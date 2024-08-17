rm(list = ls())
library(MASS)
library(MCMCpack)
library(gam)
# library(threg)
library(np)
library(kernlab)
library(randomForest)
library(parallel)
library(ggplot2)
library(viridis)
library(tidyr)
source("function/CDST.r")

for (scenario in 1:2) {
    n_iterations <- 100

    mse_STSV <- numeric(n_iterations)
    mse_ST <- numeric(n_iterations)
    mse_SA <- numeric(n_iterations)
    mse_TLR <- numeric(n_iterations)
    mse_SI <- numeric(n_iterations)
    mse_AIC <- numeric(n_iterations)
    mse_M1 <- numeric(n_iterations)
    mse_M2 <- numeric(n_iterations)
    mse_M3 <- numeric(n_iterations)
    mse_M4 <- numeric(n_iterations)

    for (it in 1:n_iterations) {
        print(it)
        set.seed(it)

        ## settings
        p <- 5 # number of covariates (5 or 15)
        add_p <- p - 5

        # sample size
        n_train <- 300 # the number of locations
        n_test <- 100 # the number of non-sampled locations
        n <- n_train + n_test

        # generation of sampling locations
        xs <- cbind(runif(n, -1, 1), runif(n, -1, 1))

        # covariates
        x1 <- xs[, 1]
        x2 <- xs[, 2]
        x3 <- rnorm(n)
        x4 <- rnorm(n)
        x5 <- rnorm(n)
        x <- cbind(x1, x2, x3, x4, x5)

        # data generation
        if (scenario == 1) {
            M1 <- 2 * (x1 + x2)
            M2 <- -x1 + 4 * x2^2
            ID1 <- ifelse(xs[, 1] < 0, 1, 0)
            ID2 <- ifelse(xs[, 1] > 0, 1, 0)
            Mu <- ID1 * M1 + ID2 * M2
        }
        if (scenario == 2) {
            Mu <- 2 * (x1 + 1) * x2 + (1 - x1) * x3^2
        }
        Sig <- (0.7)^2
        y <- rnorm(n, Mu, Sig)

        ## split data into train and test data
        xs_train <- xs[1:n_train, ]
        y_train <- y[1:n_train]
        x_train <- as.matrix(x[1:n_train, ])

        xs_test <- xs[(n_train + 1):n, ]
        y_test <- y[(n_train + 1):n]
        x_test <- as.matrix(x[(n_train + 1):n, ])

        ## prediction models
        J <- 4
        XX_train <- data.frame(x_train)
        XX_test <- data.frame(x_test)
        names(XX_train) <- names(XX_test) <- paste0("X", 1:p)
        test_pred <- matrix(NA, n_test, J)

        # linear model
        formula1 <- as.formula(paste0("y_train~", paste0("X", 1:p, collapse = "+")))
        fit1 <- lm(formula1, data = XX_train)
        test_pred[, 1] <- predict(fit1, XX_test)
        # additive model
        formula2 <- as.formula(paste0("y_train~", paste0("s(X", 1:p, ")", collapse = "+")))
        fit2 <- gam(formula2, data = XX_train)
        test_pred[, 2] <- predict(fit2, XX_test)
        # random forest
        fit3 <- randomForest(y_train ~ ., data = XX_train)
        test_pred[, 3] <- predict(fit3, XX_test)
        # gaussian process regression
        fit4 <- gausspr(y_train ~ ., data = XX_train, kernel = "rbfdot")
        test_pred[, 4] <- predict(fit4, XX_test)

        ## Threshold Regression (TR)
        reg <- function(X, y) {
            X <- qr(X)
            as.matrix(qr.coef(X, y))
        }
        thrd <- 0
        xL <- XX_train * (xs[1:n_train, 1] < thrd)
        xU <- XX_train * (xs[1:n_train, 1] > thrd)
        X_tr_train <- as.matrix(cbind(xL, xU))
        xL <- XX_test * (xs[(n_train + 1):n, 1] < thrd)
        xU <- XX_test * (xs[(n_train + 1):n, 1] > thrd)
        X_tr_test <- as.matrix(cbind(xL, xU))
        pred_TR_lin <- X_tr_test %*% reg(X_tr_train, y_train)

        # Ichimura estimator for single index model
        train_data <- data.frame(y = y_train, x_train)
        fit_ichimura <- npindex(y ~ x1 + x2 + x3 + x4 + x5, data = train_data, method = "ichimura")
        test_data <- data.frame(x_test)
        pred_SI <- predict(fit_ichimura, newdata = test_data)

        K <- n_train
        ID <- rep(1:K, rep(n_train / K, K))
        # in-sample prediction
        stack_models <- function(k, n_train, XX_train, y_train, p) {
            sub <- (1:n_train)[ID == k]

            # linear model
            formula1 <- as.formula(paste0("y_train[-sub]~", paste0("X", 1:p, collapse = "+")))
            fit1 <- lm(formula1, data = XX_train[-sub, ])
            pred1 <- predict(fit1, XX_train[sub, ])

            # additive model
            formula2 <- as.formula(paste0("y_train[-sub]~", paste0("s(X", 1:p, ")", collapse = "+")))
            fit2 <- gam(formula2, data = XX_train[-sub, ])
            pred2 <- predict(fit2, XX_train[sub, ])

            # random forest
            fit3 <- randomForest(y_train[-sub] ~ ., data = XX_train[-sub, ])
            pred3 <- predict(fit3, XX_train[sub, ])

            # gaussian process regression
            fit4 <- gausspr(y_train[-sub] ~ ., data = XX_train[-sub, ], kernel = "rbfdot")
            pred4 <- predict(fit4, XX_train[sub, ])

            return(cbind(pred1, pred2, pred3, pred4))
        }
        Pred_list <- mclapply(1:K, stack_models, n_train = n_train, XX_train = XX_train, y_train = y_train, p = p, mc.cores = detectCores())
        Pred_mat <- do.call(rbind, Pred_list)

        # Stacking with EM algorithm
        # basis function
        M <- 10
        Center <- kmeans(rbind(xs_train, xs_test), M)$center
        Base_train <- matrix(NA, n_train, M)
        Base_test <- matrix(NA, n_test, M)
        for (m in 1:M) {
            Base_train[, m] <- exp(-0.5 * apply((t(xs_train) - Center[m, ])^2, 2, sum))
            Base_test[, m] <- exp(-0.5 * apply((t(xs_test) - Center[m, ])^2, 2, sum))
        }
        # EM algorithm
        result <- em_algorithm(y_train, Pred_mat, Base_train, J, M)
        mu <- result$mu
        m_gamma <- result$m_gamma

        ## Stacking (constant)
        best_k <- 0
        best_mse <- Inf
        best_pred_ST <- NULL

        for (K in seq(n_train, n_train, by = 5)) {
            ID <- rep(1:K, rep(n_train / K, K))

            # in-sample prediction
            Pred_list <- mclapply(1:K, stack_models, n_train = n_train, XX_train = XX_train, y_train = y_train, p = p, mc.cores = detectCores())
            Pred_mat <- do.call(rbind, Pred_list)

            alpha <- as.vector(solve(t(Pred_mat) %*% Pred_mat) %*% t(Pred_mat) %*% y_train)
            pred_ST <- c(alpha %*% t(test_pred))
            mse <- mean((pred_ST - y_test)^2)

            if (mse < best_mse) {
                best_k <- K
                best_mse <- mse
                best_pred_ST <- pred_ST
            }
        }

        # Ensemble prediction
        stacking_weight <- t(mu + t(Base_test %*% matrix(m_gamma, nrow = M)))
        pred_STSV <- diag(stacking_weight %*% t(test_pred))

        # AIC weights
        AIC_values <- c(AIC(fit1), AIC(fit2), NA, NA)
        AIC_weights <- exp(-0.5 * (AIC_values)) / sum(exp(-0.5 * (AIC_values)), na.rm = TRUE)
        pred_AIC <- colSums(t(test_pred) * AIC_weights, na.rm = TRUE)

        # simple average (SA)
        pred_SA <- rowMeans(test_pred)

        # MSE
        mse_STSV[it] <- mean((pred_STSV - y_test)^2)
        mse_ST[it] <- mean((best_pred_ST - y_test)^2)
        mse_SA[it] <- mean((pred_SA - y_test)^2)
        mse_TLR[it] <- mean((pred_TR_lin - y_test)^2)
        mse_SI[it] <- mean((pred_SI - y_test)^2)
        mse_AIC[it] <- mean((pred_AIC - y_test)^2)
        mse_M1[it] <- mean((test_pred[, 1] - y)^2)
        mse_M2[it] <- mean((test_pred[, 2] - y)^2)
        mse_M3[it] <- mean((test_pred[, 3] - y)^2)
        mse_M4[it] <- mean((test_pred[, 4] - y)^2)
    }

    # データフレームの作成
    mse_data <- data.frame(
        Method = rep(c("CDST", "ST", "SAIC", "SA", "TR", "SI", "M1", "M2", "M3", "M4"), each = n_iterations),
        MSE = c(mse_STSV, mse_ST, mse_AIC, mse_SA, mse_TLR, mse_SI, mse_M1, mse_M2, mse_M3, mse_M4)
    )

    mse_data$Method <- factor(mse_data$Method, levels = c("CDST", "ST", "SAIC", "SA", "TR", "SI", "M1", "M2", "M3", "M4"))

    violin_plot <- ggplot(mse_data, aes(x = Method, y = MSE, fill = Method)) +
        geom_violin(trim = FALSE, alpha = 0.7) +
        geom_boxplot(width = 0.2, outlier.shape = NA, alpha = 0.5) +
        labs(
            title = "",
            x = "Method", y = "MSE"
        ) +
        theme_minimal() +
        theme(
            legend.position = "none",
            text = element_text(size = 14, family = "Times New Roman")
        )

    print(violin_plot)
    ggsave(paste0("sim5_", scenario, "_violin.png"), width = 6, height = 6)
}

