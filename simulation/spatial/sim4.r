rm(list = ls())
library(MASS)
library(MCMCpack)
library(gam)
library(parallel)
library(ggplot2)
library(GWmodel)
library(spdep)
library(spatialreg)
library(RandomForestsGLS)
library(viridis)
library(tidyr)
library(gamFactory)
source("function/CDST.r")


for (scenario in 1:4) {
    n_iterations <- 100
    p <- 5

    mse_CDST <- mse_ST <- mse_SA <- mse_AIC <- mse_M1 <- mse_M2 <- mse_M3 <- mse_M4 <- numeric(n_iterations)

    for (it in 1:n_iterations) {
        print(it)
        set.seed(it)

        ## settings
        add_p <- p - 5

        # sample size
        n_train <- 300 # the number of locations (including non-sampled locations)
        n_test <- 100 # the number of non-sampled locations
        n <- n_train + n_test

        ## spatial location
        coords <- cbind(runif(n, -1, 1), runif(n, -1, 1))

        ##  covariates
        phi <- 0.5 # range parameter for covariates
        dd <- as.matrix(dist(coords))
        mat <- exp(-dd / phi)
        z1 <- mvrnorm(1, rep(0, n), mat)
        z2 <- mvrnorm(1, rep(0, n), mat)
        x1 <- z1
        rr <- 0.2
        x2 <- rr * z1 + sqrt(1 - rr^2) * z2
        x3 <- rnorm(n)
        x4 <- rnorm(n)
        x5 <- rnorm(n)
        x <- cbind(x1, x2, x3, x4, x5)
        if (add_p > 0) {
            x_add <- matrix(rnorm(n * add_p), n, add_p)
            x <- cbind(x, x_add)
        }

        ## data generation
        # kernel matrix
        sp_cov <- exp(-as.matrix(dist(coords)) * 0.3)
        # mean
        if (scenario == 1) {
            w <- as.vector(mvrnorm(1, rep(0, n), sp_cov))
            Mu <- w + x3^2 * exp(-0.3 * (coords[, 1]^2 + coords[, 2]^2)) + sin(2 * x2) * coords[, 2]
        }
        if (scenario == 2) {
            w <- as.vector(mvrnorm(1, rep(0, n), sp_cov))
            Mu <- 2 * w + (10 * sin(pi * x1 * x2) + 20 * (x3 - 0.5)^2 + 10 * x4 + 5 * x5) / 20
        }
        if (scenario == 3) {
            w <- as.vector(mvrnorm(1, rep(0, n), sp_cov))
            Mu <- 2 * w + (coords[, 1] + 1) * x1 + (1 - coords[, 1]) * x3^2
        }
        if (scenario == 4) {
            w <- as.vector(mvrnorm(1, rep(0, n), sp_cov))
            Mu <- 2 * (coords[, 1] + 1) * w + x1 + (1 - coords[, 1]) * x3^2
        }

        Sig <- (0.7)^2
        y <- rnorm(n, Mu, Sig)

        ## split data into train and test data
        coords_train <- coords[1:n_train, ]
        y_train <- y[1:n_train]
        x_train <- as.matrix(x[1:n_train, ])

        coords_test <- coords[(n_train + 1):n, ]
        y_test <- y[(n_train + 1):n]
        x_test <- as.matrix(x[(n_train + 1):n, ])

        ## Four prediction models
        J <- 4
        XX_train <- data.frame(x_train, coords_train)
        XX_test <- data.frame(x_test, coords_test)
        names(XX_train) <- names(XX_test) <- paste0("X", 1:(p + 2))
        test_pred <- matrix(NA, n_test, J)

        # additive model with spatial effect
        formula3 <- as.formula(paste0("y_train~", paste0("s(X", 1:(p + 2), ")", collapse = "+")))
        fit3 <- gam(formula3, data = XX_train)
        test_pred[, 1] <- predict(fit3, XX_test)

        ## Spatial random forest (SpatialRF)
        estimation_result <- RFGLS_estimate_spatial(coords_train, y_train, x_train, ntree = 50)
        prediction_result <- RFGLS_predict_spatial(estimation_result, coords_test, x_test)
        test_pred[, 2] <- prediction_result$prediction

        # SAR
        knn <- knn2nb(knearneigh(coords_train, k = 5))
        W <- nb2listw(knn, style = "W")
        sar_model <- lagsarlm(y_train ~ ., data = as.data.frame(x_train), listw = W)
        knn_test <- knn2nb(knearneigh(coords_test, k = 5), row.names = row.names(XX_test))
        W_test <- nb2listw(knn_test, style = "W")
        test_pred[, 3] <- predict(sar_model, listw = W_test, newdata = as.data.frame(x_test))

        # GWR
        sp_train <- SpatialPointsDataFrame(coords_train, data.frame(x_train))
        sp_test <- SpatialPointsDataFrame(coords_test, data.frame(x_test))
        bw <- bw.gwr(y_train ~ ., sp_train, approach = "CV", kernel = "gaussian")
        test_pred[, 4] <- gwr.predict(y_train ~ ., sp_train, sp_test, bw = bw, kernel = "gaussian")$SDF$prediction

        K <- 5
        n_train <- length(y_train)
        ID <- rep(1:K, rep(n_train / K, K))
        Pred_mat <- matrix(NA, n_train, J)
        for (k in 1:K) {
            sub <- (1:n_train)[ID == k]

            # additive model with spatial effect
            formula3 <- as.formula(paste0("y_train[-sub]~", paste0("s(X", 1:(p + 2), ")", collapse = "+")))
            fit3 <- gam(formula3, data = XX_train[-sub, ])
            Pred_mat[sub, 1] <- predict(fit3, XX_train[sub, ])

            ## Spatial random forest (SpatialRF)
            estimation_result <- RFGLS_estimate_spatial(coords_train[-sub, ], y_train[-sub], x_train[-sub, ], ntree = 50)
            prediction_result <- RFGLS_predict_spatial(estimation_result, (coords_train[sub, ]), x_train[sub, ])
            Pred_mat[sub, 2] <- prediction_result$prediction

            # SAR
            knn <- knn2nb(knearneigh(coords_train[-sub, ], k = 5))
            W <- nb2listw(knn, style = "W")
            sar_model <- lagsarlm(y_train[-sub] ~ ., data = as.data.frame(x_train[-sub, ]), listw = W)
            knn_test <- knn2nb(knearneigh(coords_train[sub, ], k = 5), row.names = row.names(x_train[sub, ]))
            W_test <- nb2listw(knn_test, style = "W")
            res <- (predict(sar_model, listw = W_test, newdata = as.data.frame(x_train[sub, ])))
            Pred_mat[sub, 3] <- res[1:(n_train / K)]

            # GWR
            sp_train <- SpatialPointsDataFrame(coords_train[-sub, ], data.frame(x_train[-sub, ]))
            sp_test <- SpatialPointsDataFrame(coords_train[sub, ], data.frame(x_train[sub, ]))
            bw <- bw.gwr(y_train[-sub] ~ ., sp_train, approach = "CV", kernel = "gaussian")
            Pred_mat[sub, 4] <- gwr.predict(y_train[-sub] ~ ., sp_train, sp_test, bw = bw, kernel = "gaussian")$SDF$prediction
        }

        ## AIC for model selection
        ## Calculate AIC for each model
        AIC_values <- c(AIC(fit3), NA, AIC(sar_model), NA)

        ## Calculate model weights based on AIC
        AIC_weights <- exp(-0.5 * (AIC_values)) / sum(exp(-0.5 * (AIC_values)), na.rm = TRUE)

        ## Model averaging using AIC weights
        pred_AIC <- colSums(t(test_pred) * AIC_weights, na.rm = TRUE)


        # basis function
        M <- 5
        Center <- kmeans(coords, M)$center

        Base_train <- matrix(NA, n_train, M)
        Base_test <- matrix(NA, n_test, M)
        for (l in 1:M) {
            Base_train[, l] <- exp(-0.5 * apply((t(coords_train) - Center[l, ])^2, 2, sum))
            Base_test[, l] <- exp(-0.5 * apply((t(coords_test) - Center[l, ])^2, 2, sum))
        }
        Base_train <- cbind(1, Base_train)
        Base_test <- cbind(1, Base_test)

        ## Stacking (constant)
        alpha <- as.vector(solve(t(Pred_mat) %*% Pred_mat) %*% t(Pred_mat) %*% y_train)
        pred_ST <- c(alpha %*% t(test_pred))

        # simple average (SA)
        pred_SA <- rowMeans(test_pred)

        # CDST start
        M <- 10
        Center <- kmeans(rbind(coords_train, coords_test), M)$center # testデータは入れなくてもOK
        Base_train <- matrix(NA, n_train, M)
        Base_test <- matrix(NA, n_test, M)
        for (m in 1:M) {
            Base_train[, m] <- exp(-0.5 * apply((t(coords_train) - Center[m, ])^2, 2, sum))
            Base_test[, m] <- exp(-0.5 * apply((t(coords_test) - Center[m, ])^2, 2, sum))
        }

        # EM algorithm
        result <- em_algorithm(y_train, Pred_mat, Base_train, J, M)
        mu <- result$mu
        m_gamma <- result$m_gamma
        stacking_weight <- t(mu + t(Base_test %*% matrix(m_gamma, nrow = M)))
        pred_CDST <- diag(stacking_weight %*% t(test_pred))

        K <- n_train
        ID <- rep(1:K, rep(n_train / K, K))

        # in-sample prediction
        Pred_list <- mclapply(1:K, stack_models, n_train = n_train, XX_train = XX_train, y_train = y_train, p = p, mc.cores = detectCores())
        Pred_mat <- do.call(rbind, Pred_list)

        alpha <- as.vector(solve(t(Pred_mat) %*% Pred_mat) %*% t(Pred_mat) %*% y_train)
        pred_ST <- c(alpha %*% t(test_pred))


        # Probabilistic Stacking (PS)
        res <- y_train - Pred_mat
        sds <- apply(res, 2, sd)
        logP_train <- sapply(1:J, function(ii) dnorm(res[ , ii], 0, sd = sds[ii], log = TRUE))
        data_add_train <- data.frame(y = y_train,
                                 x1 = coords_train[,1],
                                 x2 = coords_train[,2],
                                 pred1 = Pred_mat[ , 1],
                                 pred2 = Pred_mat[ , 2],
                                 pred3 = Pred_mat[ , 3],
                                 pred4 = Pred_mat[ , 4])
        data_add_test <- data.frame(y = y_test,
                                x1 = coords_test[,1],
                                x2 = coords_test[,2],
                                pred1 = test_pred[ , 1],
                                pred2 = test_pred[ , 2],
                                pred3 = test_pred[ , 3],
                                pred4 = test_pred[ , 4])
        fit_PS <- gam(list(y ~ s(x1, x2, k = 10),
                       ~ s(x1, x2, k = 10),
                       ~ s(x1, x2, k = 10)
        ), data = data_add_train,
        family = fam_stackProb(logP = logP_train))
        PS_weights <- predict(fit_PS, newdata = data_add_test, type = "response")
        pred_PS <- drop(rowSums(test_pred * PS_weights))
        
        # CDST by GAM
        fit_BYGAM <- gam(y ~ s(x1, x2, by = pred1) +
                       s(x1, x2, by = pred2) +
                       s(x1, x2, by = pred3) +
                       s(x1, x2, by = pred4),
                     data = data_add_train, method = "REML")
        pred_BYGAM <- predict(fit_BYGAM, newdata = data_add_test, type = "response")

        # MSE
        mse_CDST[it] <- mean((pred_CDST - y_test)^2)
        mse_ST[it] <- mean((pred_ST - y_test)^2)
        mse_AIC[it] <- mean((pred_AIC - y_test)^2)
        mse_SA[it] <- mean((pred_SA - y_test)^2)
        mse_PS[it]     <- mean((pred_PS - y_test)^2)
        mse_BYGAM[it]  <- mean((pred_BYGAM - y_test)^2)
        mse_M1[it] <- mean((test_pred[, 1] - y_test)^2)
        mse_M2[it] <- mean((test_pred[, 2] - y_test)^2)
        mse_M3[it] <- mean((test_pred[, 3] - y_test)^2)
        mse_M4[it] <- mean((test_pred[, 4] - y_test)^2)
    }

    mse_data <- data.frame(
        Method = rep(c("CDSTE", "CDSTG","ST", "SA", "SAIC", "PS", "M1", "M2", "M3", "M4"), each = n_iterations),
        MSE = c(mse_CDST, mse_BYGAM, mse_ST, mse_SA, mse_AIC, mse_PS,mse_M1, mse_M2, mse_M3, mse_M4)
    )
    mse_data$Method <- factor(mse_data$Method, levels = c("CDSTE", "CDSTG","ST", "SA", "SAIC", "PS", "M1", "M2", "M3", "M4"))

    violin_plot <- ggplot(mse_data, aes(x = Method, y = MSE, fill = Method)) +
        geom_violin(trim = FALSE, alpha = 0.7) +
        geom_boxplot(width = 0.2, outlier.shape = NA, alpha = 0.5) +
        labs(
            title = "",
            x = paste0("Scenario", scenario), y = "MSE"
        ) +
        theme_minimal() +
        theme(
            legend.position = "none",
            text = element_text(size = 14, family = "Times New Roman")
        )
    print(violin_plot)
    #ggsave(paste0("sim4_", scenario, "_violin.png"), width = 7.5, height = 4)
}
