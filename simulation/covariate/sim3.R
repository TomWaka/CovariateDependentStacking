rm(list = ls())
set.seed(1)

library(MASS)
library(MCMCpack)
library(gam)
library(viridis)
source("function/CDST.r")
{
  # 二次元共変量依存なデータを生成し、二次元共変量空間内での重みを可視化する(2分割)

  # settings
  n_train <- 300
  n_test <- 300
  n <- n_train + n_test

  xs <- cbind(runif(n, -1, 1), runif(n, -1, 1))

  # covariates
  x1 <- xs[, 1]
  x2 <- xs[, 2]
  x3 <- rnorm(n)
  x4 <- rnorm(n)
  x5 <- rnorm(n)
  x <- cbind(x1, x2, x3, x4, x5)

  # data generation

  M1 <- 2 * (x1 + x2)
  M2 <- -x1 + 4 * x2^2
  ID1 <- ifelse(xs[, 1] < 0, 1, 0)
  ID2 <- ifelse(xs[, 1] > 0, 1, 0)
  Mu <- ID1 * M1 + ID2 * M2
  Sig <- (0.7)^2
  y <- rnorm(n, Mu, Sig)
  ID <- cbind(ID1, ID2)

  # training & test data
  xs_train <- xs[1:n_train, ]
  y_train <- y[1:n_train]
  x_train <- as.matrix(x[1:n_train, ])
  ID_train <- ID[1:n_train, ]

  xs_test <- xs[(n_train + 1):n, ]
  y_test <- y[(n_train + 1):n]
  x_test <- as.matrix(x[(n_train + 1):n, ])
  ID_test <- ID[(n_train + 1):n, ]

  EX_train <- cbind(x_train, x_train^2)
  EX_test <- cbind(x_test, x_test^2)

  J <- 2
  test_pred <- matrix(NA, n_test, J)
  for (j in 1:J) {
    sub <- (ID_train[, j] == 1)
    fit <- lm(y_train[sub] ~ EX_train[sub, ])
    test_pred[, j] <- as.vector(cbind(1, EX_test) %*% coef(fit))
  }

  ## stacking by LOOCV
  K <- n_train
  n_train <- length(y_train)
  ID_ST <- rep(1:K, rep(n_train / K, K))

  # in-sample prediction
  Pred_mat <- matrix(NA, n_train, J)
  for (k in 1:K) {
    sub_ST <- (1:n_train)[ID_ST == k]
    for (j in 1:J) {
      sub <- (ID_train[-sub_ST, j] == 1)
      fit <- lm(y_train[sub] ~ EX_train[sub, ])
      Pred_mat[sub_ST, j] <- as.vector(cbind(1, t(EX_train[sub_ST, ])) %*% coef(fit))
    }
  }

  # Stacking with EM algorithm

  # Basis function matrix
  M <- 10
  Center <- kmeans(rbind(xs_train, xs_test), M)$center
  Base_train <- matrix(NA, n_train, M)
  Base_test <- matrix(NA, n_test, M)
  for (m in 1:M) {
    Base_train[, m] <- exp(-0.5 * apply((t(xs_train) - Center[m, ])^2, 2, sum))
    Base_test[, m] <- exp(-0.5 * apply((t(xs_test) - Center[m, ])^2, 2, sum))
  }


  # EM algorithm
  # max_iter <- 20000
  # epsilon <- 1e-5
  # # initialize parameters
  # mu <- rep(1 / J, J)
  # tau2 <- rep(1 / J, J)
  # sigma2 <- 1
  # Psi <- c(mu, tau2, sigma2)

  # for (iter in 1:max_iter) {
  #   print(iter)
  #   # E-step
  #   Wmat <- do.call(cbind, lapply(1:J, function(j) Base_train * Pred_mat[, j]))
  #   Dmat <- diag(1 / tau2)
  #   L_gamma <- t(Wmat) %*% Wmat / sigma2 + kronecker(Dmat, diag(M))
  #   m_gamma <- solve(L_gamma, t(Wmat) %*% (y_train - as.vector(mu %*% t(Pred_mat))) / sigma2)

  #   # M-step
  #   y_star <- y_train - rowSums(Wmat %*% m_gamma)
  #   mu <- c(solve(t(Pred_mat) %*% Pred_mat) %*% t(Pred_mat) %*% y_star)
  #   sigma2 <- mean((y_star - Pred_mat %*% mu)^2)
  #   tau2 <- (rowSums(matrix(m_gamma, nrow = J)^2)
  #   + sapply(1:J, function(j) block_trace(L_gamma, M, j))) / M

  #   # Check convergence
  #   Psi_new <- c(mu, tau2, sigma2)
  #   if (sum(abs(Psi_new - Psi)) > epsilon) {
  #     mu <- mu
  #     tau2 <- tau2
  #     sigma2 <- sigma2
  #     Psi <- Psi_new
  #   } else {
  #     break
  #   }
  # }

  # Ensemble prediction
  result <- em_algorithm(y_train, Pred_mat, Base_train, J, M)
  mu <- result$mu
  m_gamma <- result$m_gamma
  stacking_weight <- t(mu + t(Base_test %*% matrix(m_gamma, nrow = M)))
  yhat_final <- diag(stacking_weight %*% t(test_pred))
  mean((yhat_final - y_test)^2)

  # simple average
  mean((rowMeans(test_pred) - y_test)^2)

  w1 <- stacking_weight[, 1]
  w2 <- stacking_weight[, 2]

  ##  spatial plot function
  SPlot <- function(Sp, value, ran, xlim = NULL, title = "") {
    cs <- colorRamp(c("blue", "green", "yellow", "red"), space = "rgb")
    value <- (value - ran[1]) / (ran[2] - ran[1])
    cols <- rgb(cs(value), maxColorValue = 256)
    plot(Sp, col = cols, ylab = "X2", xlab = "X1", xlim = c(-1, 1.5), main = title, pch = 20, cex = 2)
    cs <- colorRamp(c("blue", "green", "yellow", "red"), space = "rgb")
    cols <- rgb(cs(0:1000 / 1000), maxColorValue = 256)
    rect(1.25, seq(-1, 1, length = 1001), 1.45, seq(-1, 1, length = 1001), col = cols, border = cols)
    tx <- round(seq(ran[1], ran[2], length = 5), 2)
    text(x = 1.5, y = seq(-1, 1, length = 5), tx, cex = 0.7)
    yy <- seq(0, 2, length = 5)
    for (i in 1:5) {
      segments(1.1, yy[i], 1.3, yy[i], col = "white")
    }
  }

  par(mfcol = c(1, 2))
  SPlot(xs_test, w1, ran = range(w1), title = "weight for model1")
  SPlot(xs_test, w2, ran = range(w2), title = "weight for model2")

library(ggplot2)
library(viridis)

# データフレームの作成
df1 <- data.frame(x = xs_test[, 1], y = xs_test[, 2], value = w1)
df2 <- data.frame(x = xs_test[, 1], y = xs_test[, 2], value = w2)

# プロットの作成
p1 <- ggplot(df1, aes(x, y, color = value)) +
  geom_point(size = 2) +
  scale_color_viridis(option = "H", limits = range(w1)) +
  labs(title = "", x = "X1", y = "X2") +
  theme_minimal() +
  theme(text = element_text(family = "Times New Roman"))
plot(p1)
ggsave("sim3_model1.png", width = 5, height = 4)

p2 <- ggplot(df2, aes(x, y, color = value)) +
  geom_point(size = 2) +
  scale_color_viridis(option = "H", limits = range(w2)) +
  labs(title = "", x = "X1", y = "X2") +
  theme_minimal() +
  theme(text = element_text(family = "Times New Roman"))
plot(p2)
ggsave("sim3_model2.png", width = 5, height = 4)
}
