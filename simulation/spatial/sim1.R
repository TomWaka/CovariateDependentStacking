rm(list=ls())
set.seed(1) 

library(MASS)
library(MCMCpack)
library(gam)
library(ggplot2)
library(viridis)
library(tidyr)
library(tictoc)
source("function/CDST.r")


# settings 
n_train <- 300   # the number of locations (including non-sampled locations)
n_test <- 300   # the number of non-sampled locations 
n <- n_train + n_test

# generation of sampling locations
coords <- cbind(runif(n,-1,1), runif(n,-1,1))

# kernel matrix
phi <- 0.5    # range parameter for covariates
dd <- as.matrix(dist(coords))
mat <- exp(-dd/phi)  # exponential kernel

# covariates
chol_mat <- chol(mat)
z1 <- t(chol_mat) %*% matrix(rnorm(n), ncol = 1)
z2 <- t(chol_mat) %*% matrix(rnorm(n), ncol = 1)
rr <- 0.2
x1 <- z1
x2 <- rr * z1 + sqrt(1 - rr^2) * z2
x3 <- rnorm(n)
x4 <- rnorm(n)
x5 <- rnorm(n)
x <- cbind(x1, x2, x3, x4, x5) # covariates, some of which are dependent on spatial locations

# data generation 
M1 <- x1 - x2^2/2
M2 <- x1^2 + x2^2
ID1 <- ifelse(coords[,1]<0, 1, 0)
ID2 <- ifelse(coords[,1]>0, 1, 0)
sp_cov <- exp(-as.matrix(dist(coords))/0.3)
chol_sp_cov <- chol(sp_cov)
w <- as.vector(t(chol_sp_cov) %*% matrix(rnorm(n), ncol = 1))
Mu <- ID1*M1 + ID2*M2 + 0.3*w 
Sig <- 0.7^2
y <- rnorm(n, Mu, Sig)
ID <- cbind(ID1, ID2)

# training & test data 
coords_train <- coords[1:n_train,]
y_train <- y[1:n_train]
x_train <- as.matrix(x[1:n_train,])
ID_train <- ID[1:n_train,]

coords_test <- coords[(n_train+1):n,]
y_test <- y[(n_train+1):n]
x_test <- as.matrix(x[(n_train+1):n,])
ID_test <- ID[(n_train+1):n,]

EX_train <- cbind(x_train, x_train^2)
EX_test <- cbind(x_test, x_test^2)


J <- 2
test_pred <- matrix(NA, n_test, J)
for (j in 1:J){
  sub <- (ID_train[,j]==1)
  fit <- lm(y_train[sub]~EX_train[sub,])
  test_pred[,j] <- as.vector( cbind(1,EX_test)%*%coef(fit) )
}

## stacking by LOOCV
K <- n_train
n_train <- length(y_train)
ID_ST <- rep(1:K, rep(n_train/K, K))

# in-sample prediction 
Pred_mat <- matrix(NA, n_train, J)
for(k in 1:K){
  sub_ST <- (1:n_train)[ID_ST==k]
  for(j in 1:J) {
    sub <- (ID_train[-sub_ST,j]==1)
    fit <- lm(y_train[sub]~EX_train[sub,])
    Pred_mat[sub_ST, j] <- as.vector( cbind(1,t(EX_train[sub_ST,]))%*%coef(fit) )
  }
}

M <- 10
Center <- kmeans(rbind(coords_train, coords_test), M)$center   # testデータは入れなくてもOK
Base_train <- matrix(NA, n_train, M)
Base_test <- matrix(NA, n_test, M)
for(m in 1:M){
  Base_train[,m] <- exp(-0.5*apply((t(coords_train)-Center[m,])^2, 2, sum))
  Base_test[,m] <- exp(-0.5*apply((t(coords_test)-Center[m,])^2, 2, sum))
}

result <- em_algorithm(y_train, Pred_mat, Base_train, J, M)
mu <- result$mu
m_gamma <- result$m_gamma
stacking_weight <- t(mu + t(Base_test %*% matrix(m_gamma, nrow=M)))
yhat_final <- diag(stacking_weight %*% t(test_pred))
mean((yhat_final-y_test)^2)

# simple average
mean((rowMeans(test_pred) - y_test)^2)

# visualization
df1 <- data.frame(x = coords_test[,1], y = coords_test[,2], value = stacking_weight[,1]) 
p1 <- ggplot(df1, aes(x, y, color = value)) +
  geom_point(size = 2) +
  scale_color_viridis(option = "H", limits = range(stacking_weight[,1])) +
  labs(title = "", x = "Longitude", y = "Latitude") +
  theme_minimal() +
  theme(text = element_text(family = "Times New Roman"))
plot(p1)
ggsave("sim1_model1.png", width = 5, height = 4)

df2 <- data.frame(x = coords_test[,1], y = coords_test[,2], value = stacking_weight[,2]) 
p2 <- ggplot(df2, aes(x, y, color = value)) +
  geom_point(size = 2) +
  scale_color_viridis(option = "H", limits = range(stacking_weight[,2])) +
  labs(title = "", x = "Longitude", y = "Latitude") +
  theme_minimal() +
  theme(text = element_text(family = "Times New Roman"))
plot(p2)
ggsave("sim1_model2.png", width = 5, height = 4)

