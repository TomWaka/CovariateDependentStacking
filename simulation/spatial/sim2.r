rm(list=ls())
set.seed(1)
library(MASS)
library(MCMCpack)
library(gam)
library(tictoc)
source("function/CDST.r")

# 空間依存なデータを生成し、空間回帰の重みを可視化する(4分割)

# settings 
n_train <- 1000   # the number of locations (including non-sampled locations)
n_test <- 400   # the number of non-sampled locations 
n <- n_train + n_test


# generation of sampling locations
r <- sqrt(runif(n))
coords <- cbind(runif(n,-1,1), runif(n,-1,1))


# kernel matrix
phi <- 0.5    # range parameter for covariates
dd <- as.matrix(dist(coords))
mat <- exp(-dd/phi)  # exponential kernel


# covariates
chol_mat <- chol(mat)
z1 <- t(chol_mat) %*% matrix(rnorm(n), ncol = 1)
z2 <- t(chol_mat) %*% matrix(rnorm(n), ncol = 1)
z3 <- t(chol_mat) %*% matrix(rnorm(n), ncol = 1)
x1 <- z1
x2 <- z2
x3 <- z3
x4 <- rnorm(n)
x5 <- rnorm(n)
x <- cbind(x1, x2, x3, x4, x5) # covariates, some of which are dependent on spatial locations


# data generation 
M1 <- x1 - 0.5 * x2^2
M2 <- x1^2 + x2^2
M3 <- x1^2 + x3
M4 <- x2 + x3^2 + 0.5
ID1 <- ifelse(coords[,1] >=0 & coords[,2]>=0, 1, 0)
ID2 <- ifelse(coords[,1] <0  & coords[,2]>=0, 1, 0)
ID3 <- ifelse(coords[,1] >=0 & coords[,2]<0, 1, 0)
ID4 <- ifelse(coords[,1] <0  & coords[,2]<0, 1, 0)
sp_cov <- exp(-as.matrix(dist(coords))/0.3)
chol_sp_cov <- chol(sp_cov)
w <- as.vector(t(chol_sp_cov) %*% matrix(rnorm(n), ncol = 1))
Mu <- ID1*M1 + ID2*M2 + ID3*M3 + ID4*M4 + 0.3*w
Sig <- 1
y <- rnorm(n, Mu, Sig)
ID <- cbind(ID1, ID2, ID3, ID4)


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


J <- 4
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
  for(j in 1:J){
    sub <- (ID_train[-sub_ST,j]==1)
    fit <- lm(y_train[sub]~EX_train[sub,])
    Pred_mat[sub_ST, j] <- as.vector( cbind(1,t(EX_train[sub_ST,]))%*%coef(fit) )
  }
  
}


# Stacking with EM algorithm

# Basis function matrix
M <- 10
Center <- kmeans(rbind(coords_train, coords_test), M)$center   # testデータは入れなくてもOK
Base_train <- matrix(NA, n_train, M)
Base_test <- matrix(NA, n_test, M)
for(m in 1:M){
  Base_train[,m] <- exp(-0.5*apply((t(coords_train)-Center[m,])^2, 2, sum))
  Base_test[,m] <- exp(-0.5*apply((t(coords_test)-Center[m,])^2, 2, sum))
}


# Ensemble prediction
result <- em_algorithm(y_train, Pred_mat, Base_train, J, M)
mu <- result$mu
m_gamma <- result$m_gamma
stacking_weight <- t(mu + t(Base_test %*% matrix(m_gamma, nrow=M)))
yhat_final <- diag(stacking_weight %*% t(test_pred))
mean((yhat_final-y_test)^2)

# simple average
mean( (rowMeans(test_pred) - y_test)^2 )


w1 <- stacking_weight[,1]
w2 <- stacking_weight[,2]
w3 <- stacking_weight[,3]
w4 <- stacking_weight[,4]


##  spatial plot function
SPlot <- function(Sp, value, ran, xlim=NULL, title=""){
  cs <- colorRamp( c("blue", "green", "yellow", "red"), space="rgb")
  value <- (value-ran[1])/(ran[2]-ran[1])
  cols <- rgb( cs(value), maxColorValue=256 )
  plot(Sp, col=cols, ylab="Latitude", xlab="Longitude", xlim=c(-1, 1.5), main=title, pch=20, cex=2)
  cs <- colorRamp( c("blue", "green", "yellow", "red"), space="rgb")
  cols <- rgb(cs(0:1000/1000), maxColorValue=256)
  rect(1.25, seq(-1,1,length=1001), 1.45, seq(-1,1,length=1001), col=cols, border=cols)
  tx <- round(seq(ran[1], ran[2], length=5), 2)
  text(x=1.5, y=seq(-1, 1, length=5), tx, cex=0.7)
  yy <- seq(0, 2, length=5)
  for (i in 1:5){
    segments(1.1, yy[i], 1.3, yy[i], col="white")
  }
}


par(mfcol=c(2,2))
SPlot(coords_test, w1, ran=range(w1), title="weight for model1")
SPlot(coords_test, w2, ran=range(w2), title="weight for model2")
SPlot(coords_test, w3, ran=range(w3), title="weight for model3")
SPlot(coords_test, w4, ran=range(w4), title="weight for model4")



df1 <- data.frame(x = coords_test[,1], y = coords_test[,2], value = w1)
df2 <- data.frame(x = coords_test[,1], y = coords_test[,2], value = w2)
df3 <- data.frame(x = coords_test[,1], y = coords_test[,2], value = w3)
df4 <- data.frame(x = coords_test[,1], y = coords_test[,2], value = w4)

# プロットの作成
p1 <- ggplot(df1, aes(x, y, color = value)) +
  geom_point(size = 2) +
  scale_color_viridis(option = "D", limits = range(w1)) +
  labs(title = "", x = "Longitude", y = "Latitude") +
  theme_minimal() +
  theme(text = element_text(family = "Times New Roman"))
plot(p1)
ggsave("sim2_model1.png", width = 5, height = 4)

p2 <- ggplot(df2, aes(x, y, color = value)) +
  geom_point(size = 2) +
  scale_color_viridis(option = "D", limits = range(w2)) +
  labs(title = "", x = "Longitude", y = "Latitude") +
  theme_minimal() +
  theme(text = element_text(family = "Times New Roman"))
plot(p2)
ggsave("sim2_model2.png", width = 5, height = 4)


p3 <- ggplot(df3, aes(x, y, color = value)) +
  geom_point(size = 2) +
  scale_color_viridis(option = "D", limits = range(w3)) +
  labs(title = "", x = "Longitude", y = "Latitude") +
  theme_minimal() +
  theme(text = element_text(family = "Times New Roman"))
plot(p3)
ggsave("sim2_model3.png", width = 5, height = 4)


p4 <- ggplot(df4, aes(x, y, color = value)) +
  geom_point(size = 2) +
  scale_color_viridis(option = "D", limits = range(w4)) +
  labs(title = "", x = "Longitude", y = "Latitude") +
  theme_minimal() +
  theme(text = element_text(family = "Times New Roman"))
plot(p4)
ggsave("sim2_model4.png", width = 5, height = 4)
