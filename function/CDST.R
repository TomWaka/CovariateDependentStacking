#' Perform Expectation-Maximization (EM) algorithm for covariate-dependent stacking
#'
#' This function implements the EM algorithm for estimating weights of the covariate-dependent stacking. 
#' It iteratively performs the E-step and M-step
#' until convergence or the maximum number of iterations is reached.
#'
#' @param y_train A vector of observed responses.
#' @param Pred_mat A matrix of predictors.
#' @param Base_train A matrix or vector representing the base predictors.
#' @param J An integer, typically the number of predictors.
#' @param M An integer, typically the number of basis functions.
#' @param max_iter An integer, the maximum number of iterations (default: 5000).
#' @param epsilon A small positive number, the convergence threshold (default: 1e-5).
#'
#' @return A list containing:
#'   \item{mu}{Estimated mean parameters}
#'   \item{tau2}{Estimated variance parameters}
#'   \item{sigma2}{Estimated error variance}
#'   \item{m_gamma}{Estimated posterior mean}
#'   \item{iterations}{Number of iterations performed}
#'
#' @examples
#' # Assuming y_train, Pred_mat, Base_train, J, and M are properly defined:
#' result <- em_algorithm(y_train, Pred_mat, Base_train, J, M)
#' print(result)
em_algorithm <- function(y_train, Pred_mat, Base_train, J, M, max_iter, epsilon) {
  # Initialize parameters
  mu <- rep(0, J)
  tau2 <- rep(1, J)
  sigma2 <- 1
  Psi <- c(mu, tau2, sigma2)
  
  # Pre-compute constant matrices
  Wmat <- do.call(cbind, lapply(1:J, function(j) Base_train * Pred_mat[,j]))
  Pred_mat_t <- t(Pred_mat)
  Pred_mat_sq <- solve(Pred_mat_t %*% Pred_mat)
  
  for(iter in 1:max_iter) {
    print(iter)
    # E-step
    Dmat <- diag(1/tau2)
    L_gamma <- crossprod(Wmat) / sigma2 + kronecker(Dmat, diag(M))
    L_gamma_chol <- chol(L_gamma)
    m_gamma <- backsolve(L_gamma_chol, forwardsolve(t(L_gamma_chol), 
                         crossprod(Wmat, y_train - as.vector(mu %*% Pred_mat_t)) / sigma2))
    
    # M-step
    Wm_gamma <- Wmat %*% m_gamma
    y_star <- y_train - rowSums(Wm_gamma)
    mu <- c(Pred_mat_sq %*% (Pred_mat_t %*% y_star))
    sigma2 <- mean((y_star - as.vector(mu %*% Pred_mat_t))^2)
    tau2 <- (rowSums(matrix(m_gamma^2, nrow=J)) + 
             vapply(1:J, function(j) block_trace(L_gamma, M, j), numeric(1))) / M
    tau2[tau2<1e-3] <- 1e-3
    
    # Check convergence
    Psi_new <- c(mu, tau2, sigma2)
    diff <- sum(abs(Psi_new - Psi))
    if(diff <= epsilon) break
    Psi <- Psi_new
  }
  
  return(list(mu = mu, tau2 = tau2, sigma2 = sigma2, m_gamma= m_gamma, iterations = iter))
}

#' Calculate the trace of a block of the inverse of L_gamma
#'
#' This function computes the trace of a specific block of the inverse of L_gamma matrix.
#' It is used in the M-step of the EM algorithm to update tau2.
#'
#' @param L_gamma A square matrix, typically cholesky the posterior precision matrix.
#' @param M An integer, the size of each block.
#' @param k An integer, the index of the block to compute.
#'
#' @return A numeric value, the sum of the diagonal elements of the inverse of the specified block.
block_trace <- function(L_gamma, M, k) {
    start_idx <- (k - 1) * M + 1
    end_idx <- k * M
    return(sum(1 / diag(L_gamma[start_idx:end_idx, start_idx:end_idx])))
}