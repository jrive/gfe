#' @title Within‐Transformation by ID
#' @description  Subtracts each unit’s time‐series mean from Y and each column of X
#' @param Y  Numeric vector of length N*t (stacked by unit)
#' @param X  Numeric matrix with N*t rows and p columns (no intercept column)
#' @param id Vector of length N*t giving the unit identifier for each observation
#' @return A list with components  
#'   \item{Y}{the demeaned Y}  
#'   \item{X}{the demeaned X (same dimnames as input X)}  
#' @export
panel_demean <- function(Y, X, id) {
  # demean Y
  Y_d <- Y - ave(Y, id, FUN = function(x) mean(x, na.rm = TRUE))


  # demean each column of X
  X_d <- as.matrix(X) - do.call(cbind,
                                lapply(seq_len(ncol(X)), function(j) {
                                  ave(X[, j], id, FUN = function(x) mean(x, na.rm = TRUE))
                                })
  )
  colnames(X_d) <- colnames(X)
  
  list(Y = Y_d, X = X_d)
}
