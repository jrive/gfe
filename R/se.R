#' @title Compute Standard Errors for Estimators of a Linear GFE Model
#' @description Calculate standard errors from a fitted GFE/WGFE object, dispatching to the appropriate C++ routine.
#' @param gfe_fit A list-like object returned by \code{gfe} or \code{wgfe} fitting routines, containing at least:
#'   \describe{
#'     \item{\code{model}}{a \code{model.frame} used to fit the model}
#'     \item{\code{terms}}{the \code{terms} object for constructing the design matrix}
#'     \item{\code{theta}}{numeric vector or matrix of estimated slopes}
#'     \item{\code{group}}{integer vector of group assignments of length \eqn{N}}
#'     \item{\code{method}}{character, either \code{"gfe"} or \code{"wgfe"}}
#'     \item{\code{sigmas}}{numeric vector of group-specific variances}
#'   }
#' @return A numeric vector (for homogenous slopes) or appropriate structure returned by the C++ routine:
#'   \describe{
#'     \item{If homogenous (\eqn{\texttt{length(theta)} == \texttt{NCOL}(X)}):}{calls \code{se_unbalanced_cpp} and returns its output.}
#'     \item{If heterogeneous (\eqn{\texttt{length(theta)} > \texttt{NCOL}(X)}):}{calls \code{seHet_unbalanced_cpp} and returns its output.}
#'   }
#' @export
se <- function(gfe_fit) {
  mf      <- gfe_fit$model
  Y       <- stats::model.response(mf)
  Xfull   <- stats::model.matrix(gfe_fit$terms, mf)
  X       <- if (colnames(Xfull)[1] == "(Intercept)") {
    Xfull[, -1, drop = FALSE]
  } else {
    Xfull
  }
  
  theta   <- gfe_fit$theta
  groupR  <- gfe_fit$group
  N       <- length(groupR)
  t       <- length(Y) / N
  
  sigmas  <- if (identical(gfe_fit$method, "gfe")) {
    rep(1, max(groupR))
  } else {
    gfe_fit$sigmas
  }
  
  # Build list of T × (p+1) panels
  data_mat <- cbind(Y, X)
  zList    <- lapply(seq_len(N), function(i) {
    data_mat[((i - 1) * t + 1):(i * t), , drop = FALSE]
  })

  # Pre-compute residuals Z and alpha for both routines
  Z_mat   <- computeZ_unbalanced_cpp(zList, if(length(c(theta)) == ncol(X)){matrix(rep(theta, max(groupR)),
                                        nrow = ncol(X), ncol = max(groupR))}else{
                                          theta
                                        },
                          groupR)
  alpha0  <- computeAlpha_unbalanced_cpp(Z_mat, groupR)$alpha
  
  # Dispatch based on number of slope parameters
  if (length(theta) == ncol(X)) {
    return(
      se_unbalanced_cpp(
        Y      = Y,
        X      = X,
        theta0 = theta,
        groupR = groupR,
        alpha0 = alpha0,
        sigma0 = sigmas,
        t      = t
      )
    )
  }
  
  if (length(theta) > ncol(X)) {
    return(
      seHet_unbalanced_cpp(
        z       = zList,
        theta   = theta,
        groupR  = groupR,
        alpha0  = alpha0
      )
    )
  }
}
