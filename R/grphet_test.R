#' @title Group Variance Heterogeneity Test via Bootstrap
#' @description
#' Perform a group variance heterogeneity test by comparing WGFE and GFE (criterion-based test):
#' 1) compute the test statistic under the fitted models,
#' 2) generate bootstrap replicates of the statistic using parallel VNS runs.
#' @param object A fitted \code{wgfe} object (list) containing at least:
#'   \describe{
#'     \item{\code{model}}{the \code{model.frame} used for fitting}
#'     \item{\code{terms}}{the \code{terms} object for constructing the design matrix}
#'     \item{\code{theta}}{numeric vector of estimated slopes}
#'     \item{\code{group}}{integer vector of group assignments of length \eqn{N}}
#'     \item{\code{method}}{character, either \code{"gfe"} or \code{"wgfe"}}
#'     \item{\code{sigmas}}{numeric vector of group‐specific variances}
#'     \item{\code{fe}}{logical, \code{TRUE} if fixed effects were removed}
#'   }
#' @param object_gfe A fitted \code{gfe} object under the null hypothesis (same structure as \code{object}, but using \code{"gfe"} method).
#' @param M integer, number of random starts for restricted VNS (default: 10).
#' @param J integer, maximum number of local iterations per VNS call (default: 10).
#' @param neigh integer, maximum neighborhood size for random relocations per VNS call (default: 10).
#' @param B integer, number of bootstrap replicates (default: 100).
#' @param cores integer, number of CPU cores to use for parallel processing (default: \code{parallel::detectCores()}).
#' @return A list with elements:
#'   \describe{
#'     \item{\code{J_true}}{numeric, the observed test statistic}
#'     \item{\code{J_bs}}{numeric vector of length \eqn{B}, bootstrap replicates of the test statistic (may contain \code{NA} on error).}
#'   }
#' @export
grphet_test <- function(object,
                        object_gfe,
                        M     = 10,
                        J     = 10,
                        neigh = 10,
                        B     = 100,
                        cores = parallel::detectCores()) {
  # 1) Extract data from fitted object
  mf    <- object$model
  Y     <- stats::model.response(mf)
  Xfull <- stats::model.matrix(object$terms, mf)
  X     <- if (colnames(Xfull)[1] == "(Intercept)") {
    Xfull[, -1, drop = FALSE]
  } else {
    Xfull
  }
  
  if (isTRUE(object$fe)) {
    dm <- panel_demean(Y, X, id)
    Y  <- dm$Y
    X  <- dm$X
  }
  
  data_mat <- cbind(Y, X)
  grp      <- object$group
  N        <- length(grp)
  t_period <- length(Y) / N
  Seq      <- seq_len(N)
  p        <- ncol(X)
  G        <- max(grp)
  
  params <- list(M     = M,
                 J     = J,
                 neigh = neigh)
  
  # 2) Initial WGFE estimates
  wgfe_est <- object
  
  # 3) Compute alpha and sigma for restricted initialization
  z <- lapply(split(seq_len(nrow(data_mat)), id), function(rows) {
    data_mat[rows, , drop = FALSE]
  })
  Z_list   <- lapply(z, function(z_i) {
    z_i[, 1] - z_i[, 2:(p + 1), drop = FALSE] %*% wgfe_est$theta
  })
  groups   <- wgfe_est$group
  sigma_vec <- computeSigma_cpp(z, wgfe_est$theta, groups)
  init     <- list(theta = wgfe_est$theta,
                   sig   = sigma_vec)
  
  # 4) Compute observed test statistic J_true
  J_true <- 2 * sqrt(object_gfe$obj_val / (N * t_period)) -
    2 * wgfe_est$obj_val
  
  # 5) Set up cluster for bootstrap
  cl <- parallel::makeCluster(cores, type = "SOCK")
  on.exit(parallel::stopCluster(cl))
  parallel::clusterExport(cl, c(
    "Y", "X", "t_period", "G", "N", "p",
    "M", "J", "neigh", "groups",
    "sigma_vec", "wgfe_est", "params", "z"
  ), envir = environment())
  
  # Ensure package is loaded on each worker
  parallel::clusterEvalQ(cl, library(gfe))
  
  # 6) Bootstrap replicates
  J_bs <- parallel::parSapply(cl, seq_len(B), function(b) {
    # a) Draw bootstrap units
    idx_bs <- sample(N, N, replace = TRUE)
    Yb <- unlist(lapply(idx_bs, function(i) {
      Y[((i - 1) * t_period + 1):(i * t_period)]
    }))
    Xb <- do.call(rbind, lapply(idx_bs, function(i) {
      X[((i - 1) * t_period + 1):(i * t_period), , drop = FALSE]
    }))
    group_bs_init <- groups[idx_bs]
    id <- rep(seq_len(N), each = t_period)
    time <- rep(seq_len(t_period), times = N)
    df <- data.frame(id = id, time = time, Yb, Xb)
    
    # b) Build formula Yb ~ X1 + ... + Xp
    fmla <- as.formula(
      paste("Yb ~", paste(colnames(Xb), collapse = " + "))
    )
    
    # c) Unconstrained VNS (GFE fit on bootstrap)
    tryCatch({
      vns_ur <- gfe(
        fmla,
        data  = df,
        itheta = wgfe_est$theta,
        index  = c("id", "time"),
        G      = G,
        tune   = list(M = 15, J = 10, neigh = 10)
      )
      
      # d) Restricted VNS using original alpha & sigma & bootstrap init
      init_b <- list(theta = vns_ur$theta,
                     sig   = sigma_vec)
      vns_rb <- vns_restricted(
        Yb, Xb, t_period, G,
        list(M = M, J = J, neigh = neigh),
        init_b
      )
      
      # e) Bootstrap statistic
      vns_rb$opt - 2 * vns_ur$obj_val
    }, error = function(e) {
      NA_real_
    })
  })
  
  # 7) Return observed statistic and bootstrap replicates
  list(J_true = J_true,
       J_bs   = J_bs)
}
