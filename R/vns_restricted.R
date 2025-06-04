#' @title Variable-Neighborhood Search (Restricted WGFE where all variances equal)
#' @description Perform one restricted WGFE VNS run over \eqn{M} random starts:  
#'   initializes \eqn{\theta}, \eqn{\sigma}, then iteratively updates group assignments and parameters.
#'   Restriction: all group variances must be equal.
#' @param Y numeric vector of length \eqn{N \times T}, stacked by unit (each block of \eqn{t} observations).
#' @param X numeric matrix with \eqn{N \times T} rows and \eqn{p} columns of covariates, aligned with \code{Y}.
#' @param t integer, number of time periods per unit (\eqn{T}).
#' @param H integer, number of groups (\eqn{G}).
#' @param params list with elements:
#'   \describe{
#'     \item{\code{M}}{number of random replicates}
#'     \item{\code{J}}{maximum number of local iterations per replicate}
#'     \item{\code{n}}{maximum neighborhood size for random relocations per replicate}
#'   }
#' @param init list with components:
#'   \describe{
#'     \item{\code{theta}}{numeric vector of length \eqn{p}, initial slope estimates}
#'     \item{\code{sig}}{numeric vector of length \eqn{G}, initial group‐specific variances}
#'   }
#' @return A list with elements:
#'   \describe{
#'     \item{\code{theta}}{numeric vector of length \eqn{p}, best slope estimate found}
#'     \item{\code{group}}{integer vector of length \eqn{N}, best group assignments}
#'     \item{\code{opt}}{numeric, best objective value (\eqn{WGFE}-banded SSE) found}
#'     \item{\code{sigma}}{numeric, best overall \eqn{\sigma} estimate}
#'     \item{\code{time}}{difftime object, runtime for the best replicate}
#'   }
#' @export
vns_restricted <- function(Y, X, t, H, params, init) {
  N    <- length(Y) / t
  p    <- ncol(X)
  G    <- H
  M    <- params$M
  J    <- params$J
  neigh<- params$n
  Seq  <- seq_len(N)
  gee  <- seq_len(G)
  
  # Pre-build the data list of panels
  data  <- cbind(Y, X)
  zList <- lapply(Seq, function(i) {
    data[((i - 1) * t + 1):(i * t), , drop = FALSE]
  })
  
  results <- lapply(seq_len(M), function(m) {
    start_time <- Sys.time()
    
    # 1) Randomize initial theta and sigma for this replicate
    theta0 <- init$theta + runif(p, -0.05, 0.05)
    sig0   <- mean(init$sig) + (max(init$sig) - min(init$sig)) * runif(1, -1, 1)
    
    # 2) Build initial group‐specific sigma offsets
    sigsg  <- init$sig - min(init$sig)
    sigs0  <- sig0 + sigsg
    
    # 3) Compute residual matrix Z (TT × N) and initial alpha
    Z      <- computeZ_res_cpp(zList, theta0)
    alpha0 <- Z[, sample.int(N, G, replace = FALSE), drop = FALSE]
    
    # 4) Initial group assignment
    wgroups <- assignGroups_cpp(zList,
                                matrix(rep(theta0, G), nrow = p, ncol = G),
                                alpha0)
    
    # 5) One fixed-point solve to update (theta0, sig0)
    XXY     <- computeXXY_demeaned_cpp(zList, wgroups)
    delta   <- wc_fp_bs_cpp(zList,
                            rep(0, p),
                            sig0,
                            wgroups,
                            XXY$XX_demeaned,
                            XXY$y_demeaned,
                            sigsg)
    theta0  <- delta[1:p]
    sig0    <- delta[p + 1]
    
    # 6) Record the best so far for this replicate
    fobj_star   <- wgfeObj_bs_cpp(zList, theta0, wgroups, sigsg, sig0)
    slope_star  <- theta0
    sig_star    <- sig0
    groups_star <- wgroups
    
    # 7) VNS loops for this replicate
    j <- 0L
    while (j <= J) {
      n <- 1L
      while (n <= neigh) {
        # 7a) Random relocation of size n
        w2 <- wgroups
        to_move <- sample(Seq, n)
        for (i in to_move) {
          w2[i] <- sample(gee[-w2[i]], 1L)
        }
        
        # 7b) Re-assign & re-solve until stable
        refined <- refineGroups_res_cpp(zList, w2, sig0, sigsg)
        w2     <- refined$wgroups
        theta0 <- refined$slope0
        sig0   <- refined$sig0
        
        # 7c) Local-jump sweep and objective evaluation
        w2 <- localJump_res_cpp(zList, theta0, w2, sigsg, sig0)
        f2 <- wgfeObj_bs_cpp(zList, theta0, w2, sigsg, sig0)
        
        # 7d) Update global best if improved
        if (f2 < fobj_star) {
          groups_star <- w2
          fobj_star   <- f2
          n <- 1L  # restart neighborhood
        } else {
          n <- n + 1L
        }
      }
      j <- j + 1L
    }
    
    end_time <- Sys.time()
    list(
      theta = slope_star,
      group = groups_star,
      opt   = fobj_star,
      sigma = sig_star,
      time  = end_time - start_time
    )
  })
  
  # Pick the best replicate across M
  best_idx <- which.min(vapply(results, `[[`, numeric(1), "opt"))
  results[[best_idx]]
}
