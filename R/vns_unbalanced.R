#' @title Variable-Neighborhood Search (Unbalanced Panels)
#' @description Perform one VNS run on unbalanced panels: initialize \code{theta}, then iteratively update group assignments and parameters, handling missing data.
#' @param z list of length \code{N}, where each element is a \eqn{T \times (p+1)} panel matrix (may contain \code{NA}).
#' @param theta0 numeric vector of length \code{p}, the initial \eqn{\theta}.
#' @param t integer, number of time periods (\code{T}).
#' @param params list with elements:
#'   \describe{
#'     \item{\code{J}}{maximum number of local iterations}
#'     \item{\code{n}}{maximum neighborhood size for random relocations}
#'   }
#' @param G integer, number of target groups.
#' @param method string, either \code{"gfe"} or \code{"wgfe"} to choose the objective.
#' @param hetslope logical (default \code{FALSE}); \code{TRUE} for heterogeneous slopes across groups.
#' @return A list with elements:
#'   \describe{
#'     \item{\code{theta}}{estimated \eqn{p}-vector (or \eqn{p \times G} matrix if \code{hetslope=TRUE})}
#'     \item{\code{groups}}{integer vector of length \code{N} with final group assignments}
#'     \item{\code{minimum}}{numeric, the final objective value}
#'   }
#' @export
vns_unbalanced <- function(z, theta0, t, params, G, method, hetslope = FALSE) {
  N <- length(z)
  p <- length(theta0)
  Seq <- seq_len(N)
  gee <- seq_len(G)
  skipper <- FALSE
  
  tryCatch({
    # 1) Random perturbation of initial theta
    theta <- theta0 + 0.15 * runif(p) - 0.075
    
    # 2) Build unbalanced residual matrix Z (T × N) with random initial groups
    Z <- computeZ_unbalanced_cpp(z,
                                 matrix(rep(theta, G), nrow = p, ncol = G),
                                 sample(1:G, N, replace = TRUE))
    
    # 3) Initialize alpha (T × G) by spreading out samples from Z
    alpha <- matrix(0, nrow = t, ncol = G)
    chosen <- sample(Seq, G)
    for (g in gee) {
      alpha[, g] <- Z[, chosen[g]]
    }
    
    # 4) Initial group assignment (unbalanced) and local jump
    wgroups <- assignGroups_unbalanced_cpp(z,
                                           matrix(rep(theta, G), nrow = p, ncol = G),
                                           alpha)
    wgroups <- localJump_unbalanced_cpp(wgroups, Z, gee, method)
    Z_cur <- computeZ_unbalanced_cpp(z,
                                     matrix(rep(theta, G), nrow = p, ncol = G),
                                     wgroups)
    alpha_list <- computeAlpha_unbalanced_cpp(Z_cur, wgroups)
    
    # 5) Initial theta update & compute objective
    if (method != "gfe") {
      # WGFE case
      if (hetslope) {
        theta <- calcGroupSlopes_unbalanced_cpp(z, wgroups)
        Z_cur <- computeZ_unbalanced_cpp(z, theta, wgroups)
        fobj_star <- wgfeObj_unbalanced_cpp(Z_cur, wgroups)
      } else {
        XXY <- computeXXY_demeaned_unbalanced_cpp(z, wgroups)
        theta <- wc_fp_loop_unbalanced_cpp(z, wgroups,
                                           XXY$XX_demeaned, XXY$y_demeaned,
                                           gee)
        Z_cur <- computeZ_unbalanced_cpp(z, theta, wgroups)
        fobj_star <- wgfeObj_unbalanced_cpp(Z_cur, wgroups)
      }
    } else {
      # GFE case
      if (hetslope) {
        theta <- calcGroupSlopes_unbalanced_cpp(z, wgroups)
        Z_cur <- computeZ_unbalanced_cpp(z, theta, wgroups)
        fobj_star <- gfeObj_unbalanced_cpp(Z_cur, wgroups)
      } else {
        theta <- slopeGradGFE_unbalanced_cpp(z, wgroups)
        Z_cur <- computeZ_unbalanced_cpp(z, theta, wgroups)
        fobj_star <- gfeObj_unbalanced_cpp(Z_cur, wgroups)
      }
    }
    # 6) Variable-Neighborhood Search loops
    k <- 0L  # count of failed alpha validity checks
    j <- 1L
    while (j <= params$J) {
      n <- 1L
      while (n <= params$n) {
        # 6a) Random relocation of n units
        w_check <- 0
        l <- 0
        # 6a) Random relocations of size n
        while (max(w_check)< G){
          l <- l + 1
          w2 <- random_move_cpp(wgroups, gee, Seq, n)
          # 6b) Refine and jump
          w_check <- refineGroups_unbalanced_cpp(z, w2, hetslope, method)
          if (max(w_check)< G){
            l <- l + 1
            if (l >= 5){
              break
            }
            next
          }
        }
        
        w2 <- w_check
        
        # 6c) WGFE/GFE update of theta under w2
        if (method != "gfe") {
          # WGFE update
          if (hetslope) {
            theta2 <- calcGroupSlopes_unbalanced_cpp(z, w2)
          } else {
            XXY2 <- computeXXY_demeaned_unbalanced_cpp(z, w2)
            theta2 <- wc_fp_loop_unbalanced_cpp(z, w2,
                                                XXY2$XX_demeaned, XXY2$y_demeaned,
                                                gee)
          }
        } else {
          # GFE update
          if (hetslope) {
            theta2 <- calcGroupSlopes_unbalanced_cpp(z, w2)
          } else {
            theta2 <- slopeGradGFE_unbalanced_cpp(z, w2)
          }
        }
        
        # 6d) Local jump and compute new Z
        Z2 <- computeZ_unbalanced_cpp(z, theta2, w2)
        w2 <- localJump_unbalanced_cpp(w2, Z2, gee, method)
        
        # 6e) Check alpha validity
        alpha2_list <- computeAlpha_unbalanced_cpp(Z2, w2)
        if (!isTRUE(alpha2_list$valid)) {
          k <- k + 1L
          if (k > 5L) {
            k <- 0L
            n <- n + 1L
          }
          next
        }
        
        # 6f) Evaluate objective under new assignment
        if (method != "gfe") {
          f2 <- wgfeObj_unbalanced_cpp(Z2, w2)
        } else {
          f2 <- gfeObj_unbalanced_cpp(Z2, w2)
        }
        
        # 6g) Accept if improved; else expand neighborhood
        if (f2 < fobj_star) {
          wgroups <- w2
          theta <- theta2
          fobj_star <- f2
          n <- 1L
        } else {
          n <- n + 1L
        }
      }
      j <- j + 1L
    }
    
    # 7) Final validity check and return
    Z_final <- computeZ_unbalanced_cpp(z, theta, wgroups)
    alpha_final <- computeAlpha_unbalanced_cpp(Z_final, wgroups)
    if (!isTRUE(alpha_final$valid)) {
      return(list(error = "error", issue = "error in alpha", minimum = Inf))
    }
    
    return(list(theta   = theta,
                groups  = wgroups,
                minimum = fobj_star))
  },
  error = function(e) {
    skipper <<- TRUE
  })
  
  if (skipper) {
    problem <- list(error   = "error",
                    issue   = "runtime error in VNS",
                    minimum = Inf)
    return(problem)
  }
}
