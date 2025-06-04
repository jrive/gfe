#' @title Variable-Neighborhood Search
#' @description Perform one VNS run: initialize \code{theta}, then iteratively update group assignments and parameters.
#' @param z list of length \code{N}, where each element is a \eqn{T \times (p+1)} panel matrix (col 0=y, cols 1..p=x).
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
vns <- function(z, theta0, t, params, G, method, hetslope = FALSE) {
  N <- length(z)
  p <- length(theta0)
  Seq <- seq_len(N)
  gee <- seq_len(G)
  skipper <- FALSE
  
  tryCatch({
    # 1) Random perturbation of theta
    theta <- theta0 + 0.2 * runif(p) - 0.1
    
    # 2) Compute initial residual matrix Z (T × N)
    Z <- computeZ_cpp(z, matrix(rep(theta, G), nrow = p, ncol = G), sample(1:G, N, replace = TRUE))
    
    # 3) Initialize alpha (T × G) by spreading out samples from Z
    alpha <- matrix(0, nrow = t, ncol = G)
    chosen <- sample(Seq, G)
    for (g in gee) {
      alpha[, g] <- Z[, chosen[g]]
    }
    
    # 4) Initial group assignment
    wgroups <- assignGroups_cpp(z,
                                matrix(rep(theta, G), nrow = p, ncol = G),
                                alpha)
    
    # 5) Initial theta update and objective evaluation
    if (method != "gfe") {
      if (hetslope) {
        theta <- calcGroupSlopes_cpp(z, wgroups)
        Z_cur <- computeZ_cpp(z, theta, wgroups)
        fobj_star <- wgfeObj_cpp(Z_cur, wgroups)
      } else {
        XXY <- computeXXY_demeaned_cpp(z, wgroups)
        theta <- wc_fp_loop_cpp(z, wgroups,
                                XXY$XX_demeaned,
                                XXY$y_demeaned,
                                gee)
        Z_cur <- computeZ_cpp(z, theta, wgroups)
        fobj_star <- wgfeObj_cpp(Z_cur, wgroups)
      }
    } else {
      if (hetslope) {
        theta <- calcGroupSlopes_cpp(z, wgroups)
        Z_cur <- computeZ_cpp(z, theta, wgroups)
        fobj_star <- gfeObj_cpp(Z_cur, wgroups)
      } else {
        theta <- slopeGradGFE_cpp(z, wgroups)
        Z_cur <- computeZ_cpp(z, theta, wgroups)
        fobj_star <- gfeObj_cpp(Z_cur, wgroups)
      }
    }
    
    # 6) VNS main loops
    j <- 1L
    while (j <= params$J) {
      n <- 1L
      while (n <= params$n) {
        # 6a) Random relocations of size n
        w2 <- wgroups
        to_move <- sample(Seq, n)
        for (i in to_move) {
          w2[i] <- sample(gee[-w2[i]], 1)
        }
        
        # 6b) Refine and jump
        w2 <- refineGroups_cpp(z, w2, hetslope, method)
        
        if (method != "gfe") {
          if (hetslope) {
            theta2 <- calcGroupSlopes_cpp(z, w2)
          } else {
            XXY <- computeXXY_demeaned_cpp(z, w2)
            theta2 <- wc_fp_loop_cpp(z, w2,
                                     XXY$XX_demeaned,
                                     XXY$y_demeaned,
                                     gee)
          }
        } else {
          if (hetslope) {
            theta2 <- calcGroupSlopes_cpp(z, w2)
          } else {
            theta2 <- slopeGradGFE_cpp(z, w2)
          }
        }
        
        Z2 <- computeZ_cpp(z, theta2, w2)
        alpha2 <- computeAlpha_cpp(Z2, w2)
        w2 <- localJump_cpp(w2,
                            Z = Z2,
                            alpha = alpha2,
                            gee = gee,
                            method = method)
        
        # 6c) Evaluate objective under new assignment
        if (method != "gfe") {
          f2 <- wgfeObj_cpp(Z2, w2)
        } else {
          f2 <- gfeObj_cpp(Z2, w2)
        }
        
        # 6d) Accept or reject
        if (f2 < fobj_star) {
          wgroups <- w2
          theta <- theta2
          fobj_star <- f2
          # Restart neighborhood search
          n <- 1L
        } else {
          n <- n + 1L
        }
      }
      j <- j + 1L
    }
    
    return(list(theta = theta,
                groups = wgroups,
                minimum = fobj_star))
  },
  error = function(e) {
    skipper <<- TRUE
  })
  
  if (skipper) {
    problem <- list(error = TRUE,
                    issue = "VNS encountered an error",
                    minimum = Inf)
    return(problem)
  }
}
