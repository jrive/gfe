#' @title Grouped Fixed Effects Estimation
#' @description
#' Estimate a grouped fixed‐effects model (GFE or WGFE) via variable‐neighborhood search (VNS).
#' Supports balanced or unbalanced panels and optional individual fixed effects.
#' @param formula A two‐sided \code{formula}, e.g., \code{Y ~ X1 + X2 + ...}.
#' @param data A data frame containing the variables in \code{formula} and \code{index}.
#' @param index Character vector of length two: \code{c("id_col", "time_col")} identifying panel structure.
#' @param G Integer number of groups; if \code{NULL}, groups will be estimated.
#' @param itheta Numeric vector of length \eqn{p}, initial slopes; if \code{NULL}, a time fixed effects estimate is used.
#' @param fe Logical (default \code{FALSE}); \code{TRUE} to include individual fixed effects.
#' @param tune List with elements:
#'   \describe{
#'     \item{\code{M}}{number of random starts for VNS (default: 5)}
#'     \item{\code{J}}{maximum number of local iterations per start (default: 10)}
#'     \item{\code{n}}{maximum neighborhood size for random relocations per start (default: 10)}
#'   }
#' @param method Character, either \code{"wgfe"} for group heteroskedasticity or \code{"gfe"} (default: \code{"wgfe"}).
#' @param hetslope Logical (default \code{FALSE}); \code{TRUE} for group-heterogeneous coefficients.
#' @param subset An optional vector specifying a subset of observations to use.
#' @param na.action Function to handle missing values when building \code{model.frame} (default: \code{na.omit}).
#' @param parallel Logical (default \code{FALSE}); \code{TRUE} to run VNS starts in parallel.
#' @param ncores Integer number of cores to use if \code{parallel = TRUE} (default: all physical cores).
#' @param ... Additional arguments (currently unused).
#' @return A list of class \code{gfe} with components:
#'   \describe{
#'     \item{\code{call}}{matched call to this function}
#'     \item{\code{formula}}{the model formula}
#'     \item{\code{model}}{the \code{model.frame} used}
#'     \item{\code{index}}{character vector \code{c("id", "time")}}
#'     \item{\code{terms}}{the \code{terms} object for the design matrix}
#'     \item{\code{theta}}{estimated slope vector (or matrix if heterogeneous)}
#'     \item{\code{group}}{integer vector of final group assignments (length \eqn{N})}
#'     \item{\code{alphas}}{matrix of estimated group fixed effects (\eqn{T \times G})}
#'     \item{\code{sigmas}}{numeric vector of group standard deviations}
#'     \item{\code{residuals}}{vector of unit‐by‐time residuals}
#'     \item{\code{times}}{unique time values}
#'     \item{\code{obj_val}}{numeric, objective value at solution}
#'     \item{\code{gradient}}{numeric gradient of objective at \code{theta} (or \code{NA} on error)}
#'     \item{\code{tune}}{list of tuning parameters used}
#'     \item{\code{fe}}{logical, whether individual fixed effects were removed}
#'     \item{\code{method}}{character, \code{"wgfe"} or \code{"gfe"}}
#'     \item{\code{units}}{unique unit identifiers}
#'   }
#' @export
gfe <- function(formula,
                data,
                index,
                G              = NULL,
                itheta         = NULL,
                fe             = FALSE,
                tune           = list(M = 5, J = 10, n = 10),
                method         = "wgfe",
                hetslope       = FALSE,
                subset         = NULL,
                na.action      = na.omit,
                parallel       = FALSE,
                ncores         = parallel::detectCores(logical = FALSE),
                ...) {
  # 0) Initial slope guess if not provided: pooling via plm
  if (is.null(itheta)) {
    pool_coef <- plm::plm(
      formula   = formula,
      data      = data,
      index     = index,
      model     = "pooling"
    )$coefficients
    itheta <- as.numeric(pool_coef)[-1]
  }
  
  # 1) If G not specified, estimate G
  if (is.null(G)) {
    message("Number of groups G not specified: Attempting to estimate G.")
    G <- estimateG(
      formula   = formula,
      data      = data,
      index     = index,
      itheta    = itheta,
      fe        = fe,
      hetslope  = hetslope,
      method = method,
      tune      = list(M = 20, J = 10, neigh = 10)
    )
  }
  if (G < 2L){
    print("The case of G = 1 is equivalent to time fixed effect model.")
    return(plm(formula,data=data,index = index,effect = "time"))
  }
  
  cl <- match.call()
  
  # 2) Build model.frame
  if (is.null(subset)) {
    mf <- stats::model.frame(
      formula   = formula,
      data      = data,
      na.action = na.pass
    )
  } else {
    mf <- stats::model.frame(
      formula   = formula,
      data      = data,
      subset    = subset,
      na.action = na.pass
    )
  }
  
  # 3) Extract response Y and design matrix X
  Terms <- attr(mf, "terms")
  Y     <- stats::model.response(mf)
  X     <- stats::model.matrix(Terms, mf)
  
  # 4) Align and sort by id & time
  idx_data <- data[rownames(mf), index, drop = FALSE]
  id       <- idx_data[[1]]
  time     <- idx_data[[2]]
  ord      <- order(id, time)
  Y        <- Y[ord]
  X        <- X[ord, , drop = FALSE]
  id       <- id[ord]
  time     <- time[ord]
  
  # 5) Determine number of time periods and check balancedness
  times_unique <- unique(time)
  Tperiods     <- length(times_unique)
  datm         <- cbind(Y, X[, -1, drop = FALSE])  # drop intercept
  ub           <- anyNA(datm)
  
  # 6) If fixed effects requested, demean within each id
  if (isTRUE(fe)) {
    dm <- panel_demean(Y, X, id)
    Y  <- dm$Y
    X  <- dm$X
    datm <- cbind(Y, X[, -1, drop = FALSE])
  }
  
  # 8) Build list of T × (p+1) matrices by id
  z <- lapply(split(seq_len(nrow(datm)), id), function(rows) {
    datm[rows, , drop = FALSE]
  })
  # 9) Run VNS for each random start (parallel or sequential)
  if (isTRUE(parallel)) {
    clust <- parallel::makeCluster(ncores)
    on.exit(parallel::stopCluster(clust), add = TRUE)
    parallel::clusterEvalQ(clust, library(gfe))
    parallel::clusterExport(
      clust,
      varlist = c("vns", "vns_unbalanced", "z", "Tperiods",
                  "itheta", "tune", "G", "method", "hetslope"),
      envir   = environment()
    )
    results <- parallel::parLapply(
      clust,
      seq_len(tune$M),
      function(m) {
        if (!ub) {
          vns(z, itheta, Tperiods, params = tune, G, method, hetslope)
        } else {
          vns_unbalanced(z, itheta, Tperiods, params = tune, G, method, hetslope)
        }
      }
    )
  } else {
    results <- lapply(
      seq_len(tune$M),
      function(m) {
        if (!ub) {
          vns(z, itheta, Tperiods, params = tune, G, method, hetslope)
        } else {
          vns_unbalanced(z, itheta, Tperiods, params = tune, G, method, hetslope)
        }
      }
    )
  }

  # 10) Select best solution
  mins  <- vapply(results, `[[`, numeric(1), "minimum")
  best  <- which.min(mins)
  theta <- results[[best]]$theta
  if (is.null(theta)) {
    stop("A major error occurred in each search M = m. ",
         "Consider increasing M and/or lowering J, n, or G.")
  }
  group <- results[[best]]$groups
  
  # 11) Compute final Z, alphas, and sigmas
  if (!ub) {
    Z      <- computeZ_cpp(z, theta, group)
    alphas <- computeAlpha_cpp(Z, group)
    sigmas <- computeSigma_cpp(z, theta, group)
  } else {
    Z_list <- computeZ_unbalanced_cpp(z, theta, group)
    alpha_list <- computeAlpha_unbalanced_cpp(Z_list, group)
    alphas <- alpha_list$alpha
    sigmas <- computeSigma_unbalanced_cpp(Z_list, alphas, group)
  }
  
  # 12) Form objective function for gradient check
  if (!ub) {
    if (identical(method, "gfe")) {
      fobj <- function(b) gfeObj_cpp(computeZ_cpp(z, b, group), group)
    } else {
      fobj <- function(b) wgfeObj_cpp(computeZ_cpp(z, b, group), group)
    }
  } else {
    if (identical(method, "gfe")) {
      fobj <- function(b) gfeObj_unbalanced_cpp(
        computeZ_unbalanced_cpp(z, b, group), group
      )
    } else {
      fobj <- function(b) wgfeObj_unbalanced_cpp(
        computeZ_unbalanced_cpp(z, b, group), group
      )
    }
  }
  
  grad <- tryCatch(
    numDeriv::grad(fobj, theta),
    error = function(e) NA_real_
  )
  
  # 13) Compute residuals
  resids <- computeResiduals_cpp(z, theta, group, alphas)
  
  # 14) Assemble return object
  out <- list(
    call       = cl,
    formula    = formula,
    model      = mf,
    index      = index,
    terms      = Terms,
    theta      = theta,
    group      = group,
    alphas     = alphas,
    sigmas     = sigmas,
    residuals  = resids,
    times      = times_unique,
    obj_val    = mins[best],
    gradient   = grad,
    tune       = tune,
    fe         = fe,
    method     = method,
    units      = unique(id)
  )
  class(out) <- "gfe"
  out
}
