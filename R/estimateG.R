#' @export
estimateG <- function(formula, data, index, itheta,
                      fe = FALSE, hetslope = FALSE,
                      tune = list(M = 10, J = 5, neigh = 5),
                      method = "wgfe",
                      g_frac = 0.10,              # fraction of N to set Gmax (ceiling for search)
                      min_group_size = 5,         # require N/G >= this
                      workers = NULL,             # NULL => all physical cores
                      seed = 123,                 # reproducible parallel RNG
                      per_g_timeout = Inf) {      # seconds; requires R.utils if finite
  
  # --- basics
  N <- NROW(unique(data[[ index[[1]] ]]))
  t <- NROW(unique(data[[ index[[2]] ]]))
  p <- length(itheta)
  NT <- N * t
  
  # candidate Gs
  Gcap <- max(2L, floor(N * g_frac))
  g_vec <- 2:Gcap
  g_vec <- g_vec[(N / g_vec) >= min_group_size]
  if (!length(g_vec)) stop("No admissible G after min_group_size filter.")
  Gmax <- max(g_vec)  # will also be used for sigma^2 if method == "gfe"
  
  # choose workers and set plan
  if (is.null(workers)) {
    workers <- tryCatch(future::availableCores(logical = FALSE), error = function(...) 1L)
  }
  old_plan <- future::plan()
  on.exit(future::plan(old_plan), add = TRUE)
  future::plan(future::multisession, workers = workers)
  
  # helper: evaluate one G and return MSE-like objective
  eval_one <- function(G) {
    # cap threads in compiled libs to avoid oversubscription
    old_omp <- Sys.getenv("OMP_NUM_THREADS", NA)
    Sys.setenv(OMP_NUM_THREADS = "1")
    if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
      RhpcBLASctl::blas_set_num_threads(1)
      RhpcBLASctl::omp_set_num_threads(1)
    }
    on.exit({ if (!is.na(old_omp)) Sys.setenv(OMP_NUM_THREADS = old_omp) }, add = TRUE)
    
    run <- function() {
      fit <- gfe(formula,
                 data = data,
                 index = index,
                 itheta = itheta,
                 G = G,
                 fe = fe,
                 tune = tune,
                 parallel = FALSE,     # no inner parallel during G search
                 hetslope = hetslope,
                 method = method)
      val <- as.numeric(fit$obj_val)
      # --- NORMALIZE to MSE scale used downstream ---
      # Assume: WGFE already returns MSE; GFE returns RSS.
      if (identical(tolower(method), "gfe")) {
        NT <- N * t
        val <- val / NT  # convert RSS -> MSE
      }
      val
    }
    
    val <- try({
      if (is.finite(per_g_timeout) && per_g_timeout > 0 &&
          requireNamespace("R.utils", quietly = TRUE)) {
        R.utils::withTimeout(run(), timeout = per_g_timeout, onTimeout = "error")
      } else run()
    }, silent = TRUE)
    
    if (inherits(val, "try-error")) {
      list(G = G, ok = FALSE, obj = NA_real_, err = as.character(val))
    } else {
      list(G = G, ok = TRUE,  obj = val, err = NULL)  # obj is now MSE for both methods
    }
  }
 
  # progress bar + parallel map over G
  progressr::handlers(global = TRUE)
  out_list <- progressr::with_progress({
    prog <- progressr::progressor(steps = length(g_vec))
    future.apply::future_lapply(
      g_vec,
      function(G) {
        res <- eval_one(G)              # do work first
        prog(sprintf("G=%d", G))        # tick AFTER, exactly once
        res
      },
      future.seed = seed
    )
  })
  
  # collect & check
  ok <- vapply(out_list, `[[`, logical(1), "ok")
  if (!any(ok)) stop("All G candidates failed.")
  
  Gs  <- vapply(out_list, `[[`, integer(1), "G")
  obj <- vapply(out_list, `[[`, numeric(1), "obj")  # assumed MSE
  
  # keep only successes (if any failed)
  keep <- is.finite(obj)
  Gs   <- Gs[keep]
  obj  <- obj[keep]
  if (!length(obj)) stop("All successful G candidates returned non-finite obj_val.")
  
  # (B) Build criterion
  if (identical(tolower(method), "gfe")) {
    # --- GFE BIC: crit_G = MSE_G + sigma^2 * ln(NT) * (G*T + N + p) / (N*T)
    sigma2 <- tail(obj, 1)*NT/(NT - (Gmax * t) - N - p)  
    P <- sigma2 * log(NT) * (Gs*(t + N - Gs))/ NT
    crit <- obj + P
  } else {
    # --- WGFE: keep your original penalty construction
    adj  <- 1 - (2 * Gs) * (p + t + N / (2 * Gs)) / (NT)
    adj2 <- 1 - 2 * (2 * Gs) * (p + t + N / (2 * Gs)) / (NT)
    adj [adj  < 0] <- NA_real_
    adj2[adj2 < 0] <- NA_real_
    BIC_ref <- tail(obj, 1)*NT/(NT - (Gmax * t) - Gmax - N - p)               # last successful value
    P <- BIC_ref * log(NT) * (sqrt(adj) - sqrt(adj2))
    crit <- obj + P
  }
  
  # pick G
  G_est <- Gs[ which.min(crit) ]
  
  message(sprintf("estimateG: considered %d G values on %d workers; selected G = %d.",
                  length(Gs), workers, G_est))
  
  # attach diagnostics (optional)
  diag <- data.frame(G = Gs, obj = obj, penalty = P, crit = crit,
                     method = rep(method, length(Gs)),
                     NT = NT, N = N, T = t, p = p)
  diag <- diag[order(diag$G), ]
  attr(G_est, "grid") <- diag
  
  G_est
}
