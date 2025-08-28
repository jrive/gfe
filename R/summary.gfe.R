#' @title Summarize a GFE/WGFE Fit
#' @description
#' Produce a summary for a fitted GFE or WGFE model, including coefficient tables,
#' group counts, objective value, and a long-format \code{alpha} data frame for plotting.
#' @param object A fitted object returned by \code{gfe()} or \code{wgfe()}, containing at least:
#'   \describe{
#'     \item{\code{model}}{a \code{model.frame} used to fit the model}
#'     \item{\code{terms}}{the \code{terms} object for constructing the design matrix}
#'     \item{\code{theta}}{numeric vector or matrix of estimated slopes}
#'     \item{\code{group}}{integer vector of group assignments of length \eqn{N}}
#'     \item{\code{method}}{character, either \code{"gfe"} or \code{"wgfe"}}
#'     \item{\code{sigmas}}{numeric vector of group‐specific standard deviations}
#'     \item{\code{alphas}}{numeric \eqn{T \times G} matrix of group‐level effects}
#'     \item{\code{fe}}{logical, \code{TRUE} if individual fixed effects were removed}
#'     \item{\code{index}}{list of two character strings: \code{c("id", "time")}}
#'   }
#' @param ... Additional arguments (currently unused).
#' @return An object of class \code{summary.gfe} with components:
#'   \describe{
#'     \item{\code{call}}{the original call to \code{gfe()} or \code{wgfe()}}
#'     \item{\code{coefficients}}{data frame (homogeneous) or list of data frames (heterogeneous) with estimates, standard errors, t‐values, p‐values, and significance stars}
#'     \item{\code{groupCounts}}{named integer vector of group sample sizes}
#'     \item{\code{obj}}{numeric, the objective function value from the fit}
#'     \item{\code{alpha_df}}{data frame in long format with columns \code{time}, \code{group}, and \code{alpha} for plotting}
#'     \item{\code{sigmas}}{numeric vector of group‐specific standard deviations}
#'     \item{\code{method}}{character, either \code{"gfe"} or \code{"wgfe"}}
#'     \item{\code{ub}}{numeric vector of length 2: \eqn{\min} and \eqn{\max} number of complete observations per unit}
#'     \item{\code{fe}}{logical, whether individual fixed effects were included}
#'   }
#' @export
summary.gfe <- function(object, ...) {
  # 1) Extract response and design matrix
  mf    <- object$model
  Y     <- stats::model.response(mf)
  Xfull <- stats::model.matrix(object$terms, mf)
  X     <- if (colnames(Xfull)[1] == "(Intercept)") {
    Xfull[, -1, drop = FALSE]
  } else {
    Xfull
  }
  
  # 2) Demean if individual fixed effects were used
  id   <- object$index[[1]]
  if (isTRUE(object$fe)) {
    dm <- panel_demean(Y, X, id)
    Y  <- dm$Y
    X  <- dm$X
  }
  
  # 3) Reconstruct panel data and count complete rows per unit
  grp  <- object$group
  N    <- length(grp)
  t    <- length(Y) / N
  id_vec <- rep(seq_len(N), each = t)
  time   <- rep(seq_len(t), times = N)
  df     <- data.frame(id = id_vec, time = time, Y, X, check.names = FALSE)
  counts <- aggregate(
    complete.cases(df),
    by  = list(id = df$id),
    FUN = sum
  )
  n_min <- min(counts$x)
  n_max <- max(counts$x)
  
  # 4) Compute standard errors and t‐values
  s    <- se(object)
  est  <- object$theta
  tval <- est / s
  df_resid <- sum(counts$x)
  pval <- 2 * stats::pt(-abs(tval), df_resid)
  
  stars <- stats::symnum(
    pval,
    cutpoints = c(0, .001, .01, .05, .1, 1),
    symbols   = c("***", "**", "*", ".", " ")
  )

  # 5) Build coefficient table(s)
  p        <- ncol(X)
  coef_table <- NULL
  
  if (length(est) == p) {
    # Homogeneous slopes: single data frame
    coef_table <- data.frame(
      Estimate   = est,
      `Std.Error`= s,
      `t value`  = tval,
      `P(>|t|)`  = format.pval(pval, digits = 3, eps = 0),
      stars      = as.character(stars),
      check.names = FALSE
    )
    rownames(coef_table) <- names(object$model)[-1]
    names(coef_table)[5] <- ""
  } else {
    # Heterogeneous slopes: list of data frames, one per group
    G <- ncol(est)
    coef_table <- lapply(seq_len(G), function(g) {
      df_g <- data.frame(
        Estimate    = est[, g],
        `Std.Error` = s[, g],
        `t value`   = tval[, g],
        `P(>|t|)`   = format.pval(pval[, g], digits = 3, eps = 0),
        stars       = as.character(stars[, g]),
        check.names = FALSE
      )
      rownames(df_g) <- names(object$model)[-1]
      names(df_g)[5] <- ""
      df_g
    })
    names(coef_table) <- paste("Group", seq_len(G))
  }
  
  # 6) Group counts (drop the “grp” name)
  groupCounts <- table(grp)
  
  # 7) Prepare alpha data frame in long format
  alpha_mat <- object$alphas
  colnames(alpha_mat) <- as.character(seq_len(ncol(alpha_mat)))
  alpha_df <- data.frame(time = unique(object$times), alpha_mat, check.names = FALSE)
  alpha_df <- alpha_df %>%
    pivot_longer(
      cols      = -time,
      names_to  = "group",
      values_to = "alpha"
    )
  
  # 8) Return summary object
  structure(
    list(
      call         = object$call,
      coefficients = coef_table,
      groupCounts  = groupCounts,
      obj          = object$obj_val,
      alpha_df     = alpha_df,
      sigmas       = object$sigmas,
      method       = object$method,
      ub           = c(n_min, n_max),
      hetslope = length(est) > p,
      fe           = object$fe
    ),
    class = "summary.gfe"
  )
}

#' @title Print a GFE/WGFE Summary
#' @description
#' Print method for \code{summary.gfe} objects: displays the call, panel dimensions,
#' coefficient table(s), group proportions and standard deviations, objective value,
#' and, if \code{ggplot2} is installed, a plot of the \eqn{\alpha}-time series.
#' @param x An object of class \code{summary.gfe}.
#' @param ... Additional arguments (currently unused).
#' @export
print.summary.gfe <- function(x, ...) {
  # 1) Print the original call
  cat("Call:\n")
  print(x$call)
  # 2) Balanced vs. unbalanced panel info
  N_total <- sum(x$groupCounts)
  if (x$ub[1] != x$ub[2]) {
    cat("\nUnbalanced Panel: N = ", N_total,
        ", T = ", paste(x$ub[1], " - ", x$ub[2], sep = ""),
        "\n", sep = "")
  } else {
    cat("\nBalanced Panel: N = ", N_total,
        ", T = ", x$ub[2], "\n", sep = "")
  }
  
  # 3) Method and fixed effects info
  cat("\n",
      if (x$method != "gfe") "Weighted " else "",
      "Grouped Fixed Effects Estimation",
      if (isTRUE(x$fe)) " with Individual Fixed Effects" else "",
      ": G = ", max(as.numeric(x$alpha_df$group)), "\n", sep = "")

  # 4) Coefficient tables
  if (x$hetslope) {
    # Heterogeneous case
    cat("\nHeterogeneous Coefficients:\n")
    G <- length(x$coefficients)
    for (g in seq_len(G)) {
      cat("\nGroup", g, "\n")
      print(x$coefficients[[g]])
    }
  } else {
    # Homogeneous case
    cat("\nCoefficients:\n")
    print(x$coefficients)
  }
  
  cat(
    "---\n",
    "Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1\n",
    sep = ""
  )
  
  # 5) Group proportions and standard deviations
  cat("---\n", "Group Proportions and Standard Deviations:\n", sep = "")
  prop_sd_mat <- rbind(
    x$groupCounts / sum(x$groupCounts),
    c(x$sigmas)
  )
  rownames(prop_sd_mat) <- c("Pg", "sg")
  print(prop_sd_mat)
  
  # 6) Objective function value
  cat("---\n", "Objective Function Value: ", x$obj, "\n", sep = "")

  # 7) GFE (alpha)-time plot 
    p <- ggplot(
      x$alpha_df,
      aes(x = time, y = alpha, color = factor(group), group = group)
    ) +
      geom_line(size = 1.1) +
      labs(
        x     = substr(x$call[[4]][3], 1, nchar(x$call[[4]][3])),
        y     = "Grouped Fixed Effects",
        color = "Group"
      ) +
      scale_color_discrete(labels = function(lab) gsub("\\.", " ", lab)) +
      theme_minimal()
    print(p)

  
  invisible(x)
}
