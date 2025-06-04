#' @title Plot Group-Level Summary Statistics Over Time
#' @description
#' Generate and display time-series plots of group‐level summary statistics for specified variables,
#' based on a fitted GFE/WGFE object and an input data frame.
#' @param gfe_fit A fitted object returned by \code{gfe()} or \code{wgfe()}, containing at least:
#'   \describe{
#'     \item{\code{units}}{vector of unit identifiers of length \eqn{N}}
#'     \item{\code{group}}{integer vector of group assignments of length \eqn{N}}
#'     \item{\code{index}}{character vector of length 2: \code{c("id_column", "time_column")}}
#'   }
#' @param data A data frame containing at least the columns specified in \code{gfe_fit\$index},
#'   plus each variable in \code{vars}.  The data frame must include one row per observation
#'   with columns for unit ID, time, and the variables of interest.
#' @param vars Character vector of column names in \code{data} to plot.  Each variable will be
#'   summarized by group and time.
#' @param summary_stat Character: which summary statistic to compute for each group at each time.
#'   Must be one of \code{"mean"}, \code{"median"}, or \code{"sd"} (default: \code{"mean"}).
#' @return An (invisible) named list of \code{ggplot2} plot objects, one element per variable in \code{vars}.
#'   Each plot shows the chosen summary statistic over time for each group.
#' @export
plot_groups <- function(gfe_fit,
                        data,
                        vars,
                        summary_stat = c("mean", "median", "sd")) {
  # 0) Require dplyr and ggplot2 namespaces
  if (!requireNamespace("dplyr", quietly = TRUE) ||
      !requireNamespace("ggplot2", quietly = TRUE)) {
    stop("You must have both 'dplyr' and 'ggplot2' installed.")
  }
  # 1) Resolve summary_stat argument
  summary_stat <- match.arg(summary_stat)
  
  # 2) Build a data frame of unit→group mappings
  groupings <- data.frame(
    id    = gfe_fit$units,
    group = gfe_fit$group,
    stringsAsFactors = FALSE
  )
  names(groupings)[1] <- as.character(gfe_fit$index[1])
  
  # 3) Left-join the group label onto the data
  data_labeled <- dplyr::left_join(
    data,
    groupings,
    by = as.character(gfe_fit$index[1])
  )
  
  # 4) Rename the time column to "time"
  time_col_idx <- which(colnames(data_labeled) == as.character(gfe_fit$index[2]))
  colnames(data_labeled)[time_col_idx] <- "time"
  
  # 5) Initialize list to store plots
  plots_out <- list()
  
  # 6) Loop over each requested variable
  for (v in vars) {
    if (!v %in% colnames(data_labeled)) {
      warning(sprintf("Variable '%s' not found in data; skipping.", v))
      next
    }
    
    # 6a) Compute group-by-time summary
    summary_df <- data_labeled %>%
      dplyr::group_by(group, time) %>%
      dplyr::summarise(
        stat = if (summary_stat == "mean") {
          mean(!!rlang::sym(v), na.rm = TRUE)
        } else if (summary_stat == "median") {
          median(!!rlang::sym(v), na.rm = TRUE)
        } else {
          sd(!!rlang::sym(v), na.rm = TRUE)
        },
        .groups = "drop"
      )
    
    # 6b) Build the ggplot
    p <- ggplot2::ggplot(
      summary_df,
      aes(x      = time,
          y      = stat,
          colour = factor(group),
          group  = group)
    ) +
      ggplot2::geom_line(size = 1.1) +
      ggplot2::labs(
        x     = as.character(gfe_fit$index[2]),
        y     = sprintf("%s(%s)", summary_stat, v),
        color = "Group"
      ) +
      ggplot2::scale_color_discrete(
        labels = function(x) gsub("\\.", " ", x)
      ) +
      ggplot2::theme_minimal()
    
    # 6c) Store and print the plot
    plots_out[[v]] <- p
    print(p)
  }
  
  # 7) Return the list of plots invisibly
  invisible(plots_out)
}
