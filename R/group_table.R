#' @title Create a Group Membership Table
#' @description
#' Generate a rectangular table listing unit IDs by their assigned group,
#' padding shorter groups with empty strings so all columns have equal length.
#' @param gfe_fit A fitted \code{gfe} or \code{wgfe} object containing at least:
#'   \describe{
#'     \item{\code{units}}{vector of unit identifiers of length \eqn{N}}
#'     \item{\code{group}}{integer vector of group assignments of length \eqn{N}}
#'   }
#' @return A data.frame where each column corresponds to a distinct group (named \code{"Group 1"}, \code{"Group 2"}, …),
#'   and each row lists the unit IDs belonging to that group.  Columns are padded with empty strings
#'   so that all columns have the same number of rows.
#' @export
group_table <- function(gfe_fit) {
  # 1) Combine units and group assignments into a 2×N matrix
  groups <- rbind(
    gfe_fit$units,
    c(gfe_fit$group)
  )
  
  # 2) Split the unit IDs by group number
  #    Result is a named list: names = group IDs, each element = character vector of unit IDs
  group_list <- split(
    groups[1, ],
    groups[2, ]
  )
  
  # 3) Determine the largest group size
  max_size <- max(vapply(group_list, length, integer(1)))
  
  # 4) Pad each group's ID vector with NA (which will later become "")
  padded <- lapply(
    group_list,
    function(ids) {
      length(ids) <- max_size   # extends to max_size, filling with NA
      ids
    }
  )
  
  # 5) Convert the padded list into a data.frame
  df_groups <- as.data.frame(
    padded,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  
  # 6) Replace NAs with empty strings and ensure all columns are character
  df_groups[is.na(df_groups)] <- ""
  df_groups[] <- lapply(df_groups, as.character)
  
  # 7) Rename columns to "Group 1", "Group 2", etc.
  colnames(df_groups) <- paste0("Group ", names(group_list))
  
  # 8) Return the resulting data.frame
  df_groups
}
