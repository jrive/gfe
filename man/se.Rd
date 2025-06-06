% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/se.R
\name{se}
\alias{se}
\title{Compute Standard Errors for Estimators of a Linear GFE Model}
\usage{
se(gfe_fit)
}
\arguments{
\item{gfe_fit}{A list-like object returned by \code{gfe} or \code{wgfe} fitting routines, containing at least:
\describe{
\item{\code{model}}{a \code{model.frame} used to fit the model}
\item{\code{terms}}{the \code{terms} object for constructing the design matrix}
\item{\code{theta}}{numeric vector or matrix of estimated slopes}
\item{\code{group}}{integer vector of group assignments of length \eqn{N}}
\item{\code{method}}{character, either \code{"gfe"} or \code{"wgfe"}}
\item{\code{sigmas}}{numeric vector of group-specific variances}
}}
}
\value{
A numeric vector (for homogenous slopes) or appropriate structure returned by the C++ routine:
\describe{
\item{If homogenous (\eqn{\texttt{length(theta)} == \texttt{NCOL}(X)}):}{calls \code{se_unbalanced_cpp} and returns its output.}
\item{If heterogeneous (\eqn{\texttt{length(theta)} > \texttt{NCOL}(X)}):}{calls \code{seHet_unbalanced_cpp} and returns its output.}
}
}
\description{
Calculate standard errors from a fitted GFE/WGFE object, dispatching to the appropriate C++ routine.
}
