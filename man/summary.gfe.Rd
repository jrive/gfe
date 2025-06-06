% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/summary.gfe.R
\name{summary.gfe}
\alias{summary.gfe}
\title{Summarize a GFE/WGFE Fit}
\usage{
\method{summary}{gfe}(object, ...)
}
\arguments{
\item{object}{A fitted object returned by \code{gfe()} or \code{wgfe()}, containing at least:
\describe{
\item{\code{model}}{a \code{model.frame} used to fit the model}
\item{\code{terms}}{the \code{terms} object for constructing the design matrix}
\item{\code{theta}}{numeric vector or matrix of estimated slopes}
\item{\code{group}}{integer vector of group assignments of length \eqn{N}}
\item{\code{method}}{character, either \code{"gfe"} or \code{"wgfe"}}
\item{\code{sigmas}}{numeric vector of group‐specific standard deviations}
\item{\code{alphas}}{numeric \eqn{T \times G} matrix of group‐level effects}
\item{\code{fe}}{logical, \code{TRUE} if individual fixed effects were removed}
\item{\code{index}}{list of two character strings: \code{c("id", "time")}}
}}

\item{...}{Additional arguments (currently unused).}
}
\value{
An object of class \code{summary.gfe} with components:
\describe{
\item{\code{call}}{the original call to \code{gfe()} or \code{wgfe()}}
\item{\code{coefficients}}{data frame (homogeneous) or list of data frames (heterogeneous) with estimates, standard errors, t‐values, p‐values, and significance stars}
\item{\code{groupCounts}}{named integer vector of group sample sizes}
\item{\code{obj}}{numeric, the objective function value from the fit}
\item{\code{alpha_df}}{data frame in long format with columns \code{time}, \code{group}, and \code{alpha} for plotting}
\item{\code{sigmas}}{numeric vector of group‐specific standard deviations}
\item{\code{method}}{character, either \code{"gfe"} or \code{"wgfe"}}
\item{\code{ub}}{numeric vector of length 2: \eqn{\min} and \eqn{\max} number of complete observations per unit}
\item{\code{fe}}{logical, whether individual fixed effects were included}
}
}
\description{
Produce a summary for a fitted GFE or WGFE model, including coefficient tables,
group counts, objective value, and a long-format \code{alpha} data frame for plotting.
}
