#' @export
estimateG <- function(formula,data,index,itheta,fe=FALSE, hetslope = FALSE,
                      tune = list(M=20,J=5,neigh = 5)){
  
  N <- NROW(unique(data[index[[1]]]))
  t <- NROW(unique(data[index[[2]]]))
  p <- length(itheta)
  Gmax <- floor(N/10) # up to ~10% of the data

  BIC <- sapply(2:Gmax, function(g){
    gfe(formula,
        data=data,
        index = index,
        itheta = itheta,
        G = g,
        fe=fe,
        tune = tune,
        parallel = TRUE,
        hetslope = hetslope,
        method = "wgfe")$obj_val
    })
  
  P <- BIC[length(BIC)]*log(N*t)*(sqrt(1 - 2:Gmax*(p + t + N/2:Gmax)/(N*t))
    - sqrt(1 - 2*2:Gmax*(p + t + N/2:Gmax)/(N*t)))
  G_est <- which.min(BIC + P) 
  
  return(G_est)

}