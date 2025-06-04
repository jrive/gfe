// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;

// [[Rcpp::export]]
arma::mat computeZ_cpp(const Rcpp::List&         zList,
                       const Rcpp::NumericMatrix& theta,
                       const Rcpp::IntegerVector& groups) {
  int N = zList.size();
  if (groups.size() != N) {
    Rcpp::stop("`groups` must have length N = zList.size()");
  }
  if (N == 0) {
    // Nothing to compute on an empty list
    return arma::mat();
  }
  
  // Pull (T, p) from the first element
  arma::mat first = Rcpp::as<arma::mat>(zList[0]);
  int T = first.n_rows;
  int p = first.n_cols - 1;  // expect first.n_cols == p + 1
  
  // Check that every zList[[i]] has dimensions T × (p+1)
  for (int i = 0; i < N; ++i) {
    arma::mat zi = Rcpp::as<arma::mat>(zList[i]);
    if (zi.n_rows != T || zi.n_cols != (p + 1)) {
      Rcpp::stop("`zList[[%d]]` has dimensions %d×%d; expected %d×%d",
                 i + 1, zi.n_rows, zi.n_cols, T, p + 1);
    }
  }
  
  // Load theta into Armadillo
  arma::mat th = Rcpp::as<arma::mat>(theta);
  if (th.is_empty()) {
    Rcpp::stop("`theta` cannot be empty");
  }
  int R = th.n_rows;
  int C = th.n_cols;
  
  // Determine if heterogeneous (p×G) or homogeneous (p×1 or 1×p)
  bool heterogeneous = (R > 1 && C > 1);
  
  // Pre‐allocate output (T × N)
  arma::mat Zall(T, N, arma::fill::zeros);
  
  // Cast all panels once
  std::vector<arma::mat> Zvec;
  Zvec.reserve(N);
  for (int i = 0; i < N; ++i) {
    Zvec.emplace_back(Rcpp::as<arma::mat>(zList[i]));  // each is T×(p+1)
  }
  
  if (!heterogeneous) {
    // --- HOMOGENEOUS case: theta must be p×1 or 1×p ---
    arma::colvec b;
    if (R == p && C == 1) {
      // theta is p×1
      b = th.col(0);
    } else if (R == 1 && C == p) {
      // theta is 1×p, transpose to p×1
      b = th.row(0).t();
    } else {
      Rcpp::stop("For homogeneous: `theta` must be p×1 or 1×p (p = %d)", p);
    }
    
    // Compute Zall_{·,i} = Y_i − X_i * b for each unit i
    for (int i = 0; i < N; ++i) {
      const arma::mat& zi = Zvec[i];       // T×(p+1): col 0 = Y, cols 1..p = X
      Zall.col(i) = zi.col(0) - zi.cols(1, p) * b;
    }
    
  } else {
    // --- HETEROGENEOUS case: theta must be p×G ---
    if (th.n_rows != p) {
      Rcpp::stop("Heterogeneous `theta` must have %d rows (one per regressor)", p);
    }
    int G = th.n_cols;  // number of groups implied by theta’s columns
    
    for (int i = 0; i < N; ++i) {
      int g = groups[i];
      if (g < 1 || g > G) {
        Rcpp::stop("`groups` contains invalid index at position %d (found %d; valid range is 1..%d)",
                   i + 1, g, G);
      }
      arma::colvec bi = th.col(g - 1);   // p×1 slope vector for group g
      const arma::mat& zi = Zvec[i];     // T×(p+1)
      Zall.col(i) = zi.col(0) - zi.cols(1, p) * bi;
    }
  }
  
  return Zall;
}


// [[Rcpp::export]]
arma::mat computeZ_res_cpp(List zList, NumericVector theta) {
  int N = zList.size();
  arma::colvec bc = as<arma::colvec>(theta);
  arma::mat Z = as<arma::mat>(zList[0]).cols(0,0) - 
    as<arma::mat>(zList[0]).cols(1,theta.size()) * bc;
  int T = Z.n_rows;
  arma::mat Zall(T, N);
  for (int i = 0; i < N; ++i) {
    arma::mat zi = as<arma::mat>(zList[i]);
    Zall.col(i) = zi.col(0) - zi.cols(1,theta.size()) * bc;
  }
  return Zall;
}


// [[Rcpp::export]]
arma::mat computeAlpha_cpp(const arma::mat& Z, 
                           const IntegerVector& groups) {
  int T = Z.n_rows, N = Z.n_cols;
  
  // 1) discover unique group labels, in sorted order
  IntegerVector uniq = sort_unique(groups);
  int G = uniq.size();
  
  // 2) map each label → column index 0:(G-1)
  std::unordered_map<int,int> idx;
  idx.reserve(G);
  for (int g = 0; g < G; ++g) {
    idx[uniq[g]] = g;
  }
  
  // 3) accumulate sums and counts
  arma::mat alpha(T, G, arma::fill::zeros);
  arma::ivec counts(G, arma::fill::zeros);
  for (int j = 0; j < N; ++j) {
    int label = groups[j];
    int col   = idx[label];
    alpha.col(col) += Z.col(j);
    counts[col]     += 1;
  }
  
  // 4) convert sums → means
  for (int g = 0; g < G; ++g) {
    if (counts[g] > 0) {
      alpha.col(g) /= counts[g];
    }
  }
  
  return alpha;
}


// [[Rcpp::export]]
arma::vec computeSigma_cpp(List              zList,
                           const NumericMatrix& theta,
                           const IntegerVector& groupR) {
  // --- 1) Residual matrix Z (t × N)
  arma::mat Z = computeZ_cpp(zList, theta, groupR);
  
  // --- 2) Group centers α (t × G)
  arma::mat Alpha = computeAlpha_cpp(Z, groupR);
  
  // --- 3) Unique, sorted group labels
  IntegerVector gee = sort_unique(groupR);
  int G = gee.size();
  int N = groupR.size();
  
  // --- 4) Convert groupR to arma::ivec for fast indexing
  arma::ivec grp(N);
  for (int i = 0; i < N; ++i) grp[i] = groupR[i];
  
  // --- 5) Compute σ for each group
  arma::vec Sigma(G);
  for (int k = 0; k < G; ++k) {
    int label = gee[k];
    arma::uvec cols = arma::find(grp == label);
    
    if (cols.is_empty()) {
      Sigma[k] = 0.0;
    } else {
      // deviations from center α_{·,k}
      // grab as a true arma::mat
      arma::mat Zsub = Z.cols(cols);
      arma::mat dev  = Zsub.each_col() - Alpha.col(k);
      
      double msq = arma::accu(arma::square(dev)) / double(dev.n_elem);
      Sigma[k] = std::sqrt(msq);
    }
  }
  
  return Sigma;
}


// [[Rcpp::export]]
arma::vec computeResiduals_cpp(const Rcpp::List&        zList,
                               const Rcpp::NumericMatrix& theta,
                               const Rcpp::IntegerVector& groups,
                               const Rcpp::NumericMatrix& alpha) {
  int N = zList.size();
  arma::mat alpha_mat = Rcpp::as<arma::mat>(alpha);
  if ((int)groups.size() != N)
    Rcpp::stop("`groups` must have length N");
  // pull dims
  arma::mat z0 = Rcpp::as<arma::mat>(zList[0]);
  int T = z0.n_rows;
  int p = z0.n_cols - 1;
  
  // bring in θ and detect shape
  arma::mat th = Rcpp::as<arma::mat>(theta);
  int thr = th.n_rows, thc = th.n_cols;
  bool het = (thr == p && thc > 1);
  
  // pre‐allocate output
  arma::vec resid(N * T);
  
  for (int i = 0; i < N; ++i) {
    // pick β_i
    arma::colvec beta(p);
    if (het) {
      int g = groups[i] - 1;
      if (g < 0 || g >= thc) Rcpp::stop("invalid group at i=%d", i+1);
      beta = th.col(g);
    } else {
      if (thr == p && thc == 1) {
        beta = th.col(0);
      } else if (thr == 1 && thc == p) {
        beta = th.row(0).t();
      } else {
        Rcpp::stop("theta must be p×1 or 1×p for homogeneous");
      }
    }
    
    // fetch panel
    arma::mat zi = Rcpp::as<arma::mat>(zList[i]);
    // compute Y - Xβ
    arma::vec e = zi.col(0) - zi.cols(1, p) * beta;
    // subtract α_{·,g(i)}
    int gi = groups[i] - 1;
    arma::colvec alpha_i = alpha_mat.col( het ? gi : 0 );
    e -= alpha_i;
    
    // store
    for (int t = 0; t < T; ++t) {
      resid[i * T + t] = e[t];
    }
  }
  
  return resid;
}

// [[Rcpp::export]]
double gfeObj_cpp(const arma::mat& Z,
                  const Rcpp::IntegerVector& groups) {
  int T = Z.n_rows;
  int N = Z.n_cols;
  if ((int)groups.size() != N)
    Rcpp::stop("`groups` must have length N = Z.n_cols");
  
  // determine number of groups G
  int G = 0;
  for (int i = 0; i < N; ++i) {
    G = std::max(G, groups[i]);
  }
  
  // collect column‐indices for each group (0‐based)
  std::vector<std::vector<int>> idx(G);
  for (int i = 0; i < N; ++i) {
    int g = groups[i] - 1;
    if (g < 0 || g >= G)
      Rcpp::stop("`groups` contains invalid label at position %d", i+1);
    idx[g].push_back(i);
  }
  
  double total = 0.0;
  
  // loop over groups
  for (int g = 0; g < G; ++g) {
    const auto& members = idx[g];
    int sz = members.size();
    if (sz == 0) continue;  // no contribution
    
    // for each time period
    for (int t = 0; t < T; ++t) {
      // compute cross‐sectional mean at time t
      double sum = 0.0;
      for (int j : members) {
        sum += Z(t, j);
      }
      double mean = sum / sz;
      
      // accumulate squared deviations
      for (int j : members) {
        double d = Z(t, j) - mean;
        total += d * d;
      }
    }
  }
  
  return total;
}



// [[Rcpp::export]]
double wgfeObj_cpp(const arma::mat& Z,
                   const Rcpp::IntegerVector& groups) {
  int T = Z.n_rows;
  int N = Z.n_cols;
  if ((int)groups.size() != N)
    Rcpp::stop("`groups` must have length N = Z.n_cols");
  
  // determine number of groups G
  int G = 0;
  for (int i = 0; i < N; ++i) {
    G = std::max(G, groups[i]);
  }
  
  // count units per group
  arma::ivec countG(G, arma::fill::zeros);
  for (int i = 0; i < N; ++i) {
    int g = groups[i] - 1;
    if (g < 0 || g >= G)
      Rcpp::stop("invalid group label at position %d", i+1);
    ++countG[g];
  }
  
  // compute group‐wise residual SDs
  arma::vec SigmaG(G, arma::fill::zeros);
  for (int g = 0; g < G; ++g) {
    int ng = countG[g];
    if (ng == 0) continue;
    
    // collect columns for group g
    arma::mat Zg(T, ng);
    int col = 0;
    for (int i = 0; i < N; ++i) {
      if (groups[i] - 1 == g) {
        Zg.col(col++) = Z.col(i);
      }
    }
    
    // time‐specific mean (cross‐section over group)
    arma::vec mu = arma::mean(Zg, /*dim=*/1);  // length T
    // deviations
    arma::mat dev = Zg.each_col() - mu;        // T×ng
    // mean squared deviation
    double msq = arma::accu(arma::square(dev)) / double(dev.n_elem);
    SigmaG[g] = std::sqrt(msq);
  }
  
  // weighted average
  double out = 0.0;
  for (int g = 0; g < G; ++g) {
    out += SigmaG[g] * (double(countG[g]) / double(N));
  }
  
  return out;
}



// [[Rcpp::export]]
double wgfeObj_bs_cpp(
    List z,               // list of TT×(p+1) matrices
    NumericVector b,      // length-p coefficient vector
    IntegerVector group,  // length-N, values in {1,…,G}
    NumericVector sigg,   // length-G vector of group variances
    double sig            // scalar variance term
) {
  int N = z.size();            // number of individuals
  int p = b.size();            // number of regressors
  arma::vec bVec(b.begin(), p, false);
  
  // infer TT from first element of z
  arma::mat z0 = as<arma::mat>(z[0]);
  int TT = z0.n_rows;
  
  // 1) build Z matrix: TT × N, with residuals Y_i − X_i b
  arma::mat Z(TT, N);
  for (int i = 0; i < N; ++i) {
    arma::mat zi = as<arma::mat>(z[i]);
    Z.col(i) = zi.col(0) - zi.cols(1, p) * bVec;
  }
  
  // 2) compute per-group sums and counts
  int G = *std::max_element(group.begin(), group.end());
  arma::mat sumZ(TT, G, arma::fill::zeros);
  arma::vec count(G, arma::fill::zeros);
  for (int i = 0; i < N; ++i) {
    int g = group[i] - 1;           // zero-based index
    sumZ.col(g) += Z.col(i);
    count[g] += 1.0;
  }
  
  // 3) get group means alpha (TT × G)
  arma::mat alpha(TT, G);
  for (int g = 0; g < G; ++g) {
    alpha.col(g) = sumZ.col(g) / count[g];
  }
  
  // 4) accumulate SSR term exactly as in R:
  //    mean over i of [sigg[group[i]] + sig + mean((Z[,i]−α[,g])^2)/(sigg[g]+sig)]
  double acc = 0.0;
  for (int i = 0; i < N; ++i) {
    int g      = group[i] - 1;
    double sigg_val = sigg[g];
    arma::vec diff   = Z.col(i) - alpha.col(g);
    double mse       = arma::dot(diff, diff) / TT;        // mean squared deviation
    acc += sigg_val + sig + mse / (sigg_val + sig);
  }
  
  return acc / (N);
}



// [[Rcpp::export]]
arma::vec slopeGradGFE_cpp(List z, IntegerVector group) {
  int N = z.size();
  if (N == 0) return arma::vec();
  
  // 1) infer dimensions
  arma::mat Z0 = as<arma::mat>(z[0]);
  int T = Z0.n_rows, K = Z0.n_cols, p = K - 1;
  
  // 2) pull the list into a C++ vector so we only convert once
  std::vector<arma::mat> Zdata(N);
  for (int i = 0; i < N; ++i) {
    Zdata[i] = as<arma::mat>(z[i]);
  }
  
  // 3) compute group sums & counts
  int G = max(group);
  std::vector<arma::mat> gSum(G, arma::mat(T, K, arma::fill::zeros));
  std::vector<int>       gCount(G, 0);
  
  for (int i = 0; i < N; ++i) {
    int gi = group[i] - 1;           // zero‑based
    gSum[gi] += Zdata[i];
    gCount[gi] += 1;
  }
  
  // 4) compute group means
  std::vector<arma::mat> gAve(G);
  for (int g = 0; g < G; ++g) {
    if (gCount[g] > 0) 
      gAve[g] = gSum[g] / gCount[g];
    else 
      gAve[g].zeros(T, K);
  }
  
  // 5) build the big "D" matrix of within‑group deviations:
  //    dimensions (N*T) × K
  arma::mat D(N * T, K);
  for (int i = 0; i < N; ++i) {
    int row0 = i * T;
    D.rows(row0, row0 + T - 1) = Zdata[i] - gAve[group[i] - 1];
  }
  
  // 6) slice off X and y, then let BLAS do the work
  arma::mat X   = D.cols(1, K - 1); 
  arma::vec y   = D.col(0);
  arma::mat XtX = X.t() * X;      // p×p
  arma::vec Xty = X.t() * y;      // p×1
  
  // 7) solve the system
  return arma::solve(XtX, Xty);
}


// [[Rcpp::export]]
arma::mat calcGroupSlopes_cpp(const Rcpp::List&          zList,
                              const Rcpp::IntegerVector& groups) {
  int N = zList.size();
  if ((int)groups.size() != N) {
    Rcpp::stop("`groups` must have length N = zList.size()");
  }
  if (N == 0) {
    return arma::mat(); 
  }
  
  // Determine the number of groups
  int G = Rcpp::as<arma::ivec>(groups).max();
  
  // Pre‐cast all zList entries to arma::mat
  std::vector<arma::mat> Zdata(N);
  for (int i = 0; i < N; ++i) {
    Zdata[i] = Rcpp::as<arma::mat>(zList[i]);  // each is T × (p+1)
  }
  
  // Infer T and p from the first element
  int T = Zdata[0].n_rows;
  int K = Zdata[0].n_cols;   // = p + 1
  int p = K - 1;
  
  // Build list of unit‐indices for each group (1‐based labels → zero‐based index)
  std::vector<std::vector<int>> group_members(G);
  for (int i = 0; i < N; ++i) {
    int g = groups[i];
    if (g >= 1 && g <= G) {
      group_members[g - 1].push_back(i);
    }
  }
  
  // Prepare output: p × G
  arma::mat betas(p, G, arma::fill::zeros);
  
  // Loop over each group g = 1..G
  for (int g = 1; g <= G; ++g) {
    const auto& members = group_members[g - 1];
    int n_g = members.size();
    
    if (n_g == 0) {
      // No units in this group → fill NaNs
      betas.col(g - 1).fill(arma::datum::nan);
      continue;
    }
    
    // 1) Compute within‐group time‐series means for y and x
    arma::vec mean_y(T, arma::fill::zeros);
    arma::mat mean_x(T, p, arma::fill::zeros);
    
    // Sum up y‐values and x‐values across all members
    for (int idx_i : members) {
      const arma::mat& Zi = Zdata[idx_i];  // T × (p+1)
      mean_y += Zi.col(0);                // sum y_{i,t} over i
      mean_x += Zi.cols(1, p);            // sum x_{i,t,·} over i
    }
    
    // Divide by group size to get time‐by‐time means
    mean_y /= static_cast<double>(n_g);
    mean_x /= static_cast<double>(n_g);
    
    // 2) Accumulate XtX and Xty using demeaned values
    arma::mat XtX(p, p, arma::fill::zeros);
    arma::vec Xty(p,    arma::fill::zeros);
    
    for (int idx_i : members) {
      const arma::mat& Zi = Zdata[idx_i];  // T × (p+1)
      for (int t = 0; t < T; ++t) {
        double y_dev = Zi(t, 0) - mean_y[t];
        for (int a = 0; a < p; ++a) {
          double x_dev_a = Zi(t, a + 1) - mean_x(t, a);
          Xty[a] += x_dev_a * y_dev;
          for (int b = 0; b < p; ++b) {
            double x_dev_b = Zi(t, b + 1) - mean_x(t, b);
            XtX(a, b) += x_dev_a * x_dev_b;
          }
        }
      }
    }
    
    // 3) Solve the within‐group OLS: β_g = (X˜' X˜)^{-1} (X˜' y˜)
    if (XtX.is_zero() || arma::rank(XtX) < p) {
      betas.col(g - 1).fill(arma::datum::nan);
    } else {
      betas.col(g - 1) = arma::solve(XtX, Xty);
    }
  }
  
  return betas;
}



// [[Rcpp::export]]
List computeXXY_demeaned_cpp(List              zList,
                             const IntegerVector& wgroups) {
  int N = zList.size();
  // infer dimensions from first element
  NumericMatrix z0 = zList[0];
  int T = z0.nrow();
  int p = z0.ncol() - 1;
  
  // unique group labels and mapping
  IntegerVector uniq = sort_unique(wgroups);
  int G = uniq.size();
  std::unordered_map<int,int> label2k;
  label2k.reserve(G);
  for (int k = 0; k < G; ++k) {
    label2k[ uniq[k] ] = k;
  }
  
  // 1) compute group means g_ave[g] = mean of zList for group g
  std::vector<arma::mat> g_ave(G, arma::zeros<arma::mat>(T, p+1));
  arma::ivec counts(G, arma::fill::zeros);
  for (int i = 0; i < N; ++i) {
    NumericMatrix zRi = zList[i];
    arma::mat Zi(zRi.begin(), T, p+1, false);
    int k = label2k[ wgroups[i] ];
    g_ave[k] += Zi;
    counts[k] += 1;
  }
  for (int k = 0; k < G; ++k) {
    if (counts[k] > 0) {
      g_ave[k] /= double(counts[k]);
    }
  }
  
  // 2) allocate output lists
  List XX_list(N);
  List y_list(N);
  
  // 3) for each unit, form D_i and compute XX and y‐demeaned
  for (int i = 0; i < N; ++i) {
    NumericMatrix zRi = zList[i];
    arma::mat Zi(zRi.begin(), T, p+1, false);
    int k = label2k[ wgroups[i] ];
    arma::mat Di = Zi - g_ave[k];             // demeaned T×(p+1)
    
    // split into X_i (T×p) and demeaned pieces
    arma::mat X  = Zi.cols(1, p);             // T×p
    arma::mat D2 = Di.cols(1, p);             // T×p
    arma::vec D1 = Di.col(0);                 // T-vector
    
    // compute XX_i = Xᵀ D2  (p×p) and y_i = Xᵀ D1 (p-vector)
    arma::mat XX = X.t() * D2;                // p×p
    arma::vec y  = X.t() * D1;                // p
    
    XX_list[i] = XX;
    y_list[i]  = y;
  }
  
  return List::create(
    _["XX_demeaned"] = XX_list,
    _["y_demeaned"]  = y_list
  );
}



// [[Rcpp::export]]
arma::vec wc_fp_cpp(List               zList,
                    const arma::vec&   b,
                    const IntegerVector& wgroups,
                    List               XX_list,
                    List               y_list,
                    const IntegerVector& gee) {
  int N = zList.size();
  int p = b.n_elem;
  
  // infer T
  NumericMatrix z0 = zList[0];
  int T = z0.nrow();
  
  // 1) build residual matrix Z (T × N)
  arma::mat Z(T, N);
  for (int i = 0; i < N; ++i) {
    NumericMatrix zRi = zList[i];
    arma::mat Zi(zRi.begin(), T, p+1, false);
    Z.col(i) = Zi.col(0) - Zi.cols(1, p) * b;
  }
  
  // convert groups to arma::ivec
  arma::ivec grp(N);
  for (int i = 0; i < N; ++i) grp[i] = wgroups[i];
  int G = gee.size();
  
  // 2) compute σ_g for each group
  arma::vec SigmaG(G);
  for (int k = 0; k < G; ++k) {
    int label = gee[k];
    arma::uvec idx = arma::find(grp == label);
    if (idx.is_empty()) {
      SigmaG[k] = 0.0;
    } else {
      arma::mat Zg = Z.cols(idx);
      arma::vec m  = arma::mean(Zg, 1);
      arma::mat dev = Zg.each_col() - m;
      double msq   = arma::accu(arma::square(dev)) / double(dev.n_elem);
      SigmaG[k]    = std::sqrt(msq);
    }
  }
  
  // 3) assemble A and B
  arma::mat A = arma::zeros<arma::mat>(p, p);
  arma::vec B = arma::zeros<arma::vec>(p);
  for (int k = 0; k < G; ++k) {
    double sigma = SigmaG[k];
    if (sigma <= 0) continue;
    int label = gee[k];
    arma::uvec idx = arma::find(grp == label);
    
    arma::mat sumXX(p, p, arma::fill::zeros);
    arma::vec sumY(p, arma::fill::zeros);
    for (arma::uword j: idx) {
      NumericMatrix xMat = XX_list[j];
      arma::mat XXi(xMat.begin(), p, p, false);
      sumXX += XXi;
      NumericVector yVec = y_list[j];
      arma::vec yi(yVec.begin(), p, false);
      sumY += yi;
    }
    A += sumXX / sigma;
    B += sumY  / sigma;
  }
  
  // 4) solve for θ
  return arma::solve(A, B);
}



// [[Rcpp::export]]
NumericVector wc_fp_bs_cpp(
    List z,                 // list of TT×(p+1) matrices
    NumericVector b,        // length-p coefficient vector
    double s,               // scalar s
    IntegerVector groups,   // length-N, values in {1,…,G}
    List XX_demeaned,       // list of p×p matrices
    List y_demeaned,        // list of length-p vectors
    NumericVector sigs      // length-G vector of sig_g
) {
  // Dimensions
  int N = z.size();
  int p = b.size();
  
  // Load lists into Armadillo structures to avoid repeated conversions
  std::vector<arma::mat>   zList(N), XXlist(N);
  std::vector<arma::vec>   yList(N);
  for (int i = 0; i < N; ++i) {
    zList[i]   = as<arma::mat>(z[i]);
    XXlist[i]  = as<arma::mat>(XX_demeaned[i]);
    yList[i]   = as<arma::vec>(y_demeaned[i]);
  }
  
  // Infer TT from the first z
  int TT = zList[0].n_rows;
  
  // Make b into arma::vec
  arma::vec bVec(b.begin(), p, /*copy_memory=*/false);
  
  // 1) Build Z matrix of residuals: TT × N
  arma::mat Z(TT, N);
  for (int i = 0; i < N; ++i) {
    // residual = Y_i − X_i b
    Z.col(i) = zList[i].col(0)
    - zList[i].cols(1, p) * bVec;
  }
  
  // 2) Determine number of groups G
  int G = *std::max_element(groups.begin(), groups.end());
  
  // 3) Accumulate per-group sums of XX_demeaned and y_demeaned
  std::vector<arma::mat> sumXXg(G, arma::mat(p, p, arma::fill::zeros));
  std::vector<arma::vec> sumyg(G, arma::vec(p, arma::fill::zeros));
  std::vector<int>       countG(G, 0);
  
  for (int i = 0; i < N; ++i) {
    int g = groups[i] - 1;             // zero-based
    sumXXg[g] += XXlist[i];
    sumyg[g]  += yList[i];
    ++countG[g];
  }
  
  // 4) Form the weighted normal equations: A * slope = C
  arma::mat A(p, p, arma::fill::zeros);
  arma::vec C(p, arma::fill::zeros);
  for (int g = 0; g < G; ++g) {
    double denom = s + sigs[g];
    A += sumXXg[g] / denom;
    C += sumyg[g]  / denom;
  }
  arma::vec slope = arma::solve(A, C);
  
  // 5) Compute SigmaG[g] = mean of squared deviations in Z for each group
  std::vector<double> SigmaG(G, 0.0);
  for (int g = 0; g < G; ++g) {
    int k = countG[g];
    if (k == 0) continue;
    // collect columns for group g
    arma::mat Zg(TT, k);
    int idx = 0;
    for (int i = 0; i < N; ++i) {
      if (groups[i] - 1 == g)
        Zg.col(idx++) = Z.col(i);
    }
    // row-means of Zg
    arma::vec mu = arma::mean(Zg, /*dim=*/1);
    // sum of squared deviations
    double sumsq = 0;
    for (int j = 0; j < k; ++j) {
      arma::vec d = Zg.col(j) - mu;
      sumsq += arma::dot(d, d);
    }
    SigmaG[g] = sumsq / (double(k) * TT);
  }
  
  // 6) Identify the “h” group where sigs[h]==0
  int h = -1;
  for (int g = 0; g < G; ++g) {
    if (sigs[g] == 0.0) { h = g; break; }
  }
  
  // 7) Compute the “bot” term and the variance
  double invN = 1.0 / N;
  double bot  = 0.0;
  for (int g = 0; g < G; ++g) {
    if (g == h) continue;
    double denom = s + sigs[g];
    bot += invN * countG[g] * SigmaG[g] / (denom * denom);
  }
  double var = invN * countG[h] * SigmaG[h] + s*s * bot;
  double se  = std::sqrt(var);
  
  // 8) Return slope (length p) plus sqrt(var) as the (p+1)-th element
  NumericVector out(p + 1);
  for (int j = 0; j < p; ++j) out[j] = slope[j];
  out[p] = se;
  return out;
}



// [[Rcpp::export]]
arma::vec wc_fp_loop_cpp(List               zList,
                         const IntegerVector& wgroups,
                         List               XX_list,
                         List               y_list,
                         const IntegerVector& gee,
                         double             tol      = 1e-13,
                         int                max_iter = 1000) {
  // infer p from first y_list element
  NumericVector y0 = y_list[0];
  int p = y0.size();
  
  // initialize theta
  arma::vec theta0 = arma::zeros<arma::vec>(p);
  arma::vec theta  = wc_fp_cpp(zList, theta0, wgroups, XX_list, y_list, gee);
  
  // fixed-point loop
  double diff = arma::norm(theta - theta0, 2);
  int iter = 0;
  while (diff > tol && iter < max_iter) {
    theta0 = theta;
    theta  = wc_fp_cpp(zList, theta0, wgroups, XX_list, y_list, gee);
    diff   = arma::norm(theta - theta0, 2);
    ++iter;
  }
  
  return theta;
}



// [[Rcpp::export]]
arma::vec gfeJump_cpp(IntegerVector replaceR,
                      int i,
                      IntegerVector grR,
                      NumericMatrix Z_N,
                      NumericMatrix alpha_N) {
  // — 1) Convert inputs to Armadillo types
  int N   = grR.size();
  int R   = replaceR.size();
  int idx = i - 1; // one‐based -> zero‐based
  
  arma::uvec rep   = as<arma::uvec>(replaceR) - 1; // zero‐based group values
  arma::uvec gr0   = as<arma::uvec>(grR)        - 1; // zero‐based current groups
  arma::mat  Z     = as<arma::mat>(Z_N);            // T×N matrix
  arma::mat  alpha = as<arma::mat>(alpha_N);        // T×G matrix
  int        T     = Z.n_rows;
  
  // — 2) Compute gee0 = sorted unique(original gr)
  arma::uvec gee0 = arma::unique(gr0);
  gee0 = arma::sort(gee0);
  int        G    = gee0.n_elem;
  
  // — 3) Preallocate objective vector
  arma::vec obj(R);
  
  // — 4) Loop over each candidate replacement
  for (int r = 0; r < R; ++r) {
    arma::uvec grTry = gr0;
    grTry[idx]       = rep[r];
    
    double total = 0.0;
    // for each group label g in gee0
    for (int k = 0; k < G; ++k) {
      int g = gee0[k];
      arma::uvec members = arma::find(grTry == g);
      if (members.is_empty()) continue;
      
      // sub‐matrix of Z for this group
      arma::mat Zg = Z.cols(members);
      
      // the column of alpha for this group
      arma::vec a = alpha.col(g);
      
      // within‑group deviations
      arma::mat D = Zg.each_col() - a;
      
      // accumulate sum of squares
      total += arma::accu(D % D);
    }
    
    obj[r] = total;
  }
  
  // — 5) Pick the best replacement
  arma::uword best = obj.index_min();
  
  // build the output: first the min obj, then the new grouping (1‑based)
  arma::uvec grBest = gr0;
  grBest[idx]       = rep[best];
  
  arma::vec out(N + 1);
  out[0]             = obj[best];
  out.subvec(1, N)   = arma::conv_to<arma::vec>::from(grBest + 1);
  
  return out;
}


// [[Rcpp::export]]
List wgfe_loop_res_cpp(
    List            z,           // list of TT×(p+1) matrices
    NumericVector   b0,          // initial slope (length p)
    double          sig0,        // initial sigma
    IntegerVector   wgroups,     // length-N grouping (1…G)
    NumericVector   sigsg,       // group‐variance vector (length G)
    double          tol = 1e-13, // convergence tolerance
    int             maxiter = 1000
) {
  int N = z.size();
  // infer dimensions
  arma::mat zfirst = as<arma::mat>(z[0]);
  int TT = zfirst.n_rows;
  int p  = zfirst.n_cols - 1;
  
  // build list of arma::mat for z
  std::vector<arma::mat> zList(N);
  for (int i = 0; i < N; ++i)
    zList[i] = as<arma::mat>(z[i]);
  
  // unique groups
  int G = *std::max_element(wgroups.begin(), wgroups.end());
  NumericVector gee(G);
  for (int g = 0; g < G; ++g) gee[g] = g + 1;
  
  // 1) compute group means g_ave
  std::vector<arma::mat> sumZ(G, arma::mat(TT, p+1, arma::fill::zeros));
  std::vector<int>       countG(G, 0);
  for (int i = 0; i < N; ++i) {
    int g = wgroups[i] - 1;
    sumZ[g] += zList[i];
    ++countG[g];
  }
  std::vector<arma::mat> gAve(G);
  for (int g = 0; g < G; ++g)
    gAve[g] = sumZ[g] / countG[g];
  
  // 2) precompute XX_demeaned and y_demeaned
  List XXdem(N), ydem(N);
  for (int i = 0; i < N; ++i) {
    int g = wgroups[i] - 1;
    arma::mat D = zList[i] - gAve[g];
    arma::mat X = zList[i].cols(1, p);
    XXdem[i] = X.t() * D.cols(1, p);
    ydem [i] = X.t() * D.col(0);
  }
  
  // 3) fixed‐point loop
  NumericVector bR = b0;
  double s = sig0;
  NumericVector delta(p+1), delta0(p+1);
  int iter = 0;
  
  delta = wc_fp_bs_cpp(z, bR, s, wgroups, XXdem, ydem, sigsg);
  while (arma::norm(arma::vec(delta.begin(), p+1, false)
                      - arma::vec(delta0.begin(), p+1, false)) > tol
           && ++iter < maxiter) {
           delta0 = delta;
    for (int j = 0; j < p; ++j) bR[j] = delta0[j];
    s = delta0[p];
    delta = wc_fp_bs_cpp(z, bR, s, wgroups, XXdem, ydem, sigsg);
  }
  
  // return final slope and sigma
  NumericVector slope(p);
  for (int j = 0; j < p; ++j) slope[j] = bR[j];
  return List::create(
    Named("slope") = slope,
    Named("sigma") = s
  );
}


// [[Rcpp::export]]
NumericVector wgfeJump_cpp(IntegerVector replace,
                           int         i,
                           IntegerVector gr,
                           NumericMatrix Z_,
                           NumericMatrix alpha_) {
  // wrap without copying
  arma::mat Z     (Z_.begin(),      Z_.nrow(),     Z_.ncol(), false);
  arma::mat alpha (alpha_.begin(),  alpha_.nrow(), alpha_.ncol(), false);
  
  int n = gr.size();
  int G = alpha.n_cols;
  int R = replace.size();
  
  // convert everything to 0-based
  arma::uvec replace0(R);
  for(int k = 0; k < R; ++k) replace0[k] = replace[k] - 1;
  arma::uword i0 = (arma::uword)(i - 1);
  
  arma::uvec gr0(n);
  for(int j = 0; j < n; ++j) gr0[j] = gr[j] - 1;
  
  // ret: (1 + n) × R matrix of [objective; gr]
  arma::mat ret(n + 1, R);
  
  for(int k = 0; k < R; ++k) {
    // reassign
    gr0[i0] = replace0[k];
    
    arma::vec group_rmse(G);
    arma::vec weights (G);
    
    // loop over each possible group
    for(arma::uword g = 0; g < (arma::uword)G; ++g) {
      arma::uvec idx = arma::find(gr0 == g);
      if(idx.is_empty()) {
        group_rmse[g] = 0;
        weights   [g] = 0;
      } else {
        arma::mat Zsub = Z.cols(idx);
        arma::vec  a   = alpha.col(g);
        // subtract center and square
        arma::mat diff = Zsub.each_col() - a;
        double mse     = arma::accu(arma::square(diff)) / double(Zsub.n_elem);
        group_rmse[g]  = std::sqrt(mse);
        weights   [g]  = double(idx.n_elem) / double(n);
      }
    }
    
    // weighted mean of the group RMSEs
    ret(0, k) = arma::dot(group_rmse, weights);
    
    // store the updated grouping (back to 1-based)
    for(int j = 0; j < n; ++j) {
      ret(j + 1, k) = double(gr0[j] + 1);
    }
  }
  
  // pick the column with minimal objective
  arma::uword minIdx;
  ret.row(0).min(minIdx);
  arma::vec best = ret.col(minIdx);
  
  // return as R vector
  return NumericVector(best.begin(), best.end());
}


// [[Rcpp::export]]
NumericVector wgfeJump_bs_cpp(
    IntegerVector replace,        // candidate group labels (1…G)
    int            i,             // index (1-based) of unit to reassign
    IntegerVector  gr_in,         // current grouping vector (length N)
    const arma::mat& Z,           // TT × N residual matrix
    const arma::mat& alpha,       // TT × G group means
    NumericVector  sigs,          // length-G group variances
    double         sig            // scalar variance term
) {
  int N   = Z.n_cols;
  int TT  = Z.n_rows;
  int idx = i - 1;               // convert to 0-based index
  
  // Clone the original groups so we can reuse it
  IntegerVector orig_gr = clone(gr_in);
  
  // Precompute baseline per-unit contributions
  arma::vec val(N);
  for (int j = 0; j < N; ++j) {
    int g0        = orig_gr[j] - 1;
    arma::vec d   = Z.col(j) - alpha.col(g0);
    double mse_j  = arma::dot(d, d) / TT;
    val[j]        = sigs[g0] + mse_j / (sigs[g0] + sig);
  }
  double sum_val = arma::sum(val);
  
  // Search over replacements
  double best_obj = R_PosInf;
  IntegerVector best_gr;
  int R = replace.size();
  for (int k = 0; k < R; ++k) {
    int r_new      = replace[k];
    int g_new      = r_new - 1;
    
    // Compute new contribution for unit i
    arma::vec di   = Z.col(idx) - alpha.col(g_new);
    double mse_i   = arma::dot(di, di) / TT;
    double val_i   = sigs[g_new] + mse_i / (sigs[g_new] + sig);
    
    // Update objective
    double sum_r   = sum_val - val[idx] + val_i;
    double obj_r   = sum_r / N + sig;
    
    if (obj_r < best_obj) {
      best_obj = obj_r;
      best_gr  = clone(orig_gr);
      best_gr[idx] = r_new;
    }
  }
  
  // Return a numeric vector: [objective, new groups...]
  NumericVector out(N + 1);
  out[0] = best_obj;
  for (int j = 0; j < N; ++j) out[j + 1] = best_gr[j];
  return out;
}

// [[Rcpp::export]]
Rcpp::IntegerVector assignGroups_cpp(
    const Rcpp::List&         zList,
    const Rcpp::NumericMatrix& theta,
    const arma::mat&          alpha
) {
  int N = zList.size();
  int T = alpha.n_rows;
  int G = alpha.n_cols;
  
  // Load θ into Armadillo
  arma::mat th = Rcpp::as<arma::mat>(theta);
  int p = th.n_rows;   // number of covariates
  int C = th.n_cols;   // if C > 1 => heterogeneous
  
  // Decide homogeneous vs. heterogeneous
  bool het = (p > 1 && C > 1);
  arma::colvec b;
  if (!het) {
    if (th.n_cols == 1 && th.n_rows == p) {
      b = th.col(0);
    } else if (th.n_rows == 1 && th.n_cols == p) {
      b = th.row(0).t();
    } else {
      Rcpp::stop("For homogeneous theta, θ must be p×1 or 1×p.");
    }
  }
  
  Rcpp::IntegerVector out(N);
  
  // ------------------------------------------------
  // Loop over each unit i = 0..N-1
  // ------------------------------------------------
  for (int i = 0; i < N; ++i) {
    // Pull z_i (a T×(p+1) matrix) from the list
    arma::mat zi = Rcpp::as<arma::mat>(zList[i]);
    if (zi.n_rows != T || zi.n_cols != p + 1) {
      Rcpp::stop("Every zList[[i]] must be T×(p+1).");
    }
    
    // Split zi into Y_i (T×1) and X_i (T×p):
    arma::colvec Yi = zi.col(0);         // y_{i,1}, …, y_{i,T}
    arma::mat   Xi = zi.cols(1, p);      // the p covariates
    
    double bestSSE = R_PosInf;
    int    bestG   = 1;  // default to group 1 if there is a tie or no data
    
    // ------------------------------------------------
    // Loop over each group g = 0..G-1
    // ------------------------------------------------
    for (int g = 0; g < G; ++g) {
      // Choose β_g: if heterogeneous, use th.col(g); otherwise use the single b
      arma::colvec beta_g = het ? th.col(g) : b;
      
      // We accumulate SSE over those t where both y_{it} and x_{it⋅} are finite.
      // If ANY component of x_{it⋅} is NaN, or y_{it} is NaN, we skip that t entirely.
      double sse = 0.0;
      
      for (int t_idx = 0; t_idx < T; ++t_idx) {
        double y_it = Yi[t_idx];
        if (! std::isfinite(y_it)) {
          // y_{it} is NA → skip
          continue;
        }
        
        // Check X_i(t_idx, ⋅) row for finiteness:
        bool rowOK = true;
        for (int col = 0; col < p; ++col) {
          double x_val = Xi(t_idx, col);
          if (! std::isfinite(x_val)) {
            rowOK = false;
            break;
          }
        }
        if (! rowOK) {
          // at least one x_{itj} is NA → skip
          continue;
        }
        
        // Now we know y_{it} and all x_{itj} are finite.
        // Compute residual = y_{it} - x_{it⋅}'·β_g
        double dotprod = arma::as_scalar( Xi.row(t_idx) * beta_g );
        double resid = y_it - dotprod;
        
        // Subtract α_{g,t} (note that α is stored as T×G, so alpha(t_idx,g) is α_{g,t})
        double diff = resid - alpha(t_idx, g);
        
        sse += (diff * diff);
      }
      
      if (sse < bestSSE) {
        bestSSE = sse;
        bestG   = g + 1;  // convert back to 1-based indexing
      }
    }
    
    out[i] = bestG;
  }
  
  return out;
}


// [[Rcpp::export]]
Rcpp::IntegerVector assignGroups_wgfe_cpp(const Rcpp::List&         zList,
                                          const Rcpp::NumericMatrix& theta,
                                          const arma::mat&           Alpha,
                                          const arma::vec&           SigmaG) {
  int N = zList.size();
  int T = Alpha.n_rows;
  int G = Alpha.n_cols;
  if ((int)SigmaG.n_elem != G)
    Rcpp::stop("SigmaG must be length G = Alpha.n_cols");
  
  // bring theta into Armadillo
  arma::mat th = Rcpp::as<arma::mat>(theta);
  int p = th.n_rows, C = th.n_cols;
  // detect heterogeneous
  bool het = (p > 1 && C > 1);
  
  // extract homogeneous b if needed
  arma::colvec b; 
  if (!het) {
    if (th.n_cols == 1 && th.n_rows == p) {
      b = th.col(0);
    } else if (th.n_rows == 1 && th.n_cols == p) {
      b = th.row(0).t();
    } else {
      Rcpp::stop("For homogeneous: theta must be p×1 or 1×p");
    }
  }
  
  Rcpp::IntegerVector out(N);
  
  // precompute these to avoid repeated work
  arma::vec invS = 1.0 / SigmaG;
  
  for (int i = 0; i < N; ++i) {
    // load unit‐i panel
    arma::mat zi = Rcpp::as<arma::mat>(zList[i]);
    if (zi.n_rows != T || zi.n_cols != p+1)
      Rcpp::stop("All zList[[i]] must be T×(p+1)");
    
    arma::colvec Yi = zi.col(0);
    arma::mat   Xi = zi.cols(1, p);
    
    double bestObj = R_PosInf;
    int    bestG   = 1;
    
    // try each candidate group
    for (int g = 0; g < G; ++g) {
      // choose the right slope for group g
      arma::colvec beta_g = het ? th.col(g) : b;
      
      // residuals under this candidate slope
      arma::colvec resid = Yi - Xi * beta_g;
      // deviation from centroid
      arma::colvec diff  = resid - Alpha.col(g);
      double sumDevSq    = arma::dot(diff, diff);
      
      double obj = sumDevSq * invS[g]
      + double(T) * SigmaG[g];
      
      if (obj < bestObj) {
        bestObj = obj;
        bestG   = g + 1;   // back to 1-based
      }
    }
    
    out[i] = bestG;
  }
  
  return out;
}

// [[Rcpp::export]]
arma::uvec assignGroups_res_cpp(
    const arma::mat& Z,       // T × N data
    const arma::mat& alpha,   // T × G centers
    const arma::vec& sigsg,   // length-G group variances
    double sig0               // scalar extra variance
) {
  int T = Z.n_rows;
  int N = Z.n_cols;
  int G = sigsg.n_elem;
  
  // Prepare output
  arma::uvec wgroups(N);
  
  // Precompute 1/(sigsg[g]+sig0) and T*sigsg[g]
  arma::vec invDen(G), addTerm(G);
  for(int g = 0; g < G; ++g) {
    invDen[g]  = 1.0 / (sigsg[g] + sig0);
    addTerm[g] = sigsg[g] * double(T);
  }
  
  // For each column j, find the g minimizing the objective
  for(int j = 0; j < N; ++j) {
    double bestObj = arma::datum::inf;
    int    bestG   = 0;               // 0-based index
    
    for(int g = 0; g < G; ++g) {
      double sumDevSq = 0.0;
      // sum_{t=1}^T (Z(t,j)-alpha(t,g))^2
      for(int t = 0; t < T; ++t) {
        double d = Z(t,j) - alpha(t,g);
        sumDevSq += d*d;
      }
      // objective = sumDevSq/(sigsg[g]+sig0) + T*sigsg[g]
      double obj = sumDevSq * invDen[g] + addTerm[g];
      if (obj < bestObj) {
        bestObj = obj;
        bestG   = g;
      }
    }
    
    wgroups[j] = bestG + 1;
  }
  
  return wgroups;
}



// [[Rcpp::export]]
Rcpp::IntegerVector refineGroups_cpp(Rcpp::List            zList,
                                     Rcpp::IntegerVector   wgroups,
                                     bool                  heterogeneous,
                                     const std::string&    method) {
  int N = wgroups.size();
  int G = 0;
  for (int i = 0; i < N; ++i) {
    G = std::max(G, wgroups[i]);
  }
  Rcpp::IntegerVector prev(N);
  Rcpp::NumericMatrix thetaR;
  arma::mat            Z, alpha;
  arma::vec            SigmaG;
  
  Rcpp::IntegerVector gee = Rcpp::IntegerVector(G);
  for (int g = 0; g < G; ++g) {
    gee[g] = g + 1;
  }
  bool changed = true;
  
  while (changed) {
    prev = Rcpp::clone(wgroups);
    
    // --- 1) compute Z via either homogeneous or heterogeneous slopes
    arma::mat Z;
    if (!heterogeneous) {
      
      if (method == "gfe") {
        // homogeneous GFE
        arma::vec thv = slopeGradGFE_cpp(zList, wgroups);
        thetaR       = Rcpp::wrap( arma::mat(thv) );        // p×1
        Z            = computeZ_cpp(zList, thetaR, wgroups);
        alpha        = computeAlpha_cpp(Z, wgroups);
        wgroups      = assignGroups_cpp(zList, thetaR, alpha);
        
      } else {
        // homogeneous WGFE
        List tmp    = computeXXY_demeaned_cpp(zList, wgroups);
        arma::mat tm = wc_fp_loop_cpp(zList, wgroups, tmp[0], tmp[1], gee);
        thetaR       = Rcpp::wrap( tm );                    // p×1
        Z            = computeZ_cpp(zList, thetaR, wgroups);
        alpha        = computeAlpha_cpp(Z, wgroups);
        SigmaG       = computeSigma_cpp(zList, thetaR, wgroups);
        wgroups      = assignGroups_wgfe_cpp(zList, thetaR, alpha, SigmaG);
      }
      
    } else {
      
      if (method == "gfe") {
        // heterogeneous GFE
        arma::mat thm = calcGroupSlopes_cpp(zList, wgroups);
        thetaR        = Rcpp::wrap(thm);                    // p×G
        Z             = computeZ_cpp(zList, thetaR, wgroups);
        alpha         = computeAlpha_cpp(Z, wgroups);
        wgroups       = assignGroups_cpp(zList, thetaR, alpha);
        
      } else {
        // heterogeneous WGFE
        arma::mat thm = calcGroupSlopes_cpp(zList, wgroups);
        thetaR        = Rcpp::wrap(thm);                    // p×G
        Z             = computeZ_cpp(zList, thetaR, wgroups);
        alpha         = computeAlpha_cpp(Z, wgroups);
        SigmaG        = computeSigma_cpp(zList, thetaR, wgroups);
        wgroups       = assignGroups_wgfe_cpp(zList, thetaR, alpha, SigmaG);
      }
    }
    
    changed = false;
    for (int i = 0; i < N; ++i) {
      if (wgroups[i] != prev[i]) {
        changed = true;
        break;
      }
    }
  }
  
  return wgroups;
}

// [[Rcpp::export]]
List refineGroups_res_cpp(
    List            zList,
    IntegerVector   wgroups,
    double          sig0,
    NumericVector   sigsg
) {
  int N = wgroups.size();
  NumericMatrix z0R = as<NumericMatrix>(zList[0]);
  int p = z0R.ncol() - 1;
  
  NumericVector slope0(p, 0.0);
  NumericVector delta0(p + 1, 0.0);
  NumericVector delta(p + 1);
  IntegerVector groupings(N, 0);
  
  // Iterate until group labels stabilize
  while (!std::equal(wgroups.begin(), wgroups.end(), groupings.begin())) {
    groupings = clone(wgroups);
    
    // Compute demeaned XX and y
    List tmp2 = computeXXY_demeaned_cpp(zList, wgroups);
    
    // Initial parameter update
    delta = wc_fp_bs_cpp(
      zList, slope0, sig0, wgroups,
      tmp2[0], tmp2[1], sigsg
    );
    for (int i = 0; i < p; ++i) slope0[i] = delta[i];
    sig0 = delta[p];
    
    // Refine until convergence
    double diff;
    do {
      delta0 = delta;
      delta = wc_fp_bs_cpp(
        zList, slope0, sig0, wgroups,
        tmp2[0], tmp2[1], sigsg
      );
      for (int i = 0; i < p; ++i) slope0[i] = delta[i];
      sig0 = delta[p];
      
      diff = 0.0;
      for (int i = 0; i <= p; ++i) {
        double d = delta[i] - delta0[i];
        diff += d * d;
      }
    } while (diff > 1e-13);
    
    // Recompute Z and alpha and reassign groups
    arma::mat Z     = computeZ_res_cpp(zList, slope0);
    arma::mat alpha = computeAlpha_cpp(Z, wgroups);
    wgroups = assignGroups_res_cpp(Z, alpha, sigsg, sig0);
  }
  
  return List::create(
    _["wgroups"] = wgroups,
    _["slope0"]   = slope0,
    _["sig0"]     = sig0
  );
}


// [[Rcpp::export]]
IntegerVector localJump_cpp(IntegerVector  wgroups,
                            arma::mat      Z,
                            arma::mat      alpha,
                            IntegerVector  gee,
                            const std::string& method) {
  int N = wgroups.size();
  // 1) initial objective
  double oldObj;
  if (method == "gfe") {
    // use the original GFE objective
    oldObj = gfeObj_cpp(Z, wgroups);
  } else {
    // use the WGFE‐specific objective
    oldObj = wgfeObj_cpp(Z, wgroups);
  }
  
  // 2) prepare 1‑based loop index, and counter
  int i = 0, count = 0;
  
  // 3) we’ll need Z and alpha as R matrices for gfeJump_cpp:
  NumericMatrix Z_N     = wrap(Z);
  NumericMatrix alpha_N = wrap(alpha);
  
  // 4) systematic “local search”:
  while (count != N) {
    // advance i in 1..N
    i = (i % N) + 1;
    
    // build “replace” = gee[- wgroups[i] ]
    std::vector<int> tmp; tmp.reserve(gee.size()-1);
    for (int g : gee) if (g != wgroups[i-1]) tmp.push_back(g);
    IntegerVector replaceR = wrap(tmp);
    
    // call your existing C++ jump‐search:
    arma::vec nb;
    if (method == "gfe") {
      // original jump
      nb = gfeJump_cpp(replaceR, i, wgroups, Z_N, alpha_N);
    } else {
      // WGFE‐specific jump
      nb = wgfeJump_cpp(replaceR, i, wgroups, Z_N, alpha_N);
    }
    
    double obj = nb[0];  // new objective
    if (obj < oldObj) {
      // improved: adopt new grouping, reset counter
      for (int j = 0; j < N; ++j) 
        wgroups[j] = (int) nb[j+1];
      oldObj = obj;
      count  = 0;
    }
    else {
      // no improvement: move on
      ++count;
    }
  }
  
  return wgroups;
}


// [[Rcpp::export]]
IntegerVector localJump_res_cpp(
    List            zList,    // data list
    NumericVector   theta0,   // initial slope vector
    IntegerVector   wgroups,  // initial grouping (1…G)
    NumericVector   sigsg,    // group variances
    double          sig0      // scalar variance
) {
  int N = wgroups.size();
  int G = sigsg.size();
  
  // 1) Precompute residuals Z
  arma::mat Z = computeZ_res_cpp(zList, theta0);
  
  // 2) Create full group index vector gee = 1..G
  IntegerVector gee(G);
  for (int g = 0; g < G; ++g) gee[g] = g + 1;
  
  // 3) Initial objective value
  double old_obj = wgfeObj_bs_cpp(zList, theta0, wgroups, sigsg, sig0);
  
  int count = 0;
  int i     = 0;
  
  // 4) Local‐jump loop: stop after N consecutive non‐improving steps
  while (count != N) {
    // advance index 1..N
    i = (i % N) + 1;
    
    // 4a) Recompute alpha based on current grouping
    arma::mat alpha = computeAlpha_cpp(Z, wgroups);
    
    // 4b) Build candidate set = gee \ { current group of unit i }
    int curg = wgroups[i - 1];
    std::vector<int> cand;
    cand.reserve(G - 1);
    for (int g = 0; g < G; ++g) {
      if (gee[g] != curg) cand.push_back(gee[g]);
    }
    IntegerVector candidates = wrap(cand);
    
    // 4c) Evaluate jump for unit i
    NumericVector neighbor = wgfeJump_bs_cpp(
      candidates,
      i,
      wgroups,
      Z,
      alpha,
      sigsg,
      sig0
    );
    
    double new_obj = neighbor[0];
    
    // 4d) Accept or reject
    if (new_obj < old_obj) {
      // adopt new grouping
      for (int j = 0; j < N; ++j) {
        wgroups[j] = neighbor[j + 1];
      }
      old_obj = new_obj;
      count   = 0;
    } else {
      ++count;
    }
  }
  
  return wgroups;
}


// [[Rcpp::export]]
arma::vec se_cpp(const NumericVector& Y,
                 const NumericMatrix& X,
                 const NumericVector& theta0,
                 const IntegerVector& groupR,
                 const NumericMatrix& alpha0,
                 const NumericVector& sigma0,
                 int t) {
  int N   = groupR.size();
  int TT  = t;
  int p   = X.ncol();
  int G   = sigma0.size();
  
  // Convert inputs
  arma::vec theta = as<arma::vec>(theta0);
  arma::vec Yv     = as<arma::vec>(Y);
  arma::mat Xmat   = as<arma::mat>(X);
  arma::mat alpha  = as<arma::mat>(alpha0);
  arma::vec sigma  = as<arma::vec>(sigma0);
  
  // Map group labels to 0:(G-1)
  arma::ivec grp(N);
  for (int i = 0; i < N; ++i) grp[i] = groupR[i] - 1;
  
  // Precompute covariate matrix per unit: covar[i] is TT×p
  std::vector<arma::mat> covar(N, arma::mat(TT, p));
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < p; ++j) {
      // fill column j of covar[i]
      covar[i].col(j) = Xmat.col(j).rows(i*TT, i*TT + TT - 1);
    }
  }
  
  // Compute x_bar per group: TT×p means
  std::vector<arma::mat> x_bar(G, arma::mat(TT, p, arma::fill::zeros));
  arma::ivec countG(G, arma::fill::zeros);
  for (int i = 0; i < N; ++i) {
    int k = grp[i];
    x_bar[k] += covar[i];
    countG[k]++;
  }
  for (int k = 0; k < G; ++k) {
    if (countG[k] > 0) x_bar[k] /= double(countG[k]);
  }
  
  // Degrees of freedom
  double df = double(N * TT - p - G * TT - G);
  
  // Accumulators for B and V
  arma::mat SB(p,p, arma::fill::zeros);
  arma::mat SV(p,p, arma::fill::zeros);
  
  // Loop over units and time indices
  for (int i = 0; i < N; ++i) {
    int k = grp[i];
    const arma::mat& Xi = covar[i];          // TT×p
    arma::mat Di = Xi - x_bar[k];            // TT×p deviation
    
    // extract residual block
    arma::vec Zi = Yv.subvec(i*TT, i*TT + TT - 1) - Xi * theta;
    
    for (int s = 0; s < TT; ++s) {
      // d_s (1×p) row
      arma::rowvec ds = Di.row(s);
      // for SB
      SB += (ds.t() * ds) / sigma[k];
      
      // for SV: double loop
      double zs = Zi[s] - alpha(s, k);
      for (int r = 0; r < TT; ++r) {
        double zr = Zi[r] - alpha(r, k);
        arma::rowvec dr = Di.row(r);
        SV += (zs * zr) * (ds.t() * dr) / (sigma[k] * sigma[k]);
      }
    }
  }
  
  arma::mat Binv = arma::inv(SB / df);
  arma::mat V    = SV  / df;
  
  arma::mat M    = Binv * V * Binv;
  arma::vec se   = arma::sqrt(M.diag() / double(N * TT));
  return se;
}


// [[Rcpp::export]]
Rcpp::NumericMatrix seHet_cpp(const Rcpp::List&           zList,
                              const Rcpp::NumericMatrix&   theta,
                              const Rcpp::IntegerVector&   groupR,
                              const Rcpp::NumericMatrix&   alpha0) {
  int N = zList.size();
  if (groupR.size() != N) {
    Rcpp::stop("groupR must have length N = zList size");
  }
  // convert inputs
  arma::mat th  = Rcpp::as<arma::mat>(theta);    // p×G
  arma::mat alpha = Rcpp::as<arma::mat>(alpha0); // TT×G
  
  int p = th.n_rows;
  int G = th.n_cols;
  int TT = alpha.n_rows;
  if (alpha.n_cols != G) {
    Rcpp::stop("theta and alpha must share same number of columns/groups");
  }
  
  // map to 0-based groups and count
  arma::ivec grp(N);
  arma::ivec countG(G, arma::fill::zeros);
  for (int i = 0; i < N; ++i) {
    int g = groupR[i] - 1;
    if (g < 0 || g >= G) Rcpp::stop("invalid group label at position %d", i+1);
    grp[i] = g;
    countG[g]++;
  }
  
  // precompute TT×p covariate matrices for each unit
  // Precompute covariate matrix per unit: covar[i] is TT×p
  std::vector<arma::mat> covar(N);         // size N
  for (int i = 0; i < N; ++i) {
    arma::mat zi = Rcpp::as<arma::mat>(zList[i]);
    covar[i]    = zi.cols(1, p);           // directly assign into slot i
  }
  
  // compute time‐specific X means for each group
  std::vector<arma::mat> xbar(G, arma::mat(TT, p, arma::fill::zeros));
  for (int i = 0; i < N; ++i) {
    xbar[grp[i]] += covar[i];
  }
  for (int g = 0; g < G; ++g) {
    if (countG[g] > 0) xbar[g] /= double(countG[g]);
  }
  
  // output matrix p×G
  Rcpp::NumericMatrix out(p, G);
  
  // loop over groups
  for (int g = 0; g < G; ++g) {
    int ng = countG[g];
    if (ng == 0) {
      for (int j = 0; j < p; ++j) out(j, g) = NA_REAL;
      continue;
    }
    // degrees of freedom, matching original: n_g*TT - p - TT - 1
    double df = double(ng*TT - p - TT - 1);
    if (df <= 0) Rcpp::stop("non-positive degrees of freedom for group %d", g+1);
    
    arma::mat SB(p, p, arma::fill::zeros);
    arma::mat SV(p, p, arma::fill::zeros);
    
    arma::colvec betag = th.col(g);
    
    // loop units in group
    for (int i = 0; i < N; ++i) {
      if (grp[i] != g) continue;
      arma::mat& Xi = covar[i];           // TT×p
      arma::mat Di = Xi - xbar[g];        // TT×p deviations
      
      // compute residual vector e_i = Y - X*beta_g
      arma::mat zi = Rcpp::as<arma::mat>(zList[i]);
      arma::vec Yi = zi.col(0);
      arma::vec ei = Yi - Xi * betag;
      // subtract group-time centroid
      arma::vec eg = ei - alpha.col(g);
      
      // accumulate SB, SV
      for (int s = 0; s < TT; ++s) {
        arma::rowvec ds = Di.row(s);
        SB += (ds.t() * ds);
        double z_s = eg[s];
        for (int r = 0; r < TT; ++r) {
          double z_r = eg[r];
          arma::rowvec dr = Di.row(r);
          SV += (z_s * z_r) * (ds.t() * dr);
        }
      }
    }
    
    arma::mat Binv = arma::inv(SB / df);
    arma::mat V    = SV  / df;
    arma::mat M    = Binv * V * Binv;
    arma::vec se   = arma::sqrt(M.diag() / double(ng*TT));
    
    for (int j = 0; j < p; ++j) out(j, g) = se[j];
  }
  
  return out;
}

