// functions_unbalanced.cpp
// Unbalanced-panel versions of core routines
// Assumes input zList elements may have NaNs marking missing d_it, and downstream routines skip NaNs once

#include <RcppArmadillo.h>
using namespace Rcpp;

// --- computeZ_unbalanced_cpp ---
// [[Rcpp::export]]
arma::mat computeZ_unbalanced_cpp(const List& zList,
                                  const NumericMatrix& theta,
                                  const IntegerVector& groups) {
  // Number of units
  int N = zList.size();
  // Determine T and p from the first element of zList
  arma::mat first = as<arma::mat>(zList[0]);
  int T = first.n_rows;
  int p = first.n_cols - 1;
  
  // Convert theta to an Armadillo matrix
  arma::mat th = as<arma::mat>(theta);
  bool heterogeneous = (th.n_rows > 1 && th.n_cols > 1);
  
  // Pre‐allocate output (T × N), each column will be overwritten
  arma::mat Zall(T, N);
  
  // Convert each element of zList to arma::mat once
  std::vector<arma::mat> Zvec;
  Zvec.reserve(N);
  for (int i = 0; i < N; ++i) {
    Zvec.emplace_back(as<arma::mat>(zList[i]));
  }
  
  if (!heterogeneous) {
    // Homogeneous coefficients: extract b once
    arma::colvec b;
    if (th.n_rows == 1) {
      // theta is 1×p
      b = th.row(0).t();        // p×1
    } else {
      // theta is p×1
      b = th.col(0);            // p×1
    }
    
    // For each unit i, compute Zi = Y_i – X_i * b
    for (int i = 0; i < N; ++i) {
      const arma::mat& zi = Zvec[i];          // T×(p+1), col 0 = Y, cols 1..p = X
      // Submatrix X_i is zi.cols(1, p)
      // Column vector Y_i is zi.col(0)
      Zall.col(i) = zi.col(0) - zi.cols(1, p) * b;
    }
    
  } else {
    // Heterogeneous coefficients: theta is p×G
    for (int i = 0; i < N; ++i) {
      const arma::mat& zi = Zvec[i];          
      int g = groups[i] - 1;                  // zero‐based group index
      arma::colvec bi = th.col(g);            // p×1 slope vector for group g
      Zall.col(i) = zi.col(0) - zi.cols(1, p) * bi;
    }
  }
  
  return Zall;
}


// [[Rcpp::export]]
List computeAlpha_unbalanced_cpp(const arma::mat& Zall,
                                 const IntegerVector groups) {
  int T = Zall.n_rows;
  int N = Zall.n_cols;
  int G = max(groups);
  
  // Prepare output (T×G), filled with NA
  arma::mat alpha(T, G);
  alpha.fill(NA_REAL);
  
  bool allValid = true;
  
  // 1) Collect column-indices for each group (0-based)
  std::vector<std::vector<arma::uword>> cols_in_group(G);
  for (int i = 0; i < N; ++i) {
    int g = groups[i] - 1;  // 1-based → 0-based
    if (g >= 0 && g < G) {
      cols_in_group[g].push_back((arma::uword)i);
    }
  }
  
  // 2) For each group, extract its submatrix and compute row-means over finite entries
  for (int g = 0; g < G; ++g) {
    const auto& vec_idx = cols_in_group[g];
    if (vec_idx.empty()) {
      allValid = false;
      continue;
    }
    
    // Build arma::uvec of column indices
    arma::uvec cols(vec_idx.size());
    for (std::size_t j = 0; j < vec_idx.size(); ++j) {
      cols[j] = vec_idx[j];
    }
    
    // Extract T×n_g submatrix Zg
    arma::mat Zg = Zall.cols(cols);
    
    // 2a) Find all non-finite positions
    arma::uvec idx_nonfin = find_nonfinite(Zg);
    
    // 2b) Copy Zg to Zg2 and zero-out non-finite entries
    arma::mat Zg2 = Zg;
    if (!idx_nonfin.is_empty()) {
      Zg2.elem(idx_nonfin).zeros();
    }
    
    // 2c) Build a T×n_g matrix of ones, then zero-out same non-finite positions
    arma::mat ones_mat = arma::ones<arma::mat>(Zg.n_rows, Zg.n_cols);
    if (!idx_nonfin.is_empty()) {
      ones_mat.elem(idx_nonfin).zeros();
    }
    
    // 2d) Row-wise sum of finite values
    arma::vec sumVec = sum(Zg2, /*dim=*/1);
    // 2e) Row-wise count of finite entries
    arma::vec countVec = sum(ones_mat, /*dim=*/1);
    
    // 2f) Find rows with at least one finite observation
    arma::uvec has_obs = find(countVec > 0);
    if ((int)has_obs.n_elem < T) {
      allValid = false;
    }
    
    // 2g) Assign means: α(t,g) = sumVec[t]/countVec[t] for t in has_obs
    for (arma::uword idx = 0; idx < has_obs.n_elem; ++idx) {
      arma::uword t = has_obs(idx);
      alpha(t, (arma::uword)g) = sumVec(t) / countVec(t);
    }
  }
  
  return List::create(
    Named("alpha") = alpha,
    Named("valid") = allValid
  );
}


// [[Rcpp::export]]
arma::vec computeSigma_unbalanced_cpp(const arma::mat& Zall,
                                      const arma::mat& alpha,
                                      const IntegerVector& groups) {
  int T = Zall.n_rows;
  int N = Zall.n_cols;
  int G = alpha.n_cols;
  
  arma::vec Sigma(G, arma::fill::zeros);
  
  // 1) Build column‐indices for each group (0‐based)
  std::vector<std::vector<arma::uword>> cols_in_group(G);
  for (int i = 0; i < N; ++i) {
    int g = groups[i] - 1;  // convert 1‐based → 0‐based
    if (g >= 0 && g < G) {
      cols_in_group[g].push_back((arma::uword)i);
    }
  }
  
  // 2) For each group, compute Σ_g = sqrt( (1/count_g) * Σ_{i,t} (Z_{t,i} – α_{t,g})^2 )
  for (int g = 0; g < G; ++g) {
    const auto& vec_idx = cols_in_group[g];
    if (vec_idx.empty()) {
      // No units in group g ⇒ return NaN
      Sigma[g] = arma::datum::nan;
      continue;
    }
    
    // Build arma::uvec of column indices for group g
    arma::uvec cols(vec_idx.size());
    for (std::size_t j = 0; j < vec_idx.size(); ++j) {
      cols[j] = vec_idx[j];
    }
    
    // Extract T × n_g submatrix Zg
    arma::mat Zg = Zall.cols(cols);
    
    // Compute residuals Rg = Zg – α_col replicated across columns
    arma::vec a_col = alpha.col(g);      // T×1
    arma::mat Rg = Zg;                   
    Rg.each_col() -= a_col;              // subtract α(t,g) from each column
    
    // Identify positions where Zg was non‐finite
    arma::uvec idx_nonfin = find_nonfinite(Zg);
    
    // Zero‐out those positions in Rg (so they don't contribute)
    if (!idx_nonfin.is_empty()) {
      Rg.elem(idx_nonfin).zeros();
    }
    
    // Sum of squared deviations over all remaining (finite) entries
    double sumSq = accu(Rg % Rg);
    
    // Count of finite entries = total elements minus number of non‐finite positions
    arma::uword total_entries = Zg.n_elem;            // = T * n_g
    arma::uword n_nonfin    = idx_nonfin.n_elem;
    arma::uword count       = total_entries - n_nonfin;
    
    // Compute group‐specific sigma
    Sigma[g] = (count > 0)
      ? std::sqrt(sumSq / static_cast<double>(count))
        : arma::datum::nan;
  }
  
  return Sigma;
}




// --- gfeObj_unbalanced_cpp ---

// Summation over all finite Zall entries and corresponding alpha
// Skips residuals which are NaN; no per-cell valid matrix required
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
double gfeObj_unbalanced_cpp(const arma::mat& Zall,
                             const IntegerVector& groups) {
  int T = Zall.n_rows;
  int N = Zall.n_cols;
  int G = max(groups);
  
  // 1) Build column‐indices for each group (0‐based)
  std::vector<std::vector<arma::uword>> cols_in_group(G);
  for (int i = 0; i < N; ++i) {
    int g = groups[i] - 1;  // convert 1‐based → 0‐based
    if (g >= 0 && g < G) {
      cols_in_group[g].push_back((arma::uword)i);
    }
  }
  
  double totalSSE = 0.0;
  
  // 2) For each group, compute α_{t,g} and accumulate SSE
  for (int g = 0; g < G; ++g) {
    const auto& vec_idx = cols_in_group[g];
    if (vec_idx.empty()) {
      // No units in this group → skip
      continue;
    }
    
    // Build arma::uvec of column indices for group g
    arma::uvec cols(vec_idx.size());
    for (std::size_t j = 0; j < vec_idx.size(); ++j) {
      cols[j] = vec_idx[j];
    }
    
    // Extract T × n_g submatrix Zg
    arma::mat Zg = Zall.cols(cols);
    
    // 2a) Find non‐finite entries in Zg
    arma::uvec idx_nonfin = find_nonfinite(Zg);
    
    // 2b) Zero‐out non‐finite entries in a copy for sums
    arma::mat Zg2 = Zg;
    if (!idx_nonfin.is_empty()) {
      Zg2.elem(idx_nonfin).zeros();
    }
    
    // 2c) Build a T×n_g matrix of ones, then zero‐out same non‐finite positions
    arma::mat ones_mat = arma::ones<arma::mat>(Zg.n_rows, Zg.n_cols);
    if (!idx_nonfin.is_empty()) {
      ones_mat.elem(idx_nonfin).zeros();
    }
    
    // 2d) Compute row‐wise sum of finite values and count of finite entries
    arma::vec sumVec   = sum(Zg2, /*dim=*/1);      // T×1
    arma::vec countVec = sum(ones_mat, /*dim=*/1); // T×1
    
    // 2e) Build alpha_col for this group (T×1), set NaN where count==0
    arma::vec alpha_col(T);
    for (int t = 0; t < T; ++t) {
      if (countVec[t] > 0.0) {
        alpha_col[t] = sumVec[t] / countVec[t];
      } else {
        alpha_col[t] = arma::datum::nan;
      }
    }
    
    // 2f) Compute residuals Rg = Zg − α_col (broadcast across columns)
    arma::mat Rg = Zg;
    Rg.each_col() -= alpha_col;
    
    // 2g) Zero‐out residuals at non‐finite positions so they don’t contribute
    if (!idx_nonfin.is_empty()) {
      Rg.elem(idx_nonfin).zeros();
    }
    
    // 2h) Accumulate sum of squared residuals
    totalSSE += accu(Rg % Rg);
  }
  
  return totalSSE;
}


// --- wgfeObj_unbalanced_cpp ---

// Weighted GFE objective for unbalanced panels: skips NaN residuals
// Follows the same structure as wgfeObj_cpp, leveraging computeAlpha and computeSigma
// [[Rcpp::export]]
double wgfeObj_unbalanced_cpp(const arma::mat& Zall,
                              const IntegerVector& groups) {
  int T = Zall.n_rows;
  int N = Zall.n_cols;
  
  if ((int)groups.size() != N) {
    Rcpp::stop("`groups` must have length N = Zall.n_cols");
  }
  
  // 1) Determine G = max(groups)
  int G = 0;
  for (int i = 0; i < N; ++i) {
    G = std::max(G, groups[i]);
  }
  
  // 2) Build column‐indices for each group (0‐based) and check validity
  std::vector<std::vector<arma::uword>> cols_in_group(G);
  for (int i = 0; i < N; ++i) {
    int g = groups[i] - 1;  // convert 1‐based → 0‐based
    if (g < 0 || g >= G) {
      Rcpp::stop("invalid group label at position %d", i + 1);
    }
    cols_in_group[g].push_back((arma::uword)i);
  }
  
  // 3) Compute α (T×G) via the optimized computeAlpha_unbalanced_cpp
  List alphaList = computeAlpha_unbalanced_cpp(Zall, groups);
  arma::mat alpha = as<arma::mat>(alphaList["alpha"]);
  
  double total = 0.0;
  
  // 4) For each group g, compute its sigma and weight by n_g/N
  for (int g = 0; g < G; ++g) {
    const auto& vec_idx = cols_in_group[g];
    arma::uword n_g = vec_idx.size();
    if (n_g == 0) {
      continue;  // no units in group → effectively skip
    }
    
    // 4a) Build an arma::uvec of column indices for group g
    arma::uvec cols(n_g);
    for (arma::uword j = 0; j < n_g; ++j) {
      cols[j] = vec_idx[j];
    }
    
    // 4b) Extract T×n_g submatrix Zg
    arma::mat Zg = Zall.cols(cols);
    
    // 4c) Compute residuals Rg = Zg – α_col (broadcast across columns)
    arma::vec a_col = alpha.col(g);       // T×1
    arma::mat Rg = Zg;
    Rg.each_col() -= a_col;               // subtract α(t,g) from each column
    
    // 4d) Identify non‐finite positions in Zg (skip those)
    arma::uvec idx_nonfin = find_nonfinite(Zg);
    
    // 4e) Square and sum only finite residuals
    if (!idx_nonfin.is_empty()) {
      Rg.elem(idx_nonfin).zeros();
    }
    double sumSq = accu(Rg % Rg);
    
    // 4f) Count of finite entries = T * n_g – (# nonfinite positions)
    arma::uword total_entries = Zg.n_elem;       // = T * n_g
    arma::uword n_nonfin    = idx_nonfin.n_elem;
    arma::uword count       = total_entries - n_nonfin;
    
    double sigma_g = (count > 0)
      ? std::sqrt(sumSq / static_cast<double>(count))
        : arma::datum::nan;
    
    // 4g) Weight by (n_g / N)
    total += sigma_g * (static_cast<double>(n_g) / static_cast<double>(N));
  }
  
  return total;
}


// --- slopeGradGFE_unbalanced_cpp ---
// Gradient step for unbalanced panels: OLS on within-group deviations, skipping NaNs

// [[Rcpp::export]]
arma::vec slopeGradGFE_unbalanced_cpp(List z, IntegerVector group) {
  int N = z.size();
  if (N == 0) return arma::vec();
  
  // 1) infer dimensions from first element
  arma::mat Z0 = as<arma::mat>(z[0]);
  int T = Z0.n_rows;
  int K = Z0.n_cols;
  int p = K - 1;
  int G = max(group);
  
  // 2) load data once into a vector of arma::mat
  std::vector<arma::mat> Zdata(N);
  for (int i = 0; i < N; ++i) {
    Zdata[i] = as<arma::mat>(z[i]);
  }
  
  // 3) compute group‐time sums and counts using arma::cube + arma::mat
  arma::cube gSum(T, K, G, arma::fill::zeros);
  arma::mat  gCount(T, G, arma::fill::zeros);
  
  for (int i = 0; i < N; ++i) {
    int gi = group[i] - 1;              
    const arma::mat &Zi = Zdata[i];
    
    for (int t = 0; t < T; ++t) {
      arma::rowvec row = Zi.row(t);
      if (row.has_nan()) continue;   // skip if any NaN in this row
      
      gSum.slice(gi).row(t) += row;
      gCount(t, gi) += 1.0;
    }
  }
  
  // 4) compute group‐time means gAve as a cube of size T×K×G
  arma::cube gAve(T, K, G, arma::fill::zeros);
  for (int g = 0; g < G; ++g) {
    for (int t = 0; t < T; ++t) {
      double cnt = gCount(t, g);
      if (cnt > 0.0) {
        gAve.slice(g).row(t) = gSum.slice(g).row(t) / cnt;
      }
    }
  }
  
  // 5) accumulate XtX and Xty via manual loops 
  arma::mat XtX(p, p, arma::fill::zeros);
  arma::vec Xty(p, arma::fill::zeros);
  
  for (int i = 0; i < N; ++i) {
    int gi = group[i] - 1;
    const arma::mat &Zi = Zdata[i];
    
    for (int t = 0; t < T; ++t) {
      arma::rowvec row = Zi.row(t);
      if (row.has_nan()) continue;   // skip if any NaN
      
      // within‐group deviation
      arma::rowvec rowDev = row - gAve.slice(gi).row(t);
      double y = rowDev[0];
      
      for (int a = 0; a < p; ++a) {
        double xi_a = rowDev[a + 1];
        Xty[a] += xi_a * y;
        for (int b = 0; b < p; ++b) {
          XtX(a, b) += xi_a * rowDev[b + 1];
        }
      }
    }
  }
  
  // 6) solve or return NaNs if singular/zero
  if (XtX.is_zero() || arma::rank(XtX) < p) {
    return arma::vec(p).fill(arma::datum::nan);
  }
  
  return arma::solve(XtX, Xty);
}


// [[Rcpp::export]]
List computeXXY_demeaned_unbalanced_cpp(List zList,
                                        const IntegerVector& wgroups) {
  int N = zList.size();
  if (N == 0) {
    return List::create(
      Named("XX_demeaned") = List(),
      Named("y_demeaned")  = List()
    );
  }
  
  // 1) infer global dimensions from first element
  arma::mat Z0 = as<arma::mat>(zList[0]);  // T x (p+1)
  int T = Z0.n_rows;
  int p_plus1 = Z0.n_cols;  // p + 1
  int p = p_plus1 - 1;
  
  // 2) map unique group labels to 0:(G-1)
  IntegerVector uniq = sort_unique(wgroups);
  int G = uniq.size();
  std::unordered_map<int,int> label2k;
  label2k.reserve(G);
  for (int k = 0; k < G; ++k) {
    label2k[ uniq[k] ] = k;
  }
  
  // 3) preload all Z into a vector<arma::mat> for fast access
  std::vector<arma::mat> Zdata(N);
  std::vector<int>       grp_idx(N);
  for (int i = 0; i < N; ++i) {
    Zdata[i] = as<arma::mat>(zList[i]);  // T x (p+1)
    grp_idx[i] = label2k[wgroups[i]];   // zero-based group index
  }
  
  // 4) compute group-time sums and counts using arma::cube + arma::mat
  arma::cube sumZT(T, p_plus1, G, arma::fill::zeros);
  arma::mat  cntZT(T, G, arma::fill::zeros);
  
  for (int i = 0; i < N; ++i) {
    int k = grp_idx[i];
    const arma::mat& Zi = Zdata[i];
    for (int t = 0; t < T; ++t) {
      arma::rowvec row = Zi.row(t);
      if (row.has_nan()) continue; 
      sumZT.slice(k).row(t) += row;
      cntZT(t, k) += 1.0;
    }
  }
  
  // 5) compute group-time means gAve as an arma::cube of size T x (p+1) x G
  arma::cube gAve(T, p_plus1, G, arma::fill::zeros);
  for (int k = 0; k < G; ++k) {
    for (int t = 0; t < T; ++t) {
      double cnt = cntZT(t, k);
      if (cnt > 0.0) {
        gAve.slice(k).row(t) = sumZT.slice(k).row(t) / cnt;
      }
      // else leave as zeros
    }
  }
  
  // 6) build XX and y lists
  List XX_list(N), y_list(N);
  for (int i = 0; i < N; ++i) {
    int k = grp_idx[i];
    const arma::mat& Zi = Zdata[i];
    
    arma::mat XX(p, p, arma::fill::zeros);
    arma::vec y(p,    arma::fill::zeros);
    
    for (int t = 0; t < T; ++t) {
      if (cntZT(t, k) == 0.0) continue;
      
      arma::rowvec row = Zi.row(t);
      if (row.has_nan()) continue;
      
      // demean: rowDev = row - gAve[k][t,]
      arma::rowvec rowDev = row - gAve.slice(k).row(t);
      double yval = rowDev[0];
      
      // accumulate XX and y without creating temporaries
      for (int a = 0; a < p; ++a) {
        double x_a = rowDev[a + 1];
        y[a] += x_a * yval;
        for (int b = 0; b < p; ++b) {
          XX(a, b) += x_a * rowDev[b + 1];
        }
      }
    }
    
    XX_list[i] = XX;
    y_list[i]  = y;
  }
  
  return List::create(
    Named("XX_demeaned") = XX_list,
    Named("y_demeaned")  = y_list
  );
}


// [[Rcpp::export]]
arma::vec wc_fp_unbalanced_cpp(List                zList,
                               const arma::vec&    b,
                               const IntegerVector& wgroups,
                               List                XX_list,
                               List                y_list,
                               const IntegerVector& gee) {
  int N = zList.size();
  if (N == 0) return arma::vec();
  
  int p = b.n_elem;
  int G = gee.size();
  
  // 1) Build theta matrix for computeZ_unbalanced_cpp (p x 1)
  NumericMatrix thetaMat(p, 1);
  for (int j = 0; j < p; ++j) {
    thetaMat(j, 0) = b[j];
  }
  arma::mat Z = computeZ_unbalanced_cpp(zList, thetaMat, wgroups);
  
  // 2) Build mapping from group label -> index k in [0, G)
  std::unordered_map<int,int> label2k;
  label2k.reserve(G);
  for (int k = 0; k < G; ++k) {
    label2k[ gee[k] ] = k;
  }
  
  // 3) Compute alpha and SigmaG
  List alphaList = computeAlpha_unbalanced_cpp(Z, wgroups);
  arma::mat alpha = as<arma::mat>(alphaList["alpha"]);
  arma::vec SigmaG = computeSigma_unbalanced_cpp(Z, alpha, wgroups);
  
  // 4) Pre‐cast XX_list and y_list into C++ vectors of arma types
  std::vector<arma::mat> XXmats(N);
  std::vector<arma::vec> ymats(N);
  for (int i = 0; i < N; ++i) {
    XXmats[i] = as<arma::mat>(XX_list[i]);
    ymats[i]  = as<arma::vec>(y_list[i]);
  }
  
  // 5) Build unit→k mapping (or -1 if unit's group not in gee)
  std::vector<int> unit2k(N, -1);
  for (int i = 0; i < N; ++i) {
    auto it = label2k.find(wgroups[i]);
    if (it != label2k.end()) {
      unit2k[i] = it->second;
    }
  }
  
  // 6) Accumulate sumXX[k] and sumY[k] over units in group k
  std::vector<arma::mat> sumXX(G, arma::mat(p, p, arma::fill::zeros));
  std::vector<arma::vec> sumY(G,  arma::vec(p,    arma::fill::zeros));
  
  for (int i = 0; i < N; ++i) {
    int k = unit2k[i];
    if (k < 0) continue;
    sumXX[k] += XXmats[i];
    sumY[k]  += ymats[i];
  }
  
  // 7) Assemble weighted normal equations A θ = B
  arma::mat A(p, p, arma::fill::zeros);
  arma::vec B(p,    arma::fill::zeros);
  
  for (int k = 0; k < G; ++k) {
    double sigma = SigmaG[k];
    if (!(sigma > 0.0)) continue;
    
    A += (sumXX[k] / sigma);
    B += (sumY[k]  / sigma);
  }
  
  // 8) Solve for θ
  return arma::solve(A, B);
}



// [[Rcpp::export]]
arma::vec wc_fp_loop_unbalanced_cpp(List                zList,
                                         const IntegerVector& wgroups,
                                         List                XX_list,
                                         List                y_list,
                                         const IntegerVector& gee,
                                         double              tol      = 1e-13,
                                         int                 max_iter = 1000) {
  int N = zList.size();
  if (N == 0) {
    return arma::vec();
  }
  
  // 1) Infer p from first y_list element
  arma::vec y0 = as<arma::vec>(y_list[0]);
  int p = y0.n_elem;
  
  // 2) Pre‐cast zList to vector<arma::mat>
  std::vector<arma::mat> Zdata(N);
  for (int i = 0; i < N; ++i) {
    Zdata[i] = as<arma::mat>(zList[i]);  // T × (p+1) inside computeZ_unbalanced_cpp, but we need raw zList later
  }
  
  // 3) Pre‐cast XX_list and y_list to C++ vectors
  //    Also build a mapping unit → group‐index k (based on gee)
  int G = gee.size();
  std::unordered_map<int,int> label2k;
  label2k.reserve(G);
  for (int k = 0; k < G; ++k) {
    label2k[ gee[k] ] = k;
  }
  
  std::vector<arma::mat> XXmats(N);
  std::vector<arma::vec> ymats(N);
  std::vector<int>       unit2k(N, -1);
  
  for (int i = 0; i < N; ++i) {
    XXmats[i] = as<arma::mat>(XX_list[i]);
    ymats[i]  = as<arma::vec>(y_list[i]);
    
    auto it = label2k.find(wgroups[i]);
    if (it != label2k.end()) {
      unit2k[i] = it->second;  // zero‐based group index
    }
  }
  
  // 4) Precompute sumXX[k] and sumY[k] for each k = 0..G-1
  std::vector<arma::mat> sumXX(G, arma::mat(p, p, arma::fill::zeros));
  std::vector<arma::vec> sumY(G,  arma::vec(p,    arma::fill::zeros));
  
  for (int i = 0; i < N; ++i) {
    int k = unit2k[i];
    if (k < 0) continue;
    sumXX[k] += XXmats[i];
    sumY[k]  += ymats[i];
  }
  
  // 5) Initialize theta0 = 0 and prepare thetaMat container (p × 1)
  arma::vec theta0 = arma::zeros<arma::vec>(p);
  arma::vec theta  = arma::zeros<arma::vec>(p);
  NumericMatrix thetaMat(p, 1);
  
  // 6) Fixed‐point iteration
  double diff = tol + 1.0;
  int iter = 0;
  
  while (diff > tol && iter < max_iter) {
    // 6a) Build thetaMat from theta0
    for (int j = 0; j < p; ++j) {
      thetaMat(j, 0) = theta0[j];
    }
    
    // 6b) Compute Z via computeZ_unbalanced_cpp
    arma::mat Z = computeZ_unbalanced_cpp(zList, thetaMat, wgroups);
    
    // 6c) Compute alpha and SigmaG
    List alphaList = computeAlpha_unbalanced_cpp(Z, wgroups);
    arma::mat alpha = as<arma::mat>(alphaList["alpha"]);
    arma::vec SigmaG = computeSigma_unbalanced_cpp(Z, alpha, wgroups);
    
    // 6d) Assemble A and B using precomputed sumXX and sumY
    arma::mat A(p, p, arma::fill::zeros);
    arma::vec B(p,    arma::fill::zeros);
    
    for (int k = 0; k < G; ++k) {
      double sigma = SigmaG[k];
      if (!(sigma > 0.0)) continue;  // skip zero or NaN
      
      A += (sumXX[k] / sigma);
      B += (sumY[k]  / sigma);
    }
    
    // 6e) Solve for new theta
    theta = arma::solve(A, B);
    
    // 6f) Check convergence
    diff = arma::norm(theta - theta0, 2);
    theta0 = theta;
    ++iter;
  }
  
  return theta0;
}


// [[Rcpp::export]]
arma::mat calcGroupSlopes_unbalanced_cpp(const List&            zList,
                                         const IntegerVector&   groups) {
  int N = zList.size();
  if ((int)groups.size() != N) {
    Rcpp::stop("`groups` must have length N = zList.size()");
  }
  if (N == 0) {
    return arma::mat();
  }
  
  // Determine G = max group label
  int G = max(groups);
  
  // Pre‐cast all zList entries to arma::mat
  std::vector<arma::mat> Zdata(N);
  for (int i = 0; i < N; ++i) {
    Zdata[i] = as<arma::mat>(zList[i]);  // each is T×(p+1)
  }
  
  // Infer T and p from first element
  int T = Zdata[0].n_rows;
  int K = Zdata[0].n_cols; // = p + 1
  int p = K - 1;
  
  // Build list of members for each group (1‐based labels → zero‐based index)
  std::vector<std::vector<int>> group_members(G);
  group_members.reserve(G);
  for (int i = 0; i < N; ++i) {
    int g = groups[i];
    if (g >= 1 && g <= G) {
      group_members[g - 1].push_back(i);
    }
  }
  
  // Prepare output: p × G
  arma::mat betas(p, G, arma::fill::zeros);
  
  // For each group g = 1..G
  for (int g = 1; g <= G; ++g) {
    auto &members = group_members[g - 1];
    int n_g = members.size();
    if (n_g == 0) {
      // No units in this group → fill NaNs
      betas.col(g - 1).fill(arma::datum::nan);
      continue;
    }
    
    // 1) Allocate accumulators for time‐t means
    arma::vec mean_y(T, arma::fill::zeros);
    arma::mat mean_x(T, p, arma::fill::zeros);
    arma::uvec count_joint(T, arma::fill::zeros);
    
    // 2) First pass: sum up y and X for complete rows
    for (int idx_i : members) {
      const arma::mat &Zi = Zdata[idx_i];  // T × (p+1)
      for (int t = 0; t < T; ++t) {
        arma::rowvec row = Zi.row(t);
        if (!row.is_finite()) continue;
        
        mean_y[t] += row[0];
        for (int j = 0; j < p; ++j) {
          mean_x(t, j) += row[j + 1];
        }
        count_joint[t] += 1;
      }
    }
    
    // 3) Compute per‐time means (only where count_joint[t] > 0)
    for (int t = 0; t < T; ++t) {
      if (count_joint[t] > 0) {
        double inv_cnt = 1.0 / count_joint[t];
        mean_y[t] *= inv_cnt;
        for (int j = 0; j < p; ++j) {
          mean_x(t, j) *= inv_cnt;
        }
      }
      // If count_joint[t] == 0, leave mean_y[t] = 0 and mean_x(t,·) = 0,
      // since we will skip those times in the second pass.
    }
    
    // 4) Second pass: accumulate XtX and Xty using demeaned values
    arma::mat XtX(p, p, arma::fill::zeros);
    arma::vec Xty(p,    arma::fill::zeros);
    
    for (int idx_i : members) {
      const arma::mat &Zi = Zdata[idx_i];  // T × (p+1)
      for (int t = 0; t < T; ++t) {
        if (count_joint[t] == 0) continue;
        
        arma::rowvec row = Zi.row(t);
        if (!row.is_finite()) continue;
        
        double y_dev = row[0] - mean_y[t];
        for (int a = 0; a < p; ++a) {
          double x_dev_a = row[a + 1] - mean_x(t, a);
          Xty[a] += x_dev_a * y_dev;
          for (int b = 0; b < p; ++b) {
            XtX(a, b) += x_dev_a * (row[b + 1] - mean_x(t, b));
          }
        }
      }
    }
    
    // 5) Solve OLS: β = (X˜' X˜)^{-1} (X˜' y˜)
    if (XtX.is_zero() || arma::rank(XtX) < p) {
      betas.col(g - 1).fill(arma::datum::nan);
    } else {
      betas.col(g - 1) = arma::solve(XtX, Xty);
    }
  }
  
  return betas;
}




// [[Rcpp::export]]
arma::vec gfeJump_unbalanced_cpp(const IntegerVector& replaceR,
                                 int                 i,
                                 const IntegerVector& grR,
                                 const NumericMatrix& Z_N) {
  // 1) Convert inputs
  int N   = grR.size();
  int R   = replaceR.size();
  int idx = i - 1;  // convert to 0-based
  
  arma::uvec gr0   = as<arma::uvec>(grR) - 1;       // current groups, zero-based
  arma::uvec rep0  = as<arma::uvec>(replaceR) - 1;  // candidate groups, zero-based
  arma::mat  Z     = as<arma::mat>(Z_N);            // T × N
  
  int T = Z.n_rows;
  int G = (int)max(gr0) + 1;   // number of groups
  
  // 2) Build per-group summary: S(t,g), C(t,g), Q(t,g) for t=0..T-1, g=0..G-1
  arma::mat  S(T, G, fill::zeros);   // sum of Z_{t,i} for group g
  arma::imat C(T, G, fill::zeros);   // count of finite Z_{t,i} for group g
  arma::mat  Q(T, G, fill::zeros);   // sum of Z_{t,i}^2 for group g
  
  for (int j = 0; j < N; ++j) {
    int gj = gr0[j];
    for (int t = 0; t < T; ++t) {
      double z = Z(t, j);
      if (!std::isfinite(z)) continue;
      S(t, gj) += z;
      C(t, gj) += 1;
      Q(t, gj) += z * z;
    }
  }
  
  // 3) Compute initial SSE per group and total
  arma::vec SSE(G, fill::zeros);
  for (int g = 0; g < G; ++g) {
    double sse_g = 0.0;
    for (int t = 0; t < T; ++t) {
      int cnt = C(t, g);
      if (cnt > 0) {
        double sum_z  = S(t, g);
        double sum_z2 = Q(t, g);
        sse_g += sum_z2 - (sum_z * sum_z) / double(cnt);
      }
    }
    SSE[g] = sse_g;
  }
  double total0 = arma::sum(SSE);
  
  // 4) Preallocate objective values
  arma::vec obj(R, fill::zeros);
  
  int g_old = gr0[idx];
  
  // 5) For each candidate replacement r
  for (int r = 0; r < R; ++r) {
    int g_new = rep0[r];
    // If candidate is same as old group, objective stays the same
    if (g_new == g_old) {
      obj[r] = total0;
      continue;
    }
    
    // Copy original group‐specific SSE for old and new
    double sse_old0 = SSE[g_old];
    double sse_old1 = SSE[g_new];
    double delta0 = 0.0;
    double delta1 = 0.0;
    
    // For each time t where unit idx has a finite observation
    for (int t = 0; t < T; ++t) {
      double z = Z(t, idx);
      if (!std::isfinite(z)) continue;
      
      // ----- Update for old group (g_old) -----
      int    C0_old   = C(t, g_old);
      double S0_old   = S(t, g_old);
      double Q0_old   = Q(t, g_old);
      int    C0_new   = C0_old - 1;
      double S0_new   = S0_old - z;
      double Q0_new   = Q0_old - z * z;
      
      // Compute old SSE contribution at (t, g_old)
      double sse0_t_old = Q0_old - (S0_old * S0_old) / double(C0_old);
      // Compute new SSE contribution at (t, g_old)
      double sse0_t_new = 0.0;
      if (C0_new > 0) {
        sse0_t_new = Q0_new - (S0_new * S0_new) / double(C0_new);
      }
      delta0 += (sse0_t_new - sse0_t_old);
      
      // ----- Update for new group (g_new) -----
      int    C1_old   = C(t, g_new);
      double S1_old   = S(t, g_new);
      double Q1_old   = Q(t, g_new);
      int    C1_new   = C1_old + 1;
      double S1_new   = S1_old + z;
      double Q1_new   = Q1_old + z * z;
      
      // Compute old SSE contribution at (t, g_new)
      double sse1_t_old = 0.0;
      if (C1_old > 0) {
        sse1_t_old = Q1_old - (S1_old * S1_old) / double(C1_old);
      }
      // Compute new SSE contribution at (t, g_new)
      double sse1_t_new = Q1_new - (S1_new * S1_new) / double(C1_new);
      delta1 += (sse1_t_new - sse1_t_old);
    }
    
    // 4) New total = original total + Δ_old + Δ_new
    obj[r] = total0 + delta0 + delta1;
  }
  
  // 6) Select best candidate
  arma::uword best = obj.index_min();
  
  // 7) Build resulting group vector (1-based) and output
  arma::uvec grBest0 = gr0;           // copy zero-based
  grBest0[idx]      = rep0[best];     // replace one unit
  arma::uvec grBest1 = grBest0 + 1;   // back to 1-based
  
  arma::vec out(N + 1);
  out[0]           = obj[best];
  out.subvec(1, N) = arma::conv_to<arma::vec>::from(grBest1);
  
  return out;
}

// [[Rcpp::export]]
NumericVector wgfeJump_unbalanced_cpp(const IntegerVector& replaceR,
                                      int                 i,
                                      const IntegerVector& grR,
                                      const NumericMatrix& Z_N,
                                      const NumericMatrix& alpha_N) {
  // 1) Convert inputs and set up
  int N   = grR.size();
  int R   = replaceR.size();
  int idx = i - 1;  // convert 1-based → 0-based index
  
  // Current grouping (zero-based)
  arma::uvec gr0  = as<arma::uvec>(grR)       - 1;
  // Candidate groups (zero-based)
  arma::uvec rep0 = as<arma::uvec>(replaceR) - 1;
  
  // Wrap Z (T×N) and alpha (T×G) by converting (makes a copy)
  arma::mat Z     = as<arma::mat>(Z_N);
  arma::mat alpha = as<arma::mat>(alpha_N);
  
  int T = Z.n_rows;
  int G = alpha.n_cols;  // number of groups
  
  // 2) Precompute per‐group statistics under current grouping
  //    sumSq_g   = ∑_{i in group g, t: finite} (Z_{t,i} - α_{t,g})^2
  //    cntEnt_g  = ∑_{i in group g, t: finite} 1
  //    cntUnit_g = ∣{i : gr0[i] == g}∣
  arma::vec  sumSq   (G, fill::zeros);
  arma::uvec cntEnt  (G, fill::zeros);
  arma::uvec cntUnit (G, fill::zeros);
  
  // Build a list of column‐indices per group to tally units
  std::vector<std::vector<arma::uword>> cols_in_group(G);
  for (int j = 0; j < N; ++j) {
    int g = gr0[j];
    cols_in_group[g].push_back((arma::uword)j);
    cntUnit[g] += 1;
  }
  
  // Compute sumSq and cntEnt by iterating over all (t,j)
  for (int j = 0; j < N; ++j) {
    int g = gr0[j];
    for (int t = 0; t < T; ++t) {
      double z = Z(t, j);
      if (!std::isfinite(z)) continue;
      double dev = z - alpha(t, g);
      sumSq[g] += dev * dev;
      cntEnt[g] += 1;
    }
  }
  
  // 3) Precompute each group's sigma and the baseline total objective:
  //      total0 = ∑_{g=0..G-1} [ (cntUnit[g]/N) * sqrt(sumSq[g]/cntEnt[g]) ]
  arma::vec sigma0(G, fill::zeros);
  double total0 = 0.0;
  for (int g = 0; g < G; ++g) {
    if (cntEnt[g] > 0) {
      sigma0[g] = std::sqrt(sumSq[g] / double(cntEnt[g]));
      total0  += sigma0[g] * (double(cntUnit[g]) / double(N));
    } else {
      sigma0[g] = std::numeric_limits<double>::quiet_NaN();
    }
  }
  
  // 4) Extract the column of Z for unit 'idx' once, and find its finite‐time indices
  arma::vec Zcol = Z.col(idx);
  arma::uvec t_idx = arma::find_finite(Zcol);  // times t where Z(t,idx) is finite
  arma::uword nDev = t_idx.n_elem;             // number of finite observations for unit idx
  
  // Precompute squared deviations of unit idx under each group g:
  //   dev2_g(t) = (Z(t,idx) - alpha(t,g))^2, for t in t_idx
  arma::mat dev2_per_group(G, nDev);
  for (int g = 0; g < G; ++g) {
    for (arma::uword k = 0; k < nDev; ++k) {
      arma::uword t = t_idx[k];
      double diff = Zcol[t] - alpha(t, g);
      dev2_per_group(g, k) = diff * diff;
    }
  }
  
  // 5) Allocate objective vector for R candidates
  arma::vec obj(R, fill::zeros);
  
  int g_old = gr0[idx];  // original group of unit idx
  
  // 6) For each candidate r, update objective in O(T)
  for (int r = 0; r < R; ++r) {
    int g_new = rep0[r];
    if (g_new == g_old) {
      // No change if candidate equals current group
      obj[r] = total0;
      continue;
    }
    
    // Original group stats
    double sumSq_old    = sumSq[g_old];
    int    cntEnt_old   = cntEnt[g_old];
    int    cntUnit_old  = cntUnit[g_old];
    double sigma_old    = sigma0[g_old];
    
    // New group stats
    double sumSq_new    = sumSq[g_new];
    int    cntEnt_new   = cntEnt[g_new];
    int    cntUnit_new  = cntUnit[g_new];
    double sigma_new    = sigma0[g_new];
    
    // Compute sum of squared deviations for this unit in old and new groups
    double sumDev_old = 0.0;
    double sumDev_new = 0.0;
    for (arma::uword k = 0; k < nDev; ++k) {
      sumDev_old += dev2_per_group(g_old, k);
      sumDev_new += dev2_per_group(g_new, k);
    }
    int countDev = (int)nDev;  // number of finite residuals for this unit
    
    // ---- Update old group contributions ----
    double sumSq_old_p   = sumSq_old - sumDev_old;    // new sumSq for old group
    int    cntEnt_old_p  = cntEnt_old - countDev;    // new countEnt for old group
    int    cntUnit_old_p = cntUnit_old - 1;          // new unit count for old group
    
    double sigma_old_p = 0.0;
    if (cntEnt_old_p > 0) {
      sigma_old_p = std::sqrt(sumSq_old_p / double(cntEnt_old_p));
    } else {
      sigma_old_p = std::numeric_limits<double>::quiet_NaN();
    }
    
    // ---- Update new group contributions ----
    double sumSq_new_p   = sumSq_new + sumDev_new;    // new sumSq for new group
    int    cntEnt_new_p  = cntEnt_new + countDev;    // new countEnt for new group
    int    cntUnit_new_p = cntUnit_new + 1;          // new unit count for new group
    
    double sigma_new_p = std::sqrt(sumSq_new_p / double(cntEnt_new_p));
    
    // ---- Recompute weighted objective for these two groups only ----
    double old_contrib = (double(cntUnit_old) / double(N)) * sigma_old
    + (double(cntUnit_new) / double(N)) * sigma_new;
    
    double new_contrib = (double(cntUnit_old_p) / double(N)) * sigma_old_p
    + (double(cntUnit_new_p) / double(N)) * sigma_new_p;
    
    // Candidate objective = total0 − old_contrib + new_contrib
    obj[r] = total0 - old_contrib + new_contrib;
  }
  
  // 7) Identify best candidate
  arma::uword best = obj.index_min();
  
  // 8) Build output: length (N+1), out[0] = obj[best], out[1..N] = new grouping
  arma::uvec grBest0 = gr0;          // zero-based copy
  grBest0[idx]      = rep0[best];    // replace group for unit idx
  arma::uvec grBest1 = grBest0 + 1;  // back to 1-based
  
  arma::vec out(N + 1);
  out[0]           = obj[best];
  out.subvec(1, N) = arma::conv_to<arma::vec>::from(grBest1);
  
  return NumericVector(out.begin(), out.end());
}



// [[Rcpp::export]]
IntegerVector localJump_unbalanced_cpp(
    const IntegerVector& wgroups_,   // 1-based group labels
    const arma::mat&      Z,         // T × N, possibly containing non-finite entries
    const IntegerVector&  gee_,      // 1-based list of allowed groups
    const std::string&    method     // either "gfe" or anything else ("wgfe")
) {
  const int N = wgroups_.size();
  if (N == 0) Rcpp::stop("`wgroups` must be nonempty");
  if (gee_.size() == 0) Rcpp::stop("`gee` must be nonempty");
  if (Z.n_cols != static_cast<arma::uword>(N))
    Rcpp::stop("Number of columns of Z must equal length(wgroups)");
  
  int min_w = min(wgroups_), min_g = min(gee_);
  if (min_w < 1 || min_g < 1)
    Rcpp::stop("All entries of `wgroups` and `gee` must be ≥ 1 (1-based).");
  
  arma::uvec gr0  = as<arma::uvec>(wgroups_) - 1;  // 0-based
  arma::uvec gee0 = as<arma::uvec>(gee_)      - 1; // 0-based
  const arma::uword G = std::max(gr0.max(), gee0.max()) + 1;
  const arma::uword T = Z.n_rows;
  
  // Finite indices per unit
  std::vector<arma::uvec> finite_idx(N);
  for (arma::uword j = 0; j < static_cast<arma::uword>(N); ++j)
    finite_idx[j] = find_finite(Z.col(j));
  
  // Per-group aggregates
  arma::mat  S(T, G, arma::fill::zeros);
  arma::imat C(T, G, arma::fill::zeros);
  arma::mat  Q(T, G, arma::fill::zeros);
  arma::uvec cntUnit(G, arma::fill::zeros);   // units per group
  
  for (arma::uword j = 0; j < static_cast<arma::uword>(N); ++j) {
    arma::uword g = gr0[j];
    cntUnit[g] += 1u;
    const arma::uvec& t_idx = finite_idx[j];
    for (arma::uword k = 0; k < t_idx.n_elem; ++k) {
      arma::uword t = t_idx[k];
      double zval = Z(t, j);
      S(t, g) += zval;
      Q(t, g) += zval * zval;
      C(t, g) += 1;
    }
  }
  
  arma::vec  SSE(G, arma::fill::zeros);
  arma::uvec cntEnt(G, arma::fill::zeros);
  arma::vec  sigma(G, arma::fill::zeros);
  
  for (arma::uword g = 0; g < G; ++g) {
    double sse_g = 0.0; arma::uword total = 0;
    for (arma::uword t = 0; t < T; ++t) {
      int c = C((int)t, (int)g);
      if (c > 0) {
        double s = S((int)t, (int)g), q = Q((int)t, (int)g);
        sse_g += (q - (s * s) / double(c));
        total += (arma::uword)c;
      }
    }
    SSE[g]    = sse_g;
    cntEnt[g] = total;
    sigma[g]  = (total > 0u) ? std::sqrt(sse_g / double(total))
      : std::numeric_limits<double>::quiet_NaN();
  }
  
  double total0 = 0.0;
  if (method == "gfe") {
    total0 = arma::accu(SSE);
  } else {
    for (arma::uword g = 0; g < G; ++g)
      if (cntEnt[g] > 0u) total0 += sigma[g] * (double(cntUnit[g]) / double(N));
  }
  
  int count = 0, idx_i = -1;
  while (count < N) {
    idx_i = (idx_i + 1) % N;
    arma::uword g_old = gr0[(arma::uword)idx_i];
    
    // --- NEW: if moving this unit would empty its group, skip this unit
    if (cntUnit[g_old] <= 1u) { ++count; continue; }
    
    // Build candidates (gee0 \ {g_old})
    std::vector<arma::uword> candidates;
    candidates.reserve(gee0.n_elem - 1);
    for (arma::uword k = 0; k < gee0.n_elem; ++k) {
      arma::uword g_cand = gee0[k];
      if (g_cand != g_old) candidates.push_back(g_cand);
    }
    const arma::uword R = candidates.size();
    if (R == 0u) { ++count; continue; }
    
    const arma::uvec& t_idx_i = finite_idx[(arma::uword)idx_i];
    arma::uword       nDev    = t_idx_i.n_elem;
    arma::colvec      z_i     = Z.col((arma::uword)idx_i);
    
    arma::vec obj_cand(R, arma::fill::zeros);
    for (arma::uword r = 0; r < R; ++r) {
      arma::uword g_new = candidates[r];
      
      // --- NEW: quick unit-count feasibility (avoid emptying old group)
      if (cntUnit[g_old] <= 1u) { obj_cand[r] = std::numeric_limits<double>::infinity(); continue; }
      
      bool removal_invalid = false, addition_invalid = false;
      double delta0 = 0.0, delta1 = 0.0;
      
      for (arma::uword t = 0; t < T; ++t) {
        double zval = z_i[t];
        int C_old = C((int)t, (int)g_old);
        int C_new = C((int)t, (int)g_new);
        
        if (std::isfinite(zval)) {
          if (C_old == 1) { removal_invalid = true; break; }   // per-time emptiness
          double S0=S((int)t,(int)g_old), Q0=Q((int)t,(int)g_old);
          double S0p=S0 - zval, Q0p=Q0 - zval*zval; int C0p=C_old - 1;
          double sse0_old = Q0 - (S0*S0)/double(C_old);
          double sse0_new = (C0p>0) ? (Q0p - (S0p*S0p)/double(C0p)) : 0.0;
          delta0 += (sse0_new - sse0_old);
          
          double S1=S((int)t,(int)g_new), Q1=Q((int)t,(int)g_new);
          double S1p=S1 + zval, Q1p=Q1 + zval*zval; int C1p=C_new + 1;
          double sse1_old = (C_new>0) ? (Q1 - (S1*S1)/double(C_new)) : 0.0;
          double sse1_new = Q1p - (S1p*S1p)/double(C1p);
          delta1 += (sse1_new - sse1_old);
        } else {
          if (C_new == 0) { addition_invalid = true; break; }
        }
      }
      
      if (removal_invalid || addition_invalid) {
        obj_cand[r] = std::numeric_limits<double>::infinity();
      } else if (method == "gfe") {
        obj_cand[r] = total0 + delta0 + delta1;
      } else {
        double sse_old   = SSE[g_old];
        double sse_new_g = SSE[g_new];
        
        arma::uword cntEnt_old0 = cntEnt[g_old];
        arma::uword cntEnt_new0 = cntEnt[g_new];
        arma::uword n_old       = cntUnit[g_old];
        arma::uword n_new       = cntUnit[g_new];
        
        double sse_old_p = sse_old   + delta0;
        double sse_new_p = sse_new_g + delta1;
        
        arma::uword cntEnt_old_p = cntEnt_old0 - nDev;
        arma::uword cntEnt_new_p = cntEnt_new0 + nDev;
        arma::uword n_old_p      = n_old - 1u;
        arma::uword n_new_p      = n_new + 1u;
        
        double sigma_old   = (cntEnt_old0>0u) ? std::sqrt(sse_old   / double(cntEnt_old0)) : 0.0;
        double sigma_new   = (cntEnt_new0>0u) ? std::sqrt(sse_new_g / double(cntEnt_new0)) : 0.0;
        double sigma_old_p = (cntEnt_old_p>0u) ? std::sqrt(sse_old_p / double(cntEnt_old_p)) : 0.0;
        double sigma_new_p = std::sqrt(sse_new_p / double(cntEnt_new_p));
        
        double old_contrib = sigma_old * (double(n_old) / double(N))
          + sigma_new * (double(n_new) / double(N));
        double new_contrib = sigma_old_p * (double(n_old_p) / double(N))
          + sigma_new_p * (double(n_new_p) / double(N));
        
        obj_cand[r] = total0 - old_contrib + new_contrib;
      }
    }
    
    arma::uword best_idx = obj_cand.index_min();
    double obj_best = obj_cand[best_idx];
    
    if (!(obj_best < total0)) { ++count; continue; }
    
    // Apply best move
    arma::uword g_best = candidates[best_idx];
    double delta0_best = 0.0, delta1_best = 0.0;
    for (arma::uword t = 0; t < T; ++t) {
      double zval = z_i[t];
      int C_old = C((int)t,(int)g_old);
      int C_new = C((int)t,(int)g_best);
      if (std::isfinite(zval)) {
        double S0=S((int)t,(int)g_old), Q0=Q((int)t,(int)g_old);
        double S0p=S0 - zval, Q0p=Q0 - zval*zval; int C0p=C_old - 1;
        double sse0_old = Q0 - (S0*S0)/double(C_old);
        double sse0_new = (C0p>0) ? (Q0p - (S0p*S0p)/double(C0p)) : 0.0;
        delta0_best += (sse0_new - sse0_old);
        S((int)t,(int)g_old) -= zval; Q((int)t,(int)g_old) -= (zval*zval); C((int)t,(int)g_old) -= 1;
        
        double S1=S((int)t,(int)g_best), Q1=Q((int)t,(int)g_best);
        double S1p=S1 + zval, Q1p=Q1 + zval*zval; int C1p=C_new + 1;
        double sse1_old = (C_new>0) ? (Q1 - (S1*S1)/double(C_new)) : 0.0;
        double sse1_new = Q1p - (S1p*S1p)/double(C1p);
        delta1_best += (sse1_new - sse1_old);
        S((int)t,(int)g_best) += zval; Q((int)t,(int)g_best) += (zval*zval); C((int)t,(int)g_best) += 1;
      }
    }
    
    // Update summaries & objective
    SSE[g_old]  += delta0_best;
    SSE[g_best] += delta1_best;
    
    cntEnt[g_old]  -= nDev;
    cntEnt[g_best] += nDev;
    
    cntUnit[g_old]  -= 1u;
    cntUnit[g_best] += 1u;
    
    if (method != "gfe") {
      double sse_old_p = SSE[g_old], sse_new_p = SSE[g_best];
      double sigma_old_p = (cntEnt[g_old]>0u) ? std::sqrt(sse_old_p / double(cntEnt[g_old])) : 0.0;
      double sigma_new_p = std::sqrt(sse_new_p / double(cntEnt[g_best]));
      sigma[g_old] = sigma_old_p; sigma[g_best] = sigma_new_p;
      total0 = obj_best;
    } else {
      total0 = arma::accu(SSE);
    }
    
    gr0[(arma::uword)idx_i] = g_best;
    count = 0;
  }
  
  return wrap(gr0 + 1);  // back to 1-based
}


// [[Rcpp::export]]
IntegerVector localJump_het_unbalanced_cpp(IntegerVector wgroups,
                                           Rcpp::List   zList,
                                           const IntegerVector& gee,
                                           const std::string& method) {
  int N = wgroups.size();
  if ((int)gee.size() == 0)
    Rcpp::stop("`gee` must be nonempty");
  
  // ------------------------------------------------------------
  // 1) Compute initial slopes (thetaR), Z, alpha, validity, and objective
  // ------------------------------------------------------------
  arma::mat thetaR = calcGroupSlopes_unbalanced_cpp(zList, wgroups);
  Rcpp::NumericMatrix theta_N = Rcpp::wrap(thetaR);    // for passing to computeZ
  
  arma::mat Z = computeZ_unbalanced_cpp(zList, theta_N, wgroups);
  List  aList = computeAlpha_unbalanced_cpp(Z, wgroups);
  arma::mat a    = as<arma::mat>(aList["alpha"]);
  bool    valid  = as<bool>(aList["valid"]);
  
  double oldObj;
  if (method == "gfe") {
    oldObj = gfeObj_unbalanced_cpp(Z, wgroups);
  } else {
    oldObj = wgfeObj_unbalanced_cpp(Z, wgroups);
  }
  
  Rcpp::NumericMatrix Z_N     = Rcpp::wrap(Z);
  Rcpp::NumericMatrix alpha_N = Rcpp::wrap(a);
  
  // ------------------------------------------------------------
  // 2) Local search: stop only when
  //      (a) we’ve made N consecutive non‐improving steps
  //   AND (b) valid == true
  //   AND (c) no entry of thetaR is zero
  // ------------------------------------------------------------
  int count = 0;
  int i_idx = 0;  // will cycle 1..N
  
  // Note: we test (count < N || !valid || any(thetaR == 0)) to keep looping
  while (count < N) {
    // pick the next unit index i in 1..N
    i_idx = (i_idx % N) + 1;  // i_idx goes 1→2→…→N→1→…
    
    // build the set of alternative groups for unit i_idx
    std::vector<int> tmp;
    tmp.reserve(gee.size() - 1);
    for (int g : gee) {
      if (g != wgroups[i_idx - 1]) {
        tmp.push_back(g);
      }
    }
    IntegerVector replaceR = wrap(tmp);
    
    // call the appropriate unbalanced jump routine
    arma::vec nb;
    if (method == "gfe") {
      nb = gfeJump_unbalanced_cpp(replaceR, i_idx, wgroups, Z_N);
    } else {
      nb = wgfeJump_unbalanced_cpp(replaceR, i_idx, wgroups, Z_N, alpha_N);
    }
    
    double obj = nb[0];
    if (obj < oldObj) {
      // ─────────────────────────────────────────────────────────────
      // Accept the new grouping (the first element of nb is the obj,
      // the next N elements are the new group labels for all units).
      // ─────────────────────────────────────────────────────────────
      for (int j = 0; j < N; ++j) {
        wgroups[j] = (int) nb[j + 1];
      }
      oldObj = obj;
      thetaR    = calcGroupSlopes_unbalanced_cpp(zList, wgroups);
      Z          = computeZ_unbalanced_cpp(zList, wrap(thetaR), wgroups);
      aList      = computeAlpha_unbalanced_cpp(Z, wgroups);
      a          = as<arma::mat>(aList["alpha"]);
      valid      = as<bool>(aList["valid"]);
      if (valid == false){
        continue;
      }
      count  = 0;
      Z_N        = wrap(Z);
      alpha_N    = wrap(a);
      theta_N   = wrap(thetaR);
      
    } else { 
      thetaR    = calcGroupSlopes_unbalanced_cpp(zList, wgroups);
      Z          = computeZ_unbalanced_cpp(zList, wrap(thetaR), wgroups);
      aList      = computeAlpha_unbalanced_cpp(Z, wgroups);
      a          = as<arma::mat>(aList["alpha"]);
      valid      = as<bool>(aList["valid"]);
      if (valid == false){
        if (i_idx == N){
          ++count;
        }
        continue;
      }
      // ─────────────────────────────────────────────────────────────
      // No improvement → increment the “bad‐step” counter
      // ─────────────────────────────────────────────────────────────
      ++count;
    }
    
    // end of while‐body; the loop condition re‐evaluates:
    //   keep going while (count < N) OR (valid == false) OR any(thetaR == 0)
  }
  
  return wgroups;
}


// [[Rcpp::export]]
Rcpp::IntegerVector assignGroups_unbalanced_cpp(
    const Rcpp::List&           zList,
    const Rcpp::NumericMatrix&  theta,
    const arma::mat&            alpha
) {
  int N = zList.size();
  if (N == 0) {
    return Rcpp::IntegerVector();
  }
  
  int T = alpha.n_rows;
  int G = alpha.n_cols;
  
  // 1) Load θ into Armadillo
  arma::mat th = Rcpp::as<arma::mat>(theta);
  int p = th.n_rows;       // number of covariates
  int C = th.n_cols;       // if >1 ⇒ heterogeneous
  bool heterogeneous = (C > 1);
  
  if (!(C == 1 || C == G)) {
    Rcpp::stop("`theta` must be either p×1 (homogeneous) or p×G (heterogeneous).");
  }
  
  // 2) Build pointers to β_g for each group (or a single β_hom if homogeneous)
  std::vector<const double*> beta_ptrs;
  arma::colvec b;
  if (!heterogeneous) {
    // Homogeneous: either p×1 or 1×p
    if (th.n_cols == 1 && th.n_rows == p) {
      b = th.col(0);
    }
    else if (th.n_rows == 1 && th.n_cols == p) {
      b = th.row(0).t();
    }
    else {
      Rcpp::stop("For homogeneous: `theta` must be p×1 or 1×p.");
    }
  }
  else {
    // Heterogeneous: build a pointer to the start of each column
    const double* th_ptr = th.memptr();  
    beta_ptrs.resize(G);
    for (int g = 0; g < G; ++g) {
      // column g starts at offset g*p
      beta_ptrs[g] = th_ptr + (std::ptrdiff_t)g * p;
    }
  }
  const double* beta_hom = b.memptr();  // valid only if !heterogeneous
  
  // 3) Pointer to alpha data (column‐major: α(t, g) is at alpha_ptr[g*T + t])
  const double* alpha_ptr = alpha.memptr();
  
  Rcpp::IntegerVector out(N);
  
  // 4) Main loop: for each unit i, find best‐SSE group
  for (int i = 0; i < N; ++i) {
    const arma::mat& zi = Rcpp::as<arma::mat>(zList[i]);
    if (zi.n_rows != T || zi.n_cols != (p + 1)) {
      Rcpp::stop("zList[[%d]] must be %d×%d, but got %dx%d.",
                 i, T, p + 1, zi.n_rows, zi.n_cols);
    }
    const double* zi_ptr = zi.memptr();
    // Layout of zi_ptr (column‐major):
    //   – Column 0 (Y_i):     elements [0 .. T-1]
    //   – Column 1 (X_{⋅,1}): elements [  T .. 2T-1 ]
    //   – Column 2 (X_{⋅,2}): elements [2T .. 3T-1 ], etc.
    
    double bestSSE = std::numeric_limits<double>::infinity();
    int    bestG   = 1;  // 1‐based group index
    
    std::ptrdiff_t col_stride = T;  // each column in zi has T rows
    
    // 5) Loop over each group g = 0..G-1
    for (int g = 0; g < G; ++g) {
      const double* beta_ptr = heterogeneous 
      ? beta_ptrs[g] 
      : beta_hom;
      
      double sse = 0.0;
      
      // 6) Loop over each time t = 0..T-1
      for (int t = 0; t < T; ++t) {
        double y_it = zi_ptr[t];  
        if (!std::isfinite(y_it)) continue;
        
        double a_gt = alpha_ptr[(std::ptrdiff_t)g * T + t];
        if (!std::isfinite(a_gt)) continue;
        
        // Compute dot‐product x_{i,t}ᵀ β_g, checking finiteness
        double dot = 0.0;
        bool   okX = true;
        // base index for covariate X_{i,t,1} is t + 1*T
        std::ptrdiff_t base = t + col_stride;
        for (int j = 0; j < p; ++j, base += col_stride) {
          double x = zi_ptr[base];
          if (!std::isfinite(x)) {
            okX = false;
            break;
          }
          dot += x * beta_ptr[j];
        }
        if (!okX) continue;
        
        double resid = y_it - dot;
        double diff  = resid - a_gt;
        sse += diff * diff;
      } // end for(t)
      
      if (sse < bestSSE) {
        bestSSE = sse;
        bestG   = g + 1;  // convert to 1‐based
      }
    } // end for(g)
    
    out[i] = bestG;
  } // end for(i)
  
  return out;
}



// [[Rcpp::export]]
Rcpp::IntegerVector assignGroups_wgfe_unbalanced_cpp(
    const Rcpp::List&         zList,
    const Rcpp::NumericMatrix theta,
    const arma::mat&          Alpha,
    const arma::vec&          SigmaG
) {
  int N = zList.size();
  if (N == 0) {
    return Rcpp::IntegerVector();
  }
  
  // 1) Dimensions from Alpha and SigmaG
  int T = Alpha.n_rows;
  int G = Alpha.n_cols;
  if ((int)SigmaG.n_elem != G) {
    Rcpp::stop("SigmaG must have length G = Alpha.n_cols");
  }
  
  // 2) Load θ into Armadillo (p × C matrix)
  arma::mat th = Rcpp::as<arma::mat>(theta);
  int p = th.n_rows;       // number of regressors
  int C = th.n_cols;       // if >1 ⇒ heterogeneous
  
  if (!(C == 1 || C == G)) {
    Rcpp::stop("`theta` must be either p×1 (homogeneous) or p×G (heterogeneous).");
  }
  bool heterogeneous = (C > 1);
  
  // 3) Extract homogeneous slope‐vector if needed
  //    and also build pointers to each column if heterogeneous.
  std::vector<const double*> beta_ptrs;
  arma::colvec b;                // only used if !heterogeneous
  if (!heterogeneous) {
    // Accept either p×1 or 1×p
    if (th.n_cols == 1 && th.n_rows == p) {
      b = th.col(0);
    }
    else if (th.n_rows == 1 && th.n_cols == p) {
      b = th.row(0).t();
    }
    else {
      Rcpp::stop("For homogeneous: `theta` must be p×1 or 1×p.");
    }
  }
  else {
    // build pointers to each column of `th`
    const double* th_ptr = th.memptr();  // column-major storage
    beta_ptrs.resize(G);
    for (int g = 0; g < G; ++g) {
      // each column has p entries; column g starts at offset g*p
      beta_ptrs[g] = th_ptr + (std::ptrdiff_t)g * p;
    }
  }
  const double* beta_hom =  b.memptr(); // pointer to homogeneous slopes
  
  // 4) Precompute inverse of SigmaG (so we avoid division inside inner loop)
  arma::vec invS = 1.0 / SigmaG;
  const double* invS_ptr   = invS.memptr();
  const double* SigmaG_ptr = SigmaG.memptr();
  
  // 5) Cache pointer to Alpha’s data (column‐major: α(t,g) is at Alpha.memptr()[g*T + t])
  const double* alpha_ptr = Alpha.memptr();
  
  Rcpp::IntegerVector out(N);
  
  // 6) Main loop: assign each unit i to the group that minimizes WGFE‐criterion
  for (int i = 0; i < N; ++i) {
    // 6a) Load zi (T × (p+1) matrix).  
    //     We’ll only read from its memory directly—no copies into Yi/Xi.
    const arma::mat& zi = Rcpp::as<arma::mat>(zList[i]);
    if (zi.n_rows != T || zi.n_cols != p + 1) {
      Rcpp::stop("zList[[%d]] must be exactly %d×%d, but got %dx%d.",
                 i, T, p + 1, zi.n_rows, zi.n_cols);
    }
    const double* zi_ptr = zi.memptr();  
    //   Layout of zi_ptr: 
    //   – Column 0 (Y_i): T entries at indices [0 .. T-1]
    //   – Column 1 (X_{⋅,1}): T entries at indices [  T .. 2T-1 ]
    //   – Column 2 (X_{⋅,2}): T entries at indices [2T .. 3T-1], etc.
    
    double bestObj = std::numeric_limits<double>::infinity();
    int    bestG   = 1;     // 1‐based index of the best group
    
    // 6b) Try each group g = 0..(G-1)
    for (int g = 0; g < G; ++g) {
      // 6b-i) pick pointer to β_g
      const double* beta_ptr = heterogeneous 
      ? beta_ptrs[g]
      : beta_hom;
      
      double sumDevSq = 0.0;
      int    count_t  = 0;
      double invS_g   = invS_ptr[g];
      double Sigma_g  = SigmaG_ptr[g];
      
      // 6b-ii) loop over time periods t = 0..T-1
      //        check finiteness of y_it, α(t,g), each x_{i,t,j}
      //        then accumulate (y_it − x′β_g − α_{t,g})²
      const std::ptrdiff_t col_stride = T;  
      // (Because column j of zi lives at zi_ptr + j*T)
      
      for (int t = 0; t < T; ++t) {
        double y_it  = zi_ptr[t];                 
        double a_gt  = alpha_ptr[(std::ptrdiff_t)g * T + t];
        if (!std::isfinite(y_it) || !std::isfinite(a_gt)) {
          continue;
        }
        
        // check each covariate x_{i,t,j} and build the dot‐product
        double dot = 0.0;
        bool   okX = true;
        std::ptrdiff_t base = t + col_stride;  
        // base now points to element (t, j=1) in zi, i.e. x_{i,t,1}
        for (int j = 0; j < p; ++j, base += col_stride) {
          double x = zi_ptr[base];
          if (!std::isfinite(x)) {
            okX = false;
            break;
          }
          dot += x * beta_ptr[j];
        }
        if (!okX) continue;
        
        double resid = y_it - dot;
        double diff  = resid - a_gt;
        sumDevSq    += diff * diff;
        ++count_t;
      } // end t‐loop
      
      if (count_t == 0) {
        // no valid rows for this group ⇒ skip
        continue;
      }
      
      // 6b-iii) compute WGFE objective for group g:
      //       obj = (Σ diff²)/σ_g + count_t * σ_g
      double obj = sumDevSq * invS_g + (double)count_t * Sigma_g;
      
      // 6b-iv) keep strictly smaller (ties go to smaller g automatically)
      if (obj < bestObj - 1e-12) {
        bestObj = obj;
        bestG   = g + 1;  // store 1‐based index
      }
    } // end for(g)
    
    out[i] = bestG;
  } // end for(i)
  
  return out;
}



// [[Rcpp::export]]
Rcpp::IntegerVector refineGroups_unbalanced_cpp(
    const Rcpp::List&         zList,
    Rcpp::IntegerVector       wgroups,
    bool                      heterogeneous,
    const std::string&        method
) {
  int N = wgroups.size();
  if (N == 0) {
    return wgroups;  
  }
  
  // 1) Figure out G = max(wgroups)
  int G = 0;
  for (int i = 0; i < N; ++i) {
    if (wgroups[i] > G) G = wgroups[i];
  }
  
  // 2) Build gee = {1,2,...,G} once
  Rcpp::IntegerVector gee(G);
  for (int g = 0; g < G; ++g) {
    gee[g] = g + 1; 
  }
  
  // 3) Prepare “prev” vector for checking convergence
  Rcpp::IntegerVector prev(N);
  
  // 4) We will need some reusable containers:
  //    - thetaR: the R-side theta (p×1 or p×G) 
  //    - Z:      arma::mat of “Z” (T×N or whatever computeZ returns)
  //    - alpha:  the Rcpp::List returned by computeAlpha
  //    - alphaMat: arma::mat extracted from alpha["alpha"]
  //    - SigmaG: arma::vec of group variances (only for WGFE)
  //
  // We will only resize them when absolutely needed, instead of re-allocating each iteration.
  
  Rcpp::NumericMatrix thetaR; 
  arma::mat            Z;      
  Rcpp::List           alpha;  
  arma::mat            alphaMat; 
  arma::vec            SigmaG; 
  
  bool changed = true;
  int  iter    = 0;
  const int MAX_ITER = 20;
  while (changed && iter < MAX_ITER) {
    ++iter;
    prev = wgroups;  // copy old assignment
    
    if (!heterogeneous) {
      // -------------- homogeneous case --------------
      if (method == "gfe") {
        // 5a) Compute θ (p×1 vector)
        arma::vec thv = slopeGradGFE_unbalanced_cpp(zList, wgroups);
        
        // 5b) Wrap it just once into thetaR (p×1)
        //     Only resize thetaR if its size has changed.
        int p = thv.n_elem;
        if (thetaR.nrow() != p || thetaR.ncol() != 1) {
          thetaR = Rcpp::NumericMatrix(p, 1);
        }
        // Copy data from thv → thetaR’s column 0
        for (int ii = 0; ii < p; ++ii) {
          thetaR(ii, 0) = thv[ii];
        }
        
        // 5c) Compute Z ← computeZ_unbalanced_cpp(zList, thetaR, wgroups)
        //     (returns arma::mat)
        Z = computeZ_unbalanced_cpp(zList, thetaR, wgroups);
        
        // 5d) Compute alpha ← computeAlpha_unbalanced_cpp(Z, wgroups)
        alpha = computeAlpha_unbalanced_cpp(Z, wgroups);
        
        // 5e) Extract α matrix and validity flag
        alphaMat = Rcpp::as<arma::mat>( alpha["alpha"] );
        bool alpha_valid = Rcpp::as<bool>( alpha["valid"] );
        
        // 5f) If α invalid, do one local jump and return immediately
        if (!alpha_valid) {
          wgroups = localJump_unbalanced_cpp(wgroups, Z, gee, method);
          return wgroups;
        }
        
        // 5g) Re-assign groups using our (already optimized) C++ function:
        //     assignGroups_unbalanced_cpp(zList, thetaR, alphaMat)
        wgroups = assignGroups_unbalanced_cpp(zList, thetaR, alphaMat);
      }
      else {
        // ----------- homogeneous WGFE case -----------
        // 5a) Compute demeaned XX and XY (two Rcpp/List elements)
        Rcpp::List tmp = computeXXY_demeaned_unbalanced_cpp(zList, wgroups);
        
        // 5b) Solve for θ using our inner loop FP solver
        //     (returns arma::mat of size p×1)
        arma::mat thm = wc_fp_loop_unbalanced_cpp(zList, wgroups, tmp[0], tmp[1], gee);
        
        // 5c) Wrap thm into thetaR (resize once if necessary)
        int p = thm.n_rows;
        if (thetaR.nrow() != p || thetaR.ncol() != 1) {
          thetaR = Rcpp::NumericMatrix(p, 1);
        }
        for (int ii = 0; ii < p; ++ii) {
          thetaR(ii, 0) = thm(ii, 0);
        }
        
        // 5d) Compute Z ← computeZ_unbalanced_cpp(zList, thetaR, wgroups)
        Z = computeZ_unbalanced_cpp(zList, thetaR, wgroups);
        
        // 5e) Compute α and extract its matrix
        alpha = computeAlpha_unbalanced_cpp(Z, wgroups);
        alphaMat = Rcpp::as<arma::mat>( alpha["alpha"] );
        bool alpha_valid = Rcpp::as<bool>( alpha["valid"] );
        if (!alpha_valid) {
          wgroups = localJump_unbalanced_cpp(wgroups, Z, gee, method);
          return wgroups;
        }
        
        // 5f) Compute Σ_g (using our optimized C++)
        SigmaG = computeSigma_unbalanced_cpp(Z, alphaMat, wgroups);
        
        // 5g) Re-assign groups via the WGFE assignment routine
        wgroups = assignGroups_wgfe_unbalanced_cpp(zList, thetaR, alphaMat, SigmaG);
      }
    }
    else {
      // ---------------- heterogeneous case ----------------
      if (method == "gfe") {
        // 5a) Compute p×G matrix of group slopes
        arma::mat thm = calcGroupSlopes_unbalanced_cpp(zList, wgroups);
        
        // 5b) Wrap it into thetaR (resize only if dimensions changed)
        int p = thm.n_rows;
        int Pg = thm.n_cols;  // should equal G
        if (thetaR.nrow() != p || thetaR.ncol() != Pg) {
          thetaR = Rcpp::NumericMatrix(p, Pg);
        }
        for (int rr = 0; rr < p; ++rr) {
          for (int cc = 0; cc < Pg; ++cc) {
            thetaR(rr, cc) = thm(rr, cc);
          }
        }
        
        // 5c) Compute Z
        Z = computeZ_unbalanced_cpp(zList, thetaR, wgroups);
        
        // 5d) (Optionally) compute objective if you need it for debugging
        double objVal = gfeObj_unbalanced_cpp(Z, wgroups);
        
        // 5e) Compute α
        alpha = computeAlpha_unbalanced_cpp(Z, wgroups);
        alphaMat = Rcpp::as<arma::mat>( alpha["alpha"] );
        bool alpha_valid = Rcpp::as<bool>( alpha["valid"] );
        //Rcpp::Rcout <<  "obj = " << objVal<< "   " << alpha_valid << "\n";
        // 5f) If invalid, do a heterogeneous local jump and continue (or return)
        if (!alpha_valid) {
          Rcpp::Rcout << "hi" << "\n";
          wgroups = localJump_het_unbalanced_cpp(prev, zList, gee, method);
        } 
        else {
          // 5g) Otherwise assign groups as in the unbalanced‐GFE routine
          wgroups = assignGroups_unbalanced_cpp(zList, thetaR, alphaMat);
        }
      }
      else {
        // ------------ heterogeneous WGFE --------------
        // 5a) Compute slopes (same as GFE for heterogeneous)
        arma::mat thm = calcGroupSlopes_unbalanced_cpp(zList, wgroups);
        
        // 5b) Wrap it into thetaR
        int p = thm.n_rows;
        int Pg = thm.n_cols;  // should be G
        if (thetaR.nrow() != p || thetaR.ncol() != Pg) {
          thetaR = Rcpp::NumericMatrix(p, Pg);
        }
        for (int rr = 0; rr < p; ++rr) {
          for (int cc = 0; cc < Pg; ++cc) {
            thetaR(rr, cc) = thm(rr, cc);
          }
        }
        
        // 5c) Compute Z
        Z = computeZ_unbalanced_cpp(zList, thetaR, wgroups);
        
        // 5d) (Optionally) compute objective
        // double objVal = gfeObj_unbalanced_cpp(Z, wgroups);
        
        // 5e) Compute α
        alpha = computeAlpha_unbalanced_cpp(Z, wgroups);
        alphaMat = Rcpp::as<arma::mat>( alpha["alpha"] );
        bool alpha_valid = Rcpp::as<bool>( alpha["valid"] );
        
        // 5f) If invalid or if any θ = 0, bail out
        if (!alpha_valid) {
          return wgroups;
        }
        
        // 5g) Compute Σ_g
        SigmaG = computeSigma_unbalanced_cpp(Z, alphaMat, wgroups);
        
        // 5h) Re-assign groups under WGFE
        wgroups = assignGroups_wgfe_unbalanced_cpp(zList, thetaR, alphaMat, SigmaG);
      }
    }
    
    // 6) Check convergence: did any entry of wgroups change?
    changed = false;
    for (int i = 0; i < N; ++i) {
      if (wgroups[i] != prev[i]) {
        changed = true;
        break;
      }
    }
  } // end while

  return wgroups;
}



// [[Rcpp::export]]
arma::vec se_unbalanced_cpp(const Rcpp::NumericVector& Y,
                            const Rcpp::NumericMatrix& X,
                            const Rcpp::NumericVector& theta0,
                            const Rcpp::IntegerVector& groupR,
                            const Rcpp::NumericMatrix& alpha0,
                            const Rcpp::NumericVector& sigma0,
                            int t) {
  int N  = groupR.size();
  int TT = t;
  int p  = X.ncol();
  int G  = sigma0.size();
  
  // Convert inputs into Armadillo types
  arma::vec  theta = Rcpp::as<arma::vec>(theta0);        // p × 1
  arma::vec  Yv    = Rcpp::as<arma::vec>(Y);             // (N·TT) × 1
  arma::mat  Xmat  = Rcpp::as<arma::mat>(X);             // (N·TT) × p
  arma::mat  alpha = Rcpp::as<arma::mat>(alpha0);        // TT × G
  arma::vec  sigma = Rcpp::as<arma::vec>(sigma0);        // G × 1
  
  // Re‐index group labels to 0..(G−1)
  arma::ivec grp(N);
  for (int i = 0; i < N; ++i) {
    grp[i] = groupR[i] - 1;
  }
  
  // 1) Build covariate matrices per unit: covar[i] is TT × p
  std::vector<arma::mat> covar(N, arma::mat(TT, p));
  for (int i = 0; i < N; ++i) {
    int row0 = i * TT;
    for (int j = 0; j < p; ++j) {
      covar[i].col(j) = Xmat.col(j).rows(row0, row0 + TT - 1);
    }
  }
  
  // 2) Compute group‐time means of X, skipping missing (NaN) entries
  std::vector<arma::mat> x_bar(G, arma::mat(TT, p, arma::fill::zeros));
  // For counting finite observations at each (group k, time s, covariate j)
  std::vector<arma::imat> countGP(G, arma::imat(TT, p, arma::fill::zeros));
  
  for (int i = 0; i < N; ++i) {
    int k = grp[i];
    arma::mat& Xi = covar[i];  // TT × p
    for (int s = 0; s < TT; ++s) {
      for (int j = 0; j < p; ++j) {
        double val = Xi(s, j);
        if (arma::is_finite(val)) {
          x_bar[k](s, j) += val;
          countGP[k](s, j) += 1;
        }
      }
    }
  }
  
  // Finalize group‐time means
  for (int k = 0; k < G; ++k) {
    for (int s = 0; s < TT; ++s) {
      for (int j = 0; j < p; ++j) {
        int cnt = countGP[k](s, j);
        if (cnt > 0) {
          x_bar[k](s, j) /= double(cnt);
        } else {
          // If no unit in group k has a finite X at (s,j), we leave x_bar = 0.
          // Those times will be skipped later for any unit i in group k.
          x_bar[k](s, j) = 0.0;
        }
      }
    }
  }
  
  // 3) Initialize accumulators
  arma::mat SB(p, p, arma::fill::zeros);
  arma::mat SV(p, p, arma::fill::zeros);
  double totalObs = 0.0;
  
  // 4) Loop over units i, accumulating SB and SV, skipping missing time observations
  for (int i = 0; i < N; ++i) {
    int k = grp[i];
    const arma::mat& Xi = covar[i];           // TT × p
    arma::mat Di = Xi - x_bar[k];             // TT × p (may contain NaNs where Xi was NaN)
    
    // Extract Yi and compute residuals Zi = Yi - Xi·theta
    int row0 = i * TT;
    arma::vec Yi = Yv.subvec(row0, row0 + TT - 1);  // TT × 1
    arma::vec XiTheta = Xi * theta;                 // TT × 1 (NaN where Xi had NaN)
    arma::vec Zi = Yi - XiTheta;                    // TT × 1
    
    // Identify valid time indices (s) for unit i:
    // must have Yi[s] finite, Xi row s fully finite, and alpha(s, k) finite
    std::vector<int> validTimes;
    validTimes.reserve(TT);
    for (int s = 0; s < TT; ++s) {
      bool y_ok = arma::is_finite(Yi[s]);
      bool x_ok = true;
      for (std::size_t j = 0; j < Xi.n_cols; ++j) {
        if (!arma::is_finite(Xi(s, j))) {
          x_ok = false;
          break;
        }
      }
      bool a_ok = arma::is_finite(alpha(s, k));
      if (y_ok && x_ok && a_ok) {
        validTimes.push_back(s);
      }
    }
    int m = validTimes.size();
    totalObs += m;  // count each valid (i, s)
    if (m == 0) continue;
    
    double sigk = sigma[k];
    
    // Accumulate SB and SV over valid times
    for (int idx_s = 0; idx_s < m; ++idx_s) {
      int s = validTimes[idx_s];
      arma::rowvec ds = Di.row(s);                      // 1 × p, finite
      double zs = Zi[s] - alpha(s, k);                  // finite
      
      // SB contribution: (dsᵀ ds) / σ_k
      SB += (ds.t() * ds) / sigk;
      
      // SV contribution: double sum over r in validTimes
      for (int idx_r = 0; idx_r < m; ++idx_r) {
        int r = validTimes[idx_r];
        arma::rowvec dr = Di.row(r);                    // 1 × p, finite
        double zr = Zi[r] - alpha(r, k);                 // finite
        SV += (zs * zr) * (ds.t() * dr) / (sigk * sigk);
      }
    }
  }
  
  // 1) Check “all(sigma != 1)” in Armadillo (fallback to an explicit loop if needed):
  bool all_not_one = true;
  for (int k = 0; k < G; ++k) {
    if (sigma[k] == 1.0) {
      all_not_one = false;
      break;
    }
  }
  
  // 2) Use the ternary operator to pick 2 or 1:
  double multiplier = all_not_one ? 2.0 : 1.0;
  
  // 5) Degrees of freedom (unbalanced): totalObs − p − (G·TT) − G
  double df = totalObs - double(p) - double(G * TT) - double(G) * multiplier;
  if (df <= 0){
    Rcpp::stop("Not enough observations to compute standard errors (df <= 0).");
  }
  
  // 6) Compute "meat" and "bread" matrices
  arma::mat Binv = arma::inv(SB / df);  // p × p
  arma::mat V    = SV  / df;            // p × p
  
  arma::mat M    = Binv * V * Binv;     // p × p
  arma::vec se   = arma::sqrt(M.diag() / totalObs);  // p × 1
  
  return se;
}




// [[Rcpp::export]]
Rcpp::NumericMatrix seHet_unbalanced_cpp(const Rcpp::List&         zList,
                                         const Rcpp::NumericMatrix& theta0,
                                         const Rcpp::IntegerVector& groupR,
                                         const Rcpp::NumericMatrix& alpha0) {
  int N = zList.size();
  if (groupR.size() != N) {
    Rcpp::stop("groupR must have length N = zList size");
  }
  
  // Convert inputs
  arma::mat th    = Rcpp::as<arma::mat>(theta0);    // p × G
  arma::mat alpha = Rcpp::as<arma::mat>(alpha0);    // TT × G
  
  int p  = th.n_rows;
  int G  = th.n_cols;
  int TT = alpha.n_rows;
  if (alpha.n_cols != G) {
    Rcpp::stop("theta and alpha must share the same number of columns/groups");
  }
  
  // Map to 0-based groups, count units per group
  arma::ivec grp(N);
  arma::ivec countG(G, arma::fill::zeros);
  for (int i = 0; i < N; ++i) {
    int g = groupR[i] - 1;
    if (g < 0 || g >= G) {
      Rcpp::stop("invalid group label at position %d", i + 1);
    }
    grp[i]    = g;
    countG[g] += 1;
  }
  
  // Precompute TT×p covariate matrices Xi for each unit i
  // Also store Yi for each i
  std::vector<arma::mat> Xi_list(N);   // each = TT × p
  std::vector<arma::vec> Yi_list(N);   // each = TT × 1
  for (int i = 0; i < N; ++i) {
    arma::mat zi = Rcpp::as<arma::mat>(zList[i]);  // TT × (p+1), col 0 = Y, cols 1..p = X
    if (zi.n_rows != TT || zi.n_cols != (p + 1)) {
      Rcpp::stop("zList[[%d]] must be TT × (p+1)", i + 1);
    }
    Yi_list[i] = zi.col(0);           // TT × 1
    Xi_list[i] = zi.cols(1, p);       // TT × p
  }
  
  // Compute time‐specific X means xbar[g] for each group, skipping missing values
  std::vector<arma::mat> xbar(G, arma::mat(TT, p, arma::fill::zeros));
  // And count finite observations at (g, s, j)
  std::vector<arma::imat> countGP(G, arma::imat(TT, p, arma::fill::zeros));
  
  for (int i = 0; i < N; ++i) {
    int g = grp[i];
    arma::mat& Xi = Xi_list[i];  // TT × p
    for (int s = 0; s < TT; ++s) {
      for (int j = 0; j < p; ++j) {
        double xval = Xi(s, j);
        if (arma::is_finite(xval)) {
          xbar[g](s, j)   += xval;
          countGP[g](s, j) += 1;
        }
      }
    }
  }
  // Finalize xbar
  for (int g = 0; g < G; ++g) {
    for (int s = 0; s < TT; ++s) {
      for (int j = 0; j < p; ++j) {
        int cnt = countGP[g](s, j);
        if (cnt > 0) {
          xbar[g](s, j) /= double(cnt);
        } else {
          xbar[g](s, j) = 0.0;
        }
      }
    }
  }
  
  // Prepare output: p × G
  Rcpp::NumericMatrix out(p, G);
  out.attr("dimnames") = R_NilValue; // no row/col names
  
  // Loop over each group g
  for (int g = 0; g < G; ++g) {
    int ng = countG[g];
    if (ng == 0) {
      // If no units in group g, return NA for all p entries
      for (int j = 0; j < p; ++j) {
        out(j, g) = NA_REAL;
      }
      continue;
    }
    
    // We will accumulate over valid (i, s) within group g
    arma::mat SB(p, p, arma::fill::zeros);
    arma::mat SV(p, p, arma::fill::zeros);
    double groupObs = 0.0;  // count of valid (i, s)
    
    arma::colvec betag = th.col(g); // p × 1
    
    // Loop units i in group g
    for (int i = 0; i < N; ++i) {
      if (grp[i] != g) continue;
      arma::mat& Xi = Xi_list[i];    // TT × p
      arma::vec& Yi = Yi_list[i];    // TT × 1
      
      // Compute deviations Di = Xi - xbar[g]; Di(s,·) may have NaNs if Xi(s,·) had NaNs
      arma::mat Di = Xi - xbar[g];   // TT × p
      
      // Residuals before centering: ei = Yi - Xi * betag
      arma::vec ei = Yi - (Xi * betag); // TT × 1 (NaNs where Xi row or Yi was NaN)
      // Then subtract alpha_{s,g}: eg(s) = ei(s) - alpha(s,g)
      arma::vec eg = ei - alpha.col(g); // TT × 1
      
      // Now identify valid times s for this unit i:
      // Must have Yi[s] finite, all Xi(s,·) finite, and alpha(s,g) finite
      std::vector<int> validTimes;
      validTimes.reserve(TT);
      for (int s = 0; s < TT; ++s) {
        if (! arma::is_finite(Yi[s]))        continue;
        // check Xi row s
        bool x_ok = true;
        for (int j = 0; j < p; ++j) {
          if (! arma::is_finite(Xi(s, j))) {
            x_ok = false;
            break;
          }
        }
        if (! x_ok)                           continue;
        if (! arma::is_finite(alpha(s, g)))   continue;
        validTimes.push_back(s);
      }
      int m = validTimes.size();
      groupObs += m;
      if (m == 0) continue;
      
      // Accumulate SB, SV over validTimes
      for (int idx_s = 0; idx_s < m; ++idx_s) {
        int s = validTimes[idx_s];
        arma::rowvec ds = Di.row(s);             // 1 × p
        // SB contribution
        SB += (ds.t() * ds);
        
        double z_s = eg[s];                      // finite
        for (int idx_r = 0; idx_r < m; ++idx_r) {
          int r = validTimes[idx_r];
          arma::rowvec dr = Di.row(r);           // 1 × p
          double z_r = eg[r];                    // finite
          SV += (z_s * z_r) * (ds.t() * dr);
        }
      }
    }
    
    // Degrees of freedom: groupObs − p − TT − 1
    double df = groupObs - double(p) - double(TT);
    if (df <= 0) {
      Rcpp::stop("non‐positive degrees of freedom for group %d", g + 1);
    }
    
    arma::mat Binv = arma::inv(SB / df); // p × p
    arma::mat V    = SV / df;           // p × p
    arma::mat M    = Binv * V * Binv;   // p × p
    arma::vec se   = arma::sqrt(M.diag() / groupObs); // p × 1
    
    // Copy into output
    for (int j = 0; j < p; ++j) {
      out(j, g) = se[j];
    }
  }
  
  return out;
}


