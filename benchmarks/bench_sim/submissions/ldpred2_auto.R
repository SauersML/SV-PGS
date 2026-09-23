# LDpred2-auto on the summary statistics and banded LD that ldpred2_auto.py wrote to the directory in argv[1]; the
# settings are the competitor workflow's (benchmarks/compete/competitors.R, fit_ldpred2_auto). Writes beta.bin.
suppressPackageStartupMessages({library(bigstatsr); library(bigsnpr); library(Matrix)})
set.seed(1)
folder <- commandArgs(trailingOnly = TRUE)[1]
meta <- scan(file.path(folder, "meta.txt"), quiet = TRUE)
p <- meta[1]; n <- meta[2]; cores <- meta[3]
read <- function(name, what, size) { con <- file(file.path(folder, name), "rb"); on.exit(close(con)); readBin(con, what, n = file.size(file.path(folder, name)) / size, size = size, endian = "little") }
sumstats <- matrix(read("sumstats.bin", "double", 8), ncol = 2, byrow = TRUE)
frame <- data.frame(beta = sumstats[, 1], beta_se = sumstats[, 2], n_eff = n)
correlation <- sparseMatrix(i = read("ld_i.bin", "integer", 4) + 1L, j = read("ld_j.bin", "integer", 4) + 1L,
                            x = read("ld_x.bin", "double", 8), dims = c(p, p), symmetric = TRUE)
correlation <- as(correlation, "dgCMatrix")
ld <- Matrix::colSums(correlation^2)
corr <- as_SFBM(correlation, file.path(folder, "sfbm"), compact = TRUE)
heritability <- with(frame, snp_ldsc(ld, length(ld), chi2 = (beta / beta_se)^2, sample_size = n_eff, blocks = NULL))[["h2"]]
chains <- snp_ldpred2_auto(corr, frame, h2_init = heritability, vec_p_init = seq_log(1e-4, 0.2, length.out = 30),
                           ncores = cores, allow_jump_sign = FALSE, shrink_corr = 0.95)
finite <- vapply(chains, function(chain) all(is.finite(chain$beta_est)) && is.finite(chain$h2_est), logical(1))
if (!is.null(chains[[1]]$corr_est)) {
  # the tutorial's filter: chains whose predictions' correlation range is near the best chains'
  spread <- vapply(chains, function(chain) diff(range(chain$corr_est)), numeric(1))
  keep <- which(finite & spread > 0.95 * quantile(spread[finite], 0.95, na.rm = TRUE))
} else {
  # this bigsnpr returns no corr_est: the chains whose heritability estimate lies within the central 95% of the
  # converged chains' (the paper's divergence check on h2_est)
  h2 <- vapply(chains, function(chain) chain$h2_est, numeric(1))
  band <- quantile(h2[finite], c(0.025, 0.975), na.rm = TRUE)
  keep <- which(finite & h2 >= band[1] & h2 <= band[2])
}
cat("ldpred2: h2_init", heritability, "chains kept", length(keep), "of", length(chains), "finite", sum(finite), "\n")
if (length(keep) == 0) stop("no LDpred2-auto chain passed the tutorial's filter")
# sapply returns a vector, not a matrix, when one chain is kept
beta <- rowMeans(matrix(sapply(chains[keep], function(chain) chain$beta_est), nrow = p))
con <- file(file.path(folder, "beta.bin"), "wb"); writeBin(as.double(beta), con, size = 8, endian = "little"); close(con)
