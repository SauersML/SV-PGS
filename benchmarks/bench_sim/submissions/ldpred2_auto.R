# LDpred2-auto on the summary statistics and windowed LD that ldpred2_auto.py wrote to the directory in argv[1], at the
# settings of the bigsnpr tutorial and its extended documentation (bigsnpr 1.12.21). Writes beta.bin.
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
correlation <- as(correlation, "generalMatrix")
ld <- Matrix::colSums(correlation^2)
corr <- as_SFBM(correlation, file.path(folder, "sfbm"), compact = TRUE)
ldsc <- with(frame, snp_ldsc(ld, length(ld), chi2 = (beta / beta_se)^2, sample_size = n_eff, blocks = NULL))
# h2_init is only the chains' starting value (LDpred2-auto re-estimates h2); the authors floor LDSC's estimate at 0.001
# (pmax(ldsc[["h2"]], 0.001) in privefl/paper-infer code/run-ldpred2-ukbb.R), since snp_ldpred2_auto refuses h2_init <= 0
heritability <- max(ldsc[["h2"]], 0.001)
# use_MLE = FALSE: the extended documentation's setting ("use TRUE only for GWAS with large N and M"); one
# chromosome's ~12,000 HapMap3-density variants is not a large M
chains <- snp_ldpred2_auto(corr, frame, h2_init = heritability, vec_p_init = seq_log(1e-4, 0.2, length.out = 30),
                           ncores = cores, allow_jump_sign = FALSE, shrink_corr = 0.95, use_MLE = FALSE)
# the tutorial's chain filter: the chains whose predictions' correlation range is near the best chains'
spread <- vapply(chains, function(chain) diff(range(chain$corr_est)), numeric(1))
keep <- which(spread > 0.95 * quantile(spread, 0.95, na.rm = TRUE))
h2 <- vapply(chains[keep], function(chain) chain$h2_est, numeric(1))
cat("ldpred2: ldsc h2", ldsc[["h2"]], "intercept", ldsc[["int"]], "h2_init", heritability, "chains kept", length(keep),
    "of", length(chains), "h2_est", median(h2), "p_est", median(vapply(chains[keep], function(chain) chain$p_est, numeric(1))), "\n")
if (length(keep) == 0) stop("no LDpred2-auto chain passed the tutorial's filter")
# sapply returns a vector, not a matrix, when one chain is kept
beta <- rowMeans(matrix(sapply(chains[keep], function(chain) chain$beta_est), nrow = p))
con <- file(file.path(folder, "beta.bin"), "wb"); writeBin(as.double(beta), con, size = 8, endian = "little"); close(con)
