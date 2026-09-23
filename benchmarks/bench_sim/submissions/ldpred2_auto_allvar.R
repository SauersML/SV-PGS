# LDpred2-auto on the summary statistics and 3 cM band that ldpred2_auto_allvar.py wrote to the directory in argv[1],
# at ldpred2_auto.R's settings (bigsnpr 1.12.21). The band holds too many correlations for one dgCMatrix, so the compact
# SFBM is built a column block at a time with add_columns; each column's values are the contiguous rows lo .. hi - 1.
# Writes beta.bin.
suppressPackageStartupMessages({library(bigstatsr); library(bigsnpr); library(Matrix)})
set.seed(1)
folder <- commandArgs(trailingOnly = TRUE)[1]
meta <- scan(file.path(folder, "meta.txt"), quiet = TRUE)
p <- meta[1]; n <- meta[2]; cores <- meta[3]; chunk <- meta[4]
read <- function(name, what, size) { con <- file(file.path(folder, name), "rb"); on.exit(close(con)); readBin(con, what, n = file.size(file.path(folder, name)) / size, size = size, endian = "little") }
sumstats <- matrix(read("sumstats.bin", "double", 8), ncol = 2, byrow = TRUE)
frame <- data.frame(beta = sumstats[, 1], beta_se = sumstats[, 2], n_eff = n)
lo <- read("ld_lo.bin", "integer", 4); hi <- read("ld_hi.bin", "integer", 4)
ld <- read("ld_scores.bin", "double", 8)
values <- file(file.path(folder, "ld_values.bin"), "rb")
corr <- NULL
for (first in seq(1, p, by = chunk)) {
  columns <- first:min(first + chunk - 1, p)
  counts <- hi[columns] - lo[columns]
  offset <- lo[first]
  block <- new("dgCMatrix", i = as.integer(sequence(counts, from = lo[columns] - offset)),
               p = as.integer(c(0, cumsum(counts))), Dim = as.integer(c(hi[max(columns)] - offset, length(columns))),
               x = readBin(values, "double", n = sum(counts), size = 4, endian = "little"))
  if (is.null(corr)) corr <- as_SFBM(block, file.path(folder, "sfbm"), compact = TRUE) else corr$add_columns(block, offset)
}
close(values)
file.remove(file.path(folder, "ld_values.bin"))
stopifnot(nrow(corr) == p, ncol(corr) == p)
ldsc <- with(frame, snp_ldsc(ld, length(ld), chi2 = (beta / beta_se)^2, sample_size = n_eff, blocks = NULL))
heritability <- max(ldsc[["h2"]], 0.001)
chains <- snp_ldpred2_auto(corr, frame, h2_init = heritability, vec_p_init = seq_log(1e-4, 0.2, length.out = 30),
                           ncores = cores, allow_jump_sign = FALSE, shrink_corr = 0.95, use_MLE = FALSE)
spread <- vapply(chains, function(chain) diff(range(chain$corr_est)), numeric(1))
keep <- which(spread > 0.95 * quantile(spread, 0.95, na.rm = TRUE))
h2 <- vapply(chains[keep], function(chain) chain$h2_est, numeric(1))
cat("ldpred2: variants", p, "ldsc h2", ldsc[["h2"]], "intercept", ldsc[["int"]], "h2_init", heritability, "chains kept",
    length(keep), "of", length(chains), "h2_est", median(h2),
    "p_est", median(vapply(chains[keep], function(chain) chain$p_est, numeric(1))), "\n")
if (length(keep) == 0) stop("no LDpred2-auto chain passed the tutorial's filter")
beta <- rowMeans(matrix(sapply(chains[keep], function(chain) chain$beta_est), nrow = p))
con <- file(file.path(folder, "beta.bin"), "wb"); writeBin(as.double(beta), con, size = 8, endian = "little"); close(con)
