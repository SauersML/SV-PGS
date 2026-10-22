# LDpred2-auto and lassosum2 on one arm's summary statistics and 3 cM band, which aou_fastpgs.py wrote to the directory
# in argv[1]. The compact SFBM is built as ldpred2_auto_allvar.R builds it (bigsnpr 1.12.21). LDpred2-auto reads the
# GWAS of every training row (sumstats.bin) at ldpred2_auto_allvar.R's settings; lassosum2 reads the GWAS of the
# training rows outside the tuning slice (sumstats_part.bin) at snp_lassosum2's default grid. Writes beta_auto.bin (NaN
# where no chain passes the tutorial's filter), beta_lasso.bin (variants x grid, column-major) and lasso_grid.tsv.
suppressPackageStartupMessages({library(bigstatsr); library(bigsnpr); library(Matrix)})
set.seed(1)
folder <- commandArgs(trailingOnly = TRUE)[1]
meta <- scan(file.path(folder, "meta.txt"), quiet = TRUE)
p <- meta[1]; n <- meta[2]; n_part <- meta[3]; cores <- meta[4]; chunk <- meta[5]
read <- function(name, what, size) { con <- file(file.path(folder, name), "rb"); on.exit(close(con)); readBin(con, what, n = file.size(file.path(folder, name)) / size, size = size, endian = "little") }
sumstats <- function(name, count) { values <- matrix(read(name, "double", 8), ncol = 2, byrow = TRUE); data.frame(beta = values[, 1], beta_se = values[, 2], n_eff = count) }
write <- function(values, name) { con <- file(file.path(folder, name), "wb"); writeBin(as.double(values), con, size = 8, endian = "little"); close(con) }
full <- sumstats("sumstats.bin", n)
part <- sumstats("sumstats_part.bin", n_part)
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
started <- Sys.time()
ldsc <- with(full, snp_ldsc(ld, length(ld), chi2 = (beta / beta_se)^2, sample_size = n_eff, blocks = NULL))
heritability <- max(ldsc[["h2"]], 0.001)
chains <- snp_ldpred2_auto(corr, full, h2_init = heritability, vec_p_init = seq_log(1e-4, 0.2, length.out = 30),
                           ncores = cores, allow_jump_sign = FALSE, shrink_corr = 0.95, use_MLE = FALSE)
spread <- vapply(chains, function(chain) diff(range(chain$corr_est)), numeric(1))
keep <- which(spread > 0.95 * quantile(spread, 0.95, na.rm = TRUE))
cat("ldpred2: variants", p, "ldsc h2", ldsc[["h2"]], "intercept", ldsc[["int"]], "h2_init", heritability, "chains kept",
    length(keep), "of", length(chains), "h2_est", median(vapply(chains[keep], function(chain) chain$h2_est, numeric(1))),
    "p_est", median(vapply(chains[keep], function(chain) chain$p_est, numeric(1))), "minutes",
    as.numeric(difftime(Sys.time(), started, units = "mins")), "\n")
write(if (length(keep) == 0) rep(NaN, p) else rowMeans(matrix(sapply(chains[keep], function(chain) chain$beta_est), nrow = p)), "beta_auto.bin")
rm(chains)
started <- Sys.time()
lasso <- snp_lassosum2(corr, part, ncores = cores)
grid <- attr(lasso, "grid_param")
cat("lassosum2: grid", ncol(lasso), "minutes", as.numeric(difftime(Sys.time(), started, units = "mins")), "\n")
write(lasso, "beta_lasso.bin")
write.table(grid[, c("lambda", "delta", "num_iter", "sparsity")], file.path(folder, "lasso_grid.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
