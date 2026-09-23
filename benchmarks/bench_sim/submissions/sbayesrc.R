# SBayesRC on the inputs sbayesrc.py wrote to argv[1] (gwas.ma, ld/ with GCTB-format full LD per block and ldm.info,
# and the annotation file named by argv[2], empty for the annotation-free fit). The package's own steps: LDstep3 (eigen
# decomposition of each block, 99.5% variance kept), LDstep4 (snp.info), then sbayesrc() at its defaults, including
# its automatic tuning of the eigen-variance cutoff. Writes sbrc.txt (SNP, A1, BETA, ...).
suppressPackageStartupMessages(library(SBayesRC))
args <- commandArgs(trailingOnly = TRUE)
folder <- args[1]
annotation <- if (length(args) > 1 && nzchar(args[2])) file.path(folder, args[2]) else ""
ld <- file.path(folder, "ld")
blocks <- nrow(data.table::fread(file.path(ld, "ldm.info")))
for (index in seq_len(blocks)) {
  SBayesRC::LDstep3(outDir = ld, blockIndex = index)
  # the full matrix is not read again once its eigen decomposition is written
  file.remove(file.path(ld, paste0("b", index, ".ldm.full.bin")))
}
SBayesRC::LDstep4(outDir = ld)
set.seed(22)
SBayesRC::sbayesrc(mafile = file.path(folder, "gwas.ma"), LDdir = ld, outPrefix = file.path(folder, "sbrc"), annot = annotation)
