"""bench-sim submission: LDpred2-auto (bigsnpr), the genome-wide summary-statistic competitor, at its published settings.

As the competitor workflow runs it (benchmarks/compete/competitors.R, fit_ldpred2_auto; the bigsnpr tutorial): marginal
effects from the training people, in-sample LD in snp_cor's default band (500 neighbouring variants), 30 chains with
vec_p_init = seq_log(1e-4, 0.2, 30), allow_jump_sign = FALSE, shrink_corr = 0.95, h2_init from LD score regression, and
the tutorial's chain filter. LDpred2 is run at HapMap3 density, not on every sequenced record: the SNVs of minor allele
frequency at least 1% (the HapMap3 set's floor), thinned evenly in genetic distance to HapMap3's density on chr22
(about 16,000 of its 1.1 million variants genome-wide). The phenotype is residualized on
[1, covariates] before the marginal regressions, and the test prediction is the fitted per-allele effects on the test
dosages. Quantitative traits (a binary one is fitted as its 0/1 values, the liability-scale linear approximation).
Needs R with bigsnpr (module R/4.4.2-openblas-rocky8 and R_LIBS_USER at the compete library)."""
from __future__ import annotations

import os
import pathlib
import subprocess
import tempfile

import numpy as np

HAPMAP3_CHR22 = 16000
"""HapMap3's variant count on chr22 (about 16,000 of its 1.1 million genome-wide): the density LDpred2 is run at."""
MINOR_ALLELE_FLOOR = 0.01
"""HapMap3's minor allele frequency floor."""
BAND = 500
"""snp_cor's default window: 500 neighbouring variants."""
CHUNK = 4096
_SCRIPT = pathlib.Path(__file__).with_name("ldpred2_auto.R")


def _dosages(source, rows: np.ndarray) -> np.ndarray:
    return np.asarray(source.codes(rows), dtype=np.float32) / np.float32(127.0)


class Model:
    def __init__(self, rows: np.ndarray, means: np.ndarray, beta: np.ndarray, structural: np.ndarray) -> None:
        self.rows, self.means, self.beta, self.structural = rows, means, beta, structural

    def score(self, test):
        total = np.zeros(test.covariates.shape[0])
        for start in range(0, self.rows.shape[0], CHUNK):
            block = slice(start, start + CHUNK)
            dosages = _dosages(test, self.rows[block]).astype(np.float64)
            total += (dosages - self.means[block][:, None]).T @ self.beta[block]
        # LDpred2 here holds SNVs only: no part of its prediction is carried by TR or SV records
        return {"total": total, "structural": np.zeros_like(total)}


def fit(train) -> Model:
    variants = train.variants
    snv = np.flatnonzero(np.asarray(variants["cls"]) == 0)
    # minor allele frequencies of the SNVs from the training codes
    frequency = np.empty(snv.shape[0])
    for start in range(0, snv.shape[0], CHUNK):
        frequency[start : start + CHUNK] = _dosages(train, snv[start : start + CHUNK]).mean(axis=1) / 2.0
    common = snv[np.minimum(frequency, 1.0 - frequency) >= MINOR_ALLELE_FLOOR]
    cm = np.asarray(variants["cm"], dtype=np.float64)[common]
    order = np.argsort(cm, kind="stable")
    common, cm = common[order], cm[order]
    # evenly spaced in genetic distance: the first common SNV at or after each of HAPMAP3_CHR22 equally spaced positions
    targets = np.linspace(cm[0], cm[-1], min(HAPMAP3_CHR22, common.shape[0]))
    chosen = np.unique(np.minimum(np.searchsorted(cm, targets), common.shape[0] - 1))
    rows = np.sort(common[chosen])
    genotypes = np.vstack([_dosages(train, rows[start : start + CHUNK]) for start in range(0, rows.shape[0], CHUNK)])
    means = genotypes.mean(axis=1, dtype=np.float64)
    centred = genotypes - means[:, None].astype(np.float32)
    del genotypes
    count = centred.shape[1]
    design = np.column_stack([np.ones(count), np.asarray(train.covariates, dtype=np.float64)])
    basis, _ = np.linalg.qr(design)
    phenotype = np.asarray(train.phenotype, dtype=np.float64)
    residual = phenotype - basis @ (basis.T @ phenotype)
    # big_univLinReg's marginal regressions of the residualized phenotype on each centred dosage
    squares = np.einsum("ij,ij->i", centred, centred, dtype=np.float64)
    cross = centred.astype(np.float64) @ residual
    beta = cross / squares
    sse = residual @ residual - beta * cross
    beta_se = np.sqrt(sse / ((count - 2) * squares))
    # in-sample correlations with the next BAND variants (snp_cor's band), as triplets
    norms = np.sqrt(squares).astype(np.float32)
    scaled = centred / norms[:, None]
    del centred
    first, second, values = [], [], []
    total = rows.shape[0]
    for start in range(0, total, CHUNK):
        stop = min(start + CHUNK, total)
        reach = min(stop + BAND, total)
        block = scaled[start:stop] @ scaled[start:reach].T
        i, j = np.nonzero(np.abs(np.subtract.outer(np.arange(start, stop), np.arange(start, reach))) <= BAND)
        keep = (start + i) <= (start + j)
        first.append(start + i[keep]); second.append(start + j[keep]); values.append(block[i[keep], j[keep]])
    with tempfile.TemporaryDirectory(prefix="ldpred2_") as directory:
        folder = pathlib.Path(directory)
        np.column_stack([beta, beta_se]).astype("<f8").tofile(folder / "sumstats.bin")
        np.concatenate(first).astype("<i4").tofile(folder / "ld_i.bin")
        np.concatenate(second).astype("<i4").tofile(folder / "ld_j.bin")
        np.concatenate(values).astype("<f8").tofile(folder / "ld_x.bin")
        (folder / "meta.txt").write_text(f"{total} {count} {int(train.cores)}\n")
        # bigstatsr refuses two levels of parallelism: the chains run on the cores, each with single-threaded BLAS
        single = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
        subprocess.run(["Rscript", str(_SCRIPT), str(folder)], check=True, env=os.environ | single)
        effects = np.fromfile(folder / "beta.bin", dtype="<f8")
    return Model(rows, means, effects, np.zeros(0))
