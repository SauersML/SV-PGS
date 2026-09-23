"""bench-sim submission: LDpred2-auto (bigsnpr), the genome-wide summary-statistic competitor, at its published settings.

As the bigsnpr tutorial and its extended documentation run it (privefl.github.io/bigsnpr/articles/LDpred2.html,
privefl.github.io/bigsnpr-extdoc/polygenic-scores-pgs.html): marginal effects from the training people, LD in a 3 cM
window of genetic distance (snp_cor(size = 3 / 1000, infos.pos = genetic position)), h2_init from LD score regression
(floored at 0.001, as the authors' own analysis code does), 30 chains with vec_p_init = seq_log(1e-4, 0.2, 30), allow_jump_sign = FALSE, shrink_corr = 0.95, use_MLE = FALSE, and
the tutorial's chain filter on corr_est. LDpred2 is run at HapMap3 density, not on every sequenced record: the SNVs of
minor allele frequency at least 1% (the HapMap3 set's floor), thinned evenly in genetic distance to HapMap3's density
on chr22 (about 16,000 of its 1.1 million variants genome-wide). Quantitative traits (a binary one is fitted as its 0/1
values, the liability-scale linear approximation). Needs R with bigsnpr (module R/4.4.2-openblas-rocky8 and R_LIBS_USER
at the compete library).

The GWAS is big_univLinReg's with the covariates: by Frisch-Waugh-Lovell, each variant's marginal effect adjusted for
C = [1, covariates] is b_j = <P x_j, P y> / |P x_j|^2 with P = I - Q Q' (Q an orthonormal basis of C), and its standard
error has n - k - 1 degrees of freedom (k = C's columns). The LD these summary statistics imply is the correlation of
the same projected dosages: E[b_j] |P x_j| = sum_l corr(P x_j, P x_l) beta_l |P x_l|, so the in-sample LD matrix is
built from P X, not from the raw dosages. The distinction is not cosmetic in this cohort: its training people are five
ancestry groups, and raw in-sample correlations carry the ancestry structure the ten PCs remove from the phenotype
(allele-frequency differences correlate distant variants); a banded raw matrix keeps the near part of that structure,
drops the far part, and no longer matches the statistics it is paired with.
"""
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
WINDOW_CM = 3.0
"""The tutorial's LD window: snp_cor(size = 3 / 1000) on genetic positions, i.e. variants within 3 cM."""
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


def hapmap3_density_rows(train) -> np.ndarray:
    """The SNVs of minor allele frequency >= 1%, thinned evenly in genetic distance to HapMap3's chr22 count."""
    variants = train.variants
    snv = np.flatnonzero(np.asarray(variants["cls"]) == 0)
    frequency = np.empty(snv.shape[0])
    for start in range(0, snv.shape[0], CHUNK):
        frequency[start : start + CHUNK] = _dosages(train, snv[start : start + CHUNK]).mean(axis=1) / 2.0
    common = snv[np.minimum(frequency, 1.0 - frequency) >= MINOR_ALLELE_FLOOR]
    cm = np.asarray(variants["cm"], dtype=np.float64)[common]
    order = np.argsort(cm, kind="stable")
    common, cm = common[order], cm[order]
    # the first common SNV at or after each of HAPMAP3_CHR22 equally spaced genetic positions
    targets = np.linspace(cm[0], cm[-1], min(HAPMAP3_CHR22, common.shape[0]))
    chosen = np.unique(np.minimum(np.searchsorted(cm, targets), common.shape[0] - 1))
    return np.sort(common[chosen])


def covariate_adjusted(train, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """P X (float32, [rows, n]) and P y with P the projection off [1, covariates], the dosage means, and k = rank C."""
    count = train.n_samples
    design = np.column_stack([np.ones(count), np.asarray(train.covariates, dtype=np.float64)])
    basis, _ = np.linalg.qr(design)
    phenotype = np.asarray(train.phenotype, dtype=np.float64)
    residual = phenotype - basis @ (basis.T @ phenotype)
    projected = np.empty((rows.shape[0], count), dtype=np.float32)
    means = np.empty(rows.shape[0])
    basis32 = basis.astype(np.float32)
    for start in range(0, rows.shape[0], CHUNK):
        block = _dosages(train, rows[start : start + CHUNK])
        means[start : start + block.shape[0]] = block.mean(axis=1, dtype=np.float64)
        projected[start : start + block.shape[0]] = block - (block @ basis32) @ basis32.T
    return projected, residual, means, basis.shape[1]


def marginal_statistics(projected: np.ndarray, residual: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """big_univLinReg with covariates: b_j = <P x_j, P y>/|P x_j|^2, se_j^2 = (|P y|^2 - b_j <P x_j, P y>)/((n-k-1)|P x_j|^2)."""
    squares = np.einsum("ij,ij->i", projected, projected, dtype=np.float64)
    cross = projected.astype(np.float64) @ residual
    beta = cross / squares
    sse = residual @ residual - beta * cross
    beta_se = np.sqrt(sse / ((projected.shape[1] - k - 1) * squares))
    return beta, beta_se, squares


def windowed_correlation(projected: np.ndarray, squares: np.ndarray, cm: np.ndarray):
    """Upper-triangle triplets (i, j, r_ij) of the projected dosages' correlations for |cm_i - cm_j| <= WINDOW_CM."""
    scaled = projected / np.sqrt(squares).astype(np.float32)[:, None]
    first, second, values = [], [], []
    total = scaled.shape[0]
    for start in range(0, total, CHUNK):
        stop = min(start + CHUNK, total)
        reach = int(np.searchsorted(cm, cm[stop - 1] + WINDOW_CM, side="right"))
        block = scaled[start:stop] @ scaled[start:reach].T
        i, j = np.nonzero((np.arange(start, stop)[:, None] <= np.arange(start, reach)[None, :])
                          & (np.abs(cm[start:stop, None] - cm[None, start:reach]) <= WINDOW_CM))
        first.append(start + i); second.append(start + j); values.append(block[i, j])
    return np.concatenate(first), np.concatenate(second), np.concatenate(values)


def fit(train) -> Model:
    rows = hapmap3_density_rows(train)
    projected, residual, means, k = covariate_adjusted(train, rows)
    beta, beta_se, squares = marginal_statistics(projected, residual, k)
    cm = np.asarray(train.variants["cm"], dtype=np.float64)[rows]
    first, second, values = windowed_correlation(projected, squares, cm)
    del projected
    with tempfile.TemporaryDirectory(prefix="ldpred2_") as directory:
        folder = pathlib.Path(directory)
        np.column_stack([beta, beta_se]).astype("<f8").tofile(folder / "sumstats.bin")
        first.astype("<i4").tofile(folder / "ld_i.bin")
        second.astype("<i4").tofile(folder / "ld_j.bin")
        values.astype("<f8").tofile(folder / "ld_x.bin")
        (folder / "meta.txt").write_text(f"{rows.shape[0]} {train.n_samples} {int(train.cores)}\n")
        # bigstatsr refuses two levels of parallelism: the chains run on the cores, each with single-threaded BLAS
        single = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
        subprocess.run(["Rscript", str(_SCRIPT), str(folder)], check=True, env=os.environ | single)
        effects = np.fromfile(folder / "beta.bin", dtype="<f8")
    # b is per unit of the covariate-adjusted dosage, which differs from the raw dosage by a function of the
    # covariates only: the score applies it to the centred raw test dosages (the harness adjusts for covariates)
    return Model(rows, means, effects, np.zeros(0))
