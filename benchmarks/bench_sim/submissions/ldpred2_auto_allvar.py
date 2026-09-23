"""bench-sim submission: LDpred2-auto on every variant class, the all-variant arm of ldpred2_auto.py.

ldpred2_auto.py runs LDpred2 as published practice does, on HapMap3-density SNVs, so part of its gap to a method that
sees every record is missing input rather than method. This arm gives it the input SBayesRC gets: every measured record
of minor allele frequency at least 1% of every class (SNV, INDEL, TR, SV), sbayesrc.common_rows. Everything else is
ldpred2_auto.py's: the covariate-adjusted GWAS (Frisch-Waugh-Lovell), the in-sample LD of the same projected dosages in
the tutorial's 3 cM window, h2_init from LD score regression floored at 0.001, 30 chains on
vec_p_init = seq_log(1e-4, 0.2, 30), allow_jump_sign = FALSE, shrink_corr = 0.95, use_MLE = FALSE (kept as in the
SNV arm so the two arms differ only in their variants; the extended documentation reserves TRUE for GWAS of large N,
and N here is about 40,000) and the tutorial's corr_est chain filter.

What changes is only how the band is built: about 230,000 variants in a 3 cM window hold a few billion correlations,
more than one dgCMatrix can index and more than the dosages' projections fit in memory at once. The projected dosages
are computed a row block at a time and kept only while a column block's window reaches them; each column's in-window
correlations (a contiguous run of rows, the variants being sorted by genetic position) are streamed to one file, and
ldpred2_auto_allvar.R builds the compact SFBM from it a column block at a time with the SFBM's add_columns, the route
bigsnpr's documentation uses to assemble a genome-wide matrix. LD scores (the column sums of r^2 the SNV arm's R script
computes) are summed here as the band is written. The prediction's structural part is that of the TR and SV records.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import tempfile

import numpy as np

from benchmarks.bench_sim.submissions.ldpred2_auto import WINDOW_CM
from benchmarks.bench_sim.submissions.sbayesrc import Model, common_rows

__all__ = ["Model", "fit"]
CHUNK = 4096
_SCRIPT = pathlib.Path(__file__).with_name("ldpred2_auto_allvar.R")


def _dosages(source, rows: np.ndarray) -> np.ndarray:
    return np.asarray(source.codes(rows), dtype=np.float32) / np.float32(127.0)


class Adjuster:
    """The projection P off C = [1, covariates] and the GWAS on P X, P y (as ldpred2_auto.marginal_statistics)."""

    def __init__(self, train) -> None:
        count = train.n_samples
        design = np.column_stack([np.ones(count), np.asarray(train.covariates, dtype=np.float64)])
        basis, _ = np.linalg.qr(design)
        self.basis32 = basis.astype(np.float32)
        phenotype = np.asarray(train.phenotype, dtype=np.float64)
        self.residual = phenotype - basis @ (basis.T @ phenotype)
        self.degrees = count - basis.shape[1] - 1
        self.train = train

    def block(self, rows: np.ndarray):
        """Dosage means, b, se, |P x|^2 and the unit-norm projected dosages (float32) of the records `rows`."""
        dosages = _dosages(self.train, rows)
        means = dosages.mean(axis=1, dtype=np.float64)
        projected = dosages - (dosages @ self.basis32) @ self.basis32.T
        del dosages
        squares = np.einsum("ij,ij->i", projected, projected, dtype=np.float64)
        cross = projected.astype(np.float64) @ self.residual
        beta = cross / squares
        se = np.sqrt((self.residual @ self.residual - beta * cross) / (self.degrees * squares))
        projected /= np.sqrt(squares).astype(np.float32)[:, None]
        return means, beta, se, squares, projected


def write_band(train, rows: np.ndarray, folder: pathlib.Path):
    """Streams the 3 cM band of the projected dosages' correlations to folder/ld_values.bin, column by column.

    Column j holds rows lo[j] .. hi[j] - 1 (float32, diagonal exactly 1). Returns means, b, se, lo, hi, LD scores."""
    total = rows.shape[0]
    cm = np.asarray(train.variants["cm"], dtype=np.float64)[rows]
    lo = np.searchsorted(cm, cm - WINDOW_CM, side="left")
    hi = np.searchsorted(cm, cm + WINDOW_CM, side="right")
    adjuster = Adjuster(train)
    means, beta, se, scores = (np.empty(total) for _ in range(4))
    cache: dict[int, np.ndarray] = {}
    with open(folder / "ld_values.bin", "wb") as handle:
        for c0 in range(0, total, CHUNK):
            c1 = min(c0 + CHUNK, total)
            r0, r1 = int(lo[c0]), int(hi[c1 - 1])
            for index in [key for key in cache if (key + 1) * CHUNK <= r0]:
                del cache[index]
            for index in range(r0 // CHUNK, (r1 - 1) // CHUNK + 1):
                if index not in cache:
                    part = slice(index * CHUNK, min((index + 1) * CHUNK, total))
                    means[part], beta[part], se[part], _, cache[index] = adjuster.block(rows[part])
            columns = cache[c0 // CHUNK]
            # products[k, i - r0] = r(c0 + k, i) for i in [r0, r1)
            products = np.hstack([columns @ cache[index][max(r0 - index * CHUNK, 0) : r1 - index * CHUNK].T
                                  for index in range(r0 // CHUNK, (r1 - 1) // CHUNK + 1)])
            within = np.arange(c1 - c0)
            products[within, np.arange(c0, c1) - r0] = 1.0
            position = np.arange(r0, r1)[None, :]
            inside = (position >= lo[c0:c1, None]) & (position < hi[c0:c1, None])
            values = products[inside]
            handle.write(values.astype("<f4").tobytes())
            scores[c0:c1] = np.where(inside, products.astype(np.float64) ** 2, 0.0).sum(axis=1)
            del products, inside, values
    return means, beta, se, lo, hi, scores


def fit(train) -> Model:
    rows = common_rows(train)
    structural = (np.asarray(train.variants["cls"])[rows] >= 2).astype(np.float64)
    # on a network scratch TMPDIR, files the R session's forked workers still hold open linger as .nfs entries for a
    # while after it exits; a directory that cannot be removed yet must not discard a finished fit
    with tempfile.TemporaryDirectory(prefix="ldpred2_allvar_", dir=os.environ.get("TMPDIR"),
                                     ignore_cleanup_errors=True) as directory:
        folder = pathlib.Path(directory)
        means, beta, se, lo, hi, scores = write_band(train, rows, folder)
        np.column_stack([beta, se]).astype("<f8").tofile(folder / "sumstats.bin")
        lo.astype("<i4").tofile(folder / "ld_lo.bin")
        hi.astype("<i4").tofile(folder / "ld_hi.bin")
        scores.astype("<f8").tofile(folder / "ld_scores.bin")
        (folder / "meta.txt").write_text(f"{rows.shape[0]} {train.n_samples} {int(train.cores)} {CHUNK}\n")
        single = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
        subprocess.run(["Rscript", str(_SCRIPT), str(folder)], check=True, env=os.environ | single)
        effects = np.fromfile(folder / "beta.bin", dtype="<f8")
    return Model(rows, means, effects, structural)
