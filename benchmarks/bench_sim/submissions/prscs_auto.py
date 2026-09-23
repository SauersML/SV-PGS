"""bench-sim submission: PRS-CS-auto (Ge et al. 2019, Nature Communications), the continuous-shrinkage summary-statistic
method, run through its own code (github.com/getian107/PRScs) at its defaults: the Strawderman-Berger prior (a = 1,
b = 0.5), the global shrinkage phi learned from the data (the "auto" version: no --phi), 1,000 MCMC iterations with
500 burn-in and thinning 5.

Inputs, in PRS-CS's own reference formats (an ldblk_1kg_chr22.hdf5 of per-block LD matrices and snplists, and the
snpinfo_1kg_hm3 table), built like ldpred2_auto.py's: the HapMap3-density SNVs (PRS-CS's released references are
HapMap3), the GWAS with covariates (Frisch-Waugh-Lovell), and the in-sample LD of the same covariate-projected dosages
of the training people. PRS-CS treats its LD blocks as independent; its released references use LDetect's blocks,
which have no counterpart on this cohort's genetic map, so the blocks are 4 cM windows of genetic position (the width of
SBayesRC's default block file; LDetect's EUR blocks on chr22 average a comparable 1.5 Mb). The summary statistics carry
BETA and SE, so PRS-CS standardizes each as b / (se sqrt(n)); its standardized posterior effects are returned
(--beta_std=True) and put on the per-dosage scale here with the projected dosages' standard deviations,
beta_j = beta_std_j sd(P y) / sd(P x_j), the scale the statistics were computed on (PRS-CS's own unstandardizing uses
2 p (1 - p), the raw dosage variance, which the covariate projection shrinks). Quantitative traits. Needs PRS-CS and a
harness Python with h5py (the code at PRSCS below).
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile

import h5py
import numpy as np

from benchmarks.bench_sim.submissions.ldpred2_auto import (Model, covariate_adjusted, hapmap3_density_rows,
                                                           marginal_statistics)
from benchmarks.bench_sim.submissions.sbayesrc import BLOCK_CM

__all__ = ["Model", "fit"]
_HOME = pathlib.Path("/scratch.global/sauer354/svpgs-team/agents/baselines-genome")
PRSCS = _HOME / "soft" / "PRScs-master" / "PRScs.py"


def fit(train) -> Model:
    rows = hapmap3_density_rows(train)
    projected, residual, means, k = covariate_adjusted(train, rows)
    beta, beta_se, squares = marginal_statistics(projected, residual, k)
    count = projected.shape[1]
    cm = np.asarray(train.variants["cm"], dtype=np.float64)[rows]
    position = np.asarray(train.variants["pos"])[rows]
    block_of = np.floor((cm - cm[0]) / BLOCK_CM).astype(np.int64)
    names = [f"v{row}" for row in rows]
    frequency = means / 2.0
    with tempfile.TemporaryDirectory(prefix="prscs_", dir=os.environ.get("TMPDIR")) as directory:
        folder = pathlib.Path(directory)
        reference = folder / "ref_1kg"
        reference.mkdir()
        with h5py.File(reference / "ldblk_1kg_chr22.hdf5", "w") as store:
            for number, block in enumerate(np.unique(block_of), start=1):
                members = np.flatnonzero(block_of == block)
                scaled = projected[members] / np.sqrt(squares[members]).astype(np.float32)[:, None]
                correlation = (scaled.astype(np.float64) @ scaled.T.astype(np.float64))
                np.fill_diagonal(correlation, 1.0)
                group = store.create_group(f"blk_{number}")
                group.create_dataset("ldblk", data=correlation)
                group.create_dataset("snplist", data=np.array([names[member].encode() for member in members]))
        del projected
        with open(reference / "snpinfo_1kg_hm3", "w") as handle:
            handle.write("CHR\tSNP\tBP\tA1\tA2\tMAF\n")
            for index, name in enumerate(names):
                handle.write(f"22\t{name}\t{position[index]}\tA\tC\t{frequency[index]:.8f}\n")
        with open(folder / "target.bim", "w") as handle:
            for index, name in enumerate(names):
                handle.write(f"22\t{name}\t0\t{position[index]}\tA\tC\n")
        with open(folder / "sumstats.txt", "w") as handle:
            handle.write("SNP\tA1\tA2\tBETA\tSE\n")
            for index, name in enumerate(names):
                handle.write(f"{name}\tA\tC\t{beta[index]:.10g}\t{beta_se[index]:.10g}\n")
        threads = {name: str(int(train.cores)) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
        subprocess.run([sys.executable, str(PRSCS), f"--ref_dir={reference}", f"--bim_prefix={folder / 'target'}",
                        f"--sst_file={folder / 'sumstats.txt'}", f"--n_gwas={count}", "--chrom=22",
                        f"--out_dir={folder / 'out'}", "--beta_std=True", "--seed=1"],
                       check=True, env=os.environ | threads)
        effects = {}
        with open(folder / "out_pst_eff_a1_b0.5_phiauto_chr22.txt") as handle:
            for line in handle:
                fields = line.split()
                effects[fields[1]] = float(fields[5]) if fields[3] == "A" else -float(fields[5])
    standardized = np.array([effects.get(name, 0.0) for name in names])
    # per unit of the projected dosage, in phenotype units: sd(P y) / sd(P x_j)
    scale = np.sqrt(residual @ residual / count) / np.sqrt(squares / count)
    return Model(rows, means, standardized * scale, np.zeros(0))
