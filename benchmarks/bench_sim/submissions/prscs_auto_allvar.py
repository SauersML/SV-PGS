"""bench-sim submission: PRS-CS-auto on every variant class, the all-variant arm of prscs_auto.py.

prscs_auto.py runs PRS-CS on HapMap3-density SNVs, the variant set of its released references, so part of its gap to a
method that sees every record is missing input rather than method. This arm gives it SBayesRC's input: every measured
record of minor allele frequency at least 1% of every class (SNV, INDEL, TR, SV), sbayesrc.common_rows. Everything else
is prscs_auto.py's: PRS-CS's own code at its defaults (a = 1, b = 0.5, phi learned, 1,000 iterations, 500 burn-in,
thinning 5), the covariate-adjusted GWAS, the in-sample LD of the same projected dosages in the same 4 cM blocks,
written in PRS-CS's hdf5 reference format, and the standardized effects put back on the projected dosages' scale,
beta_j = beta_std_j sd(P y) / sd(P x_j). The dosages are projected a block at a time (all 230,000 of them at once would
not fit). PRS-CS's cost per iteration is a Cholesky factorization of each block's matrix, cubic in the block's size, so
at this density a fit takes hours. The prediction's structural part is that of the TR and SV records.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile

import h5py
import numpy as np

from benchmarks.bench_sim.submissions.ldpred2_auto_allvar import CHUNK, Adjuster
from benchmarks.bench_sim.submissions.prscs_auto import PRSCS
from benchmarks.bench_sim.submissions.sbayesrc import BLOCK_CM, Model, common_rows

__all__ = ["Model", "fit"]


def fit(train) -> Model:
    rows = common_rows(train)
    total = rows.shape[0]
    structural = (np.asarray(train.variants["cls"])[rows] >= 2).astype(np.float64)
    count = train.n_samples
    cm = np.asarray(train.variants["cm"], dtype=np.float64)[rows]
    position = np.asarray(train.variants["pos"])[rows]
    block_of = np.floor((cm - cm[0]) / BLOCK_CM).astype(np.int64)
    names = [f"v{row}" for row in rows]
    adjuster = Adjuster(train)
    means, beta, beta_se, squares = (np.empty(total) for _ in range(4))
    with tempfile.TemporaryDirectory(prefix="prscs_allvar_", dir=os.environ.get("TMPDIR"),
                                     ignore_cleanup_errors=True) as directory:
        folder = pathlib.Path(directory)
        reference = folder / "ref_1kg"
        reference.mkdir()
        with h5py.File(reference / "ldblk_1kg_chr22.hdf5", "w") as store:
            for number, block in enumerate(np.unique(block_of), start=1):
                members = np.flatnonzero(block_of == block)
                scaled = []
                for start in range(0, members.shape[0], CHUNK):
                    part = members[start : start + CHUNK]
                    means[part], beta[part], beta_se[part], squares[part], unit = adjuster.block(rows[part])
                    scaled.append(unit)
                scaled = np.vstack(scaled)
                correlation = scaled.astype(np.float64) @ scaled.T.astype(np.float64)
                del scaled
                np.fill_diagonal(correlation, 1.0)
                group = store.create_group(f"blk_{number}")
                group.create_dataset("ldblk", data=correlation)
                group.create_dataset("snplist", data=np.array([names[member].encode() for member in members]))
                del correlation
        frequency = means / 2.0
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
        subprocess.run([sys.executable, "-u", str(PRSCS), f"--ref_dir={reference}", f"--bim_prefix={folder / 'target'}",
                        f"--sst_file={folder / 'sumstats.txt'}", f"--n_gwas={count}", "--chrom=22",
                        f"--out_dir={folder / 'out'}", "--beta_std=True", "--seed=1"],
                       check=True, env=os.environ | threads)
        effects = {}
        with open(folder / "out_pst_eff_a1_b0.5_phiauto_chr22.txt") as handle:
            for line in handle:
                fields = line.split()
                effects[fields[1]] = float(fields[5]) if fields[3] == "A" else -float(fields[5])
    standardized = np.array([effects.get(name, 0.0) for name in names])
    scale = np.sqrt(adjuster.residual @ adjuster.residual / count) / np.sqrt(squares / count)
    return Model(rows, means, standardized * scale, structural)
