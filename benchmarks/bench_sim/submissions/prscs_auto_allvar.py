"""bench-sim submission: PRS-CS-auto on every variant class, the all-variant arm of prscs_auto.py.

prscs_auto.py runs PRS-CS on HapMap3-density SNVs, the variant set of its released references, so part of its gap to a
method that sees every record is missing input rather than method. This arm gives it SBayesRC's input: every measured
record of minor allele frequency at least 1% of every class (SNV, INDEL, TR, SV), sbayesrc.common_rows. Everything else
is prscs_auto.py's: PRS-CS's own code at its defaults (a = 1, b = 0.5, phi learned, 1,000 iterations, 500 burn-in,
thinning 5, seed 1), the covariate-adjusted GWAS, the in-sample LD of the same projected dosages in the same 4 cM
blocks, and the standardized effects put back on the projected dosages' scale, beta_j = beta_std_j sd(P y) / sd(P x_j).
The prediction's structural part is that of the TR and SV records.

PRS-CS's own functions run in this process, in the order of its PRScs.main: parse_ref, parse_bim and parse_sumstats on
the text inputs written here, then its block preparation (parse_genet.prepare_ldblk: parse_ldblk's selection, sign
flips and symmetrization, split from the hdf5 read) on the LD blocks held in memory, then mcmc_gtb.mcmc. Passing the
blocks in memory instead of through an hdf5 reference file skips writing and re-reading ~25 GB. The patched PRS-CS files
are in benchmarks/bench_sim/prscs_patch (README there): with PRSCS_DEVICE=gpu the LD products here, the
symmetrization and the MCMC's block updates run in float64 on the GPU, validated against the CPU to rounding.
"""
from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import time

import numpy as np

from benchmarks.bench_sim.submissions.ldpred2_auto_allvar import CHUNK, Adjuster
from benchmarks.bench_sim.submissions.prscs_auto import PRSCS
from benchmarks.bench_sim.submissions.sbayesrc import BLOCK_CM, Model, common_rows

__all__ = ["Model", "fit"]
A, B, N_ITER, N_BURNIN, THIN, SEED = 1, 0.5, 1000, 500, 5, 1
"""PRS-CS's defaults (PRScs.py parse_param) and prscs_auto.py's --seed=1."""


def _correlation(scaled: np.ndarray) -> np.ndarray:
    """The float64 product of the unit-norm projected dosages with themselves: on the GPU (CuPy) when PRSCS_DEVICE=gpu,
    the same float64 GEMM as numpy's up to rounding."""
    if os.environ.get("PRSCS_DEVICE") == "gpu":
        import cupy as cp
        device = cp.asarray(scaled).astype(cp.float64)
        product = cp.asnumpy(device @ device.T)
        del device
        cp.get_default_memory_pool().free_all_blocks()
        return product
    return scaled.astype(np.float64) @ scaled.T.astype(np.float64)


def fit(train) -> Model:
    started = time.time()

    def stage(label: str) -> None:
        print(f"prscs_auto_allvar: {label} at {time.time() - started:.1f} s", flush=True)

    sys.path.insert(0, str(PRSCS.parent))
    import mcmc_gtb
    import parse_genet

    rows = common_rows(train)
    stage(f"{rows.shape[0]} common records found")
    total = rows.shape[0]
    structural = (np.asarray(train.variants["cls"])[rows] >= 2).astype(np.float64)
    count = train.n_samples
    cm = np.asarray(train.variants["cm"], dtype=np.float64)[rows]
    position = np.asarray(train.variants["pos"])[rows]
    block_of = np.floor((cm - cm[0]) / BLOCK_CM).astype(np.int64)
    names = [f"v{row}" for row in rows]
    adjuster = Adjuster(train)
    means, beta, beta_se, squares = (np.empty(total) for _ in range(4))
    blocks, snplists = [], []
    for block in np.unique(block_of):
        members = np.flatnonzero(block_of == block)
        scaled = []
        for start in range(0, members.shape[0], CHUNK):
            part = members[start : start + CHUNK]
            means[part], beta[part], beta_se[part], squares[part], unit = adjuster.block(rows[part])
            scaled.append(unit)
        correlation = _correlation(np.vstack(scaled))
        del scaled
        np.fill_diagonal(correlation, 1.0)
        blocks.append(correlation)
        snplists.append([names[member] for member in members])
    stage("GWAS and LD blocks built")
    with tempfile.TemporaryDirectory(prefix="prscs_allvar_", dir=os.environ.get("TMPDIR"),
                                     ignore_cleanup_errors=True) as directory:
        folder = pathlib.Path(directory)
        frequency = means / 2.0
        with open(folder / "snpinfo_1kg_hm3", "w") as handle:
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
        ref_dict = parse_genet.parse_ref(str(folder / "snpinfo_1kg_hm3"), 22)
        vld_dict = parse_genet.parse_bim(str(folder / "target"), 22)
        sst_dict = parse_genet.parse_sumstats(ref_dict, vld_dict, str(folder / "sumstats.txt"), count)
        stage("summary statistics parsed")
        ld_blk, blk_size = parse_genet.prepare_ldblk(blocks, snplists, sst_dict)
        del blocks
        stage("LD blocks prepared")
        mcmc_gtb.mcmc(A, B, None, sst_dict, count, ld_blk, blk_size, N_ITER, N_BURNIN, THIN, 22, str(folder / "out"),
                      "TRUE", "FALSE", "FALSE", SEED)
        del ld_blk
        stage("PRS-CS MCMC done")
        effects = {}
        with open(folder / "out_pst_eff_a1_b0.5_phiauto_chr22.txt") as handle:
            for line in handle:
                fields = line.split()
                effects[fields[1]] = float(fields[5]) if fields[3] == "A" else -float(fields[5])
    standardized = np.array([effects.get(name, 0.0) for name in names])
    scale = np.sqrt(adjuster.residual @ adjuster.residual / count) / np.sqrt(squares / count)
    return Model(rows, means, standardized * scale, structural)
