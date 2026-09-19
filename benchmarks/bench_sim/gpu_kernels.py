"""bench-sim harness preparation on a GPU: genomic relationship kernels from observed codes, and the PCs.

K_simple (SNV+INDEL records) and K_structural (TR+SV records) are each the mean over measured records of outer
products of columns standardized with training-sample moments, over all N samples. 10 PCs of K_simple (the
training block's top eigenvectors, Nystrom-projected onto test samples) go to <cohort>/pcs_<arm>.npz.

    python -m benchmarks.bench_sim.gpu_kernels --cohort <cohort/chr22>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cupy as cp
import numpy as np
from cupyx.scipy.sparse.linalg import LinearOperator, eigsh

from benchmarks.bench_sim.harness import ARMS
from benchmarks.bench_sim.measurement import measured_records

CODES_PER_DOSAGE = 127
PC_COUNT = 10


def prepare(cohort: Path, block_rows: int, arm: str) -> None:
    """Peak device memory: the two N x N accumulators (2 x 4 N^2 bytes, 20 GB at N = 50,000), then the training
    block of K_simple for the PCs. A dense eigendecomposition of a 40,000-sample block exceeds cuSOLVER's syevd
    workspace limits, so the PCs come from Lanczos (the top PC_COUNT eigenpairs, residual-checked) and the
    baselines solve their ridge systems by Cholesky instead of reusing a full eigenbasis."""
    samples = np.load(cohort / "samples.npz")
    is_test = samples["is_test"]
    train, test = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    measured = measured_records(cohort)
    cls = np.load(cohort / "variants.npz")["cls"]
    observed = np.load(cohort / ARMS[arm][0], mmap_mode="r")
    n_var, size = observed.shape
    train_gpu, test_gpu = cp.asarray(train), cp.asarray(test)
    kernels = {"simple": cp.zeros((size, size), dtype=cp.float32), "structural": cp.zeros((size, size), dtype=cp.float32)}
    counts = {"simple": 0, "structural": 0}
    for first in range(0, n_var, block_rows):
        block = cp.asarray(np.asarray(observed[first:first + block_rows]), dtype=cp.float32) / CODES_PER_DOSAGE
        mean = block[:, train_gpu].mean(axis=1, keepdims=True)
        sd = block[:, train_gpu].std(axis=1, keepdims=True)
        keep = (sd[:, 0] > 0) & cp.asarray(measured[first:first + block_rows])
        standardized = (block[keep] - mean[keep]) / sd[keep]
        block_cls = cp.asarray(cls[first:first + block_rows])[keep]
        for name, members in (("simple", block_cls <= 1), ("structural", block_cls >= 2)):
            if int(members.sum()):
                part = standardized[members]
                kernels[name] += part.T @ part
                counts[name] += int(members.sum())
        if first % (block_rows * 20) == 0:
            print(f"kernel rows {first}/{n_var}", flush=True)
    (cohort / f"kernel_counts_{arm}.json").write_text(json.dumps(counts))
    for name in ("structural", "simple"):
        kernels[name] /= counts[name]
        np.save(cohort / f"kernel_{name}_{arm}.npy", cp.asnumpy(kernels[name]))
    del kernels["structural"]
    cp.get_default_memory_pool().free_all_blocks()
    train_block = kernels["simple"][train_gpu[:, None], train_gpu[None, :]]
    cross_block = kernels["simple"][test_gpu[:, None], train_gpu[None, :]]
    del kernels["simple"]
    cp.get_default_memory_pool().free_all_blocks()
    operator = LinearOperator(train_block.shape, matvec=lambda vector: train_block @ vector, dtype=cp.float32)
    values, vectors = eigsh(operator, k=PC_COUNT, which="LA")
    order = cp.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    residual = cp.linalg.norm(train_block @ vectors - vectors * values, axis=0) / values
    scale = np.sqrt(train.size)
    pcs = np.zeros((size, PC_COUNT))
    pcs[train] = cp.asnumpy(vectors) * scale
    pcs[test] = cp.asnumpy(cross_block @ vectors / values) * scale
    np.savez(cohort / f"pcs_{arm}.npz", pcs=pcs, eigenvalues=cp.asnumpy(values), relative_residual=cp.asnumpy(residual))
    print(f"PCs done; eigenvalues {cp.asnumpy(values).round(3).tolist()}, max relative residual {float(residual.max()):.2e}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--block-rows", type=int, default=2048)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    args = parser.parse_args()
    prepare(Path(args.cohort), args.block_rows, args.arm)


if __name__ == "__main__":
    main()
