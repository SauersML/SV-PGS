"""bench-sim harness preparation on a GPU: genomic relationship kernels from observed codes, and the PCs.

K_simple (SNV+INDEL records) and K_structural (TR+SV records) are each the mean over records of outer
products of columns standardized with training-sample moments, over all N samples. The training blocks of
K_simple and of the record-weighted combination are eigendecomposed for the ridge_inf baseline, and 10 PCs of
K_simple (training eigenvectors, Nystrom-projected onto test samples) go to <cohort>/pcs.npz.

    python -m benchmarks.bench_sim.gpu_kernels --cohort <cohort/chr22>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cupy as cp
import numpy as np

from benchmarks.bench_sim.harness import ARMS

CODES_PER_DOSAGE = 127
PC_COUNT = 10


def prepare(cohort: Path, block_rows: int, arm: str) -> None:
    samples = np.load(cohort / "samples.npz")
    train = np.flatnonzero(~samples["is_test"])
    cls = np.load(cohort / "variants.npz")["cls"]
    observed = np.load(cohort / ARMS[arm][0], mmap_mode="r")
    n_var, size = observed.shape
    train_gpu = cp.asarray(train)
    kernels = {"simple": cp.zeros((size, size), dtype=cp.float32), "structural": cp.zeros((size, size), dtype=cp.float32)}
    counts = {"simple": 0, "structural": 0}
    for first in range(0, n_var, block_rows):
        block = cp.asarray(np.asarray(observed[first:first + block_rows]), dtype=cp.float32) / CODES_PER_DOSAGE
        mean = block[:, train_gpu].mean(axis=1, keepdims=True)
        sd = block[:, train_gpu].std(axis=1, keepdims=True)
        keep = (sd[:, 0] > 0)
        standardized = (block[keep] - mean[keep]) / sd[keep]
        block_cls = cp.asarray(cls[first:first + block_rows])[keep]
        for name, members in (("simple", block_cls <= 1), ("structural", block_cls >= 2)):
            if int(members.sum()):
                part = standardized[members]
                kernels[name] += part.T @ part
                counts[name] += int(members.sum())
        if first % (block_rows * 20) == 0:
            print(f"kernel rows {first}/{n_var}", flush=True)
    for name in kernels:
        kernels[name] /= max(counts[name], 1)
        np.save(cohort / f"kernel_{name}_{arm}.npy", cp.asnumpy(kernels[name]))
    (cohort / f"kernel_counts_{arm}.json").write_text(json.dumps(counts))
    combined = (counts["simple"] * kernels["simple"] + counts["structural"] * kernels["structural"]) / (counts["simple"] + counts["structural"])
    del kernels["structural"]
    for name, matrix in (("simple", kernels["simple"]), ("all", combined)):
        train_block = matrix[train_gpu[:, None], train_gpu[None, :]]
        values, vectors = cp.linalg.eigh(train_block)
        np.save(cohort / f"eig_{name}_{arm}_values.npy", cp.asnumpy(values))
        np.save(cohort / f"eig_{name}_{arm}_vectors.npy", cp.asnumpy(vectors))
        if name == "simple":
            top = cp.argsort(values)[::-1][:PC_COUNT]
            scale = np.sqrt(train.size)
            pcs = np.zeros((size, PC_COUNT))
            pcs[train] = cp.asnumpy(vectors[:, top]) * scale
            test = np.flatnonzero(samples["is_test"])
            test_gpu = cp.asarray(test)
            cross = matrix[test_gpu[:, None], train_gpu[None, :]]
            pcs[test] = cp.asnumpy(cross @ vectors[:, top] / values[top]) * scale
            np.savez(cohort / f"pcs_{arm}.npz", pcs=pcs, eigenvalues=cp.asnumpy(values[top]))
        del train_block, vectors
        print(f"eigendecomposition {name} done", flush=True)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--block-rows", type=int, default=2048)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    args = parser.parse_args()
    prepare(Path(args.cohort), args.block_rows, args.arm)


if __name__ == "__main__":
    main()
