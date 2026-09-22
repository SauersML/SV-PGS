"""bench-sim measurement arm "truth": every record observed at its true genotype.

The arm's observed codes are the cohort's true genotypes in the harness's code units (code = genotype * 127),
streamed from truth_G.npy in row blocks; its imputation file reports r^2 = 1 for every record, since the
observed value is the truth. The arm has no measurement loss, so a method's shortfall against the oracle on it
is the method's own (the imputation arms add the loss of imputed TR and SV records on top of it).

    python -m benchmarks.bench_sim.measurement_truth --dir <cohort/chr22>

Writes <dir>/observed_truth.npy and <dir>/imputation_truth.npz, then the kernels and PCs follow from
gpu_kernels with --arm truth, as for every arm.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from benchmarks.bench_sim.records import STREAM_ROWS

CODES_PER_DOSAGE = 127


def write_truth_arm(root: Path) -> None:
    truth = np.load(root / "truth_G.npy", mmap_mode="r")
    n_var, size = truth.shape
    target = root / "observed_truth.npy.partial"
    observed = np.lib.format.open_memmap(target, mode="w+", dtype=np.uint8, shape=(n_var, size))
    for first in range(0, n_var, STREAM_ROWS):
        block = np.asarray(truth[first:first + STREAM_ROWS])
        if block.max() > 2:
            raise SystemExit(f"rows {first}..: a true genotype above 2 (max {int(block.max())})")
        observed[first:first + STREAM_ROWS] = block * np.uint8(CODES_PER_DOSAGE)
        if first % (STREAM_ROWS * 5) == 0:
            print(f"rows {first}/{n_var}", flush=True)
    observed.flush()
    del observed
    target.rename(root / "observed_truth.npy")
    np.savez(root / "imputation_truth.npz", info=np.ones(n_var), realized_r2=np.ones(n_var))
    print(f"truth arm written: {n_var} records x {size} samples", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()
    write_truth_arm(Path(args.dir))


if __name__ == "__main__":
    main()
