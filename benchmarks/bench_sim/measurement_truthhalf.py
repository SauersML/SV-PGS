"""bench-sim measurement arm "beagle_truthhalf" (PREREG amendment 8): the Beagle arm with a flagged training
subset observed at its true genotypes, as a long-read truth half would be.

A fixed fraction of each group's training samples (chosen with the public seed) is the truth half. Their codes
are the true genotypes (code = 127 G) at every measured record; every other sample keeps its Beagle-arm codes.
Test samples are never in the truth half. Writes <dir>/observed_beagle_truth.npy and <dir>/truth_half.npy
(one flag per cohort sample).

    python -m benchmarks.bench_sim.measurement_truthhalf --dir <cohort/chr22>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from benchmarks.bench_sim.measurement import CODES_PER_DOSAGE
from benchmarks.bench_sim.records import STREAM_ROWS, measured_records

PUBLIC_SEED = 20260919
# The truth half's share of each group's training samples: a benchmark design choice (PREREG amendment 8),
# fixed before any submission and not derived from any production cohort.
TRUTH_FRACTION = 0.2


def truth_half(samples: np.lib.npyio.NpzFile) -> np.ndarray:
    rng = np.random.default_rng([PUBLIC_SEED, 8])
    group, is_test = samples["group"], samples["is_test"]
    flags = np.zeros(group.size, dtype=bool)
    for index in np.unique(group):
        members = np.flatnonzero((group == index) & ~is_test)
        flags[rng.choice(members, size=int(round(TRUTH_FRACTION * members.size)), replace=False)] = True
    return flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()
    root = Path(args.dir)
    flags = truth_half(np.load(root / "samples.npz"))
    measured = measured_records(root)
    truth = np.load(root / "truth_G.npy", mmap_mode="r")
    observed = np.load(root / "observed_beagle.npy", mmap_mode="r")
    target = np.lib.format.open_memmap(root / "observed_beagle_truth.npy", mode="w+", dtype=np.uint8, shape=observed.shape)
    columns = np.flatnonzero(flags)
    for first in range(0, observed.shape[0], STREAM_ROWS):
        block = np.array(observed[first:first + STREAM_ROWS])
        rows = measured[first:first + STREAM_ROWS]
        true_block = np.asarray(truth[first:first + STREAM_ROWS])
        block[np.ix_(rows, columns)] = true_block[np.ix_(rows, columns)] * np.uint8(CODES_PER_DOSAGE)
        target[first:first + STREAM_ROWS] = block
    target.flush()
    np.save(root / "truth_half.npy", flags)
    print(f"truth half: {flags.sum()} of {int((~np.load(root / 'samples.npz')['is_test']).sum())} training samples", flush=True)


if __name__ == "__main__":
    main()
