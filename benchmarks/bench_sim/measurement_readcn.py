"""bench-sim measurement arm "beagle_readcn" (PREREG amendment 10): the Beagle arm plus a read-depth copy-number
channel for deletion and duplication records, simulated from the truth, so that fusing imputed DS with direct read
evidence can be tested before AoU.

For CNV record j (a measured DEL or DUP) and sample i, the true ALT count g gives the copy number c = 2 - g (DEL) or
c = 2 + g (DUP). The reads over the record's span are negative binomial:

    R_ij ~ NB(mean mu_ij, size 1/phi),   mu_ij = s_i * DEPTH * L_j / READ_LENGTH * (c_ij + 2 P_j) / (2 + 2 P_j)

- L_j is the span length. Short records carry few reads, so their evidence is weak without any length cutoff.
- s_i is the sample's depth scale, LogNormal(0, sigma_s).
- P_j counts the paralogous copies whose reads map into the span: distinct UCSC genomicSuperDups partners that overlap
  the span with fracMatch >= identity. Their reads dilute the copy-number signal, as in segmental duplications.

The caller (all that methods see) scores g in {0, 1, 2} with the same negative binomial, but it assumes unique
sequence (P = 0) and a noisy depth scale s_i exp(N(0, tau^2)). It emits phred-scaled genotype likelihoods,
min-normalized and capped at the uint8 maximum.

The generating parameters (sigma_s, phi, identity, tau) come from one of two draws. The public dev draw serves the
dev scenarios; the sealed draw serves the sealed ones, from the sealed master seed's "readcn" domain.

    python -m benchmarks.bench_sim.measurement_readcn --dir <cohort/chr22> --segdups <genomicSuperDups.txt.gz> \
        --draw dev|sealed [--master <sealed/master_seed.txt>]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import gammaln

from benchmarks.bench_sim.records import STREAM_ROWS, measured_records

# 30x whole-genome sequencing with 150 bp reads: the standard short-read design of the imputed cohorts' WGS.
DEPTH = 30.0
READ_LENGTH = 150.0
PL_CAP = 255
PUBLIC_SEED = 20260919
SV = 3
FRACMATCH_COLUMN = 26


def draw_read_parameters(rng: np.random.Generator) -> dict:
    return {
        "sigma_s": float(rng.uniform(0.05, 0.25)),
        "phi": float(np.exp(rng.uniform(np.log(0.01), np.log(0.2)))),
        "identity": float(rng.uniform(0.97, 0.995)),
        "tau": float(rng.uniform(0.0, 0.1)),
    }


def paralog_counts(path: Path, chrom: str, span_start: np.ndarray, span_end: np.ndarray, identity: float) -> np.ndarray:
    """Distinct segmental-duplication partners, at fracMatch >= identity, overlapping each span."""
    pairs = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if fields[1] == chrom and float(fields[FRACMATCH_COLUMN]) >= identity:
                pairs.append((int(fields[2]), int(fields[3]), f"{fields[7]}:{fields[8]}-{fields[9]}"))
    counts = np.zeros(span_start.size, dtype=np.int64)
    if not pairs:
        return counts
    starts = np.array([pair[0] for pair in pairs])
    ends = np.array([pair[1] for pair in pairs])
    partners = np.array([pair[2] for pair in pairs])
    for index in range(span_start.size):
        hit = (starts < span_end[index]) & (ends > span_start[index])
        counts[index] = np.unique(partners[hit]).size
    return counts


def negative_binomial_log_likelihood(reads: np.ndarray, mean: np.ndarray, size: float) -> np.ndarray:
    return (gammaln(reads + size) - gammaln(size) - gammaln(reads + 1.0)
            + size * np.log(size / (size + mean)) + reads * np.log(mean / (size + mean)))


def simulate(genotype: np.ndarray, deletion: np.ndarray, length: np.ndarray, paralogs: np.ndarray, params: dict,
             rng: np.random.Generator) -> np.ndarray:
    """PL uint8 [records, samples, 3] from true ALT counts [records, samples]."""
    samples = genotype.shape[1]
    scale = np.exp(rng.normal(0.0, params["sigma_s"], size=samples))
    believed_scale = scale * np.exp(rng.normal(0.0, params["tau"], size=samples))
    size = 1.0 / params["phi"]
    sign = np.where(deletion, -1.0, 1.0)[:, None]
    reads_per_copy = DEPTH * length[:, None] / READ_LENGTH / 2.0
    copies = 2.0 + sign * genotype
    dilution = paralogs[:, None].astype(np.float64)
    mean = scale[None, :] * reads_per_copy * 2.0 * (copies + 2.0 * dilution) / (2.0 + 2.0 * dilution)
    probability = size / (size + mean)
    reads = rng.negative_binomial(size, probability).astype(np.float64)
    log_likelihood = np.empty(genotype.shape + (3,))
    for alt_count in range(3):
        expected = believed_scale[None, :] * reads_per_copy * (2.0 + sign * alt_count)
        expected = np.maximum(expected, np.finfo(np.float64).tiny)  # a homozygous deletion expects no reads
        log_likelihood[..., alt_count] = negative_binomial_log_likelihood(reads, expected, size)
    phred = -10.0 / np.log(10.0) * (log_likelihood - log_likelihood.max(axis=-1, keepdims=True))
    return np.minimum(np.rint(phred), PL_CAP).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--segdups", required=True)
    parser.add_argument("--chrom", default="chr22")
    parser.add_argument("--draw", choices=("dev", "sealed"), required=True)
    parser.add_argument("--master", help="sealed/master_seed.txt (sealed draw only)")
    parser.add_argument("--out", help="output directory for the sealed draw (default: the cohort directory)")
    args = parser.parse_args()
    root = Path(args.dir)
    if args.draw == "dev":
        seed = [PUBLIC_SEED, 10]
        out = root
    else:
        master = Path(args.master).read_text().strip()
        seed = int.from_bytes(hashlib.sha256(f"{master}:readcn".encode()).digest()[:8], "little")
        out = Path(args.out)
    rng = np.random.default_rng(seed)
    params = draw_read_parameters(rng)
    variants = np.load(root / "variants.npz")
    measured = measured_records(root)
    svtype = variants["svtype"].astype(str)
    kind = np.where(svtype == "", np.where(variants["len_change"] < 0, "DEL", "INS"), svtype)
    rows = np.flatnonzero(measured & (variants["cls"] == SV) & np.isin(kind, ("DEL", "DUP")))
    pos = variants["pos"][rows]
    span_start = pos - 1
    span_end = np.maximum(variants["end"][rows], pos + variants["ref_len"][rows] - 1)
    length = (span_end - span_start).astype(np.float64)
    paralogs = paralog_counts(Path(args.segdups), args.chrom, span_start, span_end, params["identity"])
    truth = np.load(root / "truth_G.npy", mmap_mode="r")
    pl = np.lib.format.open_memmap(out / f"readcn_{args.draw}_pl.npy", mode="w+", dtype=np.uint8, shape=(rows.size, truth.shape[1], 3))
    for first in range(0, rows.size, STREAM_ROWS // 100):
        chunk = slice(first, first + STREAM_ROWS // 100)
        pl[chunk] = simulate(np.asarray(truth[rows[chunk]]).astype(np.float64), kind[rows[chunk]] == "DEL", length[chunk],
                             paralogs[chunk], params, rng)
    pl.flush()
    np.savez(out / f"readcn_{args.draw}.npz", rows=rows, paralogs=paralogs, length=length)
    (out / f"readcn_{args.draw}_params.json").write_text(json.dumps(params, indent=1))
    print(f"{args.draw}: {rows.size} DEL/DUP records, {int((paralogs > 0).sum())} with paralogs", flush=True)


if __name__ == "__main__":
    main()
