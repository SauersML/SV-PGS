"""bench-sim step 3: public annotations and truth LD scores for every kept record (PREREG.md sections 3.6-3.7).

Writes <dir>/annotations.npz:
  in_gene, in_exon       overlap with any RefSeq-curated transcript span / exon
  log_tss_distance       log1p of the base distance from the record span to the nearest TSS
  in_repeat              overlap with a UCSC simpleRepeat interval (the TR class uses it)
  repeat_locus           index of the merged simpleRepeat interval a record overlaps, else -1
  log_sv_length          log1p |length change| (0 for SNVs)
  ld_score               truth LD score: 1 (the record itself) plus the sum of r2 to the other records with
                         MAF >= 1% within +-1 cM, from a fixed 5,000-sample subsample of truth genotypes
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np

PUBLIC_SEED = 20260919
LD_SUBSAMPLE = 5000
LD_WINDOW_CM = 1.0
LD_REFERENCE_MAF = 0.01
BLOCK = 4096


def refseq(path: Path, chrom: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tx_start, tx_end, tss, exon_start, exon_end = [], [], [], [], []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if fields[2] != chrom:
                continue
            start, end = int(fields[4]), int(fields[5])
            tx_start.append(start)
            tx_end.append(end)
            tss.append(start if fields[3] == "+" else end - 1)
            exon_start.extend(int(value) for value in fields[9].rstrip(",").split(","))
            exon_end.extend(int(value) for value in fields[10].rstrip(",").split(","))
    return (np.asarray(tx_start), np.asarray(tx_end), np.sort(np.asarray(tss)), np.asarray(exon_start), np.asarray(exon_end))


def overlaps(starts: np.ndarray, ends: np.ndarray, query_start: np.ndarray, query_end: np.ndarray) -> np.ndarray:
    order = np.argsort(starts)
    starts, ends = starts[order], ends[order]
    running_end = np.maximum.accumulate(ends)
    index = np.searchsorted(starts, query_end, side="left") - 1
    result = np.zeros(query_start.size, dtype=bool)
    valid = index >= 0
    result[valid] = running_end[index[valid]] > query_start[valid]
    return result


def merged_intervals(path: Path, chrom: str) -> tuple[np.ndarray, np.ndarray]:
    intervals = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            fields = line.split("\t")
            if fields[1] == chrom:
                intervals.append((int(fields[2]), int(fields[3])))
    intervals.sort()
    merged: list[list[int]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    array = np.asarray(merged, dtype=np.int64)
    return array[:, 0], array[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--refseq", required=True)
    parser.add_argument("--repeats", required=True)
    args = parser.parse_args()
    root = Path(args.dir)
    variants = np.load(root / "variants.npz")
    pos, end, ref_len, cm = variants["pos"], variants["end"], variants["ref_len"], variants["cm"]
    span_start = pos - 1
    span_end = np.maximum(end, pos + ref_len - 1)
    tx_start, tx_end, tss, exon_start, exon_end = refseq(Path(args.refseq), args.chrom)
    in_gene = overlaps(tx_start, tx_end, span_start, span_end)
    in_exon = overlaps(exon_start, exon_end, span_start, span_end)
    left = np.searchsorted(tss, span_start, side="left")
    distance = np.full(pos.size, np.inf)
    for candidate in (left - 1, left):
        valid = (candidate >= 0) & (candidate < tss.size)
        site = tss[np.clip(candidate, 0, tss.size - 1)]
        gap = np.where(site < span_start, span_start - site, np.where(site >= span_end, site - span_end + 1, 0))
        distance = np.where(valid, np.minimum(distance, gap), distance)
    rep_start, rep_end = merged_intervals(Path(args.repeats), args.chrom)
    locus = np.searchsorted(rep_start, span_end, side="left") - 1
    hit = (locus >= 0) & (rep_end[np.clip(locus, 0, None)] > span_start)
    repeat_locus = np.where(hit, locus, -1)

    truth = np.load(root / "truth_G.npy", mmap_mode="r")
    rng = np.random.default_rng(PUBLIC_SEED)
    subsample = np.sort(rng.choice(truth.shape[1], size=LD_SUBSAMPLE, replace=False))

    sub = np.empty((pos.size, LD_SUBSAMPLE), dtype=np.uint8)
    for first in range(0, pos.size, 20_000):
        sub[first:first + 20_000] = np.asarray(truth[first:first + 20_000])[:, subsample]

    def standardized(rows: np.ndarray) -> np.ndarray:
        block = sub[rows].astype(np.float32)
        block -= block.mean(axis=1, keepdims=True)
        scale = block.std(axis=1, keepdims=True)
        return np.divide(block, scale, out=np.zeros_like(block), where=scale > 0)

    allele_frequency = np.zeros(pos.size)
    for first in range(0, pos.size, 20_000):
        allele_frequency[first:first + 20_000] = sub[first:first + 20_000].mean(axis=1, dtype=np.float64) / 2.0
    reference = np.flatnonzero(np.minimum(allele_frequency, 1 - allele_frequency) >= LD_REFERENCE_MAF)
    ld_score = np.zeros(pos.size)
    count = LD_SUBSAMPLE
    for first in range(0, pos.size, BLOCK):
        rows = np.arange(first, min(first + BLOCK, pos.size))
        lo = np.searchsorted(cm[reference], cm[rows[0]] - LD_WINDOW_CM, side="left")
        hi = np.searchsorted(cm[reference], cm[rows[-1]] + LD_WINDOW_CM, side="right")
        partners = reference[lo:hi]
        query = standardized(rows)
        partner_block = standardized(partners)
        correlation = query @ partner_block.T / count
        within = (np.abs(cm[rows][:, None] - cm[partners][None, :]) <= LD_WINDOW_CM) & (rows[:, None] != partners[None, :])
        ld_score[rows] = 1.0 + np.where(within, correlation * correlation, 0.0).sum(axis=1)
        if first % (BLOCK * 25) == 0:
            print(f"ld score {first}/{pos.size}", flush=True)
    np.savez(
        root / "annotations.npz",
        in_gene=in_gene, in_exon=in_exon, log_tss_distance=np.log1p(distance), in_repeat=variants["in_repeat"],
        repeat_locus=repeat_locus, log_sv_length=np.log1p(np.abs(variants["len_change"])) * (variants["cls"] >= 1),
        ld_score=ld_score, ld_subsample_af=allele_frequency,
    )
    print("annotations written", flush=True)


if __name__ == "__main__":
    main()
