"""bench-sim measurement, option C: Beagle 5 imputation of TR and SV records from called genotypes at simple sites.

The same read model as measurement.py (30x gamma-Poisson depth, gVCF hom-ref block rule) turns each
cohort member's true genotype at SNV and non-TR indel records into a call (the minimum-PL genotype). The
calls are the target, the disjoint panel's phased haplotypes are the reference, and Beagle imputes the
masked TR and SV records. Observed codes: calls for simple sites (code = call * 127), Beagle DS for TR/SV.

Writes <dir>/observed_beagle.npy and <dir>/imputation_beagle.npz (per-record DR2 for TR/SV records,
the call error rate for simple sites, and realized r2 of the observed value to truth).
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
from cyvcf2 import VCF

from benchmarks.bench_sim.measurement import (
    BASE_ERROR,
    CODES_PER_DOSAGE,
    PHASED,
    bgzip_writer,
    encode_milli,
    finish,
    header,
    run,
    simulated_pl,
)

UNPHASED = np.frombuffer(b"0/0\t0/1\t1/1\t", dtype=np.uint8).reshape(3, 4)
STREAM_ROWS = 20_000


def batch_block(truth: np.ndarray, rows: np.ndarray, first: int, last: int) -> np.ndarray:
    """truth[rows, first:last], read as sequential row blocks: strided memmap reads on a network filesystem are random I/O."""
    block = np.empty((rows.size, last - first), dtype=np.uint8)
    for start in range(0, truth.shape[0], STREAM_ROWS):
        chunk = np.asarray(truth[start:start + STREAM_ROWS])[:, first:last]
        selected = (rows >= start) & (rows < start + STREAM_ROWS)
        block[selected] = chunk[rows[selected] - start]
    return block


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--beagle", required=True, help="Beagle 5 jar")
    parser.add_argument("--map", required=True, help="PLINK-format GRCh38 map for the chromosome")
    parser.add_argument("--batch", type=int, default=5000)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-gb", type=int, default=40)
    parser.add_argument("--only-batch", type=int, default=-1)
    args = parser.parse_args()

    root = Path(args.dir)
    work = root / "beagle"
    work.mkdir(exist_ok=True)
    variants = np.load(root / "variants.npz")
    cls, pos, refs, alts = variants["cls"], variants["pos"], variants["refs"], variants["alts"]
    n_var = cls.size
    simple_rows = np.flatnonzero(cls <= 1)
    masked_rows = np.flatnonzero(cls >= 2)
    length = int(pos.max()) + 1
    sites = [f"{args.chrom}\t{pos[row]}\tv{row}\t{refs[row]}\t{alts[row]}\t.\tPASS\t.\tGT\t" for row in range(n_var)]

    reference = work / "reference.vcf.gz"
    if not reference.exists():
        panel = np.load(root / "panel_haps.npy", mmap_mode="r")
        samples = (root / "panel_samples.txt").read_text().split()
        sink, process = bgzip_writer(reference, args.threads)
        process.stdin.write(header(args.chrom, length, samples, "GT"))
        for row in range(n_var):
            haplotypes = np.asarray(panel[row])
            body = PHASED[2 * haplotypes[0::2].astype(np.intp) + haplotypes[1::2]].reshape(-1).copy()
            body[-1] = ord("\n")
            process.stdin.write(sites[row].encode() + body.tobytes())
        finish(sink, process, reference)

    truth = np.load(root / "truth_G.npy", mmap_mode="r")
    size = truth.shape[1]
    starts = list(range(0, size, args.batch))
    for batch_index, first in enumerate(starts):
        if args.only_batch >= 0 and batch_index != args.only_batch:
            continue
        flag = work / f"batch{batch_index}.done"
        if flag.exists():
            continue
        last = min(first + args.batch, size)
        names = [f"s{index}" for index in range(first, last)]
        rng = np.random.default_rng([20260919, batch_index, int(args.chrom.lstrip("chr")), 5])
        calls = np.zeros((simple_rows.size, last - first), dtype=np.uint8)
        truth_block = batch_block(truth, simple_rows, first, last)
        target = work / f"target{batch_index}.vcf.gz"
        sink, process = bgzip_writer(target, args.threads)
        process.stdin.write(header(args.chrom, length, names, "GT"))
        for offset, row in enumerate(simple_rows):
            pl = simulated_pl(truth_block[offset].astype(np.intp), BASE_ERROR[int(cls[row])], rng)
            called = np.argmin(pl, axis=-1)
            calls[offset] = called
            body = UNPHASED[called].reshape(-1).copy()
            body[-1] = ord("\n")
            process.stdin.write(sites[row].encode() + body.tobytes())
        finish(sink, process, target)
        out_prefix = work / f"imputed{batch_index}"
        run(["java", f"-Xmx{args.memory_gb}g", "-jar", args.beagle, f"ref={reference}", f"gt={target}",
             f"map={args.map}", f"out={out_prefix}", "impute=true", f"nthreads={args.threads}"])
        codes = np.zeros((n_var, last - first), dtype=np.uint8)
        codes[simple_rows] = calls * np.uint8(CODES_PER_DOSAGE)
        dr2 = np.full(n_var, np.nan)
        seen = np.zeros(n_var, dtype=bool)
        reader = VCF(f"{out_prefix}.vcf.gz")
        for record in reader:
            row = int(record.ID[1:])
            if cls[row] < 2:
                continue
            dosage = record.format("DS")[:, 0]
            codes[row] = encode_milli(np.clip(np.rint(dosage * 1000.0), 0, 2000).astype(np.int64))
            dr2[row] = float(record.INFO["DR2"])
            seen[row] = True
        reader.close()
        if not seen[masked_rows].all():
            raise SystemExit(f"batch {batch_index}: {int((~seen[masked_rows]).sum())} TR/SV records missing from the Beagle output")
        np.save(work / f"codes{batch_index}.npy", codes)
        np.save(work / f"dr2_{batch_index}.npy", dr2)
        target.unlink()
        flag.touch()
        print(f"batch {batch_index} imputed ({last}/{size})", flush=True)

    if all((work / f"batch{index}.done").exists() for index in range(len(starts))):
        observed = np.concatenate([np.load(work / f"codes{index}.npy") for index in range(len(starts))], axis=1)
        np.save(root / "observed_beagle.npy", observed)
        batch_sizes = np.array([min(first + args.batch, size) - first for first in starts], dtype=np.float64)
        dr2 = sum(np.load(work / f"dr2_{index}.npy") * batch_sizes[index] for index in range(len(starts))) / batch_sizes.sum()
        realized = np.zeros(n_var)
        for first in range(0, n_var, 5000):
            true_block = np.asarray(truth[first:first + 5000]).astype(np.float64)
            observed_block = observed[first:first + 5000].astype(np.float64) / CODES_PER_DOSAGE
            true_block -= true_block.mean(axis=1, keepdims=True)
            observed_block -= observed_block.mean(axis=1, keepdims=True)
            numerator = (true_block * observed_block).sum(axis=1) ** 2
            denominator = (true_block ** 2).sum(axis=1) * (observed_block ** 2).sum(axis=1)
            realized[first:first + 5000] = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        np.savez(root / "imputation_beagle.npz", info=dr2, realized_r2=realized)
        for code in range(4):
            members = cls == code
            print(f"class {code}: median realized r2 {np.median(realized[members]):.3f} over {members.sum()} records", flush=True)


if __name__ == "__main__":
    main()
