"""bench-sim measurement, option C: Beagle 5 imputation of TR and SV records from called genotypes at simple sites.

The same read model as measurement.py (30x gamma-Poisson depth, gVCF hom-ref block rule) turns each
cohort member's true genotype at SNV and non-TR indel records into a call (the minimum-PL genotype). The
calls are the target, the disjoint panel's phased haplotypes are the reference, and Beagle imputes the
masked TR and SV records. Observed codes: calls for simple sites (code = call * 127), Beagle DS for TR/SV.

Targets are written phased: a correct call keeps the member's true haplotype phase, and a miscalled
heterozygote gets a random phase. Beagle 5 keeps fully phased input and skips statistical phasing, which
at 5,000 targets costs about 7 minutes per iteration and window. The arm therefore carries no phasing
error, making its imputation slightly optimistic (PREREG amendment 4).

Writes <dir>/observed_beagle.npy and <dir>/imputation_beagle.npz (per-record DR2 for TR/SV records,
the call error rate for simple sites, and realized r2 of the observed value to truth).
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from cyvcf2 import VCF

from benchmarks.bench_sim.measurement import (
    BASE_ERROR,
    CODES_PER_DOSAGE,
    PHASED,
    batch_block,
    bgzip_writer,
    encode_milli,
    finish,
    header,
    realized_r2,
    run,
    simulated_pl,
)

PHASED_CALL = np.frombuffer(b"0|0\t0|1\t1|0\t1|1\t", dtype=np.uint8).reshape(4, 4)
ROW_BLOCK = 256


_BLOCK_INPUTS: dict = {}


def phased_call_index(called: np.ndarray, genotype: np.ndarray, true_first: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row of PHASED_CALL (0|0, 0|1, 1|0, 1|1) for each call: a correct call keeps the true phase, a miscalled
    heterozygote gets a random phase, and a miscalled homozygote has only one phase."""
    first_allele = np.where(called == genotype, true_first,
                            np.where(called == 1, rng.integers(0, 2, size=called.shape), called // 2)).astype(np.intp)
    return 2 * first_allele + (called.astype(np.intp) - first_allele)


def beagle_allele(alt: str, row: int) -> str:
    """Beagle rejects two records with the same CHROM, POS, REF and ALT, which symbolic SVs at one position
    (two <INS> of different lengths) share. Symbolic alleles get the record's ID inside the brackets."""
    return f"{alt[:-1]}:v{row}>" if alt.startswith("<") else alt


def _simulate_calls(block_start: int) -> tuple[int, np.ndarray, list[bytes]]:
    """One block of records: read-model calls and their target VCF lines. Runs in a forked worker."""
    truth_block, first_block, errors, sites, simple_rows, seed = (
        _BLOCK_INPUTS[key] for key in ("truth", "first", "errors", "sites", "rows", "seed"))
    stop = block_start + ROW_BLOCK
    rng = np.random.default_rng([*seed, block_start])
    genotype = truth_block[block_start:stop].astype(np.intp)
    pl = simulated_pl(genotype, errors[block_start:stop, None], rng)
    called = np.argmin(pl, axis=-1).astype(np.uint8)
    text = PHASED_CALL[phased_call_index(called, genotype, first_block[block_start:stop], rng)].reshape(called.shape[0], -1)
    text[:, -1] = ord("\n")
    lines = [sites[row].encode() + body.tobytes() for row, body in zip(simple_rows[block_start:stop], text)]
    return block_start, called, lines


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
    sites = [f"{args.chrom}\t{pos[row]}\tv{row}\t{refs[row]}\t{beagle_allele(alts[row], row)}\t.\tPASS\t.\tGT\t" for row in range(n_var)]

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
    first_haplotype = np.load(root / "truth_hapA.npy", mmap_mode="r")
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
        calls = np.zeros((simple_rows.size, last - first), dtype=np.uint8)
        truth_block = batch_block(truth, simple_rows, first, last)
        first_block = batch_block(first_haplotype, simple_rows, first, last)
        target = work / f"target{batch_index}.vcf.gz"
        sink, process = bgzip_writer(target, args.threads)
        process.stdin.write(header(args.chrom, length, names, "GT"))
        _BLOCK_INPUTS.update(
            truth=truth_block, first=first_block, errors=np.array([BASE_ERROR[int(code)] for code in cls[simple_rows]]),
            sites=sites, rows=simple_rows, seed=(20260919, batch_index, int(args.chrom.lstrip("chr")), 5),
        )
        with Pool(args.threads) as pool:
            for block_start, called, lines in pool.imap(_simulate_calls, range(0, simple_rows.size, ROW_BLOCK)):
                calls[block_start:block_start + called.shape[0]] = called
                process.stdin.write(b"".join(lines))
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
        realized = realized_r2(truth, observed)
        np.savez(root / "imputation_beagle.npz", info=dr2, realized_r2=realized)
        for code in range(4):
            members = cls == code
            print(f"class {code}: median realized r2 {np.median(realized[members]):.3f} over {members.sum()} records", flush=True)


if __name__ == "__main__":
    main()
