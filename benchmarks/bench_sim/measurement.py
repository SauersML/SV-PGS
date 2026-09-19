"""bench-sim step 2 (amendment 1): GLIMPSE2 re-imputation mirroring aou2's measurement process.

The facts, relayed from imputation-4c (process only):
  - Target input is PLs from 30x short-read gVCFs, present only at "simple sites" (SNV and simple indel
    records). Hom-ref blocks carry PL = (0, trunc(GQ/5), capped), at most (0, 19, 30).
  - Every panel record is imputed, SNVs included; SVs and complex records come from the panel alone.
  - GLIMPSE2 with --err-imp 1e-3.
Here the reference is the disjoint public panel founders' phased haplotypes (all kept records). Target PLs
come from a read-depth model on the cohort's true genotypes at SNV and non-TR indel records. DS for every
record is stored as the repo's uint8 code.

Writes <dir>/observed.npy [n_var, N] uint8 and <dir>/imputation.npz (per-record mean GLIMPSE2 INFO,
realized r2 of DS to truth).
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
from cyvcf2 import VCF

CODES_PER_DOSAGE = 127  # sv_pgs.dosage_store: code = round_half_up(milli * 127 / 1000)
PHASED = np.frombuffer(b"0|0\t0|1\t1|0\t1|1\t", dtype=np.uint8).reshape(4, 4)
MEAN_DEPTH = 30.0
DEPTH_SHAPE = 8.0
BASE_ERROR = {0: 0.005, 1: 0.01}
PL_CAP = 999
HOMREF_HOM_ALT_CAP = 30
GQ_CAP = 99
ERR_IMP = "1e-3"
# Records simulated and written per vectorized step.
ROW_BLOCK = 256


def encode_milli(milli: np.ndarray) -> np.ndarray:
    return ((milli.astype(np.uint32) * CODES_PER_DOSAGE + 500) // 1000).astype(np.uint8)


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def header(chrom: str, length: int, samples: list[str], fmt: str) -> bytes:
    lines = ["##fileformat=VCFv4.2", f"##contig=<ID={chrom},length={length}>"]
    if fmt == "GT":
        # GLIMPSE2 requires AC/AN on the reference panel.
        lines.append('##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count in the panel">')
        lines.append('##INFO=<ID=AN,Number=1,Type=Integer,Description="Allele number in the panel">')
        lines.append('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
    else:
        lines.append('##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Phred-scaled genotype likelihoods">')
    lines.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples))
    return ("\n".join(lines) + "\n").encode()


def bgzip_writer(path: Path, threads: int):
    sink = open(path, "wb")
    process = subprocess.Popen(["bgzip", "-@", str(threads), "-c"], stdin=subprocess.PIPE, stdout=sink)
    return sink, process


def finish(sink, process, path: Path) -> None:
    process.stdin.close()
    if process.wait() != 0:
        raise SystemExit(f"bgzip failed for {path}")
    sink.close()
    run(["tabix", "-f", "-p", "vcf", str(path)])


def simulated_pl(genotype: np.ndarray, error, rng: np.random.Generator) -> np.ndarray:
    """PL triplets from a gamma-Poisson read-depth model, with the gVCF hom-ref block rule.

    error is the per-read base error, a scalar or an array broadcastable to genotype (one per record).
    """
    depth = rng.poisson(rng.gamma(DEPTH_SHAPE, MEAN_DEPTH / DEPTH_SHAPE, size=genotype.shape))
    per_read = np.broadcast_to(np.asarray(error, dtype=np.float64), genotype.shape)
    fractions = np.stack([per_read, np.full(genotype.shape, 0.5), 1.0 - per_read], axis=-1)
    alt_reads = rng.binomial(depth, np.take_along_axis(fractions, genotype[..., None], axis=-1)[..., 0])
    ref_reads = depth - alt_reads
    log_likelihood = alt_reads[..., None] * np.log10(fractions) + ref_reads[..., None] * np.log10(1.0 - fractions)
    pl = -10.0 * (log_likelihood - log_likelihood.max(axis=-1, keepdims=True))
    pl = np.minimum(np.rint(pl), PL_CAP).astype(np.int64)
    ordered = np.sort(pl, axis=-1)
    gq = np.minimum(ordered[..., 1], GQ_CAP)
    homref_call = pl[..., 0] == 0
    band = gq // 5
    block = np.stack([np.zeros_like(band), band, np.minimum(2 * band, HOMREF_HOM_ALT_CAP)], axis=-1)
    return np.where(homref_call[..., None], block, pl)


def pl_text(pl: np.ndarray) -> np.ndarray:
    """[n, 3] PLs to fixed-width 'ddd,ddd,ddd\\t' bytes (htslib reads the leading zeros as integers)."""
    digits = np.stack([pl // 100, (pl // 10) % 10, pl % 10], axis=-1) + 48
    out = np.empty((pl.shape[0], 12), dtype=np.uint8)
    out[:, 0:3] = digits[:, 0]
    out[:, 3] = ord(",")
    out[:, 4:7] = digits[:, 1]
    out[:, 7] = ord(",")
    out[:, 8:11] = digits[:, 2]
    out[:, 11] = ord("\t")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--glimpse", required=True, help="directory with GLIMPSE2 static binaries and b38 maps")
    parser.add_argument("--batch", type=int, default=5000)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--only-batch", type=int, default=-1)
    parser.add_argument("--limit-samples", type=int, default=0, help="smoke runs: impute only the first samples")
    args = parser.parse_args()

    root = Path(args.dir)
    tools = Path(args.glimpse)
    work = root / "glimpse"
    work.mkdir(exist_ok=True)
    gmap = tools / f"{args.chrom}.b38.gmap.gz"
    variants = np.load(root / "variants.npz")
    cls, pos, refs, alts = variants["cls"], variants["pos"], variants["refs"], variants["alts"]
    n_var = cls.size
    simple_rows = np.flatnonzero(cls <= 1)
    length = int(pos.max()) + 1
    sites = [f"{args.chrom}\t{pos[row]}\tv{row}\t{refs[row]}\t{alts[row]}\t.\tPASS\t" for row in range(n_var)]
    prefix = [(site + ".\t").encode() for site in sites]

    reference = work / "reference.vcf.gz"
    if not reference.exists():
        panel = np.load(root / "panel_haps.npy", mmap_mode="r")
        samples = (root / "panel_samples.txt").read_text().split()
        sink, process = bgzip_writer(reference, args.threads)
        process.stdin.write(header(args.chrom, length, samples, "GT"))
        allele_number = panel.shape[1]
        for row in range(n_var):
            haplotypes = np.asarray(panel[row])
            body = PHASED[2 * haplotypes[0::2].astype(np.intp) + haplotypes[1::2]].reshape(-1).copy()
            body[-1] = ord("\n")
            info = f"AC={int(haplotypes.sum())};AN={allele_number}\t".encode()
            process.stdin.write(sites[row].encode() + info + b"GT\t" + body.tobytes())
        finish(sink, process, reference)

    chunks_file = work / "chunks.txt"
    if not chunks_file.exists():
        run([str(tools / "GLIMPSE2_chunk_static"), "--input", str(reference), "--region", args.chrom,
             "--map", str(gmap), "--sequential", "--output", str(chunks_file)])
    chunks = [line.split() for line in chunks_file.read_text().splitlines() if line.strip()]
    binaries = []
    for chunk in chunks:
        input_region, output_region = chunk[2], chunk[3]
        stem = work / "refbin" / "ref"
        stem.parent.mkdir(exist_ok=True)
        # GLIMPSE2_split_reference names each binary after the chunk's input (buffered) region.
        start, end = input_region.split(":")[1].split("-")
        binary = Path(f"{stem}_{args.chrom}_{start}_{end}.bin")
        if not binary.exists():
            run([str(tools / "GLIMPSE2_split_reference_static"), "--reference", str(reference), "--map", str(gmap),
                 "--input-region", input_region, "--output-region", output_region, "--output", str(stem),
                 "--threads", str(args.threads)])
        binaries.append(binary)

    # Whole matrices live in RAM: strided memmap access on the network filesystem is random I/O.
    truth = np.load(root / "truth_G.npy")
    size = args.limit_samples or truth.shape[1]
    suffix = "_smoke" if args.limit_samples else ""
    info_sum = np.zeros(n_var)
    info_weight = np.zeros(n_var)
    stats = work / f"stats{suffix}.npz"
    if stats.exists():
        saved = np.load(stats)
        info_sum, info_weight = saved["info_sum"], saved["info_weight"]
    starts = list(range(0, size, args.batch))
    for batch_index, first in enumerate(starts):
        if args.only_batch >= 0 and batch_index != args.only_batch:
            continue
        flag = work / f"batch{batch_index}{suffix}.done"
        if flag.exists():
            continue
        last = min(first + args.batch, size)
        names = [f"s{index}" for index in range(first, last)]
        rng = np.random.default_rng([20260919, batch_index, int(args.chrom.lstrip("chr"))])
        target = work / f"gl{batch_index}{suffix}.vcf.gz"
        sink, process = bgzip_writer(target, args.threads)
        process.stdin.write(header(args.chrom, length, names, "PL"))
        errors = np.array([BASE_ERROR[int(code)] for code in cls[simple_rows]])
        for block_start in range(0, simple_rows.size, ROW_BLOCK):
            rows = simple_rows[block_start:block_start + ROW_BLOCK]
            genotype = truth[rows, first:last].astype(np.intp)
            pl = simulated_pl(genotype, errors[block_start:block_start + ROW_BLOCK, None], rng)
            text = pl_text(pl.reshape(-1, 3)).reshape(rows.size, -1)
            text[:, -1] = ord("\n")
            for row, body in zip(rows, text):
                process.stdin.write(prefix[row] + b"PL\t" + body.tobytes())
        finish(sink, process, target)
        outputs = []
        for index, (chunk, binary) in enumerate(zip(chunks, binaries)):
            output = work / f"imp{batch_index}{suffix}_{index}.bcf"
            run([str(tools / "GLIMPSE2_phase_static"), "--input-gl", str(target), "--reference", str(binary),
                 "--output", str(output), "--threads", str(args.threads), "--err-imp", ERR_IMP])
            outputs.append(output)
        listing = work / f"ligate{batch_index}{suffix}.txt"
        listing.write_text("\n".join(str(path) for path in outputs) + "\n")
        ligated = work / f"imputed{batch_index}{suffix}.bcf"
        run([str(tools / "GLIMPSE2_ligate_static"), "--input", str(listing), "--output", str(ligated), "--threads", str(args.threads)])
        batch_codes = np.zeros((n_var, last - first), dtype=np.uint8)
        seen = np.zeros(n_var, dtype=bool)
        reader = VCF(str(ligated))
        for record in reader:
            row = int(record.ID[1:])
            dosage = record.format("DS")[:, 0]
            milli = np.clip(np.rint(dosage * 1000.0), 0, 2000).astype(np.int64)
            batch_codes[row] = encode_milli(milli)
            seen[row] = True
            info_sum[row] += float(record.INFO["INFO"]) * (last - first)
            info_weight[row] += last - first
        reader.close()
        if not seen.all():
            raise SystemExit(f"batch {batch_index}: {int((~seen).sum())} records missing from the GLIMPSE2 output")
        np.save(work / f"codes{batch_index}{suffix}.npy", batch_codes)
        np.savez(stats, info_sum=info_sum, info_weight=info_weight)
        for path in outputs:
            path.unlink()
        target.unlink()
        flag.touch()
        print(f"batch {batch_index} imputed ({last}/{size})", flush=True)

    if all((work / f"batch{index}{suffix}.done").exists() for index in range(len(starts))):
        observed = np.concatenate([np.load(work / f"codes{index}{suffix}.npy") for index in range(len(starts))], axis=1)
        np.save(root / f"observed{suffix}.npy", observed)
        realized = np.zeros(n_var)
        for first in range(0, n_var, 5000):
            true_block = truth[first:first + 5000, :size].astype(np.float64)
            observed_block = observed[first:first + 5000].astype(np.float64) / CODES_PER_DOSAGE
            true_block -= true_block.mean(axis=1, keepdims=True)
            observed_block -= observed_block.mean(axis=1, keepdims=True)
            numerator = (true_block * observed_block).sum(axis=1) ** 2
            denominator = (true_block ** 2).sum(axis=1) * (observed_block ** 2).sum(axis=1)
            realized[first:first + 5000] = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        name = "imputation_smoke.npz" if args.limit_samples else "imputation.npz"
        np.savez(root / name, info=info_sum / info_weight, realized_r2=realized)
        for code in range(4):
            members = cls == code
            print(f"class {code}: median realized r2 {np.median(realized[members]):.3f}, "
                  f"mean {realized[members].mean():.3f} over {members.sum()} records", flush=True)


if __name__ == "__main__":
    main()
