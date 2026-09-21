"""The cytotoxicity lines' genotypes: the 1kGP 30x phased panel's every record for the analysis lines, as plink beds per
chromosome (``tools/build_genotypes.sh``: plink2 with ``--set-all-var-ids '@:#:$r:$a'``, A1 = ALT).

``read_bed`` decodes plink's SNP-major bed (magic 6c 1b 01; two bits per line, four lines per byte, per variant:
00 two copies of A1, 10 one copy, 11 none, 01 missing) into ALT dosages as int8, -1 for a missing call, for the
chosen variant rows. ``variant_table`` types every record from its bim line: a symbolic ALT (``<DEL>``, ``<INS:ME:ALU>``)
is an SV of that token; a sequence-resolved record is an SV where either allele is 50 bp or longer (the 1kGP panel's
own boundary, Byrska-Bishop et al. 2022), an SNV where both are one base, an indel otherwise.

``king_kinship`` is KING-robust (Manichaikul et al. 2010; bench-real's ``relatedness.py`` has the formula) on the
autosomal SNVs, and ``related_pairs`` the pairs at third degree or closer (kinship above 2^-4.5).
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pandas as pd

GENO = pathlib.Path("/scratch.global/sauer354/svpgs-team/bench-tox/geno")
AUTOSOMES = tuple(range(1, 23))
# The 1kGP panel's SV boundary: an allele of 50 bp or longer (Byrska-Bishop et al. 2022, Cell 185:3426).
SV_LENGTH = 50
# KING's third-degree boundary, the geometric midpoint 2^-(3 + 1.5) (Manichaikul et al. 2010).
THIRD_DEGREE = 2.0 ** -4.5
_MAGIC = b"\x6c\x1b\x01"


def samples(chrom: int = 22) -> list[str]:
    fam = pd.read_csv(GENO / f"chr{chrom}.fam", sep=r"\s+", header=None, dtype=str)
    return fam[1].tolist()


def variant_table(chrom: int) -> pd.DataFrame:
    """One row per bim record: position, ref, alt, is_sv, sv_type (the symbolic token, "." otherwise), end (POS +
    len(REF) - 1), allele_length_change (ALT length - REF length for sequence-resolved records, 0 for symbolic ones,
    which ``bench_real_signed_change`` then reads from the token), sv_length (the SVLEN the id carries: 0 here, the
    panel's symbolic records give none through plink)."""
    bim = pd.read_csv(GENO / f"chr{chrom}.bim", sep=r"\s+", header=None, names=["chrom", "id", "cm", "position", "a1", "a2"], dtype={"a1": str, "a2": str})
    # Lengths and the symbolic token from the object columns: a fixed-width numpy string array of a million alleles,
    # as wide as the longest insertion, is tens of gigabytes.
    symbolic = bim["a1"].str.startswith("<").to_numpy()
    ref_length = bim["a2"].str.len().to_numpy(dtype=np.int64)
    alt_length = np.where(symbolic, ref_length, bim["a1"].str.len().to_numpy(dtype=np.int64))
    is_sv = symbolic | (np.maximum(ref_length, alt_length) >= SV_LENGTH)
    sv_type = np.where(symbolic, bim["a1"].to_numpy(dtype=object), ".")
    change = np.where(symbolic, 0, alt_length - ref_length).astype(np.int64)
    # Compact: no allele strings (25 million records genome-wide), the token as a category.
    return pd.DataFrame({
        "position": bim["position"].to_numpy(dtype=np.int64), "is_sv": is_sv, "sv_type": pd.Categorical(sv_type),
        "end": bim["position"].to_numpy(dtype=np.int64) + ref_length - 1, "allele_length_change": change,
        "is_snv": (ref_length == 1) & (alt_length == 1) & ~symbolic,
    })


def _read_rows(path: pathlib.Path, rows: np.ndarray | None, bytes_per_variant: int) -> np.ndarray:
    """The packed bytes of the given variant rows (all when None), by plain reads: a memory map's touched pages are
    charged to the task's memory budget (a runq task reading the genome's beds was killed at 16 GB of page cache), and
    the pages read here are released to the kernel at once (posix_fadvise DONTNEED)."""
    with open(path, "rb") as handle:
        if handle.read(3) != _MAGIC:
            raise ValueError(f"{path} is not a SNP-major plink bed")
        if rows is None:
            chosen = np.frombuffer(handle.read(), dtype=np.uint8)
            chosen = chosen[: (chosen.shape[0] // bytes_per_variant) * bytes_per_variant].reshape(-1, bytes_per_variant).copy()
        else:
            rows = np.asarray(rows, dtype=np.int64)
            chosen = np.empty((rows.shape[0], bytes_per_variant), dtype=np.uint8)
            # Runs of consecutive rows are one read each.
            start = 0
            while start < rows.shape[0]:
                end = start + 1
                while end < rows.shape[0] and rows[end] == rows[end - 1] + 1:
                    end += 1
                handle.seek(3 + int(rows[start]) * bytes_per_variant)
                chosen[start:end] = np.frombuffer(handle.read((end - start) * bytes_per_variant), dtype=np.uint8).reshape(end - start, bytes_per_variant)
                start = end
        try:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass
    return chosen


def read_bed(chrom: int, rows: np.ndarray | None, sample_count: int) -> np.ndarray:
    """ALT dosages (rows x samples) as int8, -1 where missing, for the given variant rows (all rows when None)."""
    bytes_per_variant = (sample_count + 3) // 4
    chosen = _read_rows(GENO / f"chr{chrom}.bed", rows, bytes_per_variant)
    # Two bits per line, the first line in the low bits.
    codes = np.stack([(chosen >> shift) & 0b11 for shift in (0, 2, 4, 6)], axis=2).reshape(chosen.shape[0], -1)[:, :sample_count]
    dosage = np.empty(codes.shape, dtype=np.int8)
    dosage[codes == 0] = 2
    dosage[codes == 2] = 1
    dosage[codes == 3] = 0
    dosage[codes == 1] = -1
    return dosage


def read_chromosome_chunks(chrom: int, sample_count: int, chunk: int):
    """(row start, dosages (chunk x samples)) over the chromosome, in bim order."""
    total = variant_count(chrom, sample_count)
    for start in range(0, total, chunk):
        yield start, read_bed(chrom, np.arange(start, min(start + chunk, total)), sample_count)


def variant_count(chrom: int, sample_count: int) -> int:
    bytes_per_variant = (sample_count + 3) // 4
    return ((GENO / f"chr{chrom}.bed").stat().st_size - 3) // bytes_per_variant


def king_kinship(genotypes: np.ndarray) -> np.ndarray:
    """KING-robust kinship (samples x samples) from hard calls (samples x markers, -1 missing, missing treated as
    unknown by dropping the marker for the pair is not done: the 30x panel has no missing calls)."""
    het = (genotypes == 1).astype(np.float64)
    hom_alt = (genotypes == 2).astype(np.float64)
    hom_ref = (genotypes == 0).astype(np.float64)
    both_het = het @ het.T
    opposite = hom_alt @ hom_ref.T + hom_ref @ hom_alt.T
    het_count = het.sum(axis=1)
    low = np.minimum.outer(het_count, het_count)
    high = np.maximum.outer(het_count, het_count)
    with np.errstate(divide="ignore", invalid="ignore"):
        kinship = (2.0 * (both_het - 2.0 * opposite) + low - high) / (4.0 * low)
    np.fill_diagonal(kinship, 0.5)
    return np.where(np.isfinite(kinship), kinship, 0.0)


def related_pairs(lines: list[str], markers_per_chromosome: int, seed: int) -> list[tuple[str, str]]:
    """Pairs of lines at third degree or closer, from ``markers_per_chromosome`` autosomal SNVs per chromosome chosen
    at random (a seeded permutation of each chromosome's SNV rows): KING-robust needs some ten thousand markers."""
    generator = np.random.default_rng(seed)
    blocks = []
    count = len(lines)
    for chrom in AUTOSOMES:
        table = variant_table(chrom)
        snv_rows = np.flatnonzero(table["is_snv"].to_numpy())
        chosen = np.sort(generator.permutation(snv_rows)[:markers_per_chromosome])
        blocks.append(read_bed(chrom, chosen, count).T)
    genotypes = np.concatenate(blocks, axis=1)
    kinship = king_kinship(genotypes)
    first, second = np.nonzero(np.triu(kinship > THIRD_DEGREE, k=1))
    return [(lines[i], lines[j]) for i, j in zip(first.tolist(), second.tolist())]
