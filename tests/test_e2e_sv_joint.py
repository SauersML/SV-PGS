"""End-to-end integration test for the ``--variants snp+sv`` joint mode.

Combines a synthetic microarray BED with two synthetic SV VCFs (chr21 +
chr22) transcoded via :func:`sv_pgs.sv_transcoder.transcode_sv_vcf_to_bed`
and asserts that ``load_multi_source_dataset_from_files`` unifies them
into a single :class:`LoadedDataset` (intersected samples, concatenated
variants, preserved provenance).

Skips cleanly when ``cyvcf2`` is unavailable or ``bgzip``/``tabix`` are
missing from ``PATH``.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cyvcf2")
pytest.importorskip("sv_pgs.sv_transcoder")

from sv_pgs.config import ModelConfig  # noqa: E402
from sv_pgs.genotype import RawGenotypeMatrix  # noqa: E402
from sv_pgs.io import load_multi_source_dataset_from_files  # noqa: E402
from sv_pgs.plink import to_bed  # noqa: E402
from sv_pgs.sv_transcoder import transcode_sv_vcf_to_bed  # noqa: E402


_HAS_BGZIP_TABIX = shutil.which("bgzip") is not None and shutil.which("tabix") is not None
pytestmark = pytest.mark.skipif(
    not _HAS_BGZIP_TABIX,
    reason="bgzip / tabix required to index synthetic SV VCFs for cyvcf2",
)


def _write_microarray_bed(
    work_dir: Path,
    *,
    n_samples: int,
    n_variants: int,
    seed: int,
) -> tuple[Path, list[str], list[str]]:
    """Write a synthetic microarray PLINK BED trio.

    Returns ``(bed_path, sample_ids, variant_ids)``.
    """
    rng = np.random.default_rng(seed)
    afs = rng.uniform(0.05, 0.45, size=n_variants).astype(np.float32)
    dosage = np.empty((n_samples, n_variants), dtype=np.float32)
    for j in range(n_variants):
        p = float(afs[j])
        col = rng.binomial(2, p, size=n_samples).astype(np.float32)
        miss = rng.random(n_samples) < 0.03  # ~3% missing
        col[miss] = np.nan
        dosage[:, j] = col

    sample_ids = [f"s{i:04d}" for i in range(n_samples)]
    variant_ids = [f"snp{j:05d}" for j in range(n_variants)]
    bed_path = work_dir / "microarray.bed"
    to_bed(
        bed_path,
        dosage,
        properties={
            "fid": sample_ids,
            "iid": sample_ids,
            "sid": variant_ids,
            "chromosome": ["1"] * n_variants,
            "bp_position": list(range(1, n_variants + 1)),
        },
    )
    return bed_path, sample_ids, variant_ids


def _build_sv_vcf_bytes(
    sample_ids: list[str],
    contig: str,
    sv_ids: list[str],
    positions: list[int],
    seed: int,
) -> bytes:
    """Build a minimal SV VCF (one contig, many DEL/DUP/INS variants)."""
    rng = np.random.default_rng(seed)
    header_lines = [
        "##fileformat=VCFv4.2",
        f"##contig=<ID={contig}>",
        '##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of SV">',
        '##INFO=<ID=END,Number=1,Type=Integer,Description="End position">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(sample_ids),
    ]
    svtypes = ["DEL", "DUP", "INS"]
    alt_for = {"DEL": "<DEL>", "DUP": "<DUP>", "INS": "<INS>"}
    body: list[str] = []
    for k, (vid, pos) in enumerate(zip(sv_ids, positions, strict=True)):
        svtype = svtypes[k % 3]
        length = int(rng.integers(50, 3000))
        end = pos + length
        # Random genotypes per sample, with ~5% missing.
        gt_codes = rng.choice(
            ["0/0", "0/1", "1/1", "./."],
            size=len(sample_ids),
            p=[0.6, 0.25, 0.10, 0.05],
        )
        info = f"SVTYPE={svtype};END={end}"
        body.append(
            f"{contig}\t{pos}\t{vid}\tN\t{alt_for[svtype]}\t.\tPASS\t{info}\tGT\t"
            + "\t".join(gt_codes)
        )
    text = "\n".join(header_lines + body) + "\n"
    return text.encode("utf-8")


def _write_sv_vcf(
    work_dir: Path,
    name: str,
    sample_ids: list[str],
    contig: str,
    sv_ids: list[str],
    positions: list[int],
    seed: int,
) -> Path:
    payload = _build_sv_vcf_bytes(sample_ids, contig, sv_ids, positions, seed)
    plain = work_dir / f"{name}.vcf"
    plain.write_bytes(payload)
    bgz = work_dir / f"{name}.vcf.gz"
    subprocess.run(
        ["bgzip", "-c", str(plain)],
        check=True,
        stdout=bgz.open("wb"),
    )
    subprocess.run(["tabix", "-p", "vcf", str(bgz)], check=True)
    return bgz


def _write_sample_table(
    path: Path,
    sample_ids: list[str],
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    n = len(sample_ids)
    targets = rng.integers(0, 2, size=n)
    pc1 = rng.standard_normal(n)
    pc2 = rng.standard_normal(n)
    lines = ["sample_id\ttarget\tPC1\tPC2\n"]
    for i, sid in enumerate(sample_ids):
        lines.append(f"{sid}\t{int(targets[i])}\t{pc1[i]}\t{pc2[i]}\n")
    path.write_text("".join(lines), encoding="utf-8")


def test_e2e_sv_joint_loader() -> None:
    n_samples_micro = 200
    n_variants_micro = 1500
    n_svs_per_chr = 50

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)

        # 1. Microarray BED (200 samples × 1500 variants).
        micro_bed, micro_sample_ids, micro_variant_ids = _write_microarray_bed(
            work,
            n_samples=n_samples_micro,
            n_variants=n_variants_micro,
            seed=42,
        )

        # 2. Two SV VCFs with overlapping sample sets. The chr21 VCF uses
        #    the first 180 microarray samples; chr22 uses samples 20:200.
        #    Intersection across all three sources: samples 20..179 (160).
        chr21_samples = micro_sample_ids[:180]
        chr22_samples = micro_sample_ids[20:]

        chr21_sv_ids = [f"sv21_{k:03d}" for k in range(n_svs_per_chr)]
        chr21_positions = [10_000 + 1000 * k for k in range(n_svs_per_chr)]
        chr22_sv_ids = [f"sv22_{k:03d}" for k in range(n_svs_per_chr)]
        chr22_positions = [20_000 + 1500 * k for k in range(n_svs_per_chr)]

        chr21_vcf = _write_sv_vcf(
            work, "chr21_svs", chr21_samples, "chr21",
            chr21_sv_ids, chr21_positions, seed=123,
        )
        chr22_vcf = _write_sv_vcf(
            work, "chr22_svs", chr22_samples, "chr22",
            chr22_sv_ids, chr22_positions, seed=456,
        )

        # 4. Transcode each SV VCF to a PLINK BED trio.
        chr21_bed = work / "chr21_svs.bed"
        chr22_bed = work / "chr22_svs.bed"
        meta21 = transcode_sv_vcf_to_bed([chr21_vcf], chr21_bed)
        meta22 = transcode_sv_vcf_to_bed([chr22_vcf], chr22_bed)
        assert meta21["n_variants"] == n_svs_per_chr
        assert meta22["n_variants"] == n_svs_per_chr

        # Expected sample intersection (preserves microarray order).
        expected_common = [
            sid for sid in micro_sample_ids
            if sid in set(chr21_samples) and sid in set(chr22_samples)
        ]

        # Sample table covers all microarray samples.
        sample_table_path = work / "samples.tsv"
        _write_sample_table(sample_table_path, micro_sample_ids, seed=7)

        # 5. Joint load.
        config = ModelConfig()
        dataset = load_multi_source_dataset_from_files(
            sources=[
                ("plink1", micro_bed),
                ("plink1", chr21_bed),
                ("plink1", chr22_bed),
            ],
            config=config,
            sample_table_path=sample_table_path,
            target_column="target",
            covariate_columns=["PC1", "PC2"],
            sample_id_column="sample_id",
        )

        # 6. Assertions.
        # Row dimension == intersection.
        assert dataset.sample_ids == expected_common
        assert dataset.genotypes.shape[0] == len(expected_common)

        # Variant count == 1500 + 50 + 50.
        expected_total_variants = n_variants_micro + 2 * n_svs_per_chr
        assert dataset.genotypes.shape[1] == expected_total_variants
        assert len(dataset.variant_records) == expected_total_variants

        # Provenance: first 1500 are microarray (chr "1"), next 50 chr21,
        # next 50 chr22.
        micro_records = dataset.variant_records[:n_variants_micro]
        chr21_records = dataset.variant_records[
            n_variants_micro : n_variants_micro + n_svs_per_chr
        ]
        chr22_records = dataset.variant_records[
            n_variants_micro + n_svs_per_chr :
        ]

        # Microarray block.
        assert all(r.chromosome == "1" for r in micro_records)
        assert [r.variant_id for r in micro_records] == micro_variant_ids
        # The microarray .bim writes positions starting at 1.
        assert [r.position for r in micro_records] == list(
            range(1, n_variants_micro + 1)
        )

        # chr21 SV block.
        assert all(r.chromosome == "chr21" for r in chr21_records)
        assert [r.variant_id for r in chr21_records] == chr21_sv_ids
        assert [r.position for r in chr21_records] == chr21_positions

        # chr22 SV block.
        assert all(r.chromosome == "chr22" for r in chr22_records)
        assert [r.variant_id for r in chr22_records] == chr22_sv_ids
        assert [r.position for r in chr22_records] == chr22_positions

        # Genotype matrix is a RawGenotypeMatrix subclass.
        assert isinstance(dataset.genotypes, RawGenotypeMatrix)

        # Covariates / targets dimensions.
        assert dataset.covariates.shape == (len(expected_common), 2)
        assert dataset.targets.shape == (len(expected_common),)
