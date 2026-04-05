from __future__ import annotations

import csv
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import r2_score, roc_auc_score

from sv_pgs import BayesianPGS
from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.cli import main
from sv_pgs.data import VariantStatistics
from sv_pgs.io import (
    _VariantDefaults,
    _inspect_delimited_table,
    _resolve_sample_id_column,
    _vcf_cache_key,
    _load_vcf_from_cache,
    _save_vcf_to_cache,
    load_dataset_from_files,
    load_multi_vcf_dataset_from_files,
    run_training_pipeline,
)
import sv_pgs.genotype as genotype_module
import sv_pgs.cli as cli_module
import sv_pgs.io as io_module
from sv_pgs.plink import to_bed


@pytest.fixture(autouse=True)
def _skip_gpu_runtime_requirement_in_cli_tests(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli_module, "require_full_gpu_runtime", lambda: None)


def test_load_dataset_from_vcf_uses_metadata_and_sample_alignment(tmp_path: Path):
    vcf_path = tmp_path / "cohort.vcf"
    vcf_path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "##contig=<ID=1>",
                "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency\">",
                "##INFO=<ID=END,Number=1,Type=Integer,Description=\"End position\">",
                "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts2\ts1",
                "1\t100\trs1\tA\tC\t50\tPASS\tAF=0.25\tGT\t0/1\t1/1",
                "1\t200\tsv1\tN\t<DEL>\t60\tPASS\tAF=0.5;END=260\tGT\t0/0\t0/1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    sample_table_path = tmp_path / "samples.tsv"
    _write_table(
        sample_table_path,
        header=("sample_id", "target", "age"),
        rows=(
            ("s1", "1", "42"),
            ("s2", "0", "35"),
        ),
    )

    metadata_path = tmp_path / "variants.tsv"
    _write_table(
        metadata_path,
        header=("variant_id", "variant_class", "training_support", "is_copy_number", "prior_continuous__sv_length_score"),
        rows=(
            ("rs1", "snv", "", "false", "0.25"),
            ("sv1", "deletion_short", "2", "true", "1.75"),
        ),
    )

    dataset = load_dataset_from_files(
        genotype_path=vcf_path,
        genotype_format="vcf",
        sample_table_path=sample_table_path,
        sample_id_column="sample_id",
        target_column="target",
        covariate_columns=("age",),
        variant_metadata_path=metadata_path,
    )

    assert dataset.sample_ids == ["s1", "s2"]
    np.testing.assert_allclose(dataset.genotypes, np.array([[2.0, 1.0], [1.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(dataset.covariates, np.array([[42.0], [35.0]], dtype=np.float32))
    np.testing.assert_allclose(dataset.targets, np.array([1.0, 0.0], dtype=np.float32))
    assert dataset.variant_stats is not None
    np.testing.assert_allclose(dataset.variant_stats.support_counts, np.array([2, 1], dtype=np.int32))
    np.testing.assert_allclose(dataset.variant_stats.means, np.array([1.5, 0.5], dtype=np.float32))
    assert dataset.variant_records[0].variant_class == VariantClass.SNV
    assert dataset.variant_records[1].variant_class == VariantClass.DELETION_SHORT
    assert dataset.variant_records[0].prior_continuous_features == {"sv_length_score": 0.25}
    assert dataset.variant_records[1].training_support == 2
    assert dataset.variant_records[1].is_copy_number is True
    assert dataset.variant_records[1].prior_continuous_features == {"sv_length_score": 1.75}


def test_load_dataset_from_vcf_auto_detects_research_id_column(tmp_path: Path):
    vcf_path = tmp_path / "cohort.vcf"
    vcf_path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "##contig=<ID=1>",
                "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency\">",
                "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\trid2\trid1",
                "1\t100\trs1\tA\tC\t50\tPASS\tAF=0.25\tGT\t0/1\t1/1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    sample_table_path = tmp_path / "samples.tsv"
    _write_table(
        sample_table_path,
        header=("research_id", "target", "age"),
        rows=(
            ("rid1", "1", "42"),
            ("rid2", "0", "35"),
        ),
    )

    dataset = load_dataset_from_files(
        genotype_path=vcf_path,
        genotype_format="vcf",
        sample_table_path=sample_table_path,
        target_column="target",
        covariate_columns=("age",),
    )

    assert dataset.sample_ids == ["rid1", "rid2"]
    np.testing.assert_allclose(dataset.genotypes, np.array([[2.0], [1.0]], dtype=np.float32))


def test_load_multi_vcf_dataset_from_files_concatenates_variants_across_chromosomes(tmp_path: Path):
    vcf1_path = tmp_path / "chr1.vcf"
    vcf1_path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "##contig=<ID=1>",
                "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency\">",
                "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts2\ts1",
                "1\t100\tchr1_var1\tA\tC\t50\tPASS\tAF=0.25\tGT\t0/1\t1/1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    vcf2_path = tmp_path / "chr2.vcf"
    vcf2_path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "##contig=<ID=2>",
                "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency\">",
                "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts2\ts1",
                "2\t200\tchr2_var1\tG\tT\t60\tPASS\tAF=0.5\tGT\t0/0\t0/1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    sample_table_path = tmp_path / "samples.tsv"
    _write_table(
        sample_table_path,
        header=("sample_id", "target", "age"),
        rows=(
            ("s1", "1", "42"),
            ("s2", "0", "35"),
        ),
    )

    dataset = load_multi_vcf_dataset_from_files(
        genotype_paths=[vcf1_path, vcf2_path],
        sample_table_path=sample_table_path,
        sample_id_column="sample_id",
        target_column="target",
        covariate_columns=("age",),
    )

    assert dataset.sample_ids == ["s1", "s2"]
    assert dataset.genotypes.shape == (2, 2)
    np.testing.assert_allclose(
        dataset.genotypes.materialize(),
        np.array(
            [
                [2.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    assert [record.variant_id for record in dataset.variant_records] == ["chr1_var1", "chr2_var1"]
    assert dataset.variant_stats is not None
    np.testing.assert_allclose(dataset.variant_stats.support_counts, np.array([2, 1], dtype=np.int32))


def test_load_multi_vcf_dataset_from_files_rejects_duplicate_paths(tmp_path: Path):
    vcf_path = tmp_path / "chr1.vcf"
    vcf_path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "##contig=<ID=1>",
                "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency\">",
                "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1",
                "1\t100\tdup\tA\tC\t50\tPASS\tAF=0.25\tGT\t0/1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    sample_table_path = tmp_path / "samples.tsv"
    _write_table(
        sample_table_path,
        header=("sample_id", "target", "age"),
        rows=(("s1", "1", "42"),),
    )

    with pytest.raises(ValueError, match="genotype_paths must be unique"):
        load_multi_vcf_dataset_from_files(
            genotype_paths=[vcf_path, vcf_path],
            sample_table_path=sample_table_path,
            sample_id_column="sample_id",
            target_column="target",
            covariate_columns=("age",),
        )


def test_load_multi_vcf_dataset_from_files_rejects_duplicate_variant_keys(tmp_path: Path):
    vcf1_path = tmp_path / "chr1_a.vcf"
    vcf1_path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "##contig=<ID=1>",
                "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency\">",
                "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1",
                "1\t100\tdup\tA\tC\t50\tPASS\tAF=0.25\tGT\t0/1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    vcf2_path = tmp_path / "chr1_b.vcf"
    vcf2_path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "##contig=<ID=1>",
                "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency\">",
                "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1",
                "1\t100\tdup\tA\tC\t50\tPASS\tAF=0.25\tGT\t0/1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    sample_table_path = tmp_path / "samples.tsv"
    _write_table(
        sample_table_path,
        header=("sample_id", "target", "age"),
        rows=(("s1", "1", "42"),),
    )

    with pytest.raises(ValueError, match="duplicate variants detected across genotype_paths"):
        load_multi_vcf_dataset_from_files(
            genotype_paths=[vcf1_path, vcf2_path],
            sample_table_path=sample_table_path,
            sample_id_column="sample_id",
            target_column="target",
            covariate_columns=("age",),
        )


def test_load_dataset_from_vcf_auto_detects_sample_id_column_after_first_1000_rows(tmp_path: Path):
    sample_table_path = tmp_path / "samples.tsv"
    _write_table(
        sample_table_path,
        header=("sample_id", "research_id", "target"),
        rows=(
            tuple((f"nomatch_sample_{row_index}", f"nomatch_research_{row_index}", "0") for row_index in range(1001))
            + (
                ("genotype_1", "other_1", "1"),
                ("genotype_2", "other_2", "0"),
            )
        ),
    )

    resolved_column = _resolve_sample_id_column(
        table_spec=_inspect_delimited_table(sample_table_path),
        requested_sample_id_column="auto",
        available_sample_ids=("genotype_1", "genotype_2"),
    )

    assert resolved_column == "sample_id"


def test_load_dataset_from_gzipped_vcf(tmp_path: Path):
    vcf_path = tmp_path / "cohort.vcf.gz"
    with gzip.open(vcf_path, "wt", encoding="utf-8") as handle:
        handle.write(
            "\n".join(
                [
                    "##fileformat=VCFv4.2",
                    "##contig=<ID=1>",
                    "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency\">",
                    "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
                    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts2\ts1",
                    "1\t100\trs1\tA\tC\t50\tPASS\tAF=0.25\tGT\t0/1\t1/1",
                    "",
                ]
            )
        )

    sample_table_path = tmp_path / "samples.tsv"
    _write_table(
        sample_table_path,
        header=("sample_id", "target", "age"),
        rows=(
            ("s1", "1", "42"),
            ("s2", "0", "35"),
        ),
    )

    dataset = load_dataset_from_files(
        genotype_path=vcf_path,
        genotype_format="vcf",
        sample_table_path=sample_table_path,
        sample_id_column="sample_id",
        target_column="target",
        covariate_columns=("age",),
    )

    assert dataset.sample_ids == ["s1", "s2"]
    np.testing.assert_allclose(dataset.genotypes, np.array([[2.0], [1.0]], dtype=np.float32))


def test_load_dataset_from_vcf_uses_record_count_hint_for_direct_preallocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    vcf_path = tmp_path / "cohort.vcf"
    vcf_path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "##contig=<ID=1>",
                "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency\">",
                "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts2\ts1",
                "1\t100\trs1\tA\tC\t50\tPASS\tAF=0.25\tGT\t0/1\t1/1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    sample_table_path = tmp_path / "samples.tsv"
    _write_table(
        sample_table_path,
        header=("sample_id", "target", "age"),
        rows=(
            ("s1", "1", "42"),
            ("s2", "0", "35"),
        ),
    )

    monkeypatch.setattr(io_module, "_vcf_record_count_hint", lambda reader: 1)

    dataset = load_dataset_from_files(
        genotype_path=vcf_path,
        genotype_format="vcf",
        sample_table_path=sample_table_path,
        sample_id_column="sample_id",
        target_column="target",
        covariate_columns=("age",),
    )

    assert dataset.genotypes.shape == (2, 1)
    assert dataset.genotypes.matrix.flags.f_contiguous
    np.testing.assert_allclose(dataset.genotypes, np.array([[2.0], [1.0]], dtype=np.float32))


def test_vcf_cache_save_uses_real_temp_file_and_roundtrips(tmp_path: Path):
    vcf_path = tmp_path / "cohort.vcf"
    vcf_path.write_text("##fileformat=VCFv4.2\n", encoding="utf-8")
    keep_sample_indices = np.array([0, 2], dtype=np.int32)
    genotype_matrix = np.array([[0, 1], [1, 2]], dtype=np.int8, order="F")
    variants = [
        _VariantDefaults(
            variant_id="variant_0",
            variant_class=VariantClass.SNV,
            chromosome="1",
            position=100,
            length=1.0,
            allele_frequency=0.25,
            quality=50.0,
        )
    ]
    variant_stats = VariantStatistics(
        means=np.array([0.5], dtype=np.float32),
        scales=np.array([0.75], dtype=np.float32),
        allele_frequencies=np.array([0.25], dtype=np.float32),
        support_counts=np.array([1], dtype=np.int32),
    )

    _save_vcf_to_cache(
        vcf_path=vcf_path,
        keep_sample_indices=keep_sample_indices,
        genotype_matrix=genotype_matrix,
        variants=variants,
        variant_stats=variant_stats,
    )

    cached = _load_vcf_from_cache(vcf_path=vcf_path, keep_sample_indices=keep_sample_indices)

    assert cached is not None
    cached_genotypes, cached_variants, cached_variant_stats = cached
    np.testing.assert_array_equal(cached_genotypes, genotype_matrix)
    assert cached_genotypes.flags.f_contiguous
    assert not cached_genotypes.flags.c_contiguous
    assert cached_variants == variants
    np.testing.assert_allclose(cached_variant_stats.means, variant_stats.means)
    np.testing.assert_allclose(cached_variant_stats.scales, variant_stats.scales)
    np.testing.assert_allclose(cached_variant_stats.allele_frequencies, variant_stats.allele_frequencies)
    np.testing.assert_array_equal(cached_variant_stats.support_counts, variant_stats.support_counts)
    assert not list((tmp_path / ".sv_pgs_cache").glob("*.tmp"))
    assert not list((tmp_path / ".sv_pgs_cache").glob("*.tmp.npy"))


def test_vcf_cache_load_upgrades_legacy_row_major_matrix(tmp_path: Path):
    vcf_path = tmp_path / "cohort.vcf"
    vcf_path.write_text("##fileformat=VCFv4.2\n", encoding="utf-8")
    genotype_matrix = np.array([[0, 1], [1, 2]], dtype=np.int8, order="C")
    variants = [
        _VariantDefaults(
            variant_id="variant_0",
            variant_class=VariantClass.SNV,
            chromosome="1",
            position=100,
            length=1.0,
            allele_frequency=0.25,
            quality=50.0,
        )
    ]
    variant_stats = VariantStatistics(
        means=np.array([0.5], dtype=np.float32),
        scales=np.array([0.75], dtype=np.float32),
        allele_frequencies=np.array([0.25], dtype=np.float32),
        support_counts=np.array([1], dtype=np.int32),
    )

    _save_vcf_to_cache(
        vcf_path=vcf_path,
        keep_sample_indices=None,
        genotype_matrix=genotype_matrix,
        variants=variants,
        variant_stats=variant_stats,
    )

    cached = _load_vcf_from_cache(vcf_path=vcf_path, keep_sample_indices=None)

    assert cached is not None
    cached_genotypes, _, _ = cached
    np.testing.assert_array_equal(cached_genotypes, genotype_matrix)
    assert cached_genotypes.flags.f_contiguous

    reloaded = _load_vcf_from_cache(vcf_path=vcf_path, keep_sample_indices=None)

    assert reloaded is not None
    reloaded_genotypes, _, _ = reloaded
    np.testing.assert_array_equal(reloaded_genotypes, genotype_matrix)
    assert reloaded_genotypes.flags.f_contiguous


def test_vcf_cache_key_ignores_mtime_for_identical_content(tmp_path: Path):
    vcf_path = tmp_path / "cohort.vcf.gz"
    vcf_path.write_bytes(b"header\nbody\ntrailer\n")
    keep_sample_indices = np.array([0, 2], dtype=np.int32)

    first_key = _vcf_cache_key(vcf_path, keep_sample_indices)
    original_mtime = vcf_path.stat().st_mtime_ns
    new_mtime = original_mtime + 1_000_000
    os.utime(vcf_path, ns=(new_mtime, new_mtime))
    second_key = _vcf_cache_key(vcf_path, keep_sample_indices)

    assert first_key == second_key


def test_vcf_cache_key_changes_when_content_changes(tmp_path: Path):
    vcf_path = tmp_path / "cohort.vcf.gz"
    vcf_path.write_bytes(b"header\nbody\ntrailer\n")
    keep_sample_indices = np.array([0, 2], dtype=np.int32)

    first_key = _vcf_cache_key(vcf_path, keep_sample_indices)
    vcf_path.write_bytes(b"header\nchanged-body\ntrailer\n")
    second_key = _vcf_cache_key(vcf_path, keep_sample_indices)

    assert first_key != second_key


def test_load_dataset_from_plink_auto_detects_person_id_column(tmp_path: Path):
    bed_path = tmp_path / "cohort.bed"
    genotype_matrix = np.array(
        [
            [0.0, 1.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )
    to_bed(
        bed_path,
        genotype_matrix,
        properties={
            "fid": ["f1", "f2"],
            "iid": ["101", "102"],
            "sid": ["rs1", "rs2"],
            "chromosome": ["1", "1"],
            "bp_position": [100, 200],
            "allele_1": ["A", "G"],
            "allele_2": ["C", "T"],
        },
    )

    sample_table_path = tmp_path / "samples.tsv"
    _write_table(
        sample_table_path,
        header=("person_id", "target", "age"),
        rows=(
            ("102", "1", "55"),
            ("101", "0", "44"),
        ),
    )

    dataset = load_dataset_from_files(
        genotype_path=bed_path,
        genotype_format="plink1",
        sample_table_path=sample_table_path,
        target_column="target",
        covariate_columns=("age",),
    )

    assert dataset.sample_ids == ["101", "102"]
    np.testing.assert_allclose(dataset.genotypes, np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32))


def test_load_dataset_from_real_plink_bed_header_bytes(tmp_path: Path):
    bed_path = tmp_path / "cohort.bed"
    bed_path.write_bytes(bytes.fromhex("6c 1b 01 0b 34"))
    bed_path.with_suffix(".fam").write_text(
        "\n".join(
            [
                "f1 s1 0 0 0 -9",
                "f2 s2 0 0 0 -9",
                "f3 s3 0 0 0 -9",
                "",
            ]
        ),
        encoding="utf-8",
    )
    bed_path.with_suffix(".bim").write_text(
        "\n".join(
            [
                "1 rs1 0 100 A C",
                "1 rs2 0 200 G T",
                "",
            ]
        ),
        encoding="utf-8",
    )

    sample_table_path = tmp_path / "samples.tsv"
    _write_table(
        sample_table_path,
        header=("sample_id", "target"),
        rows=(
            ("s3", "1"),
            ("s1", "0"),
            ("s2", "1"),
        ),
    )

    dataset = load_dataset_from_files(
        genotype_path=bed_path,
        genotype_format="plink1",
        sample_table_path=sample_table_path,
        sample_id_column="sample_id",
        target_column="target",
        covariate_columns=(),
    )

    assert dataset.sample_ids == ["s1", "s2", "s3"]
    np.testing.assert_allclose(
        dataset.genotypes,
        np.array(
            [
                [0.0, 2.0],
                [1.0, np.nan],
                [2.0, 0.0],
            ],
            dtype=np.float32,
        ),
        equal_nan=True,
    )


def test_plink_loader_uses_indexed_bed_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bed_path = tmp_path / "cohort.bed"
    genotype_matrix = np.array(
        [
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            [1.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    to_bed(
        bed_path,
        genotype_matrix,
        properties={
            "fid": ["f1", "f2", "f3"],
            "iid": ["101", "102", "103"],
            "sid": ["rs1", "rs2", "rs3"],
            "chromosome": ["1", "1", "1"],
            "bp_position": [100, 200, 300],
            "allele_1": ["A", "G", "T"],
            "allele_2": ["C", "T", "C"],
        },
    )
    sample_table_path = tmp_path / "samples.tsv"
    _write_table(
        sample_table_path,
        header=("person_id", "target"),
        rows=(
            ("102", "1"),
            ("101", "0"),
            ("103", "1"),
        ),
    )

    original_open_bed = genotype_module.open_bed
    indexed_read_calls: list[tuple[np.ndarray, np.ndarray]] = []

    class RecordingBedReader:
        def __init__(self, delegate):
            self._delegate = delegate

        def read(self, *args, **kwargs):
            index = kwargs.get("index")
            if index is None:
                raise AssertionError("PLINK path attempted an unindexed full-matrix BED read.")
            sample_indices, variant_indices = index
            indexed_read_calls.append(
                (
                    variant_indices,
                    sample_indices,
                )
            )
            return self._delegate.read(*args, **kwargs)

        def __getattr__(self, name: str):
            return getattr(self._delegate, name)

    monkeypatch.setattr(genotype_module, "open_bed", lambda *args, **kwargs: RecordingBedReader(original_open_bed(*args, **kwargs)))

    dataset = load_dataset_from_files(
        genotype_path=bed_path,
        genotype_format="plink1",
        sample_table_path=sample_table_path,
        target_column="target",
        covariate_columns=(),
    )

    assert indexed_read_calls
    assert all(isinstance(call_sample_indices, slice) for _, call_sample_indices in indexed_read_calls)
    assert indexed_read_calls[0][1] == slice(0, 3, 1)
    assert sum(
        (
            call_variant_indices.stop - call_variant_indices.start
            if isinstance(call_variant_indices, slice)
            else np.asarray(call_variant_indices, dtype=np.int32).shape[0]
        )
        for call_variant_indices, _ in indexed_read_calls
    ) >= 3
    np.testing.assert_allclose(dataset.genotypes, np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 1.0], [1.0, 2.0, 0.0]], dtype=np.float32))


def test_load_dataset_from_plink_filters_non_genotyped_sample_rows(tmp_path: Path):
    bed_path = tmp_path / "cohort.bed"
    genotype_matrix = np.array(
        [
            [0.0, 1.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )
    to_bed(
        bed_path,
        genotype_matrix,
        properties={
            "fid": ["f1", "f2"],
            "iid": ["101", "102"],
            "sid": ["rs1", "rs2"],
            "chromosome": ["1", "1"],
            "bp_position": [100, 200],
            "allele_1": ["A", "G"],
            "allele_2": ["C", "T"],
        },
    )

    sample_table_path = tmp_path / "samples.tsv"
    _write_table(
        sample_table_path,
        header=("person_id", "target", "age"),
        rows=(
            ("999", "0", "70"),
            ("102", "1", "55"),
            ("101", "0", "44"),
            ("888", "1", "63"),
        ),
    )

    dataset = load_dataset_from_files(
        genotype_path=bed_path,
        genotype_format="plink1",
        sample_table_path=sample_table_path,
        target_column="target",
        covariate_columns=("age",),
    )

    assert dataset.sample_ids == ["101", "102"]
    np.testing.assert_allclose(dataset.targets, np.array([0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(dataset.covariates, np.array([[44.0], [55.0]], dtype=np.float32))
    np.testing.assert_allclose(dataset.genotypes, np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32))


def test_effective_bed_reader_batch_size_caps_large_sample_matrices():
    capped = genotype_module._effective_bed_reader_batch_size(
        sample_count=447_278,
        requested_batch_size=4_096,
    )
    assert capped < 4_096, "should cap below requested size for large sample counts"
    assert capped >= genotype_module.MIN_BED_READER_BATCH_SIZE, "should respect minimum batch size"
    # Small sample counts should pass through the requested size unchanged.
    assert genotype_module._effective_bed_reader_batch_size(
        sample_count=100,
        requested_batch_size=4_096,
    ) == 4_096


def test_contiguous_index_or_slice_prefers_slices_for_dense_ranges():
    assert genotype_module._contiguous_index_or_slice(np.array([3, 4, 5], dtype=np.int32)) == slice(3, 6, 1)
    non_contiguous = genotype_module._contiguous_index_or_slice(np.array([3, 5, 6], dtype=np.int32))
    assert isinstance(non_contiguous, np.ndarray)
    np.testing.assert_array_equal(non_contiguous, np.array([3, 5, 6], dtype=np.intp))


def test_load_dataset_from_files_auto_detect_fails_when_no_identifier_column_matches(tmp_path: Path):
    vcf_path = tmp_path / "cohort.vcf"
    vcf_path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "##contig=<ID=1>",
                "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1",
                "1\t100\trs1\tA\tC\t50\tPASS\t.\tGT\t0/1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    sample_table_path = tmp_path / "samples.tsv"
    _write_table(
        sample_table_path,
        header=("person_id", "target"),
        rows=(("101", "1"),),
    )

    with pytest.raises(ValueError, match="Could not find a sample identifier column"):
        load_dataset_from_files(
            genotype_path=vcf_path,
            genotype_format="vcf",
            sample_table_path=sample_table_path,
            target_column="target",
            covariate_columns=(),
        )


def test_run_training_pipeline_from_plink_inputs_writes_outputs(tmp_path: Path):
    bed_path = tmp_path / "cohort.bed"
    genotype_matrix = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    to_bed(
        bed_path,
        genotype_matrix,
        properties={
            "fid": ["f1", "f2", "f3", "f4", "f5", "f6"],
            "iid": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "sid": ["rs1", "sv1", "rs2"],
            "chromosome": ["1", "1", "1"],
            "bp_position": [100, 200, 300],
            "allele_1": ["A", "N", "G"],
            "allele_2": ["C", "<DEL>", "A"],
        },
    )

    sample_table_path = tmp_path / "samples.tsv"
    _write_table(
        sample_table_path,
        header=("sample_id", "target", "age"),
        rows=(
            ("s3", "2.7", "50"),
            ("s1", "0.2", "40"),
            ("s6", "2.9", "60"),
            ("s2", "1.1", "41"),
            ("s5", "1.8", "57"),
            ("s4", "-0.3", "39"),
        ),
    )

    metadata_path = tmp_path / "variants.tsv"
    _write_table(
        metadata_path,
        header=("variant_id", "variant_class", "training_support", "is_copy_number"),
        rows=(
            ("sv1", "deletion_short", "4", "true"),
        ),
    )

    dataset = load_dataset_from_files(
        genotype_path=bed_path,
        genotype_format="plink1",
        sample_table_path=sample_table_path,
        sample_id_column="sample_id",
        target_column="target",
        covariate_columns=("age",),
        variant_metadata_path=metadata_path,
    )

    output_dir = tmp_path / "run"
    outputs = run_training_pipeline(
        dataset=dataset,
        config=ModelConfig(
            trait_type=TraitType.QUANTITATIVE,
            max_outer_iterations=2,
        ),
        output_dir=output_dir,
    )

    assert outputs.artifact_dir.is_dir()
    assert outputs.summary_path.is_file()
    assert outputs.predictions_path.is_file()
    assert outputs.coefficients_path.is_file()
    assert (outputs.artifact_dir / "arrays.npz").is_file()
    assert (outputs.artifact_dir / "metadata.json").is_file()

    summary_payload = json.loads(outputs.summary_path.read_text(encoding="utf-8"))
    assert summary_payload["sample_count"] == 6
    assert summary_payload["variant_count"] == 3
    assert summary_payload["trait_type"] == "quantitative"
    assert "training_r2" in summary_payload

    prediction_lines = outputs.predictions_path.read_text(encoding="utf-8").strip().splitlines()
    coefficient_lines = outputs.coefficients_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(prediction_lines) == 7
    assert len(coefficient_lines) == 4


def test_cli_infers_binary_trait_type(tmp_path: Path):
    vcf_path = tmp_path / "cohort.vcf"
    vcf_path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "##contig=<ID=1>",
                "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency\">",
                "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\ts4",
                "1\t100\trs1\tA\tC\t50\tPASS\tAF=0.25\tGT\t0/0\t0/1\t1/1\t0/1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    sample_table_path = tmp_path / "samples.tsv"
    _write_table(
        sample_table_path,
        header=("sample_id", "target"),
        rows=(
            ("s1", "0"),
            ("s2", "1"),
            ("s3", "1"),
            ("s4", "0"),
        ),
    )

    output_dir = tmp_path / "run"
    exit_code = main(
        [
            "run",
            "--genotypes",
            str(vcf_path),
            "--sample-table",
            str(sample_table_path),
            "--target-column",
            "target",
            "--output-dir",
            str(output_dir),
            "--max-outer-iterations",
            "1",
        ]
    )

    assert exit_code == 0
    summary_payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["trait_type"] == "binary"


def test_vcf_cli_end_to_end_recovers_binary_signal_with_symbolic_svs(tmp_path: Path):
    random_generator = np.random.default_rng(7)
    sample_count = 96
    sample_ids = [f"sample_{sample_index}" for sample_index in range(sample_count)]
    age = random_generator.normal(size=sample_count).astype(np.float32)
    base_snv = random_generator.binomial(2, 0.35, size=sample_count).astype(np.float32)
    deletion_sv = random_generator.binomial(2, 0.08, size=sample_count).astype(np.float32)
    duplication_sv = random_generator.binomial(2, 0.12, size=sample_count).astype(np.float32)
    null_snv = random_generator.binomial(2, 0.45, size=sample_count).astype(np.float32)

    linear_predictor = 2.2 * base_snv + 3.4 * deletion_sv - 2.8 * duplication_sv + 0.9 * age - 1.3
    targets = (linear_predictor > np.median(linear_predictor)).astype(np.int32)

    genotypes_by_variant = {
        "snv_causal": base_snv,
        "sv_del": deletion_sv,
        "sv_dup": duplication_sv,
        "snv_null": null_snv,
    }
    genotypes_by_variant["sv_del"][5] = np.nan
    genotypes_by_variant["sv_dup"][11] = np.nan

    vcf_path = tmp_path / "binary_signal.vcf"
    _write_vcf(
        vcf_path,
        sample_ids=sample_ids,
        records=(
            _vcf_record_payload("1", 100, "snv_causal", "A", ("C",), 90.0, "AF=0.35", genotypes_by_variant["snv_causal"]),
            _vcf_record_payload("1", 200, "sv_del", "N", ("<DEL>",), 80.0, "AF=0.08;SVTYPE=DEL;END=340", genotypes_by_variant["sv_del"]),
            _vcf_record_payload("1", 400, "sv_dup", "N", ("<DUP>",), 78.0, "AF=0.12;SVTYPE=DUP;END=600", genotypes_by_variant["sv_dup"]),
            _vcf_record_payload("1", 800, "snv_null", "G", ("T",), 70.0, "AF=0.45", genotypes_by_variant["snv_null"]),
        ),
    )

    sample_table_path = tmp_path / "binary_samples.tsv"
    _write_table(
        sample_table_path,
        header=("sample_id", "target", "age"),
        rows=tuple(
            (sample_id, str(int(target)), str(float(age_value)))
            for sample_id, target, age_value in zip(sample_ids, targets, age, strict=True)
        ),
    )

    metadata_path = tmp_path / "binary_variants.tsv"
    _write_table(
        metadata_path,
        header=("variant_id", "variant_class", "training_support", "is_copy_number"),
        rows=(
            ("sv_del", "deletion_short", str(sample_count), "true"),
            ("sv_dup", "duplication_short", str(sample_count), "true"),
        ),
    )

    output_dir = tmp_path / "binary_run"
    exit_code = main(
        [
            "run",
            "--genotypes",
            str(vcf_path),
            "--sample-table",
            str(sample_table_path),
            "--target-column",
            "target",
            "--covariate-column",
            "age",
            "--variant-metadata",
            str(metadata_path),
            "--output-dir",
            str(output_dir),
            "--max-outer-iterations",
            "6",
            "--random-seed",
            "0",
        ]
    )

    assert exit_code == 0
    summary_payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["trait_type"] == "binary"
    assert summary_payload["training_auc"] is not None
    assert summary_payload["training_auc"] > 0.8

    dataset = load_dataset_from_files(
        genotype_path=vcf_path,
        genotype_format="vcf",
        sample_table_path=sample_table_path,
        sample_id_column="sample_id",
        target_column="target",
        covariate_columns=("age",),
        variant_metadata_path=metadata_path,
    )
    loaded_model = BayesianPGS.load(output_dir / "artifact")
    loaded_probability = loaded_model.predict_proba(dataset.genotypes, dataset.covariates)[:, 1]
    assert roc_auc_score(dataset.targets, loaded_probability) > 0.8

    prediction_rows = _read_tsv_rows(output_dir / "predictions.tsv")
    file_probability = np.asarray([float(row["probability"]) for row in prediction_rows], dtype=np.float32)
    np.testing.assert_allclose(file_probability, loaded_probability, atol=1e-5)


def test_plink_end_to_end_recovers_quantitative_signal_with_sv_style_alleles(tmp_path: Path):
    random_generator = np.random.default_rng(11)
    sample_count = 84
    sample_ids = [f"sample_{sample_index}" for sample_index in range(sample_count)]
    age = random_generator.normal(size=sample_count).astype(np.float32)
    snv_causal = random_generator.binomial(2, 0.3, size=sample_count).astype(np.float32)
    deletion_variant = random_generator.binomial(2, 0.12, size=sample_count).astype(np.float32)
    duplication_variant = random_generator.binomial(2, 0.10, size=sample_count).astype(np.float32)
    null_variant = random_generator.binomial(2, 0.4, size=sample_count).astype(np.float32)
    extra_null_variant = random_generator.binomial(2, 0.25, size=sample_count).astype(np.float32)

    genotype_matrix = np.column_stack(
        [
            snv_causal,
            deletion_variant,
            duplication_variant,
            null_variant,
            extra_null_variant,
        ]
    ).astype(np.float32)
    genotype_matrix[3, 1] = np.nan
    genotype_matrix[12, 2] = np.nan
    genotype_matrix[20, 4] = np.nan

    targets = (
        1.6 * np.nan_to_num(snv_causal, nan=0.0)
        + 2.3 * np.nan_to_num(deletion_variant, nan=0.0)
        - 1.9 * np.nan_to_num(duplication_variant, nan=0.0)
        + 0.7 * age
        + random_generator.normal(scale=0.35, size=sample_count)
    ).astype(np.float32)

    bed_path = tmp_path / "quantitative_signal.bed"
    to_bed(
        bed_path,
        genotype_matrix,
        properties={
            "fid": [f"family_{sample_index}" for sample_index in range(sample_count)],
            "iid": sample_ids,
            "sid": ["snv_causal", "sv_del", "sv_dup", "null_0", "null_1"],
            "chromosome": ["1", "1", "2", "3", "4"],
            "bp_position": [100, 200, 300, 400, 500],
            "allele_1": ["A", "N", "N", "G", "TT"],
            "allele_2": ["C", "<DEL>", "<DUP>", "A", "T"],
        },
    )

    sample_table_path = tmp_path / "quantitative_samples.tsv"
    _write_table(
        sample_table_path,
        header=("sample_id", "target", "age"),
        rows=tuple(
            (sample_id, str(float(target_value)), str(float(age_value)))
            for sample_id, target_value, age_value in zip(sample_ids, targets, age, strict=True)
        ),
    )

    metadata_path = tmp_path / "quantitative_variants.tsv"
    _write_table(
        metadata_path,
        header=("variant_id", "variant_class", "training_support", "is_copy_number"),
        rows=(
            ("sv_del", "deletion_short", str(sample_count), "true"),
            ("sv_dup", "duplication_short", str(sample_count), "true"),
        ),
    )

    output_dir = tmp_path / "quantitative_run"
    exit_code = main(
        [
            "run",
            "--genotypes",
            str(bed_path),
            "--sample-table",
            str(sample_table_path),
            "--target-column",
            "target",
            "--covariate-column",
            "age",
            "--variant-metadata",
            str(metadata_path),
            "--output-dir",
            str(output_dir),
            "--max-outer-iterations",
            "6",
            "--random-seed",
            "0",
        ]
    )

    assert exit_code == 0
    summary_payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["trait_type"] == "quantitative"
    assert summary_payload["training_r2"] > 0.55

    dataset = load_dataset_from_files(
        genotype_path=bed_path,
        genotype_format="plink1",
        sample_table_path=sample_table_path,
        sample_id_column="sample_id",
        target_column="target",
        covariate_columns=("age",),
        variant_metadata_path=metadata_path,
    )
    loaded_model = BayesianPGS.load(output_dir / "artifact")
    loaded_prediction = loaded_model.predict(dataset.genotypes, dataset.covariates)
    assert r2_score(dataset.targets, loaded_prediction) > 0.55

    prediction_rows = _read_tsv_rows(output_dir / "predictions.tsv")
    file_prediction = np.asarray([float(row["prediction"]) for row in prediction_rows], dtype=np.float32)
    np.testing.assert_allclose(file_prediction, loaded_prediction, atol=1e-5)


def test_multiallelic_vcf_raises_clear_error(tmp_path: Path):
    sample_ids = ["sample_0", "sample_1", "sample_2"]
    vcf_path = tmp_path / "multiallelic.vcf"
    _write_vcf(
        vcf_path,
        sample_ids=sample_ids,
        records=(
            _vcf_record_payload("1", 100, "multiallelic", "A", ("C", "G"), 55.0, "AF=0.2,0.1", np.array([1.0, 0.0, 2.0], dtype=np.float32)),
        ),
    )
    sample_table_path = tmp_path / "samples.tsv"
    _write_table(
        sample_table_path,
        header=("sample_id", "target"),
        rows=(
            ("sample_0", "0"),
            ("sample_1", "1"),
            ("sample_2", "0"),
        ),
    )

    with pytest.raises(ValueError, match="Only biallelic VCF records are supported"):
        load_dataset_from_files(
            genotype_path=vcf_path,
            genotype_format="vcf",
            sample_table_path=sample_table_path,
            sample_id_column="sample_id",
            target_column="target",
            covariate_columns=(),
        )


def test_cli_handles_single_class_binary_targets_without_metric_crash(tmp_path: Path):
    sample_ids = [f"sample_{sample_index}" for sample_index in range(12)]
    variant_dosage = np.asarray([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0], dtype=np.float32)
    vcf_path = tmp_path / "single_class.vcf"
    _write_vcf(
        vcf_path,
        sample_ids=sample_ids,
        records=(
            _vcf_record_payload("1", 100, "rs1", "A", ("C",), 60.0, "AF=0.25", variant_dosage),
        ),
    )
    sample_table_path = tmp_path / "single_class_samples.tsv"
    _write_table(
        sample_table_path,
        header=("sample_id", "target"),
        rows=tuple((sample_id, "1") for sample_id in sample_ids),
    )

    output_dir = tmp_path / "single_class_run"
    exit_code = main(
        [
            "run",
            "--genotypes",
            str(vcf_path),
            "--sample-table",
            str(sample_table_path),
            "--target-column",
            "target",
            "--output-dir",
            str(output_dir),
            "--max-outer-iterations",
            "2",
        ]
    )

    assert exit_code == 0
    summary_payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["trait_type"] == "binary"
    assert summary_payload["training_auc"] is None
    assert "training_log_loss" in summary_payload


def test_vcf_symbolic_sv_type_is_inferred_without_metadata(tmp_path: Path):
    sample_ids = ["sample_0", "sample_1", "sample_2", "sample_3"]
    vcf_path = tmp_path / "symbolic_sv.vcf"
    _write_vcf(
        vcf_path,
        sample_ids=sample_ids,
        records=(
            _vcf_record_payload("1", 100, "sv_del", "N", ("<DEL>",), 80.0, "AF=0.1;SVTYPE=DEL;END=220", np.array([0.0, 1.0, 0.0, 2.0], dtype=np.float32)),
            _vcf_record_payload("1", 300, "sv_dup", "N", ("<DUP>",), 80.0, "AF=0.1;SVTYPE=DUP;END=420", np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float32)),
            _vcf_record_payload("1", 500, "sv_ins", "N", ("<INS:ME>",), 80.0, "AF=0.1;SVTYPE=INS;END=501", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)),
            _vcf_record_payload("1", 700, "sv_inv", "N", ("<INV>",), 80.0, "AF=0.1;SVTYPE=INV;END=900", np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)),
        ),
    )
    sample_table_path = tmp_path / "samples.tsv"
    _write_table(
        sample_table_path,
        header=("sample_id", "target"),
        rows=(
            ("sample_0", "0"),
            ("sample_1", "1"),
            ("sample_2", "0"),
            ("sample_3", "1"),
        ),
    )

    dataset = load_dataset_from_files(
        genotype_path=vcf_path,
        genotype_format="vcf",
        sample_table_path=sample_table_path,
        sample_id_column="sample_id",
        target_column="target",
        covariate_columns=(),
    )

    assert [record.variant_class for record in dataset.variant_records] == [
        VariantClass.DELETION_SHORT,
        VariantClass.DUPLICATION_SHORT,
        VariantClass.INSERTION_MEI,
        VariantClass.INVERSION_BND_COMPLEX,
    ]


def test_plink_symbolic_sv_type_is_inferred_without_metadata(tmp_path: Path):
    bed_path = tmp_path / "symbolic_sv.bed"
    genotype_matrix = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [2.0, 2.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    to_bed(
        bed_path,
        genotype_matrix,
        properties={
            "fid": ["f0", "f1", "f2", "f3"],
            "iid": ["sample_0", "sample_1", "sample_2", "sample_3"],
            "sid": ["sv_del", "sv_dup", "sv_ins", "sv_inv"],
            "chromosome": ["1", "1", "1", "1"],
            "bp_position": [100, 200, 300, 400],
            "allele_1": ["N", "N", "N", "N"],
            "allele_2": ["<DEL>", "<DUP>", "<INS:ME>", "<INV>"],
        },
    )
    sample_table_path = tmp_path / "samples.tsv"
    _write_table(
        sample_table_path,
        header=("sample_id", "target"),
        rows=(
            ("sample_0", "0"),
            ("sample_1", "1"),
            ("sample_2", "0"),
            ("sample_3", "1"),
        ),
    )

    dataset = load_dataset_from_files(
        genotype_path=bed_path,
        genotype_format="plink1",
        sample_table_path=sample_table_path,
        sample_id_column="sample_id",
        target_column="target",
        covariate_columns=(),
    )

    assert [record.variant_class for record in dataset.variant_records] == [
        VariantClass.DELETION_SHORT,
        VariantClass.DUPLICATION_SHORT,
        VariantClass.INSERTION_MEI,
        VariantClass.INVERSION_BND_COMPLEX,
    ]


def _write_table(path: Path, header: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def _read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [{str(key): str(value) for key, value in row.items()} for row in reader]


def _write_vcf(
    path: Path,
    sample_ids: list[str],
    records: tuple[tuple[str, ...], ...],
) -> None:
    header_lines = [
        "##fileformat=VCFv4.2",
        "##contig=<ID=1>",
        "##contig=<ID=2>",
        "##contig=<ID=3>",
        "##contig=<ID=4>",
        "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency\">",
        "##INFO=<ID=END,Number=1,Type=Integer,Description=\"End position\">",
        "##INFO=<ID=SVTYPE,Number=1,Type=String,Description=\"SV type\">",
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(sample_ids),
    ]
    body_lines = ["\t".join(record) for record in records]
    path.write_text("\n".join([*header_lines, *body_lines, ""]), encoding="utf-8")


def _vcf_record_payload(
    chromosome: str,
    position: int,
    variant_id: str,
    reference: str,
    alternates: tuple[str, ...],
    quality: float,
    info_field: str,
    dosage: np.ndarray,
) -> tuple[str, ...]:
    if len(alternates) == 1:
        genotype_strings = [_dosage_to_biallelic_gt(value) for value in dosage]
    else:
        genotype_strings = [_dosage_to_multiallelic_gt(value) for value in dosage]
    return (
        chromosome,
        str(position),
        variant_id,
        reference,
        ",".join(alternates),
        str(quality),
        "PASS",
        info_field,
        "GT",
        *genotype_strings,
    )


def _dosage_to_biallelic_gt(value: float) -> str:
    if np.isnan(value):
        return "./."
    rounded_value = int(value)
    if rounded_value == 0:
        return "0/0"
    if rounded_value == 1:
        return "0/1"
    if rounded_value == 2:
        return "1/1"
    raise ValueError("Unsupported biallelic dosage: " + str(value))


def _dosage_to_multiallelic_gt(value: float) -> str:
    if np.isnan(value):
        return "./."
    rounded_value = int(value)
    if rounded_value == 0:
        return "0/0"
    if rounded_value == 1:
        return "0/1"
    if rounded_value == 2:
        return "0/2"
    raise ValueError("Unsupported multiallelic dosage: " + str(value))
