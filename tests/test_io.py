from __future__ import annotations

import csv
import gzip
import json
import os
import struct
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import r2_score, roc_auc_score

from sv_pgs import BayesianPGS
from sv_pgs.artifact import _config_from_json
from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.cli import main
from sv_pgs.data import VariantRecord, VariantStatistics
from sv_pgs.genotype import Int8RawGenotypeMatrix
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
import sv_pgs.io as io_module
import sv_pgs.model as model_module
from sv_pgs.plink import to_bed


def _run_cli_without_gpu_runtime(argv: list[str]) -> int:
    return main(argv)


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
        header=(
            "variant_id",
            "variant_class",
            "training_support",
            "is_copy_number",
            "prior_binary__coding_annotation",
            "prior_continuous__sv_length_score",
            "prior_categorical__functional_state",
            "prior_membership__regulatory_mix",
            "prior_nested__gene_context",
        ),
        rows=(
            ("rs1", "snv", "", "false", "false", "0.25", "synonymous", "enhancer=0.2,promoter=0.8", "protein_coding>exon"),
            ("sv1", "deletion_short", "2", "true", "true", "1.75", "lof", "enhancer=0.7,promoter=0.3", "protein_coding>intron"),
        ),
    )

    dataset = load_dataset_from_files(
        genotype_path=vcf_path,
        config=ModelConfig(),
        genotype_format="vcf",
        sample_table_path=sample_table_path,
        sample_id_column="sample_id",
        target_column="target",
        covariate_columns=("age",),
        variant_metadata_path=metadata_path,
    )

    assert dataset.sample_ids == ["s2", "s1"]
    assert isinstance(dataset.genotypes, Int8RawGenotypeMatrix)
    np.testing.assert_allclose(dataset.genotypes, np.array([[1.0, 0.0], [2.0, 1.0]], dtype=np.float32))
    np.testing.assert_allclose(dataset.covariates, np.array([[35.0], [42.0]], dtype=np.float32))
    np.testing.assert_allclose(dataset.targets, np.array([0.0, 1.0], dtype=np.float32))
    assert dataset.variant_stats is not None
    np.testing.assert_allclose(dataset.variant_stats.support_counts, np.array([2, 1], dtype=np.int32))
    np.testing.assert_allclose(dataset.variant_stats.means, np.array([1.5, 0.5], dtype=np.float32))
    assert dataset.variant_records[0].variant_class == VariantClass.SNV
    assert dataset.variant_records[1].variant_class == VariantClass.DELETION_SHORT
    assert dataset.variant_records[0].prior_binary_features == {"coding_annotation": False}
    assert dataset.variant_records[0].prior_continuous_features == {"sv_length_score": 0.25}
    assert dataset.variant_records[0].prior_categorical_features == {"functional_state": "synonymous"}
    assert dataset.variant_records[0].prior_membership_features == {"regulatory_mix": {"enhancer": 0.2, "promoter": 0.8}}
    assert dataset.variant_records[0].prior_nested_features == {"gene_context": ("protein_coding", "exon")}
    assert dataset.variant_records[1].training_support == 2
    assert dataset.variant_records[1].is_copy_number is True
    assert dataset.variant_records[1].prior_binary_features == {"coding_annotation": True}
    assert dataset.variant_records[1].prior_continuous_features == {"sv_length_score": 1.75}
    assert dataset.variant_records[1].prior_categorical_features == {"functional_state": "lof"}
    assert dataset.variant_records[1].prior_membership_features == {"regulatory_mix": {"enhancer": 0.7, "promoter": 0.3}}
    assert dataset.variant_records[1].prior_nested_features == {"gene_context": ("protein_coding", "intron")}


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
        config=ModelConfig(),
        genotype_format="vcf",
        sample_table_path=sample_table_path,
        target_column="target",
        covariate_columns=("age",),
    )

    assert dataset.sample_ids == ["rid2", "rid1"]
    np.testing.assert_allclose(dataset.genotypes, np.array([[1.0], [2.0]], dtype=np.float32))


def test_artifact_config_parses_trait_type_without_backend():
    config = _config_from_json({"trait_type": TraitType.QUANTITATIVE.value})
    assert config.trait_type == TraitType.QUANTITATIVE


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
        config=ModelConfig(),
        sample_table_path=sample_table_path,
        sample_id_column="sample_id",
        target_column="target",
        covariate_columns=("age",),
    )

    assert dataset.sample_ids == ["s2", "s1"]
    assert dataset.genotypes.shape == (2, 2)
    np.testing.assert_allclose(
        dataset.genotypes.materialize(),
        np.array(
            [
                [1.0, 0.0],
                [2.0, 1.0],
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
            config=ModelConfig(),
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
            config=ModelConfig(),
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
        config=ModelConfig(),
        genotype_format="vcf",
        sample_table_path=sample_table_path,
        sample_id_column="sample_id",
        target_column="target",
        covariate_columns=("age",),
    )

    assert dataset.sample_ids == ["s2", "s1"]
    np.testing.assert_allclose(dataset.genotypes, np.array([[1.0], [2.0]], dtype=np.float32))


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
        config=ModelConfig(),
        genotype_format="vcf",
        sample_table_path=sample_table_path,
        sample_id_column="sample_id",
        target_column="target",
        covariate_columns=("age",),
    )

    assert dataset.genotypes.shape == (2, 1)
    assert isinstance(dataset.genotypes, Int8RawGenotypeMatrix)
    assert dataset.genotypes.matrix.flags.f_contiguous
    np.testing.assert_allclose(dataset.genotypes, np.array([[1.0], [2.0]], dtype=np.float32))


def test_vcf_cache_save_uses_real_temp_file_and_roundtrips(tmp_path: Path):
    vcf_path = tmp_path / "cohort.vcf"
    vcf_path.write_text("##fileformat=VCFv4.2\n", encoding="utf-8")
    config = ModelConfig()
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
        ),
        _VariantDefaults(
            variant_id="variant_1",
            variant_class=VariantClass.SNV,
            chromosome="1",
            position=101,
            length=1.0,
            allele_frequency=0.75,
            quality=40.0,
        ),
    ]
    variant_stats = VariantStatistics(
        means=np.array([0.5, 1.5], dtype=np.float32),
        scales=np.array([0.75, 0.75], dtype=np.float32),
        allele_frequencies=np.array([0.25, 0.75], dtype=np.float32),
        support_counts=np.array([1, 2], dtype=np.int32),
    )

    _save_vcf_to_cache(
        vcf_path=vcf_path,
        genotype_matrix=genotype_matrix,
        variants=variants,
        variant_stats=variant_stats,
        config=config,
    )

    cached = _load_vcf_from_cache(vcf_path=vcf_path, config=config)

    assert cached is not None
    cached_genotypes, cached_variants, cached_variant_stats = cached
    assert isinstance(cached_genotypes, np.memmap)
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
    config = ModelConfig()
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
        ),
        _VariantDefaults(
            variant_id="variant_1",
            variant_class=VariantClass.SNV,
            chromosome="1",
            position=101,
            length=1.0,
            allele_frequency=0.75,
            quality=40.0,
        ),
    ]
    variant_stats = VariantStatistics(
        means=np.array([0.5, 1.5], dtype=np.float32),
        scales=np.array([0.75, 0.75], dtype=np.float32),
        allele_frequencies=np.array([0.25, 0.75], dtype=np.float32),
        support_counts=np.array([1, 2], dtype=np.int32),
    )

    _save_vcf_to_cache(
        vcf_path=vcf_path,
        genotype_matrix=genotype_matrix,
        variants=variants,
        variant_stats=variant_stats,
        config=config,
    )

    cached = _load_vcf_from_cache(vcf_path=vcf_path, config=config)

    assert cached is not None
    cached_genotypes, _, _ = cached
    assert isinstance(cached_genotypes, np.memmap)
    np.testing.assert_array_equal(cached_genotypes, genotype_matrix)
    assert cached_genotypes.flags.f_contiguous

    reloaded = _load_vcf_from_cache(vcf_path=vcf_path, config=config)

    assert reloaded is not None
    reloaded_genotypes, _, _ = reloaded
    assert isinstance(reloaded_genotypes, np.memmap)
    np.testing.assert_array_equal(reloaded_genotypes, genotype_matrix)
    assert reloaded_genotypes.flags.f_contiguous


def test_vcf_cache_key_ignores_mtime_for_identical_content(tmp_path: Path):
    vcf_path = tmp_path / "cohort.vcf.gz"
    vcf_path.write_bytes(b"header\nbody\ntrailer\n")
    config = ModelConfig()

    first_key = _vcf_cache_key(vcf_path, config)
    original_mtime = vcf_path.stat().st_mtime_ns
    new_mtime = original_mtime + 1_000_000
    os.utime(vcf_path, ns=(new_mtime, new_mtime))
    second_key = _vcf_cache_key(vcf_path, config)

    assert first_key == second_key


def test_vcf_cache_key_changes_when_content_changes(tmp_path: Path):
    vcf_path = tmp_path / "cohort.vcf.gz"
    vcf_path.write_bytes(b"header\nbody\ntrailer\n")
    config = ModelConfig()

    first_key = _vcf_cache_key(vcf_path, config)
    vcf_path.write_bytes(b"header\nchanged-body\ntrailer\n")
    second_key = _vcf_cache_key(vcf_path, config)

    assert first_key != second_key


def test_vcf_cache_key_changes_when_minimum_scale_changes(tmp_path: Path):
    vcf_path = tmp_path / "cohort.vcf.gz"
    vcf_path.write_bytes(b"header\nbody\ntrailer\n")
    first_key = _vcf_cache_key(vcf_path, ModelConfig(minimum_scale=1e-6))
    second_key = _vcf_cache_key(vcf_path, ModelConfig(minimum_scale=0.25))

    assert first_key != second_key


def test_vcf_cache_load_ignores_incomplete_manifestless_new_bundle(tmp_path: Path):
    vcf_path = tmp_path / "cohort.vcf"
    vcf_path.write_text("##fileformat=VCFv4.2\n", encoding="utf-8")
    config = ModelConfig()
    key = _vcf_cache_key(vcf_path, config)
    cache_dir = tmp_path / ".sv_pgs_cache"
    cache_dir.mkdir()

    np.save(cache_dir / f"{key}.genotypes.npy", np.array([[0, 1]], dtype=np.int8), allow_pickle=False)
    io_module._write_variant_metadata(cache_dir / f"{key}.variants.npz", [])
    stats_dtype = np.dtype(
        [
            ("means", "<f4"),
            ("scales", "<f4"),
            ("allele_frequencies", "<f4"),
            ("support_counts", "<i4"),
        ]
    )
    np.save(cache_dir / f"{key}.stats.npy", np.zeros(1, dtype=stats_dtype), allow_pickle=False)

    assert _load_vcf_from_cache(vcf_path=vcf_path, config=config) is None


def test_prepare_keep_sample_selector_collapses_full_and_contiguous_ranges():
    assert io_module._prepare_keep_sample_selector(None, total_sample_count=4) is None
    assert io_module._prepare_keep_sample_selector(np.array([0, 1, 2, 3], dtype=np.intp), total_sample_count=4) is None

    contiguous = io_module._prepare_keep_sample_selector(np.array([2, 3, 4], dtype=np.intp), total_sample_count=8)
    assert isinstance(contiguous, slice)
    assert contiguous.start == 2
    assert contiguous.stop == 5

    scattered = io_module._prepare_keep_sample_selector(np.array([1, 3, 4], dtype=np.intp), total_sample_count=8)
    assert isinstance(scattered, np.ndarray)
    np.testing.assert_array_equal(scattered, np.array([1, 3, 4], dtype=np.intp))


def test_record_gt_types_to_int8_subsets_before_mapping():
    gt_map = np.array([0, 1, io_module.PLINK_MISSING_INT8, 2], dtype=np.int8)
    gt_types = np.array([0, 3, 2, 1, 0], dtype=np.int8)

    full = io_module._record_gt_types_to_int8(gt_types, gt_map, None)
    np.testing.assert_array_equal(full, np.array([0, 2, io_module.PLINK_MISSING_INT8, 1, 0], dtype=np.int8))

    contiguous = io_module._record_gt_types_to_int8(gt_types, gt_map, slice(1, 4))
    np.testing.assert_array_equal(contiguous, np.array([2, io_module.PLINK_MISSING_INT8, 1], dtype=np.int8))

    scattered = io_module._record_gt_types_to_int8(gt_types, gt_map, np.array([0, 3, 4], dtype=np.intp))
    np.testing.assert_array_equal(scattered, np.array([0, 1, 0], dtype=np.int8))


def test_precache_vcfs_parallel_reuses_completed_region_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    vcf_path = tmp_path / "chr1.vcf.gz"
    vcf_path.write_bytes(b"vcf")
    config = ModelConfig(minimum_scale=0.2)
    cache_dir = io_module._vcf_cache_dir(vcf_path)
    cache_dir.mkdir()
    key = _vcf_cache_key(vcf_path, config)
    tmp_dir = cache_dir / f"{key}.tmp_parallel"
    tmp_dir.mkdir()

    region0_prefix = tmp_dir / "region_0"
    region1_prefix = tmp_dir / "region_1"

    Path(f"{region0_prefix}.geno").write_bytes(np.array([0, 1], dtype=np.int8).tobytes())
    io_module._write_variant_metadata(
        Path(f"{region0_prefix}.variants.npz"),
        [
            _VariantDefaults(
                variant_id="var0",
                variant_class=VariantClass.SNV,
                chromosome="1",
                position=100,
                length=1.0,
                allele_frequency=0.25,
                quality=50.0,
            )
        ],
    )
    Path(f"{region0_prefix}.stats").write_bytes(struct.pack("<qqii", 1, 1, 2, 1))

    scheduled_tasks: list[tuple] = []

    class _FakePool:
        def __init__(self, processes: int) -> None:
            self.processes = processes

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _tb) -> None:
            return None

        def imap_unordered(self, _func, tasks):
            for task in tasks:
                scheduled_tasks.append(task)
                _vcf_path_str, _region, _keep_list, output_prefix, _threads = task
                Path(f"{output_prefix}.geno").write_bytes(np.array([2, 0], dtype=np.int8).tobytes())
                io_module._write_variant_metadata(
                    Path(f"{output_prefix}.variants.npz"),
                    [
                        _VariantDefaults(
                            variant_id="var1",
                            variant_class=VariantClass.SNV,
                            chromosome="1",
                            position=200,
                            length=1.0,
                            allele_frequency=0.50,
                            quality=40.0,
                        )
                    ],
                )
                Path(f"{output_prefix}.stats").write_bytes(struct.pack("<qqii", 2, 4, 2, 1))
                yield 1, str(output_prefix)

    class _FakeContext:
        def Pool(self, processes: int):
            return _FakePool(processes)

    monkeypatch.setattr(
        io_module,
        "_is_vcf_cache_bundle_complete",
        lambda paths: paths.geno_path.exists() and paths.var_path.exists() and paths.stats_path.exists() and paths.manifest_path.exists(),
    )
    monkeypatch.setattr(io_module, "_vcf_contig_info", lambda path: ("1", 200))
    monkeypatch.setattr(io_module, "_read_vcf_sample_ids", lambda path: ["s0", "s1"])
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    import multiprocessing as _multiprocessing
    monkeypatch.setattr(_multiprocessing, "get_all_start_methods", lambda: ["fork", "spawn"])
    monkeypatch.setattr(_multiprocessing, "get_context", lambda _method: _FakeContext())

    io_module.precache_vcfs_parallel([vcf_path], config)

    assert len(scheduled_tasks) == 1
    assert scheduled_tasks[0][3] == str(region1_prefix)
    assert scheduled_tasks[0][4] == 2

    cached = _load_vcf_from_cache(vcf_path=vcf_path, config=config)
    assert cached is not None
    genotype_matrix, variants, variant_stats = cached
    np.testing.assert_array_equal(
        np.asarray(genotype_matrix),
        np.array([[0, 2], [1, 0]], dtype=np.int8),
    )
    assert [variant.variant_id for variant in variants] == ["var0", "var1"]
    np.testing.assert_allclose(variant_stats.means, np.array([0.5, 1.0], dtype=np.float32))


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
        config=ModelConfig(),
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
        config=ModelConfig(),
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
        config=ModelConfig(),
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
        config=ModelConfig(),
        genotype_format="plink1",
        sample_table_path=sample_table_path,
        target_column="target",
        covariate_columns=("age",),
    )

    assert dataset.sample_ids == ["101", "102"]
    np.testing.assert_allclose(dataset.targets, np.array([0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(dataset.covariates, np.array([[44.0], [55.0]], dtype=np.float32))


def test_load_dataset_from_plink_passes_user_config_to_variant_statistics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
        header=("person_id", "target"),
        rows=(
            ("101", "0"),
            ("102", "1"),
        ),
    )

    captured_config: ModelConfig | None = None
    expected_stats = VariantStatistics(
        means=np.array([1.0, 0.5], dtype=np.float32),
        scales=np.array([0.25, 0.75], dtype=np.float32),
        allele_frequencies=np.array([0.5, 0.25], dtype=np.float32),
        support_counts=np.array([1, 1], dtype=np.int32),
    )

    def fake_compute_variant_statistics(raw_genotypes, config):
        nonlocal captured_config
        captured_config = config
        assert raw_genotypes.shape == (2, 2)
        return expected_stats

    monkeypatch.setattr(io_module, "compute_variant_statistics", fake_compute_variant_statistics)

    config = ModelConfig(minimum_scale=0.123, minimum_minor_allele_frequency=0.2)
    dataset = load_dataset_from_files(
        genotype_path=bed_path,
        genotype_format="plink1",
        sample_table_path=sample_table_path,
        target_column="target",
        covariate_columns=(),
        config=config,
    )

    assert captured_config is config
    assert dataset.variant_stats is expected_stats
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
            config=ModelConfig(),
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
        config=ModelConfig(
            trait_type=TraitType.QUANTITATIVE,
            max_outer_iterations=2,
        ),
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
    assert outputs.summary_path.name == "summary.json.gz"
    assert outputs.predictions_path.name == "predictions.tsv.gz"
    assert outputs.coefficients_path.name == "coefficients.tsv.gz"

    summary_payload = _read_json_payload(outputs.summary_path)
    assert summary_payload["sample_count"] == 6
    assert summary_payload["variant_count"] == 3
    assert summary_payload["trait_type"] == "quantitative"
    assert "training_r2" in summary_payload

    with gzip.open(outputs.predictions_path, "rt", encoding="utf-8") as handle:
        prediction_lines = handle.read().strip().splitlines()
    with gzip.open(outputs.coefficients_path, "rt", encoding="utf-8") as handle:
        coefficient_lines = handle.read().strip().splitlines()
    assert len(prediction_lines) == 7
    assert coefficient_lines[0] == "variant_id\tvariant_class\tbeta"
    for coefficient_line in coefficient_lines[1:]:
        assert float(coefficient_line.split("\t")[2]) != 0.0


def test_run_training_pipeline_rejects_mismatched_precomputed_stats_config(tmp_path: Path):
    dataset = io_module.LoadedDataset(
        sample_ids=["sample_0"],
        genotypes=np.zeros((1, 1), dtype=np.float32),
        covariates=np.zeros((1, 0), dtype=np.float32),
        targets=np.zeros(1, dtype=np.float32),
        variant_records=[],
        variant_stats=VariantStatistics(
            means=np.zeros(1, dtype=np.float32),
            scales=np.ones(1, dtype=np.float32),
            allele_frequencies=np.zeros(1, dtype=np.float32),
            support_counts=np.zeros(1, dtype=np.int32),
        ),
        variant_stats_minimum_scale=0.25,
    )

    with pytest.raises(ValueError, match="minimum_scale"):
        run_training_pipeline(
            dataset=dataset,
            config=ModelConfig(minimum_scale=0.5),
            output_dir=tmp_path / "run",
        )


def test_run_training_pipeline_fits_full_cohort_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fit_calls: list[dict[str, object]] = []

    class FakeBayesianPGS:
        def __init__(self, config: ModelConfig) -> None:
            self.config = config
            self.state = None

        def fit(
            self,
            genotypes,
            covariates,
            targets,
            variant_records,
            validation_data=None,
            variant_stats=None,
        ):
            fit_calls.append(
                {
                    "sample_count": int(genotypes.shape[0]),
                    "variant_stats_is_none": variant_stats is None,
                    "validation_sample_count": None if validation_data is None else int(validation_data[0].shape[0]),
                    "max_outer_iterations": int(self.config.max_outer_iterations),
                }
            )
            self.state = SimpleNamespace(
                active_variant_indices=np.array([0], dtype=np.int32),
                fit_result=SimpleNamespace(
                    validation_history=[],
                    selected_iteration_count=int(self.config.max_outer_iterations),
                ),
            )
            return self

        def export(self, path: Path) -> None:
            path.mkdir(parents=True, exist_ok=True)

        def coefficient_table(self, *, nonzero_only: bool = False, minimum_abs_beta: float = 0.0):
            assert nonzero_only is True
            assert minimum_abs_beta == 0.0
            return [{"variant_id": "variant_0", "variant_class": "snv", "beta": 0.75}]

        def training_decision_components(self):
            return (
                np.full(8, 0.2, dtype=np.float32),
                np.full(8, 0.1, dtype=np.float32),
            )

        def decision_components(self, genotypes, covariates):
            sample_count = int(genotypes.shape[0])
            return (
                np.full(sample_count, 0.2, dtype=np.float32),
                np.full(sample_count, 0.1, dtype=np.float32),
            )

    monkeypatch.setattr(io_module, "BayesianPGS", FakeBayesianPGS)

    dataset = io_module.LoadedDataset(
        sample_ids=[f"sample_{index}" for index in range(8)],
        genotypes=np.zeros((8, 1), dtype=np.float32),
        covariates=np.zeros((8, 0), dtype=np.float32),
        targets=np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        variant_records=[VariantRecord("variant_0", VariantClass.SNV, "1", 100)],
        variant_stats=VariantStatistics(
            means=np.zeros(1, dtype=np.float32),
            scales=np.ones(1, dtype=np.float32),
            allele_frequencies=np.array([0.25], dtype=np.float32),
            support_counts=np.full(1, 8, dtype=np.int32),
        ),
        variant_stats_minimum_scale=ModelConfig().minimum_scale,
    )

    outputs = run_training_pipeline(
        dataset=dataset,
        config=ModelConfig(
            trait_type=TraitType.BINARY,
            max_outer_iterations=9,
        ),
        output_dir=tmp_path / "run_validation",
    )

    assert fit_calls == [
        {
            "sample_count": 8,
            "variant_stats_is_none": False,
            "validation_sample_count": None,
            "max_outer_iterations": 9,
        },
    ]

    summary_payload = _read_json_payload(outputs.summary_path)
    assert summary_payload["validation_enabled"] is False
    assert summary_payload["tuning_sample_count"] == 8
    assert summary_payload["validation_sample_count"] == 0
    assert summary_payload["selected_iteration_count"] == 9
    assert summary_payload["fit_max_outer_iterations"] == 9
    assert summary_payload["validation_history"] == []


def test_run_training_pipeline_uses_validation_split_then_refits_full_cohort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fit_calls: list[dict[str, object]] = []

    class FakeBayesianPGS:
        def __init__(self, config: ModelConfig) -> None:
            self.config = config
            self.state = None

        def fit(
            self,
            genotypes,
            covariates,
            targets,
            variant_records,
            validation_data=None,
            variant_stats=None,
        ):
            fit_calls.append(
                {
                    "sample_count": int(genotypes.shape[0]),
                    "variant_stats_is_none": variant_stats is None,
                    "validation_sample_count": None if validation_data is None else int(validation_data[0].shape[0]),
                    "max_outer_iterations": int(self.config.max_outer_iterations),
                }
            )
            selected_iteration_count = 4 if validation_data is not None else int(self.config.max_outer_iterations)
            self.state = SimpleNamespace(
                active_variant_indices=np.array([0], dtype=np.int32),
                fit_result=SimpleNamespace(
                    validation_history=[0.11, 0.22] if validation_data is not None else [],
                    selected_iteration_count=selected_iteration_count,
                ),
            )
            return self

        def export(self, path: Path) -> None:
            path.mkdir(parents=True, exist_ok=True)

        def coefficient_table(self, *, nonzero_only: bool = False, minimum_abs_beta: float = 0.0):
            assert nonzero_only is True
            assert minimum_abs_beta == 0.0
            return [{"variant_id": "variant_0", "variant_class": "snv", "beta": 0.75}]

        def training_decision_components(self):
            return (
                np.full(10, 0.2, dtype=np.float32),
                np.full(10, 0.1, dtype=np.float32),
            )

        def decision_components(self, genotypes, covariates):
            sample_count = int(genotypes.shape[0])
            return (
                np.full(sample_count, 0.2, dtype=np.float32),
                np.full(sample_count, 0.1, dtype=np.float32),
            )

    monkeypatch.setattr(io_module, "BayesianPGS", FakeBayesianPGS)

    dataset = io_module.LoadedDataset(
        sample_ids=[f"sample_{index}" for index in range(10)],
        genotypes=np.zeros((10, 1), dtype=np.float32),
        covariates=np.zeros((10, 0), dtype=np.float32),
        targets=np.array([0.0, 1.0] * 5, dtype=np.float32),
        variant_records=[VariantRecord("variant_0", VariantClass.SNV, "1", 100)],
        variant_stats=VariantStatistics(
            means=np.zeros(1, dtype=np.float32),
            scales=np.ones(1, dtype=np.float32),
            allele_frequencies=np.array([0.25], dtype=np.float32),
            support_counts=np.full(1, 10, dtype=np.int32),
        ),
        variant_stats_minimum_scale=ModelConfig().minimum_scale,
    )

    outputs = run_training_pipeline(
        dataset=dataset,
        config=ModelConfig(
            trait_type=TraitType.BINARY,
            max_outer_iterations=9,
            pipeline_validation_fraction=0.2,
            pipeline_validation_min_samples=2,
            random_seed=3,
        ),
        output_dir=tmp_path / "run_with_validation",
    )

    assert fit_calls == [
        {
            "sample_count": 8,
            "variant_stats_is_none": True,
            "validation_sample_count": 2,
            "max_outer_iterations": 9,
        },
        {
            "sample_count": 10,
            "variant_stats_is_none": False,
            "validation_sample_count": None,
            "max_outer_iterations": 4,
        },
    ]

    summary_payload = _read_json_payload(outputs.summary_path)
    assert summary_payload["validation_enabled"] is True
    assert summary_payload["tuning_sample_count"] == 8
    assert summary_payload["validation_sample_count"] == 2
    assert summary_payload["selected_iteration_count"] == 4
    assert summary_payload["fit_max_outer_iterations"] == 4
    assert summary_payload["validation_history"] == [0.11, 0.22]


def test_write_predictions_and_summary_binary_uses_single_decision_pass(tmp_path: Path):
    class FakeBinaryModel:
        def __init__(self) -> None:
            self.config = ModelConfig(trait_type=TraitType.BINARY)
            self.decision_component_calls = 0

        def decision_components(self, genotypes, covariates):
            self.decision_component_calls += 1
            return (
                np.array([0.5, -0.25], dtype=np.float32),
                np.array([0.25, 0.75], dtype=np.float32),
            )

        def predict_proba(self, genotypes, covariates):
            raise AssertionError("predict_proba should not be called by _write_predictions_and_summary")

        def predict(self, genotypes, covariates):
            raise AssertionError("predict should not be called by _write_predictions_and_summary")

    dataset = io_module.LoadedDataset(
        sample_ids=["sample_0", "sample_1"],
        genotypes=np.zeros((2, 0), dtype=np.float32),
        covariates=np.zeros((2, 0), dtype=np.float32),
        targets=np.array([1.0, 0.0], dtype=np.float32),
        variant_records=[],
    )
    model = FakeBinaryModel()
    predictions_path = tmp_path / "predictions.tsv"

    summary = io_module._write_predictions_and_summary(
        predictions_path=predictions_path,
        dataset=dataset,
        model=model,
    )

    assert model.decision_component_calls == 1
    prediction_rows = _read_tsv_rows(predictions_path)
    assert [float(row["probability"]) for row in prediction_rows] == pytest.approx([
        1.0 / (1.0 + np.exp(-0.75)),
        1.0 / (1.0 + np.exp(-0.5)),
    ])
    assert [row["predicted_label"] for row in prediction_rows] == ["1", "1"]
    assert summary["training_accuracy"] == pytest.approx(0.5)


def test_write_predictions_and_summary_quantitative_uses_single_decision_pass(tmp_path: Path):
    class FakeQuantitativeModel:
        def __init__(self) -> None:
            self.config = ModelConfig(trait_type=TraitType.QUANTITATIVE)
            self.decision_component_calls = 0

        def decision_components(self, genotypes, covariates):
            self.decision_component_calls += 1
            return (
                np.array([1.25, -0.5], dtype=np.float32),
                np.array([0.25, 0.5], dtype=np.float32),
            )

        def predict(self, genotypes, covariates):
            raise AssertionError("predict should not be called by _write_predictions_and_summary")

        def predict_proba(self, genotypes, covariates):
            raise AssertionError("predict_proba should not be called by _write_predictions_and_summary")

    dataset = io_module.LoadedDataset(
        sample_ids=["sample_0", "sample_1"],
        genotypes=np.zeros((2, 0), dtype=np.float32),
        covariates=np.zeros((2, 0), dtype=np.float32),
        targets=np.array([1.0, 0.0], dtype=np.float32),
        variant_records=[],
    )
    model = FakeQuantitativeModel()
    predictions_path = tmp_path / "predictions.tsv"

    summary = io_module._write_predictions_and_summary(
        predictions_path=predictions_path,
        dataset=dataset,
        model=model,
    )

    assert model.decision_component_calls == 1
    prediction_rows = _read_tsv_rows(predictions_path)
    assert [float(row["prediction"]) for row in prediction_rows] == pytest.approx([1.5, 0.0])
    assert summary["training_r2"] == pytest.approx(0.5)
    assert summary["training_rmse"] == pytest.approx(np.sqrt(0.125))


def test_write_predictions_and_summary_uses_cached_training_scores_without_rescoring(tmp_path: Path):
    class FakeCachedModel:
        def __init__(self) -> None:
            self.config = ModelConfig(trait_type=TraitType.BINARY)

        def training_decision_components(self):
            return (
                np.array([0.2, -0.4], dtype=np.float32),
                np.array([0.3, 0.1], dtype=np.float32),
            )

        def decision_components(self, genotypes, covariates):
            raise AssertionError("decision_components should not be called when cached training scores exist")

    dataset = io_module.LoadedDataset(
        sample_ids=["sample_0", "sample_1"],
        genotypes=np.zeros((2, 3), dtype=np.float32),
        covariates=np.zeros((2, 1), dtype=np.float32),
        targets=np.array([1.0, 0.0], dtype=np.float32),
        variant_records=[],
    )
    predictions_path = tmp_path / "predictions.tsv"

    io_module._write_predictions_and_summary(
        predictions_path=predictions_path,
        dataset=dataset,
        model=FakeCachedModel(),
    )

    prediction_rows = _read_tsv_rows(predictions_path)
    assert [float(row["linear_predictor"]) for row in prediction_rows] == pytest.approx([0.5, -0.3])


def test_run_training_pipeline_keeps_full_coefficient_alignment_after_filtering(tmp_path: Path):
    config = ModelConfig(
        trait_type=TraitType.QUANTITATIVE,
        max_outer_iterations=2,
        minimum_minor_allele_frequency=0.2,
    )
    genotype_matrix = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    covariates = np.array(
        [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]],
        dtype=np.float32,
    )
    targets = np.array([0.1, 1.2, 1.7, 1.1, 0.0, 2.8], dtype=np.float32)
    variant_records = [
        VariantRecord("rare_filtered", VariantClass.SNV, "1", 100),
        VariantRecord("common_keep_1", VariantClass.DELETION_SHORT, "1", 200, length=400.0),
        VariantRecord("zero_filtered", VariantClass.SNV, "1", 300),
        VariantRecord("common_keep_2", VariantClass.DUPLICATION_SHORT, "1", 400, length=900.0),
    ]
    variant_stats = io_module.compute_variant_statistics(
        io_module.as_raw_genotype_matrix(genotype_matrix),
        config,
    )
    dataset = io_module.LoadedDataset(
        sample_ids=[f"sample_{index}" for index in range(genotype_matrix.shape[0])],
        genotypes=genotype_matrix,
        covariates=covariates,
        targets=targets,
        variant_records=variant_records,
        variant_stats=variant_stats,
        variant_stats_minimum_scale=config.minimum_scale,
    )

    def fake_fit_variational_em(
        genotypes,
        covariates,
        targets,
        records,
        tie_map,
        config,
        validation_data,
        resume_checkpoint=None,
        checkpoint_callback=None,
        predictor_offset=None,
        validation_offset=None,
    ):
        assert [record.variant_id for record in records] == ["common_keep_1", "common_keep_2"]
        return model_module.VariationalFitResult(
            alpha=np.zeros(covariates.shape[1], dtype=np.float32),
            beta_reduced=np.array([1.25, -0.5], dtype=np.float32),
            beta_variance=np.ones(genotypes.shape[1], dtype=np.float32),
            prior_scales=np.ones(len(records), dtype=np.float32),
            global_scale=1.0,
            class_tpb_shape_a=dict(config.class_tpb_shape_a()),
            class_tpb_shape_b=dict(config.class_tpb_shape_b()),
            scale_model_coefficients=np.zeros(1, dtype=np.float32),
            scale_model_feature_names=["intercept"],
            sigma_error2=1.0,
            objective_history=[0.0],
            validation_history=[],
            member_prior_variances=np.ones(len(records), dtype=np.float32),
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(model_module, "fit_variational_em", fake_fit_variational_em)

    try:
        outputs = run_training_pipeline(
            dataset=dataset,
            config=config,
            output_dir=tmp_path / "run_filtered",
        )
    finally:
        monkeypatch.undo()

    with gzip.open(outputs.coefficients_path, "rt", encoding="utf-8", newline="") as handle:
        coefficient_rows = list(csv.DictReader(handle, delimiter="\t"))

    assert [row["variant_id"] for row in coefficient_rows] == [
        "common_keep_1",
        "common_keep_2",
    ]
    assert [row["variant_class"] for row in coefficient_rows] == [
        "deletion_short",
        "duplication_short",
    ]
    assert float(coefficient_rows[0]["beta"]) == pytest.approx(1.25)
    assert float(coefficient_rows[1]["beta"]) == pytest.approx(-0.5)

    loaded_model = BayesianPGS.load(outputs.artifact_dir)
    loaded_rows = loaded_model.coefficient_table()
    assert [row["variant_id"] for row in loaded_rows] == [record.variant_id for record in variant_records]
    assert [row["variant_class"] for row in loaded_rows] == [
        record.variant_class.value for record in variant_records
    ]
    assert [float(row["beta"]) for row in loaded_rows] == pytest.approx([0.0, 1.25, 0.0, -0.5])


def test_cli_infers_binary_trait_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
    exit_code = _run_cli_without_gpu_runtime(
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
        ],
    )

    assert exit_code == 0
    summary_payload = _read_json_payload(output_dir / "summary.json.gz")
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
    exit_code = _run_cli_without_gpu_runtime(
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
        ],
    )

    assert exit_code == 0
    summary_payload = _read_json_payload(output_dir / "summary.json.gz")
    assert summary_payload["trait_type"] == "binary"
    assert summary_payload["training_auc"] is not None
    assert summary_payload["training_auc"] == pytest.approx(0.7313368055555556)

    dataset = load_dataset_from_files(
        genotype_path=vcf_path,
        config=ModelConfig(),
        genotype_format="vcf",
        sample_table_path=sample_table_path,
        sample_id_column="sample_id",
        target_column="target",
        covariate_columns=("age",),
        variant_metadata_path=metadata_path,
    )
    loaded_model = BayesianPGS.load(output_dir / "artifact")
    loaded_probability = loaded_model.predict_proba(dataset.genotypes, dataset.covariates)[:, 1]
    assert roc_auc_score(dataset.targets, loaded_probability) == pytest.approx(0.7313368055555556)

    prediction_rows = _read_tsv_rows(output_dir / "predictions.tsv.gz")
    file_probability = np.asarray([float(row["probability"]) for row in prediction_rows], dtype=np.float32)
    np.testing.assert_allclose(file_probability, loaded_probability, atol=1e-5)


def test_plink_end_to_end_recovers_quantitative_signal_with_sv_style_alleles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
    exit_code = _run_cli_without_gpu_runtime(
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
        ],
    )

    assert exit_code == 0
    summary_payload = _read_json_payload(output_dir / "summary.json.gz")
    assert summary_payload["trait_type"] == "quantitative"
    assert summary_payload["training_r2"] > 0.55

    dataset = load_dataset_from_files(
        genotype_path=bed_path,
        config=ModelConfig(),
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

    prediction_rows = _read_tsv_rows(output_dir / "predictions.tsv.gz")
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
            config=ModelConfig(),
            genotype_format="vcf",
            sample_table_path=sample_table_path,
            sample_id_column="sample_id",
            target_column="target",
            covariate_columns=(),
        )


def test_cli_handles_single_class_binary_targets_without_metric_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
    exit_code = _run_cli_without_gpu_runtime(
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
        ],
    )

    assert exit_code == 0
    summary_payload = _read_json_payload(output_dir / "summary.json.gz")
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
        config=ModelConfig(),
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
        config=ModelConfig(),
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
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [{str(key): str(value) for key, value in row.items()} for row in reader]


def _read_json_payload(path: Path) -> dict[str, object]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


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
