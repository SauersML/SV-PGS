from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from bed_reader import to_bed

from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.cli import main
from sv_pgs.io import load_dataset_from_files, run_training_pipeline


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
        header=("variant_id", "variant_class", "training_support", "is_copy_number"),
        rows=(
            ("rs1", "snv", "", "false"),
            ("sv1", "deletion_short", "2", "true"),
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
    assert dataset.variant_records[0].variant_class == VariantClass.SNV
    assert dataset.variant_records[1].variant_class == VariantClass.DELETION_SHORT
    assert dataset.variant_records[1].training_support == 2
    assert dataset.variant_records[1].is_copy_number is True


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
            minimum_structural_variant_carriers=1,
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
            "--minimum-structural-variant-carriers",
            "1",
        ]
    )

    assert exit_code == 0
    summary_payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["trait_type"] == "binary"


def _write_table(path: Path, header: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)
