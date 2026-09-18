"""Both VCF loaders keep only bi-allelic records that pass FILTER.

GATK-SV call sets (the AoU srWGS SV VCFs) carry non-PASS records such as
UNRESOLVED breakends and HIGH_NCR or VARIABLE_ACROSS_BATCHES calls, plus
multi-allelic CNVs whose copy number is in FORMAT/CN rather than GT. None of
them may enter the model as predictors, and each skip is counted by reason.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import sv_pgs.io as io_module
from sv_pgs.config import ModelConfig
from sv_pgs.io import load_dataset_from_files

_GATK_SV_LIKE_VCF = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=chr1,length=100000>\n"
    '##FILTER=<ID=PASS,Description="All filters passed">\n'
    '##FILTER=<ID=UNRESOLVED,Description="unresolved">\n'
    '##FILTER=<ID=HIGH_NCR,Description="high no-call rate">\n'
    '##FILTER=<ID=VARIABLE_ACROSS_BATCHES,Description="batch effect">\n'
    '##FILTER=<ID=MULTIALLELIC,Description="multi-allelic CNV">\n'
    '##INFO=<ID=SVTYPE,Number=1,Type=String,Description="type">\n'
    '##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="length">\n'
    '##INFO=<ID=AF,Number=A,Type=Float,Description="frequency">\n'
    '##INFO=<ID=END,Number=1,Type=Integer,Description="end">\n'
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">\n'
    '##FORMAT=<ID=CN,Number=1,Type=Integer,Description="copy number">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\tS2\n"
    "chr1\t100\tpass_del\tN\t<DEL>\t.\tPASS\tSVTYPE=DEL;SVLEN=-500;END=600\tGT:CN\t0/1:.\t0/0:.\t1/1:.\n"
    "chr1\t200\tunfiltered_ins\tN\t<INS>\t.\t.\tSVTYPE=INS;SVLEN=300\tGT:CN\t0/0:.\t0/1:.\t0/0:.\n"
    "chr1\t300\tunresolved_bnd\tN\tN]chr1:50000]\t.\tUNRESOLVED\tSVTYPE=BND\tGT:CN\t0/1:.\t0/1:.\t0/0:.\n"
    "chr1\t400\tncr_del\tN\t<DEL>\t.\tHIGH_NCR;VARIABLE_ACROSS_BATCHES\tSVTYPE=DEL;SVLEN=-800;END=1200\tGT:CN\t0/1:.\t1/1:.\t0/0:.\n"
    "chr1\t2000\tmcnv\tN\t<CN=0>,<CN=2>,<CN=3>\t.\tMULTIALLELIC\tSVTYPE=CNV;SVLEN=8000;END=10000\tGT:CN\t./.:1\t./.:2\t./.:3\n"
    "chr1\t20000\tpass_dup\tN\t<DUP>\t.\tPASS\tSVTYPE=DUP;SVLEN=2000;END=22000\tGT:CN\t0/1:.\t0/0:.\t0/0:.\n"
)
_KEPT_IDS = ["pass_del", "unfiltered_ins", "pass_dup"]
# (sample, variant) ALT dosages of the kept records.
_KEPT_DOSAGES = np.array([[1, 0, 1], [0, 1, 0], [2, 0, 0]], dtype=np.int8)
_SKIPPED = {
    "FILTER HIGH_NCR;VARIABLE_ACROSS_BATCHES": 1,
    "FILTER UNRESOLVED": 1,
    "multi-allelic CNV": 1,
}


def test_skipped_record_reason_labels() -> None:
    assert io_module._skipped_record_reason(alt_alleles=["<DEL>"], svtype="DEL", filter_value="PASS") is None
    assert io_module._skipped_record_reason(alt_alleles=["A"], svtype=None, filter_value=".") is None
    assert io_module._skipped_record_reason(alt_alleles=["<CN=0>", "<CN=2>"], svtype="CNV", filter_value="MULTIALLELIC") == "multi-allelic CNV"
    assert io_module._skipped_record_reason(alt_alleles=["<CN=0>", "<CN=2>"], svtype=None, filter_value="PASS") == "multi-allelic symbolic"
    assert io_module._skipped_record_reason(alt_alleles=["C", "G"], svtype=None, filter_value="PASS") == "multi-allelic sequence"
    assert io_module._skipped_record_reason(alt_alleles=["N]chr1:5]"], svtype="BND", filter_value="UNRESOLVED") == "FILTER UNRESOLVED"


@pytest.mark.skipif(shutil.which("bcftools") is None, reason="bcftools is not on PATH")
def test_precache_worker_keeps_only_passing_biallelic_records(tmp_path: Path) -> None:
    text_path = tmp_path / "gatk_sv.vcf"
    text_path.write_text(_GATK_SV_LIKE_VCF, encoding="utf-8")
    vcf_path = tmp_path / "gatk_sv.vcf.gz"
    subprocess.run(["bcftools", "view", "-Oz", "-o", str(vcf_path), "--write-index", str(text_path)], check=True)
    prefix = tmp_path / "region_0"

    result = io_module._region_parse_worker((str(vcf_path), "chr1:1-100000", str(prefix), 1))

    assert result == (3, str(prefix), _SKIPPED)
    variants = io_module._load_variant_metadata(Path(f"{prefix}.variants.npz"))
    assert [variant.variant_id for variant in variants] == _KEPT_IDS
    dosages = np.fromfile(f"{prefix}.geno", dtype=np.int8).reshape(len(variants), 3).T
    np.testing.assert_array_equal(dosages, _KEPT_DOSAGES)


def test_cyvcf2_loader_keeps_only_passing_biallelic_records(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    vcf_path = tmp_path / "gatk_sv.vcf"
    vcf_path.write_text(_GATK_SV_LIKE_VCF, encoding="utf-8")
    sample_table_path = tmp_path / "samples.tsv"
    sample_table_path.write_text("sample_id\ttarget\nS0\t0\nS1\t1\nS2\t0\n", encoding="utf-8")

    dataset = load_dataset_from_files(
        genotype_path=vcf_path,
        config=ModelConfig(),
        genotype_format="vcf",
        sample_table_path=sample_table_path,
        sample_id_column="sample_id",
        target_column="target",
        covariate_columns=(),
    )

    assert [record.variant_id for record in dataset.variant_records] == _KEPT_IDS
    np.testing.assert_array_equal(dataset.genotypes.materialize(), _KEPT_DOSAGES.astype(np.float32))
    assert (
        "skipped 3 records (FILTER HIGH_NCR;VARIABLE_ACROSS_BATCHES: 1, FILTER UNRESOLVED: 1, multi-allelic CNV: 1)"
        in capsys.readouterr().err
    )
