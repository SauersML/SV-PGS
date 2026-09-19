"""The bcftools region worker must emit every bi-allelic record exactly once.

`bcftools view -r` returns records that overlap a region, not only records
whose POS lies in it, and several records can share a POS. Both matter at
region boundaries and when a killed worker resumes from its checkpoint. The
worker must also parse VCFs whose header declares none of its optional INFO
tags.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import sv_pgs.io as io_module
from sv_pgs.config import VariantClass

pytestmark = pytest.mark.skipif(shutil.which("bcftools") is None, reason="bcftools is not on PATH")

_HEADER = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=chr1,length=1000>\n"
    '##INFO=<ID=SVTYPE,Number=1,Type=String,Description="type">\n'
    '##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="length">\n'
    '##INFO=<ID=AF,Number=A,Type=Float,Description="frequency">\n'
    '##INFO=<ID=END,Number=1,Type=Integer,Description="end">\n'
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
)
# v2 is a 21 bp REF starting at 95, so it overlaps every region that starts
# in 96..115; v3-v5 share POS 100; v6 is multi-allelic.
_RECORDS = (
    ("50", "v1", "A", "C", "0/1", "0/0"),
    ("95", "v2", "ACGTACGTACGTACGTACGTA", "A", "1/1", "0/1"),
    ("100", "v3", "A", "C", "0/0", "0/1"),
    ("100", "v4", "A", "G", "0/1", "0/1"),
    ("100", "v5", "A", "T", "1/1", "0/0"),
    ("150", "v6", "C", "G,T", "0/1", "0/2"),
    ("150", "v7", "C", "A", "0/0", "1/1"),
    ("160", "v8", "G", "A", "0/1", "0/0"),
)
_EXPECTED_IDS = ["v1", "v2", "v3", "v4", "v5", "v7", "v8"]
_EXPECTED_DOSAGES = np.array([[1, 0], [2, 1], [0, 1], [1, 1], [2, 0], [0, 2], [1, 0]], dtype=np.int8)


def _indexed_vcf(tmp_path: Path) -> Path:
    text_path = tmp_path / "chr1.vcf"
    lines = [
        f"chr1\t{pos}\t{variant_id}\t{ref}\t{alt}\t.\t.\t.\tGT\t{first}\t{second}\n"
        for pos, variant_id, ref, alt, first, second in _RECORDS
    ]
    text_path.write_text(_HEADER + "".join(lines), encoding="utf-8")
    vcf_path = tmp_path / "chr1.vcf.gz"
    subprocess.run(
        ["bcftools", "view", "-Oz", "-o", str(vcf_path), "--write-index", str(text_path)],
        check=True,
    )
    return vcf_path


def _region_output(prefix: Path) -> tuple[list[str], np.ndarray]:
    variants = io_module._load_variant_metadata(Path(f"{prefix}.variants.npz"))
    dosages = np.fromfile(f"{prefix}.geno", dtype=np.int8).reshape(len(variants), 2)
    return [variant.variant_id for variant in variants], dosages


def test_adjacent_regions_emit_a_spanning_record_once(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    vcf_path = _indexed_vcf(tmp_path)
    first_prefix = tmp_path / "region_0"
    second_prefix = tmp_path / "region_1"

    first_result = io_module._region_parse_worker((str(vcf_path), "chr1:1-99", str(first_prefix), 1))
    second_result = io_module._region_parse_worker((str(vcf_path), "chr1:100-1000", str(second_prefix), 1))

    first_ids, first_dosages = _region_output(first_prefix)
    second_ids, second_dosages = _region_output(second_prefix)
    assert first_ids + second_ids == _EXPECTED_IDS
    np.testing.assert_array_equal(np.vstack([first_dosages, second_dosages]), _EXPECTED_DOSAGES)
    assert "skipped 1 records (multi-allelic sequence: 1)" in capsys.readouterr().err
    assert first_result == (2, str(first_prefix), {})
    assert second_result == (5, str(second_prefix), {"multi-allelic sequence": 1})
    # A finished region reports its stored counts without re-parsing.
    assert io_module._region_parse_worker((str(vcf_path), "chr1:100-1000", str(second_prefix), 1)) == (
        0,
        str(second_prefix),
        {"multi-allelic sequence": 1},
    )


def test_resume_keeps_records_sharing_the_checkpointed_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vcf_path = _indexed_vcf(tmp_path)
    prefix = tmp_path / "region_0"
    build_defaults = io_module._variant_defaults_from_bcftools_fields
    calls = {"count": 0}

    def crash_on_fourth_record(**fields: object) -> object:
        calls["count"] += 1
        if calls["count"] == 4:
            raise RuntimeError("worker killed")
        return build_defaults(**fields)

    # The crash lands on v4: v1-v3 are checkpointed and v3 shares POS 100
    # with the two records that have not been written yet.
    monkeypatch.setattr(io_module, "_variant_defaults_from_bcftools_fields", crash_on_fourth_record)
    with pytest.raises(RuntimeError, match="worker killed"):
        io_module._region_parse_worker((str(vcf_path), "chr1:1-1000", str(prefix), 1))
    monkeypatch.setattr(io_module, "_variant_defaults_from_bcftools_fields", build_defaults)

    result = io_module._region_parse_worker((str(vcf_path), "chr1:1-1000", str(prefix), 1))

    # The multi-allelic v6 follows the checkpoint, so it is counted once.
    assert result == (7, str(prefix), {"multi-allelic sequence": 1})
    variant_ids, dosages = _region_output(prefix)
    assert variant_ids == _EXPECTED_IDS
    np.testing.assert_array_equal(dosages, _EXPECTED_DOSAGES)


def test_header_without_optional_info_tags_is_parsed(tmp_path: Path) -> None:
    # Imputed / long-read VCFs such as the aou2_50k popped BCFs declare no
    # SVTYPE, SVLEN, AF or END; bcftools query rejects undeclared tags.
    text_path = tmp_path / "popped.vcf"
    text_path.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=1000>\n"
        '##INFO=<ID=ID,Number=1,Type=String,Description="atomic id">\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
        "chr1\t50\ta1\tA\tC\t.\t.\tID=a1\tGT\t0|1\t0|0\n"
        f"chr1\t95\ta2\tA{'CGT' * 40}\tA\t.\t.\tID=a2\tGT\t1|1\t0|1\n",
        encoding="utf-8",
    )
    vcf_path = tmp_path / "popped.vcf.gz"
    subprocess.run(["bcftools", "view", "-Oz", "-o", str(vcf_path), "--write-index", str(text_path)], check=True)
    prefix = tmp_path / "region_0"

    result = io_module._region_parse_worker((str(vcf_path), "chr1:1-1000", str(prefix), 1))

    assert result == (2, str(prefix), {})
    variants = io_module._load_variant_metadata(Path(f"{prefix}.variants.npz"))
    assert [(variant.variant_id, variant.variant_class, variant.length) for variant in variants] == [
        ("a1", VariantClass.SNV, 1.0),
        ("a2", VariantClass.DELETION, 120.0),
    ]
    assert [variant.allele_frequency for variant in variants] == [-1.0, -1.0]
