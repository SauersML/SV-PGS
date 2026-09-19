"""GATK-SV store source: FILTER policy, copy-number path, classes and no-calls."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sv_pgs.config import VariantClass
from sv_pgs.gatksv_source import GatksvSource
from sv_pgs.sample_crosswalk import SampleCrosswalk, source_columns_for_store_samples

_HEADER_START = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=chr1,length=1000000>\n"
    '##FILTER=<ID=UNRESOLVED,Description="unresolved breakend">\n'
    '##FILTER=<ID=HIGH_NCR,Description="high no-call rate">\n'
    '##FILTER=<ID=VARIABLE_ACROSS_BATCHES,Description="batch effect">\n'
    '##FILTER=<ID=LIKELY_REFERENCE_ARTIFACT,Description="reference artifact">\n'
    '##FILTER=<ID=MULTIALLELIC,Description="multi-allelic CNV">\n'
    '##INFO=<ID=SVTYPE,Number=1,Type=String,Description="type">\n'
    '##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="length">\n'
    '##INFO=<ID=END,Number=1,Type=Integer,Description="end">\n'
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">\n'
    '##FORMAT=<ID=RD_CN,Number=1,Type=Integer,Description="read-depth copy state">\n'
)
_SAMPLES = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tR1\tR2\tR3\tR4\n"

# The AoU v8 layout: multi-allelic CNVs are multi-ALT <CN=k> records flagged
# MULTIALLELIC, with GT "." and an integer FORMAT/CN.
_AOU_LIKE_VCF = (
    _HEADER_START
    + '##FORMAT=<ID=CN,Number=1,Type=Integer,Description="copy number">\n'
    + _SAMPLES
    + "chr1\t1000\tdel_pass\tN\t<DEL>\t.\tPASS\tSVTYPE=DEL;SVLEN=500;END=1500\tGT:CN:RD_CN\t0/1:.:1\t./.:.:.\t./1:.:1\t1/1:.:0\n"
    + "chr1\t2000\tins_mei\tN\t<INS:ME:ALU>\t.\t.\tSVTYPE=INS;SVLEN=300;END=2001\tGT:CN:RD_CN\t0/0:.:2\t0/1:.:2\t0/0:.:2\t0/0:.:2\n"
    + "chr1\t3000\tbnd_unresolved\tN\tN]chr1:90000]\t.\tUNRESOLVED\tSVTYPE=BND\tGT:CN:RD_CN\t0/1:.:2\t0/0:.:2\t0/0:.:2\t0/0:.:2\n"
    + "chr1\t3500\tbnd_pass\tN\tN[chr1:95000[\t.\tPASS\tSVTYPE=BND\tGT:CN:RD_CN\t0/1:.:2\t0/0:.:2\t0/0:.:2\t0/0:.:2\n"
    + "chr1\t4000\tdel_ncr\tN\t<DEL>\t.\tHIGH_NCR;VARIABLE_ACROSS_BATCHES\tSVTYPE=DEL;SVLEN=800;END=4800\tGT:CN:RD_CN\t0/1:.:1\t0/0:.:2\t0/0:.:2\t0/0:.:2\n"
    + "chr1\t5000\tdup_artifact\tN\t<DUP>\t.\tLIKELY_REFERENCE_ARTIFACT\tSVTYPE=DUP;SVLEN=700;END=5700\tGT:CN:RD_CN\t0/1:.:3\t0/0:.:2\t0/0:.:2\t0/0:.:2\n"
    + "chr1\t6000\tmcnv\tN\t<CN=0>,<CN=2>,<CN=3>\t.\tMULTIALLELIC\tSVTYPE=CNV;SVLEN=9000;END=15000\tGT:CN:RD_CN\t.:2:2\t.:3:3\t.:.:.\t.:12:12\n"
    + "chr1\t20000\tmcnv_ncr\tN\t<CN=0>,<CN=2>\t.\tHIGH_NCR;MULTIALLELIC\tSVTYPE=CNV;SVLEN=6000;END=26000\tGT:CN:RD_CN\t.:2:2\t.:1:1\t.:2:2\t.:2:2\n"
    + "chr1\t30000\tinv\tN\t<INV>\t.\tPASS\tSVTYPE=INV;SVLEN=400;END=30400\tGT:CN:RD_CN\t0/0:.:2\t0/1:.:2\t0/0:.:2\t0/0:.:2\n"
    + "chr1\t40000\tcpx\tN\t<CPX>\t.\tPASS\tSVTYPE=CPX;SVLEN=2500;END=42500\tGT:CN:RD_CN\t0/1:.:2\t0/0:.:2\t0/0:.:2\t0/1:.:2\n"
    + "chr1\t50000\tdup_long\tN\t<DUP>\t.\tPASS\tSVTYPE=DUP;SVLEN=5000;END=55000\tGT:CN:RD_CN\t0/0:.:2\t0/0:.:2\t1/1:.:4\t0/1:.:3\n"
    + "chr1\t60000\tmulti_del\tN\t<DEL>,<DUP>\t.\tPASS\tSVTYPE=DEL;SVLEN=600;END=60600\tGT:CN:RD_CN\t0/1:.:1\t0/0:.:2\t0/0:.:2\t0/2:.:3\n"
)

# The public 1kGP freeze V3 layout: single-ALT <CNV> with a placeholder GT and
# a String FORMAT/CN; RD_CN covers samples whose CN is ".".
_KGP_LIKE_VCF = (
    _HEADER_START
    + '##FORMAT=<ID=CN,Number=A,Type=String,Description="copy number">\n'
    + _SAMPLES
    + "chr1\t1000\tcnv_kgp\tN\t<CNV>\t.\tPASS\tSVTYPE=CNV;SVLEN=1064;END=2064\tGT:CN:RD_CN\t0/0:2:2\t0/1:5:5\t0/1:.:3\t0/1:0:0\n"
    + "chr1\t5000\tdel_kgp\tN\t<DEL>\t.\tPASS\tSVTYPE=DEL;SVLEN=1500;END=6500\tGT:CN:RD_CN\t0/1:.:1\t0/0:.:2\t1/1:.:0\t0/0:.:2\n"
)


def _read_all(tmp_path: Path, text: str, block_records: int) -> tuple[GatksvSource, list]:
    vcf_path = tmp_path / "gatksv.vcf"
    vcf_path.write_text(text, encoding="utf-8")
    source = GatksvSource(vcf_path)
    return source, list(source.blocks(block_records))


def test_aou_layout_filter_policy_classes_and_copy_number(tmp_path: Path) -> None:
    source, blocks = _read_all(tmp_path, _AOU_LIKE_VCF, block_records=2)

    assert [block.record_count for block in blocks] == [2, 2, 2]
    kept_ids = [variant_id for block in blocks for variant_id in block.variant_ids]
    assert kept_ids == ["del_pass", "ins_mei", "mcnv", "inv", "cpx", "dup_long"]
    assert [variant_class for block in blocks for variant_class in block.variant_classes] == [
        VariantClass.DELETION_SHORT,
        VariantClass.INSERTION_MEI,
        VariantClass.COPY_NUMBER,
        VariantClass.INVERSION,
        VariantClass.OTHER_COMPLEX_SV,
        VariantClass.DUPLICATION_LONG,
    ]
    assert [svtype for block in blocks for svtype in block.svtypes] == ["DEL", "INS", "CNV", "INV", "CPX", "DUP"]
    assert np.concatenate([block.is_copy_number for block in blocks]).tolist() == [False, False, True, False, False, False]
    assert np.concatenate([block.lengths for block in blocks]).tolist() == [500.0, 300.0, 9000.0, 400.0, 2500.0, 5000.0]
    values = np.vstack([block.values for block in blocks])
    no_call = np.vstack([block.no_call for block in blocks])
    # del_pass: "./1" is a no-call, not a het; mcnv: CN, with R3's "." a no-call.
    np.testing.assert_array_equal(values[0], [1, 0, 0, 2])
    np.testing.assert_array_equal(no_call[0], [False, True, True, False])
    np.testing.assert_array_equal(values[2], [2, 3, 0, 12])
    np.testing.assert_array_equal(no_call[2], [False, False, True, False])
    assert source.skipped_records == {
        "FILTER UNRESOLVED": 1,
        "breakend": 1,
        "FILTER HIGH_NCR;VARIABLE_ACROSS_BATCHES": 1,
        "FILTER LIKELY_REFERENCE_ARTIFACT": 1,
        "FILTER HIGH_NCR;MULTIALLELIC": 1,
        "multi-allelic DEL": 1,
    }
    np.testing.assert_array_equal(source.sample_no_call_counts, [0, 1, 2, 0])
    np.testing.assert_allclose(source.sample_no_call_rates(), [0.0, 1 / 6, 2 / 6, 0.0])


def test_kgp_layout_reads_copy_number_not_the_placeholder_genotype(tmp_path: Path) -> None:
    source, blocks = _read_all(tmp_path, _KGP_LIKE_VCF, block_records=10)

    (block,) = blocks
    assert block.variant_classes == (VariantClass.COPY_NUMBER, VariantClass.DELETION_LONG)
    assert block.svtypes == ("CNV", "DEL")
    assert block.is_copy_number.tolist() == [True, False]
    # CN, falling back to RD_CN where CN is "." (sample R3); the GT is ignored.
    np.testing.assert_array_equal(block.values[0], [2, 5, 3, 0])
    assert not block.no_call.any()
    np.testing.assert_array_equal(block.values[1], [1, 0, 2, 0])
    assert source.skipped_records == {}
    assert source.sample_ids == ("R1", "R2", "R3", "R4")


def test_copy_number_above_the_stored_range_is_dropped_and_counted(tmp_path: Path) -> None:
    # A satellite-scale array (1kGP HGSV_208635 reaches 652 copies) has no uint8 code.
    source, blocks = _read_all(tmp_path, _KGP_LIKE_VCF.replace("0/1:5:5", "0/1:652:652"), block_records=10)

    assert [variant_id for block in blocks for variant_id in block.variant_ids] == ["del_kgp"]
    assert source.skipped_records == {"copy number above 254": 1}
    assert source.kept_record_count == 1


def test_blocks_align_to_store_samples_through_the_crosswalk(tmp_path: Path) -> None:
    source, blocks = _read_all(tmp_path, _KGP_LIKE_VCF, block_records=10)
    crosswalk = SampleCrosswalk(research_ids=("R1", "R2", "R4"), sequencing_ids=("D1", "D2", "D4"))
    # Store order D4, D3, D1, D2: D3 has no crosswalk row, R3 no store sample.
    columns = source_columns_for_store_samples(["D4", "D3", "D1", "D2"], source.sample_ids, crosswalk)

    aligned = blocks[0].aligned_to_store_samples(columns)

    np.testing.assert_array_equal(aligned.values, [[0, 0, 2, 5], [0, 0, 1, 0]])
    np.testing.assert_array_equal(aligned.no_call, [[False, True, False, False], [False, True, False, False]])
    assert aligned.variant_ids == blocks[0].variant_ids
