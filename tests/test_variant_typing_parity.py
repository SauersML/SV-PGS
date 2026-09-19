"""Both VCF parsers must type and size variants identically and correctly.

`_variant_defaults_from_bcftools_fields` documents that it mirrors
`_variant_defaults_from_vcf_record`; the AoU run uses the former and
`sv-pgs run` the latter, so a disagreement gives one record two priors.
"""
from __future__ import annotations

from pathlib import Path

import cyvcf2

import sv_pgs.io as io_module
from sv_pgs.config import VariantClass

_HEADER = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=chr1,length=100000>\n"
    '##INFO=<ID=SVTYPE,Number=1,Type=String,Description="type">\n'
    '##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="length">\n'
    '##INFO=<ID=AF,Number=A,Type=Float,Description="frequency">\n'
    '##INFO=<ID=END,Number=1,Type=Integer,Description="end">\n'
    '##ALT=<ID=DEL,Description="deletion">\n'
    '##ALT=<ID=CNV,Description="copy number variant">\n'
    '##ALT=<ID=INV,Description="inversion">\n'
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\n"
)
_SEQUENCE_DELETION = "A" + "CGT" * 40
# Two unrelated 80 bp alleles sharing only the anchor base: a complex SV.
_COMPLEX_REF = "T" + "ACGGT" * 15 + "ACGT"
_COMPLEX_ALT = "T" + "GTTCA" * 15 + "GTTC"
# (id, pos, ref, alt, INFO fields, expected class, expected length)
_RECORDS = (
    ("snv", 100, "A", "G", {"AF": "0.1"}, VariantClass.SNV, 1.0),
    ("small_deletion", 200, "ACG", "A", {"AF": "0.1"}, VariantClass.SMALL_INDEL, 2.0),
    ("substitution", 300, "AC", "GT", {"AF": "0.1"}, VariantClass.SMALL_INDEL, 2.0),
    ("sequence_deletion", 400, _SEQUENCE_DELETION, "A", {"SVTYPE": "DEL", "SVLEN": "-120", "AF": "0.1"}, VariantClass.DELETION, 120.0),
    ("symbolic_deletion", 900, "A", "<DEL>", {"SVTYPE": "DEL", "SVLEN": "-2000", "END": "2900", "AF": "0.1"}, VariantClass.DELETION, 2000.0),
    # GATK-SV multi-allelic CNVs are copy-number variation, not duplications,
    # and inversions have their own class.
    ("copy_number_variant", 1000, "N", "<CNV>", {"SVTYPE": "CNV", "SVLEN": "1064", "END": "2064", "AF": "0.3"}, VariantClass.COPY_NUMBER, 1064.0),
    ("inversion", 2500, "N", "<INV>", {"SVTYPE": "INV", "SVLEN": "400", "END": "2900", "AF": "0.1"}, VariantClass.INVERSION, 400.0),
    # Sequence-resolved SVs without SVTYPE/SVLEN, as long-read and imputed
    # panels write them: typed and sized from the alleles.
    ("untyped_deletion", 3000, _SEQUENCE_DELETION, "A", {}, VariantClass.DELETION, 120.0),
    ("untyped_long_deletion", 4000, "C" + "AGT" * 500, "C", {}, VariantClass.DELETION, 1500.0),
    ("untyped_insertion", 6000, "G", "G" + "ACGT" * 100, {}, VariantClass.INSERTION_MEI, 400.0),
    ("indel_49", 7000, "A" + "C" * 49, "A", {}, VariantClass.SMALL_INDEL, 49.0),
    ("deletion_50", 8000, "A" + "C" * 50, "A", {}, VariantClass.DELETION, 50.0),
    ("complex_80", 9000, _COMPLEX_REF, _COMPLEX_ALT, {}, VariantClass.OTHER_COMPLEX_SV, 79.0),
)


def _bcftools_field(info: dict[str, str], key: str) -> bytes:
    return info.get(key, ".").encode("utf-8")


def test_bcftools_and_cyvcf2_paths_assign_the_same_variant_class(tmp_path: Path) -> None:
    vcf_path = tmp_path / "typing.vcf"
    lines = [
        f"chr1\t{pos}\t{variant_id}\t{ref}\t{alt}\t50\t.\t"
        + (";".join(f"{key}={value}" for key, value in info.items()) or ".")
        + "\tGT\t0/1\n"
        for variant_id, pos, ref, alt, info, _, _ in _RECORDS
    ]
    vcf_path.write_text(_HEADER + "".join(lines), encoding="utf-8")

    reader = cyvcf2.VCF(str(vcf_path))
    cyvcf2_defaults = [io_module._variant_defaults_from_vcf_record(record) for record in reader]
    reader.close()
    bcftools_defaults = [
        io_module._variant_defaults_from_bcftools_fields(
            chrom="chr1",
            pos=pos,
            record_id_field=variant_id.encode("utf-8"),
            ref=ref,
            alt=alt,
            qual_field=b"50",
            svtype_field=_bcftools_field(info, "SVTYPE"),
            svlen_field=_bcftools_field(info, "SVLEN"),
            af_field=_bcftools_field(info, "AF"),
            end_field=_bcftools_field(info, "END"),
        )
        for variant_id, pos, ref, alt, info, _, _ in _RECORDS
    ]

    assert [defaults.variant_class for defaults in bcftools_defaults] == [
        defaults.variant_class for defaults in cyvcf2_defaults
    ]
    assert [defaults.length for defaults in bcftools_defaults] == [defaults.length for defaults in cyvcf2_defaults]
    assert [(defaults.variant_class, defaults.length) for defaults in bcftools_defaults] == [
        (expected_class, expected_length) for *_, expected_class, expected_length in _RECORDS
    ]
