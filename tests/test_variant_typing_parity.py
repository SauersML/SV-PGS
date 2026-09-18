"""The bcftools precache path must type variants like the cyvcf2 path.

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
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\n"
)
_SEQUENCE_DELETION = "A" + "CGT" * 40
# (id, pos, ref, alt, INFO fields)
_RECORDS = (
    ("snv", 100, "A", "G", {"AF": "0.1"}),
    ("small_deletion", 200, "ACG", "A", {"AF": "0.1"}),
    ("substitution", 300, "AC", "GT", {"AF": "0.1"}),
    ("sequence_deletion", 400, _SEQUENCE_DELETION, "A", {"SVTYPE": "DEL", "SVLEN": "-120", "AF": "0.1"}),
    ("symbolic_deletion", 900, "A", "<DEL>", {"SVTYPE": "DEL", "SVLEN": "-2000", "END": "2900", "AF": "0.1"}),
)


def _bcftools_field(info: dict[str, str], key: str) -> bytes:
    return info.get(key, ".").encode("utf-8")


def test_bcftools_and_cyvcf2_paths_assign_the_same_variant_class(tmp_path: Path) -> None:
    vcf_path = tmp_path / "typing.vcf"
    lines = [
        f"chr1\t{pos}\t{variant_id}\t{ref}\t{alt}\t50\t.\t"
        + ";".join(f"{key}={value}" for key, value in info.items())
        + "\tGT\t0/1\n"
        for variant_id, pos, ref, alt, info in _RECORDS
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
        for variant_id, pos, ref, alt, info in _RECORDS
    ]

    assert [defaults.variant_class for defaults in bcftools_defaults] == [
        defaults.variant_class for defaults in cyvcf2_defaults
    ]
    assert [defaults.length for defaults in bcftools_defaults] == [defaults.length for defaults in cyvcf2_defaults]
    assert [defaults.variant_class for defaults in bcftools_defaults] == [
        VariantClass.SNV,
        VariantClass.SMALL_INDEL,
        VariantClass.SMALL_INDEL,
        VariantClass.DELETION_SHORT,
        VariantClass.DELETION_LONG,
    ]
