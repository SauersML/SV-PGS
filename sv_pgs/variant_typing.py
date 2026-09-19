"""Variant class and length of a VCF/BIM record, shared by every genotype source.

A record is typed once, from its alleles and its SVTYPE/SVLEN/END fields, so the
same event gets the same prior class whichever source (imputed BCF, GATK-SV VCF,
PLINK) it comes from.
"""

from __future__ import annotations

from typing import Any

from sv_pgs.config import VariantClass

SV_LENGTH_THRESHOLD = 1_000.0
# SV size cutoff for sequence-resolved alleles: the imputation strata's rule
# (lrma-strata scripts/finalqc/strata_build.py var_class / classify).
SEQUENCE_RESOLVED_SV_MINIMUM_LENGTH = 50
_SEQUENCE_BASES = frozenset("ACGTNacgtn")


def alt_is_symbolic_or_breakend(alt: str) -> bool:
    return bool(alt) and (alt[0] == "<" or "[" in alt or "]" in alt)


def is_sequence_allele(value: str) -> bool:
    return bool(value) and all(base in _SEQUENCE_BASES for base in value)


def normalize_variant_token(value: Any) -> str | None:
    if value is None:
        return None
    normalized_value = str(value).strip().upper()
    if not normalized_value:
        return None
    if normalized_value.startswith("<") and normalized_value.endswith(">"):
        normalized_value = normalized_value[1:-1]
    return normalized_value


def structural_variant_class_from_token(token: str, length: float) -> VariantClass:
    """Class of an SV from its SVTYPE or symbolic-ALT token.

    CNV is copy-number variation, not a duplication; INV has its own class. A
    breakend (BND) keeps the legacy inversion/breakend/complex class; the
    GATK-SV store source drops breakends before typing.
    """
    if "DEL" in token:
        return VariantClass.DELETION_LONG if length >= SV_LENGTH_THRESHOLD else VariantClass.DELETION_SHORT
    if "CNV" in token:
        return VariantClass.COPY_NUMBER
    if "DUP" in token:
        return VariantClass.DUPLICATION_LONG if length >= SV_LENGTH_THRESHOLD else VariantClass.DUPLICATION_SHORT
    if "INS" in token or "ME" in token:
        return VariantClass.INSERTION_MEI
    if "INV" in token:
        return VariantClass.INVERSION
    if "BND" in token:
        return VariantClass.INVERSION_BND_COMPLEX
    if "STR" in token or "VNTR" in token or "REPEAT" in token:
        return VariantClass.STR_VNTR_REPEAT
    return VariantClass.OTHER_COMPLEX_SV


def trimmed_allele_cores(ref: str, alt: str) -> tuple[int, int, int]:
    """Shared-prefix length, then the REF and ALT lengths left after removing that prefix and the shared suffix."""
    shortest = min(len(ref), len(alt))
    prefix = 0
    while prefix < shortest and ref[prefix] == alt[prefix]:
        prefix += 1
    suffix = 0
    while suffix < shortest - prefix and ref[len(ref) - 1 - suffix] == alt[len(alt) - 1 - suffix]:
        suffix += 1
    return prefix, len(ref) - prefix - suffix, len(alt) - prefix - suffix


def sequence_resolved_kind_and_length(ref: str, alt: str) -> tuple[str, float]:
    """``DEL``, ``INS`` or ``CPX`` for a sequence-resolved allele pair, and its length.

    On the trimmed allele cores: a deletion when REF loses at least
    SEQUENCE_RESOLVED_SV_MINIMUM_LENGTH bases and the ALT core is at most
    max(10, 10% of the REF core), an insertion in the mirror case, otherwise
    complex. Length is the inserted/deleted core length, or the longer core
    for complex and equal-length alleles.
    """
    _, ref_core, alt_core = trimmed_allele_cores(ref, alt)
    deletion = ref_core - alt_core >= SEQUENCE_RESOLVED_SV_MINIMUM_LENGTH and alt_core <= max(10.0, 0.1 * ref_core)
    insertion = alt_core - ref_core >= SEQUENCE_RESOLVED_SV_MINIMUM_LENGTH and ref_core <= max(10.0, 0.1 * alt_core)
    if deletion or alt_core == 0:
        length = float(ref_core - alt_core)
    elif insertion or ref_core == 0:
        length = float(alt_core - ref_core)
    else:
        length = float(max(ref_core, alt_core))
    return ("DEL" if deletion else "INS" if insertion else "CPX"), length


def sequence_resolved_class_and_length(ref: str, alt: str) -> tuple[VariantClass, float]:
    """Type a sequence-resolved allele pair the way the imputation strata do.

    SV when max(len(REF), len(ALT)) - 1 >= SEQUENCE_RESOLVED_SV_MINIMUM_LENGTH,
    then typed by ``sequence_resolved_kind_and_length``. (Telling a DUP or an
    INV apart needs the reference sequence; supply it as variant_class in
    the variant metadata.)
    """
    if len(ref) == 1 and len(alt) == 1:
        return VariantClass.SNV, 1.0
    kind, length = sequence_resolved_kind_and_length(ref, alt)
    if max(len(ref), len(alt)) - 1 < SEQUENCE_RESOLVED_SV_MINIMUM_LENGTH:
        return VariantClass.SMALL_INDEL, length
    if kind == "CPX":
        return VariantClass.OTHER_COMPLEX_SV, length
    return structural_variant_class_from_token(kind, length), length


def variant_class_and_length(
    *,
    pos: int,
    ref: str,
    alt: str,
    svtype: str | None,
    svlen: float | None,
    info_end: int | None,
) -> tuple[VariantClass, float]:
    """Variant class and length of one bi-allelic record.

    A sequence-resolved allele pair (ACGTN REF and ALT) without SVTYPE is
    typed from its alleles by sequence_resolved_class_and_length. Records
    with SVTYPE, and symbolic/breakend ALTs, are typed from the SVTYPE (else
    the symbolic ALT) token; the token scan never runs on raw sequence.
    Length is |SVLEN| when present; symbolic alleles otherwise fall back to
    END - POS + 1, then the longest allele.
    """
    sequence_resolved = not alt_is_symbolic_or_breakend(alt) and is_sequence_allele(ref) and is_sequence_allele(alt)
    if sequence_resolved:
        sequence_class, sequence_length = sequence_resolved_class_and_length(ref, alt)
        length = float(abs(svlen)) if svlen is not None else sequence_length
        if svtype is None or sequence_class == VariantClass.SNV:
            return sequence_class, length
    elif svlen is not None:
        length = float(abs(svlen))
    elif info_end is not None and info_end >= pos:
        length = float(info_end - pos + 1)
    else:
        length = float(max(len(ref), len(alt)))
    if svtype is not None:
        variant_token = normalize_variant_token(svtype)
    elif alt_is_symbolic_or_breakend(alt):
        variant_token = normalize_variant_token(alt)
    else:
        variant_token = None
    if variant_token is None:
        return VariantClass.OTHER_COMPLEX_SV, length
    return structural_variant_class_from_token(variant_token, length), length
