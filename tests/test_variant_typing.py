"""Prior classes are mutation kinds at every size; length never splits a class."""
from __future__ import annotations

import pytest

from sv_pgs.config import VariantClass
from sv_pgs.variant_typing import (
    COMMUNITY_SV_MINIMUM_LENGTH,
    sequence_resolved_class_and_length,
    sequence_resolved_kind_and_length,
    structural_variant_class_from_token,
    variant_class_and_length,
)


@pytest.mark.parametrize("deleted", [1, COMMUNITY_SV_MINIMUM_LENGTH - 1, COMMUNITY_SV_MINIMUM_LENGTH, 5_000])
def test_a_deletion_has_one_class_at_every_size(deleted: int) -> None:
    assert sequence_resolved_class_and_length("A" + "C" * deleted, "A") == (VariantClass.DELETION, float(deleted))


@pytest.mark.parametrize("inserted", [1, COMMUNITY_SV_MINIMUM_LENGTH - 1, COMMUNITY_SV_MINIMUM_LENGTH, 5_000])
def test_an_insertion_has_one_class_at_every_size(inserted: int) -> None:
    assert sequence_resolved_class_and_length("G", "G" + "T" * inserted) == (VariantClass.INSERTION, float(inserted))


def test_single_bases_are_snvs_and_balanced_replacements_are_complex() -> None:
    assert sequence_resolved_class_and_length("A", "G") == (VariantClass.SNV, 1.0)
    assert sequence_resolved_class_and_length("AC", "GT") == (VariantClass.OTHER_COMPLEX_SV, 2.0)


def test_mobile_elements_are_their_own_insertion_class() -> None:
    assert structural_variant_class_from_token("INS:ME:ALU") == VariantClass.INSERTION_MEI
    assert structural_variant_class_from_token("INS") == VariantClass.INSERTION
    assert structural_variant_class_from_token("DEL:ME:L1") == VariantClass.DELETION
    for alt, expected in (("<INS:ME:ALU>", VariantClass.INSERTION_MEI), ("<INS>", VariantClass.INSERTION)):
        typed = variant_class_and_length(pos=2000, ref="N", alt=alt, svtype="INS", svlen=300.0, info_end=2001)
        assert typed == (expected, 300.0)


def test_external_sv_matching_keeps_the_producers_size_definition() -> None:
    below = "A" + "C" * (COMMUNITY_SV_MINIMUM_LENGTH - 1)
    at = "A" + "C" * COMMUNITY_SV_MINIMUM_LENGTH
    assert sequence_resolved_kind_and_length(below, "A")[0] == "CPX"
    assert sequence_resolved_kind_and_length(at, "A") == ("DEL", float(COMMUNITY_SV_MINIMUM_LENGTH))
