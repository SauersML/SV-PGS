"""Regression tests for three VariantRecord public-API validation bugs.

1. ``VariantRecord.__post_init__`` only checked that ``prior_class_members`` and
   ``prior_class_membership`` had equal length, so NaN, negative, zero-sum,
   non-unit-sum and duplicate class memberships were accepted and propagated an
   invalid (or NaN) prior design matrix into inference.

2. ``_build_prior_design`` filled the class-membership matrix with ``=``, so a
   record carrying a duplicated class member silently dropped mixture mass
   ((snv: 0.4, snv: 0.6) yielded 0.6 instead of 1.0).

3. Boolean fields were coerced with ``bool(...)``, so the strings "false" and
   "0" — as ingested from TSV/CSV — evaluated to True and inverted the
   annotation.
"""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.config import VariantClass
from sv_pgs.data import VariantRecord, normalize_variant_record
from sv_pgs.mixture_inference import _build_prior_design


def _record(**overrides) -> VariantRecord:
    arguments = {
        "variant_id": "v1",
        "variant_class": VariantClass.SNV,
        "chromosome": "1",
        "position": 100,
    }
    arguments.update(overrides)
    return VariantRecord(**arguments)


@pytest.mark.parametrize(
    ("members", "membership", "message_fragment"),
    [
        ((VariantClass.SNV,), (float("nan"),), "finite"),
        ((VariantClass.SNV, VariantClass.DELETION_SHORT), (float("inf"), 0.0), "finite"),
        ((VariantClass.SNV, VariantClass.DELETION_SHORT), (-0.5, 1.5), "non-negative"),
        ((VariantClass.SNV, VariantClass.DELETION_SHORT), (0.0, 0.0), "positive value"),
        ((VariantClass.SNV, VariantClass.DELETION_SHORT), (3.0, 4.0), "must sum to 1.0"),
        ((VariantClass.SNV, VariantClass.DELETION_SHORT), (0.4, 0.4), "must sum to 1.0"),
        ((VariantClass.SNV, VariantClass.SNV), (0.4, 0.6), "duplicates"),
        ((VariantClass.SNV,), (), "same length"),
        ((), (1.0,), "same length"),
    ],
)
def test_invalid_class_membership_is_rejected(members, membership, message_fragment):
    with pytest.raises(ValueError, match=message_fragment):
        _record(prior_class_members=members, prior_class_membership=membership)


@pytest.mark.parametrize(
    ("members", "membership"),
    [
        ((VariantClass.SNV,), (1.0,)),
        ((VariantClass.SNV, VariantClass.DELETION_SHORT), (0.5, 0.5)),
        ((VariantClass.SNV, VariantClass.DELETION_SHORT), (0.0, 1.0)),
        ((VariantClass.SNV, VariantClass.DELETION_SHORT), (0.25, 0.75 + 1e-9)),
    ],
)
def test_valid_class_membership_is_accepted(members, membership):
    record = _record(prior_class_members=members, prior_class_membership=membership)
    assert record.prior_class_members == members
    assert record.prior_class_membership == membership


def test_class_membership_validated_with_custom_prior_features():
    """The annotated (non-fast) construction path validates memberships too."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        _record(
            prior_binary_features={"in_gene": True},
            prior_class_members=(VariantClass.SNV, VariantClass.DELETION_SHORT),
            prior_class_membership=(3.0, 4.0),
        )


def test_omitted_class_membership_defaults_to_own_class():
    record = _record(variant_class=VariantClass.DELETION_SHORT)
    assert record.prior_class_members == (VariantClass.DELETION_SHORT,)
    assert record.prior_class_membership == (1.0,)


def test_prior_design_has_no_nan_for_valid_records():
    design = _build_prior_design([_record()])
    assert not np.isnan(design.class_membership_matrix).any()
    np.testing.assert_allclose(design.class_membership_matrix.sum(axis=1), [1.0])


def test_prior_design_accumulates_duplicate_class_members():
    """Belt-and-braces: even a record that bypassed VariantRecord validation
    (fields assigned post-construction) must not lose mixture mass."""
    record = _record()
    record.prior_class_members = (VariantClass.SNV, VariantClass.SNV)
    record.prior_class_membership = (0.4, 0.6)
    design = _build_prior_design([record])
    np.testing.assert_allclose(design.class_membership_matrix, [[1.0]])


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        (1.0, True),
        (0.0, False),
        (np.bool_(True), True),
        ("true", True),
        ("TRUE", True),
        ("True", True),
        ("t", True),
        ("yes", True),
        ("1", True),
        ("false", False),
        ("FALSE", False),
        ("f", False),
        ("no", False),
        ("0", False),
        (" 0 ", False),
    ],
)
def test_boolean_tokens_are_parsed_strictly(raw_value, expected):
    record = normalize_variant_record(
        {
            "variant_id": "v1",
            "variant_class": "snv",
            "chromosome": "1",
            "position": 100,
            "is_repeat": raw_value,
            "is_copy_number": raw_value,
            "prior_binary_features": {"in_gene": raw_value},
        }
    )
    assert record.is_repeat is expected
    assert record.is_copy_number is expected
    assert record.prior_binary_features["in_gene"] is expected


@pytest.mark.parametrize("raw_value", ["maybe", "", "2", "TRUEISH", 2, -1, 0.5, None, [], object()])
def test_invalid_boolean_tokens_are_rejected(raw_value):
    with pytest.raises(ValueError, match="is_repeat"):
        normalize_variant_record(
            {
                "variant_id": "v1",
                "variant_class": "snv",
                "chromosome": "1",
                "position": 100,
                "is_repeat": raw_value,
            }
        )


@pytest.mark.parametrize("raw_value", ["maybe", "2", 2, None])
def test_invalid_boolean_tokens_are_rejected_on_variant_record(raw_value):
    with pytest.raises(ValueError, match="is_copy_number"):
        _record(is_copy_number=raw_value)
    with pytest.raises(ValueError, match="prior_binary_features"):
        _record(prior_binary_features={"in_gene": raw_value})
