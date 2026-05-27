"""Tests for the per-variant routing classifier."""

from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.config import VariantClass
from sv_pgs.data import VariantRecord
from sv_pgs.variant_routing import RoutingDecision, classify_variants


def _record(
    variant_id: str,
    variant_class: VariantClass,
    *,
    is_repeat: bool = False,
    is_copy_number: bool = False,
) -> VariantRecord:
    return VariantRecord(
        variant_id=variant_id,
        variant_class=variant_class,
        chromosome="1",
        position=1,
        is_repeat=is_repeat,
        is_copy_number=is_copy_number,
    )


def test_mixed_cohort_routes_by_rule() -> None:
    """SNV stays dense, rare SV goes sparse, common SV stays dense, rare repeat goes sparse."""
    n_samples = 10_000
    threshold = n_samples // 64  # 156

    records = [
        _record("snv1", VariantClass.SNV),  # dense (snv_like)
        _record("del_rare", VariantClass.DELETION_SHORT),  # sparse (rare structural)
        _record("del_common", VariantClass.DELETION_LONG),  # dense (common structural)
        _record("rep_singleton", VariantClass.STR_VNTR_REPEAT, is_repeat=True),  # sparse (rare repeat)
        _record("snv2", VariantClass.SNV),  # dense (snv_like)
        _record("cnv_rare", VariantClass.SNV, is_copy_number=True),  # sparse via is_copy_number
    ]
    support_counts = np.array(
        [
            5_000,  # snv1, common
            10,  # del_rare, way below threshold
            2_000,  # del_common, above threshold
            1,  # rep_singleton
            8_000,  # snv2
            3,  # cnv_rare
        ],
        dtype=np.int64,
    )

    decision = classify_variants(records, support_counts, n_samples)

    assert isinstance(decision, RoutingDecision)
    assert decision.rationale_counts["threshold"] == threshold
    assert decision.rationale_counts["n_variants"] == len(records)

    assert sorted(decision.dense_local_indices.tolist()) == [0, 2, 4]
    assert sorted(decision.sparse_local_indices.tolist()) == [1, 3, 5]
    assert decision.dense_local_indices.dtype == np.int32
    assert decision.sparse_local_indices.dtype == np.int32

    # Rationale accounting.
    rc = decision.rationale_counts
    assert rc["dense_snv_like"] == 2
    assert rc["dense_common_structural"] == 1
    assert rc["dense_common_repeat"] == 0
    assert rc["sparse_rare_structural"] == 2  # del_rare + cnv_rare
    assert rc["sparse_rare_repeat"] == 1
    assert rc["dense_total"] == 3
    assert rc["sparse_total"] == 3
    assert rc["collapsed_total"] == 0

    # Collapse is not in scope for this classifier.
    assert decision.collapsed_representative_for == {}


def test_aou_snp_only_all_dense() -> None:
    """For an AoU-like SNP-only cohort where every variant is common, all route to dense."""
    n_samples = 250_000
    threshold = n_samples // 64

    rng = np.random.default_rng(0)
    n_variants = 5_000
    records = [_record(f"snv{i}", VariantClass.SNV) for i in range(n_variants)]
    # Carriers well above the threshold for every variant.
    support_counts = rng.integers(
        low=threshold * 2,
        high=n_samples // 2,
        size=n_variants,
        dtype=np.int64,
    )

    decision = classify_variants(records, support_counts, n_samples)

    assert decision.sparse_local_indices.size == 0
    assert decision.dense_local_indices.size == n_variants
    # Indices must enumerate 0..n_variants-1 in order.
    np.testing.assert_array_equal(
        decision.dense_local_indices,
        np.arange(n_variants, dtype=np.int32),
    )
    rc = decision.rationale_counts
    assert rc["dense_snv_like"] == n_variants
    assert rc["sparse_total"] == 0
    assert rc["collapsed_total"] == 0
    assert decision.collapsed_representative_for == {}


def test_custom_threshold_routes_boundary_correctly() -> None:
    """``carrier_count <= threshold`` is inclusive."""
    records = [
        _record("d_at", VariantClass.DELETION_SHORT),
        _record("d_above", VariantClass.DELETION_SHORT),
    ]
    support_counts = np.array([5, 6], dtype=np.int64)

    decision = classify_variants(
        records,
        support_counts,
        n_samples=1_000,
        sparse_carrier_threshold=5,
    )
    assert decision.sparse_local_indices.tolist() == [0]
    assert decision.dense_local_indices.tolist() == [1]


def test_shape_mismatch_raises() -> None:
    records = [_record("snv1", VariantClass.SNV)]
    with pytest.raises(ValueError):
        classify_variants(records, np.array([1, 2], dtype=np.int64), n_samples=100)


def test_inversion_and_duplication_match_structural_rule() -> None:
    records = [
        _record("inv1", VariantClass.INVERSION_BND_COMPLEX),
        _record("dup1", VariantClass.DUPLICATION_LONG),
        _record("mei1", VariantClass.INSERTION_MEI),
    ]
    support_counts = np.array([0, 1, 2], dtype=np.int64)
    decision = classify_variants(records, support_counts, n_samples=10_000)
    assert sorted(decision.sparse_local_indices.tolist()) == [0, 1, 2]
    assert decision.rationale_counts["sparse_rare_structural"] == 3
