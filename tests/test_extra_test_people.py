import numpy as np
import pytest

from benchmarks.bench_real.extra_test_people import (
    ModelWeights,
    aligned_covariates,
    fast_method_weights,
    finished_scores,
    joined_families,
    partial_scores,
    shown,
    weight_table,
)


def test_fast_method_weights_reproduce_the_jobs_own_score_and_drop_unused_rows(tmp_path):
    rng = np.random.default_rng(2)
    records = np.array([9, 2, 15, 4, 11])
    means = rng.uniform(0.0, 2.0, size=5)
    effects = np.array([0.3, -0.2, 0.0, 0.5, 0.1])
    other = rng.normal(size=5)
    path = tmp_path / "units.npz"
    np.savez(path, means=means, effects=np.column_stack([effects, other]),
             labels=np.array(["with_sv|ldpred2_auto|auto", "snv_indel|ldpred2_auto|auto"]))
    found = fast_method_weights(path, records, "with_sv|ldpred2_auto|auto")
    # The zero-effect record is dropped and the rest are in ascending store-row order.
    np.testing.assert_array_equal(found.rows, [2, 4, 9, 11])
    codes = rng.integers(0, 255, size=(5, 6)).astype(np.uint8)
    used = np.array([1, 3, 0, 4])
    table = weight_table([found])
    scores = finished_scores(partial_scores(codes[used], table.weights), table, np.ones(4, dtype=bool))
    expected = (effects[used, None] * (codes[used].astype(float) / 127.0 - means[used, None])).sum(axis=0)
    np.testing.assert_allclose(scores[0], expected, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="no column"):
        fast_method_weights(path, records, "with_sv|lassosum2|auto")


def test_counts_below_forty_are_masked():
    assert shown(0) == "<40"
    assert shown(1) == "<40" and shown(39) == "<40"
    assert shown(40) == "40" and shown(7185) == "7185"


def test_mean_score_matches_standardized_formula_with_missing_record_at_its_mean():
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 255, size=(5, 7)).astype(np.uint8)
    rows = np.array([2, 4, 9, 11, 15])
    means, scales, beta = rng.normal(size=5), rng.uniform(0.5, 2.0, size=5), rng.normal(size=5)
    weights = beta / scales
    table = weight_table([ModelWeights(rows, weights, weights * means, "s"), ModelWeights(rows[:2], weights[:2], (weights * means)[:2], "s")])
    found = np.array([True, True, False, True, True])
    scores = finished_scores(partial_scores(codes[found], table.weights[found]), table, found)
    signed = codes.astype(float) - 127.0
    expected = ((signed[found] - means[found, None]) / scales[found, None] * beta[found, None]).sum(axis=0)
    np.testing.assert_allclose(scores[0], expected)
    np.testing.assert_allclose(scores[1], ((signed[:2] - means[:2, None]) / scales[:2, None] * beta[:2, None]).sum(axis=0))


def test_covariates_align_by_name_and_families_join_the_cohort():
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = aligned_covariates(["intercept", "PC1"], values, ["intercept", "trait:sex=2", "PC1"])
    np.testing.assert_array_equal(out, [[1.0, 0.0, 2.0], [3.0, 0.0, 4.0]])
    # Old people's components 0, 1; new people: one in component 1 (joins family 7), two sharing a new component.
    families = joined_families(["a", "b"], np.array([3, 7]), ["c", "d", "e"], np.array([0, 1, 1, 2, 2]))
    np.testing.assert_array_equal(families, [7, 8, 8])
