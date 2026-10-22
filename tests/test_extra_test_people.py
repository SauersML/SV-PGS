import numpy as np

from benchmarks.bench_real.extra_test_people import (
    ModelWeights,
    aligned_covariates,
    finished_scores,
    joined_families,
    partial_scores,
    weight_table,
)


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
