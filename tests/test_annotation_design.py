"""The prior's annotation design from a variant table's annotation columns."""

import numpy as np
import pytest
from scipy.interpolate import BSpline
from scipy.integrate import quad

from sv_pgs.annotation_design import FREQUENCY, annotation_design, frequency_annotation, spline_basis
from sv_pgs.scale_mixture_ep import _sum_to_zero_basis, _sum_to_zero_rows, class_log_density, initial_hyperparameters, scale_mixture_prior


def _table(rng: np.random.Generator, count: int):
    classes = rng.choice(3, size=count).astype(np.int64)
    annotations = {
        "quality": rng.uniform(0.2, 1.0, count),
        "in_gene": (rng.random(count) < 0.3).astype(np.float64),
        "log_length": np.where((classes > 0) & (rng.random(count) < 0.9), rng.normal(3.0, 1.0, count), np.nan),
        "distance": rng.exponential(1.0, count),
        "constant": np.ones(count),
        "kind": rng.choice(3, size=count).astype(np.int32),
    }
    legends = {"kind": ("a", "b", "c")}
    return classes, annotations, legends


def _prior(classes, design):
    nodes = np.linspace(np.log(1e-5), np.log(0.5), 24)
    return scale_mixture_prior(
        class_index=classes, log_variance_offset=np.zeros(classes.shape[0]), annotation_design=design.design,
        annotation_groups=design.groups, nodes=nodes, floor=nodes[0] - 1.0, top=nodes[-1],
    )


def test_each_annotation_becomes_one_named_group_of_a_full_rank_class_centred_design() -> None:
    rng = np.random.default_rng(11)
    classes, annotations, legends = _table(rng, 400)
    result = annotation_design(annotations, legends, class_index=classes, exclude=("quality",))
    assert result.design.shape == (400, len(result.names))
    # the reliability column and the constant one leave no column; the rest are there
    assert not any(name.startswith("quality") or name.startswith("constant") for name in result.names)
    assert sum(name.startswith("in_gene:contrast") for name in result.names) == 1
    assert "log_length:linear" in result.names and "log_length:missing" in result.names
    assert any(name.startswith("distance:spline") for name in result.names)
    assert sum(name.startswith("kind:contrast") for name in result.names) == 2
    # one named group per annotation, covering every column once, with a penalty per column
    covered = np.concatenate([group.columns for group in result.groups])
    assert np.array_equal(np.sort(covered), np.arange(result.design.shape[1]))
    assert [group.name for group in result.groups] == ["distance", "in_gene", "kind", "log_length"]
    for group in result.groups:
        assert group.penalty.shape == (group.columns.shape[0],) * 2
    # a smooth's linear term and missing state are free; its spline directions and a factor's contrasts are penalised
    for group in result.groups:
        for column, weight in zip(group.columns, np.diag(group.penalty)):
            name = result.names[column]
            assert (weight > 0.0) == ("spline" in name or "contrast" in name), name
    # the prior accepts it, and names each annotation's smoothing weight after it
    prior = _prior(classes, result)
    assert prior.scale_design.shape == result.design.shape
    assert any("(distance)" in block.name for block in prior.smoothing_blocks)
    start = initial_hyperparameters(prior)
    assert np.all(np.isfinite(class_log_density(prior, start.coefficients)))


def test_a_column_constant_within_every_class_and_a_dependent_column_are_dropped() -> None:
    rng = np.random.default_rng(3)
    classes = np.repeat(np.arange(2), 50)
    values = rng.normal(size=100)
    annotations = {"per_class": classes.astype(np.float64), "x": values, "twice_x": 2.0 * values + 1.0}
    result = annotation_design(annotations, {}, class_index=classes)
    assert not any(name.startswith("per_class") for name in result.names)
    # x and 2x + 1 share every column after standardizing (the same knots on the same standardized values): whichever
    # comes first in name order keeps its columns, and the other's are all dependent on them and dropped
    kept = {source: sum(name.split(":")[0] == source for name in result.names) for source in ("twice_x", "x")}
    assert kept["twice_x"] >= 2 and kept["x"] == 0


def test_an_annotation_with_the_wrong_length_is_refused() -> None:
    with pytest.raises(ValueError, match="one value per variant"):
        annotation_design({"x": np.zeros(3)}, {}, class_index=np.zeros(4, dtype=np.int64))


def test_a_group_with_no_penalized_column_is_free_and_has_no_smoothing_weight() -> None:
    rng = np.random.default_rng(4)
    classes = rng.choice(2, size=200).astype(np.int64)
    # two distinct values that are not 0/1: a linear term, and no spline direction two values can support
    result = annotation_design({"binary_level": rng.choice([3.0, 7.0], size=200)}, {}, class_index=classes)
    (group,) = result.groups
    assert not np.any(group.penalty)
    prior = _prior(classes, result)
    assert not any(block.name.startswith("annotation group") for block in prior.smoothing_blocks)
    annotation_coordinates = prior.coefficient_size - result.design.shape[1] + np.arange(result.design.shape[1])
    # the free coefficient is in the penalty's null space
    assert np.linalg.matrix_rank(prior.null_basis[annotation_coordinates]) == result.design.shape[1]


def test_a_nearly_dependent_column_is_dropped_by_the_priors_own_rank_test() -> None:
    # A signed length change that is zero for almost every record: its spline directions on the handful of positive
    # records are numerically dependent at the prior's tolerance, and the prior accepts what is kept.
    rng = np.random.default_rng(12)
    count = 200_000
    classes = (rng.random(count) < 0.1).astype(np.int64)
    change = np.where((classes == 1) & (rng.random(count) < 0.02), rng.integers(1, 3, count).astype(np.float64), 0.0)
    result = annotation_design({"len_change": change, "gene": (rng.random(count) < 0.3).astype(np.float64)}, {}, class_index=classes)
    _prior(classes, result)
    covered = np.concatenate([group.columns for group in result.groups])
    assert np.array_equal(np.sort(covered), np.arange(result.design.shape[1]))


# ------------------------------------------------------------------ missingness (review F19)


def test_an_annotation_constant_where_observed_keeps_its_missing_state() -> None:
    """The audit's first failure case: constant on its observed rows, missing elsewhere, within one class. The numeric
    basis has nothing to fit, but observed-versus-missing is a real distinction the prior must keep."""
    rng = np.random.default_rng(21)
    classes = np.zeros(300, dtype=np.int64)
    values = np.where(rng.random(300) < 0.6, 5.0, np.nan)
    result = annotation_design({"length": values}, {}, class_index=classes)
    assert result.names == ("length:missing",)
    np.testing.assert_array_equal(result.design[:, 0], np.isnan(values).astype(np.float64))
    prior = _prior(classes, result)
    # theta on the missing state moves exactly the missing rows' log prior variance against the observed ones'
    np.testing.assert_allclose(np.ptp(prior.scale_design[np.isnan(values), 0]), 0.0)
    assert prior.scale_design[np.isnan(values), 0][0] != prior.scale_design[~np.isnan(values), 0][0]


def test_a_binary_annotation_keeps_unknown_apart_from_false() -> None:
    """The audit's second failure case: 0/1 with missing values. Unknown is a third level, never filled as 0."""
    rng = np.random.default_rng(22)
    count = 600
    classes = rng.choice(2, size=count).astype(np.int64)
    values = np.where(rng.random(count) < 0.2, np.nan, (rng.random(count) < 0.4).astype(np.float64))
    result = annotation_design({"in_exon": values}, {}, class_index=classes)
    assert len(result.names) == 2 and all(name.startswith("in_exon:contrast") for name in result.names)
    # three levels, rows of one level share one row of the design, and the three rows differ
    levels = np.where(np.isnan(values), 2, values).astype(int)
    rows = [result.design[levels == level][0] for level in range(3)]
    for level in range(3):
        np.testing.assert_allclose(result.design[levels == level], np.broadcast_to(rows[level], (int(np.sum(levels == level)), 2)))
    assert not np.allclose(rows[0], rows[2]) and not np.allclose(rows[1], rows[2])


def test_a_factors_missing_code_is_its_own_level() -> None:
    rng = np.random.default_rng(23)
    codes = rng.choice(3, size=500).astype(np.int32)
    codes[rng.random(500) < 0.1] = -1
    result = annotation_design({"kind": codes}, {"kind": ("a", "b", "c")}, class_index=np.zeros(500, dtype=np.int64))
    assert len(result.names) == 3 and "kind:contrast(missing)" in result.names


def test_a_partly_missing_continuous_annotation_is_zero_on_its_missing_rows_in_every_numeric_column() -> None:
    rng = np.random.default_rng(24)
    values = np.where(rng.random(400) < 0.3, np.nan, rng.normal(size=400))
    result = annotation_design({"x": values}, {}, class_index=np.zeros(400, dtype=np.int64))
    numeric = [position for position, name in enumerate(result.names) if not name.endswith(":missing")]
    assert numeric and np.all(result.design[np.isnan(values)][:, numeric] == 0.0)
    assert "x:missing" in result.names


# ------------------------------------------------------------------ the roughness penalty (review F20)


def test_the_spline_penalty_is_the_exact_integrated_squared_second_derivative() -> None:
    rng = np.random.default_rng(31)
    values = rng.normal(size=500)
    knots, basis, penalty = spline_basis(values)
    coefficients = rng.normal(size=basis.shape[1])
    second = BSpline(knots, coefficients, 3).derivative(2)
    interior = np.unique(knots)
    integral = sum(quad(lambda x: second(x) ** 2, left, right)[0] for left, right in zip(interior[:-1], interior[1:]))
    np.testing.assert_allclose(coefficients @ penalty @ coefficients, integral, rtol=1e-10)
    # a local-support basis: at most degree + 1 nonzero values per row, summing to one
    assert np.max(np.count_nonzero(basis, axis=1)) <= 4
    np.testing.assert_allclose(basis.sum(axis=1), 1.0)


def test_the_spline_penalty_null_space_is_exactly_the_linear_functions() -> None:
    values = np.random.default_rng(32).normal(size=300)
    knots, _basis, penalty = spline_basis(values)
    count = penalty.shape[0]
    greville = np.array([knots[index + 1 : index + 4].mean() for index in range(count)])
    for coefficients in (np.ones(count), greville):
        assert abs(coefficients @ penalty @ coefficients) <= 1e-10 * np.linalg.norm(penalty)
    eigenvalues = np.linalg.eigvalsh(penalty)
    assert int(np.sum(eigenvalues > 1e-10 * eigenvalues[-1])) == count - 2


def test_the_penalty_is_a_property_of_the_function_not_of_the_basis() -> None:
    """The design's penalized columns are the basis on the penalty's eigenvectors: a function's penalty computed in
    those coordinates equals its integral of f''^2, whatever linear part it has (the linear column is free)."""
    rng = np.random.default_rng(33)
    values = rng.normal(size=400)
    result = annotation_design({"x": values}, {}, class_index=np.zeros(400, dtype=np.int64))
    (group,) = result.groups
    standardized = (values - values.mean()) / values.std()
    knots, basis, penalty = spline_basis(standardized)
    spline_columns = [column for column in group.columns if "spline" in result.names[column]]
    theta = rng.normal(size=len(spline_columns))
    # the design's spline columns are B U_+, so the function they make is B (U_+ theta); its roughness integral is
    # (U_+ theta)' S (U_+ theta), which the group's diagonal penalty must give
    eigenvalues, eigenvectors = np.linalg.eigh(penalty)
    rough = eigenvalues > 1e-12 * eigenvalues[-1]
    np.testing.assert_allclose(result.design[:, spline_columns], basis @ eigenvectors[:, rough], atol=1e-12)
    local = [list(group.columns).index(column) for column in spline_columns]
    coefficients = eigenvectors[:, rough] @ theta
    np.testing.assert_allclose(theta @ group.penalty[np.ix_(local, local)] @ theta, coefficients @ penalty @ coefficients, rtol=1e-10)


# ------------------------------------------------------------------ the frequency function and the level bases


def test_the_frequency_function_is_its_own_named_smooth_group() -> None:
    rng = np.random.default_rng(41)
    classes = rng.choice(2, size=300).astype(np.int64)
    log_variance = np.log(2.0 * rng.uniform(0.01, 0.5, 300))
    result = annotation_design(frequency_annotation(log_variance), {}, class_index=classes)
    (group,) = result.groups
    assert group.name == FREQUENCY
    assert f"{FREQUENCY}:linear" in result.names and any("spline" in name for name in result.names)


def test_the_level_basis_is_orthonormal_sums_to_zero_and_gathers_rows() -> None:
    for size in (2, 3, 7, 40):
        basis = _sum_to_zero_rows(np.arange(size), size)
        np.testing.assert_allclose(basis.T @ basis, np.eye(size - 1), atol=1e-14)
        np.testing.assert_allclose(basis.sum(axis=0), 0.0, atol=1e-14)
        # the same column space as the lattice's decomposed basis: the vectors that sum to zero
        lattice = _sum_to_zero_basis(size)
        np.testing.assert_allclose(basis @ basis.T, lattice @ lattice.T, atol=1e-13)
        rows = np.array([size - 1, 0, size // 2])
        np.testing.assert_array_equal(_sum_to_zero_rows(rows, size), basis[rows])


def test_a_factor_prior_does_not_depend_on_which_level_comes_first() -> None:
    """Exchangeable levels: relabelling the levels leaves the design's column space and the penalty's induced prior on
    the level effects, (I - 11'/L) / lambda, unchanged."""
    rng = np.random.default_rng(42)
    codes = rng.choice(4, size=400).astype(np.int32)
    first = annotation_design({"kind": codes}, {"kind": ("a", "b", "c", "d")}, class_index=np.zeros(400, dtype=np.int64))
    permutation = np.array([2, 0, 3, 1])
    second = annotation_design({"kind": permutation[codes].astype(np.int32)}, {"kind": ("c", "a", "d", "b")}, class_index=np.zeros(400, dtype=np.int64))
    # the prior covariance of the per-variant contributions D theta, theta ~ N(0, P^-1), is the same
    covariances = [design.design @ np.linalg.inv(design.groups[0].penalty) @ design.design.T for design in (first, second)]
    np.testing.assert_allclose(covariances[0], covariances[1], atol=1e-12)


def test_a_near_dependent_earlier_annotation_never_drops_a_later_independent_one():
    """The screen drops a column for what it adds to the columns before it, never a later annotation's columns for an
    earlier one's near-dependence: a length smooth observed on a few rows (its spline directions barely resolved) must
    leave every column of a distance smooth observed everywhere (ENSG00000187605.16 [real] lost all of
    log_tss_distance this way)."""
    rng = np.random.default_rng(5)
    rows = 20_000
    length = np.full(rows, np.nan)
    length[:40] = rng.lognormal(5.0, 2.0, 40)
    distance = rng.uniform(0.0, 14.0, rows)
    design = annotation_design({"a_length": np.log1p(length), "b_distance": distance}, {}, class_index=np.zeros(rows, dtype=np.int64))
    distance_names = [name for name in design.names if name.startswith("b_distance:")]
    assert "b_distance:linear" in distance_names
    assert any(":spline" in name for name in distance_names)
    centred = design.design - design.design.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(centred.T @ centred)
    assert eigenvalues[0] > np.finfo(float).eps * rows * max(eigenvalues[-1], 1.0)
