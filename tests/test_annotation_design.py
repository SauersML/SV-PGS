"""The prior's annotation design from a variant table's annotation columns."""

import numpy as np
import pytest
from scipy.interpolate import BSpline
from scipy.integrate import quad

from sv_pgs.annotation_design import FREQUENCY, LD_SCORE, annotation_design, frequency_annotation, ld_score_annotation, smooth_shift_record, spline_basis
from sv_pgs.scale_mixture_ep import (
    Cavity, FixedPoint, _sum_to_zero_basis, _sum_to_zero_rows, class_log_density, derived_lattice, embed_hyperparameters, fit_hyperparameters,
    hyper_step, initial_hyperparameters, scale_mixture_prior, tilted_moments, without_annotations,
)
from tests.oracles import INDEPENDENT_EFFECTS, diagonal_posterior


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


def test_a_group_with_no_penalized_column_has_one_learned_ridge() -> None:
    rng = np.random.default_rng(4)
    classes = rng.choice(2, size=200).astype(np.int64)
    # two distinct values that are not 0/1: a linear term, and no spline direction two values can support
    result = annotation_design({"binary_level": rng.choice([3.0, 7.0], size=200)}, {}, class_index=classes)
    (group,) = result.groups
    assert not np.any(group.penalty)
    prior = _prior(classes, result)
    (block,) = (block for block in prior.smoothing_blocks if block.name.startswith("annotation group"))
    assert block.name.endswith("(binary_level) null space")
    annotation_coordinates = prior.coefficient_size - result.design.shape[1] + np.arange(result.design.shape[1])
    np.testing.assert_array_equal(np.sort(block.coordinates), annotation_coordinates)
    assert np.linalg.matrix_rank(block.factor) == result.design.shape[1]
    # a proper prior with its own weight: none of the group's coefficients is left in the profiled null space
    np.testing.assert_allclose(prior.null_basis[annotation_coordinates], 0.0, atol=1e-10)


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


def test_a_point_mass_at_the_minimum_leaves_the_spline_its_interior_knots() -> None:
    # Most records are SNVs at length 1: every row quartile sits at the minimum, the distinct values' quartiles do not.
    lengths = np.concatenate([np.ones(9_000), np.arange(2, 50, dtype=np.float64), np.geomspace(50, 2e6, 1_000)])
    knots, _basis, _penalty = spline_basis(np.log(lengths))
    interior = knots[4:-4]
    assert interior.shape[0] == 3 and np.all((interior > 0.0) & (interior < np.log(2e6)))


def test_a_length_enters_on_its_log_scale() -> None:
    lengths = np.concatenate([np.ones(50), np.geomspace(50, 1e6, 50)])
    classes = np.zeros(lengths.shape[0], dtype=np.int64)
    from_lengths = annotation_design({"sv_length": lengths}, {}, class_index=classes)
    from_logs = annotation_design({"length_log": np.log(lengths)}, {}, class_index=classes)
    np.testing.assert_allclose(from_lengths.design, from_logs.design)
    with pytest.raises(ValueError, match="must be positive"):
        annotation_design({"sv_length": np.array([0.0, 1.0, 2.0])}, {}, class_index=np.zeros(3, dtype=np.int64))


# ------------------------------------------------------------------ hierarchical class smooths (length by type)


_LABELS = ("snv", "deletion", "duplication", "insertion")


def _length_table(rng: np.random.Generator, sizes=(300, 400, 400, 400), narrow: bool = False):
    """Records by class: class 0 the SNVs, every one at length 1, the others log-uniform from 50 bp to 1 Mb; with
    ``narrow`` the last class only above 500 kb, inside the shared smooth's last knot interval, where it resolves only
    the cubics of that interval."""
    classes = np.repeat(np.arange(len(sizes)), sizes).astype(np.int64)
    lengths = np.ones(classes.shape[0])
    for position in range(1, len(sizes)):
        rows = classes == position
        low = np.log(5e5) if narrow and position == len(sizes) - 1 else np.log(50.0)
        lengths[rows] = np.exp(rng.uniform(low, np.log(1e6), int(rows.sum())))
    return classes, lengths


def _resolved_rank(columns: np.ndarray, classes: np.ndarray) -> int:
    """How many directions of ``columns`` the prior resolves after class centring: its Gram's eigenvalues above the
    prior's bound, eps p max(largest, 1)."""
    centred = np.array(columns, dtype=np.float64, copy=True)
    for position in np.unique(classes):
        rows = classes == position
        centred[rows] -= centred[rows].mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(centred.T @ centred)
    return int(np.sum(eigenvalues > np.finfo(np.float64).eps * columns.shape[0] * max(float(eigenvalues[-1]), 1.0)))


def _annotation_covariance(classes: np.ndarray, design) -> np.ndarray:
    """The prior covariance of every variant's annotation shift of its log prior variance, D theta (D the prior's own
    class-centred design), with every annotation block of the prior at unit weight."""
    prior = _prior(classes, design)
    start = prior.coefficient_size - prior.scale_size
    precision = np.zeros((prior.scale_size, prior.scale_size))
    for block in prior.smoothing_blocks:
        if block.coordinates[0] >= start:
            local = block.coordinates - start
            precision[np.ix_(local, local)] += block.factor.T @ block.factor
    return prior.scale_design @ np.linalg.solve(precision, prior.scale_design.T)


def test_without_two_varying_classes_the_class_deviations_leave_the_design_as_it_was() -> None:
    # The snv_indel arm's one class, a store whose length varies in one class only, and a name that is no annotation.
    rng = np.random.default_rng(51)
    classes, lengths = _length_table(rng, sizes=(300, 400))
    annotations = {"sv_length": lengths, "x": rng.normal(size=classes.shape[0])}
    plain = annotation_design(annotations, {}, class_index=classes)
    for by_class in (("sv_length",), ("absent",)):
        result = annotation_design(annotations, {}, class_index=classes, by_class=by_class)
        np.testing.assert_array_equal(result.design, plain.design)
        assert result.names == plain.names and [group.name for group in result.groups] == [group.name for group in plain.groups]
    one_class = annotation_design({"sv_length": lengths}, {}, class_index=np.zeros(classes.shape[0], dtype=np.int64), by_class=("sv_length",))
    assert [group.name for group in one_class.groups] == ["sv_length"]


def test_class_deviations_are_the_shared_columns_on_sum_to_zero_class_contrasts() -> None:
    rng = np.random.default_rng(52)
    classes, lengths = _length_table(rng)
    shared = annotation_design({"sv_length": lengths}, {}, class_index=classes)
    result = annotation_design({"sv_length": lengths}, {}, class_index=classes, by_class=("sv_length",), class_labels=_LABELS)
    assert [group.name for group in result.groups] == ["sv_length", "sv_length:by_class"]
    first, deviation = result.groups
    # the shared smooth is the one without deviations, column for column and penalty for penalty
    np.testing.assert_array_equal(result.design[:, first.columns], shared.design)
    np.testing.assert_array_equal(first.penalty, shared.groups[0].penalty)
    # three classes vary (the SNVs all sit at length 1): two contrasts of every shared column, each named by the class
    # it sets against the ones before it
    functions = [name.split(":", 1)[1] for name in shared.names]
    assert [result.names[column] for column in deviation.columns] == [
        f"sv_length:by_class({label}):{function}" for function in functions for label in ("duplication", "insertion")
    ]
    contrasts = _sum_to_zero_rows(np.arange(3), 3)
    level = np.maximum(classes - 1, 0)
    for position, column in enumerate(deviation.columns):
        function, contrast = divmod(position, 2)
        np.testing.assert_allclose(result.design[:, column], np.where(classes > 0, shared.design[:, function] * contrasts[level, contrast], 0.0), atol=1e-15)
    # sum_c d_c' P d_c: the function's penalty on every contrast
    np.testing.assert_allclose(deviation.penalty, np.kron(shared.groups[0].penalty, np.eye(2)))
    # the prior accepts it, with the deviations' own roughness weight and the ridge on their linear parts
    names = [block.name for block in _prior(classes, result).smoothing_blocks]
    assert any(name.endswith("(sv_length:by_class)") for name in names)
    assert any(name.endswith("(sv_length:by_class) null space") for name in names)


@pytest.mark.parametrize("narrow", [False, True])
def test_class_deviations_keep_exactly_what_the_classes_resolve_beyond_the_shared_smooth(narrow: bool) -> None:
    """Identifiability: the deviations add every direction the classes' own copies of the shared smooth resolve beyond
    it, and nothing that is a class's constant (its class density's location) or the shared curve. A class that spans
    part of the length range (``narrow``) leaves contrast directions no row resolves, and the group is restricted to
    the rest, as one subspace."""
    rng = np.random.default_rng(53)
    classes, lengths = _length_table(rng, narrow=narrow)
    shared = annotation_design({"sv_length": lengths}, {}, class_index=classes)
    result = annotation_design({"sv_length": lengths}, {}, class_index=classes, by_class=("sv_length",), class_labels=_LABELS)
    (deviation,) = [group for group in result.groups if group.name == "sv_length:by_class"]
    indicators = (classes[:, None] == np.arange(len(_LABELS))[None, :]).astype(np.float64)
    per_class = np.column_stack([shared.design * indicators[:, [position]] for position in range(len(_LABELS))])
    assert deviation.columns.shape[0] == _resolved_rank(per_class, classes) - _resolved_rank(shared.design, classes)
    assert any(":direction" in result.names[column] for column in deviation.columns) == narrow
    whole = np.column_stack([indicators, result.design])
    assert np.linalg.matrix_rank(whole) == whole.shape[1]
    _prior(classes, result)


@pytest.mark.parametrize("narrow", [False, True])
def test_the_hierarchical_prior_does_not_depend_on_the_class_order(narrow: bool) -> None:
    """Exchangeable classes: relabelling them moves the Helmert contrasts (and, where a class's support restricts the
    group, its directions) by a rotation, which leaves the prior covariance of every variant's log-variance shift
    unchanged: no class is a reference level."""
    rng = np.random.default_rng(54)
    classes, lengths = _length_table(rng, narrow=narrow)
    first = annotation_design({"sv_length": lengths}, {}, class_index=classes, by_class=("sv_length",), class_labels=_LABELS)
    relabel = np.array([2, 3, 0, 1])
    labels = [""] * len(_LABELS)
    for old, new in enumerate(relabel):
        labels[new] = _LABELS[old]
    second = annotation_design({"sv_length": lengths}, {}, class_index=relabel[classes], by_class=("sv_length",), class_labels=labels)
    assert first.names != second.names or narrow
    expected = _annotation_covariance(classes, first)
    np.testing.assert_allclose(_annotation_covariance(relabel[classes], second), expected, atol=1e-9 * np.abs(expected).max())



def test_a_missing_state_contrast_the_shared_smooth_carries_leaves_every_spline_contrast() -> None:
    # Only the first class has a record with no length, so its missing-state contrast is the shared missing state
    # itself: the one unresolved direction lies in the penalty's null space, where its roughness is roundoff, and the
    # restriction must hold that direction, not a spline contrast, so the group keeps every resolved one.
    classes = np.repeat(np.arange(2), 30).astype(np.int64)
    lengths = np.tile(np.geomspace(50.0, 1e6, 30), 2)
    lengths[0] = np.nan
    shared = annotation_design({"sv_length": lengths}, {}, class_index=classes)
    result = annotation_design({"sv_length": lengths}, {}, class_index=classes, by_class=("sv_length",), class_labels=_LABELS[:2])
    (deviation,) = [group for group in result.groups if group.name == "sv_length:by_class"]
    indicators = (classes[:, None] == np.arange(2)[None, :]).astype(np.float64)
    per_class = np.column_stack([shared.design * indicators[:, [position]] for position in range(2)])
    assert deviation.columns.shape[0] == _resolved_rank(per_class, classes) - _resolved_rank(shared.design, classes)
    _prior(classes, result)


def test_a_direction_at_the_bound_after_the_restriction_is_restricted_too() -> None:
    # A class confined to the last sliver of the length range: holding its unresolved contrasts at their smoothest can
    # leave a kept direction whose design falls to the rank bound, which the column-order screen would then drop for
    # one class order and keep for another. The restriction runs until the screen's own test passes, so both orders
    # keep the same prior.
    rng = np.random.default_rng(41)
    classes = np.repeat(np.arange(3), 100).astype(np.int64)
    lengths = np.exp(np.r_[rng.uniform(4.0, 14.0, 200), rng.uniform(13.75, 14.0, 100)])
    relabel = np.array([2, 0, 1])
    first = annotation_design({"sv_length": lengths}, {}, class_index=classes, by_class=("sv_length",))
    second = annotation_design({"sv_length": lengths}, {}, class_index=relabel[classes], by_class=("sv_length",))
    assert first.design.shape[1] == second.design.shape[1]
    expected = _annotation_covariance(classes, first)
    np.testing.assert_allclose(_annotation_covariance(relabel[classes], second), expected, atol=1e-9 * np.abs(expected).max())


def test_a_narrow_class_keeps_its_linear_deviation_free_of_roughness() -> None:
    """Where a class spans part of the length range, the deviations it cannot resolve are held at their smoothest: a
    slope that differs by class (the deviations' linear parts, under their own ridge) stays a straight line on every
    class's rows, so the roughness weight never charges a class's slope for rows it does not reach."""
    rng = np.random.default_rng(55)
    classes, lengths = _length_table(rng, narrow=True)
    result = annotation_design({"sv_length": lengths}, {}, class_index=classes, by_class=("sv_length",), class_labels=_LABELS)
    (deviation,) = [group for group in result.groups if group.name == "sv_length:by_class"]
    eigenvalues, eigenvectors = np.linalg.eigh(deviation.penalty)
    free = eigenvectors[:, eigenvalues <= np.finfo(np.float64).eps * eigenvalues.shape[0] * eigenvalues[-1]]
    # one free direction per contrast of the three varying classes' slopes
    assert free.shape[1] == 2
    log_length = np.log(lengths)
    for direction in free.T:
        shift = result.design[:, deviation.columns] @ direction
        for position in range(1, len(_LABELS)):
            rows = classes == position
            line = np.column_stack([np.ones(int(rows.sum())), log_length[rows]])
            fitted, *_ = np.linalg.lstsq(line, shift[rows], rcond=None)
            np.testing.assert_allclose(line @ fitted, shift[rows], atol=1e-10 * np.abs(shift).max())



def test_the_nested_empirical_bayes_gives_each_class_its_length_slope_and_shrinks_a_sparse_one() -> None:
    """Normal means (each effect's cavity its own likelihood): deletions' log prior variance rises with log length and
    duplications' falls (+1 and -1 per standard deviation), and six records of a fourth class rise too. One shared
    length smooth cannot carry opposite slopes; the class deviations, entered as the fit enters them (the prior without
    annotations first on the start lattice the data derive, then every annotation group at its lambda = infinity edge,
    ``embed_hyperparameters``), are released by the evidence, give each well-resolved class its own slope, and hold the
    six records' class near the shared curve."""
    rng = np.random.default_rng(56)
    classes, lengths = _length_table(rng, sizes=(200, 500, 500, 6))
    log_length = np.log(lengths)
    structural = classes > 0
    standardized = np.where(structural, (log_length - log_length[structural].mean()) / log_length[structural].std(), 0.0)
    slopes = np.array([0.0, 1.0, -1.0, 1.0])
    precision = rng.uniform(100.0, 800.0, classes.shape[0])
    effects = rng.standard_normal(classes.shape[0]) * np.exp(0.5 * (np.log(0.01) + slopes[classes] * standardized))
    cavity = Cavity(precision=precision, shift=precision * effects + np.sqrt(precision) * rng.standard_normal(classes.shape[0]))
    offset = np.zeros(classes.shape[0])
    nodes, floor, top = derived_lattice(cavity.precision, cavity.shift, offset, 1e-3)

    def prior(design):
        return scale_mixture_prior(
            class_index=classes, log_variance_offset=offset, annotation_design=design.design, annotation_groups=design.groups,
            nodes=nodes, floor=floor, top=top,
        )

    shared = annotation_design({"sv_length": lengths}, {}, class_index=classes)
    by_class = annotation_design({"sv_length": lengths}, {}, class_index=classes, by_class=("sv_length",), class_labels=_LABELS)
    base = without_annotations(prior(shared))
    base_fit = hyper_step(base, initial_hyperparameters(base), cavity, INDEPENDENT_EFFECTS, 1 << 20, 1.0 / 128.0)
    fits = {}
    for name, design in (("shared", shared), ("by_class", by_class)):
        model = prior(design)
        fits[name] = (model, hyper_step(model, embed_hyperparameters(base, model, base_fit.hyperparameters), cavity, INDEPENDENT_EFFECTS, 1 << 20, 1.0 / 128.0))
    model, fit = fits["by_class"]
    assert fit.evidence > fits["shared"][1].evidence + 10.0
    names = [block.name for block in model.smoothing_blocks]
    assert any(np.isfinite(weight) for name, weight in zip(names, fit.hyperparameters.log_smoothing) if "(sv_length:by_class)" in name)
    theta = np.asarray(fit.hyperparameters.coefficients)[model.coefficient_size - model.scale_size :]
    shift = model.scale_design @ theta
    fitted = {position: np.polyfit(standardized[classes == position], shift[classes == position], 1)[0] for position in (1, 2)}
    assert fitted[1] > 0.5 and fitted[2] < -0.5
    (deviation,) = [group for group in model.annotation_groups if group.name == "sv_length:by_class"]
    bend = model.scale_design[:, deviation.columns] @ theta[deviation.columns]
    spread = [float(np.sqrt(np.mean(np.square(bend[classes == position])))) for position in range(len(_LABELS))]
    assert spread[0] == 0.0 and spread[3] < 0.5 * min(spread[1], spread[2])


def _draw_two():
    """Draw 2 of the class-length process above (its effects drawn before the cavities' precisions): the length
    classes, their lengths, each effect's cavity (its own likelihood), and the prior with the class deviations on the
    lattice the data derive."""
    rng = np.random.default_rng(56)
    classes, lengths = _length_table(rng, sizes=(200, 500, 500, 6))
    log_length = np.log(lengths)
    structural = classes > 0
    standardized = np.where(structural, (log_length - log_length[structural].mean()) / log_length[structural].std(), 0.0)
    slopes = np.array([0.0, 1.0, -1.0, 1.0])
    effects = rng.standard_normal(classes.shape[0]) * np.exp(0.5 * (np.log(0.01) + slopes[classes] * standardized))
    precision = rng.uniform(100.0, 800.0, classes.shape[0])
    cavity = Cavity(precision=precision, shift=precision * effects + np.sqrt(precision) * rng.standard_normal(classes.shape[0]))
    offset = np.zeros(classes.shape[0])
    nodes, floor, top = derived_lattice(cavity.precision, cavity.shift, offset, 1e-3)
    design = annotation_design({"sv_length": lengths}, {}, class_index=classes, by_class=("sv_length",), class_labels=_LABELS)
    prior = scale_mixture_prior(
        class_index=classes, log_variance_offset=offset, annotation_design=design.design, annotation_groups=design.groups,
        nodes=nodes, floor=floor, top=top,
    )
    return prior, cavity


def _released_deviation(prior, hyperparameters) -> bool:
    names = [block.name for block in prior.smoothing_blocks]
    return any(np.isfinite(weight) for name, weight in zip(names, hyperparameters.log_smoothing) if "(sv_length:by_class)" in name)


@pytest.mark.slow
def test_draw_two_releases_the_class_deviations_within_a_one_mebibyte_budget() -> None:
    """The release of the class deviations on draw 2 is worth 135 nats (V 3284.1 against the base's 3148.8). The
    exact line integrals of its corrections need up to four open intervals a direction; charged as normalizers per
    variant and step (2 x 29 x 1,206 x 8 bytes an interval), 1 MiB refused every integral past its first round, so no
    deviation weight had a certified value and the search kept them all at their edge (V 3148.65). ``_lines`` holds
    only each step's class log densities whole, its variants in pieces of the budget."""
    prior, cavity = _draw_two()
    base = without_annotations(prior)
    base_fit = hyper_step(base, initial_hyperparameters(base), cavity, INDEPENDENT_EFFECTS, 1 << 20, 1.0 / 128.0)
    fit = hyper_step(prior, embed_hyperparameters(base, prior, base_fit.hyperparameters), cavity, INDEPENDENT_EFFECTS, 1 << 20, 1.0 / 128.0)
    assert _released_deviation(prior, fit.hyperparameters)
    assert fit.evidence > base_fit.evidence + 100.0


@pytest.mark.slow
def test_the_nested_outer_loop_takes_the_release_its_polished_state_plans() -> None:
    """The fit's own route on draw 2 (``fit_hyperparameters``: the prior without annotations from its start, then the
    annotated prior from that fit with every group at its lambda = infinity edge), with the exact normal-means fixed
    point. Each stage's first plan predicts no gain from an unpolished state whose error clears it, so x polishes
    first, is found at its maximum, and the state is marked polished; a polished state's hyper step frees the edges
    an unpolished one holds, so its plan is a release. Read as the same plan made twice from one state, it ended both
    stages uncertified with every deviation at its edge, V's lower bound at the returned state 3145.44 in each; the
    plans from the polished states release a block in each stage and the annotated fit returns with the deviations'
    null space freed, V's lower bound 3276.26 there (W 3286.63 less its error 10.38, still uncertified)."""
    prior, cavity = _draw_two()
    base = without_annotations(prior)

    def fixed_points(model_prior):
        def solve(hyperparameters):
            points = []
            for model in hyperparameters:
                moments = tilted_moments(model_prior, model, cavity, 1 << 28)
                points.append(FixedPoint(
                    cavity=cavity, posterior=diagonal_posterior(moments.variance), mean=moments.mean,
                    precision_norm=lambda direction, variance=moments.variance: float(np.sum(np.square(direction) / variance)),
                    effective_effects=float(np.sum(cavity.precision * moments.variance)),
                ))
            return points
        return solve

    (base_fit,) = fit_hyperparameters(base, [initial_hyperparameters(base)], fixed_points(base), 1 << 28, 1.0 / 128.0)
    start = embed_hyperparameters(base, prior, base_fit.hyperparameters)
    (fit,) = fit_hyperparameters(prior, [start], fixed_points(prior), 1 << 28, 1.0 / 128.0)
    assert base_fit.iterations >= 2 and fit.iterations >= 1
    assert _released_deviation(prior, fit.hyperparameters)


def test_the_fit_log_records_each_class_length_curve_by_decade() -> None:
    rng = np.random.default_rng(57)
    classes, lengths = _length_table(rng)
    result = annotation_design({"sv_length": lengths}, {}, class_index=classes, by_class=("sv_length",), class_labels=_LABELS)
    prior = _prior(classes, result)
    theta = rng.normal(size=prior.scale_size)
    record = smooth_shift_record(prior.scale_design, prior.annotation_groups, theta, "sv_length", classes, lengths, _LABELS)
    # the SNVs' one length has no curve; each other class has the mean shift of its members in each decade of length
    parts = dict(part.split(" [") for part in record.split("; "))
    assert sorted(parts) == ["deletion", "duplication", "insertion"]
    shift = prior.scale_design @ theta
    decade = np.floor(np.log10(lengths)).astype(int)
    for position, label in enumerate(_LABELS[1:], start=1):
        rows = classes == position
        expected = ", ".join(f"1e{value}: {shift[rows & (decade == value)].mean():+.3g}" for value in np.unique(decade[rows]))
        assert parts[label] == expected + "]"
    assert smooth_shift_record(prior.scale_design, prior.annotation_groups, theta, "absent", classes, lengths, _LABELS) == ""
    # a member with no length is in no decade
    unknown = lengths.copy()
    unknown[np.flatnonzero(classes == 1)[::2]] = np.nan
    rows = (classes == 1) & np.isfinite(unknown)
    parts = dict(part.split(" [") for part in smooth_shift_record(prior.scale_design, prior.annotation_groups, theta, "sv_length", classes, unknown, _LABELS).split("; "))
    assert parts["deletion"] == ", ".join(f"1e{value}: {shift[rows & (decade == value)].mean():+.3g}" for value in np.unique(decade[rows])) + "]"


def test_the_ld_function_is_its_own_named_smooth_on_the_log_ld_score() -> None:
    rng = np.random.default_rng(58)
    classes = rng.choice(2, size=400).astype(np.int64)
    scores = np.exp(rng.uniform(0.0, 5.0, 400))
    scores[:3] = 0.0
    (name, values), = ld_score_annotation(scores).items()
    assert name == LD_SCORE
    np.testing.assert_allclose(values[3:], np.log(scores[3:]))
    result = annotation_design({LD_SCORE: values, **frequency_annotation(np.log(rng.uniform(0.02, 0.5, 400)))}, {}, class_index=classes)
    assert [group.name for group in result.groups] == [FREQUENCY, LD_SCORE]
    # a column with no LD at all (the covariates explain it) has no finite log: the smooth's missing state, not a value
    assert f"{LD_SCORE}:linear" in result.names and f"{LD_SCORE}:missing" in result.names
    np.testing.assert_array_equal(result.design[:3, result.names.index(f"{LD_SCORE}:missing")], 1.0)
