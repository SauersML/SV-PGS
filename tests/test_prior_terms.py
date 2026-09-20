"""The scale model's terms: each one's infinite-weight limit, its derivatives and its identifiability."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.prior_terms import (
    PenaltyGroup,
    TermDesign,
    ancestry_mixed_frequency,
    frequency_design,
    heterozygosity,
    local_heterozygosity,
    measurement_offset,
    shape_functionals,
    shape_score,
    smooth_term,
    smooth_values,
    stack_terms,
    sv_context_design,
    switchable_smooth,
)
from sv_pgs.scale_mixture_ep import AnnotationGroup, derived_lattice, scale_mixture_prior
from sv_pgs.store_converter import _uniform_cubic_weights, linear_recalibration, sv_kernel_features

_EPSILON = float(np.finfo(np.float64).eps)
# A cubic fitted to four equispaced points of [0, 1] has a Vandermonde condition number of about 10^2;
# with O(1) values, O(10^2) products and that conditioning, rounding stays near 10^-11. The exact
# penalty is compared at 10^-9, two orders above it and seven below any quadrature error.
_EXACT_RELATIVE = 1e-9


def _rounding_bound(terms: int, magnitude: float) -> float:
    """The forward rounding bound n eps |x| of a sum of n products of size |x|."""
    return terms * _EPSILON * magnitude


def _exact_curvature_integral(evaluate, knots: np.ndarray) -> float:
    """The integral of f''^2 over the covered knot intervals, independently of the penalty code.

    On each interval f is a cubic: fit it through four interior points in local
    coordinates, differentiate twice, and integrate the square by Simpson's rule,
    which is exact because f''^2 is quadratic there.
    """
    spacing = float(knots[1] - knots[0])
    sample = np.array([0.2, 0.4, 0.6, 0.8])
    total = 0.0
    for left in knots[3:-4]:
        cubic = np.polynomial.Polynomial.fit(sample, evaluate(left + spacing * sample), 3, domain=[0.0, 1.0], window=[0.0, 1.0])
        curvature = cubic.deriv(2)
        values = curvature(np.array([0.0, 0.5, 1.0])) / (spacing * spacing)
        total += spacing * (values[0] ** 2 + 4.0 * values[1] ** 2 + values[2] ** 2) / 6.0
    return float(total)


def _budget() -> ComputeBudget:
    return ComputeBudget(
        device_kind="cpu",
        device_ids=(),
        device_names=(),
        device_bytes=(),
        device_compute_capabilities=(),
        host_bytes=1 << 26,
        cpu_threads=1,
    )


class TestOffset:
    """Both parts of o_j carry coefficient 1 by derivation, so neither is fitted."""

    def _recalibrated(self):
        generator = np.random.default_rng(5)
        milli = generator.integers(0, 2001, size=(3, 40))
        groups = np.zeros(40, dtype=np.int64)
        return linear_recalibration(milli, groups, np.array([[0.9], [0.6], [0.8]]))

    def test_the_local_heterozygosity_is_the_recalibrated_variance_over_the_reliability(self):
        recalibrated = self._recalibrated()
        reliability = np.array([0.81, 0.36, 0.64])
        dosage = recalibrated.dosage_milli.astype(np.float64) / 1000.0
        expected = dosage.var(axis=1) / reliability
        recovered = local_heterozygosity(recalibrated=recalibrated, reliability=reliability)
        assert np.allclose(recovered, expected, rtol=_rounding_bound(40, 1.0), atol=0.0)

    def test_a_raw_dosage_or_variance_is_refused_by_type(self):
        with pytest.raises(TypeError):
            local_heterozygosity(recalibrated=np.array([[1000, 2000, 0]]), reliability=np.array([0.5]))
        with pytest.raises(TypeError):
            local_heterozygosity(recalibrated=np.array([0.2]), reliability=np.array([0.5]))

    def test_the_offset_is_the_sum_of_the_two_logs(self):
        reliability = np.array([0.8, 0.5])
        local_variance = np.array([0.32, 0.18])
        offset = measurement_offset(reliability=reliability, local_variance=local_variance)
        assert np.allclose(offset, np.log(reliability) + np.log(local_variance), rtol=_rounding_bound(2, 1.0), atol=0.0)

    def test_an_impossible_reliability_or_variance_is_refused(self):
        with pytest.raises(ValueError):
            measurement_offset(reliability=np.array([1.5]), local_variance=np.array([0.2]))
        with pytest.raises(ValueError):
            measurement_offset(reliability=np.array([0.5]), local_variance=np.array([0.0]))
        with pytest.raises(ValueError):
            local_heterozygosity(recalibrated=self._recalibrated(), reliability=np.array([0.5, 0.0, 0.5]))


class TestAncestryMixture:
    """p^sel is a point of the profile the evidence searches, never a fitted coefficient."""

    def test_the_mixture_interpolates_the_groups_and_normalizes_its_weights(self):
        group_frequencies = np.array([[0.1, 0.4], [0.02, 0.3]])
        pooled = ancestry_mixed_frequency(group_frequencies=group_frequencies, weights=np.array([0.0, 1.0]))
        assert np.allclose(pooled, group_frequencies[:, 1])
        halved = ancestry_mixed_frequency(group_frequencies=group_frequencies, weights=np.array([2.0, 2.0]))
        assert np.allclose(halved, group_frequencies.mean(axis=1))

    def test_a_frequency_outside_the_unit_interval_is_refused(self):
        with pytest.raises(ValueError):
            ancestry_mixed_frequency(group_frequencies=np.array([[1.2, 0.3]]), weights=np.array([1.0, 1.0]))


class TestSmoothTerm:
    """A smooth's penalty is the exact integrated squared second derivative."""

    def test_the_penalty_is_the_exact_integral_of_the_squared_second_derivative(self):
        generator = np.random.default_rng(0)
        values = np.sort(generator.uniform(-3.0, 3.0, size=64))
        term = smooth_term(values, spacing=0.5, name="x")
        coefficients = generator.normal(size=term.design.shape[1])
        exact = float(coefficients @ term.penalty() @ coefficients)
        independent = _exact_curvature_integral(lambda x: smooth_values(term, x) @ coefficients, term.knots)
        assert exact == pytest.approx(independent, rel=_EXACT_RELATIVE)

    def test_the_penalty_annihilates_exactly_the_straight_lines(self):
        values = np.linspace(-2.0, 2.0, 50)
        term = smooth_term(values, spacing=0.4, name="x")
        null = term.null_basis
        assert null.shape[1] == 1, "a second-derivative penalty leaves the linear direction alone"
        fitted = term.design @ null[:, 0]
        slope, intercept = np.polyfit(values, fitted, 1)
        assert np.allclose(fitted, slope * values + intercept, atol=_rounding_bound(term.design.shape[1], 1.0) * abs(slope))
        assert abs(slope) > 0.0

    def test_a_curved_direction_is_penalized(self):
        values = np.linspace(-2.0, 2.0, 50)
        term = smooth_term(values, spacing=0.4, name="x")
        curved = np.linalg.lstsq(term.design, values * values, rcond=None)[0]
        assert float(curved @ term.penalty() @ curved) > 0.0

    def test_the_class_centred_design_keeps_full_rank_so_the_engine_accepts_it(self):
        generator = np.random.default_rng(11)
        values = np.sort(generator.uniform(-1.0, 1.0, size=64))
        classes = (generator.uniform(size=values.shape[0]) < 0.5).astype(np.int64)
        design = smooth_term(values, spacing=0.3, name="x", class_index=classes).design
        centred = design.copy()
        for value in np.unique(classes):
            rows = classes == value
            centred[rows] -= centred[rows].mean(axis=0, keepdims=True)
        singular = np.linalg.svd(centred, compute_uv=False)
        assert float(singular[-1]) > _EPSILON * max(centred.shape) * float(singular[0])

    def test_a_smooth_that_the_class_levels_already_carry_is_refused(self):
        # One variant per class: every within-class centred column is exactly zero.
        values = np.linspace(-1.0, 1.0, 6)
        with pytest.raises(ValueError):
            smooth_term(values, spacing=0.3, name="x", class_index=np.arange(values.shape[0], dtype=np.int64))


class TestSwitchableSmooth:
    """Every direction is penalized, so the term's own weights switch it off exactly."""

    def test_the_infinite_weight_limit_is_zero(self):
        values = np.linspace(0.0, 3.0, 60)
        term = switchable_smooth(values, spacing=0.5, name="w")
        assert term.null_basis.shape[1] == 0
        assert [group.name for group in term.groups] == ["w linear", "w curvature"]

    def test_the_curvature_weight_alone_leaves_exactly_a_straight_line(self):
        values = np.linspace(0.0, 3.0, 60)
        term = switchable_smooth(values, spacing=0.5, name="w")
        curvature = next(group for group in term.groups if group.name == "w curvature")
        # With only the curvature weight infinite, the surviving direction is its null space.
        rest = TermDesign(design=term.design, groups=(curvature, PenaltyGroup("free", np.array([0]), np.zeros((0, 1)))), names=term.names)
        null = rest.null_basis
        assert null.shape[1] == 1
        fitted = term.design @ null[:, 0]
        slope, intercept = np.polyfit(values, fitted, 1)
        assert np.allclose(fitted, slope * values + intercept, atol=_rounding_bound(term.design.shape[1], 1.0) * abs(slope))

    def test_the_curvature_penalty_is_exact_and_full_rank(self):
        generator = np.random.default_rng(2)
        values = np.sort(generator.uniform(0.0, 3.0, size=80))
        term = switchable_smooth(values, spacing=0.5, name="w")
        curvature = next(group for group in term.groups if group.name == "w curvature")
        assert np.linalg.matrix_rank(curvature.matrix) == curvature.columns.shape[0]
        coefficients = generator.normal(size=term.design.shape[1])
        exact = float(coefficients @ term.penalty() @ coefficients) - coefficients[0] ** 2
        independent = _exact_curvature_integral(lambda x: smooth_values(term, x) @ coefficients, term.knots)
        assert exact == pytest.approx(independent, rel=_EXACT_RELATIVE)

    def test_a_variable_constant_within_every_class_gives_no_columns(self):
        classes = np.repeat(np.arange(3, dtype=np.int64), 10)
        term = switchable_smooth(classes.astype(np.float64), spacing=0.5, name="w", class_index=classes)
        assert term.design.shape == (30, 0) and term.groups == ()


class TestFrequencyTerm:
    """At an infinite weight the frequency term is exactly the power law H^(1+S)."""

    def test_the_infinite_weight_limit_is_a_power_law_in_heterozygosity(self):
        frequencies = np.linspace(0.001, 0.5, 200)
        term = frequency_design(selection_frequency=frequencies, spacing=0.5)
        null = term.null_basis
        assert null.shape[1] == 1
        log_scale = term.design @ null[:, 0]
        log_heterozygosity = np.log(heterozygosity(frequencies))
        exponent, intercept = np.polyfit(log_heterozygosity, log_scale, 1)
        assert np.allclose(log_scale, exponent * log_heterozygosity + intercept, atol=1e-9 * max(abs(exponent), 1.0))

    def test_a_finite_weight_can_hold_a_plateau_that_no_power_law_reaches(self):
        frequencies = np.concatenate([np.linspace(0.0005, 0.01, 120), np.linspace(0.02, 0.5, 120)])
        term = frequency_design(selection_frequency=frequencies, spacing=0.5)
        log_heterozygosity = np.log(heterozygosity(frequencies))
        # The derived shape: flat where kappa H s << 1, falling toward -1 at common frequency.
        truth = -0.5 * np.log1p(np.exp(log_heterozygosity - log_heterozygosity.min() - 4.0))
        target = truth - truth.mean()
        fitted = term.design @ np.linalg.lstsq(term.design, target, rcond=None)[0]
        power_law = np.polyval(np.polyfit(log_heterozygosity, target, 1), log_heterozygosity)
        assert float(np.sum((fitted - target) ** 2)) < 0.1 * float(np.sum((power_law - target) ** 2))

    def test_a_monomorphic_selection_frequency_is_refused(self):
        with pytest.raises(ValueError):
            frequency_design(selection_frequency=np.array([0.2, 0.0]), spacing=0.5)


def _kernel_features(spacing: float, record_count: int = 5, seed: int | None = None) -> object:
    if seed is None:
        starts = np.array([0, 1000, 5000, 20000, 60000], dtype=np.int64)
        is_sv = np.array([False, True, True, True, True])
        frequencies = np.array([0.0, 0.2, 0.05, 0.4, 0.1])
        classes = np.array([0, 0, 1, 0, 1], dtype=np.int64)
        lengths = np.array([1.0, 300.0, 5000.0, 120.0, 900.0])
    else:
        generator = np.random.default_rng(seed)
        starts = np.sort(generator.choice(2_000_000, size=record_count, replace=False)).astype(np.int64)
        is_sv = generator.uniform(size=record_count) < 0.3
        frequencies = generator.uniform(0.01, 0.5, size=record_count)
        classes = generator.integers(0, 2, size=record_count).astype(np.int64)
        lengths = np.exp(generator.uniform(np.log(50.0), np.log(1e5), size=record_count))
    return sv_kernel_features(
        core_starts=starts,
        core_ends=starts + 100,
        bubble_indices=np.arange(starts.shape[0], dtype=np.int64),
        is_sv=is_sv,
        frequencies=frequencies,
        sv_classes=classes,
        sv_lengths=lengths,
        class_count=2,
        spacing=spacing,
        budget=_budget(),
    )


class TestSvContextTerm:
    """Classes share one kernel; their deviations are pooled away at an infinite weight."""

    def test_the_groups_partition_the_columns_with_their_own_weights(self):
        term = sv_context_design(_kernel_features(spacing=np.log(2.0)))
        assert [group.name for group in term.groups] == ["sv kernel roughness", "sv kernel overlap", "sv kernel class deviations"]

    def test_the_infinite_weight_limit_is_a_distance_by_length_power_law_tensor(self):
        features = _kernel_features(spacing=np.log(2.0))
        term = sv_context_design(features)
        # Only the roughness has a null space: {1, x} for each of the two length terms.
        assert term.null_basis.shape[1] == 2 * features.distance.shape[3]

    def test_the_roughness_is_the_exact_integral_on_the_store_basis(self):
        # review-mathbugs F5: the kernel's roughness is the same measure as the frequency smooth's.
        features = _kernel_features(spacing=np.log(2.0))
        term = sv_context_design(features)
        basis_count, length_count = features.distance.shape[2], features.distance.shape[3]
        roughness = next(group for group in term.groups if group.name == "sv kernel roughness")
        generator = np.random.default_rng(4)
        kernel = generator.normal(size=basis_count)
        coefficients = np.zeros(roughness.columns.shape[0])
        coefficients[0::length_count] = kernel

        def evaluate(x: np.ndarray) -> np.ndarray:
            # The store's own placement: interval floor(x/h), functions interval..interval+3.
            scaled = x / features.spacing
            interval = np.floor(scaled).astype(np.int64)
            weights = _uniform_cubic_weights(scaled - interval)
            return sum(weight * kernel[interval + shift] for shift, weight in enumerate(weights))

        knots = features.spacing * (np.arange(basis_count + 4) - 3.0)
        exact = float(coefficients @ roughness.matrix @ coefficients)
        assert exact == pytest.approx(_exact_curvature_integral(evaluate, knots), rel=_EXACT_RELATIVE)

    def test_a_pooled_column_is_the_sum_of_the_class_features(self):
        features = _kernel_features(spacing=np.log(2.0))
        term = sv_context_design(features)
        pooled_width = features.distance.shape[2] * features.distance.shape[3]
        expected = features.distance.sum(axis=1).reshape(features.distance.shape[0], -1)
        assert np.allclose(term.design[:, :pooled_width], expected)

    def test_a_malformed_feature_set_is_refused(self):
        features = _kernel_features(spacing=np.log(2.0))
        broken = type(features)(overlap=features.overlap[:, :1], distance=features.distance, spacing=features.spacing)
        with pytest.raises(ValueError):
            sv_context_design(broken)


class TestShapeDirections:
    """The shape directions are what a scale term cannot reach."""

    def test_the_functionals_are_orthogonal_to_translation_and_normalization(self):
        nodes = np.linspace(-6.0, 2.0, 41)
        log_weights = -0.5 * np.square(nodes + 2.0)
        functionals = shape_functionals(nodes=nodes, log_weights=log_weights)
        weights = np.exp(log_weights - log_weights.max())
        weights /= weights.sum()
        assert np.allclose(weights @ functionals, 0.0, atol=_rounding_bound(nodes.shape[0], 1.0))
        assert np.allclose((weights * nodes) @ functionals, 0.0, atol=_rounding_bound(nodes.shape[0], float(np.abs(nodes).max())))

    def test_a_translated_density_scores_zero_and_a_reweighted_one_does_not(self):
        nodes = np.linspace(-6.0, 2.0, 81)
        log_weights = -0.5 * np.square(nodes + 2.0)
        functionals = shape_functionals(nodes=nodes, log_weights=log_weights)
        weights = np.exp(log_weights - log_weights.max())
        weights /= weights.sum()

        translated = np.exp(-0.5 * np.square(nodes + 1.0))
        translated /= translated.sum()
        reweighted = weights * np.exp(0.6 * functionals[:, 0])
        reweighted /= reweighted.sum()

        annotation = np.ones(2)
        responsibilities = np.stack([translated, weights])
        scores = shape_score(responsibilities=responsibilities, functionals=functionals, annotation=annotation, log_weights=log_weights)
        assert abs(float(scores[1, 0])) < _rounding_bound(nodes.shape[0], 1.0)

        shifted = shape_score(
            responsibilities=np.stack([reweighted, weights]), functionals=functionals, annotation=annotation, log_weights=log_weights
        )
        assert abs(float(shifted[0, 0])) > abs(float(scores[0, 0]))


def _engine_prior(stacked, classes, offset):
    # Effects with signal above the noise: with none, derived_lattice's kernel range is empty (floor = top) and
    # its three nodes are fewer than the engine's roughness order needs, an engine-side case reported to e2e.
    generator = np.random.default_rng(9)
    effects = generator.normal(scale=1.0, size=classes.shape[0])
    nodes, floor, top = derived_lattice(np.full(classes.shape[0], 4.0), effects * 4.0, offset, tolerance=1e-3)
    return scale_mixture_prior(
        class_index=classes,
        log_variance_offset=offset,
        annotation_design=stacked.design,
        annotation_groups=[AnnotationGroup(columns=columns, penalty=penalty) for columns, penalty in stacked.groups],
        nodes=nodes,
        floor=floor,
        top=top,
    )


class TestComposition:
    """The stacked terms are what the engine takes as its annotation design and groups."""

    def test_frequency_and_kernel_terms_build_a_prior_the_engine_accepts(self):
        generator = np.random.default_rng(3)
        variant_count = 400
        features = _kernel_features(spacing=np.log(2.0), record_count=variant_count, seed=1)
        classes = generator.integers(0, 2, size=variant_count).astype(np.int64)
        frequencies = generator.uniform(0.01, 0.5, size=variant_count)
        reliability = generator.uniform(0.3, 1.0, size=variant_count)
        stacked = stack_terms(
            frequency_design(selection_frequency=frequencies, spacing=0.5, class_index=classes),
            sv_context_design(features),
            class_index=classes,
        )
        offset = measurement_offset(reliability=reliability, local_variance=heterozygosity(frequencies))
        prior = _engine_prior(stacked, classes, offset)
        assert prior.scale_size == stacked.design.shape[1]
        assert len(stacked.names) == stacked.design.shape[1]
        assert "frequency roughness" in stacked.group_names and "sv kernel roughness" in stacked.group_names

    def test_stacking_drops_what_the_engine_cannot_identify_and_maps_back(self):
        features = _kernel_features(spacing=np.log(2.0), record_count=300, seed=2)
        term = sv_context_design(features)
        stacked = stack_terms(term)
        # Bases beyond the largest gap are zero columns; they cannot survive.
        zero_columns = int(np.sum(np.all(term.design == 0.0, axis=0)))
        assert zero_columns > 0 and stacked.design.shape[1] <= term.design.shape[1] - zero_columns
        # The kept columns are exactly the term's columns through the loading.
        assert np.allclose(stacked.design, term.design @ stacked.loading, atol=_rounding_bound(term.design.shape[1], float(np.abs(term.design).max())))

    def test_stacking_keeps_the_penalty_null_space_first_so_the_power_law_survives(self):
        frequencies = np.linspace(0.001, 0.5, 200)
        term = frequency_design(selection_frequency=frequencies, spacing=0.5)
        stacked = stack_terms(term)
        penalty = stacked.groups[0][1]
        eigenvalues, eigenvectors = np.linalg.eigh(penalty)
        null = eigenvectors[:, eigenvalues <= _EPSILON * penalty.shape[0] * float(eigenvalues[-1])]
        assert null.shape[1] == 1
        log_scale = stacked.design @ null[:, 0]
        log_heterozygosity = np.log(heterozygosity(frequencies))
        exponent, intercept = np.polyfit(log_heterozygosity, log_scale, 1)
        assert np.allclose(log_scale, exponent * log_heterozygosity + intercept, atol=1e-9 * max(abs(exponent), 1.0))

    def test_stacking_refuses_terms_of_different_lengths(self):
        first = smooth_term(np.linspace(0.0, 1.0, 10), spacing=0.5, name="a")
        second = smooth_term(np.linspace(0.0, 1.0, 11), spacing=0.5, name="b")
        with pytest.raises(ValueError):
            stack_terms(first, second)

    def test_a_term_needs_one_name_per_column_and_groups_that_partition_them(self):
        with pytest.raises(ValueError):
            TermDesign(design=np.zeros((3, 2)), groups=(), names=("only_one",))
        with pytest.raises(ValueError):
            overlapping = (PenaltyGroup("a", np.array([0, 1]), np.eye(2)), PenaltyGroup("b", np.array([1]), np.eye(1)))
            TermDesign(design=np.zeros((3, 2)), groups=overlapping, names=("x", "y"))
