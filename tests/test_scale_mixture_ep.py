"""The variant-side EP-EB engine against references written independently of it.

References: quadrature of the tilted density, finite differences of the
objectives it differentiates analytically, and the REML maximum of a
Gaussian-prior regression for the noise update.
"""
from __future__ import annotations

import pathlib
import weakref
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

import sv_pgs.scale_mixture_ep as engine
from sv_pgs.scale_mixture_ep import (
    _sum_to_zero_basis,
    _sum_to_zero_rows,
    CurvatureCorrection,
    FixedPoint,
    fit_hyperparameters,
    AnnotationGroup,
    Cavity,
    GaussianPosterior,
    MixtureHyperparameters,
    _components,
    _data_objective,
    _evidence,
    _curvature_trace_gradient,
    _data_value,
    _directional_derivatives,
    _corrected,
    _ascend_evidence,
    _best_certified,
    _laplace_corrections,
    _lines,
    _column_log_sums,
    _kernel_terms,
    _moving_line_rows,
    _log_normal_start,
    _maximize_coefficients,
    _penalized,
    _penalty_matrix,
    _proposal,
    _newton_b,
    _penalty_value,
    _restricted_prior,
    _smoothing_bounds,
    curvature_correction,
    class_log_density,
    derived_lattice,
    halved_lattice,
    relattice,
    hyper_step,
    initial_hyperparameters,
    moment_start,
    kernel_floor,
    kernel_top,
    log_scale,
    moment_matched_prior_sites,
    noise_gain,
    noise_variance,
    prior_second_moment,
    roughness_factor,
    scale_mixture_prior,
    site_targets,
    spacing_bound,
    tilted_cumulants,
    tilted_moments,
)
from tests.oracles import INDEPENDENT_EFFECTS, _line, _line_log_integral, _total_curvature, cavities, diagonal_posterior, quadrature_majorant_ratio

_WORKING_BYTES = 1 << 20
# The evidence resolution a fit certifies for a scorer with 64 posterior draws, 1/(2K) nats.
_EVIDENCE_TOLERANCE = 1.0 / 128.0
# The scenario's input: the tolerance on sum_j log Z_j that the derived lattices are built to (kernel_floor and
# spacing_bound), as Stage 0 builds production's to 1/(2K).
_LATTICE_TOLERANCE = 1e-3


def _second_difference(size: int) -> np.ndarray:
    difference = np.diff(np.eye(size), n=2, axis=0)
    return difference.T @ difference


def _data(variant_count: int, seed: int):
    """Two classes, a discrete annotation column and a three-column smooth, and normal-means cavities."""
    generator = np.random.default_rng(seed)
    class_index = (generator.random(variant_count) < 0.35).astype(np.int64)
    offset = np.log(generator.uniform(0.3, 1.0, variant_count))
    position = generator.uniform(-1.0, 1.0, variant_count)
    design = np.column_stack([(generator.random(variant_count) < 0.4).astype(np.float64), position, position**2, position**3])
    groups = (
        AnnotationGroup(columns=np.array([0]), penalty=np.eye(1)),
        AnnotationGroup(columns=np.array([1, 2, 3]), penalty=_second_difference(3)),
    )
    precision = generator.uniform(50.0, 400.0, variant_count)
    effect = np.where(generator.random(variant_count) < 0.3, generator.normal(0.0, 0.25, variant_count), 0.0)
    shift = precision * (effect + generator.standard_normal(variant_count) / np.sqrt(precision))
    return class_index, offset, design, groups, Cavity(precision=precision, shift=shift)


def _problem(*, variant_count: int, seed: int, node_count: int = 0, offset_group_count: int = 0):
    """With ``node_count`` a fixed lattice whose kernels are all active; otherwise the derived floor, top and spacing.
    With ``offset_group_count`` the variants are dealt round-robin into that many offset groups with learned levels."""
    class_index, offset, design, groups, cavity = _data(variant_count, seed)
    if node_count:
        nodes = np.linspace(np.log(1e-5), np.log(0.5), node_count)
        floor, top = nodes[0] - 1.0, nodes[-1]
    else:
        floor = kernel_floor(cavity.precision, cavity.shift, offset, _LATTICE_TOLERANCE)
        top = kernel_top(cavity.precision, cavity.shift, offset, floor)
        spacing = spacing_bound(float(variant_count), _LATTICE_TOLERANCE)
        nodes = np.arange(floor - 12.0 * spacing, top + 12.0 * spacing, spacing)
    prior = scale_mixture_prior(
        class_index=class_index,
        log_variance_offset=offset,
        annotation_design=design,
        annotation_groups=groups,
        nodes=nodes,
        floor=floor,
        top=top,
        offset_groups=np.arange(variant_count) % offset_group_count if offset_group_count else None,
    )
    return prior, cavity


_FLAT_REASON = (
    "open engine work (lead, 2026-09-21): at these weights the search ends with a block's weight far down its range "
    "(rho = -3.9 on the normal-means problem) where -H is near-flat along a penalized direction, so V's inner-maximizer "
    "sensitivity bound is O(nats) at a decrement of 1e-26 and the weights' stationarity bound is infinite; the fit "
    "returns honestly uncertified. The certificate's composition (per-decision tightening) is not what stops it here."
)


_FOLD_REASON = (
    "two-sided stationarity certificate cannot certify a maximum at its basin's fold boundary: the rho-search can "
    "stop where the base's inner basin ends within 1e-5 in rho and the neighbouring basin lies inside the sides' "
    "certified errors, and which way rounding falls follows BLAS threads and kernel. Removed by lane/engine-stationarity "
    "(a boundary-aware certificate with curvature certified over the step)."
)


def _hyperparameters(prior, seed: int, log_smoothing: float | None = None) -> MixtureHyperparameters:
    """The start density with a random perturbation of every coefficient, and random or given penalty weights."""
    generator = np.random.default_rng(seed)
    start = initial_hyperparameters(prior).coefficients
    coefficients = start + 0.3 * generator.standard_normal(prior.coefficient_size)
    count = len(prior.smoothing_blocks)
    weights = generator.uniform(-1.0, 2.0, count) if log_smoothing is None else np.full(count, log_smoothing)
    return MixtureHyperparameters(coefficients=coefficients, log_smoothing=weights)


def assert_tilted_moments_match_quadrature(prior, hyperparameters, cavity, moments) -> None:
    """Every effect's log Z, tilted mean and tilted variance against quadrature of the mixture times the cavity."""
    weights = np.exp(class_log_density(prior, hyperparameters.coefficients))
    scales = np.exp(log_scale(prior, hyperparameters.coefficients))
    for variant in range(prior.variant_count):
        variances = scales[variant] * np.exp(prior.log_variance_grid)
        mixture = weights[prior.class_index[variant]]
        precision, shift = cavity.precision[variant], cavity.shift[variant]
        center = shift / (precision + 1.0 / variances.max())
        width = 12.0 / np.sqrt(precision)

        def tilted(beta: float, power: int) -> float:
            prior_density = np.sum(mixture * np.exp(-0.5 * beta * beta / variances) / np.sqrt(2.0 * np.pi * variances))
            return beta**power * prior_density * np.exp(-0.5 * precision * beta * beta + shift * beta - 0.5 * shift * shift / precision)

        integrals = [
            quad(tilted, center - width, center + width, args=(power,), points=[0.0], limit=400, epsabs=0.0, epsrel=1e-12)[0]
            for power in (0, 1, 2)
        ]
        mean = integrals[1] / integrals[0]
        log_normalizer = np.log(integrals[0]) + 0.5 * shift * shift / precision
        np.testing.assert_allclose(moments.log_normalizer[variant], log_normalizer, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(moments.mean[variant], mean, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(moments.variance[variant], integrals[2] / integrals[0] - mean * mean, rtol=1e-8)


def test_tilted_moments_match_quadrature_of_the_mixture_times_the_cavity():
    prior, cavity = _problem(variant_count=12, seed=1, node_count=9)
    hyperparameters = _hyperparameters(prior, 2)
    assert_tilted_moments_match_quadrature(prior, hyperparameters, cavity, tilted_moments(prior, hyperparameters, cavity, _WORKING_BYTES))


def test_a_site_update_gives_a_one_variant_posterior_the_exact_tilted_moments():
    prior, cavity = _problem(variant_count=30, seed=3, node_count=11)
    hyperparameters = _hyperparameters(prior, 4)
    old_precision, old_shift = moment_matched_prior_sites(prior, hyperparameters)
    # Orthogonal design: q's marginal is the likelihood times the site, so the cavity is the likelihood.
    posterior_variance = 1.0 / (cavity.precision + old_precision)
    posterior_mean = (cavity.shift + old_shift) * posterior_variance
    recovered = cavities(posterior_mean, posterior_variance, old_precision, old_shift)
    np.testing.assert_allclose(recovered.precision, cavity.precision, rtol=1e-12)
    np.testing.assert_allclose(recovered.shift, cavity.shift, rtol=1e-12)
    moments = tilted_moments(prior, hyperparameters, recovered, _WORKING_BYTES)
    new_precision, new_shift = site_targets(moments, recovered)
    variance = 1.0 / (cavity.precision + new_precision)
    np.testing.assert_allclose((cavity.shift + new_shift) * variance, moments.mean, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(variance, moments.variance, rtol=1e-10)


def test_moment_matched_prior_sites_carry_the_prior_second_moment():
    prior, _cavity = _problem(variant_count=20, seed=5, node_count=7)
    hyperparameters = _hyperparameters(prior, 6)
    precision, shift = moment_matched_prior_sites(prior, hyperparameters)
    weights = np.exp(class_log_density(prior, hyperparameters.coefficients))
    scales = np.exp(log_scale(prior, hyperparameters.coefficients))
    expected = scales * (weights[prior.class_index] @ np.exp(prior.log_variance_grid))
    np.testing.assert_allclose(prior_second_moment(prior, hyperparameters), expected, rtol=1e-12)
    np.testing.assert_allclose(precision, 1.0 / expected, rtol=1e-12)
    np.testing.assert_array_equal(shift, 0.0)


@pytest.mark.parametrize("offset_group_count", [0, 3])
def test_penalized_gradient_and_hessian_match_finite_differences(offset_group_count):
    prior, cavity = _problem(variant_count=25, seed=9, node_count=8, offset_group_count=offset_group_count)
    hyperparameters = _hyperparameters(prior, 10)
    penalty = _penalty_matrix(prior, hyperparameters.log_smoothing)
    coefficients = hyperparameters.coefficients

    def evaluate(point):
        return _penalized(prior, _data_objective(prior, point, cavity, _WORKING_BYTES), hyperparameters.log_smoothing, penalty, point)

    _value, gradient, hessian = evaluate(coefficients)
    step = 1e-6
    numerical_gradient = np.empty_like(coefficients)
    numerical_hessian = np.empty((coefficients.shape[0], coefficients.shape[0]))
    for coordinate, unit in enumerate(np.eye(coefficients.shape[0])):
        forward, backward = evaluate(coefficients + step * unit), evaluate(coefficients - step * unit)
        numerical_gradient[coordinate] = (forward[0] - backward[0]) / (2.0 * step)
        numerical_hessian[:, coordinate] = (forward[1] - backward[1]) / (2.0 * step)
    np.testing.assert_allclose(gradient, numerical_gradient, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(hessian, numerical_hessian, rtol=1e-5, atol=1e-5)


def test_component_derivatives_in_log_scale_match_finite_differences():
    prior, cavity = _problem(variant_count=15, seed=11, node_count=6)
    hyperparameters = _hyperparameters(prior, 12)
    log_density = class_log_density(prior, hyperparameters.coefficients)
    scales = log_scale(prior, hyperparameters.coefficients)

    def terms_at(variant: int, shift_in_log_scale: float):
        return _components(
            log_density[prior.class_index[variant]],
            scales[variant : variant + 1] + shift_in_log_scale,
            prior.log_variance_grid,
            cavity.precision[variant : variant + 1],
            cavity.shift[variant : variant + 1],
        )

    step, wide = 1e-4, 1e-2
    for variant in range(prior.variant_count):
        values = [terms_at(variant, multiple * step).log_normalizer[0] for multiple in (-1, 0, 1)]
        # A third difference divides rounding by step^3, so it takes a wider step (truncation O(wide^2)).
        outer = [terms_at(variant, multiple * wide).log_normalizer[0] for multiple in (-2, -1, 1, 2)]
        terms = terms_at(variant, 0.0)
        weights = terms.responsibility[0]
        mean_first = float(weights @ terms.first[0])
        centred = terms.first[0] - mean_first
        np.testing.assert_allclose(mean_first, (values[2] - values[0]) / (2.0 * step), rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(
            float(weights @ (centred**2 + terms.second[0])), (values[2] - 2.0 * values[1] + values[0]) / step**2, rtol=1e-5, atol=1e-6
        )
        third_cumulant = float(weights @ (centred**3 + 3.0 * centred * terms.second[0] + terms.third[0]))
        numerical_third = (outer[3] - 2.0 * outer[2] + 2.0 * outer[1] - outer[0]) / (2.0 * wide**3)
        np.testing.assert_allclose(third_cumulant, numerical_third, rtol=1e-3, atol=1e-4)


def test_roughness_factors_are_the_lattice_integrals_of_the_squared_derivatives():
    spacing = 0.05
    nodes = -2.0 + spacing * np.arange(101)
    for order in (1, 2, 3):
        factor = roughness_factor(nodes.shape[0], spacing, order)
        # Every polynomial below the order is in the null space; t^order has the constant difference order! h^order.
        for degree in range(order):
            np.testing.assert_allclose(factor @ nodes**degree, 0.0, atol=1e-6)
        expected = float(np.prod(np.arange(1, order + 1))) ** 2 * spacing * (nodes.shape[0] - order)
        np.testing.assert_allclose(np.sum(np.square(factor @ nodes**order)), expected, rtol=1e-6)
    # The K - 3 third differences are midpoint cells centred on t_1.5 .. t_(K-2.5): together [t_1, t_(K-2)].
    third = roughness_factor(nodes.shape[0], spacing, 3)
    start, stop = nodes[1], nodes[-2]
    exact = 0.5 * (stop - start) + 0.25 * (np.sin(2.0 * stop) - np.sin(2.0 * start))
    np.testing.assert_allclose(np.sum(np.square(third @ np.sin(nodes))), exact, rtol=2e-3)


def test_halving_the_lattice_keeps_every_class_density():
    class_index, offset, design, groups, _cavity = _data(30, 13)
    nodes = np.arange(-8.0, 3.0, 0.05)
    prior = scale_mixture_prior(
        class_index=class_index, log_variance_offset=offset, annotation_design=design, annotation_groups=groups,
        nodes=nodes, floor=nodes[0], top=nodes[-1],
    )
    # A smooth pooled shape that decays to e^-47 at the lattice ends, so the trapezoid sums have no end error.
    bump = -np.square((nodes + 2.5) / 0.8) + 0.3 * np.sin(2.0 * nodes)
    coefficients = np.zeros(prior.coefficient_size)
    coefficients[: prior.pooled_size] = prior.coefficient_map[: prior.grid_size, : prior.pooled_size].T @ (bump - bump.mean())
    hyperparameters = MixtureHyperparameters(coefficients=coefficients, log_smoothing=np.zeros(len(prior.smoothing_blocks)))
    finer, transferred = halved_lattice(prior, hyperparameters)
    fine_density = np.exp(class_log_density(finer, transferred.coefficients))
    # The children sit at different t's: compare against the same analytic density evaluated there.
    child_nodes = finer.log_variance_grid
    child_bump = -np.square((child_nodes + 2.5) / 0.8) + 0.3 * np.sin(2.0 * child_nodes)
    reference = np.exp(child_bump - np.max(child_bump))
    reference /= reference.sum()
    np.testing.assert_allclose(fine_density, np.broadcast_to(reference, fine_density.shape), rtol=2e-3, atol=1e-12)
    np.testing.assert_allclose(log_scale(finer, transferred.coefficients), log_scale(prior, hyperparameters.coefficients), atol=1e-12)


def test_halving_a_three_class_lattice_keeps_every_class_density():
    # review-mathbugs L-0: the natural end conditions were scalars for a spline over C classes, which scipy accepted
    # only when C equalled their count (two).
    generator = np.random.default_rng(71)
    count = 90
    class_index = np.repeat(np.arange(3), count // 3)
    nodes = np.linspace(np.log(1e-5), np.log(0.5), 12)
    prior = scale_mixture_prior(
        class_index=class_index, log_variance_offset=np.log(generator.uniform(0.3, 1.0, count)), annotation_design=np.zeros((count, 0)),
        annotation_groups=(), nodes=nodes, floor=nodes[0] - 1.0, top=nodes[-1],
    )
    hyperparameters = _hyperparameters(prior, 72)
    finer, moved = halved_lattice(prior, hyperparameters)
    # Compare each class independently against its scalar natural spline at the child midpoints.
    # The midpoints differ from the old nodes; comparing every other value would test different t's.
    from scipy.interpolate import make_interp_spline

    child_nodes = finer.log_variance_grid
    inside = (child_nodes >= nodes[0]) & (child_nodes <= nodes[-1])
    fine = engine._density_and_scale(finer, moved.coefficients)[0][:, inside]
    coarse = engine._density_and_scale(prior, hyperparameters.coefficients)[0]
    natural = [(3, 0.0), (4, 0.0)]
    reference = np.asarray([
        make_interp_spline(nodes, values, k=5, bc_type=(natural, natural))(child_nodes[inside])
        for values in coarse
    ])
    np.testing.assert_allclose(np.diff(fine, axis=1), np.diff(reference, axis=1), rtol=1e-9, atol=1e-9)


def test_the_layout_is_a_shared_density_plus_class_deviations_and_the_annotations():
    prior, _cavity = _problem(variant_count=40, seed=7, node_count=12)
    hyperparameters = _hyperparameters(prior, 8)
    coefficients = hyperparameters.coefficients
    basis = prior.coefficient_map[: prior.grid_size, : prior.pooled_size]
    density = (prior.coefficient_map[: prior.density_size] @ coefficients).reshape(prior.class_count, prior.grid_size)
    pooled = basis @ coefficients[: prior.pooled_size]
    for class_position in range(prior.class_count):
        start = prior.pooled_size * (class_position + 1)
        np.testing.assert_allclose(density[class_position], pooled + basis @ coefficients[start : start + prior.pooled_size], atol=1e-12)
    theta = coefficients[prior.pooled_size * (prior.class_count + 1) :]
    np.testing.assert_allclose(log_scale(prior, coefficients), prior.log_variance_offset + prior.scale_design @ theta, atol=1e-12)
    # Profiled: eta_bar's location and width (inside its coordinates) only; the deviations are fully penalized, and
    # the smooth annotation's second-difference null space (two of its three columns) has its own learned ridge.
    assert prior.null_basis.shape[1] == 2
    np.testing.assert_allclose(prior.null_basis[prior.pooled_size :], 0.0, atol=1e-10)
    names = [block.name for block in prior.smoothing_blocks]
    assert names == [
        "pooled roughness", "class 0 deviation roughness", "class 1 deviation roughness",
        "deviation location and width", "annotation group 0", "annotation group 1", "annotation group 1 null space",
    ]
    (null_block,) = (block for block in prior.smoothing_blocks if block.name == "annotation group 1 null space")
    assert null_block.factor.shape[0] == 2
    np.testing.assert_allclose(null_block.factor @ _second_difference(3), 0.0, atol=1e-12)


@pytest.mark.parametrize("node_count", [8, 17, 33])
@pytest.mark.parametrize("offset_group_count", [0, 3])
def test_known_prior_null_space_matches_the_dense_penalty_reference(node_count, offset_group_count):
    prior, _ = _problem(variant_count=40, seed=7, node_count=node_count, offset_group_count=offset_group_count)
    penalty = np.zeros((prior.coefficient_size, prior.coefficient_size))
    for block in prior.smoothing_blocks:
        penalty[np.ix_(block.coordinates, block.coordinates)] += block.factor.T @ block.factor
    values, vectors = np.linalg.eigh(penalty)
    reference = vectors[:, :2]
    assert values[2] > 1e-10
    np.testing.assert_allclose(prior.null_basis.T @ prior.null_basis, np.eye(2), atol=1e-13)
    np.testing.assert_allclose(prior.null_basis @ prior.null_basis.T, reference @ reference.T, atol=2e-9)


def test_fine_lattice_does_not_turn_a_proper_annotation_penalty_into_a_profiled_direction():
    nodes = np.linspace(-1e-3, 1e-3, 17)
    prior = scale_mixture_prior(
        class_index=np.arange(40) % 2, log_variance_offset=np.zeros(40),
        annotation_design=np.arange(40, dtype=float)[:, None],
        annotation_groups=(AnnotationGroup(columns=np.array([0]), penalty=np.eye(1)),),
        nodes=nodes, floor=nodes[0], top=nodes[-1],
    )
    assert prior.null_basis.shape == (prior.coefficient_size, 2)
    np.testing.assert_array_equal(prior.null_basis[prior.pooled_size:], 0.0)


@pytest.mark.parametrize("offset_group_count", [0, 3])
def test_curvature_trace_gradient_matches_finite_differences(offset_group_count):
    prior, cavity = _problem(variant_count=30, seed=15, node_count=12, offset_group_count=offset_group_count)
    hyperparameters = _hyperparameters(prior, 16)
    coefficients = hyperparameters.coefficients
    mapping = prior.coefficient_map
    generator = np.random.default_rng(17)
    root = generator.standard_normal((prior.coefficient_size, prior.coefficient_size))
    weight = root @ root.T / prior.coefficient_size

    def trace_at(point):
        return -float(np.sum(weight * (mapping.T @ _data_objective(prior, point, cavity, _WORKING_BYTES).hessian @ mapping)))

    analytic = _curvature_trace_gradient(prior, coefficients, cavity, weight, _WORKING_BYTES)
    step = 1e-6
    numerical = np.array([(trace_at(coefficients + step * unit) - trace_at(coefficients - step * unit)) / (2.0 * step) for unit in np.eye(coefficients.shape[0])])
    np.testing.assert_allclose(analytic, numerical, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("offset_group_count", [0, 3])
def test_evidence_gradient_in_the_log_weights_matches_finite_differences(offset_group_count):
    # Also with a correction C = B - A that is not zero (a PSD one of the form M'(B_z - A_z)M): the gradient is the
    # B-evidence's own, with W_B in its trace terms, not the fixed-cavity form's.
    prior, cavity = _problem(variant_count=60, seed=17, node_count=12, offset_group_count=offset_group_count)
    hyperparameters = _hyperparameters(prior, 18, log_smoothing=2.0)
    mapping = prior.coefficient_map
    for correction in (INDEPENDENT_EFFECTS, CurvatureCorrection(coefficient_map=mapping, matrix=mapping.T @ (0.3 * np.eye(mapping.shape[0])) @ mapping)):
        evidence = _evidence(prior, hyperparameters.log_smoothing, hyperparameters.coefficients, cavity, correction, _WORKING_BYTES, 0.0)
        assert evidence is not None and evidence.newton_decrement < 1e-12
        step = 1e-4
        numerical = []
        for unit in np.eye(hyperparameters.log_smoothing.shape[0]):
            forward = _evidence(prior, hyperparameters.log_smoothing + step * unit, evidence.coefficients, cavity, correction, _WORKING_BYTES, 0.0)
            backward = _evidence(prior, hyperparameters.log_smoothing - step * unit, evidence.coefficients, cavity, correction, _WORKING_BYTES, 0.0)
            numerical.append((forward.value - backward.value) / (2.0 * step))
        np.testing.assert_allclose(evidence.gradient, np.array(numerical), rtol=1e-5, atol=1e-7)


def test_hyper_step_reaches_a_maximum_of_the_evidence():
    prior, cavity = _problem(variant_count=150, seed=19)
    step = hyper_step(prior, initial_hyperparameters(prior), cavity, INDEPENDENT_EFFECTS, _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    fitted = step.hyperparameters
    infinite = frozenset(int(position) for position in np.flatnonzero(fitted.log_smoothing == np.inf))
    view, allowed = _restricted_prior(prior, infinite)
    weights = fitted.log_smoothing[np.isfinite(fitted.log_smoothing)]
    correction = INDEPENDENT_EFFECTS
    laplace = _evidence(view, weights, allowed.T @ fitted.coefficients, cavity, correction, _WORKING_BYTES, 0.0)
    assert laplace is not None and laplace.newton_decrement < 1e-10
    # Recomputations of this directional evidence approximation agree within its numerical tolerance.
    base = _corrected(view, weights, laplace, cavity, correction, _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    assert base is not None and abs(step.evidence - base.value) <= _EVIDENCE_TOLERANCE
    for unit in np.eye(weights.shape[0]):
        for direction in (-1.0, 1.0):
            moved_weights = weights + direction * 0.05 * unit
            moved = _corrected(
                view, moved_weights, _evidence(view, moved_weights, base.coefficients, cavity, correction, _WORKING_BYTES, 0.0), cavity, correction,
                _WORKING_BYTES, _EVIDENCE_TOLERANCE,
            )
            # The step certifies that a nearby move gains at most the tolerance, and each computed V is certified to it:
            # the computed difference is at most three tolerances.
            assert moved is None or moved.value <= base.value + 3.0 * _EVIDENCE_TOLERANCE
    # The B-evidence's own stationarity is certified: a Newton step on its differences gains at most the tolerance.
    assert step.stationarity_gain <= _EVIDENCE_TOLERANCE


def test_the_fit_does_not_depend_on_the_lattice_spacing():
    prior, cavity = _problem(variant_count=150, seed=19)
    coarse = hyper_step(prior, initial_hyperparameters(prior), cavity, INDEPENDENT_EFFECTS, _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    finer, start = halved_lattice(prior, coarse.hyperparameters)
    # At the same hyperparameters the two lattices are quadratures of one model with the same floor: the spacing bound
    # holds the trapezoid error of sum_j log Z_j to the tolerance the lattice is built to on each, so the two sums
    # agree to twice it. (Two separate fits agree only as far as V, flat to the tolerance at its maximum, determines
    # the hyperparameters, so their means are not compared.)
    coarse_moments = tilted_moments(prior, coarse.hyperparameters, cavity, _WORKING_BYTES)
    transferred_moments = tilted_moments(finer, start, cavity, _WORKING_BYTES)
    assert abs(float(np.sum(transferred_moments.log_normalizer - coarse_moments.log_normalizer))) <= 2.0 * _LATTICE_TOLERANCE
    # This fixture's refitted directional objective agrees across the grids within the numerical tolerance.
    fine = hyper_step(finer, start, cavity, INDEPENDENT_EFFECTS, _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    np.testing.assert_allclose(fine.evidence, coarse.evidence, atol=2.0 * _EVIDENCE_TOLERANCE)


def test_noise_update_reaches_the_reml_variance_of_a_gaussian_prior_regression():
    generator = np.random.default_rng(21)
    sample_count, variant_count = 150, 20
    genotypes = generator.standard_normal((sample_count, variant_count))
    covariates = np.column_stack([np.ones(sample_count), generator.standard_normal(sample_count)])
    site_precision = generator.uniform(2.0, 20.0, variant_count)
    targets = covariates @ np.array([1.0, -0.5]) + genotypes @ (generator.standard_normal(variant_count) / np.sqrt(site_precision))
    targets += 0.8 * generator.standard_normal(sample_count)
    projector = np.eye(sample_count) - covariates @ np.linalg.solve(covariates.T @ covariates, covariates.T)

    def negative_reml(log_variance: float) -> float:
        covariance = np.exp(log_variance) * np.eye(sample_count) + genotypes @ (genotypes.T / site_precision[:, None])
        inverse = np.linalg.inv(covariance)
        cross = covariates.T @ inverse @ covariates
        residual_projector = inverse - inverse @ covariates @ np.linalg.solve(cross, covariates.T @ inverse)
        return 0.5 * (np.linalg.slogdet(covariance)[1] + np.linalg.slogdet(cross)[1] + targets @ residual_projector @ targets)

    expected = np.exp(minimize_scalar(negative_reml, bounds=(-6.0, 3.0), method="bounded", options={"xatol": 1e-12}).x)
    variance = 1.0
    for _iteration in range(200):
        precision = genotypes.T @ projector @ genotypes / variance + np.diag(site_precision)
        covariance = np.linalg.inv(precision)
        mean = covariance @ (genotypes.T @ projector @ targets) / variance
        residual = projector @ (targets - genotypes @ mean)
        variance = noise_variance(
            residual_sum_of_squares=float(residual @ residual),
            sample_count=sample_count,
            covariate_count=covariates.shape[1],
            site_precision=site_precision,
            posterior_variance=np.diag(covariance),
            noise=variance,
        )
    # The reference maximizer is located to half of double precision in log variance.
    np.testing.assert_allclose(variance, expected, rtol=1e-6)


def test_the_noise_update_is_positive_and_its_gain_nonnegative_where_effects_outnumber_samples():
    # gamma near p = 400 exceeds n - k = 290: the MacKay form has no solution there, the stationarity form does.
    site_precision = np.full(400, 1e-8)
    posterior_variance = np.full(400, 1e-3)
    updated = noise_variance(
        residual_sum_of_squares=100.0, sample_count=300, covariate_count=10, site_precision=site_precision,
        posterior_variance=posterior_variance, noise=0.5,
    )
    assert updated > 0.0
    for ratio in (1e-3, 0.5, 1.0, 2.0, 1e3):
        assert noise_gain(0.5 * ratio, 0.5, 300, 10) >= 0.0


def test_the_floor_bounds_the_flat_kernel_error_and_the_top_is_the_largest_mode():
    generator = np.random.default_rng(23)
    precision = generator.uniform(10.0, 1000.0, 50)
    shift = precision * generator.normal(0.0, 0.3, 50)
    offset = np.log(generator.uniform(0.2, 1.0, 50))
    tolerance = 1e-4
    floor = kernel_floor(precision, shift, offset, tolerance)
    variance = np.exp(offset + floor)
    log_kernel = 0.5 * np.square(shift) * variance / (1.0 + variance * precision) - 0.5 * np.log1p(variance * precision)
    assert np.sum(np.abs(log_kernel)) <= tolerance
    bound = 0.5 * variance * np.abs(np.square(shift) - precision) + 0.25 * np.square(variance * precision)
    np.testing.assert_allclose(np.sum(bound), tolerance, rtol=1e-10)
    top = kernel_top(precision, shift, offset, floor)
    grid = np.linspace(top, top + 10.0, 200)
    kernel = np.exp(offset[:, None] + grid[None, :])
    values = 0.5 * np.square(shift)[:, None] * kernel / (1.0 + kernel * precision[:, None]) - 0.5 * np.log1p(kernel * precision[:, None])
    assert np.all(np.diff(values, axis=1) <= 1e-12)


def test_the_derived_spacing_certifies_the_lattice_sum():
    class_index, offset, _design, _groups, cavity = _data(80, 25)
    floor = kernel_floor(cavity.precision, cavity.shift, offset, 1e-3)
    top = kernel_top(cavity.precision, cavity.shift, offset, floor)
    centre, width = 0.5 * (floor + top), (top - floor) / 8.0
    extent = (centre - 8.0 * width, centre + 8.0 * width)

    def lattice_prior(spacing):
        nodes = np.arange(extent[0], extent[1], spacing)
        return scale_mixture_prior(
            class_index=np.zeros(class_index.shape[0], dtype=np.int64), log_variance_offset=offset,
            annotation_design=np.zeros((class_index.shape[0], 0)), annotation_groups=(), nodes=nodes, floor=floor, top=top,
        )

    def log_normal(prior):
        # The same continuous log-normal on every lattice.
        eta = -0.5 * np.square((prior.log_variance_grid - centre) / width)
        coefficients = np.zeros(prior.coefficient_size)
        coefficients[: prior.pooled_size] = prior.coefficient_map[: prior.grid_size, : prior.pooled_size].T @ (eta - eta.mean())
        return MixtureHyperparameters(coefficients=coefficients, log_smoothing=np.zeros(len(prior.smoothing_blocks)))

    tolerance = 1e-6
    reference_prior = lattice_prior((extent[1] - extent[0]) / 1000.0)
    spacing = spacing_bound(quadrature_majorant_ratio(reference_prior, log_normal(reference_prior), cavity, _WORKING_BYTES), tolerance)
    coarse_prior = lattice_prior(spacing)
    coarse = tilted_moments(coarse_prior, log_normal(coarse_prior), cavity, _WORKING_BYTES).log_normalizer
    reference = tilted_moments(reference_prior, log_normal(reference_prior), cavity, _WORKING_BYTES).log_normalizer
    assert np.sum(np.abs(coarse - reference)) <= tolerance


def test_the_derived_lattice_is_uniform_and_covers_the_kernel_range():
    _class_index, offset, _design, _groups, cavity = _data(60, 27)
    nodes, floor, top = derived_lattice(cavity.precision, cavity.shift, offset, 1e-3)
    assert floor == kernel_floor(cavity.precision, cavity.shift, offset, 1e-3)
    assert top == kernel_top(cavity.precision, cavity.shift, offset, floor)
    np.testing.assert_allclose(np.diff(nodes), spacing_bound(60.0, 1e-3), rtol=1e-12)
    assert nodes[0] <= floor - (top - floor) + 1e-12 and nodes[-1] >= top + (top - floor) - 1e-12


def test_relattice_extends_the_log_tails_linearly_and_keeps_the_density_inside():
    class_index, offset, design, groups, _cavity = _data(30, 29)
    nodes = np.arange(-8.0, 3.0, 0.05)
    prior = scale_mixture_prior(
        class_index=class_index, log_variance_offset=offset, annotation_design=design, annotation_groups=groups,
        nodes=nodes, floor=nodes[0], top=nodes[-1],
    )
    bump = -np.square((nodes + 2.5) / 0.8) + 0.3 * np.sin(2.0 * nodes)
    coefficients = np.zeros(prior.coefficient_size)
    coefficients[: prior.pooled_size] = prior.coefficient_map[: prior.grid_size, : prior.pooled_size].T @ (bump - bump.mean())
    hyperparameters = MixtureHyperparameters(coefficients=coefficients, log_smoothing=np.zeros(len(prior.smoothing_blocks)))
    wider = np.arange(-12.0, 6.0, 0.05)
    moved, transferred = relattice(prior, hyperparameters, wider, -9.0, 4.0)
    old_log_density = class_log_density(prior, hyperparameters.coefficients)
    new_log_density = class_log_density(moved, transferred.coefficients)
    inside = (wider >= nodes[0] - 1e-9) & (wider <= nodes[-1] + 1e-9)
    # Inside the old lattice the density per unit t is unchanged, up to the constant normalization shifts.
    difference = new_log_density[:, inside] - old_log_density[:, np.searchsorted(nodes, wider[inside] - 1e-9)]
    np.testing.assert_allclose(difference - difference[:, :1], 0.0, atol=1e-6)
    # Outside it the log density continues as the polynomial of degree ROUGHNESS_ORDER - 1 the end derivatives set
    # (the natural spline's own continuation, of no roughness): its differences of the roughness order vanish there.
    outside = np.flatnonzero(wider > nodes[-1] + 0.2)
    np.testing.assert_allclose(np.diff(new_log_density[:, outside], n=engine.ROUGHNESS_ORDER, axis=1), 0.0, atol=1e-8)
    assert moved.kernel_floor == -9.0 and moved.kernel_top == 4.0


def _block_residuals(prior, coefficients):
    return np.array([float(np.linalg.norm(block.factor @ coefficients[block.coordinates])) for block in prior.smoothing_blocks])


@pytest.mark.parametrize("curvature", [-0.3, 0.3])
def test_relattice_keeps_every_block_at_lambda_infinity_in_its_null_space(curvature):
    # A D3 block at lambda = infinity holds a quadratic log g; the transfer to a halved, a wider and a shifted lattice
    # must keep it one exactly, keep zero deviations zero (the least-squares transfer split a shared quadratic two
    # thirds pooled and one third per deviation, and the deviation functionals then penalized it), and leave the
    # density inside the old lattice unchanged.
    class_index, offset, design, groups, _cavity = _data(30, 29)
    nodes = np.arange(-8.0, 3.0, 0.25)
    prior = scale_mixture_prior(
        class_index=class_index, log_variance_offset=offset, annotation_design=design, annotation_groups=groups,
        nodes=nodes, floor=nodes[0], top=nodes[-1],
    )
    quadratic = curvature * np.square(nodes + 2.5) + 0.4 * nodes
    coefficients = np.zeros(prior.coefficient_size)
    coefficients[: prior.pooled_size] = prior.coefficient_map[: prior.grid_size, : prior.pooled_size].T @ (quadratic - quadratic.mean())
    hyperparameters = MixtureHyperparameters(coefficients=coefficients, log_smoothing=np.full(len(prior.smoothing_blocks), np.inf))
    scale = float(np.linalg.norm(coefficients))
    for moved, transferred in (
        halved_lattice(prior, hyperparameters),
        relattice(prior, hyperparameters, np.arange(-12.0, 7.0, 0.25), -12.0, 6.75),
        relattice(prior, hyperparameters, nodes + 2.0, nodes[0] + 2.0, nodes[-1] + 2.0),
    ):
        residuals = _block_residuals(moved, transferred.coefficients)
        assert np.all(residuals <= 64 * engine._EPSILON * moved.coefficient_size * scale * max(float(np.linalg.norm(block.factor)) for block in moved.smoothing_blocks))
        assert np.all(transferred.coefficients[moved.pooled_size : moved.pooled_size * (1 + moved.class_count)] == 0.0)
        new_nodes = moved.log_variance_grid
        inside = (new_nodes >= nodes[0] - 1e-9) & (new_nodes <= nodes[-1] + 1e-9)
        expected = curvature * np.square(new_nodes[inside] + 2.5) + 0.4 * new_nodes[inside]
        got = engine._density_and_scale(moved, transferred.coefficients)[0][0, inside]
        np.testing.assert_allclose(got - got.mean(), expected - expected.mean(), atol=1e-8)


def _log_normal_problem(seed: int):
    """One class, no annotations, effects whose variances are log-normal: the null model is the truth."""
    generator = np.random.default_rng(seed)
    variant_count = 400
    log_variance = generator.normal(np.log(0.02), 1.2, variant_count)
    precision = generator.uniform(100.0, 800.0, variant_count)
    effect = generator.standard_normal(variant_count) * np.exp(0.5 * log_variance)
    shift = precision * (effect + generator.standard_normal(variant_count) / np.sqrt(precision))
    offset = np.zeros(variant_count)
    nodes, floor, top = derived_lattice(precision, shift, offset, 1e-3)
    prior = scale_mixture_prior(
        class_index=np.zeros(variant_count, dtype=np.int64), log_variance_offset=offset, annotation_design=np.zeros((variant_count, 0)),
        annotation_groups=(), nodes=nodes, floor=floor, top=top,
    )
    return prior, Cavity(precision=precision, shift=shift)


def test_the_global_log_normal_start_reaches_the_null_models_maximum():
    prior, cavity = _log_normal_problem(31)
    view, allowed = _restricted_prior(prior, frozenset({0}))
    start = _log_normal_start(prior, initial_hyperparameters(prior).coefficients, cavity, _WORKING_BYTES)
    coefficients, objective = _maximize_coefficients(view, np.zeros(0), allowed.T @ start, cavity, _WORKING_BYTES, 0.0)
    nodes = prior.log_variance_grid
    basis = prior.coefficient_map[: prior.grid_size, : prior.pooled_size]
    brute = -np.inf
    for centre in np.linspace(prior.kernel_floor, prior.kernel_top, 60):
        for width in np.exp(np.linspace(np.log(nodes[1] - nodes[0]), np.log(prior.kernel_top - prior.kernel_floor), 40)):
            log_density = -0.5 * np.square((nodes - centre) / width)
            candidate = np.zeros(prior.coefficient_size)
            candidate[: prior.pooled_size] = basis.T @ (log_density - log_density.mean())
            brute = max(brute, _data_value(prior, candidate, cavity, _WORKING_BYTES))
    assert objective.value >= brute - 1e-6


def test_the_hyper_steps_evidence_is_at_least_the_exact_infinity_edges():
    prior, cavity = _log_normal_problem(33)
    step = hyper_step(prior, initial_hyperparameters(prior), cavity, INDEPENDENT_EFFECTS, _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    view, allowed = _restricted_prior(prior, frozenset({0}))
    start = _log_normal_start(prior, initial_hyperparameters(prior).coefficients, cavity, _WORKING_BYTES)
    correction = INDEPENDENT_EFFECTS
    edge = _corrected(
        view, np.zeros(0), _evidence(view, np.zeros(0), allowed.T @ start, cavity, correction, _WORKING_BYTES, _EVIDENCE_TOLERANCE), cavity, correction,
        _WORKING_BYTES, _EVIDENCE_TOLERANCE,
    )
    assert edge is not None
    # Both are certified V, each to the tolerance.
    assert step.evidence >= edge.value - _EVIDENCE_TOLERANCE


def _null_annotation_problem(seed: int):
    """verify-engine's edge problem: one class, ten nodes, and one annotation column with no effect on the variances."""
    generator = np.random.default_rng(seed)
    variant_count = 80
    nodes = np.linspace(np.log(1e-5), np.log(1.0), 10)
    position = generator.uniform(-1.0, 1.0, variant_count)
    offset = np.log(generator.uniform(0.2, 1.0, variant_count))
    prior = scale_mixture_prior(
        class_index=np.zeros(variant_count, np.int64), log_variance_offset=offset, annotation_design=position[:, None],
        annotation_groups=(AnnotationGroup(columns=np.array([0]), penalty=np.eye(1)),), nodes=nodes, floor=nodes[0] - 1.0, top=nodes[-1],
    )
    precision = generator.uniform(50.0, 400.0, variant_count)
    effect = np.where(generator.random(variant_count) < 0.4, generator.normal(0.0, 0.25, variant_count), 0.0)
    shift = precision * (effect + generator.standard_normal(variant_count) / np.sqrt(precision))
    return prior, Cavity(precision=precision, shift=shift)


@pytest.mark.parametrize("seed", (101, 202, 303))
def test_a_null_annotation_is_penalized_at_an_interior_optimum_or_the_infinity_edge(seed):
    # The lambda = 0 "edge" profiled a block under a flat prior, whose value bounds every proper-prior V; the search
    # took it and left a null annotation unpenalized (verify-engine). With the edge gone, every weight is either at
    # the infinity edge or inside its resolvable range, never below it.
    prior, cavity = _null_annotation_problem(seed)
    step = hyper_step(prior, initial_hyperparameters(prior), cavity, INDEPENDENT_EFFECTS, _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    weights = step.hyperparameters.log_smoothing
    assert not np.any(weights == -np.inf), weights
    bounds = _smoothing_bounds(prior, _data_objective(prior, initial_hyperparameters(prior).coefficients, cavity, _WORKING_BYTES))
    for weight, (lower, _upper) in zip(weights, bounds):
        assert weight == np.inf or weight >= lower, (weights, bounds)


def test_tilted_cumulants_are_the_shift_derivatives_of_log_z():
    # kappa_n = d^n log Z_j / d h_j^n at the cavity: central differences of the tilted mean (kappa_1) in the shift.
    prior, cavity = _problem(variant_count=12, seed=1, node_count=9)
    hyperparameters = _hyperparameters(prior, 2)
    third, fourth = tilted_cumulants(prior, hyperparameters, cavity, _WORKING_BYTES)

    def variance_at(shift):
        return tilted_moments(prior, hyperparameters, Cavity(precision=cavity.precision, shift=shift), _WORKING_BYTES).variance

    step = 1e-3 * (1.0 + np.abs(cavity.shift))
    numeric_third = (variance_at(cavity.shift + step) - variance_at(cavity.shift - step)) / (2.0 * step)
    numeric_fourth = (variance_at(cavity.shift + step) - 2.0 * variance_at(cavity.shift) + variance_at(cavity.shift - step)) / np.square(step)
    np.testing.assert_allclose(third, numeric_third, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(fourth, numeric_fourth, rtol=1e-3, atol=1e-8)


def test_directional_third_and_fourth_derivatives_match_finite_differences():
    prior, cavity = _problem(variant_count=25, seed=35, node_count=10)
    coefficients = _hyperparameters(prior, 36).coefficients
    directions = np.random.default_rng(37).standard_normal((prior.coefficient_size, 3)) * 0.2
    third, fourth = _directional_derivatives(prior, coefficients, cavity, directions, _WORKING_BYTES)
    step = 2e-2
    for column in range(3):
        values = [_data_value(prior, coefficients + multiple * step * directions[:, column], cavity, _WORKING_BYTES) for multiple in (-2, -1, 0, 1, 2)]
        numerical_third = (values[4] - 2.0 * values[3] + 2.0 * values[1] - values[0]) / (2.0 * step**3)
        numerical_fourth = (values[4] - 4.0 * values[3] + 6.0 * values[2] - 4.0 * values[1] + values[0]) / step**4
        np.testing.assert_allclose(third[column], numerical_third, rtol=2e-3, atol=1e-3)
        np.testing.assert_allclose(fourth[column], numerical_fourth, rtol=2e-2, atol=1e-2)


def test_directional_derivatives_of_every_direction_at_once_are_the_direct_cumulant_sums():
    # With an annotation design every direction comes from w-moments about each variant's modal node, over row chunks
    # and pieces; the direct sums form theta' to theta'''' per direction and take their joint cumulants under w.
    prior, cavity = _problem(variant_count=400, seed=41, node_count=24)
    coefficients = _hyperparameters(prior, 42).coefficients
    directions = np.random.default_rng(43).standard_normal((prior.coefficient_size, 9))
    working_bytes = engine._ROW_INTERMEDIATES * 8 * prior.grid_size * 37
    third, fourth = _directional_derivatives(prior, coefficients, cavity, directions, working_bytes)
    directions_z = prior.coefficient_map @ directions
    grid_size = prior.grid_size
    scale_steps = directions_z[prior.density_size : prior.density_size + prior.scale_size]
    density = np.exp(class_log_density(prior, coefficients))
    expected_third = np.zeros(directions.shape[1])
    expected_fourth = np.zeros(directions.shape[1])
    for class_position, rows, terms in engine._class_terms(prior, coefficients, cavity, working_bytes):
        weights = terms.responsibility
        for column in range(directions.shape[1]):
            density_step = directions_z[class_position * grid_size : (class_position + 1) * grid_size, column]
            scale_step = (prior.scale_design[rows] @ scale_steps[:, column])[:, None]
            first = density_step[None, :] + terms.first * scale_step
            second, third_term, fourth_term = terms.second * scale_step**2, terms.third * scale_step**3, terms.fourth * scale_step**4
            centred, centred_second, centred_third = (values - np.sum(weights * values, axis=1, keepdims=True) for values in (first, second, third_term))
            variance = np.sum(weights * centred**2, axis=1)
            expected_third[column] += np.sum(
                np.sum(weights * centred**3, axis=1) + 3.0 * np.sum(weights * centred * centred_second, axis=1) + np.sum(weights * third_term, axis=1)
            )
            expected_fourth[column] += np.sum(
                np.sum(weights * centred**4, axis=1) - 3.0 * variance**2 + 6.0 * np.sum(weights * centred**2 * centred_second, axis=1)
                + 3.0 * np.sum(weights * centred_second**2, axis=1) + 4.0 * np.sum(weights * centred * centred_third, axis=1)
                + np.sum(weights * fourth_term, axis=1)
            )
    for class_position, class_rows in enumerate(prior.class_rows):
        step = directions_z[class_position * grid_size : (class_position + 1) * grid_size]
        centred = step - density[class_position] @ step
        variance = density[class_position] @ centred**2
        expected_third -= class_rows.shape[0] * (density[class_position] @ centred**3)
        expected_fourth -= class_rows.shape[0] * (density[class_position] @ centred**4 - 3.0 * variance**2)
    np.testing.assert_allclose(third, expected_third, rtol=1e-9, atol=1e-9 * np.max(np.abs(expected_third)))
    np.testing.assert_allclose(fourth, expected_fourth, rtol=1e-9, atol=1e-9 * np.max(np.abs(expected_fourth)))


def test_directional_derivatives_in_row_blocks_and_after_refused_blocks_are_the_whole_chunks(monkeypatch):
    # With an annotation design the Laplace pass forms each chunk's moments a block of rows at a time and halves a block
    # the ledger refuses (the whole chunk at once refused 1.0 GB with 40.8 GB held on device0 [real, aou-m1d-h1]): any
    # blocking sums the same per-row terms, so the derivatives agree with the whole chunk's to rounding.
    prior, cavity = _problem(variant_count=400, seed=41, node_count=24)
    assert prior.scale_size
    coefficients = _hyperparameters(prior, 42).coefficients
    directions = np.random.default_rng(43).standard_normal((prior.coefficient_size, 9))
    working_bytes = engine._ROW_INTERMEDIATES * 8 * prior.grid_size * prior.variant_count
    monkeypatch.setattr(engine, "_directional_rows", lambda grid_size, direction_count, working_bytes: prior.variant_count)
    whole = _directional_derivatives(prior, coefficients, cavity, directions, working_bytes)
    monkeypatch.setattr(engine, "_directional_rows", lambda grid_size, direction_count, working_bytes: 7)
    sizes: list[int] = []
    moments = engine._scale_direction_moments

    def counted(xp, weights, derivatives, steps, scale_steps):
        sizes.append(int(weights.shape[0]))
        return moments(xp, weights, derivatives, steps, scale_steps)

    monkeypatch.setattr(engine, "_scale_direction_moments", counted)
    blocked = _directional_derivatives(prior, coefficients, cavity, directions, working_bytes)
    assert max(sizes) == 7 and sum(sizes) == prior.variant_count
    for got, expected in zip(blocked, whole):
        np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-11 * np.max(np.abs(expected)))

    # A refused block is halved, down to one row, and the halved size holds for the class's later blocks.
    monkeypatch.setattr(engine, "_directional_rows", lambda grid_size, direction_count, working_bytes: prior.variant_count)
    sizes.clear()

    def refusing(xp, weights, derivatives, steps, scale_steps):
        if weights.shape[0] > 5:
            raise MemoryError("a device allocation needs more than the budget holds")
        return counted(xp, weights, derivatives, steps, scale_steps)

    monkeypatch.setattr(engine, "_scale_direction_moments", refusing)
    halved = _directional_derivatives(prior, coefficients, cavity, directions, working_bytes)
    assert max(sizes) <= 5 and sum(sizes) == prior.variant_count
    for got, expected in zip(halved, whole):
        np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-11 * np.max(np.abs(expected)))

    # A refused single row is a refusal of the pass.
    monkeypatch.setattr(engine, "_scale_direction_moments", lambda *arguments: (_ for _ in ()).throw(MemoryError("refused")))
    with pytest.raises(MemoryError):
        _directional_derivatives(prior, coefficients, cavity, directions, working_bytes)


def test_directional_blocks_are_sized_from_the_room_the_device_has_left(monkeypatch):
    # Per row: the block's kernel rows, their responsibilities and its moments' rows x K arrays, each with a per-row
    # value, and its scale steps; at least one row.
    grid_size, direction_count = 24, 9
    arrays = engine._HELD_ARRAYS_PER_ROW_SET + 1 + engine._DIRECTIONAL_ROW_ARRAYS
    per_row = 8 * (arrays * (grid_size + 1) + direction_count)
    monkeypatch.setattr(engine, "_device_room", lambda working_bytes: min(working_bytes, 7 * per_row + per_row - 1))
    assert engine._directional_rows(grid_size, direction_count, 1 << 40) == 7
    assert engine._directional_rows(grid_size, direction_count, 3 * per_row) == 3
    assert engine._directional_rows(grid_size, direction_count, 1) == 1


def test_quadrature_corrections_are_the_exact_integrals_along_the_standardized_directions():
    prior, cavity = _problem(variant_count=60, seed=39, node_count=12)
    hyperparameters = _hyperparameters(prior, 40, log_smoothing=2.0)
    evidence = _evidence(prior, hyperparameters.log_smoothing, hyperparameters.coefficients, cavity, INDEPENDENT_EFFECTS, _WORKING_BYTES, 0.0)
    assert evidence is not None
    corrections, terms, directions = _laplace_corrections(prior, hyperparameters.log_smoothing, evidence, cavity, INDEPENDENT_EFFECTS, _WORKING_BYTES, 0.0)
    value = _data_value(prior, evidence.coefficients, cavity, _WORKING_BYTES) - _penalty_value(prior, hyperparameters.log_smoothing, evidence.coefficients)[0]
    steps = np.linspace(-12.0, 12.0, 4801)
    for index in np.argsort(-np.abs(terms))[:3]:
        line = np.array([
            _data_value(prior, evidence.coefficients + step * directions[:, index], cavity, _WORKING_BYTES)
            - _penalty_value(prior, hyperparameters.log_smoothing, evidence.coefficients + step * directions[:, index])[0]
            for step in steps
        ])
        reference = np.log(np.trapz(np.exp(line - value), steps)) - 0.5 * np.log(2.0 * np.pi)
        np.testing.assert_allclose(corrections[index], reference, atol=1e-6)
    # Where the Tierney-Kadane term is tiny, the exact correction is of its size.
    tiny = np.abs(terms) < 1e-5
    assert np.all(np.abs(corrections[tiny]) <= 2.0 * np.abs(terms[tiny]) + 1e-7)


def test_the_kronrod_rule_is_quadpacks():
    # The embedded Gauss rule is the 7-point Gauss-Legendre rule, and the 15-point Kronrod rule integrates every
    # polynomial of degree 22 exactly (3 n + 1 for n = 7).
    nodes = np.concatenate([-engine._KRONROD_NODES[:-1], engine._KRONROD_NODES[::-1]])
    kronrod = np.concatenate([engine._KRONROD_WEIGHTS[:-1], engine._KRONROD_WEIGHTS[::-1]])
    gauss = np.concatenate([engine._GAUSS_WEIGHTS[:-1], engine._GAUSS_WEIGHTS[::-1]])
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(7)
    np.testing.assert_allclose(np.sort(nodes[gauss > 0.0]), np.sort(gauss_nodes), atol=1e-15)
    np.testing.assert_allclose(gauss[gauss > 0.0][np.argsort(nodes[gauss > 0.0])], gauss_weights[np.argsort(gauss_nodes)], atol=1e-15)
    for degree in range(23):
        exact = (1.0 - (-1.0) ** (degree + 1)) / (degree + 1)
        assert abs(float(kronrod @ nodes**degree) - exact) <= 1e-14


def test_the_line_values_are_the_penalized_objective_at_each_step():
    # One batched pass along x + t b must give F - P at every step, also when the variants are cut into many chunks.
    prior, cavity = _problem(variant_count=60, seed=51, node_count=12)
    hyperparameters = _hyperparameters(prior, 52, log_smoothing=1.0)
    moving = 0.3 * np.random.default_rng(53).standard_normal(prior.coefficient_size)
    # A direction with no scale part keeps every kernel row: its steps are one product over the nodes.
    still = moving.copy()
    still[prior.coefficient_size - prior.scale_size :] = 0.0
    steps = np.array([-3.0, -0.4, 0.0, 0.9, 2.5])
    for working_bytes, direction in ((_WORKING_BYTES, moving), (1 << 12, moving), (_WORKING_BYTES, still), (1 << 12, still)):
        values = _line(prior, hyperparameters.log_smoothing, hyperparameters.coefficients, direction, cavity, working_bytes)(steps)
        expected = np.array([
            _data_value(prior, hyperparameters.coefficients + step * direction, cavity, _WORKING_BYTES)
            - _penalty_value(prior, hyperparameters.log_smoothing, hyperparameters.coefficients + step * direction)[0]
            for step in steps
        ])
        np.testing.assert_allclose(values, expected, rtol=1e-12, atol=1e-12 * float(np.max(np.abs(expected))))


def test_the_line_integral_matches_a_tight_quadrature_to_its_share():
    prior, cavity = _problem(variant_count=60, seed=39, node_count=12)
    hyperparameters = _hyperparameters(prior, 40, log_smoothing=2.0)
    posterior = INDEPENDENT_EFFECTS
    evidence = _evidence(prior, hyperparameters.log_smoothing, hyperparameters.coefficients, cavity, posterior, _WORKING_BYTES, 0.0)
    assert evidence is not None
    _corrections, terms, directions = _laplace_corrections(prior, hyperparameters.log_smoothing, evidence, cavity, posterior, _WORKING_BYTES, 0.0)
    value = evidence.penalized_value
    share = _EVIDENCE_TOLERANCE / 16.0
    for index in np.argsort(-np.abs(terms))[:3]:
        direction = directions[:, index]
        found = _line_log_integral(prior, hyperparameters.log_smoothing, evidence.coefficients, direction, value, cavity, _WORKING_BYTES, share)

        def integrand(step: float) -> float:
            point = evidence.coefficients + step * direction
            return float(np.exp(
                _data_value(prior, point, cavity, _WORKING_BYTES) - _penalty_value(prior, hyperparameters.log_smoothing, point)[0] - value
            ))

        integral, _error = quad(integrand, -np.inf, np.inf, epsabs=0.0, epsrel=1e-12)
        assert abs(found - (np.log(integral) - 0.5 * np.log(2.0 * np.pi))) <= share


def test_a_line_integral_whose_settled_errors_outgrow_a_falling_total_raises_the_handled_error(monkeypatch):
    # A spike on a first-round node inflates the first total; the rough left half-line settles against it, and the
    # second round misses the spike, so the total falls below what the settled error allows while every open interval
    # is within its share. It raised numpy's empty-concatenate ValueError, which no caller handles.
    from types import SimpleNamespace

    nodes = np.concatenate([-engine._KRONROD_NODES[:-1], engine._KRONROD_NODES[::-1]])
    spike = (1.0 - (0.5 + 0.5 * nodes[3])) / (0.5 + 0.5 * nodes[3])

    def log_integrand(steps: np.ndarray) -> np.ndarray:
        base = np.where(steps < 0.0, np.exp(-0.5 * (steps / 0.02) ** 2), np.exp(-0.5 * steps**2))
        with np.errstate(divide="ignore"):
            return np.log(base + 1e3 * np.exp(-0.5 * ((steps - spike) / 1e-9) ** 2))

    monkeypatch.setattr(engine, "_lines", lambda *arguments: lambda steps: {column: log_integrand(step) for column, step in steps.items()})
    with pytest.raises(FloatingPointError, match="settled against an earlier total"):
        engine._line_log_integrals(SimpleNamespace(density_size=1), None, None, np.zeros((1, 1)), 0.0, None, _WORKING_BYTES, 1e-3)


def test_a_non_finite_curvature_is_not_resolvably_positive_definite():
    # eigvalsh and cholesky pass a nan through without raising; solve_triangular's ValueError reached no handler.
    matrix = np.eye(3)
    matrix[1, 2] = matrix[2, 1] = np.nan
    with pytest.raises(np.linalg.LinAlgError, match="not finite"):
        engine._profiled_factor(matrix, np.zeros((3, 0)), np.eye(3))


def test_starts_that_find_one_basin_are_corrected_once(monkeypatch):
    prior, cavity = _problem(variant_count=60, seed=39, node_count=12)
    hyperparameters = _hyperparameters(prior, 40, log_smoothing=2.0)
    posterior = INDEPENDENT_EFFECTS
    nearby = hyperparameters.coefficients + 1e-3 * np.random.default_rng(41).standard_normal(prior.coefficient_size)
    calls = []
    corrected = engine._corrected
    monkeypatch.setattr(engine, "_corrected", lambda *arguments: calls.append(None) or corrected(*arguments))
    best = _best_certified(
        prior, hyperparameters.log_smoothing, [hyperparameters.coefficients, nearby], cavity, posterior, _WORKING_BYTES, _EVIDENCE_TOLERANCE
    )
    assert best is not None and len(calls) == 1


def test_a_trait_with_no_effect_above_the_noise_still_gets_a_lattice_and_a_fit_start():
    # prior-terms' case: every single-variant likelihood is flat to the tolerance, so the kernel range is empty.
    generator = np.random.default_rng(71)
    count = 400
    precision = np.full(count, 4.0)
    shift = 4.0 * 0.1 * generator.standard_normal(count)
    frequency = generator.uniform(0.01, 0.5, count)
    offset = np.log(generator.uniform(0.3, 1.0, count)) + np.log(2.0 * frequency * (1.0 - frequency))
    nodes, floor, top = derived_lattice(precision, shift, offset, _LATTICE_TOLERANCE)
    assert nodes.shape[0] > engine.ROUGHNESS_ORDER
    prior = scale_mixture_prior(
        class_index=np.zeros(count, dtype=np.int64), log_variance_offset=offset, annotation_design=np.zeros((count, 0)),
        annotation_groups=(), nodes=nodes, floor=floor, top=top,
    )
    moments = tilted_moments(prior, initial_hyperparameters(prior), Cavity(precision=precision, shift=shift), _WORKING_BYTES)
    assert np.all(np.isfinite(moments.mean)) and np.all(moments.variance > 0.0)


def test_the_variance_matched_start_has_the_asked_prior_variances():
    prior, _cavity = _problem(variant_count=60, seed=39, node_count=12)
    nodes = prior.log_variance_grid
    offsets = np.exp(log_scale(prior, initial_hyperparameters(prior).coefficients))
    for mean_variance in (float(np.exp(nodes[2])), float(np.exp(0.5 * (nodes[0] + nodes[-1]))), float(np.exp(nodes[-3]))):
        start = initial_hyperparameters(prior, mean_variance)
        np.testing.assert_allclose(prior_second_moment(prior, start), mean_variance * offsets, rtol=1e-10)
    # Beyond the lattice's reach the centre stops at the nearer end.
    below = initial_hyperparameters(prior, float(np.exp(nodes[0] - 1.0)))
    assert np.all(prior_second_moment(prior, below) < np.exp(nodes[2]) * offsets)


def _moments(genotypes, target, weights):
    gram = genotypes.T @ genotypes
    return dict(
        target_square=float(target @ target), residual_dimension=float(genotypes.shape[0]), score_square=float(np.sum(np.square(genotypes.T @ target))),
        gram_trace=float(np.trace(gram)), weighted_diagonal=float(weights @ np.diag(gram)),
        weighted_square=float(weights @ np.sum(np.square(gram), axis=0)), gram_square=float(np.sum(np.square(gram))),
    )


def test_the_moment_start_splits_the_phenotypic_variance_and_tracks_the_heritability():
    generator = np.random.default_rng(61)
    samples, variants = 2000, 300
    genotypes = generator.standard_normal((samples, variants))
    weights = generator.uniform(0.5, 2.0, variants)
    for heritability in (0.2, 0.6):
        effects = generator.standard_normal(variants) * np.sqrt(weights)
        signal = genotypes @ effects
        noise = generator.standard_normal(samples) * np.sqrt(np.var(signal) * (1.0 - heritability) / heritability)
        start = moment_start(**_moments(genotypes, signal + noise, weights))
        total = float((signal + noise) @ (signal + noise)) / samples
        # The split is exact: the genetic variance never exceeds the phenotypic.
        np.testing.assert_allclose(start.genetic_variance + start.noise, total, rtol=1e-12)
        assert abs(start.heritability - heritability) <= 0.15
    # Under no signal the start stays within a few resolutions of zero, and never at it.
    null = moment_start(**_moments(genotypes, generator.standard_normal(samples), weights))
    assert null.resolution <= null.heritability <= 4.0 * null.resolution


def test_the_correction_slopes_are_the_corrected_evidences_own():
    # The corrections' slopes must be those of V's corrections as V computes them at every rho (x re-maximized, the
    # standardized directions re-derived), not of held directions: on this case holding them gave +0.25 where V's own
    # slope was +0.04 [sim-only]. The central difference of V's correction part, with x re-solved, is the reference.
    prior, cavity = _annotated_problem(101)
    weights, evidence, interior = _ascended(prior, cavity)
    base = _corrected(prior, weights, _evidence(prior, weights, evidence.coefficients, cavity, INDEPENDENT_EFFECTS, _WORKING_BYTES, 0.0),
                      cavity, INDEPENDENT_EFFECTS, _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    assert base is not None and base.replaced_directions is not None and base.replaced_directions.shape[1] > 0
    slopes, errors, _second = engine._correction_slopes(prior, weights, base, interior, cavity, _WORKING_BYTES, engine._coarse_targets(base, interior, _EVIDENCE_TOLERANCE))
    step = 0.02
    for position in np.flatnonzero(interior):
        unit = np.zeros(weights.shape[0])
        unit[position] = step
        sides = []
        for sign in (1.0, -1.0):
            moved = _corrected(
                prior, weights + sign * unit,
                _evidence(prior, weights + sign * unit, base.coefficients + sign * step * base.responses[:, position], cavity, INDEPENDENT_EFFECTS, _WORKING_BYTES, 0.0),
                cavity, INDEPENDENT_EFFECTS, _WORKING_BYTES, _EVIDENCE_TOLERANCE,
            )
            assert moved is not None and moved.replaced_directions.shape[1] == base.replaced_directions.shape[1]
            sides.append(moved.value - moved.laplace_value)
        reference = (sides[0] - sides[1]) / (2.0 * step)
        # The reference's own error: each side's quadrature (its share) over 2 h, and its truncation.
        assert abs(slopes[position] - reference) <= errors[position] + base.replaced_share * base.replaced_directions.shape[1] / step + 1e-3


def _annotated_problem(seed: int):
    """One class, ten nodes and one annotation column that scales the effects (verify-engine's finding-3 case)."""
    generator = np.random.default_rng(seed)
    variant_count = 80
    nodes = np.linspace(np.log(1e-5), np.log(1.0), 10)
    position = generator.uniform(-1.0, 1.0, variant_count)
    offset = np.log(generator.uniform(0.2, 1.0, variant_count))
    prior = scale_mixture_prior(
        class_index=np.zeros(variant_count, np.int64), log_variance_offset=offset, annotation_design=position[:, None],
        annotation_groups=(AnnotationGroup(columns=np.array([0]), penalty=np.eye(1)),), nodes=nodes, floor=nodes[0] - 1.0, top=nodes[-1],
    )
    precision = generator.uniform(50.0, 400.0, variant_count)
    effect = np.where(generator.random(variant_count) < 0.4, generator.normal(0.0, 0.25, variant_count), 0.0) * np.exp(0.5 * position)
    shift = precision * (effect + generator.standard_normal(variant_count) / np.sqrt(precision))
    return prior, Cavity(precision=precision, shift=shift)


def _ascended(prior, cavity):
    flat = initial_hyperparameters(prior).coefficients
    start = _corrected(prior, np.zeros(2), _evidence(prior, np.zeros(2), flat, cavity, INDEPENDENT_EFFECTS, _WORKING_BYTES, _EVIDENCE_TOLERANCE),
                       cavity, INDEPENDENT_EFFECTS, _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    bounds = _smoothing_bounds(prior, _data_objective(prior, start.coefficients, cavity, _WORKING_BYTES))
    lower, upper = np.array([bound[0] for bound in bounds]), np.array([bound[1] for bound in bounds])
    weights, evidence = _ascend_evidence(prior, np.zeros(2), start, cavity, INDEPENDENT_EFFECTS, _WORKING_BYTES, lower, upper, _EVIDENCE_TOLERANCE, flat)
    return weights, evidence, (weights > lower) & (weights < upper)


# Seeds 101 and 303 certify since the lambda = infinity start with releases at the range's centre (lead, 2026-09-21).
@pytest.mark.parametrize("seed", (101, pytest.param(202, marks=pytest.mark.xfail(strict=True, reason=_FLAT_REASON)), 303))
def test_the_weights_remaining_gain_covers_every_nearby_weight(seed):
    # verify-engine finding 3: the old check's 1/2 (|c| + E)^2 / s used an upper bound on |V''| and so understated the
    # gain (0.0245 claimed where V rose 0.0645 within one unit of rho, seed 303). At the point hyper_step certifies,
    # the best certified V over a neighbourhood of its interior weights must stay within the claimed remaining gain,
    # up to the two sides' certified tolerances. (The claim is a Newton decrement on the local model, so it is tested
    # where it certifies, not along the way.)
    prior, cavity = _annotated_problem(seed)
    step = hyper_step(prior, initial_hyperparameters(prior), cavity, INDEPENDENT_EFFECTS, _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    assert step.stationarity_gain <= _EVIDENCE_TOLERANCE
    log_smoothing = step.hyperparameters.log_smoothing
    infinite = frozenset(int(position) for position in np.flatnonzero(log_smoothing == np.inf))
    view, allowed = _restricted_prior(prior, infinite)
    weights = log_smoothing[np.isfinite(log_smoothing)]
    coefficients = allowed.T @ step.hyperparameters.coefficients
    base = _corrected(view, weights, _evidence(view, weights, coefficients, cavity, INDEPENDENT_EFFECTS, _WORKING_BYTES, _EVIDENCE_TOLERANCE),
                      cavity, INDEPENDENT_EFFECTS, _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    assert base is not None
    bounds = _smoothing_bounds(prior, _data_objective(prior, step.hyperparameters.coefficients, cavity, _WORKING_BYTES))
    lower = np.array([bound[0] for bound in bounds])[np.isfinite(log_smoothing)]
    upper = np.array([bound[1] for bound in bounds])[np.isfinite(log_smoothing)]
    interior = np.flatnonzero((weights > lower) & (weights < upper))
    if not interior.shape[0]:
        pytest.skip("every weight is at an edge or a bound of its range")
    gains = []
    for position in interior:
        for distance in (-1.0, -0.3, -0.1, 0.1, 0.3, 1.0):
            moved_weights = weights.copy()
            moved_weights[position] += distance
            moved = _corrected(
                view, moved_weights, _evidence(view, moved_weights, base.coefficients, cavity, INDEPENDENT_EFFECTS, _WORKING_BYTES, _EVIDENCE_TOLERANCE),
                cavity, INDEPENDENT_EFFECTS, _WORKING_BYTES, _EVIDENCE_TOLERANCE,
            )
            if moved is not None:
                gains.append(moved.value - base.value)
    assert max(gains) <= step.stationarity_gain + 2.0 * _EVIDENCE_TOLERANCE, (max(gains), step.stationarity_gain)


def test_a_maximum_at_its_basins_fold_is_certified_one_sided(monkeypatch):
    # The basin ends just past the stop on the side the gradient climbs: the weight's gain is at most what V can climb
    # before the fold, and the curvature comes from the other side.
    prior, cavity = _annotated_problem(101)
    weights, evidence, interior = _ascended(prior, cavity)
    position = int(np.flatnonzero(interior)[0])
    gradient = engine._full_gradient(prior, weights, evidence, interior, cavity, _WORKING_BYTES, engine._coarse_targets(evidence, interior, _EVIDENCE_TOLERANCE))[0]
    climb = 1.0 if gradient[position] >= 0.0 else -1.0
    corrected = engine._corrected

    def folded(view, trial_weights, trial, *rest):
        if climb * (trial_weights[position] - weights[position]) > 0.0:
            return None
        return corrected(view, trial_weights, trial, *rest)

    monkeypatch.setattr(engine, "_corrected", folded)
    check = engine._stationarity(prior, weights, evidence, interior, cavity, INDEPENDENT_EFFECTS, _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    assert check.better is None and check.folds[position] > 0.0
    reach = abs(check.gradient[position]) + check.error[position]
    assert check.gain >= reach * check.folds[position]
    assert np.isfinite(check.curvature[position, position])


def test_a_certifiably_better_side_is_taken_not_certified(monkeypatch):
    prior, cavity = _annotated_problem(101)
    weights, evidence, interior = _ascended(prior, cavity)
    corrected = engine._corrected

    def lifted(view, trial_weights, trial, *rest):
        found = corrected(view, trial_weights, trial, *rest)
        if found is None or np.array_equal(trial_weights, weights):
            return found
        return replace(found, value=found.value + 1.0)

    monkeypatch.setattr(engine, "_corrected", lifted)
    check = engine._stationarity(prior, weights, evidence, interior, cavity, INDEPENDENT_EFFECTS, _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    assert check.better is not None and check.better[1].value > evidence.value + _EVIDENCE_TOLERANCE


def test_an_orthogonal_reparametrization_of_every_block_leaves_the_evidence_and_the_posterior_unchanged():
    # x = T x' with T orthogonal within each coefficient block: every log-determinant moves by log|det T| = 0,
    # so V and the tilted means must agree to rounding. A normalization or log-determinant error would not.
    prior, cavity = _problem(variant_count=60, seed=41, node_count=12)
    hyperparameters = _hyperparameters(prior, 42, log_smoothing=2.0)
    generator = np.random.default_rng(43)
    rotation = np.zeros((prior.coefficient_size, prior.coefficient_size))
    covered = np.zeros(prior.coefficient_size, dtype=bool)
    for block in prior.smoothing_blocks:
        coordinates = block.coordinates[~covered[block.coordinates]]
        if coordinates.size:
            rotation[np.ix_(coordinates, coordinates)] = np.linalg.qr(generator.standard_normal((coordinates.size, coordinates.size)))[0]
            covered[coordinates] = True
    rotation[np.ix_(~covered, ~covered)] = np.eye(int((~covered).sum()))
    rotated = replace(
        prior,
        coefficient_map=prior.coefficient_map @ rotation,
        smoothing_blocks=tuple(
            replace(block, factor=block.factor @ rotation[np.ix_(block.coordinates, block.coordinates)]) for block in prior.smoothing_blocks
        ),
        null_basis=rotation.T @ prior.null_basis,
    )
    original = _evidence(prior, hyperparameters.log_smoothing, hyperparameters.coefficients, cavity, INDEPENDENT_EFFECTS, _WORKING_BYTES, 0.0)
    transformed = _evidence(rotated, hyperparameters.log_smoothing, rotation.T @ hyperparameters.coefficients, cavity, INDEPENDENT_EFFECTS, _WORKING_BYTES, 0.0)
    assert original is not None and transformed is not None
    np.testing.assert_allclose(transformed.value, original.value, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(transformed.gradient, original.gradient, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(rotation @ transformed.coefficients, original.coefficients, rtol=0.0, atol=1e-6)
    means = [
        tilted_moments(model, MixtureHyperparameters(evidence.coefficients, hyperparameters.log_smoothing), cavity, _WORKING_BYTES).mean
        for model, evidence in ((prior, original), (rotated, transformed))
    ]
    np.testing.assert_allclose(means[1], means[0], rtol=1e-8, atol=1e-12)


def _dense_ep(prior, coefficients, likelihood_precision, linear_term, sites):
    """Damped parallel EP on the dense Gaussian likelihood exp(-b' Lambda b / 2 + l' b), run to machine precision."""
    site_precision, site_shift = (np.array(part, copy=True) for part in sites)
    hyperparameters = MixtureHyperparameters(coefficients, np.zeros(len(prior.smoothing_blocks)))
    for _sweep in range(20000):
        covariance = np.linalg.inv(likelihood_precision + np.diag(site_precision))
        mean = covariance @ (linear_term + site_shift)
        cavity = cavities(mean, np.diag(covariance).copy(), site_precision, site_shift)
        target_precision, target_shift = site_targets(tilted_moments(prior, hyperparameters, cavity, _WORKING_BYTES), cavity)
        change = max(np.max(np.abs(target_precision - site_precision) / (1.0 + np.abs(site_precision))), np.max(np.abs(target_shift - site_shift) / (1.0 + np.abs(site_shift))))
        site_precision += 0.5 * (target_precision - site_precision)
        site_shift += 0.5 * (target_shift - site_shift)
        # The finite-difference derivative divides gradient error by 1e-5. A transient
        # stall below sqrt(eps) is not precise enough for this reference calculation.
        if change < 1e-14:
            return (site_precision, site_shift), covariance, cavity
    raise AssertionError("dense EP did not converge")


def test_total_curvature_matches_ep_resolved_differences_of_the_evidence_gradient():
    generator = np.random.default_rng(45)
    variant_count, sample_count = 30, 400
    latent = generator.standard_normal((sample_count, variant_count))
    for column in range(1, variant_count):
        latent[:, column] = 0.6 * latent[:, column - 1] + 0.8 * latent[:, column]
    genotypes = (latent - latent.mean(axis=0)) / latent.std(axis=0)
    effects = np.where(generator.random(variant_count) < 0.3, generator.normal(0.0, 0.15, variant_count), 0.0)
    targets = genotypes @ effects + generator.standard_normal(sample_count)
    likelihood_precision, linear_term = genotypes.T @ genotypes, genotypes.T @ targets
    class_index, offset, design, groups, _cavity = _data(variant_count, 46)
    nodes = np.linspace(np.log(1e-4), np.log(0.2), 8)
    prior = scale_mixture_prior(
        class_index=class_index, log_variance_offset=offset, annotation_design=design, annotation_groups=groups, nodes=nodes, floor=nodes[0] - 1.0, top=nodes[-1],
    )
    coefficients = _hyperparameters(prior, 47).coefficients * 0.3 + initial_hyperparameters(prior).coefficients
    start = moment_matched_prior_sites(prior, MixtureHyperparameters(coefficients, np.zeros(len(prior.smoothing_blocks))))
    sites, covariance, cavity = _dense_ep(prior, coefficients, likelihood_precision, linear_term, start)
    posterior = GaussianPosterior(
        solve=lambda right, _relative_tolerance: covariance @ right, variance_jvp=lambda weights: -np.einsum("jk,kr,kj->jr", covariance, weights, covariance)
    )
    analytic = _total_curvature(prior, coefficients, cavity, posterior, _WORKING_BYTES, 1e-13)
    mapping = prior.coefficient_map

    def evidence_gradient(point):
        _sites, _covariance, point_cavity = _dense_ep(prior, point, likelihood_precision, linear_term, sites)
        return mapping.T @ _data_objective(prior, point, point_cavity, _WORKING_BYTES).gradient

    step = 1e-5
    numerical = np.column_stack([
        -(evidence_gradient(coefficients + step * unit) - evidence_gradient(coefficients - step * unit)) / (2.0 * step)
        for unit in np.eye(coefficients.shape[0])
    ])
    numerical = 0.5 * (numerical + numerical.T)
    np.testing.assert_allclose(analytic, numerical, rtol=1e-5, atol=1e-5 * float(np.max(np.abs(numerical))))
    fixed_cavity = -(mapping.T @ _data_objective(prior, coefficients, cavity, _WORKING_BYTES).hessian @ mapping)
    assert np.max(np.abs(analytic - fixed_cavity)) > 1e-3 * float(np.max(np.abs(fixed_cavity)))

    # A posterior that solves the linear response exactly gives the same B as GMRES.
    squared = np.square(covariance)

    def linear_response(left, right, diagonal, weight, right_hand):
        matrix = np.eye(variant_count) - (np.eye(variant_count) - weight[:, None] * squared) @ (left[:, None] * covariance * right[None, :] + np.diag(diagonal))
        return np.linalg.solve(matrix, right_hand)

    exact = _total_curvature(prior, coefficients, cavity, replace(posterior, linear_response=linear_response), _WORKING_BYTES, 1e-13)
    np.testing.assert_allclose(exact, analytic, rtol=1e-9, atol=1e-9 * float(np.max(np.abs(analytic))))

    # The correction is solved only on the directions a view asks for: first an edge's free coefficients, then the
    # released ones; on each it is the whole correction's restriction.
    whole = analytic + mapping.T @ _data_objective(prior, coefficients, cavity, _WORKING_BYTES).hessian @ mapping
    lazy = curvature_correction(prior, coefficients, cavity, posterior, _WORKING_BYTES, 1e-13 * coefficients.shape[0])
    view, allowed = _restricted_prior(prior, frozenset(range(len(prior.smoothing_blocks))))
    scale = float(np.max(np.abs(whole)))
    np.testing.assert_allclose(lazy.on(view.coefficient_map), allowed.T @ whole @ allowed, rtol=1e-8, atol=1e-8 * scale)
    # One direction per independent column of the view's map (M is not injective: eta_bar and the class deviations
    # overlap in z), and only the new ones when every edge is released.
    assert lazy.solved_directions == np.linalg.matrix_rank(view.coefficient_map) < np.linalg.matrix_rank(mapping)
    np.testing.assert_allclose(lazy.on(mapping), whole, rtol=1e-8, atol=1e-8 * scale)
    assert lazy.solved_directions == np.linalg.matrix_rank(mapping)


def test_the_outer_step_never_certifies_where_the_total_curvature_is_indefinite():
    # A correction C = B - A of the form every real one has, M'(B_z - A_z)M, here with B_z = -A_z - I: then
    # B + S = S - A - M'M, a saddle. The Newton-B model reports it as indefinite with an infinite decrement (so no
    # certificate can be issued there), and its step is the trust-region maximizer of the model inside the radius.
    prior, cavity = _problem(variant_count=60, seed=17, node_count=12)
    hyperparameters = _hyperparameters(prior, 18, log_smoothing=2.0)
    penalty = _penalty_matrix(prior, hyperparameters.log_smoothing)
    objective = _data_objective(prior, hyperparameters.coefficients, cavity, _WORKING_BYTES)
    _value, _gradient, hessian = _penalized(prior, objective, hyperparameters.log_smoothing, penalty, hyperparameters.coefficients)
    mapping = prior.coefficient_map
    saddle = CurvatureCorrection(coefficient_map=mapping, matrix=mapping.T @ (2.0 * objective.hessian - np.eye(mapping.shape[0])) @ mapping)
    expected = -hessian + saddle.matrix
    assert np.linalg.eigvalsh(0.5 * (expected + expected.T))[0] < 0.0
    point = FixedPoint(
        cavity=cavity, posterior=diagonal_posterior(np.ones(prior.variant_count)), mean=np.zeros(prior.variant_count),
        precision_norm=lambda direction: float(direction @ direction), effective_effects=float(prior.variant_count),
    )
    newton = _newton_b(prior, hyperparameters.log_smoothing, hyperparameters.coefficients, point, saddle, _WORKING_BYTES)
    assert not newton.definite and newton.decrement == np.inf
    np.testing.assert_allclose(newton.total, 0.5 * (expected + expected.T), rtol=1e-10, atol=1e-10)
    radius = 0.5
    step = _proposal(newton, radius)
    assert np.linalg.norm(step) <= radius * (1.0 + 1e-12)
    # The model rises along the step: g's - s'(B + S)s / 2 > 0.
    assert float(newton.gradient @ step) - 0.5 * float(step @ newton.total @ step) > 0.0


@pytest.mark.xfail(strict=True, reason=_FLAT_REASON)
def test_the_outer_loop_refuses_a_trial_without_a_fixed_point_and_still_certifies():
    # Normal means: each effect's cavity is its own likelihood whatever the prior, so the exact fixed point is the
    # tilted law itself. The oracle has no fixed point at its second call (the first trial): the loop must refuse that
    # trial, count it, and go on to certify.
    prior, cavity = _problem(variant_count=60, seed=17, node_count=12)
    calls = []

    solved = []

    def fixed_points(hyperparameters):
        calls.append(len(calls))
        if len(calls) == 2:
            return [None] * len(hyperparameters)
        solved.append(np.array(hyperparameters[0].coefficients, copy=True))
        points = []
        for model in hyperparameters:
            moments = tilted_moments(prior, model, cavity, _WORKING_BYTES)
            points.append(FixedPoint(
                cavity=cavity, posterior=diagonal_posterior(moments.variance), mean=moments.mean,
                precision_norm=lambda direction, variance=moments.variance: float(np.sum(np.square(direction) / variance)),
                effective_effects=float(np.sum(cavity.precision * moments.variance)),
            ))
        return points

    (fit,) = fit_hyperparameters(prior, [initial_hyperparameters(prior)], fixed_points, _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    assert fit.unresolved >= 1 and len(calls) > 2 and fit.certified
    # The oracle's last fixed point is the returned model's own (review-mathbugs E1).
    np.testing.assert_array_equal(solved[-1], fit.hyperparameters.coefficients)
    assert fit.remaining_gain <= _EVIDENCE_TOLERANCE
    assert fit.prediction_move <= fit.prediction_tolerance


@pytest.mark.xfail(strict=True, reason=_FLAT_REASON)
def test_the_prediction_check_holds_each_block_to_its_own_budget():
    # Two independently scored blocks share x: each is held to its own KL budget (fit-api P1). Normal means, as in the
    # refusal test; with block 0's metric inflated past any budget, no step that moves its mean can certify.
    prior, cavity = _problem(variant_count=60, seed=17, node_count=12)
    halves = (slice(0, 30), slice(30, 60))

    def fixed_points(hyperparameters, inflation):
        points = []
        for model in hyperparameters:
            moments = tilted_moments(prior, model, cavity, _WORKING_BYTES)
            variance = moments.variance

            def norm(direction, variance=variance):
                moves = np.array([float(np.sum(np.square(direction[part]) / variance[part])) for part in halves])
                moves[0] *= inflation
                return moves

            effective = np.array([float(np.sum(cavity.precision[part] * variance[part])) for part in halves])
            points.append(FixedPoint(cavity=cavity, posterior=diagonal_posterior(variance), mean=moments.mean, precision_norm=norm, effective_effects=effective))
        return points

    (fit,) = fit_hyperparameters(prior, [initial_hyperparameters(prior)], lambda h: fixed_points(h, 1.0), _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    assert fit.certified and fit.prediction_move <= fit.prediction_tolerance
    inflation = 1.0 / float(np.finfo(np.float64).eps) ** 2
    (starved,) = fit_hyperparameters(prior, [initial_hyperparameters(prior)], lambda h: fixed_points(h, inflation), _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    assert not starved.certified


def test_an_empirical_bayes_with_no_certified_rise_ends_uncertified_at_its_no_progress_budget(monkeypatch):
    """A walk outside every basin (no state the loop reaches has a certified value: the whole-genome arms' plans at
    "gain nan, state error inf" [real]) has no ending of its own until x halves to double precision: the no-progress
    budget ends it after ``EB_STALL_TRIALS`` trials, or ``EB_STALL_HOURS``, uncertified where it stands, with the
    reason as its certificate's refusal and the oracle at the point it returns."""
    # Normal means, whose exact fixed point is the tilted law itself; no state is certified. Each fixed point's moves
    # are inflated past the certificate's resolution, so no step reads as unmoved outside every basin (an ending of
    # its own, ``fixed point unmoved``) and only the budget can end the walk.
    prior, cavity = _problem(variant_count=60, seed=17, node_count=12)
    monkeypatch.setattr(engine, "_outer_state", lambda *arguments, **keywords: None)
    monkeypatch.setattr(engine, "EB_STALL_TRIALS", 3)
    solved: list = []

    def fixed_points(hyperparameters):
        solved.append(np.array(hyperparameters[0].coefficients, copy=True))
        points = []
        for model in hyperparameters:
            moments = tilted_moments(prior, model, cavity, _WORKING_BYTES)
            points.append(FixedPoint(
                cavity=cavity, posterior=diagonal_posterior(moments.variance), mean=moments.mean,
                precision_norm=lambda direction, variance=moments.variance: 1e6 * float(np.sum(np.square(direction) / variance)) + 1.0,
                effective_effects=float(np.sum(cavity.precision * moments.variance)),
            ))
        return points

    (fit,) = fit_hyperparameters(prior, [initial_hyperparameters(prior)], fixed_points, _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    assert not fit.certified and fit.remaining_gain == np.inf and fit.prediction_move == np.inf
    assert "no certified rise of V in 3 trials" in fit.refusal and fit.refusal.endswith("it returns where it stood, uncertified")
    # The start's solve, the three trials, and at most one more, at the returned point.
    assert 4 <= len(solved) <= 5
    np.testing.assert_array_equal(solved[-1], fit.hyperparameters.coefficients)
    # A budget of no hours ends the search before its first trial, at its start.
    monkeypatch.setattr(engine, "EB_STALL_HOURS", 0.0)
    solved.clear()
    start = initial_hyperparameters(prior)
    (stopped,) = fit_hyperparameters(prior, [start], fixed_points, _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    assert not stopped.certified and "no certified rise of V in 0 trials" in stopped.refusal
    assert len(solved) == 1 and stopped.iterations == 0
    np.testing.assert_array_equal(stopped.hyperparameters.coefficients, start.coefficients)


def test_a_density_below_the_kernel_floor_is_a_near_zero_effect_not_a_point_mass():
    # review-mathbugs N1 [real: gene 3 snv_sv's first outer trial]: with every node below the floor given v = 0, a
    # class whose density sits there had tilted variance exactly 0, so the site targets were tau = inf, nu = nan.
    prior, cavity = _problem(variant_count=60, seed=39, node_count=12)
    nodes = prior.log_variance_grid
    below = initial_hyperparameters(prior, float(np.exp(nodes[0])))
    moments = tilted_moments(prior, below, cavity, _WORKING_BYTES)
    assert np.all(moments.variance > 0.0) and np.all(np.isfinite(moments.mean))
    precision, shift = site_targets(moments, cavity)
    assert np.all(np.isfinite(precision)) and np.all(np.isfinite(shift))


def test_the_first_trust_region_trial_is_bounded_by_the_cauchy_step():
    # The first trust radius is the Cauchy step's length on |B + S|: a direction the data barely curve cannot send the
    # first trial off to the rounding floor's 1 / eps (review-mathbugs: |x| 1.3e4 on a real gene). An indefinite model
    # (the saddle correction of the outer-step test) steps inside it.
    prior, cavity = _problem(variant_count=60, seed=17, node_count=12)
    hyperparameters = _hyperparameters(prior, 18, log_smoothing=2.0)
    objective = _data_objective(prior, hyperparameters.coefficients, cavity, _WORKING_BYTES)
    mapping = prior.coefficient_map
    saddle = CurvatureCorrection(coefficient_map=mapping, matrix=mapping.T @ (2.0 * objective.hessian - np.eye(mapping.shape[0])) @ mapping)
    moments = tilted_moments(prior, hyperparameters, cavity, _WORKING_BYTES)
    point = FixedPoint(cavity=cavity, posterior=diagonal_posterior(moments.variance), mean=moments.mean,
                       precision_norm=lambda d: float(np.sum(np.square(d) / moments.variance)), effective_effects=1.0)
    newton = engine._newton_b(prior, hyperparameters.log_smoothing, hyperparameters.coefficients, point, saddle, _WORKING_BYTES)
    assert not newton.definite
    radius = engine._cauchy_radius(newton)
    assert 0.0 < radius <= float(np.linalg.norm(newton.gradient)) / float(np.min(np.abs(newton.eigenvalues)))
    assert float(np.linalg.norm(engine._proposal(newton, radius))) <= radius * (1.0 + 1e-9)


def test_a_joint_step_halved_with_no_certified_value_at_rho_hat_moves_its_weights_too(monkeypatch):
    """A refused joint step halved at rho-hat tends to (x_k, rho-hat) as the fraction falls, not to the certified state,
    so where V's formula has no certified value at rho-hat no halving need have one: on the whole-genome fits [real,
    aou-ws6 snv_indel] a polished state's step had none at fractions 1 to 0.0625, 45 minutes a trial, and halved on to
    x's resolution. Here every plan's step moves the finite weights down by one (x where the hyper step placed it), and
    V's formula has no certified value beyond 40% of that move: after one halving at rho-hat the halvings move the
    weights with x, toward the certified state, and reach a certified value."""
    prior, cavity = _problem(variant_count=60, seed=17, node_count=12)
    start = replace(initial_hyperparameters(prior), log_smoothing=np.full(len(prior.smoothing_blocks), 2.0))
    # From weights 2.0 the fit walks 17 trials outside every basin, before any plan, with no certified rise of V; the
    # no-progress budget would end it there, before its first joint trial. Twice that walk.
    monkeypatch.setattr(engine, "EB_STALL_TRIALS", 34)
    real_step, real_state = engine.hyper_step, engine._outer_state
    planned: dict[str, np.ndarray] = {}

    def step(view, hyperparameters, *arguments, **keywords):
        found = real_step(view, hyperparameters, *arguments, **keywords)
        weights = np.array(hyperparameters.log_smoothing, copy=True)
        planned["from"], planned["to"] = weights, np.where(np.isfinite(weights), weights - 1.0, weights)
        return replace(found, hyperparameters=MixtureHyperparameters(coefficients=found.hyperparameters.coefficients, log_smoothing=planned["to"]))

    def walled(view, hyperparameters, *arguments, **keywords):
        if planned:
            weights = hyperparameters.log_smoothing
            finite = np.isfinite(planned["from"]) & np.isfinite(weights)
            move = planned["to"][finite] - planned["from"][finite]
            if float((weights[finite] - planned["from"][finite]) @ move) > 0.4 * float(move @ move):
                return None
        return real_state(view, hyperparameters, *arguments, **keywords)

    monkeypatch.setattr(engine, "hyper_step", step)
    monkeypatch.setattr(engine, "_outer_state", walled)
    lines: list[str] = []
    monkeypatch.setattr(engine, "log", lines.append)

    def fixed_points(hyperparameters):
        points = []
        for model in hyperparameters:
            moments = tilted_moments(prior, model, cavity, _WORKING_BYTES)
            points.append(FixedPoint(
                cavity=cavity, posterior=diagonal_posterior(moments.variance), mean=moments.mean,
                precision_norm=lambda direction, variance=moments.variance: float(np.sum(np.square(direction) / variance)),
                effective_effects=float(np.sum(cavity.precision * moments.variance)),
            ))
        return points

    fit_hyperparameters(prior, [start], fixed_points, _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    trials = [line for line in lines if " joint trial at fraction " in line]
    run = longest = 0
    for line in trials:
        run = run + 1 if line.endswith(": no certified value") and " (x and weights) " not in line else 0
        longest = max(longest, run)
    # The whole step and one halving at rho-hat (on 11470d5 the polished state's step had 27, to x's resolution).
    assert longest <= 2, "\n".join(trials)
    assert any(" (x and weights) " in line and line.endswith((": accepted", ": refused")) for line in trials), "\n".join(trials)


def test_total_curvature_is_the_fixed_cavity_curvature_for_independent_effects():
    prior, cavity = _problem(variant_count=40, seed=48, node_count=10)
    coefficients = _hyperparameters(prior, 49).coefficients
    moments = tilted_moments(prior, MixtureHyperparameters(coefficients, np.zeros(len(prior.smoothing_blocks))), cavity, _WORKING_BYTES)
    analytic = _total_curvature(prior, coefficients, cavity, diagonal_posterior(moments.variance), _WORKING_BYTES, 1e-13)
    mapping = prior.coefficient_map
    fixed_cavity = -(mapping.T @ _data_objective(prior, coefficients, cavity, _WORKING_BYTES).hessian @ mapping)
    np.testing.assert_allclose(analytic, fixed_cavity, rtol=1e-9, atol=1e-9 * float(np.max(np.abs(fixed_cavity))))


_GAP_CASE = pathlib.Path(__file__).resolve().parent / "data" / "gap_v7_x1000_r0.npz"


def _dense_fixed_points(prior, likelihood_precision, linear_term):
    """The exact EP fixed point of a dense Gaussian likelihood exp(-b' Lambda b / 2 + l' b), warm between calls."""
    count = linear_term.shape[0]
    state = {"sites": None}

    def fixed_points(hyperparameters):
        points = []
        for model in hyperparameters:
            starts = ([state["sites"]] if state["sites"] is not None else []) + [moment_matched_prior_sites(prior, model)]
            for start in starts:
                try:
                    sites, covariance, cavity = _dense_ep(prior, model.coefficients, likelihood_precision, linear_term, start)
                    break
                except (AssertionError, FloatingPointError, np.linalg.LinAlgError):
                    continue
            else:
                points.append(None)
                continue
            state["sites"] = sites
            squared = np.square(covariance)

            def linear_response(left, right, diagonal, weight, right_hand, covariance=covariance, squared=squared):
                matrix = np.eye(count) - (np.eye(count) - weight[:, None] * squared) @ (left[:, None] * covariance * right[None, :] + np.diag(diagonal))
                return np.linalg.solve(matrix, right_hand)

            precision = likelihood_precision + np.diag(sites[0])
            points.append(FixedPoint(
                cavity=cavity,
                posterior=GaussianPosterior(solve=lambda right, _e, c=covariance: c @ right, variance_jvp=lambda w, s=squared: -(s @ w), linear_response=linear_response),
                mean=covariance @ (linear_term + sites[1]),
                precision_norm=lambda direction, a=precision: float(direction @ a @ direction),
                effective_effects=float(count - np.sum(sites[0] * np.diag(covariance))),
            ))
        return points

    return fixed_points


@pytest.mark.slow
def test_a_halved_lattice_does_not_alias_the_fitted_density_on_the_v7_gap_case():
    # oracle's finding [semi-real, bench-sim v7 x1000, 80 variants]: on the halved lattice its interior search found
    # densities narrower than one spacing, which alias to a few atoms and lifted the objective from 23.59 to 24.27, above
    # any continuous density. The engine's certified fit must give the same evidence on the lattice and on its halving.
    data = np.load(_GAP_CASE)
    likelihood_precision, linear_term = data["likelihood_precision"], data["linear_term"]
    count = linear_term.shape[0]
    offset = np.full(count, float(np.asarray(data["offset"]).ravel()[0]))
    nodes, floor, top = derived_lattice(np.diag(likelihood_precision), linear_term, offset, _EVIDENCE_TOLERANCE)
    values = []
    for lattice in (nodes, np.linspace(nodes[0], nodes[-1], 2 * nodes.shape[0] - 1)):
        prior = scale_mixture_prior(
            class_index=np.zeros(count, np.int64), log_variance_offset=offset, annotation_design=np.zeros((count, 0)), annotation_groups=(),
            nodes=lattice, floor=floor, top=top,
        )
        (fit,) = fit_hyperparameters(prior, [initial_hyperparameters(prior)], _dense_fixed_points(prior, likelihood_precision, linear_term), 1 << 26, _EVIDENCE_TOLERANCE)
        values.append(fit.step.evidence)
        assert fit.remaining_gain <= _EVIDENCE_TOLERANCE
    assert abs(values[0] - values[1]) <= 2.0 * _EVIDENCE_TOLERANCE, values


@pytest.mark.slow
def test_a_search_started_at_its_certified_optimum_certifies_again():
    """The outer loop continued from its own certified fit (an annotated search from the base fit, a lattice check's
    refit): its plan's step has zero length, whose joint trial is the state itself, so the remainder is known without
    a trial and the fit certifies, never ending "planned twice from one state" uncertified."""
    data = np.load(_GAP_CASE)
    likelihood_precision, linear_term = data["likelihood_precision"], data["linear_term"]
    count = linear_term.shape[0]
    offset = np.full(count, float(np.asarray(data["offset"]).ravel()[0]))
    nodes, floor, top = derived_lattice(np.diag(likelihood_precision), linear_term, offset, _EVIDENCE_TOLERANCE)
    prior = scale_mixture_prior(
        class_index=np.zeros(count, np.int64), log_variance_offset=offset, annotation_design=np.zeros((count, 0)), annotation_groups=(),
        nodes=nodes, floor=floor, top=top,
    )
    # One oracle, warm between the two searches, as the refit's oracle starts at the fit's own fixed point.
    fixed_points = _dense_fixed_points(prior, likelihood_precision, linear_term)
    (fit,) = fit_hyperparameters(prior, [initial_hyperparameters(prior)], fixed_points, 1 << 26, _EVIDENCE_TOLERANCE)
    assert fit.certified
    (again,) = fit_hyperparameters(prior, [fit.hyperparameters], fixed_points, 1 << 26, _EVIDENCE_TOLERANCE)
    assert again.certified and again.remaining_gain <= _EVIDENCE_TOLERANCE
    assert abs(again.step.evidence - fit.step.evidence) <= 2.0 * _EVIDENCE_TOLERANCE
    # Its first plan certifies: no joint trial had to measure the zero step's remainder (and be refused for its zero gain).
    assert again.halvings == 0 and len(again.history) == 1, (again.halvings, again.history)


def test_a_joint_trial_rises_on_vs_lower_bound_with_only_its_end_decrement_credited():
    """``_rises``: x short of its maximum leaves V above W, never below, so the end state's decrement is credited
    against the resolution (both states' errors), and the start's is not (V(rho_start) can lie above W_start by its
    whole error). A trial that realized 0.285 nats from a start whose error is 0.2 to an end whose error is 0.09, 0.085
    of it the end's decrement, was refused as a fall of 0.005 below the errors; V's lower bound rose by 0.08."""
    from sv_pgs.scale_mixture_ep import _rises, _State

    def state(error, decrement):
        return _State(value=0.0, rest=0.0, gradient=np.zeros(1), fixed_curvature=np.zeros((1, 1)), correction=None, error=error, decrement=decrement)

    gain = 0.285
    for start_decrement in (0.0, 0.2):
        start = state(0.2, start_decrement)
        assert _rises(gain, start.error + 0.09, state(0.09, 0.085), _EVIDENCE_TOLERANCE)
        # Error that is not x's decrement (the determinant's and the corrections' move with x, rounding) is a fall.
        assert not _rises(gain, start.error + 0.09, state(0.09, 0.0), _EVIDENCE_TOLERANCE)
    # The credit never exceeds the end's own error: a gain the start's error alone takes stays refused.
    assert not _rises(0.15, 0.2 + 0.09, state(0.09, 0.085), _EVIDENCE_TOLERANCE)


def test_a_refused_unmeasured_trial_halves_only_where_its_quadratic_clears_the_state_error():
    """``_halved_reach``: the halvings of a refused joint step reach at most the quadratic through the state's slope
    and the whole step's predicted gain, at half the fraction or at the quadratic's peak before it. On LDL's
    whole-genome fit [real, aou-wl1 snv_indel] the whole step from a state of error 3.75 predicted 4.964 and had no
    certified end, and its half step realized 3.684, refused on that error after one more fixed point and its
    curvature. With the slope of x's own Newton step (twice its decrement), or of the whole prediction's straight
    line, the quadratic reaches under the state's error at half the step, so no halving is tried; a slope well above
    the prediction (most of the step's gain in its first half) still halves."""
    from sv_pgs.scale_mixture_ep import _halved_reach

    error, predicted = 3.75, 4.964
    for slope in (2.0 * error, 2.0 * predicted):
        reach = _halved_reach(slope, predicted, 1.0)
        assert reach == pytest.approx(0.5 * slope + 0.25 * (predicted - slope))
        assert reach < error + _EVIDENCE_TOLERANCE
    assert _halved_reach(20.0, predicted, 1.0) > error + _EVIDENCE_TOLERANCE
    # The peak where it falls before half the fraction: s^2 / (4 |c|) at f = s / (2 |c|) = 0.25.
    assert _halved_reach(20.0, -20.0, 1.0) == pytest.approx(2.5)
    # Each halving's own fraction, and never below zero.
    assert _halved_reach(2.0 * error, predicted, 0.5) == pytest.approx(0.25 * 2.0 * error + 0.0625 * (predicted - 2.0 * error))
    assert _halved_reach(-1.0, -2.0, 1.0) == 0.0


def test_the_trust_region_step_takes_the_hard_case_exactly():
    """More and Sorensen's hard case: g = 0 at an indefinite point (and g orthogonal to -H's lowest eigenvector) puts
    mu at -lambda_min, where the lowest eigenspace's shifted eigenvalue is 0. The step reaches the boundary along the
    lowest eigenvector, with no division by zero (speed-krylov's -W error run)."""
    import warnings

    from sv_pgs.scale_mixture_ep import _trust_region_step

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        step = _trust_region_step(np.diag([-1.0, 2.0]), np.zeros(2), 1.0)
        np.testing.assert_allclose(np.abs(step), [1.0, 0.0], atol=1e-15)
        # g orthogonal to the lowest eigenvector, its own part inside the radius: the rest at mu = 1, then the boundary.
        step = _trust_region_step(np.diag([-1.0, 2.0]), np.array([0.0, 1.5]), 1.0)
        np.testing.assert_allclose(step[1], 0.5, rtol=1e-12)
        np.testing.assert_allclose(float(np.linalg.norm(step)), 1.0, rtol=1e-12)


def test_offset_group_levels_shift_every_variant_of_a_group_by_its_level_whatever_its_class():
    """review-mathbugs P2 (lead ruling): levels are gene-owned offsets, sum-to-zero over groups, never class-centred."""
    class_index = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    groups = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    nodes = np.linspace(np.log(1e-5), np.log(0.5), 8)
    prior = scale_mixture_prior(
        class_index=class_index, log_variance_offset=np.zeros(8), annotation_design=np.zeros((8, 0)), annotation_groups=(),
        nodes=nodes, floor=nodes[0] - 1.0, top=nodes[-1], offset_groups=groups,
    )
    assert prior.level_size == 1 and prior.smoothing_blocks[-1].name == "offset group levels"
    levels = np.array([0.7, -0.7])
    coefficients = np.zeros(prior.coefficient_size)
    coefficients[-1:] = _sum_to_zero_rows(np.arange(2), 2).T @ levels
    shift = log_scale(prior, coefficients) - log_scale(prior, np.zeros(prior.coefficient_size))
    np.testing.assert_allclose(shift, levels[groups], rtol=0.0, atol=8 * np.finfo(np.float64).eps)
    hyperparameters = MixtureHyperparameters(coefficients=coefficients, log_smoothing=np.zeros(len(prior.smoothing_blocks)))
    moved, moved_hyperparameters = relattice(prior, hyperparameters, np.linspace(nodes[0], nodes[-1], 12), nodes[0] - 1.0, nodes[-1])
    np.testing.assert_array_equal(moved.offset_groups, groups)
    moved_shift = log_scale(moved, moved_hyperparameters.coefficients) - log_scale(moved, np.zeros(moved.coefficient_size))
    np.testing.assert_allclose(moved_shift, levels[groups], rtol=0.0, atol=np.sqrt(np.finfo(np.float64).eps))


def test_offset_groups_that_do_not_connect_the_classes_are_refused():
    nodes = np.linspace(np.log(1e-5), np.log(0.5), 8)
    with pytest.raises(ValueError, match="connect the classes"):
        scale_mixture_prior(
            class_index=np.array([0, 0, 1, 1]), log_variance_offset=np.zeros(4), annotation_design=np.zeros((4, 0)), annotation_groups=(),
            nodes=nodes, floor=nodes[0] - 1.0, top=nodes[-1], offset_groups=np.array([0, 0, 1, 1]),
        )


def test_offset_group_levels_are_the_gathered_helmert_rows_and_many_groups_need_no_dense_indicator():
    """F18: the level design is each variant's group's row of the orthonormal sum-to-zero basis (no variant x group
    indicator, no group x group decomposition), and the identifiability test runs on grouped sums."""
    rng = np.random.default_rng(61)
    count, group_count = 3000, 400
    groups = rng.integers(0, group_count, count)
    classes = rng.integers(0, 2, count)
    nodes = np.linspace(np.log(1e-5), np.log(0.5), 8)
    prior = scale_mixture_prior(
        class_index=classes, log_variance_offset=np.zeros(count), annotation_design=np.zeros((count, 0)), annotation_groups=(),
        nodes=nodes, floor=nodes[0] - 1.0, top=nodes[-1], offset_groups=groups,
    )
    labels, group_of_row = np.unique(groups, return_inverse=True)
    np.testing.assert_array_equal(prior.scale_design, _sum_to_zero_rows(group_of_row, labels.shape[0]))
    # a level vector l = H c shifts every member of group g by l_g and sums to zero over the groups
    levels = rng.normal(size=prior.level_size)
    coefficients = np.zeros(prior.coefficient_size)
    coefficients[-prior.level_size :] = levels
    shift = engine.log_scale(prior, coefficients)
    per_group = np.array([shift[group_of_row == group][0] for group in range(labels.shape[0])])
    np.testing.assert_allclose(shift, per_group[group_of_row], atol=1e-14)
    np.testing.assert_allclose(per_group.sum(), 0.0, atol=1e-12)


def test_the_lattice_check_flags_mass_at_a_lattice_end_and_passes_a_contained_density():
    """F17: at a fitted state whose density rises to the lattice's top, the extended lattice changes the normalizers;
    a density well inside a fine lattice changes under neither refinement nor extension."""
    rng = np.random.default_rng(7)
    count = 50
    nodes = np.linspace(-6.0, 2.0, 33)
    prior = scale_mixture_prior(
        class_index=np.zeros(count, dtype=np.int64), log_variance_offset=np.zeros(count), annotation_design=np.zeros((count, 0)),
        annotation_groups=(), nodes=nodes, floor=nodes[0] + 1.0, top=nodes[-1] - 1.0,
    )
    cavity = Cavity(precision=np.full(count, 50.0), shift=rng.normal(0.0, 30.0, count))

    def at(log_density):
        coefficients = np.zeros(prior.coefficient_size)
        coefficients[: prior.pooled_size] = prior.coefficient_map[: prior.grid_size, : prior.pooled_size].T @ (log_density - log_density.mean())
        return MixtureHyperparameters(coefficients=coefficients, log_smoothing=np.zeros(len(prior.smoothing_blocks)))

    rising = engine.lattice_check(prior, at(3.0 * nodes), cavity, 1 << 24, 64)
    assert rising.extend and rising.extended_log_normalizer > rising.tolerance
    contained = engine.lattice_check(prior, at(-0.5 * ((nodes + 2.0) / 0.7) ** 2), cavity, 1 << 24, 64)
    assert not contained.extend and not contained.refine, contained.record()


def test_lattice_check_releases_each_comparison_prior_before_constructing_the_next(monkeypatch):
    prior, cavity = _problem(variant_count=30, seed=8, node_count=9)
    hyperparameters = initial_hyperparameters(prior)
    expected = engine.lattice_check(prior, hyperparameters, cavity, _WORKING_BYTES, 64)
    previous = []

    def tracked(move):
        def invoke(*args):
            assert all(reference() is None for reference in previous), "the previous dense comparison prior is still live"
            moved, coefficients = move(*args)
            previous.append(weakref.ref(moved))
            return moved, coefficients
        return invoke

    monkeypatch.setattr(engine, "halved_lattice", tracked(engine.halved_lattice))
    monkeypatch.setattr(engine, "extended_lattice", tracked(engine.extended_lattice))
    actual = engine.lattice_check(prior, hyperparameters, cavity, _WORKING_BYTES, 64)
    assert actual == expected
    assert len(previous) == 2 and all(reference() is None for reference in previous)


@pytest.mark.parametrize("move", [engine.halved_lattice, engine.extended_lattice])
def test_group_relattice_constructs_one_prior_and_preserves_separate_model_transfers(move):
    from sv_pgs.stage2_wiring import _move_lattice

    prior, _ = _problem(variant_count=30, seed=8, node_count=9)
    first = initial_hyperparameters(prior)
    second = MixtureHyperparameters(
        coefficients=np.random.default_rng(92).normal(size=prior.coefficient_size),
        log_smoothing=np.zeros(len(prior.smoothing_blocks)),
    )
    expected = [move(prior, hyperparameters) for hyperparameters in (first, second)]
    calls = []

    def counted(*args):
        calls.append(1)
        return move(*args)

    shared, transferred = _move_lattice(prior, [first, second], counted)
    assert len(calls) == 1
    for actual, (separate, reference) in zip(transferred, expected):
        np.testing.assert_array_equal(shared.coefficient_map, separate.coefficient_map)
        np.testing.assert_allclose(actual.coefficients, reference.coefficients, rtol=0, atol=0)
        np.testing.assert_array_equal(actual.log_smoothing, reference.log_smoothing)


def test_a_finite_log_variance_past_double_precision_keeps_its_finite_normalizer():
    # The audit's M18: at log v = 1000, P = 1, h = 0 the log normalizer is -1/2 log(1 + v P) = -500, not -inf.
    conditional, retained, ratio_retained, log_component, _signal = engine._kernel_terms(
        np.zeros((1, 1)), np.zeros(1), np.array([1000.0]), np.array([1.0]), np.array([0.0])
    )
    assert np.isfinite(log_component[0, 0]) and np.isclose(log_component[0, 0], -500.0, rtol=0.0, atol=1e-9)
    assert conditional[0, 0] == 1.0 and retained[0, 0] == 0.0 and ratio_retained[0, 0] == 1.0
    # With a shift the h^2 / (2P) term stays too.
    log_component = engine._kernel_terms(np.zeros((1, 1)), np.zeros(1), np.array([1000.0]), np.array([2.0]), np.array([3.0]))[3]
    assert np.isclose(log_component[0, 0], -0.5 * (1000.0 + np.log(2.0)) + 0.5 * 9.0 / 2.0, rtol=0.0, atol=1e-9)


def test_the_fused_host_objective_is_the_held_rows_objective():
    """``_host_objective`` (one fused pass per row, the device kernel's host twin) against the held kernel rows' path,
    value, gradient and Hessian, with an annotation scale design and in small chunks; and ``_data_value``'s fused
    pass against the rows' normalizers."""
    from sv_pgs.scale_mixture_ep import _held_rows_objective, _host_objective, _data_value, _kernel_chunks, class_log_density, log_scale

    prior, cavity = _problem(variant_count=60, seed=51, node_count=12)
    assert prior.scale_size
    hyperparameters = _hyperparameters(prior, 52, log_smoothing=1.0)
    expected = _held_rows_objective(prior, hyperparameters.coefficients, cavity, _WORKING_BYTES, True)
    for working_bytes in (_WORKING_BYTES, 1 << 12):
        got = _host_objective(prior, hyperparameters.coefficients, cavity, working_bytes, True)
        np.testing.assert_allclose(got.value, expected.value, rtol=1e-12)
        np.testing.assert_allclose(got.magnitude, expected.magnitude, rtol=1e-12)
        scale = float(np.max(np.abs(expected.gradient)))
        np.testing.assert_allclose(got.gradient, expected.gradient, rtol=1e-12, atol=1e-12 * scale)
        scale = float(np.max(np.abs(expected.hessian)))
        np.testing.assert_allclose(got.hessian, expected.hessian, rtol=1e-12, atol=1e-12 * scale)
    log_density = class_log_density(prior, hyperparameters.coefficients)
    scales = log_scale(prior, hyperparameters.coefficients)
    held = sum(float(np.sum(rows.normalizers(log_density[c])[0])) for c, _r, rows in _kernel_chunks(prior, scales, cavity, _WORKING_BYTES))
    np.testing.assert_allclose(_data_value(prior, hyperparameters.coefficients, cavity, _WORKING_BYTES, np), held, rtol=1e-12)


def _reference_variant_derivatives(prior, coefficients, cavity):
    """The derivatives of ``_variant_derivatives``' docstring from the dense components, two-pass centred."""
    from sv_pgs.scale_mixture_ep import _KernelRows, class_log_density, log_scale

    log_density = class_log_density(prior, coefficients)
    scales = log_scale(prior, coefficients)
    out = {name: np.empty(prior.variant_count) for name in (
        "mean", "second", "variance", "variance_by_shift", "mean_by_precision", "variance_by_precision", "mean_by_log_scale", "second_by_log_scale",
    )}
    by_mean = np.empty((prior.variant_count, prior.grid_size))
    by_second = np.empty((prior.variant_count, prior.grid_size))
    for class_position, rows in enumerate(prior.class_rows):
        kernel = _KernelRows(scales[rows], prior.log_variance_grid, cavity.precision[rows], cavity.shift[rows])
        terms = kernel.components(log_density[class_position])
        w, c, r = terms.responsibility, terms.conditional_variance, kernel.retained
        mu = cavity.shift[rows][:, None] * c
        raw = c + mu * mu
        ell = -0.5 * raw
        f = terms.first

        def expectation(values):
            return np.sum(w * values, axis=1)

        m, s2 = expectation(mu), expectation(raw)
        d, dr = mu - m[:, None], raw - s2[:, None]
        dl, df = ell - expectation(ell)[:, None], f - expectation(f)[:, None]
        m_p = expectation(dl * d) - expectation(mu * c)
        out["mean"][rows], out["second"][rows], out["variance"][rows] = m, s2, s2 - m * m
        out["variance_by_shift"][rows] = expectation(d**3) + 3.0 * expectation(c * d)
        out["mean_by_precision"][rows] = m_p
        out["variance_by_precision"][rows] = expectation(dl * dr) - expectation(c * c + 2.0 * mu * mu * c) - 2.0 * m * m_p
        out["mean_by_log_scale"][rows] = expectation(df * d) + expectation(mu * r)
        out["second_by_log_scale"][rows] = expectation(df * dr) + expectation((c + 2.0 * mu * mu) * r)
        by_mean[rows], by_second[rows] = w * d, w * dr
    return out, by_mean, by_second


def test_the_fused_variant_derivatives_are_the_dense_components_moments():
    """``_variant_derivatives``' fused host pass (``_derivative_rows``) against the moments formed densely from the
    components, with an annotation scale design."""
    from sv_pgs.scale_mixture_ep import _variant_derivatives

    prior, cavity = _problem(variant_count=60, seed=51, node_count=12)
    hyperparameters = _hyperparameters(prior, 52, log_smoothing=1.0)
    got = _variant_derivatives(prior, hyperparameters.coefficients, cavity, _WORKING_BYTES)
    expected, by_mean, by_second = _reference_variant_derivatives(prior, hyperparameters.coefficients, cavity)
    for name, values in expected.items():
        scale = float(np.max(np.abs(values)))
        np.testing.assert_allclose(getattr(got, name), values, rtol=1e-10, atol=1e-10 * scale, err_msg=name)
    np.testing.assert_allclose(got.mean_by_density, by_mean, rtol=1e-10, atol=1e-10 * float(np.max(np.abs(by_mean))))
    np.testing.assert_allclose(got.second_by_density, by_second, rtol=1e-10, atol=1e-10 * float(np.max(np.abs(by_second))))


def test_the_trust_radius_follows_the_models_measured_agreement():
    """``_next_radius``: a step that reached the radius goes to L / (2 e), e the model's measured relative error along
    it, where that is past doubling. An error >= 1/2 (measured, or the realized gain's resolution over the prediction)
    supports no longer step; a step short of the radius keeps it; a deviation within the gain's resolution doubles it;
    an undefined ratio (an unmeasured gain, or a prediction of no gain) keeps it."""
    from types import SimpleNamespace

    from sv_pgs.scale_mixture_ep import _next_radius

    newton = SimpleNamespace(gradient=np.array([2.0, 0.0]), total=np.diag([1.0, 1.0]))
    proposal = np.array([1.0, 0.0])
    predicted = 2.0 - 0.5
    assert _next_radius(newton, proposal, 1.0, 1.0, predicted * 0.99, 1e-6) == pytest.approx(1.0 / (2.0 * 0.01))
    assert _next_radius(newton, proposal, 1.0, 1.0, predicted * 0.2, 1e-6) == 1.0
    assert _next_radius(newton, proposal, 1.0, 1.0, -predicted, 1e-6) == 1.0
    assert _next_radius(newton, proposal, 1.0, 1.0, predicted, predicted / 2) == 1.0
    # The model agrees with the realized gain to within its resolution: the error is unmeasured, so no extrapolation.
    assert _next_radius(newton, proposal, 1.0, 1.0, predicted, 0.03) == 2.0
    assert _next_radius(newton, proposal, 1.0, 1.0, predicted * (1.0 - 1e-12), 1e-15) == pytest.approx(1.0 / (2.0 * 1e-12), rel=1e-3)
    assert _next_radius(newton, proposal, 1.0, 1.0, predicted * (1.0 - 1e-15), 1e-12) == 2.0
    assert _next_radius(newton, proposal, 4.0, 1.0, predicted, 1e-6) == 4.0
    assert _next_radius(newton, proposal, 1.0, 1.0, np.nan, np.nan) == 1.0
    assert _next_radius(newton, proposal, 1.0, 1.0, predicted, np.inf) == 1.0
    # B + S curves the step down by more than g rises along it: the model predicts no gain, so the ratio is undefined.
    assert _next_radius(SimpleNamespace(gradient=np.array([1.0, 0.0]), total=np.diag([3.0, 1.0])), proposal, 1.0, 1.0, 0.5, 1e-6) == 1.0


def test_an_inner_step_is_accepted_only_where_its_trust_region_ratio_is_defined():
    """``_ratio_defined``: both gains finite and positive and their quotient finite. Two finite positive gains whose
    quotient overflows (a prediction of 1e-320 against a realized 1e-3) leave the ratio undefined, so the step is
    refused."""
    from sv_pgs.scale_mixture_ep import _ratio_defined

    assert _ratio_defined(0.5, 1.0) and _ratio_defined(1e-300, 1e-300)
    assert not _ratio_defined(1e-3, 1e-320) and 1e-3 / 1e-320 == np.inf
    for realized, predicted in ((0.0, 1.0), (-1.0, 1.0), (np.nan, 1.0), (np.inf, 1.0), (1.0, 0.0), (1.0, -1.0), (1.0, np.nan), (1.0, np.inf)):
        assert not _ratio_defined(realized, predicted), (realized, predicted)


def test_the_moving_rows_kernel_is_the_log_sum_exp_of_the_kernel_terms():
    # The linear-term sum (one exponential and one square root a node) against the log-sum-exp of _kernel_terms, at
    # ordinary rows and at the edges it hands to the exact form: overflowing and underflowing scales, a near-improper
    # negative precision, a shift whose h^2 c / 2 dwarfs everything, and a density with -inf nodes.
    generator = np.random.default_rng(71)
    grid = np.linspace(-9.0, 4.0, 23)
    steps = np.array([-40.0, -2.0, -0.3, 0.0, 0.7, 3.0, 40.0])
    log_density = generator.normal(size=(steps.shape[0], grid.shape[0]))
    log_density[:, :3] = -np.inf
    log_density -= engine._log_sum_exp(log_density, axis=1, keepdims=True)
    rows = 40
    log_scales = generator.normal(scale=2.0, size=rows)
    slopes = generator.normal(scale=3.0, size=rows)
    precision = np.exp(generator.normal(scale=2.0, size=rows))
    shift = generator.normal(scale=3.0, size=rows) * np.sqrt(precision)
    log_scales[:3] = [700.0, -760.0, 40.0]
    slopes[:3] = [0.0, 0.0, 30.0]
    precision[3], log_scales[3], slopes[3] = -0.9 * np.exp(-grid[-1]), 0.0, 0.0
    shift[4] = 1e4
    out = np.empty((rows, steps.shape[0]))
    assert not _moving_line_rows(log_density, log_scales, slopes, steps, grid, precision, shift, engine._LOST_BELOW, out)
    for row in range(rows):
        scales_at = log_scales[row] + slopes[row] * steps
        for position, scale in enumerate(scales_at):
            terms = _kernel_terms(log_density[position], np.array([scale]), grid, precision[row : row + 1], shift[row : row + 1])[3]
            expected = float(engine._log_sum_exp(terms, axis=1)[0])
            assert abs(out[row, position] - expected) <= 1e-13 * max(1.0, abs(expected)), (row, position, out[row, position], expected)
    improper = precision.copy()
    improper[5] = -2.0 * np.exp(-(log_scales[5] + grid[0]))
    assert _moving_line_rows(log_density, log_scales, np.zeros(rows), steps, grid, improper, shift, engine._LOST_BELOW, out)


def test_the_column_log_sums_are_the_sums_of_the_logs():
    # Running products flushed at the square root of the normal range: the sums of the logs to rounding, with factors
    # small enough to be logged alone, and the lost flag where a term is below the threshold.
    generator = np.random.default_rng(72)
    products = np.exp(generator.uniform(-20.0, 4.6, size=(5000, 17)))
    products[7, 3] = 1e-200
    products[8:40, 5] = 1e-150
    out = np.zeros(products.shape[1])
    assert not _column_log_sums(products, engine._LOST_BELOW, out)
    expected = np.log(products).sum(axis=0)
    np.testing.assert_allclose(out, expected, rtol=0.0, atol=64 * np.finfo(np.float64).eps * np.abs(np.log(products)).sum(axis=0).max())
    products[9, 2] = 1e-300
    assert _column_log_sums(products, engine._LOST_BELOW, np.zeros(products.shape[1]))


def test_the_batched_lines_are_each_lines_own_values():
    # Directions that share their fixed rows share one GEMM over all their steps; each direction's values are its own
    # pass's, with a scale design (moving rows) and without, and in pieces of a small budget.
    prior, cavity = _problem(variant_count=60, seed=51, node_count=12)
    hyperparameters = _hyperparameters(prior, 52, log_smoothing=1.0)
    generator = np.random.default_rng(73)
    directions = 0.3 * generator.standard_normal((prior.coefficient_size, 4))
    directions[prior.coefficient_size - prior.scale_size :, :2] = 0.0
    requests = {column: generator.normal(scale=2.0, size=3 + column) for column in range(4)}
    for working_bytes in (_WORKING_BYTES, 1 << 12):
        batched = _lines(prior, hyperparameters.log_smoothing, hyperparameters.coefficients, directions, cavity, working_bytes)(requests)
        for column, steps in requests.items():
            alone = _line(prior, hyperparameters.log_smoothing, hyperparameters.coefficients, directions[:, column], cavity, working_bytes)(steps)
            np.testing.assert_allclose(batched[column], alone, rtol=1e-13, atol=1e-13 * float(np.max(np.abs(alone))))


def test_the_objective_rows_linear_sum_is_the_log_form():
    # Every output of the host objective kernel's linear-term rows against the device's log-form row, at ordinary rows
    # and at the edges handed to the log form: overflowing and underflowing scales, a v P of 1e142, a
    # near-improper negative precision, a shift whose h^2 c / 2 dwarfs everything and -inf density nodes.
    from sv_pgs.scale_mixture_ep import _objective_row_exact, _objective_rows

    generator = np.random.default_rng(74)
    grid = np.linspace(-9.0, 4.0, 23)
    log_density = generator.normal(size=grid.shape[0])
    log_density[:3] = -np.inf
    log_density -= float(engine._log_sum_exp(log_density, axis=0))
    density = np.exp(log_density)
    rows = 40
    log_scales = generator.normal(scale=2.0, size=rows)
    precision = np.exp(generator.normal(scale=2.0, size=rows))
    shift = generator.normal(scale=3.0, size=rows) * np.sqrt(precision)
    log_scales[:3] = [700.0, -760.0, 300.0]
    precision[2] = 1e10
    precision[3], log_scales[3] = -0.9 * np.exp(-grid[-1]), 0.0
    shift[4] = 1e4
    outputs = [np.empty((rows, grid.shape[0] + 1)), np.empty((rows, grid.shape[0] + 1)), np.empty(rows), np.empty(rows)]
    expected = [np.empty_like(output) for output in outputs]
    assert not _objective_rows(log_density, density, grid, log_scales, precision, shift, *outputs)
    tiny = np.finfo(np.float64).tiny
    for row in range(rows):
        assert not _objective_row_exact(row, log_density, density, grid, log_scales[row], precision[row], shift[row] ** 2, tiny, *expected)
    for got, want in zip(outputs, expected):
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12 * float(np.max(np.abs(want))))


def test_a_state_is_itself_through_a_views_basis():
    # The hyper step returns x through its view's basis, x = K K'x, which moves its last bits: the outer loop's
    # zero-length test must read that as the state itself (its remainder then needs no trial), and a move of x or a
    # weight above half precision, or another edge, as a step.
    from sv_pgs.scale_mixture_ep import _same_point

    prior, _cavity = _problem(variant_count=60, seed=51, node_count=12)
    hyperparameters = _hyperparameters(prior, 52, log_smoothing=1.0)
    _view, allowed = _restricted_prior(prior, frozenset({0}))
    x = allowed @ (allowed.T @ hyperparameters.coefficients)
    state = MixtureHyperparameters(coefficients=x, log_smoothing=np.where(np.arange(hyperparameters.log_smoothing.shape[0]) == 0, np.inf, hyperparameters.log_smoothing))
    returned = replace(state, coefficients=allowed @ (allowed.T @ x))
    assert _same_point(returned, state)
    assert not _same_point(replace(state, coefficients=x + 1e-6 * (1.0 + np.max(np.abs(x)))), state)
    moved = np.array(state.log_smoothing, copy=True)
    assert np.any(np.isfinite(moved))
    moved[np.flatnonzero(np.isfinite(moved))[0]] += 1e-6
    assert not _same_point(replace(state, log_smoothing=moved), state)
    freed = np.array(state.log_smoothing, copy=True)
    freed[0] = 1.0
    assert not _same_point(replace(state, log_smoothing=freed), state)


def test_inner_trust_agreement_uses_the_mode_objective_without_laplace_offsets():
    from types import SimpleNamespace
    from sv_pgs.scale_mixture_ep import _inner_step_agreement, _next_radius
    hessian = np.array([[2., .3], [.3, 4.]])
    gradient = np.array([.2, -.1])
    step = np.array([1e-4, -1e-4])
    end_gradient = gradient - hessian @ step
    exact = float(gradient @ step - .5 * step @ hessian @ step)
    gain, rounding = _inner_step_agreement(gradient, end_gradient, step)
    assert gain == pytest.approx(exact, rel=1e-14)
    assert rounding < 1e-14 * abs(gain)
    length = float(np.linalg.norm(step))
    newton = SimpleNamespace(gradient=gradient, total=hessian)
    # A separate absolute evidence diagnostic must not freeze this accurately modelled local step.
    assert _next_radius(newton, step, length, length, gain, rounding) > length
    assert _next_radius(newton, step, length, length, gain, .02) == length


@pytest.mark.parametrize("node_count", [8, 17, 33])
@pytest.mark.parametrize("offset_group_count", [0, 3])
def test_all_infinite_constraints_use_the_exact_null_space_without_a_dense_decomposition(monkeypatch, node_count, offset_group_count):
    prior, cavity = _problem(variant_count=60, seed=13, node_count=node_count, offset_group_count=offset_group_count)
    constraints = []
    for block in prior.smoothing_blocks:
        embedded = np.zeros((block.factor.shape[0], prior.coefficient_size))
        embedded[:, block.coordinates] = block.factor
        constraints.append(embedded)
    constraint = np.vstack(constraints)
    _, singular, right = np.linalg.svd(constraint)
    rank = np.count_nonzero(singular > np.finfo(float).eps * max(constraint.shape) * singular[0])
    reference = right[rank:].T
    assert reference.shape[1] == 2

    def forbidden(*args, **kwargs):
        raise AssertionError("all-infinite restriction needs no dense SVD")

    monkeypatch.setattr(np.linalg, "svd", forbidden)
    view, allowed = _restricted_prior(prior, frozenset(range(len(prior.smoothing_blocks))))
    np.testing.assert_allclose(allowed @ allowed.T, reference @ reference.T, atol=2e-10)
    assert view.coefficient_size == 2 and not view.smoothing_blocks
    np.testing.assert_allclose(view.null_basis @ view.null_basis.T, np.eye(2), atol=1e-14)
    coefficients = np.array([0.3, -0.7])
    np.testing.assert_allclose(
        tilted_moments(view, MixtureHyperparameters(coefficients, np.zeros(0)), cavity, _WORKING_BYTES, np).log_normalizer,
        tilted_moments(prior, MixtureHyperparameters(allowed @ coefficients, np.full(len(prior.smoothing_blocks), np.inf)), cavity, _WORKING_BYTES, np).log_normalizer,
        atol=1e-13,
    )


@pytest.mark.parametrize("move", [engine.halved_lattice, engine.extended_lattice])
def test_all_infinite_transfer_agrees_with_dense_constraint_projection(monkeypatch, move):
    prior, _ = _problem(variant_count=60, seed=13, node_count=13, offset_group_count=3)
    coefficients = np.random.default_rng(23).normal(size=prior.coefficient_size)
    # First transfer unconstrained nodal functions; independently project the result onto the new constraints.
    finite = MixtureHyperparameters(coefficients, np.zeros(len(prior.smoothing_blocks)))
    moved, unconstrained = move(prior, finite)
    constraints = []
    for block in moved.smoothing_blocks:
        embedded = np.zeros((block.factor.shape[0], moved.coefficient_size))
        embedded[:, block.coordinates] = block.factor
        constraints.append(embedded)
    constraint = np.vstack(constraints)
    expected = unconstrained.coefficients - np.linalg.lstsq(constraint, constraint @ unconstrained.coefficients, rcond=None)[0]

    def forbidden(*args, **kwargs):
        raise AssertionError("all-infinite transfer needs no dense least-squares solve")

    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    _, actual = move(prior, MixtureHyperparameters(coefficients, np.full(len(prior.smoothing_blocks), np.inf)))
    np.testing.assert_allclose(actual.coefficients, expected, atol=2e-10)
    np.testing.assert_array_equal(actual.coefficients[moved.pooled_size:], 0.0)


@pytest.mark.parametrize("slope", [-0.7, 0.5])
def test_refinement_keeps_quadrature_boundaries_and_converges_to_the_same_truncated_density(slope):
    # Density remains nonzero at the finite interval's ends. Equal-weight midpoint quadrature must refine the same
    # interval; keeping the old end nodes instead moves its bounds and produces a first-order normalization drift.
    nodes = np.linspace(-8.0, -1.0, 33)
    prior = scale_mixture_prior(
        class_index=np.zeros(1, dtype=np.int64), log_variance_offset=np.zeros(1), annotation_design=np.zeros((1, 0)),
        annotation_groups=(), nodes=nodes, floor=-7.0, top=-1.0,
    )
    log_density = slope * nodes
    hyperparameters = MixtureHyperparameters(
        prior.coefficient_map.T @ (log_density - log_density.mean()), np.full(len(prior.smoothing_blocks), np.inf),
    )
    precision, shift = 50.0, 7.0
    cavity = Cavity(precision=np.array([precision]), shift=np.array([shift]))
    spacing = nodes[1] - nodes[0]
    lower, upper = nodes[0] - spacing / 2, nodes[-1] + spacing / 2

    def integrate(function):
        return quad(function, lower, upper, epsabs=1e-12, epsrel=1e-12)[0]

    def density(t):
        return np.exp(slope * t)

    def kernel(t):
        variance = np.exp(t)
        return np.exp(-0.5 * np.log1p(precision * variance) + 0.5 * shift**2 * variance / (1 + precision * variance))

    def conditional_mean(t):
        return shift * np.exp(t) / (1 + precision * np.exp(t))

    def conditional_variance(t):
        return np.exp(t) / (1 + precision * np.exp(t))

    normalizer = integrate(lambda t: density(t) * kernel(t))
    mean = integrate(lambda t: density(t) * kernel(t) * conditional_mean(t)) / normalizer
    variance = integrate(lambda t: density(t) * kernel(t) * (conditional_variance(t) + conditional_mean(t)**2)) / normalizer - mean**2
    reference = np.array([np.log(normalizer / integrate(density)), mean, variance])
    previous = None
    for level in range(4):
        grid = prior.log_variance_grid
        step = grid[1] - grid[0]
        np.testing.assert_allclose([grid[0] - step / 2, grid[-1] + step / 2], [lower, upper], rtol=0, atol=1e-13)
        assert prior.grid_size == 33 * 2**level
        got = tilted_moments(prior, hyperparameters, cavity, _WORKING_BYTES, np)
        error = np.abs(np.array([got.log_normalizer[0], got.mean[0], got.variance[0]]) - reference)
        if previous is not None:
            assert np.all(error < 0.27 * previous + 1e-13), (previous, error)
        previous = error
        prior, hyperparameters = halved_lattice(prior, hyperparameters)


@pytest.mark.parametrize("available", [True, False])
def test_derivative_chunk_reclaims_idle_memory_or_refuses_a_tiny_chunk(monkeypatch, available):
    """A full metered device with an idle cache still gives curvature a device-wide chunk. With those same bytes
    genuinely live it refuses before launching, rather than silently scheduling one kernel per variant."""
    from types import SimpleNamespace
    from sv_pgs.memory_broker import MemoryBroker, device_pool

    device = SimpleNamespace(id=0, attributes={"MultiProcessorCount": 4, "MaxThreadsPerMultiProcessor": 32})
    xp = SimpleNamespace(cuda=SimpleNamespace(Device=lambda: device))
    pool = device_pool(0)
    capacity = 1 << 20
    live = [capacity - 1]
    broker = MemoryBroker({pool: capacity}, meters={pool: lambda: live[0]})
    if available:
        def evict():
            live[0] = 0
        lease = broker.admit(pool, live[0], "old panels", evict, allocated=True)
        assert lease is not None
    monkeypatch.setattr(engine, "current_broker", lambda: broker)
    with engine.device_scope(xp):
        if available:
            chunk, budget = engine._derivative_chunk(capacity, 146, 1000)
            assert chunk >= 128 and (1000 + chunk - 1) // chunk < 10
            assert budget == capacity and not lease.live
        else:
            with pytest.raises(MemoryError, match="EB derivative chunk"):
                engine._derivative_chunk(capacity, 146, 1000)


def test_the_device_kernels_set_their_error_flag_atomically():
    """Every thread that finds an improper cavity sets its launch's one error flag; plain stores of the same value from
    many threads race in CUDA's memory model, so each is an atomic exchange (Codex EB review 14). Read from the
    kernels' source here: they run only on a device (``test_engine_kernels_cuda``)."""
    source = engine.engine_kernels._SOURCE
    assert "*improper = 1" not in source
    assert source.count("atomicExch(improper, 1)") == 4


def test_derivative_chunk_floor_respects_small_classes_and_working_budgets(monkeypatch):
    from types import SimpleNamespace
    from sv_pgs.memory_broker import MemoryBroker, device_pool

    device = SimpleNamespace(id=0, attributes={"MultiProcessorCount": 4, "MaxThreadsPerMultiProcessor": 32})
    xp = SimpleNamespace(cuda=SimpleNamespace(Device=lambda: device))
    broker = MemoryBroker({device_pool(0): 1 << 20}, meters={device_pool(0): lambda: 0})
    monkeypatch.setattr(engine, "current_broker", lambda: broker)
    row_bytes = engine.engine_kernels.derivative_row_bytes(146)
    with engine.device_scope(xp):
        assert engine._derivative_chunk(1 << 20, 146, 7)[0] == 7
        chunk, budget = engine._derivative_chunk(16 * row_bytes, 146, 1000)
        assert 1 < chunk < 128 and budget == 16 * row_bytes
        assert chunk * row_bytes < budget


@pytest.mark.parametrize("rows", [np.arange(12), np.arange(3, 9), np.arange(0, 12, 2)])
def test_density_blocks_share_contiguous_rows_and_preserve_interleaved_classes(rows):
    from types import SimpleNamespace

    values = np.arange(60, dtype=np.float64).reshape(12, 5)
    derivatives = engine._VariantDerivatives(
        mean_by_density=values, second_by_density=values.copy(),
        **{name: np.zeros(12) for name in engine._DERIVATIVE_FIELDS},
    )
    prior = SimpleNamespace(class_rows=(rows,))
    blocks = derivatives.density_blocks("mean_by_density", prior)
    np.testing.assert_array_equal(blocks[0], values[rows])
    assert blocks[0].flags.c_contiguous
    assert np.shares_memory(blocks[0], values) == bool(np.all(np.diff(rows) == 1))
    assert derivatives.density_blocks("mean_by_density", prior) is blocks


def test_checked_hyper_steps_compose_and_release_their_correction_without_cyclic_gc(monkeypatch):
    """Resolving V and tightening the weights keep each other's progress, but replacing a step leaves no recursive
    factory cycle holding its anchor, correction and derivative arrays until cyclic GC happens to run."""
    import gc
    from types import SimpleNamespace

    prior, cavity = _problem(variant_count=20, seed=91, node_count=9)
    correction = CurvatureCorrection()
    anchored = engine._anchored(prior, correction, np.zeros(prior.coefficient_map.shape[0]))
    reference = weakref.ref(correction)
    check = engine._Stationarity(
        gradient=np.zeros(1), error=np.zeros(1), curvature=np.eye(1), steps=np.ones(1), folds=np.zeros(1), gain=1.0, better=None,
    )
    evidence = SimpleNamespace(coefficients=np.zeros(prior.coefficient_size), value=2.0, error=1.0)
    refined = SimpleNamespace(coefficients=evidence.coefficients, value=3.0, error=0.25)
    monkeypatch.setattr(engine, "_stationarity", lambda *args: replace(check, gain=0.5))
    monkeypatch.setattr(engine, "_evidence", lambda *args: refined)
    monkeypatch.setattr(engine, "_corrected", lambda *args: refined)
    enabled = gc.isenabled()
    gc.disable()
    try:
        step = engine._checked_hyper_step(
            check, evidence, anchored, np.zeros(1), np.ones(1, dtype=bool), cavity, correction, _WORKING_BYTES,
            _EVIDENCE_TOLERANCE, np.eye(prior.coefficient_size), np.zeros(1),
        )
        del anchored, correction
        resolved = step.resolve(0.25)
        tightened = resolved.tighten(0.5)
        assert tightened.evidence == 3.0 and tightened.evidence_error == 0.25 and tightened.stationarity_gain == 0.5
        again = tightened.resolve(0.25)
        assert again.stationarity_gain == 0.5
        del step, resolved, tightened, again
        assert reference() is None
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize("fault", ["zero radius", "nan radius", "quadratic", "weights quadratic", "gradient", "spectrum", "proposal", "trial gradient", "boundary", "resolution"])
def test_an_inner_fault_ends_at_the_held_point_with_a_refusal(monkeypatch, fault):
    """A numeric fault returns an uncertified model with its reason. The oracle's last point is returned, including the
    density boundary where every derivative vanishes."""
    prior, cavity = _problem(variant_count=60, seed=17, node_count=12)
    start = initial_hyperparameters(prior)
    size = start.coefficients.size
    solved = []
    restored = []

    def fixed_points(hyperparameters):
        solved.append(hyperparameters[0].coefficients.copy())
        assert len(solved) <= engine.EB_STALL_TRIALS + 2
        return [FixedPoint(
            cavity=cavity, posterior=diagonal_posterior(np.ones(prior.variant_count)), mean=np.zeros(prior.variant_count),
            precision_norm=lambda direction: np.inf, effective_effects=0.0, restore=lambda: restored.append(True),
        )]

    def no_hyper_step(*arguments, **keywords):
        if fault == "weights quadratic":
            raise np.linalg.LinAlgError("the test's weights quadratic failed")
        raise FloatingPointError("no certified maximum in the test")

    def newton(view, weights, coefficients, *arguments):
        if fault == "quadratic":
            raise np.linalg.LinAlgError("the test's quadratic failed")
        return engine._NewtonB(
            view=view, allowed=np.eye(size), log_smoothing=weights, origin=coefficients,
            gradient=np.full(size, np.nan if fault == "gradient" else 1.0), total=np.eye(size),
            eigenvectors=np.eye(size), eigenvalues=np.full(size, np.nan if fault == "spectrum" else 1.0),
            definite=True, decrement=1.0,
        )

    monkeypatch.setattr(engine, "curvature_correction", lambda *arguments: INDEPENDENT_EFFECTS)
    monkeypatch.setattr(engine, "_outer_state", lambda *arguments: None)
    monkeypatch.setattr(engine, "hyper_step", no_hyper_step)
    monkeypatch.setattr(engine, "_newton_b", newton)
    monkeypatch.setattr(engine, "_penalized_gradient", lambda *arguments: np.full(size, np.nan if fault == "trial gradient" else 0.0 if fault == "boundary" else 1.0))
    if fault in ("zero radius", "nan radius"):
        monkeypatch.setattr(engine, "_cauchy_radius", lambda newton: 0.0 if fault == "zero radius" else np.nan)
    if fault in ("proposal", "resolution"):
        monkeypatch.setattr(engine, "_proposal", lambda *arguments: np.full(size, np.nan if fault == "proposal" else 0.0))
    (fit,) = fit_hyperparameters(prior, [start], fixed_points, _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    assert not fit.certified and fit.remaining_gain == np.inf and fit.prediction_move == np.inf
    assert fit.iterations == (1 if fault == "boundary" else 0)
    reason = {
        "weights quadratic": "the test's weights quadratic failed",
        "zero radius": "trust radius 0", "nan radius": "trust radius nan", "quadratic": "the test's quadratic failed",
        "gradient": "quadratic model of the evidence at x is not finite", "spectrum": "quadratic model of the evidence at x is not finite",
        "proposal": "inner step at radius", "trial gradient": "inner trial's gradient is not finite",
        "boundary": "density collapsed to the family's boundary", "resolution": "no certified progress at the EP fixed point",
    }[fault]
    assert reason in fit.refusal
    evaluated = fault in ("trial gradient", "boundary", "resolution")
    assert len(solved) == (2 if fault == "boundary" else 3 if evaluated else 1)
    assert bool(restored) == evaluated
    if fault != "boundary":
        np.testing.assert_array_equal(fit.hyperparameters.coefficients, start.coefficients)
    np.testing.assert_array_equal(solved[-1], fit.hyperparameters.coefficients)


@pytest.mark.parametrize("start_kind", ["unplanned", "planned", "trial value", "trial error"])
def test_a_walk_that_never_held_a_certified_state_ascends_until_its_budget(monkeypatch, start_kind):
    """Until a model has held a certified state, its walk outside every basin is the search for a first one, bounded by
    the no-progress budget alone: it ascends whether or not its entering trials were planned with values, and a state
    whose value is not finite or whose error is infinite is no basin. Its fixed point moves at every step (else the walk
    ends as unmoved), so it ends at its budget, at the oracle's last point."""
    prior, cavity = _problem(variant_count=60, seed=17, node_count=12)
    start = initial_hyperparameters(prior)
    size = start.coefficients.size
    solved, plans, state_calls = [], [], []

    def fixed_points(hyperparameters):
        solved.append(hyperparameters[0].coefficients.copy())
        assert len(solved) <= engine.EB_STALL_TRIALS + 2
        return [FixedPoint(
            cavity=cavity, posterior=diagonal_posterior(np.ones(prior.variant_count)), mean=np.zeros(prior.variant_count),
            precision_norm=lambda direction: np.inf, effective_effects=0.0,
        )]

    def outer_state(*arguments):
        state_calls.append(True)
        if len(state_calls) > 1 and start_kind in ("trial value", "trial error"):
            return engine._State(
                value=np.nan if start_kind == "trial value" else 0.0, rest=0.0, gradient=np.ones(size),
                fixed_curvature=np.eye(size), correction=INDEPENDENT_EFFECTS,
                error=np.inf if start_kind == "trial error" else 0.0, decrement=1.0,
            )
        return None

    def step(view, hyperparameters, *arguments, **keywords):
        plans.append(True)
        if start_kind == "unplanned" or len(plans) == 1:
            raise FloatingPointError("no evaluated hyper step yet")
        return engine.HyperStep(
            hyperparameters=hyperparameters, evidence=0.0, smoothing_gradient=0.0,
            stationarity_steps=np.zeros(0), stationarity_errors=np.zeros(0), stationarity_gain=0.0,
        )

    def newton(view, weights, coefficients, *arguments):
        return engine._NewtonB(
            view=view, allowed=np.eye(size), log_smoothing=weights, origin=coefficients, gradient=np.ones(size),
            total=np.eye(size), eigenvectors=np.eye(size), eigenvalues=np.ones(size), definite=True, decrement=1.0,
        )

    monkeypatch.setattr(engine, "EB_STALL_TRIALS", 3)
    monkeypatch.setattr(engine, "curvature_correction", lambda *arguments: INDEPENDENT_EFFECTS)
    monkeypatch.setattr(engine, "_outer_state", outer_state)
    monkeypatch.setattr(engine, "hyper_step", step)
    monkeypatch.setattr(engine, "_newton_b", newton)
    monkeypatch.setattr(engine, "_penalized_gradient", lambda *arguments: np.ones(size))
    (fit,) = fit_hyperparameters(prior, [start], fixed_points, _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    assert not fit.certified and fit.remaining_gain == np.inf and fit.prediction_move == np.inf
    assert "no certified rise of V in 3 trials" in fit.refusal
    assert "outside every basin" not in fit.refusal
    # Every inner trial was an ascent and was taken: the walk left the start.
    assert fit.iterations >= 1
    assert not np.array_equal(fit.hyperparameters.coefficients, start.coefficients)
    np.testing.assert_array_equal(solved[-1], fit.hyperparameters.coefficients)


@pytest.mark.parametrize("step", [np.nan, np.inf])
def test_a_nonfinite_correction_step_has_unmeasured_slopes_without_line_integrals(monkeypatch, step):
    from types import SimpleNamespace

    evidence = SimpleNamespace(
        replaced_directions=np.ones((1, 1)), effective_degrees=np.ones(1), penalty_sizes=np.ones(1), magnitude=1.0,
        coefficients=np.zeros(1), responses=np.zeros((1, 1)),
    )

    def no_integral(*arguments):
        pytest.fail("a non-finite difference step must not integrate a line")

    monkeypatch.setattr(engine, "_correction_value", no_integral)
    slopes, errors, second = engine._correction_slopes(
        None, np.zeros(1), evidence, np.ones(1, dtype=bool), None, _WORKING_BYTES, np.array([step]),
    )
    np.testing.assert_array_equal(slopes, np.zeros(1))
    np.testing.assert_array_equal(errors, np.full(1, np.inf))
    np.testing.assert_array_equal(second, np.zeros(1))


@pytest.mark.parametrize("step", [np.nan, np.inf])
def test_a_nonfinite_stationarity_step_ends_before_a_side_is_evaluated(monkeypatch, step):
    from types import SimpleNamespace

    evidence = SimpleNamespace(
        effective_degrees=np.ones(1), penalty_sizes=np.ones(1), magnitude=1.0, inner_decrement=step,
        coefficients=np.zeros(1), responses=np.zeros((1, 1)), gradient=np.zeros(1), value=0.0, error=0.0,
    )
    calls = []

    def at_base(view, weights, *arguments):
        calls.append(weights.copy())
        assert len(calls) == 1, "a non-finite stationarity step must not evaluate either side"
        return evidence

    monkeypatch.setattr(engine, "_evidence", at_base)
    monkeypatch.setattr(engine, "_corrected", lambda view, weights, value, *arguments: value)
    monkeypatch.setattr(engine, "_full_gradient", lambda *arguments: (np.zeros(1), np.zeros(1), np.zeros(1)))
    with pytest.raises(FloatingPointError, match="stationarity step is not finite"):
        engine._stationarity(None, np.zeros(1), evidence, np.ones(1, dtype=bool), None, INDEPENDENT_EFFECTS, _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    assert len(calls) == 1


@pytest.mark.parametrize("search", ["coefficients", "weights"])
def test_a_nan_trust_radius_ends_the_local_ascent(monkeypatch, search):
    from types import SimpleNamespace

    prior, cavity = _problem(variant_count=60, seed=17, node_count=12)
    start = initial_hyperparameters(prior)

    def no_step(*arguments, **keywords):
        pytest.fail("a NaN radius must end the local ascent before a trust step")

    monkeypatch.setattr(engine, "_trust_region_step", no_step)
    if search == "coefficients":
        objective = SimpleNamespace(rounding=0.0)
        size = start.coefficients.size
        monkeypatch.setattr(engine, "_objective_ceiling", lambda *arguments: np.inf)
        monkeypatch.setattr(engine, "_data_objective", lambda *arguments: objective)
        monkeypatch.setattr(engine, "_penalized", lambda *arguments: (0.0, np.ones(size), -np.eye(size)))
        monkeypatch.setattr(np.linalg, "norm", lambda *arguments, **keywords: np.nan)
        coefficients, returned = engine._maximize_coefficients(prior, start.log_smoothing, start.coefficients, cavity, _WORKING_BYTES, 0.0)
        np.testing.assert_array_equal(coefficients, start.coefficients)
        assert returned is objective
    else:
        evidence = SimpleNamespace(gradient=np.full(1, np.nan), magnitude=1.0)
        weights, returned = engine._ascend_evidence(
            prior, np.zeros(1), evidence, cavity, INDEPENDENT_EFFECTS, _WORKING_BYTES,
            np.full(1, -np.inf), np.full(1, np.inf), _EVIDENCE_TOLERANCE, start.coefficients,
        )
        np.testing.assert_array_equal(weights, np.zeros(1))
        assert returned is evidence


@pytest.mark.parametrize("ending", ["unmoved", "quadratic", "budget"])
def test_a_polish_that_cannot_reenter_a_basin_returns_the_state_it_left(monkeypatch, ending):
    """A polish may leave a basin where its decrement is a sliver of the error, but a walk that ends uncertified out
    there returns the state it left: an unmoved fixed point outside every basin, a numeric fault or the no-progress
    budget each restore the oracle there before returning."""
    prior, cavity = _problem(variant_count=60, seed=17, node_count=12)
    start = initial_hyperparameters(prior)
    size = start.coefficients.size
    solved = []
    state_calls, newton_calls = [], []
    state = engine._State(
        value=0.0, rest=0.0, gradient=np.ones(size), fixed_curvature=np.eye(size), correction=INDEPENDENT_EFFECTS,
        error=4.0, decrement=1.0,
    )
    step = engine.HyperStep(
        hyperparameters=start, evidence=0.0, evidence_error=0.0, smoothing_gradient=0.0,
        stationarity_steps=np.zeros(0), stationarity_errors=np.zeros(0), stationarity_gain=0.0,
    )

    def fixed_points(hyperparameters):
        solved.append(hyperparameters[0].coefficients.copy())
        assert len(solved) <= engine.EB_STALL_TRIALS + 2
        return [FixedPoint(
            cavity=cavity, posterior=diagonal_posterior(np.ones(prior.variant_count)), mean=np.zeros(prior.variant_count),
            precision_norm=(lambda direction: 0.0) if ending == "unmoved" else (lambda direction: np.inf), effective_effects=0.0,
        )]

    def outer_state(*arguments):
        state_calls.append(True)
        return state if len(state_calls) == 1 else None

    def newton(view, weights, coefficients, *arguments):
        newton_calls.append(True)
        if ending == "quadratic" and len(newton_calls) > 1:
            raise np.linalg.LinAlgError("the departed quadratic failed")
        return engine._NewtonB(
            view=view, allowed=np.eye(size), log_smoothing=weights, origin=coefficients, gradient=np.ones(size),
            total=np.eye(size), eigenvectors=np.eye(size), eigenvalues=np.ones(size), definite=True, decrement=1.0,
        )

    monkeypatch.setattr(engine, "curvature_correction", lambda *arguments: INDEPENDENT_EFFECTS)
    monkeypatch.setattr(engine, "_outer_state", outer_state)
    monkeypatch.setattr(engine, "hyper_step", lambda *arguments, **keywords: step)
    monkeypatch.setattr(engine, "_newton_b", newton)
    monkeypatch.setattr(engine, "_penalized_gradient", lambda *arguments: np.ones(size))
    if ending == "budget":
        monkeypatch.setattr(engine, "EB_STALL_TRIALS", 1)
    (fit,) = fit_hyperparameters(prior, [start], fixed_points, _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    assert not fit.certified and fit.iterations == 1 and fit.remaining_gain == np.inf
    assert len(solved) == (4 if ending == "unmoved" else 3)
    assert {"unmoved": "fixed point unmoved outside every basin", "quadratic": "the departed quadratic failed", "budget": "no certified rise"}[ending] in fit.refusal
    np.testing.assert_array_equal(fit.hyperparameters.coefficients, start.coefficients)
    np.testing.assert_array_equal(solved[-1], start.coefficients)


def test_an_indefinite_step_with_no_certified_value_at_either_end_is_refused_and_shortened(monkeypatch):
    """aou-m3f-y1 [real]: a release stood at a fixed point with no certified value, and its polish accepted six steps
    of "decrement inf indefinite, gain nan, state error inf" on the mode objective's own ratio while the radius grew
    from 0.332 to 130, until a trial's dual solve did not fit. Here a polished state's plan frees block 0, V has no
    certified value anywhere with it free, and B + S is indefinite there: every step of the release's polish is
    refused, its radius shrinks at every refusal, and the fit ends at the state the release left."""
    prior, cavity = _problem(variant_count=60, seed=17, node_count=12)
    initial = initial_hyperparameters(prior)
    blocks = len(prior.smoothing_blocks)
    start = replace(initial, log_smoothing=np.where(np.arange(blocks) == 0, np.inf, np.full(blocks, 2.0)))
    size = start.coefficients.size
    solved, lines = [], []
    state = engine._State(
        value=0.0, rest=0.0, gradient=np.ones(size), fixed_curvature=np.eye(size), correction=INDEPENDENT_EFFECTS,
        error=1e-6, decrement=0.0, polished=True,
    )

    def free(log_smoothing) -> bool:
        return bool(np.isfinite(log_smoothing[0]))

    def fixed_points(hyperparameters):
        solved.append(hyperparameters[0].coefficients.copy())
        assert len(solved) <= engine.EB_STALL_TRIALS + 4
        return [FixedPoint(
            cavity=cavity, posterior=diagonal_posterior(np.ones(prior.variant_count)), mean=np.zeros(prior.variant_count),
            precision_norm=lambda direction: np.inf, effective_effects=0.0,
        )]

    def outer_state(view, hyperparameters, *arguments):
        return None if free(hyperparameters.log_smoothing) else state

    def step(view, hyperparameters, *arguments, held_edges=frozenset(), **keywords):
        # The release of block 0, until it is held (``refused_releases``); then the state itself.
        released = 0 not in held_edges and not free(hyperparameters.log_smoothing)
        weights = np.where(np.arange(blocks) == 0, 2.0, hyperparameters.log_smoothing) if released else hyperparameters.log_smoothing
        return engine.HyperStep(
            hyperparameters=MixtureHyperparameters(coefficients=hyperparameters.coefficients, log_smoothing=weights),
            evidence=1.0 if released else 0.0, evidence_error=0.0, smoothing_gradient=0.0,
            stationarity_steps=np.zeros(0), stationarity_errors=np.zeros(0), stationarity_gain=0.0,
        )

    def newton(view, weights, coefficients, *arguments):
        if not free(weights):
            return engine._NewtonB(
                view=view, allowed=np.eye(size), log_smoothing=weights, origin=coefficients, gradient=np.zeros(size),
                total=np.eye(size), eigenvectors=np.eye(size), eigenvalues=np.ones(size), definite=True, decrement=0.0,
            )
        # At the freed weights B + S has a negative eigenvalue: the Newton decrement is infinite, the trust step's
        # predicted gain finite and positive, and the mode objective rises along it.
        eigenvalues = np.ones(size)
        eigenvalues[0] = -1.0
        return engine._NewtonB(
            view=view, allowed=np.eye(size), log_smoothing=weights, origin=coefficients, gradient=np.ones(size),
            total=np.diag(eigenvalues), eigenvectors=np.eye(size), eigenvalues=eigenvalues, definite=False, decrement=np.inf,
        )

    monkeypatch.setattr(engine, "curvature_correction", lambda *arguments: INDEPENDENT_EFFECTS)
    monkeypatch.setattr(engine, "_outer_state", outer_state)
    monkeypatch.setattr(engine, "hyper_step", step)
    monkeypatch.setattr(engine, "_newton_b", newton)
    monkeypatch.setattr(engine, "_penalized_gradient", lambda *arguments: np.ones(size))
    monkeypatch.setattr(engine, "log", lines.append)
    (fit,) = fit_hyperparameters(prior, [start], fixed_points, _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    assert any("(a release of blocks [0])" in line for line in lines), "\n".join(lines)
    accepted = [line for line in lines if "inner step accepted" in line]
    refused = [line for line in lines if "inner step refused" in line]
    assert not accepted, "\n".join(accepted)
    assert len(refused) >= 2 and all("decrement inf," in line and line.endswith("no certified value") for line in refused), "\n".join(refused)
    radii = [float(line.split("(radius ")[1].split(")")[0]) for line in refused]
    assert all(later < earlier for earlier, later in zip(radii, radii[1:])), radii
    # The release is undone: the fit stands at the state it left, block 0 at its edge.
    np.testing.assert_array_equal(fit.hyperparameters.coefficients, start.coefficients)
    np.testing.assert_array_equal(fit.hyperparameters.log_smoothing, start.log_smoothing)
    np.testing.assert_array_equal(solved[-1], start.coefficients)
