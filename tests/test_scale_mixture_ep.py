"""The variant-side EP-EB engine against references written independently of it.

References: quadrature of the tilted density, finite differences of the
objectives it differentiates analytically, and the REML maximum of a
Gaussian-prior regression for the noise update.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from sv_pgs.scale_mixture_ep import (
    AnnotationGroup,
    Cavity,
    GaussianPosterior,
    MixtureHyperparameters,
    _components,
    _data_objective,
    _evidence,
    _curvature_trace_gradient,
    _data_value,
    _total_curvature,
    _directional_derivatives,
    _corrected,
    _laplace_corrections,
    _log_normal_start,
    _maximize_coefficients,
    _penalized,
    _penalty_matrix,
    _penalty_value,
    _restricted_prior,
    cavities,
    class_log_density,
    derived_lattice,
    diagonal_posterior,
    halved_lattice,
    relattice,
    hyper_step,
    initial_hyperparameters,
    kernel_floor,
    kernel_top,
    log_scale,
    moment_matched_prior_sites,
    noise_variance,
    normal_means_posterior,
    prior_second_moment,
    quadrature_majorant_ratio,
    roughness_factor,
    scale_mixture_prior,
    site_targets,
    spacing_bound,
    tail_mass,
    tilted_moments,
)

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


def _problem(*, variant_count: int, seed: int, node_count: int = 0):
    """With ``node_count`` a fixed lattice whose kernels are all active; otherwise the derived floor, top and spacing."""
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
    )
    return prior, cavity


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


def test_tilted_moments_match_quadrature_of_the_mixture_times_the_cavity():
    prior, cavity = _problem(variant_count=12, seed=1, node_count=9)
    hyperparameters = _hyperparameters(prior, 2)
    moments = tilted_moments(prior, hyperparameters, cavity, _WORKING_BYTES)
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


def test_penalized_gradient_and_hessian_match_finite_differences():
    prior, cavity = _problem(variant_count=25, seed=9, node_count=8)
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
            prior.kernel_floor,
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
    coarse_density = np.exp(class_log_density(prior, hyperparameters.coefficients))
    fine_density = np.exp(class_log_density(finer, transferred.coefficients))
    # Each coarse node's mass is split between two fine nodes: the density per unit t agrees.
    np.testing.assert_allclose(fine_density[:, ::2] * 2.0, coarse_density, rtol=2e-3, atol=1e-12)
    np.testing.assert_allclose(log_scale(finer, transferred.coefficients), log_scale(prior, hyperparameters.coefficients), atol=1e-12)


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
    # Profiled: eta_bar's location and width (inside its coordinates) and the smooth annotation's
    # second-difference null space (two of its three columns); the deviations are fully penalized.
    assert prior.null_basis.shape[1] == 4
    deviations = slice(prior.pooled_size, prior.pooled_size * (prior.class_count + 1))
    np.testing.assert_allclose(prior.null_basis[deviations], 0.0, atol=1e-10)
    names = [block.name for block in prior.smoothing_blocks]
    assert names == [
        "pooled roughness", "class 0 deviation roughness", "class 1 deviation roughness",
        "deviation location and width", "annotation group 0", "annotation group 1",
    ]


def test_curvature_trace_gradient_matches_finite_differences():
    prior, cavity = _problem(variant_count=30, seed=15, node_count=12)
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


def test_evidence_gradient_in_the_log_weights_matches_finite_differences():
    prior, cavity = _problem(variant_count=60, seed=17, node_count=12)
    hyperparameters = _hyperparameters(prior, 18, log_smoothing=2.0)
    evidence = _evidence(prior, hyperparameters.log_smoothing, hyperparameters.coefficients, cavity, normal_means_posterior(cavity, _WORKING_BYTES), _WORKING_BYTES, 0.0)
    assert evidence is not None and evidence.newton_decrement < 1e-12
    step = 1e-4
    numerical = []
    for unit in np.eye(hyperparameters.log_smoothing.shape[0]):
        forward = _evidence(prior, hyperparameters.log_smoothing + step * unit, evidence.coefficients, cavity, normal_means_posterior(cavity, _WORKING_BYTES), _WORKING_BYTES, 0.0)
        backward = _evidence(prior, hyperparameters.log_smoothing - step * unit, evidence.coefficients, cavity, normal_means_posterior(cavity, _WORKING_BYTES), _WORKING_BYTES, 0.0)
        numerical.append((forward.value - backward.value) / (2.0 * step))
    np.testing.assert_allclose(evidence.gradient, np.array(numerical), rtol=1e-5, atol=1e-7)


@pytest.mark.xfail(strict=False, raises=FloatingPointError, reason=_FOLD_REASON)
def test_hyper_step_reaches_a_maximum_of_the_evidence():
    prior, cavity = _problem(variant_count=150, seed=19)
    step = hyper_step(prior, initial_hyperparameters(prior), cavity, normal_means_posterior(cavity, _WORKING_BYTES), _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    fitted = step.hyperparameters
    infinite = frozenset(int(position) for position in np.flatnonzero(fitted.log_smoothing == np.inf))
    zero = frozenset(int(position) for position in np.flatnonzero(fitted.log_smoothing == -np.inf))
    view, allowed = _restricted_prior(prior, infinite, zero)
    weights = fitted.log_smoothing[np.isfinite(fitted.log_smoothing)]
    posterior = normal_means_posterior(cavity, _WORKING_BYTES)
    laplace = _evidence(view, weights, allowed.T @ fitted.coefficients, cavity, posterior, _WORKING_BYTES, 0.0)
    assert laplace is not None and laplace.newton_decrement < 1e-10
    # V is certified to the tolerance (its Tierney-Kadane corrections), so values computed apart agree to it.
    base = _corrected(view, weights, laplace, cavity, posterior, _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    assert base is not None and abs(step.evidence - base.value) <= _EVIDENCE_TOLERANCE
    for unit in np.eye(weights.shape[0]):
        for direction in (-1.0, 1.0):
            moved_weights = weights + direction * 0.05 * unit
            moved = _corrected(
                view, moved_weights, _evidence(view, moved_weights, base.coefficients, cavity, posterior, _WORKING_BYTES, 0.0), cavity, posterior,
                _WORKING_BYTES, _EVIDENCE_TOLERANCE,
            )
            # The step certifies that a nearby move gains at most the tolerance, and each computed V is certified to it:
            # the computed difference is at most three tolerances.
            assert moved is None or moved.value <= base.value + 3.0 * _EVIDENCE_TOLERANCE
    # The B-evidence's own stationarity is certified: a Newton step on its differences gains at most the tolerance.
    assert step.stationarity_gain <= _EVIDENCE_TOLERANCE


@pytest.mark.xfail(strict=False, raises=FloatingPointError, reason=_FOLD_REASON)
def test_the_fit_does_not_depend_on_the_lattice_spacing():
    prior, cavity = _problem(variant_count=150, seed=19)
    coarse = hyper_step(prior, initial_hyperparameters(prior), cavity, normal_means_posterior(cavity, _WORKING_BYTES), _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    finer, start = halved_lattice(prior, coarse.hyperparameters)
    # At the same hyperparameters the two lattices are quadratures of one model with the same floor: the spacing bound
    # holds the trapezoid error of sum_j log Z_j to the tolerance the lattice is built to on each, so the two sums
    # agree to twice it. (Two separate fits agree only as far as V, flat to the tolerance at its maximum, determines
    # the hyperparameters, so their means are not compared.)
    coarse_moments = tilted_moments(prior, coarse.hyperparameters, cavity, _WORKING_BYTES)
    transferred_moments = tilted_moments(finer, start, cavity, _WORKING_BYTES)
    assert abs(float(np.sum(transferred_moments.log_normalizer - coarse_moments.log_normalizer))) <= 2.0 * _LATTICE_TOLERANCE
    # Refitting on the finer lattice finds no better fit: each V is a maximum certified to the tolerance.
    fine = hyper_step(finer, start, cavity, normal_means_posterior(cavity, _WORKING_BYTES), _WORKING_BYTES, _EVIDENCE_TOLERANCE)
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
        )
    # The reference maximizer is located to half of double precision in log variance.
    np.testing.assert_allclose(variance, expected, rtol=1e-6)


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


def test_tail_mass_matches_numerical_integration():
    for end_value, slope, curvature in ((0.3, -0.5, -0.2), (-1.0, 0.8, -1.5), (0.0, -2.0, 0.0)):
        numerical = quad(lambda tau: np.exp(end_value + slope * tau + 0.5 * curvature * tau * tau), 0.0, np.inf, epsrel=1e-12)[0]
        np.testing.assert_allclose(tail_mass(end_value, slope, curvature), numerical, rtol=1e-9)
    assert tail_mass(0.0, 0.1, 0.0) == np.inf and tail_mass(0.0, -1.0, 0.1) == np.inf


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
    # Outside it the log density continues along the end slopes: its second differences vanish there.
    outside = np.flatnonzero(wider > nodes[-1] + 0.2)
    np.testing.assert_allclose(np.diff(new_log_density[:, outside], n=2, axis=1), 0.0, atol=1e-8)
    assert moved.kernel_floor == -9.0 and moved.kernel_top == 4.0


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
    view, allowed = _restricted_prior(prior, frozenset({0}), frozenset())
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
    step = hyper_step(prior, initial_hyperparameters(prior), cavity, normal_means_posterior(cavity, _WORKING_BYTES), _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    view, allowed = _restricted_prior(prior, frozenset({0}), frozenset())
    start = _log_normal_start(prior, initial_hyperparameters(prior).coefficients, cavity, _WORKING_BYTES)
    posterior = normal_means_posterior(cavity, _WORKING_BYTES)
    edge = _corrected(
        view, np.zeros(0), _evidence(view, np.zeros(0), allowed.T @ start, cavity, posterior, _WORKING_BYTES, _EVIDENCE_TOLERANCE), cavity, posterior,
        _WORKING_BYTES, _EVIDENCE_TOLERANCE,
    )
    assert edge is not None
    # Both are certified V, each to the tolerance.
    assert step.evidence >= edge.value - _EVIDENCE_TOLERANCE


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


def test_quadrature_corrections_are_the_exact_integrals_along_the_standardized_directions():
    prior, cavity = _problem(variant_count=60, seed=39, node_count=12)
    hyperparameters = _hyperparameters(prior, 40, log_smoothing=2.0)
    evidence = _evidence(prior, hyperparameters.log_smoothing, hyperparameters.coefficients, cavity, normal_means_posterior(cavity, _WORKING_BYTES), _WORKING_BYTES, 0.0)
    assert evidence is not None
    corrections, terms, directions = _laplace_corrections(prior, hyperparameters.log_smoothing, evidence, cavity, normal_means_posterior(cavity, _WORKING_BYTES), _WORKING_BYTES, 0.0)
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
    original = _evidence(prior, hyperparameters.log_smoothing, hyperparameters.coefficients, cavity, normal_means_posterior(cavity, _WORKING_BYTES), _WORKING_BYTES, 0.0)
    transformed = _evidence(rotated, hyperparameters.log_smoothing, rotation.T @ hyperparameters.coefficients, cavity, normal_means_posterior(cavity, _WORKING_BYTES), _WORKING_BYTES, 0.0)
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
        solve=lambda right: covariance @ right, variance_jvp=lambda weights: -np.einsum("jk,kr,kj->jr", covariance, weights, covariance)
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


def test_total_curvature_is_the_fixed_cavity_curvature_for_independent_effects():
    prior, cavity = _problem(variant_count=40, seed=48, node_count=10)
    coefficients = _hyperparameters(prior, 49).coefficients
    moments = tilted_moments(prior, MixtureHyperparameters(coefficients, np.zeros(len(prior.smoothing_blocks))), cavity, _WORKING_BYTES)
    analytic = _total_curvature(prior, coefficients, cavity, diagonal_posterior(moments.variance), _WORKING_BYTES, 1e-13)
    mapping = prior.coefficient_map
    fixed_cavity = -(mapping.T @ _data_objective(prior, coefficients, cavity, _WORKING_BYTES).hessian @ mapping)
    np.testing.assert_allclose(analytic, fixed_cavity, rtol=1e-9, atol=1e-9 * float(np.max(np.abs(fixed_cavity))))
