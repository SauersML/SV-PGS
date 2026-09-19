"""The variant-side EP-EB engine against references written independently of it.

References: quadrature of the tilted density, finite differences of the
objectives it differentiates analytically, and the REML maximum of a
Gaussian-prior regression for the noise update.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from sv_pgs.scale_mixture_ep import (
    AnnotationGroup,
    Cavity,
    MixtureHyperparameters,
    _components,
    _data_objective,
    _evidence,
    _curvature_trace_gradient,
    _penalized,
    _penalty_matrix,
    _range_functionals,
    cavities,
    class_log_density,
    halved_lattice,
    hyper_step,
    initial_hyperparameters,
    kernel_floor,
    kernel_top,
    log_scale,
    moment_matched_prior_sites,
    noise_variance,
    prior_second_moment,
    quadrature_majorant_ratio,
    roughness_penalty,
    scale_mixture_prior,
    site_targets,
    spacing_bound,
    tail_mass,
    tilted_moments,
)

_WORKING_BYTES = 1 << 20


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
        AnnotationGroup(columns=np.array([1, 2, 3]), penalty=_second_difference(3) + 1e-3 * np.eye(3)),
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
        floor = kernel_floor(cavity.precision, cavity.shift, offset, 1e-3)
        top = kernel_top(cavity.precision, cavity.shift, offset, floor)
        spacing = spacing_bound(float(variant_count), 1e-3)
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


def _hyperparameters(prior, seed: int, log_smoothing: float | None = None) -> MixtureHyperparameters:
    """The start density with a random perturbation of every coefficient, and random or given penalty weights."""
    generator = np.random.default_rng(seed)
    start = initial_hyperparameters(prior).coefficients
    coefficients = start + 0.3 * generator.standard_normal(prior.coefficient_size)
    count = len(prior.smoothing_blocks)
    weights = generator.uniform(-1.0, 2.0, count) if log_smoothing is None else np.full(count, log_smoothing)
    return MixtureHyperparameters(coefficients=coefficients, log_smoothing=weights)


_LAPLACE_NEAR_BOUNDARY = (
    "The Laplace evidence spikes where the observed -H goes near-singular along a class density's collapse "
    "direction (its boundary model is a Gaussian effect prior); the lambda step waits for the boundary-model "
    "recipe from math-density and oracle."
)


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


def test_roughness_penalty_is_the_lattice_integral_of_the_squared_third_derivative():
    spacing = 0.05
    nodes = -2.0 + spacing * np.arange(101)
    penalty = roughness_penalty(nodes.shape[0], spacing)
    for quadratic in (np.ones_like(nodes), nodes, nodes**2):
        np.testing.assert_allclose(penalty @ quadratic, 0.0, atol=1e-6)
    # The third difference of t^3 is 6 h^3 exactly, so the sum is 36 h (K - 3).
    np.testing.assert_allclose(nodes**3 @ penalty @ nodes**3, 36.0 * spacing * (nodes.shape[0] - 3), rtol=1e-6)
    # The K - 3 third differences are midpoint cells centred on t_1.5 .. t_(K-2.5): together [t_1, t_(K-2)].
    start, stop = nodes[1], nodes[-2]
    exact = 0.5 * (stop - start) + 0.25 * (np.sin(2.0 * stop) - np.sin(2.0 * start))
    np.testing.assert_allclose(np.sin(nodes) @ penalty @ np.sin(nodes), exact, rtol=2e-3)


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
    # The profiled null space is eta_bar's location and width: two directions, both inside eta_bar's coordinates.
    assert prior.null_basis.shape[1] == 2
    np.testing.assert_allclose(prior.null_basis[prior.pooled_size :], 0.0, atol=1e-10)
    location, width = _range_functionals(prior.log_variance_grid, prior.kernel_floor, prior.kernel_top)
    quadratic = basis @ prior.null_basis[: prior.pooled_size]
    assert np.linalg.matrix_rank(np.vstack([location, width]) @ quadratic) == 2
    names = [block.name for block in prior.smoothing_blocks]
    assert names == ["pooled roughness", "class 0 deviation roughness", "class 1 deviation roughness", "deviation location and width", "annotation group 0", "annotation group 1"]


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
    evidence = _evidence(prior, hyperparameters.log_smoothing, hyperparameters.coefficients, cavity, _WORKING_BYTES)
    assert evidence is not None and evidence.newton_decrement < 1e-12
    step = 1e-4
    numerical = []
    for unit in np.eye(hyperparameters.log_smoothing.shape[0]):
        forward = _evidence(prior, hyperparameters.log_smoothing + step * unit, evidence.coefficients, cavity, _WORKING_BYTES)
        backward = _evidence(prior, hyperparameters.log_smoothing - step * unit, evidence.coefficients, cavity, _WORKING_BYTES)
        numerical.append((forward.value - backward.value) / (2.0 * step))
    np.testing.assert_allclose(evidence.gradient, np.array(numerical), rtol=1e-5, atol=1e-7)


@pytest.mark.xfail(run=False, reason=_LAPLACE_NEAR_BOUNDARY)
def test_hyper_step_reaches_a_maximum_of_the_evidence():
    prior, cavity = _problem(variant_count=150, seed=19)
    step = hyper_step(prior, initial_hyperparameters(prior), cavity, _WORKING_BYTES)
    assert step.newton_decrement < 1e-10
    assert step.smoothing_gradient < 1e-5
    fitted = step.hyperparameters
    for unit in np.eye(fitted.log_smoothing.shape[0]):
        for direction in (-1.0, 1.0):
            moved = _evidence(prior, fitted.log_smoothing + direction * 0.05 * unit, fitted.coefficients, cavity, _WORKING_BYTES)
            assert moved is None or moved.value <= step.evidence + 1e-9


@pytest.mark.xfail(run=False, reason=_LAPLACE_NEAR_BOUNDARY)
def test_the_fit_does_not_depend_on_the_lattice_spacing():
    prior, cavity = _problem(variant_count=150, seed=19)
    coarse = hyper_step(prior, initial_hyperparameters(prior), cavity, _WORKING_BYTES)
    finer, start = halved_lattice(prior, coarse.hyperparameters)
    fine = hyper_step(finer, start, cavity, _WORKING_BYTES)
    np.testing.assert_allclose(fine.evidence, coarse.evidence, atol=1e-3)
    coarse_moments = tilted_moments(prior, coarse.hyperparameters, cavity, _WORKING_BYTES)
    fine_moments = tilted_moments(finer, fine.hyperparameters, cavity, _WORKING_BYTES)
    np.testing.assert_allclose(fine_moments.mean, coarse_moments.mean, rtol=1e-3, atol=1e-6)


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
    np.testing.assert_allclose(variance, expected, rtol=1e-8)


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
