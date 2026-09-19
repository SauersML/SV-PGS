"""Checks of the dense EP-EB reference (tests/ep_eb_reference.py) against first principles.

The reference is what the fast stages are gated on, so it is itself checked
where the answer is known without it:
- tilted moments against direct quadrature, and the continuous density's mixture
  integral against adaptive integration over log variance;
- EP against the exact posterior where EP is exact (an orthogonal design), including a
  brute-force double integral over the effect and the log variance;
- every derivative (gradient, Hessian, total curvature, evidence gradient, edge score)
  against finite differences;
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate
from scipy.interpolate import BSpline

from tests import ep_eb_reference as reference


def _smooth_basis(values: np.ndarray, basis_size: int) -> np.ndarray:
    """A cubic B-spline basis of a continuous annotation, on knots at its quantiles."""
    inner = np.quantile(values, np.linspace(0.0, 1.0, basis_size - 2))
    knots = np.concatenate([[inner[0]] * 3, inner, [inner[-1]] * 3])
    return BSpline.design_matrix(np.clip(values, inner[0], inner[-1]), knots, 3).toarray()


def _design(rng, variant_count: int, class_count: int = 2) -> reference.ReferenceDesign:
    log_variance_offset = np.log(rng.uniform(0.3, 1.0, size=variant_count))
    # The class centring removes the smooth's constant, so the basis is reparametrized to sum to
    # zero (Z spans the complement of the column sums). The penalty's null space is then the
    # linear effect alone, which the likelihood bounds in both directions.
    basis = _smooth_basis(rng.uniform(0.0, 1.0, size=variant_count), 5)
    _left, _singular, right = np.linalg.svd(basis.sum(axis=0)[None, :])
    absorb = right[1:].T
    smooth = basis @ absorb
    design = np.column_stack([rng.integers(0, 2, size=variant_count).astype(np.float64), smooth])
    groups = (
        reference.AnnotationGroup(columns=np.array([0]), penalty=np.eye(1)),
        reference.AnnotationGroup(
            columns=np.arange(1, 1 + smooth.shape[1]), penalty=absorb.T @ reference.second_difference_penalty(5) @ absorb
        ),
    )
    return reference.ReferenceDesign(
        class_index=rng.integers(0, class_count, size=variant_count),
        log_variance_offset=log_variance_offset,
        annotation_design=design,
        annotation_groups=groups,
    )


def _prior(rng, likelihood_precision, linear_term, degree: int = 10) -> reference.ReferencePrior:
    design = _design(rng, linear_term.shape[0])
    lower, upper = reference.initial_log_variance_range(design, likelihood_precision, linear_term)
    return reference.ReferencePrior(
        design=design,
        mixing=reference.mixing_quadrature(lower - 3.0, upper + 3.0, degree, nodes_per_panel=12),
    )


def _random_hyperparameters(rng, prior: reference.ReferencePrior) -> reference.ReferenceHyperparameters:
    # A smooth, decaying shared log density (a negative quadratic in x plus small higher terms)
    # and small class deviations from it.
    shared = rng.normal(0.0, 0.2, size=prior.degree) / np.arange(1, prior.degree + 1) ** 2
    shared[1] = -2.0
    deviations = rng.normal(0.0, 0.1, size=(prior.class_count, prior.degree)) / np.arange(1, prior.degree + 1) ** 2
    weights = np.full(reference.penalty_count(prior), 2.0)
    weights[prior.class_count + 2 :] = 3.0
    return reference.ReferenceHyperparameters(
        shared_coefficients=shared,
        deviation_coefficients=deviations,
        annotation_coefficients=rng.normal(0.0, 0.3, size=prior.feature_count),
        penalty_weights=weights,
    )


def _vector(hyperparameters: reference.ReferenceHyperparameters) -> np.ndarray:
    return reference.coefficient_vector(hyperparameters)


def _weights(hyperparameters: reference.ReferenceHyperparameters) -> np.ndarray:
    return np.array(hyperparameters.penalty_weights, dtype=np.float64)


def _components(prior, hyperparameters, variant: int):
    """The quadrature's (weight, variance) components for one variant."""
    log_scale = prior.log_variance_offset[variant] + prior.centred_design[variant] @ hyperparameters.annotation_coefficients
    weights = reference.mixing_density(prior, hyperparameters.mixing_coefficients)[prior.class_index[variant]]
    return weights, np.exp(log_scale + prior.mixing.nodes)


def _quadrature_moments(components, precision: float, shift: float):
    """Normalizer, mean and variance of Σ_k π_k N(β; 0, v_k) exp(-1/2 P β² + h β) by quadrature.

    Each component is integrated in its own coordinates, centred and scaled by
    where its integrand lives. Nothing here uses the closed form under test.
    """
    weights, variances = components
    moments = np.zeros(3)
    for weight, variance in zip(weights, variances):
        conditional_variance = 1.0 / (1.0 / variance + precision)
        centre = shift * conditional_variance
        scale = np.sqrt(conditional_variance)
        for power in range(3):

            def integrand(standard: float, power=power) -> float:
                effect = centre + scale * standard
                log_value = (
                    -0.5 * effect**2 / variance
                    - 0.5 * np.log(2.0 * np.pi * variance)
                    - 0.5 * precision * effect**2
                    + shift * effect
                )
                return effect**power * np.exp(log_value) * scale

            moments[power] += weight * integrate.quad(integrand, -40.0, 40.0, limit=200, epsabs=0.0, epsrel=1e-11)[0]
    normalizer = moments[0]
    mean = moments[1] / normalizer
    return normalizer, mean, moments[2] / normalizer - mean**2


def _ld_problem(seed: int, sample_count: int, variant_count: int, causal_count: int = 4):
    """Standardized genotypes with AR(1) LD (ρ = 0.6), a sparse effect vector, unit noise."""
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((sample_count, variant_count))
    genotypes = latent.copy()
    for column in range(1, variant_count):
        genotypes[:, column] = 0.6 * genotypes[:, column - 1] + 0.8 * latent[:, column]
    genotypes = (genotypes - genotypes.mean(axis=0)) / genotypes.std(axis=0)
    effects = np.zeros(variant_count)
    causal = rng.choice(variant_count, size=causal_count, replace=False)
    effects[causal] = rng.normal(0.0, 0.3, size=causal_count)
    phenotype = genotypes @ effects + rng.standard_normal(sample_count)
    noise_precision = 1.0 / float(np.var(phenotype - genotypes @ effects))
    return noise_precision * genotypes.T @ genotypes, noise_precision * genotypes.T @ phenotype, effects


def _cavities(rng, variant_count: int):
    return rng.uniform(20.0, 300.0, size=variant_count), rng.normal(0.0, 8.0, size=variant_count)


@pytest.mark.parametrize(("precision", "shift"), [(40.0, 3.0), (400.0, -25.0), (1e-3, 0.0)])
def test_tilted_moments_match_direct_quadrature(precision, shift) -> None:
    rng = np.random.default_rng(1)
    likelihood_precision, linear_term, _effects = _ld_problem(seed=1, sample_count=200, variant_count=20)
    prior = _prior(rng, likelihood_precision, linear_term)
    hyperparameters = _random_hyperparameters(rng, prior)
    terms = reference.tilted_terms(
        prior,
        hyperparameters.mixing_coefficients,
        hyperparameters.annotation_coefficients,
        np.full(prior.class_index.shape[0], precision),
        np.full(prior.class_index.shape[0], shift),
    )
    normalizer, mean, variance = _quadrature_moments(_components(prior, hyperparameters, 0), precision, shift)
    np.testing.assert_allclose(terms["log_normalizer"][0], np.log(normalizer), rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(terms["tilted_mean"][0], mean, rtol=1e-8, atol=1e-13)
    np.testing.assert_allclose(terms["tilted_variance"][0], variance, rtol=1e-7)


def test_quadrature_integrates_the_continuous_density() -> None:
    """Σ_k over the Gauss–Legendre nodes equals ∫ g(t) Z_j(t) dt by adaptive integration in t."""
    rng = np.random.default_rng(2)
    likelihood_precision, linear_term, _effects = _ld_problem(seed=2, sample_count=300, variant_count=12)
    prior = _prior(rng, likelihood_precision, linear_term)
    hyperparameters = _random_hyperparameters(rng, prior)
    cavity_precision, cavity_shift = _cavities(rng, 12)
    terms = reference.tilted_terms(
        prior, hyperparameters.mixing_coefficients, hyperparameters.annotation_coefficients, cavity_precision, cavity_shift
    )
    mixing = prior.mixing
    log_scale = prior.log_variance_offset + prior.centred_design @ hyperparameters.annotation_coefficients
    errors = []
    for variant in range(12):
        coefficients = hyperparameters.mixing_coefficients[prior.class_index[variant]]

        def density(log_variance: float) -> float:
            return float(np.exp(mixing.log_density_at(coefficients, np.array([log_variance]))[0]))

        def weighted(log_variance: float) -> float:
            variance = np.exp(log_scale[variant] + log_variance)
            relative = 1.0 + variance * cavity_precision[variant]
            return density(log_variance) * np.exp(-0.5 * np.log(relative) + 0.5 * cavity_shift[variant] ** 2 * variance / relative)

        normalizer = integrate.quad(density, mixing.lower, mixing.upper, limit=400, epsabs=0.0, epsrel=1e-13)[0]
        mixture = integrate.quad(weighted, mixing.lower, mixing.upper, limit=400, epsabs=0.0, epsrel=1e-13)[0]
        errors.append(abs(terms["log_normalizer"][variant] - np.log(mixture / normalizer)))
    # The certificate bounds the actual error, and at 12 nodes per panel it is below tolerance / (p + 1).
    bound = reference.quadrature_error_bound(prior, hyperparameters, cavity_precision, cavity_shift)
    assert max(errors) <= bound
    assert bound <= reference.QUADRATURE_TOLERANCE / 13


def test_orthogonal_ep_is_the_exact_posterior() -> None:
    rng = np.random.default_rng(3)
    variant_count = 24
    precisions = rng.uniform(50.0, 500.0, size=variant_count)
    linear_term = rng.normal(0.0, 10.0, size=variant_count)
    prior = _prior(rng, np.diag(precisions), linear_term)
    hyperparameters = _random_hyperparameters(rng, prior)
    vector = _vector(hyperparameters)
    sites = reference.initial_sites(prior, vector, np.diag(precisions), linear_term, np.ones(variant_count))
    np.testing.assert_allclose(sites.cavity_precision, precisions, rtol=1e-10)
    np.testing.assert_allclose(sites.cavity_shift, linear_term, rtol=1e-10, atol=1e-10)
    for variant in range(0, variant_count, 4):
        _normalizer, mean, variance = _quadrature_moments(
            _components(prior, hyperparameters, variant), precisions[variant], linear_term[variant]
        )
        np.testing.assert_allclose(sites.posterior_mean[variant], mean, rtol=1e-8, atol=1e-13)
        np.testing.assert_allclose(sites.posterior_variance[variant], variance, rtol=1e-7)
    # The cavities do not depend on the prior here, so the total curvature is the fixed-cavity
    # one and the hyperparameter step at fixed cavities is exact in one step.
    fixed_cavity = -reference.cavity_log_marginal_hessian(prior, vector, sites.cavity_precision, sites.cavity_shift)
    total = reference.total_curvature(prior, vector, sites)
    np.testing.assert_allclose(total, fixed_cavity, rtol=1e-9, atol=1e-9 * float(np.max(np.abs(fixed_cavity))))


def test_single_effect_against_brute_force_over_effect_and_log_variance() -> None:
    """For one variant EP is exact: its moments equal a double integral over (β, t)."""
    rng = np.random.default_rng(8)
    precision, shift = 120.0, 9.0
    prior = reference.ReferencePrior(
        design=reference.ReferenceDesign(
            class_index=np.zeros(1, dtype=int),
            log_variance_offset=np.zeros(1),
            annotation_design=np.zeros((1, 0)),
            annotation_groups=(),
        ),
        mixing=reference.mixing_quadrature(-np.log(precision) - 8.0, np.log((shift / precision) ** 2) + 3.0, 10, nodes_per_panel=12),
    )
    hyperparameters = _random_hyperparameters(rng, prior)
    sites = reference.initial_sites(prior, _vector(hyperparameters), np.array([[precision]]), np.array([shift]), np.ones(1))
    coefficients = hyperparameters.mixing_coefficients[0]
    mixing = prior.mixing

    def effect_integral(log_variance: float, power: int) -> float:
        """∫ β^power N(β; 0, e^t) exp(−½Pβ² + hβ) dβ, in coordinates centred where the integrand lives."""
        variance = np.exp(log_variance)
        conditional_variance = 1.0 / (1.0 / variance + precision)
        centre = shift * conditional_variance
        scale = np.sqrt(conditional_variance)

        def integrand(standard: float) -> float:
            effect = centre + scale * standard
            log_value = -0.5 * effect**2 / variance - 0.5 * np.log(2.0 * np.pi * variance) - 0.5 * precision * effect**2 + shift * effect
            return effect**power * np.exp(log_value) * scale

        return integrate.quad(integrand, -40.0, 40.0, limit=200, epsabs=0.0, epsrel=1e-12)[0]

    def moment(power: int) -> float:
        total = 0.0
        for lower, upper in zip(mixing.panel_edges[:-1], mixing.panel_edges[1:]):
            total += integrate.quad(
                lambda log_variance: float(np.exp(mixing.log_density_at(coefficients, np.array([log_variance]))[0]))
                * effect_integral(log_variance, power),
                lower,
                upper,
                limit=200,
                epsabs=0.0,
                epsrel=1e-12,
            )[0]
        return total

    normalizer = moment(0)
    mean = moment(1) / normalizer
    np.testing.assert_allclose(sites.posterior_mean[0], mean, rtol=1e-8)
    np.testing.assert_allclose(sites.posterior_variance[0], moment(2) / normalizer - mean**2, rtol=1e-7)


def test_ep_matches_posterior_marginals_to_tilted_moments_under_ld() -> None:
    rng = np.random.default_rng(9)
    likelihood_precision, linear_term, _effects = _ld_problem(seed=9, sample_count=300, variant_count=30)
    prior = _prior(rng, likelihood_precision, linear_term)
    hyperparameters = _random_hyperparameters(rng, prior)
    sites = reference.initial_sites(prior, _vector(hyperparameters), likelihood_precision, linear_term, np.ones(30))
    terms = reference.tilted_terms(
        prior, hyperparameters.mixing_coefficients, hyperparameters.annotation_coefficients, sites.cavity_precision, sites.cavity_shift
    )
    covariance = np.linalg.inv(likelihood_precision + np.diag(sites.site_precision))
    np.testing.assert_allclose(covariance @ (linear_term + sites.site_shift), terms["tilted_mean"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.diag(covariance), terms["tilted_variance"], rtol=1e-10)


def test_double_loop_reaches_the_same_ep_solution_as_newton() -> None:
    rng = np.random.default_rng(13)
    likelihood_precision, linear_term, _effects = _ld_problem(seed=13, sample_count=300, variant_count=30)
    prior = _prior(rng, likelihood_precision, linear_term)
    vector = _vector(_random_hyperparameters(rng, prior))
    start = reference.site_state(prior, vector, likelihood_precision, linear_term, np.ones(30), np.zeros(30))
    newton = reference.solve_sites(prior, vector, likelihood_precision, linear_term, start)
    double_loop = reference.double_loop_sites(prior, vector, likelihood_precision, linear_term, start)
    # The double loop stops at a loose tolerance and hands over to Newton, which lands on the same point.
    np.testing.assert_allclose(double_loop.posterior_mean, newton.posterior_mean, rtol=1e-4, atol=1e-8)
    polished = reference.solve_sites(prior, vector, likelihood_precision, linear_term, double_loop)
    np.testing.assert_allclose(polished.posterior_mean, newton.posterior_mean, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(polished.log_evidence, newton.log_evidence, rtol=0.0, atol=1e-10)


def test_cavity_log_marginal_gradient_matches_finite_differences() -> None:
    rng = np.random.default_rng(4)
    likelihood_precision, linear_term, _effects = _ld_problem(seed=4, sample_count=200, variant_count=30)
    prior = _prior(rng, likelihood_precision, linear_term)
    vector = _vector(_random_hyperparameters(rng, prior))
    cavity_precision, cavity_shift = _cavities(rng, 30)
    _value, gradient = reference.cavity_log_marginal(prior, vector, cavity_precision, cavity_shift)
    step = 1e-6
    for coordinate in range(vector.shape[0]):
        forward = vector.copy()
        backward = vector.copy()
        forward[coordinate] += step
        backward[coordinate] -= step
        difference = (
            reference.cavity_log_marginal(prior, forward, cavity_precision, cavity_shift)[0]
            - reference.cavity_log_marginal(prior, backward, cavity_precision, cavity_shift)[0]
        ) / (2.0 * step)
        np.testing.assert_allclose(gradient[coordinate], difference, rtol=1e-6, atol=1e-6)


def _central_difference(function, vector: np.ndarray, step: float) -> np.ndarray:
    columns = []
    for coordinate in range(vector.shape[0]):
        forward = vector.copy()
        backward = vector.copy()
        forward[coordinate] += step
        backward[coordinate] -= step
        columns.append((function(forward) - function(backward)) / (2.0 * step))
    return np.stack(columns, axis=-1)


def test_cavity_log_marginal_hessian_matches_finite_differences_of_the_gradient() -> None:
    rng = np.random.default_rng(7)
    likelihood_precision, linear_term, _effects = _ld_problem(seed=7, sample_count=200, variant_count=30)
    prior = _prior(rng, likelihood_precision, linear_term)
    vector = _vector(_random_hyperparameters(rng, prior))
    cavity_precision, cavity_shift = _cavities(rng, 30)
    analytic = reference.cavity_log_marginal_hessian(prior, vector, cavity_precision, cavity_shift)
    numerical = _central_difference(
        lambda candidate: reference.cavity_log_marginal(prior, candidate, cavity_precision, cavity_shift)[1], vector, 1e-5
    )
    np.testing.assert_allclose(analytic, 0.5 * (numerical + numerical.T), rtol=1e-5, atol=1e-5)


def test_total_curvature_matches_finite_differences_of_the_ep_gradient() -> None:
    """B from implicit differentiation equals −d/dx of ∇ log Z_EP with EP re-solved at each point."""
    rng = np.random.default_rng(10)
    likelihood_precision, linear_term, _effects = _ld_problem(seed=10, sample_count=200, variant_count=24)
    prior = _prior(rng, likelihood_precision, linear_term, degree=6)
    vector = _vector(_random_hyperparameters(rng, prior))
    state = reference.initial_sites(prior, vector, likelihood_precision, linear_term, np.ones(24))
    analytic = reference.total_curvature(prior, vector, state)

    def total_gradient(point):
        point_state = reference.solve_sites(prior, point, likelihood_precision, linear_term, state)
        return reference.cavity_log_marginal(prior, point, point_state.cavity_precision, point_state.cavity_shift)[1]

    numerical = -_central_difference(total_gradient, vector, 1e-5)
    np.testing.assert_allclose(analytic, 0.5 * (numerical + numerical.T), rtol=1e-5, atol=1e-6 * float(np.max(np.abs(numerical))))


def _evidence_problem(seed: int):
    rng = np.random.default_rng(seed)
    # Many causal effects of spread sizes, so neither class's density is pulled to a single variance.
    likelihood_precision, linear_term, _effects = _ld_problem(seed=seed, sample_count=400, variant_count=40, causal_count=20)
    prior = _prior(rng, likelihood_precision, linear_term, degree=8)
    hyperparameters = _random_hyperparameters(rng, prior)
    state = reference.initial_sites(prior, _vector(hyperparameters), likelihood_precision, linear_term, np.ones(40))
    return prior, hyperparameters, state, likelihood_precision, linear_term


def test_evidence_gradient_matches_finite_differences() -> None:
    """dV/dλ, which carries dx̂/dλ through D_xB, against central differences of V."""
    prior, hyperparameters, state, likelihood_precision, linear_term = _evidence_problem(11)
    weights = _weights(hyperparameters)
    point = reference.laplace_evidence(prior, weights, _vector(hyperparameters), state, likelihood_precision, linear_term)
    step = 1e-4
    for position in range(weights.shape[0]):
        values = []
        for sign in (1.0, -1.0):
            moved = weights.copy()
            moved[position] *= np.exp(sign * step)
            values.append(
                reference.laplace_evidence(
                    prior, moved, point.coefficients, point.state, likelihood_precision, linear_term, with_gradient=False
                ).value
            )
        np.testing.assert_allclose(
            weights[position] * point.weight_gradient[position], (values[0] - values[1]) / (2.0 * step), rtol=1e-5, atol=1e-7
        )


# Positions: η's roughness, class 0's deviation roughness, the deviation precision μ, the smooth annotation.
@pytest.mark.parametrize("position", [0, 1, 3, 5])
def test_evidence_is_continuous_at_infinite_weight_and_the_edge_score_is_its_slope(position) -> None:
    """V(λ) → V(∞), and −λ²·dV/dλ → dV/d(1/λ) at ∞: the boundary test is the exact one-sided derivative."""
    prior, hyperparameters, state, likelihood_precision, linear_term = _evidence_problem(12)
    weights = _weights(hyperparameters)
    at_edge = weights.copy()
    at_edge[position] = np.inf
    edge_point = reference.laplace_evidence(prior, at_edge, _vector(hyperparameters), state, likelihood_precision, linear_term)
    score = reference.edge_score(prior, at_edge, position, edge_point, likelihood_precision, linear_term)
    slopes = []
    # Large against the data's curvature on the term, small enough that log|B + S_λ| keeps its digits.
    for large in (2e3, 4e3):
        near = weights.copy()
        near[position] = large
        near_point = reference.laplace_evidence(prior, near, edge_point.coefficients, edge_point.state, likelihood_precision, linear_term)
        assert abs(near_point.value - edge_point.value) < 1e-2
        slopes.append(-(large**2) * near_point.weight_gradient[position])
    # Richardson: the O(1/λ) term of −λ² dV/dλ cancels in 2·slope(2λ) − slope(λ).
    np.testing.assert_allclose(2.0 * slopes[1] - slopes[0], score.derivative, rtol=1e-3, atol=2e-6)


def test_initial_range_spans_the_resolvable_window() -> None:
    likelihood_precision, linear_term, _effects = _ld_problem(seed=6, sample_count=300, variant_count=30)
    rng = np.random.default_rng(6)
    design = _design(rng, 30)
    lower, upper = reference.initial_log_variance_range(design, likelihood_precision, linear_term)
    scale = np.exp(design.log_variance_offset)
    assert np.exp(lower) <= float(np.min(1.0 / np.diag(likelihood_precision))) / float(np.max(scale)) * (1.0 + 1e-12)
    assert np.exp(upper) >= float(np.max(np.square(linear_term / np.diag(likelihood_precision)))) / float(np.min(scale)) * (1.0 - 1e-12)
