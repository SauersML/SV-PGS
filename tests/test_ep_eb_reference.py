"""Checks of the dense EP-EB reference (tests/ep_eb_reference.py) against first principles.

The reference is what the fast stages are gated on, so it is itself checked
where the answer is known without it:
- tilted moments against direct quadrature;
- EP against the exact posterior where EP is exact (one variant, or an
  orthogonal design);
- the penalized objective's gradient against finite differences;
- the fitted point against every fixed-point condition (sites, coefficients,
  penalty weights).
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


def _prior(rng, likelihood_precision, linear_term, class_count: int = 2) -> reference.ReferencePrior:
    variant_count = linear_term.shape[0]
    log_variance_offset = np.log(rng.uniform(0.3, 1.0, size=variant_count))
    # The class centring removes the basis's constant, so one column goes.
    smooth = _smooth_basis(rng.uniform(0.0, 1.0, size=variant_count), 5)[:, 1:]
    design = np.column_stack([rng.integers(0, 2, size=variant_count).astype(np.float64), smooth])
    groups = (
        reference.AnnotationGroup(columns=np.array([0]), penalty=np.eye(1)),
        reference.AnnotationGroup(
            columns=np.arange(1, 1 + smooth.shape[1]), penalty=reference.second_difference_penalty(smooth.shape[1])
        ),
    )
    return reference.ReferencePrior(
        class_index=rng.integers(0, class_count, size=variant_count),
        log_variance_offset=log_variance_offset,
        annotation_design=design,
        annotation_groups=groups,
        log_variance_grid=reference.data_driven_variance_grid(likelihood_precision, linear_term, log_variance_offset),
    )


def _random_hyperparameters(rng, prior: reference.ReferencePrior) -> reference.ReferenceHyperparameters:
    return reference.ReferenceHyperparameters(
        mixing_coordinates=rng.normal(0.0, 1.0, size=(prior.class_count, prior.grid_size - 1)),
        annotation_coefficients=rng.normal(0.0, 0.3, size=prior.feature_count),
        mixing_penalty=np.full((prior.class_count, 2), 2.0),
        annotation_penalty=np.full(len(prior.annotation_groups), 3.0),
    )


def _components(prior, hyperparameters, variant: int):
    """The prior's (weight, variance) components for one variant."""
    log_scale = prior.log_variance_offset[variant] + prior.centred_design[variant] @ hyperparameters.annotation_coefficients
    weights = reference.mixing_density(prior, hyperparameters.mixing_coordinates)[prior.class_index[variant]]
    return weights, np.exp(log_scale + prior.log_variance_grid)


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

            moments[power] += weight * integrate.quad(integrand, -40.0, 40.0, limit=200, epsabs=0.0, epsrel=1e-10)[0]
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


@pytest.mark.parametrize(("precision", "shift"), [(40.0, 3.0), (400.0, -25.0), (0.0, 0.0)])
def test_tilted_moments_match_direct_quadrature(precision, shift) -> None:
    rng = np.random.default_rng(1)
    likelihood_precision, linear_term, _effects = _ld_problem(seed=1, sample_count=200, variant_count=20)
    prior = _prior(rng, likelihood_precision, linear_term)
    hyperparameters = _random_hyperparameters(rng, prior)
    terms = reference.tilted_terms(
        prior,
        hyperparameters.mixing_coordinates,
        hyperparameters.annotation_coefficients,
        np.full(prior.class_index.shape[0], precision),
        np.full(prior.class_index.shape[0], shift),
    )
    normalizer, mean, variance = _quadrature_moments(_components(prior, hyperparameters, 0), precision, shift)
    np.testing.assert_allclose(terms["log_normalizer"][0], np.log(normalizer), rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(terms["tilted_mean"][0], mean, rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(terms["tilted_variance"][0], variance, rtol=1e-6)


def test_orthogonal_ep_is_the_exact_posterior() -> None:
    rng = np.random.default_rng(3)
    variant_count = 6
    precisions = rng.uniform(50.0, 500.0, size=variant_count)
    linear_term = rng.normal(0.0, 10.0, size=variant_count)
    prior = _prior(rng, np.diag(precisions), linear_term)
    hyperparameters = _random_hyperparameters(rng, prior)
    _sites, _shifts, posterior_mean, posterior_variance, _p, _h = reference.run_sites(
        prior, hyperparameters, np.diag(precisions), linear_term, np.ones(variant_count), np.zeros(variant_count), 0.5, 1e-14, 5000
    )
    for variant in range(variant_count):
        _normalizer, mean, variance = _quadrature_moments(
            _components(prior, hyperparameters, variant), precisions[variant], linear_term[variant]
        )
        np.testing.assert_allclose(posterior_mean[variant], mean, rtol=1e-6, atol=1e-12)
        # Clipped sites never take negative precision: where the exact posterior is
        # wider than the likelihood alone (a heavy prior tail pulling two ways), the
        # site precision stops at zero and only the mean is matched.
        np.testing.assert_allclose(posterior_variance[variant], min(variance, 1.0 / precisions[variant]), rtol=1e-6)
    assert np.any(1.0 / precisions > posterior_variance * (1.0 + 1e-6)), "the case where clipping is inactive is covered"


def test_penalized_objective_gradient_matches_finite_differences() -> None:
    rng = np.random.default_rng(4)
    likelihood_precision, linear_term, _effects = _ld_problem(seed=4, sample_count=200, variant_count=30)
    prior = _prior(rng, likelihood_precision, linear_term)
    hyperparameters = _random_hyperparameters(rng, prior)
    vector = reference._pack(prior, hyperparameters.mixing_coordinates, hyperparameters.annotation_coefficients)
    cavity_precision = rng.uniform(0.0, 300.0, size=linear_term.shape[0])
    cavity_shift = rng.normal(0.0, 8.0, size=linear_term.shape[0])
    _value, gradient = reference.penalized_objective(prior, hyperparameters, vector, cavity_precision, cavity_shift)
    step = 1e-6
    for coordinate in range(vector.shape[0]):
        forward = vector.copy()
        backward = vector.copy()
        forward[coordinate] += step
        backward[coordinate] -= step
        difference = (
            reference.penalized_objective(prior, hyperparameters, forward, cavity_precision, cavity_shift)[0]
            - reference.penalized_objective(prior, hyperparameters, backward, cavity_precision, cavity_shift)[0]
        ) / (2.0 * step)
        np.testing.assert_allclose(gradient[coordinate], difference, rtol=1e-6, atol=1e-6)


def test_fit_is_a_joint_fixed_point_and_deterministic() -> None:
    rng = np.random.default_rng(5)
    likelihood_precision, linear_term, effects = _ld_problem(seed=5, sample_count=400, variant_count=60, causal_count=6)
    prior = _prior(rng, likelihood_precision, linear_term)
    fit = reference.fit_reference(prior, likelihood_precision, linear_term)
    reference.assert_converged(fit)
    # Sites: one more undamped sweep at the fitted hyperparameters moves nothing.
    sites, shifts, _mean, _variance, cavity_precision, cavity_shift = reference.run_sites(
        prior, fit.hyperparameters, likelihood_precision, linear_term, fit.site_precision, fit.site_shift, 1.0, 0.0, 1
    )
    np.testing.assert_allclose(sites, fit.site_precision, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(shifts, fit.site_shift, rtol=1e-8, atol=1e-8)
    # Coefficients: they maximize the penalized cavity marginal at those cavities.
    vector = reference._pack(prior, fit.hyperparameters.mixing_coordinates, fit.hyperparameters.annotation_coefficients)
    remaximized = reference.maximize_coefficients(prior, fit.hyperparameters, vector, cavity_precision, cavity_shift)
    np.testing.assert_allclose(remaximized, vector, rtol=0.0, atol=1e-7)
    assert fit.newton_decrement < 1e-12
    # Penalty weights: the MacKay / Fellner–Schall step leaves them where they are.
    stepped = reference.update_penalties(prior, fit.hyperparameters, vector, cavity_precision, cavity_shift)
    np.testing.assert_allclose(np.log(stepped.mixing_penalty), np.log(fit.hyperparameters.mixing_penalty), atol=1e-6)
    np.testing.assert_allclose(np.log(stepped.annotation_penalty), np.log(fit.hyperparameters.annotation_penalty), atol=1e-6)
    np.testing.assert_allclose(fit.mixing_density.sum(axis=1), 1.0, rtol=1e-12)
    again = reference.fit_reference(prior, likelihood_precision, linear_term)
    np.testing.assert_array_equal(again.posterior_mean, fit.posterior_mean)
    assert np.corrcoef(fit.posterior_mean, effects)[0, 1] > 0.8


def test_grid_covers_every_resolvable_effect_and_has_no_zero_variance() -> None:
    likelihood_precision, linear_term, _effects = _ld_problem(seed=6, sample_count=300, variant_count=30)
    grid = reference.data_driven_variance_grid(likelihood_precision, linear_term, np.zeros(30))
    marginal_estimate = linear_term / np.diag(likelihood_precision)
    assert np.all(np.isfinite(grid))
    assert np.exp(grid[0]) < 0.1 * float(np.min(1.0 / np.diag(likelihood_precision)))
    assert np.exp(grid[-1]) >= float(np.max(np.square(marginal_estimate)))
    spacing = np.diff(grid)
    assert np.all(spacing <= reference.GRID_LOG_SPACING * (1.0 + 1e-12)) and np.all(spacing > 0.9 * reference.GRID_LOG_SPACING)
