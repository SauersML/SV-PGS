"""Checks of the dense EP-EB reference (tests/ep_eb_reference.py) against first principles.

The reference is what the fast stages are gated on, so it is itself checked
where the answer is known without it: tilted moments against direct quadrature,
EP against the exact posterior where EP is exact (one variant, or an orthogonal
design), the hyperparameter gradient against finite differences, and the fitted
point against the fixed-point conditions.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

from tests import ep_eb_reference as reference


def _prior(rng: np.random.Generator, variant_count: int, slab_scale: float | None, class_count: int = 2):
    return reference.ReferencePrior(
        class_index=rng.integers(0, class_count, size=variant_count),
        log_variance_offset=np.log(rng.uniform(0.3, 1.0, size=variant_count)),
        annotation_design=rng.standard_normal((variant_count, 1)),
        annotation_prior_mean=np.zeros(1),
        annotation_prior_precision=np.array([4.0]),
        shape_a=0.5,
        shape_b_pooling_variance=0.25**2,
        slab_scale=slab_scale,
    )


def _hyperparameters(level: float, class_count: int = 2) -> reference.ReferenceHyperparameters:
    return reference.ReferenceHyperparameters(
        log_variance_level=level,
        annotation_coefficients=np.array([0.2]),
        shape_b=np.linspace(0.4, 0.6, class_count),
    )


def _mixture_density(prior, hyperparameters, variant: int):
    """The prior's components (mass, variance) for one variant."""
    variances, _elasticity = reference._component_variances(
        prior, reference.log_prior_variance(prior, hyperparameters)[variant : variant + 1]
    )
    masses = np.exp(reference.log_prior_masses(prior.shape_a, hyperparameters.shape_b)[prior.class_index[variant]])
    return masses, variances[0]


def _quadrature_moments(components, precision: float, shift: float):
    """Normalizer, mean and variance of Σ_k m_k N(β; 0, v_k) exp(-1/2 P β² + h β) by quadrature.

    Each component is integrated in its own coordinates, centred and scaled by
    where its integrand lives, because the variances span ~40 orders of magnitude.
    Nothing here uses the closed form under test.
    """
    masses, variances = components
    moments = np.zeros(3)
    for mass, variance in zip(masses, variances):
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

            moments[power] += mass * integrate.quad(integrand, -40.0, 40.0, limit=200, epsabs=0.0, epsrel=1e-10)[0]
    normalizer = moments[0]
    mean = moments[1] / normalizer
    return normalizer, mean, moments[2] / normalizer - mean**2


@pytest.mark.parametrize("slab_scale", [None, 2.0])
@pytest.mark.parametrize(("precision", "shift"), [(40.0, 3.0), (400.0, -25.0), (0.0, 0.0)])
def test_tilted_moments_match_direct_quadrature(slab_scale, precision, shift) -> None:
    rng = np.random.default_rng(1)
    prior = _prior(rng, 1, slab_scale)
    hyperparameters = _hyperparameters(level=-4.0)
    terms = reference.tilted_terms(prior, hyperparameters, np.array([precision]), np.array([shift]))
    normalizer, mean, variance = _quadrature_moments(_mixture_density(prior, hyperparameters, 0), precision, shift)
    # log Z_j omits the hyperparameter-free cavity normalizer, which is 1 here (the
    # cavity factor is unnormalized exp(-1/2 P β² + h β)).
    np.testing.assert_allclose(terms["log_normalizer"][0], np.log(normalizer), rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(terms["tilted_mean"][0], mean, rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(terms["tilted_variance"][0], variance, rtol=1e-6)


def test_one_variant_ep_is_the_exact_posterior() -> None:
    rng = np.random.default_rng(2)
    prior = _prior(rng, 1, None)
    hyperparameters = _hyperparameters(level=-3.0)
    likelihood_precision = np.array([[250.0]])
    linear_term = np.array([18.0])
    _sites, _shifts, posterior_mean, posterior_variance, _p, _h = reference.run_sites(
        prior, hyperparameters, likelihood_precision, linear_term, np.array([1.0]), np.array([0.0]), 0.5, 1e-14, 5000
    )
    _normalizer, mean, variance = _quadrature_moments(_mixture_density(prior, hyperparameters, 0), 250.0, 18.0)
    np.testing.assert_allclose(posterior_mean[0], mean, rtol=1e-7)
    np.testing.assert_allclose(posterior_variance[0], variance, rtol=1e-6)


def test_orthogonal_design_is_exact_per_coordinate() -> None:
    rng = np.random.default_rng(3)
    variant_count = 6
    prior = _prior(rng, variant_count, None)
    hyperparameters = _hyperparameters(level=-3.5)
    precisions = rng.uniform(50.0, 500.0, size=variant_count)
    linear_term = rng.normal(0.0, 10.0, size=variant_count)
    _sites, _shifts, posterior_mean, posterior_variance, _p, _h = reference.run_sites(
        prior, hyperparameters, np.diag(precisions), linear_term, np.ones(variant_count), np.zeros(variant_count), 0.5, 1e-14, 5000
    )
    for variant in range(variant_count):
        _normalizer, mean, variance = _quadrature_moments(
            _mixture_density(prior, hyperparameters, variant), precisions[variant], linear_term[variant]
        )
        np.testing.assert_allclose(posterior_mean[variant], mean, rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(posterior_variance[variant], variance, rtol=1e-6)


@pytest.mark.parametrize("slab_scale", [None, 2.0])
def test_hyperparameter_gradient_matches_finite_differences(slab_scale) -> None:
    rng = np.random.default_rng(4)
    variant_count = 30
    prior = _prior(rng, variant_count, slab_scale)
    vector = _hyperparameters(level=-3.0).vector()
    cavity_precision = rng.uniform(0.0, 300.0, size=variant_count)
    cavity_shift = rng.normal(0.0, 8.0, size=variant_count)
    _value, gradient = reference.hyperparameter_objective(prior, vector, cavity_precision, cavity_shift)
    step = 1e-6
    for coordinate in range(vector.shape[0]):
        forward = vector.copy()
        backward = vector.copy()
        forward[coordinate] += step
        backward[coordinate] -= step
        difference = (
            reference.hyperparameter_objective(prior, forward, cavity_precision, cavity_shift)[0]
            - reference.hyperparameter_objective(prior, backward, cavity_precision, cavity_shift)[0]
        ) / (2.0 * step)
        np.testing.assert_allclose(gradient[coordinate], difference, rtol=1e-6, atol=1e-6)


def _ld_problem(seed: int, sample_count: int, variant_count: int):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((sample_count, variant_count))
    genotypes = np.cumsum(latent, axis=1) / np.sqrt(np.arange(1, variant_count + 1))  # AR-like LD
    genotypes = (genotypes - genotypes.mean(axis=0)) / genotypes.std(axis=0)
    effects = np.zeros(variant_count)
    causal = rng.choice(variant_count, size=4, replace=False)
    effects[causal] = rng.normal(0.0, 0.3, size=4)
    phenotype = genotypes @ effects + rng.standard_normal(sample_count)
    noise_precision = 1.0 / float(np.var(phenotype - genotypes @ effects))
    return (
        noise_precision * genotypes.T @ genotypes,
        noise_precision * genotypes.T @ phenotype,
        effects,
    )


def test_fit_is_a_joint_fixed_point_and_deterministic() -> None:
    rng = np.random.default_rng(5)
    variant_count = 40
    likelihood_precision, linear_term, effects = _ld_problem(seed=5, sample_count=300, variant_count=variant_count)
    prior = _prior(rng, variant_count, None)
    start = _hyperparameters(level=-3.0)
    fit = reference.fit_reference(prior, start, likelihood_precision, linear_term)
    reference.assert_converged(fit)
    # The sites are stationary: one more undamped sweep at the fitted hyperparameters moves nothing.
    sites, shifts, _mean, _variance, cavity_precision, cavity_shift = reference.run_sites(
        prior, fit.hyperparameters, likelihood_precision, linear_term, fit.site_precision, fit.site_shift, 1.0, 0.0, 1
    )
    np.testing.assert_allclose(sites, fit.site_precision, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(shifts, fit.site_shift, rtol=1e-8, atol=1e-8)
    # The hyperparameters maximize the cavity marginal at those cavities.
    remaximized = reference.maximize_hyperparameters(prior, fit.hyperparameters.vector(), cavity_precision, cavity_shift)
    np.testing.assert_allclose(remaximized, fit.hyperparameters.vector(), rtol=0.0, atol=1e-9)
    assert fit.newton_decrement < 1e-14
    again = reference.fit_reference(prior, start, likelihood_precision, linear_term)
    np.testing.assert_array_equal(again.posterior_mean, fit.posterior_mean)
    assert np.corrcoef(fit.posterior_mean, effects)[0, 1] > 0.8


def test_slab_bounds_an_effect_the_likelihood_does_not_limit() -> None:
    # A nearly flat likelihood with a strong pull (quasi-separation in a working
    # model): without the slab the heavy tail lets the effect run with the pull.
    rng = np.random.default_rng(6)
    means = {}
    for slab_scale in (None, 2.0):
        prior = _prior(rng, 1, slab_scale)
        prior = reference.ReferencePrior(**{**prior.__dict__, "class_index": np.array([0])})
        _sites, _shifts, posterior_mean, _variance, _p, _h = reference.run_sites(
            prior, _hyperparameters(level=-2.0), np.array([[1e-3]]), np.array([0.5]), np.array([1.0]), np.array([0.0]),
            0.5, 1e-14, 5000
        )
        means[slab_scale] = float(posterior_mean[0])
    assert means[2.0] < 8.0 < means[None]
