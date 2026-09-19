"""The learned mixing density against the fixed-shape families it nests, on exact normal means.

With an orthogonal design each effect's cavity is its own likelihood, so the engine's
hyper step at those cavities is exact empirical Bayes and every posterior below is exact.
TPB and BayesR are fitted here by exact EB, independently of the engine. The check is the
nesting property: the learned density's held-out log predictive is within the Akaike
allowance of its free coefficients of the true prior's and of each family's, whichever
family generated the truth. It checks the code and the math; it is not evidence of an
accuracy gain (the fixed families trade places with it within about a nat per replicate).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize, minimize_scalar
from scipy.special import betainc, expit, logsumexp

from sv_pgs.scale_mixture_ep import (
    Cavity,
    class_log_density,
    derived_lattice,
    hyper_step,
    initial_hyperparameters,
    normal_means_posterior,
    scale_mixture_prior,
)

_WORKING_BYTES = 1 << 24
_LOG_2PI = float(np.log(2.0 * np.pi))
_VARIANTS = 2000
_NOISE = 0.8 / 20_000
_MEAN_VARIANCE = 0.2 / 10_000
# The fixed families' own quadrature: cell masses at a spacing whose midpoint error is far below a nat.
_FAMILY_SPACING = 0.02
_BAYESR_RATIOS = np.array([1.0, 10.0, 100.0])


_BAYESR_TRUTH = (np.array([0.0, 1.0, 10.0, 100.0]) * _MEAN_VARIANCE / 0.68, np.array([0.95, 0.03, 0.015, 0.005]))
_TPB_TRUTH = (0.5, 1.5, _MEAN_VARIANCE)


def _normal_means(truth: str, seed: int):
    generator = np.random.default_rng(seed)
    if truth == "bayesr":
        variance = generator.choice(_BAYESR_TRUTH[0], size=_VARIANTS, p=_BAYESR_TRUTH[1])
    else:
        draw = generator.beta(_TPB_TRUTH[0], _TPB_TRUTH[1], size=_VARIANTS)
        variance = _TPB_TRUTH[2] * draw / (1.0 - draw)
    effect = np.sqrt(variance) * generator.standard_normal(_VARIANTS)
    estimate = effect + np.sqrt(_NOISE) * generator.standard_normal(_VARIANTS)
    replicate = effect + np.sqrt(_NOISE) * generator.standard_normal(_VARIANTS)
    return estimate, replicate


def _true_prior(truth: str, estimate):
    if truth == "bayesr":
        return _BAYESR_TRUTH[0], np.log(_BAYESR_TRUTH[1])
    centres, edges = _family_grid(estimate)
    return np.exp(centres), np.log(np.maximum(_tpb_masses(edges, *_TPB_TRUTH), 1e-300))


def _heldout_log_predictive(estimate, replicate, variances, log_weights) -> float:
    """sum_j log p(replicate_j | estimate_j) under the prior sum_k w_k N(0, v_k): exact, component by component."""
    total = _NOISE + variances[None, :]
    component = log_weights[None, :] - 0.5 * (_LOG_2PI + np.log(total) + np.square(estimate)[:, None] / total)
    responsibility = component - logsumexp(component, axis=1, keepdims=True)
    shrink = variances[None, :] / total
    mean = estimate[:, None] * shrink
    spread = _NOISE + _NOISE * shrink
    predictive = responsibility - 0.5 * (_LOG_2PI + np.log(spread) + np.square(replicate[:, None] - mean) / spread)
    return float(np.sum(logsumexp(predictive, axis=1)))


def _learned(estimate):
    """The engine's learned density at the exact cavities; nodes below the kernel floor are its effective zero."""
    precision = np.full(_VARIANTS, 1.0 / _NOISE)
    cavity = Cavity(precision=precision, shift=estimate / _NOISE)
    offset = np.zeros(_VARIANTS)
    nodes, floor, top = derived_lattice(precision, cavity.shift, offset, 1e-3)
    prior = scale_mixture_prior(
        class_index=np.zeros(_VARIANTS, dtype=np.int64),
        log_variance_offset=offset,
        annotation_design=np.zeros((_VARIANTS, 0)),
        annotation_groups=(),
        nodes=nodes,
        floor=floor,
        top=top,
    )
    step = hyper_step(prior, initial_hyperparameters(prior), cavity, normal_means_posterior(cavity, _WORKING_BYTES), _WORKING_BYTES, 1e-6)
    variances = np.where(nodes >= floor, np.exp(nodes), 0.0)
    return variances, class_log_density(prior, step.hyperparameters.coefficients)[0], nodes.shape[0]


def _family_grid(estimate):
    """Log-variance cell centres and edges spanning every variance the estimates can distinguish."""
    low = np.log(1e-4 * _NOISE)
    high = np.log(1e4 * float(np.max(np.square(estimate))))
    centres = np.arange(low, high + _FAMILY_SPACING, _FAMILY_SPACING)
    edges = np.concatenate([[-np.inf], 0.5 * (centres[1:] + centres[:-1]), [np.inf]])
    return centres, edges


def _tpb_masses(edges, shape_a, shape_b, scale):
    """Cell masses of psi = scale X / (1 - X), X ~ Beta(a, b), from the CDF in its accurate tail."""
    standardized = edges - np.log(scale)
    lower = betainc(shape_a, shape_b, expit(standardized))
    upper = betainc(shape_b, shape_a, expit(-standardized))
    below_half = expit(0.5 * (standardized[1:] + standardized[:-1])) < 0.5
    return np.maximum(np.where(below_half, np.diff(lower), -np.diff(upper)), 0.0)


def _likelihood_columns(estimate, variances):
    log_columns = -0.5 * (_LOG_2PI + np.log(_NOISE + variances[None, :]) + np.square(estimate)[:, None] / (_NOISE + variances[None, :]))
    row_max = np.max(log_columns, axis=1)
    return np.exp(log_columns - row_max[:, None]), row_max


def _tpb_fit(estimate):
    """Exact EB for TPB(a, b, scale): the marginal likelihood maximized from several shape starts."""
    centres, edges = _family_grid(estimate)
    columns, row_max = _likelihood_columns(estimate, np.exp(centres))

    def negative(parameters):
        masses = _tpb_masses(edges, np.exp(parameters[0]), np.exp(parameters[1]), np.exp(parameters[2]))
        return -float(np.sum(np.log(np.maximum(columns @ masses, 1e-300)) + row_max))

    moment = max(float(np.mean(np.square(estimate) - _NOISE)), _NOISE * 1e-3)
    bounds = [(np.log(0.02), np.log(200.0)), (np.log(0.05), np.log(200.0)), (float(centres[0]), float(centres[-1]))]
    starts = [(np.log(shape_a), np.log(shape_b), np.log(moment)) for shape_a, shape_b in ((0.5, 0.5), (0.5, 2.0), (1.0, 1.0), (5.0, 5.0))]
    best = min((minimize(negative, start, method="L-BFGS-B", bounds=bounds) for start in starts), key=lambda result: result.fun)
    masses = _tpb_masses(edges, *np.exp(best.x))
    return np.exp(centres), np.log(np.maximum(masses, 1e-300))


def _bayesr_weights(columns, row_max, weights):
    """EM for the four BayesR weights at fixed component variances (the log-likelihood is concave in them)."""
    previous = -np.inf
    value = -np.inf
    for _iteration in range(5000):
        mixture = np.maximum(columns @ weights, 1e-300)
        value = float(np.sum(np.log(mixture) + row_max))
        weights = weights * (columns.T @ (1.0 / mixture)) / columns.shape[0]
        if value - previous < 1e-12 * abs(value):
            break
        previous = value
    return weights, value


def _bayesr_fit(estimate):
    """Exact EB for BayesR: weights on {0, s, 10 s, 100 s}, with s profiled (scan, then a bounded search)."""
    start = np.array([0.9, 0.05, 0.03, 0.02])

    def profile(log_unit):
        variances = np.concatenate([[0.0], np.exp(log_unit) * _BAYESR_RATIOS])
        weights, value = _bayesr_weights(*_likelihood_columns(estimate, variances), start)
        return value, variances, weights

    scan = np.linspace(np.log(1e-3 * _NOISE), np.log(float(np.max(np.square(estimate)))), 60)
    values = [profile(log_unit)[0] for log_unit in scan]
    centre = int(np.argmax(values))
    step = float(scan[1] - scan[0])
    best = minimize_scalar(lambda log_unit: -profile(log_unit)[0], bounds=(scan[centre] - step, scan[centre] + step), method="bounded")
    _value, variances, weights = profile(float(best.x))
    return variances, np.log(np.maximum(weights, 1e-300))


@pytest.mark.parametrize("truth", ["bayesr", "tpb"])
def test_the_learned_density_is_within_its_akaike_allowance_of_every_nested_fit(truth):
    estimate, replicate = _normal_means(truth, seed=20260919)
    variances, log_weights, lattice_size = _learned(estimate)
    learned = _heldout_log_predictive(estimate, replicate, variances, log_weights)
    references = {"true prior": _true_prior(truth, estimate), "tpb": _tpb_fit(estimate), "bayesr": _bayesr_fit(estimate)}
    # Akaike: estimating d free coefficients costs about d/2 nats of expected held-out log predictive, and the
    # learned density has at most lattice_size - 1 of them.
    allowance = 0.5 * (lattice_size - 1)
    for name, reference in references.items():
        score = _heldout_log_predictive(estimate, replicate, *reference)
        assert learned >= score - allowance, f"{name}: learned {learned:.3f} vs {score:.3f}, allowance {allowance:.1f}"


@pytest.mark.xfail(
    strict=True,
    raises=FloatingPointError,
    reason="engine: hyper_step finds no certified maximum at the start weights on this TPB-truth normal-means problem",
)
def test_the_engine_certifies_a_start_on_a_tpb_truth():
    estimate, _replicate = _normal_means("tpb", seed=20260921)
    _learned(estimate)
