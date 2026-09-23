"""The gene-scale route's binary trait (``small_n``, ``mean_field`` with ``binary_likelihood``'s Polya-Gamma sites): the
whitened problem against dense weighted algebra, the fixed point's predictor moments, the joint (q, xi) fixed point,
and the whole fit's calibration and recovery. Synthetic data only [own-sim]: machinery checks, never accuracy."""

import numpy as np
import pytest
from scipy.special import expit

from sv_pgs.binary_likelihood import BernoulliSites, auc, probability
from sv_pgs.config import TraitType, VariantClass
from sv_pgs.mean_field import MeanFieldFixedPoints
from sv_pgs.small_n import dense_statistics, fit_small_n, small_n_prior, small_n_start, weighted_statistics

_WORKING_BYTES = 1 << 26


def _problem(seed: int, samples: int = 300, variants: int = 40):
    rng = np.random.default_rng(seed)
    dosage = rng.binomial(2, rng.uniform(0.05, 0.5, variants), size=(samples, variants))
    effects = np.zeros(variants)
    effects[[2, 11, 25]] = [1.2, -1.0, 0.9]
    standardized = (dosage - dosage.mean(axis=0)) / np.maximum(dosage.std(axis=0), 1e-9)
    covariate = rng.standard_normal(samples)
    eta = -0.7 + standardized @ effects + 0.4 * covariate
    labels = (rng.random(samples) < expit(eta)).astype(float)
    codes = (dosage * 127).astype(np.uint8)
    classes = np.full(variants, list(VariantClass).index(VariantClass.SNV), dtype=np.uint8)
    covariates = np.column_stack([np.ones(samples), covariate])
    return codes, covariates, labels, classes, standardized, effects


def _standardized_members(statistics, codes):
    signed = codes.astype(np.float64)[:, statistics.active_rows] - 127.0
    return (signed - statistics.means[None, :]) / statistics.scales[None, :] * statistics.signs[None, :]


def test_the_whitened_problem_is_the_dense_weighted_one():
    codes, covariates, labels, _classes, _standardized, _effects = _problem(0)
    statistics = dense_statistics(codes, covariates, labels)
    rng = np.random.default_rng(1)
    sites = BernoulliSites(labels=labels, training=np.ones(labels.shape[0], dtype=bool), xi=np.abs(rng.normal(size=labels.shape[0])))
    weighted = weighted_statistics(statistics, sites)
    weights = sites.weights
    root = np.sqrt(weights)
    members = _standardized_members(statistics, codes)
    whitened_covariates = root[:, None] * covariates
    projector = whitened_covariates @ np.linalg.solve(whitened_covariates.T @ whitened_covariates, whitened_covariates.T)
    complement = np.eye(labels.shape[0]) - projector
    assert np.allclose(weighted.projected, complement @ (root[:, None] * members), atol=1e-10)
    assert np.allclose(weighted.projected_target, complement @ (root * sites.response), atol=1e-12)
    assert np.allclose(weighted.loading, covariates.T @ (weights[:, None] * members), atol=1e-9)
    assert np.allclose(weighted.covariate_pseudo_inverse, np.linalg.inv(covariates.T @ (weights[:, None] * covariates)), rtol=1e-10)


def _binary_oracle(seed: int):
    codes, covariates, labels, classes, _standardized, _effects = _problem(seed)
    statistics = dense_statistics(codes, covariates, labels)
    sites = BernoulliSites.start(labels)
    working = weighted_statistics(statistics, sites)
    prior = small_n_prior(working, classes, np.zeros(codes.shape[1]), 64)
    start, noise, _moment = small_n_start(working, prior)
    assert noise == 1.0
    oracle = MeanFieldFixedPoints(statistics, prior, noise, 64, _WORKING_BYTES, sites=sites)
    return codes, covariates, statistics, prior, start, oracle


def test_the_binary_fixed_point_is_joint_in_q_and_xi_and_its_moments_are_the_dense_ones():
    codes, covariates, statistics, _prior, start, oracle = _binary_oracle(2)
    (point,) = oracle([start])
    assert point is not None
    assert oracle.noise == 1.0
    assert oracle.profile.get("reweights", 0) >= 1
    # The returned sites are xi's own update of q to within the tolerance: the pending gain is the certificate's.
    _updated, gain = oracle.sites.updated(*oracle.predictor_moments())
    assert gain <= 0.5 / 64
    assert oracle.noise_gain == pytest.approx(gain, abs=1e-12)
    # E eta and Var eta against dense algebra in the oracle's metric: alpha given beta is (C'WC)^-1 C'W (z - X m), and
    # Var eta_i = sum_j x_ij^2 v_j (the members independent under q) + c_i'(C'WC)^-1 c_i.
    weights = oracle.sites.weights
    members = _standardized_members(statistics, codes)
    gram = covariates.T @ (weights[:, None] * covariates)
    alpha = np.linalg.solve(gram, covariates.T @ (weights * (oracle.sites.response - members @ oracle.mean)))
    oblique = members - covariates @ np.linalg.solve(gram, covariates.T @ (weights[:, None] * members))
    mean, variance = oracle.predictor_moments()
    assert np.allclose(mean, covariates @ alpha + members @ oracle.mean, atol=1e-9)
    expected = (oblique**2) @ oracle.variance + np.einsum("ij,jk,ik->i", covariates, np.linalg.inv(gram), covariates)
    assert np.allclose(variance, expected, rtol=1e-9, atol=1e-12)


def test_a_restore_puts_back_the_metric_of_its_sites():
    _codes, _covariates, _statistics, _prior, start, oracle = _binary_oracle(3)
    entry = oracle._snapshot()
    (point,) = oracle([start])
    assert point is not None
    moved = oracle.sites
    assert moved is not entry["sites"]
    oracle._restore(entry)
    assert oracle.sites is entry["sites"]
    assert np.allclose(oracle.statistics.projected_target, weighted_statistics(oracle.base_statistics, entry["sites"]).projected_target)
    point.restore()
    assert oracle.sites is moved


def test_the_binary_fit_calibrates_and_ranks():
    """[own-sim] the fitted scoring model's predictive on the training rows has the training prevalence, and ranks held-out
    samples of the same simulation."""
    codes, covariates, labels, classes, standardized, effects = _problem(4, samples=1200)
    train, test = np.arange(900), np.arange(900, 1200)
    fit = fit_small_n(
        codes=codes[train], covariates=covariates[train], target=labels[train], variant_class=classes, log_variance_offset=None,
        draw_count=64, working_bytes=_WORKING_BYTES, seed=0, inference="mean_field", trait_type=TraitType.BINARY,
    )
    scoring = fit.scoring
    assert scoring.trait_type == TraitType.BINARY
    assert fit.noise_variance == 1.0

    def predictive(rows):
        signed = codes[rows][:, scoring.store_rows].astype(np.float64) - 127.0
        columns = (signed - scoring.signed_means[None, :]) / scoring.signed_scales[None, :]
        linear = columns @ scoring.coefficients + covariates[rows] @ scoring.alpha
        deviations = columns @ (scoring.posterior_draws - scoring.coefficients[:, None]) + covariates[rows] @ (scoring.covariate_draws - scoring.alpha[:, None])
        variance = np.mean(deviations**2, axis=1) + np.einsum("ij,jk,ik->i", covariates[rows], scoring.covariate_covariance, covariates[rows])
        return probability(linear, variance, scoring.predictive_intercept_shift)

    # Calibration in the large on the training rows holds to the Monte Carlo spread of the draws' variance.
    assert float(np.mean(predictive(train))) == pytest.approx(float(labels[train].mean()), abs=0.02)
    held_out = predictive(test)
    oracle_auc = auc(labels[test], standardized[test] @ effects)
    assert auc(labels[test], held_out) > 0.5 + 0.5 * (oracle_auc - 0.5)
    assert np.all(np.isin([2, 11, 25], np.argsort(-np.abs(scoring.coefficients))[:6]))


def test_ep_refuses_a_binary_trait():
    codes, covariates, labels, classes, _standardized, _effects = _problem(5, samples=100)
    with pytest.raises(ValueError, match="mean-field"):
        fit_small_n(
            codes=codes, covariates=covariates, target=labels, variant_class=classes, log_variance_offset=None,
            draw_count=64, working_bytes=_WORKING_BYTES, seed=0, inference="ep", trait_type=TraitType.BINARY,
        )
