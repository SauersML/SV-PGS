"""The streamed full-data route's binary trait (``full_data_fit`` with ``binary_likelihood``'s Polya-Gamma sites):
the weighted dual solve against dense algebra, the streamed fixed point's predictor moments and joint (q, xi) fixed
point, and the public fit end to end (calibrated probabilities, the deployment prevalence's offset, held-out ranking).
Synthetic data only [own-sim]: machinery checks, never accuracy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit, logit

from sv_pgs import fit_model
from sv_pgs.artifact import predict
from sv_pgs.binary_likelihood import POLYA_GAMMA_MEAN_AT_ZERO, BernoulliSites, auc
from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.dual_solve import DualGaussian, StreamedDualSource
from sv_pgs.full_data_fit import _FullDataMeanField, block_grams, moment_starts, stage0_lattice
from sv_pgs.genotype_statistics import DosageStoreTileSource, compute_genotype_statistics
from sv_pgs.scale_mixture_ep import initial_hyperparameters, scale_mixture_prior
from sv_pgs.store_block_source import StoreGenotypeBlockSource
from tests.test_full_data_fit import _BLOCK_CAP, _DRAWS, _SAMPLES, _TRAINING, _WORKSPACE_BYTES, _budget, _store


def _labels(genetic: np.ndarray, covariate: np.ndarray, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    eta = -0.8 + 2.0 * genetic / np.std(genetic) + 0.3 * covariate
    return (generator.random(genetic.shape[0]) < expit(eta)).astype(float)


def _problem(tmp_path: Path, seed: int):
    store, covariate, _targets, genetic = _store(tmp_path / "store", seed)
    labels = _labels(genetic, covariate, seed)
    training = np.arange(_TRAINING)
    mask = np.zeros(_SAMPLES, dtype=bool)
    mask[training] = True
    sites = BernoulliSites.start(labels, mask)
    training_covariates = np.column_stack([np.ones(_TRAINING), covariate[training]])
    statistics = compute_genotype_statistics(
        DosageStoreTileSource(store, np.arange(store.n_variants)), training, training_covariates, sites.response[training, None],
        ModelConfig(), _budget(), _BLOCK_CAP, tmp_path / "ld",
    )
    member_count = statistics.active_rows.shape[0]
    start_noise = 1.0 / POLYA_GAMMA_MEAN_AT_ZERO
    nodes, floor, top = stage0_lattice(statistics, 0, start_noise, np.zeros(member_count), 0.5 / _DRAWS)
    prior = scale_mixture_prior(
        class_index=np.zeros(member_count, dtype=np.int64), log_variance_offset=np.zeros(member_count),
        annotation_design=np.zeros((member_count, 0)), annotation_groups=(), nodes=nodes, floor=floor, top=top,
    )
    store_covariates = np.column_stack([np.ones(_SAMPLES), covariate])
    source = StreamedDualSource(StoreGenotypeBlockSource.from_statistics(store, statistics, _budget(), _WORKSPACE_BYTES))
    gaussian = DualGaussian(
        source=source, training=mask[:, None].astype(float), targets=sites.response[:, None], offsets=np.zeros((_SAMPLES, 1)),
        covariates=store_covariates, grams=block_grams(statistics, start_noise), probe_count=_DRAWS, seed=11,
        sample_weights=sites.weights[:, None],
    )
    signed = store.read_codes(0, store.n_variants).astype(np.float64) - 127.0
    standardized = ((signed[statistics.active_rows] - statistics.means[:, None]) / statistics.scales[:, None]).T
    return store, covariate, labels, sites, statistics, prior, gaussian, standardized, store_covariates


def test_the_weighted_dual_solve_matches_dense_algebra(tmp_path: Path) -> None:
    _store_, _covariate, _labels_, sites, _statistics, _prior, gaussian, standardized, covariates = _problem(tmp_path, 3)
    generator = np.random.default_rng(4)
    xi = np.where(sites.training, np.abs(generator.normal(scale=2.0, size=_SAMPLES)), 0.0)
    moved = BernoulliSites(labels=sites.labels, training=sites.training, xi=xi)
    weights = moved.weights
    rows = moved.training
    # The column squares in the new metric, by dense algebra: ||(I - H_W) W^1/2 x_j||^2.
    root = np.sqrt(weights[rows])
    whitened_covariates = root[:, None] * covariates[rows]
    complement = np.eye(rows.sum()) - whitened_covariates @ np.linalg.pinv(whitened_covariates)
    design = complement @ (root[:, None] * standardized[rows])
    gaussian.reweight(
        sample_weights=weights[:, None], targets=moved.response[:, None], unit_squares=np.sum(design * design, axis=0)[:, None], key=(moved.key,),
    )
    member_count = standardized.shape[1]
    precision = generator.random(member_count) + 0.5
    shift = generator.normal(size=member_count)
    certificate = gaussian.iterate(
        site_precision=precision[:, None], site_shift=shift[:, None], noise_variance=np.ones(1), error_bound=np.full(1, 1e-9), probe_residual_ratio=1e-8,
    )
    assert float(certificate.error_bound[0]) <= 1e-9
    # The joint Gaussian over (alpha, beta), alpha flat: precision [C'WC, C'WX; X'WC, X'WX + diag tau], shift
    # [C'W z; X'W z + nu], W z = kappa.
    features = np.hstack([covariates[rows], standardized[rows]])
    joint = features.T @ (weights[rows][:, None] * features)
    joint[covariates.shape[1]:, covariates.shape[1]:] += np.diag(precision)
    right = features.T @ moved.kappa[rows]
    right[covariates.shape[1]:] += shift
    expected = np.linalg.solve(joint, right)
    assert np.allclose(np.asarray(gaussian.mean)[:, 0], expected[covariates.shape[1]:], rtol=1e-6, atol=1e-7)
    assert np.allclose(np.asarray(gaussian.alpha)[:, 0], expected[: covariates.shape[1]], rtol=1e-6, atol=1e-7)


def test_the_streamed_binary_fixed_point_is_joint_and_its_moments_are_the_dense_ones(tmp_path: Path) -> None:
    _store_, _covariate, _labels_, sites, statistics, prior, gaussian, standardized, covariates = _problem(tmp_path, 5)
    moments = moment_starts(statistics, prior)
    start = initial_hyperparameters(prior, moments[0].mean_variance)
    oracle = _FullDataMeanField(gaussian, statistics, prior, _DRAWS, 1 << 22, 13, [start], np.ones(1), sites=[sites])
    (point,) = oracle([start])
    assert point is not None
    assert oracle.noise[0] == 1.0 and oracle.reweights >= 1
    final = oracle.sites[0]
    _updated, gain = final.updated(*oracle._predictor_moments(0))
    assert gain <= 0.5 / _DRAWS
    assert oracle.noise_gain[0] == pytest.approx(gain, abs=1e-12)
    rows = final.training
    weights = final.weights[rows]
    x = standardized[rows]
    c = covariates[rows]
    gram = c.T @ (weights[:, None] * c)
    effects = oracle.mean[:, 0]
    alpha = np.linalg.solve(gram, c.T @ (weights * (final.response[rows] - x @ effects)))
    oblique = x - c @ np.linalg.solve(gram, c.T @ (weights[:, None] * x))
    mean, variance = oracle._predictor_moments(0)
    assert np.allclose(mean[rows], c @ alpha + x @ effects, atol=1e-8)
    expected = (oblique**2) @ oracle.variance[:, 0] + np.einsum("ij,jk,ik->i", c, np.linalg.inv(gram), c)
    assert np.allclose(variance[rows], expected, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(oracle.covariate_coefficients(0), alpha, rtol=1e-8, atol=1e-10)


def test_the_public_binary_fit_calibrates_and_ranks(tmp_path: Path) -> None:
    store, covariate, _targets, genetic = _store(tmp_path / "store", 7)
    labels = _labels(genetic, covariate, 7)
    samples = store.n_samples
    training = np.arange(samples) < _TRAINING

    def request(prevalence):
        return fit_model.FitRequest(
            store=store, store_columns=np.arange(samples, dtype=np.int64), covariates=covariate[:, None], covariate_names=("covariate",),
            covariate_columns=np.ones((1, 1), dtype=bool), targets=np.where(training, labels, np.nan)[:, None], training=training[:, None],
            model_names=("disease/fold0",), trait_types=(TraitType.BINARY,), research_ids=tuple(f"person{index}" for index in range(samples)),
            log_variance_offset=None, budget=_budget(), work_dir=tmp_path / f"work{prevalence}", seed=3, population_prevalence=(prevalence,),
        )

    (tmp_path / "workNone").mkdir()
    model = fit_model.fit(request(None))
    assert model.scoring[0].trait_type == TraitType.BINARY
    assert model.certificate["noise_gain"][0] <= 0.5 / fit_model.DRAW_COUNT
    prediction = predict(model, store, np.arange(samples), covariate[:, None], _budget())
    chance = prediction.predictive_mean[:, 0]
    assert np.all((chance > 0.0) & (chance < 1.0))
    # Calibration in the large on the training rows, to the Monte Carlo spread of the draws' variance.
    assert float(chance[training].mean()) == pytest.approx(float(labels[training].mean()), abs=0.02)
    held_out = ~training
    assert auc(labels[held_out], chance[held_out]) > 0.5 + 0.5 * (auc(labels[held_out], genetic[held_out]) - 0.5)
    # A deployment prevalence moves only the intercept shift, by logit(K) - logit(p) (the fit is otherwise the same).
    (tmp_path / "work0.05").mkdir()
    deployed = fit_model.fit(request(0.05))
    offset = logit(0.05) - logit(float(labels[training].mean()))
    assert deployed.scoring[0].predictive_intercept_shift == pytest.approx(model.scoring[0].predictive_intercept_shift + offset, rel=1e-9, abs=1e-9)
    np.testing.assert_allclose(deployed.scoring[0].coefficients, model.scoring[0].coefficients, rtol=1e-9, atol=1e-12)
