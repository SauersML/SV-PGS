"""Stage 2 end to end: a dosage store, Stage 0, decoupled EP-EB from the prior, and scoring held-out samples.

A correctness test of the wiring on synthetic data, not an accuracy claim (accuracy comes from bench-real and
bench-sim): the fit must be certified, its in-sample predictor must equal what the scorer computes from the store,
and its held-out scores must carry the simulated signal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.dosage_store import DosageStore
from sv_pgs.dual_solve import DualGaussian, StreamedDualSource
from sv_pgs.fast_scoring import ScoringPlan, score_genetic
from sv_pgs.full_data_fit import block_grams, covariate_residual_variance, fit_full_data, scoring_models, stage0_lattice
from sv_pgs.genotype_statistics import DosageStoreTileSource, compute_genotype_statistics
from sv_pgs.scale_mixture_ep import scale_mixture_prior
from sv_pgs.store_block_source import StoreGenotypeBlockSource
from tests.stage0_support import mosaic_codes
from tests.test_dosage_store import _write_store

_SAMPLES = 500
_TRAINING = 400
_BLOCK_CAP = 64
_DRAWS = 16
_WORKSPACE_BYTES = 1 << 26


def _budget() -> ComputeBudget:
    return ComputeBudget(
        device_kind="cpu", device_ids=(), device_names=(), device_bytes=(), device_compute_capabilities=(), host_bytes=1 << 31, cpu_threads=2
    )


class _StoreCodes:
    """The store as ``fast_scoring.CodeBlockSource``."""

    def __init__(self, store: DosageStore) -> None:
        self.store = store

    @property
    def sample_count(self) -> int:
        return int(self.store.n_samples)

    def iter_code_blocks(self, variant_ranges, buffers):
        for start, stop in variant_ranges:
            yield start, stop, self.store.read_codes(start, stop)


def _store(root: Path, seed: int):
    """Two chromosomes of mosaic dosages in one store half, and a trait with 15 causal variants (h2 = 0.5)."""
    generator = np.random.default_rng(seed)
    codes = {name: mosaic_codes(generator, _SAMPLES, count) for name, count in (("chr21", 180), ("chr22", 140))}
    milli = {name: np.rint(values.astype(np.float64) / 127.0 * 1000.0).astype(np.int64) for name, values in codes.items()}
    _write_store(root, [milli])
    store = DosageStore.open(root)
    dosage = np.concatenate([store.read_codes(0, store.n_variants).astype(np.float64)]).T / 127.0
    causal = generator.choice(dosage.shape[1], size=15, replace=False)
    effects = np.zeros(dosage.shape[1])
    effects[causal] = generator.standard_normal(15)
    genetic = (dosage - dosage.mean(axis=0)) @ effects
    genetic *= np.sqrt(0.5) / np.std(genetic)
    covariate = generator.standard_normal(_SAMPLES)
    targets = 0.3 * covariate + genetic + np.sqrt(0.5) * generator.standard_normal(_SAMPLES)
    return store, covariate, targets, genetic


def test_stage2_from_the_prior_is_certified_and_scores_the_held_out_samples(tmp_path: Path) -> None:
    store, covariate, targets, genetic = _store(tmp_path / "store", 7)
    training = np.arange(_TRAINING)
    held_out = np.arange(_TRAINING, _SAMPLES)
    training_covariates = np.column_stack([np.ones(_TRAINING), covariate[training]])
    statistics = compute_genotype_statistics(
        DosageStoreTileSource(store, np.arange(store.n_variants)),
        training,
        training_covariates,
        targets[training, None],
        ModelConfig(),
        _budget(),
        _BLOCK_CAP,
        tmp_path / "ld",
    )
    reduced_count = statistics.tie_map.kept_indices.shape[0]
    store_covariates = np.column_stack([np.ones(_SAMPLES), covariate])
    mask = np.zeros((_SAMPLES, 1))
    mask[training, 0] = 1.0
    offsets = np.zeros(reduced_count)
    classes = (np.arange(reduced_count) % 5 == 0).astype(np.int64)
    # The single-variant likelihoods that set the lattice take the covariate-only residual variance as the noise.
    start_noise = float(covariate_residual_variance(targets[:, None], mask, store_covariates)[0])
    nodes, floor, top = stage0_lattice(statistics, 0, start_noise, offsets, 0.5 / _DRAWS)
    prior = scale_mixture_prior(
        class_index=classes, log_variance_offset=offsets, annotation_design=np.zeros((reduced_count, 0)), annotation_groups=(),
        nodes=nodes, floor=floor, top=top,
    )
    source = StreamedDualSource(StoreGenotypeBlockSource.from_statistics(store, statistics, _budget(), _WORKSPACE_BYTES))
    gaussian = DualGaussian(
        source=source, training=mask, targets=targets[:, None], offsets=np.zeros((_SAMPLES, 1)), covariates=store_covariates,
        grams=block_grams(statistics, start_noise), probe_count=_DRAWS, seed=11,
    )
    fit = fit_full_data(gaussian=gaussian, statistics=statistics, prior=prior, draw_count=_DRAWS, working_bytes=1 << 22, seed=13)
    certificate = fit.certificate
    assert certificate.remaining_gain[0] <= 0.5 / _DRAWS
    assert certificate.mean_move[0] <= certificate.draw_tolerance[0]
    assert certificate.noise_gain[0] <= 0.5 / _DRAWS
    scoring = scoring_models(fit, prior, statistics, [TraitType.QUANTITATIVE], _DRAWS, seed=12)
    scores = score_genetic(_StoreCodes(store), ScoringPlan.from_models(scoring), _budget())
    # The scorer's in-sample genetic score is the fitted model's own X mu, read back from the store.
    np.testing.assert_allclose(scores.means[training, 0], np.asarray(gaussian.genetic_image)[training, 0], rtol=1e-8, atol=1e-8)
    assert np.corrcoef(scores.means[held_out, 0], genetic[held_out])[0, 1] > 0.5
    assert np.all(scores.variances[held_out, 0] > 0.0)
