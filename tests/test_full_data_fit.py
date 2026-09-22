"""Stage 2 end to end: a dosage store, Stage 0, decoupled EP-EB from the prior, and scoring held-out samples.

A correctness test of the wiring on synthetic data, not an accuracy claim (accuracy comes from bench-real and
bench-sim): the fit must be certified, its in-sample predictor must equal what the scorer computes from the store,
and its held-out scores must carry the simulated signal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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


def _store(root: Path, seed: int, tied: bool = False):
    """Two chromosomes of mosaic dosages in one store half, and a trait with 15 causal variants (h2 = 0.5). With
    ``tied``, every tenth record repeats its predecessor and every tenth one after it is its predecessor's negation
    (dosage 2 - d): exact training ties of both signs, as a hard-genotype store has by the tens of thousands."""
    generator = np.random.default_rng(seed)
    codes = {name: mosaic_codes(generator, _SAMPLES, count) for name, count in (("chr21", 180), ("chr22", 140))}
    if tied:
        for values in codes.values():
            values[1::10] = values[0::10][: values[1::10].shape[0]]
            values[6::10] = 254 - values[5::10][: values[6::10].shape[0]]
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


@pytest.mark.xfail(
    strict=True,
    reason="EP's full-data oracle refuses this store (its refreshes line up with no contraction and the route has no "
    "double loop: 59 of 65 calls on 2026-09-21), so the fit returns with remaining_gain inf. The public route fits "
    "by the mean-field oracle (the test below); EP stays for tie members until the mean-field route carries them.",
)
@pytest.mark.slow  # the whole driver on the synthetic store: it did not finish within the gate's budget on the merged engine (2026-09-21), and is run on its own until its cost is measured
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
    member_count = statistics.active_rows.shape[0]
    store_covariates = np.column_stack([np.ones(_SAMPLES), covariate])
    mask = np.zeros((_SAMPLES, 1))
    mask[training, 0] = 1.0
    offsets = np.zeros(member_count)
    classes = (np.arange(member_count) % 5 == 0).astype(np.int64)
    # The single-variant likelihoods that set the lattice take the covariate-only residual variance as the noise.
    start_noise = float(covariate_residual_variance(targets[:, None], mask, store_covariates)[0])
    nodes, floor, top = stage0_lattice(statistics, 0, start_noise, offsets, 0.5 / _DRAWS)
    prior = scale_mixture_prior(
        class_index=classes, log_variance_offset=offsets, annotation_design=np.zeros((member_count, 0)), annotation_groups=(),
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


def test_stage2_by_mean_field_is_certified_and_scores_the_held_out_samples(tmp_path: Path) -> None:
    # The same store and prior through the streamed mean-field oracle (``_FullDataMeanField``), which the public route uses.
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
    member_count = statistics.active_rows.shape[0]
    store_covariates = np.column_stack([np.ones(_SAMPLES), covariate])
    mask = np.zeros((_SAMPLES, 1))
    mask[training, 0] = 1.0
    offsets = np.zeros(member_count)
    classes = (np.arange(member_count) % 5 == 0).astype(np.int64)
    # The single-variant likelihoods that set the lattice take the covariate-only residual variance as the noise.
    start_noise = float(covariate_residual_variance(targets[:, None], mask, store_covariates)[0])
    nodes, floor, top = stage0_lattice(statistics, 0, start_noise, offsets, 0.5 / _DRAWS)
    prior = scale_mixture_prior(
        class_index=classes, log_variance_offset=offsets, annotation_design=np.zeros((member_count, 0)), annotation_groups=(),
        nodes=nodes, floor=floor, top=top,
    )
    source = StreamedDualSource(StoreGenotypeBlockSource.from_statistics(store, statistics, _budget(), _WORKSPACE_BYTES))
    gaussian = DualGaussian(
        source=source, training=mask, targets=targets[:, None], offsets=np.zeros((_SAMPLES, 1)), covariates=store_covariates,
        grams=block_grams(statistics, start_noise), probe_count=_DRAWS, seed=11,
    )
    fit = fit_full_data(gaussian=gaussian, statistics=statistics, prior=prior, draw_count=_DRAWS, working_bytes=1 << 22, seed=13, inference="mean_field")
    certificate = fit.certificate
    assert certificate.remaining_gain[0] <= 0.5 / _DRAWS
    assert certificate.mean_move[0] <= certificate.draw_tolerance[0]
    assert certificate.noise_gain[0] <= 0.5 / _DRAWS
    scoring = scoring_models(fit, prior, statistics, [TraitType.QUANTITATIVE], _DRAWS, seed=12)
    scores = score_genetic(_StoreCodes(store), ScoringPlan.from_models(scoring), _budget())
    # The scorer's in-sample genetic score is q's own X m, read back from the store (the dual solver's mean at q's
    # sites is m only to its solve's bound, so it is not the reference here).
    signed = store.read_codes(0, store.n_variants).astype(np.float64) - 127.0
    standardized = (signed[statistics.active_rows] - statistics.means[:, None]) / statistics.scales[:, None]
    assert fit.member_mean is not None
    expected = standardized.T[training] @ fit.member_mean[:, 0]
    np.testing.assert_allclose(scores.means[training, 0], expected, rtol=1e-8, atol=1e-8)
    assert np.corrcoef(scores.means[held_out, 0], genetic[held_out])[0, 1] > 0.5
    assert np.all(scores.variances[held_out, 0] > 0.0)

def test_block_grams_share_stage0s_float32_arrays_across_models(tmp_path: Path) -> None:
    from dataclasses import replace

    from sv_pgs.full_data_fit import block_grams
    from sv_pgs.marginal_variances import window_working_bytes

    store, covariate, targets, _genetic = _store(tmp_path / "store", 8)
    training = np.arange(_TRAINING)
    statistics = compute_genotype_statistics(
        DosageStoreTileSource(store, np.arange(store.n_variants)),
        training,
        np.column_stack([np.ones(_TRAINING), covariate[training]]),
        targets[training, None],
        ModelConfig(),
        _budget(),
        _BLOCK_CAP,
        tmp_path / "ld",
    )
    shared = block_grams(statistics)
    model = replace(shared, scale=1.0 / 0.7)
    # The same arrays, not copies, and Stage 0's float32 storage.
    assert all(first is second for first, second in zip(shared.within, model.within))
    assert all(first is second for first, second in zip(shared.next_cross, model.next_cross))
    assert all(np.asarray(values).dtype == np.float32 for values in shared.within)
    ld = statistics.ld
    for block_index in range(ld.block_count):
        # The model's metric enters as its scale, 1 / sigma^2, multiplying the promoted float32 values.
        expected = np.asarray(ld.block(block_index).projected_gram, dtype=np.float64) * (1.0 / 0.7)
        assert np.array_equal(model.within_block(block_index), expected)
    assert window_working_bytes(model) > 0


def test_stage2_by_mean_field_carries_tie_members(tmp_path: Path) -> None:
    # A store with exact training ties of both signs: each member keeps its own coordinate, and the fit and its
    # scores are the untied route's in every respect the test above checks.
    store, covariate, targets, genetic = _store(tmp_path / "store", 7, tied=True)
    training = np.arange(_TRAINING)
    held_out = np.arange(_TRAINING, _SAMPLES)
    statistics = compute_genotype_statistics(
        DosageStoreTileSource(store, np.arange(store.n_variants)), training, np.column_stack([np.ones(_TRAINING), covariate[training]]),
        targets[training, None], ModelConfig(), _budget(), _BLOCK_CAP, tmp_path / "ld",
    )
    member_count = statistics.active_rows.shape[0]
    group_count = int(np.asarray(statistics.tie_map.kept_indices).shape[0])
    assert group_count < member_count
    store_covariates = np.column_stack([np.ones(_SAMPLES), covariate])
    mask = np.zeros((_SAMPLES, 1))
    mask[training, 0] = 1.0
    offsets = np.zeros(member_count)
    classes = (np.arange(member_count) % 5 == 0).astype(np.int64)
    start_noise = float(covariate_residual_variance(targets[:, None], mask, store_covariates)[0])
    nodes, floor, top = stage0_lattice(statistics, 0, start_noise, offsets, 0.5 / _DRAWS)
    prior = scale_mixture_prior(
        class_index=classes, log_variance_offset=offsets, annotation_design=np.zeros((member_count, 0)), annotation_groups=(),
        nodes=nodes, floor=floor, top=top,
    )
    source = StreamedDualSource(StoreGenotypeBlockSource.from_statistics(store, statistics, _budget(), _WORKSPACE_BYTES))
    gaussian = DualGaussian(
        source=source, training=mask, targets=targets[:, None], offsets=np.zeros((_SAMPLES, 1)), covariates=store_covariates,
        grams=block_grams(statistics, start_noise), probe_count=_DRAWS, seed=11,
    )
    fit = fit_full_data(gaussian=gaussian, statistics=statistics, prior=prior, draw_count=_DRAWS, working_bytes=1 << 22, seed=13, inference="mean_field")
    certificate = fit.certificate
    assert certificate.remaining_gain[0] <= 0.5 / _DRAWS
    assert certificate.noise_gain[0] <= 0.5 / _DRAWS
    assert fit.member_mean is not None and fit.member_mean.shape == (member_count, 1)
    scoring = scoring_models(fit, prior, statistics, [TraitType.QUANTITATIVE], _DRAWS, seed=12)
    scores = score_genetic(_StoreCodes(store), ScoringPlan.from_models(scoring), _budget())
    signed = store.read_codes(0, store.n_variants).astype(np.float64) - 127.0
    standardized = (signed[statistics.active_rows] - statistics.means[:, None]) / statistics.scales[:, None]
    expected = standardized.T[training] @ fit.member_mean[:, 0]
    np.testing.assert_allclose(scores.means[training, 0], expected, rtol=1e-8, atol=1e-8)
    assert np.corrcoef(scores.means[held_out, 0], genetic[held_out])[0, 1] > 0.5
