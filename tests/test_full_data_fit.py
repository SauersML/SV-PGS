"""Stage 2's decoupled EP-EB on the full data, end to end on in-memory codes.

A correctness test of the wiring (codes -> Stage 0 -> Stage 2 from the prior -> scoring),
not an accuracy claim: accuracy evidence comes from bench-sim and bench-real.
"""
from __future__ import annotations

import numpy as np

from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.exact_polish import GaussianModel
from sv_pgs.fast_scoring import ScoringPlan, score_genetic
from sv_pgs.full_data_fit import ReducedCodeBlocks, density_lattice, fit_full_data, prior_start, scoring_models
from sv_pgs.genotype_statistics import compute_genotype_statistics
from sv_pgs.scale_mixture_ep import scale_mixture_prior
from tests.stage0_support import InMemoryTileSource, bubble_groups, mosaic_codes

_SAMPLES = 500
_BLOCK_CAP = 64


def _budget() -> ComputeBudget:
    return ComputeBudget(
        device_kind="cpu", device_ids=(), device_names=(), device_bytes=(), device_compute_capabilities=(), host_bytes=2 * 10**9, cpu_threads=2
    )


class _Codes:
    """The in-memory codes as ``full_data_fit.CodeRows`` and ``fast_scoring.CodeBlockSource``."""

    def __init__(self, codes):
        self.codes = codes

    @property
    def n_samples(self) -> int:
        return self.codes.shape[1]

    @property
    def sample_count(self) -> int:
        return self.codes.shape[1]

    def read_codes(self, start, stop, sample_indices=None, out=None):
        return self.codes[start:stop]

    def iter_code_blocks(self, variant_ranges, buffers):
        for start, stop in variant_ranges:
            yield start, stop, self.codes[start:stop]


def _simulation(seed: int):
    generator = np.random.default_rng(seed)
    per_chromosome = {"chr1": 180, "chr2": 140}
    codes = {name: mosaic_codes(generator, _SAMPLES, count) for name, count in per_chromosome.items()}
    groups = {name: bubble_groups(generator, count) for name, count in per_chromosome.items()}
    all_codes = np.concatenate(list(codes.values()))
    dosage = all_codes.T.astype(np.float64) / 127.0
    causal = generator.choice(all_codes.shape[0], size=15, replace=False)
    effects = np.zeros(all_codes.shape[0])
    effects[causal] = generator.standard_normal(15)
    genetic = (dosage - dosage.mean(axis=0)) @ effects
    genetic *= np.sqrt(0.5) / np.std(genetic)
    covariate = generator.standard_normal(_SAMPLES)
    targets = 0.3 * covariate + genetic + np.sqrt(0.5) * generator.standard_normal(_SAMPLES)
    return InMemoryTileSource(codes=codes, groups=groups), all_codes, covariate, targets, genetic


def test_reduced_code_blocks_multiply_the_standardized_reduced_columns(tmp_path):
    source, all_codes, covariate, targets, _genetic = _simulation(3)
    training = np.arange(_SAMPLES)
    covariates = np.column_stack([np.ones(_SAMPLES), covariate])
    statistics = compute_genotype_statistics(
        source, training, covariates, targets[:, None], ModelConfig(minimum_minor_allele_frequency=0.01), _budget(), _BLOCK_CAP, tmp_path / "ld"
    )
    blocks = ReducedCodeBlocks(_Codes(all_codes), statistics, np, 1 << 24)
    signed = all_codes[blocks.store_rows].astype(np.float64) - 127.0
    standardized = ((signed - blocks.means[:, None]) / blocks.scales[:, None]).T
    right = np.random.default_rng(4).standard_normal((statistics.tie_map.kept_indices.shape[0], 3))
    image = np.zeros((_SAMPLES, 3))
    for block_index, tile in blocks.iter_tiles():
        columns = blocks.block_variant_indices[block_index]
        image += tile.matmat(right[columns])
    np.testing.assert_allclose(image, standardized @ right, rtol=1e-9, atol=1e-9)


def test_stage2_from_the_prior_reaches_a_certified_fit_and_scores_held_out_samples(tmp_path):
    source, all_codes, covariate, targets, genetic = _simulation(7)
    training = np.arange(400)
    held_out = np.arange(400, _SAMPLES)
    covariates = np.column_stack([np.ones(training.shape[0]), covariate[training]])
    statistics = compute_genotype_statistics(
        source, training, covariates, targets[training, None], ModelConfig(minimum_minor_allele_frequency=0.01), _budget(), _BLOCK_CAP, tmp_path / "ld"
    )
    blocks = ReducedCodeBlocks(_Codes(all_codes), statistics, np, 1 << 24)
    reduced_count = blocks.store_rows.shape[0]
    draw_count = 16
    masks = np.zeros((1, _SAMPLES))
    masks[0, training] = 1.0
    store_covariates = np.column_stack([np.ones(_SAMPLES), covariate])
    models = [GaussianModel(TraitType.QUANTITATIVE, targets, 0, np.zeros(_SAMPLES))]
    offsets = np.zeros(reduced_count)
    residual_variance = float(np.var(targets[training] - np.polyval(np.polyfit(covariate[training], targets[training], 1), covariate[training])))
    nodes, floor, top = density_lattice(statistics, 0, residual_variance, offsets, 1.0 / (6.0 * draw_count))
    prior = scale_mixture_prior(
        class_index=(np.arange(reduced_count) % 5 == 0).astype(np.int64),
        log_variance_offset=offsets,
        annotation_design=np.zeros((reduced_count, 0)),
        annotation_groups=(),
        nodes=nodes,
        floor=floor,
        top=top,
    )
    start = prior_start(prior, models, store_covariates, masks)
    fit = fit_full_data(
        source=blocks, prior=prior, models=models, covariates=store_covariates, sample_masks=masks, start=start,
        draw_count=draw_count, working_bytes=1 << 22, seed=11,
    )
    certificate = fit.certificate
    assert certificate.newton_decrement[0] <= 0.5 / draw_count
    assert certificate.mean_move[0] <= certificate.draw_tolerance[0]
    # The noise variance is identified to O(1/sqrt(n)) around the simulated 0.5.
    assert 0.3 < fit.noise_variance[0] < 0.8
    scoring = scoring_models(fit, prior, statistics, models, draw_count)
    plan = ScoringPlan.from_models(scoring)
    scores = score_genetic(_Codes(all_codes), plan, _budget(), sample_indices=held_out)
    correlation = np.corrcoef(scores.means[:, 0], genetic[held_out])[0, 1]
    assert correlation > 0.5
    assert np.all(scores.variances[:, 0] > 0.0)
