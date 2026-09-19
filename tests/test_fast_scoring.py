"""One-pass multi-model scoring over dosage codes (sv_pgs.fast_scoring).

References are dense fp64 products of the standardized codes, the posterior-draw
variance computed densely from the same draws, and the logistic-normal integral
by adaptive quadrature.
"""
from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit

from sv_pgs import TraitType
from sv_pgs.compute_budget import ComputeBudget, detect_compute_budget
from sv_pgs.data import TieGroup, TieMap
from sv_pgs.fast_scoring import (
    ScoringModel,
    ScoringPlan,
    posterior_predictive_probability,
    predictive_intercept_shift,
    score_genetic,
    score_linear_predictor,
)
from sv_pgs.genotype import _try_import_cupy


class InMemoryCodes:
    """Code source over a uint8 [variants, samples] matrix that records its reads."""

    def __init__(self, codes: np.ndarray) -> None:
        self.codes = np.ascontiguousarray(codes, dtype=np.uint8)
        self.reads: list[tuple[int, int]] = []

    @property
    def sample_count(self) -> int:
        return int(self.codes.shape[1])

    def iter_code_blocks(
        self,
        variant_ranges: Sequence[tuple[int, int]],
        buffers: Sequence[np.ndarray],
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        for position, (start, stop) in enumerate(variant_ranges):
            self.reads.append((start, stop))
            buffer = buffers[position % len(buffers)]
            buffer[: stop - start] = self.codes[start:stop]
            yield start, stop, buffer[: stop - start]


def cpu_budget(*, sample_count: int, block_rows: int, threads: int) -> ComputeBudget:
    # _block_rows gives host_bytes // (2 * 11 * samples) rows per read.
    return ComputeBudget(
        device_kind="cpu",
        device_ids=(),
        device_names=(),
        device_bytes=(),
        device_compute_capabilities=(),
        host_bytes=2 * 11 * sample_count * block_rows,
        cpu_threads=threads,
    )


def random_codes(random_generator: np.random.Generator, variant_count: int, sample_count: int) -> np.ndarray:
    frequencies = random_generator.uniform(0.05, 0.5, size=variant_count)
    hard_calls = random_generator.binomial(2, frequencies[:, None], size=(variant_count, sample_count))
    dosage = np.clip(hard_calls + random_generator.normal(0.0, 0.15, size=hard_calls.shape), 0.0, 2.0)
    return np.rint(127.0 * dosage).astype(np.uint8)


def signed_moments(codes: np.ndarray, rows: np.ndarray, training: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signed = codes[rows][:, training].astype(np.float64) - 127.0
    return signed.mean(axis=1), signed.std(axis=1)


def dense_scores(codes: np.ndarray, rows: np.ndarray, means: np.ndarray, scales: np.ndarray, effects: np.ndarray) -> np.ndarray:
    signed = codes[rows].astype(np.float64) - 127.0
    standardized = (signed - means[:, None]) / scales[:, None]
    return effects.T @ standardized


def two_fold_models(codes: np.ndarray, random_generator: np.random.Generator, draw_count: int = 0) -> list[ScoringModel]:
    variant_count, sample_count = codes.shape
    models = []
    for fold in range(2):
        rows = np.sort(random_generator.choice(variant_count, size=variant_count * 2 // 3, replace=False))
        training = np.flatnonzero(np.arange(sample_count) % 2 != fold)
        means, scales = signed_moments(codes, rows, training)
        coefficients = random_generator.normal(0.0, 0.05, size=rows.shape[0])
        models.append(
            ScoringModel(
                store_rows=rows.astype(np.int64),
                signed_means=means,
                signed_scales=scales,
                coefficients=coefficients,
                posterior_draws=coefficients[:, None]
                + random_generator.normal(0.0, 0.02, size=(rows.shape[0], draw_count)),
                alpha=np.array([0.3 * fold - 0.1, 0.5, -0.25]),
                trait_type=TraitType.QUANTITATIVE,
                predictive_intercept_shift=0.0,
            )
        )
    return models


def test_every_model_scores_to_its_dense_standardized_product_in_one_read():
    random_generator = np.random.default_rng(0)
    codes = random_codes(random_generator, variant_count=180, sample_count=90)
    models = two_fold_models(codes, random_generator)
    source = InMemoryCodes(codes)
    plan = ScoringPlan.from_models(models)

    scores = score_genetic(source, plan, cpu_budget(sample_count=90, block_rows=16, threads=3))

    assert scores.means.shape == (90, 2)
    assert np.all(np.isnan(scores.variances))
    for model_index, model in enumerate(models):
        expected = dense_scores(codes, model.store_rows, model.signed_means, model.signed_scales, model.coefficients)
        np.testing.assert_allclose(scores.means[:, model_index], expected, rtol=1e-12, atol=1e-12)
    read_rows = np.concatenate([np.arange(start, stop) for start, stop in source.reads])
    np.testing.assert_array_equal(read_rows, plan.store_rows)
    assert max(stop - start for start, stop in source.reads) <= 16


def test_posterior_draws_give_the_variance_around_the_exact_mean_score_in_the_same_read():
    random_generator = np.random.default_rng(8)
    codes = random_codes(random_generator, variant_count=90, sample_count=60)
    models = two_fold_models(codes, random_generator, draw_count=5)
    source = InMemoryCodes(codes)
    plan = ScoringPlan.from_models(models)

    scores = score_genetic(source, plan, cpu_budget(sample_count=60, block_rows=11, threads=2))

    read_rows = np.concatenate([np.arange(start, stop) for start, stop in source.reads])
    np.testing.assert_array_equal(read_rows, plan.store_rows)
    for model_index, model in enumerate(models):
        mean_score = dense_scores(codes, model.store_rows, model.signed_means, model.signed_scales, model.coefficients)
        draw_scores = dense_scores(codes, model.store_rows, model.signed_means, model.signed_scales, model.posterior_draws)
        expected_variance = np.mean((draw_scores - mean_score[None, :]) ** 2, axis=0)
        np.testing.assert_allclose(scores.means[:, model_index], mean_score, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(scores.variances[:, model_index], expected_variance, rtol=1e-10, atol=1e-14)


def test_thread_count_changes_scores_only_at_rounding_and_samples_can_be_selected():
    random_generator = np.random.default_rng(1)
    codes = random_codes(random_generator, variant_count=120, sample_count=70)
    plan = ScoringPlan.from_models(two_fold_models(codes, random_generator, draw_count=3))

    one_thread = score_genetic(InMemoryCodes(codes), plan, cpu_budget(sample_count=70, block_rows=32, threads=1))
    four_threads = score_genetic(InMemoryCodes(codes), plan, cpu_budget(sample_count=70, block_rows=32, threads=4))
    selected = np.array([69, 3, 10, 11, 40], dtype=np.int64)
    subset = score_genetic(
        InMemoryCodes(codes), plan, cpu_budget(sample_count=70, block_rows=32, threads=2), sample_indices=selected
    )

    np.testing.assert_allclose(one_thread.means, four_threads.means, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(one_thread.variances, four_threads.variances, rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(subset.means, one_thread.means[selected], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(subset.variances, one_thread.variances[selected], rtol=1e-11, atol=1e-15)


def test_linear_predictor_adds_the_intercept_and_covariates_per_model():
    random_generator = np.random.default_rng(2)
    codes = random_codes(random_generator, variant_count=60, sample_count=40)
    models = two_fold_models(codes, random_generator)
    covariates = random_generator.normal(size=(40, 2))
    genetic = score_genetic(
        InMemoryCodes(codes), ScoringPlan.from_models(models), cpu_budget(sample_count=40, block_rows=64, threads=1)
    ).means

    linear_predictor = score_linear_predictor(genetic, covariates, models)

    for model_index, model in enumerate(models):
        expected = genetic[:, model_index] + model.alpha[0] + covariates @ model.alpha[1:]
        np.testing.assert_allclose(linear_predictor[:, model_index], expected, rtol=1e-14)


def test_reduced_fit_expands_mean_and_draws_by_prior_variance_weights_and_signs():
    # Active rows 0..5; rows 1 and 4 are tied to row 0 (row 4 negated), so the
    # reduced model has representatives 0, 2, 3, 5.
    tie_map = TieMap(
        kept_indices=np.array([0, 2, 3, 5], dtype=np.int32),
        original_to_reduced=np.array([0, 0, 1, 2, 0, 3], dtype=np.int32),
        reduced_to_group=[
            TieGroup(representative_index=0, member_indices=np.array([0, 1, 4], dtype=np.int32), signs=np.array([1.0, 1.0, -1.0], dtype=np.float32)),
            TieGroup(representative_index=2, member_indices=np.array([2], dtype=np.int32), signs=np.array([1.0], dtype=np.float32)),
            TieGroup(representative_index=3, member_indices=np.array([3], dtype=np.int32), signs=np.array([1.0], dtype=np.float32)),
            TieGroup(representative_index=5, member_indices=np.array([5], dtype=np.int32), signs=np.array([1.0], dtype=np.float32)),
        ],
    )
    prior_variances = np.array([1.0, 3.0, 5.0, 7.0, 4.0, 2.0])
    beta_reduced = np.array([0.8, -0.2, 0.1, 0.05])
    draws_reduced = np.array([[0.7, 0.9], [-0.1, -0.3], [0.2, 0.0], [0.05, 0.06]])

    model = ScoringModel.from_reduced_fit(
        active_rows=np.array([10, 11, 12, 20, 21, 22], dtype=np.int64),
        signed_means=np.zeros(6),
        signed_scales=np.ones(6),
        tie_map=tie_map,
        member_prior_variances=prior_variances,
        beta_reduced=beta_reduced,
        posterior_draws_reduced=draws_reduced,
        alpha=np.array([0.0]),
        trait_type=TraitType.BINARY,
        predictive_intercept_shift=0.0,
    )

    group_weights = np.array([1.0, 3.0, 4.0]) / 8.0
    expected_mean = np.array([0.8 * group_weights[0], 0.8 * group_weights[1], -0.2, 0.1, -0.8 * group_weights[2], 0.05])
    expected_draws = np.column_stack(
        [
            [draw[0] * group_weights[0], draw[0] * group_weights[1], draw[1], draw[2], -draw[0] * group_weights[2], draw[3]]
            for draw in draws_reduced.T
        ]
    )
    np.testing.assert_allclose(model.coefficients, expected_mean, rtol=1e-15)
    np.testing.assert_allclose(model.posterior_draws, expected_draws, rtol=1e-15)
    assert model.coefficients.dtype == np.float64 and model.draw_count == 2


def test_a_binary_model_without_posterior_draws_is_rejected():
    with pytest.raises(ValueError, match="posterior draws"):
        ScoringModel(
            store_rows=np.array([0, 1], dtype=np.int64),
            signed_means=np.zeros(2),
            signed_scales=np.ones(2),
            coefficients=np.zeros(2),
            posterior_draws=np.zeros((2, 0)),
            alpha=np.array([0.0]),
            trait_type=TraitType.BINARY,
            predictive_intercept_shift=0.0,
        )


def test_the_predictive_is_the_logistic_normal_integral():
    linear_predictor = np.array([-4.0, -1.3, 0.0, 0.7, 3.2, -2.5])
    variance = np.array([0.0, 0.01, 0.5, 1.7, 4.0, 2.9])
    shift = 0.25

    probability = posterior_predictive_probability(linear_predictor, variance, shift)

    for eta, spread, value in zip(linear_predictor + shift, np.sqrt(variance), probability, strict=True):
        exact, _ = quad(
            lambda standard_normal: expit(eta + spread * standard_normal)
            * np.exp(-0.5 * standard_normal**2)
            / np.sqrt(2.0 * np.pi),
            -np.inf,
            np.inf,
            epsabs=1e-14,
            epsrel=1e-13,
        )
        assert value == pytest.approx(exact, abs=1e-10)


def test_predictive_intercept_shift_anchors_the_damped_mean_on_the_prevalence():
    random_generator = np.random.default_rng(6)
    linear_predictor = random_generator.normal(-1.5, 1.0, size=2000)
    variance = random_generator.uniform(0.0, 2.0, size=2000)
    targets = random_generator.binomial(1, 0.2, size=2000).astype(np.float64)

    shift = predictive_intercept_shift(linear_predictor, variance, targets)
    probability = posterior_predictive_probability(linear_predictor, variance, shift)

    assert float(np.mean(probability)) == pytest.approx(float(np.mean(targets)), abs=1e-12)


def test_predictive_intercept_shift_needs_cases_and_controls():
    with pytest.raises(ValueError, match="both cases and controls"):
        predictive_intercept_shift(np.zeros(4), np.zeros(4), np.zeros(4))


@pytest.mark.skipif(_try_import_cupy() is None, reason="needs a CUDA device")
def test_cuda_scores_equal_cpu_scores():
    random_generator = np.random.default_rng(7)
    codes = random_codes(random_generator, variant_count=500, sample_count=300)
    plan = ScoringPlan.from_models(two_fold_models(codes, random_generator, draw_count=4))
    budget = detect_compute_budget()
    assert budget.device_kind == "cuda"

    device_scores = score_genetic(InMemoryCodes(codes), plan, budget)
    host_scores = score_genetic(InMemoryCodes(codes), plan, cpu_budget(sample_count=300, block_rows=64, threads=2))

    np.testing.assert_allclose(device_scores.means, host_scores.means, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(device_scores.variances, host_scores.variances, rtol=1e-9, atol=1e-14)
