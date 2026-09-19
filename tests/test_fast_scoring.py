"""One-pass multi-model scoring over dosage codes (sv_pgs.fast_scoring).

References are dense fp64 products of the standardized codes, and the fitted
BayesianPGS model's own predictions for the in-sample reproduction.
"""
from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
import pytest

from sv_pgs import BayesianPGS, ModelConfig, TraitType, VariantClass, VariantRecord
from sv_pgs.compute_budget import ComputeBudget, detect_compute_budget
from sv_pgs.fast_scoring import (
    BlockPosterior,
    ScoringModel,
    ScoringPlan,
    block_predictor_variance,
    posterior_predictive_probability,
    predictive_intercept_shift,
    score_genetic,
    score_linear_predictor,
)
from sv_pgs.genotype import _try_import_cupy
from sv_pgs.numeric import logistic_normal_probit_scale, stable_sigmoid


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


def dense_scores(codes: np.ndarray, model: ScoringModel) -> np.ndarray:
    signed = codes[model.store_rows].astype(np.float64) - 127.0
    standardized = (signed - model.signed_means[:, None]) / model.signed_scales[:, None]
    return model.coefficients @ standardized


def two_fold_models(codes: np.ndarray, random_generator: np.random.Generator) -> list[ScoringModel]:
    variant_count, sample_count = codes.shape
    models = []
    for fold in range(2):
        rows = np.sort(random_generator.choice(variant_count, size=variant_count * 2 // 3, replace=False))
        training = np.flatnonzero(np.arange(sample_count) % 2 != fold)
        means, scales = signed_moments(codes, rows, training)
        models.append(
            ScoringModel(
                store_rows=rows.astype(np.int64),
                signed_means=means,
                signed_scales=scales,
                coefficients=random_generator.normal(0.0, 0.05, size=rows.shape[0]),
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

    assert scores.shape == (90, 2)
    for model_index, model in enumerate(models):
        np.testing.assert_allclose(scores[:, model_index], dense_scores(codes, model), rtol=1e-12, atol=1e-12)
    read_rows = np.concatenate([np.arange(start, stop) for start, stop in source.reads])
    np.testing.assert_array_equal(read_rows, plan.store_rows)
    assert max(stop - start for start, stop in source.reads) <= 16


def test_thread_count_changes_scores_only_at_rounding_and_samples_can_be_selected():
    random_generator = np.random.default_rng(1)
    codes = random_codes(random_generator, variant_count=120, sample_count=70)
    plan = ScoringPlan.from_models(two_fold_models(codes, random_generator))

    one_thread = score_genetic(InMemoryCodes(codes), plan, cpu_budget(sample_count=70, block_rows=32, threads=1))
    four_threads = score_genetic(InMemoryCodes(codes), plan, cpu_budget(sample_count=70, block_rows=32, threads=4))
    selected = np.array([69, 3, 10, 11, 40], dtype=np.int64)
    subset = score_genetic(
        InMemoryCodes(codes), plan, cpu_budget(sample_count=70, block_rows=32, threads=2), sample_indices=selected
    )

    np.testing.assert_allclose(one_thread, four_threads, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(subset, one_thread[selected], rtol=1e-13, atol=1e-13)


def test_linear_predictor_adds_the_intercept_and_covariates_per_model():
    random_generator = np.random.default_rng(2)
    codes = random_codes(random_generator, variant_count=60, sample_count=40)
    models = two_fold_models(codes, random_generator)
    covariates = random_generator.normal(size=(40, 2))
    genetic = score_genetic(InMemoryCodes(codes), ScoringPlan.from_models(models), cpu_budget(sample_count=40, block_rows=64, threads=1))

    linear_predictor = score_linear_predictor(genetic, covariates, models)

    for model_index, model in enumerate(models):
        expected = genetic[:, model_index] + model.alpha[0] + covariates @ model.alpha[1:]
        np.testing.assert_allclose(linear_predictor[:, model_index], expected, rtol=1e-14)


def _fit_on_codes(codes: np.ndarray, trait_type: TraitType, random_generator: np.random.Generator):
    variant_count, sample_count = codes.shape
    dosage = (codes.T.astype(np.float64) / 127.0).astype(np.float32)
    covariates = random_generator.normal(size=(sample_count, 2)).astype(np.float32)
    effects = np.zeros(variant_count)
    effects[:6] = random_generator.normal(0.0, 0.8, size=6)
    liability = (dosage - dosage.mean(axis=0)) @ effects + 0.3 * covariates[:, 0]
    if trait_type == TraitType.BINARY:
        targets = random_generator.binomial(1, 1.0 / (1.0 + np.exp(-(liability - 0.8)))).astype(np.float32)
    else:
        targets = (liability + random_generator.normal(0.0, 1.0, size=sample_count)).astype(np.float32)
    records = [
        VariantRecord(f"variant_{index}", VariantClass.SNV, "1", 100 + index, allele_frequency=0.2)
        for index in range(variant_count)
    ]
    model = BayesianPGS(
        ModelConfig(trait_type=trait_type, max_outer_iterations=8, minimum_minor_allele_frequency=0.0)
    ).fit(dosage, covariates, targets, records)
    return model, dosage, covariates


def _scoring_model_of(model: BayesianPGS, trait_type: TraitType) -> ScoringModel:
    state = model.state
    assert state is not None
    variant_count = state.full_coefficients.shape[0]
    assert state.active_variant_indices.tolist() == list(range(variant_count))
    return ScoringModel.from_reduced_fit(
        active_rows=np.arange(variant_count, dtype=np.int64),
        signed_means=127.0 * state.preprocessor.means.astype(np.float64) - 127.0,
        signed_scales=127.0 * state.preprocessor.scales.astype(np.float64),
        tie_map=state.tie_map,
        fit_result=state.fit_result,
        beta_reduced=np.asarray(state.fit_result.beta_reduced, dtype=np.float64),
        trait_type=trait_type,
    )


@pytest.mark.parametrize("trait_type", [TraitType.QUANTITATIVE, TraitType.BINARY])
def test_in_sample_scores_reproduce_the_fitted_model(trait_type):
    random_generator = np.random.default_rng(3)
    codes = random_codes(random_generator, variant_count=40, sample_count=150)
    codes[7] = codes[3]
    model, dosage, covariates = _fit_on_codes(codes, trait_type, random_generator)
    assert model.state is not None and len(model.state.tie_map.kept_indices) < codes.shape[0]
    scoring_model = _scoring_model_of(model, trait_type)

    genetic = score_genetic(InMemoryCodes(codes), ScoringPlan.from_models([scoring_model]), cpu_budget(sample_count=150, block_rows=9, threads=2))
    linear_predictor = score_linear_predictor(genetic, covariates.astype(np.float64), [scoring_model])[:, 0]

    np.testing.assert_allclose(genetic[:, 0], dense_scores(codes, scoring_model), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(scoring_model.coefficients, model.state.full_coefficients, rtol=1e-6, atol=1e-9)
    # decision_function works in float32 on float32 DS and parameters.
    np.testing.assert_allclose(linear_predictor, model.decision_function(dosage, covariates), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(linear_predictor, model.state.training_linear_predictor, rtol=1e-4, atol=1e-4)


def test_binary_posterior_predictive_matches_the_fitted_model():
    random_generator = np.random.default_rng(4)
    codes = random_codes(random_generator, variant_count=40, sample_count=150)
    codes[7] = codes[3]
    model, dosage, covariates = _fit_on_codes(codes, TraitType.BINARY, random_generator)
    state = model.state
    assert state is not None
    scoring_model = _scoring_model_of(model, TraitType.BINARY)
    source = InMemoryCodes(codes)
    genetic = score_genetic(source, ScoringPlan.from_models([scoring_model]), cpu_budget(sample_count=150, block_rows=64, threads=1))
    linear_predictor = score_linear_predictor(genetic, covariates.astype(np.float64), [scoring_model])[:, 0]
    # The fitted posterior is mean-field over the reduced coefficients: one block whose
    # precision is diag(1 / Var(beta_r)) on the representative rows.
    representatives = np.asarray(state.tie_map.kept_indices, dtype=np.int64)
    order = np.argsort(representatives)
    reduced_variance = np.asarray(state.fit_result.beta_variance, dtype=np.float64)[order]
    rows = representatives[order]
    block = BlockPosterior(
        store_rows=rows,
        signed_means=scoring_model.signed_means[rows],
        signed_scales=scoring_model.signed_scales[rows],
        precision_factor=np.diag(1.0 / np.sqrt(reduced_variance)),
    )

    variance = block_predictor_variance(source, [block])
    probability = posterior_predictive_probability(linear_predictor, variance, scoring_model.predictive_intercept_shift)

    np.testing.assert_allclose(variance, model.predictor_variance(dosage), rtol=1e-4, atol=1e-9)
    np.testing.assert_allclose(probability, model.predict_proba(dosage, covariates)[:, 1], rtol=1e-4, atol=1e-6)


def test_block_predictor_variance_is_the_block_quadratic_form():
    random_generator = np.random.default_rng(5)
    codes = random_codes(random_generator, variant_count=50, sample_count=30)
    training = np.arange(30)
    blocks = []
    expected = np.zeros(30)
    for rows in (np.array([2, 3, 5, 9]), np.array([20, 21, 22, 30, 31, 40])):
        means, scales = signed_moments(codes, rows, training)
        factor = random_generator.normal(size=(rows.size, rows.size))
        precision = factor @ factor.T + rows.size * np.eye(rows.size)
        blocks.append(
            BlockPosterior(
                store_rows=rows.astype(np.int64),
                signed_means=means,
                signed_scales=scales,
                precision_factor=np.linalg.cholesky(precision),
            )
        )
        standardized = ((codes[rows].astype(np.float64) - 127.0) - means[:, None]) / scales[:, None]
        expected += np.einsum("ij,ij->j", standardized, np.linalg.solve(precision, standardized))

    variance = block_predictor_variance(InMemoryCodes(codes), blocks)

    np.testing.assert_allclose(variance, expected, rtol=1e-12)


def test_predictive_intercept_shift_anchors_the_damped_mean_on_the_prevalence():
    random_generator = np.random.default_rng(6)
    linear_predictor = random_generator.normal(-1.5, 1.0, size=2000)
    variance = random_generator.uniform(0.0, 2.0, size=2000)
    targets = random_generator.binomial(1, 0.2, size=2000).astype(np.float64)

    shift = predictive_intercept_shift(linear_predictor, variance, targets)
    probability = posterior_predictive_probability(linear_predictor, variance, shift)

    assert float(np.mean(probability)) == pytest.approx(float(np.mean(targets)), abs=1e-9)
    np.testing.assert_allclose(
        probability,
        np.asarray(stable_sigmoid((linear_predictor + shift) / logistic_normal_probit_scale(variance))),
        rtol=1e-15,
    )


@pytest.mark.skipif(_try_import_cupy() is None, reason="needs a CUDA device")
def test_cuda_scores_equal_cpu_scores():
    random_generator = np.random.default_rng(7)
    codes = random_codes(random_generator, variant_count=500, sample_count=300)
    plan = ScoringPlan.from_models(two_fold_models(codes, random_generator))
    budget = detect_compute_budget()
    assert budget.device_kind == "cuda"

    device_scores = score_genetic(InMemoryCodes(codes), plan, budget)
    host_scores = score_genetic(InMemoryCodes(codes), plan, cpu_budget(sample_count=300, block_rows=64, threads=2))

    np.testing.assert_allclose(device_scores, host_scores, rtol=1e-12, atol=1e-12)
