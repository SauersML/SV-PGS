"""One-pass multi-model scoring over dosage codes (sv_pgs.fast_scoring).

References are dense fp64 products of the standardized codes, the posterior-draw
variance computed densely from the same draws, and the logistic-normal integral
by adaptive quadrature.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterator, Sequence

import numpy as np
import pytest
from scipy import stats
from scipy.integrate import quad
from scipy.special import expit

from sv_pgs.compute_budget import ComputeBudget, _try_import_cupy, detect_compute_budget
from sv_pgs.config import TraitType
from sv_pgs.data import TieGroup, TieMap
from sv_pgs.fast_scoring import (
    ScoringModel,
    GeneticScores,
    ScoringPlan,
    _cpu_panels,
    _batch_bytes,
    _host_bytes,
    _output_bytes,
    _trapezoid_rule,
    posterior_predictive_probability,
    predictive_intercept_shift,
    score_genetic,
    score_linear_predictor,
)


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


def cpu_budget(plan: ScoringPlan, *, sample_count: int, block_rows: int, threads: int, selected_count: int | None = None) -> ComputeBudget:
    """A CPU budget whose host memory plan fits exactly ``block_rows`` rows per read."""
    host_bytes = _host_bytes(plan, sample_count, sample_count if selected_count is None else selected_count, block_rows, "cpu")
    return ComputeBudget(
        device_kind="cpu",
        device_ids=(),
        device_names=(),
        device_bytes=(),
        device_compute_capabilities=(),
        host_bytes=host_bytes,
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


def gamma(term_count: int) -> float:
    """Higham's gamma_n = n u / (1 - n u), the relative error bound of an fp64 sum of n terms."""
    unit_roundoff = np.finfo(np.float64).eps / 2.0
    return term_count * unit_roundoff / (1.0 - term_count * unit_roundoff)


def score_rounding_bound(codes: np.ndarray, rows: np.ndarray, means: np.ndarray, scales: np.ndarray, effects: np.ndarray) -> np.ndarray:
    """Bound on |fast score - dense score|: both sum p terms of |beta_j / sigma_j| (|s_ij| + |mu_j|)
    in some order, with a few roundings per term (scaling, centring, the offset)."""
    signed = np.abs(codes[rows].astype(np.float64) - 127.0)
    magnitude = (np.abs(effects) / scales[:, None]).T @ (signed + np.abs(means)[:, None])
    return 2.0 * gamma(rows.shape[0] + 4) * magnitude


def assert_within(actual: np.ndarray, expected: np.ndarray, bound: np.ndarray) -> None:
    assert np.all(np.abs(actual - expected) <= bound)


def dense_reference(codes: np.ndarray, model: ScoringModel) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dense mean score and draw variance of one model, with their rounding bounds."""
    moments = (codes, model.store_rows, model.signed_means, model.signed_scales)
    mean_score = dense_scores(*moments, model.coefficients[:, None])[0]
    mean_bound = score_rounding_bound(*moments, model.coefficients[:, None])[0]
    draws = np.asarray(model.posterior_draws)
    draw_scores = dense_scores(*moments, draws)
    deviations = draw_scores - mean_score[None, :]
    variance = np.mean(deviations**2, axis=0)
    # each deviation moves by at most d = its draw's bound + the mean's bound, so its square by 2|dev| d + d^2
    deviation_bound = score_rounding_bound(*moments, draws) + mean_bound[None, :]
    variance_bound = np.mean(2.0 * np.abs(deviations) * deviation_bound + deviation_bound**2, axis=0)
    return mean_score, mean_bound, variance, variance_bound + gamma(model.draw_count + 2) * variance


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
                covariate_draws=np.repeat(np.array([0.3 * fold - 0.1, 0.5, -0.25])[:, None], draw_count, axis=1),
                covariate_covariance=np.zeros((3, 3)),
                gaussian_posterior=True,
            )
        )
    return models


def test_every_model_scores_to_its_dense_standardized_product_in_one_read():
    random_generator = np.random.default_rng(0)
    codes = random_codes(random_generator, variant_count=180, sample_count=90)
    models = two_fold_models(codes, random_generator)
    source = InMemoryCodes(codes)
    plan = ScoringPlan.from_models(models)

    scores = score_genetic(source, plan, cpu_budget(plan, sample_count=90, block_rows=16, threads=3))

    assert scores.means.shape == (90, 2)
    assert np.all(np.isnan(scores.variances))
    for model_index, model in enumerate(models):
        effects = model.coefficients[:, None]
        expected = dense_scores(codes, model.store_rows, model.signed_means, model.signed_scales, effects)[0]
        bound = score_rounding_bound(codes, model.store_rows, model.signed_means, model.signed_scales, effects)[0]
        assert_within(scores.means[:, model_index], expected, bound)
    read_rows = np.concatenate([np.arange(start, stop) for start, stop in source.reads])
    np.testing.assert_array_equal(read_rows, plan.store_rows)
    assert max(stop - start for start, stop in source.reads) <= 16


def test_posterior_draws_give_the_variance_around_the_exact_mean_score_in_the_same_read():
    random_generator = np.random.default_rng(8)
    codes = random_codes(random_generator, variant_count=90, sample_count=60)
    models = two_fold_models(codes, random_generator, draw_count=5)
    source = InMemoryCodes(codes)
    plan = ScoringPlan.from_models(models)

    scores = score_genetic(source, plan, cpu_budget(plan, sample_count=60, block_rows=11, threads=2))

    read_rows = np.concatenate([np.arange(start, stop) for start, stop in source.reads])
    np.testing.assert_array_equal(read_rows, plan.store_rows)
    for model_index, model in enumerate(models):
        mean_score, mean_bound, variance, variance_bound = dense_reference(codes, model)
        assert_within(scores.means[:, model_index], mean_score, mean_bound)
        assert_within(scores.variances[:, model_index], variance, variance_bound)


def test_thread_count_changes_scores_only_at_rounding_and_samples_can_be_selected():
    random_generator = np.random.default_rng(1)
    codes = random_codes(random_generator, variant_count=120, sample_count=70)
    models = two_fold_models(codes, random_generator, draw_count=3)
    plan = ScoringPlan.from_models(models)

    one_thread = score_genetic(InMemoryCodes(codes), plan, cpu_budget(plan, sample_count=70, block_rows=32, threads=1))
    four_threads = score_genetic(InMemoryCodes(codes), plan, cpu_budget(plan, sample_count=70, block_rows=32, threads=4))
    selected = np.array([69, 3, 10, 11, 40], dtype=np.int64)
    subset = score_genetic(
        InMemoryCodes(codes), plan, cpu_budget(plan, sample_count=70, block_rows=32, threads=2, selected_count=5), sample_indices=selected
    )

    # Thread and panel counts only reorder each score's sum: every run is within the rounding
    # bound of the dense reference, so two runs are within twice it.
    for model_index, model in enumerate(models):
        _mean, mean_bound, _variance, variance_bound = dense_reference(codes, model)
        assert_within(one_thread.means[:, model_index], four_threads.means[:, model_index], 2.0 * mean_bound)
        assert_within(one_thread.variances[:, model_index], four_threads.variances[:, model_index], 2.0 * variance_bound)
        assert_within(subset.means[:, model_index], one_thread.means[selected, model_index], 2.0 * mean_bound[selected])
        assert_within(subset.variances[:, model_index], one_thread.variances[selected, model_index], 2.0 * variance_bound[selected])


def test_linear_predictor_adds_the_intercept_and_covariates_per_model():
    random_generator = np.random.default_rng(2)
    codes = random_codes(random_generator, variant_count=60, sample_count=40)
    models = two_fold_models(codes, random_generator)
    covariates = random_generator.normal(size=(40, 2))
    plan = ScoringPlan.from_models(models)
    genetic = score_genetic(InMemoryCodes(codes), plan, cpu_budget(plan, sample_count=40, block_rows=60, threads=1)).means

    linear_predictor = score_linear_predictor(genetic, covariates, models)

    for model_index, model in enumerate(models):
        expected = genetic[:, model_index] + model.alpha[0] + covariates @ model.alpha[1:]
        magnitude = np.abs(genetic[:, model_index]) + abs(model.alpha[0]) + np.abs(covariates) @ np.abs(model.alpha[1:])
        assert_within(linear_predictor[:, model_index], expected, 2.0 * gamma(model.alpha.shape[0] + 1) * magnitude)


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
        covariate_draws=np.zeros((1, 2)),
        covariate_covariance=np.zeros((1, 1)),
        gaussian_posterior=True,
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
    # a weight is a ratio of a sum, times the group coefficient and a sign: a few roundings each way
    assert_within(model.coefficients, expected_mean, gamma(8) * np.abs(expected_mean))
    assert_within(np.asarray(model.posterior_draws), expected_draws, gamma(8) * np.abs(expected_draws))
    assert model.coefficients.dtype == np.float64 and model.draw_count == 2


def test_a_binary_model_without_posterior_draws_is_rejected():
    with pytest.raises(ValueError, match="posterior draws"):
        ScoringModel(
            store_rows=np.array([0, 1], dtype=np.int64),
            signed_means=np.zeros(2),
            signed_scales=np.ones(2),
            coefficients=np.zeros(2),
            posterior_draws=np.zeros((2, 0)),
            covariate_draws=np.zeros((1, 0)),
            covariate_covariance=np.zeros((1, 1)),
            gaussian_posterior=True,
            alpha=np.array([0.0]),
            trait_type=TraitType.BINARY,
            predictive_intercept_shift=0.0,
        )


def test_a_host_budget_below_one_models_block_row_is_refused_and_panels_balance():
    random_generator = np.random.default_rng(9)
    codes = random_codes(random_generator, variant_count=30, sample_count=20)
    plan = ScoringPlan.from_models(two_fold_models(codes, random_generator))
    # The result for both models plus the smaller of the two models' one-row reads, less a byte: no batch fits.
    smallest = min(_batch_bytes(plan, [model], 20, 20, 1, "cpu", "keep") for model in range(plan.model_count))
    budget = replace(cpu_budget(plan, sample_count=20, block_rows=1, threads=1), host_bytes=_output_bytes(plan, 20, "keep") + smallest - 1)
    with pytest.raises(MemoryError):
        score_genetic(InMemoryCodes(codes), plan, budget)
    widths = [stop - start for start, stop in _cpu_panels(103, 4)]
    assert sum(widths) == 103 and max(widths) - min(widths) <= 1 and len(widths) == 4
    assert _cpu_panels(3, 8) == [(0, 1), (1, 2), (2, 3)]


def test_a_one_draw_binary_model_carries_its_variance_into_the_damped_predictive():
    random_generator = np.random.default_rng(11)
    codes = random_codes(random_generator, variant_count=80, sample_count=50)
    models = [replace(model, trait_type=TraitType.BINARY) for model in two_fold_models(codes, random_generator, draw_count=1)]
    plan = ScoringPlan.from_models(models)
    scores = score_genetic(InMemoryCodes(codes), plan, cpu_budget(plan, sample_count=50, block_rows=16, threads=2))
    linear_predictor = score_linear_predictor(scores.means, np.zeros((50, 2)), models)

    for model_index, model in enumerate(models):
        _mean, _mean_bound, variance, variance_bound = dense_reference(codes, model)
        assert np.all(np.isfinite(scores.variances[:, model_index])) and np.all(scores.variances[:, model_index] >= 0.0)
        assert_within(scores.variances[:, model_index], variance, variance_bound)
        eta = linear_predictor[:, model_index]
        probability = posterior_predictive_probability(eta, scores.variances[:, model_index], 0.0)
        # E[sigmoid(eta + sqrt(v) Z)] lies strictly between 1/2 and sigmoid(eta) whenever eta != 0 and v > 0
        damped = (scores.variances[:, model_index] > 0.0) & (eta != 0.0)
        assert np.all(np.abs(probability[damped] - 0.5) < np.abs(expit(eta[damped]) - 0.5))

    # a model without draws has no variance, and the predictive refuses it rather than dropping the damping
    no_draws = score_genetic(InMemoryCodes(codes), ScoringPlan.from_models(two_fold_models(codes, random_generator)),
                             cpu_budget(plan, sample_count=50, block_rows=16, threads=1))
    with pytest.raises(ValueError, match="finite"):
        posterior_predictive_probability(no_draws.means[:, 0], no_draws.variances[:, 0], 0.0)


def exact_predictive(eta: float, variance: float) -> tuple[float, float]:
    """E[sigmoid(eta + sqrt(v) Z)] by adaptive quadrature over |z| <= 40 (the mass beyond is below 1e-340),
    split at the sigmoid's transition z = -eta / sqrt(v), with quad's error estimate."""
    spread = float(np.sqrt(variance))

    def integrand(standard_normal: float) -> float:
        return expit(eta + spread * standard_normal) * np.exp(-0.5 * standard_normal**2) / np.sqrt(2.0 * np.pi)

    breaks = [-40.0, 40.0] + ([float(np.clip(-eta / spread, -40.0, 40.0))] if spread > 0.0 else [])
    breaks = sorted(set(breaks))
    value = error = 0.0
    for low, high in zip(breaks[:-1], breaks[1:]):
        part, part_error = quad(integrand, low, high, epsabs=np.finfo(np.float64).eps, epsrel=0.0, limit=1000)
        value, error = value + part, error + part_error
    return value, error


def predictive_error_bound(variance: np.ndarray) -> np.ndarray:
    """2 eps of quadrature and truncation (see _trapezoid_rule), plus the rounding of a sum of 2K + 1
    terms of a few roundings each, all of total mass at most 1."""
    _step, half_width = _trapezoid_rule(np.asarray(variance, dtype=np.float64))
    return 2.0 * np.finfo(np.float64).eps + np.array([gamma(2 * int(width) + 9) for width in np.atleast_1d(half_width)])


def test_the_predictive_is_the_logistic_normal_integral_at_every_variance():
    linear_predictor = np.array([-4.0, -1.3, 0.0, 0.7, 3.2, -2.5, -3.0, 0.3, 2.0, -0.7])
    for variance_value in (0.0, 0.01, 0.5, 1.7, 4.0, 5.0, 30.0, 1e4):
        variance = np.full(linear_predictor.shape, variance_value)
        probability = posterior_predictive_probability(linear_predictor, variance, 0.25)
        bound = predictive_error_bound(variance)
        for eta, value, entry_bound in zip(linear_predictor + 0.25, probability, bound, strict=True):
            exact, quadrature_error = exact_predictive(float(eta), variance_value)
            assert abs(value - exact) <= entry_bound + quadrature_error


def test_the_predictive_completes_for_single_draw_variances_and_refuses_nan():
    random_generator = np.random.default_rng(12)
    linear_predictor = random_generator.normal(-1.5, 1.0, size=100_000)
    variance = 0.25 * random_generator.chisquare(1, size=100_000)
    probability = posterior_predictive_probability(linear_predictor, variance, 0.0)
    assert probability.shape == linear_predictor.shape
    assert np.all(np.isfinite(probability)) and np.all((probability > 0.0) & (probability < 1.0))
    with pytest.raises(ValueError, match="not finite"):
        posterior_predictive_probability(np.array([0.0, np.nan]), np.array([1.0, 1.0]), 0.0)


def test_credible_intervals_are_exact_student_t_for_any_draw_count():
    random_generator = np.random.default_rng(13)
    sample_count, coverage = 20_000, 0.9
    for draw_count in (1, 5):
        means = random_generator.normal(size=(sample_count, 1))
        true_variance = random_generator.uniform(0.1, 2.0, size=(sample_count, 1))
        genetic_value = means + np.sqrt(true_variance) * random_generator.standard_normal((sample_count, 1))
        draws = means + np.sqrt(true_variance) * random_generator.standard_normal((sample_count, draw_count))
        estimated = np.mean((draws - means) ** 2, axis=1, keepdims=True)
        scores = GeneticScores(means=means, variances=estimated, draw_counts=(draw_count,), draws=(draws,), gaussian_posteriors=(True,))
        lower, upper = scores.credible_interval(coverage)
        covered = int(np.sum((lower <= genetic_value) & (genetic_value <= upper)))
        assert stats.binomtest(covered, sample_count, coverage).pvalue > 0.05
        if draw_count == 1:
            normal_half_width = stats.norm.isf(0.5 * (1.0 - coverage)) * np.sqrt(estimated)
            normal_covered = int(np.sum(np.abs(genetic_value - means) <= normal_half_width))
            assert stats.binomtest(normal_covered, sample_count, coverage, alternative="less").pvalue < 0.05
    with pytest.raises(ValueError, match="no credible interval"):
        GeneticScores(means=means, variances=np.full_like(means, np.nan), draw_counts=(0,), draws=(np.zeros((sample_count, 0)),), gaussian_posteriors=(True,)).credible_interval(coverage)


def test_predictive_intercept_shift_anchors_the_damped_mean_on_the_prevalence():
    random_generator = np.random.default_rng(6)
    linear_predictor = random_generator.normal(-1.5, 1.0, size=2000)
    variance = random_generator.uniform(0.0, 2.0, size=2000)
    targets = random_generator.binomial(1, 0.2, size=2000).astype(np.float64)

    shift = predictive_intercept_shift(linear_predictor, variance, targets)
    probability = posterior_predictive_probability(linear_predictor, variance, shift)

    # brentq resolves the shift to xtol + rtol |shift|, and sigmoid' <= 1/4 carries that to the mean
    eps = np.finfo(np.float64).eps
    prevalence = float(np.mean(targets))
    resolution = 4.0 * eps * min(prevalence, 1.0 - prevalence) + 4.0 * eps * abs(shift)
    bound = 0.25 * resolution + float(np.max(predictive_error_bound(variance))) + gamma(2000)
    assert abs(float(np.mean(probability)) - prevalence) <= bound


def test_predictive_intercept_shift_needs_cases_and_controls():
    with pytest.raises(ValueError, match="both cases and controls"):
        predictive_intercept_shift(np.zeros(4), np.zeros(4), np.zeros(4))


@pytest.mark.skipif(_try_import_cupy() is None, reason="needs a CUDA device")
def test_cuda_scores_equal_cpu_scores():
    random_generator = np.random.default_rng(7)
    codes = random_codes(random_generator, variant_count=500, sample_count=300)
    models = two_fold_models(codes, random_generator, draw_count=4)
    plan = ScoringPlan.from_models(models)
    budget = detect_compute_budget()
    assert budget.device_kind == "cuda"

    device_scores = score_genetic(InMemoryCodes(codes), plan, budget)
    host_scores = score_genetic(InMemoryCodes(codes), plan, cpu_budget(plan, sample_count=300, block_rows=64, threads=2))

    for model_index, model in enumerate(models):
        _mean, mean_bound, _variance, variance_bound = dense_reference(codes, model)
        assert_within(device_scores.means[:, model_index], host_scores.means[:, model_index], 2.0 * mean_bound)
        assert_within(device_scores.variances[:, model_index], host_scores.variances[:, model_index], 2.0 * variance_bound)


class _RefusingLaw:
    """A draw law that fails if any tile is asked of it: mean prediction must never generate a draw."""

    kind = "refusing"

    def __init__(self, rows: int, draw_count: int) -> None:
        self.shape = (rows, draw_count)

    def tile(self, start: int, stop: int) -> np.ndarray:
        raise AssertionError("mean prediction asked for posterior draws")

    def tile_row_bytes(self) -> int:
        return 0


def test_mean_prediction_generates_no_draw_and_forms_no_weight_matrix_over_the_rows():
    random_generator = np.random.default_rng(21)
    codes = random_codes(random_generator, variant_count=150, sample_count=60)
    models = [replace(model, posterior_draws=_RefusingLaw(model.store_rows.shape[0], 5), covariate_draws=np.zeros((3, 5)))
              for model in two_fold_models(codes, random_generator)]
    plan = ScoringPlan.from_models(models)
    # The plan holds per-model row positions and mean offsets only: nothing of the size rows x columns.
    assert not hasattr(plan, "weights")
    budget = cpu_budget(plan, sample_count=60, block_rows=20, threads=2)
    scores = score_genetic(InMemoryCodes(codes), plan, budget, draws="none")
    assert scores.draw_counts == (0, 0) and scores.draws == () and np.all(np.isnan(scores.variances))
    for model_index, model in enumerate(models):
        effects = model.coefficients[:, None]
        expected = dense_scores(codes, model.store_rows, model.signed_means, model.signed_scales, effects)[0]
        bound = score_rounding_bound(codes, model.store_rows, model.signed_means, model.signed_scales, effects)[0]
        assert_within(scores.means[:, model_index], expected, bound)


def test_a_budget_for_one_model_at_a_time_scores_in_batches_to_the_same_values():
    random_generator = np.random.default_rng(22)
    codes = random_codes(random_generator, variant_count=120, sample_count=50)
    models = two_fold_models(codes, random_generator, draw_count=4)
    plan = ScoringPlan.from_models(models)
    whole = score_genetic(InMemoryCodes(codes), plan, cpu_budget(plan, sample_count=50, block_rows=30, threads=2))
    # Room for the result and one model's read, not both models' at once: two batches, two reads of the rows.
    one_model = max(_batch_bytes(plan, [model], 50, 50, 1, "cpu", "keep") for model in range(plan.model_count))
    both = _batch_bytes(plan, [0, 1], 50, 50, 1, "cpu", "keep")
    assert one_model < both
    source = InMemoryCodes(codes)
    batched = score_genetic(source, plan, replace(cpu_budget(plan, sample_count=50, block_rows=1, threads=2),
                                                  host_bytes=_output_bytes(plan, 50, "keep") + one_model))
    assert sum(stop - start for start, stop in source.reads) == 2 * plan.store_rows.shape[0]
    for model_index, model in enumerate(models):
        _mean, mean_bound, _variance, variance_bound = dense_reference(codes, model)
        assert_within(batched.means[:, model_index], whole.means[:, model_index], 2.0 * mean_bound)
        assert_within(batched.variances[:, model_index], whole.variances[:, model_index], 2.0 * variance_bound)


def test_a_product_mixture_law_scores_to_the_dense_reference_of_its_own_draws():
    """The law's draws are generated a block of rows at a time inside the read; the scores are those of the same draws
    formed whole (the law is counter-based, so both see the same numbers)."""
    from sv_pgs.draw_laws import ProductMixtureDraws

    random_generator = np.random.default_rng(23)
    codes = random_codes(random_generator, variant_count=100, sample_count=40)
    base = two_fold_models(codes, random_generator)[0]
    rows = base.store_rows.shape[0]
    law = ProductMixtureDraws(
        class_index=np.zeros(rows, dtype=np.int64), log_scale=np.full(rows, -4.0), omega=np.outer([40.0, 20.0], np.ones(rows)),
        log_density=np.log(np.full((1, 5), 0.2)), log_variance_grid=np.linspace(-4.0, 2.0, 5),
        shift=random_generator.normal(size=(2, rows)), weights=np.array([0.5, 0.5]), key=(3, 4), draw_count=6,
    )
    model = replace(base, posterior_draws=law, covariate_draws=np.repeat(base.alpha[:, None], 6, axis=1))
    plan = ScoringPlan.from_models([model])
    scores = score_genetic(InMemoryCodes(codes), plan, cpu_budget(plan, sample_count=40, block_rows=7, threads=2))
    mean_score, mean_bound, variance, variance_bound = dense_reference(codes, model)
    assert_within(scores.means[:, 0], mean_score, mean_bound)
    assert_within(scores.variances[:, 0], variance, variance_bound)
