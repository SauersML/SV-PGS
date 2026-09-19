"""One-pass scoring of every fitted model over the uint8 dosage store.

Model. A fitted model is a posterior over standardized genotype columns
x_j = (s_j - mu_j) / sigma_j of the signed dosage codes s = code - 127, with
mu_j and sigma_j the training-sample mean and population SD (ddof 0). Its
genetic score is

    g_i = sum_j beta_j x_ij = sum_j w_j s_ij + offset,
    w_j = beta_j / sigma_j,   offset = -sum_j w_j mu_j,

so a coefficient vector is one column of a weight matrix W and one offset.
Every model (traits x folds; each fold has its own active rows and training
moments) is scored by ONE read of the codes: each block of store rows adds
W_b' S_b to a [columns, samples] accumulator. The read is one byte per
genotype against 2 * columns flops, so the pass is bound by the read at any
realistic column count.

Exactness. Codes are small integers, exact in fp64, and the weights and the
accumulator are fp64, so a score equals X beta to fp64 rounding: in-sample
scores reproduce the fitted model's linear predictor. On the CPU each sample
column is owned by one worker that adds the blocks in store order with
single-threaded BLAS; thread and device counts change a score only at fp64
rounding (BLAS kernels may order a panel's sums by its width).

Posterior draws. A model carries K exact draws beta^(k) from its Gaussian
posterior q; Stage 2 draws them by perturb-and-solve on its own genotype
passes. Each draw is one more weight column, so the same single read gives
the draw scores g_i^(k). The posterior mean score g_i is known exactly, so

    v_i = (1 / K) sum_k (g_i^(k) - g_i)^2

is an unbiased estimate of the posterior variance of the genetic score, with
relative standard error sqrt(2 / K): 18% at Stage 2's K = 64. That moves a
damped probability by at most |sigmoid''| / 2 * 0.18 * v_i <= 0.009 v_i.
The covariate coefficients have a flat prior and an O(1/n) posterior
variance, which the predictive ignores.

Binary models. The posterior predictive is

    P(y_i = 1) = E[sigmoid(eta_i + c + sqrt(v_i) Z)],   Z ~ N(0, 1),

evaluated by Gauss-Hermite quadrature with 64 nodes: exact to below 1e-10
for v_i <= 4, far past any predictor variance a polygenic score reaches. The
shift c anchors the mean predictive on the training prevalence; the fitted
intercept anchors the plug-in predictor, and the damped one needs its own
anchor.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Protocol, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from threadpoolctl import threadpool_limits

from sv_pgs._typing import F64Array, I64Array, NDArray, U8Array
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import TraitType
from sv_pgs.data import TieMap
from sv_pgs.genotype import _cupy_device_context, _try_import_cupy
from sv_pgs.progress import log

SIGNED_CODE_OFFSET = 127.0
# Each block row costs one byte per sample in every read-ahead buffer plus
# eight in the fp64 conversion panels; a block uses half the host budget.
_READ_AHEAD_BUFFERS = 3
_HOST_BYTES_PER_BLOCK_CELL = _READ_AHEAD_BUFFERS + 8
_HOST_MEMORY_SHARES = 2
# Sample panels per CPU worker, so uneven panels still balance.
_PANELS_PER_WORKER = 4
# Device memory is split between the uploaded codes, their fp64 tile and the
# accumulator.
_DEVICE_MEMORY_SHARES = 4
# Gauss-Hermite nodes for the logistic-normal predictive (see the module docstring).
PREDICTIVE_QUADRATURE_NODES = 64
_HERMITE_NODES, _HERMITE_WEIGHTS = np.polynomial.hermite.hermgauss(PREDICTIVE_QUADRATURE_NODES)
# A binary model needs posterior draws for its damped predictive.
MINIMUM_BINARY_POSTERIOR_DRAWS = 2


class CodeBlockSource(Protocol):
    """Reader of uint8 dosage codes, variant-major, on the store's full sample axis."""

    @property
    def sample_count(self) -> int: ...

    def iter_code_blocks(
        self,
        variant_ranges: Sequence[tuple[int, int]],
        buffers: Sequence[U8Array],
    ) -> Iterator[tuple[int, int, U8Array]]:
        """Yield (start, stop, codes[stop - start, sample_count]) for each range, in order."""


@dataclass(frozen=True)
class ScoringModel:
    """One fitted model in store rows and signed-code units (see the module docstring).

    ``coefficients`` are the posterior-mean effects of the standardized columns at
    ``store_rows`` and ``posterior_draws`` [rows, K] are K exact posterior draws of
    the same effects, both tie-expanded. ``alpha`` are the covariate coefficients
    with the intercept first. ``predictive_intercept_shift`` calibrates the damped
    binary predictive (0.0 for quantitative models).
    """

    store_rows: I64Array
    signed_means: F64Array
    signed_scales: F64Array
    coefficients: F64Array
    posterior_draws: F64Array
    alpha: F64Array
    trait_type: TraitType
    predictive_intercept_shift: float

    def __post_init__(self) -> None:
        rows = np.asarray(self.store_rows)
        if rows.ndim != 1 or rows.dtype != np.int64:
            raise ValueError("store_rows must be a 1-D int64 array.")
        if rows.size and (rows[0] < 0 or np.any(np.diff(rows) <= 0)):
            raise ValueError("store_rows must be sorted, distinct and non-negative.")
        for name in ("signed_means", "signed_scales", "coefficients"):
            values = np.asarray(getattr(self, name))
            if values.shape != rows.shape or values.dtype != np.float64:
                raise ValueError(f"{name} must be float64 with one entry per store row.")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite.")
        if np.any(self.signed_scales <= 0.0):
            raise ValueError("signed_scales must be positive; inactive rows carry no model column.")
        draws = np.asarray(self.posterior_draws)
        if draws.ndim != 2 or draws.shape[0] != rows.shape[0] or draws.dtype != np.float64:
            raise ValueError("posterior_draws must be float64 [store rows, draws].")
        if not np.all(np.isfinite(draws)):
            raise ValueError("posterior_draws must be finite.")
        if self.trait_type == TraitType.BINARY and draws.shape[1] < MINIMUM_BINARY_POSTERIOR_DRAWS:
            raise ValueError(
                f"a binary model needs at least {MINIMUM_BINARY_POSTERIOR_DRAWS} posterior draws for its predictive."
            )
        alpha = np.asarray(self.alpha)
        if alpha.ndim != 1 or alpha.size < 1 or alpha.dtype != np.float64 or not np.all(np.isfinite(alpha)):
            raise ValueError("alpha must be a finite float64 vector with the intercept first.")

    @property
    def draw_count(self) -> int:
        return int(self.posterior_draws.shape[1])

    @classmethod
    def from_reduced_fit(
        cls,
        *,
        active_rows: I64Array,
        signed_means: F64Array,
        signed_scales: F64Array,
        tie_map: TieMap,
        member_prior_variances: F64Array,
        beta_reduced: F64Array,
        posterior_draws_reduced: F64Array,
        alpha: F64Array,
        trait_type: TraitType,
        predictive_intercept_shift: float,
    ) -> ScoringModel:
        """Expand a reduced-space fit to its active rows.

        ``tie_map`` is over the active rows. Each member of a tie group gets
        weight * sign * beta_group, with weights proportional to the members'
        prior variances (``TieMap.prior_variance_group_weights``); every posterior
        draw [reduced, K] expands with the same weights.
        """
        group_weights = tie_map.prior_variance_group_weights(np.asarray(member_prior_variances, dtype=np.float64))
        draws_reduced = np.asarray(posterior_draws_reduced, dtype=np.float64)
        beta = np.asarray(beta_reduced, dtype=np.float64)
        if draws_reduced.ndim != 2 or draws_reduced.shape[0] != beta.shape[0]:
            raise ValueError("posterior_draws_reduced must be [reduced coefficients, draws].")
        coefficients = np.asarray(tie_map.expand_coefficients(beta, group_weights), dtype=np.float64)
        draws = np.empty((coefficients.shape[0], draws_reduced.shape[1]), dtype=np.float64)
        for draw_index in range(draws_reduced.shape[1]):
            draws[:, draw_index] = tie_map.expand_coefficients(draws_reduced[:, draw_index], group_weights)
        return cls(
            store_rows=np.asarray(active_rows, dtype=np.int64),
            signed_means=np.asarray(signed_means, dtype=np.float64),
            signed_scales=np.asarray(signed_scales, dtype=np.float64),
            coefficients=coefficients,
            posterior_draws=draws,
            alpha=np.asarray(alpha, dtype=np.float64),
            trait_type=trait_type,
            predictive_intercept_shift=float(predictive_intercept_shift),
        )


@dataclass(frozen=True)
class ScoringPlan:
    """Every model's mean and draws packed as weight columns over the union of their store rows.

    Model m owns column ``mean_columns[m]`` and the draw columns
    ``draw_columns[m] = (start, stop)``. ``row_runs`` are the maximal runs of
    consecutive store rows, as (store_start, store_stop, plan_start); rows no model
    uses are never read.
    """

    store_rows: I64Array
    weights: F64Array
    offsets: F64Array
    mean_columns: tuple[int, ...]
    draw_columns: tuple[tuple[int, int], ...]
    row_runs: tuple[tuple[int, int, int], ...]

    @property
    def model_count(self) -> int:
        return len(self.mean_columns)

    @classmethod
    def from_models(cls, models: Sequence[ScoringModel]) -> ScoringPlan:
        if not models:
            raise ValueError("a scoring plan needs at least one model.")
        store_rows = np.unique(np.concatenate([model.store_rows for model in models]))
        column_count = sum(1 + model.draw_count for model in models)
        weights = np.zeros((store_rows.shape[0], column_count), dtype=np.float64)
        offsets = np.zeros(column_count, dtype=np.float64)
        mean_columns: list[int] = []
        draw_columns: list[tuple[int, int]] = []
        next_column = 0
        for model in models:
            positions = np.searchsorted(store_rows, model.store_rows)
            model_effects = np.column_stack([model.coefficients, model.posterior_draws])
            model_weights = model_effects / model.signed_scales[:, None]
            stop = next_column + model_effects.shape[1]
            weights[positions, next_column:stop] = model_weights
            offsets[next_column:stop] = -(model.signed_means @ model_weights)
            mean_columns.append(next_column)
            draw_columns.append((next_column + 1, stop))
            next_column = stop
        breaks = np.flatnonzero(np.diff(store_rows) != 1) + 1
        run_starts = np.concatenate([[0], breaks]).astype(np.int64)
        run_stops = np.concatenate([breaks, [store_rows.shape[0]]]).astype(np.int64)
        row_runs = tuple(
            (int(store_rows[start]), int(store_rows[stop - 1]) + 1, int(start))
            for start, stop in zip(run_starts, run_stops, strict=True)
            if stop > start
        )
        return cls(
            store_rows=store_rows,
            weights=weights,
            offsets=offsets,
            mean_columns=tuple(mean_columns),
            draw_columns=tuple(draw_columns),
            row_runs=row_runs,
        )

    def read_ranges(self, block_rows: int) -> list[tuple[int, int, int]]:
        """The row runs cut into reads of at most ``block_rows`` rows, as (store_start, store_stop, plan_start)."""
        ranges: list[tuple[int, int, int]] = []
        for store_start, store_stop, plan_start in self.row_runs:
            for piece_start in range(store_start, store_stop, block_rows):
                piece_stop = min(store_stop, piece_start + block_rows)
                ranges.append((piece_start, piece_stop, plan_start + piece_start - store_start))
        return ranges


@dataclass(frozen=True)
class GeneticScores:
    """Scores [samples, models]: posterior-mean genetic scores and their posterior variances.

    ``variances`` is NaN for a model without posterior draws.
    """

    means: F64Array
    variances: F64Array


def _block_rows(sample_count: int, budget: ComputeBudget, plan_rows: int) -> int:
    """Rows per read: the read-ahead ring and the conversion panels fit half the host budget,
    and on CUDA an uploaded block of codes fits its share of the smallest device."""
    bytes_per_row = _HOST_BYTES_PER_BLOCK_CELL * max(sample_count, 1)
    rows = max(1, int(budget.host_bytes) // (_HOST_MEMORY_SHARES * bytes_per_row))
    if budget.device_kind == "cuda":
        rows = min(rows, max(1, min(budget.device_bytes) // (_DEVICE_MEMORY_SHARES * max(sample_count, 1))))
    return min(rows, max(plan_rows, 1))


def _selected_codes(codes: U8Array, sample_indices: I64Array | None) -> U8Array:
    return codes if sample_indices is None else np.take(codes, sample_indices, axis=1)


def _cpu_panels(sample_count: int, worker_count: int) -> list[tuple[int, int]]:
    panel_width = max(1, -(-sample_count // (worker_count * _PANELS_PER_WORKER)))
    return [(start, min(sample_count, start + panel_width)) for start in range(0, sample_count, panel_width)]


def _score_cpu(
    blocks: Iterable[tuple[int, int, int, U8Array]],
    weights: F64Array,
    accumulator: F64Array,
    budget: ComputeBudget,
) -> None:
    """accumulator[columns, samples] += W_b' S_b for every block, samples split into owned panels."""
    worker_count = max(1, int(budget.cpu_threads))
    panels = _cpu_panels(accumulator.shape[1], worker_count)
    with ThreadPoolExecutor(max_workers=worker_count) as pool, threadpool_limits(limits=1, user_api="blas"):
        for plan_start, plan_stop, codes in _plan_blocks(blocks):
            block_weights = np.ascontiguousarray(weights[plan_start:plan_stop].T)

            def panel(column_range: tuple[int, int], codes: U8Array = codes, block_weights: F64Array = block_weights) -> None:
                start, stop = column_range
                signed = np.subtract(codes[:, start:stop], SIGNED_CODE_OFFSET, dtype=np.float64)
                accumulator[:, start:stop] += block_weights @ signed

            for _ in pool.map(panel, panels):
                pass


def _score_cuda(
    blocks: Iterable[tuple[int, int, int, U8Array]],
    weights: F64Array,
    accumulator: F64Array,
    budget: ComputeBudget,
) -> None:
    """Blocks round-robin over the visible devices; each device keeps its own fp64 accumulator."""
    cupy: Any = _try_import_cupy()
    if cupy is None:
        raise RuntimeError("the compute budget selected CUDA but CuPy is unavailable.")
    column_count, sample_count = accumulator.shape
    device_accumulators = []
    for device_id in budget.device_ids:
        with _cupy_device_context(cupy, device_id):
            device_accumulators.append(cupy.zeros((column_count, sample_count), dtype=cupy.float64))
    for block_position, (plan_start, plan_stop, codes) in enumerate(_plan_blocks(blocks)):
        device_position = block_position % len(budget.device_ids)
        with _cupy_device_context(cupy, budget.device_ids[device_position]):
            rows = plan_stop - plan_start
            tile_columns = max(
                1,
                int(budget.device_bytes[device_position]) // (_DEVICE_MEMORY_SHARES * 8 * max(rows, 1)),
            )
            block_weights = cupy.asarray(np.ascontiguousarray(weights[plan_start:plan_stop].T))
            device_codes = cupy.asarray(codes)
            for start in range(0, sample_count, tile_columns):
                stop = min(sample_count, start + tile_columns)
                signed = device_codes[:, start:stop].astype(cupy.float64)
                signed -= SIGNED_CODE_OFFSET
                device_accumulators[device_position][:, start:stop] += block_weights @ signed
    for device_position, device_id in enumerate(budget.device_ids):
        with _cupy_device_context(cupy, device_id):
            accumulator += cupy.asnumpy(device_accumulators[device_position])


def _plan_blocks(blocks: Iterable[tuple[int, int, int, U8Array]]) -> Iterator[tuple[int, int, U8Array]]:
    for store_start, store_stop, plan_start, codes in blocks:
        yield plan_start, plan_start + store_stop - store_start, codes


def _read_ahead_buffers(block_rows: int, sample_count: int, budget: ComputeBudget) -> list[U8Array]:
    """The read-ahead ring; page-locked on CUDA so uploads run at full bus speed."""
    shape = (block_rows, sample_count)
    if budget.device_kind != "cuda":
        return [np.empty(shape, dtype=np.uint8) for _ in range(_READ_AHEAD_BUFFERS)]
    cupy: Any = _try_import_cupy()
    if cupy is None:
        raise RuntimeError("the compute budget selected CUDA but CuPy is unavailable.")
    element_count = block_rows * sample_count
    return [
        np.frombuffer(cupy.cuda.alloc_pinned_memory(element_count), dtype=np.uint8, count=element_count).reshape(shape)
        for _ in range(_READ_AHEAD_BUFFERS)
    ]


def _read_plan_blocks(
    source: CodeBlockSource,
    ranges: Sequence[tuple[int, int, int]],
    buffers: Sequence[U8Array],
    sample_indices: I64Array | None,
) -> Iterator[tuple[int, int, int, U8Array]]:
    plan_starts = {(store_start, store_stop): plan_start for store_start, store_stop, plan_start in ranges}
    variant_ranges = [(store_start, store_stop) for store_start, store_stop, _ in ranges]
    for store_start, store_stop, codes in source.iter_code_blocks(variant_ranges, buffers):
        if codes.shape != (store_stop - store_start, source.sample_count):
            raise ValueError(f"codes for rows [{store_start}, {store_stop}) have shape {codes.shape}.")
        yield store_start, store_stop, plan_starts[(store_start, store_stop)], _selected_codes(codes, sample_indices)


def _validated_sample_indices(sample_indices: NDArray | None, sample_count: int) -> I64Array | None:
    if sample_indices is None:
        return None
    indices = np.asarray(sample_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0 or indices.min() < 0 or indices.max() >= sample_count:
        raise ValueError(f"sample_indices must be a non-empty 1-D array of store samples below {sample_count}.")
    return indices


def score_genetic(
    source: CodeBlockSource,
    plan: ScoringPlan,
    budget: ComputeBudget,
    sample_indices: NDArray | None = None,
) -> GeneticScores:
    """Posterior-mean genetic scores and their posterior variances for every planned model,
    from one read of the codes.

    ``sample_indices`` selects store samples (all of them when ``None``); scoring all
    samples once and slicing folds afterwards reads the store once for every fold.
    """
    selected = _validated_sample_indices(sample_indices, source.sample_count)
    sample_count = source.sample_count if selected is None else int(selected.shape[0])
    block_rows = _block_rows(source.sample_count, budget, int(plan.store_rows.shape[0]))
    ranges = plan.read_ranges(block_rows)
    log(
        f"  fast scoring: {plan.model_count} models ({plan.weights.shape[1]} weight columns) x {sample_count} "
        + f"samples over {plan.store_rows.shape[0]} store rows in {len(ranges)} reads of <= {block_rows} rows "
        + f"on {budget.describe()}"
    )
    accumulator = np.zeros((plan.weights.shape[1], sample_count), dtype=np.float64)
    blocks = _read_plan_blocks(source, ranges, _read_ahead_buffers(block_rows, source.sample_count, budget), selected)
    if budget.device_kind == "cuda":
        _score_cuda(blocks, plan.weights, accumulator, budget)
    else:
        _score_cpu(blocks, plan.weights, accumulator, budget)
    accumulator += plan.offsets[:, None]
    means = np.ascontiguousarray(accumulator[list(plan.mean_columns)].T)
    variances = np.full_like(means, np.nan)
    for model_index, (draw_start, draw_stop) in enumerate(plan.draw_columns):
        if draw_stop > draw_start:
            deviations = accumulator[draw_start:draw_stop] - accumulator[plan.mean_columns[model_index]]
            variances[:, model_index] = np.mean(deviations * deviations, axis=0)
    return GeneticScores(means=means, variances=variances)


def score_linear_predictor(
    genetic_scores: F64Array,
    covariates: F64Array,
    models: Sequence[ScoringModel],
) -> F64Array:
    """Linear predictors [samples, models]: intercept + covariates @ alpha + genetic score."""
    covariate_matrix = np.asarray(covariates, dtype=np.float64)
    if covariate_matrix.ndim != 2 or covariate_matrix.shape[0] != genetic_scores.shape[0]:
        raise ValueError("covariates must be [samples, covariates] aligned with the scores.")
    if genetic_scores.shape[1] != len(models):
        raise ValueError("genetic_scores must have one column per model.")
    linear_predictor = np.array(genetic_scores, dtype=np.float64)
    for model_index, model in enumerate(models):
        if model.alpha.shape[0] != covariate_matrix.shape[1] + 1:
            raise ValueError("each model's alpha must hold the intercept and one entry per covariate.")
        linear_predictor[:, model_index] += model.alpha[0] + covariate_matrix @ model.alpha[1:]
    return linear_predictor


def posterior_predictive_probability(
    linear_predictor: F64Array,
    predictor_variance: F64Array,
    intercept_shift: float,
) -> F64Array:
    """P(y = 1) = E[sigmoid(eta + c + sqrt(v) Z)] by 64-point Gauss-Hermite quadrature."""
    eta = np.asarray(linear_predictor, dtype=np.float64) + float(intercept_shift)
    variance = np.asarray(predictor_variance, dtype=np.float64)
    if eta.shape != variance.shape:
        raise ValueError("linear_predictor and predictor_variance must have the same shape.")
    if not np.all(np.isfinite(variance)) or np.any(variance < 0.0):
        raise ValueError("predictor_variance must be finite and non-negative.")
    spread = np.sqrt(2.0 * variance)[..., None]
    return (expit(eta[..., None] + spread * _HERMITE_NODES) @ _HERMITE_WEIGHTS) / np.sqrt(np.pi)


def predictive_intercept_shift(
    linear_predictor: F64Array,
    predictor_variance: F64Array,
    targets: F64Array,
) -> float:
    """The shift c with mean_i P(y_i = 1) = prevalence on the given training samples.

    The mean predictive rises strictly with c from 0 to 1, so the root is unique;
    any fixed subsample of the training set estimates c with an error that shrinks
    with its size.
    """
    outcome = np.asarray(targets, dtype=np.float64)
    if not np.all((outcome == 0.0) | (outcome == 1.0)):
        raise ValueError("targets must be 0/1.")
    prevalence = float(np.mean(outcome))
    if not 0.0 < prevalence < 1.0:
        raise ValueError("the training targets need both cases and controls.")

    def excess(shift: float) -> float:
        return float(np.mean(posterior_predictive_probability(linear_predictor, predictor_variance, shift))) - prevalence

    lower, upper = -1.0, 1.0
    while excess(lower) > 0.0:
        lower *= 2.0
    while excess(upper) < 0.0:
        upper *= 2.0
    return float(brentq(excess, lower, upper, xtol=1e-14, rtol=8.9e-16))
