"""One-pass scoring of every fitted model over the uint8 dosage store.

Model. A fitted model is a posterior over standardized genotype columns
x_j = (s_j - mu_j) / sigma_j of the signed dosage codes s = code - 127, with
mu_j and sigma_j the training-sample mean and population SD (ddof 0). Its
genetic score is

    g_i = sum_j beta_j x_ij = sum_j w_j s_ij + offset,
    w_j = beta_j / sigma_j,   offset = -sum_j w_j mu_j,

so a model is one column of a weight matrix W and one offset. Every model
(traits x folds; each fold has its own active rows and training moments) is
scored by ONE read of the codes: each block of store rows adds W_b' S_b to a
[models, samples] accumulator. The read is one byte per genotype against
2 * models flops, so the pass is bound by the read at any realistic model count.

Exactness. Codes are small integers, exact in fp64, and the weights and the
accumulator are fp64, so a score equals X beta to fp64 rounding: in-sample
scores reproduce the fitted model's linear predictor. On the CPU each sample
column is owned by one worker that adds the blocks in store order with
single-threaded BLAS, so for a given block size the result does not depend on
the thread count.

Binary models. The posterior predictive of a binary model damps its logit by
kappa_i = sqrt(1 + (pi / 8) s2_i), with s2_i the posterior variance of the
linear predictor. Under the fit's block-structured posterior (full covariance
S_b inside each LD block) that variance is exactly sum_b x_ib' S_b x_ib =
sum_b ||L_b^-1 x_ib||^2 for A_b = L_b L_b' the block precision, which
``block_predictor_variance`` evaluates from the fit's block factors.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Protocol, Sequence

import numpy as np
from scipy.linalg import solve_triangular
from threadpoolctl import threadpool_limits

from sv_pgs._typing import F64Array, I64Array, NDArray, U8Array
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import TraitType
from sv_pgs.data import TieMap
from sv_pgs.genotype import _cupy_device_context, _try_import_cupy
from sv_pgs.inference import VariationalFitResult
from sv_pgs.mixture_inference import _calibrate_binary_intercept
from sv_pgs.model import _tie_group_export_weights
from sv_pgs.numeric import logistic_normal_probit_scale, stable_sigmoid
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
    ``store_rows``, tie-expanded; ``alpha`` are the covariate coefficients with the
    intercept first. ``predictive_intercept_shift`` calibrates the damped binary
    predictive (0.0 when the fit computed no posterior variances).
    """

    store_rows: I64Array
    signed_means: F64Array
    signed_scales: F64Array
    coefficients: F64Array
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
        alpha = np.asarray(self.alpha)
        if alpha.ndim != 1 or alpha.size < 1 or alpha.dtype != np.float64 or not np.all(np.isfinite(alpha)):
            raise ValueError("alpha must be a finite float64 vector with the intercept first.")

    @classmethod
    def from_reduced_fit(
        cls,
        *,
        active_rows: I64Array,
        signed_means: F64Array,
        signed_scales: F64Array,
        tie_map: TieMap,
        fit_result: VariationalFitResult,
        beta_reduced: F64Array,
        trait_type: TraitType,
    ) -> ScoringModel:
        """Expand a reduced-space fit to its active rows with the fit's own tie convention.

        ``tie_map`` is over the active rows; each member of a tie group gets
        weight * sign * beta_group (``TieMap.expand_coefficients`` with the prior-variance
        weights of ``_tie_group_export_weights``), exactly as ``BayesianPGS.fit`` exports.
        ``beta_reduced`` is the fp64 posterior mean the fit ends with.
        """
        coefficients = np.asarray(
            tie_map.expand_coefficients(
                beta_reduced,
                group_weights=_tie_group_export_weights(tie_map=tie_map, fit_result=fit_result),
            ),
            dtype=np.float64,
        )
        return cls(
            store_rows=np.asarray(active_rows, dtype=np.int64),
            signed_means=np.asarray(signed_means, dtype=np.float64),
            signed_scales=np.asarray(signed_scales, dtype=np.float64),
            coefficients=coefficients,
            alpha=np.asarray(fit_result.alpha, dtype=np.float64),
            trait_type=trait_type,
            predictive_intercept_shift=float(fit_result.predictive_intercept_shift),
        )


@dataclass(frozen=True)
class ScoringPlan:
    """Every model packed as one weight column over the union of their store rows.

    ``row_runs`` are the maximal runs of consecutive store rows, as
    (store_start, store_stop, plan_start); rows no model uses are never read.
    """

    store_rows: I64Array
    weights: F64Array
    offsets: F64Array
    row_runs: tuple[tuple[int, int, int], ...]

    @property
    def model_count(self) -> int:
        return int(self.weights.shape[1])

    @classmethod
    def from_models(cls, models: Sequence[ScoringModel]) -> ScoringPlan:
        if not models:
            raise ValueError("a scoring plan needs at least one model.")
        store_rows = np.unique(np.concatenate([model.store_rows for model in models]))
        weights = np.zeros((store_rows.shape[0], len(models)), dtype=np.float64)
        offsets = np.zeros(len(models), dtype=np.float64)
        for model_index, model in enumerate(models):
            positions = np.searchsorted(store_rows, model.store_rows)
            model_weights = model.coefficients / model.signed_scales
            weights[positions, model_index] = model_weights
            offsets[model_index] = -float(model_weights @ model.signed_means)
        breaks = np.flatnonzero(np.diff(store_rows) != 1) + 1
        run_starts = np.concatenate([[0], breaks]).astype(np.int64)
        run_stops = np.concatenate([breaks, [store_rows.shape[0]]]).astype(np.int64)
        row_runs = tuple(
            (int(store_rows[start]), int(store_rows[stop - 1]) + 1, int(start))
            for start, stop in zip(run_starts, run_stops, strict=True)
            if stop > start
        )
        return cls(store_rows=store_rows, weights=weights, offsets=offsets, row_runs=row_runs)

    def read_ranges(self, block_rows: int) -> list[tuple[int, int, int]]:
        """The row runs cut into reads of at most ``block_rows`` rows, as (store_start, store_stop, plan_start)."""
        ranges: list[tuple[int, int, int]] = []
        for store_start, store_stop, plan_start in self.row_runs:
            for piece_start in range(store_start, store_stop, block_rows):
                piece_stop = min(store_stop, piece_start + block_rows)
                ranges.append((piece_start, piece_stop, plan_start + piece_start - store_start))
        return ranges


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
    """accumulator[models, samples] += W_b' S_b for every block, samples split into owned panels."""
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
    model_count, sample_count = accumulator.shape
    device_accumulators = []
    for device_id in budget.device_ids:
        with _cupy_device_context(cupy, device_id):
            device_accumulators.append(cupy.zeros((model_count, sample_count), dtype=cupy.float64))
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


def _read_plan_blocks(
    source: CodeBlockSource,
    ranges: Sequence[tuple[int, int, int]],
    block_rows: int,
    sample_indices: I64Array | None,
) -> Iterator[tuple[int, int, int, U8Array]]:
    buffers = [np.empty((block_rows, source.sample_count), dtype=np.uint8) for _ in range(_READ_AHEAD_BUFFERS)]
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
) -> F64Array:
    """Genetic scores [samples, models] of every planned model from one read of the codes.

    ``sample_indices`` selects store samples (all of them when ``None``); scoring all
    samples once and slicing folds afterwards reads the store once for every fold.
    """
    selected = _validated_sample_indices(sample_indices, source.sample_count)
    sample_count = source.sample_count if selected is None else int(selected.shape[0])
    block_rows = _block_rows(source.sample_count, budget, int(plan.store_rows.shape[0]))
    ranges = plan.read_ranges(block_rows)
    log(
        f"  fast scoring: {plan.model_count} models x {sample_count} samples over "
        + f"{plan.store_rows.shape[0]} store rows in {len(ranges)} reads of <= {block_rows} rows "
        + f"on {budget.describe()}"
    )
    accumulator = np.zeros((plan.model_count, sample_count), dtype=np.float64)
    blocks = _read_plan_blocks(source, ranges, block_rows, selected)
    if budget.device_kind == "cuda":
        _score_cuda(blocks, plan.weights, accumulator, budget)
    else:
        _score_cpu(blocks, plan.weights, accumulator, budget)
    accumulator += plan.offsets[:, None]
    return np.ascontiguousarray(accumulator.T)


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


@dataclass(frozen=True)
class BlockPosterior:
    """One LD block of a model's posterior: its reduced columns and precision factor.

    ``store_rows`` are the representative store rows of the block's reduced
    coefficients, in the order of ``precision_factor``, the lower Cholesky factor of
    the block precision A_b (posterior covariance S_b = A_b^-1), with the training
    moments of those rows.
    """

    store_rows: I64Array
    signed_means: F64Array
    signed_scales: F64Array
    precision_factor: F64Array

    def __post_init__(self) -> None:
        rows = np.asarray(self.store_rows)
        if rows.ndim != 1 or rows.dtype != np.int64 or rows.size == 0:
            raise ValueError("store_rows must be a non-empty 1-D int64 array.")
        if rows[0] < 0 or np.any(np.diff(rows) <= 0):
            raise ValueError("store_rows must be sorted, distinct and non-negative.")
        if self.precision_factor.shape != (rows.size, rows.size):
            raise ValueError("precision_factor must be square over the block's rows.")
        if np.asarray(self.signed_means).shape != rows.shape or np.asarray(self.signed_scales).shape != rows.shape:
            raise ValueError("signed_means and signed_scales need one entry per block row.")


def block_predictor_variance(
    source: CodeBlockSource,
    blocks: Iterable[BlockPosterior],
    sample_indices: NDArray | None = None,
) -> F64Array:
    """Posterior variance of one model's linear predictor, sum_b ||L_b^-1 x_ib||^2, per sample.

    Each block reads its rows' codes once; the cost is p_b^2 per sample and block, so
    it is meant for the samples whose predictive probabilities are needed.
    """
    selected = _validated_sample_indices(sample_indices, source.sample_count)
    sample_count = source.sample_count if selected is None else int(selected.shape[0])
    variance = np.zeros(sample_count, dtype=np.float64)
    for block in blocks:
        row_start, row_stop = int(block.store_rows[0]), int(block.store_rows[-1]) + 1
        buffers = [np.empty((row_stop - row_start, source.sample_count), dtype=np.uint8) for _ in range(2)]
        for _, _, codes in source.iter_code_blocks([(row_start, row_stop)], buffers):
            block_codes = _selected_codes(codes[block.store_rows - row_start], selected)
            standardized = (
                np.subtract(block_codes, SIGNED_CODE_OFFSET, dtype=np.float64) - block.signed_means[:, None]
            ) / block.signed_scales[:, None]
            whitened = solve_triangular(block.precision_factor, standardized, lower=True, check_finite=False)
            variance += np.einsum("ij,ij->j", whitened, whitened)
    return variance


def predictive_intercept_shift(
    linear_predictor: F64Array,
    predictor_variance: F64Array,
    targets: F64Array,
) -> float:
    """Shift c with mean sigmoid((eta_i + c) / kappa_i) = prevalence on the given training samples.

    The fitted intercept anchors the plug-in predictive; the damped one needs its own
    anchor (``_calibrate_binary_intercept`` with the probit scale). Any fixed subsample
    of the training set gives an estimate of c whose error shrinks with its size.
    """
    return _calibrate_binary_intercept(
        linear_predictor=linear_predictor,
        targets=targets,
        predictor_scale=logistic_normal_probit_scale(predictor_variance),
    )


def posterior_predictive_probability(
    linear_predictor: F64Array,
    predictor_variance: F64Array,
    intercept_shift: float,
) -> F64Array:
    """P(y = 1) = sigmoid((eta + c) / sqrt(1 + (pi / 8) s2)), the calibrated damped predictive."""
    damped = (np.asarray(linear_predictor, dtype=np.float64) + float(intercept_shift)) / logistic_normal_probit_scale(
        predictor_variance
    )
    return np.asarray(stable_sigmoid(damped), dtype=np.float64)
