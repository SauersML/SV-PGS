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

Posterior draws. A model carries K draws beta^(k) of its effects from the law
its fitting route represents the posterior by, which is not the same law for
every route. Stage 2's full-data route draws exactly from its Gaussian
posterior q by perturb-and-solve on its own genotype passes. The small-n
mean-field route (``mean_field.MeanFieldFixedPoints.draws``) draws instead
from the fitted product approximation prod_j q_j, conditional on the fitted
hyperparameters: a direction that mixes members in LD has a variance under
the product that is neither an upper nor a lower bound on the posterior's,
and hyperparameter uncertainty is left out. Everything below is a statement
about the law the draws come from, so it is a posterior statement exactly
where that law is the posterior.

Each draw is one more weight column, so the same single read gives
the draw scores g_i^(k). The mean score g_i is known exactly, so

    v_i = (1 / K) sum_k (g_i^(k) - g_i)^2

is an unbiased estimate of the variance of the genetic score under that law,
with relative standard error sqrt(2 / K). That moves a damped probability by at
most |sigmoid''| / 2 * sqrt(2 / K) * v_i <= 0.068 v_i / sqrt(K). One draw
already makes it unbiased. An interval needs more care: where the drawing law
is Gaussian, K v_i is v times a chi-square with K degrees of freedom,
independent of the genetic value, so (G_i - g_i) / sqrt(v_i) is exactly
Student-t with K degrees of freedom and ``GeneticScores.credible_interval``
uses t_K quantiles. Normal quantiles would under-cover at small K. Under a
non-Gaussian drawing law (the mean-field product's score is a sum of
independent scale mixtures) the t_K interval is an approximation, not an
exact coverage statement, for that law or for the posterior. The covariate
coefficients have a flat prior and an O(1/n) posterior variance, which the
predictive ignores.

Binary models. The posterior predictive is

    P(y_i = 1) = E[sigmoid(eta_i + c + sqrt(v_i) Z)],   Z ~ N(0, 1),

evaluated by the trapezoid rule in z with a step and a truncation derived
a priori from the integrand's strip of analyticity, so it is exact to fp64
for every variance with no iteration (see ``_trapezoid_rule``). The
shift c anchors the mean predictive on the training prevalence; the fitted
intercept anchors the plug-in predictor, and the damped one needs its own
anchor.

Memory. Every block size is the exact solution of a memory plan: the fixed
allocations (weights, accumulator) are subtracted from the budget and the
block's own buffers take the rest; on CUDA the block rows and the sample tile
width together minimize the kernel launches within the device memory.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
from typing import Any, Iterable, Iterator, Protocol, Sequence

import numpy as np
from scipy import stats
from scipy.optimize import brentq
from scipy.special import expit
from threadpoolctl import threadpool_limits

from sv_pgs._typing import F64Array, I64Array, NDArray, U8Array
from sv_pgs.compute_budget import ComputeBudget, _cupy_device_context, _try_import_cupy
from sv_pgs.config import TraitType
from sv_pgs.data import TieMap
from sv_pgs.progress import log

SIGNED_CODE_OFFSET = 127.0
_READ_AHEAD_BUFFERS = 3
_FLOAT64_BYTES = 8
_EPSILON = float(np.finfo(np.float64).eps)
_INVERSE_ROOT_TWO_PI = 1.0 / math.sqrt(2.0 * math.pi)


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
    ``store_rows`` and ``posterior_draws`` [rows, K] are K draws of the same effects
    from the law the fitting route represents the posterior by (the module docstring:
    the full-data route's exact posterior draws, or the mean-field route's conditional
    variational draws of its product approximation), both tie-expanded.
    ``alpha`` are the covariate coefficients
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
        if self.trait_type == TraitType.BINARY and draws.shape[1] == 0:
            raise ValueError("a binary model needs posterior draws for its predictive.")
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
    """Scores [samples, models]: posterior-mean genetic scores and their variances under each model's drawing law.

    ``variances`` are the K-draw estimates v_i (NaN for a model without posterior draws) and
    ``draw_counts`` holds each model's K. The drawing law is the posterior only on the full-data
    route; on the mean-field route it is that fit's product approximation (the module docstring).
    """

    means: F64Array
    variances: F64Array
    draw_counts: tuple[int, ...]

    def credible_interval(self, coverage: float) -> tuple[F64Array, F64Array]:
        """Central ``coverage`` interval of every sample's genetic value under the model's drawing law,
        exact for any K where that law is Gaussian.

        The mean score is exact and the K draws are independent draws of it, so K v_i / v is
        chi-square with K degrees of freedom and independent of G ~ N(g, v); (G - g) / sqrt(v_i) is
        then Student-t with K degrees of freedom, and the interval uses its quantiles. Under a
        non-Gaussian drawing law (the mean-field product's) G is not normal and the interval is an
        approximation; it covers the posterior only where the drawing law is the posterior
        (the module docstring).
        """
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must lie strictly between 0 and 1.")
        counts = np.asarray(self.draw_counts, dtype=np.int64)
        if np.any(counts == 0):
            raise ValueError("a model without posterior draws has no credible interval.")
        half_width = stats.t.isf(0.5 * (1.0 - coverage), counts)[None, :] * np.sqrt(self.variances)
        return self.means - half_width, self.means + half_width


def _host_bytes(plan: ScoringPlan, store_samples: int, selected_samples: int, rows: int, device_kind: str) -> int:
    """Host bytes of one scoring pass with reads of ``rows`` rows: the plan's weights, the
    accumulator, the read-ahead ring, the selected-sample copy of a block, the transposed
    block weights and, on the CPU, every panel's fp64 codes and product."""
    columns = int(plan.weights.shape[1])
    fixed = _FLOAT64_BYTES * (int(plan.weights.size) + columns * selected_samples)
    per_row = _READ_AHEAD_BUFFERS * store_samples + _FLOAT64_BYTES * columns
    if selected_samples != store_samples:
        per_row += selected_samples
    if device_kind == "cpu":
        fixed += _FLOAT64_BYTES * columns * selected_samples
        per_row += _FLOAT64_BYTES * selected_samples
    return fixed + rows * per_row


def _block_rows(plan: ScoringPlan, store_samples: int, selected_samples: int, budget: ComputeBudget) -> int:
    """The most rows per read whose host plan fits the budget, and on CUDA the device plan."""
    fixed = _host_bytes(plan, store_samples, selected_samples, 0, budget.device_kind)
    per_row = _host_bytes(plan, store_samples, selected_samples, 1, budget.device_kind) - fixed
    rows = (int(budget.host_bytes) - fixed) // per_row
    if rows < 1:
        raise MemoryError(
            f"scoring {plan.weights.shape[1]} weight columns x {selected_samples} samples needs "
            f"{(fixed + per_row) / 1e9:.2f} GB of host memory; the budget holds {budget.host_bytes / 1e9:.2f} GB"
        )
    if budget.device_kind == "cuda":
        rows = min(rows, min(_device_tile(int(plan.weights.shape[1]), selected_samples, device_bytes)[0]
                             for device_bytes in budget.device_bytes))
    return min(rows, int(plan.store_rows.shape[0]))


def _device_tile(columns: int, samples: int, device_bytes: int) -> tuple[int, int]:
    """Block rows r and sample tile width t that minimize the kernel launches (R / r)(S / t).

    The device holds the accumulator (8 C S bytes) and per block its codes (r S), its weights
    (8 r C) and per tile the fp64 codes and product (8 t (r + C)). Maximizing r t under
    r (S + 8 C) + 8 t (r + C) <= M gives r = sqrt(C^2 + M C / (S + 8 C)) - C, then the widest t.
    """
    memory = device_bytes - _FLOAT64_BYTES * columns * samples
    per_block_row = samples + _FLOAT64_BYTES * columns
    if memory < per_block_row + _FLOAT64_BYTES * (1 + columns):
        raise MemoryError(
            f"a device of {device_bytes / 1e9:.2f} GB cannot hold the {columns} x {samples} score accumulator "
            "and one block row"
        )
    rows = max(1, int(math.sqrt(columns * columns + memory * columns / per_block_row) - columns))
    rows = min(rows, (memory - _FLOAT64_BYTES * (1 + columns)) // per_block_row)
    tile = (memory - rows * per_block_row) // (_FLOAT64_BYTES * (rows + columns))
    return rows, min(tile, samples)


def _selected_codes(codes: U8Array, sample_indices: I64Array | None) -> U8Array:
    return codes if sample_indices is None else np.take(codes, sample_indices, axis=1)


def _cpu_panels(sample_count: int, worker_count: int) -> list[tuple[int, int]]:
    """One panel per worker, widths differing by at most one sample (equal work per sample)."""
    edges = [index * sample_count // worker_count for index in range(worker_count + 1)]
    return [(start, stop) for start, stop in zip(edges[:-1], edges[1:]) if stop > start]


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
            tile_columns = _device_tile(column_count, sample_count, int(budget.device_bytes[device_position]))[1]
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
    block_rows = _block_rows(plan, source.sample_count, sample_count, budget)
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
    draw_counts = tuple(draw_stop - draw_start for draw_start, draw_stop in plan.draw_columns)
    return GeneticScores(means=means, variances=variances, draw_counts=draw_counts)


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


_TRUNCATION = float(stats.norm.isf(0.5 * _EPSILON))
"""|z| beyond which the standard normal holds eps of mass in its two tails."""
_WIDEST_STRIP = math.sqrt(2.0 * math.log(2.0 / _EPSILON))
"""The strip half-width a at which the step 2 pi a / ln(1 + 2 e^(a^2/2) / eps) is widest."""


def _trapezoid_rule(variance: F64Array) -> tuple[F64Array, I64Array]:
    """Step h and half-width K (nodes k h for |k| <= K) of the trapezoid rule for E[sigmoid(eta + sqrt(v) Z)].

    In z the integrand f(z) = sigmoid(eta + sqrt(v) z) phi(z) is analytic in |Im z| < pi / sqrt(v), since
    sigmoid's poles sit at Im w = +-pi. On |Im z| <= a with a <= pi / (2 sqrt(v)), Re e^-w >= 0 gives
    |sigmoid| <= 1, and the integral of |phi(x + i y)| over x is e^(y^2/2), so M = e^(a^2/2). The infinite
    trapezoid sum with step h then errs by at most 2 M / (e^(2 pi a / h) - 1) (Trefethen and Weideman 2014,
    Theorem 5.1), which is eps at h = 2 pi a / ln(1 + 2 e^(a^2/2) / eps). The strip is
    a = min(pi / (2 sqrt(v)), _WIDEST_STRIP). Every omitted node k > K has h phi(k h) below the integral
    of phi over the step before it, and sigmoid <= 1, so the nodes beyond K h >= _TRUNCATION omit at most
    eps. The result is within 2 eps plus the rounding of its sum of the exact predictive.
    """
    with np.errstate(divide="ignore"):
        strip = np.minimum(np.pi / (2.0 * np.sqrt(variance)), _WIDEST_STRIP)
    step = 2.0 * np.pi * strip / np.log1p(2.0 * np.exp(0.5 * strip * strip) / _EPSILON)
    return step, np.ceil(_TRUNCATION / step).astype(np.int64)


def posterior_predictive_probability(
    linear_predictor: F64Array,
    predictor_variance: F64Array,
    intercept_shift: float,
) -> F64Array:
    """P(y = 1) = E[sigmoid(eta + c + sqrt(v) Z)] by the a-priori trapezoid rule of ``_trapezoid_rule``.

    Entries are processed in order of their node count, so the work is the total node count and the
    memory is linear in the entries whatever their variances.
    """
    eta = np.asarray(linear_predictor, dtype=np.float64) + float(intercept_shift)
    variance = np.asarray(predictor_variance, dtype=np.float64)
    if eta.shape != variance.shape:
        raise ValueError("linear_predictor and predictor_variance must have the same shape.")
    if not np.all(np.isfinite(variance)) or np.any(variance < 0.0):
        raise ValueError("predictor_variance must be finite and non-negative.")
    step, half_width = _trapezoid_rule(variance.ravel())
    order = np.argsort(half_width, kind="stable")
    sorted_half_width = half_width[order]
    sorted_eta = eta.ravel()[order]
    sorted_spread = np.sqrt(variance.ravel())[order]
    sorted_step = step[order]
    total = sorted_step * _INVERSE_ROOT_TWO_PI * expit(sorted_eta)
    node_count = int(sorted_half_width[-1]) if total.size else 0
    for node in range(1, node_count + 1):
        active = slice(int(np.searchsorted(sorted_half_width, node)), None)
        offset = node * sorted_step[active]
        weight = sorted_step[active] * _INVERSE_ROOT_TWO_PI * np.exp(-0.5 * offset * offset)
        shift = sorted_spread[active] * offset
        total[active] += weight * (expit(sorted_eta[active] + shift) + expit(sorted_eta[active] - shift))
    if not np.all(np.isfinite(total)):
        raise ValueError("the posterior predictive is not finite: the linear predictor holds NaN.")
    probability = np.empty_like(total)
    probability[order] = total
    return probability.reshape(eta.shape)


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

    # sigmoid' <= 1/4, so a shift resolved to 4 eps min(p, 1 - p) leaves the mean predictive
    # within fp64 resolution of the prevalence; rtol is the smallest brentq accepts.
    eps = float(np.finfo(np.float64).eps)
    lower, upper = -1.0, 1.0
    while excess(lower) > 0.0:
        lower *= 2.0
    while excess(upper) < 0.0:
        upper *= 2.0
    return float(brentq(excess, lower, upper, xtol=4.0 * eps * min(prevalence, 1.0 - prevalence), rtol=4.0 * eps))
