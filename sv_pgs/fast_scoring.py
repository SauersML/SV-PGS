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

Each draw is one more weight column, so the same read gives the draw
scores g_i^(k) (every model of a batch; a column batch is one read). The
mean score g_i is known exactly, so

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
predictive ignores; the mixture posteriors' intervals use empirical draw quantiles, and
the predictive carries the conditional covariate means for these same draws and the conditional covariate
covariance, preserving their dependence.

Binary models. The posterior predictive is

    P(y_i = 1) = E[sigmoid(eta_i + c + sqrt(v_i) Z)],   Z ~ N(0, 1),

evaluated by the trapezoid rule in z with a step and a truncation derived
a priori from the integrand's strip of analyticity, so it is exact to fp64
for every variance with no iteration (see ``_trapezoid_rule``). The
shift c anchors the mean predictive on the training prevalence; the fitted
intercept anchors the plug-in predictor, and the damped one needs its own
anchor.

Memory. No array of the size (store rows x columns) is ever formed: a model's
draws are a law read a tile of rows at a time (``draw_laws``), and each read
block's weight matrix (its rows x the batch's columns) is built from the
models' coefficients and their draws' tiles for that block alone, then
dropped. The output is scored in batches of whole models whose accumulator
(columns x samples) fits: the widest batch that fits, since every batch past
the first costs another read of the store, the dominant cost; the block rows
take what the batch leaves. Mean prediction (``draws="none"``) scores the mean
columns alone and generates no draw. Every buffer is charged to the shared
ledger (``memory_broker``) before it is allocated: the accumulator and the
per-block buffers on the host (page-locked read buffers as pinned), and on CUDA
each device's accumulator and tile buffers on its own pool; on CUDA the block
rows and the sample tile width together minimize the kernel launches within
the device memory that remains.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable, Iterator, Protocol, Sequence

import numpy as np
from scipy import stats
from scipy.optimize import brentq
from scipy.special import expit
from threadpoolctl import threadpool_limits

from sv_pgs._typing import F64Array, I64Array, NDArray, U8Array
from sv_pgs.compute_budget import ComputeBudget, _cupy_device_context, _try_import_cupy
from sv_pgs.config import TraitType
from sv_pgs.data import TieMap
from sv_pgs.draw_laws import DenseDraws, DrawLaw
from sv_pgs.memory_broker import HOST, PINNED, broker_for, device_pool
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
    ``store_rows`` and ``posterior_draws`` is the law of K draws of the same effects
    (``draw_laws``: [rows, K], read a tile of rows at a time) from the law the fitting
    route represents the posterior by (the module docstring: the full-data route's
    exact posterior draws, or the mean-field route's conditional variational draws of
    its product approximation), both tie-expanded. A matrix given here is held as
    ``draw_laws.DenseDraws``.
    ``alpha`` are the covariate coefficients
    with the intercept first. ``predictive_intercept_shift`` calibrates the damped
    binary predictive (0.0 for quantitative models).
    """

    store_rows: I64Array
    signed_means: F64Array
    signed_scales: F64Array
    coefficients: F64Array
    posterior_draws: DrawLaw
    alpha: F64Array
    trait_type: TraitType
    predictive_intercept_shift: float
    covariate_draws: F64Array
    covariate_covariance: F64Array
    gaussian_posterior: bool

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
        if isinstance(self.posterior_draws, np.ndarray):
            values = self.posterior_draws
            if values.ndim != 2 or values.dtype != np.float64:
                raise ValueError("posterior_draws must be float64 [store rows, draws].")
            object.__setattr__(self, "posterior_draws", DenseDraws(values))
        draws = self.posterior_draws
        if not hasattr(draws, "tile") or len(draws.shape) != 2 or draws.shape[0] != rows.shape[0]:
            raise ValueError("posterior_draws must be a draw law (or float64 matrix) of [store rows, draws].")
        if self.trait_type == TraitType.BINARY and draws.shape[1] == 0:
            raise ValueError("a binary model needs posterior draws for its predictive.")
        alpha = np.asarray(self.alpha)
        if alpha.ndim != 1 or alpha.size < 1 or alpha.dtype != np.float64 or not np.all(np.isfinite(alpha)):
            raise ValueError("alpha must be a finite float64 vector with the intercept first.")
        if self.covariate_draws.shape != (alpha.size, draws.shape[1]) or not np.all(np.isfinite(self.covariate_draws)):
            raise ValueError("covariate_draws must be finite [covariates, draws] conditional means paired with posterior_draws.")
        if self.covariate_covariance.shape != (alpha.size, alpha.size) or not np.all(np.isfinite(self.covariate_covariance)):
            raise ValueError("covariate_covariance must be finite [covariates, covariates].")

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
        posterior_draws_reduced: F64Array | DrawLaw,
        alpha: F64Array,
        trait_type: TraitType,
        predictive_intercept_shift: float,
        covariate_draws: F64Array,
        covariate_covariance: F64Array,
        gaussian_posterior: bool,
    ) -> ScoringModel:
        """Expand a reduced-space fit to its active rows.

        ``tie_map`` is over the active rows. Each member of a tie group gets
        weight * sign * beta_group, with weights proportional to the members'
        prior variances (``TieMap.prior_variance_group_weights``); every posterior
        draw [reduced, K] expands with the same weights.
        """
        group_weights = tie_map.prior_variance_group_weights(np.asarray(member_prior_variances, dtype=np.float64))
        law = posterior_draws_reduced if hasattr(posterior_draws_reduced, "tile") else DenseDraws(np.asarray(posterior_draws_reduced, dtype=np.float64))
        beta = np.asarray(beta_reduced, dtype=np.float64)
        if len(law.shape) != 2 or law.shape[0] != beta.shape[0]:
            raise ValueError("posterior_draws_reduced must be [reduced coefficients, draws].")
        coefficients = np.asarray(tie_map.expand_coefficients(beta, group_weights), dtype=np.float64)
        identity = (
            not tie_map.reduced_to_group and tie_map.original_to_reduced.shape[0] == law.shape[0]
            and np.array_equal(tie_map.kept_indices, np.arange(law.shape[0]))
        )
        if identity:
            # Every member is its own effect: the law is the members' already, held as it is (a p x K copy is the
            # scoring route's largest array at biobank scale).
            draws = law
        else:
            # Expanding groups to members mixes rows, so the law is drawn whole here: the routes with ties to expand
            # (the full-data EP route and the small-n routes) hold their draws as a matrix already.
            reduced = np.asarray(law, dtype=np.float64)
            expanded = np.empty((coefficients.shape[0], law.shape[1]), dtype=np.float64)
            for draw_index in range(law.shape[1]):
                expanded[:, draw_index] = tie_map.expand_coefficients(reduced[:, draw_index], group_weights)
            draws = DenseDraws(expanded)
        return cls(
            store_rows=np.asarray(active_rows, dtype=np.int64),
            signed_means=np.asarray(signed_means, dtype=np.float64),
            signed_scales=np.asarray(signed_scales, dtype=np.float64),
            coefficients=coefficients,
            posterior_draws=draws,
            alpha=np.asarray(alpha, dtype=np.float64),
            trait_type=trait_type,
            predictive_intercept_shift=float(predictive_intercept_shift),
            covariate_draws=np.asarray(covariate_draws, dtype=np.float64),
            covariate_covariance=np.asarray(covariate_covariance, dtype=np.float64),
            gaussian_posterior=gaussian_posterior,
        )


DRAW_MODES = ("none", "variance", "keep")
"""What ``score_genetic`` does with the draws: nothing (mean prediction: no draw is generated or scored), their
variance around the mean score, or that and every draw's score kept (``GeneticScores.draws``)."""


@dataclass(frozen=True)
class ScoringPlan:
    """Every model to score over the union of their store rows, with no weight matrix: each read block's weights are
    formed for that block alone (``block_weights``).

    ``positions[m]`` are model m's rows' places in ``store_rows``; ``row_runs`` are the maximal runs of consecutive
    store rows, as (store_start, store_stop, plan_start); rows no model uses are never read. ``mean_offsets[m]`` is
    the mean score's offset -sum_j mu_j beta_j / sigma_j (a draw's offset is summed block by block as its weights are
    formed).
    """

    models: tuple[ScoringModel, ...]
    store_rows: I64Array
    positions: tuple[I64Array, ...]
    mean_offsets: F64Array
    row_runs: tuple[tuple[int, int, int], ...]
    gaussian_posteriors: tuple[bool, ...]

    @property
    def model_count(self) -> int:
        return len(self.models)

    def draw_count(self, model: int, draws: str) -> int:
        """Model ``model``'s draw columns under ``draws`` (``DRAW_MODES``)."""
        return 0 if draws == "none" else self.models[model].draw_count

    def column_count(self, models: Sequence[int], draws: str) -> int:
        return sum(1 + self.draw_count(model, draws) for model in models)

    @classmethod
    def from_models(cls, models: Sequence[ScoringModel]) -> ScoringPlan:
        if not models:
            raise ValueError("a scoring plan needs at least one model.")
        store_rows = np.unique(np.concatenate([model.store_rows for model in models]))
        breaks = np.flatnonzero(np.diff(store_rows) != 1) + 1
        run_starts = np.concatenate([[0], breaks]).astype(np.int64)
        run_stops = np.concatenate([breaks, [store_rows.shape[0]]]).astype(np.int64)
        row_runs = tuple(
            (int(store_rows[start]), int(store_rows[stop - 1]) + 1, int(start))
            for start, stop in zip(run_starts, run_stops, strict=True)
            if stop > start
        )
        return cls(
            models=tuple(models),
            store_rows=store_rows,
            positions=tuple(np.searchsorted(store_rows, model.store_rows).astype(np.int64) for model in models),
            mean_offsets=np.array([-float(model.signed_means @ (model.coefficients / model.signed_scales)) for model in models]),
            row_runs=row_runs,
            gaussian_posteriors=tuple(model.gaussian_posterior for model in models),
        )

    def block_weights(self, batch: Sequence[int], draws: str, plan_start: int, plan_stop: int, offsets: F64Array) -> F64Array:
        """The weights of plan rows [plan_start, plan_stop) for the batch's columns (rows x columns; each model's mean
        column, then its draws), with each draw column's share of its offset, -mu' w, subtracted from ``offsets``.
        A model's draws for these rows are its law's tile for them, generated here and dropped with the block."""
        weights = np.zeros((plan_stop - plan_start, self.column_count(batch, draws)))
        column = 0
        for model_index in batch:
            model = self.models[model_index]
            positions = self.positions[model_index]
            first, last = (int(value) for value in np.searchsorted(positions, [plan_start, plan_stop]))
            local = positions[first:last] - plan_start
            scales = model.signed_scales[first:last]
            weights[local, column] = model.coefficients[first:last] / scales
            count = self.draw_count(model_index, draws)
            if count and last > first:
                draw_weights = model.posterior_draws.tile(first, last) / scales[:, None]
                weights[local, column + 1 : column + 1 + count] = draw_weights
                offsets[column + 1 : column + 1 + count] -= model.signed_means[first:last] @ draw_weights
            column += 1 + count
        return weights

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

    ``variances`` are the K-draw estimates v_i (NaN for a model without posterior draws, or when
    the draws were not scored) and ``draw_counts`` holds each model's K as scored. ``draws`` holds
    every draw's score (samples x K per model) when they were kept, else it is empty. The drawing
    law is the posterior only on the full-data route; on the mean-field route it is that fit's
    product approximation (the module docstring).
    """

    means: F64Array
    variances: F64Array
    draw_counts: tuple[int, ...]
    draws: tuple[F64Array, ...]
    gaussian_posteriors: tuple[bool, ...]

    def credible_interval(self, coverage: float) -> tuple[F64Array, F64Array]:
        """Central interval: a Gaussian t pivot or empirical mixture quantiles.

        The mean score is exact and the K draws are independent draws of it, so K v_i / v is
        chi-square with K degrees of freedom and independent of G ~ N(g, v); (G - g) / sqrt(v_i) is
        then Student-t with K degrees of freedom. This identity applies only to
        Gaussian posteriors. Mixture intervals use the retained draws and have
        Monte Carlo error; no exact finite-K coverage is claimed for them.
        """
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must lie strictly between 0 and 1.")
        counts = np.asarray(self.draw_counts, dtype=np.int64)
        if np.any(counts == 0):
            raise ValueError("a model without posterior draws has no credible interval.")
        half_width = stats.t.isf(0.5 * (1.0 - coverage), counts)[None, :] * np.sqrt(self.variances)
        lower, upper = self.means - half_width, self.means + half_width
        for model, gaussian in enumerate(self.gaussian_posteriors):
            if not gaussian:
                if not self.draws:
                    raise ValueError("a mixture posterior's interval needs its draws' scores: score with draws='keep'.")
                lower[:, model], upper[:, model] = np.quantile(self.draws[model], [0.5 * (1.0 - coverage), 0.5 * (1.0 + coverage)], axis=1)
        return lower, upper




def _source_owns_buffers(source: CodeBlockSource) -> bool:
    """A source that reads through its own ring (``artifact.StoreCodeBlocks``: the store's reader charges its ring to
    the ledger itself) is handed no buffers."""
    return bool(getattr(source, "owns_buffers", False))


def _output_bytes(plan: ScoringPlan, selected_samples: int, draws: str) -> int:
    """The result's arrays: each model's mean and variance scores, and every draw's score where they are kept."""
    kept = sum(plan.draw_count(model, draws) for model in range(plan.model_count)) if draws == "keep" else 0
    return _FLOAT64_BYTES * selected_samples * (2 * plan.model_count + kept)


def _batch_bytes(plan: ScoringPlan, batch: Sequence[int], store_samples: int, selected_samples: int, rows: int, device_kind: str, draws: str) -> int:
    """Host bytes of one batch's read with blocks of ``rows`` rows: the accumulator (columns x samples) and one more
    columns x samples of temporaries (on the CPU every panel's product, on CUDA each device accumulator's copy back;
    then each model's draw deviations, never live with either); per block row the read-ahead ring, the block's weights
    as built and as their transposed copy, the widest of its models' draw tiles with their working rows, the
    selected-sample copy and, on the CPU, every panel's fp64 codes."""
    columns = plan.column_count(batch, draws)
    fixed = 2 * _FLOAT64_BYTES * columns * selected_samples
    tile = max(
        (_FLOAT64_BYTES * plan.draw_count(model, draws) + plan.models[model].posterior_draws.tile_row_bytes() for model in batch if plan.draw_count(model, draws)),
        default=0,
    )
    per_row = _READ_AHEAD_BUFFERS * store_samples + 2 * _FLOAT64_BYTES * columns + tile
    if selected_samples != store_samples:
        per_row += selected_samples
    if device_kind == "cpu":
        per_row += _FLOAT64_BYTES * selected_samples
    return fixed + rows * per_row


def _host_bytes(plan: ScoringPlan, store_samples: int, selected_samples: int, rows: int, device_kind: str, draws: str = "keep") -> int:
    """Host bytes of scoring every model in one batch with reads of ``rows`` rows: the result's arrays and the batch's
    read (``_batch_bytes``)."""
    batch = list(range(plan.model_count))
    return _output_bytes(plan, selected_samples, draws) + _batch_bytes(plan, batch, store_samples, selected_samples, rows, device_kind, draws)


def _batches(
    plan: ScoringPlan, store_samples: int, selected_samples: int, host_bytes: int, device_kind: str, draws: str
) -> list[tuple[list[int], int]]:
    """Consecutive models grouped into the widest batches whose one-row read fits ``host_bytes``, each with the most
    block rows that fit beside it (the module docstring's rule); MemoryError where one model alone does not fit."""
    batches: list[tuple[list[int], int]] = []
    current: list[int] = []

    def close(batch: list[int]) -> None:
        fixed = _batch_bytes(plan, batch, store_samples, selected_samples, 0, device_kind, draws)
        per_row = _batch_bytes(plan, batch, store_samples, selected_samples, 1, device_kind, draws) - fixed
        rows = (int(host_bytes) - fixed) // per_row
        if rows < 1:
            raise MemoryError(
                f"scoring model {batch[0]} ({plan.column_count(batch, draws)} weight columns) x {selected_samples} samples needs "
                f"{(fixed + per_row) / 1e9:.3f} GB of host memory beside the result; the budget holds {int(host_bytes) / 1e9:.3f} GB"
            )
        batches.append((batch, min(rows, int(plan.store_rows.shape[0]))))

    for model in range(plan.model_count):
        widened = current + [model]
        if current and _batch_bytes(plan, widened, store_samples, selected_samples, 1, device_kind, draws) > int(host_bytes):
            close(current)
            widened = [model]
        current = widened
    close(current)
    return batches


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


def _device_bytes(columns: int, samples: int, rows: int, tile: int) -> int:
    """The device plan of ``_device_tile`` at (r, t): the accumulator, a block's codes and weights, a tile's fp64 codes
    and product."""
    return _FLOAT64_BYTES * columns * samples + rows * (samples + _FLOAT64_BYTES * columns) + _FLOAT64_BYTES * tile * (rows + columns)


def _selected_codes(codes: U8Array, sample_indices: I64Array | None) -> U8Array:
    return codes if sample_indices is None else np.take(codes, sample_indices, axis=1)


def _cpu_panels(sample_count: int, worker_count: int) -> list[tuple[int, int]]:
    """One panel per worker, widths differing by at most one sample (equal work per sample)."""
    edges = [index * sample_count // worker_count for index in range(worker_count + 1)]
    return [(start, stop) for start, stop in zip(edges[:-1], edges[1:]) if stop > start]


def _score_cpu(
    blocks: Iterable[tuple[int, int, int, U8Array]],
    weights_for: Callable[[int, int], F64Array],
    accumulator: F64Array,
    budget: ComputeBudget,
) -> None:
    """accumulator[columns, samples] += W_b' S_b for every block, samples split into owned panels."""
    worker_count = max(1, int(budget.cpu_threads))
    panels = _cpu_panels(accumulator.shape[1], worker_count)
    with ThreadPoolExecutor(max_workers=worker_count) as pool, threadpool_limits(limits=1, user_api="blas"):
        for plan_start, plan_stop, codes in _plan_blocks(blocks):
            block_weights = np.ascontiguousarray(weights_for(plan_start, plan_stop).T)

            def panel(column_range: tuple[int, int], codes: U8Array = codes, block_weights: F64Array = block_weights) -> None:
                start, stop = column_range
                signed = np.subtract(codes[:, start:stop], SIGNED_CODE_OFFSET, dtype=np.float64)
                accumulator[:, start:stop] += block_weights @ signed

            for _ in pool.map(panel, panels):
                pass


def _score_cuda(
    blocks: Iterable[tuple[int, int, int, U8Array]],
    weights_for: Callable[[int, int], F64Array],
    accumulator: F64Array,
    budget: ComputeBudget,
    tiles: Sequence[int],
) -> None:
    """Blocks round-robin over the visible devices; each device keeps its own fp64 accumulator. ``tiles`` are each
    device's sample tile widths from its planned share (``_device_tile``)."""
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
            tile_columns = int(tiles[device_position])
            block_weights = cupy.asarray(np.ascontiguousarray(weights_for(plan_start, plan_stop).T))
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


BatchReducer = Callable[[int, F64Array, F64Array], None]
"""``reduce(model, mean_scores (samples,), draw_scores (samples x K))``: called once per model with its scores, while its
batch's accumulator holds them (``artifact.predict`` forms its predictive variance there, with no draw kept)."""


def score_genetic(
    source: CodeBlockSource,
    plan: ScoringPlan,
    budget: ComputeBudget,
    sample_indices: NDArray | None = None,
    draws: str = "keep",
    reduce: BatchReducer | None = None,
) -> GeneticScores:
    """Posterior-mean genetic scores and, unless ``draws`` is "none", their variances under each model's drawing law,
    for every planned model, from one read of the codes per batch of models (the module docstring).

    ``sample_indices`` selects store samples (all of them when ``None``); scoring all samples once and slicing folds
    afterwards reads the store once for every fold. ``draws`` is one of ``DRAW_MODES``; ``reduce`` sees every model's
    mean and draw scores as its batch ends.
    """
    if draws not in DRAW_MODES:
        raise ValueError(f"draws must be one of {DRAW_MODES}, not {draws!r}")
    selected = _validated_sample_indices(sample_indices, source.sample_count)
    sample_count = source.sample_count if selected is None else int(selected.shape[0])
    broker = broker_for(budget)
    owns_buffers = _source_owns_buffers(source)
    with broker.reserve(HOST, _output_bytes(plan, sample_count, draws), "the scores"):
        means = np.empty((sample_count, plan.model_count))
        variances = np.full_like(means, np.nan)
        kept: list[F64Array] = [np.zeros((sample_count, 0)) for _ in range(plan.model_count)]
        batches = _batches(plan, source.sample_count, sample_count, broker.remaining(HOST), budget.device_kind, draws)
        for batch, block_rows in batches:
            columns = plan.column_count(batch, draws)
            tiles: list[int] = []
            device_leases = []
            if budget.device_kind == "cuda":
                for device_id in budget.device_ids:
                    rows, tile = _device_tile(columns, sample_count, broker.remaining(device_pool(device_id)))
                    block_rows = min(block_rows, rows)
                    tiles.append(tile)
                for device_id, tile in zip(budget.device_ids, tiles):
                    device_leases.append(broker.reserve(device_pool(device_id), _device_bytes(columns, sample_count, block_rows, tile), "a scoring batch's device plan"))
            host_plan = _batch_bytes(plan, batch, source.sample_count, sample_count, block_rows, budget.device_kind, draws)
            ring = 0 if owns_buffers else _READ_AHEAD_BUFFERS * block_rows * source.sample_count
            with broker.reserve(HOST, host_plan - ring, "a scoring batch's accumulator and block buffers"):
                ring_lease = broker.reserve(PINNED if budget.device_kind == "cuda" else HOST, ring, "the scorer's read-ahead ring")
                try:
                    ranges = plan.read_ranges(block_rows)
                    log(
                        f"  fast scoring: {len(batch)} of {plan.model_count} models ({columns} weight columns) x {sample_count} samples over "
                        + f"{plan.store_rows.shape[0]} store rows in {len(ranges)} reads of <= {block_rows} rows on {budget.describe()}"
                    )
                    accumulator = np.zeros((columns, sample_count), dtype=np.float64)
                    offsets = np.zeros(columns)
                    buffers = [] if owns_buffers else _read_ahead_buffers(block_rows, source.sample_count, budget)
                    blocks = _read_plan_blocks(source, ranges, buffers, selected)

                    def weights_for(plan_start: int, plan_stop: int, batch=batch, offsets=offsets) -> F64Array:
                        return plan.block_weights(batch, draws, plan_start, plan_stop, offsets)

                    if budget.device_kind == "cuda":
                        _score_cuda(blocks, weights_for, accumulator, budget, tiles)
                    else:
                        _score_cpu(blocks, weights_for, accumulator, budget)
                    del buffers, blocks
                finally:
                    ring_lease.release()
                    for lease in device_leases:
                        lease.release()
                column = 0
                for model in batch:
                    count = plan.draw_count(model, draws)
                    accumulator[column] += plan.mean_offsets[model]
                    means[:, model] = accumulator[column]
                    if count:
                        draw_scores = accumulator[column + 1 : column + 1 + count]
                        draw_scores += offsets[column + 1 : column + 1 + count, None]
                        deviations = draw_scores - accumulator[column]
                        variances[:, model] = np.mean(deviations * deviations, axis=0)
                        if draws == "keep":
                            kept[model] = np.ascontiguousarray(draw_scores.T)
                    if reduce is not None:
                        reduce(model, means[:, model], accumulator[column + 1 : column + 1 + count].T)
                    column += 1 + count
                del accumulator
    draw_counts = tuple(plan.draw_count(model, draws) for model in range(plan.model_count))
    return GeneticScores(
        means=means, variances=variances, draw_counts=draw_counts, draws=tuple(kept) if draws == "keep" else (),
        gaussian_posteriors=plan.gaussian_posteriors,
    )
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
