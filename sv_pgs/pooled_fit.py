"""One EP-EB fit of many small-n models that share their prior (batch_design.md rev 2: bench-real's pooled arm).

Each gene g has its own training codes, covariates and target, so its own posterior q_g(beta_g) and noise sigma_g^2;
the genes share the prior's hyperparameters x (the mixing density, the class deviations, the annotation terms),
learned once from all of them. Each gene's rows also carry its level l_g, a shift of their log prior variance, with
l_g ~ N(mu_l, omega^2): an annotation group "gene" of one indicator per gene (in sum-to-zero coordinates; g's location
pin carries mu_l) whose ridge weight 1 / omega^2 the EB learns like any group's. That is the identified model at
n ~ 600 per gene (review-theory): a single gene informs its level, the genes together inform x.

The genes are independent given x, so q = (x) q_g: every quantity the outer loop asks of the joint posterior is a
direct sum of the genes' own, each exact from its small-n kernel (``small_n``). ``scale_mixture_ep.fit_hyperparameters``
fits the stacked (gene, variant) rows as one model whose fixed-point oracle is ``_PooledFixedPoints``: it runs
``small_n``'s fixed-point iteration on every gene in lockstep (one tilted-moment pass over all rows per sweep), with
each gene's sites, noise update and site halving its own, and it tests the fixed point on the pooled sums (the mean
move in the joint posterior metric against sum_g p_eff,g / K, and the noise updates' summed evidence gain against
1/(2K)). The certificates are the engine's, on the pooled evidence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np

from sv_pgs._typing import F64Array
from sv_pgs.config import TraitType
from sv_pgs.fast_scoring import ScoringModel
from sv_pgs.full_data_fit import FitCertificate
from sv_pgs.scale_mixture_ep import (
    Cavity,
    FixedPoint,
    GaussianPosterior,
    MixtureHyperparameters,
    ScaleMixturePrior,
    _sum_to_zero_basis,
    _total_curvature,
    derived_lattice,
    fit_hyperparameters,
    initial_hyperparameters,
    moment_start,
    noise_variance,
    prior_second_moment,
    scale_mixture_prior,
)
from sv_pgs.small_n import (
    _EXACT_RESPONSE_MATRICES,
    _FLOAT_BYTES,
    _LIVE_FIXED_POINTS,
    DenseStatistics,
    _DensePosterior,
    _Kernel,
    _DenseFixedPoints,
    _new_profile,
    dense_statistics,
)
from sv_pgs.tie_map import _compact_identity_tie_map

_EPSILON = float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class GeneData:
    """One gene's training view: store codes [n, records], covariates [n, k] (intercept first), target [n], each
    record's variant class, and its log reliability (None: every record measured exactly)."""

    codes: np.ndarray
    covariates: F64Array
    target: F64Array
    variant_class: np.ndarray
    log_variance_offset: F64Array | None = None


@dataclass(frozen=True)
class PooledFit:
    """Each gene's scoring model (``store_rows`` index its code matrix's columns) and noise variance, the shared
    hyperparameters and prior over the stacked reduced rows (gene g's rows are ``gene_rows[g]``), the certificate and
    where the time went."""

    scoring: tuple[ScoringModel, ...]
    noise_variance: F64Array
    hyperparameters: MixtureHyperparameters
    prior: ScaleMixturePrior
    gene_rows: tuple[slice, ...]
    certificate: FitCertificate
    profile: dict = field(default_factory=dict)
    oracle: "_PooledFixedPoints | None" = field(default=None, repr=False, compare=False)


def _held_bytes(value: object, depth: int = 2) -> int:
    """Bytes of the arrays an object holds: its array attributes, tuples and lists of them, and those of the objects it
    holds, to ``depth`` (a kernel's ``design`` is the statistics', counted with them)."""
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (tuple, list)):
        return sum(_held_bytes(item, depth) for item in value)
    if depth and hasattr(value, "__dict__"):
        return sum(_held_bytes(item, depth - 1) for name, item in vars(value).items() if name != "design")
    return 0


def _gene_rows(statistics: Sequence[DenseStatistics]) -> tuple[slice, ...]:
    stops = np.cumsum([int(gene.design.variant_count) for gene in statistics])
    starts = np.concatenate([[0], stops[:-1]])
    return tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops))


def pooled_prior(
    statistics: Sequence[DenseStatistics], variant_classes: Sequence[np.ndarray], offsets: Sequence[F64Array], start_noise: F64Array, draw_count: int
) -> ScaleMixturePrior:
    """The prior over every gene's members (its active columns; review-mathbugs T1: an exact-tie member keeps its own
    class and offset), stacked in gene order: one class per variant class present in any gene, each member's log
    reliability as its offset, the genes' levels as learned offset groups (none for a single gene), and the lattice from every gene's single-variant likelihoods at its own start noise."""
    reduced = [np.asarray(classes)[gene.active_rows] for gene, classes in zip(statistics, variant_classes)]
    _classes, class_index = np.unique(np.concatenate(reduced), return_inverse=True)
    stacked_offsets = np.concatenate([np.asarray(offset, dtype=np.float64)[gene.active_rows] for gene, offset in zip(statistics, offsets)])
    single_precision = np.concatenate([gene.design.column_squares() / noise for gene, noise in zip(statistics, start_noise)])
    single_shift = np.concatenate([gene.design.back(gene.target) / noise for gene, noise in zip(statistics, start_noise)])
    nodes, floor, top = derived_lattice(single_precision, single_shift, stacked_offsets, 0.5 / draw_count)
    rows = _gene_rows(statistics)
    genes = np.concatenate([np.full(gene_rows.stop - gene_rows.start, gene, dtype=np.int64) for gene, gene_rows in enumerate(rows)])
    # Each gene's level is a gene-owned offset of its rows (review-mathbugs P2, lead ruling): the same shift for every
    # row of the gene whatever its class, sum-to-zero over genes, with one learned ridge weight; never class-centred.
    return scale_mixture_prior(
        class_index=class_index.astype(np.int64), log_variance_offset=stacked_offsets, annotation_design=np.zeros((class_index.shape[0], 0)),
        annotation_groups=(), nodes=nodes, floor=floor, top=top, offset_groups=genes if len(statistics) > 1 else None,
    )


class _PooledPosterior:
    """The joint posterior's responses as the direct sum of the genes' (``small_n._DensePosterior`` each).

    Every response is block diagonal by gene, so each gene answers its own rows, one gene at a time. A gene's exact
    response holds two matrices on its units (Sigma o Sigma's and the response's LU); the genes' together rarely fit, so
    the largest genes (whose matrices cost the most to form again, p^3 against p^2 bytes) stay resident within
    ``share`` less the room one other gene needs, and every other gene forms its matrices for the call and frees them.
    The responses are exact either way; residency only saves recomputation. Where one gene alone cannot hold its
    matrices, the exact response is unavailable and the total curvature uses GMRES on the direct sum."""

    def __init__(self, kernels: Sequence[_Kernel], noises: F64Array, rows: Sequence[slice], share: int, profile: dict) -> None:
        self.kernels = tuple(kernels)
        self.noises = np.asarray(noises, dtype=np.float64)
        self.rows = tuple(rows)
        self.share = int(share)
        self.profile = profile
        # A gene's exact response holds two matrices on its units (members sharing a tie group and a site: ``small_n``).
        need = np.array([_EXACT_RESPONSE_MATRICES * _FLOAT_BYTES * kernel.units().size ** 2 for kernel in self.kernels])
        self.exact = bool(need.max(initial=0) <= self.share)
        room = self.share - int(need.max(initial=0))
        self.resident: dict[int, _DensePosterior] = {}
        spent = 0
        for gene in np.argsort(-need, kind="stable").tolist():
            if spent + int(need[gene]) <= room:
                self.resident[gene] = self._new(gene)
                spent += int(need[gene])

    def _new(self, gene: int) -> _DensePosterior:
        # Each posterior may form its exact matrices whenever one gene's fit the share (``gaussian_posterior``).
        return _DensePosterior(self.kernels[gene], float(self.noises[gene]), self.share, self.profile)

    def _each(self, values: F64Array, apply: Callable[[_DensePosterior, F64Array, slice], F64Array]) -> F64Array:
        array = np.asarray(values, dtype=np.float64)
        result = np.empty_like(array)
        for gene, rows in enumerate(self.rows):
            posterior = self.resident.get(gene) or self._new(gene)
            result[rows] = apply(posterior, array[rows], rows)
        return result

    def solve(self, right: F64Array, relative_tolerance: float) -> F64Array:
        return self._each(right, lambda posterior, values, _rows: posterior.solve(values, relative_tolerance))

    def variance_jvp(self, weights: F64Array) -> F64Array:
        return self._each(weights, lambda posterior, values, _rows: posterior.variance_jvp(values))

    def linear_response(self, left: F64Array, right: F64Array, diagonal: F64Array, weight: F64Array, rhs: F64Array) -> F64Array:
        # The response matrix is block diagonal by gene (Sigma and Sigma o Sigma are), so each gene solves its own block:
        # a resident gene from its kept factor, every other one with matrices formed for the call and freed after it.
        values = np.asarray(rhs, dtype=np.float64)
        result = np.empty_like(values)
        for gene, rows in enumerate(self.rows):
            arguments = (left[rows], right[rows], diagonal[rows], weight[rows], values[rows])
            posterior = self.resident.get(gene) or self._new(gene)
            result[rows] = posterior.linear_response(*arguments)
        return result

    def gaussian_posterior(self) -> GaussianPosterior:
        # Every gene's response is its own dense LU, exact to rounding; the engine reads ``exact`` only where it has one.
        return GaussianPosterior(
            solve=self.solve, variance_jvp=self.variance_jvp, linear_response=self.linear_response if self.exact else None, exact=self.exact,
        )


class _PooledFixedPoints:
    """``scale_mixture_ep.FixedPoints`` for the stacked genes. The genes are independent given x, so each gene's EP fixed
    point is small_n's own (``small_n._DenseFixedPoints`` on the gene's rows of the pooled prior: its refresh and
    repair, frozen passes, noise update, double loop and certificate, unchanged), and the pooled fixed point is their
    direct sum. A gene that has no fixed point at a trial refuses the trial, and every gene returns to its state before
    it (the outer loop halves the step). Each gene is certified on its own and its prediction held to its own budget."""

    def __init__(
        self,
        statistics: Sequence[DenseStatistics],
        prior: ScaleMixturePrior,
        start: MixtureHyperparameters,
        start_noise: F64Array,
        draw_count: int,
        working_bytes: int,
    ) -> None:
        self.statistics = tuple(statistics)
        self.prior = prior
        self.rows = _gene_rows(statistics)
        self.draw_count = int(draw_count)
        self.working_bytes = int(working_bytes)
        # The genes solve one after another, so each may use the oracle's whole memory while it does.
        self.genes = tuple(
            _DenseFixedPoints(gene, _gene_prior(prior, rows), start, float(noise), draw_count, working_bytes)
            for gene, rows, noise in zip(self.statistics, self.rows, np.asarray(start_noise, dtype=np.float64))
        )
        self.mean_move = np.inf
        self.noise_gain = np.inf
        self.gene_divergence = np.full(len(self.rows), np.inf)
        self.gene_noise_gain = np.full(len(self.rows), np.inf)
        self.gene_cpu_seconds = np.zeros(len(self.rows))
        self.refusals: list[str] = []
        self._posterior_profile = _new_profile()

    @property
    def gene_count(self) -> int:
        return len(self.rows)

    @property
    def kernels(self) -> list[_Kernel | None]:
        return [gene.kernel for gene in self.genes]

    @property
    def noise(self) -> F64Array:
        return np.array([gene.noise for gene in self.genes])

    @property
    def effective(self) -> F64Array:
        return np.array([gene.effective for gene in self.genes])

    @property
    def mean(self) -> F64Array:
        return np.concatenate([gene.mean for gene in self.genes])

    @property
    def site_precision(self) -> F64Array:
        return np.concatenate([gene.site_precision for gene in self.genes])

    @property
    def site_shift(self) -> F64Array:
        return np.concatenate([gene.site_shift for gene in self.genes])

    @property
    def profile(self) -> dict:
        """Every gene's counts and seconds summed, and the pooled posterior's."""
        total = dict(self._posterior_profile)
        for gene in self.genes:
            for name, value in gene.profile.items():
                total[name] = total.get(name, 0) + value
        return total

    def _posterior(self) -> GaussianPosterior:
        # The oracle's memory holds the live kernel sets (the current fixed point's, a trial's, and the snapshot a refused
        # trial restores) and the live fixed points' posteriors (the current one and a trial's), which share the rest.
        kernel_bytes = sum(_held_bytes(kernel) for kernel in self.kernels)
        live_kernels = (_LIVE_FIXED_POINTS + 1) * kernel_bytes
        if live_kernels > self.working_bytes:
            # The kernel sets are a need, not a share: fail before the fit overruns its memory (svpgs-pooled-run).
            raise MemoryError(
                f"the pooled fit's live kernel sets need {live_kernels / 1e9:.2f} GB, more than the oracle's "
                f"{self.working_bytes / 1e9:.2f} GB: give the fit at least {2 * live_kernels / 1e9:.2f} GB after its Stage 0 arrays"
            )
        share = (self.working_bytes - live_kernels) // _LIVE_FIXED_POINTS
        return _PooledPosterior(self.kernels, self.noise, self.rows, share, self._posterior_profile).gaussian_posterior()

    def __call__(self, hyperparameters: Sequence[MixtureHyperparameters]) -> list[FixedPoint | None]:
        (model_hyperparameters,) = hyperparameters
        self._posterior_profile["fixed_point_calls"] += 1
        snapshots = [gene._snapshot() for gene in self.genes]
        tolerance = 0.5 / self.draw_count
        points = []
        for index, gene in enumerate(self.genes):
            cpu = time.process_time()
            try:
                points.append(gene._solve(model_hyperparameters))
            except FloatingPointError as error:
                # NoFixedPoint, or a tilted law outside EP's domain reached inside the gene's EP: no fixed point here.
                for other, snapshot in zip(self.genes, snapshots):
                    other._restore(snapshot)
                self.refusals.append(f"gene {index}: {error}")
                return [None]
            finally:
                self.gene_cpu_seconds[index] += time.process_time() - cpu
            self.gene_divergence[index] = 0.5 * gene.mean_move
            self.gene_noise_gain[index] = gene.noise_gain
        self.mean_move = float(np.max(self.gene_divergence)) / tolerance
        self.noise_gain = float(np.max(self.gene_noise_gain))
        norms = [point.precision_norm for point in points]
        rows = self.rows

        def precision_norm(direction: F64Array) -> F64Array:
            # One move per gene, each scored as its own model: the prediction check holds each to its own p_eff,g / K.
            values = np.asarray(direction, dtype=np.float64)
            return np.array([float(norm(values[gene_rows])) for norm, gene_rows in zip(norms, rows)])

        return [FixedPoint(
            cavity=Cavity(
                precision=np.concatenate([point.cavity.precision for point in points]),
                shift=np.concatenate([point.cavity.shift for point in points]),
            ),
            posterior=self._posterior(),
            mean=np.concatenate([point.mean for point in points]),
            precision_norm=precision_norm,
            effective_effects=np.array([float(point.effective_effects) for point in points]),
        )]


def _gene_moment(statistics: DenseStatistics, log_variance_offset: F64Array):
    """``small_n_start``'s Haseman-Elston moments of one gene, with its rows' log reliabilities."""
    design = statistics.design
    residual = statistics.projected_target
    weights = np.exp(log_variance_offset)
    kernel = design.weighted_gram(np.ones(design.variant_count))
    score = design.back(statistics.target)
    return moment_start(
        target_square=float(residual @ residual),
        residual_dimension=float(statistics.sample_count - statistics.covariate_rank),
        score_square=float(score @ score),
        gram_trace=float(np.trace(kernel)),
        weighted_diagonal=float(weights @ design.column_squares()),
        weighted_square=float(weights @ design.quadratic_diagonal(kernel)),
        gram_square=float(np.sum(kernel * kernel)),
    )


def _pooled_start(statistics: Sequence[DenseStatistics], prior: ScaleMixturePrior, rows: Sequence[slice]) -> tuple[MixtureHyperparameters, F64Array]:
    """A start, not a prior: every gene's Haseman-Elston moments give its noise and its genetic variance per variant;
    the density starts at their geometric mean, weighted by the genes' column counts, and each gene's level at its
    own log ratio to it."""
    moments = [_gene_moment(gene, prior.log_variance_offset[gene_rows]) for gene, gene_rows in zip(statistics, rows)]
    noise = np.array([float(moment.noise) for moment in moments])
    log_variance = np.log([max(float(moment.mean_variance), np.finfo(np.float64).tiny) for moment in moments])
    weights = np.array([float(gene_rows.stop - gene_rows.start) for gene_rows in rows])
    pooled = float(np.sum(weights * log_variance) / np.sum(weights))
    start = initial_hyperparameters(prior, float(np.exp(pooled)))
    coefficients = start.coefficients.copy()
    if len(rows) > 1:
        # The level group's coordinates are x's last G - 1 (``pooled_prior``'s only annotation group).
        coefficients[prior.coefficient_size - (len(rows) - 1) :] = _sum_to_zero_basis(len(rows)).T @ (log_variance - pooled)
    return MixtureHyperparameters(coefficients=coefficients, log_smoothing=start.log_smoothing), noise


def fit_pooled_small_n(genes: Sequence[GeneData], *, draw_count: int, working_bytes: int, seed: int) -> PooledFit:
    """Fit every gene's quantitative model with one prior learned from all of them (see the module docstring)."""
    started = time.perf_counter()
    offsets = [np.zeros(np.asarray(gene.codes).shape[1]) if gene.log_variance_offset is None else np.asarray(gene.log_variance_offset) for gene in genes]
    statistics = [dense_statistics(gene.codes, gene.covariates, gene.target, offset) for gene, offset in zip(genes, offsets)]
    residual_noise = np.array([float(gene.projected_target @ gene.projected_target) / (gene.sample_count - gene.covariate_rank) for gene in statistics])
    prior = pooled_prior(statistics, [gene.variant_class for gene in genes], offsets, residual_noise, draw_count)
    start, start_noise = _pooled_start(statistics, prior, _gene_rows(statistics))
    stage0_seconds = time.perf_counter() - started
    # The genes' Stage 0 arrays stay for the whole fit; the rest is shared equally by the engine's own working set
    # (tilted-moment chunks; GMRES where an exact response does not fit) and the fixed-point oracle.
    available = working_bytes - sum(_held_bytes(gene) + _held_bytes(gene.design) for gene in statistics)
    if available <= 0:
        raise MemoryError(f"the genes' Stage 0 arrays alone exceed the {working_bytes / 1e9:.2f} GB working memory.")
    engine_bytes = available // 2
    oracle = _PooledFixedPoints(statistics, prior, start, start_noise, draw_count, available - engine_bytes)
    try:
        (outer,) = fit_hyperparameters(prior, [start], oracle, engine_bytes, 0.5 / draw_count)
    except FloatingPointError as error:
        raise FloatingPointError(f"{error}; EP refusals: {oracle.refusals}") from error
    if not outer.certified:
        raise FloatingPointError(f"pooled outer fit did not converge: remaining gain {outer.remaining_gain}, unresolved {outer.unresolved}; EP refusals: {oracle.refusals}")
    second_moment = prior_second_moment(prior, outer.hyperparameters)
    generator = np.random.default_rng(seed)
    scoring = []
    for gene_index, (gene, gene_statistics, rows) in enumerate(zip(genes, statistics, oracle.rows)):
        mean = oracle.mean[rows]
        draws = mean[:, None] + np.sqrt(float(oracle.noise[gene_index])) * oracle.kernels[gene_index].draws(generator, draw_count)
        alpha = gene_statistics.covariate_pseudo_inverse @ (gene_statistics.covariates.T @ gene_statistics.target - gene_statistics.loading @ mean)
        # Every member is its own effect (review-mathbugs T1): beta_j = s_j gamma_j on its own standardized column, with
        # no split of a tie group's effect.
        members = gene_statistics.active_rows.shape[0]
        scoring.append(ScoringModel.from_reduced_fit(
            active_rows=gene_statistics.active_rows, signed_means=gene_statistics.means, signed_scales=gene_statistics.scales,
            tie_map=_compact_identity_tie_map(members), member_prior_variances=second_moment[rows], beta_reduced=gene_statistics.signs * mean,
            posterior_draws_reduced=gene_statistics.signs[:, None] * draws, alpha=alpha, trait_type=TraitType.QUANTITATIVE,
            predictive_intercept_shift=0.0,
            covariate_draws=alpha[:, None] - gene_statistics.covariate_pseudo_inverse @ gene_statistics.loading @ (draws - mean[:, None]),
            covariate_covariance=float(oracle.noise[gene_index]) * gene_statistics.covariate_pseudo_inverse,
            gaussian_posterior=True,
        ))
    certificate = FitCertificate(
        remaining_gain=np.array([outer.remaining_gain]),
        newton_decrement=np.array([outer.newton_decrement]),
        smoothing_gradient=np.array([outer.step.smoothing_gradient]),
        stationarity_steps=(outer.step.stationarity_steps,),
        stationarity_errors=(outer.step.stationarity_errors,),
        # Per gene, as the largest ratio of a gene's update divergence to 1/(2K) (so the tolerance is 1), and the
        # largest gene noise gain; the genes' own values are in the profile.
        mean_move=np.array([oracle.mean_move]),
        draw_tolerance=np.ones(1),
        noise_gain=np.array([oracle.noise_gain]),
        mean_error=np.zeros(1),
        information_bound=np.zeros(1),
        information_tolerance=np.zeros(1),
        undecided_blocks=0,
        negative_sites=np.array([int(np.sum(oracle.site_precision < 0.0))], dtype=np.int64),
        effective_effects=np.array([float(np.sum(oracle.effective))]),
        outer_iterations=np.array([outer.iterations], dtype=np.int64),
        halvings=np.array([outer.halvings], dtype=np.int64),
        prediction_move=np.array([outer.prediction_move]),
        prediction_tolerance=np.array([outer.prediction_tolerance]),
        unresolved=np.array([outer.unresolved], dtype=np.int64),
        refusals=tuple(oracle.refusals),
        outer_history=(outer.history,),
        refreshes=int(oracle.profile["refreshes"]),
        passes=int(oracle.profile["passes"]),
        certified=np.array([outer.certified], dtype=bool),
    )
    profile = dict(oracle.profile) | {
        "stage0_seconds": stage0_seconds, "total_seconds": time.perf_counter() - started, "genes": len(genes),
        "reduced": int(prior.variant_count), "coefficients": int(prior.coefficient_size), "classes": int(prior.class_count),
        "outer_iterations": int(outer.iterations),
        "gene_divergence": oracle.gene_divergence.tolist(), "gene_cpu_seconds": oracle.gene_cpu_seconds.tolist(),
        "gene_refreshes": [int(gene.profile["refreshes"]) for gene in oracle.genes],
        "gene_noise_gain": oracle.gene_noise_gain.tolist(),
    }
    return PooledFit(
        scoring=tuple(scoring), noise_variance=oracle.noise.copy(), hyperparameters=outer.hyperparameters, prior=prior,
        gene_rows=oracle.rows, certificate=certificate, profile=profile, oracle=oracle,
    )


@dataclass(frozen=True)
class CurvatureBlocks:
    """Each gene's share of the total curvature at the fitted x: ``blocks[g]`` = B_g (D x D, in x) with
    sum_g B_g = B, the outer step's B + S less S. ``penalty_blocks`` names each learned weight's x coordinates,
    ``log_smoothing`` its fitted log weight (inf: at its edge, the term in its penalty's null space), and ``null_basis``
    spans the total penalty's null space."""

    coefficients: F64Array
    blocks: F64Array
    penalty_blocks: tuple[tuple[str, np.ndarray], ...]
    log_smoothing: F64Array
    null_basis: F64Array


def gene_owned_blocks(curvature: CurvatureBlocks, prior: ScaleMixturePrior) -> tuple[F64Array, F64Array, F64Array]:
    """Each gene's block in gene-owned coordinates (theory-ep section 5): its rows see the levels only through their own
    level l_g = q_g' z (q_g the gene's row of the sum-to-zero basis), so B_g's level part is b_g q_g q_g' and its
    coupling to the shared coordinates is c_g q_g'. Returns (B_g on the shared coordinates [G, S, S], c_g [G, S],
    b_g [G]) and leaves the shared-coordinate indices to ``prior``'s layout: every x coordinate but the last G - 1."""
    level_size = prior.level_size
    genes = level_size + 1
    basis = _sum_to_zero_basis(genes)
    shared = curvature.blocks.shape[1] - level_size
    shared_blocks = curvature.blocks[:, :shared, :shared]
    coupling = np.empty((genes, shared))
    level = np.empty(genes)
    for gene in range(genes):
        row = basis[gene]
        norm = float(row @ row)
        coupling[gene] = curvature.blocks[gene, :shared, shared:] @ row / norm
        level[gene] = float(row @ curvature.blocks[gene, shared:, shared:] @ row) / norm**2
    return shared_blocks, coupling, level


def _gene_prior(prior: ScaleMixturePrior, rows: slice) -> ScaleMixturePrior:
    """The prior on one gene's rows, in the pooled x coordinates: every per-variant field restricted, the lattice and
    x's layout shared. The data objective and B are sums over rows, so the genes' parts add up to the pooled one."""
    class_index = prior.class_index[rows]
    order = np.argsort(class_index, kind="stable")
    sizes = np.bincount(class_index, minlength=prior.class_count)
    return replace(
        prior,
        class_index=class_index,
        class_rows=tuple(np.split(order, np.cumsum(sizes)[:-1])),
        log_variance_offset=prior.log_variance_offset[rows],
        scale_design=prior.scale_design[rows],
        offset_groups=None if prior.offset_groups is None else prior.offset_groups[rows],
    )


def pooled_curvature_blocks(fit: PooledFit, working_bytes: int) -> CurvatureBlocks:
    """B_g for every gene at the fitted x (theory-ep section 5): EP is re-solved there (the oracle's last fixed point
    may be a trial's), and each gene's total curvature is the engine's own, on its rows with its exact response."""
    oracle = fit.oracle
    if oracle is None:
        raise ValueError("the fit keeps no fixed-point oracle.")
    hyperparameters = fit.hyperparameters
    (point,) = oracle([hyperparameters])
    if point is None:
        raise FloatingPointError(f"no EP fixed point at the fitted hyperparameters; EP refusals: {oracle.refusals}")
    tolerance = 0.5 / oracle.draw_count
    relative_tolerance = max(tolerance / hyperparameters.coefficients.shape[0], _EPSILON)
    share = working_bytes // _LIVE_FIXED_POINTS
    blocks = []
    for gene, rows in enumerate(oracle.rows):
        posterior = _DensePosterior(oracle.kernels[gene], float(oracle.noise[gene]), share, oracle.profile).gaussian_posterior()
        cavity = Cavity(precision=point.cavity.precision[rows], shift=point.cavity.shift[rows])
        blocks.append(_total_curvature(_gene_prior(fit.prior, rows), hyperparameters.coefficients, cavity, posterior, share, relative_tolerance))
    return CurvatureBlocks(
        coefficients=hyperparameters.coefficients.copy(),
        blocks=np.stack(blocks),
        penalty_blocks=tuple((block.name, np.asarray(block.coordinates)) for block in fit.prior.smoothing_blocks),
        log_smoothing=np.asarray(hyperparameters.log_smoothing, dtype=np.float64).copy(),
        null_basis=fit.prior.null_basis.copy(),
    )
