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
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy import linalg

from sv_pgs._typing import F64Array
from sv_pgs.config import TraitType
from sv_pgs.fast_scoring import ScoringModel
from sv_pgs.full_data_fit import FitCertificate, NoFixedPoint
from sv_pgs.scale_mixture_ep import (
    AnnotationGroup,
    Cavity,
    FixedPoint,
    GaussianPosterior,
    MixtureHyperparameters,
    ScaleMixturePrior,
    _sum_to_zero_basis,
    derived_lattice,
    fit_hyperparameters,
    initial_hyperparameters,
    log_scale,
    moment_start,
    moment_matched_prior_sites,
    noise_gain,
    noise_variance,
    prior_second_moment,
    scale_mixture_prior,
    site_targets,
    tilted_moments,
)
from sv_pgs.small_n import (
    _EXACT_RESPONSE_MATRICES,
    _FLOAT_BYTES,
    _LIVE_FIXED_POINTS,
    DenseStatistics,
    _DensePosterior,
    _Kernel,
    _new_profile,
    dense_statistics,
)

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


def _gene_rows(statistics: Sequence[DenseStatistics]) -> tuple[slice, ...]:
    stops = np.cumsum([int(gene.projected.shape[1]) for gene in statistics])
    starts = np.concatenate([[0], stops[:-1]])
    return tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops))


def pooled_prior(
    statistics: Sequence[DenseStatistics], variant_classes: Sequence[np.ndarray], offsets: Sequence[F64Array], start_noise: F64Array, draw_count: int
) -> ScaleMixturePrior:
    """The prior over every gene's reduced columns, stacked in gene order: one class per variant class present in any
    gene, each representative's log reliability as its offset, the gene level as one ridge-penalized annotation group
    (none for a single gene), and the lattice from every gene's single-variant likelihoods at its own start noise."""
    reduced = [np.asarray(classes)[gene.reduced_rows] for gene, classes in zip(statistics, variant_classes)]
    _classes, class_index = np.unique(np.concatenate(reduced), return_inverse=True)
    stacked_offsets = np.concatenate([np.asarray(offset, dtype=np.float64)[gene.reduced_rows] for gene, offset in zip(statistics, offsets)])
    single_precision = np.concatenate([gene.design.column_squares() / noise for gene, noise in zip(statistics, start_noise)])
    single_shift = np.concatenate([gene.design.back(gene.target) / noise for gene, noise in zip(statistics, start_noise)])
    nodes, floor, top = derived_lattice(single_precision, single_shift, stacked_offsets, 0.5 / draw_count)
    gene_count = len(statistics)
    rows = _gene_rows(statistics)
    if gene_count > 1:
        indicator = np.zeros((class_index.shape[0], gene_count))
        for gene, gene_slice in enumerate(rows):
            indicator[gene_slice, gene] = 1.0
        design = indicator @ _sum_to_zero_basis(gene_count)
        groups = (AnnotationGroup(columns=np.arange(gene_count - 1, dtype=np.int64), penalty=np.eye(gene_count - 1)),)
    else:
        design, groups = np.zeros((class_index.shape[0], 0)), ()
    return scale_mixture_prior(
        class_index=class_index.astype(np.int64), log_variance_offset=stacked_offsets, annotation_design=design, annotation_groups=groups,
        nodes=nodes, floor=floor, top=top,
    )


def _transient_response(
    kernel: _Kernel, noise: float, left: F64Array, right: F64Array, diagonal: F64Array, weight: F64Array, rhs: F64Array, profile: dict
) -> F64Array:
    """``small_n._DensePosterior.linear_response`` for one call, in one p x p buffer: Sigma o Sigma is formed, its
    product with the low-rank factor taken, and the response matrix then built in the same buffer (Sigma o Sigma is not
    needed again) and LU-factored there. The matrix is the same as ``_DensePosterior._response_matrix``'s."""
    started = time.perf_counter()
    squared = kernel.covariance()
    squared *= noise
    np.square(squared, out=squared)
    delta, phi, psi = kernel.factors()
    d0 = diagonal + noise * left * delta * right
    factor_rows = np.vstack([phi, psi]) if psi.shape[0] else phi
    signs = np.concatenate([-np.ones(phi.shape[0]), np.ones(psi.shape[0])])
    low = (noise * left)[:, None] * (factor_rows.T * signs[None, :])  # U (p x r)
    high = factor_rows * right[None, :]  # V' (r x p)
    squared_low = squared @ low
    matrix = squared
    matrix *= d0[None, :]
    step = max(1, high.shape[0])
    for start in range(0, matrix.shape[0], step):
        rows = slice(start, min(start + step, matrix.shape[0]))
        matrix[rows] += squared_low[rows] @ high
        matrix[rows] *= weight[rows, None]
        matrix[rows] -= low[rows] @ high
    matrix[np.diag_indices_from(matrix)] += 1.0 - d0
    # LAPACK factors a Fortran-ordered array in place, and a C-ordered matrix is its transpose's Fortran form: factor the
    # transpose there and solve with it transposed.
    factor = linalg.lu_factor(matrix.T, overwrite_a=True, check_finite=False)
    solution = linalg.lu_solve(factor, rhs, trans=1, check_finite=False)
    profile["response_factorizations"] += 1
    profile["responses"] += 1
    profile["response_seconds"] += time.perf_counter() - started
    return solution


class _PooledPosterior:
    """The joint posterior's responses as the direct sum of the genes' (``small_n._DensePosterior`` each).

    Every response is block diagonal by gene, so each gene answers its own rows, one gene at a time. A gene's exact
    response holds two p_g x p_g matrices (Sigma o Sigma and the response's LU); the genes' together rarely fit, so
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
        need = np.array([_EXACT_RESPONSE_MATRICES * _FLOAT_BYTES * (gene_rows.stop - gene_rows.start) ** 2 for gene_rows in self.rows])
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
        # a resident gene from its kept factor, every other one in a single p x p buffer freed after the call.
        values = np.asarray(rhs, dtype=np.float64)
        result = np.empty_like(values)
        for gene, rows in enumerate(self.rows):
            arguments = (left[rows], right[rows], diagonal[rows], weight[rows], values[rows])
            if gene in self.resident:
                result[rows] = self.resident[gene].linear_response(*arguments)
            else:
                result[rows] = _transient_response(self.kernels[gene], float(self.noises[gene]), *arguments, self.profile)
        return result

    def gaussian_posterior(self) -> GaussianPosterior:
        return GaussianPosterior(solve=self.solve, variance_jvp=self.variance_jvp, linear_response=self.linear_response if self.exact else None)


class _PooledFixedPoints:
    """``scale_mixture_ep.FixedPoints`` for the stacked genes: ``small_n._DenseFixedPoints``' iteration on every gene
    in lockstep (see the module docstring)."""

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
        self.scores = [gene.design.back(gene.target) for gene in self.statistics]
        precision, shift = moment_matched_prior_sites(prior, start)
        self.site_precision = precision.copy()
        self.site_shift = shift.copy()
        self.noise = np.asarray(start_noise, dtype=np.float64).copy()
        self.effective = np.array([float(rows.stop - rows.start) for rows in self.rows])
        self.kernels: list[_Kernel | None] = [None] * len(self.rows)
        self.mean = np.zeros(prior.variant_count)
        self.mean_move = np.inf
        self.noise_gain = np.inf
        self.refusals: list[str] = []
        self.profile = _new_profile()

    @property
    def gene_count(self) -> int:
        return len(self.rows)

    def _iterate(self, gene: int, site_precision: F64Array, site_shift: F64Array) -> None:
        """Gene ``gene``'s exact mean at the sites; ``LinAlgError`` when its precision is not positive definite."""
        started = time.perf_counter()
        noise = float(self.noise[gene])
        kernel = _Kernel(self.statistics[gene].design, noise * site_precision)
        self.mean[self.rows[gene]] = kernel.solve(self.scores[gene] + noise * site_shift)
        self.kernels[gene] = kernel
        self.profile["factorizations"] += 1
        self.profile["passes"] += 1
        self.profile["factor_seconds"] += time.perf_counter() - started

    def _refresh(self, hyperparameters: MixtureHyperparameters) -> tuple[F64Array, F64Array]:
        """Every gene's mean, exact marginal variances and cavity precisions; a gene's negative sites halve while its
        precision is not positive definite or a cavity's tilted law is improper (``small_n``'s refresh, per gene)."""
        largest = np.exp(log_scale(self.prior, hyperparameters.coefficients) + self.prior.log_variance_grid[-1])
        variances = np.empty(self.prior.variant_count)
        cavity_precision = np.empty(self.prior.variant_count)
        for gene, rows in enumerate(self.rows):
            while True:
                try:
                    self._iterate(gene, self.site_precision[rows], self.site_shift[rows])
                except np.linalg.LinAlgError:
                    failure = f"gene {gene}: the precision is not positive definite with non-negative sites"
                else:
                    started = time.perf_counter()
                    scaled_variances, removed, scaled_precision = self.kernels[gene].cavity()
                    self.profile["variance_seconds"] += time.perf_counter() - started
                    noise = float(self.noise[gene])
                    gene_variances, gene_precision = noise * scaled_variances, scaled_precision / noise
                    if np.all(1.0 + largest[rows] * gene_precision > 0.0):
                        variances[rows], cavity_precision[rows] = gene_variances, gene_precision
                        self.effective[gene] = max(float(np.sum(removed)), _EPSILON * (rows.stop - rows.start))
                        break
                    failure = f"gene {gene}: a cavity's tilted law is improper (1 + v P <= 0 on the lattice) with non-negative sites"
                negative = self.site_precision[rows] < 0.0
                if not np.any(negative):
                    raise NoFixedPoint(failure)
                self.site_precision[rows] = np.where(negative, 0.5 * self.site_precision[rows], self.site_precision[rows])
        self.profile["refreshes"] += 1
        return variances, cavity_precision

    def _targets(self, hyperparameters: MixtureHyperparameters, cavity: Cavity) -> tuple[F64Array, F64Array]:
        started = time.perf_counter()
        targets = site_targets(tilted_moments(self.prior, hyperparameters, cavity, self.working_bytes), cavity)
        self.profile["tilted_seconds"] += time.perf_counter() - started
        return targets

    def _residual_sum_of_squares(self, gene: int) -> float:
        statistics = self.statistics[gene]
        residual = statistics.projected_target - statistics.design.image(self.mean[self.rows[gene]])
        return float(residual @ residual)

    def _noises(self, variances: F64Array) -> F64Array:
        return np.array([
            noise_variance(
                residual_sum_of_squares=self._residual_sum_of_squares(gene), sample_count=gene_statistics.sample_count,
                covariate_count=int(gene_statistics.covariates.shape[1]), site_precision=self.site_precision[rows],
                posterior_variance=variances[rows], noise=float(self.noise[gene]),
            )
            for gene, (gene_statistics, rows) in enumerate(zip(self.statistics, self.rows))
        ])

    def _solve_mean_metric(self, right: F64Array) -> float:
        """r' Sigma r in the joint posterior metric: the sum of every gene's own."""
        return float(sum(right[rows] @ (float(self.noise[gene]) * self.kernels[gene].solve(right[rows])) for gene, rows in enumerate(self.rows)))

    def _precision_norm(self) -> Callable[[F64Array], float]:
        designs = [gene.design for gene in self.statistics]
        precision, noise, rows = self.site_precision.copy(), self.noise.copy(), self.rows

        def norm(direction: F64Array) -> float:
            values = np.asarray(direction, dtype=np.float64)
            total = float(np.sum(precision * values * values))
            for design, gene_noise, gene_rows in zip(designs, noise, rows):
                image = design.image(values[gene_rows])
                total += float(image @ image) / float(gene_noise)
            return total

        return norm

    def _posterior(self) -> GaussianPosterior:
        # The live fixed points (the current one and a trial) share the working memory equally.
        return _PooledPosterior(self.kernels, self.noise.copy(), self.rows, self.working_bytes // _LIVE_FIXED_POINTS, self.profile).gaussian_posterior()

    def _snapshot(self) -> dict:
        return {
            "site_precision": self.site_precision.copy(), "site_shift": self.site_shift.copy(), "noise": self.noise.copy(),
            "effective": self.effective.copy(), "kernels": list(self.kernels), "mean": self.mean.copy(),
        }

    def _restore(self, snapshot: dict) -> None:
        self.site_precision, self.site_shift = snapshot["site_precision"].copy(), snapshot["site_shift"].copy()
        self.noise, self.effective = snapshot["noise"].copy(), snapshot["effective"].copy()
        self.kernels, self.mean = list(snapshot["kernels"]), snapshot["mean"].copy()

    def __call__(self, hyperparameters: Sequence[MixtureHyperparameters]) -> list[FixedPoint | None]:
        (model_hyperparameters,) = hyperparameters
        self.profile["fixed_point_calls"] += 1
        snapshot = self._snapshot()
        try:
            return [self._solve(model_hyperparameters)]
        except NoFixedPoint as error:
            self._restore(snapshot)
            self.refusals.append(str(error))
            return [None]

    def _solve(self, hyperparameters: MixtureHyperparameters) -> FixedPoint:
        tolerance = 0.5 / self.draw_count
        while True:
            variances, frozen = self._refresh(hyperparameters)
            mean = self.mean.copy()
            cavity = Cavity(precision=frozen, shift=mean / variances - self.site_shift)
            target_precision, target_shift = self._targets(hyperparameters, cavity)
            right = (target_shift - self.site_shift) - (target_precision - self.site_precision) * mean
            self.mean_move = self._solve_mean_metric(right)
            noises = self._noises(variances)
            gains = [
                noise_gain(float(new), float(old), gene.sample_count, int(gene.covariates.shape[1]))
                for new, old, gene in zip(noises, self.noise, self.statistics)
            ]
            # The noise updates are independent parameters, so their evidence gains add.
            self.noise_gain = float(np.sum(gains))
            if self.mean_move <= float(np.sum(self.effective)) / self.draw_count and self.noise_gain <= tolerance:
                return FixedPoint(
                    cavity=cavity, posterior=self._posterior(), mean=mean, precision_norm=self._precision_norm(),
                    effective_effects=float(np.sum(self.effective)),
                )
            self._frozen_passes(hyperparameters, frozen, target_precision, target_shift)
            self.noise = self._noises(1.0 / (frozen + self.site_precision))

    def _frozen_passes(self, hyperparameters: MixtureHyperparameters, frozen: F64Array, target_precision: F64Array, target_shift: F64Array) -> None:
        """Mean-only EP with the cavity precisions frozen, every gene stepping each sweep with its own damping, until
        the pooled frozen move is below sum_g p_eff,g / K (``small_n._DenseFixedPoints._frozen_passes``)."""
        previous = np.full(self.gene_count, np.inf)
        damping = np.ones(self.gene_count)
        while True:
            moves = np.empty(self.gene_count)
            for gene, rows in enumerate(self.rows):
                mean = self.mean[rows].copy()
                fraction = float(damping[gene])
                delta_precision = target_precision[rows] - self.site_precision[rows]
                delta_shift = target_shift[rows] - self.site_shift[rows]
                move = max(float(np.max(np.abs(delta_precision), initial=0.0)), float(np.max(np.abs(delta_shift), initial=0.0)))
                scale = 1.0 + max(float(np.max(np.abs(self.site_precision[rows]))), float(np.max(np.abs(self.site_shift[rows]))))
                if fraction * move <= _EPSILON * scale:
                    # This gene's update is below its sites' rounding: it is at its frozen fixed point while the pooled
                    # move is not yet below tolerance, so it takes no step this sweep.
                    moves[gene] = 0.0
                    continue
                while True:
                    trial_precision = self.site_precision[rows] + fraction * delta_precision
                    trial_shift = self.site_shift[rows] + fraction * delta_shift
                    try:
                        self._iterate(gene, trial_precision, trial_shift)
                        break
                    except np.linalg.LinAlgError:
                        fraction *= 0.5
                        if fraction * move <= _EPSILON * scale:
                            raise NoFixedPoint(f"gene {gene}: no damped EP pass keeps the precision positive definite")
                self.site_precision[rows], self.site_shift[rows] = trial_precision, trial_shift
                marginal = 1.0 / (frozen[rows] + self.site_precision[rows])
                moves[gene] = float(np.sum(np.square(self.mean[rows] - mean) / marginal)) / (fraction * fraction)
                if moves[gene] / previous[gene] >= 1.0:
                    damping[gene] = min(float(damping[gene]), 1.0 / (1.0 + np.sqrt(moves[gene] / previous[gene])))
            if float(np.sum(moves)) <= float(np.sum(self.effective)) / self.draw_count:
                return
            previous = moves
            cavity = Cavity(precision=frozen, shift=self.mean / (1.0 / (frozen + self.site_precision)) - self.site_shift)
            target_precision, target_shift = self._targets(hyperparameters, cavity)


def _gene_moment(statistics: DenseStatistics, log_variance_offset: F64Array):
    """``small_n_start``'s Haseman-Elston moments of one gene, with its rows' log reliabilities."""
    design = statistics.design
    residual = statistics.projected_target
    weights = np.exp(log_variance_offset)
    kernel = design.weighted_gram(np.ones(design.variant_count))
    score = design.back(statistics.target)
    return moment_start(
        target_square=float(residual @ residual),
        residual_dimension=float(statistics.sample_count - statistics.covariates.shape[1]),
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
    statistics = [dense_statistics(gene.codes, gene.covariates, gene.target) for gene in genes]
    offsets = [np.zeros(np.asarray(gene.codes).shape[1]) if gene.log_variance_offset is None else np.asarray(gene.log_variance_offset) for gene in genes]
    residual_noise = np.array([float(gene.projected_target @ gene.projected_target) / (gene.sample_count - gene.covariates.shape[1]) for gene in statistics])
    prior = pooled_prior(statistics, [gene.variant_class for gene in genes], offsets, residual_noise, draw_count)
    start, start_noise = _pooled_start(statistics, prior, _gene_rows(statistics))
    stage0_seconds = time.perf_counter() - started
    oracle = _PooledFixedPoints(statistics, prior, start, start_noise, draw_count, working_bytes)
    try:
        (outer,) = fit_hyperparameters(prior, [start], oracle, working_bytes // 2, 0.5 / draw_count)
    except FloatingPointError as error:
        raise FloatingPointError(f"{error}; EP refusals: {oracle.refusals}") from error
    second_moment = prior_second_moment(prior, outer.hyperparameters)
    generator = np.random.default_rng(seed)
    scoring = []
    for gene_index, (gene, gene_statistics, rows) in enumerate(zip(genes, statistics, oracle.rows)):
        mean = oracle.mean[rows]
        draws = mean[:, None] + np.sqrt(float(oracle.noise[gene_index])) * oracle.kernels[gene_index].draws(generator, draw_count)
        alpha = gene_statistics.covariate_pseudo_inverse @ (gene_statistics.covariates.T @ gene_statistics.target - gene_statistics.loading @ mean)
        active_to_reduced = np.asarray(gene_statistics.tie_map.original_to_reduced, dtype=np.int64)
        scoring.append(ScoringModel.from_reduced_fit(
            active_rows=gene_statistics.active_rows, signed_means=gene_statistics.means, signed_scales=gene_statistics.scales,
            tie_map=gene_statistics.tie_map, member_prior_variances=second_moment[rows][active_to_reduced], beta_reduced=mean,
            posterior_draws_reduced=draws, alpha=alpha, trait_type=TraitType.QUANTITATIVE, predictive_intercept_shift=0.0,
        ))
    certificate = FitCertificate(
        remaining_gain=np.array([outer.remaining_gain]),
        newton_decrement=np.array([outer.newton_decrement]),
        smoothing_gradient=np.array([outer.step.smoothing_gradient]),
        stationarity_steps=(outer.step.stationarity_steps,),
        stationarity_errors=(outer.step.stationarity_errors,),
        mean_move=np.array([oracle.mean_move]),
        draw_tolerance=np.array([float(np.sum(oracle.effective)) / draw_count]),
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
    )
    profile = dict(oracle.profile) | {
        "stage0_seconds": stage0_seconds, "total_seconds": time.perf_counter() - started, "genes": len(genes),
        "reduced": int(prior.variant_count), "coefficients": int(prior.coefficient_size), "classes": int(prior.class_count),
        "outer_iterations": int(outer.iterations),
    }
    return PooledFit(
        scoring=tuple(scoring), noise_variance=oracle.noise.copy(), hyperparameters=outer.hyperparameters, prior=prior,
        gene_rows=oracle.rows, certificate=certificate, profile=profile,
    )
