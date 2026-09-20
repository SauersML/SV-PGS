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

from sv_pgs._typing import BoolArray, F64Array
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
    _total_curvature,
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
    tilted_cumulants,
    tilted_moments,
)
from sv_pgs.small_n import (
    _EXACT_RESPONSE_MATRICES,
    _FLOAT_BYTES,
    _LIVE_FIXED_POINTS,
    DenseStatistics,
    _DensePosterior,
    _Kernel,
    _loop_point,
    _new_profile,
    dense_statistics,
    double_loop_sites,
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
        self.gene_divergence = np.full(len(self.rows), np.inf)
        self.gene_noise_gain = np.full(len(self.rows), np.inf)
        self.refusals: list[str] = []
        self.profile = _new_profile()
        # CPU seconds (every thread's) each gene's own work took (factorizations, cavities, double loops), the pooled
        # tilted-moment passes, and the sweeps they ran in: the per-gene cost of a fixed point, for the profiler.
        self.gene_cpu_seconds = np.zeros(len(self.rows))
        self.tilted_cpu_seconds = 0.0
        self.sweeps = 0
        # Each gene's last refresh (its marginal variances and cavity precisions), valid while its sites and noise are
        # unchanged within one fixed-point solve (at fixed x): a gene already certified is not rebuilt (theory-ep).
        self._refreshed = np.zeros(len(self.rows), dtype=bool)
        self._refresh_variances = np.empty(prior.variant_count)
        self._refresh_cavity = np.empty(prior.variant_count)

    @property
    def gene_count(self) -> int:
        return len(self.rows)

    def _iterate(self, gene: int, site_precision: F64Array, site_shift: F64Array) -> None:
        """Gene ``gene``'s exact mean at the sites; ``LinAlgError`` when its precision is not positive definite."""
        started, cpu = time.perf_counter(), time.process_time()
        noise = float(self.noise[gene])
        self._refreshed[gene] = False
        try:
            kernel = _Kernel(self.statistics[gene].design, noise * site_precision)
            self.mean[self.rows[gene]] = kernel.solve(self.scores[gene] + noise * site_shift)
        except np.linalg.LinAlgError:
            # A failed build costs its kernel's formation (n^2 p) before the Cholesky refuses: counted apart (theory-ep).
            self.profile["failed_factorizations"] = self.profile.get("failed_factorizations", 0) + 1
            self.profile["failed_factor_seconds"] = self.profile.get("failed_factor_seconds", 0.0) + time.perf_counter() - started
            raise
        finally:
            self.gene_cpu_seconds[gene] += time.process_time() - cpu
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
            if self._refreshed[gene]:
                variances[rows], cavity_precision[rows] = self._refresh_variances[rows], self._refresh_cavity[rows]
                continue
            while True:
                try:
                    self._iterate(gene, self.site_precision[rows], self.site_shift[rows])
                except np.linalg.LinAlgError:
                    failure = f"gene {gene}: the precision is not positive definite with non-negative sites"
                else:
                    started, cpu = time.perf_counter(), time.process_time()
                    scaled_variances, removed, scaled_precision = self.kernels[gene].cavity()
                    self.profile["variance_seconds"] += time.perf_counter() - started
                    self.gene_cpu_seconds[gene] += time.process_time() - cpu
                    noise = float(self.noise[gene])
                    gene_variances, gene_precision = noise * scaled_variances, scaled_precision / noise
                    if np.all(1.0 + largest[rows] * gene_precision > 0.0):
                        variances[rows], cavity_precision[rows] = gene_variances, gene_precision
                        self._refresh_variances[rows], self._refresh_cavity[rows] = gene_variances, gene_precision
                        self._refreshed[gene] = True
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
        started, cpu = time.perf_counter(), time.process_time()
        targets = site_targets(tilted_moments(self.prior, hyperparameters, cavity, self.working_bytes // _LIVE_FIXED_POINTS), cavity)
        self.profile["tilted_seconds"] += time.perf_counter() - started
        self.tilted_cpu_seconds += time.process_time() - cpu
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

    def _precision_norm(self) -> Callable[[F64Array], F64Array]:
        designs = [gene.design for gene in self.statistics]
        precision, noise, rows = self.site_precision.copy(), self.noise.copy(), self.rows

        def norm(direction: F64Array) -> F64Array:
            values = np.asarray(direction, dtype=np.float64)
            moves = []
            for design, gene_noise, gene_rows in zip(designs, noise, rows):
                image = design.image(values[gene_rows])
                moves.append(float(image @ image) / float(gene_noise) + float(np.sum(precision[gene_rows] * values[gene_rows] ** 2)))
            # One move per gene, each scored as its own model: the prediction check holds each to its own p_eff,g / K.
            return np.array(moves)

        return norm

    def _posterior(self) -> GaussianPosterior:
        # The oracle's memory holds the live kernel sets (the current fixed point's, a trial's, and the snapshot a refused
        # trial restores) and the live fixed points' posteriors (the current one and a trial's), which share the rest.
        kernel_bytes = sum(_held_bytes(kernel) for kernel in self.kernels)
        share = max(self.working_bytes - (_LIVE_FIXED_POINTS + 1) * kernel_bytes, 0) // _LIVE_FIXED_POINTS
        return _PooledPosterior(self.kernels, self.noise.copy(), self.rows, share, self.profile).gaussian_posterior()

    def _snapshot(self) -> dict:
        return {
            "site_precision": self.site_precision.copy(), "site_shift": self.site_shift.copy(), "noise": self.noise.copy(),
            "effective": self.effective.copy(), "kernels": list(self.kernels), "mean": self.mean.copy(),
            "refreshed": self._refreshed.copy(), "refresh_variances": self._refresh_variances.copy(), "refresh_cavity": self._refresh_cavity.copy(),
        }

    def _restore(self, snapshot: dict) -> None:
        self.site_precision, self.site_shift = snapshot["site_precision"].copy(), snapshot["site_shift"].copy()
        self.noise, self.effective = snapshot["noise"].copy(), snapshot["effective"].copy()
        self.kernels, self.mean = list(snapshot["kernels"]), snapshot["mean"].copy()
        self._refreshed, self._refresh_variances = snapshot["refreshed"].copy(), snapshot["refresh_variances"].copy()
        self._refresh_cavity = snapshot["refresh_cavity"].copy()

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
        # A refresh holds at one x only.
        self._refreshed[:] = False
        while True:
            variances, frozen = self._refresh(hyperparameters)
            mean = self.mean.copy()
            cavity = Cavity(precision=frozen, shift=mean / variances - self.site_shift)
            target_precision, target_shift = self._targets(hyperparameters, cavity)
            if not (np.all(np.isfinite(target_precision)) and np.all(np.isfinite(target_shift))):
                raise NoFixedPoint("non-finite EP site targets at these hyperparameters (review-mathbugs N2)")
            # Every gene is scored as its own model, so each is certified on its own (review-mathbugs P1), in evidence
            # units as small_n's fixed point is: its undamped update moves its q by KL(q || q') nats
            # (``_Kernel.update_divergence``, the mean and variance parts to second order) and its noise update gains
            # ``noise_gain`` nats, each at most 1/(2K).
            divergences = np.array([
                self.kernels[gene].update_divergence(
                    float(self.noise[gene]), target_precision[rows] - self.site_precision[rows], target_shift[rows] - self.site_shift[rows], mean[rows]
                )
                for gene, rows in enumerate(self.rows)
            ])
            noises = self._noises(variances)
            gains = np.array([
                noise_gain(float(new), float(old), gene.sample_count, int(gene.covariates.shape[1]))
                for new, old, gene in zip(noises, self.noise, self.statistics)
            ])
            self.gene_divergence, self.gene_noise_gain = divergences, gains
            self.mean_move = float(np.max(divergences)) / tolerance
            self.noise_gain = float(np.max(gains))
            if self.mean_move <= 1.0 and self.noise_gain <= tolerance:
                return FixedPoint(
                    cavity=cavity, posterior=self._posterior(), mean=mean, precision_norm=self._precision_norm(),
                    effective_effects=self.effective.copy(),
                )
            # A certified gene is at its own fixed point at this x (the genes are independent given x): it rests, keeping
            # its sites, noise and refresh; the others take frozen passes and a noise update.
            certified = (divergences <= tolerance) & (gains <= tolerance)
            self._frozen_passes(hyperparameters, frozen, target_precision, target_shift, certified)
            noises = self._noises(1.0 / (frozen + self.site_precision))
            moving = ~certified
            self.noise[moving] = noises[moving]
            self._refreshed[moving] = False

    def _double_loop(self, gene: int, hyperparameters: MixtureHyperparameters) -> None:
        """Gene ``gene``'s EP fixed point by the double loop (``small_n.double_loop_sites``) on its own rows of the pooled
        prior, from its current sites when they lie in EP's domain, else from the prior's moment-matched sites."""
        rows = self.rows[gene]
        prior = _gene_prior(self.prior, rows)
        noise = float(self.noise[gene])
        design = self.statistics[gene].design
        largest = np.exp(log_scale(prior, hyperparameters.coefficients) + prior.log_variance_grid[-1])

        def tilted(cavity_precision: F64Array, cavity_shift: F64Array) -> tuple[F64Array, F64Array, F64Array, F64Array, F64Array]:
            started = time.perf_counter()
            cavity = Cavity(precision=cavity_precision, shift=cavity_shift)
            moments = tilted_moments(prior, hyperparameters, cavity, self.working_bytes)
            third, fourth = tilted_cumulants(prior, hyperparameters, cavity, self.working_bytes)
            self.profile["tilted_seconds"] += time.perf_counter() - started
            return moments.log_normalizer, moments.mean, moments.variance, third, fourth

        self.profile["double_loops"] += 1
        cpu = time.process_time()
        start_precision, start_shift = self.site_precision[rows].copy(), self.site_shift[rows].copy()
        kernel = _Kernel(design, noise * start_precision)
        mean = kernel.solve(self.scores[gene] + noise * start_shift)
        variance = noise * kernel.variances()
        if _loop_point(design, noise, self.scores[gene], start_precision, start_shift, 1.0 / variance, mean / variance, tilted, largest) is None:
            start_precision, start_shift = moment_matched_prior_sites(prior, hyperparameters)
            try:
                kernel = _Kernel(design, noise * start_precision)
            except np.linalg.LinAlgError:
                kernel = None
            if kernel is None or _loop_point(
                design, noise, self.scores[gene], start_precision, start_shift, 1.0 / (noise * kernel.variances()),
                kernel.solve(self.scores[gene] + noise * start_shift) / (noise * kernel.variances()), tilted, largest,
            ) is None:
                # The prior's own variances leave EP's domain at these hyperparameters (they over- or underflow far from
                # where the fit lives): no fixed point exists to compute, so the trial is refused.
                raise NoFixedPoint(f"gene {gene}: the prior's moment-matched sites lie outside EP's domain at these hyperparameters")
        precision, shift = double_loop_sites(
            design, noise, self.scores[gene], start_precision, start_shift, tilted, largest, self.draw_count,
            self.working_bytes // _LIVE_FIXED_POINTS, self.profile,
        )
        self.site_precision[rows], self.site_shift[rows] = precision, shift
        self.gene_cpu_seconds[gene] += time.process_time() - cpu
        self._iterate(gene, precision, shift)

    def _frozen_passes(
        self, hyperparameters: MixtureHyperparameters, frozen: F64Array, target_precision: F64Array, target_shift: F64Array, resting: BoolArray | None = None
    ) -> None:
        """Mean-only EP with the cavity precisions frozen, every gene stepping each sweep with its own damping, until
        the pooled frozen move is below sum_g p_eff,g / K (``small_n._DenseFixedPoints._frozen_passes``)."""
        previous = np.full(self.gene_count, np.inf)
        damping = np.ones(self.gene_count)
        done = np.zeros(self.gene_count, dtype=bool) if resting is None else np.asarray(resting, dtype=bool).copy()
        while True:
            self.sweeps += 1
            moves = np.zeros(self.gene_count)
            for gene, rows in enumerate(self.rows):
                if done[gene]:
                    continue
                mean = self.mean[rows].copy()
                fraction = float(damping[gene])
                delta_precision = target_precision[rows] - self.site_precision[rows]
                delta_shift = target_shift[rows] - self.site_shift[rows]
                if not (np.all(np.isfinite(delta_precision)) and np.all(np.isfinite(delta_shift))):
                    # A non-finite target (a trial density with no mass where this gene's tilted laws live: review-mathbugs
                    # N1/N2) has no damped step to take; the trial has no fixed point here, and the outer loop halves it.
                    raise NoFixedPoint(f"gene {gene}: non-finite EP site targets at these hyperparameters")
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
                            break
                if fraction * move <= _EPSILON * scale:
                    # PD failures halved this gene's damped step to its sites' rounding: its EP falls back to the
                    # convergent double loop at these hyperparameters and its noise (MODEL.md section 4), per gene.
                    self._double_loop(gene, hyperparameters)
                    done[gene] = True
                    continue
                self.site_precision[rows], self.site_shift[rows] = trial_precision, trial_shift
                marginal = 1.0 / (frozen[rows] + self.site_precision[rows])
                moves[gene] = float(np.sum(np.square(self.mean[rows] - mean) / marginal)) / (fraction * fraction)
                if not np.isfinite(moves[gene]):
                    raise NoFixedPoint(f"gene {gene}: a non-finite EP move at these hyperparameters")
                ratio = moves[gene] / previous[gene] if previous[gene] > 0.0 else 0.0
                if 0.5 * moves[gene] > 0.5 / self.draw_count and ratio >= 1.0:
                    if damping[gene] < 1.0:
                        # A pass damped by 1/(1 + rho), exact for the map's eigenvalue at -rho^2, still does not contract:
                        # this gene's frozen-cavity map has a mode damping cannot reach, so its EP falls back to the
                        # convergent double loop (small_n 8643ab5; verify-engine's real slices of genes 3 and 4).
                        self._double_loop(gene, hyperparameters)
                        # At its EP fixed point (the double loop's), the gene leaves the frozen passes; the solve's
                        # next refresh certifies it.
                        done[gene] = True
                        continue
                    damping[gene] = 1.0 / (1.0 + np.sqrt(ratio))
            # A gene is done once its own frozen move, in nats (half its squared move in the posterior metric), is at
            # most 1/(2K), as small_n's; the passes end when every gene is.
            done |= 0.5 * moves <= 0.5 / self.draw_count
            if np.all(done):
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
    )
    profile = dict(oracle.profile) | {
        "stage0_seconds": stage0_seconds, "total_seconds": time.perf_counter() - started, "genes": len(genes),
        "reduced": int(prior.variant_count), "coefficients": int(prior.coefficient_size), "classes": int(prior.class_count),
        "outer_iterations": int(outer.iterations),
        "gene_divergence": oracle.gene_divergence.tolist(), "gene_cpu_seconds": oracle.gene_cpu_seconds.tolist(),
        "tilted_cpu_seconds": oracle.tilted_cpu_seconds, "sweeps": oracle.sweeps,
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
