"""Stage 2: each model fitted on the full data by EP-EB, and its scoring models.

For every model (a quantitative trait on its training rows) q(beta) is ``dual_solve.DualGaussian``'s Gaussian:
its mean is exact on the full-data operator and certified in the posterior metric, and non-positive sites are
eliminated exactly, so EP is unclipped. Every cavity comes from q's certified marginal variances
(``marginal_variances``, leave-block-out; lead ruling). Block-Jacobi variances are block conditionals: their
cavity-precision errors reach p99 28-58% at production (speed-floor [semi-real]), and they appear nowhere here.
The variant side, from tilted moments to the prior's empirical Bayes, is ``scale_mixture_ep``.

The fit starts from the prior itself: moment-matched sites, the start density, and each model's covariate-only
residual variance. ``scale_mixture_ep.fit_hyperparameters`` then alternates two steps.
1. The EP fixed point at the current hyperparameters (``_FullDataFixedPoints``):
   a. Refresh: solve the mean at the current sites and compute the certified marginal variances z and the
      cavities (P = 1/z - tau). Negative sites are halved while the global precision is not positive definite or a
      cavity is not proper. Non-negative sites always pass, so this ends; only the path to the fixed point changes.
      The cavity is the information the data removed, D - z, which amplifies a variance error by about 1/(D q)
      where each variant carries little data; so every refresh is certified on tr(D - z) per block
      (``block_information_certificate``, from the dual solve's own products, with the solve accuracy it derives),
      and a violated block refuses the fixed point (novel-inference: the equivalent failed there).
   b. Check, at the refresh. The undamped EP update from these cavities moves the mean by
      Sigma (delta nu - delta tau o mu), exactly to first order in the site change. The fixed point is certified when
      that move is at most p_eff / K in the posterior metric (the scorer's Monte Carlo resolution with K draws),
      and when the noise update's evidence gain, n_eff (delta log sigma^2)^2 / 4, is at most 1/(2K).
   c. Otherwise, mean-only EP with P frozen. Each site moves toward the one that makes q's marginal the tilted law,
      and the mean is re-solved; with frozen variances this is the stationarity of a convex function of the mean
      (math-epeb). A pass that does not shrink the move estimates an eigenvalue of the site map at or past -1
      (rho = sqrt of the move ratio), and the damping 1/(1 + rho) sends it to zero. It runs until the frozen move
      is below p_eff / K. Then comes the noise update sigma^2 = RSS / (n - k - gamma), with
      gamma = p - sum_j tau_j z_j (``noise_variance``), and the loop returns to (a).
2. The outer step at that fixed point: the weights by the B-evidence (``hyper_step``), and x by Newton on the
   total curvature B + S, accepted by the natural monotonicity test. It is never the plain EP-EM step, which
   diverges where (A + S)^-1 (B + S) exceeds 2; speed-floor measured up to 5.4 at production [semi-real].
The fit ends when every model's Newton-B decrement plus its weights' remaining gain is at most 1/(2K).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from sv_pgs._typing import F64Array, I64Array
from sv_pgs.config import TraitType
from sv_pgs.dual_solve import DualGaussian
from sv_pgs.fast_scoring import ScoringModel
from sv_pgs.genotype_statistics import GenotypeSufficientStatistics
from sv_pgs.marginal_variances import (
    BlockCertificate,
    BlockGrams,
    block_information_certificate,
    certificate_tolerance,
    information_products,
    information_solve_tolerance,
    marginal_variances,
    variance_jvp,
)
from sv_pgs.scale_mixture_ep import (
    Cavity,
    FixedPoint,
    GaussianPosterior,
    MixtureHyperparameters,
    ScaleMixturePrior,
    derived_lattice,
    fit_hyperparameters,
    initial_hyperparameters,
    moment_matched_prior_sites,
    noise_variance,
    prior_second_moment,
    site_targets,
    tilted_moments,
)

_HALF_PRECISION = float(np.finfo(np.float64).eps) ** 0.5


def stage0_lattice(
    statistics: GenotypeSufficientStatistics, target_index: int, noise: float, log_variance_offset: F64Array, tolerance: float
) -> tuple[F64Array, float, float]:
    """The start lattice for one trait from Stage 0's single-variant likelihoods: precision n R_jj / sigma^2 and shift
    X~_j' y / sigma^2, which bound every cavity's."""
    ld = statistics.ld
    single_precision = statistics.sample_count * ld.ld_diagonal() / noise
    single_shift = np.concatenate([ld.block(block_index).projected_score[:, target_index] for block_index in range(ld.block_count)]) / noise
    return derived_lattice(single_precision, single_shift, log_variance_offset, tolerance)


def block_grams(statistics: GenotypeSufficientStatistics, noise: float) -> BlockGrams:
    """Stage 0's projected Grams in the model's metric W = training / sigma^2: R_b within each block and R_{b,b+1}
    between neighbours, zero across a chromosome's end."""
    ld = statistics.ld
    blocks = tuple(np.asarray(ld.block(block_index).reduced_columns, dtype=np.int64) for block_index in range(ld.block_count))
    within = tuple(np.asarray(ld.block(block_index).projected_gram, dtype=np.float64) / noise for block_index in range(ld.block_count))
    next_cross = []
    for block_index in range(1, ld.block_count):
        cross = ld.adjacent_block(block_index)
        shape = (blocks[block_index - 1].shape[0], blocks[block_index].shape[0])
        next_cross.append(np.zeros(shape) if cross is None else np.asarray(cross, dtype=np.float64) / noise)
    return BlockGrams(blocks=blocks, within=within, next_cross=tuple(next_cross))


def covariate_residual_variance(targets: F64Array, training: F64Array, covariates: F64Array) -> F64Array:
    """(M,): each model's residual variance after the covariates alone, over n - k degrees of freedom."""
    noise = np.empty(int(targets.shape[1]))
    for model in range(noise.shape[0]):
        weights = training[:, model]
        normal = covariates.T @ (weights[:, None] * covariates)
        residual = weights * (targets[:, model] - covariates @ np.linalg.solve(normal, covariates.T @ (weights * targets[:, model])))
        noise[model] = float(residual @ residual) / (float(weights.sum()) - covariates.shape[1])
    return noise


@dataclass(frozen=True)
class FitCertificate:
    """Why a fit is accepted, per model; recorded in the artifact.

    - ``remaining_gain``: the last outer check's Newton-B decrement plus its weights' B-evidence gain, in nats
      (at most 1/(2K)); ``newton_decrement`` is its first part, 1/2 g'|B + S|^-1 g;
    - ``smoothing_gradient``: the B-evidence's largest |dV/drho| over interior weights, by central differences, with
      each difference's step and error bound in ``stationarity_steps`` and ``stationarity_errors``;
    - ``mean_move``: an upper bound on the undamped EP update's squared move of the mean in the posterior metric at
      the final refresh, against ``draw_tolerance`` = p_eff / K; ``noise_gain``: the noise update's evidence gain there;
    - ``mean_error``: the certified ||mu_hat - mu||_A of the final solve;
    - ``information_bound`` and ``information_tolerance``: the final refresh's largest family-wise upper bound on a
      block's relative error in tr(D - Sigma), and the approximation scale it is tested against;
    - ``negative_sites``: sites with negative precision (allowed; EP is unclipped);
    - ``effective_effects``: p_eff = p - sum_j tau_j z_j;
    - ``outer_iterations``, ``halvings`` and ``unresolved``: accepted outer steps, refused trials, and those refused for
      having no EP fixed point;
    - ``prediction_move``: the certifying Newton step's move of q's mean in q's posterior metric, against
      ``prediction_tolerance`` = p_eff / K (MODEL.md: the certificate includes the prediction change);
      ``refreshes`` and ``passes``: certified variance refreshes and mean solves over the whole fit.
    """

    remaining_gain: F64Array
    newton_decrement: F64Array
    smoothing_gradient: F64Array
    stationarity_steps: tuple[F64Array, ...]
    stationarity_errors: tuple[F64Array, ...]
    mean_move: F64Array
    draw_tolerance: F64Array
    noise_gain: F64Array
    mean_error: F64Array
    information_bound: F64Array
    information_tolerance: F64Array
    negative_sites: I64Array
    effective_effects: F64Array
    outer_iterations: I64Array
    halvings: I64Array
    prediction_move: F64Array
    prediction_tolerance: F64Array
    unresolved: I64Array
    refreshes: int
    passes: int


@dataclass(frozen=True)
class FullDataFit:
    gaussian: DualGaussian
    site_precision: F64Array
    site_shift: F64Array
    hyperparameters: tuple[MixtureHyperparameters, ...]
    noise_variance: F64Array
    certificate: FitCertificate


def _norm_bounds(products: F64Array, bound: F64Array) -> tuple[F64Array, F64Array]:
    """Bounds on ||x||_A from r'x_hat when ||x_hat - x||_A <= b, where A x = r.

    r'x_hat = ||x||_A^2 + x'A(x_hat - x), and the last term is at most ||x||_A b in size, so ||x||_A lies between
    the positive roots of t^2 - b t = r'x_hat and t^2 + b t = r'x_hat.
    """
    lower = 0.5 * (np.sqrt(np.maximum(bound * bound + 4.0 * products, 0.0)) - bound)
    upper = 0.5 * (np.sqrt(np.maximum(bound * bound + 4.0 * products, 0.0)) + bound)
    return lower, upper


def _posterior(gaussian: DualGaussian, model: int, grams: BlockGrams, variances: F64Array) -> GaussianPosterior:
    """q's responses at the current refresh for the total curvature: Sigma R by the dual solver, each column to a
    relative error in the posterior metric, and -(Sigma o Sigma) W by the leave-block-out map."""
    solve = gaussian.bulk_solves[model]

    def relative_solve(right: F64Array, relative_tolerance: float) -> F64Array:
        # A column is done when its certified error b is at most e times the lower bound on ||x||_A. The first b
        # takes ||x||_A^2 ~ sum_j r_j^2 z_j, which is exact for independent effects.
        values = np.asarray(right, dtype=np.float64)
        solution = np.zeros_like(values)
        live = np.flatnonzero(np.any(values != 0.0, axis=0))
        bound = relative_tolerance * np.sqrt(np.square(values[:, live]).T @ variances)
        while live.size:
            solved = np.asarray(gaussian.posterior_solve(values[:, live], model, bound), dtype=np.float64)
            lower, _upper = _norm_bounds(np.sum(values[:, live] * solved, axis=0), bound)
            done = bound <= relative_tolerance * lower
            solution[:, live[done]] = solved[:, done]
            bound = np.where(lower > 0.0, relative_tolerance * lower, 0.5 * bound)[~done]
            live = live[~done]
        return solution

    return GaussianPosterior(solve=relative_solve, variance_jvp=lambda weights: variance_jvp(solve, grams, weights).values)


def _precision_norm(gaussian: DualGaussian, model: int, site_precision: F64Array) -> Callable[[F64Array], float]:
    """d -> d' A d for q's posterior precision A = X' P_W X + diag tau at the model's sites, one read of the store:
    u = X d, and P_W = W - W C (C'WC)^-1 C' W with W the model's training rows over its noise variance."""
    array_module = gaussian.array_module
    weights = np.asarray(gaussian.training, dtype=np.float64)[:, model] / float(gaussian.noise_variance[model])
    covariates = np.asarray(gaussian.covariates, dtype=np.float64)
    normal = covariates.T @ (weights[:, None] * covariates)
    precision = np.array(site_precision, dtype=np.float64, copy=True)

    def norm(direction: F64Array) -> float:
        values = array_module.asarray(np.asarray(direction, dtype=np.float64)[:, None])
        image = array_module.zeros((gaussian.source.sample_count, 1))
        for start, stop, tile in gaussian.source.blocks():
            image += tile.matmat(values[start:stop])
        sample = np.asarray(image, dtype=np.float64)[:, 0]
        weighted = weights * sample
        projected = weighted - weights * (covariates @ np.linalg.solve(normal, covariates.T @ weighted))
        return float(sample @ projected + np.sum(precision * np.square(np.asarray(direction, dtype=np.float64))))

    return norm


class _FullDataFixedPoints:
    """``scale_mixture_ep.FixedPoints`` on the full data: each model's certified EP fixed point at its
    hyperparameters, with its noise variance stationary, warm from the previous call (the module docstring's step 1)."""

    def __init__(
        self, gaussian: DualGaussian, statistics: GenotypeSufficientStatistics, prior: ScaleMixturePrior, draw_count: int, working_bytes: int, seed: int
    ) -> None:
        self.gaussian = gaussian
        self.generator = np.random.default_rng(seed)
        self.statistics = statistics
        self.prior = prior
        self.draw_count = draw_count
        self.working_bytes = working_bytes
        model_count = gaussian.model_count
        precision, shift = moment_matched_prior_sites(prior, initial_hyperparameters(prior))
        self.site_precision = np.repeat(precision[:, None], model_count, axis=1)
        self.site_shift = np.repeat(shift[:, None], model_count, axis=1)
        self.noise = covariate_residual_variance(np.asarray(gaussian.targets), np.asarray(gaussian.training), np.asarray(gaussian.covariates))
        self.effective = np.full(model_count, float(prior.variant_count))
        self.probe_ratio = _HALF_PRECISION
        self.mean_move = np.full(model_count, np.inf)
        self.noise_gain = np.full(model_count, np.inf)
        self.mean_error = np.full(model_count, np.inf)
        self.information: list[BlockCertificate] = []
        self.refreshes = 0
        self.passes = 0

    def _iterate(self, site_precision: F64Array, site_shift: F64Array) -> None:
        certificate = self.gaussian.iterate(
            site_precision=site_precision, site_shift=site_shift, noise_variance=self.noise,
            error_bound=np.sqrt(self.effective / self.draw_count), probe_residual_ratio=self.probe_ratio,
        )
        self.mean_error = np.asarray(certificate.error_bound, dtype=np.float64)
        self.passes += 1

    def _refresh(self) -> tuple[F64Array, list[BlockGrams]]:
        """Solve the mean and compute the certified marginal variances (p, M) at the current sites; negative sites
        halve while the precision is not positive definite or a cavity is not proper."""
        gaussian = self.gaussian
        while True:
            try:
                self._iterate(self.site_precision, self.site_shift)
                grams = [block_grams(self.statistics, float(self.noise[model])) for model in range(gaussian.model_count)]
                variances = np.column_stack([marginal_variances(solve, model_grams) for solve, model_grams in zip(gaussian.bulk_solves, grams)])
                if np.all(1.0 / variances - self.site_precision > 0.0):
                    self.refreshes += 1
                    self.information = [self._information(model, variances[:, model], grams[model]) for model in range(gaussian.model_count)]
                    for model, certificate in enumerate(self.information):
                        if np.any(certificate.violated):
                            raise FloatingPointError(
                                f"model {model}: the marginal variances fail the cavity information certificate in "
                                f"{int(np.count_nonzero(certificate.violated))} of {certificate.violated.shape[0]} blocks"
                            )
                    self.probe_ratio = min(certificate_tolerance(solve, gaussian.probe_count) for solve in gaussian.bulk_solves)
                    self.effective = self.prior.variant_count - np.sum(self.site_precision * variances, axis=0)
                    return variances, grams
            except np.linalg.LinAlgError:
                pass
            negative = self.site_precision < 0.0
            if not np.any(negative):
                raise FloatingPointError("the full-data precision is not positive definite with non-negative sites")
            self.site_precision[negative] *= 0.5

    def _information(self, model: int, variances: F64Array, grams: BlockGrams) -> BlockCertificate:
        """Each block's tr(D - Sigma) tested against Rademacher probes of the removed information (D - Sigma) z,
        formed by the dual solve itself (``information_products``) with its derived relative residual."""
        gaussian = self.gaussian
        solve = gaussian.bulk_solves[model]
        tolerance = certificate_tolerance(solve, gaussian.probe_count)
        column_square_norms = np.asarray(gaussian.unit_squares, dtype=np.float64)[:, model] / float(self.noise[model])
        residual = information_solve_tolerance(solve, variances, grams.blocks, column_square_norms, tolerance)
        probes = self.generator.choice(np.array([-1.0, 1.0]), size=(solve.site_precision.shape[0], gaussian.probe_count))
        back_products, _coupling, _residual_norm = gaussian.information_solve(probes, model, residual)
        removed = information_products(solve, np.asarray(back_products, dtype=np.float64))
        return block_information_certificate(solve, variances, grams.blocks, probes, removed, tolerance)

    def _noise(self, variances: F64Array) -> F64Array:
        gaussian = self.gaussian
        residual_sum_of_squares = gaussian.residual_sum_of_squares()
        return np.array([
            noise_variance(
                residual_sum_of_squares=float(residual_sum_of_squares[model]),
                sample_count=int(gaussian.training_counts[model]),
                covariate_count=int(gaussian.covariates.shape[1]),
                site_precision=self.site_precision[:, model],
                posterior_variance=variances[:, model],
            )
            for model in range(gaussian.model_count)
        ])

    def _targets(self, hyperparameters: Sequence[MixtureHyperparameters], cavities: list[Cavity]) -> tuple[F64Array, F64Array]:
        columns = [site_targets(tilted_moments(self.prior, model, cavity, self.working_bytes), cavity) for model, cavity in zip(hyperparameters, cavities)]
        return np.column_stack([column[0] for column in columns]), np.column_stack([column[1] for column in columns])

    def _move_bounds(self, model: int, right: F64Array, threshold: float) -> float:
        """An upper bound on ||Sigma right||_A^2 that decides it against ``threshold``: the solve's bound halves until
        the two-sided bounds from r'x_hat fall on one side."""
        bound = np.array([0.5 * np.sqrt(threshold)])
        while True:
            solved = np.asarray(self.gaussian.posterior_solve(right[:, None], model, bound), dtype=np.float64)
            lower, upper = _norm_bounds(np.array([float(right @ solved[:, 0])]), bound)
            if upper[0] * upper[0] <= threshold or lower[0] * lower[0] > threshold:
                return float(upper[0] * upper[0])
            bound = 0.5 * bound

    def __call__(self, hyperparameters: Sequence[MixtureHyperparameters]) -> list[FixedPoint]:
        gaussian = self.gaussian
        model_count = gaussian.model_count
        tolerance = 0.5 / self.draw_count
        while True:
            variances, grams = self._refresh()
            frozen = 1.0 / variances - self.site_precision
            mean = np.asarray(gaussian.mean, dtype=np.float64).copy()
            cavities = [Cavity(precision=frozen[:, model], shift=mean[:, model] / variances[:, model] - self.site_shift[:, model]) for model in range(model_count)]
            target_precision, target_shift = self._targets(hyperparameters, cavities)
            # The undamped update moves the mean by Sigma (delta nu - delta tau o mu), to first order in the site change.
            right = (target_shift - self.site_shift) - (target_precision - self.site_precision) * mean
            draw_tolerance = self.effective / self.draw_count
            self.mean_move = np.array([self._move_bounds(model, right[:, model], float(draw_tolerance[model])) for model in range(model_count)])
            noise = self._noise(variances)
            degrees = gaussian.training_counts - int(gaussian.covariates.shape[1]) - self.effective
            self.noise_gain = 0.25 * degrees * np.square(np.log(noise / self.noise))
            if np.all(self.mean_move <= draw_tolerance) and np.all(self.noise_gain <= tolerance):
                return [
                    FixedPoint(
                        cavity=cavities[model],
                        posterior=_posterior(gaussian, model, grams[model], variances[:, model]),
                        mean=mean[:, model].copy(),
                        precision_norm=_precision_norm(gaussian, model, self.site_precision[:, model]),
                        effective_effects=float(self.effective[model]),
                    )
                    for model in range(model_count)
                ]
            self._frozen_passes(hyperparameters, frozen, target_precision, target_shift)
            self.noise = self._noise(1.0 / (frozen + self.site_precision))

    def _frozen_passes(self, hyperparameters: Sequence[MixtureHyperparameters], frozen: F64Array, target_precision: F64Array, target_shift: F64Array) -> None:
        """Mean-only EP with the cavity precisions frozen, from the check's targets, until the frozen move is below
        p_eff / K (the module docstring's step 1c)."""
        gaussian = self.gaussian
        model_count = gaussian.model_count
        previous_move, damping = np.full(model_count, np.inf), 1.0
        while True:
            mean = np.asarray(gaussian.mean, dtype=np.float64).copy()
            fraction = damping
            while True:
                trial_precision = self.site_precision + fraction * (target_precision - self.site_precision)
                trial_shift = self.site_shift + fraction * (target_shift - self.site_shift)
                try:
                    self._iterate(trial_precision, trial_shift)
                    break
                except np.linalg.LinAlgError:
                    fraction *= 0.5
            self.site_precision, self.site_shift = trial_precision, trial_shift
            marginal = 1.0 / (frozen + self.site_precision)
            # A damped pass moves fraction^2 of the full step's squared size: converge on the full step.
            mean_move = np.sum(np.square(np.asarray(gaussian.mean) - mean) / marginal, axis=0) / (fraction * fraction)
            if np.all(mean_move <= self.effective / self.draw_count):
                return
            ratio = float(np.max(mean_move / previous_move))
            if ratio >= 1.0:
                damping = min(damping, 1.0 / (1.0 + np.sqrt(ratio)))
            previous_move = mean_move
            new_mean = np.asarray(gaussian.mean, dtype=np.float64)
            cavities = [
                Cavity(precision=frozen[:, model], shift=new_mean[:, model] / marginal[:, model] - self.site_shift[:, model]) for model in range(model_count)
            ]
            target_precision, target_shift = self._targets(hyperparameters, cavities)


def fit_full_data(
    *, gaussian: DualGaussian, statistics: GenotypeSufficientStatistics, prior: ScaleMixturePrior, draw_count: int, working_bytes: int, seed: int
) -> FullDataFit:
    """Stage 2 for quantitative models, from the prior (see the module docstring); ``seed`` draws the certificate's
    variant-side probes."""
    fixed_points = _FullDataFixedPoints(gaussian, statistics, prior, draw_count, working_bytes, seed)
    starts = [initial_hyperparameters(prior) for _model in range(gaussian.model_count)]
    fits = fit_hyperparameters(prior, starts, fixed_points, working_bytes, 0.5 / draw_count)
    return FullDataFit(
        gaussian=gaussian,
        site_precision=fixed_points.site_precision,
        site_shift=fixed_points.site_shift,
        hyperparameters=tuple(fit.hyperparameters for fit in fits),
        noise_variance=fixed_points.noise,
        certificate=FitCertificate(
            remaining_gain=np.array([fit.remaining_gain for fit in fits]),
            newton_decrement=np.array([fit.newton_decrement for fit in fits]),
            smoothing_gradient=np.array([fit.step.smoothing_gradient for fit in fits]),
            stationarity_steps=tuple(fit.step.stationarity_steps for fit in fits),
            stationarity_errors=tuple(fit.step.stationarity_errors for fit in fits),
            mean_move=fixed_points.mean_move,
            draw_tolerance=fixed_points.effective / draw_count,
            noise_gain=fixed_points.noise_gain,
            mean_error=fixed_points.mean_error,
            information_bound=np.array([float(np.max(certificate.upper_bound)) for certificate in fixed_points.information]),
            information_tolerance=np.array([certificate.tolerance for certificate in fixed_points.information]),
            negative_sites=np.sum(fixed_points.site_precision < 0.0, axis=0).astype(np.int64),
            effective_effects=fixed_points.effective,
            outer_iterations=np.array([fit.iterations for fit in fits], dtype=np.int64),
            halvings=np.array([fit.halvings for fit in fits], dtype=np.int64),
            prediction_move=np.array([fit.prediction_move for fit in fits]),
            prediction_tolerance=np.array([fit.prediction_tolerance for fit in fits]),
            unresolved=np.array([fit.unresolved for fit in fits], dtype=np.int64),
            refreshes=fixed_points.refreshes,
            passes=fixed_points.passes,
        ),
    )


def scoring_models(
    fit: FullDataFit, prior: ScaleMixturePrior, statistics: GenotypeSufficientStatistics, trait_types: Sequence[TraitType], draw_count: int, seed: int
) -> list[ScoringModel]:
    """One ``fast_scoring.ScoringModel`` per model: the posterior mean, K exact posterior draws and the covariate
    coefficients, expanded from the reduced columns to every active store row. A tie group's members share their
    representative's prior, so its effect splits in proportion to their prior variances."""
    gaussian = fit.gaussian
    error_bound = np.sqrt(fit.certificate.effective_effects / draw_count)
    draws = np.asarray(gaussian.draws(draw_count=draw_count, error_bound=error_bound, seed=seed), dtype=np.float64)
    active_to_reduced = np.asarray(statistics.tie_map.original_to_reduced, dtype=np.int64)
    alpha = np.asarray(gaussian.alpha, dtype=np.float64)
    mean = np.asarray(gaussian.mean, dtype=np.float64)
    return [
        ScoringModel.from_reduced_fit(
            active_rows=np.asarray(statistics.active_rows, dtype=np.int64),
            signed_means=np.asarray(statistics.means, dtype=np.float64),
            signed_scales=np.asarray(statistics.scales, dtype=np.float64),
            tie_map=statistics.tie_map,
            member_prior_variances=prior_second_moment(prior, fit.hyperparameters[model])[active_to_reduced],
            beta_reduced=mean[:, model],
            posterior_draws_reduced=draws[:, model, :],
            alpha=alpha[:, model],
            trait_type=trait_type,
            predictive_intercept_shift=0.0,
        )
        for model, trait_type in enumerate(trait_types)
    ]
