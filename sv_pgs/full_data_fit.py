"""Stage 2: the one model fitted on the full data by decoupled EP-EB, and its scoring models.

For every model (a quantitative trait on its training rows) q(beta) is ``dual_solve.DualGaussian``'s Gaussian:
its mean is exact on the full-data operator and certified in the posterior metric, and non-positive sites are
eliminated exactly, so EP is unclipped. The EP sites are matched against q's marginal variances from
``marginal_variances`` (leave-block-out marginals), frozen between refreshes; block-Jacobi variances are block
conditionals, which bias the fixed point by 1-3% however long the fit runs (novel-inference, [sim-only]). The
variant side, from tilted moments to the prior's empirical Bayes, is ``scale_mixture_ep``.

The decoupled scheme alternates:
1. Refresh: solve the mean at the current sites, freeze the marginal variances z and the cavity precisions
   P = 1/z - tau. Negative sites are halved while the global precision is not positive definite or a cavity not
   proper; non-negative sites always pass, so this ends, and only the path to the fixed point changes.
2. Mean-only EP: with P frozen, each site moves toward the one making q's marginal the tilted law, and the mean
   is re-solved. With frozen variances this is the stationarity of a convex function of the mean (math-epeb). A
   pass that does not shrink the move estimates an eigenvalue of the site map at or past -1 (rho = sqrt of the
   move ratio), and the damping 1/(1 + rho) sends it to zero. It stops when the undamped move in the posterior
   metric, sum_j (delta mu_j)^2 / sigma_j^2, is below p_eff / K, the scorer's Monte Carlo resolution with K draws.
3. The noise update sigma^2 = RSS / (n - k - gamma), gamma = p - sum_j tau_j sigma_j^2 (``noise_variance``).
4. The hyper step at the converged EP fixed point, on the evidence with the total curvature B
   (``scale_mixture_ep.hyper_step``); a step that stops reducing what the next one still finds is damped by the
   same spectral rule.
until the hyper step finds less than 1/(2K) nats (its start decrement plus its gain).

Stage 1 fills ``WarmStart``. Until it lands, ``prior_start`` is the stand-in: q at the prior (moment-matched
sites), which reaches the same fixed point, only later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from sv_pgs._typing import F64Array, I64Array
from sv_pgs.dual_solve import DualGaussian
from sv_pgs.fast_scoring import ScoringModel
from sv_pgs.genotype_statistics import GenotypeSufficientStatistics
from sv_pgs.marginal_variances import BlockGrams, BulkSolve, certificate_tolerance, marginal_variances, variance_jvp
from sv_pgs.config import TraitType
from sv_pgs.scale_mixture_ep import (
    Cavity,
    GaussianPosterior,
    MixtureHyperparameters,
    ScaleMixturePrior,
    derived_lattice,
    hyper_step,
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


@dataclass(frozen=True)
class WarmStart:
    """What Stage 2 takes from Stage 1 for each of the M models: the sites (p, M) over Stage 0's reduced columns, each
    model's prior hyperparameters, and the noise variances (M,)."""

    site_precision: F64Array
    site_shift: F64Array
    hyperparameters: tuple[MixtureHyperparameters, ...]
    noise_variance: F64Array


def covariate_residual_variance(targets: F64Array, training: F64Array, covariates: F64Array) -> F64Array:
    """(M,): each model's residual variance after the covariates alone, over n - k degrees of freedom."""
    noise = np.empty(int(targets.shape[1]))
    for model in range(noise.shape[0]):
        weights = training[:, model]
        normal = covariates.T @ (weights[:, None] * covariates)
        residual = weights * (targets[:, model] - covariates @ np.linalg.solve(normal, covariates.T @ (weights * targets[:, model])))
        noise[model] = float(residual @ residual) / (float(weights.sum()) - covariates.shape[1])
    return noise


def prior_start(prior: ScaleMixturePrior, targets: F64Array, training: F64Array, covariates: F64Array) -> WarmStart:
    """The stand-in for Stage 1: q at the prior (moment-matched sites) at the start density, and each model's noise at
    its covariate-only residual variance (all variance attributed to noise)."""
    hyperparameters = tuple(initial_hyperparameters(prior) for _model in range(int(targets.shape[1])))
    columns = [moment_matched_prior_sites(prior, model_hyperparameters) for model_hyperparameters in hyperparameters]
    return WarmStart(
        site_precision=np.column_stack([column[0] for column in columns]),
        site_shift=np.column_stack([column[1] for column in columns]),
        hyperparameters=hyperparameters,
        noise_variance=covariate_residual_variance(targets, training, covariates),
    )


@dataclass(frozen=True)
class FitCertificate:
    """Why a fit is accepted, per model; recorded in the artifact.

    - ``remaining_gain``: what the last hyper step still found, its start decrement plus its evidence gain, in nats
      (below 1/(2K));
    - ``newton_decrement``: 1/2 g'(B + S)^-1 g at the returned hyperparameters;
    - ``smoothing_gradient``: the B-evidence's largest |dV/drho| over interior weights, by central differences;
    - ``mean_move``: the last EP pass's undamped sum_j (delta mu_j)^2 / sigma_j^2, against ``draw_tolerance`` = p_eff / K;
    - ``mean_error``: the certified ||mu_hat - mu||_A of the final solve;
    - ``negative_sites``: sites with negative precision (allowed; EP is unclipped);
    - ``effective_effects``: p_eff = p - sum_j tau_j sigma_j^2.
    """

    remaining_gain: F64Array
    newton_decrement: F64Array
    smoothing_gradient: F64Array
    mean_move: F64Array
    draw_tolerance: F64Array
    mean_error: F64Array
    negative_sites: I64Array
    effective_effects: F64Array
    outer_iterations: int


@dataclass(frozen=True)
class FullDataFit:
    gaussian: DualGaussian
    site_precision: F64Array
    site_shift: F64Array
    hyperparameters: tuple[MixtureHyperparameters, ...]
    noise_variance: F64Array
    certificate: FitCertificate


def _refresh(
    gaussian: DualGaussian,
    statistics: GenotypeSufficientStatistics,
    site_precision: F64Array,
    site_shift: F64Array,
    noise: F64Array,
    error_bound: F64Array,
    probe_ratio: float,
) -> tuple[F64Array, F64Array, list[BulkSolve], list[BlockGrams]]:
    """Solve the mean and freeze the marginal variances; negative sites halve while A is not positive definite or a
    cavity is not proper. Returns the sites, the frozen cavity precisions and each model's solve and Grams."""
    precision = np.array(site_precision, dtype=np.float64, copy=True)
    while True:
        try:
            gaussian.iterate(site_precision=precision, site_shift=site_shift, noise_variance=noise, error_bound=error_bound, probe_residual_ratio=probe_ratio)
            grams = [block_grams(statistics, float(noise[model])) for model in range(noise.shape[0])]
            variances = np.column_stack([marginal_variances(solve, model_grams) for solve, model_grams in zip(gaussian.bulk_solves, grams)])
            frozen = 1.0 / variances - precision
            if np.all(frozen > 0.0):
                return precision, frozen, list(gaussian.bulk_solves), grams
        except np.linalg.LinAlgError:
            pass
        negative = precision < 0.0
        if not np.any(negative):
            raise FloatingPointError("the full-data precision is not positive definite with non-negative sites")
        precision[negative] *= 0.5


def _posterior(gaussian: DualGaussian, solve: BulkSolve, grams: BlockGrams, model: int, error_bound: float) -> GaussianPosterior:
    """q's responses at the fixed point for the total curvature: Sigma R by the dual solver, -(Sigma o Sigma) W by the
    leave-block-out map."""
    return GaussianPosterior(
        solve=lambda right: gaussian.posterior_solve(right, model, error_bound),
        variance_jvp=lambda weights: variance_jvp(solve, grams, weights).values,
    )


def fit_full_data(
    *,
    gaussian: DualGaussian,
    statistics: GenotypeSufficientStatistics,
    prior: ScaleMixturePrior,
    start: WarmStart,
    draw_count: int,
    working_bytes: int,
) -> FullDataFit:
    """Decoupled EP-EB on the full data for quantitative models (see the module docstring)."""
    model_count = gaussian.model_count
    site_precision = np.array(start.site_precision, dtype=np.float64, copy=True)
    site_shift = np.array(start.site_shift, dtype=np.float64, copy=True)
    hyperparameters = list(start.hyperparameters)
    noise = np.array(start.noise_variance, dtype=np.float64, copy=True)
    covariate_count = int(gaussian.covariates.shape[1])
    tolerance = 0.5 / draw_count
    effective = np.full(model_count, float(prior.variant_count))
    probe_ratio = _HALF_PRECISION
    previous_remaining, outer_damping, outer = np.full(model_count, np.inf), 1.0, 0
    while True:
        outer += 1
        error_bound = np.sqrt(effective / draw_count)
        site_precision, frozen, solves, grams = _refresh(gaussian, statistics, site_precision, site_shift, noise, error_bound, probe_ratio)
        probe_ratio = min(certificate_tolerance(solve, gaussian.probe_count) for solve in solves)
        previous_move, damping = np.full(model_count, np.inf), 1.0
        while True:
            marginal = 1.0 / (frozen + site_precision)
            mean = np.asarray(gaussian.mean, dtype=np.float64).copy()
            target_precision, target_shift = site_precision.copy(), site_shift.copy()
            for model in range(model_count):
                cavity = Cavity(precision=frozen[:, model], shift=mean[:, model] / marginal[:, model] - site_shift[:, model])
                target_precision[:, model], target_shift[:, model] = site_targets(tilted_moments(prior, hyperparameters[model], cavity, working_bytes), cavity)
            fraction = damping
            while True:
                trial_precision = site_precision + fraction * (target_precision - site_precision)
                trial_shift = site_shift + fraction * (target_shift - site_shift)
                try:
                    certificate = gaussian.iterate(
                        site_precision=trial_precision, site_shift=trial_shift, noise_variance=noise, error_bound=error_bound, probe_residual_ratio=probe_ratio
                    )
                    break
                except np.linalg.LinAlgError:
                    fraction *= 0.5
            site_precision, site_shift = trial_precision, trial_shift
            marginal = 1.0 / (frozen + site_precision)
            # A damped pass moves fraction^2 of the full step's squared size: converge on the full step.
            mean_move = np.sum(np.square(np.asarray(gaussian.mean) - mean) / marginal, axis=0) / (fraction * fraction)
            effective = prior.variant_count - np.sum(site_precision * marginal, axis=0)
            if np.all(mean_move <= effective / draw_count):
                break
            ratio = float(np.max(mean_move / previous_move))
            if ratio >= 1.0:
                damping = min(damping, 1.0 / (1.0 + np.sqrt(ratio)))
            previous_move = mean_move
        residual_sum_of_squares = gaussian.residual_sum_of_squares()
        noise = np.array([
            noise_variance(
                residual_sum_of_squares=float(residual_sum_of_squares[model]),
                sample_count=int(gaussian.training_counts[model]),
                covariate_count=covariate_count,
                site_precision=site_precision[:, model],
                posterior_variance=marginal[:, model],
            )
            for model in range(model_count)
        ])
        steps = []
        for model in range(model_count):
            cavity = Cavity(precision=frozen[:, model], shift=np.asarray(gaussian.mean)[:, model] / marginal[:, model] - site_shift[:, model])
            posterior = _posterior(gaussian, solves[model], grams[model], model, float(error_bound[model]))
            steps.append(hyper_step(prior, hyperparameters[model], cavity, lambda _view, _coefficients, fixed=posterior: fixed, working_bytes, tolerance))
        remaining = np.array([step.start_decrement + step.evidence_gain for step in steps])
        if np.all(remaining <= tolerance):
            return FullDataFit(
                gaussian=gaussian,
                site_precision=site_precision,
                site_shift=site_shift,
                hyperparameters=tuple(hyperparameters),
                noise_variance=noise,
                certificate=FitCertificate(
                    remaining_gain=remaining,
                    newton_decrement=np.array([step.newton_decrement for step in steps]),
                    smoothing_gradient=np.array([step.smoothing_gradient for step in steps]),
                    mean_move=mean_move,
                    draw_tolerance=effective / draw_count,
                    mean_error=np.asarray(certificate.error_bound, dtype=np.float64),
                    negative_sites=np.sum(site_precision < 0.0, axis=0).astype(np.int64),
                    effective_effects=effective,
                    outer_iterations=outer,
                ),
            )
        ratio = float(np.max(remaining / previous_remaining))
        if ratio >= 1.0:
            outer_damping = min(outer_damping, 1.0 / (1.0 + np.sqrt(ratio)))
        previous_remaining = remaining
        hyperparameters = [_blended(old, step.hyperparameters, outer_damping) for old, step in zip(hyperparameters, steps)]


def _blended(old: MixtureHyperparameters, new: MixtureHyperparameters, fraction: float) -> MixtureHyperparameters:
    """``fraction`` of the way from ``old`` to ``new``; when a weight moved to or from an edge (0 or infinity), all of it."""
    if not np.array_equal(np.isfinite(old.log_smoothing), np.isfinite(new.log_smoothing)):
        return new
    finite = np.isfinite(new.log_smoothing)
    log_smoothing = new.log_smoothing.copy()
    log_smoothing[finite] = old.log_smoothing[finite] + fraction * (new.log_smoothing[finite] - old.log_smoothing[finite])
    return MixtureHyperparameters(coefficients=old.coefficients + fraction * (new.coefficients - old.coefficients), log_smoothing=log_smoothing)


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
