"""Stage 2: the one model fitted on the full data by decoupled EP-EB, and its scoring models.

For every model (a trait on one training set) the posterior of the effects is
the Gaussian q(beta) of ``exact_polish.FullDataGaussian``: its mean is exact
on the full-data operator, and the EP sites take the diagonal of its
block-Jacobi inverse, exact within each LD block. The variant side (tilted
moments, sites, the prior's empirical Bayes) is ``scale_mixture_ep``.

The decoupled scheme (Seeger and Nickisch 2011; the lead's ruling) alternates:
1. Refactor the blocks at the current sites, and freeze the block variances z
   and the cavity precisions P = 1/z - tau.
2. Mean-only EP: with P frozen, a site update makes q's marginal at each
   variant N(tilted mean, tilted variance) for the cavity (P, mu/sigma^2 - nu),
   and the exact mean is re-solved. With frozen variances this is the
   stationarity of a convex function of the mean (math-epeb), so it converges.
   It stops when the mean's move in the posterior metric,
   sum_j (delta mu_j)^2 / sigma_j^2, is below p_eff / K: below the Monte Carlo
   error of the scorer's K posterior draws.
3. The noise update sigma^2 = RSS / (n - k - gamma), gamma = p - sum tau_j sigma_j^2.
4. The hyper step at the converged EP fixed point, where the fixed-cavity
   gradient is the EP evidence's.
until the hyper step's Newton decrement is below 1/(2K) nats and one more EP
pass moves nothing.

Stage 1 fills ``WarmStart``: per model, the sites, the mean, the prior's
hyperparameters and the noise variance. Until Stage 1 lands, ``prior_start``
is the stand-in: q at the prior itself (moment-matched sites, zero mean), which
reaches the same fixed point, only later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Protocol, Sequence

import numpy as np

from sv_pgs._typing import F64Array, I64Array, NDArray, U8Array
from sv_pgs.code_products import CodeBlockTile
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import TraitType
from sv_pgs.exact_polish import FullDataGaussian, GaussianModel
from sv_pgs.fast_scoring import ScoringModel
from sv_pgs.genotype_buffers import SIGNED_CODE_OFFSET
from sv_pgs.genotype_statistics import GenotypeSufficientStatistics
from sv_pgs.scale_mixture_ep import (
    Cavity,
    MixtureHyperparameters,
    ScaleMixturePrior,
    hyper_step,
    initial_hyperparameters,
    moment_matched_prior_sites,
    noise_variance,
    prior_second_moment,
    kernel_floor,
    kernel_top,
    site_targets,
    spacing_bound,
    tilted_moments,
)


class CodeRows(Protocol):
    """8-bit dosage codes by store row, on the store's full sample axis (``dosage_store.DosageStore``)."""

    @property
    def n_samples(self) -> int: ...

    def read_codes(self, start: int, stop: int, sample_indices: NDArray | None = None, out: NDArray | None = None) -> U8Array: ...


class ReducedCodeBlocks:
    """Stage 0's reduced columns, one LD block at a time, as ``exact_polish.GenotypeBlockSource``.

    Each block reads its store rows once per sweep and multiplies through
    ``CodeBlockTile`` at the training means and scales Stage 0 standardized with.
    """

    def __init__(self, codes: CodeRows, statistics: GenotypeSufficientStatistics, array_module: Any, budget: ComputeBudget) -> None:
        self._codes = codes
        self._array_module = array_module
        self._budget = budget
        kept = np.asarray(statistics.tie_map.kept_indices, dtype=np.int64)
        self.store_rows = np.asarray(statistics.active_rows, dtype=np.int64)[kept]
        self.means = np.asarray(statistics.means, dtype=np.float64)[kept]
        self.scales = np.asarray(statistics.scales, dtype=np.float64)[kept]
        self._blocks = [np.asarray(statistics.ld.block(block_index).reduced_columns, dtype=np.int64) for block_index in range(statistics.ld.block_count)]

    @property
    def sample_count(self) -> int:
        return int(self._codes.n_samples)

    @property
    def array_module(self) -> Any:
        return self._array_module

    @property
    def block_variant_indices(self) -> Sequence[I64Array]:
        return self._blocks

    def iter_tiles(self) -> Iterator[tuple[int, CodeBlockTile]]:
        for block_index, columns in enumerate(self._blocks):
            rows = self.store_rows[columns]
            codes = self._codes.read_codes(int(rows[0]), int(rows[-1]) + 1)[rows - rows[0]]
            signed = (codes.astype(np.int16) - int(SIGNED_CODE_OFFSET)).astype(np.int8)
            yield block_index, CodeBlockTile(
                self._array_module.asarray(signed), self.means[columns], self.scales[columns], self._array_module, self._budget
            )


def density_lattice(
    statistics: GenotypeSufficientStatistics, target_index: int, noise: float, log_variance_offset: F64Array, tolerance: float
) -> tuple[F64Array, float, float]:
    """The start lattice (nodes, floor, top) for one trait, from Stage 0's single-variant likelihoods.

    Variant j's own likelihood has precision n R_jj / sigma^2 and shift X~_j' y / sigma^2, which bound
    every cavity's. The floor and top are ``kernel_floor`` and ``kernel_top`` at ``tolerance``; the
    spacing is ``spacing_bound`` with M_j / Z_j at its minimum of one (the fit refines it against the
    fitted density); the lattice reaches past the kernel range by the range's own width on each side.
    """
    ld = statistics.ld
    single_precision = statistics.sample_count * ld.ld_diagonal() / noise
    single_shift = np.concatenate([ld.block(block_index).projected_score[:, target_index] for block_index in range(ld.block_count)]) / noise
    floor = kernel_floor(single_precision, single_shift, log_variance_offset, tolerance)
    top = kernel_top(single_precision, single_shift, log_variance_offset, floor)
    spacing = spacing_bound(float(single_precision.shape[0]), tolerance)
    width = max(top - floor, spacing)
    nodes = np.arange(floor - width, top + width + spacing, spacing)
    return nodes, floor, top


@dataclass(frozen=True)
class WarmStart:
    """What Stage 2 takes from Stage 1, for each of the M models fitted together.

    ``site_precision``, ``site_shift`` and ``mean`` are (p, M) over Stage 0's
    reduced columns; ``hyperparameters`` holds each model's prior coefficients
    and log penalty weights; ``noise_variance`` is (M,) (1 for a binary model).
    """

    site_precision: F64Array
    site_shift: F64Array
    mean: F64Array
    hyperparameters: tuple[MixtureHyperparameters, ...]
    noise_variance: F64Array


def prior_start(prior: ScaleMixturePrior, models: Sequence[GaussianModel], covariates: NDArray, sample_masks: NDArray) -> WarmStart:
    """The stand-in for Stage 1: q at the prior (moment-matched sites, zero mean) at the start density, and each
    quantitative model's noise at its covariate-only residual variance (all variance attributed to noise)."""
    hyperparameters = tuple(initial_hyperparameters(prior) for _model in models)
    precision_columns, shift_columns, noise = [], [], []
    for model, model_hyperparameters in zip(models, hyperparameters):
        precision, shift = moment_matched_prior_sites(prior, model_hyperparameters)
        precision_columns.append(precision)
        shift_columns.append(shift)
        mask = np.asarray(sample_masks[model.sample_mask_index], dtype=np.float64)
        target = np.asarray(model.targets, dtype=np.float64) - np.asarray(model.predictor_offset, dtype=np.float64)
        weighted = covariates * mask[:, None]
        coefficients = np.linalg.lstsq(weighted.T @ covariates, weighted.T @ target, rcond=None)[0]
        residual = mask * (target - covariates @ coefficients)
        noise.append(
            float(residual @ residual) / (float(mask.sum()) - covariates.shape[1]) if model.trait_type == TraitType.QUANTITATIVE else 1.0
        )
    return WarmStart(
        site_precision=np.column_stack(precision_columns),
        site_shift=np.column_stack(shift_columns),
        mean=np.zeros((prior.variant_count, len(models))),
        hyperparameters=hyperparameters,
        noise_variance=np.asarray(noise, dtype=np.float64),
    )


@dataclass(frozen=True)
class FitCertificate:
    """Why a fit is accepted, per model; recorded in the artifact.

    - ``newton_decrement``: what the last hyper step still found, in nats: the penalized objective's decrement at
      the hyperparameters it started from plus its evidence gain (below 1/(2K) at acceptance);
    - ``smoothing_gradient``: its largest |dV/drho| over interior weights;
    - ``mean_move``: the last EP pass's sum_j (delta mu_j)^2 / sigma_j^2, against ``draw_tolerance`` = p_eff / K;
    - ``gradient_relative_norm``: the exact E-step gradient at the final state;
    - ``negative_sites``: the number of sites with negative precision (allowed; EP is unclipped);
    - ``effective_effects``: p_eff = p - sum_j tau_j sigma_j^2.
    """

    newton_decrement: F64Array
    smoothing_gradient: F64Array
    mean_move: F64Array
    draw_tolerance: F64Array
    gradient_relative_norm: F64Array
    negative_sites: I64Array
    effective_effects: F64Array
    outer_iterations: int


@dataclass(frozen=True)
class FullDataFit:
    gaussian: FullDataGaussian
    site_precision: F64Array
    site_shift: F64Array
    hyperparameters: tuple[MixtureHyperparameters, ...]
    noise_variance: F64Array
    certificate: FitCertificate


def marginal_variances(gaussian: FullDataGaussian) -> F64Array:
    """(p, M): the variances the EP sites are matched against, frozen until the next refactor.

    The block-Jacobi diagonal is each block's variance with the other blocks' effects held fixed: it
    leaves out that their signal is noise for this block, so it understates q's marginal variances and
    biases the EP fixed point (novel-inference). novel-inference's leave-block-out map replaces it.
    """
    return gaussian.block_variances()


def _positive_definite_refactor(
    gaussian: FullDataGaussian, site_precision: F64Array, site_shift: F64Array, noise: F64Array, tolerance: float, exact_curvature: NDArray
) -> tuple[Any, F64Array, F64Array]:
    """Refactor the blocks at the sites, re-solve the mean and freeze the cavity precisions P = 1/z - tau.

    Every negative site precision is halved while a block is not positive definite or a cavity is not proper
    (P <= 0: next to negative sites a block's Schur correction can exceed a variant's own data information).
    Non-negative sites always give a positive-definite precision and proper cavities, so this ends; it only
    shortens the path to the EP fixed point, which does not depend on it. Returns the certificate, the sites
    and P.
    """
    precision = np.array(site_precision, dtype=np.float64, copy=True)
    while True:
        try:
            certificate = gaussian.iterate(
                site_precision=precision, site_shift=site_shift, noise_variance=noise, tolerance=tolerance, refactor=True, exact_curvature=exact_curvature
            )
            frozen = 1.0 / marginal_variances(gaussian) - precision
            if np.all(frozen > 0.0):
                return certificate, precision, frozen
        except (FloatingPointError, np.linalg.LinAlgError):
            pass
        negative = precision < 0.0
        if not np.any(negative):
            raise FloatingPointError("the full-data precision is not positive definite with non-negative sites")
        precision[negative] *= 0.5


def _damped_site_update(
    gaussian: FullDataGaussian,
    site_precision: F64Array,
    site_shift: F64Array,
    target_precision: F64Array,
    target_shift: F64Array,
    noise: F64Array,
    tolerance: float,
    exact_curvature: NDArray,
) -> tuple[Any, F64Array, F64Array]:
    """Move the sites to their targets and re-solve the mean; halve the move while the full-data precision is
    not positive definite (a negative site can make it indefinite, and block CG then breaks down).

    Only the site step is damped: every accepted state is an exact mean for its sites, and the EP fixed point
    does not depend on the steps taken to reach it.
    """
    fraction = 1.0
    while True:
        precision = site_precision + fraction * (target_precision - site_precision)
        shift = site_shift + fraction * (target_shift - site_shift)
        try:
            certificate = gaussian.iterate(
                site_precision=precision, site_shift=shift, noise_variance=noise, tolerance=tolerance, refactor=False, exact_curvature=exact_curvature
            )
        except (FloatingPointError, np.linalg.LinAlgError):
            fraction *= 0.5
            if fraction < float(np.finfo(np.float64).eps):
                raise
            continue
        return certificate, precision, shift


def _solve_tolerance(gaussian: FullDataGaussian, effective_effects: F64Array, draw_count: int, site_precision: F64Array) -> float:
    """CG relative tolerance whose mean error, in the posterior metric, is below p_eff / K for every model.

    With the A-norm of the mean, mu' A mu = ||X~ mu||^2 / sigma^2 + sum tau mu^2, a relative error
    e has squared A-norm e^2 mu' A mu; the tolerance is the largest e with that below p_eff / K.
    Before any mean exists (zero energy) the solve runs at half of double precision.
    """
    genetic = gaussian.device.to_host(gaussian.genetic_image)
    energy = np.sum(genetic * genetic, axis=0) / gaussian.noise_variance + np.sum(site_precision * gaussian.mean * gaussian.mean, axis=0)
    half_precision = float(np.finfo(np.float64).eps) ** 0.5
    if not np.all(energy > 0.0):
        return half_precision
    return float(np.sqrt(min(float(np.min(effective_effects / draw_count / energy)), 1.0)))


def fit_full_data(
    *,
    source: Any,
    prior: ScaleMixturePrior,
    models: Sequence[GaussianModel],
    covariates: NDArray,
    sample_masks: NDArray,
    start: WarmStart,
    draw_count: int,
    working_bytes: int,
    seed: int,
) -> FullDataFit:
    """Decoupled EP-EB on the full data for quantitative models (see the module docstring)."""
    if any(model.trait_type != TraitType.QUANTITATIVE for model in models):
        raise ValueError("binary traits need logistic EP sites, which Stage 2 does not have yet")
    gaussian = FullDataGaussian(
        source=source, models=models, covariates=covariates, sample_masks=sample_masks, initial_mean=start.mean, seed=seed
    )
    model_count = len(models)
    site_precision = np.array(start.site_precision, dtype=np.float64, copy=True)
    site_shift = np.array(start.site_shift, dtype=np.float64, copy=True)
    hyperparameters = list(start.hyperparameters)
    noise = np.array(start.noise_variance, dtype=np.float64, copy=True)
    exact_curvature = np.zeros(model_count, dtype=bool)
    training_counts = np.asarray(sample_masks, dtype=np.float64)[[model.sample_mask_index for model in models]].sum(axis=1)
    covariate_count = int(np.asarray(covariates).shape[1])
    effective = np.full(model_count, float(prior.variant_count))
    hyper_tolerance = 0.5 / draw_count
    outer = 0
    while True:
        outer += 1
        tolerance = _solve_tolerance(gaussian, effective, draw_count, site_precision)
        certificate, site_precision, frozen_precision = _positive_definite_refactor(
            gaussian, site_precision, site_shift, noise, tolerance, exact_curvature
        )
        converged, previous_move = False, np.full(model_count, np.inf)
        while True:
            marginal_variance = 1.0 / (frozen_precision + site_precision)
            previous_mean = gaussian.mean.copy()
            target_precision, target_shift = site_precision.copy(), site_shift.copy()
            for model_index in range(model_count):
                cavity = Cavity(
                    precision=frozen_precision[:, model_index],
                    shift=gaussian.mean[:, model_index] / marginal_variance[:, model_index] - site_shift[:, model_index],
                )
                moments = tilted_moments(prior, hyperparameters[model_index], cavity, working_bytes)
                target_precision[:, model_index], target_shift[:, model_index] = site_targets(moments, cavity)
            certificate, site_precision, site_shift = _damped_site_update(
                gaussian, site_precision, site_shift, target_precision, target_shift, noise, tolerance, exact_curvature
            )
            marginal_variance = 1.0 / (frozen_precision + site_precision)
            mean_move = np.sum(np.square(gaussian.mean - previous_mean) / marginal_variance, axis=0)
            effective = prior.variant_count - np.sum(site_precision * marginal_variance, axis=0)
            if np.all(mean_move <= effective / draw_count):
                converged = True
                break
            # The mean-only iteration contracts only while the frozen variances are close to q's (math-epeb);
            # once a pass stops shrinking the move, refresh them.
            if np.any(mean_move >= previous_move):
                break
            previous_move = mean_move
        if not converged:
            continue
        residual_sum_of_squares = gaussian.residual_sum_of_squares()
        noise = np.array([
            noise_variance(
                residual_sum_of_squares=float(residual_sum_of_squares[model_index]),
                sample_count=int(training_counts[model_index]),
                covariate_count=covariate_count,
                site_precision=site_precision[:, model_index],
                posterior_variance=marginal_variance[:, model_index],
            )
            for model_index in range(model_count)
        ])
        steps = []
        for model_index in range(model_count):
            cavity = Cavity(
                precision=frozen_precision[:, model_index],
                shift=gaussian.mean[:, model_index] / marginal_variance[:, model_index] - site_shift[:, model_index],
            )
            steps.append(hyper_step(prior, hyperparameters[model_index], cavity, working_bytes, hyper_tolerance))
            hyperparameters[model_index] = steps[-1].hyperparameters
        remaining = np.array([step.start_decrement + step.evidence_gain for step in steps])
        if np.all(remaining <= hyper_tolerance):
            return FullDataFit(
                gaussian=gaussian,
                site_precision=site_precision,
                site_shift=site_shift,
                hyperparameters=tuple(hyperparameters),
                noise_variance=noise,
                certificate=FitCertificate(
                    newton_decrement=remaining,
                    smoothing_gradient=np.array([step.smoothing_gradient for step in steps]),
                    mean_move=mean_move,
                    draw_tolerance=effective / draw_count,
                    gradient_relative_norm=np.asarray(certificate.gradient_relative_norm, dtype=np.float64),
                    negative_sites=np.sum(site_precision < 0.0, axis=0).astype(np.int64),
                    effective_effects=effective,
                    outer_iterations=outer,
                ),
            )


def scoring_models(
    fit: FullDataFit, prior: ScaleMixturePrior, statistics: GenotypeSufficientStatistics, models: Sequence[GaussianModel], draw_count: int
) -> list[ScoringModel]:
    """One ``fast_scoring.ScoringModel`` per model: the posterior mean, K exact posterior draws and the covariate
    coefficients, expanded from the reduced columns to every active store row.

    A tie group's members share their representative's prior, so the effect splits equally among them.
    """
    gaussian = fit.gaussian
    draws = gaussian.draws(site_precision=fit.site_precision, draw_count=draw_count, tolerance=np.finfo(np.float64).eps ** 0.5)
    tie_map = statistics.tie_map
    active_to_reduced = np.asarray(tie_map.original_to_reduced, dtype=np.int64)
    alpha = gaussian.device.to_host(gaussian.alpha)
    scoring = []
    for model_index, model in enumerate(models):
        reduced_variance = prior_second_moment(prior, fit.hyperparameters[model_index])
        scoring.append(ScoringModel.from_reduced_fit(
            active_rows=np.asarray(statistics.active_rows, dtype=np.int64),
            signed_means=np.asarray(statistics.means, dtype=np.float64),
            signed_scales=np.asarray(statistics.scales, dtype=np.float64),
            tie_map=tie_map,
            member_prior_variances=reduced_variance[active_to_reduced],
            beta_reduced=gaussian.mean[:, model_index],
            posterior_draws_reduced=draws[:, model_index, :],
            alpha=alpha[:, model_index],
            trait_type=model.trait_type,
            predictive_intercept_shift=0.0,
        ))
    return scoring
