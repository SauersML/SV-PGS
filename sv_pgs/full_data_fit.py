"""Stage 2: each model fitted on the full data, and its scoring models.

Two fixed points serve ``scale_mixture_ep.fit_hyperparameters`` here: the mean-field product of ``mean_field`` on
the streamed design (``_FullDataMeanField``, ``fit_full_data(inference="mean_field")``: the public route's, since
EP's oracle refused 59 of 65 calls on the wiring store, "the EP refreshes' updates line up with no contraction",
2026-09-21) and EP as this docstring describes below (``inference="ep"``).

For every model (a quantitative trait on its training rows) q(beta) is ``dual_solve.DualGaussian``'s Gaussian:
its mean is exact on the full-data operator and certified in the posterior metric, and non-positive sites are
eliminated exactly, so EP is unclipped. Every cavity comes from q's certified marginal variances
(``marginal_variances``, leave-block-out; lead ruling). Block-Jacobi variances are block conditionals: their
cavity-precision errors reach p99 28-58% at production (speed-floor [semi-real]), and they appear nowhere here.
The variant side, from tilted moments to the prior's empirical Bayes, is ``scale_mixture_ep``.

The fit starts from the prior itself (moment-matched sites) at a start that splits each trait's residual variance
into genetic and noise parts by Haseman-Elston moments of Stage 0's statistics (``moment_starts``), so the start's
genetic variance never exceeds the phenotypic. ``scale_mixture_ep.fit_hyperparameters`` then alternates two steps.
1. The EP fixed point at the current hyperparameters (``_FullDataFixedPoints``):
   a. Refresh: solve the mean at the current sites and compute the certified marginal variances z and the
      cavities (P = 1/z - tau). Negative sites are halved while the global precision is not positive definite or a
      cavity's tilted law is not proper (1 + v P <= 0 at a lattice node). Non-negative sites always pass, so this
      ends; only the path to the fixed point changes.
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
   total curvature B + S, accepted by the natural monotonicity test, with a trust region where B + S is
   indefinite. It is never the plain EP-EM step, which diverges where (A + S)^-1 (B + S) exceeds 2 and has no
   maximum to move to where A + S is indefinite.
The fit ends when every model's Newton-B decrement plus its weights' remaining gain is at most 1/(2K).

A binary model (``binary_likelihood``) runs on the mean-field route only: its likelihood is the Polya-Gamma bound,
which at fixed sites is the weighted Gaussian problem with W = diag(omega), working response z = kappa / omega and unit
noise, known. The dual solver carries its weights as sample weights (``DualGaussian.reweight``), the oracle holds its
sites and rebuilds every metric-bound array from them (the projector, the column squares, the projected targets and
the device panels' Grams, all keyed by the sites), the ELBO it reports is the Bernoulli bound, and the sites' update
xi^2 = E eta^2 stands where the noise's does, its exact gain the certificate's ``noise_gain``. Each mixture component
keeps its own sites, and the scoring model's covariate terms and predictive intercept shift come from each
component's own metric (``BinaryComponent``).
"""

from __future__ import annotations

import contextlib
import hashlib
import weakref
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterator, Sequence

import time

import numpy as np

from sv_pgs._typing import BoolArray, F64Array, I64Array
from sv_pgs.config import TraitType
from sv_pgs.device_sweep import PIECE_COLUMNS, CodePanels, PanelGrams, sweep_piece
from sv_pgs.binary_likelihood import BernoulliSites, calibrated_shift, covariate_evidence
from sv_pgs.draw_laws import ProductMixtureDraws, seed_key, tile_rows
from sv_pgs.dual_solve import DualGaussian, DualModels, _WindowLayout, _host, column_squares
from sv_pgs.fast_scoring import ScoringModel
from sv_pgs.memory_broker import HOST, current_broker, device_pool
from sv_pgs.genotype_statistics import GenotypeSufficientStatistics
from sv_pgs.progress import log
from sv_pgs.krylov_recycle import local_response
from sv_pgs.tie_members import TieGroups, group_sites, member_draws, member_moments, member_weights, tied_groups, tied_weights
from sv_pgs.tie_map import _compact_identity_tie_map
from sv_pgs.marginal_variances import (
    BlockCertificate,
    ControlVariate,
    BlockGrams,
    block_information_certificate,
    cavity_tolerance,
    control_variate,
    probes_to_decide,
    stage_level,
    certificate_level,
    certificate_tolerance,
    information_products,
    information_solve_tolerance,
    ld_extent,
    marginal_variances,
    refined_grams,
    variance_jvp,
    window_width,
    window_working_bytes,
)
from sv_pgs.scale_mixture_ep import (
    Cavity,
    _DEVICE,
    _data_value,
    class_log_density,
    FixedPoint,
    GaussianPosterior,
    MixtureHyperparameters,
    MomentStart,
    ScaleMixturePrior,
    derived_lattice,
    fit_hyperparameters,
    embed_hyperparameters,
    initial_hyperparameters,
    without_annotations,
    log_scale,
    moment_matched_prior_sites,
    moment_start,
    noise_gain,
    noise_variance,
    prior_second_moment,
    site_targets,
    tilted_cumulants,
    tilted_moments,
)

_EPSILON = float(np.finfo(np.float64).eps)
_HALF_PRECISION = _EPSILON ** 0.5
_FLOAT_BYTES = np.dtype(np.float64).itemsize


def stage0_lattice(
    statistics: GenotypeSufficientStatistics, target_index: int, noise: float, log_variance_offset: F64Array, tolerance: float
) -> tuple[F64Array, float, float]:
    """The start lattice for one trait from Stage 0's single-variant likelihoods: precision n R_jj / sigma^2 and shift
    X~_j' y / sigma^2, which bound every cavity's; one per tie member (its group's column, signed), with
    ``log_variance_offset`` over the members (Stage 0's active rows)."""
    ld = statistics.ld
    ties = TieGroups.from_tie_map(statistics.tie_map)
    single_precision = statistics.sample_count * ld.ld_diagonal() / noise
    single_shift = np.concatenate([ld.block(block_index).projected_score[:, target_index] for block_index in range(ld.block_count)]) / noise
    return derived_lattice(single_precision[ties.group], ties.sign * single_shift[ties.group], log_variance_offset, tolerance)


def block_grams(
    statistics: GenotypeSufficientStatistics, noise: float = 1.0, working_bytes: int | None = None, array_module: Any = np,
) -> BlockGrams:
    """Stage 0's projected Grams, R_b within each block and R_{b,b+1} between neighbours (zero across a chromosome's
    end), as the stored float32 arrays themselves: memory-mapped views, no copy. A model's metric W = training /
    sigma^2 enters as ``scale = 1 / noise``; every model of a fit shares the arrays through
    ``dataclasses.replace(grams, scale=...)``, and ``marginal_variances`` promotes one window at a time to float64.
    (Building float64 copies per model and refresh held about 12 GB of each kind per model at p = 466k and ran
    e2e-scale's chr22 fit out of host memory at 45 GB.)

    With ``working_bytes`` the blocks are cut into parts no wider than the data's LD reaches (``ld_extent``) and
    whose leave-block-out windows fit the budget (``window_width`` on ``array_module``): a window of three whole Stage 0
    blocks is sized only by Stage 0's budget (256 GB of float64 working set at bench-sim's 28k-column blocks), not by
    the EP fit's, and its cost, |W|^3 per window, is set by its width. The dual solver and the
    marginal variances must be given the same partition."""
    ld = statistics.ld
    blocks = tuple(np.asarray(ld.block(block_index).reduced_columns, dtype=np.int64) for block_index in range(ld.block_count))
    within = tuple(ld.block(block_index).projected_gram for block_index in range(ld.block_count))
    next_cross = []
    for block_index in range(1, ld.block_count):
        cross = ld.adjacent_block(block_index)
        shape = (blocks[block_index - 1].shape[0], blocks[block_index].shape[0])
        next_cross.append(np.zeros(shape, dtype=np.float32) if cross is None else cross)
    grams = BlockGrams(blocks=blocks, within=within, next_cross=tuple(next_cross), scale=1.0 / noise)
    if working_bytes is None:
        return grams
    widest = max(block.shape[0] for block in blocks)
    extent = ld_extent(grams, statistics.sample_count, working_bytes, array_module)
    return refined_grams(grams, min(window_width(working_bytes, array_module, widest), extent))


def moment_starts(statistics: GenotypeSufficientStatistics, prior: ScaleMixturePrior) -> list[MomentStart]:
    """Each target's EB start (``scale_mixture_ep.moment_start``) from Stage 0's statistics, with u_j = e^(o_j) the
    start's relative prior variances. y'y after the covariates comes from the target and covariate Grams; ||X'y||^2
    from the projected scores; tr G, sum_j u_j G_jj, sum_j u_j ||G e_j||^2 and ||G||_F^2 from the projected Grams,
    within each block and with its neighbours (the LD Stage 0 keeps; farther pairs enter as zero, which only moves
    the start). The Gram sums run on the fit's device (``scale_mixture_ep.device_scope``): each block's float32 Gram
    goes up once and is squared there in float64, rather than on the host core by core (a 28k-column block is a 6 GB
    float64 square, 22 of them at 40,000 samples x 518k records)."""
    xp = _DEVICE.get()
    ld = statistics.ld
    # A group's prior variance is its members' sum (tie_members): u_g = sum_j u_j.
    ties = TieGroups.from_tie_map(statistics.tie_map)
    weights = np.zeros(ties.group_count)
    np.add.at(weights, ties.group, np.exp(prior.log_variance_offset))
    covariate_count = statistics.covariate_rank
    fitted = statistics.covariate_target.T @ statistics.covariate_gram_pseudo_inverse @ statistics.covariate_target
    target_square = np.diag(statistics.target_gram) - np.diag(fitted)
    score_square = np.zeros(target_square.shape[0])
    # Over the reduced columns (the groups), which the Grams index and the weights are summed to.
    column_square = np.zeros(ties.group_count)
    gram_trace = 0.0
    gram_square = 0.0
    weighted_diagonal = 0.0
    previous_columns = None
    for block_index in range(ld.block_count):
        block = ld.block(block_index)
        columns = np.asarray(block.reduced_columns, dtype=np.int64)
        gram = xp.asarray(np.asarray(block.projected_gram)).astype(xp.float64)
        score_square += np.sum(np.square(np.asarray(block.projected_score, dtype=np.float64)), axis=0)
        diagonal = _host(xp.diag(gram))
        gram_trace += float(diagonal.sum())
        weighted_diagonal += float(weights[columns] @ diagonal)
        gram *= gram
        column_square[columns] += _host(gram.sum(axis=0))
        gram_square += float(_host(gram.sum()))
        del gram
        cross = ld.adjacent_block(block_index) if block_index else None
        if cross is not None:
            cross_squares = xp.asarray(np.asarray(cross)).astype(xp.float64)
            cross_squares *= cross_squares
            column_square[previous_columns] += _host(cross_squares.sum(axis=1))
            column_square[columns] += _host(cross_squares.sum(axis=0))
            gram_square += 2.0 * float(_host(cross_squares.sum()))
            del cross_squares
        previous_columns = columns
    return [
        moment_start(
            target_square=float(target_square[target]), residual_dimension=float(statistics.sample_count - covariate_count),
            score_square=float(score_square[target]), gram_trace=gram_trace, weighted_diagonal=weighted_diagonal,
            weighted_square=float(weights @ column_square), gram_square=gram_square,
        )
        for target in range(target_square.shape[0])
    ]


def covariate_residual_variance(targets: F64Array, training: F64Array, covariates: F64Array) -> F64Array:
    """(M,): covariate-only residual variance, over n - rank(C) training directions."""
    noise = np.empty(int(targets.shape[1]))
    for model in range(noise.shape[0]):
        weights = training[:, model]
        selected = weights != 0.0
        design = covariates[selected]
        outcome = targets[selected, model]
        coefficients, _, rank, _ = np.linalg.lstsq(design, outcome, rcond=None)
        residual = outcome - design @ coefficients
        degrees = int(selected.sum()) - rank
        if degrees <= 0:
            raise ValueError("residual variance requires training directions outside the covariate span.")
        noise[model] = float(residual @ residual) / degrees
    return noise


@dataclass(frozen=True)
class FitCertificate:
    """Why a fit is accepted, per model; recorded in the artifact.

    - ``remaining_gain``: the last outer check's Newton-B decrement plus its weights' B-evidence gain and remaining
      gain, in nats (at most 1/(2K)); ``newton_decrement`` is its first part, 1/2 g'|B + S|^-1 g;
    - ``smoothing_gradient``: the B-evidence's largest |dV/drho| over interior weights, from its analytic gradient,
      with the curvature's difference steps in ``stationarity_steps`` and the gradient's error bounds in
      ``stationarity_errors``;
    - ``mean_move``: an upper bound on twice the KL(q || q') still to go to the EP fixed point at the final refresh,
      KL_k / (1 - rho)^2 with KL_k the undamped update's (the mean's move in the posterior metric plus half the
      variances' move, dtau'(Sigma o Sigma) dtau, halved) and rho = sqrt(KL_k / KL_(k-1)) the refreshes' measured
      contraction (theory-ep's ruled form), against ``draw_tolerance`` = 1 / K, i.e. at most 1 / (2K) nats (evidence
      units, defined as p_eff -> 0); ``noise_gain``: the noise update's evidence gain there;
    - ``mean_error``: the certified ||mu_hat - mu||_A of the final solve;
    - ``information_bound`` and ``information_tolerance``: the final refresh's largest family-wise upper bound on a
      block's relative error in tr(D - Sigma), and the smallest per-block tolerance it is tested against
      (``cavity_tolerance``); ``undecided_blocks`` counts blocks left undecided at the probe limit over the fit;
    - ``negative_sites``: sites with negative precision (allowed; EP is unclipped);
    - ``effective_effects``: p_eff = p - sum_j tau_j z_j;
    - ``outer_iterations``, ``halvings`` and ``unresolved``: accepted outer steps, refused trials, and those refused for
      having no EP fixed point, whose reasons ``refusals`` keeps; ``outer_history``: each model's decrement plus remaining
      gain at every outer evaluation (the outer rate);
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
    undecided_blocks: int
    negative_sites: I64Array
    effective_effects: F64Array
    outer_iterations: I64Array
    halvings: I64Array
    prediction_move: F64Array
    prediction_tolerance: F64Array
    unresolved: I64Array
    refusals: tuple[str, ...]
    outer_history: tuple[tuple[float, ...], ...]
    refreshes: int
    passes: int
    # Per model, the outer loop's own stopping criterion (``OuterFit.certified``): its remaining gain is within the
    # tolerance and its certifying step's prediction move within its own.
    #
    # This is NOT certification of the fit, and nothing may report it as such. ``OuterFit.fixed_point_term_measured``
    # is False for every fit this package can produce, so the outer steps' decisions charge the fixed points' own
    # error along them as zero (theory-ep: the oracles' perturbation probes are not wired yet); its docstring says a
    # caller must then treat the result as uncertified. The name says what the outer loop did establish, so no reader
    # has to know that rule to avoid overclaiming. None where a route does not record it.
    outer_criterion_met: BoolArray | None = None
    # The certificate's binding to the state the fit returns (``fit_full_data``). ``restored_best``: the model's
    # returned state is its best-ELBO one, not where the outer loop ended, so the outer loop's remaining gain, Newton
    # decrement and prediction move are withheld (NaN) and its criterion is unmet; ``budget_unresolved``: solves and
    # mixture searches that ended at their derived work budgets (a nonzero count also leaves the criterion unmet);
    # ``mixture_components``: the components of the returned mixture, whose largest mean move and noise gain the
    # fixed-point terms report; ``state_digest``: models x 32 uint8, the SHA-256 of the returned state
    # (``state_digest``). None where a route does not record them, which an artifact refuses.
    restored_best: BoolArray | None = None
    budget_unresolved: I64Array | None = None
    mixture_components: I64Array | None = None
    state_digest: np.ndarray | None = None


@dataclass(frozen=True)
class FullDataFit:
    gaussian: DualGaussian
    site_precision: F64Array
    site_shift: F64Array
    hyperparameters: tuple[MixtureHyperparameters, ...]
    noise_variance: F64Array
    certificate: FitCertificate
    # The mean-field route (``inference == "mean_field"``): each member's q_j by its pseudo-likelihood (omega, shift)
    # and its mean, and the covariate coefficients at the fixed point; None on the EP route, whose scoring reads the
    # dual solver.
    inference: str = "ep"
    member_mean: F64Array | None = None
    member_shift: F64Array | None = None
    member_omega: F64Array | None = None
    covariate_coefficients: F64Array | None = None
    working_bytes: int = 0
    # The mean-field mixture's components (``fit_full_data``): each fixed point's (shift, omega), members x models, omega
    # in the component's own metric; each scorer's draw picks a component from the weights
    # (``draw_laws.ProductMixtureDraws``), and ``member_mean`` is their weighted average.
    member_components: tuple[tuple[F64Array, F64Array], ...] = ()
    # Each component's weight per model (components x models), exp(ELBO) normalized (``small_n._mixture_weights``).
    component_weights: F64Array | None = None
    # Per component, per model: a binary model's ``BinaryComponent`` (its covariate terms and training-row predictor
    # moments in its own metric), None for a quantitative model.
    binary_components: tuple = ()


@dataclass(frozen=True)
class BinaryComponent:
    """A binary model's mixture component in its own metric W (its final Polya-Gamma sites): the covariates'
    conditional mean at the component's effects, alpha = (C'WC)^+ C'W (z - X m), their conditional loading
    (C'WC)^+ C'W X over the reduced columns (k x groups: a draw b moves them by -loading (b - m) in the group sums), their
    conditional covariance (C'WC)^+, and E eta, Var eta on the training rows (``_FullDataMeanField._predictor_moments``)
    with the rows' labels, for the predictive's calibration."""

    alpha: F64Array
    loading: F64Array
    covariance: F64Array
    # alpha + loading m_g, m_g the component's mean effects summed per reduced column (signed): a draw b's covariates'
    # conditional mean is anchor - loading b_g.
    anchor: F64Array
    predictor_mean: F64Array
    predictor_variance: F64Array
    labels: F64Array


def _norm_bounds(products: F64Array, bound: F64Array) -> tuple[F64Array, F64Array]:
    """Bounds on ||x||_A from r'x_hat when ||x_hat - x||_A <= b, where A x = r.

    r'x_hat = ||x||_A^2 + x'A(x_hat - x), and the last term is at most ||x||_A b in size, so ||x||_A lies between
    the positive roots of t^2 - b t = r'x_hat and t^2 + b t = r'x_hat.
    """
    lower = 0.5 * (np.sqrt(np.maximum(bound * bound + 4.0 * products, 0.0)) - bound)
    upper = 0.5 * (np.sqrt(np.maximum(bound * bound + 4.0 * products, 0.0)) + bound)
    return lower, upper


def _posterior(gaussian: DualGaussian, model: int, grams: BlockGrams, variances: F64Array, ensure: Callable[[], None]) -> GaussianPosterior:
    """q's responses at this refresh for the total curvature: Sigma R by the dual solver, each column to a relative
    error in the posterior metric, and -(Sigma o Sigma) W by the leave-block-out map. ``ensure`` puts the dual
    solver back at this refresh's sites before it is asked (a later trial may have moved it)."""
    solve = gaussian.bulk_solves[model]

    def relative_solve(right: F64Array, relative_tolerance: float) -> F64Array:
        # A column is done when its certified error b is at most e times the lower bound on ||x||_A. The first b
        # takes ||x||_A^2 ~ sum_j r_j^2 z_j, which is exact for independent effects.
        values = np.asarray(right, dtype=np.float64)
        solution = np.zeros_like(values)
        live = np.flatnonzero(np.any(values != 0.0, axis=0))
        bound = relative_tolerance * np.sqrt(np.square(values[:, live]).T @ variances)
        ensure()
        while live.size:
            solved, certified = gaussian.posterior_solve(values[:, live], model, bound)
            solved, certified = np.asarray(_host(solved), dtype=np.float64), np.asarray(_host(certified), dtype=np.float64)
            if not np.all(np.isfinite(certified)):
                raise ValueError("a posterior solve has no certificate at float64's accuracy")
            lower, _upper = _norm_bounds(np.sum(values[:, live] * solved, axis=0), certified)
            # A certificate above the bound asked for is float64's floor: no solve certifies the column better.
            done = (certified <= relative_tolerance * lower) | (certified > bound)
            solution[:, live[done]] = solved[:, done]
            bound = np.where(lower > 0.0, relative_tolerance * lower, 0.5 * bound)[~done]
            live = live[~done]
        return solution

    return GaussianPosterior(
        solve=relative_solve, variance_jvp=lambda weights: variance_jvp(solve, grams, weights, gaussian.array_module).values,
        local_response=local_response(solve, grams),
    )


def _member_posterior(
    reduced: GaussianPosterior, ties: TieGroups, member_precision: F64Array, group_marginals: F64Array, algebraic: bool = False
) -> GaussianPosterior:
    """q's responses over the tie members from the solver's over the groups (``tie_members``): with w_j = s_j D_j / D_g
    and the within-group conditional covariance C_w = diag(D) - D s s' D / D_g (block diagonal over the tied groups,
    zero for a singleton), Sigma_members = C_w + W Sigma_groups W', so Sigma R = C_w R + W Sigma_groups (W' R), and
    (Sigma o Sigma) V = w^2 o [(Sigma_g o Sigma_g) group_sum(w^2 V)] + (C_w o C_w + 2 C_w o (w w' Sigma_gg)) V within
    each tied group. The block-local preconditioner is the groups' own, carried over where every group is a single
    member (a permutation); a tied model's Krylov solve runs without it."""
    precision = np.asarray(member_precision, dtype=np.float64)
    weight = member_weights(ties, precision, algebraic)
    tied = tied_groups(ties)
    conditionals = []
    for members in tied:
        shares, _group_precision, _total = tied_weights(precision[members], algebraic)
        variance = 1.0 / precision[members]
        signs = ties.sign[members]
        conditionals.append(np.diag(variance) - np.outer(signs * shares, signs * variance))

    def group_sum(values: F64Array) -> F64Array:
        total = np.zeros((ties.group_count,) + values.shape[1:])
        np.add.at(total, ties.group, values)
        return total

    def solve(right: F64Array, relative_tolerance: float) -> F64Array:
        values = np.asarray(right, dtype=np.float64)
        result = weight[:, None] * reduced.solve(group_sum(weight[:, None] * values), relative_tolerance)[ties.group]
        for members, conditional in zip(tied, conditionals):
            result[members] += conditional @ values[members]
        return result

    def variance_jvp(values: F64Array) -> F64Array:
        values = np.asarray(values, dtype=np.float64)
        squared = np.square(weight)[:, None]
        result = squared * reduced.variance_jvp(group_sum(squared * values))[ties.group]
        for members, conditional in zip(tied, conditionals):
            # Var(beta_g) enters through w w' Sigma_gg, the groups' own marginal (the reduced diagonal).
            spread = np.outer(weight[members], weight[members]) * float(group_marginals[ties.group[members[0]]])
            result[members] -= (np.square(conditional) + 2.0 * conditional * spread) @ values[members]
        return result

    local = None
    if not tied and reduced.local_response is not None:
        member_of_group = np.empty(ties.group_count, dtype=np.int64)
        member_of_group[ties.group] = np.arange(ties.member_count)

        def local(left: F64Array, right: F64Array, diagonal: F64Array, weights: F64Array) -> Callable[[F64Array], F64Array]:
            inverse = reduced.local_response(left[member_of_group], right[member_of_group], diagonal[member_of_group], weights[member_of_group])
            return lambda values: inverse(np.asarray(values)[member_of_group])[ties.group]

    return GaussianPosterior(solve=solve, variance_jvp=variance_jvp, local_response=local)


def _precision_norm(gaussian: DualGaussian, model: int, site_precision: F64Array, ties: TieGroups) -> Callable[[F64Array], float]:
    """d -> d' A d for q's posterior precision over the tie members, A = X' P_W X + diag tau at the model's member
    sites, one read of the store: u = X d with each group's column once (its members' signed sum), and
    P_W = W - W C (C'WC)^-1 C' W with W the model's training rows over its noise variance."""
    array_module = gaussian.array_module
    weights = np.asarray(_host(gaussian.training), dtype=np.float64)[:, model] / float(gaussian.noise_variance[model])
    covariates = np.asarray(_host(gaussian.covariates), dtype=np.float64)
    normal = covariates.T @ (weights[:, None] * covariates)
    precision = np.array(site_precision, dtype=np.float64, copy=True)

    def norm(direction: F64Array) -> float:
        grouped = np.zeros(ties.group_count)
        np.add.at(grouped, ties.group, ties.sign * np.asarray(direction, dtype=np.float64))
        values = array_module.asarray(grouped[:, None])
        image = array_module.zeros((gaussian.source.sample_count, 1))
        for start, stop, tile in gaussian.source.blocks():
            image += tile.matmat(values[start:stop])
        sample = np.asarray(_host(image), dtype=np.float64)[:, 0]
        weighted = weights * sample
        projected = weighted - weights * (covariates @ np.linalg.solve(normal, covariates.T @ weighted))
        return float(sample @ projected + np.sum(precision * np.square(np.asarray(direction, dtype=np.float64))))

    return norm


class NoFixedPoint(FloatingPointError):
    """EP has no certified fixed point at these hyperparameters (an improper cavity, a failed cavity information
    certificate, or no positive definite precision): the outer loop refuses the trial. Any other error is a failure."""


class _FullDataFixedPoints:
    """``scale_mixture_ep.FixedPoints`` on the full data: each model's certified EP fixed point at its
    hyperparameters, with its noise variance stationary, warm from the previous call (the module docstring's step 1)."""

    def __init__(
        self,
        gaussian: DualGaussian,
        statistics: GenotypeSufficientStatistics,
        prior: ScaleMixturePrior,
        draw_count: int,
        working_bytes: int,
        seed: int,
        starts: Sequence[MixtureHyperparameters],
        noise: F64Array,
        start_sites: tuple[F64Array, F64Array] | None = None,
    ) -> None:
        self.gaussian = gaussian
        covariates = np.asarray(_host(gaussian.covariates))
        training = np.asarray(_host(gaussian.training))
        self.covariate_ranks = np.array([
            np.linalg.matrix_rank(covariates[training[:, model] != 0.0]) if covariates.shape[1] else 0
            for model in range(gaussian.model_count)
        ], dtype=np.int64)
        self.generator = np.random.default_rng(seed)
        self.statistics = statistics
        self.prior = prior
        self.draw_count = draw_count
        self.working_bytes = working_bytes
        # Stage 0's Grams, built once per fit and shared by every model and refresh (``block_grams``), over blocks cut
        # so that the window algebra's working set fits what the fit can give it: its budget, and on a device no more
        # than is free once the dual solver holds its state (the resident codes: 21 GB of an A40's 48 at bench-sim's
        # 518k x 40k), measured here. The dual solver's windows are laid on the same parts (its window cross products
        # are gathered on them); it has not solved yet, so nothing it holds depends on the old ones.
        xp = gaussian.array_module
        window_budget = int(working_bytes)
        if xp is not np:
            broker = current_broker()
            pool = device_pool(int(xp.cuda.runtime.getDevice()))
            if broker is not None and pool in broker.meters:
                # What the shared ledger has left on the device (``memory_broker``: its capacity less every live
                # allocation, the resident codes' included), not a free-memory snapshot.
                window_budget = min(window_budget, broker.remaining(pool))
            else:
                free, _total = xp.cuda.runtime.memGetInfo()
                window_budget = min(window_budget, int(free) + int(xp.get_default_memory_pool().free_bytes()))
        self.grams = block_grams(statistics, working_bytes=window_budget, array_module=xp)
        gaussian.windows = _WindowLayout(self.grams, gaussian.source)
        window_bytes = window_working_bytes(self.grams, xp)
        log(f"ep: {len(self.grams.blocks)} leave-block-out windows of at most {max(block.shape[0] for block in self.grams.blocks)} columns, {window_bytes / 1e9:.1f} GB of {window_budget / 1e9:.1f} GB")
        if window_bytes > window_budget and max(block.shape[0] for block in self.grams.blocks) > 1:
            raise MemoryError(f"the leave-block-out windows need {window_bytes} bytes of float64 working set, over the fit's {working_bytes}")
        model_count = gaussian.model_count
        # Tie members keep their own sites and priors; the solver sees each group's signed sum (tie_members).
        self.ties = TieGroups.from_tie_map(statistics.tie_map)
        if prior.variant_count != self.ties.member_count:
            raise ValueError("the prior must be over Stage 0's active rows (the tie members), in their order")
        self.member_blocks = tuple(np.flatnonzero(np.isin(self.ties.group, block)) for block in self.grams.blocks)
        sites = [moment_matched_prior_sites(prior, start) for start in starts]
        self.site_precision = np.column_stack([precision for precision, _shift in sites])
        self.site_shift = np.column_stack([shift for _precision, shift in sites])
        if start_sites is not None:
            # A warm start (``fit_full_data``): the mean-field fixed point's sites, whose Gaussian has q's means and
            # precision; a member with no mean-field variance keeps the prior's moment-matched site.
            precision, shift = (np.asarray(values, dtype=np.float64) for values in start_sites)
            usable = np.isfinite(precision) & np.isfinite(shift)
            self.site_precision = np.where(usable, precision, self.site_precision)
            self.site_shift = np.where(usable, shift, self.site_shift)
        self.noise = np.array(noise, dtype=np.float64, copy=True)
        self.effective = np.full(model_count, float(prior.variant_count))
        self.probe_ratio = _HALF_PRECISION
        self.mean_move = np.full(model_count, np.inf)
        self.noise_gain = np.full(model_count, np.inf)
        self.mean_error = np.full(model_count, np.inf)
        # The mean solve's error bound in q's metric: sqrt(1/K) (its KL 1/(2K)) until a refresh measures the step it
        # must resolve, then that relative accuracy of the step (``_solve``).
        self.mean_bound = np.full(model_count, np.sqrt(1.0 / draw_count))
        self.information: list[BlockCertificate] = []
        self.refreshes = 0
        self.passes = 0
        # The dual solver's state changes with every solve; a FixedPoint records the version it was built at.
        self.version = 0
        # Blocks the information certificate left undecided at its probe limit, over the whole fit.
        self.undecided_blocks = 0
        self.refusals: list[str] = []

    def _member_moments(self, reduced_variances: F64Array | None) -> tuple[F64Array, F64Array]:
        """The members' posterior means and variances (p_members, M) from the solver's groups at the current sites
        (variances zero where only the means are asked)."""
        mean = np.asarray(_host(self.gaussian.mean), dtype=np.float64)
        variances = np.zeros_like(mean) if reduced_variances is None else reduced_variances
        return member_moments(self.ties, self.site_precision, self.site_shift, mean, variances)

    def _iterate(self, site_precision: F64Array, site_shift: F64Array, error_bound: F64Array | None = None) -> None:
        group_precision, group_shift = group_sites(self.ties, site_precision, site_shift)
        certificate = self.gaussian.iterate(
            site_precision=group_precision, site_shift=group_shift, noise_variance=self.noise,
            # The mean's own error in q's metric (``mean_bound`` unless the caller resolves a smaller move).
            error_bound=self.mean_bound if error_bound is None else error_bound, probe_residual_ratio=self.probe_ratio,
        )
        self.mean_error = np.asarray(_host(certificate.error_bound), dtype=np.float64)
        self.passes += 1
        self.version += 1

    def _snapshot(self) -> dict:
        return {
            "site_precision": self.site_precision.copy(), "site_shift": self.site_shift.copy(), "noise": self.noise.copy(),
            "effective": self.effective.copy(), "probe_ratio": self.probe_ratio, "version": self.version,
        }

    def _restore(self, snapshot: dict) -> None:
        """The oracle and the dual solver back at a snapshot's sites."""
        self.site_precision, self.site_shift = snapshot["site_precision"].copy(), snapshot["site_shift"].copy()
        self.noise, self.effective = snapshot["noise"].copy(), snapshot["effective"].copy()
        self.probe_ratio = snapshot["probe_ratio"]
        self._iterate(self.site_precision, self.site_shift)
        snapshot["version"] = self.version

    def _ensure(self, snapshot: dict) -> None:
        if self.version != snapshot["version"]:
            self._restore(snapshot)

    def _refresh(self, hyperparameters: Sequence[MixtureHyperparameters]) -> tuple[F64Array, F64Array, F64Array, list[BlockGrams]]:
        """Solve the mean and compute the certified marginal variances at the current sites: the members' variances
        and means (p_members, M), the groups' variances (p_groups, M) and the models' Grams; negative sites halve while
        the precision is not positive definite or a cavity is not proper."""
        gaussian = self.gaussian
        while True:
            started = time.time()
            try:
                self._iterate(self.site_precision, self.site_shift)
            except np.linalg.LinAlgError:
                failure = "the full-data precision is not positive definite with non-negative sites"
            else:
                solved = time.time()
                grams = [replace(self.grams, scale=1.0 / float(self.noise[model])) for model in range(gaussian.model_count)]
                group_variances = np.column_stack([
                    marginal_variances(solve, model_grams, gaussian.array_module) for solve, model_grams in zip(gaussian.bulk_solves, grams)
                ])
                log(
                    f"ep refresh {self.refreshes + 1}: mean solve {solved - started:.1f} s, marginal variances {time.time() - solved:.1f} s "
                    f"over {len(self.grams.blocks)} windows of at most {max(block.shape[0] for block in self.grams.blocks)} columns"
                )
                try:
                    mean, variances = self._member_moments(group_variances)
                except np.linalg.LinAlgError:
                    mean = variances = None
                # A cavity is proper where the tilted law it makes is: 1 + v P > 0 at every node, v up to u_j e^(t_K) (the
                # kernel's own test). P = 1/z - tau at or just below zero is a variant the data barely inform, not an
                # improper one: a site that is an exact sum of negative sites keeps P ~ -eps there however far it halves.
                largest = np.column_stack([
                    np.exp(log_scale(self.prior, model_hyperparameters.coefficients) + self.prior.log_variance_grid[-1])
                    for model_hyperparameters in hyperparameters
                ])
                improper = np.ones_like(largest, dtype=bool) if variances is None else 1.0 + largest * (1.0 / variances - self.site_precision) <= 0.0
                if not np.any(improper):
                    self.refreshes += 1
                    self.probe_ratio = min(certificate_tolerance(solve, gaussian.probe_count) for solve in gaussian.bulk_solves)
                    # p_eff is known to its rounding, p eps: floored there, so p_eff / K is never zero.
                    count = self.prior.variant_count
                    self.effective = np.maximum(count - np.sum(self.site_precision * variances, axis=0), _EPSILON * count)
                    self.information = [
                        self._information(model, group_variances[:, model], variances[:, model], grams[model], hyperparameters[model], Cavity(
                            precision=1.0 / variances[:, model] - self.site_precision[:, model],
                            shift=mean[:, model] / variances[:, model] - self.site_shift[:, model],
                        ))
                        for model in range(gaussian.model_count)
                    ]
                    return variances, mean, group_variances, grams
                models = sorted(set(np.flatnonzero(np.any(improper, axis=0)).tolist()))
                failure = f"models {models}: a cavity is improper (1 + v (1/z - tau) <= 0 on the lattice) with non-negative sites"
            negative = self.site_precision < 0.0
            if not np.any(negative):
                raise NoFixedPoint(failure)
            self.site_precision[negative] *= 0.5

    def _information(
        self, model: int, variances: F64Array, member_variances: F64Array, grams: BlockGrams, hyperparameters: MixtureHyperparameters, cavity: Cavity
    ) -> BlockCertificate:
        """Each block's information tr(D - Sigma) certified against the error the EP fixed point can see.

        The tolerance per block is ``cavity_tolerance``: what K draws cannot see, through the tilted law's response and
        skewness at each site's cavity. The estimate is the control-variate one (the window approximation's own
        (D - Sigma_hat) z subtracted), from Rademacher probes of the removed information (D - Sigma) z formed by the
        dual solve itself, at the relative residual ``information_solve_tolerance`` derives. Stages with fresh probes
        re-test the undecided blocks at ``stage_level``, with the probe count ``probes_to_decide`` asks for: a
        violated block refuses the fixed point (``NoFixedPoint``), and a block still undecided once the probes reach
        p (where probing costs as much as the exact diagonal) is counted, not refused.

        The tolerance is the members' (their tilted laws, over each block's members); the certificate is on the
        groups' information, which the solver forms. A group's error dI_g moves its members' by
        (sum_j D_j^2 / D_g^2) dI_g, so each block's tolerance is scaled by the least D_g^2 / sum_j D_j^2 over its
        groups where that is below one (only where member sites of both signs share a group).
        """
        gaussian = self.gaussian
        solve = gaussian.bulk_solves[model]
        moments = tilted_moments(self.prior, hyperparameters, cavity, self.working_bytes)
        third, fourth = tilted_cumulants(self.prior, hyperparameters, cavity, self.working_bytes)
        response = (0.5 * fourth + moments.mean * third) / np.square(moments.variance)
        skewness = third / moments.variance ** 1.5
        blocks = grams.blocks
        tolerance = cavity_tolerance(
            self.site_precision[:, model], member_variances, response, skewness, self.member_blocks, self.draw_count, float(self.effective[model])
        )
        spread = np.ones(self.ties.group_count)
        for members in tied_groups(self.ties):
            shares, _group_precision, _total = tied_weights(self.site_precision[members, model])
            spread[self.ties.group[members[0]]] = min(1.0 / float(np.sum(np.square(shares))), 1.0)
        tolerance = tolerance * np.array([float(np.min(spread[block])) if block.shape[0] else 1.0 for block in blocks])
        column_square_norms = np.asarray(_host(gaussian.unit_squares), dtype=np.float64)[:, model] / float(self.noise[model])
        level = certificate_level(self.draw_count)
        variant_count = solve.site_precision.shape[0]
        undecided = np.arange(len(blocks))
        probe_count = gaussian.probe_count
        stage = 0
        final = None
        while True:
            subset = tuple(blocks[index] for index in undecided)
            residual = information_solve_tolerance(solve, variances, subset, column_square_norms, tolerance[undecided])
            if not np.isfinite(residual):
                # Nothing bulk in these blocks: every site is resolved, and exact.
                zeros = np.zeros(undecided.shape[0])
                return BlockCertificate(
                    relative_error=zeros, standard_error=zeros, lower_bound=zeros, upper_bound=zeros, tolerance=tolerance[undecided],
                    level=stage_level(level, stage), certified=np.ones(undecided.shape[0], dtype=bool), violated=np.zeros(undecided.shape[0], dtype=bool),
                )
            probes = self.generator.choice(np.array([-1.0, 1.0]), size=(variant_count, probe_count))
            back_products, _coupling, residual_norm = gaussian.information_solve(probes, model, residual)
            reached = float(np.max(np.asarray(_host(residual_norm), dtype=np.float64)))
            if not reached <= residual:
                # float64 stopped the solve above the residual the certificate's products need (it returns the residual
                # reached rather than raise, speed-krylov cf364bd; speed-recycle's note): nothing certifies the block.
                raise NoFixedPoint(
                    f"model {model}: the information solve stops at relative residual {reached:.3e}, above the {residual:.3e} its certificate needs"
                )
            removed = information_products(solve, np.asarray(_host(back_products), dtype=np.float64))
            control = control_variate(solve, grams, probes, gaussian.array_module)
            control = ControlVariate(
                removed_products=control.removed_products, window_information=control.window_information[undecided],
                resolvable=control.resolvable[undecided],
            )
            certificate = block_information_certificate(
                solve, variances, subset, probes, removed, tolerance[undecided], stage_level(level, stage), control
            )
            if np.any(certificate.violated):
                raise NoFixedPoint(
                    f"model {model}: the marginal variances fail the cavity information certificate in "
                    f"{int(np.count_nonzero(certificate.violated))} of {len(blocks)} blocks (stage {stage}, {probe_count} probes)"
                )
            final = certificate
            undecided = undecided[~certificate.certified]
            if not undecided.shape[0] or probe_count >= variant_count:
                self.undecided_blocks += int(undecided.shape[0])
                return certificate
            probe_count = min(max(probes_to_decide(certificate, probe_count), probe_count + 1), variant_count)
            stage += 1

    def _noise(self, variances: F64Array) -> F64Array:
        gaussian = self.gaussian
        residual_sum_of_squares = gaussian.residual_sum_of_squares()
        return np.array([
            noise_variance(
                residual_sum_of_squares=float(residual_sum_of_squares[model]),
                sample_count=int(gaussian.training_counts[model]),
                covariate_count=int(self.covariate_ranks[model]),
                site_precision=self.site_precision[:, model],
                posterior_variance=variances[:, model],
                noise=float(self.noise[model]),
            )
            for model in range(gaussian.model_count)
        ])

    def _targets(self, hyperparameters: Sequence[MixtureHyperparameters], cavities: list[Cavity]) -> tuple[F64Array, F64Array]:
        columns = [site_targets(tilted_moments(self.prior, model, cavity, self.working_bytes), cavity) for model, cavity in zip(hyperparameters, cavities)]
        return np.column_stack([column[0] for column in columns]), np.column_stack([column[1] for column in columns])

    def _move_bounds(
        self, model: int, right: F64Array, threshold: float | None = None, other: F64Array | None = None
    ) -> tuple[float, float, float, float]:
        """Two-sided bounds on ||Sigma right||_A^2 = right' Sigma right over the members: Sigma = C_w + W Sigma_groups W'
        (``_member_posterior``), so the members' within-group part is exact and the groups' part is the solver's.

        With no ``threshold`` the norm is resolved to relative accuracy sqrt(1 / K), the accuracy the mean itself is
        solved to in the posterior metric (``_iterate``), so the refreshes' ratio is measured at the scorer's own
        resolution; with one, the solve's bound halves until the bounds fall on one side of it. Where float64's floor
        stops the solve first (its certificate above the bound asked for, speed-krylov cf364bd), the bounds are
        returned as they are and the caller reads them on the cautious side.

        With ``other``, it also returns other' Sigma right from the same solve and that value's error bound
        (||W'other||_Sigma_groups <= ||W'other||_(D_groups^-1) times the solve's certificate); nan otherwise."""
        precision = self.site_precision[:, model]
        weight = member_weights(self.ties, precision)
        grouped = np.zeros(self.ties.group_count)
        np.add.at(grouped, self.ties.group, weight * right)
        other_grouped = np.zeros(self.ties.group_count)
        if other is not None:
            np.add.at(other_grouped, self.ties.group, weight * other)
        within, within_cross = 0.0, 0.0
        for members in tied_groups(self.ties):
            shares, _group_precision, _total = tied_weights(precision[members])
            variance = 1.0 / precision[members]
            signs = self.ties.sign[members]
            conditional = np.diag(variance) - np.outer(signs * shares, signs * variance)
            within += float(right[members] @ conditional @ right[members])
            if other is not None:
                within_cross += float(other[members] @ conditional @ right[members])
        remaining = None if threshold is None else threshold - within
        if remaining is not None and remaining < 0.0:
            return within, np.inf, np.nan, np.nan
        relative = np.sqrt(1.0 / self.draw_count)
        # The first bound's scale: Sigma_groups <= D_groups^-1 (the data only add precision), so r_g' D_g^-1 r_g bounds
        # the norm from above where the groups' sites are proper; later bounds follow the solve's own lower bound.
        group_precision = np.abs(np.bincount(self.ties.group, weights=precision, minlength=grouped.size))
        scale = float(np.sqrt(np.sum(np.square(grouped) / group_precision)))
        bound = np.array([0.5 * np.sqrt(remaining) if remaining is not None else relative * scale])
        while True:
            solved, certified = self.gaussian.posterior_solve(grouped[:, None], model, bound)
            solved, certified = np.asarray(_host(solved), dtype=np.float64), np.asarray(_host(certified), dtype=np.float64)
            if not np.all(np.isfinite(certified)):
                # No certificate at float64's accuracy: the move cannot be decided, so no certified fixed point here and
                # the outer loop shortens its step.
                raise NoFixedPoint(f"model {model}: a posterior solve for the EP move has no certificate at float64's accuracy")
            lower, upper = _norm_bounds(np.array([float(grouped @ solved[:, 0])]), certified)
            if remaining is None:
                done = certified[0] <= relative * lower[0]
            else:
                done = upper[0] * upper[0] <= remaining or lower[0] * lower[0] > remaining
            if done or certified[0] > bound[0]:
                cross, cross_error = np.nan, np.nan
                if other is not None:
                    cross = within_cross + float(other_grouped @ solved[:, 0])
                    cross_error = float(np.sqrt(np.sum(np.square(other_grouped) / group_precision))) * float(certified[0])
                return within + float(lower[0] * lower[0]), within + float(upper[0] * upper[0]), cross, cross_error
            bound = 0.5 * bound if remaining is not None or not lower[0] > 0.0 else np.array([relative * float(lower[0])])

    def __call__(self, hyperparameters: Sequence[MixtureHyperparameters]) -> list[FixedPoint | None]:
        """Each model's certified EP fixed point, or None for every model when EP has none at these hyperparameters
        (the dual solver's state is shared, so a refusal restores all of it); the reason is kept in ``refusals``."""
        snapshot = self._snapshot()
        try:
            return list(self._solve(hyperparameters))
        except NoFixedPoint as error:
            self._restore(snapshot)
            self.refusals.append(str(error))
            log(f"ep fixed point refused: {error}")
            return [None] * self.gaussian.model_count

    def _solve(self, hyperparameters: Sequence[MixtureHyperparameters]) -> list[FixedPoint]:
        gaussian = self.gaussian
        model_count = gaussian.model_count
        tolerance = 0.5 / self.draw_count
        # The last refresh's KL bounds per model, for the contraction rate; None where there is no earlier refresh of
        # the same map in this call (the first, or after a noise update moved the likelihood under a certified KL).
        previous: tuple[F64Array, F64Array, F64Array, float | None] | None = None
        # The share of each refresh's frozen-pass move the sites take. Undamped at first; a refresh whose KL does not fall
        # measures an oscillating mode (J's eigenvalue -lambda, lambda >= rho), which the fraction f / (1 + rho) removes
        # (the damped map's 1 - f (1 + lambda) at lambda = (1 + rho) / f - 1).
        fraction = 1.0
        # The share taken by the step into the current refresh (None before the first step in this call).
        arrived: float | None = None
        self.mean_bound = np.full(model_count, np.sqrt(1.0 / self.draw_count))
        while True:
            variances, mean, group_variances, grams = self._refresh(hyperparameters)
            frozen = 1.0 / variances - self.site_precision
            mean = mean.copy()
            cavities = [Cavity(precision=frozen[:, model], shift=mean[:, model] / variances[:, model] - self.site_shift[:, model]) for model in range(model_count)]
            target_precision, target_shift = self._targets(hyperparameters, cavities)
            if not (np.all(np.isfinite(target_precision)) and np.all(np.isfinite(target_shift))):
                # A trial whose tilted laws give no finite site (review-mathbugs N2) has no fixed point here: refused,
                # so the outer loop counts it unresolved and shortens its step.
                raise NoFixedPoint("a site target is not finite at these hyperparameters")
            # The undamped update's KL(q || q') to second order in the site change (dtau, dnu): 1/2 r'Sigma r +
            # 1/4 dtau'(Sigma o Sigma) dtau with r = dnu - dtau o mu, the Fisher metric of q's statistics (beta, -beta^2/2)
            # (speed-smalln: the mean part alone left the site variances off). The fixed point is where the refreshes
            # converge, so the certificate is on the distance to it (theory-ep's ruled form, the lead's ruling): with
            # rho = sqrt(KL_k / KL_(k-1)) the measured contraction, the remaining steps' KL sums to at most
            # KL_k / (1 - rho)^2, certified at 1/(2K) nats. For KL_(k-1) = b that is KL_k <= c b / (sqrt b + sqrt c)^2,
            # c = 1/(2K): a threshold on KL_k, taken at b's lower bound so the rate is never understated. Where the sites
            # take a share f of each move, the next step's KL is f^2 KL_k and the threshold c b / (f sqrt b + sqrt c)^2.
            snapshot = self._snapshot()
            posteriors = [
                _member_posterior(
                    _posterior(gaussian, model, grams[model], group_variances[:, model], lambda snapshot=snapshot: self._ensure(snapshot)),
                    self.ties, self.site_precision[:, model], group_variances[:, model],
                )
                for model in range(model_count)
            ]
            precision_step = target_precision - self.site_precision
            right = (target_shift - self.site_shift) - precision_step * mean
            spread = np.array([
                max(-float(precision_step[:, model] @ posteriors[model].variance_jvp(precision_step[:, [model]])[:, 0]), 0.0) for model in range(model_count)
            ])
            budget = 0.5 / self.draw_count
            damped = fraction
            lower, upper = np.empty(model_count), np.empty(model_count)
            certified = np.zeros(model_count, dtype=bool)
            for model in range(model_count):
                fixed = 0.25 * float(spread[model])
                mean_lower, mean_upper, cross, cross_error = self._move_bounds(
                    model, right[:, model], other=None if previous is None else previous[2][:, model]
                )
                lower[model], upper[model] = 0.5 * mean_lower + fixed, 0.5 * mean_upper + fixed
                if previous is None:
                    continue
                last = float(previous[0][model])
                # The ratio is the damped map's contraction only across steps at one share (theory-ep): a refresh
                # reached right after the share changed is measured, not certified.
                steady = previous[3] is None or previous[3] == arrived
                certified[model] = steady and upper[model] <= budget * last / (fraction * np.sqrt(last) + np.sqrt(budget)) ** 2
                if certified[model] or upper[model] <= last:
                    continue
                # Not within the certificate, and not measurably below the last refresh's KL (decided against that KL's
                # lower bound): no contraction at this share, so the next move is damped by the measured rate.
                if fixed < last:
                    mean_lower, mean_upper, _cross, _error = self._move_bounds(model, right[:, model], 2.0 * (last - fixed))
                    lower[model], upper[model] = 0.5 * mean_lower + fixed, 0.5 * mean_upper + fixed
                if not upper[model] <= last:
                    # Which mode fails to contract (theory-ep): successive updates that alternate, r_k' Sigma r_(k-1) < 0,
                    # are J's eigenvalue -lambda, which the damping removes; ones that line up are an eigenvalue at least
                    # 1, which no share contracts (1 - f (1 - mu) >= 1), and that needs the double loop this route lacks.
                    if cross > cross_error:
                        # No share of the refresh map contracts here: EP's fixed point is found by the convergent double
                        # loop instead (``_double_loop``), which lowers the EC free energy at every outer step.
                        log(
                            f"ep: model {model}'s refreshes line up with no contraction (KL {lower[model]:.3e}..{upper[model]:.3e} after {last:.3e}, "
                            f"r_k' Sigma r_(k-1) = {cross:.3e} +- {cross_error:.3e}): the double loop takes over"
                        )
                        return self._double_loop(hyperparameters)
                    damped = min(damped, fraction / (1.0 + float(np.sqrt(upper[model] / last))) if last > 0.0 else 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                rate = np.sqrt(upper / previous[0]) if previous is not None else np.full(model_count, np.inf)
                # Twice the certified distance's KL, against 1 / K (the certificate's units).
                self.mean_move = np.where(rate < 1.0, 2.0 * fraction * fraction * upper / np.square(1.0 - rate), np.inf)
            draw_tolerance = np.full(model_count, 2.0 * budget)
            noise = self._noise(variances)
            self.noise_gain = np.array([
                noise_gain(float(noise[model]), float(self.noise[model]), int(gaussian.training_counts[model]), int(self.covariate_ranks[model]))
                for model in range(model_count)
            ])
            if np.all(certified) and np.all(self.noise_gain <= tolerance):
                return [
                    FixedPoint(
                        cavity=cavities[model],
                        posterior=posteriors[model],
                        mean=mean[:, model].copy(),
                        precision_norm=_precision_norm(gaussian, model, self.site_precision[:, model], self.ties),
                        effective_effects=float(self.effective[model]),
                    )
                    for model in range(model_count)
                ]
            # A noise update moves the likelihood, so the next refresh is of a new map where the sites had met their
            # certificate: the rate starts again there.
            previous = None if np.all(certified) else (lower, upper, right.copy(), arrived)
            fraction = damped
            arrived = fraction
            # The next refresh's mean resolves the next step (f sqrt(2 KL_k) in q's metric) to relative accuracy
            # sqrt(1/K), as the KL itself is: a mean error at the step's own size would be what the ratio measures.
            measured = np.where(lower > 0.0, lower, upper)
            self.mean_bound = np.minimum(np.sqrt(1.0 / self.draw_count), np.sqrt(1.0 / self.draw_count) * fraction * np.sqrt(2.0 * measured))
            start_precision, start_shift = self.site_precision.copy(), self.site_shift.copy()
            self._frozen_passes(hyperparameters, frozen, target_precision, target_shift)
            if fraction < 1.0:
                change = max(float(np.max(np.abs(self.site_precision - start_precision))), float(np.max(np.abs(self.site_shift - start_shift))))
                scale = 1.0 + max(float(np.max(np.abs(start_precision))), float(np.max(np.abs(start_shift))))
                if not fraction * change > _EPSILON * scale:
                    raise NoFixedPoint("no damped refresh moves the sites past their rounding: EP does not converge at these hyperparameters")
                # A convex combination of two sets of sites that each give a positive definite precision gives one too.
                blended_precision = start_precision + fraction * (self.site_precision - start_precision)
                blended_shift = start_shift + fraction * (self.site_shift - start_shift)
                self._iterate(blended_precision, blended_shift)
                self.site_precision, self.site_shift = blended_precision, blended_shift
            self.noise = self._noise(1.0 / (frozen + self.site_precision))

    # the double loop

    def _loop_point(
        self, hyperparameters: Sequence[MixtureHyperparameters], site_precision: F64Array, site_shift: F64Array,
        marginal_precision: F64Array, marginal_shift: F64Array,
    ) -> _LoopPoint | None:
        """The inner problem at these sites (``small_n._loop_point`` on the streamed design), or None outside EP's
        domain: q's precision not positive definite, or an inner cavity (P_s - tau, h_s - nu) whose tilted law is
        improper (1 + v_max P <= 0) or has no finite moments. q's moments are the mean solve's and the windows'
        marginal variances (``marginal_variances``); the cavity certificate is the outer step's."""
        gaussian = self.gaussian
        cavity_precision = marginal_precision - site_precision
        largest = np.column_stack([
            np.exp(log_scale(self.prior, model.coefficients) + self.prior.log_variance_grid[-1]) for model in hyperparameters
        ])
        if not np.all(1.0 + largest * cavity_precision > 0.0):
            return None
        try:
            self._iterate(site_precision, site_shift)
            grams = [replace(self.grams, scale=1.0 / float(self.noise[model])) for model in range(gaussian.model_count)]
            group_variances = np.column_stack([
                marginal_variances(solve, model_grams, gaussian.array_module) for solve, model_grams in zip(gaussian.bulk_solves, grams)
            ])
            mean, variance = member_moments(
                self.ties, site_precision, site_shift, np.asarray(_host(gaussian.mean), dtype=np.float64), group_variances
            )
        except np.linalg.LinAlgError:
            return None
        if not np.all(variance > 0.0):
            return None
        tilted_mean, tilted_variance, third, fourth = np.empty_like(mean), np.empty_like(mean), np.empty_like(mean), np.empty_like(mean)
        for model, model_hyperparameters in enumerate(hyperparameters):
            cavity = Cavity(precision=cavity_precision[:, model], shift=marginal_shift[:, model] - site_shift[:, model])
            moments = tilted_moments(self.prior, model_hyperparameters, cavity, self.working_bytes)
            tilted_mean[:, model], tilted_variance[:, model] = moments.mean, moments.variance
            third[:, model], fourth[:, model] = tilted_cumulants(self.prior, model_hyperparameters, cavity, self.working_bytes)
        values = (tilted_mean, tilted_variance, third, fourth)
        if not (all(np.all(np.isfinite(value)) for value in values) and np.all(tilted_variance > 0.0)):
            return None
        gradient = np.concatenate([mean - tilted_mean, -0.5 * (variance + mean**2 - tilted_variance - tilted_mean**2)])
        return _LoopPoint(site_precision, site_shift, mean, variance, tilted_mean, tilted_variance, third, fourth, gradient)

    def _line_search(
        self, hyperparameters: Sequence[MixtureHyperparameters], point: _LoopPoint, direction: F64Array, marginal_precision: F64Array,
        marginal_shift: F64Array, tolerance: float,
    ) -> _LoopPoint | None:
        """The inner problem's minimum along ``direction`` from ``point``, to where the one-dimensional model's gain
        still to go is at most ``tolerance``: Phi is convex, so its directional derivative phi'(t) = g(t)'d rises
        along the line, the root is bracketed between a point with phi' < 0 and one with phi' >= 0 or outside the
        domain (where Phi is +inf), and the secant of phi' between them gives the next trial and the remaining gain
        phi'(lo)^2 / (2 kappa). The point returned has phi' < 0 on [0, t], so Phi fell along the whole step: a
        certified decrease with no value of Phi (no log determinant). None where no point along d has phi' < 0
        beyond the start's rounding (the start is the line's minimum)."""
        half = direction.shape[0] // 2
        slope = float(point.gradient.ravel() @ direction.ravel())
        low, low_slope, low_point = 0.0, slope, None
        high, high_slope = np.inf, np.nan
        trial = 1.0
        scale = 1.0 + max(float(np.max(np.abs(point.site_precision))), float(np.max(np.abs(point.site_shift))))
        step_size = float(np.max(np.abs(direction)))
        while True:
            if (trial - low) * step_size <= _EPSILON * scale:
                return low_point
            candidate = self._loop_point(
                hyperparameters, point.site_precision + trial * direction[half:], point.site_shift + trial * direction[:half], marginal_precision,
                marginal_shift,
            )
            if candidate is None:
                high, high_slope = trial, np.nan
            else:
                candidate_slope = float(candidate.gradient.ravel() @ direction.ravel())
                if candidate_slope < 0.0:
                    low, low_slope, low_point = trial, candidate_slope, candidate
                else:
                    high, high_slope = trial, candidate_slope
            if np.isfinite(high) and np.isfinite(high_slope):
                curvature = (high_slope - low_slope) / (high - low)
                if low_point is not None and low_slope * low_slope / (2.0 * curvature) <= tolerance:
                    return low_point
                trial = low - low_slope / curvature
                if not low < trial < high:
                    trial = 0.5 * (low + high)
            elif np.isfinite(high):
                trial = 0.5 * (low + high)
            else:
                trial = 2.0 * trial

    def _inner(
        self, hyperparameters: Sequence[MixtureHyperparameters], marginal_precision: F64Array, marginal_shift: F64Array, tolerance: float,
    ) -> tuple[F64Array, F64Array, int]:
        """The inner problem, min Phi over the sites at fixed marginals, by preconditioned nonlinear conjugate
        gradients (Polak-Ribiere+, the site blocks as the preconditioner) with ``_line_search``, from the current
        sites, until 1/2 g' Cov_r^-1 g is at most ``tolerance``: H >= Cov_r (Cov_q is positive semidefinite), so that
        bounds Newton's decrement 1/2 g'H^-1 g, the gain still to go to second order, from above (``small_n._newton_step``'s
        bound; the preconditioner's own decrement can understate it where LD couples the sites). Returns the sites and
        the count of accepted steps."""
        point = self._loop_point(hyperparameters, self.site_precision, self.site_shift, marginal_precision, marginal_shift)
        if point is None:
            # q's own cavities at its own marginals are the sites' cavities, which the outer refresh found proper.
            raise NoFixedPoint("the double loop's inner problem starts outside EP's domain")
        gradient = point.gradient
        preconditioned = _block_solve(point, gradient)
        direction = -preconditioned
        steps = 0
        while True:
            decrement = 0.5 * float(gradient.ravel() @ _block_solve(point, gradient, tilted_only=True).ravel())
            if not decrement > tolerance:
                return point.site_precision, point.site_shift, steps
            accepted = self._line_search(hyperparameters, point, direction, marginal_precision, marginal_shift, tolerance)
            if accepted is None:
                if np.array_equal(direction, -preconditioned):
                    return point.site_precision, point.site_shift, steps
                # A conjugate direction the line cannot lower Phi along: restart from the preconditioned gradient.
                direction = -preconditioned
                continue
            steps += 1
            new_gradient = accepted.gradient
            new_preconditioned = _block_solve(accepted, new_gradient)
            ratio = max(0.0, float(new_gradient.ravel() @ (new_preconditioned - preconditioned).ravel()) / float(gradient.ravel() @ preconditioned.ravel()))
            direction = -new_preconditioned + ratio * direction
            point, gradient, preconditioned = accepted, new_gradient, new_preconditioned

    def _double_loop(self, hyperparameters: Sequence[MixtureHyperparameters]) -> list[FixedPoint]:
        """EP's fixed point by the Opper-Winther double loop on the streamed design (``small_n.double_loop_sites``),
        with the noise stationary between its runs. Each outer step fixes (P_s, h_s) at q's marginals, which bounds the
        EC free energy's concave part linearly, and lowers the convex Phi (``_inner``): majorize-minimize, so every
        accepted inner step lowers the free energy. It ends where the undamped EP update's KL is within 1/(2K) nats
        and the noise update's gain within 1/(2K) (the refresh's own checks, at the certified marginal variances), or
        where an outer step leaves the sites unchanged, which is Phi stationary at q's own marginals: EP's fixed point."""
        gaussian = self.gaussian
        model_count = gaussian.model_count
        budget = 0.5 / self.draw_count
        # The inner problem's stop, on 1/2 g' Cov_r^-1 g, and the outer check, the undamped EP update's KL in q's full
        # LD metric, are different norms of one residual; where an outer step starts already inside the inner stop but
        # outside the check, the inner stop tightens by the measured ratio of the two (as the solves tighten by their
        # measured shortfall), until the inner problem moves the sites again.
        inner_tolerance = budget
        while True:
            variances, mean, group_variances, grams = self._refresh(hyperparameters)
            frozen = 1.0 / variances - self.site_precision
            cavities = [
                Cavity(precision=frozen[:, model], shift=mean[:, model] / variances[:, model] - self.site_shift[:, model]) for model in range(model_count)
            ]
            target_precision, target_shift = self._targets(hyperparameters, cavities)
            if not (np.all(np.isfinite(target_precision)) and np.all(np.isfinite(target_shift))):
                raise NoFixedPoint("a site target is not finite at the double loop's outer step")
            snapshot = self._snapshot()
            posteriors = [
                _member_posterior(
                    _posterior(gaussian, model, grams[model], group_variances[:, model], lambda snapshot=snapshot: self._ensure(snapshot)),
                    self.ties, self.site_precision[:, model], group_variances[:, model],
                )
                for model in range(model_count)
            ]
            precision_step = target_precision - self.site_precision
            right = (target_shift - self.site_shift) - precision_step * mean
            divergence = np.empty(model_count)
            for model in range(model_count):
                spread = max(-float(precision_step[:, model] @ posteriors[model].variance_jvp(precision_step[:, [model]])[:, 0]), 0.0)
                _lower, upper, _cross, _error = self._move_bounds(model, right[:, model])
                divergence[model] = 0.5 * upper + 0.25 * spread
            noise = self._noise(variances)
            self.noise_gain = np.array([
                noise_gain(float(noise[model]), float(self.noise[model]), int(gaussian.training_counts[model]), int(self.covariate_ranks[model]))
                for model in range(model_count)
            ])
            self.mean_move = 2.0 * divergence
            log(f"ep double loop: outer step at KL {np.array2string(divergence, precision=3)}, noise gain {np.array2string(self.noise_gain, precision=3)}")
            if np.all(divergence <= budget):
                if np.all(self.noise_gain <= budget):
                    self._ensure(snapshot)
                    return [
                        FixedPoint(
                            cavity=cavities[model], posterior=posteriors[model], mean=mean[:, model].copy(),
                            precision_norm=_precision_norm(gaussian, model, self.site_precision[:, model], self.ties),
                            effective_effects=float(self.effective[model]),
                        )
                        for model in range(model_count)
                    ]
                self.noise = noise
                continue
            start_precision, start_shift = self.site_precision.copy(), self.site_shift.copy()
            while True:
                precision, shift, steps = self._inner(hyperparameters, 1.0 / variances, mean / variances, inner_tolerance)
                log(f"ep double loop: inner problem {steps} steps at tolerance {inner_tolerance:.3e}")
                if steps:
                    break
                tightened = inner_tolerance * budget / float(np.max(divergence))
                if not tightened > _EPSILON * float(np.max(divergence)):
                    # The inner stop at float64's floor and still no step: Phi is stationary at q's own marginals to its
                    # rounding, yet the check above fails; the marginal variances' approximation sets that floor.
                    raise NoFixedPoint(
                        f"the double loop is stationary at KL {np.array2string(divergence, precision=3)}, above the 1/(2K) budget its check needs"
                    )
                inner_tolerance = tightened
            self.site_precision, self.site_shift = precision, shift

    def _frozen_passes(self, hyperparameters: Sequence[MixtureHyperparameters], frozen: F64Array, target_precision: F64Array, target_shift: F64Array) -> None:
        """Mean-only EP with the cavity precisions frozen, from the check's targets, until the frozen move is below
        p_eff / K (the module docstring's step 1c)."""
        gaussian = self.gaussian
        model_count = gaussian.model_count
        previous_move, damping = np.full(model_count, np.inf), 1.0
        while True:
            mean = self._member_moments(None)[0].copy()
            fraction = damping
            move = max(float(np.max(np.abs(target_precision - self.site_precision))), float(np.max(np.abs(target_shift - self.site_shift))))
            if not np.isfinite(move):
                # A non-finite target would never pass the halving test below (fraction nan is never small): refused.
                raise NoFixedPoint("a frozen pass's site target is not finite")
            scale = 1.0 + max(float(np.max(np.abs(self.site_precision))), float(np.max(np.abs(self.site_shift))))
            while True:
                if fraction * move <= _EPSILON * scale:
                    # The damped step no longer moves the sites past their rounding: no damped EP pass keeps the
                    # precision positive definite from here.
                    raise NoFixedPoint("no damped EP pass keeps the full-data precision positive definite")
                trial_precision = self.site_precision + fraction * (target_precision - self.site_precision)
                trial_shift = self.site_shift + fraction * (target_shift - self.site_shift)
                try:
                    # The pass's move is tested at fraction sqrt(1/K) in q's metric: each mean resolves that to relative
                    # accuracy sqrt(1/K), so the solves' error is not what the test measures.
                    self._iterate(trial_precision, trial_shift, np.full(model_count, fraction / self.draw_count))
                    break
                except np.linalg.LinAlgError:
                    fraction *= 0.5
            self.site_precision, self.site_shift = trial_precision, trial_shift
            marginal = 1.0 / (frozen + self.site_precision)
            # A damped pass moves fraction^2 of the full step's squared size: converge on the full step.
            new_mean = self._member_moments(None)[0]
            mean_move = np.sum(np.square(new_mean - mean) / marginal, axis=0) / (fraction * fraction)
            if not np.all(np.isfinite(mean_move)):
                raise NoFixedPoint("a frozen pass's mean move is not finite")
            if np.all(mean_move <= 1.0 / self.draw_count):
                return
            ratio = float(np.max(mean_move / previous_move))
            if ratio >= 1.0:
                # The ratio is the damped map's at this pass's fraction f: f / (1 + rho) removes the alternating mode it
                # measures (theory-ep, as for the refreshes), where 1 / (1 + rho) would hold f once it is below that.
                damping = min(damping, fraction / (1.0 + np.sqrt(ratio)))
            previous_move = mean_move
            cavities = [
                Cavity(precision=frozen[:, model], shift=new_mean[:, model] / marginal[:, model] - self.site_shift[:, model]) for model in range(model_count)
            ]
            target_precision, target_shift = self._targets(hyperparameters, cavities)


@dataclass
class _LoopPoint:
    """The double loop's inner problem at sites (tau, nu) under fixed marginals: q's members' means and marginal
    variances there, the inner cavities' tilted moments and cumulants, and Phi's gradient in (nu, tau)."""

    site_precision: F64Array
    site_shift: F64Array
    mean: F64Array
    variance: F64Array
    tilted_mean: F64Array
    tilted_variance: F64Array
    third: F64Array
    fourth: F64Array
    gradient: F64Array


def _site_blocks(point: _LoopPoint, tilted_only: bool = False) -> tuple[F64Array, F64Array, F64Array]:
    """The 2 x 2 blocks, per site, of Phi's Hessian H = Cov_q + Cov_r: Cov_r of (beta, -beta^2 / 2) under each tilted
    law alone (``tilted_only``), or plus q's own marginal part (``small_n._newton_step``'s preconditioner)."""
    mean, variance, third, fourth = point.tilted_mean, point.tilted_variance, point.third, point.fourth
    a_r = variance
    b_r = -0.5 * (third + 2.0 * mean * variance)
    c_r = 0.25 * (fourth + 2.0 * variance**2 + 4.0 * mean * third + 4.0 * mean**2 * variance)
    if tilted_only:
        return a_r, b_r, c_r
    return (
        a_r + point.variance,
        b_r - point.variance * point.mean,
        c_r + 0.5 * point.variance**2 + point.mean**2 * point.variance,
    )


def _block_solve(point: _LoopPoint, vector: F64Array, tilted_only: bool = False) -> F64Array:
    """The site blocks' inverse applied to a (nu, tau) vector (members x models each half)."""
    a, b, c = _site_blocks(point, tilted_only)
    half = vector.shape[0] // 2
    first, second = vector[:half], vector[half:]
    determinant = a * c - b * b
    return np.concatenate([(c * first - b * second) / determinant, (a * second - b * first) / determinant])


class _PassBudget(Exception):
    """A second-start solve used the first start's count of passes without converging: abandoned, not refused."""


def noise_floor(squares: F64Array, smallest_variance: F64Array, residual_dimension: float, entry_noise: float) -> float:
    """A lower bound on every noise variance a mean-field ascent reaches from ``entry_noise``, or 0 where none follows.

    The noise update is sigma^2 = (RSS + sum_j a_j v_j) / n' (a_j = ||x_j||^2, n' the residual dimension), and each
    member's variance under its q_j is at least its smallest conditional variance, v_j >= 1 / (1/V_j + a_j / s) with V_j
    its prior's smallest lattice variance and s the noise its sweep used. So the next noise is at least h(s) =
    (1/n') sum_j a_j V_j s / (s + a_j V_j): increasing in s, with h(s)/s decreasing from p_+ / n' (p_+ the members with
    a_j V_j > 0) to 0. Where p_+ > n', h has one positive fixed point s*, and s >= s* gives h(s) >= s*, s < s* gives
    h(s) > s: every noise the ascent reaches is at least min(entry_noise, s*). Where p_+ <= n' the bound is 0 (the
    likelihood alone keeps the noise from zero there, through the least-squares residual, which this bound does not
    form). s* is found by bisection on log s to float64 resolution."""
    weights = np.asarray(squares, dtype=np.float64) * np.asarray(smallest_variance, dtype=np.float64)
    weights = weights[weights > 0.0]
    dimension = float(residual_dimension)
    if weights.shape[0] <= dimension:
        return 0.0

    def excess(log_noise: float) -> float:
        return float(np.sum(weights / (np.exp(log_noise) + weights))) - dimension

    # sum_j w_j / (s + w_j) falls from p_+ > n' to 0, and each term is at most W / (s + W) (W the largest w): at
    # s = W (p_+ - n') / n' the sum is at most p_+ W / (s + W) = n', so the root lies at or below it.
    high = float(np.log(float(np.max(weights)) * (weights.shape[0] - dimension) / dimension))
    low = high
    while excess(low) <= 0.0:
        low -= 1.0
    while high - low > _EPSILON * max(abs(high), 1.0):
        middle = 0.5 * (low + high)
        if excess(middle) > 0.0:
            low = middle
        else:
            high = middle
    return min(float(entry_noise), float(np.exp(low)))


def elbo_ceiling(residual_dimension: float, floor: float) -> float:
    """An upper bound on the ELBO at every noise >= ``floor``: ELBO <= log p(y | sigma^2) <= -(n'/2) log(2 pi sigma^2),
    the Gaussian density of the projected residual at zero residual (the prior integrates to one); inf at floor 0."""
    return -0.5 * float(residual_dimension) * float(np.log(2.0 * np.pi * floor)) if floor > 0.0 else np.inf


@dataclass
class SweepBudget:
    """The work budget of a coordinate-ascent solve, derived from its problem: every iteration that does not end the
    loop raises the ELBO by more than min(rho_k, tolerance) (a sweep's gain above its rounding rho_k, else the noise
    update's gain above the tolerance, which the next ELBO includes), and every ELBO the loop reaches is at most the
    ceiling C (``noise_floor``, ``elbo_ceiling``). So after the first sweep's ELBO E_1 the loop can take at most
    (C - E_1) / min_k min(rho_k, tolerance) more iterations; one past that contradicts the bound, and the loop ends
    unresolved there rather than run on. Where no ceiling is derived (C = inf) there is no finite budget, and the count
    of iterations is what the solve reports."""

    ceiling: float
    tolerance: float
    first: float | None = None
    step: float = np.inf
    iterations: int = 0

    def exhausted(self, elbo: float, rounding: float) -> bool:
        """Counts one iteration at ELBO ``elbo`` with rounding ``rounding``; True once the count passes the budget."""
        self.iterations += 1
        if self.first is None:
            self.first = float(elbo)
            return False
        self.step = min(self.step, float(rounding), self.tolerance)
        if not np.isfinite(self.ceiling) or not self.step > 0.0:
            return False
        return self.iterations - 1 > (self.ceiling - self.first) / self.step


@dataclass(eq=False)
class _ModelState:
    """One model's q and its scalars: the snapshot the oracle keeps of a model (a candidate fixed point, the best state,
    a fixed point's restore point), and nothing of any other model. ``sites`` are a binary model's Polya-Gamma sites
    (immutable, held by reference; None for a quantitative model), which name the metric its residual and moments
    belong to. Its arrays are charged to the shared ledger for as long as it lives (``_FullDataMeanField._charged``)."""

    mean: F64Array
    variance: F64Array
    shift: F64Array
    third: F64Array
    fourth: F64Array
    residual: F64Array
    noise: float
    effective: float
    mean_move: float
    noise_gain: float
    elbo: float
    version: int
    sites: BernoulliSites | None = None

    @property
    def nbytes(self) -> int:
        return sum(int(getattr(self, name).nbytes) for name in _MODEL_STATE_ARRAYS)


_MODEL_STATE_ARRAYS = ("mean", "variance", "shift", "third", "fourth", "residual")
# The oracle's own live arrays per model beyond a model state's: over the members, the sites (precision and shift)
# and the metric's column squares (per member, per group, and the training mask's); over the samples, the metric's
# weights, root weights and targets, the training targets, the projected target and the cold start's residual.
_ORACLE_MEMBER_ARRAYS = ("site_precision", "site_shift", "member_squares", "group_squares", "training_squares")
_ORACLE_SAMPLE_ARRAYS = ("weights", "root", "targets", "training_targets", "projected_target", "cold_residual")


class _FullDataMeanField:
    """``scale_mixture_ep.FixedPoints`` on the full data by the mean-field route (``mean_field``): each model's product
    q = prod_j q_j by coordinate ascent on the ELBO, one pass over the streamed LD blocks per sweep, with the noise
    stationary between sweeps; the same fixed point, ELBO, certificate and draws as the dense route, on a design the
    store streams.

    Per block the tile's standardized columns are projected on the model's training rows and covariates,
    Xp_b = (I - H_m) X_b (``DualModels.complement``: H_m the projector of the training rows' covariates), and the
    dense sweep kernel runs over them with the residual r = y_P - Xp m carried across blocks; the pieces of a block
    are as wide as ``working_bytes`` allows two dense (n x width) arrays. ||xp_j||^2 are the dual solver's
    ``unit_squares``. The fixed point's linear responses (the total curvature's ``cavity_response``) are the dense
    route's formulas with R^-1 = (diag(tau) + Xp'Xp / sigma^2)^-1 taken from the dual solver at the sites
    tau_j = 1/v_j - omega_j and the shifts m_j / v_j - h_j, at which its Gaussian is q's mean and precision
    (``DualGaussian.iterate``, ``posterior_solve``), and Xp'Xp c by two tile passes. No leave-block-out variance,
    no cavity information certificate: q's variances are its own.

    Tie members (several members on one column, ``tie_members``) are coordinates of their own: each has its own q_j,
    class prior and offset, and its column is its group's, signed. The dual solver sees each group's sites
    (``group_sites``), and the members' responses follow from the groups' by conditioning on the sum
    (``_member_posterior``), as on the EP route.

    Memory (p members, n samples, M models; float64). Live for the oracle's life, charged to the shared ledger
    (``memory_broker``) when the oracle is built: the state, five p x M arrays (mean, variance, shift, third and fourth
    central moments) and the n x M residuals; the sites and the metric's member arrays (``_ORACLE_MEMBER_ARRAYS``); and
    the metric's sample arrays (``_ORACLE_SAMPLE_ARRAYS``: a binary model's weights and working response go through
    the ledger with them). The cold start is a recipe, not a copy: zero moments or the caller's start means (held by
    reference), the start's sites, and the residual they leave. A snapshot is one model's state (``_ModelState``: five
    p-vectors, its residual and its sites), never every model's arrays: the entry state of a call (M of them, dropped
    when the call returns), each start's solved state per model (at most two per model, dropped with the call), the
    best state per model (kept), and each returned fixed point's restore point, whose arrays are the ones its
    responses read (the outer loop keeps a model's current and trial fixed points). Every snapshot is charged to the
    ledger for as long as it lives. On a CUDA device a sweep charges its per-model device arrays, the predictor-moment
    read sizes its pieces from what the ledger can grant, and the panel Grams are the ledger's cache
    (``device_sweep.PanelGrams``)."""

    def __init__(
        self, gaussian: DualGaussian, statistics: GenotypeSufficientStatistics, prior: ScaleMixturePrior, draw_count: int, working_bytes: int, seed: int,
        starts: Sequence[MixtureHyperparameters], noise: F64Array, order_seed: int | None = None, start_mean: F64Array | None = None,
        sites: Sequence[BernoulliSites | None] | None = None,
    ) -> None:
        self.gaussian = gaussian
        self.prior = prior
        self.draw_count = int(draw_count)
        self.working_bytes = int(working_bytes)
        self.ties = TieGroups.from_tie_map(statistics.tie_map)
        if prior.variant_count != self.ties.member_count:
            raise ValueError("the prior must be over Stage 0's active rows (the tie members), in their order")
        self.grams = block_grams(statistics)
        # Each block's members in their order, and each member's group column within its block.
        self.member_blocks = tuple(np.flatnonzero(np.isin(self.ties.group, block)) for block in self.grams.blocks)
        if order_seed is not None:
            # A mixture component's sweep order (``fit_full_data``): each block's members in a random order, so coordinate
            # ascent can end in another mode between near-duplicate columns.
            order = np.random.default_rng([seed, order_seed])
            self.member_blocks = tuple(order.permutation(members) for members in self.member_blocks)
        self.sign = np.asarray(self.ties.sign, dtype=np.float64)
        model_count = gaussian.model_count
        self.model_count = model_count
        self.training = np.asarray(_host(gaussian.training), dtype=np.float64)
        self.sample_count = int(self.training.shape[0])
        self.training_counts = np.asarray(gaussian.training_counts, dtype=np.int64)
        self._broker = current_broker()
        if self._broker is not None:
            state_bytes = _FLOAT_BYTES * model_count * (
                (len(_MODEL_STATE_ARRAYS) - 1 + len(_ORACLE_MEMBER_ARRAYS)) * prior.variant_count
                + (1 + len(_ORACLE_SAMPLE_ARRAYS)) * self.sample_count
            )
            lease = self._broker.reserve(HOST, state_bytes, "the mean-field oracle's state")
            weakref.finalize(self, lease.release)
        # A binary model's Polya-Gamma sites (``binary_likelihood``): its metric is W = diag(omega) at unit noise, known,
        # rebuilt whenever its sites move; a quantitative model's is its training mask, fixed, over the noise the sweeps
        # update. Every metric-bound array (the projector, the column squares, the projected targets, the panel Grams)
        # is keyed by the sites.
        self.sites: list[BernoulliSites | None] = list(sites) if sites is not None else [None] * model_count
        if len(self.sites) != model_count:
            raise ValueError("sites needs one entry (None for a quantitative model) per model")
        self._training_targets = np.asarray(_host(gaussian.targets), dtype=np.float64).copy()
        self._training_squares = np.asarray(_host(gaussian.unit_squares), dtype=np.float64).copy()
        self._square_cache: dict[bytes, F64Array] = {}
        if gaussian.metric_key is not None:
            # The dual solver is in another oracle's binary metric: a quantitative model's squares are the mask's, which a
            # reweight never moves, and every binary model's come from its own sites below.
            binary_columns = [model for model, model_sites in enumerate(self.sites) if model_sites is not None]
            if len(binary_columns) != sum(1 for key in gaussian.metric_key if key is not None):
                raise ValueError("the dual solver's binary models must be this oracle's")
        self._install_metric()
        factor = np.asarray(_host(self.models.covariate_factor), dtype=np.float64)
        self.covariate_rank = np.array([int(np.count_nonzero(np.any(factor[model] != 0.0, axis=0))) for model in range(model_count)], dtype=np.int64)
        self.residual_dimension = self.training_counts - self.covariate_rank
        if np.any(self.residual_dimension <= 0):
            raise ValueError("a model's training rows do not outnumber its covariates' rank")
        self.class_index = np.asarray(prior.class_index, dtype=np.int64)
        self.noise = np.array(noise, dtype=np.float64, copy=True)
        self.noise[[model for model, model_sites in enumerate(self.sites) if model_sites is not None]] = 1.0
        count = prior.variant_count
        self.mean = np.zeros((count, model_count))
        self.variance = np.zeros((count, model_count))
        self.shift = np.zeros((count, model_count))
        self.third = np.zeros((count, model_count))
        self.fourth = np.zeros((count, model_count))
        self.residual = [values.copy() for values in self.projected_targets]
        self.site_precision = np.zeros((count, model_count))
        self.site_shift = np.zeros((count, model_count))
        self.effective = np.full(model_count, float(count))
        self.mean_move = np.full(model_count, np.inf)
        self.noise_gain = np.full(model_count, np.inf)
        self.mean_error = np.zeros(model_count)
        self.elbo = np.full(model_count, -np.inf)
        self.refusals: list[str] = []
        self.refreshes = 0
        self.reweights = 0
        self.passes = 0
        self.version = 0
        # Which state each model's column of the dual solver was last factored at (``_ensure``).
        self._solver_version = np.zeros(model_count, dtype=np.int64)
        self.undecided_blocks = 0
        self.information: list = []
        # Solves that ended at their derived work budget (``SweepBudget``), per model, with their reasons in refusals.
        self.unresolved = np.zeros(model_count, dtype=np.int64)
        device = self._device_pool()
        self._panel_grams = PanelGrams(self._broker, device) if device is not None else PanelGrams()
        # The cold start (a recipe: zero moments or the start means, the start's sites, and the residual they leave) and
        # the entry noise.
        self._cold_mean = None if start_mean is None else np.asarray(start_mean, dtype=np.float64)
        self._cold_noise = self.noise.copy()
        self._cold_sites = list(self.sites)
        if start_mean is not None:
            # A data start (``_mode_mixture``'s ridge component): q's means there and the residual they leave.
            self.mean[...] = self._cold_mean
            for model in range(model_count):
                self.residual[model] = self.projected_targets[model] - self._image(self.mean[:, model:model + 1], model)[:, 0]
        self._cold_residual = [values.copy() for values in self.residual]
        # Each model's highest-ELBO refreshed state and its hyperparameters (``mean_field.MeanFieldFixedPoints.best``):
        # the outer loop's objective is not the ELBO and can end below a state it visited.
        self.best_elbo = np.full(model_count, -np.inf)
        self.best_state: list[_ModelState | None] = [None] * model_count
        self.best_hyperparameters: list[MixtureHyperparameters | None] = [None] * model_count

    # the metric: each model's likelihood weights

    @property
    def metric_key(self) -> tuple | None:
        """The binary models' sites keys (None for a quantitative model), or None where every model is quantitative."""
        if all(model_sites is None for model_sites in self.sites):
            return None
        return tuple(None if model_sites is None else model_sites.key for model_sites in self.sites)

    def _group_squares(self, model: int, model_sites: BernoulliSites | None) -> F64Array:
        """||xt_g||^2 per group in the model's metric: the mask's from the dual solver's construction, or a binary model's
        at its sites by one read (``dual_solve.column_squares``), held per sites."""
        if model_sites is None:
            return self._training_squares[:, model]
        held = self._square_cache.get(model_sites.key)
        if held is None:
            xp = self.gaussian.array_module
            single = DualModels(xp.asarray(model_sites.weights[:, None]), xp.zeros((self.gaussian.source.variant_count, 1)), self.gaussian.covariates, xp)
            held = np.asarray(_host(column_squares(self.gaussian.source, single, self.gaussian.count)), dtype=np.float64)[:, 0]
            self._square_cache = {key: value for key, value in self._square_cache.items() if any(
                other is not None and other.key == key for other in self.sites
            )}
            self._square_cache[model_sites.key] = held
        return held

    def _metric_arrays(self, sites: Sequence[BernoulliSites | None]) -> tuple[F64Array, F64Array, F64Array]:
        """(weights (n, M), targets (n, M), group squares (groups, M)) of the metric the ``sites`` define."""
        weights = self.training.copy()
        targets = self._training_targets.copy()
        squares = np.empty((self.ties.group_count, self.model_count))
        for model, model_sites in enumerate(sites):
            if model_sites is not None:
                weights[:, model] = model_sites.weights
                targets[:, model] = model_sites.response
            squares[:, model] = self._group_squares(model, model_sites)
        return weights, targets, squares

    def _install_metric(self) -> None:
        """Every metric-bound array at the current sites: the projector W's (``DualModels``, its covariate factor per
        model), the root weights, the targets, the members' squares, the projected targets, and a binary model's terms of
        L beside its Gaussian value (the sites' constant and the covariates' evidence)."""
        xp = self.gaussian.array_module
        self.weights, self.targets, self.group_squares = self._metric_arrays(self.sites)
        self.root = np.sqrt(self.weights)
        # The projector alone (unit noise): Xp and y_P do not move with the noise, which the sweeps update.
        self.models = DualModels(xp.asarray(self.weights), xp.zeros((self.gaussian.source.variant_count, self.model_count)), self.gaussian.covariates, xp)
        self.member_squares = self.group_squares[self.ties.group]
        projected = np.asarray(_host(self.models.complement(xp.asarray(self.root * self.targets), xp.arange(self.model_count))), dtype=np.float64)
        self.projected_targets = [np.ascontiguousarray(projected[:, model]) for model in range(self.model_count)]
        covariates = np.asarray(_host(self.gaussian.covariates), dtype=np.float64)
        self._bernoulli_offset = np.zeros(self.model_count)
        self._bernoulli_size = np.zeros(self.model_count)
        for model, model_sites in enumerate(self.sites):
            if model_sites is not None:
                evidence = covariate_evidence(model_sites.weights, covariates)
                self._bernoulli_offset[model] = model_sites.constant() + evidence
                self._bernoulli_size[model] = model_sites.constant_size() + abs(evidence)

    def _sync(self) -> None:
        """The dual solver in this oracle's metric (it is shared with the mixture's other oracles)."""
        key = self.metric_key
        if self.gaussian.metric_key != key:
            xp = self.gaussian.array_module
            self.gaussian.reweight(
                sample_weights=xp.asarray(self.weights), targets=xp.asarray(self.targets), unit_squares=xp.asarray(self.group_squares), key=key,
            )
            self.version += 1

    def _reweight(self, model: int, sites: BernoulliSites) -> None:
        """A binary model's xi update with q held (``binary_likelihood``): the metric moves, the residual is q's in it, and
        the panel Grams of the old metric are released."""
        self.sites[model] = sites
        self._install_metric()
        self.residual[model] = self.projected_targets[model] - self._image(self.mean[:, model:model + 1], model)[:, 0]
        self._panel_grams.clear()
        self.reweights += 1

    def _moment_pieces(self, count: int) -> Iterator[tuple[int, int]]:
        """Column ranges of a block for the predictor-variance read: its three dense (n x width) float64 arrays (the
        columns, their weighted copy, their projection) fit the host working set, and on a device what the shared ledger
        can grant there (``memory_broker``: the capacity less every live allocation, with the idle caches, the resident
        codes and the panel Grams, counted as reclaimable, since the ledger's allocator evicts them before it refuses);
        the device also holds what the host budget does not see (bench-sim scenario_014 [sim]: a 4.3 GB piece against
        43 GB already held on a 48 GB A40). Outside a ledger's scope the device's free bytes and its pool's cached
        blocks, read at each call, stand in for the ledger."""
        column_bytes = 3 * self.sample_count * np.dtype(np.float64).itemsize
        width = max(1, self.working_bytes // column_bytes)
        xp = self.gaussian.array_module
        if xp is not np:
            pool = self._device_pool()
            if pool is not None:
                available = self._broker.reclaimable(pool)
            else:
                free, _total = xp.cuda.runtime.memGetInfo()
                available = int(free) + int(xp.get_default_memory_pool().free_bytes())
            width = max(1, min(width, available // column_bytes))
        for start in range(0, count, width):
            yield start, min(start + width, count)

    def _predictor_moments(self, model: int) -> tuple[F64Array, F64Array]:
        """A binary model's E eta_i and Var eta_i under q on its training rows (0 elsewhere; ``binary_likelihood``):
        z_i - r_i / sqrt(omega_i) and (sum_g Xp_ig^2 V_g + H_ii) / omega_i, V_g the group's column's effect variance (its
        members' sum: each member is its own independent factor) and H_ii = omega_i ||F'c_i||^2 the weighted covariate
        leverage (F F' = (C'WC)^+). One read of the store, the projected columns formed piece by piece."""
        rows = self.training[:, model] > 0.0
        root = self.root[:, model]
        safe = np.where(rows, root, 1.0)
        mean = np.where(rows, self.targets[:, model] - self.residual[model] / safe, 0.0)
        group_variance = np.bincount(self.ties.group, weights=self.variance[:, model], minlength=self.ties.group_count)
        xp = self.gaussian.array_module
        models = self.models
        root = xp.asarray(self.root[:, model])
        explained = xp.zeros(self.sample_count)
        for start, stop, tile in self.gaussian.source.blocks():
            for first, last in self._moment_pieces(stop - start):
                local = xp.arange(first, last, dtype=xp.int64)
                masked = xp.asarray(tile.columns(local), dtype=xp.float64) * root[:, None]
                projected = models.complement(masked, xp.full(last - first, model, dtype=xp.int64))
                del masked
                explained += xp.square(projected) @ xp.asarray(group_variance[start + first:start + last])
                del projected
        explained = np.asarray(_host(explained), dtype=np.float64)
        self.passes += 1
        covariates = np.asarray(_host(self.gaussian.covariates), dtype=np.float64)
        factor = np.asarray(_host(self.models.covariate_factor[model]), dtype=np.float64)
        whitened = covariates @ factor
        leverage = self.weights[:, model] * np.sum(whitened * whitened, axis=1)
        variance = np.where(rows, (explained + leverage) / (safe * safe), 0.0)
        return mean, variance

    # state

    def _device_pool(self) -> str | None:
        """The ledger's pool of the device the fit runs on, None on the host or outside a ledger's scope."""
        if self._broker is None or self.gaussian.array_module is np:
            return None
        pool = device_pool(int(self.gaussian.array_module.cuda.runtime.getDevice()))
        return pool if pool in self._broker.meters else None

    def _charged(self, state: _ModelState) -> _ModelState:
        """``state``'s arrays charged to the ledger for as long as the state lives."""
        if self._broker is not None:
            lease = self._broker.reserve(HOST, state.nbytes, "a mean-field model snapshot")
            weakref.finalize(state, lease.release)
        return state

    def _state(self, model: int) -> _ModelState:
        """Model ``model``'s state: copies of its columns and residual, its scalars, and its sites (by reference)."""
        return self._charged(_ModelState(
            mean=self.mean[:, model].copy(), variance=self.variance[:, model].copy(), shift=self.shift[:, model].copy(),
            third=self.third[:, model].copy(), fourth=self.fourth[:, model].copy(), residual=self.residual[model].copy(),
            noise=float(self.noise[model]), effective=float(self.effective[model]), mean_move=float(self.mean_move[model]),
            noise_gain=float(self.noise_gain[model]), elbo=float(self.elbo[model]), version=self.version, sites=self.sites[model],
        ))

    def _move_sites(self, model: int, sites: BernoulliSites | None) -> None:
        """Model ``model``'s metric at ``sites``: every metric-bound array rebuilt and the old metric's panels released."""
        if sites is not self.sites[model]:
            self.sites[model] = sites
            self._install_metric()
            self._panel_grams.clear()

    def _cold_state(self, model: int) -> None:
        """Model ``model`` back at the cold start (the recipe: zero moments or the start means, the start's sites, and
        their residual)."""
        self._move_sites(model, self._cold_sites[model])
        for name in ("variance", "shift", "third", "fourth"):
            getattr(self, name)[:, model] = 0.0
        self.mean[:, model] = 0.0 if self._cold_mean is None else self._cold_mean[:, model]
        self.residual[model] = self._cold_residual[model].copy()
        self.noise[model] = self._cold_noise[model]
        self.effective[model] = float(self.prior.variant_count)
        self.mean_move[model], self.noise_gain[model], self.elbo[model] = np.inf, np.inf, -np.inf
        self.site_precision[:, model] = 0.0
        self.site_shift[:, model] = 0.0

    def _restore_state(self, state: _ModelState, model: int) -> None:
        """Model ``model`` back at ``state``: its metric, columns and residual, its scalars, and its sites from them."""
        self._move_sites(model, state.sites)
        for name in _MODEL_STATE_ARRAYS[:-1]:
            getattr(self, name)[:, model] = getattr(state, name)
        self.residual[model] = state.residual.copy()
        self.noise[model], self.effective[model] = state.noise, state.effective
        self.mean_move[model], self.noise_gain[model], self.elbo[model] = state.mean_move, state.noise_gain, state.elbo
        _omega, tau, nu, _live = self._sites(model)
        self.site_precision[:, model], self.site_shift[:, model] = tau, nu

    def _ensure(self, state: _ModelState, model: int) -> None:
        """The dual solver's column of ``model`` back at this fixed point's sites and metric before it answers for it (a
        later trial may have moved either); the other models' columns are factored at the oracle's current sites and
        metrics, and no model's solves read another's. The oracle's own state stays where it was."""
        if self._solver_version[model] == state.version and state.sites is self.sites[model] and self.gaussian.metric_key == self.metric_key:
            return
        current = self.sites[model]
        moved = state.sites is not current
        if moved:
            self.sites[model] = state.sites
            self._install_metric()
        precision, shift, noise = self.site_precision.copy(), self.site_shift.copy(), self.noise.copy()
        omega = self.member_squares[:, model] / state.noise
        live = state.variance > 0.0
        with np.errstate(divide="ignore"):
            precision[:, model] = np.where(live, 1.0 / np.where(live, state.variance, 1.0), np.inf) - omega
            shift[:, model] = np.where(live, state.mean / np.where(live, state.variance, 1.0) - state.shift, 0.0)
        noise[model] = state.noise
        self._iterate(precision, shift, noise)
        self._solver_version[model] = state.version
        if moved:
            self.sites[model] = current
            self._install_metric()

    # the design, streamed

    def _member_pieces(self, width: int | None = None) -> Iterator[tuple[object, I64Array, I64Array]]:
        """(tile, members, their columns in the tile) per piece, the blocks and their members in order; ``width`` caps
        a piece's members (the device sweep's, from its free memory), else the host working set sets it."""
        for (start, stop, tile), members in zip(self.gaussian.source.blocks(), self.member_blocks):
            pieces = self._pieces(members.shape[0]) if width is None else ((first, min(first + width, members.shape[0])) for first in range(0, members.shape[0], width))
            for piece_start, piece_stop in pieces:
                piece = members[piece_start:piece_stop]
                yield tile, piece, self.ties.group[piece] - start

    def _group_values(self, values: F64Array) -> F64Array:
        """sum_j s_j v_j per group, for member values (p_members x r)."""
        grouped = np.zeros((self.ties.group_count,) + values.shape[1:])
        np.add.at(grouped, self.ties.group, self.sign.reshape((-1,) + (1,) * (values.ndim - 1)) * values)
        return grouped

    def _pieces(self, count: int) -> Iterator[tuple[int, int]]:
        """Column ranges of a block whose two dense (n x width) float64 arrays (the tile's columns and their
        projection) fit the working set."""
        width = max(1, self.working_bytes // (2 * self.sample_count * np.dtype(np.float64).itemsize))
        for start in range(0, count, width):
            yield start, min(start + width, count)

    def _metric(self, metric: tuple | None) -> tuple[DualModels, F64Array]:
        """(the projector, the root weights (n, M)): the current metric's, or a fixed point's own held one."""
        return (self.models, self.root) if metric is None else metric

    def _projected(self, tile, local: I64Array, model: int, metric: tuple | None = None) -> F64Array:
        """Xt_b = (I - H) W^1/2 X_b on the tile's ``local`` columns (n x |local|, F-ordered)."""
        models, root = self._metric(metric)
        xp = self.gaussian.array_module
        dense = xp.asarray(tile.columns(xp.asarray(local)), dtype=xp.float64)
        masked = dense * xp.asarray(root[:, model])[:, None]
        projected = models.complement(masked, xp.full(local.shape[0], model, dtype=xp.int64))
        return np.asfortranarray(np.asarray(_host(projected), dtype=np.float64))

    def _image(self, coefficients: F64Array, model: int, metric: tuple | None = None) -> F64Array:
        """Xp c (n x r) for c (p x r): the tiles' X_b c_b, weighted (the training mask, or W^1/2) and projected."""
        models, root = self._metric(metric)
        xp = self.gaussian.array_module
        values = self._group_values(np.asarray(coefficients, dtype=np.float64))
        image = np.zeros((self.sample_count, values.shape[1]))
        for start, stop, tile in self.gaussian.source.blocks():
            image += np.asarray(_host(tile.matmat(xp.asarray(values[start:stop]))), dtype=np.float64)
        masked = xp.asarray(image * root[:, model][:, None])
        return np.asarray(_host(models.complement(masked, xp.full(values.shape[1], model, dtype=xp.int64))), dtype=np.float64)

    def _back(self, samples: F64Array, model: int, metric: tuple | None = None) -> F64Array:
        """Xp' u (p x r) for u (n x r): (I - H) and the weights (the training mask, or W^1/2), then the tiles' X_b' u."""
        models, root = self._metric(metric)
        xp = self.gaussian.array_module
        values = np.asarray(samples, dtype=np.float64)
        projected = np.asarray(_host(models.complement(xp.asarray(values), xp.full(values.shape[1], model, dtype=xp.int64))), dtype=np.float64)
        masked = xp.asarray(projected * root[:, model][:, None])
        back = np.zeros((self.ties.group_count, values.shape[1]))
        for start, stop, tile in self.gaussian.source.blocks():
            back[start:stop] = np.asarray(_host(tile.rmatmat(masked)), dtype=np.float64)
        return self.sign[:, None] * back[self.ties.group]

    # the sweeps

    def _sweep(self, model: int, hyperparameters: MixtureHyperparameters) -> tuple[float, float, float, float]:
        if self.gaussian.array_module is not np:
            return self._device_sweep(model, hyperparameters)
        # ``mean_field`` imports ``small_n``, which imports this module's certificate: the kernel is bound at first use.
        from sv_pgs.mean_field import _sweep as mean_field_sweep

        prior = self.prior
        log_density = np.ascontiguousarray(class_log_density(prior, hyperparameters.coefficients))
        scales = log_scale(prior, hyperparameters.coefficients)
        divergence = weighted_variance = sizes = 0.0
        residual = self.residual[model]
        noise = float(self.noise[model])
        for tile, rows, local in self._member_pieces():
                projected = np.asfortranarray(self._projected(tile, local, model) * self.sign[rows][None, :])
                log_node_variance = scales[rows][:, None] + prior.log_variance_grid[None, :]
                with np.errstate(over="ignore"):
                    node_variance = np.exp(log_node_variance)
                mean, variance, shift, third, fourth = (np.ascontiguousarray(values[rows, model]) for values in (self.mean, self.variance, self.shift, self.third, self.fourth))
                part = mean_field_sweep(
                    projected, np.ascontiguousarray(self.member_squares[rows, model]), np.arange(rows.shape[0], dtype=np.int64), self.class_index[rows],
                    log_density, node_variance, log_node_variance, noise, mean, residual, variance, shift, third, fourth,
                    np.arange(rows.shape[0], dtype=np.int64),
                )
                for values, piece in ((self.mean, mean), (self.variance, variance), (self.shift, shift), (self.third, third), (self.fourth, fourth)):
                    values[rows, model] = piece
                divergence += part[0]
                weighted_variance += part[1]
                sizes += part[3]
        self.passes += 1
        return divergence, weighted_variance, float(residual @ residual), sizes

    def _device_sweep(self, model: int, hyperparameters: MixtureHyperparameters) -> tuple[float, float, float, float]:
        """The same sweep on the device (``device_sweep``): the members in the same order, the residual and the panel
        Grams held there, and only each piece's moments copied back."""
        cupy = self.gaussian.array_module
        prior = self.prior
        log_density = cupy.asarray(np.ascontiguousarray(class_log_density(prior, hyperparameters.coefficients)))
        scales = cupy.asarray(log_scale(prior, hyperparameters.coefficients))
        grid = cupy.asarray(prior.log_variance_grid)
        mask = cupy.asarray(self.root[:, model])
        residual = cupy.asarray(self.residual[model])
        squares = cupy.asarray(np.ascontiguousarray(self.member_squares[:, model]))
        classes = cupy.asarray(self.class_index)
        state = {name: cupy.asarray(np.ascontiguousarray(getattr(self, name)[:, model])) for name in ("mean", "variance", "shift", "third", "fourth")}
        pieces = cupy.zeros((prior.variant_count, PIECE_COLUMNS))
        noise = float(self.noise[model])

        masked_covariates = self.gaussian.covariates * mask[:, None]
        covariate_pinv = cupy.linalg.pinv(masked_covariates.T @ masked_covariates)
        signs = cupy.asarray(self.sign)
        for tile, members, local in self._member_pieces(None):
            rows = cupy.asarray(members)
            local_device, member_signs = cupy.asarray(local), signs[rows]

            def decode(first, last, tile=tile, local_device=local_device, member_signs=member_signs):
                return cupy.asarray(tile.columns(local_device[first:last]), dtype=cupy.float64) * member_signs[first:last][None, :]
            log_node_variance = cupy.ascontiguousarray(scales[rows][:, None] + grid[None, :])
            node_variance = cupy.exp(log_node_variance)
            piece_state = {name: cupy.ascontiguousarray(values[rows]) for name, values in state.items()}
            piece_parts = cupy.zeros((members.shape[0], PIECE_COLUMNS))
            sweep_piece(
                cupy, decode=decode, width=int(members.shape[0]), mask=mask, covariates=masked_covariates, covariate_pinv=covariate_pinv,
                residual=residual, grams=self._panel_grams,
                model=model, members=np.asarray(members), metric=b"" if self.sites[model] is None else self.sites[model].key,
                squares=cupy.ascontiguousarray(squares[rows]), class_index=cupy.ascontiguousarray(classes[rows]),
                log_density=log_density, node_variance=node_variance, log_node_variance=log_node_variance, noise=noise,
                pieces=piece_parts, **piece_state,
                panels=CodePanels(cupy, tile, local_device, member_signs) if CodePanels.supports(tile) else None,
            )
            for name, values in piece_state.items():
                state[name][rows] = values
            pieces[rows] = piece_parts
        for name, values in state.items():
            getattr(self, name)[:, model] = cupy.asnumpy(values)
        self.residual[model] = cupy.asnumpy(residual)
        totals = cupy.asnumpy(pieces.sum(axis=0))
        self.passes += 1
        residual_square = float(cupy.asnumpy(residual @ residual))
        return float(totals[0]), float(totals[1]), residual_square, float(totals[2])

    def _elbo(self, model: int, divergence: float, weighted_variance: float, residual_square: float, sizes: float) -> tuple[float, float]:
        """As ``MeanFieldFixedPoints._elbo``, on this model's training rows: a binary model's Bernoulli bound L at unit
        noise, its sites' constant and covariate evidence added."""
        if self.sites[model] is not None:
            fit_term = 0.5 * (residual_square + weighted_variance)
            value = -fit_term - divergence + float(self._bernoulli_offset[model])
            summands = 2 * self.prior.variant_count + 2 * int(self.training_counts[model]) + 1
            return value, (self.prior.grid_size + 1 + summands) * _EPSILON * (fit_term + sizes + float(self._bernoulli_size[model]))
        noise = float(self.noise[model])
        residual_term = 0.5 * float(self.residual_dimension[model]) * float(np.log(2.0 * np.pi * noise))
        fit_term = (residual_square + weighted_variance) / (2.0 * noise)
        value = -residual_term - fit_term - divergence
        summands = 2 * self.prior.variant_count + int(self.training_counts[model])
        return value, (self.prior.grid_size + 1 + summands) * _EPSILON * (abs(residual_term) + fit_term + sizes)

    def _sweep_budget(self, model: int, hyperparameters: MixtureHyperparameters) -> SweepBudget:
        """This solve's work budget (``SweepBudget``): the ELBO ceiling at the noise floor from the prior's smallest
        lattice variance per member at these hyperparameters; a binary model's ELBO bounds the Bernoulli log likelihood
        of its labels, a log probability, so its ceiling is 0."""
        if self.sites[model] is not None:
            return SweepBudget(ceiling=0.0, tolerance=0.5 / self.draw_count)
        smallest = np.exp(log_scale(self.prior, hyperparameters.coefficients) + float(np.min(self.prior.log_variance_grid)))
        floor = noise_floor(self.member_squares[:, model], smallest, float(self.residual_dimension[model]), float(self.noise[model]))
        return SweepBudget(ceiling=elbo_ceiling(float(self.residual_dimension[model]), floor), tolerance=0.5 / self.draw_count)

    def _solve_model(self, model: int, hyperparameters: MixtureHyperparameters, pass_budget: int | None = None) -> None:
        """Sweeps at the current noise, the noise moving to its stationary value between them, until the sweeps'
        measured remainder (the geometric extrapolation of the last two gains, ``mean_field``) plus the noise's
        pending gain is within the tolerance, or the solve's derived work budget (``SweepBudget``) is spent, which
        raises ``FloatingPointError`` (the trial is refused, counted in ``unresolved``, and never reported solved)."""
        tolerance = 0.5 / self.draw_count
        elbo: float | None = None
        gain: float | None = None
        previous_gain: float | None = None
        pending_noise: float | None = None
        passes_at_entry = self.passes
        budget = self._sweep_budget(model, hyperparameters)
        while True:
            if pass_budget is not None and self.passes - passes_at_entry >= pass_budget:
                raise _PassBudget()
            if pending_noise is not None:
                elbo = elbo + float(self.noise_gain[model]) if elbo is not None else None
                self.noise[model] = pending_noise
            divergence, weighted_variance, residual_square, sizes = self._sweep(model, hyperparameters)
            if not (np.isfinite(divergence) and np.isfinite(weighted_variance) and np.isfinite(residual_square)):
                raise FloatingPointError(f"model {model}: a mean-field sweep is not finite")
            value, rounding = self._elbo(model, divergence, weighted_variance, residual_square, sizes)
            if self.sites[model] is None:
                pending_noise = (residual_square + weighted_variance) / float(self.residual_dimension[model])
                self.noise_gain[model] = noise_gain(pending_noise, float(self.noise[model]), int(self.training_counts[model]), int(self.covariate_rank[model]))
            else:
                # A binary model's noise is known (1); its sites' update is taken at the fixed point below.
                self.noise_gain[model] = 0.0
            gain = (value - elbo) if elbo is not None else None
            elbo = value
            if gain is not None and gain < -rounding:
                raise FloatingPointError(f"model {model}: a mean-field sweep lowered the ELBO by {-gain:.3g} nats: the bound's ascent is broken")
            self.elbo[model] = value
            if gain is None:
                remaining = np.inf
            elif gain <= rounding:
                remaining = 0.0
            elif previous_gain is not None and previous_gain > 0.0:
                rate = gain / previous_gain
                remaining = gain * rate / (1.0 - rate) if rate < 1.0 else np.inf
            else:
                remaining = np.inf
            if gain is not None:
                previous_gain = max(float(gain), 0.0)
            self.mean_move[model] = 2.0 * remaining
            if remaining + float(self.noise_gain[model]) <= tolerance:
                model_sites = self.sites[model]
                if model_sites is None:
                    return
                # A binary model at its fixed point for these sites (``mean_field.MeanFieldFixedPoints._solve``): the xi
                # update's exact gain at q, one read for the predictor variances. Within what is left of the tolerance
                # this is the joint fixed point of (q, xi), its gain reported as the likelihood's pending one; otherwise
                # xi moves with q held, L rises by the gain, and the sweeps continue in the new metric from there.
                updated, site_gain = model_sites.updated(*self._predictor_moments(model))
                if remaining + site_gain <= tolerance:
                    self.noise_gain[model] = site_gain
                    return
                self._reweight(model, updated)
                elbo = value + site_gain
                gain = previous_gain = None
            if budget.exhausted(value, rounding):
                self.unresolved[model] += 1
                raise FloatingPointError(
                    f"model {model}: the mean-field solve spent its derived work budget ({budget.iterations} iterations against the ELBO "
                    f"ceiling {budget.ceiling:.6g} from {budget.first:.6g} in steps of at least {budget.step:.3g}) with {remaining:.3g} nats "
                    "extrapolated to go: unresolved"
                )

    def _iterate(self, site_precision: F64Array, site_shift: F64Array, noise: F64Array) -> None:
        """The dual solver at q's precision and mean: sites tau = 1/v - omega and nu = m/v - h per member (identity
        ties), whose Gaussian has precision diag(tau) + Xp'Xp / sigma^2 = R and mean m."""
        # A linear response's sites, not a Gaussian's: the members' elimination needs no sign (tied_weights).
        group_precision, group_shift = group_sites(self.ties, site_precision, site_shift, algebraic=True)
        self._sync()
        self.gaussian.iterate(
            site_precision=group_precision, site_shift=group_shift, noise_variance=noise,
            error_bound=np.full(self.model_count, np.sqrt(1.0 / self.draw_count)), probe_residual_ratio=_HALF_PRECISION,
            # q's sites: a scale-mixture member's variance can exceed 1 / omega (its tau is then negative), so R is a
            # linear response's precision, symmetric and nonsingular, and not always a Gaussian's (``mean_field``'s
            # dense route solves the same system by a symmetric indefinite factorization).
            indefinite_core=True,
        )
        self.version += 1
        self._solver_version[:] = self.version

    def _sites(self, model: int) -> tuple[F64Array, F64Array, F64Array, np.ndarray]:
        omega = self.member_squares[:, model] / float(self.noise[model])
        variance = self.variance[:, model]
        live = variance > 0.0
        with np.errstate(divide="ignore"):
            tau = np.where(live, 1.0 / np.where(live, variance, 1.0), np.inf) - omega
            nu = np.where(live, self.mean[:, model] / np.where(live, variance, 1.0) - self.shift[:, model], 0.0)
        return omega, tau, nu, live

    def __call__(self, hyperparameters: Sequence[MixtureHyperparameters]) -> list[FixedPoint | None]:
        """Each model's fixed point of the higher ELBO between the solve from the carried state and the solve from the
        start (``MeanFieldFixedPoints.__call__``: coordinate ascent has several fixed points at one x)."""
        entry = [self._state(model) for model in range(self.model_count)]
        first = self.refreshes == 0
        candidates: list[list[_ModelState]] = [[] for _ in range(self.model_count)]
        # The cold solve gets the carried solve's own count of passes per model, and no more (``mean_field``).
        budgets: list[int | None] = [None] * self.model_count
        for cold in ((False,) if first else (False, True)):
            for model in range(self.model_count):
                if cold:
                    self._cold_state(model)
                before = self.passes
                try:
                    self._solve_model(model, hyperparameters[model], pass_budget=budgets[model])
                except _PassBudget:
                    continue
                except (FloatingPointError, ZeroDivisionError) as error:
                    self.refusals.append(f"{type(error).__name__}: {error}")
                    continue
                if budgets[model] is None:
                    budgets[model] = self.passes - before
                candidates[model].append(self._state(model))
        if any(not solved for solved in candidates):
            # No start reaches some model's fixed point: the trial is refused whole, as EP's oracle refuses.
            for model, state in enumerate(entry):
                self._restore_state(state, model)
            return [None] * self.model_count
        for model, solved in enumerate(candidates):
            self._restore_state(max(solved, key=lambda state: state.elbo), model)
        del candidates
        for model in range(self.model_count):
            omega, _tau, _nu, _live = self._sites(model)
            self.effective[model] = max(float(np.sum(omega * self.variance[:, model])), _EPSILON * self.prior.variant_count)
        try:
            self._iterate(self.site_precision, self.site_shift, self.noise)
        except np.linalg.LinAlgError as error:
            self.refusals.append(f"the dual solver has no factor at q's sites: {error}")
            for model, state in enumerate(entry):
                self._restore_state(state, model)
            return [None] * self.model_count
        del entry
        self.refreshes += 1
        for model in range(self.model_count):
            if self.elbo[model] > self.best_elbo[model]:
                self.best_elbo[model], self.best_state[model], self.best_hyperparameters[model] = self.elbo[model], self._state(model), hyperparameters[model]
        return [self._fixed_point(model, hyperparameters[model]) for model in range(self.model_count)]

    def restore_best(self, hyperparameters: Sequence[MixtureHyperparameters], tolerance: float) -> tuple[tuple[MixtureHyperparameters, ...], BoolArray]:
        """Each model back at its highest-ELBO refreshed state where that is above where the outer loop ended by more than
        ``tolerance``, the dual solver refactored at the restored sites; returns each model's hyperparameters and which
        models were restored (their outer-loop diagnostics describe another state: ``fit_full_data``)."""
        chosen = list(hyperparameters)
        restored = np.array([
            self.best_state[model] is not None and self.best_elbo[model] > self.elbo[model] + tolerance for model in range(self.model_count)
        ], dtype=bool)
        for model in np.flatnonzero(restored):
            self._restore_state(self.best_state[model], int(model))  # type: ignore[arg-type]
            chosen[model] = self.best_hyperparameters[model]  # type: ignore[assignment]
        if np.any(restored):
            self._iterate(self.site_precision, self.site_shift, self.noise)
        return tuple(chosen), restored

    def _fixed_point(self, model: int, hyperparameters: MixtureHyperparameters) -> FixedPoint:
        omega, tau, _nu, live = self._sites(model)
        noise = float(self.noise[model])
        squares = self.member_squares[:, model].copy()
        # The fixed point's restore point is its model's state, and its responses read those same arrays.
        state = self._state(model)
        mean, variance, shift, third, fourth, residual = (getattr(state, name) for name in _MODEL_STATE_ARRAYS)
        mean_by_omega = np.where(live, -0.5 * (third + 2.0 * mean * variance), 0.0)
        variance_by_shift = np.where(live, third, 0.0)
        variance_by_omega = np.where(live, -0.5 * (fourth - variance * variance) - mean * third, 0.0)
        residual_dimension = float(self.residual_dimension[model])
        # The metric this fixed point was solved in, held: a later trial may move a binary model's sites.
        metric = (self.models, self.root)
        binary = self.sites[model] is not None
        # The Stage 0 Grams feed only the posterior's variance map, which the mean-field fixed point never reads (its
        # responses are the dual solver's solves, in this fixed point's own metric); a binary model's are unweighted.
        grams = replace(self.grams, scale=1.0 / noise)
        group_variance = np.bincount(self.ties.group, weights=variance, minlength=self.ties.group_count)
        dual = _member_posterior(
            _posterior(self.gaussian, model, grams, group_variance, lambda: self._ensure(state, model)), self.ties, tau, group_variance, algebraic=True,
        )
        noise_solve: dict[str, object] = {}

        def off_diagonal_gram(columns: F64Array) -> F64Array:
            return self._back(self._image(columns, model, metric), model, metric) - squares[:, None] * columns

        def solve(right: F64Array, relative_tolerance: float) -> F64Array:
            # R^-1 right: the dual solver's A = Xp'Xp / sigma^2 + diag(tau) at these sites is R itself.
            assert dual.solve is not None
            return dual.solve(right, relative_tolerance)

        def noise_terms(relative_tolerance: float) -> tuple[F64Array, F64Array, float]:
            if not noise_solve:
                coupling = np.where(live, shift + mean_by_omega * omega / np.where(live, variance, 1.0), 0.0)
                mean_one = solve(coupling[:, None], relative_tolerance)
                shift_one = off_diagonal_gram(mean_one) / noise - shift[:, None]
                scalar = residual_dimension - (
                    2.0 * float(residual @ self._image(mean_one, model, metric)[:, 0])
                    + float(squares @ (variance_by_shift * shift_one[:, 0] - variance_by_omega * omega))
                ) / noise
                noise_solve.update(mean_one=mean_one, shift_one=shift_one, scalar=scalar)
            return noise_solve["mean_one"], noise_solve["shift_one"], noise_solve["scalar"]  # type: ignore[return-value]

        def cavity_response(mean_by_z: F64Array, variance_by_z: F64Array, relative_tolerance: float) -> tuple[F64Array, F64Array]:
            # The dense route's ``cavity_response`` (``mean_field``), with R^-1 by the dual solver to the relative
            # tolerance asked and Xp'Xp by tile passes.
            scaled = np.where(live[:, None], mean_by_z / np.where(live, variance, 1.0)[:, None], 0.0)
            mean_step = solve(scaled, relative_tolerance)
            shift_step = -off_diagonal_gram(mean_step) / noise
            if binary:
                # Known noise: no noise response. The sites' own response is left out of the curvature, as in
                # ``mean_field`` (the outer loop accepts its steps on the objective itself).
                return shift_step, np.zeros_like(shift_step)
            _mean_one, shift_one, scalar = noise_terms(relative_tolerance)
            right = -2.0 * (residual @ self._image(mean_step, model, metric)) + squares @ (
                np.where(live[:, None], variance_by_z, 0.0) + variance_by_shift[:, None] * shift_step
            )
            noise_step = right / scalar
            relative = (noise_step / noise)[None, :]
            return shift_step + shift_one * relative, -omega[:, None] * relative

        def norm(direction: F64Array) -> float:
            values = np.asarray(direction, dtype=np.float64)
            if np.any((values != 0.0) & ~live):
                return np.inf
            return float(np.sum(np.square(values[live]) / variance[live]))

        posterior = GaussianPosterior(cavity_response=cavity_response, exact=False)
        cavity = Cavity(precision=omega, shift=shift)
        offset = float(self.elbo[model]) - _data_value(self.prior, hyperparameters.coefficients, cavity, self.working_bytes)
        return FixedPoint(
            cavity=cavity, posterior=posterior, mean=mean, precision_norm=norm, effective_effects=float(self.effective[model]),
            restore=lambda: self._restore_state(state, model), evidence_offset=offset,
        )

    def binary_component(self, model: int) -> BinaryComponent:
        """A binary model's ``BinaryComponent`` at this oracle's state and metric: three reads of the store (the
        covariates' coefficients, their cross-products C'W X, the predictor variances)."""
        model_sites = self.sites[model]
        assert model_sites is not None
        xp = self.gaussian.array_module
        covariates = np.asarray(_host(self.gaussian.covariates), dtype=np.float64)
        weighted = xp.asarray(self.weights[:, model:model + 1] * covariates)
        cross = np.zeros((self.ties.group_count, covariates.shape[1]))
        for start, stop, tile in self.gaussian.source.blocks():
            cross[start:stop] = np.asarray(_host(tile.rmatmat(weighted)), dtype=np.float64)
        factor = np.asarray(_host(self.models.covariate_factor[model]), dtype=np.float64)
        covariance = factor @ factor.T
        mean, variance = self._predictor_moments(model)
        rows = self.training[:, model] > 0.0
        alpha = self.covariate_coefficients(model)
        loading = covariance @ cross.T
        return BinaryComponent(
            alpha=alpha, loading=loading, covariance=covariance, anchor=alpha + loading @ self._group_values(self.mean[:, model]),
            predictor_mean=mean[rows], predictor_variance=variance[rows], labels=model_sites.labels[rows],
        )

    def covariate_coefficients(self, model: int, mean: F64Array | None = None) -> F64Array:
        """alpha = (C'WC)^+ C'W (y - X m) on the training rows, for the scoring model (m this fixed point's mean, or
        ``mean``: the mixture's); a binary model's in its metric, W = diag(omega) and y its working response z."""
        xp = self.gaussian.array_module
        values = np.zeros((self.sample_count, 1))
        grouped = self._group_values((self.mean[:, model] if mean is None else np.asarray(mean, dtype=np.float64))[:, None])
        for start, stop, tile in self.gaussian.source.blocks():
            values += np.asarray(_host(tile.matmat(xp.asarray(grouped[start:stop]))), dtype=np.float64)
        targets = self.targets[:, model]
        residual = self.weights[:, model] * (targets - values[:, 0])
        covariates = np.asarray(_host(self.gaussian.covariates), dtype=np.float64)
        right = xp.asarray((covariates.T @ residual)[:, None])
        return np.asarray(_host(self.models.covariate_solve(right, xp.asarray([model]))), dtype=np.float64)[:, 0]


def state_digest(parts: Sequence[F64Array]) -> np.ndarray:
    """SHA-256 over the returned state's arrays, in order (each one's dtype, shape and bytes), as 32 uint8: what the
    certificate is bound to, so a reader can check that the diagnostics it holds are the ones of the model it scores."""
    digest = hashlib.sha256()
    for part in parts:
        array = np.ascontiguousarray(np.asarray(part, dtype=np.float64))
        digest.update(f"{array.dtype.str}{array.shape}".encode())
        digest.update(array.tobytes())
    return np.frombuffer(digest.digest(), dtype=np.uint8).copy()


def fit_full_data(
    *, gaussian: DualGaussian, statistics: GenotypeSufficientStatistics, prior: ScaleMixturePrior, draw_count: int, working_bytes: int, seed: int,
    inference: str = "ep",
    sites: Sequence[BernoulliSites | None] | None = None,
) -> FullDataFit:
    """Stage 2 from the prior (see the module docstring); ``seed`` draws the certificate's variant-side probes.
    ``inference`` names the fixed point: "mean_field" (``_FullDataMeanField``: the product q of ``mean_field`` on the
    streamed design) or "ep" (this module's EP, started from the mean-field fixed point).

    ``sites`` gives each binary model's starting Polya-Gamma sites (None for a quantitative model;
    ``binary_likelihood``), with the dual solver already in their metric (its sample weights the sites' weights, its
    targets their working responses): such a model is fitted on its Bernoulli bound at unit noise, known, by the
    mean-field route only, and its fixed points move their own sites. Stage 0's statistics for it are those of its
    start's equal-weight problem (response 4 kappa at noise 4, ``stage2_wiring``), exact at xi = 0.

    The certificate is bound to the returned state. On the mean-field route a model whose best-ELBO state is returned
    in place of where the outer loop ended (``restore_best``) has its outer-loop terms (remaining gain, Newton
    decrement, prediction move) withheld as NaN and its outer criterion unmet, since they describe the state the loop
    ended at, and ``restored_best`` says so; the fixed-point terms (mean move, noise gain) are the returned mixture's,
    the largest over its components (each measured at its own fixed point); ``mixture_components`` counts them,
    ``budget_unresolved`` counts the solves and mixture searches that ended at their derived work budgets, and
    ``state_digest`` is the SHA-256 of the returned mean, hyperparameters, noise, components, weights and covariate
    coefficients (``state_digest``)."""
    # EB starts where each trait's genetic variance fits inside its phenotypic variance (lead ruling): the first mean
    # solve's iterations grow with the prior signal per sample, which a start at the lattice centre puts far past it.
    moments = moment_starts(statistics, prior)
    if len(moments) != gaussian.model_count:
        raise ValueError("Stage 0's targets must be the models, in order")
    if inference not in ("ep", "mean_field"):
        raise ValueError(f"inference must be 'ep' or 'mean_field', not {inference!r}")
    model_sites = list(sites) if sites is not None else [None] * gaussian.model_count
    binary = np.array([entry is not None for entry in model_sites])
    if binary.any() and inference != "mean_field":
        raise ValueError("a binary model is fitted by the mean-field route (its Polya-Gamma bound); EP has no binary likelihood")
    def solve(
        model_prior: ScaleMixturePrior, starts: list[MixtureHyperparameters], noise: F64Array, start_mean: F64Array | None,
        start_sites: Sequence[BernoulliSites | None],
    ):
        extra = {} if start_mean is None else {"start_mean": start_mean}
        extra["sites"] = list(start_sites)
        oracle = _FullDataMeanField(gaussian, statistics, model_prior, draw_count, working_bytes, seed, starts, noise, **extra)
        try:
            return oracle, fit_hyperparameters(model_prior, starts, oracle, working_bytes, 0.5 / draw_count)
        except FloatingPointError as error:
            # The oracle's refusals say why it had no fixed point; they belong with the failure.
            raise FloatingPointError(f"{error}; mean_field refusals: {oracle.refusals}") from error

    # A binary model's noise is known: 1 in its whitened coordinates.
    noise = np.where(binary, 1.0, np.array([moment.noise for moment in moments]))
    if prior.annotation_groups:
        # Nested empirical Bayes (``small_n.fit_small_n``): the prior without annotation groups first, then the annotated
        # one continued from its fit, every annotation effect zero at its lambda = infinity edge
        # (``embed_hyperparameters``), so an annotation group enters only where the evidence rises.
        base = without_annotations(prior)
        base_points, base_fits = solve(base, [initial_hyperparameters(base, moment.mean_variance) for moment in moments], noise, None, model_sites)
        starts = [embed_hyperparameters(base, prior, fit.hyperparameters) for fit in base_fits]
        fixed_points, fits = solve(
            prior, starts, np.asarray(base_points.noise, dtype=np.float64).copy(), base_points.mean.copy(), base_points.sites,
        )
    else:
        fixed_points, fits = solve(prior, [initial_hyperparameters(prior, moment.mean_variance) for moment in moments], noise, None, model_sites)
    if inference == "ep":
        # EP starts where mean field ended (its fitted hyperparameters, noise and sites: q's means and precision), so its
        # refreshes pay for EP's correction to mean field, not the path from the prior. Mean field is its own route to
        # that point; EP then moves both q and the hyperparameters to EP's own fixed point.
        field_hyperparameters = list(fixed_points.restore_best(tuple(fit.hyperparameters for fit in fits), 0.5 / draw_count))
        log(f"ep: warm start from the mean-field fixed point after {fixed_points.passes} sweeps")
        oracle = _FullDataFixedPoints(
            gaussian, statistics, prior, draw_count, working_bytes, seed, field_hyperparameters, np.asarray(fixed_points.noise, dtype=np.float64).copy(),
            start_sites=(fixed_points.site_precision.copy(), fixed_points.site_shift.copy()),
        )
        try:
            fits = fit_hyperparameters(prior, field_hyperparameters, oracle, working_bytes, 0.5 / draw_count)
        except FloatingPointError as error:
            raise FloatingPointError(f"{error}; ep refusals: {oracle.refusals}") from error
        fixed_points = oracle
    mean_field = fixed_points if inference == "mean_field" else None
    hyperparameters = tuple(fit.hyperparameters for fit in fits)
    model_count = gaussian.model_count
    restored = np.zeros(model_count, dtype=bool)
    mean_move = np.array(fixed_points.mean_move, dtype=np.float64)
    noise_gains = np.array(fixed_points.noise_gain, dtype=np.float64)
    component_count = np.ones(model_count, dtype=np.int64)
    budget_unresolved = np.zeros(model_count, dtype=np.int64)
    if prior.annotation_groups and mean_field is not None:
        budget_unresolved += np.asarray(base_points.unresolved, dtype=np.int64)
    refusals = list(fixed_points.refusals)
    components: tuple[tuple[F64Array, F64Array], ...] = ()
    member_mean = component_weights = None
    binary_parts: tuple = ()
    if mean_field is not None:
        hyperparameters, restored = mean_field.restore_best(hyperparameters, 0.5 / draw_count)
        for model in np.flatnonzero(restored):
            refusals.append(
                f"model {int(model)}: the fit returns its best-ELBO state, not the one the outer loop ended at; the outer loop's "
                "remaining gain, Newton decrement and prediction move describe that other state and are withheld"
            )
        mixture = _mode_mixture(mean_field, gaussian, statistics, prior, draw_count, working_bytes, seed, hyperparameters, moments)
        member_mean, components, component_weights, binary_parts = mixture.mean, mixture.components, mixture.weights, mixture.binaries
        mean_move, noise_gains = mixture.mean_move, mixture.noise_gain
        component_count = np.full(model_count, len(components), dtype=np.int64)
        budget_unresolved += np.asarray(mean_field.unresolved, dtype=np.int64) + mixture.unresolved
        refusals.extend(mixture.refusals)
    covariate_coefficients = None if mean_field is None else np.column_stack(
        [mean_field.covariate_coefficients(model, member_mean[:, model]) for model in range(mean_field.model_count)]
    )
    withheld = np.where(restored, np.nan, 1.0)
    returned_mean = member_mean if member_mean is not None else np.asarray(_host(gaussian.mean), dtype=np.float64)
    digests = np.stack([
        state_digest(
            [returned_mean[:, model], hyperparameters[model].coefficients, np.atleast_1d(hyperparameters[model].log_smoothing),
             np.array([float(fixed_points.noise[model])])]
            + [part[:, model] for shift, omega in components for part in (shift, omega)]
            + ([] if component_weights is None else [component_weights[:, model]])
            + ([] if covariate_coefficients is None else [covariate_coefficients[:, model]])
        )
        for model in range(model_count)
    ])
    return FullDataFit(
        gaussian=gaussian,
        site_precision=fixed_points.site_precision,
        site_shift=fixed_points.site_shift,
        inference=inference,
        member_mean=member_mean,
        member_components=components,
        binary_components=binary_parts,
        component_weights=component_weights,
        member_shift=None if mean_field is None else mean_field.shift.copy(),
        member_omega=None if mean_field is None else mean_field.member_squares / mean_field.noise[None, :],
        covariate_coefficients=covariate_coefficients,
        working_bytes=int(working_bytes),
        hyperparameters=hyperparameters,
        noise_variance=fixed_points.noise,
        certificate=FitCertificate(
            remaining_gain=np.array([fit.remaining_gain for fit in fits]) * withheld,
            newton_decrement=np.array([fit.newton_decrement for fit in fits]) * withheld,
            smoothing_gradient=np.array([fit.step.smoothing_gradient for fit in fits]),
            stationarity_steps=tuple(fit.step.stationarity_steps for fit in fits),
            stationarity_errors=tuple(fit.step.stationarity_errors for fit in fits),
            mean_move=mean_move,
            draw_tolerance=np.full(gaussian.model_count, 1.0 / draw_count),
            noise_gain=noise_gains,
            mean_error=fixed_points.mean_error,
            # The cavity information certificate is EP's (its cavities come from leave-block-out variances); the
            # mean-field route's cavities are its own pseudo-likelihoods, so it has no such term.
            information_bound=np.array([float(np.max(certificate.upper_bound)) for certificate in fixed_points.information])
            if fixed_points.information else np.full(gaussian.model_count, np.nan),
            information_tolerance=np.array([float(np.min(certificate.tolerance)) for certificate in fixed_points.information])
            if fixed_points.information else np.full(gaussian.model_count, np.nan),
            undecided_blocks=fixed_points.undecided_blocks,
            negative_sites=np.sum(fixed_points.site_precision < 0.0, axis=0).astype(np.int64),
            effective_effects=fixed_points.effective,
            outer_iterations=np.array([fit.iterations for fit in fits], dtype=np.int64),
            halvings=np.array([fit.halvings for fit in fits], dtype=np.int64),
            prediction_move=np.array([fit.prediction_move for fit in fits]) * withheld,
            prediction_tolerance=np.array([fit.prediction_tolerance for fit in fits]),
            unresolved=np.array([fit.unresolved for fit in fits], dtype=np.int64),
            refusals=tuple(refusals),
            outer_history=tuple(fit.history for fit in fits),
            refreshes=fixed_points.refreshes,
            passes=fixed_points.passes,
            outer_criterion_met=np.array([fit.certified for fit in fits], dtype=bool) & ~restored & (budget_unresolved == 0),
            restored_best=restored,
            budget_unresolved=budget_unresolved,
            mixture_components=component_count,
            state_digest=digests,
        ),
    )




def _ridge_mean(
    gaussian: DualGaussian, statistics: GenotypeSufficientStatistics, prior: ScaleMixturePrior, moments: Sequence[MomentStart], draw_count: int,
    noise: F64Array | None = None,
) -> F64Array:
    """Each model's members' posterior means under the prior family's Gaussian member, beta_j ~ N(0, v u_j) with v the
    moment start's mean variance (``moment_starts``, the heritability's moment estimate) and u_j = e^(o_j): the dense
    end of the architectures, solved by the dual solver at uniform Gaussian sites, in the dual solver's current metric
    at ``noise`` (the moment start's, or a binary model's known 1). Tied members split their group's mean by their prior
    variances (the Gaussian posterior's split between identical columns)."""
    ties = TieGroups.from_tie_map(statistics.tie_map)
    offsets = np.exp(np.asarray(prior.log_variance_offset, dtype=np.float64))
    variances = np.array([float(moment.mean_variance) for moment in moments])
    precision = 1.0 / (offsets[:, None] * variances[None, :])
    shift = np.zeros_like(precision)
    group_precision, group_shift = group_sites(ties, precision, shift)
    gaussian.iterate(
        site_precision=group_precision, site_shift=group_shift,
        noise_variance=np.array([float(moment.noise) for moment in moments]) if noise is None else np.asarray(noise, dtype=np.float64),
        error_bound=np.full(len(moments), np.sqrt(1.0 / draw_count)), probe_residual_ratio=_HALF_PRECISION,
    )
    group_mean = np.asarray(_host(gaussian.mean), dtype=np.float64)
    mean, _variance = member_moments(ties, precision, shift, group_mean, np.zeros_like(group_mean))
    return mean


@dataclass(frozen=True)
class _Mixture:
    """``_mode_mixture``'s result: the weighted mean (members x models), each component's (shift h, omega), members x
    models, the weights (components x models), each component's ``BinaryComponent`` per binary model (None per
    quantitative one), the largest mean move and noise gain over the components (each at its own fixed point), per
    model, and whether the search ended at its budget unsettled (1) with why."""

    mean: F64Array
    components: tuple[tuple[F64Array, F64Array], ...]
    weights: F64Array
    binaries: tuple
    mean_move: F64Array
    noise_gain: F64Array
    unresolved: I64Array
    refusals: tuple[str, ...]


def _mode_mixture(
    main: "_FullDataMeanField", gaussian: DualGaussian, statistics: GenotypeSufficientStatistics, prior: ScaleMixturePrior, draw_count: int,
    working_bytes: int, seed: int, hyperparameters: tuple[MixtureHyperparameters, ...], moments: Sequence[MomentStart] = (),
) -> _Mixture:
    """The mean-field fit's mixture over coordinate ascent's modes (``small_n._mode_mixture``): between near-duplicate
    columns the posterior is multimodal and the product family holds one mode, the column visited first taking the
    effect, while the posterior mean averages them. Fixed points are solved at the fitted hyperparameters from zero in
    random within-block member orders, each weighted per model by its evidence, exp(ELBO) (``small_n._mixture_weights``:
    a poor fixed point carries no weight; a repeat of a held mode is merged into it), until one more moves every model's
    fitted genetic values by at most the draws' resolution, ||Xp d||^2 / sigma^2 <= 1/K; the main fixed point is the
    first component and the fixed point from the Gaussian member's posterior mean (``_ridge_mean``, when ``moments`` are
    given) the second, and the main fixed point's dual-solver state is put back at the end.

    A binary model's components each start from the main fixed point's sites and move their own (their Bernoulli
    bounds are the evidence that weighs them); a move of the fitted genetic values is measured in the main fixed point's
    metric at its known unit noise.

    Work budget: at most K random orders are tried. The mixture reaches the scorer through its K draws, each from one
    component (``draw_laws.ProductMixtureDraws``), so no more than K components are ever represented; a search that has
    tried K orders (a refused order counts: it consumed the attempt) without settling ends there, unresolved, and says
    so rather than run on."""
    def pieces(oracle: "_FullDataMeanField") -> tuple[F64Array, F64Array, F64Array]:
        return oracle.mean.copy(), oracle.shift.copy(), oracle.member_squares / oracle.noise[None, :]

    def binary_of(oracle: "_FullDataMeanField") -> tuple:
        return tuple(None if model_sites is None else oracle.binary_component(model) for model, model_sites in enumerate(oracle.sites))

    means, components, elbos, binaries, moves, gains = [], [], [], [], [], []

    def held(oracle: "_FullDataMeanField") -> None:
        mean, shift, omega = pieces(oracle)
        means.append(mean); components.append((shift, omega)); elbos.append(np.array(oracle.elbo, dtype=np.float64)); binaries.append(binary_of(oracle))
        moves.append(np.array(oracle.mean_move, dtype=np.float64)); gains.append(np.array(oracle.noise_gain, dtype=np.float64))

    held(main)

    def weighted() -> tuple[F64Array, F64Array]:
        values = np.array(elbos)
        weights = np.exp(values - values.max(axis=0))
        weights /= weights.sum(axis=0)
        return np.einsum("cm,cjm->jm", weights, np.array(means)), weights

    def admit(oracle: "_FullDataMeanField") -> None:
        """A new mode, or a repeat of a held one merged into it (``small_n._admit``: the mixture is over distinct
        modes, each weighted by its own mass once); the repeat of the higher evidence is kept. The component's panel
        Grams are released: its order's panels are no other component's."""
        oracle._panel_grams.clear()
        elbo = np.array(oracle.elbo, dtype=np.float64)
        for index, kept in enumerate(means):
            if all(
                float(np.sum(np.square(main._image((oracle.mean - kept)[:, model:model + 1], model)))) / float(main.noise[model]) <= 1.0 / draw_count
                for model in range(main.model_count)
            ):
                if float(np.sum(elbo)) > float(np.sum(elbos[index])):
                    mean, shift, omega = pieces(oracle)
                    means[index], components[index], elbos[index], binaries[index] = mean, (shift, omega), elbo, binary_of(oracle)
                    moves[index], gains[index] = np.array(oracle.mean_move, dtype=np.float64), np.array(oracle.noise_gain, dtype=np.float64)
                return
        held(oracle)

    average, _weights = weighted()
    if moments:
        # The dense end (``_ridge_mean``): with many small effects the zero-started fixed point keeps a sparse basin and
        # over-shrinks (bench-sim scenario_001 [sim]: calibration slope 2.33, r2 below the infinitesimal ridge's), and
        # the evidence weights decide between the basins.
        main._sync()
        ridge = _FullDataMeanField(
            gaussian, statistics, prior, draw_count, working_bytes, seed, list(hyperparameters), main.noise.copy(),
            start_mean=_ridge_mean(
                gaussian, statistics, prior, moments, draw_count,
                np.where([model_sites is not None for model_sites in main.sites], 1.0, [float(moment.noise) for moment in moments]),
            ),
            sites=list(main.sites),
        )
        points = ridge(list(hyperparameters))
        if not any(point is None for point in points):
            admit(ridge)
            average, _weights = weighted()
        del ridge, points
    settled = False
    unresolved = np.zeros(main.model_count, dtype=np.int64)
    refusals: list[str] = []
    for component in range(1, draw_count + 1):
        oracle = _FullDataMeanField(
            gaussian, statistics, prior, draw_count, working_bytes, seed, list(hyperparameters), main.noise.copy(), order_seed=component,
            sites=list(main.sites),
        )
        points = oracle(list(hyperparameters))
        unresolved += oracle.unresolved
        if any(point is None for point in points):
            continue
        admit(oracle)
        del oracle, points
        updated, _weights = weighted()
        settled = all(
            float(np.sum(np.square(main._image((updated - average)[:, model:model + 1], model)))) / float(main.noise[model]) <= 1.0 / draw_count
            for model in range(main.model_count)
        )
        average = updated
        if settled:
            break
    if not settled:
        unresolved += 1
        refusals.append(
            f"the mode mixture tried its budget of {draw_count} orders (one per draw that can represent a component) without settling: "
            f"{len(components)} components, unresolved"
        )
    main._iterate(main.site_precision, main.site_shift, main.noise)
    main._panel_grams.clear()
    average, weights = weighted()
    return _Mixture(
        mean=average, components=tuple(components), weights=weights, binaries=tuple(binaries), mean_move=np.max(np.array(moves), axis=0),
        noise_gain=np.max(np.array(gains), axis=0), unresolved=unresolved, refusals=tuple(refusals),
    )


def scoring_models(
    fit: FullDataFit, prior: ScaleMixturePrior, statistics: GenotypeSufficientStatistics, trait_types: Sequence[TraitType], draw_count: int, seed: int
) -> list[ScoringModel]:
    """One ``fast_scoring.ScoringModel`` per model over every active store row: each tie member's own posterior mean
    and the law of its K posterior draws, and the covariate coefficients. Tied members are equal on the training samples
    only, so each keeps its effect.

    The mean-field route's draws are a law (``draw_laws.ProductMixtureDraws``: the mixture's components, weights and
    lattice, O(components x members), each draw's component drawn from the weights), drawn a tile of rows at a time
    when scored or exported; nothing of the size members x draws is formed here. The covariate coefficients'
    conditional means per draw are summed over tiles of members: a quantitative model's alpha - (C'C)^+ L d_k with d_k
    the draw's move of the reduced columns' effects, as (C'C)^+ L S (tile - mean) (S the members' signed group map); a
    binary model's anchor_c - loading_c b_k in the metric of the draw's own component c (``BinaryComponent``), as
    loading_c S tile. So no group x draws array is formed either. The EP route's draws are its Gaussian's
    perturb-and-solve draws, joint over every column, which exist only whole (the dual solver's p x K)."""
    gaussian = fit.gaussian
    ties = TieGroups.from_tie_map(statistics.tie_map)
    identity = _compact_identity_tie_map(ties.member_count)
    if fit.inference == "mean_field":
        assert fit.member_mean is not None and fit.member_shift is not None and fit.member_omega is not None and fit.covariate_coefficients is not None
        mean, alpha = fit.member_mean, fit.covariate_coefficients
        class_index = np.asarray(prior.class_index, dtype=np.int64)
    else:
        error_bound = np.full(len(trait_types), np.sqrt(1.0 / draw_count))
        group_draws = np.asarray(_host(gaussian.draws(draw_count=draw_count, error_bound=error_bound, seed=seed)), dtype=np.float64)
        alpha = np.asarray(_host(gaussian.alpha), dtype=np.float64)
        group_mean = np.asarray(_host(gaussian.mean), dtype=np.float64)
        mean, _variance = member_moments(ties, fit.site_precision, fit.site_shift, group_mean, np.zeros_like(group_mean))
    models = []
    loading = np.concatenate([statistics.ld.block(index).covariate_cross for index in range(statistics.ld.block_count)], axis=0).T
    conditional_loading = statistics.covariate_gram_pseudo_inverse @ loading
    # A tile of draws is as many members as the widest LD block holds (the fit's own scheduling unit).
    block_of_group = np.searchsorted(np.asarray(statistics.ld.block_boundaries), np.arange(ties.group_count), side="right") - 1
    widest = int(np.max(np.bincount(block_of_group[ties.group])))
    broker = current_broker()
    for model, trait_type in enumerate(trait_types):
        model_alpha = alpha[:, model]
        covariate_covariance = fit.noise_variance[model] * statistics.covariate_gram_pseudo_inverse
        shift_value = 0.0
        if fit.inference == "mean_field":
            parts = fit.member_components or ((fit.member_shift, fit.member_omega),)
            weights = np.ones(len(parts)) / len(parts) if fit.component_weights is None else fit.component_weights[:, model]
            coefficients = fit.hyperparameters[model].coefficients
            law = ProductMixtureDraws(
                class_index=class_index, log_scale=log_scale(prior, coefficients), log_density=class_log_density(prior, coefficients),
                log_variance_grid=np.asarray(prior.log_variance_grid, dtype=np.float64),
                shift=np.stack([np.asarray(shift, dtype=np.float64)[:, model] for shift, _omega in parts]),
                omega=np.stack([np.asarray(omega, dtype=np.float64)[:, model] for _shift, omega in parts]),
                weights=np.asarray(weights, dtype=np.float64) / float(np.sum(weights)), key=seed_key(seed, model), draw_count=draw_count,
            )
            binary = [part[model] for part in fit.binary_components] if fit.binary_components else []
            binary = binary if binary and binary[0] is not None else []
            chosen = law.components_of_draws()
            # Each (quantitative) or each component's (binary) loading over the members, signed: k x members.
            loadings = [component.loading[:, ties.group] * ties.sign[None, :] for component in binary] or [conditional_loading[:, ties.group] * ties.sign[None, :]]
            loaded = np.zeros((alpha.shape[0], draw_count))
            step = max(1, min(widest, tile_rows(law, fit.working_bytes) if fit.working_bytes > 0 else widest))
            row_bytes = law.tile_row_bytes() + _FLOAT_BYTES * draw_count
            with (broker.reserve(HOST, min(step, ties.member_count) * row_bytes, "a tile of posterior draws") if broker is not None else contextlib.nullcontext()):
                for first in range(0, ties.member_count, step):
                    last = min(first + step, ties.member_count)
                    tile = law.tile(first, last)
                    if binary:
                        for index, member_loading in enumerate(loadings):
                            columns = np.flatnonzero(chosen == index)
                            if columns.size:
                                loaded[:, columns] += member_loading[:, first:last] @ tile[:, columns]
                    else:
                        loaded += loadings[0][:, first:last] @ (tile - mean[first:last, model, None])
            if binary:
                model_alpha, covariate_draws, covariate_covariance, shift_value = _binary_covariates(binary, weights, chosen, loaded)
            else:
                covariate_draws = alpha[:, model, None] - loaded
        else:
            law = member_draws(
                ties, fit.site_precision[:, model], fit.site_shift[:, model], group_draws[:, model, :], np.random.default_rng([seed, model])
            )
            covariate_draws = alpha[:, model, None] - conditional_loading @ (group_draws[:, model, :] - group_mean[:, model, None])
        models.append(ScoringModel.from_reduced_fit(
            active_rows=np.asarray(statistics.active_rows, dtype=np.int64),
            signed_means=np.asarray(statistics.means, dtype=np.float64),
            signed_scales=np.asarray(statistics.scales, dtype=np.float64),
            tie_map=identity,
            member_prior_variances=prior_second_moment(prior, fit.hyperparameters[model]),
            beta_reduced=mean[:, model],
            posterior_draws_reduced=law,
            alpha=model_alpha,
            trait_type=trait_type,
            predictive_intercept_shift=shift_value,
            covariate_draws=covariate_draws,
            covariate_covariance=covariate_covariance,
            # The EP route's draws are its Gaussian's; the mean-field route's are the product's scale mixtures.
            gaussian_posterior=fit.inference != "mean_field",
        ))
    return models


def _binary_covariates(
    components: Sequence[BinaryComponent], weights: F64Array, chosen: I64Array, loaded: F64Array,
) -> tuple[F64Array, F64Array, F64Array, float]:
    """A binary model's covariate terms and predictive shift over its mixture (``small_n._binary_covariates``), each
    component in its own metric: alpha = sum_c w_c alpha_c, each draw k's covariates anchor_c - loading_c b_k for the
    component c = ``chosen[k]`` it came from (``loaded[:, k]`` = loading_c b_k, summed over tiles of members), the
    within-component covariance sum_c w_c (C'W_c C)^+, and the intercept shift that puts the mean predictive over the
    training rows at their prevalence under the mixture's predictor moments (``binary_likelihood.calibrated_shift``)."""
    alpha = np.einsum("c,ck->k", weights, np.array([component.alpha for component in components]))
    covariance = np.einsum("c,ckl->kl", weights, np.array([component.covariance for component in components]))
    anchors = np.array([component.anchor for component in components])
    covariate_draws = anchors[chosen].T - loaded
    means = np.array([component.predictor_mean for component in components])
    mixture_mean = np.einsum("c,ci->i", weights, means)
    mixture_variance = np.einsum("c,ci->i", weights, np.array([component.predictor_variance for component in components]) + (means - mixture_mean[None, :]) ** 2)
    shift = calibrated_shift(mixture_mean, mixture_variance, components[0].labels)
    return alpha, covariate_draws, covariance, shift
