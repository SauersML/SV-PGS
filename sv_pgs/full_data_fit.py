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
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterator, Sequence

import numpy as np

from sv_pgs._typing import BoolArray, F64Array, I64Array
from sv_pgs.config import TraitType
from sv_pgs.device_sweep import PIECE_COLUMNS, PanelGrams, sweep_piece
from sv_pgs.dual_solve import DualGaussian, DualModels, _host
from sv_pgs.fast_scoring import ScoringModel
from sv_pgs.genotype_statistics import GenotypeSufficientStatistics
from sv_pgs.krylov_recycle import local_response
from sv_pgs.tie_members import TieGroups, _group_sum, group_sites, member_draws, member_moments, member_weights, tied_groups, tied_weights
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
    marginal_variances,
    variance_jvp,
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
    initial_hyperparameters,
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


def block_grams(statistics: GenotypeSufficientStatistics, noise: float = 1.0) -> BlockGrams:
    """Stage 0's projected Grams, R_b within each block and R_{b,b+1} between neighbours (zero across a chromosome's
    end), as the stored float32 arrays themselves: memory-mapped views, no copy. A model's metric W = training /
    sigma^2 enters as ``scale = 1 / noise``; every model of a fit shares the arrays through
    ``dataclasses.replace(grams, scale=...)``, and ``marginal_variances`` promotes one window at a time to float64.
    (Building float64 copies per model and refresh held about 12 GB of each kind per model at p = 466k and ran
    e2e-scale's chr22 fit out of host memory at 45 GB.)"""
    ld = statistics.ld
    blocks = tuple(np.asarray(ld.block(block_index).reduced_columns, dtype=np.int64) for block_index in range(ld.block_count))
    within = tuple(ld.block(block_index).projected_gram for block_index in range(ld.block_count))
    next_cross = []
    for block_index in range(1, ld.block_count):
        cross = ld.adjacent_block(block_index)
        shape = (blocks[block_index - 1].shape[0], blocks[block_index].shape[0])
        next_cross.append(np.zeros(shape, dtype=np.float32) if cross is None else cross)
    return BlockGrams(blocks=blocks, within=within, next_cross=tuple(next_cross), scale=1.0 / noise)


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
        # Stage 0's Grams, built once per fit and shared by every model and refresh (``block_grams``); the window
        # algebra's float64 working set is charged against the fit's budget.
        self.grams = block_grams(statistics)
        window_bytes = window_working_bytes(self.grams)
        if window_bytes > working_bytes:
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
            try:
                self._iterate(self.site_precision, self.site_shift)
            except np.linalg.LinAlgError:
                failure = "the full-data precision is not positive definite with non-negative sites"
            else:
                grams = [replace(self.grams, scale=1.0 / float(self.noise[model])) for model in range(gaussian.model_count)]
                group_variances = np.column_stack([
                    marginal_variances(solve, model_grams, gaussian.array_module) for solve, model_grams in zip(gaussian.bulk_solves, grams)
                ])
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
                        raise NoFixedPoint(
                            f"model {model}: the EP refreshes' updates line up with no contraction (KL {lower[model]:.3e}..{upper[model]:.3e} after "
                            f"{last:.3e}, r_k' Sigma r_(k-1) = {cross:.3e} +- {cross_error:.3e}): no damping contracts it, and the full-data route "
                            "has no double loop"
                        )
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


class _PassBudget(Exception):
    """A second-start solve used the first start's count of passes without converging: abandoned, not refused."""


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
    (``_member_posterior``), as on the EP route."""

    def __init__(
        self, gaussian: DualGaussian, statistics: GenotypeSufficientStatistics, prior: ScaleMixturePrior, draw_count: int, working_bytes: int, seed: int,
        starts: Sequence[MixtureHyperparameters], noise: F64Array,
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
        self.sign = np.asarray(self.ties.sign, dtype=np.float64)
        model_count = gaussian.model_count
        self.model_count = model_count
        self.training = np.asarray(_host(gaussian.training), dtype=np.float64)
        self.sample_count = int(self.training.shape[0])
        self.training_counts = np.asarray(gaussian.training_counts, dtype=np.int64)
        # The projector alone (unit weights): Xp and y_P do not move with the noise, which the sweeps update.
        xp = gaussian.array_module
        self.models = DualModels(gaussian.training, xp.zeros((gaussian.source.variant_count, model_count)), gaussian.covariates, xp)
        factor = np.asarray(_host(self.models.covariate_factor), dtype=np.float64)
        self.covariate_rank = np.array([int(np.count_nonzero(np.any(factor[model] != 0.0, axis=0))) for model in range(model_count)], dtype=np.int64)
        self.residual_dimension = self.training_counts - self.covariate_rank
        if np.any(self.residual_dimension <= 0):
            raise ValueError("a model's training rows do not outnumber its covariates' rank")
        self.member_squares = np.asarray(_host(gaussian.unit_squares), dtype=np.float64)[self.ties.group]
        targets = np.asarray(_host(gaussian.targets), dtype=np.float64)
        projected = np.asarray(_host(self.models.complement(xp.asarray(self.training * targets), xp.arange(model_count))), dtype=np.float64)
        self.projected_targets = [np.ascontiguousarray(projected[:, model]) for model in range(model_count)]
        self.class_index = np.asarray(prior.class_index, dtype=np.int64)
        self.noise = np.array(noise, dtype=np.float64, copy=True)
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
        self.passes = 0
        self.version = 0
        self.undecided_blocks = 0
        self.information: list = []
        self._panel_grams = PanelGrams()
        self._cold = self._snapshot()
        # Each model's highest-ELBO refreshed state and its hyperparameters (``mean_field.MeanFieldFixedPoints.best``):
        # the outer loop's objective is not the ELBO and can end below a state it visited.
        self.best_elbo = np.full(model_count, -np.inf)
        self.best_state: list[dict | None] = [None] * model_count
        self.best_hyperparameters: list[MixtureHyperparameters | None] = [None] * model_count

    # state

    def _snapshot(self) -> dict:
        return {
            "mean": self.mean.copy(), "variance": self.variance.copy(), "shift": self.shift.copy(), "third": self.third.copy(),
            "fourth": self.fourth.copy(), "residual": [values.copy() for values in self.residual], "noise": self.noise.copy(),
            "site_precision": self.site_precision.copy(), "site_shift": self.site_shift.copy(), "effective": self.effective.copy(),
            "mean_move": self.mean_move.copy(), "noise_gain": self.noise_gain.copy(), "elbo": self.elbo.copy(), "version": self.version,
        }

    def _restore(self, snapshot: dict, models: Sequence[int] | None = None) -> None:
        columns = list(range(self.model_count)) if models is None else list(models)
        for name in ("mean", "variance", "shift", "third", "fourth", "site_precision", "site_shift"):
            getattr(self, name)[:, columns] = snapshot[name][:, columns]
        for name in ("noise", "effective", "mean_move", "noise_gain", "elbo"):
            getattr(self, name)[columns] = snapshot[name][columns]
        for model in columns:
            self.residual[model] = snapshot["residual"][model].copy()

    def _ensure(self, snapshot: dict) -> None:
        """The dual solver back at this fixed point's sites before it answers for it (a later trial may have moved it)."""
        if self.version != snapshot["version"]:
            self._iterate(snapshot["site_precision"], snapshot["site_shift"], snapshot["noise"])
            self.version = snapshot["version"]

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

    def _projected(self, tile, local: I64Array, model: int) -> F64Array:
        xp = self.gaussian.array_module
        dense = xp.asarray(tile.columns(xp.asarray(local)), dtype=xp.float64)
        masked = dense * xp.asarray(self.training[:, model])[:, None]
        projected = self.models.complement(masked, xp.full(local.shape[0], model, dtype=xp.int64))
        return np.asfortranarray(np.asarray(_host(projected), dtype=np.float64))

    def _image(self, coefficients: F64Array, model: int) -> F64Array:
        """Xp c (n x r) for c (p x r): the tiles' X_b c_b, masked to the training rows and projected."""
        xp = self.gaussian.array_module
        values = self._group_values(np.asarray(coefficients, dtype=np.float64))
        image = np.zeros((self.sample_count, values.shape[1]))
        for start, stop, tile in self.gaussian.source.blocks():
            image += np.asarray(_host(tile.matmat(xp.asarray(values[start:stop]))), dtype=np.float64)
        masked = xp.asarray(image * self.training[:, model][:, None])
        return np.asarray(_host(self.models.complement(masked, xp.full(values.shape[1], model, dtype=xp.int64))), dtype=np.float64)

    def _back(self, samples: F64Array, model: int) -> F64Array:
        """Xp' u (p x r) for u (n x r): (I - H) and the training mask, then the tiles' X_b' u."""
        xp = self.gaussian.array_module
        values = np.asarray(samples, dtype=np.float64)
        projected = np.asarray(_host(self.models.complement(xp.asarray(values), xp.full(values.shape[1], model, dtype=xp.int64))), dtype=np.float64)
        masked = xp.asarray(projected * self.training[:, model][:, None])
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
        mask = cupy.asarray(self.training[:, model])
        residual = cupy.asarray(self.residual[model])
        squares = cupy.asarray(np.ascontiguousarray(self.member_squares[:, model]))
        classes = cupy.asarray(self.class_index)
        state = {name: cupy.asarray(np.ascontiguousarray(getattr(self, name)[:, model])) for name in ("mean", "variance", "shift", "third", "fourth")}
        pieces = cupy.zeros((prior.variant_count, PIECE_COLUMNS))
        noise = float(self.noise[model])

        def project(values):
            return self.models.complement(values, cupy.full(values.shape[1], model, dtype=cupy.int64))

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
                cupy, decode=decode, width=int(members.shape[0]), mask=mask, project=project, residual=residual, grams=self._panel_grams,
                key_base=(model, int(members[0])), squares=cupy.ascontiguousarray(squares[rows]), class_index=cupy.ascontiguousarray(classes[rows]),
                log_density=log_density, node_variance=node_variance, log_node_variance=log_node_variance, noise=noise,
                pieces=piece_parts, **piece_state,
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
        """As ``MeanFieldFixedPoints._elbo``, on this model's training rows."""
        noise = float(self.noise[model])
        residual_term = 0.5 * float(self.residual_dimension[model]) * float(np.log(2.0 * np.pi * noise))
        fit_term = (residual_square + weighted_variance) / (2.0 * noise)
        value = -residual_term - fit_term - divergence
        summands = 2 * self.prior.variant_count + int(self.training_counts[model])
        return value, (self.prior.grid_size + 1 + summands) * _EPSILON * (abs(residual_term) + fit_term + sizes)

    def _solve_model(self, model: int, hyperparameters: MixtureHyperparameters, pass_budget: int | None = None) -> None:
        """Sweeps at the current noise, the noise moving to its stationary value between them, until the sweeps'
        measured remainder (the geometric extrapolation of the last two gains, ``mean_field``) plus the noise's
        pending gain is within the tolerance."""
        tolerance = 0.5 / self.draw_count
        elbo: float | None = None
        gain: float | None = None
        previous_gain: float | None = None
        pending_noise: float | None = None
        passes_at_entry = self.passes
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
            pending_noise = (residual_square + weighted_variance) / float(self.residual_dimension[model])
            self.noise_gain[model] = noise_gain(pending_noise, float(self.noise[model]), int(self.training_counts[model]), int(self.covariate_rank[model]))
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
                return

    def _iterate(self, site_precision: F64Array, site_shift: F64Array, noise: F64Array) -> None:
        """The dual solver at q's precision and mean: sites tau = 1/v - omega and nu = m/v - h per member (identity
        ties), whose Gaussian has precision diag(tau) + Xp'Xp / sigma^2 = R and mean m."""
        # A linear response's sites, not a Gaussian's: the members' elimination needs no sign (tied_weights).
        group_precision, group_shift = group_sites(self.ties, site_precision, site_shift, algebraic=True)
        self.gaussian.iterate(
            site_precision=group_precision, site_shift=group_shift, noise_variance=noise,
            error_bound=np.full(self.model_count, np.sqrt(1.0 / self.draw_count)), probe_residual_ratio=_HALF_PRECISION,
            # q's sites: a scale-mixture member's variance can exceed 1 / omega (its tau is then negative), so R is a
            # linear response's precision, symmetric and nonsingular, and not always a Gaussian's (``mean_field``'s
            # dense route solves the same system by a symmetric indefinite factorization).
            indefinite_core=True,
        )
        self.version += 1

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
        entry = self._snapshot()
        solved: list[list[tuple[float, dict] | None]] = []
        first = self.refreshes == 0
        # The cold solve gets the carried solve's own count of passes per model, and no more (``mean_field``).
        budgets: list[int | None] = [None] * self.model_count
        for start in ((entry,) if first else (entry, self._cold)):
            self._restore(start)
            outcome: list[tuple[float, dict] | None] = []
            for model in range(self.model_count):
                before = self.passes
                try:
                    self._solve_model(model, hyperparameters[model], pass_budget=budgets[model])
                except _PassBudget:
                    outcome.append(None)
                    continue
                except (FloatingPointError, ZeroDivisionError) as error:
                    self.refusals.append(f"{type(error).__name__}: {error}")
                    outcome.append(None)
                    continue
                if budgets[model] is None:
                    budgets[model] = self.passes - before
                outcome.append((float(self.elbo[model]), self._snapshot()))
            solved.append(outcome)
        for model in range(self.model_count):
            candidates = [outcome[model] for outcome in solved if outcome[model] is not None]
            if not candidates:
                # No start reaches this model's fixed point: the trial is refused whole, as EP's oracle refuses.
                self._restore(entry)
                return [None] * self.model_count
            _value, state = max(candidates, key=lambda item: item[0])
            self._restore(state, [model])
        for model in range(self.model_count):
            omega, tau, nu, _live = self._sites(model)
            self.site_precision[:, model] = tau
            self.site_shift[:, model] = nu
            self.effective[model] = max(float(np.sum(omega * self.variance[:, model])), _EPSILON * tau.shape[0])
        try:
            self._iterate(self.site_precision, self.site_shift, self.noise)
        except np.linalg.LinAlgError as error:
            self.refusals.append(f"the dual solver has no factor at q's sites: {error}")
            self._restore(entry)
            return [None] * self.model_count
        self.refreshes += 1
        improved = [model for model in range(self.model_count) if self.elbo[model] > self.best_elbo[model]]
        if improved:
            state = self._snapshot()
            for model in improved:
                self.best_elbo[model], self.best_state[model], self.best_hyperparameters[model] = self.elbo[model], state, hyperparameters[model]
        return [self._fixed_point(model, hyperparameters[model]) for model in range(self.model_count)]

    def restore_best(self, hyperparameters: Sequence[MixtureHyperparameters], tolerance: float) -> tuple[MixtureHyperparameters, ...]:
        """Each model back at its highest-ELBO refreshed state where that is above where the outer loop ended by more than
        ``tolerance``, the dual solver refactored at the restored sites; returns each model's hyperparameters."""
        chosen = list(hyperparameters)
        restored = [model for model in range(self.model_count)
                    if self.best_state[model] is not None and self.best_elbo[model] > self.elbo[model] + tolerance]
        for model in restored:
            self._restore(self.best_state[model], [model])  # type: ignore[arg-type]
            chosen[model] = self.best_hyperparameters[model]  # type: ignore[assignment]
        if restored:
            self._iterate(self.site_precision, self.site_shift, self.noise)
        return tuple(chosen)

    def _fixed_point(self, model: int, hyperparameters: MixtureHyperparameters) -> FixedPoint:
        omega, tau, _nu, live = self._sites(model)
        noise = float(self.noise[model])
        squares = self.member_squares[:, model].copy()
        mean, variance, shift = self.mean[:, model].copy(), self.variance[:, model].copy(), self.shift[:, model].copy()
        third, fourth = self.third[:, model].copy(), self.fourth[:, model].copy()
        residual = self.residual[model].copy()
        mean_by_omega = np.where(live, -0.5 * (third + 2.0 * mean * variance), 0.0)
        variance_by_shift = np.where(live, third, 0.0)
        variance_by_omega = np.where(live, -0.5 * (fourth - variance * variance) - mean * third, 0.0)
        residual_dimension = float(self.residual_dimension[model])
        snapshot = self._snapshot()
        grams = replace(self.grams, scale=1.0 / noise)
        group_variance = np.bincount(self.ties.group, weights=variance, minlength=self.ties.group_count)
        dual = _member_posterior(
            _posterior(self.gaussian, model, grams, group_variance, lambda: self._ensure(snapshot)), self.ties, tau, group_variance, algebraic=True,
        )
        noise_solve: dict[str, object] = {}

        def off_diagonal_gram(columns: F64Array) -> F64Array:
            return self._back(self._image(columns, model), model) - squares[:, None] * columns

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
                    2.0 * float(residual @ self._image(mean_one, model)[:, 0])
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
            _mean_one, shift_one, scalar = noise_terms(relative_tolerance)
            right = -2.0 * (residual @ self._image(mean_step, model)) + squares @ (
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
            restore=lambda: self._restore(snapshot, [model]), evidence_offset=offset,
        )

    def covariate_coefficients(self, model: int) -> F64Array:
        """alpha = (C'WC)^+ C'W (y - X m) on the training rows, for the scoring model."""
        xp = self.gaussian.array_module
        values = np.zeros((self.sample_count, 1))
        grouped = self._group_values(self.mean[:, model][:, None])
        for start, stop, tile in self.gaussian.source.blocks():
            values += np.asarray(_host(tile.matmat(xp.asarray(grouped[start:stop]))), dtype=np.float64)
        targets = np.asarray(_host(self.gaussian.targets), dtype=np.float64)[:, model]
        residual = self.training[:, model] * (targets - values[:, 0])
        covariates = np.asarray(_host(self.gaussian.covariates), dtype=np.float64)
        right = xp.asarray((covariates.T @ residual)[:, None])
        return np.asarray(_host(self.models.covariate_solve(right, xp.asarray([model]))), dtype=np.float64)[:, 0]


def fit_full_data(
    *, gaussian: DualGaussian, statistics: GenotypeSufficientStatistics, prior: ScaleMixturePrior, draw_count: int, working_bytes: int, seed: int,
    inference: str = "ep",
) -> FullDataFit:
    """Stage 2 for quantitative models, from the prior (see the module docstring); ``seed`` draws the certificate's
    variant-side probes. ``inference`` names the fixed point: "ep" (this module's EP) or "mean_field"
    (``_FullDataMeanField``: the product q of ``mean_field`` on the streamed design)."""
    # EB starts where each trait's genetic variance fits inside its phenotypic variance (lead ruling): the first mean
    # solve's iterations grow with the prior signal per sample, which a start at the lattice centre puts far past it.
    moments = moment_starts(statistics, prior)
    if len(moments) != gaussian.model_count:
        raise ValueError("Stage 0's targets must be the models, in order")
    starts = [initial_hyperparameters(prior, moment.mean_variance) for moment in moments]
    if inference not in ("ep", "mean_field"):
        raise ValueError(f"inference must be 'ep' or 'mean_field', not {inference!r}")
    oracle_class = _FullDataFixedPoints if inference == "ep" else _FullDataMeanField
    fixed_points = oracle_class(gaussian, statistics, prior, draw_count, working_bytes, seed, starts, np.array([moment.noise for moment in moments]))
    try:
        fits = fit_hyperparameters(prior, starts, fixed_points, working_bytes, 0.5 / draw_count)
    except FloatingPointError as error:
        # The oracle's refusals say why it had no fixed point; they belong with the failure.
        raise FloatingPointError(f"{error}; {inference} refusals: {fixed_points.refusals}") from error
    mean_field = fixed_points if inference == "mean_field" else None
    hyperparameters = tuple(fit.hyperparameters for fit in fits)
    if mean_field is not None:
        hyperparameters = mean_field.restore_best(hyperparameters, 0.5 / draw_count)
    return FullDataFit(
        gaussian=gaussian,
        site_precision=fixed_points.site_precision,
        site_shift=fixed_points.site_shift,
        inference=inference,
        member_mean=None if mean_field is None else mean_field.mean.copy(),
        member_shift=None if mean_field is None else mean_field.shift.copy(),
        member_omega=None if mean_field is None else mean_field.member_squares / mean_field.noise[None, :],
        covariate_coefficients=None if mean_field is None else np.column_stack([mean_field.covariate_coefficients(model) for model in range(mean_field.model_count)]),
        working_bytes=int(working_bytes),
        hyperparameters=hyperparameters,
        noise_variance=fixed_points.noise,
        certificate=FitCertificate(
            remaining_gain=np.array([fit.remaining_gain for fit in fits]),
            newton_decrement=np.array([fit.newton_decrement for fit in fits]),
            smoothing_gradient=np.array([fit.step.smoothing_gradient for fit in fits]),
            stationarity_steps=tuple(fit.step.stationarity_steps for fit in fits),
            stationarity_errors=tuple(fit.step.stationarity_errors for fit in fits),
            mean_move=fixed_points.mean_move,
            draw_tolerance=np.full(gaussian.model_count, 1.0 / draw_count),
            noise_gain=fixed_points.noise_gain,
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
            prediction_move=np.array([fit.prediction_move for fit in fits]),
            prediction_tolerance=np.array([fit.prediction_tolerance for fit in fits]),
            unresolved=np.array([fit.unresolved for fit in fits], dtype=np.int64),
            refusals=tuple(fixed_points.refusals),
            outer_history=tuple(fit.history for fit in fits),
            refreshes=fixed_points.refreshes,
            passes=fixed_points.passes,
            outer_criterion_met=np.array([fit.certified for fit in fits], dtype=bool),
        ),
    )


def scoring_models(
    fit: FullDataFit, prior: ScaleMixturePrior, statistics: GenotypeSufficientStatistics, trait_types: Sequence[TraitType], draw_count: int, seed: int
) -> list[ScoringModel]:
    """One ``fast_scoring.ScoringModel`` per model over every active store row: each tie member's own posterior mean
    and K exact posterior draws (``tie_members``: its group's, conditioned on the sum, with its own site and prior),
    and the covariate coefficients. Tied members are equal on the training samples only, so each keeps its effect."""
    gaussian = fit.gaussian
    ties = TieGroups.from_tie_map(statistics.tie_map)
    identity = _compact_identity_tie_map(ties.member_count)
    if fit.inference == "mean_field":
        # q's own means and its product draws (``mean_field.product_draws``: conditional variational draws).
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
    for model, trait_type in enumerate(trait_types):
        if fit.inference == "mean_field":
            from sv_pgs.mean_field import product_draws

            draws = product_draws(
                prior, fit.hyperparameters[model].coefficients, fit.member_omega[:, model], fit.member_shift[:, model], class_index,
                np.random.default_rng([seed, model]), draw_count, fit.working_bytes,
            )
        else:
            draws = member_draws(
                ties, fit.site_precision[:, model], fit.site_shift[:, model], group_draws[:, model, :], np.random.default_rng([seed, model])
            )
        # Each draw's move of the reduced columns' effects (the members' signed sums), on which the covariate
        # coefficients' conditional means depend through the loading.
        if fit.inference == "mean_field":
            group_deviations = _group_sum(ties, ties.sign[:, None] * (draws - mean[:, model, None]))
        else:
            group_deviations = group_draws[:, model, :] - group_mean[:, model, None]
        models.append(ScoringModel.from_reduced_fit(
            active_rows=np.asarray(statistics.active_rows, dtype=np.int64),
            signed_means=np.asarray(statistics.means, dtype=np.float64),
            signed_scales=np.asarray(statistics.scales, dtype=np.float64),
            tie_map=identity,
            member_prior_variances=prior_second_moment(prior, fit.hyperparameters[model]),
            beta_reduced=mean[:, model],
            posterior_draws_reduced=draws,
            alpha=alpha[:, model],
            trait_type=trait_type,
            predictive_intercept_shift=0.0,
            covariate_draws=alpha[:, model, None] - conditional_loading @ group_deviations,
            covariate_covariance=fit.noise_variance[model] * statistics.covariate_gram_pseudo_inverse,
            # The EP route's draws are its Gaussian's; the mean-field route's are the product's scale mixtures.
            gaussian_posterior=fit.inference != "mean_field",
        ))
    return models
