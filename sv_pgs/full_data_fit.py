"""Stage 2: each model fitted on the full data by EP-EB, and its scoring models.

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
from typing import Callable, Sequence

import numpy as np

from sv_pgs._typing import F64Array, I64Array
from sv_pgs.config import TraitType
from sv_pgs.dual_solve import DualGaussian, _host
from sv_pgs.fast_scoring import ScoringModel
from sv_pgs.genotype_statistics import GenotypeSufficientStatistics
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
    marginal_variances,
    variance_jvp,
    window_working_bytes,
)
from sv_pgs.scale_mixture_ep import (
    Cavity,
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
    the start)."""
    ld = statistics.ld
    # A group's prior variance is its members' sum (tie_members): u_g = sum_j u_j.
    ties = TieGroups.from_tie_map(statistics.tie_map)
    weights = np.zeros(ties.group_count)
    np.add.at(weights, ties.group, np.exp(prior.log_variance_offset))
    covariate_count = statistics.covariate_gram.shape[0]
    fitted = statistics.covariate_target.T @ np.linalg.pinv(statistics.covariate_gram) @ statistics.covariate_target
    target_square = np.diag(statistics.target_gram) - np.diag(fitted)
    score_square = np.zeros(target_square.shape[0])
    column_square = np.zeros(prior.variant_count)
    gram_trace = 0.0
    gram_square = 0.0
    weighted_diagonal = 0.0
    previous_columns = None
    for block_index in range(ld.block_count):
        block = ld.block(block_index)
        columns = np.asarray(block.reduced_columns, dtype=np.int64)
        gram = np.asarray(block.projected_gram, dtype=np.float64)
        score_square += np.sum(np.square(np.asarray(block.projected_score, dtype=np.float64)), axis=0)
        diagonal = np.diag(gram)
        gram_trace += float(diagonal.sum())
        weighted_diagonal += float(weights[columns] @ diagonal)
        squares = np.square(gram)
        column_square[columns] += squares.sum(axis=0)
        gram_square += float(squares.sum())
        cross = ld.adjacent_block(block_index) if block_index else None
        if cross is not None:
            cross_squares = np.square(np.asarray(cross, dtype=np.float64))
            column_square[previous_columns] += cross_squares.sum(axis=1)
            column_square[columns] += cross_squares.sum(axis=0)
            gram_square += 2.0 * float(cross_squares.sum())
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

    - ``remaining_gain``: the last outer check's Newton-B decrement plus its weights' B-evidence gain and remaining
      gain, in nats (at most 1/(2K)); ``newton_decrement`` is its first part, 1/2 g'|B + S|^-1 g;
    - ``smoothing_gradient``: the B-evidence's largest |dV/drho| over interior weights, from its analytic gradient,
      with the curvature's difference steps in ``stationarity_steps`` and the gradient's error bounds in
      ``stationarity_errors``;
    - ``mean_move``: an upper bound on the undamped EP update's squared move of the mean in the posterior metric at
      the final refresh, against ``draw_tolerance`` = p_eff / K; ``noise_gain``: the noise update's evidence gain there;
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
    # Per model: whether the outer loop certified it (``OuterFit.certified``); an uncertified fit is reported, never
    # passed as certified.
    certified: F64Array | None = None


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
            solved = np.asarray(_host(gaussian.posterior_solve(values[:, live], model, bound)), dtype=np.float64)
            lower, _upper = _norm_bounds(np.sum(values[:, live] * solved, axis=0), bound)
            done = bound <= relative_tolerance * lower
            solution[:, live[done]] = solved[:, done]
            bound = np.where(lower > 0.0, relative_tolerance * lower, 0.5 * bound)[~done]
            live = live[~done]
        return solution

    return GaussianPosterior(
        solve=relative_solve, variance_jvp=lambda weights: variance_jvp(solve, grams, weights, gaussian.array_module).values,
        local_response=local_response(solve, grams),
    )


def _member_posterior(reduced: GaussianPosterior, ties: TieGroups, member_precision: F64Array, group_marginals: F64Array) -> GaussianPosterior:
    """q's responses over the tie members from the solver's over the groups (``tie_members``): with w_j = s_j D_j / D_g
    and the within-group conditional covariance C_w = diag(D) - D s s' D / D_g (block diagonal over the tied groups,
    zero for a singleton), Sigma_members = C_w + W Sigma_groups W', so Sigma R = C_w R + W Sigma_groups (W' R), and
    (Sigma o Sigma) V = w^2 o [(Sigma_g o Sigma_g) group_sum(w^2 V)] + (C_w o C_w + 2 C_w o (w w' Sigma_gg)) V within
    each tied group. The block-local preconditioner is the groups' own, carried over where every group is a single
    member (a permutation); a tied model's Krylov solve runs without it."""
    precision = np.asarray(member_precision, dtype=np.float64)
    weight = member_weights(ties, precision)
    tied = tied_groups(ties)
    conditionals = []
    for members in tied:
        shares, _group_precision, _total = tied_weights(precision[members])
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

    def _iterate(self, site_precision: F64Array, site_shift: F64Array) -> None:
        group_precision, group_shift = group_sites(self.ties, site_precision, site_shift)
        certificate = self.gaussian.iterate(
            site_precision=group_precision, site_shift=group_shift, noise_variance=self.noise,
            error_bound=np.sqrt(self.effective / self.draw_count), probe_residual_ratio=self.probe_ratio,
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
            back_products, _coupling, _residual_norm = gaussian.information_solve(probes, model, residual)
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
                covariate_count=int(gaussian.covariates.shape[1]),
                site_precision=self.site_precision[:, model],
                posterior_variance=variances[:, model],
                noise=float(self.noise[model]),
            )
            for model in range(gaussian.model_count)
        ])

    def _targets(self, hyperparameters: Sequence[MixtureHyperparameters], cavities: list[Cavity]) -> tuple[F64Array, F64Array]:
        columns = [site_targets(tilted_moments(self.prior, model, cavity, self.working_bytes), cavity) for model, cavity in zip(hyperparameters, cavities)]
        return np.column_stack([column[0] for column in columns]), np.column_stack([column[1] for column in columns])

    def _move_bounds(self, model: int, right: F64Array, threshold: float) -> float:
        """An upper bound on ||Sigma right||_A^2 = right' Sigma right over the members that decides it against
        ``threshold``: Sigma = C_w + W Sigma_groups W' (``_member_posterior``), so the members' within-group part is
        exact and the groups' part is the solver's, whose bound halves until the two-sided bounds from r'x_hat fall on
        one side."""
        precision = self.site_precision[:, model]
        weight = member_weights(self.ties, precision)
        grouped = np.zeros(self.ties.group_count)
        np.add.at(grouped, self.ties.group, weight * right)
        within = 0.0
        for members in tied_groups(self.ties):
            shares, _group_precision, _total = tied_weights(precision[members])
            variance = 1.0 / precision[members]
            signs = self.ties.sign[members]
            conditional = np.diag(variance) - np.outer(signs * shares, signs * variance)
            within += float(right[members] @ conditional @ right[members])
        remaining = threshold - within
        if remaining < 0.0:
            return within
        bound = np.array([0.5 * np.sqrt(remaining)])
        while True:
            solved = np.asarray(_host(self.gaussian.posterior_solve(grouped[:, None], model, bound)), dtype=np.float64)
            lower, upper = _norm_bounds(np.array([float(grouped @ solved[:, 0])]), bound)
            if upper[0] * upper[0] <= remaining or lower[0] * lower[0] > remaining:
                return within + float(upper[0] * upper[0])
            bound = 0.5 * bound

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
            # The undamped update moves the mean by Sigma (delta nu - delta tau o mu), to first order in the site change.
            right = (target_shift - self.site_shift) - (target_precision - self.site_precision) * mean
            draw_tolerance = self.effective / self.draw_count
            self.mean_move = np.array([self._move_bounds(model, right[:, model], float(draw_tolerance[model])) for model in range(model_count)])
            noise = self._noise(variances)
            covariate_count = int(gaussian.covariates.shape[1])
            self.noise_gain = np.array([
                noise_gain(float(noise[model]), float(self.noise[model]), int(gaussian.training_counts[model]), covariate_count)
                for model in range(model_count)
            ])
            if np.all(self.mean_move <= draw_tolerance) and np.all(self.noise_gain <= tolerance):
                snapshot = self._snapshot()
                return [
                    FixedPoint(
                        cavity=cavities[model],
                        posterior=_member_posterior(
                            _posterior(gaussian, model, grams[model], group_variances[:, model], lambda snapshot=snapshot: self._ensure(snapshot)),
                            self.ties, self.site_precision[:, model], group_variances[:, model],
                        ),
                        mean=mean[:, model].copy(),
                        precision_norm=_precision_norm(gaussian, model, self.site_precision[:, model], self.ties),
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
                    self._iterate(trial_precision, trial_shift)
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
            if np.all(mean_move <= self.effective / self.draw_count):
                return
            ratio = float(np.max(mean_move / previous_move))
            if ratio >= 1.0:
                damping = min(damping, 1.0 / (1.0 + np.sqrt(ratio)))
            previous_move = mean_move
            cavities = [
                Cavity(precision=frozen[:, model], shift=new_mean[:, model] / marginal[:, model] - self.site_shift[:, model]) for model in range(model_count)
            ]
            target_precision, target_shift = self._targets(hyperparameters, cavities)


def fit_full_data(
    *, gaussian: DualGaussian, statistics: GenotypeSufficientStatistics, prior: ScaleMixturePrior, draw_count: int, working_bytes: int, seed: int
) -> FullDataFit:
    """Stage 2 for quantitative models, from the prior (see the module docstring); ``seed`` draws the certificate's
    variant-side probes."""
    # EB starts where each trait's genetic variance fits inside its phenotypic variance (lead ruling): the first mean
    # solve's iterations grow with the prior signal per sample, which a start at the lattice centre puts far past it.
    moments = moment_starts(statistics, prior)
    if len(moments) != gaussian.model_count:
        raise ValueError("Stage 0's targets must be the models, in order")
    starts = [initial_hyperparameters(prior, moment.mean_variance) for moment in moments]
    fixed_points = _FullDataFixedPoints(
        gaussian, statistics, prior, draw_count, working_bytes, seed, starts, np.array([moment.noise for moment in moments])
    )
    try:
        fits = fit_hyperparameters(prior, starts, fixed_points, working_bytes, 0.5 / draw_count)
    except FloatingPointError as error:
        # The oracle's refusals say why EP had no fixed point; they belong with the failure.
        raise FloatingPointError(f"{error}; EP refusals: {fixed_points.refusals}") from error
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
            information_tolerance=np.array([float(np.min(certificate.tolerance)) for certificate in fixed_points.information]),
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
            certified=np.array([fit.certified for fit in fits], dtype=bool),
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
    error_bound = np.sqrt(fit.certificate.effective_effects / draw_count)
    group_draws = np.asarray(_host(gaussian.draws(draw_count=draw_count, error_bound=error_bound, seed=seed)), dtype=np.float64)
    alpha = np.asarray(_host(gaussian.alpha), dtype=np.float64)
    group_mean = np.asarray(_host(gaussian.mean), dtype=np.float64)
    mean, _variance = member_moments(ties, fit.site_precision, fit.site_shift, group_mean, np.zeros_like(group_mean))
    identity = _compact_identity_tie_map(ties.member_count)
    models = []
    for model, trait_type in enumerate(trait_types):
        draws = member_draws(
            ties, fit.site_precision[:, model], fit.site_shift[:, model], group_draws[:, model, :], np.random.default_rng([seed, model])
        )
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
        ))
    return models
