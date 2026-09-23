"""The structured posterior at genome scale: a full-covariance Gaussian background plus local single effects, one
global residual, on the streamed store.

Model (the 2026-09-23 audit's target model, the lead's shared spec; ``structured`` is its dense gene-scale twin)

    y = C gamma + X beta + eps,   eps ~ N(0, sigma^2 I),   beta = u + sum_l b_l.

Everything lives in the projected metric of the training rows: P = (I - H) M with M the training mask and H the
projector onto the masked covariates, Xp = P X (Stage 0's projected Gram is Xp'Xp), y_P = P y, n_eff = n_train -
rank(M C). Members j are Stage 0's active rows; member j's column is s_j x_g(j) with g(j) its tie group (a reduced
column) and s_j its sign (``tie_members``).

- Background: u_j ~ N(0, D_j) independently, log D_j = o_j + e_0 + delta_c(j) + d_j' theta_u (o_j the prior's
  offset, e_0 a free common level, delta_c class deviations under N(0, 1/kappa) with kappa learned, d_j the prior's
  class-centred annotation design under its groups' penalties). A tie group carries D_g = sum_j D_j.
- Local effects: effect l lives on a window W_l of candidate members plus a null state. Exactly one candidate j carries
  a magnitude, whose prior is the learned scale mixture sum_k w_c(j)k N(0, v_jk), v_jk = exp(a_j + t_k): w and a_j are
  ``scale_mixture_ep``'s class densities and log scales of the ScaleMixturePrior ``prior`` at its coefficients x. The
  candidate probabilities are log pi_lj = phi_j - log sum_(i in W_l) exp(phi_i) with phi_j = psi_c(j) + d_j' psi_a
  (class deviations and annotation effects, each under a learned pooling penalty). The null state's probability is
  profiled per effect (structured-small's rule): with B_l = sum_j pi_lj sum_k w_jk BF_jk, the effect is on (null
  probability 0) iff log B_l > 0, else off (probability 1, mean 0, KL 0): an off effect is exactly absent.

Variational family: q = q_u prod_l q_l, with q_u one joint Gaussian of full covariance (never a product over variants)
and each q_l the exact single-effect law given the residual without it. The update of effect l (``_single_effect``)
is the spec's: d_j = ||x_g||^2 / sigma^2, h_j = x_j' r_l / sigma^2, s_jk = 1 / (d_j + 1 / v_jk), mu_jk = s_jk h_j,
log BF_jk = -1/2 log(1 + d_j v_jk) + 1/2 h_j^2 s_jk, alpha_jk by one joint log-sum-exp; m_j = sum_k alpha_jk mu_jk,
t_j = sum_k alpha_jk (s_jk + mu_jk^2), Cov(b_l) = diag(t) - m m'. No argmax anywhere: tied or identical candidates with
identical priors get identical weights whatever their order.

Objective. The ELBO of this family (declared once, ``_elbo``), at a state where q_u is optimal for (D, sigma^2):

    F = -n_eff/2 log(2 pi sigma^2) - (||r||^2 + sum_l V_l) / (2 sigma^2) - 1/2 m_u' D^-1 m_u - 1/2 log det H - sum_l KL_l,

with r = y_P - Xp (m_u + sum_l m_l) the one global residual, V_l = sum_j t_lj ||x_j||^2 - ||Xp m_l||^2 each local
factor's variance (its negative covariance included), and H = I + D^1/2 Xp'Xp D^1/2 / sigma^2. The traces cancel: q_u's
data trace tr(Xp'Xp Sigma_u) / sigma^2 and prior trace tr(D^-1 Sigma_u) sum to p at Sigma_u = A^-1, A = Xp'Xp / sigma^2
+ D^-1, which the KL's -p removes. The fit ascends J = F + the log penalty densities of the hyperparameters at fixed
smoothing weights; the weights move between iterations by their Laplace evidence (below), which changes J's
definition, so each iteration records its ascent at the weights it started from (``History.fixed_smoothing_gain``).

Background solve. The whitened system H z = b is solved in its dual form (``dual_solve``: S = I + Xt D Xt',
Xt = (I - H) W^1/2 X, W = M / sigma^2, n x n, never formed): S zeta = b with b = (r + Xp m_u) / sigma (the working
response without the background), m_u = D Xt' zeta. For any iterate with dual residual rho = b - S zeta_hat,
||m_hat - m||_A^2 = rho'(I - S^-1) rho <= ||rho||^2 (``dual_solve``'s identity). ||m_hat - m||_A is the whitened
error's H-norm, so this is the spec's certificate e'He <= ||rho||^2 in the dual's residual; rho is recomputed exactly
at the end of every solve, and the mean is solved to ||rho|| <= K^-1/2 (K = ``draw_count``), costing at most 1/(2K)
nats. The operator is ``dual_solve.apply_operator`` (one read of every tile per product, exact int8-split products).

Traces and the log determinant (``_krylov``). K Rademacher probes z on the training rows ride the mean's conjugate
gradient, each column its own Lanczos process. Their solutions w = S^-1 z give
- gamma_u = tr(Xp'Xp Sigma_u) / sigma^2 = tr(I - S^-1) = n_train - E[z'w] (the background's effective count, the
  noise M-step's trace), and
- per group, D_g (Xt'w)_g (Xt'z)_g, unbiased for D_g (Xt'S^-1 Xt)_gg = 1 - Sigma_gg / D_g (Bekas, Kokiopoulou and
  Saad 2007), so Sigma_gg and E[u_j^2] enter the M-step unbiased and unclipped (a negative sample value stays).
  Rademacher probes estimate these small quantities with an error set by S^-1's off-diagonal; perturbation draws
  would square Sigma_gg itself (their gamma_u error ~ sqrt(2 p / K), 120 at 466k groups and K = 64).
- log det S = E[z' log(S) z], each probe's quadrature bracketed by Gauss and Gauss-Radau (node at 1: S >= I) from its
  own Lanczos coefficients (Golub and Meurant 2010, ch. 6), to a width of 1/K per probe.
The probe solves stop where the trace's solve error, at most ||z|| ||rho_z|| per probe (S^-1 <= I), stays within
sqrt(2 n_eff / K): an error delta in gamma_u moves the noise M-step's sigma^2 by sigma^2 delta / n_eff and costs
delta^2 / (4 n_eff) nats, at most 1/(2K). Every estimate carries its probes' standard error (``BackgroundMoments``).

Draws by perturbation (``background_draws``): a = e2 - Xt D^1/2 e1 per draw (e1 ~ N(0, I_p) regenerated per block
from its seed, e2 ~ N(0, I_n)), S zeta_a = a, and u* - m_u = D^1/2 e1 + D Xt' zeta_a has covariance A^-1 exactly
(Matheron; ``dual_solve.draw_right_hand_side``). Each block's draws go to the caller as they form: nothing p x K.

Local effects at scale. Windows of candidates come from Stage 0's LD blocks (``candidate_windows``): each pair of
consecutive blocks of a chromosome is a window's core, and a chromosome's first and last blocks are cores of their own,
so every member lies in two cores and every two adjacent blocks share one. The product over effects is not symmetric
between two proxies that different effects can reach unequally: an effect whose set holds one and not the other takes
it whole, as mean field does. So each window carries a halo, the proxies (``proxy_threshold``, from Stage 0's
adjacent Grams: ``stage0_proxies``) its core has in the blocks beside it, and two proxies across a boundary then lie
in exactly the same windows: every effect that can take one can take the other, with equal weight where their priors
are equal (Stage 0 records no LD beyond adjacent blocks). Exact tie groups are one reduced column, so they are always
whole. The global residual couples every window: a window's sweep starts from
c = Xw' r read exactly from its tiles, updates its effects in sequence (each from the residual left by the ones
before, never several from one stale residual), and leaves r moved by exactly what its effects moved. Per effect
update the design is read twice (Xw Delta and Xw'(P Xw Delta)), exact to float64 (``code_products``). A window's
effects grow one at a time until a new one stays off (its profiled null wins): the number of effects is the data's,
and an off effect is absent, so a surplus effect costs nothing. The Bayes factors of a window's candidates over the
lattice are one vectorized device evaluation (CuPy where the source is on a device).

Prior learning at fixed q (concave M-steps with joint moments, streamed as sufficient statistics):
- local magnitudes x: sum_ck N_ck log w_ck - 1/2 sum_j [W_j a_j + A_j e^-a_j] with N_ck = sum_l sum_(j in c)
  alpha_ljk, W_j = sum_l alpha_lj and A_j = sum_l sum_k alpha_ljk (s_ljk + mu_ljk^2) e^-t_k, the joint moment
  E[1_j beta_j^2 e^-t];
- candidate logits: sum_j W_j phi_j - sum_w E_w log sum_(i in w) e^phi_i (E_w the window's live effects);
- background: -1/2 sum_j [a_j + E(u_j^2) e^-a_j] with E(u_j^2) = (D_j / D_g)^2 (m_g^2 + Sigma_gg) + D_j - D_j^2 / D_g
  (the member given its group's sum), preceded by the parameter-expanded step (Liu, Rubin and Wu 1998; Qi and
  Jaakkola 2006 for the variational form) that rescales u and D jointly by the least-squares alpha of the working
  response on Xp m_u, which leaves KL(q_u || p_u) unchanged and raises the likelihood term;
- sigma^2 = (||r||^2 + sigma^2 gamma_u + sum_l V_l) / n_eff, the local factors' negative covariance inside V_l.
Each is maximized by Newton with the penalty (the pinv step on its concave total, halved until it rises). The
smoothing weights lambda_b take one MacKay step per iteration, lambda_b <- lambda_b tr(S^+ Pi_b) /
(||R_b x||^2 + tr((-H)^-1 Pi_b)) (Pi_b = R_b'R_b, S = sum lambda_b Pi_b; rank(R_b) for a block of its own), the
fixed point of the Laplace evidence V = Q(x) - 1/2 x'Sx + 1/2 log|S|_+ - 1/2 log|-H| + 1/2 log|N'(-H)N| with its
normalizer and the unpenalized directions N profiled; a weight's lambda = infinity edge is exact, and a block moves
between the edge and the interior only where V rises (``_Hyper``).

Memory contract (one model; p members, p_g groups, n samples, K probes, B blocks of at most b groups). Persistent:
member float64 vectors (offsets, W_j, A_j, E(u_j^2), the returned mean: 5 x 8p bytes), group vectors (D_g, m_g, the
diagonal estimate, the unit squares: 4 x 8 p_g), n vectors (r, y_P, Xp m_u, the mask: 4 x 8n), the probes and the
conjugate gradient's workspace (5 x 8 n (K + 1)), and the live effects' states (per effect its window's member mean
and group products, 8 (|W_members| + |W_groups|) bytes). Leased per window: two blocks' tiles (the source's) and the
window's |W_members| x (lattice) arrays of one effect update. Nothing is p x K or n x n. At p = 518k (p_g = 466k),
n = 40k, K = 64: 21 MB of member vectors, 15 MB of group vectors, 1.3 MB of n vectors and 105 MB of Krylov workspace;
at p = 10M, n = 500k: 400 MB, 320 MB, 16 MB and 1.3 GB. Each live effect on a window of two 28k-column blocks holds
0.9 MB (effects scale with the number of on effects, not with p x K).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from sv_pgs._typing import F64Array, I64Array
from sv_pgs.dual_solve import DualModels, DualTileSource, PassCount, _host, apply_operator, column_squares
from sv_pgs.progress import log
from sv_pgs.scale_mixture_ep import MixtureHyperparameters, ScaleMixturePrior, class_log_density, initial_hyperparameters, log_scale
from sv_pgs.tie_members import TieGroups

_EPSILON = float(np.finfo(np.float64).eps)
_LOG_TWO_PI = math.log(2.0 * math.pi)


# ------------------------------------------------------------------ candidate windows


@dataclass(frozen=True)
class Window:
    """One candidate set: its core blocks (consecutive, one chromosome) and its halo, the proxies its core has in the
    neighbouring blocks (``candidate_windows``). ``groups`` lists its groups, the core's range [core_start, core_stop)
    first and then the halo's by source block (``halo``: block -> its groups there, ascending); ``members`` are the
    members on them, ascending, and ``local_group`` each member's position in ``groups``."""

    blocks: tuple[int, ...]
    core_start: int
    core_stop: int
    halo: tuple[tuple[int, I64Array], ...]
    groups: I64Array
    members: I64Array
    local_group: I64Array

    @property
    def group_count(self) -> int:
        return int(self.groups.shape[0])

    @property
    def ready(self) -> int:
        """The block whose arrival completes the window: its last core block, or a later halo source."""
        return max([self.blocks[-1]] + [block for block, _groups in self.halo])


def proxy_threshold(draw_count: int) -> float:
    """1 - r^2 at or below which two columns are proxies for the candidate sets: an effect at one standard error
    (chi^2 = 1, about the least evidence an on effect carries) moves the log Bayes factors of two columns with
    correlation r apart by about chi^2 (1 - r^2) / 2, so at 1 - r^2 <= 1/K no such effect tells them apart by more than
    the fit's tolerance 1/(2K)."""
    return 1.0 / draw_count


def stage0_proxies(statistics: Any, threshold: float, working_bytes: int) -> I64Array:
    """The cross-boundary proxy pairs (groups i in block b - 1, j in block b of one chromosome) with 1 - r^2 at most
    ``threshold`` plus the float32 rounding of Stage 0's adjacent Grams (a pair the rounding could hide is kept), read
    from its projected adjacent Grams in row chunks whose float64 copies fit ``working_bytes``."""
    ld = statistics.ld
    diagonal = np.asarray(ld.ld_diagonal(), dtype=np.float64) * float(ld.sample_count)
    # A stored entry is within one float32 rounding of its exact value relative to its size, so r^2 within two.
    limit = threshold + 2.0 * float(np.finfo(np.float32).eps)
    found = []
    for block in range(1, ld.block_count):
        cross = ld.adjacent_block(block)
        if cross is None or cross.size == 0:
            continue
        rows = np.asarray(ld.block(block - 1).reduced_columns, dtype=np.int64)
        columns = np.asarray(ld.block(block).reduced_columns, dtype=np.int64)
        column_diagonal = diagonal[columns]
        # Three float64 arrays of a chunk live at once: its values, their squares and the comparison.
        chunk = max(1, int(working_bytes) // (3 * np.dtype(np.float64).itemsize * columns.shape[0]))
        for first in range(0, rows.shape[0], chunk):
            values = np.asarray(cross[first:first + chunk], dtype=np.float64)
            squared = values * values / (diagonal[rows[first:first + chunk], None] * column_diagonal[None, :])
            left, right = np.nonzero(1.0 - squared <= limit)
            if left.size:
                found.append(np.column_stack([rows[first + left], columns[right]]))
    return np.concatenate(found).astype(np.int64) if found else np.zeros((0, 2), dtype=np.int64)


def candidate_windows(
    block_bounds: Sequence[tuple[int, int]], chromosomes: Sequence[str], ties: TieGroups, proxies: I64Array | None = None,
) -> tuple[Window, ...]:
    """Per chromosome with blocks b_0..b_k: {b_0}, {b_0, b_1}, ..., {b_(k-1), b_k}, {b_k}, in that order, so every
    member lies in two cores and every two adjacent blocks share one. ``proxies`` (pairs of groups (i, j), i in a block
    and j in the next: ``stage0_proxies``) are the cross-boundary proxies: each window takes as its halo the proxies its
    core's groups have outside it, so two proxies across a boundary lie in exactly the same windows (the one whose core
    holds both, and the two beside it through their halos): no effect can take one without the other. A chromosome of
    one block is a single window."""
    if len(block_bounds) != len(chromosomes):
        raise ValueError("one chromosome per block")
    order = np.argsort(ties.group, kind="stable")
    offsets = np.searchsorted(ties.group[order], np.arange(ties.group_count + 1))
    starts = np.asarray([start for start, _stop in block_bounds], dtype=np.int64)
    pairs = np.zeros((0, 2), dtype=np.int64) if proxies is None else np.asarray(proxies, dtype=np.int64).reshape(-1, 2)
    runs: list[list[int]] = []
    for block, chromosome in enumerate(chromosomes):
        if runs and chromosomes[runs[-1][-1]] == chromosome:
            runs[-1].append(block)
        else:
            runs.append([block])
    sets: list[tuple[int, ...]] = []
    for run in runs:
        if len(run) == 1:
            sets.append((run[0],))
            continue
        sets.append((run[0],))
        sets.extend((first, second) for first, second in zip(run[:-1], run[1:]))
        sets.append((run[-1],))
    windows = []
    for blocks in sets:
        core_start, core_stop = int(block_bounds[blocks[0]][0]), int(block_bounds[blocks[-1]][1])
        inside = (pairs >= core_start) & (pairs < core_stop)
        halo_groups = np.unique(np.concatenate([pairs[inside[:, 0] & ~inside[:, 1], 1], pairs[inside[:, 1] & ~inside[:, 0], 0]]))
        halo_blocks = np.searchsorted(starts, halo_groups, side="right") - 1
        halo = tuple((int(block), halo_groups[halo_blocks == block]) for block in np.unique(halo_blocks))
        groups = np.concatenate([np.arange(core_start, core_stop, dtype=np.int64)] + [values for _block, values in halo])
        position = np.full(ties.group_count, -1, dtype=np.int64)
        position[groups] = np.arange(groups.shape[0])
        pieces = [order[offsets[core_start]:offsets[core_stop]]] + [order[offsets[group]:offsets[group + 1]] for _block, values in halo for group in values]
        members = np.sort(np.concatenate(pieces)).astype(np.int64)
        windows.append(Window(blocks, core_start, core_stop, halo, groups, members, position[ties.group[members]]))
    return tuple(windows)


def _held(tile: Any) -> Any:
    """A tile that stays valid after its read moves on: a code tile over its own copy of the codes (a streamed read
    reuses its buffer for the block after next), any other tile (a dense source's view) as it is."""
    copy = getattr(tile, "held_copy", None)
    return tile if copy is None else copy()


def _segment_sum(xp: Any, index: Any, values: Any, size: int) -> Any:
    total = xp.zeros((size,) + tuple(values.shape[1:]))
    if xp is np:
        np.add.at(total, index, values)
    else:
        import cupyx  # noqa: PLC0415 - only on a device

        cupyx.scatter_add(total, index, values)
    return total


def _log_sum_exp(xp: Any, values: Any) -> float:
    peak = float(xp.max(values))
    if not math.isfinite(peak):
        return peak
    return peak + math.log(float(xp.sum(xp.exp(values - peak))))


# ------------------------------------------------------------------ the single effect


@dataclass
class _EffectUpdate:
    """One effect's exact update on a window: ``on`` its profiled null (log B > 0), its member means m and second
    moments t, KL(q_l || p_l), log B, and (on) its sufficient statistics: alpha summed over the lattice (W), the
    joint moment sum_k alpha (s + mu^2) e^-t (A) and the class-by-node masses N."""

    on: bool
    mean: Any
    second: Any
    divergence: float
    log_evidence: float
    weight: Any = None
    moment: Any = None
    masses: Any = None


def _single_effect(
    xp: Any, shift: Any, precision: Any, log_candidate: Any, log_mixture: Any, log_variance: Any, class_indicator: Any, inverse_nodes: Any,
) -> _EffectUpdate:
    """The spec's single-effect update for member shifts h (``shift``), d = ||x||^2 / sigma^2 (``precision``), log pi_j
    over the window (``log_candidate``), log w_c(j)k (``log_mixture``, members x nodes) and log v_jk
    (``log_variance``). ``class_indicator`` (classes x members) and ``inverse_nodes`` (e^-t_k) form the statistics.

    With B = sum_jk pi_j w_jk BF_jk, the null probability pi_0 profiled jointly with q_l maximizes
    log(pi_0 + (1 - pi_0) B): 1 (off) where B <= 1, else 0 (on), and then alpha_jk = pi_j w_jk BF_jk / B. The
    effect's KL is sum_jk alpha_jk [log BF_jk - log B + KL(N(mu, s) || N(0, v))] (log alpha - log pi - log w =
    log BF - log B), zero for the off state."""
    variance = xp.exp(log_variance)
    ratio = precision[:, None] * variance
    spread = variance / (1.0 + ratio)
    mean = spread * shift[:, None]
    log_factor = 0.5 * shift[:, None] * mean - 0.5 * xp.log1p(ratio)
    logits = log_candidate[:, None] + log_mixture + log_factor
    log_evidence = _log_sum_exp(xp, logits)
    if not log_evidence > 0.0:
        zeros = xp.zeros(shift.shape[0])
        return _EffectUpdate(False, zeros, zeros, 0.0, log_evidence)
    alpha = xp.exp(logits - log_evidence)
    second_node = spread + mean * mean
    # KL(N(mu, s) || N(0, v)) = 1/2 [(s + mu^2) / v - 1 + log(v / s)], log(v / s) = log1p(d v).
    divergence = float(xp.sum(alpha * (log_factor - log_evidence + 0.5 * (second_node / variance - 1.0 + xp.log1p(ratio)))))
    return _EffectUpdate(
        True, xp.sum(alpha * mean, axis=1), xp.sum(alpha * second_node, axis=1), divergence, log_evidence,
        weight=xp.sum(alpha, axis=1), moment=(alpha * second_node) @ inverse_nodes, masses=class_indicator @ alpha,
    )


@dataclass
class _Effect:
    """A live effect's state between sweeps (host float64): its window's member means and the group products
    Xw' Xp m_l, with its V_l and KL_l under the prior of its last update."""

    mean: F64Array
    products: F64Array
    spread: float
    divergence: float
    log_evidence: float


# ------------------------------------------------------------------ Krylov: conjugate gradient with Lanczos quadrature


def _quadrature_bracket(steps: Sequence[float], ratios: Sequence[float], square: float) -> tuple[float, float]:
    """(Gauss, Gauss-Radau) values of z' log(S) z from a CG's step lengths alpha_i and residual ratios beta_i
    (``square`` = ||z||^2), the Radau rule with its prescribed node at 1 (S >= I). The Lanczos matrix is
    T_ii = 1/alpha_i + beta_(i-1)/alpha_(i-1), T_(i,i+1) = sqrt(beta_i)/alpha_i; Radau appends the row that puts an
    eigenvalue at 1: its diagonal 1 + delta_k with (T - I) delta = T_(k,k+1)^2 e_k (Golub and Meurant 2010, 6.2).
    log's even derivatives are negative and its odd ones positive on (0, inf), so the two rules bracket the value."""
    count = len(steps)
    alpha = np.asarray(steps, dtype=np.float64)
    beta = np.asarray(ratios, dtype=np.float64)
    diagonal = 1.0 / alpha
    diagonal[1:] += beta[: count - 1] / alpha[: count - 1]
    off = np.sqrt(beta[: count - 1]) / alpha[: count - 1]
    tridiagonal = np.diag(diagonal) + np.diag(off, 1) + np.diag(off, -1)
    values, vectors = np.linalg.eigh(tridiagonal)
    gauss = square * float(np.sum(vectors[0] ** 2 * np.log(values)))
    last = math.sqrt(beta[count - 1]) / alpha[count - 1]
    if last == 0.0:
        return gauss, gauss
    unit = np.zeros(count)
    unit[-1] = last * last
    shifted = tridiagonal - np.eye(count)
    delta = np.linalg.solve(shifted, unit)
    extended = np.zeros((count + 1, count + 1))
    extended[:count, :count] = tridiagonal
    extended[count - 1, count] = extended[count, count - 1] = last
    extended[count, count] = 1.0 + delta[-1]
    values, vectors = np.linalg.eigh(extended)
    radau = square * float(np.sum(vectors[0] ** 2 * np.log(values)))
    return gauss, radau


@dataclass
class _KrylovResult:
    solution: Any
    residual: Any
    residual_norm: F64Array
    residual_bound: F64Array
    brackets: F64Array
    iterations: int


def _krylov(
    xp: Any, apply: Callable[[Any], Any], right: Any, start: Any, residual_bound: F64Array, quadrature: np.ndarray, width_bound: float,
) -> _KrylovResult:
    """Conjugate gradient on S x = right, each column its own Lanczos process, batched into one product per iteration.
    A column is done when its recursive residual meets ``residual_bound`` and, for ``quadrature`` columns (zero
    start), its Gauss / Gauss-Radau bracket of z' log(S) z is at most ``width_bound`` wide. The exact residual is then
    recomputed (the certificate), and a column that misses its bound restarts from its iterate (a restart keeps the
    bracket it has). A column whose exact residual did not fall across a restart has reached float64's floor for it:
    that residual is the bound it meets (``residual_bound`` returned), as ``certified_block_cg`` rules."""
    columns = int(right.shape[1])
    solution = start.copy()
    requested = np.asarray(residual_bound, dtype=np.float64).copy()
    brackets = np.full((columns, 2), np.nan)
    warm = np.flatnonzero(_host(xp.any(start != 0.0, axis=0)))
    residual = right.copy()
    if warm.size:
        residual[:, xp.asarray(warm)] -= apply(solution[:, xp.asarray(warm)])
    iterations = 0
    need_bracket = quadrature.copy()
    previous: F64Array | None = None
    while True:
        norms = np.sqrt(_host(xp.sum(residual * residual, axis=0)))
        if not np.all(np.isfinite(norms)):
            raise FloatingPointError("a residual is not finite: the right-hand side or the operator holds NaN or inf")
        open_ = (norms > requested) | need_bracket
        if previous is not None:
            stalled = open_ & ~need_bracket & (norms >= previous)
            requested[stalled] = norms[stalled]
            open_ &= ~stalled
        if not open_.any():
            return _KrylovResult(solution, residual, norms, requested, brackets, iterations)
        previous = norms
        live = np.flatnonzero(open_)
        direction = residual[:, xp.asarray(live)].copy()
        squares = norms[live] ** 2
        steps: list[list[float]] = [[] for _ in live]
        ratios: list[list[float]] = [[] for _ in live]
        active = np.ones(live.size, dtype=bool)
        while active.any():
            positions = np.flatnonzero(active)
            device_positions = xp.asarray(positions)
            chosen = xp.asarray(live[positions])
            image = apply(direction[:, device_positions])
            curvature = _host(xp.sum(direction[:, device_positions] * image, axis=0))
            if not np.all(curvature > 0.0):
                raise FloatingPointError("the operator is not positive definite on a search direction")
            step = squares[positions] / curvature
            device_step = xp.asarray(step)
            solution[:, chosen] += device_step[None, :] * direction[:, device_positions]
            residual[:, chosen] -= device_step[None, :] * image
            updated = _host(xp.sum(residual[:, chosen] * residual[:, chosen], axis=0))
            ratio = updated / squares[positions]
            iterations += 1
            direction[:, device_positions] = residual[:, chosen] + xp.asarray(ratio)[None, :] * direction[:, device_positions]
            for index, position in enumerate(positions):
                steps[position].append(float(step[index]))
                ratios[position].append(float(ratio[index]))
                column = int(live[position])
                done = math.sqrt(updated[index]) <= requested[column] or updated[index] == 0.0
                if need_bracket[column]:
                    gauss, radau = _quadrature_bracket(steps[position], ratios[position], float(norms[column]) ** 2)
                    brackets[column] = (min(gauss, radau), max(gauss, radau))
                    if brackets[column, 1] - brackets[column, 0] <= width_bound or updated[index] == 0.0:
                        need_bracket[column] = False
                    else:
                        done = False
                if done:
                    active[position] = False
            squares[positions] = updated
        residual = right - apply(solution)


# ------------------------------------------------------------------ penalized hyperparameters


@dataclass
class _Hyper:
    """A concave penalized M-step's parameters: coordinates x, penalty blocks (coordinates, factor R) with log weights
    (+inf: the block's directions are zero, the exact edge), and ``objective(x) -> (Q, gradient, -Hessian)``."""

    coefficients: F64Array
    blocks: tuple[tuple[I64Array, F64Array], ...]
    log_smoothing: F64Array

    @property
    def size(self) -> int:
        return int(self.coefficients.shape[0])

    def penalty(self, log_smoothing: F64Array | None = None) -> F64Array:
        weights = self.log_smoothing if log_smoothing is None else log_smoothing
        total = np.zeros((self.size, self.size))
        for (coordinates, factor), weight in zip(self.blocks, weights):
            if math.isfinite(weight):
                total[np.ix_(coordinates, coordinates)] += math.exp(weight) * (factor.T @ factor)
        return total

    def allowed(self, log_smoothing: F64Array | None = None) -> F64Array:
        """An orthonormal basis of the directions no block at its infinite edge forbids."""
        weights = self.log_smoothing if log_smoothing is None else log_smoothing
        rows = []
        for (coordinates, factor), weight in zip(self.blocks, weights):
            if not math.isfinite(weight):
                embedded = np.zeros((factor.shape[0], self.size))
                embedded[:, coordinates] = factor
                rows.append(embedded)
        if not rows:
            return np.eye(self.size)
        stacked = np.vstack(rows)
        _left, singular, right = np.linalg.svd(stacked, full_matrices=True)
        rank = int(np.sum(singular > max(stacked.shape) * _EPSILON * (singular[0] if singular.size else 0.0)))
        return right[rank:].T

    def log_prior(self, coefficients: F64Array | None = None, log_smoothing: F64Array | None = None) -> float:
        """log N(x; 0, S^+) on S's range (the proper part; the flat directions carry no density); minus infinity where a block at its
        infinite edge does not hold its directions at zero (a weight just moved to the edge, before the M-step projects)."""
        values = self.coefficients if coefficients is None else coefficients
        weights = self.log_smoothing if log_smoothing is None else log_smoothing
        for (coordinates, factor), weight in zip(self.blocks, weights):
            if not math.isfinite(weight) and float(np.linalg.norm(factor @ values[coordinates])) > factor.size * _EPSILON * float(np.linalg.norm(factor)) * float(np.linalg.norm(values)):
                return -np.inf
        penalty = self.penalty(log_smoothing)
        basis = self.allowed(log_smoothing)
        restricted = basis.T @ penalty @ basis
        eigenvalues = np.linalg.eigvalsh(0.5 * (restricted + restricted.T)) if restricted.size else np.zeros(0)
        kept = eigenvalues[eigenvalues > restricted.shape[0] * _EPSILON * (eigenvalues.max() if eigenvalues.size else 0.0)]
        return float(-0.5 * values @ penalty @ values + 0.5 * np.sum(np.log(kept)) - 0.5 * kept.size * _LOG_TWO_PI)


def _pinv_symmetric(matrix: F64Array) -> F64Array:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    largest = float(np.max(np.abs(values))) if values.size else 0.0
    kept = values > matrix.shape[0] * _EPSILON * largest
    return (vectors[:, kept] / values[kept]) @ vectors[:, kept].T


def _maximize(hyper: _Hyper, objective: Callable, log_smoothing: F64Array, tolerance: float) -> tuple[F64Array, float]:
    """The penalized maximum at fixed weights, from the current coefficients projected on the allowed directions:
    Newton on the concave total (its pseudo-inverse step: a direction with no curvature has no gradient either),
    each step halved until the total does not fall, until the Newton decrement is at most ``tolerance``."""
    basis = hyper.allowed(log_smoothing)
    penalty = hyper.penalty(log_smoothing)
    current = basis @ (basis.T @ hyper.coefficients)
    value, gradient, negative = objective(current)
    total = value - 0.5 * current @ penalty @ current
    while True:
        restricted_gradient = basis.T @ (gradient - penalty @ current)
        curvature = basis.T @ (negative + penalty) @ basis
        step = _pinv_symmetric(curvature) @ restricted_gradient
        decrement = 0.5 * float(restricted_gradient @ step)
        if not decrement > tolerance:
            return current, total
        scale = 1.0
        while True:
            trial = current + scale * (basis @ step)
            if np.array_equal(trial, current):
                return current, total
            trial_value, trial_gradient, trial_negative = objective(trial)
            trial_total = trial_value - 0.5 * trial @ penalty @ trial
            if trial_total >= total:
                current, value, gradient, negative, total = trial, trial_value, trial_gradient, trial_negative, trial_total
                break
            scale *= 0.5


def _evidence(hyper: _Hyper, objective: Callable, coefficients: F64Array, log_smoothing: F64Array) -> tuple[float, F64Array, F64Array, F64Array]:
    """The Laplace evidence V at the penalized maximum ``coefficients`` (module docstring), and the allowed basis, the
    restricted penalty and the posterior covariance pinv(-H) on the allowed directions."""
    basis = hyper.allowed(log_smoothing)
    penalty = basis.T @ hyper.penalty(log_smoothing) @ basis
    value, _gradient, negative = objective(coefficients)
    curvature = basis.T @ negative @ basis + penalty
    values, vectors = np.linalg.eigh(0.5 * (penalty + penalty.T)) if penalty.size else (np.zeros(0), np.zeros((0, 0)))
    largest = float(values.max()) if values.size else 0.0
    ranged = values > penalty.shape[0] * _EPSILON * largest
    evidence = value - 0.5 * float(coefficients @ hyper.penalty(log_smoothing) @ coefficients) + 0.5 * float(np.sum(np.log(values[ranged])))
    if ranged.any():
        range_basis, null_basis = vectors[:, ranged], vectors[:, ~ranged]
        schur = range_basis.T @ curvature @ range_basis
        if null_basis.shape[1]:
            coupling = range_basis.T @ curvature @ null_basis
            schur = schur - coupling @ _pinv_symmetric(null_basis.T @ curvature @ null_basis) @ coupling.T
        schur_values = np.linalg.eigvalsh(0.5 * (schur + schur.T))
        evidence -= 0.5 * float(np.sum(np.log(schur_values))) if np.all(schur_values > 0.0) else np.inf
    return evidence, basis, penalty, _pinv_symmetric(curvature)


def _mackay(hyper: _Hyper, objective: Callable, tolerance: float) -> tuple[F64Array, float]:
    """One smoothing step at the current coefficients: each finite weight by the MacKay fixed-point map of the Laplace
    evidence, then each block moved across its infinite edge where V rises by more than ``tolerance`` (a released
    block starts at rank(R_b) / tr(R_b (-H_data)^+ R_b'), the map's value from no penalty). Returns the new log
    weights and V's rise; the coefficients stay (the next M-step re-maximizes)."""
    weights = hyper.log_smoothing.copy()
    current, _total = _maximize(hyper, objective, weights, tolerance)
    base, basis, restricted, covariance = _evidence(hyper, objective, current, weights)
    start = base
    restricted_inverse = _pinv_symmetric(restricted)
    updated = weights.copy()
    for index, ((coordinates, factor), weight) in enumerate(zip(hyper.blocks, weights)):
        if not math.isfinite(weight):
            continue
        embedded = np.zeros((factor.shape[0], hyper.size))
        embedded[:, coordinates] = factor
        projected = embedded @ basis
        block = projected.T @ projected
        effective = math.exp(weight) * float(np.sum(restricted_inverse * block))
        denominator = float(np.sum(np.square(embedded @ current))) + float(np.sum(covariance * block))
        if effective > 0.0 and denominator > 0.0:
            updated[index] = math.log(effective / denominator)
    trial_x, _ = _maximize(hyper, objective, updated, tolerance)
    trial_v = _evidence(hyper, objective, trial_x, updated)[0]
    if trial_v >= base:
        weights, base = updated, trial_v
    for index, ((coordinates, factor), weight) in enumerate(zip(hyper.blocks, weights)):
        flipped = weights.copy()
        if math.isfinite(weight):
            flipped[index] = np.inf
        else:
            # The allowed directions with this block released (any finite weight frees them).
            basis = hyper.allowed(np.where(np.arange(weights.size) == index, 0.0, weights))
            data_curvature = basis.T @ objective(current)[2] @ basis
            embedded = np.zeros((factor.shape[0], hyper.size))
            embedded[:, coordinates] = factor
            projected = embedded @ basis
            spread = float(np.sum(_pinv_symmetric(data_curvature) * (projected.T @ projected)))
            if not spread > 0.0:
                continue
            flipped[index] = math.log(np.linalg.matrix_rank(factor) / spread)
        trial_x, _ = _maximize(hyper, objective, flipped, tolerance)
        trial_v = _evidence(hyper, objective, trial_x, flipped)[0]
        if trial_v > base + tolerance:
            weights, base = flipped, trial_v
    return weights, base - start


# ------------------------------------------------------------------ the fit


@dataclass(frozen=True)
class BackgroundMoments:
    """The background's probe estimates at the last E-step, each with its probes' standard error: gamma_u (its
    effective count) and log det S (the quadrature midpoints' mean; ``log_determinant_width`` is the mean bracket
    width, a deterministic bound on the quadrature's part), and the mean's certificate ||m_hat - m||_A."""

    effective_count: float
    effective_count_error: float
    log_determinant: float
    log_determinant_error: float
    log_determinant_width: float
    mean_certificate: float
    probe_residual: float
    iterations: int


@dataclass(frozen=True)
class History:
    """One outer iteration: J at the weights it started from before and after (``fixed_smoothing_gain``, the ascent,
    whose probe-estimate part carries ``gain_error``), the ELBO F, sigma^2, gamma_u, the live effects and their windows'
    growth, the PX scale, passes over the store and seconds."""

    objective: float
    fixed_smoothing_gain: float
    gain_error: float
    evidence_rise: float
    elbo: float
    noise: float
    effective_count: float
    live_effects: int
    expansion: float
    passes: int
    seconds: float


@dataclass(frozen=True)
class StructuredFit:
    """The returned state: every diagnostic below is of this state (``elbo`` is F at it, ``objective`` J).

    ``member_mean`` is E[beta_j] = E[u_j] + sum_l m_lj over the active rows (standardized columns, member order);
    ``inclusion`` W_j = sum_l alpha_lj; ``background_group_mean`` m_g and ``background_variance`` D (groups);
    ``covariate_coefficients`` (C'MC)^+ C'M (y - X E[beta])."""

    member_mean: F64Array
    inclusion: F64Array
    background_group_mean: F64Array
    background_variance: F64Array
    member_background_variance: F64Array
    covariate_coefficients: F64Array
    noise: float
    local_hyperparameters: MixtureHyperparameters
    background_coefficients: F64Array
    background_log_smoothing: F64Array
    candidate_coefficients: F64Array
    candidate_log_smoothing: F64Array
    elbo: float
    objective: float
    moments: BackgroundMoments
    live_effects: int
    effects_per_window: I64Array
    history: tuple[History, ...]
    passes: int
    converged: bool


class StructuredFull:
    """The fit's state and its steps (module docstring). One model: ``mask`` (n,) its training indicator, ``targets``
    (n,), ``covariates`` (n, k); ``source`` the design over the reduced columns in Stage 0's blocks, ``chromosomes``
    each block's; ``ties`` the members; ``prior`` the local magnitudes' ScaleMixturePrior over the members."""

    def __init__(
        self, *, source: DualTileSource, chromosomes: Sequence[str], ties: TieGroups, prior: ScaleMixturePrior, mask: Any, targets: Any,
        covariates: Any, noise: float, background_variance: float, draw_count: int, seed: int, proxies: I64Array | None = None,
    ) -> None:
        xp = source.array_module
        self.xp = xp
        self.source = source
        self.ties = ties
        self.prior = prior
        if prior.variant_count != ties.member_count:
            raise ValueError("the prior must be over the members, in their order")
        self.draw_count = int(draw_count)
        self.tolerance = 0.5 / self.draw_count
        self.count = PassCount()
        self.mask = xp.asarray(mask, dtype=xp.float64).reshape(-1)
        self.covariates = xp.asarray(covariates, dtype=xp.float64)
        self.targets = xp.where(self.mask == 0.0, 0.0, xp.asarray(targets, dtype=xp.float64).reshape(-1))
        self.groups = int(source.variant_count)
        self.unit = DualModels(self.mask[:, None], xp.zeros((self.groups, 1)), self.covariates, xp)
        self.single = xp.zeros(1, dtype=xp.int64)
        rank = int(np.count_nonzero(np.any(_host(self.unit.covariate_factor[0]) != 0.0, axis=0)))
        self.training_count = float(_host(self.mask.sum()))
        self.residual_dimension = self.training_count - rank
        if not self.residual_dimension > 0.0:
            raise ValueError("the training rows do not outnumber the covariates' rank")
        self.squares = np.asarray(_host(column_squares(source, self.unit, self.count)), dtype=np.float64)[:, 0]
        self.windows = candidate_windows(source.block_bounds, chromosomes, ties, proxies)
        self.effects: list[list[_Effect]] = [[] for _ in self.windows]
        self.sign = np.asarray(ties.sign, dtype=np.float64)
        self.class_index = np.asarray(prior.class_index, dtype=np.int64)
        self.class_count = int(prior.class_count)
        members = ties.member_count
        self.projected_targets = self._project(self.targets)
        self.residual = self.projected_targets.copy()
        self.background_image = xp.zeros_like(self.residual)
        self.noise = float(noise)
        # The local magnitudes (x), the candidate logits and the background's log variances, each with its blocks.
        start = initial_hyperparameters(prior, background_variance)
        self.local = _Hyper(start.coefficients.copy(), tuple((np.asarray(block.coordinates), np.asarray(block.factor)) for block in prior.smoothing_blocks), start.log_smoothing.copy())
        head = prior.coefficient_size - prior.scale_size
        indicator = np.zeros((members, self.class_count))
        indicator[np.arange(members), self.class_index] = 1.0
        annotation_blocks = [
            (np.asarray(block.coordinates) - head, np.asarray(block.factor)) for block in prior.smoothing_blocks if np.all(np.asarray(block.coordinates) >= head)
        ]
        deviations = self.class_count if self.class_count > 1 else 0
        self.candidate_design = np.column_stack([indicator[:, :deviations], prior.scale_design])
        candidate_blocks = ([(np.arange(deviations), np.eye(deviations))] if deviations else []) + [
            (coordinates + deviations, factor) for coordinates, factor in annotation_blocks
        ]
        self.candidate = _Hyper(np.zeros(self.candidate_design.shape[1]), tuple(candidate_blocks), np.full(len(candidate_blocks), np.inf))
        self.background_design = np.column_stack([np.ones(members), indicator[:, :deviations], prior.scale_design])
        background_blocks = ([(1 + np.arange(deviations), np.eye(deviations))] if deviations else []) + [
            (coordinates + 1 + deviations, factor) for coordinates, factor in annotation_blocks
        ]
        level = np.zeros(self.background_design.shape[1])
        level[0] = math.log(background_variance)
        self.background = _Hyper(level, tuple(background_blocks), np.full(len(background_blocks), np.inf))
        self.offsets = np.asarray(prior.log_variance_offset, dtype=np.float64)
        generator = np.random.default_rng([seed, 0])
        signs = generator.choice(np.array([-1.0, 1.0]), size=(int(self.mask.shape[0]), self.draw_count))
        self.probes = xp.asarray(signs) * self.mask[:, None]
        self.mean_duals = xp.zeros((int(self.mask.shape[0]), 1))
        self.background_group_mean = np.zeros(self.groups)
        self.diagonal = np.zeros(self.groups)
        self.moments: BackgroundMoments | None = None
        self.member_second = np.zeros(members)
        self.divergence_total = 0.0
        self.spread_total = 0.0
        self.statistics = self._empty_statistics()
        self.history: list[History] = []
        self.log_determinant = 0.0
        self.expanded_trace = 0.0
        self._probe_midpoints = np.zeros(self.draw_count)

    # the design, projected

    def _project(self, values: Any) -> Any:
        """P v = (I - H) M v for v (n,) or (n, c)."""
        xp = self.xp
        matrix = values[:, None] if values.ndim == 1 else values
        projected = self.unit.design_to_sample(matrix, xp.zeros(int(matrix.shape[1]), dtype=xp.int64))
        return projected[:, 0] if values.ndim == 1 else projected

    def background_variances(self) -> tuple[F64Array, F64Array]:
        """(member D_j, group D_g) at the current background coefficients."""
        member = np.exp(self.offsets + self.background_design @ self.background.coefficients)
        return member, np.bincount(self.ties.group, weights=member, minlength=self.groups)

    def _candidate_logits(self) -> F64Array:
        return self.candidate_design @ self.candidate.coefficients

    def _empty_statistics(self) -> dict:
        members = self.ties.member_count
        return {
            "weight": np.zeros(members), "moment": np.zeros(members), "masses": np.zeros((self.class_count, self.prior.grid_size)),
            "live": np.zeros(len(self.windows)),
        }

    # the local sweep

    def _window_tiles(self) -> Iterator[tuple[int, Window, dict, dict]]:
        """Each window with its core blocks' tiles and its halo's dense columns, from one read of the source: a window
        is taken up when the block that completes it arrives (``Window.ready``), a halo's columns are taken from their
        block as it passes, and a core block a later window still needs is held over (``_held``)."""
        xp = self.xp
        by_ready: dict[int, list[int]] = {}
        last_use: dict[int, int] = {}
        halo_needs: dict[int, list[tuple[int, I64Array]]] = {}
        for index, window in enumerate(self.windows):
            by_ready.setdefault(window.ready, []).append(index)
            for block in window.blocks:
                last_use[block] = max(last_use.get(block, -1), window.ready)
            for block, groups in window.halo:
                halo_needs.setdefault(block, []).append((index, groups - int(self.source.block_bounds[block][0])))
        held: dict[int, Any] = {}
        halo_columns: dict[int, dict[int, Any]] = {index: {} for index in range(len(self.windows))}
        for block, (_start, _stop, tile) in enumerate(self.source.blocks()):
            held[block] = tile
            for index, local in halo_needs.get(block, []):
                halo_columns[index][block] = tile.columns(xp.asarray(local))
            for index in by_ready.get(block, []):
                yield index, self.windows[index], held, halo_columns.pop(index)
            for kept in list(held):
                if last_use.get(kept, -1) <= block:
                    held.pop(kept)
            if block in held:
                held[block] = _held(tile)

    def _window_back(self, tiles: dict, halo: dict, window: Window, values: Any) -> Any:
        """Xw' v (the window's groups) for v (n,) in the projected space."""
        xp = self.xp
        column = values[:, None]
        parts = [tiles[block].rmatmat(column)[:, 0] for block in window.blocks] + [halo[block].T @ values for block, _groups in window.halo]
        return xp.concatenate(parts)

    def _window_image(self, tiles: dict, halo: dict, window: Window, coefficients: Any) -> Any:
        """P Xw c (n,) for coefficients c on the window's groups."""
        xp = self.xp
        image = xp.zeros_like(self.residual)
        offset = 0
        for block in window.blocks:
            start, stop = self.source.block_bounds[block]
            width = stop - start
            image += tiles[block].matmat(coefficients[offset:offset + width, None])[:, 0]
            offset += width
        for block, groups in window.halo:
            image += halo[block] @ coefficients[offset:offset + groups.shape[0]]
            offset += groups.shape[0]
        return self._project(image)

    def sweep(self) -> None:
        """Every window's effects once, in window order, then its growth (module docstring), from the one global
        residual; the statistics of the M-steps accumulate as the effects settle. Two reads of each block's tile per
        effect update, exact."""
        xp = self.xp
        prior = self.prior
        noise = self.noise
        log_density = xp.asarray(class_log_density(prior, self.local.coefficients))
        member_log_scale = log_scale(prior, self.local.coefficients)
        grid = xp.asarray(prior.log_variance_grid)
        inverse_nodes = xp.exp(-grid)
        logits = self._candidate_logits()
        statistics = self._empty_statistics()
        divergence = spread = 0.0
        for index, window, tiles, halo in self._window_tiles():
            members = window.members
            local_group = xp.asarray(window.local_group)
            signs = xp.asarray(self.sign[members])
            classes = self.class_index[members]
            candidate = logits[members]
            peak = float(candidate.max())
            log_candidate = xp.asarray(candidate - (peak + math.log(float(np.sum(np.exp(candidate - peak))))))
            log_mixture = log_density[xp.asarray(classes)]
            log_variance = xp.asarray(member_log_scale[members])[:, None] + grid[None, :]
            group_squares = xp.asarray(self.squares[window.groups])
            precision = group_squares[local_group] / noise
            indicator = xp.asarray((classes[None, :] == np.arange(self.class_count)[:, None]).astype(np.float64))
            projection = self._window_back(tiles, halo, window, self.residual)
            kept: list[_Effect] = []
            queue = list(self.effects[index])
            growing = last_on = False
            while True:
                if queue:
                    effect = queue.pop(0)
                else:
                    # Growth: one new effect at a time until one stays off. More single effects than the window's
                    # columns add no direction the ones there cannot hold, which bounds the growth.
                    if (growing and not last_on) or len(kept) >= window.group_count:
                        break
                    effect = _Effect(np.zeros(members.size), np.zeros(window.group_count), 0.0, 0.0, 0.0)
                    growing = True
                products = xp.asarray(effect.products)
                old_mean = xp.asarray(effect.mean)
                shift = signs * (projection + products)[local_group] / noise
                update = _single_effect(xp, shift, precision, log_candidate, log_mixture, log_variance, indicator, inverse_nodes)
                change = update.mean - old_mean
                if bool(xp.any(change != 0.0)):
                    group_change = _segment_sum(xp, local_group, signs * change, window.group_count)
                    image = self._window_image(tiles, halo, window, group_change)
                    moved = self._window_back(tiles, halo, window, image)
                    projection -= moved
                    products = products + moved
                    self.residual -= image
                last_on = update.on
                if not update.on:
                    continue
                group_mean = _segment_sum(xp, local_group, signs * update.mean, window.group_count)
                variance_term = float(xp.sum(update.second * precision)) * noise - float(group_mean @ products)
                kept.append(_Effect(_host(update.mean).astype(np.float64), _host(products).astype(np.float64), variance_term, update.divergence, update.log_evidence))
                divergence += update.divergence
                spread += variance_term
                np.add.at(statistics["weight"], members, _host(update.weight))
                np.add.at(statistics["moment"], members, _host(update.moment))
                statistics["masses"] += _host(update.masses)
                statistics["live"][index] += 1.0
            self.effects[index] = kept
        self.statistics = statistics
        self.divergence_total = divergence
        self.spread_total = spread

    # the local prior's M-steps

    def _local_objective(self, coefficients: F64Array) -> tuple[float, F64Array, F64Array]:
        """sum_ck N_ck log w_ck - 1/2 sum_j [W_j a_j + A_j e^-a_j], its gradient and negative Hessian in x."""
        prior = self.prior
        statistics = self.statistics
        map_ = prior.coefficient_map
        z = map_ @ coefficients
        density = z[: prior.density_size].reshape(prior.class_count, prior.grid_size)
        peak = density.max(axis=1, keepdims=True)
        log_norm = peak + np.log(np.sum(np.exp(density - peak), axis=1, keepdims=True))
        probabilities = np.exp(density - log_norm)
        masses = statistics["masses"]
        totals = masses.sum(axis=1)
        a = log_scale(prior, coefficients)
        weight, moment = statistics["weight"], statistics["moment"]
        exponent = moment * np.exp(-a)
        value = float(np.sum(masses * (density - log_norm))) - 0.5 * float(np.sum(weight * a + exponent))
        gradient_z = np.concatenate([(masses - totals[:, None] * probabilities).ravel(), -0.5 * prior.scale_design.T @ (weight - exponent)])
        negative_z = np.zeros((z.size, z.size))
        for position in range(prior.class_count):
            rows = slice(position * prior.grid_size, (position + 1) * prior.grid_size)
            p = probabilities[position]
            negative_z[rows, rows] = totals[position] * (np.diag(p) - np.outer(p, p))
        negative_z[prior.density_size:, prior.density_size:] = 0.5 * (prior.scale_design.T * exponent) @ prior.scale_design
        return value, map_.T @ gradient_z, map_.T @ negative_z @ map_

    def _candidate_objective(self, coefficients: F64Array) -> tuple[float, F64Array, F64Array]:
        """sum_j W_j phi_j - sum_w E_w log sum_(i in w) e^phi_i, its gradient and negative Hessian."""
        design = self.candidate_design
        logits = design @ coefficients
        weight, live = self.statistics["weight"], self.statistics["live"]
        value = float(weight @ logits)
        gradient = design.T @ weight
        negative = np.zeros((coefficients.size, coefficients.size))
        for window, count in zip(self.windows, live):
            if count == 0.0:
                continue
            values = logits[window.members]
            peak = float(values.max())
            share = np.exp(values - peak)
            total = float(share.sum())
            share /= total
            rows = design[window.members]
            value -= count * (peak + math.log(total))
            mean = rows.T @ share
            gradient -= count * mean
            negative += count * ((rows.T * share) @ rows - np.outer(mean, mean))
        return value, gradient, negative

    def _background_objective(self, coefficients: F64Array) -> tuple[float, F64Array, F64Array]:
        """-1/2 sum_j [a_j + E(u_j^2) e^-a_j], its gradient and negative Hessian."""
        design = self.background_design
        a = self.offsets + design @ coefficients
        exponent = self.member_second * np.exp(-a)
        value = -0.5 * float(np.sum(a + exponent))
        gradient = -0.5 * design.T @ (1.0 - exponent)
        negative = 0.5 * (design.T * exponent) @ design
        return value, gradient, negative

    def _local_value(self) -> float:
        return self._local_objective(self.local.coefficients)[0] + self._candidate_objective(self.candidate.coefficients)[0]

    def local_m_step(self) -> None:
        """x and the candidate logits at fixed q and weights; the effects' KL moves by minus the rise of their
        expected log prior (exact: q is unchanged)."""
        before = self._local_value()
        self.local.coefficients, _ = _maximize(self.local, self._local_objective, self.local.log_smoothing, self.tolerance)
        if self.candidate.size:
            self.candidate.coefficients, _ = _maximize(self.candidate, self._candidate_objective, self.candidate.log_smoothing, self.tolerance)
        self.divergence_total -= self._local_value() - before

    def expansion_step(self) -> float:
        """The parameter-expanded step for the background's scale (module docstring): alpha = y_r' Xp m_u /
        (||Xp m_u||^2 + T_u), T_u = sigma^2 gamma_u, then m_u, Xp m_u by alpha, Sigma_u and D by alpha^2."""
        assert self.moments is not None
        working = self.residual + self.background_image
        square = float(self.background_image @ self.background_image)
        trace = self.noise * self.moments.effective_count
        self.expanded_trace = trace
        denominator = square + trace
        alpha = float(working @ self.background_image) / denominator if denominator > 0.0 else 1.0
        # The likelihood term is a concave quadratic in alpha; a non-positive maximizer means no rescaling of u by
        # a positive factor helps, and alpha = 1 is the expanded family's original point.
        if not alpha > 0.0:
            return 1.0
        self.background_image = alpha * self.background_image
        self.residual = working - self.background_image
        self.background_group_mean *= alpha
        self.member_second *= alpha * alpha
        self.background.coefficients[0] += 2.0 * math.log(alpha)
        self.expanded_trace = trace * alpha * alpha
        return alpha

    def background_m_step(self) -> None:
        self.background.coefficients, _ = _maximize(self.background, self._background_objective, self.background.log_smoothing, self.tolerance)

    def noise_m_step(self, trace: float) -> None:
        residual_square = float(self.residual @ self.residual)
        self.noise = (residual_square + trace + self.spread_total) / self.residual_dimension

    # the background E-step

    def _operator(self, variances: F64Array) -> Callable[[Any], Any]:
        xp = self.xp
        weights = (self.mask / self.noise)[:, None]
        models = DualModels(weights, xp.asarray(variances)[:, None], self.covariates, xp)

        def apply(values: Any) -> Any:
            return apply_operator(self.source, models, values, xp.zeros(int(values.shape[1]), dtype=xp.int64), 0.0, self.count, "structured")

        self._models = models
        return apply

    def background_step(self) -> None:
        """q_u optimal at the current (D, sigma^2): the certified mean, the probes' traces and log det S, and the
        members' second moments for the next M-step (module docstring)."""
        xp = self.xp
        member_variance, group_variance = self.background_variances()
        apply = self._operator(group_variance)
        root = math.sqrt(self.noise)
        right = (self.residual + self.background_image) / root
        stacked = xp.concatenate([right[:, None], self.probes], axis=1)
        start = xp.concatenate([self.mean_duals, xp.zeros_like(self.probes)], axis=1)
        probe_norm = math.sqrt(self.training_count)
        bounds = np.concatenate([[math.sqrt(1.0 / self.draw_count)], np.full(self.draw_count, math.sqrt(2.0 * self.residual_dimension / self.draw_count) / probe_norm)])
        quadrature = np.concatenate([[False], np.ones(self.draw_count, dtype=bool)])
        result = _krylov(xp, apply, stacked, start, bounds, quadrature, 1.0 / self.draw_count)
        duals = result.solution
        self.mean_duals = duals[:, :1].copy()
        # One read: Xt'[zeta, w, z] per block, reduced to the mean and the diagonal estimates at once.
        models = self._models
        operand = xp.concatenate([duals, self.probes], axis=1)
        left = models.sample_to_design(operand, xp.zeros(int(operand.shape[1]), dtype=xp.int64))
        mean = np.zeros(self.groups)
        diagonal = np.zeros(self.groups)
        per_probe = np.zeros(self.draw_count)
        count = self.draw_count
        for start_row, stop_row, tile in self.source.blocks():
            products = tile.rmatmat(left)
            variances = xp.asarray(group_variance[start_row:stop_row])
            mean[start_row:stop_row] = _host(variances * products[:, 0])
            paired = products[:, 1:1 + count] * products[:, 1 + count:]
            diagonal[start_row:stop_row] = _host(paired.mean(axis=1))
            per_probe += _host(variances @ paired)
        self.count.note(int(operand.shape[1]), 0.0, "structured-moments")
        # Xp m_u = sigma Xt m_u = sigma ((S - I) zeta) = sigma (b - rho - zeta), rho the exact residual.
        exact_residual = result.residual[:, 0]
        image = root * (right - exact_residual - self.mean_duals[:, 0])
        self.residual = self.residual + self.background_image - image
        self.background_image = image
        self.background_group_mean = mean
        self.diagonal = diagonal
        group_second = mean * mean + group_variance - group_variance * group_variance * diagonal
        share = member_variance / group_variance[self.ties.group]
        self.member_second = share * share * group_second[self.ties.group] + member_variance - member_variance * share
        effective = per_probe
        brackets = result.brackets[1:]
        midpoints = brackets.mean(axis=1)
        self.log_determinant = float(midpoints.mean())
        self.moments = BackgroundMoments(
            effective_count=float(effective.mean()), effective_count_error=float(effective.std(ddof=1) / math.sqrt(count)) if count > 1 else np.inf,
            log_determinant=self.log_determinant, log_determinant_error=float(midpoints.std(ddof=1) / math.sqrt(count)) if count > 1 else np.inf,
            log_determinant_width=float(np.mean(brackets[:, 1] - brackets[:, 0])),
            mean_certificate=float(np.linalg.norm(_host(exact_residual))), probe_residual=float(result.residual_norm[1:].max()),
            iterations=result.iterations,
        )
        self._probe_midpoints = midpoints
        self._probe_brackets = brackets

    # the objective

    def elbo(self) -> float:
        """F at the current state (module docstring); q_u must be optimal for the current (D, sigma^2)."""
        _member, group_variance = self.background_variances()
        residual_square = float(self.residual @ self.residual)
        return (
            -0.5 * self.residual_dimension * (_LOG_TWO_PI + math.log(self.noise))
            - (residual_square + self.spread_total) / (2.0 * self.noise)
            - 0.5 * float(np.sum(self.background_group_mean ** 2 / group_variance))
            - 0.5 * self.log_determinant
            - self.divergence_total
        )

    def objective(self) -> float:
        return self.elbo() + self.local.log_prior() + self.candidate.log_prior() + self.background.log_prior()

    def smoothing_step(self) -> float:
        """One MacKay step of every hyper-problem's weights (``_mackay``) at the current statistics; returns the sum of
        their evidence rises. J is re-read at the new weights by the caller."""
        rise = 0.0
        for hyper, objective in ((self.local, self._local_objective), (self.candidate, self._candidate_objective), (self.background, self._background_objective)):
            if hyper.blocks:
                hyper.log_smoothing, gain = _mackay(hyper, objective, self.tolerance)
                rise += gain
        return rise

    # results

    def member_mean(self) -> F64Array:
        member_variance, group_variance = self.background_variances()
        mean = self.sign * member_variance / group_variance[self.ties.group] * self.background_group_mean[self.ties.group]
        for window, effects in zip(self.windows, self.effects):
            for effect in effects:
                mean[window.members] += effect.mean
        return mean

    def covariate_coefficients(self, member_mean: F64Array) -> F64Array:
        xp = self.xp
        group = np.bincount(self.ties.group, weights=self.sign * member_mean, minlength=self.groups)
        image = xp.zeros_like(self.residual)
        for start, stop, tile in self.source.blocks():
            image += tile.matmat(xp.asarray(group[start:stop])[:, None])[:, 0]
        self.count.note(1, 0.0, "structured-covariates")
        remainder = self.mask * (self.targets - image)
        right = (self.covariates.T @ remainder)[:, None]
        return np.asarray(_host(self.unit.covariate_solve(right, self.single)), dtype=np.float64)[:, 0]

    def background_draws(self, draw_count: int, seed: int, consume: Callable[[int, int, Any], None]) -> F64Array:
        """``draw_count`` draws of the background's groups from q_u by perturbation (module docstring), handed to
        ``consume(start, stop, draws)`` block by block as (stop - start, draw_count) arrays; returns each draw's
        certificate ||u_hat* - u*||_A (the dual residual of its solve, at most K^-1/2). Two reads plus the solve."""
        xp = self.xp
        _member, group_variance = self.background_variances()
        apply = self._operator(group_variance)
        models = self._models
        columns = xp.zeros(draw_count, dtype=xp.int64)

        def prior_noise(block: int, width: int) -> Any:
            return xp.asarray(np.random.default_rng([seed, 1, block]).standard_normal((width, draw_count)))

        image = xp.zeros((int(self.mask.shape[0]), draw_count))
        for block, (start, stop, tile) in enumerate(self.source.blocks()):
            image += tile.matmat(xp.sqrt(xp.asarray(group_variance[start:stop]))[:, None] * prior_noise(block, stop - start))
        self.count.note(draw_count, 0.0, "structured-draw-rhs")
        sample_noise = xp.asarray(np.random.default_rng([seed, 2]).standard_normal((int(self.mask.shape[0]), draw_count)))
        right = sample_noise - models.design_to_sample(image, columns)
        bound = np.full(draw_count, math.sqrt(1.0 / self.draw_count))
        result = _krylov(xp, apply, right, xp.zeros_like(right), bound, np.zeros(draw_count, dtype=bool), np.inf)
        left = models.sample_to_design(result.solution, columns)
        for block, (start, stop, tile) in enumerate(self.source.blocks()):
            variances = xp.asarray(group_variance[start:stop])[:, None]
            draws = xp.asarray(self.background_group_mean[start:stop])[:, None] + xp.sqrt(variances) * prior_noise(block, stop - start)
            consume(start, stop, draws + variances * tile.rmatmat(left))
        self.count.note(draw_count, 0.0, "structured-draws")
        return result.residual_norm


def fit_structured(
    *, source: DualTileSource, chromosomes: Sequence[str], ties: TieGroups, prior: ScaleMixturePrior, mask: Any, targets: Any, covariates: Any,
    noise: float, background_variance: float, draw_count: int, seed: int, proxies: I64Array | None = None,
) -> StructuredFit:
    """The structured fit from a start (``noise`` sigma^2 and ``background_variance`` the background's level, e.g.
    Haseman-Elston's split: ``full_data_fit.moment_starts``), to where one iteration's ascent at fixed weights and
    the weights' own evidence step are both at most 1/(2K) nats (K = ``draw_count``). Each iteration: the local sweep,
    the M-steps (local prior, candidate logits, the expanded background scale, the background log variances, sigma^2),
    the background E-step, then one smoothing step (module docstring)."""
    state = StructuredFull(
        source=source, chromosomes=chromosomes, ties=ties, prior=prior, mask=mask, targets=targets, covariates=covariates, noise=noise,
        background_variance=background_variance, draw_count=draw_count, seed=seed, proxies=proxies,
    )
    state.background_step()
    objective = state.objective()
    converged = False
    previous_midpoints = state._probe_midpoints
    while True:
        started = time.time()
        passes = state.count.passes
        state.sweep()
        state.local_m_step()
        alpha = state.expansion_step()
        state.background_m_step()
        state.noise_m_step(state.expanded_trace)
        state.background_step()
        updated = state.objective()
        gain = updated - objective
        # The gain's probe part is half the change of the log det estimate; its standard error over the probes.
        difference = state._probe_midpoints - previous_midpoints
        gain_error = 0.5 * float(difference.std(ddof=1) / math.sqrt(difference.size)) if difference.size > 1 else np.inf
        previous_midpoints = state._probe_midpoints
        evidence_rise = state.smoothing_step()
        objective = state.objective()
        state.history.append(History(
            objective=objective, fixed_smoothing_gain=gain, gain_error=gain_error, evidence_rise=evidence_rise, elbo=state.elbo(), noise=state.noise,
            effective_count=state.moments.effective_count if state.moments else np.nan,
            live_effects=sum(len(effects) for effects in state.effects), expansion=alpha, passes=state.count.passes - passes,
            seconds=time.time() - started,
        ))
        log(f"structured: iteration {len(state.history)}: {state.history[-1]}")
        if abs(gain) <= max(state.tolerance, gain_error) and evidence_rise <= state.tolerance:
            converged = True
            break
    mean = state.member_mean()
    member_variance, group_variance = state.background_variances()
    assert state.moments is not None
    return StructuredFit(
        member_mean=mean,
        inclusion=state.statistics["weight"].copy(),
        background_group_mean=state.background_group_mean.copy(),
        background_variance=group_variance,
        member_background_variance=member_variance,
        covariate_coefficients=state.covariate_coefficients(mean),
        noise=state.noise,
        local_hyperparameters=MixtureHyperparameters(coefficients=state.local.coefficients.copy(), log_smoothing=state.local.log_smoothing.copy()),
        background_coefficients=state.background.coefficients.copy(),
        background_log_smoothing=state.background.log_smoothing.copy(),
        candidate_coefficients=state.candidate.coefficients.copy(),
        candidate_log_smoothing=state.candidate.log_smoothing.copy(),
        elbo=state.elbo(),
        objective=objective,
        moments=state.moments,
        live_effects=sum(len(effects) for effects in state.effects),
        effects_per_window=np.array([len(effects) for effects in state.effects], dtype=np.int64),
        history=tuple(state.history),
        passes=state.count.passes,
        converged=converged,
    )


@dataclass(frozen=True)
class StoreFit:
    """A store's structured fit with what scoring needs: each member's store row, its signed-code mean and scale
    (Stage 0's), and the prior the fit learned on."""

    fit: StructuredFit
    store_rows: I64Array
    signed_means: F64Array
    signed_scales: F64Array
    class_index: I64Array
    prior: ScaleMixturePrior
    block_count: int


def fit_store(
    *, store: Any, training_columns: I64Array, covariates: F64Array, targets: F64Array, log_reliability: F64Array, budget: Any, work_dir: Path,
    seed: int, draw_count: int,
) -> StoreFit:
    """One quantitative model on the sorted store columns ``training_columns`` (covariates with the intercept first):
    Stage 0 and the prior exactly as ``stage2_wiring._fit_one`` builds them (every record with signal, one class per
    variant class, the reliabilities and per-unit baseline as offsets, the store's other columns as the annotation
    design), the moment start (``full_data_fit.moment_starts``), then ``fit_structured`` on the streamed blocks."""
    from sv_pgs.annotation_design import annotation_design, column_variance_annotation, per_unit_offset  # noqa: PLC0415
    from sv_pgs.config import ModelConfig  # noqa: PLC0415
    from sv_pgs.dual_solve import StreamedDualSource  # noqa: PLC0415
    from sv_pgs.full_data_fit import covariate_residual_variance, moment_starts, stage0_lattice  # noqa: PLC0415
    from sv_pgs.genotype_statistics import DosageStoreTileSource, compute_genotype_statistics  # noqa: PLC0415
    from sv_pgs.scale_mixture_ep import scale_mixture_prior  # noqa: PLC0415
    from sv_pgs.stage2_wiring import RELIABILITY_COLUMNS, _block_cap, stage0_candidates  # noqa: PLC0415
    from sv_pgs.store_block_source import StoreGenotypeBlockSource  # noqa: PLC0415

    config = ModelConfig(minimum_minor_allele_frequency=0.0)
    candidates = stage0_candidates(store, training_columns, log_reliability, config)
    if candidates.shape[0] == 0:
        raise ValueError("no store record carries signal on these training rows")
    block_cap = _block_cap(store, candidates, training_columns, covariates.shape[1], budget)
    statistics = compute_genotype_statistics(
        DosageStoreTileSource(store, candidates), training_columns, covariates, targets[:, None], config, budget, block_cap, work_dir / "ld"
    )
    member_rows = np.asarray(statistics.active_rows, dtype=np.int64)
    value_scales = np.asarray(statistics.scales) / np.asarray(store.variant_table.codes_per_unit, dtype=np.float64)[member_rows]
    offsets = log_reliability[member_rows] + per_unit_offset(value_scales)
    _classes, class_index = np.unique(store.variant_table.variant_class[member_rows], return_inverse=True)
    table = store.variant_table
    member_annotations = {name: np.asarray(values)[member_rows] for name, values in table.annotations.items()}
    member_annotations.update(column_variance_annotation(np.asarray(statistics.scales)))
    annotations = annotation_design(member_annotations, table.annotation_legends, class_index=class_index.astype(np.int64), exclude=RELIABILITY_COLUMNS)
    log(f"structured: {annotations.design.shape[1]} annotation columns in {len(annotations.groups)} groups: {', '.join(annotations.names) or 'none'}")
    mask = np.zeros(store.n_samples)
    mask[training_columns] = 1.0
    store_targets = np.zeros(store.n_samples)
    store_targets[training_columns] = targets
    store_covariates = np.zeros((store.n_samples, covariates.shape[1]))
    store_covariates[training_columns] = covariates
    start_noise = float(covariate_residual_variance(store_targets[:, None], mask[:, None], store_covariates)[0])
    nodes, floor, top = stage0_lattice(statistics, 0, start_noise, offsets, 0.5 / draw_count)
    prior = scale_mixture_prior(
        class_index=class_index.astype(np.int64), log_variance_offset=offsets, annotation_design=annotations.design,
        annotation_groups=annotations.groups, nodes=nodes, floor=floor, top=top,
    )
    (moment,) = moment_starts(statistics, prior)
    source = StreamedDualSource(StoreGenotypeBlockSource.from_statistics(store, statistics, budget, budget.working_bytes // 2))
    chromosomes = [statistics.ld.block(block).chromosome for block in range(statistics.ld.block_count)]
    proxies = stage0_proxies(statistics, proxy_threshold(draw_count), budget.working_bytes // 2)
    log(
        f"structured: {member_rows.shape[0]:,} members, {source.variant_count:,} reduced columns in {statistics.ld.block_count} blocks; "
        f"{proxies.shape[0]:,} cross-boundary proxy pairs; start h2 {moment.heritability:.4g}, noise {moment.noise:.6g}"
    )
    fit = fit_structured(
        source=source, chromosomes=chromosomes, ties=TieGroups.from_tie_map(statistics.tie_map), prior=prior, mask=mask, targets=store_targets,
        covariates=store_covariates, noise=moment.noise, background_variance=moment.mean_variance, draw_count=draw_count, seed=seed,
        proxies=proxies,
    )
    return StoreFit(
        fit=fit, store_rows=member_rows, signed_means=np.asarray(statistics.means, dtype=np.float64),
        signed_scales=np.asarray(statistics.scales, dtype=np.float64), class_index=class_index.astype(np.int64), prior=prior,
        block_count=int(statistics.ld.block_count),
    )
