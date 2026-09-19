"""Marginal posterior variances of the Stage 2 Gaussian from sample-side solves (leave-block-out, EDB-EP).

Model, per trait x fold. The metric is folded into the columns, Xt = (I - H) W^1/2 X: fold masks are
W = 0 rows, W is the likelihood curvature, H projects out the covariates. Gaussian sites give q(beta)
with precision A = Xt'Xt + diag(Pi). Stage 2's EP needs diag(A^-1) at every refresh.

Block-Jacobi inverses diag((Xt_b'Xt_b + Pi_b)^-1) are the variances of block b CONDITIONAL on every
other block's effects. They double-count block b's data by the other blocks' signal, so an EP fixed
point built on them is biased. The exact marginals follow from two identities.

1. Resolved and bulk elimination. Let L be the resolved sites (strong effects, and any Pi_k <= 0),
   S the bulk, D_S = Pi_S^-1 > 0, and K_S = I + Xt_S D_S Xt_S'. Then

       A^-1_LL = core^-1,   core = Pi_L + Xt_L' K_S^-1 Xt_L,
       diag(A^-1)_j = D_j - D_j^2 q_j + D_j^2 (c_j core^-1 c_j')   (j in S),

   with q_j = xt_j' K_S^-1 xt_j and c_j the row j of C = Xt' K_S^-1 Xt_L.
   Proof: Schur's complement of A_SS, (Xt_S'Xt_S + Pi_S)^-1 = D_S - D_S Xt_S' K_S^-1 Xt_S D_S
   (push-through), and (Xt_S'Xt_S + Pi_S)^-1 Xt_S'Xt_L = D_S Xt_S' K_S^-1 Xt_L. It needs only
   core > 0, which is q's propriety, so non-positive resolved sites are fine.

2. The bulk quadratic q_j, over a window. Take W = the blocks around b (b-1, b, b+1, where
   cross-Grams are supplied) and F = K_S - Xt_W D_W Xt_W', the far field. Then

       Xt_W' K_S^-1 Xt_W = G - G D_W^1/2 (I + D_W^1/2 G D_W^1/2)^-1 D_W^1/2 G,   G = Xt_W' F^-1 Xt_W.

   This is exact (Woodbury). F is independent of the window's columns, so the deterministic
   equivalent G = omega_F R_W (R_W = Xt_W'Xt_W, omega_F = tr(F^-1)/n) holds, with relative error of
   the order of ||F^-1||_F / tr(F^-1). omega_F solves

       omega_F = omega_S + (omega_S2 / n) sum_i lambda_i / (1 + omega_F lambda_i),

   where lambda_i are the eigenvalues of B = D_W^1/2 R_W D_W^1/2, omega_S = tr(K_S^-1)/n and
   omega_S2 = tr(K_S^-2)/n. This is Woodbury on tr(F^-1) with the same equivalent inside a
   second-order term. The right-hand side is convex and decreasing in omega_F, so the root is unique,
   and Newton from omega_S increases monotonically to it.

`variance_jvp` gives the variance map's derivative, -diag(Sigma diag(w) Sigma), in the same
representation, for the hyperparameter curvature products: resolved pairs exactly, window pairs through
identity 2, and pairs beyond the window through the block sandwich diag(Sigma_bb R_b Sigma_bb), since a
variant's LD partners absorb most of its chance coupling with distant blocks. `covariance_products`
gives Sigma v exactly from the solver's back products, for the certificate's probes.

The deterministic equivalent's own relative error, ||K_S^-1||_F / tr(K_S^-1) = sqrt(omega_S2 / n) /
omega_S, is the scale that `block_trace_certificate` tests each block against. It is the only
approximation here, and it is what the certificate checks: LD coupling a block to the far field,
beyond its window, shows up as a block whose measured trace error provably exceeds that scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as student_t


@dataclass(frozen=True)
class BulkSolve:
    """One model's refresh: every quantity is sample-side, except the p x |L| cross products.

    ``resolved_cross`` is C = Xt' K_S^-1 Xt_L, formed in the refresh's final pass: dense (p x |L|), or only
    on each block's LD window as a ``WindowCross`` (what production stores). ``bulk_trace`` and
    ``bulk_square_trace`` are tr(K_S^-1)/n and tr(K_S^-2)/n, from the bulk solve's probe columns.
    ``kernel_square_trace`` is tr(Q^2)/n for Q = K_S^-1 - Z_L core^-1 Z_L' (Z_L = K_S^-1 Xt_L), the
    sample-side operator that couples any two bulk variants once the resolved sites are eliminated. Its
    probe columns are the bulk probes corrected through core: Q z = K_S^-1 z - Z_L core^-1 Xt_L' K_S^-1 z.
    """

    site_precision: NDArray[np.float64]
    resolved: NDArray[np.int64]
    resolved_core: NDArray[np.float64]
    resolved_cross: "NDArray[np.float64] | WindowCross"
    bulk_trace: float
    bulk_square_trace: float
    kernel_square_trace: float
    sample_count: int


@dataclass(frozen=True)
class WindowCross:
    """C = Xt' K_S^-1 Xt_L kept only where it is not a chance coupling: block b's rows against the resolved
    sites inside b's window (the blocks b-1, b, b+1 when cross-Grams are supplied, else b alone).

    ``positions[b]`` indexes ``BulkSolve.resolved``, and ``values[b]`` is (|b|, len(positions[b])).
    Beyond the window, xt_j' K_S^-1 xt_l has mean zero, and it enters the maps through its expected square
    (``marginal_variances``, ``variance_jvp``). Memory is p x (resolved sites per window), not p x |L|.
    """

    positions: tuple[NDArray[np.int64], ...]
    values: tuple[NDArray[np.float64], ...]


@dataclass(frozen=True)
class BlockGrams:
    """Stage 0 Grams in the model's metric: R_b per block and R_{b,b+1} between genome-adjacent blocks.

    ``next_cross`` has one entry per adjacent pair. An empty tuple makes each window its own block
    alone; the certificate then carries any coupling across the cuts.
    """

    blocks: tuple[NDArray[np.int64], ...]
    within: tuple[NDArray[np.float64], ...]
    next_cross: tuple[NDArray[np.float64], ...]

    def __post_init__(self) -> None:
        if len(self.within) != len(self.blocks):
            raise ValueError("one within-block Gram per block")
        if self.next_cross and len(self.next_cross) != len(self.blocks) - 1:
            raise ValueError("next_cross needs one Gram per adjacent block pair, or none")


@dataclass(frozen=True)
class BlockCertificate:
    """Per block: the probe estimate of a relative error, its standard error, a two-sided interval at the
    certificate's family-wise level, and the decision.

    ``certified`` means the whole interval lies within ``tolerance``: the block's error provably does not exceed
    it. ``violated`` means the whole interval lies beyond it. A block that is neither is undecided, and more probes
    decide it (``probes_to_decide``). ``level`` is the family-wise probability that any interval misses its
    block's true error.
    """

    relative_error: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    lower_bound: NDArray[np.float64]
    upper_bound: NDArray[np.float64]
    tolerance: float
    level: float
    certified: NDArray[np.bool_]
    violated: NDArray[np.bool_]


@dataclass(frozen=True)
class JacobianProduct:
    """-diag(Sigma diag(w) Sigma) (p x r) with its error model.

    ``standard_error`` is the standard deviation of the chance-correlation part, which the equivalent
    replaces by its expectation: bulk pairs beyond each window. ``window_part`` is the part computed through the
    window Woodbury, whose Sigma entries carry the equivalent's relative error (approximation_scale),
    and the squares at most twice it. Resolved rows are exact, so both are zero there.
    """

    values: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    window_part: NDArray[np.float64]


def far_field_trace(bulk_trace: float, bulk_square_trace: float, sample_count: int, eigenvalues: NDArray[np.float64]) -> float:
    """omega_F: the root of f(w) = w - omega_S - (omega_S2 / n) sum lambda / (1 + w lambda).

    f is increasing and concave, and f(omega_S) <= 0. So Newton from omega_S never overshoots and
    increases monotonically. It stops when an iterate no longer increases in floating point.
    """
    weight = bulk_square_trace / sample_count
    current = bulk_trace
    while True:
        denominator = 1.0 + current * eigenvalues
        value = current - bulk_trace - weight * float(np.sum(eigenvalues / denominator))
        slope = 1.0 + weight * float(np.sum(np.square(eigenvalues / denominator)))
        candidate = current - value / slope
        if not candidate > current:
            return current
        current = candidate


def _whitened_spectrum(gram: NDArray[np.float64], bulk_variance: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Eigenpairs of B = D^1/2 R D^1/2 (PSD; rounding below zero is clipped to its exact bound 0)."""
    root = np.sqrt(bulk_variance)
    eigenvalues, eigenvectors = np.linalg.eigh(root[:, None] * gram * root[None, :])
    return np.maximum(eigenvalues, 0.0), eigenvectors


def _quadratic_from_spectrum(
    gram: NDArray[np.float64],
    bulk_variance: NDArray[np.float64],
    far_trace: float,
    eigenvalues: NDArray[np.float64],
    eigenvectors: NDArray[np.float64],
) -> NDArray[np.float64]:
    projected = (gram * np.sqrt(bulk_variance)[None, :]) @ eigenvectors
    correction = np.einsum("ik,k,ik->i", projected, 1.0 / (1.0 + far_trace * eigenvalues), projected)
    return far_trace * np.diag(gram) - far_trace**2 * correction


def window_bulk_quadratic(gram: NDArray[np.float64], bulk_variance: NDArray[np.float64], far_trace: float) -> NDArray[np.float64]:
    """diag(Xt_W' K_S^-1 Xt_W) given G = far_trace * R_W (exact when the far field is far_trace^-1 I).

    With B = D^1/2 R D^1/2 = V diag(lambda) V', the quadratic is
    G - far_trace^2 (R D^1/2 V) diag(1 / (1 + far_trace lambda)) (R D^1/2 V)'.
    """
    eigenvalues, eigenvectors = _whitened_spectrum(gram, bulk_variance)
    return _quadratic_from_spectrum(gram, bulk_variance, far_trace, eigenvalues, eigenvectors)


def _window(grams: BlockGrams, block: int) -> tuple[NDArray[np.float64], NDArray[np.int64], slice]:
    """The window's Gram, its variant indices, and the own block's rows in it."""
    if grams.next_cross:
        members = [neighbour for neighbour in (block - 1, block, block + 1) if 0 <= neighbour < len(grams.blocks)]
    else:
        members = [block]
    sizes = [grams.blocks[member].shape[0] for member in members]
    starts = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    gram = np.zeros((int(starts[-1]), int(starts[-1])))
    for position, member in enumerate(members):
        span = slice(int(starts[position]), int(starts[position + 1]))
        gram[span, span] = grams.within[member]
        if position + 1 < len(members):
            following = slice(int(starts[position + 1]), int(starts[position + 2]))
            gram[span, following] = grams.next_cross[member]
            gram[following, span] = grams.next_cross[member].T
    columns = np.concatenate([grams.blocks[member] for member in members])
    own_position = members.index(block)
    own = slice(int(starts[own_position]), int(starts[own_position + 1]))
    return gram, columns, own


def marginals_from_quadratics(
    site_precision: NDArray[np.float64],
    resolved: NDArray[np.int64],
    resolved_core: NDArray[np.float64],
    resolved_cross: NDArray[np.float64],
    bulk_quadratic: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Identity 1 with a dense C: diag(A^-1) from q_j = xt_j' K_S^-1 xt_j (bulk j) and the resolved core."""
    is_resolved = np.zeros(site_precision.shape[0], dtype=bool)
    is_resolved[resolved] = True
    bulk_variance = np.where(is_resolved, 0.0, 1.0 / np.where(is_resolved, 1.0, site_precision))
    spikes = np.einsum("il,lk,ik->i", resolved_cross, np.linalg.inv(resolved_core), resolved_cross) if resolved.shape[0] else 0.0
    variances = bulk_variance - np.square(bulk_variance) * (bulk_quadratic - spikes)
    variances[resolved] = np.diag(np.linalg.inv(resolved_core))
    return variances


def _block_index(grams: BlockGrams) -> NDArray[np.int64]:
    index = np.empty(sum(members.shape[0] for members in grams.blocks), dtype=np.int64)
    for block, members in enumerate(grams.blocks):
        index[members] = block
    return index


def _window_blocks(grams: BlockGrams, block: int) -> list[int]:
    if grams.next_cross:
        return [neighbour for neighbour in (block - 1, block, block + 1) if 0 <= neighbour < len(grams.blocks)]
    return [block]


def window_cross(solve: BulkSolve, grams: BlockGrams) -> WindowCross:
    """C restricted to each block's window (``WindowCross``), from a dense C or as given."""
    if isinstance(solve.resolved_cross, WindowCross):
        return solve.resolved_cross
    block_of_resolved = _block_index(grams)[solve.resolved]
    positions = []
    values = []
    for block, members in enumerate(grams.blocks):
        inside = np.nonzero(np.isin(block_of_resolved, _window_blocks(grams, block)))[0].astype(np.int64)
        positions.append(inside)
        values.append(solve.resolved_cross[np.ix_(members, inside)] if inside.shape[0] else np.zeros((members.shape[0], 0)))
    return WindowCross(positions=tuple(positions), values=tuple(values))


def _cross_rows(cross: WindowCross, block: int, positions: NDArray[np.int64]) -> NDArray[np.float64]:
    """C on block's rows against ``positions`` (indices into L); zero where the pair is beyond the window."""
    stored = {int(position): column for column, position in enumerate(cross.positions[block].tolist())}
    out = np.zeros((cross.values[block].shape[0], positions.shape[0]))
    for column, position in enumerate(positions.tolist()):
        if position in stored:
            out[:, column] = cross.values[block][:, stored[position]]
    return out


@dataclass(frozen=True)
class _BlockTerms:
    """One block's window algebra: Sigma on (own rows) x (window columns) for bulk pairs (identity 2 plus the
    window's resolved spikes), the own bulk quadratics q, Sigma_bb in full, and the window's resolved positions."""

    columns: NDArray[np.int64]
    rows: NDArray[np.float64]
    quadratic: NDArray[np.float64]
    near: NDArray[np.int64]
    loadings: NDArray[np.float64]
    covariance: NDArray[np.float64]


def _block_terms(
    solve: BulkSolve, grams: BlockGrams, cross: WindowCross, bulk_variance: NDArray[np.float64], core_inverse: NDArray[np.float64], block: int
) -> _BlockTerms:
    gram, columns, own = _window(grams, block)
    members = grams.blocks[block]
    window_variance = bulk_variance[columns]
    eigenvalues, eigenvectors = _whitened_spectrum(gram, window_variance)
    far_trace = far_field_trace(solve.bulk_trace, solve.bulk_square_trace, solve.sample_count, eigenvalues)
    projected = (gram * np.sqrt(window_variance)[None, :]) @ eigenvectors
    quadratic = far_trace * gram[own] - far_trace**2 * (projected[own] / (1.0 + far_trace * eigenvalues)[None, :]) @ projected.T
    near = cross.positions[block]
    near_inverse = core_inverse[np.ix_(near, near)]
    window_cross_rows = np.concatenate([_cross_rows(cross, member, near) for member in _window_blocks(grams, block)], axis=0)
    spikes = cross.values[block] @ near_inverse @ window_cross_rows.T
    own_variance = bulk_variance[members]
    diagonal = (np.arange(members.shape[0]), np.arange(own.start, own.stop))
    rows = -own_variance[:, None] * (quadratic - spikes) * window_variance[None, :]
    rows[diagonal] += own_variance
    # Sigma_jL = -D_j (C core^-1)_jL, with C_j known on the window's resolved sites.
    loadings = cross.values[block] @ core_inverse[near, :]
    covariance = rows[:, own].copy()
    local = np.nonzero(np.isin(solve.resolved, members))[0]
    if local.shape[0]:
        member_positions = np.array([int(np.nonzero(members == variant)[0][0]) for variant in solve.resolved[local].tolist()])
        bulk_to_resolved = -own_variance[:, None] * loadings[:, local]
        covariance[:, member_positions] = bulk_to_resolved
        covariance[member_positions, :] = bulk_to_resolved.T
        covariance[np.ix_(member_positions, member_positions)] = core_inverse[np.ix_(local, local)]
    return _BlockTerms(columns=columns, rows=rows, quadratic=quadratic[diagonal], near=near, loadings=loadings, covariance=covariance)


def _prepare(solve: BulkSolve, grams: BlockGrams) -> tuple[WindowCross, NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    variant_count = solve.site_precision.shape[0]
    is_resolved = np.zeros(variant_count, dtype=bool)
    is_resolved[solve.resolved] = True
    bulk_variance = np.where(is_resolved, 0.0, 1.0 / np.where(is_resolved, 1.0, solve.site_precision))
    core_inverse = np.linalg.inv(solve.resolved_core) if solve.resolved.shape[0] else np.zeros((0, 0))
    return window_cross(solve, grams), bulk_variance, core_inverse, is_resolved


def marginal_variances(solve: BulkSolve, grams: BlockGrams) -> NDArray[np.float64]:
    """(p,) diag(A^-1) for one model, by identities 1 and 2 (module docstring).

    With C kept on LD windows only, a bulk j's coupling to resolved sites beyond its window, D_j^2 c_j,far
    core^-1 c_j,far', enters by its expectation. For far pairs Sigma_jl ~ -D_j c_jl (core^-1)_ll and
    E[Sigma_jl^2] = u_j u_l tr(Q^2) / n^2 (the far-field model of ``variance_jvp``), so the term is
    u_j tr(Q^2) / n^2 sum_{l far} u_l / Sigma_ll, with u = diag(Sigma_bb R_b Sigma_bb).
    """
    cross, bulk_variance, core_inverse, is_resolved = _prepare(solve, grams)
    variant_count = solve.site_precision.shape[0]
    near_variance = np.empty(variant_count)
    sandwich = np.zeros(variant_count)
    near_totals = np.zeros(len(grams.blocks))
    resolved_variance = np.diag(core_inverse)
    for block, members in enumerate(grams.blocks):
        terms = _block_terms(solve, grams, cross, bulk_variance, core_inverse, block)
        near_variance[members] = np.diag(terms.covariance)
        sandwich[members] = np.einsum("ij,jk,ki->i", terms.covariance, grams.within[block], terms.covariance)
    resolved_weight = sandwich[solve.resolved] / resolved_variance if solve.resolved.shape[0] else np.zeros(0)
    for block in range(len(grams.blocks)):
        near_totals[block] = float(np.sum(resolved_weight[cross.positions[block]]))
    far_scale = solve.kernel_square_trace / solve.sample_count
    block_of = _block_index(grams)
    far = sandwich * far_scale * (float(np.sum(resolved_weight)) - near_totals[block_of])
    variances = np.where(is_resolved, near_variance, near_variance + far)
    variances[solve.resolved] = resolved_variance
    return variances


def approximation_scale(solve: BulkSolve) -> float:
    """||K_S^-1||_F / tr(K_S^-1): the deterministic equivalent's relative error scale (Theorem 2)."""
    return float(np.sqrt(solve.bulk_square_trace / solve.sample_count) / solve.bulk_trace)


def certificate_level(draw_count: int) -> float:
    """The certificate's family-wise error probability: 1/K for a scorer of K posterior draws.

    The scorer resolves posterior probabilities only to 1/K: an event rarer than that is, on average, absent from
    its K draws. ep_eb.md §3.3 sets every numerical tolerance to what the K draws cannot see, and this is the same
    rule applied to the certificate's chance of letting a bad block through.
    """
    return 1.0 / draw_count


def _certificate(
    estimate: NDArray[np.float64], per_probe: list[NDArray[np.float64]], tolerance: float, level: float
) -> BlockCertificate:
    """Intervals from k probe values per block, at family-wise ``level`` over the B blocks (Bonferroni, two-sided).

    (mean - estimate) / (sd / sqrt k) is referred to Student's t with k - 1 degrees of freedom. That is exact for
    Gaussian probe values. A block's probe value is a Rademacher quadratic form over many pairs, which is close to
    Gaussian, so the level is approximate, and conservative in the tail compared with the normal quantile. A
    block with zero estimate and zero spread (every site resolved, so exact) is certified.
    """
    block_count = len(per_probe)
    probe_count = per_probe[0].shape[0]
    quantile = float(student_t.isf(0.5 * level / block_count, probe_count - 1))
    relative = np.zeros(block_count)
    standard = np.zeros(block_count)
    for position, values in enumerate(per_probe):
        computed = float(estimate[position])
        spread = float(np.std(values, ddof=1)) / np.sqrt(probe_count)
        if computed == 0.0:
            if spread != 0.0:
                raise ValueError(f"block {position}: zero estimate with nonzero probe spread; a zero estimate must mean an exact block")
            continue
        relative[position] = (float(np.mean(values)) - computed) / computed
        standard[position] = spread / abs(computed)
    lower = relative - quantile * standard
    upper = relative + quantile * standard
    certified = (lower >= -tolerance) & (upper <= tolerance)
    violated = (lower > tolerance) | (upper < -tolerance)
    return BlockCertificate(
        relative_error=relative, standard_error=standard, lower_bound=lower, upper_bound=upper,
        tolerance=tolerance, level=level, certified=certified, violated=violated,
    )


def block_trace_certificate(
    variances: NDArray[np.float64],
    blocks: tuple[NDArray[np.int64], ...],
    probes: NDArray[np.float64],
    covariance_probes: NDArray[np.float64],
    tolerance: float,
    level: float,
) -> BlockCertificate:
    """Test each block's tr(Sigma_bb) against Rademacher probes z (p x k) and Sigma z (from the solver).

    For each probe, z_b' (Sigma z)_b is an unbiased estimate of tr(Sigma_bb). The block's relative error is
    (mean - sum_b variances) / sum_b variances, and its standard error is the probes' own spread over sqrt(k),
    with intervals as in ``_certificate``.
    """
    per_probe = [np.sum(probes[members] * covariance_probes[members], axis=0) for members in blocks]
    estimate = np.array([float(np.sum(variances[members])) for members in blocks])
    return _certificate(estimate, per_probe, tolerance, level)


def probes_to_decide(certificate: BlockCertificate, probe_count: int) -> int:
    """The probe count that would decide every undecided block, if the standard errors fall as 1/sqrt(k).

    The quantile is held at its k-probe value, which is larger than at more probes, so this is conservative.
    """
    undecided = ~(certificate.certified | certificate.violated)
    if not np.any(undecided):
        return probe_count
    half_width = (certificate.upper_bound - certificate.lower_bound)[undecided] / 2.0
    margin = np.abs(np.abs(certificate.relative_error[undecided]) - certificate.tolerance)
    ratio = float(np.max(half_width / np.maximum(margin, np.finfo(np.float64).tiny)))
    return int(np.ceil(probe_count * ratio * ratio))


def information_products(solve: BulkSolve, back_products: NDArray[np.float64]) -> NDArray[np.float64]:
    """(D - Sigma) v on bulk sites, formed directly as D_S (Xt' w) from ``covariance_products``' back product.

    Forming D v - Sigma v by subtraction loses about log10(tr Sigma_b / tr(D - Sigma)_b) digits to
    cancellation, which is large when there is little data per variant. The back product gives the difference
    itself, so its accuracy is the dual solve's. Resolved rows are zero: the resolved sites are exact in core.
    """
    is_bulk = np.ones(solve.site_precision.shape[0], dtype=bool)
    is_bulk[solve.resolved] = False
    bulk_variance = np.where(is_bulk, 1.0 / np.where(is_bulk, solve.site_precision, 1.0), 0.0)
    return bulk_variance[:, None] * back_products


def information_solve_tolerance(
    solve: BulkSolve,
    variances: NDArray[np.float64],
    blocks: tuple[NDArray[np.int64], ...],
    column_square_norms: NDArray[np.float64],
    tolerance: float,
) -> float:
    """The relative residual |r| / |u| that the certificate's K_S solves need.

    Below it, the solve moves no block's information estimate, averaged over the probes, by more than half the
    certificate tolerance (in expectation over Rademacher z).

    For a probe z, block b's estimate is a_b' w with a_b = Xt_b D_b z_b, w = K_S^-1 u and u = Xt D_S z. A
    residual r leaves the error a_b' K_S^-1 r, and |a_b' K_S^-1 r| <= |a_b| |r|, since K_S >= I. For Rademacher
    z, exactly, E|a_b|^2 = M_b = sum_{j in b, bulk} D_j^2 |xt_j|^2 and E|u|^2 = M, the same sum over every bulk
    site. So E[|a_b| |u|] <= sqrt(M_b M) (Cauchy-Schwarz). With T_b = tr(D - Sigma)_b the block's information
    from ``variances``, a relative residual of (tolerance / 2) min_b T_b / sqrt(M_b M) suffices.
    """
    is_bulk = np.ones(solve.site_precision.shape[0], dtype=bool)
    is_bulk[solve.resolved] = False
    bulk_variance = np.where(is_bulk, 1.0 / np.where(is_bulk, solve.site_precision, 1.0), 0.0)
    mass = np.square(bulk_variance) * column_square_norms
    removed = np.where(is_bulk, bulk_variance - variances, 0.0)
    total = float(np.sum(mass))
    ratios = [float(np.sum(removed[members])) / np.sqrt(float(np.sum(mass[members])) * total) for members in blocks if float(np.sum(mass[members])) > 0.0]
    return 0.5 * tolerance * min(ratios)


def block_information_certificate(
    solve: BulkSolve,
    variances: NDArray[np.float64],
    blocks: tuple[NDArray[np.int64], ...],
    probes: NDArray[np.float64],
    removed_products: NDArray[np.float64],
    tolerance: float,
    level: float,
) -> BlockCertificate:
    """Test each block's data information tr(D_b - Sigma_bb), over its bulk sites, against probes.

    EP matches sites to the cavity P_j = 1/Sigma_jj - Pi_j. For a bulk site, Sigma_jj = D_j - D_j^2 q_j, so
    P_j = q_j / (1 - D_j q_j): what P needs is the relative accuracy of D_j - Sigma_jj, the variance the data
    removed, not of Sigma_jj itself. When D_j q_j is small (little data per variant, the production regime),
    a variance correct to 1e-3 can leave a cavity tens of percent off. ``block_trace_certificate`` cannot see
    that, and this certificate can.

    ``removed_products`` is (D - Sigma) z on bulk sites, from ``information_products``. For Rademacher z,
    z_b' ((D - Sigma) z)_b is an unbiased estimate of the block's information, tested as in
    ``block_trace_certificate``: relative error, the probes' standard error, family-wise level 1/B.
    """
    is_bulk = np.ones(solve.site_precision.shape[0], dtype=bool)
    is_bulk[solve.resolved] = False
    bulk_variance = np.where(is_bulk, 1.0 / np.where(is_bulk, solve.site_precision, 1.0), 0.0)
    removed = np.where(is_bulk, bulk_variance - variances, 0.0)
    return block_trace_certificate(removed, blocks, np.where(is_bulk[:, None], probes, 0.0), np.where(is_bulk[:, None], removed_products, 0.0), tolerance, level)


def certificate_tolerance(solve: BulkSolve, bulk_probe_count: int) -> float:
    """The certificate's tolerance when omega_S comes from k sample-side Rademacher probes.

    Hutchinson's variance is at most 2 ||K_S^-1||_F^2 / k, so the probes add relative error
    sqrt(2/k) times the equivalent's own scale. The two are independent, so they add in quadrature.
    """
    return approximation_scale(solve) * float(np.sqrt(1.0 + 2.0 / bulk_probe_count))


def covariance_products(
    solve: BulkSolve, probes: NDArray[np.float64], back_products: NDArray[np.float64], resolved_coupling: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Sigma v for variant-side vectors v (p x k), exactly, from two solver products.

    With u = Xt (D_S v) (forward pass), t = Z_L' u (sample-side, |L| x k) and
    w = K_S^-1 u - Z_L core^-1 (t - v_L), the solver returns ``back_products`` = Xt' w and
    ``resolved_coupling`` = t. Block elimination (identity 1) gives
        (Sigma v)_S = D v - D (Xt' w),   (Sigma v)_L = core^-1 (v_L - t).
    The forward product, the K_S solve and the back product ride existing passes (the certificate is lagged
    one refresh). No p x |L| array is needed.
    """
    is_resolved = np.zeros(solve.site_precision.shape[0], dtype=bool)
    is_resolved[solve.resolved] = True
    bulk_variance = np.where(is_resolved, 0.0, 1.0 / np.where(is_resolved, 1.0, solve.site_precision))
    products = bulk_variance[:, None] * (probes - back_products)
    products[solve.resolved] = np.linalg.solve(solve.resolved_core, probes[solve.resolved] - resolved_coupling)
    return products


def variance_jvp(solve: BulkSolve, grams: BlockGrams, direction: NDArray[np.float64]) -> JacobianProduct:
    """d diag(Sigma) / d Pi applied to w (p x r): -diag(Sigma diag(w) Sigma), i.e. -sum_k Sigma_jk^2 w_k.

    - **Resolved pairs:** Sigma_LL = core^-1 exactly. Sigma_Lk = -D_k (C_k core^-1)_L for bulk k whose window
      holds the resolved site, with C_k taken on k's window.
    - **Bulk pairs inside a window:** through identity 2 and the window's resolved spikes.
    - **Every pair beyond the window (bulk or resolved):** by its expected square. For blocks b and c
      beyond each other's windows, the two-block precision's off-diagonal block is the chance coupling
      X_b' Q X_c, with Q the operator in ``kernel_square_trace``. To first order,
      Sigma_bc = -Sigma_bb (X_b' Q X_c) Sigma_cc. With exchangeable rows and unlinked blocks,
          E[Sigma_jk^2] = u_j u_k tr(Q^2) / n^2,   u = diag(Sigma_bb R_b Sigma_bb) (the block sandwich),
      with resolved sites as members of their blocks. Treating the entries as Gaussian, the variance of the
      weighted sum is 2 u_j^2 (tr(Q^2)/n^2)^2 sum_k u_k^2 w_k^2.
    - **The block sandwich, not |xt_j|^2 Sigma_jj^2:** a variant's LD partners absorb most of its chance
      coupling with distant blocks.
    """
    cross, bulk_variance, core_inverse, is_resolved = _prepare(solve, grams)
    variant_count = solve.site_precision.shape[0]
    squared_variance = np.square(bulk_variance)
    sandwich = np.zeros(variant_count)
    for block, members in enumerate(grams.blocks):
        terms = _block_terms(solve, grams, cross, bulk_variance, core_inverse, block)
        sandwich[members] = np.einsum("ij,jk,ki->i", terms.covariance, grams.within[block], terms.covariance)
    pair_scale = solve.kernel_square_trace / solve.sample_count
    chance_weight = sandwich[:, None] * direction
    chance_total = chance_weight.sum(axis=0)
    chance_square_total = np.square(chance_weight).sum(axis=0)
    # Resolved rows take their resolved partners exactly from core^-1, so their chance sums run over bulk partners only.
    bulk_chance_weight = np.where(is_resolved[:, None], 0.0, chance_weight)
    block_of = _block_index(grams)
    values = np.zeros_like(direction)
    variance = np.zeros_like(direction)
    window_part = np.zeros_like(direction)
    resolved_bulk_near = np.zeros((solve.resolved.shape[0], direction.shape[1]))
    window_chance = np.zeros((len(grams.blocks), direction.shape[1]))
    window_chance_square = np.zeros((len(grams.blocks), direction.shape[1]))
    window_bulk_chance = np.zeros((len(grams.blocks), direction.shape[1]))
    window_bulk_chance_square = np.zeros((len(grams.blocks), direction.shape[1]))
    for block, members in enumerate(grams.blocks):
        window_members = np.concatenate([grams.blocks[member] for member in _window_blocks(grams, block)])
        window_chance[block] = chance_weight[window_members].sum(axis=0)
        window_chance_square[block] = np.square(chance_weight[window_members]).sum(axis=0)
        window_bulk_chance[block] = bulk_chance_weight[window_members].sum(axis=0)
        window_bulk_chance_square[block] = np.square(bulk_chance_weight[window_members]).sum(axis=0)
    for block, members in enumerate(grams.blocks):
        terms = _block_terms(solve, grams, cross, bulk_variance, core_inverse, block)
        window_sum = np.square(terms.rows) @ direction[terms.columns]
        loadings = terms.loadings  # (|b|, |L|)
        near = terms.near
        resolved_sum = squared_variance[members][:, None] * (np.square(loadings[:, near]) @ direction[solve.resolved[near]])
        row_scale = sandwich[members] * pair_scale
        chance_far = row_scale[:, None] * (chance_total - window_chance[block])[None, :]
        values[members] = -(window_sum + resolved_sum + chance_far)
        window_part[members] = window_sum
        variance[members] = 2.0 * np.square(row_scale)[:, None] * (chance_square_total - window_chance_square[block])[None, :]
        # Resolved rows: bulk partners whose window holds the resolved site, exactly.
        weighted = squared_variance[members][:, None, None] * np.square(loadings)[:, :, None] * direction[members][:, None, :]
        resolved_bulk_near[near] += weighted.sum(axis=0)[near]
    if solve.resolved.shape[0]:
        resolved_blocks = block_of[solve.resolved]
        resolved_scale = sandwich[solve.resolved] * pair_scale
        resolved_rows = np.square(core_inverse) @ direction[solve.resolved]
        bulk_total = bulk_chance_weight.sum(axis=0)
        bulk_square_total = np.square(bulk_chance_weight).sum(axis=0)
        chance_resolved = resolved_scale[:, None] * (bulk_total[None, :] - window_bulk_chance[resolved_blocks])
        values[solve.resolved] = -(resolved_rows + resolved_bulk_near + chance_resolved)
        variance[solve.resolved] = 2.0 * np.square(resolved_scale)[:, None] * (bulk_square_total[None, :] - window_bulk_chance_square[resolved_blocks])
        window_part[solve.resolved] = resolved_bulk_near
    return JacobianProduct(values=values, standard_error=np.sqrt(variance), window_part=window_part)

