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
from scipy.stats import norm


@dataclass(frozen=True)
class BulkSolve:
    """One model's refresh: every quantity is sample-side, except the p x |L| cross products.

    ``resolved_cross`` is C = Xt' K_S^-1 Xt_L, formed in the refresh's final pass. ``bulk_trace`` and
    ``bulk_square_trace`` are tr(K_S^-1)/n and tr(K_S^-2)/n, from the bulk solve's probe columns.
    ``kernel_square_trace`` is tr(Q^2)/n for Q = K_S^-1 - Z_L core^-1 Z_L' (Z_L = K_S^-1 Xt_L), the
    sample-side operator that couples any two bulk variants once the resolved sites are eliminated. Its
    probe columns are the bulk probes corrected through core: Q z = K_S^-1 z - Z_L core^-1 Xt_L' K_S^-1 z.
    """

    site_precision: NDArray[np.float64]
    resolved: NDArray[np.int64]
    resolved_core: NDArray[np.float64]
    resolved_cross: NDArray[np.float64]
    bulk_trace: float
    bulk_square_trace: float
    kernel_square_trace: float
    sample_count: int


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
    """Per block: the probe estimate of the relative error of tr(Sigma_bb), its standard error, the
    family-wise upper bound on its size, and whether it provably exceeds the approximation's scale."""

    relative_error: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    upper_bound: NDArray[np.float64]
    tolerance: float
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
    """Identity 1: diag(A^-1) from q_j = xt_j' K_S^-1 xt_j (bulk j) and the resolved core and cross products."""
    is_resolved = np.zeros(site_precision.shape[0], dtype=bool)
    is_resolved[resolved] = True
    bulk_variance = np.where(is_resolved, 0.0, 1.0 / np.where(is_resolved, 1.0, site_precision))
    spikes = np.einsum("il,lk,ik->i", resolved_cross, np.linalg.inv(resolved_core), resolved_cross) if resolved.shape[0] else 0.0
    variances = bulk_variance - np.square(bulk_variance) * (bulk_quadratic - spikes)
    variances[resolved] = np.diag(np.linalg.inv(resolved_core))
    return variances


def marginal_variances(solve: BulkSolve, grams: BlockGrams) -> NDArray[np.float64]:
    """(p,) diag(A^-1) for one model, by identities 1 and 2 (module docstring)."""
    variant_count = solve.site_precision.shape[0]
    is_resolved = np.zeros(variant_count, dtype=bool)
    is_resolved[solve.resolved] = True
    bulk_variance = np.where(is_resolved, 0.0, 1.0 / np.where(is_resolved, 1.0, solve.site_precision))
    bulk_quadratic = np.empty(variant_count)
    for block in range(len(grams.blocks)):
        gram, columns, own = _window(grams, block)
        window_variance = bulk_variance[columns]
        eigenvalues, eigenvectors = _whitened_spectrum(gram, window_variance)
        far_trace = far_field_trace(solve.bulk_trace, solve.bulk_square_trace, solve.sample_count, eigenvalues)
        quadratic = _quadratic_from_spectrum(gram, window_variance, far_trace, eigenvalues, eigenvectors)
        bulk_quadratic[grams.blocks[block]] = quadratic[own]
    return marginals_from_quadratics(solve.site_precision, solve.resolved, solve.resolved_core, solve.resolved_cross, bulk_quadratic)


def approximation_scale(solve: BulkSolve) -> float:
    """||K_S^-1||_F / tr(K_S^-1): the deterministic equivalent's relative error scale (Theorem 2)."""
    return float(np.sqrt(solve.bulk_square_trace / solve.sample_count) / solve.bulk_trace)


def block_trace_certificate(
    variances: NDArray[np.float64],
    blocks: tuple[NDArray[np.int64], ...],
    probes: NDArray[np.float64],
    covariance_probes: NDArray[np.float64],
    tolerance: float,
) -> BlockCertificate:
    """Test each block's tr(Sigma_bb) against Rademacher probes z (p x k) and Sigma z (from the solver).

    For each probe, z_b' (Sigma z)_b is an unbiased estimate of tr(Sigma_bb). The block's relative error
    is (mean - sum_b variances) / sum_b variances, and its standard error is the probes' own spread
    over sqrt(k). The bound uses the normal quantile at family-wise level 1/B over the B blocks, so
    across all blocks at most about one false flag is expected per model. A block is violated when
    even its lower bound exceeds ``tolerance``, i.e. its error provably exceeds the approximation's
    scale.
    """
    probe_count = probes.shape[1]
    quantile = float(norm.isf(0.5 / len(blocks)))
    relative = np.empty(len(blocks))
    standard = np.empty(len(blocks))
    for position, members in enumerate(blocks):
        per_probe = np.sum(probes[members] * covariance_probes[members], axis=0)
        computed = float(np.sum(variances[members]))
        relative[position] = (float(np.mean(per_probe)) - computed) / computed
        standard[position] = float(np.std(per_probe, ddof=1)) / np.sqrt(probe_count) / computed
    upper = np.abs(relative) + quantile * standard
    violated = np.abs(relative) - quantile * standard > tolerance
    return BlockCertificate(relative_error=relative, standard_error=standard, upper_bound=upper, tolerance=tolerance, violated=violated)


def certificate_tolerance(solve: BulkSolve, bulk_probe_count: int) -> float:
    """The certificate's tolerance when omega_S comes from k sample-side Rademacher probes.

    Hutchinson's variance is at most 2 ||K_S^-1||_F^2 / k, so the probes add relative error
    sqrt(2/k) times the equivalent's own scale. The two are independent, so they add in quadrature.
    """
    return approximation_scale(solve) * float(np.sqrt(1.0 + 2.0 / bulk_probe_count))


def covariance_products(solve: BulkSolve, probes: NDArray[np.float64], back_products: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sigma v for variant-side vectors v (p x k), from the solver's back products Xt' K_S^-1 Xt (D_S v).

    Block elimination (module docstring, identity 1), with C = Xt' K_S^-1 Xt_L and t = C' D v:
        (Sigma v)_S = D v - D (Xt' K_S^-1 Xt D v) + D C core^-1 (t - v_L),
        (Sigma v)_L = core^-1 (v_L - t).
    The solver pushes D_S v forward in one pass, solves K_S in the next refresh's passes, and forms the back
    product in that refresh's final pass, so the certificate costs no pass of its own.
    """
    is_resolved = np.zeros(solve.site_precision.shape[0], dtype=bool)
    is_resolved[solve.resolved] = True
    bulk_variance = np.where(is_resolved, 0.0, 1.0 / np.where(is_resolved, 1.0, solve.site_precision))
    scaled = bulk_variance[:, None] * probes
    coupling = solve.resolved_cross.T @ scaled
    resolved_part = np.linalg.solve(solve.resolved_core, coupling - probes[solve.resolved])
    products = scaled - bulk_variance[:, None] * back_products + bulk_variance[:, None] * (solve.resolved_cross @ resolved_part)
    products[solve.resolved] = -resolved_part
    return products


def _window_rows(
    solve: BulkSolve, grams: BlockGrams, bulk_variance: NDArray[np.float64], block: int
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """Sigma restricted to (block b's rows) x (its window's columns), bulk parts only, by identity 2,
    and the own block's bulk quadratics q_j = xt_j' K_S^-1 xt_j."""
    gram, columns, own = _window(grams, block)
    window_variance = bulk_variance[columns]
    eigenvalues, eigenvectors = _whitened_spectrum(gram, window_variance)
    far_trace = far_field_trace(solve.bulk_trace, solve.bulk_square_trace, solve.sample_count, eigenvalues)
    projected = (gram * np.sqrt(window_variance)[None, :]) @ eigenvectors
    quadratic = far_trace * gram[own] - far_trace**2 * (projected[own] / (1.0 + far_trace * eigenvalues)[None, :]) @ projected.T
    spikes = solve.resolved_cross[grams.blocks[block]] @ np.linalg.solve(solve.resolved_core, solve.resolved_cross[columns].T)
    own_variance = bulk_variance[grams.blocks[block]]
    rows = -own_variance[:, None] * (quadratic - spikes) * window_variance[None, :]
    rows[np.arange(own_variance.shape[0]), np.arange(own.start, own.stop)] += own_variance
    return rows, columns, quadratic[np.arange(own_variance.shape[0]), np.arange(own.start, own.stop)]


def _own_block_covariance(solve: BulkSolve, grams: BlockGrams, block: int, rows: NDArray[np.float64], columns: NDArray[np.int64],
                          bulk_variance: NDArray[np.float64], loadings: NDArray[np.float64], core_inverse: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sigma_bb in full: bulk-bulk from the window rows, bulk-resolved and resolved-resolved exactly."""
    members = grams.blocks[block]
    own = np.isin(columns, members)
    covariance = rows[:, own].copy()
    resolved_position = {variant: position for position, variant in enumerate(solve.resolved.tolist())}
    local = [(position, resolved_position[variant]) for position, variant in enumerate(members.tolist()) if variant in resolved_position]
    if local:
        member_positions = np.array([position for position, _ in local])
        resolved_positions = np.array([index for _, index in local])
        bulk_to_resolved = -bulk_variance[members][:, None] * loadings[members][:, resolved_positions]
        covariance[:, member_positions] = bulk_to_resolved
        covariance[member_positions, :] = bulk_to_resolved.T
        covariance[np.ix_(member_positions, member_positions)] = core_inverse[np.ix_(resolved_positions, resolved_positions)]
    return covariance


def variance_jvp(solve: BulkSolve, grams: BlockGrams, direction: NDArray[np.float64]) -> JacobianProduct:
    """d diag(Sigma) / d Pi applied to w (p x r): -diag(Sigma diag(w) Sigma), i.e. -sum_k Sigma_jk^2 w_k.

    Exact for every pair involving a resolved site: Sigma_LL = core^-1, and Sigma_Lk = -D_k (core^-1 C_k')
    for bulk k. Exact up to the far-field equivalent for bulk pairs inside a window (identity 2).

    Bulk pairs beyond the window. For blocks b and c beyond each other's windows, integrating the rest
    of the model out leaves a two-block precision whose off-diagonal block is the chance coupling
    X_b' Q X_c, with Q the operator in ``kernel_square_trace``. To first order in that coupling,
    Sigma_bc = -Sigma_bb (X_b' Q X_c) Sigma_cc. With exchangeable rows and unlinked blocks, the entries
    of X_b' Q X_c have covariance R_b (x) R_c tr(Q^2) / n^2, so

        E[Sigma_jk^2] = u_j u_k tr(Q^2) / n^2,   u = diag(Sigma_bb R_b Sigma_bb) (the block sandwich).

    Here, not |xt_j|^2 Sigma_jj^2: a variant's LD partners absorb most of its chance coupling, and
    only the sandwich carries that. Treating the entries as Gaussian, the variance of the weighted sum
    is 2 u_j^2 (tr(Q^2)/n^2)^2 sum_k u_k^2 w_k^2. Resolved k are excluded, since they are exact.
    """
    variant_count = solve.site_precision.shape[0]
    is_resolved = np.zeros(variant_count, dtype=bool)
    is_resolved[solve.resolved] = True
    bulk_variance = np.where(is_resolved, 0.0, 1.0 / np.where(is_resolved, 1.0, solve.site_precision))
    core_inverse = np.linalg.inv(solve.resolved_core)
    loadings = solve.resolved_cross @ core_inverse  # (p, |L|): Sigma_jL = -D_j loadings_jL (bulk j)
    squared_variance = np.square(bulk_variance)
    window_rows = []
    sandwich = np.zeros(variant_count)
    for block in range(len(grams.blocks)):
        rows, columns, _own_quadratic = _window_rows(solve, grams, bulk_variance, block)
        window_rows.append((rows, columns))
        covariance = _own_block_covariance(solve, grams, block, rows, columns, bulk_variance, loadings, core_inverse)
        sandwich[grams.blocks[block]] = np.einsum("ij,jk,ki->i", covariance, grams.within[block], covariance)
    pair_scale = solve.kernel_square_trace / solve.sample_count  # tr(Q^2) / n^2
    bulk_sandwich = np.where(is_resolved, 0.0, sandwich)
    chance_weight = bulk_sandwich[:, None] * direction
    chance_square = np.square(chance_weight)
    values = np.zeros_like(direction)
    variance = np.zeros_like(direction)
    window_part = np.zeros_like(direction)
    for block, (rows, columns) in enumerate(window_rows):
        members = grams.blocks[block]
        window_sum = np.square(rows) @ direction[columns]
        resolved_sum = squared_variance[members][:, None] * (np.square(loadings[members]) @ direction[solve.resolved])
        row_scale = bulk_sandwich[members] * pair_scale
        chance_far = row_scale[:, None] * (chance_weight.sum(axis=0) - chance_weight[columns].sum(axis=0))[None, :]
        values[members] = -(window_sum + resolved_sum + chance_far)
        window_part[members] = window_sum
        variance[members] = 2.0 * np.square(row_scale)[:, None] * (chance_square.sum(axis=0) - chance_square[columns].sum(axis=0))[None, :]
    if solve.resolved.shape[0]:
        resolved_rows = np.square(core_inverse) @ direction[solve.resolved]
        bulk_columns = np.square(loadings).T @ (squared_variance[:, None] * direction)
        values[solve.resolved] = -(resolved_rows + bulk_columns)
        variance[solve.resolved] = 0.0
        window_part[solve.resolved] = 0.0
    return JacobianProduct(values=values, standard_error=np.sqrt(variance), window_part=window_part)
