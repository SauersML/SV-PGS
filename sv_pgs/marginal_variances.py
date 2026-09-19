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
    """

    site_precision: NDArray[np.float64]
    resolved: NDArray[np.int64]
    resolved_core: NDArray[np.float64]
    resolved_cross: NDArray[np.float64]
    bulk_trace: float
    bulk_square_trace: float
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
