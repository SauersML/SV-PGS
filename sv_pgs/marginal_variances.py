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
from typing import Any
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular
from scipy.optimize import brentq
from scipy.stats import norm
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
    """Stage 0 Grams: R_b per block and R_{b,b+1} between genome-adjacent blocks, in the model's metric times
    ``scale``.

    The arrays may be Stage 0's stored float32 Grams themselves (``LdGramStore``, memory-mapped). Every model of a
    fit shares them, and a model's metric enters only through ``scale`` (1 / sigma^2 for a quantitative model):
    ``dataclasses.replace(grams, scale=...)`` copies no array. They are promoted to float64 per window, inside the
    window algebra, the only place their precision matters. ``next_cross`` has one entry per adjacent pair. An empty
    tuple makes each window its own block alone; the certificate then carries any coupling across the cuts.
    """

    blocks: tuple[NDArray[np.int64], ...]
    within: tuple[NDArray, ...]
    next_cross: tuple[NDArray, ...]
    scale: float = 1.0

    def __post_init__(self) -> None:
        if len(self.within) != len(self.blocks):
            raise ValueError("one within-block Gram per block")
        if self.next_cross and len(self.next_cross) != len(self.blocks) - 1:
            raise ValueError("next_cross needs one Gram per adjacent block pair, or none")

    def within_block(self, block: int) -> NDArray[np.float64]:
        """R_b in the model's metric, in float64 (a new array, one block at a time)."""
        return np.asarray(self.within[block], dtype=np.float64) * self.scale

    def cross_block(self, block: int) -> NDArray[np.float64]:
        """R_{b,b+1} in the model's metric, in float64."""
        return np.asarray(self.next_cross[block], dtype=np.float64) * self.scale

    def column_square_norms(self) -> NDArray[np.float64]:
        """|xt_j|^2 for every variant, in variant order: the within blocks' diagonals."""
        norms = np.empty(sum(members.shape[0] for members in self.blocks))
        for members, within in zip(self.blocks, self.within):
            norms[members] = np.diagonal(within).astype(np.float64) * self.scale
        return norms


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
    tolerance: "float | NDArray[np.float64]"
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
    """The window's Gram as Stage 0 stores it, its variant indices, and the own block's rows in it.

    Stage 0 supplies each block's Gram and the Gram with its successor, not the (b-1, b+1) corner, which is left at
    zero here: ``_whitened_window`` completes it in the model's metric, where its rounding allowance is defined.
    """
    members = _window_blocks(grams, block)
    sizes = [grams.blocks[member].shape[0] for member in members]
    starts = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    gram = np.zeros((int(starts[-1]), int(starts[-1])))
    for position, member in enumerate(members):
        span = slice(int(starts[position]), int(starts[position + 1]))
        gram[span, span] = grams.within_block(member)
        if position + 1 < len(members):
            following = slice(int(starts[position + 1]), int(starts[position + 2]))
            cross = grams.cross_block(member)
            gram[span, following] = cross
            gram[following, span] = cross.T
    columns = np.concatenate([grams.blocks[member] for member in members])
    own_position = members.index(block)
    own = slice(int(starts[own_position]), int(starts[own_position + 1]))
    return gram, columns, own


def _storage_allowance(whitened: Any, array_module: Any) -> float:
    """How far below zero a stored Gram's whitened spectrum may fall by rounding alone: u32 |B|_F.

    Stored in float32 (LdGramStore), each entry is rounded once, |E_ij| <= u32 |B_ij|, so by Weyl's inequality every
    eigenvalue moves by at most |E|_2 <= |E|_F <= u32 |B|_F.
    """
    return np.finfo(np.float32).eps / 2 * float(_to_host(array_module.linalg.norm(whitened)))


def _whitened_window(
    gram: NDArray[np.float64], window_variance: NDArray[np.float64], own: slice, block: int, array_module: Any
) -> tuple[Any, Any, Any]:
    """The window's R and B = D^1/2 R D^1/2 on ``array_module``, with the (b-1, b+1) corner completed, and D^1/2.

    The corner is the maximum-determinant positive completion of the tridiagonal pattern (Grone, Johnson, Sa and
    Wolkowicz 1984), b-1 and b+1 conditionally uncorrelated given b, taken of B + eps I with eps the two stored
    two-block windows' rounding allowance: B_{b-1,b+1} = B_{b-1,b} (B_bb + eps I)^-1 B_{b,b+1}. Each stored two-block
    window T is within eps of a Gram matrix, so T + eps I is positive semidefinite; the Schur complement of the
    completed B + eps I with respect to B_bb + eps I is then block-diagonal (the two T + eps I's Schur complements),
    and B >= -eps I, inside the guard's allowance. Without eps (e2e-scale, full chr22, blocks up to 4,009: a window at
    lambda_min -2.6e-4 against an allowance of 2.8e-5) R_bb^-1 amplifies the float32 rounding of R_{b,b+1} along R_bb's
    smallest eigenvalues, which are rounding themselves when LD is near-collinear or n < |b|. The completed corner
    is written into R as D_{b-1}^-1/2 B_{b-1,b+1} D_{b+1}^-1/2 = R_{b-1,b} D_b^1/2 (B_bb + eps I)^-1 D_b^1/2 R_{b,b+1},
    which needs no inverse of D (zero on resolved sites).
    """
    xp = array_module
    device_gram = xp.asarray(gram)
    root = xp.sqrt(xp.asarray(window_variance))
    whitened = root[:, None] * device_gram * root[None, :]
    if own.start > 0 and own.stop < whitened.shape[0]:
        first, last = slice(0, own.start), slice(own.stop, whitened.shape[0])
        allowance = max(_storage_allowance(whitened[: own.stop, : own.stop], xp), _storage_allowance(whitened[own.start :, own.start :], xp))
        centre = whitened[own, own].copy()
        centre[xp.arange(centre.shape[0]), xp.arange(centre.shape[0])] += allowance
        lower = _positive_cholesky(centre, xp)
        if lower is None:
            raise ValueError(f"block {block}: its Gram is not positive semidefinite within rounding: {_inconsistency(whitened, own, xp)}")
        del centre
        solved = _cholesky_solve(xp, lower, root[own, None] * device_gram[own, last])  # (B_bb + eps I)^-1 D_b^1/2 R_{b,b+1}
        del lower
        corner = device_gram[first, own] @ (root[own, None] * solved)
        device_gram[first, last] = corner
        device_gram[last, first] = corner.T
        whitened[first, last] = root[first, None] * corner * root[None, last]
        whitened[last, first] = whitened[first, last].T
    return device_gram, whitened, root


def _positive_cholesky(matrix: Any, array_module: Any) -> Any:
    """The lower Cholesky factor of ``matrix``, or None when it is not numerically positive definite (numpy raises;
    a device factorization leaves a non-positive pivot on the diagonal)."""
    try:
        lower = array_module.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return None
    return lower if bool(_to_host(array_module.all(array_module.diagonal(lower) > 0.0))) else None


def _inconsistency(whitened: Any, own: slice, array_module: Any) -> str:
    """The stored pieces' smallest whitened eigenvalues against their rounding allowances, for a refusal message:
    a stored piece below its allowance is inconsistent in Stage 0's store; otherwise the completion is at fault."""
    pieces = {"own block": (own.start, own.stop)}
    if own.start > 0:
        pieces["with the previous block"] = (0, own.stop)
    if own.stop < whitened.shape[0]:
        pieces["with the next block"] = (own.start, whitened.shape[0])
    reports = []
    for name, (start, stop) in pieces.items():
        piece = whitened[start:stop, start:stop]
        smallest = float(_to_host(array_module.linalg.eigvalsh(piece))[0])
        reports.append(f"{name} {smallest:.3e} (allows {-_storage_allowance(piece, array_module):.3e})")
    return "; ".join(reports)


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
    spikes = np.sum((resolved_cross @ np.linalg.inv(resolved_core)) * resolved_cross, axis=1) if resolved.shape[0] else 0.0
    variances = bulk_variance - np.square(bulk_variance) * (bulk_quadratic - spikes)
    variances[resolved] = np.diag(np.linalg.inv(resolved_core))
    return variances


def sandwich_diagonal(covariance: NDArray[np.float64], gram: NDArray[np.float64]) -> NDArray[np.float64]:
    """diag(S R S) as sum_k (S R)_ik S_ki: one BLAS product and an elementwise sum.

    Unoptimized three-operand einsum loops over all (i, j, k) without BLAS; on one 4,430-variant block it was 99.8%
    of a fit's time (engine's bench-real profile).
    """
    return np.sum((covariance @ gram) * covariance.T, axis=1)


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


def _window_quadratic(
    gram: NDArray[np.float64], window_variance: NDArray[np.float64], own: slice, block: int, solve: BulkSolve, array_module: Any
) -> NDArray[np.float64]:
    """The window's bulk quadratic rows (own x window): (omega_F R - omega_F^2 R D^1/2 (I + omega_F B)^-1 D^1/2 R)[own].

    B's spectrum is needed only for omega_F, and (I + omega_F B)^-1 only on the own block's columns: eigenvalues
    (about 4/3 |W|^3), one Cholesky (|W|^3 / 3) and solves for |b| columns, not a full eigendecomposition (about
    9 |W|^3; e2e-scale: 37 min per model at |W| up to 3 x 2,144 on host). The dense algebra runs on
    ``array_module`` (numpy, or cupy on a device), in float64; omega_F's scalar root is found on host.
    """
    xp = array_module
    device_gram, whitened, root = _whitened_window(gram, window_variance, own, block, xp)
    raw_eigenvalues = _to_host(xp.linalg.eigvalsh(whitened))
    # A window Gram must be positive semidefinite, and then every quadratic here is >= 0: omega R (I + omega D R)^-1
    # = omega R^1/2 (I + omega R^1/2 D R^1/2)^-1 R^1/2. The stored pieces' rounding and the completion keep B above
    # -u32 |B|_F (``_whitened_window``), and the float64 eigensolver adds |W| u64 |B|_F. Anything beyond that means
    # the within- and cross-block Grams are not the Gram of one design (different rows, units, projection or block
    # order): the map would be wrong, so refuse.
    rounding = _storage_allowance(whitened, xp) + whitened.shape[0] * np.finfo(np.float64).eps / 2 * float(_to_host(xp.linalg.norm(whitened)))
    if raw_eigenvalues.shape[0] and float(raw_eigenvalues[0]) < -rounding:
        raise ValueError(
            f"block {block}: a window Gram is not positive semidefinite (smallest whitened eigenvalue "
            f"{float(raw_eigenvalues[0]):.3e}, rounding allows {-rounding:.3e}): the within- and cross-block Grams do "
            f"not come from one design; {_inconsistency(whitened, own, xp)}"
        )
    eigenvalues = np.maximum(raw_eigenvalues, 0.0)
    far_trace = far_field_trace(solve.bulk_trace, solve.bulk_square_trace, solve.sample_count, eigenvalues)
    whitened *= far_trace  # in place: I + omega_F B, with no second |W| x |W| array
    whitened[xp.arange(whitened.shape[0]), xp.arange(whitened.shape[0])] += 1.0
    lower = xp.linalg.cholesky(whitened)
    del whitened
    right = root[:, None] * device_gram[:, own]  # D^1/2 R[:, own]
    solved = _cholesky_solve(xp, lower, right)  # (I + omega_F B)^-1 D^1/2 R[:, own]
    del lower
    quadratic = far_trace * device_gram[:, own] - far_trace**2 * (device_gram @ (root[:, None] * solved))
    return _to_host(quadratic.T)


def window_working_bytes(grams: BlockGrams) -> int:
    """The window algebra's peak float64 working set over the blocks, for the fit's working_bytes budget.

    At most four |W| x |W| arrays are live in a window (the assembled Gram, its whitened form, the eigensolver's
    workspace, the Cholesky factor) plus four |W| x |b| ones (the right-hand sides, their solve, the quadratic
    rows and the covariance rows), with |W| the window's and |b| the block's size. The shared Grams themselves stay
    where the caller keeps them (Stage 0's memory map).
    """
    itemsize = np.dtype(np.float64).itemsize
    peak = 0
    for block in range(len(grams.blocks)):
        window = sum(grams.blocks[member].shape[0] for member in _window_blocks(grams, block))
        own = grams.blocks[block].shape[0]
        peak = max(peak, (4 * window * window + 4 * window * own) * itemsize)
    return peak


def _to_host(values: Any) -> NDArray[np.float64]:
    return values.get() if hasattr(values, "get") else np.asarray(values)


def _cholesky_solve(xp: Any, lower: Any, right: Any) -> Any:
    """(L L')^-1 right by two triangular solves, on numpy or cupy."""
    if xp is np:
        return solve_triangular(lower.T, solve_triangular(lower, right, lower=True), lower=False)
    from cupyx.scipy.linalg import solve_triangular as device_triangular

    return device_triangular(lower.T, device_triangular(lower, right, lower=True), lower=False)


def _block_terms(
    solve: BulkSolve,
    grams: BlockGrams,
    cross: WindowCross,
    bulk_variance: NDArray[np.float64],
    core_inverse: NDArray[np.float64],
    block: int,
    array_module: Any = np,
) -> _BlockTerms:
    gram, columns, own = _window(grams, block)
    members = grams.blocks[block]
    window_variance = bulk_variance[columns]
    quadratic = _window_quadratic(gram, window_variance, own, block, solve, array_module)
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


@dataclass(frozen=True)
class WindowPreparation:
    """What every block's window algebra shares for one model: C on the LD windows, D on bulk sites, core^-1 and
    the resolved mask. Build it once with ``prepare_windows`` and pass it to ``block_covariance`` for many blocks."""

    cross: WindowCross
    bulk_variance: NDArray[np.float64]
    core_inverse: NDArray[np.float64]
    is_resolved: NDArray[np.bool_]


def prepare_windows(solve: BulkSolve, grams: BlockGrams) -> WindowPreparation:
    """The shared state of every block's window algebra for one model (one core inversion, one window_cross)."""
    cross, bulk_variance, core_inverse, is_resolved = _prepare(solve, grams)
    return WindowPreparation(cross=cross, bulk_variance=bulk_variance, core_inverse=core_inverse, is_resolved=is_resolved)


def block_covariance(
    solve: BulkSolve, grams: BlockGrams, block: int, array_module: Any = np, prepared: WindowPreparation | None = None
) -> NDArray[np.float64]:
    """Sigma_bb for one block, (|b| x |b|) host float64: the window algebra's own-block covariance.

    Bulk-bulk entries come from identity 2 and the window's resolved spikes; bulk-resolved and resolved-resolved
    entries are exact through core. Resolved sites beyond the window do not enter (their coupling to block b is a
    chance term, which ``marginal_variances`` adds to the diagonal by its expectation). ``grams`` may be Stage 0's
    shared float32 Grams with the model's ``scale``; each window is promoted to float64. The window algebra runs on
    ``array_module`` (cupy for a device). ``prepared`` (``prepare_windows``) saves recomputing the model's shared
    state for every block.
    """
    state = prepared if prepared is not None else prepare_windows(solve, grams)
    terms = _block_terms(solve, grams, state.cross, state.bulk_variance, state.core_inverse, block, array_module)
    covariance = terms.covariance
    return 0.5 * (covariance + covariance.T)


def marginal_variances(solve: BulkSolve, grams: BlockGrams, array_module: Any = np) -> NDArray[np.float64]:
    """(p,) diag(A^-1) for one model, by identities 1 and 2 (module docstring). The windows' dense algebra runs on
    ``array_module`` (numpy by default; pass cupy to use a device).

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
        terms = _block_terms(solve, grams, cross, bulk_variance, core_inverse, block, array_module)
        near_variance[members] = np.diag(terms.covariance)
        sandwich[members] = sandwich_diagonal(terms.covariance, grams.within_block(block))
    resolved_weight = sandwich[solve.resolved] / resolved_variance if solve.resolved.shape[0] else np.zeros(0)
    for block in range(len(grams.blocks)):
        near_totals[block] = float(np.sum(resolved_weight[cross.positions[block]]))
    far_scale = solve.kernel_square_trace / solve.sample_count
    block_of = _block_index(grams)
    far = sandwich * far_scale * (float(np.sum(resolved_weight)) - near_totals[block_of])
    variances = np.where(is_resolved, near_variance, near_variance + far)
    # Every marginal obeys 1 / A_jj <= Sigma_jj (Cauchy-Schwarz, for any positive-definite A). The upper bound
    # Sigma_jj <= 1 / Pi_j holds only when every site precision is positive: A >= diag(Pi) gives
    # A^-1 <= diag(Pi)^-1 only for diag(Pi) > 0, and a non-positive resolved site breaks it, since the bulk block
    # of Sigma^-1 is then Pi_S + Xt_S' (I + Xt_L Pi_L^-1 Xt_L')^-1 Xt_S with an indefinite middle factor
    # (verify-stage2's counterexample: Xt'Xt = [[1, 1], [1, 1]], Pi = (2, -1/2) gives Sigma_11 = 1 > 1/2). The
    # approximation can cross the bounds, e.g. when a window's resolved spikes over-subtract, so project onto
    # the ones that hold, as exact_polish does for its estimates. The certificate still sees the error.
    column_square_norms = grams.column_square_norms()
    lower = 1.0 / (column_square_norms + solve.site_precision)
    every_site_positive = bool(np.all(solve.site_precision > 0.0))
    upper = bulk_variance if every_site_positive else np.full(variant_count, np.inf)
    variances = np.where(is_resolved, variances, np.clip(variances, lower, upper))
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


def _edgeworth_correction(skewness: float, probe_count: int, cut: float) -> float:
    """The one-term Edgeworth correction to the studentized mean's tail beyond ``cut``, on its heavy side.

    For the mean of k i.i.d. values of skewness g, T = (mean - mu) / (sd / sqrt k) has
    P(T <= x) = Phi(x) + k^(-1/2) g (2 x^2 + 1) phi(x) / 6 + O(1/k) (Hall 1992, The Bootstrap and Edgeworth
    Expansion, section 2.6), so the tail on the side the skewness weights gains |g| (2 x^2 + 1) phi(x) / (6 sqrt k).
    """
    return abs(skewness) * (2.0 * cut * cut + 1.0) * float(norm.pdf(cut)) / (6.0 * np.sqrt(probe_count))


def _heavy_cut(skewness: float, probe_count: int, side: float, quantile: float) -> tuple[float, bool]:
    """The heavy tail's cut x >= ``quantile`` whose corrected tail t_{k-1}(x) + correction(x) spends ``side``, and
    whether the expansion is usable there: its first-order term below its zeroth-order term, the normal tail
    beyond x (Hall's expansion is about Phi; the Student-t base only adds the O(1/k) term exact for Gaussian values).

    Both terms fall as x grows past the t quantile, so the cut is bracketed by doubling from it. With zero
    skewness it is the t quantile itself: the certificate is unchanged for symmetric probe values.
    [sim-only, review-stats stopgap_sim2: comparing against the Student-t tail instead left rank-1 blocks at k = 16
    missing 4x the level.]
    """
    def excess(cut: float) -> float:
        return float(student_t.sf(cut, probe_count - 1)) + _edgeworth_correction(skewness, probe_count, cut) - side

    if excess(quantile) <= 0.0:
        cut = quantile
    else:
        upper = 2.0 * quantile
        while excess(upper) > 0.0:
            upper *= 2.0
        cut = brentq(excess, quantile, upper)
    return cut, _edgeworth_correction(skewness, probe_count, cut) < float(norm.sf(cut))


def _certificate(
    estimate: NDArray[np.float64],
    per_probe: list[NDArray[np.float64]],
    tolerance: "float | NDArray[np.float64]",
    level: float,
    skewness: "NDArray[np.float64] | None" = None,
) -> BlockCertificate:
    """Intervals from k probe values per block, at family-wise ``level`` over the B blocks (Bonferroni, two-sided).

    (mean - estimate) / (sd / sqrt k) is referred to Student's t with k - 1 degrees of freedom, which is exact for
    Gaussian probe values. A block's probe value is a Rademacher quadratic form, and a block whose matrix has low
    effective rank (strong LD: a few directions carry it) gives skewed values, close to tr * chi2_r / r. The
    studentized mean of skewed values has a heavy tail on one side: a draw that lacks the rare large values has a
    small mean and a small sd together. So the heavy side's cut is moved out by the one-term Edgeworth correction
    (``_heavy_cut``, from the probes' own sample skewness) until that side spends its share of the level, and the
    light side keeps the t quantile. The interval only widens, so a block can move from certified or violated to
    undecided, never between the two. Where the expansion is not usable at the cut (its correction at least the
    term it corrects), the block is undecided: k probes cannot place that tail, and more probes (or the exact
    diagonal) must decide it. [sim-only, review-stats certsim: with the plain t cut, rank-1 blocks at k = 16 missed
    on the heavy side 9.5x the nominal rate for one block and ~280x under Bonferroni over 144 blocks.]

    ``skewness``, when given, is each block's probe-value skewness known from structure (e.g. the exact Rademacher
    cumulants of a deflated remainder's proxy); otherwise the probes' own sample skewness is used, which is
    conservative only through the undecided rule, because a draw lacking the large values also looks less skewed.

    A block whose estimate is zero has no finite relative error. With zero probe spread too (every site resolved,
    so the block is exact), it is certified. Otherwise the probes see information the estimate says is absent
    (e.g. a marginal clamped to its bound), so the relative error is infinite: the block is violated when the
    probes' own interval for the absolute value excludes zero, and undecided when it does not.
    """
    block_count = len(per_probe)
    probe_count = per_probe[0].shape[0]
    side = 0.5 * level / block_count
    quantile = float(student_t.isf(side, probe_count - 1))
    relative = np.zeros(block_count)
    standard = np.zeros(block_count)
    below = np.full(block_count, quantile)
    above = np.full(block_count, quantile)
    unusable = np.zeros(block_count, dtype=bool)
    unresolved_zero = np.zeros(block_count, dtype=bool)
    excludes_zero = np.zeros(block_count, dtype=bool)
    for position, values in enumerate(per_probe):
        computed = float(estimate[position])
        spread = float(np.std(values, ddof=1)) / np.sqrt(probe_count)
        if computed == 0.0:
            if spread == 0.0:
                continue
            centre = float(np.mean(values))
            unresolved_zero[position] = True
            excludes_zero[position] = abs(centre) > quantile * spread
            relative[position] = np.copysign(np.inf, centre)
            standard[position] = np.inf
            continue
        relative[position] = (float(np.mean(values)) - computed) / computed
        standard[position] = spread / abs(computed)
        if skewness is None:
            centred = values - float(np.mean(values))
            second = float(np.mean(centred * centred))
            block_skewness = float(np.mean(centred**3)) / second**1.5 if second > 0.0 else 0.0
        else:
            block_skewness = float(skewness[position])
        heavy, usable = _heavy_cut(block_skewness, probe_count, side, quantile)
        unusable[position] = not usable
        # Right-skewed values make the studentized mean's lower tail the heavy one: the truth lies above the mean more
        # often than t says, so the mean's interval moves out above (and below for left skew).
        mean_above, mean_below = (heavy, quantile) if block_skewness > 0.0 else (quantile, heavy)
        # The relative error (mean - computed) / computed falls with the mean when the estimate is negative.
        below[position], above[position] = (mean_below, mean_above) if computed > 0.0 else (mean_above, mean_below)
    with np.errstate(invalid="ignore"):
        lower = np.where(unresolved_zero, np.where(excludes_zero, relative, -np.inf), relative - below * standard)
        upper = np.where(unresolved_zero, np.where(excludes_zero, relative, np.inf), relative + above * standard)
    bound = np.broadcast_to(np.asarray(tolerance, dtype=np.float64), relative.shape)
    certified = (lower >= -bound) & (upper <= bound) & ~unusable
    violated = ((lower > bound) | (upper < -bound)) & ~unusable
    return BlockCertificate(
        relative_error=relative, standard_error=standard, lower_bound=lower, upper_bound=upper,
        tolerance=tolerance, level=level, certified=certified, violated=violated,
    )


def block_trace_certificate(
    variances: NDArray[np.float64],
    blocks: tuple[NDArray[np.int64], ...],
    probes: NDArray[np.float64],
    covariance_probes: NDArray[np.float64],
    tolerance: "float | NDArray[np.float64]",
    level: float,
    skewness: "NDArray[np.float64] | None" = None,
) -> BlockCertificate:
    """Test each block's tr(Sigma_bb) against Rademacher probes z (p x k) and Sigma z (from the solver).

    For each probe, z_b' (Sigma z)_b is an unbiased estimate of tr(Sigma_bb). The block's relative error is
    (mean - sum_b variances) / sum_b variances, and its standard error is the probes' own spread over sqrt(k),
    with intervals as in ``_certificate``.
    """
    per_probe = [np.sum(probes[members] * covariance_probes[members], axis=0) for members in blocks]
    estimate = np.array([float(np.sum(variances[members])) for members in blocks])
    return _certificate(estimate, per_probe, tolerance, level, skewness)


def stage_level(level: float, stage: int) -> float:
    """The level for stage s = 0, 1, ... of a sequential certificate with fresh probes at each stage.

    Stage s spends level * 2^-(s+1), so the stages together never exceed ``level`` (Bonferroni over stages). A
    stage re-tests only the blocks the last one left undecided. A violation at any stage is decisive, so the
    probability of refusing a fixed point whose blocks are all within tolerance is at most ``level`` overall.
    """
    return level * 2.0 ** -(stage + 1)


def probes_to_decide(certificate: BlockCertificate, probe_count: int) -> int:
    """The probe count that would decide every undecided block, if the standard errors fall as 1/sqrt(k).

    The quantile is held at its k-probe value, which is larger than at more probes, so this is conservative.
    """
    undecided = ~(certificate.certified | certificate.violated)
    if not np.any(undecided):
        return probe_count
    half_width = (certificate.upper_bound - certificate.lower_bound)[undecided] / 2.0
    tolerance = np.broadcast_to(np.asarray(certificate.tolerance, dtype=np.float64), certificate.relative_error.shape)
    margin = np.abs(np.abs(certificate.relative_error[undecided]) - tolerance[undecided])
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
    tolerance: "float | NDArray[np.float64]",
) -> float:
    """The relative residual |r| / |u| that the certificate's K_S solves need.

    Below it, the solve moves no block's information estimate, averaged over the probes, by more than half the
    certificate tolerance (in expectation over Rademacher z).

    For a probe z, block b's estimate is a_b' w with a_b = Xt_b D_b z_b, w = K_S^-1 u and u = Xt D_S z. A
    residual r leaves the error a_b' K_S^-1 r, and |a_b' K_S^-1 r| <= |a_b| |r|, since K_S >= I. For Rademacher
    z, exactly, E|a_b|^2 = M_b = sum_{j in b, bulk} D_j^2 |xt_j|^2 and E|u|^2 = M, the same sum over every bulk
    site. So E[|a_b| |u|] <= sqrt(M_b M) (Cauchy-Schwarz). With T_b = tr(D - Sigma)_b the block's information
    from ``variances``, a relative residual of min_b (tolerance_b / 2) |T_b| / sqrt(M_b M) suffices (with the ceiling
    U_b in place of a zero T_b). It is +infinity only when no block has bulk mass (every site resolved): then no
    solve is needed at all.
    """
    is_bulk = np.ones(solve.site_precision.shape[0], dtype=bool)
    is_bulk[solve.resolved] = False
    bulk_variance = np.where(is_bulk, 1.0 / np.where(is_bulk, solve.site_precision, 1.0), 0.0)
    mass = np.square(bulk_variance) * column_square_norms
    removed = np.where(is_bulk, bulk_variance - variances, 0.0)
    total = float(np.sum(mass))
    bound = np.broadcast_to(np.asarray(tolerance, dtype=np.float64), (len(blocks),))
    # The certificate tests (probe mean - T_b) / |T_b|, so the solve must hold its move below (tolerance / 2) |T_b|
    # for the estimate T_b it is tested against, whatever its sign; a non-positive-site model can make it negative.
    # A zero estimate is decided on the probes' absolute value (_certificate), so its scale is the largest
    # information the block can hold, U_b = sum_{j in b, bulk} (D_j - 1/A_jj), since Sigma_jj >= 1/A_jj for any
    # positive-definite A. U_b > 0 whenever the block has bulk mass. Only a block with no bulk mass (every site
    # resolved, so exact) constrains nothing; with none left, the minimum over the empty set is +infinity.
    ceiling = information_ceiling(solve, blocks, column_square_norms)
    resolvable = resolvable_blocks(solve, blocks, ceiling)
    ratios = []
    for position, members in enumerate(blocks):
        # A block whose ceiling float64 cannot resolve holds no information the arithmetic can see, whatever
        # rounding-level column mass it carries (a column that is zero on the fold's training rows): it is
        # exact, and constrains nothing (``resolvable_blocks``).
        if not resolvable[position]:
            continue
        estimate = abs(float(np.sum(removed[members])))
        scale = estimate if estimate > 0.0 else float(ceiling[position])
        ratios.append(float(bound[position]) * scale / np.sqrt(float(np.sum(mass[members])) * total))
    return 0.5 * min(ratios) if ratios else np.inf


def resolvable_blocks(solve: BulkSolve, blocks: tuple[NDArray[np.int64], ...], ceiling: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Blocks whose information ceiling U_b exceeds the rounding of their own information in float64.

    The information sum_{j in b} (D_j - Sigma_jj) over m bulk sites is computed from values of size D_j, so its
    rounding is at most gamma_m sum_{j in b} D_j, with gamma_m = m u / (1 - m u) and u the unit roundoff
    (Higham, Thm 4.3). A block with U_b at or below that holds no information float64 can resolve.
    """
    is_bulk = np.ones(solve.site_precision.shape[0], dtype=bool)
    is_bulk[solve.resolved] = False
    prior = np.where(is_bulk, 1.0 / np.where(is_bulk, solve.site_precision, 1.0), 0.0)
    unit = np.finfo(np.float64).eps / 2.0
    out = np.zeros(len(blocks), dtype=bool)
    for position, members in enumerate(blocks):
        count = int(np.sum(is_bulk[members]))
        rounding = count * unit / (1.0 - count * unit) * float(np.sum(prior[members]))
        out[position] = ceiling[position] > rounding
    return out


def information_ceiling(solve: BulkSolve, blocks: tuple[NDArray[np.int64], ...], column_square_norms: NDArray[np.float64]) -> NDArray[np.float64]:
    """U_b = sum over block b's bulk sites of D_j - 1/A_jj: the most information the block can hold.

    Sigma_jj >= 1/A_jj for any positive-definite A (Cauchy-Schwarz), with A_jj = |xt_j|^2 + Pi_j, so
    D_j - Sigma_jj <= D_j - 1/A_jj. U_b is zero exactly when the block has no bulk site with a column that
    float64 can tell from zero next to its site precision.
    """
    is_bulk = np.ones(solve.site_precision.shape[0], dtype=bool)
    is_bulk[solve.resolved] = False
    precision = np.where(is_bulk, solve.site_precision, 1.0)
    per_site = np.where(is_bulk, 1.0 / precision - 1.0 / (column_square_norms + precision), 0.0)
    return np.array([float(np.sum(per_site[members])) for members in blocks])


def block_information_certificate(
    solve: BulkSolve,
    variances: NDArray[np.float64],
    blocks: tuple[NDArray[np.int64], ...],
    probes: NDArray[np.float64],
    removed_products: NDArray[np.float64],
    tolerance: "float | NDArray[np.float64]",
    level: float,
    control: ControlVariate,
    skewness: "NDArray[np.float64] | None" = None,
) -> BlockCertificate:
    """Test each block's data information tr(D_b - Sigma_bb), over its bulk sites, against probes.

    EP matches sites to the cavity P_j = 1/Sigma_jj - Pi_j. For a bulk site, Sigma_jj = D_j - D_j^2 q_j, so
    P_j = q_j / (1 - D_j q_j): what P needs is the relative accuracy of D_j - Sigma_jj, the variance the data
    removed, not of Sigma_jj itself. When D_j q_j is small (little data per variant, the production regime),
    a variance correct to 1e-3 can leave a cavity tens of percent off. ``block_trace_certificate`` cannot see
    that, and this certificate can.

    ``removed_products`` is (D - Sigma) z on bulk sites, from ``information_products``. The estimator is the
    control-variate Hutchinson identity,

        T_b = tr(D - Sigma_hat)_b + E_z[ z_b' ((D - Sigma) z - (D - Sigma_hat) z)_b ],

    with Sigma_hat the window approximation (``control_variate``). Without the control variate, the probe
    values' spread is set by Sigma's off-diagonal LD mass, which dwarfs the small diagonal D - Sigma: on a
    synthetic LD store (engine's test), 16 probes gave standard errors of 2-6% on errors below 3.3%. With it,
    the spread comes from Sigma_hat - Sigma only, which is small exactly when the approximation is right.
    The test is ``_certificate``'s.
    """
    if control.window_information.shape[0] != len(blocks):
        raise ValueError("the control variate was built on different blocks")
    is_bulk = np.ones(solve.site_precision.shape[0], dtype=bool)
    is_bulk[solve.resolved] = False
    bulk_variance = np.where(is_bulk, 1.0 / np.where(is_bulk, solve.site_precision, 1.0), 0.0)
    removed = np.where(is_bulk, bulk_variance - variances, 0.0)
    difference = np.where(is_bulk[:, None], removed_products - control.removed_products, 0.0)
    # A block that can hold no resolvable information (``resolvable_blocks``) is exact: test it as such.
    per_probe = [
        control.window_information[position] + np.sum(probes[members] * difference[members], axis=0)
        if control.resolvable[position] else np.zeros(probes.shape[1])
        for position, members in enumerate(blocks)
    ]
    estimate = np.array([float(np.sum(removed[members])) if control.resolvable[position] else 0.0 for position, members in enumerate(blocks)])
    return _certificate(estimate, per_probe, tolerance, level, skewness)


@dataclass(frozen=True)
class ControlVariate:
    """The window approximation's own (D - Sigma_hat) z on bulk rows, and each block's window information.

    Subtracting it from the solver's exact (D - Sigma) z leaves an estimator of the approximation's error whose
    spread comes only from (Sigma_hat - Sigma), not from Sigma's off-diagonal LD mass. ``resolvable`` marks the
    blocks float64 can resolve (``resolvable_blocks``); the others are exact and certified as such.
    """

    removed_products: NDArray[np.float64]
    window_information: NDArray[np.float64]
    resolvable: NDArray[np.bool_]


def control_variate(solve: BulkSolve, grams: BlockGrams, probes: NDArray[np.float64], array_module: Any = np) -> ControlVariate:
    """(D - Sigma_hat) z on bulk rows, with Sigma_hat the window approximation, and each block's tr(D - Sigma_hat).

    For bulk j in block b, Sigma_hat_{j,:} z = sum over window columns (identity 2 plus the window's resolved
    spikes) + Sigma_hat_{j,L} z_L, with Sigma_hat_{jL} = -D_j (C core^-1)_jL from the window's cross products.
    The window information is taken on those window diagonals, so the certificate's estimator has the expectation
    of the exact information.
    """
    cross, bulk_variance, core_inverse, is_resolved = _prepare(solve, grams)
    removed = np.zeros_like(probes)
    information = np.zeros(len(grams.blocks))
    column_square_norms = grams.column_square_norms()
    resolvable = resolvable_blocks(solve, grams.blocks, information_ceiling(solve, grams.blocks, column_square_norms))
    for block, members in enumerate(grams.blocks):
        terms = _block_terms(solve, grams, cross, bulk_variance, core_inverse, block, array_module)
        own_variance = bulk_variance[members]
        products = terms.rows @ probes[terms.columns] - own_variance[:, None] * (terms.loadings @ probes[solve.resolved])
        bulk_rows = ~is_resolved[members]
        removed[members] = np.where(bulk_rows[:, None], own_variance[:, None] * probes[members] - products, 0.0)
        information[block] = float(np.sum(np.where(bulk_rows, own_variance - np.diag(terms.covariance), 0.0)))
    return ControlVariate(removed_products=removed, window_information=information, resolvable=resolvable)


def cavity_tolerance(
    site_precision: NDArray[np.float64],
    variances: NDArray[np.float64],
    tilted_response: NDArray[np.float64],
    tilted_skewness: NDArray[np.float64],
    blocks: tuple[NDArray[np.int64], ...],
    draw_count: int,
    effective_parameters: float,
) -> NDArray[np.float64]:
    """Per block: the relative error of tr(D - Sigma)_b that the EP fixed point cannot see through K draws.

    **What an information error does.** A relative error eps_j in site j's information I_j = D_j - Sigma_jj
    is a relative variance error e_j = -eps_j w_j / (1 - w_j), where w_j = P_j Sigma_jj is the data share and
    I_j / Sigma_jj = P_j / tau_j = w_j / (1 - w_j). The decoupled EP freezes the variances in the cavities,
    so its cavity moves by -e_j / Sigma_jj, amplified by 1/w_j. But the site responds only through the tilted
    law's non-Gaussianity, and at the fixed point Sigma_jj (P_j + tau_j) = 1 cancels the 1/w_j. To first order:

        delta Sigma_jj / Sigma_jj = r_j e_j,   |delta mu|^2_{Sigma^-1} = sum_j (gamma_j e_j / 2)^2,

    with r_j = d tau_j / d P_j = (kappa4 / 2 + m kappa3) / V^2 (``tilted_response``) and gamma_j = kappa3 / V^1.5
    (``tilted_skewness``), the tilted cumulants at the cavity. Measured against exact perturbed fixed points
    [sim-only: AR(1) LD, mixture priors, n = 400-600]: correlation 0.93-0.998 with the variance prediction,
    slope 0.59-0.98 (the prediction is the larger), the mean prediction within 5%, and improper cavities
    exactly where predicted.

    **The tolerance** is the smallest of three:
    - variance channel: sqrt(2/K) posterior sds of a variance is invisible to K draws (ep_eb.md §3.3). With a
      uniform eps over the block, that is eps_b <= sqrt(2/K) / rms_{j in b}(|r_j| w_j / (1 - w_j));
    - mean channel: |delta mu|^2 <= p_eff / K, split evenly over the model, is
      eps <= sqrt(p_eff / K) / sqrt(sum_j (gamma_j w_j / (2 (1 - w_j)))^2), for every block;
    - properness: the cavity stays proper iff the estimated information stays positive, eps_j < 1.
    Resolved sites are exact in core and are excluded.
    """
    data_share = 1.0 - site_precision * variances
    ratio = data_share / (1.0 - data_share)
    is_bulk = (site_precision > 0.0) & np.isfinite(ratio)
    variance_weight = np.where(is_bulk, np.abs(tilted_response) * ratio, 0.0)
    mean_weight = np.where(is_bulk, 0.5 * tilted_skewness * ratio, 0.0)
    mean_total = float(np.sqrt(np.sum(np.square(mean_weight))))
    mean_bound = float(np.sqrt(effective_parameters / draw_count)) / mean_total if mean_total > 0.0 else np.inf
    variance_limit = float(np.sqrt(2.0 / draw_count))
    tolerance = np.empty(len(blocks))
    for position, members in enumerate(blocks):
        weights = variance_weight[members][is_bulk[members]]
        spread = float(np.sqrt(np.mean(np.square(weights)))) if weights.shape[0] else 0.0
        variance_bound = variance_limit / spread if spread > 0.0 else np.inf
        tolerance[position] = min(variance_bound, mean_bound, 1.0)
    return tolerance


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


def variance_jvp(solve: BulkSolve, grams: BlockGrams, direction: NDArray[np.float64], array_module: Any = np) -> JacobianProduct:
    """d diag(Sigma) / d Pi applied to w (p x r): -diag(Sigma diag(w) Sigma), i.e. -sum_k Sigma_jk^2 w_k.
    The windows' dense algebra runs on ``array_module`` (numpy by default; pass cupy to use a device).

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
        terms = _block_terms(solve, grams, cross, bulk_variance, core_inverse, block, array_module)
        sandwich[members] = sandwich_diagonal(terms.covariance, grams.within_block(block))
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
        terms = _block_terms(solve, grams, cross, bulk_variance, core_inverse, block, array_module)
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
        resolved_bulk_near[near] += (np.square(loadings).T @ (squared_variance[members][:, None] * direction[members]))[near]
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




# --------------------------------------------------------------------------- the exact dual route (small n)


@dataclass(frozen=True)
class KernelFactor:
    """The bulk kernel K_S = I + Xt_S D_S Xt_S' factored densely: for shapes whose n x n fits in memory.

    ``lower`` is its Cholesky factor, ``resolved_solves`` Z_L = K_S^-1 Xt_L, and ``resolved_core``
    core = Pi_L + Xt_L' Z_L. Then Q = K^-1 = K_S^-1 - Z_L core^-1 Z_L' (Woodbury, valid for any core > 0).
    """

    lower: NDArray[np.float64]
    resolved_solves: NDArray[np.float64]
    resolved_core: NDArray[np.float64]

    @cached_property
    def core_inverse(self) -> NDArray[np.float64]:
        """core^-1, formed once per factor and shared by every block's call (speed-krylov)."""
        return np.linalg.inv(self.resolved_core) if self.resolved_core.shape[0] else np.zeros((0, 0))


def exact_block_information(factor: KernelFactor, bulk_variance: NDArray[np.float64], columns: NDArray[np.float64]) -> NDArray[np.float64]:
    """D_j - Sigma_jj for one block's variants, exactly: D_j^2 (xt_j' K_S^-1 xt_j - c_j core^-1 c_j').

    ``columns`` is the block's Xt_b (n x |b|) and ``bulk_variance`` its D_b (zero on resolved sites, whose rows
    the caller takes from core^-1). The cost is n^2 |b| for the triangular solve plus n |b| |L|.
    """
    whitened = solve_triangular(factor.lower, columns, lower=True)
    quadratic = np.sum(np.square(whitened), axis=0)
    cross = columns.T @ factor.resolved_solves
    spikes = np.sum((cross @ factor.core_inverse) * cross, axis=1) if cross.shape[1] else 0.0
    return np.square(bulk_variance) * (quadratic - spikes)


def exact_bulk_diagonal(factor: KernelFactor) -> NDArray[np.float64]:
    """diag(K_S^-1), exactly: the column sums of squares of L^-1, with K_S = L L'. The cost is n^3 / 3.

    This is the bulk diagonal only. The resolved sites' correction (Z_L core^-1 Z_L')_ii and the covariate
    leverage are added once, by ``dual_solve.SampleDiagonal.predictor_variance``, the single assembler for the
    exact and the windowed routes alike (binary-ep). Subtracting them here too would count them twice.
    """
    inverse_lower = solve_triangular(factor.lower, np.eye(factor.lower.shape[0]), lower=True)
    return np.sum(np.square(inverse_lower), axis=0)


def exact_route_is_cheaper(sample_count: int, grams: BlockGrams, working_bytes: int) -> bool:
    """Whether the exact dual route costs fewer flops than the window route and its n x n factor fits.

    Leading-order counts (Golub and Van Loan, Matrix Computations, 4th ed.: Cholesky n^3 / 3; a triangular solve
    n^2 per column; the symmetric eigendecomposition with vectors about 9 n^3):
    - exact: n^2 p (forming K_S) + n^3 / 3 (its factor) + n^2 p (the triangular solves);
    - window: the sum over blocks of 9 |W_b|^3, with |W_b| the block's window size.
    The exact route needs n^2 float64 for the factor and n^2 for its inverse (the leverages).
    """
    variant_count = sum(members.shape[0] for members in grams.blocks)
    exact = 2.0 * sample_count**2 * variant_count + sample_count**3 / 3.0
    window = 0.0
    for block in range(len(grams.blocks)):
        size = sum(grams.blocks[member].shape[0] for member in _window_blocks(grams, block))
        window += 9.0 * float(size) ** 3
    fits = 2 * sample_count * sample_count * np.dtype(np.float64).itemsize <= working_bytes
    return bool(fits and exact < window)
