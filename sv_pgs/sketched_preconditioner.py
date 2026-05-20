"""Subsampled Randomized Hadamard Transform (SRHT) preconditioner.

We solve, in the inner E-step of the polygenic score solver, systems

    A beta = c,    A = X^T W X + D^{-1}

with X in R^{n x p}, W = diag(w), D = diag(tau^2). For large n (samples)
and moderate p (variants), Cholesky on A is infeasible per iteration when W
changes, but a sketched approximation gives an excellent preconditioner.

Let B = diag(sqrt(w)) @ X in R^{n x p}, so B^T B = X^T W X. Define the
SRHT sketching matrix

    S = sqrt(n_pow / r) * P @ H @ D_rand

where H is the orthonormal Hadamard matrix on the next power of two
n_pow >= n, D_rand a diagonal of random +/-1, and P selects r rows
uniformly without replacement. (B is zero-padded to n_pow rows.) Then
with high probability

    (1 - eps) B^T B  <=  (SB)^T (SB)  <=  (1 + eps) B^T B

for r = O(p log p / eps^2). The preconditioner is the Cholesky factor of

    M^{-1} = (SB)^T (SB) + D^{-1}

so PCG with M brings cond(M^{1/2} A M^{1/2}) <= (1+eps)/(1-eps).

This module is pure numpy + scipy. It is intended to be wired into
``linear_solvers.py`` in a follow-up change.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np
from scipy.linalg import solve_triangular

try:  # scipy >= 0.8 ships dense hadamard; we fall back to FWHT if huge
    from scipy.linalg import hadamard as _scipy_hadamard
except Exception:  # pragma: no cover - defensive
    _scipy_hadamard = None


__all__ = [
    "SRHTPreconditioner",
    "srht_apply",
    "build_srht_preconditioner",
    "apply_srht_preconditioner",
]


# Memory threshold above which we refuse to materialize H_n explicitly
# (n_pow x n_pow float64). 2**14 = 16384 -> 2 GiB. Use FWHT below this
# whenever we can; only fall back to dense when scipy.linalg.hadamard is
# unavailable.
_DENSE_HADAMARD_LIMIT = 1 << 13  # 8192


@dataclass(slots=True)
class SRHTPreconditioner:
    """Cached factor of (SB)^T (SB) + D^{-1}.

    Apply M v = (cached)^{-1} v with two triangular solves.
    Rebuild whenever the prior variances D or weights W change materially.
    """

    cholesky_factor: np.ndarray   # lower-triangular L with L L^T = M^{-1}
    sketch_rows: int              # r
    relative_error: float         # nominal eps used to choose r


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (int(value - 1).bit_length())


def _fwht_inplace(array: np.ndarray) -> None:
    """In-place Fast Walsh-Hadamard Transform along axis 0.

    Implements the *unnormalized* Hadamard transform: after the call,
    ``array`` holds H_unnorm @ array where H_unnorm has entries +/- 1.
    The orthonormal transform is this divided by sqrt(n_pow). We keep the
    unnormalized form here for speed and fold the normalization into the
    SRHT scaling factor in :func:`srht_apply`.
    Requires ``array.shape[0]`` to be a power of two.
    """
    n_pow = array.shape[0]
    if n_pow & (n_pow - 1):
        raise ValueError("FWHT requires a power-of-two leading dimension")
    h = 1
    while h < n_pow:
        # Process pairs of slabs of height h.
        for i in range(0, n_pow, h * 2):
            a = array[i : i + h]
            b = array[i + h : i + 2 * h]
            # In-place butterfly. Use a temporary because a and b alias views.
            tmp = a + b
            np.subtract(a, b, out=b)
            a[...] = tmp
        h *= 2


def _hadamard_matmul(matrix: np.ndarray) -> np.ndarray:
    """Return H_unnorm @ matrix where H_unnorm is the +/-1 Hadamard matrix.

    Uses scipy.linalg.hadamard for small n_pow (dense BLAS path) and the
    in-place FWHT for larger n_pow to avoid the n_pow^2 storage cost.
    Always returns a fresh contiguous float64 array.
    """
    n_pow = matrix.shape[0]
    if n_pow <= _DENSE_HADAMARD_LIMIT and _scipy_hadamard is not None:
        h_matrix = _scipy_hadamard(n_pow).astype(np.float64, copy=False)
        return h_matrix @ matrix
    work = np.ascontiguousarray(matrix, dtype=np.float64).copy()
    _fwht_inplace(work)
    return work


def srht_apply(
    rng: np.random.Generator,
    weights: np.ndarray,
    design: np.ndarray,
    sketch_rows: int,
) -> np.ndarray:
    """Apply an SRHT sketch to B = diag(sqrt(weights)) @ design.

    Returns ``B' = S @ B`` with shape ``(sketch_rows, p)``. ``S`` is the
    SRHT operator described in the module docstring.
    """
    weights = np.asarray(weights, dtype=np.float64)
    design = np.asarray(design, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError("weights must be a 1-D array")
    if design.ndim != 2:
        raise ValueError("design must be a 2-D array")
    n_samples = design.shape[0]
    if weights.shape[0] != n_samples:
        raise ValueError("weights length must match design rows")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    if sketch_rows <= 0:
        raise ValueError("sketch_rows must be positive")

    n_pow = _next_power_of_two(n_samples)
    if sketch_rows > n_pow:
        raise ValueError("sketch_rows cannot exceed next-power-of-two(n)")

    # Build B = diag(sqrt(w)) X, padded to n_pow rows with zeros.
    sqrt_w = np.sqrt(weights)
    if n_pow == n_samples:
        b_padded = design * sqrt_w[:, None]
    else:
        b_padded = np.zeros((n_pow, design.shape[1]), dtype=np.float64)
        b_padded[:n_samples] = design * sqrt_w[:, None]

    # Random +/-1 diagonal.
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=n_pow)
    b_padded *= signs[:, None]

    # H @ b_padded (unnormalized; sqrt(n_pow) is folded into S scaling).
    transformed = _hadamard_matmul(b_padded)

    # Sample r rows without replacement.
    selected = rng.choice(n_pow, size=sketch_rows, replace=False)
    sketched = transformed[selected]

    # SRHT scaling: sqrt(n_pow / r) * (1/sqrt(n_pow)) = 1/sqrt(r).
    # (The 1/sqrt(n_pow) is the orthonormal-Hadamard correction we omitted
    # when using the unnormalized +/-1 Hadamard matrix above.)
    sketched *= 1.0 / np.sqrt(sketch_rows)
    return sketched


def _default_sketch_rows(n_samples: int, n_variants: int) -> int:
    p_int = max(n_variants, 1)
    base = max(4 * p_int + 50, int(4 * p_int * log(max(p_int, 2))))
    r = min(n_samples, base)
    # Round up to satisfy r <= n_pow trivially and keep sampling sensible.
    return max(r, 1)


def build_srht_preconditioner(
    *,
    design: np.ndarray,
    weights: np.ndarray,
    prior_variances: np.ndarray,
    sketch_rows: int | None = None,
    relative_error_target: float = 0.1,
    random_seed: int = 0,
) -> SRHTPreconditioner:
    """Build the SRHT preconditioner for A = X^T W X + diag(1/tau^2).

    The returned object caches the lower-triangular Cholesky factor L of
    M^{-1} = (SB)^T (SB) + D^{-1}, so ``M v = L^{-T} L^{-1} v`` via two
    triangular solves.

    Notes
    -----
    The preconditioner factorizes a ``p x p`` matrix. This is appropriate
    when ``p`` is at most a few thousand; above ~10k the Cholesky cost
    dominates and a different strategy is needed.
    """
    design = np.asarray(design, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    prior_variances = np.asarray(prior_variances, dtype=np.float64)

    if design.ndim != 2:
        raise ValueError("design must be a 2-D array")
    n_samples, n_variants = design.shape
    if weights.shape != (n_samples,):
        raise ValueError("weights must have shape (n_samples,)")
    if prior_variances.shape != (n_variants,):
        raise ValueError("prior_variances must have shape (n_variants,)")
    if np.any(prior_variances <= 0):
        raise ValueError("prior_variances must be strictly positive")

    if sketch_rows is None:
        sketch_rows = _default_sketch_rows(n_samples, n_variants)
    sketch_rows = int(sketch_rows)
    n_pow = _next_power_of_two(n_samples)
    if sketch_rows > n_pow:
        sketch_rows = n_pow

    rng = np.random.default_rng(random_seed)
    sketched = srht_apply(rng, weights, design, sketch_rows)

    # Form (SB)^T (SB) + diag(1/tau^2). This is the p x p Gram matrix.
    gram = sketched.T @ sketched
    gram_diag = gram.diagonal().copy()
    gram_diag += 1.0 / prior_variances
    np.fill_diagonal(gram, gram_diag)
    # Symmetrize to suppress drift; the prior makes it SPD.
    gram = 0.5 * (gram + gram.T)

    # Plain Cholesky -> lower-triangular L with L L^T = gram. We use
    # np.linalg.cholesky because it returns L directly (no factor tuple).
    cholesky_factor = np.linalg.cholesky(gram)
    return SRHTPreconditioner(
        cholesky_factor=cholesky_factor,
        sketch_rows=sketch_rows,
        relative_error=float(relative_error_target),
    )


def apply_srht_preconditioner(
    preconditioner: SRHTPreconditioner, vector: np.ndarray
) -> np.ndarray:
    """Return ``M @ v`` via two triangular solves on the cached Cholesky.

    With ``L L^T = M^{-1}``, solving ``L y = v`` then ``L^T x = y`` yields
    ``x = (L L^T)^{-1} v = M v``.
    """
    vector = np.asarray(vector, dtype=np.float64)
    factor = preconditioner.cholesky_factor
    if vector.ndim != 1:
        raise ValueError("vector must be 1-D")
    if vector.shape[0] != factor.shape[0]:
        raise ValueError("vector length must match preconditioner dimension")
    intermediate = solve_triangular(factor, vector, lower=True, check_finite=False)
    result = solve_triangular(
        factor.T, intermediate, lower=False, check_finite=False
    )
    return result
