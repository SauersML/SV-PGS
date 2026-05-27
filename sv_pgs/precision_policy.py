"""Centralised dtype policy for numerically sensitive linear-algebra subroutines.

The mixed-precision GPU path (``gpu_compute_numpy_dtype()``) reasonably keeps
the bulk-cost GEMMs (X^T W X, X @ beta, etc.) in fp32 on Turing/Volta. But a
handful of *factorisations* in that pipeline operate on small, structurally
ill-conditioned Gram matrices where fp32 silently produces garbage and the
downstream Schur subtraction makes a much larger matrix non-PSD. The crash
diagnosed in bc1c13e (variant_precision non-PSD on AoU after the one-hot
dummy trap) was exactly this failure mode for the C^T W C bundle.

This module owns the per-site decision so the rule is global instead of
point-local.

Policy summary, by ``matrix_kind``:

- ``covariate_precision`` — C^T W C, n_cov x n_cov (~35x35). Categorical
  dummy traps push cond(C^T W C) ~1e13; ALWAYS fp64. Negligible cost.
- ``variant_precision_full`` — X^T W X + diag(prior_precision), p x p exact
  Cholesky path. After ill-conditioned fp32 GEMM accumulation this can lose
  10+ digits across diagonal entries; ALWAYS fp64 factor. The fp32 GEMM
  that built the input is preserved upstream; only the factor is promoted.
- ``variant_precision_subset`` — small k x k (k = active set) Gram on the
  active-coordinate descent inner solves. fp32 would in principle be ok if
  the active set is well-screened, but cost is tiny so we still pin fp64.
- ``low_rank`` — k x k Gram from the sample-space low-rank preconditioner
  (k = preconditioner rank, ~64-256). Class B: condition is bounded by the
  Nystrom construction. Match the bundle's compute dtype (fp32 ok).
- ``gls`` — n_cov x n_cov GLS normal matrix C^T M^{-1} C from restricted
  mean. Same dummy-trap risk as ``covariate_precision``; ALWAYS fp64. Tiny.
- ``exact_variant`` — p x p exact variant-space Cholesky (the bc1c13e
  follow-on path, promoted in-place where the factorisation happens). The
  matrix is dense and large but the factor cost is dwarfed by the upstream
  GEMM and the failure mode is silent NaN-fill; ALWAYS fp64.
- ``probe_norm`` — vector norms / QR / SVD for stochastic probes. Class C:
  condition-insensitive Lanczos-style ops; default compute dtype.

The "small ill-conditioned Gram" sites cost <1ms at fp64, so we don't gate
on size — just return fp64 unconditionally.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

MatrixKind = Literal[
    "covariate_precision",
    "variant_precision_full",
    "variant_precision_subset",
    "low_rank",
    "gls",
    "exact_variant",
    "probe_norm",
]


# Class A: small, structurally ill-conditioned, must factor in fp64.
_CLASS_A_FP64: frozenset[str] = frozenset(
    {
        "covariate_precision",
        "variant_precision_full",
        "variant_precision_subset",
        "gls",
        "exact_variant",
    }
)

# Class B/C: dtype follows the caller's compute precision.
_CLASS_BC_COMPUTE: frozenset[str] = frozenset(
    {
        "low_rank",
        "probe_norm",
    }
)


def factor_dtype(matrix_kind: MatrixKind) -> np.dtype:
    """Return the numpy dtype to use when factorising the named matrix.

    Class A kinds always return ``np.float64``. Class B/C kinds defer to the
    runtime compute dtype (``gpu_compute_numpy_dtype()``); on Turing the
    mixed-precision build returns fp32, on Ampere+ it returns fp64.
    """
    if matrix_kind in _CLASS_A_FP64:
        return np.dtype(np.float64)
    if matrix_kind in _CLASS_BC_COMPUTE:
        # Local import keeps this module free of jax-config side effects at
        # import time and avoids a circular dependency at module load.
        from sv_pgs._jax import gpu_compute_numpy_dtype

        return gpu_compute_numpy_dtype()
    raise ValueError(f"unknown matrix_kind for factor_dtype: {matrix_kind!r}")
