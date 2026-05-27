"""GPU kernels for sparse standardized carrier matvec / rmatvec.

For a standardized genotype matrix X with shape ``(n_samples, n_variants)``,
where::

    X[i, j] = (raw[i, j] - means[j]) / scales[j]

most entries of the *raw* genotype matrix are zero (non-carriers), so we store
only the carrier entries per variant:

    carriers[j]            -> int32[K_j] sample indices that carry variant j
    carrier_genotypes[j]   -> float32[K_j] raw dosage at those sample indices

The standardized matvec ``y = X @ x`` decomposes as::

    raw[:, j] @ x[j] / scales[j]    -- sparse scatter-add over carriers
    -(means[j] / scales[j]) * x[j]  -- constant subtracted from every sample

So::

    y = sparse_part - (sum_j x[j] * means[j] / scales[j]) * ones(n_samples)

The right-multiply ``z = X.T @ y`` decomposes as::

    z[j] = (1 / scales[j]) * sum_k carrier_geno[j][k] * y[carriers[j][k]]
           - (means[j] / scales[j]) * sum(y)

Both kernels are O(sum_j K_j) work plus O(n_samples) for the dense constant
broadcast in the matvec and O(n_samples) for the ``sum(y)`` reduction in the
rmatvec.  They never materialise the full dense ``X``.

Implementation note: rather than authoring a hand-rolled ``cupy.RawKernel``
(which is hard to test on CI without a real GPU), this module is written purely
in terms of cupy / numpy vector operations.  Tests use the standard
``pytest.importorskip("cupy")`` and a numpy fallback path: when ``cupy`` is the
host's ``numpy`` (via duck typing) the same code runs as a CPU reference, which
is exactly what we compare against the dense formulation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


ArrayLike = "np.ndarray | object"  # cupy.ndarray duck-typed


def _xp_of(array: object):
    """Return the array module (cupy or numpy) that owns ``array``."""

    module = type(array).__module__
    if module.startswith("cupy"):
        import cupy as cp  # type: ignore[import-not-found]

        return cp
    return np


def sparse_matvec(
    carriers: Sequence[object],
    carrier_genotypes: Sequence[object],
    means: object,
    scales: object,
    x: object,
    n_samples: int,
    n_variants: int,
) -> object:
    """Compute ``y = X_std @ x`` for the sparse carrier representation.

    Parameters
    ----------
    carriers:
        Length-``n_variants`` sequence of int arrays (``int32``/``int64``) of
        sample indices that carry each variant.
    carrier_genotypes:
        Length-``n_variants`` sequence of float arrays of raw genotype dosages
        at the corresponding carrier indices.
    means, scales:
        1-D arrays of shape ``(n_variants,)`` with the per-variant mean / scale
        used for standardization.
    x:
        1-D array of shape ``(n_variants,)``.
    n_samples, n_variants:
        Output / problem dimensions.  Must match the array shapes.
    """

    if len(carriers) != n_variants or len(carrier_genotypes) != n_variants:
        raise ValueError(
            "carriers and carrier_genotypes must have length n_variants="
            f"{n_variants}"
        )

    xp = _xp_of(x)
    dtype = x.dtype

    means_arr = xp.asarray(means, dtype=dtype)
    scales_arr = xp.asarray(scales, dtype=dtype)

    # Sparse scatter-add part.  Concatenate all carrier indices / genotypes
    # once so that we issue a single scatter-add (O(total_nnz)).
    if n_variants == 0:
        return xp.zeros(n_samples, dtype=dtype)

    sizes = np.fromiter(
        (int(_length(c)) for c in carriers), dtype=np.int64, count=n_variants
    )
    total_nnz = int(sizes.sum())

    y = xp.zeros(n_samples, dtype=dtype)

    if total_nnz > 0:
        # Per-variant scaling x[j] / scales[j] applied to each carrier's
        # genotype value.  We expand to a flat representation.
        all_idx = xp.concatenate(
            [xp.asarray(carriers[j], dtype=xp.int64) for j in range(n_variants)
             if int(sizes[j]) > 0]
        )
        all_geno = xp.concatenate(
            [xp.asarray(carrier_genotypes[j], dtype=dtype)
             for j in range(n_variants) if int(sizes[j]) > 0]
        )
        # Build a parallel array of (x[j] / scales[j]) replicated K_j times.
        coeff_per_variant = (x / scales_arr).astype(dtype, copy=False)
        # numpy.repeat broadcasts to cupy.repeat too.
        sizes_dev = xp.asarray(sizes, dtype=xp.int64)
        coeff_flat = xp.repeat(coeff_per_variant, sizes_dev)
        contrib = all_geno * coeff_flat
        # Scatter-add into y.  cupy and numpy both expose `add.at`, but
        # cupy's ``ndarray.scatter_add`` is the GPU-native primitive.
        _scatter_add(xp, y, all_idx, contrib)

    # Dense constant part: y -= (sum_j x[j] * means[j] / scales[j]) * ones.
    constant = xp.sum((x * means_arr) / scales_arr)
    y = y - constant
    return y


def sparse_rmatvec(
    carriers: Sequence[object],
    carrier_genotypes: Sequence[object],
    means: object,
    scales: object,
    y: object,
    n_samples: int,
    n_variants: int,
) -> object:
    """Compute ``z = X_std.T @ y`` for the sparse carrier representation."""

    if len(carriers) != n_variants or len(carrier_genotypes) != n_variants:
        raise ValueError(
            "carriers and carrier_genotypes must have length n_variants="
            f"{n_variants}"
        )

    xp = _xp_of(y)
    dtype = y.dtype

    means_arr = xp.asarray(means, dtype=dtype)
    scales_arr = xp.asarray(scales, dtype=dtype)

    z = xp.zeros(n_variants, dtype=dtype)
    y_sum = xp.sum(y)

    for j in range(n_variants):
        idx = xp.asarray(carriers[j], dtype=xp.int64)
        geno = xp.asarray(carrier_genotypes[j], dtype=dtype)
        if int(idx.size) > 0:
            z[j] = xp.sum(geno * y[idx])
        # else: z[j] stays zero before the dense correction
    z = z / scales_arr - (means_arr / scales_arr) * y_sum
    return z


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _length(obj: object) -> int:
    size = getattr(obj, "size", None)
    if size is not None:
        return int(size)
    return len(obj)  # type: ignore[arg-type]


def _scatter_add(xp, target, indices, values) -> None:
    """Device-native scatter-add.

    On cupy uses ``cupyx.scatter_add`` (atomic on GPU); on numpy falls back to
    ``np.add.at``.
    """

    if xp is np:
        np.add.at(target, indices, values)
        return
    try:  # pragma: no cover - exercised on GPU only
        import cupyx  # type: ignore[import-not-found]

        cupyx.scatter_add(target, indices, values)
    except Exception:  # pragma: no cover
        # Fallback: cupy.ndarray.scatter_add exists on older releases.
        target.scatter_add(indices, values)
