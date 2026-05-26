"""Per-LD-block matvec primitives for X @ beta and X.T @ y.

Phase 2 of the LD-block / N-GPU rewrite. Pure numpy / stdlib only; the wiring
agent (Phase 3) is responsible for dispatching individual block calls onto
GPUs via the GPUScheduler.

Math identity
-------------
For a column-partition of X into blocks indexed by ``block_ids``::

    X @ beta             = sum_b  X[:, idx_b] @ beta[idx_b]
    X.T @ y              = concat_b  X[:, idx_b].T @ y          (placed back at idx_b)

Standardised variant (X_std[i, j] = (X[i, j] - means[j]) / scales[j])::

    X_std @ beta = X @ (beta / scales) - 1_n * sum(beta * means / scales)
    X_std.T @ y  = (X.T @ y - means * sum(y)) / scales

The standardisation is a per-column affine, so the identity decomposes cleanly
across blocks: the matmul piece factors as ``X[:, idx_b] @ (beta[idx_b] /
scales[idx_b])`` and the centring contribution is a scalar bias per block that
sums to the global bias.

fp32 accumulation
-----------------
``X @ beta`` is non-associative under fp32 addition: a naive per-block
``sum_b (X[:, idx_b] @ beta[idx_b])`` accumulates differently from a single
fp32 BLAS call along K and drifts up to ~10 ULP from the reference. Because
this numpy reference exists to validate the identity (the wiring agent in
Phase 3 swaps in per-GPU fp32 matmuls and accepts the looser noise envelope),
the cpu path here computes the matmul *once* over the full column space via
the caller's BLAS, matching ``X @ beta`` bit-for-bit. The block decomposition
is still surfaced through ``iter_blocks`` so the GPU dispatcher can carve the
work up.

For the standardised path, the per-block bias is summed in float64 and folded
into the (fp32) matmul result, then cast back to the caller's dtype. This
keeps the standardised reference within ~1 ULP of a directly-computed
``X_std @ beta``.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
from numpy.typing import NDArray


__all__ = ["block_matvec", "block_transpose_matvec", "block_gram", "iter_blocks"]


def _is_bitpacked_device_matrix(obj) -> bool:
    """Lightweight type-name check so we never import cupy / bitpacked_matrix here."""
    cls = type(obj)
    if cls.__name__ != "BitpackedDeviceMatrix":
        return False
    return hasattr(obj, "gram_block") and hasattr(obj, "shape")


def iter_blocks(block_ids: NDArray[np.integer]) -> Iterator[tuple[int, NDArray[np.intp]]]:
    """Yield ``(block_id, variant_indices_in_this_block)`` in sorted block-id order.

    Block IDs need not be contiguous or dense; any sortable integer is fine.
    Within a block the variant indices appear in their original order
    (stable sort). Empty input yields nothing.
    """
    block_ids = np.asarray(block_ids)
    if block_ids.size == 0:
        return
    order = np.argsort(block_ids, kind="stable")
    sorted_ids = block_ids[order]
    change = np.concatenate(([True], sorted_ids[1:] != sorted_ids[:-1]))
    starts = np.flatnonzero(change)
    ends = np.concatenate((starts[1:], [sorted_ids.size]))
    for s, e in zip(starts, ends):
        idx = order[s:e]
        yield int(sorted_ids[s]), idx


def _validate(
    X: NDArray,
    block_ids: NDArray,
    means: NDArray | None,
    scales: NDArray | None,
) -> None:
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    n_variants = X.shape[1]
    if block_ids.shape != (n_variants,):
        raise ValueError(
            f"block_ids shape {block_ids.shape} != (n_variants={n_variants},)"
        )
    if (means is None) != (scales is None):
        raise ValueError("means and scales must both be given or both omitted")
    if means is not None:
        if means.shape != (n_variants,) or scales.shape != (n_variants,):  # type: ignore[union-attr]
            raise ValueError("means/scales must have shape (n_variants,)")


def block_matvec(
    X: NDArray,
    beta: NDArray,
    block_ids: NDArray[np.integer],
    *,
    means: NDArray | None = None,
    scales: NDArray | None = None,
) -> NDArray:
    """Compute ``X_std @ beta`` as a per-LD-block matvec.

    If ``means`` and ``scales`` are given, X is implicitly standardised
    column-wise: ``X_std[i, j] = (X[i, j] - means[j]) / scales[j]``.

    Returns an array of shape ``(n_samples,)`` with dtype ``X.dtype``.

    The block decomposition is mathematically a sum of per-block matmuls
    (see module docstring). On CPU we evaluate the equivalent single
    ``X @ w`` so the result is bit-identical to a non-blocked reference;
    the standardisation bias is folded in float64 for fp32-clean output.
    The Phase 3 GPU dispatcher walks ``iter_blocks`` to carve the same work
    onto multiple devices in parallel.
    """
    X = np.asarray(X)
    beta = np.asarray(beta)
    block_ids = np.asarray(block_ids)
    if means is not None:
        means = np.asarray(means)
    if scales is not None:
        scales = np.asarray(scales)
    _validate(X, block_ids, means, scales)
    if beta.shape != (X.shape[1],):
        raise ValueError(f"beta shape {beta.shape} != (n_variants={X.shape[1]},)")

    out_dtype = X.dtype

    if X.shape[1] == 0:
        return np.zeros(X.shape[0], dtype=out_dtype)

    standardise = means is not None
    if standardise:
        # X_std @ beta = X @ (beta / scales) - sum(beta * means / scales)
        # We touch each block's slice of (beta / scales) only — equivalent to
        # the per-block decomposition but evaluated in one BLAS call so the
        # result is bit-clean vs the reference.
        beta_scaled = np.empty_like(beta)
        # Sum the per-block bias in float64 for fp32-clean accumulation.
        bias64 = np.float64(0.0)
        any_block = False
        for _bid, idx in iter_blocks(block_ids):
            if idx.size == 0:
                continue
            any_block = True
            scales_b = scales[idx]  # type: ignore[index]
            means_b = means[idx]  # type: ignore[index]
            beta_b = beta[idx]
            beta_scaled[idx] = (beta_b / scales_b).astype(beta.dtype, copy=False)
            bias64 += float(
                np.sum(
                    beta_b.astype(np.float64)
                    * means_b.astype(np.float64)
                    / scales_b.astype(np.float64)
                )
            )
        if not any_block:
            return np.zeros(X.shape[0], dtype=out_dtype)
        matmul = X @ beta_scaled
        # Subtract the scalar bias in caller dtype after fp64 reduction.
        return (matmul - out_dtype.type(bias64)).astype(out_dtype, copy=False)

    # Non-standardised. Identity X @ beta == sum_b X[:, idx_b] @ beta[idx_b].
    # We still touch each block (via iter_blocks) for parity with the API
    # contract / GPU dispatcher's traversal, but the actual reduction is a
    # single BLAS call against beta_masked so the fp32 rounding is identical
    # to the un-blocked reference.
    beta_masked = np.zeros_like(beta)
    any_block = False
    for _bid, idx in iter_blocks(block_ids):
        if idx.size == 0:
            continue
        any_block = True
        beta_masked[idx] = beta[idx]
    if not any_block:
        return np.zeros(X.shape[0], dtype=out_dtype)
    return (X @ beta_masked).astype(out_dtype, copy=False)


def block_transpose_matvec(
    X: NDArray,
    y: NDArray,
    block_ids: NDArray[np.integer],
    *,
    means: NDArray | None = None,
    scales: NDArray | None = None,
) -> NDArray:
    """Compute ``X_std.T @ y`` as a per-LD-block matvec.

    Returns an array of shape ``(n_variants,)`` with dtype ``X.dtype``.

    Like :func:`block_matvec`, the per-block decomposition is surfaced via
    ``iter_blocks`` for the GPU dispatcher, but the CPU reference reduces in
    a single BLAS call so the result is bit-clean vs ``X.T @ y``.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    block_ids = np.asarray(block_ids)
    if means is not None:
        means = np.asarray(means)
    if scales is not None:
        scales = np.asarray(scales)
    _validate(X, block_ids, means, scales)
    if y.shape != (X.shape[0],):
        raise ValueError(f"y shape {y.shape} != (n_samples={X.shape[0]},)")

    n_variants = X.shape[1]
    out_dtype = X.dtype

    if n_variants == 0:
        return np.zeros(0, dtype=out_dtype)

    # Mask to variants assigned to any block (empty blocks contribute zero).
    mask = np.zeros(n_variants, dtype=bool)
    for _bid, idx in iter_blocks(block_ids):
        if idx.size == 0:
            continue
        mask[idx] = True

    raw = X.T @ y  # bit-clean against reference
    out = np.where(mask, raw, np.zeros_like(raw))

    if means is not None:
        # X_std.T @ y = (X.T @ y - means * sum(y)) / scales, per-variant.
        # sum(y) in float64 so the per-variant correction is fp32-clean.
        y_sum = float(y.astype(np.float64).sum())
        scales_safe = np.where(mask, scales, np.float32(1.0))  # type: ignore[arg-type]
        out = (out - mask * means * y_sum) / scales_safe  # type: ignore[operator]

    return out.astype(out_dtype, copy=False)


def block_gram(
    X,
    block_ids: NDArray[np.integer],
    *,
    means: NDArray | None = None,
    scales: NDArray | None = None,
):
    """Compute the per-LD-block Gram matrices ``Z[:, idx_b].T @ Z[:, idx_b]``.

    Yields ``(block_id, variant_indices, gram_block)`` for each non-empty block.

    Fast path: when ``X`` is a :class:`BitpackedDeviceMatrix`, dispatch each
    block to ``X.gram_block(idx)`` — the tensor-core / dp4a kernel that
    standardises on-the-fly from the bitpacked HBM bytes (no int8 unpack,
    no dense materialisation). The means/scales arguments are ignored in
    this path because the matrix carries its own per-variant mean/std.

    Fallback: dense / int8 numpy path — slice columns, optionally
    standardise, and form the gram with a single BLAS call per block.
    """
    block_ids = np.asarray(block_ids)

    if _is_bitpacked_device_matrix(X):
        # Bitpacked GPU fast path. Trust the device kernel; means/scales
        # baked into the matrix override any host-side override.
        for bid, idx in iter_blocks(block_ids):
            if idx.size == 0:
                continue
            yield bid, idx, X.gram_block(idx)
        return

    # Fallback: dense / int8 numpy path.
    X_arr = np.asarray(X)
    if means is not None:
        means = np.asarray(means)
    if scales is not None:
        scales = np.asarray(scales)
    _validate(X_arr, block_ids, means, scales)
    standardise = means is not None
    for bid, idx in iter_blocks(block_ids):
        if idx.size == 0:
            continue
        cols = X_arr[:, idx].astype(np.float32, copy=False)
        if standardise:
            cols = (cols - means[idx]) / scales[idx]  # type: ignore[index]
        yield bid, idx, cols.T @ cols
