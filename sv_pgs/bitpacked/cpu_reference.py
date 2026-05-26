"""CPU reference implementations of the bitpacked GEMV / GEMM / screening kernels.

These are the canonical (pure-numpy) ground truth used by the test suite to
validate the CuPy RawKernels in sibling modules. They MUST be bit-for-bit
identical to ``sv_pgs.plink._decode_payload`` for genotype decoding and to
``sv_pgs.preprocessing._means_and_scales_with_floor`` for standardization.

Conventions (see ``BITPACKED_SPEC.md``):

    - ``packed`` is uint8 ``(n_variants, bytes_per_variant)`` where
      ``bytes_per_variant = (n_samples + 3) // 4``.
    - Inside each byte 4 samples are packed low-bit-first.
    - Two-bit codes under ``count_a1=True``:
        0b00 -> 2, 0b01 -> missing, 0b10 -> 1, 0b11 -> 0
      Under ``count_a1=False``:
        0b00 -> 0, 0b01 -> missing, 0b10 -> 1, 0b11 -> 2
    - Missing genotypes contribute 0 to every reduction (mean-imputed).
    - Standardized value ``z[i, v] = (raw - mean[v]) / scale[v]`` for
      non-missing samples, else 0.

Standardization (REPO-EXACT, load-bearing — see spec "Conventions"):

    N         = total sample count
    count[v]  = number of non-missing samples for variant v
    sum[v]    = Σ_{i: observed} dosage[i, v]
    sumsq[v]  = Σ_{i: observed} dosage[i, v]^2
    css[v]    = sumsq[v] - sum[v]^2 / max(count[v], 1)

    if count[v] > 0:
        mean[v]      = sum[v] / count[v]
        scale_raw[v] = sqrt(css[v] / max(N, 1))      # denom is N, not count
    else:
        mean[v]      = 0.0
        scale_raw[v] = 0.0

    if count[v] == 0:
        mean[v]  = 0.0
        scale[v] = 1.0
    elif scale_raw[v] < minimum_scale:
        mean[v]  = sum[v] / count[v]
        scale[v] = 1.0
    else:
        scale[v] = scale_raw[v]

Invariant: under no missingness and no floor, ``Σ_i z[i, v]^2 == N`` exactly.

No GPU dependency — pure numpy, runs anywhere.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

try:  # Prefer the shared LUT module if available.
    from sv_pgs.bitpacked.lut import make_decode_lut as _make_decode_lut
except Exception:  # pragma: no cover - inline fallback per spec.
    _LOOKUP_A1 = np.array([2, np.int8(-127), 1, 0], dtype=np.int8)
    _LOOKUP_A2 = np.array([0, np.int8(-127), 1, 2], dtype=np.int8)

    def _make_decode_lut(count_a1: bool = True) -> np.ndarray:
        per_code = _LOOKUP_A1 if count_a1 else _LOOKUP_A2
        lut = np.empty((256, 4), dtype=np.int8)
        for b in range(256):
            for k in range(4):
                lut[b, k] = per_code[(b >> (2 * k)) & 0b11]
        return lut


_PLINK_MISSING_INT8 = np.int8(-127)
DEFAULT_MINIMUM_SCALE: float = 1e-6


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _decode_packed(
    packed: np.ndarray,
    n_samples: int,
    count_a1: bool,
) -> np.ndarray:
    """Decode ``packed`` to int8 ``(n_variants, n_samples)`` matrix.

    Missing slots are ``-127`` (``PLINK_MISSING_INT8``). Trailing padding bits
    past ``n_samples`` are dropped.
    """
    if packed.ndim != 2:
        raise ValueError(
            f"packed must be 2-D (n_variants, bytes_per_variant); got shape {packed.shape!r}"
        )
    if packed.dtype != np.uint8:
        raise ValueError(f"packed must be uint8; got dtype {packed.dtype!r}")
    n_variants, bytes_per_variant = packed.shape
    expected_bpv = (n_samples + 3) // 4
    if bytes_per_variant != expected_bpv:
        raise ValueError(
            f"bytes_per_variant mismatch: packed has {bytes_per_variant} "
            f"bytes/variant, expected {expected_bpv} for n_samples={n_samples}"
        )
    lut = _make_decode_lut(count_a1=count_a1)
    decoded_padded = lut[packed]  # (V, bpv, 4) int8
    return decoded_padded.reshape(n_variants, bytes_per_variant * 4)[:, :n_samples]


def _resolve_scale(
    scale: np.ndarray | None,
    std: np.ndarray | None,
    func_name: str,
) -> np.ndarray:
    """Resolve the ``scale`` / legacy ``std`` parameter, warning on legacy use."""
    if scale is not None and std is not None:
        raise TypeError(
            f"{func_name}(): pass either `scale=` (preferred) or `std=` (deprecated), not both."
        )
    if std is not None:
        warnings.warn(
            f"{func_name}(): the `std=` keyword is deprecated; use `scale=` instead. "
            "Note that `scale` follows the REPO-EXACT contract with denominator N, "
            "not non-missing count.",
            DeprecationWarning,
            stacklevel=3,
        )
        return std
    if scale is None:
        raise TypeError(f"{func_name}(): missing required keyword `scale`.")
    return scale


def compute_mean_scale(
    packed: np.ndarray,
    n_samples: int,
    count_a1: bool = True,
    minimum_scale: float = DEFAULT_MINIMUM_SCALE,
    dtype: Any = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute REPO-EXACT (mean, scale) for each variant from packed bytes.

    Matches ``sv_pgs.preprocessing._means_and_scales_with_floor`` exactly:

      - ``mean[v] = sum[v] / count[v]`` over non-missing samples (where count>0).
      - ``scale_raw[v] = sqrt(css[v] / max(N, 1))`` — denominator is TOTAL N.
      - All-missing column (count==0): mean=0, scale=1.
      - Below-floor (``scale_raw < minimum_scale``): keep mean, scale=1.

    Parameters
    ----------
    packed : np.ndarray
        ``(n_variants, bytes_per_variant)`` uint8 PLINK 1.9 packed bytes.
    n_samples : int
        Total sample count ``N`` (used as the variance denominator).
    count_a1 : bool, optional
        PLINK ``count_A1`` convention.
    minimum_scale : float, optional
        Floor below which the raw scale is replaced by 1.0 (mean kept).
    dtype : numpy dtype, optional
        Output dtype for ``mean`` and ``scale``. Default ``np.float32``.

    Returns
    -------
    (mean, scale) : tuple[np.ndarray, np.ndarray]
        Each of shape ``(n_variants,)``.
    """
    decoded = _decode_packed(packed, n_samples, count_a1)
    missing = decoded == _PLINK_MISSING_INT8
    observed = ~missing
    raw = decoded.astype(np.float64)
    raw_obs = np.where(observed, raw, 0.0)

    count = observed.sum(axis=1).astype(np.int64)
    sum_ = raw_obs.sum(axis=1)
    sumsq = (raw_obs * raw_obs).sum(axis=1)

    safe_count = np.maximum(count, 1)
    css = sumsq - (sum_ * sum_) / safe_count
    css = np.maximum(css, 0.0)  # guard fp noise

    n_total_denom = max(int(n_samples), 1)
    has_obs = count > 0
    mean = np.where(has_obs, sum_ / safe_count, 0.0)
    scale_raw = np.where(has_obs, np.sqrt(css / n_total_denom), 0.0)
    scale = np.where(
        ~has_obs,
        1.0,
        np.where(scale_raw < minimum_scale, 1.0, scale_raw),
    )
    return mean.astype(dtype), scale.astype(dtype)


def _standardize(
    decoded: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Return float64 ``z`` matrix of shape ``(n_variants, n_samples)``.

    Missing slots (-127) -> 0. Otherwise ``(raw - mean[v]) / scale[v]``.
    """
    n_variants = decoded.shape[0]
    if mean.shape != (n_variants,):
        raise ValueError(f"mean shape {mean.shape} != ({n_variants},)")
    if scale.shape != (n_variants,):
        raise ValueError(f"scale shape {scale.shape} != ({n_variants},)")
    missing_mask = decoded == _PLINK_MISSING_INT8
    raw = decoded.astype(np.float64)
    z = (raw - mean.astype(np.float64)[:, None]) / scale.astype(np.float64)[:, None]
    z[missing_mask] = 0.0
    return z


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cpu_gemv_nt(
    packed: np.ndarray,
    n_samples: int,
    x: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray | None = None,
    count_a1: bool = True,
    *,
    std: np.ndarray | None = None,  # deprecated alias
) -> np.ndarray:
    """Compute ``y[i] = Σ_v z[i, v] * x[v]`` where ``z`` is standardized dosage.

    Parameters
    ----------
    packed : np.ndarray
        ``(n_variants, bytes_per_variant)`` uint8 PLINK 1.9 packed bytes.
    n_samples : int
        Number of samples (trailing padding bits ignored).
    x : np.ndarray
        ``(n_variants,)`` weights vector.
    mean, scale : np.ndarray
        ``(n_variants,)`` per-variant standardization parameters from the
        REPO-EXACT contract.
    count_a1 : bool, optional
        PLINK ``count_A1`` convention.
    std : np.ndarray, optional
        Deprecated alias for ``scale``. Emits ``DeprecationWarning``.

    Returns
    -------
    np.ndarray
        ``(n_samples,)`` array. Dtype is float32 if all of ``x``, ``mean``, and
        the resolved ``scale`` are float32, else float64.
    """
    scale_arr = _resolve_scale(scale, std, "cpu_gemv_nt")
    decoded = _decode_packed(packed, n_samples, count_a1)
    z = _standardize(decoded, mean, scale_arr)
    y = z.T @ x.astype(np.float64)
    all_f32 = (
        x.dtype == np.float32
        and mean.dtype == np.float32
        and scale_arr.dtype == np.float32
    )
    return y.astype(np.float32 if all_f32 else np.float64)


def cpu_gemv_tn(
    packed: np.ndarray,
    n_samples: int,
    y: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray | None = None,
    count_a1: bool = True,
    *,
    std: np.ndarray | None = None,  # deprecated alias
) -> np.ndarray:
    """Compute ``g[v] = Σ_i z[i, v] * y[i]``.

    Parameters
    ----------
    packed : np.ndarray
        ``(n_variants, bytes_per_variant)`` uint8 PLINK 1.9 packed bytes.
    n_samples : int
        Number of samples (trailing padding bits ignored).
    y : np.ndarray
        ``(n_samples,)`` per-sample vector.
    mean, scale : np.ndarray
        ``(n_variants,)`` per-variant standardization parameters.
    count_a1 : bool, optional
        PLINK ``count_A1`` convention.
    std : np.ndarray, optional
        Deprecated alias for ``scale``.

    Returns
    -------
    np.ndarray
        ``(n_variants,)`` array. Dtype is float32 if all of ``y``, ``mean``,
        and the resolved ``scale`` are float32, else float64.
    """
    scale_arr = _resolve_scale(scale, std, "cpu_gemv_tn")
    decoded = _decode_packed(packed, n_samples, count_a1)
    z = _standardize(decoded, mean, scale_arr)
    g = z @ y.astype(np.float64)
    all_f32 = (
        y.dtype == np.float32
        and mean.dtype == np.float32
        and scale_arr.dtype == np.float32
    )
    return g.astype(np.float32 if all_f32 else np.float64)


def cpu_gemm_gram(
    packed: np.ndarray,
    n_samples: int,
    mean: np.ndarray,
    scale: np.ndarray | None = None,
    count_a1: bool = True,
    *,
    std: np.ndarray | None = None,  # deprecated alias
) -> np.ndarray:
    """Compute the gram matrix ``B = Z.T @ Z`` of shape ``(n_variants, n_variants)``.

    ``Z`` is the standardized dosage matrix with missing slots set to zero.
    Reduction is in float64; output dtype is float32 if both ``mean`` and the
    resolved ``scale`` are float32, else float64.

    Parameters
    ----------
    packed : np.ndarray
        ``(n_variants, bytes_per_variant)`` uint8 PLINK 1.9 packed bytes.
    n_samples : int
        Number of samples (trailing padding bits ignored).
    mean, scale : np.ndarray
        ``(n_variants,)`` per-variant standardization parameters.
    count_a1 : bool, optional
        PLINK ``count_A1`` convention.
    std : np.ndarray, optional
        Deprecated alias for ``scale``.

    Returns
    -------
    np.ndarray
        Gram matrix of shape ``(n_variants, n_variants)``.
    """
    scale_arr = _resolve_scale(scale, std, "cpu_gemm_gram")
    decoded = _decode_packed(packed, n_samples, count_a1)
    z = _standardize(decoded, mean, scale_arr)
    b = z @ z.T
    all_f32 = mean.dtype == np.float32 and scale_arr.dtype == np.float32
    return b.astype(np.float32 if all_f32 else np.float64)


def cpu_screen(
    packed: np.ndarray,
    n_samples: int,
    rhs: np.ndarray | None = None,
    count_a1: bool = True,
    *,
    y_resid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """One-pass screening reductions over packed genotypes.

    Parameters
    ----------
    packed : np.ndarray
        ``(n_variants, bytes_per_variant)`` uint8 PLINK 1.9 packed bytes.
    n_samples : int
        Number of samples (trailing padding bits ignored).
    rhs : np.ndarray, optional
        ``(n_samples,)`` or ``(n_samples, k)`` right-hand-side matrix. If
        supplied, the result includes:

            ``dosage_rhs[v, k]   = Σ_{i: observed} dosage[i, v] * rhs[i, k]``
            ``observed_rhs[v, k] = Σ_{i: observed} 1            * rhs[i, k]``

        From these the standardized inner product is recovered as
        ``Z.T @ rhs[:, k] = (dosage_rhs - mean[v] * observed_rhs) / scale[v]``
        — see :func:`finalize_standardized_rhs`.
    count_a1 : bool, optional
        PLINK ``count_A1`` convention.

    Returns
    -------
    dict[str, np.ndarray]
        Keys (always):
          ``'count'`` (n_variants,) int32 — non-missing sample count.
          ``'sum'``   (n_variants,) float64 — sum of raw dosage.
          ``'sumsq'`` (n_variants,) float64 — sum of squared raw dosage.
        Keys (iff ``rhs`` provided), each shape ``(n_variants, k)`` float64
        (or ``(n_variants,)`` if ``rhs`` was 1-D):
          ``'dosage_rhs'``   — Σ_observed dosage[i, v] * rhs[i, k].
          ``'observed_rhs'`` — Σ_observed                 rhs[i, k].
    """
    if y_resid is not None:
        if rhs is not None:
            raise TypeError("Pass either rhs or y_resid, not both.")
        import warnings as _warnings
        _warnings.warn(
            "cpu_screen(y_resid=...) is deprecated; use rhs=... and read "
            "'dosage_rhs'/'observed_rhs' from the result.",
            DeprecationWarning,
            stacklevel=2,
        )
        rhs = y_resid
    decoded = _decode_packed(packed, n_samples, count_a1)
    missing_mask = decoded == _PLINK_MISSING_INT8
    observed_mask = ~missing_mask
    raw = decoded.astype(np.float64)
    raw_obs = np.where(observed_mask, raw, 0.0)

    count = observed_mask.sum(axis=1).astype(np.int32)
    sum_ = raw_obs.sum(axis=1).astype(np.float64)
    sumsq = (raw_obs * raw_obs).sum(axis=1).astype(np.float64)

    out: dict[str, np.ndarray] = {
        "count": count,
        "sum": sum_,
        "sumsq": sumsq,
    }

    if rhs is not None:
        rhs_arr = np.asarray(rhs)
        squeeze = rhs_arr.ndim == 1
        if squeeze:
            rhs_arr = rhs_arr[:, None]
        if rhs_arr.ndim != 2 or rhs_arr.shape[0] != n_samples:
            raise ValueError(
                f"rhs must be (n_samples,) or (n_samples, k); got shape {np.asarray(rhs).shape!r}"
            )
        rhs_f64 = rhs_arr.astype(np.float64)
        observed_f64 = observed_mask.astype(np.float64)

        dosage_rhs = raw_obs @ rhs_f64           # (V, k)
        observed_rhs = observed_f64 @ rhs_f64    # (V, k)
        if squeeze:
            dosage_rhs = dosage_rhs[:, 0]
            observed_rhs = observed_rhs[:, 0]
        out["dosage_rhs"] = dosage_rhs
        out["observed_rhs"] = observed_rhs

    return out


def finalize_standardized_rhs(
    dosage_rhs: np.ndarray,
    observed_rhs: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Convert raw screening accumulators to the standardized inner product.

    Returns ``Z.T @ rhs`` where ``Z`` is the standardized dosage matrix:

        result[v, k] = (dosage_rhs[v, k] - mean[v] * observed_rhs[v, k]) / scale[v]

    This identity follows from ``z[i, v] = (d[i, v] - mean[v]) / scale[v]``
    summed over the observed set (missing entries contribute 0 to both
    accumulators).

    Parameters
    ----------
    dosage_rhs, observed_rhs : np.ndarray
        ``(n_variants,)`` or ``(n_variants, k)`` accumulators from
        :func:`cpu_screen`.
    mean, scale : np.ndarray
        ``(n_variants,)`` per-variant standardization parameters.

    Returns
    -------
    np.ndarray
        float64 array with the same shape as ``dosage_rhs``.
    """
    if dosage_rhs.shape != observed_rhs.shape:
        raise ValueError(
            f"shape mismatch: dosage_rhs {dosage_rhs.shape} vs observed_rhs {observed_rhs.shape}"
        )
    n_variants = dosage_rhs.shape[0]
    if mean.shape != (n_variants,):
        raise ValueError(f"mean shape {mean.shape} != ({n_variants},)")
    if scale.shape != (n_variants,):
        raise ValueError(f"scale shape {scale.shape} != ({n_variants},)")

    mean_f64 = mean.astype(np.float64)
    scale_f64 = scale.astype(np.float64)
    if dosage_rhs.ndim == 2:
        centered = (
            dosage_rhs.astype(np.float64)
            - mean_f64[:, None] * observed_rhs.astype(np.float64)
        )
        return centered / scale_f64[:, None]
    centered = dosage_rhs.astype(np.float64) - mean_f64 * observed_rhs.astype(np.float64)
    return centered / scale_f64
