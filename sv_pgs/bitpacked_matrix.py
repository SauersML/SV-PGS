"""Device-resident bitpacked genotype matrix.

A drop-in for ``Int8RawGenotypeMatrix`` that keeps the PLINK 1.9 2-bit
encoding all the way on GPU HBM and uses the custom kernels in
``sv_pgs.bitpacked`` (gemv_nt / gemv_tn / gemm_gram) for linear algebra.

Implements the ``RawGenotypeMatrix`` ABC plus the extended method surface
documented in ``BITPACKED_SPEC.md`` (matvec/rmatvec, matmat,
transpose_matmat_numpy, gram_block, subset, column_means/stds,
to_host_int8, properties, close).
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from sv_pgs.genotype import RawGenotypeBatch, RawGenotypeMatrix

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    import cupy as cp


def _cupy():
    """Lazy cupy import. Raises ImportError with a clear message if unavailable."""
    import cupy as cp  # noqa: WPS433 - lazy by design

    return cp


class _BitpackedFns:
    """Cached handle to the bitpacked kernel callables.

    We deliberately bind the concrete submodule functions here rather than
    going through ``sv_pgs.bitpacked``'s package-level ``__getattr__``.
    Importing any submodule has the side effect of binding the submodule
    onto the package's ``__dict__``, after which ``bp.gemv_nt`` resolves to
    the *module* (not the callable) and ``__getattr__`` is never invoked.
    Binding the callables on this small object sidesteps that pitfall.
    """

    __slots__ = ("gemv_nt", "gemv_tn", "gemm_gram", "screen", "make_decode_lut")

    def __init__(self) -> None:
        from sv_pgs.bitpacked.gemv_nt import gemv_nt
        from sv_pgs.bitpacked.gemv_tn import gemv_tn
        from sv_pgs.bitpacked.gemm_gram import gemm_gram
        from sv_pgs.bitpacked.screening import screen
        from sv_pgs.bitpacked.lut import make_decode_lut

        self.gemv_nt = gemv_nt
        self.gemv_tn = gemv_tn
        self.gemm_gram = gemm_gram
        self.screen = screen
        self.make_decode_lut = make_decode_lut


_BITPACKED_FNS: _BitpackedFns | None = None


def _bitpacked() -> _BitpackedFns:
    """Lazy, cached handle to the bitpacked kernel callables."""
    global _BITPACKED_FNS
    if _BITPACKED_FNS is None:
        _BITPACKED_FNS = _BitpackedFns()
    return _BITPACKED_FNS


def _bytes_per_variant(n_samples: int) -> int:
    return (int(n_samples) + 3) // 4


class BitpackedDeviceMatrix(RawGenotypeMatrix):
    """Bitpacked, device-resident (samples x variants) genotype matrix.

    Storage
    -------
    ``_packed``: ``cupy.ndarray`` of shape ``(n_variants, bytes_per_variant)``
    dtype uint8, variant-major (row v = variant v's packed sample bytes).
    ``_mean``/``_std``: ``cupy.ndarray`` of shape ``(n_variants,)`` float32.

    The class deliberately uses lazy ``cupy`` imports so that the module can
    be imported on hosts without CUDA / cupy installed.
    """

    __slots__ = (
        "_packed",
        "_n_samples",
        "_n_variants",
        "_mean",
        "_std",
        "_count_a1",
        "_properties",
    )

    def __init__(
        self,
        packed: "cp.ndarray",
        n_samples: int,
        mean: "cp.ndarray",
        std: "cp.ndarray",
        *,
        count_a1: bool = True,
        properties: dict[str, Any] | None = None,
    ) -> None:
        cp = _cupy()
        if packed.ndim != 2:
            raise ValueError("packed must be 2D (n_variants, bytes_per_variant).")
        if packed.dtype != cp.uint8:
            raise ValueError("packed must be uint8.")
        n_variants = int(packed.shape[0])
        expected_bpv = _bytes_per_variant(n_samples)
        if int(packed.shape[1]) != expected_bpv:
            raise ValueError(
                f"packed has {int(packed.shape[1])} bytes/variant but n_samples={n_samples} "
                f"requires {expected_bpv}."
            )
        if mean.shape != (n_variants,) or std.shape != (n_variants,):
            raise ValueError("mean and std must be shape (n_variants,).")
        if mean.dtype != cp.float32:
            mean = mean.astype(cp.float32)
        if std.dtype != cp.float32:
            std = std.astype(cp.float32)

        self._packed = packed
        self._n_samples = int(n_samples)
        self._n_variants = int(n_variants)
        self._mean = mean
        self._std = std
        self._count_a1 = bool(count_a1)
        self._properties = dict(properties) if properties is not None else {}
        # Loud size log at construction so every BitpackedDeviceMatrix that
        # gets built leaves a greppable record of its HBM footprint. The
        # ratio against ``memGetInfo`` lets operators verify the active
        # matrix hasn't accidentally eaten the entire device budget.
        try:
            from sv_pgs.progress import log as _log
            packed_bytes = int(packed.nbytes)
            side_bytes = int(mean.nbytes) + int(std.nbytes)
            total_bytes = packed_bytes + side_bytes
            try:
                free_bytes, total_dev_bytes = cp.cuda.runtime.memGetInfo()
                hbm_field = (
                    f"hbm_used={total_bytes / 1e9:.3f}/{total_dev_bytes / 1e9:.3f} GB "
                    f"(free={free_bytes / 1e9:.3f} GB)"
                )
            except Exception as _exc:  # noqa: BLE001 - memGetInfo is best-effort
                hbm_field = f"hbm_used={total_bytes / 1e9:.3f} GB (memGetInfo failed: {_exc!r})"
            _log(
                f"BitpackedDeviceMatrix: n_samples={self._n_samples}, "
                f"n_variants={self._n_variants}, "
                f"packed_bytes={packed_bytes}, side_bytes={side_bytes}, "
                f"{hbm_field}"
            )
        except Exception:  # noqa: BLE001 - logging never blocks construction
            pass

    def __repr__(self) -> str:
        # HBM footprint reports packed + per-variant float side arrays. Wrapped
        # in try/except because attributes may be cleared after ``close()`` and
        # ``__repr__`` must never raise.
        try:
            packed = self._packed
            mean = self._mean
            std = self._std
            if packed is None:
                return (
                    f"BitpackedDeviceMatrix(n_samples={self._n_samples}, "
                    f"n_variants={self._n_variants}, closed=True)"
                )
            hbm_bytes = int(packed.nbytes) + int(mean.nbytes) + int(std.nbytes)
            return (
                f"BitpackedDeviceMatrix(n_samples={self._n_samples}, "
                f"n_variants={self._n_variants}, hbm_gb={hbm_bytes / 1e9:.2f})"
            )
        except Exception:  # noqa: BLE001 - __repr__ must not raise
            return (
                f"BitpackedDeviceMatrix(n_samples={getattr(self, '_n_samples', '?')}, "
                f"n_variants={getattr(self, '_n_variants', '?')})"
            )

    # ------------------------------------------------------------------
    # RawGenotypeMatrix ABC surface
    # ------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return (self._n_samples, self._n_variants)

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def n_variants(self) -> int:
        return self._n_variants

    @property
    def dtype(self) -> Any:
        cp = _cupy()
        return cp.dtype("float32")

    @property
    def properties(self) -> dict[str, Any]:
        return self._properties

    def iter_column_batches(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
        batch_size: int = 256,
    ) -> Iterator[RawGenotypeBatch]:
        """Decode columns on demand by materializing a host int8 slice.

        This is a slow compatibility path for callers that still drive
        the matrix through the streaming ABC. Device kernels should
        instead use ``matvec`` / ``rmatvec`` / ``gram_block`` directly.
        """
        resolved = self._resolve_variant_indices(variant_indices)
        if resolved.size == 0:
            return
        safe_batch = max(int(batch_size), 1)
        for start in range(0, resolved.shape[0], safe_batch):
            batch_idx = resolved[start : start + safe_batch]
            block_int8 = self._materialize_int8(batch_idx)  # (n_samples, k) int8
            values = block_int8.astype(np.float32)
            # Replace PLINK missing sentinel (-127) with NaN to match
            # the float32 contract used by DenseRawGenotypeMatrix.
            values[block_int8 == -127] = np.nan
            yield RawGenotypeBatch(
                variant_indices=batch_idx.astype(np.int32, copy=False),
                values=values,
            )

    def materialize(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
    ) -> NDArray:
        resolved = self._resolve_variant_indices(variant_indices)
        if resolved.size == 0:
            return np.empty((self._n_samples, 0), dtype=np.float32)
        block_int8 = self._materialize_int8(resolved)
        values = block_int8.astype(np.float32)
        values[block_int8 == -127] = np.nan
        return values

    # ------------------------------------------------------------------
    # Matvec / rmatvec
    # ------------------------------------------------------------------
    def matvec(self, x_dev: "cp.ndarray") -> "cp.ndarray":
        """Compute G @ x on device. Returns (n_samples,) float32."""
        from sv_pgs.bitpacked_profile import record as _profile_record  # local import

        cp = _cupy()
        bp = _bitpacked()
        if self._n_variants == 0:
            return cp.zeros((self._n_samples,), dtype=cp.float32)
        x32 = cp.ascontiguousarray(x_dev, dtype=cp.float32)
        if x32.shape != (self._n_variants,):
            raise ValueError(
                f"matvec: x has shape {tuple(x32.shape)}, expected ({self._n_variants},)."
            )
        out = cp.zeros((self._n_samples,), dtype=cp.float32)
        with _profile_record("matvec"):
            bp.gemv_nt(
                self._packed,
                self._n_samples,
                x32,
                self._mean,
                self._std,
                out,
                count_a1=self._count_a1,
            )
        return out

    def rmatvec(self, y_dev: "cp.ndarray") -> "cp.ndarray":
        """Compute G.T @ y on device. Returns (n_variants,) float32."""
        from sv_pgs.bitpacked_profile import record as _profile_record  # local import

        cp = _cupy()
        bp = _bitpacked()
        if self._n_variants == 0:
            return cp.zeros((0,), dtype=cp.float32)
        y32 = cp.ascontiguousarray(y_dev, dtype=cp.float32)
        if y32.shape != (self._n_samples,):
            raise ValueError(
                f"rmatvec: y has shape {tuple(y32.shape)}, expected ({self._n_samples},)."
            )
        out = cp.zeros((self._n_variants,), dtype=cp.float32)
        with _profile_record("rmatvec"):
            bp.gemv_tn(
                self._packed,
                self._n_samples,
                y32,
                self._mean,
                self._std,
                out,
                count_a1=self._count_a1,
            )
        return out

    def matvec_numpy(self, x_np: NDArray) -> NDArray:
        cp = _cupy()
        x_dev = cp.asarray(x_np, dtype=cp.float32)
        return cp.asnumpy(self.matvec(x_dev))

    def rmatvec_numpy(self, y_np: NDArray) -> NDArray:
        cp = _cupy()
        y_dev = cp.asarray(y_np, dtype=cp.float32)
        return cp.asnumpy(self.rmatvec(y_dev))

    # ------------------------------------------------------------------
    # Matmat / transpose matmat
    # ------------------------------------------------------------------
    def matmat(self, X_np: NDArray) -> NDArray:
        """Compute G @ X for X of shape (n_variants, k). Returns host float32."""
        X = np.asarray(X_np, dtype=np.float32)
        if X.ndim == 1:
            return self.matvec_numpy(X)
        if X.ndim != 2:
            raise ValueError("matmat: X must be 1D or 2D.")
        if X.shape[0] != self._n_variants:
            raise ValueError(
                f"matmat: X has shape {X.shape}, expected first dim {self._n_variants}."
            )
        cp = _cupy()
        k = int(X.shape[1])
        out = np.empty((self._n_samples, k), dtype=np.float32)
        if self._n_variants == 0:
            out.fill(0.0)
            return out
        for j in range(k):
            col = cp.asarray(X[:, j], dtype=cp.float32)
            out[:, j] = cp.asnumpy(self.matvec(col))
        return out

    def transpose_matmat_numpy(self, Y_np: NDArray) -> NDArray:
        """Compute G.T @ Y for Y of shape (n_samples, k). Returns host float32."""
        Y = np.asarray(Y_np, dtype=np.float32)
        if Y.ndim == 1:
            return self.rmatvec_numpy(Y)
        if Y.ndim != 2:
            raise ValueError("transpose_matmat_numpy: Y must be 1D or 2D.")
        if Y.shape[0] != self._n_samples:
            raise ValueError(
                f"transpose_matmat_numpy: Y has shape {Y.shape}, "
                f"expected first dim {self._n_samples}."
            )
        cp = _cupy()
        k = int(Y.shape[1])
        out = np.empty((self._n_variants, k), dtype=np.float32)
        if self._n_variants == 0:
            return out
        for j in range(k):
            col = cp.asarray(Y[:, j], dtype=cp.float32)
            out[:, j] = cp.asnumpy(self.rmatvec(col))
        return out

    # ------------------------------------------------------------------
    # Gram block
    # ------------------------------------------------------------------
    def gram_block(self, variant_indices: NDArray | "cp.ndarray") -> "cp.ndarray":
        """Compute Z_S.T @ Z_S for the subset S of variants. Returns device float32."""
        from sv_pgs.bitpacked_profile import record as _profile_record  # local import

        cp = _cupy()
        bp = _bitpacked()
        idx_dev = cp.asarray(variant_indices, dtype=cp.int64)
        k = int(idx_dev.shape[0])
        if k == 0:
            return cp.empty((0, 0), dtype=cp.float32)
        sub_packed = cp.take(self._packed, idx_dev, axis=0)
        sub_mean = cp.take(self._mean, idx_dev, axis=0)
        sub_std = cp.take(self._std, idx_dev, axis=0)
        out = cp.zeros((k, k), dtype=cp.float32)
        with _profile_record("gram"):
            bp.gemm_gram(
                sub_packed,
                self._n_samples,
                sub_mean,
                sub_std,
                out,
                count_a1=self._count_a1,
            )
        return out

    # ------------------------------------------------------------------
    # Subset
    # ------------------------------------------------------------------
    def subset(self, variant_indices: NDArray) -> "BitpackedDeviceMatrix":
        """Return a new BitpackedDeviceMatrix whose variants are the given subset.

        ``mean``/``std`` are sliced with the same index → row v of the new
        matrix corresponds exactly to ``_mean[variant_indices[v]]`` /
        ``_std[variant_indices[v]]``, preserving alignment.
        """
        cp = _cupy()
        idx_dev = cp.asarray(variant_indices, dtype=cp.int64)
        k = int(idx_dev.shape[0])
        if k == 0:
            bpv = _bytes_per_variant(self._n_samples)
            empty_packed = cp.empty((0, bpv), dtype=cp.uint8)
            empty_mean = cp.empty((0,), dtype=cp.float32)
            empty_std = cp.empty((0,), dtype=cp.float32)
            return BitpackedDeviceMatrix(
                empty_packed,
                self._n_samples,
                empty_mean,
                empty_std,
                count_a1=self._count_a1,
                properties=self._subset_properties(np.asarray(variant_indices, dtype=np.int64)),
            )
        new_packed = cp.take(self._packed, idx_dev, axis=0)
        new_mean = cp.take(self._mean, idx_dev, axis=0)
        new_std = cp.take(self._std, idx_dev, axis=0)
        host_idx = cp.asnumpy(idx_dev)
        return BitpackedDeviceMatrix(
            new_packed,
            self._n_samples,
            new_mean,
            new_std,
            count_a1=self._count_a1,
            properties=self._subset_properties(host_idx),
        )

    # ------------------------------------------------------------------
    # Column statistics
    # ------------------------------------------------------------------
    def column_means(self) -> NDArray:
        cp = _cupy()
        return cp.asnumpy(self._mean)

    def column_stds(self) -> NDArray:
        cp = _cupy()
        return cp.asnumpy(self._std)

    # ------------------------------------------------------------------
    # Debug / test path: full int8 host array
    # ------------------------------------------------------------------
    def to_host_int8(self) -> NDArray:
        """Decode the full matrix to host int8 in Fortran (variant-major) order.

        Slow; intended for tests. Uses :mod:`sv_pgs.bitpacked.cpu_reference`
        semantics (PLINK_MISSING_INT8 = -127 for missing).
        """
        cp = _cupy()
        bp = _bitpacked()
        if self._n_variants == 0:
            return np.empty((self._n_samples, 0), dtype=np.int8, order="F")
        packed_host = cp.asnumpy(self._packed)
        lut = bp.make_decode_lut(count_a1=self._count_a1)  # (256, 4) int8
        decoded = lut[packed_host]  # (n_variants, bpv, 4) int8
        decoded = decoded.reshape(self._n_variants, -1)[:, : self._n_samples]
        # Result is (n_samples, n_variants), Fortran order so each variant
        # column is contiguous.
        return np.asfortranarray(decoded.T)

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release device memory by dropping the cupy references."""
        self._packed = None  # type: ignore[assignment]
        self._mean = None  # type: ignore[assignment]
        self._std = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_variant_indices(
        self,
        variant_indices: Sequence[int] | NDArray | None,
    ) -> NDArray:
        if variant_indices is None:
            return np.arange(self._n_variants, dtype=np.int64)
        idx = np.asarray(variant_indices, dtype=np.int64).ravel()
        if idx.size and (idx.min() < 0 or idx.max() >= self._n_variants):
            raise ValueError("variant_indices out of range.")
        return idx

    def _materialize_int8(self, variant_indices: NDArray) -> NDArray:
        """Decode the given variant subset to a host (n_samples, k) int8 array."""
        cp = _cupy()
        bp = _bitpacked()
        idx_dev = cp.asarray(variant_indices, dtype=cp.int64)
        sub_packed_host = cp.asnumpy(cp.take(self._packed, idx_dev, axis=0))
        lut = bp.make_decode_lut(count_a1=self._count_a1)
        decoded = lut[sub_packed_host]  # (k, bpv, 4) int8
        k = int(variant_indices.shape[0])
        decoded = decoded.reshape(k, -1)[:, : self._n_samples]
        return np.ascontiguousarray(decoded.T)

    def _subset_properties(self, host_idx: NDArray) -> dict[str, Any]:
        """Slice array-like entries of self._properties along the variant axis.

        Any entry whose first dimension equals ``n_variants`` is gathered
        with ``host_idx``; other entries are passed through untouched.
        """
        if not self._properties:
            return {}
        out: dict[str, Any] = {}
        for key, value in self._properties.items():
            if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == self._n_variants:
                out[key] = value[host_idx]
            elif isinstance(value, (list, tuple)) and len(value) == self._n_variants:
                out[key] = [value[int(i)] for i in host_idx]
            else:
                out[key] = value
        return out
