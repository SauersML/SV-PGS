"""Hybrid dense + sparse genotype matrix scaffolding.

This module provides the *math* for routing variants between a dense bitpacked
representation (for common variants, where most samples are carriers) and a
sparse carrier-list representation (for rare variants, where the vast majority
of samples are non-carriers and the standardized column is dominated by a
constant ``-mu/s`` value).

The classes here are a pure scaffold:

* They are importable without CuPy (they fall back to NumPy when CuPy is
  unavailable, via the ``xp_backend`` parameter -- this is what makes the
  numpy-stub unit test work).
* They are math-correct: ``matvec``/``rmatvec`` of ``BioHybridGenotypeMatrix``
  agree with a dense reference computation up to floating-point tolerance.
* They are *not* wired into the rest of the pipeline. Integration (population
  from a ``BitpackedDeviceMatrix``, carrier-count thresholding during
  preprocessing, swap-in to the mixture inference operator) is the
  responsibility of a follow-up patch.

Carrier-count threshold
-----------------------
The default threshold is ``n_samples // 64``. For a variant with ``c``
carriers, the sparse representation costs roughly ``c`` ints per variant; the
dense bitpacked representation costs ``n_samples / 4`` bytes per variant. The
sparse matvec/rmatvec is *also* O(c) per variant vs O(n_samples) for the
dense kernel. ``c < n_samples / 64`` means sparse is at least ~16x cheaper in
both storage *and* compute. Singleton/doubleton SVs from AoU sit deep below
that cutoff (a typical SV has carrier count in single digits out of ~250k
samples) so they migrate to the sparse side, while common SNPs (MAF > 1%)
stay on the dense side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from sv_pgs.genotype import RawGenotypeBatch, RawGenotypeMatrix

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sv_pgs.bitpacked_matrix import BitpackedDeviceMatrix


# Default fraction of samples that must be carriers for a variant to stay
# dense. ``n_samples / 64`` means 1/64 ~= 1.5 %; rarer variants go sparse.
DEFAULT_CARRIER_DENSE_THRESHOLD_DIVISOR: int = 64


def default_carrier_threshold(n_samples: int) -> int:
    """Return the default carrier-count threshold for hybrid splitting.

    Variants with ``carrier_count > threshold`` are routed to the dense
    bitpacked backend; variants with ``carrier_count <= threshold`` are
    routed to the sparse carrier-list backend.
    """
    return max(int(n_samples) // DEFAULT_CARRIER_DENSE_THRESHOLD_DIVISOR, 1)


class _Backend:
    """Tiny shim so tests can run with NumPy without importing CuPy.

    The real GPU path uses CuPy and the bitpacked kernels. The scaffolding
    intentionally implements the math in terms of array-library primitives
    (``add.at`` / ``scatter_add``, ``take``, etc.) so the same code is
    correct on either backend.
    """

    __slots__ = ("xp", "_is_cupy")

    def __init__(self, xp: Any) -> None:
        self.xp = xp
        self._is_cupy = getattr(xp, "__name__", "") == "cupy"

    def scatter_add(self, out: Any, indices: Any, values: Any) -> None:
        """In-place ``out[indices] += values`` with duplicate-index safety."""
        if self._is_cupy:
            # cupy provides scatter_add on the module
            self.xp.scatter_add(out, indices, values)
        else:
            np.add.at(out, indices, values)


def _resolve_backend(xp_backend: Any | None) -> _Backend:
    if xp_backend is not None:
        return _Backend(xp_backend)
    try:  # pragma: no cover - exercised on GPU hosts only
        import cupy as cp  # type: ignore[import-not-found]

        return _Backend(cp)
    except Exception:
        return _Backend(np)


@dataclass(slots=True)
class _SparseVariant:
    """Per-variant carrier index/genotype payload (internal representation)."""

    carrier_indices: NDArray  # int32, shape (n_carriers,)
    carrier_genotypes: NDArray  # int8, shape (n_carriers,)


class GpuSparseCarrierMatrix(RawGenotypeMatrix):
    """Sparse standardized genotype matrix keyed by carrier lists.

    Storage (per variant ``v``):

    * ``carrier_indices[v]``: sample indices ``i`` with ``g_{i,v} != 0``.
    * ``carrier_genotypes[v]``: the (raw, unstandardized) dosage ``g_{i,v}``
      at those indices, as int8.
    * ``mean[v]``, ``scale[v]``: standardization constants applied at
      matvec/rmatvec time.

    The standardized column is::

        z_{i,v} = (g_{i,v} - mean[v]) / scale[v]                if i in carriers
                = (0       - mean[v]) / scale[v] = -mean[v]/scale[v]   else

    so the column is a *constant* ``-mean/scale`` everywhere except at the
    carrier positions, where it picks up the extra term ``g_{i,v}/scale``.
    This decomposition is what makes ``matvec`` / ``rmatvec`` fast for rare
    variants.
    """

    __slots__ = (
        "_n_samples",
        "_n_variants",
        "_carrier_indices",
        "_carrier_genotypes",
        "_mean",
        "_scale",
        "_backend",
    )

    def __init__(
        self,
        n_samples: int,
        carrier_indices: Sequence[NDArray],
        carrier_genotypes: Sequence[NDArray],
        mean: NDArray,
        scale: NDArray,
        *,
        xp_backend: Any | None = None,
    ) -> None:
        if len(carrier_indices) != len(carrier_genotypes):
            raise ValueError(
                "carrier_indices and carrier_genotypes must have the same length."
            )
        n_variants = len(carrier_indices)
        mean_arr = np.asarray(mean, dtype=np.float32)
        scale_arr = np.asarray(scale, dtype=np.float32)
        if mean_arr.shape != (n_variants,) or scale_arr.shape != (n_variants,):
            raise ValueError("mean and scale must have shape (n_variants,).")
        if np.any(scale_arr == 0):
            raise ValueError("scale must be non-zero for every variant.")

        # Store as (n_variants,) object arrays of per-variant int arrays.
        ci = np.empty(n_variants, dtype=object)
        cg = np.empty(n_variants, dtype=object)
        for v in range(n_variants):
            idx = np.asarray(carrier_indices[v], dtype=np.int32)
            gen = np.asarray(carrier_genotypes[v], dtype=np.int8)
            if idx.shape != gen.shape:
                raise ValueError(
                    f"variant {v}: carrier_indices/genotypes shape mismatch "
                    f"({idx.shape} vs {gen.shape})."
                )
            if idx.size and (idx.min() < 0 or idx.max() >= n_samples):
                raise ValueError(f"variant {v}: carrier index out of range.")
            ci[v] = idx
            cg[v] = gen

        self._n_samples = int(n_samples)
        self._n_variants = int(n_variants)
        self._carrier_indices = ci
        self._carrier_genotypes = cg
        self._mean = mean_arr
        self._scale = scale_arr
        self._backend = _resolve_backend(xp_backend)

    # ------------------------------------------------------------------
    # RawGenotypeMatrix surface
    # ------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return self._n_samples, self._n_variants

    @property
    def mean(self) -> NDArray:
        return self._mean

    @property
    def scale(self) -> NDArray:
        return self._scale

    def iter_column_batches(  # type: ignore[override]
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
        batch_size: int = 256,
    ) -> Iterator[RawGenotypeBatch]:
        if variant_indices is None:
            resolved = np.arange(self._n_variants, dtype=np.int32)
        else:
            resolved = np.asarray(variant_indices, dtype=np.int32)
        step = max(int(batch_size), 1)
        for start in range(0, resolved.shape[0], step):
            batch_idx = resolved[start : start + step]
            values = self._materialize_standardized(batch_idx)
            yield RawGenotypeBatch(variant_indices=batch_idx, values=values)

    def materialize(  # type: ignore[override]
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
    ) -> NDArray:
        if variant_indices is None:
            resolved = np.arange(self._n_variants, dtype=np.int32)
        else:
            resolved = np.asarray(variant_indices, dtype=np.int32)
        return self._materialize_standardized(resolved)

    def _materialize_standardized(self, variant_indices: NDArray) -> NDArray:
        out = np.empty((self._n_samples, variant_indices.shape[0]), dtype=np.float32)
        for col, v in enumerate(variant_indices):
            mu = float(self._mean[v])
            sc = float(self._scale[v])
            base = -mu / sc
            col_vec = np.full((self._n_samples,), base, dtype=np.float32)
            idx = self._carrier_indices[v]
            gen = self._carrier_genotypes[v]
            if idx.size:
                col_vec[idx] = (gen.astype(np.float32) - mu) / sc
            out[:, col] = col_vec
        return out

    # ------------------------------------------------------------------
    # matvec / rmatvec
    # ------------------------------------------------------------------
    def matvec(self, x_dev: Any) -> Any:
        """Compute ``X @ x`` (sparse-side). Returns shape ``(n_samples,)``."""
        xp = self._backend.xp
        x = xp.asarray(x_dev, dtype=xp.float32)
        if x.shape != (self._n_variants,):
            raise ValueError(
                f"matvec: x has shape {tuple(x.shape)}, expected ({self._n_variants},)."
            )
        mean = xp.asarray(self._mean, dtype=xp.float32)
        scale = xp.asarray(self._scale, dtype=xp.float32)
        # Constant per-column contribution: -(mu/s) * x_v, broadcast across rows.
        const_total = float(xp.sum(-(mean / scale) * x).item()) if self._n_variants else 0.0
        out = xp.full((self._n_samples,), const_total, dtype=xp.float32)
        # Carrier corrections: for each variant v and carrier i in C_v,
        # add (g_{i,v}/s_v) * x_v at row i.
        for v in range(self._n_variants):
            idx = self._carrier_indices[v]
            gen = self._carrier_genotypes[v]
            if idx.size == 0:
                continue
            idx_d = xp.asarray(idx, dtype=xp.int64)
            gen_d = xp.asarray(gen, dtype=xp.float32)
            sc = float(self._scale[v])
            xv = float(x[v].item()) if hasattr(x[v], "item") else float(x[v])
            contrib = (gen_d / sc) * xv
            self._backend.scatter_add(out, idx_d, contrib)
        return out

    def rmatvec(self, y_dev: Any) -> Any:
        """Compute ``X.T @ y`` (sparse-side). Returns shape ``(n_variants,)``."""
        xp = self._backend.xp
        y = xp.asarray(y_dev, dtype=xp.float32)
        if y.shape != (self._n_samples,):
            raise ValueError(
                f"rmatvec: y has shape {tuple(y.shape)}, expected ({self._n_samples},)."
            )
        if self._n_variants == 0:
            return xp.zeros((0,), dtype=xp.float32)
        mean = xp.asarray(self._mean, dtype=xp.float32)
        scale = xp.asarray(self._scale, dtype=xp.float32)
        y_sum = float(xp.sum(y).item())
        out = (-(mean / scale) * y_sum).astype(xp.float32)
        # Carrier corrections: out[v] += sum_{i in C_v} (g_{i,v}/s_v) * y[i].
        # Gather y at carrier positions, multiply by g/s, sum.
        out_list = xp.zeros((self._n_variants,), dtype=xp.float32)
        for v in range(self._n_variants):
            idx = self._carrier_indices[v]
            gen = self._carrier_genotypes[v]
            if idx.size == 0:
                continue
            idx_d = xp.asarray(idx, dtype=xp.int64)
            gen_d = xp.asarray(gen, dtype=xp.float32)
            sc = float(self._scale[v])
            yi = xp.take(y, idx_d)
            out_list[v] = xp.sum((gen_d / sc) * yi)
        return out + out_list

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_dense(
        cls,
        dense: NDArray,
        mean: NDArray,
        scale: NDArray,
        *,
        xp_backend: Any | None = None,
    ) -> "GpuSparseCarrierMatrix":
        """Construct from an (n_samples, n_variants) int dosage matrix.

        Non-zero entries are treated as carriers. Used by tests and by the
        eventual pipeline-side ingestion code.
        """
        arr = np.asarray(dense)
        if arr.ndim != 2:
            raise ValueError("dense must be 2D.")
        n_samples, n_variants = arr.shape
        ci: list[NDArray] = []
        cg: list[NDArray] = []
        for v in range(n_variants):
            col = arr[:, v]
            mask = col != 0
            ci.append(np.flatnonzero(mask).astype(np.int32))
            cg.append(col[mask].astype(np.int8))
        return cls(
            n_samples=n_samples,
            carrier_indices=ci,
            carrier_genotypes=cg,
            mean=mean,
            scale=scale,
            xp_backend=xp_backend,
        )


class BioHybridGenotypeMatrix(RawGenotypeMatrix):
    """Hybrid dense + sparse genotype matrix.

    A unified view over the *global* variant index space. The matrix is the
    column-concatenation of:

    * a :class:`BitpackedDeviceMatrix` (or any object with ``matvec`` /
      ``rmatvec`` of compatible shape) holding the **common** variants, and
    * a :class:`GpuSparseCarrierMatrix` holding the **rare** variants,

    with two index maps ``dense_to_global`` / ``sparse_to_global`` that say
    where each local block's columns live in the global ``(n_samples,
    n_global_variants)`` layout.

    Math:

    * ``matvec(x)`` with ``x`` of shape ``(n_global_variants,)`` returns
      ``X @ x = dense.matvec(x[dense_to_global]) + sparse.matvec(x[sparse_to_global])``
      of shape ``(n_samples,)``.

    * ``rmatvec(y)`` with ``y`` of shape ``(n_samples,)`` returns a
      ``(n_global_variants,)`` vector whose entries at ``dense_to_global`` are
      ``dense.rmatvec(y)`` and whose entries at ``sparse_to_global`` are
      ``sparse.rmatvec(y)``. (The two index sets are disjoint; positions not
      covered by either map are zero, which would only happen if the caller
      passes an incomplete partition -- normally
      ``len(dense_to_global) + len(sparse_to_global) == n_global_variants``.)
    """

    __slots__ = (
        "_dense",
        "_sparse",
        "_dense_to_global",
        "_sparse_to_global",
        "_n_samples",
        "_n_global_variants",
        "_backend",
    )

    def __init__(
        self,
        dense: "BitpackedDeviceMatrix | Any",
        sparse: GpuSparseCarrierMatrix,
        dense_to_global: NDArray,
        sparse_to_global: NDArray,
        *,
        n_global_variants: int | None = None,
        xp_backend: Any | None = None,
    ) -> None:
        d2g = np.asarray(dense_to_global, dtype=np.int64)
        s2g = np.asarray(sparse_to_global, dtype=np.int64)
        if d2g.ndim != 1 or s2g.ndim != 1:
            raise ValueError("dense_to_global / sparse_to_global must be 1D.")

        dense_shape = tuple(dense.shape)
        sparse_shape = tuple(sparse.shape)
        if dense_shape[0] != sparse_shape[0]:
            raise ValueError(
                f"dense and sparse must agree on n_samples "
                f"({dense_shape[0]} vs {sparse_shape[0]})."
            )
        if d2g.shape[0] != dense_shape[1]:
            raise ValueError(
                f"dense_to_global has length {d2g.shape[0]} but dense has "
                f"{dense_shape[1]} variants."
            )
        if s2g.shape[0] != sparse_shape[1]:
            raise ValueError(
                f"sparse_to_global has length {s2g.shape[0]} but sparse has "
                f"{sparse_shape[1]} variants."
            )

        if n_global_variants is None:
            n_global_variants = int(d2g.shape[0] + s2g.shape[0])
        # Disjointness check (cheap, catches bugs early).
        if d2g.size and s2g.size:
            overlap = np.intersect1d(d2g, s2g, assume_unique=False)
            if overlap.size:
                raise ValueError(
                    f"dense_to_global and sparse_to_global overlap at "
                    f"{overlap.size} global index/indices."
                )

        self._dense = dense
        self._sparse = sparse
        self._dense_to_global = d2g
        self._sparse_to_global = s2g
        self._n_samples = int(dense_shape[0])
        self._n_global_variants = int(n_global_variants)
        self._backend = _resolve_backend(xp_backend)

    # ------------------------------------------------------------------
    # RawGenotypeMatrix surface
    # ------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return self._n_samples, self._n_global_variants

    @property
    def dense(self) -> Any:
        return self._dense

    @property
    def sparse(self) -> GpuSparseCarrierMatrix:
        return self._sparse

    @property
    def dense_to_global(self) -> NDArray:
        return self._dense_to_global

    @property
    def sparse_to_global(self) -> NDArray:
        return self._sparse_to_global

    def iter_column_batches(  # type: ignore[override]
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
        batch_size: int = 256,
    ) -> Iterator[RawGenotypeBatch]:
        raise NotImplementedError(
            "BioHybridGenotypeMatrix is operator-only; use matvec/rmatvec."
        )

    def materialize(  # type: ignore[override]
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
    ) -> NDArray:
        raise NotImplementedError(
            "BioHybridGenotypeMatrix is operator-only; use matvec/rmatvec."
        )

    # ------------------------------------------------------------------
    # matvec / rmatvec
    # ------------------------------------------------------------------
    def matvec(self, x: Any) -> Any:
        xp = self._backend.xp
        x_arr = xp.asarray(x, dtype=xp.float32)
        if x_arr.shape != (self._n_global_variants,):
            raise ValueError(
                f"matvec: x has shape {tuple(x_arr.shape)}, "
                f"expected ({self._n_global_variants},)."
            )
        d_idx = xp.asarray(self._dense_to_global, dtype=xp.int64)
        s_idx = xp.asarray(self._sparse_to_global, dtype=xp.int64)
        x_dense = xp.take(x_arr, d_idx)
        x_sparse = xp.take(x_arr, s_idx)
        # Each operand returns shape (n_samples,). Sum them.
        if d_idx.size:
            y_dense = self._dense.matvec(x_dense)
        else:
            y_dense = xp.zeros((self._n_samples,), dtype=xp.float32)
        if s_idx.size:
            y_sparse = self._sparse.matvec(x_sparse)
        else:
            y_sparse = xp.zeros((self._n_samples,), dtype=xp.float32)
        return y_dense + y_sparse

    def rmatvec(self, y: Any) -> Any:
        xp = self._backend.xp
        y_arr = xp.asarray(y, dtype=xp.float32)
        if y_arr.shape != (self._n_samples,):
            raise ValueError(
                f"rmatvec: y has shape {tuple(y_arr.shape)}, "
                f"expected ({self._n_samples},)."
            )
        out = xp.zeros((self._n_global_variants,), dtype=xp.float32)
        if self._dense_to_global.size:
            d_local = self._dense.rmatvec(y_arr)
            d_idx = xp.asarray(self._dense_to_global, dtype=xp.int64)
            self._backend.scatter_add(out, d_idx, d_local)
        if self._sparse_to_global.size:
            s_local = self._sparse.rmatvec(y_arr)
            s_idx = xp.asarray(self._sparse_to_global, dtype=xp.int64)
            self._backend.scatter_add(out, s_idx, s_local)
        return out


__all__ = [
    "DEFAULT_CARRIER_DENSE_THRESHOLD_DIVISOR",
    "default_carrier_threshold",
    "GpuSparseCarrierMatrix",
    "BioHybridGenotypeMatrix",
]
