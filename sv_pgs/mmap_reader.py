"""mmap-based PLINK 1.9 .bed reader.

This reader is intended for SEQUENTIAL screening passes — single linear
sweep through the variant-major payload, where the kernel can prefault
pages via MADV_SEQUENTIAL | MADV_WILLNEED and discard them after use.

Random-access (``read_packed_indexed``) is precisely the case
``sv_pgs.plink.open_bed`` deliberately avoids: under sustained
multi-threaded random prefetch a page-fault into an evicted region can
surface as SIGBUS (process-terminating) instead of an EIO/OSError that
the executor can recover from. We still offer the indexed path because
some callers will want it for small, in-cache gathers, but it warns and
defensively falls back to ``os.preadv`` (via ``sv_pgs.plink.open_bed``)
on any read error.
"""

from __future__ import annotations

import mmap
import os
import sys
import warnings
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

PLINK1_MAGIC = b"\x6c\x1b\x01"
PLINK1_HEADER_SIZE = 3


def _bytes_per_variant(sample_count: int) -> int:
    return (int(sample_count) + 3) // 4


class BedMmapReader:
    """mmap-based BED reader with a sequential-prefetch fast path.

    The mmap region is opened ``ACCESS_READ`` and (on Linux) advised
    ``MADV_SEQUENTIAL | MADV_WILLNEED``. Arrays returned by
    ``read_all_packed`` / ``read_packed_range`` are zero-copy views over
    the mmap; **the caller MUST keep this reader alive** for as long as
    those arrays are in use. After ``close()`` (or ``__exit__``) the
    mmap is unmapped and any outstanding view becomes invalid.

    Random-access gathers (``read_packed_indexed``) are the SIGBUS risk
    case documented in ``sv_pgs.plink.open_bed``; we emit a warning and
    fall back to ``preadv`` on the first sign of trouble.
    """

    __slots__ = (
        "_count_a1",
        "_bpv",
        "_fd",
        "_file_size",
        "_mmap",
        "_n_samples",
        "_n_variants",
        "_path",
        "_payload_size",
    )

    def __init__(
        self,
        path: str | os.PathLike[str],
        n_samples: int,
        n_variants: int,
        count_a1: bool = True,
        allow_gcsfuse: bool = False,
    ) -> None:
        self._path: Path = Path(path)
        if n_samples < 1 or n_variants < 1:
            raise ValueError("n_samples and n_variants must be positive.")
        self._n_samples: int = int(n_samples)
        self._n_variants: int = int(n_variants)
        self._count_a1: bool = bool(count_a1)
        self._bpv: int = _bytes_per_variant(self._n_samples)
        self._payload_size: int = self._n_variants * self._bpv
        expected_size: int = PLINK1_HEADER_SIZE + self._payload_size

        self._fd: int | None = None
        self._mmap: mmap.mmap | None = None

        fd = os.open(str(self._path), os.O_RDONLY)
        try:
            actual_size = os.fstat(fd).st_size
            if actual_size != expected_size:
                raise ValueError(
                    "PLINK 1 .bed size does not match n_samples/n_variants: "
                    f"expected {expected_size} bytes, found {actual_size}"
                )
            header = os.pread(fd, PLINK1_HEADER_SIZE, 0)
            if header != PLINK1_MAGIC:
                raise ValueError(
                    "Invalid PLINK 1 .bed header: expected "
                    + PLINK1_MAGIC.hex(" ")
                    + ", got "
                    + header.hex(" ")
                )
            if not allow_gcsfuse:
                from sv_pgs.gcsfuse_staging import is_gcsfuse_path  # noqa: PLC0415

                if is_gcsfuse_path(self._path):
                    raise RuntimeError(
                        "Refusing to mmap a gcsfuse-backed BED: stage to local "
                        "NVMe first via sv_pgs.gcsfuse_staging.stage_bed_trio_to_local"
                    )
            self._file_size: int = actual_size
            mm = mmap.mmap(fd, actual_size, access=mmap.ACCESS_READ)
        except BaseException:
            os.close(fd)
            raise

        self._fd = fd
        self._mmap = mm
        self._advise_sequential()

    def _advise_sequential(self) -> None:
        """Hint the kernel to prefetch pages linearly; no-op where unsupported."""
        mm = self._mmap
        if mm is None:
            return
        madvise = getattr(mm, "madvise", None)
        if madvise is None:
            return
        seq = getattr(mmap, "MADV_SEQUENTIAL", None)
        will = getattr(mmap, "MADV_WILLNEED", None)
        for advice in (seq, will):
            if advice is None:
                continue
            try:
                madvise(advice)
            except (OSError, ValueError):
                # macOS / older kernels may reject; safe to ignore.
                pass

    # ------------------------------------------------------------------ utils
    def _check_open(self) -> mmap.mmap:
        mm = self._mmap
        if mm is None:
            raise RuntimeError("BedMmapReader is closed.")
        return mm

    @property
    def path(self) -> Path:
        return self._path

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def n_variants(self) -> int:
        return self._n_variants

    @property
    def bytes_per_variant(self) -> int:
        return self._bpv

    @property
    def count_a1(self) -> bool:
        return self._count_a1

    # ------------------------------------------------------------------ reads
    def read_all_packed(self) -> NDArray[np.uint8]:
        """Return the entire packed region as ``(n_variants, bpv)`` uint8.

        Zero-copy view over the mmap. The caller MUST keep this reader
        alive for as long as the returned array is in use; closing the
        reader unmaps the underlying memory.
        """
        mm = self._check_open()
        arr = np.frombuffer(
            mm,
            dtype=np.uint8,
            count=self._payload_size,
            offset=PLINK1_HEADER_SIZE,
        )
        # frombuffer always produces a non-owning view; copy=False is
        # implicit. Reshape to variant-major (n_variants, bpv).
        return arr.reshape(self._n_variants, self._bpv)

    def read_packed_range(self, start: int, stop: int) -> NDArray[np.uint8]:
        """Return ``packed[start:stop]`` as a zero-copy view."""
        mm = self._check_open()
        start_i = int(start)
        stop_i = int(stop)
        if start_i < 0 or stop_i < start_i or stop_i > self._n_variants:
            raise IndexError(
                f"variant range [{start_i}, {stop_i}) out of bounds for "
                f"n_variants={self._n_variants}"
            )
        n = stop_i - start_i
        if n == 0:
            return np.empty((0, self._bpv), dtype=np.uint8)
        byte_offset = PLINK1_HEADER_SIZE + start_i * self._bpv
        byte_count = n * self._bpv
        arr = np.frombuffer(
            mm,
            dtype=np.uint8,
            count=byte_count,
            offset=byte_offset,
        )
        return arr.reshape(n, self._bpv)

    def read_packed_indexed(self, indices: NDArray[np.integer]) -> NDArray[np.uint8]:
        """Gather variants by index.

        WARNING: random-access reads against an mmap are exactly the
        SIGBUS risk case documented in :class:`sv_pgs.plink.open_bed`.
        We attempt the gather, but wrap it in a guard that falls back to
        ``os.preadv`` (via ``sv_pgs.plink.open_bed``) on any
        OSError/SystemError. Prefer ``read_packed_range`` for hot paths.
        """
        warnings.warn(
            "BedMmapReader.read_packed_indexed: random-access mmap reads "
            "carry SIGBUS risk under sustained prefetch; will fall back to "
            "preadv on any read error. Prefer read_packed_range for "
            "sequential scans.",
            RuntimeWarning,
            stacklevel=2,
        )
        idx = np.ascontiguousarray(np.asarray(indices), dtype=np.int64)
        if idx.ndim != 1:
            raise ValueError("indices must be 1D.")
        if idx.size == 0:
            return np.empty((0, self._bpv), dtype=np.uint8)
        if idx.size > 0 and (int(idx.min()) < 0 or int(idx.max()) >= self._n_variants):
            raise IndexError("variant index out of bounds.")

        try:
            return self._gather_via_mmap(idx)
        except (OSError, SystemError, BufferError) as exc:
            warnings.warn(
                f"BedMmapReader.read_packed_indexed: mmap gather failed "
                f"({type(exc).__name__}: {exc}); falling back to preadv via "
                f"sv_pgs.plink.open_bed.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._gather_via_preadv(idx)

    # ------------------------------------------------------------ gather impls
    def _gather_via_mmap(self, idx: NDArray[np.int64]) -> NDArray[np.uint8]:
        mm = self._check_open()
        n = int(idx.size)
        out = np.empty((n, self._bpv), dtype=np.uint8)
        # We always allocate a fresh owned buffer here — copy=True semantics —
        # because the gather is by definition non-contiguous: there is no
        # single zero-copy view over the union of arbitrary rows. Materializing
        # the requested rows into a private array also drops the SIGBUS-risk
        # mmap reference before the caller starts using the result.
        bpv = self._bpv
        base = PLINK1_HEADER_SIZE
        # Use buffer-protocol slicing on the mmap object directly; each slice
        # copies bytes out of the mapping into the output row.
        for out_row, v in enumerate(idx.tolist()):
            offset = base + v * bpv
            out[out_row, :] = np.frombuffer(
                mm[offset : offset + bpv], dtype=np.uint8, count=bpv
            )
        return out

    def _gather_via_preadv(self, idx: NDArray[np.int64]) -> NDArray[np.uint8]:
        # Import lazily to avoid a circular import at module load.
        from sv_pgs.plink import open_bed  # noqa: PLC0415

        with open_bed(
            self._path,
            iid_count=self._n_samples,
            sid_count=self._n_variants,
            count_A1=self._count_a1,
        ) as reader:
            payload = reader._pread_indexed_variant_payload(  # noqa: SLF001
                idx, bytes_per_variant=self._bpv
            )
        return np.ascontiguousarray(payload).reshape(int(idx.size), self._bpv)

    # ------------------------------------------------------------- lifecycle
    def close(self) -> None:
        """Unmap and close. Idempotent."""
        mm = self._mmap
        if mm is not None:
            self._mmap = None
            try:
                mm.close()
            except (BufferError, OSError):
                # Outstanding numpy views can block close(); surface via
                # warning rather than raising on cleanup.
                print(
                    "BedMmapReader.close: outstanding views over the mmap "
                    "prevented unmap; release them before closing.",
                    file=sys.stderr,
                )
        fd = self._fd
        if fd is not None:
            self._fd = None
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self) -> BedMmapReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
