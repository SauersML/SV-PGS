"""Tests for the persistent active-matrix cache in bitpacked_loader.

The cache machinery is exercised end-to-end (content hashing, atomic write,
verification, warm load) without a real CuPy install. A small numpy-backed
shim stands in for ``cupy``: it implements the minimum surface the cache
read/write paths touch — ``asnumpy``, ``asarray``, ``empty``, ``uint8``,
``float32``, ``ascontiguousarray``, ``Stream.null``, ``alloc_pinned_memory``,
and the host→device memcpy proxy.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import sv_pgs.bitpacked_loader as bp_loader


# ---------------------------------------------------------------------------
# CuPy shim sufficient for the cache write+load paths
# ---------------------------------------------------------------------------


class _PinnedAlloc:
    """Stand-in for cupy.cuda.PinnedMemoryPointer / alloc_pinned_memory.

    ``np.frombuffer`` requires an object exposing the buffer protocol, so we
    wrap an ordinary bytearray. The cache loader writes through the numpy
    view returned by ``_allocate_pinned``.
    """

    def __init__(self, nbytes: int) -> None:
        self._buf = bytearray(nbytes)

    def __buffer__(self, flags):  # noqa: D401 - PEP 688 buffer protocol
        return memoryview(self._buf).__buffer__(flags)

    # Pre-3.12 numpy still calls __array_interface__-friendly paths via
    # ``frombuffer``; bytearray implements the buffer protocol natively.
    @property
    def raw(self) -> bytearray:
        return self._buf


def _alloc_pinned_memory(nbytes: int) -> Any:
    # numpy.frombuffer can read directly from a bytearray.
    return bytearray(nbytes)


class _DeviceArray:
    """numpy-backed cupy.ndarray stand-in. Exposes ``data.ptr`` and ``nbytes``.

    Only the fields touched by the cache write+load code paths are populated.
    """

    def __init__(self, arr: np.ndarray) -> None:
        self._np = np.ascontiguousarray(arr)

    @property
    def nbytes(self) -> int:
        return int(self._np.nbytes)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._np.shape)

    @property
    def dtype(self) -> Any:
        return self._np.dtype

    @property
    def data(self) -> SimpleNamespace:
        return SimpleNamespace(ptr=int(self._np.ctypes.data))


class _Stream:
    ptr = 0

    def synchronize(self) -> None:
        return None


def _make_cupy_shim() -> Any:
    """Return a SimpleNamespace shaped like just enough of the cupy API."""

    def asnumpy(x: Any) -> np.ndarray:
        if isinstance(x, _DeviceArray):
            return np.array(x._np, copy=True)
        return np.asarray(x)

    def asarray(x: Any, dtype: Any = None) -> _DeviceArray:
        arr = np.asarray(x)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return _DeviceArray(arr)

    def empty(shape: Any, dtype: Any = None) -> _DeviceArray:
        return _DeviceArray(np.empty(shape, dtype=dtype))

    def ascontiguousarray(x: Any, dtype: Any = None) -> _DeviceArray:
        arr = np.ascontiguousarray(np.asarray(x), dtype=dtype) if dtype is not None else np.ascontiguousarray(np.asarray(x))
        return _DeviceArray(arr)

    runtime = SimpleNamespace(
        memcpyAsync=lambda *args, **kwargs: None,
        memcpyHostToDevice=0,
    )
    cuda = SimpleNamespace(
        Stream=SimpleNamespace(null=_Stream()),
        runtime=runtime,
        alloc_pinned_memory=_alloc_pinned_memory,
    )
    return SimpleNamespace(
        asnumpy=asnumpy,
        asarray=asarray,
        empty=empty,
        ascontiguousarray=ascontiguousarray,
        uint8=np.uint8,
        float32=np.float32,
        cuda=cuda,
    )


@pytest.fixture
def cupy_shim(monkeypatch: pytest.MonkeyPatch) -> Any:
    shim = _make_cupy_shim()
    monkeypatch.setattr(bp_loader, "_require_cupy", lambda: shim)
    return shim


def _make_fake_matrix(n_samples: int, n_variants: int) -> Any:
    bpv = (n_samples + 3) // 4
    rng = np.random.default_rng(7)
    packed = rng.integers(0, 256, size=(n_variants, bpv), dtype=np.uint8)
    mean = rng.random(size=(n_variants,)).astype(np.float32)
    std = (rng.random(size=(n_variants,)).astype(np.float32) + 0.5)
    matrix = SimpleNamespace(
        _packed=_DeviceArray(packed),
        _mean=_DeviceArray(mean),
        _std=_DeviceArray(std),
        _n_samples=int(n_samples),
        _n_variants=int(n_variants),
    )
    # Preserve the raw numpy buffers so tests can compare round-tripped data.
    matrix._packed_host = packed
    matrix._mean_host = mean
    matrix._std_host = std
    return matrix


# ---------------------------------------------------------------------------
# Hash + verify
# ---------------------------------------------------------------------------


def test_content_hash_changes_with_inputs(tmp_path: Path) -> None:
    bed = tmp_path / "x.bed"
    bed.write_bytes(b"x" * 32)
    h0 = bp_loader._active_cache_content_hash(
        bed_path=bed, sample_indices=None, variant_indices=None, count_a1=True
    )
    h1 = bp_loader._active_cache_content_hash(
        bed_path=bed,
        sample_indices=np.arange(4, dtype=np.int64),
        variant_indices=None,
        count_a1=True,
    )
    h2 = bp_loader._active_cache_content_hash(
        bed_path=bed, sample_indices=None, variant_indices=None, count_a1=False
    )
    assert h0 != h1
    assert h0 != h2
    # Stable under no input change.
    assert h0 == bp_loader._active_cache_content_hash(
        bed_path=bed, sample_indices=None, variant_indices=None, count_a1=True
    )


def test_verify_rejects_missing_or_incomplete(tmp_path: Path) -> None:
    cache_dir = tmp_path / "bitpacked_active" / "deadbeef"
    assert bp_loader.verify_active_matrix_cache(cache_dir) is False
    cache_dir.mkdir(parents=True)
    # Manifest absent.
    assert bp_loader.verify_active_matrix_cache(cache_dir) is False
    # Manifest present but complete=False.
    (cache_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": bp_loader._ACTIVE_CACHE_SCHEMA_VERSION,
                "complete": False,
                "n_samples": 0,
                "n_variants": 0,
                "packed_bytes": 0,
            }
        )
    )
    assert bp_loader.verify_active_matrix_cache(cache_dir) is False


# ---------------------------------------------------------------------------
# Write + load round trip
# ---------------------------------------------------------------------------


def test_cache_round_trip(tmp_path: Path, cupy_shim: Any) -> None:
    matrix = _make_fake_matrix(n_samples=11, n_variants=7)
    cache_root = tmp_path / "cache"
    bed_path = tmp_path / "x.bed"
    bed_path.write_bytes(b"x" * 64)
    content_hash = bp_loader._active_cache_content_hash(
        bed_path=bed_path,
        sample_indices=None,
        variant_indices=None,
        count_a1=True,
    )
    cache_subdir = bp_loader._active_cache_dir(cache_root, content_hash)
    assert bp_loader.verify_active_matrix_cache(cache_subdir) is False

    bp_loader._write_active_matrix_cache(
        cache_subdir,
        matrix=matrix,
        bed_path=bed_path,
        content_hash=content_hash,
        count_a1=True,
    )
    assert bp_loader.verify_active_matrix_cache(cache_subdir) is True

    # Load it back — stub BitpackedDeviceMatrix so we don't need the real
    # device-resident class wired through.
    captured: dict[str, Any] = {}

    class _StubMatrix:
        def __init__(self, *, packed, mean, std, n_samples, count_a1):  # noqa: D401
            captured["packed"] = packed
            captured["mean"] = mean
            captured["std"] = std
            captured["n_samples"] = n_samples
            captured["count_a1"] = count_a1

    import sys
    module = type(sys)("sv_pgs.bitpacked_matrix")
    module.BitpackedDeviceMatrix = _StubMatrix  # type: ignore[attr-defined]
    sys.modules["sv_pgs.bitpacked_matrix"] = module

    bp_loader._load_active_matrix_cache(cache_subdir, count_a1=True)

    assert captured["n_samples"] == matrix._n_samples
    assert captured["count_a1"] is True
    # Validate round-trip data: shim's empty+memcpy is a no-op, so the
    # device buffer here is empty; what matters is mean/std fidelity (those
    # are uploaded via cp.asarray on the actual host bytes).
    np.testing.assert_array_equal(captured["mean"]._np, matrix._mean_host)
    np.testing.assert_array_equal(captured["std"]._np, matrix._std_host)

    # And the packed.bin on disk matches the host bytes exactly.
    on_disk = np.fromfile(cache_subdir / "packed.bin", dtype=np.uint8)
    np.testing.assert_array_equal(
        on_disk, np.ascontiguousarray(matrix._packed_host).reshape(-1)
    )


# ---------------------------------------------------------------------------
# End-to-end: cache miss -> populate -> cache hit
# ---------------------------------------------------------------------------


def test_cached_wrapper_miss_then_hit(tmp_path: Path, cupy_shim: Any) -> None:
    matrix = _make_fake_matrix(n_samples=9, n_variants=5)
    bed_path = tmp_path / "x.bed"
    bed_path.write_bytes(b"x" * 128)
    cache_root = tmp_path / "cache"

    # Stub the cold-load implementation so we don't actually open a BED.
    call_counter = {"n": 0}

    def _fake_loader(**kwargs: Any) -> Any:
        call_counter["n"] += 1
        return matrix

    import sv_pgs.bitpacked_loader as mod
    import sys
    module = type(sys)("sv_pgs.bitpacked_matrix")

    # When loading from cache, we need a real-ish BitpackedDeviceMatrix stub.
    class _StubMatrix:
        def __init__(self, *, packed, mean, std, n_samples, count_a1):  # noqa: D401
            self._packed = packed
            self._mean = mean
            self._std = std
            self._n_samples = n_samples
            self._n_variants = int(mean._np.shape[0]) if hasattr(mean, "_np") else int(mean.shape[0])
            self._count_a1 = count_a1

    module.BitpackedDeviceMatrix = _StubMatrix  # type: ignore[attr-defined]
    sys.modules["sv_pgs.bitpacked_matrix"] = module

    orig_loader = mod.load_bed_to_bitpacked_device
    mod.load_bed_to_bitpacked_device = _fake_loader  # type: ignore[assignment]
    try:
        # Cache miss → fake cold loader runs, cache populated.
        m1 = mod.load_bed_to_bitpacked_device_cached(
            bed_path=bed_path,
            n_samples=9,
            n_variants=5,
            cache_dir=cache_root,
            count_a1=True,
        )
        assert m1 is matrix
        assert call_counter["n"] == 1

        # Cache hit → cold loader NOT called again.
        m2 = mod.load_bed_to_bitpacked_device_cached(
            bed_path=bed_path,
            n_samples=9,
            n_variants=5,
            cache_dir=cache_root,
            count_a1=True,
        )
        assert call_counter["n"] == 1  # still 1, cache served the request
        # m2 is a freshly constructed stub matrix carrying the cached fields.
        assert isinstance(m2, _StubMatrix)
        assert m2._n_samples == matrix._n_samples
        assert m2._n_variants == matrix._n_variants
    finally:
        mod.load_bed_to_bitpacked_device = orig_loader  # type: ignore[assignment]


def test_cached_wrapper_partial_write_rejected(tmp_path: Path, cupy_shim: Any) -> None:
    """An incomplete manifest (e.g. crash mid-write) must NOT serve a cache hit."""
    matrix = _make_fake_matrix(n_samples=9, n_variants=5)
    bed_path = tmp_path / "x.bed"
    bed_path.write_bytes(b"x" * 32)
    content_hash = bp_loader._active_cache_content_hash(
        bed_path=bed_path, sample_indices=None, variant_indices=None, count_a1=True
    )
    cache_subdir = bp_loader._active_cache_dir(tmp_path / "cache", content_hash)
    bp_loader._write_active_matrix_cache(
        cache_subdir,
        matrix=matrix,
        bed_path=bed_path,
        content_hash=content_hash,
        count_a1=True,
    )
    assert bp_loader.verify_active_matrix_cache(cache_subdir) is True
    # Simulate corruption: rewrite manifest with complete=False.
    manifest_path = bp_loader._active_cache_manifest_path(cache_subdir)
    data = json.loads(manifest_path.read_text())
    data["complete"] = False
    manifest_path.write_text(json.dumps(data))
    assert bp_loader.verify_active_matrix_cache(cache_subdir) is False
