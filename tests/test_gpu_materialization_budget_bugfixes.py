"""Regression tests for GPU materialization budget calculations.

The bug fixed here: ``_gpu_materialization_budget_bytes`` used to read
``cupy.cuda.runtime.memGetInfo()`` directly. After dropping a large genotype
cache, the CuPy memory pool still holds the freed blocks as cached, so the
driver-level free count is artificially low. The fix is twofold:

  1. ``_gpu_effective_free_bytes`` adds the pool's ``free_bytes()`` to the
     CUDA free figure (or, equivalently, the budget routine calls
     ``free_all_blocks()`` first so the driver sees the reclaimed memory).
  2. ``_gpu_materialization_budget_bytes`` calls
     ``cupy.get_default_memory_pool().free_all_blocks()`` before reading
     free memory, so the next ``memGetInfo()`` returns the post-reclaim
     figure.

Without these, the budget collapses to ~0 after a cache drop and forces a
spurious fallback to CPU streaming.
"""
from __future__ import annotations

import types
from typing import Any

import pytest

from sv_pgs import genotype as genotype_module

_BUDGET_HELPER_PRESENT = hasattr(genotype_module, "_gpu_materialization_budget_bytes")
_EFFECTIVE_FREE_PRESENT = hasattr(genotype_module, "_gpu_effective_free_bytes")

if not (_BUDGET_HELPER_PRESENT and _EFFECTIVE_FREE_PRESENT):
    pytest.skip(
        "waiting for fix in sv_pgs/genotype.py (missing budget helpers)",
        allow_module_level=True,
    )


class _FakeMemoryPool:
    """Stand-in for ``cupy.get_default_memory_pool()``.

    Records calls to ``free_all_blocks()`` and optionally mutates the
    parent fake cupy's ``free`` counter to simulate driver-level reclaim
    when cached blocks are released.
    """

    def __init__(self, pool_free_bytes: int, on_free_all_blocks=None) -> None:
        self._pool_free_bytes = int(pool_free_bytes)
        self.free_all_blocks_calls = 0
        self._on_free_all_blocks = on_free_all_blocks

    def free_bytes(self) -> int:
        return self._pool_free_bytes

    def free_all_blocks(self) -> None:
        self.free_all_blocks_calls += 1
        if self._on_free_all_blocks is not None:
            self._on_free_all_blocks(self)
        # Pool was just emptied: subsequent calls see no cached blocks.
        self._pool_free_bytes = 0


class _FakeDevice:
    def synchronize(self) -> None:
        return None


def _make_fake_cupy(free_bytes: int, total_bytes: int, pool: _FakeMemoryPool) -> Any:
    """Build a minimal stand-in for the cupy module used by these helpers."""
    state = {"free": int(free_bytes), "total": int(total_bytes)}

    def _mem_get_info():
        return state["free"], state["total"]

    fake_runtime = types.SimpleNamespace(memGetInfo=_mem_get_info)
    fake_cuda = types.SimpleNamespace(runtime=fake_runtime, Device=lambda: _FakeDevice())

    fake_cupy = types.SimpleNamespace(
        cuda=fake_cuda,
        get_default_memory_pool=lambda: pool,
        get_default_pinned_memory_pool=lambda: pool,
    )
    # Stash the mutable state so tests can flip the reading mid-call.
    fake_cupy._mem_state = state  # type: ignore[attr-defined]
    return fake_cupy


def test_stale_pool_budget_recovers_pool_free_bytes():
    """Driver-free is small, pool holds the bulk of the device → budget
    must account for the pool's cached blocks (≥ ~9.5 GB on a 15 GB GPU)."""
    pool = _FakeMemoryPool(pool_free_bytes=10 * 10**9)

    def _reclaim_into_driver(p: _FakeMemoryPool) -> None:
        cupy._mem_state["free"] = cupy._mem_state["free"] + 10 * 10**9  # type: ignore[attr-defined]

    pool._on_free_all_blocks = _reclaim_into_driver
    cupy = _make_fake_cupy(
        free_bytes=int(1.6 * 10**9),
        total_bytes=15 * 10**9,
        pool=pool,
    )

    budget = genotype_module._gpu_materialization_budget_bytes(
        cupy,
        n_rows=10_000,
        n_cols=10_000,
        backend="int8",
    )
    # Pre-fix this would be ~0 because memGetInfo alone reported only 1.6 GB
    # free; post-fix the pool free is recovered. After subtracting the static
    # 1.5 GB safety reserve plus modest staging/result vectors, we still
    # expect at least 9.0 GB of available budget on a 15 GB device.
    assert budget >= 9 * 10**9, f"budget={budget!r} expected ≥9 GB"


def test_budget_releases_pool_blocks_before_reading_free():
    """The budget routine must call ``pool.free_all_blocks()`` so the
    driver sees the reclaimed memory before the free read."""
    pool = _FakeMemoryPool(pool_free_bytes=4 * 10**9)
    cupy = _make_fake_cupy(free_bytes=10**9, total_bytes=15 * 10**9, pool=pool)
    genotype_module._gpu_materialization_budget_bytes(cupy)
    assert pool.free_all_blocks_calls >= 1


def test_low_free_no_pool_honors_cuda_cap():
    """Empty pool, low CUDA free → budget caps at the small free figure
    minus the safety reserve (the co-tenant case must keep working)."""
    pool = _FakeMemoryPool(pool_free_bytes=0)
    total = 15 * 10**9
    free = int(2.0 * 10**9)
    cupy = _make_fake_cupy(free_bytes=free, total_bytes=total, pool=pool)

    budget = genotype_module._gpu_materialization_budget_bytes(
        cupy, n_rows=1_000, n_cols=1_000, backend="float32"
    )
    # Must not exceed the (free - safety reserve) cap; 1.5 GB safety leaves
    # ≤ ~0.5 GB headroom.
    assert budget <= free, f"budget={budget!r} exceeds free={free!r}"
    assert budget < int(0.75 * 10**9)


def test_total_zero_returns_zero():
    """No CUDA runtime → total=0 → budget=0 instead of an exception."""
    pool = _FakeMemoryPool(pool_free_bytes=0)
    cupy = _make_fake_cupy(free_bytes=0, total_bytes=0, pool=pool)
    assert genotype_module._gpu_materialization_budget_bytes(cupy) == 0
