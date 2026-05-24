"""Pin the wave-5713d85 GPU budget win on a V100-class device.

Wave 5713d85 made the CG working-set fit GPU-resident as int8, which is
the single biggest perf win in the recent perf-commit batch (measured 118x
matvec speedup vs streaming on a small synthetic on V100; see
``swarm_findings/verdicts/v100_benchmark.md``).

The win is structural: at the real AoU working-set shape (~332k samples by
~18.8k variants in int8), the storage cost is ~6.3 GB which **must** fit
under the int8-resident solver budget on a 16 GB device. If anyone tweaks
``_gpu_solver_headroom_bytes``, ``_GPU_RESERVED_OVERHEAD_BYTES``, or the
``int8-resident`` budget math such that this stops fitting, CG falls back
to BED streaming and the AoU run-time regresses from minutes to hours.

These tests use the same fake-cupy fixture pattern as the budget bugfix
suite so they run on any host (no GPU required); the measured numbers
were cross-checked against a real V100 16 GB box.
"""
from __future__ import annotations

import types
from typing import Any

import numpy as np
import pytest

from sv_pgs import genotype as genotype_module


_PLAN_HELPER_PRESENT = hasattr(
    genotype_module, "_estimate_gpu_materialization_memory_plan"
)

if not _PLAN_HELPER_PRESENT:
    pytest.skip(
        "waiting for _estimate_gpu_materialization_memory_plan in sv_pgs/genotype.py",
        allow_module_level=True,
    )


class _FakeMemoryPool:
    def __init__(self, pool_free_bytes: int = 0) -> None:
        self._pool_free_bytes = int(pool_free_bytes)
        self.free_all_blocks_calls = 0

    def free_bytes(self) -> int:
        return self._pool_free_bytes

    def free_all_blocks(self) -> None:
        self.free_all_blocks_calls += 1


class _FakeDevice:
    def __enter__(self) -> "_FakeDevice":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def synchronize(self) -> None:
        return None


def _make_fake_cupy(free_bytes: int, total_bytes: int) -> Any:
    pool = _FakeMemoryPool(pool_free_bytes=0)
    state = {"free": int(free_bytes), "total": int(total_bytes)}

    def _mem_get_info():
        return state["free"], state["total"]

    fake_runtime = types.SimpleNamespace(memGetInfo=_mem_get_info)
    fake_cuda = types.SimpleNamespace(runtime=fake_runtime, Device=lambda: _FakeDevice())

    return types.SimpleNamespace(
        cuda=fake_cuda,
        get_default_memory_pool=lambda: pool,
        get_default_pinned_memory_pool=lambda: pool,
    )


# V100 16 GB-class device free figures, as reported by the actual box.
_V100_TOTAL_BYTES = 16_945_512_448  # ~16.94 GB
_V100_FREE_BYTES = 16_605_741_056   # ~16.59 GB (cold device)


def test_cg_workset_fits_on_v100_int8_resident():
    """The wave-5713d85 win: 332k samples x 18.8k variants int8 must fit
    under the int8-resident budget on a fresh 16 GB V100."""
    cupy = _make_fake_cupy(free_bytes=_V100_FREE_BYTES, total_bytes=_V100_TOTAL_BYTES)
    plan = genotype_module._estimate_gpu_materialization_memory_plan(
        n_rows=332_000,
        n_cols=18_865,
        dtype=np.int8,
        backend="int8-resident",
        cupy=cupy,
    )
    # ~6.26 GB required; budget ≥ ~13 GB on a cold V100. Pin generously
    # so chunk-size knobs can move without breaking the test.
    assert plan.fits, (
        f"CG working set must fit on 16 GB V100 int8-resident; "
        f"required={plan.required_bytes/1e9:.2f}GB "
        f"budget={plan.budget_bytes/1e9:.2f}GB"
    )
    assert plan.required_bytes <= 7 * 10**9
    assert plan.budget_bytes >= 10 * 10**9


def test_full_active_variants_correctly_does_not_fit_int8_resident():
    """Sanity: at 332k samples x 695k variants the int8 store is ~231 GB,
    which obviously cannot fit a 16 GB V100. This pin guards against a
    silent ``fits=True`` regression that would let install attempt a
    doomed allocation."""
    cupy = _make_fake_cupy(free_bytes=_V100_FREE_BYTES, total_bytes=_V100_TOTAL_BYTES)
    plan = genotype_module._estimate_gpu_materialization_memory_plan(
        n_rows=332_000,
        n_cols=695_875,
        dtype=np.int8,
        backend="int8-resident",
        cupy=cupy,
    )
    assert not plan.fits
    assert plan.required_bytes > plan.budget_bytes


def test_dense_fp32_oversize_correctly_does_not_fit():
    """100k x 50k float32 = 20 GB > 16 GB V100 — must report fits=False."""
    cupy = _make_fake_cupy(free_bytes=_V100_FREE_BYTES, total_bytes=_V100_TOTAL_BYTES)
    plan = genotype_module._estimate_gpu_materialization_memory_plan(
        n_rows=100_000,
        n_cols=50_000,
        dtype=np.float32,
        backend="dense",
        cupy=cupy,
    )
    assert not plan.fits


def test_int8_resident_700k_columns_correctly_does_not_fit():
    """100k samples x 700k variants int8 = 70 GB > 16 GB V100."""
    cupy = _make_fake_cupy(free_bytes=_V100_FREE_BYTES, total_bytes=_V100_TOTAL_BYTES)
    plan = genotype_module._estimate_gpu_materialization_memory_plan(
        n_rows=100_000,
        n_cols=700_000,
        dtype=np.int8,
        backend="int8-resident",
        cupy=cupy,
    )
    assert not plan.fits


def test_modest_workset_fits_with_room_to_spare():
    """A small synthetic the V100 micro-benchmark uses (8k x 4k float32)
    must fit comfortably; pin so the dense path keeps working."""
    cupy = _make_fake_cupy(free_bytes=_V100_FREE_BYTES, total_bytes=_V100_TOTAL_BYTES)
    plan = genotype_module._estimate_gpu_materialization_memory_plan(
        n_rows=8_000,
        n_cols=4_000,
        dtype=np.float32,
        backend="fp32-resident",
        cupy=cupy,
    )
    assert plan.fits
    assert plan.required_bytes < 200 * 10**6  # ~0.13 GB
