"""OOM-safety guards for the prefetch autotune sizing.

Regression: on a 30 GB AoU container with MemAvailable=16.9 GB and 2 visible
GPUs, the autotune sized prefetch as ``depth=8 × batch=976 MB = ~8 GB`` for
EACH of 2 shards, which combined with the fit's ~5 GB working set + result
accumulators tripped the OOM killer at RSS=11 GB. The fix:

1. Reserve a fixed amount of RAM (4 GB by default) for the fit working set
   BEFORE sizing the prefetch queue.
2. When multi-GPU sharding is enabled (cupy device_count >= 2), divide the
   prefetch RAM budget across shards so the TOTAL inflight bytes
   (``depth × batch_bytes × n_shards``) fits in the usable budget.

These tests pin those invariants on a synthesized 16 GB box and a 256 GB
production AoU box, plus the single-GPU case where ``n_shards=1`` must
not regress versus the historical sizing.
"""
from __future__ import annotations

import builtins
import io
from typing import Any

import pytest

from sv_pgs import genotype


def _patch_proc_meminfo(monkeypatch: pytest.MonkeyPatch, mem_available_bytes: int) -> None:
    """Patch ``/proc/meminfo`` reads to advertise the given MemAvailable."""
    kb = mem_available_bytes // 1024
    contents = (
        f"MemTotal:       {kb + 2_000_000} kB\n"
        f"MemFree:           200000 kB\n"
        f"MemAvailable:   {kb} kB\n"
        f"Cached:         200000000 kB\n"
        f"SReclaimable:    10000000 kB\n"
    )
    real_open = builtins.open

    def fake_open(path: Any, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        if str(path) == "/proc/meminfo":
            return io.StringIO(contents)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)


def _patch_no_gpu_detection(monkeypatch: pytest.MonkeyPatch, *, device_count: int) -> None:
    """Force ``_detect_cuda_device_count`` / ``_detect_gpu_free_bytes``."""
    monkeypatch.setattr(genotype, "_detect_cuda_device_count", lambda: int(device_count))
    monkeypatch.setattr(genotype, "_detect_gpu_free_bytes", lambda: 0)


def test_usable_prefetch_reserves_fit_working_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """16 GB MemAvailable -> usable prefetch RAM = 16 - 4 reserve = 12 GB."""
    host_ram = 16 * 1024**3
    usable = genotype.compute_usable_prefetch_ram_bytes(host_ram)
    assert usable == 12 * 1024**3


def test_usable_prefetch_floor_on_tiny_box() -> None:
    """A 2 GB box (smaller than the reserve) still gets the 1 GB minimum."""
    usable = genotype.compute_usable_prefetch_ram_bytes(2 * 1024**3)
    assert usable == genotype._AUTO_TUNE_MIN_USABLE_PREFETCH_BYTES


def test_two_shard_total_inflight_fits_within_usable(monkeypatch: pytest.MonkeyPatch) -> None:
    """16 GB MemAvailable + 2 GPUs: depth × batch × 2 shards <= 12 GB usable."""
    _patch_proc_meminfo(monkeypatch, 16 * 1024**3)
    _patch_no_gpu_detection(monkeypatch, device_count=2)
    monkeypatch.setattr(genotype.os, "cpu_count", lambda: 22)

    bed_bytes = genotype.compute_bed_reader_target_batch_bytes()
    depth = genotype.compute_plink_int8_max_prefetch_depth(target_batch_bytes=bed_bytes)
    usable = genotype.compute_usable_prefetch_ram_bytes()

    total_inflight = depth * bed_bytes * 2  # 2 shards
    assert total_inflight <= usable, (
        f"OOM risk: depth={depth} × batch={bed_bytes / 1024**3:.2f} GiB × 2 shards "
        f"= {total_inflight / 1024**3:.2f} GiB > usable={usable / 1024**3:.2f} GiB"
    )

    # And the grand RAM budget (prefetch + reserve) stays inside MemAvailable.
    grand_total = total_inflight + genotype._AUTO_TUNE_FIT_WORKING_MEM_RESERVE_BYTES
    assert grand_total <= 16 * 1024**3


def test_snapshot_emits_new_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """The autotune snapshot exposes fit reserve + usable + per-shard / total depth."""
    _patch_proc_meminfo(monkeypatch, 16 * 1024**3)
    _patch_no_gpu_detection(monkeypatch, device_count=2)
    monkeypatch.setattr(genotype.os, "cpu_count", lambda: 22)
    state = genotype._snapshot_autotune_state()
    assert state["fit_working_mem_reserve_bytes"] == 4 * 1024**3
    assert state["usable_prefetch_ram_bytes"] == 12 * 1024**3
    assert state["n_shards"] == 2
    assert state["plink_int8_max_prefetch_depth_total"] == (
        state["plink_int8_max_prefetch_depth"] * state["n_shards"]
    )


def test_production_256gb_box_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    """A production 256 GB AoU box still respects the 4 GB reserve."""
    host_ram = 256 * 1024**3
    _patch_proc_meminfo(monkeypatch, host_ram)
    _patch_no_gpu_detection(monkeypatch, device_count=2)
    monkeypatch.setattr(genotype.os, "cpu_count", lambda: 64)

    usable = genotype.compute_usable_prefetch_ram_bytes(host_ram)
    assert usable == (256 - 4) * 1024**3

    bed_bytes = genotype.compute_bed_reader_target_batch_bytes(host_ram_bytes=host_ram, n_shards=2)
    depth = genotype.compute_plink_int8_max_prefetch_depth(
        host_ram_bytes=host_ram,
        target_batch_bytes=bed_bytes,
        n_shards=2,
    )
    # Hard cap: a single batch never exceeds 4 GB even on huge boxes.
    assert bed_bytes <= genotype._AUTO_TUNE_BED_BATCH_HARD_CAP_BYTES
    # Total inflight fits comfortably.
    total_inflight = depth * bed_bytes * 2
    assert total_inflight <= usable
    # And depth stays at least 1.
    assert depth >= 1


def test_single_gpu_path_does_not_regress(monkeypatch: pytest.MonkeyPatch) -> None:
    """n_shards=1 (single GPU / CPU-only) sizing must remain sensible.

    Specifically: depth >= 1, batch_bytes >= floor, and inflight fits.
    """
    host_ram = 16 * 1024**3
    _patch_proc_meminfo(monkeypatch, host_ram)
    _patch_no_gpu_detection(monkeypatch, device_count=1)
    monkeypatch.setattr(genotype.os, "cpu_count", lambda: 16)

    bed_bytes = genotype.compute_bed_reader_target_batch_bytes()
    depth = genotype.compute_plink_int8_max_prefetch_depth(target_batch_bytes=bed_bytes)
    usable = genotype.compute_usable_prefetch_ram_bytes()
    assert bed_bytes >= genotype._AUTO_TUNE_BED_BATCH_FLOOR_BYTES
    assert depth >= 1
    # Single-shard total inflight fits within usable.
    assert depth * bed_bytes <= usable


def test_per_batch_ceiling_caps_huge_box(monkeypatch: pytest.MonkeyPatch) -> None:
    """A batch never exceeds 50% of usable RAM (so >=2 slots stay queueable)."""
    host_ram = 16 * 1024**3
    _patch_proc_meminfo(monkeypatch, host_ram)
    _patch_no_gpu_detection(monkeypatch, device_count=1)
    monkeypatch.setattr(genotype.os, "cpu_count", lambda: 1)  # collapses host_share

    bed_bytes = genotype.compute_bed_reader_target_batch_bytes()
    usable = genotype.compute_usable_prefetch_ram_bytes()
    assert bed_bytes <= usable * genotype._AUTO_TUNE_BED_BATCH_USABLE_FRACTION + 1
