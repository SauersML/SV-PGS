"""Tests for the bitpacked engagement / fallback logging contract.

The production path silently fell back to int8 in three places (pipeline
upgrade, io variant stats, model marginal-z screen). These tests pin the
explicit ENGAGED/SKIPPED log lines and the host-RAM guardrail that refuses
the legacy int8 stats path when there is not enough free memory.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import sv_pgs.io as io_module
import sv_pgs.pipeline as pipeline_module


def _make_loaded_dataset(genotypes: Any) -> Any:
    """Build a minimal LoadedDataset around a fake genotype matrix."""
    return io_module.LoadedDataset(
        sample_ids=["s0", "s1", "s2"],
        genotypes=genotypes,
        covariates=np.zeros((3, 0), dtype=np.float32),
        targets=np.zeros((3,), dtype=np.float32),
        variant_records=[],
        variant_stats=None,
        variant_stats_minimum_scale=None,
    )


def _make_config_bitpacked() -> Any:
    """Return a stand-in ModelConfig that requests the bitpacked backend.

    SimpleNamespace is enough because ``_maybe_upgrade_to_bitpacked`` only
    reads ``genotype_backend``; nothing else on ``config`` is touched.
    """
    return SimpleNamespace(genotype_backend="bitpacked")


class _FakeBitpackedMatrix:
    """Stand-in BitpackedDeviceMatrix carrying just enough surface for the
    pipeline upgrade's success-log path (``_packed.nbytes`` etc.)."""

    def __init__(self, n_samples: int, n_variants: int) -> None:
        self.shape = (n_samples, n_variants)
        bytes_per_variant = (n_samples + 3) // 4
        self._packed = SimpleNamespace(nbytes=n_variants * bytes_per_variant)
        self._mean = SimpleNamespace(nbytes=n_variants * 4)
        self._std = SimpleNamespace(nbytes=n_variants * 4)


def test_pipeline_maybe_upgrade_logs_engaged(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_genotypes = SimpleNamespace(shape=(3, 4))
    dataset = _make_loaded_dataset(fake_genotypes)
    config = _make_config_bitpacked()

    fake_reader = SimpleNamespace(sid_count=4)
    monkeypatch.setattr(
        "sv_pgs.genotype._resolve_plink_pread_context",
        lambda _raw: (fake_reader, np.arange(3, dtype=np.int64), 3, "/dev/null/fake.bed"),
    )
    # cupy import probe must succeed; install a dummy module if cupy isn't
    # actually present in the CPU test env.
    import sys
    if "cupy" not in sys.modules:
        monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())

    bp_matrix = _FakeBitpackedMatrix(n_samples=3, n_variants=4)
    import sv_pgs.bitpacked_loader as bp_loader_module
    monkeypatch.setattr(
        bp_loader_module,
        "load_bed_to_bitpacked_device",
        lambda **_kwargs: bp_matrix,
    )

    out = pipeline_module._maybe_upgrade_to_bitpacked(dataset, config)
    assert out.genotypes is bp_matrix

    captured = capsys.readouterr().err
    assert "bitpacked upgrade: ENGAGED" in captured
    assert "active=4 variants" in captured


def test_pipeline_maybe_upgrade_logs_skipped_with_reason(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_genotypes = SimpleNamespace(shape=(3, 4))
    dataset = _make_loaded_dataset(fake_genotypes)
    config = _make_config_bitpacked()

    fake_reader = SimpleNamespace(sid_count=4)
    monkeypatch.setattr(
        "sv_pgs.genotype._resolve_plink_pread_context",
        lambda _raw: (fake_reader, np.arange(3, dtype=np.int64), 3, "/dev/null/fake.bed"),
    )
    import sys
    if "cupy" not in sys.modules:
        monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())

    import sv_pgs.bitpacked_loader as bp_loader_module

    def _explode(**_kwargs: Any) -> None:
        raise RuntimeError("loader test failure")

    monkeypatch.setattr(bp_loader_module, "load_bed_to_bitpacked_device", _explode)

    out = pipeline_module._maybe_upgrade_to_bitpacked(dataset, config)
    # On failure the dataset is returned unchanged (int8 fallback).
    assert out.genotypes is fake_genotypes

    captured = capsys.readouterr().err
    assert "bitpacked upgrade: SKIPPED" in captured
    assert "loader test failure" in captured


def test_io_bitpacked_stats_fallback_logs_reason(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Any,
) -> None:
    """When the screening pass raises, the helper returns None *and* logs a reason."""
    # Patch the cache lookup so we always take the live-compute branch.
    monkeypatch.setattr(io_module, "_load_plink_stats_from_cache", lambda _path: None)

    def _exploding_screen(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("screening_pipeline test failure")

    import sv_pgs.screening_pipeline as screening_module
    monkeypatch.setattr(screening_module, "run_screening_pass", _exploding_screen)

    bed_path = tmp_path / "fake.bed"
    bed_path.write_bytes(b"")
    config = SimpleNamespace(minimum_scale=1e-6)
    sample_indices = np.arange(8, dtype=np.int64)

    result = io_module._try_bitpacked_plink_variant_stats(
        bed_path=bed_path,
        sample_indices=sample_indices,
        n_samples=8,
        n_variants=16,
        config=config,  # type: ignore[arg-type]
    )
    assert result is None
    captured = capsys.readouterr().err
    assert "bitpacked path failed in run_screening_pass" in captured
    assert "screening_pipeline test failure" in captured


def test_io_memory_guardrail_refuses_when_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1 GB available — way below the multi-GB threshold the AoU shape demands.
    monkeypatch.setattr(io_module, "_available_host_memory_bytes", lambda: 1 * 1024**3)
    with pytest.raises(RuntimeError) as exc_info:
        io_module._legacy_int8_memory_guardrail(
            sample_count=80_000,
            variant_count=695_000,
        )
    message = str(exc_info.value)
    assert "bitpacked" in message.lower()
    assert "available" in message.lower()


def test_io_memory_guardrail_passes_when_ample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1 TiB available — must not refuse.
    monkeypatch.setattr(io_module, "_available_host_memory_bytes", lambda: 1024**4)
    # No exception expected.
    io_module._legacy_int8_memory_guardrail(
        sample_count=80_000,
        variant_count=695_000,
    )


def test_io_memory_guardrail_skips_when_probe_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No psutil + no /proc/meminfo → degrades to a no-op (avoids false-positive
    # refusals on macOS dev hosts).
    monkeypatch.setattr(io_module, "_available_host_memory_bytes", lambda: None)
    io_module._legacy_int8_memory_guardrail(
        sample_count=80_000,
        variant_count=695_000,
    )
