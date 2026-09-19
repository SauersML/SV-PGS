"""Tests for the bitpacked variant-stats fallback logging contract.

The io variant-stats path used to fall back to int8 silently. This test pins
the explicit log line that names why the bitpacked path was abandoned.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import sv_pgs.io as io_module


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


