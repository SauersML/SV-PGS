from __future__ import annotations

import shutil

import pytest

pytest.importorskip("sv_pgs.preflight")

from sv_pgs import preflight
from sv_pgs.preflight import (
    AouPreflightReport,
    assert_preflight_ok,
    check_aou_preflight,
)


def _set_baseline_env(monkeypatch):
    """Provide a 'happy-path' env so any test only needs to break one knob."""
    monkeypatch.setenv("CDR_STORAGE_PATH", "gs://fake-cdr/path")
    monkeypatch.setenv("WORKSPACE_BUCKET", "gs://fake-ws")
    monkeypatch.setenv("GOOGLE_PROJECT", "fake-project")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _stub_gpu_available(monkeypatch):
    monkeypatch.setattr(
        preflight, "_probe_nvidia_smi", lambda: ["GPU 0: NVIDIA T4 (UUID: GPU-fake)"]
    )
    monkeypatch.setattr(preflight, "_probe_cupy", lambda: (True, 1, None))
    monkeypatch.setattr(preflight, "_probe_cupy_nvrtc", lambda: None)


def _stub_lots_of_disk(monkeypatch):
    # Stub disk_usage to a generous 1 TB free so it never trips the budget test.
    fake_usage = shutil._ntuple_diskusage(total=2 * 10**12, used=10**12, free=10**12)
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda _p: fake_usage)


def test_preflight_local_cache_dir_passes(tmp_path, monkeypatch):
    _set_baseline_env(monkeypatch)
    _stub_gpu_available(monkeypatch)
    _stub_lots_of_disk(monkeypatch)
    # gcsfuse classifier should say "not gcsfuse"
    monkeypatch.setattr(preflight, "is_gcsfuse_path", lambda _p: False)

    report = check_aou_preflight(tmp_path, require_gpu=True)
    assert isinstance(report, AouPreflightReport)
    assert report.ok, report.fatal_errors
    assert report.cache_storage_class == "local_hot"


def test_preflight_missing_cdr_path_warns_not_fatal(tmp_path, monkeypatch):
    _set_baseline_env(monkeypatch)
    monkeypatch.delenv("CDR_STORAGE_PATH", raising=False)
    _stub_gpu_available(monkeypatch)
    _stub_lots_of_disk(monkeypatch)
    monkeypatch.setattr(preflight, "is_gcsfuse_path", lambda _p: False)

    report = check_aou_preflight(tmp_path, require_gpu=True)
    assert report.ok, report.fatal_errors
    assert any("CDR_STORAGE_PATH" in w for w in report.warnings)
    assert not any("CDR_STORAGE_PATH" in e for e in report.fatal_errors)


def test_preflight_jax_preallocate_unset_is_fatal(tmp_path, monkeypatch):
    _set_baseline_env(monkeypatch)
    # Now break preallocate.
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION", raising=False)
    _stub_gpu_available(monkeypatch)
    _stub_lots_of_disk(monkeypatch)
    monkeypatch.setattr(preflight, "is_gcsfuse_path", lambda _p: False)

    report = check_aou_preflight(tmp_path, require_gpu=True)
    assert not report.ok
    assert any("XLA_PYTHON_CLIENT_PREALLOCATE" in e for e in report.fatal_errors)


def test_preflight_insufficient_disk_is_fatal(tmp_path, monkeypatch):
    _set_baseline_env(monkeypatch)
    _stub_gpu_available(monkeypatch)
    monkeypatch.setattr(preflight, "is_gcsfuse_path", lambda _p: False)

    tiny_usage = shutil._ntuple_diskusage(total=10**9, used=10**9 - 1024, free=1024)
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda _p: tiny_usage)

    report = check_aou_preflight(tmp_path, require_gpu=True)
    assert not report.ok
    assert any("Insufficient free space" in e for e in report.fatal_errors)


def test_preflight_gpu_required_but_absent_is_fatal(tmp_path, monkeypatch):
    _set_baseline_env(monkeypatch)
    _stub_lots_of_disk(monkeypatch)
    monkeypatch.setattr(preflight, "is_gcsfuse_path", lambda _p: False)

    # No nvidia-smi devices, no cupy.
    monkeypatch.setattr(preflight, "_probe_nvidia_smi", lambda: [])
    monkeypatch.setattr(
        preflight, "_probe_cupy", lambda: (False, 0, "cupy import failed: stubbed")
    )

    report = check_aou_preflight(tmp_path, require_gpu=True)
    assert not report.ok
    joined = " | ".join(report.fatal_errors)
    assert "nvidia-smi" in joined
    assert "CuPy" in joined or "cupy" in joined


def test_preflight_nvrtc_missing_is_fatal(tmp_path, monkeypatch):
    _set_baseline_env(monkeypatch)
    _stub_lots_of_disk(monkeypatch)
    monkeypatch.setattr(preflight, "is_gcsfuse_path", lambda _p: False)
    monkeypatch.setattr(
        preflight, "_probe_nvidia_smi", lambda: ["GPU 0: NVIDIA A100 (UUID: GPU-fake)"]
    )
    monkeypatch.setattr(preflight, "_probe_cupy", lambda: (True, 1, None))
    # Simulate the AoU container symptom: runtime ok, but NVRTC dlopen fails.
    monkeypatch.setattr(
        preflight,
        "_probe_cupy_nvrtc",
        lambda: "RuntimeError: CuPy failed to load libnvrtc.so.12",
    )

    report = check_aou_preflight(tmp_path, require_gpu=True)
    assert not report.ok
    joined = " | ".join(report.fatal_errors)
    assert "NVRTC" in joined or "nvrtc" in joined
    assert "LD_LIBRARY_PATH" in joined


def test_assert_preflight_ok_raises_with_all_errors_listed(tmp_path, monkeypatch):
    _set_baseline_env(monkeypatch)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION", raising=False)
    monkeypatch.setattr(preflight, "is_gcsfuse_path", lambda _p: False)
    # No GPU and no disk.
    monkeypatch.setattr(preflight, "_probe_nvidia_smi", lambda: [])
    monkeypatch.setattr(
        preflight, "_probe_cupy", lambda: (False, 0, "cupy import failed: stubbed")
    )
    tiny_usage = shutil._ntuple_diskusage(total=10**9, used=10**9 - 1024, free=1024)
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda _p: tiny_usage)

    report = check_aou_preflight(tmp_path, require_gpu=True)
    assert not report.ok
    assert len(report.fatal_errors) >= 2

    with pytest.raises(RuntimeError) as excinfo:
        assert_preflight_ok(report)
    msg = str(excinfo.value)
    for err in report.fatal_errors:
        # Each fatal error message should be referenced (at least its first few words).
        head = err.split(":", 1)[0]
        assert head in msg
