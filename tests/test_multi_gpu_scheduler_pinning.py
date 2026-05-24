"""Tests for the All of Us all-disease scheduler.

Covers `sv_pgs.aou_runner.run_all_of_us_all_diseases` and its helpers:

  * `_detect_gpu_count` returns 0 on the no-GPU path and integers otherwise.
  * Pool size derives correctly from (detected_gpu_count, n_diseases,
    explicit user override).
  * The default sweep keeps one disease in-process at a time so each fit sees
    all visible GPUs for model-level sharding.
  * Explicit per-disease concurrency pins subprocesses with
    `CUDA_VISIBLE_DEVICES=<gpu_id>` round-robin across the pool.
  * A subprocess failure causes the orchestrator to log + continue, then
    return a non-zero exit code at the end so CI catches it.
  * Sequential fallback (pool_size==1) still goes through the in-process
    `run_all_of_us` path rather than spawning subprocesses.

These tests intentionally do NOT exercise real CUDA, gsutil, or the
training pipeline — they mock at the subprocess.Popen and
run_all_of_us boundaries so they run on a laptop in milliseconds.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from sv_pgs import aou_runner


@dataclass
class _StubDisease:
    canonical_name: str
    snomed_code: str = "0"
    snomed_concept_name: str = ""


def _patch_diseases(monkeypatch: pytest.MonkeyPatch, names: list[str]) -> None:
    monkeypatch.setattr(
        aou_runner,
        "DISEASE_DEFINITIONS",
        [_StubDisease(canonical_name=name) for name in names],
    )


def _patch_pre_flight_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aou_runner, "_log_all_cached_test_evals", lambda *_a, **_kw: 0)


class _FakePopen:
    """Records the env it was launched with and returns a configurable exit code."""

    instances: list["_FakePopen"] = []

    def __init__(self, cmd: list[str], env: dict[str, str] | None = None, **_kwargs: Any):
        self.cmd = list(cmd)
        self.env = dict(env or {})
        # Default success; tests override via `_FakePopen.returncode_for`.
        disease = self._extract_disease(cmd)
        self.returncode = _FakePopen.returncode_for.get(disease, 0)
        self.disease = disease
        _FakePopen.instances.append(self)

    @staticmethod
    def _extract_disease(cmd: list[str]) -> str:
        if "--disease" in cmd:
            return cmd[cmd.index("--disease") + 1]
        return ""

    def communicate(self) -> tuple[str, str]:
        return "", ("simulated stderr tail" if self.returncode != 0 else "")

    def terminate(self) -> None:  # pragma: no cover - not exercised here
        pass

    def kill(self) -> None:  # pragma: no cover
        pass

    def wait(self, timeout: float | None = None) -> int:  # pragma: no cover
        return self.returncode

    # Per-test overrides for which disease should fail.
    returncode_for: dict[str, int] = {}


@pytest.fixture(autouse=True)
def _reset_fake_popen() -> None:
    _FakePopen.instances = []
    _FakePopen.returncode_for = {}
    yield
    _FakePopen.instances = []
    _FakePopen.returncode_for = {}


# ---------------------------------------------------------------------------
# _detect_gpu_count
# ---------------------------------------------------------------------------


def test_detect_gpu_count_returns_zero_when_cupy_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """No cupy installed -> 0 (sequential CPU fallback)."""
    import builtins

    real_import = builtins.__import__

    def _import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "cupy":
            raise ImportError("cupy not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    assert aou_runner._detect_gpu_count() == 0


# ---------------------------------------------------------------------------
# Pool sizing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "detected,n_diseases,override,expected_pool,expected_wave_count",
    [
        # 0 GPUs -> sequential.
        (0, 5, None, 1, 5),
        # 1 GPU -> sequential (single-disease at a time).
        (1, 5, None, 1, 5),
        # 2 GPUs, 5 diseases -> default is sequential, each fit sees both GPUs.
        (2, 5, None, 1, 5),
        # 4 GPUs, 10 diseases -> default is still sequential all-GPU per fit.
        (4, 10, None, 1, 10),
        # 4 GPUs but only 2 diseases -> default sequential all-GPU per fit.
        (4, 2, None, 1, 2),
        # Explicit override: user forces sequential even on a 4-GPU host.
        (4, 8, 1, 1, 8),
        # Explicit override above detected count is capped at detected GPUs.
        (4, 6, 3, 3, 2),
        # Explicit override is ignored when no GPUs are visible.
        (0, 6, 3, 1, 6),
    ],
)
def test_pool_size_derivation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    detected: int,
    n_diseases: int,
    override: int | None,
    expected_pool: int,
    expected_wave_count: int,
) -> None:
    """`run_all_of_us_all_diseases` should size the pool from detect/override."""
    diseases = [f"disease_{i}" for i in range(n_diseases)]
    _patch_diseases(monkeypatch, diseases)
    _patch_pre_flight_noop(monkeypatch)
    monkeypatch.setattr(aou_runner, "_detect_gpu_count", lambda: detected)

    # Capture scheduling banner so we can assert the announced wave count.
    captured_logs: list[str] = []
    monkeypatch.setattr(aou_runner, "log", lambda msg: captured_logs.append(str(msg)))

    if expected_pool == 1:
        # Sequential path: stub the in-process run_all_of_us so no real work
        # happens. Track the call count to confirm we used the in-process path.
        calls: list[str] = []

        def _fake_run(disease: str, **_kwargs: Any) -> None:
            calls.append(disease)

        monkeypatch.setattr(aou_runner, "run_all_of_us", _fake_run)
        rc = aou_runner.run_all_of_us_all_diseases(
            chromosomes=[1],
            output_base=str(tmp_path),
            max_parallel_gpus=override,
        )
        assert rc == 0
        assert calls == diseases
        # No subprocesses should have been spawned in the sequential path.
        assert _FakePopen.instances == []
    else:
        monkeypatch.setattr(aou_runner.subprocess, "Popen", _FakePopen)
        rc = aou_runner.run_all_of_us_all_diseases(
            chromosomes=[1],
            output_base=str(tmp_path),
            max_parallel_gpus=override,
        )
        assert rc == 0
        assert len(_FakePopen.instances) == n_diseases

    banner = next(
        (line for line in captured_logs if line.startswith("multi-GPU scheduling")),
        None,
    )
    assert banner is not None, f"missing scheduling banner; logs={captured_logs}"
    assert f"{expected_pool} concurrent fit" in banner
    assert f"{expected_wave_count} wave" in banner


# ---------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES pinning
# ---------------------------------------------------------------------------


def test_default_sweep_runs_in_process_with_all_gpus_visible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    diseases = [f"d{i}" for i in range(4)]
    _patch_diseases(monkeypatch, diseases)
    _patch_pre_flight_noop(monkeypatch)
    monkeypatch.setattr(aou_runner, "_detect_gpu_count", lambda: 2)
    calls: list[str] = []

    def _fake_run(disease: str, **_kwargs: Any) -> None:
        calls.append(disease)

    monkeypatch.setattr(aou_runner, "run_all_of_us", _fake_run)
    monkeypatch.setattr(aou_runner, "log", lambda _msg: None)

    rc = aou_runner.run_all_of_us_all_diseases(
        chromosomes=[1, 2],
        output_base=str(tmp_path),
    )

    assert rc == 0
    assert calls == diseases
    assert _FakePopen.instances == []


def test_explicit_parallel_cuda_visible_devices_round_robin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Explicit concurrency pins subprocesses round-robin across GPU ids."""
    diseases = [f"d{i}" for i in range(6)]
    _patch_diseases(monkeypatch, diseases)
    _patch_pre_flight_noop(monkeypatch)
    monkeypatch.setattr(aou_runner, "_detect_gpu_count", lambda: 3)
    monkeypatch.setattr(aou_runner.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(aou_runner, "log", lambda _msg: None)

    rc = aou_runner.run_all_of_us_all_diseases(
        chromosomes=[1, 2],
        output_base=str(tmp_path),
        max_parallel_gpus=3,
    )
    assert rc == 0

    # Each disease should have been spawned exactly once, with its disease
    # name embedded in argv and CUDA_VISIBLE_DEVICES pinned per submission
    # index mod pool_size. We rebuild the mapping by submission order
    # (instances are appended at submission time).
    assert len(_FakePopen.instances) == len(diseases)
    submission_order = [inst.disease for inst in _FakePopen.instances]
    assert submission_order == diseases

    for i, inst in enumerate(_FakePopen.instances):
        expected_gpu = str(i % 3)
        assert inst.env.get("CUDA_VISIBLE_DEVICES") == expected_gpu, (
            f"{inst.disease} got CUDA_VISIBLE_DEVICES="
            f"{inst.env.get('CUDA_VISIBLE_DEVICES')!r}, expected {expected_gpu!r}"
        )
        # argv must invoke the same interpreter via `-m sv_pgs run-all-of-us`.
        assert "sv_pgs" in inst.cmd
        assert "run-all-of-us" in inst.cmd
        assert "--disease" in inst.cmd
        assert inst.cmd[inst.cmd.index("--disease") + 1] == inst.disease


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------


def test_subprocess_failure_continues_and_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One failing subprocess must not abort the sweep; final rc must be non-zero."""
    diseases = ["alpha", "beta", "gamma", "delta"]
    _patch_diseases(monkeypatch, diseases)
    _patch_pre_flight_noop(monkeypatch)
    monkeypatch.setattr(aou_runner, "_detect_gpu_count", lambda: 2)
    monkeypatch.setattr(aou_runner.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(aou_runner, "log", lambda _msg: None)

    # `beta` fails with a non-zero exit code; the other three succeed.
    _FakePopen.returncode_for = {"beta": 17}

    rc = aou_runner.run_all_of_us_all_diseases(
        chromosomes=[1],
        output_base=str(tmp_path),
        max_parallel_gpus=2,
    )
    # All 4 diseases still attempted...
    assert len(_FakePopen.instances) == 4
    spawned = {inst.disease for inst in _FakePopen.instances}
    assert spawned == set(diseases)
    # ...and the orchestrator returns non-zero so CI catches the failure.
    assert rc != 0


def test_sequential_path_failure_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Even in the no-GPU sequential path, in-process failures yield rc!=0."""
    diseases = ["alpha", "beta"]
    _patch_diseases(monkeypatch, diseases)
    _patch_pre_flight_noop(monkeypatch)
    monkeypatch.setattr(aou_runner, "_detect_gpu_count", lambda: 0)

    def _fake_run(disease: str, **_kwargs: Any) -> None:
        if disease == "alpha":
            raise RuntimeError("simulated failure")

    monkeypatch.setattr(aou_runner, "run_all_of_us", _fake_run)
    monkeypatch.setattr(aou_runner, "log", lambda _msg: None)

    rc = aou_runner.run_all_of_us_all_diseases(
        chromosomes=[1],
        output_base=str(tmp_path),
    )
    assert rc == 1
