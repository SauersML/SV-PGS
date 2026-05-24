"""Self-pin to the most-free CUDA device on multi-GPU boxes.

The orchestrator (``sv_pgs.aou_runner``) already pins per-disease subprocesses
via ``CUDA_VISIBLE_DEVICES`` before spawning. But for direct single-fit
invocations on a 2-V100 box, nobody was pinning the process and CuPy would
silently default to device 0 while JAX could spread work across both devices.

``sv_pgs._jax._self_pin_cuda_visible_device`` fixes this: it probes for the
most-free device and sets ``CUDA_VISIBLE_DEVICES`` BEFORE jax initializes,
but only when (a) the caller hasn't already pinned and (b) more than one
device is visible (single-GPU is a no-op).

These tests pin that behaviour using a fake cupy namespace — no real GPU
required, same pattern as ``tests/test_v100_budget_pinning.py``.
"""
from __future__ import annotations

import importlib
import os
import sys
import types
from typing import Any

import pytest


def _make_fake_cupy(devices: list[tuple[int, int]]) -> Any:
    """Build a SimpleNamespace mimicking cupy.cuda.* used by the pin helper.

    ``devices`` is a list of ``(free_bytes, total_bytes)`` tuples; one entry
    per visible CUDA device.
    """
    state = {"current": 0, "devices": list(devices)}

    class _FakeDevice:
        def __init__(self, device_id: int = 0) -> None:
            self._device_id = int(device_id)
            self._prev = state["current"]

        def __enter__(self) -> "_FakeDevice":
            self._prev = state["current"]
            state["current"] = self._device_id
            return self

        def __exit__(self, *args: Any) -> None:
            state["current"] = self._prev

    def _get_device_count() -> int:
        return len(state["devices"])

    def _mem_get_info() -> tuple[int, int]:
        return state["devices"][state["current"]]

    fake_runtime = types.SimpleNamespace(
        getDeviceCount=_get_device_count,
        memGetInfo=_mem_get_info,
    )
    fake_cuda = types.SimpleNamespace(runtime=fake_runtime, Device=_FakeDevice)
    return types.SimpleNamespace(cuda=fake_cuda)


@pytest.fixture
def jax_module():
    """Import sv_pgs._jax fresh so we can call its helpers without touching
    the real cached SELECTED_CUDA_DEVICE / env mutations."""
    return importlib.import_module("sv_pgs._jax")


def _patch_cupy(monkeypatch: pytest.MonkeyPatch, fake_cupy: Any | None) -> None:
    """Install ``fake_cupy`` as ``import cupy`` for the duration of the test.

    Pass ``None`` to simulate "cupy not importable" (ImportError on import).
    """
    if fake_cupy is None:
        monkeypatch.setitem(sys.modules, "cupy", None)  # type: ignore[arg-type]
    else:
        monkeypatch.setitem(sys.modules, "cupy", fake_cupy)


def test_self_pin_noop_when_caller_already_pinned(
    monkeypatch: pytest.MonkeyPatch, jax_module: Any
) -> None:
    """Aou_runner subprocess case: CUDA_VISIBLE_DEVICES already set by the
    parent — self-pin must NOT mutate the env (subprocess pin takes precedence)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    _patch_cupy(
        monkeypatch,
        _make_fake_cupy([(10 * 10**9, 16 * 10**9), (15 * 10**9, 16 * 10**9)]),
    )
    result = jax_module._self_pin_cuda_visible_device()
    # Helper may return a probed triple (None when CUDA_VISIBLE_DEVICES is set,
    # since _most_free_cuda_device early-exits) but MUST NOT change the env.
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert result is None  # _most_free_cuda_device early-exits when env is set


def test_self_pin_noop_when_zero_devices_visible(
    monkeypatch: pytest.MonkeyPatch, jax_module: Any
) -> None:
    """No GPU box: helper must NOT introduce a stray CUDA_VISIBLE_DEVICES."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _patch_cupy(monkeypatch, _make_fake_cupy([]))
    result = jax_module._self_pin_cuda_visible_device()
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert result is None


def test_self_pin_noop_when_single_device_visible(
    monkeypatch: pytest.MonkeyPatch, jax_module: Any
) -> None:
    """Single-GPU box: pinning gains nothing; helper must leave env clean."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _patch_cupy(
        monkeypatch,
        _make_fake_cupy([(12 * 10**9, 16 * 10**9)]),
    )
    result = jax_module._self_pin_cuda_visible_device()
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    # Helper still returns the probed triple so the banner can show free/total.
    assert result is not None
    device_id, free_bytes, total_bytes = result
    assert device_id == 0
    assert free_bytes == 12 * 10**9
    assert total_bytes == 16 * 10**9


def test_self_pin_picks_most_free_when_multi_gpu(
    monkeypatch: pytest.MonkeyPatch, jax_module: Any
) -> None:
    """2-V100 box: helper must pin to the device with the most free bytes."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    # device 1 is the most-free.
    _patch_cupy(
        monkeypatch,
        _make_fake_cupy(
            [
                (5 * 10**9, 16 * 10**9),
                (15 * 10**9, 16 * 10**9),
            ]
        ),
    )
    result = jax_module._self_pin_cuda_visible_device()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    assert result is not None
    device_id, free_bytes, total_bytes = result
    assert device_id == 1
    assert free_bytes == 15 * 10**9
    assert total_bytes == 16 * 10**9
