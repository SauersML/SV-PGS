"""CUDA visibility reporting must not initialize or hide GPUs."""
from __future__ import annotations

from typing import Any

import pytest

from sv_pgs import _jax as jax_module


def test_most_free_cuda_device_noop_when_caller_already_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(
        jax_module,
        "_query_nvidia_smi",
        lambda _field: ("0, 100 MiB, 1000 MiB", "1, 900 MiB, 1000 MiB"),
    )

    assert jax_module._most_free_cuda_device() is None
    assert jax_module.os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_most_free_cuda_device_uses_nvidia_smi_without_env_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        jax_module,
        "_query_nvidia_smi",
        lambda _field: ("0, 5000 MiB, 16384 MiB", "1, 15000 MiB, 16384 MiB"),
    )

    assert jax_module._most_free_cuda_device() == (
        1,
        15000 * 1024 * 1024,
        16384 * 1024 * 1024,
    )
    assert "CUDA_VISIBLE_DEVICES" not in jax_module.os.environ


def test_most_free_cuda_device_ignores_bad_nvidia_smi_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def _fake_query(_field: str) -> tuple[str, ...]:
        return (
            "bad row",
            "0, missing, 16384 MiB",
            "1, 12000 MiB, 16384 MiB",
        )

    monkeypatch.setattr(jax_module, "_query_nvidia_smi", _fake_query)

    assert jax_module._most_free_cuda_device() == (
        1,
        12000 * 1024 * 1024,
        16384 * 1024 * 1024,
    )


def test_most_free_cuda_device_does_not_import_cupy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        jax_module,
        "_query_nvidia_smi",
        lambda _field: ("0, 1000 MiB, 16384 MiB",),
    )

    original_import = __import__

    def _blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "cupy":
            raise AssertionError("CUDA detection must not import cupy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _blocked_import)

    assert jax_module._most_free_cuda_device() == (
        0,
        1000 * 1024 * 1024,
        16384 * 1024 * 1024,
    )
