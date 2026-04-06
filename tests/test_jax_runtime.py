from __future__ import annotations

import sv_pgs._jax as jax_module
from sv_pgs._jax import _is_turing_gpu


def test_turing_detection_uses_compute_capability() -> None:
    assert _is_turing_gpu(device_names=(), compute_capabilities=("7.5",))


def test_turing_detection_uses_device_name() -> None:
    assert _is_turing_gpu(device_names=("NVIDIA Tesla T4",), compute_capabilities=())


def test_non_turing_gpu_does_not_enable_workarounds() -> None:
    assert not _is_turing_gpu(device_names=("NVIDIA A100-SXM4-40GB",), compute_capabilities=("8.0",))


def test_turing_gpu_disables_dense_jax_linear_algebra_preference(monkeypatch) -> None:
    monkeypatch.setattr(jax_module, "turing_workarounds_enabled", lambda: True)
    assert not jax_module.jax_dense_linear_algebra_preferred()


def test_turing_gpu_enables_fast_math_only_with_cupy_runtime(monkeypatch) -> None:
    monkeypatch.setattr(jax_module, "turing_workarounds_enabled", lambda: True)
    monkeypatch.setattr(jax_module, "_cupy_runtime_status", lambda: (True, "cupy_devices=1"))
    assert jax_module.t4_fast_math_enabled()

    monkeypatch.setattr(jax_module, "_cupy_runtime_status", lambda: (False, "cupy_unavailable"))
    assert not jax_module.t4_fast_math_enabled()
