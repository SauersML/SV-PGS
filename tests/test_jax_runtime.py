from __future__ import annotations

from sv_pgs._jax import _is_turing_gpu


def test_turing_detection_uses_compute_capability() -> None:
    assert _is_turing_gpu(device_names=(), compute_capabilities=("7.5",))


def test_turing_detection_uses_device_name() -> None:
    assert _is_turing_gpu(device_names=("NVIDIA Tesla T4",), compute_capabilities=())


def test_non_turing_gpu_does_not_enable_workarounds() -> None:
    assert not _is_turing_gpu(device_names=("NVIDIA A100-SXM4-40GB",), compute_capabilities=("8.0",))
