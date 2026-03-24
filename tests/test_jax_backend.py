from __future__ import annotations

from dataclasses import dataclass

import pytest

from sv_pgs.config import JaxDevicePreference, ModelConfig
from sv_pgs.jax_backend import resolve_single_device


@dataclass
class _FakeDevice:
    platform: str
    id: int


def test_resolve_single_device_prefers_gpu(monkeypatch):
    fake_devices = [
        _FakeDevice(platform="cpu", id=0),
        _FakeDevice(platform="gpu", id=0),
        _FakeDevice(platform="gpu", id=1),
    ]
    monkeypatch.setattr("sv_pgs.jax_backend.jax.devices", lambda: fake_devices)

    device = resolve_single_device(ModelConfig(jax_device_preference=JaxDevicePreference.GPU))

    assert device.platform == "gpu"
    assert device.id == 0


def test_resolve_single_device_honors_device_index(monkeypatch):
    fake_devices = [
        _FakeDevice(platform="gpu", id=0),
        _FakeDevice(platform="gpu", id=1),
    ]
    monkeypatch.setattr("sv_pgs.jax_backend.jax.devices", lambda: fake_devices)

    device = resolve_single_device(
        ModelConfig(
            jax_device_preference=JaxDevicePreference.GPU,
            jax_device_index=1,
        )
    )

    assert device.platform == "gpu"
    assert device.id == 1


def test_resolve_single_device_raises_when_required_platform_missing(monkeypatch):
    fake_devices = [_FakeDevice(platform="cpu", id=0)]
    monkeypatch.setattr("sv_pgs.jax_backend.jax.devices", lambda: fake_devices)

    with pytest.raises(RuntimeError, match="unavailable"):
        resolve_single_device(
            ModelConfig(
                jax_device_preference=JaxDevicePreference.GPU,
                require_jax_device=True,
            )
        )
