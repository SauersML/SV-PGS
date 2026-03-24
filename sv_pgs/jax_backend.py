from __future__ import annotations

import jax

from sv_pgs.config import JaxDevicePreference, ModelConfig


def resolve_single_device(config: ModelConfig) -> jax.Device:
    available_devices = list(jax.devices())
    if not available_devices:
        raise RuntimeError("JAX did not report any available devices.")

    preferred_platform = config.jax_device_preference.value
    if config.jax_device_preference == JaxDevicePreference.DEFAULT:
        return _select_indexed_device(
            available_devices,
            device_index=config.jax_device_index,
            description="default JAX device",
        )

    matching_devices = [device for device in available_devices if device.platform == preferred_platform]
    if matching_devices:
        return _select_indexed_device(
            matching_devices,
            device_index=config.jax_device_index,
            description=preferred_platform + " device",
        )

    if config.require_jax_device:
        raise RuntimeError(
            "Requested JAX device platform "
            + preferred_platform
            + " is unavailable. Available platforms: "
            + ", ".join(sorted({device.platform for device in available_devices}))
            + "."
        )

    return _select_indexed_device(
        available_devices,
        device_index=min(config.jax_device_index, len(available_devices) - 1),
        description="fallback JAX device",
    )


def _select_indexed_device(
    devices: list[jax.Device],
    device_index: int,
    description: str,
) -> jax.Device:
    if device_index >= len(devices):
        raise ValueError(
            "Requested "
            + description
            + " index "
            + str(device_index)
            + " but only "
            + str(len(devices))
            + " device(s) are available."
        )
    return devices[device_index]
