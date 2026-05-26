from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal, Union

from sv_pgs._jax import (
    _compute_capabilities,
    _device_names,
)

Family = Literal["volta", "turing", "ampere", "ada", "hopper", "unknown"]
LegacyFamily = Literal["t4", "volta", "ampere", "hopper", "unknown"]

_LEGACY_TO_FAMILY: dict[str, Family] = {
    "t4": "turing",
    "volta": "volta",
    "ampere": "ampere",
    "hopper": "hopper",
    "unknown": "unknown",
}

_FAMILY_TO_LEGACY: dict[Family, LegacyFamily] = {
    "volta": "volta",
    "turing": "t4",
    "ampere": "ampere",
    "ada": "ampere",
    "hopper": "hopper",
    "unknown": "unknown",
}


@dataclass(frozen=True, eq=False)
class GpuArch:
    device_id: int
    compute_capability: tuple[int, int]
    sm: int
    family: Family
    name: str = field(default="")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GpuArch):
            return (
                self.device_id == other.device_id
                and self.compute_capability == other.compute_capability
                and self.sm == other.sm
                and self.family == other.family
                and self.name == other.name
            )
        if isinstance(other, str):
            if other == self.family:
                return True
            return other == _FAMILY_TO_LEGACY.get(self.family, "unknown")
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("GpuArch", self.family, self.sm))

    def __str__(self) -> str:
        return self.family


ArchLike = Union[GpuArch, str]


def _family_from_sm(sm: int) -> Family:
    if sm >= 90:
        return "hopper"
    if sm == 89:
        return "ada"
    if 80 <= sm < 89:
        return "ampere"
    if sm == 75:
        return "turing"
    if sm == 70 or sm == 72:
        return "volta"
    return "unknown"


def _parse_cc(capability: str) -> tuple[int, int] | None:
    token = str(capability).strip()
    if not token:
        return None
    parts = token.split(".", 1)
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return None
    return (major, minor)


def _probe_via_cupy() -> tuple[int, tuple[int, int], str] | None:
    try:
        import cupy as cp
    except (ImportError, OSError, RuntimeError):
        return None
    try:
        device = cp.cuda.Device()
        device_id = int(device.id)
        cc = device.compute_capability
        cc_str = str(cc)
        if cc_str.isdigit() and len(cc_str) >= 2:
            major = int(cc_str[0])
            minor = int(cc_str[1:])
            cc_tuple = (major, minor)
        else:
            parsed = _parse_cc(cc_str)
            if parsed is None:
                return None
            cc_tuple = parsed
        name = ""
        try:
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            raw_name = props.get("name", b"")
            if isinstance(raw_name, bytes):
                name = raw_name.decode("utf-8", errors="replace")
            else:
                name = str(raw_name)
        except (AttributeError, KeyError, OSError, RuntimeError):
            name = ""
        return device_id, cc_tuple, name
    except (AttributeError, OSError, RuntimeError):
        return None


def _probe_via_nvidia_smi() -> tuple[int, tuple[int, int], str] | None:
    capabilities = _compute_capabilities()
    names = _device_names()
    if not capabilities:
        return None
    parsed = _parse_cc(capabilities[0])
    if parsed is None:
        return None
    name = names[0] if names else ""
    return 0, parsed, name


@lru_cache(maxsize=1)
def gpu_arch() -> GpuArch:
    probe = _probe_via_cupy()
    if probe is None:
        probe = _probe_via_nvidia_smi()
    if probe is None:
        return GpuArch(
            device_id=-1,
            compute_capability=(0, 0),
            sm=0,
            family="unknown",
            name="",
        )
    device_id, (major, minor), name = probe
    sm = major * 10 + minor
    return GpuArch(
        device_id=device_id,
        compute_capability=(major, minor),
        sm=sm,
        family=_family_from_sm(sm),
        name=name,
    )


def gpu_arch_family() -> str:
    """Deprecated. Returns the legacy family string ("t4"|"ampere"|"hopper"|"unknown")."""
    warnings.warn(
        "gpu_arch_family() is deprecated; use gpu_arch().family instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _FAMILY_TO_LEGACY.get(gpu_arch().family, "unknown")


def _resolve_arch(arch: ArchLike) -> tuple[Family, int]:
    if isinstance(arch, GpuArch):
        return arch.family, arch.sm
    if isinstance(arch, str):
        token = arch.strip().lower()
        if token in _LEGACY_TO_FAMILY:
            family: Family = _LEGACY_TO_FAMILY[token]
        elif token in ("volta", "turing", "ampere", "ada", "hopper", "unknown"):
            family = token  # type: ignore[assignment]
        else:
            family = "unknown"
        sm_default = {
            "volta": 70,
            "turing": 75,
            "ampere": 80,
            "ada": 89,
            "hopper": 90,
            "unknown": 0,
        }[family]
        return family, sm_default
    return "unknown", 0


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def gemv_nt_config(n_samples: int, n_variants: int, arch: ArchLike) -> dict:
    family, _sm = _resolve_arch(arch)
    if family == "hopper":
        block_x = 384
        shmem = 64 * 1024
    elif family == "ampere":
        block_x = 256
        shmem = 48 * 1024
    elif family == "ada":
        block_x = 256
        shmem = 32 * 1024
    elif family == "turing":
        block_x = 256
        shmem = 16 * 1024
    elif family == "volta":
        block_x = 256
        shmem = 16 * 1024
    else:
        block_x = 256
        shmem = 16 * 1024
    grid_x = max(1, _ceil_div(n_samples, block_x))
    return {
        "grid": (grid_x, 1, 1),
        "block": (block_x, 1, 1),
        "shmem_bytes": shmem,
    }


def gemv_tn_config(n_samples: int, n_variants: int, arch: ArchLike) -> dict:
    family, _sm = _resolve_arch(arch)
    if family == "hopper":
        block_x = 384
        shmem = 64 * 1024
    elif family == "ampere":
        block_x = 256
        shmem = 48 * 1024
    elif family == "ada":
        block_x = 256
        shmem = 32 * 1024
    elif family == "turing":
        block_x = 256
        shmem = 16 * 1024
    elif family == "volta":
        block_x = 256
        shmem = 16 * 1024
    else:
        block_x = 256
        shmem = 16 * 1024
    grid_x = max(1, n_variants)
    return {
        "grid": (grid_x, 1, 1),
        "block": (block_x, 1, 1),
        "shmem_bytes": shmem,
    }


def gemm_gram_config(n_samples: int, n_variants: int, arch: ArchLike) -> dict:
    family, _sm = _resolve_arch(arch)
    if family == "hopper":
        tile_m, tile_n = 128, 256
        block_x = 256
        shmem = 96 * 1024
    elif family == "ampere":
        tile_m, tile_n = 128, 128
        block_x = 256
        shmem = 64 * 1024
    elif family == "ada":
        tile_m, tile_n = 128, 128
        block_x = 256
        shmem = 48 * 1024
    elif family == "turing":
        tile_m, tile_n = 64, 64
        block_x = 128
        shmem = 16 * 1024
    elif family == "volta":
        # V100: 64x64 output tile, 128 threads (4 warps), ~16 KB shmem cap
        # leaves room for the WMMA scratch buffer used in the volta_mma kernel.
        tile_m, tile_n = 64, 64
        block_x = 128
        shmem = 16 * 1024
    else:
        tile_m, tile_n = 64, 64
        block_x = 128
        shmem = 16 * 1024
    grid_x = max(1, _ceil_div(n_variants, tile_m))
    grid_y = max(1, _ceil_div(n_variants, tile_n))
    return {
        "grid": (grid_x, grid_y, 1),
        "block": (block_x, 1, 1),
        "shmem_bytes": shmem,
        "tile_m": tile_m,
        "tile_n": tile_n,
    }


def screening_config(n_samples: int, n_variants: int, arch: ArchLike) -> dict:
    family, _sm = _resolve_arch(arch)
    if family == "hopper":
        block_x = 384
        shmem = 32 * 1024
    elif family == "ampere":
        block_x = 256
        shmem = 24 * 1024
    elif family == "ada":
        block_x = 256
        shmem = 20 * 1024
    elif family == "turing":
        block_x = 256
        shmem = 12 * 1024
    elif family == "volta":
        block_x = 256
        shmem = 12 * 1024
    else:
        block_x = 256
        shmem = 12 * 1024
    grid_x = max(1, n_variants)
    return {
        "grid": (grid_x, 1, 1),
        "block": (block_x, 1, 1),
        "shmem_bytes": shmem,
    }
