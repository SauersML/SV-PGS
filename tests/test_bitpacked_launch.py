from __future__ import annotations

import pytest

pytest.importorskip("sv_pgs.bitpacked.launch")

from sv_pgs.bitpacked.launch import (  # noqa: E402
    gemm_gram_config,
    gemv_nt_config,
    gemv_tn_config,
    gpu_arch,
    screening_config,
)

_KNOWN_ARCHS = ("t4", "volta", "ampere", "hopper", "unknown")
_REQUIRED_KEYS = ("grid", "block", "shmem_bytes")
_CONFIG_FNS = (
    gemv_nt_config,
    gemv_tn_config,
    gemm_gram_config,
    screening_config,
)


def test_gpu_arch_returns_known_value() -> None:
    arch = gpu_arch()
    assert arch in _KNOWN_ARCHS, f"unexpected arch: {arch!r}"


def test_launch_configs_return_dicts_with_required_keys() -> None:
    n_samples = 1024
    n_variants = 256
    arch = "ampere"
    for fn in _CONFIG_FNS:
        cfg = fn(n_samples, n_variants, arch)
        assert isinstance(cfg, dict), f"{fn.__name__} did not return a dict"
        for key in _REQUIRED_KEYS:
            assert key in cfg, f"{fn.__name__} missing key {key!r}"
        assert isinstance(cfg["grid"], tuple), (
            f"{fn.__name__}['grid'] should be a tuple"
        )
        assert isinstance(cfg["block"], tuple), (
            f"{fn.__name__}['block'] should be a tuple"
        )
        assert isinstance(cfg["shmem_bytes"], int), (
            f"{fn.__name__}['shmem_bytes'] should be an int"
        )


def test_launch_configs_for_all_archs() -> None:
    n_samples = 512
    n_variants = 128
    for arch in _KNOWN_ARCHS:
        for fn in _CONFIG_FNS:
            cfg = fn(n_samples, n_variants, arch)
            assert cfg, f"{fn.__name__}({arch!r}) returned empty config"
            assert len(cfg["grid"]) > 0
            assert len(cfg["block"]) > 0
