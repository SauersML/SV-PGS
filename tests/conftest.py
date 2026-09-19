from __future__ import annotations

import gc
import sys

import pytest


@pytest.fixture(autouse=True)
def clear_accelerator_caches():
    yield
    gc.collect()

    jax_module = sys.modules.get("jax")
    if jax_module is not None:
        clear_caches = getattr(jax_module, "clear_caches", None)
        if callable(clear_caches):
            clear_caches()

    cupy_module = sys.modules.get("cupy")
    if cupy_module is not None:
        get_default_memory_pool = getattr(cupy_module, "get_default_memory_pool", None)
        if callable(get_default_memory_pool):
            get_default_memory_pool().free_all_blocks()
        get_default_pinned_memory_pool = getattr(cupy_module, "get_default_pinned_memory_pool", None)
        if callable(get_default_pinned_memory_pool):
            get_default_pinned_memory_pool().free_all_blocks()
