from __future__ import annotations

import gc
import sys

import pytest


@pytest.fixture(autouse=True)
def clear_accelerator_caches():
    """Return every test's device memory to the driver, so GPU tests don't inherit each other's pools."""
    yield
    gc.collect()
    cupy_module = sys.modules.get("cupy")
    if cupy_module is not None:
        cupy_module.get_default_memory_pool().free_all_blocks()
        cupy_module.get_default_pinned_memory_pool().free_all_blocks()
