from __future__ import annotations

import gc
import sys

import numpy as np
import pytest

from sv_pgs.config import VariantClass
from sv_pgs.data import VariantRecord


@pytest.fixture
def random_generator() -> np.random.Generator:
    return np.random.default_rng(42)


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


def make_variant_records(
    variant_count: int,
    variant_class: VariantClass = VariantClass.SNV,
    chromosome: str = "chr1",
) -> list[VariantRecord]:
    structural_variant_classes = {
        VariantClass.DELETION_SHORT,
        VariantClass.DELETION_LONG,
        VariantClass.DUPLICATION_SHORT,
        VariantClass.DUPLICATION_LONG,
        VariantClass.INSERTION_MEI,
        VariantClass.INVERSION_BND_COMPLEX,
        VariantClass.STR_VNTR_REPEAT,
        VariantClass.OTHER_COMPLEX_SV,
    }
    return [
        VariantRecord(
            variant_id="variant_" + str(variant_index),
            variant_class=variant_class,
            chromosome=chromosome,
            position=variant_index * 100,
            length=1.0,
            allele_frequency=0.1,
            quality=1.0,
            training_support=32 if variant_class in structural_variant_classes else None,
        )
        for variant_index in range(variant_count)
    ]
