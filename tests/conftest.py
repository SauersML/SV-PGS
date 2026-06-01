from __future__ import annotations

import gc
import sys
import types
from typing import Any

import numpy as np
import pytest

from sv_pgs.config import VariantClass
from sv_pgs.data import VariantRecord


def make_fake_cupy(**overrides: Any) -> types.ModuleType:
    """A numpy-backed stand-in for the ``cupy`` module.

    Any attribute that is not explicitly overridden falls back (via PEP 562
    module ``__getattr__``) to the matching numpy symbol, so production code
    that reaches for a new ``cp.<fn>`` keeps working without per-test fake
    maintenance — the single reason the previous hand-rolled fakes broke when
    a code path started calling ``cp.isfinite``. Device-only helpers
    (``asnumpy``) and a minimal ``cuda`` namespace are provided; pass keyword
    overrides to replace any attribute (e.g. ``linalg=...``).
    """
    module = types.ModuleType("cupy")

    def _module_getattr(name: str) -> Any:
        attribute = getattr(np, name, None)
        if attribute is None:
            raise AttributeError(f"fake cupy has no attribute {name!r}")
        return attribute

    module.__getattr__ = _module_getattr  # type: ignore[attr-defined]
    module.asnumpy = lambda array: np.asarray(array)  # type: ignore[attr-defined]
    module.get_default_memory_pool = lambda: types.SimpleNamespace(  # type: ignore[attr-defined]
        free_all_blocks=lambda: None
    )
    module.get_default_pinned_memory_pool = lambda: types.SimpleNamespace(  # type: ignore[attr-defined]
        free_all_blocks=lambda: None
    )
    module.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        Device=lambda *args, **kwargs: types.SimpleNamespace(synchronize=lambda: None),
        runtime=types.SimpleNamespace(getDeviceCount=lambda: 1),
        Stream=types.SimpleNamespace(null=types.SimpleNamespace(synchronize=lambda: None)),
    )
    for attribute_name, attribute_value in overrides.items():
        setattr(module, attribute_name, attribute_value)
    return module


@pytest.fixture
def random_generator() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(autouse=True)
def reset_cupy_import_cache():
    """Reset ``genotype._try_import_cupy``'s process-global module cache.

    ``_try_import_cupy`` memoizes the imported module in module globals
    (``_cupy_checked`` / ``_cupy_module``). On a GPU host the first test that
    touches CuPy caches the *real* module, and later tests that monkeypatch
    ``sys.modules['cupy']`` with a numpy-backed fake would still receive the
    cached real module — leaking real device arrays into numpy-only code and
    causing CuPy-13 implicit-conversion errors. Clearing the cache before and
    after each test makes ``_try_import_cupy`` honor each test's own
    ``sys.modules`` patch and removes the cross-test pollution.
    """
    import sv_pgs.genotype as genotype_module

    genotype_module._cupy_checked = False
    genotype_module._cupy_module = None
    yield
    genotype_module._cupy_checked = False
    genotype_module._cupy_module = None


@pytest.fixture(autouse=True)
def reset_bitpacked_profiler_globals():
    """Reset the process-global bitpacked-profiler state between tests.

    ``bitpacked_profile`` keeps a process-global cumulative accumulator and a
    ``_SYNC_ENABLED`` flag. ``model.fit`` flips ``enable_cuda_sync(True)`` and
    any test running bitpacked GEMVs accumulates into ``_CUMULATIVE``; neither
    is reset, so state leaks across tests — e.g. ``test_record_and_reset_round_trip``
    asserts ``cumulative == per_iter`` from a clean slate, which only holds if
    no earlier test (or a still-running EM thread) polluted the global. Drain
    and reset before and after each test for isolation.
    """
    bitpacked_profile = sys.modules.get("sv_pgs.bitpacked_profile")

    def _reset() -> None:
        module = sys.modules.get("sv_pgs.bitpacked_profile")
        if module is None:
            return
        reset_cumulative = getattr(module, "reset_cumulative", None)
        if callable(reset_cumulative):
            reset_cumulative()
        snapshot_and_reset = getattr(module, "snapshot_and_reset", None)
        if callable(snapshot_and_reset):
            snapshot_and_reset()  # drop this thread's leftover per-iter bucket
        enable_cuda_sync = getattr(module, "enable_cuda_sync", None)
        if callable(enable_cuda_sync):
            enable_cuda_sync(False)

    if bitpacked_profile is not None:
        _reset()
    yield
    _reset()


@pytest.fixture(autouse=True)
def join_bitpacked_cache_writers():
    """Join the bitpacked active-matrix cache-writer daemon threads each test.

    ``load_bed_to_bitpacked_device_cached`` dispatches the on-disk cache write
    to a background daemon thread and returns immediately. Those threads are
    not joined by the production code (the process exits when done), so across
    a test session a writer from an earlier test can still be running — and
    running bitpacked GEMVs that mutate the process-global
    ``bitpacked_profile._CUMULATIVE`` accumulator, or holding GPU/cache state —
    while a later test executes. That cross-test leakage is what makes
    ``test_record_and_reset_round_trip`` / the bitpacked PLINK e2e tests pass in
    isolation but flake under full-suite ordering. Draining the writers at
    teardown restores per-test isolation.
    """
    yield
    bitpacked_loader = sys.modules.get("sv_pgs.bitpacked_loader")
    if bitpacked_loader is None:
        return
    writer_threads = getattr(bitpacked_loader, "_ACTIVE_CACHE_WRITER_THREADS", None)
    if not writer_threads:
        return
    for writer_thread in list(writer_threads):
        writer_thread.join(timeout=60)


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
