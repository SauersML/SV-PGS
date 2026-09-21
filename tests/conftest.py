from __future__ import annotations

import gc
import sys

import pytest

# Tests over 60 s, measured by bug-recent's MSI durations run (cn1001, 2 BLAS threads, main 8666cfa), with
# seconds and owner lane. They carry the `slow` marker: CI and the MSI landing gate run -m "not slow", and
# bug-recent's periodic MSI job runs -m slow. A test sped up below 60 s leaves this table; the owner lane
# derives the speed-up rather than shrinking the problem arbitrarily.
SLOW_TESTS = {
    "tests/test_learned_density_nests_families.py::test_the_engine_certifies_a_start_on_a_tpb_truth": (893, "e2e"),
    "tests/test_learned_density_nests_families.py::test_the_learned_density_is_within_its_akaike_allowance_of_every_nested_fit[bayesr]": (696, "e2e"),
    "tests/test_learned_density_nests_families.py::test_the_learned_density_is_within_its_akaike_allowance_of_every_nested_fit[tpb]": (330, "e2e"),
    "tests/test_scale_mixture_ep.py::test_the_hyper_steps_evidence_is_at_least_the_exact_infinity_edges": (411, "e2e"),
    "tests/test_scale_mixture_ep.py::test_hyper_step_reaches_a_maximum_of_the_evidence": (379, "e2e"),
    "tests/test_scale_mixture_ep.py::test_the_fit_does_not_depend_on_the_lattice_spacing": (368, "e2e"),
    "tests/test_phenotype_measurement.py::test_gross_errors_are_downweighted_by_the_learned_density": (174, "deslop-hygiene"),
    "tests/test_all_of_us_sql.py::test_gross_errors_reach_the_model_and_are_downweighted_by_its_learned_density": (161, "deslop-hygiene"),
    # λ ascent at an interior optimum on 40 real-LD variants (acl42, 1 BLAS thread).
    "tests/test_ep_eb_reference.py::test_fit_on_real_ld_is_certified[0.3-40]": (225, "oracle"),
}


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark the SLOW_TESTS, before -m and -k deselect anything.

    When whole modules are collected (no ``::`` selection), an entry whose module was collected but which
    itself was not is stale, and collection fails, so the table cannot outlive the tests it names.
    """
    collected = {item.nodeid for item in items}
    if not any("::" in argument for argument in config.args):
        modules = {nodeid.split("::")[0] for nodeid in collected}
        stale = sorted(nodeid for nodeid in SLOW_TESTS if nodeid.split("::")[0] in modules and nodeid not in collected)
        if stale:
            raise pytest.UsageError("SLOW_TESTS names tests that no longer exist: " + ", ".join(stale))
    for item in items:
        if item.nodeid in SLOW_TESTS:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def clear_accelerator_caches():
    """Return every test's device memory to the driver, so GPU tests don't inherit each other's pools."""
    yield
    gc.collect()
    cupy_module = sys.modules.get("cupy")
    if cupy_module is not None:
        cupy_module.get_default_memory_pool().free_all_blocks()
        cupy_module.get_default_pinned_memory_pool().free_all_blocks()
