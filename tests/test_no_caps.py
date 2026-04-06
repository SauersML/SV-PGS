"""Tests that enforce SPEC.md: no caps on variants or samples.

These tests exist because the linter repeatedly re-adds maximum_active_variants
and similar caps. If any of these tests fail, someone added a cap that violates
the spec. Delete the cap, not the test.
"""
import inspect
import pytest

from sv_pgs.config import ModelConfig
from sv_pgs.preprocessing import select_active_variant_indices
from sv_pgs.data import VariantRecord
from sv_pgs.config import VariantClass


# ---------------------------------------------------------------------------
# Config must not have any variant or sample cap fields
# ---------------------------------------------------------------------------

def test_config_has_no_maximum_active_variants():
    assert not hasattr(ModelConfig, "maximum_active_variants"), \
        "ModelConfig must not have maximum_active_variants (SPEC.md: no variant caps)"


def test_config_has_no_maximum_samples():
    assert not hasattr(ModelConfig, "maximum_samples"), \
        "ModelConfig must not have maximum_samples (SPEC.md: no sample caps)"


def test_config_has_no_max_variants():
    assert not hasattr(ModelConfig, "max_variants"), \
        "ModelConfig must not have max_variants (SPEC.md: no variant caps)"


def test_config_has_no_max_samples():
    assert not hasattr(ModelConfig, "max_samples"), \
        "ModelConfig must not have max_samples (SPEC.md: no sample caps)"


def test_config_has_no_screen_max_fields():
    for field_name in dir(ModelConfig):
        assert "screen_max" not in field_name.lower(), \
            f"ModelConfig.{field_name} looks like a screening cap (SPEC.md: no variant caps)"


def test_config_has_no_budget_fields():
    for field_name in dir(ModelConfig):
        if field_name.startswith("_"):
            continue
        assert "variant_budget" not in field_name.lower(), \
            f"ModelConfig.{field_name} looks like a variant budget (SPEC.md: no variant caps)"


def test_config_constructor_rejects_maximum_active_variants_kwarg():
    with pytest.raises(TypeError):
        ModelConfig(maximum_active_variants=100)


def test_config_constructor_rejects_max_variants_kwarg():
    with pytest.raises(TypeError):
        ModelConfig(max_variants=100)


def test_config_constructor_rejects_maximum_samples_kwarg():
    with pytest.raises(TypeError):
        ModelConfig(maximum_samples=100)


# ---------------------------------------------------------------------------
# select_active_variant_indices must not cap
# ---------------------------------------------------------------------------

def test_select_active_keeps_all_variants_above_maf():
    n = 10000
    records = [
        VariantRecord(f"v{i}", VariantClass.DELETION_SHORT, "1", i, allele_frequency=0.05)
        for i in range(n)
    ]
    result = select_active_variant_indices(records, ModelConfig(minimum_minor_allele_frequency=0.001))
    assert len(result) == n, f"Expected all {n} variants kept, got {len(result)}"


def test_select_active_keeps_100k_variants():
    n = 100_000
    records = [
        VariantRecord(f"v{i}", VariantClass.SNV, "1", i, allele_frequency=0.1)
        for i in range(n)
    ]
    result = select_active_variant_indices(records, ModelConfig(minimum_minor_allele_frequency=0.0))
    assert len(result) == n


def test_select_active_has_no_screening_params():
    sig = inspect.signature(select_active_variant_indices)
    param_names = set(sig.parameters.keys())
    banned = {"max_active", "maximum_active", "budget", "screen", "cap", "limit"}
    for name in param_names:
        for ban in banned:
            assert ban not in name.lower(), \
                f"select_active_variant_indices has param '{name}' that looks like a cap"


# ---------------------------------------------------------------------------
# Source code scanning: no caps hiding anywhere
# ---------------------------------------------------------------------------

def test_config_source_has_no_maximum_active_variants():
    source = inspect.getsource(ModelConfig)
    assert "maximum_active_variants" not in source, \
        "ModelConfig source contains 'maximum_active_variants' — delete it"


def test_config_source_has_no_maximum_samples():
    source = inspect.getsource(ModelConfig)
    assert "maximum_samples" not in source, \
        "ModelConfig source contains 'maximum_samples' — delete it"


def test_config_source_has_no_screen_max():
    source = inspect.getsource(ModelConfig)
    assert "screen_max" not in source.lower(), \
        "ModelConfig source contains screening cap — delete it"


def test_preprocessing_source_has_no_maximum_active_variants():
    import sv_pgs.preprocessing as mod
    source = inspect.getsource(mod)
    assert "maximum_active_variants" not in source, \
        "preprocessing module contains 'maximum_active_variants' — delete it"


def test_no_production_module_contains_maximum_active_variants():
    """Scan ALL production modules for the banned string."""
    import sv_pgs.config
    import sv_pgs.model
    import sv_pgs.preprocessing
    import sv_pgs.mixture_inference
    import sv_pgs.genotype
    import sv_pgs.io
    import sv_pgs.linear_solvers
    import sv_pgs.aou_runner
    modules = [
        sv_pgs.config, sv_pgs.model, sv_pgs.preprocessing,
        sv_pgs.mixture_inference, sv_pgs.genotype, sv_pgs.io,
        sv_pgs.linear_solvers, sv_pgs.aou_runner,
    ]
    for mod in modules:
        source = inspect.getsource(mod)
        assert "maximum_active_variants" not in source, \
            f"{mod.__name__} contains 'maximum_active_variants' — delete it"


def test_no_production_module_contains_screening_cap_functions():
    """Ensure dead screening functions don't come back."""
    import sv_pgs.preprocessing
    source = inspect.getsource(sv_pgs.preprocessing)
    banned = [
        "_covariate_adjusted_marginal_scores",
        "_top_scoring_variant_indices",
        "_binary_screening_state",
        "_quantitative_screening_state",
    ]
    for name in banned:
        assert name not in source, \
            f"preprocessing contains '{name}' — this is a banned screening function, delete it"
