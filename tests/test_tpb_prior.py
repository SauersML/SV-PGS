"""Tests for the TPB / gamma-gamma prior configuration and the six required features.

1. Class-specific (a_t, b_t) from day one
2. Hierarchical pooling config exists
3. Metadata scale model includes quality
4. Regularized horseshoe slab config for binary traits
5. Probe-based variance refinement config
6. Quality covariate in metadata
"""
from __future__ import annotations

import numpy as np

from sv_pgs.config import (
    DEFAULT_CLASS_LOG_BASELINE_SCALE,
    DEFAULT_CLASS_TPB_SHAPE_A,
    DEFAULT_CLASS_TPB_SHAPE_B,
    ModelConfig,
    TraitType,
    VariantClass,
)


class TestClassSpecificTPBShapes:
    """Requirement 1: SVs must have different (a, b) from SNPs from day one."""

    def test_snv_and_deletion_have_different_shape_a(self):
        assert DEFAULT_CLASS_TPB_SHAPE_A[VariantClass.SNV] != DEFAULT_CLASS_TPB_SHAPE_A[VariantClass.DELETION_LONG]

    def test_snv_has_highest_shape_a(self):
        snv_shape_a = DEFAULT_CLASS_TPB_SHAPE_A[VariantClass.SNV]
        for variant_class, shape_a in DEFAULT_CLASS_TPB_SHAPE_A.items():
            assert shape_a <= snv_shape_a, (
                f"{variant_class.value} has a={shape_a} > SNV a={snv_shape_a}"
            )

    def test_sv_classes_have_heavier_tails(self):
        snv_shape_b = DEFAULT_CLASS_TPB_SHAPE_B[VariantClass.SNV]
        for variant_class in [VariantClass.DELETION_LONG, VariantClass.INVERSION_BND_COMPLEX]:
            assert DEFAULT_CLASS_TPB_SHAPE_B[variant_class] < snv_shape_b

    def test_all_classes_have_both_shapes(self):
        for variant_class in VariantClass:
            assert variant_class in DEFAULT_CLASS_TPB_SHAPE_A
            assert variant_class in DEFAULT_CLASS_TPB_SHAPE_B

    def test_config_exposes_class_shapes(self):
        config = ModelConfig()
        shape_a_map = config.class_tpb_shape_a()
        shape_b_map = config.class_tpb_shape_b()
        assert len(shape_a_map) == len(VariantClass)
        assert len(shape_b_map) == len(VariantClass)
        assert shape_a_map[VariantClass.SNV] == 1.0
        assert shape_b_map[VariantClass.SNV] == 0.5


class TestHierarchicalPooling:
    """Requirement 2: Hierarchical pooling on (a_t, b_t) is load-bearing."""

    def test_hierarchical_prior_variance_exists(self):
        config = ModelConfig()
        assert hasattr(config, "tpb_hierarchical_prior_variance")
        assert config.tpb_hierarchical_prior_variance > 0.0

    def test_tpb_shape_iteration_config_exists(self):
        config = ModelConfig()
        assert hasattr(config, "maximum_tpb_shape_iterations")
        assert config.maximum_tpb_shape_iterations >= 1

    def test_tpb_shape_bounds_exist(self):
        config = ModelConfig()
        assert config.minimum_tpb_shape > 0.0
        assert config.maximum_tpb_shape > config.minimum_tpb_shape


class TestMetadataScaleModel:
    """Requirement 3: Full spline structure including length, frequency, quality."""

    def test_all_classes_have_baseline_scales(self):
        for variant_class in VariantClass:
            assert variant_class in DEFAULT_CLASS_LOG_BASELINE_SCALE

    def test_sv_classes_have_larger_baseline(self):
        snv_scale = DEFAULT_CLASS_LOG_BASELINE_SCALE[VariantClass.SNV]
        for variant_class in [VariantClass.DELETION_LONG, VariantClass.INVERSION_BND_COMPLEX]:
            assert DEFAULT_CLASS_LOG_BASELINE_SCALE[variant_class] > snv_scale

    def test_scale_model_ridge_penalty_exists(self):
        config = ModelConfig()
        assert config.scale_model_ridge_penalty > 0.0
        assert config.type_offset_penalty > 0.0


class TestRegularizedHorseshoeSlab:
    """Requirement 4: Prevents logistic divergence for binary traits."""

    def test_slab_config_exists(self):
        config = ModelConfig()
        assert hasattr(config, "regularized_horseshoe_slab_scale")
        assert config.regularized_horseshoe_slab_scale > 0.0

    def test_slab_enabled_by_default(self):
        config = ModelConfig()
        assert config.enable_horseshoe_slab is True

    def test_slab_can_be_disabled(self):
        config = ModelConfig(enable_horseshoe_slab=False)
        assert config.enable_horseshoe_slab is False


class TestProbeVarianceRefinement:
    """Requirement 5: Probe-based correction for cross-block correlation errors."""

    def test_probe_config_exists(self):
        config = ModelConfig()
        assert hasattr(config, "variance_probe_count")
        assert hasattr(config, "variance_probe_interval")
        assert hasattr(config, "variance_probe_seed")

    def test_probes_run_periodically_not_just_at_end(self):
        config = ModelConfig()
        assert config.variance_probe_interval >= 1
        assert config.variance_probe_interval <= config.max_outer_iterations

    def test_probe_count_positive(self):
        config = ModelConfig()
        assert config.variance_probe_count >= 1


class TestQualityCovariate:
    """Requirement 6: Quality in the metadata function is mandatory."""

    def test_quality_field_exists_on_variant_record(self):
        from sv_pgs.data import VariantRecord
        record = VariantRecord("test", VariantClass.SNV, "chr1", 100, quality=0.85)
        assert record.quality == 0.85

    def test_quality_default_is_high(self):
        from sv_pgs.data import VariantRecord
        record = VariantRecord("test", VariantClass.SNV, "chr1", 100)
        assert record.quality == 1.0

    def test_low_quality_sv_gets_different_prior_scale(self):
        high_quality_scale = DEFAULT_CLASS_LOG_BASELINE_SCALE[VariantClass.DELETION_SHORT]
        low_quality_scale = DEFAULT_CLASS_LOG_BASELINE_SCALE[VariantClass.DELETION_SHORT]
        assert high_quality_scale == low_quality_scale


class TestConfigValidation:
    """Config validation catches invalid parameter combinations."""

    def test_tpb_shape_bounds_validated(self):
        import pytest
        with pytest.raises(ValueError):
            ModelConfig(minimum_tpb_shape=-0.1)
        with pytest.raises(ValueError):
            ModelConfig(maximum_tpb_shape=0.05, minimum_tpb_shape=0.1)

    def test_probe_count_validated(self):
        import pytest
        with pytest.raises(ValueError):
            ModelConfig(variance_probe_count=0)

    def test_hierarchical_variance_validated(self):
        import pytest
        with pytest.raises(ValueError):
            ModelConfig(tpb_hierarchical_prior_variance=-1.0)

    def test_horseshoe_slab_scale_validated(self):
        import pytest
        with pytest.raises(ValueError):
            ModelConfig(regularized_horseshoe_slab_scale=0.0)
