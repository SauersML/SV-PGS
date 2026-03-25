"""Tests for the TPB / gamma-gamma prior configuration and active solver features.

1. Class-specific (a_t, b_t) from day one
2. Hierarchical pooling config exists
3. Metadata scale model includes quality
4. Regularized horseshoe slab config for binary traits
5. Quality covariate in metadata
"""
from __future__ import annotations

import numpy as np

from sv_pgs.config import (
    DEFAULT_CLASS_LOG_BASELINE_SCALE,
    DEFAULT_CLASS_TPB_SHAPE_A,
    DEFAULT_CLASS_TPB_SHAPE_B,
    JaxDevicePreference,
    ModelConfig,
    TraitType,
    VariantClass,
)
from sv_pgs.data import VariantRecord
from sv_pgs.mixture_inference import (
    _build_prior_design,
    _metadata_baseline_scales_from_coefficients,
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


class TestQualityCovariate:
    """Requirement 5: Quality in the metadata function is mandatory."""

    def test_quality_field_exists_on_variant_record(self):
        from sv_pgs.data import VariantRecord
        record = VariantRecord("test", VariantClass.SNV, "chr1", 100, quality=0.85)
        assert record.quality == 0.85

    def test_quality_default_is_high(self):
        from sv_pgs.data import VariantRecord
        record = VariantRecord("test", VariantClass.SNV, "chr1", 100)
        assert record.quality == 1.0

    def test_low_quality_sv_gets_different_prior_scale(self):
        records = [
            VariantRecord("high_q", VariantClass.DELETION_SHORT, "chr1", 100, quality=0.95),
            VariantRecord("low_q", VariantClass.DELETION_SHORT, "chr1", 101, quality=0.50),
        ]
        prior_design = _build_prior_design(records)
        coefficients = np.zeros(len(prior_design.feature_names), dtype=np.float64)
        quality_feature_index = prior_design.feature_names.index("quality_linear")
        coefficients[quality_feature_index] = -0.75

        baseline_scales = _metadata_baseline_scales_from_coefficients(
            coefficients,
            prior_design.design_matrix,
            ModelConfig(),
        )

        assert baseline_scales[0] != baseline_scales[1]


class TestConfigValidation:
    """Config validation catches invalid parameter combinations."""

    def test_tpb_shape_bounds_validated(self):
        import pytest
        with pytest.raises(ValueError):
            ModelConfig(minimum_tpb_shape=-0.1)
        with pytest.raises(ValueError):
            ModelConfig(maximum_tpb_shape=0.05, minimum_tpb_shape=0.1)

    def test_scale_model_iteration_count_validated(self):
        import pytest
        with pytest.raises(ValueError):
            ModelConfig(maximum_scale_model_iterations=0)

    def test_tpb_shape_iteration_count_validated(self):
        import pytest
        with pytest.raises(ValueError):
            ModelConfig(maximum_tpb_shape_iterations=0)

    def test_hierarchical_variance_validated(self):
        import pytest
        with pytest.raises(ValueError):
            ModelConfig(tpb_hierarchical_prior_variance=-1.0)

    def test_tpb_shape_learning_rate_validated(self):
        import pytest
        with pytest.raises(ValueError):
            ModelConfig(tpb_shape_learning_rate=0.0)

    def test_horseshoe_slab_scale_validated(self):
        import pytest
        with pytest.raises(ValueError):
            ModelConfig(regularized_horseshoe_slab_scale=0.0)

    def test_binary_solver_config_validated(self):
        import pytest
        with pytest.raises(ValueError):
            ModelConfig(max_inner_newton_iterations=0)
        with pytest.raises(ValueError):
            ModelConfig(newton_gradient_tolerance=0.0)
        with pytest.raises(ValueError):
            ModelConfig(trust_region_initial_damping=0.0)
        with pytest.raises(ValueError):
            ModelConfig(trust_region_damping_increase_factor=1.0)
        with pytest.raises(ValueError):
            ModelConfig(trust_region_damping_decrease_factor=1.0)
        with pytest.raises(ValueError):
            ModelConfig(trust_region_success_threshold=1.0)
        with pytest.raises(ValueError):
            ModelConfig(trust_region_minimum_damping=0.0)

    def test_jax_device_index_validated(self):
        import pytest
        with pytest.raises(ValueError):
            ModelConfig(jax_device_index=-1)

    def test_jax_device_preference_enum_available(self):
        assert JaxDevicePreference.GPU.value == "gpu"
