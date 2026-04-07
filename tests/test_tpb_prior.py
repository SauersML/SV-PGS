"""Tests for the active TPB / gamma-gamma prior configuration and metadata contract.

1. Class-specific (a_t, b_t) from day one
2. Hierarchical pooling config exists
3. Metadata scale model only uses stable training-safe features
4. Solver config only exposes active operator knobs
"""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.config import (
    DEFAULT_CLASS_LOG_BASELINE_SCALE,
    DEFAULT_CLASS_TPB_SHAPE_A,
    DEFAULT_CLASS_TPB_SHAPE_B,
    ModelConfig,
    VariantClass,
)
from sv_pgs.data import VariantRecord
from sv_pgs.mixture_inference import (
    _build_prior_design,
    _design_matrix_for_feature_specs,
    _metadata_baseline_scales_from_coefficients,
    _parse_scale_model_feature_names,
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
    """Requirement 3: Metadata scale model only uses stable, training-safe features."""

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


    def test_quality_allele_frequency_and_support_enter_scale_design(self):
        records = [
            VariantRecord("sv_a", VariantClass.DELETION_SHORT, "chr1", 100, quality=0.95, allele_frequency=0.20),
            VariantRecord("sv_b", VariantClass.DELETION_SHORT, "chr1", 101, quality=0.50, allele_frequency=0.05, training_support=8),
            VariantRecord("sv_c", VariantClass.DELETION_SHORT, "chr1", 102, quality=0.25, allele_frequency=0.01, training_support=3),
        ]
        prior_design = _build_prior_design(records)
        assert "continuous_shared::quality" in prior_design.feature_names
        assert "continuous_linear::quality::deletion_short" in prior_design.feature_names
        assert "continuous_linear::logit_allele_frequency::deletion_short" in prior_design.feature_names
        assert "continuous_linear::log_training_support::deletion_short" in prior_design.feature_names

    def test_annotation_binary_features_and_frequency_bins_enter_scale_design(self):
        records = [
            VariantRecord(
                "sv_a",
                VariantClass.DELETION_SHORT,
                "chr1",
                100,
                allele_frequency=5e-4,
                prior_continuous_features={"coding_annotation": 1.0, "constraint_score": 0.2},
            ),
            VariantRecord(
                "sv_b",
                VariantClass.DELETION_SHORT,
                "chr1",
                101,
                allele_frequency=5e-3,
                prior_continuous_features={"coding_annotation": 0.0, "constraint_score": 0.7},
            ),
            VariantRecord(
                "sv_c",
                VariantClass.DELETION_SHORT,
                "chr1",
                102,
                allele_frequency=2e-2,
                prior_continuous_features={"coding_annotation": 1.0, "constraint_score": 0.9},
            ),
            VariantRecord(
                "sv_d",
                VariantClass.DELETION_SHORT,
                "chr1",
                103,
                allele_frequency=0.2,
                prior_continuous_features={"coding_annotation": 0.0, "constraint_score": 1.1},
            ),
        ]
        prior_design = _build_prior_design(records)

        assert "binary_indicator::coding_annotation" in prior_design.feature_names
        assert "binary_interaction::coding_annotation::deletion_short" in prior_design.feature_names
        assert "continuous_shared::constraint_score" in prior_design.feature_names
        assert "binary_indicator::maf_ultra_rare_indicator" in prior_design.feature_names
        assert "binary_interaction::maf_rare_indicator::deletion_short" in prior_design.feature_names
        assert "binary_interaction::maf_low_frequency_indicator::deletion_short" in prior_design.feature_names

        coefficients = np.zeros(len(prior_design.feature_names), dtype=np.float64)
        coefficients[prior_design.feature_names.index("binary_indicator::coding_annotation")] = 0.3
        coefficients[prior_design.feature_names.index("binary_interaction::coding_annotation::deletion_short")] = 0.5
        coefficients[prior_design.feature_names.index("continuous_shared::constraint_score")] = 0.4
        baseline_scales = _metadata_baseline_scales_from_coefficients(
            coefficients,
            prior_design.design_matrix,
            ModelConfig(),
        )
        assert np.unique(np.round(baseline_scales, 6)).shape[0] > 1

    def test_feature_name_parser_round_trips_new_annotation_features(self):
        records = [
            VariantRecord(
                "sv_a",
                VariantClass.DELETION_SHORT,
                "chr1",
                100,
                allele_frequency=5e-4,
                prior_continuous_features={"coding_annotation": 1.0, "constraint_score": 0.2},
            ),
            VariantRecord(
                "sv_b",
                VariantClass.DELETION_SHORT,
                "chr1",
                101,
                allele_frequency=5e-3,
                prior_continuous_features={"coding_annotation": 0.0, "constraint_score": 0.7},
            ),
            VariantRecord(
                "sv_c",
                VariantClass.DELETION_SHORT,
                "chr1",
                102,
                allele_frequency=2e-2,
                prior_continuous_features={"coding_annotation": 1.0, "constraint_score": 0.9},
            ),
        ]
        feature_specs = _parse_scale_model_feature_names(
            [
                "continuous_shared::constraint_score",
                "binary_indicator::coding_annotation",
                "binary_interaction::coding_annotation::deletion_short",
                "binary_interaction::maf_ultra_rare_indicator::deletion_short",
            ]
        )

        design_matrix = _design_matrix_for_feature_specs(records, feature_specs)

        assert design_matrix.shape == (3, 4)
        assert np.all(np.isfinite(design_matrix))

    def test_structural_annotations_still_affect_prior_scale(self):
        records = [
            VariantRecord("sv_a", VariantClass.DELETION_SHORT, "chr1", 100, length=100.0, is_copy_number=False),
            VariantRecord("sv_b", VariantClass.DELETION_SHORT, "chr1", 101, length=2_000.0, is_copy_number=True),
            VariantRecord("sv_c", VariantClass.DELETION_SHORT, "chr1", 102, length=6_000.0, is_repeat=True),
        ]
        prior_design = _build_prior_design(records)
        coefficients = np.zeros(len(prior_design.feature_names), dtype=np.float64)
        coefficients[prior_design.feature_names.index("continuous_linear::log_length::deletion_short")] = 0.5
        coefficients[prior_design.feature_names.index("copy_number_indicator")] = 0.5
        coefficients[prior_design.feature_names.index("repeat_indicator")] = -0.5
        baseline_scales = _metadata_baseline_scales_from_coefficients(coefficients, prior_design.design_matrix, ModelConfig())
        assert np.unique(np.round(baseline_scales, 6)).shape[0] > 1

    def test_custom_continuous_features_enter_scale_design_and_affect_prior_scale(self):
        records = [
            VariantRecord(
                "sv_a",
                VariantClass.DELETION_SHORT,
                "chr1",
                100,
                prior_continuous_features={"sv_length_score": 0.1},
            ),
            VariantRecord(
                "sv_b",
                VariantClass.DELETION_SHORT,
                "chr1",
                101,
                prior_continuous_features={"sv_length_score": 0.4},
            ),
            VariantRecord(
                "sv_c",
                VariantClass.DELETION_SHORT,
                "chr1",
                102,
                prior_continuous_features={"sv_length_score": 0.9},
            ),
        ]
        prior_design = _build_prior_design(records)
        coefficients = np.zeros(len(prior_design.feature_names), dtype=np.float64)
        coefficients[prior_design.feature_names.index("continuous_linear::sv_length_score::deletion_short")] = 0.5
        coefficients[prior_design.feature_names.index("continuous_quadratic::sv_length_score::deletion_short")] = -0.25
        baseline_scales = _metadata_baseline_scales_from_coefficients(coefficients, prior_design.design_matrix, ModelConfig())
        assert "continuous_linear::sv_length_score::deletion_short" in prior_design.feature_names
        assert "continuous_quadratic::sv_length_score::deletion_short" in prior_design.feature_names
        assert np.unique(np.round(baseline_scales, 6)).shape[0] > 1

    def test_custom_continuous_features_cannot_override_reserved_names(self):
        with pytest.raises(ValueError, match="built-in features: log_length"):
            VariantRecord(
                "sv_a",
                VariantClass.DELETION_SHORT,
                "chr1",
                100,
                prior_continuous_features={"log_length": 1.0},
            )

    def test_custom_continuous_features_cannot_contain_feature_delimiter(self):
        with pytest.raises(ValueError, match="cannot contain '::'"):
            VariantRecord(
                "sv_a",
                VariantClass.DELETION_SHORT,
                "chr1",
                100,
                prior_continuous_features={"bad::name": 1.0},
            )


class TestConfigValidation:
    """Config validation catches invalid parameter combinations."""

    def test_tpb_shape_bounds_validated(self):
        with pytest.raises(ValueError):
            ModelConfig(minimum_tpb_shape=-0.1)
        with pytest.raises(ValueError):
            ModelConfig(maximum_tpb_shape=0.05, minimum_tpb_shape=0.1)

    def test_scale_model_iteration_count_validated(self):
        with pytest.raises(ValueError):
            ModelConfig(maximum_scale_model_iterations=0)

    def test_tpb_shape_iteration_count_validated(self):
        with pytest.raises(ValueError):
            ModelConfig(maximum_tpb_shape_iterations=0)

    def test_hierarchical_variance_validated(self):
        with pytest.raises(ValueError):
            ModelConfig(tpb_hierarchical_prior_variance=-1.0)

    def test_tpb_shape_learning_rate_validated(self):
        with pytest.raises(ValueError):
            ModelConfig(tpb_shape_learning_rate=0.0)

    def test_active_solver_config_validated(self):
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

    def test_linear_solver_config_validated(self):
        with pytest.raises(ValueError):
            ModelConfig(linear_solver_tolerance=0.0)
        with pytest.raises(ValueError):
            ModelConfig(maximum_linear_solver_iterations=0)
        with pytest.raises(ValueError):
            ModelConfig(logdet_probe_count=0)
        with pytest.raises(ValueError):
            ModelConfig(logdet_lanczos_steps=1)
        with pytest.raises(ValueError):
            ModelConfig(exact_solver_matrix_limit=0)
        with pytest.raises(ValueError):
            ModelConfig(posterior_variance_batch_size=0)
        with pytest.raises(ValueError):
            ModelConfig(validation_interval=0)
