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
    _class_membership_by_class,
    _design_matrix_for_feature_specs,
    _metadata_baseline_scales_from_coefficients,
    _prior_annotation_tables,
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


    def test_vcf_fields_do_not_enter_scale_design_as_annotations(self):
        records = [
            VariantRecord("sv_a", VariantClass.DELETION_SHORT, "chr1", 100, quality=0.95, allele_frequency=0.20),
            VariantRecord("sv_b", VariantClass.DELETION_SHORT, "chr1", 101, quality=0.50, allele_frequency=0.05, training_support=8),
            VariantRecord("sv_c", VariantClass.DELETION_SHORT, "chr1", 102, quality=0.25, allele_frequency=0.01, training_support=3),
        ]
        prior_design = _build_prior_design(records)
        assert all("quality" not in feature_name for feature_name in prior_design.feature_names)
        assert all("allele_frequency" not in feature_name for feature_name in prior_design.feature_names)
        assert all("training_support" not in feature_name for feature_name in prior_design.feature_names)

    def test_annotation_binary_features_enter_scale_design(self):
        records = [
            VariantRecord(
                "sv_a",
                VariantClass.DELETION_SHORT,
                "chr1",
                100,
                allele_frequency=5e-4,
                prior_binary_features={"coding_annotation": True},
                prior_continuous_features={"constraint_score": 0.2},
            ),
            VariantRecord(
                "sv_b",
                VariantClass.DELETION_SHORT,
                "chr1",
                101,
                allele_frequency=5e-3,
                prior_binary_features={"coding_annotation": False},
                prior_continuous_features={"constraint_score": 0.7},
            ),
            VariantRecord(
                "sv_c",
                VariantClass.DELETION_SHORT,
                "chr1",
                102,
                allele_frequency=2e-2,
                prior_binary_features={"coding_annotation": True},
                prior_continuous_features={"constraint_score": 0.9},
            ),
            VariantRecord(
                "sv_d",
                VariantClass.DELETION_SHORT,
                "chr1",
                103,
                allele_frequency=0.2,
                prior_binary_features={"coding_annotation": False},
                prior_continuous_features={"constraint_score": 1.1},
            ),
        ]
        prior_design = _build_prior_design(records)

        assert "factor_level::coding_annotation::true" in prior_design.feature_names
        assert not any("factor_interaction::" in name for name in prior_design.feature_names)
        assert "continuous_spline::constraint_score::basis_0" in prior_design.feature_names

        coefficients = np.zeros(len(prior_design.feature_names), dtype=np.float64)
        coefficients[prior_design.feature_names.index("factor_level::coding_annotation::true")] = 0.3
        coefficients[prior_design.feature_names.index("continuous_spline::constraint_score::basis_0")] = 0.4
        baseline_scales = _metadata_baseline_scales_from_coefficients(
            coefficients,
            prior_design.design_matrix,
            ModelConfig(),
        )
        assert np.unique(np.round(baseline_scales, 6)).shape[0] > 1

    def test_feature_specs_round_trip_design_matrix(self):
        records = [
            VariantRecord(
                "sv_a",
                VariantClass.DELETION_SHORT,
                "chr1",
                100,
                allele_frequency=5e-4,
                prior_binary_features={"coding_annotation": True},
                prior_continuous_features={"constraint_score": 0.2},
                prior_categorical_features={"functional_state": "lof"},
                prior_membership_features={"regulatory_mix": {"enhancer": 0.75, "promoter": 0.25}},
                prior_nested_features={"gene_context": ("protein_coding", "exon")},
            ),
            VariantRecord(
                "sv_b",
                VariantClass.DELETION_SHORT,
                "chr1",
                101,
                allele_frequency=5e-3,
                prior_binary_features={"coding_annotation": False},
                prior_continuous_features={"constraint_score": 0.7},
                prior_categorical_features={"functional_state": "missense"},
                prior_membership_features={"regulatory_mix": {"enhancer": 0.10, "promoter": 0.90}},
                prior_nested_features={"gene_context": ("protein_coding", "intron")},
            ),
            VariantRecord(
                "sv_c",
                VariantClass.DELETION_SHORT,
                "chr1",
                102,
                allele_frequency=2e-2,
                prior_binary_features={"coding_annotation": True},
                prior_continuous_features={"constraint_score": 0.9},
                prior_categorical_features={"functional_state": "lof"},
                prior_membership_features={"regulatory_mix": {"enhancer": 0.40, "promoter": 0.60}},
                prior_nested_features={"gene_context": ("lncRNA", "exon")},
            ),
        ]
        prior_design = _build_prior_design(records)
        design_matrix = _design_matrix_for_feature_specs(
            records=records,
            feature_specs=prior_design.feature_specs,
            annotation_tables=_prior_annotation_tables(records),
            class_membership_by_class=_class_membership_by_class(
                records,
                {
                    feature_spec.variant_class
                    for feature_spec in prior_design.feature_specs
                    if feature_spec.variant_class is not None
                },
            ),
        )

        np.testing.assert_allclose(design_matrix, prior_design.design_matrix)

    def test_vcf_structural_fields_do_not_enter_scale_design_as_annotations(self):
        records = [
            VariantRecord("sv_a", VariantClass.DELETION_SHORT, "chr1", 100, length=100.0, is_copy_number=False),
            VariantRecord("sv_b", VariantClass.DELETION_SHORT, "chr1", 101, length=2_000.0, is_copy_number=True),
            VariantRecord("sv_c", VariantClass.DELETION_SHORT, "chr1", 102, length=6_000.0, is_repeat=True),
        ]
        prior_design = _build_prior_design(records)
        assert all("log_length" not in feature_name for feature_name in prior_design.feature_names)
        assert all("copy_number" not in feature_name for feature_name in prior_design.feature_names)
        assert all("repeat" not in feature_name for feature_name in prior_design.feature_names)

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
        coefficients[prior_design.feature_names.index("continuous_spline::sv_length_score::basis_0")] = 0.5
        coefficients[prior_design.feature_names.index("continuous_spline::sv_length_score::basis_1")] = -0.25
        baseline_scales = _metadata_baseline_scales_from_coefficients(coefficients, prior_design.design_matrix, ModelConfig())
        assert "continuous_spline::sv_length_score::basis_0" in prior_design.feature_names
        assert "continuous_spline::sv_length_score::basis_1" in prior_design.feature_names
        assert np.unique(np.round(baseline_scales, 6)).shape[0] > 1

    def test_schema_driven_annotations_compile_categorical_nested_and_membership_terms(self):
        nested_paths = [
            ("protein_coding", "exon"),
            ("protein_coding", "intron"),
            ("lncRNA", "enhancer"),
            ("lncRNA", "promoter"),
        ]
        records = []
        for record_index, (nested_path, functional_state, promoter_weight) in enumerate(
            (
                (nested_path, functional_state, promoter_weight)
                for nested_path in nested_paths
                for functional_state in ("lof", "missense")
                for promoter_weight in (0.1, 0.35, 0.65, 0.9)
            )
        ):
            records.append(
                VariantRecord(
                    f"sv_{record_index}",
                    VariantClass.DELETION_SHORT,
                    "chr1",
                    100 + record_index,
                    prior_categorical_features={"functional_state": functional_state},
                    prior_membership_features={
                        "regulatory_mix": {
                            "enhancer": 1.0 - promoter_weight,
                            "promoter": promoter_weight,
                        }
                    },
                    prior_nested_features={"gene_context": nested_path},
                )
            )
        prior_design = _build_prior_design(records)

        assert "factor_level::functional_state::missense" in prior_design.feature_names
        assert "factor_level::regulatory_mix::promoter" in prior_design.feature_names
        assert "nested_level::gene_context::0::lncRNA" not in prior_design.feature_names
        assert "nested_level::gene_context::0::protein_coding" in prior_design.feature_names
        assert "nested_level::gene_context::1::protein_coding>intron" in prior_design.feature_names

        coefficients = np.zeros(len(prior_design.feature_names), dtype=np.float64)
        coefficients[prior_design.feature_names.index("factor_level::functional_state::missense")] = 0.4
        coefficients[prior_design.feature_names.index("factor_level::regulatory_mix::promoter")] = 0.5
        coefficients[prior_design.feature_names.index("nested_level::gene_context::1::protein_coding>intron")] = 0.6
        baseline_scales = _metadata_baseline_scales_from_coefficients(coefficients, prior_design.design_matrix, ModelConfig())

        assert np.unique(np.round(baseline_scales, 6)).shape[0] > 1

    def test_prior_design_is_full_rank_centered_and_unit_rms(self):
        random_generator = np.random.default_rng(8)
        variant_classes = [VariantClass.SNV, VariantClass.DELETION_SHORT, VariantClass.DUPLICATION_SHORT]
        functional_states = ["lof", "missense", "neutral"]
        nested_paths = [
            ("coding", "exon"),
            ("coding", "intron"),
            ("noncoding", "enhancer"),
            ("noncoding", "promoter"),
        ]
        records = [
            VariantRecord(
                f"variant_{record_index}",
                variant_classes[int(random_generator.integers(len(variant_classes)))],
                "chr1",
                record_index,
                prior_categorical_features={
                    "functional_state": functional_states[int(random_generator.integers(len(functional_states)))]
                },
                prior_nested_features={
                    "gene_context": nested_paths[int(random_generator.integers(len(nested_paths)))]
                },
                prior_continuous_features={"constraint_score": float(random_generator.normal())},
            )
            for record_index in range(160)
        ]

        design_matrix = _build_prior_design(records).design_matrix
        singular_values = np.linalg.svd(design_matrix, compute_uv=False)

        assert np.linalg.matrix_rank(design_matrix, tol=1e-10) == design_matrix.shape[1]
        np.testing.assert_allclose(np.mean(design_matrix, axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.sqrt(np.mean(design_matrix * design_matrix, axis=0)), 1.0, atol=1e-12)
        assert singular_values[0] / singular_values[-1] < 500.0

    def test_reference_coding_preserves_saturated_class_varying_factor_predictors(self):
        variant_classes = [VariantClass.SNV, VariantClass.DELETION_SHORT, VariantClass.DUPLICATION_SHORT]
        records = [
            VariantRecord(
                f"variant_{record_index}",
                variant_classes[record_index % len(variant_classes)],
                "chr1",
                record_index,
                prior_binary_features={"coding": (record_index * 5 + record_index // 3) % 7 < 3},
            )
            for record_index in range(84)
        ]
        prior_design = _build_prior_design(records)
        class_membership = np.column_stack(
            [
                np.asarray([record.variant_class == variant_class for record in records], dtype=np.float64)
                for variant_class in variant_classes
            ]
        )
        coding = np.asarray([record.prior_binary_features["coding"] for record in records], dtype=np.float64)

        def centered(values: np.ndarray) -> np.ndarray:
            return values - np.mean(values)

        redundant_design = np.column_stack(
            [
                np.ones(len(records), dtype=np.float64),
                *(centered(class_membership[:, class_index]) for class_index in range(len(variant_classes))),
                centered(coding),
                *(centered(coding * class_membership[:, class_index]) for class_index in range(len(variant_classes))),
            ]
        )
        reference_coded_design = np.column_stack(
            [np.ones(len(records), dtype=np.float64), prior_design.design_matrix]
        )
        redundant_coefficients = np.linspace(-0.7, 0.9, redundant_design.shape[1])
        redundant_predictor = redundant_design @ redundant_coefficients
        reference_coefficients, *_ = np.linalg.lstsq(reference_coded_design, redundant_predictor, rcond=None)

        np.testing.assert_allclose(reference_coded_design @ reference_coefficients, redundant_predictor, atol=1e-12)

    def test_rank_screen_prefers_first_main_effect_over_duplicate_annotation(self):
        records = [
            VariantRecord(
                f"variant_{record_index}",
                VariantClass.SNV,
                "chr1",
                record_index,
                prior_binary_features={
                    "canonical_annotation": record_index % 2 == 0,
                    "duplicate_annotation": record_index % 2 == 0,
                },
            )
            for record_index in range(12)
        ]

        prior_design = _build_prior_design(records)

        assert prior_design.feature_names == ["factor_level::canonical_annotation::true"]
        assert np.linalg.matrix_rank(prior_design.design_matrix) == 1

    def test_compiled_scaling_is_reused_for_a_shifted_member_distribution(self):
        training_records = [
            VariantRecord("deletion", VariantClass.DELETION_SHORT, "chr1", 1),
            VariantRecord("snv_1", VariantClass.SNV, "chr1", 2),
            VariantRecord("snv_2", VariantClass.SNV, "chr1", 3),
            VariantRecord("snv_3", VariantClass.SNV, "chr1", 4),
        ]
        prior_design = _build_prior_design(training_records)
        member_records = [
            VariantRecord("member_deletion", VariantClass.DELETION_SHORT, "chr1", 5),
            VariantRecord("member_snv", VariantClass.SNV, "chr1", 6),
        ]

        member_design = _design_matrix_for_feature_specs(
            records=member_records,
            feature_specs=prior_design.feature_specs,
            annotation_tables=_prior_annotation_tables(member_records),
            class_membership_by_class=_class_membership_by_class(
                member_records,
                {
                    feature_spec.variant_class
                    for feature_spec in prior_design.feature_specs
                    if feature_spec.variant_class is not None
                },
            ),
        )

        assert prior_design.feature_names == ["type_offset::deletion_short"]
        training_rms = np.sqrt(3.0) / 4.0
        np.testing.assert_allclose(
            member_design[:, 0],
            np.asarray([(1.0 - 0.25) / training_rms, (0.0 - 0.25) / training_rms]),
        )
        assert not np.isclose(np.mean(member_design[:, 0]), 0.0)

    def test_user_annotation_names_are_not_reserved_by_old_built_ins(self):
        record = VariantRecord(
            "sv_a",
            VariantClass.DELETION_SHORT,
            "chr1",
            100,
            prior_continuous_features={"log_length": 1.0},
        )
        assert record.prior_continuous_features == {"log_length": 1.0}

    def test_custom_continuous_features_cannot_contain_feature_delimiter(self):
        with pytest.raises(ValueError, match="cannot contain '::'"):
            VariantRecord(
                "sv_a",
                VariantClass.DELETION_SHORT,
                "chr1",
                100,
                prior_continuous_features={"bad::name": 1.0},
            )

    def test_custom_prior_feature_names_must_be_unique_across_annotation_families(self):
        with pytest.raises(ValueError, match="unique across annotation families"):
            VariantRecord(
                "sv_a",
                VariantClass.DELETION_SHORT,
                "chr1",
                100,
                prior_binary_features={"shared_name": True},
                prior_categorical_features={"shared_name": "lof"},
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

    def test_active_solver_config_validated(self):
        with pytest.raises(ValueError):
            ModelConfig(max_inner_newton_iterations=0)
        with pytest.raises(ValueError):
            ModelConfig(binary_inner_tolerance=0.0)

    def test_binary_inner_tolerance_default_matches_outer_tolerance(self):
        config = ModelConfig()
        assert config.binary_inner_tolerance == 1e-4
        assert config.binary_inner_tolerance == config.convergence_tolerance

    def test_removed_newton_gradient_tolerance_is_not_accepted(self):
        with pytest.raises(TypeError, match="newton_gradient_tolerance"):
            ModelConfig(**{"newton_gradient_tolerance": 1e-6})

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
