import numpy as np
import pytest

from sv_pgs.config import ModelConfig, VariantClass
from sv_pgs.data import VariantRecord
from sv_pgs.genotype import as_raw_genotype_matrix
from sv_pgs.plink import PLINK_MISSING_INT8
import sv_pgs.preprocessing as preprocessing_module
from sv_pgs.preprocessing import (
    Preprocessor,
    build_tie_map,
    collapse_tie_groups,
    compute_variant_statistics,
    fit_preprocessor,
    select_active_variant_indices,
)


def test_fold_preprocessing_and_exact_ties_ignore_variant_class():
    genotype_matrix = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, -1.0, np.nan],
            [2.0, 2.0, -2.0, 0.0],
            [np.nan, np.nan, np.nan, 1.0],
        ],
        dtype=np.float32,
    )
    covariate_matrix = np.zeros((4, 1), dtype=np.float32)
    target_vector = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    variant_records = [
        VariantRecord("variant_0", VariantClass.SNV, "1", 100, length=1.0, allele_frequency=0.10, quality=1.0),
        VariantRecord("variant_1", VariantClass.DELETION_SHORT, "1", 101, length=500.0, allele_frequency=0.02, quality=0.8),
        VariantRecord("variant_2", VariantClass.DUPLICATION_SHORT, "1", 102, length=800.0, allele_frequency=0.01, quality=0.7),
        VariantRecord("variant_3", VariantClass.SNV, "1", 103, length=1.0, allele_frequency=0.10, quality=0.80),
    ]

    prepared_arrays = fit_preprocessor(genotype_matrix, covariate_matrix, target_vector, ModelConfig())
    standardized_genotypes = Preprocessor(means=prepared_arrays.means, scales=prepared_arrays.scales).transform(genotype_matrix)
    standardized_dense = np.asarray(standardized_genotypes, dtype=np.float32)
    tie_map = build_tie_map(standardized_dense, variant_records, ModelConfig())

    assert standardized_genotypes.shape == genotype_matrix.shape
    np.testing.assert_allclose(standardized_dense.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.mean(standardized_dense**2, axis=0), 1.0, atol=1e-5)
    assert tie_map.kept_indices.tolist() == [0, 3]
    assert tie_map.original_to_reduced.tolist() == [0, 0, 0, 1]
    np.testing.assert_allclose(tie_map.reduced_to_group[0].signs, [1.0, 1.0, -1.0])


def test_mixed_class_tie_group_uses_symmetric_latent_class():
    genotype_matrix = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, -1.0],
        ],
        dtype=np.float32,
    )
    variant_records = [
        VariantRecord("variant_0", VariantClass.SNV, "1", 100, length=1.0, allele_frequency=0.10, quality=1.0),
        VariantRecord("variant_1", VariantClass.DELETION_SHORT, "1", 101, length=500.0, allele_frequency=0.02, quality=0.8),
        VariantRecord("variant_2", VariantClass.SNV, "1", 200, length=1.0, allele_frequency=0.10, quality=0.9),
    ]

    tie_map = build_tie_map(genotype_matrix, variant_records, ModelConfig())
    collapsed_records = collapse_tie_groups(variant_records, tie_map)

    assert tie_map.kept_indices.tolist() == [0, 2]
    assert collapsed_records[0].variant_class == VariantClass.OTHER_COMPLEX_SV
    assert collapsed_records[0].position == 100
    assert collapsed_records[0].is_repeat is False
    assert collapsed_records[0].prior_class_members == (
        VariantClass.DELETION_SHORT,
        VariantClass.SNV,
    )
    np.testing.assert_allclose(collapsed_records[0].prior_class_membership, [0.5, 0.5])


def test_exact_ties_require_exact_float32_equality():
    base = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    almost_equal = base.copy()
    almost_equal[1] = np.nextafter(np.float32(1.0), np.float32(2.0))
    genotype_matrix = np.column_stack([base, almost_equal]).astype(np.float32)
    variant_records = [
        VariantRecord("variant_0", VariantClass.SNV, "1", 100),
        VariantRecord("variant_1", VariantClass.SNV, "1", 101),
    ]

    tie_map = build_tie_map(genotype_matrix, variant_records, ModelConfig())

    assert tie_map.kept_indices.tolist() == [0, 1]
    assert tie_map.original_to_reduced.tolist() == [0, 1]


def test_tie_map_uses_missingness_aware_signatures():
    genotype_matrix = np.array(
        [
            [0.0, 0.0],
            [np.nan, 0.0],
            [0.0, np.nan],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    variant_records = [
        VariantRecord("variant_0", VariantClass.SNV, "1", 100),
        VariantRecord("variant_1", VariantClass.SNV, "1", 101),
    ]

    tie_map = build_tie_map(genotype_matrix, variant_records, ModelConfig())

    assert tie_map.kept_indices.tolist() == [0, 1]
    assert tie_map.original_to_reduced.tolist() == [0, 1]


def test_tie_map_groups_binary_columns_with_same_standardized_pattern():
    raw_genotype_matrix = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    covariate_matrix = np.zeros((raw_genotype_matrix.shape[0], 1), dtype=np.float32)
    target_vector = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    variant_records = [
        VariantRecord("variant_0", VariantClass.SNV, "1", 100),
        VariantRecord("variant_1", VariantClass.SNV, "1", 101),
        VariantRecord("variant_2", VariantClass.SNV, "1", 102),
    ]
    prepared_arrays = fit_preprocessor(raw_genotype_matrix, covariate_matrix, target_vector, ModelConfig())
    standardized_genotypes = as_raw_genotype_matrix(raw_genotype_matrix).standardized(
        prepared_arrays.means,
        prepared_arrays.scales,
    )

    tie_map = build_tie_map(standardized_genotypes, variant_records, ModelConfig())

    assert tie_map.kept_indices.tolist() == [0]
    assert tie_map.original_to_reduced.tolist() == [0, 0, 0]
    np.testing.assert_allclose(tie_map.reduced_to_group[0].signs, [1.0, 1.0, -1.0])


def test_tie_map_groups_monomorphic_columns_regardless_of_raw_level():
    raw_genotype_matrix = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    covariate_matrix = np.zeros((raw_genotype_matrix.shape[0], 1), dtype=np.float32)
    target_vector = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    variant_records = [
        VariantRecord("variant_0", VariantClass.SNV, "1", 100),
        VariantRecord("variant_1", VariantClass.SNV, "1", 101),
        VariantRecord("variant_2", VariantClass.SNV, "1", 102),
    ]
    prepared_arrays = fit_preprocessor(raw_genotype_matrix, covariate_matrix, target_vector, ModelConfig())
    standardized_genotypes = as_raw_genotype_matrix(raw_genotype_matrix).standardized(
        prepared_arrays.means,
        prepared_arrays.scales,
    )

    tie_map = build_tie_map(standardized_genotypes, variant_records, ModelConfig())

    assert tie_map.kept_indices.tolist() == [0]
    assert tie_map.original_to_reduced.tolist() == [0, 0, 0]


def test_tie_map_groups_monomorphic_raw_columns_even_with_different_missingness():
    raw_genotype_matrix = np.array(
        [
            [0, 2],
            [PLINK_MISSING_INT8, 2],
            [0, 2],
            [0, PLINK_MISSING_INT8],
        ],
        dtype=np.int8,
    )
    covariate_matrix = np.zeros((raw_genotype_matrix.shape[0], 1), dtype=np.float32)
    target_vector = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    variant_records = [
        VariantRecord("variant_0", VariantClass.SNV, "1", 100),
        VariantRecord("variant_1", VariantClass.SNV, "1", 101),
    ]
    raw_genotypes = as_raw_genotype_matrix(raw_genotype_matrix)
    prepared_arrays = fit_preprocessor(raw_genotypes, covariate_matrix, target_vector, ModelConfig())
    standardized_genotypes = raw_genotypes.standardized(
        prepared_arrays.means,
        prepared_arrays.scales,
    )

    tie_map = build_tie_map(standardized_genotypes, variant_records, ModelConfig())

    assert tie_map.kept_indices.tolist() == [0]
    assert tie_map.original_to_reduced.tolist() == [0, 0]


def test_tie_map_groups_int8_sign_flipped_hardcalls():
    raw_genotype_matrix = np.array(
        [
            [0, 1, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
        ],
        dtype=np.int8,
    )
    covariate_matrix = np.zeros((raw_genotype_matrix.shape[0], 1), dtype=np.float32)
    target_vector = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    variant_records = [
        VariantRecord("variant_0", VariantClass.SNV, "1", 100),
        VariantRecord("variant_1", VariantClass.SNV, "1", 101),
        VariantRecord("variant_2", VariantClass.SNV, "1", 102),
    ]
    raw_genotypes = as_raw_genotype_matrix(raw_genotype_matrix)
    prepared_arrays = fit_preprocessor(raw_genotypes, covariate_matrix, target_vector, ModelConfig())
    standardized_genotypes = raw_genotypes.standardized(
        prepared_arrays.means,
        prepared_arrays.scales,
    )

    tie_map = build_tie_map(standardized_genotypes, variant_records, ModelConfig())

    assert tie_map.kept_indices.tolist() == [0, 2]
    assert tie_map.original_to_reduced.tolist() == [0, 0, 1]
    np.testing.assert_allclose(tie_map.reduced_to_group[0].signs, [1.0, -1.0])


def test_hardcall_batch_canonicalization_matches_expected_state_mappings():
    batch_values = np.array(
        [
            [PLINK_MISSING_INT8, 0, 1, 2, 0, 0, 1, 0],
            [PLINK_MISSING_INT8, 0, 1, 2, 1, 2, 2, 1],
            [PLINK_MISSING_INT8, PLINK_MISSING_INT8, PLINK_MISSING_INT8, PLINK_MISSING_INT8, PLINK_MISSING_INT8, PLINK_MISSING_INT8, PLINK_MISSING_INT8, 2],
            [PLINK_MISSING_INT8, 0, 1, 2, 0, 0, 1, PLINK_MISSING_INT8],
        ],
        dtype=np.int8,
    )

    state_masks = preprocessing_module._hardcall_state_masks(batch_values)
    canonical, sign_flipped = preprocessing_module._canonicalize_hardcall_tie_columns_i8(
        batch_values,
        state_masks,
    )

    np.testing.assert_array_equal(state_masks, np.array([0, 1, 2, 4, 3, 5, 6, 7], dtype=np.uint8))
    np.testing.assert_array_equal(
        canonical,
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 3, 3, 3, 2],
                [0, 0, 0, 0, 0, 0, 0, 3],
            ],
            dtype=np.int8,
        ),
    )
    np.testing.assert_array_equal(
        sign_flipped,
        np.array(
            [
                [0, 0, 0, 0, 1, 1, 1, 2],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 3, 3, 3, 0],
                [0, 0, 0, 0, 1, 1, 1, 3],
            ],
            dtype=np.int8,
        ),
    )


def test_tie_map_handles_dense_float_columns_with_more_than_127_distinct_values():
    sample_count = 256
    dense_genotypes = np.column_stack(
        [
            np.linspace(-3.0, 3.0, sample_count, dtype=np.float32),
            np.linspace(-3.0, 3.0, sample_count, dtype=np.float32),
            np.linspace(3.0, -3.0, sample_count, dtype=np.float32),
        ]
    )
    variant_records = [
        VariantRecord("variant_0", VariantClass.SNV, "1", 100),
        VariantRecord("variant_1", VariantClass.SNV, "1", 101),
        VariantRecord("variant_2", VariantClass.SNV, "1", 102),
    ]

    tie_map = build_tie_map(dense_genotypes, variant_records, ModelConfig())

    assert tie_map.kept_indices.tolist() == [0]
    assert tie_map.original_to_reduced.tolist() == [0, 0, 0]
    np.testing.assert_allclose(tie_map.reduced_to_group[0].signs, [1.0, 1.0, -1.0])


def test_tie_map_verifies_hash_collisions_before_grouping(monkeypatch: pytest.MonkeyPatch):
    genotype_matrix = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    variant_records = [
        VariantRecord("variant_0", VariantClass.SNV, "1", 100),
        VariantRecord("variant_1", VariantClass.SNV, "1", 101),
    ]

    monkeypatch.setattr(
        preprocessing_module,
        "_hashed_tie_signature",
        lambda standardized_column, missing_mask: (b"forced", b"forced"),
    )

    tie_map = build_tie_map(genotype_matrix, variant_records, ModelConfig())

    assert tie_map.kept_indices.tolist() == [0, 1]
    assert tie_map.original_to_reduced.tolist() == [0, 1]


def test_select_active_variant_indices_filters_low_maf_variants_regardless_of_class():
    variant_records = [
        VariantRecord("snv_drop", VariantClass.SNV, "1", 100, allele_frequency=0.0002),
        VariantRecord("sv_keep", VariantClass.DELETION_SHORT, "1", 101, allele_frequency=0.0020),
        VariantRecord("snv_keep", VariantClass.SNV, "1", 102, allele_frequency=0.0200),
        VariantRecord("common_alt_keep", VariantClass.SNV, "1", 103, allele_frequency=0.9990),
    ]

    result = select_active_variant_indices(
        variant_records=variant_records,
        config=ModelConfig(minimum_minor_allele_frequency=0.001),
    )
    assert result.tolist() == [1, 2, 3]


def test_select_active_variant_indices_keeps_all_variants_when_maf_filter_is_disabled():
    variant_records = [
        VariantRecord("snv_0", VariantClass.SNV, "1", 100, allele_frequency=0.0001),
        VariantRecord("sv_1", VariantClass.DELETION_SHORT, "1", 101, allele_frequency=0.0002),
        VariantRecord("snv_2", VariantClass.SNV, "1", 102, allele_frequency=0.25),
    ]

    result = select_active_variant_indices(
        variant_records=variant_records,
        config=ModelConfig(minimum_minor_allele_frequency=0.0),
    )

    assert result.tolist() == [0, 1, 2]


def test_select_active_variant_indices_uses_only_maf_filter():
    variant_records = [
        VariantRecord("rare_drop", VariantClass.SNV, "1", 100, allele_frequency=0.0005),
        VariantRecord("structural_keep", VariantClass.DELETION_SHORT, "1", 101, allele_frequency=0.0020),
        VariantRecord("snv_keep", VariantClass.SNV, "1", 102, allele_frequency=0.0100),
        VariantRecord("common_ref_keep", VariantClass.SNV, "1", 103, allele_frequency=0.4000),
        VariantRecord("common_alt_keep", VariantClass.SNV, "1", 104, allele_frequency=0.9990),
    ]

    result = select_active_variant_indices(
        variant_records=variant_records,
        config=ModelConfig(minimum_minor_allele_frequency=0.001),
    )

    assert result.tolist() == [1, 2, 3, 4]


def test_select_active_variant_indices_keeps_all_post_maf_variants():
    variant_records = [
        VariantRecord("snv_signal", VariantClass.SNV, "1", 100),
        VariantRecord("sv_keep", VariantClass.DELETION_SHORT, "1", 101, training_support=1),
        VariantRecord("snv_noise_0", VariantClass.SNV, "1", 102),
        VariantRecord("snv_noise_1", VariantClass.SNV, "1", 103),
        VariantRecord("snv_noise_2", VariantClass.SNV, "1", 104),
    ]

    result = select_active_variant_indices(
        variant_records=variant_records,
        config=ModelConfig(
            minimum_minor_allele_frequency=0.0,
        ),
    )

    assert result.tolist() == [0, 1, 2, 3, 4]


def test_select_active_variant_indices_keeps_structural_and_snv_variants_after_maf_filter():
    variant_records = [
        VariantRecord("snv_signal", VariantClass.SNV, "1", 100, allele_frequency=0.1),
        VariantRecord("structural_keep", VariantClass.DELETION_SHORT, "1", 101, allele_frequency=0.1),
        VariantRecord("snv_noise", VariantClass.SNV, "1", 102, allele_frequency=0.1),
    ]

    result = select_active_variant_indices(
        variant_records=variant_records,
        config=ModelConfig(
            minimum_minor_allele_frequency=0.0,
        ),
    )

    assert result.tolist() == [0, 1, 2]


def test_fit_preprocessor_matches_streaming_variant_statistics_with_missing_values():
    genotype_matrix = np.array(
        [
            [0.0, 1.0, np.nan],
            [1.0, np.nan, 2.0],
            [2.0, 0.0, 1.0],
            [np.nan, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    covariate_matrix = np.zeros((4, 1), dtype=np.float32)
    target_vector = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    config = ModelConfig()

    prepared_arrays = fit_preprocessor(genotype_matrix, covariate_matrix, target_vector, config)
    variant_statistics = compute_variant_statistics(
        raw_genotypes=as_raw_genotype_matrix(genotype_matrix),
        config=config,
    )

    np.testing.assert_allclose(prepared_arrays.means, variant_statistics.means)
    np.testing.assert_allclose(prepared_arrays.scales, variant_statistics.scales)


def test_compute_variant_statistics_uses_neutral_af_for_non_dosage_float_columns():
    genotype_matrix = np.array(
        [
            [-1.0, 1.0],
            [0.0, 0.0],
            [1.0, -1.0],
            [0.5, -0.5],
        ],
        dtype=np.float32,
    )

    variant_statistics = compute_variant_statistics(
        raw_genotypes=as_raw_genotype_matrix(genotype_matrix),
        config=ModelConfig(),
    )

    np.testing.assert_allclose(variant_statistics.allele_frequencies, np.array([0.5, 0.5], dtype=np.float32))


def test_compute_variant_statistics_preserves_af_for_continuous_dosage_columns():
    genotype_matrix = np.array(
        [
            [0.2, 1.8],
            [0.4, 1.6],
            [0.6, 1.4],
            [0.8, 1.2],
        ],
        dtype=np.float32,
    )

    variant_statistics = compute_variant_statistics(
        raw_genotypes=as_raw_genotype_matrix(genotype_matrix),
        config=ModelConfig(),
    )

    np.testing.assert_allclose(variant_statistics.allele_frequencies, np.array([0.25, 0.75], dtype=np.float32))


def test_collapse_tie_groups_preserves_support_and_continuous_features():
    genotype_matrix = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    variant_records = [
        VariantRecord(
            "variant_0",
            VariantClass.DELETION_SHORT,
            "1",
            100,
            training_support=6,
            prior_binary_features={"coding_annotation": True},
            prior_continuous_features={"sv_length_score": 1.0},
        ),
        VariantRecord(
            "variant_1",
            VariantClass.DELETION_SHORT,
            "1",
            101,
            training_support=8,
            prior_binary_features={"coding_annotation": False},
            prior_continuous_features={"sv_length_score": 3.0},
        ),
        VariantRecord("variant_2", VariantClass.SNV, "1", 102),
    ]

    tie_map = build_tie_map(genotype_matrix, variant_records, ModelConfig())
    collapsed_records = collapse_tie_groups(variant_records, tie_map)

    assert collapsed_records[0].training_support == 7
    assert collapsed_records[0].prior_binary_features == {"coding_annotation": True}
    assert collapsed_records[0].prior_continuous_features == {"sv_length_score": 2.0}
