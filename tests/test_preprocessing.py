import numpy as np
import pytest

from sv_pgs.config import ModelConfig, VariantClass
from sv_pgs.data import VariantRecord
from sv_pgs.genotype import as_raw_genotype_matrix
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
    tie_map = build_tie_map(standardized_genotypes, variant_records, ModelConfig())

    assert standardized_genotypes.shape == genotype_matrix.shape
    np.testing.assert_allclose(standardized_genotypes.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.mean(standardized_genotypes**2, axis=0), 1.0, atol=1e-5)
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


def test_select_active_variant_indices_keeps_low_support_structural_variants():
    variant_records = [
        VariantRecord("sv_0", VariantClass.DELETION_SHORT, "1", 100, training_support=1),
        VariantRecord("sv_1", VariantClass.DELETION_SHORT, "1", 101, training_support=5),
        VariantRecord("snp_2", VariantClass.SNV, "1", 102),
    ]

    result = select_active_variant_indices(
        variant_records=variant_records,
        config=ModelConfig(),
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
            prior_continuous_features={"sv_length_score": 1.0},
        ),
        VariantRecord(
            "variant_1",
            VariantClass.DELETION_SHORT,
            "1",
            101,
            training_support=8,
            prior_continuous_features={"sv_length_score": 3.0},
        ),
        VariantRecord("variant_2", VariantClass.SNV, "1", 102),
    ]

    tie_map = build_tie_map(genotype_matrix, variant_records, ModelConfig())
    collapsed_records = collapse_tie_groups(variant_records, tie_map)

    assert collapsed_records[0].training_support == 7
    assert collapsed_records[0].prior_continuous_features == {"sv_length_score": 2.0}
