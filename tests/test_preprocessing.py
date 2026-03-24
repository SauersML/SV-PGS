import numpy as np

from sv_pgs.config import ModelConfig, VariantClass
from sv_pgs.data import VariantRecord
from sv_pgs.preprocessing import build_tie_map, collapse_tie_groups, fit_preprocessor


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
    tie_map = build_tie_map(prepared_arrays.genotypes, variant_records, ModelConfig())

    assert prepared_arrays.genotypes.shape == genotype_matrix.shape
    np.testing.assert_allclose(prepared_arrays.genotypes.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.mean(prepared_arrays.genotypes**2, axis=0), 1.0, atol=1e-5)
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
