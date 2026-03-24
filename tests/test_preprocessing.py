import numpy as np

from sv_pgs.config import ModelConfig, VariantClass
from sv_pgs.data import VariantRecord
from sv_pgs.preprocessing import build_tie_map, fit_preprocessor


def test_fold_preprocessing_and_tie_map():
    genotypes = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, -1.0, np.nan],
            [2.0, 2.0, -2.0, 0.0],
            [np.nan, np.nan, np.nan, 1.0],
        ],
        dtype=np.float32,
    )
    covariates = np.zeros((4, 1), dtype=np.float32)
    targets = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    records = [
        VariantRecord("v0", VariantClass.SNV, "na", "1", 100),
        VariantRecord("v1", VariantClass.SNV, "na", "1", 101),
        VariantRecord("v2", VariantClass.SNV, "na", "1", 102),
        VariantRecord("v3", VariantClass.DELETION_SHORT, "short", "1", 103),
    ]

    prepared = fit_preprocessor(genotypes, covariates, targets, ModelConfig())
    tie_map = build_tie_map(prepared.genotypes, records)

    assert prepared.genotypes.shape == genotypes.shape
    np.testing.assert_allclose(prepared.genotypes.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.mean(prepared.genotypes**2, axis=0), 1.0, atol=1e-5)
    assert tie_map.kept_indices.tolist() == [0, 3]
    assert tie_map.original_to_reduced.tolist() == [0, 0, 0, 1]
    np.testing.assert_allclose(tie_map.reduced_to_group[0].signs, [1.0, 1.0, -1.0])
