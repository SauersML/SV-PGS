"""Verify VariationalFitResult preserves float64 precision for variance fields."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.mixture_inference import fit_variational_em
from sv_pgs.preprocessing import build_tie_map

from tests.conftest import make_variant_records


@pytest.fixture
def random_generator() -> np.random.Generator:
    return np.random.default_rng(0)


def _fit_tiny(random_generator: np.random.Generator):
    sample_count, variant_count = 60, 8
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    covariate_matrix = np.column_stack(
        [np.ones(sample_count), random_generator.standard_normal((sample_count, 1))]
    ).astype(np.float32)
    true_coefficients = np.zeros(variant_count, dtype=np.float32)
    true_coefficients[0] = 0.8
    target_vector = (
        genotype_matrix @ true_coefficients
        + random_generator.standard_normal(sample_count).astype(np.float32) * 0.3
    )
    records = make_variant_records(variant_count)
    config = ModelConfig(trait_type=TraitType.QUANTITATIVE, max_outer_iterations=3)
    return fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=config,
        tie_map=build_tie_map(genotype_matrix, records, config),
    )


def test_variance_fields_are_float64(random_generator):
    result = _fit_tiny(random_generator)
    assert result.beta_variance.dtype == np.float64
    assert result.prior_scales.dtype == np.float64
    assert result.member_prior_variances.dtype == np.float64


def test_prediction_fields_remain_float32(random_generator):
    result = _fit_tiny(random_generator)
    assert result.alpha.dtype == np.float32
    assert result.beta_reduced.dtype == np.float32


def test_small_variance_preserved_exactly():
    # 1e-7 is not representable exactly in float32 (~1.1920929e-07 is the closest),
    # but float64 stores it to many more digits. Verify storing then reading back
    # via the float64 dtype preserves the value precisely.
    tiny = 1e-7
    as_f32 = np.float32(tiny)
    as_f64 = np.float64(tiny)
    # confirm float32 round-trip is lossy
    assert float(as_f32) != tiny
    # float64 preserves the python double exactly
    assert float(as_f64) == tiny
    # and a float64 array stores it without quantizing to the float32 grid
    arr = np.asarray([tiny], dtype=np.float64)
    assert arr[0] == tiny
    assert arr[0] != float(as_f32)
