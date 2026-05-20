"""Tests for the automatic E[1/lambda] CAVI-correct prior-precision construction.

The beta precision uses ``E[1/lambda] / (sigma_g^2 * s_j^2)`` whenever the
previous posterior second moment is available.
"""

from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.inference import fit_variational_em
from sv_pgs.mixture_inference import (
    _build_cavi_correct_prior_precision,
    _update_local_scales,
)
from sv_pgs.optimizer_helpers import gig_inverse_first_moment
from sv_pgs.preprocessing import build_tie_map

from tests.conftest import make_variant_records


@pytest.fixture
def random_generator() -> np.random.Generator:
    return np.random.default_rng(2026)


def test_inverse_first_moment_matches_local_scale_parameters(random_generator):
    """E[lambda] and E[1/lambda] are computed from the same GIG parameters."""
    config = ModelConfig()
    rng = random_generator
    p = 12
    coefficient_second_moment = rng.uniform(1e-4, 1.0, size=p)
    baseline_prior_variances = rng.uniform(1e-3, 1e-1, size=p)
    local_shape_a = np.full(p, 1.0)
    local_shape_b = np.full(p, 0.5)
    auxiliary_delta = rng.uniform(0.5, 2.0, size=p)

    lam_default, delta_default = _update_local_scales(
        coefficient_second_moment=coefficient_second_moment,
        baseline_prior_variances=baseline_prior_variances,
        local_shape_a=local_shape_a,
        local_shape_b=local_shape_b,
        auxiliary_delta=auxiliary_delta,
        config=config,
    )
    chi = coefficient_second_moment / np.maximum(baseline_prior_variances, 1e-12)
    inv_lam = gig_inverse_first_moment(
        p_parameter=local_shape_a - 0.5,
        chi=chi,
        psi=2.0 * np.maximum(auxiliary_delta, config.local_scale_floor),
    )

    assert lam_default.shape == inv_lam.shape
    assert delta_default.shape == inv_lam.shape
    assert np.all(np.isfinite(inv_lam))
    assert np.all(inv_lam >= 0.0)


def test_build_cavi_correct_prior_precision_is_finite_for_extreme_inputs():
    """The precision builder should produce finite positive values for extreme
    but valid GIG parameters."""
    config = ModelConfig()
    p = 6
    baseline = np.full(p, 1e-2)
    # Second moment of zero forces chi -> floor; combined with delta -> floor
    # the kve ratio still returns finite values, so we explicitly verify the
    # function output is finite for an extreme but representative input.
    second_moment = np.zeros(p)
    aux_delta = np.full(p, 1e-12)
    precision = _build_cavi_correct_prior_precision(
        reduced_second_moment=second_moment,
        baseline_prior_variances=baseline,
        local_shape_a=np.full(p, 1.0),
        auxiliary_delta=aux_delta,
        config=config,
    )
    assert precision.shape == (p,)
    assert np.all(np.isfinite(precision))
    assert np.all(precision > 0.0)


def _fit_once(*, seed: int = 12345):
    rng = np.random.default_rng(seed)
    sample_count, variant_count = 96, 14
    genotype_matrix = rng.standard_normal((sample_count, variant_count)).astype(np.float32)
    covariate_matrix = np.column_stack(
        [np.ones(sample_count), rng.standard_normal((sample_count, 1))]
    ).astype(np.float32)
    true_coefficients = np.zeros(variant_count, dtype=np.float32)
    true_coefficients[0] = 1.2
    true_coefficients[3] = -0.8
    target_vector = (
        genotype_matrix @ true_coefficients
        + rng.standard_normal(sample_count).astype(np.float32) * 0.3
    )
    records = make_variant_records(variant_count)
    config = ModelConfig(
        trait_type=TraitType.QUANTITATIVE,
        max_outer_iterations=25,
        random_seed=0,
    )
    result = fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=config,
        tie_map=build_tie_map(genotype_matrix, records, config),
    )
    return np.asarray(result.beta_reduced, dtype=np.float64)


def test_inverse_first_moment_path_recovers_signal_direction():
    beta_inverse = _fit_once()
    expected = np.zeros_like(beta_inverse)
    expected[0] = 1.2
    expected[3] = -0.8

    corr = float(np.corrcoef(beta_inverse, expected)[0, 1])
    assert corr > 0.80, f"betas correlate at {corr:.3f}, expected > 0.80"
