"""Regression tests for the numerical guard fixes in ``mixture_inference``.

Each test corresponds to one of the six high-priority audit fixes:

1. ``np.log`` of a saturated sigmoid must clamp via ``np.maximum`` (no NaN).
2. ``_gig_moment`` must remain finite when both Bessel-K evaluations underflow.
3. The Polya-Gamma pseudo-response must not blow up when weights are tiny.
4. ``sigma_e^2`` must stay bounded when total leverage approaches the sample count.
5. ``_relative_change`` must return a bounded value when the previous norm is tiny.
6. Resuming from a checkpoint must produce a finite ``parameter_change`` diagnostic.
"""

from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.inference import fit_variational_em
from sv_pgs.mixture_inference import (
    VariationalFitCheckpoint,
    _gig_moment,
    _relative_change,
    _stochastic_epoch_objective,
)
from sv_pgs.preprocessing import build_tie_map

from tests.conftest import make_variant_records


# ---------- Fix 1: saturated-sigmoid log clamp ------------------------------


def test_stochastic_epoch_objective_finite_at_saturated_sigmoid():
    """eta = +700 saturates sigma(eta) to exactly 1.0 in float64.  Without the
    np.maximum clamp, log(1.0 - 1.0 + 1e-12) = log(1e-12) is fine but
    log(p + 1e-12) for p = 0 (when eta = -700) is log(1e-12), and the
    intermediate ``1.0 - probabilities`` can go (very slightly) negative due
    to fp rounding of stable_sigmoid, producing NaN under ``log``.  With the
    np.maximum clamp the result must always be finite.
    """
    target = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    saturated_predictor = np.array([700.0, -700.0, 700.0, -700.0], dtype=np.float64)
    objective = _stochastic_epoch_objective(
        trait_type=TraitType.BINARY,
        targets=target,
        linear_predictor=saturated_predictor,
        beta=np.zeros(2, dtype=np.float64),
        reduced_prior_variances=np.ones(2, dtype=np.float64),
        local_scale=np.ones(2, dtype=np.float64),
        auxiliary_delta=np.ones(2, dtype=np.float64),
        local_shape_a=np.ones(2, dtype=np.float64),
        local_shape_b=np.ones(2, dtype=np.float64),
        scale_model_coefficients=np.zeros(1, dtype=np.float64),
        scale_penalty=np.zeros(1, dtype=np.float64),
    )
    assert np.isfinite(objective), "objective must remain finite at sigmoid saturation"


# ---------- Fix 2: Bessel-K ratio underflow ---------------------------------


def test_gig_moment_finite_when_both_bessels_underflow():
    """With chi=psi~1e-300, both Bessel evaluations may underflow to 0; the ratio
    must therefore be guarded on BOTH ends, not only the denominator."""
    moment = _gig_moment(
        p_parameter=np.array(-1.0),
        chi=np.array(1e-300),
        psi=np.array(1e-300),
        moment_power=1.0,
    )
    assert np.isfinite(np.asarray(moment)).all()


# ---------- Fix 3: division by near-zero PG weight --------------------------


def test_binary_fit_stable_when_weights_are_tiny(random_generator):
    """Drive the binary fit through the IRLS path: with extreme initial-coef
    behaviour the Polya-Gamma weights can be near the minimum floor; the
    re-floor before ``pseudo_response = kappa / current_weights`` must keep
    the fit finite end-to-end.
    """
    sample_count, variant_count = 60, 5
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    covariate_matrix = np.ones((sample_count, 1), dtype=np.float32)
    target_vector = (random_generator.random(sample_count) < 0.5).astype(np.float32)
    records = make_variant_records(variant_count)
    config = ModelConfig(trait_type=TraitType.BINARY, max_outer_iterations=3)
    result = fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=config,
        tie_map=build_tie_map(genotype_matrix, records, config),
    )
    assert np.all(np.isfinite(result.beta_reduced))
    assert np.all(np.isfinite(result.alpha))
    assert all(np.isfinite(value) for value in result.objective_history)


# ---------- Fix 4: effective_dof floor --------------------------------------


def test_effective_dof_floor_keeps_sigma_e2_bounded(random_generator):
    """A high-variant / low-sample QTL fit pushes total leverage close to n.  The
    fix raises the effective-dof floor from 1.0 to max(2, 1% n) — verify that
    sigma_e^2 stays bounded (no inf/nan) under that regime.
    """
    sample_count, variant_count = 30, 25
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    covariate_matrix = np.ones((sample_count, 1), dtype=np.float32)
    target_vector = random_generator.standard_normal(sample_count).astype(np.float32)
    records = make_variant_records(variant_count)
    config = ModelConfig(trait_type=TraitType.QUANTITATIVE, max_outer_iterations=4)
    result = fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=config,
        tie_map=build_tie_map(genotype_matrix, records, config),
    )
    assert np.isfinite(result.sigma_error2)
    assert result.sigma_error2 < 1e6, "sigma_e^2 should stay bounded under high leverage"


# ---------- Fix 5: relative-change with near-zero baseline ------------------


def test_relative_change_bounded_when_previous_is_zero():
    previous = np.zeros(8, dtype=np.float64)
    current = 1e-10 * np.random.default_rng(0).standard_normal(8)
    value = _relative_change(current, previous)
    assert np.isfinite(value)
    # With the new adaptive floor (norm(current) * 1e-8) the ratio is roughly
    # 1 / 1e-8 = 1e8 at worst — but with both norm-zero and tiny-current paths
    # the value should remain a sane bounded ratio, not 1e10 or infinity.
    assert value < 1e9


def test_relative_change_handles_both_norms_tiny():
    previous = 1e-20 * np.ones(4, dtype=np.float64)
    current = 1e-20 * np.ones(4, dtype=np.float64)
    value = _relative_change(current, previous)
    assert np.isfinite(value)
    assert value == pytest.approx(0.0, abs=1e-6)


# ---------- Fix 6: parameter_change finite after checkpoint resume ----------


def test_parameter_change_finite_after_resume(random_generator):
    """Run two iterations, save a checkpoint at iteration 1, resume, and verify
    the post-resume diagnostic is finite (not NaN from a missing
    previous_linear_predictor).
    """
    sample_count, variant_count = 40, 6
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    covariate_matrix = np.column_stack(
        [np.ones(sample_count, dtype=np.float32), random_generator.standard_normal(sample_count).astype(np.float32)]
    )
    target_vector = random_generator.standard_normal(sample_count).astype(np.float32)
    records = make_variant_records(variant_count)
    tie_map = build_tie_map(genotype_matrix, records, ModelConfig(trait_type=TraitType.QUANTITATIVE))

    captured: list[VariationalFitCheckpoint] = []

    def capture(checkpoint: VariationalFitCheckpoint) -> None:
        captured.append(checkpoint)

    config_pass_one = ModelConfig(trait_type=TraitType.QUANTITATIVE, max_outer_iterations=2)
    fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=config_pass_one,
        tie_map=tie_map,
        checkpoint_callback=capture,
    )
    assert captured, "expected at least one checkpoint from the first pass"
    mid_checkpoint = captured[0]

    config_pass_two = ModelConfig(trait_type=TraitType.QUANTITATIVE, max_outer_iterations=3)
    resumed = fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=config_pass_two,
        tie_map=tie_map,
        resume_checkpoint=mid_checkpoint,
    )
    # The convergence diagnostic on the first post-resume iteration is the
    # one that previously could be NaN — assert it is finite.
    assert resumed.final_parameter_change is None or np.isfinite(resumed.final_parameter_change)
    assert resumed.final_predictor_change is None or np.isfinite(resumed.final_predictor_change)
    assert all(np.isfinite(value) for value in resumed.objective_history)
