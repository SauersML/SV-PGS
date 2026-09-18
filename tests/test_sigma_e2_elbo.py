"""Tests for the exact CAVI stationary-point sigma_e^2 update.

With Z = [W | X] and Cov the joint posterior covariance of (alpha, beta) under
the flat covariate prior,

    sigma_e^2_new = (RSS + tr(Z'Z Cov)) / n,
    tr(Z'Z Cov) = sigma_e^2 * (k + sum_j (1 - Cov_jj / tau_j^2)),

because Cov = (Z'Z / sigma_e^2 + diag(0, 1/tau^2))^{-1}. The identity needs only the diagonal.
The former n * sum_j Cov_jj equals it only for orthogonal genotype columns, so
these tests use strongly correlated (LD-like) columns and a dense exact reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from sv_pgs import mixture_inference
from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.genotype import as_raw_genotype_matrix
from sv_pgs.inference import fit_variational_em
from sv_pgs.mixture_inference import (
    _build_prior_design,
    _quantitative_posterior_state,
    _scale_state_reduced_prior_variances,
)
from sv_pgs.preprocessing import build_tie_map

from tests.conftest import make_variant_records


def _correlated_standardized_genotypes(
    random_generator: np.random.Generator,
    sample_count: int,
    variant_count: int,
    correlation: float = 0.95,
    block_size: int = 10,
) -> np.ndarray:
    latent = np.empty((sample_count, variant_count), dtype=np.float64)
    for block_start in range(0, variant_count, block_size):
        block_stop = min(block_start + block_size, variant_count)
        block = random_generator.standard_normal((sample_count, block_stop - block_start))
        for column_index in range(1, block_stop - block_start):
            block[:, column_index] = (
                correlation * block[:, column_index - 1]
                + np.sqrt(1.0 - correlation**2) * block[:, column_index]
            )
        latent[:, block_start:block_stop] = block
    latent -= latent.mean(axis=0, keepdims=True)
    latent /= latent.std(axis=0, keepdims=True)
    return latent.astype(np.float32)


def _standardized_matrix(genotype_values: np.ndarray):
    variant_count = genotype_values.shape[1]
    standardized = as_raw_genotype_matrix(genotype_values).standardized(
        means=np.zeros(variant_count, dtype=np.float32),
        scales=np.ones(variant_count, dtype=np.float32),
    )
    standardized._dense_cache = standardized.materialize()
    return standardized


def _exact_sigma_error2_update(
    genotype_values: np.ndarray,
    covariate_matrix: np.ndarray,
    targets: np.ndarray,
    prior_precision: np.ndarray,
    sigma_error2: float,
) -> tuple[float, float]:
    design = np.hstack([covariate_matrix.astype(np.float64), genotype_values.astype(np.float64)])
    precision = np.concatenate([np.zeros(covariate_matrix.shape[1]), prior_precision])
    posterior_precision = design.T @ design / sigma_error2 + np.diag(precision)
    covariance = np.linalg.inv(posterior_precision)
    mean = covariance @ design.T @ targets.astype(np.float64) / sigma_error2
    residual = targets.astype(np.float64) - design @ mean
    residual_sum_squares = float(residual @ residual)
    trace = float(np.sum((design.T @ design) * covariance))
    sample_count = targets.shape[0]
    diagonal_only_trace = sample_count * float(np.sum(np.diag(covariance)[covariate_matrix.shape[1]:]))
    return (residual_sum_squares + trace) / sample_count, (residual_sum_squares + diagonal_only_trace) / sample_count


@pytest.mark.parametrize(
    ("sample_count", "variant_count"),
    [(300, 120), (120, 300)],
    ids=["variant_space", "sample_space"],
)
def test_sigma_e2_matches_exact_cavi_update_under_ld(sample_count: int, variant_count: int):
    random_generator = np.random.default_rng(7)
    genotype_values = _correlated_standardized_genotypes(random_generator, sample_count, variant_count)
    covariate_matrix = np.column_stack(
        [np.ones(sample_count), random_generator.standard_normal(sample_count)]
    ).astype(np.float32)
    true_beta = np.zeros(variant_count)
    true_beta[random_generator.choice(variant_count, 6, replace=False)] = 0.4
    targets = (
        genotype_values.astype(np.float64) @ true_beta + random_generator.standard_normal(sample_count)
    ).astype(np.float32)
    prior_variances = random_generator.uniform(0.005, 0.05, size=variant_count)
    sigma_error2 = 0.9

    *_unused, sigma_error2_new = _quantitative_posterior_state(
        genotype_matrix=_standardized_matrix(genotype_values),
        covariate_matrix=covariate_matrix,
        targets=targets,
        prior_variances=prior_variances,
        sigma_error2=sigma_error2,
        sigma_error_floor=1e-8,
        solver_tolerance=1e-12,
        compute_logdet=False,
        compute_beta_variance=True,
    )

    expected, diagonal_only = _exact_sigma_error2_update(
        genotype_values, covariate_matrix, targets, 1.0 / prior_variances, sigma_error2
    )
    # The correlated design must make the orthogonal-column shortcut wrong.
    assert diagonal_only > 1.2 * expected
    assert sigma_error2_new == pytest.approx(expected, rel=1e-6)


def test_stochastic_epoch_sigma_e2_uses_the_prior_the_block_solves_used(monkeypatch: pytest.MonkeyPatch):
    """Epoch-end sigma_e^2 = RSS / (n - sum_j (1 - Sigma_jj / tau_j^2)) with the
    tau^2 the blocks were solved under, not the tau^2 re-estimated after the epoch."""
    # Pin the block partition the reference below reproduces.
    monkeypatch.setattr(
        mixture_inference,
        "_adaptive_stochastic_variant_block_size",
        lambda _genotype_matrix, configured_block_size: configured_block_size,
    )
    random_generator = np.random.default_rng(3)
    sample_count, variant_count, block_size = 200, 40, 20
    genotype_values = _correlated_standardized_genotypes(random_generator, sample_count, variant_count)
    covariate_matrix = np.ones((sample_count, 1), dtype=np.float32)
    true_beta = np.zeros(variant_count)
    true_beta[[2, 17, 31]] = [0.6, -0.5, 0.4]
    targets = (
        genotype_values.astype(np.float64) @ true_beta + random_generator.standard_normal(sample_count)
    ).astype(np.float32)
    records = make_variant_records(variant_count)
    config = ModelConfig(
        trait_type=TraitType.QUANTITATIVE,
        max_outer_iterations=2,
        beta_variance_update_interval=1,
        stochastic_variational_updates=True,
        stochastic_min_variant_count=0,
        stochastic_variant_batch_size=block_size,
        final_posterior_diagnostics=False,
        linear_solver_tolerance=1e-12,
    )
    epoch_end_checkpoints = []
    epoch_snapshots = []
    fit_variational_em(
        genotypes=genotype_values,
        covariates=covariate_matrix,
        targets=targets,
        records=records,
        config=config,
        tie_map=build_tie_map(genotype_values, records, config),
        checkpoint_callback=lambda checkpoint: (
            epoch_end_checkpoints.append(checkpoint)
            if checkpoint.completed_blocks_in_iteration == 0
            else None
        ),
        per_epoch_eval_callback=epoch_snapshots.append,
    )
    assert [checkpoint.completed_iterations for checkpoint in epoch_end_checkpoints] == [1, 2]

    first_epoch = epoch_end_checkpoints[0]
    solve_prior_variances = _scale_state_reduced_prior_variances(
        global_scale=first_epoch.global_scale,
        scale_model_coefficients=first_epoch.scale_model_coefficients,
        local_scale=first_epoch.local_scale,
        design_matrix=_build_prior_design(records).design_matrix,
        config=config,
    )
    second_epoch_prior_variances = _scale_state_reduced_prior_variances(
        global_scale=epoch_end_checkpoints[1].global_scale,
        scale_model_coefficients=epoch_end_checkpoints[1].scale_model_coefficients,
        local_scale=epoch_end_checkpoints[1].local_scale,
        design_matrix=_build_prior_design(records).design_matrix,
        config=config,
    )
    # The hyperparameter update inside epoch 2 must move the prior, otherwise
    # the two leverage definitions coincide and the test cannot tell them apart.
    assert np.max(np.abs(np.log(second_epoch_prior_variances / solve_prior_variances))) > 0.05

    genotypes64 = genotype_values.astype(np.float64)
    leverage = 0.0
    for block_start in range(0, variant_count, block_size):
        block = slice(block_start, block_start + block_size)
        block_genotypes = genotypes64[:, block]
        block_covariance = np.linalg.inv(
            block_genotypes.T @ block_genotypes / first_epoch.sigma_error2
            + np.diag(1.0 / solve_prior_variances[block])
        )
        leverage += float(np.sum(1.0 - np.diag(block_covariance) / solve_prior_variances[block]))
    second_epoch = epoch_snapshots[1]
    residual = (
        targets.astype(np.float64)
        - covariate_matrix.astype(np.float64) @ second_epoch["alpha_reduced"]
        - genotypes64 @ second_epoch["beta_reduced"]
    )
    expected = float(residual @ residual) / (sample_count - leverage)
    assert second_epoch["sigma_error2"] == pytest.approx(expected, rel=1e-5)
