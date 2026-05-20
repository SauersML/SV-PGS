"""Focused tests for the L-BFGS-B based TPB shape update.

Verifies that `_update_tpb_shape_vectors` recovers known TPB shape parameters
from synthetic data in far fewer iterations than the legacy projected
gradient ascent loop required.
"""

from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.mixture_inference import _update_tpb_shape_vectors


def _draw_tpb_local_scale(
    shape_a: float,
    shape_b: float,
    num_variants: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw (local_scale, auxiliary_delta) matching the variational sufficient
    statistics for TPB shape inference.

    The shape-update objective in `_update_tpb_shape_vectors` is a sum of
    Gamma log-likelihood terms whose stationary equations are
        digamma(a) = E[log δ] + E[log λ]
        digamma(b) = E[log δ]
    So we sample
        δ ~ Gamma(b*, 1)            (=> E[log δ] = digamma(b*))
        λ = γ / δ, γ ~ Gamma(a*, 1) (=> E[log λ] = digamma(a*) − digamma(b*))
    which makes (a*, b*) the population maximizer of the objective.
    """
    delta = rng.gamma(shape=shape_b, scale=1.0, size=num_variants)
    gamma_a = rng.gamma(shape=shape_a, scale=1.0, size=num_variants)
    delta = np.maximum(delta, 1e-12)
    local_scale = np.maximum(gamma_a / delta, 1e-12)
    auxiliary_delta = delta
    return local_scale.astype(np.float64), auxiliary_delta.astype(np.float64)


def _analytical_gradient_at(
    class_membership_matrix: np.ndarray,
    shape_a: np.ndarray,
    shape_b: np.ndarray,
    local_scale: np.ndarray,
    auxiliary_delta: np.ndarray,
    config: ModelConfig,
) -> np.ndarray:
    """Analytical gradient of the TPB shape objective w.r.t. (log a, log b).

    Mirrors the inner objective in `_update_tpb_shape_vectors` exactly,
    using scipy's special functions to avoid a JAX round-trip.
    """
    from scipy.special import digamma as scipy_digamma

    log_a = np.log(shape_a)
    log_b = np.log(shape_b)
    a_vec = np.exp(log_a)
    b_vec = np.exp(log_b)
    local_a = class_membership_matrix @ a_vec
    local_b = class_membership_matrix @ b_vec
    log_ls = np.log(np.maximum(local_scale, config.local_scale_floor))
    log_ad = np.log(np.maximum(auxiliary_delta, config.local_scale_floor))
    digamma_a = scipy_digamma(local_a)
    digamma_b = scipy_digamma(local_b)
    score_a = log_ad - digamma_a + log_ls
    score_b = log_ad - digamma_b
    centered_a = log_a - np.mean(log_a)
    centered_b = log_b - np.mean(log_b)
    grad_a = a_vec * (class_membership_matrix.T @ score_a) - centered_a / config.tpb_hierarchical_prior_variance
    grad_b = b_vec * (class_membership_matrix.T @ score_b) - centered_b / config.tpb_hierarchical_prior_variance
    return np.concatenate([grad_a, grad_b])


def test_lbfgs_recovers_known_tpb_shapes_quickly() -> None:
    rng = np.random.default_rng(20260520)
    class_count = 2
    num_variants_per_class = 100
    true_a = np.array([2.5, 4.0], dtype=np.float64)
    true_b = np.array([1.5, 3.0], dtype=np.float64)

    blocks = []
    membership_rows = []
    for class_index in range(class_count):
        ls, ad = _draw_tpb_local_scale(
            float(true_a[class_index]),
            float(true_b[class_index]),
            num_variants_per_class,
            rng,
        )
        blocks.append((ls, ad))
        membership_block = np.zeros((num_variants_per_class, class_count), dtype=np.float64)
        membership_block[:, class_index] = 1.0
        membership_rows.append(membership_block)

    local_scale = np.concatenate([b[0] for b in blocks])
    auxiliary_delta = np.concatenate([b[1] for b in blocks])
    class_membership_matrix = np.vstack(membership_rows)

    initial_a = np.ones(class_count, dtype=np.float64)
    initial_b = np.ones(class_count, dtype=np.float64)

    # Tight tolerance, but cap iterations far below the legacy default to
    # demonstrate L-BFGS-B converges in ≤ 30 iterations on this problem.
    config = ModelConfig(
        trait_type=TraitType.QUANTITATIVE,
        max_outer_iterations=1,
        maximum_tpb_shape_iterations=30,
        convergence_tolerance=1e-6,
    )

    recovered_a, recovered_b = _update_tpb_shape_vectors(
        class_membership_matrix,
        initial_a,
        initial_b,
        local_scale,
        auxiliary_delta,
        config,
    )

    # Recovered shapes finite and within bounds.
    assert np.all(np.isfinite(recovered_a))
    assert np.all(np.isfinite(recovered_b))
    assert np.all(recovered_a >= config.minimum_tpb_shape)
    assert np.all(recovered_a <= config.maximum_tpb_shape)
    assert np.all(recovered_b >= config.minimum_tpb_shape)
    assert np.all(recovered_b <= config.maximum_tpb_shape)

    # Within 50% of truth elementwise.
    relative_error_a = np.abs(recovered_a - true_a) / true_a
    relative_error_b = np.abs(recovered_b - true_b) / true_b
    assert np.all(relative_error_a < 0.5), (
        f"shape_a recovery off: got {recovered_a}, truth {true_a}"
    )
    assert np.all(relative_error_b < 0.5), (
        f"shape_b recovery off: got {recovered_b}, truth {true_b}"
    )

    # Gradient norm at the returned point is small. Because the unscaled
    # objective sums over all variants, we compare the per-variant gradient
    # norm against the requested 1e-3 tolerance.
    grad_vec = _analytical_gradient_at(
        class_membership_matrix,
        recovered_a,
        recovered_b,
        local_scale,
        auxiliary_delta,
        config,
    )
    num_variants = class_membership_matrix.shape[0]
    per_variant_grad_norm = float(np.linalg.norm(grad_vec)) / max(num_variants, 1)
    assert per_variant_grad_norm < 1e-3, (
        f"per-variant gradient norm at solution too large: {per_variant_grad_norm}"
    )


def test_lbfgs_empty_class_count_early_return() -> None:
    config = ModelConfig(
        trait_type=TraitType.QUANTITATIVE,
        max_outer_iterations=1,
    )
    empty_a = np.zeros(0, dtype=np.float64)
    empty_b = np.zeros(0, dtype=np.float64)
    membership = np.zeros((10, 0), dtype=np.float64)
    local_scale = np.ones(10, dtype=np.float64)
    aux = np.ones(10, dtype=np.float64)

    out_a, out_b = _update_tpb_shape_vectors(
        membership, empty_a, empty_b, local_scale, aux, config
    )
    assert out_a.shape == (0,)
    assert out_b.shape == (0,)
