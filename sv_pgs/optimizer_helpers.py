"""Helpers for cleaner CAVI updates: closed-form sigma_g and GIG inverse moment.

These replace pieces of the existing damped Newton on theta = (log sigma_g, scale_model_coefs).
For the sigma_g part only, a closed-form inverse-Gamma update is exact CAVI - the
existing Newton was wasted work along that direction.
"""

from __future__ import annotations
import numpy as np
from scipy.special import gammaln as scipy_gammaln
from scipy.special import kve as scipy_bessel_kve

from sv_pgs._typing import F64Array


def closed_form_global_scale(
    *,
    coefficient_second_moment: F64Array,   # E_q[beta_j^2]
    metadata_baseline_scales: F64Array,    # s_j (exp of design @ theta excluding intercept)
    local_scale: F64Array,                 # lambda_j
    prior_shape: float = 0.0,                # inverse-gamma alpha (use 0 for flat)
    prior_rate: float = 0.0,                 # inverse-gamma beta (use 0 for flat)
    floor: float = 1e-8,
    ceiling: float = 1e8,
) -> float:
    """Closed-form CAVI update for sigma_g given current (beta^2, lambda, scale-model).

    Mathematically: with tau_j^2 = (sigma_g * s_j)^2 * lambda_j and N(0, tau^2) prior on beta,
    the conditional log-posterior in sigma_g^2 (given everything else) is
    proportional to inverse-Gamma; its mode is

        sigma_g^2* = (beta_prior + 0.5 Sum_j E[beta^2]/(s^2 lambda)) / (alpha_prior + p/2 + 1)

    Result is clipped to [floor, ceiling].
    """
    p = int(coefficient_second_moment.shape[0])
    scaled = np.asarray(coefficient_second_moment, dtype=np.float64) / (
        np.maximum(np.asarray(metadata_baseline_scales, dtype=np.float64) ** 2, 1e-12)
        * np.maximum(np.asarray(local_scale, dtype=np.float64), 1e-12)
    )
    numerator = float(prior_rate) + 0.5 * float(np.sum(scaled))
    denominator = float(prior_shape) + 0.5 * float(p) + 1.0
    sigma_g_squared = max(numerator / max(denominator, 1e-12), floor ** 2)
    return float(np.clip(np.sqrt(sigma_g_squared), floor, ceiling))


def gig_inverse_first_moment(
    *,
    p_parameter: F64Array,
    chi: F64Array,
    psi: F64Array,
) -> F64Array:
    """E[1/X] for X ~ GIG(p, chi, psi) - used in the beta precision when we collapse lambda.

    Formula:  E[1/X] = sqrt(psi/chi) * K_{p-1}(sqrt(chi*psi)) / K_p(sqrt(chi*psi))   (when chi > 0)
            = -2p / chi                             (when psi -> 0 and p < 0)

    Uses exponentially-scaled kve for numerical stability.
    """
    chi_arr = np.asarray(chi, dtype=np.float64)
    psi_arr = np.asarray(psi, dtype=np.float64)
    p_arr = np.asarray(p_parameter, dtype=np.float64)
    z = np.sqrt(np.maximum(chi_arr * psi_arr, 1e-300))
    safe_chi = np.maximum(chi_arr, 1e-300)
    safe_psi = np.maximum(psi_arr, 1e-300)

    # Use ratio identity: E[1/X] = sqrt(psi/chi) * K_{p-1}(z) / K_p(z).
    # For tiny z, direct Bessel calls can overflow to inf and produce inf/inf.
    # The psi -> 0 limit is common in the optimizer and is exact for p < 0.
    result = np.empty(np.broadcast_shapes(p_arr.shape, chi_arr.shape, psi_arr.shape), dtype=np.float64)
    p_b, chi_b, psi_b, z_b = np.broadcast_arrays(p_arr, safe_chi, safe_psi, z)
    small_z = z_b < 1e-8

    inverse_gamma_limit = small_z & (p_b < 0.0)
    result[inverse_gamma_limit] = -2.0 * p_b[inverse_gamma_limit] / chi_b[inverse_gamma_limit]

    positive_small = small_z & ~inverse_gamma_limit
    if np.any(positive_small):
        p_small = p_b[positive_small]
        chi_small = chi_b[positive_small]
        psi_small = psi_b[positive_small]
        z_small = z_b[positive_small]
        small_result = np.empty_like(p_small, dtype=np.float64)

        near_one = np.isclose(p_small, 1.0, rtol=0.0, atol=1e-8)
        small_result[near_one] = psi_small[near_one] * np.maximum(
            -np.log(np.maximum(z_small[near_one] * 0.5, 1e-300)) - np.euler_gamma,
            0.0,
        )

        greater_than_one = (p_small > 1.0) & ~near_one
        small_result[greater_than_one] = psi_small[greater_than_one] / (2.0 * (p_small[greater_than_one] - 1.0))

        between_zero_and_one = ~(near_one | greater_than_one)
        if np.any(between_zero_and_one):
            p_mid = np.maximum(p_small[between_zero_and_one], 1e-12)
            log_value = (
                scipy_gammaln(1.0 - p_mid)
                - scipy_gammaln(p_mid)
                + p_mid * np.log(psi_small[between_zero_and_one])
                + (p_mid - 1.0) * np.log(chi_small[between_zero_and_one])
                - (2.0 * p_mid - 1.0) * np.log(2.0)
            )
            small_result[between_zero_and_one] = np.exp(np.clip(log_value, -745.0, 709.0))
        result[positive_small] = small_result

    regular = ~small_z
    if np.any(regular):
        numerator = scipy_bessel_kve(np.abs(p_b[regular] - 1.0), z_b[regular])
        denominator = scipy_bessel_kve(np.abs(p_b[regular]), z_b[regular])
        result[regular] = np.sqrt(psi_b[regular] / chi_b[regular]) * (numerator / denominator)

    return np.maximum(result, np.finfo(np.float64).tiny)


def pack_em_hyperparameters(
    *,
    log_global_scale: float,
    scale_model_coefficients: F64Array,
    log_tpb_shape_a_vector: F64Array,
    log_tpb_shape_b_vector: F64Array,
) -> F64Array:
    return np.concatenate([
        np.asarray([log_global_scale], dtype=np.float64),
        np.asarray(scale_model_coefficients, dtype=np.float64),
        np.asarray(log_tpb_shape_a_vector, dtype=np.float64),
        np.asarray(log_tpb_shape_b_vector, dtype=np.float64),
    ])


def unpack_em_hyperparameters(
    packed: F64Array,
    *,
    scale_model_dim: int,
    tpb_class_count: int,
) -> tuple[float, F64Array, F64Array, F64Array]:
    cursor = 0
    log_global_scale = float(packed[cursor])
    cursor += 1
    scale_coefs = packed[cursor:cursor + scale_model_dim].copy()
    cursor += scale_model_dim
    log_a = packed[cursor:cursor + tpb_class_count].copy()
    cursor += tpb_class_count
    log_b = packed[cursor:cursor + tpb_class_count].copy()
    cursor += tpb_class_count
    return log_global_scale, scale_coefs, log_a, log_b
