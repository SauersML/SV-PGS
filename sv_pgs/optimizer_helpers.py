"""Helpers for cleaner CAVI updates: closed-form sigma_g and GIG inverse moment.

These replace pieces of the existing damped Newton on theta = (log sigma_g, scale_model_coefs).
For the sigma_g part only, a closed-form inverse-Gamma update is exact CAVI - the
existing Newton was wasted work along that direction.
"""

from __future__ import annotations
import numpy as np
from scipy.special import kve as scipy_bessel_kve


def closed_form_global_scale(
    *,
    coefficient_second_moment: np.ndarray,   # E_q[beta_j^2]
    metadata_baseline_scales: np.ndarray,    # s_j (exp of design @ theta excluding intercept)
    local_scale: np.ndarray,                 # lambda_j
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
    p_parameter: np.ndarray,
    chi: np.ndarray,
    psi: np.ndarray,
) -> np.ndarray:
    """E[1/X] for X ~ GIG(p, chi, psi) - used in the beta precision when we collapse lambda.

    Formula:  E[1/X] = sqrt(psi/chi) * K_{p-1}(sqrt(chi*psi)) / K_p(sqrt(chi*psi))   (when chi > 0)
            = (2 p / chi) * (mean^-1 correction)   (when psi = 0 limit, Gamma case)

    Uses exponentially-scaled kve for numerical stability.
    """
    chi_arr = np.asarray(chi, dtype=np.float64)
    psi_arr = np.asarray(psi, dtype=np.float64)
    p_arr = np.asarray(p_parameter, dtype=np.float64)
    z = np.sqrt(np.maximum(chi_arr * psi_arr, 1e-300))
    safe_chi = np.maximum(chi_arr, 1e-300)
    # Use ratio identity: E[1/X] = (sqrt(psi/chi)) K_{p-1}/K_p
    # K_{p-1}(z) / K_p(z), via kve which is K_v(z) * exp(z):
    numerator = scipy_bessel_kve(np.abs(p_arr - 1.0), z)
    denominator = np.maximum(scipy_bessel_kve(np.abs(p_arr), z), 1e-300)
    return np.sqrt(np.maximum(psi_arr / safe_chi, 1e-300)) * (numerator / denominator)


def pack_em_hyperparameters(
    *,
    log_global_scale: float,
    scale_model_coefficients: np.ndarray,
    log_tpb_shape_a_vector: np.ndarray,
    log_tpb_shape_b_vector: np.ndarray,
) -> np.ndarray:
    return np.concatenate([
        np.asarray([log_global_scale], dtype=np.float64),
        np.asarray(scale_model_coefficients, dtype=np.float64),
        np.asarray(log_tpb_shape_a_vector, dtype=np.float64),
        np.asarray(log_tpb_shape_b_vector, dtype=np.float64),
    ])


def unpack_em_hyperparameters(
    packed: np.ndarray,
    *,
    scale_model_dim: int,
    tpb_class_count: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
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
