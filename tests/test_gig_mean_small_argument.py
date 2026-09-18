"""E[lambda] for the GIG local-scale posterior when chi * psi or chi / psi is tiny.

For half-integer orders the Bessel ratio is elementary, which gives an exact
reference: K_{3/2}(z) / K_{1/2}(z) = 1 + 1/z, so for p = 1/2 (shape a = 1)

    E[lambda] = sqrt(chi / psi) * (1 + 1/z) = sqrt(chi / psi) + 1 / psi,

and K_{5/2}(z) / K_{3/2}(z) = (z^2 + 3z + 3) / (z (z + 1)) for p = 3/2 (a = 2).
"""

from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.config import ModelConfig
from sv_pgs.mixture_inference import _gig_moment, _update_local_scales


def _half_integer_gig_mean(p_parameter: float, chi: np.ndarray, psi: np.ndarray) -> np.ndarray:
    z_value = np.sqrt(chi * psi)
    if p_parameter == 0.5:
        ratio = 1.0 + 1.0 / z_value
    else:
        ratio = (z_value * z_value + 3.0 * z_value + 3.0) / (z_value * (z_value + 1.0))
    return np.sqrt(chi / psi) * ratio


@pytest.mark.parametrize("p_parameter", [0.5, 1.5])
def test_gig_mean_is_exact_when_chi_psi_or_chi_over_psi_is_tiny(p_parameter: float) -> None:
    # chi * psi below 1e-12, chi / psi below 1e-12, and the caller's floors
    # (chi = 1e-12, psi = 2 * local_scale_floor).
    chi = np.array([1e-10, 1e-12, 1e-12, 1e-6])
    psi = np.array([1e-3, 4.0, 2e-8, 2.0])
    moment = _gig_moment(
        p_parameter=np.full(chi.shape, p_parameter),
        chi=chi,
        psi=psi,
        moment_power=1.0,
    )
    np.testing.assert_allclose(moment, _half_integer_gig_mean(p_parameter, chi, psi), rtol=1e-10)


def test_local_scale_update_at_the_chi_floor_uses_the_gamma_limit() -> None:
    # A heavily shrunk variant with a large baseline hits the chi = 1e-12 floor.
    # With shape a = 1 its GIG posterior is essentially Gamma(1/2, rate delta),
    # whose mean 1/psi = 1/(2 delta) the update must return (not 2x that).
    config = ModelConfig()
    auxiliary_delta = np.array([2.0])
    local_scale, _ = _update_local_scales(
        coefficient_second_moment=np.array([1e-12]),
        baseline_prior_variances=np.array([1.0]),
        local_shape_a=np.array([1.0]),
        local_shape_b=np.array([1.0]),
        auxiliary_delta=auxiliary_delta,
        config=config,
    )
    expected = _half_integer_gig_mean(0.5, np.array([1e-12]), 2.0 * auxiliary_delta)
    np.testing.assert_allclose(local_scale, expected, rtol=1e-10)
