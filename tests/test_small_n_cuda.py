"""The small-n fit with its variant side on the device against the same fit on the host: one route, two array modules,
the same certified answer (``scale_mixture_ep.device_scope``)."""
from __future__ import annotations

import sys

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")

sys.path.insert(0, "tests")
from test_mean_field import _WORKING_BYTES, _problem  # noqa: E402

from sv_pgs.small_n import fit_small_n  # noqa: E402


def test_the_device_fit_is_the_host_fit():
    codes, covariates, target, classes = _problem(21, samples=160, variants=50)
    fits = [
        fit_small_n(
            codes=codes, covariates=covariates, target=target, variant_class=classes, log_variance_offset=None, draw_count=64,
            working_bytes=_WORKING_BYTES, seed=0, inference="mean_field", array_module=module,
        )
        for module in (None, cupy)
    ]
    host, device = fits
    assert host.profile["outer_criterion_met"] == device.profile["outer_criterion_met"]
    # The device kernels agree with the host to rounding on every value (test_engine_kernels_cuda), so the searches
    # take the same path: the same weights and coefficients, and the same scoring model, to the fit's own resolution.
    np.testing.assert_array_equal(np.isinf(host.hyperparameters.log_smoothing), np.isinf(device.hyperparameters.log_smoothing))
    np.testing.assert_allclose(host.hyperparameters.coefficients, device.hyperparameters.coefficients, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(host.scoring.coefficients, device.scoring.coefficients, rtol=0.0, atol=1e-8)
    assert abs(host.noise_variance - device.noise_variance) <= 1e-10
