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
    # The device fit's mean-field oracle ascends the ELBO by parallel passes (``mean_field.MeanFieldFixedPoints._pass``)
    # where the host's sweeps, so the two searches are not one path to rounding: they are two certified answers to
    # one problem, and they agree at the certificate's own resolution. The remaining gain is within
    # tol = 0.5 / draws nats at each fixed point, so the ELBOs are within 2 tol; the noise's stationary gain for a
    # relative miss d is n_r d^2 / 4 to leading order (``mean_field.noise_gain``), so a certified noise is within
    # sqrt(4 tol / n_r); and a shift D of the training predictions costs ||D||^2 / (2 sigma^2) in the data term, so
    # two answers within tol of one maximum differ by ||D||^2 <= 2 tol sigma^2 there.
    codes, covariates, target, classes = _problem(21, samples=160, variants=50)
    fits = [
        fit_small_n(
            codes=codes, covariates=covariates, target=target, variant_class=classes, log_variance_offset=None, draw_count=64,
            working_bytes=_WORKING_BYTES, seed=0, inference="mean_field", array_module=module,
        )
        for module in (None, cupy)
    ]
    host, device = fits
    tolerance = 0.5 / 64
    assert host.profile["outer_criterion_met"] and device.profile["outer_criterion_met"]
    assert abs(host.profile["elbo"] - device.profile["elbo"]) <= 2 * tolerance
    residual_dimension = host.statistics.sample_count - host.statistics.covariate_rank
    assert abs(host.noise_variance - device.noise_variance) <= np.sqrt(4 * tolerance / residual_dimension) * host.noise_variance
    shift = host.statistics.projected @ (host.statistics.signs * (host.scoring.coefficients - device.scoring.coefficients))
    assert float(shift @ shift) <= 2 * tolerance * host.noise_variance


def test_the_device_pass_reaches_the_host_sweeps_fixed_point():
    # The parallel pass (``mean_field.MeanFieldFixedPoints._pass``) and the sweeps ascend the same ELBO to the same
    # certificate: the fixed points agree within the certificate's tolerance in the ELBO, and the sites' gradient
    # v (h* - h) is within it on the device as on the host.
    from sv_pgs import small_n
    from sv_pgs.mean_field import MeanFieldFixedPoints
    from sv_pgs.scale_mixture_ep import device_scope

    codes, covariates, target, classes = _problem(9, samples=160, variants=50)
    statistics = small_n.dense_statistics(codes, covariates, target)
    prior = small_n.small_n_prior(statistics, classes, np.zeros(codes.shape[1]), 64)
    start, start_noise, _moment = small_n.small_n_start(statistics, prior)
    tolerance = 0.5 / 64
    host = MeanFieldFixedPoints(statistics, prior, start_noise, 64, _WORKING_BYTES)
    (host_point,) = host([start])
    with device_scope(cupy):
        device = MeanFieldFixedPoints(statistics, prior, start_noise, 64, _WORKING_BYTES)
        (device_point,) = device([start])
    assert host_point is not None and device_point is not None
    assert device.profile["parallel_passes"] > 0 and host.profile.get("parallel_passes", 0) == 0
    assert abs(host.profile["elbo"] - device.profile["elbo"]) <= 2 * tolerance
    # A certified noise is within sqrt(4 tol / n_r) of the stationary one (test_the_device_fit_is_the_host_fit).
    assert abs(host.noise - device.noise) <= np.sqrt(4 * tolerance / host.residual_dimension) * host.noise
    shift = statistics.projected @ (device.mean - host.mean)
    assert float(shift @ shift) <= 2 * tolerance * host.noise
