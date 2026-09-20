"""The dual E-step on a CUDA device gives the host's answer."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs import dual_solve
from sv_pgs.compute_budget import _try_import_cupy
from tests.test_dual_solve import EPS, MODEL_COUNT, _CodeTileSource, _coded_problem, _grams, _problem, _solve_bound

cupy = _try_import_cupy()
pytestmark = pytest.mark.skipif(cupy is None, reason="needs a CUDA device")


def test_cuda_certified_solve_with_deflation_matches_the_host() -> None:
    genotypes, bounds, covariates, weights, variances, prior_mean, response = _problem(31)
    results = []
    for array_module in (np, cupy):
        source = dual_solve.DenseDualSource(array_module.asarray(genotypes), bounds, array_module)
        models = dual_solve.DualModels(array_module.asarray(weights), array_module.asarray(variances), array_module.asarray(covariates), array_module)
        right = dual_solve.mean_right_hand_side(models, array_module.asarray(response), array_module.asarray(genotypes @ prior_mean))
        count = dual_solve.PassCount()
        deflation, _resolved = dual_solve.spike_deflation(source, models, count)
        bound = array_module.asarray(_solve_bound(np.asarray(right.get() if hasattr(right, "get") else right)))
        result = dual_solve.certified_block_cg(source, models, right, array_module.zeros_like(right), array_module.arange(MODEL_COUNT), bound, count, deflation=deflation)
        solution = result.solution.get() if hasattr(result.solution, "get") else result.solution
        results.append((solution, np.asarray(bound.get() if hasattr(bound, "get") else bound)))
    (host, host_bound), (device, _device_bound) = results
    # Both are certified to their bound, so they agree within twice the bound per column.
    assert np.all(np.linalg.norm(host - device, axis=0) <= 2.0 * host_bound * (1.0 + genotypes.shape[0] * EPS))


def test_cuda_code_tiles_with_relaxed_digits_keep_the_certificate() -> None:
    standardized, codes, means, scales, bounds, covariates, weights, variances, prior_mean, response = _coded_problem(43)
    source = dual_solve.StreamedDualSource(_CodeTileSource(codes, means, scales, bounds, cupy, 1 << 28))
    models = dual_solve.DualModels(cupy.asarray(weights), cupy.asarray(variances), cupy.asarray(covariates), cupy)
    right = dual_solve.mean_right_hand_side(models, cupy.asarray(response), cupy.asarray(standardized @ prior_mean))
    count = dual_solve.PassCount()
    deflation, _resolved = dual_solve.spike_deflation(source, models, count)
    host_right = cupy.asnumpy(right)
    bound = cupy.asarray(_solve_bound(host_right))
    result = dual_solve.certified_block_cg(source, models, right, cupy.zeros_like(right), cupy.arange(MODEL_COUNT), bound, count, deflation=deflation)
    assert bool(cupy.all(result.residual_norm <= bound))
    assert any(relative_error > 0.0 for relative_error in result.relative_errors)
    dense_models = dual_solve.DualModels(weights, variances, covariates)
    exact = host_right - np.column_stack([
        _dense_operator(standardized, covariates, weights, variances, model) @ cupy.asnumpy(result.solution[:, model]) for model in range(MODEL_COUNT)
    ])
    assert np.all(np.linalg.norm(exact, axis=0) <= cupy.asnumpy(bound) * (1.0 + standardized.shape[0] * EPS) + standardized.shape[0] * EPS * np.linalg.norm(host_right, axis=0))
    del dense_models


def _dense_operator(genotypes, covariates, weights, variances, model):
    from tests.test_dual_solve import _dense

    return _dense(genotypes, covariates, weights, variances, model)[3]


def test_cuda_exact_marginals_match_the_host() -> None:
    from tests.test_dual_solve import _gaussian_problem

    genotypes, bounds, covariates, training, noise, precision, shift, response, offsets, _negative = _gaussian_problem(82)
    outputs = []
    for array_module in (np, cupy):
        source = dual_solve.DenseDualSource(array_module.asarray(genotypes), bounds, array_module)
        gaussian = dual_solve.DualGaussian(
            source=source, training=array_module.asarray(training), targets=array_module.asarray(response), offsets=array_module.asarray(offsets),
            covariates=array_module.asarray(covariates), grams=_grams(bounds, True), probe_count=2, seed=14,
        )
        gaussian.iterate(site_precision=array_module.asarray(precision), site_shift=array_module.asarray(shift), noise_variance=noise,
                         error_bound=np.full(MODEL_COUNT, np.sqrt(EPS)), probe_residual_ratio=np.sqrt(EPS))
        outputs.append(gaussian.exact_marginals(0, gaussian.kernel_factor(0)))
    np.testing.assert_allclose(outputs[1], outputs[0], rtol=genotypes.shape[0] * np.sqrt(EPS), atol=0.0)


def test_cuda_dual_gaussian_matches_the_host() -> None:
    from tests.test_dual_solve import _gaussian_problem

    genotypes, bounds, covariates, training, noise, precision, shift, response, offsets, _negative = _gaussian_problem(55)
    means = []
    for array_module in (np, cupy):
        source = dual_solve.DenseDualSource(array_module.asarray(genotypes), bounds, array_module)
        gaussian = dual_solve.DualGaussian(
            source=source, training=array_module.asarray(training), targets=array_module.asarray(response), offsets=array_module.asarray(offsets),
            covariates=array_module.asarray(covariates), grams=_grams(bounds, True), probe_count=2, seed=6,
        )
        certificate = gaussian.iterate(
            site_precision=array_module.asarray(precision), site_shift=array_module.asarray(shift), noise_variance=noise,
            error_bound=np.full(MODEL_COUNT, np.sqrt(EPS)), probe_residual_ratio=np.sqrt(EPS),
        )
        assert bool(np.all(np.asarray(certificate.error_bound.get() if hasattr(certificate.error_bound, "get") else certificate.error_bound) <= np.sqrt(EPS)))
        means.append(np.asarray(gaussian.mean.get() if hasattr(gaussian.mean, "get") else gaussian.mean))
    from tests.test_dual_solve import _dense_gaussian

    for model in range(MODEL_COUNT):
        posterior_precision = _dense_gaussian(genotypes, covariates, training, noise, precision, shift, response, offsets, model)[0]
        difference = means[0][:, model] - means[1][:, model]
        # Both are within sqrt(eps) of the exact mean in the A-norm.
        assert np.sqrt(float(difference @ posterior_precision @ difference)) <= 2.0 * np.sqrt(EPS) * (1.0 + np.linalg.cond(posterior_precision) * genotypes.shape[1] * EPS)


def test_cuda_posterior_and_information_solves_match_the_host() -> None:
    from tests.test_dual_solve import _gaussian_problem

    genotypes, bounds, covariates, training, noise, precision, shift, response, offsets, _negative = _gaussian_problem(61)
    rng = np.random.default_rng(62)
    right = rng.standard_normal((genotypes.shape[1], 3))
    outputs = []
    for array_module in (np, cupy):
        source = dual_solve.DenseDualSource(array_module.asarray(genotypes), bounds, array_module)
        gaussian = dual_solve.DualGaussian(
            source=source, training=array_module.asarray(training), targets=array_module.asarray(response), offsets=array_module.asarray(offsets),
            covariates=array_module.asarray(covariates), grams=_grams(bounds, True), probe_count=2, seed=9,
        )
        gaussian.iterate(site_precision=array_module.asarray(precision), site_shift=array_module.asarray(shift), noise_variance=noise,
                         error_bound=np.full(MODEL_COUNT, np.sqrt(EPS)), probe_residual_ratio=np.sqrt(EPS))
        solved = gaussian.posterior_solve(array_module.asarray(right), 0, np.sqrt(EPS))
        back, coupling, _norms = gaussian.information_solve(array_module.asarray(right), 0, np.sqrt(EPS))
        outputs.append([np.asarray(item.get() if hasattr(item, "get") else item) for item in (solved, back, coupling)])
    for host, device in zip(*outputs):
        scale = max(float(np.abs(host).max()), np.finfo(np.float64).tiny)
        # Both runs are certified to sqrt(eps); their difference is at most twice that, times the conditioning seen.
        assert float(np.abs(host - device).max()) <= np.sqrt(EPS) * scale * genotypes.shape[0]


def test_cuda_rank_losing_covariates_and_tied_spikes_match_the_host() -> None:
    from tests.test_dual_solve import _rank_losing_problem

    genotypes, bounds, covariates, weights, variances, prior_mean, response = _rank_losing_problem(33)
    results = []
    for array_module in (np, cupy):
        source = dual_solve.DenseDualSource(array_module.asarray(genotypes), bounds, array_module)
        models = dual_solve.DualModels(array_module.asarray(weights), array_module.asarray(variances), array_module.asarray(covariates), array_module)
        right = dual_solve.mean_right_hand_side(models, array_module.asarray(response), array_module.asarray(genotypes @ prior_mean))
        count = dual_solve.PassCount()
        deflation, resolved = dual_solve.spike_deflation(source, models, count)
        assert resolved[1] >= 2
        host_right = np.asarray(right.get() if hasattr(right, "get") else right)
        assert np.all(np.isfinite(host_right))
        bound = array_module.asarray(_solve_bound(host_right))
        result = dual_solve.certified_block_cg(source, models, right, array_module.zeros_like(right), array_module.arange(MODEL_COUNT), bound, count, deflation=deflation)
        solution = np.asarray(result.solution.get() if hasattr(result.solution, "get") else result.solution)
        assert np.all(np.isfinite(solution))
        results.append((solution, np.asarray(bound.get() if hasattr(bound, "get") else bound)))
    (host, host_bound), (device, _device_bound) = results
    assert np.all(np.linalg.norm(host - device, axis=0) <= 2.0 * host_bound * (1.0 + genotypes.shape[0] * EPS))
