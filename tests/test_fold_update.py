"""Checks of the fixed-site fold identities and the honest nested plan (synthetic data; math checks only)."""
import numpy as np
from scipy import linalg, optimize

from sv_pgs.fold_update import (
    FixedSites,
    check_honest,
    leave_blocks_out,
    nested_fold_plan,
    rescale_sites,
    run_nested,
)
from sv_pgs.fold_update import _kernel_leave_blocks_out, _primal_leave_blocks_out


def _problem(seed, samples=40, variants=60, negative=0):
    generator = np.random.default_rng(seed)
    genotypes = generator.binomial(2, 0.3, size=(samples, variants)).astype(np.float64)
    covariates = np.column_stack([np.ones(samples), generator.normal(size=samples)])
    target = covariates @ np.array([0.5, -0.2]) + genotypes[:, :3] @ np.array([0.4, -0.3, 0.2]) + generator.normal(size=samples)
    precision = generator.uniform(0.5, 3.0, size=variants)
    precision[:negative] = -generator.uniform(0.01, 0.05, size=negative)
    shift = generator.normal(scale=0.1, size=variants)
    blocks = np.array_split(generator.permutation(samples), 4)
    return genotypes, covariates, target, FixedSites(precision=precision, shift=shift, noise=0.8), blocks


def _refit(genotypes, covariates, target, sites, rows):
    """The fixed-site Gaussian fitted on the other rows only, then predicting ``rows``."""
    rest = np.setdiff1d(np.arange(genotypes.shape[0]), rows)
    design = np.hstack([covariates, genotypes])
    count = covariates.shape[1]
    precision = design[rest].T @ design[rest] / sites.noise + np.diag(np.concatenate([np.zeros(count), sites.precision]))
    information = design[rest].T @ target[rest] / sites.noise + np.concatenate([np.zeros(count), sites.shift])
    covariance = linalg.inv(precision)
    coefficients = covariance @ information
    prediction = design[rows] @ coefficients
    variance = sites.noise + np.einsum("ij,jk,ik->i", design[rows], covariance, design[rows])
    return prediction, variance, coefficients[count:]


def test_kernel_and_joint_forms_equal_a_refit_on_the_other_rows():
    genotypes, covariates, target, sites, blocks = _problem(1)
    for form in (_kernel_leave_blocks_out, _primal_leave_blocks_out):
        for result in form(genotypes, covariates, target, sites, blocks):
            prediction, variance, mean = _refit(genotypes, covariates, target, sites, result.rows)
            np.testing.assert_allclose(result.prediction, prediction, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(result.variance, variance, rtol=1e-9)
            np.testing.assert_allclose(result.mean, mean, rtol=1e-8, atol=1e-10)


def test_joint_form_is_exact_with_negative_sites():
    genotypes, covariates, target, sites, blocks = _problem(2, negative=4)
    results = leave_blocks_out(genotypes, covariates, target, sites, blocks)
    for result in results:
        prediction, variance, mean = _refit(genotypes, covariates, target, sites, result.rows)
        np.testing.assert_allclose(result.prediction, prediction, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(result.variance, variance, rtol=1e-9)
        np.testing.assert_allclose(result.mean, mean, rtol=1e-8, atol=1e-10)


def test_rescaled_sites_are_the_same_gaussian_on_raw_coefficients():
    genotypes, covariates, target, sites, blocks = _problem(3)
    old_scale = genotypes.std(axis=0) + 0.1
    new_scale = old_scale * np.random.default_rng(4).uniform(0.5, 2.0, size=old_scale.shape)
    precision, shift = rescale_sites(sites.precision, sites.shift, old_scale, new_scale)
    old = leave_blocks_out(genotypes / old_scale, covariates, target, sites, blocks)
    new = leave_blocks_out(genotypes / new_scale, covariates, target, FixedSites(precision, shift, sites.noise), blocks)
    for first, second in zip(old, new):
        np.testing.assert_allclose(first.prediction, second.prediction, rtol=1e-9, atol=1e-9)
        # Raw coefficients beta = b / s agree.
        np.testing.assert_allclose(first.mean / old_scale, second.mean / new_scale, rtol=1e-8, atol=1e-10)


def test_nested_plan_is_honest_and_ends_in_each_folds_training_rows():
    generator = np.random.default_rng(5)
    samples = 53
    for fold_count in (2, 5, 10):
        held_out = np.array_split(generator.permutation(samples), fold_count)
        plan = nested_fold_plan(held_out, samples)
        check_honest(plan, held_out, samples)
        leaves = [node for node in plan if len(node.folds) == 1]
        assert sorted(node.folds[0] for node in leaves) == list(range(fold_count))
        assert len(plan) == 2 * fold_count - 1
        # The root fits the rows every fold trains on: none, when the held-out blocks cover the samples.
        assert plan[0].rows.size == 0


def test_run_nested_never_shows_a_fit_a_held_out_row():
    generator = np.random.default_rng(6)
    samples, fold_count = 40, 5
    held_out = np.array_split(generator.permutation(samples), fold_count)
    plan = nested_fold_plan(held_out, samples)
    calls = []

    def fit(rows, warm):
        calls.append(rows.copy())
        return {"rows": rows.copy(), "warm_rows": None if warm is None else warm["rows"]}

    leaves = run_nested(plan, fit)
    for fold, result in leaves.items():
        assert np.array_equal(result["rows"], np.setdiff1d(np.arange(samples), held_out[fold]))
        if result["warm_rows"] is not None:
            # Its warm start was fitted on a subset of its own training rows.
            assert not np.intersect1d(result["warm_rows"], held_out[fold]).size
            assert not np.setdiff1d(result["warm_rows"], result["rows"]).size
    assert len(calls) == 2 * fold_count - 2


def _reml(log_parameters, genotypes, covariates, target):
    """-log restricted likelihood of y ~ N(C alpha, sigma^2 I + s^2 G G') with alpha flat (ridge EB's evidence)."""
    genetic, noise = np.exp(log_parameters)
    kernel = noise * np.eye(genotypes.shape[0]) + genetic * genotypes @ genotypes.T
    factor = linalg.cho_factor(kernel)
    inverse_covariates = linalg.cho_solve(factor, covariates)
    information = covariates.T @ inverse_covariates
    inverse_target = linalg.cho_solve(factor, target)
    alpha = linalg.solve(information, covariates.T @ inverse_target)
    residual = target - covariates @ alpha
    quadratic = residual @ linalg.cho_solve(factor, residual)
    log_det = 2.0 * np.sum(np.log(np.diag(factor[0])))
    return 0.5 * (log_det + np.linalg.slogdet(information)[1] + quadratic)


def _eb_fit(genotypes, covariates, target, rows, start):
    result = optimize.minimize(_reml, start, args=(genotypes[rows], covariates[rows], target[rows]), method="BFGS",
                               options={"gtol": 1e-10})
    return result.x, result.nfev


def test_honest_warm_starts_reach_the_independent_fold_optimum_and_full_data_alo_does_not():
    generator = np.random.default_rng(7)
    samples, variants, fold_count = 150, 40, 5
    genotypes = generator.binomial(2, 0.3, size=(samples, variants)).astype(np.float64)
    genotypes -= genotypes.mean(axis=0)
    covariates = np.ones((samples, 1))
    target = genotypes @ generator.normal(scale=0.15, size=variants) + generator.normal(size=samples)
    held_out = np.array_split(generator.permutation(samples), fold_count)
    cold_start = np.zeros(2)
    independent, cold_evaluations = {}, 0
    for fold in range(fold_count):
        rows = np.setdiff1d(np.arange(samples), held_out[fold])
        independent[fold], evaluations = _eb_fit(genotypes, covariates, target, rows, cold_start)
        cold_evaluations += evaluations
    warm_evaluations = [0]

    def fit(rows, warm):
        start = cold_start if warm is None else warm
        parameters, evaluations = _eb_fit(genotypes, covariates, target, rows, start)
        warm_evaluations[0] += evaluations
        return parameters

    nested = run_nested(nested_fold_plan(held_out, samples), fit)
    for fold in range(fold_count):
        np.testing.assert_allclose(nested[fold], independent[fold], atol=1e-6)
    # The leak: fixed-site leave-out at the FULL-data EB optimum is not the fold's own fit.
    full, _ = _eb_fit(genotypes, covariates, target, np.arange(samples), cold_start)
    genetic, noise = np.exp(full)
    sites = FixedSites(precision=np.full(variants, 1.0 / genetic), shift=np.zeros(variants), noise=noise)
    approximate = leave_blocks_out(genotypes, covariates, target, sites, held_out)
    gaps = []
    for fold, result in enumerate(approximate):
        fold_genetic, fold_noise = np.exp(independent[fold])
        fold_sites = FixedSites(precision=np.full(variants, 1.0 / fold_genetic), shift=np.zeros(variants), noise=fold_noise)
        honest = _refit(genotypes, covariates, target, fold_sites, held_out[fold])[0]
        gaps.append(np.max(np.abs(result.prediction - honest)))
    assert max(gaps) > 1e-6
