"""Exact block quadratics against the dense posterior covariance.

On a block's bulk sites identity 1 gives Sigma_bb = D_b - D_b M_b D_b, so M_b = D_b^-1 (D_b - Sigma_bb) D_b^-1
exactly; every returned entry must lie within its certified bound of that, up to float64 rounding through the
dense inverse (dimension * kappa * eps, as in test_dual_solve).
"""

from __future__ import annotations

import numpy as np

from sv_pgs import dual_solve
from sv_pgs.exact_quadratics import exact_block_quadratics
from sv_pgs.marginal_variances import BlockGrams

EPS = np.finfo(np.float64).eps


def _problem(seed: int, duplicated: int = 0):
    """One model on a fold mask, LD running across the block cuts, two strong sites and one negative site."""
    rng = np.random.default_rng(seed)
    sample_count, variant_count, block = 60, 90, 15
    raw = rng.standard_normal((sample_count, variant_count))
    for column in range(1, variant_count):
        raw[:, column] = 0.8 * raw[:, column - 1] + 0.6 * raw[:, column]
    genotypes = (raw - raw.mean(0)) / raw.std(0)
    for target in range(duplicated):
        genotypes[:, block + 2 * target + 1] = genotypes[:, block + 2 * target]
    bounds = [(start, start + block) for start in range(0, variant_count, block)]
    covariates = np.column_stack([np.ones(sample_count), rng.standard_normal(sample_count)])
    training = np.ones((sample_count, 1))
    training[rng.permutation(sample_count)[: sample_count // 4], 0] = 0.0
    noise = np.ones(1)
    # D ||xt||^2 of order 1: few sites stand out as spikes besides the two strong ones
    variances = np.exp(rng.normal(0.0, 1.0, (variant_count, 1))) / sample_count
    variances[[3, 40], 0] *= 50.0
    precision = 1.0 / variances
    design = _design(genotypes, covariates, training)
    held = precision[:, 0].copy()
    held[70] = 0.0
    limit = 1.0 / np.linalg.inv(design.T @ design + np.diag(held))[70, 70]
    precision[70, 0] = -0.5 * limit
    shift = rng.standard_normal((variant_count, 1)) * np.sqrt(np.abs(precision))
    response = rng.standard_normal((sample_count, 1))
    return genotypes, bounds, covariates, training, noise, precision, shift, response


def _design(genotypes, covariates, training):
    root = np.sqrt(training[:, 0])
    weighted = root[:, None] * covariates
    projector = np.eye(genotypes.shape[0]) - weighted @ np.linalg.pinv(weighted)
    return projector @ (root[:, None] * genotypes)


def _grams(design, bounds):
    blocks = tuple(np.arange(start, stop) for start, stop in bounds)
    within = tuple(design[:, members].T @ design[:, members] for members in blocks)
    cross = tuple(design[:, blocks[index]].T @ design[:, blocks[index + 1]] for index in range(len(blocks) - 1))
    return BlockGrams(blocks=blocks, within=within, next_cross=cross)


def _fit(seed: int, bound: float, duplicated: int = 0):
    genotypes, bounds, covariates, training, noise, precision, shift, response = _problem(seed, duplicated)
    design = _design(genotypes, covariates, training)
    grams = _grams(design, bounds)
    gaussian = dual_solve.DualGaussian(source=dual_solve.DenseDualSource(genotypes, bounds), training=training, targets=response, offsets=np.zeros_like(response),
                                       covariates=covariates, grams=grams, probe_count=2, seed=seed)
    gaussian.iterate(site_precision=precision, site_shift=shift, noise_variance=noise, error_bound=np.full(1, bound), probe_residual_ratio=bound)
    posterior_precision = design.T @ design + np.diag(precision[:, 0])
    return gaussian, grams, posterior_precision


def _exact(gaussian, posterior_precision, members):
    solve = gaussian.bulk_solves[0]
    bulk = members[~np.isin(members, solve.resolved)]
    variance = 1.0 / solve.site_precision[bulk]
    covariance = np.linalg.inv(posterior_precision)[np.ix_(bulk, bulk)]
    exact = (np.diag(variance) - covariance) / np.outer(variance, variance)
    rounding = posterior_precision.shape[0] * np.linalg.cond(posterior_precision) * EPS * np.abs(exact).max()
    return np.searchsorted(members, bulk), exact, rounding


def test_exact_block_quadratics_are_the_dense_quadratics_within_their_bounds() -> None:
    gaussian, grams, posterior_precision = _fit(71, np.sqrt(EPS))
    assert gaussian.bulk_solves[0].resolved.size
    relative_error = EPS ** 0.25
    for result in exact_block_quadratics(gaussian, 0, grams, [0, 2, 4], relative_error):
        positions, exact, rounding = _exact(gaussian, posterior_precision, result.sites)
        computed = result.quadratic[np.ix_(positions, positions)]
        assert np.all(np.abs(computed - exact) <= result.error[np.ix_(positions, positions)] + rounding)
        assert result.met
        diagonal = np.diag(computed)
        assert np.all(np.diag(result.error[np.ix_(positions, positions)]) <= relative_error * np.abs(diagonal))


def test_the_bounds_cover_a_loose_refresh_s_resolved_block() -> None:
    # A refresh at a quarter of float64's digits leaves Z_L that far off; the bounds must still hold.
    gaussian, grams, posterior_precision = _fit(73, EPS ** 0.25)
    for result in exact_block_quadratics(gaussian, 0, grams, [0, 2, 4], np.sqrt(EPS)):
        positions, exact, rounding = _exact(gaussian, posterior_precision, result.sites)
        assert np.all(np.abs(result.quadratic[np.ix_(positions, positions)] - exact) <= result.error[np.ix_(positions, positions)] + rounding)


def test_a_rank_deficient_block_takes_its_eigenvectors() -> None:
    # Three duplicated column pairs in block 1: its bulk Gram loses one rank per pair left in the bulk, so it
    # takes fewer directions than columns.
    gaussian, grams, posterior_precision = _fit(75, np.sqrt(EPS), duplicated=3)
    [result] = exact_block_quadratics(gaussian, 0, grams, [1], EPS ** 0.25)
    resolved = gaussian.bulk_solves[0].resolved
    bulk = np.setdiff1d(result.sites, resolved)
    pairs = [(15 + 2 * index, 16 + 2 * index) for index in range(3)]
    deficiency = sum(first not in resolved and second not in resolved for first, second in pairs)
    assert deficiency
    assert result.columns == bulk.size - deficiency
    positions, exact, rounding = _exact(gaussian, posterior_precision, result.sites)
    assert np.all(np.abs(result.quadratic[np.ix_(positions, positions)] - exact) <= result.error[np.ix_(positions, positions)] + rounding)
