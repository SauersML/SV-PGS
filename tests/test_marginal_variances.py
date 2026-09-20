"""Leave-block-out marginal variances against the dense inverse diag((Xt'Xt + diag(Pi))^-1).

The two identities (resolved/bulk elimination, and the window Woodbury) are checked exactly. The
deterministic equivalent is checked against the dense inverse on genotypes whose LD crosses every block
cut, next to the block-Jacobi variances it replaces. The certificate is checked to flag a block whose
variances are wrong and to leave the others alone.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm
from scipy.stats import t as student_t

from sv_pgs.marginal_variances import (
    BlockGrams,
    BulkSolve,
    approximation_scale,
    block_information_certificate,
    control_variate,
    stage_level,
    information_products,
    information_ceiling,
    information_solve_tolerance,
    resolvable_blocks,
    sandwich_diagonal,
    KernelFactor,
    exact_block_information,
    exact_route_is_cheaper,
    exact_bulk_diagonal,
    block_trace_certificate,
    cavity_tolerance,
    certificate_level,
    probes_to_decide,
    certificate_tolerance,
    covariance_products,
    far_field_trace,
    marginal_variances,
    marginals_from_quadratics,
    variance_jvp,
    window_bulk_quadratic,
    window_cross,
)
from sv_pgs.marginal_variances import _heavy_cut


def _genotypes(generator, sample_count: int, variant_count: int, correlation: float) -> np.ndarray:
    """Two AR(1) haplotypes per sample run along the whole region (LD crosses every cut), thresholded."""
    latent = generator.standard_normal((2, sample_count, variant_count))
    innovation = np.sqrt(1.0 - correlation**2)
    for column in range(1, variant_count):
        latent[:, :, column] = correlation * latent[:, :, column - 1] + innovation * latent[:, :, column]
    dosage = (latent > norm.isf(generator.uniform(0.05, 0.5, variant_count))).sum(axis=0).astype(np.float64)
    return (dosage - dosage.mean(axis=0)) / dosage.std(axis=0)


def _resolved(variance: np.ndarray, sample_count: int) -> np.ndarray:
    resolved = np.zeros(variance.shape[0], dtype=bool)
    while True:
        updated = sample_count * variance >= 1.0 + np.sum(variance[~resolved])
        if np.array_equal(updated, resolved):
            return np.nonzero(resolved)[0]
        resolved = updated


def _solve(columns: np.ndarray, precision: np.ndarray, resolved: np.ndarray) -> BulkSolve:
    """The refresh's outputs, formed densely."""
    sample_count = columns.shape[0]
    bulk = 1.0 / precision
    bulk[resolved] = 0.0
    kernel_inverse = np.linalg.inv(np.eye(sample_count) + (columns * bulk) @ columns.T)
    z_resolved = kernel_inverse @ columns[:, resolved]
    core = np.diag(precision[resolved]) + columns[:, resolved].T @ z_resolved
    coupling = kernel_inverse - z_resolved @ np.linalg.solve(core, z_resolved.T)
    return BulkSolve(
        site_precision=precision,
        resolved=resolved,
        resolved_core=core,
        resolved_cross=columns.T @ z_resolved,
        bulk_trace=float(np.trace(kernel_inverse)) / sample_count,
        bulk_square_trace=float(np.sum(kernel_inverse**2)) / sample_count,
        kernel_square_trace=float(np.sum(coupling**2)) / sample_count,
        sample_count=sample_count,
    )


def _grams(columns: np.ndarray, blocks: tuple[np.ndarray, ...]) -> BlockGrams:
    gram = columns.T @ columns
    return BlockGrams(
        blocks=blocks,
        within=tuple(gram[np.ix_(block, block)] for block in blocks),
        next_cross=tuple(gram[np.ix_(blocks[position], blocks[position + 1])] for position in range(len(blocks) - 1)),
    )


def test_elimination_is_exact_with_non_positive_resolved_sites():
    generator = np.random.default_rng(1)
    columns = generator.standard_normal((200, 120))
    gram = columns.T @ columns
    precision = generator.uniform(1.0, 30.0, 120)
    resolved = np.array([4, 50, 99])
    smallest = float(np.linalg.eigvalsh(gram)[0])
    precision[resolved] = [-0.5 * smallest, 0.0, 0.01]
    is_resolved = np.isin(np.arange(120), resolved)
    bulk = np.where(is_resolved, 0.0, 1.0 / np.where(is_resolved, 1.0, precision))
    kernel_inverse = np.linalg.inv(np.eye(200) + (columns * bulk) @ columns.T)
    quadratic = np.einsum("ij,ik,kj->j", columns, kernel_inverse, columns)
    z_resolved = kernel_inverse @ columns[:, resolved]
    core = np.diag(precision[resolved]) + columns[:, resolved].T @ z_resolved
    variances = marginals_from_quadratics(precision, resolved, core, columns.T @ z_resolved, quadratic)
    assert np.allclose(variances, np.diag(np.linalg.inv(gram + np.diag(precision))), rtol=1e-9)


def test_window_quadratic_is_exact_against_a_white_far_field():
    generator = np.random.default_rng(2)
    columns = generator.standard_normal((90, 150))
    variance = generator.uniform(0.001, 0.05, 150)
    variance[[7, 30]] = 0.0
    far_trace = 1.7
    kernel = np.eye(90) / far_trace + (columns * variance) @ columns.T
    exact = np.einsum("ij,ik,kj->j", columns, np.linalg.inv(kernel), columns)
    assert np.allclose(window_bulk_quadratic(columns.T @ columns, variance, far_trace), exact, rtol=1e-9)


def test_far_field_trace_is_the_increasing_root():
    generator = np.random.default_rng(3)
    eigenvalues = generator.exponential(2.0, 300)
    root = far_field_trace(0.6, 0.45, 1000, eigenvalues)
    terms = 0.45 / 1000 * eigenvalues / (1.0 + root * eigenvalues)
    residual = root - 0.6 - np.sum(terms)
    # Rounding of the residual itself: a sum of N = len(terms) + 2 terms carries at most
    # gamma_N = N u / (1 - N u) times the sum of their magnitudes (Higham, Thm 4.3).
    count = terms.shape[0] + 2
    unit = np.finfo(np.float64).eps / 2
    assert root > 0.6
    assert abs(residual) <= count * unit / (1 - count * unit) * (root + 0.6 + np.sum(terms))


def test_marginals_track_the_dense_inverse_across_cuts_and_block_jacobi_does_not():
    generator = np.random.default_rng(4)
    sample_count, variant_count, heritability = 1500, 600, 0.5
    columns = _genotypes(generator, sample_count, variant_count, 0.97) / np.sqrt(1.0 - heritability)
    variance = heritability / variant_count * np.exp(generator.normal(0.0, 1.0, variant_count))
    variance[generator.choice(variant_count, 4, replace=False)] *= 400.0
    precision = 1.0 / variance
    blocks = tuple(np.arange(start, start + 100) for start in range(0, variant_count, 100))
    solve = _solve(columns, precision, _resolved(variance, sample_count))
    grams = _grams(columns, blocks)
    exact = np.diag(np.linalg.inv(columns.T @ columns + np.diag(precision)))
    windowed = marginal_variances(solve, grams)
    jacobi = np.concatenate([np.diag(np.linalg.inv(grams.within[position] + np.diag(precision[block]))) for position, block in enumerate(blocks)])
    scale = approximation_scale(solve)
    windowed_error = np.max(np.abs(windowed - exact) / exact)
    jacobi_error = np.max(np.abs(jacobi - exact) / exact)
    assert windowed_error <= scale
    assert jacobi_error > scale


def test_certificate_flags_only_the_wrong_block():
    generator = np.random.default_rng(5)
    sample_count, variant_count, heritability = 1500, 600, 0.5
    columns = _genotypes(generator, sample_count, variant_count, 0.97) / np.sqrt(1.0 - heritability)
    precision = variant_count / heritability * np.exp(generator.normal(0.0, 1.0, variant_count))
    blocks = tuple(np.arange(start, start + 100) for start in range(0, variant_count, 100))
    solve = _solve(columns, precision, _resolved(1.0 / precision, sample_count))
    covariance = np.linalg.inv(columns.T @ columns + np.diag(precision))
    variances = marginal_variances(solve, _grams(columns, blocks))
    variances[blocks[3]] *= 1.0 + 20 * approximation_scale(solve)
    probes = generator.choice([-1.0, 1.0], size=(variant_count, 256))
    certificate = block_trace_certificate(variances, blocks, probes, covariance @ probes, approximation_scale(solve), certificate_level(64))
    assert certificate.violated.tolist() == [False, False, False, True, False, False]
    assert not certificate.certified[3]


def _strong_case(seed: int):
    generator = np.random.default_rng(seed)
    sample_count, variant_count, heritability = 1500, 600, 0.5
    columns = _genotypes(generator, sample_count, variant_count, 0.97) / np.sqrt(1.0 - heritability)
    variance = heritability / variant_count * np.exp(generator.normal(0.0, 1.0, variant_count))
    variance[generator.choice(variant_count, 4, replace=False)] *= 400.0
    precision = 1.0 / variance
    blocks = tuple(np.arange(start, start + 100) for start in range(0, variant_count, 100))
    return generator, columns, precision, blocks, _solve(columns, precision, _resolved(variance, sample_count))


def test_covariance_products_are_exact_given_the_solver_products():
    generator, columns, precision, _blocks, solve = _strong_case(6)
    bulk = 1.0 / precision
    bulk[solve.resolved] = 0.0
    kernel_inverse = np.linalg.inv(np.eye(columns.shape[0]) + (columns * bulk) @ columns.T)
    z_resolved = kernel_inverse @ columns[:, solve.resolved]
    probes = generator.choice([-1.0, 1.0], size=(columns.shape[1], 5))
    forward = columns @ (bulk[:, None] * probes)
    coupling = z_resolved.T @ forward
    back = columns.T @ (kernel_inverse @ forward - z_resolved @ np.linalg.solve(solve.resolved_core, coupling - probes[solve.resolved]))
    covariance = np.linalg.inv(columns.T @ columns + np.diag(precision))
    assert np.allclose(covariance_products(solve, probes, back, coupling), covariance @ probes, rtol=1e-8, atol=1e-12)


def test_variance_jvp_tracks_the_dense_derivative():
    generator, columns, precision, blocks, solve = _strong_case(7)
    covariance = np.linalg.inv(columns.T @ columns + np.diag(precision))
    direction = generator.uniform(0.0, 1.0, size=(columns.shape[1], 3)) * precision[:, None]
    exact = -np.einsum("jk,kr,jk->jr", covariance, direction, covariance)
    product = variance_jvp(solve, _grams(columns, blocks), direction)
    # Sums of squared entries carry at most twice the entries' relative error (d x^2 / x^2 = 2 dx / x),
    # and the B-products consume block sums.
    scale = approximation_scale(solve)
    for block in blocks:
        assert np.all(np.abs(product.values[block].sum(axis=0) - exact[block].sum(axis=0)) <= 2 * scale * np.abs(exact[block].sum(axis=0)))
    resolved = solve.resolved
    assert np.all(np.abs(product.values[resolved].sum(axis=0) - exact[resolved].sum(axis=0)) <= 2 * scale * np.abs(exact[resolved].sum(axis=0)))


def test_window_cross_matches_the_dense_maps():
    _generator, columns, precision, blocks, solve = _strong_case(12)
    grams = _grams(columns, blocks)
    dense = marginal_variances(solve, grams)
    windowed = BulkSolve(**{**solve.__dict__, "resolved_cross": window_cross(solve, grams)})
    assert np.array_equal(marginal_variances(windowed, grams), dense)


def test_certificate_tolerance_adds_the_probe_error_in_quadrature():
    _generator, _columns, _precision, _blocks, solve = _strong_case(8)
    scale = approximation_scale(solve)
    assert np.isclose(certificate_tolerance(solve, 2), scale * np.sqrt(2.0))
    assert certificate_tolerance(solve, 10**9) < scale * (1 + 1e-8)



def test_information_certificate_flags_a_cavity_error_the_trace_certificate_misses():
    generator = np.random.default_rng(13)
    sample_count, variant_count = 1500, 600
    columns = _genotypes(generator, sample_count, variant_count, 0.97)
    # Little data per variant (D_j |xt_j|^2 << 1): the variances sit next to the prior's, and a cavity
    # P_j = q_j / (1 - D_j q_j) error hides inside a tiny variance error.
    precision = variant_count / 1e-3 * np.exp(generator.normal(0.0, 1.0, variant_count))
    blocks = tuple(np.arange(start, start + 100) for start in range(0, variant_count, 100))
    solve = _solve(columns, precision, _resolved(1.0 / precision, sample_count))
    covariance = np.linalg.inv(columns.T @ columns + np.diag(precision))
    variances = np.diag(covariance).copy()
    prior = 1.0 / precision
    scale = approximation_scale(solve)
    # The premise: the injected information error moves block 3's variance trace by under half the tolerance.
    information_share = np.sum(prior[blocks[3]] - variances[blocks[3]]) / np.sum(variances[blocks[3]])
    assert 20 * scale * information_share <= 0.5 * scale
    variances[blocks[3]] = prior[blocks[3]] - (prior[blocks[3]] - variances[blocks[3]]) * (1.0 + 20 * scale)
    probes = generator.choice([-1.0, 1.0], size=(variant_count, 256))
    bulk = prior.copy()
    bulk[solve.resolved] = 0.0
    kernel_inverse = np.linalg.inv(np.eye(sample_count) + (columns * bulk) @ columns.T)
    z_resolved = kernel_inverse @ columns[:, solve.resolved]
    forward = columns @ (bulk[:, None] * probes)
    coupling = z_resolved.T @ forward
    back = columns.T @ (kernel_inverse @ forward - z_resolved @ np.linalg.solve(solve.resolved_core, coupling - probes[solve.resolved]))
    removed = information_products(solve, back)
    is_bulk = ~np.isin(np.arange(variant_count), solve.resolved)
    assert np.allclose(removed[is_bulk], (prior[:, None] * probes - covariance @ probes)[is_bulk], rtol=1e-6, atol=1e-12 * np.abs(removed).max())
    level = certificate_level(64)
    trace = block_trace_certificate(variances, blocks, probes, covariance @ probes, scale, level)
    grams = _grams(columns, blocks)
    information = block_information_certificate(solve, variances, blocks, probes, removed, scale, level, control_variate(solve, grams, probes))
    assert not trace.violated.any()
    assert information.violated.tolist() == [False, False, False, True, False, False]


def test_information_solve_tolerance_bounds_the_worst_residual():
    generator = np.random.default_rng(14)
    sample_count, variant_count = 1500, 600
    columns = _genotypes(generator, sample_count, variant_count, 0.97)
    precision = variant_count / 1e-2 * np.exp(generator.normal(0.0, 1.0, variant_count))
    blocks = tuple(np.arange(start, start + 100) for start in range(0, variant_count, 100))
    solve = _solve(columns, precision, _resolved(1.0 / precision, sample_count))
    covariance = np.linalg.inv(columns.T @ columns + np.diag(precision))
    variances = np.diag(covariance)
    scale = approximation_scale(solve)
    norms = np.sum(columns**2, axis=0)
    relative_residual = information_solve_tolerance(solve, variances, blocks, norms, scale)
    bulk = 1.0 / precision
    bulk[solve.resolved] = 0.0
    kernel = np.eye(sample_count) + (columns * bulk) @ columns.T
    # The two expectations the bound uses hold exactly: E|Xt D z|^2 = sum D^2 |xt|^2 for Rademacher z.
    gram = columns.T @ columns
    assert np.isclose(np.trace(bulk[:, None] * gram * bulk[None, :]), np.sum(bulk**2 * norms), rtol=1e-12)
    # For every block, the worst residual of the allowed size moves the probe-averaged estimate by at most
    # half the tolerance, with the probes' realized norms in place of their expectations.
    probes = generator.choice([-1.0, 1.0], size=(variant_count, 64))
    forward = columns @ (bulk[:, None] * probes)
    for members in blocks:
        information = float(np.sum(bulk[members] - variances[members]))
        direction = columns[:, members] @ (bulk[members, None] * probes[members])  # a_b per probe
        worst = np.linalg.solve(kernel, direction)
        worst *= relative_residual * np.linalg.norm(forward, axis=0) / np.linalg.norm(worst, axis=0)
        moved = np.sum(direction * np.linalg.solve(kernel, worst), axis=0)
        realized = np.mean(np.linalg.norm(direction, axis=0) * np.linalg.norm(forward, axis=0))
        expected = np.sqrt(np.sum(bulk[members] ** 2 * norms[members]) * np.sum(bulk**2 * norms))
        assert np.mean(np.abs(moved)) <= 0.5 * scale * information * realized / expected


def test_certificate_intervals_cover_at_their_level_and_probes_to_decide_decides():
    generator = np.random.default_rng(15)
    sample_count, variant_count, heritability = 1500, 600, 0.5
    columns = _genotypes(generator, sample_count, variant_count, 0.97) / np.sqrt(1.0 - heritability)
    precision = variant_count / heritability * np.exp(generator.normal(0.0, 1.0, variant_count))
    blocks = tuple(np.arange(start, start + 100) for start in range(0, variant_count, 100))
    covariance = np.linalg.inv(columns.T @ columns + np.diag(precision))
    exact = np.diag(covariance)
    level = certificate_level(64)
    # With the exact variances the true relative error is zero, so an interval misses only when it excludes 0.
    misses = 0
    trials = 200
    for _trial in range(trials):
        probes = generator.choice([-1.0, 1.0], size=(variant_count, 8))
        certificate = block_trace_certificate(exact, blocks, probes, covariance @ probes, 1.0, level)
        misses += int(np.any((certificate.lower_bound > 0.0) | (certificate.upper_bound < 0.0)))
    # The family-wise miss rate is at most the level, up to the binomial spread of 200 trials.
    assert misses <= level * trials + 3 * np.sqrt(level * trials)
    probes = generator.choice([-1.0, 1.0], size=(variant_count, 8))
    tolerance = 0.01
    undecided = block_trace_certificate(exact, blocks, probes, covariance @ probes, tolerance, level)
    needed = probes_to_decide(undecided, 8)
    probes = generator.choice([-1.0, 1.0], size=(variant_count, needed))
    decided = block_trace_certificate(exact, blocks, probes, covariance @ probes, tolerance, level)
    assert decided.certified.all()


def test_control_variate_leaves_the_information_estimate_unbiased_and_shrinks_its_spread():
    generator = np.random.default_rng(16)
    sample_count, variant_count = 1500, 600
    columns = _genotypes(generator, sample_count, variant_count, 0.97)
    precision = variant_count / 1e-2 * np.exp(generator.normal(0.0, 1.0, variant_count))
    blocks = tuple(np.arange(start, start + 100) for start in range(0, variant_count, 100))
    solve = _solve(columns, precision, _resolved(1.0 / precision, sample_count))
    grams = _grams(columns, blocks)
    covariance = np.linalg.inv(columns.T @ columns + np.diag(precision))
    prior = 1.0 / precision
    probes = generator.choice([-1.0, 1.0], size=(variant_count, 64))
    removed_exact = prior[:, None] * probes - covariance @ probes
    control = control_variate(solve, grams, probes)
    # A two-sided family-wise interval at the certificate's own level (Student t, k - 1 degrees of freedom).
    quantile = student_t.isf(0.5 * certificate_level(64) / len(blocks), probes.shape[1] - 1)
    for position, members in enumerate(blocks):
        plain = np.sum(probes[members] * removed_exact[members], axis=0)
        controlled = control.window_information[position] + np.sum(probes[members] * (removed_exact - control.removed_products)[members], axis=0)
        exact_information = float(np.sum(prior[members] - np.diag(covariance)[members]))
        # Unbiased: the controlled mean sits within its own standard error scale of the exact information.
        spread = np.std(controlled, ddof=1) / np.sqrt(probes.shape[1])
        assert abs(np.mean(controlled) - exact_information) <= quantile * spread
        assert np.std(controlled) < np.std(plain)


def test_stage_levels_spend_at_most_the_level():
    level = certificate_level(64)
    assert sum(stage_level(level, stage) for stage in range(60)) <= level


def test_cavity_tolerance_follows_its_three_bounds():
    generator = np.random.default_rng(17)
    variant_count = 300
    blocks = tuple(np.arange(start, start + 100) for start in range(0, variant_count, 100))
    site_precision = generator.uniform(10.0, 100.0, variant_count)
    data_share = generator.uniform(0.01, 0.5, variant_count)
    variances = (1.0 - data_share) / site_precision  # w = 1 - tau Sigma
    # An information error eps is the relative variance error -eps w / (1 - w).
    information = 1.0 / site_precision - variances
    eps = 0.1
    perturbed = 1.0 / site_precision - information * (1.0 + eps)
    assert np.allclose(perturbed / variances - 1.0, -eps * data_share / (1.0 - data_share), rtol=1e-12)
    draws, effective = 64, 50.0
    # A Gaussian tilted law (no response, no skewness) leaves only properness.
    zero = np.zeros(variant_count)
    assert np.all(cavity_tolerance(site_precision, variances, zero, zero, blocks, draws, effective) == 1.0)
    response = generator.uniform(0.5, 20.0, variant_count)
    skewness = generator.normal(0.0, 1.0, variant_count)
    tolerance = cavity_tolerance(site_precision, variances, response, skewness, blocks, draws, effective)
    ratio = data_share / (1.0 - data_share)
    mean_bound = np.sqrt(effective / draws) / np.sqrt(np.sum(np.square(0.5 * skewness * ratio)))
    for position, members in enumerate(blocks):
        variance_bound = np.sqrt(2.0 / draws) / np.sqrt(np.mean(np.square(response[members] * ratio[members])))
        assert np.isclose(tolerance[position], min(variance_bound, mean_bound, 1.0), rtol=1e-12)


def test_information_solve_tolerance_is_infinite_when_every_site_is_resolved():
    generator = np.random.default_rng(18)
    columns = generator.standard_normal((80, 120))
    precision = generator.uniform(1.0, 30.0, 120)
    # The case the Stage 2 sweep hit: a solver that resolved every site (e.g. all improper, or all spikes).
    solve = _solve(columns, precision, np.arange(120))
    blocks = tuple(np.arange(start, start + 40) for start in range(0, 120, 40))
    variances = np.diag(np.linalg.inv(columns.T @ columns + np.diag(precision)))
    assert information_solve_tolerance(solve, variances, blocks, np.sum(columns**2, axis=0), 0.01) == np.inf


def test_marginals_respect_the_exact_bounds():
    _generator, columns, precision, blocks, solve = _strong_case(19)
    variances = marginal_variances(solve, _grams(columns, blocks))
    upper = 1.0 / precision
    lower = 1.0 / (np.sum(columns**2, axis=0) + precision)
    bulk = ~np.isin(np.arange(precision.shape[0]), solve.resolved)
    assert np.all(variances[bulk] <= upper[bulk]) and np.all(variances[bulk] >= lower[bulk])


def test_no_upper_clamp_when_a_resolved_site_is_non_positive():
    # verify-stage2's counterexample: with Pi = (2, -1/2) the bulk site's exact marginal exceeds 1/Pi_1.
    columns = np.array([[1.0, 1.0]])
    precision = np.array([2.0, -0.5])
    exact = np.linalg.inv(columns.T @ columns + np.diag(precision))
    assert exact[0, 0] > 1.0 / precision[0]
    solve = _solve(columns, precision, np.array([1]))
    grams = BlockGrams(blocks=(np.array([0, 1]),), within=(columns.T @ columns,), next_cross=())
    variances = marginal_variances(solve, grams)
    assert variances[0] > 1.0 / precision[0]
    assert np.isclose(variances[1], exact[1, 1], rtol=1e-12)


def test_a_zero_estimate_with_probe_signal_is_violated_not_an_error():
    generator = np.random.default_rng(20)
    sample_count, variant_count = 1500, 600
    columns = _genotypes(generator, sample_count, variant_count, 0.97)
    precision = variant_count / 1e-2 * np.exp(generator.normal(0.0, 1.0, variant_count))
    blocks = tuple(np.arange(start, start + 100) for start in range(0, variant_count, 100))
    covariance = np.linalg.inv(columns.T @ columns + np.diag(precision))
    variances = np.diag(covariance).copy()
    variances[blocks[2]] = 1.0 / precision[blocks[2]]  # pinned to the prior: zero information estimated
    probes = generator.choice([-1.0, 1.0], size=(variant_count, 64))
    removed = probes / precision[:, None] - covariance @ probes
    solve = _solve(columns, precision, _resolved(1.0 / precision, sample_count))
    removed_estimate = np.where(np.isin(np.arange(variant_count), solve.resolved), 0.0, 1.0 / precision - variances)
    certificate = block_trace_certificate(removed_estimate, blocks, probes, removed, 0.5, certificate_level(64))
    assert certificate.violated[2] and not certificate.certified[2]
    assert np.isinf(certificate.relative_error[2])


def test_information_solve_tolerance_stays_finite_with_non_positive_estimates():
    # verify-stage2's case: a model whose estimated block information is <= 0 everywhere still has bulk mass,
    # so the probe solve must run (a +inf tolerance skipped it).
    generator = np.random.default_rng(21)
    columns = generator.standard_normal((200, 120))
    precision = generator.uniform(1.0, 30.0, 120)
    blocks = tuple(np.arange(start, start + 40) for start in range(0, 120, 40))
    solve = _solve(columns, precision, np.array([5]))
    norms = np.sum(columns**2, axis=0)
    too_large = 1.0 / precision * 1.5  # every bulk marginal above D: negative estimates
    zero = 1.0 / precision  # every bulk marginal at D: zero estimates
    for variances in (too_large, zero):
        tolerance = information_solve_tolerance(solve, variances, blocks, norms, 0.01)
        assert np.isfinite(tolerance) and tolerance > 0.0


def test_a_block_of_rounding_level_columns_is_exact_not_a_zero_tolerance():
    # verify-stage2's case: a fold whose training rows make a block's columns zero up to rounding.
    generator = np.random.default_rng(22)
    columns = generator.standard_normal((200, 120))
    columns[:, 40:80] = 1e-30 * generator.standard_normal((200, 40))
    precision = generator.uniform(1.0, 30.0, 120)
    blocks = tuple(np.arange(start, start + 40) for start in range(0, 120, 40))
    solve = _solve(columns, precision, np.array([5]))
    norms = np.sum(columns**2, axis=0)
    assert information_ceiling(solve, blocks, norms)[1] == 0.0
    assert resolvable_blocks(solve, blocks, information_ceiling(solve, blocks, norms)).tolist() == [True, False, True]
    variances = np.diag(np.linalg.inv(columns.T @ columns + np.diag(precision)))
    tolerance = information_solve_tolerance(solve, variances, blocks, norms, 0.01)
    assert np.isfinite(tolerance) and tolerance > 0.0
    probes = generator.choice([-1.0, 1.0], size=(120, 16))
    covariance = np.linalg.inv(columns.T @ columns + np.diag(precision))
    grams = BlockGrams(blocks=blocks, within=tuple((columns.T @ columns)[np.ix_(b, b)] for b in blocks), next_cross=())
    removed = probes / precision[:, None] - covariance @ probes
    certificate = block_information_certificate(solve, variances, blocks, probes, removed, 0.5, certificate_level(64), control_variate(solve, grams, probes))
    assert certificate.certified[1]


def test_sandwich_diagonal_equals_the_three_operand_contraction():
    generator = np.random.default_rng(23)
    covariance = generator.standard_normal((40, 40))
    gram = generator.standard_normal((40, 40))
    reference = np.einsum("ij,jk,ki->i", covariance, gram, covariance)
    assert np.allclose(sandwich_diagonal(covariance, gram), reference, rtol=1e-12, atol=1e-12 * np.abs(reference).max())


def _low_rank_block(generator, size: int, rank: int) -> np.ndarray:
    """A PSD block whose trace is spread evenly over ``rank`` random orthonormal directions (effective rank ``rank``)."""
    basis = np.linalg.qr(generator.standard_normal((size, rank)))[0]
    return basis @ basis.T


@pytest.mark.parametrize("rank", [1, 2, 20])
def test_certificate_keeps_its_level_on_low_effective_rank_blocks(rank):
    # review-stats: strong LD makes a block's probe values z'Az skewed (close to tr chi2_r / r), and the plain t cut
    # then missed on the heavy side up to 9.5x its level. With the exact variances the true relative error is zero and
    # the tolerance is zero, so a block is violated exactly when its interval misses: the family-wise miss rate must
    # stay within the binomial spread of the level.
    generator = np.random.default_rng(100 + rank)
    size, block_count, probe_count = 24, 4, 16
    level = certificate_level(8)
    blocks = tuple(np.arange(start, start + size) for start in range(0, size * block_count, size))
    covariance = np.zeros((size * block_count, size * block_count))
    for members in blocks:
        covariance[np.ix_(members, members)] = _low_rank_block(generator, size, rank)
    variances = np.diag(covariance).copy()
    trials = int(np.ceil(300 / level))
    misses = 0
    for _trial in range(trials):
        probes = generator.choice([-1.0, 1.0], size=(size * block_count, probe_count))
        certificate = block_trace_certificate(variances, blocks, probes, covariance @ probes, 0.0, level)
        assert not certificate.certified.any()
        misses += int(certificate.violated.any())
    assert misses <= level * trials + 3 * np.sqrt(level * trials)


def test_heavy_cut_is_the_t_quantile_for_symmetric_values_and_moves_out_with_skewness():
    side, probe_count = 1e-3, 16
    quantile = float(student_t.isf(side, probe_count - 1))
    assert _heavy_cut(0.0, probe_count, side, quantile) == (quantile, True)
    cut, usable = _heavy_cut(0.5, probe_count, side, quantile)
    assert cut > quantile and usable
    # A skewness large enough that the one-term correction is not below the tail it corrects leaves the block undecided.
    assert not _heavy_cut(10.0, probe_count, side, quantile)[1]


def _kernel_factor(columns: np.ndarray, precision: np.ndarray, resolved: np.ndarray) -> KernelFactor:
    bulk = 1.0 / precision
    bulk[resolved] = 0.0
    kernel = np.eye(columns.shape[0]) + (columns * bulk) @ columns.T
    solves = np.linalg.solve(kernel, columns[:, resolved])
    return KernelFactor(lower=np.linalg.cholesky(kernel), resolved_solves=solves,
                        resolved_core=np.diag(precision[resolved]) + columns[:, resolved].T @ solves)


def test_exact_dual_route_matches_the_dense_inverse_with_a_non_positive_site():
    generator = np.random.default_rng(24)
    columns = generator.standard_normal((60, 150))
    precision = generator.uniform(1.0, 30.0, 150)
    precision[[4, 90]] = [-0.5 * float(np.linalg.eigvalsh(columns.T @ columns)[0]), 0.01]
    resolved = np.array([4, 90])
    factor = _kernel_factor(columns, precision, resolved)
    covariance = np.linalg.inv(columns.T @ columns + np.diag(precision))
    bulk = 1.0 / precision
    bulk[resolved] = 0.0
    block = np.arange(30, 80)
    removed = exact_block_information(factor, bulk[block], columns[:, block])
    expected = np.where(np.isin(block, resolved), 0.0, bulk[block] - np.diag(covariance)[block])
    assert np.allclose(np.where(np.isin(block, resolved), 0.0, removed), expected, rtol=1e-8, atol=1e-14)
    kernel = np.eye(columns.shape[0]) + (columns * bulk) @ columns.T
    assert np.allclose(exact_bulk_diagonal(factor), np.diag(np.linalg.inv(kernel)), rtol=1e-8)
    # With the resolved correction added once, 1 - Q_ii is the exact leverage h_i = xt_i' Sigma xt_i.
    resolved_term = np.sum((factor.resolved_solves @ np.linalg.inv(factor.resolved_core)) * factor.resolved_solves, axis=1)
    leverage = 1.0 - (exact_bulk_diagonal(factor) - resolved_term)
    assert np.allclose(leverage, np.einsum("ij,jk,ik->i", columns, covariance, columns), rtol=1e-8)


def test_exact_route_rule_prefers_the_dual_when_samples_are_few():
    blocks = (np.arange(4430),)
    grams = BlockGrams(blocks=blocks, within=(np.eye(4430),), next_cross=())
    assert exact_route_is_cheaper(580, grams, 2**30)
    assert not exact_route_is_cheaper(50_000, grams, 2**30)


def test_kernel_factor_inverts_its_core_once():
    generator = np.random.default_rng(25)
    columns = generator.standard_normal((40, 60))
    precision = generator.uniform(1.0, 30.0, 60)
    factor = _kernel_factor(columns, precision, np.array([3, 7]))
    assert factor.core_inverse is factor.core_inverse
    assert np.allclose(factor.core_inverse @ factor.resolved_core, np.eye(2), atol=1e-12)
