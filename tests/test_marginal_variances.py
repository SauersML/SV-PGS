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

from sv_pgs import marginal_variances as marginal_variances_module
from sv_pgs.marginal_variances import (
    BlockCertificate,
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
    block_covariance,
    block_trace_certificate,
    prepare_windows,
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
import sv_pgs.marginal_variances as marginal_variances_module
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


@pytest.mark.parametrize(("rank", "probe_count"), [(1, 16), (2, 16), (20, 16), (20, 64)])
def test_certificate_keeps_its_level_on_low_effective_rank_blocks(rank, probe_count):
    # review-stats: strong LD makes a block's probe values z'Az skewed (close to tr chi2_r / r), and the plain t cut
    # then missed on the heavy side up to 9.5x its level. With the exact variances the true relative error is zero and
    # the tolerance is zero, so a block is violated exactly when its interval misses: the family-wise miss rate must
    # stay within the binomial spread of the level.
    # 400 families give the test power against the plain t cut, whose family-wise miss rate at rank 1 is ~3x the level.
    generator = np.random.default_rng(100 + rank + probe_count)
    size, block_count = 24, 4
    level = certificate_level(8)
    blocks = tuple(np.arange(start, start + size) for start in range(0, size * block_count, size))
    covariance = np.zeros((size * block_count, size * block_count))
    for members in blocks:
        covariance[np.ix_(members, members)] = _low_rank_block(generator, size, rank)
    variances = np.diag(covariance).copy()
    trials = 400
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
    cut, usable = _heavy_cut(0.05, probe_count, side, quantile)
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


def _rademacher_skewness(matrix: np.ndarray) -> float:
    """The exact skewness of z'Az for Rademacher z: variance 2 (||A||_F^2 - sum a_ii^2) and third cumulant
    8 (tr A^3 - 3 sum_i a_ii (A^2)_ii + 2 sum_i a_ii^3), the triangle terms of the off-diagonal chaos."""
    diagonal = np.diag(matrix)
    square = matrix @ matrix
    variance = 2.0 * (float(np.sum(matrix * matrix)) - float(np.sum(diagonal**2)))
    third = 8.0 * (float(np.trace(square @ matrix)) - 3.0 * float(np.sum(diagonal * np.diag(square))) + 2.0 * float(np.sum(diagonal**3)))
    return third / variance**1.5


@pytest.mark.parametrize("rank", [2, 20])
def test_certificate_keeps_its_level_with_structural_skewness(rank):
    generator = np.random.default_rng(200 + rank)
    size, block_count, probe_count = 24, 4, 16
    level = certificate_level(8)
    blocks = tuple(np.arange(start, start + size) for start in range(0, size * block_count, size))
    covariance = np.zeros((size * block_count, size * block_count))
    skewness = np.zeros(block_count)
    for position, members in enumerate(blocks):
        block = _low_rank_block(generator, size, rank)
        covariance[np.ix_(members, members)] = block
        skewness[position] = _rademacher_skewness(block)
    variances = np.diag(covariance).copy()
    trials = 400
    misses = 0
    for _trial in range(trials):
        probes = generator.choice([-1.0, 1.0], size=(size * block_count, probe_count))
        certificate = block_trace_certificate(variances, blocks, probes, covariance @ probes, 0.0, level, skewness)
        misses += int(certificate.violated.any())
    assert misses <= level * trials + 3 * np.sqrt(level * trials)


def test_zero_skewness_reproduces_the_student_t_interval():
    generator = np.random.default_rng(24)
    values = [generator.standard_normal(16) + 5.0 for _ in range(3)]
    estimate = np.array([5.0, 5.0, 5.0])
    level = certificate_level(64)
    certificate = marginal_variances_module._certificate(estimate, values, 0.1, level, np.zeros(3))
    quantile = float(student_t.isf(0.5 * level / 3, 15))
    spread = np.array([np.std(v, ddof=1) / 4.0 / 5.0 for v in values])
    relative = np.array([(np.mean(v) - 5.0) / 5.0 for v in values])
    assert np.allclose(certificate.lower_bound, relative - quantile * spread, rtol=0, atol=1e-15)
    assert np.allclose(certificate.upper_bound, relative + quantile * spread, rtol=0, atol=1e-15)


def test_an_inconsistent_window_gram_is_refused():
    generator, columns, precision, blocks, solve = _strong_case(27)
    grams = _grams(columns, blocks)
    # Cross-block Grams from another design: the assembled window is no longer a Gram matrix.
    other = generator.standard_normal(columns.shape)
    wrong = BlockGrams(blocks=grams.blocks, within=grams.within,
                       next_cross=tuple(3.0 * (other[:, blocks[i]].T @ other[:, blocks[i + 1]]) for i in range(len(blocks) - 1)))
    with pytest.raises(ValueError, match="not positive semidefinite"):
        marginal_variances(solve, wrong)


def test_a_thin_middle_block_leaves_the_window_positive_semidefinite():
    # Strong LD across a 5-variant middle block: the zero corner would make the three-block window indefinite.
    generator = np.random.default_rng(28)
    columns = _genotypes(generator, 2000, 205, 0.99)
    blocks = (np.arange(0, 100), np.arange(100, 105), np.arange(105, 205))
    grams = _grams(columns, blocks)
    gram, _columns, own = marginal_variances_module._window(grams, 1)
    assert np.linalg.eigvalsh(gram)[0] < 0.0
    _completed, whitened, _root = marginal_variances_module._whitened_window(gram, np.ones(gram.shape[0]), own, 1, np)
    assert np.linalg.eigvalsh(whitened)[0] >= -marginal_variances_module._storage_allowance(whitened, np)


def _unregularized_corner(gram: np.ndarray, own: slice) -> np.ndarray:
    """The window with the completion R_{b-1,b} R_bb^+ R_{b,b+1} and no rounding allowance."""
    completed = gram.copy()
    first, last = slice(0, own.start), slice(own.stop, gram.shape[0])
    corner = completed[first, own] @ np.linalg.lstsq(completed[own, own], completed[own, last], rcond=None)[0]
    completed[first, last] = corner
    completed[last, first] = corner.T
    return completed


def test_a_singular_float32_middle_block_completes_within_its_rounding():
    # e2e-scale's full chr22 failure in small: n < |b| makes R_bb singular, float32 storage leaves its null space at
    # the rounding level, and R_bb^+ amplifies the rounding of R_{b,b+1} there past the allowance. The completion of
    # B + eps I stays inside it, and the marginals are computed.
    generator = np.random.default_rng(32)
    columns = _genotypes(generator, 120, 600, 0.99)
    blocks = (np.arange(0, 200), np.arange(200, 400), np.arange(400, 600))
    exact = _grams(columns, blocks)
    stored = BlockGrams(blocks=blocks, within=tuple(w.astype(np.float32) for w in exact.within),
                        next_cross=tuple(c.astype(np.float32) for c in exact.next_cross))
    gram, _columns, own = marginal_variances_module._window(stored, 1)
    unregularized = _unregularized_corner(gram, own)
    assert np.linalg.eigvalsh(unregularized)[0] < -marginal_variances_module._storage_allowance(unregularized, np)
    _completed, whitened, _root = marginal_variances_module._whitened_window(gram, np.ones(gram.shape[0]), own, 1, np)
    assert np.linalg.eigvalsh(whitened)[0] >= -marginal_variances_module._storage_allowance(whitened, np)
    precision = np.full(600, 600 / 0.5)
    solve = _solve(columns, precision, np.zeros(0, dtype=np.int64))
    variances = marginal_variances(solve, stored)
    assert np.all(np.isfinite(variances)) and np.all(variances > 0.0)


def test_a_refused_window_names_its_block_and_the_inconsistent_piece():
    generator, columns, precision, blocks, solve = _strong_case(27)
    grams = _grams(columns, blocks)
    other = generator.standard_normal(columns.shape)
    cross = list(grams.next_cross)
    cross[1] = 3.0 * (other[:, blocks[1]].T @ other[:, blocks[2]])
    wrong = BlockGrams(blocks=grams.blocks, within=grams.within, next_cross=tuple(cross))
    with pytest.raises(ValueError, match=r"^block [12]: .*with the next block|^block [12]: .*with the previous block"):
        marginal_variances(solve, wrong)


def test_probes_to_decide_takes_a_per_block_tolerance_with_mixed_decisions():
    # speed-recycle's case: a per-block tolerance array and a stage with certified and undecided blocks.
    # Dyadic values, so the expected probe count is exact in floating point.
    relative = np.array([0.0, 0.0, 0.5, 0.5])
    standard = np.full(4, 0.125)
    tolerance = np.array([2.0, 2.0, 0.75, 0.75])
    width = 3.0 * standard
    certificate = BlockCertificate(
        relative_error=relative, standard_error=standard, lower_bound=relative - width, upper_bound=relative + width,
        tolerance=tolerance, level=certificate_level(64),
        certified=(relative - width >= -tolerance) & (relative + width <= tolerance),
        violated=(relative - width > tolerance) | (relative + width < -tolerance),
    )
    assert certificate.certified.tolist() == [True, True, False, False]
    # The undecided blocks sit 0.25 inside their tolerance with half-width 0.375: (0.375 / 0.25)^2 = 2.25 times the probes.
    assert probes_to_decide(certificate, 16) == 36


def test_probes_to_decide_counts_the_probes_that_place_a_zero_estimate():
    # speed-recycle's and svpgs-integrator's NaN: a zero estimate whose probes see information has infinite relative
    # bounds, and inf / inf gave NaN. It is decided once the absolute interval excludes zero, at k (q s / |m|)^2 probes.
    k = 16
    level = certificate_level(64)
    quantile = float(student_t.isf(0.5 * level / 2, k - 1))
    base = np.array([1.0, -1.0] * (k // 2))
    straddling = 0.5 * quantile * np.std(base, ddof=1) / np.sqrt(k) + base  # mean = half the interval's half-width
    exact = np.zeros(k)
    certificate = marginal_variances_module._certificate(np.array([0.0, 0.0]), [straddling, exact], 0.1, level)
    assert certificate.certified.tolist() == [False, True] and not certificate.violated.any()
    assert np.isclose(certificate.zero_estimate_ratio[0], 2.0) and np.isnan(certificate.zero_estimate_ratio[1])
    assert probes_to_decide(certificate, k) == int(np.ceil(k * certificate.zero_estimate_ratio[0] ** 2))
    centred = marginal_variances_module._certificate(np.array([0.0]), [base], 0.1, level)
    assert probes_to_decide(centred, k) == np.inf



def test_shared_float32_grams_with_a_scale_give_the_float64_answer():
    _generator, columns, precision, blocks, solve = _strong_case(30)
    grams = _grams(columns, blocks)
    noise = 0.8
    scaled = BlockGrams(blocks=grams.blocks, within=tuple(w * noise for w in grams.within),
                        next_cross=tuple(c * noise for c in grams.next_cross))
    stored = BlockGrams(blocks=scaled.blocks, within=tuple(w.astype(np.float32) for w in scaled.within),
                        next_cross=tuple(c.astype(np.float32) for c in scaled.next_cross), scale=1.0 / noise)
    reference = marginal_variances(solve, grams)
    shared = marginal_variances(solve, stored)
    # float32 storage rounds each Gram entry by u32; the variances move by at most that times the window's
    # conditioning, which the tolerance bounds (I + omega_F B has eigenvalues >= 1).
    window = max(sum(grams.blocks[m].shape[0] for m in marginal_variances_module._window_blocks(grams, b)) for b in range(len(blocks)))
    bound = np.finfo(np.float32).eps * window * (1.0 + solve.bulk_trace * float(np.linalg.eigvalsh(columns.T @ columns)[-1]) * np.max(1.0 / precision))
    assert np.all(np.abs(shared - reference) <= bound * np.abs(reference))
    assert marginal_variances_module.window_working_bytes(stored) == marginal_variances_module.window_working_bytes(grams)


def test_block_covariance_is_the_window_maps_own_block_and_shares_its_preparation():
    generator = np.random.default_rng(31)
    sample_count, variant_count, heritability = 1500, 600, 0.5
    columns = _genotypes(generator, sample_count, variant_count, 0.97) / np.sqrt(1.0 - heritability)
    # Small per-variant prior variances: no spike rises far above the bulk level, so taking every site as bulk (an
    # empty resolved set, which the split allows for any positive D) keeps the equivalent in its regime. With no
    # resolved site there is no far-resolved term, and the marginals are the blocks' own diagonals.
    precision = variant_count / (0.1 * heritability) * np.exp(0.3 * generator.normal(0.0, 1.0, variant_count))
    blocks = tuple(np.arange(start, start + 100) for start in range(0, variant_count, 100))
    solve = _solve(columns, precision, np.zeros(0, dtype=np.int64))
    grams = _grams(columns, blocks)
    prepared = prepare_windows(solve, grams)
    variances = marginal_variances(solve, grams)
    exact = np.linalg.inv(columns.T @ columns + np.diag(precision))
    for block, members in enumerate(blocks):
        covariance = block_covariance(solve, grams, block, prepared=prepared)
        assert covariance.shape == (members.shape[0], members.shape[0]) and covariance.dtype == np.float64
        assert np.array_equal(covariance, covariance.T)
        assert np.array_equal(covariance, block_covariance(solve, grams, block))
        assert np.allclose(np.diag(covariance), variances[members], rtol=1e-12)
        # Off the diagonal too, the block's covariance tracks the exact one to the equivalent's scale.
        error = np.linalg.norm(covariance - exact[np.ix_(members, members)]) / np.linalg.norm(exact[np.ix_(members, members)])
        assert error <= approximation_scale(solve)
