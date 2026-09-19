"""Leave-block-out marginal variances against the dense inverse diag((Xt'Xt + diag(Pi))^-1).

The two identities (resolved/bulk elimination, and the window Woodbury) are checked exactly. The
deterministic equivalent is checked against the dense inverse on genotypes whose LD crosses every block
cut, next to the block-Jacobi variances it replaces. The certificate is checked to flag a block whose
variances are wrong and to leave the others alone.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from sv_pgs.marginal_variances import (
    BlockGrams,
    BulkSolve,
    approximation_scale,
    block_trace_certificate,
    certificate_tolerance,
    covariance_products,
    far_field_trace,
    marginal_variances,
    marginals_from_quadratics,
    variance_jvp,
    window_bulk_quadratic,
)


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
    certificate = block_trace_certificate(variances, blocks, probes, covariance @ probes, approximation_scale(solve))
    assert certificate.violated.tolist() == [False, False, False, True, False, False]


def _strong_case(seed: int):
    generator = np.random.default_rng(seed)
    sample_count, variant_count, heritability = 1500, 600, 0.5
    columns = _genotypes(generator, sample_count, variant_count, 0.97) / np.sqrt(1.0 - heritability)
    variance = heritability / variant_count * np.exp(generator.normal(0.0, 1.0, variant_count))
    variance[generator.choice(variant_count, 4, replace=False)] *= 400.0
    precision = 1.0 / variance
    blocks = tuple(np.arange(start, start + 100) for start in range(0, variant_count, 100))
    return generator, columns, precision, blocks, _solve(columns, precision, _resolved(variance, sample_count))


def test_covariance_products_are_exact_given_the_back_products():
    generator, columns, precision, _blocks, solve = _strong_case(6)
    bulk = 1.0 / precision
    bulk[solve.resolved] = 0.0
    kernel_inverse = np.linalg.inv(np.eye(columns.shape[0]) + (columns * bulk) @ columns.T)
    probes = generator.choice([-1.0, 1.0], size=(columns.shape[1], 5))
    back = columns.T @ (kernel_inverse @ (columns @ (bulk[:, None] * probes)))
    covariance = np.linalg.inv(columns.T @ columns + np.diag(precision))
    assert np.allclose(covariance_products(solve, probes, back), covariance @ probes, rtol=1e-8, atol=1e-12)


def test_variance_jvp_tracks_the_dense_derivative():
    generator, columns, precision, blocks, solve = _strong_case(7)
    covariance = np.linalg.inv(columns.T @ columns + np.diag(precision))
    direction = generator.uniform(0.0, 1.0, size=(columns.shape[1], 3)) * precision[:, None]
    exact = -np.einsum("jk,kr,jk->jr", covariance, direction, covariance)
    product = variance_jvp(solve, _grams(columns, blocks), direction)
    error = np.abs(product.values - exact)
    assert np.all(error[solve.resolved] <= 1e-8 * np.abs(exact[solve.resolved]))
    # Sums of squared entries carry at most twice the entries' relative error (d x^2 / x^2 = 2 dx / x),
    # and the B-products consume block sums.
    scale = approximation_scale(solve)
    for block in blocks:
        assert np.all(np.abs(product.values[block].sum(axis=0) - exact[block].sum(axis=0)) <= 2 * scale * np.abs(exact[block].sum(axis=0)))


def test_certificate_tolerance_adds_the_probe_error_in_quadrature():
    _generator, _columns, _precision, _blocks, solve = _strong_case(8)
    scale = approximation_scale(solve)
    assert np.isclose(certificate_tolerance(solve, 2), scale * np.sqrt(2.0))
    assert certificate_tolerance(solve, 10**9) < scale * (1 + 1e-8)
