"""The small-n route (sv_pgs/small_n.py): its kernel-form algebra against brute-force dense inverses, its Stage 0
against the definitions, and one end-to-end fit on synthetic genotypes. Synthetic data only."""

import numpy as np
import pytest

from sv_pgs.config import VariantClass
from sv_pgs.fast_scoring import SIGNED_CODE_OFFSET
from sv_pgs.small_n import _DensePosterior, _Design, _Kernel, _new_profile, dense_statistics, fit_small_n

_ROUNDING = 1e-9


def _design(rng, samples, variants):
    return np.asfortranarray(rng.standard_normal((samples, variants)))


def _dense_inverse(design, precision):
    return np.linalg.inv(design.T @ design + np.diag(precision))


@pytest.mark.parametrize("negative", [0, 3])
def test_kernel_solve_variances_and_jvp_match_the_dense_inverse(negative):
    rng = np.random.default_rng(1 + negative)
    design = _design(rng, 12, 30)
    precision = rng.uniform(0.5, 2.0, 30)
    # Negative sites on columns the data resolve: A stays positive definite.
    precision[:negative] = -0.05
    inverse = _dense_inverse(design, precision)
    assert np.all(np.linalg.eigvalsh(design.T @ design + np.diag(precision)) > 0.0)
    kernel = _Kernel(_Design.dense(design), precision)
    right = rng.standard_normal((30, 4))
    np.testing.assert_allclose(kernel.solve(right), inverse @ right, rtol=_ROUNDING, atol=_ROUNDING)
    np.testing.assert_allclose(kernel.variances(), np.diag(inverse), rtol=_ROUNDING, atol=_ROUNDING)
    weights = rng.standard_normal((30, 5))
    noise = 1.7
    expected = -(noise * noise) * (inverse * inverse) @ weights
    for jvp_bytes in (10**9, 0):  # formed Sigma o Sigma, and the factor form
        posterior = _DensePosterior(kernel, noise, jvp_bytes, _new_profile())
        np.testing.assert_allclose(posterior.variance_jvp(weights), expected, rtol=_ROUNDING, atol=_ROUNDING)
        np.testing.assert_allclose(posterior.solve(right, 0.0), noise * inverse @ right, rtol=_ROUNDING, atol=_ROUNDING)


def test_the_cavity_precision_keeps_its_digits_where_the_data_barely_inform_a_column():
    rng = np.random.default_rng(9)
    design = _design(rng, 10, 6)
    design[:, 0] *= 1e-7  # a column whose data information q ~ 1e-14 of its site precision
    precision = rng.uniform(0.5, 2.0, 6)
    precision[0] = 1e3
    _variances, removed, cavity = _Kernel(_Design.dense(design), precision).cavity()
    # Reference, no cancellation: the cavity precision of column j is x_j' (I + X_-j T_-j^-1 X_-j')^-1 x_j.
    for column in range(6):
        others = np.delete(np.arange(6), column)
        kernel = np.eye(10) + design[:, others] @ np.diag(1.0 / precision[others]) @ design[:, others].T
        expected = float(design[:, column] @ np.linalg.solve(kernel, design[:, column]))
        np.testing.assert_allclose(cavity[column], expected, rtol=1e-9)
        np.testing.assert_allclose(removed[column], expected / (expected + precision[column]), rtol=1e-9)
    # The naive 1/z - t has no digits left for column 0.
    naive = 1.0 / _Kernel(_Design.dense(design), precision).variances()[0] - precision[0]
    assert abs(naive - cavity[0]) > 1e-6 * abs(cavity[0])


def test_a_column_spanned_by_negative_site_columns_has_a_small_negative_cavity():
    """x3 = x1 + x2 (a doubleton, the sum of two singleton columns) with slightly negative sites on x1 and x2: x3's
    cavity precision is negative and proportional to those sites, never positive however small they get. The refresh
    must therefore accept it whenever the tilted law is proper (1 + v_max P > 0), not ask P > 0."""
    rng = np.random.default_rng(12)
    design = _design(rng, 15, 6)
    design[:, 2] = design[:, 0] + design[:, 1]
    for scale in (1e-3, 1e-6, 1e-9):
        precision = rng.uniform(0.5, 2.0, 6)
        precision[[0, 1]] = -scale
        _variances, _removed, cavity = _Kernel(_Design.dense(design), precision).cavity()
        others = np.array([0, 1, 3, 4, 5])
        # Reference (T invertible here): x3' (I + X_-3 T_-3^-1 X_-3')^-1 x3.
        kernel = np.eye(15) + design[:, others] @ np.diag(1.0 / precision[others]) @ design[:, others].T
        expected = float(design[:, 2] @ np.linalg.solve(kernel, design[:, 2]))
        assert expected < 0.0 and cavity[2] < 0.0
        np.testing.assert_allclose(cavity[2], expected, rtol=1e-6, atol=10 * _ROUNDING * scale)
        assert abs(cavity[2]) <= 10.0 * scale


def test_the_sparse_carrier_design_gives_the_dense_design_quantities():
    """Stage 0's sparse G (minor-allele codes over their SD) with the intercept projected out equals the dense projected
    standardized design in every kernel quantity, including negative sites."""
    rng = np.random.default_rng(21)
    samples, variants = 60, 90
    frequency = rng.uniform(0.005, 0.6, variants)
    dosage = rng.binomial(2, frequency, size=(samples, variants))
    dosage[:, 0] = 0
    dosage[3, 0] = 1  # a singleton
    codes = (dosage * 127).astype(np.uint8)
    statistics = dense_statistics(codes, np.column_stack([np.ones(samples), rng.standard_normal(samples)]), rng.standard_normal(samples))
    assert statistics.design.is_sparse
    dense = _Design.dense(statistics.projected)
    count = statistics.design.variant_count
    precision = rng.uniform(0.5, 3.0, count)
    precision[:2] = -0.01
    sparse_kernel, dense_kernel = _Kernel(statistics.design, precision), _Kernel(dense, precision)
    right = rng.standard_normal((count, 3))
    np.testing.assert_allclose(sparse_kernel.solve(right), dense_kernel.solve(right), rtol=1e-9, atol=1e-11)
    for sparse_part, dense_part in zip(sparse_kernel.cavity(), dense_kernel.cavity()):
        np.testing.assert_allclose(sparse_part, dense_part, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(sparse_kernel.covariance(), dense_kernel.covariance(), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(statistics.design.column_squares(), np.einsum("ij,ij->j", statistics.projected, statistics.projected), rtol=1e-10)
    # X~'s covariate loading, against the definition.
    signed = codes[:, statistics.reduced_rows].astype(np.float64) - SIGNED_CODE_OFFSET
    kept = np.asarray(statistics.tie_map.kept_indices)
    standardized = (signed - statistics.means[kept]) / statistics.scales[kept]
    np.testing.assert_allclose(statistics.loading, statistics.covariates.T @ standardized, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("negative", [0, 2])
def test_the_exact_linear_response_solves_the_curvature_fixed_point(negative):
    rng = np.random.default_rng(31 + negative)
    design = _design(rng, 10, 25)
    precision = rng.uniform(0.5, 2.0, 25)
    precision[:negative] = -0.05
    noise = 0.8
    sigma = noise * _dense_inverse(design, precision)
    left, right, diagonal, weight = (rng.standard_normal(25) for _ in range(4))
    weight = np.abs(weight)
    rhs = rng.standard_normal((25, 4))
    matrix = np.eye(25) - (np.eye(25) - weight[:, None] * (sigma * sigma)) @ (left[:, None] * sigma * right[None, :] + np.diag(diagonal))
    posterior = _DensePosterior(_Kernel(_Design.dense(design), precision), noise, 10**9, _new_profile())
    np.testing.assert_allclose(posterior.linear_response(left, right, diagonal, weight, rhs), np.linalg.solve(matrix, rhs), rtol=1e-8, atol=1e-10)


def test_the_total_curvature_by_the_exact_response_equals_gmres():
    from sv_pgs.scale_mixture_ep import Cavity, GaussianPosterior, _total_curvature, derived_lattice, initial_hyperparameters, scale_mixture_prior

    rng = np.random.default_rng(41)
    samples, variants = 30, 40
    design = _design(rng, samples, variants)
    precision = rng.uniform(0.5, 2.0, variants)
    precision[0] = -0.02
    noise = 1.3
    posterior = _DensePosterior(_Kernel(_Design.dense(design), precision), noise, 10**9, _new_profile())
    single_precision = np.einsum("ij,ij->j", design, design) / noise
    single_shift = rng.standard_normal(variants) * 2.0
    nodes, floor, top = derived_lattice(single_precision, single_shift, np.zeros(variants), 1.0 / 128)
    prior = scale_mixture_prior(
        class_index=np.arange(variants) % 2, log_variance_offset=np.zeros(variants), annotation_design=np.zeros((variants, 0)),
        annotation_groups=(), nodes=nodes, floor=floor, top=top,
    )
    coefficients = initial_hyperparameters(prior).coefficients
    variances, _removed, cavity_precision = posterior.kernel.cavity()
    cavity = Cavity(precision=cavity_precision / noise, shift=rng.standard_normal(variants))
    exact = _total_curvature(prior, coefficients, cavity, posterior.gaussian_posterior(), 10**9, 1e-13)
    iterative = _total_curvature(prior, coefficients, cavity, GaussianPosterior(solve=posterior.solve, variance_jvp=posterior.variance_jvp), 10**9, 1e-13)
    np.testing.assert_allclose(exact, iterative, rtol=1e-7, atol=1e-9 * float(np.max(np.abs(iterative))))


def test_kernel_refuses_an_indefinite_precision():
    rng = np.random.default_rng(7)
    design = _design(rng, 5, 20)
    precision = np.ones(20)
    precision[0] = -50.0
    with pytest.raises(np.linalg.LinAlgError):
        _Kernel(_Design.dense(design), precision)


def test_draws_have_the_posterior_covariance():
    rng = np.random.default_rng(3)
    design = _design(rng, 6, 8)
    precision = rng.uniform(0.5, 2.0, 8)
    precision[0] = -0.1
    inverse = _dense_inverse(design, precision)
    draws = _Kernel(_Design.dense(design), precision).draws(np.random.default_rng(11), 200_000)
    empirical = draws @ draws.T / draws.shape[1]
    # Monte Carlo error of a covariance entry: sqrt((S_ii S_jj + S_ij^2) / K) <= sqrt(2 / K) max S_ii.
    bound = 5.0 * np.sqrt(2.0 / draws.shape[1]) * float(np.max(np.diag(inverse)))
    assert float(np.max(np.abs(empirical - inverse))) <= bound


def test_stage0_standardizes_and_merges_exact_ties():
    rng = np.random.default_rng(5)
    samples = 40
    dosage = rng.integers(0, 3, size=(samples, 6))
    dosage[:, 2] = dosage[:, 0]          # a copy
    dosage[:, 3] = 2 - dosage[:, 0]      # a negated copy
    dosage[:, 4] = 1                     # every sample heterozygous: no variance
    carrier = rng.integers(0, 2, size=samples)
    dosage[:, 5] = carrier               # 0/1 ...
    extra = 2 * carrier                  # ... and its 0/2 multiple: an exact tie of r = 1
    dosage = np.column_stack([dosage, extra])
    codes = (dosage * 127).astype(np.uint8)
    covariates = np.ones((samples, 1))
    target = rng.standard_normal(samples)
    statistics = dense_statistics(codes, covariates, target)
    np.testing.assert_array_equal(statistics.active_rows, [0, 1, 2, 3, 5, 6])
    signed = codes[:, statistics.active_rows].astype(np.float64) - SIGNED_CODE_OFFSET
    np.testing.assert_allclose(statistics.means, signed.mean(axis=0), rtol=1e-14)
    np.testing.assert_allclose(statistics.scales, signed.std(axis=0), rtol=1e-14)
    tie_map = statistics.tie_map
    # Active positions: 0 -> column 0, 1 -> 1, 2 -> 2 (copy of 0), 3 -> 3 (negated 0), 4 -> 5, 5 -> 6 (tie of 5).
    np.testing.assert_array_equal(tie_map.kept_indices, [0, 1, 4])
    np.testing.assert_array_equal(tie_map.original_to_reduced, [0, 1, 0, 0, 2, 2])
    signs = {int(group.representative_index): dict(zip(group.member_indices.tolist(), group.signs.tolist())) for group in tie_map.reduced_to_group}
    assert signs[0] == {0: 1.0, 2: 1.0, 3: -1.0}
    assert signs[4] == {4: 1.0, 5: 1.0}
    # (I - H_C) X~ with the intercept: centred standardized columns of the representatives.
    representatives = signed[:, [0, 1, 4]]
    standardized = (representatives - representatives.mean(axis=0)) / representatives.std(axis=0)
    np.testing.assert_allclose(statistics.projected, standardized, atol=1e-12)
    np.testing.assert_allclose(statistics.projected_target, target - target.mean(), atol=1e-12)


@pytest.mark.slow  # the whole outer loop on 120 columns: about a minute on one core
def test_the_small_n_fit_certifies_and_scores():
    """Machinery only (own simulation): the outer loop certifies, and the scoring model carries the fit."""
    rng = np.random.default_rng(17)
    samples, variants = 150, 120
    frequency = rng.uniform(0.05, 0.5, variants)
    dosage = rng.binomial(2, frequency, size=(samples, variants))
    effects = np.zeros(variants)
    effects[[10, 50, 90]] = [0.8, -0.6, 0.5]
    standardized = (dosage - dosage.mean(axis=0)) / dosage.std(axis=0)
    target = standardized @ effects + rng.standard_normal(samples)
    codes = (dosage * 127).astype(np.uint8)
    classes = np.full(variants, list(VariantClass).index(VariantClass.SNV), dtype=np.uint8)
    fit = fit_small_n(
        codes=codes, covariates=np.ones((samples, 1)), target=target, variant_class=classes, log_variance_offset=None,
        draw_count=64, working_bytes=2 * 10**9, seed=0,
    )
    assert fit.certificate.remaining_gain[0] <= 0.5 / 64
    assert fit.certificate.prediction_move[0] <= fit.certificate.prediction_tolerance[0]
    assert np.all(np.isfinite(fit.scoring.coefficients)) and fit.scoring.posterior_draws.shape == (variants, 64)
    assert np.all(np.isfinite(fit.scoring.posterior_draws)) and fit.noise_variance > 0.0
    np.testing.assert_array_equal(fit.scoring.store_rows, np.arange(variants))
