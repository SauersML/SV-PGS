"""The small-n route (sv_pgs/small_n.py): its kernel-form algebra against brute-force dense inverses, its Stage 0
against the definitions, and one end-to-end fit on synthetic genotypes. Synthetic data only."""

import numpy as np
import pytest

from sv_pgs.config import VariantClass
from sv_pgs.fast_scoring import SIGNED_CODE_OFFSET
from sv_pgs.small_n import _DensePosterior, _Kernel, _new_profile, dense_statistics, fit_small_n

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
    kernel = _Kernel(design, precision)
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
    _variances, removed, cavity = _Kernel(np.asfortranarray(design), precision).cavity()
    # Reference, no cancellation: the cavity precision of column j is x_j' (I + X_-j T_-j^-1 X_-j')^-1 x_j.
    for column in range(6):
        others = np.delete(np.arange(6), column)
        kernel = np.eye(10) + design[:, others] @ np.diag(1.0 / precision[others]) @ design[:, others].T
        expected = float(design[:, column] @ np.linalg.solve(kernel, design[:, column]))
        np.testing.assert_allclose(cavity[column], expected, rtol=1e-9)
        np.testing.assert_allclose(removed[column], expected / (expected + precision[column]), rtol=1e-9)
    # The naive 1/z - t has no digits left for column 0.
    naive = 1.0 / _Kernel(np.asfortranarray(design), precision).variances()[0] - precision[0]
    assert abs(naive - cavity[0]) > 1e-6 * abs(cavity[0])


def test_kernel_refuses_an_indefinite_precision():
    rng = np.random.default_rng(7)
    design = _design(rng, 5, 20)
    precision = np.ones(20)
    precision[0] = -50.0
    with pytest.raises(np.linalg.LinAlgError):
        _Kernel(design, precision)


def test_draws_have_the_posterior_covariance():
    rng = np.random.default_rng(3)
    design = _design(rng, 6, 8)
    precision = rng.uniform(0.5, 2.0, 8)
    precision[0] = -0.1
    inverse = _dense_inverse(design, precision)
    draws = _Kernel(design, precision).draws(np.random.default_rng(11), 200_000)
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


def test_the_small_n_fit_runs_and_recovers_a_sparse_signal():
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
    coefficients = fit.scoring.coefficients
    assert set(np.argsort(-np.abs(coefficients))[:3].tolist()) == {10, 50, 90}
    assert fit.scoring.posterior_draws.shape == (variants, 64)
    assert 0.5 < fit.noise_variance < 2.0
