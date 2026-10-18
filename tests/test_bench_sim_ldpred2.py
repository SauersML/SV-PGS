"""LDpred2's GWAS and streamed LD must agree with direct covariate-adjusted regression."""
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.bench_sim.submissions import ldpred2_auto, ldpred2_auto_allvar


def training_view(redundant):
    rng = np.random.default_rng(24)
    codes = rng.integers(0, 255, size=(23, 80), dtype=np.uint8)
    covariates = rng.normal(size=(80, 3))
    if redundant:
        covariates = np.column_stack([covariates, covariates[:, 0], np.ones(80), np.zeros(80)])
    return SimpleNamespace(n_samples=80, covariates=covariates, phenotype=rng.normal(size=80),
                           variants={"cm": np.linspace(0.0, 10.0, 23), "cls": np.zeros(23, dtype=int)},
                           codes=lambda rows: codes[rows])


@pytest.mark.parametrize("redundant", [False, True])
def test_gwas_matches_direct_least_squares_with_the_actual_covariate_rank(redundant):
    train = training_view(redundant)
    rows = np.arange(23)
    projected, residual, means, rank = ldpred2_auto.covariate_adjusted(train, rows)
    beta, se, squares = ldpred2_auto.marginal_statistics(projected, residual, rank)
    assert rank == 4
    design = np.column_stack([np.ones(80), train.covariates])
    dosages = train.codes(rows).astype(np.float32) / np.float32(127.0)
    for index, dosage in enumerate(dosages):
        full = np.column_stack([design, dosage])
        coefficients = np.linalg.lstsq(full, train.phenotype, rcond=None)[0]
        errors = train.phenotype - full @ coefficients
        covariance = np.linalg.pinv(full.T @ full) * (errors @ errors) / (80 - rank - 1)
        assert beta[index] == pytest.approx(coefficients[-1], abs=2e-7)
        assert se[index] == pytest.approx(np.sqrt(covariance[-1, -1]), abs=2e-7)
    np.testing.assert_allclose(means, dosages.mean(axis=1, dtype=np.float64))
    adjusted = ldpred2_auto_allvar.Adjuster(train).block(rows)
    for actual, expected in zip(adjusted[:4], (means, beta, se, squares)):
        np.testing.assert_allclose(actual, expected)


def test_streamed_ld_band_matches_dense_projection_across_block_boundaries(tmp_path, monkeypatch):
    train = training_view(True)
    rows = np.arange(23)
    monkeypatch.setattr(ldpred2_auto_allvar, "CHUNK", 4)
    means, beta, se, lower, upper, scores = ldpred2_auto_allvar.write_band(train, rows, tmp_path)
    projected, residual, direct_means, rank = ldpred2_auto.covariate_adjusted(train, rows)
    direct_beta, direct_se, squares = ldpred2_auto.marginal_statistics(projected, residual, rank)
    scaled = projected / np.sqrt(squares)[:, None]
    dense = scaled @ scaled.T
    np.fill_diagonal(dense, 1.0)
    cm = train.variants["cm"]
    inside = np.abs(cm[:, None] - cm[None, :]) <= ldpred2_auto.WINDOW_CM
    np.testing.assert_allclose(np.fromfile(tmp_path / "ld_values.bin", dtype="<f4"), dense[inside], atol=2e-7)
    np.testing.assert_allclose(scores, np.where(inside, dense**2, 0.0).sum(axis=1), rtol=2e-7)
    np.testing.assert_array_equal(upper - lower, inside.sum(axis=1))
    for actual, expected in zip((means, beta, se), (direct_means, direct_beta, direct_se)):
        np.testing.assert_allclose(actual, expected, atol=2e-7)


def test_empty_common_snv_set_is_an_explicit_failure():
    train = training_view(False)
    train.codes = lambda rows: np.zeros((len(rows), train.n_samples), dtype=np.uint8)
    with pytest.raises(ValueError, match="at least one SNV"):
        ldpred2_auto.hapmap3_density_rows(train)


def test_saturated_covariates_cannot_produce_gwas_standard_errors():
    train = SimpleNamespace(n_samples=4, covariates=np.eye(4))
    with pytest.raises(ValueError, match="residual degrees of freedom"):
        ldpred2_auto.covariate_basis(train)
