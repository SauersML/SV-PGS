"""Bit-exact equivalence between the new one-pass marginal-z code and the
mathematical definition, plus equivalence of chunked-vs-whole computation.

These cover the refactor that turns 17 sequential transpose_matvec calls into
a single transpose_matmat call: the math must be unchanged. They also cover
the model.py chunking wrapper: computing per-chunk and concatenating must
match the whole-set computation.

No GPU required.
"""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.preprocessing import compute_marginal_z_scores


class _DenseStandardizedView:
    """Minimal stand-in matching the StandardizedGenotypeMatrix surface used by
    compute_marginal_z_scores (subset + transpose_matmat_numpy + shape)."""

    def __init__(self, x_std: np.ndarray):
        self._x = np.asarray(x_std, dtype=np.float32)

    @property
    def shape(self) -> tuple[int, int]:
        return self._x.shape

    def subset(self, active_indices: np.ndarray) -> "_DenseStandardizedView":
        return _DenseStandardizedView(self._x[:, np.asarray(active_indices, dtype=np.int64)])

    def transpose_matvec_numpy(self, vec: np.ndarray, batch_size: int = 0) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32)
        return np.asarray(self._x.T @ v, dtype=np.float32)

    def transpose_matmat_numpy(self, mat: np.ndarray, batch_size: int = 0) -> np.ndarray:
        m = np.asarray(mat, dtype=np.float32)
        return np.asarray(self._x.T @ m, dtype=np.float32)


def _make_synthetic(n_samples: int, n_variants: int, n_covariates: int, seed: int):
    rng = np.random.default_rng(seed)
    # Build standardized genotypes (mean 0, var 1) and a random covariate matrix +
    # a binary-ish target with some true signal in the first 3 variants.
    raw = rng.normal(size=(n_samples, n_variants)).astype(np.float32)
    raw = (raw - raw.mean(axis=0, keepdims=True)) / raw.std(axis=0, keepdims=True).clip(min=1e-6)
    covariates = rng.normal(size=(n_samples, n_covariates)).astype(np.float32)
    beta = np.zeros(n_variants, dtype=np.float32)
    beta[:3] = rng.normal(scale=0.5, size=3)
    target = (raw @ beta + covariates @ rng.normal(size=n_covariates).astype(np.float32)
              + rng.normal(scale=1.0, size=n_samples)).astype(np.float32)
    return raw, covariates, target


def _reference_z_scores(
    x_std: np.ndarray, active_indices: np.ndarray, covariates: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Direct numpy implementation of the same formula compute_marginal_z_scores
    encodes, used as the ground truth."""
    n = target.shape[0]
    target_f64 = target.astype(np.float64).reshape(-1)
    cov_f64 = covariates.astype(np.float64)
    has_const = False
    for k in range(cov_f64.shape[1]):
        col = cov_f64[:, k]
        if np.allclose(col, col[0], atol=1e-12) and abs(col[0]) > 1e-12:
            has_const = True
            break
    if cov_f64.shape[1] == 0:
        y_resid = target_f64 - target_f64.mean()
        proj = np.zeros((n, 0))
        ctc_inv = None
    else:
        if has_const:
            proj = cov_f64
        else:
            proj = np.concatenate([np.ones((n, 1)), cov_f64], axis=1)
        alpha, *_ = np.linalg.lstsq(proj, target_f64, rcond=None)
        y_resid = target_f64 - proj @ alpha
        ctc_inv = np.linalg.pinv(proj.T @ proj)
    sigma2 = float(y_resid @ y_resid / max(n, 1))
    if sigma2 <= 0.0:
        return np.zeros(active_indices.size, dtype=np.float32)
    x_active = x_std[:, active_indices].astype(np.float64)
    sum_xy = x_active.T @ y_resid
    if proj.shape[1] > 0:
        cprime_x = proj.T @ x_active                       # (p, m_active)
        proj_diag = np.einsum("ki,kl,li->i", cprime_x, ctc_inv, cprime_x)
    else:
        proj_diag = np.zeros(active_indices.size)
    xpx = np.maximum(np.full(active_indices.size, float(n)) - proj_diag, 0.0)
    denom = np.sqrt(sigma2 * xpx)
    z = np.zeros(active_indices.size, dtype=np.float64)
    safe = denom > 0.0
    z[safe] = sum_xy[safe] / denom[safe]
    return z.astype(np.float32)


@pytest.mark.parametrize("n_samples,n_variants,n_cov", [
    (200, 50, 0),       # no covariates branch
    (200, 50, 5),       # standard
    (200, 50, 10),
    (1000, 500, 15),    # AoU-shaped
])
def test_one_pass_matches_reference(n_samples: int, n_variants: int, n_cov: int) -> None:
    raw, cov, y = _make_synthetic(n_samples, n_variants, n_cov, seed=7 + n_cov)
    view = _DenseStandardizedView(raw)
    active = np.arange(n_variants, dtype=np.int32)
    got = compute_marginal_z_scores(
        standardized_genotypes=view,
        active_variant_indices=active,
        covariate_matrix=cov,
        target_vector=y,
    )
    expected = _reference_z_scores(raw, active, cov, y)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


def test_chunked_matches_whole() -> None:
    """The chunking wrapper used in model.py computes z per chunk and
    concatenates. The result must equal a whole-set computation."""
    raw, cov, y = _make_synthetic(n_samples=500, n_variants=300, n_covariates=8, seed=99)
    view = _DenseStandardizedView(raw)
    active = np.arange(300, dtype=np.int32)
    whole = compute_marginal_z_scores(
        standardized_genotypes=view,
        active_variant_indices=active,
        covariate_matrix=cov,
        target_vector=y,
    )
    chunk_size = 50
    chunked = np.empty(active.size, dtype=np.float32)
    for c_start in range(0, active.size, chunk_size):
        c_stop = min(c_start + chunk_size, active.size)
        chunked[c_start:c_stop] = compute_marginal_z_scores(
            standardized_genotypes=view,
            active_variant_indices=active[c_start:c_stop],
            covariate_matrix=cov,
            target_vector=y,
        )
    np.testing.assert_allclose(chunked, whole, rtol=1e-5, atol=1e-6)


def test_active_subset_independence() -> None:
    """The z-score for a given variant depends only on its own data + targets +
    covariates, not on the rest of the active set. Subsetting must not change
    per-variant values."""
    raw, cov, y = _make_synthetic(n_samples=400, n_variants=200, n_covariates=6, seed=11)
    view = _DenseStandardizedView(raw)
    full = compute_marginal_z_scores(
        standardized_genotypes=view,
        active_variant_indices=np.arange(200, dtype=np.int32),
        covariate_matrix=cov,
        target_vector=y,
    )
    # Pick a scattered subset.
    sub_idx = np.array([5, 17, 88, 142, 199], dtype=np.int32)
    sub = compute_marginal_z_scores(
        standardized_genotypes=view,
        active_variant_indices=sub_idx,
        covariate_matrix=cov,
        target_vector=y,
    )
    np.testing.assert_allclose(sub, full[sub_idx], rtol=1e-5, atol=1e-6)


def test_zero_target_returns_zeros() -> None:
    """Defensive: sigma2_resid == 0 (all-zero residuals) should yield zeros."""
    raw, cov, _ = _make_synthetic(n_samples=200, n_variants=50, n_covariates=3, seed=3)
    view = _DenseStandardizedView(raw)
    y_zero = np.zeros(200, dtype=np.float32)
    got = compute_marginal_z_scores(
        standardized_genotypes=view,
        active_variant_indices=np.arange(50, dtype=np.int32),
        covariate_matrix=cov,
        target_vector=y_zero,
    )
    assert np.allclose(got, 0.0)
