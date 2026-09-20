"""Exactness of sv_pgs.fold_share's sharing identities against direct computation, on synthetic data."""
import numpy as np
import pytest
from scipy import linalg

from sv_pgs import fold_share

EPS = np.finfo(np.float64).eps


def _kernel(rng, samples, columns):
    genotypes = rng.integers(0, 3, size=(samples, columns)).astype(np.float64)
    weights = rng.gamma(0.5, size=columns)
    return genotypes, weights, (genotypes * weights) @ genotypes.T + np.eye(samples)


def _factor_bound(factor):
    """γ_{2n} ‖|L||Lᵀ|‖: Cholesky then QR, each backward stable (Higham 2002, Theorems 10.3 and 19.4)."""
    depth = 2 * factor.shape[0]
    return depth * EPS / (1 - depth * EPS) * np.linalg.norm(np.abs(factor) @ np.abs(factor).T)


@pytest.mark.parametrize("always_trained", [0, 7])
def test_fold_factors_reproduce_each_training_kernel(always_trained):
    rng = np.random.default_rng(1)
    samples = 60
    _, _, kernel = _kernel(rng, samples, 200)
    permutation = rng.permutation(samples)
    held = permutation[always_trained:]
    blocks = np.array_split(held, 5)
    blocks[1] = blocks[1][:3]  # unequal blocks, as leave-one-group-out gives
    folds = fold_share.FoldOrder.from_held_out(blocks, samples)
    full = linalg.cholesky(kernel[np.ix_(folds.order, folds.order)], lower=True)
    seen = 0
    for fold, factor in fold_share.fold_factors(full, folds):
        training = folds.training_order(fold)
        direct = kernel[np.ix_(training, training)]
        assert np.linalg.norm(factor @ factor.T - direct) <= _factor_bound(factor)
        assert np.allclose(np.tril(factor), factor)
        sign, logdet = np.linalg.slogdet(direct)
        assert sign > 0
        assert abs(fold_share.logdet_from_factor(factor) - logdet) <= _factor_bound(factor) * np.linalg.norm(linalg.inv(direct), 2)
        seen += 1
    assert seen == len(blocks)


def test_fold_order_rejects_overlapping_blocks():
    with pytest.raises(ValueError):
        fold_share.FoldOrder.from_held_out([np.array([0, 1]), np.array([1, 2])], 4)


def test_adding_nonnegative_columns_matches_the_refit():
    rng = np.random.default_rng(2)
    genotypes, weights, kernel = _kernel(rng, 50, 120)
    extra = rng.integers(0, 3, size=(50, 9)).astype(np.float64)
    extra_weights = rng.gamma(0.5, size=9)
    base = linalg.cholesky(kernel, lower=True)
    updated = fold_share.add_nonnegative_columns(base, extra, extra_weights)
    direct = kernel + (extra * extra_weights) @ extra.T
    assert np.linalg.norm(updated @ updated.T - direct) <= _factor_bound(updated)
    with pytest.raises(ValueError):
        fold_share.add_nonnegative_columns(base, extra, -extra_weights)


def test_signed_update_solves_and_logdet_are_certified():
    rng = np.random.default_rng(3)
    _, _, kernel = _kernel(rng, 40, 100)
    extra = rng.normal(size=(40, 6))
    base = linalg.cholesky(kernel, lower=True)
    gram = extra.T @ linalg.cho_solve((base, True), extra)
    # Negative weights that keep the update positive definite: c_j > −1/λ_max(G).
    signed = np.array([0.7, -0.4, 1.3, -0.2, 0.0, 0.5]) / np.linalg.eigvalsh(gram).max()
    update = fold_share.SignedColumnUpdate(base, extra, signed)
    direct = kernel + (extra * signed) @ extra.T
    right = rng.normal(size=(40, 3))
    result = update.solve(right)
    assert result.certified
    exact = np.linalg.solve(direct, right)
    assert np.linalg.norm(result.solution - exact) <= np.linalg.cond(direct) * result.rounding_bound / np.linalg.norm(direct, 2) + np.linalg.cond(direct) * EPS * np.linalg.norm(exact)
    sign, logdet = update.logdet()
    direct_sign, direct_logdet = np.linalg.slogdet(direct)
    assert sign == direct_sign
    assert abs(logdet - direct_logdet) <= np.linalg.cond(direct) * 40 * EPS * abs(direct_logdet) + 40 * EPS


def test_window_kernels_equal_direct_grams():
    rng = np.random.default_rng(4)
    samples, columns = 30, 400
    genotypes = rng.integers(0, 3, size=(samples, columns)).astype(np.float64)
    weights = rng.gamma(0.5, size=columns)
    starts = np.sort(rng.integers(0, columns - 60, size=12))
    windows = [np.arange(start, start + rng.integers(20, 120)) for start in starts]
    windows = [window[window < columns] for window in windows]
    windows[3] = np.concatenate([windows[3], [columns - 1]])  # a long SV straddling into a far window
    windows.append(np.array([], dtype=np.int64))
    emitted = {}
    for window, kernel in fold_share.window_kernels(lambda cols: genotypes[:, cols], weights, windows, samples):
        emitted[window] = kernel
    assert set(emitted) == set(range(len(windows)))
    for window, members in enumerate(windows):
        direct = (genotypes[:, members] * weights[members]) @ genotypes[:, members].T
        bound = (len(members) + len(windows)) * EPS * np.linalg.norm((np.abs(genotypes[:, members]) * weights[members]) @ np.abs(genotypes[:, members]).T)
        assert np.linalg.norm(emitted[window] - direct) <= bound
    with pytest.raises(ValueError):
        list(fold_share.window_kernels(lambda cols: genotypes[:, cols], -weights, windows, samples))


def test_segments_partition_every_window():
    windows = [np.array([0, 1, 2, 3]), np.array([2, 3, 4]), np.array([3, 9])]
    segments = fold_share.window_segments(windows)
    for window, members in enumerate(windows):
        covered = np.sort(np.concatenate([segment.columns for segment in segments if window in segment.windows]))
        assert np.array_equal(covered, np.sort(members))
    assert sum(segment.columns.shape[0] for segment in segments) == 6


def test_site_state_transfers_by_variant_identity():
    source = fold_share.transfer_site_state(
        np.array(["a", "b", "c"]), {"precision": np.array([1.0, 2.0, 3.0])},
        np.array(["c", "x", "a"]), {"precision": np.array([9.0, 9.0, 9.0])})
    assert np.array_equal(source["precision"], [3.0, 9.0, 1.0])


def test_common_weight_fold_kernels_are_submatrices():
    rng = np.random.default_rng(5)
    genotypes, weights, _ = _kernel(rng, 25, 80)
    full = (genotypes * weights) @ genotypes.T
    training = [np.arange(0, 20), np.arange(5, 25)]
    for fold, kernel in fold_share.common_weight_fold_kernels(full, training):
        rows = training[fold]
        direct = (genotypes[rows] * weights) @ genotypes[rows].T
        assert np.linalg.norm(kernel - direct) <= 80 * EPS * np.linalg.norm((np.abs(genotypes[rows]) * weights) @ np.abs(genotypes[rows]).T)


def test_sharing_cost_counts():
    cost = fold_share.sharing_cost(sample_count=10, held_out_sizes=[5, 5], window_sizes=[4, 4], unique_columns=6,
                                   segment_window_memberships=4, added_columns=[2])
    assert cost.window_direct == 100 * 8
    assert cost.window_shared == 100 * (6 + 4)
    assert cost.fold_kernel_direct == 2 * 25 * 4
    assert cost.fold_kernel_shared == 100 * 4
    assert cost.fold_factor_direct == 2 * 125 / 3
    assert cost.nested_shared == 100 * 2
    assert cost.ratios()["fold_factors"] < 1
