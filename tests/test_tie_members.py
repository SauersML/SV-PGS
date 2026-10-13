"""Tied members keep their own sites: the grouped solve plus conditioning equals the dense member posterior."""
from __future__ import annotations

import numpy as np

from sv_pgs.data import TieGroup, TieMap
from sv_pgs.tie_members import TieGroups, group_sites, member_draws, member_moments


def _problem(seed: int):
    generator = np.random.default_rng(seed)
    samples, groups = 60, 5
    reduced = generator.standard_normal((samples, groups))
    # Members: group 0 alone, group 1 a copy pair, group 2 a triple with one negated copy, groups 3 and 4 alone.
    group = np.array([0, 1, 1, 2, 2, 2, 3, 4])
    sign = np.array([1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0])
    members = reduced[:, group] * sign[None, :]
    precision = generator.uniform(0.5, 4.0, group.shape[0])
    shift = generator.normal(0.0, 1.0, group.shape[0])
    target = generator.standard_normal(samples)
    return reduced, members, group, sign, precision, shift, target


def _posterior(design, precision, shift, target):
    matrix = design.T @ design + np.diag(precision)
    covariance = np.linalg.inv(matrix)
    return covariance @ (design.T @ target + shift), covariance


def test_the_grouped_solve_and_conditioning_give_the_dense_member_posterior():
    reduced, members, group, sign, precision, shift, target = _problem(3)
    ties = TieGroups(group=group, sign=sign, group_count=reduced.shape[1])
    dense_mean, dense_covariance = _posterior(members, precision, shift, target)
    group_precision, group_shift = group_sites(ties, precision, shift)
    reduced_mean, reduced_covariance = _posterior(reduced, group_precision, group_shift, target)
    mean, variance = member_moments(ties, precision, shift, reduced_mean, np.diag(reduced_covariance))
    np.testing.assert_allclose(mean, dense_mean, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(variance, np.diag(dense_covariance), rtol=1e-10, atol=1e-12)
    # Several models at once (columns).
    both = member_moments(ties, np.column_stack([precision, precision]), np.column_stack([shift, shift]),
                          np.column_stack([reduced_mean, reduced_mean]), np.column_stack([np.diag(reduced_covariance)] * 2))
    np.testing.assert_allclose(both[0][:, 1], dense_mean, rtol=1e-10, atol=1e-12)


def test_member_draws_sum_to_their_groups_and_have_the_member_covariance():
    reduced, members, group, sign, precision, shift, target = _problem(5)
    ties = TieGroups(group=group, sign=sign, group_count=reduced.shape[1])
    dense_mean, dense_covariance = _posterior(members, precision, shift, target)
    group_precision, group_shift = group_sites(ties, precision, shift)
    reduced_mean, reduced_covariance = _posterior(reduced, group_precision, group_shift, target)
    generator = np.random.default_rng(7)
    count = 40000
    group_draws = reduced_mean[:, None] + np.linalg.cholesky(reduced_covariance) @ generator.standard_normal((reduced.shape[1], count))
    draws = member_draws(ties, precision, shift, group_draws, generator)
    # Exactly: every group's signed member sum is its draw.
    sums = np.zeros_like(group_draws)
    np.add.at(sums, group, sign[:, None] * draws)
    np.testing.assert_allclose(sums, group_draws, rtol=1e-12, atol=1e-12)
    # In distribution: the dense member posterior, to Monte Carlo error (5 standard errors).
    empirical = np.cov(draws)
    scale = np.sqrt(np.outer(np.diag(dense_covariance), np.diag(dense_covariance)))
    assert np.max(np.abs(empirical - dense_covariance) / scale) <= 5.0 * np.sqrt(2.0 / count)
    assert np.max(np.abs(draws.mean(axis=1) - dense_mean) / np.sqrt(np.diag(dense_covariance))) <= 5.0 / np.sqrt(count)


def test_from_tie_map_reads_groups_and_signs():
    tie_map = TieMap(
        kept_indices=np.array([0, 1, 3], dtype=np.int32),
        original_to_reduced=np.array([0, 1, 1, 2], dtype=np.int32),
        reduced_to_group=[
            TieGroup(representative_index=0, member_indices=np.array([0], dtype=np.int32), signs=np.array([1.0], dtype=np.float32)),
            TieGroup(representative_index=1, member_indices=np.array([1, 2], dtype=np.int32), signs=np.array([1.0, -1.0], dtype=np.float32)),
            TieGroup(representative_index=3, member_indices=np.array([3], dtype=np.int32), signs=np.array([1.0], dtype=np.float32)),
        ],
    )
    ties = TieGroups.from_tie_map(tie_map)
    np.testing.assert_array_equal(ties.group, [0, 1, 1, 2])
    np.testing.assert_array_equal(ties.sign, [1.0, 1.0, -1.0, 1.0])
    assert ties.group_count == 3


def test_the_drivers_member_posterior_is_the_dense_member_posterior():
    # full_data_fit's responses over the members (Sigma R and -(Sigma o Sigma) W, for the total curvature) from the
    # solver's over the groups.
    from sv_pgs.full_data_fit import _member_posterior
    from sv_pgs.scale_mixture_ep import GaussianPosterior

    reduced, members, group, sign, precision, shift, _target = _problem(9)
    ties = TieGroups(group=group, sign=sign, group_count=reduced.shape[1])
    group_precision, _group_shift = group_sites(ties, precision, shift)
    reduced_covariance = np.linalg.inv(reduced.T @ reduced + np.diag(group_precision))
    member_covariance = np.linalg.inv(members.T @ members + np.diag(precision))
    posterior = _member_posterior(
        GaussianPosterior(solve=lambda right, _e: reduced_covariance @ right, variance_jvp=lambda weights: -(np.square(reduced_covariance) @ weights)),
        ties, precision, np.diag(reduced_covariance),
    )
    generator = np.random.default_rng(11)
    right = generator.standard_normal((group.shape[0], 3))
    np.testing.assert_allclose(posterior.solve(right, 1e-12), member_covariance @ right, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(posterior.variance_jvp(right), -(np.square(member_covariance) @ right), rtol=1e-9, atol=1e-12)


def test_a_genome_scale_tie_map_is_built_and_read_in_linear_numpy():
    # Stage 0's map over millions of active variants with exact ties among them (rare variants carried by the same
    # samples tie under standardization). A Python object per group held about 530 bytes per member and took about 20 s
    # for these 2M; the arrays hold 24 bytes per member at their peak and take a fraction of one sort of the members.
    import time
    import tracemalloc

    from sv_pgs.tie_map import tie_map_from_representatives

    count = 2_000_000
    generator = np.random.default_rng(5)
    representative = np.arange(count, dtype=np.int64)
    tied = np.arange(1, count, 4)
    representative[tied] = tied - 1
    sign = np.ones(count, dtype=np.float32)
    sign[tied] = generator.choice([-1.0, 1.0], size=tied.shape[0])

    def build() -> TieGroups:
        return TieGroups.from_tie_map(tie_map_from_representatives(representative, sign))

    def seconds(work) -> float:
        times = []
        for _ in range(3):
            started = time.perf_counter()
            work()
            times.append(time.perf_counter() - started)
        return min(times)

    shuffled = generator.permutation(count)
    assert seconds(build) < seconds(lambda: np.sort(shuffled))
    tracemalloc.start()
    try:
        before = len(tracemalloc.take_snapshot().traces)
        tracemalloc.reset_peak()
        ties = build()
        _current, peak = tracemalloc.get_traced_memory()
        objects = len(tracemalloc.take_snapshot().traces) - before
    finally:
        tracemalloc.stop()
    assert peak <= 32 * count and objects < 100
    assert ties.group_count == count - tied.shape[0]
    np.testing.assert_array_equal(ties.group[tied], ties.group[tied - 1])
    np.testing.assert_array_equal(ties.sign, sign)
    group = tie_map_from_representatives(representative, sign).reduced_to_group[int(ties.group[tied[-1]])]
    assert group.representative_index == tied[-1] - 1
    np.testing.assert_array_equal(group.member_indices, [tied[-1] - 1, tied[-1]])
    np.testing.assert_array_equal(group.signs, [1.0, sign[tied[-1]]])
