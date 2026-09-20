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
