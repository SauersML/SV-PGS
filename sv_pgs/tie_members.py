"""Tied columns keep their own effects: members of a tie group (columns exactly equal, or negated, on the training
samples) each carry their own effect, EP site and class prior, and only the solver sees their signed sum (lead ruling
on review-mathbugs T1: a merged column took its lowest-index member's class prior, so an SV tied to an SNV was fitted
under the SNV's prior and the effect split b / M).

With independent member sites N(beta_j; mu_j, D_j) (D_j = 1 / tau_j, mu_j = nu_j / tau_j) and a group's column the
same x_g for every member up to its sign s_j, the likelihood sees beta_g = sum_j s_j beta_j only. So the solver takes
the group's site, the law of that sum under the member sites,
    D_g = sum_j D_j,    mu_g = sum_j s_j mu_j,
and each member's posterior follows from the group's by conditioning on the sum (members given beta_g are the sites'
Gaussian restricted to the hyperplane, independent of the data):
    E[beta_j] = mu_j + s_j (D_j / D_g) (E[beta_g] - mu_g),
    Var(beta_j) = D_j - D_j^2 / D_g + (D_j / D_g)^2 Var(beta_g),
    Cov(beta_j, beta_k) = -D_j D_k / D_g + s_j s_k (D_j D_k / D_g^2) Var(beta_g)   (j != k).
A singleton group is the identity. The effective number of effects sum_j (1 - tau_j Var(beta_j)) over a group equals
the group's own 1 - tau_g Var(beta_g), so the noise update is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sv_pgs._typing import F64Array, I64Array
from sv_pgs.data import TieMap


@dataclass(frozen=True)
class TieGroups:
    """``group[j]`` is member j's reduced column and ``sign[j]`` its sign (+1 a copy, -1 a negated copy), over the
    active rows in their order; ``group_count`` reduced columns."""

    group: I64Array
    sign: F64Array
    group_count: int

    @classmethod
    def from_tie_map(cls, tie_map: TieMap) -> TieGroups:
        group = np.asarray(tie_map.original_to_reduced, dtype=np.int64)
        if np.any(group < 0):
            raise ValueError("every member must be active: the tie map covers the active rows")
        sign = np.ones(group.shape[0])
        for tie_group in tie_map.reduced_to_group:
            sign[np.asarray(tie_group.member_indices, dtype=np.int64)] = np.asarray(tie_group.signs, dtype=np.float64)
        return cls(group=group, sign=sign, group_count=int(np.asarray(tie_map.kept_indices).shape[0]))

    @property
    def member_count(self) -> int:
        return int(self.group.shape[0])


def _group_sum(ties: TieGroups, values: F64Array) -> F64Array:
    values = np.asarray(values, dtype=np.float64)
    total = np.zeros((ties.group_count,) + values.shape[1:])
    np.add.at(total, ties.group, values)
    return total


def _column(ties: TieGroups, values: F64Array) -> F64Array:
    """``ties.sign`` shaped to broadcast against ``values`` ((p,) or (p, M))."""
    return ties.sign.reshape((-1,) + (1,) * (np.ndim(values) - 1))


def _site_moments(precision: F64Array, shift: F64Array) -> tuple[F64Array, F64Array]:
    """(D, mu) of sites (precision, shift); a flat site (precision 0) has D = inf, and must have shift 0."""
    precision = np.asarray(precision, dtype=np.float64)
    shift = np.asarray(shift, dtype=np.float64)
    flat = precision == 0.0
    if np.any(flat & (shift != 0.0)):
        raise ValueError("a flat site (precision 0) must have shift 0")
    with np.errstate(divide="ignore"):
        variance = np.where(flat, np.inf, 1.0 / np.where(flat, 1.0, precision))
    return variance, np.where(flat, 0.0, shift * variance)


def group_sites(ties: TieGroups, precision: F64Array, shift: F64Array) -> tuple[F64Array, F64Array]:
    """The groups' sites (precision, shift) from the members': the law of beta_g = sum_j s_j beta_j under them."""
    variance, mean = _site_moments(precision, shift)
    group_variance = _group_sum(ties, variance)
    group_mean = _group_sum(ties, _column(ties, mean) * mean)
    if np.any(group_variance == 0.0):
        raise ValueError("a tie group's member site variances sum to zero: its site has no finite precision")
    with np.errstate(divide="ignore"):
        group_precision = np.where(np.isinf(group_variance), 0.0, 1.0 / group_variance)
    return group_precision, group_mean * group_precision


def member_moments(
    ties: TieGroups, precision: F64Array, shift: F64Array, group_mean: F64Array, group_variance: F64Array
) -> tuple[F64Array, F64Array]:
    """Each member's posterior mean and variance from its group's (``group_mean``, ``group_variance``: the solver's
    posterior of beta_g), by conditioning on the sum."""
    variance, mean = _site_moments(precision, shift)
    total_variance = _group_sum(ties, variance)
    total_mean = _group_sum(ties, _column(ties, mean) * mean)
    ratio = variance / total_variance[ties.group]
    member_mean = mean + _column(ties, mean) * ratio * (np.asarray(group_mean)[ties.group] - total_mean[ties.group])
    member_variance = variance - variance * ratio + np.square(ratio) * np.asarray(group_variance)[ties.group]
    return member_mean, member_variance


def member_draws(
    ties: TieGroups, precision: F64Array, shift: F64Array, group_draws: F64Array, generator: np.random.Generator
) -> F64Array:
    """Exact posterior draws of every member (p_members, K) from draws of the groups' effects (p_groups, K): each
    group's members given its sum are the sites' Gaussian on the hyperplane sum_j s_j beta_j = beta_g, independent of
    the data, sampled by its Cholesky factor on the hyperplane (a singleton is its group's draw). Raises when that
    restriction is not positive definite (a negative site the sum cannot absorb)."""
    variance, mean = _site_moments(precision, shift)
    draws = np.asarray(group_draws, dtype=np.float64)
    member = draws[ties.group] * ties.sign[:, None]
    sizes = np.bincount(ties.group, minlength=ties.group_count)
    for group in np.flatnonzero(sizes > 1):
        members = np.flatnonzero(ties.group == group)
        signs = ties.sign[members]
        site_precision = 1.0 / variance[members]
        # A basis of the hyperplane s'beta = 0 and the sites' precision restricted to it.
        basis = np.linalg.svd(signs[None, :])[2][1:].T
        restricted = basis.T @ (site_precision[:, None] * basis)
        factor = np.linalg.cholesky(0.5 * (restricted + restricted.T))
        # The conditional mean given the sum: the sites' mean moved along D s to meet the sum.
        weights = variance[members] * signs / float(np.sum(variance[members]))
        base = mean[members][:, None] + weights[:, None] * (draws[group][None, :] - float(signs @ mean[members]))
        noise = np.linalg.solve(factor.T, generator.standard_normal((basis.shape[1], draws.shape[1])))
        member[members] = base + basis @ noise
    return member
