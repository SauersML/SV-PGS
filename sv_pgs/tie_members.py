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


def tied_groups(ties: TieGroups) -> list[I64Array]:
    """The members of every group with more than one member (singletons pass through exactly)."""
    sizes = np.bincount(ties.group, minlength=ties.group_count)
    order = np.argsort(ties.group, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    return [order[bounds[group] : bounds[group + 1]] for group in np.flatnonzero(sizes > 1)]


def _as_columns(values: F64Array) -> F64Array:
    values = np.asarray(values, dtype=np.float64)
    return values[:, None] if values.ndim == 1 else values


def tied_weights(precision: F64Array) -> tuple[F64Array, float, float]:
    """For one tied group's member precisions t_j: the weights w_j = D_j / D_g, and the group's precision
    t_g = 1 / D_g, computed from the ratios a_j = t_ref / t_j (|a_j| <= 1, t_ref the least |t_j|) so a near-flat member
    neither overflows nor cancels; with the group's variance's sign. The members' law restricted to the hyperplane
    s'beta = beta_g is proper exactly when every D_j > 0, or when one D_j < 0 and D_g < 0 (Haynsworth's inertia of
    diag(t) on s-perp; speed-smalln): any other sites raise LinAlgError, as a precision that is not positive definite
    does, so EP halves its negative sites."""
    negative = int(np.count_nonzero(precision < 0.0))
    if np.any(precision == 0.0):
        raise np.linalg.LinAlgError("a tied member's site is flat: the members' posterior is improper along their difference")
    reference = float(np.min(np.abs(precision)))
    ratios = reference / precision
    total = float(np.sum(ratios))
    if negative > 1 or (negative == 1 and not total < 0.0) or total == 0.0:
        raise np.linalg.LinAlgError("a tie group's member sites are not positive definite on their difference directions")
    return ratios / total, reference / total, total


def member_weights(ties: TieGroups, precision: F64Array) -> F64Array:
    """w_j = s_j D_j / D_g for every member at one model's sites (p_members,): each member's share of its group's
    posterior move (a singleton's is its sign)."""
    weights = np.array(ties.sign, copy=True)
    member_precision = np.asarray(precision, dtype=np.float64)
    for members in tied_groups(ties):
        weights[members] = ties.sign[members] * tied_weights(member_precision[members])[0]
    return weights


def group_sites(ties: TieGroups, precision: F64Array, shift: F64Array) -> tuple[F64Array, F64Array]:
    """The groups' sites (precision, shift) from the members': the law of beta_g = sum_j s_j beta_j under them,
    t_g = 1 / sum_j D_j and nu_g = t_g sum_j s_j nu_j / t_j, in natural parameters (a singleton's is its own, signed)."""
    shape = np.shape(precision)
    member_precision, member_shift = _as_columns(precision), _as_columns(shift)
    group_precision = np.zeros((ties.group_count, member_precision.shape[1]))
    group_shift = np.zeros_like(group_precision)
    group_precision[ties.group] = member_precision
    group_shift[ties.group] = ties.sign[:, None] * member_shift
    for members in tied_groups(ties):
        group = int(ties.group[members[0]])
        for column in range(member_precision.shape[1]):
            weights, precision_g, _total = tied_weights(member_precision[members, column])
            group_precision[group, column] = precision_g
            # nu_g = t_g sum_j s_j nu_j / t_j = sum_j s_j nu_j w_j t_g / t_j ... = sum_j s_j nu_j (D_j / D_g).
            group_shift[group, column] = float(np.sum(ties.sign[members] * member_shift[members, column] * weights))
    output_shape = (ties.group_count,) + tuple(shape[1:])
    return group_precision.reshape(output_shape), group_shift.reshape(output_shape)


def member_moments(
    ties: TieGroups, precision: F64Array, shift: F64Array, group_mean: F64Array, group_variance: F64Array
) -> tuple[F64Array, F64Array]:
    """Each member's posterior mean and variance from its group's (``group_mean``, ``group_variance``: the solver's
    posterior of beta_g), by conditioning on the sum: m_j = mu_j + s_j w_j (m_g - mu_g) and
    Var_j = D_j (1 - w_j) + w_j^2 Var_g with w_j = D_j / D_g (a singleton's are its group's, signed)."""
    shape = np.shape(precision)
    member_precision, member_shift = _as_columns(precision), _as_columns(shift)
    means, variances = _as_columns(group_mean), _as_columns(group_variance)
    member_mean = ties.sign[:, None] * means[ties.group]
    member_variance = np.array(variances[ties.group], copy=True)
    for members in tied_groups(ties):
        group = int(ties.group[members[0]])
        signs = ties.sign[members]
        for column in range(member_precision.shape[1]):
            weights, precision_g, _total = tied_weights(member_precision[members, column])
            site_means = member_shift[members, column] / member_precision[members, column]
            group_site_mean = float(np.sum(signs * site_means))
            member_mean[members, column] = site_means + signs * weights * (means[group, column] - group_site_mean)
            member_variance[members, column] = (1.0 - weights) / member_precision[members, column] + np.square(weights) * variances[group, column]
    if not (np.all(np.isfinite(member_mean)) and np.all(np.isfinite(member_variance))):
        raise np.linalg.LinAlgError("a tie group's member moments are not finite at these sites")
    return member_mean.reshape(shape), member_variance.reshape(shape)


def member_draws(
    ties: TieGroups, precision: F64Array, shift: F64Array, group_draws: F64Array, generator: np.random.Generator
) -> F64Array:
    """Exact posterior draws of every member (p_members, K) from draws of the groups' effects (p_groups, K): each
    group's members given its sum are the sites' Gaussian on the hyperplane sum_j s_j beta_j = beta_g, independent of
    the data, sampled by its Cholesky factor on the hyperplane (a singleton is its group's draw, signed)."""
    draws = np.asarray(group_draws, dtype=np.float64)
    member = draws[ties.group] * ties.sign[:, None]
    member_precision = np.asarray(precision, dtype=np.float64)
    member_shift = np.asarray(shift, dtype=np.float64)
    for members in tied_groups(ties):
        group = int(ties.group[members[0]])
        signs = ties.sign[members]
        weights, _precision_g, _total = tied_weights(member_precision[members])
        site_means = member_shift[members] / member_precision[members]
        # A basis of the hyperplane s'beta = 0 and the sites' precision restricted to it.
        basis = np.linalg.svd(signs[None, :])[2][1:].T
        restricted = basis.T @ (member_precision[members][:, None] * basis)
        factor = np.linalg.cholesky(0.5 * (restricted + restricted.T))
        # The conditional mean given the sum: the sites' mean moved along D s to meet it.
        base = site_means[:, None] + (signs * weights)[:, None] * (draws[group][None, :] - float(signs @ site_means))
        noise = np.linalg.solve(factor.T, generator.standard_normal((basis.shape[1], draws.shape[1])))
        member[members] = base + basis @ noise
    return member
