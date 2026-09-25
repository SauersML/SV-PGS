"""The power screen: the sites whose data cannot move the posterior mean by more than the scorer resolves.

For a prior member j with the fitted prior (class density pi_ck on the lattice t_k, scale u_j = e^{log u_j}, read
from ``scale_mixture_ep.class_log_density`` and ``log_scale``, so the learned frequency function and every other
annotation enter through u_j) and data precision w_j = x_j'x_j / sigma^2 (its standardized column's; an imputed
record's reliability is in its Var(D) and so in u_j), the effect's law given node k is N(0, c_jk), c_jk = u_j e^{t_k},
and its posterior mean from any data is a shrinkage lambda_jk = w_j c_jk / (1 + w_j c_jk) of its least squares.
Averaged over the prior predictive:

    E[m_j^2]            <=  s_j sum_k pi_ck c_jk lambda_jk            =: B_j   (per unit of the column's variance s_j)
    E[Var(beta_j | y)]  >=  s_j sum_k pi_ck c_jk (1 - lambda_jk)       =: E_j

(law of total variance, with the node conditioned on; both exact given the prior and w_j, whatever y is). B_j is the
Bayes risk of dropping j: the expected rise in the predictor's squared error when its posterior mean is replaced by
the prior's, 0. The scorer's K posterior draws resolve its mean only to the posterior variance over K (the
certificate's level, ``marginal_variances.certificate_level``), so a set D whose total risk is within
(1/K) sum_j E_j moves the prediction by less than the draws can see.

A site is one unbreakable group of records (its bubble, same-POS set or TR locus, ``VariantTable.group_first``) joined
with every record whose reference span overlaps it: a common multiallelic locus split into rare allele records is
one site, and its alleles' risks add. A site is dropped whole or kept whole, smallest total risk first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from sv_pgs._typing import BoolArray, F64Array, I64Array
from sv_pgs.scale_mixture_ep import MixtureHyperparameters, ScaleMixturePrior, class_log_density, log_scale

_FLOAT_BYTES = np.dtype(np.float64).itemsize


@dataclass(frozen=True)
class PowerScreen:
    """Per member: the risk bound ``risk`` (B_j), the variance bound ``spread`` (E_j) and its site; per site the summed
    risk; ``dropped`` marks the members of the dropped sites, and ``budget`` is (1/K) sum_j E_j."""

    risk: F64Array
    spread: F64Array
    site: I64Array
    site_risk: F64Array
    dropped: BoolArray
    budget: float

    def record(self) -> dict:
        return {
            "members": int(self.risk.shape[0]), "sites": int(self.site_risk.shape[0]),
            "dropped_members": int(self.dropped.sum()), "dropped_sites": int(np.unique(self.site[self.dropped]).shape[0]),
            "dropped_risk": float(self.risk[self.dropped].sum()), "budget": self.budget,
        }


def site_index(chromosome: np.ndarray, position: I64Array, ref_length: np.ndarray, group_first: I64Array) -> I64Array:
    """A site per record (0..S-1, in record order): records sharing ``group_first`` or whose reference spans
    [position, position + max(ref_length, 1)) overlap on one chromosome are one site. Records must be sorted by
    (chromosome, position), as a store's are."""
    chromosome = np.asarray(chromosome)
    position = np.asarray(position, dtype=np.int64)
    end = position + np.maximum(np.asarray(ref_length, dtype=np.int64), 1)
    count = position.shape[0]
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    if np.any((chromosome[1:] == chromosome[:-1]) & (position[1:] < position[:-1])):
        raise ValueError("records must be sorted by chromosome and position")
    # A record overlaps the span before it when it begins before the furthest end reached so far on its chromosome:
    # the running maximum of the ends, restarted at each chromosome.
    chromosome_start = np.r_[True, chromosome[1:] != chromosome[:-1]]
    reach = np.empty(count, dtype=np.int64)
    for first, stop in zip(np.flatnonzero(chromosome_start), np.r_[np.flatnonzero(chromosome_start)[1:], count]):
        reach[first:stop] = np.maximum.accumulate(end[first:stop])
    joined = np.zeros(count, dtype=bool)
    joined[1:] = ~chromosome_start[1:] & (position[1:] < reach[:-1])
    # Edges: each overlapping record to the one before it, and each record to its group's first record.
    rows = np.r_[np.flatnonzero(joined), np.arange(count)]
    columns = np.r_[np.flatnonzero(joined) - 1, np.asarray(group_first, dtype=np.int64)]
    graph = coo_matrix((np.ones(rows.shape[0]), (rows, columns)), shape=(count, count))
    _count, label = connected_components(graph, directed=False)
    # Sites numbered in order of their first record.
    _labels, first_record = np.unique(label, return_index=True)
    rank = np.empty(first_record.shape[0], dtype=np.int64)
    rank[np.argsort(first_record, kind="stable")] = np.arange(first_record.shape[0])
    return rank[label]


def power_screen(
    prior: ScaleMixturePrior, hyperparameters: MixtureHyperparameters, precision: F64Array, column_variance: F64Array,
    site: I64Array, draw_count: int, working_bytes: int,
) -> PowerScreen:
    """The screen at ``hyperparameters`` for members with data precision ``precision`` (w_j) and column variance
    ``column_variance`` (s_j, x_j'x_j / n), grouped by ``site`` (``site_index`` over the members)."""
    precision = np.asarray(precision, dtype=np.float64)
    column_variance = np.asarray(column_variance, dtype=np.float64)
    site = np.asarray(site, dtype=np.int64)
    count = prior.variant_count
    if precision.shape != (count,) or column_variance.shape != (count,) or site.shape != (count,):
        raise ValueError("precision, column variance and site need one entry per prior member")
    grid = np.asarray(prior.log_variance_grid, dtype=np.float64)
    log_density = class_log_density(prior, hyperparameters.coefficients)
    scales = log_scale(prior, hyperparameters.coefficients)
    risk = np.empty(count)
    spread = np.empty(count)
    # Three (rows x K) float64 intermediates per chunk.
    chunk = max(1, int(working_bytes) // (3 * _FLOAT_BYTES * grid.shape[0]))
    for first in range(0, count, chunk):
        rows = slice(first, min(count, first + chunk))
        variance = np.exp(scales[rows, None] + grid[None, :])
        product = precision[rows, None] * variance
        weighted = np.exp(log_density[prior.class_index[rows]]) * variance
        # lambda = wc / (1 + wc) and 1 - lambda = 1 / (1 + wc), each formed without cancellation.
        risk[rows] = column_variance[rows] * np.sum(weighted * product / (1.0 + product), axis=1)
        spread[rows] = column_variance[rows] * np.sum(weighted / (1.0 + product), axis=1)
    site_count = int(site.max()) + 1 if count else 0
    site_risk = np.bincount(site, weights=risk, minlength=site_count)
    budget = float(np.sum(spread)) / draw_count
    order = np.argsort(site_risk, kind="stable")
    cumulative = np.cumsum(site_risk[order])
    dropped_sites = order[: int(np.searchsorted(cumulative, budget, side="right"))]
    dropped_site = np.zeros(site_count, dtype=bool)
    dropped_site[dropped_sites] = True
    return PowerScreen(risk=risk, spread=spread, site=site, site_risk=site_risk, dropped=dropped_site[site], budget=budget)
