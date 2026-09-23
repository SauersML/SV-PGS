"""The empirical Bayes objective of the unified prior on the alias groups' sums at fixed EP cavities, with its exact
gradient: sum_g log Z_g (``alias_laws``) less the smoothing penalty, as a function of the prior's coefficients x.

At an EP fixed point the evidence's derivative in the prior's hyperparameters is the tilted normalizers' at the fixed
cavities (EP's stationarity in its sites), so this is the objective the outer search climbs between fixed points, as
``scale_mixture_ep._data_objective`` is per variant. In z = M x, with member j of class c at its group's cavity:

    d/d eta_ck = sum_{j in c} P(k_j = k | y) - n_c pi_ck,   d/d(scale coefficients) = D' (d log Z / d log u),

D the scale design, both from ``alias_laws.group_terms``; the penalty's value and gradient are
``scale_mixture_ep._penalty_value``'s."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sv_pgs._typing import F64Array, I64Array
from sv_pgs.alias_laws import GroupLaws, GroupTerms, group_terms
from sv_pgs.scale_mixture_ep import ScaleMixturePrior, _penalty_value, class_log_density, log_scale


@dataclass(frozen=True)
class GroupObjective:
    """The penalized objective at x, its gradient in x, and the laws and terms it was read from."""

    value: float
    gradient: F64Array
    data_value: float
    laws: GroupLaws
    terms: GroupTerms


def group_objective(
    prior: ScaleMixturePrior, coefficients: F64Array, log_smoothing: F64Array, groups: I64Array, cavity_precision: F64Array,
    cavity_shift: F64Array, refinement: int,
) -> GroupObjective:
    """sum_g log Z_g - 1/2 sum_i lambda_i ||R_i x_i||^2 and its gradient in x (module docstring); ``groups`` is each
    member's alias group, the cavities are on the groups' sums."""
    log_density = class_log_density(prior, coefficients)
    laws = GroupLaws.of(groups, log_density, prior.class_index, log_scale(prior, coefficients), prior.log_variance_grid, refinement)
    terms = group_terms(laws, cavity_precision, cavity_shift)
    gradient_z = np.zeros(prior.density_size + prior.scale_size)
    density = np.exp(log_density)
    for class_position, rows in enumerate(prior.class_rows):
        span = slice(class_position * prior.grid_size, (class_position + 1) * prior.grid_size)
        gradient_z[span] = terms.marginal[rows].sum(axis=0) - rows.shape[0] * density[class_position]
    if prior.scale_size:
        gradient_z[prior.density_size :] = prior.scale_design.T @ terms.scale_derivative
    penalty, penalty_gradient = _penalty_value(prior, log_smoothing, coefficients)
    data_value = float(np.sum(terms.log_normalizer))
    return GroupObjective(
        value=data_value - penalty, gradient=prior.coefficient_map.T @ gradient_z - penalty_gradient, data_value=data_value, laws=laws, terms=terms,
    )
