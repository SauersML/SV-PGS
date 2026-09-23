"""The alias groups' empirical Bayes objective (sv_pgs/alias_eb.py): with every member its own group it is the
per-variant objective (``scale_mixture_ep._data_objective`` less the penalty), value and gradient; with pairs and a
larger group its gradient is the derivative of its value. Synthetic data only."""

import numpy as np
import pytest

from sv_pgs.alias_eb import group_objective
from sv_pgs.scale_mixture_ep import _data_objective, _penalized, _penalty_matrix
from tests.test_scale_mixture_ep import _WORKING_BYTES, _hyperparameters, _problem


def test_singleton_groups_are_the_per_variant_objective():
    prior, cavity = _problem(variant_count=25, seed=9, node_count=8)
    hyperparameters = _hyperparameters(prior, 10)
    penalty = _penalty_matrix(prior, hyperparameters.log_smoothing)
    value, gradient, _hessian = _penalized(prior, _data_objective(prior, hyperparameters.coefficients, cavity, _WORKING_BYTES), hyperparameters.log_smoothing,
                                           penalty, hyperparameters.coefficients)
    objective = group_objective(prior, hyperparameters.coefficients, hyperparameters.log_smoothing, np.arange(prior.variant_count), cavity.precision,
                                cavity.shift, 1)
    np.testing.assert_allclose(objective.value, value, rtol=1e-12)
    np.testing.assert_allclose(objective.gradient, gradient, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("largest", [2, 3])
def test_the_gradient_is_the_derivative_of_the_value(largest):
    """Pairs are exact; a group of three is binned (refinement 16), whose identities hold to its splitting's
    resolution."""
    prior, cavity = _problem(variant_count=25, seed=9, node_count=8)
    hyperparameters = _hyperparameters(prior, 10)
    groups = np.repeat(np.arange(13), 2)[: prior.variant_count]
    if largest == 3:
        groups[2] = 0
        groups = np.unique(groups, return_inverse=True)[1]
    count = int(groups.max()) + 1
    precision, shift = cavity.precision[:count] / 2, cavity.shift[:count] / 2

    def value(point):
        return group_objective(prior, point, hyperparameters.log_smoothing, groups, precision, shift, 16).value

    gradient = group_objective(prior, hyperparameters.coefficients, hyperparameters.log_smoothing, groups, precision, shift, 16).gradient
    step = 1e-6
    numerical = np.array([
        (value(hyperparameters.coefficients + step * unit) - value(hyperparameters.coefficients - step * unit)) / (2 * step)
        for unit in np.eye(prior.coefficient_size)
    ])
    tolerance = 1e-6 if largest == 2 else 1e-2
    np.testing.assert_allclose(gradient, numerical, rtol=tolerance, atol=tolerance * np.max(np.abs(numerical)))
