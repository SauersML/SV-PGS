"""Alias groups' laws (sv_pgs/alias_laws.py) against enumeration of their members' node tuples: log Z, the members'
node marginals, posterior means and variances, and the empirical Bayes gradient's pieces against finite differences.
Synthetic data only."""

import itertools

import numpy as np
import pytest

from sv_pgs.alias_laws import GroupLaws, group_terms, law_resolution


def _enumerate(log_weight, log_variance, precision, shift):
    """(log Z, marginals, means, variances) of one group by its node tuples (members x K inputs)."""
    count, nodes = log_weight.shape
    terms, tuples = [], []
    for combination in itertools.product(range(nodes), repeat=count):
        v = np.exp(log_variance[np.arange(count), combination])
        total = v.sum()
        log_k = -0.5 * np.log1p(total * precision) + 0.5 * shift**2 * total / (1 + total * precision)
        terms.append(np.sum(log_weight[np.arange(count), combination]) + log_k)
        tuples.append((combination, v, total))
    terms = np.array(terms)
    peak = terms.max()
    weights = np.exp(terms - peak)
    z = weights.sum()
    weights /= z
    marginal = np.zeros((count, nodes))
    first, second = np.zeros(count), np.zeros(count)
    for weight, (combination, v, total) in zip(weights, tuples):
        conditional = total / (1 + total * precision)
        mean_t = shift * conditional
        for j in range(count):
            marginal[j, combination[j]] += weight
            share = v[j] / total
            first[j] += weight * share * mean_t
            second[j] += weight * (v[j] * (total - v[j]) / total + share**2 * (conditional + mean_t**2))
    return peak + np.log(z), marginal, first, second - first**2


def _laws(log_scale, classes, log_density, grid, groups, refinement=16):
    return GroupLaws.of(np.asarray(groups), log_density, np.asarray(classes), np.asarray(log_scale), grid, refinement)


@pytest.mark.parametrize("groups", [[0], [0, 0]])
def test_singles_and_pairs_are_exact(groups):
    grid = np.log([1e-3, 1e-2, 1e-1, 1.0])
    log_density = np.log([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]])
    classes = [0, 1][: len(groups)]
    log_scale = [0.0, np.log(3.0)][: len(groups)]
    laws = _laws(log_scale, classes, log_density, grid, groups)
    precision, shift = 5.0, 4.0
    terms = group_terms(laws, np.array([precision]), np.array([shift]))
    rows = np.arange(len(groups))
    expected = _enumerate(log_density[classes], np.asarray(log_scale)[:, None] + grid[None, :], precision, shift)
    np.testing.assert_allclose(terms.log_normalizer[0], expected[0], rtol=1e-12)
    np.testing.assert_allclose(terms.marginal[rows], expected[1], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(terms.mean[rows], expected[2], rtol=1e-10)
    np.testing.assert_allclose(terms.variance[rows], expected[3], rtol=1e-9)


def test_a_larger_groups_splitting_error_falls_as_the_spacings_square_to_the_enumeration():
    grid = np.linspace(-6.0, 1.0, 15)
    log_density = (-0.5 * np.square((grid + 2.5) / 1.5))[None, :]
    log_density -= np.log(np.exp(log_density).sum())
    log_scale = np.array([0.0, 0.3, -0.2])
    precision, shift = np.array([4.0]), np.array([3.0])
    expected = _enumerate(log_density[[0, 0, 0]], log_scale[:, None] + grid[None, :], precision[0], shift[0])
    errors = []
    for refinement in (1, 2, 4, 8, 16):
        laws = GroupLaws.of(np.zeros(3, dtype=np.int64), log_density, np.zeros(3, dtype=np.int64), log_scale, grid, refinement)
        terms = group_terms(laws, precision, shift)
        errors.append(abs(terms.log_normalizer[0] - expected[0]))
    print("log Z error by refinement", errors)
    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
    assert errors[-1] < errors[0] / 50
    np.testing.assert_allclose(terms.mean, expected[2], rtol=2e-3)
    np.testing.assert_allclose(terms.variance, expected[3], rtol=5e-3)
    np.testing.assert_allclose(terms.marginal, expected[1], atol=2e-3)
    coarse = GroupLaws.of(np.zeros(3, dtype=np.int64), log_density, np.zeros(3, dtype=np.int64), log_scale, grid, 8)
    assert law_resolution(coarse, laws, precision, shift) > errors[-1]


@pytest.mark.parametrize("groups", [[0], [0, 0], [0, 0, 0]])
def test_the_gradient_pieces_are_the_derivatives_of_log_z(groups):
    """d log Z / d log u_j is ``scale_derivative``; d log Z / d log pi_ck (unnormalized weights) is the sum of class c's
    members' marginals at node k."""
    grid = np.linspace(-5.0, 1.0, 13)
    log_density = np.stack([-0.5 * np.square((grid + 2.0) / 1.2), -0.5 * np.square((grid + 1.0) / 1.0)])
    log_density -= np.log(np.exp(log_density).sum(axis=1, keepdims=True))
    count = len(groups)
    classes = np.array([0, 1, 0][:count])
    log_scale = np.array([0.0, 0.4, -0.3][:count])
    precision, shift = np.array([3.0]), np.array([2.5])
    terms = group_terms(_laws(log_scale, classes, log_density, grid, groups), precision, shift)
    step = 1e-6
    for j in range(count):
        up, down = log_scale.copy(), log_scale.copy()
        up[j] += step
        down[j] -= step
        numeric = (group_terms(_laws(up, classes, log_density, grid, groups), precision, shift).log_normalizer[0]
                   - group_terms(_laws(down, classes, log_density, grid, groups), precision, shift).log_normalizer[0]) / (2 * step)
        # The larger group's laws are split on nodes, so its identities agree with the split log Z to the splitting's resolution.
        np.testing.assert_allclose(terms.scale_derivative[j], numeric, rtol=1e-6 if count < 3 else 1e-2, atol=1e-8)
    for c in range(2):
        for k in (3, 7):
            up, down = log_density.copy(), log_density.copy()
            up[c, k] += step
            down[c, k] -= step
            numeric = (group_terms(_laws(log_scale, classes, up, grid, groups), precision, shift).log_normalizer[0]
                       - group_terms(_laws(log_scale, classes, down, grid, groups), precision, shift).log_normalizer[0]) / (2 * step)
            analytic = float(np.sum(terms.marginal[np.flatnonzero(classes == c), k]))
            np.testing.assert_allclose(analytic, numeric, rtol=1e-6 if count < 3 else 1e-2, atol=1e-8)
