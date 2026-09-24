"""Sequential EP on the alias groups' sums (sv_pgs/sequential_ep.py, sv_pgs/alias_groups.py) against exact references:
one sweep against a dense reference that re-inverts A' before every update (with the domain's halving), the induced
prior and the decode against enumeration, and the brief's identical-proxy example, which compiled to its sum is exact.
Synthetic data only."""

import itertools

import numpy as np
import pytest

from sv_pgs.alias_groups import decode, group_priors, tilted_sum
from sv_pgs.sequential_ep import SequentialSweep

_HALF = np.finfo(np.float64).eps ** 0.5


def _components(priors, group):
    start, stop = priors.component_start[group], priors.component_start[group + 1]
    return priors.log_weight[start:stop], priors.log_variance[start:stop]


def _reference_sweep(gram, score, t, nu, noise, priors, largest, coupling=None, skip=(), residuals=None):
    """One sweep over the groups' sums: dense A' = X'X + diag t (+ a cluster's scaled off-diagonal ``coupling``)
    re-inverted before every update; a step that would make another unclustered cavity improper halves; ``skip``'s
    groups (a cluster's) keep their sites."""
    t, nu = t.copy(), nu.copy()
    coupling = np.zeros_like(gram) if coupling is None else coupling
    for group in range(t.shape[0]):
        if group in skip:
            continue
        inverse = np.linalg.inv(gram + np.diag(t) + coupling)
        mean = inverse @ (score + noise * nu)
        marginal = noise * inverse[group, group]
        cavity_precision = 1.0 / marginal - t[group] / noise
        cavity_shift = mean[group] / marginal - nu[group]
        proper, _log_z, tilted_mean, tilted_variance = tilted_sum(*_components(priors, group), cavity_precision, cavity_shift)
        if not proper:
            continue
        target_t = noise * (1.0 / tilted_variance - cavity_precision)
        target_nu = tilted_mean / tilted_variance - cavity_shift
        fraction = 1.0
        while True:
            trial_t, trial_nu = t.copy(), nu.copy()
            trial_t[group] = t[group] + fraction * (target_t - t[group])
            trial_nu[group] = nu[group] + fraction * (target_nu - nu[group])
            trial = np.linalg.inv(gram + np.diag(trial_t) + coupling)
            cavities = (1.0 / (noise * np.diag(trial)) - trial_t / noise)
            others = (np.arange(t.shape[0]) != group) & ~np.isin(np.arange(t.shape[0]), list(skip))
            valid = np.all((cavities[others] >= 0.0) | (1.0 + largest[others] * cavities[others] > 0.0))
            if valid:
                t, nu = trial_t, trial_nu
                if residuals is not None:
                    q_variance = 1.0 / (cavity_precision + t[group] / noise)
                    q_mean = q_variance * (cavity_shift + nu[group])
                    residuals[group] = 0.5 * (tilted_variance / q_variance + (q_mean - tilted_mean) ** 2 / q_variance - 1.0
                                              + np.log(q_variance / tilted_variance))
                break
            fraction *= 0.5
            if fraction * abs(target_t - t[group]) <= np.finfo(np.float64).eps * abs(t[group]):
                break
    return t, nu


def _problem(seed, samples, groups, members_of):
    rng = np.random.default_rng(seed)
    rows = rng.standard_normal((groups, samples))
    rows[1] = 0.9 * rows[0] + 0.1 * rows[1]  # a near proxy
    members = np.repeat(np.arange(groups), members_of)
    grid = np.linspace(-8.0, 2.0, 11)
    log_density = np.log(np.stack([np.linspace(1.0, 3.0, grid.shape[0]), np.linspace(3.0, 1.0, grid.shape[0])]))
    log_density -= np.log(np.exp(log_density).sum(axis=1, keepdims=True))
    class_index = rng.integers(0, 2, members.shape[0])
    log_scale = rng.normal(0.0, 0.3, members.shape[0])
    priors = group_priors(members, log_density, class_index, log_scale, grid)
    target = rows[0] * 0.8 + rows[2] * 0.4 + rng.standard_normal(samples)
    return rows, members, priors, target


@pytest.mark.parametrize("members_of", [1, 2])
def test_one_sweep_is_the_dense_sequential_update(members_of):
    rows, _members, priors, target = _problem(5 + members_of, 30, 20, members_of)
    noise = 0.8
    gram = rows @ rows.T
    score = rows @ target
    second = np.array([float(np.exp(_components(priors, g)[0]) @ np.exp(_components(priors, g)[1])) for g in range(rows.shape[0])])
    t, nu = noise / second, np.zeros(rows.shape[0])
    with np.errstate(over="ignore"):
        largest = np.exp(priors.largest_log_variance())
    expected_t, expected_nu = _reference_sweep(gram, score, t, nu, noise, priors, largest)
    sweep = SequentialSweep(rows, np.einsum("ij,ij->i", rows, rows), score, priors, noise)
    got_t, got_nu = t.copy(), nu.copy()
    assert sweep.run(got_t, got_nu, np.arange(rows.shape[0], dtype=np.int64)) is not None
    np.testing.assert_allclose(got_t, expected_t, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(got_nu, expected_nu, rtol=1e-8, atol=1e-10)


def test_the_induced_prior_of_a_pair_and_its_decode_are_the_enumeration():
    """A pair's sum law is its K^2 tuples; decoding at a cavity gives the members' exact posterior moments."""
    grid = np.log([1e-3, 1e-1, 1.0])
    log_density = np.log([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    log_scale = np.array([0.0, np.log(2.0)])
    priors = group_priors(np.array([0, 0]), log_density, np.array([0, 1]), log_scale, grid)
    precision, shift = 3.0, 2.5
    mean, variance, proper = decode(priors, np.array([precision]), np.array([shift]), 2)
    assert proper.all()
    # Enumeration: the members are independent scale mixtures; the data see their sum through exp(-P T^2/2 + h T).
    total, first, second = 0.0, np.zeros(2), np.zeros(2)
    for i, j in itertools.product(range(3), range(3)):
        v = np.array([np.exp(log_scale[0] + grid[i]), np.exp(log_scale[1] + grid[j])])
        weight = np.exp(log_density[0, i] + log_density[1, j])
        precision_matrix = np.diag(1.0 / v) + precision
        covariance = np.linalg.inv(precision_matrix)
        m = covariance @ np.full(2, shift)
        evidence = np.sqrt(np.linalg.det(covariance) / np.prod(v)) * np.exp(0.5 * np.full(2, shift) @ m)
        total += weight * evidence
        first += weight * evidence * m
        second += weight * evidence * (np.diag(covariance) + m * m)
    np.testing.assert_allclose(mean, first / total, rtol=1e-10)
    np.testing.assert_allclose(variance, second / total - (first / total) ** 2, rtol=1e-9)


def test_identical_proxies_compiled_to_their_sum_are_exact():
    """The brief's example (100 samples, two identical columns of +-1 x 50, y = 1.5 x, noise 1, priors 0.95 N(0, 1e-5)
    + 0.05 N(0, 1)): exact posterior means 0.742691 each. Compiled to their sum, the group is one coordinate with one
    non-Gaussian factor, so EP on it is exact and the decode gives both members exactly equal means."""
    x = np.repeat([1.0, -1.0], 50)
    grid = np.log([1e-5, 1.0])
    priors = group_priors(np.array([0, 0]), np.log([[0.95, 0.05]]), np.array([0, 0]), np.zeros(2), grid)
    score = np.array([150.0])
    second = float(np.exp(priors.log_weight) @ np.exp(priors.log_variance))
    t, nu = np.array([1.0 / second]), np.zeros(1)
    sweep = SequentialSweep(x[None, :], np.array([100.0]), score, priors, 1.0)
    for _ in range(3):
        assert sweep.run(t, nu, np.array([0])) == 0
    cavity_precision, cavity_shift = sweep.cavities(t, nu)
    np.testing.assert_allclose([cavity_precision[0], cavity_shift[0]], [100.0, 150.0], rtol=1e-9)
    mean, _variance, _proper = decode(priors, cavity_precision, cavity_shift, 2)
    assert mean[0] == mean[1]
    np.testing.assert_allclose(mean, 0.742691, atol=5e-7)


def test_a_larger_groups_convolved_law_decodes_close_to_the_enumeration():
    """A group of three: the member-by-member convolution on the lattice's spacing (a quadrature in log variance, as the
    lattice itself) against the full K^3 enumeration; its members' means agree to the quadrature's resolution, and
    exchangeable members come out exactly equal."""
    grid = np.linspace(-6.0, 1.0, 15)
    log_density = np.log(np.exp(-0.5 * np.square((grid + 2.0) / 1.5)))[None, :]
    log_density -= np.log(np.exp(log_density).sum())
    priors = group_priors(np.array([0, 0, 0]), log_density, np.array([0, 0, 0]), np.zeros(3), grid)
    precision, shift = 4.0, 3.0
    mean, _variance, _proper = decode(priors, np.array([precision]), np.array([shift]), 3)
    assert mean[0] == mean[1] == mean[2]
    total, first = 0.0, 0.0
    weights = np.exp(log_density[0])
    for i, j, k in itertools.product(range(grid.shape[0]), repeat=3):
        v = np.exp(grid[[i, j, k]])
        variance_sum = v.sum()
        weight = weights[i] * weights[j] * weights[k] * np.exp(-0.5 * np.log1p(variance_sum * precision) + 0.5 * shift**2 * variance_sum / (1 + variance_sum * precision))
        total += weight
        first += weight * v[0] / variance_sum * shift * variance_sum / (1 + variance_sum * precision)
    np.testing.assert_allclose(mean[0], first / total, rtol=2e-2)


def test_a_group_whose_cavity_integral_is_infinite_is_reported_not_decoded():
    grid = np.log([1e-3, 1e3])
    priors = group_priors(np.array([0, 1]), np.log([[0.5, 0.5]]), np.array([0, 0]), np.zeros(2), grid)
    mean, variance, proper = decode(priors, np.array([1.0, -1e-2]), np.array([0.5, 0.5]), 2)
    assert proper.tolist() == [True, False]
    assert np.isfinite(mean[0]) and np.isnan(mean[1]) and np.isnan(variance[1])


def test_a_cluster_site_is_the_dense_block_and_the_sweep_moves_around_it():
    """A cluster of three groups with a joint site (a full block on their sums): its cavity and every other group's
    are the dense A' = X'X + sites' (Sigma_C^-1 less the block), and one sweep of the other groups is the dense
    sequential update with the block held."""
    rows, _members, priors, target = _problem(11, 30, 20, 1)
    noise = 0.8
    gram = rows @ rows.T
    score = rows @ target
    second = np.array([float(np.exp(_components(priors, g)[0]) @ np.exp(_components(priors, g)[1])) for g in range(rows.shape[0])])
    t, nu = noise / second, np.zeros(rows.shape[0])
    cluster = np.array([0, 1, 5])
    rng = np.random.default_rng(3)
    factor = rng.standard_normal((3, 3))
    block = factor @ factor.T / 3 + np.diag(1.0 / second[cluster])
    block_shift = rng.standard_normal(3)
    sweep = SequentialSweep(rows, np.einsum("ij,ij->i", rows, rows), score, priors, noise)
    sweep.set_clusters([cluster])
    sweep.set_cluster_site(0, block, block_shift, t, nu)
    scaled_coupling = np.zeros_like(gram)
    scaled_coupling[np.ix_(cluster, cluster)] = noise * (block - np.diag(np.diag(block)))
    inverse = np.linalg.inv(gram + np.diag(t) + scaled_coupling)
    covariance = noise * inverse
    mean = inverse @ (score + noise * nu)
    cavity_precision, cavity_shift, cluster_mean, cluster_covariance = sweep.cluster_cavity(0, t, nu)
    expected_inverse = np.linalg.inv(covariance[np.ix_(cluster, cluster)])
    np.testing.assert_allclose(cluster_covariance, covariance[np.ix_(cluster, cluster)], rtol=1e-9)
    np.testing.assert_allclose(cluster_mean, mean[cluster], rtol=1e-9)
    np.testing.assert_allclose(cavity_precision, expected_inverse - block, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(cavity_shift, expected_inverse @ mean[cluster] - block_shift, rtol=1e-8, atol=1e-8)
    single_precision, single_shift = sweep.cavities(t, nu)
    others = np.setdiff1d(np.arange(rows.shape[0]), cluster)
    np.testing.assert_allclose(single_precision[others], 1.0 / np.diag(covariance)[others] - t[others] / noise, rtol=1e-8)
    np.testing.assert_allclose(single_shift[others], mean[others] / np.diag(covariance)[others] - nu[others], rtol=1e-8, atol=1e-10)
    with np.errstate(over="ignore"):
        largest = np.exp(priors.largest_log_variance())
    expected_t, expected_nu = _reference_sweep(gram, score, t, nu, noise, priors, largest, scaled_coupling, set(cluster.tolist()))
    got_t, got_nu = t.copy(), nu.copy()
    assert sweep.run(got_t, got_nu, np.arange(rows.shape[0], dtype=np.int64)) is not None
    np.testing.assert_allclose(got_t, expected_t, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(got_nu, expected_nu, rtol=1e-8, atol=1e-10)
    assert sweep.valid(got_t, got_nu)


def test_the_blocked_sweep_is_the_per_step_sweep():
    """Blocks with their checks deferred to one product (restored and redone step by step where a state is not
    proper) give the per-step sweep's sites, over several sweeps of p >> n' with a tight near-proxy cluster."""
    rng = np.random.default_rng(17)
    rows, _members, priors, _target = _problem(17, 40, 200, 1)
    rows[3:12] = rows[2] + 0.02 * rng.standard_normal((9, 40))
    target = 1.5 * rows[2] + 0.6 * rows[40] + rng.standard_normal(40)
    noise = 0.5
    score = rows @ target
    squares = np.einsum("ij,ij->i", rows, rows)
    second = np.array([float(np.exp(_components(priors, g)[0]) @ np.exp(_components(priors, g)[1])) for g in range(rows.shape[0])])
    order = np.arange(rows.shape[0], dtype=np.int64)
    results = []
    for width in (1, None):
        sweep = SequentialSweep(rows, squares, score, priors, noise)
        if width is None:
            width = sweep.block_width
            assert width > 1
        sweep.block_width = width
        t, nu = noise / second, np.zeros(rows.shape[0])
        refused = [sweep.run(t, nu, order) for _ in range(6)]
        assert all(count is not None for count in refused)
        results.append((t, nu, refused))
    np.testing.assert_allclose(results[1][0], results[0][0], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(results[1][1], results[0][1], rtol=1e-9, atol=1e-12)
    assert results[1][2] == results[0][2]


def test_a_cluster_step_by_woodbury_is_the_rebuilt_state():
    """``step_cluster``'s rank-m update of S^-1, every I and the rest's mean gives the cavities a fresh build gives at
    the new site."""
    rows, _members, priors, target = _problem(23, 30, 20, 1)
    noise = 0.8
    score = rows @ target
    second = np.array([float(np.exp(_components(priors, g)[0]) @ np.exp(_components(priors, g)[1])) for g in range(rows.shape[0])])
    t, nu = noise / second, np.zeros(rows.shape[0])
    sweep = SequentialSweep(rows, np.einsum("ij,ij->i", rows, rows), score, priors, noise)
    sweep.set_clusters([np.array([0, 1, 4])])
    assert sweep.run(t, nu, np.arange(rows.shape[0], dtype=np.int64)) is not None
    before = sweep.current_cluster_cavity(0, t, nu)
    rng = np.random.default_rng(1)
    factor = rng.standard_normal((3, 3))
    site = (np.diag(t[[0, 1, 4]]) + sweep.coupling[0]) / noise + 0.1 * (factor @ factor.T)
    site_shift = nu[[0, 1, 4]] + rng.standard_normal(3)
    assert sweep.step_cluster(0, site, site_shift, t, nu)
    assert sweep.blocking.size == 0
    stepped = sweep.current_cluster_cavity(0, t, nu)
    stepped_informed = sweep.informed.copy()
    rebuilt = sweep.cluster_cavity(0, t, nu)
    for got, expected in zip(stepped, rebuilt):
        np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(stepped_informed, sweep.informed, rtol=1e-8, atol=1e-12)
    assert not np.allclose(before[2], stepped[2])


def test_each_steps_own_residual_is_the_tilted_laws_kl_from_the_new_marginal():
    """``SequentialSweep.own_residual``: after each group's step, the KL of its tilted law from q's marginal at its new
    site, zero for a full step and positive for a damped one, as the dense reference computes it."""
    rows, _members, priors, target = _problem(31, 20, 60, 1)
    rows[5:15] = rows[4] + 0.01 * np.random.default_rng(2).standard_normal((10, 20))
    target = 2.0 * rows[4] + np.random.default_rng(3).standard_normal(20)
    noise = 0.5
    gram = rows @ rows.T
    score = rows @ target
    second = np.array([float(np.exp(_components(priors, g)[0]) @ np.exp(_components(priors, g)[1])) for g in range(rows.shape[0])])
    t, nu = noise / second, np.zeros(rows.shape[0])
    with np.errstate(over="ignore"):
        largest = np.exp(priors.largest_log_variance())
    expected = np.full(rows.shape[0], np.nan)
    _reference_sweep(gram, score, t, nu, noise, priors, largest, residuals=expected)
    sweep = SequentialSweep(rows, np.einsum("ij,ij->i", rows, rows), score, priors, noise)
    assert sweep.run(t.copy(), nu.copy(), np.arange(rows.shape[0], dtype=np.int64)) is not None
    swept = ~np.isnan(expected)
    np.testing.assert_allclose(sweep.own_residual[swept], expected[swept], rtol=1e-6, atol=1e-9)
    assert np.all(expected[swept] >= -1e-12)
