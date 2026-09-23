"""Alias EP with joint cluster sites (sv_pgs/alias_ep.py): on a synthetic design its fixed point is one, every unit's
tilted law matching q's marginal to the resolution, and a cluster's site is the exact joint tilted law's Gaussian
projection divided by its cavity (the pair's law atoms enumerated). Synthetic data only."""

import numpy as np

from sv_pgs.alias_ep import AliasEP, _gaussian_kl
from sv_pgs.alias_laws import GroupLaws
from sv_pgs.scale_mixture_ep import class_log_density, log_scale
from sv_pgs.scale_sampler import NodePrior
from tests.test_scale_mixture_ep import _hyperparameters, _problem

_DRAWS = 64


def _engine(seed):
    """Twelve groups (one pair of aliased members, the rest single) on n' = 30 rows, two of the groups' columns
    nearly collinear and carrying the signal."""
    prior, _cavity = _problem(variant_count=13, seed=seed, node_count=8)
    hyperparameters = _hyperparameters(prior, seed)
    groups = np.concatenate([[0], np.arange(12)])
    generator = np.random.default_rng(seed)
    rows = generator.standard_normal((12, 30))
    rows[1] = rows[2] + 0.05 * generator.standard_normal(30)
    target = 0.4 * rows[2] + generator.standard_normal(30)
    noise = 1.0
    laws = GroupLaws.of(groups, class_log_density(prior, hyperparameters.coefficients), prior.class_index,
                        log_scale(prior, hyperparameters.coefficients), prior.log_variance_grid, 4)
    nodes = NodePrior.of(prior, hyperparameters.coefficients)
    engine = AliasEP(rows, rows @ target, laws, nodes, groups, noise, _DRAWS, seed)
    second = np.array([float(np.exp(laws.law_log_mass[a:b]) @ np.exp(laws.law_log_variance[a:b]))
                       for a, b in zip(laws.law_start[:-1], laws.law_start[1:])])
    return engine, noise / second


def test_the_fixed_point_matches_every_units_tilted_law():
    engine, start = _engine(5)
    fit = engine.fit(start, np.zeros(start.shape[0]))
    assert fit.converged
    assert np.all(np.isfinite(fit.mean)) and np.all(fit.variance > 0.0)
    residual, proper = engine._single_residuals(fit.precision, fit.shift)
    alone = ~engine.sweep.clustered
    assert proper[alone].all()
    assert residual[alone].sum() <= 2.0 * engine.resolution


def test_a_clusters_site_is_its_exact_joint_tilted_laws_projection():
    engine, start = _engine(7)
    fit = engine.fit(start, np.zeros(start.shape[0]), clusters=[np.array([1, 2])])
    assert fit.converged
    assert any(set(cluster.tolist()) >= {1, 2} for cluster in fit.clusters)
    for index, cluster in enumerate(engine.sweep.clusters):
        cavity_precision, cavity_shift, mean, covariance = engine.sweep.cluster_cavity(index, fit.precision, fit.shift)
        proper, tilted_mean, tilted_covariance, floor = engine.cluster_tilted(cluster, cavity_precision, cavity_shift, 0)
        assert proper
        assert _gaussian_kl(tilted_mean, tilted_covariance, mean, covariance) <= engine.resolution + floor
    assert np.all(np.isfinite(fit.mean))


def test_the_enumerated_cluster_law_is_the_mixture_of_its_tuples_gaussians_and_stays_finite_at_a_flat_cavity():
    """``alias_groups.cluster_tilted`` against the tuples' Gaussian mixture formed densely, and at a near-flat cavity
    with a large pull (a variance past double precision among the atoms) finite with a positive definite covariance."""
    from itertools import product

    from sv_pgs.alias_groups import cluster_tilted

    counts = np.array([3, 2], dtype=np.int64)
    log_weight = np.log(np.array([0.5, 0.3, 0.2, 0.6, 0.4]))
    log_variance = np.array([-4.0, -1.0, 1.0, -2.0, 0.5])
    precision = np.array([[3.0, 2.5], [2.5, 3.0]])
    shift = np.array([1.2, -0.4])
    proper, _log_z, mean, covariance = cluster_tilted(counts, log_weight, log_variance, precision, shift)
    assert proper
    weights, means, covariances = [], [], []
    for first, second in product(range(3), range(3, 5)):
        variances = np.exp(log_variance[[first, second]])
        matrix = np.diag(1.0 / variances) + precision
        inverse = np.linalg.inv(matrix)
        location = inverse @ shift
        log_term = (log_weight[first] + log_weight[second] - 0.5 * (np.sum(np.log(variances)) + np.linalg.slogdet(matrix)[1])
                    + 0.5 * shift @ location)
        weights.append(log_term)
        means.append(location)
        covariances.append(inverse)
    weights = np.exp(np.array(weights) - max(weights))
    weights /= weights.sum()
    expected_mean = np.einsum("t,ti->i", weights, np.array(means))
    centred = np.array(means) - expected_mean
    expected_covariance = np.einsum("t,tij->ij", weights, np.array(covariances)) + np.einsum("t,ti,tj->ij", weights, centred, centred)
    np.testing.assert_allclose(mean, expected_mean, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(covariance, expected_covariance, rtol=1e-11, atol=1e-14)
    flat = np.array([[1e-12, 0.0], [0.0, 2.0]])
    log_variance_far = log_variance.copy()
    log_variance_far[2] = 800.0
    proper, _log_z, mean, covariance = cluster_tilted(counts, log_weight, log_variance_far, flat, np.array([80.0, 1.0]))
    assert proper and np.all(np.isfinite(mean)) and np.all(np.isfinite(covariance))
    assert np.linalg.eigvalsh(0.5 * (covariance + covariance.T))[0] > 0.0
