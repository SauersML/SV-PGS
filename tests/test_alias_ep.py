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
