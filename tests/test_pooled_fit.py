"""The pooled small-n fit (sv_pgs/pooled_fit.py): one gene is exactly the small-n fit, and several genes share one prior
with a learned level per gene. Machinery only (own simulations): accuracy comes from bench-real."""

import numpy as np
import pytest

from sv_pgs.config import VariantClass
from sv_pgs.pooled_fit import GeneData, _PooledPosterior, fit_pooled_small_n, pooled_prior
from sv_pgs.small_n import _DensePosterior, _Design, _Kernel, _new_profile, dense_statistics, fit_small_n

_SNV = list(VariantClass).index(VariantClass.SNV)
_DELETION = list(VariantClass).index(VariantClass.DELETION)


def _gene(rng, samples, variants, effects):
    frequency = rng.uniform(0.05, 0.5, variants)
    dosage = rng.binomial(2, frequency, size=(samples, variants))
    standardized = (dosage - dosage.mean(axis=0)) / dosage.std(axis=0)
    target = standardized @ effects + rng.standard_normal(samples)
    classes = np.where(np.arange(variants) % 7 == 0, _DELETION, _SNV).astype(np.uint8)
    return GeneData(codes=(dosage * 127).astype(np.uint8), covariates=np.ones((samples, 1)), target=target, variant_class=classes)


def _sparse_effects(rng, variants, count):
    effects = np.zeros(variants)
    effects[rng.choice(variants, size=count, replace=False)] = rng.normal(size=count)
    return effects


@pytest.mark.slow
def test_one_gene_is_exactly_the_small_n_fit():
    rng = np.random.default_rng(3)
    gene = _gene(rng, 150, 120, _sparse_effects(rng, 120, 3))
    pooled = fit_pooled_small_n([gene], draw_count=64, working_bytes=2 * 10**9, seed=5)
    single = fit_small_n(
        codes=gene.codes, covariates=gene.covariates, target=gene.target, variant_class=gene.variant_class, log_variance_offset=None,
        draw_count=64, working_bytes=2 * 10**9, seed=5,
    )
    np.testing.assert_array_equal(pooled.hyperparameters.coefficients, single.hyperparameters.coefficients)
    np.testing.assert_array_equal(pooled.scoring[0].coefficients, single.scoring.coefficients)
    np.testing.assert_array_equal(pooled.scoring[0].posterior_draws, single.scoring.posterior_draws)
    assert float(pooled.noise_variance[0]) == single.noise_variance


def test_the_pooled_posterior_is_the_direct_sum_of_the_genes():
    rng = np.random.default_rng(4)
    designs = [np.asfortranarray(rng.standard_normal((12, width))) for width in (20, 9)]
    precisions = [rng.uniform(0.5, 2.0, design.shape[1]) for design in designs]
    noises = [1.3, 0.7]
    posteriors = [
        _DensePosterior(_Kernel(_Design.dense(design), noise * precision), noise, 10**9, _new_profile())
        for design, precision, noise in zip(designs, precisions, noises)
    ]
    rows = (slice(0, 20), slice(20, 29))
    pooled = _PooledPosterior(posteriors, rows)
    covariance = np.zeros((29, 29))
    for design, precision, noise, gene_rows in zip(designs, precisions, noises, rows):
        covariance[gene_rows, gene_rows] = np.linalg.inv(design.T @ design / noise + np.diag(precision))
    right = rng.standard_normal((29, 3))
    rounding = np.sqrt(np.finfo(np.float64).eps)
    np.testing.assert_allclose(pooled.solve(right, 0.0), covariance @ right, rtol=rounding, atol=rounding)
    np.testing.assert_allclose(pooled.variance_jvp(right), -(covariance * covariance) @ right, rtol=rounding, atol=rounding)


def test_the_pooled_prior_stacks_the_genes_with_one_level_group():
    rng = np.random.default_rng(6)
    genes = [_gene(rng, 60, width, _sparse_effects(rng, width, 1)) for width in (30, 25, 40)]
    statistics = [dense_statistics(gene.codes, gene.covariates, gene.target) for gene in genes]
    prior = pooled_prior(statistics, [gene.variant_class for gene in genes], [np.zeros(gene.codes.shape[1]) for gene in genes], np.ones(3), 64)
    assert prior.variant_count == sum(int(gene.projected.shape[1]) for gene in statistics)
    assert prior.class_count == 2 and len(prior.annotation_groups) == 1 and prior.scale_size == 2


@pytest.mark.slow
def test_several_genes_fit_one_certified_prior():
    rng = np.random.default_rng(7)
    genes = [_gene(rng, 120, width, _sparse_effects(rng, width, count)) for width, count in ((80, 2), (60, 0), (100, 3))]
    fit = fit_pooled_small_n(genes, draw_count=64, working_bytes=2 * 10**9, seed=1)
    assert fit.certificate.remaining_gain[0] <= 0.5 / 64
    assert fit.certificate.prediction_move[0] <= fit.certificate.prediction_tolerance[0]
    assert len(fit.scoring) == 3 and fit.noise_variance.shape == (3,) and np.all(fit.noise_variance > 0.0)
    for gene, scoring in zip(genes, fit.scoring):
        assert np.all(np.isfinite(scoring.coefficients)) and scoring.posterior_draws.shape[1] == 64
        assert scoring.store_rows.max() < gene.codes.shape[1]
