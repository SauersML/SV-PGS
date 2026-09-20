"""The pooled small-n fit (sv_pgs/pooled_fit.py): one gene is exactly the small-n fit, and several genes share one prior
with a learned level per gene. Machinery only (own simulations): accuracy comes from bench-real."""

import numpy as np
import pytest

from sv_pgs.config import VariantClass
from sv_pgs.pooled_fit import GeneData, _PooledPosterior, fit_pooled_small_n, pooled_prior
from sv_pgs.small_n import _Design, _Kernel, _new_profile, dense_statistics, fit_small_n

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


def _two_genes(rng):
    designs = [np.asfortranarray(rng.standard_normal((12, width))) for width in (20, 9)]
    precisions = [rng.uniform(0.5, 2.0, design.shape[1]) for design in designs]
    noises = np.array([1.3, 0.7])
    kernels = [_Kernel(_Design.dense(design), noise * precision) for design, precision, noise in zip(designs, precisions, noises)]
    rows = (slice(0, 20), slice(20, 29))
    covariance = np.zeros((29, 29))
    for design, precision, noise, gene_rows in zip(designs, precisions, noises, rows):
        covariance[gene_rows, gene_rows] = np.linalg.inv(design.T @ design / noise + np.diag(precision))
    return kernels, noises, rows, covariance


@pytest.mark.parametrize("share", [10**9, 2 * 8 * 20**2, 2 * 8 * 20**2 + 2 * 8 * 9**2])
def test_the_pooled_posterior_is_the_direct_sum_of_the_genes_at_any_residency(share):
    rng = np.random.default_rng(4)
    kernels, noises, rows, covariance = _two_genes(rng)
    pooled = _PooledPosterior(kernels, noises, rows, share, _new_profile())
    right = rng.standard_normal((29, 3))
    rounding = np.sqrt(np.finfo(np.float64).eps)
    np.testing.assert_allclose(pooled.solve(right, 0.0), covariance @ right, rtol=rounding, atol=rounding)
    squared = covariance * covariance
    np.testing.assert_allclose(pooled.variance_jvp(right), -squared @ right, rtol=rounding, atol=rounding)
    left, response_right, diagonal = rng.uniform(0.1, 0.5, 29), rng.uniform(0.1, 0.5, 29), rng.uniform(-0.2, 0.2, 29)
    weight = rng.uniform(0.0, 1.0, 29)
    system = np.eye(29) - (np.eye(29) - np.diag(weight) @ squared) @ (np.diag(left) @ covariance @ np.diag(response_right) + np.diag(diagonal))
    posterior = pooled.gaussian_posterior()
    assert posterior.linear_response is not None
    for _call in range(2):  # a resident gene answers from its kept factor the second time
        np.testing.assert_allclose(
            posterior.linear_response(left, response_right, diagonal, weight, right), np.linalg.solve(system, right), rtol=rounding, atol=rounding
        )


def test_the_exact_response_needs_room_for_the_largest_gene():
    kernels, noises, rows, _covariance = _two_genes(np.random.default_rng(5))
    assert _PooledPosterior(kernels, noises, rows, 2 * 8 * 20**2 - 1, _new_profile()).gaussian_posterior().linear_response is None


def test_the_pooled_prior_stacks_the_genes_with_their_levels_as_offset_groups():
    rng = np.random.default_rng(6)
    genes = [_gene(rng, 60, width, _sparse_effects(rng, width, 1)) for width in (30, 25, 40)]
    statistics = [dense_statistics(gene.codes, gene.covariates, gene.target) for gene in genes]
    prior = pooled_prior(statistics, [gene.variant_class for gene in genes], [np.zeros(gene.codes.shape[1]) for gene in genes], np.ones(3), 64)
    assert prior.variant_count == sum(int(gene.projected.shape[1]) for gene in statistics)
    # The genes' levels are offset groups (review-mathbugs P2): no annotation group, G - 1 level coordinates.
    assert prior.class_count == 2 and len(prior.annotation_groups) == 0 and prior.level_size == 2 and prior.scale_size == 2


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


def test_a_gene_at_its_frozen_fixed_point_takes_no_step_while_the_others_move():
    """Regression (bench-real diag2 [real], 3 of 3 groups): a gene whose frozen update was below its sites' rounding was
    taken for one whose damped passes all failed, and refused the whole pool."""
    from sv_pgs.pooled_fit import _PooledFixedPoints, _gene_rows, _pooled_start

    rng = np.random.default_rng(8)
    genes = [_gene(rng, 80, width, _sparse_effects(rng, width, count)) for width, count in ((40, 2), (30, 1))]
    statistics = [dense_statistics(gene.codes, gene.covariates, gene.target) for gene in genes]
    prior = pooled_prior(statistics, [gene.variant_class for gene in genes], [np.zeros(gene.codes.shape[1]) for gene in genes], np.ones(2), 64)
    start, noise = _pooled_start(statistics, prior, _gene_rows(statistics))
    oracle = _PooledFixedPoints(statistics, prior, start, noise, 64, 10**9)
    _variances, frozen = oracle._refresh(start)
    rows = oracle.rows
    target_precision, target_shift = oracle.site_precision.copy(), oracle.site_shift.copy()
    target_precision[rows[1]] *= 1.5  # gene 0's targets are its own sites: its update is exactly zero
    oracle._frozen_passes(start, frozen, target_precision, target_shift)  # refused before the fix
    assert np.all(np.isfinite(oracle.site_precision)) and np.all(np.isfinite(oracle.mean))


@pytest.mark.slow
def test_the_genes_curvature_blocks_add_up_to_the_pooled_curvature():
    from sv_pgs.pooled_fit import _PooledPosterior, _total_curvature, pooled_curvature_blocks
    from sv_pgs.scale_mixture_ep import Cavity

    rng = np.random.default_rng(9)
    genes = [_gene(rng, 100, width, _sparse_effects(rng, width, count)) for width, count in ((50, 2), (40, 1))]
    fit = fit_pooled_small_n(genes, draw_count=64, working_bytes=2 * 10**9, seed=2)
    curvature = pooled_curvature_blocks(fit, 2 * 10**9)
    oracle = fit.oracle
    (point,) = oracle([fit.hyperparameters])
    joint = _PooledPosterior(oracle.kernels, oracle.noise, oracle.rows, 10**9, _new_profile()).gaussian_posterior()
    relative = max(0.5 / 64 / fit.hyperparameters.coefficients.shape[0], np.finfo(np.float64).eps)
    pooled = _total_curvature(fit.prior, fit.hyperparameters.coefficients, point.cavity, joint, 10**9, relative)
    scale = np.max(np.abs(pooled))
    np.testing.assert_allclose(curvature.blocks.sum(axis=0), pooled, rtol=0.0, atol=np.sqrt(np.finfo(np.float64).eps) * scale)
    assert curvature.blocks.shape == (2,) + pooled.shape
    # Gene-owned levels (review-mathbugs P2): each gene's level block is rank one along its own basis row, exactly.
    from sv_pgs.pooled_fit import gene_owned_blocks
    from sv_pgs.scale_mixture_ep import _sum_to_zero_basis

    shared, coupling, level = gene_owned_blocks(curvature, fit.prior)
    basis = _sum_to_zero_basis(2)
    for gene in range(2):
        rebuilt = level[gene] * np.outer(basis[gene], basis[gene])
        np.testing.assert_allclose(curvature.blocks[gene, -1:, -1:], rebuilt, rtol=0.0, atol=np.sqrt(np.finfo(np.float64).eps) * scale)
        np.testing.assert_allclose(curvature.blocks[gene, :-1, -1:], np.outer(coupling[gene], basis[gene]), rtol=0.0, atol=np.sqrt(np.finfo(np.float64).eps) * scale)


def test_a_genes_double_loop_is_small_ns_on_its_rows():
    """The pooled fallback (MODEL.md section 4) runs small_n's double loop on the gene's own rows of the pooled prior:
    for one gene it is small_n's exactly."""
    from sv_pgs.pooled_fit import _PooledFixedPoints, _gene_rows, _pooled_start
    from sv_pgs.small_n import _DenseFixedPoints

    rng = np.random.default_rng(10)
    gene = _gene(rng, 60, 40, _sparse_effects(rng, 40, 2))
    statistics = [dense_statistics(gene.codes, gene.covariates, gene.target)]
    prior = pooled_prior(statistics, [gene.variant_class], [np.zeros(40)], np.ones(1), 64)
    start, noise = _pooled_start(statistics, prior, _gene_rows(statistics))
    pooled = _PooledFixedPoints(statistics, prior, start, noise, 64, 10**9)
    single = _DenseFixedPoints(statistics[0], prior, start, float(noise[0]), 64, 10**9)
    pooled._double_loop(0, start)
    single._double_loop(start)
    np.testing.assert_array_equal(pooled.site_precision, single.site_precision)
    np.testing.assert_array_equal(pooled.site_shift, single.site_shift)
    np.testing.assert_array_equal(pooled.mean, single.mean)


def test_non_finite_site_targets_refuse_the_trial_instead_of_looping():
    """review-mathbugs N2: a NaN target made the damped halving and the sweep loop run forever."""
    from sv_pgs.full_data_fit import NoFixedPoint
    from sv_pgs.pooled_fit import _PooledFixedPoints, _gene_rows, _pooled_start

    rng = np.random.default_rng(11)
    genes = [_gene(rng, 60, width, _sparse_effects(rng, width, 1)) for width in (30, 20)]
    statistics = [dense_statistics(gene.codes, gene.covariates, gene.target) for gene in genes]
    prior = pooled_prior(statistics, [gene.variant_class for gene in genes], [np.zeros(gene.codes.shape[1]) for gene in genes], np.ones(2), 64)
    start, noise = _pooled_start(statistics, prior, _gene_rows(statistics))
    oracle = _PooledFixedPoints(statistics, prior, start, noise, 64, 10**9)
    _variances, frozen = oracle._refresh(start)
    target_precision, target_shift = oracle.site_precision.copy(), oracle.site_shift.copy()
    target_precision[3] = np.nan
    with pytest.raises(NoFixedPoint, match="non-finite"):
        oracle._frozen_passes(start, frozen, target_precision, target_shift)


def test_every_row_of_a_gene_carries_exactly_its_level():
    """review-mathbugs P2 (lead ruling): gene levels are gene-owned offsets, sum-to-zero over genes, never class-centred.
    Two genes, each with SNV and deletion rows: every row of gene g must shift its log prior variance by l_g exactly."""
    from sv_pgs.scale_mixture_ep import _sum_to_zero_basis, log_scale

    rng = np.random.default_rng(12)
    genes = [_gene(rng, 60, width, _sparse_effects(rng, width, 1)) for width in (28, 14)]
    statistics = [dense_statistics(gene.codes, gene.covariates, gene.target) for gene in genes]
    prior = pooled_prior(statistics, [gene.variant_class for gene in genes], [np.zeros(gene.codes.shape[1]) for gene in genes], np.ones(2), 64)
    levels = np.array([1.0, -1.0])
    coefficients = np.zeros(prior.coefficient_size)
    coefficients[prior.coefficient_size - 1 :] = _sum_to_zero_basis(2).T @ levels
    shift = log_scale(prior, coefficients) - log_scale(prior, np.zeros(prior.coefficient_size))
    rows = [slice(0, statistics[0].design.variant_count), slice(statistics[0].design.variant_count, prior.variant_count)]
    for gene, gene_rows in enumerate(rows):
        np.testing.assert_allclose(shift[gene_rows], levels[gene], rtol=0.0, atol=np.sqrt(np.finfo(np.float64).eps))
