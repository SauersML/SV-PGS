"""Stage 1 (LD-space EB fit) against exact references.

- The grid tilted moments against adaptive quadrature of the continuous
  BetaPrime prior, and the M-step derivatives against finite differences.
- The whole quantitative fit, from sufficient statistics only, against the same
  scheme run on the individual-level data: dense X̃ᵀX̃, the full inverse and the
  same fixed noise variance rᵀr / (n − q). The design is built so that the
  projected cross-block LD is exactly zero, so the block-diagonal LD is exact
  and the two paths must agree to rounding error, for p < n and for p > n.
- A binary trait after the mean-weight fit plus one weight refresh, against the
  individual-level Laplace-EP fit with exact per-sample weights.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from scipy import integrate
from scipy.special import betaln, digamma, expit

from sv_pgs import ld_space_fit
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import ModelConfig, VariantClass
from sv_pgs.mixture_inference import _build_prior_design
from sv_pgs.ld_space_fit import (
    ExpectationPropagationSites,
    InMemoryLDBlocks,
    LDPriorHypermodel,
    QuantitativeTraitStatistics,
    binary_statistics_at,
    fit_ld_space,
    gig_expected_log,
    hypermodel_from_prior_design,
    inverse_digamma,
    quantitative_trait_statistics,
)
from sv_pgs.genotype_statistics import compute_genotype_statistics
from tests.conftest import make_variant_records
from tests.stage0_support import InMemoryTileSource, bubble_groups, mosaic_codes

SCHEMES = ("expectation_propagation", "coherent_vb", "plug_in")
CPU_BUDGET = ComputeBudget(
    device_kind="cpu",
    device_ids=(),
    device_names=(),
    device_bytes=(),
    device_compute_capabilities=(),
    host_bytes=4 * 1024**3,
    cpu_threads=4,
)
HOST = ld_space_fit._HostBackend(ThreadPoolExecutor(max_workers=2), 2, CPU_BUDGET.host_bytes)


def test_host_block_posterior_matches_the_dense_inverse() -> None:
    rng = np.random.default_rng(4)
    width = 700
    samples = rng.standard_normal((2000, width)) @ np.triu(rng.uniform(0.0, 0.2, size=(width, width)))
    correlation = samples.T @ samples / 2000.0
    correlation = 0.5 * (correlation + correlation.T)
    precision = rng.uniform(1.0, 1e5, size=width)
    linear = rng.standard_normal(width)
    system = 1500.0 * correlation + np.diag(precision)
    inverse = np.linalg.inv(system)
    mean, variance = HOST.block_posterior(correlation, 1500.0, precision, linear)
    np.testing.assert_allclose(mean, inverse @ linear, rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(variance, np.diag(inverse), rtol=1e-9)
    with pytest.raises(np.linalg.LinAlgError):
        HOST.block_posterior(np.array([[1.0, 2.0], [2.0, 1.0]]), 1.0, np.zeros(2), np.ones(2))


def test_anderson_extrapolation_is_capped_on_the_log_scale() -> None:
    # x -> 1 + 0.5 x has its fixed point at 2: Anderson reaches it from two steps.
    acceleration = ld_space_fit.AndersonState(memory_depth=3)
    ld_space_fit._accelerated_hyperparameters(acceleration, np.array([0.0]), np.array([1.0]))
    accelerated = ld_space_fit._accelerated_hyperparameters(acceleration, np.array([1.0]), np.array([1.5]))
    np.testing.assert_allclose(accelerated, [2.0])
    # x -> 1 + 0.999 x would extrapolate to 1000, far beyond the cap: plain step, fresh history.
    acceleration = ld_space_fit.AndersonState(memory_depth=3)
    ld_space_fit._accelerated_hyperparameters(acceleration, np.array([0.0]), np.array([1.0]))
    capped = ld_space_fit._accelerated_hyperparameters(acceleration, np.array([1.0]), np.array([1.999]))
    np.testing.assert_array_equal(capped, [1.999])
    assert not acceleration.residuals


def test_hypermodel_from_prior_design_maps_classes_and_precisions() -> None:
    records = make_variant_records(30) + make_variant_records(10, VariantClass.DELETION_SHORT)
    prior_design = _build_prior_design(records)
    config = ModelConfig()
    offset = np.log(np.linspace(0.4, 1.0, 40))
    hypermodel = hypermodel_from_prior_design(prior_design, config, offset, shape_a=0.5)
    assert hypermodel.class_names == ("deletion_short", "snv")
    np.testing.assert_array_equal(hypermodel.variant_class_index, np.r_[np.ones(30), np.zeros(10)])
    np.testing.assert_array_equal(hypermodel.initial_shape_b, [config.class_tpb_shape_b()[VariantClass.DELETION_SHORT], 0.5])
    # The one type-offset column carries the type-offset penalty, quartered on the log-variance scale.
    np.testing.assert_array_equal(hypermodel.annotation_prior_precision, [config.type_offset_penalty / 4.0])
    np.testing.assert_array_equal(hypermodel.log_variance_offset, offset)


def test_host_worker_count_is_bounded_by_threads_and_memory() -> None:
    assert ld_space_fit._host_worker_count(CPU_BUDGET, 100) == CPU_BUDGET.cpu_threads
    one_worker_bytes = ld_space_fit._HOST_BLOCK_WORKER_MATRICES * 8 * 4096 * 4096
    assert ld_space_fit._host_worker_count(CPU_BUDGET, 4096) == min(CPU_BUDGET.cpu_threads, CPU_BUDGET.host_bytes // one_worker_bytes)
    with pytest.raises(MemoryError):
        ld_space_fit._host_worker_count(CPU_BUDGET, 20000)


def _initial_model(scheme, ld_blocks, statistics, hypermodel):
    return ld_space_fit._initial_model_state(scheme, ld_blocks.ld_diagonal(), ld_blocks.ld_scores(), statistics, hypermodel)


# --------------------------------------------------------------------------- data


def _haplotype_block(rng: np.random.Generator, sample_count: int, width: int) -> np.ndarray:
    """Dosages of `width` variants in LD (thresholded AR(1) haplotypes)."""
    correlation = rng.uniform(0.7, 0.95)
    frequency = rng.uniform(0.05, 0.5, size=width)
    threshold = np.quantile(rng.standard_normal(100_000), 1.0 - frequency)
    dosage = np.zeros((sample_count, width))
    for _haplotype in range(2):
        latent = np.empty((sample_count, width))
        latent[:, 0] = rng.standard_normal(sample_count)
        for column in range(1, width):
            latent[:, column] = correlation * latent[:, column - 1] + np.sqrt(1.0 - correlation**2) * rng.standard_normal(sample_count)
        dosage += latent > threshold[None, :]
    return dosage


def _orthogonal_block_design(
    rng: np.random.Generator, train_count: int, test_count: int, block_widths: list[int], covariate_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Covariates W and genotype blocks, each regressed on W and every earlier block.

    The regression coefficients come from the first ``train_count`` rows and are
    applied to every row, so the result is one linear map of the dosages. On the
    training rows the LD within a block is kept and X̃_bᵀX̃_c = 0 across blocks
    exactly, which is the Stage 1 model. Columns get unit training variance.
    """
    sample_count = train_count + test_count
    covariates = np.column_stack([np.ones(sample_count), rng.standard_normal((sample_count, covariate_count - 1))])
    predictors = covariates
    blocks = []
    for width in block_widths:
        block = _haplotype_block(rng, sample_count, width)
        for _reorthogonalization in range(2):
            coefficients, *_ = np.linalg.lstsq(predictors[:train_count], block[:train_count], rcond=None)
            block = block - predictors @ coefficients
        block /= np.sqrt(np.mean(block[:train_count] ** 2, axis=0))[None, :]
        blocks.append(block)
        predictors = np.column_stack([predictors, block])
    boundaries = np.concatenate([[0], np.cumsum(block_widths)]).astype(np.int64)
    return covariates, np.column_stack(blocks), boundaries


def _hypermodel(variant_count: int, class_index: np.ndarray, rng: np.random.Generator) -> LDPriorHypermodel:
    sv_indicator = (class_index == 1).astype(np.float64)
    annotation = rng.standard_normal(variant_count)
    design = np.column_stack([sv_indicator - sv_indicator.mean(), annotation - annotation.mean()])
    return LDPriorHypermodel(
        annotation_design=design,
        annotation_prior_mean=np.zeros(2),
        annotation_prior_precision=np.array([1.0, 4.0]),
        log_variance_offset=np.log(rng.uniform(0.5, 1.0, size=variant_count)),
        variant_class_index=class_index,
        class_names=("snv", "sv"),
        shape_a=0.5,
        initial_shape_b=np.array([0.5, 0.4]),
        shape_b_pooling_variance=0.25,
    )


def _sparse_effects(rng: np.random.Generator, variant_count: int, class_index: np.ndarray, causal_count: int) -> np.ndarray:
    effects = np.zeros(variant_count)
    causal = rng.choice(variant_count, causal_count, replace=False)
    effects[causal] = rng.standard_normal(causal_count) * np.where(class_index[causal] == 1, 2.0, 1.0)
    return effects


def _ld_blocks(genotypes: np.ndarray, boundaries: np.ndarray) -> InMemoryLDBlocks:
    sample_count = genotypes.shape[0]
    blocks = []
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        block = genotypes[:, start:stop].T @ genotypes[:, start:stop] / sample_count
        blocks.append(0.5 * (block + block.T))
    return InMemoryLDBlocks(block_boundaries=boundaries, correlation_blocks=tuple(blocks))


# --------------------------------------------------------------------------- individual-level reference


def _reference_quantitative_fit(
    scheme_name: str,
    genotypes: np.ndarray,
    residual: np.ndarray,
    hypermodel: LDPriorHypermodel,
    warm_model: ld_space_fit._ModelState,
    pass_count: int,
) -> ld_space_fit._ModelState:
    """The same scheme on individual-level data with dense linear algebra."""
    scheme = ld_space_fit._local_scheme(scheme_name)
    model = warm_model
    variant_count = genotypes.shape[1]
    gram = genotypes.T @ genotypes
    everything = slice(0, variant_count)
    for _pass in range(pass_count):
        grid = ld_space_fit._device_grid(HOST, hypermodel, model)
        block_prior = ld_space_fit._block_prior(HOST, model, hypermodel, grid, everything)
        block_state = scheme.load_block(HOST, model.local, everything)
        for _local_iteration in range(ld_space_fit._LOCAL_ITERATIONS_PER_PASS):
            precision, shift = scheme.solve_terms(HOST, block_state, block_prior)
            system = model.likelihood_precision * gram + np.diag(precision)
            covariance = np.linalg.inv(system)
            mean = covariance @ (model.likelihood_precision * (genotypes.T @ residual) + shift)
            block_state = scheme.update_block(HOST, block_state, mean, np.diag(covariance).copy(), block_prior)
        scheme.store_block(HOST, model.local, everything, block_state)
        model.posterior_mean = mean
        model.posterior_variance = np.diag(covariance).copy()
        ld_space_fit._update_scale_model(scheme, HOST, model, hypermodel, [everything])
        ld_space_fit._update_shapes(scheme, HOST, model, hypermodel, [everything])
    return model


def _quantitative_problem(seed: int, sample_count: int, block_widths: list[int]):
    rng = np.random.default_rng(seed)
    covariate_count = 3
    covariates, genotypes, boundaries = _orthogonal_block_design(rng, sample_count, 0, block_widths, covariate_count)
    variant_count = genotypes.shape[1]
    class_index = (rng.uniform(size=variant_count) < 0.15).astype(np.int64)
    effects = _sparse_effects(rng, variant_count, class_index, causal_count=max(variant_count // 20, 4))
    genetic = genotypes @ effects
    effects *= np.sqrt(0.4 / np.var(genetic))
    phenotype = covariates @ np.array([2.0, 0.5, -0.3]) + genotypes @ effects + np.sqrt(0.6) * rng.standard_normal(sample_count)
    coefficients, *_ = np.linalg.lstsq(covariates, phenotype, rcond=None)
    residual = phenotype - covariates @ coefficients
    statistics = QuantitativeTraitStatistics(
        sample_count=sample_count,
        covariate_count=covariate_count,
        score=genotypes.T @ residual,
        residual_sum_of_squares=float(residual @ residual),
    )
    return (
        genotypes,
        residual,
        covariate_count,
        boundaries,
        statistics,
        _hypermodel(variant_count, class_index, rng),
        genotypes @ effects,
    )


EXACTNESS_CASES = [
    pytest.param(scheme_name, sample_count, block_widths, id=f"{scheme_name}-{label}")
    for scheme_name in SCHEMES
    for sample_count, block_widths, label in [(600, [40, 25, 60, 35], "p_below_n_four_blocks"), (150, [230], "p_above_n_one_block")]
    # EP refuses a block with p_b >= n - q (see test_ep_refuses_a_block_wider_than_the_sample_rank).
    if not (scheme_name == "expectation_propagation" and sum(block_widths) > sample_count)
]


@pytest.mark.parametrize(("scheme_name", "sample_count", "block_widths"), EXACTNESS_CASES)
def test_quantitative_fit_matches_individual_level_scheme(scheme_name, sample_count, block_widths, monkeypatch) -> None:
    genotypes, residual, _covariate_count, boundaries, statistics, hypermodel, _genetic = _quantitative_problem(
        seed=7, sample_count=sample_count, block_widths=block_widths
    )
    ld_blocks = _ld_blocks(genotypes, boundaries)
    scheme = ld_space_fit._local_scheme(scheme_name)
    reference = _reference_quantitative_fit(
        scheme_name,
        genotypes,
        residual,
        hypermodel,
        _initial_model(scheme, ld_blocks, statistics, hypermodel),
        pass_count=12,
    )
    # Exactly the reference's 12 plain passes: no early stop and no acceleration.
    monkeypatch.setattr(ld_space_fit, "_MAXIMUM_PASSES", 12)
    monkeypatch.setattr(ld_space_fit, "_CONVERGENCE_TOLERANCE", 0.0)
    monkeypatch.setattr(ld_space_fit, "anderson_step", lambda state, *, x_current, map_value: map_value)
    (fit,) = fit_ld_space(ld_blocks, [statistics], hypermodel, CPU_BUDGET, scheme_name=scheme_name)
    # Rounding enters the Newton M-steps' stopping points, so agreement is ~1e-8, not 1e-15.
    np.testing.assert_allclose(fit.log_variance_level, reference.log_variance_level, rtol=1e-7, atol=1e-6)
    np.testing.assert_allclose(fit.annotation_coefficients, reference.annotation_coefficients, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(fit.shape_b, reference.shape_b, rtol=1e-6)
    scale = np.max(np.abs(reference.posterior_mean))
    np.testing.assert_allclose(fit.posterior_mean, reference.posterior_mean, rtol=0.0, atol=1e-6 * scale)
    np.testing.assert_allclose(fit.posterior_variance, reference.posterior_variance, rtol=1e-6)


def test_quantitative_fit_converges_and_recovers_the_signal() -> None:
    genotypes, _residual, _covariates, boundaries, statistics, hypermodel, genetic = _quantitative_problem(
        seed=3, sample_count=800, block_widths=[50, 50, 50, 50]
    )
    (fit,) = fit_ld_space(_ld_blocks(genotypes, boundaries), [statistics], hypermodel, CPU_BUDGET)
    assert fit.converged and fit.passes < 30
    assert fit.likelihood_precision == (statistics.sample_count - statistics.covariate_count) / statistics.residual_sum_of_squares
    assert np.corrcoef(genotypes @ fit.posterior_mean, genetic)[0, 1] ** 2 > 0.9


# --------------------------------------------------------------------------- 1-D numerics


def _betaprime_tilted_quadrature(cavity_mean, cavity_variance, prior_variance, shape_a, shape_b):
    """log Z, E[β] and Var[β] for the continuous BetaPrime(a, b) scale mixture times N(β; m, v)."""

    def log_prior(log_scale):
        return shape_a * log_scale - (shape_a + shape_b) * np.logaddexp(0.0, log_scale) - betaln(shape_a, shape_b)

    def integrand(log_scale, power):
        slab = prior_variance * np.exp(log_scale)
        total = cavity_variance + slab
        shrinkage = slab / total
        density = np.exp(log_prior(log_scale) - 0.5 * (np.log(2 * np.pi * total) + cavity_mean**2 / total))
        return density * [1.0, cavity_mean * shrinkage, cavity_mean**2 * shrinkage**2 + cavity_variance * shrinkage][power]

    edges = sorted([-200.0, np.log(cavity_variance / prior_variance), np.log(max(cavity_mean**2, cavity_variance) / prior_variance), 200.0])
    moments = []
    for power in range(3):
        total = 0.0
        for lower, upper in zip(edges[:-1], edges[1:], strict=True):
            piece, _error = integrate.quad(lambda log_scale: integrand(log_scale, power), lower, upper, limit=4000, epsabs=0.0, epsrel=1e-13)
            total += piece
        moments.append(total)
    normalizer, first, second = moments
    return np.log(normalizer), first / normalizer, second / normalizer - (first / normalizer) ** 2


@pytest.mark.parametrize(
    ("cavity_mean", "cavity_variance", "prior_variance", "shape_a", "shape_b"),
    [
        (0.0, 1e-5, 1e-7, 0.5, 0.5),
        (0.02, 1e-5, 1e-7, 0.5, 0.5),
        (0.05, 1e-5, 1e-6, 0.5, 0.3),
        (0.003, 1e-5, 1e-8, 0.5, 2.0),
        (1.0, 0.5, 0.2, 0.5, 0.8),
        (0.1, 1e-5, 1e-9, 0.5, 0.1),
        (0.2, 1e-5, 1e-8, 1.0, 0.5),
    ],
)
def test_grid_tilted_moments_match_quadrature(cavity_mean, cavity_variance, prior_variance, shape_a, shape_b) -> None:
    grid = ld_space_fit._log_local_scale_grid(shape_a, np.array([shape_b]))
    cavity_precision = np.array([1.0 / cavity_variance])
    cavity_shift = np.array([cavity_mean / cavity_variance])
    log_normalizer, weight, component_variance, _relative_precision = ld_space_fit._tilted_weights(
        cavity_precision, cavity_shift, np.array([prior_variance]), grid.class_log_prior_mass, grid.local_scale
    )
    mean, variance = ld_space_fit._tilted_mean_and_variance(cavity_shift, weight, component_variance)
    exact_log_normalizer, exact_mean, exact_variance = _betaprime_tilted_quadrature(
        cavity_mean, cavity_variance, prior_variance, shape_a, shape_b
    )
    # The grid's log Z omits the hyperparameter-free cavity normalizer log N(m; 0, v).
    cavity_log_normalizer = -0.5 * (np.log(2 * np.pi * cavity_variance) + cavity_mean**2 / cavity_variance)
    np.testing.assert_allclose(log_normalizer[0] + cavity_log_normalizer, exact_log_normalizer, rtol=0.0, atol=5e-8)
    np.testing.assert_allclose(mean[0], exact_mean, rtol=1e-7, atol=1e-14)
    np.testing.assert_allclose(variance[0], exact_variance, rtol=1e-7)


def test_grid_prior_mass_outside_the_grid_is_negligible() -> None:
    for shape_a in (0.5, 1.0):
        for shape_b in (ld_space_fit._MINIMUM_SHAPE_B, 0.5, ld_space_fit._MAXIMUM_SHAPE_B):
            grid = ld_space_fit._log_local_scale_grid(shape_a, np.array([shape_b]))
            spike_mass = float(np.exp(shape_a * np.log(grid.local_scale[0]) - betaln(shape_a, shape_b)) / shape_a)
            assert spike_mass < 1e-7
            # E_prior[log(1 + λ)] is closed form; the grid reproduces it up to the dropped λ > e^40 tail.
            grid_mean = float(np.exp(grid.class_log_prior_mass[0]) @ grid.log_one_plus_local_scale)
            assert grid_mean <= grid.class_prior_mean_log_one_plus[0] + 1e-9


def _ep_sites_state(cavity_mean, cavity_variance):
    return ExpectationPropagationSites(
        site_precision=np.zeros_like(cavity_mean),
        site_shift=np.zeros_like(cavity_mean),
        cavity_precision=1.0 / cavity_variance,
        cavity_shift=cavity_mean / cavity_variance,
    )


def test_ep_scale_derivatives_match_finite_differences() -> None:
    rng = np.random.default_rng(1)
    variant_count = 40
    cavity_variance = rng.uniform(1e-6, 1e-4, size=variant_count)
    cavity_mean = rng.standard_normal(variant_count) * np.sqrt(cavity_variance) * rng.choice([0.5, 1.0, 8.0], size=variant_count)
    log_prior_variance = np.log(rng.uniform(1e-8, 1e-5, size=variant_count))
    class_index = rng.integers(0, 2, size=variant_count)
    scheme = ld_space_fit._ExpectationPropagationScheme()
    state = _ep_sites_state(cavity_mean, cavity_variance)
    shape_b = np.array([0.5, 0.35])
    everything = slice(0, variant_count)

    def per_variant_log_normalizer(eta):
        grid = ld_space_fit._log_local_scale_grid(0.5, shape_b)
        log_normalizer, *_rest = ld_space_fit._tilted_weights(
            1.0 / cavity_variance, cavity_mean / cavity_variance, np.exp(eta), grid.class_log_prior_mass[class_index], grid.local_scale
        )
        return log_normalizer

    grid = ld_space_fit._log_local_scale_grid(0.5, shape_b)
    device_grid = ld_space_fit._DeviceGrid(local_scale=grid.local_scale, class_log_prior_mass=grid.class_log_prior_mass)
    _value, first, second = scheme.scale_objective(HOST, state, everything, log_prior_variance, class_index, device_grid)
    step = 1e-4
    upper = per_variant_log_normalizer(log_prior_variance + step)
    lower = per_variant_log_normalizer(log_prior_variance - step)
    middle = per_variant_log_normalizer(log_prior_variance)
    np.testing.assert_allclose(first, (upper - lower) / (2 * step), rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(second, (upper - 2 * middle + lower) / step**2, rtol=1e-3, atol=1e-5)


def test_ep_shape_derivatives_match_finite_differences() -> None:
    rng = np.random.default_rng(2)
    variant_count = 60
    cavity_variance = rng.uniform(1e-6, 1e-4, size=variant_count)
    cavity_mean = rng.standard_normal(variant_count) * np.sqrt(cavity_variance) * rng.choice([0.5, 1.0, 6.0], size=variant_count)
    log_prior_variance = np.log(rng.uniform(1e-8, 1e-5, size=variant_count))
    class_index = rng.integers(0, 3, size=variant_count)
    scheme = ld_space_fit._ExpectationPropagationScheme()
    state = _ep_sites_state(cavity_mean, cavity_variance)
    shape_b = np.array([0.5, 0.35, 1.3])
    everything = [slice(0, variant_count)]
    _value, gradient, hessian = scheme.shape_objective(HOST, state, log_prior_variance, class_index, 0.5, shape_b, everything)
    step = 1e-5
    for class_position in range(3):
        offset = np.zeros(3)
        offset[class_position] = step
        upper_value, upper_gradient, _upper_hessian = scheme.shape_objective(
            HOST, state, log_prior_variance, class_index, 0.5, shape_b + offset, everything
        )
        lower_value, lower_gradient, _lower_hessian = scheme.shape_objective(
            HOST, state, log_prior_variance, class_index, 0.5, shape_b - offset, everything
        )
        np.testing.assert_allclose(gradient[class_position], (upper_value - lower_value) / (2 * step), rtol=1e-5)
        np.testing.assert_allclose(
            hessian[class_position], (upper_gradient[class_position] - lower_gradient[class_position]) / (2 * step), rtol=1e-4
        )


@pytest.mark.parametrize(("order", "chi", "psi"), [(0.0, 1e-3, 2.0), (-0.3, 0.5, 1.0), (0.5, 20.0, 0.1), (-0.45, 1e-6, 3.0)])
def test_gig_expected_log_matches_quadrature(order, chi, psi) -> None:
    def log_density(log_value):
        value = np.exp(log_value)
        return order * log_value - 0.5 * (chi / value + psi * value)

    mode = np.log(((order - 1.0) + np.sqrt((order - 1.0) ** 2 + chi * psi)) / psi)
    reference_log = log_density(mode)
    normalizer, _error = integrate.quad(lambda log_value: np.exp(log_density(log_value) - reference_log), mode - 60, mode + 60, limit=2000, epsrel=1e-13)
    first, _error = integrate.quad(
        lambda log_value: log_value * np.exp(log_density(log_value) - reference_log), mode - 60, mode + 60, limit=2000, epsrel=1e-13
    )
    expected = gig_expected_log(np.array([order]), np.array([chi]), np.array([psi]))[0]
    np.testing.assert_allclose(expected, first / normalizer, rtol=0.0, atol=1e-8)


def test_inverse_digamma_inverts_digamma() -> None:
    values = np.array([-12.0, -3.0, -2.22, -0.5, 0.0, 1.3, 7.0])
    np.testing.assert_allclose(digamma(inverse_digamma(values)), values, rtol=0.0, atol=1e-12)


# --------------------------------------------------------------------------- binary


def _logistic_offset_fit(covariates, labels, offset):
    coefficients = np.zeros(covariates.shape[1])
    for _newton_iteration in range(100):
        probability = expit(covariates @ coefficients + offset)
        gradient = covariates.T @ (labels - probability)
        hessian = covariates.T @ (covariates * (probability * (1 - probability))[:, None])
        step = np.linalg.solve(hessian, gradient)
        coefficients += step
        if np.max(np.abs(step)) < 1e-12:
            break
    return coefficients


def _reference_binary_fit(genotypes, covariates, labels, hypermodel, warm_model, pass_count):
    """Laplace-EP-EB on the individual-level logistic likelihood with exact weights.

    Each local iteration finds the mode of the logistic likelihood times the sites
    (covariates flat), then takes the Laplace covariance of β with α profiled out.
    """
    scheme = ld_space_fit._ExpectationPropagationScheme()
    model = warm_model
    covariate_count = covariates.shape[1]
    variant_count = genotypes.shape[1]
    design = np.column_stack([covariates, genotypes])
    coefficients = np.concatenate([_logistic_offset_fit(covariates, labels, np.zeros(labels.shape[0])), np.zeros(variant_count)])
    everything = slice(0, variant_count)
    for _pass in range(pass_count):
        grid = ld_space_fit._device_grid(HOST, hypermodel, model)
        block_prior = ld_space_fit._block_prior(HOST, model, hypermodel, grid, everything)
        block_state = scheme.load_block(HOST, model.local, everything)
        for _local_iteration in range(ld_space_fit._LOCAL_ITERATIONS_PER_PASS):
            precision, shift = scheme.solve_terms(HOST, block_state, block_prior)
            penalty = np.concatenate([np.zeros(covariate_count), precision])
            linear = np.concatenate([np.zeros(covariate_count), shift])
            for _newton_iteration in range(100):
                probability = expit(design @ coefficients)
                gradient = design.T @ (labels - probability) - penalty * coefficients + linear
                hessian = design.T @ (design * (probability * (1 - probability))[:, None]) + np.diag(penalty)
                step = np.linalg.solve(hessian, gradient)
                coefficients = coefficients + step
                if np.max(np.abs(step)) < 1e-12:
                    break
            probability = expit(design @ coefficients)
            weighted = design * (probability * (1 - probability))[:, None]
            hessian = design.T @ weighted + np.diag(penalty)
            covariance = np.linalg.inv(hessian)[covariate_count:, covariate_count:]
            block_state = scheme.update_block(
                HOST, block_state, coefficients[covariate_count:].copy(), np.diag(covariance).copy(), block_prior
            )
        scheme.store_block(HOST, model.local, everything, block_state)
        model.posterior_mean = coefficients[covariate_count:].copy()
        ld_space_fit._update_scale_model(scheme, HOST, model, hypermodel, [everything])
        ld_space_fit._update_shapes(scheme, HOST, model, hypermodel, [everything])
    return model, coefficients[:covariate_count]


def _auc(labels, scores):
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.shape[0])
    ranks[order] = np.arange(1, scores.shape[0] + 1)
    positives = labels == 1
    positive_count = int(np.sum(positives))
    return (np.sum(ranks[positives]) - positive_count * (positive_count + 1) / 2) / (positive_count * (labels.shape[0] - positive_count))


def test_binary_fit_after_one_refresh_matches_individual_level_laplace_ep() -> None:
    rng = np.random.default_rng(11)
    train_count, test_count = 3000, 20000
    covariate_count = 3
    block_widths = [40, 40, 40, 40, 40]
    covariates, genotypes, boundaries = _orthogonal_block_design(rng, train_count, test_count, block_widths, covariate_count)
    variant_count = genotypes.shape[1]
    class_index = (rng.uniform(size=variant_count) < 0.15).astype(np.int64)
    effects = _sparse_effects(rng, variant_count, class_index, causal_count=15)
    effects *= 0.6 / np.std(genotypes @ effects)
    logit = covariates @ np.array([-1.5, 0.3, -0.2]) + genotypes @ effects
    labels = (rng.uniform(size=logit.shape[0]) < expit(logit)).astype(np.float64)
    train, test = slice(0, train_count), slice(train_count, None)
    train_covariates, train_genotypes, train_labels = covariates[train], genotypes[train], labels[train]
    ld_blocks = _ld_blocks(train_genotypes, boundaries)
    hypermodel = _hypermodel(variant_count, class_index, rng)

    alpha = _logistic_offset_fit(train_covariates, train_labels, np.zeros(train_count))
    probability = expit(train_covariates @ alpha)
    first_statistics = binary_statistics_at(
        residual_score=train_genotypes.T @ (train_labels - probability),
        gram_times_expansion=np.zeros(variant_count),
        fitted_probability=probability,
        covariate_count=covariate_count,
    )
    (first_fit,) = fit_ld_space(ld_blocks, [first_statistics], hypermodel, CPU_BUDGET)
    offset = train_genotypes @ first_fit.posterior_mean
    alpha = _logistic_offset_fit(train_covariates, train_labels, offset)
    probability = expit(train_covariates @ alpha + offset)
    refreshed_statistics = binary_statistics_at(
        residual_score=train_genotypes.T @ (train_labels - probability),
        gram_times_expansion=train_genotypes.T @ offset,
        fitted_probability=probability,
        covariate_count=covariate_count,
    )
    (refreshed_fit,) = fit_ld_space(ld_blocks, [refreshed_statistics], hypermodel, CPU_BUDGET, warm_starts=[first_fit])
    assert first_fit.converged and refreshed_fit.converged

    scheme = ld_space_fit._ExpectationPropagationScheme()
    reference_model, _reference_alpha = _reference_binary_fit(
        train_genotypes,
        train_covariates,
        train_labels,
        hypermodel,
        _initial_model(scheme, ld_blocks, first_statistics, hypermodel),
        pass_count=25,
    )

    def held_out_auc(effect_estimate):
        train_offset = train_genotypes @ effect_estimate
        fitted_alpha = _logistic_offset_fit(train_covariates, train_labels, train_offset)
        return _auc(labels[test], covariates[test] @ fitted_alpha + genotypes[test] @ effect_estimate)

    reference_auc = held_out_auc(reference_model.posterior_mean)
    refreshed_auc = held_out_auc(refreshed_fit.posterior_mean)
    predictor_correlation = np.corrcoef(genotypes[test] @ refreshed_fit.posterior_mean, genotypes[test] @ reference_model.posterior_mean)[0, 1]
    assert abs(refreshed_auc - reference_auc) < 0.003, (refreshed_auc, reference_auc)
    assert predictor_correlation > 0.99, predictor_correlation
    assert abs(refreshed_fit.log_variance_level - reference_model.log_variance_level) < 0.25


def test_ep_refuses_a_block_wider_than_the_sample_rank() -> None:
    genotypes, _residual, _covariates, boundaries, statistics, hypermodel, _genetic = _quantitative_problem(
        seed=7, sample_count=150, block_widths=[230]
    )
    with pytest.raises(ValueError, match="EP needs every block narrower"):
        fit_ld_space(_ld_blocks(genotypes, boundaries), [statistics], hypermodel, CPU_BUDGET)


def test_hyperparameter_blocks_meet_the_variant_and_class_targets(monkeypatch) -> None:
    monkeypatch.setattr(ld_space_fit, "_HYPERPARAMETER_SUBSET_VARIANTS", 300)
    monkeypatch.setattr(ld_space_fit, "_HYPERPARAMETER_SUBSET_CLASS_VARIANTS", 40)
    boundaries = np.arange(0, 2001, 50, dtype=np.int64)
    class_index = np.zeros(2000, dtype=np.int64)
    class_index[1500:1560] = 1
    blocks = ld_space_fit._hyperparameter_blocks(boundaries, class_index, 2)
    np.testing.assert_array_equal(blocks, ld_space_fit._hyperparameter_blocks(boundaries, class_index, 2))
    chosen = np.concatenate([np.arange(boundaries[block], boundaries[block + 1]) for block in blocks])
    assert chosen.shape[0] >= 300
    assert np.sum(class_index[chosen] == 1) >= 40
    # Everything is taken when the problem is smaller than the targets.
    small = ld_space_fit._hyperparameter_blocks(boundaries[:5], class_index[:200], 2)
    np.testing.assert_array_equal(small, np.arange(4))


def test_subset_hyperparameters_then_fixed_pass_stays_close_to_the_full_fit(monkeypatch) -> None:
    genotypes, _residual, _covariates, boundaries, statistics, hypermodel, genetic = _quantitative_problem(
        seed=3, sample_count=1200, block_widths=[40] * 12
    )
    ld_blocks = _ld_blocks(genotypes, boundaries)
    (full,) = fit_ld_space(ld_blocks, [statistics], hypermodel, CPU_BUDGET)
    monkeypatch.setattr(ld_space_fit, "_HYPERPARAMETER_SUBSET_VARIANTS", 240)
    monkeypatch.setattr(ld_space_fit, "_HYPERPARAMETER_SUBSET_CLASS_VARIANTS", 20)
    assert ld_space_fit._hyperparameter_blocks(boundaries, hypermodel.variant_class_index, 2).shape[0] < 12
    (two_phase,) = fit_ld_space(ld_blocks, [statistics], hypermodel, CPU_BUDGET)
    assert full.converged and two_phase.converged
    assert abs(two_phase.log_variance_level - full.log_variance_level) < 0.5
    full_accuracy = np.corrcoef(genotypes @ full.posterior_mean, genetic)[0, 1] ** 2
    two_phase_accuracy = np.corrcoef(genotypes @ two_phase.posterior_mean, genetic)[0, 1] ** 2
    assert two_phase_accuracy > full_accuracy - 0.02, (two_phase_accuracy, full_accuracy)


def test_stage1_on_stage0_output_matches_dense_sufficient_statistics(tmp_path) -> None:
    rng = np.random.default_rng(12)
    sample_count, variant_count = 900, 360
    codes = mosaic_codes(rng, sample_count, variant_count)
    source = InMemoryTileSource(codes={"chr1": codes}, groups={"chr1": bubble_groups(rng, variant_count)})
    covariates = np.column_stack([np.ones(sample_count), rng.standard_normal((sample_count, 2))])
    dosage = codes.T.astype(np.float64) / 127.0
    phenotype = covariates @ np.array([1.0, 0.4, -0.2]) + rng.standard_normal(sample_count)
    phenotype += 0.3 * (dosage[:, rng.choice(variant_count, 12, replace=False)] @ rng.standard_normal(12))
    config = ModelConfig(minimum_minor_allele_frequency=0.01)
    statistics = compute_genotype_statistics(
        source, np.arange(sample_count), covariates, phenotype[:, None], config, CPU_BUDGET, 128, tmp_path / "ld"
    )
    ld = statistics.ld
    trait = quantitative_trait_statistics(statistics, 0)

    active = dosage[:, statistics.active_rows]
    standardized = (active - active.mean(axis=0)) / active.std(axis=0)
    reduced = standardized[:, statistics.tie_map.kept_indices]
    projected = reduced - covariates @ np.linalg.lstsq(covariates, reduced, rcond=None)[0]
    residual = phenotype - covariates @ np.linalg.lstsq(covariates, phenotype, rcond=None)[0]
    dense = _ld_blocks(projected, np.asarray(ld.block_boundaries))
    dense_trait = QuantitativeTraitStatistics(
        sample_count=sample_count, covariate_count=3, score=projected.T @ residual, residual_sum_of_squares=float(residual @ residual)
    )
    np.testing.assert_allclose(trait.score, dense_trait.score, rtol=1e-8, atol=1e-6)
    np.testing.assert_allclose(trait.residual_sum_of_squares, dense_trait.residual_sum_of_squares, rtol=1e-10)

    reduced_count = projected.shape[1]
    hypermodel = LDPriorHypermodel(
        annotation_design=np.zeros((reduced_count, 0)),
        annotation_prior_mean=np.zeros(0),
        annotation_prior_precision=np.zeros(0),
        log_variance_offset=np.zeros(reduced_count),
        variant_class_index=np.zeros(reduced_count, dtype=np.int64),
        class_names=("snv",),
        shape_a=0.5,
        initial_shape_b=np.array([0.5]),
        shape_b_pooling_variance=0.25,
    )
    (from_stage0,) = fit_ld_space(ld, [trait], hypermodel, CPU_BUDGET)
    (from_dense,) = fit_ld_space(dense, [dense_trait], hypermodel, CPU_BUDGET)
    assert ld.block_count > 1 and from_stage0.converged and from_dense.converged
    # Stage 0 stores the Grams in fp32.
    np.testing.assert_allclose(from_stage0.log_variance_level, from_dense.log_variance_level, rtol=0.0, atol=1e-4)
    scale = np.max(np.abs(from_dense.posterior_mean))
    np.testing.assert_allclose(from_stage0.posterior_mean, from_dense.posterior_mean, rtol=0.0, atol=1e-4 * scale)


def test_several_traits_share_one_pass_and_match_separate_fits() -> None:
    genotypes, residual, _covariates, boundaries, statistics, hypermodel, genetic = _quantitative_problem(
        seed=4, sample_count=500, block_widths=[30, 30, 30]
    )
    rng = np.random.default_rng(8)
    # A second trait on the same genotypes: half the genetic signal, fresh noise.
    other_residual = 0.5 * genetic + rng.standard_normal(500)
    other_residual -= other_residual.mean()
    other_statistics = QuantitativeTraitStatistics(
        sample_count=500, covariate_count=3, score=genotypes.T @ other_residual, residual_sum_of_squares=float(other_residual @ other_residual)
    )
    ld_blocks = _ld_blocks(genotypes, boundaries)
    joint = fit_ld_space(ld_blocks, [statistics, other_statistics], hypermodel, CPU_BUDGET)
    separate = [fit_ld_space(ld_blocks, [single], hypermodel, CPU_BUDGET)[0] for single in (statistics, other_statistics)]
    for joint_fit, separate_fit in zip(joint, separate, strict=True):
        assert joint_fit.passes == separate_fit.passes
        np.testing.assert_allclose(joint_fit.posterior_mean, separate_fit.posterior_mean, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(joint_fit.log_variance_level, separate_fit.log_variance_level, rtol=1e-12)
