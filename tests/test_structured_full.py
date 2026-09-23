"""The structured fit at genome scale (``structured_full``) against dense algebra on tiny problems.

The dense reference here is the model's own definition written with explicit matrices: Xp = P X with P = (I - H) M the
training rows' covariate complement, A = Xp'Xp / sigma^2 + D^-1, S = I + Xt D Xt' (Xt = Xp / sigma), the single
effect by enumerating its (candidate, node) states, and the ELBO term by term. Every CPU test also runs on a CUDA
device when one is present (the ``array_module`` fixture).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import logm

from sv_pgs.dual_solve import DenseDualSource
from sv_pgs.scale_mixture_ep import class_log_density, log_scale, scale_mixture_prior
from sv_pgs.structured_full import (
    StructuredFull,
    _quadrature_bracket,
    _single_effect,
    candidate_windows,
    fit_structured,
)
from sv_pgs.tie_members import TieGroups

_NODES = np.linspace(-8.0, 2.0, 11)


def _modules() -> list:
    modules = [np]
    try:
        import cupy  # noqa: PLC0415

        if cupy.cuda.runtime.getDeviceCount() > 0:
            modules.append(cupy)
    except Exception:  # noqa: BLE001 - no device, no CUDA run
        pass
    return modules


@pytest.fixture(params=_modules(), ids=lambda module: module.__name__)
def array_module(request):
    return request.param


def _host(values):
    return np.asarray(values.get() if hasattr(values, "get") else values)


def _identity_ties(count: int) -> TieGroups:
    return TieGroups(group=np.arange(count, dtype=np.int64), sign=np.ones(count), group_count=count)


def _prior(classes: np.ndarray):
    count = classes.shape[0]
    return scale_mixture_prior(
        class_index=classes.astype(np.int64), log_variance_offset=np.zeros(count), annotation_design=np.zeros((count, 0)), annotation_groups=(),
        nodes=_NODES, floor=float(_NODES[0]), top=float(_NODES[-1]),
    )


def _problem(seed: int, sizes=(10, 10, 10), samples: int = 90, duplicate: bool = False, causal=(3, 17), size: float = 0.6, spread: float = 0.05):
    """Standardized Gaussian genotypes in blocks of ``sizes``, an intercept and one covariate, a mask that holds out
    the last tenth, and a trait with two effects plus a small polygenic part. With ``duplicate``, the last column of the
    first block is copied into the first column of the second."""
    generator = np.random.default_rng(seed)
    count = int(sum(sizes))
    genotypes = generator.standard_normal((samples, count))
    if duplicate:
        genotypes[:, sizes[0]] = genotypes[:, sizes[0] - 1]
    genotypes = (genotypes - genotypes.mean(axis=0)) / genotypes.std(axis=0)
    covariates = np.column_stack([np.ones(samples), generator.standard_normal(samples)])
    effects = spread * generator.standard_normal(count)
    for index in causal:
        effects[index] += size
    targets = genotypes @ effects + 0.4 * covariates[:, 1] + generator.standard_normal(samples)
    mask = np.ones(samples)
    mask[-samples // 10:] = 0.0
    bounds = list(zip(np.cumsum((0,) + tuple(sizes))[:-1], np.cumsum(sizes)))
    return genotypes, covariates, targets, mask, bounds


def _projector(covariates: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weighted = mask[:, None] * covariates
    hat = weighted @ np.linalg.pinv(weighted.T @ weighted) @ weighted.T
    return (np.eye(mask.shape[0]) - hat) @ np.diag(mask)


def _state(array_module, genotypes, covariates, targets, mask, bounds, draw_count, proxies=None, classes=None, chromosomes=None, background_variance=0.02):
    count = genotypes.shape[1]
    classes = np.zeros(count, dtype=np.int64) if classes is None else classes
    source = DenseDualSource(array_module.asarray(genotypes), bounds, array_module)
    return StructuredFull(
        source=source, chromosomes=chromosomes or ["1"] * len(bounds), ties=_identity_ties(count), prior=_prior(classes),
        mask=array_module.asarray(mask), targets=array_module.asarray(targets), covariates=array_module.asarray(covariates), noise=0.8,
        background_variance=background_variance, local_variance=0.02, draw_count=draw_count, seed=5, proxies=proxies,
    )


def _dense_elbo(state: StructuredFull, genotypes, covariates, targets, mask) -> tuple[float, float]:
    """F of the state by dense algebra (exact log det S) and the module's F."""
    projector = _projector(covariates, mask)
    projected = projector @ genotypes
    _member, group_variance = state.background_variances()
    kernel = np.eye(mask.shape[0]) + projected @ np.diag(group_variance) @ projected.T / state.noise
    sign, log_determinant = np.linalg.slogdet(kernel)
    assert sign > 0.0
    mean = state.background_group_mean.copy()
    for window, effects in zip(state.windows, state.effects):
        for effect in effects:
            mean[window.members] += effect.mean
    residual = projector @ targets - projected @ mean
    np.testing.assert_allclose(_host(state.residual), residual, rtol=1e-9, atol=1e-9)
    dense = (
        -0.5 * state.residual_dimension * math.log(2.0 * math.pi * state.noise)
        - (residual @ residual + state.spread_total) / (2.0 * state.noise)
        - 0.5 * float(np.sum(state.background_group_mean ** 2 / group_variance))
        - 0.5 * log_determinant
        - state.divergence_total
    )
    return dense, state.elbo()


# ------------------------------------------------------------------ the single effect


def test_single_effect_is_exact_enumeration(array_module) -> None:
    """One effect over five candidates and a two-class lattice: the update's weights, moments, log B and KL are the
    enumeration of its (candidate, node) states by their marginal likelihoods N(y; 0, sigma^2 I + v x x')."""
    xp = array_module
    generator = np.random.default_rng(3)
    samples, count = 40, 5
    design = generator.standard_normal((samples, count))
    noise = 0.7
    response = 0.9 * design[:, 2] + math.sqrt(noise) * generator.standard_normal(samples)
    classes = np.array([0, 1, 0, 1, 0])
    prior = _prior(classes)
    coefficients = generator.standard_normal(prior.coefficient_size) * 0.3
    log_density = class_log_density(prior, coefficients)
    scales = log_scale(prior, coefficients)
    log_candidate = np.log(np.array([0.1, 0.3, 0.2, 0.25, 0.15]))
    log_variance = scales[:, None] + _NODES[None, :]
    indicator = (classes[None, :] == np.arange(2)[:, None]).astype(np.float64)
    update = _single_effect(
        xp, xp.asarray(design.T @ response / noise), xp.asarray(np.sum(design ** 2, axis=0) / noise), xp.asarray(log_candidate),
        xp.asarray(log_density[classes]), xp.asarray(log_variance), xp.asarray(indicator), xp.asarray(np.exp(-_NODES)),
    )
    null = -0.5 * response @ response / noise - 0.5 * samples * math.log(2.0 * math.pi * noise)
    log_weights = np.empty((count, _NODES.size))
    posterior_mean = np.empty_like(log_weights)
    posterior_variance = np.empty_like(log_weights)
    for j in range(count):
        column = design[:, j]
        for k, node_variance in enumerate(np.exp(log_variance[j])):
            covariance = noise * np.eye(samples) + node_variance * np.outer(column, column)
            sign, determinant = np.linalg.slogdet(covariance)
            marginal = -0.5 * response @ np.linalg.solve(covariance, response) - 0.5 * determinant - 0.5 * samples * math.log(2.0 * math.pi)
            log_weights[j, k] = log_candidate[j] + log_density[classes[j], k] + marginal - null
            precision = column @ column / noise + 1.0 / node_variance
            posterior_variance[j, k] = 1.0 / precision
            posterior_mean[j, k] = (column @ response / noise) / precision
    log_evidence = float(np.log(np.sum(np.exp(log_weights))))
    assert update.on == (log_evidence > 0.0)
    assert update.log_evidence == pytest.approx(log_evidence, rel=1e-10, abs=1e-10)
    alpha = np.exp(log_weights - log_evidence)
    np.testing.assert_allclose(_host(update.mean), np.sum(alpha * posterior_mean, axis=1), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(_host(update.second), np.sum(alpha * (posterior_variance + posterior_mean ** 2), axis=1), rtol=1e-9, atol=1e-12)
    prior_mass = np.exp(log_candidate[:, None] + log_density[classes])
    node_variance = np.exp(log_variance)
    gaussian_divergence = 0.5 * ((posterior_variance + posterior_mean ** 2) / node_variance - 1.0 - np.log(posterior_variance / node_variance))
    divergence = float(np.sum(alpha * (np.log(alpha / prior_mass) + gaussian_divergence)))
    assert update.divergence == pytest.approx(divergence, rel=1e-9)
    np.testing.assert_allclose(_host(update.masses), indicator @ alpha, rtol=1e-10, atol=1e-14)


def test_identical_candidates_get_identical_weights(array_module) -> None:
    """Two identical columns with identical priors: equal weights and means, in either order (no argmax)."""
    xp = array_module
    shift = xp.asarray(np.array([2.0, 5.0, 5.0, 1.0]))
    precision = xp.asarray(np.array([3.0, 4.0, 4.0, 2.0]))
    log_variance = xp.asarray(np.tile(_NODES, (4, 1)))
    log_mixture = xp.asarray(np.tile(-np.log(_NODES.size) * np.ones(_NODES.size), (4, 1)))
    update = _single_effect(
        xp, shift, precision, xp.asarray(np.full(4, -np.log(4.0))), log_mixture, log_variance, xp.ones((1, 4)), xp.asarray(np.exp(-_NODES)),
    )
    mean = _host(update.mean)
    assert update.on
    assert mean[1] == mean[2]
    assert _host(update.weight)[1] == _host(update.weight)[2]


def test_quadrature_brackets_the_log_quadratic_form() -> None:
    """Gauss and Gauss-Radau (node at 1) bracket z' log(S) z at every step of CG on S >= I."""
    generator = np.random.default_rng(1)
    size = 30
    factor = generator.standard_normal((size, 8))
    operator = np.eye(size) + factor @ np.diag(np.linspace(0.1, 20.0, 8)) @ factor.T
    probe = generator.choice([-1.0, 1.0], size=size)
    exact = float(probe @ logm(operator).real @ probe)
    solution, residual = np.zeros(size), probe.copy()
    direction = residual.copy()
    steps, ratios = [], []
    for _ in range(9):
        image = operator @ direction
        step = residual @ residual / (direction @ image)
        solution += step * direction
        updated = residual - step * image
        ratio = updated @ updated / (residual @ residual)
        steps.append(step)
        ratios.append(ratio)
        direction = updated + ratio * direction
        residual = updated
        low, high = sorted(_quadrature_bracket(steps, ratios, float(probe @ probe)))
        assert low - 1e-9 * abs(exact) <= exact <= high + 1e-9 * abs(exact)
    assert high - low < 1e-6 * abs(exact)


# ------------------------------------------------------------------ windows


def test_windows_cover_every_member_twice_and_keep_proxies_together() -> None:
    bounds = [(0, 5), (5, 9), (9, 15), (15, 20), (20, 24)]
    chromosomes = ["1", "1", "1", "2", "2"]
    proxies = np.array([[4, 5], [8, 9], [14, 15]])  # the last pair crosses a chromosome: never a proxy pair from Stage 0
    windows = candidate_windows(bounds, chromosomes, _identity_ties(24), proxies[:2])
    cores = np.zeros(24, dtype=int)
    for window in windows:
        cores[np.arange(window.core_start, window.core_stop)] += 1
    assert np.all(cores == 2)
    for first, second in proxies[:2]:
        holding = [{index for index, window in enumerate(windows) if member in set(window.members)} for member in (first, second)]
        assert holding[0] == holding[1]
    assert [window.blocks for window in windows] == [(0,), (0, 1), (1, 2), (2,), (3,), (3, 4), (4,)]


# ------------------------------------------------------------------ the background


def test_background_step_is_certified_and_its_probe_moments_are_unbiased(array_module) -> None:
    """The mean against A^-1 Xp'y / sigma^2 within its certificate; each probe's log-det bracket holds its exact
    quadratic form; gamma_u and log det S within four standard errors of their exact values."""
    genotypes, covariates, targets, mask, bounds = _problem(11)
    state = _state(array_module, genotypes, covariates, targets, mask, bounds, draw_count=256)
    state.background_step()
    projector = _projector(covariates, mask)
    projected = projector @ genotypes
    _member, variance = state.background_variances()
    precision = projected.T @ projected / state.noise + np.diag(1.0 / variance)
    mean = np.linalg.solve(precision, projected.T @ (projector @ targets) / state.noise)
    error = state.background_group_mean - mean
    assert math.sqrt(error @ precision @ error) <= state.moments.mean_certificate * (1.0 + 1e-8) + 1e-12
    assert state.moments.mean_certificate <= math.sqrt(1.0 / state.draw_count)
    kernel = np.eye(mask.shape[0]) + projected @ np.diag(variance) @ projected.T / state.noise
    effective = float(np.trace(np.eye(mask.shape[0]) - np.linalg.inv(kernel)))
    assert abs(state.moments.effective_count - effective) <= 4.0 * state.moments.effective_count_error
    log_kernel = logm(kernel).real
    probes = _host(state.probes)
    exact = np.einsum("ik,ij,jk->k", probes, log_kernel, probes)
    brackets = state._probe_brackets
    assert np.all(brackets[:, 0] - 1e-9 * np.abs(exact) <= exact) and np.all(exact <= brackets[:, 1] + 1e-9 * np.abs(exact))
    _sign, log_determinant = np.linalg.slogdet(kernel)
    assert abs(state.moments.log_determinant - log_determinant) <= 4.0 * state.moments.log_determinant_error + state.moments.log_determinant_width
    # The image the residual carries is Xp m_u of the returned mean.
    np.testing.assert_allclose(_host(state.background_image), projected @ state.background_group_mean, rtol=1e-8, atol=1e-8)


def test_background_draws_have_the_posterior_covariance(array_module) -> None:
    genotypes, covariates, targets, mask, bounds = _problem(12, sizes=(6, 6), causal=(3, 8))
    state = _state(array_module, genotypes, covariates, targets, mask, bounds, draw_count=16)
    state.background_step()
    count = 4000
    draws = np.zeros((genotypes.shape[1], count))
    widths = []

    def consume(start, stop, values):
        widths.append(stop - start)
        draws[start:stop] = _host(values)

    certificates = state.background_draws(count, 9, consume)
    assert widths == [6, 6]
    assert np.all(certificates <= math.sqrt(1.0 / state.draw_count))
    projector = _projector(covariates, mask)
    projected = projector @ genotypes
    _member, variance = state.background_variances()
    covariance = np.linalg.inv(projected.T @ projected / state.noise + np.diag(1.0 / variance))
    sample = np.cov(draws)
    # Each entry's Monte Carlo standard error is sqrt((C_ii C_jj + C_ij^2) / count).
    error = np.sqrt((np.outer(np.diag(covariance), np.diag(covariance)) + covariance ** 2) / count)
    assert np.all(np.abs(sample - covariance) <= 5.0 * error)
    np.testing.assert_allclose(draws.mean(axis=1), state.background_group_mean, atol=5.0 * float(np.sqrt(np.diag(covariance).max() / count)))


# ------------------------------------------------------------------ the fit


def test_sweep_keeps_the_global_residual_and_the_elbo_matches_dense_algebra(array_module) -> None:
    """Across iterations the residual is y_P - Xp (m_u + sum_l m_l) exactly, every live effect's stored products are
    Xw'Xp m_l, F equals its dense value within the log det's probe error, and J rises at fixed weights."""
    # The effects-first start (the background at the lattice floor), where effects are live from the first sweep.
    genotypes, covariates, targets, mask, bounds = _problem(13, samples=200)
    state = _state(array_module, genotypes, covariates, targets, mask, bounds, draw_count=256, background_variance=math.exp(_NODES[0]))
    state.background_step()
    projector = _projector(covariates, mask)
    projected = projector @ genotypes
    previous = state.objective()
    for _ in range(4):
        state.sweep()
        for window, effects in zip(state.windows, state.effects):
            for effect in effects:
                group_mean = np.zeros(window.group_count)
                np.add.at(group_mean, window.local_group, effect.mean)
                image = projected[:, window.groups] @ group_mean
                np.testing.assert_allclose(effect.products, projected[:, window.groups].T @ image, rtol=1e-9, atol=1e-9)
        state.local_m_step()
        state.expansion_step()
        state.background_m_step()
        state.noise_m_step(state.expanded_trace)
        state.background_step()
        dense, reported = _dense_elbo(state, genotypes, covariates, targets, mask)
        assert abs(dense - reported) <= 4.0 * 0.5 * state.moments.log_determinant_error + 0.5 * state.moments.log_determinant_width
        updated = state.objective()
        assert updated >= previous - max(state.tolerance, 4.0 * 0.5 * state.moments.log_determinant_error)
        state.smoothing_step()
        previous = state.objective()
    assert sum(len(effects) for effects in state.effects) >= 1


def test_identical_proxies_across_a_block_boundary_are_allocated_equally(array_module) -> None:
    """A column copied across the boundary of two blocks (not a Stage 0 tie: they straddle a cut), with the signal on
    it: with the proxy pair in the windows' halos every effect that can take one can take the other, and the fit gives
    both the same mean and inclusion. Without the halos the first window's effect takes one whole."""
    genotypes, covariates, targets, mask, bounds = _problem(14, sizes=(10, 10, 10), samples=300, duplicate=True, causal=(9, 25), size=1.0, spread=0.01)
    fitted = fit_structured(
        source=DenseDualSource(array_module.asarray(genotypes), bounds, array_module), chromosomes=["1"] * 3, ties=_identity_ties(30),
        prior=_prior(np.zeros(30, dtype=np.int64)), mask=array_module.asarray(mask), targets=array_module.asarray(targets),
        covariates=array_module.asarray(covariates), noise=0.8, background_variance=0.02, draw_count=32, seed=5, proxies=np.array([[9, 10]]),
    )
    assert fitted.converged
    assert fitted.member_mean[9] == pytest.approx(fitted.member_mean[10], rel=1e-9)
    assert fitted.inclusion[9] == pytest.approx(fitted.inclusion[10], rel=1e-9)
    assert fitted.inclusion[9] > 0.0
    split = fit_structured(
        source=DenseDualSource(array_module.asarray(genotypes), bounds, array_module), chromosomes=["1"] * 3, ties=_identity_ties(30),
        prior=_prior(np.zeros(30, dtype=np.int64)), mask=array_module.asarray(mask), targets=array_module.asarray(targets),
        covariates=array_module.asarray(covariates), noise=0.8, background_variance=0.02, draw_count=32, seed=5,
    )
    assert abs(split.member_mean[9] - split.member_mean[10]) > 1e-3 * abs(split.member_mean[9] + split.member_mean[10])


def test_fit_converges_recovers_the_effects_and_holds_no_p_by_k_or_n_by_n_array(array_module) -> None:
    """The whole fit on 60 columns, 90 samples and K = 8 probes: it converges, its ascent history never falls beyond
    its probe error, the two large effects are found, and no array the state keeps is p x K or n x n (the memory
    contract: p, n, n x (K + 1) and window-sized arrays only)."""
    genotypes, covariates, targets, mask, bounds = _problem(15, sizes=(20, 20, 20), causal=(5, 44))
    draw_count = 8
    source = DenseDualSource(array_module.asarray(genotypes), bounds, array_module)
    fitted = fit_structured(
        source=source, chromosomes=["1"] * 3, ties=_identity_ties(60), prior=_prior(np.zeros(60, dtype=np.int64)),
        mask=array_module.asarray(mask), targets=array_module.asarray(targets), covariates=array_module.asarray(covariates), noise=0.8,
        background_variance=0.02, draw_count=draw_count, seed=5,
    )
    assert fitted.converged
    for record in fitted.history:
        assert record.fixed_smoothing_gain >= -max(0.5 / draw_count, 4.0 * record.gain_error)
    assert np.argsort(-np.abs(fitted.member_mean))[:2].tolist() in ([5, 44], [44, 5])
    samples, count = genotypes.shape
    state = _state(array_module, genotypes, covariates, targets, mask, bounds, draw_count=draw_count)
    state.background_step()
    state.sweep()
    state.local_m_step()
    state.background_step()
    for name, value in vars(state).items():
        if isinstance(value, np.ndarray) or type(value).__module__.startswith("cupy"):
            assert int(value.size) <= max(samples * (draw_count + 1), count), name
            assert tuple(value.shape) not in ((count, draw_count), (samples, samples)), name
    for window, effects in zip(state.windows, state.effects):
        for effect in effects:
            assert effect.mean.shape == window.members.shape and effect.products.shape == window.groups.shape


@pytest.mark.slow  # Stage 0 on a synthetic store plus the whole fit: its cost is measured on its own first
def test_store_fit_scores_held_out_samples(tmp_path: Path) -> None:
    """``fit_store`` on the full-data tests' synthetic store (two chromosomes, 15 causal variants, h2 = 0.5): the
    held-out genetic score correlates with the truth."""
    from sv_pgs.structured_full import fit_store  # noqa: PLC0415
    from tests.test_full_data_fit import _SAMPLES, _TRAINING, _budget, _store  # noqa: PLC0415

    store, covariate, targets, genetic = _store(tmp_path / "store", 7)
    training = np.arange(_TRAINING)
    covariates = np.column_stack([np.ones(_TRAINING), covariate[training]])
    fitted = fit_store(
        store=store, training_columns=training, covariates=covariates, targets=targets[training], log_reliability=np.zeros(store.n_variants),
        budget=_budget(), work_dir=tmp_path, seed=3, draw_count=16,
    )
    assert fitted.fit.converged
    codes = store.read_codes(0, store.n_variants).astype(np.float64)[fitted.store_rows]
    standardized = (codes - 127.0 - fitted.signed_means[:, None]) / fitted.signed_scales[:, None]
    score = fitted.fit.member_mean @ standardized
    held_out = np.arange(_TRAINING, _SAMPLES)
    assert np.corrcoef(score[held_out], genetic[held_out])[0, 1] > 0.5
