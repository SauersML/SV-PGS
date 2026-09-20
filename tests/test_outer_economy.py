"""outer_economy against the parent solver and dense linear algebra: the same certified answers in fewer reads.

Reads are counted as passes over the source's blocks. Every tolerance is a rounding bound, as in tests/test_dual_solve.py.
"""

from __future__ import annotations

import numpy as np

from sv_pgs import dual_solve, outer_economy
from tests.test_dual_solve import EPS, MODEL_COUNT, _dense_gaussian, _gaussian_problem, _grams


class _CountedSource(dual_solve.DenseDualSource):
    """A dense source that counts full passes over its blocks, whatever PassCount records."""

    def __init__(self, genotypes, bounds) -> None:
        super().__init__(genotypes, bounds)
        self.reads = 0

    def blocks(self):
        self.reads += 1
        return super().blocks()


def _pair(seed: int, probe_count: int = 3):
    genotypes, bounds, covariates, training, noise, precision, shift, response, offsets, negative = _gaussian_problem(seed)
    keywords = dict(training=training, targets=response, offsets=offsets, covariates=covariates, grams=_grams(bounds, True), probe_count=probe_count, seed=5)
    parent = dual_solve.DualGaussian(source=_CountedSource(genotypes, bounds), **keywords)
    economy = outer_economy.EconomicalDualGaussian(source=_CountedSource(genotypes, bounds), **keywords)
    return parent, economy, (genotypes, covariates, training, noise, precision, shift, response, offsets)


def _scales(problem, precision=None, shift=None):
    genotypes, covariates, training, noise, base_precision, base_shift, response, offsets = problem
    precision = base_precision if precision is None else precision
    shift = base_shift if shift is None else shift
    return np.array([
        np.sqrt(float(mean @ posterior_precision @ mean))
        for posterior_precision, mean in (
            _dense_gaussian(genotypes, covariates, training, noise, precision, shift, response, offsets, model)[:2] for model in range(MODEL_COUNT)
        )
    ])


def _check_mean(economy, problem, bound, precision=None, shift=None):
    genotypes, covariates, training, noise, base_precision, base_shift, response, offsets = problem
    precision = base_precision if precision is None else precision
    shift = base_shift if shift is None else shift
    scales = _scales(problem, precision, shift)
    for model in range(MODEL_COUNT):
        posterior_precision, mean, alpha = _dense_gaussian(genotypes, covariates, training, noise, precision, shift, response, offsets, model)[:3]
        rounding = np.linalg.cond(posterior_precision) * genotypes.shape[1] * EPS * scales[model]
        error = economy.mean[:, model] - mean
        assert np.sqrt(float(error @ posterior_precision @ error)) <= bound[model] + rounding
        image = genotypes @ economy.mean[:, model]
        np.testing.assert_allclose(economy.genetic_image[:, model], image, rtol=0.0, atol=np.sqrt(EPS) * np.abs(image).max() * np.linalg.cond(posterior_precision))
        np.testing.assert_allclose(economy.alpha[:, model], alpha, rtol=0.0, atol=np.sqrt(EPS) * np.abs(alpha).max() * np.linalg.cond(posterior_precision))


def test_the_forcing_is_the_minimum_of_its_read_model() -> None:
    for contraction, rate, fixed in ((0.3, 0.2, 3.0), (0.8, 0.05, 5.0), (0.05, 1.0, 0.0), (0.95, 0.5, 2.0)):
        eta = outer_economy.optimal_forcing(contraction, rate, fixed)
        grid = np.linspace(0.0, 1.0 - contraction, 200_001)[1:-1]
        cost = (fixed + np.log1p(contraction / grid) / rate) / -np.log(contraction + grid)
        assert abs(eta - grid[np.argmin(cost)]) <= 2 * (grid[1] - grid[0])
    assert outer_economy.optimal_forcing(1.0, 0.3, 2.0) == 0.0
    assert outer_economy.optimal_forcing(0.5, 0.0, 2.0) == 0.0


def test_the_fused_mean_is_the_dense_posterior_in_fewer_reads() -> None:
    parent, economy, problem = _pair(51)
    genotypes, covariates, training, noise, precision, shift, response, offsets = problem
    bound = np.sqrt(EPS) * _scales(problem)
    keywords = dict(noise_variance=noise, error_bound=bound, probe_residual_ratio=np.sqrt(EPS))
    moved = shift * (1.0 + 1e-3 * np.random.default_rng(52).standard_normal(shift.shape))
    reads = {}
    for name, gaussian in (("parent", parent), ("economy", economy)):
        before = gaussian.source.reads
        for sites in (shift, moved):
            certificate = gaussian.iterate(site_precision=precision, site_shift=sites, **keywords)
            assert np.all(certificate.error_bound <= bound)
        reads[name] = gaussian.source.reads - before
    _check_mean(economy, problem, bound, shift=moved)
    for parent_solve, economy_solve in zip(parent.bulk_solves, economy.bulk_solves):
        np.testing.assert_array_equal(parent_solve.resolved, economy_solve.resolved)
        np.testing.assert_allclose(economy_solve.resolved_core, parent_solve.resolved_core, rtol=np.sqrt(EPS), atol=np.sqrt(EPS) * max(1.0, np.abs(parent_solve.resolved_core).max()))
        assert abs(parent_solve.bulk_trace - economy_solve.bulk_trace) <= np.sqrt(EPS) * max(abs(parent_solve.bulk_trace), 1.0) * genotypes.shape[1]
        for parent_values, economy_values in zip(parent_solve.resolved_cross.values, economy_solve.resolved_cross.values):
            np.testing.assert_allclose(economy_values, parent_values, rtol=0.0, atol=np.sqrt(EPS) * max(1.0, np.abs(parent_values).max() if parent_values.size else 1.0) * genotypes.shape[0])
    assert reads["economy"] < reads["parent"]


def test_a_frozen_pass_without_probes_keeps_the_refresh_quantities_and_warm_starts_the_next_refresh() -> None:
    _parent, economy, problem = _pair(55)
    genotypes, covariates, training, noise, precision, shift, response, offsets = problem
    bound = np.sqrt(EPS) * _scales(problem)
    keywords = dict(noise_variance=noise, error_bound=bound, probe_residual_ratio=np.sqrt(EPS))
    economy.iterate(site_precision=precision, site_shift=shift, **keywords)
    kept = economy.bulk_solves
    moved = shift * 1.001
    economy.iterate(site_precision=precision, site_shift=moved, with_probes=False, **keywords)
    assert economy.bulk_solves is kept
    _check_mean(economy, problem, bound, shift=moved)
    certificate = economy.iterate(site_precision=precision, site_shift=moved, **keywords)
    assert np.all(certificate.error_bound <= bound)
    _check_mean(economy, problem, bound, shift=moved)


def test_the_posterior_solve_is_the_dense_inverse_and_a_tightening_continues_without_re_reading_its_right_hand_side() -> None:
    parent, economy, problem = _pair(57, probe_count=2)
    genotypes, covariates, training, noise, precision, shift, response, offsets = problem
    for gaussian in (parent, economy):
        gaussian.iterate(site_precision=precision, site_shift=shift, noise_variance=noise, error_bound=np.full(MODEL_COUNT, np.sqrt(EPS)), probe_residual_ratio=np.sqrt(EPS))
    right = np.random.default_rng(58).standard_normal((genotypes.shape[1], 4))
    for model in (0, 1):
        posterior_precision = _dense_gaussian(genotypes, covariates, training, noise, precision, shift, response, offsets, model)[0]
        exact = np.linalg.solve(posterior_precision, right)
        scale = np.sqrt(np.einsum("pc,pq,qc->c", exact, posterior_precision, exact))
        rounding = np.linalg.cond(posterior_precision) * genotypes.shape[1] * EPS * scale
        loose, state = economy.posterior_solve_state(right, model, 1e-3 * scale)
        before = economy.count.passes
        tight, state = economy.posterior_solve_state(right, model, np.sqrt(EPS) * scale, state)
        labels = [label for label, _columns, _error in economy.count.records[before:]]
        assert "bulk-image" not in labels and not any(label.endswith(":exact") for label in labels)
        for solved, bound in ((loose, 1e-3 * scale), (tight, np.sqrt(EPS) * scale)):
            error = solved - exact
            assert np.all(np.sqrt(np.einsum("pc,pq,qc->c", error, posterior_precision, error)) <= bound + rounding)
        before = {name: gaussian.source.reads for name, gaussian in (("parent", parent), ("economy", economy))}
        parent.posterior_solve(right, model, np.sqrt(EPS) * scale)
        economy.posterior_solve(right, model, np.sqrt(EPS) * scale)
        assert economy.source.reads - before["economy"] < parent.source.reads - before["parent"]


def test_the_information_solve_matches_the_parent() -> None:
    parent, economy, problem = _pair(59, probe_count=2)
    genotypes, covariates, training, noise, precision, shift, response, offsets = problem
    for gaussian in (parent, economy):
        gaussian.iterate(site_precision=precision, site_shift=shift, noise_variance=noise, error_bound=np.full(MODEL_COUNT, np.sqrt(EPS)), probe_residual_ratio=np.sqrt(EPS))
    probes = np.random.default_rng(60).choice(np.array([-1.0, 1.0]), size=(genotypes.shape[1], 3))
    for model in (0, 2):
        expected, expected_coupling, _norms = parent.information_solve(probes, model, np.sqrt(EPS))
        back_products, coupling, residual_norm = economy.information_solve(probes, model, np.sqrt(EPS))
        assert np.all(residual_norm <= np.sqrt(EPS))
        scale = max(1.0, float(np.abs(expected).max()))
        np.testing.assert_allclose(back_products, expected, rtol=0.0, atol=1e-6 * scale)
        if coupling.size:
            np.testing.assert_allclose(coupling, expected_coupling, rtol=0.0, atol=1e-6 * max(1.0, float(np.abs(expected_coupling).max())))


def test_a_saved_state_restores_the_solver_without_a_read() -> None:
    _parent, economy, problem = _pair(61, probe_count=2)
    genotypes, covariates, training, noise, precision, shift, response, offsets = problem
    bound = np.sqrt(EPS) * _scales(problem)
    keywords = dict(noise_variance=noise, error_bound=bound, probe_residual_ratio=np.sqrt(EPS))
    # The oracle snapshots before the first mean solve too.
    economy.save()
    economy.iterate(site_precision=precision, site_shift=shift, **keywords)
    saved = economy.save()
    economy.iterate(site_precision=precision * 1.05, site_shift=shift * 0.95, **keywords)
    before = economy.source.reads
    economy.load(saved)
    assert economy.source.reads == before
    _check_mean(economy, problem, bound)
    # A posterior solve after the restore is at the restored sites.
    model = 1
    posterior_precision = _dense_gaussian(genotypes, covariates, training, noise, precision, shift, response, offsets, model)[0]
    right = np.random.default_rng(62).standard_normal((genotypes.shape[1], 2))
    exact = np.linalg.solve(posterior_precision, right)
    scale = np.sqrt(np.einsum("pc,pq,qc->c", exact, posterior_precision, exact))
    error = economy.posterior_solve(right, model, np.sqrt(EPS) * scale) - exact
    assert np.all(np.sqrt(np.einsum("pc,pq,qc->c", error, posterior_precision, error)) <= np.sqrt(EPS) * scale + np.linalg.cond(posterior_precision) * genotypes.shape[1] * EPS * scale)
