"""Small numerical counterexamples from the September 2026 accuracy audit."""
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from benchmarks.bench_sim.harness import auc
from benchmarks import svpgs_method
from sv_pgs import scale_mixture_ep as mixture
from sv_pgs.full_data_fit import covariate_residual_variance, moment_starts
from sv_pgs.small_n import dense_statistics, small_n_prior, small_n_start, _DenseFixedPoints


def test_auc_ties_are_order_invariant():
    for labels in ([0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 1]):
        assert auc(np.array(labels), np.zeros(4)) == 0.5
    assert auc(np.array([0, 1, 0, 1]), np.array([0, 1, 1, 2])) == 0.875
    with pytest.raises(ValueError, match="both"):
        auc(np.ones(4), np.arange(4))


def test_residual_variance_ignores_redundant_covariates_and_held_out_nan():
    outcome = np.array([1., 3., 2., 6., np.nan])[:, None]
    training = np.array([1., 1., 1., 1., 0.])[:, None]
    covariates = np.ones((5, 2))
    covariates[-1] = np.nan
    actual = covariate_residual_variance(outcome, training, covariates)
    np.testing.assert_allclose(actual, np.var(outcome[:4], ddof=1))


def test_dense_rank_and_zero_reliability_preserve_model():
    rng = np.random.default_rng(22)
    codes = rng.integers(0, 3, size=(12, 3), dtype=np.uint8) * 127
    target = rng.normal(size=12)
    offsets = np.array([0., -np.inf, 0.])
    redundant = dense_statistics(codes, np.ones((12, 2)), target, offsets)
    simple = dense_statistics(codes[:, [0, 2]], np.ones((12, 1)), target)
    assert redundant.covariate_rank == simple.covariate_rank == 1
    np.testing.assert_array_equal(redundant.active_rows, [0, 2])
    prior = small_n_prior(redundant, np.zeros(3, dtype=int), offsets, 64)
    simple_prior = small_n_prior(simple, np.zeros(2, dtype=int), np.zeros(2), 64)
    start, noise, moment = small_n_start(redundant, prior)
    _, simple_noise, simple_moment = small_n_start(simple, simple_prior)
    np.testing.assert_allclose(noise, simple_noise)
    np.testing.assert_allclose(moment.mean_variance, simple_moment.mean_variance)
    solver = _DenseFixedPoints(redundant, prior, start, noise, 64, 1 << 20)
    solver._refresh(start)
    assert np.all(np.isfinite(solver.mean))


def test_kernel_classes_are_streamed_before_allocating_over_budget(monkeypatch):
    classes = tuple(np.arange(group * 100, (group + 1) * 100) for group in range(10))
    prior = SimpleNamespace(class_rows=classes, grid_size=4, log_variance_grid=np.linspace(-2, 2, 4))
    original = mixture._KernelRows
    references = []
    def tracked(*args):
        value = original(*args)
        references.append(weakref.ref(value))
        return value
    monkeypatch.setattr(mixture, "_KernelRows", tracked)
    token = mixture._STEP_CACHE.set({})
    try:
        iterator = mixture._kernel_chunks(prior, np.zeros(1000), mixture.Cavity(np.ones(1000), np.ones(1000)), 64000)
        for _ in range(10):
            item = next(iterator)
            assert sum(reference() is not None for reference in references) <= 1
            del item
        assert "rows" not in mixture._STEP_CACHE.get()
        iterator.close()
    finally:
        mixture._STEP_CACHE.reset(token)


def test_benchmark_preserves_dosage_cn_and_reliability():
    values = np.array([[0., 0.], [0.5, 3.], [2., 6.]])
    codes, units = svpgs_method.bench_real_encoding(values)
    np.testing.assert_allclose(codes / units, values, atol=0.5 / units.min())
    np.testing.assert_array_equal(codes[-1], [254, 254])
    offsets = svpgs_method.bench_real_log_reliability(SimpleNamespace(reliability=np.array([1., 0.25, 0.])))
    np.testing.assert_allclose(offsets, [0., np.log(0.25), -np.inf])
    with pytest.raises(ValueError, match="defined"):
        svpgs_method.bench_real_log_reliability(SimpleNamespace(reliability=np.array([np.nan])))
    assert svpgs_method._training_seed(values, np.arange(3)) != svpgs_method._training_seed(np.floor(values), np.arange(3))


def test_real_stage0_tie_moment_start(tmp_path):
    from tests.test_dosage_store import _write_store
    from sv_pgs.dosage_store import DosageStore
    from sv_pgs.genotype_statistics import compute_genotype_statistics, DosageStoreTileSource
    from sv_pgs.compute_budget import ComputeBudget
    from sv_pgs.config import ModelConfig
    rng = np.random.default_rng(22)
    dosage = rng.integers(0, 3, size=(12, 3))
    dosage[:, 1] = dosage[:, 0]
    root = tmp_path / "store"
    _write_store(root, [{"chr22": dosage.T * 1000}])
    store = DosageStore.open(root)
    budget = ComputeBudget(device_kind="cpu", device_ids=(), device_names=(), device_bytes=(), device_compute_capabilities=(), host_bytes=1 << 27, cpu_threads=1)
    try:
        stats = compute_genotype_statistics(DosageStoreTileSource(store, np.arange(3)), np.arange(12), np.ones((12, 2)), rng.normal(size=(12, 1)), ModelConfig(minimum_minor_allele_frequency=0.), budget, 256, tmp_path / "ld")
        assert len(stats.active_rows) == 3 and len(stats.tie_map.kept_indices) == 2
        assert stats.covariate_rank == 1
        (start,) = moment_starts(stats, SimpleNamespace(variant_count=3, log_variance_offset=np.zeros(3)))
        assert np.isfinite(start.noise) and start.noise > 0.
    finally:
        store.close()


def test_calibration_does_not_square_the_design_condition_number():
    from sv_pgs.measurement_model import calibration_moments, fit_calibration_curve
    dosage = np.tile(np.array([0., 1., 2.]), (4, 1))
    design = np.column_stack([np.ones(4), 1e-9 * np.array([-1., 0., 1., 2.])])
    slope = np.array([1., 2., 3., 4.])
    moments = calibration_moments(dosage, slope[:, None] * dosage)
    curve = fit_calibration_curve(moments, np.zeros(4, dtype=int), design)
    fitted, _ = curve.predict(np.zeros(4, dtype=int), design)
    np.testing.assert_allclose(fitted, slope, atol=1e-12)


def test_mixture_intervals_use_draw_quantiles():
    from sv_pgs.fast_scoring import GeneticScores
    draws = np.array([[-9., -9., -9., 1., 1., 1., 1., 1., 1., 1.]])
    scores = GeneticScores(means=np.array([[-2.]]), variances=np.array([[21.]]), draw_counts=(10,), draws=(draws,), gaussian_posteriors=(False,))
    lower, upper = scores.credible_interval(0.8)
    np.testing.assert_array_equal(lower, [[-9.]])
    np.testing.assert_array_equal(upper, [[1.]])
