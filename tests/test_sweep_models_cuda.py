"""The batched mean-field sweep (``device_sweep.sweep_models``) on CUDA devices: its kernels against the host sweep
(``mean_field._sweep``) and their numpy reference (``HostPanelKernels``) on the same problem, a model's sweep the same
to the bit whichever models share it, a layout over two devices (the residuals handed between them) the one-device
sweep's, the oracle's batched sweeps and calls on a store's held codes against the host oracle's, and the timing script
on a small problem."""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from sv_pgs.compute_budget import _try_import_cupy
from sv_pgs.device_sweep import MOMENTS, PANEL, CudaPanelKernels, HostPanelKernels, PanelGrams
from tests.test_sweep_models import _NODES, _layout, batched_sweep, host_reference, sweep_problem

cupy = _try_import_cupy()
DEVICE_COUNT = 0 if cupy is None else int(cupy.cuda.runtime.getDeviceCount())
pytestmark = pytest.mark.skipif(DEVICE_COUNT == 0, reason="needs a CUDA device")


def _sweeps(xp, kernels, problem, models, count):
    """``count`` batched sweeps of ``models`` from zero moments and the problem's residuals: each sweep's (residuals,
    totals, residual squares, moments)."""
    layout = _layout(xp, problem)
    members = sum(block.members.shape[0] for block in layout.blocks)
    state = {name: np.zeros((members, len(models))) for name in MOMENTS}
    residuals = problem["residuals"][list(models)].copy()
    grams, outcomes = PanelGrams(), []
    for _sweep in range(count):
        swept, totals, squares = batched_sweep(xp, kernels, problem, layout, models, state, residuals, grams)
        residuals = np.stack(swept)
        outcomes.append((residuals.copy(), totals.copy(), squares.copy(), {name: values.copy() for name, values in state.items()}))
    return outcomes


def test_the_cuda_sweep_is_the_host_sweep_and_its_numpy_reference() -> None:
    with cupy.cuda.Device(0):
        problem = sweep_problem(cupy, 3, devices=(0, 0, 0))
        kernels = CudaPanelKernels(cupy, _NODES)
        device = _sweeps(cupy, kernels, problem, (0, 1, 2), 3)
        reference = _sweeps(np, HostPanelKernels(kernels.threads(problem["masks"].shape[1])), sweep_problem(np, 3, devices=(0, 0, 0)), (0, 1, 2), 3)
    order = np.concatenate(problem["member_blocks"])
    host = [{name: np.zeros(order.shape[0]) for name in MOMENTS} for _model in range(3)]
    host_residuals = [problem["residuals"][model].copy() for model in range(3)]
    for (residuals, totals, squares, state), (_r, reference_totals, _s, reference_state) in zip(device, reference):
        for model in range(3):
            parts = host_reference(problem, model, order, host[model], host_residuals[model])
            # The same members in the same order: equal to rounding (fused multiply-adds, other summation orders).
            for name in MOMENTS:
                expected = host[model][name]
                scale = 1.0 + np.max(np.abs(expected))
                np.testing.assert_allclose(state[name][order, model], expected, rtol=1e-9, atol=1e-11 * scale, err_msg=name)
                np.testing.assert_allclose(state[name][:, model], reference_state[name][:, model], rtol=1e-10, atol=1e-12 * scale, err_msg=name)
            np.testing.assert_allclose(residuals[model], host_residuals[model], rtol=1e-9, atol=1e-10 * np.linalg.norm(host_residuals[model]))
            assert totals[model, 0] == pytest.approx(parts[0], rel=1e-9, abs=1e-9 * parts[3])
            assert totals[model, 1] == pytest.approx(parts[1], rel=1e-9)
            assert squares[model] == pytest.approx(parts[2], rel=1e-9)
            np.testing.assert_allclose(totals[model], reference_totals[model], rtol=1e-10, atol=1e-10 * parts[3])


def test_a_model_s_cuda_sweep_is_the_same_whichever_models_share_it() -> None:
    with cupy.cuda.Device(0):
        problem = sweep_problem(cupy, 5, devices=(0, 0, 0))
        kernels = CudaPanelKernels(cupy, _NODES)
        together = _sweeps(cupy, kernels, problem, (0, 1, 2), 3)
        for model in range(3):
            for (residuals, totals, squares, state), (alone_residuals, alone_totals, alone_squares, alone_state) in zip(
                together, _sweeps(cupy, kernels, problem, (model,), 3),
            ):
                np.testing.assert_array_equal(residuals[model], alone_residuals[0])
                np.testing.assert_array_equal(totals[model], alone_totals[0])
                assert squares[model] == alone_squares[0]
                for name in MOMENTS:
                    np.testing.assert_array_equal(state[name][:, model], alone_state[name][:, 0])


@pytest.mark.skipif(DEVICE_COUNT < 2, reason="needs two CUDA devices")
def test_a_sweep_over_two_devices_is_the_one_device_sweep() -> None:
    """Blocks on devices 0, 1 and 0: each block's launches on its own device, the residuals handed over between them.
    The members' arithmetic is the one-device sweep's to the bit (same kernels, same sample blocks, exact copies); the
    ELBO pieces are summed per device, so their totals agree to rounding."""
    with cupy.cuda.Device(0):
        one = _sweeps(cupy, CudaPanelKernels(cupy, _NODES), sweep_problem(cupy, 9, devices=(0, 0, 0)), (0, 1, 2), 3)
        two = _sweeps(cupy, CudaPanelKernels(cupy, _NODES), sweep_problem(cupy, 9, devices=(0, 1, 0)), (0, 1, 2), 3)
    for (residuals, totals, squares, state), (two_residuals, two_totals, two_squares, two_state) in zip(one, two):
        np.testing.assert_array_equal(two_residuals, residuals)
        np.testing.assert_array_equal(two_squares, squares)
        np.testing.assert_allclose(two_totals, totals, rtol=1e-12, atol=1e-12 * np.max(np.abs(totals)))
        for name in MOMENTS:
            np.testing.assert_array_equal(two_state[name], state[name])


def _cuda_budget(device_count: int):
    from sv_pgs.compute_budget import detect_compute_budget

    budget = detect_compute_budget()
    return replace(
        budget, device_ids=budget.device_ids[:device_count], device_names=budget.device_names[:device_count],
        device_bytes=budget.device_bytes[:device_count], device_compute_capabilities=budget.device_compute_capabilities[:device_count],
    )


def _store_oracles(tmp_path: Path, budget):
    """test_full_data_fit's two-trait oracle over one store and Stage 0, built on the host and on ``budget``'s CUDA
    devices (``stage2_wiring.stage2_source``: streamed on one, sharded over two)."""
    from sv_pgs.config import ModelConfig
    from sv_pgs.dual_solve import DualGaussian, StreamedDualSource
    from sv_pgs.full_data_fit import _FullDataMeanField, block_grams, covariate_residual_variance, moment_starts, stage0_lattice
    from sv_pgs.genotype_statistics import DosageStoreTileSource, compute_genotype_statistics
    from sv_pgs.scale_mixture_ep import initial_hyperparameters, scale_mixture_prior
    from sv_pgs.stage2_wiring import stage2_source
    from sv_pgs.store_block_source import StoreGenotypeBlockSource
    from tests.test_full_data_fit import _BLOCK_CAP, _DRAWS, _SAMPLES, _TRAINING, _WORKSPACE_BYTES, _budget, _store

    store, covariate, targets, genetic = _store(tmp_path / "store", 7)
    generator = np.random.default_rng(3)
    traits = np.column_stack([targets, 0.6 * genetic - 0.2 * covariate + 0.8 * generator.standard_normal(_SAMPLES)])
    training = np.arange(_TRAINING)
    statistics = compute_genotype_statistics(
        DosageStoreTileSource(store, np.arange(store.n_variants)), training, np.column_stack([np.ones(_TRAINING), covariate[training]]),
        traits[training], ModelConfig(), _budget(), _BLOCK_CAP, tmp_path / "ld",
    )
    members = statistics.active_rows.shape[0]
    covariates = np.column_stack([np.ones(_SAMPLES), covariate])
    mask = np.zeros((_SAMPLES, 2))
    mask[training] = 1.0
    noise = covariate_residual_variance(traits, mask, covariates)
    nodes, floor, top = stage0_lattice(statistics, 0, float(noise[0]), np.zeros(members), 0.5 / _DRAWS)
    prior = scale_mixture_prior(
        class_index=np.zeros(members, dtype=np.int64), log_variance_offset=np.zeros(members), annotation_design=np.zeros((members, 0)),
        annotation_groups=(), nodes=nodes, floor=floor, top=top,
    )
    starts = [initial_hyperparameters(prior, moment.mean_variance) for moment in moment_starts(statistics, prior)]
    sources = {
        "host": StreamedDualSource(StoreGenotypeBlockSource.from_statistics(store, statistics, _budget(), _WORKSPACE_BYTES)),
        "cuda": stage2_source(store, statistics, budget, _WORKSPACE_BYTES, np.arange(_SAMPLES)),
    }

    def oracle(kind: str) -> _FullDataMeanField:
        gaussian = DualGaussian(
            source=sources[kind], training=mask, targets=traits, offsets=np.zeros((_SAMPLES, 2)), covariates=covariates,
            grams=block_grams(statistics), probe_count=_DRAWS, seed=11,
        )
        return _FullDataMeanField(gaussian, statistics, prior, _DRAWS, 1 << 22, 13, starts, noise)

    return oracle, starts


@pytest.mark.parametrize("devices", [1, 2])
def test_the_oracle_s_batched_sweeps_and_calls_on_held_codes_are_the_host_oracle_s(tmp_path: Path, devices: int) -> None:
    from sv_pgs.memory_broker import memory_scope

    if DEVICE_COUNT < devices:
        pytest.skip("needs two CUDA devices")
    budget = _cuda_budget(devices)
    # Inside the fit's ledger, as the fits run: the panel Grams are admitted to each device's own pool.
    with memory_scope(budget):
        _batched_sweeps_and_calls(*_store_oracles(tmp_path, budget))


def _batched_sweeps_and_calls(oracle_of, starts) -> None:
    host, device = oracle_of("host"), oracle_of("cuda")
    requests = {0: starts[0], 1: starts[1]}
    for _sweep in range(3):
        expected = {model: host._sweep(model, hyperparameters) for model, hyperparameters in requests.items()}
        swept = device._panel_sweeps(requests)
        assert swept is not None, "the store's codes did not take the batched route"
        for model in requests:
            np.testing.assert_allclose(swept[model][1:], expected[model][1:], rtol=1e-9)
            assert swept[model][0] == pytest.approx(expected[model][0], rel=1e-9, abs=1e-9 * expected[model][3])
            for name in MOMENTS:
                values = getattr(host, name)[:, model]
                np.testing.assert_allclose(getattr(device, name)[:, model], values, rtol=1e-9, atol=1e-11 * (1.0 + np.max(np.abs(values))), err_msg=name)
    assert device.passes == 3 and host.passes == 6
    # Whole calls (the second adds the cold solves): the same fixed points to the smoke tolerance of a fit.
    host, device = oracle_of("host"), oracle_of("cuda")
    for _call in range(2):
        assert all(point is not None for point in host(list(starts))) and all(point is not None for point in device(list(starts)))
        np.testing.assert_allclose(device.mean, host.mean, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(device.noise, host.noise, rtol=1e-4)


@pytest.mark.parametrize("devices", [1, 2])
def test_the_sweep_benchmark_passes_on_a_small_problem(tmp_path: Path, devices: int) -> None:
    if DEVICE_COUNT < devices:
        pytest.skip("needs two CUDA devices")
    environment = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    script = "import sys; sys.modules['pytest'] = None; from benchmarks.sweep_bench import main; raise SystemExit(main(sys.argv[1:]))"
    completed = subprocess.run(
        [sys.executable, "-c", script, "--samples", "4000", "--members", str(64 * PANEL), "--blocks", "3", "--devices", str(devices)], env=environment,
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "FAIL " not in completed.stdout and "PASS sweep_bench " in completed.stdout
