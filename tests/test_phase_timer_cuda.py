"""A timed device sweep (``phase_timer.timed_sweep`` with SVPGS_SWEEP_BUSY set) times real kernels with CUDA events and
leaves the sweep's every value as an untimed sweep gives it."""
from __future__ import annotations

import numpy as np
import pytest

from benchmarks import profile_log
from sv_pgs.compute_budget import _try_import_cupy
from sv_pgs.phase_timer import SWEEP_BUSY, launch_timer, timed_sweep

cupy = _try_import_cupy()
pytestmark = pytest.mark.skipif(cupy is None, reason="needs a CUDA device")


def _sweeps(data) -> tuple[np.ndarray, dict]:
    from sv_pgs.device_sweep import PanelGrams, sweep_piece

    count = len(data["squares"])
    state = {name: cupy.zeros(count) for name in ("mean", "variance", "shift", "third", "fourth")}
    residual = cupy.asarray(data["target"])
    dense = cupy.asarray(data["dense"])
    grams = PanelGrams()
    for _sweep_index in range(3):
        with timed_sweep(cupy):
            sweep_piece(
                cupy, decode=lambda first, last: dense[:, first:last], width=count, mask=cupy.asarray(data["mask"]),
                covariates=cupy.asarray(data["covariates"]), covariate_pinv=cupy.asarray(data["solve"]), residual=residual,
                grams=grams, model=0, members=np.arange(count), squares=cupy.asarray(data["squares"]),
                class_index=cupy.asarray(data["classes"]), log_density=cupy.asarray(data["log_density"]),
                node_variance=cupy.exp(cupy.asarray(data["log_node_variance"])), log_node_variance=cupy.asarray(data["log_node_variance"]),
                noise=0.9, pieces=cupy.zeros((count, 3)), **state,
            )
    return cupy.asnumpy(residual), {name: cupy.asnumpy(values) for name, values in state.items()}


def test_a_timed_sweep_is_the_untimed_sweep_and_reports_its_device_s_busy_share(monkeypatch, capsys) -> None:
    from tests.test_device_sweep_cuda import _problem

    data = _problem(5, 400, 3 * 32 + 7, 41)
    monkeypatch.delenv(SWEEP_BUSY, raising=False)
    untimed = _sweeps(data)
    assert "timing:" not in capsys.readouterr().err
    monkeypatch.setenv(SWEEP_BUSY, "1")
    timed = _sweeps(data)
    # the events are recorded beside the launches: the same kernels on the same inputs, so the same bits
    assert np.array_equal(timed[0], untimed[0]) and all(np.array_equal(timed[1][name], untimed[1][name]) for name in untimed[1])
    phases = profile_log.parse(capsys.readouterr().err.splitlines()).phases
    assert [phase.name for phase in phases] == ["device_sweep"] * 3
    device = f"gpu{int(cupy.cuda.runtime.getDevice())}"
    for phase in phases:
        # every step launch timed (one per panel); a share of the wall, the launch latency on an idle stream included
        assert int(phase.extra["sampled"]) == int(phase.extra["launches"]) > 0
        assert 0.0 < float(phase.extra[device]) <= 1.05


def test_a_sampled_timer_times_a_real_kernel(monkeypatch, capsys) -> None:
    monkeypatch.setenv(SWEEP_BUSY, "4")
    values = cupy.ones(1 << 22)
    with timed_sweep(cupy):
        for _ in range(64):
            timer = launch_timer()
            mark = timer.before()
            values *= 1.0
            timer.after(mark)
        cupy.cuda.Device().synchronize()
    (phase,) = profile_log.parse(capsys.readouterr().err.splitlines()).phases
    assert phase.extra["launches"] == "64" and 0 < int(phase.extra["sampled"]) < 64 and "unavailable" not in phase.extra
