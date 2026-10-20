"""The phase timing lines (``phase_timer``) are in the one form benchmarks/profile_log.py reads, hand every answer
through, and never stop or change the work they time.

The device sweep's busy shares are checked here against a stand-in for CuPy's events; test_phase_timer_cuda times real
kernels on a device.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import profile_log
from sv_pgs import phase_timer
from sv_pgs.phase_timer import SWEEP_BUSY, TimedOracle, launch_timer, timed_sweep, timing


def _timing_lines(capsys) -> list[profile_log.Phase]:
    return profile_log.parse(capsys.readouterr().err.splitlines()).phases


def test_a_timing_line_reads_back_as_its_phase_seconds_and_extras(capsys) -> None:
    timing("stage0", 2581.04, blocks=55, ld_gb=150.123456, root="run/step one")
    (phase,) = _timing_lines(capsys)
    assert (phase.name, phase.seconds) == ("stage0", 2581.0)
    # floats to four significant digits, anything else one token
    assert phase.extra == {"blocks": "55", "ld_gb": "150.1", "root": "run/step_one"}


def test_the_oracle_s_rounds_and_whole_are_timed_and_its_answers_pass_through(capsys) -> None:
    answers = []

    def oracle(hyperparameters):
        time.sleep(0.02)
        answers.append(list(hyperparameters))
        return [value * 2 for value in hyperparameters]

    timed = TimedOracle(oracle)
    assert timed([1.0, 2.0]) == [2.0, 4.0]
    time.sleep(0.03)
    assert timed([3.0]) == [6.0]
    timed.close()
    assert answers == [[1.0, 2.0], [3.0]] and timed.rounds == 2
    phases = _timing_lines(capsys)
    assert [phase.name for phase in phases] == ["eb_iteration", "eb_iteration", "eb"]
    first, second, whole = phases
    assert [first.extra["round"], second.extra["round"]] == ["1", "2"] and whole.extra["rounds"] == "2"
    # a round is one call to the next: its solve the call's own time, the rest what the loop did between calls
    assert float(first.extra["solve"]) >= 0.02 and float(first.extra["other"]) >= 0.03
    assert float(whole.extra["solve"]) >= 0.04


def test_an_oracle_s_failure_reaches_the_loop_and_the_rounds_before_it_are_still_written(capsys) -> None:
    def failing(hyperparameters):
        raise FloatingPointError("no fixed point")

    timed = TimedOracle(failing)
    with pytest.raises(FloatingPointError, match="no fixed point"):
        timed([1.0])
    timed.close()
    assert [(phase.name, phase.extra["round" if phase.name == "eb_iteration" else "rounds"]) for phase in _timing_lines(capsys)] == [
        ("eb_iteration", "1"), ("eb", "1"),
    ]


def test_a_sweep_is_not_timed_unless_asked_and_on_a_device(monkeypatch, capsys) -> None:
    monkeypatch.delenv(SWEEP_BUSY, raising=False)
    with timed_sweep(_FakeCupy()):
        assert launch_timer() is phase_timer._UNTIMED
    for every in ("0", "-3", "often", ""):
        monkeypatch.setenv(SWEEP_BUSY, every)
        with timed_sweep(_FakeCupy()):
            assert launch_timer() is phase_timer._UNTIMED
    # NumPy (the host route) has no CUDA events
    monkeypatch.setenv(SWEEP_BUSY, "1")
    with timed_sweep(np):
        assert launch_timer() is phase_timer._UNTIMED
    assert capsys.readouterr().err == ""
    mark = launch_timer().before()
    assert mark is None and launch_timer().after(mark) is None


class _FakeEvent:
    def __init__(self, cupy: "_FakeCupy") -> None:
        self.cupy = cupy

    def record(self) -> None:
        if self.cupy.broken:
            raise RuntimeError("cudaErrorLaunchFailure")
        self.at = self.cupy.clock

    def synchronize(self) -> None:
        return None


class _FakeCupy:
    """CuPy's event surface the timer uses, on a clock the test advances: each launch runs ``kernel`` seconds on the
    current device."""

    def __init__(self, broken: bool = False) -> None:
        self.broken = broken
        self.clock = 0.0
        self.device = 0
        self.cuda = SimpleNamespace(
            Event=lambda: _FakeEvent(self),
            runtime=SimpleNamespace(getDevice=lambda: self.device),
            Device=self._device,
            get_elapsed_time=lambda start, end: (end.at - start.at) * 1e3,
        )

    @contextmanager
    def _device(self, index: int):
        yield

    def launch(self, kernel: float) -> None:
        timer = launch_timer()
        mark = timer.before()
        self.clock += kernel
        timer.after(mark)


def test_a_sampled_sweep_estimates_each_device_s_kernel_seconds(monkeypatch, capsys) -> None:
    cupy = _FakeCupy()
    monkeypatch.setenv(SWEEP_BUSY, "1")
    with timed_sweep(cupy):
        for device in (0, 1):
            cupy.device = device
            for _ in range(10):
                cupy.launch(0.001)
        time.sleep(0.05)
    (phase,) = _timing_lines(capsys)
    assert phase.name == "device_sweep" and phase.extra["launches"] == "20" and phase.extra["sampled"] == "20"
    # every launch timed: each device's 10 ms of kernels over the sweep's wall
    for device in ("gpu0", "gpu1"):
        assert 0.01 / 5.0 < float(phase.extra[device]) <= 0.01 / 0.05
    assert launch_timer() is phase_timer._UNTIMED


def test_one_launch_in_k_is_sampled_and_each_stands_for_k(monkeypatch) -> None:
    cupy = _FakeCupy()
    busy = phase_timer.DeviceBusy(cupy, 8)
    for _ in range(4000):
        mark = busy.before()
        cupy.clock += 0.001
        busy.after(mark)
    assert busy.launches == 4000 and 4000 / 8 * 0.8 < busy.sampled < 4000 / 8 * 1.2
    assert busy.kernel_seconds()[0] == pytest.approx(busy.sampled * 0.001 * 8)


def test_a_timer_that_cannot_record_reports_why_and_never_stops_the_sweep(monkeypatch, capsys) -> None:
    cupy = _FakeCupy(broken=True)
    monkeypatch.setenv(SWEEP_BUSY, "1")
    with timed_sweep(cupy):
        for _ in range(3):
            cupy.launch(0.001)
    (phase,) = _timing_lines(capsys)
    assert phase.extra == {"launches": "3", "unavailable": "RuntimeError"}


def test_a_failing_sweep_raises_unchanged_and_writes_no_line(monkeypatch, capsys) -> None:
    monkeypatch.setenv(SWEEP_BUSY, "1")
    with pytest.raises(ValueError, match="sweep"):
        with timed_sweep(_FakeCupy()):
            raise ValueError("sweep")
    assert capsys.readouterr().err == "" and launch_timer() is phase_timer._UNTIMED
