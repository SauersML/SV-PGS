"""Parseable phase timing in the progress log, which benchmarks/profile_log.py reads into a run's phase timeline.

A whole-genome fit's one window while it runs is its progress log (launcher/workspace/run_step.sh copies the driver's
log to the ops bucket every five minutes), and its lines said what a phase did, not in one form how long it took. Each
major phase now ends with one line in a fixed form,

    timing: <phase> <seconds> s [<name>=<value> ...]

the phase one word, the seconds its wall time, and each extra a duration, a size, a rate, a share or a count of the
run's own work (sweeps, rounds, blocks, models), never a count of people and never a value of the data. The line is
written when the phase ends, so the phase began ``seconds`` before it and the timeline needs no start lines. The phases
are the store's restore (``run_mirror.restore``) and digest (``content_identity.tree_digest``), Stage 0
(``genotype_statistics.compute_genotype_statistics``), the far field and the summary band (``gram_space``), the resident
genotype codes' fill and each later streamed read (``store_block_source``), each mean-field solve with its sweeps and
its seconds per sweep, the mean-field oracle's dual iterate and each response solve with their reads (``full_data_fit``),
each empirical Bayes round and the empirical Bayes as a whole (``TimedOracle``), each scoring
batch (``workspace_pipeline._score_step``), and, where SWEEP_BUSY is set, each device sweep's busy share
(``timed_sweep``). launcher/workspace/sample_resources.py writes the VM's utilization into the same log once a minute.

Nothing here changes what a fit computes: the lines are written beside its work, and the oracle wrapper hands every
answer through unchanged.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from sv_pgs.progress import log

GB = 1e9
"""Bytes per GB, the unit of the timing lines' sizes (as of the progress lines')."""
SWEEP_BUSY = "SVPGS_SWEEP_BUSY"
"""The environment variable that, set to a whole number k, has each device sweep time one kernel launch in k (chosen at
random) with CUDA events and log the devices' busy shares (``timed_sweep``); unset, no launch is timed."""


def _shown(value: object) -> str:
    """An extra's value in one token: a float to four significant digits, anything else as written."""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value).replace(" ", "_")


def timing(phase: str, seconds: float, **extra: object) -> None:
    """Log the line ``timing: <phase> <seconds> s`` with each extra as ``name=value``, in its keyword order."""
    log(f"timing: {phase} {seconds:.1f} s" + "".join(f" {name}={_shown(value)}" for name, value in extra.items()))


class TimedOracle:
    """An empirical Bayes oracle (``scale_mixture_ep.fit_hyperparameters``' ``fixed_points``) that times the outer loop's
    rounds by its calls: the loop asks it once per pass, every model's trial solved together, so from one call's start to
    the next is one round, the call's own wall time that round's solve, and the rest its curvature, its hyper steps and
    its trials' bookkeeping. The curvature's share is in the round's ``eb curvature:`` lines, and whether the round was a
    halving of a refused step (the line search) in its ``joint trial at fraction`` lines: benchmarks/profile_log.py
    reads both between one round's line and the next. A round's line is written at the next call; ``close`` writes the
    last and the loop's own total. The answers pass through unchanged, so the fit is the one an unwrapped oracle gives."""

    def __init__(self, oracle: Callable[[Sequence[Any]], Sequence[Any]], phase: str = "eb") -> None:
        self._oracle = oracle
        self._phase = phase
        self._opened = time.perf_counter()
        self._round_start: float | None = None
        self._solve = 0.0
        self._solves = 0.0
        self.rounds = 0

    def __call__(self, hyperparameters: Sequence[Any]) -> Sequence[Any]:
        called = time.perf_counter()
        self._close_round(called)
        self._round_start = called
        try:
            return self._oracle(hyperparameters)
        finally:
            self._solve = time.perf_counter() - called
            self._solves += self._solve

    def _close_round(self, now: float) -> None:
        if self._round_start is None:
            return
        self.rounds += 1
        seconds = now - self._round_start
        timing(f"{self._phase}_iteration", seconds, round=self.rounds, solve=self._solve, other=seconds - self._solve)
        self._round_start = None

    def close(self) -> None:
        """The last round's line and the loop's: its wall time from this wrapper's making, its rounds and their solves."""
        now = time.perf_counter()
        self._close_round(now)
        timing(self._phase, now - self._opened, rounds=self.rounds, solve=self._solves)


class _Untimed:
    """The launch hooks' stand-in where no sweep is timed: each hook returns at once."""

    def before(self) -> None:
        return None

    def after(self, mark: object) -> None:
        return None


_UNTIMED = _Untimed()
_ACTIVE: ContextVar[Any] = ContextVar("timed_sweep", default=_UNTIMED)


def launch_timer() -> Any:
    """The current device sweep's launch timer for a kernel launch's hooks, ``mark = timer.before()`` just before the
    launch and ``timer.after(mark)`` just after it: the sweep's ``DeviceBusy`` inside ``timed_sweep``, else a stand-in
    whose hooks cost a call each."""
    return _ACTIVE.get()


class DeviceBusy:
    """A device sweep's kernel time, sampled: each launch is timed with probability 1 / ``every`` by a CUDA event pair on
    its device's current stream, and each timed launch stands for ``every`` launches, so a device's summed event times
    times ``every`` estimate its kernel seconds and their share of the sweep's wall time its busy share. The rest of the
    wall is the device waiting on the host (the launches' Python, decoding, the moments' copies back). On a stream with
    queued work the pair times the kernel alone; on an idle one it also holds the launch's own latency between the two
    records, so a share of short kernels issued one at a time reads high, never low. A sampled launch costs two event
    records and one elapsed-time read after the sweep; any other launch a uniform draw. A timer that cannot make or read
    an event stops sampling and reports no share: it never stops the sweep it times."""

    def __init__(self, cupy: Any, every: int) -> None:
        self._cupy = cupy
        self._every = every
        self._draws = random.Random(every)
        self._pairs: list[tuple[int, Any, Any]] = []
        self.failed: str | None = None
        self.launches = 0

    def before(self) -> tuple[int, Any] | None:
        self.launches += 1
        if self.failed is not None or self._draws.random() * self._every >= 1.0:
            return None
        try:
            start = self._cupy.cuda.Event()
            start.record()
            return int(self._cupy.cuda.runtime.getDevice()), start
        except Exception as error:  # the timer's own failure, never the sweep's
            self.failed = type(error).__name__
            return None

    def after(self, mark: tuple[int, Any] | None) -> None:
        if mark is None or self.failed is not None:
            return
        try:
            end = self._cupy.cuda.Event()
            end.record()
        except Exception as error:  # the timer's own failure, never the sweep's
            self.failed = type(error).__name__
            return
        self._pairs.append((mark[0], mark[1], end))

    @property
    def sampled(self) -> int:
        return len(self._pairs)

    def kernel_seconds(self) -> dict[int, float]:
        """Each device's estimated kernel seconds (the sampled pairs' elapsed times, each times ``every``), read on the
        pair's own device once its end event has run."""
        seconds: dict[int, float] = {}
        for device, start, end in self._pairs:
            with self._cupy.cuda.Device(device):
                end.synchronize()
                # get_elapsed_time reports milliseconds.
                seconds[device] = seconds.get(device, 0.0) + self._cupy.cuda.get_elapsed_time(start, end) / 1e3 * self._every
        return seconds


@contextmanager
def timed_sweep(array_module: Any) -> Iterator[None]:
    """One device sweep, timed for its devices' busy shares where SWEEP_BUSY names a whole number k: the launches the
    sweep's hooks see (``launch_timer``) are sampled one in k (``DeviceBusy``), and at the sweep's end one line
    ``timing: device_sweep <wall> s launches=.. sampled=.. gpu<i>=<busy share>``. Elsewhere, on the host, or where the
    sweep raises, nothing is timed or written."""
    every = os.environ.get(SWEEP_BUSY, "")
    if not every.isdigit() or int(every) < 1 or not hasattr(array_module, "cuda"):
        yield
        return
    busy = DeviceBusy(array_module, int(every))
    token = _ACTIVE.set(busy)
    started = time.perf_counter()
    try:
        yield
    finally:
        _ACTIVE.reset(token)
    wall = time.perf_counter() - started
    if busy.failed is None:
        try:
            seconds = busy.kernel_seconds()
        except Exception as error:  # the timer's own failure, never the sweep's
            busy.failed = type(error).__name__
    if busy.failed is not None or not wall > 0.0:
        timing("device_sweep", wall, launches=busy.launches, unavailable=busy.failed)
        return
    timing("device_sweep", wall, launches=busy.launches, sampled=busy.sampled, **{f"gpu{device}": seconds[device] / wall for device in sorted(seconds)})
