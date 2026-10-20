"""Changing worker scheduling preserves numerical results and the caller's mask."""
import threading

import numba
import numpy as np
import pytest

from sv_pgs import host_kernels
from sv_pgs.code_products import _fp64_codes
from sv_pgs.host_kernels import row_kernel
from sv_pgs.scale_mixture_ep import _LOST_BELOW, _moving_line_rows, _objective_rows


def test_kernel_schedule_is_measured_cached_and_restores_worker_mask(monkeypatch):
    workers = numba.get_num_threads()
    if workers == 1:
        pytest.skip("needs more than one configured Numba worker")
    ticks = iter([0., 1., 2., 5.])
    monkeypatch.setattr(host_kernels.time, "perf_counter", lambda: next(ticks))
    calls = []

    def kernel():
        calls.append(numba.get_num_threads())
        return False

    assert row_kernel(kernel, 1) is False
    assert calls == [workers, 1, workers]
    assert numba.get_num_threads() == workers
    assert row_kernel(kernel, 1) is False
    assert calls[-1] == 1
    assert numba.get_num_threads() == workers

    def fail():
        if numba.get_num_threads() == 1:
            raise ValueError("test failure")

    monkeypatch.setattr(host_kernels.time, "perf_counter", lambda: 0.)
    with pytest.raises(ValueError, match="test failure"):
        row_kernel(fail, 1)
    assert numba.get_num_threads() == workers


@pytest.mark.parametrize("rows", [16, 600])
def test_worker_scheduling_preserves_likelihood_rows_bitwise(rows):
    rng = np.random.default_rng(817)
    grid = np.linspace(-9., 4., 33)
    steps = np.linspace(-1., 1., 15)
    density = rng.normal(size=(steps.size, grid.size))
    density -= np.logaddexp.reduce(density, axis=1)[:, None]
    scales, slopes, shift = rng.normal(size=(3, rows))
    precision = np.exp(rng.normal(size=rows))
    # Exercise the log-space fallback as well as ordinary finite scales.
    scales[:2], slopes[:2] = [700., -760.], 0.
    args = (density, scales, slopes, steps, grid, precision, shift, _LOST_BELOW)
    expected, actual = np.empty((rows, steps.size)), np.empty((rows, steps.size))
    bad = _moving_line_rows(*args, expected)
    assert row_kernel(_moving_line_rows, rows * steps.size * grid.size, *args, actual) == bad
    np.testing.assert_array_equal(actual, expected)

    def outputs():
        return (np.empty((rows, grid.size + 1)), np.empty((rows, grid.size + 1)), np.empty(rows), np.empty(rows))

    expected_outputs, actual_outputs = outputs(), outputs()
    args = (density[0], np.exp(density[0]), grid, scales, precision, shift)
    bad = _objective_rows(*args, *expected_outputs)
    assert row_kernel(_objective_rows, rows * grid.size, *args, *actual_outputs) == bad
    for actual, expected in zip(actual_outputs, expected_outputs):
        np.testing.assert_array_equal(actual, expected)


def test_parallel_launches_from_several_threads_take_turns():
    """The stage-0 host pool converts code tiles on several threads at once. Under Numba's workqueue threading layer
    (macOS, or a Linux host without libgomp such as the Batch VM image) two overlapping launches abort the process."""
    codes = np.random.default_rng(0).integers(0, 256, size=(64, 500), dtype=np.uint8)
    expected = codes.astype(np.float64)
    failures = []

    def convert():
        for _ in range(50):
            if not np.array_equal(_fp64_codes(codes, 0, codes.shape[1]), expected):
                failures.append("converted codes differ")

    threads = [threading.Thread(target=convert) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == []
