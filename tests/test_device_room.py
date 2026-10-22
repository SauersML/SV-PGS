"""Line passes and kernel chunks use current device room and recover from refused allocations on CPU fakes."""
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from sv_pgs import scale_mixture_ep as engine
from tests.test_scale_mixture_ep import _problem, _hyperparameters, _WORKING_BYTES
from tests.test_memory_broker import _metered_device, _UNIT


def test_device_room_returns_idle_pool_blocks_but_does_not_count_resident_codes(monkeypatch):
    broker, allocate, state = _metered_device(1000)
    allocate(600)
    state.update(used=400, cached=200)
    dropped = []
    assert broker.admit("device0", 400 * _UNIT, "codes", lambda: dropped.append(True), allocated=True)
    monkeypatch.setattr(engine, "current_broker", lambda: broker)
    xp = SimpleNamespace(cuda=SimpleNamespace(Device=lambda: SimpleNamespace(id=0)))
    with engine.device_scope(xp):
        assert engine._device_room(1000 * _UNIT) == 600 * _UNIT
        state["free"] -= 100
        assert engine._device_room(1000 * _UNIT) == 500 * _UNIT
        assert engine._device_room(_UNIT) == _UNIT
    assert not dropped and state["cached"] == 0
    assert engine._device_room(_WORKING_BYTES) == _WORKING_BYTES


@pytest.mark.parametrize("held", [False, True])
def test_kernel_chunks_retry_and_keep_the_smaller_size_for_later_classes(monkeypatch, held):
    prior, cavity = _problem(variant_count=60, seed=51, node_count=12)
    scales = engine.log_scale(prior, _hyperparameters(prior, 52).coefficients)
    original = engine._KernelRows
    calls, references = [], []

    def form(scale, grid, precision, shift):
        calls.append(len(scale))
        # On the held route, let the first class be formed before refusing the second.
        if len(scale) > 3 and (not held or len(calls) > 1):
            raise MemoryError("device room fell")
        result = original(scale, grid, precision, shift)
        references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(engine, "_KernelRows", form)
    monkeypatch.setattr(engine, "_device_room", lambda budget: budget)
    cache = {} if held else None
    token = engine._STEP_CACHE.set(cache)
    try:
        chunks = list(engine._kernel_chunks(prior, scales, cavity, _WORKING_BYTES))
        if held:
            assert "rows" not in cache
            assert references[0]() is None
    finally:
        engine._STEP_CACHE.reset(token)
    assert any(size > 3 for size in calls)
    accepted = [len(rows) for _, rows, _ in chunks]
    assert max(accepted) <= 3
    for position, rows in enumerate(prior.class_rows):
        np.testing.assert_array_equal(np.concatenate([part for c, part, _ in chunks if c == position]), rows)
    for _, rows, actual in chunks:
        expected = original(scales[rows], prior.log_variance_grid, cavity.precision[rows], cavity.shift[rows])
        np.testing.assert_array_equal(actual.kernel, expected.kernel)


def test_kernel_chunks_propagate_a_one_row_refusal(monkeypatch):
    prior, cavity = _problem(variant_count=6, seed=51, node_count=12)
    scales = engine.log_scale(prior, _hyperparameters(prior, 52).coefficients)
    calls = []

    def refused(scale, *arguments):
        calls.append(len(scale))
        raise MemoryError("one row does not fit")

    monkeypatch.setattr(engine, "_KernelRows", refused)
    with pytest.raises(MemoryError, match="one row"):
        list(engine._kernel_chunks(prior, scales, cavity, _WORKING_BYTES))
    assert calls[-1] == 1


def _line_problem():
    prior, cavity = _problem(variant_count=60, seed=51, node_count=12)
    hyper = _hyperparameters(prior, 52, log_smoothing=1.0)
    directions = np.random.default_rng(73).normal(size=(prior.coefficient_size, 2))
    directions[prior.coefficient_size - prior.scale_size:] = 0.0
    requests = {0: np.array([-1.0, 0.0, 0.7]), 1: np.array([0.2, 0.5])}
    arguments = (prior, hyper.log_smoothing, hyper.coefficients, directions, cavity, _WORKING_BYTES)
    return arguments, requests


def test_lines_refresh_room_each_pass_and_reuse_the_log_buffer_after_a_refusal(monkeypatch):
    arguments, requests = _line_problem()
    expected = engine._lines(*arguments)(requests)
    attempts, logs, budgets = [], [], []
    limit = [3]

    class Array(np.ndarray):
        def __matmul__(self, right):
            attempts.append(self.shape[0])
            if self.shape[0] > limit[0]:
                raise MemoryError("products refused")
            return np.asarray(self) @ right

    class Device:
        def __getattr__(self, name):
            return getattr(np, name)

        def log(self, values, out=None):
            assert out is values
            logs.append(values.shape)
            return np.log(values, out=out)

    original = engine._kernel_terms

    def terms(*args):
        result = list(original(*args))
        result[3] = result[3].view(Array)
        return tuple(result)

    def room(working_bytes):
        budgets.append(working_bytes)
        return working_bytes

    monkeypatch.setattr(engine, "_kernel_terms", terms)
    monkeypatch.setattr(engine, "_device_room", room)
    with engine.device_scope(Device()):
        evaluate = engine._lines(*arguments)
        count = len(budgets)
        for _ in range(2):
            actual = evaluate(requests)
            for column in requests:
                np.testing.assert_allclose(actual[column], expected[column], rtol=1e-13, atol=1e-13)
        assert len(budgets) == count + 2
        limit[0] = 0
        with pytest.raises(MemoryError, match="products refused"):
            evaluate(requests)
    assert logs and max(attempts) > 3 and attempts[-1] == 1


def test_lines_retry_without_adding_a_failed_pieces_peak_twice(monkeypatch):
    arguments, requests = _line_problem()
    expected = engine._lines(*arguments)(requests)
    original = engine._column_log_sums
    calls = []

    def refused_after_peak(products, lost, out):
        calls.append(products.shape[0])
        if products.shape[0] > 2:
            raise MemoryError("log workspace refused")
        return original(products, lost, out)

    monkeypatch.setattr(engine, "_column_log_sums", refused_after_peak)
    actual = engine._lines(*arguments)(requests)
    assert max(calls) > 2
    for column in requests:
        np.testing.assert_allclose(actual[column], expected[column], rtol=1e-13, atol=1e-13)


def test_kernel_chunks_use_the_current_room_in_each_pass(monkeypatch):
    prior, cavity = _problem(variant_count=60, seed=51, node_count=12)
    scales = engine.log_scale(prior, _hyperparameters(prior, 52).coefficients)
    row_bytes = engine._ROW_INTERMEDIATES * prior.grid_size * np.dtype(np.float64).itemsize
    room = [4 * row_bytes]
    monkeypatch.setattr(engine, "_device_room", lambda budget: room[0])
    token = engine._STEP_CACHE.set(None)
    try:
        for width in (4, 2):
            room[0] = width * row_bytes
            chunks = list(engine._kernel_chunks(prior, scales, cavity, _WORKING_BYTES))
            assert max(len(rows) for _, rows, _ in chunks) == width
    finally:
        engine._STEP_CACHE.reset(token)
