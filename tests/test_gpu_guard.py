"""CUDA fault reports use CPU exceptions and the device plan's NumPy stand-ins."""
import inspect
import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from sv_pgs import gpu_guard as guard
from sv_pgs.device_plan import _PeerTile, moved, ShardedDualSource, plan_blocks
from sv_pgs.memory_broker import _DeviceMeter
from tests.test_memory_broker import _fake_device


class CUDARuntimeError(RuntimeError):
    pass


@pytest.fixture
def traced(monkeypatch):
    messages = []
    monkeypatch.setattr(guard, "_MODE", "1")
    monkeypatch.setattr(guard, "_REPORTED", False)
    monkeypatch.setattr(guard, "log", messages.append)
    guard._TRAIL.clear()
    yield messages
    guard._TRAIL.clear()


def test_off_returns_the_original_function_without_a_call_wrapper(monkeypatch):
    monkeypatch.setattr(guard, "_MODE", "0")
    for function in (moved, _PeerTile.rmatmat, ShardedDualSource.map_reduce, _DeviceMeter.settle):
        original = inspect.unwrap(function)
        assert guard.traced(original) is original


@pytest.mark.parametrize("mode", ["0", "1", "sync", "harden"])
def test_environment_selects_the_guard_before_device_modules_are_imported(mode):
    code = '''
from sv_pgs import gpu_guard
from sv_pgs.device_plan import moved
import os
assert hasattr(moved, '__wrapped__') == (os.environ['SVPGS_GPU_GUARD'] != '0')
'''
    subprocess.run([sys.executable, "-c", code], env={**os.environ, "SVPGS_GPU_GUARD": mode}, check=True)


def test_the_first_cuda_error_logs_only_the_last_128_operations(traced):
    @guard.traced
    def operation(value):
        return value

    @guard.traced
    def fault():
        raise CUDARuntimeError("cudaErrorLaunchFailure")

    for value in range(guard.TRAIL + 1):
        operation(np.ones(value))
    for _ in range(2):
        with pytest.raises(CUDARuntimeError, match="cudaErrorLaunchFailure"):
            fault()
    entries = [line for line in traced if line.startswith("gpu guard:   -")]
    assert len(entries) == guard.TRAIL
    assert "(2,)" in entries[0] and "fault" in entries[-1]
    assert sum("first CUDA error" in line for line in traced) == 1
    assert any("Python stack" in line for line in traced)


def test_non_cuda_errors_are_not_reported(traced):
    @guard.traced
    def fault():
        raise ValueError("host failure")

    with pytest.raises(ValueError, match="host failure"):
        fault()
    assert not traced


def test_peer_tile_fault_names_the_operation_and_device(traced):
    def fail(values):
        raise CUDARuntimeError("peer product")

    tile = _PeerTile(SimpleNamespace(rmatmat=fail), 1, 0, np)
    call = guard.traced(inspect.unwrap(_PeerTile.rmatmat))
    with pytest.raises(CUDARuntimeError, match="peer product"):
        call(tile, np.ones((4, 2)))
    assert any("_PeerTile.rmatmat" in line and "devices=(1,)" in line for line in traced)


def test_settle_reports_the_fault_and_keeps_the_original_exception(traced, monkeypatch):
    xp, _state = _fake_device(10)
    meter = _DeviceMeter(xp, 0)
    error = CUDARuntimeError("pool fault")

    def fail():
        raise error

    monkeypatch.setattr(meter.pool, "free_all_blocks", fail)
    with pytest.raises(CUDARuntimeError) as caught:
        guard.traced(inspect.unwrap(_DeviceMeter.settle))(meter)
    assert caught.value is error
    assert any("_DeviceMeter.settle" in line and "devices=(0,)" in line for line in traced)


def test_map_reduce_reports_the_workers_block_before_the_main_thread_reraises(traced):
    class Shard:
        def iter_tiles(self):
            yield 0, np.ones((2, 2))

    source = ShardedDualSource([Shard()], plan_blocks((0,), [2], 2), [(0, 2)], 2, np)

    def work(start, stop, tile, shared, rows, image):
        raise CUDARuntimeError("worker fault")

    call = guard.traced(inspect.unwrap(ShardedDualSource.map_reduce))
    with pytest.raises(CUDARuntimeError, match="worker fault"):
        call(source, work, {}, {}, (2,))
    assert any("block [0:2)" in line and "stage2-device-0" in line for line in traced)
    assert sum("first CUDA error" in line for line in traced) == 1


def test_simultaneous_cuda_errors_have_one_report(traced):
    barrier = threading.Barrier(2)

    def fault():
        barrier.wait()
        guard.report("worker", CUDARuntimeError("fault"))

    workers = [threading.Thread(target=fault) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert sum("first CUDA error" in line for line in traced) == 1


def test_ignored_destructor_error_is_reported_and_forwarded(traced, monkeypatch):
    forwarded = []
    monkeypatch.setattr(guard, "_PREVIOUS_UNRAISABLE", forwarded.append)
    arguments = SimpleNamespace(exc_value=CUDARuntimeError("event destructor"), err_msg=None, object=object())
    guard._unraisable(arguments)
    assert forwarded == [arguments]
    assert any("event destructor" in line for line in traced)


def test_moved_reports_the_copy_target(traced):
    def fail(values):
        raise CUDARuntimeError("copy fault")

    xp = SimpleNamespace(asarray=fail)
    with pytest.raises(CUDARuntimeError, match="copy fault"):
        guard.traced(inspect.unwrap(moved))(xp, np.ones((4, 2)), 1)
    assert any("moved" in line and "devices=(1,)" in line and "(4, 2)" in line for line in traced)


@pytest.mark.parametrize("mode", ["sync", "harden"])
def test_checked_settle_reports_a_fault_before_freeing_pool_blocks(traced, monkeypatch, mode):
    monkeypatch.setattr(guard, "_MODE", mode)
    xp, _state = _fake_device(10)
    meter = _DeviceMeter(xp, 0)
    freed = []

    def fail():
        raise CUDARuntimeError("pending launch failure")

    xp.cuda.Device = lambda device: SimpleNamespace(synchronize=fail)
    monkeypatch.setattr(meter.pool, "free_all_blocks", lambda: freed.append(True))
    with pytest.raises(CUDARuntimeError, match="pending launch failure"):
        guard.traced(inspect.unwrap(_DeviceMeter.settle))(meter)
    assert not freed and any("_DeviceMeter.settle" in line for line in traced)
