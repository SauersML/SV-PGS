"""Tests for the Phase-3 N-GPU LD-block scheduler.

Covers ``sv_pgs.gpu_scheduler.GPUScheduler``:

* Detection for 0/1/2/4/8 visible devices via a mocked cupy.
* CPU fallback when cupy is missing.
* Round-robin block dispatch table.
* ``device_context`` enter/exit semantics on cupy.cuda.Device + Stream.
* ``synchronize`` calls ``.synchronize()`` on every per-device stream.
* ``aggregate_to_host`` sums per-device dicts.
* Single-device behavior: one stream, every block on device 0.

No real GPU required. Mocks follow the SimpleNamespace pattern from
``tests/test_v100_budget_pinning.py`` and ``tests/test_multi_gpu_scheduler_pinning.py``.
"""
from __future__ import annotations

import builtins
import types
from typing import Any

import numpy as np
import pytest

from sv_pgs import gpu_scheduler as gs


# ---------------------------------------------------------------------------
# fake cupy helpers
# ---------------------------------------------------------------------------


class _FakeStream:
    instances: list["_FakeStream"] = []

    def __init__(self, non_blocking: bool = False) -> None:
        self.non_blocking = non_blocking
        self.entered = 0
        self.exited = 0
        self.synchronized = 0
        _FakeStream.instances.append(self)

    def __enter__(self) -> "_FakeStream":
        self.entered += 1
        return self

    def __exit__(self, *_: Any) -> None:
        self.exited += 1

    def synchronize(self) -> None:
        self.synchronized += 1


class _FakeDevice:
    enter_log: list[int] = []
    exit_log: list[int] = []

    def __init__(self, device_id: int = 0) -> None:
        self.device_id = int(device_id)

    def __enter__(self) -> "_FakeDevice":
        _FakeDevice.enter_log.append(self.device_id)
        return self

    def __exit__(self, *_: Any) -> None:
        _FakeDevice.exit_log.append(self.device_id)


def _make_fake_cupy(device_count: int) -> Any:
    state = {"count": int(device_count), "device_syncs": 0}

    def _get_device_count() -> int:
        return state["count"]

    def _device_synchronize() -> None:
        state["device_syncs"] += 1

    runtime = types.SimpleNamespace(
        getDeviceCount=_get_device_count,
        deviceSynchronize=_device_synchronize,
    )
    cuda = types.SimpleNamespace(
        runtime=runtime,
        Device=_FakeDevice,
        Stream=_FakeStream,
    )
    fake = types.SimpleNamespace(
        cuda=cuda,
        ndarray=np.ndarray,  # so isinstance checks in aggregate_to_host stay safe
        asnumpy=np.asarray,
        _state=state,
    )
    return fake


@pytest.fixture(autouse=True)
def _reset_fake_state() -> None:
    _FakeStream.instances = []
    _FakeDevice.enter_log = []
    _FakeDevice.exit_log = []
    yield
    _FakeStream.instances = []
    _FakeDevice.enter_log = []
    _FakeDevice.exit_log = []


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 2, 4, 8])
def test_detect_with_mock_cupy_reports_correct_device_count(
    monkeypatch: pytest.MonkeyPatch, n: int
) -> None:
    fake = _make_fake_cupy(device_count=n)
    real_import = builtins.__import__

    def _import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "cupy":
            return fake
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)

    sched = gs.GPUScheduler.detect()
    if n == 0:
        # 0 visible devices -> CPU fallback.
        assert sched.device_count == 1
        assert sched.device_ids == (-1,)
        assert sched.is_cpu_fallback
    else:
        assert sched.device_count == n
        assert sched.device_ids == tuple(range(n))
        assert not sched.is_cpu_fallback


def test_detect_cpu_fallback_when_cupy_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def _import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "cupy":
            raise ImportError("cupy not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    sched = gs.GPUScheduler.detect()
    assert sched.device_count == 1
    assert sched.device_ids == (-1,)
    assert sched.is_cpu_fallback


# ---------------------------------------------------------------------------
# round-robin
# ---------------------------------------------------------------------------


def test_round_robin_10_blocks_across_4_devices() -> None:
    sched = gs.GPUScheduler(device_ids=(0, 1, 2, 3), cupy=_make_fake_cupy(4))
    assignments = list(sched.assign(range(10)))

    by_device: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
    for a in assignments:
        by_device[a.device_id].append(a.block_id)
        assert a.stream_id == 0

    assert by_device[0] == [0, 4, 8]
    assert by_device[1] == [1, 5, 9]
    assert by_device[2] == [2, 6]
    assert by_device[3] == [3, 7]


def test_round_robin_single_device_pins_all_blocks_to_zero() -> None:
    sched = gs.GPUScheduler(device_ids=(0,), cupy=_make_fake_cupy(1))
    assignments = list(sched.assign(range(7)))
    assert all(a.device_id == 0 for a in assignments)
    assert [a.block_id for a in assignments] == list(range(7))


def test_round_robin_cpu_fallback_emits_minus_one_device() -> None:
    sched = gs.GPUScheduler(device_ids=(-1,), cupy=None)
    assignments = list(sched.assign(range(5)))
    assert all(a.device_id == -1 for a in assignments)
    assert [a.block_id for a in assignments] == list(range(5))


# ---------------------------------------------------------------------------
# device_context
# ---------------------------------------------------------------------------


def test_device_context_enters_and_exits_cupy_device_and_stream() -> None:
    fake = _make_fake_cupy(2)
    sched = gs.GPUScheduler(device_ids=(0, 1), cupy=fake)

    with sched.device_context(1):
        assert _FakeDevice.enter_log == [1]
        # Stream is lazily created on first context use.
        assert len(_FakeStream.instances) == 1
        assert _FakeStream.instances[0].entered == 1
        assert _FakeStream.instances[0].exited == 0
    assert _FakeDevice.exit_log == [1]
    assert _FakeStream.instances[0].exited == 1


def test_device_context_reuses_one_stream_per_device() -> None:
    fake = _make_fake_cupy(2)
    sched = gs.GPUScheduler(device_ids=(0, 1), cupy=fake)

    with sched.device_context(0):
        pass
    with sched.device_context(0):
        pass
    with sched.device_context(1):
        pass
    # 2 distinct streams total (one per device used), not 3.
    assert len(_FakeStream.instances) == 2
    assert _FakeStream.instances[0].entered == 2
    assert _FakeStream.instances[1].entered == 1


def test_device_context_cpu_fallback_is_noop() -> None:
    sched = gs.GPUScheduler(device_ids=(-1,), cupy=None)
    with sched.device_context(-1):
        pass
    assert _FakeDevice.enter_log == []
    assert _FakeStream.instances == []


def test_device_context_rejects_unknown_device_id() -> None:
    fake = _make_fake_cupy(2)
    sched = gs.GPUScheduler(device_ids=(0, 1), cupy=fake)
    with pytest.raises(ValueError):
        with sched.device_context(7):
            pass


# ---------------------------------------------------------------------------
# synchronize
# ---------------------------------------------------------------------------


def test_synchronize_waits_on_every_per_device_stream() -> None:
    fake = _make_fake_cupy(3)
    sched = gs.GPUScheduler(device_ids=(0, 1, 2), cupy=fake)
    # Touch each device to materialize a stream.
    for d in (0, 1, 2):
        with sched.device_context(d):
            pass

    sched.synchronize()
    assert len(_FakeStream.instances) == 3
    for s in _FakeStream.instances:
        assert s.synchronized == 1
    # deviceSynchronize called once per device.
    assert fake._state["device_syncs"] == 3


def test_synchronize_skips_devices_with_no_stream_touched() -> None:
    fake = _make_fake_cupy(3)
    sched = gs.GPUScheduler(device_ids=(0, 1, 2), cupy=fake)
    with sched.device_context(1):
        pass
    sched.synchronize()
    # Only device 1 had a stream; the other two still issue a deviceSynchronize
    # (paranoia) but no stream syncs happen for them.
    assert len(_FakeStream.instances) == 1
    assert _FakeStream.instances[0].synchronized == 1
    assert fake._state["device_syncs"] == 3


def test_synchronize_cpu_fallback_is_noop() -> None:
    sched = gs.GPUScheduler(device_ids=(-1,), cupy=None)
    sched.synchronize()  # must not raise


# ---------------------------------------------------------------------------
# aggregate_to_host
# ---------------------------------------------------------------------------


def test_aggregate_to_host_sums_numpy_per_device_dict() -> None:
    sched = gs.GPUScheduler(device_ids=(0, 1, 2), cupy=_make_fake_cupy(3))
    per_device = {
        0: np.array([1.0, 2.0, 3.0]),
        1: np.array([10.0, 20.0, 30.0]),
        2: np.array([100.0, 200.0, 300.0]),
    }
    result = sched.aggregate_to_host(per_device)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([111.0, 222.0, 333.0]))


def test_aggregate_to_host_cpu_fallback_returns_single_array() -> None:
    sched = gs.GPUScheduler(device_ids=(-1,), cupy=None)
    arr = np.array([1.0, 2.0, 3.0])
    out = sched.aggregate_to_host({-1: arr})
    np.testing.assert_array_equal(out, arr)
    # must be a copy, not the same buffer
    assert out is not arr


def test_aggregate_to_host_empty_dict_raises() -> None:
    sched = gs.GPUScheduler(device_ids=(0,), cupy=_make_fake_cupy(1))
    with pytest.raises(ValueError):
        sched.aggregate_to_host({})


# ---------------------------------------------------------------------------
# construction guards
# ---------------------------------------------------------------------------


def test_construction_rejects_empty_device_tuple() -> None:
    with pytest.raises(ValueError):
        gs.GPUScheduler(device_ids=(), cupy=None)
