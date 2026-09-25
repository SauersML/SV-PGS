"""``memory_broker``: one ledger per pool, mandatory leases before caches, and the operating system's own cap.

The invariant (held bytes <= capacity at every moment) is checked on random sequences of leases, caches, releases and
evictions; the OS-level test runs a whole streamed fit (``stage2_wiring.fit_models``) and its scoring in a child whose
address space RLIMIT_AS caps at what it holds after a warm-up plus the budget the fit is given."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.memory_broker import HOST, PINNED, MemoryBroker, broker_for, current_broker, memory_scope


def _budget(host_bytes: int) -> ComputeBudget:
    return ComputeBudget(
        device_kind="cpu", device_ids=(), device_names=(), device_bytes=(), device_compute_capabilities=(), host_bytes=host_bytes, cpu_threads=1
    )


def test_the_held_bytes_never_exceed_the_capacity_over_random_leases_caches_and_releases():
    generator = np.random.default_rng(0)
    broker = MemoryBroker({HOST: 1000, PINNED: 1000, "device0": 700})
    live: list = []
    evicted: list[int] = []
    for step in range(3000):
        action = generator.integers(0, 4)
        pool = [HOST, PINNED, "device0"][int(generator.integers(0, 3))]
        size = int(generator.integers(0, 400))
        if action == 0:
            try:
                live.append(broker.reserve(pool, size, f"mandatory {step}"))
            except MemoryError:
                # Refused only where the mandatory leases alone leave no room (every idle cache was evicted first).
                charged = (PINNED, HOST) if pool == PINNED else (pool,)
                assert any(
                    sum(lease.nbytes for lease in broker.leases if charged_pool in lease.pools and not lease.cache) + size > broker.capacities[charged_pool]
                    for charged_pool in charged
                )
        elif action == 1:
            lease = broker.admit(pool, size, f"cache {step}", lambda step=step: evicted.append(step))
            if lease is not None:
                live.append(lease)
        elif live:
            live.pop(int(generator.integers(0, len(live)))).release()
        for name, capacity in broker.capacities.items():
            assert broker.held(name) <= capacity
        # A pinned lease is a host lease too.
        assert broker.held(PINNED) <= broker.held(HOST)
    assert evicted, "the random sequence should have evicted caches for mandatory leases"


def test_a_mandatory_lease_evicts_idle_caches_newest_first_and_never_one_in_use():
    broker = MemoryBroker({HOST: 100})
    dropped: list[str] = []
    old = broker.admit(HOST, 40, "old", lambda: dropped.append("old"))
    new = broker.admit(HOST, 40, "new", lambda: dropped.append("new"))
    assert broker.admit(HOST, 30, "no room", lambda: dropped.append("never")) is None
    broker.reserve(HOST, 50, "needs one cache's bytes")
    assert dropped == ["new"] and not new.live and old.live
    with old.in_use():
        with pytest.raises(MemoryError, match="old"):
            broker.reserve(HOST, 20, "needs the cache in use")
    broker.reserve(HOST, 10, "fits beside")
    assert broker.held(HOST) == 100


class _Refused(Exception):
    pass


# The fake device's unit: CuPy's allocation alignment, so every request is its own rounded size.
_UNIT = 512


def _fake_device(free: int):
    """A device of ``free`` units with CuPy's pool on it: ``used`` live units, ``cached`` free blocks it can return,
    ``fragments`` free parts of partly used blocks it cannot; lowering ``free`` is a library taking device bytes past
    the pool. Every count is in units of ``_UNIT`` bytes."""
    from types import SimpleNamespace

    state = {"free": free, "used": 0, "cached": 0, "fragments": 0, "calls": 0, "limit": 0}

    class Pool:
        def total_bytes(self):
            return (state["used"] + state["cached"] + state["fragments"]) * _UNIT

        def free_bytes(self):
            return (state["cached"] + state["fragments"]) * _UNIT

        def get_limit(self):
            return state["limit"]

        def set_limit(self, size):
            state["limit"] = size

        def free_all_blocks(self):
            state["free"] += state["cached"]
            state["cached"] = 0

        def malloc(self, size):
            units = size // _UNIT
            state["calls"] += 1
            if state["cached"] >= units:
                state["cached"] -= units
            elif state["fragments"] >= units:
                state["fragments"] -= units
            elif units > state["free"] or (state["limit"] and self.total_bytes() + size > state["limit"]):
                raise _Refused(size)
            else:
                state["free"] -= units
            state["used"] += units
            return units

    pool = Pool()
    runtime = SimpleNamespace(getDevice=lambda: 0, memGetInfo=lambda: (state["free"] * _UNIT, free * _UNIT))
    cupy = SimpleNamespace(get_default_memory_pool=lambda: pool, cuda=SimpleNamespace(runtime=runtime, memory=SimpleNamespace(OutOfMemoryError=_Refused)))
    return cupy, state


def _metered_device(free: int):
    from sv_pgs.memory_broker import _LedgerAllocator, meter_device

    cupy, state = _fake_device(free)
    broker = MemoryBroker({"device0": free * _UNIT})
    meter_device(broker, cupy, 0)
    allocator = _LedgerAllocator(broker, cupy)
    return broker, lambda units: allocator(units * _UNIT), state


def test_the_device_meter_counts_the_pools_fragments_and_refuses_what_they_leave_no_room_for():
    # bench-sim v7 chr22 001 on a 40 GB A100 [bench]: the pool's live bytes admitted 1.2 GB that the device, with 40.9 GB
    # reserved, refused after every cache was dropped. The reserved bytes the pool cannot return are held.
    broker, allocate, state = _metered_device(1000)
    allocate(400)
    # Most of the block freed, its rest still live: 300 units the pool keeps and cannot return.
    state.update(used=100, fragments=300)
    assert broker.held("device0") == 400 * _UNIT
    calls = state["calls"]
    with pytest.raises(_Refused):
        allocate(700)
    assert state["calls"] == calls, "the ledger refuses before the device is asked"
    assert allocate(600) == 600 and broker.held("device0") == 1000 * _UNIT == broker.peak("device0")


def test_a_full_ledger_still_admits_what_the_pool_serves_from_its_free_blocks():
    # A request the pool serves from a free part of a partly used block leaves its reserved bytes where they are: the
    # held bytes do not move, so the ledger admits it at capacity; one the free parts cannot serve is refused.
    broker, allocate, state = _metered_device(1000)
    allocate(1000)
    state.update(used=700, fragments=300)
    assert allocate(200) == 200 and state["free"] == 0 and broker.held("device0") == 1000 * _UNIT
    assert state["limit"] == 0, "the pool's limit is put back"
    calls = state["calls"]
    with pytest.raises(_Refused):
        allocate(200)
    assert state["calls"] == calls


def test_an_allocation_the_device_refuses_after_the_ledger_admitted_it_measures_the_outside_bytes_and_evicts_for_them():
    # The ledger's bound reads the device's bytes outside the pool (a library's workspace) as last measured; when the
    # device refuses what the bound admitted, the meter measures them again and the ledger makes room against the
    # device's own count: the pool's free blocks returned, then idle caches evicted newest first.
    broker, allocate, state = _metered_device(1000)
    dropped: list[str] = []
    allocate(300)

    def evict() -> None:
        dropped.append("idle")
        state["used"] -= 300
        state["cached"] += 300

    busy = broker.admit("device0", 0, "busy", lambda: dropped.append("busy"), allocated=True)
    broker.admit("device0", 300 * _UNIT, "idle", evict, allocated=True)
    state["free"] -= 200  # a library's workspace
    with busy.in_use():
        assert allocate(600) == 600
    assert dropped == ["idle"] and broker.held("device0") == 800 * _UNIT and state["free"] == 200
    calls = state["calls"]
    with busy.in_use(), pytest.raises(_Refused):
        allocate(300)
    assert state["calls"] == calls


def test_a_decision_the_held_bytes_refuse_first_returns_the_pools_free_blocks():
    broker, allocate, state = _metered_device(1000)
    dropped: list[str] = []
    allocate(700)
    state.update(used=0, cached=700)
    assert broker.held("device0") == 700 * _UNIT and broker.remaining("device0") == 1000 * _UNIT
    assert state["cached"] == 0
    allocate(700)
    assert broker.admit("device0", 400 * _UNIT, "a cache", lambda: dropped.append("cache")) is None
    state.update(used=0, cached=700)
    assert broker.admit("device0", 400 * _UNIT, "a cache", lambda: dropped.append("cache")) is not None and not dropped


def test_a_metered_pool_counts_its_live_bytes_and_admits_caches_from_what_they_leave():
    live = {"bytes": 0}
    broker = MemoryBroker({"device0": 100}, meters={"device0": lambda: live["bytes"]})
    dropped: list[int] = []

    def evict() -> None:
        dropped.append(1)
        live["bytes"] -= 30

    live["bytes"] = 50
    assert broker.admit("device0", 60, "too big before allocation", evict) is None
    cache = broker.admit("device0", 30, "a cache about to be allocated", evict)
    live["bytes"] += 30
    assert cache is not None and broker.held("device0") == 80
    # An allocation of 40 needs the cache's 30: it is evicted, and the meter falls with it.
    broker.make_room("device0", 40, "an allocation")
    assert dropped == [1] and broker.held("device0") == 50
    with pytest.raises(MemoryError):
        broker.make_room("device0", 60, "past what anything frees")


def test_a_scope_shares_one_ledger_and_outside_it_each_caller_has_its_own():
    assert current_broker() is None
    with memory_scope(_budget(1000)) as broker:
        assert current_broker() is broker and broker_for(_budget(5)) is broker
        with memory_scope(_budget(7)) as inner:
            assert inner is broker
    assert current_broker() is None
    assert broker_for(_budget(5)).capacities[HOST] == 5


def test_the_dosage_ring_and_the_scorer_charge_one_ledger(tmp_path: Path):
    """The store reader's ring and the scorer's buffers are leases on the scope's ledger: a scope too small for both
    refuses before allocating, and one that holds them leaves the ledger empty afterwards."""
    from sv_pgs.artifact import StoreCodeBlocks
    from sv_pgs.dosage_store import DosageStore
    from sv_pgs.fast_scoring import ScoringModel, ScoringPlan, score_genetic
    from sv_pgs.config import TraitType
    from tests.test_dosage_store import _two_half_dosage, _write_store

    _write_store(tmp_path / "store", _two_half_dosage(), "zstd")
    with DosageStore.open(tmp_path / "store") as store:
        rows = np.arange(0, store.n_variants, 3, dtype=np.int64)
        model = ScoringModel(
            store_rows=rows, signed_means=np.zeros(rows.shape[0]), signed_scales=np.ones(rows.shape[0]), coefficients=np.full(rows.shape[0], 0.01),
            posterior_draws=np.zeros((rows.shape[0], 2)), alpha=np.zeros(1), trait_type=TraitType.QUANTITATIVE, predictive_intercept_shift=0.0,
            covariate_draws=np.zeros((1, 2)), covariate_covariance=np.zeros((1, 1)), gaussian_posterior=True,
        )
        plan = ScoringPlan.from_models([model])
        budget = _budget(1 << 24)
        with memory_scope(budget) as broker:
            reference = score_genetic(StoreCodeBlocks(store, budget), plan, budget)
            assert broker.held(HOST) == 0 and broker.peak(HOST) > 0
            with pytest.raises(MemoryError):
                broker.reserve(HOST, (1 << 24) - store.n_samples, "most of the budget")
                score_genetic(StoreCodeBlocks(store, budget), plan, budget)
        signed = store.read_codes(0, store.n_variants)[rows].astype(np.float64) - 127.0
        np.testing.assert_allclose(reference.means[:, 0], signed.T @ model.coefficients, rtol=1e-12, atol=1e-12)


_CHILD = r"""
import resource, sys
from pathlib import Path
import numpy as np
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import TraitType
from sv_pgs.fast_scoring import ScoringPlan, score_genetic
from sv_pgs.artifact import StoreCodeBlocks
from sv_pgs.stage2_wiring import fit_models
from tests.test_full_data_fit import _SAMPLES, _store

root, budget_bytes = Path(sys.argv[1]), int(sys.argv[2])
store, covariate, targets, _genetic = _store(root / "store", 7)
budget = ComputeBudget("cpu", (), (), (), (), budget_bytes, 1)
warm = ComputeBudget("cpu", (), (), (), (), 1 << 30, 1)


def run(work, budget):
    work.mkdir()
    fitted = fit_models(
        store=store, store_columns=np.arange(_SAMPLES), covariates=np.column_stack([np.ones(_SAMPLES), covariate]),
        covariate_columns=np.ones((1, 2), dtype=bool), targets=targets[:, None], training=np.ones((_SAMPLES, 1), dtype=bool),
        trait_types=[TraitType.QUANTITATIVE], log_variance_offset=None, budget=budget, work_dir=work, seed=3, draw_count=8,
    )
    return score_genetic(StoreCodeBlocks(store, budget), ScoringPlan.from_models(fitted.scoring), budget)


run(root / "warm", warm)  # every kernel compiled and every library loaded before the cap
with open("/proc/self/status") as status:
    held = next(int(line.split()[1]) * 1024 for line in status if line.startswith("VmSize:"))
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (held + budget_bytes, hard))
try:
    scores = run(root / "capped", budget)
except MemoryError:
    print("memory-error")
    raise SystemExit(0)
assert np.all(np.isfinite(scores.means))
print("fitted-and-scored", held)
"""


def _capped_fit(tmp_path: Path, budget_bytes: int) -> str:
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1", MALLOC_ARENA_MAX="1")
    environment["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1]), environment.get("PYTHONPATH", "")])
    completed = subprocess.run([sys.executable, "-c", _CHILD, str(tmp_path), str(budget_bytes)], env=environment, capture_output=True, text=True)
    assert completed.returncode == 0, f"the capped fit ended with status {completed.returncode}:\n{completed.stderr[-4000:]}"
    return completed.stdout


def test_a_streamed_fit_and_its_scoring_stay_under_an_address_space_cap_of_their_budget(tmp_path: Path):
    """``ulimit -v`` in effect: the child's address space may grow by the fit's budget beyond what it holds after an
    identical warm-up fit, and the kernel refuses any mapping past that. The capped fit completes and scores."""
    assert "fitted-and-scored" in _capped_fit(tmp_path, 1 << 30)


def test_a_budget_the_fit_cannot_run_in_ends_in_memory_error_not_a_kill(tmp_path: Path):
    """Under a cap of a few megabytes the fit refuses (the planners' or the allocator's MemoryError), never dies."""
    assert "memory-error" in _capped_fit(tmp_path, 1 << 22)


def test_a_bounded_pools_allocations_under_its_bound_never_read_its_meter():
    """A metered pool with an O(1) bound (a device's reserved bytes and its outside bytes as last measured): allocations
    that fit under the bound record the bound's peak with the allocation and never read the meter (the pool's live bytes,
    the meter read on every allocation before, were 74% of a genome fit's sweeps); one that does not fit under the bound
    reads it."""
    reads = {"meter": 0}
    state = {"used": 10, "reserved": 60}

    def meter() -> int:
        reads["meter"] += 1
        return state["used"]

    broker = MemoryBroker({"device0": 100}, meters={"device0": meter}, bounds={"device0": lambda: state["reserved"]})
    for _ in range(50):
        broker.make_room("device0", 30, "an allocation under the bound")
    assert reads["meter"] == 0 and broker.peak("device0") == 60 + 30 >= state["used"]
    broker.make_room("device0", 60, "past the bound: the meter decides")
    assert reads["meter"] > 0


_REFAULT = """
import resource
import numpy as np
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.memory_broker import memory_scope

# A block above glibc's largest mmap threshold (32 MB on 64-bit), as the fit's response blocks are.
count = 2 * 32 * 2**20 // 8

def refaults() -> int:
    np.empty(count).fill(1.0)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
    np.empty(count).fill(1.0)
    return resource.getrusage(resource.RUSAGE_SELF).ru_minflt - before

outside = refaults()
with memory_scope(ComputeBudget("cpu", (), (), (), (), 2**34, 1)):
    inside = refaults()
print(outside, inside)
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the host allocator setting is Linux/glibc only")
def test_a_fit_scope_keeps_a_freed_block_above_the_mmap_cap_for_its_next_allocation():
    """Outside a scope glibc unmaps a freed block above its mmap cap and the next one faults its pages in again;
    inside a fit's scope the heap keeps the block, so the next allocation of its size faults nothing."""
    result = subprocess.run([sys.executable, "-c", _REFAULT], capture_output=True, text=True, check=True, cwd=Path(__file__).resolve().parents[1])
    outside, inside = map(int, result.stdout.split())
    assert outside > 0 and inside == 0


def test_the_host_allocator_setting_does_nothing_off_glibc(monkeypatch):
    from sv_pgs import memory_broker

    monkeypatch.setattr(memory_broker.sys, "platform", "darwin")
    with memory_broker._retain_freed_host_memory() as set_:
        assert set_ is False
