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
    """A metered pool with an O(1) bound (a device's reserved bytes): allocations that fit under the bound record the
    bound's peak and never read the meter (the meter walks CuPy's free lists: read on every allocation, it was 74% of a
    genome fit's sweeps); one that does not fit under the bound reads it."""
    reads = {"meter": 0}
    state = {"used": 10, "reserved": 60}

    def meter() -> int:
        reads["meter"] += 1
        return state["used"]

    broker = MemoryBroker({"device0": 100}, meters={"device0": meter}, bounds={"device0": lambda: state["reserved"]})
    for _ in range(50):
        broker.make_room("device0", 30, "an allocation under the bound")
    assert reads["meter"] == 0 and broker.peak("device0") == 60 >= state["used"]
    broker.make_room("device0", 60, "past the bound: the meter decides")
    assert reads["meter"] > 0
