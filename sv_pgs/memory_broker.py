"""The one ledger of a fit's and a scorer's memory: the host, page-locked host memory and each device, every
substantial allocation charged before it is made, and every cache admitted only from what the rest leaves.

``compute_budget`` measures what the machine allows (``ComputeBudget``: host bytes after MemAvailable, the memory
cgroups, the runner's allotment and the address-space limit; each device's free bytes once its libraries exist). Every
consumer used to plan its own buffers against that whole number, so two consumers that were each within the budget
could together exceed it, and a cache sized from a free-memory snapshot (the resident genotype codes, the panel Grams)
could take what a later mandatory buffer needed. The broker keeps one account per pool instead.

Host pools (``host``, ``pinned``) are ledgers of leases:

- ``reserve(pool, bytes, purpose)``: a mandatory lease (a buffer the computation cannot run without), granted when the
  pool's held bytes plus the request fit its capacity, after evicting as many idle caches as that takes (newest
  first); otherwise ``MemoryError`` naming the pool, the request and every live lease, before anything is allocated.
- ``admit(pool, bytes, purpose, evict)``: a cache lease (it only saves recomputation or rereading), granted only from
  what the pool has left, never by evicting anything; None where it does not fit. ``evict`` drops the cache when a
  later mandatory lease needs its bytes, except while it is in use (``Lease.in_use``), when that lease is refused.
- ``release``: the bytes return to the pool (a lease is also a context manager).
A page-locked lease is charged to both the ``pinned`` and the ``host`` pool, since pinned pages are resident host
pages; the ``pinned`` pool's capacity is the host's (CUDA pins its pages in the driver, not by ``mlock``, so
RLIMIT_MEMLOCK does not bound them), and it keeps the page-locked bytes' own account.

Device pools are metered: their held bytes are the bytes CuPy's memory pool has live on the device
(``used_bytes``), read at every decision, and inside a CUDA scope every CuPy allocation goes through the ledger's
allocator (``_LedgerAllocator``), which makes room for it (evicting idle device caches, newest first) or raises
``MemoryError`` before the allocation is made. So on a device "every allocation is charged before it happens" holds
for every array CuPy allocates, by construction rather than by each consumer's accounting. A device cache (the
resident genotype codes, the panel Grams) is a cache lease that records what to drop; its arrays are already live
bytes of the meter, so it adds nothing to the held bytes. ``reserve`` on a device pool makes room for bytes about to
be allocated (the allocations themselves are then metered) and holds nothing.

Invariant. For every pool P with capacity C_P, at every moment, H_P <= C_P, where H_P is the sum of P's live leases
(host pools) or the device's live CuPy bytes (device pools).

Proof. Host pools: H_P changes only in three ways. (1) A release or an eviction removes a live lease: H_P falls.
(2) ``reserve`` grants b bytes only after checking H_P' + b <= C_P, H_P' the holding after the evictions it made (each
an instance of 1), with nothing else between the check and the grant (the ledger's lock). (3) ``admit`` grants b only
if H_P + b <= C_P. A pinned lease applies (2) or (3) to both pools, checking both before charging either. Device
pools: H_P rises only by an allocation, which the allocator makes only after checking H_P + b <= C_P (after evictions,
each of which only frees); it falls when an array is freed. Every transition from a state that satisfies the
invariant ends in one that does, and the empty ledger does: it holds at every moment. Caches enter a host pool only by
(3), so from what the mandatory leases then leave; a mandatory request evicts every idle cache before it is refused,
so caches never make a request fail that the mandatory leases alone would admit (unless a cache is in use at that
moment, which the refusal names). On a device a cache is never in use in this sense: its holder reads it a block at a
time and switches to recomputing once it is dropped (``store_block_source``, ``device_sweep``), so an eviction frees
all of it but what the running step still references, which the meter counts as live.

What the invariant covers. Host: the bytes the leases stand for, which is the process's memory exactly where every
substantial allocation is charged before it is made and released no earlier than it is freed; each consumer's
docstring states which arrays its leases cover. Allocations too small to charge (Python objects, per-block
temporaries of O(block) bytes) are outside it. Device: every CuPy allocation inside a scope; memory CuPy does not
allocate (the CUDA context and the libraries' own handles, measured before the budget was) is outside it, as is the
pool's fragmentation (a request the meter admits can still find no contiguous block).

Scope. ``memory_scope(budget)`` makes one broker current for the code under it (a ``ContextVar``; threads started
inside inherit it only through ``contextvars.copy_context``) and, on CUDA, installs the ledger's allocator for its
duration; ``broker_for(budget)`` returns the current broker, or, outside every scope, a fresh broker of that budget
whose ledger is then the caller's alone; ``current_broker()`` returns the current one or None.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from sv_pgs.compute_budget import ComputeBudget, _try_import_cupy

HOST = "host"
PINNED = "pinned"
_CUPY_ALLOCATION_ALIGNMENT = 512
"""CuPy's memory pool rounds every allocation up to a multiple of 512 bytes (``cupy.cuda.memory._round_size``), which
is what its ``used_bytes`` counts."""


def device_pool(device_id: int) -> str:
    """The pool name of CUDA device ``device_id``."""
    return f"device{int(device_id)}"


@dataclass(eq=False)
class Lease:
    """Bytes of one pool held for one purpose (``MemoryBroker``). ``cache`` leases carry the ``evict`` callback that
    drops what they hold; ``pools`` are the pools charged (a pinned lease is also a host lease)."""

    broker: "MemoryBroker"
    pools: tuple[str, ...]
    nbytes: int
    purpose: str
    evict: Callable[[], None] | None = None
    live: bool = True
    users: int = 0

    @property
    def cache(self) -> bool:
        return self.evict is not None

    def release(self) -> None:
        self.broker._release(self)

    def resize(self, nbytes: int) -> None:
        """Grow (as a mandatory request, evicting caches if it must) or shrink the lease in place."""
        self.broker._resize(self, int(nbytes))

    @contextmanager
    def in_use(self) -> Iterator["Lease"]:
        """While inside, a cache lease is not evicted: its holder is reading what it holds."""
        with self.broker._lock:
            self.users += 1
        try:
            yield self
        finally:
            with self.broker._lock:
                self.users -= 1

    def __enter__(self) -> "Lease":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


@dataclass
class MemoryBroker:
    """Capacities per pool, the meters of the metered (device) pools, and the live leases (the module docstring states
    the invariant)."""

    capacities: dict[str, int]
    meters: dict[str, Callable[[], int]] = field(default_factory=dict)
    # A metered pool's O(1) upper bound on its meter (CuPy's reserved bytes >= its used bytes): where the bound
    # already decides a question (a request fits, or no new peak), the meter, which walks the pool's free lists, is
    # not read. Reading it on every device allocation was 92% of a genome fit's stage 2 (bench-sim 013 [sim], py-spy).
    bounds: dict[str, Callable[[], int]] = field(default_factory=dict)
    leases: list[Lease] = field(default_factory=list)
    peaks: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for pool, capacity in self.capacities.items():
            if int(capacity) < 0:
                raise ValueError(f"pool {pool} has a negative capacity {capacity}")
        self.capacities = {pool: int(capacity) for pool, capacity in self.capacities.items()}
        self.peaks = {pool: 0 for pool in self.capacities}
        self._lock = threading.RLock()

    @classmethod
    def from_budget(cls, budget: ComputeBudget) -> "MemoryBroker":
        """The host's bytes, the pinned pool (within the host's) and each device's."""
        capacities = {HOST: int(budget.host_bytes), PINNED: int(budget.host_bytes)}
        for device_id, device_bytes in zip(budget.device_ids, budget.device_bytes):
            capacities[device_pool(device_id)] = int(device_bytes)
        return cls(capacities)

    # the ledger

    def held(self, pool: str) -> int:
        with self._lock:
            meter = self.meters.get(pool)
            if meter is not None:
                return int(meter())
            return sum(lease.nbytes for lease in self.leases if pool in lease.pools)

    def remaining(self, pool: str) -> int:
        """What a new lease of ``pool`` could take without evicting anything."""
        with self._lock:
            return min(self.capacities[charged] - self.held(charged) for charged in self._charged(pool))

    def reclaimable(self, pool: str) -> int:
        """What a mandatory lease of ``pool`` could take: the remainder plus every idle cache charged to it."""
        with self._lock:
            return min(
                self.capacities[charged] - self.held(charged)
                + sum(lease.nbytes for lease in self.leases if lease.cache and not lease.users and charged in lease.pools)
                for charged in self._charged(pool)
            )

    def peak(self, pool: str) -> int:
        """The pool's largest held bytes seen at a lease or allocation; for a pool with a bound (a device's), the
        bound's largest, an upper bound on that."""
        with self._lock:
            return self.peaks[pool]

    def describe(self) -> str:
        with self._lock:
            pools = ", ".join(f"{pool} {self.held(pool) / 1e9:.3f}/{capacity / 1e9:.3f} GB" for pool, capacity in self.capacities.items())
            leases = "; ".join(
                f"{lease.purpose} {lease.nbytes / 1e9:.3f} GB on {'+'.join(lease.pools)}{' (cache)' if lease.cache else ''}" for lease in self.leases
            )
            return f"{pools} | leases: {leases or 'none'}"

    def _charged(self, pool: str) -> tuple[str, ...]:
        if pool not in self.capacities:
            raise KeyError(f"no memory pool {pool!r}: this budget has {sorted(self.capacities)}")
        return (PINNED, HOST) if pool == PINNED else (pool,)

    def _counted(self, lease: Lease, pool: str) -> bool:
        """Whether ``lease``'s bytes are part of ``pool``'s held bytes (a metered pool counts its live arrays instead)."""
        return pool in lease.pools and pool not in self.meters

    def _fits(self, charged: tuple[str, ...], nbytes: int, ignoring: Lease | None = None) -> bool:
        def fits(pool: str) -> bool:
            bound = self.bounds.get(pool)
            if bound is not None and int(bound()) + nbytes <= self.capacities[pool]:
                return True  # held <= bound: the request fits whatever the pool's used bytes are
            return self.held(pool) - (ignoring.nbytes if ignoring is not None and self._counted(ignoring, pool) else 0) + nbytes <= self.capacities[pool]

        return all(fits(pool) for pool in charged)

    def _note_peaks(self, charged: tuple[str, ...]) -> None:
        for pool in charged:
            bound = self.bounds.get(pool)
            if bound is not None:
                # A pool with an O(1) bound records the bound's peak, an upper bound on its held bytes' peak. Reading
                # the meter whenever the bound was above the recorded peak (the pool's reserved bytes usually are
                # above its used bytes' peak, so nearly always) was 74% of a genome fit's mean-field sweeps
                # (bench-sim chr22 004 [sim], py-spy: used <- held <- _note_peaks on every device allocation).
                self.peaks[pool] = max(self.peaks[pool], int(bound()))
                continue
            self.peaks[pool] = max(self.peaks[pool], self.held(pool))

    def _make_room(self, charged: tuple[str, ...], nbytes: int, purpose: str, ignoring: Lease | None = None) -> None:
        """Evict idle caches charged to ``charged``, newest first, until ``nbytes`` more fit; MemoryError if they cannot."""
        if self._fits(charged, nbytes, ignoring):
            return
        for lease in reversed([lease for lease in self.leases if lease.cache and not lease.users and set(lease.pools) & set(charged)]):
            self._evict(lease)
            if self._fits(charged, nbytes, ignoring):
                return
        raise MemoryError(
            f"{purpose} needs {nbytes / 1e9:.3f} GB of {'+'.join(charged)} memory that the budget does not hold: {self.describe()}"
        )

    def _evict(self, lease: Lease) -> None:
        self.leases.remove(lease)
        lease.live = False
        assert lease.evict is not None
        lease.evict()

    # the leases

    def reserve(self, pool: str, nbytes: int, purpose: str) -> Lease:
        """A mandatory lease (the module docstring): granted, evicting idle caches if it must, or MemoryError. On a
        metered pool it makes room for bytes about to be allocated and holds nothing itself."""
        size = int(nbytes)
        if size < 0:
            raise ValueError(f"{purpose}: a lease of {size} bytes")
        with self._lock:
            charged = self._charged(pool)
            self._make_room(charged, size, purpose)
            metered = all(charged_pool in self.meters for charged_pool in charged)
            lease = Lease(self, charged, 0 if metered else size, purpose, live=not metered)
            if not metered:
                self.leases.append(lease)
            self._note_peaks(charged)
            return lease

    def make_room(self, pool: str, nbytes: int, purpose: str) -> None:
        """Room for ``nbytes`` more on ``pool`` (a metered pool's allocation), evicting idle caches if it must."""
        with self._lock:
            self._make_room(self._charged(pool), int(nbytes), purpose)
            self._note_peaks(self._charged(pool))

    def admit(self, pool: str, nbytes: int, purpose: str, evict: Callable[[], None], allocated: bool = False) -> Lease | None:
        """A cache lease from what the pool has left (no eviction), or None where it does not fit. ``allocated``: the
        cache's arrays already exist (a metered pool then already counts them, and admits them while it is within its
        capacity)."""
        size = int(nbytes)
        with self._lock:
            charged = self._charged(pool)
            needed = 0 if allocated and all(charged_pool in self.meters for charged_pool in charged) else size
            if size < 0 or not self._fits(charged, needed):
                return None
            lease = Lease(self, charged, size, purpose, evict=evict)
            self.leases.append(lease)
            self._note_peaks(charged)
            return lease

    def _release(self, lease: Lease) -> None:
        with self._lock:
            if lease.live:
                self.leases.remove(lease)
                lease.live = False

    def _resize(self, lease: Lease, nbytes: int) -> None:
        with self._lock:
            if not lease.live:
                raise ValueError(f"{lease.purpose}: a released lease cannot be resized")
            if nbytes > lease.nbytes:
                if lease.cache:
                    if not self._fits(lease.pools, nbytes, ignoring=lease):
                        raise MemoryError(f"{lease.purpose}: a cache lease grows only into free bytes: {self.describe()}")
                else:
                    self._make_room(lease.pools, nbytes, lease.purpose, ignoring=lease)
            lease.nbytes = int(nbytes)
            self._note_peaks(lease.pools)


class _LedgerAllocator:
    """CuPy's allocator inside a CUDA scope: every device allocation makes room on the ledger before the default pool
    allocates it (the module docstring's device pools)."""

    def __init__(self, broker: MemoryBroker, cupy: Any) -> None:
        self.broker = broker
        self.cupy = cupy
        self.pool = cupy.get_default_memory_pool()

    def __call__(self, size: int) -> Any:
        rounded = -(-int(size) // _CUPY_ALLOCATION_ALIGNMENT) * _CUPY_ALLOCATION_ALIGNMENT
        name = device_pool(int(self.cupy.cuda.runtime.getDevice()))
        if name in self.broker.capacities:
            try:
                self.broker.make_room(name, rounded, "a device allocation")
            except MemoryError as error:
                raise self.cupy.cuda.memory.OutOfMemoryError(rounded, self.broker.capacities[name] - self.broker.held(name), 0) from error
        return self.pool.malloc(size)


def _device_meter(cupy: Any, device_id: int, reserved: bool = False) -> Callable[[], int]:
    """The device pool's used bytes, or with ``reserved`` its reserved bytes (every block the pool holds, used or
    free: an upper bound on the used bytes, read in O(1))."""
    pool = cupy.get_default_memory_pool()
    read = pool.total_bytes if reserved else pool.used_bytes

    def used() -> int:
        if int(cupy.cuda.runtime.getDevice()) == device_id:
            return int(read())
        with cupy.cuda.Device(device_id):
            return int(read())

    return used


_CURRENT: ContextVar[MemoryBroker | None] = ContextVar("svpgs_memory_broker", default=None)


@contextmanager
def memory_scope(budget: ComputeBudget) -> Iterator[MemoryBroker]:
    """Make one broker of ``budget`` current for the code inside (an enclosing scope's broker, where there is one) and,
    on CUDA, meter its devices and route CuPy's allocations through its allocator."""
    current = _CURRENT.get()
    if current is not None:
        yield current
        return
    broker = MemoryBroker.from_budget(budget)
    cupy = _try_import_cupy() if budget.device_kind == "cuda" else None
    previous = None
    if cupy is not None:
        for device_id in budget.device_ids:
            broker.meters[device_pool(device_id)] = _device_meter(cupy, device_id)
            broker.bounds[device_pool(device_id)] = _device_meter(cupy, device_id, reserved=True)
        previous = cupy.cuda.get_allocator() if hasattr(cupy.cuda, "get_allocator") else cupy.get_default_memory_pool().malloc
        cupy.cuda.set_allocator(_LedgerAllocator(broker, cupy))
    token = _CURRENT.set(broker)
    try:
        yield broker
    finally:
        _CURRENT.reset(token)
        if cupy is not None:
            cupy.cuda.set_allocator(previous)


def broker_for(budget: ComputeBudget) -> MemoryBroker:
    """The current scope's broker, or a fresh broker of ``budget`` outside every scope."""
    current = _CURRENT.get()
    return current if current is not None else MemoryBroker.from_budget(budget)


def current_broker() -> MemoryBroker | None:
    """The current scope's broker, None outside every scope (for code that is handed no budget: it charges the scope's
    ledger where there is one)."""
    return _CURRENT.get()
