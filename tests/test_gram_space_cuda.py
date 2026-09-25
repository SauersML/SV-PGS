"""``gram_space`` on a CUDA device: the band's sweep (``device_sweep``'s panel kernel on the band's panels) and its
posterior solves agree with the host's.

The sweep's two sides differ only in summation order: each member's field is s_g less a product over the band
(<= band width terms, float64), and its node sums run over the lattice (``mean_field._sweep``'s own terms), so each
coordinate agrees to a small multiple of the band's summation rounding, carried through the sweeps it takes."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.compute_budget import ComputeBudget, _try_import_cupy
from sv_pgs.gram_space import GramBand, GramGaussian
from sv_pgs.memory_broker import device_pool, memory_scope, transient_device_arrays
from tests.test_gram_space import _banded_design, _prior_arrays

cupy = _try_import_cupy()
pytestmark = pytest.mark.skipif(cupy is None, reason="needs a CUDA device")

_EPSILON = float(np.finfo(np.float64).eps)


def _band(generator: np.random.Generator, widths: list[int], dtype=np.float64):
    design = _banded_design(generator, widths, 40)
    gram = design.T @ design
    starts = np.concatenate([[0], np.cumsum(widths)])
    outcome = design @ (generator.normal(size=design.shape[1]) * 0.3) + generator.normal(size=design.shape[0])
    band = GramBand.from_arrays(
        within=[gram[starts[b]:starts[b + 1], starts[b]:starts[b + 1]].astype(dtype) for b in range(len(widths))],
        cross=[gram[starts[b]:starts[b + 1], starts[b + 1]:starts[b + 2]].astype(dtype) for b in range(len(widths) - 1)],
        scores=design.T @ outcome, target_square=float(outcome @ outcome), sample_count=design.shape[0],
        residual_dimension=float(design.shape[0]), working_bytes=1 << 22,
    )
    return band, starts


@pytest.mark.parametrize("dtype, tempered", [(np.float64, False), (np.float32, False), (np.float64, True)])
def test_device_band_sweep_is_the_host_band_sweep(dtype, tempered) -> None:
    generator = np.random.default_rng(11)
    # Blocks wider than a panel (32), so a block takes several panels and a tie shares one.
    widths = [70, 45, 90]
    band, starts = _band(generator, widths, dtype)
    if tempered:
        # The tempered band (the summary route's far field): each block its own lambda, the kernels its noise / lambda.
        band.set_tempering(generator.uniform(0.4, 1.0, size=len(widths)))
    count = band.group_count
    groups = np.concatenate([np.arange(count), [3, 3]])
    signs = np.concatenate([np.ones(count), [1.0, -1.0]])
    grid, log_density, scales = _prior_arrays(generator, groups.shape[0])
    block_of_group = np.searchsorted(starts, groups, side="right") - 1
    member_blocks = tuple(generator.permutation(np.flatnonzero(block_of_group == b)) for b in range(len(widths)))
    class_index = np.zeros(groups.shape[0], dtype=np.int64)
    states = {side: [np.zeros(groups.shape[0]) for _ in range(5)] for side in ("host", "device")}
    for _sweep in range(3):
        results = {}
        for side, xp in (("host", np), ("device", cupy)):
            state = states[side]
            results[side] = band.sweep(
                xp, member_blocks=member_blocks, group=groups, sign=signs, class_index=class_index, log_density=log_density, scales=scales,
                grid=grid, noise=0.9, mean=state[0], variance=state[1], shift=state[2], third=state[3], fourth=state[4],
            )
        bound = 256 * band.band_width * _EPSILON
        scale = max(float(np.max(np.abs(states["host"][0]))), 1.0)
        np.testing.assert_allclose(states["device"][0], states["host"][0], rtol=0, atol=bound * scale)
        np.testing.assert_allclose(states["device"][1], states["host"][1], rtol=bound)
        np.testing.assert_allclose(results["device"].residual_square, results["host"].residual_square, rtol=0, atol=bound * results["host"].residual_size)
        np.testing.assert_allclose(results["device"].divergence, results["host"].divergence, rtol=bound)


def test_device_band_posterior_solve_is_the_host_solve() -> None:
    generator = np.random.default_rng(12)
    band, _starts = _band(generator, [30, 50, 40])
    count = band.group_count
    precision = generator.uniform(0.05, 2.0, size=count)
    precision[[5, 60]] = [-0.02, 0.0]
    samples = 4 * 40
    right = generator.normal(size=(count, 3))
    solutions = []
    for xp in (np, cupy):
        solver = GramGaussian(band, training=np.ones((samples, 1)), targets=np.zeros((samples, 1)), covariates=np.ones((samples, 1)), array_module=xp)
        solver.iterate(site_precision=precision[:, None], site_shift=np.zeros((count, 1)), noise_variance=np.array([1.1]))
        solution, certificate = solver.posterior_solve(right, 0, np.full(3, 1e-10))
        assert np.all(np.isfinite(certificate))
        solutions.append(solution)
    exact = np.linalg.solve(band.product(np.eye(count)) / 1.1 + np.diag(precision), right)
    for solution in solutions:
        np.testing.assert_allclose(solution, exact, rtol=0, atol=1e-7 * float(np.max(np.abs(exact))))


class _ReservedPeak(cupy.cuda.memory_hook.MemoryHook if cupy is not None else object):
    """The pool's reserved bytes after every allocation it makes: the device's own count, beside the ledger's."""

    name = "reserved_peak"

    def __init__(self) -> None:
        self.peak = 0

    def malloc_postprocess(self, **_arguments) -> None:
        self.peak = max(self.peak, int(cupy.get_default_memory_pool().total_bytes()))


def _device_budget(device_bytes: int) -> ComputeBudget:
    return ComputeBudget("cuda", (0,), ("capped",), (int(device_bytes),), ((0, 0),), 1 << 34, 1)


def _capped_solve(generator_seed: int, device_bytes: int | None, keep: bool = True):
    """A device band solve (resolved sites included) inside a scope of ``device_bytes`` beyond what the pool holds at
    its start (None: the device's own), its Gram blocks cached on the device where ``keep``; returns (solution, exact, the ledger's peak, the pool's reserved peak, the pool
    at the start, the band's float32 bytes, the blocks the ledger did not keep)."""
    generator = np.random.default_rng(generator_seed)
    pool = cupy.get_default_memory_pool()
    pool.free_all_blocks()
    start = int(pool.total_bytes())
    free, _total = cupy.cuda.runtime.memGetInfo()
    hook = _ReservedPeak()
    with memory_scope(_device_budget(start + (int(free) if device_bytes is None else device_bytes))) as broker, hook:
        # The Gram blocks the ledger did not keep: evicted, or refused a cache lease.
        dropped: list[str] = []
        evict, admit = broker._evict, broker.admit
        broker._evict = lambda lease: (dropped.append(lease.purpose), evict(lease))

        def counted(name, nbytes, purpose, drop, allocated=False):
            lease = admit(name, nbytes, purpose, drop, allocated)
            if lease is None:
                dropped.append(purpose)
            return lease

        broker.admit = counted
        # Blocks wide enough that the device's page rounding of each reservation (2 MiB) is small beside them.
        band, _starts = _band(generator, [3000, 4000, 3500], np.float32)
        if not keep:
            band._keep = lambda *_arguments, **_keywords: None
        band.working_bytes = 1 << 30
        count = band.group_count
        precision = generator.uniform(0.05, 2.0, size=count)
        precision[[5, 6000]] = [-0.02, 0.0]
        samples = 4 * 40
        right = generator.normal(size=(count, 3))
        solver = GramGaussian(band, training=np.ones((samples, 1)), targets=np.zeros((samples, 1)), covariates=np.ones((samples, 1)), array_module=cupy)
        solver.iterate(site_precision=precision[:, None], site_shift=np.zeros((count, 1)), noise_variance=np.array([1.1]))
        solution, certificate = solver.posterior_solve(right, 0, np.full(3, 1e-10))
        assert np.all(np.isfinite(certificate))
        stored = sum(int(values.nbytes) for values in band._arrays.values())
        peak = broker.peak(device_pool(0))
        band.release()
    exact = np.linalg.solve(band.product(np.eye(count)) / 1.1 + np.diag(precision), right)
    return solution, exact, peak, hook.peak, start, stored, dropped.count("a cached Stage 0 Gram block")


def test_a_device_band_solve_under_a_budget_below_its_unconstrained_peak_completes_within_it() -> None:
    """The ledger's own mechanism caps the device below the solve's unconstrained peak, at its peak with no Gram block
    kept past its use. The solve completes, the ledger's peak and the pool's reserved bytes after every allocation stay
    within the budget, and the solution is the unconstrained one's."""
    free_solution, exact, free_peak, _reserved, free_start, _stored, dropped = _capped_solve(21, None)
    assert dropped == 0
    _solution, _exact, working_peak, _reserved, working_start, _stored, _dropped = _capped_solve(21, None, keep=False)
    budget = working_peak - working_start
    assert budget < free_peak - free_start, "the budget binds: below the unconstrained solve's peak"
    solution, capped_exact, peak, reserved, start, _stored, _dropped = _capped_solve(21, budget)
    assert peak <= start + budget and reserved <= start + budget
    for values in (free_solution, solution):
        np.testing.assert_allclose(values, capped_exact, rtol=0, atol=1e-7 * float(np.max(np.abs(capped_exact))))
    np.testing.assert_array_equal(exact, capped_exact)


def test_the_device_ledger_holds_the_fragments_its_pool_cannot_return() -> None:
    """A block freed and then split by a smaller allocation leaves a fragment the pool keeps and cannot return: the
    ledger counts it, and refuses what the live bytes alone would have admitted (bench-sim v7 chr22 001 [bench]: 1.2 GB
    admitted and then refused by a 40 GB A100 with 40.9 GB reserved)."""
    pool = cupy.get_default_memory_pool()
    pool.free_all_blocks()
    start = int(pool.total_bytes())
    size = 64 << 20
    with memory_scope(_device_budget(start + 2 * size + size // 2)) as broker:
        block = cupy.empty(size, dtype=cupy.uint8)
        del block
        small = cupy.empty(1 << 20, dtype=cupy.uint8)
        assert int(pool.total_bytes()) - int(pool.used_bytes()) >= size - (1 << 20)
        assert broker.held(device_pool(0)) >= start + size
        assert int(pool.used_bytes()) + 2 * size <= start + 2 * size + size // 2
        with pytest.raises(cupy.cuda.memory.OutOfMemoryError):
            cupy.empty(2 * size, dtype=cupy.uint8)
        del small
    pool.free_all_blocks()


def test_a_transient_device_array_is_reserved_outside_the_pool_charged_first_and_returned_when_freed() -> None:
    """``transient_device_arrays``: the array's bytes are the device's, not the pool's (so no free block of its size is
    left in the pool for a smaller array to split), the ledger counts them while it lives and admits it only within the
    budget, and freeing it returns them to the device."""
    pool = cupy.get_default_memory_pool()
    pool.free_all_blocks()
    start = int(pool.total_bytes())
    size = 64 << 20
    with memory_scope(_device_budget(start + 2 * size)) as broker:
        free = int(cupy.cuda.runtime.memGetInfo()[0])
        with transient_device_arrays(cupy):
            block = cupy.ones(size, dtype=cupy.uint8)
        assert int(pool.total_bytes()) == start and broker.held(device_pool(0)) >= start + size
        assert free - int(cupy.cuda.runtime.memGetInfo()[0]) >= size
        del block
        assert int(cupy.cuda.runtime.memGetInfo()[0]) >= free and int(pool.total_bytes()) == start
        with pytest.raises(cupy.cuda.memory.OutOfMemoryError), transient_device_arrays(cupy):
            cupy.empty(3 * size, dtype=cupy.uint8)
