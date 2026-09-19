"""How every visible CUDA device shares Stage 2's genotype work (SPEC: all visible devices, sharded by LD block).

Stage 0 already runs one buffer per device, a chromosome at a time (``genotype_statistics``), and
scoring spreads its blocks over the devices (``fast_scoring``). This module shards Stage 2's reads:

* :func:`plan_blocks` assigns each LD block to one device, longest block first onto the least
  loaded device (Graham's LPT rule; its makespan is within ``4/3 - 1/(3D)`` of the best for ``D``
  devices), with a block's load its width times the sample count, the int8 product work of a read.
* :class:`ShardedDualSource` is a ``dual_solve.DualTileSource`` over one tile source per device.
  Its ``blocks()`` keeps every existing read correct on several devices: each block's products run
  on the block's device, with the home device's operands moved there and the results moved back.
  Its ``map_reduce`` runs a read's blocks on all devices at once, each into its own sample-side
  image, and sums the images in device order, so the result is deterministic.
* :func:`check_visible_devices` makes an exposed device that CUDA cannot use an error, and
  :func:`log_device_plan` records the decision when a fit starts.

Integer products are computed wholly on one device, so they are the same bits whichever device ran
them. Any association of a fp64 sum over ``B`` block images is within ``gamma_(B-1) * sum_b |image_b|``
of the exact sum elementwise (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
eq. 4.4), so the devices' sum and the single-device sum differ by at most twice that, which
:func:`reassociation_bound` returns.
"""
from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from sv_pgs.compute_budget import ComputeBudget, _cupy_device_context
from sv_pgs.progress import log


def exposed_device_count() -> int | None:
    """Devices exposed to this process: the entries of ``CUDA_VISIBLE_DEVICES``, else the
    ``/dev/nvidia<N>`` device nodes, else None when neither says anything."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        entries = [entry for entry in visible.split(",") if entry.strip() not in ("", "-1")]
        return len(entries)
    nodes = [path for path in Path("/dev").glob("nvidia*") if re.fullmatch(r"nvidia\d+", path.name)]
    return len(nodes) if nodes else None


def check_visible_devices(budget: ComputeBudget) -> None:
    """Raise if a device is exposed to this process but the budget cannot use it."""
    exposed = exposed_device_count()
    if exposed is None:
        return
    usable = len(budget.device_ids) if budget.device_kind == "cuda" else 0
    if usable != exposed:
        raise RuntimeError(
            f"{exposed} CUDA device(s) are exposed to this process but {usable} are usable; "
            "every exposed device must work, or be hidden with CUDA_VISIBLE_DEVICES."
        )


@dataclass(frozen=True, slots=True)
class DevicePlan:
    """Which device reads each LD block: ``block_devices[b]`` is a position in ``device_ids``."""

    device_ids: tuple[int, ...]
    block_devices: NDArray[np.int64]
    loads: tuple[int, ...]

    def blocks_on(self, position: int) -> NDArray[np.int64]:
        """The blocks of one device, ascending."""
        return np.flatnonzero(self.block_devices == position).astype(np.int64)

    def describe(self) -> str:
        total = sum(self.loads)
        shares = ", ".join(
            f"device {device_id}: {int(np.sum(self.block_devices == position))} blocks, {load / total:.1%} of the work"
            for position, (device_id, load) in enumerate(zip(self.device_ids, self.loads, strict=True))
        ) if total else "no blocks"
        return f"{len(self.device_ids)} device(s): {shares}"


def plan_blocks(device_ids: Sequence[int], block_widths: Sequence[int], sample_count: int) -> DevicePlan:
    """Graham's longest-processing-time rule: blocks widest first onto the least loaded device
    (ties: the lower device position, then the lower block index)."""
    if not device_ids:
        raise ValueError("a device plan needs at least one device")
    widths = np.asarray(block_widths, dtype=np.int64)
    if widths.ndim != 1 or np.any(widths <= 0):
        raise ValueError("block widths must be positive")
    loads = [0] * len(device_ids)
    block_devices = np.empty(widths.shape[0], dtype=np.int64)
    for block in sorted(range(widths.shape[0]), key=lambda index: (-int(widths[index]), index)):
        position = min(range(len(device_ids)), key=lambda index: (loads[index], index))
        block_devices[block] = position
        loads[position] += int(widths[block]) * int(sample_count)
    return DevicePlan(tuple(int(device_id) for device_id in device_ids), block_devices, tuple(loads))


def log_device_plan(budget: ComputeBudget, plan: DevicePlan) -> None:
    """The device decision, logged once when a fit starts."""
    log(f"  device plan ({budget.device_kind}): {plan.describe()}")


def reassociation_bound(block_images: Sequence[Any], array_module: Any) -> Any:
    """Elementwise bound on how far two summation orders of ``block_images`` can differ in fp64.

    Each order is within ``gamma_(B-1) M`` of the exact sum, ``M = sum_b |image_b|``. The bound is
    evaluated in fp64 as ``2 gamma_(B+1) M~``: the computed M~ is at least ``(1 - gamma_(B-1)) M``
    and the bound's own arithmetic rounds by a relative ``4u`` or less, while
    ``gamma_(B+1) / gamma_(B-1) >= 1 + 2 / (B - 1)`` covers both for any ``B`` below ``u^-1/2``.
    """
    count = len(block_images)
    if count <= 1:
        return array_module.zeros_like(block_images[0]) if count else 0.0
    unit = np.finfo(np.float64).eps / 2.0
    if count * count * unit >= 1.0:
        raise ValueError(f"{count} blocks are too many for the fp64 reassociation bound")
    gamma = (count + 1) * unit / (1.0 - (count + 1) * unit)
    magnitude = array_module.zeros_like(block_images[0])
    for image in block_images:
        magnitude += array_module.abs(image)
    return 2.0 * gamma * magnitude


class ShardTileSource(Protocol):
    """One device's share of the blocks: ``iter_tiles`` yields ``(local_index, tile)`` in the order
    of ``DevicePlan.blocks_on`` for that device, each tile valid until the next is requested
    (``store_block_source.StoreGenotypeBlockSource`` built on that device, say)."""

    def iter_tiles(self) -> Iterator[tuple[int, Any]]: ...


def moved(cupy: Any, values: Any, device_id: int) -> Any:
    """``values`` (a cupy array on any device, or a numpy array) as a contiguous array on ``device_id``,
    with the copy ordered after the work that wrote ``values`` on its own device."""
    if isinstance(values, np.ndarray):
        with _cupy_device_context(cupy, device_id):
            return cupy.asarray(values)
    source_device = int(values.device.id)
    if source_device == device_id:
        return values
    with _cupy_device_context(cupy, source_device):
        contiguous = cupy.ascontiguousarray(values)
        written = cupy.cuda.Event()
        written.record()
    with _cupy_device_context(cupy, device_id):
        stream = cupy.cuda.get_current_stream()
        stream.wait_event(written)
        target = cupy.empty(contiguous.shape, dtype=contiguous.dtype)
        target.data.copy_from_device_async(contiguous.data, contiguous.nbytes, stream)
        copied = cupy.cuda.Event()
        copied.record(stream)
    # the source buffer must outlive the asynchronous copy
    with _cupy_device_context(cupy, source_device):
        cupy.cuda.get_current_stream().wait_event(copied)
    return target


class _PeerOperand:
    """A read's sample-side operand, prepared once per device on first use there."""

    def __init__(self, left: Any, relative_error: float) -> None:
        self.left = left
        self.relative_error = relative_error
        self.prepared: dict[int, Any] = {}


class _PeerTile:
    """A tile on ``device_id`` whose inputs and outputs live on ``home_id`` (``dual_solve.DualTile``)."""

    def __init__(self, tile: Any, device_id: int, home_id: int, cupy: Any) -> None:
        self._tile = tile
        self._device_id = device_id
        self._home_id = home_id
        self._cupy = cupy

    def _there(self, values: Any) -> Any:
        return moved(self._cupy, values, self._device_id)

    def _home(self, values: Any) -> Any:
        return moved(self._cupy, values, self._home_id)

    def _run(self, method: Callable[..., Any], *arguments: Any) -> Any:
        with _cupy_device_context(self._cupy, self._device_id):
            return self._home(method(*arguments))

    def sample_operand(self, left: Any, relative_error: float) -> _PeerOperand:
        return _PeerOperand(left, relative_error)

    def rmatmat(self, left: Any) -> Any:
        if not isinstance(left, _PeerOperand):
            return self._run(self._tile.rmatmat, self._there(left))
        if self._device_id not in left.prepared:
            with _cupy_device_context(self._cupy, self._device_id):
                left.prepared[self._device_id] = self._tile.sample_operand(self._there(left.left), left.relative_error)
        return self._run(self._tile.rmatmat, left.prepared[self._device_id])

    def matmat(self, right: Any) -> Any:
        return self._run(self._tile.matmat, self._there(right))

    def accumulate_matmat(self, right: Any, image: Any, relative_error: float) -> None:
        with _cupy_device_context(self._cupy, self._device_id):
            partial = self._cupy.zeros(tuple(image.shape), dtype=image.dtype)
            self._tile.accumulate_matmat(self._there(right), partial, relative_error)
        image += self._home(partial)

    def weighted_column_squares(self, weights: Any) -> Any:
        return self._run(self._tile.weighted_column_squares, self._there(weights))

    def columns(self, local: Any) -> Any:
        return self._run(self._tile.columns, self._there(local))


BlockWork = Callable[[int, int, Any, dict[str, Any], dict[str, Any], Any], None]
"""``work(start, stop, tile, shared, rows, image)`` for one block on its device: ``shared`` holds the
read's arrays on that device (the work may cache per-device state in it, such as the read's
prepared operand), ``rows`` the block's ``[start:stop]`` slices of the variant-side arrays, and
``image`` the device's sample-side accumulator."""


class ShardedDualSource:
    """A ``dual_solve.DualTileSource`` whose LD blocks are spread over several CUDA devices by a
    :class:`DevicePlan`. ``shards[position]`` streams the blocks of ``plan.device_ids[position]``;
    the home device, where callers keep their operands, is ``plan.device_ids[0]``."""

    def __init__(self, shards: Sequence[ShardTileSource], plan: DevicePlan, block_bounds: Sequence[tuple[int, int]], sample_count: int, cupy: Any) -> None:
        if len(shards) != len(plan.device_ids):
            raise ValueError("one shard per planned device")
        if len(block_bounds) != plan.block_devices.shape[0]:
            raise ValueError("one planned device per block")
        previous = 0
        for start, stop in block_bounds:
            if start != previous or stop <= start:
                raise ValueError("blocks must tile the variant axis contiguously, in order")
            previous = stop
        self._shards = list(shards)
        self.plan = plan
        self.block_bounds = [(int(start), int(stop)) for start, stop in block_bounds]
        self.variant_count = previous
        self.sample_count = int(sample_count)
        self.array_module = cupy
        self._cupy = cupy
        self.home_id = plan.device_ids[0]

    @classmethod
    def from_statistics(cls, store: Any, statistics: Any, budget: ComputeBudget, workspace_bytes: int) -> ShardedDualSource:
        """One Stage 0 pass's reduced LD blocks over every device of ``budget``, each device streaming
        its own blocks from ``store``. The devices read at once, so each shard's read-ahead plans
        against an equal share of the host memory."""
        from sv_pgs.compute_budget import _try_import_cupy
        from sv_pgs.store_block_source import StoreGenotypeBlockSource, reduced_block_layout

        if budget.device_kind != "cuda":
            raise ValueError("a sharded source needs CUDA devices; one host source serves the CPU")
        cupy = _try_import_cupy()
        if cupy is None:
            raise RuntimeError("the compute budget names CUDA devices but CuPy cannot be imported")
        check_visible_devices(budget)
        block_rows, block_indices, means, scales = reduced_block_layout(statistics)
        plan = plan_blocks(budget.device_ids, [rows.shape[0] for rows in block_rows], int(store.n_samples))
        log_device_plan(budget, plan)
        shard_host_bytes = budget.host_bytes // len(budget.device_ids)
        shards = []
        for position, device_id in enumerate(budget.device_ids):
            blocks = plan.blocks_on(position)
            shard_budget = replace(
                budget,
                device_ids=(device_id,),
                device_names=(budget.device_names[position],),
                device_bytes=(budget.device_bytes[position],),
                device_compute_capabilities=(budget.device_compute_capabilities[position],),
                host_bytes=shard_host_bytes,
            )
            with _cupy_device_context(cupy, device_id):
                shards.append(StoreGenotypeBlockSource(
                    store,
                    [block_rows[block] for block in blocks],
                    [block_indices[block] for block in blocks],
                    np.concatenate([means[block] for block in blocks]),
                    np.concatenate([scales[block] for block in blocks]),
                    shard_budget,
                    workspace_bytes,
                ))
        bounds = [(int(indices[0]), int(indices[-1]) + 1) for indices in block_indices]
        return cls(shards, plan, bounds, int(store.n_samples), cupy)

    def blocks(self) -> Iterator[tuple[int, int, Any]]:
        """Every block in variant order, as a tile that takes and returns home-device arrays."""
        iterators = [iter(shard.iter_tiles()) for shard in self._shards]
        for block, (start, stop) in enumerate(self.block_bounds):
            position = int(self.plan.block_devices[block])
            device_id = self.plan.device_ids[position]
            with _cupy_device_context(self._cupy, device_id):
                _local, tile = next(iterators[position])
            yield start, stop, _PeerTile(tile, device_id, self.home_id, self._cupy)

    def map_reduce(self, work: BlockWork, shared: Mapping[str, Any], rows: Mapping[str, Any], image_shape: tuple[int, ...]) -> Any:
        """Run ``work`` on every block, all devices at once, and return the sum of the devices'
        images on the home device, summed in device order."""
        images: list[Any] = [None] * len(self._shards)
        failures: list[BaseException] = []

        def device_worker(position: int) -> None:
            device_id = self.plan.device_ids[position]
            blocks = self.plan.blocks_on(position)
            try:
                with _cupy_device_context(self._cupy, device_id):
                    device_shared = {name: moved(self._cupy, values, device_id) for name, values in shared.items()}
                    image = self._cupy.zeros(image_shape, dtype=self._cupy.float64)
                    for block, (_index, tile) in zip(blocks, self._shards[position].iter_tiles(), strict=True):
                        start, stop = self.block_bounds[int(block)]
                        block_rows = {name: moved(self._cupy, values[start:stop], device_id) for name, values in rows.items()}
                        work(start, stop, tile, device_shared, block_rows, image)
                    self._cupy.cuda.get_current_stream().synchronize()
                    images[position] = image
            except BaseException as error:
                failures.append(error)

        workers = [
            threading.Thread(target=device_worker, args=(position,), name=f"stage2-device-{device_id}")
            for position, device_id in enumerate(self.plan.device_ids)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        if failures:
            raise failures[0]
        with _cupy_device_context(self._cupy, self.home_id):
            total = images[0]
            for image in images[1:]:
                total = total + moved(self._cupy, image, self.home_id)
            return total
