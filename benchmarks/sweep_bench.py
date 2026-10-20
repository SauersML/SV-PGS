"""Old against new seconds per mean-field sweep on held int8 codes, synthetic, pytest-free, on one GPU by default:

    python -m benchmarks.sweep_bench [--samples 44000] [--members 524288] [--blocks 28] [--devices 1]

Both routes sweep the same synthetic codes, members and models, the blocks planned over the devices as Stage 2 plans
them (``device_plan.plan_blocks``), one piece per block:

- old: the route the whole-genome fits took (``ShardedDualSource.blocks``' peer tiles: each panel decoded on its
  block's device, copied to the residual's and swept by ``sweep_piece``, one model at a time; ``full_data_fit
  ._device_sweep``). Two sweeps of model 0 (the first builds its panel Grams); the second is timed, per model. On one
  device no panel is copied between devices, so its time is a floor for the two-device fits' route.
- new: ``device_sweep.sweep_models``, every model of the group in one launch per panel on the device that holds the
  panel's codes. Three sweeps of three models (the first builds the Grams; the next two are timed as the warm sweeps a
  solve repeats); one with every launch made to an empty kernel with the same arguments, the host's share (the
  launches' submissions, the state's copies, the Grams' stacking: where it is below the full sweep's time the GPU is
  the bottleneck); and two of nine models (three groups' models, or three traits' groups, read in one pass of the
  codes).

Each warm time is extrapolated per member to the All of Us whole-genome fits' 1.0e6 members [real], three models a
group [real]: a sweep's work is per panel, and every panel has the same work. The codes are uniform random int8 (no
participant data; a panel's chain does not depend on their values), 146 lattice nodes and 21 covariates [real], 44,000
samples [real: 1.7e4 to 4.4e4 per group], blocks of ~18,700 members (1.0e6 members in ~55 LD blocks [real]). At the
defaults the codes take 23 GB of one A100-40GB, and the whole command about 2-4 minutes.

Checks print PASS/FAIL with their times: model 0's old sweep against its new sweep from the same start (rtol 1e-8:
the same members, order and panels, other summation orders), and model 0 swept alone and among nine models against its
sweep among three (bit for bit). A RESULT line gives old and new seconds per sweep at 1.0e6 members. A ten-minute
subprocess deadline includes imports, the codes' synthesis and kernel compilation; a timeout is a FAIL. ``--host``
runs the new route's harness on numpy with its reference kernels (``HostPanelKernels``), at a size the host can take,
and skips the old route, which needs CUDA.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from benchmarks.gpu_smoke import Checks, _require

# Scenario sizes belong to this benchmark, not to the inference package.
SAMPLES = 44_000
MEMBERS = 524_288
BLOCKS = 28
DEVICES = 1
MODELS = 3
WIDE_MODELS = 9
NODES = 146
COVARIATES = 21
WHOLE_GENOME_MEMBERS = 1_000_000
TIMEOUT_SECONDS = 600
CHECK_RTOL = 1e-8
WORKSPACE_BYTES = 1 << 26
CODE_ROWS_PER_DRAW = 1024
GRAM_COLUMNS = 2048
MASK_FRACTIONS = (0.9, 0.8, 0.7)
EMPTY_SOURCE = 'extern "C" __global__ void empty_step() {}'


class _Shard:
    """One device's blocks as ``device_plan.ShardedDualSource`` reads a shard: its held tiles, in plan order."""

    def __init__(self, tiles):
        self.tiles = tiles

    def iter_tiles(self):
        yield from enumerate(self.tiles)

    def holds_codes(self):
        return True


def _host(values):
    import numpy as np

    return np.asarray(values.get() if hasattr(values, "get") else values)


def _problem(xp, device_ids, samples, members, blocks, models):
    """Codes on their devices; ``models`` models' masks, priors and starting residuals (synthetic)."""
    import numpy as np
    from sv_pgs.code_products import CodeBlockTile
    from sv_pgs.compute_budget import _cupy_device_context
    from sv_pgs.device_plan import ShardedDualSource, plan_blocks

    rng = np.random.default_rng(29)
    widths = [members // blocks + (index < members % blocks) for index in range(blocks)]
    plan = plan_blocks(device_ids, widths, samples)
    shard_tiles = [[] for _device in device_ids]
    bounds, member_blocks, devices = [], [], []
    start = 0
    for block, width in enumerate(widths):
        position = int(plan.block_devices[block])
        device = plan.device_ids[position]
        with _cupy_device_context(xp, device):
            generator = xp.random.RandomState(block)
            codes = xp.empty((width, samples), dtype=xp.int8)
            means, scales = xp.empty(width), xp.empty(width)
            for first in range(0, width, CODE_ROWS_PER_DRAW):
                last = min(first + CODE_ROWS_PER_DRAW, width)
                drawn = generator.randint(-60, 61, size=(last - first, samples), dtype=xp.int32).astype(xp.float64)
                codes[first:last] = drawn.astype(xp.int8)
                means[first:last], scales[first:last] = drawn.mean(axis=1), drawn.std(axis=1)
                del drawn
            shard_tiles[position].append(CodeBlockTile(codes, means, scales, xp, WORKSPACE_BYTES))
            del codes
        bounds.append((start, start + width))
        member_blocks.append(np.arange(start, start + width))
        devices.append(device)
        start += width
    sharded = ShardedDualSource([_Shard(tiles) for tiles in shard_tiles], plan, bounds, samples, xp)
    covariates = np.column_stack([np.ones(samples), rng.normal(size=(samples, COVARIATES - 1))])
    masks = np.stack([(rng.random(samples) < MASK_FRACTIONS[model % len(MASK_FRACTIONS)]).astype(np.float64) for model in range(models)])
    pinvs, residuals = [], []
    for model in range(models):
        masked = covariates * masks[model][:, None]
        pinv = np.linalg.pinv(masked.T @ masked)
        target = rng.normal(size=samples) * masks[model]
        residuals.append(target - masked @ (pinv @ (masked.T @ target)))
        pinvs.append(pinv)
    grid = np.linspace(np.log(1e-6), 0.0, NODES)
    log_density = rng.normal(size=(models, 1, NODES))
    log_density -= np.log(np.exp(log_density).sum(axis=2, keepdims=True))
    return dict(
        sharded=sharded, bounds=bounds, devices=devices, member_blocks=member_blocks, covariates=covariates, masks=masks, pinvs=pinvs,
        residuals=np.stack(residuals), grid=grid, log_density=log_density, log_scale=rng.normal(-2.0, 0.5, (members, models)),
        # ||x_j||^2 over a model's training rows for standardized columns; the projection's small share left out.
        squares=np.tile(masks.sum(axis=1), (members, 1)),
        noise=np.ones(models),
    )


def _new(xp, problem, kernels, models, state, residuals, grams, layout):
    import numpy as np
    from sv_pgs.device_plan import native_blocks
    from sv_pgs.device_sweep import sweep_models

    columns = list(models)
    return sweep_models(
        xp, kernels, layout, native_blocks(problem["sharded"], xp), grams=grams, keys=[(model, b"") for model in columns],
        masks=problem["masks"][columns], residuals=residuals, noise=problem["noise"][columns], log_density=problem["log_density"][columns],
        grid=problem["grid"], log_scale=np.ascontiguousarray(problem["log_scale"][:, columns]),
        squares=np.ascontiguousarray(problem["squares"][:, columns]), state=state, covariates=xp.asarray(problem["covariates"]),
        covariate_pinvs=[xp.asarray(problem["pinvs"][model]) for model in columns], gram_columns=lambda width: GRAM_COLUMNS,
    )


def _old(cp, problem, model, state, residual, grams):
    """One sweep of ``model`` as ``full_data_fit._device_sweep`` ran it on a sharded source: peer tiles, decoded panels."""
    import numpy as np
    from sv_pgs.device_sweep import PIECE_COLUMNS, CodePanels, sweep_piece

    mask = cp.asarray(problem["masks"][model])
    masked_covariates = cp.asarray(problem["covariates"]) * mask[:, None]
    pinv = cp.asarray(problem["pinvs"][model])
    log_density = cp.asarray(problem["log_density"][model])
    scales = cp.asarray(problem["log_scale"][:, model])
    grid = cp.asarray(problem["grid"])
    squares = cp.asarray(np.ascontiguousarray(problem["squares"][:, model]))
    totals = cp.zeros(PIECE_COLUMNS)
    for (start, _stop, tile), members in zip(problem["sharded"].blocks(), problem["member_blocks"]):
        rows = cp.asarray(members)
        local = cp.asarray(members - start)

        def decode(first, last, tile=tile, local=local):
            return cp.asarray(tile.columns(local[first:last]), dtype=cp.float64)

        log_node_variance = cp.ascontiguousarray(scales[rows][:, None] + grid[None, :])
        piece_state = {name: cp.ascontiguousarray(values[rows]) for name, values in state.items()}
        pieces = cp.zeros((members.shape[0], PIECE_COLUMNS))
        sweep_piece(
            cp, decode=decode, width=int(members.shape[0]), mask=mask, covariates=masked_covariates, covariate_pinv=pinv, residual=residual,
            grams=grams, model=model, members=members, squares=cp.ascontiguousarray(squares[rows]),
            class_index=cp.zeros(members.shape[0], dtype=cp.int64), log_density=log_density, node_variance=cp.exp(log_node_variance),
            log_node_variance=log_node_variance, noise=float(problem["noise"][model]), pieces=pieces, **piece_state,
            panels=CodePanels(cp, tile, local, cp.ones(members.shape[0])) if CodePanels.supports(tile) else None,
        )
        for name, values in piece_state.items():
            state[name][rows] = values
        totals += pieces.sum(axis=0)
    return totals


class _NoKernels:
    """``sweep_models``' kernels with nothing launched: the host's share of a sweep on numpy."""

    def __init__(self, kernels):
        self._kernels = kernels

    def threads(self, sample_count):
        return self._kernels.threads(sample_count)

    def node_terms(self, arguments, total):
        return None

    def panel_step(self, blocks, threads, models, arguments):
        return None


class _EmptyKernels(_NoKernels):
    """``sweep_models``' launches made to an empty kernel with the same grid, block, shared memory and arguments (the
    driver reads a kernel's parameters from its image, so the empty kernel ignores them; CuPy packs every one): the
    host's share of a sweep on a device, its submissions included, with next to no GPU time behind it."""

    def __init__(self, cupy, kernels):
        super().__init__(kernels)
        self._cupy = cupy
        self._empty = {}

    def _kernel(self):
        device = int(self._cupy.cuda.runtime.getDevice())
        if device not in self._empty:
            self._empty[device] = self._cupy.RawKernel(EMPTY_SOURCE, "empty_step")
        return self._empty[device]

    def node_terms(self, arguments, total):
        self._kernel()((1,), (1,), arguments)

    def panel_step(self, blocks, threads, models, arguments):
        from sv_pgs.device_sweep import _panel_step_shared_bytes

        self._kernel()((blocks,), (threads,), arguments, shared_mem=_panel_step_shared_bytes(threads))


def _worker(samples, members, blocks, devices, host):
    import numpy as np
    from sv_pgs.device_sweep import MOMENTS, PANEL, CudaPanelKernels, HostPanelKernels, PanelGrams, SweepLayout

    checks = Checks()
    if host:
        xp, device_ids = np, (0,)
    else:
        import cupy as xp

        count = min(devices, int(xp.cuda.runtime.getDeviceCount()))
        _require(count > 0, "no CUDA GPU visible")
        device_ids = tuple(range(count))
        xp.cuda.Device(0).use()

    def synchronize():
        if xp is not np:
            for device_id in device_ids:
                xp.cuda.Device(device_id).synchronize()

    checks.synchronize = synchronize
    problem = checks.run(
        f"synthetic codes ({members} members x {samples} samples on {len(device_ids)} {'host' if host else 'GPU'} devices)",
        lambda: _problem(xp, device_ids, samples, members, blocks, WIDE_MODELS),
    )
    if problem is None:
        return 1
    kernels = HostPanelKernels(2 * PANEL) if host else CudaPanelKernels(xp, NODES)
    layout = SweepLayout(
        xp, problem["devices"], problem["member_blocks"], problem["bounds"], np.arange(members), np.ones(members), np.zeros(members, dtype=np.int64),
        lambda width: iter(((0, width),)),
    )
    panels = layout.panel_count
    times = {}
    first_sweep = {}

    def start(models):
        return {"state": {name: np.zeros((members, models)) for name in MOMENTS}, "residuals": problem["residuals"][:models].copy(), "grams": PanelGrams()}

    def sweep(run, models, label, sweep_kernels=None):
        started = time.perf_counter()
        swept, totals, squares = _new(xp, problem, kernels if sweep_kernels is None else sweep_kernels, range(models), run["state"], run["residuals"], run["grams"], layout)
        synchronize()
        times[label] = time.perf_counter() - started
        if sweep_kernels is None:
            _require(np.all(np.isfinite(totals)) and np.all(np.isfinite(squares)), "a new sweep is not finite")
            run["residuals"] = np.stack(swept)
        return swept, totals

    def model_0_is_the_first_sweep(swept, totals, state):
        _require(bool(first_sweep), "the first new sweep failed")
        np.testing.assert_array_equal(swept[0], first_sweep["residual"])
        np.testing.assert_array_equal(totals[0], first_sweep["totals"])
        for name in MOMENTS:
            np.testing.assert_array_equal(state[name][:, 0], first_sweep["state"][name])

    new = start(MODELS)

    def cold():
        swept, totals = sweep(new, MODELS, "new cold")
        first_sweep.update(residual=swept[0].copy(), totals=totals[0].copy(), state={name: values[:, 0].copy() for name, values in new["state"].items()})

    checks.run(f"new sweep 1 of {MODELS} models (builds the panel Grams)", cold)
    checks.run(f"new sweep 2 of {MODELS} models", lambda: sweep(new, MODELS, "new warm 2"))
    checks.run(f"new sweep 3 of {MODELS} models", lambda: sweep(new, MODELS, "new warm 3"))
    saved = {name: values.copy() for name, values in new["state"].items()}
    checks.run(
        f"new sweep, host share ({'no kernels' if host else 'every launch made to an empty kernel'})",
        lambda: sweep(new, MODELS, "host share", _NoKernels(kernels) if host else _EmptyKernels(xp, kernels)),
    )
    new["state"] = saved

    def alone():
        run = start(1)
        swept, totals = sweep(run, 1, "new alone")
        model_0_is_the_first_sweep(swept, totals, run["state"])

    checks.run("model 0 alone is its sweep among three, bit for bit", alone)
    del new

    if not host:
        old = {"state": {name: xp.zeros(members) for name in MOMENTS}, "residual": xp.asarray(problem["residuals"][0]), "grams": PanelGrams()}

        def old_sweep(label):
            started = time.perf_counter()
            totals = _old(xp, problem, 0, old["state"], old["residual"], old["grams"])
            synchronize()
            times[label] = time.perf_counter() - started
            return totals

        def old_first():
            old_sweep("old cold")
            _require(bool(first_sweep), "the first new sweep failed")
            scale = 1.0 + max(float(np.max(np.abs(values))) for values in first_sweep["state"].values())
            for name in MOMENTS:
                np.testing.assert_allclose(_host(old["state"][name]), first_sweep["state"][name], rtol=CHECK_RTOL, atol=CHECK_RTOL * scale, err_msg=name)
            residual = first_sweep["residual"]
            np.testing.assert_allclose(_host(old["residual"]), residual, rtol=CHECK_RTOL, atol=CHECK_RTOL * float(np.linalg.norm(residual)))

        checks.run("old sweep 1 of model 0 is its new sweep 1", old_first)
        checks.run("old sweep 2 of model 0", lambda: old_sweep("old warm"))
        del old

    wide = start(WIDE_MODELS)

    def wide_cold():
        swept, totals = sweep(wide, WIDE_MODELS, "wide cold")
        model_0_is_the_first_sweep(swept, totals, wide["state"])

    checks.run(f"new sweep 1 of {WIDE_MODELS} models: model 0 is its sweep among three, bit for bit", wide_cold)
    checks.run(f"new sweep 2 of {WIDE_MODELS} models", lambda: sweep(wide, WIDE_MODELS, "wide warm"))
    del wide

    scale = WHOLE_GENOME_MEMBERS / members
    name = "host" if host else xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"INFO {members} members in {panels} panels over {blocks} blocks, {samples} samples, {NODES} nodes, {COVARIATES} covariates, "
          f"{len(device_ids)} device(s) ({name})", flush=True)
    warm = min(times.get("new warm 2", float("inf")), times.get("new warm 3", float("inf")))
    if warm < float("inf"):
        print(f"INFO new: cold {times.get('new cold', float('nan')):.3f} s, warm {warm:.3f} s a sweep of all {MODELS} models "
              f"({1e6 * warm / panels:.1f} us a panel); host share {times.get('host share', float('nan')):.3f} s "
              f"({1e6 * times.get('host share', float('nan')) / panels:.1f} us a panel)", flush=True)
    if "wide warm" in times:
        print(f"INFO new, {WIDE_MODELS} models a launch: cold {times.get('wide cold', float('nan')):.3f} s, warm {times['wide warm']:.3f} s "
              f"({1e6 * times['wide warm'] / panels:.1f} us a panel)", flush=True)
    if "old warm" in times:
        print(f"INFO old: cold {times.get('old cold', float('nan')):.3f} s, warm {times['old warm']:.3f} s a sweep of one model "
              f"({1e3 * times['old warm'] / panels:.2f} ms a panel)", flush=True)
    result = [f"RESULT seconds per sweep at {WHOLE_GENOME_MEMBERS:.1e} members, {samples} samples:"]
    if "old warm" in times:
        result.append(f"old {times['old warm'] * scale:.1f} s a model, {MODELS * times['old warm'] * scale:.1f} s for {MODELS};")
    if warm < float("inf"):
        result.append(f"new {warm * scale:.2f} s for all {MODELS}")
        if "wide warm" in times:
            result.append(f"({times['wide warm'] * scale:.2f} s for {WIDE_MODELS})")
        if "old warm" in times:
            result.append(f"; speedup {MODELS * times['old warm'] / warm:.0f}x")
    print(" ".join(result), flush=True)
    return int(checks.failures > 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--members", type=int, default=MEMBERS)
    parser.add_argument("--blocks", type=int, default=BLOCKS)
    parser.add_argument("--devices", type=int, default=DEVICES, help="GPUs to plan the blocks over (default: 1)")
    parser.add_argument("--host", action="store_true", help="the new route's harness on numpy, its reference kernels, no old route")
    parser.add_argument("--timeout-seconds", type=float, default=TIMEOUT_SECONDS, help="whole command deadline, including compilation (default: 600)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        try:
            return _worker(args.samples, args.members, args.blocks, args.devices, args.host)
        except Exception as error:
            traceback.print_exc()
            print(f"FAIL harness {type(error).__name__}: {' '.join(str(error).split())}", flush=True)
            return 1
    if not 0 < args.timeout_seconds < float("inf"):
        parser.error("--timeout-seconds must be positive and finite")
    if args.devices < 1:
        parser.error("--devices must be at least 1")
    started = time.perf_counter()
    environment = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMBA_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    command = [sys.executable, "-m", "benchmarks.sweep_bench", "--worker", "--samples", str(args.samples), "--members", str(args.members),
               "--blocks", str(args.blocks), "--devices", str(args.devices)] + (["--host"] if args.host else [])
    try:
        completed = subprocess.run(command, env=environment, cwd=Path(__file__).resolve().parents[1], timeout=args.timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        print(f"FAIL deadline {time.perf_counter() - started:.3f}s exceeded {args.timeout_seconds:g}s", flush=True)
        return 1
    elapsed = time.perf_counter() - started
    print(f"{'PASS' if completed.returncode == 0 else 'FAIL'} sweep_bench {elapsed:.3f}s", flush=True)
    return int(completed.returncode != 0)


if __name__ == "__main__":
    raise SystemExit(main())
