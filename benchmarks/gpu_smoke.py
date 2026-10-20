"""A bounded, pytest-free check of the full-data CUDA route, from a staged source tree.

    python -m benchmarks.gpu_smoke

The store is the mosaic fixture of test_full_data_fit, with its writer and generator shared with the tests.
Stage 0 is computed once per trait on the host; the two fits then see exactly the same statistics, lattice,
starts, sample order and seeds. No summary-band route is selected. The working set cuts a block into pieces
of PANEL + 7 members, so a piece crosses a warp boundary and later pieces cannot silently be omitted.

The fit comparisons are smoke tolerances, not claims of statistical accuracy: means and finite hyperparameters
use rtol=1e-4, atol=1e-6, and the EB fixed point's ELBO uses 1e-4 nats absolute. The ELBO is read after restore_best,
before the mode mixture (FullDataFit does not expose an ELBO for that mixture); the means compared are the returned
mixture's. Infinite smoothing weights must have identical signs and positions. Direct kernels and streamed reads
use rtol=1e-10 and atol=1e-10 times the reference's largest magnitude, as test_engine_kernels_cuda does. NaNs fail.

Observers only count completed calls and copy diagnostics: they never replace a numerical result, shorten the EB
loop or loosen its certificates. Each fit must take at least two accepted EB steps and complete curvature responses.
A five-minute subprocess deadline includes imports and cold compilation; a timeout is a FAIL, never a skipped check.
The small problem targets that budget on the A100 VMs; cold-cache runtime must be measured on the target machine.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
from unittest.mock import patch

# Scenario sizes and smoke thresholds belong to this benchmark, not to the inference package.
SAMPLES = 500
TRAINING = 400
DRAWS = 16
WORKSPACE_BYTES = 1 << 26
TIMEOUT_SECONDS = 300
FIT_RTOL = 1e-4
FIT_ATOL = 1e-6
ELBO_ATOL = 1e-4
KERNEL_RTOL = 1e-10


class Checks:
    """Report each check even after another failed, keeping asynchronous CUDA work inside its measured interval."""

    def __init__(self, synchronize=lambda: None):
        self.synchronize = synchronize
        self.failures = 0

    def run(self, name, work):
        started = time.perf_counter()
        try:
            result = work()
            self.synchronize()
        except Exception as error:
            self.failures += 1
            traceback.print_exc(file=sys.stderr)
            detail = " ".join(str(error).split())
            print(f"FAIL {name} {time.perf_counter() - started:.3f}s {type(error).__name__}: {detail}", flush=True)
            return None
        print(f"PASS {name} {time.perf_counter() - started:.3f}s", flush=True)
        return result


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _close(actual, expected, *, rtol=KERNEL_RTOL, atol=None):
    import numpy as np

    actual, expected = np.asarray(actual), np.asarray(expected)
    _require(actual.shape == expected.shape, f"shapes differ: {actual.shape}, {expected.shape}")
    _require(np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)), "non-finite comparison")
    if atol is None:
        atol = KERNEL_RTOL * float(np.max(np.abs(expected)))
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False)


def _smoothing_close(actual, expected):
    import numpy as np

    actual, expected = np.asarray(actual), np.asarray(expected)
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    _require(not np.isnan(actual).any() and not np.isnan(expected).any(), "NaN smoothing weight")
    if finite.any():
        _close(actual[finite], expected[finite], rtol=FIT_RTOL, atol=FIT_ATOL)


def _store(root):
    """test_full_data_fit's seed-7 quantitative store, with binary labels drawn from the same genetic signal."""
    import numpy as np
    from scipy.special import expit
    from benchmarks.gpu_smoke_fixtures import _write_store, mosaic_codes
    from sv_pgs.dosage_store import DosageStore

    generator = np.random.default_rng(7)
    codes = {name: mosaic_codes(generator, SAMPLES, count) for name, count in (("chr21", 180), ("chr22", 140))}
    milli = {name: np.rint(values.astype(np.float64) / 127.0 * 1000.0).astype(np.int64) for name, values in codes.items()}
    _write_store(root, [milli])
    store = DosageStore.open(root)
    dosage = store.read_codes(0, store.n_variants).astype(np.float64).T / 127.0
    causal = generator.choice(dosage.shape[1], size=15, replace=False)
    effects = np.zeros(dosage.shape[1])
    effects[causal] = generator.standard_normal(15)
    genetic = (dosage - dosage.mean(axis=0)) @ effects
    genetic *= np.sqrt(0.5) / np.std(genetic)
    covariate = generator.standard_normal(SAMPLES)
    quantitative = 0.3 * covariate + genetic + np.sqrt(0.5) * generator.standard_normal(SAMPLES)
    binary = (generator.random(SAMPLES) < expit(0.3 * covariate + 2.0 * genetic)).astype(np.float64)
    return store, np.column_stack([np.ones(TRAINING), covariate[:TRAINING]]), (quantitative[:TRAINING], binary[:TRAINING])


def _derivatives(cp):
    """The fused derivative entry point against its numba host twin, including the optional scale fields.

    As in test_engine_kernels_cuda, inspect all eight scalar fields and both density matrices; checking only the
    posterior mean misses the variance/precision derivatives that curvature's response consumes.
    """
    import numpy as np
    from scipy.special import log_softmax
    from sv_pgs import engine_kernels, scale_mixture_ep as engine

    rng = np.random.default_rng(51)
    rows, nodes = 257, 40
    grid = np.linspace(np.log(1e-5), np.log(0.5), nodes)
    density = log_softmax(rng.normal(size=nodes))
    scales = rng.uniform(-1.0, 1.0, rows)
    precision = rng.uniform(50.0, 400.0, rows)
    shift = rng.normal(size=rows) * np.sqrt(precision)
    for with_scale in (False, True):
        fields = np.empty((rows, len(engine._DERIVATIVE_FIELDS)))
        mean, second = np.empty((rows, nodes)), np.empty((rows, nodes))
        _require(not engine._derivative_rows(density, grid, scales, precision, shift, with_scale, fields, mean, second), "host cavity improper")
        got = engine_kernels.variant_derivatives(cp, density, scales, grid, precision, shift, with_scale)
        engine_kernels._raise_if_improper(cp.asnumpy(got[-1]))
        for column in range(len(engine._DERIVATIVE_FIELDS)):
            _close(cp.asnumpy(got[0][:, column]), fields[:, column])
        for actual, expected in zip(got[1:3], (mean, second), strict=True):
            _close(cp.asnumpy(actual), expected)


def _products(cp):
    """The same unaligned signed-code products as test_code_products_cuda, against a decoded numpy matrix."""
    import numpy as np
    from sv_pgs.code_products import CodeBlockTile

    rng = np.random.default_rng(7)
    variants, samples = 61, 5003
    codes = rng.integers(-127, 128, size=(variants, samples), dtype=np.int16).astype(np.int8)
    values = codes.astype(np.float64)
    means, scales = values.mean(axis=1), values.std(axis=1)
    dense = ((values - means[:, None]) / scales[:, None]).T
    right, left = rng.normal(size=(variants, 5)), rng.normal(size=(samples, 3))
    weights = rng.uniform(0.05, 0.25, samples)
    for workspace in (WORKSPACE_BYTES, 3_000_000):
        tile = CodeBlockTile(cp.asarray(codes), cp.asarray(means), cp.asarray(scales), cp, workspace)
        for actual, expected in (
            (tile.matmat(cp.asarray(right)), dense @ right),
            (tile.rmatmat(cp.asarray(left)), dense.T @ left),
            (tile.weighted_gram(cp.asarray(weights)), dense.T @ (weights[:, None] * dense)),
            (tile.weighted_column_squares(cp.asarray(weights[:, None])), (dense * dense).T @ weights[:, None]),
        ):
            _close(cp.asnumpy(actual), expected)
        operand = tile.sample_operand(cp.asarray(left), np.finfo(np.float64).eps / 2.0)
        _close(cp.asnumpy(tile.rmatmat(operand)), dense.T @ left)
        image = cp.zeros((samples, right.shape[1]))
        tile.accumulate_matmat(cp.asarray(right), image, np.finfo(np.float64).eps / 2.0)
        _close(cp.asnumpy(image), dense @ right)



def _sweep_panels(cp):
    """test_device_sweep_cuda's code-panel sweep against the host sweep on the same projected problem.

    The sharded fit's peer tiles use decoded panels, so this separate check also runs CodePanels' held-code
    projection and residual kernels on every visible device. Three sweeps reuse the panel Grams, including
    the last partial panel, and check every moment and both scalar ELBO pieces against the host sweep.
    """
    import numpy as np
    from sv_pgs.code_products import CodeBlockTile
    from sv_pgs.device_sweep import PANEL, CodePanels, PanelGrams, sweep_piece
    from sv_pgs.mean_field import _sweep as host_sweep

    rng = np.random.default_rng(11)
    samples, members, nodes = 160, 3 * PANEL + 5, 31
    codes = rng.integers(-60, 61, size=(members, samples)).astype(np.int8)
    means, scales = codes.mean(axis=1), codes.std(axis=1)
    dense = ((codes - means[:, None]) / scales[:, None]).T
    covariates = np.column_stack([np.ones(samples), rng.normal(size=(samples, 2))])
    solve = np.linalg.pinv(covariates.T @ covariates)
    projected = np.asfortranarray(dense - covariates @ (solve @ (covariates.T @ dense)))
    target = rng.normal(size=samples)
    target -= covariates @ (solve @ (covariates.T @ target))
    grid = np.linspace(np.log(1e-5), np.log(0.5), nodes)
    density = np.full((1, nodes), -np.log(nodes))
    log_variance = np.tile(grid, (members, 1))
    variance = np.exp(log_variance)
    squares = np.sum(projected * projected, axis=0)
    rows, classes = np.arange(members), np.zeros(members, dtype=np.int64)
    host = {name: np.zeros(members) for name in ("mean", "variance", "shift", "third", "fourth")}
    device = {name: cp.zeros(members) for name in host}
    host_residual, device_residual = target.copy(), cp.asarray(target)
    tile = CodeBlockTile(cp.asarray(codes), cp.asarray(means), cp.asarray(scales), cp, WORKSPACE_BYTES)
    panels = CodePanels(cp, tile, cp.asarray(rows), cp.ones(members))
    grams = PanelGrams()
    for _ in range(3):
        parts = host_sweep(projected, squares, rows, classes, density, variance, log_variance, 0.9,
                           host["mean"], host_residual, host["variance"], host["shift"], host["third"], host["fourth"], rows)
        pieces = cp.zeros((members, 3))
        sweep_piece(
            cp, decode=lambda first, last: cp.asarray(dense[:, first:last]), width=members, mask=cp.ones(samples),
            covariates=cp.asarray(covariates), covariate_pinv=cp.asarray(solve), residual=device_residual, grams=grams,
            model=0, members=rows, squares=cp.asarray(squares), class_index=cp.asarray(classes), log_density=cp.asarray(density),
            node_variance=cp.asarray(variance), log_node_variance=cp.asarray(log_variance), noise=0.9, pieces=pieces,
            panels=panels, **device,
        )
        for name in host:
            _close(cp.asnumpy(device[name]), host[name], rtol=1e-9, atol=1e-11 * (1.0 + np.abs(host[name]).max()))
        _close(cp.asnumpy(device_residual), host_residual)
        _close(cp.asnumpy(pieces.sum(axis=0))[:2], np.asarray(parts[:2]), rtol=1e-9, atol=1e-9 * parts[3])
    _require(len(grams._grams) == -(-members // PANEL), "the sweep did not retain every panel Gram")

def _resident_shards(source):
    from sv_pgs.device_plan import ShardedDualSource

    shards = source._shards if isinstance(source, ShardedDualSource) else [source.source]
    for shard in shards:
        _require(shard._resident_complete and shard._resident is not None, "a store shard did not become resident")
        _require(all(code is not None for code in shard._resident), "a resident shard has missing blocks")
        _require(all(int(code.device.id) == shard._budget.device_ids[0] for code in shard._resident),
                 "resident codes are on the wrong device")
    return shards


def _reads(source, store, statistics, cp, device_ids):
    """Read every block to exhaustion, then forbid disk reads while checking blocks and the sharded reduction.

    ``holds_codes`` also answers true for a cache that could be admitted: the completed cache itself is checked,
    and a second read must succeed with the store reader disabled. This catches the generator-exhaustion failure
    that leaves a sharded source streaming despite enough device memory to hold its codes.
    """
    import numpy as np
    from sv_pgs.device_plan import ShardedDualSource
    from sv_pgs.store_block_source import reduced_block_layout

    if len(device_ids) > 1:
        _require(isinstance(source, ShardedDualSource), "two visible devices did not select the sharded route")
        _require(set(source.plan.block_devices.tolist()) == set(range(len(device_ids))), "an assigned GPU has no blocks")
    rows, _indices, means, scales = reduced_block_layout(statistics)
    signed = np.concatenate([store.read_codes(int(row), int(row) + 1) for block in rows for row in block]).astype(np.float64) - 127.0
    dense = ((signed[:, :TRAINING] - np.concatenate(means)[:, None]) / np.concatenate(scales)[:, None]).T
    for _start, _stop, _tile in source.blocks():
        pass
    _resident_shards(source)
    rng = np.random.default_rng(47)
    left, right = rng.normal(size=(TRAINING, 3)), rng.uniform(0.1, 1.0, (source.variant_count, 3))

    def work(start, stop, tile, shared, rows, image):
        image += tile.matmat(rows["right"] * tile.rmatmat(shared["left"]))

    with patch("sv_pgs.store_block_source.StoreGenotypeBlockSource._iter_streamed", side_effect=AssertionError("resident read touched disk")):
        covered = []
        for start, stop, tile in source.blocks():
            covered.extend(range(start, stop))
            _close(cp.asnumpy(tile.columns(cp.arange(stop - start))), dense[:, start:stop])
            _close(cp.asnumpy(tile.rmatmat(cp.asarray(left))), dense[:, start:stop].T @ left)
        np.testing.assert_array_equal(covered, np.arange(source.variant_count))
        total = source.map_reduce(work, {"left": cp.asarray(left)}, {"right": cp.asarray(right)}, left.shape)
        _close(cp.asnumpy(total), dense @ (right * (dense.T @ left)))
    return dense


def _fit(source, statistics, prior, covariates, target, binary):
    """Observe the real fit without changing its results, retaining only host diagnostics after it returns."""
    import numpy as np
    from sv_pgs import full_data_fit as full, scale_mixture_ep as engine
    from sv_pgs.binary_likelihood import BernoulliSites
    from sv_pgs.device_sweep import PANEL
    from sv_pgs.dual_solve import DualGaussian

    sites = BernoulliSites.start(target) if binary else None
    response = target if sites is None else sites.response
    gaussian = DualGaussian(
        source=source, training=np.ones((TRAINING, 1)), targets=response[:, None], offsets=np.zeros((TRAINING, 1)),
        covariates=covariates, grams=full.block_grams(statistics), probe_count=DRAWS, seed=11,
        sample_weights=None if sites is None else sites.weights[:, None], probe_sample_indices=np.arange(TRAINING),
    )
    observed = {"curvature": 0, "responses": 0, "derivatives": 0, "sweeps": 0, "widths": []}
    curvature, derivatives, sweep, mixture = engine.curvature_correction, engine._variant_derivatives, full.sweep_piece, full._mode_mixture

    response_columns = engine._total_curvature_columns

    def counted_response(*args, **kwargs):
        _require(args[6].shape[1] > 0, "curvature was asked for no directions")
        result = response_columns(*args, **kwargs)
        observed["responses"] += 1
        return result

    def counted_curvature(*args, **kwargs):
        result = curvature(*args, **kwargs)
        observed["curvature"] += 1
        return result

    def counted_derivatives(*args, **kwargs):
        result = derivatives(*args, **kwargs)
        _require(engine._DEVICE.get() is source.array_module, "derivatives ran on the wrong array module")
        observed["derivatives"] += 1
        return result

    def counted_sweep(*args, **kwargs):
        result = sweep(*args, **kwargs)
        observed["sweeps"] += 1
        observed["widths"].append(kwargs["width"])
        return result

    def measured_mixture(main, *args, **kwargs):
        observed["elbo"] = main.elbo.copy()
        return mixture(main, *args, **kwargs)

    working_bytes = 2 * TRAINING * np.dtype(np.float64).itemsize * (PANEL + 7)
    # Residency is required throughout the fit, not just on the warm-up read. With this tiny store an eviction or
    # an incomplete shard read is a harness failure; neither route may quietly stream its way to an equal answer.
    resident_read = nullcontext() if source.array_module is np else patch(
        "sv_pgs.store_block_source.StoreGenotypeBlockSource._iter_streamed", side_effect=AssertionError("resident fit touched disk")
    )
    with resident_read, patch.object(engine, "curvature_correction", counted_curvature), patch.object(engine, "_variant_derivatives", counted_derivatives), \
            patch.object(full, "sweep_piece", counted_sweep), patch.object(full, "_mode_mixture", measured_mixture), \
            patch.object(engine, "_total_curvature_columns", counted_response):
        fit = full.fit_full_data(
            gaussian=gaussian, statistics=statistics, prior=prior, draw_count=DRAWS, working_bytes=working_bytes,
            seed=13, inference="mean_field", sites=[sites],
        )
    _require(int(fit.certificate.outer_iterations[0]) >= 2, f"fewer than two accepted EB steps: {fit.certificate.outer_iterations}")
    _require(observed["responses"] >= 2 and observed["derivatives"] >= 2, f"curvature coverage missing: {observed}")
    if source.array_module is not np:
        _require(observed["sweeps"] > 0 and max(observed["widths"]) > PANEL, "device panels were not swept")
        _require(max(observed["widths"]) <= PANEL + 7, "the piece working set was not respected")
        _resident_shards(source)
    print(f"INFO EB steps={fit.certificate.outer_iterations.tolist()} curvature={observed['curvature']} "
          f"responses={observed['responses']} derivatives={observed['derivatives']} device_pieces={observed['sweeps']} "
          f"certified={fit.certificate.outer_criterion_met.tolist()}", flush=True)
    return dict(mean=fit.member_mean.copy(), elbo=observed["elbo"], noise=fit.noise_variance.copy(),
                coefficients=fit.hyperparameters[0].coefficients.copy(), smoothing=fit.hyperparameters[0].log_smoothing.copy())


def _worker(root):
    import numpy as np
    from dataclasses import replace
    from threadpoolctl import threadpool_limits
    from sv_pgs.compute_budget import detect_compute_budget

    checks = Checks()

    def devices():
        import cupy as cp
        count = min(2, int(cp.cuda.runtime.getDeviceCount()))
        _require(count > 0, "no CUDA GPU visible; CPU fallback is not a smoke pass")
        budget = detect_compute_budget()
        _require(budget.device_kind == "cuda" and len(budget.device_ids) >= count, "budget omitted visible CUDA GPUs")
        budget = replace(budget, device_ids=budget.device_ids[:count], device_names=budget.device_names[:count],
                         device_bytes=budget.device_bytes[:count], device_compute_capabilities=budget.device_compute_capabilities[:count])
        for device_id in budget.device_ids:
            with cp.cuda.Device(device_id):
                _require(float(cp.arange(3, dtype=cp.float64).sum()) == 3.0, "CUDA compile/execute failed")
                cp.cuda.Device(device_id).synchronize()
        cp.cuda.Device(budget.device_ids[0]).use()
        return cp, budget

    detected = checks.run("CUDA devices", devices)
    if detected is None:
        return 1
    cp, budget = detected

    def synchronize():
        for device_id in budget.device_ids:
            cp.cuda.Device(device_id).synchronize()

    checks.synchronize = synchronize
    print(f"INFO GPUs={budget.device_ids} fit rtol={FIT_RTOL} atol={FIT_ATOL}; EB ELBO atol={ELBO_ATOL} nats; "
          f"kernels rtol={KERNEL_RTOL}, atol={KERNEL_RTOL} * reference max", flush=True)
    with threadpool_limits(limits=2):
        for device_id in budget.device_ids:
            with cp.cuda.Device(device_id):
                checks.run(f"variant_derivatives GPU {device_id}", lambda: _derivatives(cp))
                checks.run(f"code_products GPU {device_id}", lambda: _products(cp))
                checks.run(f"device_sweep code panels GPU {device_id}", lambda: _sweep_panels(cp))
        problem = checks.run("mosaic store", lambda: _store(root / "store"))
        if problem is None:
            return 1
        store, covariates, targets = problem
        host_budget = replace(budget, device_kind="cpu", device_ids=(), device_names=(), device_bytes=(), device_compute_capabilities=(), cpu_threads=2)
        from sv_pgs.binary_likelihood import BernoulliSites
        from sv_pgs.config import ModelConfig
        from sv_pgs.device_sweep import PANEL
        from sv_pgs.genotype_statistics import DosageStoreTileSource, compute_genotype_statistics
        from sv_pgs.memory_broker import memory_scope
        from sv_pgs.scale_mixture_ep import scale_mixture_prior
        from sv_pgs.stage2_wiring import stage2_source

        for label, target, binary in zip(("quantitative", "binary"), targets, (False, True), strict=True):
            def prepare():
                response = BernoulliSites.start(target).response if binary else target
                statistics = compute_genotype_statistics(
                    DosageStoreTileSource(store, np.arange(store.n_variants)), np.arange(TRAINING), covariates, response[:, None],
                    ModelConfig(), host_budget, 3 * PANEL + 7, root / label,
                )
                widths = [statistics.ld.block(index).reduced_columns.size for index in range(statistics.ld.block_count)]
                _require(len(widths) >= 3 and sum(width > PANEL + 7 for width in widths) >= 2,
                         f"not enough blocks spanning multiple pieces: {widths}")
                members = statistics.active_rows.size
                # A fixed small lattice, as in the kernel fixtures: smoke both routes on one finite model, without
                # adding the public route's adaptive relattice searches to this wiring check.
                nodes = np.linspace(np.log(1e-5), np.log(0.5), 17)
                prior = scale_mixture_prior(
                    class_index=np.zeros(members, dtype=np.int64), log_variance_offset=np.zeros(members),
                    annotation_design=np.zeros((members, 0)), annotation_groups=(), nodes=nodes, floor=nodes[0] - 1.0, top=nodes[-1],
                )
                print(f"INFO {label} LD block widths={widths}; piece cap={PANEL + 7}; nodes={nodes.size}", flush=True)
                return statistics, prior

            prepared = checks.run(f"{label} Stage 0 and piece coverage", prepare)
            if prepared is None:
                continue
            statistics, prior = prepared
            with memory_scope(budget):
                source = checks.run(f"{label} CUDA source", lambda: stage2_source(store, statistics, budget, WORKSPACE_BYTES, np.arange(TRAINING)))
                if source is not None:
                    checks.run(f"{label} resident blocks/map_reduce ({len(budget.device_ids)} GPUs)",
                               lambda: _reads(source, store, statistics, cp, budget.device_ids))
                    gpu = checks.run(f"{label} CUDA full-data fit", lambda: _fit(source, statistics, prior, covariates, target, binary))
                else:
                    gpu = None
                del source
                gc.collect()
            source = stage2_source(store, statistics, host_budget, WORKSPACE_BYTES, np.arange(TRAINING))
            cpu = checks.run(f"{label} CPU full-data fit", lambda: _fit(source, statistics, prior, covariates, target, binary))
            for field in ("mean", "elbo", "noise", "coefficients", "smoothing"):
                def compare(field=field):
                    _require(gpu is not None and cpu is not None, "a prerequisite fit failed")
                    if field == "smoothing":
                        _smoothing_close(gpu[field], cpu[field])
                    else:
                        _close(gpu[field], cpu[field], rtol=0.0 if field == "elbo" else FIT_RTOL,
                               atol=ELBO_ATOL if field == "elbo" else FIT_ATOL)
                checks.run(f"{label} CPU/CUDA {field}", compare)
            del source
            gc.collect()
    return int(checks.failures > 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timeout-seconds", type=float, default=TIMEOUT_SECONDS, help="whole command deadline, including cold compilation (default: 300)")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker is not None:
        try:
            return _worker(args.worker)
        except Exception as error:
            traceback.print_exc()
            print(f"FAIL harness {type(error).__name__}: {' '.join(str(error).split())}", flush=True)
            return 1
    if not 0 < args.timeout_seconds < float("inf"):
        parser.error("--timeout-seconds must be positive and finite")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="svpgs-gpu-smoke-") as root:
        environment = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMBA_NUM_THREADS="2", PYTHONUNBUFFERED="1")
        try:
            completed = subprocess.run([sys.executable, "-m", "benchmarks.gpu_smoke", "--worker", root],
                                       env=environment, timeout=args.timeout_seconds, check=False)
        except subprocess.TimeoutExpired:
            print(f"FAIL deadline {time.perf_counter() - started:.3f}s exceeded {args.timeout_seconds:g}s", flush=True)
            return 1
    elapsed = time.perf_counter() - started
    print(f"{'PASS' if completed.returncode == 0 else 'FAIL'} gpu_smoke {elapsed:.3f}s", flush=True)
    return int(completed.returncode != 0)


if __name__ == "__main__":
    raise SystemExit(main())
