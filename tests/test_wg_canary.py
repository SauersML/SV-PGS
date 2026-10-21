"""The whole-genome canary's small deterministic input and its failure contracts, without remote services.

The ordinary tests generate only a few LD blocks and run the production pipeline through measurement on the CPU;
they target less than a minute together on the supported Linux VM. The separately marked tests in
test_wg_canary_integration.py run real EB, lattice checks and mode mixtures, so their convergence time is not hidden inside that unit-test runtime claim. No test calls the Batch launcher.
"""
from __future__ import annotations

import dataclasses
import gzip
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import wg_canary as canary
from sv_pgs import workspace_pipeline as pipeline
from sv_pgs.compute_budget import ComputeBudget

BUDGET = ComputeBudget("cpu", (), (), (), (), 1 << 29, 1)
TINY = canary.Scenario(columns=64, samples=128, chromosomes=2, block_columns=16, sv_every=4)


def test_generator_bytes_are_deterministic_and_the_seed_changes_dosages(tmp_path):
    """The same fixture in different directories has identical inputs, including compressed VCF headers."""
    first = canary.generate(tmp_path / "a" / "inputs", TINY)
    second = canary.generate(tmp_path / "b" / "inputs", TINY)
    assert first.cross_fit is False and first.fold_count == 5
    for path in first.crosswalk.parent.iterdir():
        assert path.read_bytes() == (second.crosswalk.parent / path.name).read_bytes()
    codes = canary.mosaic_block(TINY, 0, 0, 16)
    np.testing.assert_array_equal(codes, canary.mosaic_block(TINY, 0, 0, 16))
    assert not np.array_equal(codes, canary.mosaic_block(dataclasses.replace(TINY, seed=TINY.seed + 1), 0, 0, 16))
    assert codes.shape == (16, TINY.samples) and np.all((codes >= 0) & (codes <= 2))
    # Quantization must not collapse these variant columns into founder duplicates.
    assert np.unique(np.rint(codes * 127).astype(np.uint8), axis=0).shape[0] == codes.shape[0]


def test_full_size_plan_keeps_each_column_and_chromosome_without_generating_it():
    scenario = canary.Scenario()
    counts = scenario.counts(scenario.columns)
    assert len(counts) == 22 and sum(counts) == 1_060_000 and min(counts) > 0
    assert counts[0] > counts[-1]
    assert len(canary.SV_TYPES) == len(set(canary.SV_TYPES)) == 8
    assert "CNV" not in canary.SV_TYPES  # The ALT-dosage converter correctly refuses copy-number units.


@pytest.mark.parametrize("changes", [{"records": 32}, {"samples": 31}, {"chromosomes": 23}, {"block_columns": 1}, {"sv_every": 1}])
def test_invalid_scenario_fails_before_any_write(changes):
    with pytest.raises(ValueError):
        dataclasses.replace(TINY, **changes)


def test_converter_measurement_filter_annotations_and_family_split_are_real(tmp_path):
    """No synthetic store or model is injected: the VCF decoder, store writer, measurement model and family splitter
    consume the same generator the million-column command uses. Extra records traverse the converter but are filtered
    by their reported reliability; every structural class supported by this input route is represented."""
    scenario = dataclasses.replace(TINY, columns=192, records=208)
    config = canary.generate(tmp_path / "inputs", scenario)
    bindings = dataclasses.replace(pipeline.workspace_bindings(), bigquery_client=canary.no_remote)
    with canary.instrumentation(scenario, {}):
        summaries = pipeline.run_pipeline(config, bindings, BUDGET, through="direct_sv")
    assert "measurement" in summaries
    run = pipeline._Run(config, bindings, BUDGET, {})
    cohort = pipeline._read_cohort(run)
    assert len(cohort.model_names) == 1
    # No family crosses the training/held-out boundary.
    samples = pipeline._read_table(config.run_directory / "samples" / "rows.tsv")
    family_folds = {}
    for row in samples:
        family_folds.setdefault(row["family"], set()).add(row["fold"])
    assert all(len(folds) == 1 for folds in family_folds.values())
    model = bindings.load_measurement_model(pipeline._final_measurement_model(run))
    with pipeline.DosageStore.open(pipeline._final_store(run)) as store:
        assert store.n_variants == scenario.records
        assert canary.ANNOTATIONS <= set(store.variant_table.annotations)
        assert set(store.variant_table.annotation_legends["cx"]) == {"simple", "nested", "complex"}
        assert len(np.unique(store.variant_table.variant_class)) == 9
        assert np.max(store.variant_table.annotations["sv_length"]) > 50
        selected = pipeline.fit_filtered(store.variant_table, store.n_samples, model.log_reliability, 0, .1)
        assert np.isfinite(selected).sum() == scenario.columns
        short = pipeline.arm_log_reliability(store.variant_table, selected, "snv_indel")
        assert 0 < np.isfinite(short).sum() < scenario.columns
        assert store.read_codes(0, 8, np.arange(scenario.samples)).shape == (8, scenario.samples)
    finished = json.loads((config.run_directory / "store" / pipeline.COMPLETE_FILE).read_text())["finished"]
    assert not finished.endswith("Z") and ("-" in finished or "+" in finished)


def test_extra_record_axis_is_explicit_in_the_generated_input(tmp_path):
    scenario = dataclasses.replace(TINY, records=80)
    config = canary.generate(tmp_path / "inputs", scenario)
    rows = []
    for chromosome in config.chromosomes:
        with gzip.open(config.crosswalk.parent / f"{chromosome}.vcf.gz", "rt") as handle:
            rows.extend(line for line in handle if not line.startswith("#"))
    assert len(rows) == 80
    assert sum(";R2=0.95" in row for row in rows) == 64


def test_mixed_budget_preserves_column_reserve_and_refuses_overcommit():
    detected = dataclasses.replace(BUDGET, host_bytes=240 * canary.GIB, device_kind="cuda", device_ids=(0, 1),
                                   device_names=("fixture", "fixture"), device_bytes=(40 * canary.GIB,) * 2,
                                   device_compute_capabilities=((8, 0),) * 2)
    budget = canary.planned_budget(detected, 512, None, None)
    assert budget.host_bytes == int((24 + 136 * 512 / 50_000) * canary.GIB)
    assert budget.device_bytes == (int((8 + 32 * 512 / 50_000) * canary.GIB),) * 2
    assert budget.device_ids == (0, 1)
    with pytest.raises(MemoryError):
        canary.planned_budget(BUDGET, 512, None, None)
    assert canary.planned_budget(BUDGET, 512, .25, .25).host_bytes == canary.GIB // 4


@pytest.mark.parametrize("metric", ["rss", "python_live", "device0_live", "ledger_host", "ledger_leases"])
def test_sustained_growth_is_a_hard_failure(metric):
    guard = canary.GrowthGuard(window=6, warmup=2, tolerance=1)
    with pytest.raises(canary.CanaryFailure, match=metric):
        for call in range(20):
            guard.observe({metric: call * 10})
    assert not issubclass(canary.CanaryFailure, FloatingPointError)


def test_a_lease_a_call_fails_and_a_working_set_that_moves_by_a_few_passes():
    """One lease leaked per call raises each third's floor by its width; the outer loop's kept snapshots and latest
    warm starts rise by one now and then (the two-arm fixture's last loop: 26, 27, 28 over twelve calls)."""
    guard = canary.GrowthGuard(window=12, warmup=0)
    for value in (26, 27, 27, 26, 27, 27, 27, 27, 28, 28, 29, 28, 28, 29, 29, 28):
        guard.observe({"ledger_leases": value})
    leaking = canary.GrowthGuard(window=12, warmup=0)
    with pytest.raises(canary.CanaryFailure, match="ledger_leases"):
        for call in range(12):
            leaking.observe({"ledger_leases": 26 + call})


def test_growth_guard_allows_a_cache_warmup_and_a_transient_trial():
    guard = canary.GrowthGuard(window=6, warmup=2, tolerance=1)
    for value in (0, 1, 100, 100, 1000, 100, 100, 100, 100, 100, 100):
        guard.observe({"rss": value, "ledger_leases": 2})
    assert len(guard.history) == 6


def test_process_tree_peak_counts_workers_and_tolerates_an_exited_child(tmp_path):
    for pid, children, kib in ((100, "200 300", 4096), (200, "", 2048)):
        (tmp_path / str(pid) / "task" / str(pid)).mkdir(parents=True)
        (tmp_path / str(pid) / "status").write_text(f"VmRSS:\t{kib} kB\n")
        (tmp_path / str(pid) / "task" / str(pid) / "children").write_text(children)
    assert canary.process_tree_rss(100, tmp_path) == (4096 + 2048) * 1024


def _partial_arms(root, key_b="same"):
    for step in pipeline.STEP_NAMES[:pipeline.STEP_NAMES.index("fit")]:
        (root / "prepare" / step).mkdir(parents=True)
    for arm, key in zip(pipeline.ARMS, ("same", key_b)):
        partial = root / arm / "fit.partial"
        models = partial / "models" / arm
        models.mkdir(parents=True)
        (partial / pipeline.KEY_FILE).write_text(key)
        (models / "payload").write_text(arm)


def test_merge_copies_each_whole_arm_and_leaves_sources_untouched(tmp_path):
    _partial_arms(tmp_path)
    canary.merge_arms(tmp_path)
    for arm in pipeline.ARMS:
        merged = tmp_path / "merge" / "fit.partial" / "models" / arm / "payload"
        assert merged.read_text() == arm
        merged.write_text("changed copy")
        assert (tmp_path / arm / "fit.partial" / "models" / arm / "payload").read_text() == arm
    assert (tmp_path / "merge" / "store").is_symlink()


def test_mismatched_arm_keys_fail_before_finalization(tmp_path):
    _partial_arms(tmp_path, key_b="different")
    with pytest.raises(canary.CanaryFailure, match="different fit keys"):
        canary.merge_arms(tmp_path)


def test_null_or_uncertified_fit_cannot_pass_canary(tmp_path, monkeypatch):
    monkeypatch.setattr(canary, "acceptance", lambda model: {"accepted": False, "issues": [{"severity": "fatal", "code": "outer_criterion_met"}]})
    config = SimpleNamespace(run_directory=tmp_path, diseases=(), traits=TINY.traits)
    bindings = SimpleNamespace(load_model=lambda path: object())
    with pytest.raises(canary.CanaryFailure, match="uncertified"):
        canary.validate_models(config, bindings, TINY)


def test_no_remote_and_no_refit_are_hard_failures():
    with pytest.raises(RuntimeError, match="network"):
        canary.no_remote()
    with pytest.raises(RuntimeError, match="without another fit"):
        canary.no_fit()



def test_nonfinite_accepted_step_fails_and_hooks_are_restored():
    """A step accepted outside every basin (EB's design: gain nan, no state) passes; a broken step or a line the canary
    cannot read any more fails, and the hooks and the environment are restored either way."""
    original = canary.scale_mixture_ep.log
    workspace = {name: os.environ.get(name) for name in canary.SYNTHETIC_WORKSPACE}
    line = "eb outer: model 0 inner step accepted: length {} (radius 4.57e+03), decrement {} {}, gain {}, state error {} (decrement {})"
    outside = line.format("0.536", "inf", "indefinite", "nan", "inf", "inf")  # the first CPU run's line [sim]
    assert canary.accepted_step_problem(outside) is None
    assert canary.accepted_step_problem(line.format("3.55e+03", "0.00551", "definite", "0.00678", "0.00148", "0.00139")) is None
    for broken, problem in (
        (line.format("nan", "0.0055", "definite", "0.0068", "0.0015", "0.0014"), "length or radius"),
        (line.format("3.5", "inf", "definite", "0.0068", "0.0015", "0.0014"), "definite model"),
        (line.format("3.5", "0.0055", "definite", "0.0068", "nan", "0.0014"), "state error"),
        (line.format("3.5", "0.0055", "definite", "inf", "0.0015", "0.0014"), "nonfinite gain"),
        ("eb outer: model 0 inner step accepted: gain nan, state error inf", "no longer reads"),
    ):
        assert problem in canary.accepted_step_problem(broken)
    with pytest.raises(canary.CanaryFailure, match="nonfinite"), canary.instrumentation(TINY, {}):
        assert {name: os.environ[name] for name in canary.SYNTHETIC_WORKSPACE} == canary.SYNTHETIC_WORKSPACE
        canary.scale_mixture_ep.log(outside)
        canary.scale_mixture_ep.log(line.format("inf", "0.0055", "definite", "0.0068", "0.0015", "0.0014"))
    assert canary.scale_mixture_ep.log is original
    assert {name: os.environ.get(name) for name in canary.SYNTHETIC_WORKSPACE} == workspace


@pytest.mark.parametrize("failure", ["exit", "deadline"])
def test_supervisor_never_reports_a_failed_or_timed_out_process_as_complete(tmp_path, failure):
    command = [sys.executable, "-c", "raise SystemExit(3)" if failure == "exit" else "import time; time.sleep(60)"]
    metrics = {"passed": False}
    expected = subprocess.CalledProcessError if failure == "exit" else canary.CanaryFailure
    with (tmp_path / "progress.log").open("w") as handle, pytest.raises(expected):
        canary.supervise(command, {}, handle, .05 if failure == "deadline" else 5, metrics)
    assert metrics["passed"] is False and metrics["seconds"] > 0
    assert metrics["peak_process_tree_rss_bytes"] >= 0
