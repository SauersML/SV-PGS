"""The EB-work benchmark's bookkeeping: every sweep reaches a phase, the report compares, failures reach Batch.

The benchmark itself runs on a GPU (benchmarks/eb_sweeps.sh); these host tests run its worker on a store small enough
for the test suite, so a sweep the ledger misses or misfiles, or a report ``compare`` cannot read, fails here first.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import eb_sweeps


def test_every_sweep_reaches_a_phase_and_compare_reads_the_report(tmp_path: Path) -> None:
    from sv_pgs.full_data_fit import _FullDataMeanField

    arguments = SimpleNamespace(
        root=tmp_path, samples=600, variants=400, causal=40, heritability=0.5, founders=6, segment=[10, 30], flip=0.05, seed=4,
        label="tiny", draw_count=16, route="samples", traits=3,
    )
    sweep = _FullDataMeanField._sweep
    report = eb_sweeps._worker(arguments)
    # The ledger puts the engine's methods back.
    assert _FullDataMeanField._sweep is sweep
    assert report["total_sweeps"] == sum(report["sweeps"].values()) > 0
    assert set(report["sweeps"]) <= {f"eb:{stage}:{role}" for stage in ("prior", "base", "annotated", "continued") for role in ("first", "carried", "cold")} | {"mixture", "other"}
    # The frequency annotation's groups give the nested EB both stages, and the three traits are one group.
    assert [run["stage"] for run in report["eb_runs"]][:2] == ["base", "annotated"] and report["eb_runs"][0]["models"] == 3
    assert sum(run["sweeps"] for run in report["eb_runs"]) == sum(count for phase, count in report["sweeps"].items() if phase.startswith("eb:"))
    assert report["sweeps"].get("mixture", 0) > 0 and report["sweeps"].get("other", 0) == 0
    assert len(report["models"]) == 3 and all(np.isfinite(model["held_out_partial_r2"]) for model in report["models"])
    lines = eb_sweeps._compare(report, json.loads(json.dumps(report)))
    assert "(1.00x fewer)" in lines[-5] and all("effects correlation 1.000000, relative distance 0" in line for line in lines[-3:])


def test_a_device_s_batched_pass_counts_one_sweep_per_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a device the models of a solve round are swept in one batched pass (``_panel_sweeps``), never through
    ``_sweep``: the ledger counts each model's share, and nothing for a pass that fell back to the decoded route."""
    import contextlib

    from sv_pgs.full_data_fit import _FullDataMeanField

    answers = [{0: (0.0, 0.0, 0.0, 0.0), 1: (0.0, 0.0, 0.0, 0.0)}, None]
    monkeypatch.setattr(_FullDataMeanField, "_panel_sweeps", lambda self, requests, kernels=None: answers.pop(0), raising=False)
    ledger = eb_sweeps._Ledger()
    with contextlib.ExitStack() as stack:
        ledger.install(stack)
        ledger.phase.append("mixture")
        assert _FullDataMeanField._panel_sweeps(None, {0: None, 1: None}) is not None
        assert _FullDataMeanField._panel_sweeps(None, {0: None, 1: None}) is None
    assert ledger.sweeps == {"mixture": 2}


@pytest.mark.parametrize("outcome", [0, 1, "timeout"])
def test_the_parent_propagates_exit_status_and_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], outcome) -> None:
    out = tmp_path / "report.json"

    def run(command, **keywords):
        assert command[1:4] == ["-m", "benchmarks.eb_sweeps", "run"] and "--worker" in command
        assert keywords["timeout"] == eb_sweeps.DEADLINE_SECONDS
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, keywords["timeout"])
        out.write_text(json.dumps({"total_sweeps": 7, "sweeps": {"mixture": 7}}))
        return SimpleNamespace(returncode=outcome)

    monkeypatch.setattr(eb_sweeps.subprocess, "run", run)
    assert eb_sweeps.main(["run", "--out", str(out)]) == int(outcome != 0)
    assert ("PASS eb_sweeps" if outcome == 0 else "FAIL eb_sweeps") in capsys.readouterr().out


def test_the_batch_spec_runs_one_a100_without_retries() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PROJECT_ID="eb-project", REGION="us-central1", OPS_BUCKET="eb-ops", RUN_ID="eb", SERVICE_ACCOUNT="eb@example.invalid",
                       NETWORK="default", SUBNETWORK="default", LABEL="after", EB_SWEEPS_ARGS="--samples 20000")
    for name in ("MACHINE_TYPE", "JOB_ID", "GPU_COUNT"):
        environment.pop(name, None)
    spec = json.loads(subprocess.run(["bash", "benchmarks/eb_sweeps.sh", "spec"], cwd=root, env=environment, capture_output=True, text=True, check=True).stdout)
    (group,) = spec["taskGroups"]
    task = group["taskSpec"]
    policy = spec["allocationPolicy"]["instances"][0]["policy"]
    assert policy["machineType"] == "a2-highgpu-1g" and policy["accelerators"] == [{"type": "nvidia-tesla-a100", "count": 1}]
    assert task["maxRetryCount"] == 0 and task["runnables"][0]["script"]["text"] == 'bash "$OPS/source/benchmarks/eb_sweeps.sh" task'
    variables = task["environment"]["variables"]
    assert variables["OPS"] == "/mnt/disks/ops/svpgs/eb" and variables["LABEL"] == "after" and variables["EB_SWEEPS_ARGS"] == "--samples 20000"
