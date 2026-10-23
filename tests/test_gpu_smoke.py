"""The smoke command's failures must reach Batch, and sharing its fixtures must preserve the tests' inputs.

The end-to-end CUDA checks live in test_gpu_smoke_cuda: these host tests cover the reporting and deadline around
that command, where a swallowed exception or a NaN accepted as equal would turn a failed GPU run into a pass.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import gpu_smoke as smoke


@pytest.mark.parametrize("actual,expected", [(np.nan, np.nan), (np.inf, np.inf), (1.01, 1.0)])
def test_nonfinite_or_changed_results_cannot_pass(actual, expected):
    with pytest.raises(AssertionError):
        smoke._close(np.array([actual]), np.array([expected]))


def test_smoothing_edges_must_match_in_position_and_sign():
    smoke._smoothing_close(np.array([np.inf, -np.inf, 0.0]), np.array([np.inf, -np.inf, 0.0]))
    for changed in (np.array([-np.inf, -np.inf, 0.0]), np.array([np.inf, 0.0, -np.inf]), np.array([np.inf, -np.inf, np.nan])):
        with pytest.raises(AssertionError):
            smoke._smoothing_close(changed, np.array([np.inf, -np.inf, 0.0]))


def test_an_asynchronous_device_failure_is_reported_and_later_checks_run(capsys):
    def failed():
        raise RuntimeError("device synchronization failed")

    checks = smoke.Checks(failed)
    assert checks.run("first", lambda: 1) is None
    checks.synchronize = lambda: None
    assert checks.run("second", lambda: 2) == 2
    assert checks.failures == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("FAIL first ") and lines[1].startswith("PASS second ")


@pytest.mark.parametrize("outcome", [0, 1, -11, "timeout"])
def test_the_parent_propagates_exit_status_and_deadline(monkeypatch, capsys, outcome):
    def run(command, **kwargs):
        assert command[1:3] == ["-m", "benchmarks.gpu_smoke"]
        assert kwargs["timeout"] == smoke.TIMEOUT_SECONDS
        assert Path(command[-1]).is_dir()
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return SimpleNamespace(returncode=outcome)

    monkeypatch.setattr(smoke.subprocess, "run", run)
    assert smoke.main([]) == int(outcome != 0)
    assert ("PASS gpu_smoke" if outcome == 0 else "FAIL") in capsys.readouterr().out


def test_the_shared_mosaic_store_preserves_the_full_data_fixture(tmp_path):
    from tests.test_full_data_fit import _store

    expected, covariate, target, _genetic = _store(tmp_path / "expected", 7)
    actual, covariates, (quantitative, binary) = smoke._store(tmp_path / "actual")
    np.testing.assert_array_equal(actual.read_codes(0, actual.n_variants), expected.read_codes(0, expected.n_variants))
    np.testing.assert_array_equal(covariates[:, 1], covariate[:smoke.TRAINING])
    np.testing.assert_array_equal(quantitative, target[:smoke.TRAINING])
    assert set(np.unique(binary)) == {0.0, 1.0}


@pytest.mark.parametrize("devices", [1, 2])
def test_batch_spec_uses_staged_offline_runtime_and_no_retries(devices):
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PROJECT_ID="smoke-project", REGION="us-central1", OPS_BUCKET="smoke-ops", RUN_ID="smoke",
                       SERVICE_ACCOUNT="smoke@example.invalid", NETWORK="default", SUBNETWORK="default", GPU_COUNT=str(devices))
    environment.pop("MACHINE_TYPE", None)
    environment.pop("JOB_ID", None)
    environment.pop("STEPS", None)
    spec = json.loads(subprocess.run(["bash", "benchmarks/gpu_smoke.sh", "spec"], cwd=root, env=environment,
                                    capture_output=True, text=True, check=True).stdout)
    task = spec["taskGroups"][0]["taskSpec"]
    assert task["maxRetryCount"] == 0
    assert task["environment"]["variables"]["OPS"] == "/mnt/disks/ops/svpgs/smoke"
    assert '"$OPS/source/benchmarks/gpu_smoke.sh" task' in task["runnables"][0]["script"]["text"]
    # The workspace cannot read Cloud Logging: the task copies its output to smoke/<job>.log on the writable ops mount.
    assert task["environment"]["variables"]["JOB"].startswith("svpgs-gpu-smoke-")
    assert task["environment"]["variables"]["STEPS"] == "smoke bench tests"
    assert "-o ro" not in task["volumes"][0]["mountOptions"]
    policy = spec["allocationPolicy"]["instances"][0]["policy"]
    assert policy["accelerators"][0]["count"] == devices
    assert policy["machineType"] == f"a2-highgpu-{devices}g"


def test_batch_spec_takes_the_sweep_benchmark_alone_and_refuses_an_unknown_step():
    """STEPS=bench on one GPU is the sweep benchmark's own cheap job, STEPS=stage0 the Stage 0 benchmark's; a misspelt step
    is refused before any submission."""
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PROJECT_ID="smoke-project", REGION="us-central1", OPS_BUCKET="smoke-ops", RUN_ID="smoke",
                       SERVICE_ACCOUNT="smoke@example.invalid", NETWORK="default", SUBNETWORK="default", GPU_COUNT="1", STEPS="bench")
    environment.pop("MACHINE_TYPE", None)
    spec = json.loads(subprocess.run(["bash", "benchmarks/gpu_smoke.sh", "spec"], cwd=root, env=environment,
                                    capture_output=True, text=True, check=True).stdout)
    assert spec["taskGroups"][0]["taskSpec"]["environment"]["variables"]["STEPS"] == "bench"
    assert spec["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "a2-highgpu-1g"
    stage0 = json.loads(subprocess.run(["bash", "benchmarks/gpu_smoke.sh", "spec"], cwd=root, env=dict(environment, STEPS="stage0"),
                                      capture_output=True, text=True, check=True).stdout)
    assert stage0["taskGroups"][0]["taskSpec"]["environment"]["variables"]["STEPS"] == "stage0"
    refused = subprocess.run(["bash", "benchmarks/gpu_smoke.sh", "spec"], cwd=root, env=dict(environment, STEPS="smoke benchmark"),
                             capture_output=True, text=True)
    assert refused.returncode == 2 and "not benchmark" in refused.stderr


def test_batch_spec_runs_the_restore_benchmark_on_the_fit_jobs_disks():
    """STEPS=restore makes, restores and removes a synthetic mirror under the run's ops prefix, on the disk type and
    provisioning given (the fit jobs' four local SSDs on Spot); the defaults stay pd-balanced and STANDARD."""
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PROJECT_ID="smoke-project", REGION="us-central1", OPS_BUCKET="smoke-ops", RUN_ID="smoke",
                       SERVICE_ACCOUNT="smoke@example.invalid", NETWORK="default", SUBNETWORK="default", STEPS="restore seeds",
                       RUN_DISK_TYPE="local-ssd", RUN_DISK_GB="1500", PROVISIONING_MODEL="SPOT")
    environment.pop("MACHINE_TYPE", None)
    spec = json.loads(subprocess.run(["bash", "benchmarks/gpu_smoke.sh", "spec"], cwd=root, env=environment,
                                    capture_output=True, text=True, check=True).stdout)
    variables = spec["taskGroups"][0]["taskSpec"]["environment"]["variables"]
    assert variables["STEPS"] == "restore seeds" and variables["BENCH_MIRROR"] == "gs://smoke-ops/svpgs/smoke/bench_mirror"
    policy = spec["allocationPolicy"]["instances"][0]["policy"]
    assert policy["disks"][0]["newDisk"] == {"type": "local-ssd", "sizeGb": 1500} and policy["provisioningModel"] == "SPOT"
    for name in ("RUN_DISK_TYPE", "RUN_DISK_GB", "PROVISIONING_MODEL"):
        environment.pop(name)
    policy = json.loads(subprocess.run(["bash", "benchmarks/gpu_smoke.sh", "spec"], cwd=root, env=environment,
                                      capture_output=True, text=True, check=True).stdout)["allocationPolicy"]["instances"][0]["policy"]
    assert policy["disks"][0]["newDisk"] == {"type": "pd-balanced", "sizeGb": 100} and policy["provisioningModel"] == "STANDARD"


def test_the_restore_benchmark_s_synthetic_stream_repeats_its_block_and_seeks_like_a_file():
    from benchmarks.restore_bench import _Synthetic

    stream = _Synthetic(b"abcde", 12)
    assert stream.read(3) == b"abc" and stream.read(4) == b"deab" and stream.read() == b"cdeab" and stream.read() == b""
    assert stream.seek(-4, os.SEEK_END) == 8 and stream.read(2) == b"de" and stream.tell() == 10
    assert stream.seek(0) == 0 and stream.read(100) == b"abcdeabcdeab"


def test_the_seeds_benchmark_passes_on_the_host(capsys):
    from benchmarks import restore_bench

    assert restore_bench.seeds(members=3000, nodes=16, components=3) == 0
    out = capsys.readouterr().out
    assert "PASS 3 components read at once are the bits read one at a time" in out and "RESULT seeds 3000 members" in out
