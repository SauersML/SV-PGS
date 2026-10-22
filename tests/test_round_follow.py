"""round.py --follow left to run a round alone (launcher/workspace/round.py's Follower, watch and Lease, and
launcher/workspace/follow.sh), against the fixture repository and fake gcloud of tests/test_round.py, never a cloud
account: why a failed job failed; which failures it resubmits, on which slot, how often, and which it never does; the
spend cap; a time cap resubmitted only after progress; the round's end, its STOP object and its one follower; and the
follower started detached, stopped, and started again after a restart."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from launcher.workspace import round as planner
from tests.test_round import BUCKET, LAUNCHER, _bucket, _repository, _spec, cloud  # noqa: F401 - cloud is a fixture

MARK = "20261022T120000Z"
"""A marker time after every fixture job's creation (CREATED)."""
CREATED = "2026-10-22T11:00:00Z"


def _job(state: str, exits=(), hours: float = 0.0, machine: str = "a2-highgpu-2g") -> tuple:
    """A job's record as gcloud batch jobs describe gives it, and its one task's exits as gcloud batch tasks list does."""
    record = {
        "createTime": CREATED, "status": {
            "state": state, "runDuration": f"{hours * 3600:.0f}s",
            "statusEvents": [{"eventTime": "2026-10-22T11:05:00Z", "description": "Job state is set from SCHEDULED to RUNNING"}],
        },
        "allocationPolicy": {"instances": [{"policy": {"machineType": machine, "provisioningModel": "SPOT"}}]},
    }
    tasks = [{"status": {"statusEvents": [{"taskExecution": {"exitCode": code}} for code in exits]}}]
    return record, tasks


def _set(fake: Path, name: str, state: str, exits=(), hours: float = 0.0) -> None:
    record, tasks = _job(state, exits, hours)
    (fake / "jobs").mkdir(exist_ok=True)
    (fake / "jobs" / name).write_text(json.dumps(record))
    (fake / "jobs" / f"{name}.tasks").write_text(json.dumps(tasks))


def _mark(fake: Path, run: str, *events: str) -> None:
    markers = fake / "gcs" / BUCKET / "svpgs" / run / "markers"
    markers.mkdir(parents=True, exist_ok=True)
    for event in events:
        (markers / f"{MARK}_{event}").write_text("")


def _submitted(fake: Path) -> list:
    path = fake / "submitted.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _round(tmp_path: Path, **changes):
    repository, commit = _repository(tmp_path / "repo", partner=True)
    fake = _bucket(tmp_path, repository, "l")
    plan = planner.build(planner.read_spec(_spec(commit, count=1, **changes)), *planner.read_timeline("budget"), planner.Gcloud(BUCKET, True), repository)
    return plan, fake


def test_why_a_job_failed_is_batch_s_exit_first_then_the_task_s_and_the_driver_s_markers() -> None:
    record = {"status": {"state": "FAILED"}}
    assert planner.classify(record, [[50001, 50001, 50001, 50001]], []) == ("preempted", "exit 50001")
    # An attempt preempted, then the driver ran out of memory: the driver's error, deterministic.
    assert planner.classify(record, [[50001, 1]], [f"{MARK}_task_terminated", f"{MARK}_fit_failed-OutOfMemoryError", f"{MARK}_task_exit-1"]) == (
        "driver", "fit_failed-OutOfMemoryError")
    assert planner.classify(record, [[1]], [f"{MARK}_fit_failed-CUDARuntimeError", f"{MARK}_task_exit-push", f"{MARK}_task_exit-1"])[0] == "driver"
    assert planner.classify(record, [[50005]], [])[0] == "time-cap"
    assert planner.classify(record, [[50001, 50004]], [])[0] == "vm-lost"
    assert planner.classify(record, [[3]], [f"{MARK}_task_push-stuck", f"{MARK}_task_exit-push-stuck"])[0] == "push-stuck"
    assert planner.classify(record, [[1]], [f"{MARK}_task_exit-no-gpu"])[0] == "no-gpu"
    assert planner.classify(record, [[1]], [f"{MARK}_task_exit-push", f"{MARK}_task_exit-1"])[0] == "push"
    # A store fan-out's task markers carry its chromosome index.
    assert planner.classify(record, [[0], [1]], [f"{MARK}_task_7-exit-restore"])[0] == "restore"
    assert planner.classify(record, [[1]], [f"{MARK}_task_exit-install"])[0] == "install"
    assert planner.classify(record, [[137]], [])[0] == "unknown"


def test_a_preempted_fit_is_resubmitted_on_a_free_slot_and_a_driver_error_never(tmp_path, cloud) -> None:  # noqa: F811
    plan, fake = _round(tmp_path)
    gcloud = planner.Gcloud(BUCKET, True)
    follower = planner.Follower(plan, gcloud, cloud, "p", "r", log=tmp_path / "follow.log")
    # snv_indel's job used its four attempts on preemptions; with_sv's driver ran out of memory.
    _set(fake, "svpgs-aou-r7-l1-train-1", "FAILED", [50001, 50001, 50001, 50001], hours=2)
    _set(fake, "svpgs-aou-r7-l2-train-1", "FAILED", [1], hours=3)
    _mark(fake, "aou-r7-l2", "fit_failed-OutOfMemoryError", "task_exit-1")
    follower.forget()
    follower.resubmit()
    (resubmitted,) = _submitted(fake)
    assert (resubmitted["RUN_ID"], resubmitted["BUDGET_SLOT"], resubmitted["FIT_ARMS"], resubmitted["PARTNER_RUN"]) == ("aou-r7-l1", "train-3", "snv_indel", "aou-r7-l2")
    assert {key: resubmitted[key] for key in ("MACHINE_TYPE", "STAGED_SOURCE", "MAX_RETRIES")} == {
        key: plan.jobs[0]["environment"][key] for key in ("MACHINE_TYPE", "STAGED_SOURCE", "MAX_RETRIES")}
    chain = json.loads((fake / "gcs" / BUCKET / "svpgs/_rounds/r7/follow/chains/svpgs-aou-r7-l1-train-1.json").read_text())["jobs"]
    assert [(job["name"], job["reason"]) for job in chain] == [("svpgs-aou-r7-l1-train-1", "planned"), ("svpgs-aou-r7-l1-train-3", "preempted")]
    assert follower.ended["svpgs-aou-r7-l2-train-1"][0] == "FAILED" and "driver" in follower.ended["svpgs-aou-r7-l2-train-1"][1]
    logged = [json.loads(line) for line in (tmp_path / "follow.log").read_text().splitlines()]
    assert [line["action"] for line in logged] == ["submitted", "ended"] and "UTC" not in (tmp_path / "follow.log").read_text()
    assert (fake / "gcs" / BUCKET / "svpgs/_rounds/r7/follow/follow.log").read_text() == (tmp_path / "follow.log").read_text()
    # A later pass: the resubmission runs, nothing more is submitted, and the out-of-memory job stays ended.
    _set(fake, "svpgs-aou-r7-l1-train-3", "RUNNING")
    follower.forget()
    follower.resubmit()
    assert len(_submitted(fake)) == 1 and follower.state(plan.jobs[0]) == "RUNNING"
    # A VM that stopped answering is given one more job; the second time, the chain ends.
    _set(fake, "svpgs-aou-r7-l1-train-3", "FAILED", [50004], hours=1)
    follower.forget()
    follower.resubmit()
    assert _submitted(fake)[-1]["BUDGET_SLOT"] == "train-4"
    _set(fake, "svpgs-aou-r7-l1-train-4", "FAILED", [50002], hours=1)
    follower.forget()
    follower.resubmit()
    assert len(_submitted(fake)) == 2 and "again" in follower.ended["svpgs-aou-r7-l1-train-1"][1]
    # Both fit chains ended failed without their arms' parts: the CPU merge is not submitted, and nothing waits.
    assert planner.follow(plan, gcloud, cloud, "p", "r", follower) == [] and len(_submitted(fake)) == 2
    assert follower.state(plan.finishes[0]) == "FAILED"


def test_a_time_cap_is_resubmitted_only_after_progress_and_the_spend_cap_stops_submissions(tmp_path, cloud) -> None:  # noqa: F811
    plan, fake = _round(tmp_path, spend_cap_usd=50)
    gcloud = planner.Gcloud(BUCKET, True)
    follower = planner.Follower(plan, gcloud, cloud, "p", "r")
    assert follower.cap == 50
    # The attempt reached its cap after the mirror last changed: stuck, not resubmitted.
    restart = fake / "gcs" / BUCKET / "svpgs/aou-r7-l1/run/restart/snv_indel/eb/k"
    restart.mkdir(parents=True)
    (restart / "model0.npz").write_text("start")
    os.utime(restart / "model0.npz", (1_000_000_000, 1_000_000_000))
    _set(fake, "svpgs-aou-r7-l1-train-1", "FAILED", [50005], hours=8)
    _set(fake, "svpgs-aou-r7-l2-train-1", "SUCCEEDED", hours=1)
    follower.forget()
    follower.resubmit()
    assert not _submitted(fake) and "stuck" in follower.ended["svpgs-aou-r7-l1-train-1"][1]
    # The same, with an EB start written during the attempt: resubmitted (9 h at $4.41 is $40 of the $50 cap).
    follower = planner.Follower(plan, gcloud, cloud, "p", "r")
    os.utime(restart / "model0.npz", None)
    follower.forget()
    follower.resubmit()
    assert [entry["BUDGET_SLOT"] for entry in _submitted(fake)] == ["train-3"]
    # Another 8 h: $75 spent, over the cap, so the next failure is not resubmitted.
    _set(fake, "svpgs-aou-r7-l1-train-3", "FAILED", [50001], hours=8)
    follower.forget()
    follower.resubmit()
    assert len(_submitted(fake)) == 1 and follower.ended["svpgs-aou-r7-l1-train-1"] == ("FAILED", "spend cap")


def test_max_resubmissions_and_a_deleted_job_end_a_chain(tmp_path, cloud) -> None:  # noqa: F811
    plan, fake = _round(tmp_path, max_resubmissions=1)
    gcloud = planner.Gcloud(BUCKET, True)
    follower = planner.Follower(plan, gcloud, cloud, "p", "r")
    _set(fake, "svpgs-aou-r7-l1-train-1", "FAILED", [50001])
    _set(fake, "svpgs-aou-r7-l2-train-1", "SUCCEEDED")
    follower.forget()
    follower.resubmit()
    _set(fake, "svpgs-aou-r7-l1-train-3", "FAILED", [50001])
    follower.forget()
    follower.resubmit()
    assert len(_submitted(fake)) == 1 and "max_resubmissions" in follower.ended["svpgs-aou-r7-l1-train-1"][1]
    # A job someone deleted is not resubmitted.
    follower = planner.Follower(plan, gcloud, cloud, "p", "r")
    chains = fake / "gcs" / BUCKET / "svpgs/_rounds/r7/follow/chains"
    shutil.rmtree(chains)
    (fake / "jobs/svpgs-aou-r7-l1-train-1").unlink()
    chains.mkdir()
    (chains / "svpgs-aou-r7-l1-train-1.json").write_text(json.dumps({"jobs": [
        {"name": "svpgs-aou-r7-l1-train-1", "run_id": "aou-r7-l1", "slot": "train-1", "environment": {}, "reason": "planned", "time": None}]}))
    follower.forget()
    follower.resubmit()
    assert follower.ended["svpgs-aou-r7-l1-train-1"] == ("FAILED", "deleted") and len(_submitted(fake)) == 1


def test_the_follower_ends_with_the_round_stops_on_stop_and_is_the_round_s_only_one(tmp_path, cloud) -> None:  # noqa: F811
    plan, fake = _round(tmp_path)
    gcloud = planner.Gcloud(BUCKET, True)
    state = tmp_path / "state"
    (state / "rounds").mkdir(parents=True)
    (state / "rounds/r7").write_text("spec\n10\n")
    svpgs = fake / "gcs" / BUCKET / "svpgs"
    # Another machine's follower holds the round and was seen a minute ago: this one does not start.
    lease = svpgs / "_rounds/r7/follow/lease.json"
    lease.parent.mkdir(parents=True)
    lease.write_text(json.dumps({"host": "elsewhere", "pid": 1, "time": time.time() - 60}))
    with pytest.raises(planner.RoundError, match="followed by pid 1 on elsewhere"):
        planner.watch(plan, gcloud, cloud, "p", "r", 10, state)
    # Seen an hour ago, it is gone: this one takes the round. Both arms' jobs SUCCEEDED and the later one finished the
    # run, so the CPU merge is not needed: the round has ended.
    lease.write_text(json.dumps({"host": "elsewhere", "pid": 1, "time": time.time() - 3600}))
    _set(fake, "svpgs-aou-r7-l1-train-1", "SUCCEEDED")
    _set(fake, "svpgs-aou-r7-l2-train-1", "SUCCEEDED")
    (svpgs / "aou-r7-l2/run/report").mkdir(parents=True)
    (svpgs / "aou-r7-l2/run/report/_COMPLETE.json").write_text("{}")
    assert planner.watch(plan, gcloud, cloud, "p", "r", 10, state) == 0
    logged = [json.loads(line) for line in (state / "r7.log").read_text().splitlines()]
    assert logged[0]["action"] == "follow started" and logged[-1]["action"] == "follow ended" and logged[-1]["failed"] == {}
    assert not lease.exists() and not (state / "r7.pid").exists() and not (state / "rounds/r7").exists()
    # A job failed for good: the round ends with status 1, naming it.
    (svpgs / "aou-r7-l2/run/report/_COMPLETE.json").unlink()
    _set(fake, "svpgs-aou-r7-l2-train-1", "FAILED", [1])
    _mark(fake, "aou-r7-l2", "fit_failed-TypeError")
    assert planner.watch(plan, gcloud, cloud, "p", "r", 10, state) == 1
    assert "driver" in json.loads((state / "r7.log").read_text().splitlines()[-1])["failed"]["svpgs-aou-r7-l2-train-1"]
    # A STOP object stops it before it submits anything.
    _set(fake, "svpgs-aou-r7-l2-train-1", "FAILED", [50001])
    (svpgs / "_rounds/r7/follow/STOP").write_text("")
    assert planner.watch(plan, gcloud, cloud, "p", "r", 10, state) == 0 and not _submitted(fake)
    assert json.loads((state / "r7.log").read_text().splitlines()[-1])["action"] == "follow stopped"


def test_a_gpu_job_s_attempt_is_capped_and_the_cap_is_within_the_slot(tmp_path) -> None:
    repository, commit = _repository(tmp_path / "repo", partner=True)
    plan = planner.build(planner.read_spec(_spec(commit, count=1)), *planner.read_timeline("budget"), None, repository)
    assert all(entry["environment"]["MAX_RUN_HOURS"] == "8" for entry in plan.jobs)
    assert "MAX_RUN_HOURS" not in plan.finishes[0]["environment"]
    plan = planner.build(planner.read_spec(_spec(commit, count=1, max_run_hours=None)), *planner.read_timeline("budget"), None, repository)
    assert not any("MAX_RUN_HOURS" in entry["environment"] for entry in plan.jobs)
    for wrong in (0, 2.5, 100, True):
        with pytest.raises(planner.RoundError, match="max_run_hours"):
            planner.read_spec(_spec(commit, max_run_hours=wrong))


FAKE_ROUND = """import os, signal, sys, time
from pathlib import Path

WORKSPACE_ENVIRONMENT = ("OPS_BUCKET",)

if __name__ == "__main__":
    state = Path(sys.argv[sys.argv.index("--state") + 1])
    stopped = []
    signal.signal(signal.SIGTERM, lambda *_: stopped.append(1))
    (state / "r7.pid").write_text(f"{os.getpid()}\\n")
    while not stopped:
        time.sleep(0.1)
    with open(state / "r7.log", "a") as log:
        log.write("stopped\\n")
"""


@pytest.mark.skipif(not Path("/proc/self/cmdline").exists() or shutil.which("setsid") is None, reason="follow.sh runs on the Linux app")
def test_follow_sh_starts_one_detached_follower_stops_it_and_starts_recorded_rounds_again(tmp_path) -> None:
    launcher = tmp_path / "launcher"
    launcher.mkdir()
    shutil.copy(LAUNCHER / "follow.sh", launcher)
    (launcher / "round.py").write_text(FAKE_ROUND)
    spec = tmp_path / "r7.json"
    spec.write_text(json.dumps({"round": "r7"}))
    state = tmp_path / "state"
    environment = dict(os.environ, SVPGS_FOLLOW_STATE=str(state), OPS_BUCKET="ops-bucket", HOME=str(tmp_path / "home"), PATH=os.environ["PATH"])

    def follow(*arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(launcher / "follow.sh"), *arguments], env=environment, capture_output=True, text=True, timeout=120)

    started = follow("start", str(spec), "5")
    assert started.returncode == 0 and "follower started" in started.stdout, started.stderr
    pid = int((state / "r7.pid").read_text())
    try:
        # Its own session, so the terminal that started it can close; the round is recorded with its settings.
        assert os.getsid(pid) == pid != os.getsid(0)
        assert (state / "rounds/r7").read_text().splitlines() == [str(spec), "5"]
        assert "export OPS_BUCKET=ops-bucket" in (state / "r7.env").read_text()
        assert "already runs" in follow("start", str(spec)).stdout
        assert "running (pid" in follow("status").stdout
        stopped = follow("stop", "r7")
        assert stopped.returncode == 0 and "stopped" in stopped.stdout
        assert not Path(f"/proc/{pid}").exists() and not (state / "rounds/r7").exists()
        assert (state / "r7.log").read_text() == "stopped\n"
        # A follower killed with its machine is started again by ensure, as the app's restart runs it; a stopped one not.
        assert follow("start", str(spec), "5").returncode == 0
        os.kill(int((state / "r7.pid").read_text()), signal.SIGKILL)
        time.sleep(0.5)
        assert "not running" in follow("status").stdout
        assert follow("ensure").returncode == 0
        pid = int((state / "r7.pid").read_text())
        assert Path(f"/proc/{pid}").exists()
        follow("stop", "r7")
        assert follow("ensure").returncode == 0 and not Path(f"/proc/{int((state / 'r7.pid').read_text())}").exists()
    finally:
        follow("stop", "r7")
    # install-boot adds a hook to the Jupyter server's config once, and the hook compiles.
    assert "installed" in follow("install-boot").stdout and "already installed" in follow("install-boot").stdout
    config = (tmp_path / "home/.jupyter/jupyter_server_config.py").read_text()
    compile(config, "jupyter_server_config.py", "exec")
    assert config.count("svpgs-follow") == 1 and f"{launcher}/follow.sh" in config
