"""round.py --follow left to run a round alone (launcher/workspace/round.py's Follower, watch and Lease, and
launcher/workspace/follow.sh), against the fixture repository and fake gcloud of tests/test_round.py, never a cloud
account: why a failed job failed; which failures it resubmits, on which slot, how often, and which it never does; the
spend cap; a time cap resubmitted only after progress; the round's end, its STOP object and its one follower; and the
follower started detached, stopped, and started again after a restart."""
from __future__ import annotations

import ctypes
import datetime
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

from launcher.workspace import round as planner
from tests.test_round import BUCKET, LAUNCHER, _arrays, _bucket, _repository, _spec, cloud  # noqa: F401 - cloud is a fixture

MARK = "20261022T120000Z"
"""A marker time after every fixture job's creation (CREATED)."""
CREATED = "2026-10-22T11:00:00Z"
PR_SET_CHILD_SUBREAPER = 36


def _job(state: str, exits=(), hours: float = 0.0, machine: str = "a2-highgpu-2g") -> tuple:
    """A job's record as gcloud batch jobs describe gives it, and its one task's exits as gcloud batch tasks list does. It
    became RUNNING a minute ago, so a running job's spend is its ``hours`` whenever the test runs."""
    began = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = {
        "createTime": CREATED, "status": {
            "state": state, "runDuration": f"{hours * 3600:.0f}s",
            "statusEvents": [{"eventTime": began, "description": "Job state is set from SCHEDULED to RUNNING"}],
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
    assert planner.classify(record, [[2]], [])[0] == "unknown"
    # A driver the kernel killed is not resubmitted, though its last push failed after it.
    assert planner.classify(record, [[137]], [])[0] == "killed"
    assert planner.classify(record, [[1]], [f"{MARK}_task_exit-137", f"{MARK}_task_exit-push", f"{MARK}_task_exit-1"])[0] == "killed"
    # A fan-out task's own failure decides, though another task's VM was preempted.
    assert planner.classify(record, [[1], [50001]], [f"{MARK}_store_failed-ValueError"])[0] == "driver"
    assert planner.classify(record, [[50001], [50001, 50005]], [])[0] == "time-cap"
    # A driver interrupted from outside names no error of its own.
    assert planner.classify(record, [[3]], [f"{MARK}_fit_failed-KeyboardInterrupt", f"{MARK}_task_exit-push-stuck"])[0] == "push-stuck"


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


def test_a_failed_read_decides_nothing_and_a_failed_pass_is_tried_again(tmp_path, cloud, monkeypatch) -> None:  # noqa: F811
    plan, fake = _round(tmp_path, spend_cap_usd=1000)
    gcloud = planner.Gcloud(BUCKET, True)
    _set(fake, "svpgs-aou-r7-l1-train-1", "FAILED", [50001], hours=1)
    _set(fake, "svpgs-aou-r7-l2-train-1", "RUNNING", hours=1)
    # The cloud does not answer: neither chain is ended or resubmitted.
    monkeypatch.setenv("FAKE_GCLOUD_DOWN", "1")
    follower = planner.Follower(plan, gcloud, cloud, "p", "r")
    with pytest.raises(planner.Unavailable, match="auth tokens"):
        follower.resubmit()
    assert not follower.ended and not _submitted(fake)
    monkeypatch.delenv("FAKE_GCLOUD_DOWN")
    follower.forget()
    follower.resubmit()
    assert [entry["BUDGET_SLOT"] for entry in _submitted(fake)] == ["train-3"]
    # A pass that fails is logged and the next one runs: here the round then ends.
    state = tmp_path / "state"
    _set(fake, "svpgs-aou-r7-l1-train-3", "SUCCEEDED")
    _set(fake, "svpgs-aou-r7-l2-train-1", "SUCCEEDED")
    report = fake / "gcs" / BUCKET / "svpgs/aou-r7-l2/run/report"
    report.mkdir(parents=True)
    (report / "_COMPLETE.json").write_text("{}")
    resubmit = planner.Follower.resubmit
    calls = []

    def flaky(self):
        calls.append(1)
        if len(calls) == 1:
            raise planner.Unavailable("gcloud storage ls failed: connection reset")
        return resubmit(self)

    monkeypatch.setattr(planner.Follower, "resubmit", flaky)
    assert planner.watch(plan, gcloud, cloud, "p", "r", 0.001, state) == 0
    logged = [json.loads(line) for line in (state / "r7.log").read_text().splitlines()]
    assert [line["action"] for line in logged] == ["follow started", "pass failed", "follow ended"] and len(calls) == 2


def test_the_lease_is_taken_and_kept_only_over_its_own_generation(tmp_path, cloud, monkeypatch) -> None:  # noqa: F811
    plan, fake = _round(tmp_path)
    gcloud = planner.Gcloud(BUCKET, True)
    path = f"{plan.staging}/follow/lease.json"
    with planner.Lease(gcloud, plan, tmp_path / "a", 1800) as first:
        holder = json.loads(gcloud.read(path))
        assert holder["token"] == first.me["token"]
        # Another follower that read no lease before this one wrote it cannot write its own.
        second = planner.Lease(planner.Gcloud(BUCKET, True), plan, tmp_path / "b", 1800)
        with monkeypatch.context() as patched:
            patched.setattr(second.cloud, "generation", lambda _path: None)
            with pytest.raises(planner.Contended):
                second.__enter__()
        first.renew()
        # Another follower takes the lease (this one's had gone stale): this one submits nothing more.
        gcloud.write(path, json.dumps({"host": "elsewhere", "pid": 1, "time": time.time(), "token": "other"}))
        follower = planner.Follower(plan, gcloud, cloud, "p", "r")
        follower.lease = first
        _set(fake, "svpgs-aou-r7-l1-train-1", "FAILED", [50001])
        _set(fake, "svpgs-aou-r7-l2-train-1", "RUNNING")
        follower.forget()
        with pytest.raises(planner.Contended, match="another follower"):
            follower.resubmit()
        assert not _submitted(fake) and not (fake / "gcs" / BUCKET / "svpgs/_rounds/r7/follow/chains/svpgs-aou-r7-l1-train-1.json").exists()
    # Leaving, it does not remove the other follower's lease.
    assert json.loads(gcloud.read(path))["token"] == "other"


def test_a_refused_submission_leaves_the_chain_as_it_was_and_spend_counts_each_task(tmp_path, cloud, monkeypatch) -> None:  # noqa: F811
    plan, fake = _round(tmp_path, spend_cap_usd=1000)
    gcloud = planner.Gcloud(BUCKET, True)
    follower = planner.Follower(plan, gcloud, cloud, "p", "r")
    _set(fake, "svpgs-aou-r7-l1-train-1", "FAILED", [50001])
    _set(fake, "svpgs-aou-r7-l2-train-1", "RUNNING")

    def refused(entry, launcher):
        raise planner.RoundError(f"{entry['name']} was not submitted: quota")

    chain = fake / "gcs" / BUCKET / "svpgs/_rounds/r7/follow/chains/svpgs-aou-r7-l1-train-1.json"
    with monkeypatch.context() as patched:
        patched.setattr(planner, "submit", refused)
        follower.forget()
        with pytest.raises(planner.RoundError, match="quota"):
            follower.resubmit()
    assert [job["reason"] for job in json.loads(chain.read_text())["jobs"]] == ["planned"] and "svpgs-aou-r7-l1-train-1" not in follower.ended
    follower.forget()
    follower.resubmit()
    assert [job["reason"] for job in json.loads(chain.read_text())["jobs"]] == ["planned", "preempted"]
    # A fan-out of 22 tasks ran 2 hours: 44 VM-hours.
    record, _tasks = _job("SUCCEEDED", hours=2, machine="n2-standard-8")
    record["taskGroups"] = [{"taskCount": 22}]
    (fake / "jobs" / "svpgs-aou-r7-l1-train-1").write_text(json.dumps(record))
    _set(fake, "svpgs-aou-r7-l2-train-1", "SUCCEEDED")
    follower.records.clear()
    many = follower.spent()
    (fake / "jobs" / "svpgs-aou-r7-l1-train-1").write_text(json.dumps(dict(record, taskGroups=[{"taskCount": 1}])))
    follower.records.clear()
    one = follower.spent()
    assert one > 0 and many == pytest.approx(22 * one)


def test_a_root_job_execute_did_not_submit_is_submitted_and_a_deleted_one_is_not(tmp_path, cloud, monkeypatch) -> None:  # noqa: F811
    plan, fake = _round(tmp_path)
    gcloud = planner.Gcloud(BUCKET, True)
    follower = planner.Follower(plan, gcloud, cloud, "p", "r")
    # No round.json: --execute has not run, so nothing is submitted.
    planner.follow(plan, gcloud, cloud, "p", "r", follower)
    assert not _submitted(fake)
    svpgs = fake / "gcs" / BUCKET / "svpgs"
    (svpgs / plan.staging / "round.json").parent.mkdir(parents=True, exist_ok=True)
    (svpgs / plan.staging / "round.json").write_text("{}")
    # l1's job is in Batch. l2's slot was reserved 20 minutes ago and Batch has no job: someone deleted it, so it is
    # ended, not submitted again.
    _set(fake, "svpgs-aou-r7-l1-train-1", "RUNNING")
    slots = svpgs / "aou-r7-l2/budget/slots"
    slots.mkdir(parents=True)
    (slots / "train-1.json").write_text("{}")
    os.utime(slots / "train-1.json", (time.time() - 1200, time.time() - 1200))
    follower.forget()
    planner.follow(plan, gcloud, cloud, "p", "r", follower)
    assert not _submitted(fake) and "reserved" in follower.ended["svpgs-aou-r7-l2-train-1"][1]
    # With no reservation, --execute stopped before it: submitted on its own slot.
    (slots / "train-1.json").unlink()
    follower = planner.Follower(plan, gcloud, cloud, "p", "r")
    planner.follow(plan, gcloud, cloud, "p", "r", follower)
    assert [(entry["RUN_ID"], entry["BUDGET_SLOT"]) for entry in _submitted(fake)] == [("aou-r7-l2", "train-1")]
    # This follower's own submission reserved the slot and failed: once the reservation is old, another slot.
    (svpgs / "_rounds/r7/follow/chains/svpgs-aou-r7-l2-train-1.json").unlink()
    follower = planner.Follower(plan, gcloud, cloud, "p", "r")

    def refused(entry, launcher):
        (slots / f"{entry['environment']['BUDGET_SLOT']}.json").write_text("{}")
        raise planner.RoundError(f"{entry['name']} was not submitted: quota")

    with monkeypatch.context() as patched:
        patched.setattr(planner, "submit", refused)
        with pytest.raises(planner.RoundError, match="quota"):
            planner.follow(plan, gcloud, cloud, "p", "r", follower)
    follower.forget()
    waiting = planner.follow(plan, gcloud, cloud, "p", "r", follower)
    assert "svpgs-aou-r7-l2-train-1" in dict(waiting) and len(_submitted(fake)) == 1
    os.utime(slots / "train-1.json", (time.time() - 1200, time.time() - 1200))
    follower.forget()
    planner.follow(plan, gcloud, cloud, "p", "r", follower)
    assert [(entry["RUN_ID"], entry["BUDGET_SLOT"]) for entry in _submitted(fake)][1:] == [("aou-r7-l2", "train-3")]


def test_a_pair_eb_fitted_no_genetic_signal_in_is_marked_and_not_merged(tmp_path, cloud) -> None:  # noqa: F811
    plan, fake = _round(tmp_path)
    gcloud = planner.Gcloud(BUCKET, True)
    svpgs = fake / "gcs" / BUCKET / "svpgs"
    trait, merge = plan.finishes[0]["trait"], plan.finishes[0]["name"]

    def model(run: str, arm: str, scale: float) -> None:
        folder = svpgs / run / "run/fit.partial/models" / arm / trait
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model.json").write_text("{}")
        _arrays(folder / "arrays.npz", scale)

    # Both arms' jobs ended, neither finished the run, and each pushed its handoff: the CPU merge would run. The
    # snv_indel arm stayed at the null, as aou-m3-s1/s2 did from the collapsed seed aou-ws6.
    model("aou-r7-l1", "snv_indel", 2.8e-21)
    model("aou-r7-l2", "with_sv", 4e-3)
    for run in ("aou-r7-l1", "aou-r7-l2"):
        (svpgs / run / "run/effects.partial/batches/a").mkdir(parents=True)
        (svpgs / run / "run/effects.partial/batches/a/handoff.json").write_text("{}")
    _set(fake, "svpgs-aou-r7-l1-train-1", "SUCCEEDED")
    _set(fake, "svpgs-aou-r7-l2-train-1", "SUCCEEDED")
    follower = planner.Follower(plan, gcloud, cloud, "p", "r", log=tmp_path / "follow.log")
    assert planner.follow(plan, gcloud, cloud, "p", "r", follower) == [] and not _submitted(fake)
    why = "EB fitted no genetic signal: the snv_indel arm's mean |b| is 2.8e-21, under 1e-12; the pair is no SV test"
    assert follower.ended[merge] == ("FAILED", why)
    for run in ("aou-r7-l1", "aou-r7-l2"):
        marked = json.loads((svpgs / run / planner.NULL_COLLAPSE).read_text())
        assert marked["null_collapse"] and marked["why"] == why and marked["runs"] == ["aou-r7-l1", "aou-r7-l2"]
        assert marked["arms"]["with_sv"]["mean_abs_effect"] == pytest.approx(4e-3)
    assert json.loads((svpgs / f"_rounds/r7/follow/signal/{trait}.json").read_text()) == marked
    assert [json.loads(line)["action"] for line in (tmp_path / "follow.log").read_text().splitlines()] == ["null collapse", "ended"]
    # A follower started again takes the verdict from the bucket and reads no model again.
    (fake / "calls.log").write_text("")
    follower = planner.Follower(plan, gcloud, cloud, "p", "r")
    assert planner.follow(plan, gcloud, cloud, "p", "r", follower) == [] and follower.ended[merge] == ("FAILED", why) and not _submitted(fake)
    assert not [line for line in (fake / "calls.log").read_text().splitlines() if "arrays.npz" in line]
    # Both arms fitted a signal: no mark, and the merge is submitted.
    (svpgs / f"_rounds/r7/follow/signal/{trait}.json").unlink()
    for run in ("aou-r7-l1", "aou-r7-l2"):
        (svpgs / run / planner.NULL_COLLAPSE).unlink()
    model("aou-r7-l1", "snv_indel", 9e-4)
    follower = planner.Follower(plan, gcloud, cloud, "p", "r")
    assert planner.follow(plan, gcloud, cloud, "p", "r", follower) == []
    assert [entry["PARTNER_RUN"] for entry in _submitted(fake)] == ["aou-r7-l2"] and not (svpgs / "aou-r7-l1" / planner.NULL_COLLAPSE).exists()
    assert not json.loads((svpgs / f"_rounds/r7/follow/signal/{trait}.json").read_text())["null_collapse"]


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
    environment = dict(os.environ, SVPGS_FOLLOW_STATE=str(state), SVPGS_FOLLOW_RESTART_SECONDS="1", OPS_BUCKET="ops-bucket", HOME=str(tmp_path / "home"),
                       PATH=os.environ["PATH"])

    def follow(*arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(launcher / "follow.sh"), *arguments], env=environment, capture_output=True, text=True, timeout=120)

    def gone(pid) -> bool:
        try:
            return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"
        except FileNotFoundError:
            return True

    def started_pid() -> int:
        for _ in range(100):
            if (state / "r7.pid").is_file() and not gone((state / "r7.pid").read_text().strip()):
                return int((state / "r7.pid").read_text())
            time.sleep(0.1)
        raise AssertionError("no follower started")

    # As on the app, where PID 1 (the Jupyter server) reaps no orphan: the supervisor, orphaned when "start" exits, is
    # this test's child and stays a zombie once it exits, which kill -0 still finds.
    libc = ctypes.CDLL(None, use_errno=True)
    assert libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == 0
    started = follow("start", str(spec), "5")
    assert started.returncode == 0 and "follower started" in started.stdout, started.stderr
    pid, supervisor = started_pid(), int((state / "r7.supervisor").read_text())
    try:
        # Its own session, so the terminal that started it can close; the round is recorded with its settings.
        assert os.getsid(pid) == supervisor != os.getsid(0)
        assert (state / "rounds/r7").read_text().splitlines() == [str(spec), "5"]
        assert "export OPS_BUCKET=ops-bucket" in (state / "r7.env").read_text()
        assert "already runs" in follow("start", str(spec)).stdout
        assert "running (pid" in follow("status").stdout
        began = time.monotonic()
        stopped = follow("stop", "r7")
        assert stopped.returncode == 0 and "stopped" in stopped.stdout and time.monotonic() - began < 30
        assert gone(pid) and gone(supervisor) and Path(f"/proc/{supervisor}").exists() and not (state / "rounds/r7").exists()
        assert (state / "r7.log").read_text() == "stopped\n"
        # A follower that dies while its round is recorded is started again by its supervisor.
        assert follow("start", str(spec), "5").returncode == 0
        pid = started_pid()
        os.kill(pid, signal.SIGKILL)
        for _ in range(100):
            if (state / "r7.pid").read_text().strip() != str(pid) and not gone((state / "r7.pid").read_text().strip()):
                break
            time.sleep(0.1)
        assert int((state / "r7.pid").read_text()) != pid and "started again" in (state / "r7.out").read_text()
        # Killed with its machine (the supervisor too), it is started again by ensure, as the app's restart runs it; a
        # stopped one not.
        os.kill(int((state / "r7.supervisor").read_text()), signal.SIGKILL)
        os.kill(started_pid(), signal.SIGKILL)
        time.sleep(0.5)
        assert "not running" in follow("status").stdout
        assert follow("ensure").returncode == 0
        assert not gone(started_pid())
        follow("stop", "r7")
        assert follow("ensure").returncode == 0 and gone(int((state / "r7.pid").read_text()))
        assert f"export PATH={os.environ['PATH']}" in (state / "r7.env").read_text()
    finally:
        follow("stop", "r7")
        libc.prctl(PR_SET_CHILD_SUBREAPER, 0, 0, 0, 0)
        while True:
            try:
                if os.waitpid(-1, os.WNOHANG)[0] == 0:
                    break
            except ChildProcessError:
                break
    # install-boot adds a hook to the Jupyter server's config once, and the hook compiles.
    assert "installed" in follow("install-boot").stdout and "already installed" in follow("install-boot").stdout
    config = (tmp_path / "home/.jupyter/jupyter_server_config.py").read_text()
    compile(config, "jupyter_server_config.py", "exec")
    assert config.count("svpgs-follow") == 1 and f"{launcher}/follow.sh" in config and f"{state}/boot.log" in config
    # On the app, ~/.jupyter's file is root's: the hook goes in the next config directory under HOME that jupyter --paths
    # names, and none outside HOME.
    home = tmp_path / "app"
    (home / ".jupyter").mkdir(parents=True)
    (home / ".jupyter/jupyter_server_config.py").write_text("c = get_config()\n")
    (home / ".jupyter/jupyter_server_config.py").chmod(0o444)
    tools = tmp_path / "jupyter-bin"
    tools.mkdir()
    (tools / "jupyter").write_text(f"#!/bin/sh\nprintf 'config:\\n    {home}/.jupyter\\n    {tmp_path}/etc/jupyter\\n    {home}/.local/etc/jupyter\\n"
                                   f"data:\\n    {home}/.local/share/jupyter\\n'\n")
    (tools / "jupyter").chmod(0o755)
    environment.update(HOME=str(home), PATH=f"{tools}:{os.environ['PATH']}")
    assert f"installed in {home}/.local/etc/jupyter/jupyter_server_config.py" in follow("install-boot").stdout
    assert "already installed" in follow("install-boot").stdout and not (tmp_path / "etc").exists()
    assert (home / ".jupyter/jupyter_server_config.py").read_text() == "c = get_config()\n"
    compile((home / ".local/etc/jupyter/jupyter_server_config.py").read_text(), "jupyter_server_config.py", "exec")
