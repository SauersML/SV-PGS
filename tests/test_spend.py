"""The Batch spend estimate (launcher/workspace/spend.py), with no cloud calls: it lists jobs and their tasks, nothing else."""
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from launcher.workspace import budget_guard, spend

LAUNCHER = Path(__file__).parents[1] / "launcher/workspace"
NOW = datetime.datetime(2026, 10, 20, 20, 0, tzinfo=datetime.timezone.utc)


def _stamp(moment: datetime.datetime) -> str:
    return moment.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _job(name, machine, model, disk_gb, state, *, tasks=1, run_seconds=None, script=None, events=(), environment=None):
    """A job as `gcloud batch jobs list --format=json` shows it: int64 fields as strings, the status's instances."""
    task_spec = {"runnables": [{"script": {"text": script or f"bash /mnt/disks/ops/svpgs/{name}/run_step.sh"}}]}
    if environment:
        task_spec["environment"] = {"variables": environment}
    status = {"state": state, "statusEvents": list(events), "taskGroups": {"group0": {"counts": {state: str(tasks)}, "instances": [{"machineType": machine, "provisioningModel": model}]}}}
    if run_seconds is not None:
        status["runDuration"] = f"{run_seconds}s"
    return {
        "name": f"projects/workspace-project/locations/us-central1/jobs/{name}",
        "createTime": _stamp(NOW - datetime.timedelta(hours=6)),
        "taskGroups": [{"taskCount": str(tasks), "taskSpec": task_spec}],
        "allocationPolicy": {"instances": [{"policy": {
            "machineType": machine, "provisioningModel": model,
            "disks": [{"deviceName": "run", "newDisk": {"type": "pd-balanced", "sizeGb": str(disk_gb)}}],
        }}]},
        "status": status,
    }


def _changed(before, after, moment):
    return {"type": "STATUS_CHANGED", "description": f"Job state is set from {before} to {after} for job projects/p/locations/us-central1/jobs/j.", "eventTime": _stamp(moment)}


def _task_event(state, moment, exit_code=None):
    event = {"type": "STATUS_CHANGED", "description": f"Task state is updated to {state}.", "eventTime": _stamp(moment), "taskState": state}
    if exit_code is not None:
        event["taskExecution"] = {"exitCode": exit_code}
    return event


def _rate(machine, model, disk_gb):
    column = 1 if model == "SPOT" else 0
    return spend.MACHINE_USD_PER_HOUR[machine][column] + spend.DISK_USD_PER_GIB_MONTH["pd-balanced"][column] * disk_gb / spend.HOURS_PER_MONTH


def _study_jobs():
    launched = "bash /mnt/disks/ops/svpgs/{run}/launcher/{slot}/run_step.sh"
    return [
        # A fit arm, preempted once and retried, then finished.
        _job("svpgs-aou-wa1-train-6", "a2-highgpu-2g", "SPOT", 2000, "SUCCEEDED", run_seconds=7200, script=launched.format(run="aou-wa1", slot="train-6"),
             events=[_changed("SCHEDULED", "RUNNING", NOW - datetime.timedelta(hours=4)), _task_event("FAILED", NOW - datetime.timedelta(hours=3), 50001)]),
        # A measurement job running on STANDARD for an hour and a half.
        _job("svpgs-aou-wa2-train-1", "n2d-highmem-96", "STANDARD", 1000, "RUNNING", script=launched.format(run="aou-wa2", slot="train-1"),
             events=[_changed("QUEUED", "SCHEDULED", NOW - datetime.timedelta(hours=3)), _changed("SCHEDULED", "RUNNING", NOW - datetime.timedelta(hours=1.5))]),
        # A store fan-out of 22 chromosome tasks.
        _job("svpgs-aou-wa1-train-5", "n2d-standard-128", "SPOT", 500, "SUCCEEDED", tasks=22, run_seconds=3600, script=launched.format(run="aou-wa1", slot="train-5")),
        # A job still waiting for capacity, and one on a machine without a price.
        _job("svpgs-aou-wa3-train-6", "a2-highgpu-2g", "SPOT", 2000, "QUEUED", script=launched.format(run="aou-wa3", slot="train-6")),
        _job("svpgs-aou-wa3-train-2", "g2-standard-24", "SPOT", 2000, "FAILED", run_seconds=600, script=launched.format(run="aou-wa3", slot="train-2")),
        # The GPU smoke job reads a run's staged source through its OPS variable.
        _job("svpgs-gsmoke-c66695d", "a2-highgpu-2g", "STANDARD", 100, "FAILED", run_seconds=1260, script='bash "$OPS/source/benchmarks/gpu_smoke.sh" task',
             environment={"OPS": "/mnt/disks/ops/svpgs/aou-smoke", "RUN": "/mnt/disks/run", "JOB": "svpgs-gsmoke-c66695d"}),
    ]


def test_the_script_s_constants_are_the_launcher_s() -> None:
    assert spend.STUDY_USD == budget_guard.MAX_STUDY_USD
    # The preemption it counts is one the job templates retry.
    for template in ("job_fit.json.template", "job_store.json.template"):
        (codes,) = re.findall(r'"action": "RETRY_TASK", "actionCondition": \{"exitCodes": \[([0-9, ]+)\]\}', (LAUNCHER / template).read_text())
        assert spend.PREEMPTED in [int(code) for code in codes.split(",")]
    for standard, spot in [*spend.MACHINE_USD_PER_HOUR.values(), *spend.DISK_USD_PER_GIB_MONTH.values()]:
        assert 0 < spot <= standard


def test_each_job_is_priced_by_its_shape_and_time_and_each_run_is_totalled() -> None:
    text = spend.report(_study_jobs(), NOW)
    rows = {line.split()[1]: line for line in text.splitlines()[1:7]}
    fit = rows["svpgs-aou-wa1-train-6"].split()
    assert fit[:9] == ["aou-wa1", "svpgs-aou-wa1-train-6", "train-6", "SUCCEEDED", "a2-highgpu-2g", "SPOT", "1", "2.00", f"{_rate('a2-highgpu-2g', 'SPOT', 2000):.2f}"]
    assert fit[9] == f"{2 * _rate('a2-highgpu-2g', 'SPOT', 2000):.2f}" and "1 preempted" in rows["svpgs-aou-wa1-train-6"]
    measuring = rows["svpgs-aou-wa2-train-1"]
    assert measuring.split()[7] == "1.50" and measuring.split()[9] == f"{1.5 * _rate('n2d-highmem-96', 'STANDARD', 1000):.2f}"
    assert "RUNNING on 1 VM(s)" in measuring
    fan_out = rows["svpgs-aou-wa1-train-5"]
    assert fan_out.split()[6:8] == ["22", "22.00"] and "at most: 22 tasks x the job's time" in fan_out
    assert rows["svpgs-aou-wa3-train-6"].split()[7] == "0.00" and rows["svpgs-aou-wa3-train-2"].split()[8:10] == ["?", "?"]
    smoke = rows["svpgs-gsmoke-c66695d"].split()
    assert smoke[0] == "aou-smoke" and smoke[2] == "-" and smoke[9] == f"{0.35 * _rate('a2-highgpu-2g', 'STANDARD', 100):.2f}"
    totals = {line.split()[0]: line.split()[1:] for line in text.split("\n\n")[1].splitlines()[1:]}
    wa1 = 2 * _rate("a2-highgpu-2g", "SPOT", 2000) + 22 * _rate("n2d-standard-128", "SPOT", 500)
    assert totals["aou-wa1"] == ["2", f"{wa1:.2f}"] and totals["aou-wa3"] == ["2", "0.00"]
    total = wa1 + 1.5 * _rate("n2d-highmem-96", "STANDARD", 1000) + 0.35 * _rate("a2-highgpu-2g", "STANDARD", 100)
    assert totals["all"] == ["runs", "6", f"{total:.2f}"]
    assert f"running now: 1 job(s), ${_rate('n2d-highmem-96', 'STANDARD', 1000):.2f} per hour" in text
    assert f"estimated Batch spend ${total:.2f}, {100 * total / 3000:.1f}% of the $3,000 envelope" in text
    assert "unpriced, not in the total: svpgs-aou-wa3-train-2" in text
    # Durations only: no clock time of any zone.
    assert not re.search(r"\d\d:\d\d", text)


def test_batch_times_are_read_in_any_offset() -> None:
    assert spend.moment("2026-10-20T11:30:00.5-07:00") == datetime.datetime(2026, 10, 20, 18, 30, 0, 500000, tzinfo=datetime.timezone.utc)
    assert spend.moment("2026-10-20T18:30:00Z") == spend.moment("2026-10-20T18:30:00.000000000Z")


def test_per_task_spend_lists_only_jobs_and_their_tasks(tmp_path: Path) -> None:
    """Against a local fake gcloud that answers the two listings and refuses every other command."""
    now = datetime.datetime.now(datetime.timezone.utc)
    start = now - datetime.timedelta(hours=5)
    fan_out = _job("svpgs-aou-wa1-train-5", "n2d-standard-128", "SPOT", 500, "SUCCEEDED", tasks=2, run_seconds=8000,
                   script="bash /mnt/disks/ops/svpgs/aou-wa1/launcher/train-5/run_step.sh")
    fit = _job("svpgs-aou-wa1-train-6", "a2-highgpu-2g", "SPOT", 2000, "RUNNING", script="bash /mnt/disks/ops/svpgs/aou-wa1/launcher/train-6/run_step.sh",
               events=[_changed("SCHEDULED", "RUNNING", now - datetime.timedelta(hours=1))])

    def at(hours):
        return start + datetime.timedelta(hours=hours)

    tasks = {
        # One task preempted after an hour on its VM and finished on a second VM in another hour; one done in half an hour.
        "svpgs-aou-wa1-train-5": [
            {"name": "task0", "status": {"state": "SUCCEEDED", "statusEvents": [
                _task_event("ASSIGNED", at(0)), _task_event("RUNNING", at(0.1)), _task_event("FAILED", at(1), 50001),
                _task_event("PENDING", at(1.05)), _task_event("ASSIGNED", at(1.2)), _task_event("RUNNING", at(1.25)), _task_event("SUCCEEDED", at(2.2))]}},
            {"name": "task1", "status": {"state": "SUCCEEDED", "statusEvents": [
                _task_event("ASSIGNED", at(0)), _task_event("RUNNING", at(0.1)), _task_event("SUCCEEDED", at(0.5))]}},
        ],
        "svpgs-aou-wa1-train-6": [
            {"name": "task0", "status": {"state": "RUNNING", "statusEvents": [_task_event("ASSIGNED", now - datetime.timedelta(hours=1)), _task_event("RUNNING", now - datetime.timedelta(hours=0.9))]}},
        ],
    }
    (tmp_path / "jobs.json").write_text(json.dumps([fan_out, fit]))
    listings = tmp_path / "tasks"
    listings.mkdir()
    for name, listed in tasks.items():
        (listings / f"{name}.json").write_text(json.dumps(listed))
    executable = tmp_path / "gcloud"
    executable.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
Path(os.environ["CALLS"]).open("a").write(json.dumps(args) + "\\n")
root = Path(os.environ["FAKE"])
if args[:3] == ["batch", "jobs", "list"]:
    print((root / "jobs.json").read_text())
elif args[:3] == ["batch", "tasks", "list"]:
    print((root / "tasks" / (args[args.index("--job") + 1] + ".json")).read_text())
else:
    sys.exit(f"refused: {args}")
''')
    executable.chmod(0o755)
    environment = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"], CALLS=str(tmp_path / "calls"), FAKE=str(tmp_path))
    script = LAUNCHER / "spend.py"
    shown = subprocess.run([sys.executable, str(script), "--location", "us-central1", "--per-task"], env=environment, capture_output=True, text=True)
    assert shown.returncode == 0, shown.stderr
    calls = [json.loads(line) for line in (tmp_path / "calls").read_text().splitlines()]
    assert calls == [
        ["batch", "jobs", "list", "--location", "us-central1", "--format=json"],
        ["batch", "tasks", "list", "--job", "svpgs-aou-wa1-train-5", "--location", "us-central1", "--format=json"],
        ["batch", "tasks", "list", "--job", "svpgs-aou-wa1-train-6", "--location", "us-central1", "--format=json"],
    ]
    rows = {line.split()[1]: line for line in shown.stdout.splitlines()[1:3]}
    # Each attempt's VM time, not 2 tasks x the job's 2.2 h.
    assert rows["svpgs-aou-wa1-train-5"].split()[7] == "2.50" and "1 preempted" in rows["svpgs-aou-wa1-train-5"]
    assert "at most" not in rows["svpgs-aou-wa1-train-5"]
    assert rows["svpgs-aou-wa1-train-6"].split()[7] == "1.00" and "RUNNING on 1 VM(s)" in rows["svpgs-aou-wa1-train-6"]
    # The same from saved listings, with no gcloud call.
    os.remove(tmp_path / "calls")
    saved = subprocess.run([sys.executable, str(script), "--listing", str(tmp_path / "jobs.json"), "--task-listings", str(listings)], env=environment, capture_output=True, text=True)
    assert saved.returncode == 0, saved.stderr
    assert not (tmp_path / "calls").exists()
    assert saved.stdout.splitlines()[1] == shown.stdout.splitlines()[1]
