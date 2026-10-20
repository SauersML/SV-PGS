"""Estimate what the study's Google Batch jobs have cost, to track the $3,000 envelope (budget_guard.py reserves each
job's slot before it runs; this is what the jobs then ran). It submits, changes and deletes nothing: it reads one
`gcloud batch jobs list` (and, with --per-task, each job's `gcloud batch tasks list`), from gcloud or from saved files.
It prints job names, machine shapes, durations and dollars, nothing about participants, and no clock times.

A job costs its VM hours times its machine's hourly price for its provisioning model (an A2's price includes its GPUs),
plus its disks at their price per GiB-month over HOURS_PER_MONTH. VM hours are the time Batch reports the job RUNNING
(status.runDuration; for a running job, since it became RUNNING), times its task count for a job of several tasks, an
upper bound, since each task has its own VM and they end at different times. --per-task counts each task's spans on a VM
instead (from its assignment to a VM to its end, every attempt). A VM's boot before the job is RUNNING and a boot disk
the job does not name (tens of GB, under a cent an hour) are not counted. Prices are list prices (PRICES_SOURCE): Spot
prices can change daily, and a machine without a price is listed as unpriced, never guessed.

  python3 launcher/workspace/spend.py --location us-central1
  python3 launcher/workspace/spend.py --location us-central1 --per-task
  gcloud batch jobs list --location us-central1 --format=json > jobs.json
  python3 launcher/workspace/spend.py --listing jobs.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import datetime
import json
from pathlib import Path
import re
import subprocess

PRICES_SOURCE = (
    "us-central1 list prices of Cyclenerd's google-cloud-pricing-cost-calculator pricing.yml (generated 2026-10-15), "
    "a copy of Google's Cloud Billing catalog, which the workspace cannot read"
)
MACHINE_USD_PER_HOUR = {
    # machine type: (STANDARD, SPOT), whole VM
    "a2-highgpu-1g": (3.673385, 2.20401),
    "a2-highgpu-2g": (7.34677, 4.40802),
    "a2-highgpu-4g": (14.69354, 8.81604),
    "a2-highgpu-8g": (29.38708, 17.63208),
    "a2-ultragpu-1g": (5.068798, 3.038823),
    "a2-ultragpu-2g": (10.137596, 6.077646),
    "n2d-standard-8": (0.337968, 0.163504),
    "n2d-standard-16": (0.675936, 0.327008),
    "n2d-standard-32": (1.351872, 0.654016),
    "n2d-standard-48": (2.027808, 0.981024),
    "n2d-standard-64": (2.703744, 1.308032),
    "n2d-standard-80": (3.37968, 1.63504),
    "n2d-standard-96": (4.055616, 1.962048),
    "n2d-standard-128": (5.407488, 2.616064),
    "n2d-standard-224": (9.463104, 4.578112),
    "n2d-highmem-8": (0.45592, 0.220528),
    "n2d-highmem-16": (0.91184, 0.441056),
    "n2d-highmem-32": (1.82368, 0.882112),
    "n2d-highmem-48": (2.73552, 1.323168),
    "n2d-highmem-64": (3.64736, 1.764224),
    "n2d-highmem-80": (4.5592, 2.20528),
    "n2d-highmem-96": (5.47104, 2.646336),
}
"""The machine shapes the study runs (A2 for fits; N2D for the store fan-out, measurement and LDpred2), from PRICES_SOURCE."""
DISK_USD_PER_GIB_MONTH = {
    # disk type: (with a STANDARD VM, with a SPOT VM)
    "pd-standard": (0.04, 0.04),
    "pd-balanced": (0.10, 0.10),
    "pd-ssd": (0.17, 0.17),
    "local-ssd": (0.08, 0.0433),
}
HOURS_PER_MONTH = 730
"""Compute Engine's month for prorating a monthly price by the hour."""
STUDY_USD = 3000
"""budget_guard.MAX_STUDY_USD: the envelope the total is set against."""
PREEMPTED = 50001
"""Batch's exit code for a task whose Spot VM was preempted."""
ON_VM = ("ASSIGNED", "RUNNING")
"""A task's states while a VM is held for it."""
LAUNCHED = re.compile(r"/svpgs/([a-z][a-z0-9-]{0,25})/launcher/([a-z][a-z0-9-]{0,20})/run_step\.sh")
"""submit.sh's task script: svpgs/<RUN_ID>/launcher/<BUDGET_SLOT>/run_step.sh on the ops bucket mount."""
STAGED = re.compile(r"/svpgs/([a-z][a-z0-9-]{0,25})(?=[/\"]|$)")
"""Any other job reading a run's staged files (benchmarks/gpu_smoke.sh: OPS=/mnt/disks/ops/svpgs/<RUN_ID>)."""
OTHER = "(no run)"


def moment(text: str) -> datetime.datetime:
    """An RFC 3339 time as Batch writes it (Z or an offset, 0 to 9 fractional digits), in UTC for arithmetic only."""
    matched = re.fullmatch(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d)", text)
    if matched is None:
        raise ValueError(f"not an RFC 3339 time: {text}")
    base = datetime.datetime.strptime(matched.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=datetime.timezone.utc)
    fraction = datetime.timedelta(seconds=float("0." + (matched.group(2) or "0")))
    zone = matched.group(3)
    offset = datetime.timedelta(0) if zone == "Z" else (-1 if zone[0] == "-" else 1) * datetime.timedelta(hours=int(zone[1:3]), minutes=int(zone[4:6]))
    return base + fraction - offset


def hours_between(start: datetime.datetime, end: datetime.datetime) -> float:
    return max((end - start).total_seconds(), 0.0) / 3600


def run_of(job: dict) -> tuple[str, str]:
    """The run and budget slot a job belongs to, from the task script or staged files it names."""
    spec = json.dumps(job.get("taskGroups", []))
    launched = LAUNCHED.search(spec)
    if launched is not None:
        return launched.group(1), launched.group(2)
    staged = STAGED.search(spec)
    return (staged.group(1) if staged is not None else OTHER), "-"


def shape_of(job: dict) -> tuple[str, str, list[dict]]:
    """The machine type and provisioning model the job ran on (the status's, else the requested), and its named disks."""
    policy = job.get("allocationPolicy", {}).get("instances", [{}])[0].get("policy", {})
    ran = [instance for group in job.get("status", {}).get("taskGroups", {}).values() for instance in group.get("instances", [])]
    machine = (ran[0].get("machineType") if ran else None) or policy.get("machineType", "?")
    model = (ran[0].get("provisioningModel") if ran else None) or policy.get("provisioningModel", "STANDARD")
    disks = [disk["newDisk"] for disk in policy.get("disks", []) if "newDisk" in disk]
    if "bootDisk" in policy:
        disks.append(policy["bootDisk"])
    return machine, model, disks


def usd_per_hour(machine: str, model: str, disks: list[dict]) -> tuple[float | None, list[str]]:
    """One VM's price per hour with its disks, or None for a machine without a price; and what was left unpriced."""
    column = 1 if model == "SPOT" else 0
    unpriced = []
    if machine not in MACHINE_USD_PER_HOUR:
        return None, [f"no price for {machine}"]
    rate = MACHINE_USD_PER_HOUR[machine][column]
    for disk in disks:
        kind = disk.get("type", "pd-standard")
        if kind not in DISK_USD_PER_GIB_MONTH:
            unpriced.append(f"no price for disk {kind}")
            continue
        rate += DISK_USD_PER_GIB_MONTH[kind][column] * float(disk.get("sizeGb", 0)) / HOURS_PER_MONTH
    return rate, unpriced


def job_hours(job: dict, now: datetime.datetime) -> tuple[float, str]:
    """The hours Batch reports the job RUNNING (for a running job, since it became RUNNING), and a note on how."""
    status = job.get("status", {})
    reported = float(str(status.get("runDuration", "0s")).rstrip("s") or 0) / 3600
    events = status.get("statusEvents", [])
    started = [moment(event["eventTime"]) for event in events if "taskState" not in event and "to RUNNING" in event.get("description", "")]
    if status.get("state") == "RUNNING":
        if started:
            return max(reported, hours_between(max(started), now)), ""
        return max(reported, hours_between(moment(job["createTime"]), now)), "counted from creation"
    if reported == 0 and started:
        return hours_between(max(started), max(moment(event["eventTime"]) for event in events)), ""
    return reported, ""


def task_hours(tasks: list[dict], now: datetime.datetime) -> float:
    """Every task's spans on a VM: from each assignment to the attempt's end, and to now for a task still on one."""
    total = 0.0
    for task in tasks:
        status = task.get("status", {})
        start = None
        for event in sorted(status.get("statusEvents", []), key=lambda event: moment(event["eventTime"])):
            state = event.get("taskState")
            if state in ON_VM:
                start = start or moment(event["eventTime"])
            elif state is not None and start is not None:
                total += hours_between(start, moment(event["eventTime"]))
                start = None
        if start is not None and status.get("state") in ON_VM:
            total += hours_between(start, now)
    return total


def preemptions(events: list[dict]) -> int:
    return sum(1 for event in events if event.get("taskExecution", {}).get("exitCode") == PREEMPTED)


def report(jobs: list[dict], now: datetime.datetime, tasks_of: dict[str, list[dict]] | None = None, study_usd: float = STUDY_USD) -> str:
    """One line per job, then each run's total, the running jobs' rate, and the total against the envelope."""
    rows = []
    for job in jobs:
        name = str(job.get("name", "?")).rsplit("/", 1)[-1]
        run, slot = run_of(job)
        state = job.get("status", {}).get("state", "?")
        machine, model, disks = shape_of(job)
        rate, notes = usd_per_hour(machine, model, disks)
        tasks = int(job.get("taskGroups", [{}])[0].get("taskCount", 1))
        if tasks_of is not None and name in tasks_of:
            hours = task_hours(tasks_of[name], now)
            preempted = sum(preemptions(task.get("status", {}).get("statusEvents", [])) for task in tasks_of[name])
        else:
            hours, how = job_hours(job, now)
            if how:
                notes.append(how)
            if tasks > 1:
                hours *= tasks
                notes.append(f"at most: {tasks} tasks x the job's time")
            preempted = preemptions(job.get("status", {}).get("statusEvents", []))
        if preempted:
            notes.append(f"{preempted} preempted")
        running = 0
        if state == "RUNNING":
            counts = [int(group.get("counts", {}).get("RUNNING", 0)) for group in job.get("status", {}).get("taskGroups", {}).values()]
            running = sum(counts) or tasks
            notes.insert(0, f"RUNNING on {running} VM(s)")
        rows.append((run, name, slot, state, machine, model, tasks, hours, rate, running, notes))
    rows.sort(key=lambda row: (row[0], row[1]))
    # Columns are joined by a space, so a run or job name longer than its column still stands apart.
    lines = [" ".join((f"{'run':<15}", f"{'job':<33}", f"{'slot':<9}", f"{'state':<10}", f"{'machine':<17}", f"{'model':<8}", f"{'tasks':>5}", f"{'VM h':>8}", f"{'$/h':>7}", f"{'$':>9}", " note"))]
    spent: dict[str, float] = defaultdict(float)
    counted: dict[str, int] = defaultdict(int)
    burning = 0.0
    unpriced = []
    for run, name, slot, state, machine, model, tasks, hours, rate, running, notes in rows:
        usd = hours * rate if rate is not None else None
        if usd is None:
            unpriced.append(name)
        else:
            spent[run] += usd
            burning += rate * running
        counted[run] += 1
        priced = (f"{rate:>7.2f}", f"{usd:>9.2f}") if usd is not None else (f"{'?':>7}", f"{'?':>9}")
        lines.append(" ".join((
            f"{run:<15}", f"{name:<33}", f"{slot:<9}", f"{state:<10}", f"{machine:<17}", f"{model:<8}", f"{tasks:>5}", f"{hours:>8.2f}",
            *priced, " " + "; ".join(notes),
        )))
    lines.append("")
    lines.append(f"{'run':<15} {'jobs':>5} {'$':>9}")
    for run in sorted(counted):
        lines.append(f"{run:<15} {counted[run]:>5} {spent[run]:>9.2f}")
    total = sum(spent.values())
    lines.append(f"{'all runs':<15} {sum(counted.values()):>5} {total:>9.2f}")
    lines.append("")
    lines.append(f"running now: {sum(1 for row in rows if row[3] == 'RUNNING')} job(s), ${burning:.2f} per hour")
    lines.append(f"estimated Batch spend ${total:.2f}, {100 * total / study_usd:.1f}% of the ${study_usd:,.0f} envelope")
    if unpriced:
        lines.append(f"unpriced, not in the total: {', '.join(unpriced)}")
    lines.append(f"VM and disk time only (not storage, the workspace app or Spot price changes); prices: {PRICES_SOURCE}")
    return "\n".join(lines)


def gcloud_json(arguments: list[str]) -> list[dict]:
    shown = subprocess.run(["gcloud", *arguments, "--format=json"], capture_output=True, check=True, text=True)
    return json.loads(shown.stdout or "[]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--location", default="us-central1", help="The Batch region the jobs ran in.")
    parser.add_argument("--project", help="The workspace's Google project, if gcloud's configuration does not name it.")
    parser.add_argument("--listing", type=Path, help="A saved `gcloud batch jobs list --format=json`, instead of listing.")
    parser.add_argument("--per-task", action="store_true", help="Count each task's VM spans (one `gcloud batch tasks list` per job).")
    parser.add_argument("--task-listings", type=Path, help="With --listing: a directory of saved task lists, <job>.json each.")
    parser.add_argument("--study-usd", type=float, default=STUDY_USD, help="The envelope the total is set against.")
    args = parser.parse_args()
    scope = ["--location", args.location] + (["--project", args.project] if args.project else [])
    jobs = json.loads(args.listing.read_text()) if args.listing is not None else gcloud_json(["batch", "jobs", "list", *scope])
    tasks_of = None
    if args.task_listings is not None:
        tasks_of = {path.stem: json.loads(path.read_text()) for path in sorted(args.task_listings.glob("*.json"))}
    elif args.per_task:
        names = [str(job.get("name", "")).rsplit("/", 1)[-1] for job in jobs]
        tasks_of = {name: gcloud_json(["batch", "tasks", "list", "--job", name, *scope]) for name in names if name}
    print(report(jobs, datetime.datetime.now(datetime.timezone.utc), tasks_of, args.study_usd))


if __name__ == "__main__":
    main()
