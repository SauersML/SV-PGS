"""Reserve each job once against an immutable, workspace-local $3,000 study envelope.

No cloud client library is needed on the submission host. Reservations use GCS generation
preconditions, so two launchers cannot spend the same slot. A failed submission retains its
reservation: it is safer to inspect it than to automatically release an ambiguous charge.

The allowance covers task runtime at the supplied whole-VM hourly rate, including every
retry and an allowance for startup. Persistent storage, queries, idle launch machines and
billing latency use the separate reserve; this is not a provider-enforced billing cap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

MAX_STUDY_USD = 3000


def positive(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (value == 0 and not zero):
        raise ValueError(f"{name} must be a finite {'non-negative' if zero else 'positive'} number")
    return float(value)


def validate_plan(plan):
    total = positive(plan["total_usd"], "total_usd")
    if total > MAX_STUDY_USD:
        raise ValueError("the study budget may not exceed $3,000")
    reserve = positive(plan["non_task_reserve_usd"], "non_task_reserve_usd")
    positive(plan["startup_hours_per_attempt"], "startup_hours_per_attempt")
    slots = plan["slots"]
    if not isinstance(slots, dict) or not slots:
        raise ValueError("slots must be a nonempty allocation map")
    for name, amount in slots.items():
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,20}", name):
            raise ValueError("slot names must be short lowercase identifiers")
        positive(amount, name)
    if math.fsum(slots.values()) + reserve > total:
        raise ValueError("job allocations plus the non-task reserve exceed the study budget")


def duration_seconds(plan, slot, hourly_usd, retries):
    validate_plan(plan)
    rate = positive(hourly_usd, "hourly_usd")
    if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= 10:
        raise ValueError("retries must be an integer between 0 and 10")
    seconds = math.floor((plan["slots"][slot] / (rate * (retries + 1)) - plan["startup_hours_per_attempt"]) * 3600)
    if seconds < 1:
        raise ValueError("this allocation cannot cover startup and one second of task runtime")
    # Batch's job lifetime is at most 14 days, including repeated attempts.
    lifetime = math.floor((14 * 24 / (retries + 1) - plan["startup_hours_per_attempt"]) * 3600)
    if lifetime < 1:
        raise ValueError("startup allowances exhaust Batch's maximum job lifetime")
    return min(seconds, lifetime)


def validate_job(spec, seconds, retries):
    groups = spec["taskGroups"]
    instances = spec["allocationPolicy"]["instances"]
    if len(groups) != 1 or len(instances) != 1 or groups[0].get("taskCount") != 1:
        raise ValueError("budgeted jobs must allocate one task on one VM")
    group = groups[0]
    if group.get("parallelism", 1) != 1 or group.get("taskCountPerNode", 1) != 1:
        raise ValueError("budgeted jobs cannot multiply tasks or nodes")
    task = group["taskSpec"]
    if task.get("maxRetryCount") != retries or task.get("maxRunDuration") != f"{seconds}s":
        raise ValueError("the job's retries or timeout do not match its reservation")
    if "instanceTemplate" in instances[0]:
        raise ValueError("use an explicit machine policy so the reservation identifies the VM")
    policy = instances[0]["policy"]
    if not policy.get("machineType") or policy.get("provisioningModel") not in ("SPOT", "STANDARD"):
        raise ValueError("the VM type and provisioning model must be explicit")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def reserve(plan, slot, spec, rate, retries, root, run=subprocess.run):
    seconds = duration_seconds(plan, slot, rate, retries)
    validate_job(spec, seconds, retries)
    if not root.startswith("gs://") or "#" in root or "*" in root:
        raise ValueError("the ledger must be a workspace GCS prefix")
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "plan.json"
        source.write_bytes(canonical(plan))
        plan_uri = root.rstrip("/") + "/plan.json"
        created = run(["gcloud", "storage", "cp", str(source), plan_uri, "--if-generation-match=0"], capture_output=True)
        if created.returncode:
            existing = run(["gcloud", "storage", "cat", plan_uri], capture_output=True, check=True)
            if canonical(json.loads(existing.stdout)) != canonical(plan):
                raise ValueError("the study already has a different budget plan; its envelope cannot be reset")
        record = {
            "slot": slot, "reserved_usd": plan["slots"][slot], "hourly_usd": rate,
            "max_retry_count": retries, "max_run_seconds": seconds,
            "job_sha256": hashlib.sha256(canonical(spec)).hexdigest(),
            "machine": spec["allocationPolicy"]["instances"][0]["policy"],
        }
        source.write_bytes(canonical(record))
        reserved = run(["gcloud", "storage", "cp", str(source), root.rstrip("/") + f"/slots/{slot}.json", "--if-generation-match=0"], capture_output=True)
        if reserved.returncode:
            raise RuntimeError("slot reservation failed or was already used; no job may be submitted")


def render(template, output, variables):
    """Expand only the named deployment placeholders, preserving the task's shell variables."""
    content = template.read_text()
    for name in variables:
        if name not in os.environ:
            raise ValueError(f"missing deployment variable {name}")
        value = os.environ[name]
        # Values are embedded in shell double quotes or JSON: refuse shell syntax, quotes and controls.
        if not re.fullmatch(r"[a-zA-Z0-9_./:@+ -]*", value):
            raise ValueError(f"unsafe deployment variable {name}")
        content = content.replace("${" + name + "}", value)
    output.write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("duration", "reserve", "render"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--slot")
    parser.add_argument("--hourly-usd", type=float)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--ledger")
    parser.add_argument("--template", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--variables", nargs="+")
    args = parser.parse_args()
    if args.action == "render":
        render(args.template, args.output, args.variables)
        return
    plan = json.loads(args.plan.read_text())
    if args.action == "duration":
        print(duration_seconds(plan, args.slot, args.hourly_usd, args.retries))
    else:
        reserve(plan, args.slot, json.loads(args.spec.read_text()), args.hourly_usd, args.retries, args.ledger)


if __name__ == "__main__":
    main()
