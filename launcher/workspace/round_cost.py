"""Project what a whole-genome round will cost before any of its jobs is submitted: per arm (one fit job), per merge
(the job, or the part of a job, that brings a trait's two arms to its report), per trait (both arms and the merge) and per
round. Each job's machine, provisioning model and run disk are priced at launcher/workspace/spend.py's list prices and
multiplied by its phases' minutes, and the round's wall clock is followed along the order its jobs wait on each other.
It submits, reads and writes nothing in the cloud. spend.py measures afterwards what the jobs did cost, against these
same prices.

A round takes every trait from its last step before the fit to its report. A plan (a JSON file, below) lists its jobs,
or --layout builds one of three layouts from phase minutes (MINUTES):
  today              One fit job per (trait, arm) on an a2-highgpu-2g with a 2000 GiB pd-balanced disk. Then one job
                     per trait, on the same shape, that restores the store, finds both arms and scores them. This is how
                     the AoU rounds have run.
  one-gpu            One fit job per (trait, arm) on one A100, on the compact fit store (fit_store). Each job scores and
                     summarizes its own arm, and the arm job that finishes last writes the trait's report: the merge is
                     minutes on that job's VM, with no start or restore of its own. An arm whose resident genotype codes
                     fit an A100-40GB (FIT_DEVICE_GB less FIT_DEVICE_RESERVE_GB) runs on an a2-highgpu-1g; a larger arm
                     runs on an a2-ultragpu-1g (A100-80GB).
  one-gpu-whole-store  The same jobs on the whole 547 GB store: bigger disks, and the restore, Stage 0, fill and
                     scoring of today's store.
--canary TRAIT:ARM (repeated) puts only those arms on their one-GPU shape and every other arm on an a2-highgpu-2g, the
round's shape until one-GPU arms have been measured: Stage 0 and the fill decode the store on an a2-highgpu-1g's 12
vCPUs, and its 85 GB of host memory hold the fit's buffers, neither yet run. The report then says how many arms are
canaries.
One A100 does what two do for this fit's largest part. The fused mean-field sweep (sv_pgs/device_sweep.sweep_models) is
one serial Gauss-Seidel chain whose residual is handed from device to device, so a second A100 shards only the products
and Stage 0's reads. An arm whose codes one device holds therefore sweeps as fast on one A100, at half the price.
An arm's resident codes are its kept rows times its training rows, one byte each: 1.06e6 x n bytes for the AoU arms
[real rows]. A fit's log prints them per device as "resident genotype codes: X GB admitted on ...", so an arm fitted on
two devices holds the sum of its two lines. The AoU arms logged so far hold 16.6-29.6 GB [real].

A preemption redoes a job's work since its last durable point. Its expected cost is preemptions per hour of runtime
times the job's hours times the minutes each one redoes, at the job's rate (first order). A resumable job (--resumable,
the restartable fit) redoes only its start, its restore, its digest and half a mirror push interval. Otherwise it redoes
everything before its empirical Bayes and half of that.

Every report sets the projection against today's layout for the same traits at the same preemption rate, thing by
thing, with a fifth of today's as the target (TARGET_FRACTION).

  python3 launcher/workspace/round_cost.py --layout today
  python3 launcher/workspace/round_cost.py --layout one-gpu --arm ldl:snv_indel:16.6 --arm ldl:with_sv:17.8 --minutes eb=40
  python3 launcher/workspace/round_cost.py --layout one-gpu --traits 11 --resumable --submit-env
  python3 launcher/workspace/round_cost.py --layout one-gpu-whole-store --canary trait7:snv_indel --canary trait7:with_sv
  python3 launcher/workspace/round_cost.py --plan round.json

A plan is {"jobs": [job, ...], "preemptions_per_hour": 0.05}. Each job is {"name": ..., "machine": ..., "model": "SPOT"
or "STANDARD", "disk": {"type": ..., "sizeGb": ...}, "minutes": {phase: minutes, ...}, "redo_minutes": minutes a
preemption redoes, "after": [names of the jobs it waits for], "trait": ..., "kind": "fit" or "merge"}. Only "name",
"machine" and "minutes" are required. A job without a kind is a fit if it has an "eb" phase and a merge otherwise. A
job without a trait belongs to the part of its name before its last "-". launcher/workspace/round.py's cost_jobs tuples
become such jobs through from_planner. It prints machine shapes, minutes and dollars only, never a count of people.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

try:
    from launcher.workspace import spend
except ImportError:  # run as a script, from its own directory
    import spend

PHASES = ("start", "restore", "store_digest", "fit_setup", "stage0", "band_fill", "eb", "score", "merge")
"""A job's phases in order, named as launcher/workspace/round.py names them: queue, boot and environment; the mirror
restore; the store's digest; the fit's set-up before Stage 0; Stage 0; the summary band, far field, moment starts and
resident fill; the empirical Bayes; scoring and effects; a merge's work toward the report. A plan may name others, which
are summed after these."""
PRE_EB = PHASES[: PHASES.index("eb")]
RESUME = ("start", "restore", "store_digest")
MINUTES = {
    "today": {
        "fit": {"start": 10, "restore": 17, "store_digest": 12, "fit_setup": 5.3, "stage0": 46.5, "band_fill": 55, "eb": 120},
        "merge": {"start": 10, "restore": 17, "store_digest": 12, "merge": 36},
    },
    "one-gpu": {
        "fit": {"start": 7, "restore": 1.2, "store_digest": 0, "fit_setup": 1, "stage0": 3, "band_fill": 1.2, "eb": 40, "score": 4},
        "merge": {"merge": 4},
    },
    "one-gpu-whole-store": {
        "fit": {"start": 7, "restore": 12, "store_digest": 5, "fit_setup": 1, "stage0": 46.5, "band_fill": 21, "eb": 40, "score": 7},
        "merge": {"merge": 4},
    },
}
"""Each layout's minutes by job kind.
today [real]: per arm, boot and environment ~10 (an estimate, unmeasured), restore 17, the store's digest 12, set-up 5.3
(key checks 4.9, store open and wiring 0.4), Stage 0 43-50 (53-73 projected in the live round), band 9-14 + moment starts
25+ + fill ~20, and EB 1-5 h (2 h here). A merge job per trait runs 1-1.5 h: its own restore and digest, then scoring
both arms.
one-gpu, the owners' estimates, none measured on a GPU yet:
- start 5-9.5 (SHARED_RUNTIME);
- restore 0.6-1.6 of the 20-53 GB compact store, digest 0 with restore receipts (0.4-1.2 without);
- set-up 0.5-1 with the identity calls run concurrently;
- Stage 0 2-4 on the copy, band and fill 1-1.5 (Stage 0's stored sums, the sample route, the fill from the copy);
- EB 15-75 (sweeps 1.4-12.5, non-sweep work <= 15, and the outer iterations);
- scoring 1-3 and effects 1-2 in the arm's job;
- the report 3-5 on the last arm's VM.
one-gpu-whole-store: the same, but the 547 GB restore on four local SSDs of a 12-vCPU VM (~12, between the 2g's 8.5
and pd-balanced's 17), its digest at ~1.3-2.6 GB/s (~5), today's Stage 0, the band and moment starts from Stage 0's sums
with today's fill (~21), and scoring 3-8 + effects 1-2 (~7). Stage 0 and the fill decode the store on half the 2g's
vCPUs, and may take up to twice as long where decoding bounds them (unmeasured)."""
PUSH_MINUTES = 15
"""run_step.sh pushes the mirror every 15 minutes, so a resumed job loses half an interval on average."""
PREEMPTIONS_PER_HOUR = 0.05
"""An assumption, not a measurement: Google publishes no Spot preemption rate. Spot GPU VMs were preempted "often" in the
AoU rounds, and a round of 22 arms of ~4.5 h (~100 VM-hours) with ~5 restarts is 0.05 an hour of runtime. A higher rate
weighs on long jobs that redo much: at 0.2 an hour today's layout loses ~$330 a round to preemptions."""
TARGET_FRACTION = 0.2
"""The cost goal: each thing (an arm, a merge, a trait, a round) at a fifth of what today's layout costs for it."""
FIT_DEVICE_GB = {"a2-highgpu-1g": 40, "a2-ultragpu-1g": 80}
"""Each one-GPU fit shape's device memory, smallest first."""
FIT_DEVICE_RESERVE_GB = 4
"""What a fit's device holds beside its resident codes: the CUDA context, the panel Grams (0.42 GB a model) and the
workspace. One A100-40GB holds ~36 GB of codes, n <= ~34k training rows at the AoU arms' 1.06M kept rows."""
CODES_GB = 29.6
"""The largest arm's resident codes the AoU whole-genome logs show [real] (16.6-29.6 GB over the arms logged): the
default where --arm does not give an arm's own."""
GPU_TYPE = {"a2-highgpu": "nvidia-tesla-a100", "a2-ultragpu": "nvidia-a100-80gb"}
TODAY_DISK = ("pd-balanced", 2000)
"""The AoU rounds' run disk."""
FIT_DISK = {
    ("a2-highgpu-2g", True): ("local-ssd", 750), ("a2-highgpu-2g", False): ("local-ssd", 1500),
    ("a2-highgpu-1g", True): ("local-ssd", 750), ("a2-highgpu-1g", False): ("local-ssd", 1500),
    ("a2-ultragpu-1g", True): ("pd-balanced", 500), ("a2-ultragpu-1g", False): ("pd-balanced", 1000),
}
"""Each fit shape's run disk, on the compact fit store (True) or the whole store (False).
- The whole store's arm holds ~830-880 GB: the 547 GB store, the 150-169 GB LD store, up to 98 GB of band and the
  environment. That is four local SSDs (1500 GiB), since two (750 GiB, 805 GB) are too small.
- The compact store's arm holds ~230-330 GB: 20-53 GB of store, the LD store and what the sample route writes of the
  band. That is two local SSDs, at 1,320 MiB/s read and 700 write.
- An a2-ultragpu-1g bundles a 375 GiB local SSD in its price, and how Batch mounts it is undocumented, so it takes a
  pd-balanced disk.
- An a2-highgpu-2g that is not a canary's takes the same local SSDs as an a2-highgpu-1g."""
TWO_GPU = "a2-highgpu-2g"
"""The shape of an arm that is not a canary while one-GPU arms are unmeasured (--canary)."""
ARMS = ("snv_indel", "with_sv")


def rate(job: dict) -> float:
    """The job's VM and disk price per hour (spend.usd_per_hour), refusing a machine spend.py has no price for."""
    disk = job.get("disk")
    usd, unpriced = spend.usd_per_hour(job["machine"], job.get("model", "SPOT"), [disk] if disk else [])
    if usd is None or unpriced:
        raise ValueError(f"{job['name']}: {'; '.join(unpriced)}")
    return usd


def fit_shape(codes_gb: float) -> str:
    """The cheapest one-GPU fit shape whose device holds the arm's resident codes beside the fit's reserve."""
    for machine, device_gb in FIT_DEVICE_GB.items():
        if codes_gb <= device_gb - FIT_DEVICE_RESERVE_GB:
            return machine
    raise ValueError(f"{codes_gb} GB of resident codes fit no one-GPU shape; fit the arm on two devices (layout today)")


def gpu_count(machine: str) -> int:
    """An A2's GPUs, from its name (a2-highgpu-2g: 2); 0 for a machine without them."""
    suffix = machine.rsplit("-", 1)[-1]
    return int(suffix[:-1]) if machine.startswith("a2-") and suffix.endswith("g") and suffix[:-1].isdigit() else 0


def redo_minutes(minutes: dict, resumable: bool) -> float:
    """What one preemption redoes: the job's start, restore and digest and half a push interval if it resumes from the
    mirror; otherwise everything before its empirical Bayes and half of that, its expected progress when preempted."""
    if resumable:
        return sum(minutes.get(phase, 0) for phase in RESUME) + PUSH_MINUTES / 2
    return sum(minutes.get(phase, 0) for phase in PRE_EB) + minutes.get("eb", 0) / 2


def _job(name, trait, kind, machine, disk, minutes, *, after=(), resumable=False, model="SPOT"):
    return {"name": name, "trait": trait, "kind": kind, "machine": machine, "model": model, "disk": {"type": disk[0], "sizeGb": disk[1]},
            "minutes": dict(minutes), "redo_minutes": redo_minutes(minutes, resumable), "after": list(after)}


def layout_today(traits: list[str], minutes: dict) -> list[dict]:
    jobs = []
    for trait in traits:
        arms = [_job(f"{trait}-{arm}", trait, "fit", "a2-highgpu-2g", TODAY_DISK, minutes["fit"]) for arm in ARMS]
        merge = _job(f"{trait}-merge", trait, "merge", "a2-highgpu-2g", TODAY_DISK, minutes["merge"], after=[job["name"] for job in arms])
        jobs += [*arms, merge]
    return jobs


def layout_one_gpu(arms: list[tuple[str, str, float]], minutes: dict, resumable: bool, compact: bool = True,
                   canaries: set | None = None) -> list[dict]:
    """One fit job per arm on its fit_shape (only the (trait, arm) pairs in ``canaries`` where it is given, the others on
    TWO_GPU), and the trait's merge: minutes on the VM of the arm that ends last (priced at the dearer of its arms'
    rates, after both), with no start or restore of its own."""
    jobs = []
    for trait in dict.fromkeys(trait for trait, _arm, _codes in arms):
        fits = []
        for _trait, arm, codes_gb in (entry for entry in arms if entry[0] == trait):
            machine = fit_shape(codes_gb) if canaries is None or (trait, arm) in canaries else TWO_GPU
            fit = _job(f"{trait}-{arm}", trait, "fit", machine, FIT_DISK[(machine, compact)], minutes["fit"], resumable=resumable)
            fit["fit_arms"], fit["codes_gb"] = arm, codes_gb
            fits.append(fit)
        last = max(fits, key=rate)
        merge = _job(f"{trait}-merge", trait, "merge", last["machine"], (last["disk"]["type"], last["disk"]["sizeGb"]), minutes["merge"],
                     after=[fit["name"] for fit in fits], resumable=resumable)
        jobs += [*fits, merge]
    return jobs


def from_planner(listed: list, end: int = 1) -> list[dict]:
    """launcher/workspace/round.py's cost_jobs tuples (name, machine, provisioning model, disk type, disk GB, {phase:
    (low, high) minutes}, and optionally a dict of the job's other fields here: "trait", "kind", "after", "redo_minutes")
    as jobs here, at the low (0) or high (1) end of each phase. The planner's run IDs differ by arm, so a trait's jobs are
    grouped by the "trait" the dict names."""
    jobs = []
    for name, machine, model, disk_type, disk_gb, phases, *extra in listed:
        minutes = {phase: float(span[end] if isinstance(span, (list, tuple)) else span) for phase, span in phases.items()}
        job = {"name": name, "machine": machine, "model": model, "disk": {"type": disk_type, "sizeGb": int(disk_gb)}, "minutes": minutes}
        jobs.append({**(extra[0] if extra else {}), **job})
    return jobs


def project(jobs: list[dict], preemptions_per_hour: float) -> dict:
    """Each job's rate, minutes, expected preemption minutes and dollars; the dollars by phase; the dollars by thing (each
    fit job, each merge, each trait's jobs together); the total; and the wall clock, the longest chain of jobs through
    "after" (each job's own minutes with its expected preemptions)."""
    names = {job["name"] for job in jobs}
    if len(names) != len(jobs):
        raise ValueError("job names must be unique")
    rows, by_phase, by_trait = [], {}, {}
    for job in jobs:
        missing = set(job.get("after", ())) - names
        if missing:
            raise ValueError(f"{job['name']} waits for unknown jobs {sorted(missing)}")
        usd_per_hour = rate(job)
        minutes = job["minutes"]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 or not math.isfinite(value) for value in minutes.values()):
            raise ValueError(f"{job['name']}: phase minutes must be finite and non-negative")
        own = sum(minutes.values())
        lost = preemptions_per_hour * own / 60 * job.get("redo_minutes", redo_minutes(minutes, False))
        for phase, value in [*minutes.items(), ("preemptions", lost)]:
            by_phase[phase] = by_phase.get(phase, 0.0) + usd_per_hour * value / 60
        kind = job.get("kind") or ("fit" if "eb" in minutes else "merge")
        trait = job.get("trait") or job["name"].rsplit("-", 1)[0]
        usd = usd_per_hour * (own + lost) / 60
        by_trait[trait] = by_trait.get(trait, 0.0) + usd
        rows.append({"name": job["name"], "trait": trait, "kind": kind, "machine": job["machine"], "model": job.get("model", "SPOT"),
                     "gpus": gpu_count(job["machine"]),
                     "usd_per_hour": usd_per_hour, "minutes": own, "preemption_minutes": lost, "usd": usd})
    span = {}
    pending = {job["name"]: job for job in jobs}
    while pending:
        ready = [job for job in pending.values() if all(name in span for name in job.get("after", ()))]
        if not ready:
            raise ValueError(f"the jobs' order has a cycle among {sorted(pending)}")
        for job in ready:
            row = next(row for row in rows if row["name"] == job["name"])
            span[job["name"]] = max((span[name] for name in job.get("after", ())), default=0.0) + row["minutes"] + row["preemption_minutes"]
            del pending[job["name"]]
    order = [phase for phase in PHASES if phase in by_phase] + sorted(set(by_phase) - set(PHASES) - {"preemptions"}) + ["preemptions"]
    things = {
        "arm": [row["usd"] for row in rows if row["kind"] == "fit"],
        "merge": [row["usd"] for row in rows if row["kind"] == "merge"],
        "trait": list(by_trait.values()),
    }
    return {"jobs": rows, "by_phase": {phase: by_phase[phase] for phase in order}, "things": things,
            "usd": sum(row["usd"] for row in rows), "wall_minutes": max(span.values(), default=0.0)}


def _mean(values: list) -> float | None:
    return sum(values) / len(values) if values else None


def report(projection: dict, preemptions_per_hour: float, today: dict | None = None) -> str:
    lines = [f"preemptions: {preemptions_per_hour:g} per hour of a job's runtime, each redoing the job's redo minutes"]
    lines.append(" ".join((f"{'job':<24}", f"{'machine':<17}", f"{'model':<8}", f"{'$/h':>6}", f"{'min':>7}", f"{'preempt':>8}", f"{'$':>8}")))
    for row in projection["jobs"]:
        lines.append(" ".join((f"{row['name']:<24}", f"{row['machine']:<17}", f"{row['model']:<8}", f"{row['usd_per_hour']:>6.3f}",
                               f"{row['minutes']:>7.1f}", f"{row['preemption_minutes']:>8.1f}", f"{row['usd']:>8.2f}")))
    lines += ["", f"{'phase':<12} {'$':>8}"]
    lines += [f"{phase:<12} {usd:>8.2f}" for phase, usd in projection["by_phase"].items()]
    lines += ["", f"{'thing':<8} {'n':>3} {'mean $':>8} {'max $':>8}" + (f" {'today $':>8} {'target $':>8} {'of today':>8}" if today else "")]
    for thing, values in (*projection["things"].items(), ("round", [projection["usd"]])):
        if not values:
            continue
        cells = f"{thing:<8} {len(values):>3} {_mean(values):>8.2f} {max(values):>8.2f}"
        before = (today["things"].get(thing) if thing != "round" else [today["usd"]]) if today else None
        if before:
            cells += f" {_mean(before):>8.2f} {TARGET_FRACTION * _mean(before):>8.2f} {_mean(values) / _mean(before):>8.2f}"
        lines.append(cells)
    fits = [row for row in projection["jobs"] if row["kind"] == "fit"]
    one = sum(1 for row in fits if row["gpus"] == 1)
    if 0 < one < len(fits):
        lines += ["", f"canaries: {one} of {len(fits)} arms on one GPU, the others on {sorted({row['machine'] for row in fits if row['gpus'] != 1})}; "
                  "put every arm on one GPU once the canaries' Stage 0, fill and host memory lines are read"]
    lines += ["", f"projected round: ${projection['usd']:.2f} over {len(projection['jobs'])} jobs, "
              f"{projection['wall_minutes']:.0f} min from the first job's start to the last job's end"]
    if today:
        verdict = "within" if projection["usd"] <= TARGET_FRACTION * today["usd"] else "OVER"
        lines.append(f"{verdict} a fifth of today's ${today['usd']:.2f} for the same traits (${TARGET_FRACTION * today['usd']:.2f})")
    lines.append(f"VM and disk time only, at spend.py's list prices ({spend.PRICES_SOURCE}); not storage or Spot price changes")
    return "\n".join(lines)


def submit_environment(jobs: list[dict]) -> str:
    """For each fit job of a one-GPU layout, the submit.sh fit variables of its shape, HOURLY_USD at its rate rounded up
    to the cent."""
    lines = []
    for job in jobs:
        if "fit_arms" not in job:
            continue
        machine = job["machine"]
        settings = {
            "FIT_ARMS": job["fit_arms"], "MACHINE_TYPE": machine, "GPU_TYPE": GPU_TYPE[machine.rsplit("-", 1)[0]], "GPU_COUNT": gpu_count(machine),
            "RUN_DISK_TYPE": job["disk"]["type"], "RUN_DISK_GB": job["disk"]["sizeGb"], "PROVISIONING_MODEL": job["model"],
            "HOURLY_USD": f"{math.ceil(rate(job) * 100) / 100:.2f}",
        }
        lines.append(f"# {job['name']} ({job['codes_gb']:g} GB of resident codes)")
        lines.append(" ".join(f"{name}={value}" for name, value in settings.items()))
    return "\n".join(lines)


def minutes_of(layout: str, changes: list[str]) -> dict:
    """The layout's MINUTES with each phase=minutes change applied to every job kind that has the phase."""
    minutes = {kind: dict(phases) for kind, phases in MINUTES[layout].items()}
    for change in changes:
        phase, _, value = change.partition("=")
        found = [kind for kind in minutes if phase in minutes[kind]]
        if not found or not value:
            raise ValueError(f"--minutes takes phase=minutes for a phase of the layout's jobs: {change}")
        for kind in found:
            minutes[kind][phase] = float(value)
    return minutes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, help="A round's jobs as JSON (the module docstring).")
    parser.add_argument("--layout", choices=tuple(MINUTES), help="Build the round from phase minutes instead.")
    parser.add_argument("--arm", action="append", default=[], metavar="TRAIT:ARM:CODES_GB",
                        help="An arm of a one-GPU layout and its resident genotype codes in GB; repeat for each.")
    parser.add_argument("--traits", type=int, default=11, help="Without --arm: this many traits, both arms each.")
    parser.add_argument("--codes-gb", type=float, default=CODES_GB, help="Without --arm: every arm's resident codes in GB.")
    parser.add_argument("--canary", action="append", default=[], metavar="TRAIT:ARM",
                        help="Only these arms on one GPU, the others on a2-highgpu-2g; repeat for each.")
    parser.add_argument("--minutes", action="append", default=[], metavar="PHASE=MINUTES", help="Change a phase's minutes.")
    parser.add_argument("--resumable", action="store_true", help="A preempted fit resumes from the mirror.")
    parser.add_argument("--preemptions-per-hour", type=float, help=f"Spot preemptions per hour of runtime (default {PREEMPTIONS_PER_HOUR}, assumed).")
    parser.add_argument("--submit-env", action="store_true", help="Also print each fit job's submit.sh variables.")
    parser.add_argument("--json", action="store_true", help="Print the projection as JSON instead of the table.")
    args = parser.parse_args(argv)
    if (args.plan is None) == (args.layout is None):
        parser.error("give --plan or --layout")
    per_hour = PREEMPTIONS_PER_HOUR
    if args.plan is not None:
        plan = json.loads(args.plan.read_text())
        jobs = plan["jobs"]
        per_hour = plan.get("preemptions_per_hour", per_hour)
    else:
        minutes = minutes_of(args.layout, args.minutes)
        arms = []
        for entry in args.arm:
            trait, arm, codes = entry.split(":")
            if arm not in ARMS:
                parser.error(f"--arm takes an arm of {ARMS}: {entry}")
            arms.append((trait, arm, float(codes)))
        arms = arms or [(f"trait{index + 1}", arm, args.codes_gb) for index in range(args.traits) for arm in ARMS]
        if args.layout == "today":
            jobs = layout_today(list(dict.fromkeys(trait for trait, _arm, _codes in arms)), minutes)
        else:
            canaries = {tuple(entry.split(":")) for entry in args.canary} or None
            unknown = canaries and canaries - {(trait, arm) for trait, arm, _codes in arms}
            if unknown:
                parser.error(f"--canary names arms the round does not have: {sorted(unknown)}")
            jobs = layout_one_gpu(arms, minutes, args.resumable, compact=args.layout == "one-gpu", canaries=canaries)
    if args.preemptions_per_hour is not None:
        per_hour = args.preemptions_per_hour
    projection = project(jobs, per_hour)
    traits = list(dict.fromkeys(row["trait"] for row in projection["jobs"]))
    today = project(layout_today(traits, MINUTES["today"]), per_hour)
    print(json.dumps({**projection, "today": today}, indent=1) if args.json else report(projection, per_hour, today))
    if args.submit_env:
        print(submit_environment(jobs))


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        print(f"round_cost: {error}", file=sys.stderr)
        sys.exit(2)
