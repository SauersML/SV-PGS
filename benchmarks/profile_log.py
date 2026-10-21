"""A run's phase timeline from its progress log, with each phase's GPU, CPU, disk, network and memory use, and its stalls.

    python3 benchmarks/profile_log.py progress-task0.full.log [--all]

The log is a Batch task's driver log as launcher/workspace/run_step.sh copies it beside the task log
(progress-task<i>.full.log whole, progress-task<i>.log its last 300 lines) or the task log itself. It reads three kinds
of line:
- ``timing: <phase> <seconds> s [name=value ...]`` (sv_pgs/phase_timer.py, and the restore's from run_step.sh), each
  written at its phase's end;
- ``resources: ...`` once a minute (launcher/workspace/sample_resources.py): the GPUs' utilization, the CPU, the run
  disk's and the network's MiB/s, the host memory in use, each over the minute before the line;
- the driver's other progress lines: their resident memory (the ``| <n> MB]`` of each), the empirical Bayes'
  ``eb curvature:`` and ``joint trial at fraction`` lines, and the quiet-stack lines (sv_pgs/progress.py) that name where
  the driver's main thread was in a silent stretch.

It prints, with every time an offset from the log's start (its first line, or the restore's start where that is
earlier; never a clock time):
1. the timeline: each phase's start, its seconds, the median and 95th percentile of the GPUs' mean utilization over the
   resource lines inside it, its CPU share, its mean disk read and write and network receive rates, and its peak driver
   resident memory; the repeated phases (each mean-field solve, empirical Bayes round, device sweep, scoring batch and
   streamed store read) are summarized after it instead (all of them listed with --all);
2. the repeated phases: count, total, median, 95th percentile and largest seconds, and the medians of their extras;
3. each empirical Bayes loop: its rounds and seconds split into the oracle's solves, the curvature (the round's
   ``eb curvature:`` derivatives and responses), and the rest (the hyper steps and the trials' bookkeeping), and the
   rounds that solved a halved step's trial (the line search) with their seconds;
4. the stalls: each stretch longer than --stall-minutes with no progress line and the GPUs' utilization under
   --idle-util percent at every resource line in it (or no resource line at all), the one still going at the log's end
   included, with the innermost quiet-stack frames seen there and how often.

Only durations, rates, sizes, shares and code locations are printed, never a line's other text: the extras shown are a
fixed list of the timing lines' own (sweeps, seconds, GB, blocks, models, rounds), none of them a count of people. The
standard library alone, on any Python 3.8 or later (the workspace app's).
"""
from __future__ import annotations

import argparse
import math
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

STAMP = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
RESIDENT = re.compile(r"\| (\d+) MB\] ")
MESSAGE = re.compile(r"^.*?\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:,\d+ \S+ [A-Z]+ |[^\]]*\] )(.*)$")
TIMING = re.compile(r"^timing: (\S+) ([0-9.]+) s(.*)$")
EXTRA = re.compile(r"(\w+)=(\S+)")
QUIET = re.compile(r"^quiet \d+ s; main thread at (.*)$")
CURVATURE = re.compile(r"^eb curvature: .*derivatives ([0-9.]+) s, response ([0-9.]+) s")
TRIAL = re.compile(r"^eb outer: model \d+ joint trial at fraction ([0-9.eE+-]+) of a step")
LEGACY_SOLVE = re.compile(r"^mean-field solve: model \d+ (\w+) after (\d+) sweeps in ([0-9.]+) s")
GPU = re.compile(r"gpu\d+ (\d+)% ([0-9.]+)/([0-9.]+) GiB")
CPU = re.compile(r"cpu (\d+)% iowait (\d+)%")
MEMORY = re.compile(r"mem ([0-9.]+)/([0-9.]+) GiB")
DISK = re.compile(r"disk r ([0-9.]+) w ([0-9.]+) MiB/s")
NET = re.compile(r"net rx ([0-9.]+) tx ([0-9.]+) MiB/s")
REPEATED = ("mean_field_solve", "eb_iteration", "device_sweep", "score_batch", "store_read")
"""Phases a run has many of: summarized after the timeline (listed in it with --all)."""
SHOWN_EXTRAS = (
    "sweeps", "per_sweep", "solve", "other", "load", "predict", "gb", "ld_gb", "genotype_pass", "blocks", "parts", "files", "columns",
    "models", "rounds", "launches", "sampled", "codes",
)
"""The timing lines' extras the report shows (with each ``gpu<i>`` busy share): durations, sizes and counts of the run's
own work, never of people, and where a read's codes came from (``codes``: the store, or Stage 0's kept codes)."""
FRAMES_SHOWN = 4


@dataclass
class Phase:
    name: str
    end: float
    seconds: float
    extra: dict

    @property
    def start(self) -> float:
        return self.end - self.seconds


@dataclass
class Sample:
    """A resources line: the minute before ``at``."""

    at: float
    gpu: float | None
    cpu: float | None
    disk_read: float | None
    disk_write: float | None
    net_rx: float | None
    memory: float | None


@dataclass
class Round:
    phase: Phase
    derivatives: float
    response: float
    halved: bool


@dataclass
class Loop:
    phase: Phase
    rounds: list


@dataclass
class Profile:
    phases: list = field(default_factory=list)
    samples: list = field(default_factory=list)
    resident: list = field(default_factory=list)
    activity: list = field(default_factory=list)
    quiet: list = field(default_factory=list)
    loops: list = field(default_factory=list)
    span: float = 0.0


def _sample(at: float, message: str) -> Sample:
    gpus = [float(match.group(1)) for match in GPU.finditer(message)]
    cpu, memory, disk, net = CPU.search(message), MEMORY.search(message), DISK.search(message), NET.search(message)
    return Sample(
        at=at,
        gpu=sum(gpus) / len(gpus) if gpus else None,
        cpu=float(cpu.group(1)) if cpu else None,
        disk_read=float(disk.group(1)) if disk else None,
        disk_write=float(disk.group(2)) if disk else None,
        net_rx=float(net.group(1)) if net else None,
        memory=float(memory.group(1)) if memory else None,
    )


def parse(lines) -> Profile:
    """The profile of a log's lines: its phases in the order their lines were written (each at its phase's end); a line
    with no timestamp (a traceback, a partly copied last line) is skipped."""
    profile = Profile()
    first = None
    derivatives = response = 0.0
    halved = False
    rounds: list = []
    legacy: list = []
    for line in lines:
        stamp, message = STAMP.search(line), MESSAGE.match(line.rstrip("\n"))
        if stamp is None or message is None:
            continue
        clock = datetime.strptime(stamp.group(1), "%Y-%m-%d %H:%M:%S")
        first = clock if first is None else first
        at = (clock - first).total_seconds()
        profile.span = max(profile.span, at)
        text = message.group(1)
        resident = RESIDENT.search(line)
        if resident:
            profile.resident.append((at, int(resident.group(1)) / 1024))
        if text.startswith("resources: "):
            profile.samples.append(_sample(at, text))
            continue
        quiet = QUIET.match(text)
        if quiet:
            profile.quiet.append((at, quiet.group(1)))
            continue
        profile.activity.append(at)
        curvature, trial, timing, solve = CURVATURE.match(text), TRIAL.match(text), TIMING.match(text), LEGACY_SOLVE.match(text)
        if curvature:
            derivatives += float(curvature.group(1))
            response += float(curvature.group(2))
        elif trial:
            halved = halved or float(trial.group(1)) < 1.0
        elif solve:
            seconds = float(solve.group(3))
            legacy.append(Phase("mean_field_solve", at, seconds, {"sweeps": solve.group(2), "per_sweep": str(seconds / max(int(solve.group(2)), 1))}))
        elif timing:
            phase = Phase(timing.group(1), at, float(timing.group(2)), dict(EXTRA.findall(timing.group(3))))
            profile.phases.append(phase)
            if phase.name == "eb_iteration":
                rounds.append(Round(phase, derivatives, response, halved))
                derivatives = response = 0.0
                halved = False
            elif phase.name == "eb":
                profile.loops.append(Loop(phase, rounds))
                rounds = []
    if not any(phase.name == "mean_field_solve" for phase in profile.phases):
        # A log from before the timing lines: its solves from their own lines.
        profile.phases.extend(legacy)
    # Offsets from the earliest moment the log shows: the restore's start precedes its line, and the sampler's first.
    origin = min([0.0] + [phase.start for phase in profile.phases])
    for phase in profile.phases:
        phase.end -= origin
    for sample in profile.samples:
        sample.at -= origin
    profile.resident = [(at - origin, value) for at, value in profile.resident]
    profile.activity = [at - origin for at in profile.activity]
    profile.quiet = [(at - origin, chain) for at, chain in profile.quiet]
    profile.span -= origin
    return profile


def _quantile(values: list, share: float) -> float:
    """The nearest-rank quantile."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(share * len(ordered)) - 1)]


def _offset(seconds: float) -> str:
    whole = int(round(seconds))
    return f"+{whole // 3600}:{whole // 60 % 60:02d}:{whole % 60:02d}"


def _number(value, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _mean(values: list):
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _usage(profile: Profile, start: float, end: float) -> dict:
    """The resource lines and resident memory inside (start, end]."""
    inside = [sample for sample in profile.samples if start < sample.at <= end]
    gpus = [sample.gpu for sample in inside if sample.gpu is not None]
    resident = [value for at, value in profile.resident if start <= at <= end]
    return {
        "gpu_median": statistics.median(gpus) if gpus else None,
        "gpu_p95": _quantile(gpus, 0.95) if gpus else None,
        "cpu": _mean([sample.cpu for sample in inside]),
        "disk_read": _mean([sample.disk_read for sample in inside]),
        "disk_write": _mean([sample.disk_write for sample in inside]),
        "net_rx": _mean([sample.net_rx for sample in inside]),
        "resident": max(resident) if resident else None,
    }


def _extras(phase: Phase) -> str:
    shown = [(name, value) for name, value in phase.extra.items() if name in SHOWN_EXTRAS or re.fullmatch(r"gpu\d+", name)]
    return " ".join(f"{name}={value}" for name, value in shown)


def _timeline(profile: Profile, show_all: bool) -> list:
    rows = [
        f"{'phase':<20} {'start':>9} {'seconds':>9} {'gpu% med':>8} {'p95':>5} {'cpu%':>5} {'disk r':>7} {'disk w':>7} {'net rx':>7} "
        f"{'rss GiB':>8}  extras"
    ]
    # By start, an enclosing phase before the phases inside it.
    for phase in sorted(profile.phases, key=lambda phase: (phase.start, -phase.seconds)):
        if phase.name in REPEATED and not show_all:
            continue
        usage = _usage(profile, phase.start, phase.end)
        rows.append(
            f"{phase.name:<20} {_offset(phase.start):>9} {phase.seconds:>9.1f} {_number(usage['gpu_median'], 0):>8} {_number(usage['gpu_p95'], 0):>5} "
            f"{_number(usage['cpu'], 0):>5} {_number(usage['disk_read']):>7} {_number(usage['disk_write']):>7} {_number(usage['net_rx']):>7} "
            f"{_number(usage['resident']):>8}  {_extras(phase)}"
        )
    return rows


def _repeated(profile: Profile) -> list:
    rows = [f"{'phase':<20} {'count':>6} {'total s':>9} {'median s':>9} {'p95 s':>8} {'max s':>8}  extras (medians)"]
    for name in REPEATED:
        phases = [phase for phase in profile.phases if phase.name == name]
        if not phases:
            continue
        seconds = [phase.seconds for phase in phases]
        medians = []
        for extra in sorted({key for phase in phases for key in phase.extra}):
            if extra not in SHOWN_EXTRAS and not re.fullmatch(r"gpu\d+", extra):
                continue
            values = []
            for phase in phases:
                try:
                    values.append(float(phase.extra[extra]))
                except (KeyError, ValueError):
                    continue
            if values:
                medians.append(f"{extra}={statistics.median(values):.4g}")
        rows.append(
            f"{name:<20} {len(phases):>6} {sum(seconds):>9.1f} {statistics.median(seconds):>9.1f} {_quantile(seconds, 0.95):>8.1f} "
            f"{max(seconds):>8.1f}  {' '.join(medians)}"
        )
    return rows


def _loops(profile: Profile) -> list:
    rows = []
    for index, loop in enumerate(profile.loops, 1):
        rounds = loop.rounds
        solve = sum(float(entry.phase.extra.get("solve", 0.0)) for entry in rounds)
        curvature = sum(entry.derivatives + entry.response for entry in rounds)
        derivatives = sum(entry.derivatives for entry in rounds)
        other = sum(entry.phase.seconds for entry in rounds) - solve - curvature
        searched = [entry for entry in rounds if entry.halved]
        usage = _usage(profile, loop.phase.start, loop.phase.end)
        rows.append(
            f"eb loop {index}: {loop.phase.seconds:.1f} s from {_offset(loop.phase.start)}, {len(rounds)} rounds: solve {solve:.1f} s, "
            f"curvature {curvature:.1f} s (derivatives {derivatives:.1f} s), other {other:.1f} s; line search {len(searched)} rounds, "
            f"{sum(entry.phase.seconds for entry in searched):.1f} s; gpu% median {_number(usage['gpu_median'], 0)}"
        )
        if rounds:
            seconds = [entry.phase.seconds for entry in rounds]
            rows.append(f"  rounds: median {statistics.median(seconds):.1f} s, p95 {_quantile(seconds, 0.95):.1f} s, last {seconds[-1]:.1f} s")
    return rows


def _stalls(profile: Profile, stall_seconds: float, idle_util: float) -> list:
    """Each stretch over ``stall_seconds`` without a progress line in which every resource line shows the GPUs under
    ``idle_util`` percent (a line without GPUs counts as idle; a stretch with no resource line at all is listed too),
    with the quiet-stack frames seen in it. The stretch from the last progress line to the log's end is one (an ongoing
    stall of a live run). A resource line covers the minute before it, so a stretch runs from its gap's start, or from
    the last busy line before it, to the gap's end, or to its own last idle line where a busy line follows."""
    rows = []
    activity = sorted(set(profile.activity))
    gaps = list(zip(activity, activity[1:]))
    if activity and profile.span > activity[-1]:
        gaps.append((activity[-1], profile.span))
    for before, after in gaps:
        if after - before <= stall_seconds:
            continue
        ongoing = after > activity[-1]
        stretches, start, low = [], before, []
        for sample in (sample for sample in profile.samples if before < sample.at <= after):
            if sample.gpu is None or sample.gpu < idle_util:
                low.append(sample)
                continue
            stretches.append((start, low[-1].at if low else start, low))
            start, low = sample.at, []
        stretches.append((start, after, low))
        for start, end, low in stretches:
            if end - start <= stall_seconds:
                continue
            gpus = [sample.gpu for sample in low if sample.gpu is not None]
            frames: dict = {}
            for at, chain in profile.quiet:
                if start <= at <= end:
                    shown = " < ".join(chain.split(" < ")[:FRAMES_SHOWN])
                    frames[shown] = frames.get(shown, 0) + 1
            rows.append(
                f"stall {_offset(start)} for {end - start:.0f} s{' to the log end' if ongoing and end == after else ''}: "
                f"gpu% max {_number(max(gpus) if gpus else None, 0)}, disk r {_number(_mean([sample.disk_read for sample in low]))} MiB/s, "
                f"net rx {_number(_mean([sample.net_rx for sample in low]))} MiB/s, {len(low)} resource lines"
            )
            for chain, seen in sorted(frames.items(), key=lambda item: -item[1]):
                rows.append(f"  quiet x{seen}: {chain}")
    return rows


def report(profile: Profile, show_all: bool = False, stall_seconds: float = 600.0, idle_util: float = 20.0) -> str:
    sections = [
        f"log span {_offset(profile.span)}; {len(profile.phases)} timing lines, {len(profile.samples)} resource lines "
        "(offsets from the log's start; rates in MiB/s, the GPUs' mean utilization, the driver's resident memory)",
        "",
        *_timeline(profile, show_all),
        "",
        *_repeated(profile),
    ]
    loops = _loops(profile)
    if loops:
        sections += ["", *loops]
    stalls = _stalls(profile, stall_seconds, idle_util)
    sections += ["", *(stalls or [f"no stall: no stretch over {stall_seconds / 60:.0f} minutes without a progress line and with the GPUs under {idle_util:.0f}%"])]
    return "\n".join(sections)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path, help="A progress log (progress-task<i>.full.log) or Batch task log.")
    parser.add_argument("--all", action="store_true", help="List the repeated phases in the timeline too.")
    parser.add_argument("--stall-minutes", type=float, default=10.0, help="A stall's least length without a progress line.")
    parser.add_argument("--idle-util", type=float, default=20.0, help="The GPU utilization, in percent, below which a stall's GPUs count as idle.")
    args = parser.parse_args(argv)
    with args.log.open(encoding="utf-8", errors="replace") as lines:
        profile = parse(lines)
    print(report(profile, args.all, args.stall_minutes * 60.0, args.idle_util))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
