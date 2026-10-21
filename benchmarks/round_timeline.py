"""A run's step-by-step timeline from its progress log, and the round's end-to-end gate on a CPU fixture.

    python3 benchmarks/round_timeline.py progress-task0.full.log [--baseline BASE.json] [--save SIGNATURE.json]
    python -m benchmarks.round_timeline --fixture DIRECTORY [--baseline BASE.json] [--save SIGNATURE.json]

The log is a driver's progress log (launcher/workspace/run_step.sh's progress-task<i>.full.log, or the fixture's). Each
pipeline step ends with one line ``timing: step_<name> <seconds> s state=<reused|computed|part|failed>``
(``workspace_pipeline._run_step``), and the phases inside a step end with their own timing lines (sv_pgs/phase_timer.py),
which benchmarks/profile_log.py reads. A line belongs to the step whose line follows it: one driver writes its log in
order, so everything between one step's line and the next is the next step's work (a log of several drivers at once,
interleaved, cannot be read this way). It prints:
1. each step: its state, its start as an offset from the log's start (never a clock time), its seconds, the phases
   inside it with their counts and seconds, and its unattributed seconds, the part of the step no phase line covers
   (a fit's set-up before its first Stage 0 that fit_open and fit_filter do not cover);
2. every phase's count and seconds over the log, with its sweeps, rounds or models where the phase has them;
3. the problems: a driver run (``Timeline.drivers``) with a step missing between its first and its last, a step that
   failed, a phase no rule here knows, an engine phase outside the fit, a scoring batch outside the score step or a
   step's key line in another step, a reused step that did work beyond its key, a fitted group (a ``stage2 wiring:
   group i of G`` line) without its own Stage 0, far field and summary band line or without a solve and an empirical
   Bayes loop, a computed fit that filtered an arm for the fitter and fitted or resumed no group, and a computed score
   step with neither a scoring batch nor an arm's scores (a run joining arms scored in their own runs reads only the
   latter). A phase silently skipped is one of these: its work happened, or had to, and its line
   is missing.
With --save the log's signature, its steps' states and each phase's count, sweeps, rounds and models (the counts of the
run's own work, which a deterministic fixture repeats exactly) and seconds, goes to a JSON file; with --baseline each is
printed beside an earlier signature's, so a change's claim (fewer sweeps, a digest gone, a faster phase) is read
against the same fixture before and after it.

--fixture runs the whole pipeline, samples to export, with the production measurement model, engine, artifacts and
scorer on the CPU (``workspace_pipeline.workspace_bindings``), on the synthetic workspace inputs of
tests/test_workspace_pipeline.py (two chromosomes of popped batches in two imputation halves, long-read truth calls, a
GATK-SV call set, a disease and a lab trait), as the round fits them: one model per trait and arm (cross_fit false),
both arms, each trait on its own panel weight. It is deterministic: one CPU thread, the inputs' and the fit's seeds
fixed, and the fit's route pinned (``FIXTURE_ROUTE``), which the engine otherwise picks by timing a pass on each. It
then runs the pipeline again on the same run directory, where every step is reused. It reads the repository's tests
package, so it runs from the repository root; the log's analysis needs only the standard library (Python 3.8 or
later, the workspace app's).

Only durations, step names and states, phase names and the counts of the run's own work are printed, never a line's
other text: no count of people (a group line's training rows are not read).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    from benchmarks import profile_log
except ImportError:  # run as a script, from its own directory
    import profile_log

STEP_NAMES = ("samples", "store", "phenotypes", "cohort", "measurement", "direct_sv", "fit", "score", "effects", "report", "export")
"""``workspace_pipeline.STEP_NAMES``, in order (tests/test_round_timeline.py checks they agree); the log's analysis does
not import sv_pgs."""
STEP = "step_"
"""A step line's phase prefix: ``step_<name>``."""
STATES = ("reused", "computed", "part", "failed")
GROUP_PHASES = ("stage0", "far_field", "summary_band")
"""The phases each fitted group runs once: its Stage 0, its far field and its summary band (``stage2_wiring``)."""
LOOP_PHASES = ("mean_field_solve", "eb")
"""The phases each fitted group runs at least once: a mean-field solve and an empirical Bayes loop."""
CANARY_PHASES = ("moment_starts", "annotation_design", "mode_mixture")
"""The engine's phases benchmarks/wg_canary.py times around the production functions, inside the fit like the rest."""
ENGINE_PHASES = (*GROUP_PHASES, "resident_fill", "store_read", "mean_field_solve", "eb_iteration", "eb", "device_sweep", *CANARY_PHASES)
"""The engine's phases, which belong inside the fit step; the resident fill, the streamed store reads and the device
sweeps' busy shares are the device route's, so a CPU run has none."""
FIT_PHASES = ("fit_open", "fit_filter")
"""The fit step's own set-up: opening its inputs (``timing: fit_open``) and each arm's filter (``fit_filter``)."""
KEY_PHASES = (*(name + "_key" for name in STEP_NAMES), "file_digest", "genotype_file_fingerprint")
"""A step's key (``timing: <step>_key``, ``workspace_pipeline._run_step``) and its inputs' identities (``_each``): every
step pays them, a reused one too, so they are its key's work, not the step's."""
SCORE_PHASES = ("score_batch", "score_arm")
"""A scoring batch (its models loaded, read with the store and committed) and an arm's whole (its batches made or read
back, ``workspace_pipeline._arm_scores``)."""
OTHER_PHASES = ("store_restore", "tree_digest")
"""The job's restore (run_step.sh, before the driver's first step) and a store's digest (a fit's, a score's)."""
JOB_PHASES = ("store_restore", "canary_generate", "canary_prepare", "canary_snv_indel", "canary_with_sv", "canary_merge")
"""The phases of a job, not of its driver's steps: the restore before the driver, and the canary's input generation
and each of its processes' whole (wg_canary.run_stage)."""
KNOWN_PHASES = frozenset((*ENGINE_PHASES, *FIT_PHASES, *KEY_PHASES, *SCORE_PHASES, *OTHER_PHASES, *JOB_PHASES, *(STEP + name for name in STEP_NAMES)))
GROUP_LINE = re.compile(r"^stage2 wiring: group \d+ of \d+: ")
RESUMED_LINE = re.compile(r"^stage2 wiring: resumed completed models ")
COUNTED_EXTRAS = {"mean_field_solve": "sweeps", "eb": "rounds", "score_batch": "models"}
"""The extra each phase's lines sum into its count of work: a solve's sweeps, a loop's rounds, a batch's models."""
FIXTURE_ROUTE = (0.0, float("inf"))
"""The fixture's pass costs (``gram_space.pass_costs``: seconds over the samples, over the summary band), so its fit
takes the full-data route over the samples on every machine: the engine picks the cheaper pass by timing one, which a
fast or a loaded machine could flip, and the fixture's counts must repeat."""


@dataclass
class Step:
    name: str
    state: str
    phase: profile_log.Phase
    inside: list = field(default_factory=list)
    groups: int = 0
    resumed: int = 0

    @property
    def unattributed(self) -> float:
        """The step's seconds no phase inside it covers (the union of their spans, clipped to the step's)."""
        spans = sorted((max(phase.start, self.phase.start), min(phase.end, self.phase.end)) for phase in self.inside)
        covered, reach = 0.0, self.phase.start
        for start, end in spans:
            if end > reach:
                covered += end - max(start, reach)
                reach = end
        return max(self.phase.seconds - covered, 0.0)


@dataclass
class Timeline:
    steps: list
    phases: list
    outside: list
    """The phases after the last step line (a driver that stopped mid-step), or of a log without step lines."""
    job: list
    """The job's own phases (JOB_PHASES), which belong to no step."""

    def drivers(self) -> list:
        """The steps by driver run: a run passes the steps in STEP_NAMES' order, so a step at or before the one logged
        before it begins another (the next driver of the job, or a region's run, which begins at its fit)."""
        runs: list = []
        for step in self.steps:
            if not runs or step.name not in STEP_NAMES or runs[-1][-1].name not in STEP_NAMES or (
                STEP_NAMES.index(step.name) <= STEP_NAMES.index(runs[-1][-1].name)
            ):
                runs.append([])
            runs[-1].append(step)
        return runs


def read(lines) -> Timeline:
    """The log's steps in order, each with the phases and group lines written since the step before it."""
    lines = list(lines)
    profile = profile_log.parse(lines)
    # profile_log keeps the timing lines in the order written; the same filter here walks them alongside the group lines.
    phases = iter(profile.phases)
    steps, pending, job, groups, resumed = [], [], [], 0, 0
    for line in lines:
        stamp, message = profile_log.STAMP.search(line), profile_log.MESSAGE.match(line.rstrip("\n"))
        if stamp is None or message is None:
            continue
        text = message.group(1)
        if GROUP_LINE.match(text):
            groups += 1
        elif RESUMED_LINE.match(text):
            resumed += 1
        elif profile_log.TIMING.match(text) and not text.startswith("resources: "):
            phase = next(phases)
            if phase.name in JOB_PHASES:
                job.append(phase)
            elif not phase.name.startswith(STEP):
                pending.append(phase)
            else:
                steps.append(Step(phase.name[len(STEP):], phase.extra.get("state", "?"), phase, pending, groups, resumed))
                pending, groups, resumed = [], 0, 0
    return Timeline(steps, profile.phases, pending, job)


def problems(timeline: Timeline) -> list:
    """What the log says went unlogged or wrong (the module's list 3); empty for a whole, clean run."""
    found = []
    for run in timeline.drivers():
        names = [step.name for step in run]
        if any(name not in STEP_NAMES for name in names):
            found.append(f"unknown steps logged: {' '.join(name for name in names if name not in STEP_NAMES)}")
            continue
        first = STEP_NAMES.index(names[0])
        # A driver passes every step from the first (a region's run, from its fit), each with its line.
        if names[0] not in ("samples", "fit") or names != list(STEP_NAMES[first:first + len(names)]):
            found.append(f"a driver run logged {' '.join(names)}: a step between these has no line")
    for phase in timeline.phases:
        if phase.name not in KNOWN_PHASES:
            found.append(f"phase {phase.name} has no rule in benchmarks/round_timeline.py (KNOWN_PHASES): its time is counted nowhere")
    for step in timeline.steps:
        names = [phase.name for phase in step.inside]
        work = [name for name in names if name not in KEY_PHASES]
        if step.state not in STATES:
            found.append(f"step {step.name}: no state on its line")
        if step.state == "failed":
            found.append(f"step {step.name} failed")
        if step.state == "reused" and (work or step.groups):
            found.append(f"step {step.name} was reused but did work: {' '.join(sorted(set(work)))}")
        misplaced = sorted({name for name in names if (name in (*ENGINE_PHASES, *FIT_PHASES) and step.name != "fit") or (name in SCORE_PHASES and step.name != "score")
                            or (name.endswith("_key") and name != step.name + "_key")})
        if misplaced:
            found.append(f"step {step.name} holds another step's phases: {' '.join(misplaced)}")
        if step.name == "fit" and step.state in ("computed", "part"):
            for name in GROUP_PHASES:
                if names.count(name) != step.groups:
                    found.append(f"fit: {step.groups} groups fitted, {names.count(name)} {name} lines")
            for name in LOOP_PHASES:
                if names.count(name) < step.groups:
                    found.append(f"fit: {step.groups} groups fitted, {names.count(name)} {name} lines")
            # A fit that set out to fit an arm (its filter line, just before the fitter) fits or resumes a group; one
            # whose every model was on disk (a merge of the arms' runs) has no filter line and fits nothing.
            if step.groups + step.resumed == 0 and step.state == "computed" and "fit_filter" in names:
                found.append("fit: computed with no group fitted or resumed")
        if step.name == "score" and step.state == "computed" and "score_batch" not in names and "score_arm" not in names:
            found.append("score: computed with no scoring batch and no arm's scores")
    if timeline.outside and timeline.steps:
        found.append(f"{len(timeline.outside)} phase lines after the last step's line: a step stopped before its line")
    return found


def signature(timeline: Timeline) -> dict:
    """The log's steps and states, and per phase its lines, its counted work (``COUNTED_EXTRAS``) and its seconds."""
    phases: dict = {}
    for phase in timeline.phases:
        entry = phases.setdefault(phase.name, {"count": 0, "work": 0, "seconds": 0.0})
        entry["count"] += 1
        entry["seconds"] = round(entry["seconds"] + phase.seconds, 1)
        extra = COUNTED_EXTRAS.get(phase.name)
        if extra is not None:
            try:
                entry["work"] += int(phase.extra.get(extra, 0))
            except ValueError:
                pass
    return {
        "steps": [[step.name, step.state] for step in timeline.steps],
        "groups": sum(step.groups for step in timeline.steps),
        "resumed": sum(step.resumed for step in timeline.steps),
        "phases": phases,
    }


def counts(signature: dict) -> dict:
    """A signature without its seconds: what a deterministic fixture repeats exactly."""
    return {
        "steps": signature["steps"], "groups": signature["groups"], "resumed": signature["resumed"],
        "phases": {name: {"count": entry["count"], "work": entry["work"]} for name, entry in signature["phases"].items()},
    }


def report(timeline: Timeline, baseline: dict | None = None) -> str:
    rows = [f"{'step':<12} {'state':<9} {'start':>9} {'seconds':>9} {'unattr. s':>9}  phases inside (count, seconds)"]
    for step in timeline.steps:
        inside: dict = {}
        for phase in step.inside:
            count, seconds = inside.get(phase.name, (0, 0.0))
            inside[phase.name] = (count + 1, seconds + phase.seconds)
        shown = ", ".join(f"{name} {count} {seconds:.1f}" for name, (count, seconds) in inside.items())
        groups = f"; {step.groups} groups fitted, {step.resumed} resumed" if step.name == "fit" else ""
        rows.append(
            f"{step.name:<12} {step.state:<9} {profile_log._offset(step.phase.start):>9} {step.phase.seconds:>9.1f} {step.unattributed:>9.1f}  {shown}{groups}"
        )
    current = signature(timeline)
    rows += ["", f"{'phase':<20} {'count':>6} {'work':>7} {'seconds':>9}" + ("  baseline: count work seconds" if baseline else "")]
    names = list(current["phases"]) + [name for name in (baseline or {}).get("phases", {}) if name not in current["phases"]]
    for name in names:
        entry = current["phases"].get(name, {"count": 0, "work": 0, "seconds": 0.0})
        line = f"{name:<20} {entry['count']:>6} {entry['work']:>7} {entry['seconds']:>9.1f}"
        if baseline:
            was = baseline.get("phases", {}).get(name)
            line += "  (none)" if was is None else f"  {was['count']:>5} {was['work']:>6} {was['seconds']:>9.1f}"
        rows.append(line)
    if baseline:
        same = counts(current) == counts(baseline)
        rows += ["", "counts equal the baseline's" if same else "counts differ from the baseline's (steps, states, groups or a phase's count or work)"]
    found = problems(timeline)
    rows += ["", *(f"problem: {problem}" for problem in found)] if found else ["", "no problem: every step and phase logged"]
    return "\n".join(rows)


def run_fixture(directory: Path) -> str:
    """The fixture (the module's --fixture) in ``directory``, run twice on one run directory; its progress log."""
    import dataclasses
    import os

    from sv_pgs import all_of_us, gram_space, stage2_wiring
    from sv_pgs.workspace_pipeline import (
        WorkspaceConfig,
        run_pipeline,
        workspace_bindings,
    )
    from tests.oracles import fit_at_exponent
    from tests.test_workspace_pipeline import (
        BUDGET,
        DIRECT_SV_NULL,
        _FakeBigQuery,
        _omop_rows,
        _persons,
        _write_inputs,
    )

    assert stage2_wiring.pass_costs is gram_space.pass_costs
    synthetic = _write_inputs(Path(directory) / "inputs")
    client = _FakeBigQuery(*_omop_rows(_persons()))
    raw = dict(
        synthetic.raw, cross_fit=False, direct_sv={**DIRECT_SV_NULL, "gatksv_calls": synthetic.gatksv_pattern},
        panel={"primary_fold": 0, "weights": {name: 1.0 for name in synthetic.raw["diseases"] + synthetic.raw["traits"]}},
    )
    config = WorkspaceConfig.from_mapping(raw)
    bindings = dataclasses.replace(workspace_bindings(), bigquery_client=lambda: client)
    patched = {"GOOGLE_PROJECT": "billing-project", "WORKSPACE_CDR": "aou_workspace.cdr_dataset"}
    saved_environment = {name: os.environ.get(name) for name in patched}
    saved = (all_of_us.fit_occasion_model, stage2_wiring.pass_costs)
    log = io.StringIO()
    os.environ.update(patched)
    # The occasion model at the identity transform (as tests/test_workspace_pipeline.py's dry run), and the pinned route.
    all_of_us.fit_occasion_model = lambda occasions, working_bytes: fit_at_exponent(occasions, 1.0, min(working_bytes, 1 << 28))
    stage2_wiring.pass_costs = lambda band, source, array_module: FIXTURE_ROUTE
    try:
        with contextlib.redirect_stderr(log):
            run_pipeline(config, bindings, BUDGET)
            run_pipeline(config, bindings, BUDGET)
    finally:
        all_of_us.fit_occasion_model, stage2_wiring.pass_costs = saved
        for name, value in saved_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    (Path(directory) / "progress.log").write_text(log.getvalue(), encoding="utf-8")
    return log.getvalue()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("log", nargs="?", type=Path, help="A progress log (progress-task<i>.full.log).")
    source.add_argument("--fixture", type=Path, help="Run the CPU fixture in this new directory and read its log.")
    parser.add_argument("--baseline", type=Path, help="An earlier --save's signature to print beside this one.")
    parser.add_argument("--save", type=Path, help="Write this log's signature here.")
    args = parser.parse_args(argv)
    if args.fixture is not None:
        args.fixture.mkdir(parents=True)
        lines = run_fixture(args.fixture).splitlines()
    else:
        with args.log.open(encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()
    timeline = read(lines)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline is not None else None
    print(report(timeline, baseline))
    if args.save is not None:
        args.save.write_text(json.dumps(signature(timeline), indent=2) + "\n", encoding="utf-8")
    return 1 if problems(timeline) else 0


if __name__ == "__main__":
    sys.exit(main())
