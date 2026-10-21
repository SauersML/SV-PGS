"""The round's end-to-end gate (benchmarks/round_timeline.py): the whole pipeline on the CPU fixture logs every step and
every phase of its work, a rerun reuses every step and does no work, the fixture repeats its counts exactly, and a log
with a step or a phase missing, a phase no rule knows or a reused step that did work is flagged.

The lead runs this file after integrating each patch, and ``python -m benchmarks.round_timeline --fixture DIR --baseline
BASE.json`` for the fixture's phase counts and seconds beside the last integration's."""
from __future__ import annotations

import dataclasses
import json
import re
import shutil
from pathlib import Path

import pytest

from benchmarks import profile_log, round_timeline
from sv_pgs import workspace_pipeline
from sv_pgs.workspace_pipeline import STEP_NAMES, WorkspaceConfig, run_pipeline
from tests.test_workspace_pipeline import (  # noqa: F401
    BUDGET,
    _bindings,
    _Recorder,
    workspace,
)

PACKAGE = Path(__file__).resolve().parents[1] / "sv_pgs"
LAUNCHER = Path(__file__).resolve().parents[1] / "launcher/workspace"


def test_the_gate_knows_every_step_and_every_phase_the_package_logs() -> None:
    """A new timing line elsewhere fails here until the gate has a rule for it, so no phase's time is counted nowhere."""
    assert round_timeline.STEP_NAMES == STEP_NAMES
    named = set()
    for path in PACKAGE.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for first, second in re.findall(r'timing\(\s*"(\w+)"(?:\s+if\s+[^,]+?\s+else\s+"(\w+)")?', text):
            named |= {first, second} - {""}
        for phase in re.findall(r'TimedOracle\([^)]*phase\s*=\s*"(\w+)"', text):
            named |= {phase, f"{phase}_iteration"}
    named |= {"eb", "eb_iteration"}  # TimedOracle's default phase and its rounds
    named |= {f"step_{name}" for name in STEP_NAMES} | {f"{name}_key" for name in STEP_NAMES}  # workspace_pipeline._run_step
    # workspace_pipeline._each's line for each input identity it times, named after the identity function.
    named |= {identity.lstrip("_") for identity in re.findall(r"_each\(\s*(\w+)\s*,", (PACKAGE / "workspace_pipeline.py").read_text(encoding="utf-8"))}
    named |= set(re.findall(r"timing: (\w+) ", (LAUNCHER / "run_step.sh.template").read_text(encoding="utf-8")))
    assert {"stage0", "summary_band", "score_batch", "store_restore", "tree_digest", "fit_key", "file_digest", "genotype_file_fingerprint"} <= named
    assert named <= round_timeline.KNOWN_PHASES, sorted(named - round_timeline.KNOWN_PHASES)


def _fixture(directory: Path) -> tuple[round_timeline.Timeline, str]:
    log = round_timeline.run_fixture(directory)
    return round_timeline.read(log.splitlines()), log


@pytest.fixture(scope="module")
def fixture_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[round_timeline.Timeline, str, Path]:
    """The CPU fixture's two driver runs, once for the module's tests (~35 s in the VM)."""
    directory = tmp_path_factory.mktemp("round") / "first"
    timeline, log = _fixture(directory)
    return timeline, log, directory


def test_the_whole_round_on_the_cpu_fixture_logs_every_step_and_phase(fixture_run) -> None:
    """Samples to export with the production measurement model, engine, artifacts and scorer: each step has its line,
    computed the first time and reused the second with no work inside; each fitted group (two traits, two arms, one
    model each) its own Stage 0, far field, summary band, solves and empirical Bayes loop inside the fit; every model
    scored inside the score step."""
    timeline, log, directory = fixture_run
    assert round_timeline.problems(timeline) == []
    first, second = timeline.drivers()
    assert [step.name for step in first] == [step.name for step in second] == list(STEP_NAMES)
    assert {step.state for step in first} == {"computed"} and {step.state for step in second} == {"reused"}
    # A reused step pays its key and its inputs' identities (round_timeline.KEY_PHASES), nothing else.
    assert all(not step.groups and all(phase.name in round_timeline.KEY_PHASES for phase in step.inside) for step in second)
    assert {phase.name for step in second for phase in step.inside} >= {f"{name}_key" for name in STEP_NAMES}
    fit = first[STEP_NAMES.index("fit")]
    assert fit.groups == 4 and fit.resumed == 0
    names = [phase.name for phase in fit.inside]
    assert all(names.count(name) == 4 for name in round_timeline.GROUP_PHASES)
    assert names.count("eb") >= 4 and names.count("mean_field_solve") >= 4 and "tree_digest" in names
    score = first[STEP_NAMES.index("score")]
    assert sum(int(phase.extra["models"]) for phase in score.inside if phase.name == "score_batch") == 4
    # Nothing outside a step but the job's own restore, which the fixture has none of.
    assert not timeline.outside and not timeline.job
    # The step lines carry their state into profile_log's timeline, and the gate's report reads clean.
    shown = profile_log.report(profile_log.parse(log.splitlines()))
    assert "state=computed" in shown and "state=reused" in shown
    shown = round_timeline.report(timeline)
    assert "no problem" in shown and "training rows" not in shown
    assert (directory / "progress.log").read_text(encoding="utf-8") == log


def test_the_fixture_repeats_its_counts_in_another_directory(fixture_run, tmp_path: Path) -> None:
    """Every step's state, every group and every phase's lines, sweeps, rounds and models, exactly: what a baseline
    signature is compared on after each integration."""
    timeline, _log, _directory = fixture_run
    again, _again_log = _fixture(tmp_path / "second")
    assert round_timeline.counts(round_timeline.signature(again)) == round_timeline.counts(round_timeline.signature(timeline))


def _line(seconds: int, message: str) -> str:
    return f"[2026-10-20 10:{seconds // 60:02d}:{seconds % 60:02d} | 1.0s | 100 MB] {message}"


def _run_log(skip: tuple[str, ...] = (), extra: tuple[str, ...] = (), reused: bool = False) -> list[str]:
    """A driver run's log in the pipeline's line forms: one fitted group, its phases, a scoring batch; ``skip`` drops
    the timing lines of those phases or steps, ``extra`` adds a line before the fit's."""
    lines = []
    for at, name in enumerate(STEP_NAMES, start=1):
        if name == "fit" and not reused:
            lines += [
                _line(at, "stage2 wiring: group 1 of 1: models 0 on 300 training rows"),
                *(_line(at, f"timing: {phase} 0.5 s") for phase in ("tree_digest", "stage0", "far_field", "summary_band") if phase not in skip),
                *(_line(at, "timing: mean_field_solve 0.2 s model=0 outcome=solved sweeps=7") for _ in range(2) if "mean_field_solve" not in skip),
                *(_line(at, line) for line in ("timing: eb_iteration 0.3 s round=1", "timing: eb 0.4 s rounds=1") if "eb" not in skip),
                *(_line(at, line) for line in extra),
            ]
        if name == "score" and not reused and "score_batch" not in skip:
            lines.append(_line(at, "timing: score_batch 0.1 s models=1"))
        # Every step keys itself, a reused one too (the store's inputs with their identities).
        lines += [_line(at, f"timing: {identity} 0.1 s files=2") for identity in ("file_digest", "genotype_file_fingerprint") if name == "store"]
        lines.append(_line(at, f"timing: {name}_key 0.1 s"))
        if f"step_{name}" not in skip:
            lines.append(_line(at, f"timing: step_{name} {3.0 if name == 'fit' else 0.1} s state={'reused' if reused else 'computed'}"))
    return lines


def _problems(lines: list[str]) -> list[str]:
    return round_timeline.problems(round_timeline.read(lines))


def test_a_step_or_phase_missing_an_unknown_phase_and_a_reused_step_that_worked_are_flagged() -> None:
    assert _problems(_run_log()) == [] and _problems(_run_log() + _run_log(reused=True)) == []
    assert any("stage0" in problem for problem in _problems(_run_log(skip=("stage0",))))
    assert any("mean_field_solve" in problem for problem in _problems(_run_log(skip=("mean_field_solve",))))
    assert any("eb lines" in problem for problem in _problems(_run_log(skip=("eb",))))
    assert any("no scoring batch" in problem for problem in _problems(_run_log(skip=("score_batch",))))
    assert any("no line" in problem for problem in _problems(_run_log(skip=("step_cohort",))))
    assert any("KNOWN_PHASES" in problem for problem in _problems(_run_log(extra=("timing: an_unknown_phase 1.0 s",))))
    assert any("another step's phases: score_key" in problem for problem in _problems(_run_log(extra=("timing: score_key 1.0 s",))))
    # A reused step that read the store (its key matched, yet it digested or fitted) did work it should not have.
    worked = _run_log(reused=True)
    worked.insert(len(worked) - 5, _line(30, "timing: tree_digest 9.0 s"))
    assert any("reused but did work" in problem for problem in _problems(worked))
    failed = _run_log()
    failed[-1] = failed[-1].replace("state=computed", "state=failed")
    assert any("export failed" in problem for problem in _problems(failed))
    # A merge's fit finds every arm's models on disk and fits nothing; a fit that filtered an arm for the fitter and
    # logged no group skipped its work.
    merged = [line for line in _run_log() if not any(marker in line for marker in ("stage2 wiring", "stage0", "far_field", "summary_band", "mean_field_solve", "eb"))]
    fit_at = next(index for index, line in enumerate(merged) if "step_fit " in line)
    merged.insert(fit_at, _line(30, "timing: fit_open 0.1 s load=0.1"))
    assert _problems(merged) == []
    merged.insert(fit_at + 1, _line(30, "timing: fit_filter 0.1 s"))
    assert any("no group fitted or resumed" in problem for problem in _problems(merged))
    # A driver that stopped inside a step leaves its phases after the last step's line.
    assert any("after the last step" in problem for problem in _problems(_run_log() + [_line(59, "timing: stage0 1.0 s")]))
    # The job's restore precedes the driver and belongs to no step.
    restored = round_timeline.read([_line(0, "timing: store_restore 60 s")] + _run_log(reused=True))
    assert round_timeline.problems(restored) == [] and [phase.name for phase in restored.job] == ["store_restore"]
    # So are the canary's input generation and each of its processes' whole (benchmarks/wg_canary.py), and its timers
    # around the engine's functions belong inside the fit.
    canary = round_timeline.read(
        [_line(0, "timing: canary_generate 5 s")] + _run_log(extra=("timing: moment_starts 0.1 s", "timing: mode_mixture 0.2 s"))
        + [_line(59, "timing: canary_prepare 60 s gb=1.0")]
    )
    assert round_timeline.problems(canary) == [] and [phase.name for phase in canary.job] == ["canary_generate", "canary_prepare"]


def test_a_region_s_run_begins_at_its_fit() -> None:
    # It neither keys nor runs a step before its fit.
    before = STEP_NAMES[:STEP_NAMES.index("fit")]
    region = [line for line in _run_log() if not any(f"step_{name} " in line or f" {name}_key " in line for name in before)
              and not any(identity in line for identity in ("file_digest", "genotype_file_fingerprint"))]
    timeline = round_timeline.read(_run_log() + region)
    assert round_timeline.problems(timeline) == [] and [run[0].name for run in timeline.drivers()] == ["samples", "fit"]


def test_the_signature_saves_and_reads_back_as_the_baseline(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    log = tmp_path / "progress.log"
    log.write_text("\n".join(_run_log()) + "\n", encoding="utf-8")
    assert round_timeline.main([str(log), "--save", str(tmp_path / "base.json")]) == 0
    saved = json.loads((tmp_path / "base.json").read_text(encoding="utf-8"))
    assert saved["phases"]["mean_field_solve"] == {"count": 2, "work": 14, "seconds": 0.4} and saved["groups"] == 1
    capsys.readouterr()
    assert round_timeline.main([str(log), "--baseline", str(tmp_path / "base.json")]) == 0
    assert "counts equal the baseline's" in capsys.readouterr().out
    fewer = tmp_path / "fewer.log"
    fewer.write_text("\n".join(line.replace("sweeps=7", "sweeps=3") for line in _run_log()) + "\n", encoding="utf-8")
    assert round_timeline.main([str(fewer), "--baseline", str(tmp_path / "base.json")]) == 0
    out = capsys.readouterr().out
    assert "counts differ" in out and re.search(r"mean_field_solve\s+2\s+6\s+0\.4\s+2\s+14", out)
    broken = tmp_path / "broken.log"
    broken.write_text("\n".join(_run_log(skip=("summary_band",))) + "\n", encoding="utf-8")
    assert round_timeline.main([str(broken)]) == 1


def test_a_step_s_line_says_when_it_stopped_after_its_part_or_failed(workspace, capsys: pytest.CaptureFixture[str]) -> None:  # noqa: F811
    """The step lines of runs with the dry run's fitter double: an arm-only run's fit is a part, and so are its score
    and effects (ARM_PARTS), and a fit that raises is failed, and either way the run's steps before it are each logged
    once. A finishing run then computes every step from the fit on, its score from the arms' own scores alone."""
    synthetic, client = workspace
    config = WorkspaceConfig.from_mapping(synthetic.raw)
    run_pipeline(config, _bindings(_Recorder(), client), BUDGET, fit_arms=("snv_indel",))
    timeline = round_timeline.read(capsys.readouterr().err.splitlines())
    assert [(step.name, step.state) for step in timeline.steps][-4:] == [
        ("direct_sv", "computed"), ("fit", "part"), ("score", "part"), ("effects", "part"),
    ]
    assert round_timeline.problems(timeline) == []
    with pytest.raises(RuntimeError, match="preempted"):
        run_pipeline(dataclasses.replace(config), _bindings(_Recorder(fits_before_failing=0), client), BUDGET)
    timeline = round_timeline.read(capsys.readouterr().err.splitlines())
    assert [step.state for step in timeline.steps] == ["reused"] * STEP_NAMES.index("fit") + ["failed"]
    # The other arm's run, on a disk restored from this run's mirror, and this run finishing from its parts.
    from sv_pgs.run_mirror import DirectoryRemote, adopt, push, restore

    root = config.run_directory.parent
    partner, mine, theirs = dataclasses.replace(config, run_directory=root / "partner"), DirectoryRemote(root / "mine"), DirectoryRemote(root / "theirs")
    push(config.run_directory, mine)
    restore(partner.run_directory, mine)
    for step in ("fit", "score", "effects"):
        shutil.rmtree(partner.run_directory / f"{step}.partial")
    run_pipeline(partner, _bindings(_Recorder(), client), BUDGET, fit_arms=("with_sv",))
    push(partner.run_directory, theirs)
    assert adopt(config.run_directory, theirs) is True
    run_pipeline(dataclasses.replace(config), _bindings(_Recorder(fits_before_failing=0), client), BUDGET, finish=True)
    timeline = round_timeline.read(capsys.readouterr().err.splitlines())
    finishing = timeline.drivers()[-1]
    fit = STEP_NAMES.index("fit")
    assert [(step.name, step.state) for step in finishing] == [(name, "reused") for name in STEP_NAMES[:fit]] + [(name, "computed") for name in STEP_NAMES[fit:]]
    score = finishing[STEP_NAMES.index("score")]
    assert [phase.name for phase in score.inside if phase.name in round_timeline.SCORE_PHASES] == ["score_arm", "score_arm"]
    assert round_timeline.problems(timeline) == []
    assert workspace_pipeline._run_step.__doc__ and "round_timeline" in workspace_pipeline._run_step.__doc__
