"""SPEC: a preemption costs time, never a different result.

Every step of the workspace pipeline is stopped part way through, at a call inside its own work, and the run is
started again, on the same disk or on a new one restored from the run mirror (as each Batch job is). The restarted
run's outputs are then those of one uninterrupted run, byte for byte: every file of every step, with gzip's tables
compared by their bytes, not their wall-clock header, NPZ archives by their arrays, and each step's completion record
by its key and summary, not its time.

Each case also counts the calls the restart makes again, which states what each step keeps when it stops: the
store and direct_sv steps keep each chromosome they finished, the fit each arm, score and effects each batch; the
samples, phenotypes, cohort, measurement, report and export steps keep nothing and start again (minutes at the All
of Us workload but for measurement's store rewrite, which ``workspace_pipeline._measurement_step`` keeps whole).
"""
from __future__ import annotations

import dataclasses
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from sv_pgs import workspace_pipeline
from sv_pgs.run_mirror import DirectoryRemote, push, restore
from sv_pgs.workspace_pipeline import COMPLETE_FILE, STEP_NAMES, WorkspaceConfig, run_pipeline
from tests.test_workspace_pipeline import DIRECT_SV_NULL, BUDGET, _bindings, _Recorder, _single_model_batches, workspace  # noqa: F401 (a fixture)

BOOKKEEPING = ("batches",)
"""The score and effects steps' committed batches (``postfit_checkpoint``): restart state whose records carry a fresh
generation each time they are written, not an output; the step's outputs are assembled from them and compared."""
HANDOFF = ("fit_key", "model_files")
"""What the mirror adds to a fit step's model record as it pushes it (``run_mirror._push_partial``): the key the model
was fitted under and its files' sizes, so an arm handed to another run carries them. The models themselves, and every
other field of the record, are compared byte for byte."""


class _Preempted(RuntimeError):
    pass


def _outputs(run: Path) -> dict[str, object]:
    """Every file of every step: its bytes, a gzip table's content, an NPZ's arrays, a completion record's key and
    summary."""
    found: dict[str, object] = {}
    for step in STEP_NAMES:
        for path in sorted((run / step).rglob("*")):
            relative = path.relative_to(run)
            if not path.is_file() or relative.parts[1] in BOOKKEEPING:
                continue
            if path.name == COMPLETE_FILE:
                record = json.loads(path.read_text(encoding="utf-8"))
                found[relative.as_posix()] = (record["key"], json.dumps(record["summary"], sort_keys=True))
            elif relative.match("fit/models/*/*.json"):
                record = json.loads(path.read_text(encoding="utf-8"))
                found[relative.as_posix()] = json.dumps({name: value for name, value in record.items() if name not in HANDOFF}, sort_keys=True)
            elif path.suffix == ".npz":
                with np.load(path, allow_pickle=False) as arrays:
                    found[relative.as_posix()] = {name: arrays[name] for name in arrays.files}
            elif path.suffix == ".gz":
                found[relative.as_posix()] = gzip.decompress(path.read_bytes())
            else:
                found[relative.as_posix()] = path.read_bytes()
    return found


def _assert_same(actual: dict[str, object], expected: dict[str, object]) -> None:
    assert sorted(actual) == sorted(expected)
    for name, value in expected.items():
        if isinstance(value, dict):
            assert sorted(actual[name]) == sorted(value), name
            for array, values in value.items():
                np.testing.assert_array_equal(actual[name][array], values, err_msg=f"{name}:{array}")
        else:
            assert actual[name] == value, name


# (step, where it stops: a workspace_pipeline function or a binding, the calls that finish before it stops, whether
# the step keeps them: the restart makes again only the calls it does not keep)
STOPS = [
    ("samples", "_calibration_pairs", 0, False),
    ("store", "_convert_chromosome", 1, True),
    ("phenotypes", "prepare_all_of_us_measurement_sample_table", 0, False),
    ("cohort", "development_calibration", 0, False),
    ("measurement", "assemble_half", 1, False),
    ("direct_sv", "_direct_chromosome", 1, True),
    ("fit", "bindings.fit", 1, True),
    ("score", "bindings.predict", 1, True),
    ("effects", "variant_effects", 1, True),
    ("report", "_paired", 1, False),
    ("export", "check_exportable", 1, False),
]


@pytest.mark.parametrize("mirrored", [True, False], ids=["new-disk", "same-disk"])
@pytest.mark.parametrize("step,where,finished,kept", STOPS, ids=[stop[0] for stop in STOPS])
def test_a_step_stopped_part_way_and_started_again_makes_an_uninterrupted_run_s_outputs(
    workspace, monkeypatch, tmp_path, request, step, where, finished, kept, mirrored  # noqa: F811 (the fixture)
) -> None:
    if step == "direct_sv" and mirrored:
        # The mirror carries a finished chromosome's record and its store files but not its direct rows,
        # direct_sv.partial/rows/<chromosome>.npz, so the restarted step cannot finish (FileNotFoundError).
        request.applymarker(pytest.mark.xfail(strict=True, raises=FileNotFoundError, reason="run_mirror.STORE_CHROMOSOME_FILES lacks direct_sv's rows"))
    synthetic, client = workspace
    # With a GATK-SV call set, so the direct_sv step has chromosomes to merge and with_sv fits its rows.
    config = WorkspaceConfig.from_mapping({
        **synthetic.raw, "calibration_fraction": .3, "annotated_prior": False,
        "direct_sv": {**DIRECT_SV_NULL, "gatksv_calls": synthetic.gatksv_pattern},
    })
    # Several score batches, as the whole-genome workload's memory makes, so the score step has batches to keep.
    monkeypatch.setattr(workspace_pipeline, "score_batches", _single_model_batches)
    run = config.run_directory
    calls: list[int] = []
    stop_at: list[int | None] = [None]

    def counted(original):
        def call(*args, **kwargs):
            if stop_at[0] is not None and len(calls) == stop_at[0]:
                raise _Preempted(f"preempted in {step} after {len(calls)} calls of {where}")
            calls.append(1)
            return original(*args, **kwargs)

        return call

    def bindings():
        made = _bindings(_Recorder(), client)
        if where.startswith("bindings."):
            name = where.removeprefix("bindings.")
            made = dataclasses.replace(made, **{name: counted(getattr(made, name))})
        return made

    if not where.startswith("bindings."):
        monkeypatch.setattr(workspace_pipeline, where, counted(getattr(workspace_pipeline, where)))

    def through(name: str) -> None:
        run_pipeline(config, bindings(), BUDGET, through=name)

    # The calls are counted in this step alone: a function may serve another step too (assemble_half the store's).
    previous = STEP_NAMES[STEP_NAMES.index(step) - 1] if STEP_NAMES.index(step) else None
    if previous is not None:
        through(previous)
    calls.clear()
    through(step)
    total = len(calls)
    assert total > finished, f"{where} is called {total} times in {step}"
    through(STEP_NAMES[-1])
    expected = _outputs(run)
    shutil.rmtree(run)
    if previous is not None:
        through(previous)
    calls.clear()
    stop_at[0] = finished
    with pytest.raises(_Preempted):
        through(step)
    assert len(calls) == finished and not (run / step / COMPLETE_FILE).exists()
    if mirrored:
        remote = DirectoryRemote(tmp_path / "mirror")
        push(run, remote)
        shutil.rmtree(run)
        restore(run, remote)
    calls.clear()
    stop_at[0] = None
    through(step)
    assert len(calls) == (total - finished if kept else total)
    through(STEP_NAMES[-1])
    _assert_same(_outputs(run), expected)
