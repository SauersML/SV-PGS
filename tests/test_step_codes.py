"""The step-code comparison of two source trees (launcher/workspace/step_codes.py): a change outside the fit step's
code keeps the fit key a live run's retries need, and a change to a module the fit imports does not; before the fit, a
tree is compared by what each step runs."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from launcher.workspace import step_codes
from sv_pgs.step_code import STEP_ROOTS

PACKAGE = Path(__file__).resolve().parents[1] / "sv_pgs"


def _tree(root: Path) -> Path:
    shutil.copytree(PACKAGE, root / "sv_pgs", ignore=shutil.ignore_patterns("__pycache__"))
    return root


def test_only_a_change_to_the_fit_step_s_code_moves_the_fit_key(tmp_path: Path) -> None:
    staged, candidate = _tree(tmp_path / "staged"), _tree(tmp_path / "candidate")
    # A report-only module: the report step's code moves, the fit step's does not.
    with open(candidate / "sv_pgs" / "held_out_comparison.py", "a", encoding="utf-8") as handle:
        handle.write("\n# a post-fit change\n")
    lines, changed = step_codes.compare(staged, candidate)
    assert "report" in changed and step_codes.FIT not in changed
    assert any(line.startswith("fit\tsame\t") for line in lines)
    assert step_codes.main([str(staged), str(candidate)]) == 0
    # A module the fit imports (the engine's wiring): the fit step's code moves.
    with open(candidate / "sv_pgs" / "stage2_wiring.py", "a", encoding="utf-8") as handle:
        handle.write("\n# an engine change\n")
    _lines, changed = step_codes.compare(staged, candidate)
    assert step_codes.FIT in changed
    assert step_codes.main([str(staged), str(candidate)]) == 1
    assert not list(staged.rglob("__pycache__"))


def test_before_the_fit_two_trees_differ_only_where_a_step_runs_other_code(tmp_path: Path) -> None:
    earlier, candidate = _tree(tmp_path / "earlier"), _tree(tmp_path / "candidate")
    # The engine and the log: no step before the fit runs either.
    for name in ("stage2_wiring.py", "progress.py"):
        with open(candidate / "sv_pgs" / name, "a", encoding="utf-8") as handle:
            handle.write("\n\ndef _added():\n    return None\n")
    lines, changed = step_codes.compare_before_fit(earlier, candidate)
    assert changed == [] and len(lines) == 6 and step_codes.main(["--before-fit", str(earlier), str(candidate)]) == 0
    # The measurements' query: only the phenotypes step runs it, and only a run with traits does.
    path = candidate / "sv_pgs" / "all_of_us.py"
    text = path.read_text()
    line = "    definition = resolve_measurement_definition(trait)\n    rows = fetch_all_of_us_measurement_rows(definition, client=client)\n"
    assert text.count(line) == 1
    path.write_text(text.replace(line, line + "    rows = list(rows)\n"))
    assert step_codes.compare_before_fit(earlier, candidate)[1] == ["phenotypes"]
    diseases = tmp_path / "diseases.json"
    diseases.write_text(json.dumps({"diseases": ["gout"], "traits": []}))
    assert step_codes.compare_before_fit(earlier, candidate, {"diseases": ["gout"], "traits": []})[1] == []
    assert step_codes.main(["--before-fit", "--config", str(diseases), str(earlier), str(candidate)]) == 0
    assert step_codes.main(["--before-fit", str(earlier), str(candidate)]) == 1
    # A tree without a step's body differs there.
    shutil.rmtree(earlier / "sv_pgs")
    assert set(step_codes.compare_before_fit(earlier, candidate)[1]) == set(STEP_ROOTS)
