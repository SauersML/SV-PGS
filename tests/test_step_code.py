"""What keys a step before the fit (``step_code``): the definitions its roots reach, read from a tree's text, without
docstrings, comments, annotations, the log or the trait tables; and on this tree, that every step's roots resolve and
an edit outside what a step runs leaves its digest as it was."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from sv_pgs import step_code, workspace_pipeline
from sv_pgs.step_code import (
    PHENOTYPE_ENTRY_POINTS,
    STEP_ROOTS,
    UNKEYED_TABLES,
    closure,
    code_digests,
    left_out,
)

_PACKAGE = {
    "__init__.py": "",
    "progress.py": "def log(message):\n    print(message)\n",
    "helpers.py": '''"""Helpers."""
from __future__ import annotations

LIMIT = 3


def used(value: Unused) -> int:
    """Its docstring."""
    return value + LIMIT  # a comment


def unused(value):
    return value


class Unused:
    pass
''',
    "lazy.py": "def loaded():\n    return 1\n",
    "dynamic.py": "def first():\n    return 1\n\n\ndef second():\n    return 2\n",
    "tables.py": "ROWS = (1, 2)\n\n\ndef lookup(name):\n    return ROWS[name]\n",
    "step.py": '''import importlib

from sv_pgs import helpers
from sv_pgs.helpers import unused
from sv_pgs.progress import log
from sv_pgs.tables import lookup


def body(run, unused):
    from sv_pgs.lazy import loaded

    log("started")
    importlib.import_module("sv_pgs.dynamic")
    return helpers.used(loaded()) + unused + lookup(0) + [unused for unused in range(2)][0]
''',
}


def _tree(root: Path, **edits: str) -> Path:
    """The small package of _PACKAGE under ``root``, with ``edits`` (module file name to its text) in place."""
    package = root / "sv_pgs"
    package.mkdir(parents=True)
    for name, text in {**_PACKAGE, **edits}.items():
        (package / name).write_text(text)
    return root


def test_a_step_is_keyed_by_what_its_roots_reach_and_nothing_else(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(step_code, "UNKEYED_TABLES", (("sv_pgs.tables", "ROWS"),))
    base = _tree(tmp_path / "base")
    reached = closure(base, ("sv_pgs.step:body",))
    # Its body, the names it loads through the module's imports and its own, a whole module it imports by name, and
    # the constants its callees read; not the log, a table, an annotation's class, a parameter or a comprehension's
    # variable that shares a module name, or a definition nobody names.
    assert set(reached) == {
        "sv_pgs.step:", "sv_pgs.step:body", "sv_pgs.helpers:", "sv_pgs.helpers:used", "sv_pgs.helpers:LIMIT", "sv_pgs.lazy:",
        "sv_pgs.lazy:loaded", "sv_pgs.dynamic:", "sv_pgs.dynamic:first", "sv_pgs.dynamic:second", "sv_pgs.tables:", "sv_pgs.tables:lookup",
    }
    assert "Its docstring" not in reached["sv_pgs.helpers:used"] and "#" not in reached["sv_pgs.helpers:used"]
    digest = code_digests(base, {"step": ("sv_pgs.step:body",)})["step"]

    def digest_with(name: str, old: str, new: str) -> str:
        assert old in _PACKAGE[name]
        tree = _tree(tmp_path / f"edit{len(list(tmp_path.iterdir()))}", **{name: _PACKAGE[name].replace(old, new)})
        return code_digests(tree, {"step": ("sv_pgs.step:body",)})["step"]

    # Prose, the log, a table's rows and code the step never reaches leave the key.
    assert digest_with("helpers.py", "Its docstring.", "Another docstring.") == digest
    assert digest_with("helpers.py", "# a comment", "# another comment") == digest
    assert digest_with("helpers.py", "    return value\n", "    return -value\n") == digest
    assert digest_with("progress.py", "print(message)", "print(message, flush=True)") == digest
    assert digest_with("tables.py", "ROWS = (1, 2)", "ROWS = (1, 2, 3)") == digest
    # What it runs moves it: its callees, the constants they read, a lazily imported or dynamically imported module.
    assert digest_with("helpers.py", "LIMIT = 3", "LIMIT = 4") != digest
    assert digest_with("helpers.py", "value + LIMIT", "value - LIMIT") != digest
    assert digest_with("lazy.py", "return 1", "return 2") != digest
    assert digest_with("dynamic.py", "return 2", "return 3") != digest
    assert digest_with("tables.py", "return ROWS[name]", "return ROWS[-name]") != digest
    # A root or a named definition the tree lacks is no digest.
    with pytest.raises(LookupError, match="defines no"):
        closure(base, ("sv_pgs.step:missing",))
    with pytest.raises(LookupError, match="has no module"):
        closure(base, ("sv_pgs.absent:body",))


def test_every_step_before_the_fit_resolves_on_this_tree_without_the_log_or_the_tables() -> None:
    tree = workspace_pipeline.SOURCE_TREE
    assert tuple(STEP_ROOTS) == workspace_pipeline.STEP_NAMES[: workspace_pipeline.STEP_NAMES.index("fit")]
    assert all(step.code == STEP_ROOTS[step.name] for step in workspace_pipeline._STEPS if step.name in STEP_ROOTS)
    for name, roots in STEP_ROOTS.items():
        reached = closure(tree, roots)
        assert not any(label.startswith("sv_pgs.progress:") for label in reached), name
        assert not any(f"{module}:{table}" in reached for module, table in UNKEYED_TABLES), name
        # A step before the fit never reaches the driver's steps after it, nor the step table.
        assert not any(label in reached for label in ("sv_pgs.workspace_pipeline:_STEPS", "sv_pgs.workspace_pipeline:_fit_step", "sv_pgs.workspace_pipeline:run_pipeline")), name
    # A run of diseases alone never reaches the measurements' occasion model, and one of traits alone the disease query.
    diseases = closure(tree, STEP_ROOTS["phenotypes"], unreached=left_out({"diseases": ["gout"], "traits": []})["phenotypes"])
    traits = closure(tree, STEP_ROOTS["phenotypes"], unreached=left_out({"diseases": [], "traits": ["ldl_cholesterol"]})["phenotypes"])
    assert not any(label in diseases for label in PHENOTYPE_ENTRY_POINTS["traits"]) and not any(label in traits for label in PHENOTYPE_ENTRY_POINTS["diseases"])
    assert not any(label.startswith("sv_pgs.phenotype_measurement:") for label in diseases)
    assert left_out({"diseases": ["gout"], "traits": ["ldl_cholesterol"]}) == {"phenotypes": ()}


def test_an_edit_outside_what_a_step_runs_leaves_its_digest(tmp_path: Path) -> None:
    """On copies of this tree: another disease's table entry and the log move no step; a step's own body moves it and
    the steps whose code reaches it, and nothing else."""
    source = workspace_pipeline.SOURCE_TREE / "sv_pgs"

    def copy(name: str, edits: dict[str, tuple[str, str]]) -> Path:
        shutil.copytree(source, tmp_path / name / "sv_pgs", ignore=shutil.ignore_patterns("__pycache__"))
        for module, (old, new) in edits.items():
            path = tmp_path / name / "sv_pgs" / module
            text = path.read_text()
            assert text.count(old) == 1, (module, old)
            path.write_text(text.replace(old, new))
        return tmp_path / name

    unreached = left_out({"diseases": ["gout"], "traits": []})
    base = code_digests(copy("base", {}), STEP_ROOTS, unreached)
    table = code_digests(copy("table", {
        "all_of_us.py": (
            "DISEASE_DEFINITIONS: tuple[DiseaseDefinition, ...] = (\n",
            'DISEASE_DEFINITIONS: tuple[DiseaseDefinition, ...] = (\n    DiseaseDefinition(canonical_name="another", description="another"),\n',
        ),
        "progress.py": ("\ndef log(", "\ndef _note():\n    return None\n\n\ndef log("),
    }), STEP_ROOTS, unreached)
    assert table == base
    cohort = code_digests(copy("cohort", {
        "workspace_pipeline.py": ("def _cohort_step(run: _Run, directory: Path) -> dict[str, Any]:\n    config = run.config\n",
                                  "def _cohort_step(run: _Run, directory: Path) -> dict[str, Any]:\n    config = run.config\n    config = config\n"),
    }), STEP_ROOTS, unreached)
    assert {name for name in base if cohort[name] != base[name]} == {"cohort"}
