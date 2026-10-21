"""Which pipeline steps' code differs between two source trees: the "code" part of each step's key
(``sv_pgs.workspace_pipeline._run_step``), so a tree is staged into a live run only where that keeps the keys its jobs
were started under. It reads the two trees and writes nothing.

A run's every job start, a Batch retry of a preempted task included, copies the run's staged source ($OPS/source,
run_step.sh.template) and keys its steps by that copy. So a source restaged into a run whose fit is still running
gives its next retry another fit key wherever the fit step's code moved: the retry starts its fit again from the
start, and its models no longer merge with the other arm's (a model record of another fit key is refused by the mirror,
``run_mirror``, and by the fit step's handoff check, ``workspace_pipeline._check_fit_records``). A change outside the
fit step's code (its functions and every ``sv_pgs`` module ``fit_model`` imports, ``module_closure``) leaves the fit key
as it was: post-fit fixes to the score, effects or report steps were restaged into live runs this way. Each tree is read
by its own Python process, which imports that tree's ``sv_pgs``, so the environment that runs this needs sv-pgs's
dependencies (the VM's, or the job's).

  python launcher/workspace/step_codes.py <staged source tree> <tree to stage>

It prints each step, "same" or "differs", and the two code digests' first 16 hex digits, and exits 1 where the fit
step's code differs (restaging would cost every running arm's retry its fit), else 0.

With --before-fit it compares the steps before the fit alone, as today's driver keys them (``sv_pgs.step_code``: what
each step's roots reach), reading both trees' text with this tree's step_code.py and importing nothing, so the
Workbench app's python3 (3.8 or later, no sv-pgs environment) runs it. A round planner uses it to tell whether a new
source keeps an earlier run's steps before the fit (its run config's accept_steps_from; the driver decides from the
run's inputs too, ``workspace_pipeline._accepted``). --config names a run config, whose lists of diseases and traits
leave out the phenotype queries the run never runs (``step_code.left_out``); without it both are compared.

  python3 launcher/workspace/step_codes.py --before-fit [--config run_config.json] <earlier source tree> <new source tree>

It exits 1 where any of them differs, else 0.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

FIT = "fit"
_DIGESTS = """
import json
from sv_pgs import workspace_pipeline as pipeline


def code(step):
    # A step keyed by its roots' reach (sv_pgs.step_code), or by its listed code's source.
    if step.code and all(isinstance(item, str) for item in step.code):
        from sv_pgs.step_code import code_digests

        return code_digests(pipeline.SOURCE_TREE, {step.name: step.code})[step.name]
    return pipeline._source_digest(step.code)


print(json.dumps({step.name: code(step) for step in pipeline._STEPS}))
"""
_SHOWN = 16
"""Hex digits of a digest shown: 64 bits, enough to tell two trees' digests apart by eye."""


def step_codes(tree: Path) -> dict[str, str]:
    """Each step's code digest in ``tree`` (``workspace_pipeline._source_digest`` of the step's code, or for a step
    before the fit in a tree that keys it by its roots, ``step_code.code_digests``), by its name. The
    import writes no bytecode, and numba's compiled-function cache (``cache=True``, beside each module by default) goes
    to a directory removed afterwards, so the tree is left as it was."""
    with tempfile.TemporaryDirectory() as cache:
        environment = {**os.environ, "PYTHONPATH": str(tree), "PYTHONDONTWRITEBYTECODE": "1", "NUMBA_CACHE_DIR": cache}
        shown = subprocess.run([sys.executable, "-c", _DIGESTS], cwd=tree, env=environment, capture_output=True, check=True, text=True)
    return json.loads(shown.stdout.strip().splitlines()[-1])


def compare(staged: Path, candidate: Path) -> tuple[list[str], list[str]]:
    """The lines to print, a step a line, and the steps whose code differs between the two trees."""
    first, second = step_codes(staged), step_codes(candidate)
    names = list(first) + [name for name in second if name not in first]
    changed = [name for name in names if first.get(name) != second.get(name)]
    lines = [
        f"{name}\t{'differs' if name in changed else 'same'}\t{first.get(name, '-')[:_SHOWN]}\t{second.get(name, '-')[:_SHOWN]}" for name in names
    ]
    return lines, changed


def _step_code():
    """This tree's sv_pgs/step_code.py, loaded from its file alone (it imports only the standard library)."""
    path = Path(__file__).resolve().parents[2] / "sv_pgs" / "step_code.py"
    spec = importlib.util.spec_from_file_location("_step_code", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_before_fit(earlier: Path, candidate: Path, config: dict | None = None) -> tuple[list[str], list[str]]:
    """The lines to print, a step a line, and the steps before the fit whose code differs between the two trees by
    today's formula (``step_code.code_digests``); a tree that lacks a step's root differs there."""
    step_code = _step_code()
    unreached = step_code.left_out(config) if config is not None else None
    digests = []
    for tree in (earlier, candidate):
        found = {}
        for name, roots in step_code.STEP_ROOTS.items():
            try:
                found.update(step_code.code_digests(tree, {name: roots}, unreached))
            except (LookupError, OSError, SyntaxError) as error:
                found[name] = f"missing ({error})"
        digests.append(found)
    first, second = digests
    changed = [name for name in step_code.STEP_ROOTS if first[name] != second[name]]
    lines = [f"{name}\t{'differs' if name in changed else 'same'}\t{first[name][:_SHOWN]}\t{second[name][:_SHOWN]}" for name in step_code.STEP_ROOTS]
    return lines, changed


def main(arguments: list[str]) -> int:
    if arguments[:1] == ["--before-fit"]:
        config = None
        rest = arguments[1:]
        if rest[:1] == ["--config"] and len(rest) > 1:
            config = json.loads(Path(rest[1]).read_text(encoding="utf-8"))
            rest = rest[2:]
        if len(rest) != 2:
            print(__doc__, file=sys.stderr)
            return 2
        lines, changed = compare_before_fit(*(Path(argument).resolve() for argument in rest), config)
        print("\n".join(lines))
        return 1 if changed else 0
    if len(arguments) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    lines, changed = compare(*(Path(argument).resolve() for argument in arguments))
    print("\n".join(lines))
    if FIT in changed:
        print("the fit step's code differs: staging this tree into a run whose fit is running gives its retries another fit key", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
