"""What sv_pgs imports to run must be a runtime dependency, not only a dev or optional one.

numba was in the dev dependency group alone while sv_pgs/mean_field.py imported it at module scope, so a plain
install of the package could not run the small-n route's inference at all: importing it raised ModuleNotFoundError.
An import a module makes unconditionally is a requirement of the package, and this states that as a check.
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys
from importlib import metadata

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 has no tomllib
    tomllib = None

pytestmark = pytest.mark.skipif(tomllib is None, reason="tomllib is in the standard library from Python 3.11")

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Modules no runtime dependency provides: the package itself, and the GPU backend, which every caller loads behind a
# guard and pyproject declares as the optional "gpu" extra.
NOT_REQUIRED = {"sv_pgs", "cupy", "cupyx"}
REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def normalized(name: str) -> str:
    """A distribution name in its comparison form (PyPA core metadata: case-folded, runs of -_. as one -)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def declared(section) -> set:
    return {normalized(REQUIREMENT_NAME.match(requirement).group()) for requirement in section}


def manifest() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def module_scope_imports(path: pathlib.Path):
    """The top-level modules a file imports unconditionally. Imports inside a try, a function or a guard are not
    statements of the module body, so they are not counted: only what importing the file always needs."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Import):
            yield from (alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            yield node.module.split(".")[0]


def test_every_module_sv_pgs_imports_unconditionally_is_a_runtime_dependency():
    runtime = declared(manifest()["project"]["dependencies"])
    provided = metadata.packages_distributions()
    outside = {}
    for path in sorted((ROOT / "sv_pgs").rglob("*.py")):
        for module in module_scope_imports(path):
            if module in sys.stdlib_module_names or module in NOT_REQUIRED:
                continue
            distributions = provided.get(module)
            # A module this environment has not installed cannot be traced to the distribution that provides it.
            if distributions and not any(normalized(name) in runtime for name in distributions):
                outside[module] = (str(path.relative_to(ROOT)), sorted(distributions))
    assert outside == {}


def test_numba_is_a_runtime_dependency_because_the_mean_field_route_imports_it():
    assert "numba" in set(module_scope_imports(ROOT / "sv_pgs" / "mean_field.py"))
    pyproject = manifest()
    assert "numba" in declared(pyproject["project"]["dependencies"])
    for name, group in pyproject.get("dependency-groups", {}).items():
        assert "numba" not in declared(group), f"numba is a runtime dependency, so the {name} group must not repeat it"
