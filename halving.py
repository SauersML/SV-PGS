"""The outer loop's halving fix (``halving.patch``, ag-eb's diff of 793e4db's ``sv_pgs/scale_mixture_ep.py``) applied in
memory, so the staged file, and every key that reads it, stays as it is.

A refused joint trial of ``scale_mixture_ep.fit_hyperparameters`` is halved at rho-hat: its weights stay at the step's
end, so as the fraction falls the trial tends to (x_k, rho-hat), not to the certified state. Where V's formula has no
certified value there, no halving has one, and a polished state's step halves on to x's resolution: on the All of Us
whole-genome arms about 26 halvings of 45 minutes each [real], and a retry resumes into the same loop. The fix moves
the finite weights by the fraction too, once a halving at rho-hat has no certified value.

Editing the file would move the phenotypes key (its step lists the whole module) and the fit key (``module_closure``),
and ``artifact.code_digest``, which keys the engine's checkpoints and this package's units. So ``install`` reads the
file, refuses unless its bytes are 793e4db's (``SOURCE_MD5``), applies the patch to the text (every hunk at its stated
line, or nothing), checks the result is ag-eb's patched file (``PATCHED_MD5``), and executes only the top-level
definitions the patch changes (``_OuterTrial`` and ``fit_hyperparameters``) in the module's own namespace: every other
object, its classes among them, stays the one already imported. Each module that imported a changed definition by name
(``full_data_fit``, ``small_n``) is rebound to it; a module imported later reads the patched one. The new definitions'
source is registered under a name of its own (``FILENAME``), so a traceback shows the patched lines, while the module's
own source, which the keys read, is the file's."""
from __future__ import annotations

import __future__
import ast
import hashlib
import linecache
import re
import sys
from pathlib import Path

from sv_pgs.progress import log

PATCH = Path(__file__).with_name("halving.patch")
SOURCE_MD5 = "6fa293fb9abafb005012eee664830e71"
"""``sv_pgs/scale_mixture_ep.py`` at 793e4db, as staged in every live run."""
PATCHED_MD5 = "4f224a5acb3a0819491035a8dd04041d"
"""The md5 of ag-eb's patched file: what ``PATCH`` must make of the staged one."""
FILENAME = "<live resume halving fix: sv_pgs/scale_mixture_ep.py>"
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_INSTALLED: list = []
"""What ``install`` replaced, as (namespace, name, previous value or ``_ABSENT``), for ``uninstall``."""
_ABSENT = object()


def apply(text: str, patch: str) -> str:
    """``patch``, a unified diff of one file, applied to ``text``: each hunk's old lines must be at its stated line, as
    written; otherwise ValueError."""
    source = text.splitlines(keepends=True)
    result: list[str] = []
    position = 0
    lines = patch.splitlines()
    index = 0
    while index < len(lines) and not lines[index].startswith("@@"):
        index += 1
    if index == len(lines):
        raise ValueError("the patch has no hunk")
    while index < len(lines):
        header = _HUNK.match(lines[index])
        if header is None:
            raise ValueError(f"the patch's line {index + 1} is not a hunk header: {lines[index][:60]!r}")
        start, old_count, new_count = int(header[1]), int(header[2] or 1), int(header[4] or 1)
        old: list[str] = []
        new: list[str] = []
        index += 1
        while index < len(lines) and not lines[index].startswith("@@"):
            line = lines[index]
            if line.startswith("\\"):
                raise ValueError("the patch changes the file's last line ending")
            kind, body = (line[0], line[1:]) if line else (" ", "")
            if kind not in " -+":
                raise ValueError(f"the patch's line {index + 1} is not a hunk line: {line[:60]!r}")
            if kind in " -":
                old.append(body)
            if kind in " +":
                new.append(body)
            index += 1
        if len(old) != old_count or len(new) != new_count:
            raise ValueError(f"the hunk at line {start} holds {len(old)} old and {len(new)} new lines, not {old_count} and {new_count}")
        first = start - 1
        if first < position or [line.rstrip("\n") for line in source[first:first + len(old)]] != old:
            raise ValueError(f"the hunk at line {start} does not match the file there")
        result.extend(source[position:first])
        result.extend(f"{line}\n" for line in new)
        position = first + len(old)
    result.extend(source[position:])
    return "".join(result)


def _statements(text: str) -> tuple[dict[str, tuple[int, str]], list[str]]:
    """``text``'s top-level definitions by name (functions, classes and single-name assignments), each with its first
    line and its source as written, decorators included; and its other top-level statements' source."""
    lines = text.splitlines(keepends=True)
    named: dict[str, tuple[int, str]] = {}
    other = []
    for node in ast.parse(text).body:
        first = min([node.lineno, *(decorator.lineno for decorator in getattr(node, "decorator_list", []))])
        segment = "".join(lines[first - 1:node.end_lineno])
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        else:
            other.append(segment)
            continue
        if name in named:
            raise ValueError(f"{name} is defined twice at the top level")
        named[name] = (first, segment)
    return named, other


def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _changes(text: str, patched: str) -> list[tuple[str, int, str]]:
    """The top-level definitions ``patched`` adds or changes, in its order, each with its first line and source; ValueError
    where it changes anything else (another statement, or a definition it removes)."""
    before, before_other = _statements(text)
    after, after_other = _statements(patched)
    if before_other != after_other:
        raise ValueError("the patch changes a top-level statement that is not a definition")
    if set(before) - set(after):
        raise ValueError(f"the patch removes {sorted(set(before) - set(after))}")
    changed = [(name, first, segment) for name, (first, segment) in after.items() if before.get(name, (0, None))[1] != segment]
    if not changed:
        raise ValueError("the patch changes no definition")
    return sorted(changed, key=lambda entry: entry[1])


def install() -> bool:
    """Put the patched definitions in place (the module docstring) and log one line; False, with a line saying why,
    where the staged file is not 793e4db's or the patch does not make ag-eb's file (nothing is then changed)."""
    from sv_pgs import scale_mixture_ep

    if _INSTALLED:
        return True
    try:
        text = Path(scale_mixture_ep.__file__).read_bytes().decode()
        if _md5(text) != SOURCE_MD5:
            raise ValueError(f"sv_pgs/scale_mixture_ep.py is not 793e4db's (md5 {_md5(text)[:12]}, not {SOURCE_MD5[:12]})")
        patch = PATCH.read_text(encoding="utf-8")
        patched = apply(text, patch)
        if _md5(patched) != PATCHED_MD5:
            raise ValueError(f"the patched file's md5 is {_md5(patched)[:12]}, not ag-eb's {PATCHED_MD5}")
        changes = _changes(text, patched)
        flags = 0
        for value in vars(scale_mixture_ep).values():
            if isinstance(value, __future__._Feature):
                flags |= value.compiler_flag
        # Each definition compiled at its line in the patched file, so its code and tracebacks point at those lines.
        codes = [(name, compile("\n" * (first - 1) + segment, FILENAME, "exec", flags=flags, dont_inherit=True)) for name, first, segment in changes]
    except Exception as error:
        log(f"live resume: halving fix not installed ({type(error).__name__}: {error}); EB runs unpatched")
        return False
    namespace = vars(scale_mixture_ep)
    try:
        previous = {name: namespace.get(name, _ABSENT) for name, _code in codes}
        for name, code in codes:
            _INSTALLED.append((namespace, name, previous[name]))
            exec(code, namespace)
        rebound = []
        for module_name, module in list(sys.modules.items()):
            if module is scale_mixture_ep or not (module_name == "sv_pgs" or module_name.startswith("sv_pgs.")):
                continue
            held = getattr(module, "__dict__", None)
            for name, value in list(held.items()) if isinstance(held, dict) else []:
                for changed, old in previous.items():
                    if old is not _ABSENT and value is old:
                        _INSTALLED.append((held, name, old))
                        held[name] = namespace[changed]
                        rebound.append(f"{module_name}.{name}")
        linecache.cache[FILENAME] = (len(patched), None, patched.splitlines(keepends=True), FILENAME)
    except Exception as error:
        uninstall()
        log(f"live resume: halving fix not installed ({type(error).__name__}: {error}); EB runs unpatched")
        return False
    log(f"live resume: halving fix installed ({_md5(patch)}): {', '.join(name for name, _code in codes)} of sv_pgs.scale_mixture_ep"
        f"{'; rebound ' + ', '.join(sorted(rebound)) if rebound else ''}")
    return True


def uninstall() -> None:
    """Put back every replaced definition and binding (tests)."""
    while _INSTALLED:
        namespace, name, value = _INSTALLED.pop()
        if value is _ABSENT:
            namespace.pop(name, None)
        else:
            namespace[name] = value
    linecache.cache.pop(FILENAME, None)


def installed() -> bool:
    return bool(_INSTALLED)


def patched_source() -> str:
    """The patched file's text, as ``install`` makes it (tests)."""
    from sv_pgs import scale_mixture_ep

    return apply(Path(scale_mixture_ep.__file__).read_text(encoding="utf-8"), PATCH.read_text(encoding="utf-8"))
