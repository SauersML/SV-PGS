"""The code a pipeline step before the fit executes, read from a source tree's text, so that its key moves only when that
code does.

A step was keyed by the source of the modules and functions it listed (``workspace_pipeline._source_digest``), whole
modules among them, so an edit anywhere in one moved every step that listed it:
- adding Alzheimer's disease to ``all_of_us``'s table moved every run's phenotypes step, and with it the cohort,
  measurement and direct_sv steps after it, for traits whose definitions had not changed;
- a log line's change in ``progress`` moved the measurement and direct_sv steps through ``module_closure``.

A step here is keyed by the definitions it can reach instead. From its roots (``STEP_ROOTS``: the step's body and the
entry points it is handed, ``workspace_pipeline.WorkspaceBindings``) it follows:
- every module-level function, class and assignment of ``sv_pgs`` that a reached definition names, through the
  module's imports, a function's own imports, and ``import_module("sv_pgs.<module>")``, which imports a whole module;
- each reached module's other top-level statements, which run when it is imported.

Three things are not read:
- names in annotations, which the package never evaluates (``from __future__ import annotations``);
- docstrings and comments, so an edit to the prose moves nothing;
- ``UNKEYED_MODULES`` and ``UNKEYED_TABLES``. In a key, a table's place is taken by what the step reads of it: the run's
  own trait definitions (``workspace_pipeline._definition_entries``).

Lines are compared as text with their trailing whitespace, comments and blank lines removed; a multi-line string's lines
are kept as they are. It reads text only, with the standard library, and runs on Python 3.8. So the Workbench app's
python3 can compare two staged trees before a round is submitted (launcher/workspace/step_codes.py --before-fit), and
the driver keys its own tree's steps by the same digests (``workspace_pipeline._run_step``).
"""
from __future__ import annotations

import ast
import hashlib
import io
import tokenize
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

PACKAGE = "sv_pgs"
UNKEYED_MODULES = ("sv_pgs.progress",)
"""Modules no key reads: ``progress`` writes the log (its lines, timers and progress bars), which no step's outputs
depend on."""
UNKEYED_TABLES = (
    ("sv_pgs.all_of_us", "DISEASE_DEFINITIONS"),
    ("sv_pgs.all_of_us", "MEASUREMENT_DEFINITIONS"),
    ("sv_pgs.all_of_us", "LAB_ANALYTE_DEFINITIONS"),
)
"""The trait-definition tables. A run reads only its own traits' entries and the lab measurements they name, and those
entries key the phenotypes step in the tables' place, so an entry added for one study moves no other run's steps."""
_PIPELINE = "sv_pgs.workspace_pipeline"
_MEASUREMENT = "sv_pgs.measurement_model"
STEP_ROOTS = {
    "samples": (f"{_PIPELINE}:_samples_step",),
    "store": (f"{_PIPELINE}:_store_step",),
    "phenotypes": (f"{_PIPELINE}:_phenotypes_step",),
    "cohort": (f"{_PIPELINE}:_cohort_step",),
    # The measurement model's entry points reach these steps through the run's bindings (workspace_bindings), not by name.
    "measurement": (
        f"{_PIPELINE}:_measurement_step", f"{_MEASUREMENT}:calibration_moments", f"{_MEASUREMENT}:concatenate_calibration_moments",
        f"{_MEASUREMENT}:CalibrationPairs", f"{_MEASUREMENT}:fit_measurement_model", f"{_MEASUREMENT}:pooled_measurement_model",
    ),
    "direct_sv": (
        f"{_PIPELINE}:_direct_sv_step", f"{_MEASUREMENT}:fit_measurement_model", f"{_MEASUREMENT}:pooled_measurement_model",
        f"{_MEASUREMENT}:MeasurementModel",
    ),
}
"""Each step before the fit's roots, "module:name" (a definition) or "module" (all of it): its body, and the bindings
it calls (``workspace_pipeline.WorkspaceBindings``) in their production form."""
PHENOTYPE_ENTRY_POINTS = {
    "diseases": (
        "sv_pgs.all_of_us:prepare_all_of_us_disease_sample_table", "sv_pgs.all_of_us:build_all_of_us_disease_sql",
        "sv_pgs.all_of_us:build_all_of_us_disease_query_parameters",
    ),
    "traits": ("sv_pgs.all_of_us:prepare_all_of_us_measurement_sample_table",),
}
"""The phenotypes step's queries for each of the run config's lists, which it calls only for a list with a name in it
(``phenotype_fingerprint`` names the disease query too, in its branch for a disease). A whole-genome run fits one
trait, so a disease's run never runs the measurements' occasion model and a measurement's never runs the disease query
(``left_out``)."""
_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_IMPORTERS = ("import_module", "__import__")


def _target_names(target: ast.expr) -> list[str] | None:
    """The names an assignment target binds, or None where it binds no plain name (an attribute or an item)."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for element in target.elts:
            inner = _target_names(element.value if isinstance(element, ast.Starred) else element)
            if inner is None:
                return None
            names.extend(inner)
        return names
    return None


def _defined(node: ast.stmt) -> list[str] | None:
    """The names a module-level statement defines, or None where it is no definition (an import, a call, a loop)."""
    if isinstance(node, _DEFINITIONS):
        return [node.name]
    if isinstance(node, ast.Assign):
        found = [_target_names(target) for target in node.targets]
        return None if any(names is None for names in found) else [name for names in found for name in names or ()]
    if isinstance(node, ast.AnnAssign):
        return _target_names(node.target)
    return None


def _type_checking(node: ast.stmt) -> bool:
    """Whether ``node`` is an ``if TYPE_CHECKING:`` block, which never runs."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


def _is_string(node: ast.AST) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


class _Module:
    """One module's top-level definitions by name, its imports (each local name to the dotted path it binds) and its
    other top-level statements, over its text."""

    def __init__(self, name: str, path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        self.name = name
        self.package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        self.lines = text.split("\n")
        self.comments: dict[int, int] = {}
        self.inside_strings: set[int] = set()
        starts: list[int] = []
        string_tokens = (tokenize.STRING,)
        f_start, f_end = getattr(tokenize, "FSTRING_START", None), getattr(tokenize, "FSTRING_END", None)
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT:
                self.comments[token.start[0]] = token.start[1]
            elif token.type in string_tokens:
                self.inside_strings.update(range(token.start[0] + 1, token.end[0] + 1))
            elif f_start is not None and token.type == f_start:
                starts.append(token.start[0])
            elif f_end is not None and token.type == f_end and starts:
                self.inside_strings.update(range(starts.pop() + 1, token.end[0] + 1))
        self.definitions: dict[str, ast.stmt] = {}
        self.imports: dict[str, str] = {}
        self.preamble: list[ast.stmt] = []
        self.preamble_names: set[str] = set()
        for node in ast.parse(text).body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self.imports.update(self.bound(node))
                continue
            if _is_string(node) or _type_checking(node):
                continue
            names = _defined(node)
            if names:
                self.definitions.update((defined, node) for defined in names)
                continue
            self.preamble.append(node)
            for inner in ast.walk(node):
                if isinstance(inner, (ast.Import, ast.ImportFrom)):
                    self.imports.update(self.bound(inner))
                elif isinstance(inner, _DEFINITIONS):
                    self.preamble_names.add(inner.name)
                elif isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Store):
                    self.preamble_names.add(inner.id)

    def bound(self, node: ast.Import | ast.ImportFrom) -> dict[str, str]:
        """Each name an import statement binds, to the dotted path it names."""
        if isinstance(node, ast.Import):
            return {alias.asname or alias.name.partition(".")[0]: alias.name if alias.asname else alias.name.partition(".")[0] for alias in node.names}
        base = node.module or ""
        if node.level:
            parts = self.package.split(".")
            parent = ".".join(parts[:len(parts) - node.level + 1])
            base = f"{parent}.{base}" if base else parent
        return {alias.asname or alias.name: f"{base}.{alias.name}" for alias in node.names if alias.name != "*"}

    def text(self, nodes: Sequence[ast.stmt]) -> str:
        """The nodes' lines, decorators included, without their docstrings and string statements, comments, trailing
        whitespace and blank lines; a multi-line string's lines as they are."""
        kept: list[str] = []
        for node in nodes:
            first = min([node.lineno, *(decorator.lineno for decorator in getattr(node, "decorator_list", ()))])
            skipped: set[int] = set()
            for inner in ast.walk(node):
                if not _is_string(inner):
                    continue
                before = self.lines[inner.lineno - 1].encode("utf-8")[:inner.col_offset]
                after = self.lines[inner.end_lineno - 1].encode("utf-8")[inner.end_col_offset:].strip()
                if not before.strip() and (not after or after.startswith(b"#")):
                    skipped.update(range(inner.lineno, inner.end_lineno + 1))
            for number in range(first, node.end_lineno + 1):
                if number in skipped:
                    continue
                line = self.lines[number - 1]
                if number in self.inside_strings:
                    kept.append(line)
                    continue
                column = self.comments.get(number)
                line = (line if column is None else line[:column]).rstrip()
                if line:
                    kept.append(line)
        return "\n".join(kept)


class _Tree:
    """The package's modules in one source tree, each parsed once, on first use."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.modules: dict[str, _Module | None] = {}

    def module(self, name: str) -> _Module | None:
        if name not in self.modules:
            base = self.root.joinpath(*name.split("."))
            path = base.with_name(base.name + ".py")
            if not path.is_file():
                path = base / "__init__.py"
            self.modules[name] = _Module(name, path) if path.is_file() else None
        return self.modules[name]

    def target(self, path: str) -> tuple[str, str | None] | None:
        """The package definition a dotted path names, (module, name), or (module, None) for a whole module; None outside
        the package."""
        if path != PACKAGE and not path.startswith(PACKAGE + "."):
            return None
        parts = path.split(".")
        for end in range(len(parts), 0, -1):
            if self.module(".".join(parts[:end])) is not None:
                return ".".join(parts[:end]), (parts[end] if end < len(parts) else None)
        return None


_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)


def _locals(function: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> set[str]:
    """The names a function binds itself (its parameters, and whatever its body assigns, loops over, catches or
    defines, outside nested functions and comprehensions), which hide the module's names of the same spelling there.
    Its imports are not among them: those name package code (``closure``)."""
    arguments = function.args
    found = {argument.arg for argument in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]}
    found.update(argument.arg for argument in (arguments.vararg, arguments.kwarg) if argument is not None)
    declared: set[str] = set()
    pending: list[ast.AST] = list(function.body) if isinstance(function.body, list) else [function.body]
    while pending:
        item = pending.pop()
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(item.name)
        elif isinstance(item, (ast.Global, ast.Nonlocal)):
            declared.update(item.names)
        elif isinstance(item, ast.Name) and not isinstance(item.ctx, ast.Load):
            found.add(item.id)
        elif isinstance(item, ast.ExceptHandler) and item.name:
            found.add(item.name)
        if not isinstance(item, (*_FUNCTIONS, ast.ClassDef, *_COMPREHENSIONS)):
            pending.extend(ast.iter_child_nodes(item))
    return found - declared


def _references(node: ast.AST) -> Iterator[tuple[str, str]]:
    """What a definition names, outside its annotations and the names its functions bind themselves (``_locals``):
    ("name", a dotted name it loads) and ("path", the module an ``import_module`` or ``__import__`` call names). Its own
    imports are the caller's (``closure``)."""
    pending: list[tuple[ast.AST, frozenset[str]]] = [(node, frozenset())]
    while pending:
        item, hidden = pending.pop()
        if isinstance(item, _FUNCTIONS):
            # Decorators and defaults run where the function is defined; its body sees its own names.
            outer = [*([] if isinstance(item, ast.Lambda) else item.decorator_list), *item.args.defaults]
            outer.extend(default for default in item.args.kw_defaults if default is not None)
            pending.extend((child, hidden) for child in outer)
            inner = hidden | _locals(item)
            pending.extend((child, inner) for child in (item.body if isinstance(item.body, list) else [item.body]))
        elif isinstance(item, _COMPREHENSIONS):
            targets = {name.id for generator in item.generators for name in ast.walk(generator.target) if isinstance(name, ast.Name)}
            pending.extend((child, hidden | targets) for child in ast.iter_child_nodes(item))
        elif isinstance(item, ast.AnnAssign):
            pending.extend((child, hidden) for child in (item.target, item.value) if child is not None)
        elif isinstance(item, ast.arg):
            continue
        elif isinstance(item, ast.Name):
            if isinstance(item.ctx, ast.Load) and item.id not in hidden:
                yield "name", item.id
        elif isinstance(item, ast.Attribute):
            chain = [item.attr]
            base: ast.expr = item.value
            while isinstance(base, ast.Attribute):
                chain.append(base.attr)
                base = base.value
            if isinstance(base, ast.Name):
                if base.id not in hidden:
                    yield "name", ".".join([base.id, *reversed(chain)])
            else:
                pending.append((base, hidden))
        else:
            if isinstance(item, ast.Call) and item.args and isinstance(item.args[0], ast.Constant) and isinstance(item.args[0].value, str):
                called = item.func.attr if isinstance(item.func, ast.Attribute) else getattr(item.func, "id", None)
                if called in _IMPORTERS:
                    yield "path", item.args[0].value
            pending.extend((child, hidden) for child in ast.iter_child_nodes(item))


def left_out(config: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """The definitions a run with this config never calls, by step (``code_digests``): the phenotypes step's query for
    an empty list of the config (its "diseases" or "traits", a run config's own keys)."""
    return {"phenotypes": tuple(point for key, points in PHENOTYPE_ENTRY_POINTS.items() if not config.get(key) for point in points)}


def _root(item: str) -> tuple[str, str | None]:
    module, _, name = item.partition(":")
    return module, name or None


def closure(root: Path, roots: Sequence[str], tree: _Tree | None = None, unreached: Sequence[str] = ()) -> dict[str, str]:
    """Each definition ``roots`` reach in the source tree at ``root`` (the directory holding ``sv_pgs``), as
    "module:name" (names a statement defines together, comma-joined; a module's other top-level statements as
    "module:") to its text (``_Module.text``), past none of ``unreached`` ("module:name", definitions the run never
    calls). A root or a named package definition the tree lacks raises LookupError."""
    tree = tree or _Tree(root)
    found: dict[str, str] = {}
    seen: set[tuple[str, str | None]] = {_root(item) for item in unreached}
    pending = [_root(item) for item in roots]
    while pending:
        reference = pending.pop()
        if reference in seen:
            continue
        seen.add(reference)
        module_name, name = reference
        if module_name in UNKEYED_MODULES or reference in UNKEYED_TABLES:
            continue
        module = tree.module(module_name)
        if module is None:
            raise LookupError(f"{root} has no module {module_name}.")
        if name is None:
            pending.extend((module_name, defined) for defined in module.definitions)
            pending.append((module_name, ""))
            continue
        if name == "" or (name not in module.definitions and name in module.preamble_names):
            nodes, label = module.preamble, f"{module_name}:"
        elif name in module.definitions:
            node = module.definitions[name]
            nodes, label = [node], f"{module_name}:{','.join(_defined(node) or ())}"
            pending.append((module_name, ""))
        elif name in module.imports:
            target = tree.target(module.imports[name])
            if target is not None:
                pending.append(target)
            continue
        else:
            raise LookupError(f"{module_name} in {root} defines no {name}.")
        if label in found:
            continue
        found[label] = module.text(nodes)
        for node in nodes:
            own: dict[str, str] = {}
            for inner in ast.walk(node):
                if inner is not node and isinstance(inner, (ast.Import, ast.ImportFrom)):
                    own.update(module.bound(inner))
            # A definition's own imports run when it does, and bind names that shadow the module's.
            pending.extend(target for target in map(tree.target, own.values()) if target is not None)
            for kind, value in _references(node):
                if kind == "path":
                    target = tree.target(value)
                else:
                    head, _, rest = value.partition(".")
                    if head in own:
                        target = tree.target(f"{own[head]}.{rest}" if rest else own[head])
                    elif head in module.definitions or head in module.preamble_names:
                        target = module_name, head
                    elif head in module.imports:
                        target = tree.target(f"{module.imports[head]}.{rest}" if rest else module.imports[head])
                    else:
                        target = None
                if target is not None:
                    pending.append(target)
    return found


def code_digests(root: Path, steps: Mapping[str, Sequence[str]], unreached: Mapping[str, Sequence[str]] | None = None) -> dict[str, str]:
    """Each step's code digest in the source tree at ``root``: SHA-256 over the definitions its roots reach
    (``closure``) past its ``unreached`` ones (``left_out``), each as its label and text, in label order."""
    tree = _Tree(root)
    digests = {}
    for step, roots in steps.items():
        digest = hashlib.sha256()
        for label, text in sorted(closure(root, roots, tree, (unreached or {}).get(step, ())).items()):
            digest.update(f"{label}\n{text}\n\0".encode())
        digests[step] = digest.hexdigest()
    return digests
