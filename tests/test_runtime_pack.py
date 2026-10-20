"""A job's prebuilt environment (launcher/workspace/runtime_pack.py) packs in parts that run_step.sh's unpack restores
whole: every file's bytes and mode, every symlink as a link, every directory."""
import os
from pathlib import Path
import re
import subprocess

from launcher.workspace import runtime_pack

LAUNCHER = Path(__file__).parents[1] / "launcher/workspace"


def _environment(root: Path) -> None:
    """A venv and its interpreter as uv lays them out: links to the interpreter, packages of very different sizes, an
    empty directory and a linked one."""
    interpreter = root / "python/cpython-3.12-linux-x86_64-gnu/bin/python3.12"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"\x7fELF interpreter")
    interpreter.chmod(0o755)
    packages = root / "venv/lib/python3.12/site-packages"
    (packages / "nvidia/cublas/lib").mkdir(parents=True)
    (packages / "nvidia/cublas/lib/libcublas.so.12").write_bytes(os.urandom(300_000))
    (packages / "nvidia/cusparse/lib").mkdir(parents=True)
    (packages / "nvidia/cusparse/lib/libcusparse.so.12").write_bytes(os.urandom(200_000))
    (packages / "cupy").mkdir()
    for index in range(30):
        (packages / f"cupy/module{index}.py").write_text(f"VALUE = {index}\n")
    (packages / "empty").mkdir()
    (packages / "nvidia/cublas/lib64").symlink_to("lib")
    (root / "venv/bin").mkdir()
    (root / "venv/bin/python").symlink_to(interpreter)
    (root / "venv/bin/python3").symlink_to("python")
    script = root / "venv/bin/sv-pgs"
    script.write_text(f"#!{root}/venv/bin/python\n")
    script.chmod(0o755)
    (root / "venv/pyvenv.cfg").write_text(f"home = {interpreter.parent}\n")


def _tree(root: Path) -> dict[str, tuple]:
    shown = {}
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            shown[name] = ("link", os.readlink(path))
        elif path.is_dir():
            shown[name] = ("directory",)
        else:
            shown[name] = ("file", path.read_bytes(), path.stat().st_mode & 0o777)
    return shown


def test_entries_go_largest_first_to_the_lightest_part() -> None:
    groups = runtime_pack.parts_of([("a", 10), ("b", 9), ("c", 8), ("d", 1), ("e", 1), ("f", 1)], 3)
    sizes = dict([("a", 10), ("b", 9), ("c", 8), ("d", 1), ("e", 1), ("f", 1)])
    assert sorted(sum(sizes[name] for name in group) for group in groups) == [10, 10, 10]
    assert sorted(name for group in groups for name in group) == ["a", "b", "c", "d", "e", "f"]


def test_a_packed_environment_unpacks_whole_with_run_step_s_unpack(tmp_path: Path) -> None:
    built, published, fresh = tmp_path / "built", tmp_path / "published/job", tmp_path / "fresh"
    _environment(built)
    (built / "source").mkdir()
    (built / "source/uv.lock").write_text("not part of the environment\n")
    runtime_pack.pack(built, published, 3)
    assert (published / runtime_pack.COMPLETE).read_text() == "3\n"
    parts = sorted(published.glob("part-*.tar"))
    assert [part.name for part in parts] == ["part-0.tar", "part-1.tar", "part-2.tar"]
    # The two large libraries land in different parts.
    listed = [subprocess.run(["tar", "-tf", str(part)], capture_output=True, text=True, check=True).stdout for part in parts]
    assert sum("libcublas.so.12" in text for text in listed) == 1 and sum("libcusparse.so.12" in text for text in listed) == 1
    assert not any("libcublas.so.12" in text and "libcusparse.so.12" in text for text in listed)
    template = (LAUNCHER / "run_step.sh.template").read_text()
    (unpack,) = re.findall(r"^unpack\(\) \{\n.*?^\}\n", template, re.MULTILINE | re.DOTALL)
    fresh.mkdir()
    unpacked = subprocess.run(["bash", "-c", f'set -uo pipefail\n{unpack}RUN="$1"; unpack "$2"', "-", str(fresh), str(published)], capture_output=True, text=True)
    assert unpacked.returncode == 0, unpacked.stderr
    expected = {name: shown for name, shown in _tree(built).items() if not name.startswith("source")}
    assert _tree(fresh) == expected
    # A publication missing a part is refused, never half unpacked.
    parts[1].unlink()
    refused = subprocess.run(["bash", "-c", f'set -uo pipefail\n{unpack}RUN="$1"; unpack "$2"', "-", str(tmp_path / "again"), str(published)], capture_output=True, text=True)
    assert refused.returncode != 0 and not (tmp_path / "again").exists()
