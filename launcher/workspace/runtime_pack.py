"""Pack a Batch job's built environment, run_step.sh's $RUN/venv and $RUN/python, as PARTS tar files of about equal
size beside a COMPLETE file written last, so a later job given SHARED_RUNTIME unpacks them in parallel (run_step.sh's
unpack) instead of extracting env.tar and running uv sync again. A venv is not relocatable: its scripts and pyvenv.cfg
name the paths it was built at, which every job's run disk shares (/mnt/disks/run), and the editable sv-pgs install
names $RUN/source, which every job copies from the staged source before it starts.

One tar read in one stream over the bucket mount is bound by that stream; PARTS of them read at once are not. Every
directory goes in the first part (run_step.sh's tar creates a missing parent anyway), and every file and symlink in the
part with the fewest bytes so far, largest first. It writes only under --into; run_step.sh names a new directory there
for each publishing job, so two jobs publishing at once never write the same file.

  python launcher/workspace/runtime_pack.py --root /mnt/disks/run --into /mnt/disks/ops/svpgs/_runtime/<key>/<job>
"""
from __future__ import annotations

import argparse
import heapq
import os
from pathlib import Path
import subprocess
import tempfile

TREES = ("venv", "python")
"""What run_step.sh builds from env.tar and uv sync: the project's environment and the interpreter it links to."""
PARTS = 8
COMPLETE = "COMPLETE"
"""Written last; its one line is the number of parts, which run_step.sh checks before it unpacks them."""


def members(root: Path) -> tuple[list[str], list[tuple[str, int]]]:
    """The TREES' directories, and their files and symlinks with their sizes (a symlink counts as nothing), relative
    to ``root``, in name order. A symlink to a directory is an entry of its own, never followed."""
    directories, entries = [], []
    for tree in TREES:
        for directory, subdirectories, files in os.walk(root / tree):
            subdirectories.sort()
            relative = Path(directory).relative_to(root)
            directories.append(relative.as_posix())
            for name in sorted(files + [name for name in subdirectories if (Path(directory) / name).is_symlink()]):
                path = Path(directory) / name
                entries.append(((relative / name).as_posix(), 0 if path.is_symlink() else path.stat().st_size))
    return directories, entries


def parts_of(entries: list[tuple[str, int]], parts: int) -> list[list[str]]:
    """The entries in ``parts`` groups of about equal bytes: each entry, largest first, to the lightest group so far."""
    heap = [(0, index) for index in range(parts)]
    groups: list[list[str]] = [[] for _ in range(parts)]
    for name, size in sorted(entries, key=lambda entry: (-entry[1], entry[0])):
        held, index = heapq.heappop(heap)
        groups[index].append(name)
        heapq.heappush(heap, (held + size, index))
    return [sorted(group) for group in groups]


def pack(root: Path, into: Path, parts: int = PARTS) -> None:
    directories, entries = members(root)
    groups = parts_of(entries, parts)
    groups[0] = directories + groups[0]
    into.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as listed:
        running = []
        for index, group in enumerate(groups):
            listing = Path(listed) / f"part-{index}"
            listing.write_text("".join(f"{name}\0" for name in group))
            running.append(subprocess.Popen(
                ["tar", "-cf", str(into / f"part-{index}.tar"), "-C", str(root), "--no-recursion", "--null", "-T", str(listing)]
            ))
        failed = [process.args for process in running if process.wait() != 0]
    if failed:
        raise RuntimeError(f"tar failed: {failed}")
    (into / COMPLETE).write_text(f"{parts}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="The job's run disk, which holds venv/ and python/.")
    parser.add_argument("--into", type=Path, required=True, help="A new directory for the parts and COMPLETE.")
    parser.add_argument("--parts", type=int, default=PARTS)
    args = parser.parse_args()
    pack(args.root, args.into, args.parts)


if __name__ == "__main__":
    main()
