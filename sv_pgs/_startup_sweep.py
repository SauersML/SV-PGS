"""Aggressive startup-time sweep of stale sv-pgs siblings.

When a prior sv-pgs run is killed mid-fit (Ctrl-C from a notebook tab,
SIGSTOP from a debugger, OOM-killer truncating one child of a multi-proc
job, etc.) the Python process can persist holding many GB of host RAM
and a CUDA context. The bash sweep in run.sh only sometimes catches
these; this Python-side sweep is the belt-and-suspenders pass.

Runs at the very top of cli.main(), before any heavy imports allocate
memory, so the auto-tuner downstream sees the real free RAM.
"""
from __future__ import annotations

import os
import signal
import sys
import time


_PROTECTED_MARKERS = ("claude", "codex")


def _read_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return ""


def _read_status(pid: int) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                key, _, value = line.partition(":")
                result[key.strip()] = value.strip()
    except OSError:
        pass
    return result


def _proc_uid(pid: int) -> int | None:
    status = _read_status(pid)
    uid_field = status.get("Uid", "")
    parts = uid_field.split()
    if not parts:
        return None
    try:
        return int(parts[0])
    except ValueError:
        return None


def _ppid(pid: int) -> int:
    status = _read_status(pid)
    try:
        return int(status.get("PPid", "0"))
    except ValueError:
        return 0


def _ancestor_chain(pid: int) -> set[int]:
    chain = {pid}
    cur = pid
    while cur > 1:
        cur = _ppid(cur)
        if cur in chain or cur <= 0:
            break
        chain.add(cur)
    return chain


def _rss_kb(pid: int) -> int:
    status = _read_status(pid)
    raw = status.get("VmRSS", "0 kB").split()
    try:
        return int(raw[0])
    except (ValueError, IndexError):
        return 0


def _iter_pids() -> list[int]:
    try:
        entries = os.listdir("/proc")
    except OSError:
        return []
    pids = []
    for name in entries:
        if name.isdigit():
            pids.append(int(name))
    return pids


def _matches_aou_run(cmdline: str) -> bool:
    if not cmdline:
        return False
    lowered = cmdline.lower()
    if any(marker in lowered for marker in _PROTECTED_MARKERS):
        return False
    if "run-all-of-us" in cmdline:
        return True
    return "sv-pgs" in cmdline and "run-all-of-us" in cmdline


def kill_stale_sv_pgs_siblings(verbose: bool = True) -> int:
    """Find and kill stale sv-pgs / sv_pgs Python processes owned by us.

    Safety rules:
      - never kill self or any ancestor (would SIGKILL the shell / jupyter)
      - never kill a process owned by a different uid
      - never kill Claude/Codex processes
      - only kill stale AoU `sv-pgs run-all-of-us` processes

    Returns the number of processes killed. Best-effort: errors are swallowed
    so a hostile /proc layout never blocks startup.
    """
    if sys.platform != "linux":
        return 0
    try:
        my_uid = os.getuid()
    except AttributeError:
        return 0
    my_pid = os.getpid()
    ancestors = _ancestor_chain(my_pid)
    candidates: list[tuple[int, int, str]] = []
    for pid in _iter_pids():
        if pid in ancestors or pid == my_pid:
            continue
        if _proc_uid(pid) != my_uid:
            continue
        cmd = _read_cmdline(pid)
        if not _matches_aou_run(cmd):
            continue
        candidates.append((pid, _rss_kb(pid), cmd))
    if not candidates:
        return 0
    if verbose:
        for pid, rss, cmd in candidates:
            print(
                f"  startup sweep: killing stale sv-pgs pid={pid} "
                f"rss={rss}kB cmd={cmd[:100]}",
                file=sys.stderr,
                flush=True,
            )
    # SIGTERM first so processes that have signal handlers get a chance to
    # release resources cleanly; SIGKILL stragglers below.
    for pid, _rss, _cmd in candidates:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    # Give SIGTERM up to 2 s, then SIGKILL anything still alive.
    deadline = time.monotonic() + 2.0
    pending = {pid for pid, _, _ in candidates}
    while pending and time.monotonic() < deadline:
        time.sleep(0.1)
        for pid in list(pending):
            try:
                os.kill(pid, 0)  # alive?
            except OSError:
                pending.discard(pid)
    killed = 0
    for pid in pending:
        try:
            os.kill(pid, signal.SIGKILL)
            killed += 1
        except OSError:
            pass
    # Brief settle so the kernel reclaims pages before downstream RAM probes.
    if candidates:
        time.sleep(0.5)
    return len(candidates)
