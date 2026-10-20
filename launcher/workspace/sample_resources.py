"""One line of the VM's resources a minute on stdout, which run_step.sh appends to the driver's log beside its progress lines:

    [2026-10-20 09:13:41] resources: gpu0 97% 31.2/40.0 GiB, gpu1 95% 30.8/40.0 GiB; cpu 12% iowait 1% load 3.1;
    mem 61.2/170.0 GiB; disk r 410.2 w 3.1 MiB/s; net rx 0.1 tx 0.0 MiB/s

(one line in the log). Each figure covers the interval before its line: a GPU's utilization is the mean of nvidia-smi's
samples every --gpu-every seconds (each the share of the driver's own sample period in which a kernel ran; one query a
minute would be that one second's), its memory the last sample's used and total (CuPy's pool counts as used); the CPU's
busy and iowait shares come from /proc/stat's jiffies, the load is the 1-minute average, the memory the host's in use
(MemTotal less MemAvailable, so the page cache a read fills is not counted); the disk's rates are the --disk mount's
device in /proc/diskstats (512-byte sectors whatever the device's own, Documentation/admin-guide/iostats.rst) and the
network's every interface's but the loopback's in /proc/net/dev. Sizes, rates and shares of the machine only: nothing
of the data, and no count of people. benchmarks/profile_log.py reads the lines into a run's phase timeline.

nvidia-smi (the PATH's, or a driver directory's) runs once for the task, looping (--loop), on a pseudo-terminal so
that its C library writes each sample as it takes it rather than when a pipe's buffer fills (on a pipe where no
terminal can be had); where it is missing or stops, the line says "gpu n/a" and the next interval starts it again. The
sampler's own cost is a sleep, a handful of small /proc reads an interval, and nvidia-smi's queries every --gpu-every
seconds: far under 1% of one CPU. It exits on SIGTERM, stopping nvidia-smi.

    sample_resources.py --every 60 --gpu-every 5 --disk /mnt/disks/run
"""
from __future__ import annotations

import argparse
import io
import os
import pty
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

GPU_QUERY = ("--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits")
NVIDIA_SMI_DIRECTORIES = ("/var/lib/nvidia/bin", "/usr/local/nvidia/bin")
"""Where a driver installed outside the PATH keeps nvidia-smi (Container-Optimized OS's installer, GKE's), tried after it."""
SECTOR_BYTES = 512
KIB = 1 << 10
MIB = 1 << 20
GIB = 1 << 30


def nvidia_smi() -> str:
    """nvidia-smi on the PATH, else in a driver directory that has it, else the bare name (whose start then fails)."""
    found = shutil.which("nvidia-smi") or shutil.which("nvidia-smi", path=os.pathsep.join(NVIDIA_SMI_DIRECTORIES))
    return found or "nvidia-smi"


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii", errors="replace")
    except OSError:
        return None


def cpu_jiffies(proc: Path) -> tuple[int, int, int] | None:
    """(total, idle, iowait) jiffies of every CPU since boot, from /proc/stat's first line: user, nice, system, idle,
    iowait, irq, softirq and steal (its guest columns are already counted in user and nice)."""
    text = _read(proc / "stat")
    if text is None or not text.startswith("cpu "):
        return None
    fields = [int(value) for value in text.split("\n", 1)[0].split()[1:]]
    return sum(fields[:8]), fields[3], fields[4]


def load_average(proc: Path) -> float | None:
    text = _read(proc / "loadavg")
    return None if not text else float(text.split()[0])


def memory_bytes(proc: Path) -> tuple[int, int] | None:
    """(in use, total) host bytes: MemTotal less MemAvailable, and MemTotal (/proc/meminfo reports kB)."""
    text = _read(proc / "meminfo")
    values = {}
    for line in (text or "").splitlines():
        name, _, rest = line.partition(":")
        if name in ("MemTotal", "MemAvailable"):
            values[name] = int(rest.split()[0]) * KIB
    if len(values) < 2:
        return None
    return values["MemTotal"] - values["MemAvailable"], values["MemTotal"]


def disk_device(proc: Path, path: str) -> str | None:
    """The name in /proc/diskstats of the block device holding ``path``: the device of the longest mount point that
    contains it (/proc/self/mounts), its symbolic links resolved (a Batch disk mounts by /dev/disk/by-id)."""
    text = _read(proc / "self" / "mounts")
    target = os.path.realpath(path)
    found: tuple[str, str] | None = None
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        device, mount = parts[0], parts[1].replace("\\040", " ")
        inside = target == mount or target.startswith(mount.rstrip("/") + "/")
        if inside and (found is None or len(mount) > len(found[1])):
            found = device, mount
    if found is None or not found[0].startswith("/dev/"):
        return None
    return os.path.basename(os.path.realpath(found[0]))


def disk_bytes(proc: Path, device: str | None) -> tuple[int, int] | None:
    """(read, written) bytes of ``device`` since boot: /proc/diskstats' sectors read (field 6) and written (field 10)."""
    text = _read(proc / "diskstats") if device is not None else None
    for line in (text or "").splitlines():
        fields = line.split()
        if len(fields) > 9 and fields[2] == device:
            return int(fields[5]) * SECTOR_BYTES, int(fields[9]) * SECTOR_BYTES
    return None


def net_bytes(proc: Path) -> tuple[int, int] | None:
    """(received, transmitted) bytes since boot over every interface but the loopback, from /proc/net/dev."""
    text = _read(proc / "net" / "dev")
    if text is None:
        return None
    received = transmitted = 0
    for line in text.splitlines()[2:]:
        name, _, counters = line.partition(":")
        fields = counters.split()
        if name.strip() != "lo" and len(fields) > 8:
            received += int(fields[0])
            transmitted += int(fields[8])
    return received, transmitted


@dataclass(frozen=True)
class Counters:
    """The cumulative counters an interval's rates are differences of, and when they were read (time.monotonic)."""

    at: float
    cpu: tuple[int, int, int] | None
    disk: tuple[int, int] | None
    net: tuple[int, int] | None


def counters(proc: Path, device: str | None) -> Counters:
    return Counters(time.monotonic(), cpu_jiffies(proc), disk_bytes(proc, device), net_bytes(proc))


class GpuSamples:
    """nvidia-smi's samples, gathered as it writes them: per GPU index, (utilization %, memory used MiB, memory total
    MiB) since the last ``drain``."""

    def __init__(self, every: int) -> None:
        self._every = every
        self._lock = threading.Lock()
        self._samples: dict[int, list[tuple[float, float, float]]] = {}
        self._process: subprocess.Popen | None = None

    def start(self) -> None:
        """Start nvidia-smi's loop, unless it runs: a missing nvidia-smi or a terminal that cannot be had leaves none."""
        if self._process is not None and self._process.poll() is None:
            return
        self._process = None
        try:
            leader, follower = pty.openpty()
        except OSError:
            leader, follower = os.pipe()
        try:
            self._process = subprocess.Popen(
                [nvidia_smi(), *GPU_QUERY, f"--loop={self._every}"], stdin=subprocess.DEVNULL, stdout=follower, stderr=subprocess.DEVNULL,
            )
        except OSError:
            os.close(leader)
            return
        finally:
            os.close(follower)
        threading.Thread(target=self._gather, args=(leader,), name="nvidia-smi", daemon=True).start()

    def _gather(self, leader: int) -> None:
        pending = b""
        try:
            while chunk := os.read(leader, io.DEFAULT_BUFFER_SIZE):
                *lines, pending = (pending + chunk).split(b"\n")
                for line in lines:
                    self.take(line.decode("ascii", errors="replace"))
        except OSError:
            # The terminal reads as an error once nvidia-smi has exited.
            pass
        finally:
            os.close(leader)

    def take(self, line: str) -> None:
        """One CSV line of the query (index, utilization, used, total); a line of anything else (a "[N/A]") is skipped."""
        fields = [field.strip() for field in line.strip().split(",")]
        try:
            index, utilization, used, total = int(fields[0]), float(fields[1]), float(fields[2]), float(fields[3])
        except (IndexError, ValueError):
            return
        with self._lock:
            self._samples.setdefault(index, []).append((utilization, used, total))

    def drain(self) -> dict[int, tuple[float, float, float]]:
        """Per GPU, the mean utilization of its samples since the last call and its last sample's memory used and total
        (MiB); the samples are then dropped."""
        with self._lock:
            samples, self._samples = self._samples, {}
        return {
            index: (sum(sample[0] for sample in taken) / len(taken), taken[-1][1], taken[-1][2])
            for index, taken in sorted(samples.items())
        }

    def stop(self) -> None:
        if self._process is None or self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=self._every)
        except subprocess.TimeoutExpired:
            self._process.kill()


def resource_line(gpus: dict[int, tuple[float, float, float]], before: Counters, after: Counters, memory: tuple[int, int] | None,
                  load: float | None) -> str:
    """The interval's line, without its timestamp: each part "n/a" where its counters could not be read."""
    seconds = after.at - before.at
    parts = [", ".join(f"gpu{index} {utilization:.0f}% {used * MIB / GIB:.1f}/{total * MIB / GIB:.1f} GiB"
                       for index, (utilization, used, total) in gpus.items()) or "gpu n/a"]
    if before.cpu is not None and after.cpu is not None and after.cpu[0] > before.cpu[0]:
        total, idle, iowait = (now - then for now, then in zip(after.cpu, before.cpu))
        cpu = f"cpu {100 * (total - idle - iowait) / total:.0f}% iowait {100 * iowait / total:.0f}%"
    else:
        cpu = "cpu n/a"
    parts.append(cpu if load is None else f"{cpu} load {load:.1f}")
    parts.append("mem n/a" if memory is None else f"mem {memory[0] / GIB:.1f}/{memory[1] / GIB:.1f} GiB")
    for name, labels, old, new in (("disk", ("r", "w"), before.disk, after.disk), ("net", ("rx", "tx"), before.net, after.net)):
        if old is None or new is None or not seconds > 0:
            parts.append(f"{name} n/a")
            continue
        rates = [(now - then) / MIB / seconds for now, then in zip(new, old)]
        parts.append(f"{name} {labels[0]} {rates[0]:.1f} {labels[1]} {rates[1]:.1f} MiB/s")
    return "resources: " + "; ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--every", type=float, required=True, help="Seconds between lines.")
    parser.add_argument("--gpu-every", type=int, required=True, help="Seconds between nvidia-smi's samples.")
    parser.add_argument("--disk", required=True, help="A path on the disk whose reads and writes are reported.")
    parser.add_argument("--proc", type=Path, default=Path("/proc"), help="Where procfs is mounted.")
    parser.add_argument("--lines", type=int, help="Stop after this many lines (default: run until terminated).")
    args = parser.parse_args(argv)
    # SIGTERM (run_step.sh's stop) ends the sleep as an exit, so nvidia-smi is stopped with the sampler.
    signal.signal(signal.SIGTERM, lambda _signal, _frame: sys.exit(0))
    device = disk_device(args.proc, args.disk)
    gpus = GpuSamples(args.gpu_every)
    written = 0
    try:
        gpus.start()
        before = counters(args.proc, device)
        while args.lines is None or written < args.lines:
            time.sleep(args.every)
            after = counters(args.proc, device)
            line = resource_line(gpus.drain(), before, after, memory_bytes(args.proc), load_average(args.proc))
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {line}", flush=True)
            before = after
            written += 1
            # Again where nvidia-smi stopped, for the next interval's samples.
            gpus.start()
    finally:
        gpus.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
