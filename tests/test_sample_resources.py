"""The run's resource sampler (launcher/workspace/sample_resources.py) reads /proc and nvidia-smi into one line a
minute that benchmarks/profile_log.py reads back, survives a VM without nvidia-smi, and stops with its task.

/proc is a directory of the files' formats here and nvidia-smi a script that loops two GPUs' samples, so the tests
need neither a GPU nor a particular kernel.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from benchmarks import profile_log
from launcher.workspace import sample_resources as sampler

SCRIPT = Path(__file__).parents[1] / "launcher/workspace/sample_resources.py"
FAKE_NVIDIA_SMI = """#!/bin/bash
# two GPUs' samples every --loop seconds, as nvidia-smi --query-gpu writes them; its pid where the test reads it
echo $$ > "$(dirname "$0")/nvidia-smi.pid"
every=1
for argument in "$@"; do case "$argument" in --loop=*) every=${argument#--loop=} ;; esac; done
while true; do echo "0, 97, 31948, 40960"; echo "1, [N/A], 0, 40960"; echo "1, 95, 31539, 40960"; sleep "$every"; done
"""


def _proc(root: Path, *, jiffies: tuple[int, ...], read_sectors: int, written_sectors: int, received: int, sent: int) -> Path:
    (root / "self").mkdir(parents=True, exist_ok=True)
    (root / "net").mkdir(exist_ok=True)
    (root / "stat").write_text("cpu  " + " ".join(map(str, jiffies)) + " 0 0\ncpu0 1 2 3 4 5 6 7 8 0 0\nintr 9\n")
    (root / "loadavg").write_text("3.84 2.10 1.00 4/812 4242\n")
    (root / "meminfo").write_text("MemTotal:       41943040 kB\nMemFree:         1048576 kB\nMemAvailable:   40894464 kB\n")
    (root / "self" / "mounts").write_text(
        "/dev/root / ext4 rw 0 0\n"
        "/dev/nvme9n9 /mnt/disks/run ext4 rw 0 0\n"
        "gcsfuse /mnt/disks/run/ops fuse rw 0 0\n"
        "tmpfs /mnt/disks/runs tmpfs rw 0 0\n"
    )
    (root / "diskstats").write_text(
        f" 259       0 nvme9n9 100 0 {read_sectors} 50 200 0 {written_sectors} 80 0 120 130 0 0 0 0\n"
        "   8       0 sda 1 0 999999 1 1 0 999999 1 0 1 1\n"
    )
    (root / "net" / "dev").write_text(
        "Inter-|   Receive                                                |  Transmit\n"
        " face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed\n"
        "    lo: 999999 1 0 0 0 0 0 0 999999 1 0 0 0 0 0 0\n"
        f"  ens4: {received} 10 0 0 0 0 0 0 {sent} 10 0 0 0 0 0 0\n"
    )
    return root


def test_each_counter_is_read_from_its_proc_file(tmp_path: Path) -> None:
    proc = _proc(tmp_path, jiffies=(100, 0, 50, 800, 50, 0, 0, 0), read_sectors=2048, written_sectors=4096, received=10, sent=20)
    assert sampler.cpu_jiffies(proc) == (1000, 800, 50)
    assert sampler.load_average(proc) == 3.84
    # MemTotal less MemAvailable: the page cache is not memory in use
    assert sampler.memory_bytes(proc) == ((41943040 - 40894464) * 1024, 41943040 * 1024)
    # the longest mount holding the path, not a mount whose name only begins the same
    assert sampler.disk_device(proc, "/mnt/disks/run/step/store") == "nvme9n9"
    assert sampler.disk_device(proc, "/mnt/disks/run/ops/svpgs") is None
    assert sampler.disk_bytes(proc, "nvme9n9") == (2048 * 512, 4096 * 512)
    assert sampler.disk_bytes(proc, None) is None
    # every interface but the loopback
    assert sampler.net_bytes(proc) == (10, 20)
    for missing in ("stat", "meminfo", "diskstats"):
        (proc / missing).unlink()
    assert sampler.cpu_jiffies(proc) is None and sampler.memory_bytes(proc) is None and sampler.disk_bytes(proc, "nvme9n9") is None


def test_a_line_is_each_interval_s_rates_and_reads_back_into_the_profile(tmp_path: Path) -> None:
    mib = 1 << 20
    before = sampler.Counters(10.0, (1000, 800, 50), (0, 0), (0, 0))
    after = sampler.Counters(70.0, (2000, 1500, 100), (600 * mib, 60 * mib), (6 * mib, 6 * mib))
    gpus = {0: (96.4, 31948.0, 40960.0), 1: (12.0, 512.0, 40960.0)}
    line = sampler.resource_line(gpus, before, after, (8 << 30, 40 << 30), 3.84)
    assert line == (
        "resources: gpu0 96% 31.2/40.0 GiB, gpu1 12% 0.5/40.0 GiB; cpu 25% iowait 5% load 3.8; mem 8.0/40.0 GiB; "
        "disk r 10.0 w 1.0 MiB/s; net rx 0.1 tx 0.1 MiB/s"
    )
    sample = profile_log.parse([f"[2026-10-20 09:00:00] {line}"]).samples[0]
    assert (sample.gpu, sample.cpu, sample.disk_read, sample.disk_write, sample.net_rx, sample.memory) == (
        (96.0 + 12.0) / 2, 25.0, 10.0, 1.0, 0.1, 8.0,
    )


def test_a_counter_that_could_not_be_read_is_n_a_and_the_line_is_still_written() -> None:
    empty = sampler.Counters(0.0, None, None, None)
    line = sampler.resource_line({}, empty, sampler.Counters(60.0, None, None, None), None, None)
    assert line == "resources: gpu n/a; cpu n/a; mem n/a; disk n/a; net n/a"
    sample = profile_log.parse([f"[2026-10-20 09:00:00] {line}"]).samples[0]
    assert sample.gpu is None and sample.disk_read is None


def test_nvidia_smi_s_samples_average_per_gpu_and_its_other_lines_are_skipped() -> None:
    gpus = sampler.GpuSamples(5)
    for line in ("0, 90, 1000, 40960", "0, 100, 2000, 40960", "1, [N/A], [N/A], 40960", "garbage", "", "1, 50, 3000, 40960"):
        gpus.take(line)
    assert gpus.drain() == {0: (95.0, 2000.0, 40960.0), 1: (50.0, 3000.0, 40960.0)}
    assert gpus.drain() == {}


def _fake_nvidia_smi(directory: Path) -> Path:
    directory.mkdir()
    fake = directory / "nvidia-smi"
    fake.write_text(FAKE_NVIDIA_SMI)
    fake.chmod(0o755)
    return directory


def test_nvidia_smi_is_the_path_s_else_a_driver_directory_s(tmp_path: Path, monkeypatch) -> None:
    fake = _fake_nvidia_smi(tmp_path / "driver")
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    monkeypatch.setattr(sampler, "NVIDIA_SMI_DIRECTORIES", (str(tmp_path / "absent"), str(fake)))
    assert sampler.nvidia_smi() == str(fake / "nvidia-smi")
    monkeypatch.setattr(sampler, "NVIDIA_SMI_DIRECTORIES", ())
    assert sampler.nvidia_smi() == "nvidia-smi"


def _run(tmp_path: Path, path: str, *extra: str) -> subprocess.Popen:
    proc = _proc(tmp_path / "proc", jiffies=(100, 0, 50, 800, 50, 0, 0, 0), read_sectors=0, written_sectors=0, received=0, sent=0)
    return subprocess.Popen(
        [sys.executable, str(SCRIPT), "--every", "1", "--gpu-every", "1", "--disk", "/mnt/disks/run", "--proc", str(proc), *extra],
        env=dict(os.environ, PATH=path), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def test_the_sampler_writes_gpu_lines_and_stops_nvidia_smi_when_terminated(tmp_path: Path) -> None:
    fake = _fake_nvidia_smi(tmp_path / "bin")
    running = _run(tmp_path, f"{fake}{os.pathsep}{os.environ['PATH']}")
    lines = [running.stdout.readline() for _ in range(2)]
    running.send_signal(signal.SIGTERM)
    _, errors = running.communicate(timeout=10)
    assert running.returncode == 0, errors
    assert all(profile_log.STAMP.match(line[1:]) and " resources: " in line for line in lines)
    # nvidia-smi's first samples can land after the first line; by the second its loop has written
    assert "gpu0 97% 31.2/40.0 GiB, gpu1 95% 30.8/40.0 GiB; cpu " in lines[1]
    # the sampler's own nvidia-smi is gone with it (stopped and reaped, not left looping on the VM)
    pid = int((fake / "nvidia-smi.pid").read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("nvidia-smi still runs after the sampler stopped")


def test_without_nvidia_smi_the_sampler_writes_gpu_n_a_and_exits_cleanly(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    finished = _run(tmp_path, str(empty), "--lines", "2")
    output, errors = finished.communicate(timeout=10)
    assert finished.returncode == 0 and errors == ""
    lines = output.splitlines()
    assert len(lines) == 2 and all(" resources: gpu n/a; cpu " in line and "; disk r 0.0 w 0.0 MiB/s;" in line for line in lines)


TEMPLATE = Path(__file__).parents[1] / "launcher/workspace/run_step.sh.template"


def test_the_task_samples_from_before_the_restore_to_the_driver_s_end_into_the_driver_s_log() -> None:
    script = TEMPLATE.read_text()
    driver = script.index('wait "$driver"')
    assert script.index("sample_resources.py") < script.index("mirror restore") < script.index("timing: store_restore") < script.index("workspace-run")
    # every writer appends, and the whole log goes beside the task log for benchmarks/profile_log.py
    assert '2> >(tee -a "$RUN/driver.log" >&2)' in script and 'cp "$RUN/driver.log" "${progress%.log}.full.log"' in script
    assert driver < script.index('kill "$sampler"', driver) < script.index("copy_progress\n", driver)


def test_the_sampler_stops_with_the_task_and_never_changes_its_exit_status(tmp_path: Path) -> None:
    """The template's own sampler lines, run with a stand-in run directory: a task that succeeds, fails, or is
    terminated (the template's TERM trap exits 75) leaves no sampler behind and keeps its status."""
    script = TEMPLATE.read_text()
    lines = script[script.index(': > "$RUN/driver.log"') : script.index("restore_started=$SECONDS")]
    run = tmp_path / "run"
    (run / "venv" / "bin").mkdir(parents=True)
    (run / "venv" / "bin" / "python").symlink_to(sys.executable)
    (run / "source" / "launcher" / "workspace").mkdir(parents=True)
    (run / "source" / "launcher" / "workspace" / "sample_resources.py").write_text(SCRIPT.read_text())
    for ending, status in (("exit 0", 0), ("exit 3", 3), ("(sleep 0.2; kill -TERM $$) & sleep 30 > /dev/null 2>&1 & wait $!", 75)):
        task = subprocess.run(
            ["bash", "-c", f"trap 'exit 75' TERM\nRUN={run}\n{lines}\necho $sampler > {run}/sampler.pid\nsleep 0.5\n{ending}\n"],
            capture_output=True, text=True, timeout=60,
        )
        assert task.returncode == status, task.stderr
        pid = int((run / "sampler.pid").read_text())
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            pytest.fail(f"the sampler outlived a task that ended with {ending}")
