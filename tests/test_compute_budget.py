from pathlib import Path

import pytest

import sv_pgs.compute_budget as compute_budget


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_cgroup_v1_headroom_is_limit_minus_usage(tmp_path: Path) -> None:
    proc_file = tmp_path / "proc_cgroup"
    _write(proc_file, "12:pids:/slurm/job_1\n11:memory:/slurm/uid_7/job_1\n10:freezer:/\n")
    group = tmp_path / "root" / "memory" / "slurm" / "uid_7" / "job_1"
    _write(group / "memory.limit_in_bytes", "17179869184\n")
    _write(group / "memory.usage_in_bytes", "4294967296\n")
    headroom = compute_budget._cgroup_memory_headroom_bytes(proc_file, tmp_path / "root")
    assert headroom == 17179869184 - 4294967296


def test_cgroup_v2_unlimited_and_v1_sentinel_are_none(tmp_path: Path) -> None:
    proc_v2 = tmp_path / "proc_v2"
    _write(proc_v2, "0::/user.slice/session\n")
    group_v2 = tmp_path / "root_v2" / "user.slice" / "session"
    _write(group_v2 / "memory.max", "max\n")
    _write(group_v2 / "memory.current", "123\n")
    assert compute_budget._cgroup_memory_headroom_bytes(proc_v2, tmp_path / "root_v2") is None

    proc_v1 = tmp_path / "proc_v1"
    _write(proc_v1, "11:memory:/\n")
    group_v1 = tmp_path / "root_v1" / "memory"
    _write(group_v1 / "memory.limit_in_bytes", str(compute_budget._CGROUP_V1_UNLIMITED) + "\n")
    _write(group_v1 / "memory.usage_in_bytes", "123\n")
    assert compute_budget._cgroup_memory_headroom_bytes(proc_v1, tmp_path / "root_v1") is None


def test_cgroup_v2_limit_binds(tmp_path: Path) -> None:
    proc_file = tmp_path / "proc"
    _write(proc_file, "0::/job\n")
    _write(tmp_path / "root" / "job" / "memory.max", "1000\n")
    _write(tmp_path / "root" / "job" / "memory.current", "250\n")
    assert compute_budget._cgroup_memory_headroom_bytes(proc_file, tmp_path / "root") == 750


def test_cgroup_v1_limit_on_the_slurm_job_binds_under_unlimited_step_and_task(tmp_path: Path) -> None:
    # MSI Slurm: the task and step cgroups report the unlimited sentinel, the job holds --mem.
    proc_file = tmp_path / "proc"
    _write(proc_file, "3:memory:/slurm/uid_7/job_42/step_batch/task_0\n2:cpuset:/slurm/uid_7/job_42\n")
    job = tmp_path / "root" / "memory" / "slurm" / "uid_7" / "job_42"
    for level, limit, usage in (
        (job / "step_batch" / "task_0", compute_budget._CGROUP_V1_UNLIMITED, 3 * 2**30),
        (job / "step_batch", compute_budget._CGROUP_V1_UNLIMITED, 3 * 2**30),
        (job, 48 * 2**30, 10 * 2**30),
        (job.parent, compute_budget._CGROUP_V1_UNLIMITED, 500 * 2**30),
    ):
        _write(level / "memory.limit_in_bytes", f"{limit}\n")
        _write(level / "memory.usage_in_bytes", f"{usage}\n")
    headroom = compute_budget._cgroup_memory_headroom_bytes(proc_file, tmp_path / "root")
    assert headroom == 38 * 2**30


def test_cgroup_v2_tightest_ancestor_binds(tmp_path: Path) -> None:
    proc_file = tmp_path / "proc"
    _write(proc_file, "0::/machine.slice/job/step\n")
    root = tmp_path / "root"
    for level, limit, usage in (
        (root / "machine.slice" / "job" / "step", "max", "300"),
        (root / "machine.slice" / "job", "1000", "400"),
        (root / "machine.slice", "5000", "4800"),
    ):
        _write(level / "memory.max", limit + "\n")
        _write(level / "memory.current", usage + "\n")
    assert compute_budget._cgroup_memory_headroom_bytes(proc_file, root) == 200


def test_cpu_budget_when_no_device_is_exposed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compute_budget, "_try_import_cupy", lambda: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    budget = compute_budget.detect_compute_budget()
    assert budget.device_kind == "cpu"
    assert budget.device_ids == ()
    assert budget.cpu_threads >= 1
    assert budget.host_bytes > 0
    assert budget.working_bytes == budget.host_bytes


def test_exposed_device_without_usable_cupy_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compute_budget, "_try_import_cupy", lambda: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="CuPy cannot use them"):
        compute_budget.detect_compute_budget()
