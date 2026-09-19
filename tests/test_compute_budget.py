from pathlib import Path

import pytest

import sv_pgs.compute_budget as compute_budget


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_stat(level: Path, key: str, inactive_file: int) -> None:
    """A memory.stat with the reclaimable page-cache entry next to its neighbours."""
    _write(level / "memory.stat", f"anon 4096\nfile {inactive_file + 8192}\n{key} {inactive_file}\nactive_file 8192\n")


def test_cgroup_v1_headroom_is_limit_minus_usage(tmp_path: Path) -> None:
    proc_file = tmp_path / "proc_cgroup"
    _write(proc_file, "12:pids:/slurm/job_1\n11:memory:/slurm/uid_7/job_1\n10:freezer:/\n")
    group = tmp_path / "root" / "memory" / "slurm" / "uid_7" / "job_1"
    _write(group / "memory.limit_in_bytes", "17179869184\n")
    _write(group / "memory.usage_in_bytes", "4294967296\n")
    _write_stat(group, "total_inactive_file", 0)
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
    _write_stat(tmp_path / "root" / "job", "inactive_file", 0)
    assert compute_budget._cgroup_memory_headroom_bytes(proc_file, tmp_path / "root") == 750


def test_cgroup_page_cache_the_kernel_reclaims_counts_as_headroom(tmp_path: Path) -> None:
    # After a pass over the store the job's charged page cache fills its limit: usage is at the
    # limit, but the inactive file pages are reclaimed before the cgroup could run out.
    proc_v2 = tmp_path / "proc_v2"
    _write(proc_v2, "0::/job\n")
    _write(tmp_path / "root_v2" / "job" / "memory.max", "1000\n")
    _write(tmp_path / "root_v2" / "job" / "memory.current", "990\n")
    _write_stat(tmp_path / "root_v2" / "job", "inactive_file", 600)
    assert compute_budget._cgroup_memory_headroom_bytes(proc_v2, tmp_path / "root_v2") == 610

    proc_v1 = tmp_path / "proc_v1"
    _write(proc_v1, "11:memory:/slurm/job_3\n")
    job = tmp_path / "root_v1" / "memory" / "slurm" / "job_3"
    _write(job / "memory.limit_in_bytes", "4096\n")
    _write(job / "memory.usage_in_bytes", "4000\n")
    _write(job / "memory.stat", "cache 3500\ninactive_file 100\ntotal_inactive_file 3000\n")
    assert compute_budget._cgroup_memory_headroom_bytes(proc_v1, tmp_path / "root_v1") == 3096


def test_a_limited_cgroup_without_its_memory_stat_entry_raises(tmp_path: Path) -> None:
    proc_file = tmp_path / "proc"
    _write(proc_file, "0::/job\n")
    _write(tmp_path / "root" / "job" / "memory.max", "1000\n")
    _write(tmp_path / "root" / "job" / "memory.current", "250\n")
    _write(tmp_path / "root" / "job" / "memory.stat", "anon 250\n")
    with pytest.raises(RuntimeError, match="inactive_file"):
        compute_budget._cgroup_memory_headroom_bytes(proc_file, tmp_path / "root")


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
        _write_stat(level, "total_inactive_file", 0)
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
        _write_stat(level, "inactive_file", 0)
    assert compute_budget._cgroup_memory_headroom_bytes(proc_file, root) == 200


def test_the_runner_allotment_and_the_cgroup_both_cap_host_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compute_budget, "_detect_available_host_ram_bytes", lambda: 100 * 2**30)
    monkeypatch.setattr(compute_budget, "_cgroup_memory_headroom_bytes", lambda: 60 * 2**30)
    monkeypatch.delenv(compute_budget.RUNQ_MEMORY_VARIABLE, raising=False)
    assert compute_budget._usable_host_bytes() == 60 * 2**30
    monkeypatch.setenv(compute_budget.RUNQ_MEMORY_VARIABLE, str(24 * 2**30))
    assert compute_budget._usable_host_bytes() == 24 * 2**30
    monkeypatch.setenv(compute_budget.RUNQ_MEMORY_VARIABLE, str(80 * 2**30))
    assert compute_budget._usable_host_bytes() == 60 * 2**30
    monkeypatch.setattr(compute_budget, "_cgroup_memory_headroom_bytes", lambda: None)
    assert compute_budget._usable_host_bytes() == 80 * 2**30
    for invalid in ("", "0", "-5", "12GB", "1.5e9"):
        monkeypatch.setenv(compute_budget.RUNQ_MEMORY_VARIABLE, invalid)
        with pytest.raises(ValueError, match="RUNQ_MEM_BYTES"):
            compute_budget._usable_host_bytes()


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
