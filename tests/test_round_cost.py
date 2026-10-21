"""The round's dry-run cost projection (launcher/workspace/round_cost.py): spend.py's prices, the one-GPU shape choice,
the arithmetic of dollars by phase and by thing (arm, merge, trait, round), preemptions and wall clock, the planner's
tuples, and the submit.sh variables it prints. No cloud calls."""
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

from launcher.workspace import round_cost, spend

LAUNCHER = Path(__file__).parents[1] / "launcher/workspace"


def test_every_shape_the_layouts_use_has_a_spend_price():
    for machine, _compact in round_cost.FIT_DISK:
        assert machine in spend.MACHINE_USD_PER_HOUR
    for disk_type, _gb in round_cost.FIT_DISK.values():
        assert disk_type in spend.DISK_USD_PER_GIB_MONTH
    assert set(round_cost.FIT_DEVICE_GB) <= {machine for machine, _compact in round_cost.FIT_DISK}
    for layout in round_cost.MINUTES.values():
        assert set(layout) == {"fit", "merge"}
        assert all(phase in round_cost.PHASES for minutes in layout.values() for phase in minutes)


def test_rate_is_spends_vm_and_disk_price():
    job = {"name": "a", "machine": "a2-highgpu-1g", "model": "SPOT", "disk": {"type": "local-ssd", "sizeGb": 750}}
    assert round_cost.rate(job) == pytest.approx(2.20401 + 0.0433 * 750 / 730)
    job = {"name": "b", "machine": "a2-highgpu-2g", "model": "STANDARD", "disk": {"type": "pd-balanced", "sizeGb": 2000}}
    assert round_cost.rate(job) == pytest.approx(7.34677 + 0.10 * 2000 / 730)
    with pytest.raises(ValueError, match="no price for"):
        round_cost.rate({"name": "c", "machine": "a9-unknown", "minutes": {}})


def test_fit_shape_takes_the_smallest_device_that_holds_the_codes():
    reserve = round_cost.FIT_DEVICE_RESERVE_GB
    assert round_cost.fit_shape(round_cost.CODES_GB) == "a2-highgpu-1g"
    assert round_cost.fit_shape(40 - reserve) == "a2-highgpu-1g"
    assert round_cost.fit_shape(40 - reserve + 0.1) == "a2-ultragpu-1g"
    assert round_cost.fit_shape(46.6) == "a2-ultragpu-1g"
    with pytest.raises(ValueError, match="two devices"):
        round_cost.fit_shape(80 - reserve + 0.1)


def test_redo_minutes_resumed_or_from_the_start():
    minutes = {"start": 7, "restore": 1, "store_digest": 2, "fit_setup": 3, "stage0": 4, "band_fill": 5, "eb": 40, "score": 3}
    assert round_cost.redo_minutes(minutes, True) == 7 + 1 + 2 + round_cost.PUSH_MINUTES / 2
    assert round_cost.redo_minutes(minutes, False) == 7 + 1 + 2 + 3 + 4 + 5 + 40 / 2


def test_one_gpu_layout_routes_each_arm_by_its_codes_and_merges_on_the_last_arms_vm():
    minutes = round_cost.minutes_of("one-gpu", [])
    jobs = round_cost.layout_one_gpu([("ldl", "snv_indel", 16.6), ("ldl", "with_sv", 40.0), ("t2d", "snv_indel", 24.2)], minutes, True)
    by_name = {job["name"]: job for job in jobs}
    assert list(by_name) == ["ldl-snv_indel", "ldl-with_sv", "ldl-merge", "t2d-snv_indel", "t2d-merge"]
    assert by_name["ldl-snv_indel"]["machine"] == "a2-highgpu-1g"
    assert by_name["ldl-snv_indel"]["disk"] == {"type": "local-ssd", "sizeGb": 750}
    assert by_name["ldl-with_sv"]["machine"] == "a2-ultragpu-1g"
    # The merge is minutes on the VM of the arm that ends last, priced at the dearer arm's rate, with no start or restore.
    assert by_name["ldl-merge"]["machine"] == "a2-ultragpu-1g" and by_name["ldl-merge"]["kind"] == "merge"
    assert by_name["ldl-merge"]["after"] == ["ldl-snv_indel", "ldl-with_sv"] and set(by_name["ldl-merge"]["minutes"]) == {"merge"}
    assert by_name["t2d-merge"]["after"] == ["t2d-snv_indel"]
    whole = round_cost.layout_one_gpu([("ldl", "snv_indel", 16.6)], round_cost.minutes_of("one-gpu-whole-store", []), True, compact=False)
    assert whole[0]["disk"] == {"type": "local-ssd", "sizeGb": 1500}


def test_canaries_alone_take_one_gpu_and_the_report_says_so():
    arms = [(trait, arm, 29.6) for trait in ("af", "ldl") for arm in round_cost.ARMS]
    jobs = round_cost.layout_one_gpu(arms, round_cost.minutes_of("one-gpu-whole-store", []), True, compact=False,
                                     canaries={("af", "snv_indel"), ("af", "with_sv")})
    shapes = {job["name"]: (job["machine"], job["disk"]["sizeGb"]) for job in jobs}
    assert shapes["af-snv_indel"] == shapes["af-with_sv"] == ("a2-highgpu-1g", 1500)
    assert shapes["ldl-snv_indel"] == shapes["ldl-with_sv"] == shapes["ldl-merge"] == (round_cost.TWO_GPU, 1500)
    assert [round_cost.gpu_count(machine) for machine in ("a2-highgpu-1g", "a2-highgpu-2g", "a2-ultragpu-1g", "n2d-standard-8")] == [1, 2, 1, 0]
    shown = round_cost.report(round_cost.project(jobs, 0.0), 0.0)
    assert "canaries: 2 of 4 arms on one GPU, the others on ['a2-highgpu-2g']" in shown
    assert "canaries" not in round_cost.report(round_cost.project(round_cost.layout_one_gpu(arms, round_cost.MINUTES["one-gpu"], True), 0.0), 0.0)
    settings = [dict(pair.split("=") for pair in line.split()) for line in round_cost.submit_environment(jobs).splitlines() if line.startswith("FIT_ARMS=")]
    assert [setting["GPU_COUNT"] for setting in settings] == ["1", "1", "2", "2"]


def test_project_sums_dollars_by_phase_and_thing_preemptions_and_the_longest_chain():
    disk = {"type": "pd-balanced", "sizeGb": 100}
    jobs = [
        {"name": "x-snv_indel", "machine": "a2-highgpu-1g", "disk": disk, "minutes": {"start": 10, "eb": 50}, "redo_minutes": 30},
        {"name": "x-with_sv", "machine": "a2-highgpu-1g", "disk": disk, "minutes": {"start": 10, "eb": 110}, "redo_minutes": 30},
        {"name": "x-report", "machine": "n2d-standard-8", "disk": disk, "minutes": {"merge": 20}, "redo_minutes": 0, "after": ["x-snv_indel", "x-with_sv"]},
        {"name": "solo", "trait": "y", "kind": "fit", "machine": "n2d-standard-8", "disk": disk, "minutes": {"merge": 6}, "redo_minutes": 0},
    ]
    gpu = round_cost.rate(jobs[0])
    cpu = round_cost.rate(jobs[2])
    projection = round_cost.project(jobs, 0.5)
    rows = {row["name"]: row for row in projection["jobs"]}
    assert rows["x-snv_indel"]["preemption_minutes"] == pytest.approx(0.5 * 1.0 * 30)
    assert rows["x-with_sv"]["preemption_minutes"] == pytest.approx(0.5 * 2.0 * 30)
    assert rows["x-snv_indel"]["usd"] == pytest.approx(gpu * (60 + 15) / 60)
    assert projection["usd"] == pytest.approx(gpu * (75 + 150) / 60 + cpu * (20 + 6) / 60)
    assert projection["by_phase"]["eb"] == pytest.approx(gpu * 160 / 60)
    assert list(projection["by_phase"]) == ["start", "eb", "merge", "preemptions"]
    # Kinds and traits: a job with an "eb" phase is a fit, one without a merge, unless it says; the trait is the name's
    # part before its last "-", unless it says.
    assert [rows[name]["kind"] for name in ("x-snv_indel", "x-with_sv", "x-report", "solo")] == ["fit", "fit", "merge", "fit"]
    assert projection["things"]["arm"] == pytest.approx([rows["x-snv_indel"]["usd"], rows["x-with_sv"]["usd"], rows["solo"]["usd"]])
    assert projection["things"]["merge"] == pytest.approx([rows["x-report"]["usd"]])
    assert projection["things"]["trait"] == pytest.approx([gpu * 225 / 60 + cpu * 20 / 60, cpu * 6 / 60])
    assert projection["wall_minutes"] == pytest.approx(120 + 30 + 20)
    with pytest.raises(ValueError, match="unknown jobs"):
        round_cost.project([{**jobs[2], "after": ["z"]}], 0.0)
    with pytest.raises(ValueError, match="cycle"):
        round_cost.project([{**jobs[0], "after": ["x-with_sv"]}, {**jobs[1], "after": ["x-snv_indel"]}], 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        round_cost.project([{**jobs[0], "minutes": {"eb": -1}}], 0.0)


def test_today_layout_is_two_arm_jobs_and_a_merge_per_trait_on_2g():
    jobs = round_cost.layout_today([f"t{index}" for index in range(11)], round_cost.minutes_of("today", []))
    assert len(jobs) == 33 and {job["machine"] for job in jobs} == {"a2-highgpu-2g"}
    projection = round_cost.project(jobs, 0.0)
    rate = 4.40802 + 0.10 * 2000 / 730
    fit = sum(round_cost.MINUTES["today"]["fit"].values())
    merge = sum(round_cost.MINUTES["today"]["merge"].values())
    assert projection["usd"] == pytest.approx(rate * (22 * fit + 11 * merge) / 60)
    assert projection["things"]["trait"] == pytest.approx([rate * (2 * fit + merge) / 60] * 11)
    assert projection["wall_minutes"] == pytest.approx(fit + merge)


def test_minutes_changes_reach_every_job_kind_with_that_phase():
    minutes = round_cost.minutes_of("today", ["eb=12", "start=3"])
    assert minutes["fit"]["eb"] == 12 and minutes["fit"]["start"] == 3 and minutes["merge"]["start"] == 3
    assert round_cost.MINUTES["today"]["fit"]["eb"] != 12
    with pytest.raises(ValueError, match="phase=minutes"):
        round_cost.minutes_of("one-gpu", ["stage1=4"])


def test_planner_tuples_become_jobs_at_either_end_of_their_spans():
    listed = [
        ("svpgs-aou-r7l1-train-1", "a2-highgpu-1g", "SPOT", "local-ssd", 750, {"start": (6, 9.5), "eb": (2, 28)}),
        ("svpgs-aou-r7l1-train-2", "a2-highgpu-1g", "SPOT", "local-ssd", 750, {"merge": (23, 40)}),
    ]
    low, high = round_cost.from_planner(listed, 0), round_cost.from_planner(listed, 1)
    assert low[0]["minutes"] == {"start": 6.0, "eb": 2.0} and high[0]["minutes"] == {"start": 9.5, "eb": 28.0}
    assert high[1]["disk"] == {"type": "local-ssd", "sizeGb": 750}
    projection = round_cost.project(high, 0.0)
    assert [row["kind"] for row in projection["jobs"]] == ["fit", "merge"]
    assert projection["usd"] == pytest.approx(round_cost.rate(high[0]) * (37.5 + 40) / 60)
    # The planner's run IDs differ by arm: a seventh element names each job's trait (and may name its kind and order).
    listed = [(*entry, {"trait": "ldl", "after": [] if index == 0 else [listed[0][0]]}) for index, entry in enumerate(listed)]
    projection = round_cost.project(round_cost.from_planner(listed, 1), 0.0)
    assert projection["things"]["trait"] == pytest.approx([projection["usd"]])
    assert projection["wall_minutes"] == pytest.approx(37.5 + 40)


def test_cli_projects_a_plan_against_today_and_prints_submit_variables_submit_sh_accepts(tmp_path):
    plan = {"preemptions_per_hour": 0.0, "jobs": [
        {"name": "x-snv_indel", "machine": "a2-highgpu-1g", "disk": {"type": "local-ssd", "sizeGb": 750}, "minutes": {"eb": 60}},
    ]}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    # As a script from its own directory, as the workspace app's python3 runs it (no repository package on the path).
    shown = subprocess.run([sys.executable, "round_cost.py", "--plan", str(path)], cwd=LAUNCHER, capture_output=True, text=True, check=True).stdout
    assert re.search(r"projected round: \$2\.25 over 1 jobs, 60 min", shown)
    today = round_cost.project(round_cost.layout_today(["x"], round_cost.MINUTES["today"]), 0.0)
    assert f"within a fifth of today's ${today['usd']:.2f}" in shown
    assert re.search(r"^arm +1 +2\.25 +2\.25 +" + re.escape(f"{today['things']['arm'][0]:.2f}"), shown, re.MULTILINE)
    shown = subprocess.run([sys.executable, str(LAUNCHER / "round_cost.py"), "--layout", "one-gpu", "--arm", "ldl:snv_indel:16.6",
                            "--arm", "ldl:with_sv:40", "--resumable", "--submit-env", "--json"], capture_output=True, text=True, check=True).stdout
    projection = json.loads(shown[: shown.rindex("}") + 1])
    assert [row["name"] for row in projection["jobs"]] == ["ldl-snv_indel", "ldl-with_sv", "ldl-merge"]
    assert len(projection["today"]["jobs"]) == 3
    settings = [dict(pair.split("=") for pair in line.split()) for line in shown.splitlines() if line.startswith("FIT_ARMS=")]
    assert [setting["FIT_ARMS"] for setting in settings] == ["snv_indel", "with_sv"]
    for setting in settings:
        assert setting["GPU_COUNT"] == "1" and setting["PROVISIONING_MODEL"] == "SPOT"
        assert setting["RUN_DISK_TYPE"] in ("pd-balanced", "pd-ssd", "local-ssd")
        assert setting["RUN_DISK_TYPE"] != "local-ssd" or int(setting["RUN_DISK_GB"]) % 375 == 0
        assert float(setting["HOURLY_USD"]) >= spend.MACHINE_USD_PER_HOUR[setting["MACHINE_TYPE"]][1]
    assert settings[0]["GPU_TYPE"] == "nvidia-tesla-a100" and settings[1]["GPU_TYPE"] == "nvidia-a100-80gb"
    failed = subprocess.run([sys.executable, str(LAUNCHER / "round_cost.py"), "--layout", "one-gpu", "--arm", "x:with_sv:90"],
                            capture_output=True, text=True)
    assert failed.returncode == 2 and "fit no one-GPU shape" in failed.stderr
