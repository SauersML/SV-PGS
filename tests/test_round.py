"""The round launcher (launcher/workspace/round.py) on fixture specs, against a fixture git repository and a fake
gcloud over a local directory, never a cloud account: the plan of an 11-trait round on each timeline with its minutes
and dollars, which steps each run takes from its base and why the others are its job's, each arm's warm or cold start
and why, that a dry run changes nothing, that --execute stages once and checks it, copies a step's record last and submits each job with the round's
settings, the shapes (one GPU by default, canaries), the merges with and without PARTNER_RUN and arm_vms' store job
(--follow), the dollars through round_cost, and a job built from a round's staging that checks its copy of the
source."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from launcher.workspace import round as planner
from sv_pgs import workspace_pipeline
from tests.test_study_budget import _UV, _fake_cloud

LAUNCHER = Path(__file__).resolve().parents[1] / "launcher/workspace"
TRAITS = [("ldl_cholesterol", "l"), ("egfr_ckd_epi_2021", "e"), ("hypertension", "h"), ("asthma", "a"), ("type2_diabetes", "t"),
          ("gout", "g"), ("copd", "o"), ("coronary_artery_disease", "c"), ("hypothyroidism", "y"), ("psoriasis", "s"), ("cataract", "k")]
BUCKET = "ops-bucket"

FAKE_GCLOUD = r'''
import json, os, shutil, sys, time
from pathlib import Path
root = Path(os.environ["FAKE_GCS"])
args = sys.argv[1:]
with (root / "calls.log").open("a") as log:
    log.write(" ".join(args) + "\n")
def local(url):
    return root / "gcs" / url[len("gs://"):] if url.startswith("gs://") else Path(url)
def files(path):
    return sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else []
def show(path):
    return "gs://" + path.relative_to(root / "gcs").as_posix()
def absent(what="The following URLs matched no objects or files"):
    sys.exit(f"ERROR: (gcloud.{'.'.join(args[:2])}) {what}")
# Each object's generation, as Cloud Storage keeps it: one written outside this gcloud is at generation 1.
generations = root / "generations.json"
def generation(path):
    return json.loads(generations.read_text()).get(str(path), 1) if generations.is_file() and path.is_file() else (1 if path.is_file() else 0)
def written(path):
    known = json.loads(generations.read_text()) if generations.is_file() else {}
    known[str(path)] = max(known.values(), default=1) + 1
    generations.write_text(json.dumps(known))
def matches(path):
    wanted = [value.split("=", 1)[1] for value in args if value.startswith("--if-generation-match=")]
    if wanted and int(wanted[0]) != generation(path):
        sys.exit("ERROR: (gcloud.storage) HTTPError 412: At least one of the pre-conditions you specified did not hold.")
if os.environ.get("FAKE_GCLOUD_DOWN"):
    sys.exit("ERROR: (gcloud) There was a problem refreshing your current auth tokens: connection reset")
if args[:2] == ["storage", "cat"]:
    path = local(args[2])
    if not path.is_file():
        absent()
    sys.stdout.write(path.read_text())
elif args[:3] == ["storage", "objects", "describe"]:
    path = local(args[3])
    if not path.is_file():
        absent("NOT_FOUND: No such object")
    print(json.dumps({"name": args[3], "generation": str(generation(path))}))
elif args[:2] == ["storage", "ls"]:
    url = args[-1]
    path = local(url)
    if "-l" in args:
        listed = files(path) if url.endswith("/") and "-r" in args else [path] if path.is_file() else []
        if not listed:
            absent("One or more URLs matched no objects.")
        for item in listed:
            written = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(item.stat().st_mtime))
            print(f"{item.stat().st_size:>10}  {written}  {show(item)}")
        print(f"TOTAL: {len(listed)} objects")
    elif "*" in url:
        matched = sorted(item for item in (root / "gcs").glob(url[len("gs://"):]) if item.is_file())
        if not matched:
            absent("One or more URLs matched no objects.")
        for item in matched:
            print(show(item))
    else:
        if not path.is_dir() or not any(path.iterdir()):
            absent("One or more URLs matched no objects.")
        for item in sorted(path.iterdir()):
            print(show(item) + ("/" if item.is_dir() else ""))
elif args[:2] == ["storage", "cp"] and "-r" in args:
    # A directory into a destination ending in "/" lands under its own name, as gcloud storage cp -r does.
    source, target = [value for value in args[2:] if not value.startswith("-")]
    assert target.endswith("/"), target
    into = local(target.rstrip("/")) / local(source).name
    if not files(local(source)):
        sys.exit(1)
    for item in files(local(source)):
        (into / item.relative_to(local(source))).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item, into / item.relative_to(local(source)))
elif args[:2] == ["storage", "cp"]:
    source, target = [local(value) for value in args[2:] if not value.startswith("-")]
    if not source.is_file():
        absent()
    matches(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    written(target)
elif args[:3] == ["batch", "jobs", "describe"]:
    # A job's file holds its state, or its whole record as JSON.
    state = root / "jobs" / args[3]
    if not state.is_file():
        absent(f"NOT_FOUND: Resource 'jobs/{args[3]}' was not found")
    text = state.read_text().strip()
    record = json.loads(text) if text.startswith("{") else {"status": {"state": text}}
    print(json.dumps(record) if "--format=json" in args else record["status"]["state"])
elif args[:3] == ["batch", "tasks", "list"]:
    tasks = root / "jobs" / (next(value.split("=", 1)[1] for value in args if value.startswith("--job=")) + ".tasks")
    print(tasks.read_text() if tasks.is_file() else "[]")
elif args[:2] == ["storage", "rm"]:
    path = local(args[2])
    if not path.is_file():
        absent()
    matches(path)
    path.unlink()
else:
    sys.exit(f"unexpected gcloud {args}")
'''
FAKE_SUBMIT = """#!/bin/bash
python3 -c 'import json, os, sys; print(json.dumps({name: os.environ.get(name, "") for name in sys.argv[1:]}))' \
  RUN_ID BUDGET_SLOT THROUGH STAGED_SOURCE BASE_RUN FIT_ARMS PARALLEL_ARMS PARTNER_RUN MACHINE_TYPE GPU_TYPE GPU_COUNT PROVISIONING_MODEL \
  MAX_RETRIES RUN_DISK_TYPE RUN_DISK_GB SHARED_RUNTIME UV_EXTRAS HOURLY_USD PROJECT_ID >> "$FAKE_GCS/submitted.jsonl"
echo "$1" >> "$FAKE_GCS/commands.log"
"""
STEP_CODES = """import sys
assert sys.argv[1] == "--before-fit"
for step in ("samples", "store", "phenotypes", "cohort", "measurement", "direct_sv"):
    print(f"{step}\\t{'differs' if step == 'cohort' else 'same'}\\t0\\t1")
"""


def _repository(root: Path, step_codes: bool = False, accepts: bool = True, partner: bool = False, driver: str = "", artifact: str = "") -> tuple[Path, str]:
    """A git repository standing in for the study's: the files round.py reads of a commit (``partner``: its task has
    ag-score's PARTNER_RUN; ``driver``: more of its workspace_pipeline.py; ``artifact``: its sv_pgs/artifact.py)."""
    files = {
        "sv_pgs/workspace_pipeline.py": ("accept_steps_from = None\n" if accepts else "STEPS = ()\n") + driver,
        "sv_pgs/store.py": "STORE = 1\n",
        "launcher/workspace/run_step.sh.template": 'STAGED_SOURCE base=(--base "x")\n' + ("PARTNER_RUN\n" if partner else ""),
        "launcher/workspace/submit.sh": "PARALLEL_ARMS\n",
        "uv.lock": "version = 1\n",
        "pyproject.toml": "[project]\nname = 'sv-pgs'\n",
    }
    if step_codes:
        files["launcher/workspace/step_codes.py"] = STEP_CODES
    if artifact:
        files["sv_pgs/artifact.py"] = artifact
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    identity = ["-c", "user.name=fixture", "-c", "user.email=fixture@example.test"]
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), *identity, "commit", "-q", "-m", "fixture"], check=True)
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    return root, commit


def _spec(commit: str, count: int = 11, **changes) -> dict:
    spec = {
        "round": "r7", "source_commit": commit, "layout": "arm_jobs",
        "traits": [{"name": name, "code": code, "base_run": f"aou-w{code}5"} for name, code in TRAITS[:count]],
        "machine_type": "a2-highgpu-2g", "gpu_type": "nvidia-tesla-a100", "gpu_count": 2,
    }
    spec.update(changes)
    return spec


def _bucket(tmp_path: Path, repository: Path, codes: str, changed: str = "") -> Path:
    """The ops bucket's base runs, one per trait code: each run's config, its staged source (the repository's tree, with
    ``changed`` appended to its sv_pgs/store.py) and env.tar, and every step before the fit completed in its mirror."""
    runs = tmp_path / "fake/gcs" / BUCKET / "svpgs"
    for code in codes:
        run = runs / f"aou-w{code}5"
        name = next(name for name, letter in TRAITS if letter == code)
        config = {"marker_directory": f"/mnt/disks/ops/svpgs/aou-w{code}5/markers", "diseases": [name], "traits": [], "seed": 7,
                  "chromosomes": ["chr1"], "panel": {"primary_fold": 0, "weights": {name: 1.0}}}
        run.mkdir(parents=True)
        (run / "run_config.json").write_text(json.dumps(config))
        subprocess.run(["git", "-C", str(repository), "archive", "-o", str(run / "source.tar"), "HEAD"], check=True)
        (run / "source").mkdir()
        with tarfile.open(run / "source.tar") as archive:
            archive.extractall(run / "source")
        (run / "source.tar").unlink()
        (run / "source/sv_pgs/store.py").write_text("STORE = 1\n" + changed)
        (run / "env.tar").write_bytes(b"environment" * 100)
        for step in planner.PREPARED:
            (run / "run" / step / "part").mkdir(parents=True)
            (run / "run" / step / "part/data.bin").write_bytes(step.encode() * 10)
            (run / "run" / step / planner.COMPLETE).write_text(json.dumps({"key": f"{step}-key"}))
            (run / "run" / f"{step}{planner.DIRECTORIES}").write_text('["part"]')
    return tmp_path / "fake"


@pytest.fixture
def cloud(tmp_path, monkeypatch):
    """The fake gcloud on PATH and the workspace's settings in the environment."""
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "gcloud").write_text(f"#!{sys.executable}\n{FAKE_GCLOUD}")
    (tools / "gcloud").chmod(0o755)
    (tools / "submit.sh").write_text(FAKE_SUBMIT)
    (tmp_path / "fake").mkdir()
    monkeypatch.setenv("PATH", f"{tools}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_GCS", str(tmp_path / "fake"))
    for name in planner.WORKSPACE_ENVIRONMENT:
        monkeypatch.setenv(name, BUCKET if name == "OPS_BUCKET" else "x")
    return tools


def _calls(fake: Path) -> list[str]:
    return (fake / "calls.log").read_text().splitlines() if (fake / "calls.log").exists() else []


def test_round_py_runs_on_the_workspace_python_and_knows_the_pipeline_s_steps() -> None:
    """Standard library and Python 3.8 syntax (the workspace app's python3), and the steps, arms and upstream links of
    the pipeline it plans."""
    source = (LAUNCHER / "round.py").read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 8))
    imported = {alias.name.split(".")[0] for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom) and node.module}
    assert imported <= set(sys.stdlib_module_names) | {"launcher", "spend", "round_cost", "__future__"}
    assert planner.STEPS == workspace_pipeline.STEP_NAMES and planner.ARMS == workspace_pipeline.ARMS
    upstream = {step.name: step.upstream for step in workspace_pipeline._STEPS}
    assert {step: tuple(upstream[step]) for step in planner.PREPARED} == planner.UPSTREAM
    assert planner.PREPARED == workspace_pipeline.STEP_NAMES[:workspace_pipeline.STEP_NAMES.index("fit")]
    assert {"BASE_RUN", "STAGED_SOURCE", "SHARED_RUNTIME", "RUN_DISK_TYPE", "FIT_ARMS"} <= {
        word for line in (LAUNCHER / "submit.sh").read_text().splitlines() for word in line.split() if word.isupper()
    }


def test_a_spec_takes_the_round_s_defaults_and_refuses_what_would_move_a_key_or_misplace_a_job(tmp_path) -> None:
    spec = planner.read_spec(_spec("abc1234"))
    assert (spec["reuse"], spec["max_retries"], spec["run_disk_type"], spec["run_disk_gb"], spec["provisioning_model"]) == ("base", 3, "local-ssd", 1500, "SPOT")
    assert spec["reuse_steps"] == list(planner.PREPARED) and spec["arms"] == list(planner.ARMS) and spec["env_from"] == "aou-wl5"
    assert planner.read_spec(_spec("abc1234", provisioning_model="STANDARD"))["max_retries"] == 0
    # The fit's shape: the layout's where the spec names none of it (one A100 for arm_jobs), a named A2's GPUs from its
    # name; canaries by a trait's code or name.
    unshaped = {key: value for key, value in _spec("abc1234").items() if key not in planner.SHAPE_KEYS}
    assert {key: planner.read_spec(unshaped)[key] for key in planner.SHAPE_KEYS} == planner.ONE_GPU
    assert {key: planner.read_spec(dict(unshaped, layout="trait_jobs"))[key] for key in planner.SHAPE_KEYS} == planner.TWO_GPU
    named = planner.read_spec(dict(unshaped, machine_type="a2-ultragpu-1g"))
    assert (named["gpu_count"], named["gpu_type"]) == (1, "nvidia-a100-80gb")
    assert planner.read_spec(dict(unshaped, canaries=["ldl_cholesterol:snv_indel", "l:with_sv"]))["canaries"] == ["l:snv_indel", "l:with_sv"]
    # Warm starts by default, each trait from its base run unless it names its own warm_from runs; none where the spec
    # turns them off or names warm_start_models itself. The spec's own trait entries are left as they were.
    raw = _spec("abc1234", count=2)
    raw["traits"][1]["warm_from"] = ["aou-we6", "aou-we5"]
    spec = planner.read_spec(raw)
    assert spec["warm_start"] is True and [trait["warm_from"] for trait in spec["traits"]] == [["aou-wl5"], ["aou-we6", "aou-we5"]]
    assert "warm_from" not in raw["traits"][0]
    assert planner.warm_patterns(spec["traits"]) == [f"/mnt/disks/ops/svpgs/{run}/run/fit*/models/{{arm}}/{{trait}}" for run in ("aou-wl5", "aou-we6", "aou-we5")]
    assert all(trait["warm_from"] == [] for trait in planner.read_spec(_spec("abc1234", warm_start=False))["traits"])
    assert all(trait["warm_from"] == [] for trait in planner.read_spec(_spec("abc1234", config={"warm_start_models": ["/x/{arm}/{trait}"]}))["traits"])
    for changes, message in (
        ({"warm_start": False}, "warm_start is false"), ({"config": {"warm_start_models": []}}, "config sets warm_start_models"),
        ({"warm_start": True, "warm_from_value": ["aou-wl1", "aou-wl1"]}, "each once"), ({"warm_start": True, "warm_from_value": ["Bad"]}, "warm_from"),
    ):
        raw = _spec("abc1234", count=1, **{key: value for key, value in changes.items() if key != "warm_from_value"})
        raw["traits"][0]["warm_from"] = changes.get("warm_from_value", ["aou-wl1"])
        with pytest.raises(planner.RoundError, match=message):
            planner.read_spec(raw)
    with pytest.raises(planner.RoundError, match="true .* or false"):
        planner.read_spec(_spec("abc1234", warm_start="yes"))
    for changes, message in (
        ({"config": {"seed": 3}}, "moves a reused step's key"), ({"layout": "one_vm"}, "layout"), ({"run_disk_gb": 1000}, "multiple of 375"),
        ({"merge_slot": "train-1"}, "own slot"), ({"wanted": 1}, "unknown keys"), ({"arms": ["with_sv", "with_sv"]}, "arms"),
        ({"traits": [{"name": "asthma", "code": "a", "base_run": "x"}, {"name": "gout", "code": "a", "base_run": "y"}]}, "code is its own"),
        ({"canaries": ["af:snv_indel"]}, "a canary is TRAIT:ARM"), ({"canaries": ["l:snv_indel", "ldl_cholesterol:snv_indel"]}, "once"),
        ({"canaries": ["l:snv_indel"], "layout": "trait_jobs"}, "arms of arm_jobs"), ({"layout": "arm_vms", "store_machine_type": 32}, "store_machine_type"),
    ):
        with pytest.raises(planner.RoundError, match=message):
            planner.read_spec(_spec("abc1234", **changes))


def test_the_eleven_trait_round_on_each_timeline_with_its_minutes(tmp_path, capsys) -> None:
    """Offline (no bucket read): 22 runs and jobs and 11 merge jobs (the fixture's task has no PARTNER_RUN), every job
    with the round's settings, and the budget's minutes inside the 150-minute goal and today's not."""
    repository, commit = _repository(tmp_path / "repo")
    (tmp_path / "round.json").write_text(json.dumps(_spec(commit)))
    plans = {}
    for timeline in ("budget", "today", "projected"):
        plan = planner.build(planner.read_spec(_spec(commit)), *planner.read_timeline(timeline), None, repository)
        plans[timeline] = planner.projection(plan)
        shown = planner.render(plan)
        assert len(plan.runs) == len(plan.jobs) == 22 and len(plan.merges) == 11
        assert "offline: the bucket was not read" in shown and "dry run: nothing was changed" in shown
    for entry, (run_id, traits, arms) in zip(plan.jobs, planner.runs_of(planner.read_spec(_spec(commit)))):
        assert entry["environment"]["RUN_ID"] == run_id == f"aou-r7-{traits[0]['code']}{planner.ARM_DIGITS[arms[0]]}"
        assert {key: entry["environment"][key] for key in ("SHARED_RUNTIME", "RUN_DISK_TYPE", "RUN_DISK_GB", "MAX_RETRIES", "PROVISIONING_MODEL",
                                                           "STAGED_SOURCE", "FIT_ARMS", "BUDGET_SLOT", "HOURLY_USD")} == {
            "SHARED_RUNTIME": "1", "RUN_DISK_TYPE": "local-ssd", "RUN_DISK_GB": "1500", "MAX_RETRIES": "3", "PROVISIONING_MODEL": "SPOT",
            "STAGED_SOURCE": "_rounds/r7", "FIT_ARMS": arms[0], "BUDGET_SLOT": "train-1", "HOURLY_USD": "5"}
    merges = [entry["environment"] for entry in plan.merges]
    assert all("FIT_ARMS" not in environment and environment["BUDGET_SLOT"] == "train-2" for environment in merges)
    # Budget: 5 + 8 + 0 + 1 + 8 + 10 + 75 = 107 a fit job (its scoring in the merge job), 15 a merge job: 122 against the
    # goal of 150, 132 preempted once.
    assert plans["budget"]["job"] == (107, 107) and plans["budget"]["wall"] == (122, 122) and plans["budget"]["wall_preempted"] == (132, 132)
    assert plans["today"]["wall"] == (261, 566) and plans["today"]["wall"][0] > planner.TARGET_MINUTES
    assert planner.main([str(tmp_path / "round.json"), "--offline", "--repository", str(repository), "--timeline", "today"]) == 0
    shown = capsys.readouterr().out
    assert "goal 150: not met" in shown and "over its budget of 75" in shown and "svpgs-aou-r7-l1-train-1" in shown
    # ag-cost's tuple per job, with its trait, kind and the jobs it waits for.
    listed = planner.cost_jobs(plan)
    name, machine, model, disk, gigabytes, phases, extra = listed[0]
    assert (name, machine, model, disk, gigabytes) == ("svpgs-aou-r7-l1-train-1", "a2-highgpu-2g", "SPOT", "local-ssd", 1500) and phases["eb"] == (2, 28)
    assert extra == {"trait": "ldl_cholesterol", "kind": "fit", "after": []} and len(listed) == 33
    assert listed[22][5] == {"merge": (23, 40)} and listed[22][6] == {"trait": "ldl_cholesterol", "kind": "merge", "after": ["svpgs-aou-r7-l1-train-1", "svpgs-aou-r7-l2-train-1"]}


def test_the_other_layouts_one_run_per_trait_or_per_arm(tmp_path) -> None:
    repository, commit = _repository(tmp_path / "repo")
    plan = planner.build(planner.read_spec(_spec(commit, layout="trait_jobs", one_gpu_slowdown=1.3)), *planner.read_timeline("budget"), None, repository)
    assert len(plan.jobs) == 11 and not plan.merges and plan.jobs[0]["environment"]["PARALLEL_ARMS"] == "1"
    assert "FIT_ARMS" not in plan.jobs[0]["environment"]
    rows = {phase: (low, high) for phase, low, high, _note in planner.fit_rows(plan.timeline, plan.spec)}
    assert rows["eb"] == pytest.approx((97.5, 97.5)) and rows["score"] == (15, 15)
    assert planner.read_spec(_spec(commit, layout="arm_vms"))["store_machine_type"] == "n2d-highmem-32"
    plan = planner.build(planner.read_spec(_spec(commit, layout="arm_vms", store_machine_type="n2d-standard-32")), *planner.read_timeline("budget"), None, repository)
    assert [entry["environment"]["RUN_ID"] for entry in plan.jobs] == ["aou-r7-all1", "aou-r7-all2"] and len(plan.merges) == 1
    assert plan.store["run_id"] == "aou-r7-all0" and all(entry["after"] == [plan.store["name"]] for entry in plan.jobs)
    assert {entry["environment"]["BASE_RUN"] for entry in plan.jobs} == {"aou-r7-all0"}
    assert plan.store["environment"]["THROUGH"] == "direct_sv" and "GPU_TYPE" not in plan.store["environment"]
    # Each trait in turn: its set-up, Stage 0, band and EB (its scoring in the merge job: no PARTNER_RUN here).
    assert planner.projection(plan)["job"] == (5 + 8 + 0 + 11 * (1 + 8 + 10 + 75),) * 2
    # A trait with its own shape (codes too big for one A100-40) gets its own machine and price.
    traits = _spec(commit)["traits"]
    traits[0].update(machine_type="a2-ultragpu-1g", gpu_type="nvidia-a100-80gb", gpu_count=1, run_disk_type="pd-balanced", run_disk_gb=500)
    plan = planner.build(planner.read_spec(_spec(commit, traits=traits)), *planner.read_timeline("budget"), None, repository)
    assert plan.jobs[0]["environment"]["MACHINE_TYPE"] == "a2-ultragpu-1g" and plan.jobs[0]["rate"] == pytest.approx(3.038823 + 0.1 * 500 / 730)
    assert plan.jobs[2]["environment"]["MACHINE_TYPE"] == "a2-highgpu-2g"


def test_two_canaries_on_one_gpu_the_other_arms_on_two(tmp_path) -> None:
    """One A100 is the round's shape, but no arm has run there: the canaries take it, every other arm the live rounds'
    a2-highgpu-2g, and the plan says so; with no canaries every arm is on one GPU, with a warning."""
    repository, commit = _repository(tmp_path / "repo")
    unshaped = {key: value for key, value in _spec(commit).items() if key not in planner.SHAPE_KEYS}
    plan = planner.build(planner.read_spec(dict(unshaped, canaries=["l:snv_indel", "l:with_sv"])), *planner.read_timeline("budget"), None, repository)
    shapes = [(entry["environment"]["MACHINE_TYPE"], entry["environment"]["GPU_COUNT"]) for entry in plan.jobs]
    assert shapes[:2] == [("a2-highgpu-1g", "1")] * 2 and set(shapes[2:]) == {("a2-highgpu-2g", "2")}
    assert plan.merges[0]["environment"]["MACHINE_TYPE"] == "a2-highgpu-1g" and plan.merges[1]["environment"]["GPU_COUNT"] == "2"
    shown = planner.render(plan)
    assert "canaries: 2 of 22 arms on a2-highgpu-1g (l:snv_indel, l:with_sv), the others on a2-highgpu-2g" in shown
    assert "jobs: a2-highgpu-2g x2" in shown and shown.count("shape: a2-highgpu-1g x1") == 2
    plan = planner.build(planner.read_spec(unshaped), *planner.read_timeline("budget"), None, repository)
    assert {entry["environment"]["MACHINE_TYPE"] for entry in plan.jobs} == {"a2-highgpu-1g"}
    assert any(state == "warn" and "name two canaries first" in text for state, text in plan.checks)


def test_with_partner_run_the_later_arm_finishes_the_run_and_a_cpu_merge_is_the_fallback(tmp_path, cloud) -> None:
    """ag-score's layout: each arm's job scores its own arm and names the other's run, the later finishes the run on
    its VM (priced as a merge on the dearer arm's machine after both), and --follow submits a CPU merge job only where
    both jobs have ended, neither run holds the report and each holds its arm's handoff."""
    repository, commit = _repository(tmp_path / "repo", partner=True)
    plan = planner.build(planner.read_spec(_spec(commit)), *planner.read_timeline("budget"), None, repository)
    assert plan.partner and not plan.merges and len(plan.finishes) == 11
    assert [entry["environment"]["PARTNER_RUN"] for entry in plan.jobs[:2]] == ["aou-r7-l2", "aou-r7-l1"]
    fallback = plan.finishes[0]
    assert fallback["command"] == "merge" and fallback["environment"]["MACHINE_TYPE"] == "n2d-standard-16" and fallback["environment"]["PARTNER_RUN"] == "aou-r7-l2"
    assert "GPU_TYPE" not in fallback["environment"] and fallback["environment"]["UV_EXTRAS"] == ""
    # Budget: 5 + 8 + 0 + 1 + 8 + 10 + 75 + 10 = 117 a fit job, its score in the job, and 5 to finish: 122.
    projected = planner.projection(plan)
    assert projected["job"] == (117, 117) and projected["wall"] == (122, 122)
    listed = planner.cost_jobs(plan)
    assert len(listed) == 33 and listed[22][0] == "svpgs-aou-r7-l1-finish" and listed[22][5] == {"merge": (5, 5)}
    assert listed[22][6]["after"] == ["svpgs-aou-r7-l1-train-1", "svpgs-aou-r7-l2-train-1"] and listed[22][1] == "a2-highgpu-2g"
    assert not planner.build(planner.read_spec(_spec(commit, through="fit")), *planner.read_timeline("budget"), None, repository).partner
    # --follow against the fake bucket and jobs.
    fake = _bucket(tmp_path, repository, "l")
    plan = planner.build(planner.read_spec(_spec(commit, count=1)), *planner.read_timeline("budget"), planner.Gcloud(BUCKET, True), repository)
    (fake / "jobs").mkdir()
    (fake / "jobs/svpgs-aou-r7-l1-train-1").write_text("SUCCEEDED")
    (fake / "jobs/svpgs-aou-r7-l2-train-1").write_text("RUNNING")
    assert planner.follow(plan, planner.Gcloud(BUCKET, True), cloud, "p", "r")[0][1]["svpgs-aou-r7-l2-train-1"] == "RUNNING"
    (fake / "jobs/svpgs-aou-r7-l2-train-1").write_text("FAILED")
    svpgs = fake / "gcs" / BUCKET / "svpgs"
    (svpgs / "aou-r7-l1/run/effects.partial/batches/a1").mkdir(parents=True)
    (svpgs / "aou-r7-l1/run/effects.partial/batches/a1/handoff.json").write_text("{}")
    # aou-r7-l2's job failed without its arm's parts: the merge cannot run, and the follower says so and stops waiting.
    follower = planner.Follower(plan, planner.Gcloud(BUCKET, True), cloud, "p", "r", resubmitting=False)
    assert planner.follow(plan, planner.Gcloud(BUCKET, True), cloud, "p", "r", follower) == []
    assert "aou-r7-l2 no handoff" in follower.ended["svpgs-aou-r7-l1-train-2"][1]
    (svpgs / "aou-r7-l2/run/effects.partial/batches/b2").mkdir(parents=True)
    (svpgs / "aou-r7-l2/run/effects.partial/batches/b2/handoff.json").write_text("{}")
    (svpgs / "aou-r7-l2/run/report").mkdir(parents=True)
    (svpgs / "aou-r7-l2/run/report/_COMPLETE.json").write_text("{}")
    assert planner.follow(plan, planner.Gcloud(BUCKET, True), cloud, "p", "r") == [] and not (fake / "submitted.jsonl").exists()
    (svpgs / "aou-r7-l2/run/report/_COMPLETE.json").unlink()
    assert planner.follow(plan, planner.Gcloud(BUCKET, False), cloud, "p", "r")[0][1]["svpgs-aou-r7-l1-train-1"].endswith("--execute submits it")
    assert planner.follow(plan, planner.Gcloud(BUCKET, True), cloud, "p", "r") == []
    (submitted,) = [json.loads(line) for line in (fake / "submitted.jsonl").read_text().splitlines()]
    assert (submitted["RUN_ID"], submitted["PARTNER_RUN"], submitted["MACHINE_TYPE"]) == ("aou-r7-l1", "aou-r7-l2", "n2d-standard-16")
    assert (fake / "commands.log").read_text().split() == ["merge"]
    # The follower recorded the merge it submitted: while Batch does not show it yet, nothing is submitted again.
    chain = svpgs / "_rounds/r7/follow/chains/svpgs-aou-r7-l1-train-2.json"
    assert [job["name"] for job in json.loads(chain.read_text())["jobs"]] == ["svpgs-aou-r7-l1-train-2"]
    assert planner.follow(plan, planner.Gcloud(BUCKET, True), cloud, "p", "r") == []
    assert len((fake / "submitted.jsonl").read_text().splitlines()) == 1
    chain.unlink()
    # The run that got further is the merge's RUN_ID (ag-score's rule): aou-r7-l2 adopted l1's parts and pushed both
    # arms' handoffs before its finish failed, so the merge finishes in it, not in l1.
    (svpgs / "aou-r7-l2/run/effects.partial/batches/c3").mkdir(parents=True)
    (svpgs / "aou-r7-l2/run/effects.partial/batches/c3/handoff.json").write_text("{}")
    assert planner.follow(plan, planner.Gcloud(BUCKET, True), cloud, "p", "r") == []
    submitted = json.loads((fake / "submitted.jsonl").read_text().splitlines()[-1])
    assert (submitted["RUN_ID"], submitted["PARTNER_RUN"], submitted["BUDGET_SLOT"]) == ("aou-r7-l2", "aou-r7-l1", "train-2")
    # Once a merge job of either run exists, nothing more is submitted.
    (fake / "jobs/svpgs-aou-r7-l2-train-2").write_text("RUNNING")
    assert planner.follow(plan, planner.Gcloud(BUCKET, True), cloud, "p", "r") == []
    assert len((fake / "submitted.jsonl").read_text().splitlines()) == 2


def test_arm_vms_submits_its_store_job_first_and_its_fit_jobs_after_it(tmp_path, cloud) -> None:
    """One run of every trait's cohort (the store job, ag-share's config keys where the driver has them), then a run per
    arm that restores it through BASE_RUN, submitted by --follow once the store job has SUCCEEDED."""
    repository, commit = _repository(tmp_path / "repo", driver="cross_fit = True\ncompact_fit_store = False\n")
    fake = _bucket(tmp_path, repository, "le")
    spec = planner.read_spec(_spec(commit, count=2, layout="arm_vms", store_machine_type="n2d-standard-32"))
    plan = planner.build(spec, *planner.read_timeline("budget"), planner.Gcloud(BUCKET, True), repository)
    assert not [text for state, text in plan.checks if state == "FAIL"]
    assert any(state == "warn" and "no arm_pass" in text for state, text in plan.checks)
    store = plan.runs[0]
    assert store["reuse"] == ["samples", "store"] and store["made"]["cohort"] == "the run's own traits"
    assert store["config"]["cross_fit"] is False and store["config"]["compact_fit_store"] is True and "arm_pass" not in store["config"]
    assert store["config"]["diseases"] == ["ldl_cholesterol", "egfr_ckd_epi_2021"] and plan.store["environment"]["THROUGH"] == "fit_store"
    assert plan.runs[1]["config"] == dict(store["config"], marker_directory="/mnt/disks/ops/svpgs/aou-r7-all1/markers")
    planner.execute(plan, planner.Gcloud(BUCKET, True), cloud)
    assert [json.loads(line)["RUN_ID"] for line in (fake / "submitted.jsonl").read_text().splitlines()] == ["aou-r7-all0"]
    (fake / "jobs").mkdir()
    (fake / "jobs/svpgs-aou-r7-all0-prepare").write_text("RUNNING")
    waiting = planner.follow(plan, planner.Gcloud(BUCKET, True), cloud, "p", "r")
    assert [name for name, _states in waiting] == ["svpgs-aou-r7-all1-train-1", "svpgs-aou-r7-all2-train-1", "svpgs-aou-r7-all1-train-2"]
    (fake / "jobs/svpgs-aou-r7-all0-prepare").write_text("SUCCEEDED")
    planner.follow(plan, planner.Gcloud(BUCKET, True), cloud, "p", "r")
    submitted = [json.loads(line) for line in (fake / "submitted.jsonl").read_text().splitlines()]
    assert [(entry["RUN_ID"], entry["BASE_RUN"], entry["FIT_ARMS"]) for entry in submitted[1:]] == [("aou-r7-all1", "aou-r7-all0", "snv_indel"), ("aou-r7-all2", "aou-r7-all0", "with_sv")]
    assert (fake / "commands.log").read_text().split() == ["store", "fit", "fit"]


def test_the_dollars_come_from_round_cost(tmp_path) -> None:
    """round_cost.py prices the plan's jobs (cost_jobs) against today's layout."""
    repository, commit = _repository(tmp_path / "repo", partner=True)
    plan = planner.build(planner.read_spec(_spec(commit)), *planner.read_timeline("budget"), None, repository)
    shown = "\n".join(planner.dollars(plan))
    assert "launcher/workspace/round_cost.py" in shown and "svpgs-aou-r7-l1-finish" in shown and "today $" in shown


def test_a_dry_run_takes_every_step_from_the_base_and_changes_nothing(tmp_path, cloud, capsys) -> None:
    repository, commit = _repository(tmp_path / "repo")
    fake = _bucket(tmp_path, repository, "le")
    (tmp_path / "round.json").write_text(json.dumps(_spec(commit, count=2)))
    assert planner.main([str(tmp_path / "round.json"), "--repository", str(repository)]) == 0
    shown = capsys.readouterr().out
    assert "[FAIL]" not in shown and "--execute stages, writes, copies and submits" in shown
    assert shown.count("kept: samples store phenotypes cohort measurement direct_sv; remade: -") == 4 and "BASE_RUN=aou-wl5" in shown
    assert shown.count("a record of the earlier key formula kept where /mnt/disks/ops/svpgs/aou-wl5/source made it") == 2
    calls = _calls(fake)
    # Reads only: no copy into the bucket (the base's source comes down to the work directory), nothing submitted.
    assert [call for call in calls if call.startswith("storage cp -r gs://")]
    assert not [call for call in calls if call.startswith("batch jobs submit") or (call.startswith("storage cp") and call.split()[-1].startswith("gs://"))]
    assert not (fake / "submitted.jsonl").exists()
    plan = planner.build(planner.read_spec(_spec(commit, count=2)), *planner.read_timeline("budget"), planner.Gcloud(BUCKET, False), repository)
    config = plan.runs[0]["config"]
    assert config["marker_directory"] == "/mnt/disks/ops/svpgs/aou-r7-l1/markers" and config["accept_steps_from"] == "/mnt/disks/ops/svpgs/aou-wl5/source"
    assert config["diseases"] == ["ldl_cholesterol"] and config["seed"] == 7
    # This fixture's driver has no warm_start_models: every arm cold, said once, and no config names it.
    assert "warm_start_models" not in config and any(state == "warn" and "every arm starts cold" in text for state, text in plan.checks)
    with pytest.raises(RuntimeError, match="outside --execute"):
        planner.Gcloud(BUCKET, False).copy("a", "b")


def test_each_arm_starts_from_its_own_earlier_model_and_the_plan_names_each_start(tmp_path, cloud) -> None:
    """warm_start_models by default: a run's config names its trait's warm_from runs' saved models, the same patterns
    for both arms so their fit keys agree, and the plan reads the bucket for each arm's start: warm from a mean-field
    model on the store the run keeps, cold where its runs hold none (and why), and a trait whose two arms would start
    unlike fails the plan (the study's rule: both cold, or each from its own arm)."""
    repository, commit = _repository(tmp_path / "repo", driver="warm_start_models = ()\n", artifact='MODEL_FORMAT = "svpgs-model v3"\n')
    fake = _bucket(tmp_path, repository, "le")
    svpgs = fake / "gcs" / BUCKET / "svpgs"
    # A live pair before its merge: aou-wl5 holds with_sv's model in its fit partial and aou-wl1 snv_indel's, on a copy
    # of the same store.
    shutil.copytree(svpgs / "aou-wl5/run/store", svpgs / "aou-wl1/run/store")

    def save(run: str, directory: str, arm: str, laws=("product_mixture",), accepted: bool = True, model_format: str = "svpgs-model v3") -> Path:
        folder = svpgs / run / "run" / directory / "models" / arm / "ldl_cholesterol"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model.json").write_text(json.dumps({"format": model_format, "draw_laws": list(laws), "acceptance": {"accepted": accepted}, "provenance": {}}))
        (folder / "arrays.npz").write_bytes(b"a")
        return folder

    save("aou-wl5", "fit.partial", "with_sv")
    save("aou-wl1", "fit.partial", "snv_indel", accepted=False)
    raw = _spec(commit, count=2)
    raw["traits"][0]["warm_from"] = ["aou-wl5", "aou-wl1"]
    raw["traits"][1]["warm_from"] = []

    def plan_of():
        return planner.build(planner.read_spec(raw), *planner.read_timeline("budget"), planner.Gcloud(BUCKET, False), repository)

    plan = plan_of()
    patterns = [f"/mnt/disks/ops/svpgs/{run}/run/fit*/models/{{arm}}/{{trait}}" for run in ("aou-wl5", "aou-wl1")]
    assert [run["config"].get("warm_start_models") for run in plan.runs] == [patterns, patterns, None, None]
    assert plan.runs[0]["starts"] == {("ldl_cholesterol", "snv_indel"): ("warm", "svpgs/aou-wl1/run/fit.partial/models/snv_indel/ldl_cholesterol (a fit not accepted)")}
    assert plan.runs[1]["starts"] == {("ldl_cholesterol", "with_sv"): ("warm", "svpgs/aou-wl5/run/fit.partial/models/with_sv/ldl_cholesterol (an accepted fit)")}
    assert plan.runs[3]["starts"] == {("egfr_ckd_epi_2021", "with_sv"): ("cold", "no warm_from run named")}
    assert not [text for state, text in plan.checks if state == "FAIL"]
    assert ("warn", "warm starts: 2 of 4 arms start from an earlier model, 2 cold (each named under its run)") in plan.checks
    shown = planner.render(plan)
    assert "warm snv_indel ldl_cholesterol: svpgs/aou-wl1/run/fit.partial/models/snv_indel/ldl_cholesterol (a fit not accepted)" in shown
    assert "cold with_sv egfr_ckd_epi_2021: no warm_from run named" in shown
    # One arm's model on another store, or not a mean-field model: that arm is cold, and the trait's arms would start
    # unlike, which fails the plan.
    (svpgs / "aou-wl1/run/store" / planner.COMPLETE).write_text(json.dumps({"key": "another-store"}))
    plan = plan_of()
    assert plan.runs[0]["starts"][("ldl_cholesterol", "snv_indel")] == ("cold", "no saved model in aou-wl5; aou-wl1's store is not the one this run keeps")
    assert [text for state, text in plan.checks if state == "FAIL"] == [
        "warm starts: each trait's arms start alike, but not ldl_cholesterol: name the run holding the cold arm's model in its warm_from, or [] for both cold"]
    (svpgs / "aou-wl1/run/store" / planner.COMPLETE).write_text(json.dumps({"key": "store-key"}))
    save("aou-wl1", "fit.partial", "snv_indel", laws=("gaussian",))
    assert plan_of().runs[0]["starts"][("ldl_cholesterol", "snv_indel")][1].endswith("aou-wl1/run/fit.partial/models/snv_indel/ldl_cholesterol holds no mean-field model")
    save("aou-wl1", "fit.partial", "snv_indel", model_format="svpgs-model v2")
    assert plan_of().runs[0]["starts"][("ldl_cholesterol", "snv_indel")][1].endswith("ldl_cholesterol is a 'svpgs-model v2' model, not 'svpgs-model v3'")
    # After the merge both arms' models are in one run's completed fit: each arm finds two, and the job ranks them.
    save("aou-wl1", "fit.partial", "snv_indel")
    save("aou-wl5", "fit", "snv_indel")
    assert plan_of().runs[0]["starts"][("ldl_cholesterol", "snv_indel")] == (
        "warm", "svpgs/aou-wl5/run/fit/models/snv_indel/ldl_cholesterol (an accepted fit; 2 models, the job takes the best)")
    # Offline the patterns are named and the starts unchecked.
    plan = planner.build(planner.read_spec(raw), *planner.read_timeline("budget"), None, repository)
    assert "starts unchecked (the bucket was not read): warm_start_models " + " ".join(patterns) in planner.render(plan)


def test_a_step_whose_code_differs_and_the_steps_reading_it_are_the_job_s_to_make(tmp_path, cloud, monkeypatch) -> None:
    """A base staged from other sv_pgs: the commit's step_codes.py says which steps' code differs (here the cohort's);
    without it nothing can be taken, and the round is refused rather than converting the store again in every job."""
    repository, commit = _repository(tmp_path / "repo", step_codes=True)
    _bucket(tmp_path, repository, "l", changed="CHANGED = 1\n")
    plan = planner.build(planner.read_spec(_spec(commit, count=1)), *planner.read_timeline("budget"), planner.Gcloud(BUCKET, False), repository)
    run = plan.runs[0]
    assert run["reuse"] == ["samples", "store", "phenotypes"]
    assert run["made"] == {"cohort": "its code differs from the base's", "measurement": "reads cohort, made again", "direct_sv": "reads cohort, measurement, made again"}
    assert not [text for state, text in plan.checks if state == "FAIL"]
    bare, commit = _repository(tmp_path / "bare")
    _bucket(tmp_path / "other", bare, "l", changed="CHANGED = 1\n")
    monkeypatch.setenv("FAKE_GCS", str(tmp_path / "other/fake"))
    plan = planner.build(planner.read_spec(_spec(commit, count=1)), *planner.read_timeline("budget"), planner.Gcloud(BUCKET, False), bare)
    failed = [text for state, text in plan.checks if state == "FAIL"]
    assert any("convert the store again" in text for text in failed) and any("unchecked" in text for text in failed)
    with pytest.raises(planner.RoundError, match="refused"):
        planner.execute(plan, planner.Gcloud(BUCKET, True), cloud)


def test_a_base_that_carries_its_steps_through_its_own_base_is_followed(tmp_path, cloud) -> None:
    """A base made by a round (its source the round's staging, named by its STAGED_SOURCE object) that holds only its
    samples and carries the rest through its own base (.bases): every step is found where a job's restore finds it,
    and accept_steps_from names the source of the run holding the store."""
    repository, commit = _repository(tmp_path / "repo")
    fake = _bucket(tmp_path, repository, "l")
    runs = fake / "gcs" / BUCKET / "svpgs"
    shutil.move(runs / "aou-wl5", runs / "aou-wl4")
    shutil.copytree(runs / "aou-wl4/run/samples", runs / "aou-wl5/run/samples")
    (runs / "aou-wl5/run/.bases" / BUCKET / "svpgs/aou-wl4").mkdir(parents=True)
    (runs / "aou-wl5/run/.bases" / BUCKET / "svpgs/aou-wl4/run").write_text("")
    shutil.copy(runs / "aou-wl4/run_config.json", runs / "aou-wl5/run_config.json")
    (runs / "aou-wl5/STAGED_SOURCE").write_text("_rounds/r6\n")
    shutil.copytree(runs / "aou-wl4/source", runs / "_rounds/r6/source")
    plan = planner.build(planner.read_spec(_spec(commit, count=1)), *planner.read_timeline("budget"), planner.Gcloud(BUCKET, False), repository)
    run = plan.runs[0]
    assert run["reuse"] == list(planner.PREPARED) and run["holders"] == {step: "aou-wl5" if step == "samples" else "aou-wl4" for step in planner.PREPARED}
    assert run["config"]["accept_steps_from"] == "/mnt/disks/ops/svpgs/aou-wl4/source"
    assert any(state == "warn" and "but samples are held by a run staged from _rounds/r6/source" in text for state, text in plan.checks)
    assert plan.jobs[0]["environment"]["BASE_RUN"] == "aou-wl5"


def test_execute_stages_once_writes_each_run_and_submits_every_job(tmp_path, cloud) -> None:
    repository, commit = _repository(tmp_path / "repo")
    fake = _bucket(tmp_path, repository, "le")
    plan = planner.build(planner.read_spec(_spec(commit, count=2)), *planner.read_timeline("budget"), planner.Gcloud(BUCKET, True), repository)
    planner.execute(plan, planner.Gcloud(BUCKET, True), cloud)
    svpgs = fake / "gcs" / BUCKET / "svpgs"
    staged = svpgs / "_rounds/r7"
    assert planner.tree_files(staged / "source") == plan.files and (staged / "env.tar").read_bytes() == (svpgs / "aou-wl5/env.tar").read_bytes()
    lines = (staged / "source.sha256").read_text().splitlines()
    assert sorted(line.split("  ", 1)[1] for line in lines) == sorted(plan.files)
    assert all(hashlib.sha256((staged / "source" / name).read_bytes()).hexdigest() == digest for digest, name in (line.split("  ", 1) for line in lines))
    for run_id in ("aou-r7-l1", "aou-r7-l2", "aou-r7-e1", "aou-r7-e2"):
        assert json.loads((svpgs / run_id / "run_config.json").read_text())["marker_directory"] == f"/mnt/disks/ops/svpgs/{run_id}/markers"
        assert (svpgs / run_id / "STAGED_SOURCE").read_text() == "_rounds/r7\n" and not (svpgs / run_id / "run").exists()
    submitted = [json.loads(line) for line in (fake / "submitted.jsonl").read_text().splitlines()]
    assert [entry["RUN_ID"] for entry in submitted] == ["aou-r7-l1", "aou-r7-l2", "aou-r7-e1", "aou-r7-e2"]
    assert {entry["BASE_RUN"] for entry in submitted[:2]} == {"aou-wl5"} and submitted[1]["FIT_ARMS"] == "with_sv"
    assert json.loads((staged / "round.json").read_text())["jobs"] == [f"svpgs-{entry['RUN_ID']}-train-1" for entry in submitted]
    # A second round of the same name is another tree: refused before anything is written.
    (tmp_path / "repo/sv_pgs/store.py").write_text("STORE = 2\n")
    subprocess.run(["git", "-C", str(tmp_path / "repo"), "-c", "user.name=f", "-c", "user.email=f@e.t", "commit", "-qam", "next"], check=True)
    other = subprocess.run(["git", "-C", str(tmp_path / "repo"), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    with pytest.raises(planner.RoundError, match="another tree"):
        planner.stage(planner.build(planner.read_spec(_spec(other, count=1)), *planner.read_timeline("budget"), None, repository), planner.Gcloud(BUCKET, True))


def test_copy_mode_copies_each_step_whole_before_its_record(tmp_path, cloud) -> None:
    repository, commit = _repository(tmp_path / "repo")
    fake = _bucket(tmp_path, repository, "l")
    plan = planner.build(planner.read_spec(_spec(commit, count=1, reuse="copy", arms=["with_sv"])), *planner.read_timeline("budget"), planner.Gcloud(BUCKET, True), repository)
    assert "BASE_RUN" not in plan.jobs[0]["environment"] and len(plan.copies) == len(planner.PREPARED)
    planner.execute(plan, planner.Gcloud(BUCKET, True), cloud)
    svpgs = fake / "gcs" / BUCKET / "svpgs"
    for step in planner.PREPARED:
        assert planner.tree_files(svpgs / "aou-r7-l2/run" / step) == planner.tree_files(svpgs / "aou-wl5/run" / step)
        assert (svpgs / "aou-r7-l2/run" / f"{step}{planner.DIRECTORIES}").is_file()
    calls = _calls(fake)
    for step in planner.PREPARED:
        record = next(index for index, call in enumerate(calls) if call.startswith("storage cp") and call.endswith(f"aou-r7-l2/run/{step}/{planner.COMPLETE}"))
        objects = [index for index, call in enumerate(calls) if call.startswith("storage cp") and f"aou-r7-l2/run/{step}/" in call.split()[-1]]
        assert calls[objects[0]].startswith("storage cp -r") and calls[objects[0]].endswith(f"aou-r7-l2/run/{step}/")
        assert objects[-1] == record and all(index < record for index in objects[:-1])


def test_without_partner_run_follow_waits_for_both_arms_then_moves_the_models_and_submits_the_merge(tmp_path, cloud, capsys) -> None:
    repository, commit = _repository(tmp_path / "repo")
    fake = _bucket(tmp_path, repository, "l")
    plan = planner.build(planner.read_spec(_spec(commit, count=1)), *planner.read_timeline("budget"), planner.Gcloud(BUCKET, True), repository)
    (fake / "jobs").mkdir()
    (fake / "jobs/svpgs-aou-r7-l1-train-1").write_text("SUCCEEDED")
    (fake / "jobs/svpgs-aou-r7-l2-train-1").write_text("RUNNING")
    waiting = planner.follow(plan, planner.Gcloud(BUCKET, True), cloud, "p", "r")
    assert waiting == [("svpgs-aou-r7-l1-train-2", {"svpgs-aou-r7-l1-train-1": "SUCCEEDED", "svpgs-aou-r7-l2-train-1": "RUNNING"})]
    (fake / "jobs/svpgs-aou-r7-l2-train-1").write_text("SUCCEEDED")
    svpgs = fake / "gcs" / BUCKET / "svpgs"
    for run_id, arm in (("aou-r7-l1", "snv_indel"), ("aou-r7-l2", "with_sv")):
        models = svpgs / run_id / "run/fit.partial/models" / arm
        (models / "ldl_cholesterol").mkdir(parents=True)
        (models / "ldl_cholesterol/weights.npz").write_bytes(b"w" * 7)
        (models / "ldl_cholesterol.json").write_text("{}")
        (svpgs / run_id / "run/fit.partial/_KEY").write_text("fit-key")
    ready = "SUCCEEDED: ready, --execute submits it"
    assert planner.follow(plan, planner.Gcloud(BUCKET, False), cloud, "p", "r")[0][1] == {"svpgs-aou-r7-l1-train-1": ready, "svpgs-aou-r7-l2-train-1": ready}
    assert planner.follow(plan, planner.Gcloud(BUCKET, True), cloud, "p", "r") == []
    merged = svpgs / "aou-r7-l1/run/fit.partial/models/with_sv"
    assert (merged / "ldl_cholesterol/weights.npz").read_bytes() == b"w" * 7 and (merged / "ldl_cholesterol.json").is_file()
    calls = _calls(fake)
    files = next(index for index, call in enumerate(calls) if call.startswith("storage cp -r") and call.endswith("aou-r7-l1/run/fit.partial/models/with_sv/"))
    record = next(index for index, call in enumerate(calls) if call.startswith("storage cp") and call.endswith("models/with_sv/ldl_cholesterol.json"))
    assert files < record
    (submitted,) = [json.loads(line) for line in (fake / "submitted.jsonl").read_text().splitlines()]
    assert submitted["RUN_ID"] == "aou-r7-l1" and submitted["BUDGET_SLOT"] == "train-2" and submitted["FIT_ARMS"] == ""


@pytest.mark.skipif(shutil.which("flock") is None or shutil.which("sha256sum") is None, reason="run_step.sh runs on Linux")
def test_a_job_builds_from_the_round_s_staging_and_stops_on_a_source_that_is_not_the_staged_one(tmp_path) -> None:
    """The rendered task script with STAGED_SOURCE, uv and sv-pgs stubbed as in test_study_budget: the run has no source
    of its own, the job copies and checks the round's, and a staged file changed since is refused before anything runs."""
    launcher, environment = _fake_cloud(tmp_path)
    environment.update(UV_EXTRAS="", STAGED_SOURCE="_rounds/r7/../x")
    refused = subprocess.run(["bash", str(launcher), "fit"], env=environment, capture_output=True, text=True, check=False)
    assert refused.returncode == 2 and "STAGED_SOURCE" in refused.stderr
    environment["STAGED_SOURCE"] = "_rounds/r7"
    submitted = subprocess.run(["bash", str(launcher), "fit"], env=environment, capture_output=True, text=True, check=False)
    assert submitted.returncode == 0, submitted.stderr
    script = json.loads((tmp_path / "objects.json").read_text())["gs://workspace/svpgs/study/launcher/pilot/run_step.sh"]
    ops = tmp_path / "ops/svpgs/study"
    ops.mkdir(parents=True)
    (ops / "run_config.json").write_text(json.dumps({"run_directory": str(tmp_path / "pipeline")}))
    staging = tmp_path / "ops/svpgs/_rounds/r7"
    (staging / "source/launcher/workspace").mkdir(parents=True)
    (staging / "source/uv.lock").write_text("version = 1\n")
    (staging / "source/pyproject.toml").write_text("[project]\nname = 'sv-pgs'\n")
    (staging / "source.sha256").write_text(planner.manifest(planner.tree_files(staging / "source")))
    tools = tmp_path / "tools"
    (tools / "bin").mkdir(parents=True)
    (tools / "bin/uv").write_text(_UV.format(python=sys.executable))
    (tools / "python/cpython-3.12-linux/bin").mkdir(parents=True)
    (tools / "python/cpython-3.12-linux/bin/python3.12").write_text("#!/bin/sh\n")
    with tarfile.open(staging / "env.tar", "w") as archive:
        for part in ("bin", "python"):
            archive.add(tools / part, arcname=part)
    for job, changed in (("first", False), ("second", True)):
        if changed:
            (staging / "source/uv.lock").write_text("version = 2\n")
        run = tmp_path / job
        run.mkdir()
        task = tmp_path / f"{job}-launcher/run_step.sh"
        task.parent.mkdir(parents=True)
        task.write_text(script.replace("OPS=/mnt/disks/ops/svpgs/study\n", f"OPS={ops}\n").replace("RUN=/mnt/disks/run\n", f"RUN={run}\n"))
        ran = subprocess.run(["bash", str(task)], env=dict(os.environ, BATCH_TASK_INDEX="0"), capture_output=True, text=True, timeout=120, check=False)
        markers = sorted(path.name.split("_task_", 1)[1] for path in (ops / "markers").iterdir())
        if changed:
            assert ran.returncode == 1 and "exit-source" in markers and not (run / "sv-pgs-calls").exists()
        else:
            assert ran.returncode == 0, ran.stderr
            assert (run / "source/uv.lock").read_text() == "version = 1\n" and not (ops / "source").exists()
            assert (run / "sv-pgs-calls").read_text().splitlines()[1].startswith("workspace-run")
        shutil.rmtree(ops / "markers")
