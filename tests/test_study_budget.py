"""Budget validation and duplicate submission protection, with no cloud calls."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from launcher.workspace import budget_guard as guard


@pytest.fixture
def plan():
    return json.loads((Path(__file__).parents[1] / "launcher/workspace/study_budget.template.json").read_text())


def job(seconds, retries=0):
    return {
        "taskGroups": [{"taskCount": 1, "taskSpec": {"maxRetryCount": retries, "maxRunDuration": f"{seconds}s"}}],
        "allocationPolicy": {"instances": [{"policy": {"machineType": "a2-highgpu-1g", "provisioningModel": "SPOT"}}]},
    }


def test_all_attempts_plus_startup_fit_inside_each_slot(plan):
    for rate in (0.1, 2.4, 5.07, 29.0):
        for retries in (0, 1, 5):
            if plan["slots"]["train-1"] / (rate * (retries + 1)) <= 1:
                continue
            seconds = guard.duration_seconds(plan, "train-1", rate, retries)
            assert (seconds / 3600 + 1) * rate * (retries + 1) <= 325
            assert seconds * (retries + 1) <= 14 * 24 * 3600
    assert sum(plan["slots"].values()) + plan["non_task_reserve_usd"] == 3000


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf"), True])
def test_invalid_quotes_cannot_launch(plan, rate):
    with pytest.raises(ValueError):
        guard.duration_seconds(plan, "train-1", rate, 0)


def test_over_budget_plans_and_unbounded_jobs_are_refused(plan):
    excessive = copy.deepcopy(plan)
    excessive["slots"]["train-1"] += 1
    with pytest.raises(ValueError, match="exceed"):
        guard.validate_plan(excessive)
    excessive["total_usd"] = 3001
    with pytest.raises(ValueError, match="3,000"):
        guard.validate_plan(excessive)
    with pytest.raises(ValueError, match="timeout"):
        guard.validate_job(job(999999), 3600, 0)
    multiplied = job(3600)
    multiplied["taskGroups"][0]["taskCount"] = 2
    with pytest.raises(ValueError, match="one task"):
        guard.validate_job(multiplied, 3600, 0)


class Storage:
    def __init__(self):
        self.objects = {}
        self.lock = threading.Lock()

    def __call__(self, arguments, **kwargs):
        with self.lock:
            assert arguments[:2] == ["gcloud", "storage"]
            if arguments[2] == "cat":
                result = subprocess.CompletedProcess(arguments, 0, self.objects[arguments[3]], b"")
            else:
                assert arguments[-1] == "--if-generation-match=0"
                source, target = arguments[3:5]
                exists = target in self.objects
                if not exists:
                    self.objects[target] = Path(source).read_bytes()
                result = subprocess.CompletedProcess(arguments, int(exists), b"", b"")
            if kwargs.get("check") and result.returncode:
                raise subprocess.CalledProcessError(result.returncode, arguments)
            return result


def test_two_launchers_cannot_spend_the_same_slot(plan):
    storage = Storage()
    seconds = guard.duration_seconds(plan, "pilot", 2.4, 0)
    def launch():
        try:
            guard.reserve(plan, "pilot", job(seconds), 2.4, 0, "gs://workspace/study/budget", run=storage)
            return "reserved"
        except RuntimeError:
            return "refused"
    with ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(lambda _: launch(), range(2))) == ["refused", "reserved"]
    assert len(storage.objects) == 2


def test_a_new_slot_cannot_reset_the_study_envelope(plan):
    storage = Storage()
    seconds = guard.duration_seconds(plan, "pilot", 2.4, 0)
    guard.reserve(plan, "pilot", job(seconds), 2.4, 0, "gs://workspace/study/budget", run=storage)
    changed = copy.deepcopy(plan)
    changed["slots"]["another-pilot"] = changed["slots"].pop("pilot")
    with pytest.raises(ValueError, match="cannot be reset"):
        guard.reserve(changed, "another-pilot", job(seconds), 2.4, 0, "gs://workspace/study/budget", run=storage)
    assert len(storage.objects) == 2


def test_render_preserves_runtime_variables_and_refuses_shell_injection(tmp_path, monkeypatch):
    template = tmp_path / "in"
    output = tmp_path / "out"
    template.write_text('${RUN_ID}: $RUN ${status} $(date) $!')
    monkeypatch.setenv("RUN_ID", "study-1")
    guard.render(template, output, ["RUN_ID"])
    assert output.read_text() == 'study-1: $RUN ${status} $(date) $!'
    monkeypatch.setenv("RUN_ID", "$(cat /private)")
    with pytest.raises(ValueError, match="unsafe"):
        guard.render(template, output, ["RUN_ID"])


def _fake_cloud(tmp_path):
    """A local fake gcloud (storage cp/cat with generation preconditions, batch submit), never a cloud account, and the
    launcher's environment pointing at it."""
    executable = tmp_path / "gcloud"
    executable.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
from pathlib import Path
root = Path(os.environ["FAKE_CLOUD"])
database = root / "objects.json"
objects = json.loads(database.read_text()) if database.exists() else {}
args = sys.argv[1:]
if args[:2] == ["storage", "cp"]:
    source, target = args[2:4]
    if "--if-generation-match=0" in args and target in objects:
        sys.exit(1)
    objects[target] = Path(source).read_text()
    database.write_text(json.dumps(objects))
elif args[:2] == ["storage", "cat"]:
    print(objects[args[2]])
elif args[:3] == ["batch", "jobs", "submit"]:
    spec = next(value.split("=", 1)[1] for value in args if value.startswith("--config="))
    with (root / "submitted.jsonl").open("a") as output:
        output.write(json.dumps(json.loads(Path(spec).read_text())) + "\\n")
else:
    raise RuntimeError(args)
''')
    executable.chmod(0o755)
    launcher = Path(__file__).parents[1] / "launcher/workspace/submit.sh"
    environment = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"], FAKE_CLOUD=str(tmp_path))
    environment.update(
        PROJECT_ID="test-project", REGION="us-central1", ZONE="us-central1-a", OPS_BUCKET="workspace",
        RUN_ID="study", RUN_DISK="disk", HOURLY_USD="2.4", THROUGH="fit", BUDGET_SLOT="pilot",
        WORKSPACE_CDR="cdr", GOOGLE_PROJECT="test-project", UV_EXTRAS="--extra gpu",
        MACHINE_TYPE="a2-highgpu-1g", PROVISIONING_MODEL="SPOT", GPU_TYPE="nvidia-tesla-a100",
        GPU_COUNT="1", NETWORK="network", SUBNETWORK="subnetwork", SERVICE_ACCOUNT="account@example.test",
        MAX_RETRIES="0", STUDY_BUDGET=str(launcher.with_name("study_budget.template.json")),
    )
    return launcher, environment


def test_submit_script_reserves_once_and_preserves_task_shell(tmp_path):
    """Exercise the real shell/CLI integration against a local fake gcloud, never a cloud account."""
    launcher, environment = _fake_cloud(tmp_path)
    first = subprocess.run(["bash", str(launcher), "fit"], env=environment, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    second = subprocess.run(["bash", str(launcher), "fit"], env=environment, capture_output=True, text=True)
    assert second.returncode != 0 and "already used" in second.stderr
    submitted = (tmp_path / "submitted.jsonl").read_text().splitlines()
    assert len(submitted) == 1
    spec = json.loads(submitted[0])
    guard.validate_job(spec, 146400, 0)
    script = json.loads((tmp_path / "objects.json").read_text())["gs://workspace/svpgs/study/launcher/pilot/run_step.sh"]
    assert "$RUN/venv/bin/sv-pgs" in script and "${status}" in script and "${RUN_ID}" not in script
    # The fit job shares the store job's run disk and venv, so its sync with the GPU extras runs on every start, outside
    # the one-time staging copy, and a GPU it cannot reach stops it before the pipeline starts.
    staging_end = script.index("\nfi\n", script.index('if [ ! -f "$RUN/tools/staged" ]'))
    assert script.index("sync --project") > staging_end and "sync --project" in script and "--extra gpu" in script
    assert script.index("exit-no-gpu") < script.index("workspace-run")


def test_the_store_job_mounts_the_workspace_cdr_read_only(tmp_path):
    launcher, environment = _fake_cloud(tmp_path)
    environment.update(THROUGH="direct_sv", BUDGET_SLOT="prepare", MACHINE_TYPE="n2d-standard-32")
    environment.pop("CDR_STORAGE_PATH", None)
    refused = subprocess.run(["bash", str(launcher), "store"], env=environment, capture_output=True, text=True)
    assert refused.returncode != 0 and "CDR_STORAGE_PATH" in refused.stderr
    environment["CDR_STORAGE_PATH"] = "gs://fc-aou-datasets-controlled/v8/"
    submitted = subprocess.run(["bash", str(launcher), "store"], env=environment, capture_output=True, text=True)
    assert submitted.returncode == 0, submitted.stderr
    spec = json.loads((tmp_path / "submitted.jsonl").read_text().splitlines()[-1])
    volumes = {volume["mountPath"]: volume for volume in spec["taskGroups"][0]["taskSpec"]["volumes"]}
    cdr = volumes["/mnt/disks/cdr"]
    assert cdr["gcs"]["remotePath"] == "fc-aou-datasets-controlled/v8"
    assert "-o ro" in cdr["mountOptions"] and "--billing-project=test-project" in cdr["mountOptions"]
