"""Budget validation and duplicate submission protection, with no cloud calls."""
import copy
import json
import os
import re
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
            assert (seconds / 3600 + 1) * rate * (retries + 1) <= plan["slots"]["train-1"]
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
    with pytest.raises(ValueError, match="one VM each"):
        guard.validate_job(multiplied, 3600, 0)


def test_a_fan_out_reserves_every_task_s_vm(plan):
    """A store job of 22 chromosome tasks, each on its own VM: every attempt of every task fits inside the slot, and
    the job must run exactly the reserved tasks, one per VM."""
    seconds = guard.duration_seconds(plan, "train-1", 2.5, 1, 22)
    assert (seconds / 3600 + 1) * 2.5 * 22 * 2 <= plan["slots"]["train-1"] < (seconds / 3600 + 1.01) * 2.5 * 22 * 2
    fan_out = job(seconds, 1)
    fan_out["taskGroups"][0].update(taskCount=22, parallelism=22)
    # Batch packs tasks without taskCountPerNode, so a fan-out must name one per VM.
    with pytest.raises(ValueError, match="one task per VM"):
        guard.validate_job(fan_out, seconds, 1, 22)
    fan_out["taskGroups"][0]["taskCountPerNode"] = 1
    guard.validate_job(fan_out, seconds, 1, 22)
    with pytest.raises(ValueError, match="one VM each"):
        guard.validate_job(fan_out, seconds, 1)
    for change in ({"parallelism": 11}, {"taskCountPerNode": 2}):
        changed = copy.deepcopy(fan_out)
        changed["taskGroups"][0].update(change)
        with pytest.raises(ValueError, match="multiply"):
            guard.validate_job(changed, seconds, 1, 22)
    with pytest.raises(ValueError, match="tasks"):
        guard.duration_seconds(plan, "train-1", 2.5, 1, 0)


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
        PROJECT_ID="test-project", REGION="us-central1", OPS_BUCKET="workspace",
        RUN_ID="study", RUN_DISK_GB="2000", HOURLY_USD="2.4", THROUGH="fit", BUDGET_SLOT="pilot",
        WORKSPACE_CDR="cdr", GOOGLE_PROJECT="test-project", UV_EXTRAS="--extra gpu", CDR_STORAGE_PATH="gs://cdr-bucket/v8",
        IMPUTED_BUCKET="imputed-bucket", PANEL_BUCKET="panel-bucket",
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
    # The sync with the GPU extras runs on every start, outside the one-time staging copy, and a GPU the job cannot
    # reach stops it before the pipeline starts.
    staging_end = script.index("\nfi\n", script.index('if [ ! -f "$RUN/tools/staged" ]'))
    assert script.index("sync --project") > staging_end and "sync --project" in script and "--extra gpu" in script
    assert script.index("exit-no-gpu") < script.index("workspace-run")
    # Batch cannot write to an existing disk: the job's disk is new, and the run is restored from the workspace bucket
    # before the driver and pushed back after it.
    assert script.index("mirror restore") < script.index("workspace-run") < script.rindex("mirror push")
    assert '--mirror "gs://workspace/svpgs/study/run"' in script
    # Input files on the bucket mounts are keyed by their objects' generations, not a pass over their bytes.
    assert (
        """SVPGS_GCS_MOUNTS='{"/mnt/disks/ops": "gs://workspace", "/mnt/disks/cdr": "gs://cdr-bucket/v8", """
        """"/mnt/disks/imputed": "gs://imputed-bucket", "/mnt/disks/panel": "gs://panel-bucket"}'"""
    ) in script
    task = spec["taskGroups"][0]["taskSpec"]
    assert task["runnables"][0]["script"]["text"] == "bash /mnt/disks/ops/svpgs/study/launcher/pilot/run_step.sh"
    assert spec["logsPolicy"]["logsPath"] == "/mnt/disks/ops/svpgs/study/launcher/pilot/batch.log"
    (disk,) = spec["allocationPolicy"]["instances"][0]["policy"]["disks"]
    assert disk == {"deviceName": "run", "newDisk": {"type": "pd-balanced", "sizeGb": 2000}}
    # Any zone of the region with capacity takes the job.
    assert spec["allocationPolicy"]["location"] == {"allowedLocations": ["regions/us-central1"]}
    ops = next(volume for volume in task["volumes"] if volume["mountPath"] == "/mnt/disks/ops")
    assert ops["mountOptions"] == ["--implicit-dirs"]
    # Batch takes one lifecycle policy: exit 1 (the driver's or the install's failure) fails the task, and any other
    # non-zero exit (a spot preemption's 50001, the TERM trap's 75) is retried up to maxRetryCount by default.
    assert task["lifecyclePolicies"] == [{"action": "FAIL_TASK", "actionCondition": {"exitCodes": [1]}}]


def test_a_store_fan_out_gives_each_task_its_chromosome_and_the_fit_job_has_one_task(tmp_path):
    launcher, environment = _fake_cloud(tmp_path)
    environment.update(THROUGH="store", BUDGET_SLOT="train-1", TASK_COUNT="22", HOURLY_USD="2.5", MAX_RETRIES="1", UV_EXTRAS="")
    submitted = subprocess.run(["bash", str(launcher), "store"], env=environment, capture_output=True, text=True)
    assert submitted.returncode == 0, submitted.stderr
    spec = json.loads((tmp_path / "submitted.jsonl").read_text().splitlines()[-1])
    group = spec["taskGroups"][0]
    assert group["taskCount"] == 22 and group["parallelism"] == 22 and group["taskCountPerNode"] == 1
    guard.validate_job(spec, guard.duration_seconds(json.loads(Path(environment["STUDY_BUDGET"]).read_text()), "train-1", 2.5, 1, 22), 1, 22)
    script = json.loads((tmp_path / "objects.json").read_text())["gs://workspace/svpgs/study/launcher/train-1/run_step.sh"]
    assert 'shard=(--store-shard "$index")' in script and '--through "store" ${shard[@]+"${shard[@]}"}' in script
    assert '_task_${tag}$1' in script and 'if [ "22" != 1 ] || [ "${#shards[@]}" != 0 ]' in script and "shards=()" in script
    environment.update(THROUGH="report", BUDGET_SLOT="train-2", UV_EXTRAS="--extra gpu")
    refused = subprocess.run(["bash", str(launcher), "fit"], env=environment, capture_output=True, text=True)
    assert refused.returncode != 0 and "TASK_COUNT" in refused.stderr
    # A rerun of the chromosomes a fan-out did not finish names their indices, one per task.
    environment.update(THROUGH="store", BUDGET_SLOT="train-3", UV_EXTRAS="", TASK_COUNT="2", STORE_SHARDS="1 16 3")
    refused = subprocess.run(["bash", str(launcher), "store"], env=environment, capture_output=True, text=True)
    assert refused.returncode != 0 and "STORE_SHARDS" in refused.stderr
    environment["STORE_SHARDS"] = "1 16"
    rerun = subprocess.run(["bash", str(launcher), "store"], env=environment, capture_output=True, text=True)
    assert rerun.returncode == 0, rerun.stderr
    script = json.loads((tmp_path / "objects.json").read_text())["gs://workspace/svpgs/study/launcher/train-3/run_step.sh"]
    assert 'shards=(1 16)' in script and 'index="${shards[$BATCH_TASK_INDEX]:-$BATCH_TASK_INDEX}"' in script
    for task, expected in ((0, "1"), (1, "16")):
        chosen = subprocess.run(
            ["bash", "-c", 'shards=(1 16); BATCH_TASK_INDEX=%d; echo "${shards[$BATCH_TASK_INDEX]:-$BATCH_TASK_INDEX}"' % task],
            capture_output=True, text=True,
        )
        assert chosen.stdout.strip() == expected


def test_a_fit_job_given_its_arms_passes_them_to_the_driver(tmp_path):
    launcher, environment = _fake_cloud(tmp_path)
    environment.update(THROUGH="report", BUDGET_SLOT="train-2", UV_EXTRAS="--extra gpu", FIT_ARMS="with_sv")
    submitted = subprocess.run(["bash", str(launcher), "fit"], env=environment, capture_output=True, text=True)
    assert submitted.returncode == 0, submitted.stderr
    script = json.loads((tmp_path / "objects.json").read_text())["gs://workspace/svpgs/study/launcher/train-2/run_step.sh"]
    assert "for arm in with_sv; do" in script and '${shard[@]+"${shard[@]}"} ${fit_arm[@]+"${fit_arm[@]}"}' in script
    arms = subprocess.run(["bash", "-c", 'fit_arm=(); for arm in snv_indel with_sv; do fit_arm+=(--fit-arm "$arm"); done; echo "${fit_arm[@]}"'], capture_output=True, text=True)
    assert arms.stdout.split() == ["--fit-arm", "snv_indel", "--fit-arm", "with_sv"]
    for refused_arms in ("with_sv snv", "all"):
        environment["FIT_ARMS"] = refused_arms
        refused = subprocess.run(["bash", str(launcher), "fit"], env=environment, capture_output=True, text=True)
        assert refused.returncode != 0 and "FIT_ARMS" in refused.stderr


def test_submit_refuses_a_disk_size_that_is_not_a_whole_number_of_gb(tmp_path):
    launcher, environment = _fake_cloud(tmp_path)
    environment["RUN_DISK_GB"] = "2T"
    refused = subprocess.run(["bash", str(launcher), "fit"], env=environment, capture_output=True, text=True)
    assert refused.returncode != 0 and "RUN_DISK_GB" in refused.stderr and not (tmp_path / "submitted.jsonl").exists()


@pytest.mark.parametrize("job, slot, through", [("store", "prepare", "direct_sv"), ("fit", "pilot", "fit")])
def test_both_jobs_mount_the_workspace_cdr_read_only(tmp_path, job, slot, through):
    """The store job reads the GATK-SV call set there; the fit job hashes it to key the direct_sv step it skips."""
    launcher, environment = _fake_cloud(tmp_path)
    environment.update(THROUGH=through, BUDGET_SLOT=slot)
    environment.pop("CDR_STORAGE_PATH", None)
    refused = subprocess.run(["bash", str(launcher), job], env=environment, capture_output=True, text=True)
    assert refused.returncode != 0 and "CDR_STORAGE_PATH" in refused.stderr
    environment["CDR_STORAGE_PATH"] = "gs://fc-aou-datasets-controlled/v8/"
    submitted = subprocess.run(["bash", str(launcher), job], env=environment, capture_output=True, text=True)
    assert submitted.returncode == 0, submitted.stderr
    spec = json.loads((tmp_path / "submitted.jsonl").read_text().splitlines()[-1])
    volumes = {volume["mountPath"]: volume for volume in spec["taskGroups"][0]["taskSpec"]["volumes"]}
    cdr = volumes["/mnt/disks/cdr"]
    assert cdr["gcs"]["remotePath"] == "fc-aou-datasets-controlled/v8"
    assert "-o ro" in cdr["mountOptions"] and "--billing-project=test-project" in cdr["mountOptions"]
    # The imputed batches with their strata sidecar, and the panel's bubble.split sites, are mounted read-only too.
    assert volumes["/mnt/disks/imputed"]["gcs"]["remotePath"] == "imputed-bucket" and "-o ro" in volumes["/mnt/disks/imputed"]["mountOptions"]
    assert volumes["/mnt/disks/panel"]["gcs"]["remotePath"] == "panel-bucket" and "-o ro" in volumes["/mnt/disks/panel"]["mountOptions"]
    # Either job takes the provisioning model it is given: a store's partial work is not mirrored, so it can run STANDARD.
    assert spec["allocationPolicy"]["instances"][0]["policy"]["provisioningModel"] == "SPOT"
    environment.update(PROVISIONING_MODEL="STANDARD", BUDGET_SLOT="train-1")
    standard = subprocess.run(["bash", str(launcher), job], env=environment, capture_output=True, text=True)
    assert standard.returncode == 0, standard.stderr
    spec = json.loads((tmp_path / "submitted.jsonl").read_text().splitlines()[-1])
    assert spec["allocationPolicy"]["instances"][0]["policy"]["provisioningModel"] == "STANDARD"


def test_staging_refuses_a_run_config_with_a_placeholder_or_a_missing_key(tmp_path: Path) -> None:
    stage = Path(__file__).parents[1] / "launcher/workspace/stage.sh"
    template = (Path(__file__).parents[1] / "launcher/workspace/run_config.template.json").read_text()
    environment = {**os.environ, "OPS_BUCKET": "bucket", "RUN_ID": "run"}
    unfilled = tmp_path / "unfilled.json"
    unfilled.write_text(template)
    result = subprocess.run(["bash", str(stage), str(unfilled)], capture_output=True, text=True, env=environment)
    assert result.returncode != 0 and "placeholders" in result.stderr and "${FOLD_SEED}" in result.stderr
    filled = json.loads(re.sub(r"\$\{[A-Z_0-9]+\}", "0", template))
    del filled["crosswalk"]
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps(filled))
    result = subprocess.run(["bash", str(stage), str(partial)], capture_output=True, text=True, env=environment)
    assert result.returncode != 0 and "lacks the template's keys ['crosswalk']" in result.stderr
