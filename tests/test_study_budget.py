"""Budget validation and duplicate submission protection, with no cloud calls."""
import copy
import json
from pathlib import Path
import subprocess
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
