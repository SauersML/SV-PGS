"""VM integration of the whole-genome canary on tiny inputs, with the real EB stopping rules.

These are separate from the sub-minute generator/contract tests. CUDA skips cleanly where unavailable; a numerical
failure, incomplete mixture, wrong route or deadline is a failure. Nothing here launches a cloud job.
"""
from __future__ import annotations

import json

import pytest

from benchmarks import round_timeline
from benchmarks import wg_canary as canary
from sv_pgs import workspace_pipeline as pipeline


@pytest.mark.slow
@pytest.mark.parametrize("device,gpus", [("cpu", 1), ("cuda", 1), ("cuda", 2)])
def test_real_eb_two_arm_handoff_to_report(tmp_path, device, gpus):
    """No fitter double or sweep cap: the same command as Batch on a tiny input, including its completion gates.

    This is a VM integration test, deliberately separate from the sub-minute generator/contract tests. A route mismatch,
    unresolved EB or a deadline is a failure, not a skip or a successful early exit. Only missing CUDA skips.
    """
    if device == "cuda":
        try:
            cupy = pytest.importorskip("cupy")
            available = cupy.cuda.runtime.getDeviceCount()
        except (RuntimeError, OSError):
            pytest.skip("CUDA runtime is unavailable")
        if available < gpus:
            pytest.skip("the requested CUDA devices are unavailable")
    out = tmp_path / "canary"
    # One SV in eight records (the whole-genome default is one in twenty): with one in four on 128 to 512 people, the
    # with_sv arm's empirical Bayes had not ended after 4 to 7 minutes in the VM, its lattice check unresolved again and
    # again, where this scenario's whole run took about 30 s.
    assert canary.main([
        "--out", str(out), "--columns", "128", "--samples", "1024", "--chromosomes", "2", "--block-columns", "32",
        "--sv-every", "8", "--device", device, "--gpus", str(gpus), "--host-gib", "1", "--device-gib", "1",
        "--deadline-seconds", "600", "--allow-band-accelerator",
    ]) == 0
    assert json.loads((out / "result.json").read_text())["passed"]
    for arm in pipeline.ARMS:
        metrics = json.loads((out / f"{arm}.metrics.json").read_text())
        assert metrics["eb_calls"] and metrics["mode_mixture"] and metrics["full_data"] and metrics["loops_checked_for_warm_starts"]
    assert (out / "merge" / "report" / pipeline.COMPLETE_FILE).is_file()
    # One gate reads the canary's log as a round's: four driver runs (the preparation, each arm's fit, the merge to the
    # report), each step with its line, each group's phases inside its fit.
    timeline = round_timeline.read((out / "progress.log").read_text().splitlines())
    assert round_timeline.problems(timeline) == []
    assert [(run[0].name, run[-1].name, run[-1].state) for run in timeline.drivers()] == [
        ("samples", "direct_sv", "computed"), ("samples", "fit", "part"), ("samples", "fit", "part"), ("samples", "report", "computed")]
    assert {phase.name for phase in timeline.job} >= {"canary_generate", "canary_prepare", "canary_merge"}
    assert "step_report" in (out / "profile.txt").read_text() and "no problem" in (out / "profile.txt").read_text()
