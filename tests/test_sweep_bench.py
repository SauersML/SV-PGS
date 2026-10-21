"""The sweep benchmark's harness on the host (``--host``: the new route on numpy with its reference kernels): its
checks run and pass, and its failures reach the exit status. The CUDA run is test_sweep_models_cuda's."""
from __future__ import annotations

from types import SimpleNamespace

from benchmarks import sweep_bench as bench


def test_the_host_harness_runs_its_checks_and_passes(capfd):
    assert bench.main(["--host", "--samples", "300", "--members", "256", "--blocks", "3"]) == 0
    out = capfd.readouterr().out
    assert "FAIL" not in out and "PASS sweep_bench " in out
    for check in (
        "new sweep 1 of 3 models", "new sweep 3 of 3 models", "new sweep, host share (no kernels)", "model 0 alone is its sweep among three, bit for bit",
        "new sweep 1 of 9 models: model 0 is its sweep among three, bit for bit", "new sweep 2 of 9 models",
        "an arm's 11 traits, each standardized on its training rows (ArmTraits)",
        "arm sweep 1 of 11 traits, each its own standardization, members and lattice (builds the panel Grams)", "arm sweep 3 of 11 traits",
        "arm trait 0 alone is its sweep among 11, bit for bit", "arm sweep 2 of trait 0 alone",
    ):
        assert f"PASS {check}" in out
    assert "INFO new: cold" in out and "INFO new, 9 models a launch" in out and "INFO arm, 11 traits a launch" in out
    result = next(line for line in out.splitlines() if line.startswith("RESULT seconds per sweep"))
    assert "; arm: " in result and "for 11 traits in one pass against" in result


def test_a_failed_worker_or_a_deadline_fails_the_command(monkeypatch, capsys):
    def failed(command, **kwargs):
        assert command[1:4] == ["-m", "benchmarks.sweep_bench", "--worker"] and kwargs["timeout"] == bench.TIMEOUT_SECONDS
        assert command[command.index("--devices") + 1] == "1"
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(bench.subprocess, "run", failed)
    assert bench.main([]) == 1
    assert "FAIL sweep_bench" in capsys.readouterr().out

    def late(command, **kwargs):
        raise bench.subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(bench.subprocess, "run", late)
    assert bench.main([]) == 1
    assert "FAIL deadline" in capsys.readouterr().out
