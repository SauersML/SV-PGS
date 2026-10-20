"""A run's progress log reads into its phase timeline, its repeated phases, its empirical Bayes rounds and its stalls
(benchmarks/profile_log.py), with offsets in place of clock times and only the timing lines' own extras.

The log is a synthetic whole-genome run in the driver's line formats: the restore's shell line, the sampler's resource
lines, the driver's timing, curvature, trial and quiet-stack lines, one logging-module line, a traceback and a partly
copied last line.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from benchmarks import profile_log

START = datetime(2026, 10, 20, 10, 0, 0)


def _clock(seconds: float) -> str:
    return (START + timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S")


def _line(seconds: float, message: str, megabytes: int = 1000) -> tuple[float, str]:
    return seconds, f"[{_clock(seconds)} | 1h2m3s | {megabytes} MB] {message}"


def _resources(seconds: float, gpu: int, disk: float) -> tuple[float, str]:
    return seconds, (
        f"[{_clock(seconds)}] resources: gpu0 {gpu}% 31.2/40.0 GiB, gpu1 {gpu}% 30.8/40.0 GiB; cpu 10% iowait 0% load 1.0; "
        f"mem 10.0/170.0 GiB; disk r {disk:.1f} w 0.0 MiB/s; net rx 0.0 tx 0.0 MiB/s"
    )


def _whole_genome_run() -> list[str]:
    """The restore (0-1020 s), the store digest, Stage 0, a summary band that is a 900 s stall at 5% GPU, the resident
    fill, three mean-field solves and two empirical Bayes rounds (the second a halved step), two scoring batches, and a
    stall still going at the log's end."""
    lines = []
    for at in range(60, 10801, 60):
        gpu, disk = next((gpu, disk) for end, gpu, disk in (
            (1020, 0, 500.0), (1745, 0, 700.0), (4445, 80, 300.0), (5345, 5, 0.0), (6545, 30, 900.0), (9000, 95, 50.0),
            (10000, 60, 20.0), (10800, 0, 0.0),
        ) if at <= end)
        lines.append(_resources(at, gpu, disk))
    lines.append((1020, f"[{_clock(1020)}] timing: store_restore 1020 s"))
    for at in range(1025, 10000, 300):
        if not 4445 < at <= 5345:
            lines.append(_line(at, "stage 0: block done", 20480 if 1745 < at < 4445 else 1000))
    lines.append((3000, f"{_clock(3000)},123 sv_pgs.workspace_pipeline INFO step stage0 running"))
    for at in range(4565, 5346, 120):
        lines.append(_line(at, "quiet 120 s; main thread at gram_space.py:700 build_band_store < stage2_wiring.py:600 _band < fit_model.py:10 fit < cli.py:5 main < runpy.py:1 run"))
    for at in range(10120, 10801, 120):
        lines.append(_line(at, "quiet 120 s; main thread at workspace_pipeline.py:3438 _score_step < workspace_pipeline.py:100 run < cli.py:5 main"))
    lines += [
        _line(1745, "timing: tree_digest 720.0 s files=500 gb=547 root=run/store"),
        _line(4445, "timing: stage0 2700.0 s blocks=55 ld_gb=150 genotype_pass=2400 samples=48123"),
        _line(5345, "timing: summary_band 900.0 s blocks=55 parts=98 gb=90"),
        _line(6545, "timing: resident_fill 1200.0 s gb=12 pool=device:0"),
        _line(7145, "timing: mean_field_solve 600.0 s model=0 outcome=converged sweeps=12 per_sweep=50"),
        _line(7265, "timing: mean_field_solve 120.0 s model=1 outcome=converged sweeps=3 per_sweep=40"),
        _line(7365, "timing: mean_field_solve 100.0 s model=2 outcome=converged sweeps=2 per_sweep=50"),
        _line(7400, "eb curvature: 3 directions over 1000 variants, 1 classes x 140 nodes; derivatives 2.0 s, response 18.0 s"),
        _line(7500, "timing: eb_iteration 955.0 s round=1 solve=820 other=135"),
        _line(7600, "eb outer: model 0 joint trial at fraction 0.5 of a step predicted 1: realized 0.9 (resolution 0.01), trial state error 0.001, decrement 0.1: accepted"),
        _line(7700, "eb curvature: 3 directions over 1000 variants, 1 classes x 140 nodes; derivatives 3.0 s, response 17.0 s"),
        _line(8000, "timing: eb_iteration 500.0 s round=2 solve=400 other=100"),
        _line(8000, "timing: eb 1455.0 s rounds=2 solve=1220"),
        _line(9500, "timing: score_batch 500.0 s models=6 load=50 predict=440"),
        _line(10000, "timing: score_batch 500.0 s models=6 load=40 predict=450"),
        (5000, "Traceback (most recent call last):"),
    ]
    ordered = [text for _, text in sorted(lines, key=lambda entry: entry[0])]
    return [line + "\n" for line in ordered] + [f"[{_clock(10860)[:13]}"]


def _row(report: str, name: str) -> list[str]:
    return next(line for line in report.splitlines() if line.split()[:1] == [name]).split()


def test_the_timeline_puts_each_phase_at_its_start_with_its_resources() -> None:
    profile = profile_log.parse(_whole_genome_run())
    report = profile_log.report(profile)
    # the restore's line comes 17 minutes in, after the sampler's: its start is the log's start
    assert _row(report, "store_restore")[1:3] == ["+0:00:00", "1020.0"]
    assert _row(report, "store_restore")[3:9] == ["0", "0", "10", "500.0", "0.0", "0.0"]
    assert _row(report, "tree_digest")[1:3] == ["+0:17:05", "720.0"] and _row(report, "tree_digest")[6] == "700.0"
    stage0 = _row(report, "stage0")
    # GPU median and p95, the driver's peak resident memory (20480 MB), its extras
    assert stage0[1:5] == ["+0:29:05", "2700.0", "80", "80"] and stage0[9] == "20.0"
    assert stage0[10:] == ["blocks=55", "ld_gb=150", "genotype_pass=2400"]
    assert _row(report, "resident_fill")[1:4] == ["+1:29:05", "1200.0", "30"] and _row(report, "resident_fill")[6] == "900.0"
    assert _row(report, "eb")[1:5] == ["+1:49:05", "1455.0", "95", "95"]
    # the repeated phases are summarized, not listed
    assert not any(line.startswith("mean_field_solve ") and "+" in line for line in report.splitlines())


def test_repeated_phases_are_summarized_and_listed_with_all() -> None:
    profile = profile_log.parse(_whole_genome_run())
    report = profile_log.report(profile)
    solves = _row(report, "mean_field_solve")
    assert solves[1:6] == ["3", "820.0", "120.0", "600.0", "600.0"] and "sweeps=3" in solves and "per_sweep=50" in solves
    assert _row(report, "score_batch")[1:3] == ["2", "1000.0"] and "predict=445" in _row(report, "score_batch")
    assert _row(report, "eb_iteration")[1:3] == ["2", "1455.0"]
    listed = profile_log.report(profile, show_all=True)
    assert sum(line.startswith("mean_field_solve ") and "+" in line for line in listed.splitlines()) == 3


def test_an_eb_loop_splits_into_solves_curvature_the_rest_and_its_line_search() -> None:
    report = profile_log.report(profile_log.parse(_whole_genome_run()))
    (loop,) = [line for line in report.splitlines() if line.startswith("eb loop")]
    assert loop.startswith(
        "eb loop 1: 1455.0 s from +1:49:05, 2 rounds: solve 1220.0 s, curvature 40.0 s (derivatives 5.0 s), other 195.0 s; "
        "line search 1 rounds, 500.0 s; gpu% median 95"
    )


def test_a_silent_stretch_with_idle_gpus_is_a_stall_with_its_quiet_frames() -> None:
    report = profile_log.report(profile_log.parse(_whole_genome_run()))
    stalls = report.split("\n\n")[-1].splitlines()
    assert stalls[0] == "stall +1:14:05 for 900 s: gpu% max 5, disk r 0.0 MiB/s, net rx 0.0 MiB/s, 15 resource lines"
    # the innermost frames of the quiet stacks, by how often they were seen
    assert stalls[1] == "  quiet x7: gram_space.py:700 build_band_store < stage2_wiring.py:600 _band < fit_model.py:10 fit < cli.py:5 main"
    assert stalls[2].startswith("stall +2:46:40 for 800 s to the log end: gpu% max 0,")
    assert stalls[3] == "  quiet x6: workspace_pipeline.py:3438 _score_step < workspace_pipeline.py:100 run < cli.py:5 main"
    assert len(stalls) == 4


def test_a_silent_stretch_with_busy_gpus_is_not_a_stall_and_one_without_resource_lines_is() -> None:
    busy = [_line(0, "start")[1]] + [_resources(at, 90, 0.0)[1] for at in range(60, 1201, 60)] + [_line(1200, "next")[1]]
    assert profile_log._stalls(profile_log.parse(busy), 600.0, 20.0) == []
    # busy for its first five minutes, then eleven idle: the idle part is the stall
    late = [_line(0, "start")[1]] + [_resources(at, 90 if at <= 300 else 3, 0.0)[1] for at in range(60, 1021, 60)] + [_line(1020, "next")[1]]
    assert profile_log._stalls(profile_log.parse(late), 600.0, 20.0)[0].startswith("stall +0:05:00 for 720 s: gpu% max 3,")
    unsampled = [_line(0, "start")[1], _line(1200, "next")[1]]
    assert profile_log._stalls(profile_log.parse(unsampled), 600.0, 20.0)[0].startswith("stall +0:00:00 for 1200 s: gpu% max -,")


def test_the_report_has_no_clock_time_and_no_extra_outside_its_list() -> None:
    report = profile_log.report(profile_log.parse(_whole_genome_run()), show_all=True)
    assert "2026" not in report and "10:00:00" not in report
    # an extra the report does not list (here a count of people) is never printed
    assert "48123" not in report and "samples" not in report


def test_a_log_from_before_the_timing_lines_still_times_its_solves(capsys, tmp_path: Path) -> None:
    log = tmp_path / "progress-task0.log"
    log.write_text("".join(
        _line(at, f"mean-field solve: model 0 converged after {sweeps} sweeps in {seconds:.1f} s, 30 passes")[1] + "\n"
        for at, sweeps, seconds in ((600, 12, 600.0), (700, 3, 90.0))
    ))
    assert profile_log.main([str(log)]) == 0
    solves = _row(capsys.readouterr().out, "mean_field_solve")
    assert solves[1:3] == ["2", "690.0"] and "per_sweep=40" in solves
