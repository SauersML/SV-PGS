"""A driver's silent stretches report where its main thread is."""
from __future__ import annotations

import time

from sv_pgs import progress


def _waiting_in_a_named_function(seconds: float) -> None:
    time.sleep(seconds)


def test_a_quiet_stretch_logs_the_main_thread_s_innermost_frame_and_a_line_resets_it(capsys) -> None:
    progress.log("start")
    stop = progress.sample_quiet_stacks(0.05)
    try:
        _waiting_in_a_named_function(0.4)
    finally:
        stop.set()
    lines = [line for line in capsys.readouterr().err.splitlines() if "quiet" in line]
    assert lines and all("main thread at test_progress.py:" in line and "_waiting_in_a_named_function" in line for line in lines)
    # Each report is itself a line, so a silent stretch is sampled once per period, not on every check.
    assert len(lines) <= 0.4 / 0.05
