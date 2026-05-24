"""Entry point for `python -m sv_pgs <command>`.

Exists so the orchestrator in `aou_runner.run_all_of_us_all_diseases` can
spawn per-disease subprocesses with the current interpreter (`sys.executable
-m sv_pgs ...`) without depending on the `sv-pgs` console-script being on
PATH inside the child's environment.
"""
from __future__ import annotations

import sys

from sv_pgs.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
