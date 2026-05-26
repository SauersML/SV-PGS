import subprocess
import sys


def test_doctor_runs():
    r = subprocess.run(
        [sys.executable, "-m", "sv_pgs.cli", "doctor"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    # On a CPU-only host CuPy probe will FAIL and rc=1 is expected; that's OK.
    assert r.returncode in (0, 1)
    assert "[OK]" in r.stdout or "[WARN]" in r.stdout or "[FAIL]" in r.stdout
    assert "sv-pgs doctor:" in r.stdout
