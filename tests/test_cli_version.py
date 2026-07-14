import subprocess
import sys


def test_version_runs():
    r = subprocess.run([sys.executable, "-m", "sv_pgs.cli", "version"], capture_output=True, text=True, timeout=5)
    assert r.returncode == 0
    assert "sv-pgs" in r.stdout
