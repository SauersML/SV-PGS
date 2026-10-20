"""The staged smoke command runs without pytest, on one device and sharded across two where visible.

Run these on Linux with the GPU extra. Each subprocess has its own cold-start deadline, and disabling pytest
imports in the worker makes an accidental dependency on a test module fail even on a developer's environment.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from sv_pgs.compute_budget import _try_import_cupy

cupy = _try_import_cupy()
DEVICE_COUNT = 0 if cupy is None else int(cupy.cuda.runtime.getDeviceCount())
pytestmark = pytest.mark.skipif(DEVICE_COUNT == 0, reason="needs a CUDA device")


@pytest.mark.slow
@pytest.mark.parametrize("devices", [1, 2])
def test_gpu_smoke_without_pytest(devices, tmp_path):
    if DEVICE_COUNT < devices:
        pytest.skip("needs two CUDA devices")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    ids = visible.split(",") if visible else [str(index) for index in range(DEVICE_COUNT)]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(ids[:devices]), OMP_NUM_THREADS="2", NUMBA_NUM_THREADS="2",
                       OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    script = (
        "import sys; sys.modules['pytest'] = None; "
        "from pathlib import Path; from benchmarks.gpu_smoke import main; "
        "raise SystemExit(main(['--worker', sys.argv[1]]))"
    )
    completed = subprocess.run([sys.executable, "-c", script, str(tmp_path)], env=environment,
                               cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=300)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "FAIL " not in completed.stdout
    for trait in ("quantitative", "binary"):
        for check in ("CUDA full-data fit", "CPU full-data fit", "CPU/CUDA mean", "CPU/CUDA elbo", "CPU/CUDA coefficients", "CPU/CUDA smoothing"):
            assert f"PASS {trait} {check} " in completed.stdout
        assert f"PASS {trait} resident blocks/map_reduce ({devices} GPUs) " in completed.stdout
