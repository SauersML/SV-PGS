"""Run a budget-planned computation under an address-space limit taken from its own working-memory budget.

Linux overcommit lets a process reserve far more memory than it ever touches: ``np.empty`` of an n x n
basis that is barely written costs address space but no resident pages. A resident-memory limit never sees
such a reservation, although a machine without overcommit refuses it (scipy's gmres with restart = n once
reserved about 150 GiB this way). ``run_within_budget`` runs a function in a fresh interpreter whose
RLIMIT_AS is the address space it holds once its module, its arguments and BLAS are loaded, plus the
``working_bytes`` the computation was given. Any allocation past the plan then raises MemoryError, touched
or not, and the test fails.

Only numpy/scipy code belongs here. DuckDB, jemalloc and CUDA reserve address space they never commit, so
their processes have no meaningful address-space budget.

The child runs single-threaded with one glibc arena and a fixed mmap threshold, so that its address space
is the computation's own: no per-thread BLAS buffers, thread stacks or malloc arenas, and no heap growth
from glibc raising its mmap threshold after a large free.
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

# glibc's documented default M_MMAP_THRESHOLD (mallopt(3)). Setting it explicitly turns off the dynamic
# threshold, so large blocks keep coming from mmap and return to the system when freed.
_GLIBC_MMAP_THRESHOLD = 128 * 1024

_CHILD = r"""
import pickle, resource, sys, traceback
payload_path, result_path = sys.argv[1], sys.argv[2]
with open(payload_path, "rb") as handle:
    search_path, module_name, qualified_name, args, kwargs, working_bytes = pickle.load(handle)
sys.path[:0] = search_path
import importlib
import numpy as np
function = importlib.import_module(module_name)
for part in qualified_name.split("."):
    function = getattr(function, part)
# BLAS allocates its buffers on first use; do that before the baseline so they are not charged to the plan.
np.ones((2, 2)) @ np.ones((2, 2))
with open("/proc/self/status", encoding="ascii") as status:
    baseline = next(int(line.split()[1]) * 1024 for line in status if line.startswith("VmSize:"))
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
limit = baseline + working_bytes
if hard != resource.RLIM_INFINITY and hard < limit:
    raise SystemExit(f"the hard address-space limit {hard} is below the budget's {limit}")
resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
try:
    outcome = ("ok", function(*args, **kwargs), baseline)
except MemoryError:
    outcome = ("memory", traceback.format_exc(), baseline)
except BaseException as error:
    outcome = ("error", error, baseline)
resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
with open(result_path, "wb") as handle:
    pickle.dump(outcome, handle)
"""


def run_within_budget(function: Callable[..., Any], *args: Any, working_bytes: int, **kwargs: Any) -> Any:
    """``function(*args, **kwargs)`` in a child whose address space may grow by at most ``working_bytes``.

    ``function`` must be importable by module and qualified name, and its arguments and result picklable.
    Arguments are loaded before the limit is set, so they are not charged to the budget. An exception other
    than MemoryError is raised again here. A MemoryError fails the test: the computation reserved address
    space past the budget it plans with.
    """
    if not working_bytes > 0:
        raise ValueError("working_bytes must be positive.")
    environment = dict(os.environ)
    environment.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        MALLOC_ARENA_MAX="1",
        MALLOC_MMAP_THRESHOLD_=str(_GLIBC_MMAP_THRESHOLD),
    )
    with tempfile.TemporaryDirectory() as directory:
        payload_path = Path(directory) / "payload.pickle"
        result_path = Path(directory) / "result.pickle"
        with payload_path.open("wb") as handle:
            pickle.dump(
                (list(sys.path), function.__module__, function.__qualname__, args, kwargs, int(working_bytes)), handle
            )
        completed = subprocess.run(
            [sys.executable, "-c", _CHILD, str(payload_path), str(result_path)],
            env=environment,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not result_path.exists():
            raise AssertionError(
                f"{function.__qualname__} ended with exit status {completed.returncode} under its "
                f"{working_bytes}-byte budget:\n{completed.stderr}"
            )
        with result_path.open("rb") as handle:
            status, value, baseline = pickle.load(handle)
    if status == "memory":
        raise AssertionError(
            f"{function.__qualname__} reserved address space past its working_bytes budget of {working_bytes} bytes "
            f"(above the {baseline} bytes held after setup):\n{value}"
        )
    if status == "error":
        raise value
    return value
