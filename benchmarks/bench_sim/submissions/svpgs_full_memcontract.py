"""bench-sim submission: ``svpgs_full`` under its own results name, with the process's peak memory measured: the peak
resident set (``getrusage``'s ``ru_maxrss``, page-locked host buffers included) and, on a CUDA device, the peak of the
CuPy pool's held bytes and of the device's used bytes (total less free), sampled by a daemon thread for the whole run.
Written for the memory contract's before/after measurement; the fit and the score are ``svpgs_full``'s own."""
from __future__ import annotations

import resource
import threading
import time

from benchmarks.bench_sim.submissions import svpgs_full
from sv_pgs.progress import log

_PEAKS = {"pool_bytes": 0, "device_used_bytes": 0}
_SAMPLE_SECONDS = 0.25
"""The sampler's period: the device peaks are the largest values seen at this resolution (a lower bound on the true
peak; an allocation that lives for less than a period can be missed)."""


def _sample() -> None:
    try:
        import cupy
    except ImportError:
        return
    pool = cupy.get_default_memory_pool()
    while True:
        try:
            free, total = cupy.cuda.runtime.memGetInfo()
            _PEAKS["pool_bytes"] = max(_PEAKS["pool_bytes"], int(pool.total_bytes()))
            _PEAKS["device_used_bytes"] = max(_PEAKS["device_used_bytes"], int(total) - int(free))
        except Exception:  # noqa: BLE001 - a measurement thread must never stop the run
            return
        time.sleep(_SAMPLE_SECONDS)


threading.Thread(target=_sample, daemon=True).start()


def report(stage: str) -> None:
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    log(
        f"memcontract peak after {stage}: rss {peak_rss / 1e9:.2f} GB, cupy pool {_PEAKS['pool_bytes'] / 1e9:.2f} GB, "
        f"device used {_PEAKS['device_used_bytes'] / 1e9:.2f} GB"
    )


class Model(svpgs_full.Model):
    def score(self, test) -> dict:
        result = super().score(test)
        report("score")
        return result


def fit(train) -> Model:
    fitted = svpgs_full.fit(train)
    report("fit")
    return Model(fitted.scoring, fitted.order, fitted.structural, fitted.profile)
