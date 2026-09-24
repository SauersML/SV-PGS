"""bench-sim submission: ``svpgs_full`` by the summary route (``gram_space``: every step after Stage 0 on the summary
band, its far field in the likelihood) whatever the measured pass costs say, for the accuracy gate against the exact
route (``benchmarks.summary_gate``) and its wall time. The process's peak memory is logged after the fit and the score:
the peak resident set (``getrusage``'s ``ru_maxrss``) and, on a CUDA device, the peaks of the CuPy pool's held bytes
and of the device's used bytes, sampled by a daemon thread (a lower bound on the true peaks at its period)."""
from __future__ import annotations

import resource
import threading
import time

from benchmarks.bench_sim.submissions import svpgs_full
from sv_pgs import stage2_wiring
from sv_pgs.progress import log

# The measured choice (``gram_space.pass_costs``) replaced by the arm's route: a Gram pass is taken as free.
stage2_wiring.pass_costs = lambda band, source, array_module: (float("inf"), 0.0)

_PEAKS = {"pool_bytes": 0, "device_used_bytes": 0}
_SAMPLE_SECONDS = 0.25
"""The sampler's period: the device peaks are the largest values seen at this resolution."""


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


def report(stage: str) -> None:
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    log(
        f"memory peak after {stage}: rss {peak_rss / 1e9:.2f} GB, cupy pool {_PEAKS['pool_bytes'] / 1e9:.2f} GB, "
        f"device used {_PEAKS['device_used_bytes'] / 1e9:.2f} GB"
    )


class Model(svpgs_full.Model):
    def score(self, test) -> dict:
        result = super().score(test)
        report("score")
        return result


def fit(train) -> Model:
    # The sampler starts with the fit: polling CuPy before the device's context exists raced its creation.
    threading.Thread(target=_sample, daemon=True).start()
    fitted = svpgs_full.fit(train)
    report("fit")
    return Model(fitted.scoring, fitted.order, fitted.structural, fitted.profile)
