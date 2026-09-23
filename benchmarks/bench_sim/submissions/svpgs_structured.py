"""bench-sim submission: SV-PGS's structured posterior at genome scale (``structured_full.fit_store``) on the arm's
observed codes: a full-covariance Gaussian background with the SV-aware log-variance hierarchy plus local single
effects over Stage 0's LD blocks, one global residual.

The training store, its variant table and annotations, the budget and the scoring are ``svpgs_full``'s (the same
store, cached per arm); only the fit differs. The profile records the fit's time, the peak host resident set and the
device pool's high-water mark, the learned background variance per variant class (the sum of D_j and its mean), the
live local effects, and both starts' ELBOs.
"""

from __future__ import annotations

import os
import resource
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmarks.bench_sim.submissions.svpgs_full import Model, _array_module, cached_store, task_budget
from sv_pgs.dosage_store import VARIANT_CLASSES, DosageStore
from sv_pgs.fit_model import DRAW_COUNT
from sv_pgs.progress import log
from sv_pgs.scale_mixture_ep import device_scope
from sv_pgs.stage2_wiring import store_log_reliability
from sv_pgs.structured_full import fit_store


@dataclass(frozen=True)
class _Scoring:
    """What ``svpgs_full.Model.score`` reads: each member's store row, signed-code mean and scale, and coefficient."""

    store_rows: np.ndarray
    signed_means: np.ndarray
    signed_scales: np.ndarray
    coefficients: np.ndarray


def _device_high_water(array_module) -> int | None:
    if array_module is None:
        return None
    return int(array_module.get_default_memory_pool().total_bytes())


def fit(train) -> Model:
    work = Path(tempfile.mkdtemp(prefix="svpgs_structured_", dir=os.environ.get("TMPDIR")))
    try:
        store_path, order = cached_store(train, work)
        budget = task_budget()
        phenotype = np.asarray(train.phenotype, dtype=np.float64)
        covariates = np.column_stack([np.ones(train.n_samples), np.asarray(train.covariates, dtype=np.float64)])
        started = time.time()
        (work / "fit").mkdir()
        array_module = _array_module(budget)
        with DosageStore.open(store_path) as store, device_scope(array_module):
            fitted = fit_store(
                store=store, training_columns=np.arange(train.n_samples, dtype=np.int64), covariates=covariates, targets=phenotype,
                log_reliability=store_log_reliability(store), budget=budget, work_dir=work / "fit", seed=20260923, draw_count=DRAW_COUNT,
            )
            store_classes = np.asarray(store.variant_table.variant_class)[fitted.store_rows]
        result = fitted.fit
        variance = result.member_background_variance
        per_class = {
            VARIANT_CLASSES[int(code)].value: {
                "members": int(np.sum(store_classes == code)),
                "background_variance_sum": float(variance[store_classes == code].sum()),
                "background_variance_mean": float(variance[store_classes == code].mean()),
                "inclusion_sum": float(result.inclusion[store_classes == code].sum()),
                "mean_square": float(np.sum(result.member_mean[store_classes == code] ** 2)),
            }
            for code in np.unique(store_classes)
        }
        profile = {
            "fit_seconds": time.time() - started,
            "device": budget.device_kind,
            "rows": int(fitted.store_rows.shape[0]),
            "blocks": fitted.block_count,
            "noise_variance": float(result.noise),
            "phenotype_variance": float(phenotype.var()),
            "elbo": float(result.elbo),
            "start": result.start,
            "start_elbos": [float(value) for value in result.start_elbos],
            "converged": bool(result.converged),
            "iterations": len(result.history),
            "passes": int(result.passes),
            "live_effects": int(result.live_effects),
            "effective_count": float(result.moments.effective_count),
            "effective_count_error": float(result.moments.effective_count_error),
            "background_classes": per_class,
            "peak_host_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            "device_pool_bytes": _device_high_water(array_module),
        }
        log(f"svpgs_structured: {profile}")
        scoring = _Scoring(fitted.store_rows, fitted.signed_means, fitted.signed_scales, result.member_mean)
        return Model(scoring, order, np.asarray(train.variants["cls"]) >= 2, profile)
    finally:
        shutil.rmtree(work, ignore_errors=True)
