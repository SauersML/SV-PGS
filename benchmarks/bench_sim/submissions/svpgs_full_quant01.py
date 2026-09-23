"""bench-sim comparator: SV-PGS's quantitative route fitted on a binary trait's 0/1 outcome (``svpgs_full``'s fit), for the
binary route's paired comparison (``svpgs_full_binary``).

A Gaussian fit of a 0/1 outcome has no probability of its own (its predictor leaves [0, 1]), so its probability is its
full linear predictor through the logistic recalibration fitted on the training rows' in-sample predictors (Platt
scaling: logit P = a + b predictor, ``binary_likelihood.calibration``), training data only. The prediction carries the
harness's ``total`` and ``structural`` genetic scores, the full ``linear`` predictor and that ``probability``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.special import expit

from benchmarks.bench_sim.submissions.svpgs_full import _array_module, cached_store, task_budget
from benchmarks.bench_sim.submissions.svpgs_full_binary import linear_scores
from sv_pgs.binary_likelihood import calibration
from sv_pgs.config import TraitType
from sv_pgs.dosage_store import DosageStore
from sv_pgs.fast_scoring import ScoringModel
from sv_pgs.fit_model import DRAW_COUNT
from sv_pgs.progress import log
from sv_pgs.scale_mixture_ep import device_scope
from sv_pgs.stage2_wiring import fit_models


class Model:
    def __init__(self, scoring: ScoringModel, order: np.ndarray, structural: np.ndarray, recalibration, profile: dict) -> None:
        self.scoring, self.order, self.structural, self.recalibration, self.profile = scoring, order, structural, recalibration, profile

    def score(self, test) -> dict:
        result = linear_scores(self.scoring, self.order, self.structural, test, False)
        if self.recalibration is not None:
            result["probability"] = expit(self.recalibration.intercept + self.recalibration.slope * result["linear"])
        return result


def fit(train) -> Model:
    work = Path(tempfile.mkdtemp(prefix="svpgs_full_quant01_", dir=os.environ.get("TMPDIR")))
    try:
        store_path, order = cached_store(train, work)
        budget = task_budget()
        phenotype = np.asarray(train.phenotype, dtype=np.float64)
        covariates = np.column_stack([np.ones(train.n_samples), np.asarray(train.covariates, dtype=np.float64)])
        started = time.time()
        (work / "fit").mkdir()
        with DosageStore.open(store_path) as store, device_scope(_array_module(budget)):
            fitted = fit_models(
                store=store, store_columns=np.arange(train.n_samples, dtype=np.int64), covariates=covariates,
                covariate_columns=np.ones((1, covariates.shape[1]), dtype=bool), targets=phenotype[:, None],
                training=np.ones((train.n_samples, 1), dtype=bool), trait_types=[TraitType.QUANTITATIVE], log_variance_offset=None,
                budget=budget, work_dir=work / "fit", seed=20260922, draw_count=DRAW_COUNT,
            )
        (scoring,) = fitted.scoring
        structural = np.asarray(train.variants["cls"]) >= 2
        recalibration = None
        if train.trait_type == "binary":
            # The in-sample linear predictor as a logit through Platt's recalibration: calibration() fits
            # logit P = a + b logit(p), so p = expit(predictor) makes it a + b predictor.
            in_sample = linear_scores(scoring, order, structural, train, False)["linear"]
            recalibration = calibration(phenotype, expit(in_sample))
        profile = {"fit_seconds": time.time() - started, "device": budget.device_kind, "recalibration": None if recalibration is None else vars(recalibration)}
        log(f"svpgs_full_quant01: {profile}")
        return Model(scoring, order, structural, recalibration, profile)
    finally:
        shutil.rmtree(work, ignore_errors=True)
