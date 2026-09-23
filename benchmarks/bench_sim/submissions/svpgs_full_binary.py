"""bench-sim submission: SV-PGS by the full-data route with the trait's own likelihood.

A binary trait is fitted on its Bernoulli likelihood (``sv_pgs.binary_likelihood``: the Polya-Gamma bound on the
streamed mean-field route), a quantitative one exactly as ``svpgs_full`` fits it. The prediction carries, besides the
harness's genetic ``total`` and ``structural`` scores, the full linear predictor with the covariates (``linear``) and,
for a binary trait, the posterior predictive P(y = 1) (``probability``: E sigmoid(eta + shift) under the draws' spread
and the covariates' covariance, as ``artifact.predict`` forms it) with its predictor variance (``variance``).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from benchmarks.bench_sim.submissions.svpgs_full import BLOCK_ROWS, _array_module, cached_store, task_budget
from sv_pgs.config import TraitType
from sv_pgs.dosage_store import DosageStore
from sv_pgs.fast_scoring import SIGNED_CODE_OFFSET, ScoringModel, posterior_predictive_probability
from sv_pgs.fit_model import DRAW_COUNT
from sv_pgs.progress import log
from sv_pgs.scale_mixture_ep import device_scope
from sv_pgs.stage2_wiring import fit_models


def linear_scores(scoring: ScoringModel, order: np.ndarray, structural_rows: np.ndarray, data, draws: bool) -> dict[str, np.ndarray]:
    """The genetic mean score, its structural part, the full linear predictor (intercept and covariates added) and,
    with ``draws``, the predictor variance the scorer's draws and the covariates' covariance give (``artifact.predict``)."""
    total = np.zeros(data.n_samples)
    structural = np.zeros(data.n_samples)
    draw_scores = np.zeros((data.n_samples, scoring.draw_count)) if draws else None
    store_rows = np.asarray(scoring.store_rows)
    for first in range(0, store_rows.shape[0], BLOCK_ROWS):
        rows = store_rows[first:first + BLOCK_ROWS]
        codes = np.asarray(data.codes(order[rows]), dtype=np.float64)
        standardized = (codes - SIGNED_CODE_OFFSET - scoring.signed_means[first:first + BLOCK_ROWS, None]) / scoring.signed_scales[first:first + BLOCK_ROWS, None]
        weights = scoring.coefficients[first:first + BLOCK_ROWS]
        total += weights @ standardized
        members = structural_rows[order[rows]]
        structural += weights[members] @ standardized[members]
        if draw_scores is not None:
            draw_scores += standardized.T @ scoring.posterior_draws[first:first + BLOCK_ROWS]
    fixed = np.column_stack([np.ones(data.n_samples), np.asarray(data.covariates, dtype=np.float64)])
    result = {"total": total, "structural": structural, "linear": total + fixed @ scoring.alpha}
    if draw_scores is not None:
        deviations = draw_scores - total[:, None] + fixed @ (scoring.covariate_draws - scoring.alpha[:, None])
        result["variance"] = np.mean(deviations * deviations, axis=1) + np.einsum("ij,jk,ik->i", fixed, scoring.covariate_covariance, fixed)
    return result


class Model:
    def __init__(self, scoring: ScoringModel, order: np.ndarray, structural: np.ndarray, binary: bool, profile: dict) -> None:
        self.scoring, self.order, self.structural, self.binary, self.profile = scoring, order, structural, binary, profile

    def score(self, test) -> dict:
        started = time.time()
        result = linear_scores(self.scoring, self.order, self.structural, test, self.binary)
        if self.binary:
            result["probability"] = posterior_predictive_probability(result["linear"], result["variance"], self.scoring.predictive_intercept_shift)
        log(f"svpgs_full_binary: scored {test.n_samples:,} samples in {time.time() - started:.0f} s")
        return result


def fit(train) -> Model:
    work = Path(tempfile.mkdtemp(prefix="svpgs_full_binary_", dir=os.environ.get("TMPDIR")))
    try:
        store_path, order = cached_store(train, work)
        budget = task_budget()
        binary = train.trait_type == "binary"
        phenotype = np.asarray(train.phenotype, dtype=np.float64)
        covariates = np.column_stack([np.ones(train.n_samples), np.asarray(train.covariates, dtype=np.float64)])
        started = time.time()
        (work / "fit").mkdir()
        with DosageStore.open(store_path) as store, device_scope(_array_module(budget)):
            fitted = fit_models(
                store=store,
                store_columns=np.arange(train.n_samples, dtype=np.int64),
                covariates=covariates,
                covariate_columns=np.ones((1, covariates.shape[1]), dtype=bool),
                targets=phenotype[:, None],
                training=np.ones((train.n_samples, 1), dtype=bool),
                trait_types=[TraitType.BINARY if binary else TraitType.QUANTITATIVE],
                log_variance_offset=None,
                budget=budget,
                work_dir=work / "fit",
                seed=20260922,
                draw_count=DRAW_COUNT,
            )
        (scoring,) = fitted.scoring
        certificate = fitted.certificate
        profile = {
            "fit_seconds": time.time() - started,
            "device": budget.device_kind,
            "trait_type": train.trait_type,
            "rows": int(np.asarray(scoring.store_rows).shape[0]),
            "noise_variance": float(fitted.noise_variance[0]),
            "intercept_shift": float(scoring.predictive_intercept_shift),
            "likelihood_gain": float(np.max(certificate.noise_gain)),
            "certified": None if certificate.outer_criterion_met is None else bool(np.all(certificate.outer_criterion_met)),
            "remaining_gain": float(np.max(certificate.remaining_gain)) if np.asarray(certificate.remaining_gain).size else None,
            "refusals": len(certificate.refusals),
        }
        log(f"svpgs_full_binary: {profile}")
        return Model(scoring, order, np.asarray(train.variants["cls"]) >= 2, binary, profile)
    finally:
        shutil.rmtree(work, ignore_errors=True)
