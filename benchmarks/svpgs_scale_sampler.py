"""SV-PGS on bench-real by the exact reference (sv_pgs/scale_sampler.py): the full arm of ``svpgs_small_n`` (the same
classes, annotations, covariates, codes and seed), with the posterior computed exactly by the collapsed scale sampler
and the hyperparameters by empirical Bayes on the marginal likelihood itself, not by EP or the mean-field ELBO. The
prediction is the Rao-Blackwellized posterior-mean genetic score plus the intercept.

Every call of ``predict`` also prints one ``SCALE_SAMPLER`` JSON line: the precision dials of that prediction (each
person's Monte Carlo standard error from the two chains' batch means, relative to the predictions' spread, and the
per-person split R-hat) and the fit's own (empirical Bayes and posterior status, sweeps, seconds), so a run's log
carries every fit's measured precision beside its score.

Like its sibling, this file is loaded by path without being registered as a module, so it loads ``svpgs_small_n.py``
by path too.
"""

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.seeds import seed_from_name
from sv_pgs.fast_scoring import SIGNED_CODE_OFFSET
from sv_pgs.fit_model import DRAW_COUNT
from sv_pgs.scale_sampler import ScaleSamplerFit, fit_scale_sampler, split_rhat

_SPEC = importlib.util.spec_from_file_location("svpgs_small_n_for_scale_sampler", Path(__file__).with_name("svpgs_small_n.py"))
_SMALL_N = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SMALL_N)
_METHOD = _SMALL_N._METHOD

ASSUMPTIONS = _SMALL_N.ASSUMPTIONS


@dataclass(frozen=True)
class ScaleSamplerPredictor:
    fit: ScaleSamplerFit
    codes_per_unit: np.ndarray
    gene_id: str

    @property
    def fitted_schema(self) -> dict:
        from sv_pgs.small_n import fitted_schema

        return fitted_schema(self.fit.prior, "scale_sampler", None) | {"annotation_groups": len(self.fit.prior.annotation_groups)}

    @property
    def profile(self) -> dict:
        return self.fit.profile

    def _standardized(self, genotypes: np.ndarray) -> np.ndarray:
        statistics = self.fit.statistics
        rows = statistics.active_rows
        dosages = np.asarray(genotypes, dtype=np.float64)[:, rows]
        return (self.codes_per_unit[rows] * dosages - SIGNED_CODE_OFFSET - statistics.means) / statistics.scales

    def predict(self, genotypes: np.ndarray, covariates: np.ndarray | None = None) -> np.ndarray:
        """The genetic score plus the intercept (and the covariates' effects when given), and its precision dials."""
        standardized = self._standardized(genotypes)
        score = standardized @ self.fit.coefficients + self.fit.alpha[0]
        if covariates is not None and self.fit.alpha.shape[0] > 1:
            score = score + np.asarray(covariates, dtype=np.float64) @ self.fit.alpha[1:]
        batches = [values @ standardized.T for values in self.fit.coefficient_batches]
        genetic = standardized @ self.fit.coefficients
        record = {"gene": self.gene_id, "rows": int(standardized.shape[0]), "variants": int(standardized.shape[1])}
        if all(values.shape[0] >= 2 for values in batches):
            error = np.sqrt(np.sum([values.var(axis=0, ddof=1) / values.shape[0] for values in batches], axis=0)) / len(batches)
            per_person, aggregate = split_rhat([np.asarray(values) for values in batches])
            spread = float(np.std(genetic))
            record |= {
                "relative_mc_error_max": float(np.max(error) / spread) if spread > 0 else None,
                "relative_mc_error_median": float(np.median(error) / spread) if spread > 0 else None,
                "batch_rhat_max": float(np.nanmax(per_person)),
                "batch_rhat_aggregate": aggregate,
            }
        record |= {key: value for key, value in self.fit.profile.items() if not isinstance(value, (dict, list))}
        print("SCALE_SAMPLER " + json.dumps(record, default=float), flush=True)
        return score


def fit_expression_scale_sampler(train: Any) -> ScaleSamplerPredictor:
    """bench-real: the full arm of ``svpgs_small_n.fit_expression`` with the exact posterior."""
    genotypes = np.asarray(train.genotypes)
    codes, units = _METHOD.bench_real_encoding(genotypes)
    annotations = _SMALL_N._arm_annotations(train.variants, "full")
    fit = fit_scale_sampler(
        codes=codes,
        covariates=_METHOD.bench_real_covariates(train),
        target=np.asarray(train.phenotype, dtype=np.float64),
        variant_class=_SMALL_N.classes_for_arm(train.variants, "full"),
        log_variance_offset=_METHOD.bench_real_log_reliability(train.variants),
        draw_count=DRAW_COUNT,
        working_bytes=_METHOD.one_core_budget().working_bytes,
        seed=seed_from_name(str(train.gene_id)),
        annotations=annotations,
        codes_per_unit=units,
    )
    return ScaleSamplerPredictor(fit=fit, codes_per_unit=units, gene_id=str(train.gene_id))
