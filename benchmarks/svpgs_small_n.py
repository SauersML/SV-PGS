"""SV-PGS on bench-real by the small-n route (sv_pgs/small_n.py): the same model as ``svpgs_method.fit_expression``,
fitted from the training genotypes in memory with exact dense algebra (no dosage store, no LD blocks).

The arms are ``svpgs_method``'s: the full model, ``fit_expression_no_sv_terms`` and ``fit_expression_no_annotations``
(the variant classes each arm's prior sees). The prediction is the posterior-mean genetic score plus the intercept.

Both harnesses load this file without registering it as a module, so it has no ``from __future__ import
annotations`` and loads its sibling ``svpgs_method.py`` by path.
"""

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sv_pgs.dosage_store import CODES_PER_DOSAGE
from sv_pgs.fast_scoring import SIGNED_CODE_OFFSET, ScoringModel
from sv_pgs.fit_model import DRAW_COUNT
from sv_pgs.small_n import fit_small_n

_SPEC = importlib.util.spec_from_file_location("svpgs_method_for_small_n", Path(__file__).with_name("svpgs_method.py"))
_METHOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_METHOD)


@dataclass(frozen=True)
class SmallNPredictor:
    scoring: ScoringModel
    profile: dict

    def predict(self, genotypes: np.ndarray) -> np.ndarray:
        """The genetic score plus the intercept, in closed form from dosages: x_j = (127 d_j - 127 - mu_j) / sigma_j."""
        dosages = np.asarray(genotypes, dtype=np.float64)[:, self.scoring.store_rows]
        standardized = (CODES_PER_DOSAGE * dosages - SIGNED_CODE_OFFSET - self.scoring.signed_means) / self.scoring.signed_scales
        return standardized @ self.scoring.coefficients + self.scoring.alpha[0]


def _fit(train: Any, arm: str) -> SmallNPredictor:
    genotypes = np.asarray(train.genotypes)
    if not np.all(np.isin(genotypes, (0, 1, 2))):
        raise ValueError("bench-real training genotypes must be allele counts 0, 1 or 2.")
    codes = genotypes.astype(np.uint8) * np.uint8(CODES_PER_DOSAGE)
    samples = genotypes.shape[0]
    fit = fit_small_n(
        codes=codes,
        covariates=np.ones((samples, 1)),
        target=np.asarray(train.phenotype, dtype=np.float64),
        variant_class=_METHOD.bench_real_classes_for_arm(train.variants, arm),
        log_variance_offset=None,
        draw_count=DRAW_COUNT,
        working_bytes=_METHOD.one_core_budget().working_bytes,
        seed=int.from_bytes(str(train.gene_id).encode()[:8].ljust(8, b"\0"), "big"),
    )
    return SmallNPredictor(scoring=fit.scoring, profile=fit.profile)


def fit_expression(train: Any) -> SmallNPredictor:
    """bench-real: SV-PGS on one gene's cis window by the small-n route."""
    return _fit(train, "full")


def fit_expression_no_sv_terms(train: Any) -> SmallNPredictor:
    """bench-real ablation arm: the SV-specific prior terms withheld (``svpgs_method``)."""
    return _fit(train, "no_sv_terms")


def fit_expression_no_annotations(train: Any) -> SmallNPredictor:
    """bench-real ablation arm: the genotypes alone, every annotation withheld (``svpgs_method``)."""
    return _fit(train, "no_annotations")
