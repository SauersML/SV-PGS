"""SV-PGS on bench-real by the small-n route (sv_pgs/small_n.py): the same model as ``svpgs_method.fit_expression``,
fitted from the training genotypes in memory with exact dense algebra (no dosage store, no LD blocks).

The arms are ``svpgs_method``'s: the full model, ``fit_expression_no_sv_terms`` and ``fit_expression_no_annotations``
(the variant classes each arm's prior sees). The prediction is the posterior-mean genetic score plus the intercept.

Every arm here passes ``log_variance_offset=None``, so the fit takes every record as measured exactly: its prior
log-variance offset is log r^2 = 0. That is what bench-real's genotypes are, the panel's hard calls, and it is a
statement about this benchmark, not about the model. A number from these arms measures the prior and the inference,
never the measurement model, and none of them may be cited as evidence that the r^2-scaled prior helps or does not
(``svpgs_method``'s module docstring says the same of its own bench-real arms).

Both harnesses load this file without registering it as a module, so it has no ``from __future__ import
annotations`` and loads its sibling ``svpgs_method.py`` by path.
"""

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from types import ModuleType

import numpy as np

from sv_pgs.compute_budget import detect_compute_budget

from benchmarks.seeds import seed_from_name
from sv_pgs.config import VariantClass
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

    def predict(self, genotypes: np.ndarray, covariates: np.ndarray | None = None) -> np.ndarray:
        """The genetic score plus the intercept, in closed form from dosages: x_j = (127 d_j - 127 - mu_j) / sigma_j, and
        the fixed covariate effects when the harness passes the scored samples' covariates."""
        dosages = np.asarray(genotypes, dtype=np.float64)[:, self.scoring.store_rows]
        standardized = (CODES_PER_DOSAGE * dosages - SIGNED_CODE_OFFSET - self.scoring.signed_means) / self.scoring.signed_scales
        score = standardized @ self.scoring.coefficients + self.scoring.alpha[0]
        if covariates is not None and self.scoring.alpha.shape[0] > 1:
            score = score + np.asarray(covariates, dtype=np.float64) @ self.scoring.alpha[1:]
        return score


_CLASS_CODES = {variant_class: index for index, variant_class in enumerate(VariantClass)}
_LOSS = {_CLASS_CODES[VariantClass.DELETION]}
_GAIN = {_CLASS_CODES[variant_class] for variant_class in (VariantClass.INSERTION, VariantClass.DUPLICATION, VariantClass.INSERTION_MEI)}


def signed_length_change(variants: Any) -> np.ndarray:
    """Each record's signed allele-length change. bench-real stores 0 for a symbolic ALT (review F1), so a symbolic
    record (an SV type token and change 0) takes -SVLEN for a loss (DEL), +SVLEN for a gain (INS, DUP, MEI) and 0
    otherwise (INV, BND, CNV); sequence-resolved records keep their own change."""
    change = np.asarray(variants.allele_length_change, dtype=np.int64).copy()
    token_classes = _METHOD.bench_real_classes(variants)
    length_only = _METHOD.bench_real_classes_for_arm(variants, "no_sv_terms")
    symbolic = (token_classes != length_only) & (change == 0)
    length = np.abs(np.asarray(variants.sv_length, dtype=np.int64))
    change[symbolic & np.isin(token_classes, list(_LOSS))] = -length[symbolic & np.isin(token_classes, list(_LOSS))]
    change[symbolic & np.isin(token_classes, list(_GAIN))] = length[symbolic & np.isin(token_classes, list(_GAIN))]
    return change


def classes_for_arm(variants: Any, arm: str) -> np.ndarray:
    """``svpgs_method``'s classes, with the length rule applied to the signed change (review F1): the full model's
    SV-type classes; the small-variant rule on (span, span + signed change) for every record without the SV terms, so
    a symbolic DEL is a deletion and a symbolic INS/DUP an insertion; one class without any annotation."""
    if arm != "no_sv_terms":
        return _METHOD.bench_real_classes_for_arm(variants, arm)
    reference_length = np.asarray(variants.end, dtype=np.int64) - np.asarray(variants.position, dtype=np.int64) + 1
    return _METHOD._length_class(reference_length, reference_length + signed_length_change(variants))


def _fit(train: Any, arm: str, inference: str = "ep") -> SmallNPredictor:
    genotypes = np.asarray(train.genotypes)
    if not np.all(np.isin(genotypes, (0, 1, 2))):
        raise ValueError("bench-real training genotypes must be allele counts 0, 1 or 2.")
    codes = genotypes.astype(np.uint8) * np.uint8(CODES_PER_DOSAGE)
    fit = fit_small_n(
        codes=codes,
        # bench-real's fixed-effect covariates [1, C] (review-mathbugs C2), as svpgs_method's arms pass them.
        covariates=_METHOD.bench_real_covariates(train),
        target=np.asarray(train.phenotype, dtype=np.float64),
        variant_class=classes_for_arm(train.variants, arm),
        log_variance_offset=None,
        draw_count=DRAW_COUNT,
        working_bytes=_METHOD.one_core_budget().working_bytes,
        seed=seed_from_name(str(train.gene_id)),
        inference=inference,
        array_module=_device(),
    )
    return SmallNPredictor(scoring=fit.scoring, profile=fit.profile)


def _device() -> ModuleType | None:
    """CuPy where the machine has a CUDA device the budget can see, else None (numpy): the variant side's objective
    runs 28x faster there (7 ms against 196 ms on one core, ENSG00000254709.8 [real]), and its per-fit working set
    (one chunk of rows x nodes) is small enough for every forked worker to hold its own on one device."""
    if detect_compute_budget().device_kind != "cuda":
        return None
    import cupy  # noqa: PLC0415 - only where the budget found a device

    return cupy


def fit_expression(train: Any) -> SmallNPredictor:
    """bench-real: SV-PGS on one gene's cis window by the small-n route."""
    return _fit(train, "full")


def fit_expression_no_sv_terms(train: Any) -> SmallNPredictor:
    """bench-real ablation arm: the SV-specific prior terms withheld (``svpgs_method``)."""
    return _fit(train, "no_sv_terms")


def fit_expression_no_annotations(train: Any) -> SmallNPredictor:
    """bench-real ablation arm: the genotypes alone, every annotation withheld (``svpgs_method``)."""
    return _fit(train, "no_annotations")


def fit_expression_mean_field(train: Any) -> SmallNPredictor:
    """bench-real: the full model with the small-n route's mean-field inference (``sv_pgs.mean_field``), the VB side
    of the definition of done's inference measurement against ``fit_expression``'s EP."""
    return _fit(train, "full", inference="mean_field")
