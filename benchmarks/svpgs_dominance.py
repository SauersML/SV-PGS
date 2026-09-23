"""bench-real: SV-PGS with the full genotype model of each biallelic SV, additive plus dominance, as extra design columns.

A three-state genotype d in {0, 1, 2} has three means, which an additive term (d) and one dominance term
(h = d (2 - d), the heterozygote indicator on hard calls) span exactly; the additive model alone forces the
heterozygote to the midpoint of the homozygotes. For an SV that need not hold: a homozygous deletion removes an
element entirely, and a dose-sensitive element need not respond to losing two copies as twice losing one. Each SV
gets its dominance column, with its own prior class whose density empirical Bayes learns, so the columns shrink to
nothing where the data carry no dominance. A column is added only where it is identified on the training people:
with no homozygous alternate carrier h equals d there, and with no heterozygote h is constant.

The dominance columns are functions of the window's genotypes alone (never of the phenotype), rebuilt identically for
the people scored. Everything else is ``svpgs_small_n.fit_expression_mean_field``'s full arm: annotations, classes,
offsets, covariates; a dominance column carries no annotation (missing) and its SV's measurement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from benchmarks import svpgs_small_n as small
from benchmarks.svpgs_method import _CLASS_CODES
from sv_pgs.small_n import fit_small_n

DOMINANCE_CLASS = len(_CLASS_CODES)
"""The dominance columns' prior class: the first code after the variant classes'."""


def dominance_columns(train: Any) -> np.ndarray:
    """The SV records whose dominance is identified on the training people: heterozygotes and homozygous alternates both
    present among the hard calls."""
    calls = np.rint(np.asarray(train.genotypes, dtype=np.float64))
    return np.flatnonzero(np.asarray(train.variants.is_sv, dtype=bool) & np.any(calls == 1.0, axis=0) & np.any(calls == 2.0, axis=0))


def dominance_values(genotypes: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """h = d (2 - d) for the given records (people x records)."""
    dosages = np.asarray(genotypes, dtype=np.float64)[:, rows]
    return dosages * (2.0 - dosages)


@dataclass(frozen=True)
class DominancePredictor:
    inner: small.SmallNPredictor
    rows: np.ndarray

    @property
    def fitted_schema(self) -> dict | None:
        schema = dict(self.inner.fitted_schema or {})
        schema["dominance_columns"] = int(self.rows.shape[0])
        return schema

    def predict(self, genotypes: np.ndarray, covariates: np.ndarray | None = None) -> np.ndarray:
        dosages = np.asarray(genotypes, dtype=np.float64)
        return self.inner.predict(np.hstack([dosages, dominance_values(dosages, self.rows)]), covariates=covariates)


def fit_expression_dominance(train: Any) -> DominancePredictor:
    """bench-real: the full mean-field arm with each identified SV's dominance column (module docstring)."""
    rows = dominance_columns(train)
    genotypes = np.asarray(train.genotypes, dtype=np.float64)
    codes, units = small._METHOD.bench_real_encoding(np.hstack([genotypes, dominance_values(genotypes, rows)]))
    classes = np.concatenate([small.classes_for_arm(train.variants, "full"), np.full(rows.shape[0], DOMINANCE_CLASS, dtype=np.uint8)])
    offsets = small._METHOD.bench_real_log_reliability(train.variants)
    if offsets is not None:
        offsets = np.concatenate([offsets, offsets[rows]])
    annotations = small._arm_annotations(train.variants, "full")
    if annotations is not None:
        annotations = {name: np.concatenate([values, np.full(rows.shape[0], np.nan)]) for name, values in annotations.items()}
    fit = fit_small_n(
        codes=codes, covariates=small._METHOD.bench_real_covariates(train), target=np.asarray(train.phenotype, dtype=np.float64),
        variant_class=classes, log_variance_offset=offsets, draw_count=small.DRAW_COUNT,
        working_bytes=small._METHOD.one_core_budget().working_bytes, seed=small.seed_from_name(str(train.gene_id)), inference="mean_field",
        array_module=small._device(), annotations=annotations, codes_per_unit=units,
    )
    return DominancePredictor(inner=small.SmallNPredictor(scoring=fit.scoring, profile=fit.profile, codes_per_unit=units), rows=rows)
