"""Fast architecture: one genotype pass, LD-space EB fit, exact certified polish.

Stage 0 (phenotype independent): one pass over the dosage store produces the
per-LD-block Grams, the per-variant training statistics and X'[C | Y].
Stage 1: per trait, the EB hyperparameters are fitted in LD space from those
sufficient statistics (cost independent of the sample count).
Stage 2: exact passes over the individual-level data drive every trait's
posterior mean to the exact joint fixed point, certified by the exact
full-data gradient, with the same M-step as ``fit_variational_em``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from sv_pgs.compute_budget import ComputeBudget, detect_compute_budget
from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.dosage_store import DosageStore
from sv_pgs.exact_polish import polish_exact
from sv_pgs.genotype_statistics import compute_genotype_statistics
from sv_pgs.ld_partition import partition_ld_blocks
from sv_pgs.ld_space_fit import build_prior_design_from_table, fit_ld_space
from sv_pgs.progress import log


@dataclass(slots=True)
class FastFitReport:
    stage_seconds: dict[str, float] = field(default_factory=dict)
    stage_reports: dict[str, object] = field(default_factory=dict)


def candidate_rows(store: DosageStore, training_sample_count: int, config: ModelConfig) -> NDArray[np.int64]:
    """Rows that can reach the training MAF floor, from the store's all-sample sums.

    A training minor-allele frequency f >= t on n of the store's N samples puts at least
    2 n t copies of that allele in the store, so its all-sample frequency is >= t n / N.
    """
    table = store.variant_table
    allele_frequency = table.sum_code.astype(np.float64) / (254.0 * store.n_samples)
    minor_frequency = np.minimum(allele_frequency, 1.0 - allele_frequency)
    floor = config.minimum_minor_allele_frequency * training_sample_count / store.n_samples
    return np.flatnonzero(minor_frequency >= floor).astype(np.int64)
