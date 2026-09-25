"""bench-sim prototype arm: ``svpgs_full`` with each non-TR multi-allelic site's REF state given its own column,
2 - sum_a D_a over the site's records (MODEL.md section 1: exchangeable allele states, so the prediction does not
depend on which allele the reference carries). The site's ALT records keep their columns.

A site is a position with two or more records none of which is a TR (class 2); the REF column takes the class of the
site's first non-SNV record (SNV where there is none) and is structural where any of its records is. It is stored and
scored as ``svpgs_full_derived`` states, with that prototype's two departures (its raw unit and its training
quantization)."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from benchmarks.bench_sim.submissions.svpgs_full import variant_classes
from benchmarks.bench_sim.submissions.svpgs_full_derived import Derived, fit_derived
from sv_pgs.dosage_store import VARIANT_CLASSES
from sv_pgs.config import VariantClass

TR = 2


def reference_columns(variants: dict) -> Derived:
    positions = np.asarray(variants["pos"], dtype=np.int64)
    cls = np.asarray(variants["cls"])
    _unique, site, counts = np.unique(positions, return_inverse=True, return_counts=True)
    has_tr = np.bincount(site, weights=(cls == TR).astype(np.float64)) > 0
    kept_sites = np.flatnonzero((counts > 1) & ~has_tr)
    number = np.full(counts.shape[0], -1, dtype=np.int64)
    number[kept_sites] = np.arange(kept_sites.shape[0])
    records = np.flatnonzero(number[site] >= 0)
    weights = sparse.csr_matrix((-np.ones(records.shape[0]), (number[site[records]], records)), shape=(kept_sites.shape[0], positions.shape[0]))
    classes = variant_classes(cls, np.asarray(variants["len_change"]))
    snv = VARIANT_CLASSES.index(VariantClass.SNV)
    site_class = []
    for row in range(kept_sites.shape[0]):
        members = weights.indices[weights.indptr[row]:weights.indptr[row + 1]]
        other = members[classes[members] != snv]
        site_class.append(VARIANT_CLASSES[int(classes[other[0]])] if other.shape[0] else VariantClass.SNV)
    structural = np.bincount(number[site[records]], weights=(cls[records] >= TR).astype(np.float64), minlength=kept_sites.shape[0]) > 0
    return Derived(
        weights=weights, offset=np.full(kept_sites.shape[0], 2.0), replaced=np.zeros(0, dtype=np.int64),
        variant_class=tuple(site_class), structural=structural, name="refstate",
    )


def fit(train):
    return fit_derived(train, reference_columns(train.variants))
