"""bench-sim prototype arm: ``svpgs_full`` with each tandem-repeat locus's length-changing records replaced by its one
signed-length column Z = sum_a dlen_a D_a (MODEL.md section 1: the allele effects' rank-one prior beta_a = dlen_a theta).

A locus is the public ``repeat_locus`` (the merged simpleRepeat interval a record overlaps); its members are its
records with a length change, of every class (an SNV's change is zero, so it keeps its own column). The column is
stored and scored as ``svpgs_full_derived`` states, with that prototype's two departures (its raw unit and its
training quantization)."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from benchmarks.bench_sim.submissions.svpgs_full_derived import Derived, fit_derived
from sv_pgs.config import VariantClass


def locus_columns(variants: dict) -> Derived:
    locus = np.asarray(variants["repeat_locus"], dtype=np.int64)
    change = np.asarray(variants["len_change"], dtype=np.float64)
    members = np.flatnonzero((locus >= 0) & (change != 0.0))
    loci, of_member = np.unique(locus[members], return_inverse=True)
    weights = sparse.csr_matrix((change[members], (of_member, members)), shape=(loci.shape[0], locus.shape[0]))
    return Derived(
        weights=weights, offset=np.zeros(loci.shape[0]), replaced=members,
        variant_class=(VariantClass.STR_VNTR_REPEAT,) * loci.shape[0], structural=np.ones(loci.shape[0], dtype=bool), name="trlocus",
    )


def fit(train):
    return fit_derived(train, locus_columns(train.variants))
