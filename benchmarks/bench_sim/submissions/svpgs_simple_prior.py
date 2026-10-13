"""Matched simpler-prior arm: retain class and frequency, omit store annotation features.

Use this and svpgs_full.py in separate power-grid output directories, with identical
cohort, scenario, training count and replicate. Each runner refits both feature arms.
"""
from benchmarks.bench_sim.submissions.svpgs_full import fit_with


def fit(train):
    return fit_with(train, "mean_field", annotated_prior=False)
