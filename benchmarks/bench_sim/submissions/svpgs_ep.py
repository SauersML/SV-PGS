"""bench-sim submission: SV-PGS by the full-data route with EP's fixed points (``full_data_fit._FullDataFixedPoints``)
for the one scale-mixture prior, everything else as ``svpgs_full``: the same cached training store, Stage 0, prior
and scoring, and the same profile."""

from __future__ import annotations

from benchmarks.bench_sim.submissions.svpgs_full import Model, fit_with


def fit(train) -> Model:
    return fit_with(train, "ep")
