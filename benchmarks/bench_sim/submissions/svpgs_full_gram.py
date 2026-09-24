"""bench-sim submission: ``svpgs_full`` by the summary route (``gram_space``: every step after Stage 0 on the summary
band, its far field in the likelihood) whatever the measured pass costs say, for the accuracy gate against the exact
route (``benchmarks.summary_gate``) and its wall time; the peaks of ``svpgs_full_memcontract`` are logged after the fit
and the score."""
from __future__ import annotations

from benchmarks.bench_sim.submissions import svpgs_full, svpgs_full_memcontract
from sv_pgs import stage2_wiring

# The measured choice (``gram_space.pass_costs``) replaced by the arm's route: a Gram pass is taken as free.
stage2_wiring.pass_costs = lambda band, source, array_module: (float("inf"), 0.0)


class Model(svpgs_full_memcontract.Model):
    pass


def fit(train) -> Model:
    fitted = svpgs_full.fit(train)
    svpgs_full_memcontract.report("fit")
    return Model(fitted.scoring, fitted.order, fitted.structural, fitted.profile)
