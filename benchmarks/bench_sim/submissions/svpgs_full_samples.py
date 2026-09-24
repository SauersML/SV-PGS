"""bench-sim submission: ``svpgs_full`` with the quantitative fit in sample space (``dual_solve``, the store's codes)
whatever the measured pass costs say: the reference arm for ``svpgs_full_gram`` on the same commit; the peaks of
``svpgs_full_memcontract`` are logged after the fit and the score."""
from __future__ import annotations

from benchmarks.bench_sim.submissions import svpgs_full, svpgs_full_memcontract
from sv_pgs import stage2_wiring

# The measured choice (``gram_space.pass_costs``) replaced by the arm's route: a sample pass is taken as free.
stage2_wiring.pass_costs = lambda band, source, array_module: (0.0, float("inf"))


class Model(svpgs_full_memcontract.Model):
    pass


def fit(train) -> Model:
    fitted = svpgs_full.fit(train)
    svpgs_full_memcontract.report("fit")
    return Model(fitted.scoring, fitted.order, fitted.structural, fitted.profile)
