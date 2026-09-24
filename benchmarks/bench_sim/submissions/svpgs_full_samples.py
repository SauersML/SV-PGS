"""bench-sim submission: ``svpgs_full`` with the quantitative fit in sample space (``dual_solve``, the store's codes)
whatever the measured pass costs say: the reference arm for ``svpgs_full_gram`` on the same commit, with its peak
memory logged as ``svpgs_full_gram`` logs it."""
from __future__ import annotations

import threading

from benchmarks.bench_sim.submissions import svpgs_full, svpgs_full_gram
from sv_pgs import stage2_wiring

# The measured choice (``gram_space.pass_costs``) replaced by the arm's route: a sample pass is taken as free. (Importing
# ``svpgs_full_gram`` set the Gram route's; this arm's own assignment comes after it.)
stage2_wiring.pass_costs = lambda band, source, array_module: (0.0, float("inf"))


class Model(svpgs_full_gram.Model):
    pass


def fit(train) -> Model:
    threading.Thread(target=svpgs_full_gram._sample, daemon=True).start()
    fitted = svpgs_full.fit(train)
    svpgs_full_gram.report("fit")
    return Model(fitted.scoring, fitted.order, fitted.structural, fitted.profile)
