"""Sweeps to convergence vs local iterations per pass and M-step depth (Newton iterations per pass)."""
import sys
import time
import numpy as np
sys.path.insert(0, ".")
import bench_stage1 as bench
from sv_pgs import ld_space_fit as L
from sv_pgs.compute_budget import detect_compute_budget

bench.POOL_WIDTHS = (400, 450, 500, 520, 480, 500, 510, 490)
budget = detect_compute_budget()
L._HYPERPARAMETER_SUBSET_VARIANTS = 10**9
rng = np.random.default_rng(1)
ld, statistics, hypermodel = bench.simulate(int(sys.argv[1]), 1, rng)
reference = None
for local in (1, 2, 5):
    for newton in (1, 50):
        L._LOCAL_ITERATIONS_PER_PASS = local
        L._MAXIMUM_NEWTON_ITERATIONS = newton
        L._MAXIMUM_PASSES = 200
        start = time.perf_counter()
        (fit,) = L.fit_ld_space(ld, statistics, hypermodel, budget)
        seconds = time.perf_counter() - start
        if reference is None:
            reference = fit
        diff = np.max(np.abs(fit.posterior_mean - reference.posterior_mean)) / np.max(np.abs(reference.posterior_mean))
        print(f"SCHED local={local} newton={newton} passes={fit.passes} sweeps={fit.passes * local} converged={fit.converged} "
              f"s={seconds:.1f} level={fit.log_variance_level:.5f} b={np.round(fit.shape_b, 4)} sigma2={1/fit.likelihood_precision:.5f} "
              f"mean_rel_diff_vs_first={diff:.2e}", flush=True)
