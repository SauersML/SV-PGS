"""Which quantity keeps the benchmark problem from converging: hyperparameters or posterior means?"""
import sys
import numpy as np
sys.path.insert(0, ".")
import bench_stage1 as bench
from sv_pgs import ld_space_fit as L
from sv_pgs.compute_budget import detect_compute_budget

bench.POOL_WIDTHS = tuple(int(width) for width in sys.argv[2].split(","))
use_anderson = sys.argv[3] == "anderson"
budget = detect_compute_budget()
ld, statistics, hypermodel = bench.simulate(int(sys.argv[1]), 1, np.random.default_rng(0))
boundaries = ld.block_boundaries
scheme = L._local_scheme("expectation_propagation")
model = L._initial_model_state(scheme, ld.ld_diagonal(), ld.ld_scores(), statistics[0], hypermodel)
everything = [slice(0, int(boundaries[-1]))]
with L._array_backend(budget, int(np.max(np.diff(boundaries)))) as backend:
    for pass_index in range(int(sys.argv[4])):
        before_mean = model.posterior_mean.copy()
        def work(block_index):
            variants = slice(int(boundaries[block_index]), int(boundaries[block_index + 1]))
            return L._update_model_block(scheme, backend, model, hypermodel, L._device_grid(backend, hypermodel, model),
                                         backend.to_device(ld.correlation_block(block_index)), variants, False)
        backend.map_blocks(work, list(range(boundaries.shape[0] - 1)))
        current = L._hyperparameter_vector(model)
        L._update_scale_model(scheme, backend, model, hypermodel, everything)
        L._update_shapes(scheme, backend, model, hypermodel, everything)
        mapped = L._hyperparameter_vector(model)
        change = np.abs(model.posterior_mean - before_mean)
        worst = int(np.argmax(change))
        sites = model.local
        print(f"pass {pass_index + 1}: level {mapped[0]:.4f} coef {np.round(mapped[1:3], 4)} logb {np.round(mapped[3:], 4)} "
              f"hyper change {np.max(np.abs(mapped - current)):.2e} mean change rel {np.linalg.norm(change) / np.linalg.norm(model.posterior_mean):.2e} "
              f"worst variant {worst} mean {before_mean[worst]:.4e}->{model.posterior_mean[worst]:.4e} site prec {sites.site_precision[worst]:.3e} "
              f"zero sites {int(np.sum(sites.site_precision == 0))}", flush=True)
        if use_anderson:
            L._set_hyperparameters(model, L._accelerated_hyperparameters(model.acceleration, current, mapped))
