import sys
import numpy as np
from scipy.special import expit
sys.path.insert(0, ".")
from tests import test_ld_space_fit as T
from sv_pgs import ld_space_fit as L
from sv_pgs.anderson import AndersonState, anderson_step

def vec(model):
    return np.concatenate([[model.log_variance_level], model.annotation_coefficients, np.log(model.shape_b), [np.log(model.likelihood_precision)]])

def setvec(model, v, nfeat):
    model.log_variance_level = float(v[0]); model.annotation_coefficients = v[1:1 + nfeat].copy()
    model.shape_b = np.exp(np.clip(v[1 + nfeat:-1], np.log(0.1), np.log(10.0))); model.likelihood_precision = float(np.exp(v[-1]))

def run(scheme_name, ld, stats, hyper, boundaries, mode, max_passes=150, tol=1e-5):
    scheme = L._local_scheme(scheme_name)
    model = L._initial_model_state(scheme, ld, stats, hyper)
    state = AndersonState(memory_depth=5)
    previous = np.inf
    nfeat = hyper.annotation_design.shape[1]
    for pass_index in range(max_passes):
        model.score_dot_mean = model.mean_quadratic_form = model.effective_parameter_count = model.squared_mean_change = 0.0
        current = vec(model)
        for block_index in range(len(boundaries) - 1):
            L._update_model_block(scheme, model, hyper, ld.correlation_block(block_index), slice(int(boundaries[block_index]), int(boundaries[block_index + 1])))
        L._update_noise_variance(model); L._update_scale_model(scheme, model, hyper); L._update_shapes(scheme, model, hyper)
        mapped = vec(model)
        residual = float(np.max(np.abs(mapped - current)))
        mean_change = np.sqrt(model.squared_mean_change) / max(np.linalg.norm(model.posterior_mean), 1e-300)
        if max(residual, mean_change) < tol:
            return pass_index + 1, mapped, model
        if mode == "plain":
            continue
        if mode == "restart" and residual > previous:
            state.reset()
        previous = residual
        setvec(model, anderson_step(state, x_current=current, map_value=mapped), nfeat)
    return None, vec(model), model

# quantitative convergence problem
genotypes, residual, q, boundaries, stats, hyper, genetic = T._quantitative_problem(3, 800, [50, 50, 50, 50])
ld = T._ld_blocks(genotypes, boundaries)
for scheme_name in ("expectation_propagation", "coherent_vb", "plug_in"):
    for mode in ("plain", "anderson", "restart"):
        passes, v, model = run(scheme_name, ld, stats, hyper, boundaries, mode)
        c2 = np.corrcoef(genotypes @ model.posterior_mean, genetic)[0, 1] ** 2
        print(f"quant {scheme_name:24s} {mode:8s} passes={passes} level={v[0]:.5f} b={np.round(np.exp(v[3:5]), 4)} sigma2={np.exp(-v[-1]):.4f} corr2={c2:.4f}", flush=True)

# binary first fit
rng = np.random.default_rng(11)
train_count = 3000
covariates, genotypes, boundaries = T._orthogonal_block_design(rng, train_count, 20000, [40] * 5, 3)
p = genotypes.shape[1]
class_index = (rng.uniform(size=p) < 0.15).astype(np.int64)
effects = T._sparse_effects(rng, p, class_index, causal_count=15)
effects *= 0.6 / np.std(genotypes @ effects)
logit = covariates @ np.array([-1.5, 0.3, -0.2]) + genotypes @ effects
labels = (rng.uniform(size=logit.shape[0]) < expit(logit)).astype(np.float64)
Xtr, Wtr, ytr = genotypes[:train_count], covariates[:train_count], labels[:train_count]
ld = T._ld_blocks(Xtr, boundaries)
hyper = T._hypermodel(p, class_index, rng)
alpha = T._logistic_offset_fit(Wtr, ytr, np.zeros(train_count))
prob = expit(Wtr @ alpha)
stats = L.binary_statistics_at(score=Xtr.T @ (ytr - prob), fitted_probability=prob, expansion_point=np.zeros(p))
for scheme_name in ("expectation_propagation", "coherent_vb"):
    for mode in ("plain", "anderson", "restart"):
        passes, v, model = run(scheme_name, ld, stats, hyper, boundaries, mode)
        print(f"binary {scheme_name:24s} {mode:8s} passes={passes} level={v[0]:.5f} b={np.round(np.exp(v[3:5]), 4)} corr={np.corrcoef(model.posterior_mean, effects)[0,1]:.4f}", flush=True)
