import sys
import numpy as np
from scipy.special import expit
sys.path.insert(0, ".")
from tests import test_ld_space_fit as T
from sv_pgs import ld_space_fit as L

rng = np.random.default_rng(11)
train_count, test_count = 3000, 20000
covariates, genotypes, boundaries = T._orthogonal_block_design(rng, train_count, test_count, [40] * 5, 3)
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
print("wbar", stats.mean_fisher_weight, "max|score|", np.max(np.abs(stats.score)), "R diag range", ld.ld_diagonal().min(), ld.ld_diagonal().max())
for scheme_name in ("expectation_propagation", "coherent_vb"):
    scheme = L._local_scheme(scheme_name)
    model = L._initial_model_state(scheme, ld, stats, hyper)
    print(scheme_name, "init level", model.log_variance_level)
    for pass_index in range(30):
        model.score_dot_mean = model.mean_quadratic_form = model.effective_parameter_count = model.squared_mean_change = 0.0
        for block_index in range(len(boundaries) - 1):
            L._update_model_block(scheme, model, hyper, ld.correlation_block(block_index), slice(int(boundaries[block_index]), int(boundaries[block_index + 1])))
        extra = ""
        if scheme_name == "expectation_propagation":
            s = model.local
            extra = f"site prec [{s.site_precision.min():.3g},{s.site_precision.max():.3g}] zero sites {np.sum(s.site_precision == 0)} cavP [{s.cavity_precision.min():.3g},{s.cavity_precision.max():.3g}]"
        L._update_scale_model(scheme, model, hyper)
        L._update_shapes(scheme, model, hyper)
        print(f"  pass {pass_index} level {model.log_variance_level:.4f} coef {np.round(model.annotation_coefficients, 4)} b {np.round(model.shape_b, 4)} max|mu| {np.max(np.abs(model.posterior_mean)):.4g} corr {np.corrcoef(model.posterior_mean, effects)[0,1]:.3f} {extra}")
