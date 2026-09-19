import sys
import numpy as np
sys.path.insert(0, ".")
from tests import test_ld_space_fit as T
from sv_pgs import ld_space_fit as L

genotypes, residual, q, boundaries, statistics, hyper, genetic = T._quantitative_problem(7, 600, [40, 25, 60, 35])
ld = T._ld_blocks(genotypes, boundaries)
scheme = L._local_scheme("expectation_propagation")
cross = genotypes[:, :40].T @ genotypes[:, 40:65]
print("max cross-block gram", np.max(np.abs(cross)), "diag scale", np.max(np.abs(genotypes[:, :40].T @ genotypes[:, :40])))
# module path, pass by pass
module = L._initial_model_state(scheme, ld, statistics, hyper)
reference = L._initial_model_state(scheme, ld, statistics, hyper)
gram = genotypes.T @ genotypes
p = genotypes.shape[1]
everything = slice(0, p)
for pass_index in range(12):
    module.score_dot_mean = module.mean_quadratic_form = module.effective_parameter_count = module.squared_mean_change = 0.0
    for block_index in range(len(boundaries) - 1):
        L._update_model_block(scheme, module, hyper, ld.correlation_block(block_index), slice(int(boundaries[block_index]), int(boundaries[block_index + 1])))
    prior_variance = np.exp(hyper.log_prior_variance(reference.log_variance_level, reference.annotation_coefficients, everything))
    for _ in range(L._LOCAL_ITERATIONS_PER_PASS):
        precision, shift = scheme.solve_terms(reference.local, everything, prior_variance)
        cov = np.linalg.inv(reference.likelihood_precision * gram + np.diag(precision))
        mean = cov @ (reference.likelihood_precision * (genotypes.T @ residual) + shift)
        scheme.update_block(reference.local, everything, mean, np.diag(cov).copy(), prior_variance, hyper.variant_class_index, hyper.shape_a, reference.shape_b)
    ers = float(np.sum((residual - genotypes @ mean) ** 2) + np.sum(gram * cov))
    reference.likelihood_precision = (600 - q) / ers
    L._update_noise_variance(module)
    print(pass_index, "kappa", module.likelihood_precision, reference.likelihood_precision,
          "cavity diff", np.max(np.abs(module.local.cavity_mean - reference.local.cavity_mean)) / np.max(np.abs(reference.local.cavity_mean)),
          np.max(np.abs(module.local.cavity_variance / reference.local.cavity_variance - 1)),
          "mean diff", np.max(np.abs(module.posterior_mean - mean)) / np.max(np.abs(mean)))
    L._update_scale_model(scheme, module, hyper); L._update_scale_model(scheme, reference, hyper)
    L._update_shapes(scheme, module, hyper); L._update_shapes(scheme, reference, hyper)
    print("   level", module.log_variance_level, reference.log_variance_level, "coef", module.annotation_coefficients, reference.annotation_coefficients, "b", module.shape_b, reference.shape_b)
