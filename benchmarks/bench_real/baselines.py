"""Baseline methods for the bench-real harness. Each is a ``fit(train) -> predictor`` callable.

top_variant   the single variant with the largest training |correlation|, fitted by OLS (the "lead eQTL" predictor).
gblup_reml    GBLUP on the standardized cis genotypes, with h^2 chosen by REML (spectral form: Kang et al. 2008,
              Genetics 178:1709), and the GLS intercept.
mr_ash        a faithful port of mr.ash.alpha (Kim, Wang, Carbonetto & Stephens 2024, arXiv 2208.10910; R package
              commit 8e257fd, 2023-10-27) with all of its published defaults: intercept, no standardization,
              sa2 = (2^((0:19)/20) - 1)^2 / median(w) * n, sigma2 = var(y), uniform pi, beta = 0, sequential order,
              method_q = "sigma_dep_q", max.iter = 1000, convtol = 1e-8, epstol = 1e-12. These are the published
              method's own settings, cited in place; they are the comparator, not our model.
"""
import numpy as np
import numba
from scipy import linalg, optimize

DOUBLE_EPSILON = np.finfo(np.float64).eps


class LinearPredictor:
    def __init__(self, intercept, coefficients, center=None, scale=None):
        self.intercept = float(intercept)
        self.coefficients = coefficients
        self.center = center
        self.scale = scale

    def predict(self, genotypes):
        values = np.asarray(genotypes, dtype=np.float64)
        if self.center is not None:
            values = (values - self.center) / self.scale
        return self.intercept + values @ self.coefficients


class ZeroPredictor:
    def __init__(self, level):
        self.level = float(level)

    def predict(self, genotypes):
        return np.full(genotypes.shape[0], self.level)


def top_variant(train):
    genotypes = np.asarray(train.genotypes, dtype=np.float64)
    phenotype = train.phenotype
    if genotypes.shape[1] == 0:
        return ZeroPredictor(phenotype.mean())
    centered = genotypes - genotypes.mean(axis=0)
    centered_phenotype = phenotype - phenotype.mean()
    correlation = (centered.T @ centered_phenotype) / np.sqrt((centered ** 2).sum(axis=0) * (centered_phenotype ** 2).sum())
    best = int(np.argmax(np.abs(correlation)))
    slope = (centered[:, best] @ centered_phenotype) / (centered[:, best] @ centered[:, best])
    coefficients = np.zeros(genotypes.shape[1])
    coefficients[best] = slope
    return LinearPredictor(phenotype.mean() - genotypes[:, best].mean() * slope, coefficients)


class GblupPredictor:
    def __init__(self, standardized_train, dual_weights, intercept, center, scale, heritability):
        self.heritability = heritability
        self.standardized_train = standardized_train
        self.dual_weights = dual_weights
        self.intercept = intercept
        self.center = center
        self.scale = scale

    def predict(self, genotypes):
        standardized = (np.asarray(genotypes, dtype=np.float64) - self.center) / self.scale
        return self.intercept + standardized @ (self.standardized_train.T @ self.dual_weights)


def gblup_reml(train):
    genotypes = np.asarray(train.genotypes, dtype=np.float64)
    phenotype = train.phenotype
    sample_count, variant_count = genotypes.shape
    if variant_count == 0:
        return ZeroPredictor(phenotype.mean())
    center = genotypes.mean(axis=0)
    scale = genotypes.std(axis=0)
    standardized = (genotypes - center) / scale / np.sqrt(variant_count)
    kernel = standardized @ standardized.T
    # REML with the intercept as the only fixed effect: work in an orthonormal basis of the complement of 1.
    complement = linalg.null_space(np.ones((1, sample_count)))
    eigenvalues, eigenvectors = np.linalg.eigh(complement.T @ kernel @ complement)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    rotated_squared = (eigenvectors.T @ (complement.T @ phenotype)) ** 2
    degrees = sample_count - 1

    def negative_reml(heritability):
        # V = h K + (1-h) I up to the total variance, which is profiled out.
        variance = heritability * eigenvalues + (1.0 - heritability)
        return 0.5 * (degrees * np.log(np.sum(rotated_squared / variance)) + np.sum(np.log(variance)))

    interior = optimize.minimize_scalar(negative_reml, bounds=(0.0, 1.0), method="bounded", options={"xatol": np.sqrt(DOUBLE_EPSILON)})
    candidates = [(negative_reml(0.0), 0.0), (negative_reml(1.0 - np.sqrt(DOUBLE_EPSILON)), 1.0 - np.sqrt(DOUBLE_EPSILON)), (interior.fun, interior.x)]
    heritability = min(candidates)[1]
    if heritability == 0.0:
        return ZeroPredictor(phenotype.mean())
    variance = heritability * kernel + (1.0 - heritability) * np.eye(sample_count)
    factor = np.linalg.cholesky(variance)
    solve = lambda vector: np.linalg.solve(factor.T, np.linalg.solve(factor, vector))
    inverse_ones, inverse_phenotype = solve(np.ones(sample_count)), solve(phenotype)
    intercept = inverse_phenotype.sum() / inverse_ones.sum()
    dual_weights = heritability * (inverse_phenotype - intercept * inverse_ones)
    return GblupPredictor(standardized, dual_weights, intercept, center, scale * np.sqrt(variant_count), heritability)


MR_ASH_GRID_SIZE = 20
MR_ASH_MAX_ITERATIONS = 1000
MR_ASH_CONVERGENCE_TOLERANCE = 1e-8
MR_ASH_EPSILON_TOLERANCE = 1e-12


@numba.njit(cache=True)
def _mr_ash_sweeps(design, squared_norms, prior_variances, mixture, coefficients, residual, noise_variance,
                   max_iterations, convergence_tolerance, epsilon_tolerance):
    sample_count, variant_count = design.shape
    component_count = prior_variances.shape[0]
    posterior_scale = np.empty((component_count, variant_count))
    for component in range(component_count):
        for variant in range(variant_count):
            posterior_scale[component, variant] = 1.0 / (1.0 / prior_variances[component] + squared_norms[variant]) if prior_variances[component] > 0 else epsilon_tolerance
    previous_objective = np.inf
    iterations = 0
    for iteration in range(max_iterations):
        iterations = iteration + 1
        linear_accumulator = 0.0
        entropy_accumulator = 0.0
        old_mixture = mixture.copy()
        new_mixture = np.zeros(component_count)
        old_coefficients = coefficients.copy()
        for variant in range(variant_count):
            column = design[:, variant]
            projection = 0.0
            for sample in range(sample_count):
                projection += residual[sample] * column[sample]
            projection += coefficients[variant] * squared_norms[variant]
            for sample in range(sample_count):
                residual[sample] += column[sample] * coefficients[variant]
            means = np.empty(component_count)
            log_weights = np.empty(component_count)
            for component in range(component_count):
                means[component] = projection * posterior_scale[component, variant] if component > 0 else 0.0
                log_weights[component] = (np.log(old_mixture[component] + epsilon_tolerance) - np.log(1.0 + prior_variances[component] * squared_norms[variant]) / 2.0
                                          + means[component] * (projection / 2.0 / noise_variance))
            largest = log_weights.max()
            total = 0.0
            for component in range(component_count):
                log_weights[component] = np.exp(log_weights[component] - largest)
                total += log_weights[component]
            posterior_mean = 0.0
            for component in range(component_count):
                log_weights[component] /= total
                new_mixture[component] += log_weights[component] / variant_count
                posterior_mean += log_weights[component] * means[component]
            coefficients[variant] = posterior_mean
            for sample in range(sample_count):
                residual[sample] -= column[sample] * posterior_mean
            linear_accumulator += projection * posterior_mean
            for component in range(component_count):
                entropy_accumulator += log_weights[component] * np.log(log_weights[component] + epsilon_tolerance)
            for component in range(1, component_count):
                entropy_accumulator -= log_weights[component] * np.log(posterior_scale[component, variant]) / 2.0
        residual_norm = 0.0
        for sample in range(sample_count):
            residual_norm += residual[sample] * residual[sample]
        weighted_coefficients = 0.0
        for variant in range(variant_count):
            weighted_coefficients += coefficients[variant] * coefficients[variant] * squared_norms[variant]
        objective = residual_norm - weighted_coefficients + linear_accumulator
        noise_variance = objective / sample_count
        mixture[:] = new_mixture
        objective = objective / noise_variance / 2.0 + np.log(2.0 * np.pi * noise_variance) / 2.0 * sample_count + entropy_accumulator
        for component in range(component_count):
            objective -= mixture[component] * np.log(mixture[component] + epsilon_tolerance) * variant_count
        for component in range(1, component_count):
            objective += mixture[component] * np.log(prior_variances[component]) * variant_count / 2.0
        change = 0.0
        for variant in range(variant_count):
            change += (old_coefficients[variant] - coefficients[variant]) ** 2
        if np.sqrt(change) < convergence_tolerance * variant_count:
            break
        if iteration > 0 and objective > previous_objective:
            break
        previous_objective = objective
    return coefficients, noise_variance, iterations


def mr_ash(train):
    genotypes = np.asarray(train.genotypes, dtype=np.float64)
    phenotype = train.phenotype
    sample_count, variant_count = genotypes.shape
    if variant_count == 0:
        return ZeroPredictor(phenotype.mean())
    center = genotypes.mean(axis=0)
    design = np.ascontiguousarray(genotypes - center)
    response = phenotype - phenotype.mean()
    squared_norms = (design ** 2).sum(axis=0)
    prior_variances = (2.0 ** (np.arange(MR_ASH_GRID_SIZE) / MR_ASH_GRID_SIZE) - 1.0) ** 2 / np.median(squared_norms) * sample_count
    mixture = np.full(MR_ASH_GRID_SIZE, 1.0 / MR_ASH_GRID_SIZE)
    coefficients = np.zeros(variant_count)
    residual = response.copy()
    noise_variance = float(np.mean((response - response.mean()) ** 2))
    coefficients, _, _ = _mr_ash_sweeps(np.asfortranarray(design), squared_norms, prior_variances, mixture, coefficients, residual,
                                        noise_variance, MR_ASH_MAX_ITERATIONS, MR_ASH_CONVERGENCE_TOLERANCE, MR_ASH_EPSILON_TOLERANCE)
    return LinearPredictor(phenotype.mean() - center @ coefficients, coefficients)
