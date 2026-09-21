"""Baseline methods for the bench-real harness. Each is a ``fit(train) -> predictor`` callable.

top_variant   the single variant with the largest training |correlation|, fitted by OLS (the "lead eQTL" predictor).
gblup_reml    GBLUP on the standardized cis genotypes, with h^2 chosen by REML (spectral form: Kang et al. 2008,
              Genetics 178:1709) and the GLS fixed effects, both under the harness's fixed-effect design [1, covariates].
mr.ash        is not ported here: the published R package (mr.ash.alpha) is run as itself for the comparison; a Python
              port gave identical SNV and SNV+SV predictions on bench-real chr22 (lead, 2026-09-21) and was deleted.
"""
import numpy as np
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


def covariate_projected(train, genotypes):
    """The training genotypes with [1, covariates] projected out (Frisch-Waugh-Lovell): fitting effects on these and
    the phenotype (already orthogonal to [1, covariates]) is least squares with the covariates as fixed effects. Without
    covariates it is centring. The harness applies the matching test-side projection to every score (predict_for_truth)."""
    covariates = getattr(train, "covariates", None)
    design = np.ones((genotypes.shape[0], 1)) if covariates is None else np.column_stack([np.ones(genotypes.shape[0]), covariates])
    coefficients, *_ = np.linalg.lstsq(design, genotypes, rcond=None)
    return genotypes - design @ coefficients


def top_variant(train):
    genotypes = np.asarray(train.genotypes, dtype=np.float64)
    phenotype = train.phenotype
    if genotypes.shape[1] == 0:
        return ZeroPredictor(phenotype.mean())
    if not np.all(genotypes.var(axis=0) > 0):
        raise ValueError("top_variant needs every column to vary in the training samples")
    centered = covariate_projected(train, genotypes)
    centered_phenotype = phenotype - phenotype.mean()
    squared_norms = (centered ** 2).sum(axis=0)
    # A column that varies in training can still lie in the span of [1, covariates]: it adds nothing to the covariates
    # and has no slope. Its projection is then rounding error, not zero, so "positive norm" is not the test; the test is
    # the numerical rank of the projection. Computing g - design @ coefficients leaves a residual of order
    # max(design.shape) * eps * ||g|| (Golub and Van Loan 2013, §5.3), which is the same criterion numpy's own lstsq
    # uses for rank (its rcond default). Below it a column is in the span: its correlation is a ratio of two rounding
    # errors, which reaches +-1 as readily as a real association, and its slope is that error divided by its square.
    covariates = getattr(train, "covariates", None)
    design_columns = 1 if covariates is None else 1 + np.asarray(covariates).shape[1]
    span_level = max(genotypes.shape[0], design_columns) * DOUBLE_EPSILON * np.linalg.norm(genotypes, axis=0)
    outside_span = squared_norms > span_level ** 2
    usable = np.flatnonzero(outside_span)
    if usable.size == 0:
        # No variant adds anything to [1, covariates], so the covariates are the whole fit.
        return ZeroPredictor(phenotype.mean())
    correlation = np.divide(centered.T @ centered_phenotype, np.sqrt(squared_norms * (centered_phenotype ** 2).sum()),
                            out=np.zeros(genotypes.shape[1]), where=outside_span)
    best = int(usable[np.argmax(np.abs(correlation[usable]))])
    slope = (centered[:, best] @ centered_phenotype) / squared_norms[best]
    coefficients = np.zeros(genotypes.shape[1])
    coefficients[best] = slope
    return LinearPredictor(phenotype.mean() - genotypes[:, best].mean() * slope, coefficients)


class GblupPredictor(LinearPredictor):
    """The GLS fit: intercept + (x - center) / scale @ coefficients, plus the covariate effects when the caller
    passes the covariates (the harness does; predict_for_truth then removes the whole covariate combination from the
    score, so they change no score of this benchmark).

    The dual weights are turned into those primal coefficients once, in the fit: the dual form recomputed
    standardized_train.T @ dual_weights on every predict, and the harness predicts four times per fit (train and test,
    full and SV-muted), each time multiplying the whole training matrix again and holding it alive until then."""

    def __init__(self, coefficients, fixed_effects, center, scale, heritability):
        super().__init__(fixed_effects[0], coefficients, center, scale)
        self.heritability = heritability
        self.fixed_effects = fixed_effects

    def predict(self, genotypes, covariates=None):
        score = super().predict(genotypes)
        if covariates is None or len(self.fixed_effects) == 1:
            return score
        return score + np.asarray(covariates, dtype=np.float64) @ self.fixed_effects[1:]


def fixed_effect_design(train, sample_count):
    """[1, covariates]: the fixed-effect design every method of the benchmark fits under (harness.predict_for_truth)."""
    covariates = getattr(train, "covariates", None)
    if covariates is None:
        return np.ones((sample_count, 1))
    return np.column_stack([np.ones(sample_count), covariates])


def gblup_reml(train):
    genotypes = np.asarray(train.genotypes, dtype=np.float64)
    phenotype = train.phenotype
    sample_count, variant_count = genotypes.shape
    if variant_count == 0:
        return ZeroPredictor(phenotype.mean())
    center = genotypes.mean(axis=0)
    scale = genotypes.std(axis=0)
    if not np.all(scale > 0):
        raise ValueError("gblup_reml needs every column to vary in the training samples")
    standardized = (genotypes - center) / scale / np.sqrt(variant_count)
    kernel = standardized @ standardized.T
    # REML under the harness's own fixed-effect design [1, covariates], not the intercept alone: the restricted
    # likelihood is the likelihood of the data in the complement of the design's column span, and the covariance
    # K enters it through that complement, so removing a different design changes h^2 itself, not just the fitted
    # mean. null_space's rank-revealing SVD handles a rank-deficient design; the complement's width is n - rank.
    design = fixed_effect_design(train, sample_count)
    complement = linalg.null_space(design.T)
    eigenvalues, eigenvectors = np.linalg.eigh(complement.T @ kernel @ complement)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    rotated_squared = (eigenvectors.T @ (complement.T @ phenotype)) ** 2
    degrees = complement.shape[1]

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
    # One Cholesky and one triangular solve of the design and the phenotype together: cho_solve tells LAPACK the factor
    # is triangular, which np.linalg.solve (an LU of a triangular matrix, twice) does not.
    factor = linalg.cho_factor(variance, lower=True)
    inverse = linalg.cho_solve(factor, np.column_stack([design, phenotype]))
    inverse_design, inverse_phenotype = inverse[:, :-1], inverse[:, -1]
    # The generalized least squares fixed effects of the whole design, under the fitted covariance; lstsq because a
    # rank-deficient design has no unique solution, only a unique fit.
    fixed_effects, *_ = np.linalg.lstsq(design.T @ inverse_design, design.T @ inverse_phenotype, rcond=None)
    dual_weights = heritability * (inverse_phenotype - inverse_design @ fixed_effects)
    return GblupPredictor(standardized.T @ dual_weights, fixed_effects, center, scale * np.sqrt(variant_count), heritability)


