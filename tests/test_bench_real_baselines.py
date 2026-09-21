"""Math checks of the baselines on small synthetic designs (correctness only, never accuracy evidence)."""
import dataclasses

import numpy as np

from benchmarks.bench_real import baselines

EPSILON = np.finfo(np.float64).eps


@dataclasses.dataclass
class FakeTrain:
    genotypes: np.ndarray
    phenotype: np.ndarray
    covariates: np.ndarray = None


def simulated(seed, samples=300, variants=200, causal=3):
    generator = np.random.default_rng(seed)
    frequency = generator.uniform(0.05, 0.5, variants)
    genotypes = generator.binomial(2, frequency, size=(samples, variants)).astype(np.float64)
    effects = np.zeros(variants)
    effects[generator.choice(variants, causal, replace=False)] = generator.choice([-1.0, 1.0], causal) * np.linspace(0.5, 0.3, causal)
    return genotypes, genotypes @ effects + generator.normal(0, 1, samples), effects


def direct_negative_reml(heritability, kernel, phenotype, design=None):
    """-log of the restricted Gaussian likelihood of y under V = h K + (1 - h) I, profiled over the total variance,
    written directly from its definition: |V|, |D' V^-1 D| and y' P y with P the generalized residual maker of the
    fixed-effect design D (Patterson and Thompson 1971). D defaults to the intercept alone."""
    count = len(phenotype)
    design = np.ones((count, 1)) if design is None else design
    variance = heritability * kernel + (1 - heritability) * np.eye(count)
    inverse = np.linalg.inv(variance)
    information = design.T @ inverse @ design
    projector = inverse - inverse @ design @ np.linalg.solve(information, design.T @ inverse)
    degrees = count - np.linalg.matrix_rank(design)
    return 0.5 * (np.linalg.slogdet(variance)[1] + np.linalg.slogdet(information)[1] + degrees * np.log(phenotype @ projector @ phenotype))


def correlated_covariates(seed, samples, count=4):
    """Covariates with a real correlation among themselves, so the intercept-only and the full-design restricted
    likelihoods are genuinely different functions of h."""
    generator = np.random.default_rng(seed)
    factor = generator.normal(size=(samples, 1))
    return factor + 0.6 * generator.normal(size=(samples, count))


def test_gblup_heritability_maximizes_the_direct_reml():
    genotypes, phenotype, _ = simulated(1)
    predictor = baselines.gblup_reml(FakeTrain(genotypes, phenotype))
    standardized = (genotypes - genotypes.mean(0)) / genotypes.std(0) / np.sqrt(genotypes.shape[1])
    kernel = standardized @ standardized.T
    grid = np.linspace(0.0, 1.0, 1001)[:-1]
    values = np.array([direct_negative_reml(value, kernel, phenotype) for value in grid])
    assert direct_negative_reml(predictor.heritability, kernel, phenotype) <= values.min() + np.sqrt(EPSILON) * np.abs(values).max()


def test_gblup_reml_uses_the_supplied_fixed_effects_not_the_intercept_alone():
    """With correlated covariates the restricted likelihood of the design [1, C] is not that of the intercept, and
    its maximizer is a different h^2. The fit must maximize the one its own covariates define."""
    genotypes, phenotype, _ = simulated(7, samples=120, variants=80)
    covariates = correlated_covariates(8, len(phenotype))
    phenotype = phenotype + covariates @ np.array([1.5, -0.9, 0.7, 1.1])
    design = np.column_stack([np.ones(len(phenotype)), covariates])
    predictor = baselines.gblup_reml(FakeTrain(genotypes, phenotype, covariates=covariates))
    standardized = (genotypes - genotypes.mean(0)) / genotypes.std(0) / np.sqrt(genotypes.shape[1])
    kernel = standardized @ standardized.T
    grid = np.linspace(0.0, 1.0, 2001)[:-1]
    full = np.array([direct_negative_reml(value, kernel, phenotype, design) for value in grid])
    intercept_only = np.array([direct_negative_reml(value, kernel, phenotype) for value in grid])
    assert direct_negative_reml(predictor.heritability, kernel, phenotype, design) <= full.min() + np.sqrt(EPSILON) * np.abs(full).max()
    # The two likelihoods disagree about h^2 by far more than the grid can resolve, so fitting the wrong one is visible.
    assert abs(grid[full.argmin()] - grid[intercept_only.argmin()]) > 10 * (grid[1] - grid[0])


def test_gblup_prediction_is_the_mixed_model_blup_of_its_own_design():
    """The predictor is D b + h K V^-1 (y - D b) with b the generalized least squares fixed effects: the textbook
    mixed-model solution under the fitted covariance, not a phenotype residualized by ordinary least squares."""
    genotypes, phenotype, _ = simulated(9, samples=90, variants=60)
    covariates = correlated_covariates(10, len(phenotype))
    phenotype = phenotype + covariates @ np.array([0.8, -1.2, 0.4, 0.6])
    design = np.column_stack([np.ones(len(phenotype)), covariates])
    predictor = baselines.gblup_reml(FakeTrain(genotypes, phenotype, covariates=covariates))
    standardized = (genotypes - genotypes.mean(0)) / genotypes.std(0) / np.sqrt(genotypes.shape[1])
    kernel = standardized @ standardized.T
    variance = predictor.heritability * kernel + (1 - predictor.heritability) * np.eye(len(phenotype))
    inverse = np.linalg.inv(variance)
    fixed = np.linalg.solve(design.T @ inverse @ design, design.T @ inverse @ phenotype)
    expected = design @ fixed + predictor.heritability * kernel @ inverse @ (phenotype - design @ fixed)
    predicted = predictor.predict(genotypes, covariates=covariates)
    condition = np.linalg.cond(variance)
    assert np.max(np.abs(predicted - expected)) <= condition * EPSILON * len(phenotype) * max(np.abs(expected).max(), 1.0)
    # Without the covariates the predictor is the same score minus that covariate combination.
    assert np.allclose(predictor.predict(genotypes), predicted - covariates @ fixed[1:])


def test_gblup_dual_prediction_equals_primal_ridge():
    genotypes, phenotype, _ = simulated(5)
    predictor = baselines.gblup_reml(FakeTrain(genotypes, phenotype))
    standardized = (genotypes - genotypes.mean(0)) / genotypes.std(0) / np.sqrt(genotypes.shape[1])
    heritability = predictor.heritability
    penalty = (1 - heritability) / heritability
    coefficients = np.linalg.solve(standardized.T @ standardized + penalty * np.eye(genotypes.shape[1]), standardized.T @ (phenotype - predictor.intercept))
    primal = predictor.intercept + standardized @ coefficients
    dual = predictor.predict(genotypes)
    condition = np.linalg.cond(standardized.T @ standardized + penalty * np.eye(genotypes.shape[1]))
    assert np.max(np.abs(primal - dual)) <= condition * EPSILON * np.abs(primal).max() * genotypes.shape[1]


def test_gblup_coefficients_decompose_its_score_column_by_column():
    """The fit carries the primal coefficients, so one column's contribution to the score is effect (x - train mean):
    the contract harness.sv_coefficients reads a linear predictor under, and what makes SV credit decomposable."""
    genotypes, phenotype, _ = simulated(13, samples=80, variants=50)
    predictor = baselines.gblup_reml(FakeTrain(genotypes, phenotype))
    assert predictor.heritability > 0
    effects = predictor.coefficients / predictor.scale
    muted = genotypes.copy()
    muted[:, 7] = genotypes[:, 7].mean()
    contribution = predictor.predict(genotypes) - predictor.predict(muted)
    expected = effects[7] * (genotypes[:, 7] - genotypes[:, 7].mean())
    assert np.allclose(contribution, expected, rtol=0, atol=64 * EPSILON * max(np.abs(predictor.predict(genotypes)).max(), 1.0))


def test_top_variant_picks_the_causal_variant():
    genotypes, phenotype, effects = simulated(4, causal=1)
    predictor = baselines.top_variant(FakeTrain(genotypes, phenotype))
    assert np.flatnonzero(predictor.coefficients)[0] == np.flatnonzero(effects)[0]


def test_top_variant_is_the_covariate_only_fit_when_every_column_is_in_the_covariate_span():
    """Columns that vary in training but lie in the span of [1, covariates] project to rounding error, not to zero."""
    genotypes, phenotype, _ = simulated(11, samples=40, variants=6)
    predictor = baselines.top_variant(FakeTrain(genotypes, phenotype, covariates=genotypes))
    assert (genotypes.var(axis=0) > 0).all()
    assert isinstance(predictor, baselines.ZeroPredictor)
    assert np.allclose(predictor.predict(genotypes), phenotype.mean())


def test_top_variant_picks_a_column_that_survives_the_projection():
    """Every column but the first is a covariate, so only the first can carry a slope; the rest are rounding error,
    whose correlation reaches +-1 as readily as a real association."""
    genotypes, phenotype, _ = simulated(12, samples=40, variants=6)
    predictor = baselines.top_variant(FakeTrain(genotypes, phenotype, covariates=genotypes[:, 1:]))
    assert np.flatnonzero(predictor.coefficients).tolist() == [0]
    assert np.isfinite(predictor.coefficients).all() and np.isfinite(predictor.intercept)
    assert np.isfinite(predictor.predict(genotypes)).all()
