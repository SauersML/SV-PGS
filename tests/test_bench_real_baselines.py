"""Math checks of the baselines on small synthetic designs (correctness only, never accuracy evidence)."""
import dataclasses

import numpy as np

from benchmarks.bench_real import baselines

EPSILON = np.finfo(np.float64).eps


@dataclasses.dataclass
class FakeTrain:
    genotypes: np.ndarray
    phenotype: np.ndarray


def simulated(seed, samples=300, variants=200, causal=3):
    generator = np.random.default_rng(seed)
    frequency = generator.uniform(0.05, 0.5, variants)
    genotypes = generator.binomial(2, frequency, size=(samples, variants)).astype(np.float64)
    effects = np.zeros(variants)
    effects[generator.choice(variants, causal, replace=False)] = generator.choice([-1.0, 1.0], causal) * np.linspace(0.5, 0.3, causal)
    return genotypes, genotypes @ effects + generator.normal(0, 1, samples), effects


def direct_negative_reml(heritability, kernel, phenotype):
    count = len(phenotype)
    variance = heritability * kernel + (1 - heritability) * np.eye(count)
    inverse = np.linalg.inv(variance)
    ones = np.ones(count)
    information = ones @ inverse @ ones
    projector = inverse - np.outer(inverse @ ones, ones @ inverse) / information
    return 0.5 * (np.linalg.slogdet(variance)[1] + np.log(information) + (count - 1) * np.log(phenotype @ projector @ phenotype))


def test_gblup_heritability_maximizes_the_direct_reml():
    genotypes, phenotype, _ = simulated(1)
    predictor = baselines.gblup_reml(FakeTrain(genotypes, phenotype))
    standardized = (genotypes - genotypes.mean(0)) / genotypes.std(0) / np.sqrt(genotypes.shape[1])
    kernel = standardized @ standardized.T
    grid = np.linspace(0.0, 1.0, 1001)[:-1]
    values = np.array([direct_negative_reml(value, kernel, phenotype) for value in grid])
    assert direct_negative_reml(predictor.heritability, kernel, phenotype) <= values.min() + np.sqrt(EPSILON) * np.abs(values).max()


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


def test_top_variant_picks_the_causal_variant():
    genotypes, phenotype, effects = simulated(4, causal=1)
    predictor = baselines.top_variant(FakeTrain(genotypes, phenotype))
    assert np.flatnonzero(predictor.coefficients)[0] == np.flatnonzero(effects)[0]
