"""The AoU fit settings (variance never refreshed, no final diagnostics) must
not inflate the prior until the model loses to its own covariates.

With the posterior variance replaced by the prior variance, E[beta^2] =
m^2 + tau^2 always exceeds tau^2, so every local-scale and global-scale
update widened the prior; the genetic score overfit the training samples
and the held-out AUC fell below a covariates-only logistic fit.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import VariantRecord
from sv_pgs.inference import fit_variational_em
from sv_pgs.preprocessing import build_tie_map

from tests.test_prediction_accuracy import _covariate_only_holdout_probability


@pytest.mark.slow
def test_unrefreshed_binary_fit_beats_its_covariates() -> None:
    random_generator = np.random.default_rng(1)
    train_count, test_count, variant_count, causal_count = 1000, 3000, 6000, 200
    frequencies = random_generator.uniform(0.05, 0.5, size=variant_count)
    genotypes = random_generator.binomial(2, frequencies, size=(train_count + test_count, variant_count)).astype(np.float32)
    standardized = (
        (genotypes - genotypes[:train_count].mean(axis=0)) / genotypes[:train_count].std(axis=0)
    ).astype(np.float32)
    effects = np.zeros(variant_count)
    effects[random_generator.choice(variant_count, causal_count, replace=False)] = random_generator.standard_normal(causal_count)
    genetic_value = standardized @ effects
    genetic_value *= np.sqrt(0.5) / genetic_value.std()
    age = random_generator.standard_normal(train_count + test_count)
    logits = -1.0 + 0.8 * age + genetic_value
    targets = (random_generator.uniform(size=logits.shape[0]) < 1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    covariates = np.column_stack([np.ones(age.shape[0]), age]).astype(np.float32)
    is_structural = random_generator.uniform(size=variant_count) < 0.1
    records = [
        VariantRecord(
            variant_id=f"v{variant_index}",
            variant_class=VariantClass.DELETION_SHORT if structural else VariantClass.SNV,
            chromosome="1",
            position=variant_index + 1,
        )
        for variant_index, structural in enumerate(is_structural)
    ]
    config = ModelConfig(
        trait_type=TraitType.BINARY,
        max_outer_iterations=12,
        beta_variance_update_interval=13,
        final_posterior_diagnostics=False,
        minimum_minor_allele_frequency=0.0,
        stochastic_variational_updates=True,
        stochastic_min_variant_count=0,
        stochastic_variant_batch_size=2000,
        random_seed=0,
    )
    tie_map = build_tie_map(standardized[:train_count], records, config)
    result = fit_variational_em(
        genotypes=standardized[:train_count],
        covariates=covariates[:train_count],
        targets=targets[:train_count],
        records=records,
        config=config,
        tie_map=tie_map,
    )

    beta = np.zeros(variant_count)
    beta[tie_map.kept_indices] = result.beta_reduced
    held_out_predictor = standardized[train_count:] @ beta + covariates[train_count:] @ np.asarray(result.alpha, dtype=np.float64)
    model_auc = roc_auc_score(targets[train_count:], held_out_predictor)
    covariate_only_auc = roc_auc_score(
        targets[train_count:],
        _covariate_only_holdout_probability(covariates[:, 1:], targets, train_count),
    )
    assert model_auc > covariate_only_auc
    # The prior genetic variance must stay on the scale of the true one (0.5),
    # not grow with every epoch.
    assert float(np.sum(result.prior_scales)) < 2.0
