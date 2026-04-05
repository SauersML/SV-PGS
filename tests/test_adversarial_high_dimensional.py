from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score

from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import VariantRecord
from sv_pgs.model import BayesianPGS


def _high_dimensional_sparse_quantitative_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[VariantRecord], list[int]]:
    random_generator = np.random.default_rng(321)
    sample_count = 90
    variant_count = 260
    latent_factor_matrix = random_generator.standard_normal((sample_count, 6)).astype(np.float32)
    factor_loadings = random_generator.standard_normal((6, variant_count)).astype(np.float32)
    genotype_matrix = (
        latent_factor_matrix @ factor_loadings
        + 0.2 * random_generator.standard_normal((sample_count, variant_count))
    ).astype(np.float32)
    covariate_matrix = random_generator.standard_normal((sample_count, 2)).astype(np.float32)
    causal_indices = [5, 40, 100, 150]
    true_coefficients = np.zeros(variant_count, dtype=np.float32)
    true_coefficients[causal_indices] = [1.8, -1.4, 1.2, -1.0]
    target_vector = (
        genotype_matrix @ true_coefficients
        + 0.4 * covariate_matrix[:, 0]
        + random_generator.standard_normal(sample_count) * 1.0
    ).astype(np.float32)
    variant_records = [
        VariantRecord(
            variant_id=f"variant_{variant_index}",
            variant_class=VariantClass.DELETION_SHORT if variant_index % 7 == 0 else VariantClass.SNV,
            chromosome="1",
            position=variant_index,
            training_support=sample_count if variant_index % 7 == 0 else None,
        )
        for variant_index in range(variant_count)
    ]
    return genotype_matrix, covariate_matrix, target_vector, variant_records, causal_indices


def _high_dimensional_pure_noise_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[VariantRecord]]:
    random_generator = np.random.default_rng(5)
    sample_count = 120
    variant_count = 140
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    covariate_matrix = random_generator.standard_normal((sample_count, 2)).astype(np.float32)
    target_vector = random_generator.standard_normal(sample_count).astype(np.float32)
    variant_records = [
        VariantRecord(f"noise_{variant_index}", VariantClass.SNV, "1", variant_index)
        for variant_index in range(variant_count)
    ]
    return genotype_matrix, covariate_matrix, target_vector, variant_records


def test_p_greater_than_n_sparse_quantitative_should_still_recover_positive_holdout_signal():
    genotype_matrix, covariate_matrix, target_vector, variant_records, _causal_indices = (
        _high_dimensional_sparse_quantitative_dataset()
    )
    train_stop = 60
    model = BayesianPGS(
        ModelConfig(
            trait_type=TraitType.QUANTITATIVE,
            max_outer_iterations=12,
        )
    ).fit(
        genotype_matrix[:train_stop],
        covariate_matrix[:train_stop],
        target_vector[:train_stop],
        variant_records,
    )

    held_out_prediction = model.predict(
        genotype_matrix[train_stop:],
        covariate_matrix[train_stop:],
    )
    held_out_r2 = r2_score(target_vector[train_stop:], held_out_prediction)
    assert held_out_r2 > 0.05, (
        "Expected the joint model to keep positive held-out signal in a p >> n sparse regime, "
        f"got R^2={held_out_r2:.4f}"
    )


def test_p_greater_than_n_sparse_quantitative_should_rank_multiple_causal_variants_near_the_top():
    genotype_matrix, covariate_matrix, target_vector, variant_records, causal_indices = (
        _high_dimensional_sparse_quantitative_dataset()
    )
    train_stop = 60
    model = BayesianPGS(
        ModelConfig(
            trait_type=TraitType.QUANTITATIVE,
            max_outer_iterations=12,
        )
    ).fit(
        genotype_matrix[:train_stop],
        covariate_matrix[:train_stop],
        target_vector[:train_stop],
        variant_records,
    )

    standardized_train_genotypes = (
        np.nan_to_num(genotype_matrix[:train_stop], nan=model.state.preprocessor.means)
        - model.state.preprocessor.means
    ) / model.state.preprocessor.scales
    coefficient_magnitudes = np.abs(np.asarray([row["beta"] for row in model.coefficient_table()], dtype=np.float64))
    top_variant_indices = set(np.argsort(-coefficient_magnitudes)[:12].tolist())

    recovered_latent_signals: set[int] = set()
    for top_variant_index in top_variant_indices:
        top_variant_values = standardized_train_genotypes[:, top_variant_index]
        for causal_index in causal_indices:
            causal_values = standardized_train_genotypes[:, causal_index]
            correlation = float(np.corrcoef(top_variant_values, causal_values)[0, 1])
            if abs(correlation) >= 0.75:
                recovered_latent_signals.add(causal_index)

    assert len(recovered_latent_signals) >= 2, (
        "Expected the top-loaded variants to recover at least two planted latent signals up to high within-train "
        f"correlation, recovered only {sorted(recovered_latent_signals)} from causal set {causal_indices}"
    )


def test_high_dimensional_pure_noise_should_not_overfit_below_a_small_negative_holdout_r2_floor():
    genotype_matrix, covariate_matrix, target_vector, variant_records = _high_dimensional_pure_noise_dataset()
    train_stop = 90
    model = BayesianPGS(
        ModelConfig(
            trait_type=TraitType.QUANTITATIVE,
            max_outer_iterations=8,
        )
    ).fit(
        genotype_matrix[:train_stop],
        covariate_matrix[:train_stop],
        target_vector[:train_stop],
        variant_records,
    )

    held_out_prediction = model.predict(
        genotype_matrix[train_stop:],
        covariate_matrix[train_stop:],
    )
    held_out_r2 = r2_score(target_vector[train_stop:], held_out_prediction)
    coefficient_magnitudes = np.abs(np.asarray([row["beta"] for row in model.coefficient_table()], dtype=np.float64))
    prediction_scale = float(np.std(held_out_prediction))
    target_scale = float(np.std(target_vector[train_stop:]))

    assert held_out_r2 > -0.10, (
        "Expected the model to avoid catastrophic high-dimensional overfit on pure noise, "
        f"got R^2={held_out_r2:.4f}"
    )
    assert float(np.max(coefficient_magnitudes)) < 0.05, (
        "Expected pure-noise training to stay strongly shrunk; "
        f"max |beta| was {float(np.max(coefficient_magnitudes)):.4f}"
    )
    assert prediction_scale < 0.2 * target_scale, (
        "Expected pure-noise predictions to remain small relative to phenotype scale, "
        f"prediction std={prediction_scale:.4f}, target std={target_scale:.4f}"
    )
