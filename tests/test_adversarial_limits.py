from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import VariantRecord
from sv_pgs.model import BayesianPGS
from tests.test_prediction_accuracy import _build_sparse_dataset


def test_rare_event_binary_signal_should_still_beat_chance():
    sample_count = 1_200
    variant_count = 80
    train_stop = 900
    genotype_matrix, covariate_matrix, _targets, variant_records = _build_sparse_dataset(
        sample_count=sample_count,
        variant_count=variant_count,
        causal_snp_indices=[7, 25],
        causal_sv_indices=[40],
        causal_snp_effects=[1.4, -1.0],
        causal_sv_effects=[1.8],
        ld_block_size=8,
        trait_type=TraitType.BINARY,
        noise_scale=0.0,
        missing_rate=0.01,
        random_seed=77,
    )

    linear_predictor = (
        1.3 * np.nan_to_num(genotype_matrix[:, 7], nan=0.0)
        - 1.0 * np.nan_to_num(genotype_matrix[:, 25], nan=0.0)
        + 1.8 * np.nan_to_num(genotype_matrix[:, 40], nan=0.0)
        - 4.5
    )
    rare_event_probability = 1.0 / (1.0 + np.exp(-linear_predictor))
    target_vector = np.random.default_rng(7).binomial(1, rare_event_probability).astype(np.float32)
    variant_records[40] = VariantRecord(
        variant_id=variant_records[40].variant_id,
        variant_class=variant_records[40].variant_class,
        chromosome=variant_records[40].chromosome,
        position=variant_records[40].position,
        length=variant_records[40].length,
        allele_frequency=variant_records[40].allele_frequency,
        quality=variant_records[40].quality,
        training_support=sample_count,
        is_repeat=variant_records[40].is_repeat,
        is_copy_number=variant_records[40].is_copy_number,
    )

    model = BayesianPGS(
        ModelConfig(
            trait_type=TraitType.BINARY,
            max_outer_iterations=10,
            ld_block_max_variants=10,
            ld_block_window_bp=200_000,
        )
    ).fit(
        genotype_matrix[:train_stop],
        covariate_matrix[:train_stop],
        target_vector[:train_stop],
        variant_records,
    )

    held_out_probability = model.predict_proba(
        genotype_matrix[train_stop:],
        covariate_matrix[train_stop:],
    )[:, 1]
    held_out_auc = roc_auc_score(target_vector[train_stop:], held_out_probability)
    assert held_out_auc > 0.65, f"Expected rare-event AUC > 0.65, got {held_out_auc:.4f}"


def test_mixed_exact_tie_set_should_not_collapse_onto_one_member():
    sample_count = 500
    random_generator = np.random.default_rng(9)
    base_column = random_generator.standard_normal(sample_count).astype(np.float32)
    genotype_matrix = np.column_stack(
        [
            base_column,
            base_column,
            -base_column,
            0.95 * base_column + 0.05 * random_generator.standard_normal(sample_count),
        ]
    ).astype(np.float32)
    covariate_matrix = random_generator.standard_normal((sample_count, 2)).astype(np.float32)
    target_vector = (
        1.5 * base_column
        + 0.2 * covariate_matrix[:, 0]
        + random_generator.standard_normal(sample_count) * 0.4
    ).astype(np.float32)
    variant_records = [
        VariantRecord("snp_0", VariantClass.SNV, "1", 100),
        VariantRecord("del_1", VariantClass.DELETION_SHORT, "1", 101, length=500.0, training_support=sample_count),
        VariantRecord("dup_2", VariantClass.DUPLICATION_SHORT, "1", 102, length=700.0, training_support=sample_count),
        VariantRecord("snp_3", VariantClass.SNV, "1", 50_000),
    ]

    model = BayesianPGS(
        ModelConfig(
            trait_type=TraitType.QUANTITATIVE,
            max_outer_iterations=12,
        )
    ).fit(
        genotype_matrix,
        covariate_matrix,
        target_vector,
        variant_records,
    )

    coefficient_table = model.coefficient_table()
    tied_betas = np.abs([float(coefficient_table[index]["beta"]) for index in range(3)])
    assert np.min(tied_betas) > 0.1 * np.max(tied_betas), (
        "Expected mixed exact-tie members to retain at least 10% of the largest tied weight, "
        f"got {tied_betas.tolist()}"
    )
