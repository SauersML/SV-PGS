"""End-to-end prediction accuracy tests on simulated sparse genetic architectures.

These tests verify that the model actually learns signal — not just that it runs.
Each test simulates a realistic genetic architecture with:
  - Sparse causal variants among many null variants
  - Mixed SNP + SV causal effects
  - LD structure (correlated variant blocks)
  - Missing genotype data
  - Held-out prediction evaluated by R² or AUC

The thresholds are deliberately conservative: the model must beat chance
by a meaningful margin on held-out data, not just on training data.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score, roc_auc_score

from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import VariantRecord
from sv_pgs.model import BayesianPGS


def _build_sparse_dataset(
    sample_count: int,
    variant_count: int,
    causal_snp_indices: list[int],
    causal_sv_indices: list[int],
    causal_snp_effects: list[float],
    causal_sv_effects: list[float],
    ld_block_size: int,
    trait_type: TraitType,
    noise_scale: float,
    missing_rate: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[VariantRecord]]:
    random_gen = np.random.default_rng(random_seed)
    block_count = max(variant_count // ld_block_size, 1)
    genotype_matrix = np.zeros((sample_count, variant_count), dtype=np.float32)

    for block_idx in range(block_count):
        block_start = block_idx * ld_block_size
        block_end = min(block_start + ld_block_size, variant_count)
        actual_block_size = block_end - block_start
        latent_factor = random_gen.standard_normal(sample_count)
        for offset in range(actual_block_size):
            loading = 0.6 + 0.3 * random_gen.random()
            noise = random_gen.standard_normal(sample_count) * 0.5
            genotype_matrix[:, block_start + offset] = (loading * latent_factor + noise).astype(np.float32)

    covariate_matrix = random_gen.standard_normal((sample_count, 2)).astype(np.float32)
    covariate_effects = np.array([0.5, -0.3], dtype=np.float32)
    genetic_signal = np.zeros(sample_count, dtype=np.float32)

    for variant_idx, effect in zip(causal_snp_indices, causal_snp_effects, strict=True):
        genetic_signal += effect * genotype_matrix[:, variant_idx]
    for variant_idx, effect in zip(causal_sv_indices, causal_sv_effects, strict=True):
        genetic_signal += effect * genotype_matrix[:, variant_idx]

    linear_predictor = genetic_signal + covariate_matrix @ covariate_effects

    if trait_type == TraitType.QUANTITATIVE:
        target_vector = linear_predictor + random_gen.standard_normal(sample_count).astype(np.float32) * noise_scale
    else:
        probabilities = 1.0 / (1.0 + np.exp(-linear_predictor))
        target_vector = random_gen.binomial(1, probabilities).astype(np.float32)

    if missing_rate > 0.0:
        missing_mask = random_gen.random(genotype_matrix.shape) < missing_rate
        genotype_matrix[missing_mask] = np.nan

    variant_records: list[VariantRecord] = []
    sv_variant_set = set(causal_sv_indices)
    for variant_idx in range(variant_count):
        block_idx = variant_idx // ld_block_size
        chromosome = "chr" + str(1 + block_idx % 5)
        position = block_idx * 100_000 + (variant_idx % ld_block_size) * 500

        if variant_idx in sv_variant_set:
            variant_records.append(VariantRecord(
                variant_id="sv_" + str(variant_idx),
                variant_class=VariantClass.DELETION_SHORT,
                chromosome=chromosome,
                position=position,
                length=800.0 + 200.0 * random_gen.random(),
                allele_frequency=0.05 + 0.1 * random_gen.random(),
                quality=0.8 + 0.15 * random_gen.random(),
                training_support=sample_count,
                is_copy_number=True,
            ))
        else:
            variant_records.append(VariantRecord(
                variant_id="snp_" + str(variant_idx),
                variant_class=VariantClass.SNV,
                chromosome=chromosome,
                position=position,
                allele_frequency=0.1 + 0.3 * random_gen.random(),
                quality=0.95 + 0.05 * random_gen.random(),
            ))

    return genotype_matrix, covariate_matrix, target_vector, variant_records


class TestQuantitativePrediction:
    """Quantitative trait with sparse architecture: model must beat R²=0 on held-out data."""

    def test_sparse_quantitative_recovers_signal(self):
        sample_count = 500
        variant_count = 80
        train_stop = 380

        genotype_matrix, covariate_matrix, target_vector, variant_records = _build_sparse_dataset(
            sample_count=sample_count,
            variant_count=variant_count,
            causal_snp_indices=[3, 18, 42],
            causal_sv_indices=[10, 55],
            causal_snp_effects=[1.2, -0.8, 0.6],
            causal_sv_effects=[1.5, -1.0],
            ld_block_size=16,
            trait_type=TraitType.QUANTITATIVE,
            noise_scale=0.8,
            missing_rate=0.02,
            random_seed=42,
        )

        config = ModelConfig(
            trait_type=TraitType.QUANTITATIVE,
            max_outer_iterations=15,
        )
        model = BayesianPGS(config).fit(
            genotype_matrix[:train_stop],
            covariate_matrix[:train_stop],
            target_vector[:train_stop],
            variant_records,
        )

        test_predictions = model.predict(
            genotype_matrix[train_stop:],
            covariate_matrix[train_stop:],
        )
        assert np.all(np.isfinite(test_predictions))
        test_r2 = r2_score(target_vector[train_stop:], test_predictions)
        assert test_r2 > 0.05, f"Held-out R²={test_r2:.4f} too low for this architecture"

    def test_causal_snp_gets_nonzero_coefficient(self):
        genotype_matrix, covariate_matrix, target_vector, variant_records = _build_sparse_dataset(
            sample_count=400,
            variant_count=40,
            causal_snp_indices=[5],
            causal_sv_indices=[],
            causal_snp_effects=[2.5],
            causal_sv_effects=[],
            ld_block_size=10,
            trait_type=TraitType.QUANTITATIVE,
            noise_scale=0.5,
            missing_rate=0.0,
            random_seed=99,
        )
        config = ModelConfig(
            trait_type=TraitType.QUANTITATIVE,
            max_outer_iterations=15,
        )
        model = BayesianPGS(config).fit(
            genotype_matrix, covariate_matrix, target_vector, variant_records,
        )
        coefficient_table = model.coefficient_table()
        causal_beta = abs(float(coefficient_table[5]["beta"]))
        median_null_beta = float(np.median([
            abs(float(row["beta"])) for idx, row in enumerate(coefficient_table) if idx != 5
        ]))
        assert causal_beta > median_null_beta * 2.0, (
            f"Causal variant beta={causal_beta:.4f} not sufficiently larger than "
            f"null median={median_null_beta:.4f}"
        )


class TestBinaryPrediction:
    """Binary trait: model must achieve AUC > 0.55 on held-out data."""

    def test_sparse_binary_beats_chance(self):
        sample_count = 600
        variant_count = 60
        train_stop = 450

        genotype_matrix, covariate_matrix, target_vector, variant_records = _build_sparse_dataset(
            sample_count=sample_count,
            variant_count=variant_count,
            causal_snp_indices=[7, 25],
            causal_sv_indices=[40],
            causal_snp_effects=[1.0, -0.7],
            causal_sv_effects=[1.3],
            ld_block_size=15,
            trait_type=TraitType.BINARY,
            noise_scale=0.0,
            missing_rate=0.01,
            random_seed=77,
        )

        config = ModelConfig(
            trait_type=TraitType.BINARY,
            max_outer_iterations=12,
        )
        model = BayesianPGS(config).fit(
            genotype_matrix[:train_stop],
            covariate_matrix[:train_stop],
            target_vector[:train_stop],
            variant_records,
        )

        test_proba = model.predict_proba(
            genotype_matrix[train_stop:],
            covariate_matrix[train_stop:],
        )[:, 1]
        assert np.all(np.isfinite(test_proba))
        assert np.all(test_proba >= 0.0)
        assert np.all(test_proba <= 1.0)
        test_auc = roc_auc_score(target_vector[train_stop:], test_proba)
        assert test_auc > 0.55, f"Held-out AUC={test_auc:.4f} too close to chance"


class TestJointSNVSVBenefit:
    """Joint SNP+SV model should outperform SNP-only when SVs carry signal."""

    def test_joint_model_beats_snp_only(self):
        sample_count = 500
        variant_count = 60
        train_stop = 380

        genotype_matrix, covariate_matrix, target_vector, variant_records = _build_sparse_dataset(
            sample_count=sample_count,
            variant_count=variant_count,
            causal_snp_indices=[5, 20],
            causal_sv_indices=[35, 50],
            causal_snp_effects=[0.8, -0.6],
            causal_sv_effects=[1.5, -1.2],
            ld_block_size=15,
            trait_type=TraitType.QUANTITATIVE,
            noise_scale=0.7,
            missing_rate=0.01,
            random_seed=55,
        )

        snv_mask = np.array([rec.variant_class == VariantClass.SNV for rec in variant_records])
        snv_records = [rec for rec in variant_records if rec.variant_class == VariantClass.SNV]

        base_config = ModelConfig(
            trait_type=TraitType.QUANTITATIVE,
            max_outer_iterations=12,
        )

        snv_only_model = BayesianPGS(base_config).fit(
            genotype_matrix[:train_stop, :][:, snv_mask],
            covariate_matrix[:train_stop],
            target_vector[:train_stop],
            snv_records,
        )
        snv_only_pred = snv_only_model.predict(
            genotype_matrix[train_stop:, :][:, snv_mask],
            covariate_matrix[train_stop:],
        )
        snv_only_r2 = r2_score(target_vector[train_stop:], snv_only_pred)

        joint_model = BayesianPGS(base_config).fit(
            genotype_matrix[:train_stop],
            covariate_matrix[:train_stop],
            target_vector[:train_stop],
            variant_records,
        )
        joint_pred = joint_model.predict(
            genotype_matrix[train_stop:],
            covariate_matrix[train_stop:],
        )
        joint_r2 = r2_score(target_vector[train_stop:], joint_pred)

        assert joint_r2 > snv_only_r2, (
            f"Joint R²={joint_r2:.4f} should exceed SNV-only R²={snv_only_r2:.4f} "
            "when SVs carry causal signal"
        )


class TestExactCorrelationSetHandling:
    """Perfectly collinear SNP+SV must not crash or produce NaN."""

    def test_perfect_collinearity_produces_finite_predictions(self):
        random_gen = np.random.default_rng(33)
        sample_count = 200
        base_column = random_gen.standard_normal(sample_count).astype(np.float32)
        genotype_matrix = np.column_stack([
            base_column,
            base_column,
            -base_column,
            random_gen.standard_normal(sample_count),
            random_gen.standard_normal(sample_count),
        ]).astype(np.float32)
        covariate_matrix = random_gen.standard_normal((sample_count, 1)).astype(np.float32)
        target_vector = (1.5 * base_column + 0.3 * covariate_matrix[:, 0]
                         + random_gen.standard_normal(sample_count) * 0.5).astype(np.float32)

        variant_records = [
            VariantRecord("snp_0", VariantClass.SNV, "chr1", 100),
            VariantRecord("del_1", VariantClass.DELETION_SHORT, "chr1", 100, length=500.0, training_support=sample_count),
            VariantRecord("dup_2", VariantClass.DUPLICATION_SHORT, "chr1", 100, length=500.0, training_support=sample_count),
            VariantRecord("snp_3", VariantClass.SNV, "chr1", 50_000),
            VariantRecord("snp_4", VariantClass.SNV, "chr1", 100_000),
        ]

        config = ModelConfig(
            trait_type=TraitType.QUANTITATIVE,
            max_outer_iterations=10,
        )
        model = BayesianPGS(config).fit(
            genotype_matrix, covariate_matrix, target_vector, variant_records,
        )

        predictions = model.predict(genotype_matrix, covariate_matrix)
        assert np.all(np.isfinite(predictions))
        train_r2 = r2_score(target_vector, predictions)
        assert train_r2 > 0.1, f"Training R²={train_r2:.4f} too low with collinear causal variant"

        coefficient_table = model.coefficient_table()
        tied_betas = [abs(float(coefficient_table[idx]["beta"])) for idx in range(3)]
        assert all(beta_val > 0.0 for beta_val in tied_betas), (
            "All tied variants should get nonzero coefficients via prior-weighted splitting"
        )
