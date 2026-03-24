from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score, roc_auc_score

from sv_pgs.artifact import ModelArtifact, load_artifact, save_artifact
from sv_pgs.benchmark import run_benchmark_suite
from sv_pgs.config import BenchmarkConfig, ModelConfig, TraitType, VariantClass
from sv_pgs.data import TieGroup, TieMap, VariantRecord
from sv_pgs.model import BayesianPGS


VARIANT_CLASS_CYCLE = (
    VariantClass.SNV,
    VariantClass.DELETION_SHORT,
    VariantClass.DUPLICATION_SHORT,
    VariantClass.INSERTION_MEI,
    VariantClass.STR_VNTR_REPEAT,
    VariantClass.SNV,
)


def _variant_metadata(variant_index: int, block_index: int) -> VariantRecord:
    variant_class = VARIANT_CLASS_CYCLE[variant_index % len(VARIANT_CLASS_CYCLE)]
    chromosome = "chr" + str(1 + block_index // 4)
    position = 100_000 * block_index + 100 * (variant_index % 16)
    length = 1.0 if variant_class == VariantClass.SNV else 100.0 + 25.0 * (variant_index % 9)
    allele_frequency = 0.05 + 0.35 * ((variant_index % 11) / 10.0)
    quality = 0.7 + 0.03 * (variant_index % 8)
    return VariantRecord(
        variant_id="variant_" + str(variant_index),
        variant_class=variant_class,
        chromosome=chromosome,
        position=position,
        length=length,
        allele_frequency=allele_frequency,
        quality=min(quality, 0.98),
        training_support=64 if variant_class != VariantClass.SNV else None,
        is_repeat=variant_class == VariantClass.STR_VNTR_REPEAT,
        is_copy_number=variant_class in {VariantClass.DELETION_SHORT, VariantClass.DUPLICATION_SHORT},
    )


def _make_correlated_genotypes(
    sample_count: int,
    block_count: int,
    variants_per_block: int,
    random_generator: np.random.Generator,
) -> tuple[np.ndarray, list[VariantRecord]]:
    variant_count = block_count * variants_per_block
    genotype_matrix = np.zeros((sample_count, variant_count), dtype=np.float32)
    variant_records: list[VariantRecord] = []

    for block_index in range(block_count):
        block_factor = random_generator.normal(size=sample_count)
        secondary_factor = random_generator.normal(size=sample_count)
        for offset in range(variants_per_block):
            variant_index = block_index * variants_per_block + offset
            loading_primary = 0.75 + 0.15 * random_generator.random()
            loading_secondary = 0.10 * random_generator.normal()
            noise = random_generator.normal(scale=0.45, size=sample_count)
            genotype_matrix[:, variant_index] = (
                loading_primary * block_factor
                + loading_secondary * secondary_factor
                + noise
            ).astype(np.float32)
            variant_records.append(_variant_metadata(variant_index, block_index))

    genotype_matrix[:, 1] = genotype_matrix[:, 0]
    genotype_matrix[:, 2] = -genotype_matrix[:, 0]
    variant_records[1] = VariantRecord(
        variant_id=variant_records[1].variant_id,
        variant_class=VariantClass.DELETION_SHORT,
        chromosome=variant_records[1].chromosome,
        position=variant_records[1].position,
        length=600.0,
        allele_frequency=0.03,
        quality=0.82,
        training_support=sample_count,
        is_repeat=False,
        is_copy_number=True,
    )
    variant_records[2] = VariantRecord(
        variant_id=variant_records[2].variant_id,
        variant_class=VariantClass.DUPLICATION_SHORT,
        chromosome=variant_records[2].chromosome,
        position=variant_records[2].position,
        length=900.0,
        allele_frequency=0.03,
        quality=0.84,
        training_support=sample_count,
        is_repeat=False,
        is_copy_number=True,
    )

    genotype_matrix[:, 30] = 0.0
    genotype_matrix[0, 30] = 1.0
    variant_records[30] = VariantRecord(
        variant_id=variant_records[30].variant_id,
        variant_class=VariantClass.DUPLICATION_SHORT,
        chromosome=variant_records[30].chromosome,
        position=variant_records[30].position,
        length=2_000.0,
        allele_frequency=0.002,
        quality=0.9,
        training_support=1,
        is_repeat=False,
        is_copy_number=True,
    )

    missing_mask = random_generator.random(genotype_matrix.shape) < 0.015
    missing_mask[:, :3] = False
    missing_mask[:, 30] = False
    genotype_matrix[missing_mask] = np.nan
    return genotype_matrix, variant_records


def _binary_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[VariantRecord]]:
    random_generator = np.random.default_rng(123)
    sample_count = 640
    genotype_matrix, variant_records = _make_correlated_genotypes(
        sample_count=sample_count,
        block_count=12,
        variants_per_block=16,
        random_generator=random_generator,
    )
    genotype_matrix[:, 70] = (
        0.35 * np.nan_to_num(genotype_matrix[:, 70], nan=0.0)
        + random_generator.normal(scale=1.25, size=sample_count)
    ).astype(np.float32)
    genotype_matrix[:, 140] = (
        0.35 * np.nan_to_num(genotype_matrix[:, 140], nan=0.0)
        + random_generator.normal(scale=1.20, size=sample_count)
    ).astype(np.float32)
    covariate_matrix = random_generator.normal(size=(sample_count, 3)).astype(np.float32)
    filled_genotypes = np.nan_to_num(genotype_matrix, nan=0.0)
    linear_predictor = (
        1.05 * filled_genotypes[:, 0]
        - 0.90 * filled_genotypes[:, 35]
        + 2.80 * filled_genotypes[:, 70]
        - 2.40 * filled_genotypes[:, 140]
        + 0.70 * covariate_matrix[:, 0]
        - 0.45 * covariate_matrix[:, 1]
        + 0.25 * covariate_matrix[:, 2]
    )
    target_probability = 1.0 / (1.0 + np.exp(-linear_predictor))
    target_vector = random_generator.binomial(1, target_probability).astype(np.float32)
    return genotype_matrix, covariate_matrix, target_vector, variant_records


def _quantitative_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[VariantRecord]]:
    random_generator = np.random.default_rng(321)
    sample_count = 520
    genotype_matrix, variant_records = _make_correlated_genotypes(
        sample_count=sample_count,
        block_count=10,
        variants_per_block=16,
        random_generator=random_generator,
    )
    genotype_matrix[:, 49] = (
        0.30 * np.nan_to_num(genotype_matrix[:, 49], nan=0.0)
        + random_generator.normal(scale=1.15, size=sample_count)
    ).astype(np.float32)
    genotype_matrix[:, 98] = (
        0.30 * np.nan_to_num(genotype_matrix[:, 98], nan=0.0)
        + random_generator.normal(scale=1.10, size=sample_count)
    ).astype(np.float32)
    covariate_matrix = random_generator.normal(size=(sample_count, 2)).astype(np.float32)
    filled_genotypes = np.nan_to_num(genotype_matrix, nan=0.0)
    target_vector = (
        1.2 * filled_genotypes[:, 0]
        - 2.0 * filled_genotypes[:, 49]
        + 1.8 * filled_genotypes[:, 98]
        + 0.6 * covariate_matrix[:, 0]
        - 0.3 * covariate_matrix[:, 1]
        + random_generator.normal(scale=0.55, size=sample_count)
    ).astype(np.float32)
    return genotype_matrix, covariate_matrix, target_vector, variant_records


def test_large_scale_binary_end_to_end_roundtrip(tmp_path: Path):
    genotype_matrix, covariate_matrix, target_vector, variant_records = _binary_dataset()
    train_stop = 480
    model = BayesianPGS(
        ModelConfig(
            trait_type=TraitType.BINARY,
            max_outer_iterations=12,
            minimum_structural_variant_carriers=2,
            ld_block_max_variants=32,
            ld_block_window_bp=250_000,
        )
    ).fit(
        genotype_matrix[:train_stop],
        covariate_matrix[:train_stop],
        target_vector[:train_stop],
        variant_records,
        validation_data=(
            genotype_matrix[train_stop:],
            covariate_matrix[train_stop:],
            target_vector[train_stop:],
        ),
    )

    assert model.state is not None
    assert model.state.tie_map.original_to_reduced[0] == model.state.tie_map.original_to_reduced[1]
    assert model.state.tie_map.original_to_reduced[0] == model.state.tie_map.original_to_reduced[2]
    assert model.state.tie_map.original_to_reduced[30] == -1

    test_probability = model.predict_proba(genotype_matrix[train_stop:], covariate_matrix[train_stop:])[:, 1]
    assert np.all(np.isfinite(test_probability))
    assert roc_auc_score(target_vector[train_stop:], test_probability) > 0.52

    artifact_path = tmp_path / "large_binary_artifact"
    model.export(artifact_path)
    loaded_model = BayesianPGS.load(artifact_path)
    np.testing.assert_allclose(
        loaded_model.decision_function(genotype_matrix[train_stop:], covariate_matrix[train_stop:]),
        model.decision_function(genotype_matrix[train_stop:], covariate_matrix[train_stop:]),
        atol=1e-5,
    )


def test_large_scale_benchmark_and_quantitative_fit():
    binary_genotypes, binary_covariates, binary_targets, binary_records = _binary_dataset()
    train_stop = 480
    benchmark_metrics = run_benchmark_suite(
        train_genotypes=binary_genotypes[:train_stop],
        train_covariates=binary_covariates[:train_stop],
        train_targets=binary_targets[:train_stop],
        test_genotypes=binary_genotypes[train_stop:],
        test_covariates=binary_covariates[train_stop:],
        test_targets=binary_targets[train_stop:],
        records=binary_records,
        benchmark_config=BenchmarkConfig(
                shared_config=ModelConfig(
                    trait_type=TraitType.BINARY,
                    max_outer_iterations=10,
                    minimum_structural_variant_carriers=2,
                    ld_block_max_variants=32,
                ld_block_window_bp=250_000,
            )
        ),
    )

    assert benchmark_metrics["joint_snv_sv_continuous"].auc is not None
    assert benchmark_metrics["snv_only_continuous"].auc is not None
    assert benchmark_metrics["joint_snv_sv_continuous"].log_loss is not None
    assert benchmark_metrics["joint_snv_sv_continuous"].top_tail_enrichment > 0.9

    quantitative_genotypes, quantitative_covariates, quantitative_targets, quantitative_records = _quantitative_dataset()
    quantitative_benchmark = run_benchmark_suite(
        train_genotypes=quantitative_genotypes[:390],
        train_covariates=quantitative_covariates[:390],
        train_targets=quantitative_targets[:390],
        test_genotypes=quantitative_genotypes[390:],
        test_covariates=quantitative_covariates[390:],
        test_targets=quantitative_targets[390:],
        records=quantitative_records,
        benchmark_config=BenchmarkConfig(
            shared_config=ModelConfig(
                trait_type=TraitType.QUANTITATIVE,
                max_outer_iterations=7,
                minimum_structural_variant_carriers=2,
                ld_block_max_variants=32,
                ld_block_window_bp=250_000,
            )
        ),
    )
    assert quantitative_benchmark["joint_snv_sv_continuous"].r2 is not None
    assert quantitative_benchmark["snv_only_continuous"].r2 is not None
    assert quantitative_benchmark["joint_snv_sv_continuous"].top_tail_enrichment > 5.0
    assert quantitative_benchmark["joint_snv_sv_continuous"].r2 > 0.03

    quantitative_model = BayesianPGS(
        ModelConfig(
            trait_type=TraitType.QUANTITATIVE,
            max_outer_iterations=7,
            minimum_structural_variant_carriers=2,
            ld_block_max_variants=32,
            ld_block_window_bp=250_000,
        )
    ).fit(
        quantitative_genotypes[:390],
        quantitative_covariates[:390],
        quantitative_targets[:390],
        quantitative_records,
        validation_data=(
            quantitative_genotypes[390:],
            quantitative_covariates[390:],
            quantitative_targets[390:],
        ),
    )
    quantitative_prediction = quantitative_model.predict(
        quantitative_genotypes[390:],
        quantitative_covariates[390:],
    )
    assert np.all(np.isfinite(quantitative_prediction))
    assert r2_score(quantitative_targets[390:], quantitative_prediction) > 0.03


def test_artifact_roundtrip_preserves_prior_membership_metadata(tmp_path: Path):
    artifact = ModelArtifact(
        config=ModelConfig(),
        records=[
            VariantRecord(
                variant_id="latent_0",
                variant_class=VariantClass.OTHER_COMPLEX_SV,
                chromosome="chr1",
                position=100,
                length=750.0,
                allele_frequency=0.03,
                quality=0.9,
                prior_class_members=(VariantClass.DELETION_SHORT, VariantClass.SNV),
                prior_class_membership=(0.25, 0.75),
            )
        ],
        means=np.zeros(1, dtype=np.float32),
        scales=np.ones(1, dtype=np.float32),
        alpha=np.zeros(2, dtype=np.float32),
        beta_reduced=np.zeros(1, dtype=np.float32),
        beta_full=np.zeros(1, dtype=np.float32),
        beta_variance=np.ones(1, dtype=np.float32),
        tie_map=TieMap(
            kept_indices=np.array([0], dtype=np.int32),
            original_to_reduced=np.array([0], dtype=np.int32),
            reduced_to_group=[
                TieGroup(
                    representative_index=0,
                    member_indices=np.array([0], dtype=np.int32),
                    signs=np.array([1.0], dtype=np.float32),
                )
            ],
        ),
        sigma_e2=1.0,
        prior_scales=np.ones(1, dtype=np.float32),
        global_scale=1.0,
        class_tpb_shape_a={VariantClass.OTHER_COMPLEX_SV: 1.0},
        class_tpb_shape_b={VariantClass.OTHER_COMPLEX_SV: 0.5},
        scale_model_coefficients=np.zeros(3, dtype=np.float32),
        scale_model_feature_names=["intercept", "quality_linear", "copy_number_indicator"],
        objective_history=[-10.0, -9.0],
        validation_history=[0.7],
    )

    artifact_path = tmp_path / "artifact_roundtrip"
    save_artifact(artifact_path, artifact)
    restored_artifact = load_artifact(artifact_path)

    restored_record = restored_artifact.records[0]
    assert restored_record.prior_class_members == (VariantClass.DELETION_SHORT, VariantClass.SNV)
    np.testing.assert_allclose(restored_record.prior_class_membership, [0.25, 0.75])
