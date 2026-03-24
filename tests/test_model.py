from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from sv_pgs import BayesianPGS, BenchmarkConfig, ModelConfig, TraitType, VariantClass, VariantRecord, run_benchmark_suite


def _synthetic_binary_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[VariantRecord]]:
    random_generator = np.random.default_rng(7)
    sample_count = 160
    covariate_matrix = random_generator.normal(size=(sample_count, 2)).astype(np.float32)
    signal_variant = random_generator.normal(size=sample_count).astype(np.float32)
    duplicate_variant = signal_variant.copy()
    correlated_structural_variant = (
        signal_variant + random_generator.normal(scale=0.05, size=sample_count)
    ).astype(np.float32)
    nuisance_variant = random_generator.normal(size=sample_count).astype(np.float32)
    structural_signal_variant = random_generator.normal(size=sample_count).astype(np.float32)
    genotype_matrix = np.column_stack(
        [
            signal_variant,
            duplicate_variant,
            correlated_structural_variant,
            nuisance_variant,
            structural_signal_variant,
        ]
    ).astype(np.float32)
    genotype_matrix[random_generator.choice(sample_count, size=12, replace=False), 4] = np.nan
    linear_predictor = (
        1.4 * signal_variant
        - 1.0 * structural_signal_variant
        + 0.6 * covariate_matrix[:, 0]
        - 0.4 * covariate_matrix[:, 1]
    )
    target_probabilities = 1.0 / (1.0 + np.exp(-linear_predictor))
    target_vector = random_generator.binomial(1, target_probabilities).astype(np.float32)
    variant_records = [
        VariantRecord("variant_0", VariantClass.SNV, "na", "1", 100),
        VariantRecord("variant_1", VariantClass.SNV, "na", "1", 101),
        VariantRecord("variant_2", VariantClass.DELETION_SHORT, "short", "1", 105, cluster_id="cluster_a"),
        VariantRecord("variant_3", VariantClass.SNV, "na", "1", 2_000_000),
        VariantRecord("variant_4", VariantClass.DUPLICATION_SHORT, "short", "1", 110, cluster_id="cluster_a"),
    ]
    return genotype_matrix, covariate_matrix, target_vector, variant_records


def test_binary_model_fit_roundtrip_and_rare_sv_filter(tmp_path):
    genotype_matrix, covariate_matrix, target_vector, variant_records = _synthetic_binary_dataset()
    genotype_matrix[:159, 4] = 0.0
    genotype_matrix[159, 4] = 1.0
    model = BayesianPGS(
        ModelConfig(
            trait_type=TraitType.BINARY,
            max_outer_iters=12,
            max_inner_pcg_iters=80,
            correlation_threshold=0.95,
            tile_size=8,
            minimum_structural_variant_carriers=2,
        )
    ).fit(genotype_matrix, covariate_matrix, target_vector, variant_records)

    assert model.state is not None
    assert 4 not in model.state.active_variant_indices.tolist()
    assert model.state.tie_map.kept_indices.tolist() == [0, 2, 3]
    predicted_probabilities = model.predict_proba(genotype_matrix, covariate_matrix)[:, 1]
    assert roc_auc_score(target_vector, predicted_probabilities) > 0.75

    artifact_directory = tmp_path / "artifact"
    model.export(artifact_directory)
    loaded_model = BayesianPGS.load(artifact_directory)
    np.testing.assert_allclose(
        loaded_model.decision_function(genotype_matrix, covariate_matrix),
        model.decision_function(genotype_matrix, covariate_matrix),
        atol=1e-5,
    )


def test_benchmark_suite_runs_from_shared_trainer():
    genotype_matrix, covariate_matrix, target_vector, variant_records = _synthetic_binary_dataset()
    train_stop = 120
    benchmark_metrics = run_benchmark_suite(
        train_genotypes=genotype_matrix[:train_stop],
        train_covariates=covariate_matrix[:train_stop],
        train_targets=target_vector[:train_stop],
        test_genotypes=genotype_matrix[train_stop:],
        test_covariates=covariate_matrix[train_stop:],
        test_targets=target_vector[train_stop:],
        records=variant_records,
        benchmark_config=BenchmarkConfig(
            shared_config=ModelConfig(
                trait_type=TraitType.BINARY,
                max_outer_iters=8,
                max_inner_pcg_iters=60,
                correlation_threshold=0.95,
                tile_size=8,
            )
        ),
    )

    assert set(benchmark_metrics) == {"current_snv_score", "snv_only_bayesr", "joint_snv_sv_bayesr"}
    assert benchmark_metrics["joint_snv_sv_bayesr"].auc is not None
    assert benchmark_metrics["joint_snv_sv_bayesr"].log_loss is not None
