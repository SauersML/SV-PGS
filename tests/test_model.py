from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

import sv_pgs.genotype as genotype_module
from sv_pgs import BayesianPGS, BenchmarkConfig, ModelConfig, TraitType, VariantClass, VariantRecord, run_benchmark_suite
from sv_pgs.data import TieGroup, TieMap
from sv_pgs.genotype import RawGenotypeBatch, RawGenotypeMatrix
from sv_pgs.inference import VariationalFitResult
from sv_pgs.model import (
    _raw_standardized_subset_matvec,
    _tie_group_export_weights,
    _training_records_from_stats,
)


def _synthetic_binary_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[VariantRecord]]:
    random_generator = np.random.default_rng(7)
    sample_count = 160
    covariate_matrix = random_generator.normal(size=(sample_count, 2)).astype(np.float32)
    signal_variant = random_generator.normal(size=sample_count).astype(np.float32)
    duplicate_variant = signal_variant.copy()
    sign_flipped_duplicate = -signal_variant.copy()
    nuisance_variant = random_generator.normal(size=sample_count).astype(np.float32)
    structural_signal_variant = random_generator.normal(size=sample_count).astype(np.float32)
    genotype_matrix = np.column_stack(
        [
            signal_variant,
            duplicate_variant,
            sign_flipped_duplicate,
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
        VariantRecord("variant_0", VariantClass.SNV, "1", 100, length=1.0, allele_frequency=0.12, quality=1.0),
        VariantRecord("variant_1", VariantClass.DELETION_SHORT, "1", 101, length=600.0, allele_frequency=0.02, quality=0.9, training_support=sample_count),
        VariantRecord("variant_2", VariantClass.DUPLICATION_SHORT, "1", 102, length=1_200.0, allele_frequency=0.02, quality=0.9, training_support=sample_count),
        VariantRecord("variant_3", VariantClass.SNV, "1", 2_000_000, length=1.0, allele_frequency=0.40, quality=1.0),
        VariantRecord(
            "variant_4",
            VariantClass.DUPLICATION_SHORT,
            "1",
            110,
            length=2_000.0,
            allele_frequency=0.02,
            quality=0.95,
            training_support=sample_count,
            is_copy_number=True,
        ),
    ]
    return genotype_matrix, covariate_matrix, target_vector, variant_records


def test_binary_model_fit_roundtrip_and_keeps_all_variants(tmp_path):
    genotype_matrix, covariate_matrix, target_vector, variant_records = _synthetic_binary_dataset()
    genotype_matrix[:159, 4] = 0.0
    genotype_matrix[159, 4] = 1.0
    variant_records[4] = VariantRecord(
        variant_id=variant_records[4].variant_id,
        variant_class=variant_records[4].variant_class,
        chromosome=variant_records[4].chromosome,
        position=variant_records[4].position,
        length=variant_records[4].length,
        allele_frequency=variant_records[4].allele_frequency,
        quality=variant_records[4].quality,
        training_support=None,
        is_repeat=variant_records[4].is_repeat,
        is_copy_number=variant_records[4].is_copy_number,
    )
    model = BayesianPGS(
        ModelConfig(
            trait_type=TraitType.BINARY,
            max_outer_iterations=10,
        )
    ).fit(genotype_matrix, covariate_matrix, target_vector, variant_records)

    assert model.state is not None
    assert model.state.active_variant_indices.tolist() == [0, 1, 2, 3, 4]
    assert model.state.tie_map.kept_indices.tolist() == [0, 3, 4]
    predicted_probabilities = model.predict_proba(genotype_matrix, covariate_matrix)[:, 1]
    assert roc_auc_score(target_vector, predicted_probabilities) > 0.55

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
                max_outer_iterations=8,
            )
        ),
    )

    assert set(benchmark_metrics) == {
        "snv_only_without_hyperparameter_updates",
        "snv_only_continuous",
        "joint_snv_sv_continuous",
    }
    assert benchmark_metrics["joint_snv_sv_continuous"].auc is not None
    assert benchmark_metrics["joint_snv_sv_continuous"].log_loss is not None


def test_tie_group_export_weights_are_proportional_to_member_variances():
    tie_map = TieMap(
        kept_indices=np.array([0], dtype=np.int32),
        original_to_reduced=np.array([0, 0], dtype=np.int32),
        reduced_to_group=[
            TieGroup(
                representative_index=0,
                member_indices=np.array([0, 1], dtype=np.int32),
                signs=np.array([1.0, 1.0], dtype=np.float32),
            )
        ],
    )
    fit_result = VariationalFitResult(
        alpha=np.zeros(1, dtype=np.float32),
        beta_reduced=np.zeros(1, dtype=np.float32),
        beta_variance=np.ones(1, dtype=np.float32),
        prior_scales=np.array([9.0, 1.0], dtype=np.float32),
        global_scale=1.0,
        class_tpb_shape_a={VariantClass.SNV: 1.0},
        class_tpb_shape_b={VariantClass.SNV: 0.5},
        scale_model_coefficients=np.zeros(1, dtype=np.float32),
        scale_model_feature_names=["copy_number_indicator"],
        sigma_error2=1.0,
        objective_history=[],
        validation_history=[],
        member_prior_variances=np.array([9.0, 1.0], dtype=np.float32),
    )

    weights = _tie_group_export_weights(
        tie_map=tie_map,
        fit_result=fit_result,
    )

    np.testing.assert_allclose(weights[0], np.array([0.9, 0.1], dtype=np.float32))
def test_training_records_from_stats_preserve_prior_continuous_features():
    records = [
        VariantRecord(
            "variant_0",
            VariantClass.DELETION_SHORT,
            "1",
            100,
            prior_continuous_features={"sv_length_score": 1.5},
        )
    ]

    training_records = _training_records_from_stats(
        records,
        variant_stats=type(
            "Stats",
            (),
            {
                "support_counts": np.array([7], dtype=np.int32),
            },
        )(),
    )

    assert training_records[0].training_support == 7
    assert training_records[0].prior_continuous_features == {"sv_length_score": 1.5}


def test_fit_rejects_variant_stats_shape_mismatch():
    genotype_matrix, covariate_matrix, target_vector, variant_records = _synthetic_binary_dataset()

    with np.testing.assert_raises_regex(ValueError, "variant_stats.means must match genotype column count."):
        BayesianPGS(
            ModelConfig(
                trait_type=TraitType.BINARY,
                max_outer_iterations=2,
            )
        ).fit(
            genotype_matrix,
            covariate_matrix,
            target_vector,
            variant_records,
            variant_stats=type(
                "Stats",
                (),
                {
                    "means": np.zeros(genotype_matrix.shape[1] - 1, dtype=np.float32),
                    "scales": np.ones(genotype_matrix.shape[1], dtype=np.float32),
                    "allele_frequencies": np.zeros(genotype_matrix.shape[1], dtype=np.float32),
                    "support_counts": np.ones(genotype_matrix.shape[1], dtype=np.int32),
                },
            )(),
        )


def test_validation_path_keeps_raw_genotypes_streaming():
    class NonMaterializingValidationMatrix(RawGenotypeMatrix):
        def __init__(self, matrix: np.ndarray) -> None:
            self.matrix = np.asarray(matrix, dtype=np.float32)
            self.materialize_calls = 0

        @property
        def shape(self) -> tuple[int, int]:
            return self.matrix.shape

        def iter_column_batches(self, variant_indices=None, batch_size: int = 1024):
            resolved_indices = (
                np.arange(self.matrix.shape[1], dtype=np.int32)
                if variant_indices is None
                else np.asarray(variant_indices, dtype=np.int32)
            )
            for start_index in range(0, resolved_indices.shape[0], batch_size):
                batch_indices = resolved_indices[start_index : start_index + batch_size]
                yield RawGenotypeBatch(
                    variant_indices=batch_indices,
                    values=np.asarray(self.matrix[:, batch_indices], dtype=np.float32),
                )

        def materialize(self, variant_indices=None) -> np.ndarray:
            self.materialize_calls += 1
            raise AssertionError("validation genotypes should stay streaming")

    train_genotypes = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    covariate_matrix = np.zeros((train_genotypes.shape[0], 1), dtype=np.float32)
    target_vector = np.array([0.0, 1.0, 2.0, 0.5, 1.5, 2.5], dtype=np.float32)
    validation_genotypes = NonMaterializingValidationMatrix(train_genotypes.copy())
    variant_records = [
        VariantRecord("variant_0", VariantClass.SNV, "1", 100),
        VariantRecord("variant_1", VariantClass.SNV, "1", 101),
        VariantRecord("variant_2", VariantClass.SNV, "1", 102),
    ]

    model = BayesianPGS(
        ModelConfig(
            trait_type=TraitType.QUANTITATIVE,
            max_outer_iterations=2,
        )
    ).fit(
        train_genotypes,
        covariate_matrix,
        target_vector,
        variant_records,
        validation_data=(validation_genotypes, covariate_matrix, target_vector),
    )

    assert model.state is not None
    assert validation_genotypes.materialize_calls == 0


def test_model_fit_keeps_streaming_when_materialization_is_skipped(monkeypatch):
    genotype_matrix, covariate_matrix, target_vector, variant_records = _synthetic_binary_dataset()

    monkeypatch.setattr(genotype_module.StandardizedGenotypeMatrix, "try_materialize_gpu", lambda self: False)
    monkeypatch.setattr(genotype_module.StandardizedGenotypeMatrix, "try_materialize", lambda self: False)
    monkeypatch.setattr(
        genotype_module.StandardizedGenotypeMatrix,
        "release_raw_storage",
        lambda self: (_ for _ in ()).throw(AssertionError("release_raw_storage should not be called for streaming fits")),
    )

    model = BayesianPGS(
        ModelConfig(
            trait_type=TraitType.BINARY,
            max_outer_iterations=2,
        )
    ).fit(genotype_matrix, covariate_matrix, target_vector, variant_records)

    assert model.state is not None

def test_raw_standardized_subset_matvec_reads_only_requested_columns():
    class SpyRawGenotypeMatrix(RawGenotypeMatrix):
        def __init__(self, matrix: np.ndarray) -> None:
            self.matrix = np.asarray(matrix, dtype=np.float32)
            self.requested_variant_indices: list[list[int]] = []

        @property
        def shape(self) -> tuple[int, int]:
            return self.matrix.shape

        def iter_column_batches(
            self,
            variant_indices=None,
            batch_size: int = 1024,
        ):
            resolved_indices = np.arange(self.matrix.shape[1], dtype=np.int32) if variant_indices is None else np.asarray(variant_indices, dtype=np.int32)
            self.requested_variant_indices.append(resolved_indices.tolist())
            safe_batch_size = max(int(batch_size), 1)
            for start_index in range(0, resolved_indices.shape[0], safe_batch_size):
                batch_indices = resolved_indices[start_index : start_index + safe_batch_size]
                yield RawGenotypeBatch(
                    variant_indices=batch_indices,
                    values=np.asarray(self.matrix[:, batch_indices], dtype=np.float32),
                )

        def materialize(self, variant_indices=None) -> np.ndarray:
            resolved_indices = np.arange(self.matrix.shape[1], dtype=np.int32) if variant_indices is None else np.asarray(variant_indices, dtype=np.int32)
            return np.asarray(self.matrix[:, resolved_indices], dtype=np.float32)

    raw_matrix = np.array(
        [
            [0.0, 1.0, 0.0, np.nan],
            [1.0, 0.0, 2.0, 1.0],
            [2.0, 1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    raw_genotypes = SpyRawGenotypeMatrix(raw_matrix)
    variant_indices = np.array([1, 3], dtype=np.int32)
    means = np.array([2.0 / 3.0, 0.5], dtype=np.float32)
    scales = np.array([0.47140452, 0.5], dtype=np.float32)
    coefficients = np.array([0.25, -0.75], dtype=np.float32)

    scores = _raw_standardized_subset_matvec(
        raw_genotypes=raw_genotypes,
        variant_indices=variant_indices,
        means=means,
        scales=scales,
        coefficients=coefficients,
        batch_size=1,
    )

    dense_subset = np.asarray(raw_matrix[:, variant_indices], dtype=np.float32)
    imputed = np.where(np.isnan(dense_subset), means[None, :], dense_subset)
    expected_scores = ((imputed - means[None, :]) / scales[None, :]) @ coefficients

    assert raw_genotypes.requested_variant_indices == [[1, 3]]
    np.testing.assert_allclose(scores, expected_scores.astype(np.float32))
