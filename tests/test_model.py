from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

import sv_pgs.genotype as genotype_module
import sv_pgs.mixture_inference as mixture_module
import sv_pgs.model as model_module
import sv_pgs.runtime_policy as runtime_policy_module
from sv_pgs import (
    BayesianPGS,
    BenchmarkConfig,
    ModelConfig,
    TraitType,
    VariantClass,
    VariantRecord,
    run_benchmark_suite,
)
from sv_pgs.data import TieGroup, TieMap, VariantStatistics
from sv_pgs.genotype import RawGenotypeBatch, RawGenotypeMatrix, as_raw_genotype_matrix
from sv_pgs.inference import VariationalFitCheckpoint, VariationalFitResult
from sv_pgs.model import (
    _FitStageCachePaths,
    _active_indices_cover_original,
    _fit_stage_cache_paths,
    _fit_checkpoint_config_hash,
    _persistent_raw_signature,
    _raw_standardized_subset_matvec,
    _runtime_tuned_config_for_fit,
    _normalize_variant_records,
    _tie_group_export_weights,
    _tie_map_keeps_all_active_variants,
    _training_records_from_stats,
)
from sv_pgs.preprocessing import compute_variant_statistics
from tests.conftest import make_fake_cupy


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


def test_concat_marginal_z_plink_streams_chunks_without_child_contrib_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cupy = make_fake_cupy()
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setattr(
        model_module,
        "_marginal_int8_chunk_size",
        lambda **kwargs: 2,
    )

    genotype_i8 = np.array(
        [
            [0, 1, 2, 0, 1],
            [1, 2, 0, 1, 2],
            [2, 0, 1, 2, 0],
            [0, 1, 2, 0, 1],
            [1, 2, 0, 1, 2],
            [2, 0, 1, 2, 0],
        ],
        dtype=np.int8,
    )
    means = genotype_i8.astype(np.float64).mean(axis=0).astype(np.float32)
    scales = genotype_i8.astype(np.float64).std(axis=0, ddof=0).astype(np.float32)
    raw = genotype_module.PlinkRawGenotypeMatrix(
        bed_path=Path("synthetic.bed"),
        sample_indices=np.arange(genotype_i8.shape[0], dtype=np.intp),
        variant_count=genotype_i8.shape[1],
        total_sample_count=genotype_i8.shape[0],
    )
    concat = genotype_module.ConcatenatedRawGenotypeMatrix(children=(raw,))
    standardized = genotype_module.StandardizedGenotypeMatrix(
        raw=concat,
        means=means,
        scales=scales,
        variant_indices=np.arange(genotype_i8.shape[1], dtype=np.int32),
        support_counts=np.full(genotype_i8.shape[1], genotype_i8.shape[0], dtype=np.int32),
        sample_count=genotype_i8.shape[0],
        _enable_hybrid_backend=False,
    )

    closed_readers: list[int] = []
    read_batches: list[np.ndarray] = []
    released_paths: list[tuple[str, ...]] = []

    class FakeReader:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            if not self.closed:
                closed_readers.append(id(self))
                self.closed = True

    def fake_bed_reader(self: genotype_module.PlinkRawGenotypeMatrix) -> FakeReader:
        reader = FakeReader()
        self._reader = reader
        return reader

    def fake_read_batch_i8(
        self: genotype_module.PlinkRawGenotypeMatrix,
        reader: FakeReader,
        batch_indices: np.ndarray,
        *,
        num_threads: int | None = None,
    ) -> np.ndarray:
        assert not reader.closed
        resolved = np.asarray(batch_indices, dtype=np.int64)
        read_batches.append(resolved.copy())
        return np.asfortranarray(genotype_i8[:, resolved])

    def fake_release_host_caches(cupy: object, *, fadvise_paths=()) -> None:
        released_paths.append(tuple(str(path) for path in fadvise_paths))

    monkeypatch.setattr(genotype_module.PlinkRawGenotypeMatrix, "_bed_reader", fake_bed_reader)
    monkeypatch.setattr(genotype_module.PlinkRawGenotypeMatrix, "_read_batch_i8", fake_read_batch_i8)
    monkeypatch.setattr(model_module, "_release_host_caches", fake_release_host_caches)

    target = np.array([0.25, 1.0, -0.5, 0.75, -1.25, 0.5], dtype=np.float64)
    active = np.arange(genotype_i8.shape[1], dtype=np.int32)
    observed = model_module._compute_marginal_z_scores_concat(
        standardized_genotypes=standardized,
        active_variant_indices=active,
        covariate_matrix=np.zeros((genotype_i8.shape[0], 0), dtype=np.float64),
        target_vector=target,
    )

    y_resid = target - target.mean()
    sigma2_resid = float(np.dot(y_resid, y_resid) / genotype_i8.shape[0])
    standardized_host = (genotype_i8.astype(np.float64) - means[None, :]) / scales[None, :]
    expected_sum_xy = standardized_host.T @ y_resid
    expected = expected_sum_xy / np.sqrt(sigma2_resid * genotype_i8.shape[0])

    assert observed is not None
    np.testing.assert_allclose(observed, expected.astype(np.float32), rtol=1e-5, atol=1e-6)
    assert [batch.tolist() for batch in read_batches] == [[0, 1], [2, 3], [4]]
    assert len(closed_readers) == 3
    assert released_paths == [
        ("synthetic.bed",),
        (),
        ("synthetic.bed",),
        (),
        ("synthetic.bed",),
        (),
    ]


class ShapeOnlyRawGenotypeMatrix(RawGenotypeMatrix):
    def __init__(self, sample_count: int, variant_count: int) -> None:
        self._shape = (int(sample_count), int(variant_count))

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    def iter_column_batches(self, _variant_indices=None, _batch_size: int = 1024):
        raise AssertionError("shape-only test double should not be iterated")

    def materialize(self, _variant_indices=None) -> np.ndarray:
        raise AssertionError("shape-only test double should not be materialized")


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
            minimum_minor_allele_frequency=0.0,
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


def test_runtime_tuned_config_for_gpu_uses_budget_driven_solver_limit(monkeypatch):
    raw_genotypes = ShapeOnlyRawGenotypeMatrix(sample_count=1_000, variant_count=50_000)
    config = ModelConfig(
        trait_type=TraitType.BINARY,
        exact_solver_matrix_limit=20_000,
        sample_space_preconditioner_rank=256,
        stochastic_variant_batch_size=2_048,
    )

    monkeypatch.setattr(runtime_policy_module, "_try_import_cupy", lambda: object())
    monkeypatch.setattr(
        runtime_policy_module,
        "_gpu_materialization_budget_bytes",
        lambda _cupy: 1_000 * 4 * 10_000,
    )

    tuned_config, summary = _runtime_tuned_config_for_fit(config, raw_genotypes)

    assert tuned_config.exact_solver_matrix_limit == 9_000
    assert tuned_config.sample_space_preconditioner_rank == 400
    assert tuned_config.stochastic_variant_batch_size == 8_500
    assert summary is not None
    assert "gpu_profile=budget-driven" in summary


def test_runtime_tuned_config_preserves_disabled_sample_space_preconditioner(monkeypatch):
    raw_genotypes = ShapeOnlyRawGenotypeMatrix(sample_count=1_000, variant_count=50_000)
    config = ModelConfig(
        trait_type=TraitType.BINARY,
        exact_solver_matrix_limit=20_000,
        sample_space_preconditioner_rank=0,
        stochastic_variant_batch_size=2_048,
    )

    monkeypatch.setattr(runtime_policy_module, "_try_import_cupy", lambda: object())
    monkeypatch.setattr(
        runtime_policy_module,
        "_gpu_materialization_budget_bytes",
        lambda _cupy: 1_000 * 4 * 10_000,
    )

    tuned_config, summary = _runtime_tuned_config_for_fit(config, raw_genotypes)

    assert tuned_config.exact_solver_matrix_limit == 9_000
    assert tuned_config.sample_space_preconditioner_rank == 0
    assert tuned_config.stochastic_variant_batch_size == 8_500
    assert summary is not None
    assert "sample_space_preconditioner_rank=0->0" in summary


def test_runtime_tuned_config_caps_binary_stochastic_batch_size_on_small_gpu(monkeypatch):
    raw_genotypes = ShapeOnlyRawGenotypeMatrix(sample_count=1_000, variant_count=50_000)
    config = ModelConfig(
        trait_type=TraitType.BINARY,
        exact_solver_matrix_limit=2_048,
        sample_space_preconditioner_rank=256,
        stochastic_variant_batch_size=8_192,
    )

    monkeypatch.setattr(runtime_policy_module, "_try_import_cupy", lambda: object())
    monkeypatch.setattr(
        runtime_policy_module,
        "_gpu_materialization_budget_bytes",
        lambda _cupy: 1_000 * 4 * 5_844,
    )

    tuned_config, summary = _runtime_tuned_config_for_fit(config, raw_genotypes)

    assert tuned_config.exact_solver_matrix_limit == 2_048
    assert tuned_config.stochastic_variant_batch_size == 4_967
    assert summary is not None
    assert "stochastic_variant_batch_size=8192->4967" in summary


def test_runtime_tuned_config_caps_stochastic_batch_work_on_large_cohort(monkeypatch):
    raw_genotypes = ShapeOnlyRawGenotypeMatrix(sample_count=245_000, variant_count=1_700_000)
    config = ModelConfig(
        trait_type=TraitType.BINARY,
        exact_solver_matrix_limit=2_048,
        sample_space_preconditioner_rank=256,
        stochastic_variant_batch_size=8_192,
    )

    monkeypatch.setattr(runtime_policy_module, "_try_import_cupy", lambda: object())
    monkeypatch.setattr(
        runtime_policy_module,
        "_gpu_materialization_budget_bytes",
        lambda _cupy: 245_000 * 4 * 40_000,
    )

    tuned_config, summary = _runtime_tuned_config_for_fit(config, raw_genotypes)

    assert tuned_config.stochastic_variant_batch_size == 6_998
    assert summary is not None
    assert "stochastic_variant_batch_size=8192->6998" in summary


def test_fit_stage_structure_cache_key_is_shared_across_traits(monkeypatch):
    genotype_matrix, covariate_matrix, target_vector, variant_records = _synthetic_binary_dataset()
    raw_genotypes = as_raw_genotype_matrix(genotype_matrix)
    monkeypatch.setattr(model_module, "_persistent_raw_signature", lambda _genotype_matrix: "synthetic-raw-signature")
    variant_stats = compute_variant_statistics(
        raw_genotypes=raw_genotypes,
        config=ModelConfig(
            trait_type=TraitType.BINARY,
            max_outer_iterations=1,
            minimum_minor_allele_frequency=0.0,
        ),
    )
    prepared_binary = model_module.fit_preprocessor_from_stats(
        variant_stats,
        np.column_stack([np.ones(covariate_matrix.shape[0], dtype=np.float32), covariate_matrix]).astype(np.float32),
        target_vector,
    )
    binary_paths = _fit_stage_cache_paths(
        genotype_matrix=raw_genotypes,
        allele_frequencies=np.asarray([record.allele_frequency for record in variant_records], dtype=np.float32),
        means=prepared_binary.means,
        scales=prepared_binary.scales,
        covariates=prepared_binary.covariates,
        targets=prepared_binary.targets,
        config=ModelConfig(
            trait_type=TraitType.BINARY,
            minimum_minor_allele_frequency=0.0,
        ),
    )
    quantitative_targets = np.asarray(target_vector * 2.0 - 0.5, dtype=np.float32)
    prepared_quantitative = model_module.fit_preprocessor_from_stats(
        variant_stats,
        np.column_stack([np.ones(covariate_matrix.shape[0], dtype=np.float32), covariate_matrix]).astype(np.float32),
        quantitative_targets,
    )
    quantitative_paths = _fit_stage_cache_paths(
        genotype_matrix=raw_genotypes,
        allele_frequencies=np.asarray([record.allele_frequency for record in variant_records], dtype=np.float32),
        means=prepared_quantitative.means,
        scales=prepared_quantitative.scales,
        covariates=prepared_quantitative.covariates,
        targets=prepared_quantitative.targets,
        config=ModelConfig(
            trait_type=TraitType.QUANTITATIVE,
            minimum_minor_allele_frequency=0.0,
        ),
    )

    assert binary_paths is not None
    assert quantitative_paths is not None
    assert binary_paths.key == quantitative_paths.key
    assert binary_paths.manifest_path == quantitative_paths.manifest_path
    assert binary_paths.reduced_raw_i8_path == quantitative_paths.reduced_raw_i8_path
    assert binary_paths.em_checkpoint_path != quantitative_paths.em_checkpoint_path
    assert binary_paths.fit_key != quantitative_paths.fit_key


def test_fit_stage_structure_cache_key_includes_targets_when_marginal_screening(monkeypatch):
    genotype_matrix, covariate_matrix, target_vector, variant_records = _synthetic_binary_dataset()
    raw_genotypes = as_raw_genotype_matrix(genotype_matrix)
    monkeypatch.setattr(model_module, "_persistent_raw_signature", lambda _genotype_matrix: "synthetic-raw-signature")
    variant_stats = compute_variant_statistics(
        raw_genotypes=raw_genotypes,
        config=ModelConfig(
            trait_type=TraitType.BINARY,
            max_outer_iterations=1,
            minimum_minor_allele_frequency=0.0,
        ),
    )
    covariates = np.column_stack(
        [np.ones(covariate_matrix.shape[0], dtype=np.float32), covariate_matrix]
    ).astype(np.float32)

    first_paths = _fit_stage_cache_paths(
        genotype_matrix=raw_genotypes,
        allele_frequencies=np.asarray([record.allele_frequency for record in variant_records], dtype=np.float32),
        means=variant_stats.means,
        scales=variant_stats.scales,
        covariates=covariates,
        targets=target_vector,
        config=ModelConfig(
            trait_type=TraitType.BINARY,
            minimum_minor_allele_frequency=0.0,
            marginal_screen_min_abs_z=1.5,
        ),
    )
    second_paths = _fit_stage_cache_paths(
        genotype_matrix=raw_genotypes,
        allele_frequencies=np.asarray([record.allele_frequency for record in variant_records], dtype=np.float32),
        means=variant_stats.means,
        scales=variant_stats.scales,
        covariates=covariates,
        targets=1.0 - target_vector,
        config=ModelConfig(
            trait_type=TraitType.BINARY,
            minimum_minor_allele_frequency=0.0,
            marginal_screen_min_abs_z=1.5,
        ),
    )

    assert first_paths is not None
    assert second_paths is not None
    assert first_paths.key != second_paths.key
    assert first_paths.active_indices_path != second_paths.active_indices_path


def test_persistent_raw_signature_ignores_mtime_for_cache_backed_memmaps(tmp_path: Path):
    cache_backed_path = tmp_path / ".sv_pgs_cache" / "synthetic.genotypes.npy"
    cache_backed_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_backed_path, np.arange(12, dtype=np.int8).reshape(3, 4))

    raw_genotypes = as_raw_genotype_matrix(np.load(cache_backed_path, mmap_mode="r"))
    signature_before = _persistent_raw_signature(raw_genotypes)
    assert signature_before is not None

    original_stat = cache_backed_path.stat()
    touched_mtime_ns = original_stat.st_mtime_ns + 5_000_000_000
    os.utime(cache_backed_path, ns=(original_stat.st_atime_ns, touched_mtime_ns))

    signature_after = _persistent_raw_signature(raw_genotypes)
    assert signature_after == signature_before


def test_fit_stage_cache_survives_cache_backed_memmap_mtime_changes(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache_backed_path = tmp_path / ".sv_pgs_cache" / "synthetic.genotypes.npy"
    cache_backed_path.parent.mkdir(parents=True, exist_ok=True)
    genotype_matrix = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
        ],
        dtype=np.int8,
    )
    covariate_matrix = np.zeros((genotype_matrix.shape[0], 1), dtype=np.float32)
    target_vector = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    variant_records = [
        VariantRecord("variant_0", VariantClass.SNV, "1", 100, allele_frequency=0.25),
        VariantRecord("variant_1", VariantClass.SNV, "1", 101, allele_frequency=0.25),
        VariantRecord("variant_2", VariantClass.SNV, "1", 102, allele_frequency=0.5),
        VariantRecord("variant_3", VariantClass.SNV, "1", 103, allele_frequency=0.25),
    ]
    np.save(cache_backed_path, genotype_matrix)

    raw_genotypes = as_raw_genotype_matrix(np.load(cache_backed_path, mmap_mode="r"))
    build_tie_map_calls = 0
    original_build_tie_map = model_module.build_tie_map

    def counting_build_tie_map(genotypes, records, config):
        nonlocal build_tie_map_calls
        build_tie_map_calls += 1
        return original_build_tie_map(genotypes, records, config)

    def fake_fit_variational_em(genotypes, covariates, records, **_kwargs):
        return VariationalFitResult(
            alpha=np.zeros(covariates.shape[1], dtype=np.float32),
            beta_reduced=np.zeros(genotypes.shape[1], dtype=np.float32),
            beta_variance=np.ones(genotypes.shape[1], dtype=np.float32),
            prior_scales=np.ones(len(records), dtype=np.float32),
            global_scale=1.0,
            class_tpb_shape_a={VariantClass.SNV: 1.0},
            class_tpb_shape_b={VariantClass.SNV: 0.5},
            scale_model_coefficients=np.zeros(1, dtype=np.float32),
            scale_model_feature_names=["intercept"],
            sigma_error2=1.0,
            objective_history=[0.0],
            validation_history=[],
            member_prior_variances=np.ones(len(records), dtype=np.float32),
        )

    monkeypatch.setattr(model_module, "build_tie_map", counting_build_tie_map)
    monkeypatch.setattr(model_module, "fit_variational_em", fake_fit_variational_em)

    config = ModelConfig(
        trait_type=TraitType.BINARY,
        max_outer_iterations=1,
        minimum_minor_allele_frequency=0.0,
        sample_space_preconditioner_rank=0,
        stochastic_variational_updates=False,
    )

    first_model = BayesianPGS(config).fit(
        raw_genotypes,
        covariate_matrix,
        target_vector,
        variant_records,
    )
    assert first_model.state is not None
    assert build_tie_map_calls == 1

    original_stat = cache_backed_path.stat()
    os.utime(cache_backed_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 5_000_000_000))
    touched_raw_genotypes = as_raw_genotype_matrix(np.load(cache_backed_path, mmap_mode="r"))

    second_model = BayesianPGS(config).fit(
        touched_raw_genotypes,
        covariate_matrix,
        target_vector,
        variant_records,
    )
    assert second_model.state is not None
    assert build_tie_map_calls == 1


def test_fit_stage_cache_persists_cohort_artifacts_for_repeated_fits(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(genotype_module, "require_gpu", lambda: None)
    monkeypatch.setattr(genotype_module.StandardizedGenotypeMatrix, "try_materialize_gpu", lambda self: False)
    cache_backed_path = tmp_path / ".sv_pgs_cache" / "synthetic.genotypes.npy"
    cache_backed_path.parent.mkdir(parents=True, exist_ok=True)
    genotype_matrix = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
        ],
        dtype=np.int8,
    )
    covariate_matrix = np.zeros((genotype_matrix.shape[0], 1), dtype=np.float32)
    target_vector = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    variant_records = [
        VariantRecord("variant_0", VariantClass.SNV, "1", 100, allele_frequency=0.25),
        VariantRecord("variant_1", VariantClass.SNV, "1", 101, allele_frequency=0.25),
        VariantRecord("variant_2", VariantClass.SNV, "1", 102, allele_frequency=0.5),
        VariantRecord("variant_3", VariantClass.SNV, "1", 103, allele_frequency=0.25),
    ]
    np.save(cache_backed_path, genotype_matrix)

    raw_genotypes = as_raw_genotype_matrix(np.load(cache_backed_path, mmap_mode="r"))
    original_compute_variant_statistics = model_module.compute_variant_statistics
    original_build_tie_map = model_module.build_tie_map
    compute_variant_statistics_calls = 0
    build_tie_map_calls = 0

    def counting_compute_variant_statistics(*args, **kwargs):
        nonlocal compute_variant_statistics_calls
        compute_variant_statistics_calls += 1
        return original_compute_variant_statistics(*args, **kwargs)

    def counting_build_tie_map(genotypes, records, config):
        nonlocal build_tie_map_calls
        build_tie_map_calls += 1
        return original_build_tie_map(genotypes, records, config)

    def fake_fit_variational_em(
        genotypes,
        covariates,
        targets,
        records,
        tie_map,
        config,
        resume_checkpoint=None,
        checkpoint_callback=None,
        **_kwargs,
    ):
        return VariationalFitResult(
            alpha=np.zeros(covariates.shape[1], dtype=np.float32),
            beta_reduced=np.zeros(genotypes.shape[1], dtype=np.float32),
            beta_variance=np.ones(genotypes.shape[1], dtype=np.float32),
            prior_scales=np.ones(len(records), dtype=np.float32),
            global_scale=1.0,
            class_tpb_shape_a={VariantClass.SNV: 1.0},
            class_tpb_shape_b={VariantClass.SNV: 0.5},
            scale_model_coefficients=np.zeros(1, dtype=np.float32),
            scale_model_feature_names=["intercept"],
            sigma_error2=1.0,
            objective_history=[0.0],
            validation_history=[],
            member_prior_variances=np.ones(len(records), dtype=np.float32),
        )

    monkeypatch.setattr(model_module, "compute_variant_statistics", counting_compute_variant_statistics)
    monkeypatch.setattr(model_module, "build_tie_map", counting_build_tie_map)
    monkeypatch.setattr(model_module, "fit_variational_em", fake_fit_variational_em)

    config = ModelConfig(
        trait_type=TraitType.BINARY,
        max_outer_iterations=1,
        minimum_minor_allele_frequency=0.0,
        sample_space_preconditioner_rank=0,
        stochastic_variational_updates=False,
    )

    first_model = BayesianPGS(config).fit(
        raw_genotypes,
        covariate_matrix,
        target_vector,
        variant_records,
    )
    assert first_model.state is not None
    fit_stage_dir = tmp_path / ".sv_pgs_cache" / "fit_stage"
    assert len(list(fit_stage_dir.glob("*.variant_stats.npz"))) == 1
    assert len(list(fit_stage_dir.glob("*.reduced_raw_i8.npy"))) == 1
    assert compute_variant_statistics_calls == 1
    assert build_tie_map_calls == 1

    second_raw_genotypes = as_raw_genotype_matrix(np.load(cache_backed_path, mmap_mode="r"))
    second_model = BayesianPGS(config).fit(
        second_raw_genotypes,
        covariate_matrix,
        target_vector,
        variant_records,
    )
    assert second_model.state is not None
    assert compute_variant_statistics_calls == 1
    assert build_tie_map_calls == 1


def test_sample_space_basis_cache_round_trip(tmp_path):
    genotype_matrix, covariate_matrix, target_vector, variant_records = _synthetic_binary_dataset()
    raw_genotypes = as_raw_genotype_matrix(genotype_matrix)
    variant_stats = compute_variant_statistics(
        raw_genotypes=raw_genotypes,
        config=ModelConfig(
            trait_type=TraitType.BINARY,
            max_outer_iterations=1,
            minimum_minor_allele_frequency=0.0,
        ),
    )
    prepared = model_module.fit_preprocessor_from_stats(
        variant_stats,
        np.column_stack([np.ones(covariate_matrix.shape[0], dtype=np.float32), covariate_matrix]).astype(np.float32),
        target_vector,
    )
    standardized = raw_genotypes.standardized(
        prepared.means,
        prepared.scales,
        support_counts=prepared.support_counts,
    )
    cache_paths = _FitStageCachePaths(
        key="basis-structure",
        fit_key="basis-fit",
        cache_dir=tmp_path,
        manifest_path=tmp_path / "basis-structure.manifest.json",
        active_indices_path=tmp_path / "basis-structure.active.npy",
        tie_map_path=tmp_path / "basis-structure.tie.pkl",
        reduced_raw_i8_path=tmp_path / "basis-structure.reduced_raw_i8.npy",
        em_checkpoint_path=tmp_path / "basis-fit.em.pkl",
    )
    basis = np.arange(standardized.shape[0] * 3, dtype=np.float64).reshape(standardized.shape[0], 3)
    standardized._sample_space_nystrom_basis_cpu_cache[(3, 19)] = basis

    assert model_module._save_sample_space_basis_cache(
        cache_paths,
        standardized,
        rank=3,
        random_seed=19,
    )

    restored = standardized.subset(np.arange(standardized.shape[1], dtype=np.int32))
    restored.clear_sample_space_nystrom_cache()

    assert model_module._try_restore_sample_space_basis_cache(
        cache_paths,
        restored,
        rank=3,
        random_seed=19,
    )
    np.testing.assert_allclose(restored._sample_space_nystrom_basis_cpu_cache[(3, 19)], basis)


def test_fit_resumes_from_variational_checkpoint(tmp_path, monkeypatch):
    genotype_matrix, covariate_matrix, target_vector, variant_records = _synthetic_binary_dataset()
    cache_paths = _FitStageCachePaths(
        key="resume-test",
        cache_dir=tmp_path,
        manifest_path=tmp_path / "resume-test.manifest.json",
        active_indices_path=tmp_path / "resume-test.active.npy",
        tie_map_path=tmp_path / "resume-test.tie.pkl",
        reduced_raw_i8_path=tmp_path / "resume-test.reduced_raw_i8.npy",
        em_checkpoint_path=tmp_path / "resume-test.em.pkl",
    )
    call_log: list[int | None] = []

    monkeypatch.setattr(
        model_module,
        "_fit_stage_cache_paths",
        lambda **kwargs: cache_paths,
    )

    def fake_fit_variational_em(
        genotypes,
        covariates,
        targets,
        records,
        tie_map,
        config,
        resume_checkpoint=None,
        checkpoint_callback=None,
        **_kwargs,
    ):
        completed_iterations = None if resume_checkpoint is None else resume_checkpoint.completed_iterations
        call_log.append(completed_iterations)
        reduced_records = mixture_module.collapse_tie_groups(list(records), tie_map)
        prior_design = mixture_module._build_prior_design(reduced_records)
        assert checkpoint_callback is not None
        if resume_checkpoint is None:
            checkpoint_callback(
                VariationalFitCheckpoint(
                    config_signature="resume-config",
                    prior_design_signature="resume-design",
                    validation_enabled=False,
                    completed_iterations=1,
                    alpha_state=np.linspace(0.5, -0.25, covariates.shape[1], dtype=np.float64),
                    beta_state=np.linspace(0.1, -0.2, genotypes.shape[1], dtype=np.float64),
                    local_scale=np.ones(genotypes.shape[1], dtype=np.float64),
                    auxiliary_delta=np.ones(genotypes.shape[1], dtype=np.float64) * 0.5,
                    sigma_error2=1.0,
                    global_scale=0.75,
                    scale_model_coefficients=np.zeros(prior_design.design_matrix.shape[1], dtype=np.float64),
                    tpb_shape_a_vector=np.ones(prior_design.class_membership_matrix.shape[1], dtype=np.float64),
                    tpb_shape_b_vector=np.ones(prior_design.class_membership_matrix.shape[1], dtype=np.float64) * 0.5,
                    objective_history=[-1.0],
                    validation_history=[],
                    previous_alpha=np.linspace(0.5, -0.25, covariates.shape[1], dtype=np.float64),
                    previous_beta=np.linspace(0.1, -0.2, genotypes.shape[1], dtype=np.float64),
                    previous_local_scale=np.ones(genotypes.shape[1], dtype=np.float64),
                    previous_theta=np.zeros(1 + prior_design.design_matrix.shape[1], dtype=np.float64),
                    previous_tpb_shape_a_vector=np.ones(prior_design.class_membership_matrix.shape[1], dtype=np.float64),
                    previous_tpb_shape_b_vector=np.ones(prior_design.class_membership_matrix.shape[1], dtype=np.float64) * 0.5,
                    best_validation_metric=None,
                    best_alpha=None,
                    best_beta=None,
                    best_beta_variance=None,
                    best_local_scale=None,
                    best_theta=None,
                    best_sigma_error2=None,
                    best_tpb_shape_a_vector=None,
                    best_tpb_shape_b_vector=None,
                )
            )
            raise RuntimeError("stop-after-checkpoint")
        return VariationalFitResult(
            alpha=np.zeros(covariates.shape[1], dtype=np.float32),
            beta_reduced=np.zeros(genotypes.shape[1], dtype=np.float32),
            beta_variance=np.ones(genotypes.shape[1], dtype=np.float32),
            prior_scales=np.ones(len(records), dtype=np.float32),
            global_scale=1.0,
            class_tpb_shape_a={VariantClass.SNV: 1.0},
            class_tpb_shape_b={VariantClass.SNV: 0.5},
            scale_model_coefficients=np.zeros(prior_design.design_matrix.shape[1], dtype=np.float32),
            scale_model_feature_names=[f"feature_{index}" for index in range(prior_design.design_matrix.shape[1])],
            sigma_error2=1.0,
            objective_history=[-1.0, -0.5],
            validation_history=[],
            member_prior_variances=np.ones(len(records), dtype=np.float32),
        )

    monkeypatch.setattr(model_module, "fit_variational_em", fake_fit_variational_em)

    with pytest.raises(RuntimeError, match="stop-after-checkpoint"):
        BayesianPGS(
            ModelConfig(
                trait_type=TraitType.BINARY,
                max_outer_iterations=2,
                minimum_minor_allele_frequency=0.0,
            )
        ).fit(genotype_matrix, covariate_matrix, target_vector, variant_records)

    assert cache_paths.em_checkpoint_path.exists()

    resumed_model = BayesianPGS(
        ModelConfig(
            trait_type=TraitType.BINARY,
            max_outer_iterations=2,
            minimum_minor_allele_frequency=0.0,
        )
    ).fit(genotype_matrix, covariate_matrix, target_vector, variant_records)

    assert resumed_model.state is not None
    assert call_log == [None, 1]
    assert not cache_paths.em_checkpoint_path.exists()


def test_fit_checkpoint_persists_basis_cache_during_interrupted_run(tmp_path, monkeypatch):
    genotype_matrix, covariate_matrix, target_vector, variant_records = _synthetic_binary_dataset()
    cache_paths = _FitStageCachePaths(
        key="runtime-basis-test",
        cache_dir=tmp_path,
        manifest_path=tmp_path / "runtime-basis-test.manifest.json",
        active_indices_path=tmp_path / "runtime-basis-test.active.npy",
        tie_map_path=tmp_path / "runtime-basis-test.tie.pkl",
        reduced_raw_i8_path=tmp_path / "runtime-basis-test.reduced_raw_i8.npy",
        em_checkpoint_path=tmp_path / "runtime-basis-test.em.pkl",
    )

    monkeypatch.setattr(model_module, "_fit_stage_cache_paths", lambda **kwargs: cache_paths)
    # Pin the configured rank/seed: the device-driven runtime tuner would
    # otherwise auto-size sample_space_preconditioner_rank (3 -> 512 on a
    # 16GB GPU), so the basis would be cached under key (512, seed) while the
    # fake stores it under (3, 19) — and the asserted r3.seed19 path would
    # never be written.
    monkeypatch.setattr(
        model_module,
        "_runtime_tuned_config_for_fit",
        lambda *, config, genotype_matrix: (config, None),
    )

    def fake_fit_variational_em(
        genotypes,
        covariates,
        targets,
        records,
        tie_map,
        config,
        resume_checkpoint=None,
        checkpoint_callback=None,
        **_kwargs,
    ):
        reduced_records = mixture_module.collapse_tie_groups(list(records), tie_map)
        prior_design = mixture_module._build_prior_design(reduced_records)
        basis = np.arange(genotypes.shape[0] * 3, dtype=np.float64).reshape(genotypes.shape[0], 3)
        genotypes._sample_space_nystrom_basis_cpu_cache[(3, 19)] = basis
        assert checkpoint_callback is not None
        checkpoint_callback(
            VariationalFitCheckpoint(
                config_signature="runtime-basis-config",
                prior_design_signature="runtime-basis-design",
                validation_enabled=False,
                completed_iterations=0,
                alpha_state=np.zeros(covariates.shape[1], dtype=np.float64),
                beta_state=np.zeros(genotypes.shape[1], dtype=np.float64),
                local_scale=np.ones(genotypes.shape[1], dtype=np.float64),
                auxiliary_delta=np.ones(genotypes.shape[1], dtype=np.float64),
                sigma_error2=1.0,
                global_scale=1.0,
                scale_model_coefficients=np.zeros(prior_design.design_matrix.shape[1], dtype=np.float64),
                tpb_shape_a_vector=np.ones(prior_design.class_membership_matrix.shape[1], dtype=np.float64),
                tpb_shape_b_vector=np.ones(prior_design.class_membership_matrix.shape[1], dtype=np.float64),
                objective_history=[],
                validation_history=[],
                previous_alpha=None,
                previous_beta=None,
                previous_local_scale=None,
                previous_theta=None,
                previous_tpb_shape_a_vector=None,
                previous_tpb_shape_b_vector=None,
                best_validation_metric=None,
                best_alpha=None,
                best_beta=None,
                best_beta_variance=None,
                best_local_scale=None,
                best_theta=None,
                best_sigma_error2=None,
                best_tpb_shape_a_vector=None,
                best_tpb_shape_b_vector=None,
                completed_blocks_in_iteration=1,
                beta_variance_state=np.ones(genotypes.shape[1], dtype=np.float64),
                reduced_second_moment=np.ones(genotypes.shape[1], dtype=np.float64),
                epoch_reduced_prior_variances=np.ones(genotypes.shape[1], dtype=np.float64),
            )
        )
        raise RuntimeError("stop-after-runtime-basis-checkpoint")

    monkeypatch.setattr(model_module, "fit_variational_em", fake_fit_variational_em)

    with pytest.raises(RuntimeError, match="stop-after-runtime-basis-checkpoint"):
        BayesianPGS(
            ModelConfig(
                trait_type=TraitType.BINARY,
                max_outer_iterations=2,
                minimum_minor_allele_frequency=0.0,
                sample_space_preconditioner_rank=3,
                random_seed=19,
            )
        ).fit(genotype_matrix, covariate_matrix, target_vector, variant_records)

    basis_path = model_module._sample_space_basis_cache_path(cache_paths, rank=3, random_seed=19)
    assert cache_paths.em_checkpoint_path.exists()
    assert basis_path.exists()
    restored_basis = np.load(basis_path)
    assert restored_basis.shape[1] == 3


def test_try_load_variational_checkpoint_accepts_legacy_checkpoint_pickle(tmp_path):
    cache_paths = _FitStageCachePaths(
        key="legacy-checkpoint-test",
        cache_dir=tmp_path,
        manifest_path=tmp_path / "legacy-checkpoint-test.manifest.json",
        active_indices_path=tmp_path / "legacy-checkpoint-test.active.npy",
        tie_map_path=tmp_path / "legacy-checkpoint-test.tie.pkl",
        reduced_raw_i8_path=tmp_path / "legacy-checkpoint-test.reduced_raw_i8.npy",
        em_checkpoint_path=tmp_path / "legacy-checkpoint-test.em.pkl",
    )
    checkpoint = VariationalFitCheckpoint(
        config_signature="legacy-config",
        prior_design_signature="legacy-design",
        validation_enabled=False,
        completed_iterations=1,
        alpha_state=np.array([1.0, -0.5], dtype=np.float64),
        beta_state=np.array([0.25, -0.75], dtype=np.float64),
        local_scale=np.array([1.0, 1.0], dtype=np.float64),
        auxiliary_delta=np.array([0.5, 0.5], dtype=np.float64),
        sigma_error2=1.0,
        global_scale=0.8,
        scale_model_coefficients=np.array([0.0], dtype=np.float64),
        tpb_shape_a_vector=np.array([1.0], dtype=np.float64),
        tpb_shape_b_vector=np.array([0.5], dtype=np.float64),
        objective_history=[-1.0],
        validation_history=[],
        previous_alpha=None,
        previous_beta=None,
        previous_local_scale=None,
        previous_theta=None,
        previous_tpb_shape_a_vector=None,
        previous_tpb_shape_b_vector=None,
        best_validation_metric=None,
        best_alpha=None,
        best_beta=None,
        best_beta_variance=None,
        best_local_scale=None,
        best_theta=None,
        best_sigma_error2=None,
        best_tpb_shape_a_vector=None,
        best_tpb_shape_b_vector=None,
        completed_blocks_in_iteration=1,
        beta_variance_state=np.array([0.4, 0.6], dtype=np.float64),
        reduced_second_moment=np.array([0.5, 0.7], dtype=np.float64),
        epoch_reduced_prior_variances=np.array([0.9, 1.1], dtype=np.float64),
        binary_block_resume_state={
            "block_indices": np.array([0, 1], dtype=np.int32),
            "solver_state": {"completed_iterations": 1},
        },
    )
    legacy_state = checkpoint.__getstate__()
    legacy_state.pop("binary_block_resume_state")
    legacy_state.pop("stochastic_block_size")

    class _LegacyCheckpointPayload:
        def __reduce__(self):
            return (object.__new__, (VariationalFitCheckpoint,), (None, legacy_state))

    cache_paths.cache_dir.mkdir(parents=True, exist_ok=True)
    with cache_paths.em_checkpoint_path.open("wb") as handle:
        pickle.dump(_LegacyCheckpointPayload(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    restored = model_module._try_load_variational_checkpoint(cache_paths)

    assert restored is not None
    assert restored.binary_block_resume_state is None
    assert restored.stochastic_block_size is None
    np.testing.assert_allclose(restored.alpha_state, checkpoint.alpha_state)
    np.testing.assert_allclose(restored.beta_state, checkpoint.beta_state)
    assert cache_paths.em_checkpoint_path.exists()


def test_corrupt_variational_checkpoint_is_ignored(tmp_path, monkeypatch):
    genotype_matrix, covariate_matrix, target_vector, variant_records = _synthetic_binary_dataset()
    cache_paths = _FitStageCachePaths(
        key="corrupt-checkpoint-test",
        cache_dir=tmp_path,
        manifest_path=tmp_path / "corrupt-checkpoint-test.manifest.json",
        active_indices_path=tmp_path / "corrupt-checkpoint-test.active.npy",
        tie_map_path=tmp_path / "corrupt-checkpoint-test.tie.pkl",
        reduced_raw_i8_path=tmp_path / "corrupt-checkpoint-test.reduced_raw_i8.npy",
        em_checkpoint_path=tmp_path / "corrupt-checkpoint-test.em.pkl",
    )
    cache_paths.em_checkpoint_path.write_bytes(b"not-a-pickle")
    observed_resume_values: list[object] = []

    monkeypatch.setattr(model_module, "_fit_stage_cache_paths", lambda **kwargs: cache_paths)

    def fake_fit_variational_em(
        genotypes,
        covariates,
        targets,
        records,
        tie_map,
        config,
        resume_checkpoint=None,
        checkpoint_callback=None,
        **_kwargs,
    ):
        observed_resume_values.append(resume_checkpoint)
        return VariationalFitResult(
            alpha=np.zeros(covariates.shape[1], dtype=np.float32),
            beta_reduced=np.zeros(genotypes.shape[1], dtype=np.float32),
            beta_variance=np.ones(genotypes.shape[1], dtype=np.float32),
            prior_scales=np.ones(len(records), dtype=np.float32),
            global_scale=1.0,
            class_tpb_shape_a={VariantClass.SNV: 1.0},
            class_tpb_shape_b={VariantClass.SNV: 0.5},
            scale_model_coefficients=np.zeros(1, dtype=np.float32),
            scale_model_feature_names=["feature_0"],
            sigma_error2=1.0,
            objective_history=[0.0],
            validation_history=[],
            member_prior_variances=np.ones(len(records), dtype=np.float32),
        )

    monkeypatch.setattr(model_module, "fit_variational_em", fake_fit_variational_em)

    model = BayesianPGS(
        ModelConfig(
            trait_type=TraitType.BINARY,
            max_outer_iterations=1,
            minimum_minor_allele_frequency=0.0,
        )
    ).fit(genotype_matrix, covariate_matrix, target_vector, variant_records)

    assert model.state is not None
    assert observed_resume_values == [None]
    assert not cache_paths.em_checkpoint_path.exists()


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
                minimum_minor_allele_frequency=0.0,
            )
        ),
    )

    assert set(benchmark_metrics) == {
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
        scale_model_feature_names=["user_annotation"],
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


def test_compact_identity_tie_map_expands_coefficients_without_group_objects():
    tie_map = TieMap(
        kept_indices=np.array([0, 1, 2], dtype=np.int32),
        original_to_reduced=np.array([0, 1, 2], dtype=np.int32),
        reduced_to_group=[],
    )

    expanded = tie_map.expand_coefficients(
        np.array([0.1, -0.2, 0.3], dtype=np.float32),
        group_weights=[],
    )

    np.testing.assert_allclose(expanded, np.array([0.1, -0.2, 0.3], dtype=np.float32))


def test_training_records_from_stats_preserve_prior_continuous_features():
    records = [
        VariantRecord(
            "variant_0",
            VariantClass.DELETION_SHORT,
            "1",
            100,
            prior_binary_features={"coding_annotation": True},
            prior_continuous_features={"sv_length_score": 1.5},
            prior_categorical_features={"functional_state": "lof"},
            prior_membership_features={"regulatory_mix": {"enhancer": 0.75, "promoter": 0.25}},
            prior_nested_features={"gene_context": ("protein_coding", "exon")},
        )
    ]

    training_records = _training_records_from_stats(
        records,
        variant_stats=VariantStatistics(
            means=np.zeros(1, dtype=np.float32),
            scales=np.ones(1, dtype=np.float32),
            allele_frequencies=np.array([0.2], dtype=np.float32),
            support_counts=np.array([7], dtype=np.int32),
        ),
        variant_indices=np.array([0], dtype=np.int32),
    )

    assert training_records[0].training_support == 7
    np.testing.assert_allclose(training_records[0].allele_frequency, 0.2)
    assert training_records[0].prior_binary_features == {"coding_annotation": True}
    assert training_records[0].prior_continuous_features == {"sv_length_score": 1.5}
    assert training_records[0].prior_categorical_features == {"functional_state": "lof"}
    assert training_records[0].prior_membership_features == {"regulatory_mix": {"enhancer": 0.75, "promoter": 0.25}}
    assert training_records[0].prior_nested_features == {"gene_context": ("protein_coding", "exon")}
    assert training_records[0].prior_continuous_features is records[0].prior_continuous_features


def test_normalize_variant_records_reuses_already_normalized_list():
    records = [
        VariantRecord("variant_0", VariantClass.SNV, "1", 100),
        VariantRecord("variant_1", VariantClass.DELETION_SHORT, "1", 101),
    ]

    normalized = _normalize_variant_records(records)

    assert normalized is records


def test_fit_checkpoint_hash_avoids_per_variant_json_for_plain_records(monkeypatch):
    records = [
        VariantRecord("variant_0", VariantClass.SNV, "1", 100),
        VariantRecord("variant_1", VariantClass.DELETION_SHORT, "1", 101),
    ]
    raw_genotypes = as_raw_genotype_matrix(np.zeros((3, 2), dtype=np.int8))
    covariates = np.ones((3, 2), dtype=np.float32)
    targets = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    real_json_dumps = model_module.json.dumps

    def fail_variant_json_dumps(payload, *args, **kwargs):
        if isinstance(payload, dict) and "prior_binary_features" in payload:
            raise AssertionError("plain VariantRecord hashing should not JSON-dump per record")
        return real_json_dumps(payload, *args, **kwargs)

    monkeypatch.setattr(model_module.json, "dumps", fail_variant_json_dumps)

    first = _fit_checkpoint_config_hash(
        genotype_matrix=raw_genotypes,
        covariates=covariates,
        targets=targets,
        variant_records=records,
        config=ModelConfig(trait_type=TraitType.BINARY, max_outer_iterations=1),
    )
    second = _fit_checkpoint_config_hash(
        genotype_matrix=raw_genotypes,
        covariates=covariates,
        targets=targets,
        variant_records=records,
        config=ModelConfig(trait_type=TraitType.BINARY, max_outer_iterations=1),
    )

    assert first == second
    assert len(first) == 64


def test_tie_map_keeps_all_active_variants_detects_identity_kept_indices():
    identity_tie_map = TieMap(
        kept_indices=np.array([0, 1, 2], dtype=np.int32),
        original_to_reduced=np.array([0, 1, 2], dtype=np.int32),
        reduced_to_group=[
            TieGroup(0, np.array([0], dtype=np.int32), np.array([1.0], dtype=np.float32)),
            TieGroup(1, np.array([1], dtype=np.int32), np.array([1.0], dtype=np.float32)),
            TieGroup(2, np.array([2], dtype=np.int32), np.array([1.0], dtype=np.float32)),
        ],
    )
    skipped_tie_map = TieMap(
        kept_indices=np.array([0, 2], dtype=np.int32),
        original_to_reduced=np.array([0, -1, 1], dtype=np.int32),
        reduced_to_group=[],
    )

    assert _tie_map_keeps_all_active_variants(identity_tie_map, 3)
    assert not _tie_map_keeps_all_active_variants(skipped_tie_map, 3)


def test_active_indices_cover_original_detects_identity_indices():
    assert _active_indices_cover_original(np.array([0, 1, 2], dtype=np.int32), 3)
    assert not _active_indices_cover_original(np.array([0, 2], dtype=np.int32), 3)


def test_fit_uses_cohort_allele_frequencies_for_maf_filter(monkeypatch):
    genotype_matrix = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [2.0, 1.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )
    covariate_matrix = np.zeros((genotype_matrix.shape[0], 1), dtype=np.float32)
    target_vector = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    variant_records = [
        VariantRecord("variant_0", VariantClass.SNV, "1", 100, allele_frequency=1e-5),
        VariantRecord("variant_1", VariantClass.SNV, "1", 101, allele_frequency=0.25),
    ]
    config = ModelConfig(
        trait_type=TraitType.BINARY,
        minimum_minor_allele_frequency=0.1,
        max_outer_iterations=1,
    )
    variant_stats = compute_variant_statistics(
        raw_genotypes=as_raw_genotype_matrix(genotype_matrix),
        config=config,
    )

    def fake_fit_variational_em(
        genotypes,
        covariates,
        targets,
        records,
        tie_map,
        config,
        resume_checkpoint=None,
        checkpoint_callback=None,
        **_kwargs,
    ):
        reduced_count = genotypes.shape[1]
        active_count = len(records)
        return VariationalFitResult(
            alpha=np.zeros(covariates.shape[1], dtype=np.float32),
            beta_reduced=np.zeros(reduced_count, dtype=np.float32),
            beta_variance=np.ones(reduced_count, dtype=np.float32),
            prior_scales=np.ones(active_count, dtype=np.float32),
            global_scale=1.0,
            class_tpb_shape_a=dict(config.class_tpb_shape_a()),
            class_tpb_shape_b=dict(config.class_tpb_shape_b()),
            scale_model_coefficients=np.zeros(1, dtype=np.float32),
            scale_model_feature_names=["intercept"],
            sigma_error2=1.0,
            objective_history=[0.0],
            validation_history=[],
            member_prior_variances=np.ones(active_count, dtype=np.float32),
        )

    monkeypatch.setattr(model_module, "fit_variational_em", fake_fit_variational_em)

    model = BayesianPGS(config).fit(
        genotype_matrix,
        covariate_matrix,
        target_vector,
        variant_records,
        variant_stats=variant_stats,
    )

    assert model.state is not None
    assert model.state.variant_records[0].allele_frequency > 0.1
    assert model.state.active_variant_indices.tolist() == [0, 1]


def test_coefficient_table_preserves_full_variant_alignment_after_filtering(tmp_path: Path, monkeypatch):
    genotype_matrix = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    covariate_matrix = np.zeros((genotype_matrix.shape[0], 1), dtype=np.float32)
    target_vector = np.array([0.1, 1.0, 1.8, 1.1, 0.0, 2.7], dtype=np.float32)
    variant_records = [
        VariantRecord("rare_filtered", VariantClass.SNV, "1", 100, allele_frequency=0.01),
        VariantRecord("common_keep_1", VariantClass.DELETION_SHORT, "1", 200, allele_frequency=0.25, length=400.0),
        VariantRecord("zero_filtered", VariantClass.SNV, "1", 300, allele_frequency=0.0),
        VariantRecord("common_keep_2", VariantClass.DUPLICATION_SHORT, "1", 400, allele_frequency=0.5, length=900.0),
    ]
    config = ModelConfig(
        trait_type=TraitType.QUANTITATIVE,
        minimum_minor_allele_frequency=0.2,
        max_outer_iterations=1,
    )
    variant_stats = compute_variant_statistics(
        raw_genotypes=as_raw_genotype_matrix(genotype_matrix),
        config=config,
    )

    def fake_fit_variational_em(
        genotypes,
        covariates,
        targets,
        records,
        tie_map,
        config,
        resume_checkpoint=None,
        checkpoint_callback=None,
        **_kwargs,
    ):
        assert [record.variant_id for record in records] == ["common_keep_1", "common_keep_2"]
        return VariationalFitResult(
            alpha=np.zeros(covariates.shape[1], dtype=np.float32),
            beta_reduced=np.array([1.5, -0.25], dtype=np.float32),
            beta_variance=np.ones(genotypes.shape[1], dtype=np.float32),
            prior_scales=np.ones(len(records), dtype=np.float32),
            global_scale=1.0,
            class_tpb_shape_a=dict(config.class_tpb_shape_a()),
            class_tpb_shape_b=dict(config.class_tpb_shape_b()),
            scale_model_coefficients=np.zeros(1, dtype=np.float32),
            scale_model_feature_names=["intercept"],
            sigma_error2=1.0,
            objective_history=[0.0],
            validation_history=[],
            member_prior_variances=np.ones(len(records), dtype=np.float32),
        )

    monkeypatch.setattr(model_module, "fit_variational_em", fake_fit_variational_em)

    model = BayesianPGS(config).fit(
        genotype_matrix,
        covariate_matrix,
        target_vector,
        variant_records,
        variant_stats=variant_stats,
    )

    training_components = model.training_decision_components()
    assert training_components is not None
    recomputed_components = model.decision_components(genotype_matrix, covariate_matrix)
    np.testing.assert_allclose(training_components[0], recomputed_components[0], atol=1e-6)
    np.testing.assert_allclose(training_components[1], recomputed_components[1], atol=1e-6)

    coefficient_rows = model.coefficient_table()
    assert [row["variant_id"] for row in coefficient_rows] == [record.variant_id for record in variant_records]
    assert [row["variant_class"] for row in coefficient_rows] == [
        record.variant_class.value for record in variant_records
    ]
    assert [float(cast(float, row["beta"])) for row in coefficient_rows] == pytest.approx([0.0, 1.5, 0.0, -0.25])

    artifact_dir = tmp_path / "filtered_artifact"
    model.export(artifact_dir)
    loaded_rows = BayesianPGS.load(artifact_dir).coefficient_table()
    assert [row["variant_id"] for row in loaded_rows] == [record.variant_id for record in variant_records]
    assert [float(cast(float, row["beta"])) for row in loaded_rows] == pytest.approx([0.0, 1.5, 0.0, -0.25])


def test_model_fit_supports_covariates_only_when_no_variants_survive(tmp_path):
    genotype_matrix = np.zeros((6, 3), dtype=np.float32)
    alternate_genotypes = np.array(
        [
            [0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 2.0, 1.0],
            [2.0, 0.0, 2.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    covariate_matrix = np.array(
        [[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]],
        dtype=np.float32,
    )
    target_vector = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
    variant_records = [
        VariantRecord("variant_0", VariantClass.SNV, "1", 100, allele_frequency=0.2),
        VariantRecord("variant_1", VariantClass.SNV, "1", 101, allele_frequency=0.2),
        VariantRecord("variant_2", VariantClass.SNV, "1", 102, allele_frequency=0.2),
    ]

    model = BayesianPGS(
        ModelConfig(
            trait_type=TraitType.BINARY,
            minimum_minor_allele_frequency=0.01,
            max_outer_iterations=5,
        )
    ).fit(genotype_matrix, covariate_matrix, target_vector, variant_records)

    assert model.state is not None
    assert model.state.active_variant_indices.shape == (0,)
    assert model.state.tie_map.kept_indices.shape == (0,)
    assert np.all(model.state.tie_map.original_to_reduced == -1)
    np.testing.assert_array_equal(model.state.full_coefficients, np.zeros(genotype_matrix.shape[1], dtype=np.float32))
    probabilities = model.predict_proba(genotype_matrix, covariate_matrix)[:, 1]
    assert roc_auc_score(target_vector, probabilities) > 0.95
    np.testing.assert_allclose(
        model.decision_function(genotype_matrix, covariate_matrix),
        model.decision_function(alternate_genotypes, covariate_matrix),
    )

    artifact_directory = tmp_path / "covariates_only_artifact"
    model.export(artifact_directory)
    loaded_model = BayesianPGS.load(artifact_directory)
    np.testing.assert_allclose(
        loaded_model.predict_proba(alternate_genotypes, covariate_matrix),
        model.predict_proba(alternate_genotypes, covariate_matrix),
    )


def test_fit_rejects_invalid_variant_stats_allele_frequencies():
    genotype_matrix, covariate_matrix, target_vector, variant_records = _synthetic_binary_dataset()

    with np.testing.assert_raises_regex(
        ValueError,
        r"variant_stats.allele_frequencies must be finite and lie in \[0.0, 1.0\]\.",
    ):
        BayesianPGS(
            ModelConfig(
                trait_type=TraitType.BINARY,
                minimum_minor_allele_frequency=0.0,
                max_outer_iterations=2,
            )
        ).fit(
            genotype_matrix,
            covariate_matrix,
            target_vector,
            variant_records,
            variant_stats=VariantStatistics(
                means=np.zeros(genotype_matrix.shape[1], dtype=np.float32),
                scales=np.ones(genotype_matrix.shape[1], dtype=np.float32),
                allele_frequencies=np.array([0.2, 0.1, 1.2, 0.3, 0.4], dtype=np.float32),
                support_counts=np.ones(genotype_matrix.shape[1], dtype=np.int32),
            ),
        )


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
            variant_stats=VariantStatistics(
                means=np.zeros(genotype_matrix.shape[1] - 1, dtype=np.float32),
                scales=np.ones(genotype_matrix.shape[1], dtype=np.float32),
                allele_frequencies=np.zeros(genotype_matrix.shape[1], dtype=np.float32),
                support_counts=np.ones(genotype_matrix.shape[1], dtype=np.int32),
            ),
        )


def test_validation_path_keeps_raw_genotypes_streaming():
    class NonMaterializingValidationMatrix(RawGenotypeMatrix):
        def __init__(self, matrix: np.ndarray) -> None:
            self.matrix = np.asarray(matrix, dtype=np.float32)
            self.materialize_calls = 0

        @property
        def shape(self) -> tuple[int, int]:
            return int(self.matrix.shape[0]), int(self.matrix.shape[1])

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


def test_validation_reduction_does_not_build_full_width_hybrid_backend(monkeypatch):
    class SpyInt8ValidationMatrix(RawGenotypeMatrix):
        def __init__(self, matrix: np.ndarray) -> None:
            self.matrix = np.asarray(matrix, dtype=np.int8)
            self.i8_requests: list[list[int]] = []

        @property
        def shape(self) -> tuple[int, int]:
            return int(self.matrix.shape[0]), int(self.matrix.shape[1])

        def iter_column_batches_i8(self, variant_indices=None, batch_size: int = 1024):
            resolved_indices = (
                np.arange(self.matrix.shape[1], dtype=np.int32)
                if variant_indices is None
                else np.asarray(variant_indices, dtype=np.int32)
            )
            self.i8_requests.append(resolved_indices.tolist())
            for start_index in range(0, resolved_indices.shape[0], max(int(batch_size), 1)):
                batch_indices = resolved_indices[start_index : start_index + batch_size]
                yield RawGenotypeBatch(
                    variant_indices=batch_indices,
                    values=np.asarray(self.matrix[:, batch_indices], dtype=np.int8),
                )

        def iter_column_batches(self, variant_indices=None, batch_size: int = 1024):
            raise AssertionError("validation reduction should not use float raw batches")

        def materialize(self, variant_indices=None) -> np.ndarray:
            resolved_indices = (
                np.arange(self.matrix.shape[1], dtype=np.int32)
                if variant_indices is None
                else np.asarray(variant_indices, dtype=np.int32)
            )
            return np.asarray(self.matrix[:, resolved_indices], dtype=np.float32)

    sample_count = 8
    variant_count = 80
    active_indices = np.array([2, 17, 61], dtype=np.int32)
    rng = np.random.default_rng(123)
    train_genotypes = rng.integers(0, 3, size=(sample_count, variant_count)).astype(np.float32)
    covariate_matrix = np.zeros((sample_count, 1), dtype=np.float32)
    target_vector = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
    variant_records = [
        VariantRecord(f"variant_{index}", VariantClass.SNV, "1", 100 + index)
        for index in range(variant_count)
    ]
    allele_frequencies = np.full(variant_count, 0.001, dtype=np.float32)
    allele_frequencies[active_indices] = 0.25
    variant_stats = VariantStatistics(
        means=np.zeros(variant_count, dtype=np.float32),
        scales=np.ones(variant_count, dtype=np.float32),
        allele_frequencies=allele_frequencies,
        support_counts=np.ones(variant_count, dtype=np.int32),
    )
    validation_genotypes = SpyInt8ValidationMatrix(train_genotypes.astype(np.int8))

    monkeypatch.setattr(genotype_module, "require_gpu", lambda: None)
    monkeypatch.setattr(
        model_module,
        "build_tie_map",
        lambda _genotypes, _records, _config: TieMap(
            kept_indices=np.arange(active_indices.shape[0], dtype=np.int32),
            original_to_reduced=np.arange(active_indices.shape[0], dtype=np.int32),
            reduced_to_group=[],
        ),
    )
    monkeypatch.setattr(
        genotype_module.StandardizedGenotypeMatrix,
        "try_materialize_gpu",
        lambda self: False,
    )
    monkeypatch.setattr(
        genotype_module.StandardizedGenotypeMatrix,
        "try_materialize",
        lambda self: False,
    )

    def fake_fit_variational_em(**kwargs):
        reduced_variant_count = int(kwargs["genotypes"].shape[1])
        return VariationalFitResult(
            alpha=np.zeros(covariate_matrix.shape[1] + 1, dtype=np.float32),
            beta_reduced=np.zeros(reduced_variant_count, dtype=np.float32),
            beta_variance=np.ones(reduced_variant_count, dtype=np.float64),
            prior_scales=np.ones(reduced_variant_count, dtype=np.float64),
            global_scale=1.0,
            class_tpb_shape_a={VariantClass.SNV: 1.0},
            class_tpb_shape_b={VariantClass.SNV: 1.0},
            scale_model_coefficients=np.zeros(1, dtype=np.float64),
            scale_model_feature_names=["intercept"],
            sigma_error2=1.0,
            objective_history=[0.0],
            validation_history=[],
            member_prior_variances=np.ones(reduced_variant_count, dtype=np.float64),
            linear_predictor=np.zeros(sample_count, dtype=np.float32),
            selected_iteration_count=1,
            converged=True,
        )

    monkeypatch.setattr(model_module, "fit_variational_em", fake_fit_variational_em)

    model = BayesianPGS(
        ModelConfig(
            trait_type=TraitType.QUANTITATIVE,
            max_outer_iterations=1,
            minimum_minor_allele_frequency=0.05,
        )
    ).fit(
        train_genotypes,
        covariate_matrix,
        target_vector,
        variant_records,
        validation_data=(validation_genotypes, covariate_matrix, target_vector),
        variant_stats=variant_stats,
    )

    assert model.state is not None
    assert validation_genotypes.i8_requests == []


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
            minimum_minor_allele_frequency=0.0,
        )
    ).fit(genotype_matrix, covariate_matrix, target_vector, variant_records)

    assert model.state is not None


def test_aou_scale_fit_skips_full_active_set_materialization():
    matrix = type("ShapeOnly", (), {"shape": (331_945, 695_875)})()
    config = ModelConfig(
        trait_type=TraitType.BINARY,
        stochastic_variant_batch_size=6_185,
    )

    assert model_module._skip_full_matrix_materialization_for_blocks(
        matrix,
        config,
        label="training",
    )


def test_small_fit_allows_full_materialization():
    matrix = type("ShapeOnly", (), {"shape": (16, 24)})()
    config = ModelConfig(
        trait_type=TraitType.BINARY,
        stochastic_variant_batch_size=8,
    )

    assert not model_module._skip_full_matrix_materialization_for_blocks(
        matrix,
        config,
        label="training",
    )


def test_large_stochastic_fit_skips_full_active_set_materialization(monkeypatch):
    sample_count = 16
    variant_count = 24
    rng = np.random.default_rng(456)
    genotype_matrix = rng.integers(0, 3, size=(sample_count, variant_count)).astype(np.float32)
    covariate_matrix = np.zeros((sample_count, 1), dtype=np.float32)
    target_vector = rng.binomial(1, 0.5, size=sample_count).astype(np.float32)
    variant_records = [
        VariantRecord(f"variant_{index}", VariantClass.SNV, "1", 100 + index)
        for index in range(variant_count)
    ]
    variant_stats = VariantStatistics(
        means=np.zeros(variant_count, dtype=np.float32),
        scales=np.ones(variant_count, dtype=np.float32),
        allele_frequencies=np.full(variant_count, 0.25, dtype=np.float32),
        support_counts=np.ones(variant_count, dtype=np.int32),
    )

    def forbid_full_cache(self):
        if int(self.shape[1]) > 8:
            raise AssertionError("full active-set materialization should be skipped")
        return False

    monkeypatch.setattr(genotype_module.StandardizedGenotypeMatrix, "try_materialize_gpu", forbid_full_cache)
    monkeypatch.setattr(genotype_module.StandardizedGenotypeMatrix, "try_materialize", forbid_full_cache)
    monkeypatch.setattr(genotype_module.StandardizedGenotypeMatrix, "try_cache_locally", forbid_full_cache)
    monkeypatch.setattr(
        model_module,
        "_skip_full_matrix_materialization_for_blocks",
        lambda *args, **kwargs: True,
    )

    captured_shape: tuple[int, int] | None = None

    def fake_fit_variational_em(**kwargs):
        nonlocal captured_shape
        captured_shape = tuple(kwargs["genotypes"].shape)
        return VariationalFitResult(
            alpha=np.zeros(covariate_matrix.shape[1] + 1, dtype=np.float32),
            beta_reduced=np.zeros(variant_count, dtype=np.float32),
            beta_variance=np.ones(variant_count, dtype=np.float64),
            prior_scales=np.ones(variant_count, dtype=np.float64),
            global_scale=1.0,
            class_tpb_shape_a={VariantClass.SNV: 1.0},
            class_tpb_shape_b={VariantClass.SNV: 1.0},
            scale_model_coefficients=np.zeros(1, dtype=np.float64),
            scale_model_feature_names=["intercept"],
            sigma_error2=1.0,
            objective_history=[0.0],
            validation_history=[],
            member_prior_variances=np.ones(variant_count, dtype=np.float64),
            linear_predictor=np.zeros(sample_count, dtype=np.float32),
            selected_iteration_count=1,
            converged=True,
        )

    monkeypatch.setattr(model_module, "fit_variational_em", fake_fit_variational_em)

    model = BayesianPGS(
        ModelConfig(
            trait_type=TraitType.BINARY,
            max_outer_iterations=1,
            minimum_minor_allele_frequency=0.0,
            stochastic_variant_batch_size=8,
        )
    ).fit(
        genotype_matrix,
        covariate_matrix,
        target_vector,
        variant_records,
        variant_stats=variant_stats,
    )

    assert model.state is not None
    assert captured_shape == (sample_count, variant_count)


def test_raw_standardized_subset_matvec_reads_only_requested_columns():
    class SpyRawGenotypeMatrix(RawGenotypeMatrix):
        def __init__(self, matrix: np.ndarray) -> None:
            self.matrix = np.asarray(matrix, dtype=np.float32)
            self.requested_variant_indices: list[list[int]] = []

        @property
        def shape(self) -> tuple[int, int]:
            return int(self.matrix.shape[0]), int(self.matrix.shape[1])

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
