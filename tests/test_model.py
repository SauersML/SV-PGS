from __future__ import annotations

import os
from pathlib import Path

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
from sv_pgs.data import TieGroup, TieMap
from sv_pgs.genotype import RawGenotypeBatch, RawGenotypeMatrix, as_raw_genotype_matrix
from sv_pgs.inference import VariationalFitCheckpoint, VariationalFitResult
from sv_pgs.model import (
    _FitStageCachePaths,
    _fit_stage_cache_paths,
    _persistent_raw_signature,
    _raw_standardized_subset_matvec,
    _runtime_tuned_config_for_fit,
    _tie_group_export_weights,
    _training_records_from_stats,
)
from sv_pgs.preprocessing import compute_variant_statistics


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


class ShapeOnlyRawGenotypeMatrix(RawGenotypeMatrix):
    def __init__(self, sample_count: int, variant_count: int) -> None:
        self._shape = (int(sample_count), int(variant_count))

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    def iter_column_batches(self, variant_indices=None, batch_size: int = 1024):
        raise AssertionError("shape-only test double should not be iterated")

    def materialize(self, variant_indices=None) -> np.ndarray:
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


def test_runtime_tuned_config_for_t4_caps_solver_from_gpu_budget(monkeypatch):
    raw_genotypes = ShapeOnlyRawGenotypeMatrix(sample_count=1_000, variant_count=50_000)
    config = ModelConfig(
        trait_type=TraitType.BINARY,
        exact_solver_matrix_limit=2_048,
        sample_space_preconditioner_rank=256,
    )

    monkeypatch.setattr(runtime_policy_module, "_try_import_cupy", lambda: object())
    monkeypatch.setattr(
        runtime_policy_module,
        "_gpu_materialization_budget_bytes",
        lambda _cupy: 1_000 * 4 * 10_000,
    )

    tuned_config, summary = _runtime_tuned_config_for_fit(config, raw_genotypes)

    assert tuned_config.exact_solver_matrix_limit == 1_024
    assert tuned_config.sample_space_preconditioner_rank == 256
    assert tuned_config.final_posterior_refinement is False
    assert summary is not None


def test_fit_stage_structure_cache_key_is_shared_across_traits(monkeypatch):
    genotype_matrix, covariate_matrix, target_vector, variant_records = _synthetic_binary_dataset()
    raw_genotypes = as_raw_genotype_matrix(genotype_matrix)
    monkeypatch.setattr(model_module, "_persistent_raw_signature", lambda genotype_matrix: "synthetic-raw-signature")
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

    def fake_fit_variational_em(
        genotypes,
        covariates,
        targets,
        records,
        tie_map,
        config,
        validation_data,
        resume_checkpoint=None,
        checkpoint_callback=None,
        predictor_offset=None,
        validation_offset=None,
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

    monkeypatch.setattr(model_module, "build_tie_map", counting_build_tie_map)
    monkeypatch.setattr(model_module, "fit_variational_em", fake_fit_variational_em)

    config = ModelConfig(
        trait_type=TraitType.BINARY,
        max_outer_iterations=1,
        minimum_minor_allele_frequency=0.0,
        sample_space_preconditioner_rank=0,
        final_posterior_refinement=False,
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
        validation_data,
        resume_checkpoint=None,
        checkpoint_callback=None,
        predictor_offset=None,
        validation_offset=None,
    ):
        completed_iterations = None if resume_checkpoint is None else resume_checkpoint.completed_iterations
        call_log.append(completed_iterations)
        reduced_records = mixture_module.collapse_tie_groups(list(records), tie_map)
        prior_design = mixture_module._build_prior_design(reduced_records)
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

    def fake_fit_variational_em(
        genotypes,
        covariates,
        targets,
        records,
        tie_map,
        config,
        validation_data,
        resume_checkpoint=None,
        checkpoint_callback=None,
        predictor_offset=None,
        validation_offset=None,
    ):
        reduced_records = mixture_module.collapse_tie_groups(list(records), tie_map)
        prior_design = mixture_module._build_prior_design(reduced_records)
        basis = np.arange(genotypes.shape[0] * 3, dtype=np.float64).reshape(genotypes.shape[0], 3)
        genotypes._sample_space_nystrom_basis_cpu_cache[(3, 19)] = basis
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
        validation_data,
        resume_checkpoint=None,
        checkpoint_callback=None,
        predictor_offset=None,
        validation_offset=None,
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
            prior_binary_features={"coding_annotation": True},
            prior_continuous_features={"sv_length_score": 1.5},
        )
    ]

    training_records = _training_records_from_stats(
        records,
        variant_stats=type(
            "Stats",
            (),
            {
                "allele_frequencies": np.array([0.2], dtype=np.float32),
                "support_counts": np.array([7], dtype=np.int32),
            },
        )(),
    )

    assert training_records[0].training_support == 7
    np.testing.assert_allclose(training_records[0].allele_frequency, 0.2)
    assert training_records[0].prior_binary_features == {"coding_annotation": True}
    assert training_records[0].prior_continuous_features == {"sv_length_score": 1.5}


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
        validation_data,
        resume_checkpoint=None,
        checkpoint_callback=None,
        predictor_offset=None,
        validation_offset=None,
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
            variant_stats=type(
                "Stats",
                (),
                {
                    "means": np.zeros(genotype_matrix.shape[1], dtype=np.float32),
                    "scales": np.ones(genotype_matrix.shape[1], dtype=np.float32),
                    "allele_frequencies": np.array([0.2, 0.1, 1.2, 0.3, 0.4], dtype=np.float32),
                    "support_counts": np.ones(genotype_matrix.shape[1], dtype=np.int32),
                },
            )(),
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
