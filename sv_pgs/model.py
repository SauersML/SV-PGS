from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import sv_pgs._jax  # noqa: F401
import jax.numpy as jnp

from sv_pgs.artifact import ModelArtifact, load_artifact, save_artifact
from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.data import TieGroup, TieMap, VariantRecord, VariantStatistics, normalize_variant_records
from sv_pgs.genotype import (
    RawGenotypeMatrix,
    _standardize_batch,
    as_raw_genotype_matrix,
    auto_batch_size,
)
from sv_pgs.inference import VariationalFitResult, fit_variational_em
from sv_pgs.numeric import stable_sigmoid
from sv_pgs.preprocessing import (
    Preprocessor,
    build_tie_map,
    fit_preprocessor,
    fit_preprocessor_from_stats,
    select_active_variant_indices,
)
from sv_pgs.progress import log, mem

STRUCTURAL_VARIANT_CLASSES = set(ModelConfig.structural_variant_classes())


@dataclass(slots=True)
class FittedState:
    variant_records: list[VariantRecord]
    active_variant_indices: np.ndarray
    preprocessor: Preprocessor
    tie_map: TieMap
    fit_result: VariationalFitResult
    full_coefficients: np.ndarray
    nonzero_coefficient_indices: np.ndarray
    nonzero_coefficients: np.ndarray
    nonzero_means: np.ndarray
    nonzero_scales: np.ndarray


class BayesianPGS:
    """Main entry point for fitting and applying a Bayesian Polygenic Score.

    A PGS predicts a trait (disease risk or continuous measurement) from an
    individual's genotypes across many variants.  This implementation uses a
    Bayesian approach that:

      1. Automatically learns which variants matter (most get shrunk to ~zero)
      2. Gives structural variants (deletions, duplications, etc.) wider priors
         than SNVs, reflecting their potentially larger per-variant effects
      3. Handles hundreds of thousands of samples and variants via streaming
         genotype I/O, GPU-accelerated statistics, and efficient linear algebra

    Typical workflow:
        model = BayesianPGS(config)
        model.fit(genotypes, covariates, targets, variant_records)
        predictions = model.predict(new_genotypes, new_covariates)
        model.export("model.npz")
    """
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.state: FittedState | None = None

    def fit(
        self,
        genotypes: RawGenotypeMatrix | np.ndarray,
        covariates: np.ndarray,
        targets: np.ndarray,
        variant_records: Sequence[VariantRecord | dict],
        validation_data: tuple[RawGenotypeMatrix | np.ndarray, np.ndarray, np.ndarray] | None = None,
        variant_stats: VariantStatistics | None = None,
    ) -> BayesianPGS:
        log(f"=== MODEL FIT START ===  genotypes={genotypes.shape}  covariates={covariates.shape}  targets={targets.shape}  pre_computed_stats={'YES' if variant_stats else 'NO'}")
        raw_genotype_matrix = as_raw_genotype_matrix(genotypes)

        # Use pre-computed stats if available (saves 3 full data passes)
        if variant_stats is not None:
            log("using pre-computed variant statistics (means, scales, support) [NO DATA PASSES]")
            normalized_records = _training_records_from_stats(
                normalize_variant_records(variant_records), variant_stats,
            )
            covariate_matrix = self._with_intercept(covariates)
            prepared_arrays = fit_preprocessor_from_stats(variant_stats, covariate_matrix, targets)
        else:
            log("normalizing variant records and computing training support...")
            normalized_records = _training_records(raw_genotype_matrix, normalize_variant_records(variant_records), self.config)
            covariate_matrix = self._with_intercept(covariates)
            log(f"fitting preprocessor (streaming mean/scale over {raw_genotype_matrix.shape[1]} variants)...")
            prepared_arrays = fit_preprocessor(raw_genotype_matrix, covariate_matrix, targets, self.config)
        preprocessor = Preprocessor(means=prepared_arrays.means, scales=prepared_arrays.scales)
        log(f"preprocessor ready  {len(normalized_records)} variant records  mem={mem()}")

        log("creating standardized genotype view...")
        standardized_genotypes = raw_genotype_matrix.standardized(prepared_arrays.means, prepared_arrays.scales)

        log("selecting active variant indices (filtering low-carrier SVs)...")
        active_variant_indices = select_active_variant_indices(
            variant_records=normalized_records,
            support_counts=variant_stats.support_counts if variant_stats is not None else _compute_support_counts(raw_genotype_matrix, normalized_records, self.config),
            config=self.config,
            marginal_scores=None if variant_stats is None else variant_stats.marginal_scores,
        )
        log(f"active variants: {len(active_variant_indices)} / {len(normalized_records)} ({100.0*len(active_variant_indices)/max(len(normalized_records),1):.1f}%)")
        active_records = [normalized_records[int(variant_index)] for variant_index in active_variant_indices]
        active_genotypes = standardized_genotypes.subset(active_variant_indices)

        if active_variant_indices.shape[0] > self.config.maximum_tie_map_variants:
            log(
                f"skipping tie map for {active_variant_indices.shape[0]} active variants "
                f"(limit={self.config.maximum_tie_map_variants}); using identity mapping"
            )
            reduced_tie_map = _identity_tie_map(active_variant_indices.shape[0])
        else:
            log("building tie map (detecting identical/negated genotype columns)...")
            reduced_tie_map = build_tie_map(active_genotypes, active_records, self.config)
        original_space_tie_map = _project_tie_map_to_original_space(
            reduced_tie_map=reduced_tie_map,
            active_variant_indices=active_variant_indices,
            original_variant_count=len(normalized_records),
        )
        # Combine active + tie-map indices into one subset call to avoid
        # intermediate GPU/RAM copies.
        combined_indices = active_variant_indices[reduced_tie_map.kept_indices]
        reduced_genotypes = standardized_genotypes.subset(combined_indices)
        log(f"tie map: {len(active_variant_indices)} active -> {len(reduced_tie_map.kept_indices)} unique ({len(reduced_tie_map.reduced_to_group)} groups)  mem={mem()}")

        reduced_validation = None
        if validation_data is not None:
            validation_genotypes, validation_covariates, validation_targets = validation_data
            standardized_validation = as_raw_genotype_matrix(validation_genotypes).standardized(
                prepared_arrays.means,
                prepared_arrays.scales,
            )
            combined_validation_indices = active_variant_indices[reduced_tie_map.kept_indices]
            reduced_validation = (
                standardized_validation.subset(combined_validation_indices),
                self._with_intercept(np.asarray(validation_covariates, dtype=np.float32)),
                np.asarray(validation_targets, dtype=np.float32),
            )

        # Materialize the reduced genotype matrix (RAM or GPU via CuPy).
        cached = reduced_genotypes.try_materialize_gpu()
        if not cached:
            cached = reduced_genotypes.try_materialize()
        if cached:
            # After materialization, reduced_genotypes no longer needs raw.
            reduced_genotypes.release_raw_storage()
        else:
            log("keeping reduced genotype matrix streaming (no RAM/GPU cache)  mem=" + mem())
        del raw_genotype_matrix, standardized_genotypes, active_genotypes
        import gc
        gc.collect()
        log(f"memory freed after materialization  mem={mem()}")
        log(
            f"starting variational EM  max_iterations={self.config.max_outer_iterations}  "
            f"reduced_matrix={reduced_genotypes.shape}  in_memory={cached}  "
            f"on_gpu={reduced_genotypes._cupy_cache is not None}  "
            f"mem={mem()}"
        )
        fit_result = fit_variational_em(
            genotypes=reduced_genotypes,
            covariates=prepared_arrays.covariates,
            targets=prepared_arrays.targets,
            records=active_records,
            tie_map=reduced_tie_map,
            config=self.config,
            validation_data=reduced_validation,
        )
        log(f"variational EM converged in {len(fit_result.objective_history)} iterations  final_obj={fit_result.objective_history[-1]:.4f}  mem={mem()}")

        log("expanding coefficients from reduced to full space...")
        tie_group_weights = _tie_group_export_weights(
            tie_map=reduced_tie_map,
            fit_result=fit_result,
        )
        active_coefficients = reduced_tie_map.expand_coefficients(
            fit_result.beta_reduced,
            group_weights=tie_group_weights,
        )
        full_coefficients = np.zeros(len(normalized_records), dtype=np.float32)
        full_coefficients[active_variant_indices] = active_coefficients
        nonzero_coefficient_indices, nonzero_coefficients = _nonzero_coefficient_cache(full_coefficients)
        nonzero_means = np.asarray(prepared_arrays.means[nonzero_coefficient_indices], dtype=np.float32)
        nonzero_scales = np.asarray(prepared_arrays.scales[nonzero_coefficient_indices], dtype=np.float32)
        nonzero_count = int(np.count_nonzero(full_coefficients))
        log(f"coefficients: {nonzero_count} non-zero out of {len(normalized_records)} total")

        self.state = FittedState(
            variant_records=normalized_records,
            active_variant_indices=active_variant_indices,
            preprocessor=preprocessor,
            tie_map=original_space_tie_map,
            fit_result=fit_result,
            full_coefficients=full_coefficients,
            nonzero_coefficient_indices=nonzero_coefficient_indices,
            nonzero_coefficients=nonzero_coefficients,
            nonzero_means=nonzero_means,
            nonzero_scales=nonzero_scales,
        )
        log(f"=== MODEL FIT DONE ===  mem={mem()}")
        return self

    def decision_function(self, genotypes: RawGenotypeMatrix | np.ndarray, covariates: np.ndarray) -> np.ndarray:
        """Compute the raw linear predictor (before sigmoid for binary traits).

        For each individual: score = sum_j(beta_j * standardized_genotype_j) + covariates @ alpha.
        Only reads variants with non-zero coefficients (typically <1% of all variants),
        making prediction fast even on huge genotype files.
        """
        genotype_component, covariate_component = self.decision_components(genotypes, covariates)
        return genotype_component + covariate_component

    def decision_components(
        self,
        genotypes: RawGenotypeMatrix | np.ndarray,
        covariates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the separate genetic and covariate contributions to the predictor."""
        fitted_state = self._require_state()
        covariate_matrix = self._with_intercept(np.asarray(covariates, dtype=np.float32))
        covariate_component = np.asarray(covariate_matrix @ fitted_state.fit_result.alpha, dtype=np.float32)

        # Only read variants with non-zero coefficients (skip 99%+ of the file)
        nonzero_indices = fitted_state.nonzero_coefficient_indices
        nonzero_coefficients = fitted_state.nonzero_coefficients
        log(f"decision_function: {genotypes.shape[0]} samples, {len(nonzero_indices)} non-zero coefficients (of {len(fitted_state.full_coefficients)} total)  mem={mem()}")

        if len(nonzero_indices) == 0:
            return np.zeros(genotypes.shape[0], dtype=np.float32), covariate_component

        raw_genotypes = as_raw_genotype_matrix(genotypes)
        genotype_component = _raw_standardized_subset_matvec(
            raw_genotypes=raw_genotypes,
            variant_indices=nonzero_indices,
            means=fitted_state.nonzero_means,
            scales=fitted_state.nonzero_scales,
            coefficients=nonzero_coefficients,
            batch_size=auto_batch_size(raw_genotypes.shape[0]),
        )
        log(f"  decision_function done  mem={mem()}")
        return np.asarray(genotype_component, dtype=np.float32), covariate_component

    def predict_proba(self, genotypes: RawGenotypeMatrix | np.ndarray, covariates: np.ndarray) -> np.ndarray:
        """For binary traits: convert the linear predictor to probabilities via sigmoid.

        Returns an (n_samples, 2) array with columns [P(control), P(case)].
        """
        if self.config.trait_type != TraitType.BINARY:
            raise ValueError("predict_proba is only available for binary traits.")
        linear_predictor = self.decision_function(genotypes, covariates)
        positive_probability = np.asarray(stable_sigmoid(linear_predictor), dtype=np.float32)
        return np.column_stack([1.0 - positive_probability, positive_probability])

    def predict(self, genotypes: RawGenotypeMatrix | np.ndarray, covariates: np.ndarray) -> np.ndarray:
        linear_predictor = self.decision_function(genotypes, covariates)
        if self.config.trait_type == TraitType.BINARY:
            return (np.asarray(stable_sigmoid(linear_predictor), dtype=np.float32) >= 0.5).astype(np.int32)
        return linear_predictor

    def export(self, path: str | Path) -> None:
        fitted_state = self._require_state()
        artifact = ModelArtifact(
            config=self.config,
            records=fitted_state.variant_records,
            means=fitted_state.preprocessor.means,
            scales=fitted_state.preprocessor.scales,
            alpha=fitted_state.fit_result.alpha,
            beta_reduced=fitted_state.fit_result.beta_reduced,
            beta_full=fitted_state.full_coefficients,
            beta_variance=fitted_state.fit_result.beta_variance,
            tie_map=fitted_state.tie_map,
            sigma_e2=fitted_state.fit_result.sigma_error2,
            prior_scales=fitted_state.fit_result.prior_scales,
            global_scale=fitted_state.fit_result.global_scale,
            class_tpb_shape_a=fitted_state.fit_result.class_tpb_shape_a,
            class_tpb_shape_b=fitted_state.fit_result.class_tpb_shape_b,
            scale_model_coefficients=fitted_state.fit_result.scale_model_coefficients,
            scale_model_feature_names=fitted_state.fit_result.scale_model_feature_names,
            objective_history=fitted_state.fit_result.objective_history,
            validation_history=fitted_state.fit_result.validation_history,
        )
        save_artifact(path, artifact)

    @classmethod
    def load(cls, path: str | Path) -> BayesianPGS:
        artifact = load_artifact(path)
        loaded_model = cls(config=artifact.config)
        nonzero_coefficient_indices, nonzero_coefficients = _nonzero_coefficient_cache(artifact.beta_full)
        nonzero_means = np.asarray(artifact.means[nonzero_coefficient_indices], dtype=np.float32)
        nonzero_scales = np.asarray(artifact.scales[nonzero_coefficient_indices], dtype=np.float32)
        loaded_model.state = FittedState(
            variant_records=artifact.records,
            active_variant_indices=np.where(artifact.tie_map.original_to_reduced >= 0)[0].astype(np.int32),
            preprocessor=Preprocessor(means=artifact.means, scales=artifact.scales),
            tie_map=artifact.tie_map,
            fit_result=VariationalFitResult(
                alpha=artifact.alpha,
                beta_reduced=artifact.beta_reduced,
                beta_variance=artifact.beta_variance,
                prior_scales=artifact.prior_scales,
                global_scale=artifact.global_scale,
                class_tpb_shape_a=artifact.class_tpb_shape_a,
                class_tpb_shape_b=artifact.class_tpb_shape_b,
                scale_model_coefficients=artifact.scale_model_coefficients,
                scale_model_feature_names=artifact.scale_model_feature_names,
                sigma_error2=artifact.sigma_e2,
                objective_history=artifact.objective_history,
                validation_history=artifact.validation_history,
                member_prior_variances=artifact.prior_scales,
            ),
            full_coefficients=artifact.beta_full,
            nonzero_coefficient_indices=nonzero_coefficient_indices,
            nonzero_coefficients=nonzero_coefficients,
            nonzero_means=nonzero_means,
            nonzero_scales=nonzero_scales,
        )
        return loaded_model

    def coefficient_table(self) -> list[dict[str, object]]:
        fitted_state = self._require_state()
        return [
            {
                "variant_id": variant_record.variant_id,
                "variant_class": variant_record.variant_class.value,
                "beta": float(coefficient),
            }
            for variant_record, coefficient in zip(
                fitted_state.variant_records,
                fitted_state.full_coefficients,
                strict=True,
            )
        ]

    def _require_state(self) -> FittedState:
        if self.state is None:
            raise ValueError("Model is not fitted.")
        return self.state

    @staticmethod
    def _with_intercept(covariates: np.ndarray) -> np.ndarray:
        covariate_matrix = np.asarray(covariates, dtype=np.float32)
        if covariate_matrix.ndim != 2:
            raise ValueError("covariates must be 2D.")
        output = np.empty((covariate_matrix.shape[0], covariate_matrix.shape[1] + 1), dtype=np.float32)
        output[:, 0] = 1.0
        output[:, 1:] = covariate_matrix
        return output


def _nonzero_coefficient_cache(coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coefficient_array = np.asarray(coefficients, dtype=np.float32)
    nonzero_indices = np.flatnonzero(np.abs(coefficient_array) > 0.0).astype(np.int32)
    return nonzero_indices, np.asarray(coefficient_array[nonzero_indices], dtype=np.float32)


# Compute genotypes @ coefficients for a subset of variants, standardizing
# on the fly.  This avoids materializing the full standardized genotype
# matrix — instead we read raw genotypes in batches, standardize each batch,
# multiply by the corresponding coefficients, and accumulate the result.
# This is the inner loop of prediction and is I/O-bound for PLINK files.
def _raw_standardized_subset_matvec(
    raw_genotypes: RawGenotypeMatrix,
    variant_indices: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    coefficients: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if variant_indices.shape[0] == 0:
        return np.zeros(raw_genotypes.shape[0], dtype=np.float32)
    result = np.zeros(raw_genotypes.shape[0], dtype=np.float32)
    offset = 0
    for batch in raw_genotypes.iter_column_batches(variant_indices, batch_size=batch_size):
        batch_width = batch.variant_indices.shape[0]
        batch_slice = slice(offset, offset + batch_width)
        standardized_batch = _standardize_batch(batch.values, means[batch_slice], scales[batch_slice])
        result += standardized_batch @ coefficients[batch_slice]
        offset += batch_width
    return result


# The tie map was built in "active variant space" (indices 0..n_active-1).
# This function translates those indices back to the original variant numbering
# so the exported model can be applied to new genotype files that use the
# original variant ordering.
def _project_tie_map_to_original_space(
    reduced_tie_map: TieMap,
    active_variant_indices: np.ndarray,
    original_variant_count: int,
) -> TieMap:
    kept_indices = active_variant_indices[reduced_tie_map.kept_indices]
    original_to_reduced = np.full(original_variant_count, -1, dtype=np.int32)
    original_to_reduced[active_variant_indices] = reduced_tie_map.original_to_reduced
    original_groups: list[TieGroup] = []
    for tie_group in reduced_tie_map.reduced_to_group:
        original_groups.append(
            TieGroup(
                representative_index=int(active_variant_indices[tie_group.representative_index]),
                member_indices=active_variant_indices[tie_group.member_indices].astype(np.int32),
                signs=tie_group.signs.astype(np.float32),
            )
        )
    return TieMap(
        kept_indices=kept_indices.astype(np.int32),
        original_to_reduced=original_to_reduced,
        reduced_to_group=original_groups,
    )


# When expanding a single group effect back to individual members, how much
# weight does each member get?  We split proportional to each member's prior
# variance — variants the model trusted more a priori get a larger share of
# the group's fitted coefficient.  This is fairer than equal splitting when
# group members differ in type (e.g. a deletion and a duplication tied
# together).
def _tie_group_export_weights(
    tie_map: TieMap,
    fit_result: VariationalFitResult,
) -> list[np.ndarray]:
    member_prior_variances = np.asarray(fit_result.member_prior_variances, dtype=np.float32)
    group_weights: list[np.ndarray] = []
    for tie_group in tie_map.reduced_to_group:
        member_variances = np.asarray(member_prior_variances[tie_group.member_indices], dtype=np.float32)
        normalized_weights = member_variances / np.maximum(np.sum(member_variances), 1e-12)
        group_weights.append(normalized_weights.astype(np.float32))
    return group_weights


def _training_records(
    raw_genotypes: RawGenotypeMatrix,
    records: Sequence[VariantRecord],
    config: ModelConfig,
) -> list[VariantRecord]:
    training_supports = [record.training_support for record in records]
    unresolved_variant_indices = [
        variant_index
        for variant_index, record in enumerate(records)
        if record.variant_class in STRUCTURAL_VARIANT_CLASSES and record.training_support is None
    ]
    log(f"  training records: {len(records)} total, {len(unresolved_variant_indices)} SVs need support counts  mem={mem()}")
    if unresolved_variant_indices:
        unresolved_supports = np.zeros(len(unresolved_variant_indices), dtype=np.int32)
        offset = 0
        for batch in raw_genotypes.iter_column_batches(unresolved_variant_indices, batch_size=auto_batch_size(raw_genotypes.shape[0])):
            # Vectorized support count: count non-zero non-NaN values per column
            batch_jax = jnp.asarray(batch.values)
            valid = ~jnp.isnan(batch_jax)
            counts = jnp.sum((jnp.abs(batch_jax) > 0.5) & valid, axis=0)
            batch_len = len(batch.variant_indices)
            unresolved_supports[offset:offset + batch_len] = np.asarray(counts, dtype=np.int32)
            offset += batch_len
        for i, variant_index in enumerate(unresolved_variant_indices):
            training_supports[variant_index] = int(unresolved_supports[i])

    training_records: list[VariantRecord] = []
    for variant_index, record in enumerate(records):
        training_records.append(
            VariantRecord(
                variant_id=record.variant_id,
                variant_class=record.variant_class,
                chromosome=record.chromosome,
                position=record.position,
                length=record.length,
                allele_frequency=record.allele_frequency,
                quality=record.quality,
                training_support=training_supports[variant_index],
                is_repeat=record.is_repeat,
                is_copy_number=record.is_copy_number,
                prior_continuous_features=dict(record.prior_continuous_features),
                prior_class_members=record.prior_class_members,
                prior_class_membership=record.prior_class_membership,
            )
        )
    return training_records


def _training_records_from_stats(
    records: Sequence[VariantRecord],
    variant_stats: VariantStatistics,
) -> list[VariantRecord]:
    """Build training records using pre-computed support counts (no data pass)."""
    training_records: list[VariantRecord] = []
    for variant_index, record in enumerate(records):
        support = record.training_support
        if support is None and record.variant_class in STRUCTURAL_VARIANT_CLASSES:
            support = int(variant_stats.support_counts[variant_index])
        training_records.append(
            VariantRecord(
                variant_id=record.variant_id,
                variant_class=record.variant_class,
                chromosome=record.chromosome,
                position=record.position,
                length=record.length,
                allele_frequency=record.allele_frequency,
                quality=record.quality,
                training_support=support,
                is_repeat=record.is_repeat,
                is_copy_number=record.is_copy_number,
                prior_continuous_features=dict(record.prior_continuous_features),
                prior_class_members=record.prior_class_members,
                prior_class_membership=record.prior_class_membership,
            )
        )
    log(f"  training records from stats: {len(training_records)} records [NO DATA PASS]")
    return training_records


def _compute_support_counts(
    raw_genotypes: RawGenotypeMatrix,
    records: Sequence[VariantRecord],
    config: ModelConfig,
) -> np.ndarray:
    """Count how many individuals carry each structural variant.

    "Support" = number of samples with a non-zero dosage (|dosage| > 0.5).
    Only computed for SVs that don't already have a support count from
    metadata.  SNVs/indels skip this step entirely.  Low-support SVs will
    be filtered out before model fitting to avoid noisy estimates.
    """
    support_counts = np.zeros(len(records), dtype=np.int32)
    unresolved_structural_variant_indices: list[int] = []
    for variant_index, record in enumerate(records):
        if record.variant_class not in STRUCTURAL_VARIANT_CLASSES:
            continue
        if record.training_support is not None:
            support_counts[variant_index] = int(record.training_support)
            continue
        unresolved_structural_variant_indices.append(variant_index)

    if not unresolved_structural_variant_indices:
        return support_counts

    for batch in raw_genotypes.iter_column_batches(
        unresolved_structural_variant_indices,
        batch_size=auto_batch_size(raw_genotypes.shape[0]),
    ):
        batch_jax = jnp.asarray(batch.values)
        counts = jnp.sum(jnp.abs(jnp.where(jnp.isnan(batch_jax), 0.0, batch_jax)) > 0.5, axis=0)
        support_counts[batch.variant_indices] = np.asarray(counts, dtype=np.int32)
    return support_counts



def _identity_tie_map(variant_count: int) -> TieMap:
    kept_indices = np.arange(variant_count, dtype=np.int32)
    return TieMap(
        kept_indices=kept_indices,
        original_to_reduced=kept_indices.copy(),
        reduced_to_group=[
            TieGroup(
                representative_index=int(variant_index),
                member_indices=np.asarray([variant_index], dtype=np.int32),
                signs=np.asarray([1.0], dtype=np.float32),
            )
            for variant_index in range(variant_count)
        ],
    )
