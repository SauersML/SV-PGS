from __future__ import annotations

from dataclasses import dataclass, replace

from sv_pgs._jax import t4_fast_math_enabled
from sv_pgs.config import ModelConfig
from sv_pgs.genotype import RawGenotypeMatrix, _gpu_materialization_budget_bytes, _try_import_cupy

# Algorithmic limits — not GPU-memory-dependent.
# The exact solver limit caps dense Cholesky factorizations on GPU to avoid
# excessive O(p^3) cost. The preconditioner rank bounds the Nyström approximation.
GPU_FINAL_REFINEMENT_VARIANT_MULTIPLIER = 2
T4_GPU_PRECONDITIONER_RANK_LIMIT = 384


@dataclass(frozen=True, slots=True)
class RuntimeTrainingPolicy:
    tuned_config: ModelConfig
    gpu_budget_bytes: int | None
    cacheable_dense_variants: int | None


def runtime_training_policy_for_fit(
    config: ModelConfig,
    genotype_matrix: RawGenotypeMatrix,
) -> RuntimeTrainingPolicy:
    cupy = _try_import_cupy()
    if cupy is None:
        return RuntimeTrainingPolicy(
            tuned_config=config,
            gpu_budget_bytes=None,
            cacheable_dense_variants=None,
        )
    sample_count = int(genotype_matrix.shape[0])
    if sample_count < 1:
        return RuntimeTrainingPolicy(
            tuned_config=config,
            gpu_budget_bytes=None,
            cacheable_dense_variants=None,
        )
    gpu_budget_bytes = _gpu_materialization_budget_bytes(cupy)
    cacheable_dense_variants = max(int(gpu_budget_bytes // max(sample_count * 4, 1)), 1)
    preconditioner_rank_limit = (
        T4_GPU_PRECONDITIONER_RANK_LIMIT
        if t4_fast_math_enabled()
        else cacheable_dense_variants
    )
    tuned_exact_solver_limit = min(
        int(config.exact_solver_matrix_limit),
        max(int(cacheable_dense_variants * 0.9), 1),
    )
    max_gpu_preconditioner_rank = max(1, min(cacheable_dense_variants, preconditioner_rank_limit))
    tuned_preconditioner_rank = min(
        int(config.sample_space_preconditioner_rank),
        max_gpu_preconditioner_rank,
    )
    # Use up to 75% of GPU budget for stochastic blocks — larger blocks mean
    # fewer blocks per epoch, fewer preconditioner builds, better convergence.
    # On GPU we want stochastic blocks large enough to amortize upload,
    # preconditioner, and CG startup costs. The static default (8,192) is too
    # conservative on T4-class cards once the block solver is iterative.
    tuned_stochastic_batch_size = max(
        max(int(config.stochastic_variant_batch_size), max(int(cacheable_dense_variants * 0.75), 256)),
        256,
    )
    tuned_final_posterior_refinement = (
        bool(config.final_posterior_refinement)
        and int(genotype_matrix.shape[1])
        <= max(
            int(cacheable_dense_variants) * GPU_FINAL_REFINEMENT_VARIANT_MULTIPLIER,
            tuned_exact_solver_limit,
        )
    )
    tuned_config = replace(
        config,
        exact_solver_matrix_limit=tuned_exact_solver_limit,
        sample_space_preconditioner_rank=tuned_preconditioner_rank,
        stochastic_variant_batch_size=max(tuned_stochastic_batch_size, 1),
        final_posterior_refinement=tuned_final_posterior_refinement,
    )
    return RuntimeTrainingPolicy(
        tuned_config=tuned_config,
        gpu_budget_bytes=gpu_budget_bytes,
        cacheable_dense_variants=cacheable_dense_variants,
    )


def runtime_training_policy_summary(policy: RuntimeTrainingPolicy, original_config: ModelConfig) -> str | None:
    if policy.gpu_budget_bytes is None or policy.cacheable_dense_variants is None:
        return None
    tuned_config = policy.tuned_config
    if tuned_config == original_config:
        return (
            "GPU runtime profile active: "
            + f"gpu_budget={policy.gpu_budget_bytes / 1e9:.1f} GB "
            + f"cacheable_dense_variants~{policy.cacheable_dense_variants} "
            + f"t4_profile={'on' if t4_fast_math_enabled() else 'off'} "
            + "(user config already fits GPU profile)"
        )
    return (
        "GPU runtime profile active: "
        + f"gpu_budget={policy.gpu_budget_bytes / 1e9:.1f} GB "
        + f"cacheable_dense_variants~{policy.cacheable_dense_variants} "
        + f"t4_profile={'on' if t4_fast_math_enabled() else 'off'} "
        + f"exact_solver_matrix_limit={original_config.exact_solver_matrix_limit}->{tuned_config.exact_solver_matrix_limit} "
        + f"sample_space_preconditioner_rank={original_config.sample_space_preconditioner_rank}->{tuned_config.sample_space_preconditioner_rank} "
        + f"stochastic_variant_batch_size={original_config.stochastic_variant_batch_size}->{tuned_config.stochastic_variant_batch_size} "
        + f"final_posterior_refinement={original_config.final_posterior_refinement}->{tuned_config.final_posterior_refinement}"
    )
