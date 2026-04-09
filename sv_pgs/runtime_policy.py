from __future__ import annotations

from dataclasses import dataclass, replace

from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.genotype import RawGenotypeMatrix, _gpu_materialization_budget_bytes, _try_import_cupy

# Algorithmic limits — not GPU-memory-dependent.
# The exact solver limit caps dense Cholesky factorizations on GPU to avoid
# excessive O(p^3) cost. The preconditioner rank bounds the Nyström approximation.
GPU_FINAL_REFINEMENT_VARIANT_MULTIPLIER = 2
GPU_PRECONDITIONER_RANK_FLOOR = 128
GPU_PRECONDITIONER_RANK_CEILING = 512
GPU_PRECONDITIONER_RANK_FRACTION = 0.04


@dataclass(frozen=True, slots=True)
class RuntimeTrainingPolicy:
    tuned_config: ModelConfig
    gpu_budget_bytes: int | None
    cacheable_dense_variants: int | None


def _recommended_gpu_preconditioner_rank(cacheable_dense_variants: int) -> int:
    if cacheable_dense_variants < 1:
        return GPU_PRECONDITIONER_RANK_FLOOR
    budget_scaled_rank = int(round(float(cacheable_dense_variants) * GPU_PRECONDITIONER_RANK_FRACTION))
    return max(
        GPU_PRECONDITIONER_RANK_FLOOR,
        min(
            GPU_PRECONDITIONER_RANK_CEILING,
            int(cacheable_dense_variants),
            budget_scaled_rank,
        ),
    )


def _recommended_gpu_stochastic_batch_size(
    *,
    cacheable_dense_variants: int,
    exact_solver_matrix_limit: int,
    trait_type: TraitType,
) -> int:
    if cacheable_dense_variants < 1:
        return 256
    budget_target = max(int(cacheable_dense_variants * 0.85), 256)
    if trait_type == TraitType.BINARY:
        # On smaller GPUs, very large binary blocks push the sample-space CG
        # system into a hard regime where iterative refinement can fail before
        # the Newton update is useful. Keep binary blocks within a modest
        # multiple of the exact-solver limit so the stochastic path stays in the
        # well-conditioned regime for the available GPU memory.
        budget_target = min(
            budget_target,
            max(int(exact_solver_matrix_limit) * 2, 256),
        )
    return max(budget_target, 256)


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
    tuned_exact_solver_limit = min(
        int(config.exact_solver_matrix_limit),
        max(int(cacheable_dense_variants * 0.9), 1),
    )
    max_gpu_preconditioner_rank = max(1, int(cacheable_dense_variants))
    recommended_preconditioner_rank = _recommended_gpu_preconditioner_rank(cacheable_dense_variants)
    tuned_preconditioner_rank = min(
        max(int(config.sample_space_preconditioner_rank), recommended_preconditioner_rank),
        max_gpu_preconditioner_rank,
    )
    # Use up to 85% of GPU budget for stochastic blocks — larger blocks mean
    # fewer blocks per epoch, fewer preconditioner builds, better convergence.
    # On GPU we want stochastic blocks large enough to amortize upload,
    # preconditioner, and CG startup costs. The static default (8,192) is too
    # conservative on modern GPUs once the block solver is iterative.
    tuned_stochastic_batch_size = max(
        _recommended_gpu_stochastic_batch_size(
            cacheable_dense_variants=cacheable_dense_variants,
            exact_solver_matrix_limit=tuned_exact_solver_limit,
            trait_type=config.trait_type,
        ),
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
            + "gpu_profile=budget-driven "
            + "(user config already fits GPU profile)"
        )
    return (
        "GPU runtime profile active: "
        + f"gpu_budget={policy.gpu_budget_bytes / 1e9:.1f} GB "
        + f"cacheable_dense_variants~{policy.cacheable_dense_variants} "
        + "gpu_profile=budget-driven "
        + f"exact_solver_matrix_limit={original_config.exact_solver_matrix_limit}->{tuned_config.exact_solver_matrix_limit} "
        + f"sample_space_preconditioner_rank={original_config.sample_space_preconditioner_rank}->{tuned_config.sample_space_preconditioner_rank} "
        + f"stochastic_variant_batch_size={original_config.stochastic_variant_batch_size}->{tuned_config.stochastic_variant_batch_size} "
        + f"final_posterior_refinement={original_config.final_posterior_refinement}->{tuned_config.final_posterior_refinement}"
    )
