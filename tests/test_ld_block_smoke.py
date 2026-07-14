"""Phase 4 LD-block / N-GPU wiring smoke test.

Verifies that flipping ``config.use_ld_blocks`` on routes the sample-space
matvec through the per-block decomposition while still landing within fp32
noise of the legacy single-monolithic-matmul path. Also pins that the
GPU scheduler's CPU fallback (``device_ids=(-1,)``) is exercised when no
CUDA devices are visible (which is the case in CI).
"""

from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.genotype import DenseRawGenotypeMatrix, StandardizedGenotypeMatrix
from sv_pgs.gpu_scheduler import GPUScheduler
from sv_pgs.ld_block_partition import LdBlockPartition, build_ld_block_partition
from sv_pgs.ld_blocks import load_ld_blocks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_synthetic_standardized_matrix(
    n_samples: int = 512,
    n_variants: int = 1000,
    seed: int = 0,
) -> StandardizedGenotypeMatrix:
    rng = np.random.default_rng(seed)
    # Synthetic dosages in {0, 1, 2}; impute by sampling uniformly.
    dosages = rng.integers(0, 3, size=(n_samples, n_variants)).astype(np.float32)
    means = dosages.mean(axis=0).astype(np.float32)
    scales = dosages.std(axis=0).astype(np.float32)
    scales[scales < 1e-3] = 1.0
    # Cap support counts so the hybrid sparse backend is not triggered.
    support_counts = np.full(n_variants, n_samples, dtype=np.int32)
    raw = DenseRawGenotypeMatrix(matrix=dosages)
    return StandardizedGenotypeMatrix(
        raw=raw,
        means=means,
        scales=scales,
        variant_indices=np.arange(n_variants, dtype=np.int32),
        support_counts=support_counts,
        _enable_hybrid_backend=False,
    )


def _build_real_partition_for_synthetic(
    n_variants: int = 1000,
    blocks_per_chrom: int = 5,
) -> LdBlockPartition:
    """Build a synthetic but Berisa-Pickrell-consistent partition.

    Pick the first ``blocks_per_chrom`` blocks on chr1, place
    ``n_variants/blocks_per_chrom`` variants inside each, and let the real
    ``assign_ld_blocks`` produce the per-variant ids — exercising the actual
    bisect-into-block table.
    """
    blocks = load_ld_blocks(build="hg38", ancestry="EUR")
    chr1 = blocks[blocks[:, 0] == 1][:blocks_per_chrom]
    assert chr1.shape[0] == blocks_per_chrom

    per_block = n_variants // blocks_per_chrom
    assert per_block * blocks_per_chrom == n_variants

    chroms: list[str] = []
    positions: list[int] = []
    for i in range(blocks_per_chrom):
        start, end = int(chr1[i, 1]), int(chr1[i, 2])
        # Pick deterministically spread positions within (start, end).
        block_positions = np.linspace(
            start + 1, end - 1, num=per_block, dtype=np.int64
        ).tolist()
        chroms.extend(["chr1"] * per_block)
        positions.extend(block_positions)

    # Build pseudo-records for build_ld_block_partition.
    class _Rec:
        __slots__ = ("chromosome", "position")

        def __init__(self, chrom: str, pos: int) -> None:
            self.chromosome = chrom
            self.position = pos

    records = [_Rec(c, p) for c, p in zip(chroms, positions)]
    partition = build_ld_block_partition(records, population="EUR", build="hg38")
    assert partition.block_count == blocks_per_chrom, partition.block_count
    return partition


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_gpu_scheduler_cpu_fallback_is_a_noop_context() -> None:
    """``GPUScheduler.detect()`` must yield a CPU-only scheduler in CI."""
    sched = GPUScheduler.detect()
    # In CI (no cupy / no CUDA) we should land on the CPU fallback sentinel.
    if sched.is_cpu_fallback:
        assert sched.device_ids == (-1,)
        assert sched.device_count == 1
        # device_context(-1) must be a no-op contextmanager.
        with sched.device_context(-1):
            pass
        sched.synchronize()
    else:
        # On a GPU box: must enumerate at least one device.
        assert sched.device_count >= 1


def test_build_ld_block_partition_from_records() -> None:
    partition = _build_real_partition_for_synthetic(n_variants=1000, blocks_per_chrom=5)
    assert partition.block_count == 5
    # 1000 variants, 200 per block.
    sizes = [int(idx.shape[0]) for _bid, idx in partition.iter_blocks()]
    assert sorted(sizes) == [200, 200, 200, 200, 200]
    # Hashing surface is stable across rebuilds.
    sig_a = partition.signature_sha256()
    partition_b = _build_real_partition_for_synthetic(n_variants=1000, blocks_per_chrom=5)
    assert partition_b.signature_sha256() == sig_a


def test_solve_restricted_mean_only_block_off_vs_on_match_within_fp32_noise() -> None:
    """Beta and linear-predictor must agree across the legacy and LD-block paths."""
    from sv_pgs.mixture_inference import _solve_restricted_mean_only

    rng = np.random.default_rng(7)
    n_samples = 512
    n_variants = 1000
    matrix_off = _build_synthetic_standardized_matrix(
        n_samples=n_samples, n_variants=n_variants, seed=11
    )
    matrix_on = _build_synthetic_standardized_matrix(
        n_samples=n_samples, n_variants=n_variants, seed=11
    )
    partition = _build_real_partition_for_synthetic(
        n_variants=n_variants, blocks_per_chrom=5
    )
    matrix_on._ld_block_partition = partition
    matrix_on._ld_block_scheduler = GPUScheduler.detect()

    covariates = rng.standard_normal((n_samples, 2), dtype=np.float64).astype(np.float64)
    # Intercept column.
    covariates = np.concatenate([np.ones((n_samples, 1)), covariates], axis=1)
    targets = rng.standard_normal(n_samples).astype(np.float64)
    prior_variances = np.full(n_variants, 0.01, dtype=np.float64)
    diagonal_noise = np.full(n_samples, 1.0, dtype=np.float64)

    common_kwargs = dict(
        covariate_matrix=covariates,
        targets=targets,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        solver_tolerance=1e-8,
        maximum_linear_solver_iterations=512,
        exact_solver_matrix_limit=2048,
        posterior_variance_batch_size=256,
        random_seed=0,
        allow_working_set=False,
    )

    alpha_off, beta_off, _proj_off, pred_off, _q_off = _solve_restricted_mean_only(
        genotype_matrix=matrix_off, **common_kwargs
    )
    alpha_on, beta_on, _proj_on, pred_on, _q_on = _solve_restricted_mean_only(
        genotype_matrix=matrix_on, **common_kwargs
    )

    # The block-decomposed path must reproduce the legacy result within
    # fp32 BLAS noise (the smoke spec asks for atol=1e-5, rtol=1e-4).
    assert np.allclose(beta_off, beta_on, atol=1e-5, rtol=1e-4), (
        float(np.max(np.abs(beta_off - beta_on))),
    )
    assert np.allclose(pred_off, pred_on, atol=1e-5, rtol=1e-4), (
        float(np.max(np.abs(pred_off - pred_on))),
    )
    assert np.allclose(alpha_off, alpha_on, atol=1e-5, rtol=1e-4), (
        float(np.max(np.abs(alpha_off - alpha_on))),
    )


def test_genotype_matvec_with_partition_matches_unblocked() -> None:
    """Sanity: the helper that threads the partition through is loss-free."""
    from sv_pgs.mixture_inference import (
        _genotype_matvec_result_numpy,
        _genotype_transpose_matvec_result_numpy,
    )

    rng = np.random.default_rng(13)
    n_samples, n_variants = 128, 400
    matrix = _build_synthetic_standardized_matrix(
        n_samples=n_samples, n_variants=n_variants, seed=2
    )
    partition = _build_real_partition_for_synthetic(
        n_variants=n_variants, blocks_per_chrom=5
    )
    beta = rng.standard_normal(n_variants).astype(np.float64)
    y = rng.standard_normal(n_samples).astype(np.float64)

    got_legacy = _genotype_matvec_result_numpy(
        matrix, beta, batch_size=128, dtype=np.float64
    )
    matrix._ld_block_partition = partition
    got_block = _genotype_matvec_result_numpy(
        matrix, beta, batch_size=128, dtype=np.float64
    )
    assert np.allclose(got_legacy, got_block, atol=1e-9, rtol=1e-9)

    matrix._ld_block_partition = None
    got_legacy_t = _genotype_transpose_matvec_result_numpy(
        matrix, y, batch_size=128, dtype=np.float64
    )
    matrix._ld_block_partition = partition
    got_block_t = _genotype_transpose_matvec_result_numpy(
        matrix, y, batch_size=128, dtype=np.float64
    )
    assert np.allclose(got_legacy_t, got_block_t, atol=1e-9, rtol=1e-9)


def test_config_use_ld_blocks_defaults_off_and_validates_population() -> None:
    from sv_pgs.config import ModelConfig

    default = ModelConfig()
    assert default.use_ld_blocks is False
    assert default.ld_block_population == "EUR"
    assert default.ld_block_build == "hg38"

    # AFR/EAS/AMR strings validate (even though only EUR is shipped today).
    for pop in ("AFR", "EAS", "AMR", "EUR"):
        ModelConfig(ld_block_population=pop)

    with pytest.raises(ValueError):
        ModelConfig(ld_block_population="ZZZ")
    with pytest.raises(ValueError):
        ModelConfig(ld_block_build="hg18")
    with pytest.raises(ValueError):
        ModelConfig(ld_block_singleton_chunk_size=0)
    with pytest.raises(ValueError):
        ModelConfig(ld_block_pipeline_depth=0)
