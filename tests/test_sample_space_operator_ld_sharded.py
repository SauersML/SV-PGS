"""Sample-space operator on a column-sharded GPU cache with an LD partition.

The AoU runner attaches an LD-block partition (use_ld_blocks=True) and a
multi-GPU host shards the resident genotype cache by columns once the int8
matrix no longer fits one device. The LD branch of
``_apply_sample_space_operator_gpu`` sliced ``_cupy_cache[:, block]``, which a
``_CupyShardedStandardizedCache`` does not support, so the first CG
application raised TypeError. The operator must instead equal the dense
``diag(noise) v + X diag(tau^2) X^T v``.
"""
from __future__ import annotations

import numpy as np

import sv_pgs.genotype as genotype_module
from sv_pgs.genotype import as_raw_genotype_matrix
from sv_pgs.ld_block_partition import LdBlockPartition
from sv_pgs.mixture_inference import _apply_sample_space_operator_gpu


class _FakeCupy:
    float32 = np.float32
    float64 = np.float64
    int64 = np.int64

    @staticmethod
    def asarray(array, dtype=None):
        return np.asarray(array, dtype=dtype)

    @staticmethod
    def zeros(shape, dtype=None):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def empty(shape, dtype=None, order=None):
        return np.empty(shape, dtype=dtype, order="C" if order is None else order)

    @staticmethod
    def concatenate(arrays, axis=0):
        return np.concatenate(arrays, axis=axis)

    @staticmethod
    def any(array):
        return np.any(array)


def test_ld_partitioned_operator_on_sharded_cache_matches_dense_reference() -> None:
    rng = np.random.default_rng(11)
    raw_matrix = rng.integers(0, 3, size=(6, 4)).astype(np.float32)
    means = raw_matrix.mean(axis=0)
    scales = np.maximum(raw_matrix.std(axis=0), 0.5).astype(np.float32)
    standardized = as_raw_genotype_matrix(raw_matrix).standardized(means, scales)
    dense_matrix = standardized.materialize().astype(np.float64, copy=False)
    fake_cupy = _FakeCupy()
    standardized._cupy_cache = genotype_module._CupyShardedStandardizedCache(
        (
            genotype_module._CupyDeviceCacheShard(
                device_id=0,
                column_start=0,
                cache=dense_matrix[:, :2].astype(np.float32, copy=False),
            ),
            genotype_module._CupyDeviceCacheShard(
                device_id=1,
                column_start=2,
                cache=dense_matrix[:, 2:].astype(np.float32, copy=False),
            ),
        ),
        cupy=fake_cupy,
    )
    standardized._dense_cache = None
    # Blocks straddle the shard boundary so a per-block slice would need both devices.
    standardized._ld_block_partition = LdBlockPartition(
        block_ids=np.array([0, 1, 0, 1], dtype=np.int64),
        partition={0: np.array([0, 2], dtype=np.int64), 1: np.array([1, 3], dtype=np.int64)},
    )
    prior_variances = np.array([0.5, 0.25, 1.5, 0.75], dtype=np.float64)
    diagonal_noise = np.linspace(0.5, 1.0, 6)
    right_hand_side = rng.standard_normal((6, 2))

    applied = _apply_sample_space_operator_gpu(
        genotype_matrix=standardized,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        matrix_gpu=right_hand_side,
        batch_size=2,
        cp=fake_cupy,
        dtype=np.float64,
    )

    expected = diagonal_noise[:, None] * right_hand_side + dense_matrix @ (
        prior_variances[:, None] * (dense_matrix.T @ right_hand_side)
    )
    np.testing.assert_allclose(np.asarray(applied, dtype=np.float64), expected, rtol=1e-6, atol=1e-6)
