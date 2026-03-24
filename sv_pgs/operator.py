"""Tiled single-device JAX genotype operator for memory-efficient matvecs.

Genotypes are stored variant-major in tiles of shape (num_tiles, tile_size, n).
The scan-based kernels stream through tiles without materializing X^T X or
the full dense product. This is the right layout for p >> n biobank workloads.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GenotypeOperator:
    genotype_tiles: jnp.ndarray
    tile_mask: jnp.ndarray
    variant_count: int
    tile_size: int

    @classmethod
    def from_numpy(cls, genotypes: np.ndarray, tile_size: int = 256) -> GenotypeOperator:
        variant_major = np.asarray(np.transpose(genotypes), dtype=np.float32)
        variant_count, sample_count = variant_major.shape
        tile_count = int(np.ceil(variant_count / tile_size))
        padded_count = tile_count * tile_size

        padded = np.zeros((padded_count, sample_count), dtype=np.float32)
        padded[:variant_count] = variant_major

        mask = np.zeros((tile_count, tile_size), dtype=np.float32)
        mask.reshape(-1)[:variant_count] = 1.0

        return cls(
            genotype_tiles=jnp.asarray(
                padded.reshape(tile_count, tile_size, sample_count),
            ),
            tile_mask=jnp.asarray(mask),
            variant_count=variant_count,
            tile_size=tile_size,
        )

    def pad_vector(self, vector: jnp.ndarray) -> jnp.ndarray:
        padded = jnp.zeros(
            self.genotype_tiles.shape[0] * self.tile_size,
            dtype=vector.dtype,
        )
        return padded.at[: self.variant_count].set(vector)

    def truncate_vector(self, vector: jnp.ndarray) -> jnp.ndarray:
        return vector[: self.variant_count]

    def tree_flatten(self):
        children = (self.genotype_tiles, self.tile_mask)
        auxiliary = (self.variant_count, self.tile_size)
        return children, auxiliary

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        variant_count, tile_size = auxiliary
        return cls(
            genotype_tiles=children[0],
            tile_mask=children[1],
            variant_count=variant_count,
            tile_size=tile_size,
        )


def _scan_forward(
    genotype_tiles: jnp.ndarray,
    coefficient_tiles: jnp.ndarray,
    tile_mask: jnp.ndarray,
) -> jnp.ndarray:
    """X @ beta via tiled scan. Returns (n,)."""

    def body(
        accumulated: jnp.ndarray,
        scan_inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, None]:
        tile, coeff_tile, mask_tile = scan_inputs
        contribution = jnp.sum(
            tile * (coeff_tile * mask_tile)[:, None], axis=0,
        )
        return accumulated + contribution, None

    initial = jnp.zeros(genotype_tiles.shape[-1], dtype=jnp.float32)
    return lax.scan(
        body, initial, (genotype_tiles, coefficient_tiles, tile_mask),
    )[0]


def _scan_adjoint(
    genotype_tiles: jnp.ndarray,
    sample_vector: jnp.ndarray,
    tile_mask: jnp.ndarray,
) -> jnp.ndarray:
    """X^T @ u via tiled scan. Returns (padded_p,)."""

    def body(
        unused_carry: None,
        scan_inputs: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[None, jnp.ndarray]:
        del unused_carry
        tile, mask_tile = scan_inputs
        projection = jnp.sum(tile * sample_vector[None, :], axis=1)
        return None, projection * mask_tile

    return lax.scan(
        body, None, (genotype_tiles, tile_mask),
    )[1].reshape(-1)


@jax.jit
def matvec(
    operator: GenotypeOperator, coefficients: jnp.ndarray,
) -> jnp.ndarray:
    """X @ coefficients: (p,) -> (n,). Tiled scan, O(tile_size * n) memory."""
    padded = operator.pad_vector(coefficients)
    tiles = padded.reshape(
        operator.genotype_tiles.shape[0], operator.tile_size,
    )
    return _scan_forward(operator.genotype_tiles, tiles, operator.tile_mask)


@jax.jit
def rmatvec(
    operator: GenotypeOperator, sample_vector: jnp.ndarray,
) -> jnp.ndarray:
    """X^T @ sample_vector: (n,) -> (p,). Tiled scan, O(tile_size * n) memory."""
    padded = _scan_adjoint(
        operator.genotype_tiles, sample_vector, operator.tile_mask,
    )
    return operator.truncate_vector(padded)


@jax.jit
def weighted_rmatvec(
    operator: GenotypeOperator,
    sample_weights: jnp.ndarray,
    response_vector: jnp.ndarray,
) -> jnp.ndarray:
    """X^T @ (w * y) without host/device transfers."""
    return rmatvec(operator, sample_weights * response_vector)


@jax.jit
def apply_precision_matrix(
    operator: GenotypeOperator,
    sample_weights: jnp.ndarray,
    prior_precision: jnp.ndarray,
    coefficient_vector: jnp.ndarray,
) -> jnp.ndarray:
    """Apply X^T W X + D to coefficients inside one compiled JAX kernel."""
    sample_projection = matvec(operator, coefficient_vector)
    return rmatvec(operator, sample_weights * sample_projection) + prior_precision * coefficient_vector
