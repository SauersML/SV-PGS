"""Matrix-free genotype operator for tiled single-device JAX matvecs."""

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
    def from_numpy(cls, genotypes: np.ndarray, tile_size: int) -> "GenotypeOperator":
        variant_major_genotypes = np.asarray(np.transpose(genotypes), dtype=np.float32)
        variant_count, sample_count = variant_major_genotypes.shape
        tile_count = int(np.ceil(variant_count / tile_size))
        padded_variant_count = tile_count * tile_size

        padded_genotypes = np.zeros((padded_variant_count, sample_count), dtype=np.float32)
        padded_genotypes[:variant_count] = variant_major_genotypes

        tile_mask = np.zeros((tile_count, tile_size), dtype=np.float32)
        tile_mask.reshape(-1)[:variant_count] = 1.0

        return cls(
            genotype_tiles=jnp.asarray(padded_genotypes.reshape(tile_count, tile_size, sample_count)),
            tile_mask=jnp.asarray(tile_mask),
            variant_count=variant_count,
            tile_size=tile_size,
        )

    def pad_vector(self, vector: jnp.ndarray) -> jnp.ndarray:
        padded_vector = jnp.zeros(self.genotype_tiles.shape[0] * self.tile_size, dtype=vector.dtype)
        return padded_vector.at[: self.variant_count].set(vector)

    def tree_flatten(self):
        children = (self.genotype_tiles, self.tile_mask)
        auxiliary_data = (self.variant_count, self.tile_size)
        return children, auxiliary_data

    @classmethod
    def tree_unflatten(cls, auxiliary_data, children):
        variant_count, tile_size = auxiliary_data
        return cls(
            genotype_tiles=children[0],
            tile_mask=children[1],
            variant_count=variant_count,
            tile_size=tile_size,
        )


def _scan_matvec(
    genotype_tiles: jnp.ndarray,
    coefficient_tiles: jnp.ndarray,
    tile_mask: jnp.ndarray,
) -> jnp.ndarray:
    def scan_body(
        accumulated_prediction: jnp.ndarray,
        scan_inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, None]:
        genotype_tile, coefficient_tile, mask_tile = scan_inputs
        updated_prediction = accumulated_prediction + jnp.sum(
            genotype_tile * (coefficient_tile * mask_tile)[:, None],
            axis=0,
        )
        return updated_prediction, None

    initial_prediction = jnp.zeros(genotype_tiles.shape[-1], dtype=jnp.float32)
    final_prediction = lax.scan(scan_body, initial_prediction, (genotype_tiles, coefficient_tiles, tile_mask))[0]
    return final_prediction


@jax.jit
def matvec(operator: GenotypeOperator, coefficients: jnp.ndarray) -> jnp.ndarray:
    padded_coefficients = operator.pad_vector(coefficients)
    coefficient_tiles = padded_coefficients.reshape(operator.genotype_tiles.shape[0], operator.tile_size)
    return _scan_matvec(operator.genotype_tiles, coefficient_tiles, operator.tile_mask)
