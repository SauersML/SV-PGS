"""Matrix-free genotype operator for tiled JAX matvecs and PCG solver.

The hot loop uses custom scan-based tiled matvecs rather than
jax.experimental.sparse. Genotype data is stored variant-major in tiles
and accumulated in float32 via lax.Precision.HIGHEST.

Single-device only. Buffer donation is used on the PCG solve to let XLA
recycle work arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax


from sv_pgs.config import ModelConfig
from sv_pgs.data import GraphEdges


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GenotypeOperator:
    """Encapsulates tiled variant-major genotype matrix and graph Laplacian.

    Genotypes are stored as (num_tiles, tile_size, n) for scan-based matvec.
    A mask array zeros out padding variants in the last tile.
    """

    x_tiles: jnp.ndarray       # (num_tiles, tile_size, n)
    tile_mask: jnp.ndarray      # (num_tiles, tile_size)
    graph_src: jnp.ndarray      # (E,)
    graph_dst: jnp.ndarray      # (E,)
    graph_sign: jnp.ndarray     # (E,)
    graph_weight: jnp.ndarray   # (E,)
    variant_count: int
    sample_count: int
    tile_size: int

    @classmethod
    def from_numpy(
        cls,
        genotypes: np.ndarray,
        graph: GraphEdges,
        config: ModelConfig,
    ) -> GenotypeOperator:
        # Store variant-major: (p, n) → tiled as (num_tiles, tile_size, n).
        x_variant_major = np.asarray(genotypes.T, dtype=np.float32)
        p, n = x_variant_major.shape
        tile_size = config.tile_size
        tile_count = int(np.ceil(p / tile_size))
        padded_p = tile_count * tile_size

        padded = np.zeros((padded_p, n), dtype=np.float32)
        padded[:p] = x_variant_major

        tile_mask = np.zeros((tile_count, tile_size), dtype=np.float32)
        tile_mask.reshape(-1)[:p] = 1.0

        return cls(
            x_tiles=jnp.asarray(padded.reshape(tile_count, tile_size, n)),
            tile_mask=jnp.asarray(tile_mask),
            graph_src=jnp.asarray(graph.src, dtype=jnp.int32),
            graph_dst=jnp.asarray(graph.dst, dtype=jnp.int32),
            graph_sign=jnp.asarray(graph.sign, dtype=jnp.float32),
            graph_weight=jnp.asarray(graph.weight, dtype=jnp.float32),
            variant_count=p,
            sample_count=n,
            tile_size=tile_size,
        )

    def pad_vector(self, vector: jnp.ndarray) -> jnp.ndarray:
        """Zero-pad a (p,) vector to (padded_p,) for tiled operations."""
        padded = jnp.zeros(self.x_tiles.shape[0] * self.tile_size, dtype=vector.dtype)
        return padded.at[:self.variant_count].set(vector)

    def truncate_vector(self, vector: jnp.ndarray) -> jnp.ndarray:
        """Truncate a (padded_p,) vector back to (p,)."""
        return vector[:self.variant_count]

    def tree_flatten(self):
        children = (
            self.x_tiles, self.tile_mask,
            self.graph_src, self.graph_dst, self.graph_sign, self.graph_weight,
        )
        aux = (self.variant_count, self.sample_count, self.tile_size)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        variant_count, sample_count, tile_size = aux_data
        return cls(
            x_tiles=children[0], tile_mask=children[1],
            graph_src=children[2], graph_dst=children[3],
            graph_sign=children[4], graph_weight=children[5],
            variant_count=variant_count, sample_count=sample_count,
            tile_size=tile_size,
        )


# ---------------------------------------------------------------------------
# Tiled scan-based matvec / rmatvec
# ---------------------------------------------------------------------------


def _scan_matvec(
    x_tiles: jnp.ndarray,
    beta_tiles: jnp.ndarray,
    tile_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Compute X @ beta via scan over tiles. Returns (n,)."""

    def body(acc, inputs):
        x_tile, beta_tile, mask = inputs
        # x_tile: (tile_size, n), beta_tile: (tile_size,), mask: (tile_size,)
        acc = acc + jnp.einsum(
            "vn,v->n", x_tile, beta_tile * mask,
            precision=lax.Precision.HIGHEST,
        )
        return acc, None

    init = jnp.zeros(x_tiles.shape[-1], dtype=jnp.float32)
    result, _ = lax.scan(body, init, (x_tiles, beta_tiles, tile_mask))
    return result


def _scan_rmatvec(
    x_tiles: jnp.ndarray,
    residual: jnp.ndarray,
    tile_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Compute X^T @ residual via scan over tiles. Returns (padded_p,)."""

    def body(_, inputs):
        x_tile, mask = inputs
        local = jnp.einsum(
            "vn,n->v", x_tile, residual,
            precision=lax.Precision.HIGHEST,
        )
        return None, local * mask

    _, values = lax.scan(body, None, (x_tiles, tile_mask))
    return values.reshape(-1)


# ---------------------------------------------------------------------------
# Graph Laplacian matvec
# ---------------------------------------------------------------------------


def _graph_laplacian_matvec(
    vector: jnp.ndarray,
    src: jnp.ndarray,
    dst: jnp.ndarray,
    sign: jnp.ndarray,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    """Compute L_G @ v where L_G is the signed graph Laplacian.

    For each edge (i,j) with sign s and weight w:
      L_G[i,i] += w,  L_G[j,j] += w
      L_G[i,j] -= s*w, L_G[j,i] -= s*w

    So (L_G v)[i] = sum_j w * (v[i] - s * v[j])  over neighbors j of i.
    """
    if src.shape[0] == 0:
        return jnp.zeros_like(vector)

    out = jnp.zeros_like(vector)
    # For edge (src, dst): src gets w*(v[src] - s*v[dst]), dst gets w*(v[dst] - s*v[src])
    diff_src = weight * (vector[src] - sign * vector[dst])
    diff_dst = weight * (vector[dst] - sign * vector[src])
    out = out.at[src].add(diff_src)
    out = out.at[dst].add(diff_dst)
    return out


# ---------------------------------------------------------------------------
# Public JIT-compiled operations
# ---------------------------------------------------------------------------


@jax.jit
def matvec(operator: GenotypeOperator, beta: jnp.ndarray) -> jnp.ndarray:
    """Compute X @ beta. Input: (p,), output: (n,)."""
    padded = operator.pad_vector(beta)
    beta_tiles = padded.reshape(operator.x_tiles.shape[0], operator.tile_size)
    return _scan_matvec(operator.x_tiles, beta_tiles, operator.tile_mask)


@jax.jit
def rmatvec(operator: GenotypeOperator, residual: jnp.ndarray) -> jnp.ndarray:
    """Compute X^T @ residual. Input: (n,), output: (p,)."""
    raw = _scan_rmatvec(operator.x_tiles, residual, operator.tile_mask)
    return operator.truncate_vector(raw)


@jax.jit
def weighted_column_norms(
    operator: GenotypeOperator,
    sample_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Compute diag(X^T W X) = sum_i W_i * X_{ij}^2 for each j. Output: (p,)."""

    def body(_, inputs):
        x_tile, mask = inputs
        local = jnp.einsum(
            "vn,n,vn->v", x_tile, sample_weights, x_tile,
            precision=lax.Precision.HIGHEST,
        )
        return None, local * mask

    _, values = lax.scan(body, None, (operator.x_tiles, operator.tile_mask))
    return operator.truncate_vector(values.reshape(-1))


@jax.jit
def apply_hessian(
    operator: GenotypeOperator,
    vector: jnp.ndarray,
    sample_weights: jnp.ndarray,
    prior_precision: jnp.ndarray,
) -> jnp.ndarray:
    """Apply H v = X^T W X v + diag(d) v + L_G v. Never forms X^T X."""
    Xv = matvec(operator, vector)
    XtWXv = rmatvec(operator, sample_weights * Xv)
    Lv = _graph_laplacian_matvec(
        vector,
        operator.graph_src, operator.graph_dst,
        operator.graph_sign, operator.graph_weight,
    )
    return XtWXv + prior_precision * vector + Lv


@jax.jit
def pcg_solve(
    operator: GenotypeOperator,
    rhs: jnp.ndarray,
    sample_weights: jnp.ndarray,
    prior_precision: jnp.ndarray,
    precond_diag: jnp.ndarray,
    x0: jnp.ndarray,
    tol: float,
    maxiter: int,
) -> jnp.ndarray:
    """Preconditioned conjugate gradient with early exit via while_loop.

    Solves H x = rhs where H = X^T W X + diag(d) + L_G.
    Preconditioner M^{-1} = 1 / precond_diag.
    """
    M_inv = 1.0 / jnp.maximum(precond_diag, 1e-10)

    r0 = rhs - apply_hessian(operator, x0, sample_weights, prior_precision)
    z0 = M_inv * r0
    rz0 = jnp.dot(r0, z0)
    rhs_norm = jnp.linalg.norm(rhs) + 1e-30

    # State: (x, r, p, rz, iteration)
    init_state = (x0, r0, z0, rz0, jnp.int32(0))

    def cond_fn(state):
        _, r, _, _, k = state
        return (jnp.linalg.norm(r) / rhs_norm > tol) & (k < maxiter)

    def body_fn(state):
        x, r, p, rz, k = state
        Hp = apply_hessian(operator, p, sample_weights, prior_precision)
        pHp = jnp.dot(p, Hp)
        alpha = rz / jnp.maximum(pHp, 1e-30)
        x_new = x + alpha * p
        r_new = r - alpha * Hp
        z_new = M_inv * r_new
        rz_new = jnp.dot(r_new, z_new)
        beta = rz_new / jnp.maximum(rz, 1e-30)
        p_new = z_new + beta * p
        return (x_new, r_new, p_new, rz_new, k + 1)

    x_final, _, _, _, _ = lax.while_loop(cond_fn, body_fn, init_state)
    return x_final
