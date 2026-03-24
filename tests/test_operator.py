"""Tests for the JAX genotype operator."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from sv_pgs.config import ModelConfig
from sv_pgs.data import GraphEdges
from sv_pgs.operator import GenotypeOperator, matvec, rmatvec, weighted_column_norms, apply_precision, pcg_solve


def _empty_graph(p: int) -> GraphEdges:
    return GraphEdges(
        src=np.array([], dtype=np.int32),
        dst=np.array([], dtype=np.int32),
        sign=np.array([], dtype=np.float32),
        weight=np.array([], dtype=np.float32),
        block_ids=np.arange(p, dtype=np.int32),
    )


class TestGenotypeOperator:
    def test_matvec_matches_numpy(self):
        rng = np.random.default_rng(100)
        n, p = 50, 30
        X = rng.standard_normal((n, p)).astype(np.float32)
        beta = rng.standard_normal(p).astype(np.float32)
        config = ModelConfig(tile_size=8)
        graph = _empty_graph(p)
        op = GenotypeOperator.from_numpy(X, graph, config)

        result = np.asarray(matvec(op, jnp.asarray(beta)))
        expected = X @ beta
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_rmatvec_matches_numpy(self):
        rng = np.random.default_rng(101)
        n, p = 50, 30
        X = rng.standard_normal((n, p)).astype(np.float32)
        r = rng.standard_normal(n).astype(np.float32)
        config = ModelConfig(tile_size=8)
        graph = _empty_graph(p)
        op = GenotypeOperator.from_numpy(X, graph, config)

        result = np.asarray(rmatvec(op, jnp.asarray(r)))
        expected = X.T @ r
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_weighted_column_norms(self):
        rng = np.random.default_rng(102)
        n, p = 50, 20
        X = rng.standard_normal((n, p)).astype(np.float32)
        W = np.abs(rng.standard_normal(n)).astype(np.float32)
        config = ModelConfig(tile_size=8)
        graph = _empty_graph(p)
        op = GenotypeOperator.from_numpy(X, graph, config)

        result = np.asarray(weighted_column_norms(op, jnp.asarray(W)))
        expected = np.sum(X ** 2 * W[:, None], axis=0)
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_pcg_solve_recovers_solution(self):
        rng = np.random.default_rng(103)
        n, p = 80, 15
        X = rng.standard_normal((n, p)).astype(np.float32)
        W = np.ones(n, dtype=np.float32)
        prior_prec = np.full(p, 1.0, dtype=np.float32)
        config = ModelConfig(tile_size=8)
        graph = _empty_graph(p)
        op = GenotypeOperator.from_numpy(X, graph, config)

        # True solution.
        H = X.T @ X + np.diag(prior_prec)
        true_beta = rng.standard_normal(p).astype(np.float32)
        rhs = H @ true_beta

        diag = np.diag(H).astype(np.float32)
        x0 = jnp.zeros(p, dtype=jnp.float32)
        result = np.asarray(pcg_solve(
            op, jnp.asarray(rhs), jnp.asarray(W), jnp.asarray(prior_prec),
            jnp.asarray(np.maximum(diag, 1e-4)),
            x0, 1e-6, 200,
        ))
        np.testing.assert_allclose(result, true_beta, atol=1e-3)
