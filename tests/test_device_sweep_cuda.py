"""The device sweep (``device_sweep``) against the host sweep (``mean_field._sweep``) on the same projected problem."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.compute_budget import _try_import_cupy
from sv_pgs.device_sweep import PANEL, PanelGrams, sweep_piece
from sv_pgs.mean_field import _sweep as host_sweep

cupy = _try_import_cupy()
pytestmark = pytest.mark.skipif(cupy is None, reason="needs a CUDA device")


def _problem(seed: int, samples: int, members: int, nodes: int):
    rng = np.random.default_rng(seed)
    mask = (rng.random(samples) < 0.8).astype(np.float64)
    codes = rng.binomial(2, rng.uniform(0.02, 0.5, members), size=(samples, members)).astype(np.float64)
    # correlated neighbours: the within-panel Gram terms carry real weight
    codes[:, 1::2] = np.where(rng.random((samples, members // 2)) < 0.9, codes[:, 0::2][:, : members // 2], codes[:, 1::2])
    dense = (codes - codes.mean(axis=0)) / np.maximum(codes.std(axis=0), 1e-12)
    covariates = np.column_stack([np.ones(samples), rng.normal(size=(samples, 3))]) * mask[:, None]
    solve = np.linalg.pinv(covariates.T @ covariates)

    def project_host(values):
        return values - covariates @ (solve @ (covariates.T @ values))

    projected = project_host(dense * mask[:, None])
    effects = np.where(rng.random(members) < 0.1, rng.normal(0.0, 0.3, members), 0.0)
    target = project_host(((dense @ effects) + rng.normal(size=samples)) * mask)
    grid = np.linspace(np.log(1e-5), np.log(0.5), nodes)
    classes = rng.integers(0, 2, members).astype(np.int64)
    log_density = rng.normal(size=(2, nodes))
    log_density -= np.log(np.exp(log_density).sum(axis=1, keepdims=True))
    log_node_variance = rng.normal(0.0, 0.3, members)[:, None] + grid[None, :]
    return dict(mask=mask, dense=dense, covariates=covariates, solve=solve, projected=projected, target=target, classes=classes,
                log_density=log_density, log_node_variance=log_node_variance, squares=np.sum(projected * projected, axis=0))


@pytest.mark.parametrize("members", [PANEL, 3 * PANEL + 7])
def test_the_device_sweep_is_the_host_sweep(members: int) -> None:
    data = _problem(5, 400, members, 41)
    noise = 0.9
    count = members
    host = {name: np.zeros(count) for name in ("mean", "variance", "shift", "third", "fourth")}
    host_residual = data["target"].copy()
    device = {name: cupy.zeros(count) for name in host}
    device_residual = cupy.asarray(data["target"])
    pieces = cupy.zeros((count, 3))
    covariates, solve = cupy.asarray(data["covariates"]), cupy.asarray(data["solve"])

    def project(values):
        return values - covariates @ (solve @ (covariates.T @ values))

    grams = PanelGrams()
    for _sweep_index in range(3):
        parts = host_sweep(
            np.asfortranarray(data["projected"]), data["squares"], np.arange(count), data["classes"], data["log_density"],
            np.exp(data["log_node_variance"]), data["log_node_variance"], noise, host["mean"], host_residual, host["variance"],
            host["shift"], host["third"], host["fourth"],
        )
        pieces[...] = 0.0
        sweep_piece(
            cupy, dense=cupy.asarray(data["dense"]), mask=cupy.asarray(data["mask"]), project=project, residual=device_residual,
            grams=grams, key_base=(0, 0), squares=cupy.asarray(data["squares"]), class_index=cupy.asarray(data["classes"]),
            log_density=cupy.asarray(data["log_density"]), node_variance=cupy.exp(cupy.asarray(data["log_node_variance"])),
            log_node_variance=cupy.asarray(data["log_node_variance"]), noise=noise, pieces=pieces, **device,
        )
        totals = cupy.asnumpy(pieces.sum(axis=0))
        # the same members in the same order: the moments, the residual and the ELBO's pieces agree to rounding
        scale = 1.0 + np.max(np.abs(host["mean"]))
        for name in host:
            assert np.allclose(cupy.asnumpy(device[name]), host[name], rtol=1e-9, atol=1e-11 * scale), name
        assert np.allclose(cupy.asnumpy(device_residual), host_residual, rtol=1e-9, atol=1e-10 * np.linalg.norm(host_residual))
        assert totals[0] == pytest.approx(parts[0], rel=1e-9, abs=1e-9 * parts[3])
        assert totals[1] == pytest.approx(parts[1], rel=1e-9)
    # the panel Grams were formed once and reused by the later sweeps
    assert len(grams._grams) == -(-members // PANEL)
