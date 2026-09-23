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

    grams = PanelGrams()
    for _sweep_index in range(3):
        parts = host_sweep(
            np.asfortranarray(data["projected"]), data["squares"], np.arange(count), data["classes"], data["log_density"],
            np.exp(data["log_node_variance"]), data["log_node_variance"], noise, host["mean"], host_residual, host["variance"],
            host["shift"], host["third"], host["fourth"], np.arange(count, dtype=np.int64),
        )
        pieces[...] = 0.0
        dense = cupy.asarray(data["dense"])
        sweep_piece(
            cupy, decode=lambda first, last: dense[:, first:last], width=count, mask=cupy.asarray(data["mask"]), covariates=covariates, covariate_pinv=solve, residual=device_residual,
            grams=grams, model=0, members=np.arange(len(data["squares"])), squares=cupy.asarray(data["squares"]), class_index=cupy.asarray(data["classes"]),
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


def test_the_code_panels_sweep_is_the_decoded_sweep() -> None:
    from sv_pgs.code_products import CodeBlockTile
    from sv_pgs.device_sweep import CodePanels

    rng = np.random.default_rng(11)
    samples, variants, nodes = 700, 3 * PANEL + 5, 31
    codes = rng.integers(-60, 61, size=(variants, samples)).astype(np.int8)
    means = codes.mean(axis=1) + rng.normal(0.0, 0.1, variants)
    scales = codes.std(axis=1) * rng.uniform(0.8, 1.2, variants)
    tile = CodeBlockTile(codes, means, scales, cupy, 1 << 26)
    order = rng.permutation(variants)
    signs = np.where(rng.random(variants) < 0.5, -1.0, 1.0)
    mask = (rng.random(samples) < 0.85).astype(np.float64)
    covariates = np.column_stack([np.ones(samples), rng.normal(size=(samples, 2))]) * mask[:, None]
    solve = np.linalg.pinv(covariates.T @ covariates)
    dense = ((codes.T - means) / scales)[:, order] * signs[order]
    target = rng.normal(size=samples) * mask
    target -= covariates @ (solve @ (covariates.T @ target))
    grid = np.linspace(np.log(1e-5), np.log(0.5), nodes)
    log_density = rng.normal(size=(1, nodes))
    log_density -= np.log(np.exp(log_density).sum())
    log_node_variance = np.tile(grid, (variants, 1))
    masked = dense * mask[:, None]
    projected = masked - covariates @ (solve @ (covariates.T @ masked))
    squares = np.sum(projected * projected, axis=0)
    outcomes = []
    for use_codes in (False, True):
        state = {name: cupy.zeros(variants) for name in ("mean", "variance", "shift", "third", "fourth")}
        residual = cupy.asarray(target)
        device_order, device_signs = cupy.asarray(order), cupy.asarray(signs[order])
        panels = CodePanels(cupy, tile, device_order, device_signs) if use_codes else None
        for _sweep_index in range(3):
            sweep_piece(
                cupy, decode=lambda first, last: cupy.asarray(tile.columns(device_order[first:last])) * device_signs[first:last][None, :],
                width=variants, mask=cupy.asarray(mask), covariates=cupy.asarray(covariates), covariate_pinv=cupy.asarray(solve),
                residual=residual, grams=PanelGrams(), model=0, members=order, squares=cupy.asarray(squares),
                class_index=cupy.zeros(variants, dtype=cupy.int64), log_density=cupy.asarray(log_density),
                node_variance=cupy.exp(cupy.asarray(log_node_variance)), log_node_variance=cupy.asarray(log_node_variance), noise=0.8,
                pieces=cupy.zeros((variants, 3)), panels=panels, **state,
            )
        outcomes.append((cupy.asnumpy(residual), {name: cupy.asnumpy(values) for name, values in state.items()}))
    (decoded_residual, decoded), (coded_residual, coded) = outcomes
    # The same float64 sums in another order: equal to rounding, the residual on its own norm's scale.
    assert np.allclose(coded_residual, decoded_residual, rtol=1e-10, atol=1e-11 * np.linalg.norm(decoded_residual))
    for name in decoded:
        assert np.allclose(coded[name], decoded[name], rtol=1e-9, atol=1e-12 * (1.0 + np.max(np.abs(decoded[name])))), name
