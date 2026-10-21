"""One arm's traits in one sweep pass (``device_sweep.sweep_models`` with each trait's own standardization, members and
lattice; ``ArmTraits``, ``trait_lattices``), on the host through the numpy reference kernels: every trait's sweep is
the host sweep (``mean_field._sweep``) of that trait's own fit, which holds only its training rows, its own members
(the rows not constant there, less any its filters left out) standardized on those rows, and its own lattice, to
rounding; a trait's sweep is the same to the bit whichever traits share the pass; and ``ArmTraits`` is Stage 0's
standardization of each trait's training rows.

numpy stands in for CuPy as in test_sweep_models (a layout over two device ids runs its blocks and hand-offs as on two
GPUs); test_sweep_arm_cuda checks the CUDA kernels against the same reference."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.code_products import CodeBlockTile
from sv_pgs.config import ModelConfig
from sv_pgs.device_sweep import MOMENTS, PANEL, ArmTraits, HostPanelKernels, PanelGrams, SweepLayout, sweep_models, trait_lattices
from sv_pgs.genotype_buffers import SIGNED_CODE_OFFSET
from sv_pgs.mean_field import _sweep as host_sweep

# Three blocks over 240 samples; eleven traits [real: an arm's 11 traits] on training rows of 35-90% of them [real:
# 1.7e4-4.4e4 of ~5e4], lattices of 20-40 nodes; pieces of 100 members and Grams decoded 64 columns at a time.
_BLOCKS = (60, 45, 70)
_SAMPLES = 240
_TRAITS = 11
_PIECE = 100
_GRAM_COLUMNS = 2 * PANEL
_STAGE2 = ModelConfig(minimum_minor_allele_frequency=0.0)


def arm_problem(xp, seed: int, devices: tuple[int, ...] = (0, 1, 0)):
    """An arm's codes over every sample on ``xp``'s devices and eleven traits' training rows, metrics, priors and
    starting residuals; each trait's own fit (its rows, members, standardized projected design) for the reference."""
    rng = np.random.default_rng(seed)
    blocks = []
    for size in _BLOCKS:
        codes = rng.integers(-60, 61, size=(size, _SAMPLES)).astype(np.int8)
        # correlated neighbours: the within-panel Gram terms carry real weight
        codes[1::2] = np.where(rng.random((size // 2, _SAMPLES)) < 0.8, codes[0::2][: size // 2], codes[1::2])
        blocks.append(codes)
    codes = np.concatenate(blocks)
    members = codes.shape[0]
    training = np.zeros((_TRAITS, _SAMPLES), dtype=bool)
    for trait, fraction in enumerate(np.linspace(0.35, 0.9, _TRAITS)):
        training[trait, rng.choice(_SAMPLES, int(round(fraction * _SAMPLES)), replace=False)] = True
    # Rare rows: constant on one trait's training rows (not its member there), varying on three samples outside them.
    for row, trait in ((5, 0), (50, 3), (61, 3), (120, 10), (174, 6)):
        codes[row] = -60
        codes[row, rng.choice(np.flatnonzero(~training[trait]), 3, replace=False)] = 60
    # Each trait's own members: the rows varying on its training rows, less a few its own filters left out.
    varying = np.stack([codes[:, training[trait]].std(axis=1) > 0 for trait in range(_TRAITS)], axis=1)
    live = varying & (rng.random((members, _TRAITS)) > 0.03)
    tiles, bounds, start = [], [], 0
    for index, block in enumerate(blocks):
        size = block.shape[0]
        values = codes[start:start + size]
        # the tile's own standardization (over every sample) is not any trait's: each trait's replaces it
        means, scales = values.mean(axis=1), values.std(axis=1) + 1.0
        if xp is np:
            tiles.append(CodeBlockTile(values, means, scales, np, 1 << 24))
        else:
            with xp.cuda.Device(devices[index]):
                tiles.append(CodeBlockTile(xp.asarray(values), xp.asarray(means), xp.asarray(scales), xp, 1 << 24))
        bounds.append((start, start + size))
        start += size
    covariates = np.column_stack([np.ones(_SAMPLES), rng.normal(size=(_SAMPLES, 2))])
    # Quantitative traits' masks, and a binary trait's root weights sqrt(omega) on its training rows (the last).
    masks = training.astype(np.float64)
    masks[-1] = np.where(training[-1], rng.uniform(0.2, 0.5, _SAMPLES), 0.0)
    classes = rng.integers(0, 2, members).astype(np.int64)
    grids, densities, fits, pinvs, residuals = [], [], [], [], []
    squares = np.ones((members, _TRAITS))
    for trait in range(_TRAITS):
        nodes = 20 + 2 * trait
        grids.append(np.linspace(np.log(1e-5), np.log(0.5) + 0.1 * trait, nodes))
        density = rng.normal(size=(2, nodes))
        densities.append(density - np.log(np.exp(density).sum(axis=1, keepdims=True)))
        rows, own = np.flatnonzero(training[trait]), np.flatnonzero(live[:, trait])
        raw = codes[own][:, rows].T.astype(np.float64)
        weights = masks[trait, rows]
        masked_covariates = covariates[rows] * weights[:, None]
        pinv = np.linalg.pinv(masked_covariates.T @ masked_covariates)

        def project(values, masked_covariates=masked_covariates, pinv=pinv, weights=weights):
            masked = values * weights.reshape((-1,) + (1,) * (values.ndim - 1))
            return masked - masked_covariates @ (pinv @ (masked_covariates.T @ masked))

        # the trait's own fit: its training rows only, its members standardized on them (numpy's own mean and sd)
        projected = project((raw - raw.mean(axis=0)) / raw.std(axis=0))
        squares[own, trait] = np.sum(projected * projected, axis=0)
        start_residual = project(projected @ np.where(rng.random(own.shape[0]) < 0.1, rng.normal(0.0, 0.3, own.shape[0]), 0.0) + rng.normal(size=rows.shape[0]))
        residual = np.zeros(_SAMPLES)
        residual[rows] = start_residual
        residuals.append(residual)
        pinvs.append(pinv)
        fits.append(dict(rows=rows, members=own, projected=projected))
    grid, node_counts, log_density = trait_lattices(grids, densities)
    return dict(
        codes=codes, tiles=tiles, bounds=bounds, devices=devices, member_blocks=[np.arange(first, last) for first, last in bounds],
        classes=classes, covariates=covariates, training=training, masks=masks, live=live, pinvs=pinvs, residuals=np.stack(residuals),
        grids=grids, densities=densities, grid=grid, node_counts=node_counts, log_density=log_density, fits=fits, squares=squares,
        log_scale=rng.normal(0.0, 0.3, (members, _TRAITS)), noise=np.linspace(0.8, 1.2, _TRAITS),
    )


def _pieces(count: int):
    for first in range(0, count, _PIECE):
        yield first, min(first + _PIECE, count)


def arm_layout(xp, problem) -> SweepLayout:
    members = problem["codes"].shape[0]
    return SweepLayout(xp, problem["devices"], problem["member_blocks"], problem["bounds"], np.arange(members), np.ones(members), problem["classes"], _pieces)


def _tiles(problem):
    return ((start, stop, tile, device) for (start, stop), tile, device in zip(problem["bounds"], problem["tiles"], problem["devices"]))


def arm_sweep(xp, kernels, problem, layout, traits, standards, state, residuals, grams):
    """One ``sweep_models`` pass of ``traits`` (columns of the problem's trait axis), each with its own standardization
    (``standards``, an ``ArmTraits``), members and lattice."""
    columns = list(traits)
    return sweep_models(
        xp, kernels, layout, _tiles(problem), grams=grams, keys=[(trait, b"") for trait in columns], masks=problem["masks"][columns],
        residuals=residuals, noise=problem["noise"][columns], log_density=problem["log_density"][columns], grid=problem["grid"][columns],
        log_scale=np.ascontiguousarray(problem["log_scale"][:, columns]), squares=np.ascontiguousarray(problem["squares"][:, columns]),
        state=state, covariates=xp.asarray(problem["covariates"]), covariate_pinvs=[xp.asarray(problem["pinvs"][trait]) for trait in columns],
        gram_columns=lambda width: _GRAM_COLUMNS, node_counts=problem["node_counts"][columns],
        standardization=(np.ascontiguousarray(standards.centers[:, columns]), np.ascontiguousarray(standards.scales[:, columns])),
        live=np.ascontiguousarray(problem["live"][:, columns]),
    )


def test_arm_traits_is_stage0_s_standardization_of_each_trait_s_rows() -> None:
    problem = arm_problem(np, 3)
    standards = ArmTraits.from_tiles(np, _tiles(problem), problem["training"], _STAGE2, chunk_rows=16)
    codes = problem["codes"].astype(np.float64)
    for trait in range(_TRAITS):
        rows = codes[:, problem["training"][trait]]
        varying = rows.std(axis=1) > 0
        np.testing.assert_array_equal(standards.live[:, trait], varying)
        np.testing.assert_allclose(standards.centers[varying, trait], rows.mean(axis=1)[varying], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(standards.scales[varying, trait], rows.std(axis=1)[varying], rtol=1e-12)
        # a row the trait does not have carries a finite, unit scale
        assert np.all(standards.centers[~varying, trait] == 0.0) and np.all(standards.scales[~varying, trait] == 1.0)
    # Stage 0's live test also needs the scale above its floor, in code offsets
    tight = ArmTraits.from_tiles(np, _tiles(problem), problem["training"], ModelConfig(minimum_minor_allele_frequency=0.0, minimum_scale=0.2), 16)
    expected = standards.live & (standards.scales / SIGNED_CODE_OFFSET >= 0.2)
    assert expected.any() and not expected[[5, 50, 61, 120, 174]].any()
    np.testing.assert_array_equal(tight.live, expected)


def test_each_trait_s_sweep_in_the_arm_s_pass_is_its_own_fit_s_host_sweep() -> None:
    problem = arm_problem(np, 5)
    layout = arm_layout(np, problem)
    standards = ArmTraits.from_tiles(np, _tiles(problem), problem["training"], _STAGE2, chunk_rows=16)
    members = problem["codes"].shape[0]
    assert layout.panel_count > len(_BLOCKS) and any((last - first) % PANEL for block in layout.blocks for first, last in block.pieces)
    assert len(set(problem["node_counts"])) == _TRAITS and not problem["live"].all()
    state = {name: np.zeros((members, _TRAITS)) for name in MOMENTS}
    residuals = problem["residuals"].copy()
    host = [{name: np.zeros(fit["members"].shape[0]) for name in MOMENTS} for fit in problem["fits"]]
    host_residuals = [problem["residuals"][trait, fit["rows"]].copy() for trait, fit in enumerate(problem["fits"])]
    grams = PanelGrams()
    for _sweep in range(3):
        swept, totals, squares = arm_sweep(np, HostPanelKernels(2 * PANEL), problem, layout, range(_TRAITS), standards, state, residuals, grams)
        residuals = np.stack(swept)
        for trait, fit in enumerate(problem["fits"]):
            own, rows = fit["members"], fit["rows"]
            log_node_variance = problem["log_scale"][own, trait][:, None] + problem["grids"][trait][None, :]
            parts = host_sweep(
                np.asfortranarray(fit["projected"]), np.ascontiguousarray(problem["squares"][own, trait]), np.arange(own.shape[0]), problem["classes"][own],
                problem["densities"][trait], np.exp(log_node_variance), log_node_variance, float(problem["noise"][trait]), host[trait]["mean"],
                host_residuals[trait], host[trait]["variance"], host[trait]["shift"], host[trait]["third"], host[trait]["fourth"],
                np.arange(own.shape[0], dtype=np.int64),
            )
            for name in MOMENTS:
                expected = host[trait][name]
                np.testing.assert_allclose(state[name][own, trait], expected, rtol=1e-9, atol=1e-11 * (1.0 + np.max(np.abs(expected))), err_msg=name)
                # a member the trait does not have is left where it started
                assert not np.any(state[name][~problem["live"][:, trait], trait]), name
            np.testing.assert_allclose(swept[trait][rows], host_residuals[trait], rtol=1e-9, atol=1e-10 * np.linalg.norm(host_residuals[trait]))
            outside = np.ones(_SAMPLES, dtype=bool)
            outside[rows] = False
            assert not np.any(swept[trait][outside])
            assert totals[trait, 0] == pytest.approx(parts[0], rel=1e-9, abs=1e-9 * parts[3])
            assert totals[trait, 1] == pytest.approx(parts[1], rel=1e-9)
            assert squares[trait] == pytest.approx(parts[2], rel=1e-9)
            assert totals[trait, 2] == pytest.approx(parts[3], rel=1e-9)
    # every piece's Grams built once per trait and kept, on its block's device
    assert len(grams._grams) == _TRAITS * sum(len(block.pieces) for block in layout.blocks)


def test_a_trait_s_sweep_is_the_same_whichever_traits_share_the_pass() -> None:
    problem = arm_problem(np, 7)
    layout = arm_layout(np, problem)
    standards = ArmTraits.from_tiles(np, _tiles(problem), problem["training"], _STAGE2, chunk_rows=16)
    members = problem["codes"].shape[0]

    def run(columns):
        state = {name: np.zeros((members, len(columns))) for name in MOMENTS}
        residuals = problem["residuals"][list(columns)].copy()
        grams = PanelGrams()
        outcomes = []
        for _sweep in range(2):
            swept, totals, squares = arm_sweep(np, HostPanelKernels(2 * PANEL), problem, layout, columns, standards, state, residuals, grams)
            residuals = np.stack(swept)
            outcomes.append((residuals.copy(), totals.copy(), squares.copy(), {name: values.copy() for name, values in state.items()}))
        return outcomes

    together = run(tuple(range(_TRAITS)))
    for trait in (0, 4, _TRAITS - 1):
        for (residuals, totals, squares, state), (alone_residuals, alone_totals, alone_squares, alone_state) in zip(together, run((trait,))):
            np.testing.assert_array_equal(residuals[trait], alone_residuals[0])
            np.testing.assert_array_equal(totals[trait], alone_totals[0])
            assert squares[trait] == alone_squares[0]
            for name in MOMENTS:
                np.testing.assert_array_equal(state[name][:, trait], alone_state[name][:, 0])


def test_a_sweep_refuses_a_member_scale_that_is_not_finite_and_nonzero_and_a_lattice_wider_than_its_grid() -> None:
    problem = arm_problem(np, 11)
    layout = arm_layout(np, problem)
    standards = ArmTraits.from_tiles(np, _tiles(problem), problem["training"], _STAGE2, chunk_rows=16)
    members = problem["codes"].shape[0]
    state = {name: np.zeros((members, _TRAITS)) for name in MOMENTS}
    broken = ArmTraits(standards.centers, np.where(standards.live, standards.scales, 0.0), standards.live)
    with pytest.raises(ValueError, match="scale is finite and nonzero"):
        arm_sweep(np, HostPanelKernels(2 * PANEL), problem, layout, range(_TRAITS), broken, state, problem["residuals"].copy(), PanelGrams())
    problem["node_counts"] = problem["node_counts"] + 1
    with pytest.raises(ValueError, match="node count"):
        arm_sweep(np, HostPanelKernels(2 * PANEL), problem, layout, range(_TRAITS), standards, state, problem["residuals"].copy(), PanelGrams())
    assert all(not np.any(values) for values in state.values())
