"""The batched mean-field sweep (``device_sweep.sweep_models``) on the host, through its numpy reference kernels
(``HostPanelKernels``): each model's sweep is the host sweep (``mean_field._sweep``) on its own projected design, to
rounding; a model's sweep is the same to the bit whichever models share it; and the oracle's side-by-side solves
(``full_data_fit._FullDataMeanField._solve_models``) reach, bit for bit, the states its one-model solves reach.

numpy stands in for CuPy (``compute_budget._cupy_device_context`` needs no device there), so a layout over two device
ids runs its blocks, hand-offs and per-device sums as on two GPUs; the CUDA kernels are checked against the same
reference in test_sweep_models_cuda."""
from __future__ import annotations

from pathlib import Path
import types

import numpy as np
import pytest

from sv_pgs.code_products import CodeBlockTile
from sv_pgs.device_sweep import MOMENTS, PANEL, HostPanelKernels, PanelGrams, SweepLayout, sweep_models
from sv_pgs.mean_field import _sweep as host_sweep

# Three blocks of 70, 45 and 90 columns, every fifth column of each also a second (signed) tie member; pieces of 100
# members and Grams decoded 64 columns at a time, so pieces end in partial panels and Gram chunks end mid-piece.
_BLOCKS = (70, 45, 90)
_PIECE = 100
_GRAM_COLUMNS = 2 * PANEL
_SAMPLES = 300
_NODES = 37


def sweep_problem(
    xp, seed: int, devices: tuple[int, ...] = (0, 1, 0), samples: int = _SAMPLES, blocks: tuple[int, ...] = _BLOCKS, collinear: float = 0.0,
):
    """Codes, members and three models' metrics, priors and starting residuals; the tiles on ``xp``'s devices
    (``devices`` per block) and each model's dense projected design (host) for the reference. ``collinear`` (if not
    0) makes the second covariate the first block's first column to within that fraction of its scale."""
    rng = np.random.default_rng(seed)
    tiles, bounds, member_blocks, groups, signs, dense_columns = [], [], [], [], [], []
    start = member = 0
    for index, size in enumerate(blocks):
        codes = rng.integers(-60, 61, size=(size, samples)).astype(np.int8)
        # correlated neighbours: the within-panel Gram terms carry real weight
        codes[1::2] = np.where(rng.random((size // 2, samples)) < 0.8, codes[0::2][: size // 2], codes[1::2])
        means = codes.mean(axis=1) + rng.normal(0.0, 0.1, size)
        scales = codes.std(axis=1) * rng.uniform(0.8, 1.2, size)
        dense_columns.append((codes.T - means) / scales)
        device = devices[index]
        if xp is np:
            tiles.append(CodeBlockTile(codes, means, scales, np, 1 << 24))
        else:
            with xp.cuda.Device(device):
                tiles.append(CodeBlockTile(xp.asarray(codes), xp.asarray(means), xp.asarray(scales), xp, 1 << 24))
        block_groups = np.concatenate([np.arange(size), np.arange(0, size, 5)]) + start
        block_signs = np.concatenate([np.ones(size), np.where(rng.random(-(-size // 5)) < 0.5, -1.0, 1.0)])
        count = block_groups.shape[0]
        groups.append(block_groups)
        signs.append(block_signs)
        member_blocks.append(rng.permutation(np.arange(member, member + count)))
        bounds.append((start, start + size))
        start += size
        member += count
    group, sign = np.concatenate(groups), np.concatenate(signs)
    design = np.concatenate(dense_columns, axis=1)[:, group] * sign
    members = group.shape[0]
    classes = rng.integers(0, 2, members).astype(np.int64)
    covariates = np.column_stack([np.ones(samples), rng.normal(size=(samples, 2))])
    if collinear:
        covariates[:, 1] = dense_columns[0][:, 0] + collinear * rng.normal(size=samples)
    training = rng.random((samples, 2)) < 0.85
    # Two quantitative models' training masks and a binary model's root weights sqrt(omega) on its training rows.
    masks = np.stack([training[:, 0], training[:, 1], np.where(training[:, 0], rng.uniform(0.2, 0.5, samples), 0.0)]).astype(np.float64)
    models = masks.shape[0]
    pinvs, projected, residuals = [], [], []
    for model in range(models):
        masked_covariates = covariates * masks[model][:, None]
        pinv = np.linalg.pinv(masked_covariates.T @ masked_covariates)
        pinvs.append(pinv)

        def project(values, masked_covariates=masked_covariates, pinv=pinv, mask=masks[model]):
            masked = values * mask.reshape((-1,) + (1,) * (values.ndim - 1))
            return masked - masked_covariates @ (pinv @ (masked_covariates.T @ masked))

        projected.append(project(design))
        residuals.append(project(design @ np.where(rng.random(members) < 0.1, rng.normal(0.0, 0.3, members), 0.0) + rng.normal(size=samples)))
    grid = np.linspace(np.log(1e-5), np.log(0.5), _NODES)
    log_density = rng.normal(size=(models, 2, _NODES))
    log_density -= np.log(np.exp(log_density).sum(axis=2, keepdims=True))
    return dict(
        tiles=tiles, bounds=bounds, devices=devices, member_blocks=member_blocks, group=group, sign=sign, classes=classes,
        covariates=covariates, masks=masks, pinvs=pinvs, projected=projected, residuals=np.stack(residuals), grid=grid,
        log_density=log_density, log_scale=rng.normal(0.0, 0.3, (members, models)),
        squares=np.column_stack([np.sum(values * values, axis=0) for values in projected]), noise=np.array([0.9, 1.1, 1.0]),
    )


def _pieces(count: int):
    for first in range(0, count, _PIECE):
        yield first, min(first + _PIECE, count)


def _layout(xp, problem) -> SweepLayout:
    return SweepLayout(xp, problem["devices"], problem["member_blocks"], problem["bounds"], problem["group"], problem["sign"], problem["classes"], _pieces)


def batched_sweep(xp, kernels, problem, layout, models, state, residuals, grams):
    """One ``sweep_models`` pass of ``models`` (columns of the problem's model axis) from ``state`` (name -> (members,
    len(models)), updated in place) and ``residuals`` (len(models), n)."""
    tiles = ((start, stop, tile, device) for (start, stop), tile, device in zip(problem["bounds"], problem["tiles"], problem["devices"]))
    columns = list(models)
    return sweep_models(
        xp, kernels, layout, tiles, grams=grams, keys=[(model, b"") for model in columns], masks=problem["masks"][columns], residuals=residuals,
        noise=problem["noise"][columns], log_density=problem["log_density"][columns], grid=problem["grid"],
        log_scale=np.ascontiguousarray(problem["log_scale"][:, columns]), squares=np.ascontiguousarray(problem["squares"][:, columns]),
        state=state, covariates=xp.asarray(problem["covariates"]), covariate_pinvs=[xp.asarray(problem["pinvs"][model]) for model in columns],
        gram_columns=lambda width: _GRAM_COLUMNS,
    )


def host_reference(problem, model, order, state, residual):
    """``mean_field._sweep`` on the model's projected design, the members in sweep ``order``; ``state`` and
    ``residual`` in place (members in sweep order)."""
    log_node_variance = problem["log_scale"][order, model][:, None] + problem["grid"][None, :]
    return host_sweep(
        np.asfortranarray(problem["projected"][model][:, order]), np.ascontiguousarray(problem["squares"][order, model]), np.arange(order.shape[0]),
        problem["classes"][order], problem["log_density"][model], np.exp(log_node_variance), log_node_variance, float(problem["noise"][model]),
        state["mean"], residual, state["variance"], state["shift"], state["third"], state["fourth"], np.arange(order.shape[0], dtype=np.int64),
    )


def test_each_model_s_batched_sweep_is_its_host_sweep() -> None:
    problem = sweep_problem(np, 3)
    layout = _layout(np, problem)
    order = np.concatenate(problem["member_blocks"])
    members, models = order.shape[0], problem["masks"].shape[0]
    assert layout.panel_count > len(_BLOCKS) and any((last - first) % PANEL for block in layout.blocks for first, last in block.pieces)
    state = {name: np.zeros((members, models)) for name in MOMENTS}
    residuals = problem["residuals"].copy()
    host = [{name: np.zeros(members) for name in MOMENTS} for _model in range(models)]
    host_residuals = [problem["residuals"][model].copy() for model in range(models)]
    grams = PanelGrams()
    for _sweep in range(3):
        swept, totals, squares = batched_sweep(np, HostPanelKernels(2 * PANEL), problem, layout, range(models), state, residuals, grams)
        residuals = np.stack(swept)
        for model in range(models):
            parts = host_reference(problem, model, order, host[model], host_residuals[model])
            # The same members in the same order: the moments, the residual and the ELBO's pieces agree to rounding.
            for name in MOMENTS:
                expected = host[model][name]
                np.testing.assert_allclose(state[name][order, model], expected, rtol=1e-9, atol=1e-11 * (1.0 + np.max(np.abs(expected))), err_msg=name)
            np.testing.assert_allclose(swept[model], host_residuals[model], rtol=1e-9, atol=1e-10 * np.linalg.norm(host_residuals[model]))
            assert totals[model, 0] == pytest.approx(parts[0], rel=1e-9, abs=1e-9 * parts[3])
            assert totals[model, 1] == pytest.approx(parts[1], rel=1e-9)
            assert squares[model] == pytest.approx(parts[2], rel=1e-9)
            assert totals[model, 2] == pytest.approx(parts[3], rel=1e-9)
    # Every piece's Grams were built once per model and kept for the later sweeps, on its block's device.
    pieces = sum(len(block.pieces) for block in layout.blocks)
    assert len(grams._grams) == models * pieces
    assert {key[0] for key in grams._grams} == {"device0", "device1"}


def test_a_model_s_sweep_is_the_same_whichever_models_share_it() -> None:
    problem = sweep_problem(np, 5)
    layout = _layout(np, problem)
    members, models = sum(block.members.shape[0] for block in layout.blocks), problem["masks"].shape[0]

    def run(columns):
        state = {name: np.zeros((members, len(columns))) for name in MOMENTS}
        residuals = problem["residuals"][list(columns)].copy()
        grams = PanelGrams()
        outcomes = []
        for _sweep in range(3):
            swept, totals, squares = batched_sweep(np, HostPanelKernels(2 * PANEL), problem, layout, columns, state, residuals, grams)
            residuals = np.stack(swept)
            outcomes.append((residuals.copy(), totals.copy(), squares.copy(), {name: values.copy() for name, values in state.items()}))
        return outcomes

    together = run(tuple(range(models)))
    for model in range(models):
        for (residuals, totals, squares, state), (alone_residuals, alone_totals, alone_squares, alone_state) in zip(together, run((model,))):
            np.testing.assert_array_equal(residuals[model], alone_residuals[0])
            np.testing.assert_array_equal(totals[model], alone_totals[0])
            assert squares[model] == alone_squares[0]
            for name in MOMENTS:
                np.testing.assert_array_equal(state[name][:, model], alone_state[name][:, 0])


def test_a_block_whose_tile_holds_no_codes_is_refused_before_any_state_moves() -> None:
    from sv_pgs.device_sweep import NoCodes

    problem = sweep_problem(np, 7)
    layout = _layout(np, problem)
    members = sum(block.members.shape[0] for block in layout.blocks)
    state = {name: np.zeros((members, 3)) for name in MOMENTS}
    problem["tiles"][1] = types.SimpleNamespace(columns=problem["tiles"][1].columns)
    with pytest.raises(NoCodes):
        batched_sweep(np, HostPanelKernels(2 * PANEL), problem, layout, range(3), state, problem["residuals"].copy(), PanelGrams())
    assert all(not np.any(values) for values in state.values())


def _oracle(tmp_path: Path):
    from tests.test_full_data_fit import _two_trait_oracle

    return _two_trait_oracle(tmp_path)


def test_the_oracle_s_batched_sweeps_are_its_host_sweeps(tmp_path: Path) -> None:
    """``_FullDataMeanField._panel_sweeps`` on the store's codes against the oracle's host sweeps (``_sweep``: the
    projected pieces through ``mean_field._sweep``), both models of a group in each batched pass."""
    oracle_of, starts, _statistics, _prior = _oracle(tmp_path)
    host, batched = oracle_of([0, 1]), oracle_of([0, 1])
    requests = {0: starts[0], 1: starts[1]}
    kernels = HostPanelKernels(2 * PANEL)
    for _sweep in range(3):
        expected = {model: host._sweep(model, hyperparameters) for model, hyperparameters in requests.items()}
        swept = batched._panel_sweeps(requests, kernels)
        assert swept is not None
        for model in requests:
            divergence, weighted_variance, residual_square, sizes = swept[model]
            assert divergence == pytest.approx(expected[model][0], rel=1e-9, abs=1e-9 * expected[model][3])
            assert weighted_variance == pytest.approx(expected[model][1], rel=1e-9)
            assert residual_square == pytest.approx(expected[model][2], rel=1e-9)
            assert sizes == pytest.approx(expected[model][3], rel=1e-9)
            for name in MOMENTS:
                values = getattr(host, name)[:, model]
                np.testing.assert_allclose(getattr(batched, name)[:, model], values, rtol=1e-9, atol=1e-11 * (1.0 + np.max(np.abs(values))), err_msg=name)
            np.testing.assert_allclose(batched.residual[model], host.residual[model], rtol=1e-9, atol=1e-10 * np.linalg.norm(host.residual[model]))
    # One read of the store per batched pass, one pass per model.
    assert batched.passes == 3 and host.passes == 6
    np.testing.assert_array_equal(batched.model_passes, host.model_passes)


def test_side_by_side_solves_are_each_model_s_own_solve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On the batched route (its numpy reference), the group's solves run side by side, every round's sweeps one pass,
    and reach the states the one-model solves reach, bit for bit, in the same passes of each model; an oracle call,
    whose second call adds the cold solves under the carried ones' pass budgets, is the one-at-a-time call's."""
    from sv_pgs.full_data_fit import _FullDataMeanField, _GramMeanField
    from sv_pgs.scale_mixture_ep import MixtureHyperparameters

    kernels = HostPanelKernels(2 * PANEL)

    def batched_sweeps(self, requests):
        return self._panel_sweeps(requests, kernels)

    def one_sweep(self, model, hyperparameters):
        return self._panel_sweeps({model: hyperparameters}, kernels)[model]

    monkeypatch.setattr(_FullDataMeanField, "_sweeps", batched_sweeps)
    monkeypatch.setattr(_FullDataMeanField, "_sweep", one_sweep)
    oracle_of, starts, _statistics, _prior = _oracle(tmp_path)
    together, apart = oracle_of([0, 1]), oracle_of([0, 1])
    assert together._solve_models([0, 1], list(starts), [None, None]) == [None, None]
    for model in range(2):
        apart._solve_model(model, starts[model])

    def same(left, right):
        for name in MOMENTS:
            np.testing.assert_array_equal(getattr(left, name), getattr(right, name), err_msg=name)
        for model in range(2):
            np.testing.assert_array_equal(left.residual[model], right.residual[model])
        np.testing.assert_array_equal(left.noise, right.noise)
        np.testing.assert_array_equal(left.elbo, right.elbo)
        np.testing.assert_array_equal(left.model_passes, right.model_passes)

    same(together, apart)
    assert together.passes == int(np.max(together.model_passes)) < apart.passes

    # Whole calls: the second, at moved hyperparameters, solves from the carried state and from the start.
    together, apart = oracle_of([0, 1]), oracle_of([0, 1])
    apart._solve_models = types.MethodType(_GramMeanField._solve_models, apart)
    moved = [MixtureHyperparameters(coefficients=start.coefficients * 0.9, log_smoothing=start.log_smoothing.copy()) for start in starts]
    for hyperparameters in (list(starts), moved):
        points = together(hyperparameters), apart(hyperparameters)
        assert all(point is not None for point in points[0]) and all(point is not None for point in points[1])
        same(together, apart)
    assert together.refusals == apart.refusals
