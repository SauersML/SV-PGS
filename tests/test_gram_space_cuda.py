"""``gram_space`` on a CUDA device: the band's sweep (``device_sweep``'s panel kernel on the band's panels) and its
posterior solves agree with the host's.

The sweep's two sides differ only in summation order: each member's field is s_g less a product over the band
(<= band width terms, float64), and its node sums run over the lattice (``mean_field._sweep``'s own terms), so each
coordinate agrees to a small multiple of the band's summation rounding, carried through the sweeps it takes."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.compute_budget import _try_import_cupy
from sv_pgs.gram_space import GramBand, GramGaussian
from tests.test_gram_space import _banded_design, _prior_arrays

cupy = _try_import_cupy()
pytestmark = pytest.mark.skipif(cupy is None, reason="needs a CUDA device")

_EPSILON = float(np.finfo(np.float64).eps)


def _band(generator: np.random.Generator, widths: list[int], dtype=np.float64):
    design = _banded_design(generator, widths, 40)
    gram = design.T @ design
    starts = np.concatenate([[0], np.cumsum(widths)])
    outcome = design @ (generator.normal(size=design.shape[1]) * 0.3) + generator.normal(size=design.shape[0])
    band = GramBand.from_arrays(
        within=[gram[starts[b]:starts[b + 1], starts[b]:starts[b + 1]].astype(dtype) for b in range(len(widths))],
        cross=[gram[starts[b]:starts[b + 1], starts[b + 1]:starts[b + 2]].astype(dtype) for b in range(len(widths) - 1)],
        scores=design.T @ outcome, target_square=float(outcome @ outcome), sample_count=design.shape[0],
        residual_dimension=float(design.shape[0]), working_bytes=1 << 22,
    )
    return band, starts


@pytest.mark.parametrize("dtype, tempered", [(np.float64, False), (np.float32, False), (np.float64, True)])
def test_device_band_sweep_is_the_host_band_sweep(dtype, tempered) -> None:
    generator = np.random.default_rng(11)
    # Blocks wider than a panel (32), so a block takes several panels and a tie shares one.
    widths = [70, 45, 90]
    band, starts = _band(generator, widths, dtype)
    if tempered:
        # The tempered band (the summary route's far field): each block its own lambda, the kernels its noise / lambda.
        band.set_tempering(generator.uniform(0.4, 1.0, size=len(widths)))
    count = band.group_count
    groups = np.concatenate([np.arange(count), [3, 3]])
    signs = np.concatenate([np.ones(count), [1.0, -1.0]])
    grid, log_density, scales = _prior_arrays(generator, groups.shape[0])
    block_of_group = np.searchsorted(starts, groups, side="right") - 1
    member_blocks = tuple(generator.permutation(np.flatnonzero(block_of_group == b)) for b in range(len(widths)))
    class_index = np.zeros(groups.shape[0], dtype=np.int64)
    states = {side: [np.zeros(groups.shape[0]) for _ in range(5)] for side in ("host", "device")}
    for _sweep in range(3):
        results = {}
        for side, xp in (("host", np), ("device", cupy)):
            state = states[side]
            results[side] = band.sweep(
                xp, member_blocks=member_blocks, group=groups, sign=signs, class_index=class_index, log_density=log_density, scales=scales,
                grid=grid, noise=0.9, mean=state[0], variance=state[1], shift=state[2], third=state[3], fourth=state[4],
            )
        bound = 256 * band.band_width * _EPSILON
        scale = max(float(np.max(np.abs(states["host"][0]))), 1.0)
        np.testing.assert_allclose(states["device"][0], states["host"][0], rtol=0, atol=bound * scale)
        np.testing.assert_allclose(states["device"][1], states["host"][1], rtol=bound)
        np.testing.assert_allclose(results["device"].residual_square, results["host"].residual_square, rtol=0, atol=bound * results["host"].residual_size)
        np.testing.assert_allclose(results["device"].divergence, results["host"].divergence, rtol=bound)


def test_device_band_posterior_solve_is_the_host_solve() -> None:
    generator = np.random.default_rng(12)
    band, _starts = _band(generator, [30, 50, 40])
    count = band.group_count
    precision = generator.uniform(0.05, 2.0, size=count)
    precision[[5, 60]] = [-0.02, 0.0]
    samples = 4 * 40
    right = generator.normal(size=(count, 3))
    solutions = []
    for xp in (np, cupy):
        solver = GramGaussian(band, training=np.ones((samples, 1)), targets=np.zeros((samples, 1)), covariates=np.ones((samples, 1)), array_module=xp)
        solver.iterate(site_precision=precision[:, None], site_shift=np.zeros((count, 1)), noise_variance=np.array([1.1]))
        solution, certificate = solver.posterior_solve(right, 0, np.full(3, 1e-10))
        assert np.all(np.isfinite(certificate))
        solutions.append(solution)
    exact = np.linalg.solve(band.product(np.eye(count)) / 1.1 + np.diag(precision), right)
    for solution in solutions:
        np.testing.assert_allclose(solution, exact, rtol=0, atol=1e-7 * float(np.max(np.abs(exact))))
