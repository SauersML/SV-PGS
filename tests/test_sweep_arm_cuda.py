"""One arm's eleven traits in one sweep pass on CUDA devices (``device_sweep.sweep_models`` with each trait's own
standardization, members and lattice): the kernels against their numpy reference on test_sweep_arm's problem (which
test_sweep_arm checks against each trait's own fit), a trait's sweep the same to the bit whichever traits share the
pass and whichever devices hold the blocks, and ``ArmTraits`` on the device its host value to the bit."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.compute_budget import _try_import_cupy
from sv_pgs.device_sweep import MOMENTS, ArmTraits, CudaPanelKernels, HostPanelKernels, PanelGrams
from tests.test_sweep_arm import _STAGE2, _TRAITS, _tiles, arm_layout, arm_problem, arm_sweep

cupy = _try_import_cupy()
DEVICE_COUNT = 0 if cupy is None else int(cupy.cuda.runtime.getDeviceCount())
pytestmark = pytest.mark.skipif(DEVICE_COUNT == 0, reason="needs a CUDA device")


def _sweeps(xp, kernels, problem, traits, count):
    """``count`` arm passes of ``traits`` from zero moments and the problem's residuals: each pass's (residuals,
    totals, residual squares, moments)."""
    layout = arm_layout(xp, problem)
    standards = ArmTraits.from_tiles(xp, _tiles(problem), problem["training"], _STAGE2, chunk_rows=16)
    members = problem["codes"].shape[0]
    state = {name: np.zeros((members, len(traits))) for name in MOMENTS}
    residuals = problem["residuals"][list(traits)].copy()
    grams, outcomes = PanelGrams(), []
    for _sweep in range(count):
        swept, totals, squares = arm_sweep(xp, kernels, problem, layout, traits, standards, state, residuals, grams)
        residuals = np.stack(swept)
        outcomes.append((residuals.copy(), totals.copy(), squares.copy(), {name: values.copy() for name, values in state.items()}))
    return outcomes


def test_arm_traits_on_the_device_is_its_host_value_bit_for_bit() -> None:
    """The sums are of integers, so every order gives the same bits; the division and the square root round alike."""
    with cupy.cuda.Device(0):
        device = ArmTraits.from_tiles(cupy, _tiles(arm_problem(cupy, 3, devices=(0, 0, 0))), arm_problem(np, 3)["training"], _STAGE2, chunk_rows=16)
    host_problem = arm_problem(np, 3)
    host = ArmTraits.from_tiles(np, _tiles(host_problem), host_problem["training"], _STAGE2, chunk_rows=16)
    np.testing.assert_array_equal(device.centers, host.centers)
    np.testing.assert_array_equal(device.scales, host.scales)
    np.testing.assert_array_equal(device.live, host.live)


def test_the_cuda_arm_pass_is_its_numpy_reference() -> None:
    traits = tuple(range(_TRAITS))
    with cupy.cuda.Device(0):
        problem = arm_problem(cupy, 5, devices=(0, 0, 0))
        kernels = CudaPanelKernels(cupy, int(problem["grid"].shape[1]))
        device = _sweeps(cupy, kernels, problem, traits, 3)
        reference = _sweeps(np, HostPanelKernels(kernels.threads(problem["masks"].shape[1])), arm_problem(np, 5, devices=(0, 0, 0)), traits, 3)
    for (residuals, totals, squares, state), (reference_residuals, reference_totals, reference_squares, reference_state) in zip(device, reference):
        for trait in traits:
            live = problem["live"][:, trait]
            for name in MOMENTS:
                expected = reference_state[name][:, trait]
                np.testing.assert_allclose(state[name][:, trait], expected, rtol=1e-10, atol=1e-12 * (1.0 + np.max(np.abs(expected))), err_msg=name)
                assert not np.any(state[name][~live, trait]), name
            np.testing.assert_allclose(residuals[trait], reference_residuals[trait], rtol=1e-10, atol=1e-11 * np.linalg.norm(reference_residuals[trait]))
            assert not np.any(residuals[trait][~problem["training"][trait]])
            assert squares[trait] == pytest.approx(reference_squares[trait], rel=1e-10)
            np.testing.assert_allclose(totals[trait], reference_totals[trait], rtol=1e-10, atol=1e-10 * abs(reference_totals[trait, 2]))


def test_a_trait_s_cuda_sweep_is_the_same_whichever_traits_share_the_pass() -> None:
    with cupy.cuda.Device(0):
        problem = arm_problem(cupy, 7, devices=(0, 0, 0))
        kernels = CudaPanelKernels(cupy, int(problem["grid"].shape[1]))
        together = _sweeps(cupy, kernels, problem, tuple(range(_TRAITS)), 2)
        for trait in (0, _TRAITS - 1):
            for (residuals, totals, squares, state), (alone_residuals, alone_totals, alone_squares, alone_state) in zip(
                together, _sweeps(cupy, kernels, problem, (trait,), 2),
            ):
                np.testing.assert_array_equal(residuals[trait], alone_residuals[0])
                np.testing.assert_array_equal(totals[trait], alone_totals[0])
                assert squares[trait] == alone_squares[0]
                for name in MOMENTS:
                    np.testing.assert_array_equal(state[name][:, trait], alone_state[name][:, 0])


@pytest.mark.skipif(DEVICE_COUNT < 2, reason="needs two CUDA devices")
def test_an_arm_pass_over_two_devices_is_the_one_device_pass() -> None:
    """A block's panels run on the device that holds its codes and hand the residuals on in order, so the moments and
    residuals are the one-device pass's to the bit; the ELBO pieces are summed per device, so their totals to rounding."""
    with cupy.cuda.Device(0):
        problem = arm_problem(cupy, 9, devices=(0, 0, 0))
        kernels = CudaPanelKernels(cupy, int(problem["grid"].shape[1]))
        one = _sweeps(cupy, kernels, problem, tuple(range(_TRAITS)), 2)
        two = _sweeps(cupy, kernels, arm_problem(cupy, 9, devices=(0, 1, 0)), tuple(range(_TRAITS)), 2)
    for (residuals, totals, squares, state), (two_residuals, two_totals, two_squares, two_state) in zip(one, two):
        np.testing.assert_array_equal(two_residuals, residuals)
        np.testing.assert_array_equal(two_squares, squares)
        np.testing.assert_allclose(two_totals, totals, rtol=1e-12, atol=1e-12 * np.max(np.abs(totals)))
        for name in MOMENTS:
            np.testing.assert_array_equal(two_state[name], state[name])
