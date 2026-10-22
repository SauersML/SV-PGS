"""The first batched sweep is verified once per fit and failures leave the reference route's state untouched."""
import numpy as np
import pytest

from sv_pgs import device_sweep as sweep
from tests.test_sweep_models import sweep_problem, _layout, batched_sweep, _oracle


def _case():
    problem = sweep_problem(np, 7, devices=(0, 1), samples=30, blocks=(6, 4))
    layout = _layout(np, problem)
    count = sum(block.members.size for block in layout.blocks)
    state = {name: np.zeros((count, 3)) for name in sweep.MOMENTS}
    return problem, layout, state


def _run(problem, layout, state):
    return batched_sweep(np, sweep.HostPanelKernels(2 * sweep.PANEL), problem, layout, range(3), state,
                         problem["residuals"].copy(), sweep.PanelGrams())


def test_sweep_checks_once_per_fit_and_a_new_fit_checks_again(monkeypatch):
    original = sweep._SweepReference.block
    calls = []

    def checked(self, block, item):
        calls.append(self.layout)
        return original(self, block, item)

    monkeypatch.setattr(sweep._SweepReference, "block", checked)
    for _fit in range(2):
        problem, layout, state = _case()
        for _pass in range(3):
            _run(problem, layout, state)
        assert layout.checked and not layout.disabled
        assert sum(value is layout for value in calls) == len(layout.blocks)


@pytest.mark.parametrize("field", [*sweep.MOMENTS, "residual", "divergence", "weighted_variance", "sizes", "squares"])
def test_sweep_mismatch_disables_the_fit_without_committing_state(monkeypatch, field):
    problem, layout, state = _case()
    original = sweep._sweep_models
    messages = []
    monkeypatch.setattr(sweep, "log", messages.append)

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        if field in sweep.MOMENTS:
            kwargs["state"][field][0, 0] = np.nan
        elif field == "residual":
            result[0][0][0] += 1.0
        elif field == "squares":
            result[2][0] += 1.0
        else:
            column = {"divergence": 0, "weighted_variance": 1, "sizes": 2}[field]
            result[1][0, column] += 1.0
        return result

    monkeypatch.setattr(sweep, "_sweep_models", corrupt)
    with pytest.raises(sweep.NoCodes, match="disagreed"):
        _run(problem, layout, state)
    assert layout.disabled and all(not values.any() for values in state.values())
    with pytest.raises(sweep.NoCodes, match="disabled"):
        _run(problem, layout, state)
    assert len(messages) == 1 and "mismatch" in messages[0]


@pytest.mark.parametrize("checked", [False, True])
def test_cuda_failure_restores_even_a_partly_downloaded_sweep(monkeypatch, checked):
    problem, layout, state = _case()
    layout.checked = checked
    messages = []
    monkeypatch.setattr(sweep, "log", messages.append)

    class CUDARuntimeError(RuntimeError):
        pass

    def fault(*args, **kwargs):
        kwargs["state"]["mean"][:] = 1.0
        raise CUDARuntimeError("cudaErrorLaunchFailure")

    monkeypatch.setattr(sweep, "_sweep_models", fault)
    with pytest.raises(sweep.NoCodes) as caught:
        _run(problem, layout, state)
    assert isinstance(caught.value.__cause__, CUDARuntimeError)
    assert layout.disabled and all(not values.any() for values in state.values())
    assert "cudaErrorLaunchFailure" in messages[0]


def test_non_cuda_failure_keeps_its_exception_and_does_not_disable(monkeypatch):
    problem, layout, state = _case()

    def invalid(*args, **kwargs):
        raise ValueError("invalid layout")

    monkeypatch.setattr(sweep, "_sweep_models", invalid)
    with pytest.raises(ValueError, match="invalid layout"):
        _run(problem, layout, state)
    assert not layout.disabled and all(not values.any() for values in state.values())


def test_the_oracle_takes_the_reference_route_after_a_rejected_sweep(tmp_path, monkeypatch):
    oracle_of, starts, _statistics, _prior = _oracle(tmp_path)
    reference, oracle = oracle_of([0, 1]), oracle_of([0, 1])
    expected = {model: reference._sweep(model, starts[model]) for model in (0, 1)}
    original = sweep._sweep_models

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        kwargs["state"]["mean"][:] = np.nan
        return result

    monkeypatch.setattr(sweep, "_sweep_models", corrupt)
    assert oracle._panel_sweeps({0: starts[0], 1: starts[1]}, sweep.HostPanelKernels(2 * sweep.PANEL)) is None
    assert oracle.passes == 0 and not oracle.model_passes.any()
    assert oracle._layout is False
    for model in (0, 1):
        np.testing.assert_array_equal(oracle._sweep(model, starts[model]), expected[model])
    for name in sweep.MOMENTS:
        np.testing.assert_array_equal(getattr(oracle, name), getattr(reference, name))


def test_verification_uses_the_live_sweeps_tolerance_and_rejects_nonfinite_values():
    expected = np.array([1.0, -1.0])
    assert sweep._sweep_agrees(expected * (1.0 + sweep.SWEEP_RTOL / 2), expected)
    assert not sweep._sweep_agrees(expected * (1.0 + 2 * sweep.SWEEP_RTOL), expected)
    assert not sweep._sweep_agrees(np.array([np.nan, -1.0]), expected)
    assert not sweep._sweep_agrees(np.array([np.inf, -1.0]), expected)
    assert not sweep._sweep_agrees(expected[:, None], expected)
