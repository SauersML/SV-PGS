"""The first batched sweep of a configuration is checked once per process (``device_sweep._SweepCheck``: each
member's step against the sweep's own inputs to it), a failed check or CUDA error leaves the reference route's state
untouched, and a fault inside the kernels' chain is caught."""
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


def test_the_first_sweep_of_a_configuration_is_checked_once_per_process(monkeypatch):
    original = sweep._SweepCheck.block
    calls = []
    messages = []
    monkeypatch.setattr(sweep, "log", messages.append)

    def counted(self, block, *args):
        calls.append(self.layout)
        return original(self, block, *args)

    monkeypatch.setattr(sweep._SweepCheck, "block", counted)
    layouts = []
    for _fit in range(2):
        problem, layout, state = _case()
        for _pass in range(3):
            _run(problem, layout, state)
        assert layout.checked and not layout.disabled
        layouts.append(layout)
    # The first layout's first sweep is checked; a later layout (another fit or member order) runs unchecked.
    assert len(calls) == len(layouts[0].blocks) and all(value is layouts[0] for value in calls)
    assert len(messages) == 1 and messages[0].startswith("batched sweep check passed: largest error")
    worst = float(messages[0].split("largest error ")[1].split(" ")[0])
    assert 0.0 < worst < 1.0
    # A new process (no verdict) checks again.
    sweep._VERDICTS.clear()
    problem, layout, state = _case()
    _run(problem, layout, state)
    assert len(calls) == 2 * len(layout.blocks)


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
    assert len(messages) == 1 and messages[0].startswith("batched sweep disabled (check failed: ")
    # A sample's residual is never written to the log: only the largest error and its tolerance.
    assert ("member" in messages[0]) == (field in sweep.MOMENTS)
    # Another layout of the same configuration takes the reference route without sweeping.
    monkeypatch.setattr(sweep, "_sweep_models", lambda *args, **kwargs: pytest.fail("swept after a failed check"))
    other_problem, other, other_state = _case()
    with pytest.raises(sweep.NoCodes, match="earlier in this process"):
        _run(other_problem, other, other_state)
    assert other.disabled


class _Faulty(sweep.HostPanelKernels):
    """The numpy kernels with one fault on the third launch that projects a panel."""

    def __init__(self, fault):
        super().__init__(2 * sweep.PANEL)
        self.fault, self.launches = fault, 0

    def panel_step(self, blocks, threads, models, arguments):
        projected_width = int(arguments[11])
        self.launches += projected_width > 0
        third = projected_width > 0 and self.launches == 3
        residuals, grams, mean, variance = arguments[15], arguments[30], arguments[33], arguments[34]
        projected, panel = int(arguments[10]), int(arguments[32])
        if third and self.fault == "residual":
            # a residual move off by a millionth of the residual's size at one sample
            residuals[0, 0] += 1e-4 * np.max(np.abs(residuals[0]))
        if third and self.fault == "gram":
            grams[panel, 0, 1, 0] *= 1.0 + 1e-4
            grams[panel, 0, 0, 1] *= 1.0 + 1e-4
        super().panel_step(blocks, threads, models, arguments)
        if third and self.fault == "variance":
            variance[projected + 1, 0] *= 1.0 + 1e-5
        if third and self.fault == "mean":
            mean[projected + 2, 0] += 1e-5 * abs(mean[projected + 2, 0])


@pytest.mark.parametrize("fault, quantities", [("residual", ("shift", "residual")), ("gram", ("shift",)), ("variance", ("variance",)), ("mean", ("shift", "mean"))])
def test_a_fault_inside_the_kernels_chain_fails_the_check(monkeypatch, fault, quantities):
    # three blocks of several panels each, over two devices
    problem = sweep_problem(np, 7)
    layout = _layout(np, problem)
    count = sum(block.members.size for block in layout.blocks)
    state = {name: np.zeros((count, 3)) for name in sweep.MOMENTS}
    messages = []
    monkeypatch.setattr(sweep, "log", messages.append)
    with pytest.raises(sweep.NoCodes, match="disagreed"):
        batched_sweep(np, _Faulty(fault), problem, layout, range(3), state, problem["residuals"].copy(), sweep.PanelGrams())
    assert all(not values.any() for values in state.values())
    assert any(messages[0].startswith(f"batched sweep disabled (check failed: {quantity} of model 0") for quantity in quantities), messages[0]


def test_the_check_passes_every_kernel_without_a_fault_on_the_larger_problem(monkeypatch):
    problem = sweep_problem(np, 7)
    layout = _layout(np, problem)
    count = sum(block.members.size for block in layout.blocks)
    state = {name: np.zeros((count, 3)) for name in sweep.MOMENTS}
    messages = []
    monkeypatch.setattr(sweep, "log", messages.append)
    batched_sweep(np, _Faulty(None), problem, layout, range(3), state, problem["residuals"].copy(), sweep.PanelGrams())
    assert layout.checked and messages[0].startswith("batched sweep check passed")


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


def test_the_check_holds_each_quantity_to_its_own_scale_and_rejects_nonfinite_values():
    def outcome(actual, expected, scale):
        check = object.__new__(sweep._SweepCheck)
        check.worst, check.failure = (0.0, "nothing compared"), None
        check.compare("mean", 0, np.asarray(actual), np.asarray(expected), scale, np.array([4, 7]))
        return check

    expected = np.array([1.0, -1.0])
    within = outcome(expected + sweep.SWEEP_RTOL * 3.0, expected, 4.0)
    assert within.failure is None and within.worst == (pytest.approx(0.75), "mean of model 0, member 4")
    outside = outcome(expected + np.array([0.0, sweep.SWEEP_RTOL * 5.0]), expected, 4.0)
    assert outside.failure.startswith("mean of model 0, member 7: ")
    for value in (np.nan, np.inf):
        assert outcome(np.array([value, -1.0]), expected, 1.0).failure.startswith("mean of model 0, member 4: ")
    # an exact comparison (scale 0) admits no difference beyond the floor
    assert outcome(np.nextafter(expected, 2.0), expected, 0.0).failure is not None
    assert outcome(expected, expected, 0.0).failure is None
