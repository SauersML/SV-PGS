"""The batched sweep's check (``device_sweep._SweepCheck``: each member's step against the sweep's own inputs to it):
the whole sweep on a configuration's first layout in a process, the first block on each later layout; a failed check
or CUDA error leaves the reference route's state untouched; a fault inside the kernels' chain is caught, also one that
shows only on a later member order; and a failure line carries no value."""
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


def test_a_configuration_s_first_layout_is_checked_whole_and_each_later_one_on_its_first_block(monkeypatch):
    messages = []
    monkeypatch.setattr(sweep, "log", messages.append)
    layouts = []
    for _fit in range(2):
        problem, layout, state = _case()
        for _pass in range(3):
            _run(problem, layout, state)
        assert layout.checked and not layout.disabled
        layouts.append(layout)
    # one check per layout, on its first sweep: the whole sweep, then the later layout's first block
    assert [message.split(")")[0] for message in messages] == [
        "batched sweep check passed (whole sweep", "batched sweep check passed (first block of a new member order",
    ]
    worst = float(messages[0].split("largest error ")[1].split(" ")[0])
    assert 0.0 < worst < 1.0
    # Another configuration (one model, not three) is checked whole; so is the first one in a new process.
    problem, layout, state = _case()
    batched_sweep(np, sweep.HostPanelKernels(2 * sweep.PANEL), problem, layout, range(1), {name: values[:, :1].copy() for name, values in state.items()},
                  problem["residuals"][:1].copy(), sweep.PanelGrams())
    sweep._VERDICTS.clear()
    problem, layout, state = _case()
    _run(problem, layout, state)
    assert [message.split(")")[0].split("(")[1] for message in messages[2:]] == ["whole sweep", "whole sweep"]


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
    assert ("member" in messages[0]) == (field in sweep.MOMENTS)
    # Another layout of the same configuration takes the reference route without sweeping.
    monkeypatch.setattr(sweep, "_sweep_models", lambda *args, **kwargs: pytest.fail("swept after a failed check"))
    other_problem, other, other_state = _case()
    with pytest.raises(sweep.NoCodes, match="earlier in this process"):
        _run(other_problem, other, other_state)
    assert other.disabled


class _Faulty(sweep.HostPanelKernels):
    """The numpy kernels with one fault on the third launch that projects a panel."""

    def __init__(self, fault, launch=3):
        super().__init__(2 * sweep.PANEL)
        self.fault, self.launch, self.launches = fault, launch, 0

    def panel_step(self, blocks, threads, models, arguments):
        projected_width = int(arguments[11])
        self.launches += projected_width > 0
        third = projected_width > 0 and self.launches == self.launch
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
        if third and self.fault == "leak":
            # a wrong pointer: a sample's residual lands in a member's fourth moment
            self.leaked = float(residuals[0, np.flatnonzero(residuals[0])[0]])
            arguments[37][projected + 1, 0] = self.leaked


@pytest.mark.parametrize("fault, quantities", [("residual", ("shift", "residual")), ("gram", ("shift",)), ("variance", ("variance",)), ("mean", ("shift", "mean"))])
def test_a_fault_inside_the_kernels_chain_fails_the_check(monkeypatch, fault, quantities):
    # three blocks of several panels each, over two devices
    problem, layout, state = _larger()
    messages = []
    monkeypatch.setattr(sweep, "log", messages.append)
    with pytest.raises(sweep.NoCodes, match="disagreed"):
        batched_sweep(np, _Faulty(fault), problem, layout, range(3), state, problem["residuals"].copy(), sweep.PanelGrams())
    assert all(not values.any() for values in state.values())
    assert any(messages[0].startswith(f"batched sweep disabled (check failed: {quantity} of model 0") for quantity in quantities), messages[0]


def _larger(seed=7, **options):
    problem = sweep_problem(np, seed, **options)
    layout = _layout(np, problem)
    count = sum(block.members.size for block in layout.blocks)
    return problem, layout, {name: np.zeros((count, 3)) for name in sweep.MOMENTS}


def test_a_fault_on_a_later_member_order_fails_that_layout_s_check(monkeypatch):
    """A kernel right on the configuration's first layout and wrong on a later one (an indexing fault that a member
    order reaches): the later layout's first-block check catches it before its moments are committed."""
    messages = []
    monkeypatch.setattr(sweep, "log", messages.append)
    problem, first, state = _larger()
    batched_sweep(np, _Faulty(None), problem, first, range(3), state, problem["residuals"].copy(), sweep.PanelGrams())
    problem, later, state = _larger()
    later.blocks[0] = sweep._BlockLayout(later.blocks[0].device, later.blocks[0].offset, later.blocks[0].start, later.blocks[0].members[::-1].copy(),
                                         later.blocks[0].pieces)
    with pytest.raises(sweep.NoCodes, match="disagreed"):
        batched_sweep(np, _Faulty("gram", launch=1), problem, later, range(3), state, problem["residuals"].copy(), sweep.PanelGrams())
    assert all(not values.any() for values in state.values())
    assert messages[1].startswith("batched sweep disabled (check failed: shift of model 0, member ")


def test_a_failure_line_carries_no_value(monkeypatch):
    messages = []
    monkeypatch.setattr(sweep, "log", messages.append)
    problem, layout, state = _larger()
    kernels = _Faulty("leak")
    with pytest.raises(sweep.NoCodes, match="disagreed"):
        batched_sweep(np, kernels, problem, layout, range(3), state, problem["residuals"].copy(), sweep.PanelGrams())
    assert messages[0].startswith("batched sweep disabled (check failed: fourth of model 0, member ")
    assert "times its tolerance" in messages[0] and abs(kernels.leaked) > 1e-3
    for digits in (2, 3, 6, 17):
        assert f"{abs(kernels.leaked):.{digits}g}" not in messages[0]


def test_a_strong_effect_s_moments_are_held_to_their_own_size():
    """One unit-variance node, n = 44,000, h = 8,800: q is N(0.2, 1 / 44,001), its fourth central moment 3 c^2 =
    1.5e-9. A fourth moment of 0 (or a variance of 0) is far outside its tolerance."""
    (mean, variance, third, fourth, pieces), scales = sweep._tilted(
        np.array([8800.0]), np.array([44000.0]), 1.0, np.zeros(1), np.zeros(1), np.zeros((1, 1)),
    )
    conditional = 1.0 / (1.0 + 44000.0)
    assert mean[0] == pytest.approx(8800.0 * conditional) and variance[0] == pytest.approx(conditional)
    assert fourth[0] == pytest.approx(3.0 * conditional ** 2)
    for values, scale in ((variance, scales[1]), (fourth, scales[3])):
        assert sweep.SWEEP_RTOL * scale[0] < 1e-3 * values[0]


def test_a_column_nearly_spanned_by_a_covariate_passes_the_check(monkeypatch):
    """The kernels form x_j' r from the unprojected codes; a column that a covariate nearly spans leaves a projected
    product far smaller than the rounding of the unprojected one, which the shift's tolerance carries."""
    messages = []
    monkeypatch.setattr(sweep, "log", messages.append)
    problem, layout, state = _larger(samples=20_000, collinear=True)
    batched_sweep(np, _Faulty(None), problem, layout, range(3), state, problem["residuals"].copy(), sweep.PanelGrams())
    assert messages[0].startswith("batched sweep check passed (whole sweep)")


def test_the_check_passes_every_kernel_without_a_fault_on_the_larger_problem(monkeypatch):
    problem, layout, state = _larger()
    messages = []
    monkeypatch.setattr(sweep, "log", messages.append)
    batched_sweep(np, _Faulty(None), problem, layout, range(3), state, problem["residuals"].copy(), sweep.PanelGrams())
    assert layout.checked and messages[0].startswith("batched sweep check passed (whole sweep)")


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
