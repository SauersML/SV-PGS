"""The derived work budgets of the ascent and mixture loops (``full_data_fit.SweepBudget``, ``noise_floor``,
``elbo_ceiling``; the mixtures' K-order budgets): the bounds are what they claim, and a loop that spends its budget
ends unresolved and says so rather than run on or report success."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.full_data_fit import SweepBudget, elbo_ceiling, noise_floor


def test_the_noise_floor_is_the_fixed_point_of_the_variance_floor_map():
    generator = np.random.default_rng(0)
    squares, smallest = generator.uniform(10.0, 50.0, 400), generator.uniform(1e-4, 1e-2, 400)
    floor = noise_floor(squares, smallest, 100.0, entry_noise=10.0)
    weights = squares * smallest
    # s* solves sum_j w_j / (s + w_j) = n'.
    assert abs(float(np.sum(weights / (floor + weights))) - 100.0) <= 1e-9 * 100.0
    # From below the map raises the noise, from above it stays above s*: the floor is min(entry, s*).
    assert noise_floor(squares, smallest, 100.0, entry_noise=floor / 3.0) == floor / 3.0
    # No more members than residual dimensions: no floor from the prior.
    assert noise_floor(squares[:50], smallest[:50], 100.0, entry_noise=1.0) == 0.0
    assert elbo_ceiling(100.0, 0.0) == np.inf and elbo_ceiling(100.0, 1.0) == -50.0 * np.log(2.0 * np.pi)


def test_a_sweep_budget_counts_iterations_against_the_ceiling_in_its_smallest_step():
    budget = SweepBudget(ceiling=10.0, tolerance=0.5)
    assert not budget.exhausted(0.0, 1.0)  # the first ELBO: the budget's origin
    spent = [budget.exhausted(0.0, 1.0) for _ in range(21)]
    # (C - E_1) / min(rho, tolerance) = 20 iterations past the first; the 21st past it is over.
    assert not any(spent[:20]) and spent[20]
    assert not any(SweepBudget(ceiling=np.inf, tolerance=0.5).exhausted(0.0, 1.0) for _ in range(100))


def test_a_mean_field_solve_that_spends_its_budget_is_refused_as_unresolved(monkeypatch: pytest.MonkeyPatch):
    from sv_pgs import mean_field
    from tests.test_mean_field import _oracle

    _statistics, _prior, start, oracle = _oracle(13)
    # A ceiling below every ELBO leaves no iteration to spend: the solve ends at its second.
    monkeypatch.setattr(mean_field, "elbo_ceiling", lambda _dimension, _floor: -1e300)
    (point,) = oracle([start])
    assert point is None
    assert any("work budget" in refusal and "unresolved" in refusal for refusal in oracle.refusals)
    assert oracle.profile["unresolved_solves"] >= 1


def test_the_small_n_mode_mixture_ends_unresolved_at_its_budget_of_k_orders(monkeypatch: pytest.MonkeyPatch):
    """Every order refused: the search stops after K attempts and reports unresolved (it once looped for ever)."""
    from sv_pgs import small_n
    from sv_pgs.mean_field import MeanFieldFixedPoints
    from tests.test_mean_field import _oracle

    statistics, prior, start, oracle = _oracle(13)
    (point,) = oracle([start])
    assert point is not None
    solve = small_n._SmallNSolve(oracle=oracle, outer=None, hyperparameters=start, inference="mean_field", draw_count=8)
    calls = []

    def refuse(self, hyperparameters):
        calls.append(1)
        return [None]

    monkeypatch.setattr(MeanFieldFixedPoints, "__call__", refuse)
    components, unresolved = small_n._mode_mixture(statistics, prior, [solve], [np.asarray(oracle.mean)], float(oracle.noise), 8, 1 << 24, 0)
    assert unresolved and len(calls) == 8 and len(components) == 1


def test_the_small_n_mode_mixture_searches_its_whole_budget_of_orders(monkeypatch: pytest.MonkeyPatch):
    """All K permutations are tried in the generator's sequence, even after a repeat, with the same alternating
    starts and normalized evidence weights."""
    from sv_pgs import small_n
    from sv_pgs.mean_field import MeanFieldFixedPoints
    from tests.test_mean_field import _oracle

    statistics, prior, start, oracle = _oracle(13)
    (point,) = oracle([start])
    assert point is not None
    draw_count = 8
    solve = small_n._SmallNSolve(oracle=oracle, outer=None, hyperparameters=start, inference="mean_field", draw_count=draw_count)
    initialize, called = MeanFieldFixedPoints.__init__, MeanFieldFixedPoints.__call__
    orders, starts, calls = [], [], []

    def recording(self, *arguments, **keywords):
        orders.append(keywords["order"].copy())
        starts.append(keywords["start_means"])
        initialize(self, *arguments, **keywords)

    def counted(self, hyperparameters):
        calls.append(1)
        return called(self, hyperparameters)

    monkeypatch.setattr(MeanFieldFixedPoints, "__init__", recording)
    monkeypatch.setattr(MeanFieldFixedPoints, "__call__", counted)
    components, _unresolved = small_n._mode_mixture(
        statistics, prior, [solve], [np.asarray(oracle.mean)], float(oracle.noise), draw_count, 1 << 24, 0,
    )
    generator = np.random.default_rng([0, 1])
    np.testing.assert_array_equal(orders, [generator.permutation(oracle.mean.shape[0]) for _ in range(draw_count)])
    assert len(calls) == draw_count
    for index, means in enumerate(starts):
        if index % 2 == 0:
            np.testing.assert_array_equal(means, (oracle.mean,))
        else:
            assert means == ()
    weights = small_n._mixture_weights(components)
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-12)
    assert np.all(np.isfinite(weights)) and np.all(weights >= 0.0)


@pytest.mark.parametrize(("orders", "unresolved"), [
    ([2.0], False),
    ([4.0], True),
    ([0.0, 8.0, None], True),
    ([0.0, None, 8.0], True),
    ([8.0, 8.0, None], False),
    ([8.0, None, 8.0], False),
    ([None, None, None], True),
])
def test_the_small_n_mode_mixture_settles_from_its_last_admitted_order(monkeypatch: pytest.MonkeyPatch, orders, unresolved):
    """An early repeat leaves later modes to search; a refused last order leaves the last admitted move standing,
    and a search with no admitted order is unresolved."""
    from types import SimpleNamespace
    from sv_pgs import mean_field, small_n

    calls = []

    class Oracle:
        noise = 1.0
        profile = {"elbo": 0.0}

        def __init__(self, *arguments, **keywords):
            self.mean = np.zeros(1)

        def __call__(self, hyperparameters):
            value = orders[len(calls)]
            calls.append(value)
            if value is None:
                return [None]
            self.mean[:] = value
            return [self]

    monkeypatch.setattr(mean_field, "MeanFieldFixedPoints", Oracle)
    statistics = SimpleNamespace(design=SimpleNamespace(image=lambda move: move))
    oracle = Oracle()
    solve = small_n._SmallNSolve(oracle=oracle, outer=None, hyperparameters=None, inference="mean_field", draw_count=len(orders))
    components, ended = small_n._mode_mixture(statistics, None, [solve], [oracle.mean], oracle.noise, len(orders), 1 << 24, 0)
    assert calls == orders and ended == unresolved
    np.testing.assert_allclose(small_n._mixture_weights(components).sum(), 1.0, rtol=1e-12)
