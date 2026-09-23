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
