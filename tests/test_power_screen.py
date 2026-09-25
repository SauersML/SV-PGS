"""The power screen (``sv_pgs.power_screen``): sites, the prior-predictive bounds, and whole-site dropping."""

from __future__ import annotations

import numpy as np

from sv_pgs.power_screen import power_screen
from sv_pgs.scale_mixture_ep import (
    Cavity, class_log_density, derived_lattice, initial_hyperparameters, log_scale, scale_mixture_prior, tilted_moments,
)

_DRAWS = 64
_BYTES = 1 << 24


def _prior(offsets: np.ndarray, precision: float):
    count = offsets.shape[0]
    generator = np.random.default_rng(7)
    shift = precision * (generator.normal(0.0, 0.05, count) + generator.standard_normal(count) / np.sqrt(precision))
    nodes, floor, top = derived_lattice(np.full(count, precision), shift, offsets, 0.5 / _DRAWS)
    prior = scale_mixture_prior(
        class_index=np.zeros(count, dtype=np.int64), log_variance_offset=offsets, annotation_design=np.zeros((count, 0)),
        annotation_groups=(), nodes=nodes, floor=floor, top=top,
    )
    return prior, initial_hyperparameters(prior, 1.0 / precision)


def test_sites_are_the_records_group_first() -> None:
    # Members keyed by their records' group_first (three alleles at one locus, a deletion with an SNV inside it, an
    # SNV alone): a site per key, numbered in record order, its risk the sum of its members'.
    precision = 400.0
    group_first = np.array([10, 10, 10, 13, 13, 15])
    prior, hyperparameters = _prior(np.zeros(6), precision)
    screen = power_screen(prior, hyperparameters, np.full(6, precision), np.ones(6), group_first, _DRAWS, _BYTES)
    assert screen.site.tolist() == [0, 0, 0, 1, 1, 2]
    np.testing.assert_allclose(screen.site_risk, np.bincount(screen.site, weights=screen.risk))
    assert screen.record()["sites"] == 3


def test_the_bounds_hold_over_the_prior_predictive() -> None:
    # B_j bounds E[m_j^2] and E_j bounds E[Var(beta_j | y)] from below, y drawn from the prior predictive.
    precision = 400.0
    offsets = np.log(np.array([1.0, 0.1, 0.01]))
    prior, hyperparameters = _prior(offsets, precision)
    screen = power_screen(prior, hyperparameters, np.full(3, precision), np.ones(3), np.arange(3), _DRAWS, _BYTES)
    generator = np.random.default_rng(11)
    draws = 20000
    grid = prior.log_variance_grid

    weights = np.exp(class_log_density(prior, hyperparameters.coefficients)[0])
    scales = log_scale(prior, hyperparameters.coefficients)
    for member in range(3):
        node = generator.choice(grid.shape[0], size=draws, p=weights / weights.sum())
        effect = np.sqrt(np.exp(scales[member] + grid[node])) * generator.standard_normal(draws)
        estimate = effect + generator.standard_normal(draws) / np.sqrt(precision)
        rows = np.full(draws, member)
        many = scale_mixture_prior(
            class_index=np.zeros(draws, dtype=np.int64), log_variance_offset=offsets[rows], annotation_design=np.zeros((draws, 0)),
            annotation_groups=(), nodes=grid, floor=prior.kernel_floor, top=prior.kernel_top,
        )
        moments = tilted_moments(many, hyperparameters, Cavity(precision=np.full(draws, precision), shift=precision * estimate), _BYTES, np)
        # Monte Carlo standard errors of the two averages.
        square_error = np.std(moments.mean ** 2) / np.sqrt(draws)
        variance_error = np.std(moments.variance) / np.sqrt(draws)
        assert np.mean(moments.mean ** 2) <= screen.risk[member] + 4.0 * square_error
        assert np.mean(moments.variance) >= screen.spread[member] - 4.0 * variance_error
    # Rarer (smaller u_j) members carry less risk.
    assert screen.risk[0] > screen.risk[1] > screen.risk[2]


def test_a_common_locus_split_into_rare_alleles_is_kept_whole() -> None:
    # Twenty common records and three rare allele records of one locus, each allele's risk about half the budget:
    # screened one record at a time the alleles would go; as one site their risks add past the budget and it stays.
    precision = 1e4
    common = np.zeros(20)
    low, high = -30.0, 0.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        offsets = np.r_[common, np.full(3, middle)]
        prior, hyperparameters = _prior(offsets, precision)
        alone = power_screen(prior, hyperparameters, np.full(23, precision), np.ones(23), np.arange(23), _DRAWS, _BYTES)
        if alone.risk[-1] < 0.5 * alone.budget:
            low = middle
        else:
            high = middle
    assert 0.4 * alone.budget < alone.risk[-1] < 0.6 * alone.budget
    assert alone.dropped[-3:].any()
    site = np.r_[np.arange(20), np.full(3, 20)]
    whole = power_screen(prior, hyperparameters, np.full(23, precision), np.ones(23), site, _DRAWS, _BYTES)
    assert not whole.dropped[-3:].any()
    assert whole.site_risk[20] > whole.budget
