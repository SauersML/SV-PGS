"""The power screen (``sv_pgs.power_screen``): sites, the prior-predictive bounds, and whole-site dropping."""

from __future__ import annotations

import numpy as np

from sv_pgs.power_screen import power_screen, site_index
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


def test_sites_join_split_alleles_overlaps_and_groups() -> None:
    # Three alleles at one position; an SNV alone; a 50 bp deletion with an SNV inside it; two records joined only by
    # their unbreakable group; a record on the next chromosome at an overlapping coordinate.
    chromosome = np.array([22, 22, 22, 22, 22, 22, 22, 22, 23])
    position = np.array([100, 100, 100, 200, 300, 310, 500, 600, 300])
    ref_length = np.array([1, 4, 9, 1, 50, 1, 1, 1, 1])
    group_first = np.array([0, 1, 2, 3, 4, 5, 6, 6, 8])
    assert site_index(chromosome, position, ref_length, group_first).tolist() == [0, 0, 0, 1, 2, 2, 3, 3, 4]


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
