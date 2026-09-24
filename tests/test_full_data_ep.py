"""Genome-scale EP's pieces (``full_data_fit`` EP route): the leave-block-out windows cut to the fit's budget."""

from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.marginal_variances import BlockGrams, refined_grams, window_width, window_working_bytes


def _banded(seed: int, sizes=(15, 25, 9)) -> tuple[np.ndarray, BlockGrams]:
    generator = np.random.default_rng(seed)
    count = int(sum(sizes))
    design = generator.standard_normal((120, count))
    design[:, 1:] += 0.7 * design[:, :-1]
    starts = np.concatenate([[0], np.cumsum(sizes)])
    blocks = tuple(np.arange(starts[index], starts[index + 1]) for index in range(len(sizes)))
    gram = design.T @ design
    within = tuple(gram[np.ix_(block, block)].astype(np.float32) for block in blocks)
    cross = tuple(gram[np.ix_(blocks[index], blocks[index + 1])].astype(np.float32) for index in range(len(sizes) - 1))
    return gram, BlockGrams(blocks=blocks, within=within, next_cross=cross, scale=0.5)


def test_refined_grams_are_slices_of_the_stored_grams() -> None:
    gram, grams = _banded(1)
    refined = refined_grams(grams, 4)
    assert np.array_equal(np.concatenate(refined.blocks), np.concatenate(grams.blocks))
    assert max(block.shape[0] for block in refined.blocks) <= 4
    assert len(refined.next_cross) == len(refined.blocks) - 1
    for index, block in enumerate(refined.blocks):
        np.testing.assert_allclose(refined.within_block(index), 0.5 * gram[np.ix_(block, block)], rtol=1e-6)
        if index + 1 < len(refined.blocks):
            np.testing.assert_allclose(refined.cross_block(index), 0.5 * gram[np.ix_(block, refined.blocks[index + 1])], rtol=1e-6)
    # Views of the stored arrays, not copies.
    assert np.shares_memory(refined.within[0], grams.within[0])


def test_window_width_keeps_every_window_within_the_budget() -> None:
    _gram, grams = _banded(2, sizes=(400, 300))
    for budget in (10_000, 1 << 20, 1 << 24):
        width = window_width(budget)
        refined = refined_grams(grams, width)
        assert window_working_bytes(refined) <= budget or width == 1
        # The next width up would not fit a window of three blocks at that width.
        assert window_working_bytes(refined_grams(BlockGrams(
            blocks=(np.arange(3 * (width + 1)),), within=(np.zeros((3 * (width + 1),) * 2),), next_cross=(),
        ), width + 1)) > budget


def test_streamed_double_loop_reaches_the_dense_double_loops_fixed_point(tmp_path, monkeypatch) -> None:
    """The streamed EP double loop (``_FullDataFixedPoints._double_loop``) on the full-data tests' synthetic store, with
    its first chromosome in one Stage 0 block, so the window covers the whole design and the leave-block-out variances
    are exact (Stage 0 records no LD across chromosomes, which is the windows' own approximation there), against small_n's dense double loop (``double_loop_sites``) on the same projected design, prior and noise
    (held fixed in both): the same EP fixed point, q's marginal means and variances."""
    from pathlib import Path

    from sv_pgs.config import ModelConfig
    from sv_pgs.dual_solve import DualGaussian, StreamedDualSource
    from sv_pgs.full_data_fit import _FullDataFixedPoints, block_grams, covariate_residual_variance, stage0_lattice
    from sv_pgs.genotype_statistics import DosageStoreTileSource, compute_genotype_statistics
    from sv_pgs.scale_mixture_ep import (
        Cavity, initial_hyperparameters, log_scale, moment_matched_prior_sites, scale_mixture_prior, tilted_cumulants, tilted_moments,
    )
    from sv_pgs.marginal_variances import approximation_scale
    from sv_pgs.small_n import _Design, _new_profile, double_loop_sites
    from sv_pgs.store_block_source import StoreGenotypeBlockSource
    from tests.test_full_data_fit import _SAMPLES, _TRAINING, _budget, _store

    store, covariate, targets, _genetic = _store(Path(tmp_path) / "store", 7)
    training = np.arange(_TRAINING)
    covariates = np.column_stack([np.ones(_TRAINING), covariate[training]])
    statistics = compute_genotype_statistics(
        DosageStoreTileSource(store, np.arange(int(store.chromosome_starts[1]))), training, covariates, targets[training, None], ModelConfig(), _budget(), 256,
        Path(tmp_path) / "ld",
    )
    assert statistics.ld.block_count == 1
    count = statistics.active_rows.shape[0]
    mask = np.zeros((_SAMPLES, 1))
    mask[training, 0] = 1.0
    store_covariates = np.column_stack([np.ones(_SAMPLES), covariate])
    noise = float(covariate_residual_variance(targets[:, None], mask, store_covariates)[0]) * 0.6
    draw_count = 2**14
    nodes, floor, top = stage0_lattice(statistics, 0, noise, np.zeros(count), 1.0 / 64)
    prior = scale_mixture_prior(
        class_index=np.zeros(count, dtype=np.int64), log_variance_offset=np.zeros(count), annotation_design=np.zeros((count, 0)),
        annotation_groups=(), nodes=nodes, floor=floor, top=top,
    )
    start = initial_hyperparameters(prior)
    working_bytes = 1 << 30
    source = StreamedDualSource(StoreGenotypeBlockSource.from_statistics(store, statistics, _budget(), 1 << 26))
    gaussian = DualGaussian(
        source=source, training=mask, targets=targets[:, None], offsets=np.zeros((_SAMPLES, 1)), covariates=store_covariates,
        grams=block_grams(statistics, noise, working_bytes=working_bytes, level=1.0 / draw_count), probe_count=16, seed=11,
    )
    oracle = _FullDataFixedPoints(gaussian, statistics, prior, draw_count, working_bytes, 13, [start], np.array([noise]))
    monkeypatch.setattr(oracle, "_noise", lambda _variances: oracle.noise.copy())
    monkeypatch.setattr(oracle, "_information", lambda *arguments: None)
    oracle._double_loop([start])
    streamed_mean = np.asarray(gaussian.mean)[:, 0]

    # The dense design: the kept columns standardized as Stage 0 did, masked and projected off the covariates.
    kept = np.asarray(statistics.active_rows)[np.asarray(statistics.tie_map.kept_indices)]
    codes = store.read_codes(0, store.n_variants).astype(np.float64)[kept].T - 127.0
    means = np.asarray(statistics.means)[np.asarray(statistics.tie_map.kept_indices)]
    scales = np.asarray(statistics.scales)[np.asarray(statistics.tie_map.kept_indices)]
    columns = (codes - means) / scales
    weighted = mask * store_covariates
    projector = (np.eye(_SAMPLES) - weighted @ np.linalg.pinv(weighted.T @ weighted) @ weighted.T) @ np.diag(mask[:, 0])
    design = projector @ columns
    response = projector @ targets

    def tilted(cavity_precision, cavity_shift):
        cavity = Cavity(precision=cavity_precision, shift=cavity_shift)
        moments = tilted_moments(prior, start, cavity, 10**8)
        third, fourth = tilted_cumulants(prior, start, cavity, 10**8)
        return moments.log_normalizer, moments.mean, moments.variance, third, fourth

    largest = np.exp(log_scale(prior, start.coefficients) + prior.log_variance_grid[-1])
    precision, shift = double_loop_sites(
        _Design.dense(design), noise, design.T @ response, *moment_matched_prior_sites(prior, start), tilted, largest, draw_count, 10**9, _new_profile(),
    )
    inverse = np.linalg.inv(design.T @ design / noise + np.diag(precision))
    dense_mean = inverse @ (design.T @ response / noise + shift)
    dense_variance = np.diag(inverse)
    # The streamed fixed point uses the windows' marginal variances, whose one approximation (the deterministic
    # equivalent's far-field trace, from the solver's probes) has relative error scale ||K^-1||_F / tr(K^-1)
    # (``approximation_scale``): the two EP fixed points agree within it (their variances differed by at most 0.6% here).
    scale = approximation_scale(gaussian.bulk_solves[0])
    streamed_inverse = np.linalg.inv(design.T @ design / noise + np.diag(oracle.site_precision[:, 0]))
    np.testing.assert_allclose(np.diag(streamed_inverse), dense_variance, rtol=scale)
    np.testing.assert_allclose(streamed_mean, dense_mean, rtol=scale, atol=scale * float(np.max(np.sqrt(dense_variance))))


def test_prediction_error_bound_holds_for_every_row() -> None:
    """|x'(mu_hat - mu)| <= sqrt(x'A^-1 x) ||mu_hat - mu||_A for rows x, with a perturbed mean and its exact A-norm."""
    from sv_pgs.dual_solve import prediction_error_bound

    generator = np.random.default_rng(5)
    design = generator.standard_normal((40, 12))
    precision = design.T @ design + np.diag(generator.uniform(0.5, 2.0, 12))
    error = generator.standard_normal(12) * 1e-3
    norm = float(np.sqrt(error @ precision @ error))
    rows = generator.standard_normal((25, 12))
    variance = np.einsum("ij,jk,ik->i", rows, np.linalg.inv(precision), rows)
    assert np.all(np.abs(rows @ error) <= prediction_error_bound(norm, variance) * (1.0 + 1e-12))



def test_ld_extent_is_the_lag_where_the_excess_ld_is_unresolved() -> None:
    """A design whose LD reaches exactly three variants (a moving sum of four independent columns) has extent at most
    a few lags beyond three and at least three; independent columns have extent 1 (nothing to reach)."""
    from sv_pgs.marginal_variances import ld_extent

    generator = np.random.default_rng(7)
    samples, count = 4000, 300
    noise = generator.standard_normal((samples, count + 3))
    moving = sum(noise[:, shift:shift + count] for shift in range(4))
    for design, low, high in ((moving, 3, 8), (generator.standard_normal((samples, count)), 1, 2)):
        gram = design.T @ design
        blocks = (np.arange(count // 2), np.arange(count // 2, count))
        grams = BlockGrams(
            blocks=blocks, within=tuple(gram[np.ix_(block, block)].astype(np.float32) for block in blocks),
            next_cross=(gram[np.ix_(blocks[0], blocks[1])].astype(np.float32),),
        )
        extent = ld_extent(grams, samples, 1 << 24, 1.0 / 64)
        assert low <= extent <= high, extent


def test_factored_far_field_trace_matches_the_spectral_one() -> None:
    """omega_F from Cholesky factors (``_far_field_factored``) equals ``far_field_trace`` from B's eigenvalues, and
    the factor returned is of I + omega_F B."""
    from sv_pgs.marginal_variances import BulkSolve, WindowCross, _far_field_factored, far_field_trace

    generator = np.random.default_rng(9)
    design = generator.standard_normal((50, 30))
    whitened = design.T @ design * 0.01
    solve = BulkSolve(
        site_precision=np.ones(30), resolved=np.zeros(0, np.int64), resolved_core=np.zeros((0, 0)),
        resolved_cross=WindowCross(positions=(), values=()), bulk_trace=0.7, bulk_square_trace=0.5, kernel_square_trace=0.5, sample_count=400,
    )
    omega, lower = _far_field_factored(whitened, solve, np)
    expected = far_field_trace(0.7, 0.5, 400, np.linalg.eigvalsh(whitened))
    assert omega == pytest.approx(expected, rel=1e-12)
    np.testing.assert_allclose(lower @ lower.T, np.eye(30) + omega * whitened, rtol=1e-12, atol=1e-12)


def test_window_width_at_a_huge_budget_stops_at_the_widest_block() -> None:
    assert window_width(1 << 62, np, 41_984) == 41_984
    assert window_width(1 << 62, np, 1) == 1


def test_factored_far_field_trace_from_a_warm_start_reaches_the_same_root() -> None:
    """Newton from above the root (a neighbouring window's omega) lands on the same omega_F as from omega_S."""
    from sv_pgs.marginal_variances import BulkSolve, WindowCross, _far_field_factored, far_field_trace

    generator = np.random.default_rng(11)
    design = generator.standard_normal((50, 30))
    whitened = design.T @ design * 0.01
    solve = BulkSolve(
        site_precision=np.ones(30), resolved=np.zeros(0, np.int64), resolved_core=np.zeros((0, 0)),
        resolved_cross=WindowCross(positions=(), values=()), bulk_trace=0.7, bulk_square_trace=0.5, kernel_square_trace=0.5, sample_count=400,
    )
    expected = far_field_trace(0.7, 0.5, 400, np.linalg.eigvalsh(whitened))
    for start in (0.7, 2.0 * expected, 0.9 * expected):
        omega, _lower = _far_field_factored(whitened, solve, np, start)
        assert omega == pytest.approx(expected, rel=1e-12)


def test_warm_start_zeroes_negative_sites_and_every_cavity_is_proper() -> None:
    """A mixed-sign start (as mean field leaves), after ``warm_start_sites``, becomes a start
    whose every cavity precision 1/Sigma_jj - tau_j is non-negative (dense algebra), with the non-negative sites kept."""
    from sv_pgs.full_data_fit import warm_start_sites

    generator = np.random.default_rng(13)
    design = generator.standard_normal((60, 20))
    design[:, 1] = design[:, 0] + 0.05 * generator.standard_normal(60)
    likelihood = design.T @ design / 0.8
    precision = generator.uniform(0.5, 3.0, 20)
    precision[[1, 4, 9]] = -0.4
    shift = generator.standard_normal(20)
    kept_precision, kept_shift = warm_start_sites(precision, shift)
    assert np.all(kept_precision >= 0.0)
    np.testing.assert_array_equal(kept_precision[precision >= 0.0], precision[precision >= 0.0])
    np.testing.assert_array_equal(kept_shift[precision >= 0.0], shift[precision >= 0.0])
    assert np.all(kept_shift[precision < 0.0] == 0.0)
    covariance = np.linalg.inv(likelihood + np.diag(kept_precision))
    assert np.all(1.0 / np.diag(covariance) - kept_precision >= 0.0)


def test_ld_lag_profiles_give_each_region_its_own_extent() -> None:
    """Per-block lag profiles: a region whose LD reaches three variants has an extent of at least three, an
    independent region's is one, and their rows sum to the whole design's profile (``ld_extent``)."""
    from sv_pgs.marginal_variances import extent_from_profile, ld_extent, ld_lag_profiles

    generator = np.random.default_rng(17)
    samples, count = 4000, 150
    noise = generator.standard_normal((samples, count + 3))
    linked = sum(noise[:, shift:shift + count] for shift in range(4))
    independent = generator.standard_normal((samples, count))
    design = np.column_stack([linked, independent])
    gram = design.T @ design
    blocks = (np.arange(count), np.arange(count, 2 * count))
    grams = BlockGrams(
        blocks=blocks, within=tuple(gram[np.ix_(block, block)].astype(np.float32) for block in blocks),
        next_cross=(gram[np.ix_(blocks[0], blocks[1])].astype(np.float32),),
    )
    excess, pairs = ld_lag_profiles(grams, samples, 1 << 24)
    assert excess.shape == pairs.shape == (2, count)
    assert 3 <= extent_from_profile(excess[0], pairs[0], samples, count, 1.0 / 64) <= 8
    assert extent_from_profile(excess[1], pairs[1], samples, count, 1.0 / 64) <= 2
    assert ld_extent(grams, samples, 1 << 24, 1.0 / 64) == extent_from_profile(excess.sum(axis=0), pairs.sum(axis=0), samples, 2 * count, 1.0 / 64)


def test_a_regions_extent_read_against_its_far_pairs_measured_null_is_its_true_short_one() -> None:
    """A pooled cohort of two groups whose genotype variances differ alike at every variant (1.5 and 0.5 of the pooled
    one: variance heterogeneity, no correlation) puts every unlinked pair's r^2 at kappa / n, kappa = E[v_x v_y] =
    (1.5^2 + 0.5^2) / 2 = 1.25, at every lag. Against chance's 1/n that excess over thousands of far pairs reads as LD
    and the extent runs far out; read against the far pairs' measured kappa / n (``null_scale``) the region's extent is
    its true one, its LD reaching three variants."""
    from sv_pgs.marginal_variances import extent_from_profile, ld_lag_profiles

    generator = np.random.default_rng(23)
    samples, count = 8000, 300
    scale = np.where(np.arange(samples) < samples // 2, np.sqrt(1.5), np.sqrt(0.5))
    noise = generator.standard_normal((samples, count + 3)) * scale[:, None]
    design = sum(noise[:, shift:shift + count] for shift in range(4))
    design -= design.mean(axis=0)
    gram = design.T @ design
    grams = BlockGrams(blocks=(np.arange(count),), within=(gram.astype(np.float32),), next_cross=())
    excess, pairs = ld_lag_profiles(grams, samples, 1 << 24)
    far = np.arange(excess.shape[1]) > 20
    kappa = 1.0 + float(excess[0, far].sum() / pairs[0, far].sum()) * samples
    assert 1.15 < kappa < 1.35
    assert extent_from_profile(excess[0], pairs[0], samples, count, 1.0 / 64) > 20
    assert 3 <= extent_from_profile(excess[0], pairs[0], samples, count, 1.0 / 64, null_scale=kappa) <= 8


def test_warm_start_gives_a_tied_members_negative_site_a_positive_one() -> None:
    """A negative site inside a tie group becomes its group's smallest positive site (a zero would leave the group's
    split improper: ``tied_weights``); an untied negative site is zeroed; a group with no positive site is left for the
    prior (not finite)."""
    from sv_pgs.full_data_fit import warm_start_sites
    from sv_pgs.tie_members import TieGroups, group_sites

    ties = TieGroups(group=np.array([0, 0, 1, 2, 2, 3]), sign=np.ones(6), group_count=4)
    precision = np.array([[2.0], [-0.5], [-0.3], [-1.0], [-2.0], [1.5]])
    shift = np.ones((6, 1))
    kept, kept_shift = warm_start_sites(precision, shift, ties)
    assert kept[1, 0] == 2.0 and kept_shift[1, 0] == 0.0
    assert kept[2, 0] == 0.0
    assert not np.isfinite(kept[3, 0]) and not np.isfinite(kept[4, 0])
    assert kept[0, 0] == 2.0 and kept[5, 0] == 1.5
    group_sites(ties, np.where(np.isfinite(kept), kept, 1.0)[[0, 1, 2, 3, 4, 5]], kept_shift)
