"""Genome-scale EP's pieces (``full_data_fit`` EP route): the leave-block-out windows cut to the fit's budget."""

from __future__ import annotations

import numpy as np

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
        grams=block_grams(statistics, noise, working_bytes=working_bytes), probe_count=16, seed=11,
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
