"""Stage 2 in Gram space (``gram_space``): the band's sweep is the sample-space sweep where the band holds all the LD,
its solves are the dense ones within their certificates, Stage 0's blocks read back as stored, and the fit on the
band agrees with the fit on the samples where the band is the whole Gram."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sv_pgs.config import ModelConfig
from sv_pgs.dual_solve import DualGaussian, StreamedDualSource
from sv_pgs.full_data_fit import block_grams, covariate_residual_variance, fit_full_data, stage0_lattice
from sv_pgs.genotype_statistics import DosageStoreTileSource, compute_genotype_statistics
from sv_pgs.gram_space import GramBand, GramGaussian, pass_costs
from sv_pgs.mean_field import _sweep as sample_sweep
from sv_pgs.scale_mixture_ep import scale_mixture_prior
from sv_pgs.store_block_source import StoreGenotypeBlockSource
from tests.test_full_data_fit import _BLOCK_CAP, _DRAWS, _SAMPLES, _TRAINING, _WORKSPACE_BYTES, _budget, _store

_EPSILON = float(np.finfo(np.float64).eps)


def _banded_design(generator: np.random.Generator, widths: list[int], segment: int) -> np.ndarray:
    """Columns of block b supported on sample segments b and b + 1, each +-1 with equal counts there and 0 elsewhere:
    centred, so blocks two apart share no sample and are exactly orthogonal, while neighbours share a segment."""
    samples = segment * (len(widths) + 1)
    columns = []
    for block, width in enumerate(widths):
        support = np.arange(block * segment, (block + 2) * segment)
        for _ in range(width):
            values = np.zeros(samples)
            signs = np.repeat([-1.0, 1.0], support.shape[0] // 2)
            values[support] = generator.permutation(signs)
            columns.append(values)
    return np.column_stack(columns)


def _prior_arrays(generator: np.random.Generator, members: int):
    grid = np.linspace(-6.0, 1.0, 9)
    weights = generator.dirichlet(np.ones(grid.shape[0]))
    log_density = np.log(weights)[None, :]
    scales = generator.normal(-1.0, 0.3, size=members)
    return grid, log_density, scales


def test_band_sweep_is_the_sample_sweep_where_the_band_holds_all_the_ld() -> None:
    generator = np.random.default_rng(3)
    widths = [5, 7, 4, 6]
    design = _banded_design(generator, widths, 12)
    # A tie: the last group's column appears twice more, once negated, as members of its block.
    groups = np.concatenate([np.arange(design.shape[1]), [design.shape[1] - 1, design.shape[1] - 1]])
    signs = np.concatenate([np.ones(design.shape[1]), [1.0, -1.0]])
    gram = design.T @ design
    starts = np.concatenate([[0], np.cumsum(widths)])
    outcome = design @ (generator.normal(size=design.shape[1]) * 0.4) + generator.normal(size=design.shape[0])
    band = GramBand.from_arrays(
        within=[gram[starts[b]:starts[b + 1], starts[b]:starts[b + 1]] for b in range(len(widths))],
        cross=[gram[starts[b]:starts[b + 1], starts[b + 1]:starts[b + 2]] for b in range(len(widths) - 1)],
        scores=design.T @ outcome, target_square=float(outcome @ outcome), sample_count=design.shape[0],
        residual_dimension=float(design.shape[0]), working_bytes=1 << 20,
    )
    grid, log_density, scales = _prior_arrays(generator, groups.shape[0])
    block_of_group = np.searchsorted(starts, groups, side="right") - 1
    member_blocks = tuple(generator.permutation(np.flatnonzero(block_of_group == b)) for b in range(len(widths)))
    order = np.concatenate(member_blocks)
    class_index = np.zeros(groups.shape[0], dtype=np.int64)
    noise = 0.8
    band_state = [np.zeros(groups.shape[0]) for _ in range(5)]
    sample_state = [np.zeros(groups.shape[0]) for _ in range(5)]
    residual = outcome.copy()
    member_design = np.asfortranarray(design[:, groups] * signs[None, :])
    log_node_variance = scales[:, None] + grid[None, :]
    for _sweep_index in range(4):
        result = band.sweep(
            np, member_blocks=member_blocks, group=groups, sign=signs, class_index=class_index, log_density=log_density, scales=scales, grid=grid,
            noise=noise, mean=band_state[0], variance=band_state[1], shift=band_state[2], third=band_state[3], fourth=band_state[4],
        )
        divergence, weighted, residual_square, _sizes = sample_sweep(
            member_design, np.sum(member_design * member_design, axis=0), np.arange(groups.shape[0]), class_index, log_density,
            np.exp(log_node_variance), log_node_variance, noise, sample_state[0], residual, sample_state[1], sample_state[2], sample_state[3],
            sample_state[4], order,
        )
        # Update for update: each member's projection is x_j'r to the rounding of a band's sums (<= |band| terms).
        scale = np.max(np.abs(band_state[0])) + np.max(np.abs(sample_state[0]))
        np.testing.assert_allclose(band_state[0], sample_state[0], rtol=0, atol=64 * band.band_width * _EPSILON * max(scale, 1.0))
        np.testing.assert_allclose(result.residual_square, residual_square, rtol=0, atol=64 * band.band_width * _EPSILON * result.residual_size)
        np.testing.assert_allclose(result.divergence, divergence, rtol=64 * band.band_width * _EPSILON)
        np.testing.assert_allclose(result.weighted_variance, weighted, rtol=64 * band.band_width * _EPSILON)


def _solver(generator: np.random.Generator, precision: np.ndarray) -> tuple[GramGaussian, np.ndarray, float]:
    widths = [6, 9, 5]
    design = _banded_design(generator, widths, 10)
    gram = design.T @ design
    starts = np.concatenate([[0], np.cumsum(widths)])
    band = GramBand.from_arrays(
        within=[gram[starts[b]:starts[b + 1], starts[b]:starts[b + 1]] for b in range(len(widths))],
        cross=[gram[starts[b]:starts[b + 1], starts[b + 1]:starts[b + 2]] for b in range(len(widths) - 1)],
        scores=np.zeros(gram.shape[0]), target_square=0.0, sample_count=design.shape[0], residual_dimension=float(design.shape[0]),
        working_bytes=8 * gram.shape[0] * 4,
    )
    samples = design.shape[0]
    solver = GramGaussian(band, training=np.ones((samples, 1)), targets=np.zeros((samples, 1)), covariates=np.ones((samples, 1)))
    noise = 1.3
    solver.iterate(site_precision=precision[:, None], site_shift=np.zeros((gram.shape[0], 1)), noise_variance=np.array([noise]))
    return solver, gram, noise


def test_band_posterior_solve_is_the_dense_solve_within_its_certificate() -> None:
    generator = np.random.default_rng(5)
    count = 20
    precision = generator.uniform(0.05, 2.0, size=count)
    solver, gram, noise = _solver(generator, precision)
    matrix = gram / noise + np.diag(precision)
    right = generator.normal(size=(count, 3))
    bound = np.array([1e-3, 1e-6, 1e-9])
    solution, certificate = solver.posterior_solve(right, 0, bound)
    exact = np.linalg.solve(matrix, right)
    errors = np.sqrt(np.einsum("ik,ij,jk->k", solution - exact, matrix, solution - exact))
    assert np.all(errors <= certificate * (1 + 1e-6) + 1e-12)
    assert np.all((certificate <= bound) | (certificate <= 1e3 * _EPSILON * np.linalg.norm(right, axis=0)))


def test_band_posterior_solve_eliminates_non_positive_sites_exactly() -> None:
    generator = np.random.default_rng(6)
    count = 20
    precision = generator.uniform(0.5, 2.0, size=count)
    # A linear response's sites: two of them non-positive, with A still nonsingular.
    precision[[3, 11]] = [-0.05, 0.0]
    solver, gram, noise = _solver(generator, precision)
    matrix = gram / noise + np.diag(precision)
    right = generator.normal(size=(count, 2))
    solution, certificate = solver.posterior_solve(right, 0, np.array([1e-9, 1e-9]))
    np.testing.assert_allclose(solution, np.linalg.solve(matrix, right), rtol=0, atol=1e-6 * np.max(np.abs(np.linalg.solve(matrix, right))))
    assert np.all(np.isfinite(certificate))


def test_store_block_reads_are_the_mapped_blocks(tmp_path: Path) -> None:
    store, covariate, targets, _genetic = _store(tmp_path / "store", 7)
    training = np.arange(_TRAINING)
    statistics = compute_genotype_statistics(
        DosageStoreTileSource(store, np.arange(store.n_variants)), training, np.column_stack([np.ones(_TRAINING), covariate[training]]),
        targets[training, None], ModelConfig(), _budget(), _BLOCK_CAP, tmp_path / "ld",
    )
    ld = statistics.ld
    assert ld.block_count > 2
    for block in range(ld.block_count):
        np.testing.assert_array_equal(ld.read_gram(block), ld.block(block).projected_gram)
        if ld.has_adjacent(block):
            np.testing.assert_array_equal(ld.read_adjacent(block), ld.adjacent_block(block))
        else:
            assert ld.adjacent_block(block) is None
    band = GramBand(statistics, 1 << 20)
    np.testing.assert_array_equal(band.squares, np.concatenate([np.diagonal(ld.block(b).projected_gram).astype(np.float64) for b in range(ld.block_count)]))


def _fits(tmp_path: Path, block_cap: int, seed: int = 7):
    store, covariate, targets, genetic = _store(tmp_path / "store", seed)
    training = np.arange(_TRAINING)
    training_covariates = np.column_stack([np.ones(_TRAINING), covariate[training]])
    statistics = compute_genotype_statistics(
        DosageStoreTileSource(store, np.arange(store.n_variants)), training, training_covariates, targets[training, None], ModelConfig(), _budget(),
        block_cap, tmp_path / "ld",
    )
    member_count = statistics.active_rows.shape[0]
    store_covariates = np.column_stack([np.ones(_SAMPLES), covariate])
    mask = np.zeros((_SAMPLES, 1))
    mask[training, 0] = 1.0
    offsets = np.zeros(member_count)
    classes = (np.arange(member_count) % 5 == 0).astype(np.int64)
    start_noise = float(covariate_residual_variance(targets[:, None], mask, store_covariates)[0])
    nodes, floor, top = stage0_lattice(statistics, 0, start_noise, offsets, 0.5 / _DRAWS)
    prior = scale_mixture_prior(
        class_index=classes, log_variance_offset=offsets, annotation_design=np.zeros((member_count, 0)), annotation_groups=(),
        nodes=nodes, floor=floor, top=top,
    )
    source = StreamedDualSource(StoreGenotypeBlockSource.from_statistics(store, statistics, _budget(), _WORKSPACE_BYTES))
    sample_space = DualGaussian(
        source=source, training=mask, targets=targets[:, None], offsets=np.zeros((_SAMPLES, 1)), covariates=store_covariates,
        grams=block_grams(statistics, start_noise), probe_count=_DRAWS, seed=11,
    )
    band = GramBand(statistics, 1 << 22)
    gram_space = GramGaussian(band, training=mask, targets=targets[:, None], covariates=store_covariates, probe_count=_DRAWS)
    exact_space = DualGaussian(
        source=source, training=mask, targets=targets[:, None], offsets=np.zeros((_SAMPLES, 1)), covariates=store_covariates,
        grams=block_grams(statistics, start_noise), probe_count=_DRAWS, seed=11,
    )
    sample_seconds, gram_seconds = pass_costs(band, source, np)
    assert sample_seconds > 0.0 and gram_seconds > 0.0
    fits = [
        fit_full_data(gaussian=gaussian, statistics=statistics, prior=prior, draw_count=_DRAWS, working_bytes=1 << 22, seed=13, inference="mean_field", band=route_band)
        for gaussian, route_band in ((sample_space, None), (gram_space, None), (exact_space, band))
    ]
    return fits, statistics, store, genetic


def _held_out_predictions(fits, statistics, store) -> list[np.ndarray]:
    signed = store.read_codes(0, store.n_variants).astype(np.float64) - 127.0
    standardized = (signed[statistics.active_rows] - statistics.means[:, None]) / statistics.scales[:, None]
    return [standardized.T[_TRAINING:_SAMPLES] @ fit.member_mean[:, 0] for fit in fits]


def _exact_is_the_sample_fit(sample_fit, exact_fit, statistics, store) -> None:
    """The exact route's fit is the sample-space fit's model: no far field left out, its fixed points solved to the
    fit's resolution, and its predictions those of the sample-space fit. The two reach their fixed points by different
    paths (the band's sweeps against the far field held for a round, against the samples' Gauss-Seidel), so where the
    mean field has several modes they may settle in different ones, and their outer loops end at different gains."""
    assert exact_fit.certificate.far_field[0] == 0.0 and exact_fit.certificate.budget_unresolved[0] == 0
    assert exact_fit.certificate.mean_move[0] <= exact_fit.certificate.draw_tolerance[0]
    sample, exact = _held_out_predictions([sample_fit, exact_fit], statistics, store)
    print("exact route: remaining gains", sample_fit.certificate.remaining_gain[0], exact_fit.certificate.remaining_gain[0], "prediction correlation", np.corrcoef(sample, exact)[0, 1])
    assert np.corrcoef(sample, exact)[0, 1] > 0.99


@pytest.mark.slow  # three mean-field fits of the synthetic store: about two minutes on acl42
def test_band_fit_is_near_the_sample_fit_with_one_block_per_chromosome(tmp_path: Path) -> None:
    # One block per chromosome, and the chromosomes' cross-Gram is chance LD only: the band is G but for the chance
    # coupling of the two chromosomes and float32's rounding of the stored Gram.
    (sample_fit, band_fit, exact_fit), statistics, store, _genetic = _fits(tmp_path, 256)
    assert statistics.ld.block_count == 2
    _exact_is_the_sample_fit(sample_fit, exact_fit, statistics, store)
    for fit in (sample_fit, band_fit):
        assert fit.certificate.budget_unresolved[0] == 0
        assert fit.certificate.mean_move[0] <= fit.certificate.draw_tolerance[0]
    assert band_fit.certificate.far_field[0] > 0.0 and sample_fit.certificate.far_field[0] == 0.0
    difference = np.linalg.norm(band_fit.member_mean - sample_fit.member_mean) / np.linalg.norm(sample_fit.member_mean)
    # The chance coupling between the chromosomes is O(1/sqrt(n)) per pair: the two fits differ at that order.
    assert difference < 4.0 / np.sqrt(_TRAINING)


@pytest.mark.slow  # three mean-field fits of the synthetic store: about two minutes on acl42
def test_band_fit_on_many_blocks_predicts_as_the_sample_fit(tmp_path: Path) -> None:
    (sample_fit, band_fit, exact_fit), statistics, store, genetic = _fits(tmp_path, _BLOCK_CAP)
    assert statistics.ld.block_count > 2
    _exact_is_the_sample_fit(sample_fit, exact_fit, statistics, store)
    held_out = np.arange(_TRAINING, _SAMPLES)
    predictions = _held_out_predictions([sample_fit, band_fit], statistics, store)
    accuracy = [np.corrcoef(values, genetic[held_out])[0, 1] for values in predictions]
    assert accuracy[1] > 0.5
    assert np.corrcoef(predictions[0], predictions[1])[0, 1] > 0.9


@pytest.mark.slow  # two engine fits of a synthetic chromosome: about 40 s on acl42
def test_band_route_certifies_as_the_sample_route_where_the_band_is_the_whole_gram(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # One chromosome in one LD block: the band is G up to float32's rounding of the stored Gram, so the whole fit, its
    # outer loop and certificate included, is the sample-space fit's to that rounding.
    from sv_pgs import fit_model, stage2_wiring
    from sv_pgs.config import TraitType
    from sv_pgs.dosage_store import DosageStore
    from tests.stage0_support import mosaic_codes
    from tests.test_dosage_store import _write_store

    generator = np.random.default_rng(7)
    codes = mosaic_codes(generator, _SAMPLES, 240)
    _write_store(tmp_path / "store", [{"chr22": np.rint(codes.astype(np.float64) / 127.0 * 1000.0).astype(np.int64)}])
    store = DosageStore.open(tmp_path / "store")
    dosage = store.read_codes(0, store.n_variants).astype(np.float64).T / 127.0
    causal = generator.choice(dosage.shape[1], size=15, replace=False)
    effects = np.zeros(dosage.shape[1])
    effects[causal] = generator.standard_normal(15)
    genetic = (dosage - dosage.mean(axis=0)) @ effects
    genetic *= np.sqrt(0.5) / np.std(genetic)
    covariate = generator.standard_normal(_SAMPLES)
    targets = 0.3 * covariate + genetic + np.sqrt(0.5) * generator.standard_normal(_SAMPLES)
    training = np.arange(_SAMPLES) < _TRAINING
    models = {}
    for route, costs in (("samples", (0.0, np.inf)), ("gram", (np.inf, 0.0))):
        monkeypatch.setattr(stage2_wiring, "pass_costs", lambda band, source, xp, costs=costs: costs)
        (tmp_path / route).mkdir()
        models[route] = fit_model.fit(fit_model.FitRequest(
            store=store, store_columns=np.arange(_SAMPLES, dtype=np.int64), covariates=covariate[:, None], covariate_names=("covariate",),
            covariate_columns=np.ones((1, 1), dtype=bool), targets=np.where(training, targets, np.nan)[:, None], training=training[:, None],
            model_names=("trait",), trait_types=(TraitType.QUANTITATIVE,), research_ids=tuple(f"person{index}" for index in range(_SAMPLES)),
            log_variance_offset=None, budget=_budget(), work_dir=tmp_path / route, seed=3,
        ))
    samples, gram = models["samples"].certificate, models["gram"].certificate
    assert gram["far_field"][0] == 0.0
    assert samples["remaining_gain"][0] <= 0.5 / fit_model.DRAW_COUNT and gram["remaining_gain"][0] <= 0.5 / fit_model.DRAW_COUNT
    assert samples["outer_iterations"][0] == gram["outer_iterations"][0]
    # Float32's rounding of the stored Gram, carried through the fit: the gains agree far inside the tolerance.
    assert abs(samples["remaining_gain"][0] - gram["remaining_gain"][0]) <= 1e-3 * 0.5 / fit_model.DRAW_COUNT
    coefficients = [np.asarray(models[route].scoring[0].coefficients) for route in ("samples", "gram")]
    assert np.linalg.norm(coefficients[1] - coefficients[0]) <= 1e-4 * np.linalg.norm(coefficients[0])
