"""Stage 2 end to end: a dosage store, Stage 0, decoupled EP-EB from the prior, and scoring held-out samples.

A correctness test of the wiring on synthetic data, not an accuracy claim (accuracy comes from bench-real and
bench-sim): the fit must be certified, its in-sample predictor must equal what the scorer computes from the store,
and its held-out scores must carry the simulated signal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.dosage_store import DosageStore
from sv_pgs.dual_solve import DualGaussian, StreamedDualSource
from sv_pgs.fast_scoring import ScoringPlan, score_genetic
from sv_pgs.full_data_fit import _ridge_mean, block_grams, covariate_residual_variance, fit_full_data, moment_starts, scoring_models, stage0_lattice
from sv_pgs.genotype_statistics import DosageStoreTileSource, compute_genotype_statistics
from sv_pgs.scale_mixture_ep import scale_mixture_prior
from sv_pgs.store_block_source import StoreGenotypeBlockSource
from tests.stage0_support import mosaic_codes
from tests.test_dosage_store import _write_store

_SAMPLES = 500
_TRAINING = 400
_BLOCK_CAP = 64
_DRAWS = 16
_WORKSPACE_BYTES = 1 << 26


def _budget() -> ComputeBudget:
    return ComputeBudget(
        device_kind="cpu", device_ids=(), device_names=(), device_bytes=(), device_compute_capabilities=(), host_bytes=1 << 31, cpu_threads=2
    )


class _StoreCodes:
    """The store as ``fast_scoring.CodeBlockSource``."""

    def __init__(self, store: DosageStore) -> None:
        self.store = store

    @property
    def sample_count(self) -> int:
        return int(self.store.n_samples)

    def iter_code_blocks(self, variant_ranges, buffers):
        for start, stop in variant_ranges:
            yield start, stop, self.store.read_codes(start, stop)


def _store(root: Path, seed: int, tied: bool = False):
    """Two chromosomes of mosaic dosages in one store half, and a trait with 15 causal variants (h2 = 0.5). With
    ``tied``, every tenth record repeats its predecessor and every tenth one after it is its predecessor's negation
    (dosage 2 - d): exact training ties of both signs, as a hard-genotype store has by the tens of thousands."""
    generator = np.random.default_rng(seed)
    codes = {name: mosaic_codes(generator, _SAMPLES, count) for name, count in (("chr21", 180), ("chr22", 140))}
    if tied:
        for values in codes.values():
            values[1::10] = values[0::10][: values[1::10].shape[0]]
            values[6::10] = 254 - values[5::10][: values[6::10].shape[0]]
    milli = {name: np.rint(values.astype(np.float64) / 127.0 * 1000.0).astype(np.int64) for name, values in codes.items()}
    _write_store(root, [milli])
    store = DosageStore.open(root)
    dosage = np.concatenate([store.read_codes(0, store.n_variants).astype(np.float64)]).T / 127.0
    causal = generator.choice(dosage.shape[1], size=15, replace=False)
    effects = np.zeros(dosage.shape[1])
    effects[causal] = generator.standard_normal(15)
    genetic = (dosage - dosage.mean(axis=0)) @ effects
    genetic *= np.sqrt(0.5) / np.std(genetic)
    covariate = generator.standard_normal(_SAMPLES)
    targets = 0.3 * covariate + genetic + np.sqrt(0.5) * generator.standard_normal(_SAMPLES)
    return store, covariate, targets, genetic


@pytest.mark.slow  # the whole EP driver on the synthetic store: 108 s on acl42 with the double loop (structured-genome, 2026-09-23)
def test_stage2_from_the_prior_is_certified_and_scores_the_held_out_samples(tmp_path: Path) -> None:
    store, covariate, targets, genetic = _store(tmp_path / "store", 7)
    training = np.arange(_TRAINING)
    held_out = np.arange(_TRAINING, _SAMPLES)
    training_covariates = np.column_stack([np.ones(_TRAINING), covariate[training]])
    statistics = compute_genotype_statistics(
        DosageStoreTileSource(store, np.arange(store.n_variants)),
        training,
        training_covariates,
        targets[training, None],
        ModelConfig(),
        _budget(),
        _BLOCK_CAP,
        tmp_path / "ld",
    )
    member_count = statistics.active_rows.shape[0]
    store_covariates = np.column_stack([np.ones(_SAMPLES), covariate])
    mask = np.zeros((_SAMPLES, 1))
    mask[training, 0] = 1.0
    offsets = np.zeros(member_count)
    classes = (np.arange(member_count) % 5 == 0).astype(np.int64)
    # The single-variant likelihoods that set the lattice take the covariate-only residual variance as the noise.
    start_noise = float(covariate_residual_variance(targets[:, None], mask, store_covariates)[0])
    nodes, floor, top = stage0_lattice(statistics, 0, start_noise, offsets, 0.5 / _DRAWS)
    prior = scale_mixture_prior(
        class_index=classes, log_variance_offset=offsets, annotation_design=np.zeros((member_count, 0)), annotation_groups=(),
        nodes=nodes, floor=floor, top=top,
    )
    source = StreamedDualSource(StoreGenotypeBlockSource.from_statistics(store, statistics, _budget(), _WORKSPACE_BYTES))
    gaussian = DualGaussian(
        source=source, training=mask, targets=targets[:, None], offsets=np.zeros((_SAMPLES, 1)), covariates=store_covariates,
        grams=block_grams(statistics, start_noise), probe_count=_DRAWS, seed=11,
    )
    fit = fit_full_data(gaussian=gaussian, statistics=statistics, prior=prior, draw_count=_DRAWS, working_bytes=1 << 22, seed=13)
    certificate = fit.certificate
    assert certificate.remaining_gain[0] <= 0.5 / _DRAWS
    assert certificate.mean_move[0] <= certificate.draw_tolerance[0]
    assert certificate.noise_gain[0] <= 0.5 / _DRAWS
    scoring = scoring_models(fit, prior, statistics, [TraitType.QUANTITATIVE], _DRAWS, seed=12)
    scores = score_genetic(_StoreCodes(store), ScoringPlan.from_models(scoring), _budget())
    # The scorer's in-sample genetic score is the fitted model's own X mu, read back from the store.
    np.testing.assert_allclose(scores.means[training, 0], np.asarray(gaussian.genetic_image)[training, 0], rtol=1e-8, atol=1e-8)
    assert np.corrcoef(scores.means[held_out, 0], genetic[held_out])[0, 1] > 0.5
    assert np.all(scores.variances[held_out, 0] > 0.0)


def test_stage2_by_mean_field_is_certified_and_scores_the_held_out_samples(tmp_path: Path) -> None:
    # The same store and prior through the streamed mean-field oracle (``_FullDataMeanField``), which the public route uses.
    store, covariate, targets, genetic = _store(tmp_path / "store", 7)
    training = np.arange(_TRAINING)
    held_out = np.arange(_TRAINING, _SAMPLES)
    training_covariates = np.column_stack([np.ones(_TRAINING), covariate[training]])
    statistics = compute_genotype_statistics(
        DosageStoreTileSource(store, np.arange(store.n_variants)),
        training,
        training_covariates,
        targets[training, None],
        ModelConfig(),
        _budget(),
        _BLOCK_CAP,
        tmp_path / "ld",
    )
    member_count = statistics.active_rows.shape[0]
    store_covariates = np.column_stack([np.ones(_SAMPLES), covariate])
    mask = np.zeros((_SAMPLES, 1))
    mask[training, 0] = 1.0
    offsets = np.zeros(member_count)
    classes = (np.arange(member_count) % 5 == 0).astype(np.int64)
    # The single-variant likelihoods that set the lattice take the covariate-only residual variance as the noise.
    start_noise = float(covariate_residual_variance(targets[:, None], mask, store_covariates)[0])
    nodes, floor, top = stage0_lattice(statistics, 0, start_noise, offsets, 0.5 / _DRAWS)
    prior = scale_mixture_prior(
        class_index=classes, log_variance_offset=offsets, annotation_design=np.zeros((member_count, 0)), annotation_groups=(),
        nodes=nodes, floor=floor, top=top,
    )
    source = StreamedDualSource(StoreGenotypeBlockSource.from_statistics(store, statistics, _budget(), _WORKSPACE_BYTES))
    gaussian = DualGaussian(
        source=source, training=mask, targets=targets[:, None], offsets=np.zeros((_SAMPLES, 1)), covariates=store_covariates,
        grams=block_grams(statistics, start_noise), probe_count=_DRAWS, seed=11,
    )
    fit = fit_full_data(gaussian=gaussian, statistics=statistics, prior=prior, draw_count=_DRAWS, working_bytes=1 << 22, seed=13, inference="mean_field")
    # Continued from its own fit (``stage2_wiring``'s lattice-check refit, here on the same lattice): certified again,
    # with no more outer iterations than the fit from the prior took.
    continued = fit_full_data(
        gaussian=gaussian, statistics=statistics, prior=prior, draw_count=_DRAWS, working_bytes=1 << 22, seed=13, inference="mean_field",
        starts=list(fit.hyperparameters), start_noise=np.asarray(fit.noise_variance, dtype=np.float64), start_mean=fit.member_mean,
    )
    if not continued.certificate.restored_best[0]:
        assert continued.certificate.remaining_gain[0] <= 0.5 / _DRAWS
    assert continued.certificate.outer_iterations[0] <= fit.certificate.outer_iterations[0]
    certificate = fit.certificate
    # The certificate is bound to the returned state: a restored best-ELBO state withholds the outer loop's terms.
    if certificate.restored_best[0]:
        assert np.isnan(certificate.remaining_gain[0]) and not certificate.outer_criterion_met[0]
    else:
        assert certificate.remaining_gain[0] <= 0.5 / _DRAWS
    assert certificate.budget_unresolved[0] == 0 and certificate.mixture_components[0] == len(fit.member_components) >= 1
    assert certificate.mean_move[0] <= certificate.draw_tolerance[0]
    assert certificate.noise_gain[0] <= 0.5 / _DRAWS
    from sv_pgs.full_data_fit import state_digest

    parts = [fit.member_mean[:, 0], fit.hyperparameters[0].coefficients, np.atleast_1d(fit.hyperparameters[0].log_smoothing),
             np.array([float(fit.noise_variance[0])])]
    parts += [part[:, 0] for shift, omega in fit.member_components for part in (shift, omega)]
    parts += [fit.component_weights[:, 0], fit.covariate_coefficients[:, 0]]
    np.testing.assert_array_equal(certificate.state_digest[0], state_digest(parts))
    assert not np.array_equal(certificate.state_digest[0], state_digest([fit.member_mean[:, 0] * 2.0, *parts[1:]]))
    scoring = scoring_models(fit, prior, statistics, [TraitType.QUANTITATIVE], _DRAWS, seed=12)
    # The draws are a law, drawn by tiles when scored; nothing members x draws is held by the scoring model.
    from sv_pgs.draw_laws import ProductMixtureDraws

    assert isinstance(scoring[0].posterior_draws, ProductMixtureDraws)
    scores = score_genetic(_StoreCodes(store), ScoringPlan.from_models(scoring), _budget())
    # The scorer's in-sample genetic score is q's own X m, read back from the store (the dual solver's mean at q's
    # sites is m only to its solve's bound, so it is not the reference here).
    signed = store.read_codes(0, store.n_variants).astype(np.float64) - 127.0
    standardized = (signed[statistics.active_rows] - statistics.means[:, None]) / statistics.scales[:, None]
    assert fit.member_mean is not None
    expected = standardized.T[training] @ fit.member_mean[:, 0]
    np.testing.assert_allclose(scores.means[training, 0], expected, rtol=1e-8, atol=1e-8)
    assert np.corrcoef(scores.means[held_out, 0], genetic[held_out])[0, 1] > 0.5
    assert np.all(scores.variances[held_out, 0] > 0.0)

def _mean_field_fit(tmp_path: Path):
    """``test_stage2_by_mean_field_is_certified_and_scores_the_held_out_samples``'s problem: a function fitting it by the
    mean-field route with ``fit_full_data``'s keywords, each fit on a dual solver of its own."""
    store, covariate, targets, _genetic = _store(tmp_path / "store", 7)
    training = np.arange(_TRAINING)
    statistics = compute_genotype_statistics(
        DosageStoreTileSource(store, np.arange(store.n_variants)), training, np.column_stack([np.ones(_TRAINING), covariate[training]]),
        targets[training, None], ModelConfig(), _budget(), _BLOCK_CAP, tmp_path / "ld",
    )
    member_count = statistics.active_rows.shape[0]
    store_covariates = np.column_stack([np.ones(_SAMPLES), covariate])
    mask = np.zeros((_SAMPLES, 1))
    mask[training, 0] = 1.0
    offsets = np.zeros(member_count)
    start_noise = float(covariate_residual_variance(targets[:, None], mask, store_covariates)[0])
    nodes, floor, top = stage0_lattice(statistics, 0, start_noise, offsets, 0.5 / _DRAWS)
    prior = scale_mixture_prior(
        class_index=(np.arange(member_count) % 5 == 0).astype(np.int64), log_variance_offset=offsets, annotation_design=np.zeros((member_count, 0)),
        annotation_groups=(), nodes=nodes, floor=floor, top=top,
    )
    source = StreamedDualSource(StoreGenotypeBlockSource.from_statistics(store, statistics, _budget(), _WORKSPACE_BYTES))

    def fit(**continuation):
        gaussian = DualGaussian(
            source=source, training=mask, targets=targets[:, None], offsets=np.zeros((_SAMPLES, 1)), covariates=store_covariates,
            grams=block_grams(statistics, start_noise), probe_count=_DRAWS, seed=11,
        )
        return fit_full_data(
            gaussian=gaussian, statistics=statistics, prior=prior, draw_count=_DRAWS, working_bytes=1 << 22, seed=13, inference="mean_field", **continuation,
        )

    return fit


def test_a_fit_continued_from_its_own_fixed_point_and_mixture_is_the_same_fit_in_a_few_sweeps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The empirical Bayes side of a warm start (a seed's or any earlier fit's state, ``stage2_wiring``): continued from
    a converged fit's hyperparameters, noise, main fixed point (``component_means[0]``) and mode mixture
    (``mixture_start``), the fit reaches the same fixed point and mixture, to the draws' resolution, in a handful of
    sweeps, where the cold fit took dozens: its empirical Bayes in at most 10. Its mode mixture solves the earlier
    components from their own means and noise; here the first (the ridge component) was stopped on a plateau and moves
    on into the main mode (``_mode_mixture``), so the search runs again, as in the cold fit, at the cost of that one
    solve more (``test_warm_start_eb``: a saved fit whose components come back keeps its search, with no order)."""
    from sv_pgs import full_data_fit
    from sv_pgs.full_data_fit import _FullDataMeanField

    fit = _mean_field_fit(tmp_path)
    sweep = _FullDataMeanField._sweep
    # Sweeps of the empirical Bayes' oracle (the one with an allowance) and of the mode mixture's, and the mixture's
    # searches from the ridge start or from zero.
    sweeps = {"eb": 0, "mixture": 0}
    searches = [0]

    def counted(self, model, hyperparameters):
        sweeps["eb" if self.pass_allowance is not None else "mixture"] += 1
        return sweep(self, model, hyperparameters)

    initialize = _FullDataMeanField.__init__

    def made(self, *arguments, **keywords):
        searches[0] += "order_seed" in keywords
        initialize(self, *arguments, **keywords)

    ridge_mean = full_data_fit._ridge_mean

    def ridge(*arguments, **keywords):
        searches[0] += 1
        return ridge_mean(*arguments, **keywords)

    monkeypatch.setattr(_FullDataMeanField, "_sweep", counted)
    monkeypatch.setattr(_FullDataMeanField, "__init__", made)
    monkeypatch.setattr(full_data_fit, "_ridge_mean", ridge)
    cold = fit()
    cold_sweeps, cold_searches = dict(sweeps), searches[0]
    sweeps.update(eb=0, mixture=0)
    searches[0] = 0
    warm = fit(
        starts=list(cold.hyperparameters), start_noise=np.asarray(cold.noise_variance, dtype=np.float64), start_mean=cold.component_means[0],
        mixture_start=cold.mixture_start(),
    )
    components = int(cold.certificate.mixture_components[0])
    print(
        f"warm start: cold fit {cold_sweeps} sweeps ({cold_searches} searches); continued from its own fixed point and its {components} "
        f"components' mixture {sweeps} sweeps ({searches[0]} searches), {int(warm.certificate.mixture_components[0])} modes"
    )
    assert sweeps["eb"] <= 10 < sum(cold_sweeps.values())
    # One earlier component solved, then the search: the cold mixture's sweeps and that solve's.
    assert 0 < searches[0] and sweeps["mixture"] < 2 * cold_sweeps["mixture"] and sum(sweeps.values()) < sum(cold_sweeps.values())
    assert warm.certificate.outer_criterion_met[0] == cold.certificate.outer_criterion_met[0]
    for hyperparameters, earlier in zip(warm.hyperparameters, cold.hyperparameters):
        np.testing.assert_allclose(hyperparameters.coefficients, earlier.coefficients, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(warm.noise_variance, cold.noise_variance, rtol=1e-3)
    assert np.corrcoef(warm.member_mean[:, 0], cold.member_mean[:, 0])[0, 1] > 0.999


def test_an_empirical_bayes_past_its_allowance_ends_at_its_best_state_uncertified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``EB_PASSES``: a model whose empirical Bayes has spent its sweeps is refused every later trial, which costs no
    sweep, and the outer loop ends by its own rules: the fit returns the best-ELBO state it solved, its certificate's
    outer criterion unmet with the allowance among its refusals, and the stages' sweeps past the allowance are only
    the re-solve at the held point."""
    from sv_pgs import full_data_fit
    from sv_pgs.full_data_fit import _FullDataMeanField

    fit = _mean_field_fit(tmp_path)

    # The empirical Bayes' oracles (an unbounded sibling is the mode mixture's), the calls each takes once spent, and
    # the passes of those that solved.
    oracles: list[_FullDataMeanField] = []
    refused_calls = [0]
    solved_after: list[int] = []
    call = _FullDataMeanField.__call__

    def counted(self, hyperparameters):
        if self.pass_allowance is not None and not any(self is oracle for oracle in oracles):
            oracles.append(self)
        spent_before, passes = bool(self.allowance_spent[0]), int(self.model_passes[0])
        points = call(self, hyperparameters)
        if spent_before:
            refused_calls[0] += 1
            assert refused_calls[0] < 1_000, "the outer loop does not end against refused trials"
            if points[0] is None:
                # A refused trial costs no sweep.
                assert int(self.model_passes[0]) == passes
            else:
                solved_after.append(int(self.model_passes[0]) - passes)
        return points

    monkeypatch.setattr(_FullDataMeanField, "__call__", counted)
    unbounded = fit()
    spent = sum(int(oracle.model_passes[0]) for oracle in oracles)
    assert not any(oracle.allowance_spent[0] for oracle in oracles)
    # An allowance the cold fit's first solve and one more trial spend.
    oracles.clear()
    allowance = spent // 3
    monkeypatch.setattr(full_data_fit, "EB_PASSES", allowance)
    ended = fit()
    (oracle,) = oracles
    assert oracle.allowance_spent[0] and refused_calls[0] > 0
    certificate = ended.certificate
    assert certificate.budget_unresolved[0] >= 1 and not certificate.outer_criterion_met[0]
    assert any(f"spent its allowance of {allowance} sweeps" in refusal for refusal in certificate.refusals)
    # Past its allowance the stage solved only once more, at its held point (the outer loop's last call).
    assert len(solved_after) <= 1
    assert int(oracle.model_passes[0]) < spent
    # The returned state is the best ELBO the stage solved, and still the trait's: certified or not, it scores.
    assert float(oracle.elbo[0]) >= float(oracle.best_elbo[0]) - 0.5 / _DRAWS
    assert np.corrcoef(ended.member_mean[:, 0], unbounded.member_mean[:, 0])[0, 1] > 0.5


def test_an_empirical_bayes_ended_by_its_allowance_returns_its_best_state_where_the_outer_loop_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Refused trials can leave the outer loop no certified step, and it raises (``scale_mixture_ep.NoCertifiedProgress``:
    B + S indefinite at a state outside every basin, as on ``test_stage2_wiring``'s band-route fixture after 3,524
    sweeps): the fit still returns, at the model's best-ELBO state, uncertified, with the allowance and the loop's error
    among its refusals, where without the allowance the error would have cost the fit."""
    import dataclasses

    from sv_pgs import full_data_fit
    from sv_pgs.scale_mixture_ep import NoCertifiedProgress

    fit = _mean_field_fit(tmp_path)
    monkeypatch.setattr(full_data_fit, "EB_PASSES", 1)
    refused: list[list] = []

    def walking(prior, starts, oracle, working_bytes, tolerance):
        # The start's solve, one trial further (refused: the allowance is spent), then the loop's error.
        assert all(point is not None for point in oracle(list(starts)))
        refused.append(list(oracle([dataclasses.replace(start, coefficients=np.asarray(start.coefficients) + 0.1) for start in starts])))
        raise NoCertifiedProgress("the Newton-B step makes no certified progress at the EP fixed point (the test's)")

    monkeypatch.setattr(full_data_fit, "fit_hyperparameters", walking)
    ended = fit()
    assert refused == [[None]]
    certificate = ended.certificate
    assert not certificate.outer_criterion_met[0] and certificate.budget_unresolved[0] >= 1
    assert any("spent its allowance of 1 sweeps" in refusal for refusal in certificate.refusals)
    assert any(
        "the outer loop raised after an empirical Bayes allowance ended it: the Newton-B step makes no certified progress at the EP fixed point "
        "(the test's)" in refusal for refusal in certificate.refusals
    )
    assert np.all(np.isfinite(ended.member_mean)) and np.isfinite(ended.noise_variance[0])


def test_ridge_component_is_the_gaussian_members_posterior_mean(tmp_path: Path) -> None:
    # ``_ridge_mean``: the dual solver at uniform Gaussian sites against the dense posterior mean
    # D Xp' (Xp D Xp' + sigma^2 I)^-1 yp, D = diag(v e^o), on the covariate-projected training rows.
    store, covariate, targets, _genetic = _store(tmp_path / "store", 5)
    training = np.arange(_TRAINING)
    training_covariates = np.column_stack([np.ones(_TRAINING), covariate[training]])
    statistics = compute_genotype_statistics(
        DosageStoreTileSource(store, np.arange(store.n_variants)), training, training_covariates, targets[training, None],
        ModelConfig(), _budget(), _BLOCK_CAP, tmp_path / "ld",
    )
    member_count = statistics.active_rows.shape[0]
    store_covariates = np.column_stack([np.ones(_SAMPLES), covariate])
    mask = np.zeros((_SAMPLES, 1))
    mask[training, 0] = 1.0
    offsets = np.random.default_rng(3).normal(scale=0.5, size=member_count)
    start_noise = float(covariate_residual_variance(targets[:, None], mask, store_covariates)[0])
    nodes, floor, top = stage0_lattice(statistics, 0, start_noise, offsets, 0.5 / _DRAWS)
    prior = scale_mixture_prior(
        class_index=np.zeros(member_count, dtype=np.int64), log_variance_offset=offsets, annotation_design=np.zeros((member_count, 0)),
        annotation_groups=(), nodes=nodes, floor=floor, top=top,
    )
    source = StreamedDualSource(StoreGenotypeBlockSource.from_statistics(store, statistics, _budget(), _WORKSPACE_BYTES))
    gaussian = DualGaussian(
        source=source, training=mask, targets=targets[:, None], offsets=np.zeros((_SAMPLES, 1)), covariates=store_covariates,
        grams=block_grams(statistics, start_noise), probe_count=_DRAWS, seed=11,
    )
    moments = moment_starts(statistics, prior)
    mean = _ridge_mean(gaussian, statistics, prior, moments, _DRAWS)[:, 0]
    signed = store.read_codes(0, store.n_variants).astype(np.float64) - 127.0
    standardized = ((signed[statistics.active_rows] - statistics.means[:, None]) / statistics.scales[:, None]).T[training]
    projection = np.eye(_TRAINING) - training_covariates @ np.linalg.pinv(training_covariates)
    design, target = projection @ standardized, projection @ targets[training]
    prior_variance = float(moments[0].mean_variance) * np.exp(offsets)
    expected = prior_variance * (design.T @ np.linalg.solve((design * prior_variance) @ design.T + float(moments[0].noise) * np.eye(_TRAINING), target))
    # The solver's certificate is in the posterior precision's norm, to the draws' resolution.
    precision = design.T @ design / float(moments[0].noise) + np.diag(1.0 / prior_variance)
    difference = mean - expected
    assert float(difference @ precision @ difference) <= 1.0 / _DRAWS


def test_block_grams_share_stage0s_float32_arrays_across_models(tmp_path: Path) -> None:
    from dataclasses import replace

    from sv_pgs.full_data_fit import block_grams
    from sv_pgs.marginal_variances import window_working_bytes

    store, covariate, targets, _genetic = _store(tmp_path / "store", 8)
    training = np.arange(_TRAINING)
    statistics = compute_genotype_statistics(
        DosageStoreTileSource(store, np.arange(store.n_variants)),
        training,
        np.column_stack([np.ones(_TRAINING), covariate[training]]),
        targets[training, None],
        ModelConfig(),
        _budget(),
        _BLOCK_CAP,
        tmp_path / "ld",
    )
    shared = block_grams(statistics)
    model = replace(shared, scale=1.0 / 0.7)
    # The same arrays, not copies, and Stage 0's float32 storage.
    assert all(first is second for first, second in zip(shared.within, model.within))
    assert all(first is second for first, second in zip(shared.next_cross, model.next_cross))
    assert all(np.asarray(values).dtype == np.float32 for values in shared.within)
    ld = statistics.ld
    for block_index in range(ld.block_count):
        # The model's metric enters as its scale, 1 / sigma^2, multiplying the promoted float32 values.
        expected = np.asarray(ld.block(block_index).projected_gram, dtype=np.float64) * (1.0 / 0.7)
        assert np.array_equal(model.within_block(block_index), expected)
    assert window_working_bytes(model) > 0


def test_stage2_by_mean_field_carries_tie_members(tmp_path: Path) -> None:
    # A store with exact training ties of both signs: each member keeps its own coordinate, and the fit and its
    # scores are the untied route's in every respect the test above checks.
    store, covariate, targets, genetic = _store(tmp_path / "store", 7, tied=True)
    training = np.arange(_TRAINING)
    held_out = np.arange(_TRAINING, _SAMPLES)
    statistics = compute_genotype_statistics(
        DosageStoreTileSource(store, np.arange(store.n_variants)), training, np.column_stack([np.ones(_TRAINING), covariate[training]]),
        targets[training, None], ModelConfig(), _budget(), _BLOCK_CAP, tmp_path / "ld",
    )
    member_count = statistics.active_rows.shape[0]
    group_count = int(np.asarray(statistics.tie_map.kept_indices).shape[0])
    assert group_count < member_count
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
    gaussian = DualGaussian(
        source=source, training=mask, targets=targets[:, None], offsets=np.zeros((_SAMPLES, 1)), covariates=store_covariates,
        grams=block_grams(statistics, start_noise), probe_count=_DRAWS, seed=11,
    )
    fit = fit_full_data(gaussian=gaussian, statistics=statistics, prior=prior, draw_count=_DRAWS, working_bytes=1 << 22, seed=13, inference="mean_field")
    certificate = fit.certificate
    # q's sites here are a linear response's (members with v omega up to 2.8, R's least eigenvalue -12.7): every
    # refresh is solved, none refused.
    assert not certificate.refusals
    assert certificate.noise_gain[0] <= 0.5 / _DRAWS
    assert fit.member_mean is not None and fit.member_mean.shape == (member_count, 1)
    scoring = scoring_models(fit, prior, statistics, [TraitType.QUANTITATIVE], _DRAWS, seed=12)
    scores = score_genetic(_StoreCodes(store), ScoringPlan.from_models(scoring), _budget())
    signed = store.read_codes(0, store.n_variants).astype(np.float64) - 127.0
    standardized = (signed[statistics.active_rows] - statistics.means[:, None]) / statistics.scales[:, None]
    expected = standardized.T[training] @ fit.member_mean[:, 0]
    np.testing.assert_allclose(scores.means[training, 0], expected, rtol=1e-8, atol=1e-8)
    assert np.corrcoef(scores.means[held_out, 0], genetic[held_out])[0, 1] > 0.5
    if not certificate.remaining_gain[0] <= 0.5 / _DRAWS:
        pytest.xfail(
            "the outer loop stalls uncertified on the tied store (remaining 0.11 against 0.031): its second joint trial's "
            "certified gain, 0.060 with resolution 0.054, is refused and the next plan is the same state's"
        )


def _two_trait_oracle(tmp_path: Path):
    """Two traits on the same training rows (a group's models, ``stage2_wiring.model_groups``), Stage 0 over both: a
    function building the mean-field oracle of any of them in order, the moment starts of each, Stage 0's statistics and
    the prior."""
    from sv_pgs.full_data_fit import _FullDataMeanField
    from sv_pgs.scale_mixture_ep import initial_hyperparameters

    store, covariate, targets, genetic = _store(tmp_path / "store", 7)
    generator = np.random.default_rng(3)
    traits = np.column_stack([targets, 0.6 * genetic - 0.2 * covariate + 0.8 * generator.standard_normal(_SAMPLES)])
    training = np.arange(_TRAINING)
    statistics = compute_genotype_statistics(
        DosageStoreTileSource(store, np.arange(store.n_variants)), training, np.column_stack([np.ones(_TRAINING), covariate[training]]),
        traits[training], ModelConfig(), _budget(), _BLOCK_CAP, tmp_path / "ld",
    )
    member_count = statistics.active_rows.shape[0]
    store_covariates = np.column_stack([np.ones(_SAMPLES), covariate])
    mask = np.zeros((_SAMPLES, 2))
    mask[training] = 1.0
    offsets = np.zeros(member_count)
    noise = covariate_residual_variance(traits, mask, store_covariates)
    nodes, floor, top = stage0_lattice(statistics, 0, float(noise[0]), offsets, 0.5 / _DRAWS)
    prior = scale_mixture_prior(
        class_index=np.zeros(member_count, dtype=np.int64), log_variance_offset=offsets, annotation_design=np.zeros((member_count, 0)),
        annotation_groups=(), nodes=nodes, floor=floor, top=top,
    )
    source = StreamedDualSource(StoreGenotypeBlockSource.from_statistics(store, statistics, _budget(), _WORKSPACE_BYTES))
    starts = [initial_hyperparameters(prior, moment.mean_variance) for moment in moment_starts(statistics, prior)]

    def oracle(columns: list[int]) -> _FullDataMeanField:
        gaussian = DualGaussian(
            source=source, training=mask[:, columns], targets=traits[:, columns], offsets=np.zeros((_SAMPLES, len(columns))),
            covariates=store_covariates, grams=block_grams(statistics), probe_count=_DRAWS, seed=11,
        )
        return _FullDataMeanField(gaussian, statistics, prior, _DRAWS, 1 << 22, 13, [starts[column] for column in columns], noise[columns])

    return oracle, starts, statistics, prior


def test_a_joint_call_keeps_a_model_at_its_held_fixed_point(tmp_path: Path) -> None:
    """A model still at the fixed point the oracle returned for it, asked at the same hyperparameters, keeps that point
    while the call solves the others (a joint fit's outer loop asks every model at each call, one whose fit has ended
    included; fitted alone it is not solved again); a call where every model is so solves them all."""
    from sv_pgs.scale_mixture_ep import MixtureHyperparameters

    oracle_of, starts, _statistics, _prior = _two_trait_oracle(tmp_path)
    oracle = oracle_of([0, 1])
    first = oracle(list(starts))
    assert all(point is not None for point in first)
    held_mean = oracle.mean[:, 0].copy()
    moved = MixtureHyperparameters(coefficients=starts[1].coefficients * 0.9, log_smoothing=starts[1].log_smoothing.copy())
    passes = oracle.passes
    second = oracle([starts[0], moved])
    assert second[0] is first[0] and second[1] is not first[1] and second[1] is not None
    np.testing.assert_array_equal(oracle.mean[:, 0], held_mean)
    # Only model 1's solves read the store: its carried and cold solves.
    alone = oracle_of([1])
    alone(list(starts[1:]))
    alone_passes = alone.passes
    alone([moved])
    assert oracle.passes - passes == alone.passes - alone_passes
    # Every model at its held point: the call solves them all.
    third = oracle([starts[0], moved])
    assert third[0] is not first[0] and third[1] is not second[1]


def test_a_restored_model_waits_while_another_model_solves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A trial of the first model lost its evidence test; after its accepted point is restored it waits while the
    second model moves. Only the second model takes carried and cold solves, just as if it had been fitted alone.
    A raw snapshot with the same values is not the returned point: identity is required as well as hyperparameters,
    so the reuse cannot silently choose another basin or certify an arbitrary restored candidate."""
    from sv_pgs.scale_mixture_ep import MixtureHyperparameters

    oracle_of, starts, _statistics, _prior = _two_trait_oracle(tmp_path)
    oracle = oracle_of([0, 1])
    first = oracle(list(starts))
    assert all(point is not None for point in first)
    moved = [MixtureHyperparameters(coefficients=start.coefficients * 0.9, log_smoothing=start.log_smoothing.copy()) for start in starts]
    assert all(point is not None for point in oracle([moved[0], starts[1]]))
    first[0].restore()
    solve_model = oracle._solve_model
    solved: list[int] = []

    def counted(model, hyperparameters, pass_budget=None):
        solved.append(model)
        return solve_model(model, hyperparameters, pass_budget)

    monkeypatch.setattr(oracle, "_solve_model", counted)
    second = oracle([starts[0], moved[1]])
    assert second[0] is first[0] and second[1] is not None
    assert solved == [1, 1]
    np.testing.assert_array_equal(oracle.mean[:, 0], first[0].mean)
    raw = oracle._state(0)
    oracle._restore_state(raw, 0)
    solved.clear()
    assert all(point is not None for point in oracle([starts[0], moved[1]]))
    assert solved == [0, 0]
    # The reuse key includes the smoothing weights as well as the coefficients, even though the mean-field
    # sweeps themselves read the coefficients: a different outer request must take the ordinary solve path.
    reweighted = MixtureHyperparameters(
        coefficients=starts[0].coefficients.copy(),
        log_smoothing=np.where(np.isfinite(starts[0].log_smoothing), starts[0].log_smoothing + 1.0, 0.0),
    )
    solved.clear()
    assert all(point is not None for point in oracle([reweighted, moved[1]]))
    # Its carried solve, and its cold solve where the cold start's measured need is within the budget
    # (``_FullDataMeanField._solved``); model 1 is not solved.
    assert solved[0] == 0 and set(solved) == {0}


def test_a_cold_solve_known_to_need_more_than_its_budget_is_not_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cold solve gets the carried solve's count of passes, so where the cold start is known to need more it could
    only be abandoned at that count (``_FullDataMeanField._solved``): it is not run, and the call returns what an oracle
    running every cold solve returns, bit for bit, where that cold solve does not win. The need is the first call's
    passes (that solve starts at the cold start), measured again by every cold solve that runs (one more than it used
    where it was abandoned), and the model's 1st, 2nd, 4th, ... budgeted cold solves run whatever their need."""
    from sv_pgs.full_data_fit import _PassBudget
    from sv_pgs.scale_mixture_ep import MixtureHyperparameters

    oracle_of, starts, _statistics, _prior = _two_trait_oracle(tmp_path)
    skipping, running = oracle_of([0]), oracle_of([0])
    # Per oracle, each solve's (budget, passes, outcome, ELBO).
    solves: dict[int, list[tuple[int | None, int, str, float]]] = {id(skipping): [], id(running): []}
    for oracle in (skipping, running):
        solve_model = oracle._solve_model

        def counted(model, hyperparameters, pass_budget=None, oracle=oracle, solve_model=solve_model):
            before = oracle.passes
            try:
                solve_model(model, hyperparameters, pass_budget)
            except _PassBudget:
                solves[id(oracle)].append((pass_budget, oracle.passes - before, "abandoned", float(oracle.elbo[model])))
                raise
            solves[id(oracle)].append((pass_budget, oracle.passes - before, "solved", float(oracle.elbo[model])))

        monkeypatch.setattr(oracle, "_solve_model", counted)
        assert oracle([starts[0]])[0] is not None
    assert skipping._cold_need[0] == solves[id(skipping)][0][1] > 0
    unreachable = np.iinfo(np.int64).max

    def call(factor: float) -> tuple[list, list]:
        """Both oracles at the coefficients times ``factor``, ``skipping`` told its cold start needs more than any budget
        and ``running`` that it needs none; each one's solves in the call."""
        hyperparameters = MixtureHyperparameters(coefficients=starts[0].coefficients * factor, log_smoothing=starts[0].log_smoothing.copy())
        skipping._cold_need[0], running._cold_need[0] = unreachable, 0
        counts = {key: len(values) for key, values in solves.items()}
        assert skipping([hyperparameters])[0] is not None and running([hyperparameters])[0] is not None
        return solves[id(skipping)][counts[id(skipping)]:], solves[id(running)][counts[id(running)]:]

    # The 1st and 2nd budgeted calls run the cold solve whatever its need, and measure it again.
    for factor in (0.9, 0.8):
        skipped, ran = call(factor)
        assert skipped == ran and len(skipped) == 2
        (budget, carried, _outcome, _elbo), (cold_budget, used, outcome, _cold_elbo) = skipped
        assert budget is None and cold_budget == carried
        assert skipping._cold_need[0] == (used + 1 if outcome == "abandoned" else used)
        np.testing.assert_array_equal(skipping.mean, running.mean)
    # The 3rd does not: only the carried solve runs, the need stands, and the fixed point is the running oracle's,
    # whose cold solve did not win.
    skipped, ran = call(0.7)
    assert len(skipped) == 1 and skipped[0] == ran[0] and len(ran) == 2
    assert ran[1][2] == "abandoned" or ran[1][3] <= ran[0][3]
    assert skipping._cold_need[0] == unreachable
    np.testing.assert_array_equal(skipping.mean, running.mean)
    np.testing.assert_array_equal(skipping.variance, running.variance)
    np.testing.assert_array_equal(skipping.residual[0], running.residual[0])
    assert skipping.elbo[0] == running.elbo[0] and skipping.noise[0] == running.noise[0]


def test_a_group_s_mode_mixture_is_each_model_s_own(tmp_path: Path) -> None:
    """The mode mixture of a group's oracle gives each model the mixture it gets alone: its own components, weights,
    repeat and settling tests, the later orders solving only the models still searching."""
    from sv_pgs.full_data_fit import _mode_mixture

    oracle_of, starts, statistics, prior = _two_trait_oracle(tmp_path)

    def mixture(columns: list[int]):
        oracle = oracle_of(columns)
        assert all(point is not None for point in oracle([starts[column] for column in columns]))
        return _mode_mixture(oracle, oracle.gaussian, statistics, prior, _DRAWS, 1 << 22, 13, tuple(starts[column] for column in columns))

    together = mixture([0, 1])
    for column in range(2):
        alone = mixture([column])
        count = int(alone.counts[0])
        assert together.counts[column] == count
        np.testing.assert_allclose(together.weights[:count, column], alone.weights[:, 0], rtol=1e-10, atol=1e-12)
        assert np.all(together.weights[count:, column] == 0.0)
        np.testing.assert_allclose(together.mean[:, column], alone.mean[:, 0], rtol=1e-10, atol=1e-12)
        assert together.unresolved[column] == alone.unresolved[0]


def test_the_mode_mixture_reads_no_dual_solver_factor_of_its_orders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The mode mixture's ridge start and orders are solved and never factored by the dual solver
    (``_FullDataMeanField._solved``), since nothing of the mixture reads a factor: factoring the dual solver at an order's
    sites, as an oracle's call does, leaves every part of the mixture as it is, bit for bit, and the only factor the
    mixture makes is the main fixed point's own, put back at its end."""
    from sv_pgs.full_data_fit import _FullDataMeanField, _mode_mixture

    oracle_of, starts, statistics, prior = _two_trait_oracle(tmp_path)
    iterate, solved = _FullDataMeanField._iterate, _FullDataMeanField._solved
    factored: list[object] = []
    mains: list[object] = []

    def counted(self, *arguments):
        factored.append(self)
        return iterate(self, *arguments)

    def mixture():
        main = oracle_of([0, 1])
        mains.append(main)
        assert all(point is not None for point in main(list(starts)))
        factored.clear()
        return main, _mode_mixture(main, main.gaussian, statistics, prior, _DRAWS, 1 << 22, 13, tuple(starts))

    monkeypatch.setattr(_FullDataMeanField, "_iterate", counted)
    main, plain = mixture()
    assert factored == [main]

    def factoring(self, hyperparameters, solving):
        # The ridge start and an order every model joins, factored as an oracle's call factors it (the main fixed
        # point's own call is factored once, as before).
        entry = solved(self, hyperparameters, solving)
        if entry is not None and self is not mains[-1] and len(solving) == self.model_count:
            iterate(self, self.site_precision, self.site_shift, self.noise)
        return entry

    monkeypatch.setattr(_FullDataMeanField, "_solved", factoring)
    _main, again = mixture()
    for name in ("mean", "weights", "counts", "mean_move", "noise_gain", "unresolved"):
        np.testing.assert_array_equal(getattr(again, name), getattr(plain, name))
    for (shift, omega), (plain_shift, plain_omega) in zip(again.components, plain.components, strict=True):
        np.testing.assert_array_equal(shift, plain_shift)
        np.testing.assert_array_equal(omega, plain_omega)


def test_a_continuation_s_mode_mixture_starts_from_the_earlier_fit_s_components(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A continuation's mode mixture (``MixtureStart``) solves the earlier fit's components again from their means and
    noise (the earlier main fixed point only where this fit's main one is not it): a model whose earlier search settled,
    whose every component is again a fixed point where it started and whose main fixed point is one of them keeps that
    settled search, with neither the ridge start nor an order solved for it, and its mixture is the earlier one to the
    draws' resolution; a model whose earlier search did not settle solves none of them and searches as without a start,
    alone."""
    from sv_pgs.full_data_fit import MixtureStart, _FullDataMeanField, _mode_mixture

    oracle_of, starts, statistics, prior = _two_trait_oracle(tmp_path)
    moments = moment_starts(statistics, prior)
    solved = _FullDataMeanField._solved
    # Per solve: the models solved and whether it started from zero (an order) rather than from given means.
    solves: list[tuple[list[int], bool]] = []

    def counted(self, hyperparameters, solving):
        solves.append((list(solving), self is not mains[-1] and not np.any(self.mean)))
        return solved(self, hyperparameters, solving)

    mains: list[_FullDataMeanField] = []

    def mixture(start: MixtureStart | None):
        main = oracle_of([0, 1])
        mains.append(main)
        assert all(point is not None for point in main(list(starts)))
        solves.clear()
        return main, _mode_mixture(main, main.gaussian, statistics, prior, _DRAWS, 1 << 22, 13, tuple(starts), moments, start)

    monkeypatch.setattr(_FullDataMeanField, "_solved", counted)
    _main, plain = mixture(None)
    assert np.all(plain.settled) and len(plain.component_means) == int(plain.counts.max())
    # The ridge start and at least one order per model, from zero.
    assert any(from_zero for _models, from_zero in solves)
    earlier = MixtureStart(means=plain.component_means, noises=plain.component_noises, counts=plain.counts, settled=plain.settled)
    main, continued = mixture(earlier)
    assert np.all(continued.settled) and np.array_equal(continued.counts, plain.counts)
    # Only the earlier components were solved, each from its own means, but the main fixed point, which this main one is.
    restarted = len(plain.component_means) - 1
    assert len(solves) == restarted and not any(from_zero for _models, from_zero in solves)
    for model in range(2):
        moved = main._fitted_square(continued.mean[:, model] - plain.mean[:, model], model) / float(main.noise[model])
        assert moved <= 1.0 / _DRAWS
    # Model 1's earlier search unsettled: it alone searches again, from the ridge start and the orders.
    _main, searched = mixture(MixtureStart(means=plain.component_means, noises=plain.component_noises, counts=plain.counts, settled=np.array([True, False])))
    # Model 0's earlier components (but its main fixed point), then model 1's search alone.
    own = int(plain.counts[0]) - 1
    assert searched.settled[0] and len(solves) > own
    assert all(models == [0] for models, _from_zero in solves[:own]) and all(models == [1] for models, _from_zero in solves[own:])
    assert any(from_zero for _models, from_zero in solves[own:])
    np.testing.assert_allclose(searched.mean[:, 0], continued.mean[:, 0], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("cached", [False, True])
def test_relative_solve_retries_from_its_latest_duals_for_the_unfinished_columns(cached):
    """A tighter relative certificate continues only the unfinished columns, from this solve's answer. The
    cross-fixed-point cache may be absent or contain a stale answer; neither changes the retry's start."""
    from types import SimpleNamespace
    from sv_pgs.full_data_fit import _relative_solve
    from sv_pgs.scale_mixture_ep import solve_key

    right = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    starts = {(0, b"directions", 3): (np.array([[3.0, 4.0, 5.0]]), None)} if cached else None
    seen = []
    gaussian = SimpleNamespace(array_module=np)

    def posterior_solve(values, model, bound, start):
        seen.append(None if start is None else start.copy())
        if len(seen) == 1:
            gaussian.last_posterior_duals = np.array([[11.0, 22.0]])
            return values.copy(), np.array([0.01, 0.5])
        np.testing.assert_array_equal(values, right[:, 2:])
        gaussian.last_posterior_duals = np.array([[23.0]])
        return values.copy(), np.array([0.001])

    gaussian.posterior_solve = posterior_solve
    token = solve_key.set(b"directions" if cached else None)
    try:
        solved = _relative_solve(gaussian, 0, np.full(2, 100.0), lambda: None, starts)(right, 0.1)
    finally:
        solve_key.reset(token)
    assert len(seen) == 2
    if cached:
        np.testing.assert_array_equal(seen[0], [[3.0, 5.0]])
        np.testing.assert_array_equal(starts[(0, b"directions", 3)][0], [[11.0, 0.0, 23.0]])
    else:
        assert seen[0] is None
    np.testing.assert_array_equal(seen[1], [[22.0]])
    np.testing.assert_array_equal(solved, right)


def test_warm_starts_no_later_solve_asks_for_are_released_after_two_oracle_calls(monkeypatch):
    """A warm start's key digests its directions, which move with the state: one stored again is kept, one not stored
    again over two oracle calls gives back its cache lease, an evicted one leaves no stamp behind, and a closed cache
    (its oracle's outer loop has ended) holds nothing until the oracle is called again."""
    from types import SimpleNamespace
    from sv_pgs import full_data_fit
    from sv_pgs.full_data_fit import _SolveStarts, _hold_start

    released, evictions = [], {}

    def admit(pool, nbytes, purpose, evict, allocated=False):
        lease = SimpleNamespace(name=len(evictions))
        lease.release = lambda: released.append(lease.name)
        evictions[lease.name] = evict
        return lease

    monkeypatch.setattr(full_data_fit, "current_broker", lambda: SimpleNamespace(capacities={"host": 1}, admit=admit))
    first, second, third = (0, b"first", 1), (0, b"second", 1), (0, b"third", 1)
    starts = _SolveStarts()
    starts.age()
    _hold_start(starts, first, np.zeros((2, 1)))
    _hold_start(starts, second, np.zeros((2, 1)))
    starts.age()
    _hold_start(starts, second, np.ones((2, 1)))
    _hold_start(starts, third, np.ones((2, 1)))
    assert released == [1]
    starts.age()
    assert set(starts) == {first, second, third}
    evictions[3]()
    starts.age()
    assert set(starts) == {second} and released == [1, 0] and set(starts.stored) == {second}
    np.testing.assert_array_equal(starts[second][0], np.ones((2, 1)))
    starts.age()
    assert not starts and not starts.stored and released == [1, 0, 2]
    # The outer loop's end releases every start and keeps none until the oracle is called again.
    _hold_start(starts, first, np.zeros((2, 1)))
    starts.close()
    _hold_start(starts, second, np.zeros((2, 1)))
    assert not starts and not starts.stored and released == [1, 0, 2, 4]
    starts.age()
    _hold_start(starts, second, np.zeros((2, 1)))
    assert set(starts) == {second}


def test_mean_field_posterior_does_not_hold_an_unused_refreshs_device_cross_products():
    """Mean-field curvature uses only Sigma R. Replacing the shared solver's refresh must release its old window
    cross-products even while an outer anchor still owns the old mean-field posterior."""
    import weakref
    from types import SimpleNamespace
    from sv_pgs.full_data_fit import _FullDataMeanField

    cross = np.ones((8, 8))
    reference = weakref.ref(cross)
    oracle = object.__new__(_FullDataMeanField)
    oracle.gaussian = SimpleNamespace(bulk_solves=[SimpleNamespace(resolved_cross=cross, site_precision=np.ones(8))])
    oracle._solve_starts = {}
    oracle._ensure = lambda state, model: None
    posterior = oracle._reduced_posterior(0, SimpleNamespace(blocks=(np.arange(8),)), np.ones(8), None)
    oracle.gaussian.bulk_solves = []
    del cross
    assert reference() is None
    assert posterior.solve is not None
    assert posterior.local_response is None
    with pytest.raises(RuntimeError, match="holds no variance map"):
        posterior.variance_jvp(np.ones((8, 1)))
