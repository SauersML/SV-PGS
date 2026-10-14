"""RUN-ONLY (run/svpgs-bench-1): the real engine behind fit_model.fit and the bench-real adapter, on synthetic data.

Machinery checks only: the fit is certified at the draw count's tolerance, the artifact round-trips, and scoring the
store reproduces the fitted model's in-sample predictor. Accuracy comes from bench-real and bench-sim.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import svpgs_method
from benchmarks.bench_real import harness as bench_real
from sv_pgs import fit_model
from sv_pgs.artifact import load_model, predict, save_model
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import TraitType
from sv_pgs.fast_scoring import ScoringPlan, score_genetic
from tests.test_full_data_fit import _store


def _budget() -> ComputeBudget:
    return ComputeBudget(
        device_kind="cpu", device_ids=(), device_names=(), device_bytes=(), device_compute_capabilities=(), host_bytes=1 << 31, cpu_threads=2
    )


@pytest.mark.parametrize("route", ["samples", "gram"])
def test_fit_runs_the_engine_and_the_saved_model_scores_the_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str) -> None:
    # Timing selects an accelerator only. Both routes retain the full-data likelihood.
    from sv_pgs import stage2_wiring

    monkeypatch.setattr(stage2_wiring, "pass_costs", (lambda band, source, xp: (0.0, np.inf)) if route == "samples" else (lambda band, source, xp: (np.inf, 0.0)))
    store, covariate, targets, _genetic = _store(tmp_path / "store", 7)
    samples = store.n_samples
    training = np.arange(samples) < samples * 4 // 5
    original_gaussian = stage2_wiring.DualGaussian
    fitted_sample_counts = []

    def training_only_gaussian(**arguments):
        fitted_sample_counts.append(arguments["source"].sample_count)
        assert arguments["training"].shape == (int(training.sum()), 1)
        assert np.all(arguments["training"] == 1)
        np.testing.assert_array_equal(arguments["probe_sample_indices"], np.flatnonzero(training))
        return original_gaussian(**arguments)

    monkeypatch.setattr(stage2_wiring, "DualGaussian", training_only_gaussian)
    model = fit_model.fit(
        fit_model.FitRequest(
            store=store,
            store_columns=np.arange(samples, dtype=np.int64),
            covariates=covariate[:, None],
            covariate_names=("covariate",),
            covariate_columns=np.ones((1, 1), dtype=bool),
            targets=np.where(training, targets, np.nan)[:, None],
            training=training[:, None],
            model_names=("trait/fold0",),
            trait_types=(TraitType.QUANTITATIVE,),
            research_ids=tuple(f"person{index}" for index in range(samples)),
            log_variance_offset=None,
            budget=_budget(),
            work_dir=tmp_path,
            seed=3,
        )
    )
    assert model.certificate["remaining_gain"][0] <= 0.5 / fit_model.DRAW_COUNT
    assert model.certificate["far_field"][0] == 0.0
    assert model.certificate["far_field_scale"][0] == 0.0
    assert model.certificate["budget_unresolved"][0] == 0
    save_model(tmp_path / "model", model)
    loaded = load_model(tmp_path / "model")
    prediction = predict(loaded, store, np.arange(samples), covariate[:, None], _budget())
    assert np.all(np.isfinite(prediction.predictive_mean)) and np.all(prediction.predictive_variance > 0.0)
    assert fitted_sample_counts == [int(training.sum())]
    assert prediction.predictive_mean.shape == (samples, 1)


def test_the_bench_real_adapter_fits_a_synthetic_gene_with_the_engine() -> None:
    generator = np.random.default_rng(8)
    samples, columns = 160, 120
    genotypes = generator.binomial(2, generator.uniform(0.05, 0.5, size=columns), size=(samples + 40, columns)).astype(np.float32)
    effects = np.zeros(columns)
    effects[generator.choice(columns, size=4, replace=False)] = generator.normal(size=4)
    phenotype = (genotypes - genotypes.mean(axis=0)) @ effects + generator.normal(size=samples + 40)
    position = np.arange(columns) * 500 + 10_000
    variants = bench_real.Variants(
        position=position, end=position, distance_to_tss=position - 40_000, is_sv=np.zeros(columns, dtype=bool),
        sv_type=np.array(["."] * columns), sv_length=np.zeros(columns, dtype=np.int64), allele_length_change=np.zeros(columns, dtype=np.int64),
        train_allele_frequency=genotypes[:samples].mean(axis=0) / 2, source=np.array(["panel"] * columns),
    )
    train = bench_real.TrainData(
        gene_id="g", chrom="chr22", tss=40_000, genotypes=genotypes[:samples], phenotype=phenotype[:samples] - phenotype[:samples].mean(),
        variants=variants, superpopulation=np.array(["AFR"] * samples), population=np.array(["YRI"] * samples),
        gene_start=39_000, gene_end=41_000, strand="+", exons=np.zeros((0, 2), dtype=np.int64), coding_exons=np.zeros((0, 2), dtype=np.int64),
    )
    prediction = svpgs_method.fit_expression(train).predict(genotypes[samples:])
    assert prediction.shape == (40,) and np.all(np.isfinite(prediction))


def test_both_reliability_sources_meet_one_contract(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A store's quality column and a caller's offsets are validated by the same function."""
    from sv_pgs.dosage_store import DosageStore
    from sv_pgs.stage2_wiring import fit_models, store_log_reliability
    from tests.test_dosage_store import _write_store

    generator = np.random.default_rng(6)
    samples, records = 12, 5
    milli = (generator.binomial(2, 0.3, size=(records, samples)) * 1000).astype(np.int64)
    _write_store(tmp_path / "store", [{"chr22": milli}])
    store = DosageStore.open(tmp_path / "store")

    # No quality column: perfect measurement, said out loud rather than assumed in silence.
    np.testing.assert_array_equal(store_log_reliability(store), np.zeros(records))
    assert "measured exactly" in capsys.readouterr().err

    store.variant_table.annotations["quality"] = np.array([1.0, 0.5, 0.0, 0.25, 0.75])
    np.testing.assert_allclose(store_log_reliability(store), [0.0, np.log(0.5), -np.inf, np.log(0.25), np.log(0.75)])

    for corrupt in ([1.0, 1.4, 0.5, 0.25, 0.75], [1.0, -0.3, 0.5, 0.25, 0.75], [1.0, np.nan, 0.5, 0.25, 0.75]):
        store.variant_table.annotations["quality"] = np.array(corrupt)
        with pytest.raises(ValueError, match="the store's quality column: every record's log reliability"):
            store_log_reliability(store)

    with pytest.raises(ValueError, match="the caller's log_variance_offset: every record's log reliability"):
        fit_models(
            store=store,
            store_columns=np.arange(samples, dtype=np.int64),
            covariates=np.ones((samples, 1)),
            covariate_columns=np.ones((1, 1), dtype=bool),
            targets=np.zeros((samples, 1)),
            training=np.ones((samples, 1), dtype=bool),
            trait_types=(TraitType.QUANTITATIVE,),
            log_variance_offset=np.array([0.0, 0.0, 0.3, 0.0, 0.0]),
            budget=_budget(),
            work_dir=tmp_path / "fit",
            seed=1,
            draw_count=8,
        )


def _made(path: Path) -> Path:
    path.mkdir(parents=True)
    return path


def _fit_one_model(root: Path, milli: np.ndarray, training: np.ndarray, targets: np.ndarray, work_dir: Path):
    from sv_pgs.dosage_store import DosageStore
    from sv_pgs.stage2_wiring import fit_models
    from tests.test_dosage_store import _write_store

    _write_store(root, [{"chr22": milli}])
    store = DosageStore.open(root)
    work_dir.mkdir(parents=True)
    return store, fit_models(
        store=store,
        store_columns=np.arange(store.n_samples, dtype=np.int64),
        covariates=np.ones((store.n_samples, 1)),
        covariate_columns=np.ones((1, 1), dtype=bool),
        targets=targets[:, None],
        training=training[:, None],
        trait_types=(TraitType.QUANTITATIVE,),
        log_variance_offset=None,
        budget=_budget(),
        work_dir=work_dir,
        seed=1,
        draw_count=4,
    )


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("monomorphic_store", "no store record carries signal"),
        ("monomorphic_on_training", "monomorphic on these training rows"),
        ("covariates_explain_the_target", "explain every training target to working precision"),
    ],
)
def test_a_training_set_with_no_genetic_column_fits_the_covariates_alone(tmp_path: Path, case: str, reason: str) -> None:
    """I07: no candidate record, every candidate monomorphic on the training rows, and a phenotype the covariates
    explain to working precision are covariate-only outcomes of the model, not crashes. They raised ValueError from
    .max() on an empty bincount and from arange in the start lattice at a zero residual variance."""
    generator = np.random.default_rng(6)
    samples, records = 40, 12
    training = np.ones(samples, dtype=bool)
    targets = generator.normal(size=samples)
    if case == "monomorphic_store":
        milli = np.zeros((records, samples), dtype=np.int64)
    elif case == "monomorphic_on_training":
        training = np.arange(samples) < samples // 2
        milli = (generator.binomial(2, 0.3, size=(records, samples)) * 1000).astype(np.int64)
        milli[:, training] = 0
    else:
        milli = (generator.binomial(2, 0.3, size=(records, samples)) * 1000).astype(np.int64)
        targets = np.full(samples, 3.5)
    store, fitted = _fit_one_model(tmp_path / "store", milli, training, targets, tmp_path / "work")
    (scoring,) = fitted.scoring
    expected = float(np.mean(targets[training]))
    assert scoring.store_rows.shape == (0,) and scoring.coefficients.shape == (0,)
    np.testing.assert_allclose(scoring.alpha, [expected], rtol=0.0, atol=1e-12)
    (refusal,) = fitted.certificate.refusals
    assert refusal.startswith("model 0: ") and reason in refusal
    assert fitted.certificate.outer_criterion_met.tolist() == [True]
    rows = int(training.sum())
    if case == "covariates_explain_the_target":
        # The residual is zero to the covariates' own rank tolerance, so the reported variance is below it, not a floor.
        resolution = max(rows, 1) * np.finfo(np.float64).eps * float(np.linalg.norm(targets[training]))
        expected_noise = 0.0
        assert 0.0 <= fitted.noise_variance[0] <= resolution * resolution / (rows - 1)
    else:
        expected_noise = float(np.sum((targets[training] - expected) ** 2) / (rows - 1))
        np.testing.assert_allclose(fitted.noise_variance, [expected_noise], rtol=1e-12, atol=1e-12)
    # The covariate-only model saves, loads and scores like any other: every person gets the covariate prediction.
    model = fit_model.fit(
        fit_model.FitRequest(
            store=store, store_columns=np.arange(samples, dtype=np.int64), covariates=np.zeros((samples, 0)),
            covariate_names=(), covariate_columns=np.zeros((1, 0), dtype=bool), targets=targets[:, None],
            training=training[:, None], model_names=("trait/fold0",), trait_types=(TraitType.QUANTITATIVE,),
            research_ids=tuple(f"person{index}" for index in range(samples)), log_variance_offset=None,
            budget=_budget(), work_dir=_made(tmp_path / "fit"), seed=2,
        )
    )
    save_model(tmp_path / "model", model)
    prediction = predict(load_model(tmp_path / "model"), store, np.arange(samples), np.zeros((samples, 0)), _budget())
    np.testing.assert_allclose(prediction.predictive_mean[:, 0], expected, rtol=0.0, atol=1e-12)
    # Its predictive variance is the noise's plus the intercept's own posterior variance, sigma^2 / n.
    np.testing.assert_allclose(prediction.predictive_variance[:, 0], model.noise_variance[0] * (1.0 + 1.0 / rows), rtol=1e-12, atol=0.0)
    assert expected_noise == 0.0 or abs(model.noise_variance[0] - expected_noise) <= 1e-12


def test_the_candidate_prefilter_keeps_every_record_stage0_keeps(tmp_path: Path) -> None:
    from sv_pgs.config import ModelConfig
    from sv_pgs.dosage_store import DosageStore
    from sv_pgs.genotype_statistics import DosageStoreTileSource, compute_genotype_statistics
    from sv_pgs.stage2_wiring import stage0_candidates
    from tests.test_dosage_store import _write_store

    generator = np.random.default_rng(4)
    samples, records = 90, 200
    frequency = np.concatenate([np.zeros(10), generator.uniform(0.0, 0.03, size=100), generator.uniform(0.03, 0.5, size=90)])
    milli = (generator.binomial(2, frequency[:, None], size=(records, samples)) * 1000).astype(np.int64)
    _write_store(tmp_path / "store", [{"chr22": milli}])
    store = DosageStore.open(tmp_path / "store")
    columns = np.arange(samples, dtype=np.int64)
    everything = np.arange(records, dtype=np.int64)
    candidates = stage0_candidates(store, columns, np.zeros(records), ModelConfig(minimum_minor_allele_frequency=0.0))
    polymorphic = int(np.sum(np.ptp(milli, axis=1) > 0))
    assert candidates.shape[0] == polymorphic  # only the monomorphic records go
    candidates = stage0_candidates(store, columns, np.zeros(records), ModelConfig())
    statistics = compute_genotype_statistics(
        DosageStoreTileSource(store, everything), columns, np.ones((samples, 1)), generator.normal(size=(samples, 1)), ModelConfig(), _budget(), 256, tmp_path / "ld"
    )
    assert set(np.asarray(statistics.active_rows).tolist()) <= set(candidates.tolist())
    assert candidates.shape[0] < records


def test_models_with_the_same_rows_covariates_and_likelihood_form_one_group() -> None:
    from sv_pgs.stage2_wiring import model_groups

    generator = np.random.default_rng(3)
    rows = 30
    store_columns = generator.permutation(rows + 10)[:rows].astype(np.int64)
    first = generator.uniform(size=rows) < 0.7
    second = first.copy()
    second[np.flatnonzero(first)[0]] = False
    covariates = np.array([True, False, True])
    # Models: two traits on the first rows, one on other rows, one on the first rows with other covariates, two binary
    # models on the first rows (a binary model is a group of its own), and a third trait on the first rows again.
    training = np.column_stack([first, first, second, first, first, first, first])
    covariate_columns = np.array([covariates, covariates, covariates, ~covariates, covariates, covariates, covariates])
    quantitative, binary = TraitType.QUANTITATIVE, TraitType.BINARY
    groups = model_groups(store_columns, covariate_columns, training, (quantitative, quantitative, quantitative, quantitative, binary, binary, quantitative))
    assert [group.tolist() for group in groups] == [[0, 1, 6], [2], [3], [4], [5]]


def _fitted_group(store, covariate: np.ndarray, targets: np.ndarray, training: np.ndarray, work_dir: Path, seed: int, **arguments):
    from sv_pgs.stage2_wiring import fit_models

    work_dir.mkdir(parents=True)
    return fit_models(
        store=store, store_columns=np.arange(store.n_samples, dtype=np.int64), covariates=np.column_stack([np.ones(store.n_samples), covariate]),
        covariate_columns=np.ones((training.shape[1], 2), dtype=bool), targets=np.where(training, targets, 0.0), training=training,
        trait_types=(TraitType.QUANTITATIVE,) * training.shape[1], log_variance_offset=None, budget=_budget(), work_dir=work_dir, seed=seed,
        draw_count=16, **arguments,
    )


def _three_traits(root: Path):
    """The synthetic store (``test_full_data_fit._store``), its covariate, and three traits on its genetic value each on
    two training sets (all samples but the last fifth, all but the first): (store, covariate, model targets [n, 6],
    training [n, 6], the two held-out masks), the models in trait-major order: (trait 0, rows 0), (trait 0, rows 1), ..."""
    store, covariate, targets, genetic = _store(root, 7)
    samples = store.n_samples
    generator = np.random.default_rng(12)
    traits = np.column_stack([
        targets, 0.4 * covariate + 0.8 * genetic + 0.6 * generator.standard_normal(samples),
        -0.2 * covariate - genetic + 0.9 * generator.standard_normal(samples),
    ])
    held_out = [np.arange(samples) >= samples * 4 // 5, np.arange(samples) < samples // 5]
    training = np.column_stack([~held_out[rows] for _trait in range(3) for rows in range(2)])
    return store, covariate, np.repeat(traits, 2, axis=1), training, held_out


@pytest.mark.slow  # seven fits of the synthetic store per route: 3-10 min each in the shared 6-CPU VM (batch-fit, 2026-10-13)
@pytest.mark.parametrize("route", ["samples", "gram"])
def test_a_group_fits_each_model_as_it_would_be_fitted_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str) -> None:
    """Three traits on two training sets: one fit per training set, with the traits as its columns, gives each model the
    fit it gets alone, up to the Monte Carlo error of the probes and draws it shares with its group and the start lattice
    that covers its group's. Whether the outer criterion is met is the engine's own outcome on this store either way, and
    it flips both ways between a model's fits.

    Each model alone is fitted on the sample route, the full-data likelihood both routes share and the reference route:
    alone on the band route this store's model 4 ends uncertified (its coefficients correlate 0.33 with its sample-route
    fit's) and is fitted again on the sample route (the next test), while its group's band-route fit correlates 0.997 with
    the group's sample-route fit. Timing alone chooses the route, so both are pinned."""
    from sv_pgs import stage2_wiring
    from tests.test_full_data_fit import _StoreCodes

    store, covariate, model_targets, training, held_out = _three_traits(tmp_path / "store")
    monkeypatch.setattr(stage2_wiring, "pass_costs", (lambda band, source, xp: (0.0, np.inf)) if route == "samples" else (lambda band, source, xp: (np.inf, 0.0)))
    together = _fitted_group(store, covariate, model_targets, training, tmp_path / "together", 5)
    monkeypatch.setattr(stage2_wiring, "pass_costs", lambda band, source, xp: (0.0, np.inf))
    for model in range(6):
        alone = _fitted_group(store, covariate, model_targets[:, [model]], training[:, [model]], tmp_path / f"alone{model}", 5)
        joint, single = together.scoring[model], alone.scoring[0]
        np.testing.assert_array_equal(joint.store_rows, single.store_rows)
        assert np.corrcoef(joint.coefficients, single.coefficients)[0, 1] > 0.99
        rows = held_out[model % 2]
        scores = [
            score_genetic(_StoreCodes(store), ScoringPlan.from_models([scoring]), _budget()).means[rows, 0] for scoring in (joint, single)
        ]
        assert np.corrcoef(*scores)[0, 1] > 0.99
        assert float(np.max(np.abs(scores[0] - scores[1]))) <= 0.25 * float(np.std(scores[1]))
        np.testing.assert_allclose(together.noise_variance[model], alone.noise_variance[0], rtol=0.05)


@pytest.mark.parametrize("refused", ["joint", "every"])
def test_a_band_route_start_with_no_fixed_point_costs_no_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refused: str) -> None:
    """A group whose band-route joint fit has no certified start is fitted model by model on the same Stage 0, and a
    model whose band-route fit alone has none either is fitted on the sample route; each one's refusals say why. A stub
    refuses the band route's starts: the joint ones, or every one."""
    from sv_pgs import stage2_wiring

    store, covariate, model_targets, training, _held_out = _three_traits(tmp_path / "store")
    monkeypatch.setattr(stage2_wiring, "pass_costs", lambda band, source, xp: (np.inf, 0.0))
    engine = stage2_wiring.fit_full_data
    calls: list[tuple[int, bool]] = []

    def refusing(**arguments):
        on_band = arguments["band"] is not None
        if "starts" in arguments:
            # A lattice check's refit, continued from a fit that had its start.
            return engine(**arguments)
        calls.append((arguments["gaussian"].model_count, on_band))
        if on_band and (refused == "every" or arguments["gaussian"].model_count > 1):
            raise FloatingPointError("a starting point has no certified EP fixed point (stub)")
        return engine(**arguments)

    monkeypatch.setattr(stage2_wiring, "fit_full_data", refusing)
    # Two traits on the first training set: one group.
    fitted = _fitted_group(store, covariate, model_targets[:, [0, 2]], training[:, [0, 2]], tmp_path / "work", 5)
    # The joint start, then each model's own; where that is refused too, the model's sample-route fit. (A model the gate
    # would quarantine is fitted on the sample route after these.)
    expected = [(2, True), (1, True), (1, True)] if refused == "joint" else [(2, True), (1, True), (1, False), (1, True), (1, False)]
    assert calls[: len(expected)] == expected
    for model in range(2):
        reasons = [reason for reason in fitted.certificate.refusals if reason.startswith(f"model {model}: ")]
        assert any("joint fit of models 0, 1 had no certified start, so each was fitted on its own" in reason for reason in reasons)
        assert any("fitted on the sample route" in reason for reason in reasons) == (refused == "every")
        assert fitted.scoring[model].store_rows.shape[0] > 0


def test_a_band_route_fit_the_gate_would_quarantine_is_fitted_again_on_the_sample_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Model 4 alone on the band route, seed 5: its fit ends uncertified (its outer loop returns from outside every
    basin; ``model_acceptance`` would quarantine it, and its coefficients correlate 0.33 with the sample route's), so it
    is fitted again alone on the sample route, which the gate accepts, and its refusals say why."""
    from sv_pgs import stage2_wiring
    from sv_pgs.fit_model import certificate_parts
    from sv_pgs.model_acceptance import acceptance

    store, covariate, model_targets, training, _held_out = _three_traits(tmp_path / "store")
    monkeypatch.setattr(stage2_wiring, "pass_costs", lambda band, source, xp: (np.inf, 0.0))
    fitted = _fitted_group(store, covariate, model_targets[:, [4]], training[:, [4]], tmp_path / "work", 5)
    (reason,) = [reason for reason in fitted.certificate.refusals if "quarantined" in reason]
    assert reason.startswith("model 0: the band route's fit would be quarantined (outer_criterion_met") and reason.endswith("fitted again alone on the sample route")
    parts = certificate_parts(fitted.certificate, 1)
    decision = acceptance(SimpleNamespace(certificate=parts.terms, refusals=parts.refusals, model_names=("model 4",), scoring=tuple(fitted.scoring)))
    assert decision["accepted"], decision["issues"]


def test_a_group_resumes_only_when_every_model_is_checkpointed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sv_pgs import stage2_wiring

    generator = np.random.default_rng(9)
    samples, records = 40, 12
    milli = (generator.binomial(2, 0.3, size=(records, samples)) * 1000).astype(np.int64)
    from sv_pgs.dosage_store import DosageStore
    from tests.test_dosage_store import _write_store

    _write_store(tmp_path / "store", [{"chr22": milli}])
    store = DosageStore.open(tmp_path / "store")
    fitted_groups: list[list[int]] = []

    def covariates_alone(_store, _columns, covariates, targets, _measurement, _budget, work_dir, _seed, draw_count, models, *_rest):
        # A stand-in for the engine: each model's covariates-only fit, which checkpoints like any other.
        assert work_dir.is_dir()
        fitted_groups.append(models.tolist())
        return [
            stage2_wiring._null_genetic_model(stage2_wiring._covariate_least_squares(covariates, targets[:, model]), draw_count, "stand-in")
            for model in range(targets.shape[1])
        ]

    monkeypatch.setattr(stage2_wiring, "_fit_group", covariates_alone)
    training = np.column_stack([np.ones(samples, dtype=bool)] * 2 + [np.arange(samples) < samples // 2])
    targets = generator.normal(size=(samples, 3))

    def fit():
        return stage2_wiring.fit_models(
            store=store, store_columns=np.arange(samples, dtype=np.int64), covariates=np.ones((samples, 1)),
            covariate_columns=np.ones((3, 1), dtype=bool), targets=targets, training=training, trait_types=(TraitType.QUANTITATIVE,) * 3,
            log_variance_offset=None, budget=_budget(), work_dir=tmp_path / "work", seed=4, draw_count=4, checkpoint_key="key",
        )

    first = fit()
    assert fitted_groups == [[0, 1], [2]]
    checkpoints = tmp_path / "work" / "checkpoints" / "key"
    assert sorted(path.name for path in checkpoints.iterdir()) == ["model0.npz", "model1.npz", "model2.npz"]
    resumed = fit()
    assert fitted_groups == [[0, 1], [2]]
    np.testing.assert_array_equal(resumed.noise_variance, first.noise_variance)
    # One model of the pair lost its checkpoint: the pair is fitted again, the other group resumes.
    (checkpoints / "model1.npz").unlink()
    again = fit()
    assert fitted_groups == [[0, 1], [2], [0, 1]]
    np.testing.assert_array_equal(again.noise_variance, first.noise_variance)
    assert again.certificate.refusals == ("model 0: stand-in", "model 1: stand-in", "model 2: stand-in")
