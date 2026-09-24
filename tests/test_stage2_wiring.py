"""RUN-ONLY (run/svpgs-bench-1): the real engine behind fit_model.fit and the bench-real adapter, on synthetic data.

Machinery checks only: the fit is certified at the draw count's tolerance, the artifact round-trips, and scoring the
store reproduces the fitted model's in-sample predictor. Accuracy comes from bench-real and bench-sim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmarks import svpgs_method
from benchmarks.bench_real import harness as bench_real
from sv_pgs import fit_model
from sv_pgs.artifact import load_model, predict, save_model
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import TraitType
from tests.test_full_data_fit import _store


def _budget() -> ComputeBudget:
    return ComputeBudget(
        device_kind="cpu", device_ids=(), device_names=(), device_bytes=(), device_compute_capabilities=(), host_bytes=1 << 31, cpu_threads=2
    )


@pytest.mark.parametrize("route", ["samples", "gram"])
def test_fit_runs_the_engine_and_the_saved_model_scores_the_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str) -> None:
    # Each route by name (the measured pass costs tie at this size, so either could be chosen); both are exact (the Gram
    # route corrects the band's far field, the chance LD between the store's two chromosomes, from the samples), so
    # both certify and leave no far field out.
    from sv_pgs import stage2_wiring

    monkeypatch.setattr(stage2_wiring, "pass_costs", (lambda band, source, xp: (0.0, np.inf)) if route == "samples" else (lambda band, source, xp: (np.inf, 0.0)))
    store, covariate, targets, _genetic = _store(tmp_path / "store", 7)
    samples = store.n_samples
    training = np.arange(samples) < samples * 4 // 5
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
    assert model.certificate["remaining_gain"][0] <= 0.5 / fit_model.DRAW_COUNT and model.certificate["far_field"][0] == 0.0
    save_model(tmp_path / "model", model)
    loaded = load_model(tmp_path / "model")
    prediction = predict(loaded, store, np.arange(samples), covariate[:, None], _budget())
    assert np.all(np.isfinite(prediction.predictive_mean)) and np.all(prediction.predictive_variance > 0.0)


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
