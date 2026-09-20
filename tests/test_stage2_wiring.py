"""RUN-ONLY (run/svpgs-bench-1): the real engine behind fit_model.fit and the bench-real adapter, on synthetic data.

Machinery checks only: the fit is certified at the draw count's tolerance, the artifact round-trips, and scoring the
store reproduces the fitted model's in-sample predictor. Accuracy comes from bench-real and bench-sim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

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


def test_fit_runs_the_engine_and_the_saved_model_scores_the_store(tmp_path: Path) -> None:
    store, covariate, targets, _genetic = _store(tmp_path / "store", 7)
    samples = store.n_samples
    training = np.arange(samples) < samples * 4 // 5
    model = fit_model.fit(
        store=store,
        store_columns=np.arange(samples, dtype=np.int64),
        covariates=covariate[:, None],
        covariate_names=("covariate",),
        covariate_columns=np.ones((1, 1), dtype=bool),
        targets=np.where(training, targets, np.nan)[:, None],
        training=training[:, None],
        model_names=("trait/fold0",),
        trait_types=(TraitType.QUANTITATIVE,),
        research_ids=[f"person{index}" for index in range(samples)],
        log_variance_offset=None,
        budget=_budget(),
        work_dir=tmp_path,
        seed=3,
    )
    assert model.certificate["remaining_gain"][0] <= 0.5 / fit_model.DRAW_COUNT
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
