"""SV-PGS as a bench-sim and bench-real method: the store each adapter writes from a harness view, and its scores.

The engine driver is the stub of tests/test_fit_model.py, so these tests pin the adapters, not the fit.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks import svpgs_method
from benchmarks.bench_real import harness as bench_real
from benchmarks.bench_sim import harness as bench_sim
from sv_pgs import fit_model
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import VariantClass
from sv_pgs.dosage_store import CODES_PER_DOSAGE, DosageStore
from sv_pgs.fast_scoring import SIGNED_CODE_OFFSET, ScoringModel
from tests.test_fit_model import _StubDriver

_EPSILON = np.finfo(np.float64).eps
_CLASSES = tuple(VariantClass)


@pytest.fixture
def driver(monkeypatch: pytest.MonkeyPatch) -> _StubDriver:
    stub = _StubDriver()
    monkeypatch.setattr(fit_model, "fit_models", stub)
    return stub


def _scores(scoring: ScoringModel, signed: np.ndarray) -> tuple[np.ndarray, float]:
    """X beta over signed codes [rows, samples] in the model's store rows, and its fp64 rounding bound."""
    standardized = (signed - scoring.signed_means[:, None]) / scoring.signed_scales[:, None]
    score = standardized.T @ scoring.coefficients
    return score, float(np.max(_EPSILON * (np.abs(standardized.T) @ np.abs(scoring.coefficients)) * signed.shape[0]))


def _expected_class(code: int, ref_len: int, alt_len: int) -> VariantClass:
    if code == 2:
        return VariantClass.STR_VNTR_REPEAT
    if ref_len == alt_len == 1:
        return VariantClass.SNV
    if ref_len != alt_len:
        return VariantClass.DELETION if ref_len > alt_len else VariantClass.INSERTION
    return VariantClass.OTHER_COMPLEX_SV


# ---------------------------------------------------------------------------
# bench-sim
# ---------------------------------------------------------------------------

_RECORDS = 40
_SIM_SAMPLES = 70


def _bench_sim_views(generator: np.random.Generator, trait_type: str) -> tuple[bench_sim.TrainData, bench_sim.ScoreData, dict]:
    observed = generator.integers(0, 255, size=(_RECORDS + 5, _SIM_SAMPLES), dtype=np.uint8)
    records = np.sort(generator.choice(_RECORDS + 5, size=_RECORDS, replace=False))
    cls = np.tile(np.arange(4, dtype=np.int8), _RECORDS // 4)
    ref_len = np.where(cls == 1, generator.integers(1, 4, size=_RECORDS), 1)
    alt_len = np.where(cls == 1, generator.integers(1, 4, size=_RECORDS), 1)
    alt_len = np.where(cls >= 2, generator.integers(2, 900, size=_RECORDS), alt_len)
    position = np.sort(generator.integers(10_000, 10_060, size=_RECORDS))
    quality = generator.uniform(size=_RECORDS)
    quality[[3, 17]] = 0.0
    variants = {
        "pos": position,
        "cm": position / 1e6,
        "cls": cls,
        "len_change": alt_len - ref_len,
        "ref_len": ref_len,
        "alt_len": alt_len,
        "in_gene": generator.uniform(size=_RECORDS) < 0.5,
        "in_exon": np.zeros(_RECORDS, dtype=bool),
        "log_tss_distance": generator.uniform(0, 12, size=_RECORDS),
        "in_repeat": generator.uniform(size=_RECORDS) < 0.3,
        "log_sv_length": np.where(cls == 3, np.log(alt_len), 0.0),
        "imputation_info": quality,
        "class_names": np.array(["SNV", "INDEL", "TR", "SV"]),
    }
    columns = generator.permutation(_SIM_SAMPLES)
    train_columns, test_columns = np.sort(columns[:50]), np.sort(columns[50:])
    covariates = generator.normal(size=(_SIM_SAMPLES, 3))
    phenotype = generator.normal(size=_SIM_SAMPLES) if trait_type == "quantitative" else (generator.uniform(size=_SIM_SAMPLES) < 0.3).astype(np.float64)
    names = ("sex", "age", "pc1")
    train = bench_sim.TrainData(
        variants=variants, covariates=covariates[train_columns], covariate_names=names, phenotype=phenotype[train_columns],
        trait_type=trait_type, prevalence=None if trait_type == "quantitative" else 0.3, cores=1,
        _observed=observed, _columns=train_columns, _records=records,
    )
    test = bench_sim.ScoreData(variants=variants, covariates=covariates[test_columns], covariate_names=names,
                               _observed=observed, _columns=test_columns, _records=records)
    return train, test, variants


def test_the_bench_sim_store_holds_the_training_codes_and_the_public_table(driver: _StubDriver) -> None:
    train, _test, variants = _bench_sim_views(np.random.default_rng(1), "quantitative")
    svpgs_method.fit(train)
    store: DosageStore = driver.calls[0]["store"]
    kept = np.flatnonzero(variants["imputation_info"] > 0.0)
    np.testing.assert_array_equal(store.read_codes(0, store.n_variants), train.codes(kept))
    table = store.variant_table
    np.testing.assert_array_equal(table.position, variants["pos"][kept])
    assert [_CLASSES[code] for code in table.variant_class] == [
        _expected_class(*row) for row in zip(variants["cls"][kept], variants["ref_len"][kept], variants["alt_len"][kept])
    ]
    firsts = np.array([np.flatnonzero(table.position == position)[0] for position in table.position])
    np.testing.assert_array_equal(table.group_first, firsts)
    assert sorted(table.annotations) == ["in_gene", "in_repeat", "len_change", "log_sv_length", "log_tss_distance", "quality"]
    np.testing.assert_array_equal(table.annotations["quality"], variants["imputation_info"][kept])
    np.testing.assert_array_equal(driver.calls[0]["covariates"][:, 1:], train.covariates)


@pytest.mark.parametrize("trait_type", ["quantitative", "binary"])
def test_bench_sim_scores_are_the_fitted_scores_of_the_test_codes(driver: _StubDriver, trait_type: str) -> None:
    train, test, variants = _bench_sim_views(np.random.default_rng(2), trait_type)
    prediction = svpgs_method.fit(train).score(test)
    scoring = driver.results[0].scoring[0]
    kept = np.flatnonzero(variants["imputation_info"] > 0.0)
    signed = test.codes(kept).astype(np.float64) - SIGNED_CODE_OFFSET
    total, rounding = _scores(scoring, signed[scoring.store_rows])
    np.testing.assert_allclose(prediction["total"], total, rtol=0.0, atol=rounding)
    structural_rows = np.isin(variants["cls"][kept][scoring.store_rows], (2, 3))
    structural, rounding = _scores(svpgs_method._restricted(scoring, structural_rows), signed[scoring.store_rows[structural_rows]])
    np.testing.assert_allclose(prediction["structural"], structural, rtol=0.0, atol=rounding)


def test_bench_sim_refuses_an_imputation_quality_outside_zero_one(driver: _StubDriver) -> None:
    train, _test, variants = _bench_sim_views(np.random.default_rng(3), "quantitative")
    variants["imputation_info"][0] = np.nan
    with pytest.raises(ValueError, match="r\\^2 in \\[0, 1\\]"):
        svpgs_method.fit(train)
    assert driver.calls == []


# ---------------------------------------------------------------------------
# bench-real
# ---------------------------------------------------------------------------

_COLUMNS = 12
_REAL_SAMPLES = 60


def _bench_real_train(generator: np.random.Generator) -> tuple[bench_real.TrainData, np.ndarray]:
    # Panel rows in position order, then PanGenie rows appended out of order, as the dataset stores them.
    position = np.array([100, 150, 150, 220, 300, 410, 500, 640, 700, 810, 140, 505])
    end = position + np.array([0, 0, 2, 0, 350, 0, 0, 0, 0, 0, 0, 90])
    sv_type = np.array([".", ".", "DEL", ".", "DEL", "INS", ".", "DUP", ".", ".", "INS", "DEL"])
    is_sv = np.array([False, False, False, False, True, False, False, True, False, False, True, True])
    change = np.array([0, 0, -2, 0, -350, 3, 0, 0, 0, 0, 480, -90])
    variants = bench_real.Variants(
        position=position, end=end, distance_to_tss=position - 400, is_sv=is_sv, sv_type=sv_type,
        sv_length=np.abs(change), allele_length_change=change, train_allele_frequency=np.full(_COLUMNS, 0.3),
        source=np.array(["panel"] * 10 + ["pangenie"] * 2),
    )
    genotypes = generator.binomial(2, 0.35, size=(_REAL_SAMPLES + 20, _COLUMNS)).astype(np.float32)
    train = bench_real.TrainData(
        gene_id="g", chrom="chr6", tss=400, genotypes=genotypes[:_REAL_SAMPLES], phenotype=generator.normal(size=_REAL_SAMPLES),
        variants=variants, superpopulation=np.array(["AFR"] * _REAL_SAMPLES), population=np.array(["YRI"] * _REAL_SAMPLES),
        gene_start=380, gene_end=420, strand="+", exons=np.zeros((0, 2), dtype=np.int64), coding_exons=np.zeros((0, 2), dtype=np.int64),
    )
    return train, genotypes[_REAL_SAMPLES:]


def test_the_bench_real_store_holds_the_calls_in_position_order(driver: _StubDriver) -> None:
    train, _test = _bench_real_train(np.random.default_rng(4))
    svpgs_method.fit_expression(train)
    store: DosageStore = driver.calls[0]["store"]
    order = np.argsort(train.variants.position, kind="stable")
    np.testing.assert_array_equal(store.read_codes(0, store.n_variants), (train.genotypes.T[order] * CODES_PER_DOSAGE).astype(np.uint8))
    table = store.variant_table
    assert list(store.chromosomes) == ["chr6"]
    np.testing.assert_array_equal(table.position, train.variants.position[order])
    expected = {".": VariantClass.SNV, "DEL": VariantClass.DELETION, "INS": VariantClass.INSERTION, "DUP": VariantClass.DUPLICATION}
    assert [_CLASSES[code] for code in table.variant_class] == [expected[token] for token in train.variants.sv_type[order]]
    assert "train_allele_frequency" not in table.annotations
    assert sorted(table.annotations) == ["allele_length_change", "log1p_sv_length", "log1p_tss_distance", "source", "sv_type"]
    assert table.annotation_legends["source"] == tuple(sorted({"panel", "pangenie"}))
    assert driver.calls[0]["covariates"].shape == (_REAL_SAMPLES, 1)


def test_bench_real_predictions_match_the_store_scorer_and_mask_svs_exactly(tmp_path, driver: _StubDriver) -> None:
    train, test = _bench_real_train(np.random.default_rng(5))
    predictor = svpgs_method.fit_expression(train)
    scoring = driver.results[0].scoring[0]
    order = np.argsort(train.variants.position, kind="stable")
    signed = test.T[order].astype(np.float64) * CODES_PER_DOSAGE - SIGNED_CODE_OFFSET
    expected, rounding = _scores(scoring, signed[scoring.store_rows])
    np.testing.assert_allclose(predictor.predict(test), expected + scoring.alpha[0], rtol=0.0, atol=rounding + _EPSILON * abs(scoring.alpha[0]))
    masked = bench_real._without_structural_variants(train, test)
    masked_signed = masked.T[order].astype(np.float64) * CODES_PER_DOSAGE - SIGNED_CODE_OFFSET
    _masked_scores, masked_rounding = _scores(scoring, masked_signed[scoring.store_rows])
    sv_rows = train.variants.is_sv[order][scoring.store_rows]
    # The harness's arrays are float32; the difference is taken in float64, as predict widens them before any arithmetic.
    widened = test[:, predictor.columns].astype(np.float64) - masked[:, predictor.columns].astype(np.float64)
    change = CODES_PER_DOSAGE * widened / scoring.signed_scales
    sv_part = change[:, sv_rows] @ scoring.coefficients[sv_rows]
    sv_rounding = float(np.max(_EPSILON * (np.abs(change) @ np.abs(scoring.coefficients)) * _COLUMNS))
    # Each prediction is within its rounding bound of X beta, and the SV part within its own.
    np.testing.assert_allclose(
        predictor.predict(test) - predictor.predict(masked), sv_part, rtol=0.0, atol=rounding + masked_rounding + sv_rounding
    )


def test_bench_real_refuses_training_genotypes_that_are_not_allele_counts(driver: _StubDriver) -> None:
    train, _test = _bench_real_train(np.random.default_rng(6))
    train.genotypes[0, 0] = 0.5
    with pytest.raises(ValueError, match="allele counts"):
        svpgs_method.fit_expression(train)
    assert driver.calls == []


def test_each_bench_real_fit_gets_one_cores_share(monkeypatch: pytest.MonkeyPatch) -> None:
    machine = ComputeBudget(
        device_kind="cuda", device_ids=(0,), device_names=("gpu",), device_bytes=(1 << 34,), device_compute_capabilities=((8, 6),),
        host_bytes=(1 << 36) + 5, cpu_threads=16,
    )
    monkeypatch.setattr(svpgs_method, "detect_compute_budget", lambda: machine)
    budget = svpgs_method.one_core_budget()
    assert (budget.device_kind, budget.cpu_threads, budget.host_bytes) == ("cpu", 1, machine.host_bytes // machine.cpu_threads)
