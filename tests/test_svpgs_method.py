"""SV-PGS as a bench-sim and bench-real method: the store each adapter writes from a harness view, and its scores.

The engine driver is the stub of tests/test_fit_model.py, so these tests pin the adapters, not the fit.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks import svpgs_method
from benchmarks.bench_real import harness as bench_real
from benchmarks.bench_sim import harness as bench_sim
from sv_pgs import fit_model
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import TraitType, VariantClass
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
        truth_half=np.zeros(train_columns.shape[0], dtype=bool), reads=None,
        _observed=observed, _columns=train_columns, _records=records,
    )
    test = bench_sim.ScoreData(variants=variants, covariates=covariates[test_columns], covariate_names=names, reads=None,
                               _observed=observed, _columns=test_columns, _records=records)
    return train, test, variants


def test_the_bench_sim_store_holds_the_training_codes_and_the_public_table(driver: _StubDriver) -> None:
    train, _test, variants = _bench_sim_views(np.random.default_rng(1), "quantitative")
    svpgs_method.fit(train)
    # The store as the fit saw it (the adapter closes and removes it once the fit returns).
    codes, table = driver.store_contents[0]
    kept = np.flatnonzero(variants["imputation_info"] > 0.0)
    np.testing.assert_array_equal(codes, train.codes(kept))
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


class _SmallNStub:
    """Marginal regression of the centred target on the standardized training codes, as ``fit_small_n`` returns it."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **arguments: Any) -> Any:
        self.calls.append(arguments)
        signed = arguments["codes"].astype(np.float64) - SIGNED_CODE_OFFSET
        live = np.flatnonzero(signed.std(axis=0) > 0.0)
        means, scales = signed[:, live].mean(axis=0), signed[:, live].std(axis=0)
        standardized = (signed[:, live] - means) / scales
        target = arguments["target"] - arguments["target"].mean()
        coefficients = standardized.T @ target / (target.shape[0] * live.shape[0])
        generator = np.random.default_rng(arguments["seed"])
        alpha = np.linalg.lstsq(arguments["covariates"], arguments["target"], rcond=None)[0]
        scoring = ScoringModel(
            store_rows=live.astype(np.int64), signed_means=means, signed_scales=scales, coefficients=coefficients,
            posterior_draws=coefficients[:, None] + generator.normal(scale=np.abs(coefficients).mean(), size=(live.shape[0], 3)),
            alpha=alpha, trait_type=TraitType.QUANTITATIVE,
            covariate_draws=np.repeat(alpha[:, None], 3, axis=1),
            covariate_covariance=np.zeros((alpha.size, alpha.size)),
            gaussian_posterior=True,
            predictive_intercept_shift=0.0,
        )
        return type("SmallNFit", (), {"scoring": scoring})()


@pytest.fixture
def small_n(monkeypatch: pytest.MonkeyPatch) -> _SmallNStub:
    stub = _SmallNStub()
    monkeypatch.setattr(svpgs_method, "fit_small_n", stub)
    return stub


def _bench_real_train(generator: np.random.Generator) -> tuple[bench_real.TrainData, np.ndarray]:
    # Panel rows in position order, then PanGenie rows appended out of order, as the dataset stores them. Rows 4, 7 and
    # 11 are symbolic SVs (harness change 0, END past POS): a deletion, a duplication and a deletion; row 10 is a
    # symbolic insertion (END = POS) and row 2 a sequence-resolved 2 bp deletion typed DEL.
    position = np.array([100, 150, 150, 220, 300, 410, 500, 640, 700, 810, 140, 505])
    end = position + np.array([0, 0, 2, 0, 350, 0, 0, 900, 0, 0, 0, 90])
    sv_type = np.array([".", ".", "DEL", ".", "DEL", "INS", ".", "DUP", ".", ".", "INS", "DEL"])
    is_sv = np.array([False, False, False, False, True, False, False, True, False, False, True, True])
    change = np.array([0, 0, -2, 0, 0, 3, 0, 0, 0, 0, 0, 0])
    sv_length = np.array([0, 0, 0, 0, 350, 0, 0, 900, 0, 0, 480, 90])
    variants = bench_real.Variants(
        position=position, end=end, distance_to_tss=position - 400, is_sv=is_sv, sv_type=sv_type,
        sv_length=sv_length, allele_length_change=change, train_allele_frequency=np.full(_COLUMNS, 0.3),
        source=np.array(["panel"] * 10 + ["pangenie"] * 2),
    )
    genotypes = generator.binomial(2, 0.35, size=(_REAL_SAMPLES + 20, _COLUMNS)).astype(np.float32)
    train = bench_real.TrainData(
        gene_id="g", chrom="chr6", tss=400, genotypes=genotypes[:_REAL_SAMPLES], phenotype=generator.normal(size=_REAL_SAMPLES),
        variants=variants, superpopulation=np.array(["AFR"] * _REAL_SAMPLES), population=np.array(["YRI"] * _REAL_SAMPLES),
        gene_start=380, gene_end=420, strand="+", exons=np.zeros((0, 2), dtype=np.int64), coding_exons=np.zeros((0, 2), dtype=np.int64),
    )
    return train, genotypes[_REAL_SAMPLES:]


def test_symbolic_svs_get_their_signed_length_change_from_their_type() -> None:
    train, _test = _bench_real_train(np.random.default_rng(3))
    np.testing.assert_array_equal(svpgs_method.bench_real_signed_change(train.variants), [0, 0, -2, 0, -350, 3, 0, 900, 0, 0, 480, -90])


def test_the_full_model_gets_the_training_codes_and_the_sv_classes(small_n: _SmallNStub) -> None:
    train, _test = _bench_real_train(np.random.default_rng(4))
    svpgs_method.fit_expression(train)
    (call,) = small_n.calls
    np.testing.assert_array_equal(call["codes"], (train.genotypes * CODES_PER_DOSAGE).astype(np.uint8))
    np.testing.assert_array_equal(call["target"], train.phenotype)
    np.testing.assert_array_equal(call["covariates"], np.ones((_REAL_SAMPLES, 1)))
    expected = {".": VariantClass.SNV, "DEL": VariantClass.DELETION, "INS": VariantClass.INSERTION, "DUP": VariantClass.DUPLICATION}
    assert [_CLASSES[code] for code in call["variant_class"]] == [expected[token] for token in train.variants.sv_type]
    assert call["log_variance_offset"] is None and call["draw_count"] == fit_model.DRAW_COUNT


@pytest.mark.parametrize("centering", ["training", "target"])
def test_bench_real_predictions_are_the_fitted_scores_and_mask_svs_exactly(small_n: _SmallNStub, centering: str) -> None:
    train, test = _bench_real_train(np.random.default_rng(5))
    arm = svpgs_method.fit_expression if centering == "training" else svpgs_method.fit_expression_target_centered
    predictor = arm(train)
    scoring = predictor.scoring
    signed = test[:, scoring.store_rows].astype(np.float64) * CODES_PER_DOSAGE - SIGNED_CODE_OFFSET
    centre = scoring.signed_means if centering == "training" else signed.mean(axis=0)
    expected, rounding = _scores(scoring, (signed - centre).T + scoring.signed_means[:, None])
    np.testing.assert_allclose(predictor.predict(test), expected + scoring.alpha[0], rtol=0.0, atol=rounding + _EPSILON * abs(scoring.alpha[0]))
    masked = bench_real._without_structural_variants(train, test)
    sv_rows = train.variants.is_sv[scoring.store_rows]
    widened = test[:, scoring.store_rows].astype(np.float64) - masked[:, scoring.store_rows].astype(np.float64)
    change = CODES_PER_DOSAGE * widened / scoring.signed_scales
    if centering == "target":
        change -= change.mean(axis=0)
    sv_part = change[:, sv_rows] @ scoring.coefficients[sv_rows]
    bound = 4.0 * rounding + float(np.max(_EPSILON * (np.abs(change) @ np.abs(scoring.coefficients)) * _COLUMNS))
    np.testing.assert_allclose(predictor.predict(test) - predictor.predict(masked), sv_part, rtol=0.0, atol=bound)


def test_bench_real_accepts_fractional_dosages(small_n: _SmallNStub) -> None:
    train, _test = _bench_real_train(np.random.default_rng(6))
    train.genotypes[0, 0] = 0.5
    predictor = svpgs_method.fit_expression(train)
    assert small_n.calls[0]["codes"][0, 0] == round(0.5 * CODES_PER_DOSAGE)
    assert np.all(np.isfinite(predictor.predict(train.genotypes)))


def test_each_bench_real_fit_gets_one_cores_share(monkeypatch: pytest.MonkeyPatch) -> None:
    machine = ComputeBudget(
        device_kind="cuda", device_ids=(0,), device_names=("gpu",), device_bytes=(1 << 34,), device_compute_capabilities=((8, 6),),
        host_bytes=(1 << 36) + 5, cpu_threads=16,
    )
    monkeypatch.setattr(svpgs_method, "detect_compute_budget", lambda: machine)
    monkeypatch.delenv("RUNQ_MEM_BYTES", raising=False)
    budget = svpgs_method.one_core_budget()
    assert (budget.device_kind, budget.cpu_threads, budget.host_bytes) == ("cpu", 1, machine.host_bytes // machine.cpu_threads)
    # Under the runner's allotment, a worker's share is reduced by what it already holds.
    monkeypatch.setattr(svpgs_method, "_resident_bytes", lambda: 1 << 20)
    monkeypatch.setenv("RUNQ_MEM_BYTES", str(1 << 34))
    assert svpgs_method.one_core_budget().host_bytes == (1 << 34) // 16 - (1 << 20)


def test_both_harnesses_load_the_method_file_their_own_way() -> None:
    path = Path(svpgs_method.__file__)
    assert callable(bench_real.load_method(f"{path}:fit_expression"))
    assert callable(bench_real.load_method(f"{path}:fit_expression_no_sv_terms"))
    assert callable(bench_real.load_method(f"{path}:fit_expression_no_annotations"))
    assert callable(bench_real.load_method(f"{path}:fit_expression_target_centered"))
    assert callable(bench_sim.load_method(path).fit)


@pytest.mark.parametrize("arm", ["no_sv_terms", "no_annotations"])
def test_the_ablation_arms_withhold_their_prior_terms_and_nothing_else(small_n: _SmallNStub, arm: str) -> None:
    train, _test = _bench_real_train(np.random.default_rng(7))
    getattr(svpgs_method, f"fit_expression_{arm}")(train)
    (call,) = small_n.calls
    classes = [_CLASSES[code] for code in call["variant_class"]]
    if arm == "no_annotations":
        assert set(classes) == {VariantClass.SNV}
    else:
        # The small-variant rule on allele lengths, symbolic SVs by their signed change: no SV type is read.
        assert classes == [
            VariantClass.SNV, VariantClass.SNV, VariantClass.DELETION, VariantClass.SNV, VariantClass.DELETION, VariantClass.INSERTION,
            VariantClass.SNV, VariantClass.INSERTION, VariantClass.SNV, VariantClass.SNV, VariantClass.INSERTION, VariantClass.DELETION,
        ]
    np.testing.assert_array_equal(call["codes"], (train.genotypes * CODES_PER_DOSAGE).astype(np.uint8))


def test_the_coefficients_are_the_genotype_scale_effects_of_the_prediction(small_n: _SmallNStub) -> None:
    train, test = _bench_real_train(np.random.default_rng(9))
    predictor = svpgs_method.fit_expression(train)
    scoring = predictor.scoring

    def rounding(genotypes: np.ndarray) -> float:
        signed = genotypes[:, scoring.store_rows].astype(np.float64).T * CODES_PER_DOSAGE - SIGNED_CODE_OFFSET
        return _scores(scoring, signed)[1] + _EPSILON * abs(float(scoring.alpha[0]))

    base = predictor.predict(test)
    for column in range(_COLUMNS):
        moved = test.astype(np.float64)
        moved[:, column] += 1.0
        change = predictor.predict(moved) - base
        np.testing.assert_allclose(change, predictor.coefficients[column], rtol=0.0, atol=rounding(test) + rounding(moved))


def test_bench_reals_covariates_are_the_fits_fixed_effects(small_n: _SmallNStub) -> None:
    """review-mathbugs C2: the phenotype was residualized on [1, C], so the fit projects on the same [1, C]."""
    train, test = _bench_real_train(np.random.default_rng(15))
    covariates = np.random.default_rng(16).normal(size=(_REAL_SAMPLES + 20, 2))
    with_covariates = dataclasses.replace(train, covariates=covariates[:_REAL_SAMPLES])
    train = dataclasses.replace(train, covariates=None)
    predictor = svpgs_method.fit_expression(with_covariates)
    np.testing.assert_array_equal(small_n.calls[0]["covariates"], np.column_stack([np.ones(_REAL_SAMPLES), covariates[:_REAL_SAMPLES]]))
    alpha = predictor.scoring.alpha
    np.testing.assert_allclose(
        predictor.predict(test, covariates=covariates[_REAL_SAMPLES:]) - predictor.predict(test), covariates[_REAL_SAMPLES:] @ alpha[1:],
        rtol=0.0, atol=64 * _EPSILON * (1.0 + np.abs(covariates[_REAL_SAMPLES:]) @ np.abs(alpha[1:])).max(),
    )
    svpgs_method.fit_expression(train)
    np.testing.assert_array_equal(small_n.calls[1]["covariates"], np.ones((_REAL_SAMPLES, 1)))
