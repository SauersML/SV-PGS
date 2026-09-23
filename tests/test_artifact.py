"""The fitted-model artifact: an exact round trip, strict loading, atomic writes, and scoring from the store."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sv_pgs.artifact import (
    CERTIFICATE_STATUS,
    MODEL_FORMAT,
    FittedModel,
    Provenance,
    cohort_digest,
    code_digest,
    load_model,
    named_digest,
    offset_digest,
    predict,
    save_model,
    sites_digest,
    store_digest,
    write_predictions,
)
from sv_pgs.cli import main
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import TraitType
from sv_pgs.dosage_store import DosageStore
from sv_pgs.fast_scoring import ScoringModel, posterior_predictive_probability
from sv_pgs.fit_model import certificate_parts
from sv_pgs.full_data_fit import FitCertificate
from sv_pgs.scale_mixture_ep import MixtureHyperparameters
from tests.test_dosage_store import _write_store
from tests.test_fit_model import _certificate

_SAMPLES = 40
_VARIANTS = 30
_DRAWS = 5
_COVARIATES = 2


def _budget() -> ComputeBudget:
    return ComputeBudget(
        device_kind="cpu", device_ids=(), device_names=(), device_bytes=(), device_compute_capabilities=(), host_bytes=1 << 28, cpu_threads=1
    )


def _scoring(generator: np.random.Generator, trait_type: TraitType, rows: np.ndarray) -> ScoringModel:
    alpha = generator.normal(size=_COVARIATES + 1)
    return ScoringModel(
        store_rows=rows,
        signed_means=generator.normal(size=rows.shape[0]),
        signed_scales=generator.uniform(10.0, 60.0, size=rows.shape[0]),
        coefficients=generator.normal(size=rows.shape[0]),
        posterior_draws=generator.normal(size=(rows.shape[0], _DRAWS)),
        alpha=alpha,
        covariate_draws=np.repeat(alpha[:, None], _DRAWS, axis=1),
        covariate_covariance=np.zeros((alpha.size, alpha.size)),
        gaussian_posterior=True,
        trait_type=trait_type,
        predictive_intercept_shift=float(generator.normal()) if trait_type == TraitType.BINARY else 0.0,
    )


def _model(generator: np.random.Generator, store_root: Path) -> FittedModel:
    rows_quantitative = np.sort(generator.choice(_VARIANTS, size=20, replace=False)).astype(np.int64)
    rows_binary = np.sort(generator.choice(_VARIANTS, size=12, replace=False)).astype(np.int64)
    binary = _scoring(generator, TraitType.BINARY, rows_binary)
    return FittedModel(
        model_names=("ldl/fold0", "t2d/fold0"),
        covariate_names=("age", "t2d:sex=female"),
        covariate_columns=np.array([[True, False], [True, True]]),
        scoring=(
            replace(_scoring(generator, TraitType.QUANTITATIVE, rows_quantitative), alpha=np.array([0.3, -1.2, 0.0]),
                    covariate_draws=np.repeat(np.array([0.3, -1.2, 0.0])[:, None], _DRAWS, axis=1)),
            binary,
        ),
        noise_variance=np.array([0.7, 1.0]),
        hyperparameters=tuple(
            MixtureHyperparameters(coefficients=generator.normal(size=9), log_smoothing=generator.normal(size=3)) for _model in range(2)
        ),
        certificate={
            "remaining_gain": np.array([0.01, 0.02]),
            "negative_sites": np.array([0, 3]),
            "outer_iterations": np.array([4, 6]),
            "outer_criterion_met": np.array([True, False]),
        },
        fit_counts={"refreshes": 3, "passes": 41},
        refusals=("model 1: no damped EP pass keeps the full-data precision positive definite",),
        provenance=Provenance(
            code_digest=code_digest(),
            store_digest=store_digest(store_root),
            sites_digest=sites_digest(store_root),
            cohort_digest=cohort_digest(["b", "a"]),
            offset_digest=offset_digest(np.log(generator.uniform(size=_VARIANTS))),
            problem_digest=named_digest({"cohort": np.array(["a", "b"]), "seed": "5"}),
            prior_digest=named_digest({"nodes": np.linspace(-9.0, 1.0, 11)}),
        ),
    )


@pytest.fixture
def store_root(tmp_path: Path) -> Path:
    generator = np.random.default_rng(5)
    root = tmp_path / "store"
    _write_store(root, [{"chr22": generator.integers(0, 2001, size=(_VARIANTS, _SAMPLES))}])
    return root


def test_a_saved_model_loads_back_exactly(tmp_path: Path, store_root: Path) -> None:
    model = _model(np.random.default_rng(1), store_root)
    save_model(tmp_path / "model", model)
    loaded = load_model(tmp_path / "model")
    assert loaded.model_names == model.model_names
    assert loaded.covariate_names == model.covariate_names
    np.testing.assert_array_equal(loaded.covariate_columns, model.covariate_columns)
    assert loaded.trait_types == model.trait_types
    assert loaded.provenance == model.provenance
    assert loaded.fit_counts == model.fit_counts
    assert loaded.refusals == model.refusals
    np.testing.assert_array_equal(loaded.noise_variance, model.noise_variance)
    for original, restored in zip(model.scoring, loaded.scoring, strict=True):
        for field_name in ("store_rows", "signed_means", "signed_scales", "coefficients", "posterior_draws", "alpha", "covariate_draws", "covariate_covariance"):
            np.testing.assert_array_equal(np.asarray(getattr(restored, field_name)), np.asarray(getattr(original, field_name)))
        assert restored.predictive_intercept_shift == original.predictive_intercept_shift
        assert restored.gaussian_posterior == original.gaussian_posterior
    for original, restored in zip(model.hyperparameters, loaded.hyperparameters, strict=True):
        np.testing.assert_array_equal(restored.coefficients, original.coefficients)
        np.testing.assert_array_equal(restored.log_smoothing, original.log_smoothing)
    assert sorted(loaded.certificate) == sorted(model.certificate)
    for name, values in model.certificate.items():
        np.testing.assert_array_equal(loaded.certificate[name], values)


def test_a_product_mixture_law_is_saved_as_its_parameters_and_draws_the_same_numbers(tmp_path: Path, store_root: Path) -> None:
    """The mean-field route's draws are a law (``draw_laws.ProductMixtureDraws``): the artifact holds its O(components x
    rows) parameters, never the rows x K draws, and the loaded law draws exactly what the saved one draws."""
    from sv_pgs.draw_laws import ProductMixtureDraws

    model = _model(np.random.default_rng(3), store_root)
    first = model.scoring[0]
    rows, draws = first.store_rows.shape[0], first.draw_count
    generator = np.random.default_rng(5)
    law = ProductMixtureDraws(
        class_index=np.zeros(rows, dtype=np.int64), log_scale=generator.normal(-3.0, 0.5, rows), omega=generator.uniform(50.0, 100.0, (2, rows)),
        log_density=np.log(np.full((1, 7), 1.0 / 7.0)), log_variance_grid=np.linspace(-6.0, 2.0, 7),
        shift=generator.normal(size=(2, rows)), weights=np.array([0.7, 0.3]), key=(11, 12), draw_count=draws,
    )
    model = replace(model, scoring=(replace(first, posterior_draws=law), *model.scoring[1:]))
    save_model(tmp_path / "model", model)
    with np.load(tmp_path / "model" / "arrays.npz") as archive:
        assert not any(archive[name].size >= rows * draws for name in archive.files if name.startswith("scoring/0/draw_law/"))
    loaded = load_model(tmp_path / "model").scoring[0].posterior_draws
    assert isinstance(loaded, ProductMixtureDraws)
    np.testing.assert_array_equal(loaded.tile(0, rows), law.tile(0, rows))


def test_a_model_needs_a_well_formed_certificate_with_its_fit_status(tmp_path: Path, store_root: Path) -> None:
    """I12: the leading dimension alone let an empty mapping, a mapping without the fit's status, and an object array
    all pass. The status term is the one ``full_data_fit.FitCertificate`` records for every route."""
    model = _model(np.random.default_rng(9), store_root)
    assert CERTIFICATE_STATUS in {field.name for field in dataclasses.fields(FitCertificate)}
    with pytest.raises(ValueError, match=CERTIFICATE_STATUS):
        replace(model, certificate={})
    with pytest.raises(ValueError, match=CERTIFICATE_STATUS):
        replace(model, certificate={name: values for name, values in model.certificate.items() if name != CERTIFICATE_STATUS})
    with pytest.raises(ValueError, match="numeric or boolean"):
        replace(model, certificate=model.certificate | {"remaining_gain": np.array(["a", "b"])})
    # A padded per-model term keeps its NaN pads: only the kind is checked.
    padded = replace(model, certificate=model.certificate | {"stationarity_steps": np.array([[0.5, np.nan], [0.25, 0.125]])})
    save_model(tmp_path / "model", padded)
    assert np.isnan(load_model(tmp_path / "model").certificate["stationarity_steps"][0, 1])


def test_an_artifact_is_never_written_from_a_certificate_term_the_fit_did_not_record() -> None:
    """I12: ``certified`` defaulted to None and reached ``int(None)``; a term a route leaves unrecorded is now named."""
    certificate = _certificate(2, np.random.default_rng(11))
    assert certificate_parts(certificate, 2).terms[CERTIFICATE_STATUS].shape == (2,)
    with pytest.raises(ValueError, match=CERTIFICATE_STATUS):
        certificate_parts(replace(certificate, **{CERTIFICATE_STATUS: None}), 2)


def test_a_model_is_never_overwritten(tmp_path: Path, store_root: Path) -> None:
    model = _model(np.random.default_rng(2), store_root)
    save_model(tmp_path / "model", model)
    with pytest.raises(FileExistsError):
        save_model(tmp_path / "model", model)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["model", "store"]


def test_loading_refuses_a_model_that_is_not_exactly_what_was_written(tmp_path: Path, store_root: Path) -> None:
    save_model(tmp_path / "model", _model(np.random.default_rng(3), store_root))
    metadata_path = tmp_path / "model" / "model.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["format"] = "svpgs-model v0"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="is not a"):
        load_model(tmp_path / "model")
    metadata["format"] = MODEL_FORMAT
    metadata["arrays"] = metadata["arrays"][1:]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="exactly the arrays"):
        load_model(tmp_path / "model")


def test_a_model_has_no_effect_of_a_covariate_it_did_not_adjust_for(store_root: Path) -> None:
    model = _model(np.random.default_rng(10), store_root)
    leaking = replace(model.scoring[0], alpha=np.array([0.3, -1.2, 0.4]))
    with pytest.raises(ValueError, match="did not adjust for"):
        replace(model, scoring=(leaking, model.scoring[1]))
    with pytest.raises(ValueError, match="covariate_columns"):
        replace(model, covariate_columns=np.ones((2, 3), dtype=bool))


def test_the_cohort_digest_ignores_order_and_refuses_repeats() -> None:
    assert cohort_digest(["a", "b", "c"]) == cohort_digest(["c", "a", "b"])
    with pytest.raises(ValueError, match="repeats"):
        cohort_digest(["a", "a"])


def test_prediction_equals_the_standardized_scores_read_from_the_store(tmp_path: Path, store_root: Path) -> None:
    generator = np.random.default_rng(4)
    model = _model(generator, store_root)
    store = DosageStore.open(store_root)
    signed = store.read_codes(0, store.n_variants).astype(np.float64) - 127.0
    samples = np.sort(generator.choice(_SAMPLES, size=15, replace=False)).astype(np.int64)
    covariates = generator.normal(size=(samples.shape[0], _COVARIATES))
    prediction = predict(model, store, samples, covariates, _budget())
    for index, scoring in enumerate(model.scoring):
        standardized = (signed[scoring.store_rows][:, samples] - scoring.signed_means[:, None]) / scoring.signed_scales[:, None]
        genetic = standardized.T @ scoring.coefficients
        draw_scores = standardized.T @ np.asarray(scoring.posterior_draws)
        variance = np.mean(np.square(draw_scores - genetic[:, None]), axis=1)
        linear = genetic + scoring.alpha[0] + covariates @ scoring.alpha[1:]
        rounding = np.finfo(np.float64).eps * (np.abs(standardized.T) @ np.abs(scoring.coefficients) + np.abs(linear)) * store.n_variants
        np.testing.assert_allclose(prediction.genetic.means[:, index], genetic, rtol=0.0, atol=float(np.max(rounding)))
        np.testing.assert_allclose(prediction.linear_predictor[:, index], linear, rtol=0.0, atol=float(np.max(rounding)))
        np.testing.assert_allclose(prediction.genetic.variances[:, index], variance, rtol=np.sqrt(np.finfo(np.float64).eps))
        if scoring.trait_type == TraitType.BINARY:
            probability = posterior_predictive_probability(
                prediction.linear_predictor[:, index], prediction.genetic.variances[:, index], scoring.predictive_intercept_shift
            )
            np.testing.assert_array_equal(prediction.predictive_mean[:, index], probability)
            np.testing.assert_array_equal(prediction.predictive_variance[:, index], probability * (1.0 - probability))
        else:
            np.testing.assert_array_equal(prediction.predictive_mean[:, index], prediction.linear_predictor[:, index])
            np.testing.assert_array_equal(
                prediction.predictive_variance[:, index], prediction.genetic.variances[:, index] + model.noise_variance[index]
            )


def test_prediction_refuses_covariates_of_the_wrong_shape(store_root: Path) -> None:
    model = _model(np.random.default_rng(6), store_root)
    with pytest.raises(ValueError, match="covariates"):
        predict(model, DosageStore.open(store_root), np.arange(4), np.zeros((4, _COVARIATES + 1)), _budget())


def test_prediction_refuses_a_store_with_another_variant_layout(tmp_path: Path, store_root: Path) -> None:
    model = _model(np.random.default_rng(7), store_root)
    other_root = tmp_path / "other"
    _write_store(other_root, [{"chr22": np.random.default_rng(8).integers(0, 2001, size=(_VARIANTS + 1, _SAMPLES))}])
    with pytest.raises(ValueError, match="variant layout"):
        predict(model, DosageStore.open(other_root), np.arange(4), np.zeros((4, _COVARIATES)), _budget())


def test_the_score_command_writes_the_predictions_of_the_people_file(tmp_path: Path, store_root: Path) -> None:
    generator = np.random.default_rng(9)
    model = _model(generator, store_root)
    save_model(tmp_path / "model", model)
    samples = np.arange(0, _SAMPLES, 3, dtype=np.int64)
    covariates = generator.normal(size=(samples.shape[0], _COVARIATES))
    np.savez(tmp_path / "people.npz", sample_indices=samples, covariates=covariates)
    assert main(["score", str(tmp_path / "model"), str(store_root), str(tmp_path / "people.npz"), str(tmp_path / "scores.npz")]) == 0
    expected = predict(model, DosageStore.open(store_root), samples, covariates, _budget())
    with np.load(tmp_path / "scores.npz") as scores:
        assert tuple(scores["model_names"]) == model.model_names
        np.testing.assert_allclose(scores["genetic_mean"], expected.genetic.means, rtol=0.0, atol=np.finfo(np.float64).eps * _VARIANTS)
        np.testing.assert_allclose(scores["predictive_mean"], expected.predictive_mean, rtol=0.0, atol=np.finfo(np.float64).eps * _VARIANTS)
    with pytest.raises(FileExistsError):
        write_predictions(tmp_path / "model", store_root, tmp_path / "people.npz", tmp_path / "scores.npz", _budget())


def test_prediction_preserves_genetic_covariate_cancellation(store_root: Path) -> None:
    model = _model(np.random.default_rng(54), store_root)
    with DosageStore.open(store_root) as store:
        samples = np.arange(store.n_samples)
        code = store.read_codes(0, 1)[0].astype(np.float64)
        mean, scale = (code - 127.).mean(), code.std()
        standardized = (code - 127. - mean) / scale
        beta_draws = np.array([[-10., -5., 5., 10.]])
        scoring = ScoringModel(
            store_rows=np.array([0], dtype=np.int64), signed_means=np.array([mean]), signed_scales=np.array([scale]),
            coefficients=np.zeros(1), posterior_draws=beta_draws, alpha=np.zeros(3), trait_type=TraitType.QUANTITATIVE,
            predictive_intercept_shift=0., covariate_draws=np.vstack([np.zeros(4), -beta_draws, np.zeros(4)]),
            covariate_covariance=np.diag([0.2, 0.3, 0.]), gaussian_posterior=True,
        )
        model = replace(model, model_names=(model.model_names[0],), scoring=(scoring,), covariate_columns=model.covariate_columns[:1],
                        noise_variance=model.noise_variance[:1], hyperparameters=model.hyperparameters[:1],
                        certificate={name: np.asarray(values)[:1] for name, values in model.certificate.items()})
        result = predict(model, store, samples, np.column_stack([standardized, np.zeros(samples.size)]), _budget())
        np.testing.assert_allclose(result.predictive_variance[:, 0], model.noise_variance[0] + 0.2 + 0.3 * standardized ** 2, atol=1e-12)
        assert result.genetic.variances.max() > 10.
