"""The public fit: cohort checks, what reaches the engine driver, the artifact it returns, and store -> fit -> score.

The driver is replaced by a stub that fits a real (if simple) model from the store, so these tests pin the wiring
around ``full_data_fit.fit_models``; the driver's own accuracy is tested with it.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sv_pgs import fit_model
from sv_pgs.artifact import cohort_digest, load_model, offset_digest, predict, save_model, sites_digest, store_digest
from sv_pgs.cli import main
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import TraitType
from sv_pgs.dosage_store import CODES_PER_DOSAGE, DosageStore
from sv_pgs.fast_scoring import ScoringModel
from sv_pgs.full_data_fit import FitCertificate
from sv_pgs.scale_mixture_ep import MixtureHyperparameters, OuterFit
from tests.test_dosage_store import _write_store

_SAMPLES = 60
_VARIANTS = 25
_COHORT = 48
_DRAWS = 4


def _budget() -> ComputeBudget:
    return ComputeBudget(
        device_kind="cpu", device_ids=(), device_names=(), device_bytes=(), device_compute_capabilities=(), host_bytes=1 << 28, cpu_threads=1
    )


def _request(store_root: Path, arguments: dict[str, Any], work_dir: Path, seed: int) -> fit_model.FitRequest:
    work_dir.mkdir(parents=True, exist_ok=True)
    return fit_model.FitRequest(store=DosageStore.open(store_root), **arguments, budget=_budget(), work_dir=work_dir, seed=seed)


def _certificate(model_count: int, generator: np.random.Generator) -> FitCertificate:
    """A certificate with every field filled by its annotated kind, so a new driver field needs no test change."""
    values: dict[str, Any] = {}
    for field in dataclasses.fields(FitCertificate):
        kind = str(field.type)
        if kind == "tuple[str, ...]":
            values[field.name] = ("model 1: no damped EP pass keeps the full-data precision positive definite",)
        elif kind.startswith("tuple["):
            values[field.name] = tuple(generator.normal(size=model + 1) for model in range(model_count))
        elif kind == "int":
            values[field.name] = int(generator.integers(1, 50))
        elif kind.startswith("I64Array"):
            values[field.name] = generator.integers(0, 9, size=model_count)
        elif kind.startswith("BoolArray"):
            values[field.name] = generator.uniform(size=model_count) < 0.5
        else:
            values[field.name] = generator.uniform(size=model_count)
    return FitCertificate(**values)


@dataclasses.dataclass(frozen=True)
class _StubFit:
    scoring: list[ScoringModel]
    noise_variance: np.ndarray
    hyperparameters: tuple[MixtureHyperparameters, ...]
    certificate: FitCertificate
    prior_digests: tuple[str, ...]


class _StubDriver:
    """Marginal regression of each model's covariate-adjusted training targets on the standardized store codes."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.results: list[_StubFit] = []
        self.store_contents: list[tuple[np.ndarray, Any]] = []

    def __call__(self, **arguments: Any) -> _StubFit:
        self.calls.append(arguments)
        assert arguments["work_dir"].is_dir()
        store: DosageStore = arguments["store"]
        # What the store held during the call: a caller may close and remove it once the fit returns.
        self.store_contents.append((store.read_codes(0, store.n_variants).copy(), store.variant_table))
        signed = store.read_codes(0, store.n_variants).astype(np.float64) - CODES_PER_DOSAGE
        generator = np.random.default_rng(arguments["seed"])
        scoring, noise = [], []
        for model, trait_type in enumerate(arguments["trait_types"]):
            rows = arguments["training"][:, model]
            codes = signed[:, arguments["store_columns"][rows]]
            means, scales = codes.mean(axis=1), codes.std(axis=1)
            standardized = (codes - means[:, None]) / scales[:, None]
            adjusted = arguments["covariate_columns"][model]
            design = arguments["covariates"][rows][:, adjusted]
            fitted, *_ = np.linalg.lstsq(design, arguments["targets"][rows, model], rcond=None)
            residual = arguments["targets"][rows, model] - design @ fitted
            alpha = np.zeros(adjusted.shape[0])
            alpha[adjusted] = fitted
            coefficients = standardized @ residual / (rows.sum() * store.n_variants)
            scoring.append(
                ScoringModel(
                    store_rows=np.arange(store.n_variants, dtype=np.int64),
                    signed_means=means,
                    signed_scales=scales,
                    coefficients=coefficients,
                    posterior_draws=coefficients[:, None] + generator.normal(scale=np.abs(coefficients).mean(), size=(store.n_variants, _DRAWS)),
                    alpha=alpha,
                    covariate_draws=np.repeat(alpha[:, None], _DRAWS, axis=1),
                    covariate_covariance=np.zeros((alpha.size, alpha.size)),
                    gaussian_posterior=True,
                    trait_type=trait_type,
                    predictive_intercept_shift=float(generator.normal()) if trait_type == TraitType.BINARY else 0.0,
                )
            )
            noise.append(float(residual @ residual) / rows.sum())
        result = _StubFit(
            scoring=scoring,
            noise_variance=np.array(noise),
            hyperparameters=tuple(
                MixtureHyperparameters(coefficients=generator.normal(size=7), log_smoothing=generator.normal(size=2)) for _ in scoring
            ),
            certificate=_certificate(len(scoring), generator),
            prior_digests=tuple(f"prior-schema-of-model-{index}" for index in range(len(scoring))),
        )
        self.results.append(result)
        return result


@dataclasses.dataclass(frozen=True)
class _Cohort:
    research_ids: list[str]
    store_columns: np.ndarray
    covariates: np.ndarray
    covariate_names: tuple[str, ...]
    covariate_columns: np.ndarray
    targets: np.ndarray
    training: np.ndarray
    model_names: tuple[str, ...]
    trait_types: tuple[TraitType, ...]

    def arguments(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in dataclasses.fields(self)} | {"log_variance_offset": None}


def _cohort(generator: np.random.Generator) -> _Cohort:
    store_columns = generator.permutation(_SAMPLES)[:_COHORT].astype(np.int64)
    quantitative = generator.normal(size=_COHORT)
    binary = (generator.uniform(size=_COHORT) < 0.3).astype(np.float64)
    fold = np.arange(_COHORT) % 2
    training = np.column_stack([fold == 0, fold == 1, fold == 0])
    targets = np.column_stack([quantitative, quantitative, binary])
    targets[~training] = np.nan
    return _Cohort(
        research_ids=[f"person{column}" for column in store_columns],
        store_columns=store_columns,
        covariates=generator.normal(size=(_COHORT, 2)),
        covariate_names=("age", "t2d:sex=female"),
        covariate_columns=np.array([[True, False], [True, False], [True, True]]),
        targets=targets,
        training=training,
        model_names=("ldl/fold0", "ldl/fold1", "t2d/fold0"),
        trait_types=(TraitType.QUANTITATIVE, TraitType.QUANTITATIVE, TraitType.BINARY),
    )


@pytest.fixture
def store_root(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    _write_store(root, [{"chr22": np.random.default_rng(3).integers(0, 2001, size=(_VARIANTS, _SAMPLES))}])
    return root


@pytest.fixture
def driver(monkeypatch: pytest.MonkeyPatch) -> _StubDriver:
    stub = _StubDriver()
    monkeypatch.setattr(fit_model, "fit_models", stub)
    return stub


def test_the_driver_gets_the_intercept_the_training_targets_and_the_draw_count(tmp_path: Path, store_root: Path, driver: _StubDriver) -> None:
    cohort = _cohort(np.random.default_rng(1))
    fit_model.fit(_request(store_root, cohort.arguments(), tmp_path, 11))
    (call,) = driver.calls
    np.testing.assert_array_equal(call["covariates"], np.column_stack([np.ones(_COHORT), cohort.covariates]))
    np.testing.assert_array_equal(call["covariate_columns"], np.column_stack([np.ones(3, dtype=bool), cohort.covariate_columns]))
    np.testing.assert_array_equal(call["training"], cohort.training)
    np.testing.assert_array_equal(call["targets"][cohort.training], cohort.targets[cohort.training])
    assert np.all(call["targets"][~cohort.training] == 0.0)
    np.testing.assert_array_equal(call["store_columns"], cohort.store_columns)
    assert call["trait_types"] == cohort.trait_types
    assert (call["draw_count"], call["seed"], call["work_dir"]) == (fit_model.DRAW_COUNT, 11, tmp_path)
    assert call["log_variance_offset"] is None


def test_the_driver_gets_the_records_log_reliabilities(tmp_path: Path, store_root: Path, driver: _StubDriver) -> None:
    offset = np.log(np.random.default_rng(12).uniform(size=_VARIANTS))
    offset[4] = -np.inf
    arguments = _cohort(np.random.default_rng(1)).arguments() | {"log_variance_offset": offset}
    model = fit_model.fit(_request(store_root, arguments, tmp_path, 11))
    np.testing.assert_array_equal(driver.calls[0]["log_variance_offset"], offset)
    assert model.provenance.offset_digest == offset_digest(offset) != offset_digest(None)


def test_the_artifact_carries_the_whole_certificate_and_the_provenance(tmp_path: Path, store_root: Path, driver: _StubDriver) -> None:
    cohort = _cohort(np.random.default_rng(2))
    model = fit_model.fit(_request(store_root, cohort.arguments(), tmp_path, 5))
    fitted = driver.results[0].certificate
    assert model.model_names == cohort.model_names and model.trait_types == cohort.trait_types
    np.testing.assert_array_equal(model.covariate_columns, cohort.covariate_columns)
    assert all(scoring.alpha[2] == 0.0 for scoring in model.scoring[:2]) and model.scoring[2].alpha[2] != 0.0
    for field in dataclasses.fields(FitCertificate):
        value = getattr(fitted, field.name)
        if str(field.type) == "tuple[str, ...]":
            assert model.refusals == value
        elif str(field.type).startswith("tuple["):
            counts = model.certificate[f"{field.name}_count"]
            np.testing.assert_array_equal(counts, [entry.shape[0] for entry in value])
            for index, entry in enumerate(value):
                np.testing.assert_array_equal(model.certificate[field.name][index, : counts[index]], entry)
                assert np.all(np.isnan(model.certificate[field.name][index, counts[index] :]))
        elif str(field.type) == "int":
            assert model.fit_counts[field.name] == value
        else:
            np.testing.assert_array_equal(model.certificate[field.name], value)
    assert model.provenance.store_digest == store_digest(store_root)
    assert model.provenance.sites_digest == sites_digest(store_root)
    assert model.provenance.cohort_digest == cohort_digest(cohort.research_ids)
    assert model.provenance.offset_digest == offset_digest(None)


def test_the_artifact_reports_the_outer_criterion_and_never_certification(tmp_path: Path, store_root: Path, driver: _StubDriver) -> None:
    """M04: ``OuterFit.fixed_point_term_measured`` is False for every fit this package produces, and its docstring
    says a caller must then treat the fit as uncertified. So the status the artifact carries is named for what the
    outer loop did establish, and no public surface carries a ``certified`` flag to be misread."""
    assert OuterFit.fixed_point_term_measured is False
    names = {field.name for field in dataclasses.fields(FitCertificate)}
    assert "certified" not in names and "outer_criterion_met" in names
    cohort = _cohort(np.random.default_rng(7))
    model = fit_model.fit(_request(store_root, cohort.arguments(), tmp_path, 7))
    assert "certified" not in model.certificate and "certified" not in model.fit_counts
    np.testing.assert_array_equal(model.certificate["outer_criterion_met"], driver.results[0].certificate.outer_criterion_met)
    save_model(tmp_path / "model", model)
    assert "certified" not in json.loads((tmp_path / "model" / "model.json").read_text())["certificate_terms"]


def test_the_provenance_identifies_the_whole_fitting_problem(tmp_path: Path, store_root: Path, driver: _StubDriver) -> None:
    """I11: the cohort digest sorts, so it is blind to which store column each person was read from and to every
    model's own training set. The problem digest is not, and the prior digest identifies the schema the saved
    hyperparameters are written on."""
    cohort = _cohort(np.random.default_rng(8))
    base = fit_model.fit(_request(store_root, cohort.arguments(), tmp_path / "a", 5)).provenance
    order = np.argsort(cohort.research_ids)

    def reordered(**change: Any) -> Any:
        rows = {name: getattr(cohort, name) for name in ("research_ids", "store_columns", "covariates", "targets", "training")}
        moved = {name: [values[index] for index in order] if name == "research_ids" else values[order] for name, values in rows.items()}
        return dataclasses.replace(cohort, **(moved | change))

    # The same people, the same store columns, the same models: one cohort, a different fitting problem.
    permuted = fit_model.fit(_request(store_root, reordered().arguments(), tmp_path / "b", 5)).provenance
    assert permuted.cohort_digest == base.cohort_digest and permuted.problem_digest != base.problem_digest
    # The same people in the same order, re-bound to other store columns.
    rebound = reordered(store_columns=cohort.store_columns[order][::-1])
    other = fit_model.fit(_request(store_root, rebound.arguments(), tmp_path / "c", 5)).provenance
    assert other.cohort_digest == base.cohort_digest and other.problem_digest != permuted.problem_digest
    # One model's training set moved, with every other input the same.
    moved = cohort.training.copy()
    moved[np.flatnonzero(moved[:, 1])[0], 1] = False
    held = fit_model.fit(_request(store_root, cohort.arguments() | {"training": moved}, tmp_path / "d", 5)).provenance
    assert held.problem_digest != base.problem_digest
    # The prior schema is the driver's, and the same problem twice gives the same provenance.
    assert base.prior_digest == fit_model.fit(_request(store_root, cohort.arguments(), tmp_path / "e", 5)).provenance.prior_digest
    assert base.problem_digest == fit_model.fit(_request(store_root, cohort.arguments(), tmp_path / "f", 5)).provenance.problem_digest


def test_the_cohort_digest_covers_only_the_training_rows(tmp_path: Path, store_root: Path, driver: _StubDriver) -> None:
    cohort = _cohort(np.random.default_rng(3))
    training = cohort.training.copy()
    training[0] = False
    arguments = cohort.arguments() | {"training": training}
    model = fit_model.fit(_request(store_root, arguments, tmp_path, 5))
    assert model.provenance.cohort_digest == cohort_digest(cohort.research_ids[1:])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda cohort: {"store_columns": np.concatenate([cohort.store_columns[:1], cohort.store_columns[:-1]])}, "distinct store samples"),
        (lambda cohort: {"store_columns": np.where(np.arange(_COHORT) == 0, _SAMPLES, cohort.store_columns)}, "distinct store samples"),
        (lambda cohort: {"research_ids": cohort.research_ids[:-1]}, "one distinct id"),
        (lambda cohort: {"research_ids": [cohort.research_ids[1], *cohort.research_ids[1:]]}, "one distinct id"),
        (lambda cohort: {"covariates": np.column_stack([np.ones(_COHORT), cohort.covariates])}, "covariates"),
        (lambda cohort: {"covariate_columns": cohort.covariate_columns[:2]}, "covariate_columns"),
        (lambda cohort: {"covariate_columns": cohort.covariate_columns.astype(np.int64)}, "covariate_columns"),
        (lambda cohort: {"targets": np.where(cohort.training, cohort.targets, np.nan)[:, :2]}, "targets and training"),
        (lambda cohort: {"targets": np.where(np.arange(_COHORT)[:, None] == 0, np.nan, cohort.targets)}, "finite"),
        (lambda cohort: {"targets": np.where(cohort.training & (np.arange(3) == 2), 0.5, cohort.targets)}, "other than 0 and 1"),
        (lambda cohort: {"trait_types": cohort.trait_types[:2]}, "one entry per model"),
        (lambda cohort: {"log_variance_offset": np.zeros(_VARIANTS - 1)}, "log reliability"),
        (lambda cohort: {"log_variance_offset": np.full(_VARIANTS, 0.1)}, "log reliability"),
        (lambda cohort: {"log_variance_offset": np.full(_VARIANTS, np.nan)}, "log reliability"),
        # Validated before conversion: a coercion that would silently change what the input means is refused.
        (lambda cohort: {"training": cohort.training.astype(np.int64)}, "not a training flag"),
        (lambda cohort: {"training": cohort.training.astype(np.float64)}, "not a training flag"),
        (lambda cohort: {"store_columns": cohort.store_columns.astype(np.float64) + 0.5}, "1-D integer array"),
        (lambda cohort: {"store_columns": cohort.store_columns[0]}, "1-D integer array"),
        (lambda cohort: {"store_columns": cohort.store_columns.astype(str)}, "1-D integer array"),
        (lambda cohort: {"covariates": cohort.covariates.astype(str)}, "covariates"),
        (lambda cohort: {"covariates": cohort.covariates[:, 0]}, "covariates"),
        (lambda cohort: {"targets": cohort.targets.astype(str)}, "targets and training"),
        (lambda cohort: {"log_variance_offset": np.zeros(_VARIANTS).astype(str)}, "log reliability"),
        # Refused before Stage 0 reads the store, not after the fit.
        (lambda cohort: {"model_names": (cohort.model_names[0], *cohort.model_names[1:-1], cohort.model_names[0])}, "distinct model names"),
        (lambda cohort: {"training": np.column_stack([np.zeros(_COHORT, bool), cohort.training[:, 1:]])}, "no training rows"),
    ],
)
def test_fit_refuses_a_cohort_that_does_not_line_up(tmp_path: Path, store_root: Path, driver: _StubDriver, change: Any, message: str) -> None:
    cohort = _cohort(np.random.default_rng(4))
    with pytest.raises(ValueError, match=message):
        _request(store_root, cohort.arguments() | change(cohort), tmp_path, 5)
    assert driver.calls == []


def test_a_refused_offset_is_refused_by_the_one_reliability_contract(tmp_path: Path, store_root: Path, driver: _StubDriver) -> None:
    """The request's offsets meet ``imputation_reliability.checked_log_reliability``, the contract every source of a
    record's reliability meets, so the refusal names the first offending record and how many there are rather than
    repeating the rule in a second place that could drift from it."""
    cohort = _cohort(np.random.default_rng(4))
    offset = np.zeros(_VARIANTS)
    offset[2] = 0.25
    with pytest.raises(ValueError, match=r"log_variance_offset: .*record 2 has 0\.25 \(1 of "):
        _request(store_root, cohort.arguments() | {"log_variance_offset": offset}, tmp_path, 5)
    assert driver.calls == []


def test_fit_refuses_a_seed_that_is_not_an_integer(tmp_path: Path, store_root: Path, driver: _StubDriver) -> None:
    """A float seed would be truncated by the conversion, so a different run would be reported as the same one."""
    cohort = _cohort(np.random.default_rng(4))
    with pytest.raises(ValueError, match="seed must be an integer"):
        _request(store_root, cohort.arguments(), tmp_path, 5.5)  # type: ignore[arg-type]
    assert driver.calls == []


def test_the_seed_is_fixed_by_the_store_and_the_cohort(tmp_path: Path, store_root: Path) -> None:
    ids = [f"person{index}" for index in range(5)]
    assert fit_model.cohort_seed(store_root, ids) == fit_model.cohort_seed(store_root, ids[::-1])
    assert fit_model.cohort_seed(store_root, ids) != fit_model.cohort_seed(store_root, ids[1:])


def _save_cohort(path: Path, cohort: _Cohort) -> None:
    np.savez(
        path,
        research_ids=np.array(cohort.research_ids),
        store_columns=cohort.store_columns,
        covariates=cohort.covariates,
        covariate_names=np.array(cohort.covariate_names),
        covariate_columns=cohort.covariate_columns,
        targets=cohort.targets,
        training=cohort.training,
        model_names=np.array(cohort.model_names),
        trait_types=np.array([trait_type.value for trait_type in cohort.trait_types]),
    )


def test_a_store_goes_through_fit_and_score_to_the_held_out_predictions(tmp_path: Path, store_root: Path, driver: _StubDriver) -> None:
    cohort = _cohort(np.random.default_rng(5))
    _save_cohort(tmp_path / "cohort.npz", cohort)
    assert main(["fit", str(store_root), str(tmp_path / "cohort.npz"), str(tmp_path / "model")]) == 0
    assert sorted(path.name for path in tmp_path.iterdir()) == ["cohort.npz", "model", "store"]
    assert driver.calls[0]["seed"] == fit_model.cohort_seed(store_root, cohort.research_ids)
    held_out = ~cohort.training[:, 0]
    np.savez(tmp_path / "people.npz", sample_indices=cohort.store_columns[held_out], covariates=cohort.covariates[held_out])
    assert main(["score", str(tmp_path / "model"), str(store_root), str(tmp_path / "people.npz"), str(tmp_path / "scores.npz")]) == 0
    model = load_model(tmp_path / "model")
    signed = DosageStore.open(store_root).read_codes(0, _VARIANTS).astype(np.float64) - CODES_PER_DOSAGE
    with np.load(tmp_path / "scores.npz") as scores:
        assert tuple(scores["model_names"]) == cohort.model_names
        for index, scoring in enumerate(model.scoring):
            standardized = (signed[:, cohort.store_columns[held_out]] - scoring.signed_means[:, None]) / scoring.signed_scales[:, None]
            linear = standardized.T @ scoring.coefficients + scoring.alpha[0] + cohort.covariates[held_out] @ scoring.alpha[1:]
            rounding = np.finfo(np.float64).eps * (np.abs(standardized.T) @ np.abs(scoring.coefficients) + np.abs(linear)) * _VARIANTS
            np.testing.assert_allclose(scores["linear_predictor"][:, index], linear, rtol=0.0, atol=float(np.max(rounding)))
    expected = predict(model, DosageStore.open(store_root), cohort.store_columns[held_out], cohort.covariates[held_out], _budget())
    with np.load(tmp_path / "scores.npz") as scores:
        # The command and this call reduce the same sums in whatever order their thread counts give them (the command
        # took 128 threads on a compute node where this call takes the budget's 1), so they agree to the reduction's
        # rounding, not bit for bit: eps per variant of the size the sum reaches.
        rounding = np.finfo(np.float64).eps * _VARIANTS * np.maximum(np.abs(expected.predictive_mean), 1.0)
        np.testing.assert_allclose(scores["predictive_mean"], expected.predictive_mean, rtol=0.0, atol=float(np.max(rounding)))
    with pytest.raises(FileExistsError):
        main(["fit", str(store_root), str(tmp_path / "cohort.npz"), str(tmp_path / "model")])
    assert len(driver.calls) == 1


def test_the_fit_command_refuses_a_cohort_file_with_other_arrays(tmp_path: Path, store_root: Path, driver: _StubDriver) -> None:
    cohort = _cohort(np.random.default_rng(6))
    _save_cohort(tmp_path / "cohort.npz", cohort)
    with np.load(tmp_path / "cohort.npz") as arrays:
        np.savez(tmp_path / "extra.npz", **{name: arrays[name] for name in arrays.files}, weights=np.ones(_COHORT))
    with pytest.raises(ValueError, match="must hold exactly"):
        main(["fit", str(store_root), str(tmp_path / "extra.npz"), str(tmp_path / "model")])
    assert not (tmp_path / "model").exists() and driver.calls == []
