"""The in-workspace pipeline's synthetic dry run: every step on a synthetic store and synthetic OMOP.

The inputs mimic the imputation's deliverables (docs/design/WORKSPACE_PIPELINE.md, "Inputs"):
popped GT:DS:GP batches in two imputation halves, long-read hard calls on the same sites, the
strata sidecar v2 with its _done manifest and md5s, bubble.split path records, a repeat BED, the
CDR crosswalk, ancestry and relatedness tables, and OMOP rows through a fake BigQuery client. The
measurement model, fit, model files and prediction are bound to doubles with the production
signatures; they are other lanes' modules.
"""
from __future__ import annotations

import dataclasses
import datetime
import gzip
import hashlib
import inspect
import json
import pickle
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sv_pgs.all_of_us import MEASUREMENT_EXCLUSION_REASONS, build_all_of_us_measurement_sql
from sv_pgs.cohort import pipeline_half_levels
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import TraitType
from sv_pgs.dosage_store import VARIANT_CLASS_LEGEND, DosageStore, HalfSamples, encode_dosage_milli
from sv_pgs.fast_scoring import (
    ScoringModel,
    ScoringPlan,
    posterior_predictive_probability,
    score_genetic,
    score_linear_predictor,
)
from sv_pgs.fit_model import FitRequest
from sv_pgs.phenotype_measurement import fit_at_exponent
from sv_pgs.store_converter import linear_recalibration, refalt_digest, value_matched_background
from sv_pgs import workspace_pipeline
from sv_pgs.workspace_pipeline import (
    STEP_NAMES,
    StrataSites,
    WorkspaceBindings,
    WorkspaceConfig,
    check_exportable,
    path_counts,
    run_pipeline,
)

CHROMOSOMES = ("chr21", "chr22")
# (POS, REF, ALT, n_paths, n_paths_total, SVLEN) per popped record, the same layout on each chromosome.
RECORDS = (
    (1000, "A", "G", 1, 1, None),
    (2000, "ACGT", "A", 2, 3, None),
    (2000, "A", "AT", 2, 3, None),
    (3000, "GA", "G", 3, 12, None),
    (3000, "G", "GTT", 2, 12, None),
    (3000, "GAA", "G", 8, 12, None),
    (4000, "CA", "C", 1, 1, None),
    (5000, "C", "T", 1, 1, None),
    (6000, "T", "T" + "ACGT" * 13, 1, 1, 52),
)
# bubble.split: each bubble's paths as the records (indices into RECORDS) they carry; one path is
# written with ',' and one with ':' between its IDs.
BUBBLES = ((2000, ((1,), (1, 2), (2,))), (3000, ((3,), (4,)) + ((5,),) * 8 + ((3, 4), (3,))))
REPEATS = ((3995, 4010),)
HALVES = (("A", (("001", 20), ("002", 20))), ("B", (("001", 24),)))
TRUTH = ("7001", "7002", "7003", "7004", "7005", "7006", "7099")
# Imputed samples the crosswalk maps to truth persons (three per ancestry group, by parity); everyone
# else is research ID 5000 + index. 7099 has only a truth row.
TRUTH_OF_SEQUENCING = {"SA001": "7001", "SA002": "7003", "SA003": "7005", "SB001": "7002", "SB002": "7004", "SB003": "7006"}
KINSHIP = (("7003", "5010", 0.45), ("5020", "5021", 0.25), ("5030", "5031", 0.05))
FEMALE, MALE = 45878463, 45880669
BUDGET = ComputeBudget(
    device_kind="cpu", device_ids=(), device_names=(), device_bytes=(), device_compute_capabilities=(), host_bytes=1 << 29, cpu_threads=1
)


def _identifier(chromosome: str, record: int) -> str:
    position, _ref, alt, *_rest = RECORDS[record]
    return f"{chromosome}-{position}-allele{record}-{len(alt)}"


def _sequencing_names() -> list[list[list[str]]]:
    return [
        [[f"S{label}{start + column + 1:03d}" for column in range(width)] for start, (_batch, width) in zip(np.cumsum([0] + [w for _b, w in batches]), batches)]
        for label, batches in HALVES
    ]


def _research_of(name: str) -> str:
    """Imputed samples of half A are 5000..5039 and of half B 5040..5063, except the truth persons'."""
    if name in TRUTH_OF_SEQUENCING:
        return TRUTH_OF_SEQUENCING[name]
    offset = 0 if name[1] == "A" else sum(width for _batch, width in HALVES[0][1])
    return str(5000 + offset + int(name[2:]) - 1)


def _vcf_header(samples: list[str], formats: tuple[str, ...]) -> list[str]:
    lines = [
        "##fileformat=VCFv4.2",
        '##INFO=<ID=ID,Number=A,Type=String,Description="atomic ids">',
        '##INFO=<ID=CM,Number=1,Type=Float,Description="genetic position">',
        '##INFO=<ID=INFO,Number=1,Type=Float,Description="imputation r2">',
        '##INFO=<ID=SVLEN,Number=A,Type=Integer,Description="SV length">',
        *(f'##FORMAT=<ID={name},Number={"G" if name == "GP" else 1},Type={"String" if name == "GT" else "Float"},Description="{name}">' for name in formats),
        *(f"##contig=<ID={chromosome},length=50000000>" for chromosome in CHROMOSOMES),
    ]
    return lines + ["\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", *(["FORMAT"] if formats else []), *samples])]


def _reported_info(record: int) -> float:
    """The synthetic INFO/INFO of a record, the same in every batch."""
    return 0.5 + 0.05 * record


def _info(chromosome: str, record: int) -> str:
    fields = [f"ID={_identifier(chromosome, record)}", f"CM={0.001 * RECORDS[record][0]:.4f}", f"INFO={_reported_info(record):.3f}"]
    if RECORDS[record][5] is not None:
        fields.append(f"SVLEN={RECORDS[record][5]}")
    return ";".join(fields)


def _persons() -> list[str]:
    return sorted({_research_of(name) for half in _sequencing_names() for batch in half for name in batch} | set(TRUTH))


@dataclasses.dataclass
class _Synthetic:
    root: Path
    raw: dict
    genotypes: dict[tuple[str, str], np.ndarray]
    dosages: dict[tuple[str, str], np.ndarray]


def _write_inputs(root: Path) -> _Synthetic:
    """Every input file, and the genotypes and imputed dosages [records] per (chromosome, person) they were written from."""
    rng = np.random.default_rng(20260919)
    root.mkdir(parents=True)
    names = _sequencing_names()
    persons = _persons()
    frequencies = rng.uniform(0.15, 0.5, len(RECORDS))
    genotypes, dosages = {}, {}
    for chromosome in CHROMOSOMES:
        called = rng.binomial(2, frequencies[:, None], (len(RECORDS), len(persons)))
        called[:, persons.index("5010")] = called[:, persons.index("7003")]  # one genome under two research IDs
        noisy = np.round(np.clip(called + rng.normal(0.0, 0.2, called.shape), 0.0, 2.0), 3)
        single_path = np.array([record[4] == 1 for record in RECORDS])[:, None]
        # The single-path background q = w / (1 + w) rounds to 0.001: half of the zero dosages carry it.
        noisy[(noisy == 0.0) & single_path & (rng.random(noisy.shape) < 0.5)] = 0.001
        for column, person in enumerate(persons):
            genotypes[(chromosome, person)] = called[:, column]
            dosages[(chromosome, person)] = noisy[:, column]

    for half_index, (label, batches) in enumerate(HALVES):
        (root / label).mkdir()
        for (batch, _width), samples in zip(batches, names[half_index]):
            for chromosome in CHROMOSOMES:
                lines = _vcf_header(samples, ("GT", "DS", "GP"))
                for record, (position, ref, alt, *_rest) in enumerate(RECORDS):
                    fields = []
                    for sample in samples:
                        dosage = float(dosages[(chromosome, _research_of(sample))][record])
                        second = max(dosage - 1.0, 0.0)
                        first = dosage - 2.0 * second
                        fields.append(f"0|0:{dosage:.3f}:{1.0 - first - second:.3f},{first:.3f},{second:.3f}")
                    lines.append("\t".join([chromosome, str(position), ".", ref, alt, ".", "PASS", _info(chromosome, record), "GT:DS:GP", *fields]))
                (root / label / f"{chromosome}.{batch}.vcf").write_text("\n".join(lines) + "\n")
    (root / "truth").mkdir()
    for chromosome in CHROMOSOMES:
        lines = _vcf_header(list(TRUTH), ("GT",))
        for record, (position, ref, alt, *_rest) in enumerate(RECORDS):
            calls = [("0/0", "0/1", "1/1")[int(genotypes[(chromosome, person)][record])] for person in TRUTH]
            if chromosome == "chr21" and record == 0:
                calls[1] = "./."
            lines.append("\t".join([chromosome, str(position), ".", ref, alt, ".", "PASS", _info(chromosome, record), "GT", *calls]))
        (root / "truth" / f"{chromosome}.vcf").write_text("\n".join(lines) + "\n")

    strata = root / "strata"
    (strata / "_done").mkdir(parents=True)
    (root / "bubbles").mkdir()
    for chromosome in CHROMOSOMES:
        rows = ["#idx\tpos\tid\trefalt_md5\tref_len\talt_len\tclass\tn_paths\tn_paths_total\tcx"]
        sites, ids = hashlib.md5(), hashlib.md5()
        for record, (position, ref, alt, carried, total, _length) in enumerate(RECORDS):
            identifier = _identifier(chromosome, record)
            rows.append(f"{record}\t{position}\t{identifier}\t{refalt_digest(ref, alt):016x}\t{len(ref)}\t{len(alt)}\tsnv\t{carried}\t{total}\t{int(total > 1)}")
            sites.update(f"{chromosome}\t{position}\t{ref}\t{alt}\n".encode())
            ids.update(f"{identifier}\n".encode())
        with gzip.open(strata / f"{chromosome}.strata.tsv.gz", "wt") as handle:
            handle.write("\n".join(rows) + "\n")
        (strata / "_done" / f"{chromosome}.json").write_text(json.dumps({"sites_md5": sites.hexdigest(), "ids_md5": ids.hexdigest()}))
        lines = _vcf_header([], ())
        for bubble, (position, paths) in enumerate(BUBBLES):
            for path, carried in enumerate(paths):
                separator = "," if (bubble, path) == (1, 10) else ":"
                listed = separator.join(_identifier(chromosome, record) for record in carried)
                lines.append("\t".join([chromosome, str(position), ".", "A", "C", ".", "PASS", f"ID={listed}"]))
        (root / "bubbles" / f"{chromosome}.bubble.split.vcf").write_text("\n".join(lines) + "\n")
    with gzip.open(root / "repeats.bed.gz", "wt") as handle:
        handle.write("".join(f"{chromosome}\t{start}\t{end}\tTR\n" for chromosome in CHROMOSOMES for start, end in REPEATS))

    all_names = [name for half in names for batch in half for name in batch]
    (root / "crosswalk.tsv").write_text("research_id\tsequencing_id\n" + "".join(f"{_research_of(name)}\t{name}\n" for name in all_names))
    (root / "ancestry.tsv").write_text(
        "research_id\tancestry_pred\tpca_features\n"
        + "".join(f"{person}\t{('afr', 'eur')[int(person) % 2]}\t{json.dumps(rng.normal(size=3).round(4).tolist())}\n" for person in persons)
    )
    (root / "relatedness.tsv").write_text("i.s\tj.s\tkin\n" + "".join(f"{first}\t{second}\t{kinship}\n" for first, second, kinship in KINSHIP))
    raw = {
        "run_directory": str(root.parent / "run"),
        "marker_directory": str(root.parent / "markers"),
        "chromosomes": list(CHROMOSOMES),
        "imputed_halves": [
            {"label": label, "batches": [batch for batch, _width in batches], "batch_path": str(root / label / "{chromosome}.{batch}.vcf")}
            for label, batches in HALVES
        ],
        "truth_calls": str(root / "truth" / "{chromosome}.vcf"),
        "strata_directory": str(strata),
        "bubble_split": str(root / "bubbles" / "{chromosome}.bubble.split.vcf"),
        "tandem_repeats": str(root / "repeats.bed.gz"),
        "crosswalk": str(root / "crosswalk.tsv"),
        "crosswalk_columns": ["research_id", "sequencing_id"],
        "ancestry": str(root / "ancestry.tsv"),
        "relatedness": str(root / "relatedness.tsv"),
        "relatedness_columns": ["i.s", "j.s", "kin"],
        "diseases": ["atrial_fibrillation"],
        "traits": ["total_bilirubin"],
        "fold_count": 2,
        "seed": 7,
        "codec": "zstd",
        "export": ["accuracy.tsv"],
    }
    return _Synthetic(root=root, raw=raw, genotypes=genotypes, dosages=dosages)


def _omop_rows(persons: list[str]) -> tuple[list[dict], list[dict]]:
    """Disease-query and measurement-query rows: every person has EHR and 9001 has no genotypes. 5049 has one
    afib date (neither case nor control, so no disease row) and, like every person numbered 1 mod 8, no
    bilirubin occasion, so no table lists it."""
    rng = np.random.default_rng(3)
    disease, measurement = [], []
    for person in persons + ["9001"]:
        number = int(person)
        sex = FEMALE if number % 2 == 0 else MALE
        occurrences = 1 if person == "5049" else (3 if number % 3 == 0 else 0)
        disease.append({
            "sample_id": person, "person_id": person, "phenotype_occurrence_count": occurrences,
            "first_condition_date": "2020-01-01" if occurrences else None, "observation_start_date": "2015-01-01",
            "observation_end_date": "2024-01-01", "primary_consent_date": "2018-06-01", "year_of_birth": 1975,
            "age_at_first_condition": 40.0 + number % 30 if occurrences else None, "age_at_first_case_procedure": None,
            "age_at_observation_end": 45.0 + number % 30, "pre_landmark_condition_dates": 6 + number % 7,
            "has_control_exclusion_code": False, "has_ambiguous_code": False, "case_medication_dates": 0,
            "has_control_exclusion_medication": False, "case_procedure_dates": 0, "sex_at_birth_concept_id": sex,
            "sex_at_birth_name": "female" if sex == FEMALE else "male",
        })
        if number % 8 == 1:
            continue
        for day in range(int(rng.integers(2, 5))):
            measurement.append({
                "sample_id": person, "person_id": person, "measurement_date": datetime.date(2016, 1, 1) + datetime.timedelta(days=day),
                "age_at_occasion": float(rng.uniform(30.0, 70.0)), "treated": False, "row_count": 1,
                **{f"{reason}_row_count": 0 for reason in MEASUREMENT_EXCLUSION_REASONS}, "retained_row_count": 1,
                "occasion_value": round(float(1.0 + 0.2 * rng.normal()), 1), "unrecognized_unit_labels": [],
                "sex_at_birth_concept_id": sex, "sex_at_birth_name": "female" if sex == FEMALE else "male",
            })
    return disease, measurement


class _FakeBigQuery:
    project = "billing-project"

    def __init__(self, disease: list[dict], measurement: list[dict]) -> None:
        self.disease, self.measurement = disease, measurement

    def query(self, sql: str, job_config):
        rows = self.measurement if sql == build_all_of_us_measurement_sql() else self.disease
        return SimpleNamespace(result=lambda: [SimpleNamespace(items=row.items) for row in rows])


@dataclasses.dataclass(frozen=True)
class _Moments:
    pair_counts: np.ndarray
    dosage_mean: np.ndarray
    truth_mean: np.ndarray
    dosage_variance: np.ndarray
    truth_variance: np.ndarray
    covariance: np.ndarray


def _calibration_moments(dosage: np.ndarray, truth: np.ndarray) -> _Moments:
    complete = np.isfinite(dosage) & np.isfinite(truth)
    counts = complete.sum(axis=1)
    d, t = np.where(complete, dosage, 0.0), np.where(complete, truth, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        dm, tm = d.sum(axis=1) / counts, t.sum(axis=1) / counts
        return _Moments(
            counts, dm, tm, (np.where(complete, d - dm[:, None], 0.0) ** 2).sum(axis=1) / counts,
            (np.where(complete, t - tm[:, None], 0.0) ** 2).sum(axis=1) / counts,
            (np.where(complete, (d - dm[:, None]) * (t - tm[:, None]), 0.0)).sum(axis=1) / counts,
        )


def _fit_measurement_model(calibration, cohort_dosage_variance, strata, design=None, reported_reliability=None):
    """A stand-in with the production signature: kappa = Cov(T, D) / Var(D) where there are three pairs, else 1."""
    assert strata.shape == cohort_dosage_variance.shape and set(strata) <= set(VARIANT_CLASS_LEGEND)
    assert reported_reliability is not None and np.all((reported_reliability >= 0.0) & (reported_reliability <= 1.0))
    scales = np.ones_like(cohort_dosage_variance)
    if calibration is not None:
        moments = calibration.moments
        usable = (moments.pair_counts >= 3) & (moments.dosage_variance > 0.0) & (moments.covariance > 0.0)
        scales[usable] = moments.covariance[usable] / moments.dosage_variance[usable]
    certificate = {"recalibration": "not applied: no truth genotypes were supplied" if calibration is None else "applied"}
    return SimpleNamespace(scales=scales, residual_variance=np.zeros_like(scales), log_reliability=np.zeros_like(scales), certificate=certificate)


@dataclasses.dataclass(frozen=True)
class _MeasurementModel:
    """A stand-in for measurement_model.MeasurementModel's arrays and its npz save and load."""

    scales: np.ndarray
    residual_variance: np.ndarray
    log_reliability: np.ndarray

    def save(self, path: Path) -> None:
        np.savez(path, **dataclasses.asdict(self))

    @classmethod
    def load(cls, path: Path) -> _MeasurementModel:
        with np.load(path) as archive:
            return cls(**{name: archive[name] for name in archive.files})


def _pooled_measurement_model(models, cohort_dosage_mean, cohort_dosage_variance, group_counts) -> _MeasurementModel:
    """A stand-in with the production signature: log Var(D*) / (Var(D*) + v) of the stacked groups, -inf without variance."""
    scales = np.column_stack([model.scales for model in models])
    residual = np.column_stack([model.residual_variance for model in models])
    weights = group_counts / group_counts.sum()
    mean = cohort_dosage_mean @ weights
    variance = (scales**2 * cohort_dosage_variance + (cohort_dosage_mean - mean[:, None]) ** 2) @ weights
    total = variance + residual @ weights
    with np.errstate(divide="ignore", invalid="ignore"):
        log_reliability = np.where(total > 0.0, np.log(variance / total), -np.inf)
    return _MeasurementModel(np.ones_like(mean), residual @ weights, log_reliability)


@dataclasses.dataclass
class _Fitted:
    model_names: tuple[str, ...]
    covariate_names: tuple[str, ...]
    scoring: tuple[ScoringModel, ...]
    noise_variance: np.ndarray
    refusals: tuple[str, ...] = ()


class _Recorder:
    def __init__(self, failing_fit: bool = False) -> None:
        self.fit_calls: list[FitRequest] = []
        self.failing_fit = failing_fit

    def fit(self, request: FitRequest) -> _Fitted:
        """A stand-in for the engine behind the real ``fit_model.FitRequest``: each model's own covariates by least
        squares, then marginal effects of the standardized codes on the training rows."""
        if self.failing_fit:
            raise RuntimeError("preempted")
        assert isinstance(request, FitRequest)
        self.fit_calls.append(request)
        store, store_columns, covariates = request.store, request.store_columns, request.covariates
        covariate_columns, targets, training = request.covariate_columns, request.targets, request.training
        model_names, trait_types, work_dir = request.model_names, request.trait_types, request.work_dir
        offset = request.log_variance_offset
        assert offset.shape == (store.n_variants,) and np.all(offset <= 0.0) and work_dir.is_dir()
        codes = store.read_codes(0, store.n_variants, np.asarray(store_columns)).astype(np.float64) - 127.0
        models = []
        for model, trait_type in enumerate(trait_types):
            rows, own = training[:, model], covariate_columns[model]
            signed = codes[:, rows]
            means, scales = signed.mean(axis=1), signed.std(axis=1)
            active = np.flatnonzero(scales > 0.0)
            target = targets[rows, model]
            design = np.column_stack([np.ones(rows.sum()), covariates[rows][:, own]])
            coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
            standardized = (signed[active] - means[active, None]) / scales[active, None]
            effects = standardized @ (target - design @ coefficients) / rows.sum() / len(active)
            draws = effects[:, None] + np.random.default_rng(model).normal(0.0, 1e-3, (len(active), 4))
            alpha = np.zeros(1 + covariates.shape[1])
            if trait_type != TraitType.BINARY:
                alpha[0], alpha[1:][own] = coefficients[0], coefficients[1:]
            models.append(ScoringModel(active.astype(np.int64), means[active], scales[active], effects, draws, alpha, trait_type, 0.0))
        return _Fitted(tuple(model_names), tuple(request.covariate_names), tuple(models), np.ones(len(models)))


def _save_model(path: Path, model: _Fitted) -> None:
    path.mkdir()
    (path / "model.pickle").write_bytes(pickle.dumps(model))


def _load_model(path: Path) -> _Fitted:
    return pickle.loads((path / "model.pickle").read_bytes())


class _StoreBlocks:
    def __init__(self, store: DosageStore, budget: ComputeBudget) -> None:
        self.store, self.budget = store, budget

    @property
    def sample_count(self) -> int:
        return int(self.store.n_samples)

    def iter_code_blocks(self, variant_ranges, buffers):
        yield from self.store.iter_codes(variant_ranges, None, self.budget)


def _predict(model: _Fitted, store: DosageStore, sample_indices, covariates, budget):
    genetic = score_genetic(_StoreBlocks(store, budget), ScoringPlan.from_models(model.scoring), budget, np.asarray(sample_indices))
    linear = score_linear_predictor(genetic.means, covariates, model.scoring)
    mean = linear.copy()
    for index, scoring in enumerate(model.scoring):
        if scoring.trait_type == TraitType.BINARY:
            mean[:, index] = posterior_predictive_probability(linear[:, index], genetic.variances[:, index], 0.0)
    return SimpleNamespace(genetic=genetic, linear_predictor=linear, predictive_mean=mean, predictive_variance=np.ones_like(mean))


def _bindings(recorder: _Recorder, client: _FakeBigQuery) -> WorkspaceBindings:
    return WorkspaceBindings(
        bigquery_client=lambda: client,
        calibration_moments=_calibration_moments,
        concatenate_calibration_moments=lambda parts: _Moments(
            *(np.concatenate([getattr(part, field.name) for part in parts]) for field in dataclasses.fields(_Moments))
        ),
        calibration_pairs=lambda sample_ids, moments, blocks=(): SimpleNamespace(sample_ids=sample_ids, moments=moments, blocks=blocks),
        fit_measurement_model=_fit_measurement_model,
        pooled_measurement_model=_pooled_measurement_model,
        load_measurement_model=_MeasurementModel.load,
        fit=recorder.fit,
        save_model=_save_model,
        load_model=_load_model,
        predict=_predict,
    )


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GOOGLE_PROJECT", "billing-project")
    monkeypatch.setenv("WORKSPACE_CDR", "aou_workspace.cdr_dataset")
    # The occasion model at the identity transform in a small budget; its exponent search is tested in test_all_of_us_phenotypes.
    monkeypatch.setattr("sv_pgs.all_of_us.fit_occasion_model", lambda occasions, working_bytes: fit_at_exponent(occasions, 1.0, min(working_bytes, 1 << 28)))
    return _write_inputs(tmp_path / "inputs"), _FakeBigQuery(*_omop_rows(_persons()))


def _completed_at(run: Path, step: str) -> float:
    return (run / step / "_COMPLETE.json").stat().st_mtime_ns


def test_the_production_bindings_hand_the_fit_step_s_request_to_the_real_fitter() -> None:
    """No stand-in: the modules the run imports are the real ones, and ``fit`` takes the ``FitRequest`` the fit step
    builds, so the step's call cannot drift from the fitter's input."""
    from sv_pgs import artifact, fit_model, measurement_model
    from sv_pgs.workspace_pipeline import workspace_bindings

    bindings = workspace_bindings()
    assert bindings.fit is fit_model.fit and bindings.save_model is artifact.save_model
    assert bindings.load_model is artifact.load_model and bindings.predict is artifact.predict
    assert bindings.load_measurement_model == measurement_model.MeasurementModel.load
    (parameter,) = inspect.signature(fit_model.fit).parameters.values()
    assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and parameter.default is inspect.Parameter.empty
    # The request the fit step builds is the fitter's own class, not a copy of its fields.
    assert workspace_pipeline.FitRequest is fit_model.FitRequest is FitRequest


def test_a_dry_run_builds_every_step_from_synthetic_inputs(workspace) -> None:
    synthetic, client = workspace
    recorder = _Recorder()
    config = WorkspaceConfig.from_mapping(synthetic.raw)
    summaries = run_pipeline(config, _bindings(recorder, client), BUDGET)
    run = config.run_directory
    assert tuple(summaries) == STEP_NAMES and all((run / step / "_COMPLETE.json").is_file() for step in STEP_NAMES)
    assert {path.name.split("_", 1)[1] for path in config.marker_directory.iterdir()} >= {f"{step}_completed" for step in STEP_NAMES}

    # Cohort rows: each truth row wins over its imputed row; 5010, which duplicates 7003's genome, is dropped.
    rows = {row.split("\t")[0]: row.split("\t") for row in (run / "samples" / "rows.tsv").read_text().splitlines()[1:]}
    assert {person for person, row in rows.items() if row[1] == "long_read"} == set(TRUTH)
    assert "5010" not in rows and rows["5020"][5] == rows["5021"][5]
    # Second-degree relatives form one family; a pair below the KING second-degree bound does not.
    assert rows["5020"][6] == rows["5021"][6] and rows["5030"][6] != rows["5031"][6]
    assert summaries["samples"]["calibration_pairs"] == len(TRUTH_OF_SEQUENCING)

    # The store: typed half manifests, background-corrected imputed codes, long-read hard calls.
    with DosageStore.open(run / "store" / "store") as store:
        halves = store.half_samples()
        assert [half.namespace for half in halves] == ["dragen_sample", "dragen_sample", "research_id"]
        assert halves[2] == HalfSamples("research_id", TRUTH)
        names = [name for batch in _sequencing_names()[0] for name in batch]
        stored = store.read_codes(0, len(RECORDS), np.arange(len(names)))
        milli = np.rint(np.column_stack([synthetic.dosages[("chr21", _research_of(name))] for name in names]) * 1000).astype(np.uint16)
        kept = np.array([min(10, record[4]) for record in RECORDS])
        carrying = np.array([1, 2, 2, 1, 1, 8, 1, 1, 1])
        np.testing.assert_array_equal(stored, encode_dosage_milli(value_matched_background(milli, kept, carrying).dosage_milli))
        truth_codes = store.read_codes(0, len(RECORDS), np.arange(store.n_samples - len(TRUTH), store.n_samples))
        called = np.column_stack([synthetic.genotypes[("chr21", person)] for person in TRUTH]) * 127
        # 7002's no-call at the first record is filled with its group's measured mean; every call is its ALT count.
        assert truth_codes[0, 1] != 255
        np.testing.assert_array_equal(np.delete(truth_codes[0], 1), np.delete(called[0], 1))
        np.testing.assert_array_equal(truth_codes[1:], called[1:])
        table = store.variant_table
        classes = [VARIANT_CLASS_LEGEND[code] for code in table.variant_class[: len(RECORDS)]]
        assert classes == ["snv", "deletion", "insertion", "deletion", "insertion", "deletion", "str_vntr_repeat", "snv", "insertion"]
        assert table.group_first[: len(RECORDS)].tolist() == [0, 1, 1, 3, 3, 3, 6, 7, 8]
        assert table.annotations["n_paths_total"][: len(RECORDS)].tolist() == [record[4] for record in RECORDS]
    assert summaries["store"]["background_zeroed"] > 0 and summaries["store"]["no_calls"] == 1
    # The reported r^2 the measurement model falls back on is the batches' INFO/INFO, not a dosage-variance ratio.
    np.testing.assert_allclose(np.load(run / "store" / "reported" / "chr21.npy"), [_reported_info(record) for record in range(len(RECORDS))])

    # D*: the imputed halves were rewritten with the measurement's scales; the MANIFEST says so.
    measurement = np.load(run / "measurement" / "measurement.npz")
    assert summaries["measurement"]["recalibrated"] and np.any(measurement["scales"] != 1.0)
    with DosageStore.open(run / "store" / "store", half_indices=[0]) as before, DosageStore.open(run / "measurement" / "store", half_indices=[0]) as after:
        groups = np.load(run / "samples" / "groups_half0.npy")
        stored_milli = ((before.read_codes(0, before.n_variants).astype(np.int64) * 2000 + 127) // 254).astype(np.uint16)
        expected = linear_recalibration(stored_milli, groups, measurement["scales"][:, : groups.max() + 1]).dosage_milli
        np.testing.assert_array_equal(after.read_codes(0, after.n_variants), encode_dosage_milli(expected))
        assert after.manifest_attributes["recalibrated"] is True

    # The fit saw every (trait, fold) training set, with no held-out row, no intercept column, and each
    # model's own trait's covariates only.
    (call,) = recorder.fit_calls
    cohort = np.load(run / "cohort" / "cohort.npz")
    design = json.loads((run / "cohort" / "cohort.json").read_text())
    assert design["model_names"] == ["atrial_fibrillation/fold0", "atrial_fibrillation/fold1", "total_bilirubin/fold0", "total_bilirubin/fold1"]
    names = design["covariate_names"]
    structure = {"intercept", "pipeline_half=B", "genotype_source=long_read", "PC1", "PC2", "PC3"}
    own = {
        "atrial_fibrillation": {
            "atrial_fibrillation:age_at_observation_end", "atrial_fibrillation:age_at_observation_end_squared",
            "atrial_fibrillation:age_at_observation_end_x_female", "atrial_fibrillation:log1p_pre_landmark_condition_dates",
            "atrial_fibrillation:sex_at_birth_concept_id=45880669",
        },
        "total_bilirubin": {
            "total_bilirubin:age_at_measurement", "total_bilirubin:age_at_measurement_squared",
            "total_bilirubin:age_at_measurement_x_female", "total_bilirubin:sex_at_birth_concept_id=45880669",
        },
    }
    assert set(names) == structure | own["atrial_fibrillation"] | own["total_bilirubin"]
    assert summaries["cohort"]["covariates"] == {trait: [name for name in names if name in structure | columns] for trait, columns in own.items()}
    assert call.covariates.shape[1] == len(call.covariate_names) == len(names) - 1
    for model, trait in enumerate(["atrial_fibrillation"] * 2 + ["total_bilirubin"] * 2):
        assert {name for name, used in zip(names[1:], call.covariate_columns[model]) if used} == (structure | own[trait]) - {"intercept"}
    # A trait's own columns are 0 on the rows it does not observe.
    bilirubin_age = cohort["covariates"][:, names.index("total_bilirubin:age_at_measurement")]
    assert np.all(bilirubin_age[~np.isfinite(cohort["targets"][:, 1])] == 0.0) and np.all(bilirubin_age[np.isfinite(cohort["targets"][:, 1])] > 0.0)
    families = cohort["families"]
    assert np.array_equal(np.unique(families), np.arange(families.max() + 1)) and len(set(families.tolist())) == len(families) - 1
    fitted_rows = cohort["training"].any(axis=1)
    folds = cohort["folds"][fitted_rows]
    for model in range(4):
        assert not np.any(call.training[:, model] & (folds == model % 2))
        assert np.all(np.isnan(call.targets[~call.training[:, model], model]))
    # 5049 is genotyped but in no table: a cohort row with no target and no fit row.
    untabled = list(cohort["research_ids"]).index("5049")
    assert np.all(np.isnan(cohort["targets"][untabled])) and not fitted_rows[untabled] and "7099" in cohort["research_ids"]
    assert len(call.research_ids) == len(call.store_columns)
    saved = _MeasurementModel.load(run / "measurement" / "measurement_model.npz")
    np.testing.assert_array_equal(call.log_variance_offset, saved.log_reliability)

    # Scores: every observed target has its own fold's held-out prediction.
    held_out = np.load(run / "score" / "predictions.npz")["held_out"]
    assert np.array_equal(np.isfinite(held_out), np.isfinite(cohort["targets"]))

    # The report suppresses cells of 1-20 people; export stages the approved file and nothing is sent.
    report = [line.split("\t") for line in (run / "report" / "accuracy.tsv").read_text().splitlines()]
    long_read = [line for line in report if line[1:3] == ["half", "long_read"]]
    assert all(line[3] == "1-20 (suppressed)" and line[5] == "" for line in long_read)
    assert any(line[1] == "fold" and line[5] for line in report[1:])
    assert (run / "export" / "accuracy.tsv").read_text() == (run / "report" / "accuracy.tsv").read_text()


def test_a_rerun_skips_completed_steps_and_resumes_after_a_failure(workspace) -> None:
    synthetic, client = workspace
    config = WorkspaceConfig.from_mapping(synthetic.raw)
    with pytest.raises(RuntimeError, match="preempted"):
        run_pipeline(config, _bindings(_Recorder(failing_fit=True), client), BUDGET)
    run = config.run_directory
    assert (run / "fit.partial").is_dir() and any(path.name.endswith("_fit_failed-RuntimeError") for path in config.marker_directory.iterdir())
    finished = {step: _completed_at(run, step) for step in STEP_NAMES[:5]}
    recorder = _Recorder()
    run_pipeline(config, _bindings(recorder, client), BUDGET)
    assert {step: _completed_at(run, step) for step in STEP_NAMES[:5]} == finished
    assert len(recorder.fit_calls) == 1 and (run / "export" / "_COMPLETE.json").is_file()

    # A changed phenotype set re-runs the steps that read it and keeps the store.
    fewer = WorkspaceConfig.from_mapping({**synthetic.raw, "traits": []})
    run_pipeline(fewer, _bindings(_Recorder(), client), BUDGET)
    assert _completed_at(run, "store") == finished["store"] and _completed_at(run, "samples") == finished["samples"]
    assert json.loads((run / "cohort" / "cohort.json").read_text())["trait_names"] == ["atrial_fibrillation"]
    assert any(path.name.startswith("phenotypes.superseded-") for path in run.iterdir())


def test_without_truth_calls_the_run_says_so_and_keeps_the_stored_dosages(workspace) -> None:
    synthetic, client = workspace
    two_threads = dataclasses.replace(BUDGET, cpu_threads=2)
    config = WorkspaceConfig.from_mapping({**synthetic.raw, "truth_calls": None})
    summaries = run_pipeline(config, _bindings(_Recorder(), client), two_threads, through="measurement")
    run = config.run_directory
    assert summaries["samples"]["calibration_pairs"] == 0 and summaries["store"]["halves"] == ["imputed_dosage", "imputed_dosage"]
    assert not summaries["measurement"]["recalibrated"] and summaries["measurement"]["store"] == "store/store"
    certificate = json.loads((run / "measurement" / "certificate.json").read_text())
    assert all(entry["pairs_supplied"] == 0 and entry["recalibration"].startswith("not applied") for entry in certificate.values())
    rows = (run / "samples" / "rows.tsv").read_text()
    assert "long_read" not in rows and "\n5010\t" in rows


def test_a_sidecar_that_disagrees_with_the_popped_records_stops_the_store(workspace) -> None:
    synthetic, client = workspace
    done = synthetic.root / "strata" / "_done" / "chr22.json"
    done.write_text(json.dumps({**json.loads(done.read_text()), "ids_md5": "0" * 32}))
    with pytest.raises(ValueError, match="md5"):
        run_pipeline(WorkspaceConfig.from_mapping(synthetic.raw), _bindings(_Recorder(), client), BUDGET, through="store")


def _strata(identifiers, carried, totals) -> StrataSites:
    count = len(identifiers)
    zeros = np.zeros(count, dtype=np.int64)
    return StrataSites(zeros, tuple(identifiers), zeros.astype(np.uint64), zeros, zeros, np.array(carried), np.array(totals), zeros.astype(float), None, "", "")


def test_path_counts_count_only_the_first_kept_paths_and_check_the_sidecar() -> None:
    bubble = [frozenset({"x"})] + [frozenset({"y"})] * 10 + [frozenset({"x", "y"})]
    counts = path_counts(_strata(["x", "y", "z"], [2, 11, 1], [12, 12, 1]), [bubble])
    assert counts.kept.tolist() == [10, 10, 1] and counts.carrying.tolist() == [1, 9, 1] and counts.bubble.tolist() == [0, 0, -1]
    with pytest.raises(ValueError, match="sidecar says"):
        path_counts(_strata(["x"], [3], [12]), [bubble])
    with pytest.raises(ValueError, match="no bubble.split path"):
        path_counts(_strata(["w"], [1], [2]), [bubble])


def test_export_refuses_a_count_of_one_to_twenty(tmp_path: Path) -> None:
    table = tmp_path / "accuracy.tsv"
    table.write_text("trait\tcell\tlevel\tn\tmetric\tvalue\nldl\tfold\t0\t25\tr2\t0.1\nldl\thalf\tlong_read\t1-20 (suppressed)\t\t\n")
    check_exportable(table)
    table.write_text("trait\tcell\tlevel\tn\tmetric\tvalue\nldl\thalf\tlong_read\t7\tr2\t0.3\n")
    with pytest.raises(ValueError, match="below 21"):
        check_exportable(table)


def test_a_long_read_row_takes_the_reference_half_so_the_indicators_differ() -> None:
    assert pipeline_half_levels(["B", "A", "long_read"], ["imputed", "imputed", "long_read"]) == ("B", "A", "A")


def test_the_config_names_every_key_and_canonical_phenotypes(workspace) -> None:
    synthetic, _client = workspace
    with pytest.raises(ValueError, match="every key is required"):
        WorkspaceConfig.from_mapping({key: value for key, value in synthetic.raw.items() if key != "truth_calls"})
    with pytest.raises(ValueError, match="canonical name"):
        WorkspaceConfig.from_mapping({**synthetic.raw, "diseases": ["afib"]})


LAUNCHER = Path(__file__).resolve().parents[1] / "launcher" / "workspace"


def _filled(text: str) -> str:
    """A template with its numeric placeholders set to 1, the two half labels to A and B, and every other
    placeholder to its own name in lower case."""
    for name, value in (("FOLD_SEED", "1"), ("MAX_RETRIES", "1"), ("GPU_COUNT", "1"), ("HALF_A_LABEL", "A"), ("HALF_B_LABEL", "B")):
        text = text.replace("${" + name + "}", value)
    return re.sub(r"\$\{([A-Z_]+)\}", lambda match: match.group(1).lower(), text)


def test_the_launcher_templates_parse_and_name_the_preregistered_panel() -> None:
    raw = json.loads(_filled((LAUNCHER / "run_config.template.json").read_text()))
    config = WorkspaceConfig.from_mapping(raw)
    assert (len(config.diseases), len(config.traits), config.fold_count) == (10, 11, 5) and config.truth_calls is None
    for job in ("job_store.json.template", "job_fit.json.template"):
        network = json.loads(_filled((LAUNCHER / job).read_text()))["allocationPolicy"]["network"]["networkInterfaces"][0]
        assert network["noExternalIpAddress"] is True


def test_the_launcher_sends_nothing_outside_the_workspace() -> None:
    text = "\n".join(path.read_text().lower() for path in LAUNCHER.iterdir() if path.name != "README.md")
    # The service account's "email" key names an identity inside the workspace; it sends nothing.
    assert not re.search(r"https?://|curl|wget|webhook|notif|sendmail|smtp|\bmail\b|slack|pubsub|scp |rsync", text)
