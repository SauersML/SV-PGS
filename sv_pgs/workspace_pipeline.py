"""The pipeline that runs inside the All of Us workspace: popped BCFs to held-out scores.

One driver, restartable step by step (docs/design/WORKSPACE_PIPELINE.md). A step writes into
``<run>/<step>.partial`` and is complete once that directory, holding ``_COMPLETE.json`` with the
step's key, is renamed to ``<run>/<step>``. The key is a digest of the configuration the step
reads, the keys of the steps it depends on, and the source of the code it runs. A rerun:

- skips every step whose key is unchanged;
- resumes a partial step from its sub-checkpoints (the store's decoded batches and finished
  chromosomes);
- moves a step whose key changed aside as ``<step>.superseded-<key>``.

Nothing is deleted except the store step's own decoded batch files, once their chromosome is
written.

Steps:

1. ``samples``: the halves' typed sample manifests from the batch headers. Then the cohort rows
   (``cohort.resolve_cohort_rows``: the truth row wins, an imputed sample maps to a person only
   through the crosswalk, KING duplicates of a truth genome are dropped), and the trait-agnostic
   kinship folds over both halves, frozen before any phenotype exists (EVALUATION.md, "Folds").
2. ``phenotypes``: each configured disease's and trait's sample table from the CDR (``all_of_us``).
3. ``cohort``: over the cohort rows, the structure covariates every trait shares, each trait's own
   covariates from its own table, one target column per trait (``cohort.build_cohort``), and every
   (trait, fold) training set.
4. ``store``: the dosage store from the popped batches (``store_converter``). Every batch is
   checked in lockstep against the strata sidecar, and the popped records against the sidecar's
   sites and ids md5. Each half's sample manifest is typed by its namespace.
5. ``measurement``: the measurement model, fitted on the calibration pairs from the truth rows, or
   its explicit no-truth form, and the fit's prior offset from it. Where it recalibrates, the store
   is rewritten with D*.
6. ``fit``: every (trait, fold) model in one call, each projecting out its own trait's covariates,
   saved to disk.
7. ``score``: every model's predictions for the whole cohort from one read of the store. Each
   person's held-out prediction is the one from their own fold's model.
8. ``report``: held-out accuracy per trait, in cells by fold, ancestry and half, with any count
   of 1 to 20 suppressed.
9. ``export``: stages in ``<run>/export`` the report files the user named in the config, and
   nothing else.

Nothing is ever sent anywhere. Moving a staged file out of the workspace is the user's action.
Everything here reads and writes participant-derived data, so every output stays in the
workspace (STORE.md, "Data class"). Progress is visible only as empty marker files whose names
hold the step and the event.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import csv
import dataclasses
from dataclasses import dataclass
import gzip
import hashlib
import importlib
import inspect
import io
import json
import logging
import multiprocessing
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Mapping, Sequence

from cyvcf2 import VCF
import numpy as np
import pandas as pd

from sv_pgs import all_of_us, cohort as cohort_module, dosage_store, sample_crosswalk, store_converter, variant_typing
from sv_pgs._typing import BoolArray, F64Array, I64Array, NDArray, U8Array
from sv_pgs.all_of_us import (
    MINIMUM_REPORTED_PARTICIPANTS,
    phenotype_fingerprint,
    prepare_all_of_us_disease_sample_table,
    prepare_all_of_us_measurement_sample_table,
    resolve_disease_definition,
    resolve_measurement_definition,
)
from sv_pgs.cohort import (
    LONG_READ_SOURCE,
    AncestryPcs,
    TraitTable,
    build_cohort,
    kinship_components,
    kinship_folds,
    pipeline_half_levels,
    read_kinship_pairs,
    read_predicted_ancestry,
    reportable_count,
    resolve_cohort_rows,
)
from sv_pgs.copy_number import allele_count_decode
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.fit_model import FitRequest
from sv_pgs.config import TraitType, VariantClass
from sv_pgs.dosage_store import (
    CODES_PER_DOSAGE,
    MISSING_CODE,
    VARIANT_CLASS_LEGEND,
    VARIANT_CLASSES,
    Codec,
    DosageStore,
    HalfSamples,
    VariantTable,
    chromosome_number,
    dosage_array_directory,
    read_manifest,
    sites_md5,
    statistic_column_directory,
    write_column,
    write_variant_columns,
)
from sv_pgs.sample_crosswalk import SampleCrosswalk, store_research_ids
from sv_pgs.sample_ids import ResearchId
from sv_pgs.store_converter import (
    MAXIMUM_KEPT_PATHS,
    NO_LOCUS,
    ExpectedSites,
    TrLoci,
    assemble_half,
    core_spans,
    decode_batch,
    decode_called_batch,
    no_call_fill,
    refalt_digest,
    tr_loci,
    unbreakable_group_first,
    write_store_manifest,
)
from sv_pgs.variant_typing import variant_class_and_length

LOGGER = logging.getLogger(__name__)

STEP_NAMES = ("samples", "phenotypes", "cohort", "store", "measurement", "fit", "score", "report", "export")
COMPLETE_FILE = "_COMPLETE.json"
KEY_FILE = "_KEY"
EXPORTABLE_FILE = "_EXPORTABLE.json"
CONFIG_KEYS = (
    "run_directory",
    "marker_directory",
    "chromosomes",
    "imputed_halves",
    "truth_calls",
    "strata_directory",
    "bubble_split",
    "tandem_repeats",
    "crosswalk",
    "crosswalk_columns",
    "ancestry",
    "relatedness",
    "relatedness_columns",
    "diseases",
    "traits",
    "fold_count",
    "seed",
    "codec",
    "export",
)
STRATA_COLUMNS = ("idx", "pos", "id", "refalt_md5", "ref_len", "alt_len", "n_paths", "n_paths_total", "cx")
"""The strata sidecar v2 columns the store needs (the imputation's chrK.strata.tsv.gz, header after '#')."""
SEX_COLUMN = "sex_at_birth_concept_id"
"""The categorical covariate the sample tables write one-hot, as ``sex_at_birth_concept_id_<concept>``."""
UNRECORDED_SEX = "unrecorded"
MEASUREMENT_MODEL_FILE = "measurement_model.npz"
_INT64_BYTES = np.dtype(np.int64).itemsize
_FLOAT64_BYTES = np.dtype(np.float64).itemsize


# ---------------------------------------------------------------------------
# Configuration and bindings
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ImputedHalfInput:
    """One imputation half: its popped batches in batch order, and each (chromosome, batch) file's path pattern."""

    label: str
    batches: tuple[str, ...]
    batch_path: str

    def path(self, chromosome: str, batch: str) -> Path:
        return Path(self.batch_path.format(chromosome=chromosome, batch=batch))


@dataclass(frozen=True, slots=True)
class WorkspaceConfig:
    """The run's inputs, from the JSON config supplied inside the workspace.

    Every key is required, so nothing runs on a default nobody chose. ``truth_calls`` is either a
    path pattern over ``{chromosome}`` for the long-read hard calls on the popped site list
    (samples named by research ID), or an explicit null. Null runs without truth rows: no stacked
    long-read half, no calibration pairs, no D*, and each step's summary says so. ``export`` lists
    the report files the user has approved for staging, by name.
    """

    run_directory: Path
    marker_directory: Path | None
    chromosomes: tuple[str, ...]
    imputed_halves: tuple[ImputedHalfInput, ...]
    truth_calls: str | None
    strata_directory: Path
    bubble_split: str
    tandem_repeats: Path
    crosswalk: Path
    crosswalk_columns: tuple[str, str]
    ancestry: Path
    relatedness: Path
    relatedness_columns: tuple[str, str, str]
    diseases: tuple[str, ...]
    traits: tuple[str, ...]
    fold_count: int
    seed: int
    codec: Codec
    export: tuple[str, ...]

    @classmethod
    def read(cls, path: str | Path) -> WorkspaceConfig:
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> WorkspaceConfig:
        missing = sorted(set(CONFIG_KEYS) - set(raw))
        unknown = sorted(set(raw) - set(CONFIG_KEYS))
        if missing or unknown:
            raise ValueError(f"the run config lacks {missing} and has unknown keys {unknown}; every key is required.")
        halves = tuple(
            ImputedHalfInput(label=str(half["label"]), batches=tuple(str(batch) for batch in half["batches"]), batch_path=str(half["batch_path"]))
            for half in raw["imputed_halves"]
        )
        labels = [half.label for half in halves]
        if not halves or len(set(labels)) != len(labels) or LONG_READ_SOURCE in labels or any(not half.batches for half in halves):
            raise ValueError(f"imputed_halves needs distinct labels other than {LONG_READ_SOURCE!r}, each with its batches.")
        diseases = tuple(str(name) for name in raw["diseases"])
        traits = tuple(str(name) for name in raw["traits"])
        for name in diseases:
            if resolve_disease_definition(name).canonical_name != name:
                raise ValueError(f"name the disease {name!r} by its canonical name, which model names and files carry.")
        for name in traits:
            if resolve_measurement_definition(name).canonical_name != name:
                raise ValueError(f"name the trait {name!r} by its canonical name, which model names and files carry.")
        if len(set(diseases + traits)) != len(diseases + traits) or not diseases + traits:
            raise ValueError("diseases and traits must name at least one phenotype, each once.")
        if raw["codec"] not in ("raw", "zstd"):
            raise ValueError("codec must be 'raw' or 'zstd' (STORE.md).")
        chromosomes = tuple(str(chromosome) for chromosome in raw["chromosomes"])
        for chromosome in chromosomes:
            chromosome_number(chromosome)
        if not chromosomes or len(set(chromosomes)) != len(chromosomes):
            raise ValueError("chromosomes must list each store chromosome once.")
        try:
            research_column, sequencing_column = (str(column) for column in raw["crosswalk_columns"])
            first_column, second_column, kinship_column = (str(column) for column in raw["relatedness_columns"])
        except ValueError as error:
            raise ValueError("crosswalk_columns names (research, sequencing); relatedness_columns (first, second, kinship).") from error
        marker_directory = raw["marker_directory"]
        return cls(
            run_directory=Path(raw["run_directory"]),
            marker_directory=None if marker_directory is None else Path(marker_directory),
            chromosomes=chromosomes,
            imputed_halves=halves,
            truth_calls=None if raw["truth_calls"] is None else str(raw["truth_calls"]),
            strata_directory=Path(raw["strata_directory"]),
            bubble_split=str(raw["bubble_split"]),
            tandem_repeats=Path(raw["tandem_repeats"]),
            crosswalk=Path(raw["crosswalk"]),
            crosswalk_columns=(research_column, sequencing_column),
            ancestry=Path(raw["ancestry"]),
            relatedness=Path(raw["relatedness"]),
            relatedness_columns=(first_column, second_column, kinship_column),
            diseases=diseases,
            traits=traits,
            fold_count=int(raw["fold_count"]),
            seed=int(raw["seed"]),
            codec=raw["codec"],
            export=tuple(str(name) for name in raw["export"]),
        )

    def truth_path(self, chromosome: str) -> Path:
        if self.truth_calls is None:
            raise ValueError("this run has no truth calls.")
        return Path(self.truth_calls.format(chromosome=chromosome))

    def half_labels(self) -> tuple[str, ...]:
        """Each store half's label: the imputed halves in config order, then the long-read half if there is one."""
        imputed = tuple(half.label for half in self.imputed_halves)
        return imputed if self.truth_calls is None else (*imputed, LONG_READ_SOURCE)


@dataclass(frozen=True, slots=True)
class WorkspaceBindings:
    """The entry points of the steps that other modules own.

    ``workspace_bindings`` binds the production ones. A test binds doubles of the same signatures; ``fit`` is called
    with a real ``fit_model.FitRequest`` either way, so a double cannot stand in for the fitter's own input.
    """

    bigquery_client: Callable[[], Any]
    calibration_moments: Callable[..., Any]
    concatenate_calibration_moments: Callable[..., Any]
    calibration_pairs: Callable[..., Any]
    fit_measurement_model: Callable[..., Any]
    pooled_measurement_model: Callable[..., Any]
    load_measurement_model: Callable[..., Any]
    fit: Callable[..., Any]
    save_model: Callable[..., None]
    load_model: Callable[..., Any]
    predict: Callable[..., Any]


def workspace_bindings() -> WorkspaceBindings:
    """The production entry points: the measurement model (``measurement_model``) and the fit, model files and
    prediction (``fit_model``, ``artifact``).

    They are imported when the run starts, so a checkout without them fails before any step runs. BigQuery gets
    None, from which all_of_us builds the workspace's own client (GOOGLE_PROJECT).
    """
    measurement_model = importlib.import_module("sv_pgs.measurement_model")
    fit_model = importlib.import_module("sv_pgs.fit_model")
    artifact = importlib.import_module("sv_pgs.artifact")
    return WorkspaceBindings(
        bigquery_client=lambda: None,
        calibration_moments=measurement_model.calibration_moments,
        concatenate_calibration_moments=measurement_model.concatenate_calibration_moments,
        calibration_pairs=measurement_model.CalibrationPairs,
        fit_measurement_model=measurement_model.fit_measurement_model,
        pooled_measurement_model=measurement_model.pooled_measurement_model,
        load_measurement_model=measurement_model.MeasurementModel.load,
        fit=fit_model.fit,
        save_model=artifact.save_model,
        load_model=artifact.load_model,
        predict=artifact.predict,
    )


# ---------------------------------------------------------------------------
# Checkpoints and markers
# ---------------------------------------------------------------------------


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(io.DEFAULT_BUFFER_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_digest(code: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for item in code:
        digest.update(inspect.getsource(item).encode())
    return digest.hexdigest()


def _utc() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _write_json(path: Path, value: Any) -> None:
    """Write JSON atomically: a reader sees the old file or the whole new one."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _mark(config: WorkspaceConfig, step: str, event: str) -> None:
    """An empty marker file named by time, step and event, and never by any value computed from data."""
    if config.marker_directory is None:
        return
    config.marker_directory.mkdir(parents=True, exist_ok=True)
    (config.marker_directory / f"{_utc()}_{step}_{event}").touch()


def _set_aside(directory: Path, key: str) -> None:
    target = directory.with_name(f"{directory.name}.superseded-{key[:16]}")
    if target.exists():
        target = directory.with_name(f"{target.name}-{_utc()}")
    LOGGER.info("workspace pipeline: %s was made from other inputs; kept as %s", directory.name, target.name)
    os.rename(directory, target)


def _remove_own(path: Path, owner: Path) -> None:
    """Remove a file or directory this driver wrote inside ``owner``, refusing any path outside it."""
    if not path.resolve().is_relative_to(owner.resolve()):
        raise ValueError(f"refusing to remove {path}, which is outside {owner}.")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


@dataclass(slots=True)
class _Run:
    config: WorkspaceConfig
    bindings: WorkspaceBindings
    budget: ComputeBudget
    keys: dict[str, str]

    def output(self, step: str) -> Path:
        return self.config.run_directory / step

    def summary(self, step: str) -> dict[str, Any]:
        record: dict[str, Any] = json.loads((self.output(step) / COMPLETE_FILE).read_text(encoding="utf-8"))
        return dict(record["summary"])


@dataclass(frozen=True, slots=True)
class _Step:
    name: str
    upstream: tuple[str, ...]
    inputs: Callable[[WorkspaceConfig], Any]
    code: tuple[Any, ...]
    body: Callable[[_Run, Path], dict[str, Any]]


def _run_step(run: _Run, step: _Step) -> dict[str, Any]:
    key = _digest(
        {
            "step": step.name,
            "inputs": step.inputs(run.config),
            "upstream": {name: run.keys[name] for name in step.upstream},
            "code": _source_digest(step.code),
        }
    )
    root = run.config.run_directory
    final = root / step.name
    if (final / COMPLETE_FILE).is_file():
        record = json.loads((final / COMPLETE_FILE).read_text(encoding="utf-8"))
        if record["key"] == key:
            run.keys[step.name] = key
            _mark(run.config, step.name, "skipped")
            return dict(record["summary"])
        _set_aside(final, record["key"])
    elif final.exists():
        raise RuntimeError(f"{final} exists without {COMPLETE_FILE}, so this driver did not write it; move it away first.")
    partial = root / f"{step.name}.partial"
    if partial.exists():
        written_key = (partial / KEY_FILE).read_text(encoding="utf-8") if (partial / KEY_FILE).is_file() else "unkeyed"
        if written_key != key:
            _set_aside(partial, written_key)
    partial.mkdir(parents=True, exist_ok=True)
    (partial / KEY_FILE).write_text(key, encoding="utf-8")
    _mark(run.config, step.name, "started")
    try:
        summary = step.body(run, partial)
    except BaseException as error:
        _mark(run.config, step.name, f"failed-{type(error).__name__}")
        raise
    _write_json(partial / COMPLETE_FILE, {"key": key, "summary": summary, "finished": _utc()})
    os.rename(partial, final)
    run.keys[step.name] = key
    _mark(run.config, step.name, "completed")
    return summary


# ---------------------------------------------------------------------------
# Step 1: samples, cohort rows and folds
# ---------------------------------------------------------------------------


def _header_samples(path: Path) -> tuple[str, ...]:
    reader = VCF(str(path))
    try:
        return tuple(str(sample) for sample in reader.samples)
    finally:
        reader.close()


def _same_samples(paths: Sequence[Path]) -> tuple[str, ...]:
    """The one sample list of a batch's files, which must be the same on every chromosome."""
    lists = {_header_samples(path) for path in paths}
    if len(lists) != 1:
        raise ValueError(f"{paths[0]} and its other chromosomes list different samples.")
    return lists.pop()


def store_half_samples(config: WorkspaceConfig) -> tuple[HalfSamples, ...]:
    """Each store half's sample manifest: an imputed half's batches' header names in batch order, named by DRAGEN
    sample, then the long-read half's, named by research ID."""
    halves = []
    for half in config.imputed_halves:
        names: list[str] = []
        for batch in half.batches:
            names.extend(_same_samples([half.path(chromosome, batch) for chromosome in config.chromosomes]))
        halves.append(HalfSamples("dragen_sample", tuple(names)))
    if config.truth_calls is not None:
        halves.append(HalfSamples("research_id", _same_samples([config.truth_path(chromosome) for chromosome in config.chromosomes])))
    return tuple(halves)


def _half_research_ids(half: HalfSamples, crosswalk: SampleCrosswalk) -> tuple[str, ...]:
    if half.namespace == "research_id":
        return half.names
    return tuple(research_id.value for research_id in store_research_ids(half, crosswalk))


def _calibration_pairs(halves: Sequence[HalfSamples], crosswalk: SampleCrosswalk) -> list[tuple[str, int, int, int]]:
    """(research ID, imputed half, its column there, its long-read column) for every long-read person with an imputed row."""
    if not halves or halves[-1].namespace != "research_id":
        return []
    sequencing_of = dict(zip(crosswalk.research_ids, crosswalk.sequencing_ids))
    imputed_column = {
        name: (half_index, column)
        for half_index, half in enumerate(halves[:-1])
        for column, name in enumerate(half.names)
    }
    pairs = []
    for truth_column, research_id in enumerate(halves[-1].names):
        sequencing_id = sequencing_of.get(research_id)
        if sequencing_id is not None and sequencing_id in imputed_column:
            half_index, column = imputed_column[sequencing_id]
            pairs.append((research_id, half_index, column, truth_column))
    return pairs


def _samples_inputs(config: WorkspaceConfig) -> Any:
    return {
        "chromosomes": config.chromosomes,
        "halves": [
            (half.label, half.batches, [half.path(chromosome, batch).stat().st_size for chromosome in config.chromosomes for batch in half.batches])
            for half in config.imputed_halves
        ],
        "truth": None if config.truth_calls is None else [config.truth_path(chromosome).stat().st_size for chromosome in config.chromosomes],
        "crosswalk": (_file_digest(config.crosswalk), config.crosswalk_columns),
        "ancestry": _file_digest(config.ancestry),
        "relatedness": (_file_digest(config.relatedness), config.relatedness_columns),
        "folds": (config.fold_count, config.seed),
    }


def _samples_step(run: _Run, directory: Path) -> dict[str, Any]:
    config = run.config
    halves = store_half_samples(config)
    labels = config.half_labels()
    crosswalk = SampleCrosswalk.read(config.crosswalk, *config.crosswalk_columns)
    rows = resolve_cohort_rows(halves, crosswalk, read_kinship_pairs(config.relatedness, *config.relatedness_columns))
    ancestry = read_predicted_ancestry(config.ancestry)
    store_research = [_half_research_ids(half, crosswalk) for half in halves]
    unlabelled = {research_id for ids in store_research for research_id in ids if research_id not in ancestry}
    if unlabelled:
        raise ValueError(f"{reportable_count(len(unlabelled))} store samples have no predicted ancestry.")
    legend = sorted({ancestry[research_id] for ids in store_research for research_id in ids})
    for half_index, ids in enumerate(store_research):
        np.save(directory / f"groups_half{half_index}.npy", np.array([legend.index(ancestry[research_id]) for research_id in ids], dtype=np.int64))
    half_starts = np.concatenate([[0], np.cumsum([len(half.names) for half in halves])]).astype(np.int64)
    research_ids = [research_id.value for research_id in rows.research_ids]
    row_labels = [labels[half] for half in rows.store_half]
    strata = [f"{label}|{ancestry[research_id]}" for label, research_id in zip(row_labels, research_ids)]
    components = kinship_components(
        research_ids,
        [first.value for first, _second, _kinship in rows.kinship_pairs],
        [second.value for _first, second, _kinship in rows.kinship_pairs],
        [kinship for _first, _second, kinship in rows.kinship_pairs],
    )
    folds = kinship_folds(components, strata, config.fold_count, config.seed)
    with (directory / "rows.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("research_id", "genotype_source", "half", "store_column", "ancestry", "fold", "family"))
        for row, research_id in enumerate(research_ids):
            store_column = int(half_starts[rows.store_half[row]]) + rows.store_column[row]
            writer.writerow(
                (research_id, rows.genotype_source[row], row_labels[row], store_column, ancestry[research_id], int(folds[row]), int(components[row]))
            )
    pairs = _calibration_pairs(halves, crosswalk)
    _write_json(
        directory / "samples.json",
        {
            "halves": [{"label": label, "namespace": half.namespace, "names": list(half.names)} for label, half in zip(labels, halves)],
            "groups": legend,
            "calibration_pairs": [list(pair) for pair in pairs],
        },
    )
    return {
        "cohort_rows": len(research_ids),
        "half_samples": [len(half.names) for half in halves],
        "ancestry_groups": legend,
        "calibration_pairs": len(pairs),
        "truth": "long-read half stacked" if config.truth_calls is not None else "absent: no long-read half, no calibration pairs",
    }


@dataclass(frozen=True, slots=True)
class _Samples:
    halves: tuple[HalfSamples, ...]
    labels: tuple[str, ...]
    groups: tuple[I64Array, ...]
    legend: tuple[str, ...]
    pairs: tuple[tuple[str, int, int, int], ...]


def _read_samples(run: _Run) -> _Samples:
    directory = run.output("samples")
    record = json.loads((directory / "samples.json").read_text(encoding="utf-8"))
    halves = tuple(HalfSamples(half["namespace"], tuple(half["names"])) for half in record["halves"])
    return _Samples(
        halves=halves,
        labels=tuple(half["label"] for half in record["halves"]),
        groups=tuple(np.load(directory / f"groups_half{index}.npy") for index in range(len(halves))),
        legend=tuple(record["groups"]),
        pairs=tuple((str(pair[0]), int(pair[1]), int(pair[2]), int(pair[3])) for pair in record["calibration_pairs"]),
    )


# ---------------------------------------------------------------------------
# Step 2: phenotypes
# ---------------------------------------------------------------------------


def _phenotype_inputs(config: WorkspaceConfig) -> Any:
    return {
        "diseases": [(name, phenotype_fingerprint(resolve_disease_definition(name))) for name in config.diseases],
        "traits": [(name, phenotype_fingerprint(resolve_measurement_definition(name))) for name in config.traits],
        "cdr": os.environ.get("WORKSPACE_CDR"),
    }


def _phenotypes_step(run: _Run, directory: Path) -> dict[str, Any]:
    client = run.bindings.bigquery_client()
    for disease in run.config.diseases:
        prepare_all_of_us_disease_sample_table(disease, directory / f"{disease}.tsv", client=client)
    for trait in run.config.traits:
        prepare_all_of_us_measurement_sample_table(trait, directory / f"{trait}.tsv", client=client)
    return {"diseases": list(run.config.diseases), "traits": list(run.config.traits)}


# ---------------------------------------------------------------------------
# Step 3: cohort
# ---------------------------------------------------------------------------


def _read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def trait_table(rows: Sequence[Mapping[str, str]], covariate_columns: Sequence[str]) -> TraitTable:
    """A sample table's targets and its own covariates (its metadata's ``covariate_columns``), keyed by person_id.

    Every covariate but sex at birth is numeric. Sex at birth is written one-hot; a person's level is
    the concept ID whose column is 1, or UNRECORDED_SEX where none is.
    """
    prefix = SEX_COLUMN + "_"
    numeric = [name for name in covariate_columns if name != SEX_COLUMN]
    levels: dict[str, str] = {}
    for row in rows:
        ones = [name[len(prefix):] for name, value in row.items() if name.startswith(prefix) and value == "1"]
        if len(ones) > 1:
            raise ValueError(f"person {row['person_id']} has two sex-at-birth levels.")
        levels[row["person_id"]] = ones[0] if ones else UNRECORDED_SEX
    return TraitTable(
        targets={row["person_id"]: float(row["target"]) for row in rows},
        numeric={name: {row["person_id"]: float(row[name]) for row in rows} for name in numeric},
        categorical={SEX_COLUMN: levels} if SEX_COLUMN in covariate_columns else {},
    )


def _cohort_step(run: _Run, directory: Path) -> dict[str, Any]:
    config = run.config
    kept = _read_table(run.output("samples") / "rows.tsv")
    trait_names = (*config.diseases, *config.traits)
    tables = {}
    for name in trait_names:
        metadata = json.loads((run.output("phenotypes") / f"{name}.tsv.metadata.json").read_text(encoding="utf-8"))
        tables[name] = trait_table(_read_table(run.output("phenotypes") / f"{name}.tsv"), metadata["covariate_columns"])
    research_ids = [ResearchId(row["research_id"]) for row in kept]
    sources = [row["genotype_source"] for row in kept]
    cohort = build_cohort(
        research_ids,
        ancestry=AncestryPcs.read(config.ancestry),
        pipeline_half=pipeline_half_levels([row["half"] for row in kept], sources),
        genotype_source=sources,
        traits=tables,
    )
    folds = np.array([int(row["fold"]) for row in kept], dtype=np.int64)
    # The kinship components as family clusters 0..C-1, the resampling unit of the held-out tests.
    families = np.unique(np.array([int(row["family"]) for row in kept], dtype=np.int64), return_inverse=True)[1].astype(np.int64)
    model_traits = np.repeat(np.arange(len(trait_names)), config.fold_count)
    model_folds = np.tile(np.arange(config.fold_count), len(trait_names))
    observed = cohort.observed[:, model_traits]
    training = observed & (folds[:, None] != model_folds[None, :])
    held_out = observed & (folds[:, None] == model_folds[None, :])
    np.savez(
        directory / "cohort.npz",
        research_ids=np.array([research_id.value for research_id in research_ids]),
        store_columns=np.array([int(row["store_column"]) for row in kept], dtype=np.int64),
        covariates=cohort.covariates,
        covariate_columns=cohort.covariate_columns,
        targets=cohort.targets,
        folds=folds,
        families=families,
        training=training,
        held_out=held_out,
        model_traits=model_traits,
    )
    trait_types = [TraitType.BINARY.value] * len(config.diseases) + [TraitType.QUANTITATIVE.value] * len(config.traits)
    _write_json(
        directory / "cohort.json",
        {
            "covariate_names": list(cohort.covariate_names),
            "trait_names": list(trait_names),
            "trait_types": trait_types,
            "model_names": [f"{trait_names[trait]}/fold{fold}" for trait, fold in zip(model_traits.tolist(), model_folds.tolist())],
            "ancestry": [row["ancestry"] for row in kept],
            "half": [row["half"] for row in kept],
        },
    )
    return {
        "rows": len(kept),
        "fit_rows": int(training.any(axis=1).sum()),
        "covariates": {
            trait: [name for name, used in zip(cohort.covariate_names, cohort.covariate_columns[index]) if used]
            for index, trait in enumerate(trait_names)
        },
        "models": int(model_traits.shape[0]),
    }


@dataclass(frozen=True, slots=True)
class _Cohort:
    research_ids: tuple[str, ...]
    store_columns: I64Array
    covariates: F64Array
    covariate_names: tuple[str, ...]
    covariate_columns: BoolArray
    targets: F64Array
    folds: I64Array
    training: BoolArray
    held_out: BoolArray
    model_traits: I64Array
    model_names: tuple[str, ...]
    trait_names: tuple[str, ...]
    trait_types: tuple[TraitType, ...]
    ancestry: tuple[str, ...]
    half: tuple[str, ...]

    @property
    def fit_rows(self) -> BoolArray:
        return np.asarray(self.training.any(axis=1))


def _read_cohort(run: _Run) -> _Cohort:
    directory = run.output("cohort")
    arrays = np.load(directory / "cohort.npz")
    record = json.loads((directory / "cohort.json").read_text(encoding="utf-8"))
    return _Cohort(
        research_ids=tuple(str(value) for value in arrays["research_ids"]),
        store_columns=arrays["store_columns"],
        covariates=arrays["covariates"],
        covariate_names=tuple(record["covariate_names"]),
        covariate_columns=arrays["covariate_columns"],
        targets=arrays["targets"],
        folds=arrays["folds"],
        training=arrays["training"],
        held_out=arrays["held_out"],
        model_traits=arrays["model_traits"],
        model_names=tuple(record["model_names"]),
        trait_names=tuple(record["trait_names"]),
        trait_types=tuple(TraitType(value) for value in record["trait_types"]),
        ancestry=tuple(record["ancestry"]),
        half=tuple(record["half"]),
    )


# ---------------------------------------------------------------------------
# Step 4: the store
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StrataSites:
    """One chromosome's strata sidecar (v2) rows, in popped-record order, and its md5s of the popped records."""

    positions: I64Array
    identifiers: tuple[str, ...]
    refalt_digests: NDArray
    ref_lengths: I64Array
    alt_lengths: I64Array
    n_paths: I64Array
    n_paths_total: I64Array
    complexity: NDArray
    complexity_legend: tuple[str, ...] | None
    sites_md5: str
    ids_md5: str


def read_strata_sites(directory: Path, chromosome: str) -> StrataSites:
    """The STRATA_COLUMNS of ``chrK.strata.tsv.gz``, once ``_done/chrK.json`` marks the contig complete.

    ``cx`` stays numeric when it is, and is otherwise coded by its sorted levels.
    """
    done = directory / "_done" / f"{chromosome}.json"
    if not done.is_file():
        raise ValueError(f"the strata sidecar of {chromosome} is not complete: {done} does not exist.")
    manifest = json.loads(done.read_text(encoding="utf-8"))
    path = directory / f"{chromosome}.strata.tsv.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n")
    if not header.startswith("#"):
        raise ValueError(f"{path}: the header line must start with '#'.")
    names = header[1:].split("\t")
    missing = [column for column in STRATA_COLUMNS if column not in names]
    if missing:
        raise ValueError(f"{path} lacks the columns {missing}.")
    table = pd.read_csv(
        path, sep="\t", skiprows=1, header=None, names=names, usecols=list(STRATA_COLUMNS), dtype={"id": str, "refalt_md5": str}, keep_default_na=False
    )
    index = table["idx"].to_numpy(dtype=np.int64)
    if index.size and np.any(np.diff(index) != 1):
        raise ValueError(f"{path}: idx must count the popped records in file order.")
    complexity = table["cx"]
    numeric = pd.api.types.is_numeric_dtype(complexity)
    legend = None if numeric else tuple(sorted(set(complexity.astype(str))))
    return StrataSites(
        positions=table["pos"].to_numpy(dtype=np.int64),
        identifiers=tuple(table["id"].astype(str)),
        refalt_digests=np.array([int.from_bytes(bytes.fromhex(value), "big") for value in table["refalt_md5"]], dtype=np.uint64),
        ref_lengths=table["ref_len"].to_numpy(dtype=np.int64),
        alt_lengths=table["alt_len"].to_numpy(dtype=np.int64),
        n_paths=table["n_paths"].to_numpy(dtype=np.int64),
        n_paths_total=table["n_paths_total"].to_numpy(dtype=np.int64),
        complexity=complexity.to_numpy(dtype=np.float64) if legend is None else np.array([legend.index(value) for value in complexity.astype(str)], dtype=np.int32),
        complexity_legend=legend,
        sites_md5=str(manifest["sites_md5"]),
        ids_md5=str(manifest["ids_md5"]),
    )


@dataclass(frozen=True, slots=True)
class PoppedSites:
    """REF, ALT, genetic position (INFO/CM) and |INFO/SVLEN| (NaN where absent) of every popped record."""

    refs: tuple[str, ...]
    alts: tuple[str, ...]
    genetic_position_cm: F64Array
    sv_lengths: F64Array


def read_popped_sites(path: Path, strata: StrataSites) -> PoppedSites:
    """One popped batch file's site fields, checked against the strata sidecar.

    Each record is checked against its sidecar row (POS, INFO/ID, REF/ALT md5, allele lengths).
    The whole file is then checked against the sidecar's sites md5 (the md5 of the records'
    "CHROM\\tPOS\\tREF\\tALT\\n" lines) and ids md5 (the same over "INFO/ID\\n").
    """
    sites = hashlib.md5()
    ids = hashlib.md5()
    refs: list[str] = []
    alts: list[str] = []
    genetic: list[float] = []
    lengths: list[float] = []
    count = strata.positions.shape[0]
    reader = VCF(str(path))
    try:
        for row, record in enumerate(reader):
            if row >= count:
                raise ValueError(f"{path}: more records than the strata sidecar's {count}.")
            alt = ",".join(record.ALT)
            identifier = str(record.INFO.get("ID"))
            sites.update(f"{record.CHROM}\t{record.POS}\t{record.REF}\t{alt}\n".encode())
            ids.update(f"{identifier}\n".encode())
            if (
                record.POS != int(strata.positions[row])
                or identifier != strata.identifiers[row]
                or refalt_digest(record.REF, alt) != int(strata.refalt_digests[row])
                or (len(record.REF), len(alt)) != (int(strata.ref_lengths[row]), int(strata.alt_lengths[row]))
            ):
                raise ValueError(f"{path} record {row}: POS, INFO/ID, REF/ALT or allele lengths differ from the strata sidecar.")
            centimorgans = record.INFO.get("CM")
            if centimorgans is None:
                raise ValueError(f"{path} record {row}: no INFO/CM, so the record has no genetic position.")
            sv_length = record.INFO.get("SVLEN")
            refs.append(record.REF)
            alts.append(alt)
            genetic.append(float(centimorgans))
            lengths.append(np.nan if sv_length is None else float(np.abs(np.atleast_1d(sv_length)).max()))
    finally:
        reader.close()
    if len(refs) != count:
        raise ValueError(f"{path}: {len(refs)} records, not the strata sidecar's {count}.")
    if sites.hexdigest() != strata.sites_md5 or ids.hexdigest() != strata.ids_md5:
        raise ValueError(f"{path}: the popped records' sites or ids md5 differs from the strata sidecar's.")
    centimorgan_array = np.array(genetic, dtype=np.float64)
    if not np.all(np.isfinite(centimorgan_array)) or np.any(np.diff(centimorgan_array) < 0):
        raise ValueError(f"{path}: INFO/CM must be finite and non-decreasing along the chromosome.")
    return PoppedSites(tuple(refs), tuple(alts), centimorgan_array, np.array(lengths, dtype=np.float64))


def read_bubble_paths(path: Path) -> list[list[frozenset[str]]]:
    """Each bubble's paths, in file order, as the sets of atomic IDs they carry.

    bubble.split holds one biallelic record per path, with a bubble's paths as consecutive
    records at its POS. INFO/ID lists the path's atomic IDs separated by ':', with ',' read as ':'.
    """
    bubbles: list[list[frozenset[str]]] = []
    previous: int | None = None
    reader = VCF(str(path))
    try:
        for record in reader:
            carried = frozenset(part for part in str(record.INFO.get("ID")).replace(",", ":").split(":") if part)
            if not carried:
                raise ValueError(f"{path}: a path record at {record.POS} carries no atomic ID.")
            if record.POS != previous:
                bubbles.append([])
            bubbles[-1].append(carried)
            previous = record.POS
    finally:
        reader.close()
    return bubbles


@dataclass(frozen=True, slots=True)
class PathCounts:
    """Each record's bubble (its index in bubble.split, -1 for a single-path record), K and m (STORE.md, background)."""

    bubble: I64Array
    kept: I64Array
    carrying: I64Array


def path_counts(strata: StrataSites, bubbles: Sequence[Sequence[frozenset[str]]]) -> PathCounts:
    """K = min(MAXIMUM_KEPT_PATHS, N_PATHS_TOTAL) and m = the record's paths among its bubble's first K, per record.

    bubble.split must agree with the sidecar on every multi-path record: one bubble carries it, and
    that bubble has the sidecar's N_PATHS_TOTAL paths, the sidecar's N_PATHS of which carry the
    record's ID. A single-path record no bubble lists has K = m = 1.
    """
    carriers: dict[str, list[tuple[int, int]]] = {}
    for bubble_index, paths in enumerate(bubbles):
        for path_index, carried in enumerate(paths):
            for identifier in carried:
                carriers.setdefault(identifier, []).append((bubble_index, path_index))
    count = len(strata.identifiers)
    bubble = np.full(count, -1, dtype=np.int64)
    kept = np.ones(count, dtype=np.int64)
    carrying = np.ones(count, dtype=np.int64)
    for row, identifier in enumerate(strata.identifiers):
        total, carried_by = int(strata.n_paths_total[row]), int(strata.n_paths[row])
        hits = carriers.get(identifier, [])
        if not hits:
            if (total, carried_by) != (1, 1):
                raise ValueError(f"{identifier}: the sidecar gives {carried_by} of {total} paths, but no bubble.split path carries it.")
            continue
        owners = {owner for owner, _path in hits}
        if len(owners) != 1:
            raise ValueError(f"{identifier}: paths of {len(owners)} bubbles carry it.")
        owner = owners.pop()
        if (len(bubbles[owner]), len(hits)) != (total, carried_by):
            raise ValueError(
                f"{identifier}: bubble.split has {len(hits)} of {len(bubbles[owner])} paths carrying it; the sidecar says {carried_by} of {total}."
            )
        kept[row] = min(MAXIMUM_KEPT_PATHS, total)
        carrying[row] = sum(1 for _owner, path in hits if path < kept[row])
        bubble[row] = owner
    return PathCounts(bubble=bubble, kept=kept, carrying=carrying)


def read_tandem_repeats(path: Path) -> dict[str, tuple[I64Array, I64Array]]:
    """The repeat BED (GIAB AllTandemRepeatsandHomopolymers_slop5; 0-based half-open) as (starts, ends) per chromosome."""
    table = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2], names=["chromosome", "start", "end"], dtype={"chromosome": str}, comment="#")
    intervals = {}
    for chromosome, rows in table.groupby("chromosome", sort=False):
        ordered = rows.sort_values("start", kind="stable")
        intervals[str(chromosome)] = (ordered["start"].to_numpy(dtype=np.int64), ordered["end"].to_numpy(dtype=np.int64))
    return intervals


def store_variant_classes(sites: PoppedSites, positions: NDArray, record_locus: NDArray) -> tuple[U8Array, F64Array]:
    """Each record's store class code (an index of VARIANT_CLASSES) and length, by STORE.md's first-match rules.

    A record that is not an SNV and whose core overlaps a tandem-repeat locus is str_vntr_repeat.
    Every other record is typed from its alleles and |SVLEN| by variant_typing.
    """
    code = {variant_class: index for index, variant_class in enumerate(VARIANT_CLASSES)}
    classes = np.empty(len(sites.refs), dtype=np.uint8)
    lengths = np.empty(len(sites.refs), dtype=np.float64)
    for row, (ref, alt) in enumerate(zip(sites.refs, sites.alts, strict=True)):
        sv_length = float(sites.sv_lengths[row])
        variant_class, length = variant_class_and_length(
            pos=int(positions[row]), ref=ref, alt=alt, svtype=None, svlen=None if np.isnan(sv_length) else sv_length, info_end=None
        )
        if variant_class != VariantClass.SNV and record_locus[row] != NO_LOCUS:
            variant_class = VariantClass.STR_VNTR_REPEAT
        classes[row] = code[variant_class]
        lengths[row] = length
    return classes, lengths


@dataclass(frozen=True, slots=True)
class _DecodeTask:
    half: int
    batch: int
    path: Path
    called: bool
    groups: I64Array
    codes_path: Path
    statistics_path: Path


_DECODER: dict[str, Any] = {}


def _start_decoder(expected: ExpectedSites, group_count: int, budget: ComputeBudget) -> None:
    _DECODER.update(expected=expected, group_count=group_count, budget=budget)


def _decode(task: _DecodeTask) -> None:
    """Decode one batch file and write its codes, then its statistics, whose presence marks the batch done."""
    decode = decode_called_batch if task.called else decode_batch
    decoded = decode(task.path, _DECODER["expected"], task.groups, _DECODER["group_count"], task.codes_path, _DECODER["budget"])
    temporary = task.statistics_path.with_name(task.statistics_path.stem + ".partial.npz")
    np.savez(
        temporary,
        group_sums=decoded.group_sums,
        group_counts=decoded.group_counts,
        zeroed=decoded.zeroed,
        unmatched_low=decoded.unmatched_low,
        no_calls=decoded.no_calls,
        sample_ids=np.array(decoded.sample_ids),
        reported_info=decoded.reported_info,
    )
    os.replace(temporary, task.statistics_path)


def _decode_all(tasks: Sequence[_DecodeTask], expected: ExpectedSites, group_count: int, budget: ComputeBudget) -> None:
    """Decode every batch whose statistics are not written yet, one process per CPU thread, each with its share of host memory."""
    pending = [task for task in tasks if not task.statistics_path.is_file()]
    if not pending:
        return
    workers = min(budget.cpu_threads, len(pending))
    worker_budget = dataclasses.replace(
        budget, device_kind="cpu", device_ids=(), device_names=(), device_bytes=(), device_compute_capabilities=(),
        host_bytes=budget.host_bytes // workers, cpu_threads=1,
    )
    if workers == 1:
        _start_decoder(expected, group_count, worker_budget)
        for task in pending:
            _decode(task)
        return
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_start_decoder, initargs=(expected, group_count, worker_budget)
    ) as pool:
        list(pool.map(_decode, pending))


def _decode_tasks(config: WorkspaceConfig, directory: Path, chromosome: str, samples: _Samples) -> list[_DecodeTask]:
    batches = directory / "batches" / chromosome
    batches.mkdir(parents=True, exist_ok=True)
    tasks = []
    for half_index, half in enumerate(config.imputed_halves):
        start = 0
        for batch_index, batch in enumerate(half.batches):
            path = half.path(chromosome, batch)
            width = len(_header_samples(path))
            tasks.append(
                _DecodeTask(
                    half_index, batch_index, path, False, samples.groups[half_index][start : start + width],
                    batches / f"half{half_index}.batch{batch_index}.npy", batches / f"half{half_index}.batch{batch_index}.statistics.npz",
                )
            )
            start += width
    if config.truth_calls is not None:
        half_index = len(config.imputed_halves)
        tasks.append(
            _DecodeTask(
                half_index, 0, config.truth_path(chromosome), True, samples.groups[half_index],
                batches / f"half{half_index}.batch0.npy", batches / f"half{half_index}.batch0.statistics.npz",
            )
        )
    return tasks


def _clear_chromosome(root: Path, chromosome: str, half_count: int, owner: Path) -> None:
    """Remove a chromosome's store arrays left by an interrupted attempt, so it is written again from the start."""
    for half_index in range(half_count):
        _remove_own(dosage_array_directory(root, half_index, chromosome), owner)
        _remove_own(statistic_column_directory(root, half_index, chromosome, "sum_code").parent, owner)
    _remove_own(root / "variants" / chromosome, owner)
    _remove_own(root / "loci" / chromosome, owner)


def _write_loci(root: Path, chromosome: str, loci: TrLoci) -> None:
    """The chromosome's TR loci table (STORE.md, "Loci")."""
    for name, values in (
        ("tr_start", loci.starts),
        ("tr_end", loci.ends),
        ("n_intervals", loci.interval_counts),
        ("n_records", loci.record_counts),
        ("n_dlen_nonzero", loci.length_changing_record_counts),
    ):
        write_column(root / "loci" / chromosome / name, np.asarray(values, dtype=np.int64))


def _gather_calibration(tasks: Sequence[_DecodeTask], samples: _Samples, directory: Path, chromosome: str) -> None:
    """Every calibration pair's stored imputed codes and long-read codes, the latter before any no-call fill
    (MISSING_CODE marks a no-call), read from the decoded batches in one pass over each batch."""
    if not samples.pairs:
        return
    by_half: dict[int, list[_DecodeTask]] = {}
    for task in tasks:
        by_half.setdefault(task.half, []).append(task)
    record_count = int(np.load(tasks[0].codes_path, mmap_mode="r").shape[0])
    dosage = np.empty((record_count, len(samples.pairs)), dtype=np.uint8)
    truth = np.empty((record_count, len(samples.pairs)), dtype=np.uint8)
    truth_codes = np.load(by_half[len(samples.halves) - 1][0].codes_path, mmap_mode="r")
    truth[:] = truth_codes[:, [pair[3] for pair in samples.pairs]]
    for half_index, half_tasks in by_half.items():
        if half_index == len(samples.halves) - 1:
            continue
        start = 0
        for task in half_tasks:
            codes = np.load(task.codes_path, mmap_mode="r")
            wanted = [(position, pair[2] - start) for position, pair in enumerate(samples.pairs) if pair[1] == half_index and start <= pair[2] < start + codes.shape[1]]
            if wanted:
                dosage[:, [position for position, _column in wanted]] = codes[:, [column for _position, column in wanted]]
            start += codes.shape[1]
    np.savez(directory / "calibration" / f"{chromosome}.npz", dosage=dosage, truth=truth)


def _convert_chromosome(run: _Run, directory: Path, root: Path, chromosome: str, samples: _Samples, repeats: Mapping[str, tuple[I64Array, I64Array]]) -> dict[str, Any]:
    config = run.config
    strata = read_strata_sites(config.strata_directory, chromosome)
    first_half = config.imputed_halves[0]
    sites = read_popped_sites(first_half.path(chromosome, first_half.batches[0]), strata)
    counts = path_counts(strata, read_bubble_paths(Path(config.bubble_split.format(chromosome=chromosome))))
    expected = ExpectedSites(
        positions=strata.positions,
        refalt_digests=strata.refalt_digests,
        identifiers=strata.identifiers,
        kept_paths=counts.kept,
        carrying_paths=counts.carrying,
    )
    core_starts, core_ends = core_spans(strata.positions, sites.refs, sites.alts)
    empty = np.zeros(0, dtype=np.int64)
    interval_starts, interval_ends = repeats.get(chromosome, (empty, empty))
    loci = tr_loci(interval_starts, interval_ends, core_starts, core_ends, strata.alt_lengths - strata.ref_lengths)
    classes, lengths = store_variant_classes(sites, strata.positions, loci.record_locus)
    locus_groups = np.where(loci.record_locus == NO_LOCUS, -1, loci.record_locus.astype(np.int64))
    group_first = unbreakable_group_first(counts.bubble, strata.positions, locus_groups)

    tasks = _decode_tasks(config, directory, chromosome, samples)
    _decode_all(tasks, expected, len(samples.legend), run.budget)
    statistics = [np.load(task.statistics_path) for task in tasks]
    for half_index, half in enumerate(samples.halves):
        decoded_names = tuple(name for task, values in zip(tasks, statistics) if task.half == half_index for name in values["sample_ids"].tolist())
        if decoded_names != half.names:
            raise ValueError(f"{chromosome}: half {half_index}'s batches list other samples than its manifest.")
    fill = no_call_fill(sum(values["group_sums"] for values in statistics), sum(values["group_counts"] for values in statistics))
    np.save(directory / "fill" / f"{chromosome}.npy", fill)
    # The imputation's reported r^2 over every imputed sample: the batches' INFO weighted by their sample counts.
    imputed = [(len(values["sample_ids"]), values["reported_info"]) for task, values in zip(tasks, statistics) if not task.called]
    np.save(directory / "reported" / f"{chromosome}.npy", sum(width * info for width, info in imputed) / sum(width for width, _info in imputed))
    record_count = strata.positions.shape[0]
    sums = np.zeros(record_count, dtype=np.int64)
    squares = np.zeros(record_count, dtype=np.int64)
    for half_index in range(len(samples.halves)):
        half_sums, half_squares = assemble_half(
            root, half_index, chromosome, [np.load(task.codes_path, mmap_mode="r") for task in tasks if task.half == half_index],
            samples.groups[half_index], None, fill, codec=config.codec, budget=run.budget,
        )
        sums += half_sums
        squares += half_squares
    _gather_calibration(tasks, samples, directory, chromosome)
    encoded_ids = [identifier.encode() for identifier in strata.identifiers]
    complexity_legends = {} if strata.complexity_legend is None else {"cx": strata.complexity_legend}
    table = VariantTable(
        chromosome=np.full(record_count, chromosome_number(chromosome), dtype=np.int8),
        position=strata.positions,
        genetic_position_cm=sites.genetic_position_cm,
        ref_length=strata.ref_lengths.astype(np.int32),
        alt_length=strata.alt_lengths.astype(np.int32),
        variant_class=classes,
        # Every record the driver stores is an ALT count from the popped BCFs: its value is code / 127.
        codes_per_unit=allele_count_decode(record_count)[0],
        value_origin=allele_count_decode(record_count)[1],
        group_first=group_first,
        sum_code=sums.astype(np.uint64),
        sum_code2=squares.astype(np.uint64),
        annotations={
            "n_paths": strata.n_paths.astype(np.float64),
            "n_paths_total": strata.n_paths_total.astype(np.float64),
            "cx": strata.complexity,
            "tr_locus": loci.record_locus.astype(np.uint32),
            "sv_length": lengths,
        },
        annotation_legends=complexity_legends,
        id_bytes=np.frombuffer(b"".join(encoded_ids), dtype=np.uint8),
        id_offsets=np.concatenate([[0], np.cumsum([len(identifier) for identifier in encoded_ids])]).astype(np.int64),
    )
    write_variant_columns(root, chromosome, table, slice(0, record_count))
    _write_loci(root, chromosome, loci)
    return {
        "records": int(record_count),
        "sites_md5": sites_md5(strata.positions, strata.ref_lengths, strata.alt_lengths),
        "background_zeroed": int(sum(int(values["zeroed"].sum()) for values in statistics)),
        "no_calls": int(sum(int(values["no_calls"].sum()) for values in statistics)),
    }


def _store_inputs(config: WorkspaceConfig) -> Any:
    return {
        "chromosomes": config.chromosomes,
        "halves": [
            (half.label, half.batches, [half.path(chromosome, batch).stat().st_size for chromosome in config.chromosomes for batch in half.batches])
            for half in config.imputed_halves
        ],
        "truth": None if config.truth_calls is None else [config.truth_path(chromosome).stat().st_size for chromosome in config.chromosomes],
        "strata": [_file_digest(config.strata_directory / "_done" / f"{chromosome}.json") for chromosome in config.chromosomes],
        "bubble_split": [Path(config.bubble_split.format(chromosome=chromosome)).stat().st_size for chromosome in config.chromosomes],
        "tandem_repeats": _file_digest(config.tandem_repeats),
        "codec": config.codec,
    }


def _store_step(run: _Run, directory: Path) -> dict[str, Any]:
    config = run.config
    samples = _read_samples(run)
    root = directory / "store"
    for subdirectory in ("chromosomes", "fill", "calibration", "reported"):
        (directory / subdirectory).mkdir(exist_ok=True)
    repeats = read_tandem_repeats(config.tandem_repeats)
    facts = []
    for chromosome in config.chromosomes:
        done = directory / "chromosomes" / f"{chromosome}.json"
        if not done.is_file():
            _clear_chromosome(root, chromosome, len(samples.halves), directory)
            _write_json(done, _convert_chromosome(run, directory, root, chromosome, samples, repeats))
            _remove_own(directory / "batches" / chromosome, directory)
        facts.append(json.loads(done.read_text(encoding="utf-8")))
    measurements = ["imputed_dosage"] * len(config.imputed_halves) + (["long_read_calls"] if config.truth_calls is not None else [])
    write_store_manifest(
        root,
        chromosomes=config.chromosomes,
        record_counts=[fact["records"] for fact in facts],
        chromosome_sites_md5=[fact["sites_md5"] for fact in facts],
        half_sample_counts=[len(half.names) for half in samples.halves],
        half_samples=samples.halves,
        half_measurements=measurements,
        gates={"G2_lockstep": "PASS", "G4_format": "PASS", "strata_sites_md5": "PASS", "strata_ids_md5": "PASS"},
        recalibrated=False,
    )
    with DosageStore.open(root) as store:
        if store.half_samples() != samples.halves:
            raise ValueError("the written store's sample manifests differ from the samples step's.")
    return {
        "records": int(sum(fact["records"] for fact in facts)),
        "halves": measurements,
        "recalibrated": False,
        "background_zeroed": int(sum(fact["background_zeroed"] for fact in facts)),
        "no_calls": int(sum(fact["no_calls"] for fact in facts)),
    }


# ---------------------------------------------------------------------------
# Step 5: the measurement model and D*
# ---------------------------------------------------------------------------


def _group_code_moments(store: DosageStore, columns: I64Array, column_groups: I64Array, group_count: int, budget: ComputeBudget) -> tuple[I64Array, I64Array, I64Array]:
    """Per record and group, the count, sum and sum of squares of the codes of ``columns`` (exact, in int64), from one read.

    Half the host budget goes to the store's read-ahead ring, the other half to each block's
    int64 copies of one group's codes and their squares.
    """
    record_count = store.n_variants
    sums = np.zeros((record_count, group_count), dtype=np.int64)
    squares = np.zeros((record_count, group_count), dtype=np.int64)
    counts = np.bincount(column_groups, minlength=group_count).astype(np.int64)
    rows = max(1, int(budget.host_bytes // 2 // (columns.shape[0] * (1 + 2 * _INT64_BYTES))))
    ranges = [(start, min(start + rows, record_count)) for start in range(0, record_count, rows)]
    reader_budget = dataclasses.replace(budget, host_bytes=budget.host_bytes // 2)
    members = [column_groups == group for group in range(group_count)]
    for start, stop, codes in store.iter_codes(ranges, columns, reader_budget):
        for group, member in enumerate(members):
            values = codes[:, member].astype(np.int64)
            sums[start:stop, group] = values.sum(axis=1)
            squares[start:stop, group] = (values * values).sum(axis=1)
    return counts, sums, squares


def _group_calibration(run: _Run, directory: Path, chromosomes: Sequence[str], pair_positions: Sequence[int]) -> Any:
    """The calibration moments of one group's pairs over every record, in record chunks that fit the host budget.

    Dosage and truth enter in dosage units, the truth NaN at a no-call.
    """
    rows = max(1, int(run.budget.host_bytes // 2 // (max(len(pair_positions), 1) * 2 * _FLOAT64_BYTES)))
    parts = []
    for chromosome in chromosomes:
        arrays = np.load(directory / "calibration" / f"{chromosome}.npz")
        dosage_codes, truth_codes = arrays["dosage"], arrays["truth"]
        for start in range(0, dosage_codes.shape[0], rows):
            dosage = dosage_codes[start : start + rows][:, pair_positions].astype(np.float64) / CODES_PER_DOSAGE
            truth_block = truth_codes[start : start + rows][:, pair_positions]
            truth = np.where(truth_block == MISSING_CODE, np.nan, truth_block.astype(np.float64) / CODES_PER_DOSAGE)
            parts.append(run.bindings.calibration_moments(dosage, truth))
    return run.bindings.concatenate_calibration_moments(parts)


class _StoredHalfCodes:
    """One half's stored codes of one chromosome, read by row slices as assemble_half reads a batch's codes."""

    def __init__(self, store: DosageStore, start: int, record_count: int) -> None:
        self._store = store
        self._start = start
        self.shape = (record_count, store.n_samples)

    def __getitem__(self, rows: slice) -> U8Array:
        return self._store.read_codes(self._start + rows.start, self._start + rows.stop)


def rewrite_recalibrated_store(
    source_root: Path, target_root: Path, scales: F64Array, groups: Sequence[I64Array], fill_directory: Path, codec: Codec, budget: ComputeBudget
) -> None:
    """The store again with every imputed half's codes recalibrated to D* (scales[record, group], STORE.md).

    D* is computed from the stored codes. A long-read half is written through unchanged. The
    variant columns and loci are copied, and the MANIFEST records the recalibration.
    """
    manifest = read_manifest(source_root)
    with DosageStore.open(source_root) as source:
        halves = source.half_samples()
        measurements = list(source.manifest_attributes["half_measurements"])
        gates = dict(source.manifest_attributes["gates"])
        chromosomes, starts = source.chromosomes, source.chromosome_starts
    for half_index, measurement in enumerate(measurements):
        # linear_recalibration takes a kappa for each group up to the half's highest one.
        half_groups = int(groups[half_index].max()) + 1
        with DosageStore.open(source_root, half_indices=[half_index]) as half:
            for position, chromosome in enumerate(chromosomes):
                start, stop = int(starts[position]), int(starts[position + 1])
                assemble_half(
                    target_root, half_index, chromosome, [_StoredHalfCodes(half, start, stop - start)], groups[half_index],
                    scales[start:stop, :half_groups] if measurement == "imputed_dosage" else None,
                    np.load(fill_directory / f"{chromosome}.npy"), codec=codec, budget=budget,
                )
    shutil.copytree(source_root / "variants", target_root / "variants")
    shutil.copytree(source_root / "loci", target_root / "loci")
    write_store_manifest(
        target_root,
        chromosomes=chromosomes,
        record_counts=manifest["record_counts"],
        chromosome_sites_md5=[manifest["sites_md5"][chromosome] for chromosome in chromosomes],
        half_sample_counts=manifest["half_sample_counts"],
        half_samples=halves,
        half_measurements=measurements,
        gates={**gates, "recalibration": "APPLIED"},
        recalibrated=True,
    )


def _measurement_step(run: _Run, directory: Path) -> dict[str, Any]:
    samples = _read_samples(run)
    cohort = _read_cohort(run)
    store_directory = run.output("store")
    source_root = store_directory / "store"
    group_count = len(samples.legend)
    column_groups = np.concatenate(samples.groups)
    fit_columns = np.sort(cohort.store_columns[cohort.fit_rows])
    with DosageStore.open(source_root) as store:
        counts, sums, squares = _group_code_moments(store, fit_columns, column_groups[fit_columns], group_count, run.budget)
        strata = np.asarray(VARIANT_CLASS_LEGEND)[store.variant_table.variant_class]
        chromosomes = store.chromosomes
    pooled = sums.sum(axis=1) / fit_columns.shape[0], squares.sum(axis=1) / fit_columns.shape[0]
    reported = np.concatenate([np.load(store_directory / "reported" / f"{chromosome}.npy") for chromosome in chromosomes])
    pair_groups = [int(samples.groups[pair[1]][pair[2]]) for pair in samples.pairs]
    results = []
    certificates = {}
    calibrated = []
    group_means, group_variances = [], []
    for group, label in enumerate(samples.legend):
        # A group with no fitted rows takes the pooled moments; its scales then serve only its unfitted store samples.
        mean, second = (sums[:, group] / counts[group], squares[:, group] / counts[group]) if counts[group] else pooled
        variance = np.maximum(second - mean * mean, 0.0) / CODES_PER_DOSAGE**2
        group_means.append(mean / CODES_PER_DOSAGE)
        group_variances.append(variance)
        members = [position for position, pair_group in enumerate(pair_groups) if pair_group == group]
        calibration = None
        if members:
            calibrated.append(label)
            calibration = run.bindings.calibration_pairs(
                sample_ids=tuple(ResearchId(samples.pairs[position][0]) for position in members),
                moments=_group_calibration(run, store_directory, chromosomes, members),
            )
        result = run.bindings.fit_measurement_model(calibration, variance, strata, reported_reliability=reported)
        results.append(result)
        certificates[label] = {"pairs_supplied": len(members), **dict(result.certificate)}
    scales = np.column_stack([np.asarray(result.scales, dtype=np.float64) for result in results])
    residual_variance = np.column_stack([np.asarray(result.residual_variance, dtype=np.float64) for result in results])
    np.savez(
        directory / "measurement.npz",
        groups=np.array(samples.legend),
        fit_row_counts=counts,
        scales=scales,
        residual_variance=residual_variance,
        log_reliability=np.column_stack([np.asarray(result.log_reliability, dtype=np.float64) for result in results]),
    )
    # The fit takes one model over the store's records until it takes one per ancestry group: the groups' models
    # pooled over the fit rows (its log reliability is the stacked D*'s r^2), as measurement_model defines it.
    pooled = run.bindings.pooled_measurement_model(results, np.column_stack(group_means), np.column_stack(group_variances), counts)
    pooled.save(directory / MEASUREMENT_MODEL_FILE)
    _write_json(directory / "certificate.json", certificates)
    if calibrated:
        rewrite_recalibrated_store(source_root, directory / "store", scales, samples.groups, store_directory / "fill", run.config.codec, run.budget)
    return {
        "store": "measurement/store" if calibrated else "store/store",
        "recalibrated": bool(calibrated),
        "groups_with_calibration_pairs": calibrated,
        "leakage_maps": "not built: the LD blocks are the fit's, so no mapped block pairs are passed",
    }


def _final_store(run: _Run) -> Path:
    return run.config.run_directory / run.summary("measurement")["store"]


# ---------------------------------------------------------------------------
# Steps 6-7: fit and score
# ---------------------------------------------------------------------------


def _fit_step(run: _Run, directory: Path) -> dict[str, Any]:
    cohort = _read_cohort(run)
    rows = cohort.fit_rows
    targets = np.where(cohort.training, cohort.targets[:, cohort.model_traits], np.nan)[rows]
    work = directory / "work"
    work.mkdir(exist_ok=True)
    measurement = run.bindings.load_measurement_model(run.output("measurement") / MEASUREMENT_MODEL_FILE)
    with DosageStore.open(_final_store(run)) as store:
        fitted = run.bindings.fit(
            FitRequest(
                store=store,
                store_columns=cohort.store_columns[rows],
                covariates=cohort.covariates[rows][:, 1:],
                covariate_names=cohort.covariate_names[1:],
                # Each model projects out its own trait's columns and the structure columns, never another trait's.
                covariate_columns=cohort.covariate_columns[cohort.model_traits][:, 1:],
                targets=targets,
                training=cohort.training[rows],
                model_names=cohort.model_names,
                trait_types=tuple(cohort.trait_types[trait] for trait in cohort.model_traits.tolist()),
                research_ids=tuple(research_id for research_id, fitted_row in zip(cohort.research_ids, rows) if fitted_row),
                # The measurement model reaches the fit as the records' log reliabilities (its scales already rewrote
                # the store's dosages in the measurement step). The fit has no term for its residual variances or its
                # leakage maps, and none for a target variance, so it is not handed either.
                log_variance_offset=measurement.log_reliability,
                budget=run.budget,
                work_dir=work,
                seed=run.config.seed,
            )
        )
    if tuple(fitted.model_names) != cohort.model_names:
        raise ValueError("the fit returned other models than the cohort's (trait, fold) training sets.")
    run.bindings.save_model(directory / "model", fitted)
    _remove_own(work, directory)
    return {
        "models": list(fitted.model_names),
        "refusals": [str(refusal) for refusal in getattr(fitted, "refusals", ())],
        "measurement_terms": (
            "log_variance_offset: the log reliability of the groups' measurement models pooled over the fit rows' "
            "ancestry groups; the fit has no target-variance term, so every target is taken as measured exactly, "
            "and none for the measurement model's residual variances or leakage maps"
        ),
    }


def _score_step(run: _Run, directory: Path) -> dict[str, Any]:
    cohort = _read_cohort(run)
    fitted = run.bindings.load_model(run.output("fit") / "model")
    with DosageStore.open(_final_store(run)) as store:
        prediction = run.bindings.predict(fitted, store, cohort.store_columns, cohort.covariates[:, 1:], run.budget)
    predictive_mean = np.asarray(prediction.predictive_mean, dtype=np.float64)
    held_out = np.full(cohort.targets.shape, np.nan)
    for model, trait in enumerate(cohort.model_traits.tolist()):
        rows = cohort.held_out[:, model]
        held_out[rows, trait] = predictive_mean[rows, model]
    np.savez(
        directory / "predictions.npz",
        model_names=np.array(cohort.model_names),
        predictive_mean=predictive_mean,
        predictive_variance=np.asarray(prediction.predictive_variance, dtype=np.float64),
        linear_predictor=np.asarray(prediction.linear_predictor, dtype=np.float64),
        held_out=held_out,
    )
    return {"models": len(cohort.model_names), "scored_rows": int(cohort.store_columns.shape[0])}


# ---------------------------------------------------------------------------
# Steps 8-9: report and export
# ---------------------------------------------------------------------------


def _held_out_metric(trait_type: TraitType, targets: F64Array, predictions: F64Array) -> tuple[str, float]:
    """Squared correlation of target and prediction (quantitative), or the mean log loss of the predicted probability (binary)."""
    if trait_type == TraitType.BINARY:
        with np.errstate(divide="ignore"):
            return "log_loss", float(-np.mean(np.where(targets > 0.5, np.log(predictions), np.log1p(-predictions))))
    if np.std(targets) == 0.0 or np.std(predictions) == 0.0:
        return "r2", float("nan")
    return "r2", float(np.corrcoef(targets, predictions)[0, 1] ** 2)


def _report_step(run: _Run, directory: Path) -> dict[str, Any]:
    cohort = _read_cohort(run)
    held_out = np.load(run.output("score") / "predictions.npz")["held_out"]
    lines = [("trait", "cell", "level", "n", "metric", "value")]
    for trait, name in enumerate(cohort.trait_names):
        observed = np.isfinite(cohort.targets[:, trait])
        cells = (
            [("fold", str(fold), cohort.folds == fold) for fold in range(run.config.fold_count)]
            + [("ancestry", label, np.array(cohort.ancestry) == label) for label in sorted(set(cohort.ancestry))]
            + [("half", label, np.array(cohort.half) == label) for label in sorted(set(cohort.half))]
        )
        for cell, level, member in cells:
            rows = observed & member
            count = int(rows.sum())
            metric, value = _held_out_metric(cohort.trait_types[trait], cohort.targets[rows, trait], held_out[rows, trait]) if count else ("", float("nan"))
            shown = count >= MINIMUM_REPORTED_PARTICIPANTS
            lines.append((name, cell, level, reportable_count(count), metric if shown else "", f"{value:.6g}" if shown else ""))
    with (directory / "accuracy.tsv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, delimiter="\t").writerows(lines)
    _write_json(directory / EXPORTABLE_FILE, ["accuracy.tsv"])
    return {"files": ["accuracy.tsv"]}


def check_exportable(path: Path) -> None:
    """Refuse a report table with a count of 1 to 20, or a value shown for a cell below MINIMUM_REPORTED_PARTICIPANTS."""
    for row in _read_table(path):
        count = row["n"]
        if not count.isdigit():
            if row["value"]:
                raise ValueError(f"{path}: a suppressed cell shows a value.")
            continue
        if 0 < int(count) < MINIMUM_REPORTED_PARTICIPANTS or (row["value"] and int(count) < MINIMUM_REPORTED_PARTICIPANTS):
            raise ValueError(f"{path}: a count below {MINIMUM_REPORTED_PARTICIPANTS} is shown.")


def _export_step(run: _Run, directory: Path) -> dict[str, Any]:
    report = run.output("report")
    exportable = set(json.loads((report / EXPORTABLE_FILE).read_text(encoding="utf-8")))
    for name in run.config.export:
        if name not in exportable:
            raise ValueError(f"{name!r} is not a report output; only {sorted(exportable)} can be staged.")
        check_exportable(report / name)
        shutil.copy2(report / name, directory / name)
    return {"staged": list(run.config.export), "sent": "nothing: moving a staged file out of the workspace is the user's action"}


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


_STEPS = (
    _Step("samples", (), _samples_inputs, (cohort_module, sample_crosswalk, dosage_store, _samples_step, store_half_samples, _calibration_pairs), _samples_step),
    _Step("phenotypes", (), _phenotype_inputs, (all_of_us, _phenotypes_step), _phenotypes_step),
    _Step("cohort", ("samples", "phenotypes"), lambda config: None, (cohort_module, _cohort_step, trait_table), _cohort_step),
    _Step(
        "store",
        ("samples",),
        _store_inputs,
        (
            store_converter, dosage_store, variant_typing, _store_step, _convert_chromosome, read_strata_sites, read_popped_sites,
            read_bubble_paths, path_counts, read_tandem_repeats, store_variant_classes, _decode, _decode_tasks, _gather_calibration,
        ),
        _store_step,
    ),
    _Step(
        "measurement",
        ("samples", "cohort", "store"),
        lambda config: None,
        (store_converter, dosage_store, _measurement_step, _group_code_moments, _group_calibration, rewrite_recalibrated_store),
        _measurement_step,
    ),
    _Step("fit", ("cohort", "measurement"), lambda config: config.seed, (_fit_step,), _fit_step),
    _Step("score", ("cohort", "fit"), lambda config: None, (_score_step,), _score_step),
    _Step("report", ("cohort", "score"), lambda config: None, (_report_step, _held_out_metric), _report_step),
    _Step("export", ("report",), lambda config: config.export, (_export_step, check_exportable), _export_step),
)


def run_pipeline(config: WorkspaceConfig, bindings: WorkspaceBindings, budget: ComputeBudget, *, through: str | None = None) -> dict[str, dict[str, Any]]:
    """Run every step in order, through ``through`` when given, resuming from the run directory's checkpoints."""
    if through is not None and through not in STEP_NAMES:
        raise ValueError(f"through must be one of {STEP_NAMES}.")
    config.run_directory.mkdir(parents=True, exist_ok=True)
    run = _Run(config, bindings, budget, {})
    summaries = {}
    for step in _STEPS:
        summaries[step.name] = _run_step(run, step)
        if step.name == through:
            break
    return summaries
