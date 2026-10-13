"""The pipeline that runs inside the All of Us workspace: imputed batches to held-out scores.

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
4. ``store``: the dosage store from the imputed batches (``store_converter``): popped BCFs, every
   batch checked in lockstep against the strata sidecar and the popped records against the
   sidecar's sites and ids md5; or plain imputed VCF/BCFs (GLIMPSE2, Beagle 5, Minimac4, IMPUTE5),
   every batch checked in lockstep against the first batch's records. Each half's sample manifest
   is typed by its namespace.
5. ``measurement``: the measurement model, fitted on the calibration pairs from the truth rows
   outside the imputation reference panel and its members' close relatives (``calibrating_pairs``),
   or its explicit no-truth form, and the fit's prior offset from it. Where it recalibrates, the store
   is rewritten with D*.
6. ``fit``: both arms (``ARMS``: SNVs and short indels alone, then with every structural record)
   of every trait, each trait's fold models and its all-rows model in one call, each projecting
   out its own trait's covariates. Each (arm, trait) model is saved as it finishes, so a rerun
   resumes at the first one not on disk.
7. ``score``: every model of both arms scored for the whole cohort from one read of the store.
   Each person's held-out prediction is the one from their own fold's model.
8. ``effects``: each trait's all-rows model's per-record effects (kept in the workspace) and
   their summaries by variant class and length.
9. ``report``: held-out accuracy of both arms and their paired comparison per trait, in cells by
   fold, ancestry and half, with any count of 1 to 20 suppressed.
10. ``export``: stages in ``<run>/export`` the report and effects summaries the user named in the
    config, and nothing else.

Nothing is ever sent anywhere. Moving a staged file out of the workspace is the user's action.
Everything here reads and writes participant-derived data, so every output stays in the
workspace (STORE.md, "Data class"). Progress is visible only as empty marker files whose names
hold the step and the event.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import ast
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
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

from cyvcf2 import VCF
import numpy as np
import pandas as pd
from scipy import stats

from sv_pgs import (
    all_of_us,
    cohort as cohort_module,
    dosage_store,
    fit_model,
    held_out_comparison,
    measurement_model,
    phenotype_measurement,
    probability_calibration,
    sample_crosswalk,
    scale_mixture_ep,
    store_converter,
    variant_typing,
)
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
    SECOND_DEGREE_KINSHIP,
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
from sv_pgs.artifact import CERTIFICATE_STATUS, model_part
from sv_pgs.copy_number import allele_count_decode
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.draw_laws import tile_rows
from sv_pgs.fast_scoring import SIGNED_CODE_OFFSET
from sv_pgs.fit_model import FitRequest
from sv_pgs.held_out_comparison import PairedComparison, auc_components, paired_delta_auc, paired_delta_r2
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
    encoding_flattened,
    PlainExpectedSites,
    TrLoci,
    assemble_half,
    core_spans,
    decode_batch,
    decode_called_batch,
    decode_plain_batch,
    gene_overlap,
    info_sv_length,
    no_call_fill,
    plain_batch_fields,
    plain_record_site,
    refalt_digest,
    tr_loci,
    unbreakable_group_first,
    write_store_manifest,
)
from sv_pgs.variant_typing import COMMUNITY_SV_MINIMUM_LENGTH, alt_is_symbolic_or_breakend, variant_class_and_length
from sv_pgs.fit_checkpoint import checkpoint_timings

LOGGER = logging.getLogger(__name__)

STEP_NAMES = ("samples", "store", "phenotypes", "cohort", "measurement", "fit", "score", "effects", "report", "export")
ARMS = ("snv_indel", "with_sv")
"""The study's two arms, fitted on the same people, folds and seeds so that their held-out predictions are paired.
``snv_indel`` fits every record shorter than a structural variant (``COMMUNITY_SV_MINIMUM_LENGTH``: SNVs, short indels
and short repeat-length changes) and no copy-number record; ``with_sv`` fits every record. The two differ by exactly the
structural variation of the abstract's definition (>= 50 bp, and copy number), which the baseline leaves out through a
-inf log reliability: the fit drops a record with no signal exactly (``fit_model.FitRequest``)."""
ARM_KEEPS_STRUCTURAL = {"snv_indel": False, "with_sv": True}
ALL_ROWS_MODEL = "all"
"""The model of a trait trained on every person who has the trait: the released score and the effects step's source.
SPEC: the fit itself never holds rows out; the folds' models exist only to measure held-out accuracy."""
STRUCTURAL_PART = "structural"
"""The name suffix of an all-rows model's structural part (``artifact.model_part`` on ``structural_records``), whose
genetic score the score step writes beside the model's own for the effects step's variance decomposition."""
COMPLETE_FILE = "_COMPLETE.json"
KEY_FILE = "_KEY"
EXPORTABLE_FILE = "_EXPORTABLE.json"
CONFIG_KEYS = (
    "run_directory",
    "marker_directory",
    "chromosomes",
    "imputed_halves",
    "truth_calls",
    "imputation_reference_ids",
    "strata_directory",
    "bubble_split",
    "genetic_map",
    "tandem_repeats",
    "gene_annotation",
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
KEY_HINTS = {
    "gene_annotation": (
        "gene_annotation is the public GENCODE v38 basic annotation GTF, gencode.v38.basic.annotation.gtf.gz from "
        "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/ (GRCh38, chr-prefixed contigs)."
    ),
    "genetic_map": (
        "genetic_map is null for popped batches, which carry INFO/CM. For plain imputed batches without INFO/CM it is a path "
        "pattern over {chromosome} to the public Beagle GRCh38 PLINK maps, plink.GRCh38.map.zip from "
        "https://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/plink.GRCh38.map.zip, unzipped: e.g. "
        "<dir>/no_chr_in_chrom_field/plink.{chromosome}.GRCh38.map."
    ),
    "imputation_reference_ids": (
        "imputation_reference_ids is a one-column file headed research_id listing everyone whose genome was in the "
        "imputation's reference panel, or null; with truth_calls set and it null, no calibration pair is used."
    ),
}
"""What to point a key at, for the keys whose file is a public download or whose null changes what the run does."""
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
    """One imputation half: its imputed batches in batch order, and each (chromosome, batch) file's path pattern."""

    label: str
    batches: tuple[str, ...]
    batch_path: str

    def path(self, chromosome: str, batch: str) -> Path:
        return Path(self.batch_path.format(chromosome=chromosome, batch=batch))


@dataclass(frozen=True, slots=True)
class PoppedSidecars:
    """The popped batches' sidecars: the strata directory, and bubble.split's path pattern over ``{chromosome}``."""

    strata_directory: Path
    bubble_split: str

    def bubble_split_path(self, chromosome: str) -> Path:
        return Path(self.bubble_split.format(chromosome=chromosome))


@dataclass(frozen=True, slots=True)
class WorkspaceConfig:
    """The run's inputs, from the JSON config supplied inside the workspace.

    Every key is required, so nothing runs on a default nobody chose. ``truth_calls`` is either a
    path pattern over ``{chromosome}`` for the long-read hard calls on the imputed site list
    (samples named by research ID), or an explicit null. Null runs without truth rows: no stacked
    long-read half, no calibration pairs, no D*, and each step's summary says so. ``export`` lists
    the report files the user has approved for staging, by name.

    ``imputation_reference_ids`` is the file of research IDs whose genomes were in the imputation's
    reference panel, which the calibration pairs leave out with their close relatives
    (``calibrating_pairs``), or null. It is read only with truth calls; with truth calls and null,
    no calibration pair is used.

    ``strata_directory`` and ``bubble_split`` are set together for popped batches (``popped``), or
    null together for plain imputed VCF/BCFs (``popped`` None). ``genetic_map`` is a path pattern
    over ``{chromosome}`` to a PLINK-format map, read for a plain chromosome whose records lack
    INFO/CM, or null; popped batches carry INFO/CM, so it is null for them.
    """

    run_directory: Path
    marker_directory: Path | None
    chromosomes: tuple[str, ...]
    imputed_halves: tuple[ImputedHalfInput, ...]
    truth_calls: str | None
    imputation_reference_ids: Path | None
    popped: PoppedSidecars | None
    genetic_map: str | None
    tandem_repeats: Path
    gene_annotation: Path
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
    genome_build: str = "unspecified"
    disease_likelihood: str = "gaussian_working"
    calibration_fraction: float = 0.0
    annotated_prior: bool = True

    @classmethod
    def read(cls, path: str | Path) -> WorkspaceConfig:
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> WorkspaceConfig:
        missing = sorted(set(CONFIG_KEYS) - set(raw))
        unknown = sorted(set(raw) - set(CONFIG_KEYS) - {"genome_build", "disease_likelihood", "calibration_fraction", "annotated_prior"})
        if missing or unknown:
            hints = "".join(f" {KEY_HINTS[key]}" for key in missing if key in KEY_HINTS)
            raise ValueError(f"the run config lacks {missing} and has unknown keys {unknown}; every key is required." + hints)
        if not isinstance(raw.get("calibration_fraction", 0.0), (int, float)) or not 0 <= raw.get("calibration_fraction", 0.0) < 1:
            raise ValueError("calibration_fraction must be in [0, 1)")
        if raw.get("disease_likelihood", "gaussian_working") not in ("gaussian_working", "bernoulli"):
            raise ValueError("disease_likelihood must be gaussian_working or bernoulli")
        if not isinstance(raw.get("annotated_prior", True), bool):
            raise ValueError("annotated_prior must be a boolean")
        for key, minimum in (("fold_count", 2), ("seed", 0)):
            value = raw[key]
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}; it is never rounded or coerced.")
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
        if not chromosomes or len({chromosome_number(chromosome) for chromosome in chromosomes}) != len(chromosomes):
            raise ValueError("chromosomes must list each store chromosome once.")
        try:
            research_column, sequencing_column = (str(column) for column in raw["crosswalk_columns"])
            first_column, second_column, kinship_column = (str(column) for column in raw["relatedness_columns"])
        except ValueError as error:
            raise ValueError("crosswalk_columns names (research, sequencing); relatedness_columns (first, second, kinship).") from error
        if (raw["strata_directory"] is None) != (raw["bubble_split"] is None):
            raise ValueError(
                "strata_directory and bubble_split are set together (popped batches) or null together (plain imputed VCF/BCFs)."
            )
        if raw["strata_directory"] is not None and raw["genetic_map"] is not None:
            raise ValueError("popped batches carry INFO/CM, so genetic_map is read only for plain input; set it null.")
        if raw["truth_calls"] is None and raw["imputation_reference_ids"] is not None:
            raise ValueError("imputation_reference_ids is read only for the calibration pairs of truth_calls; set it null.")
        marker_directory = raw["marker_directory"]
        return cls(
            run_directory=Path(raw["run_directory"]),
            marker_directory=None if marker_directory is None else Path(marker_directory),
            chromosomes=chromosomes,
            imputed_halves=halves,
            truth_calls=None if raw["truth_calls"] is None else str(raw["truth_calls"]),
            imputation_reference_ids=None if raw["imputation_reference_ids"] is None else Path(raw["imputation_reference_ids"]),
            popped=None if raw["strata_directory"] is None else PoppedSidecars(Path(raw["strata_directory"]), str(raw["bubble_split"])),
            genetic_map=None if raw["genetic_map"] is None else str(raw["genetic_map"]),
            tandem_repeats=Path(raw["tandem_repeats"]),
            gene_annotation=Path(raw["gene_annotation"]),
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
            genome_build=str(raw.get("genome_build", "unspecified")),
            disease_likelihood=str(raw.get("disease_likelihood", "gaussian_working")),
            calibration_fraction=float(raw.get("calibration_fraction", 0.0)),
            annotated_prior=raw.get("annotated_prior", True),
        )

    def truth_path(self, chromosome: str) -> Path:
        if self.truth_calls is None:
            raise ValueError("this run has no truth calls.")
        return Path(self.truth_calls.format(chromosome=chromosome))

    def genetic_map_path(self, chromosome: str) -> Path:
        if self.genetic_map is None:
            raise ValueError(
                f"{chromosome}: the plain batches' records lack INFO/CM, so their genetic positions come from genetic_map, which is "
                f"null; nothing assumes a rate of cM per Mb. {KEY_HINTS['genetic_map']}"
            )
        return Path(self.genetic_map.format(chromosome=chromosome))

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
        predict=artifact.predict_models,
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


def genotype_file_fingerprint(path: Path) -> tuple[int, str]:
    """Size and complete content hash, including same-size interior edits."""
    from sv_pgs.content_identity import file_digest

    return path.stat().st_size, file_digest(path)


def module_closure(*roots: ModuleType) -> tuple[ModuleType, ...]:
    """The ``roots`` and every ``sv_pgs`` module they import, transitively, in name order: the package code a step
    that runs the roots executes, read from their import statements (function-level ones included), and none it does
    not. A step keyed by the whole package refitted every model on an edit to the report or a docstring; a step keyed
    by its own functions alone reused a fit across a change to the engine underneath."""
    found: dict[str, ModuleType] = {}
    pending = list(roots)
    while pending:
        module = pending.pop()
        if module.__name__ in found:
            continue
        found[module.__name__] = module
        for node in ast.walk(ast.parse(inspect.getsource(module))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
                names = [f"{node.module}.{alias.name}" for alias in node.names] if node.module == "sv_pgs" else [node.module]
            else:
                continue
            pending.extend(importlib.import_module(name) for name in names if name.startswith("sv_pgs."))
    return tuple(found[name] for name in sorted(found))


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
    """(research ID, imputed half, its column there, its long-read column) for every long-read person with an imputed row:
    the pairs whose codes the store gathers, of which ``calibrating_pairs`` picks those the measurement model is fitted on."""
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


def read_imputation_reference_ids(path: Path) -> frozenset[ResearchId]:
    """The research IDs whose genomes were in the imputation's reference panel: one column headed ``research_id``."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if not rows or rows[0] != ["research_id"] or any(len(row) != 1 for row in rows[1:]):
        raise ValueError(f"{path}: imputation_reference_ids is one column headed research_id, one research ID per line.")
    research_ids = [ResearchId(row[0]) for row in rows[1:]]
    if len(set(research_ids)) != len(research_ids):
        raise ValueError(f"{path}: imputation_reference_ids lists a research ID more than once.")
    return frozenset(research_ids)


def calibrating_pairs(
    config: WorkspaceConfig, pairs: Sequence[tuple[str, int, int, int]], kinship_pairs: Sequence[tuple[ResearchId, ResearchId, float]]
) -> tuple[list[int], dict[str, str]]:
    """The positions of ``pairs`` the measurement model is fitted on, and the account of the others.

    The long-read people come from the assembly panel the imputation used as its reference. A panel member's imputed row
    is a leave-in imputation, and a close relative's is partly one: the panel holds the haplotypes they share (a KING
    pair above SECOND_DEGREE_KINSHIP, the bound the folds group relatives by). Calibrating on either makes the
    imputation look better than it is for a new person, so both are left out. The list is the user's statement of the
    panel: the run cannot see which genomes the imputation read, and the account says so. With truth calls and no
    reference list that overlap is unchecked, so no pair is used: the measurement model takes its no-truth form rather
    than an optimistic calibration. Counts of 1 to 20 are suppressed.
    """
    if config.truth_calls is None:
        return [], {"reference_overlap": "not applicable: no long-read half"}
    if config.imputation_reference_ids is None:
        LOGGER.warning("workspace pipeline: imputation_reference_ids is null, so no calibration pair is used")
        return [], {
            "reference_overlap": (
                "unchecked: imputation_reference_ids is null, so any pair may be a leave-in imputation of a reference "
                "genome; no calibration pair is used, and the measurement model takes its no-truth form"
            ),
            "pairs_withheld": reportable_count(len(pairs)),
        }
    reference = read_imputation_reference_ids(config.imputation_reference_ids)
    relatives = {
        candidate
        for first, second, kinship in kinship_pairs
        if kinship > SECOND_DEGREE_KINSHIP
        for member, candidate in ((first, second), (second, first))
        if member in reference
    } - reference
    research_ids = [ResearchId(pair[0]) for pair in pairs]
    left_out = reference | relatives
    return [position for position, research_id in enumerate(research_ids) if research_id not in left_out], {
        "reference_overlap": (
            "checked: the members of the panel imputation_reference_ids lists, and their relatives closer than second degree "
            "(KING), left out; that the imputation read this panel is the list's statement, which the run cannot verify"
        ),
        "reference_members_left_out": reportable_count(sum(research_id in reference for research_id in research_ids)),
        "relatives_left_out": reportable_count(sum(research_id in relatives for research_id in research_ids)),
    }


def _samples_inputs(config: WorkspaceConfig) -> Any:
    return {
        "chromosomes": config.chromosomes,
        "halves": [
            (half.label, half.batches, [genotype_file_fingerprint(half.path(chromosome, batch)) for chromosome in config.chromosomes for batch in half.batches])
            for half in config.imputed_halves
        ],
        "truth": None if config.truth_calls is None else [genotype_file_fingerprint(config.truth_path(chromosome)) for chromosome in config.chromosomes],
        "imputation_reference_ids": None if config.imputation_reference_ids is None else _file_digest(config.imputation_reference_ids),
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
    calibrating, calibration = calibrating_pairs(config, pairs, rows.kinship_pairs)
    _write_json(
        directory / "samples.json",
        {
            "halves": [{"label": label, "namespace": half.namespace, "names": list(half.names)} for label, half in zip(labels, halves)],
            "groups": legend,
            "calibration_pairs": [list(pair) for pair in pairs],
            "calibrating": calibrating,
            "calibration": calibration,
        },
    )
    return {
        "cohort_rows": len(research_ids),
        "half_samples": [len(half.names) for half in halves],
        "ancestry_groups": legend,
        "calibration_pairs": len(calibrating),
        "calibration": calibration,
        "truth": "long-read half stacked" if config.truth_calls is not None else "absent: no long-read half, no calibration pairs",
    }


@dataclass(frozen=True, slots=True)
class _Samples:
    halves: tuple[HalfSamples, ...]
    labels: tuple[str, ...]
    groups: tuple[I64Array, ...]
    legend: tuple[str, ...]
    pairs: tuple[tuple[str, int, int, int], ...]
    calibrating: tuple[int, ...]
    calibration: dict[str, str]


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
        calibrating=tuple(int(position) for position in record["calibrating"]),
        calibration=dict(record["calibration"]),
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
    # Per trait, one model per fold (trained without that fold, scored on it) and then the all-rows model (fold index
    # fold_count, which no person belongs to: it trains on every observed row and holds none out).
    models_per_trait = config.fold_count + 1
    model_traits = np.repeat(np.arange(len(trait_names)), models_per_trait)
    model_folds = np.tile(np.arange(models_per_trait), len(trait_names))
    observed = cohort.observed[:, model_traits]
    training = observed & (folds[:, None] != model_folds[None, :])
    held_out = observed & (folds[:, None] == model_folds[None, :])
    from sv_pgs.probability_calibration import calibration_mask
    calibration = np.zeros_like(training)
    for model, trait in enumerate(model_traits):
        if trait < len(config.diseases):
            calibration[:, model] = calibration_mask(training[:, model], families, config.calibration_fraction, [config.seed, int(trait), int(model_folds[model])])
    training &= ~calibration
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
        calibration=calibration,
        model_traits=model_traits,
        model_folds=model_folds,
    )
    trait_types = [TraitType.BINARY.value] * len(config.diseases) + [TraitType.QUANTITATIVE.value] * len(config.traits)
    fold_names = [f"fold{fold}" for fold in range(config.fold_count)] + [ALL_ROWS_MODEL]
    _write_json(
        directory / "cohort.json",
        {
            "covariate_names": list(cohort.covariate_names),
            "trait_names": list(trait_names),
            "trait_types": trait_types,
            "model_names": [f"{trait_names[trait]}/{fold_names[fold]}" for trait, fold in zip(model_traits.tolist(), model_folds.tolist())],
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
    families: I64Array
    training: BoolArray
    held_out: BoolArray
    model_traits: I64Array
    model_folds: I64Array
    model_names: tuple[str, ...]
    trait_names: tuple[str, ...]
    trait_types: tuple[TraitType, ...]
    ancestry: tuple[str, ...]
    half: tuple[str, ...]
    calibration: BoolArray | None = None

    @property
    def fit_rows(self) -> BoolArray:
        return np.asarray(self.training.any(axis=1))

    def trait_models(self, trait: int) -> I64Array:
        """The models of one trait, in model order: its fold models, then its all-rows model."""
        return np.flatnonzero(self.model_traits == trait)

    def all_rows_model(self, trait: int) -> int:
        """The all-rows model of one trait (ALL_ROWS_MODEL), the last of its models."""
        return int(self.trait_models(trait)[-1])


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
        families=arrays["families"],
        training=arrays["training"],
        held_out=arrays["held_out"],
        calibration=arrays["calibration"] if "calibration" in arrays else np.zeros_like(arrays["training"]),
        model_traits=arrays["model_traits"],
        model_folds=arrays["model_folds"],
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


def strata_complexity_legend(directory: Path, chromosomes: Sequence[str]) -> tuple[str, ...] | None:
    """The store's one legend for the sidecar's ``cx``: None where every chromosome's ``cx`` is numeric, else the sorted
    levels of every chromosome's together. A legend per chromosome either failed the finished store's read (a legend
    that changes at a chromosome) or, where one chromosome's ``cx`` alone held a non-numeric token, cast every later
    chromosome's numbers to that legend's codes beside the earlier ones' values."""
    columns = [
        pd.read_csv(directory / f"{chromosome}.strata.tsv.gz", sep="\t", comment=None, skiprows=1, header=None,
                    names=_strata_names(directory / f"{chromosome}.strata.tsv.gz"), usecols=["cx"], keep_default_na=False)["cx"]
        for chromosome in chromosomes
    ]
    if all(pd.api.types.is_numeric_dtype(column) for column in columns):
        return None
    return tuple(sorted({str(value) for column in columns for value in column.astype(str)}))


def _strata_names(path: Path) -> list[str]:
    """A strata sidecar's column names, from its '#' header line, which must hold every STRATA_COLUMNS name."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n")
    if not header.startswith("#"):
        raise ValueError(f"{path}: the header line must start with '#'.")
    names = header[1:].split("\t")
    missing = [column for column in STRATA_COLUMNS if column not in names]
    if missing:
        raise ValueError(f"{path} lacks the columns {missing}.")
    return names


def read_strata_sites(directory: Path, chromosome: str, complexity_legend: tuple[str, ...] | None) -> StrataSites:
    """The STRATA_COLUMNS of ``chrK.strata.tsv.gz``, once ``_done/chrK.json`` marks the contig complete.

    ``cx`` is numeric where ``complexity_legend`` is None, and is otherwise coded by that legend, the store's one legend
    over every chromosome (``strata_complexity_legend``).
    """
    done = directory / "_done" / f"{chromosome}.json"
    if not done.is_file():
        raise ValueError(f"the strata sidecar of {chromosome} is not complete: {done} does not exist.")
    manifest = json.loads(done.read_text(encoding="utf-8"))
    path = directory / f"{chromosome}.strata.tsv.gz"
    names = _strata_names(path)
    table = pd.read_csv(
        path, sep="\t", skiprows=1, header=None, names=names, usecols=list(STRATA_COLUMNS), dtype={"id": str, "refalt_md5": str}, keep_default_na=False
    )
    index = table["idx"].to_numpy(dtype=np.int64)
    if index.size and np.any(np.diff(index) != 1):
        raise ValueError(f"{path}: idx must count the popped records in file order.")
    complexity = table["cx"]
    legend = complexity_legend
    if legend is None and not pd.api.types.is_numeric_dtype(complexity):
        raise ValueError(f"{path}: cx is not numeric although every chromosome's was when the store's legend was read.")
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
class RecordSites:
    """Every record's REF, ALT, INFO/SVTYPE (None where absent), |INFO/SVLEN| (NaN where absent), INFO/END (None where
    absent) and genetic position (cM), in batch-file order: what the store's classes, cores and variant table read."""

    refs: tuple[str, ...]
    alts: tuple[str, ...]
    sv_types: tuple[str | None, ...]
    sv_lengths: F64Array
    ends: tuple[int | None, ...]
    genetic_position_cm: F64Array


def _checked_genetic_positions(path: Path, centimorgans: F64Array, source: str) -> F64Array:
    if not np.all(np.isfinite(centimorgans)) or np.any(np.diff(centimorgans) < 0):
        raise ValueError(f"{path}: {source} must be finite and non-decreasing along the chromosome.")
    return centimorgans


def read_popped_sites(path: Path, strata: StrataSites) -> RecordSites:
    """One popped batch file's site fields, checked against the strata sidecar.

    Each record is checked against its sidecar row (POS, INFO/ID, REF/ALT md5, allele lengths).
    The whole file is then checked against the sidecar's sites md5 (the md5 of the records'
    "CHROM\\tPOS\\tREF\\tALT\\n" lines) and ids md5 (the same over "INFO/ID\\n"). The popped
    records are sequence-resolved atomic alleles, typed from their alleles and |SVLEN|, so no
    SVTYPE or END is read.
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
            identifier = ExpectedSites.record_identifier(record)
            sites.update(ExpectedSites.site_line(record).encode())
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
            refs.append(record.REF)
            alts.append(alt)
            genetic.append(float(centimorgans))
            lengths.append(info_sv_length(record))
    finally:
        reader.close()
    if len(refs) != count:
        raise ValueError(f"{path}: {len(refs)} records, not the strata sidecar's {count}.")
    if sites.hexdigest() != strata.sites_md5 or ids.hexdigest() != strata.ids_md5:
        raise ValueError(f"{path}: the popped records' sites or ids md5 differs from the strata sidecar's.")
    return RecordSites(
        refs=tuple(refs),
        alts=tuple(alts),
        sv_types=(None,) * count,
        sv_lengths=np.array(lengths, dtype=np.float64),
        ends=(None,) * count,
        genetic_position_cm=_checked_genetic_positions(path, np.array(genetic, dtype=np.float64), "INFO/CM"),
    )


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


def _chromosome_spellings(chromosome: str) -> tuple[str, str]:
    """The two names a plain batch's CHROM or a genetic map's chromosome field may give a store chromosome: as the store
    writes it ('chr21') and by its number ('21')."""
    return chromosome, str(chromosome_number(chromosome))


@dataclass(frozen=True, slots=True)
class PlainSites:
    """One plain batch file's site list: the keys every batch is decoded against, each record's site fields with its
    INFO/CM (NaN where absent), and the md5 of its site lines (``PlainExpectedSites.site_line``)."""

    expected: PlainExpectedSites
    records: RecordSites
    sites_md5: str


def read_plain_sites(path: Path, chromosome: str) -> PlainSites:
    """The site list of a plain imputed batch file: every record's POS, identifier (``plain_record_identifier``: its ID,
    or CHROM:POS:REF:ALT where that is '.'), REF, ALT, INFO/SVTYPE, |INFO/SVLEN|, INFO/END and INFO/CM where present.

    Every record must have one ALT allele, lie on ``chromosome`` (CHROM as ``_chromosome_spellings`` names it) in POS
    order, and carry an identifier no other record of the chromosome has.
    """
    spellings = _chromosome_spellings(chromosome)
    positions: list[int] = []
    identifiers: list[str] = []
    refs: list[str] = []
    alts: list[str] = []
    sv_types: list[str | None] = []
    sv_lengths: list[float] = []
    ends: list[int | None] = []
    genetic: list[float] = []
    sites = hashlib.md5()
    reader = VCF(str(path))
    try:
        for record in reader:
            if len(record.ALT) != 1:
                raise ValueError(
                    f"{path}: the record at {record.CHROM}:{record.POS} has {len(record.ALT)} ALT alleles; the store reads biallelic "
                    "records, so split multiallelic ones first with `bcftools norm -m-any`."
                )
            if record.CHROM not in spellings:
                raise ValueError(f"{path}: a record lies on {record.CHROM}, not on {chromosome} (named {' or '.join(spellings)}).")
            _chromosome, position, identifier, ref, alt, sv_type, sv_length, end = plain_record_site(record)
            sites.update(PlainExpectedSites.site_line(record).encode())
            centimorgans = record.INFO.get("CM")
            positions.append(position)
            identifiers.append(identifier)
            refs.append(ref)
            alts.append(alt)
            sv_types.append(sv_type)
            sv_lengths.append(sv_length)
            ends.append(end)
            genetic.append(np.nan if centimorgans is None else float(centimorgans))
    finally:
        reader.close()
    position_array = np.array(positions, dtype=np.int64)
    if np.any(np.diff(position_array) < 0):
        raise ValueError(f"{path}: the records must be in POS order.")
    repeated = sorted(identifier for identifier, count in Counter(identifiers).items() if count > 1)
    if repeated:
        raise ValueError(
            f"{path}: {len(repeated)} identifiers name more than one record, the first {repeated[0]!r}. Give every record its own ID "
            "(e.g. `bcftools annotate --set-id '%CHROM:%POS:%REF:%FIRST_ALT'`), or '.' to name it CHROM:POS:REF:ALT."
        )
    return PlainSites(
        expected=PlainExpectedSites.from_records(position_array, refs, alts, identifiers),
        records=RecordSites(
            refs=tuple(refs),
            alts=tuple(alts),
            sv_types=tuple(sv_types),
            sv_lengths=np.array(sv_lengths, dtype=np.float64),
            ends=tuple(ends),
            genetic_position_cm=np.array(genetic, dtype=np.float64),
        ),
        sites_md5=sites.hexdigest(),
    )


GENETIC_MAP_COLUMNS = ("chromosome", "identifier", "centimorgans", "position")
"""A PLINK-format genetic map's four whitespace-separated fields in order (the Beagle maps': chromosome, variant ID, cM,
bp)."""


def read_genetic_map(path: Path, chromosome: str) -> tuple[I64Array, F64Array]:
    """One store chromosome's PLINK-format genetic map (the public Beagle GRCh38 maps) as positions (bp, strictly
    increasing) and genetic positions (cM, finite and non-decreasing), at least two rows.

    The chromosome field names the store chromosome as ``_chromosome_spellings`` does: the Beagle zip ships both
    spellings, plink.chrchr21.GRCh38.map under chr_in_chrom_field/ and plink.chr21.GRCh38.map under
    no_chr_in_chrom_field/. A row naming any other chromosome is an error.
    """
    table = pd.read_csv(
        path, sep=r"\s+", header=None, names=GENETIC_MAP_COLUMNS,
        dtype={"chromosome": str, "identifier": str, "centimorgans": np.float64, "position": np.int64}, comment="#",
    )
    spellings = _chromosome_spellings(chromosome)
    foreign = sorted(set(table["chromosome"]) - set(spellings))
    if foreign:
        raise ValueError(f"{path}: rows name the chromosomes {foreign}; a map of {chromosome} names it {' or '.join(spellings)}.")
    positions = table["position"].to_numpy(dtype=np.int64)
    centimorgans = table["centimorgans"].to_numpy(dtype=np.float64)
    if positions.shape[0] < 2 or np.any(np.diff(positions) <= 0) or not np.all(np.isfinite(centimorgans)) or np.any(np.diff(centimorgans) < 0):
        raise ValueError(f"{path}: a genetic map needs at least two rows, bp strictly increasing and cM finite and non-decreasing.")
    return positions, centimorgans


def interpolated_genetic_positions(map_positions: I64Array, map_centimorgans: F64Array, positions: NDArray) -> F64Array:
    """Each position's genetic position (cM): linear in bp between the map's positions, and before its first or after
    its last position along its first or last segment's rate."""
    positions = np.asarray(positions, dtype=np.int64)
    rates = np.diff(map_centimorgans) / np.diff(map_positions)
    before = map_centimorgans[0] + rates[0] * (positions - map_positions[0])
    after = map_centimorgans[-1] + rates[-1] * (positions - map_positions[-1])
    inside = np.interp(positions, map_positions, map_centimorgans)
    return np.where(positions < map_positions[0], before, np.where(positions > map_positions[-1], after, inside))


def read_tandem_repeats(path: Path) -> dict[str, tuple[I64Array, I64Array]]:
    """The repeat BED (GIAB AllTandemRepeatsandHomopolymers_slop5; 0-based half-open) as (starts, ends) per chromosome."""
    table = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2], names=["chromosome", "start", "end"], dtype={"chromosome": str}, comment="#")
    intervals = {}
    for chromosome, rows in table.groupby("chromosome", sort=False):
        ordered = rows.sort_values("start", kind="stable")
        intervals[str(chromosome)] = (ordered["start"].to_numpy(dtype=np.int64), ordered["end"].to_numpy(dtype=np.int64))
    return intervals


GTF_COLUMNS = ("chromosome", "source", "feature", "start", "end", "score", "strand", "frame", "attributes")
"""The GTF's nine tab-separated fields in order (GENCODE and Ensembl GTF 2.2: seqname, source, feature, start, end, score,
strand, frame, attributes); the reader keeps the chromosome, feature, span and strand."""


@dataclass(frozen=True, slots=True)
class GeneIntervals:
    """One chromosome's genes from the public GENCODE GTF, 0-based half-open: gene bodies, exons (of every
    transcript) and transcription starts (a transcript's 5' end on its strand)."""

    gene_starts: I64Array
    gene_ends: I64Array
    exon_starts: I64Array
    exon_ends: I64Array
    tss: I64Array


def _feature_spans(rows: pd.DataFrame, feature: str) -> tuple[I64Array, I64Array]:
    """A GTF feature's spans, 0-based half-open."""
    chosen = rows[rows["feature"] == feature]
    return chosen["start"].to_numpy(dtype=np.int64) - 1, chosen["end"].to_numpy(dtype=np.int64)


def read_gene_annotation(path: Path) -> dict[str, GeneIntervals]:
    """The public GENCODE GTF (1-based closed; e.g. gencode.v38.basic.annotation.gtf.gz from EBI) as each
    chromosome's ``GeneIntervals``: the store's in_gene, in_exon and log_tss_distance (``store_converter.gene_overlap``)."""
    table = pd.read_csv(
        path, sep="\t", header=None, names=GTF_COLUMNS, usecols=["chromosome", "feature", "start", "end", "strand"],
        dtype={"chromosome": str, "feature": str, "start": np.int64, "end": np.int64, "strand": str}, comment="#",
    )
    genes = {}
    for chromosome, rows in table.groupby("chromosome", sort=False):
        gene_starts, gene_ends = _feature_spans(rows, "gene")
        exon_starts, exon_ends = _feature_spans(rows, "exon")
        transcripts = rows[rows["feature"] == "transcript"]
        forward = transcripts["strand"].to_numpy() == "+"
        tss = np.where(forward, transcripts["start"].to_numpy(dtype=np.int64) - 1, transcripts["end"].to_numpy(dtype=np.int64) - 1)
        genes[str(chromosome)] = GeneIntervals(gene_starts, gene_ends, exon_starts, exon_ends, tss)
    return genes


def store_core_spans(positions: NDArray, sites: RecordSites) -> tuple[I64Array, I64Array]:
    """Each record's core, the reference bases it changes (``core_spans``, 0-based half-open).

    A symbolic or breakend ALT (a plain record's; a popped record is sequence-resolved) carries no
    bases to trim: its core is the bases after its padding base through INFO/END, [POS, END), where
    END passes POS, and otherwise, as an insertion's, the two bases around its padding base.
    """
    positions = np.asarray(positions, dtype=np.int64)
    starts, ends = core_spans(positions, sites.refs, sites.alts)
    symbolic = np.array([alt_is_symbolic_or_breakend(alt) for alt in sites.alts], dtype=bool)
    info_ends = np.array([-1 if end is None else end for end in sites.ends], dtype=np.int64)
    spanning = symbolic & (info_ends > positions)
    point = symbolic & ~spanning
    starts[spanning], ends[spanning] = positions[spanning], info_ends[spanning]
    starts[point], ends[point] = positions[point] - 1, positions[point] + 1
    return starts, ends


def store_variant_classes(sites: RecordSites, positions: NDArray, record_locus: NDArray) -> tuple[U8Array, F64Array]:
    """Each record's store class code (an index of VARIANT_CLASSES) and length, by STORE.md's first-match rules.

    A record that is not an SNV and whose core overlaps a tandem-repeat locus is str_vntr_repeat.
    Every other record is typed from its alleles, INFO/SVTYPE, |SVLEN| and INFO/END by
    variant_typing. A record typed copy_number is refused: that class's stored value is a copy
    number, and the store holds every record's DS as an ALT count.
    """
    code = {variant_class: index for index, variant_class in enumerate(VARIANT_CLASSES)}
    classes = np.empty(len(sites.refs), dtype=np.uint8)
    lengths = np.empty(len(sites.refs), dtype=np.float64)
    for row, (ref, alt) in enumerate(zip(sites.refs, sites.alts, strict=True)):
        sv_length = float(sites.sv_lengths[row])
        variant_class, length = variant_class_and_length(
            pos=int(positions[row]), ref=ref, alt=alt, svtype=sites.sv_types[row], svlen=None if np.isnan(sv_length) else sv_length,
            info_end=sites.ends[row],
        )
        if variant_class == VariantClass.COPY_NUMBER:
            raise ValueError(
                f"the record at {int(positions[row])} ({ref}>{alt}) types as copy_number, a copy number's class; its DS is an ALT "
                "count, so give it the SVTYPE or ALT of a deletion or duplication."
            )
        if variant_class != VariantClass.SNV and record_locus[row] != NO_LOCUS:
            variant_class = VariantClass.STR_VNTR_REPEAT
        classes[row] = code[variant_class]
        lengths[row] = length
    return classes, lengths


def store_group_first(bubble: NDArray, positions: NDArray, classes: NDArray, record_locus: NDArray) -> I64Array:
    """Each record's ``group_first``, the store's site key: its bubble, its same-POS set and its TR locus are one
    unsplittable group (``unbreakable_group_first``), a TR locus grouping only its TR-class records, as
    ``store_converter.overlap_group_first`` groups bench-sim's: a locus's SNVs join a group only through a bubble or a
    shared position, so a locus spans no more records than its repeat holds."""
    tr_class = np.asarray(classes) == VARIANT_CLASSES.index(VariantClass.STR_VNTR_REPEAT)
    # A TR-class record outside every repeat interval (typed by its SVTYPE or ALT token) has no locus: NO_LOCUS, a valid
    # non-negative id, grouped every such record of a chromosome into one span past any block cap.
    locus_groups = np.where(tr_class & (np.asarray(record_locus) != NO_LOCUS), np.asarray(record_locus).astype(np.int64), -1)
    return unbreakable_group_first(bubble, positions, locus_groups)


@dataclass(frozen=True, slots=True)
class ChromosomeSites:
    """One chromosome's store records as either input gives them, in batch-file order.

    ``expected`` holds the keys every batch is decoded against by ``decoder`` (or, for the
    long-read half, by ``decode_called_batch``), and ``sites_md5`` the md5 every imputed batch's
    site lines must give. ``records`` are each record's site fields and ``bubble`` its bubble (-1
    outside one). ``annotations`` are the variant-table columns only this input carries, and
    ``facts`` what the store step reports of it: the genetic positions' source, the reported r^2
    fields, and how many imputed batch files gate G4 checked of all of them.
    """

    expected: ExpectedSites
    sites_md5: str
    decoder: Callable[..., Any]
    records: RecordSites
    bubble: I64Array
    annotations: dict[str, NDArray]
    annotation_legends: dict[str, tuple[str, ...]]
    facts: dict[str, Any]


def _popped_chromosome_sites(popped: PoppedSidecars, config: WorkspaceConfig, chromosome: str, complexity_legend: tuple[str, ...] | None) -> ChromosomeSites:
    """The popped site list: the strata sidecar's rows, the first batch's records checked against them, and each
    record's bubble, K and m from bubble.split; the sidecar's n_paths, n_paths_total and cx are its annotations."""
    strata = read_strata_sites(popped.strata_directory, chromosome, complexity_legend)
    first_half = config.imputed_halves[0]
    records = read_popped_sites(first_half.path(chromosome, first_half.batches[0]), strata)
    counts = path_counts(strata, read_bubble_paths(popped.bubble_split_path(chromosome)))
    batch_count = sum(len(half.batches) for half in config.imputed_halves)
    return ChromosomeSites(
        expected=ExpectedSites(
            positions=strata.positions,
            refalt_digests=strata.refalt_digests,
            identifiers=strata.identifiers,
            kept_paths=counts.kept,
            carrying_paths=counts.carrying,
        ),
        sites_md5=strata.sites_md5,
        decoder=decode_batch,
        records=records,
        bubble=counts.bubble,
        annotations={
            "n_paths": strata.n_paths.astype(np.float64),
            "n_paths_total": strata.n_paths_total.astype(np.float64),
            "cx": strata.complexity,
        },
        annotation_legends={} if strata.complexity_legend is None else {"cx": strata.complexity_legend},
        facts={"genetic_position": "INFO/CM", "reported_r2_fields": ["INFO"], "probability_checked_batches": batch_count, "imputed_batches": batch_count},
    )


def _plain_chromosome_sites(config: WorkspaceConfig, chromosome: str) -> ChromosomeSites:
    """The plain site list: the first half's first batch's records (``read_plain_sites``), each its own single path.

    The genetic position is INFO/CM where every record carries it, and is otherwise interpolated
    from ``genetic_map``. Every batch's header must declare DS and one r^2 field
    (``plain_batch_fields``), read here before any batch is decoded. A plain record has no strata
    annotations: its own are its TR locus, length and gene columns.
    """
    first_half = config.imputed_halves[0]
    path = first_half.path(chromosome, first_half.batches[0])
    listed = read_plain_sites(path, chromosome)
    fields = [plain_batch_fields(half.path(chromosome, batch)) for half in config.imputed_halves for batch in half.batches]
    record_count = len(listed.expected.identifiers)
    carried = int(np.isfinite(listed.records.genetic_position_cm).sum())
    if carried == record_count:
        centimorgans, source = listed.records.genetic_position_cm, "INFO/CM"
    else:
        map_path = config.genetic_map_path(chromosome)
        centimorgans = interpolated_genetic_positions(*read_genetic_map(map_path, chromosome), listed.expected.positions)
        source = f"genetic_map {map_path.name}, as INFO/CM is on {carried} of {record_count} records"
    return ChromosomeSites(
        expected=listed.expected,
        sites_md5=listed.sites_md5,
        decoder=decode_plain_batch,
        records=dataclasses.replace(listed.records, genetic_position_cm=_checked_genetic_positions(path, centimorgans, source)),
        bubble=np.full(record_count, -1, dtype=np.int64),
        annotations={},
        annotation_legends={},
        facts={
            "genetic_position": source,
            "reported_r2_aggregation": "sample_weighted_batch_quality_not_pooled_INFO",
        "reported_r2_fields": sorted({batch.reported_r2 for batch in fields}),
            "probability_checked_batches": sum(batch.probabilities for batch in fields),
            "imputed_batches": len(fields),
        },
    )


@dataclass(frozen=True, slots=True)
class _DecodeTask:
    half: int
    batch: int
    path: Path
    decoder: Callable[..., Any]
    groups: I64Array
    codes_path: Path
    statistics_path: Path

    @property
    def called(self) -> bool:
        return self.decoder is decode_called_batch


_DECODER: dict[str, Any] = {}


def _start_decoder(expected: ExpectedSites, group_count: int, budget: ComputeBudget) -> None:
    _DECODER.update(expected=expected, group_count=group_count, budget=budget)


def _decode(task: _DecodeTask) -> None:
    """Decode one batch file and write its codes, then its statistics, whose presence marks the batch done."""
    decoded = task.decoder(task.path, _DECODER["expected"], task.groups, _DECODER["group_count"], task.codes_path, _DECODER["budget"])
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
        sites_md5=np.array(decoded.sites_md5),
        dosage_range=decoded.dosage_range,
        code_range=decoded.code_range,
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


def _decode_tasks(
    config: WorkspaceConfig, directory: Path, chromosome: str, samples: _Samples, decoder: Callable[..., Any]
) -> list[_DecodeTask]:
    """Every batch file of the chromosome, each imputed batch decoded by ``decoder`` and the long-read calls by
    ``decode_called_batch``."""
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
                    half_index, batch_index, path, decoder, samples.groups[half_index][start : start + width],
                    batches / f"half{half_index}.batch{batch_index}.npy", batches / f"half{half_index}.batch{batch_index}.statistics.npz",
                )
            )
            start += width
    if config.truth_calls is not None:
        half_index = len(config.imputed_halves)
        tasks.append(
            _DecodeTask(
                half_index, 0, config.truth_path(chromosome), decode_called_batch, samples.groups[half_index],
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


def _convert_chromosome(
    run: _Run, directory: Path, root: Path, chromosome: str, samples: _Samples, repeats: Mapping[str, tuple[I64Array, I64Array]],
    genes: Mapping[str, GeneIntervals], complexity_legend: tuple[str, ...] | None,
) -> dict[str, Any]:
    config = run.config
    sites = _plain_chromosome_sites(config, chromosome) if config.popped is None else _popped_chromosome_sites(config.popped, config, chromosome, complexity_legend)
    positions = sites.expected.positions
    ref_lengths = np.array([len(ref) for ref in sites.records.refs], dtype=np.int64)
    alt_lengths = np.array([len(alt) for alt in sites.records.alts], dtype=np.int64)
    core_starts, core_ends = store_core_spans(positions, sites.records)
    interval_starts, interval_ends = repeats[chromosome]
    loci = tr_loci(interval_starts, interval_ends, core_starts, core_ends, alt_lengths - ref_lengths)
    classes, lengths = store_variant_classes(sites.records, positions, loci.record_locus)
    group_first = store_group_first(sites.bubble, positions, classes, loci.record_locus)
    gene = genes[chromosome]
    gene_columns = gene_overlap(gene.gene_starts, gene.gene_ends, gene.exon_starts, gene.exon_ends, gene.tss, core_starts, core_ends)

    tasks = _decode_tasks(config, directory, chromosome, samples, sites.decoder)
    _decode_all(tasks, sites.expected, len(samples.legend), run.budget)
    statistics = [np.load(task.statistics_path) for task in tasks]
    for half_index, half in enumerate(samples.halves):
        decoded_names = tuple(name for task, values in zip(tasks, statistics) if task.half == half_index for name in values["sample_ids"].tolist())
        if decoded_names != half.names:
            raise ValueError(f"{chromosome}: half {half_index}'s batches list other samples than its manifest.")
    for task, values in zip(tasks, statistics):
        if not task.called and str(values["sites_md5"]) != sites.sites_md5:
            raise ValueError(f"{task.path}: its records' site lines differ from the site list's; every imputed batch holds the same records in order.")
    fill = no_call_fill(sum(values["group_sums"] for values in statistics), sum(values["group_counts"] for values in statistics))
    np.save(directory / "fill" / f"{chromosome}.npy", fill)
    # Descriptive sample-weighted batch quality, NOT pooled IMPUTE INFO (its denominator is not additive). Each
    # above 1 (Minimac4's R2, a variance ratio, exceeds it) set to 1 and counted in the facts.
    imputed = [(len(values["sample_ids"]), values["reported_info"]) for task, values in zip(tasks, statistics) if not task.called]
    np.save(
        directory / "reported" / f"{chromosome}.npy", sum(width * np.minimum(info, 1.0) for width, info in imputed) / sum(width for width, _info in imputed)
    )
    record_count = positions.shape[0]
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
    encoded_ids = [identifier.encode() for identifier in sites.expected.identifiers]
    table = VariantTable(
        chromosome=np.full(record_count, chromosome_number(chromosome), dtype=np.int8),
        position=positions,
        genetic_position_cm=sites.records.genetic_position_cm,
        ref_length=ref_lengths.astype(np.int32),
        alt_length=alt_lengths.astype(np.int32),
        variant_class=classes,
        # Every record the driver stores is an ALT count from the imputed batches: its value is code / 127.
        codes_per_unit=allele_count_decode(record_count)[0],
        value_origin=allele_count_decode(record_count)[1],
        group_first=group_first,
        sum_code=sums.astype(np.uint64),
        sum_code2=squares.astype(np.uint64),
        annotations={**sites.annotations, "tr_locus": loci.record_locus.astype(np.uint32), "sv_length": lengths, **gene_columns},
        annotation_legends=dict(sites.annotation_legends),
        id_bytes=np.frombuffer(b"".join(encoded_ids), dtype=np.uint8),
        id_offsets=np.concatenate([[0], np.cumsum([len(identifier) for identifier in encoded_ids])]).astype(np.int64),
    )
    write_variant_columns(root, chromosome, table, slice(0, record_count))
    _write_loci(root, chromosome, loci)
    from sv_pgs.feature_identity import chromosome_identity
    identity = chromosome_identity(
        genome_build=config.genome_build, chromosome=chromosome, positions=positions,
        refs=sites.records.refs, alts=sites.records.alts, identifiers=sites.expected.identifiers,
        ends=sites.records.ends, sv_types=sites.records.sv_types, sv_lengths=sites.records.sv_lengths,
        codes_per_unit=table.codes_per_unit, value_origin=table.value_origin,
    )
    return {
        "feature_identity": identity,
        "records": int(record_count),
        "sites_md5": sites_md5(positions, ref_lengths, alt_lengths),
        "background_zeroed": int(sum(int(values["zeroed"].sum()) for values in statistics)),
        "no_calls": int(sum(int(values["no_calls"].sum()) for values in statistics)),
        # Records whose dosages vary but whose 8-bit codes do not (DS 0.001-0.003 all store as 0): stored monomorphic.
        "encoding_flattened": int(encoding_flattened([values["dosage_range"] for values in statistics], [values["code_range"] for values in statistics]).shape[0]),
        "reported_r2_above_one": int(sum(int((info > 1.0).sum()) for _width, info in imputed)),
        "reported_r2_largest": max(float(info.max(initial=0.0)) for _width, info in imputed),
        **sites.facts,
    }


def _store_inputs(config: WorkspaceConfig) -> Any:
    popped = config.popped
    return {
        "input": "plain" if popped is None else "popped",
        "genome_build": config.genome_build,
        "chromosomes": config.chromosomes,
        "halves": [
            (half.label, half.batches, [genotype_file_fingerprint(half.path(chromosome, batch)) for chromosome in config.chromosomes for batch in half.batches])
            for half in config.imputed_halves
        ],
        "truth": None if config.truth_calls is None else [genotype_file_fingerprint(config.truth_path(chromosome)) for chromosome in config.chromosomes],
        "strata": None if popped is None else [_file_digest(popped.strata_directory / "_done" / f"{chromosome}.json") for chromosome in config.chromosomes],
        "bubble_split": None if popped is None else [genotype_file_fingerprint(popped.bubble_split_path(chromosome)) for chromosome in config.chromosomes],
        "genetic_map": None if config.genetic_map is None else [_file_digest(config.genetic_map_path(chromosome)) for chromosome in config.chromosomes],
        "tandem_repeats": _file_digest(config.tandem_repeats),
        "gene_annotation": _file_digest(config.gene_annotation),
        "codec": config.codec,
        "samples": _store_sample_layout(config),
    }


def _store_sample_layout(config: WorkspaceConfig) -> str:
    """The samples step's outputs the store reads (each half's typed manifest, its samples' ancestry groups, and the
    calibration pairs whose codes it gathers), by content: the store depends on those alone, not on the whole samples
    key, whose folds, seed, relatedness file and reference list would otherwise rebuild a genome's store when they
    change."""
    directory = config.run_directory / "samples"
    record = json.loads((directory / "samples.json").read_text(encoding="utf-8"))
    return _digest({
        "halves": record["halves"],
        "groups": record["groups"],
        "calibration_pairs": record["calibration_pairs"],
        "group_files": [_file_digest(directory / f"groups_half{index}.npy") for index in range(len(record["halves"]))],
    })


def _store_gates(config: WorkspaceConfig, facts: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """The MANIFEST's gate results, each saying what it checked: G4 only where a batch file writes GP, the strata md5s
    only for popped input, and the reported r^2 values above 1 that were set to 1."""
    checked = sum(fact["probability_checked_batches"] for fact in facts)
    batch_files = sum(fact["imputed_batches"] for fact in facts)
    above_one = sum(fact["reported_r2_above_one"] for fact in facts)
    strata = "PASS" if config.popped is not None else "NOT_APPLICABLE: plain input has no strata sidecar"
    return {
        "G2_lockstep": "PASS",
        "G4_format": "PASS" if checked == batch_files else (
            f"NOT_CHECKED on {batch_files - checked} of {batch_files} imputed batch files, which write no FORMAT/GP; PASS on the other {checked}"
        ),
        "strata_sites_md5": strata,
        "strata_ids_md5": strata,
        "reported_r2": "PASS" if above_one == 0 else (
            f"TRUNCATED: {above_one} reported r^2 values above 1 (largest {max(fact['reported_r2_largest'] for fact in facts):.6g}) set to 1"
        ),
    }


def _store_step(run: _Run, directory: Path) -> dict[str, Any]:
    config = run.config
    samples = _read_samples(run)
    root = directory / "store"
    for subdirectory in ("chromosomes", "fill", "calibration", "reported"):
        (directory / subdirectory).mkdir(exist_ok=True)
    repeats = read_tandem_repeats(config.tandem_repeats)
    genes = read_gene_annotation(config.gene_annotation)
    # Every store chromosome must be in both by the store's own spelling: a chr-less file (Ensembl's "21") converted
    # every chromosome with no TR loci, no gene and a NaN TSS distance, in silence.
    for name, table in (("tandem_repeats", repeats), ("gene_annotation", genes)):
        absent = [chromosome for chromosome in config.chromosomes if chromosome not in table]
        if absent:
            raise ValueError(f"{name} has no record on {absent}; its chromosomes are named {sorted(table)[:3]}..., the store's {list(config.chromosomes[:3])}...")
    complexity_legend = None if config.popped is None else strata_complexity_legend(config.popped.strata_directory, config.chromosomes)
    facts = []
    for chromosome in config.chromosomes:
        done = directory / "chromosomes" / f"{chromosome}.json"
        if not done.is_file():
            _clear_chromosome(root, chromosome, len(samples.halves), directory)
            _write_json(done, _convert_chromosome(run, directory, root, chromosome, samples, repeats, genes, complexity_legend))
            _remove_own(directory / "batches" / chromosome, directory)
        facts.append(json.loads(done.read_text(encoding="utf-8")))
    from sv_pgs.feature_identity import IDENTITY_FILE
    _write_json(root / IDENTITY_FILE, {chromosome: fact["feature_identity"] for chromosome, fact in zip(config.chromosomes, facts)})
    measurements = ["imputed_dosage"] * len(config.imputed_halves) + (["long_read_calls"] if config.truth_calls is not None else [])
    gates = _store_gates(config, facts)
    write_store_manifest(
        root,
        chromosomes=config.chromosomes,
        record_counts=[fact["records"] for fact in facts],
        chromosome_sites_md5=[fact["sites_md5"] for fact in facts],
        half_sample_counts=[len(half.names) for half in samples.halves],
        half_samples=samples.halves,
        half_measurements=measurements,
        gates=gates,
        recalibrated=False,
    )
    with DosageStore.open(root) as store:
        if store.half_samples() != samples.halves:
            raise ValueError("the written store's sample manifests differ from the samples step's.")
    return {
        "input": "plain" if config.popped is None else "popped",
        "records": int(sum(fact["records"] for fact in facts)),
        "halves": measurements,
        "recalibrated": False,
        "background_zeroed": int(sum(fact["background_zeroed"] for fact in facts)),
        "no_calls": int(sum(fact["no_calls"] for fact in facts)),
        "encoding_flattened": int(sum(fact["encoding_flattened"] for fact in facts)),
        "gates": gates,
        "genetic_position": {chromosome: fact["genetic_position"] for chromosome, fact in zip(config.chromosomes, facts)},
        "reported_r2_fields": sorted({field for fact in facts for field in fact["reported_r2_fields"]}),
        "reported_r2_above_one": int(sum(fact["reported_r2_above_one"] for fact in facts)),
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
    # A resumed measurement step rewrites into the same partial directory, so a copy an interrupted attempt left is
    # overwritten (copytree raised FileExistsError on every resume).
    from sv_pgs.feature_identity import IDENTITY_FILE
    if (source_root / IDENTITY_FILE).exists():
        shutil.copy2(source_root / IDENTITY_FILE, target_root / IDENTITY_FILE)
    shutil.copytree(source_root / "variants", target_root / "variants", dirs_exist_ok=True)
    shutil.copytree(source_root / "loci", target_root / "loci", dirs_exist_ok=True)
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
        members = [position for position in samples.calibrating if pair_groups[position] == group]
        calibration = None
        if members:
            calibrated.append(label)
            calibration = run.bindings.calibration_pairs(
                sample_ids=tuple(ResearchId(samples.pairs[position][0]) for position in members),
                moments=_group_calibration(run, store_directory, chromosomes, members),
            )
        result = run.bindings.fit_measurement_model(calibration, variance, strata, reported_reliability=reported)
        results.append(result)
        certificates[label] = {"pairs_supplied": len(members), "calibration": samples.calibration, **dict(result.certificate)}
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
        "calibration": samples.calibration,
        "leakage_maps": "not built: the LD blocks are the fit's, so no mapped block pairs are passed",
    }


def _final_store(run: _Run) -> Path:
    return run.config.run_directory / run.summary("measurement")["store"]


# ---------------------------------------------------------------------------
# Steps 6-7: fit and score
# ---------------------------------------------------------------------------


def structural_records(table: VariantTable) -> BoolArray:
    """Whether each store record is structural: its typed length (the store's ``sv_length``, written by the store step
    for every record) is at least COMMUNITY_SV_MINIMUM_LENGTH, or it is a copy number."""
    if "sv_length" not in table.annotations:
        raise ValueError("the store has no sv_length column, so its structural records cannot be told apart for the arms.")
    lengths = np.asarray(table.annotations["sv_length"], dtype=np.float64)
    if not np.all(np.isfinite(lengths)):
        raise ValueError("every store record needs a finite sv_length to be assigned to an arm.")
    return (lengths >= COMMUNITY_SV_MINIMUM_LENGTH) | (np.asarray(table.variant_class) == VARIANT_CLASSES.index(VariantClass.COPY_NUMBER))


def arm_log_reliability(table: VariantTable, log_reliability: F64Array, arm: str) -> F64Array:
    """The records' log reliability as ``arm`` (ARMS) fits them: unchanged where the arm fits a record, -inf where it
    leaves a structural record (``structural_records``) out."""
    return np.where(structural_records(table) & (not ARM_KEEPS_STRUCTURAL[arm]), -np.inf, np.asarray(log_reliability, dtype=np.float64))


FIT_TRAIT_TYPE = TraitType.QUANTITATIVE
"""Default working likelihood. ``disease_likelihood=bernoulli`` selects the binary likelihood explicitly.
Gaussian disease coefficients are in observed 0/1 status units, not log odds or liability units. Both execution
routes retain the full-data operator; an LD band accelerates it without dropping realized far-field correlations.
A disease's linear predictor is made a probability by a logistic calibration fitted only on reserved development
families, whose scores and labels are independent of the outer evaluation labels."""


def trait_seed(seed: int, trait: int) -> int:
    """One trait's fit seed from the run's: both arms of a trait share it, so their probes and draws are common random
    numbers where the models' layouts permit it; this does not eliminate their Monte Carlo error."""
    return int(np.random.SeedSequence([seed, trait]).generate_state(1, dtype=np.uint64)[0])


def trait_request(
    cohort: _Cohort, trait: int, store: DosageStore, log_variance_offset: F64Array, budget: ComputeBudget, work_dir: Path, seed: int,
    disease_likelihood: str = "gaussian_working", annotated_prior: bool = True,
) -> FitRequest:
    """One trait's models (its folds' and its all-rows model) as one fit, on the rows any of them trains on.

    Every model adjusts for its own trait's covariates and the structure columns, never another trait's; the
    covariate names are the cohort's whole union, so every trait's fitted model scores with the same covariates
    (``artifact.predict_models``). A row's target is read only where its model trains on it.

    The configured disease likelihood is explicit; quantitative traits always use Gaussian regression.
    """
    models = cohort.trait_models(trait)
    training = cohort.training[:, models]
    rows = training.any(axis=1)
    adjusted = cohort.covariate_columns[np.full(models.size, trait)]
    return FitRequest(
        store=store,
        store_columns=cohort.store_columns[rows],
        covariates=cohort.covariates[rows][:, 1:],
        covariate_names=cohort.covariate_names[1:],
        covariate_columns=adjusted[:, 1:],
        targets=np.where(training, cohort.targets[:, [trait]], np.nan)[rows],
        training=training[rows],
        model_names=tuple(cohort.model_names[model] for model in models),
        trait_types=((TraitType.BINARY if disease_likelihood == "bernoulli" and cohort.trait_types[trait] == TraitType.BINARY else TraitType.QUANTITATIVE),) * models.size,
        research_ids=tuple(np.asarray(cohort.research_ids)[rows].tolist()),
        log_variance_offset=log_variance_offset,
        budget=budget,
        work_dir=work_dir,
        seed=seed,
        annotated_prior=annotated_prior,
    )


def _fitted_model_directory(run: _Run, arm: str, trait_name: str) -> Path:
    return run.output("fit") / "models" / arm / trait_name


def _fit_step(run: _Run, directory: Path) -> dict[str, Any]:
    """Every (arm, trait) fit, each saved on its own as it finishes: a rerun after a preemption resumes at the first
    (arm, trait) whose model is not on disk (``artifact.save_model`` renames a complete model into place, so a model
    directory that exists is whole). Each saved model's outcome is recorded beside it, from the saved model.

    The fits run trait by trait in the config's order, both arms of a trait back to back, so a run its budget stops
    (``launcher/workspace``: each job's runtime is bounded) holds complete arm comparisons for the traits it reached.
    After each trait the log gives the hours its fits took and the step's projected total."""
    cohort = _read_cohort(run)
    # The measurement model reaches the fit as the records' log reliabilities (its scales already rewrote the store's
    # dosages in the measurement step). The fit has no term for its residual variances or its leakage maps, and none for
    # a target variance, so it is not handed either.
    measurement = run.bindings.load_measurement_model(run.output("measurement") / MEASUREMENT_MODEL_FILE)
    records: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    fitted_seconds: list[float] = []
    with DosageStore.open(_final_store(run)) as store:
        offsets_of = {arm: arm_log_reliability(store.variant_table, measurement.log_reliability, arm) for arm in ARMS}
        for trait, trait_name in enumerate(cohort.trait_names):
            started = time.perf_counter()
            fitted_here = False
            for arm in ARMS:
                offsets = offsets_of[arm]
                target = directory / "models" / arm / trait_name
                record = target.with_name(target.name + ".json")
                if not record.is_file():
                    work = directory / "work" / arm / trait_name
                    # A Spot restart keeps completed fold checkpoints; the engine discards only
                    # the interrupted model's scratch and verifies each checkpoint's input digest.
                    work.mkdir(parents=True, exist_ok=True)
                    request = trait_request(cohort, trait, store, offsets, run.budget, work, trait_seed(run.config.seed, trait),
                                            run.config.disease_likelihood, run.config.annotated_prior)
                    if not target.is_dir():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        LOGGER.info("workspace pipeline: fitting %s / %s", arm, trait_name)
                        run.bindings.save_model(target, run.bindings.fit(request))
                        fitted_here = True
                    fitted = run.bindings.load_model(target)
                    if tuple(fitted.model_names) != request.model_names:
                        raise ValueError(f"the {arm} fit of {trait_name} returned other models than the trait's training sets.")
                    _write_json(
                        record,
                        {
                            "models": list(fitted.model_names),
                            "completed_fold_timings": checkpoint_timings(work),
                            "fitted_records": [int(scoring.store_rows.shape[0]) for scoring in fitted.scoring],
                            CERTIFICATE_STATUS: [bool(met) for met in np.asarray(fitted.certificate[CERTIFICATE_STATUS])],
                            "refusals": [str(refusal) for refusal in fitted.refusals],
                        },
                    )
                    _remove_own(work, directory)
                records[arm][trait_name] = json.loads(record.read_text(encoding="utf-8"))
            if fitted_here:
                fitted_seconds.append(time.perf_counter() - started)
                remaining = len(cohort.trait_names) - trait - 1
                LOGGER.info(
                    "workspace pipeline: %s's fits took %.2f h in this run; at this run's mean of %.2f h a trait, the %d traits "
                    "left need %.1f h more", trait_name, fitted_seconds[-1] / 3600.0, np.mean(fitted_seconds) / 3600.0, remaining,
                    remaining * np.mean(fitted_seconds) / 3600.0,
                )
    _remove_own(directory / "work", directory)
    return {
        "arms": {
            arm: {
                "models": sum(len(record["models"]) for record in by_trait.values()),
                "outer_criterion_met": sum(sum(record[CERTIFICATE_STATUS]) for record in by_trait.values()),
                "refusals": sum(len(record["refusals"]) for record in by_trait.values()),
            }
            for arm, by_trait in records.items()
        },
        "measurement_terms": (
            "log_variance_offset: the log reliability of the groups' measurement models pooled over the fit rows' "
            "ancestry groups (-inf on the records an arm leaves out); the fit has no target-variance term, so every "
            "target is taken as measured exactly, and none for the measurement model's residual variances or leakage maps"
        ),
    }


@dataclass(frozen=True, slots=True)
class _Scores:
    """The score step's arrays: [arms, cohort rows, models] over every model, each person's own-fold held-out
    values [arms, cohort rows, traits] (NaN where the person has no held-out prediction of the trait), and
    ``structural_genetic`` [cohort rows, traits], the genetic score of the structural part (STRUCTURAL_PART) of each
    trait's all-rows ``with_sv`` model."""

    predictive_mean: F64Array
    linear_predictor: F64Array
    genetic_mean: F64Array
    held_out_predictive_mean: F64Array
    held_out_linear_predictor: F64Array
    held_out_genetic: F64Array
    structural_genetic: F64Array
    held_out_calibrated_logits: F64Array | None = None


def score_batches(model_bytes: Sequence[int], host_bytes: int) -> list[range]:
    """Consecutive runs of the saved fitted models whose files together hold at most a quarter of ``host_bytes`` (each
    run at least one model): loaded together, they are scored from one read of the store, and the scorer's plan and
    accumulators take the rest of the host. Every model at once held some 200 GB at All of Us scale (2 arms x 15
    diseases x 6 models, each ~20M records of means, scales, coefficients and draw-law parameters)."""
    batches: list[range] = []
    start, held = 0, 0
    for position, size in enumerate(model_bytes):
        if position > start and held + size > host_bytes // 4:
            batches.append(range(start, position))
            start, held = position, 0
        held += size
    batches.append(range(start, len(model_bytes)))
    return batches


def _score_step(run: _Run, directory: Path) -> dict[str, Any]:
    """Every model of both arms scored for the whole cohort (``predict_models``), in as few reads of the store as the
    host's memory allows the saved models to be held together (``score_batches``)."""
    from sv_pgs.model_acceptance import require_accepted
    cohort = _read_cohort(run)
    directories = [_fitted_model_directory(run, arm, trait_name) for arm in ARMS for trait_name in cohort.trait_names]
    shape = (len(ARMS), cohort.store_columns.shape[0], len(cohort.model_names))
    columns = {name: np.empty((shape[1], shape[0] * shape[2])) for name in ("predictive_mean", "predictive_variance", "linear_predictor", "genetic_mean")}
    names: list[str] = []
    trait_count = len(cohort.trait_names)
    structural_genetic = np.empty((shape[1], trait_count))
    batches = score_batches([sum(path.stat().st_size for path in folder.iterdir()) for folder in directories], run.budget.host_bytes)
    with DosageStore.open(_final_store(run)) as store:
        structural = structural_records(store.variant_table)
        for batch in batches:
            fitted = [run.bindings.load_model(directories[position]) for position in batch]
            for model in fitted:
                require_accepted(model)
            # The structural part of each trait's all-rows with_sv model, by trait, scored in the same read as the
            # models themselves, after them.
            parts = {}
            for position, model in zip(batch, fitted):
                if ARMS[position // trait_count] == "with_sv":
                    all_rows_name = cohort.model_names[cohort.all_rows_model(position % trait_count)]
                    parts[position % trait_count] = model_part(model, all_rows_name, structural, f"{all_rows_name}/{STRUCTURAL_PART}")
            prediction = run.bindings.predict([*fitted, *parts.values()], store, cohort.store_columns, cohort.covariates[:, 1:], run.budget)
            first = len(names)
            names.extend(name for model in fitted for name in model.model_names)
            for name, values in (
                ("predictive_mean", prediction.predictive_mean), ("predictive_variance", prediction.predictive_variance),
                ("linear_predictor", prediction.linear_predictor), ("genetic_mean", prediction.genetic.means),
            ):
                columns[name][:, first : len(names)] = np.asarray(values, dtype=np.float64)[:, : len(names) - first]
            structural_genetic[:, list(parts)] = np.asarray(prediction.genetic.means, dtype=np.float64)[:, len(names) - first :]
            del fitted, parts, prediction
    for arm_index, arm in enumerate(ARMS):
        if tuple(names[arm_index * shape[2] : (arm_index + 1) * shape[2]]) != cohort.model_names:
            raise ValueError(f"the {arm} arm's saved models are not the cohort's models in order.")
    LOGGER.info("workspace pipeline: scored %d models in %d reads of the store", len(names), len(batches))

    def by_arm(values: Any) -> F64Array:
        # [rows, arms x models] in arm-major model order, as [arms, rows, models].
        return np.asarray(values, dtype=np.float64).reshape(shape[1], shape[0], shape[2]).transpose(1, 0, 2)

    def held_out(values: F64Array) -> F64Array:
        own = np.full((len(ARMS), shape[1], len(cohort.trait_names)), np.nan)
        for model, trait in enumerate(cohort.model_traits.tolist()):
            rows = cohort.held_out[:, model]
            own[:, rows, trait] = values[:, rows, model]
        return own

    predictive_mean = by_arm(columns["predictive_mean"])
    linear_predictor = by_arm(columns["linear_predictor"])
    genetic_mean = by_arm(columns["genetic_mean"])
    calibrated_logits = np.full_like(linear_predictor, np.nan)
    calibrators = []
    from sv_pgs.probability_calibration import FrozenCalibration
    if run.config.calibration_fraction:
        for model, trait in enumerate(cohort.model_traits):
            if cohort.trait_types[trait] != TraitType.BINARY:
                continue
            reserved = cohort.calibration[:, model]
            if np.any(reserved & (cohort.training[:, model] | cohort.held_out[:, model])):
                raise ValueError("Calibration overlaps training or evaluation")
            for arm_index, arm in enumerate(ARMS):
                calibration = FrozenCalibration.fit(linear_predictor[arm_index, reserved, model], cohort.targets[reserved, trait])
                calibrated_logits[arm_index, :, model] = calibration.logits(linear_predictor[arm_index, :, model])
                calibrators.append({"arm": arm, "model": cohort.model_names[model], **dataclasses.asdict(calibration),
                                    "source": "independent_development_families", "evaluation_labels_used": False})
    _write_json(directory / "calibrators.json", calibrators)
    np.savez(
        directory / "predictions.npz",
        arms=np.array(ARMS),
        model_names=np.array(cohort.model_names),
        predictive_mean=predictive_mean,
        predictive_variance=by_arm(columns["predictive_variance"]),
        linear_predictor=linear_predictor,
        genetic_mean=genetic_mean,
        held_out_predictive_mean=held_out(predictive_mean),
        held_out_linear_predictor=held_out(linear_predictor),
        held_out_genetic=held_out(genetic_mean),
        structural_genetic=structural_genetic,
        calibrated_logits=calibrated_logits,
        held_out_calibrated_logits=held_out(calibrated_logits),
    )
    return {"arms": list(ARMS), "models": len(names), "store_reads": len(batches), "scored_rows": int(cohort.store_columns.shape[0])}


def _read_scores(run: _Run) -> _Scores:
    arrays = np.load(run.output("score") / "predictions.npz")
    if tuple(str(arm) for arm in arrays["arms"]) != ARMS:
        raise ValueError("the score step's arms are not this driver's.")
    return _Scores(**{field.name: arrays[field.name] for field in dataclasses.fields(_Scores)})


# ---------------------------------------------------------------------------
# Steps 8-10: effects, report and export
# ---------------------------------------------------------------------------


LENGTH_DECADE = 10
"""The effects summary bins a structural record's length by decades from COMMUNITY_SV_MINIMUM_LENGTH (50-499 bp,
500-4,999 bp, ...): a display grouping, never an input to any fit."""
SHORT_RECORDS = "<50"
"""The length bin of every record shorter than a structural variant."""


@dataclass(frozen=True, slots=True)
class VariantEffects:
    """One fitted model's effects per fitted store record, in the record's own units (one ALT allele, or one copy for a
    copy number), on the fitted likelihood's scale: trait/status units for Gaussian fits, log odds only for binary-likelihood fits.

    The effect per unit is beta / sigma x codes_per_unit (``fast_scoring``: beta multiplies the standardized signed
    code, whose SD is sigma). Every moment is under the law the model's draws come from (``fast_scoring``: the
    posterior on the full-data route, the fitted product approximation on the mean-field route).
    """

    store_rows: I64Array
    mean: F64Array
    standard_deviation: F64Array
    probability_positive: F64Array
    probability_negative: F64Array


def variant_effects(scoring: Any, codes_per_unit: NDArray, working_bytes: int) -> VariantEffects:
    """The effects of ``scoring`` (a ``fast_scoring.ScoringModel``) at its store rows, reading its draws a tile of rows
    at a time within ``working_bytes``."""
    rows = np.asarray(scoring.store_rows, dtype=np.int64)
    unit = np.asarray(codes_per_unit, dtype=np.float64)[rows] / np.asarray(scoring.signed_scales, dtype=np.float64)
    law = scoring.posterior_draws
    variance = np.empty(rows.shape[0])
    coefficients = np.asarray(scoring.coefficients, dtype=np.float64)
    positive = np.empty(rows.shape[0])
    negative = np.empty(rows.shape[0])
    step = tile_rows(law, working_bytes)
    for start in range(0, rows.shape[0], step):
        stop = min(start + step, rows.shape[0])
        if hasattr(law, "marginal_summary"):
            variance[start:stop], positive[start:stop], negative[start:stop] = law.marginal_summary(
                start, stop, coefficients[start:stop], include_negative=True)
        else:
            draws = law.tile(start, stop)
            variance[start:stop] = np.mean((draws - coefficients[start:stop, None]) ** 2, axis=1)
            positive[start:stop] = np.mean(draws > 0.0, axis=1)
            negative[start:stop] = np.mean(draws < 0.0, axis=1)
    coefficients = np.asarray(scoring.coefficients, dtype=np.float64)
    return VariantEffects(
        store_rows=rows,
        mean=coefficients * unit,
        standard_deviation=np.sqrt(variance) * unit,
        probability_positive=positive,
        probability_negative=negative,
    )


def length_bins(lengths: F64Array) -> NDArray:
    """Each record's length bin: SHORT_RECORDS below COMMUNITY_SV_MINIMUM_LENGTH, else its decade from it."""
    ratio = np.maximum(lengths, COMMUNITY_SV_MINIMUM_LENGTH) / COMMUNITY_SV_MINIMUM_LENGTH
    decades, decade_of_record = np.unique(np.floor(np.log(ratio) / np.log(LENGTH_DECADE)).astype(np.int64), return_inverse=True)
    lower = [COMMUNITY_SV_MINIMUM_LENGTH * LENGTH_DECADE**decade for decade in decades.tolist()]
    labels = np.array([f"{low}-{low * LENGTH_DECADE - 1}" for low in lower], dtype=object)[decade_of_record]
    return np.where(lengths < COMMUNITY_SV_MINIMUM_LENGTH, SHORT_RECORDS, labels)


EFFECT_SUMMARY_COLUMNS = (
    "trait", "arm", "variant_class", "length_bin", "variants", "positive_mean", "negative_mean", "expected_positive",
    "expected_negative", "mean_absolute_effect", "median_absolute_effect", "root_mean_square_effect",
)
"""The effects summary: per (trait, arm, class, length bin) the fitted records, how many have a positive and a negative
posterior mean, the posterior-expected counts of positive and negative effects (sum of P(beta > 0) and of P(beta < 0)),
and the mean, median and root-mean-square posterior effect per unit. Records, not participants, are counted. A bin's
share of the genetic variance is not a sum over its records under LD; GENETIC_VARIANCE_COLUMNS gives the structural
records' share from the scores themselves."""
GENETIC_VARIANCE_COLUMNS = ("trait", "n", "quantity", "value")
"""Per trait, the variance of its all-rows ``with_sv`` model's genetic score over the trait's observed cohort rows (n of
them), split between the score's structural part and the rest (``genetic_variance_decomposition``), one quantity a
row. Every value is an aggregate over the n rows; a trait observed on 1 to 20 rows shows neither its count nor a value."""
LENGTH_CORRELATION_COLUMNS = ("trait", "arm", "variant_class", "variants", "quantity", "spearman_rho", "interpretation")
"""Per (trait, arm, structural class, and every structural record together), Spearman's rho between log length and
the absolute fitted mean and uncertainty separately. Descriptive only: LD and prior features invalidate an independent-record significance claim."""


def genetic_variance_decomposition(total: F64Array, structural: F64Array) -> dict[str, float]:
    """The variance of the genetic score ``total`` over its rows, split between its ``structural`` part and the rest.

    A genetic score is g = X beta over standardized columns, so Var(g) = beta' Sigma beta for Sigma the rows' genotype
    covariance. Its structural part s = X_S beta_S and its small part r = g - s split it as
    Var(g) = Var(s) + Var(r) + 2 Cov(s, r), whose cross term (``covariance_term``) is the LD between structural and small
    records that a sum over records misses. The structural share is Cov(s, g) / Var(g) = (Var(s) + Cov(s, r)) / Var(g),
    the part of Var(g) the structural records carry: the two parts' shares sum to one, and either leaves [0, 1] when the
    parts are negatively correlated. Moments are population moments over the rows, as the scores' standardization is
    (``fast_scoring``); a score without variance has no share (NaN).
    """
    small = total - structural
    covariance = np.cov(np.vstack([structural, small]), bias=True)
    total_variance = float(np.var(total))
    return {
        "total_variance": total_variance,
        "structural_variance": float(covariance[0, 0]),
        "small_variance": float(covariance[1, 1]),
        "covariance_term": 2.0 * float(covariance[0, 1]),
        "structural_share": float(covariance[0, 0] + covariance[0, 1]) / total_variance if total_variance > 0.0 else float("nan"),
    }


def _effects_step(run: _Run, directory: Path) -> dict[str, Any]:
    """Each trait's all-rows model, in both arms: every fitted record's effect (``effects/<trait>.<arm>.tsv.gz``, kept in
    the workspace: it carries each record's mean dosage, from which a rare record's carrier count follows), two
    summaries by variant class and length that carry no participant count (``effects_by_class.tsv``,
    ``effect_length_correlation.tsv``), and the structural records' share of the ``with_sv`` score's variance over each
    trait's observed rows, from the score step's scores (``genetic_variance.tsv``, GENETIC_VARIANCE_COLUMNS)."""
    cohort = _read_cohort(run)
    scores = _read_scores(run)
    variance_lines: list[tuple[Any, ...]] = [GENETIC_VARIANCE_COLUMNS]
    for trait, trait_name in enumerate(cohort.trait_names):
        observed = np.flatnonzero(np.isfinite(cohort.targets[:, trait]))
        count = observed.shape[0]
        decomposition = genetic_variance_decomposition(
            scores.genetic_mean[ARMS.index("with_sv"), observed, cohort.all_rows_model(trait)], scores.structural_genetic[observed, trait]
        )
        variance_lines.extend(
            (trait_name, reportable_count(count), quantity, f"{value:.6g}" if _shown(count, None) else "") for quantity, value in decomposition.items()
        )
    summary_lines: list[tuple[Any, ...]] = [EFFECT_SUMMARY_COLUMNS]
    correlation_lines: list[tuple[Any, ...]] = [LENGTH_CORRELATION_COLUMNS]
    with DosageStore.open(_final_store(run)) as store:
        table = store.variant_table
        classes = np.asarray(VARIANT_CLASS_LEGEND, dtype=object)[np.asarray(table.variant_class)]
        lengths = np.asarray(table.annotations["sv_length"], dtype=np.float64)
        bins = length_bins(lengths)
        codes_per_unit = np.asarray(table.codes_per_unit, dtype=np.float64)
        working_bytes = run.budget.host_bytes // 2
        for arm in ARMS:
            for trait, trait_name in enumerate(cohort.trait_names):
                fitted = run.bindings.load_model(_fitted_model_directory(run, arm, trait_name))
                from sv_pgs.model_acceptance import require_accepted
                require_accepted(fitted)
                scoring = fitted.scoring[list(fitted.model_names).index(cohort.model_names[cohort.all_rows_model(trait)])]
                effects = variant_effects(scoring, codes_per_unit, working_bytes)
                rows = effects.store_rows
                root_mean_square = np.sqrt(effects.mean * effects.mean + effects.standard_deviation * effects.standard_deviation)
                pd.DataFrame(
                    {
                        "chromosome": np.asarray(table.chromosome)[rows].astype(np.int64),
                        "position": np.asarray(table.position)[rows],
                        "id": table.variant_ids(rows),
                        "variant_class": classes[rows],
                        "length": lengths[rows],
                        "mean_value": (np.asarray(scoring.signed_means) + SIGNED_CODE_OFFSET) / codes_per_unit[rows] + np.asarray(table.value_origin)[rows],
                        "effect": effects.mean,
                        "posterior_sd": effects.standard_deviation,
                        "likelihood": "bernoulli" if scoring.trait_type == TraitType.BINARY else "gaussian",
                        "effect_scale": "log_odds" if scoring.trait_type == TraitType.BINARY else ("observed_status" if cohort.trait_types[trait] == TraitType.BINARY else "trait_units"),
                        "sign_method": "analytic_represented_mixture" if hasattr(scoring.posterior_draws, "marginal_summary") else "monte_carlo",
                        "sign_draw_count": 0 if hasattr(scoring.posterior_draws, "marginal_summary") else scoring.draw_count,
                        "sign_probability_resolution": 0.0 if hasattr(scoring.posterior_draws, "marginal_summary") else 1.0 / scoring.draw_count,
                        "probability_positive": effects.probability_positive,
                        "probability_negative": effects.probability_negative,
                    }
                ).to_csv(
                    directory / f"{trait_name}.{arm}.tsv.gz", sep="\t", index=False, float_format="%.6g",
                    # gzip's fastest level: a genome's records are tens of millions of rows per table.
                    compression={"method": "gzip", "compresslevel": 1},
                )
                frame = pd.DataFrame(
                    {
                        "variant_class": classes[rows], "length_bin": bins[rows], "mean": effects.mean,
                        "probability_positive": effects.probability_positive, "probability_negative": effects.probability_negative,
                        "root_mean_square": root_mean_square,
                    }
                )
                for (variant_class, length_bin), group in frame.groupby(["variant_class", "length_bin"], sort=True):
                    absolute = np.abs(group["mean"].to_numpy())
                    summary_lines.append((
                        trait_name, arm, variant_class, length_bin, len(group), int((group["mean"] > 0.0).sum()), int((group["mean"] < 0.0).sum()),
                        f"{group['probability_positive'].sum():.6g}", f"{group['probability_negative'].sum():.6g}",
                        f"{absolute.mean():.6g}", f"{np.median(absolute):.6g}", f"{np.sqrt(np.mean(group['root_mean_square'].to_numpy() ** 2)):.6g}",
                    ))
                structural = lengths[rows] >= COMMUNITY_SV_MINIMUM_LENGTH
                for variant_class in [*sorted(set(classes[rows][structural])), "all_structural"]:
                    member = structural & ((classes[rows] == variant_class) | (variant_class == "all_structural"))
                    if np.unique(lengths[rows][member]).shape[0] < 2:
                        continue
                    for quantity, values in (("absolute_mean", np.abs(effects.mean)), ("posterior_sd", effects.standard_deviation)):
                        selected = values[member]
                        rho = stats.spearmanr(np.log(lengths[rows][member]), selected).statistic if np.ptp(selected) > 0 else float("nan")
                        correlation_lines.append((trait_name, arm, variant_class, int(member.sum()), quantity, f"{rho:.6g}", "descriptive_prior_and_LD_dependent"))
    for name, lines in (("effects_by_class.tsv", summary_lines), ("effect_length_correlation.tsv", correlation_lines), ("genetic_variance.tsv", variance_lines)):
        with (directory / name).open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle, delimiter="\t").writerows(lines)
    exportable = ["effects_by_class.tsv", "effect_length_correlation.tsv", "genetic_variance.tsv"]
    _write_json(directory / EXPORTABLE_FILE, exportable)
    return {"files": exportable, "variant_tables": f"{len(ARMS) * len(cohort.trait_names)} per-record effect tables, kept in the workspace"}


ACCURACY_COLUMNS = ("trait", "arm", "cell", "level", "n", "cases", "metric", "value")
COMPARISON_COLUMNS = ("trait", "cell", "level", "n", "cases", "metric", "estimate", "standard_error", "z", "p_value", "inference")
PARTICIPANT_COUNT_COLUMNS = ("n", "cases")
"""The report columns that count participants, which the All of Us rule never lets show 1 to 20."""


def _residualized(values: F64Array, covariates: F64Array) -> F64Array:
    """``values`` less their least-squares projection on ``covariates`` (whose span holds the intercept)."""
    return values - covariates @ np.linalg.lstsq(covariates, values, rcond=None)[0]


def _partial_r2(target: F64Array, score: F64Array, covariates: F64Array) -> float:
    """The squared correlation of a score with a target after both are residualized on the covariates."""
    target_residual, score_residual = _residualized(target, covariates), _residualized(score, covariates)
    if np.std(target_residual) == 0.0 or np.std(score_residual) == 0.0:
        return float("nan")
    return float(np.corrcoef(target_residual, score_residual)[0, 1] ** 2)


@dataclass(frozen=True, slots=True)
class _HeldOutCell:
    """One report cell of one trait, split by fold: each fold's held-out rows in the cell (its own model scored them)."""

    folds: tuple[I64Array, ...]

    @property
    def count(self) -> int:
        return int(sum(rows.shape[0] for rows in self.folds))


def _cells(cohort: _Cohort, trait: int, fold_count: int) -> list[tuple[str, str, _HeldOutCell]]:
    """Every report cell of a trait: all its held-out rows, each fold, each ancestry group and each half."""
    observed = np.isfinite(cohort.targets[:, trait])
    ancestry, half = np.array(cohort.ancestry), np.array(cohort.half)

    def cell(member: BoolArray) -> _HeldOutCell:
        return _HeldOutCell(tuple(np.flatnonzero(observed & member & (cohort.folds == fold)) for fold in range(fold_count)))

    everyone = np.ones(cohort.folds.shape[0], dtype=bool)
    return (
        [("all", "all", cell(everyone))]
        + [("fold", str(fold), cell(cohort.folds == fold)) for fold in range(fold_count)]
        + [("ancestry", label, cell(ancestry == label)) for label in sorted(set(cohort.ancestry))]
        + [("half", label, cell(half == label)) for label in sorted(set(cohort.half))]
    )


def _fold_weighted(values: Sequence[tuple[int, float]]) -> float:
    """The n-weighted mean of per-fold values: each fold's own model on its own fold, never one pooled correlation of
    the K folds' differently scaled predictors (``held_out_comparison``)."""
    counts = np.array([count for count, _value in values], dtype=np.float64)
    return float(np.dot(counts, [value for _count, value in values]) / counts.sum()) if counts.sum() else float("nan")


def _fold_combined(comparisons: Sequence[tuple[int, PairedComparison]]) -> PairedComparison:
    """Per-fold paired estimates combined with weights n_f / n and conditional variance in quadrature.

    Disjoint held-out sets do not make the fitted predictors independent. The cross-fold pair term needs each arm's
    design and posterior covariance, which a biobank-scale fit never forms, and is left out: the folds' shared training rows make
    that term the part of the variance this standard error does not hold."""
    counts = np.array([count for count, _comparison in comparisons], dtype=np.float64)
    weights = counts / counts.sum()
    estimate = float(np.dot(weights, [comparison.estimate for _count, comparison in comparisons]))
    standard_error = float(np.sqrt(np.dot(weights**2, [comparison.standard_error**2 for _count, comparison in comparisons])))
    return held_out_comparison.comparison_from_estimate(estimate, standard_error)


def _shown(count: int, cases: int | None) -> bool:
    """Whether a cell's values may be shown: at least MINIMUM_REPORTED_PARTICIPANTS rows, and for a disease as many
    cases and as many controls, so no count of 1 to 20 is shown or follows from the shown ones."""
    if count < MINIMUM_REPORTED_PARTICIPANTS:
        return False
    return cases is None or min(cases, count - cases) >= MINIMUM_REPORTED_PARTICIPANTS


def _cases_text(count: int, cases: int | None) -> str:
    """A disease cell's case count as the report writes it: shown only where neither it nor the controls it implies
    (the shown total less it) is 1 to 20; empty for a quantitative trait."""
    if cases is None:
        return ""
    controls = count - cases
    if all(part == 0 or part >= MINIMUM_REPORTED_PARTICIPANTS for part in (cases, controls)):
        return str(cases)
    return "suppressed"


def _scored_folds(held: _HeldOutCell, targets: F64Array, binary: bool) -> list[I64Array]:
    """The folds of a cell that a held-out metric can be read on: a disease's with at least two cases and two controls
    (DeLong's components need a variance of each), a quantitative trait's whose targets vary."""
    if binary:
        return [rows for rows in held.folds if min(int(targets[rows].sum()), rows.shape[0] - int(targets[rows].sum())) >= 2]
    return [rows for rows in held.folds if rows.shape[0] and np.ptp(targets[rows]) > 0.0]


PROPER_SCORES = ("log_loss", "brier", "log_likelihood_gain", "brier_reduction")
"""The metrics of a disease's frozen probabilities, which a fold with one outcome class still has (``_report_step``)."""


def _evaluated_counts(selected: Sequence[I64Array], targets: F64Array, binary: bool) -> tuple[int, int | None]:
    """The rows, and a disease's cases, of the folds a metric is read on."""
    return sum(rows.shape[0] for rows in selected), int(sum(targets[rows].sum() for rows in selected)) if binary else None


def _proper_folds(nonempty: Sequence[I64Array], scored: Sequence[I64Array], targets: F64Array, binary: bool) -> Sequence[I64Array]:
    """The folds a cell's frozen-probability metrics are read on: every nonempty fold where the folds that leaves beside
    ``scored`` (its subset, the other metrics') differ from ``scored`` by no count of 1 to 20 in rows, cases or
    controls; ``scored`` otherwise. Each row of a cell shows its metric's own counts, so two rows' counts differ by the
    folds between them, and that difference is a count the report shows by subtraction."""
    wide_rows, wide_cases = _evaluated_counts(nonempty, targets, binary)
    narrow_rows, narrow_cases = _evaluated_counts(scored, targets, binary)
    left_out = [wide_rows - narrow_rows]
    if binary:
        assert wide_cases is not None and narrow_cases is not None
        left_out += [wide_cases - narrow_cases, (wide_rows - wide_cases) - (narrow_rows - narrow_cases)]
    return nonempty if all(part == 0 or part >= MINIMUM_REPORTED_PARTICIPANTS for part in left_out) else scored


def _held_out_metric(metric: str, targets: F64Array, linear: F64Array, genetic: F64Array, frozen: F64Array, covariates: F64Array) -> float:
    """One arm's held-out metric on one fold's rows (``_report_step``). ``frozen`` holds the rows' frozen calibrated
    logits (the score step's independent development calibration); a metric of them on a fold whose calibration has no
    maximum is NaN."""
    if metric in ("log_loss", "brier", "calibration_intercept", "calibration_slope") and not np.all(np.isfinite(frozen)):
        return float("nan")
    if metric in ("log_loss", "brier"):
        return float(np.mean(probability_calibration.proper_scores(targets, frozen)[metric]))
    if metric in ("calibration_intercept", "calibration_slope"):
        # Descriptive: the outcome's logistic fit on the frozen logit over the evaluated rows (intercept 0 and slope 1
        # where the frozen probabilities are calibrated there). It never changes a frozen probability or a proper score.
        try:
            intercept, slope = held_out_comparison.logistic_recalibration_coefficients(frozen, targets)
        except held_out_comparison.RecalibrationSeparationError:
            return float("nan")
        return float(intercept if metric == "calibration_intercept" else slope)
    if metric == "auc":
        return auc_components(targets, linear).auc
    if metric == "pgs_auc":
        return auc_components(targets, genetic).auc
    if metric == "r2":
        return _partial_r2(targets, linear, np.ones((targets.shape[0], 1)))
    if metric == "pgs_partial_r2":
        return _partial_r2(targets, genetic, covariates)
    raise ValueError(f"unknown metric {metric!r}")


def _paired(
    metric: str, targets: F64Array, linear: F64Array, genetic: F64Array, frozen: F64Array, covariates: F64Array, families: I64Array
) -> PairedComparison:
    """One fold's paired comparison of the arms on its held-out rows, from their linear predictors, genetic scores and
    frozen calibrated logits, each [arms, rows] (positive favours ``with_sv``). A comparison of the frozen probabilities
    on a fold whose calibration has no maximum (the score step's independent development calibration) is NaN."""
    baseline, candidate = ARMS.index("snv_indel"), ARMS.index("with_sv")
    if metric in ("log_likelihood_gain", "brier_reduction"):
        if not np.all(np.isfinite(frozen)):
            return PairedComparison(float("nan"), float("nan"), float("nan"))
        return probability_calibration.paired_frozen_gain(
            targets, frozen[baseline], frozen[candidate], families, metric="log_loss" if metric == "log_likelihood_gain" else "brier"
        )
    if metric == "delta_auc":
        return paired_delta_auc(targets, linear[baseline], linear[candidate], families)
    if metric == "delta_pgs_auc":
        return paired_delta_auc(targets, genetic[baseline], genetic[candidate], families)
    if metric == "delta_pgs_partial_r2":
        return paired_delta_r2(
            _residualized(targets, covariates), _residualized(genetic[baseline], covariates), _residualized(genetic[candidate], covariates), families
        )
    raise ValueError(f"unknown comparison {metric!r}")


def _report_step(run: _Run, directory: Path) -> dict[str, Any]:
    """Held-out accuracy of both arms (``accuracy.tsv``) and the paired SNV+SV against SNV/indel comparison
    (``comparison.tsv``), per trait in cells of all held-out rows, fold, ancestry and half.

    Every value is each fold's own model evaluated on its own fold, then weighted by the fold's rows. A disease's linear
    predictor is not a probability (FIT_TRAIT_TYPE): each arm's is made one by a frozen logistic calibration, fold k's
    fitted only on its reserved development families. Neither those labels nor their scores depend on outer test labels. A disease is scored by the AUC of its linear predictor, the AUC
    of its genetic score alone, the log loss and Brier score of its frozen probabilities, and the calibration intercept
    and slope of the outcome on its frozen logit over the evaluated rows (descriptive); a quantitative trait by the r^2
    of its linear predictor and the partial r^2 of its genetic score given the trait's covariates. The comparison is
    paired on the same people: a disease's DeLong delta AUC of the linear predictors and of the genetic scores (with
    family-clustered variance) and its family-clustered gain in log likelihood and reduction in Brier score of the
    frozen probabilities; a quantitative trait's family-clustered delta partial r^2 of the genetic scores. Positive
    favours ``with_sv``. A frozen-probability metric is read on every fold of a cell, one with a single outcome class
    included; the others on the folds ``_scored_folds`` keeps. Across multiple folds the estimate and conditional
    standard error are descriptive: shared-training covariance is not estimated, so no combined z or p-value is emitted.
    """
    cohort = _read_cohort(run)
    scores = _read_scores(run)
    fold_count = run.config.fold_count
    accuracy: list[tuple[Any, ...]] = [ACCURACY_COLUMNS]
    comparison: list[tuple[Any, ...]] = [COMPARISON_COLUMNS]
    for trait, name in enumerate(cohort.trait_names):
        binary = cohort.trait_types[trait] == TraitType.BINARY
        covariates = cohort.covariates[:, cohort.covariate_columns[trait]]
        targets = cohort.targets[:, trait]
        linear = scores.held_out_linear_predictor[:, :, trait]
        genetic = scores.held_out_genetic[:, :, trait]
        # Frozen probabilities are produced by this outer model's reserved development families.
        # Other folds' out-of-fold models trained on this fold's labels: calibrating from
        # those models would indirectly reuse evaluation labels and is not outer validation.
        frozen = np.full(linear.shape, np.nan) if scores.held_out_calibrated_logits is None else scores.held_out_calibrated_logits[:, :, trait]
        metric_names = ("auc", "pgs_auc", "log_loss", "brier", "calibration_intercept", "calibration_slope") if binary else ("r2", "pgs_partial_r2")
        comparison_names = ("delta_auc", "delta_pgs_auc", "log_likelihood_gain", "brier_reduction") if binary else ("delta_pgs_partial_r2",)
        if binary and not run.config.calibration_fraction:
            metric_names = ("auc", "pgs_auc")
            comparison_names = ("delta_auc", "delta_pgs_auc")
        for cell, level, held in _cells(cohort, trait, fold_count):
            count = held.count
            cases = int(sum(int(targets[rows].sum()) for rows in held.folds)) if binary else None
            counts = (reportable_count(count), _cases_text(count, cases))
            shown = _shown(count, cases)
            scored = _scored_folds(held, targets, binary) if shown else []
            proper = _proper_folds([rows for rows in held.folds if rows.shape[0]], scored, targets, binary) if shown else []

            def evaluated(metric: str) -> tuple[Sequence[I64Array], tuple[str, str]]:
                # The folds a metric is read on and the counts its row shows; none where those counts are not
                # themselves shown (a metric read on fewer than MINIMUM_REPORTED_PARTICIPANTS rows, cases or controls).
                if not shown:
                    return [], counts
                selected = proper if metric in PROPER_SCORES else scored
                used, used_cases = _evaluated_counts(selected, targets, binary)
                return (selected if _shown(used, used_cases) else []), (reportable_count(used), _cases_text(used, used_cases))
            for arm_index, arm in enumerate(ARMS):
                for metric in metric_names:
                    selected, metric_counts = evaluated(metric)
                    values = [
                        (rows.shape[0], _held_out_metric(metric, targets[rows], linear[arm_index, rows], genetic[arm_index, rows], frozen[arm_index, rows], covariates[rows]))
                        for rows in selected
                    ]
                    accuracy.append((name, arm, cell, level, *metric_counts, *((metric, f"{_fold_weighted(values):.6g}") if values else ("", ""))))
            # The paired comparison per cell of all held-out rows and per ancestry group (a fold is one term of
            # every cell's sum, and the halves are a measurement split rather than a population).
            if cell not in ("all", "ancestry"):
                continue
            for metric in comparison_names:
                selected, metric_counts = evaluated(metric)
                per_fold = [
                    (rows.shape[0], _paired(metric, targets[rows], linear[:, rows], genetic[:, rows], frozen[:, rows], covariates[rows], cohort.families[rows]))
                    for rows in selected
                ]
                if not per_fold:
                    comparison.append((name, cell, level, *metric_counts, "", "", "", "", "", ""))
                    continue
                combined = _fold_combined(per_fold)
                if len(per_fold) > 1:
                    z_text, p_text, inference = "", "", "descriptive_cross_fit"
                elif np.isfinite(combined.z_score):
                    z_text = f"{combined.z_score:.6g}"
                    p_text = f"{2.0 * stats.norm.sf(abs(combined.z_score)):.6g}"
                    inference = "conditional_on_fitted_models"
                else:
                    z_text, p_text, inference = "", "", "undefined"
                comparison.append((
                    name, cell, level, *metric_counts, metric, f"{combined.estimate:.6g}", f"{combined.standard_error:.6g}",
                    z_text, p_text, inference,
                ))
    for file_name, lines in (("accuracy.tsv", accuracy), ("comparison.tsv", comparison)):
        with (directory / file_name).open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle, delimiter="\t").writerows(lines)
    exportable = ["accuracy.tsv", "comparison.tsv"]
    _write_json(directory / EXPORTABLE_FILE, exportable)
    return {
        "files": exportable,
        "comparison_inference": (
            "Standard errors are conditional on fitted models. Multi-fold comparisons are descriptive: "
            "shared-training covariance is not estimated, so combined z and p-values are omitted."
        ),
    }


def check_exportable(path: Path) -> None:
    """Refuse a report table that shows a participant count of 1 to 20, or a value for a cell below
    MINIMUM_REPORTED_PARTICIPANTS (any column of PARTICIPANT_COUNT_COLUMNS the table has)."""
    for row in _read_table(path):
        shown_value = any(row.get(column) for column in ("value", "estimate"))
        for column in PARTICIPANT_COUNT_COLUMNS:
            count = row.get(column)
            if count is None or count == "":
                continue
            if not count.isdigit():
                if shown_value:
                    raise ValueError(f"{path}: a suppressed cell shows a value.")
                continue
            if 0 < int(count) < MINIMUM_REPORTED_PARTICIPANTS or (shown_value and int(count) < MINIMUM_REPORTED_PARTICIPANTS):
                raise ValueError(f"{path}: a count below {MINIMUM_REPORTED_PARTICIPANTS} is shown.")


EXPORT_SOURCES = ("report", "effects")
"""The steps whose files the export step may stage, each listing its own in its EXPORTABLE_FILE."""


def _export_step(run: _Run, directory: Path) -> dict[str, Any]:
    exportable = {
        name: run.output(step) / name
        for step in EXPORT_SOURCES
        for name in json.loads((run.output(step) / EXPORTABLE_FILE).read_text(encoding="utf-8"))
    }
    for name in run.config.export:
        if name not in exportable:
            raise ValueError(f"{name!r} is not a report output; only {sorted(exportable)} can be staged.")
        check_exportable(exportable[name])
        shutil.copy2(exportable[name], directory / name)
    return {"staged": list(run.config.export), "sent": "nothing: moving a staged file out of the workspace is the user's action"}


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


_STEPS = (
    _Step(
        "samples",
        (),
        _samples_inputs,
        (cohort_module, sample_crosswalk, dosage_store, _samples_step, store_half_samples, _calibration_pairs, calibrating_pairs, read_imputation_reference_ids),
        _samples_step,
    ),
    # The store reads only the samples step's half manifests and groups (``_store_sample_layout``), so it follows samples
    # directly and is keyed by them, not by phenotypes, cohort or the folds.
    _Step(
        "store",
        (),
        _store_inputs,
        (
            store_converter, dosage_store, variant_typing, _store_step, _store_gates, _convert_chromosome, _popped_chromosome_sites,
            _plain_chromosome_sites, strata_complexity_legend, _strata_names, read_strata_sites, read_popped_sites, _checked_genetic_positions, read_bubble_paths, path_counts,
            _chromosome_spellings, read_plain_sites, read_genetic_map, interpolated_genetic_positions, read_tandem_repeats,
            read_gene_annotation, store_core_spans, store_variant_classes, store_group_first, _decode, _decode_tasks, _gather_calibration,
        ),
        _store_step,
    ),
    # The targets come from the occasion model (``phenotype_measurement``, on ``scale_mixture_ep``'s lattice), so a change
    # to either is a change to the phenotypes, not only a change to ``all_of_us``.
    _Step("phenotypes", (), _phenotype_inputs, (all_of_us, phenotype_measurement, scale_mixture_ep, _phenotypes_step), _phenotypes_step),
    _Step("cohort", ("samples", "phenotypes"), lambda config: config.calibration_fraction, (cohort_module, _cohort_step, trait_table), _cohort_step),
    _Step(
        "measurement",
        ("samples", "cohort", "store"),
        lambda config: None,
        # The measurement model it fits and loads (workspace_bindings) is measurement_model's, which the key covers.
        (store_converter, dosage_store, _measurement_step, _group_code_moments, _group_calibration, rewrite_recalibrated_store, *module_closure(measurement_model)),
        _measurement_step,
    ),
    # The fit runs fit_model and everything it imports (the engine, the store reader, the scorer the artifact holds).
    _Step("fit", ("cohort", "measurement"), lambda config: (config.seed, config.disease_likelihood, config.annotated_prior), (_fit_step, structural_records, arm_log_reliability, trait_request, trait_seed, *module_closure(fit_model)), _fit_step),
    _Step("score", ("cohort", "fit"), lambda config: None, (_score_step, score_batches, structural_records, model_part), _score_step),
    _Step(
        "effects", ("cohort", "fit", "score"), lambda config: None,
        (_effects_step, variant_effects, length_bins, genetic_variance_decomposition, _shown), _effects_step,
    ),
    _Step(
        "report",
        ("cohort", "score"),
        lambda config: None,
        (
            _report_step, _cells, _scored_folds, _held_out_metric, _paired, _fold_combined, _fold_weighted, _partial_r2, _shown, _cases_text,
            held_out_comparison, probability_calibration,
        ),
        _report_step,
    ),
    _Step("export", ("report", "effects"), lambda config: config.export, (_export_step, check_exportable), _export_step),
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
