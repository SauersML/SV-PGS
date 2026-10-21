"""The pipeline that runs inside the All of Us workspace: imputed batches to held-out scores.

One driver, restartable step by step. A step writes into
``<run>/<step>.partial`` and is complete once that directory, holding ``_COMPLETE.json`` with the
step's key, is renamed to ``<run>/<step>``. The key is a digest of the configuration the step
reads, the keys of the steps it depends on, and the source of the code it runs. A rerun:

- skips every step whose key is unchanged;
- resumes a partial step from its sub-checkpoints (the store's decoded batches and finished
  chromosomes);
- moves a step whose key changed aside as ``<step>.superseded-<key>``.

Nothing is deleted except the store step's own decoded batch files and the direct_sv step's blocks
of rows, once their chromosome is written.

Steps:

1. ``samples``: the halves' typed sample manifests from the batch headers. Then the cohort rows
   (``cohort.resolve_cohort_rows``: the truth row wins, an imputed sample maps to a person only
   through the crosswalk, KING duplicates of a truth genome are dropped), and the trait-agnostic
   kinship folds over both halves, frozen before any phenotype exists.
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
5b. ``direct_sv``: the direct read-based channel. Every kept record of the GATK-SV call set becomes
   a store row of its own beside the imputed records, its no-calls filled from a paired imputed DS
   or its observed mean, with the measurement model's no-truth reliability; the store and the model
   the fit reads cover the merged rows. With ``direct_sv`` null the step adds nothing and says so.
6. ``fit``: both arms (``ARMS``: SNVs and short indels alone, then with every structural record)
   of every trait, each trait's fold models and its all-rows model in one call, each projecting
   out its own trait's covariates. Each (arm, trait) model is saved as it finishes, so a rerun
   resumes at the first one not on disk.
7. ``score``: every model of both arms scored for the whole cohort from one read of the store.
   Each person's held-out prediction is the one from their own fold's model.
8. ``effects``: each trait's all-rows model's per-record effects (kept in the workspace) and
   their summaries by variant class and length, the learned prior's among them.
9. ``report``: held-out accuracy of both arms and their paired comparison per trait, in cells by
   fold, ancestry and half, with any count of 1 to 20 suppressed (and any the shown ones would give
   back); each arm's held-out partial r^2 beside a baseline method's on the same people, with
   bootstrap intervals, paired differences and their meta-analysis across traits.
10. ``export``: stages in ``<run>/export`` the report and effects summaries the user named in the
    config, and nothing else.

Nothing is ever sent anywhere. Moving a staged file out of the workspace is the user's action.
Everything here reads and writes participant-derived data, so every output stays in the
workspace. Progress is visible only as empty marker files whose names
hold the step and the event.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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
import re
import shutil
import time
from types import ModuleType
from typing import Any, Callable, Iterator, Mapping, Sequence

from cyvcf2 import VCF
import numpy as np
import pandas as pd
from scipy import special, stats

from sv_pgs import (
    all_of_us,
    cohort as cohort_module,
    dosage_store,
    fit_model,
    gatksv_store_rows,
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
    reportable_count,
    reportable_partition,
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
    resolve_cohort_rows,
)
from sv_pgs.artifact import CERTIFICATE_STATUS, cohort_digest, model_part, model_subset
from sv_pgs.copy_number import allele_count_decode
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.draw_laws import tile_rows
from sv_pgs.fast_scoring import SIGNED_CODE_OFFSET
from sv_pgs.eb_monitor import EvaluationRows
from sv_pgs.fit_model import FitRequest
from sv_pgs.model_acceptance import acceptance, require_scorable, unscorable_codes
from sv_pgs.held_out_comparison import PairedComparison, auc_components, paired_delta_auc, paired_delta_r2
from sv_pgs.config import TraitType, VariantClass
from sv_pgs.dosage_store import (
    CODES_PER_DOSAGE,
    DEFAULT_INNER_CHUNK_ROWS,
    MISSING_CODE,
    VARIANT_CLASS_LEGEND,
    VARIANT_CLASSES,
    Codec,
    DosageStore,
    HalfSamples,
    VariantTable,
    chromosome_number,
    dosage_array_directory,
    open_column,
    read_manifest,
    sites_md5,
    statistic_column_directory,
    write_column,
    write_half_codes,
    write_variant_columns,
)
from sv_pgs.gatksv_source import GatksvSource
from sv_pgs.gatksv_store_rows import RELIABILITY_SOURCES, ImputedSvRecords, PairEvidence, gatksv_block_rows, gatksv_sites, one_to_one
from sv_pgs.sample_crosswalk import ABSENT_COLUMN, SampleCrosswalk, source_columns_for_store_samples, store_research_ids
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
from sv_pgs.sv_fusion import MINIMUM_SIZE_RATIO, SvSites, candidate_pairs, sequence_resolved_sites
from sv_pgs.variant_typing import COMMUNITY_SV_MINIMUM_LENGTH, alt_is_symbolic_or_breakend, is_sequence_allele, variant_class_and_length
from sv_pgs.fit_checkpoint import checkpoint_timings

LOGGER = logging.getLogger(__name__)

STEP_NAMES = ("samples", "store", "phenotypes", "cohort", "measurement", "direct_sv", "fit", "score", "effects", "report", "export")
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
REGIONS_DIRECTORY = "regions"
"""The run directory's directory of its regions' runs (``WorkspaceConfig.fit_regions``), one directory each by name."""
REGION_NAME = re.compile(r"[a-z0-9_]+")
"""A fit region's name, which names its directory (and its mirror's): lowercase letters, digits and underscores."""
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
    "direct_sv",
)
KEY_HINTS = {
    "gene_annotation": (
        "gene_annotation is the public GENCODE v38 basic annotation GTF, gencode.v38.basic.annotation.gtf.gz from "
        "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/ (GRCh38, chr-prefixed contigs)."
    ),
    "genetic_map": (
        "genetic_map is null where every imputed record carries INFO/CM. For batches whose records lack it (AoU's popped "
        "batches declare it but carry it on no record) it is a path pattern over {chromosome} to the public Beagle GRCh38 PLINK maps, plink.GRCh38.map.zip from "
        "https://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/plink.GRCh38.map.zip, unzipped: e.g. "
        "<dir>/no_chr_in_chrom_field/plink.{chromosome}.GRCh38.map."
    ),
    "imputation_reference_ids": (
        "imputation_reference_ids is a one-column file headed research_id listing everyone whose genome was in the "
        "imputation's reference panel, or null; with truth_calls set and it null, no calibration pair is used."
    ),
    "direct_sv": (
        "direct_sv is null (no direct read-based rows) or {\"gatksv_calls\": a path pattern over {chromosome} to the CDR's "
        "GATK-SV call set, one VCF per chromosome, e.g. "
        "${CDR_STORAGE_PATH}/wgs/short_read/structural_variants/vcf/full/AoU_srWGS_SV.v8.{chromosome}.vcf.gz, "
        "or null; \"read_depth_cnv_calls\", \"ctyper_copy_numbers\", \"cn_estimator_copy_numbers\" and "
        "\"truth_copy_numbers\": null, as none of them is implemented}."
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


DIRECT_SV_KEYS = ("gatksv_calls", "read_depth_cnv_calls", "ctyper_copy_numbers", "cn_estimator_copy_numbers", "truth_copy_numbers")
"""The ``direct_sv`` config's keys: the GATK-SV call set,
the one source implemented, then the optional sources, each of which must be null."""


@dataclass(frozen=True, slots=True)
class PanelInputs:
    """The pre-registered panel test, frozen before any test phenotype is
    read: the outer fold its per-trait tests are read on, and each trait's non-negative weight, proportional to the
    trait's expected z under a power model that uses no SV biology."""

    primary_fold: int
    weights: tuple[tuple[str, float], ...]

    @classmethod
    def from_mapping(cls, raw: Any, phenotypes: Sequence[str], fold_count: int) -> PanelInputs | None:
        if raw is None:
            return None
        if not isinstance(raw, Mapping) or set(raw) != {"primary_fold", "weights"}:
            raise ValueError("panel is null or {primary_fold, weights}.")
        fold = raw["primary_fold"]
        if isinstance(fold, bool) or not isinstance(fold, (int, np.integer)) or not 0 <= fold < fold_count:
            raise ValueError(f"panel primary_fold must be an integer fold in [0, {fold_count}).")
        weights = raw["weights"]
        if not isinstance(weights, Mapping) or set(weights) != set(phenotypes):
            raise ValueError("panel weights name every disease and trait of the run, each once.")
        values = [weights[name] for name in phenotypes]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0 for value in values):
            raise ValueError("panel weights are finite and non-negative.")
        if not sum(values) > 0:
            raise ValueError("panel weights cannot all be 0.")
        return cls(primary_fold=int(fold), weights=tuple((name, float(weights[name])) for name in phenotypes))


@dataclass(frozen=True, slots=True)
class DirectSvInputs:
    """The ``direct_sv`` config: the GATK-SV call set as a path pattern over ``{chromosome}``, one VCF per store
    chromosome (samples named by research ID), or None, which adds no direct row."""

    gatksv_calls: str | None

    @classmethod
    def from_mapping(cls, raw: Any) -> DirectSvInputs | None:
        """None for a null ``direct_sv``; otherwise every DIRECT_SV_KEYS key is required, and every source but the
        GATK-SV call set must be null: a non-null one is refused, never ignored."""
        if raw is None:
            return None
        if not isinstance(raw, Mapping) or set(raw) != set(DIRECT_SV_KEYS):
            raise ValueError(f"direct_sv is null or names exactly {list(DIRECT_SV_KEYS)}. {KEY_HINTS['direct_sv']}")
        for key in DIRECT_SV_KEYS[1:]:
            if raw[key] is not None:
                raise ValueError(
                    f"direct_sv.{key} is not implemented: the driver reads only the GATK-SV call set (gatksv_calls); set {key} to null."
                )
        calls = raw["gatksv_calls"]
        if calls is not None and "{chromosome}" not in str(calls):
            raise ValueError("direct_sv.gatksv_calls is a path pattern over {chromosome}: the call set is one VCF per chromosome.")
        return cls(None if calls is None else str(calls))

    def gatksv_path(self, chromosome: str) -> Path:
        if self.gatksv_calls is None:
            raise ValueError("this run reads no GATK-SV call set.")
        return Path(self.gatksv_calls.format(chromosome=chromosome))


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
    over ``{chromosome}`` to a PLINK-format map, read for a chromosome whose records (popped or plain)
    do not all carry INFO/CM, or null.

    ``direct_sv`` is the direct read-based channel's sources (``DirectSvInputs``), or None. With it None or its
    GATK-SV call set None, the ``direct_sv`` step adds no row and says so.
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
    direct_sv: DirectSvInputs | None
    genome_build: str = "unspecified"
    disease_likelihood: str = "gaussian_working"
    calibration_fraction: float = 0.0
    annotated_prior: bool = True
    length_by_class: bool = False
    """The hierarchical length-by-type prior (``fit_model.FitRequest.length_by_class``): each variant class's own learned
    deviation from the shared length smooth, shrunk to it where the class's records resolve little; False (the default)
    keeps one length curve for every class, the prior of the runs so far."""
    ld_score_prior: bool = False
    """The LD function (``fit_model.FitRequest.ld_score_prior``): each record's Stage 0 LD score as a learned smooth of its
    prior variance; False (the default) leaves it out, as in the runs so far."""
    panel: PanelInputs | None = None
    cross_fit: bool = True
    """True: one model per fold, each scored on the fold it left out, and an all-rows model per trait. False: a single
    train/test split, one model per trait trained on every fold but the test fold (the panel's primary fold, else fold
    0) and scored on it, which also serves as the trait's released model: 1 / fold_count of the rows test, the rest train."""
    fit_minimum_minor_allele_frequency: float = 0.0
    fit_minimum_r2: float = 0.0
    """The fit leaves out every record whose minor allele frequency over the store's samples (its code sums) or whose
    reliability (the measurement model's r^2) is below these; 0 keeps every record. A fit's cost grows with
    its records: the All of Us chr22 store's 1.8M candidates made a 364 GB LD store and a 10-minute pass on an A100."""
    fit_variant_sites: str | None = None
    """A site list (``listed_sites``): the fit keeps only the single-base substitutions at its sites and every structural
    record, each still subject to the floors above; None keeps every record. The floors alone left 8M of the All of Us
    genome's 128M records, 67 times chr22's, whose filtered fit took half an hour an arm on two A100s; HapMap3's 1.05M
    sites are the usual polygenic-score set."""
    fit_region: tuple[int, int, int] | None = None
    """A window of one store chromosome, (its number, the first position, the last position) in the store's build, both
    ends included: the fit keeps only the records whose position lies in it (``region_records``), SNVs, short indels and
    structural records alike, each still subject to the floors and the site list above; None keeps every chromosome
    whole. A regional score is fitted on one locus's records alone: the APOE locus's for Alzheimer's disease."""
    fit_regions: tuple[tuple[str, tuple[int, int, int]], ...] | None = None
    """Named windows, (name, fit_region's window), each fitted as its own model set: a region's fit, score, effects,
    report and export steps run in REGIONS_DIRECTORY/<name> of the run directory, as a run whose fit_region is its
    window (``region_runs``), on the run's steps before the fit, which every region shares (``run_pipeline``). Names
    are distinct and REGION_NAME's; windows may overlap, each fitted alone. None fits the run's own models; a run sets
    fit_region or fit_regions, never both. A window's fit is minutes on a small machine, so one job fits a panel of
    loci on one store."""
    test_fold: int | None = None
    """A single split's validation fold for method iteration, or None for the final split. The final split tests on its
    reserved fold (the panel's primary fold, else fold 0: ``model_folds``) and trains on the rest; with ``test_fold`` the
    model trains on every fold but these two and is scored on ``test_fold``, so the reserved fold is neither trained on
    nor seen while priors, annotations and fit settings are chosen, and stays untouched for the final numbers. Only the
    steps that read labels see it (``_split_cohort``): the cohort step's masks are the final split's, and the store,
    phenotypes, cohort, measurement and direct-SV steps keep their keys (measurement and direct SV read only the
    genotypes of the final split's fit rows), while the fit's key names the fold, so the fit and every step after it are
    the validation run's own."""
    baseline_directory: Path | None = None
    """Where the report reads other methods' held-out scores (``baselines``), or None for the run directory's
    BASELINE_DIRECTORY. The run mirror carries completed steps, not other files of the run directory, so a baseline
    written after a job's driver (benchmarks/bench_real/aou_ldpred2.sh) is kept in the workspace bucket's mount and
    named here for the next job's report."""
    locus_panel: str | None = None
    """A curated panel of genome windows (benchmarks/bench_real/locus_panel.json's format: its ``loci``, each with a
    ``name``, ``kind``, store ``chromosome``, first and last position and the ``traits`` it bears on), whose genetic
    variance the effects step reports per trait, arm and class (``locus_variance``), or None for none."""

    @classmethod
    def read(cls, path: str | Path) -> WorkspaceConfig:
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> WorkspaceConfig:
        missing = sorted(set(CONFIG_KEYS) - set(raw))
        unknown = sorted(set(raw) - set(CONFIG_KEYS) - {
            "genome_build", "disease_likelihood", "calibration_fraction", "annotated_prior", "length_by_class", "ld_score_prior", "panel", "cross_fit",
            "fit_variant_sites", "fit_region", "fit_regions", "test_fold", "locus_panel", "baseline_directory", *_FIT_FILTER_KEYS,
        })
        if missing or unknown:
            hints = "".join(f" {KEY_HINTS[key]}" for key in missing if key in KEY_HINTS)
            raise ValueError(f"the run config lacks {missing} and has unknown keys {unknown}; every key is required." + hints)
        if not isinstance(raw.get("calibration_fraction", 0.0), (int, float)) or not 0 <= raw.get("calibration_fraction", 0.0) < 1:
            raise ValueError("calibration_fraction must be in [0, 1)")
        if raw.get("disease_likelihood", "gaussian_working") not in ("gaussian_working", "bernoulli"):
            raise ValueError("disease_likelihood must be gaussian_working or bernoulli")
        if not isinstance(raw.get("annotated_prior", True), bool):
            raise ValueError("annotated_prior must be a boolean")
        for key in ("length_by_class", "ld_score_prior"):
            if not isinstance(raw.get(key, False), bool):
                raise ValueError(f"{key} must be a boolean")
        if not isinstance(raw.get("cross_fit", True), bool):
            raise ValueError("cross_fit must be a boolean")
        for key in _FIT_FILTER_KEYS:
            value = raw.get(key, 0.0)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
                raise ValueError(f"{key} must be a number in [0, 1)")
        if raw.get("fit_variant_sites") is not None and (not isinstance(raw["fit_variant_sites"], str) or not raw["fit_variant_sites"]):
            raise ValueError("fit_variant_sites must be a site list's path or null")
        for key in ("locus_panel", "baseline_directory"):
            if raw.get(key) is not None and (not isinstance(raw[key], str) or not raw[key]):
                raise ValueError(f"{key} must be a path or null")
        test_fold = raw.get("test_fold")
        if test_fold is not None:
            reserved = 0 if raw.get("panel") is None else raw["panel"].get("primary_fold")
            if raw.get("cross_fit", True) or isinstance(test_fold, bool) or not isinstance(test_fold, (int, np.integer)):
                raise ValueError("test_fold is an integer fold of a single split (cross_fit false), or null.")
            # Some fold besides the reserved and the test fold trains.
            if not 0 <= test_fold < raw["fold_count"] or test_fold == reserved or raw["fold_count"] <= len({reserved, test_fold}):
                raise ValueError(f"test_fold must be a fold in [0, fold_count) other than the reserved test fold {reserved}, with fold_count >= 3 so some fold trains.")
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
            raise ValueError("codec must be 'raw' or 'zstd'.")
        chromosomes = tuple(str(chromosome) for chromosome in raw["chromosomes"])
        for chromosome in chromosomes:
            chromosome_number(chromosome)
        if not chromosomes or len({chromosome_number(chromosome) for chromosome in chromosomes}) != len(chromosomes):
            raise ValueError("chromosomes must list each store chromosome once.")
        numbers = {chromosome_number(chromosome) for chromosome in chromosomes}
        region = None if raw.get("fit_region") is None else _fit_window(raw["fit_region"], numbers, "fit_region")
        regions = raw.get("fit_regions")
        if regions is not None:
            if region is not None:
                raise ValueError("a run sets fit_region (its own models' window) or fit_regions (a model set per named window), never both.")
            if not isinstance(regions, Sequence) or isinstance(regions, str) or not regions or not all(isinstance(entry, Mapping) for entry in regions):
                raise ValueError("fit_regions is null or a non-empty list of {name, chromosome, start, end}.")
            names = [entry.get("name") for entry in regions]
            if any(not isinstance(name, str) or not REGION_NAME.fullmatch(name) for name in names) or len(set(names)) != len(names):
                raise ValueError(f"each of fit_regions has a distinct name matching {REGION_NAME.pattern}: it names the region's directory.")
            regions = tuple(
                (name, _fit_window({key: value for key, value in entry.items() if key != "name"}, numbers, f"fit_regions[{index}]"))
                for index, (name, entry) in enumerate(zip(names, regions))
            )
        try:
            research_column, sequencing_column = (str(column) for column in raw["crosswalk_columns"])
            first_column, second_column, kinship_column = (str(column) for column in raw["relatedness_columns"])
        except ValueError as error:
            raise ValueError("crosswalk_columns names (research, sequencing); relatedness_columns (first, second, kinship).") from error
        if (raw["strata_directory"] is None) != (raw["bubble_split"] is None):
            raise ValueError(
                "strata_directory and bubble_split are set together (popped batches) or null together (plain imputed VCF/BCFs)."
            )
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
            direct_sv=DirectSvInputs.from_mapping(raw["direct_sv"]),
            genome_build=str(raw.get("genome_build", "unspecified")),
            disease_likelihood=str(raw.get("disease_likelihood", "gaussian_working")),
            calibration_fraction=float(raw.get("calibration_fraction", 0.0)),
            annotated_prior=raw.get("annotated_prior", True),
            length_by_class=raw.get("length_by_class", False),
            ld_score_prior=raw.get("ld_score_prior", False),
            panel=PanelInputs.from_mapping(raw.get("panel"), diseases + traits, int(raw["fold_count"])),
            cross_fit=raw.get("cross_fit", True),
            fit_minimum_minor_allele_frequency=float(raw.get("fit_minimum_minor_allele_frequency", 0.0)),
            fit_minimum_r2=float(raw.get("fit_minimum_r2", 0.0)),
            fit_variant_sites=raw.get("fit_variant_sites"),
            fit_region=region,
            fit_regions=regions,
            test_fold=None if test_fold is None else int(test_fold),
            locus_panel=raw.get("locus_panel"),
            baseline_directory=None if raw.get("baseline_directory") is None else Path(raw["baseline_directory"]),
        )

    def region_runs(self) -> tuple[tuple[str, WorkspaceConfig], ...]:
        """Each of ``fit_regions`` by name, as the run its steps from the fit on make: in REGIONS_DIRECTORY/<name> of the
        run directory (and of the marker and baseline directories), with the region's window as its fit_region. None
        without fit_regions."""
        return tuple(
            (name, dataclasses.replace(
                self, run_directory=self.run_directory / REGIONS_DIRECTORY / name, fit_region=window, fit_regions=None,
                marker_directory=None if self.marker_directory is None else self.marker_directory / REGIONS_DIRECTORY / name,
                baseline_directory=None if self.baseline_directory is None else self.baseline_directory / REGIONS_DIRECTORY / name,
            ))
            for name, window in self.fit_regions or ()
        )

    def baselines(self) -> Path:
        """The directory of other methods' held-out scores (``baseline_directory``)."""
        return self.run_directory / BASELINE_DIRECTORY if self.baseline_directory is None else self.baseline_directory

    def model_folds(self) -> tuple[int, ...]:
        """Each trait's models by the fold they leave out, in model order: every fold then fold_count (the all-rows
        model, which leaves out no one) when cross-fitting; the test fold alone for a single split."""
        if self.cross_fit:
            return tuple(range(self.fold_count + 1))
        return (0 if self.panel is None else self.panel.primary_fold,)

    def truth_path(self, chromosome: str) -> Path:
        if self.truth_calls is None:
            raise ValueError("this run has no truth calls.")
        return Path(self.truth_calls.format(chromosome=chromosome))

    def genetic_map_path(self, chromosome: str) -> Path:
        if self.genetic_map is None:
            raise ValueError(
                f"{chromosome}: the batches' records lack INFO/CM, so their genetic positions come from genetic_map, which is "
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


_FIT_FILTER_KEYS = ("fit_minimum_minor_allele_frequency", "fit_minimum_r2")


def _fit_window(window: Any, chromosomes: set[int], key: str) -> tuple[int, int, int]:
    """A fit window (``WorkspaceConfig.fit_region``) from the config's {chromosome, start, end}: the number of one of
    the store's ``chromosomes`` and the window's first and last positions, integers never rounded or coerced."""
    if (
        not isinstance(window, Mapping) or set(window) != {"chromosome", "start", "end"}
        or any(isinstance(window[name], bool) or not isinstance(window[name], (int, np.integer)) for name in window)
        or window["chromosome"] not in chromosomes or not 1 <= window["start"] <= window["end"]
    ):
        raise ValueError(
            f"{key} is {{chromosome, start, end}}: the number of one of the store's chromosomes and the window's first and last "
            "positions (1-based, both included, in the store's build); never rounded or coerced."
        )
    return int(window["chromosome"]), int(window["start"]), int(window["end"])


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
    """An empty marker file named by time, step and event, and never by any value computed from data.

    A marker is a status signal only, so one that cannot be written is logged and the run goes on: on the bucket mount,
    a store fan-out's tasks touching the same name in the same second failed with "Stale file handle", which ended two
    chromosome tasks before they converted anything."""
    if config.marker_directory is None:
        return
    try:
        config.marker_directory.mkdir(parents=True, exist_ok=True)
        (config.marker_directory / f"{_utc()}_{step}_{event}").touch()
    except OSError as error:
        LOGGER.warning("workspace pipeline: marker %s_%s not written: %s", step, event, error)


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
    store_chromosomes: tuple[str, ...] | None = None
    """The chromosomes this run's store step converts, or None for every one (``run_pipeline``'s ``store_shard``)."""
    fit_arms: tuple[str, ...] | None = None
    """The arms this run's fit step fits, or None for every one (``run_pipeline``'s ``fit_arms``)."""
    step: str | None = None
    """The step running now (``_run_step``), which decides the cohort it reads (``_read_cohort``); None outside one."""

    def output(self, step: str) -> Path:
        return self.config.run_directory / step

    def summary(self, step: str) -> dict[str, Any]:
        record: dict[str, Any] = json.loads((self.output(step) / COMPLETE_FILE).read_text(encoding="utf-8"))
        return dict(record["summary"])


class StepPart(Exception):
    """A step body did the part of its work its run was given, and the step is not complete: its partial directory holds
    that part for another run (or a later one) to finish."""


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
            # A completed fit was checked when its partial was finalized (below); checking it again here would load
            # every saved model on each later pass, which a resumed score or effects step must not do.
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
    run.step = step.name
    try:
        if step.name == "fit":
            _check_fit_records(run, partial, key)
        summary = step.body(run, partial)
        if step.name == "fit":
            _check_fit_records(run, partial, key)
    except StepPart:
        if step.name == "fit":
            _check_fit_records(run, partial, key)
        _mark(run.config, step.name, "part-done")
        raise
    except BaseException as error:
        _mark(run.config, step.name, f"failed-{type(error).__name__}")
        raise
    finally:
        run.step = None
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
    # fold_count, which no person belongs to: it trains on every observed row and holds none out); a single split
    # (cross_fit false) fits only the test fold's model.
    trait_folds = config.model_folds()
    model_traits = np.repeat(np.arange(len(trait_names)), len(trait_folds))
    model_folds = np.tile(np.array(trait_folds, dtype=np.int64), len(trait_names))
    observed = cohort.observed[:, model_traits]
    training = observed & (folds[:, None] != model_folds[None, :])
    held_out = observed & (folds[:, None] == model_folds[None, :])
    calibration = development_calibration(training, model_traits, model_folds, len(config.diseases), families, config.calibration_fraction, config.seed)
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
    fold_names = {fold: ALL_ROWS_MODEL if fold == config.fold_count else f"fold{fold}" for fold in trait_folds}
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


def development_calibration(
    training: BoolArray, model_traits: I64Array, model_folds: I64Array, disease_count: int, families: I64Array, fraction: float, seed: int,
) -> BoolArray:
    """Each disease model's reserved calibration rows (``probability_calibration.calibration_mask``): whole families of
    its development rows, drawn by the run's seed and the model's fold alone, never its trait.

    Every disease's fold-k families are disjoint from its own training and evaluation rows whatever the draw, so the
    outer validation does not depend on it. Drawn without the trait, two diseases with the same eligible rows reserve
    the same families and keep the same training rows, so the fit fits them together and they share its store passes
    (``stage2_wiring.model_groups``); drawn per trait, no two diseases' fold models shared a training row set."""
    calibration = np.zeros_like(training)
    for model, trait in enumerate(np.asarray(model_traits).tolist()):
        if trait < disease_count:
            calibration[:, model] = probability_calibration.calibration_mask(training[:, model], families, fraction, [seed, int(model_folds[model])])
    return calibration


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
        """The released model of one trait, the last of its models: the all-rows model (ALL_ROWS_MODEL) when
        cross-fitting, else the single split's one model, trained on every fold but the test fold."""
        return int(self.trait_models(trait)[-1])

    def evaluated_folds(self, trait: int, fold_count: int) -> tuple[int, ...]:
        """The folds a trait's held-out predictions cover: those its fold models leave out (never the all-rows model's)."""
        return tuple(sorted(int(fold) for fold in self.model_folds[self.model_traits == trait] if fold < fold_count))


GENOTYPE_STEPS = ("samples", "store", "phenotypes", "cohort", "measurement", "direct_sv")
"""The steps before the fit, which read the cohort step's own masks (``_read_cohort``): measurement and direct SV read
only its fit rows' genotypes, never a label, so a validation split (``WorkspaceConfig.test_fold``) reuses them."""


def _split_cohort(cohort: _Cohort, config: WorkspaceConfig) -> _Cohort:
    """The cohort as a validation split reads it (``WorkspaceConfig.test_fold``): each trait's one model trains on its
    observed rows outside the test fold and the reserved fold, less its disease's calibration families drawn as the cohort
    step draws them (``development_calibration``), and holds out its observed rows in the test fold, named for that fold.
    The final split's cohort is returned as it is."""
    if config.test_fold is None:
        return cohort
    reserved = config.model_folds()[0]
    model_folds = np.full_like(cohort.model_folds, config.test_fold)
    observed = np.isfinite(cohort.targets)[:, cohort.model_traits]
    training = observed & (cohort.folds[:, None] != config.test_fold) & (cohort.folds[:, None] != reserved)
    held_out = observed & (cohort.folds[:, None] == config.test_fold)
    calibration = development_calibration(training, cohort.model_traits, model_folds, len(config.diseases), cohort.families, config.calibration_fraction, config.seed)
    return dataclasses.replace(
        cohort, training=training & ~calibration, held_out=held_out, calibration=calibration, model_folds=model_folds,
        model_names=tuple(f"{cohort.trait_names[trait]}/fold{config.test_fold}" for trait in cohort.model_traits.tolist()),
    )


def _read_cohort(run: _Run) -> _Cohort:
    """The cohort step's cohort, as the running step reads it: a validation split's (``_split_cohort``) in every step
    after GENOTYPE_STEPS and outside the driver (a baseline method reading a finished run)."""
    cohort = _read_cohort_step(run)
    return cohort if run.step in GENOTYPE_STEPS else _split_cohort(cohort, run.config)


def _read_cohort_step(run: _Run) -> _Cohort:
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


def strata_manifest(directory: Path, chromosome: str) -> Path:
    """A contig's strata manifest, ``chrK.json``: its sites and ids md5s, and the md5 of every sidecar file."""
    return directory / f"{chromosome}.json"


def read_strata_sites(directory: Path, chromosome: str, complexity_legend: tuple[str, ...] | None) -> StrataSites:
    """The STRATA_COLUMNS of ``chrK.strata.tsv.gz``, once ``_done/chrK.json`` marks the contig complete, with the
    popped records' sites and ids md5s from the contig's manifest ``chrK.json`` (the strata build's counts, md5s and
    checks).

    ``cx`` is numeric where ``complexity_legend`` is None, and is otherwise coded by that legend, the store's one legend
    over every chromosome (``strata_complexity_legend``).
    """
    done = directory / "_done" / f"{chromosome}.json"
    if not done.is_file():
        raise ValueError(f"the strata sidecar of {chromosome} is not complete: {done} does not exist.")
    manifest = json.loads(strata_manifest(directory, chromosome).read_text(encoding="utf-8"))
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
    """One popped batch file's site fields, checked against the strata sidecar, with INFO/CM NaN where a record has none
    (``_popped_chromosome_sites`` fills those).

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
            refs.append(record.REF)
            alts.append(alt)
            genetic.append(np.nan if centimorgans is None else float(centimorgans))
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
        genetic_position_cm=np.array(genetic, dtype=np.float64),
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
    """Each record's bubble (its index in bubble.split, -1 for a single-path record), K and m."""

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
    """Each record's store class code (an index of VARIANT_CLASSES) and length, by the store's first-match rules.

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
    path = first_half.path(chromosome, first_half.batches[0])
    records = read_popped_sites(path, strata)
    # AoU's popped batches declare INFO/CM but carry it on no record: their genetic positions come from genetic_map, as
    # a plain batch's do.
    record_count = strata.positions.shape[0]
    carried = int(np.isfinite(records.genetic_position_cm).sum())
    if carried == record_count:
        centimorgans, source = records.genetic_position_cm, "INFO/CM"
    else:
        map_path = config.genetic_map_path(chromosome)
        centimorgans = interpolated_genetic_positions(*read_genetic_map(map_path, chromosome), strata.positions)
        source = f"genetic_map {map_path.name}, as INFO/CM is on {carried} of {record_count} records"
    records = dataclasses.replace(records, genetic_position_cm=_checked_genetic_positions(path, centimorgans, source))
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
        facts={"genetic_position": source, "reported_r2_fields": ["INFO"], "probability_checked_batches": batch_count, "imputed_batches": batch_count},
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
    """The chromosome's TR loci table."""
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
        "strata": None if popped is None else [_file_digest(strata_manifest(popped.strata_directory, chromosome)) for chromosome in config.chromosomes],
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
    # A store shard (run.store_chromosomes) converts its own chromosomes and leaves the rest to the shards that hold them;
    # the run that finds every chromosome converted (the run mirror carries each shard's to it) writes the store.
    for chromosome in config.chromosomes if run.store_chromosomes is None else run.store_chromosomes:
        done = directory / "chromosomes" / f"{chromosome}.json"
        if not done.is_file():
            _clear_chromosome(root, chromosome, len(samples.halves), directory)
            _write_json(done, _convert_chromosome(run, directory, root, chromosome, samples, repeats, genes, complexity_legend))
            _remove_own(directory / "batches" / chromosome, directory)
            _mark(config, "store", f"converted-{chromosome}")
    remaining = [chromosome for chromosome in config.chromosomes if not (directory / "chromosomes" / f"{chromosome}.json").is_file()]
    if remaining:
        raise StepPart(f"the store holds {len(config.chromosomes) - len(remaining)} of its chromosomes; {remaining} are another shard's.")
    facts = [json.loads((directory / "chromosomes" / f"{chromosome}.json").read_text(encoding="utf-8")) for chromosome in config.chromosomes]
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
    copies of one group's codes and their squares. Each block's rows are split over the budget's
    CPU threads (numpy's take and reduce release the GIL): on one thread, the whole-genome All of
    Us store's 6e12 fit entries took hours on the GPU VM.
    """
    record_count = store.n_variants
    sums = np.zeros((record_count, group_count), dtype=np.int64)
    squares = np.zeros((record_count, group_count), dtype=np.int64)
    counts = np.bincount(column_groups, minlength=group_count).astype(np.int64)
    rows = max(1, int(budget.host_bytes // 2 // (columns.shape[0] * (1 + 2 * _INT64_BYTES))))
    ranges = [(start, min(start + rows, record_count)) for start in range(0, record_count, rows)]
    reader_budget = dataclasses.replace(budget, host_bytes=budget.host_bytes // 2)
    members = [np.flatnonzero(column_groups == group) for group in range(group_count)]
    # A code's square, exact in uint16 (255**2 < 2**16).
    square_of = np.arange(256, dtype=np.uint16) ** 2
    threads = max(1, budget.cpu_threads)

    def moments(codes: U8Array, start: int, low: int, high: int) -> None:
        for group, member in enumerate(members):
            values = np.take(codes[low:high], member, axis=1)
            sums[start + low : start + high, group] = np.add.reduce(values, axis=1, dtype=np.int64)
            squares[start + low : start + high, group] = np.add.reduce(np.take(square_of, values), axis=1, dtype=np.int64)

    with ThreadPoolExecutor(max_workers=threads) as pool:
        for start, stop, codes in store.iter_codes(ranges, columns, reader_budget):
            step = -(-(stop - start) // threads)
            for _ in pool.map(lambda low: moments(codes, start, low, min(low + step, stop - start)), range(0, stop - start, step)):
                pass
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
    """The store again with every imputed half's codes recalibrated to D* (scales[record, group]).

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


# ---------------------------------------------------------------------------
# Step 6: the direct read-based SV/CNV channel
# ---------------------------------------------------------------------------


ROW_SOURCE = "row_source"
ROW_SOURCES = ("imputed", "direct")
"""The ``row_source`` annotation's legend: a record of the imputed site list (popped or plain batches, with the long-read
half's calls on it), or a direct read-based row, a GATK-SV record called from the cohort's short reads."""
GATKSV_FILTER_POLICY = (
    "gatksv_source.GatksvSource: a record is kept where every FILTER value is PASS (or '.'), with MULTIALLELIC also allowed "
    "on copy-number records; breakends, multi-ALT records that are not copy numbers and copy numbers above the code range "
    "are dropped, each counted by reason"
)
DIRECT_RELIABILITY = (
    "no truth pairs, so every direct row takes the measurement model's no-truth form (fit_measurement_model without "
    "calibration, per ancestry group, pooled over the fit rows' groups) from its reported r^2 (gatksv_store_rows): 1 for "
    "a hard-call ALT count, the call set's own claim of exactness; the CNQ bound 1 - sum e_i / (n Var(called CN)), "
    "e_i = 10^(-CNQ_i / 10), for a copy number (CN reliability from CNQ, at least one copy per miscall: an upper bound); "
    "0, a -inf offset that leaves the row out, for a row whose called samples share one value, a bound at or below 0, or "
    "a called copy number without a quality"
)
DIRECT_FUSION = (
    "not applied: fit_model.fit takes no leakage map, so each one-to-one pair stays two columns, kept in one unbreakable "
    "group (group_first) so that no engine block splits it"
)
_CALL_SET_BYTES_PER_ENTRY = np.dtype(np.uint8).itemsize + np.dtype(np.bool_).itemsize + np.dtype(np.uint16).itemsize
"""A call-set entry's bytes in a ``GatksvBlock``: its value, its no-call flag and its quality."""
_MERGED_COPIES = 3
"""A merged code block's copies of one byte per entry: the imputed rows read, the direct rows read and the block."""


def call_set_columns(halves: Sequence[HalfSamples], call_set_ids: Sequence[str], crosswalk: SampleCrosswalk) -> I64Array:
    """For every store sample, in store column order over the halves, its column in a call set named by research ID,
    or ABSENT_COLUMN.

    An imputed half's samples reach it through the crosswalk (``source_columns_for_store_samples``). The long-read
    half is named by research ID, the call set's own namespace, so its names are looked up there directly: no name is
    compared across namespaces."""
    if len(set(call_set_ids)) != len(call_set_ids):
        raise ValueError("the call set repeats a research ID.")
    column_of = {research_id: column for column, research_id in enumerate(call_set_ids)}
    parts = []
    for half in halves:
        if half.namespace == "research_id":
            parts.append(np.array([column_of.get(name, ABSENT_COLUMN) for name in half.names], dtype=np.int64))
        else:
            parts.append(source_columns_for_store_samples(half, call_set_ids, crosswalk))
    return np.concatenate(parts).astype(np.int64)


def _imputed_site_list(config: WorkspaceConfig, chromosome: str, complexity_legend: tuple[str, ...] | None) -> tuple[tuple[str, ...], RecordSites]:
    """The imputed site list's identifiers and site fields, read as the store step read them (``read_plain_sites``, or
    the strata sidecar and ``read_popped_sites``): the store keeps neither the records' REF/ALT sequences nor INFO/END."""
    first_half = config.imputed_halves[0]
    path = first_half.path(chromosome, first_half.batches[0])
    if config.popped is None:
        listed = read_plain_sites(path, chromosome)
        return tuple(listed.expected.identifiers), listed.records
    strata = read_strata_sites(config.popped.strata_directory, chromosome, complexity_legend)
    return tuple(strata.identifiers), read_popped_sites(path, strata)


def imputed_sv_sites(chromosome: str, positions: NDArray, records: RecordSites, classes: NDArray) -> tuple[I64Array, SvSites]:
    """The imputed records a direct row may pair with, as rows of the chromosome and ``sv_fusion`` sites: every record
    that is not an SNV and whose REF and ALT are both sequence (``sequence_resolved_sites``). A symbolic imputed record
    carries no changed sequence to place on those coordinates, so it pairs with no direct row."""
    snv = VARIANT_CLASSES.index(VariantClass.SNV)
    rows = np.array(
        [
            row for row, (ref, alt) in enumerate(zip(records.refs, records.alts))
            if int(classes[row]) != snv and is_sequence_allele(ref) and is_sequence_allele(alt)
        ],
        dtype=np.int64,
    )
    sites = sequence_resolved_sites(
        np.full(rows.shape[0], chromosome), np.asarray(positions, dtype=np.int64)[rows],
        [records.refs[row] for row in rows.tolist()], [records.alts[row] for row in rows.tolist()],
    )
    return rows, sites


def _site_rows(sites: SvSites, rows: I64Array) -> SvSites:
    return SvSites(
        chromosomes=sites.chromosomes[rows], starts=sites.starts[rows], ends=sites.ends[rows], sizes=sites.sizes[rows],
        kinds=sites.kinds[rows], duplications_are_insertions=sites.duplications_are_insertions,
    )


def _store_rows(store: DosageStore, rows: I64Array, host_bytes: int) -> U8Array:
    """The codes of ``rows`` (ascending store rows) over every store sample. Rows within an inner chunk
    (DEFAULT_INNER_CHUNK_ROWS) of each other share one read, as their chunks are decoded either way, and a read spans
    at most ``host_bytes``."""
    codes = np.empty((rows.shape[0], store.n_samples), dtype=np.uint8)
    span = max(1, int(host_bytes // max(store.n_samples, 1)))
    first = 0
    while first < rows.shape[0]:
        last = first
        while last + 1 < rows.shape[0] and rows[last + 1] - rows[last] <= DEFAULT_INNER_CHUNK_ROWS and rows[last + 1] - rows[first] < span:
            last += 1
        read = store.read_codes(int(rows[first]), int(rows[last]) + 1)
        codes[first : last + 1] = read[rows[first : last + 1] - rows[first]]
        first = last + 1
    return codes


class _StackedRows:
    """Row blocks saved one .npy file each, read as one array [rows, columns] by row ranges."""

    def __init__(self, paths: Sequence[Path], width: int) -> None:
        self._arrays = [np.load(path, mmap_mode="r") for path in paths]
        self._starts = np.concatenate([[0], np.cumsum([array.shape[0] for array in self._arrays])]).astype(np.int64)
        self._width = width

    def rows(self, first: int, last: int, columns: slice) -> U8Array:
        pieces = [
            array[max(first - start, 0) : min(last, stop) - start, columns]
            for array, start, stop in zip(self._arrays, self._starts[:-1].tolist(), self._starts[1:].tolist())
            if start < last and stop > first
        ]
        return np.concatenate(pieces) if pieces else np.empty((0, self._width), dtype=np.uint8)[:, columns]


def _merged_code_blocks(
    source: DosageStore, start: int, is_direct: BoolArray, direct_codes: _StackedRows, columns: slice, block_rows: int
) -> Iterator[U8Array]:
    """One half's codes of a merged chromosome in row blocks: the imputed rows read from the source store (rows
    ``start`` onward), the direct rows from their saved blocks, each in merged order."""
    direct_merged = np.flatnonzero(is_direct)
    half_columns = np.arange(columns.start, columns.stop)
    for first in range(0, is_direct.shape[0], block_rows):
        last = min(first + block_rows, is_direct.shape[0])
        piece = is_direct[first:last]
        direct_first = int(np.searchsorted(direct_merged, first))
        direct_last = direct_first + int(piece.sum())
        imputed_first = first - direct_first
        imputed_last = imputed_first + (last - first) - (direct_last - direct_first)
        codes = np.empty((last - first, half_columns.shape[0]), dtype=np.uint8)
        if imputed_last > imputed_first:
            codes[~piece] = source.read_codes(start + imputed_first, start + imputed_last, half_columns)
        if direct_last > direct_first:
            codes[piece] = direct_codes.rows(direct_first, direct_last, columns)
        yield codes


def merged_measurement_model(imputed: Any, direct: Any, is_direct: BoolArray) -> Any:
    """The measurement model over a store holding direct rows: ``imputed``'s values at the imputed records, in their
    order, and ``direct``'s at the direct rows, both models of one class (``measurement_model.MeasurementModel``). A
    leakage map's records are renumbered to the merged rows, and the certificate keeps both models'."""
    imputed_rows = np.flatnonzero(~is_direct)
    direct_rows = np.flatnonzero(is_direct)
    if np.asarray(imputed.log_reliability).shape != imputed_rows.shape or np.asarray(direct.log_reliability).shape != direct_rows.shape:
        raise ValueError("the imputed and direct measurement models need one record per imputed and per direct row.")

    def merged(name: str) -> F64Array:
        values = np.empty(is_direct.shape[0], dtype=np.float64)
        values[imputed_rows] = getattr(imputed, name)
        values[direct_rows] = getattr(direct, name)
        return values

    maps = tuple(
        dataclasses.replace(leakage, records=rows[np.asarray(leakage.records, dtype=np.int64)])
        for model, rows in ((imputed, imputed_rows), (direct, direct_rows))
        for leakage in model.leakage_maps
    )
    return dataclasses.replace(
        imputed,
        scales=merged("scales"),
        residual_variance=merged("residual_variance"),
        log_reliability=merged("log_reliability"),
        leakage_maps=maps,
        certificate={"imputed_records": imputed.certificate, "direct_rows": direct.certificate},
    )


def _direct_chromosome(
    config: WorkspaceConfig, budget: ComputeBudget, directory: Path, source: DosageStore, source_root: Path, root: Path, chromosome: str,
    crosswalk: SampleCrosswalk, gene: GeneIntervals, complexity_legend: tuple[str, ...] | None, fit_columns: I64Array, fit_groups: I64Array,
    group_count: int,
) -> dict[str, Any]:
    """One chromosome of the store with its GATK-SV rows merged in, and the direct rows' reported r^2 and fitted-cohort moments for the measurement model.

    The call set is read in blocks of records that hold half the host (``gatksv_block_rows``), each block's rows saved
    as they are built; the pairs of every block are resolved one to one together (``one_to_one``). The direct rows go in
    coordinate order after the imputed records at their POS, and ``group_first`` keeps each imputed group, each same-POS
    set and each pair unbroken (``unbreakable_group_first``).
    """
    assert config.direct_sv is not None
    index = source.chromosomes.index(chromosome)
    start, stop = int(source.chromosome_starts[index]), int(source.chromosome_starts[index + 1])
    table = source.variant_table
    positions = np.asarray(table.position[start:stop], dtype=np.int64)
    if np.any(np.diff(positions) < 0):
        raise ValueError(f"{chromosome}: the store's records are not in position order, so direct rows have no place among them.")
    identifiers, records = _imputed_site_list(config, chromosome, complexity_legend)
    if identifiers != tuple(table.variant_ids(np.arange(start, stop))):
        raise ValueError(f"{chromosome}: the imputed site list names other records than the store holds.")
    candidate_rows, candidate_sites = imputed_sv_sites(chromosome, positions, records, table.variant_class[start:stop])

    path = config.direct_sv.gatksv_path(chromosome)
    call_set = GatksvSource(path)
    halves = source.half_samples()
    columns = call_set_columns(halves, call_set.sample_ids, crosswalk)
    sample_count = columns.shape[0]
    per_record = len(call_set.sample_ids) * _CALL_SET_BYTES_PER_ENTRY + sample_count * (_CALL_SET_BYTES_PER_ENTRY + np.dtype(np.uint8).itemsize)
    # Half the host holds a block of records with its aligned copy and its rows; the other half the imputed codes the
    # block pairs with and the reads behind them.
    block_records = max(1, int(budget.host_bytes // 2 // per_record))
    blocks = directory / "blocks" / chromosome
    blocks.mkdir(parents=True, exist_ok=True)
    spellings = _chromosome_spellings(chromosome)
    half_starts = np.asarray(source.half_sample_starts, dtype=np.int64)
    kept: dict[str, list[Any]] = {name: [] for name in (
        "records", "positions", "ends", "lengths", "ids", "refs", "alts", "svtypes", "classes", "codes_per_unit", "value_origin",
        "filled", "clipped", "reliability", "sources", "no_calls", "sums", "squares",
    )}
    evidence_parts: list[PairEvidence] = []
    block_count = offset = unobserved = 0
    for block in call_set.blocks(block_records):
        foreign = sorted(set(block.chromosomes) - set(spellings))
        if foreign:
            raise ValueError(f"{path}: records lie on {foreign}, not on {chromosome}.")
        aligned = block.aligned_to_store_samples(columns)
        sites = gatksv_sites(aligned)
        # A pair needs sizes within a factor of MINIMUM_SIZE_RATIO, so no smaller imputed record pairs with this block.
        reachable = np.flatnonzero(candidate_sites.sizes >= MINIMUM_SIZE_RATIO * sites.sizes.min())
        needed = reachable[np.unique(candidate_pairs(_site_rows(candidate_sites, reachable), sites).first_rows)]
        imputed = ImputedSvRecords(
            sites=_site_rows(candidate_sites, needed),
            codes=_store_rows(source, start + candidate_rows[needed], budget.host_bytes // 2),
        )
        rows, evidence = gatksv_block_rows(aligned, imputed)
        np.save(blocks / f"block{block_count}.npy", rows.codes)
        block_count += 1
        evidence_parts.append(dataclasses.replace(
            evidence, imputed_records=candidate_rows[needed[evidence.imputed_records]], gatksv_records=offset + evidence.gatksv_records,
        ))
        own = rows.gatksv_records
        kept["records"].append(offset + own)
        kept["positions"].append(aligned.positions[own])
        kept["ends"].append(aligned.ends[own])
        kept["lengths"].append(rows.lengths)
        for name, values in (("ids", aligned.variant_ids), ("refs", aligned.refs), ("alts", aligned.alts), ("svtypes", aligned.svtypes)):
            kept[name].extend(values[record] for record in own.tolist())
        kept["classes"].append(np.array([VARIANT_CLASSES.index(aligned.variant_classes[record]) for record in own.tolist()], dtype=np.uint8))
        kept["codes_per_unit"].append(rows.codes_per_unit)
        kept["value_origin"].append(rows.value_origin)
        kept["filled"].append(rows.filled_from_imputed)
        kept["clipped"].append(rows.clipped_counts)
        kept["reliability"].append(rows.reported_reliability)
        kept["sources"].append(rows.reliability_sources)
        kept["no_calls"].append(np.column_stack([
            aligned.no_call[own, int(half_starts[half]) : int(half_starts[half + 1])].sum(axis=1) for half in range(len(halves))
        ]).reshape(own.shape[0], len(halves)))
        sums = np.zeros((own.shape[0], group_count))
        squares = np.zeros((own.shape[0], group_count))
        for row in range(own.shape[0]):
            # The fitted cohort's moments in the row's own units (copies for a copy number): code / codes_per_unit + origin.
            values = rows.codes[row, fit_columns] / float(rows.codes_per_unit[row]) + float(rows.value_origin[row])
            sums[row] = np.bincount(fit_groups, weights=values, minlength=group_count)
            squares[row] = np.bincount(fit_groups, weights=values * values, minlength=group_count)
        kept["sums"].append(sums)
        kept["squares"].append(squares)
        unobserved += int(rows.unobserved_records.shape[0])
        offset += block.record_count
        del aligned, imputed, rows

    def joined(name: str, dtype: Any, width: int | None = None) -> NDArray:
        parts = kept[name]
        if parts:
            return np.concatenate(parts).astype(dtype)
        return np.zeros((0,) if width is None else (0, width), dtype=dtype)

    direct_records = joined("records", np.int64)
    direct_positions = joined("positions", np.int64)
    if np.any(np.diff(direct_positions) < 0):
        raise ValueError(f"{path}: the call set's records are not in position order.")
    direct_count = direct_positions.shape[0]
    evidence = PairEvidence(*(
        np.concatenate([getattr(part, field.name) for part in evidence_parts]) if evidence_parts else np.zeros(0)
        for field in dataclasses.fields(PairEvidence)
    ))
    chosen = one_to_one(evidence)
    pair_imputed = evidence.imputed_records[chosen].astype(np.int64)
    pair_records = evidence.gatksv_records[chosen].astype(np.int64)
    pair_direct = np.searchsorted(direct_records, pair_records)
    # A pair's z is read over its record's called samples, so a record with none (no row) pairs with nothing; a pair
    # whose record has no row would otherwise take the next row's place.
    if np.any(pair_direct >= direct_count) or not np.array_equal(direct_records[np.minimum(pair_direct, max(direct_count - 1, 0))], pair_records):
        raise ValueError(f"{path}: a paired GATK-SV record has no row of its own.")

    # The merged order: each direct row after the imputed records at or before its POS.
    record_count = positions.shape[0]
    merged_count = record_count + direct_count
    direct_merged = np.searchsorted(positions, direct_positions, side="right") + np.arange(direct_count)
    is_direct = np.zeros(merged_count, dtype=bool)
    is_direct[direct_merged] = True
    imputed_merged = np.flatnonzero(~is_direct)

    def merged(imputed_values: Any, direct_values: Any, dtype: Any) -> NDArray:
        values = np.empty(merged_count, dtype=dtype)
        values[imputed_merged] = imputed_values
        values[direct_merged] = direct_values
        return values

    direct_lengths = joined("lengths", np.float64)
    direct_refs, direct_alts, direct_svtypes = tuple(kept["refs"]), tuple(kept["alts"]), tuple(kept["svtypes"])
    direct_ends = tuple(int(end) for end in joined("ends", np.int64).tolist())
    unique_positions, first_of_position = np.unique(positions, return_index=True)
    if direct_count and unique_positions.shape[0] < 2:
        raise ValueError(f"{chromosome}: a direct row's genetic position is interpolated between the imputed records', which needs two positions.")
    direct_centimorgans = (
        interpolated_genetic_positions(unique_positions, table.genetic_position_cm[start:stop][first_of_position], direct_positions)
        if direct_count else np.zeros(0)
    )
    direct_sites = RecordSites(
        refs=direct_refs, alts=direct_alts, sv_types=direct_svtypes, sv_lengths=direct_lengths, ends=direct_ends,
        genetic_position_cm=direct_centimorgans,
    )
    core_starts, core_ends = store_core_spans(direct_positions, direct_sites)
    gene_columns = gene_overlap(gene.gene_starts, gene.gene_ends, gene.exon_starts, gene.exon_ends, gene.tss, core_starts, core_ends)
    annotations: dict[str, NDArray] = {}
    legends = dict(table.annotation_legends)
    for name, values in table.annotations.items():
        own_values = np.asarray(values[start:stop])
        if name == ROW_SOURCE:
            raise ValueError(f"{source_root} already holds direct rows.")
        if name == "sv_length":
            direct_values: NDArray = direct_lengths
        elif name == "tr_locus":
            # A direct row joins no repeat locus: it keeps its GATK-SV class.
            direct_values = np.full(direct_count, NO_LOCUS)
        elif name in gene_columns:
            direct_values = gene_columns[name]
        elif name in legends:
            # The legend's missing state (annotation_design): the imputed sidecar's own fields say nothing of a direct row.
            direct_values = np.full(direct_count, -1)
        else:
            direct_values = np.full(direct_count, np.nan)
        annotations[name] = merged(own_values, direct_values, own_values.dtype)
    annotations[ROW_SOURCE] = is_direct.astype(np.int32)
    legends[ROW_SOURCE] = ROW_SOURCES
    merged_positions = merged(positions, direct_positions, np.int64)
    pair_groups = np.full(merged_count, -1, dtype=np.int64)
    pair_groups[imputed_merged[pair_imputed]] = np.arange(chosen.shape[0])
    pair_groups[direct_merged[pair_direct]] = np.arange(chosen.shape[0])
    group_first = unbreakable_group_first(
        merged(np.asarray(table.group_first[start:stop], dtype=np.int64) - start, np.full(direct_count, -1), np.int64), merged_positions, pair_groups,
    )

    direct_codes = _StackedRows([blocks / f"block{block}.npy" for block in range(block_count)], sample_count)
    widest = int(np.diff(half_starts).max())
    merge_rows = max(1, int(budget.host_bytes // 2 // (_MERGED_COPIES * widest)))
    direct_no_calls = joined("no_calls", np.uint32, len(halves))
    sums_total = np.zeros(merged_count, dtype=np.uint64)
    squares_total = np.zeros(merged_count, dtype=np.uint64)
    for half in range(len(halves)):
        half_columns = slice(int(half_starts[half]), int(half_starts[half + 1]))
        sums, squares = write_half_codes(
            root, half, chromosome, merged_count, half_columns.stop - half_columns.start,
            _merged_code_blocks(source, start, is_direct, direct_codes, half_columns, merge_rows), codec=config.codec,
        )
        sums_total += sums.astype(np.uint64)
        squares_total += squares.astype(np.uint64)
        imputed_no_calls, _ = open_column(statistic_column_directory(source_root, half, chromosome, "no_calls"))
        write_column(
            statistic_column_directory(root, half, chromosome, "no_calls"), merged(np.asarray(imputed_no_calls), direct_no_calls[:, half], np.uint32)
        )
    del direct_codes

    direct_ids = tuple(kept["ids"])
    merged_ids = merged(np.array(identifiers, dtype=object), np.array(direct_ids, dtype=object), object).tolist()
    encoded = [identifier.encode() for identifier in merged_ids]
    direct_classes = joined("classes", np.uint8)
    merged_table = VariantTable(
        chromosome=np.full(merged_count, chromosome_number(chromosome), dtype=np.int8),
        position=merged_positions,
        genetic_position_cm=merged(table.genetic_position_cm[start:stop], direct_centimorgans, np.float64),
        ref_length=merged(table.ref_length[start:stop], [len(ref) for ref in direct_refs], np.int32),
        alt_length=merged(table.alt_length[start:stop], [len(alt) for alt in direct_alts], np.int32),
        variant_class=merged(table.variant_class[start:stop], direct_classes, np.uint8),
        codes_per_unit=merged(table.codes_per_unit[start:stop], joined("codes_per_unit", np.uint8), np.uint8),
        value_origin=merged(table.value_origin[start:stop], joined("value_origin", np.int64), np.int64),
        group_first=group_first,
        sum_code=sums_total,
        sum_code2=squares_total,
        annotations=annotations,
        annotation_legends=legends,
        id_bytes=np.frombuffer(b"".join(encoded), dtype=np.uint8),
        id_offsets=np.concatenate([[0], np.cumsum([len(identifier) for identifier in encoded])]).astype(np.int64),
    )
    write_variant_columns(root, chromosome, merged_table, slice(0, merged_count))
    shutil.copytree(source_root / "loci" / chromosome, root / "loci" / chromosome, dirs_exist_ok=True)
    from sv_pgs.feature_identity import chromosome_identity
    identity = chromosome_identity(
        genome_build=config.genome_build, chromosome=chromosome, positions=merged_positions,
        refs=merged(np.array(records.refs, dtype=object), np.array(direct_refs, dtype=object), object).tolist(),
        alts=merged(np.array(records.alts, dtype=object), np.array(direct_alts, dtype=object), object).tolist(),
        identifiers=merged_ids,
        ends=merged(np.array(records.ends, dtype=object), np.array(direct_ends, dtype=object), object).tolist(),
        sv_types=merged(np.array(records.sv_types, dtype=object), np.array(direct_svtypes, dtype=object), object).tolist(),
        sv_lengths=merged(records.sv_lengths, direct_lengths, np.float64),
        codes_per_unit=merged_table.codes_per_unit, value_origin=merged_table.value_origin,
    )
    sources = joined("sources", np.uint8)
    filled = joined("filled", bool)
    np.savez(
        directory / "rows" / f"{chromosome}.npz",
        merged_rows=direct_merged.astype(np.int64),
        classes=direct_classes,
        reported_reliability=joined("reliability", np.float64),
        sums=joined("sums", np.float64, group_count),
        squares=joined("squares", np.float64, group_count),
    )
    return {
        "feature_identity": identity,
        "records": int(merged_count),
        "sites_md5": sites_md5(merged_positions, merged_table.ref_length, merged_table.alt_length),
        "call_set": {"file": path.name, "bytes": int(path.stat().st_size)},
        "direct_rows": int(direct_count),
        "kept_records": int(call_set.kept_record_count),
        "skipped_records": dict(call_set.skipped_records),
        "uncalled_records": int(unobserved),
        "candidate_pairs": int(evidence.pairing_z.shape[0]),
        "pairs": int(chosen.shape[0]),
        "filled_from_paired_imputed_ds": int(filled.sum()),
        "filled_from_observed_mean": int(direct_count - filled.sum()),
        "clipped_entries": int(joined("clipped", np.int64).sum()),
        "reliability_sources": {name: int(np.sum(sources == code)) for code, name in enumerate(RELIABILITY_SOURCES)},
    }


def _direct_chromosome_task(task: tuple[Any, ...]) -> str:
    """One chromosome of the direct_sv step in its own process (``_direct_chromosome``), written with its fact file,
    whose presence marks it done, and its blocks of rows removed."""
    config, budget, directory, source_root, root, chromosome, *rest = task
    with DosageStore.open(source_root) as source:
        fact = _direct_chromosome(config, budget, directory, source, source_root, root, chromosome, *rest)
    _write_json(directory / "chromosomes" / f"{chromosome}.json", fact)
    _remove_own(directory / "blocks" / chromosome, directory)
    return chromosome


def _direct_sv_inputs(config: WorkspaceConfig) -> Any:
    """The call set's files by size and content (``genotype_file_fingerprint``) and the codec the merged store is
    written with; the crosswalk, the store's inputs and the measurement model enter through the upstream keys."""
    if config.direct_sv is None:
        return {"direct_sv": None}
    if config.direct_sv.gatksv_calls is None:
        return {"gatksv_calls": None}
    return {
        "gatksv_calls": [
            (config.direct_sv.gatksv_path(chromosome).name, genotype_file_fingerprint(config.direct_sv.gatksv_path(chromosome)))
            for chromosome in config.chromosomes
        ],
        "codec": config.codec,
        "genome_build": config.genome_build,
    }


def _direct_sv_step(run: _Run, directory: Path) -> dict[str, Any]:
    """The direct read-based SV/CNV channel: every kept GATK-SV record a store row of its own
    beside the imputed records, and the measurement model over the merged store's rows.

    Each chromosome is a sub-checkpoint, merged in a process of its own with its share of the host, its blocks of rows
    removed once it is written. A direct row's reliability is
    the measurement model's no-truth form from its reported r^2 (DIRECT_RELIABILITY); no pair is fused (DIRECT_FUSION).
    With ``direct_sv`` or its call set null the step adds nothing: the measurement step's store and model pass through
    unchanged, and the summary and certificate say so.
    """
    config = run.config
    measured = run.summary("measurement")["store"]
    optional = {key: "null: not implemented, not read" for key in DIRECT_SV_KEYS[1:]}
    if config.direct_sv is None or config.direct_sv.gatksv_calls is None:
        absent = "direct_sv is null" if config.direct_sv is None else "direct_sv.gatksv_calls is null"
        statement = (
            f"not read: {absent}, so no direct row is added; the measurement step's store and measurement model pass "
            "through unchanged"
        )
        _write_json(directory / "certificate.json", {"gatksv_calls": statement, "optional_sources": optional})
        return {
            "store": measured,
            "measurement_model": (Path("measurement") / MEASUREMENT_MODEL_FILE).as_posix(),
            "direct_rows": 0,
            "gatksv_calls": statement,
            "optional_sources": optional,
        }
    samples = _read_samples(run)
    cohort = _read_cohort(run)
    crosswalk = SampleCrosswalk.read(config.crosswalk, *config.crosswalk_columns)
    source_root = config.run_directory / measured
    root = directory / "store"
    for subdirectory in ("chromosomes", "rows", "blocks"):
        (directory / subdirectory).mkdir(exist_ok=True)
    genes = read_gene_annotation(config.gene_annotation)
    complexity_legend = None if config.popped is None else strata_complexity_legend(config.popped.strata_directory, config.chromosomes)
    group_count = len(samples.legend)
    fit_columns = np.sort(cohort.store_columns[cohort.fit_rows])
    fit_groups = np.concatenate(samples.groups)[fit_columns]
    with DosageStore.open(source_root) as source:
        halves = source.half_samples()
        attributes = dict(source.manifest_attributes)
        half_sample_counts = [int(count) for count in np.diff(source.half_sample_starts)]
    pending = [chromosome for chromosome in config.chromosomes if not (directory / "chromosomes" / f"{chromosome}.json").is_file()]
    for chromosome in pending:
        _clear_chromosome(root, chromosome, len(halves), directory)
    # The chromosomes are independent: one process each, every one with its share of the host and the threads.
    workers = max(1, min(run.budget.cpu_threads, len(pending)))
    worker_budget = dataclasses.replace(
        run.budget, device_kind="cpu", device_ids=(), device_names=(), device_bytes=(), device_compute_capabilities=(),
        host_bytes=run.budget.host_bytes // workers, cpu_threads=max(1, run.budget.cpu_threads // workers),
    )
    tasks = [
        (config, worker_budget, directory, source_root, root, chromosome, crosswalk, genes[chromosome], complexity_legend, fit_columns, fit_groups, group_count)
        for chromosome in pending
    ]
    if workers == 1:
        for task in tasks:
            _direct_chromosome_task(task)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            list(pool.map(_direct_chromosome_task, tasks))
    facts = [json.loads((directory / "chromosomes" / f"{chromosome}.json").read_text(encoding="utf-8")) for chromosome in config.chromosomes]
    from sv_pgs.feature_identity import IDENTITY_FILE
    _write_json(root / IDENTITY_FILE, {chromosome: fact["feature_identity"] for chromosome, fact in zip(config.chromosomes, facts)})
    call_set = {
        "caller": "GATK-SV",
        "files": {chromosome: fact["call_set"] for chromosome, fact in zip(config.chromosomes, facts)},
        "filter_policy": GATKSV_FILTER_POLICY,
        "rows": int(sum(fact["direct_rows"] for fact in facts)),
    }
    write_store_manifest(
        root,
        chromosomes=config.chromosomes,
        record_counts=[fact["records"] for fact in facts],
        chromosome_sites_md5=[fact["sites_md5"] for fact in facts],
        half_sample_counts=half_sample_counts,
        half_samples=halves,
        half_measurements=attributes["half_measurements"],
        gates=attributes["gates"],
        recalibrated=bool(attributes["recalibrated"]),
        direct_calls=call_set,
    )
    with DosageStore.open(root) as store:
        if store.half_samples() != halves:
            raise ValueError("the merged store's sample manifests differ from the measurement step's store.")

    # The direct rows' measurement model: the no-truth form per ancestry group from their reported r^2 and the fitted
    # cohort's moments in their own units, pooled over the groups as the measurement step pools the imputed records'.
    rows = [np.load(directory / "rows" / f"{chromosome}.npz") for chromosome in config.chromosomes]
    chromosome_starts = np.concatenate([[0], np.cumsum([fact["records"] for fact in facts])]).astype(np.int64)
    is_direct = np.zeros(int(chromosome_starts[-1]), dtype=bool)
    for chromosome_start, arrays in zip(chromosome_starts[:-1].tolist(), rows):
        is_direct[chromosome_start + arrays["merged_rows"]] = True
    reliability = np.concatenate([arrays["reported_reliability"] for arrays in rows])
    strata = np.asarray(VARIANT_CLASS_LEGEND)[np.concatenate([arrays["classes"] for arrays in rows]).astype(np.int64)]
    sums = np.concatenate([arrays["sums"] for arrays in rows])
    squares = np.concatenate([arrays["squares"] for arrays in rows])
    counts = np.bincount(fit_groups, minlength=group_count).astype(np.int64)
    pooled = sums.sum(axis=1) / fit_columns.shape[0], squares.sum(axis=1) / fit_columns.shape[0]
    results, group_means, group_variances = [], [], []
    for group in range(group_count):
        # A group with no fitted rows takes the pooled moments, as in the measurement step.
        mean, second = (sums[:, group] / counts[group], squares[:, group] / counts[group]) if counts[group] else pooled
        variance = np.maximum(second - mean * mean, 0.0)
        group_means.append(mean)
        group_variances.append(variance)
        results.append(run.bindings.fit_measurement_model(None, variance, strata, reported_reliability=reliability))
    direct_model = run.bindings.pooled_measurement_model(
        results, np.column_stack(group_means).reshape(-1, group_count), np.column_stack(group_variances).reshape(-1, group_count), counts
    )
    imputed_model = run.bindings.load_measurement_model(run.output("measurement") / MEASUREMENT_MODEL_FILE)
    merged_measurement_model(imputed_model, direct_model, is_direct).save(directory / MEASUREMENT_MODEL_FILE)

    total = {
        name: int(sum(fact[name] for fact in facts))
        for name in (
            "direct_rows", "kept_records", "uncalled_records", "candidate_pairs", "pairs", "filled_from_paired_imputed_ds",
            "filled_from_observed_mean", "clipped_entries",
        )
    }
    skipped: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    for fact in facts:
        skipped.update(fact["skipped_records"])
        sources.update(fact["reliability_sources"])
    excluded = int(sum(count for name, count in sources.items() if name not in ("hard_call", "cnq_bound")))
    certificate = {
        "caller": "GATK-SV",
        "call_set": call_set,
        "uncalibrated_records": total["direct_rows"],
        "reliability_source": DIRECT_RELIABILITY,
        "reliability_rules": dict(sources),
        "excluded_without_reliability": excluded,
        "direct_calls_fused": 0,
        "fusion": DIRECT_FUSION,
        "optional_sources": optional,
        "measurement_model": direct_model.certificate,
    }
    _write_json(directory / "certificate.json", certificate)
    return {
        "store": (Path("direct_sv") / "store").as_posix(),
        "measurement_model": (Path("direct_sv") / MEASUREMENT_MODEL_FILE).as_posix(),
        **total,
        "records": int(chromosome_starts[-1]),
        "skipped_records": dict(skipped),
        "filter_policy": GATKSV_FILTER_POLICY,
        "reliability": DIRECT_RELIABILITY,
        "reliability_rules": dict(sources),
        "excluded_without_reliability": excluded,
        "fusion": DIRECT_FUSION,
        "no_call_fill": (
            "E_lin[B | DS] from the paired imputed DS with the largest Fisher z (a defined slope), else the record's observed "
            "mean; never zero. E[B | SL] is not built"
        ),
        "arms": "snv_indel leaves every direct row out (-inf log reliability); with_sv fits them",
        "optional_sources": optional,
    }


def _final_store(run: _Run) -> Path:
    """The store the fit, score and effects read: the direct_sv step's (the measurement step's when it adds no row)."""
    return run.config.run_directory / run.summary("direct_sv")["store"]


def _final_measurement_model(run: _Run) -> Path:
    """The measurement model over the final store's records, the one the fit reads."""
    return run.config.run_directory / run.summary("direct_sv")["measurement_model"]


# ---------------------------------------------------------------------------
# Steps 6-7: fit and score
# ---------------------------------------------------------------------------


def structural_records(table: VariantTable) -> BoolArray:
    """Whether each store record is structural: its typed length (the store's ``sv_length``, written by the store step
    for every record) is at least COMMUNITY_SV_MINIMUM_LENGTH, it is a copy number, or it is a direct read-based row
    (ROW_SOURCE), whatever its length: a GATK-SV insertion of unknown length writes SVLEN -1."""
    if "sv_length" not in table.annotations:
        raise ValueError("the store has no sv_length column, so its structural records cannot be told apart for the arms.")
    lengths = np.asarray(table.annotations["sv_length"], dtype=np.float64)
    if not np.all(np.isfinite(lengths)):
        raise ValueError("every store record needs a finite sv_length to be assigned to an arm.")
    structural = (lengths >= COMMUNITY_SV_MINIMUM_LENGTH) | (np.asarray(table.variant_class) == VARIANT_CLASSES.index(VariantClass.COPY_NUMBER))
    if ROW_SOURCE in table.annotations:
        if tuple(table.annotation_legends.get(ROW_SOURCE, ())) != ROW_SOURCES:
            raise ValueError(f"the store's {ROW_SOURCE} legend is not {ROW_SOURCES}.")
        structural |= np.asarray(table.annotations[ROW_SOURCE]) == ROW_SOURCES.index("direct")
    return structural


_COMPLEMENT = str.maketrans("ACGT", "TGCA")
_ALLELE_ID = re.compile(r"[^:]+:[0-9]+:([ACGTacgt]+):([ACGTacgt]+)")
"""A record ID that names its alleles: CHROM:POS:REF:ALT."""


def listed_sites(table: VariantTable, path: str | Path) -> BoolArray:
    """Whether each record is a single-base substitution at a site of the list at ``path`` (``WorkspaceConfig.
    fit_variant_sites``): a tab-separated file, one site a line under a header naming its columns chromosome (the
    autosome's number), position (in the store's build), allele_a and allele_b. A record whose ID names its alleles
    (``_ALLELE_ID``) matches a site only with the site's two alleles, in either order or both complemented (a list
    lifted over from another build can hold a site on the other strand); a record whose ID does not name them matches
    on its chromosome and position alone."""
    sites = pd.read_csv(path, sep="\t", usecols=["chromosome", "position", "allele_a", "allele_b"], dtype={"chromosome": np.int64, "position": np.int64, "allele_a": str, "allele_b": str})
    position = np.asarray(table.position, dtype=np.int64)
    rows = np.flatnonzero((np.asarray(table.ref_length) == 1) & (np.asarray(table.alt_length) == 1) & np.isin(position, sites["position"].to_numpy()))
    candidates = pd.DataFrame({"row": rows, "chromosome": np.asarray(table.chromosome, dtype=np.int64)[rows], "position": position[rows]})
    matched = candidates.merge(sites, on=["chromosome", "position"])
    listed = np.zeros(table.variant_count, dtype=bool)
    named = 0
    for row, identifier, first, second in zip(matched["row"].tolist(), table.variant_ids(matched["row"].to_numpy()), matched["allele_a"], matched["allele_b"]):
        alleles = _ALLELE_ID.fullmatch(identifier)
        if alleles is None:
            listed[row] = True
            continue
        named += 1
        pair = (first.upper(), second.upper())
        listed[row] |= frozenset(allele.upper() for allele in alleles.groups()) in (frozenset(pair), frozenset(allele.translate(_COMPLEMENT) for allele in pair))
    LOGGER.info(
        "workspace pipeline: %d records are at %d listed sites (%d of the %d at a listed site's position matched on their ID's alleles, the rest on position)",
        int(listed.sum()), sites.shape[0], named, matched.shape[0],
    )
    return listed


def region_records(table: VariantTable, region: tuple[int, int, int]) -> BoolArray:
    """Whether each record lies in ``region`` (``WorkspaceConfig.fit_region``): on its chromosome, with its position (the
    record's POS in the store's build; a structural record's first base) from the region's first to its last position."""
    chromosome, first, last = region
    position = np.asarray(table.position, dtype=np.int64)
    return (np.asarray(table.chromosome, dtype=np.int64) == chromosome) & (position >= first) & (position <= last)


def fit_filtered(
    table: VariantTable, sample_count: int, log_reliability: F64Array, minimum_frequency: float, minimum_r2: float, variant_sites: str | Path | None = None,
    region: tuple[int, int, int] | None = None,
) -> F64Array:
    """``log_reliability`` with -inf on every record the fit leaves out (``WorkspaceConfig.fit_minimum_r2``, its
    frequency floor, its site list and its region): one whose minor allele frequency over the store's ``sample_count``
    samples, from its code sums, or whose reliability (the measurement model's r^2, ``log_reliability``) is below its
    floor; with a site list, a record that is neither structural (``structural_records``) nor at a listed site
    (``listed_sites``); with a region, a record outside it (``region_records``). A region's kept records are logged by
    variant class, short and structural apart: the snv_indel arm fits the short ones, with_sv every one."""
    kept = np.ones(np.asarray(table.sum_code).shape[0], dtype=bool)
    if minimum_frequency > 0.0:
        frequency = np.asarray(table.sum_code, dtype=np.float64) / (2.0 * CODES_PER_DOSAGE * sample_count)
        kept &= np.minimum(frequency, 1.0 - frequency) >= minimum_frequency
    if minimum_r2 > 0.0:
        kept &= np.asarray(log_reliability, dtype=np.float64) >= np.log(minimum_r2)
    if variant_sites is not None:
        kept &= structural_records(table) | listed_sites(table, variant_sites)
    if region is not None:
        kept &= region_records(table, region)
    LOGGER.info(
        "workspace pipeline: the fit keeps %d of %d records (minor allele frequency >= %g, r^2 >= %g, sites %s, region %s)",
        int(kept.sum()), kept.shape[0], minimum_frequency, minimum_r2, "all" if variant_sites is None else variant_sites,
        "none" if region is None else "chr{}:{}-{}".format(*region),
    )
    if region is not None:
        classes = Counter(
            f"{VARIANT_CLASS_LEGEND[code]} {'structural' if structural else 'short'}"
            for code, structural in zip(np.asarray(table.variant_class)[kept].tolist(), structural_records(table)[kept].tolist())
        )
        LOGGER.info(
            "workspace pipeline: the fit region chr%d:%d-%d keeps %s", *region, ", ".join(f"{count} {name}" for name, count in sorted(classes.items())),
        )
        if not classes:
            raise ValueError("the fit region chr{}:{}-{} holds no record the fit keeps; its positions are in the store's build.".format(*region))
    return np.where(kept, np.asarray(log_reliability, dtype=np.float64), -np.inf)


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


def fit_request(
    cohort: _Cohort, store: DosageStore, log_variance_offset: F64Array, budget: ComputeBudget, work_dir: Path, seed: int,
    disease_likelihood: str = "gaussian_working", annotated_prior: bool = True, length_by_class: bool = False, ld_score_prior: bool = False,
) -> FitRequest:
    """Every trait's models (each trait's folds' and its all-rows model, in the cohort's model order) as one fit, on the
    rows any of them trains on, so the models of different traits that share their training rows and covariates are
    fitted together (``stage2_wiring.model_groups``) and share the store's passes.

    Every model adjusts for its own trait's covariates and the structure columns, never another trait's; the
    covariate names are the cohort's whole union, so every trait's fitted model scores with the same covariates
    (``artifact.predict_models``). A row's target is read only where its model trains on it; each model's held-out
    rows (its evaluation fold) go to the fit's EB monitor alone (``eb_monitor``), which logs each empirical-Bayes call's
    held-out partial r^2 (and a disease's AUC) and nothing else, so a fit's signal shows before it ends.

    The configured disease likelihood is explicit; quantitative traits always use Gaussian regression. Both arms' requests
    hold the same models in the same order with the run's ``seed``, so each group's probes and draws are common random
    numbers across the arms where the models' layouts permit it; this does not eliminate their Monte Carlo error.
    """
    rows = cohort.fit_rows
    traits = np.asarray(cohort.model_traits)
    # The EB monitor's rows: each model's held-out fold where its target and the covariates are finite (a monitor's
    # input never refuses a fit).
    held = np.asarray(cohort.held_out) & np.isfinite(cohort.targets[:, traits]) & np.all(np.isfinite(cohort.covariates), axis=1)[:, None]
    evaluated = held.any(axis=1)
    evaluation = EvaluationRows(
        store_columns=cohort.store_columns[evaluated],
        covariates=cohort.covariates[evaluated][:, 1:],
        targets=np.where(held, cohort.targets[:, traits], np.nan)[evaluated],
        disease=np.array([cohort.trait_types[trait] == TraitType.BINARY for trait in traits.tolist()], dtype=bool),
    ) if evaluated.any() else None
    return FitRequest(
        store=store,
        store_columns=cohort.store_columns[rows],
        covariates=cohort.covariates[rows][:, 1:],
        covariate_names=cohort.covariate_names[1:],
        covariate_columns=cohort.covariate_columns[traits][:, 1:],
        targets=np.where(cohort.training, cohort.targets[:, traits], np.nan)[rows],
        training=cohort.training[rows],
        model_names=tuple(cohort.model_names),
        trait_types=tuple(
            TraitType.BINARY if disease_likelihood == "bernoulli" and cohort.trait_types[trait] == TraitType.BINARY else FIT_TRAIT_TYPE
            for trait in traits.tolist()
        ),
        research_ids=tuple(np.asarray(cohort.research_ids)[rows].tolist()),
        log_variance_offset=log_variance_offset,
        budget=budget,
        work_dir=work_dir,
        seed=seed,
        annotated_prior=annotated_prior,
        evaluation=evaluation,
        length_by_class=length_by_class,
        ld_score_prior=ld_score_prior,
    )


def _fitted_model_directory(run: _Run, arm: str, trait_name: str) -> Path:
    return run.output("fit") / "models" / arm / trait_name


def _checked_trait_model(run: _Run, cohort: _Cohort, arm: str, trait: int, folder: Path) -> Any:
    """Read one saved trait in the cohort's model order, checking its named axes before any score is used.

    A copied arm can list the same fold models in a different order; names bind those models, never their positions
    in a directory. Covariates are different: the prediction API takes a matrix without column names, so the saved
    names and each model's mask must agree with the cohort before that matrix is handed to it. A different name,
    likelihood or covariate design is a different fitting problem, not a reason to refit an expensive arm in silence.
    """
    if sum(path.stat().st_size for path in folder.iterdir() if path.is_file()) > run.budget.host_bytes:
        raise MemoryError(f"saved model {folder} exceeds the host budget before it is loaded.")
    fitted = run.bindings.load_model(folder)
    indices = cohort.trait_models(trait)
    expected = tuple(cohort.model_names[index] for index in indices)
    if len(fitted.model_names) != len(expected) or set(fitted.model_names) != set(expected):
        raise ValueError(f"the {arm} model of {cohort.trait_names[trait]} does not hold its trait's models {expected}.")
    if tuple(fitted.model_names) != expected:
        fitted = model_subset(fitted, expected)
    if tuple(fitted.covariate_names) != cohort.covariate_names[1:]:
        raise ValueError(f"the {arm} model of {cohort.trait_names[trait]} has different covariate names or order.")
    if not np.array_equal(fitted.covariate_columns, cohort.covariate_columns[cohort.model_traits[indices], 1:]):
        raise ValueError(f"the {arm} model of {cohort.trait_names[trait]} has different covariate columns.")
    likelihood = TraitType.BINARY if run.config.disease_likelihood == "bernoulli" and cohort.trait_types[trait] == TraitType.BINARY else FIT_TRAIT_TYPE
    if any(scoring.trait_type != likelihood for scoring in fitted.scoring):
        raise ValueError(f"the {arm} model of {cohort.trait_names[trait]} has a different likelihood.")
    return fitted


def _check_fit_records(run: _Run, directory: Path, key: str) -> None:
    """Check the saved-model handoff, without changing the fitting computation or its key.

    The mirror binds each copied record to its source partial's key. Old local records have only their enclosing
    KEY_FILE; they are checked against their saved model here and bound on the next push. A record without its
    directory, or an arm with only some traits, is an incomplete transfer: stop before invoking the fitter, so a
    resubmission cannot spend hours replacing a model that merely has not arrived. A directory saved just before a
    preemption, without its record yet, is left for the fit step to record without fitting it again.

    This validation belongs to the driver boundary, outside _fit_step's source digest: a post-fit validation fix
    does not change a fitted coefficient. The fit's existing key still checks every engine and input dependency.
    """
    cohort = _read_cohort(run)
    expected_offsets, expected_problem = None, None
    for arm in ARMS:
        root = directory / "models" / arm
        present = [name for name in cohort.trait_names if (root / name).exists() or (root / (name + ".json")).exists()]
        if len(present) != len(cohort.trait_names) and (not directory.name.endswith(".partial") or (present and not (directory / "work" / arm).is_dir())):
            missing = sorted(set(cohort.trait_names) - set(present))
            raise ValueError(f"incomplete {arm} model transfer: missing traits {missing}; restore them before finalizing.")
        for trait, name in enumerate(cohort.trait_names):
            folder, record_path = root / name, root / (name + ".json")
            if not record_path.is_file():
                if not directory.name.endswith(".partial"):
                    raise ValueError(f"completed fit is missing model record {record_path}.")
                continue
            if not folder.is_dir():
                raise ValueError(f"model record {record_path} has no model directory; restore it before finalizing.")
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("fit_key", key) != key:
                raise ValueError(f"model record {record_path} belongs to a different fit key.")
            for relative, size in record.get("model_files", {}).items():
                path = folder / relative
                if Path(relative).is_absolute() or ".." in Path(relative).parts or not path.is_file() or path.stat().st_size != size:
                    raise ValueError(f"incomplete model transfer at {folder}: missing or truncated {relative}.")
            fitted = _checked_trait_model(run, cohort, arm, trait, folder)
            if hasattr(fitted, "provenance"):
                if expected_offsets is None:
                    # Both arms have the same model names and training problem. Their record offsets distinguish
                    # them; copying snv_indel into with_sv must not turn a paired comparison into a self-comparison.
                    measurement = run.bindings.load_measurement_model(_final_measurement_model(run))
                    with DosageStore.open(_final_store(run)) as store:
                        kept = fit_filtered(store.variant_table, store.n_samples, measurement.log_reliability,
                                            run.config.fit_minimum_minor_allele_frequency, run.config.fit_minimum_r2, run.config.fit_variant_sites,
                                            run.config.fit_region)
                        expected_offsets = {name: fit_model.offset_digest(arm_log_reliability(store.variant_table, kept, name)) for name in ARMS}
                    del measurement, kept
                if fitted.provenance.offset_digest != expected_offsets[arm]:
                    raise ValueError(f"model {folder} belongs to different arm offsets or variant filters.")
                if expected_problem is None:
                    rows, traits = cohort.fit_rows, cohort.model_traits
                    expected_problem = fit_model.problem_digest(
                        research_ids=tuple(np.asarray(cohort.research_ids)[rows].tolist()), store_columns=cohort.store_columns[rows],
                        model_names=cohort.model_names,
                        trait_types=tuple(TraitType.BINARY if run.config.disease_likelihood == "bernoulli" and cohort.trait_types[index] == TraitType.BINARY else FIT_TRAIT_TYPE for index in traits),
                        covariate_names=cohort.covariate_names[1:], covariate_columns=cohort.covariate_columns[traits, 1:],
                        covariates=cohort.covariates[rows, 1:], targets=cohort.targets[rows][:, traits], training=cohort.training[rows],
                        draw_count=fit_model.DRAW_COUNT, seed=run.config.seed,
                    )
                if fitted.provenance.problem_digest != expected_problem:
                    raise ValueError(f"model {folder} belongs to a different cohort or training problem.")
            if "fit_key" not in record:
                LOGGER.warning("workspace pipeline: legacy record %s has no source fit key; using its enclosing key after model validation", record_path)
            if len(record["models"]) != len(fitted.model_names) or set(record["models"]) != set(fitted.model_names):
                raise ValueError(f"model record {record_path} names different models from its directory.")
            # The artifact is the source of the verdict, including for records written before acceptance was saved.
            # Refresh it in cohort order so a swapped record cannot quarantine the wrong trait or certify a bad fit.
            record.update(models=list(fitted.model_names), acceptance=acceptance(fitted),
                          fitted_records=[int(scoring.store_rows.shape[0]) for scoring in fitted.scoring],
                          refusals=list(fitted.refusals))
            record[CERTIFICATE_STATUS] = [bool(met) for met in np.asarray(fitted.certificate[CERTIFICATE_STATUS])]
            if directory.name.endswith(".partial"):
                record["fit_key"] = key
                if record != json.loads(record_path.read_text(encoding="utf-8")):
                    _write_json(record_path, record)
            elif record["acceptance"] != json.loads(record_path.read_text(encoding="utf-8")).get("acceptance"):
                raise ValueError(f"completed model record {record_path} disagrees with its saved model.")
            del fitted


def _fit_step(run: _Run, directory: Path) -> dict[str, Any]:
    """Each arm's fit of every trait's models as one request (``fit_request``), so the models of different traits that
    share training rows and covariates share their store passes (``stage2_wiring.model_groups``); each trait's models
    are then saved on their own (``artifact.model_subset``), in the layout a fit of that trait alone has, so the score,
    effects and report steps read one model directory per (arm, trait). ``artifact.save_model`` renames a complete model
    into place, so a model directory that exists is whole, and each saved model's outcome is recorded beside it, from
    the saved model. A rerun after a preemption fits only an arm with a trait not on disk, and every group of that arm
    whose models all finished resumes from their checkpoints (``stage2_wiring.fit_models``).

    A run given ``fit_arms`` (``run_pipeline``) fits only those arms and leaves the step to the run that finds every arm
    on disk (the run mirror carries each run's models to it), so the arms can be fitted at once on their own machines.

    The arms run in ARMS order, each whole before the next, so a run its budget stops (``launcher/workspace``: each
    job's runtime is bounded) holds the first arm's every trait before the second begins; the comparison of the arms
    needs both. After each arm the log gives the hours its fit took and the step's projected total."""
    cohort = _read_cohort(run)
    # The measurement model reaches the fit as the records' log reliabilities (its scales already rewrote the store's
    # dosages in the measurement step). The fit has no term for its residual variances or its leakage maps, and none for
    # a target variance, so it is not handed either.
    measurement = run.bindings.load_measurement_model(_final_measurement_model(run))
    records: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    fitted_seconds: list[float] = []
    trait_model_names = {trait: tuple(cohort.model_names[model] for model in cohort.trait_models(trait)) for trait in range(len(cohort.trait_names))}
    with DosageStore.open(_final_store(run)) as store:
        others = []
        for position, arm in enumerate(ARMS):
            targets = [directory / "models" / arm / trait_name for trait_name in cohort.trait_names]
            unrecorded = [trait for trait, target in enumerate(targets) if not target.with_name(target.name + ".json").is_file()]
            if unrecorded and run.fit_arms is not None and arm not in run.fit_arms:
                others.append(arm)
                continue
            if unrecorded:
                work = directory / "work" / arm
                # A Spot restart keeps completed models' checkpoints; the engine discards only the interrupted group's
                # scratch and verifies each checkpoint's input digest.
                work.mkdir(parents=True, exist_ok=True)
                unsaved = [trait for trait in unrecorded if not targets[trait].is_dir()]
                if unsaved:
                    kept = fit_filtered(
                        store.variant_table, store.n_samples, measurement.log_reliability, run.config.fit_minimum_minor_allele_frequency, run.config.fit_minimum_r2,
                        run.config.fit_variant_sites, run.config.fit_region,
                    )
                    request = fit_request(
                        cohort, store, arm_log_reliability(store.variant_table, kept, arm), run.budget, work, run.config.seed,
                        run.config.disease_likelihood, run.config.annotated_prior, run.config.length_by_class, run.config.ld_score_prior,
                    )
                    LOGGER.info("workspace pipeline: fitting %s: %d models of %d traits", arm, len(request.model_names), len(cohort.trait_names))
                    started = time.perf_counter()
                    fitted = run.bindings.fit(request)
                    if tuple(fitted.model_names) != request.model_names:
                        raise ValueError(f"the {arm} fit returned other models than the cohort's training sets.")
                    for trait in unsaved:
                        targets[trait].parent.mkdir(parents=True, exist_ok=True)
                        run.bindings.save_model(targets[trait], model_subset(fitted, trait_model_names[trait]))
                    fitted_seconds.append(time.perf_counter() - started)
                    remaining = len(ARMS) - position - 1
                    LOGGER.info(
                        "workspace pipeline: the %s fit took %.2f h in this run; at this run's mean of %.2f h an arm, the %d arms "
                        "left need %.1f h more", arm, fitted_seconds[-1] / 3600.0, np.mean(fitted_seconds) / 3600.0, remaining,
                        remaining * np.mean(fitted_seconds) / 3600.0,
                    )
                timings = checkpoint_timings(work)
                for trait in unrecorded:
                    fitted = run.bindings.load_model(targets[trait])
                    if tuple(fitted.model_names) != trait_model_names[trait]:
                        raise ValueError(f"the {arm} fit of {cohort.trait_names[trait]} returned other models than the trait's training sets.")
                    models = set(cohort.trait_models(trait).tolist())
                    _write_json(
                        targets[trait].with_name(targets[trait].name + ".json"),
                        {
                            "models": list(fitted.model_names),
                            # Each checkpointed model's share of its group's time, by its index among the cohort's models.
                            "completed_fold_timings": [timing for timing in timings if timing["model_index"] in models],
                            "fitted_records": [int(scoring.store_rows.shape[0]) for scoring in fitted.scoring],
                            CERTIFICATE_STATUS: [bool(met) for met in np.asarray(fitted.certificate[CERTIFICATE_STATUS])],
                            "refusals": [str(refusal) for refusal in fitted.refusals],
                            # model_acceptance's verdict, recorded when the fit ends so the summary names an uncertified
                            # trait at once and the score step leaves out an unscorable one before reading the store.
                            "acceptance": acceptance(fitted),
                        },
                    )
                _remove_own(work, directory)
            for target, trait_name in zip(targets, cohort.trait_names):
                records[arm][trait_name] = json.loads(target.with_name(target.name + ".json").read_text(encoding="utf-8"))
    if others:
        raise StepPart(f"the fit holds its arms but {others}, which are another run's.")
    _remove_own(directory / "work", directory)
    return {
        "arms": {
            arm: {
                "models": sum(len(record["models"]) for record in by_trait.values()),
                "outer_criterion_met": sum(sum(record[CERTIFICATE_STATUS]) for record in by_trait.values()),
                "refusals": sum(len(record["refusals"]) for record in by_trait.values()),
                "uncertified": [trait_name for trait_name, record in by_trait.items() if not record["acceptance"]["accepted"]],
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
    values [arms, cohort rows, traits] (NaN where the person has no held-out prediction of the trait),
    ``structural_genetic`` [cohort rows, traits], the genetic score of the structural part (STRUCTURAL_PART) of each
    trait's all-rows ``with_sv`` model, and ``held_out_structural`` [cohort rows, traits], each person's own-fold
    ``with_sv`` model's structural part, as ``held_out_genetic`` is its whole score (the report's stacked baselines: the
    all-rows model trained on everyone, so its part is never a held-out score)."""

    predictive_mean: F64Array
    linear_predictor: F64Array
    genetic_mean: F64Array
    held_out_predictive_mean: F64Array
    held_out_linear_predictor: F64Array
    held_out_genetic: F64Array
    structural_genetic: F64Array
    held_out_calibrated_logits: F64Array | None = None
    held_out_structural: F64Array | None = None


def score_batches(model_bytes: Sequence[int], host_bytes: int) -> list[range]:
    """Consecutive runs of the saved fitted models whose files together hold at most a quarter of ``host_bytes`` (each
    run at least one model): loaded together, they are scored from one read of the store, and the scorer's plan and
    accumulators take the rest of the host. Every model at once held some 200 GB at All of Us scale (2 arms x 15
    diseases x 6 models, each ~20M records of means, scales, coefficients and draw-law parameters)."""
    batches: list[range] = []
    start, held = 0, 0
    for position, size in enumerate(model_bytes):
        if size > host_bytes:
            raise MemoryError(f"saved model {position} exceeds the host budget before it is loaded.")
        if position > start and held + size > host_bytes // 4:
            batches.append(range(start, position))
            start, held = position, 0
        held += size
    if start < len(model_bytes):
        batches.append(range(start, len(model_bytes)))
    return batches


def _array_bytes(value: Any) -> int:
    """Bytes of the distinct NumPy allocations reachable from the cohort and fitted models, without drawing a law.

    Model subsets share arrays, and a view can keep its whole parent alive; count each owning allocation once.
    Draw laws are dataclasses just as fitted models are, so their parameter arrays, not variants x draws made from
    them, are the resident bytes. Python metadata and the store's mapped columns are outside this array count.
    """
    seen: set[int] = set()

    def size(item: Any) -> int:
        if isinstance(item, np.ndarray):
            while isinstance(item.base, np.ndarray):
                item = item.base
            if id(item) in seen:
                return 0
            seen.add(id(item))
            return int(item.nbytes)
        if dataclasses.is_dataclass(item):
            return sum(size(getattr(item, field.name)) for field in dataclasses.fields(item))
        if isinstance(item, Mapping):
            return sum(size(part) for part in item.values())
        if isinstance(item, (tuple, list)):
            return sum(size(part) for part in item)
        return 0

    return size(value)


def _prediction_budget(budget: ComputeBudget, cohort: _Cohort, models: Sequence[Any], columns: Mapping[str, Any], structural: Any) -> ComputeBudget:
    """The scorer's workspace after the loaded arrays and its caller's unmetered temporaries are accounted for.

    The saved-file batching limit alone leaves the scorer entitled to the full host again. Its plan can hold a
    concatenation, a unique-work copy, the union and per-model positions; the prediction wrapper holds fixed design,
    linear variances and the three prediction arrays beside the scorer's result. During variance reduction the
    scorer's draw deviations stay live beside the wrapper's deviations and squared deviations: two draw-wide
    temporaries beyond the one the scorer accounts for. The remaining host is shared by the scorer's ledger and
    the store reader's ledger, which are separate outside a memory scope. This leaves the reader room for its thread-dependent ring even where it is wider than the scorer's assumed ring.
    """
    scorings = [scoring for model in models for scoring in model.scoring]
    plan_bytes = 4 * sum(scoring.store_rows.nbytes for scoring in scorings)
    prediction_columns = 4 * len(scorings)
    reduction_columns = 2 * max((scoring.draw_count for scoring in scorings), default=0)
    wrapper_bytes = np.dtype(np.float64).itemsize * cohort.store_columns.size * (prediction_columns + reduction_columns + cohort.covariates.shape[1])
    remaining = budget.host_bytes - _array_bytes((cohort, models, columns, structural)) - plan_bytes - wrapper_bytes
    if remaining <= 0:
        raise MemoryError("loaded models and prediction arrays leave no host memory for scoring; use a smaller model batch.")
    return dataclasses.replace(budget, host_bytes=remaining // 2)


def _fit_verdicts(run: _Run, cohort: _Cohort, unscorable: bool) -> dict[str, list[str]]:
    """From the acceptance each fit record holds (written when the fit ended): with ``unscorable``, the traits that
    cannot be scored (``model_acceptance.unscorable_codes``: non-finite parameters, or a fit that failed and returned its
    covariates alone), else the traits with an uncertified fit; each with its arms and their models' fatal codes. A trait
    is compared across the arms, so an unscorable model in either arm takes the trait out of every table. An uncertified
    trait is scored and reported like any other: held-out accuracy measures a fit whatever its convergence."""
    found: dict[str, list[str]] = {}
    for arm in ARMS:
        for trait_name in cohort.trait_names:
            folder = _fitted_model_directory(run, arm, trait_name)
            verdict = json.loads(folder.with_name(folder.name + ".json").read_text(encoding="utf-8"))["acceptance"]
            codes = unscorable_codes(verdict) if unscorable else sorted({issue["code"] for issue in verdict["issues"] if issue["severity"] == "fatal"})
            if codes:
                found.setdefault(trait_name, []).append(f"{arm} ({', '.join(codes)})")
    return found


def _score_step(run: _Run, directory: Path) -> dict[str, Any]:
    """Every model of both arms scored for the whole cohort (``predict_models``), in as few reads of the store as the
    host's memory allows the saved models to be held together (``score_batches``). An unscorable trait
    (``_fit_verdicts``) is not read: its scores stay NaN, and the step's summary names it with its codes under
    ``quarantined``; the summary names each uncertified trait, scored like the rest, under ``uncertified``.

    Each completed model batch is durable in the keyed partial: a restart fills its columns from that batch before
    loading any fitted model. The exact ordered membership and output columns belong to the batch key, so a different
    memory plan cannot mistake an earlier batch number for its own models. Final calibration and held-out assembly
    run again from these same float64 values; neither the prediction call nor its arithmetic changes. A batch's read of
    the store is kept part way through as well (``postfit_checkpoint.batch_reads``), so a restart reads only its rest
    and scores it to the bit; its kept reads go once the batch is committed."""
    from sv_pgs.phase_timer import timing
    from sv_pgs.postfit_checkpoint import BatchCheckpoint, batch_reads, discard_reads

    step_key = (directory / KEY_FILE).read_text(encoding="utf-8")
    cohort = _read_cohort(run)
    quarantined = _fit_verdicts(run, cohort, unscorable=True)
    uncertified = {name: arms for name, arms in _fit_verdicts(run, cohort, unscorable=False).items() if name not in quarantined}
    for trait_name, arms in quarantined.items():
        LOGGER.warning("workspace pipeline: %s is not scored or reported: its %s fit cannot be scored", trait_name, "; ".join(arms))
    for trait_name, arms in uncertified.items():
        LOGGER.warning("workspace pipeline: %s is scored and reported, uncertified: %s", trait_name, "; ".join(arms))
    trait_count = len(cohort.trait_names)
    scored = [(arm_index, trait) for arm_index in range(len(ARMS)) for trait in range(trait_count) if cohort.trait_names[trait] not in quarantined]
    directories = [_fitted_model_directory(run, ARMS[arm_index], cohort.trait_names[trait]) for arm_index, trait in scored]
    shape = (len(ARMS), cohort.store_columns.shape[0], len(cohort.model_names))
    columns = {name: np.full((shape[1], shape[0] * shape[2]), np.nan) for name in ("predictive_mean", "predictive_variance", "linear_predictor", "genetic_mean")}
    structural_genetic = np.full((shape[1], trait_count), np.nan)
    # [cohort rows, models]: the structural part of each with_sv model that holds someone out.
    model_structural = np.full((shape[1], shape[2]), np.nan)
    batches = score_batches([sum(path.stat().st_size for path in folder.iterdir()) for folder in directories], run.budget.host_bytes)
    restored_batches = 0
    with DosageStore.open(_final_store(run)) as store:
        structural = structural_records(store.variant_table) if batches else None
        for batch in batches:
            batch_models = [scored[position] for position in batch]
            targets = [arm_index * shape[2] + int(index) for arm_index, trait in batch_models for index in cohort.trait_models(trait)]
            part_traits = [trait for arm_index, trait in batch_models if ARMS[arm_index] == "with_sv"]
            part_models = [int(index) for trait in part_traits for index in cohort.trait_models(trait) if cohort.held_out[:, index].any()]
            checkpoint = BatchCheckpoint(directory, step_key, {
                "models": batch_models, "targets": targets, "parts": part_traits, "model_parts": part_models, "shape": shape,
            }, ("predictions.npz",))
            saved = checkpoint.read()
            if saved is not None:
                with np.load(saved / "predictions.npz", allow_pickle=False) as arrays:
                    for name in columns:
                        columns[name][:, targets] = arrays[name]
                    structural_genetic[:, part_traits] = arrays["structural_genetic"]
                    model_structural[:, part_models] = arrays["model_structural"]
                restored_batches += 1
                discard_reads(directory, checkpoint)
                continue
            started = time.perf_counter()
            fitted = [_checked_trait_model(run, cohort, ARMS[scored[position][0]], scored[position][1], directories[position]) for position in batch]
            # Each fitted model's column in [arms x models] (arm-major), from its own names.
            for position, model in zip(batch, fitted):
                require_scorable(model)
                arm_index, trait = scored[position]
                expected = tuple(cohort.model_names[index] for index in cohort.trait_models(trait))
                if tuple(model.model_names) != expected:
                    raise ValueError(f"the {ARMS[arm_index]} model of {cohort.trait_names[trait]} holds {model.model_names}, not its trait's models {expected}.")
            # The structural part of each trait's all-rows with_sv model, by trait, and of each of its models that
            # holds someone out, by model, scored in the same read as the models themselves, after them.
            parts = {}
            for position, model in zip(batch, fitted):
                arm_index, trait = scored[position]
                if ARMS[arm_index] == "with_sv":
                    all_rows_name = cohort.model_names[cohort.all_rows_model(trait)]
                    parts[trait] = model_part(model, all_rows_name, structural, f"{all_rows_name}/{STRUCTURAL_PART}")
            held_parts = {}
            for position, model in zip(batch, fitted):
                arm_index, trait = scored[position]
                for index in (part_models if ARMS[arm_index] == "with_sv" else []):
                    if cohort.model_traits[index] == trait:
                        held_parts[index] = model_part(model, cohort.model_names[index], structural, f"{cohort.model_names[index]}/{STRUCTURAL_PART}")
            models = [*fitted, *parts.values(), *held_parts.values()]
            try:
                budget = _prediction_budget(run.budget, cohort, models, columns, (structural_genetic, model_structural))
            except MemoryError:
                if len(batch) == 1:
                    raise
                batches.extend(range(position, position + 1) for position in batch)
                del fitted, parts, held_parts, models, model
                continue
            try:
                loaded = time.perf_counter()
                prediction = run.bindings.predict(models, store, cohort.store_columns, cohort.covariates[:, 1:], budget, checkpoint=batch_reads(checkpoint))
                # Production prediction checks these too; bindings must obey the same contract. A failed fused read
                # is retried one trait at a time so one overflow cannot hide the other traits' valid predictions.
                values_to_check = (prediction.predictive_mean, prediction.predictive_variance, prediction.linear_predictor, prediction.genetic.means)
                expected_shape = (shape[1], sum(len(model.model_names) for model in models))
                if any(np.shape(values) != expected_shape for values in values_to_check):
                    raise ValueError("misaligned prediction columns")
                if any(not np.all(np.isfinite(values)) for values in values_to_check) or np.any(prediction.predictive_variance < 0):
                    raise ValueError("Non-finite predictions or invalid predictive variances")
                predicted = time.perf_counter()
            except ValueError as error:
                if not str(error).startswith("Non-finite predictions"):
                    raise
                if len(batch) > 1:
                    batches.extend(range(position, position + 1) for position in batch)
                else:
                    arm_index, trait = scored[batch.start]
                    quarantined.setdefault(cohort.trait_names[trait], []).append(f"{ARMS[arm_index]} (invalid_predictions)")
                    uncertified.pop(cohort.trait_names[trait], None)
                # A previous successful iteration must not keep a model or prediction alive into the retry.
                prediction, values_to_check = None, ()
                del fitted, parts, held_parts, models, model
                continue
            for name, values in (
                ("predictive_mean", prediction.predictive_mean), ("predictive_variance", prediction.predictive_variance),
                ("linear_predictor", prediction.linear_predictor), ("genetic_mean", prediction.genetic.means),
            ):
                columns[name][:, targets] = np.asarray(values, dtype=np.float64)[:, : len(targets)]
            means = np.asarray(prediction.genetic.means, dtype=np.float64)
            structural_genetic[:, list(parts)] = means[:, len(targets) : len(targets) + len(parts)]
            model_structural[:, list(held_parts)] = means[:, len(targets) + len(parts) :]
            del fitted, parts, held_parts, prediction, models, model, values, values_to_check, means
            with checkpoint.write() as temporary:
                np.savez(temporary / "predictions.npz", **{name: values[:, targets] for name, values in columns.items()},
                         structural_genetic=structural_genetic[:, part_traits], model_structural=model_structural[:, part_models])
            discard_reads(directory, checkpoint)
            # The batch's models loaded (and checked), then read from the store in one prediction, then committed.
            timing("score_batch", time.perf_counter() - started, models=len(batch), load=loaded - started, predict=predicted - loaded)
    # A batch split after a failed read leaves that read's kept part behind.
    discard_reads(directory)
    # Comparisons need the same trait in both arms. If one arm failed during prediction, erase the other arm's
    # already computed columns too; every downstream table uses this trait-wide quarantine.
    for trait, name in enumerate(cohort.trait_names):
        if name in quarantined:
            for arm_index in range(len(ARMS)):
                targets = arm_index * shape[2] + cohort.trait_models(trait)
                for values in columns.values():
                    values[:, targets] = np.nan
            structural_genetic[:, trait] = np.nan
            model_structural[:, cohort.trait_models(trait)] = np.nan
    LOGGER.info("workspace pipeline: submitted %d models in %d prediction batches, reused %d batches",
                sum(len(cohort.trait_models(trait)) for _arm, trait in scored), len(batches) - restored_batches, restored_batches)

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
            if cohort.trait_types[trait] != TraitType.BINARY or cohort.trait_names[trait] in quarantined:
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
        held_out_structural=held_out(model_structural[None])[0],
        calibrated_logits=calibrated_logits,
        held_out_calibrated_logits=held_out(calibrated_logits),
    )
    return {
        "arms": list(ARMS), "models": sum(len(cohort.trait_models(trait)) for _arm, trait in scored if cohort.trait_names[trait] not in quarantined), "prediction_batches": len(batches),
        "scored_rows": int(cohort.store_columns.shape[0]), "quarantined": quarantined, "uncertified": uncertified,
    }


def _read_scores(run: _Run) -> _Scores:
    arrays = np.load(run.output("score") / "predictions.npz")
    if tuple(str(arm) for arm in arrays["arms"]) != ARMS:
        raise ValueError("the score step's arms are not this driver's.")
    if tuple(str(name) for name in arrays["model_names"]) != _read_cohort(run).model_names:
        raise ValueError("the score step's model names or order differ from the cohort's.")
    # A field with a default is absent from an older score step's arrays.
    return _Scores(**{field.name: arrays[field.name] for field in dataclasses.fields(_Scores) if field.name in arrays.files or field.default is dataclasses.MISSING})


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
PRIOR_REFERENCE_CLASS = VariantClass.SNV.value
"""The class every prior and posterior enrichment is relative to: SNVs, the class every arm fits."""
PRIOR_BY_LENGTH_COLUMNS = (
    "trait", "arm", "variant_class", "length_bin", "variants", "prior_variance", "prior_variance_geometric", "prior_expected_absolute_effect",
    "prior_enrichment", "posterior_second_moment", "posterior_enrichment",
)
"""``prior_by_length.tsv``: the learned prior by (trait, arm, class, length bin), read from the fitted model
(``prior_moments``), as means over the bin's fitted records: the prior variance of a record's effect on its standardized
genotype (its expected share of the trait's variance before the data, LD aside), its geometric mean, the prior's
expected |effect| per unit (one allele, or one copy), the prior variance relative to the SNVs' mean
(PRIOR_REFERENCE_CLASS), and the posterior second moment E[beta^2 | data] of the standardized effect and its enrichment
on the same reference. Each mean is over the bin's records with every other annotation (frequency, gene context) as the
genome has it, so a bin is the prior an SV of that class and length gets in this fit, not a length effect alone."""
PRIOR_ENRICHMENT_COLUMNS = (
    "trait", "arm", "variant_class", "variants", "prior_variance", "prior_enrichment", "prior_share", "posterior_second_moment",
    "posterior_enrichment", "posterior_enrichment_standard_error", "posterior_enrichment_ci_low", "posterior_enrichment_ci_high",
    "posterior_share", "posterior_share_ci_low", "posterior_share_ci_high", "variant_share", "heritability_enrichment",
    "heritability_enrichment_ci_low", "heritability_enrichment_ci_high", "jackknife_blocks",
)
"""``prior_enrichment.tsv``: per (trait, arm, class, and every structural record together), the mean prior variance of
the standardized effect and its enrichment over SNVs, the class's share of the summed prior variance, the same three of
the posterior second moment, and the posterior enrichment's delete-one-block jackknife standard error and normal
interval (``jackknife_ratio``); then the class's share of the posterior genetic variance (its records' summed
E[beta^2 | data] on standardized genotypes over every fitted record's, each record's own variance with LD between
records aside) with its jackknife interval, its share of the fitted records, and their ratio, the heritability
enrichment of LD score regression's partitioning (Finucane et al. 2015, Nat Genet 47:1228) read from the posterior,
with its jackknife interval. A structural class's row is its structural records (COMMUNITY_SV_MINIMUM_LENGTH and
longer), as every structural record together is; its shorter records are in the length table's SHORT_RECORDS bin. The
prior's enrichment has no interval: the empirical Bayes keeps point hyperparameters, so the fit holds no uncertainty
about them."""
JACKKNIFE_BLOCKS = 200
"""The posterior enrichment's jackknife blocks: contiguous runs of the fitted records in store (genome) order, as LD
score regression's 200-block jackknife (Bulik-Sullivan et al. 2015, Nat Genet 47:291), each long against LD."""


@dataclass(frozen=True, slots=True)
class PriorMoments:
    """The learned prior at each fitted record of one model (``prior_moments``): the prior variance of its effect on its
    standardized genotype, and its prior expected |effect| per unit."""

    variance: F64Array
    absolute_effect: F64Array


def prior_moments(scoring: Any, unit: F64Array) -> PriorMoments | None:
    """The prior at each of ``scoring``'s records (a ``fast_scoring.ScoringModel``; ``unit`` each record's units per
    standardized genotype, codes_per_unit / sigma), from the fitted hyperparameters its draw law carries: the mean-field
    law holds each member's log u_j = o_j + d_j' theta (its unit offset and every annotation smooth at the fitted
    coefficients) and its class's lattice masses log pi_ck on the nodes t_k (``draw_laws.ProductMixtureDraws``), so the
    prior beta_j ~ sum_k pi_ck N(0, u_j e^t_k) has variance u_j sum_k pi_ck e^t_k and expected |beta_j|
    sqrt(2 / pi) sqrt(u_j) sum_k pi_ck e^(t_k / 2), per unit times ``unit``. None for a law that does not hold the prior
    (dense draws, the full-data and small-n routes)."""
    law = scoring.posterior_draws
    if not hasattr(law, "log_density"):
        return None
    grid = np.asarray(law.log_variance_grid, dtype=np.float64)
    density = np.asarray(law.log_density, dtype=np.float64)
    classes = np.asarray(law.class_index, dtype=np.int64)
    log_scale = np.asarray(law.log_scale, dtype=np.float64)
    return PriorMoments(
        variance=np.exp(log_scale + special.logsumexp(density + grid, axis=1)[classes]),
        absolute_effect=np.sqrt(2.0 / np.pi) * np.exp(0.5 * log_scale + special.logsumexp(density + 0.5 * grid, axis=1)[classes]) * unit,
    )


def jackknife_ratio(values: F64Array, member: BoolArray, reference: BoolArray, blocks: int, means: bool = True) -> tuple[float, float, int]:
    """The mean of ``values`` over ``member`` relative to their mean over ``reference`` (with ``means`` false, their sum
    relative to their sum), its delete-one-block jackknife standard error over ``blocks`` contiguous runs of the rows
    (fewer where there are fewer rows), and the block count. The error is NaN where deleting a block leaves either set
    empty."""
    count = values.shape[0]
    blocks = min(blocks, count)
    block = np.repeat(np.arange(blocks), np.diff(np.linspace(0, count, blocks + 1).astype(np.int64)))

    def sums(mask: BoolArray) -> tuple[F64Array, F64Array]:
        return np.bincount(block[mask], weights=values[mask], minlength=blocks), np.bincount(block[mask], minlength=blocks).astype(np.float64)

    member_sum, member_count = sums(member)
    reference_sum, reference_count = sums(reference)
    if not means:
        member_count, reference_count = np.ones(blocks), np.ones(blocks)
    with np.errstate(divide="ignore", invalid="ignore"):
        estimate = (member_sum.sum() / member_count.sum()) / (reference_sum.sum() / reference_count.sum())
        deleted = ((member_sum.sum() - member_sum) / (member_count.sum() - member_count)) / (
            (reference_sum.sum() - reference_sum) / (reference_count.sum() - reference_count)
        )
    if blocks < 2 or not np.all(np.isfinite(deleted)):
        return float(estimate), float("nan"), blocks
    return float(estimate), float(np.sqrt((blocks - 1) / blocks * np.sum((deleted - deleted.mean()) ** 2))), blocks


def _hidden_records(count: int) -> str:
    """A hidden summary's record count, as a hidden cell's (``_hidden_counts``)."""
    return _hidden_counts(count, False)[0]


def effect_summaries(
    trait: str, arm: str, classes: NDArray, bins: NDArray, lengths: F64Array, effects: VariantEffects, unit: F64Array, prior: PriorMoments | None,
) -> dict[str, list[tuple[Any, ...]]]:
    """One model's summary lines by table name (EFFECT_SUMMARY_COLUMNS, LENGTH_CORRELATION_COLUMNS,
    PRIOR_BY_LENGTH_COLUMNS, PRIOR_ENRICHMENT_COLUMNS), from its fitted records' ``classes``, length ``bins`` and
    ``lengths``, their ``effects`` and ``unit``s, and the learned ``prior`` at them (no prior lines where it is None).

    Record counts are held to the participant rule, and every count a table shows is a sum of the parts of one tree, so
    no difference of shown counts isolates 1 to 20 records: every fitted record is in one part, a class's structural
    records (COMMUNITY_SV_MINIMUM_LENGTH and longer, the structural classes') or a class's shorter ones; every structural
    record together is the structural parts, and a structural part is its length bins. In each partition (the structural
    parts, the shorter parts, a structural part's bins) the parts of 1 to 20 records are hidden with the smallest others
    until the hidden ones hold 0 or more than 20 (``suppressed_levels``); a hidden structural part hides its bins. So no
    mean over 1 to 20 records follows from the shown means and counts. Where every structural record together, or every
    shorter one, is 1 to 20 records, nothing relative to every fitted record is shown (the heritability enrichment and the
    variant share would give back their count), and where the SNVs' part is hidden nothing is shown relative to them. A
    bin's positive and negative counts are parts of its records with the zero means beside them, suppressed the same
    way."""
    structural = lengths >= COMMUNITY_SV_MINIMUM_LENGTH
    structural_classes = sorted(set(classes[structural].tolist()))
    short_counts = Counter(classes[~structural].tolist())
    structural_counts = Counter(classes[structural].tolist())
    hidden_structural = suppressed_levels({name: (members,) for name, members in structural_counts.items()})
    hidden_short = suppressed_levels({name: (members,) for name, members in short_counts.items()})
    cells = Counter(zip(classes.tolist(), bins.tolist()))
    hidden_bins: set[tuple[str, str]] = {(name, SHORT_RECORDS) for name in hidden_short}
    for name in structural_classes:
        sizes = {level: (members,) for (owner, level), members in cells.items() if owner == name and level != SHORT_RECORDS}
        hidden_bins |= {(name, level) for level in (sizes if name in hidden_structural else suppressed_levels(sizes))}
    # The class rows: a structural class's structural part, any other class's whole (shorter) part.
    hidden_classes = hidden_structural | {name for name in hidden_short if name not in structural_counts}
    totals_shown = all(not 0 < int(total) < MINIMUM_REPORTED_PARTICIPANTS for total in (structural.sum(), (~structural).sum()))
    lines: dict[str, list[tuple[Any, ...]]] = {"effects_by_class.tsv": [], "effect_length_correlation.tsv": [], "prior_by_length.tsv": [], "prior_enrichment.tsv": []}
    root_mean_square = np.sqrt(effects.mean * effects.mean + effects.standard_deviation * effects.standard_deviation)
    pairs = sorted(set(zip(classes.tolist(), bins.tolist())))
    summary_blanks = ("",) * (len(EFFECT_SUMMARY_COLUMNS) - EFFECT_SUMMARY_COLUMNS.index("positive_mean"))
    for name, level in pairs:
        member = (classes == name) & (bins == level)
        members = int(member.sum())
        if (name, level) in hidden_bins:
            lines["effects_by_class.tsv"].append((trait, arm, name, level, _hidden_records(members), *summary_blanks))
            continue
        mean = effects.mean[member]
        positive, negative = int(np.sum(mean > 0.0)), int(np.sum(mean < 0.0))
        signs = suppressed_levels({"positive": (positive,), "negative": (negative,), "zero": (members - positive - negative,)})
        absolute = np.abs(mean)
        lines["effects_by_class.tsv"].append((
            trait, arm, name, level, members, "suppressed" if "positive" in signs else positive, "suppressed" if "negative" in signs else negative,
            f"{effects.probability_positive[member].sum():.6g}", f"{effects.probability_negative[member].sum():.6g}", f"{absolute.mean():.6g}",
            f"{np.median(absolute):.6g}", f"{np.sqrt(np.mean(root_mean_square[member] ** 2)):.6g}",
        ))
    for name in [*structural_classes, "all_structural"]:
        member = structural & ((classes == name) | (name == "all_structural"))
        if np.unique(lengths[member]).shape[0] < 2:
            continue
        hidden = name in hidden_classes if name != "all_structural" else int(member.sum()) < MINIMUM_REPORTED_PARTICIPANTS
        for quantity, values in (("absolute_mean", np.abs(effects.mean)), ("posterior_sd", effects.standard_deviation)):
            if hidden:
                lines["effect_length_correlation.tsv"].append((trait, arm, name, _hidden_records(int(member.sum())), quantity, "", "descriptive_prior_and_LD_dependent"))
                continue
            selected = values[member]
            rho = stats.spearmanr(np.log(lengths[member]), selected).statistic if np.ptp(selected) > 0 else float("nan")
            lines["effect_length_correlation.tsv"].append((trait, arm, name, int(member.sum()), quantity, f"{rho:.6g}", "descriptive_prior_and_LD_dependent"))
    if prior is None:
        return lines
    reference = classes == PRIOR_REFERENCE_CLASS
    second = (effects.mean * effects.mean + effects.standard_deviation * effects.standard_deviation) / (unit * unit)
    # A hidden reference shows nothing relative to it: its mean would follow from a shown mean and its enrichment.
    reference_shown = bool(reference.any()) and PRIOR_REFERENCE_CLASS not in hidden_short
    with np.errstate(divide="ignore", invalid="ignore"):
        prior_reference = float(prior.variance[reference].mean()) if reference_shown else float("nan")
        posterior_reference = float(second[reference].mean()) if reference_shown else float("nan")
    length_blanks = ("",) * (len(PRIOR_BY_LENGTH_COLUMNS) - PRIOR_BY_LENGTH_COLUMNS.index("prior_variance"))
    for name, level in pairs:
        member = (classes == name) & (bins == level)
        if (name, level) in hidden_bins:
            lines["prior_by_length.tsv"].append((trait, arm, name, level, _hidden_records(int(member.sum())), *length_blanks))
            continue
        variance = prior.variance[member]
        lines["prior_by_length.tsv"].append((
            trait, arm, name, level, int(member.sum()), _number(variance.mean()), _number(np.exp(np.mean(np.log(variance)))),
            _number(prior.absolute_effect[member].mean()), _number(variance.mean() / prior_reference), _number(second[member].mean()),
            _number(second[member].mean() / posterior_reference),
        ))
    quantile = stats.norm.ppf((1.0 + CONFIDENCE_LEVEL) / 2.0)
    every = np.ones(classes.shape[0], dtype=bool)

    def interval(estimate: float, standard_error: float) -> tuple[str, str]:
        return _number(estimate - quantile * standard_error), _number(estimate + quantile * standard_error)

    class_blanks = ("",) * (len(PRIOR_ENRICHMENT_COLUMNS) - PRIOR_ENRICHMENT_COLUMNS.index("prior_variance"))
    for name in [*sorted(set(classes.tolist())), *(["all_structural"] if structural.any() else [])]:
        if name == "all_structural":
            member, hidden = structural, not totals_shown or int(structural.sum()) < MINIMUM_REPORTED_PARTICIPANTS
        else:
            member, hidden = (classes == name) & (structural if name in structural_counts else ~structural), name in hidden_classes
        members = int(member.sum())
        if hidden:
            lines["prior_enrichment.tsv"].append((trait, arm, name, _hidden_records(members), *class_blanks))
            continue
        enrichment, standard_error, blocks = jackknife_ratio(second, member, reference, JACKKNIFE_BLOCKS) if reference_shown else (np.nan, np.nan, 0)
        share, share_error, _blocks = jackknife_ratio(second, member, every, JACKKNIFE_BLOCKS, means=False)
        heritability, heritability_error, blocks = jackknife_ratio(second, member, every, JACKKNIFE_BLOCKS) if totals_shown else (np.nan, np.nan, blocks)
        lines["prior_enrichment.tsv"].append((
            trait, arm, name, members, _number(prior.variance[member].mean()), _number(prior.variance[member].mean() / prior_reference),
            _number(prior.variance[member].sum() / prior.variance.sum()), _number(second[member].mean()), _number(enrichment),
            _number(standard_error), *interval(enrichment, standard_error), _number(share), *interval(share, share_error),
            _number(members / classes.shape[0]) if totals_shown else "", _number(heritability), *interval(heritability, heritability_error), blocks,
        ))
    return lines


LOCUS_VARIANCE_COLUMNS = (
    "trait", "arm", "locus", "kind", "chromosome", "start", "end", "variant_class", "variants", "rare_left_out", "genetic_variance", "genome_share",
    "window_share",
)
"""``locus_variance.tsv`` (with the config's ``locus_panel``): per trait, arm, panel window bearing on the trait and
variant class in it (then ``all`` its classes), the all-rows model's genetic variance there, the summed posterior
E[beta^2 | data] on standardized genotypes (each record's beta^2 times its genotype variance, LD between records aside),
its share of the model's whole genome's and of the window's. A record, structural or not, enters only with at least
MINIMUM_REPORTED_PARTICIPANTS expected carriers of its minor allele among the model's training rows (``locus_variance``);
the others are counted in ``rare_left_out`` and in no value, so no value rests on fewer carriers. The counts are of
records, never of participants."""


def read_locus_panel(path: str | Path) -> list[dict[str, Any]]:
    """The panel's windows (``WorkspaceConfig.locus_panel``), each with its name, kind, store chromosome, first and last
    position (both included) and traits."""
    loci = json.loads(Path(path).read_text(encoding="utf-8"))["loci"]
    for locus in loci:
        if not {"name", "kind", "chromosome", "start", "end", "traits"} <= set(locus) or not int(locus["start"]) <= int(locus["end"]):
            raise ValueError(f"{path}: a locus needs its name, kind, chromosome, start <= end and traits: {locus.get('name')!r}.")
    return loci


def locus_variance(
    trait: str, arm: str, loci: Sequence[Mapping[str, Any]], chromosomes: I64Array, positions: I64Array, classes: NDArray, second: F64Array,
    carriers: F64Array,
) -> list[tuple[Any, ...]]:
    """One model's ``locus_variance.tsv`` lines over its fitted records (their store ``chromosomes``, ``positions``,
    ``classes``, posterior second moments on standardized genotypes ``second``, and expected minor-allele ``carriers``)
    at the panel windows that list ``trait``: a record is in a window when its first base is. Carriers are read as the
    training rows times the minor allele frequency, the fewest the minor allele copies allow (every copy in a
    homozygote), so a record kept here has at least MINIMUM_REPORTED_PARTICIPANTS of them in expectation."""
    total = float(second.sum())
    rare = carriers < MINIMUM_REPORTED_PARTICIPANTS
    lines: list[tuple[Any, ...]] = []
    for locus in loci:
        if trait not in locus["traits"]:
            continue
        window = (chromosomes == int(locus["chromosome"])) & (positions >= int(locus["start"])) & (positions <= int(locus["end"]))
        kept = window & ~rare
        window_total = float(second[kept].sum())
        head = (trait, arm, locus["name"], locus["kind"], int(locus["chromosome"]), int(locus["start"]), int(locus["end"]))
        for name in [*sorted(set(classes[window].tolist())), "all"]:
            member = window if name == "all" else window & (classes == name)
            variance = float(second[member & kept].sum())
            shown = bool((member & kept).any())
            lines.append((
                *head, name, int((member & kept).sum()), int((member & rare).sum()), _number(variance) if shown else "",
                _number(variance / total) if shown and total > 0.0 else "", _number(variance / window_total) if shown and window_total > 0.0 else "",
            ))
    return lines


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
    ``effect_length_correlation.tsv``), the learned prior by class and length and its enrichment over SNVs beside the
    posterior's (``prior_by_length.tsv``, ``prior_enrichment.tsv``: ``prior_moments``, ``effect_summaries``), and the
    structural records' share of the ``with_sv`` score's variance over each trait's observed rows, from the score step's
    scores (``genetic_variance.tsv``, GENETIC_VARIANCE_COLUMNS). Only summaries the export check passes are listed as
    exportable. A trait the score step left out as unscorable (``_fit_verdicts``) has no effects here.

    A batch is one (arm, trait)'s all-rows model: its compressed per-record table and its already formatted summary
    rows are committed together. A restart reuses only whole, matching batches, in the original arm/trait order;
    the small score-derived variance table and final summary tables are assembled again. Per-record tables link to
    their durable copies on the local disk, so keeping a partial does not hold a second copy of the largest files."""
    from sv_pgs.postfit_checkpoint import BatchCheckpoint

    step_key = (directory / KEY_FILE).read_text(encoding="utf-8")
    cohort = _read_cohort(run)
    scores = _read_scores(run)
    quarantined = run.summary("score")["quarantined"]
    reported = [(trait, trait_name) for trait, trait_name in enumerate(cohort.trait_names) if trait_name not in quarantined]
    variance_lines: list[tuple[Any, ...]] = [GENETIC_VARIANCE_COLUMNS]
    for trait, trait_name in reported:
        observed = np.flatnonzero(np.isfinite(cohort.targets[:, trait]))
        count = observed.shape[0]
        decomposition = genetic_variance_decomposition(
            scores.genetic_mean[ARMS.index("with_sv"), observed, cohort.all_rows_model(trait)], scores.structural_genetic[observed, trait]
        )
        variance_lines.extend(
            (trait_name, reportable_count(count), quantity, f"{value:.6g}" if _shown(count, None) else "") for quantity, value in decomposition.items()
        )
    summaries: dict[str, list[tuple[Any, ...]]] = {
        "effects_by_class.tsv": [EFFECT_SUMMARY_COLUMNS], "effect_length_correlation.tsv": [LENGTH_CORRELATION_COLUMNS],
        "prior_by_length.tsv": [PRIOR_BY_LENGTH_COLUMNS], "prior_enrichment.tsv": [PRIOR_ENRICHMENT_COLUMNS],
    }
    loci = [] if run.config.locus_panel is None else read_locus_panel(run.config.locus_panel)
    if run.config.locus_panel is not None:
        summaries["locus_variance.tsv"] = [LOCUS_VARIANCE_COLUMNS]
    priors = 0
    if reported:
        with DosageStore.open(_final_store(run)) as store:
            table = store.variant_table
            classes = np.asarray(VARIANT_CLASS_LEGEND, dtype=object)[np.asarray(table.variant_class)]
            lengths = np.asarray(table.annotations["sv_length"], dtype=np.float64)
            bins = length_bins(lengths)
            # A store without direct rows holds imputed records only.
            row_sources = np.asarray(ROW_SOURCES, dtype=object)[
                np.asarray(table.annotations[ROW_SOURCE], dtype=np.int64) if ROW_SOURCE in table.annotations else np.zeros(lengths.shape[0], dtype=np.int64)
            ]
            codes_per_unit = np.asarray(table.codes_per_unit, dtype=np.float64)
            working_bytes = run.budget.host_bytes // 2
            for arm in ARMS:
                for trait, trait_name in reported:
                    table_name = f"{trait_name}.{arm}.tsv.gz"
                    checkpoint = BatchCheckpoint(directory, step_key, {"arm": arm, "trait": trait_name}, (table_name, "summary.json"))
                    saved = checkpoint.read()
                    if saved is None:
                        with checkpoint.write() as temporary:
                            fitted = _checked_trait_model(run, cohort, arm, trait, _fitted_model_directory(run, arm, trait_name))
                            require_scorable(fitted)
                            scoring = fitted.scoring[list(fitted.model_names).index(cohort.model_names[cohort.all_rows_model(trait)])]
                            effects = variant_effects(scoring, codes_per_unit, working_bytes)
                            rows = effects.store_rows
                            pd.DataFrame(
                                {
                                    "chromosome": np.asarray(table.chromosome)[rows].astype(np.int64),
                                    "position": np.asarray(table.position)[rows],
                                    "id": table.variant_ids(rows),
                                    "variant_class": classes[rows],
                                    "length": lengths[rows],
                                    "row_source": row_sources[rows],
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
                                temporary / table_name, sep="\t", index=False, float_format="%.6g",
                                # gzip's fastest level: a genome's records are tens of millions of rows per table.
                                compression={"method": "gzip", "compresslevel": 1},
                            )
                            unit = codes_per_unit[rows] / np.asarray(scoring.signed_scales, dtype=np.float64)
                            prior = prior_moments(scoring, unit)
                            tables = effect_summaries(trait_name, arm, classes[rows], bins[rows], lengths[rows], effects, unit, prior)
                            if loci:
                                frequency = np.asarray(table.sum_code, dtype=np.float64)[rows] / (2.0 * CODES_PER_DOSAGE * store.n_samples)
                                trained = int(cohort.training[:, cohort.all_rows_model(trait)].sum())
                                tables["locus_variance.tsv"] = locus_variance(
                                    trait_name, arm, loci, np.asarray(table.chromosome)[rows].astype(np.int64), np.asarray(table.position)[rows].astype(np.int64),
                                    classes[rows], (effects.mean * effects.mean + effects.standard_deviation * effects.standard_deviation) / (unit * unit),
                                    trained * np.minimum(frequency, 1.0 - frequency),
                                )
                            _write_json(temporary / "summary.json", {"tables": tables, "prior": prior is not None})
                        saved = checkpoint.path
                    record = json.loads((saved / "summary.json").read_text(encoding="utf-8"))
                    for name, lines in record["tables"].items():
                        summaries[name].extend(lines)
                    priors += bool(record["prior"])
                    output = directory / table_name
                    output.unlink(missing_ok=True)
                    os.link(saved / table_name, output)
    summaries["genetic_variance.tsv"] = variance_lines
    for name, lines in summaries.items():
        with (directory / name).open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle, delimiter="\t").writerows(lines)
    exportable, refused = _checked_exportable(directory, list(summaries))
    _write_json(directory / EXPORTABLE_FILE, exportable)
    return {
        "files": exportable, "not_exportable": refused,
        "variant_tables": f"{len(ARMS) * len(reported)} per-record effect tables, kept in the workspace",
        "prior": f"the learned prior read from {priors} of {len(ARMS) * len(reported)} models' draw laws (mean-field models hold it; others have none)",
    }


ACCURACY_COLUMNS = ("trait", "arm", "cell", "level", "n", "cases", "metric", "value")
COMPARISON_COLUMNS = ("trait", "cell", "level", "n", "cases", "metric", "estimate", "standard_error", "z", "p_value", "inference")
PARTICIPANT_COUNT_COLUMNS = ("n", "cases")
"""The report columns that count participants, which the All of Us rule never lets show 1 to 20."""


def _residualized(values: F64Array, covariates: F64Array) -> F64Array:
    """``values`` less their least-squares projection on ``covariates`` (whose span holds the intercept)."""
    return values - covariates @ np.linalg.lstsq(covariates, values, rcond=None)[0]


def _partial_r2(target: F64Array, score: F64Array, covariates: F64Array) -> float:
    """The squared correlation of a score with a target after both are residualized on the covariates."""
    if not target.size or not all(np.all(np.isfinite(values)) for values in (target, score, covariates)):
        return float("nan")
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


def held_out_cells(observed: BoolArray, folds: I64Array, evaluated: Sequence[int], groupings: Mapping[str, Sequence[str]]) -> list[tuple[str, str, _HeldOutCell]]:
    """The report cells of one trait's held-out rows (``observed`` rows in the ``evaluated`` folds): all of them, each
    evaluated fold, then each level of every grouping in ``groupings`` (a label per row, by the grouping's name)."""

    def cell(member: BoolArray) -> _HeldOutCell:
        return _HeldOutCell(tuple(np.flatnonzero(observed & member & (folds == fold)) for fold in evaluated))

    cells = [("all", "all", cell(np.ones(folds.shape[0], dtype=bool)))] + [("fold", str(fold), cell(folds == fold)) for fold in evaluated]
    for grouping, labels in groupings.items():
        labelled = np.asarray(labels)
        cells += [(grouping, label, cell(labelled == label)) for label in sorted(set(labelled.tolist()))]
    return cells


def _held_out_rows(cohort: _Cohort, trait: int) -> BoolArray:
    """A trait's rows that one of its models holds out and that observe it: the people its held-out metrics are read on."""
    return np.isfinite(cohort.targets[:, trait]) & cohort.held_out[:, cohort.trait_models(trait)].any(axis=1)


def _cells(cohort: _Cohort, trait: int, fold_count: int) -> list[tuple[str, str, _HeldOutCell]]:
    """Every report cell of a trait: all its held-out rows, each evaluated fold, each ancestry group and each half."""
    return held_out_cells(
        _held_out_rows(cohort, trait), cohort.folds, cohort.evaluated_folds(trait, fold_count), {"ancestry": cohort.ancestry, "half": cohort.half}
    )


def suppressed_levels(parts: Mapping[str, Sequence[int]]) -> set[str]:
    """The levels of a partition that are not shown beside its total, each level given by its counts (rows; a disease's
    cases and controls beside them): every level with a count of 1 to 20, then the smallest of the others until each
    count's suppressed total is 0 or at least MINIMUM_REPORTED_PARTICIPANTS. ``all_of_us.reportable_partition``'s rule
    over several counts at once, so the total less the shown levels gives no count of 1 to 20 in any of them."""
    suppressed = {level for level, counts in parts.items() if any(0 < count < MINIMUM_REPORTED_PARTICIPANTS for count in counts)}
    others = sorted((counts[0], level) for level, counts in parts.items() if level not in suppressed and counts[0])
    width = len(next(iter(parts.values()), ()))

    def leaking() -> bool:
        return any(0 < sum(parts[level][index] for level in suppressed) < MINIMUM_REPORTED_PARTICIPANTS for index in range(width))

    while others and leaking():
        suppressed.add(others.pop(0)[1])
    return suppressed


def _hidden_counts(count: int, binary: bool, hidden: str = "rows") -> tuple[str, str]:
    """The counts a hidden cell (``_partition_hidden``) shows: its rows where only its cases are hidden (``hidden``
    "cases"), or where they are 1 to 20 and reportable_count writes them as suppressed itself; none of its own else."""
    return (reportable_count(count) if hidden == "cases" or count < MINIMUM_REPORTED_PARTICIPANTS else "suppressed", "suppressed" if binary else "")


def _number(value: float) -> str:
    """A table's number as the report writes it, or nothing where it is not finite."""
    return f"{value:.6g}" if np.isfinite(value) else ""


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


def _partition_hidden(cells: Sequence[tuple[str, str, _HeldOutCell]], targets: F64Array, binary: bool) -> dict[tuple[str, str], str]:
    """The cells of one trait (``_cells``) whose counts complementary suppression hides, each with "rows" where its row
    count is hidden or "cases" where only a disease's case count is (its controls follow from its rows less its cases).

    Each family of cells (the folds, the ancestry groups, the halves) partitions the trait's held-out rows, the cell
    "all", so one level's count of 1 to 20 hidden alone follows from the total less the other levels' shown counts: a
    disease's few cases leave several ancestry groups with fewer than MINIMUM_REPORTED_PARTICIPANTS. Rows, cases and
    controls are each hidden as ``all_of_us.reportable_partition`` hides the parts of a written total: every part of 1
    to 20, then the smallest others until the hidden parts hold MINIMUM_REPORTED_PARTICIPANTS, and every part of a total
    not written. A disease's total controls are written where its rows and cases are (the one follows from the other
    two)."""
    whole = next(held for cell, _level, held in cells if cell == "all")

    def cases_of(held: _HeldOutCell) -> int:
        return int(sum(int(targets[rows].sum()) for rows in held.folds))

    totals = {"rows": reportable_count(whole.count)}
    if binary:
        totals["cases"] = _cases_text(whole.count, cases_of(whole))
        written = totals["rows"].isdigit() and totals["cases"].isdigit()
        totals["controls"] = str(whole.count - cases_of(whole)) if written else "suppressed"
    hidden: dict[tuple[str, str], str] = {}
    for family in dict.fromkeys(cell for cell, _level, _held in cells if cell != "all"):
        members = {level: held for cell, level, held in cells if cell == family}
        parts = {"rows": {level: held.count for level, held in members.items()}}
        if binary:
            parts["cases"] = {level: cases_of(held) for level, held in members.items()}
            parts["controls"] = {level: parts["rows"][level] - parts["cases"][level] for level in members}
        for kind, counts in parts.items():
            for level, text in reportable_partition(totals[kind], counts).items():
                if not text.isdigit() and hidden.get((family, level)) != "rows":
                    hidden[(family, level)] = "rows" if kind == "rows" else "cases"
    return hidden


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


def _primary_influence(
    metric: str, targets: F64Array, linear: F64Array, genetic: F64Array, frozen: F64Array, covariates: F64Array
) -> F64Array:
    """The per-person influence of ``_paired``'s ``metric`` on one fold's rows, whose family-summed squares are its
    variance (``held_out_comparison.cluster_totals``)."""
    baseline, candidate = ARMS.index("snv_indel"), ARMS.index("with_sv")
    if metric == "log_likelihood_gain":
        return probability_calibration.frozen_gain_influence(targets, frozen[baseline], frozen[candidate], metric="log_loss")[1]
    if metric == "delta_auc":
        return held_out_comparison.delta_auc_influence(targets, linear[baseline], linear[candidate])[1]
    if metric == "delta_pgs_partial_r2":
        outcome = _residualized(targets, covariates)
        _delta, influence = held_out_comparison.delta_r2_influence(
            outcome, _residualized(genetic[baseline], covariates), _residualized(genetic[candidate], covariates)
        )
        return influence / outcome.shape[0]
    raise ValueError(f"no primary influence for {metric!r}")


BOOTSTRAP_RESAMPLES = 2000
"""The resamples behind every held-out partial r^2 interval and paired difference (``partial_r2_tables``): Efron and
Tibshirani's 1,000 to 2,000 for a percentile interval (An Introduction to the Bootstrap, 1993, section 19.3)."""
CONFIDENCE_LEVEL = 0.95
"""The two-sided level of every interval the partial r^2 tables and the prior summaries give."""
BASELINE_DIRECTORY = "baselines"
"""``<run>/baselines/<method>.npz``, or the config's ``baseline_directory``: another method's held-out genetic scores on
this run's cohort (``write_baseline_scores``; benchmarks/bench_real/aou_ldpred2.py writes LDpred2-auto's as
``ldpred2_auto.npz``), which the report compares with both arms on the same held-out people and the same covariates
(``partial_r2_tables``). One score per held-out person, so it stays in the workspace as the score step's predictions do,
and no table lists it for export."""
STACKING_SPLITS = 5
"""The parts a held-out fold's families are dealt into for a stacked score (``stacked_scores``): each part's weights are
fitted on the other four, so every weight trains on 80% of the fold's held-out people and never on the person it
scores, as five-fold cross-validation is usually run."""
STACKED_PARTS = ("with_sv", "structural")
"""What a stacked score adds to a baseline: the ``with_sv`` arm's whole held-out score, or only its structural part
(``_Scores.held_out_structural``). A stacked method is named ``<baseline>+<part>``."""
PARTIAL_R2_COLUMNS = ("trait", "method", "cell", "level", "n", "cases", "partial_r2", "standard_error", "ci_low", "ci_high", "resamples", "inference")
"""``partial_r2.tsv``: per trait, method (each arm, then each baseline, then each stacked score: ``stacked_scores``) and
cell (all held-out rows, each ancestry group), the held-out partial r^2 of the method's genetic score (``_partial_r2``:
the squared correlation with the outcome after both are residualized on the trait's covariates, PC1-16, age, age^2,
age x female and sex among them; a disease's 0/1 outcome is read the same way), with its family-clustered bootstrap
standard error and percentile interval."""
PAIRED_DIFFERENCE_COLUMNS = (
    "trait", "candidate", "baseline", "cell", "level", "n", "cases", "estimate", "standard_error", "ci_low", "ci_high", "z", "p_value",
    "p_bootstrap", "relative_estimate", "relative_ci_low", "relative_ci_high", "resamples", "inference",
)
"""``paired_differences.tsv``: per trait, pair of methods and cell, the candidate's partial r^2 less the baseline's on the
same held-out people, from the same resamples: its bootstrap standard error and percentile interval, z = estimate / SE
with its two-sided normal p, the bootstrap's own two-sided p (twice the smaller share of resampled differences on either
side of 0, each counted with the estimate itself), and the difference relative to the baseline's partial r^2."""
META_ANALYSIS_COLUMNS = (
    "candidate", "baseline", "cell", "level", "model", "traits", "estimate", "standard_error", "ci_low", "ci_high", "z", "p_value",
    "heterogeneity_q", "heterogeneity_p", "i_squared", "tau_squared", "positive_traits", "covariance",
)
"""``meta_analysis.tsv``: per pair of methods and cell (all held-out rows, each ancestry group), the traits' held-out
partial r^2 differences combined (``held_out_comparison.meta_analysis``), fixed and random effects, over the traits
whose single test fold gives each a p-value there, with the traits' covariance from the joint resamples (``covariance``
says which), and how many of the traits' differences are positive (descriptive)."""
TRAIT_COVARIANCE_COLUMNS = ("candidate", "baseline", "cell", "level", "trait", "other_trait", "covariance")
"""``trait_covariance.tsv``: the joint-resample covariance of every two traits' paired differences behind each
meta-analysis, so meta-analyses across runs (docs/poster/ashg2026/aou_results.py) keep each run's own correlations."""
HETEROGENEITY_COLUMNS = ("trait", "candidate", "baseline", "cell", "levels", "estimates", "q", "degrees", "p_value")
"""``heterogeneity.tsv``: per trait, and for the traits combined (trait ``combined``: each level's fixed effect over the
traits), whether a pair's paired difference differs between a grouping's levels (each ancestry group): Cochran's Q of
the levels' differences about their common value under their joint-resample covariance, on levels - 1 degrees of
freedom. The levels are those whose difference is shown with a p-value; ``estimates`` lists their differences."""


@dataclass(frozen=True, slots=True)
class HeldOutTrait:
    """One trait's held-out genetic scores on a cohort's rows, as ``partial_r2_tables`` reads them: its outcome (NaN off
    its held-out rows), its own covariates (the intercept among them), each method's genetic score (finite on the
    trait's held-out rows) and the folds its held-out rows come from."""

    name: str
    binary: bool
    targets: F64Array
    covariates: F64Array
    scores: Mapping[str, F64Array]
    folds: tuple[int, ...]


def method_pairs(methods: Sequence[str]) -> list[tuple[str, str]]:
    """The (candidate, baseline) pairs the tables compare: with_sv against snv_indel, each arm against each baseline
    method, and each stacked score against the baseline it adds to (``stacked_scores``) where that baseline is compared."""
    pairs = [("with_sv", "snv_indel")]
    for method in methods:
        if method in ARMS:
            continue
        if "+" in method:
            pairs.extend((method, baseline) for baseline in [method.split("+")[0]] if baseline in methods)
        else:
            pairs.extend((arm, method) for arm in reversed(ARMS))
    return pairs


def stacked_scores(
    targets: F64Array, covariates: F64Array, scores: Mapping[str, F64Array], structural: F64Array | None, families: I64Array, folds: I64Array,
    evaluated: Sequence[int], seed: int,
) -> dict[str, F64Array]:
    """For every baseline method among ``scores`` (one not in ARMS), each STACKED_PARTS score added to it: per evaluated
    fold, the least-squares weights of the baseline and the part given the covariates, cross-fitted over the fold's
    held-out families (``held_out_comparison.cross_fitted_stack``, STACKING_SPLITS parts), so a person's stacked score
    comes from weights fitted on other families of their fold and never on them; NaN off the held-out rows
    (``targets`` is NaN there). Every stack of a fold deals its families the same way (seeded by ``seed`` and the fold), so
    two stacked scores differ by what they add alone. ``structural`` (the with_sv arm's held-out structural part) None
    leaves the structural stack out. A stack is what the best SNV score gains from the SV-PGS score on the same people:
    the gain of ``<baseline>+structural`` over the baseline is the structural records' alone."""
    parts = {"with_sv": scores["with_sv"], "structural": structural}
    stacked: dict[str, F64Array] = {}
    for baseline, values in scores.items():
        if baseline in ARMS:
            continue
        for part in STACKED_PARTS:
            if parts[part] is None:
                continue
            result = np.full(targets.shape[0], np.nan)
            for fold in evaluated:
                rows = np.flatnonzero((folds == fold) & np.isfinite(targets))
                result[rows] = held_out_comparison.cross_fitted_stack(
                    targets[rows], covariates[rows], np.vstack([values[rows], parts[part][rows]]), families[rows], STACKING_SPLITS,
                    np.random.default_rng([seed, int(fold)]),
                )
            stacked[f"{baseline}+{part}"] = result
    return stacked


def partial_r2_tables(
    traits: Sequence[HeldOutTrait], families: I64Array, folds: I64Array, ancestry: Sequence[str], seed: int, resamples: int, working_bytes: int,
) -> tuple[dict[str, list[tuple[Any, ...]]], list[str]]:
    """The lines of ``partial_r2.tsv``, ``paired_differences.tsv``, ``meta_analysis.tsv``, ``trait_covariance.tsv`` and
    ``heterogeneity.tsv`` by file name for ``traits`` on one cohort's rows (``families``, ``folds`` and ``ancestry`` per
    row), and the baselines left out of a trait because they did not score every held-out row its cells read ("trait:
    method"; an arm's missing score is an error).

    Each trait's cells are the report's (``held_out_cells``: all held-out rows, each ancestry group), shown as the
    accuracy table shows them (``_partition_hidden``, ``_shown``), each method's partial r^2 read on each evaluated fold's
    rows with that fold's covariates and weighted by the fold's rows, as accuracy.tsv's pgs_partial_r2. Its uncertainty is
    a cluster bootstrap of the held-out people (``held_out_comparison.cluster_bootstrap_counts``: each fold's families
    drawn with replacement within it), every method scored on the same resample, so a paired difference is resampled whole
    and holds the methods' correlation; the fitted models are fixed, so, as comparison.tsv's, the inference is
    conditional on them and across several folds descriptive (no z or p). Every trait is resampled from the same
    ``seed``, so resample r weights the same families in every trait, and the meta-analysis reads the traits'
    covariance from them (traits observed on overlapping people, or genetically correlated, are not independent), as the
    test of a difference between ancestry groups reads the groups'. A cell's counts are those of the folds its partial
    r^2 is read on, as accuracy.tsv's pgs_partial_r2 row shows them, so the two tables never differ by a dropped fold."""
    # The resampled clusters: a fold's members of one family (the kinship folds keep a family in one fold, so these are
    # the families), each in its fold's stratum.
    span = int(families.max(initial=0)) + 1
    keys, clusters = np.unique(np.asarray(folds, dtype=np.int64) * span + families, return_inverse=True)
    cluster_folds = keys // span
    lower, upper = (1.0 - CONFIDENCE_LEVEL) / 2.0, (1.0 + CONFIDENCE_LEVEL) / 2.0
    chunk = max(1, min(resamples, working_bytes // (np.dtype(np.float64).itemsize * (cluster_folds.shape[0] + 2 * families.shape[0]))))
    partial_lines: list[tuple[Any, ...]] = [PARTIAL_R2_COLUMNS]
    paired_lines: list[tuple[Any, ...]] = [PAIRED_DIFFERENCE_COLUMNS]
    # An unshown cell's values, every column after its counts.
    partial_blanks = ("",) * (len(PARTIAL_R2_COLUMNS) - PARTIAL_R2_COLUMNS.index("partial_r2"))
    paired_blanks = ("",) * (len(PAIRED_DIFFERENCE_COLUMNS) - PAIRED_DIFFERENCE_COLUMNS.index("estimate"))
    # Each tested difference's resamples, by (candidate, baseline, cell, level) across traits and by (trait, candidate,
    # baseline, cell) across a grouping's levels.
    joint: dict[tuple[str, str, str, str], list[tuple[str, float, F64Array]]] = {}
    across_levels: dict[tuple[str, str, str, str], list[tuple[str, float, F64Array]]] = {}
    incomplete: list[str] = []
    for trait in traits:
        observed = np.isfinite(trait.targets)
        cells = [entry for entry in held_out_cells(observed, folds, trait.folds, {"ancestry": ancestry}) if entry[0] != "fold"]
        hidden = _partition_hidden(cells, trait.targets, trait.binary)
        # Per cell: its counts as shown, and the folds its partial r^2 is read on (none where it is not shown).
        read: list[tuple[str, str, tuple[str, str], Sequence[I64Array]]] = []
        for cell, level, held in cells:
            cases = int(sum(int(trait.targets[rows].sum()) for rows in held.folds)) if trait.binary else None
            if (cell, level) in hidden:
                read.append((cell, level, _hidden_counts(held.count, trait.binary, hidden[(cell, level)]), []))
                continue
            selected = _scored_folds(held, trait.targets, trait.binary) if _shown(held.count, cases) else []
            used, used_cases = _evaluated_counts(selected, trait.targets, trait.binary)
            shown = bool(selected) and _shown(used, used_cases)
            counts = (reportable_count(used), _cases_text(used, used_cases)) if shown else (reportable_count(held.count), _cases_text(held.count, cases))
            read.append((cell, level, counts, selected if shown else []))
        methods = []
        for method, values in trait.scores.items():
            if all(np.all(np.isfinite(values[rows])) for _cell, _level, _counts, selected in read for rows in selected):
                methods.append(method)
            elif method in ARMS:
                raise ValueError(f"{trait.name}: the {method} genetic score is missing on held-out rows the report reads.")
            else:
                incomplete.append(f"{trait.name}: {method}")
        pairs = method_pairs(methods)
        stacked = np.vstack([np.asarray(trait.scores[method], dtype=np.float64) for method in methods])

        def fold_weighted(weights: Callable[[I64Array], F64Array], selected: Sequence[I64Array]) -> F64Array:
            # [resamples, methods]: each fold's partial r^2 under the resamples' weights, weighted by the fold's
            # resampled rows.
            total, mass = 0.0, 0.0
            for rows in selected:
                fold_weights = weights(rows)
                total = total + fold_weights.sum(axis=1)[:, None] * held_out_comparison.weighted_partial_r2(
                    fold_weights, trait.covariates[rows], trait.targets[rows], stacked[:, rows]
                )
                mass = mass + fold_weights.sum(axis=1)[:, None]
            # A resample that draws none of a small cell's families has no partial r^2 there (NaN).
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.asarray(total / mass)

        points = {(cell, level): fold_weighted(lambda rows: np.ones((1, rows.shape[0])), selected)[0] for cell, level, _counts, selected in read if selected}
        replicates = {key: np.empty((resamples, len(methods))) for key in points}
        generator = np.random.default_rng(seed)
        for start in range(0, resamples, chunk):
            counts = held_out_comparison.cluster_bootstrap_counts(cluster_folds, min(chunk, resamples - start), generator)
            for cell, level, _counts, selected in read:
                if selected:
                    replicates[cell, level][start : start + counts.shape[0]] = fold_weighted(lambda rows: counts[:, clusters[rows]], selected)
        for cell, level, shown_counts, selected in read:
            single = len(selected) == 1
            inference = "conditional_on_fitted_models" if single else "descriptive_cross_fit"
            if not selected:
                partial_lines.extend((trait.name, method, cell, level, *shown_counts, *partial_blanks) for method in methods)
                paired_lines.extend((trait.name, candidate, baseline, cell, level, *shown_counts, *paired_blanks) for candidate, baseline in pairs)
                continue
            point, resampled = points[cell, level], replicates[cell, level]
            for index, method in enumerate(methods):
                finite = resampled[np.isfinite(resampled[:, index]), index]
                low, high = np.quantile(finite, [lower, upper]) if finite.shape[0] else (np.nan, np.nan)
                partial_lines.append((
                    trait.name, method, cell, level, *shown_counts, _number(point[index]),
                    _number(float(np.std(finite, ddof=1)) if finite.shape[0] > 1 else float("nan")), _number(low), _number(high), resamples, inference,
                ))
            for candidate, baseline in pairs:
                first, second = methods.index(candidate), methods.index(baseline)
                estimate = float(point[first] - point[second])
                differences = resampled[:, first] - resampled[:, second]
                finite = differences[np.isfinite(differences)]
                standard_error = float(np.std(finite, ddof=1)) if finite.shape[0] > 1 else float("nan")
                comparison = held_out_comparison.comparison_from_estimate(estimate, standard_error)
                low, high = np.quantile(finite, [lower, upper]) if finite.shape[0] else (np.nan, np.nan)
                bootstrap_p = min(1.0, 2.0 * min(np.sum(finite <= 0.0) + 1, np.sum(finite >= 0.0) + 1) / (finite.shape[0] + 1)) if finite.shape[0] else np.nan
                with np.errstate(divide="ignore", invalid="ignore"):
                    relative = differences / resampled[:, second]
                relative = relative[np.isfinite(relative)]
                relative_low, relative_high = np.quantile(relative, [lower, upper]) if relative.shape[0] else (np.nan, np.nan)
                tested = single and np.isfinite(comparison.z_score)
                paired_lines.append((
                    trait.name, candidate, baseline, cell, level, *shown_counts, _number(estimate), _number(standard_error), _number(low), _number(high),
                    _number(comparison.z_score) if tested else "", _number(2.0 * stats.norm.sf(abs(comparison.z_score))) if tested else "",
                    _number(bootstrap_p) if single else "", _number(estimate / point[second]) if np.isfinite(point[second]) and point[second] else "", _number(relative_low),
                    _number(relative_high), resamples, inference,
                ))
                if tested and standard_error > 0.0:
                    joint.setdefault((candidate, baseline, cell, level), []).append((trait.name, estimate, differences))
                    if cell != "all":
                        across_levels.setdefault((trait.name, candidate, baseline, cell), []).append((level, estimate, differences))
    meta_lines: list[tuple[Any, ...]] = [META_ANALYSIS_COLUMNS]
    covariance_lines: list[tuple[Any, ...]] = [TRAIT_COVARIANCE_COLUMNS]
    heterogeneity_lines: list[tuple[Any, ...]] = [HETEROGENEITY_COLUMNS]
    quantile = stats.norm.ppf(upper)

    def jointly(entries: list[tuple[str, float, F64Array]]) -> tuple[F64Array, F64Array] | None:
        # The entries' estimates and covariance over the resamples every one of them has, or None with fewer than two.
        resampled = np.vstack([differences for _name, _estimate, differences in entries])
        complete = resampled[:, np.all(np.isfinite(resampled), axis=0)]
        if complete.shape[1] < 2:
            return None
        return np.array([estimate for _name, estimate, _differences in entries]), np.atleast_2d(np.cov(complete))

    def heterogeneity(trait_name: str, candidate: str, baseline: str, cell: str, entries: list[tuple[str, float, F64Array]]) -> None:
        found = jointly(entries) if len(entries) > 1 else None
        if found is None:
            return
        result = held_out_comparison.meta_analysis(*found)
        heterogeneity_lines.append((
            trait_name, candidate, baseline, cell, ";".join(name for name, _estimate, _differences in entries), ";".join(_number(value) for value in found[0]),
            _number(result.heterogeneity_q), len(entries) - 1, _number(result.heterogeneity_p),
        ))

    # Each level's fixed effect over the traits, resampled with the same weights, by (candidate, baseline, cell).
    combined_levels: dict[tuple[str, str, str], list[tuple[str, float, F64Array]]] = {}
    for (candidate, baseline, cell, level), entries in joint.items():
        found = jointly(entries) if len(entries) > 1 else None
        if found is None:
            continue
        estimates, covariance = found
        result = held_out_comparison.meta_analysis(estimates, covariance)
        names = [name for name, _estimate, _differences in entries]
        for model, combined in (("fixed", result.fixed), ("random", result.random)):
            meta_lines.append((
                candidate, baseline, cell, level, model, ";".join(names), _number(combined.estimate), _number(combined.standard_error),
                _number(combined.estimate - quantile * combined.standard_error), _number(combined.estimate + quantile * combined.standard_error),
                _number(combined.z_score), _number(2.0 * stats.norm.sf(abs(combined.z_score))), _number(result.heterogeneity_q),
                _number(result.heterogeneity_p), _number(result.i_squared), _number(result.tau_squared) if model == "random" else "",
                result.positive, "joint_bootstrap",
            ))
        covariance_lines.extend(
            (candidate, baseline, cell, level, names[first], names[second], _number(covariance[first, second]))
            for first in range(len(names)) for second in range(first, len(names))
        )
        if cell != "all" and np.isfinite(result.fixed.estimate):
            precision = np.linalg.pinv(covariance, hermitian=True)
            weights = precision.sum(axis=1) / precision.sum()
            resampled = np.vstack([differences for _name, _estimate, differences in entries])
            combined_levels.setdefault((candidate, baseline, cell), []).append((level, result.fixed.estimate, weights @ resampled))
    for (trait_name, candidate, baseline, cell), entries in across_levels.items():
        heterogeneity(trait_name, candidate, baseline, cell, entries)
    for (candidate, baseline, cell), entries in combined_levels.items():
        heterogeneity("combined", candidate, baseline, cell, entries)
    tables = {
        "partial_r2.tsv": partial_lines, "paired_differences.tsv": paired_lines, "meta_analysis.tsv": meta_lines,
        "trait_covariance.tsv": covariance_lines, "heterogeneity.tsv": heterogeneity_lines,
    }
    return tables, incomplete


def _training_digest(cohort: _Cohort, model: int) -> str:
    """The digest of one model's training research IDs (``artifact.cohort_digest``)."""
    return cohort_digest([cohort.research_ids[row] for row in np.flatnonzero(cohort.training[:, model])])


def _held_out_models(cohort: _Cohort, trait: int) -> list[int]:
    """A trait's models that hold someone out: its fold models, never the all-rows model (which holds no one out)."""
    return [int(model) for model in cohort.trait_models(trait) if cohort.held_out[:, model].any()]


def write_baseline_scores(path: Path, cohort: _Cohort, scores: Mapping[str, F64Array]) -> None:
    """A method's held-out genetic scores, ``scores[trait name]`` one value per cohort row, finite on exactly the held-out
    rows of the trait's models it fitted (``_held_out_models``, whole models), as the report reads them
    (BASELINE_DIRECTORY): per trait the scored research IDs, their scores and folds, and per fitted model its name and the
    digest of its training research IDs, from which the report knows the method trained on those models' rows and scored
    their held-out people. A trait with no finite score is not written."""
    arrays: dict[str, np.ndarray] = {}
    names = []
    for name, values in scores.items():
        trait = cohort.trait_names.index(name)
        values = np.asarray(values, dtype=np.float64)
        rows = np.flatnonzero(np.isfinite(values))
        if not rows.shape[0]:
            continue
        models = [model for model in _held_out_models(cohort, trait) if np.all(np.isfinite(values[cohort.held_out[:, model]]))]
        if not np.array_equal(rows, np.flatnonzero(cohort.held_out[:, models].any(axis=1))):
            raise ValueError(f"{name}: the scores are not the held-out rows of whole models.")
        names.append(name)
        arrays |= {
            f"{name}/research_ids": np.array([cohort.research_ids[row] for row in rows], dtype=str), f"{name}/scores": values[rows],
            f"{name}/folds": cohort.folds[rows], f"{name}/models": np.array([cohort.model_names[model] for model in models], dtype=str),
            f"{name}/training": np.array([_training_digest(cohort, model) for model in models], dtype=str),
        }
    with Path(path).open("wb") as handle:
        np.savez(handle, traits=np.array(names, dtype=str), **arrays)


def _baseline_scores(run: _Run, cohort: _Cohort) -> tuple[dict[str, F64Array], dict[str, str]]:
    """Each baseline method's held-out genetic scores (BASELINE_DIRECTORY) [cohort rows, traits], NaN off the rows it
    scored, and why each other baseline file was left out. A file is read only if, for every trait of this cohort it
    scored, its models are this cohort's models of that trait that hold someone out, each trained on exactly that model's
    training rows (by digest), and its scored people are exactly those models' held-out people, each in their own fold: a
    comparison is on the same people and the same split, or it is not made."""
    directory = run.config.baselines()
    found: dict[str, F64Array] = {}
    refused: dict[str, str] = {}
    paths = sorted(directory.glob("*.npz")) if directory.is_dir() else []
    position = {research_id: row for row, research_id in enumerate(cohort.research_ids)} if paths else {}
    for path in paths:
        values = np.full(cohort.targets.shape, np.nan)
        try:
            with np.load(path) as arrays:
                for name in [str(name) for name in arrays["traits"]]:
                    if name not in cohort.trait_names:
                        continue
                    trait = cohort.trait_names.index(name)
                    held_out = {cohort.model_names[model]: model for model in _held_out_models(cohort, trait)}
                    trained = dict(zip((str(model) for model in arrays[f"{name}/models"]), (str(digest) for digest in arrays[f"{name}/training"])))
                    if not trained or any(model not in held_out or digest != _training_digest(cohort, held_out[model]) for model, digest in trained.items()):
                        raise ValueError(f"its {name} models are not this report's fold models trained on their rows")
                    rows = np.array([position.get(str(research_id), -1) for research_id in arrays[f"{name}/research_ids"]], dtype=np.int64)
                    expected = np.flatnonzero(cohort.held_out[:, [held_out[model] for model in trained]].any(axis=1))
                    if np.any(rows < 0) or not np.array_equal(np.sort(rows), expected) or not np.array_equal(arrays[f"{name}/folds"], cohort.folds[rows]):
                        raise ValueError(f"its {name} scores are not on those models' held-out people in their folds")
                    values[rows, trait] = arrays[f"{name}/scores"]
        except (KeyError, ValueError) as error:
            refused[path.stem] = f"{path.name}: {error}; it was made on another cohort or split, and is not compared."
            LOGGER.warning("workspace pipeline: baseline %s", refused[path.stem])
            continue
        found[path.stem] = values
    return found, refused


def _report_inputs(config: WorkspaceConfig) -> dict[str, Any]:
    """What the report reads beyond its upstream steps: the pre-registered panel, its bootstrap's resamples and level,
    and each baseline file's digest (BASELINE_DIRECTORY), so a baseline placed in the run after its report makes the
    report again."""
    directory = config.baselines()
    return {
        "panel": None if config.panel is None else [config.panel.primary_fold, [list(weight) for weight in config.panel.weights]],
        "bootstrap": [BOOTSTRAP_RESAMPLES, CONFIDENCE_LEVEL],
        "stacking": [STACKING_SPLITS, list(STACKED_PARTS)],
        "baselines": {path.stem: _file_digest(path) for path in sorted(directory.glob("*.npz"))} if directory.is_dir() else {},
    }


QUARANTINE_COLUMNS = ("trait", "quarantined_fits")
"""``quarantined.tsv``: each trait left out of every table as unscorable, with its arms and their models' codes."""
UNCERTIFIED_COLUMNS = ("trait", "uncertified_fits")
"""``uncertified.tsv``: each reported trait with a fit that missed certification, with its arms and their fatal codes."""
PANEL_COLUMNS = ("trait", "n", "cases", "metric", "estimate", "standard_error", "z", "p_one_sided", "p_hommel", "weight")
"""``panel.tsv``: each trait's pre-registered test on the panel's primary fold, then the panel's weighted z."""


def _panel_rows(run: _Run, cohort: _Cohort, scores: Any, quarantined: Mapping[str, Any]) -> tuple[list[tuple[Any, ...]], dict[str, Any]]:
    """The pre-registered panel test on the config's frozen primary fold.

    Each trait's test is one-sided (the with_sv arm predicts better), paired on the primary fold's held-out people:
    a disease's gain in log likelihood of its frozen probabilities (Delta log loss; its delta AUC where the run has no
    frozen calibration), a quantitative trait's delta partial r^2 of its genetic scores. One outer fold, so each z is
    exact conditional on that fold's fitted models: the pre-registered cross-fit pair term that would let every fold
    enter needs each arm's posterior covariance, which a biobank-scale fit never forms. The panel's statistic is
    w'z / sqrt(w'Rw), one-sided, with the frozen weights and R the correlation of the traits' family-summed influences
    (``held_out_comparison.panel_z``, ``influence_correlation``); the per-trait p-values are Hommel-adjusted across the
    panel. A trait whose statistic is undefined on the fold (one outcome class, no variance), or that the score step left
    out as unscorable, leaves the panel, and the summary names it."""
    panel = run.config.panel
    assert panel is not None
    weights = dict(panel.weights)
    tested: list[tuple[str, float, F64Array]] = []
    table: list[list[Any]] = []
    for trait, name in enumerate(cohort.trait_names):
        binary = cohort.trait_types[trait] == TraitType.BINARY
        targets = cohort.targets[:, trait]
        eligible = cohort.held_out[:, cohort.trait_models(trait)].any(axis=1)
        rows = np.flatnonzero(np.isfinite(targets) & eligible & (cohort.folds == panel.primary_fold))
        cases = int(targets[rows].sum()) if binary else None
        metric = ("log_likelihood_gain" if run.config.calibration_fraction else "delta_auc") if binary else "delta_pgs_partial_r2"
        linear = scores.held_out_linear_predictor[:, rows, trait]
        genetic = scores.held_out_genetic[:, rows, trait]
        frozen = np.full(linear.shape, np.nan) if scores.held_out_calibrated_logits is None else scores.held_out_calibrated_logits[:, rows, trait]
        covariates = cohort.covariates[np.ix_(rows, np.flatnonzero(cohort.covariate_columns[trait]))]
        row: list[Any] = [name, reportable_count(rows.shape[0]), _cases_text(rows.shape[0], cases), metric, "", "", "", "", "", f"{weights[name]:.6g}"]
        if name in quarantined:
            row[3] = "quarantined"
            table.append(row)
            continue
        both_classes = cases is None or min(cases, rows.shape[0] - cases) >= 2
        calibrated = metric != "log_likelihood_gain" or bool(np.all(np.isfinite(frozen)))
        if rows.shape[0] and both_classes and calibrated:
            comparison = _paired(metric, targets[rows], linear, genetic, frozen, covariates, cohort.families[rows])
            if np.isfinite(comparison.z_score) and comparison.standard_error > 0:
                influence = np.zeros(cohort.targets.shape[0])
                influence[rows] = _primary_influence(metric, targets[rows], linear, genetic, frozen, covariates)
                tested.append((name, comparison.z_score, influence))
                if _shown(rows.shape[0], cases):
                    row[4:8] = [f"{comparison.estimate:.6g}", f"{comparison.standard_error:.6g}", f"{comparison.z_score:.6g}", f"{stats.norm.sf(comparison.z_score):.6g}"]
        table.append(row)
    names = [entry[0] for entry in tested]
    summary: dict[str, Any] = {"primary_fold": panel.primary_fold, "left_out": [name for name in cohort.trait_names if name not in names]}
    if tested:
        z = np.array([entry[1] for entry in tested])
        for name, adjusted in zip(names, held_out_comparison.hommel_adjusted(stats.norm.sf(z))):
            row = table[cohort.trait_names.index(name)]
            if row[6]:
                row[8] = f"{adjusted:.6g}"
        weight = np.array([weights[name] for name in names])
        if weight.sum() > 0:
            correlation = held_out_comparison.influence_correlation([entry[2] for entry in tested], cohort.families)
            statistic = held_out_comparison.panel_z(z, weight, correlation)
            summary.update(panel_z=statistic, panel_p_one_sided=float(stats.norm.sf(statistic)), traits_in_panel=names)
            table.append(["panel", "", "", "weighted_z", "", "", f"{statistic:.6g}", f"{stats.norm.sf(statistic):.6g}", "", ""])
    return [PANEL_COLUMNS, *(tuple(row) for row in table)], summary


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

    Every trait, a disease's included, is also read on the LDpred2 baseline's scale (``partial_r2_tables``): each arm's,
    each baseline method's (``WorkspaceConfig.baselines``) and each baseline's stacked with the with_sv score or its
    structural part alone (``stacked_scores``) held-out partial r^2 with a family-clustered bootstrap interval
    (``partial_r2.tsv``), each pair's paired difference on the same resamples (``paired_differences.tsv``), the traits'
    differences combined per cell with their covariance (``meta_analysis.tsv``, ``trait_covariance.tsv``), and the test of
    a difference between ancestry groups (``heterogeneity.tsv``). A grouping's levels whose counts the all cell would give
    back are hidden with the small ones (``_partition_hidden``), and only tables the export check passes are listed as
    exportable.

    A trait the score step left out as unscorable (``_fit_verdicts``) is in no table but ``quarantined.tsv``
    (QUARANTINE_COLUMNS). An uncertified trait is in every table, and ``uncertified.tsv`` (UNCERTIFIED_COLUMNS) names it
    with its arms and their models' fatal codes.
    """
    cohort = _read_cohort(run)
    scores = _read_scores(run)
    quarantined = run.summary("score")["quarantined"]
    uncertified = run.summary("score").get("uncertified", {})
    fold_count = run.config.fold_count
    accuracy: list[tuple[Any, ...]] = [ACCURACY_COLUMNS]
    comparison: list[tuple[Any, ...]] = [COMPARISON_COLUMNS]
    for trait, name in enumerate(cohort.trait_names):
        if name in quarantined:
            continue
        binary = cohort.trait_types[trait] == TraitType.BINARY
        covariates = cohort.covariates[:, cohort.covariate_columns[trait]]
        targets = cohort.targets[:, trait]
        linear = scores.held_out_linear_predictor[:, :, trait]
        genetic = scores.held_out_genetic[:, :, trait]
        # Frozen probabilities are produced by this outer model's reserved development families.
        # Other folds' out-of-fold models trained on this fold's labels: calibrating from
        # those models would indirectly reuse evaluation labels and is not outer validation.
        frozen = np.full(linear.shape, np.nan) if scores.held_out_calibrated_logits is None else scores.held_out_calibrated_logits[:, :, trait]
        # A disease's genetic score is also read as a quantitative trait's is (pgs_partial_r2: its squared correlation with
        # the outcome after both are residualized on the trait's covariates), the measure the LDpred2 baseline reports
        # (benchmarks/bench_real/aou_ldpred2.py), so every trait compares with it on one scale.
        metric_names = ("auc", "pgs_auc", "pgs_partial_r2", "log_loss", "brier", "calibration_intercept", "calibration_slope") if binary else ("r2", "pgs_partial_r2")
        comparison_names = ("delta_auc", "delta_pgs_auc", "log_likelihood_gain", "brier_reduction") if binary else ("delta_pgs_partial_r2",)
        if binary and not run.config.calibration_fraction:
            metric_names = ("auc", "pgs_auc", "pgs_partial_r2")
            comparison_names = ("delta_auc", "delta_pgs_auc")
        cells = _cells(cohort, trait, fold_count)
        hidden = _partition_hidden(cells, targets, binary)
        for cell, level, held in cells:
            count = held.count
            cases = int(sum(int(targets[rows].sum()) for rows in held.folds)) if binary else None
            counts = (reportable_count(count), _cases_text(count, cases))
            shown = _shown(count, cases)
            if (cell, level) in hidden:
                # Hidden with its family's other small levels, so that no count of 1 to 20 follows from "all" less them.
                counts, shown = _hidden_counts(count, binary, hidden[(cell, level)]), False
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
                    accuracy.append((name, arm, cell, level, *metric_counts, metric, f"{_fold_weighted(values):.6g}" if values else ""))
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
                    comparison.append((name, cell, level, *metric_counts, metric, "", "", "", "", ""))
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
    baselines, refused_baselines = _baseline_scores(run, cohort)
    held_out = []
    for trait, name in enumerate(cohort.trait_names):
        if name in quarantined:
            continue
        targets = np.where(_held_out_rows(cohort, trait), cohort.targets[:, trait], np.nan)
        covariates = cohort.covariates[:, cohort.covariate_columns[trait]]
        evaluated_folds = cohort.evaluated_folds(trait, fold_count)
        methods = {arm: scores.held_out_genetic[arm_index, :, trait] for arm_index, arm in enumerate(ARMS)}
        methods |= {baseline: values[:, trait] for baseline, values in baselines.items() if np.isfinite(values[:, trait]).any()}
        structural = None if scores.held_out_structural is None else scores.held_out_structural[:, trait]
        methods |= stacked_scores(targets, covariates, methods, structural, cohort.families, cohort.folds, evaluated_folds, run.config.seed)
        held_out.append(HeldOutTrait(name, cohort.trait_types[trait] == TraitType.BINARY, targets, covariates, methods, evaluated_folds))
    held_out_tables, incomplete = partial_r2_tables(
        held_out, cohort.families, cohort.folds, cohort.ancestry, run.config.seed, BOOTSTRAP_RESAMPLES, run.budget.host_bytes // 2
    )
    tables = [
        ("accuracy.tsv", accuracy), ("comparison.tsv", comparison),
        ("quarantined.tsv", [QUARANTINE_COLUMNS, *((name, "; ".join(arms)) for name, arms in quarantined.items())]),
        ("uncertified.tsv", [UNCERTIFIED_COLUMNS, *((name, "; ".join(arms)) for name, arms in uncertified.items())]),
        *held_out_tables.items(),
    ]
    panel_summary: dict[str, Any] | str = "no pre-registered panel in the run config"
    if run.config.panel is not None and run.config.test_fold is not None:
        panel_summary = "a validation split (test_fold) tests no pre-registered panel: its primary fold is reserved"
    elif run.config.panel is not None:
        panel_lines, panel_summary = _panel_rows(run, cohort, scores, quarantined)
        tables.append(("panel.tsv", panel_lines))
    for file_name, lines in tables:
        with (directory / file_name).open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle, delimiter="\t").writerows(lines)
    exportable, refused = _checked_exportable(directory, [file_name for file_name, _lines in tables])
    _write_json(directory / EXPORTABLE_FILE, exportable)
    return {
        "files": exportable,
        "not_exportable": refused,
        "quarantined": sorted(quarantined),
        "uncertified": sorted(uncertified),
        "panel": panel_summary,
        "baselines": sorted(baselines),
        "baselines_refused": refused_baselines,
        "baselines_incomplete": incomplete,
        "comparison_inference": (
            "Standard errors are conditional on fitted models. Multi-fold comparisons are descriptive: "
            "shared-training covariance is not estimated, so combined z and p-values are omitted."
        ),
        "partial_r2_inference": (
            f"{BOOTSTRAP_RESAMPLES} family-clustered bootstrap resamples of the held-out people within each fold, every method "
            "and trait on the same resamples; conditional on the fitted models and on each stacked score's weights, which are "
            f"cross-fitted over {STACKING_SPLITS} parts of each fold's held-out families and never fitted on the person scored."
        ),
    }


@dataclass(frozen=True, slots=True)
class ExportRule:
    """The columns the export check reads in one table (``check_exportable``): ``sizes``, the counts whose suppression
    hides the row's ``values``; and ``partitions``, each a shown count with the counts of its parts, shown or suppressed
    on their own (a part's suppression hides no value), whose unshown rest must not be a count of 1 to 20."""

    sizes: tuple[str, ...] = ()
    values: tuple[str, ...] = ()
    partitions: tuple[tuple[str, tuple[str, ...]], ...] = ()


def export_rules() -> dict[str, ExportRule]:
    """Every table a step may list as exportable, by file name, with its rule: participant counts (n, and a disease's
    cases) in the report's tables; record counts in the effects summaries, held to the same rule though no participant is
    counted there, so no summary describes 1 to 20 records. A table no rule names is never exportable."""
    participants = PARTICIPANT_COUNT_COLUMNS
    return {
        "accuracy.tsv": ExportRule(participants, ("value",)),
        "comparison.tsv": ExportRule(participants, ("estimate", "standard_error", "z", "p_value")),
        "quarantined.tsv": ExportRule(),
        "uncertified.tsv": ExportRule(),
        "panel.tsv": ExportRule(participants, ("estimate", "standard_error", "z", "p_one_sided", "p_hommel")),
        "partial_r2.tsv": ExportRule(participants, PARTIAL_R2_COLUMNS[PARTIAL_R2_COLUMNS.index("partial_r2") :]),
        "paired_differences.tsv": ExportRule(participants, PAIRED_DIFFERENCE_COLUMNS[PAIRED_DIFFERENCE_COLUMNS.index("estimate") :]),
        "meta_analysis.tsv": ExportRule(),
        "trait_covariance.tsv": ExportRule(),
        "heterogeneity.tsv": ExportRule(),
        "effects_by_class.tsv": ExportRule(
            ("variants",), EFFECT_SUMMARY_COLUMNS[EFFECT_SUMMARY_COLUMNS.index("expected_positive") :], (("variants", ("positive_mean", "negative_mean")),)
        ),
        "effect_length_correlation.tsv": ExportRule(("variants",), ("spearman_rho",)),
        "genetic_variance.tsv": ExportRule(("n",), ("value",)),
        "prior_by_length.tsv": ExportRule(("variants",), PRIOR_BY_LENGTH_COLUMNS[PRIOR_BY_LENGTH_COLUMNS.index("prior_variance") :]),
        "prior_enrichment.tsv": ExportRule(("variants",), PRIOR_ENRICHMENT_COLUMNS[PRIOR_ENRICHMENT_COLUMNS.index("prior_variance") :]),
        # Its record counts are not participants', and every record in a value has 21 or more expected carriers.
        "locus_variance.tsv": ExportRule(),
        # benchmarks/bench_real/aou_compare.py's comparison of the arms and LDpred2 on the same held-out people.
        "compare.tsv": ExportRule(participants, ("estimate", "standard_error", "ci_lower", "ci_upper")),
    }


def check_exportable(path: Path) -> None:
    """Refuse a table no export rule names (``export_rules``), or one whose rule's counts show a count of 1 to 20, a
    value beside a suppressed count or beside a count of 0, a partition whose shown parts leave a rest of 1 to 20, or,
    in a table of cells, an all cell whose count (rows, a disease's cases or its controls) less one grouping's shown
    levels' is 1 to 20. Rows are compared within the rows that agree on every column but the cell, the counts and the
    values (one trait's one arm and metric, say), as a reader would set them side by side."""
    rule = export_rules().get(Path(path).name)
    if rule is None:
        raise ValueError(f"{path}: no export rule names this table, so nothing says which of its columns count.")
    rows = _read_table(path)
    parts = [part for _total, members in rule.partitions for part in members]

    def leaves(rest: int) -> bool:
        return 0 < rest < MINIMUM_REPORTED_PARTICIPANTS

    for row in rows:
        for column in (*rule.sizes, *parts):
            text = row.get(column) or ""
            if text.isdigit() and leaves(int(text)):
                raise ValueError(f"{path}: a count below {MINIMUM_REPORTED_PARTICIPANTS} is shown.")
        shown_value = any(row.get(column) for column in rule.values)
        for column in rule.sizes:
            text = row.get(column) or ""
            if shown_value and text and (not text.isdigit() or int(text) == 0):
                raise ValueError(f"{path}: a suppressed cell shows a value.")
        for total, members in rule.partitions:
            if (row.get(total) or "").isdigit():
                shown = sum(int(row[member]) for member in members if (row.get(member) or "").isdigit())
                if leaves(int(row[total]) - shown):
                    raise ValueError(f"{path}: the parts of {total} it shows leave a count below {MINIMUM_REPORTED_PARTICIPANTS}.")
    if not rows or not {"cell", "level", "n"} <= set(rows[0]):
        return
    key_columns = [column for column in rows[0] if column not in {"cell", "level", *rule.sizes, *parts, *rule.values}]
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[column] for column in key_columns), []).append(row)

    def counted(row: dict[str, str]) -> tuple[int | None, int | None, int | None]:
        # (rows, cases, controls) as the row shows them; None where it does not.
        count = int(row["n"]) if row["n"].isdigit() else None
        cases = int(row["cases"]) if (row.get("cases") or "").isdigit() else None
        return count, cases, None if count is None or cases is None else count - cases

    for members in groups.values():
        for whole in (row for row in members if row["cell"] == "all"):
            for grouping in {row["cell"] for row in members} - {"all"}:
                levels = [counted(row) for row in members if row["cell"] == grouping]
                for index, total in enumerate(counted(whole)):
                    if total is not None and leaves(total - sum(level[index] or 0 for level in levels)):
                        raise ValueError(f"{path}: the all cell less the {grouping} levels it shows leaves a count below {MINIMUM_REPORTED_PARTICIPANTS}.")


def _checked_exportable(directory: Path, names: Sequence[str]) -> tuple[list[str], dict[str, str]]:
    """The tables of ``names`` in ``directory`` that the export check passes (``check_exportable``), and why it refuses
    each of the others: a refused table stays in the step's directory and is never listed for export."""
    passed: list[str] = []
    refused: dict[str, str] = {}
    for name in names:
        try:
            check_exportable(directory / name)
        except ValueError as error:
            LOGGER.warning("workspace pipeline: %s is not exportable: %s", name, error)
            refused[name] = str(error)
        else:
            passed.append(name)
    return passed, refused


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
    _Step(
        "cohort", ("samples", "phenotypes"),
        lambda config: config.calibration_fraction if config.cross_fit else (config.calibration_fraction, config.model_folds()),
        (cohort_module, _cohort_step, development_calibration, trait_table), _cohort_step,
    ),
    _Step(
        "measurement",
        ("samples", "cohort", "store"),
        lambda config: None,
        # The measurement model it fits and loads (workspace_bindings) is measurement_model's, which the key covers.
        (store_converter, dosage_store, _measurement_step, _group_code_moments, _group_calibration, rewrite_recalibrated_store, *module_closure(measurement_model)),
        _measurement_step,
    ),
    _Step(
        "direct_sv",
        ("samples", "cohort", "store", "measurement"),
        _direct_sv_inputs,
        # The direct rows' measurement model is measurement_model's (workspace_bindings), which the key covers.
        (
            _direct_sv_step, _direct_chromosome_task, _direct_chromosome, call_set_columns, _imputed_site_list, imputed_sv_sites, _site_rows, _store_rows, _StackedRows,
            _merged_code_blocks, merged_measurement_model, read_plain_sites, read_strata_sites, read_popped_sites, store_core_spans,
            read_gene_annotation, interpolated_genetic_positions, _clear_chromosome,
            *module_closure(gatksv_store_rows, store_converter, measurement_model),
        ),
        _direct_sv_step,
    ),
    # The fit runs fit_model and everything it imports (the engine, the store reader, the scorer the artifact holds).
    # A validation split (test_fold) names its fold in the fit's key; the final split's key is as it was.
    _Step("fit", ("cohort", "measurement", "direct_sv"), lambda config: (
        config.seed, config.disease_likelihood, config.annotated_prior, config.fit_minimum_minor_allele_frequency, config.fit_minimum_r2,
        None if config.fit_variant_sites is None else _file_digest(Path(config.fit_variant_sites)), config.fit_region,
        config.length_by_class, config.ld_score_prior,
        *(() if config.test_fold is None else (("test_fold", config.test_fold),)),
    ), (_fit_step, fit_filtered, listed_sites, region_records, _final_store, _final_measurement_model, structural_records, arm_log_reliability, fit_request, model_subset, *module_closure(fit_model)), _fit_step),
    _Step("score", ("cohort", "fit"), lambda config: None, (_score_step, _split_cohort, _fit_verdicts, _checked_trait_model, _prediction_budget, _array_bytes, _final_store, score_batches, structural_records, model_part, importlib.import_module("sv_pgs.postfit_checkpoint")), _score_step),
    _Step(
        "effects", ("cohort", "fit", "score"), lambda config: {
            "jackknife_blocks": JACKKNIFE_BLOCKS, "confidence": CONFIDENCE_LEVEL,
            "locus_panel": None if config.locus_panel is None else _file_digest(Path(config.locus_panel)),
        },
        (
            _effects_step, _read_scores, _split_cohort, _checked_trait_model, _final_store, variant_effects, length_bins, genetic_variance_decomposition,
            _shown, PriorMoments, prior_moments, jackknife_ratio, effect_summaries, suppressed_levels, _hidden_counts, _hidden_records, _number,
            read_locus_panel, locus_variance, _checked_exportable, check_exportable, ExportRule, export_rules,
            importlib.import_module("sv_pgs.postfit_checkpoint"),
        ),
        _effects_step,
    ),
    _Step(
        "report",
        ("cohort", "score"),
        _report_inputs,
        (
            _report_step, _read_scores, _split_cohort, _Cohort.evaluated_folds, _proper_folds, _evaluated_counts, _residualized, held_out_cells, _held_out_rows,
            _cells, _partition_hidden, _scored_folds, _held_out_metric, _paired, _fold_combined, _fold_weighted, _partial_r2, _shown, _cases_text, _primary_influence,
            _panel_rows, _hidden_counts, _number, HeldOutTrait, method_pairs, stacked_scores,
            partial_r2_tables, _training_digest,
            _held_out_models, _baseline_scores, _checked_exportable, check_exportable, ExportRule, export_rules,
            held_out_comparison, probability_calibration,
        ),
        _report_step,
    ),
    _Step("export", ("report", "effects"), lambda config: config.export, (_export_step, check_exportable, ExportRule, export_rules), _export_step),
)


def run_pipeline(
    config: WorkspaceConfig, bindings: WorkspaceBindings, budget: ComputeBudget, *, through: str | None = None, store_shard: int | None = None,
    fit_arms: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run every step in order, through ``through`` when given, resuming from the run directory's checkpoints.

    With ``store_shard`` the run is one of a fan-out over the config's chromosomes: its store step converts only the
    chromosome at that index, and the run stops there unless the store is then whole. The store step's key does not
    name the shard, so every shard's converted chromosomes belong to one store. With ``fit_arms`` the fit step fits only
    those of ARMS, and the run stops there unless every arm is then on disk; the fit step's key does not name them either.
    On the All of Us chr22 store an arm took half an hour on two A100s, the second waiting on the first.

    With ``fit_regions`` the steps before the fit are made once, in the run directory, and each region's fit and the
    steps after it in the region's own, in the config's order (``_region_steps``); a region's summaries are named
    REGIONS_DIRECTORY/<name>/<step>."""
    if through is not None and through not in STEP_NAMES:
        raise ValueError(f"through must be one of {STEP_NAMES}.")
    if store_shard is not None and not 0 <= store_shard < len(config.chromosomes):
        raise ValueError(f"store_shard indexes the config's {len(config.chromosomes)} chromosomes; got {store_shard}.")
    config.run_directory.mkdir(parents=True, exist_ok=True)
    if fit_arms is not None and (not fit_arms or not set(fit_arms) <= set(ARMS)):
        raise ValueError(f"fit_arms names some of {ARMS}; got {fit_arms}.")
    run = _Run(config, bindings, budget, {}, None if store_shard is None else (config.chromosomes[store_shard],), None if fit_arms is None else tuple(fit_arms))
    summaries = {}
    for step in _STEPS:
        if step.name == "fit" and config.fit_regions is not None:
            for name, region in config.region_runs():
                summaries.update({f"{REGIONS_DIRECTORY}/{name}/{done}": summary for done, summary in _region_steps(run, region, through).items()})
            break
        try:
            summaries[step.name] = _run_step(run, step)
        except StepPart as part:
            LOGGER.info("workspace pipeline: %s stopped after its part: %s", step.name, part)
            break
        if step.name == through:
            break
    return summaries


def _region_steps(shared: _Run, config: WorkspaceConfig, through: str | None) -> dict[str, dict[str, Any]]:
    """One region's fit and the steps after it (``WorkspaceConfig.region_runs``), on the shared run's steps before the
    fit. Each of those is linked into the region's run directory, where the later steps read it as their own
    (``_final_store`` names the store by its path in the run directory), and keeps the shared run's key, so no region
    makes it again. A region's fit stopped after its part (``fit_arms``) ends that region's steps alone."""
    config.run_directory.mkdir(parents=True, exist_ok=True)
    later = [step for step in _STEPS if STEP_NAMES.index(step.name) >= STEP_NAMES.index("fit")]
    for name in STEP_NAMES[:STEP_NAMES.index("fit")]:
        link = config.run_directory / name
        if link.exists() and not link.is_symlink():
            raise RuntimeError(f"{link} is not the shared run's {name}; move it away first.")
        if not link.is_symlink():
            link.symlink_to(os.path.relpath(shared.output(name), config.run_directory), target_is_directory=True)
    run = _Run(config, shared.bindings, shared.budget, dict(shared.keys), None, shared.fit_arms)
    summaries = {}
    for step in later:
        try:
            summaries[step.name] = _run_step(run, step)
        except StepPart as part:
            LOGGER.info("workspace pipeline: region %s: %s stopped after its part: %s", config.run_directory.name, step.name, part)
            break
        if step.name == through:
            break
    return summaries


def completed_through(config: WorkspaceConfig, through: str) -> bool:
    """Whether the run holds ``through`` completed (launcher/workspace runs its AFTER_SCRIPT only then): in every
    region's run directory when the run fits regions and ``through`` is the fit or a later step, else in its own."""
    if through not in STEP_NAMES:
        raise ValueError(f"through must be one of {STEP_NAMES}.")
    regional = config.fit_regions is not None and STEP_NAMES.index(through) >= STEP_NAMES.index("fit")
    return all((run.run_directory / through / COMPLETE_FILE).is_file() for run in ([region for _name, region in config.region_runs()] if regional else [config]))
