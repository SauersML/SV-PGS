"""The All of Us results: the study's summary tables and figures, from the aggregate tables SV-PGS runs export.

Inputs. Each RUN_DIR holds one run's exported tables (``<run>/export``, which the export step stages after
``workspace_pipeline.check_exportable``) and, where an LDpred2 job ran beside the run, its compare.tsv
(benchmarks/bench_real/aou_compare.py) and ldpred2.jsonl (aou_ldpred2.py ``--out``):
- the report's partial_r2.tsv, paired_differences.tsv, meta_analysis.tsv, trait_covariance.tsv and heterogeneity.tsv:
  each method's held-out partial r^2 with its family-clustered bootstrap interval, overall and per ancestry group, for
  both arms, LDpred2-auto on the same people and LDpred2-auto stacked with SV-PGS (where the run's baseline directory
  held LDpred2's scores), and every paired difference;
- accuracy.tsv and comparison.tsv: the arms' AUCs and a disease's paired AUC differences; for a run exported before the
  tables above, each arm's partial r^2 (a point, with no interval) and the SV gain with its influence-function
  interval;
- compare.tsv: LDpred2-auto's partial r^2 and its paired differences with the arms on the same held-out people
  (influence-function intervals), read where the report has no paired LDpred2 and only for a trait whose arms there
  are the report's (the same held-out people and partial r^2); ldpred2.jsonl: an unpaired LDpred2-auto point, read only
  where the report has neither and its held-out people (and a disease's cases) number the report's;
- the effects step's prior_enrichment.tsv, prior_by_length.tsv, genetic_variance.tsv and locus_variance.tsv: the learned
  prior by SV class and length, each class's share of the genetic variance, and each panel window's;
- quarantined.tsv and uncertified.tsv: a flagged trait is marked (STATUS_MARK) and left out of every meta-analysis made
  here.
``--region`` selects one region's compare.tsv and ldpred2.jsonl lines (default: the whole genome's); a region run's
report tables are in its own export directory.

Validation. Every input is read, and refused whole (naming its file, never a value in it), before anything is written:
a table must have the pipeline's columns for its name and pass check_exportable, with no disease metric shown without
its case count; an ldpred2.jsonl line is held to the same count rule. A trait is one run's: a trait in two run
directories is refused, so none is counted twice. Nothing here reads a participant, and a suppressed cell stays blank,
never filled from another table.

Outputs, in the output directory:
- ``aou_summary.tsv`` (SUMMARY_COLUMNS): one row per trait.
- ``aou_ancestry.tsv`` (ANCESTRY_COLUMNS): one row per trait and ancestry group shown.
- ``aou_meta.tsv`` (workspace_pipeline.META_ANALYSIS_COLUMNS): each paired difference combined over the unflagged traits
  whose single test fold gives it a p-value, overall and per ancestry group, fixed and random effects. Where one run
  holds every such trait, its own meta-analysis is copied; otherwise the traits are combined here
  (``held_out_comparison.meta_analysis``) with each run's traits' covariance from its own bootstrap resamples
  (trait_covariance.tsv) and the runs as independent, which their shared people make optimistic; ``covariance`` says
  which.
- ``aou_loci.tsv`` (LOCI_COLUMNS): per panel window and trait, the with_sv model's genetic variance there split between
  SNVs, short indels and structural variants.
- ``fig_aou_diseases`` and ``fig_aou_effects``: the poster's two All of Us figures (its ph_aou_diseases and
  ph_aou_hypertension slots), SVGs in figures.py's units and, where figures.py's Chrome is installed, their PNGs.
- With matplotlib (``uv run --with matplotlib python -m docs.poster.ashg2026.aou_results ...``), the review figures
  (REVIEW_FIGURES), each a vector PDF and a 600 dpi PNG in sim_gallery.py's journal style, all of them in
  ``aou_review.pdf``, and ``aou_review_notes.json``, what each figure's intervals do and do not hold.
A figure without the data it needs is not drawn, and its previous files are removed. The default output directory is the
poster's figures directory, where build_pptx.py places the PNGs in their slots; git ignores these outputs there
(.gitignore), so no All of Us result enters the repository before it is released.

    python -m docs.poster.ashg2026.aou_results RUN_DIR [RUN_DIR ...] [--ldpred2 FILE ...] [--region NAME] [--trait NAME] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from docs.poster.ashg2026 import figures
except ImportError:  # run as a script from this directory, as figures.py is
    import figures  # type: ignore[no-redef]

from benchmarks.bench_real.aou_compare import COLUMNS as COMPARE_COLUMNS
from sv_pgs import workspace_pipeline as pipeline
from sv_pgs.all_of_us import MINIMUM_REPORTED_PARTICIPANTS
from sv_pgs.held_out_comparison import meta_analysis

ARMS = pipeline.ARMS
BASELINE = "ldpred2_auto"
STACKED = f"{BASELINE}+with_sv"
STACKED_STRUCTURAL = f"{BASELINE}+structural"
PAIRS = (
    ("with_sv", "snv_indel", "sv_gain"), ("with_sv", BASELINE, "with_sv_vs_ldpred2"), ("snv_indel", BASELINE, "snv_indel_vs_ldpred2"),
    (STACKED, BASELINE, "stack_gain"), (STACKED_STRUCTURAL, BASELINE, "stack_sv_gain"),
)
"""Each (candidate, baseline) difference the summary reads, with its column prefix: the SV gain within SV-PGS, each arm
against LDpred2-auto, and what LDpred2-auto gains when SV-PGS's score (``stack_gain``) or only its structural part
(``stack_sv_gain``) is stacked with it on the same people (workspace_pipeline.stacked_scores)."""
COMPARE_PAIRS = {"with_sv - snv_indel": "sv_gain", "with_sv - ldpred2": "with_sv_vs_ldpred2", "snv_indel - ldpred2": "snv_indel_vs_ldpred2"}
"""compare.tsv's paired scores (aou_compare.PAIRS, first less second) by the PAIRS prefix each gives."""
CONFIDENCE_QUANTILE = 1.959963984540054
"""The two-sided 95% normal quantile, for an interval made here from a standard error."""
SINGLE_FOLD = "conditional_on_fitted_models"
"""The report's inference where a trait has one test fold, whose paired difference has a p-value (the report's
``descriptive_cross_fit`` has none)."""
SCHEMAS = {
    "partial_r2.tsv": pipeline.PARTIAL_R2_COLUMNS, "paired_differences.tsv": pipeline.PAIRED_DIFFERENCE_COLUMNS,
    "meta_analysis.tsv": pipeline.META_ANALYSIS_COLUMNS, "trait_covariance.tsv": pipeline.TRAIT_COVARIANCE_COLUMNS,
    "heterogeneity.tsv": pipeline.HETEROGENEITY_COLUMNS, "accuracy.tsv": pipeline.ACCURACY_COLUMNS, "comparison.tsv": pipeline.COMPARISON_COLUMNS,
    "quarantined.tsv": pipeline.QUARANTINE_COLUMNS, "uncertified.tsv": pipeline.UNCERTIFIED_COLUMNS,
    "genetic_variance.tsv": pipeline.GENETIC_VARIANCE_COLUMNS, "prior_enrichment.tsv": pipeline.PRIOR_ENRICHMENT_COLUMNS,
    "prior_by_length.tsv": pipeline.PRIOR_BY_LENGTH_COLUMNS, "locus_variance.tsv": pipeline.LOCUS_VARIANCE_COLUMNS,
    "compare.tsv": COMPARE_COLUMNS,
}
"""Every table read, by file name, with the columns its writer gives it; a file of another name is never read."""
TRAIT_TABLES = (
    "partial_r2", "accuracy", "comparison", "quarantined", "uncertified", "paired_differences", "compare", "genetic_variance", "prior_enrichment",
    "prior_by_length", "locus_variance", "heterogeneity",
)
"""The tables whose ``trait`` column names a run's own traits (heterogeneity's ``combined`` aside), in the order a run's
traits are listed."""
DISEASE_METRICS = (
    "auc", "pgs_auc", "log_loss", "brier", "calibration_intercept", "calibration_slope", "delta_auc", "delta_pgs_auc", "log_likelihood_gain",
    "brier_reduction",
)
"""The report's metrics of a disease alone, each always beside its case count (shown or suppressed)."""
LDPRED2_KEYS = ("trait", "held_out_rows", "variants", "pgs_partial_r2")
"""An ldpred2.jsonl line's keys; ``region`` and a disease's ``cases`` may join them."""
STATUS_MARK = "†"
"""The mark of a flagged (quarantined or uncertified) trait in every figure."""
SUMMARY_COLUMNS = (
    "trait", "status", "n", "cases",
    "snv_indel_partial_r2", "snv_indel_ci_low", "snv_indel_ci_high",
    "with_sv_partial_r2", "with_sv_ci_low", "with_sv_ci_high",
    "ldpred2_partial_r2", "ldpred2_ci_low", "ldpred2_ci_high", "ldpred2_source",
    "stacked_partial_r2", "stacked_ci_low", "stacked_ci_high",
    "stacked_structural_partial_r2", "stacked_structural_ci_low", "stacked_structural_ci_high", "partial_r2_interval",
    *(f"{prefix}{suffix}" for _candidate, _baseline, prefix in PAIRS for suffix in ("", "_ci_low", "_ci_high", "_p")),
    "sv_gain_relative", "snv_indel_pgs_auc", "with_sv_pgs_auc", "ldpred2_pgs_auc",
    "delta_pgs_auc", "delta_pgs_auc_ci_low", "delta_pgs_auc_ci_high", "delta_pgs_auc_p", "delta_auc", "delta_auc_ci_low", "delta_auc_ci_high",
    "delta_auc_p", "sv_prior_enrichment", "sv_posterior_enrichment", "sv_posterior_enrichment_ci_low", "sv_posterior_enrichment_ci_high",
    "sv_variance_share", "sv_variance_share_ci_low", "sv_variance_share_ci_high", "sv_variant_share", "sv_heritability_enrichment",
    "sv_heritability_enrichment_ci_low", "sv_heritability_enrichment_ci_high", "sv_score_variance_share", "ancestry_heterogeneity_p",
    "inference", "source",
)
"""``aou_summary.tsv``: per trait its flags (``status``: each flagged arm and its flag), held-out people (n, a disease's
cases), each method's partial r^2 with its 95% interval (``stacked``: LDpred2-auto stacked with SV-PGS;
``stacked_structural``: with SV-PGS's structural part; ``partial_r2_interval``: "bootstrap", or "none" for points from
accuracy.tsv), LDpred2-auto's source ("paired": the report's bootstrap on the same people; "paired_compare":
compare.tsv's influence functions on the same people; "unpaired": an ldpred2.jsonl point), each paired difference of
PAIRS (candidate less baseline on the same people) with its interval and two-sided p, the arms' and LDpred2's genetic-
score AUC and a disease's paired AUC differences (``delta_pgs_auc``: the genetic scores alone; ``delta_auc``: the whole
predictor), the with_sv model's prior and posterior enrichment of every structural record over SNVs, their share of its
genetic variance against their share of its records and the ratio (the heritability enrichment), their share of the
genetic score's variance over the observed rows (genetic_variance.tsv; their LD with the rest included), the p of the SV
gain's difference between ancestry groups, the inference the run's single or several test folds allow, and the run's
directory."""
ANCESTRY_COLUMNS = (
    "trait", "level", "n", "cases", *(f"{prefix}{suffix}" for _candidate, _baseline, prefix in PAIRS for suffix in ("", "_ci_low", "_ci_high", "_p")),
)
"""``aou_ancestry.tsv``: per trait and ancestry group shown, its held-out people and each paired difference of PAIRS."""
LOCI_COLUMNS = (
    "trait", "locus", "kind", "chromosome", "start", "end", "genome_share", "snv_share", "short_indel_share", "sv_share", "top_sv_class",
    "top_sv_class_share", "snv_indel_genome_share", "rare_left_out",
)
"""``aou_loci.tsv``: per panel window and trait (locus_variance.tsv), the with_sv model's share of its genetic variance in
the window (``genome_share``), the window's split between SNVs, short indels and structural variants (each a share of
the window's), its largest structural class and that class's share, the snv_indel model's share in the same window, and
the records left out for their few carriers (records, not people)."""
TRAIT_LABELS = {
    "hypertension": "Hypertension", "coronary_artery_disease": "CAD", "asthma": "Asthma", "hyperlipidemia": "Hyperlipidemia",
    "breast_cancer": "Breast cancer", "type2_diabetes": "T2D", "atrial_fibrillation": "AF", "hypothyroidism": "Hypothyroidism",
    "copd": "COPD", "depression": "Depression", "gout": "Gout", "cataract": "Cataract", "chronic_kidney_disease": "CKD",
    "psoriasis": "Psoriasis", "prostate_cancer": "Prostate cancer", "egfr_ckd_epi_2021": "eGFR", "ldl_cholesterol": "LDL",
    "body_mass_index": "BMI", "systolic_blood_pressure": "Systolic BP", "hemoglobin_a1c": "HbA1c", "combined": "All traits",
    "alzheimers_disease": "Alzheimer's",
}
CLASS_LABELS = {
    "deletion": "Deletion", "insertion": "Insertion", "duplication": "Duplication", "insertion_mei": "Mobile element",
    "inversion_bnd_complex": "Inversion/BND", "str_vntr_repeat": "STR/VNTR", "other_complex_sv": "Complex",
    "copy_number": "Copy number", "inversion": "Inversion", "all_structural": "All SVs", "snv": "SNV", "short_indel": "Short indel",
}
CLASS_COLOURS = {"deletion": figures.NAVY, "insertion": figures.TEAL, "duplication": figures.AMBER, "insertion_mei": figures.PURPLE}
"""The length panel's series, each class always in its own colour; the other classes are in the class panel alone."""
METHODS = (
    # (method, summary prefix, key label, colour, filled marker, horizontal offset)
    (BASELINE, "ldpred2", "LDpred2-auto", figures.AMBER, True, -18.0),
    ("snv_indel", "snv_indel", "SV-PGS without SVs", figures.NAVY, False, 0.0),
    ("with_sv", "with_sv", "SV-PGS with SVs", figures.TEAL, True, 18.0),
)
"""The poster's partial r^2 panel's methods; the stacked scores are in the summary and, as gains, in the gain panel."""
GAINS = (
    # (summary prefix, key label, colour, horizontal offset)
    ("sv_gain", "within SV-PGS", figures.TEAL, -12.0),
    ("stack_sv_gain", "on top of LDpred2-auto", figures.PURPLE, 12.0),
)
"""The poster's gain panel's two measures of what SVs add, each per trait and combined over the traits: with_sv less
snv_indel, and LDpred2-auto stacked with SV-PGS's structural part less LDpred2-auto alone."""


def number(text: object) -> float:
    """A table's value, or NaN where it is blank or suppressed."""
    try:
        return float(text)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan


def text(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}g}" if math.isfinite(value) else ""


def label(name: str) -> str:
    return TRAIT_LABELS.get(name, name.replace("_", " ").capitalize())


# ------------------------------------------------------------------ inputs


def read_table(path: Path) -> list[dict[str, str]]:
    """One exported table, refused whole unless its columns are its writer's (SCHEMAS), every row has them all, it passes
    the export check, and no disease metric is shown without its case count. A refusal names the file and the rule, never
    a value."""
    with path.open(newline="", encoding="utf-8") as handle:
        lines = list(csv.reader(handle, delimiter="\t"))
    if not lines or tuple(lines[0]) != SCHEMAS[path.name] or any(len(line) != len(lines[0]) for line in lines[1:]):
        raise ValueError(f"{path}: its columns are not those of the pipeline's {path.name}; refused.")
    try:
        pipeline.check_exportable(path)
    except ValueError as error:
        raise ValueError(f"{path}: refused by the export check ({str(error).rsplit(': ', 1)[-1].rstrip('.')}).") from None
    rows = [dict(zip(lines[0], line)) for line in lines[1:]]
    if any(row.get("metric") in DISEASE_METRICS and not row.get("cases") for row in rows):
        raise ValueError(f"{path}: a disease metric is shown without its case count; refused.")
    return rows


def counts_refused(count: str, cases: str, shown: bool) -> bool:
    """Whether counts break the export rule as check_exportable reads it: a count of 1 to 20, cases leaving 1 to 20
    controls (or more cases than rows), or a value beside a count that is suppressed or 0."""
    small = [text_count for text_count in (count, cases) if text_count.isdigit() and 0 < int(text_count) < MINIMUM_REPORTED_PARTICIPANTS]
    if small or (count.isdigit() and cases.isdigit() and (int(cases) > int(count) or 0 < int(count) - int(cases) < MINIMUM_REPORTED_PARTICIPANTS)):
        return True
    return shown and any(text_count and (not text_count.isdigit() or int(text_count) == 0) for text_count in (count, cases))


def read_ldpred2(path: Path) -> list[dict]:
    """aou_ldpred2.py's aggregate lines, refused whole as read_table refuses a table: unknown keys, or counts the export
    rule would not show beside the line's partial r^2."""
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            raise ValueError(f"{path}: not aou_ldpred2.py's aggregate lines; refused.") from None
        if not isinstance(entry, dict) or not set(LDPRED2_KEYS) <= set(entry) or set(entry) - {*LDPRED2_KEYS, "region", "cases"}:
            raise ValueError(f"{path}: a line's keys are not aou_ldpred2.py's; refused.")
        value = entry["pgs_partial_r2"]
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError(f"{path}: a line's partial r^2 is not a number; refused.")
        if counts_refused(str(entry["held_out_rows"]), str(entry.get("cases", "")), value is not None):
            raise ValueError(f"{path}: refused by the export rule (a count of 1 to 20, or a value beside a hidden count).")
        entries.append(entry)
    return entries


def flagged_arms(detail: str) -> list[str]:
    """The arms a quarantined or uncertified entry names (every arm where it names none)."""
    return [arm for arm in ARMS if re.search(rf"\b{arm}\b", detail)] or list(ARMS)


@dataclass
class Exports:
    """Every run's tables (each row tagged with its run's directory, ``source``), the traits in run order, each trait's
    flagged arms, and the LDpred2 lines of the selected region by trait."""

    tables: dict[str, list[dict[str, str]]]
    traits: list[str]
    flags: dict[str, dict[str, str]]
    ldpred2: dict[str, dict]

    def status(self, trait: str) -> str:
        return "; ".join(f"{arm} {flag}" for arm, flag in sorted(self.flags.get(trait, {}).items()))


def load(directories: list[Path], ldpred2_paths: list[Path] = (), region: str = "") -> Exports:  # type: ignore[assignment]
    """Every run directory's tables and LDpred2 lines, each validated (read_table, read_ldpred2) before any is used; a
    trait in two run directories, or in two different LDpred2 lines of the region, is refused."""
    tables: dict[str, list[dict[str, str]]] = {name.removesuffix(".tsv"): [] for name in SCHEMAS}
    owner: dict[str, str] = {}
    order: list[str] = []
    lines: list[dict] = []
    for directory in directories:
        if not directory.is_dir():
            raise ValueError(f"{directory} is not a run's export directory.")
        found = {name.removesuffix(".tsv"): read_table(directory / name) for name in SCHEMAS if (directory / name).is_file()}
        own = list(dict.fromkeys(row["trait"] for name in TRAIT_TABLES for row in found.get(name, []) if row["trait"] != "combined"))
        for trait in own:
            if trait in owner:
                raise ValueError(f"{trait} is in both {owner[trait]} and {directory}; give each trait's run once.")
            owner[trait] = str(directory)
        order += own
        for name, rows in found.items():
            tables[name] += [{**row, "source": str(directory)} for row in rows]
        if (directory / "ldpred2.jsonl").is_file():
            lines += read_ldpred2(directory / "ldpred2.jsonl")
    for path in ldpred2_paths:
        lines += read_ldpred2(path)
    ldpred2: dict[str, dict] = {}
    for entry in lines:
        if entry.get("region", "") != region:
            continue
        if ldpred2.get(entry["trait"], entry) != entry:
            raise ValueError(f"two different LDpred2 lines give {entry['trait']}; give each trait's line once.")
        ldpred2[entry["trait"]] = entry
    tables["compare"] = [row for row in tables["compare"] if row["region"] == region]
    flags: dict[str, dict[str, str]] = {}
    for kind in ("quarantined", "uncertified"):
        for row in tables[kind]:
            for arm in flagged_arms(row[f"{kind}_fits"]):
                flags.setdefault(row["trait"], {}).setdefault(arm, kind)
    return Exports(tables, order, flags, ldpred2)


# ------------------------------------------------------------------ tables


def interval_entry(estimate: str, standard_error: str, p_value: str, source: str, ci_low: str = "", ci_high: str = "") -> dict[str, str]:
    """A paired difference as the summary holds it, its interval the given one or the estimate plus and minus
    CONFIDENCE_QUANTILE standard errors."""
    value, error = number(estimate), number(standard_error)
    if not ci_low and math.isfinite(value) and math.isfinite(error):
        ci_low, ci_high = text(value - CONFIDENCE_QUANTILE * error), text(value + CONFIDENCE_QUANTILE * error)
    return {"estimate": estimate, "standard_error": standard_error, "ci_low": ci_low, "ci_high": ci_high, "p_value": p_value, "source": source}


def single_fold(exports: Exports) -> dict[str, bool]:
    """Whether each trait's report reads one test fold, whose differences have p-values (SINGLE_FOLD)."""
    single: dict[str, bool] = {}
    for name in ("partial_r2", "comparison"):
        for row in exports.tables[name]:
            if row["cell"] == "all":
                single[row["trait"]] = single.get(row["trait"], False) or row["inference"] == SINGLE_FOLD
    return single


def arm_points(exports: Exports) -> dict[tuple[str, str], dict[str, str]]:
    """(trait, method) -> the all-rows cell's partial r^2 with its interval, n and cases: partial_r2.tsv's, else (a run
    exported before it) accuracy.tsv's point with no interval."""
    points = {(row["trait"], row["method"]): {**row, "interval": "bootstrap"} for row in exports.tables["partial_r2"] if row["cell"] == "all"}
    for row in exports.tables["accuracy"]:
        if row["cell"] == "all" and row["metric"] == "pgs_partial_r2" and row["arm"] in ARMS:
            points.setdefault((row["trait"], row["arm"]), {
                "partial_r2": row["value"], "ci_low": "", "ci_high": "", "n": row["n"], "cases": row["cases"], "inference": "", "source": row["source"],
                "interval": "none",
            })
    return points


def compare_usable(exports: Exports, points: dict[tuple[str, str], dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Each trait's compare.tsv rows, for the traits whose report has no paired LDpred2 and whose compare.tsv arms are on
    the report's held-out people (n and a disease's cases) with the report's partial r^2 (to its six digits); a trait
    whose compare.tsv arms are not is named on stderr and not read."""
    rows: dict[str, list[dict[str, str]]] = {}
    for row in exports.tables["compare"]:
        rows.setdefault(row["trait"], []).append(row)
    usable = {}
    for trait, entries in rows.items():
        if (trait, BASELINE) in points:
            continue
        arms = {row["score"]: row for row in entries if row["metric"] == "pgs_partial_r2" and row["score"] in ARMS}
        agree = len(arms) == len(ARMS) and all(
            (trait, arm) in points and row["n"] == points[trait, arm]["n"] and row["cases"] == points[trait, arm]["cases"]
            and math.isclose(number(row["estimate"]), number(points[trait, arm]["partial_r2"]), rel_tol=1e-4, abs_tol=1e-6)
            for arm, row in arms.items()
        )
        if agree:
            usable[trait] = entries
        else:
            print(f"aou_results: compare.tsv's arms of {trait} are not the report's held-out people or values; its LDpred2 rows are not read.", file=sys.stderr)
    return usable


def pair_entries(exports: Exports, compare: dict[str, list[dict[str, str]]]) -> dict[tuple[str, str, str, str], dict[str, str]]:
    """(trait, PAIRS prefix, cell, level) -> each paired difference: paired_differences.tsv's (bootstrap), else
    comparison.tsv's SV gain and compare.tsv's LDpred2 differences (influence functions; a p-value only where the trait's
    report reads one test fold)."""
    prefixes = {(candidate, baseline): prefix for candidate, baseline, prefix in PAIRS}
    single = single_fold(exports)
    entries: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in exports.tables["paired_differences"]:
        prefix = prefixes.get((row["candidate"], row["baseline"]))
        if prefix is not None:
            entries[row["trait"], prefix, row["cell"], row["level"]] = {
                **interval_entry(row["estimate"], row["standard_error"], row["p_value"], row["source"], row["ci_low"], row["ci_high"]),
                "relative": row["relative_estimate"], "origin": "bootstrap",
            }
    for row in exports.tables["comparison"]:
        if row["metric"] == "delta_pgs_partial_r2":
            entries.setdefault((row["trait"], "sv_gain", row["cell"], row["level"]), {
                **interval_entry(row["estimate"], row["standard_error"], row["p_value"], row["source"]), "origin": "influence",
            })
    for trait, rows in compare.items():
        for row in rows:
            if row["metric"] == "delta_pgs_partial_r2" and row["score"] in COMPARE_PAIRS:
                z = number(row["estimate"]) / number(row["standard_error"]) if number(row["standard_error"]) > 0.0 else math.nan
                p_value = text(math.erfc(abs(z) / math.sqrt(2.0))) if single.get(trait) else ""
                entries.setdefault((trait, COMPARE_PAIRS[row["score"]], "all", "all"), {
                    **interval_entry(row["estimate"], row["standard_error"], p_value, row["source"], row["ci_lower"], row["ci_upper"]), "origin": "influence",
                })
    return entries


def summary_rows(exports: Exports) -> list[dict[str, str]]:
    """One row per trait (SUMMARY_COLUMNS), from the all-held-out-rows cells, in run order. LDpred2-auto comes from the
    report's paired scores, else compare.tsv's (compare_usable), else an ldpred2.jsonl point where its held-out people
    (and a disease's cases) are the trait's as the run shows them."""
    tables = exports.tables
    points = arm_points(exports)
    compare = compare_usable(exports, points)
    paired = pair_entries(exports, compare)
    accuracy = {(row["trait"], row["arm"], row["metric"]): row["value"] for row in tables["accuracy"] if row["cell"] == "all"}
    comparison = {(row["trait"], row["metric"]): row for row in tables["comparison"] if row["cell"] == "all"}
    enrichment = {row["trait"]: row for row in tables["prior_enrichment"] if row["arm"] == "with_sv" and row["variant_class"] == "all_structural"}
    shares = {row["trait"]: row["value"] for row in tables["genetic_variance"] if row["quantity"] == "structural_share"}
    ancestry = {row["trait"]: row for row in tables["heterogeneity"] if (row["candidate"], row["baseline"], row["cell"]) == ("with_sv", "snv_indel", "ancestry")}
    rows = []
    for trait in exports.traits:
        reference = points.get((trait, "with_sv")) or points.get((trait, "snv_indel")) or {}
        row = {
            "trait": trait, "status": exports.status(trait), "n": reference.get("n", ""), "cases": reference.get("cases", ""),
            "inference": reference.get("inference") or comparison.get((trait, "delta_pgs_partial_r2"), {}).get("inference", ""),
            "source": reference.get("source", ""), "partial_r2_interval": reference.get("interval", ""),
        }
        for method, prefix in (("snv_indel", "snv_indel"), ("with_sv", "with_sv"), (BASELINE, "ldpred2"), (STACKED, "stacked"), (STACKED_STRUCTURAL, "stacked_structural")):
            entry = points.get((trait, method), {})
            row |= {f"{prefix}_partial_r2": entry.get("partial_r2", ""), f"{prefix}_ci_low": entry.get("ci_low", ""), f"{prefix}_ci_high": entry.get("ci_high", "")}
        row["ldpred2_source"] = "paired" if (trait, BASELINE) in points else ""
        compared = {(entry["score"], entry["metric"]): entry for entry in compare.get(trait, [])}
        if not row["ldpred2_source"] and ("ldpred2", "pgs_partial_r2") in compared:
            entry = compared["ldpred2", "pgs_partial_r2"]
            row |= {"ldpred2_partial_r2": entry["estimate"], "ldpred2_ci_low": entry["ci_lower"], "ldpred2_ci_high": entry["ci_upper"], "ldpred2_source": "paired_compare"}
        line = exports.ldpred2.get(trait, {})
        same_people = str(line.get("held_out_rows", "")) == row["n"] and str(line.get("cases", "")) == row["cases"]
        if not row["ldpred2_source"] and line.get("pgs_partial_r2") is not None and same_people:
            row["ldpred2_partial_r2"], row["ldpred2_source"] = text(float(line["pgs_partial_r2"])), "unpaired"
        for _candidate, _baseline, prefix in PAIRS:
            entry = paired.get((trait, prefix, "all", "all"), {})
            row |= {prefix: entry.get("estimate", ""), f"{prefix}_ci_low": entry.get("ci_low", ""), f"{prefix}_ci_high": entry.get("ci_high", ""),
                    f"{prefix}_p": entry.get("p_value", "")}
        row["sv_gain_relative"] = paired.get((trait, "sv_gain", "all", "all"), {}).get("relative", "")
        row |= {f"{arm}_pgs_auc": accuracy.get((trait, arm, "pgs_auc"), "") for arm in ARMS}
        row["ldpred2_pgs_auc"] = compared.get(("ldpred2", "pgs_auc"), {}).get("estimate", "")
        for metric in ("delta_pgs_auc", "delta_auc"):
            entry = comparison.get((trait, metric))
            found = interval_entry(entry["estimate"], entry["standard_error"], entry["p_value"], entry["source"]) if entry else {}
            row |= {metric: found.get("estimate", ""), f"{metric}_ci_low": found.get("ci_low", ""), f"{metric}_ci_high": found.get("ci_high", ""),
                    f"{metric}_p": found.get("p_value", "")}
        prior = enrichment.get(trait, {})
        row |= {
            "sv_prior_enrichment": prior.get("prior_enrichment", ""), "sv_posterior_enrichment": prior.get("posterior_enrichment", ""),
            "sv_posterior_enrichment_ci_low": prior.get("posterior_enrichment_ci_low", ""),
            "sv_posterior_enrichment_ci_high": prior.get("posterior_enrichment_ci_high", ""),
            "sv_variance_share": prior.get("posterior_share", ""), "sv_variance_share_ci_low": prior.get("posterior_share_ci_low", ""),
            "sv_variance_share_ci_high": prior.get("posterior_share_ci_high", ""), "sv_variant_share": prior.get("variant_share", ""),
            "sv_heritability_enrichment": prior.get("heritability_enrichment", ""),
            "sv_heritability_enrichment_ci_low": prior.get("heritability_enrichment_ci_low", ""),
            "sv_heritability_enrichment_ci_high": prior.get("heritability_enrichment_ci_high", ""),
            "sv_score_variance_share": shares.get(trait, ""), "ancestry_heterogeneity_p": ancestry.get(trait, {}).get("p_value", ""),
        }
        rows.append(row)
    return rows


def ancestry_rows(exports: Exports) -> list[dict[str, str]]:
    """One row per trait and ancestry group shown (ANCESTRY_COLUMNS): partial_r2.tsv's groups, else (a run exported
    before it) comparison.tsv's."""
    paired = pair_entries(exports, {})
    groups = {(row["trait"], row["level"]): row for row in exports.tables["comparison"] if row["cell"] == "ancestry" and row["metric"] == "delta_pgs_partial_r2"}
    groups |= {(row["trait"], row["level"]): row for row in exports.tables["partial_r2"] if row["cell"] == "ancestry" and row["method"] == "with_sv"}
    rows = []
    for (trait, level), entry in groups.items():
        row = {"trait": trait, "level": level, "n": entry["n"], "cases": entry["cases"]}
        for _candidate, _baseline, prefix in PAIRS:
            found = paired.get((trait, prefix, "ancestry", level), {})
            row |= {prefix: found.get("estimate", ""), f"{prefix}_ci_low": found.get("ci_low", ""), f"{prefix}_ci_high": found.get("ci_high", ""),
                    f"{prefix}_p": found.get("p_value", "")}
        rows.append(row)
    return rows


def meta_rows(exports: Exports) -> list[dict[str, str]]:
    """Each pair's combined difference per cell (workspace_pipeline.META_ANALYSIS_COLUMNS) over the unflagged traits
    whose differences have a p-value: a run's own meta-analysis where it holds exactly those traits; else the traits
    combined here, each run's traits with their joint-bootstrap covariance (trait_covariance.tsv) and the runs taken as
    independent of each other (their shared people make that optimistic); a run without its covariance table, or a
    difference from compare.tsv or comparison.tsv, counts as independent too, and ``covariance`` says which."""
    paired = pair_entries(exports, compare_usable(exports, arm_points(exports)))
    pairs = {prefix: (candidate, baseline) for candidate, baseline, prefix in PAIRS}
    tested: dict[tuple[str, str, str], list[tuple[str, dict[str, str]]]] = {}
    for (trait, prefix, cell, level), entry in paired.items():
        if entry["p_value"] and number(entry["standard_error"]) > 0.0 and trait not in exports.flags:
            tested.setdefault((prefix, cell, level), []).append((trait, entry))
    covariances = {
        (row["source"], row["candidate"], row["baseline"], row["cell"], row["level"], *sorted((row["trait"], row["other_trait"]))): number(row["covariance"])
        for row in exports.tables["trait_covariance"]
    }
    rows = []
    for (prefix, cell, level), entries in tested.items():
        candidate, baseline = pairs[prefix]
        traits = sorted(trait for trait, _entry in entries)
        if len(traits) < 2:
            continue
        own = [row for row in exports.tables["meta_analysis"] if (row["candidate"], row["baseline"], row["cell"], row["level"]) == (candidate, baseline, cell, level)
               and sorted(row["traits"].split(";")) == traits and all(entry["origin"] == "bootstrap" for _trait, entry in entries)]
        if own:
            rows += [{column: row[column] for column in pipeline.META_ANALYSIS_COLUMNS} for row in own]
            continue
        estimates = np.array([number(entry["estimate"]) for _trait, entry in entries])
        covariance = np.diag([number(entry["standard_error"]) ** 2 for _trait, entry in entries])
        within = True
        for first, (one, one_entry) in enumerate(entries):
            for second, (other, other_entry) in enumerate(entries):
                if first != second and one_entry["source"] == other_entry["source"]:
                    value = covariances.get((one_entry["source"], candidate, baseline, cell, level, *sorted((one, other))))
                    value = value if one_entry["origin"] == other_entry["origin"] == "bootstrap" else None
                    within &= value is not None
                    covariance[first, second] = 0.0 if value is None else value
        result = meta_analysis(estimates, covariance)
        for model, combined in (("fixed", result.fixed), ("random", result.random)):
            rows.append({
                "candidate": candidate, "baseline": baseline, "cell": cell, "level": level, "model": model,
                "traits": ";".join(trait for trait, _entry in entries), "estimate": text(combined.estimate), "standard_error": text(combined.standard_error),
                "ci_low": text(combined.estimate - CONFIDENCE_QUANTILE * combined.standard_error),
                "ci_high": text(combined.estimate + CONFIDENCE_QUANTILE * combined.standard_error), "z": text(combined.z_score),
                "p_value": text(math.erfc(abs(combined.z_score) / math.sqrt(2.0))), "heterogeneity_q": text(result.heterogeneity_q),
                "heterogeneity_p": text(result.heterogeneity_p), "i_squared": text(result.i_squared),
                "tau_squared": text(result.tau_squared) if model == "random" else "", "positive_traits": str(result.positive),
                "covariance": "within_run_joint_bootstrap_runs_independent" if within else "independent_traits_assumed",
            })
    return rows


def loci_rows(exports: Exports) -> list[dict[str, str]]:
    """Per panel window and trait (LOCI_COLUMNS), the with_sv model's genetic variance in the window split by class:
    SNVs, short indels and every structural class (workspace_pipeline.locus_classes); blank where locus_variance.tsv
    shows no value."""
    windows: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in exports.tables["locus_variance"]:
        windows.setdefault((row["trait"], row["locus"]), {}).setdefault(row["arm"], {})[row["variant_class"]] = row
    rows = []
    for (trait, locus), arms in windows.items():
        classes = arms.get("with_sv", {})
        whole = classes.get("all")
        if whole is None:
            continue
        structural = {name: number(row["window_share"]) for name, row in classes.items() if name not in ("all", "snv", pipeline.SHORT_INDEL_CLASS)}
        shown = {name: share for name, share in structural.items() if math.isfinite(share)}
        top = max(shown, key=shown.get) if shown else ""
        rows.append({
            "trait": trait, "locus": locus, "kind": whole["kind"], "chromosome": whole["chromosome"], "start": whole["start"], "end": whole["end"],
            "genome_share": whole["genome_share"], "snv_share": classes.get("snv", {}).get("window_share", ""),
            "short_indel_share": classes.get(pipeline.SHORT_INDEL_CLASS, {}).get("window_share", ""),
            "sv_share": text(sum(shown.values())) if shown else "", "top_sv_class": top, "top_sv_class_share": text(shown[top]) if top else "",
            "snv_indel_genome_share": arms.get("snv_indel", {}).get("all", {}).get("genome_share", ""), "rare_left_out": whole["rare_left_out"],
        })
    return rows


def write_table(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        writer.writerows([[row.get(column, "") for column in columns] for row in rows])
    return path


# ------------------------------------------------------------------ figures


def log_ticks(low: float, high: float) -> list[float]:
    """The powers of ten from below ``low`` to above ``high``, at least two."""
    first = math.floor(math.log10(low))
    return [10.0 ** power for power in range(first, max(math.ceil(math.log10(high)), first + 1) + 1)]


def power_text(value: float, unit: str = "") -> str:
    """A power of ten as a tick label: 1, 10 and 100 written out, others as 10 with a raised exponent."""
    power = round(math.log10(value))
    if 0 <= power <= 2:
        return f"{10 ** power}{unit}"
    return f'10<tspan dy="-14" font-size="70%">{str(power).replace("-", "−")}</tspan><tspan dy="14">{unit}</tspan>'


def bp_text(value: float) -> str:
    """A length in base pairs as a tick label: 50, 500, 5k, 50k, 5M."""
    for size, suffix in ((1e6, "M"), (1e3, "k")):
        if value >= size:
            return f"{value / size:g}{suffix}"
    return f"{value:g}"


def axis_left(parts: list[str], left: float, top: float, bottom: float, right: float) -> None:
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{figures.GREY}" stroke-width="3"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="{figures.GREY}" stroke-width="3"/>')


def trait_labels(parts: list[str], names: list[str], x, bottom: float, flagged: set[str] = frozenset()) -> None:  # type: ignore[assignment]
    for name in names:
        shown = label(name) + (STATUS_MARK if name in flagged else "")
        parts.append(f'<text class="row" transform="translate({x(name) + 8:.1f} {bottom + 22:.1f}) rotate(-30)" text-anchor="end">{html.escape(shown)}</text>')


def diseases_figure(summary: list[dict[str, str]], meta: list[dict[str, str]], width: float = 15.40, height: float = 4.60) -> str | None:
    """Two panels over one trait axis, the traits ordered by the with_sv partial r^2: above, each method's held-out
    partial r^2 (log scale) with its 95% bootstrap interval; below, what SVs add on the same held-out people (GAINS: to
    SV-PGS, and to LDpred2-auto when SV-PGS's structural part is stacked with it) with its interval, then the traits
    combined (fixed effect, a diamond) where a meta-analysis exists. An unpaired LDpred2 value is an open square without
    an interval; a flagged trait's label carries STATUS_MARK."""
    rows = [row for row in summary if math.isfinite(number(row["with_sv_partial_r2"])) or math.isfinite(number(row["snv_indel_partial_r2"]))]
    if not rows:
        return None
    rows.sort(key=lambda row: -number(row["with_sv_partial_r2"]) if math.isfinite(number(row["with_sv_partial_r2"])) else 0.0)
    names = [row["trait"] for row in rows]
    pairs = {prefix: (candidate, baseline) for candidate, baseline, prefix in PAIRS}
    combined = {
        prefix: entry for prefix, *_rest in GAINS
        for entry in meta if (entry["candidate"], entry["baseline"], entry["cell"], entry["model"]) == (*pairs[prefix], "all", "fixed")
        and math.isfinite(number(entry["estimate"]))
    }
    columns = names + (["combined"] if combined else [])
    w, h = width * 100, height * 100
    # each panel's title over its plot, beside its tick labels: above the upper one with the key, between the two
    key_y, top, middle, lower_top, bottom = 30.0, 58.0, 170.0, 214.0, h - 128.0
    left, right = 150.0, 30.0
    pitch = (w - left - right) / len(columns)
    style = figures.SIM_STYLE + f".row {{ font-size: {20 * figures.PT:.1f}px; fill: {figures.INK}; }}"
    parts = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" xmlns="http://www.w3.org/2000/svg"><style>{style}</style>']

    def x(name: str) -> float:
        return left + pitch * (columns.index(name) + 0.5)

    def key(cursor: float, y: float, entries: list[tuple[str, str, bool]]) -> None:
        for key_label, colour, filled in entries:
            parts.append(f'<circle cx="{cursor + 12}" cy="{y - 12}" r="11" fill="{colour if filled else "#fff"}" stroke="{colour}" stroke-width="5"/>')
            parts.append(f'<text class="key" x="{cursor + 32}" y="{y}">{key_label}</text>')
            cursor += 32 + figures.text_width(key_label, 26) + 30

    # the partial r^2 panel, from the decade below the smallest estimate (an interval reaching under it is cut there)
    prefixes = [prefix for _method, prefix, *_rest in METHODS]
    points = [number(row[f"{prefix}_partial_r2"]) for row in rows for prefix in prefixes]
    uppers = [number(row[f"{prefix}_ci_high"]) for row in rows for prefix in prefixes]
    positive = [value for value in points if math.isfinite(value) and value > 0.0]
    if not positive:
        return None
    ticks = log_ticks(min(positive), max(value for value in positive + uppers if math.isfinite(value) and value > 0.0))
    low, high = ticks[0], ticks[-1]

    def y1(value: float) -> float:
        value = min(max(value, low), high)
        return middle - (math.log10(value) - math.log10(low)) / (math.log10(high) - math.log10(low)) * (middle - top)

    axis_left(parts, left, top, middle, w - right)
    # every decade is marked; where they crowd, every other one is labelled, the top one always
    stride = 1 if (middle - top) / max(len(ticks) - 1, 1) >= 40.0 else 2
    for index, tick in enumerate(ticks):
        parts.append(f'<line x1="{left - 10}" y1="{y1(tick):.1f}" x2="{left}" y2="{y1(tick):.1f}" stroke="{figures.GREY}" stroke-width="3"/>')
        if (len(ticks) - 1 - index) % stride == 0:
            parts.append(f'<text class="tick" x="{left - 16}" y="{y1(tick) + 9:.1f}" text-anchor="end">{power_text(tick)}</text>')
    title = "Partial r²"
    parts.append(f'<text class="axis" x="{left + 12}" y="{key_y + 12}">{title}</text>')
    key(left + 12 + figures.text_width(title, 26) + 40, key_y + 12, [(key_label, colour, filled) for _method, _prefix, key_label, colour, filled, _offset in METHODS])
    for row in rows:
        for method, prefix, _key_label, colour, filled, offset in METHODS:
            point = number(row[f"{prefix}_partial_r2"])
            if not math.isfinite(point) or point <= 0.0:
                continue
            px = x(row["trait"]) + offset
            lower, upper = number(row[f"{prefix}_ci_low"]), number(row[f"{prefix}_ci_high"])
            if math.isfinite(lower) and math.isfinite(upper):
                parts.append(f'<line x1="{px:.1f}" y1="{y1(max(lower, low)):.1f}" x2="{px:.1f}" y2="{y1(upper):.1f}" stroke="{colour}" stroke-width="4"/>')
            if method == BASELINE and row["ldpred2_source"] == "unpaired":
                parts.append(f'<rect x="{px - 9:.1f}" y="{y1(point) - 9:.1f}" width="18" height="18" fill="#fff" stroke="{colour}" stroke-width="5"/>')
            else:
                parts.append(f'<circle cx="{px:.1f}" cy="{y1(point):.1f}" r="10" fill="{colour if filled else "#fff"}" stroke="{colour}" stroke-width="5"/>')

    # the gain panel: each GAINS difference on the same people, then the traits combined
    gains = [number(row[f"{prefix}{suffix}"]) for row in rows for prefix, *_rest in GAINS for suffix in ("", "_ci_low", "_ci_high")]
    gains += [number(entry[key_name]) for entry in combined.values() for key_name in ("estimate", "ci_low", "ci_high")]
    finite = [0.0] + [value for value in gains if math.isfinite(value)]
    glow, ghigh, step = figures.nice_range(min(finite), max(finite), steps=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05), most=int((bottom - lower_top) // 36))

    def y2(value: float) -> float:
        return bottom - (value - glow) / (ghigh - glow) * (bottom - lower_top)

    axis_left(parts, left, lower_top, bottom, w - right)
    parts.append(f'<line x1="{left}" y1="{y2(0.0):.1f}" x2="{w - right}" y2="{y2(0.0):.1f}" stroke="{figures.GREY}" stroke-width="2.5" stroke-dasharray="8 8"/>')
    # as the upper panel: where the steps crowd, every other one is labelled, zero always
    stride = 1 if (bottom - lower_top) / round((ghigh - glow) / step) >= 36.0 else 2
    for index in range(round(glow / step), round(ghigh / step) + 1):
        tick = index * step
        parts.append(f'<line x1="{left - 10}" y1="{y2(tick):.1f}" x2="{left}" y2="{y2(tick):.1f}" stroke="{figures.GREY}" stroke-width="3"/>')
        if index % stride == 0:
            parts.append(f'<text class="tick" x="{left - 16}" y="{y2(tick) + 9:.1f}" text-anchor="end">{figures.tick_text(tick, step) if index else "0"}</text>')
    title = "Gain in partial r² from SVs"
    parts.append(f'<text class="axis" x="{left + 12}" y="{lower_top - 12}">{title}</text>')
    drawn = [entry for entry in GAINS if any(math.isfinite(number(row[entry[0]])) for row in rows) or entry[0] in combined]
    key(left + 12 + figures.text_width(title, 26) + 40, lower_top - 12, [(key_label, colour, True) for _prefix, key_label, colour, _offset in drawn])
    for prefix, _key_label, colour, offset in drawn:
        for row in rows:
            gain = number(row[prefix])
            if not math.isfinite(gain):
                continue
            px = x(row["trait"]) + offset
            lower, upper = number(row[f"{prefix}_ci_low"]), number(row[f"{prefix}_ci_high"])
            if math.isfinite(lower) and math.isfinite(upper):
                parts.append(f'<line x1="{px:.1f}" y1="{y2(lower):.1f}" x2="{px:.1f}" y2="{y2(upper):.1f}" stroke="{colour}" stroke-width="4"/>')
            parts.append(f'<circle cx="{px:.1f}" cy="{y2(gain):.1f}" r="10" fill="{colour}" stroke="#fff" stroke-width="3"/>')
        if prefix in combined:
            px, estimate = x("combined") + offset, number(combined[prefix]["estimate"])
            lower, upper = number(combined[prefix]["ci_low"]), number(combined[prefix]["ci_high"])
            parts.append(f'<line x1="{px:.1f}" y1="{y2(lower):.1f}" x2="{px:.1f}" y2="{y2(upper):.1f}" stroke="{colour}" stroke-width="4"/>')
            parts.append(f'<polygon points="{px - 16:.1f},{y2(estimate):.1f} {px:.1f},{y2(estimate) - 16:.1f} {px + 16:.1f},{y2(estimate):.1f} {px:.1f},{y2(estimate) + 16:.1f}" fill="{colour}" stroke="#fff" stroke-width="3"/>')
    trait_labels(parts, columns, x, bottom, {row["trait"] for row in rows if row.get("status")})
    parts.append("</svg>")
    return "".join(parts)


def effects_figure(tables: dict, trait: str, width: float = 15.40, height: float = 5.00) -> str | None:
    """For one trait's with_sv model: per SV class (and every SV together), the prior variance of an effect relative to
    SNVs (filled) beside the posterior second moment's (open, with its 95% block-jackknife interval), on a log scale; and
    the prior's enrichment by length bin for the four main classes (CLASS_COLOURS). A suppressed class or bin is not
    drawn."""
    classes = [row for row in tables["prior_enrichment"] if row["trait"] == trait and row["arm"] == "with_sv"
               and row["variant_class"] != "snv" and math.isfinite(number(row["prior_enrichment"]))]
    lengths = [row for row in tables["prior_by_length"] if row["trait"] == trait and row["arm"] == "with_sv"
               and row["length_bin"] != "<50" and math.isfinite(number(row["prior_enrichment"]))]
    if not classes:
        return None
    classes.sort(key=lambda row: (row["variant_class"] != "all_structural", -number(row["prior_enrichment"])))
    names = [row["variant_class"] for row in classes]
    w, h = width * 100, height * 100
    key_y, top, bottom = 34.0, 74.0, h - 150.0
    left, gap, right = 150.0, 170.0, 30.0
    first_right = left + (w - left - gap - right) * 0.48
    second_left = first_right + gap
    style = figures.SIM_STYLE + f".row {{ font-size: {22 * figures.PT:.1f}px; fill: {figures.INK}; }}"
    parts = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" xmlns="http://www.w3.org/2000/svg"><style>{style}</style>']
    values = [number(row[key]) for row in classes for key in ("prior_enrichment", "posterior_enrichment", "posterior_enrichment_ci_low", "posterior_enrichment_ci_high")]
    values += [number(row["prior_enrichment"]) for row in lengths] + [1.0]
    ticks = log_ticks(min(value for value in values if math.isfinite(value) and value > 0.0), max(value for value in values if math.isfinite(value)))
    low, high = ticks[0], ticks[-1]

    def y(value: float) -> float:
        value = min(max(value, low), high)
        return bottom - (math.log10(value) - math.log10(low)) / (math.log10(high) - math.log10(low)) * (bottom - top)

    def ticks_on(x0: float) -> None:
        for tick in ticks:
            parts.append(f'<line x1="{x0 - 10}" y1="{y(tick):.1f}" x2="{x0}" y2="{y(tick):.1f}" stroke="{figures.GREY}" stroke-width="3"/>')
            parts.append(f'<text class="tick" x="{x0 - 16}" y="{y(tick) + 9:.1f}" text-anchor="end">{power_text(tick, "×")}</text>')

    pitch = (first_right - left) / len(names)

    def x1(name: str) -> float:
        return left + pitch * (names.index(name) + 0.5)

    axis_left(parts, left, top, bottom, first_right)
    parts.append(f'<line x1="{left}" y1="{y(1.0):.1f}" x2="{first_right}" y2="{y(1.0):.1f}" stroke="{figures.GREY}" stroke-width="2.5" stroke-dasharray="8 8"/>')
    ticks_on(left)
    parts.append(f'<text class="axis" transform="translate(30 {(top + bottom) / 2:.1f}) rotate(-90)" text-anchor="middle">Effect variance vs. SNVs</text>')
    cursor = left
    for key_label, filled in (("Learned prior", True), ("Posterior (95% CI)", False)):
        parts.append(f'<circle cx="{cursor + 12}" cy="{key_y}" r="11" fill="{figures.TEAL if filled else "#fff"}" stroke="{figures.TEAL if filled else figures.NAVY}" stroke-width="5"/>')
        parts.append(f'<text class="key" x="{cursor + 32}" y="{key_y + 12}">{key_label}</text>')
        cursor += 32 + figures.text_width(key_label, 26) + 40
    for row in classes:
        px = x1(row["variant_class"])
        posterior, lower, upper = (number(row[key]) for key in ("posterior_enrichment", "posterior_enrichment_ci_low", "posterior_enrichment_ci_high"))
        if math.isfinite(lower) and math.isfinite(upper):
            parts.append(f'<line x1="{px + 14:.1f}" y1="{y(max(lower, low)):.1f}" x2="{px + 14:.1f}" y2="{y(upper):.1f}" stroke="{figures.NAVY}" stroke-width="4"/>')
        if math.isfinite(posterior) and posterior > 0.0:
            parts.append(f'<circle cx="{px + 14:.1f}" cy="{y(posterior):.1f}" r="10" fill="#fff" stroke="{figures.NAVY}" stroke-width="5"/>')
        parts.append(f'<circle cx="{px - 14:.1f}" cy="{y(number(row["prior_enrichment"])):.1f}" r="10" fill="{figures.TEAL}" stroke="{figures.TEAL}" stroke-width="5"/>')
    for name in names:
        parts.append(f'<text class="row" transform="translate({x1(name):.1f} {bottom + 30:.1f}) rotate(-35)" text-anchor="end">{html.escape(CLASS_LABELS.get(name, name))}</text>')

    # the length panel: each bin between its edges (decades from 50 bp), each class labelled at its line's end
    bins = sorted({row["length_bin"] for row in lengths}, key=lambda level: number(level.split("-")[0]))
    if bins:
        edges = [number(level.split("-")[0]) for level in bins] + [number(bins[-1].split("-")[1]) + 1.0]
        drawn = [CLASS_LABELS[name] for name in CLASS_COLOURS if any(row["variant_class"] == name for row in lengths)]
        plot_right = w - right - 40.0 - max(figures.text_width(name, 22) for name in drawn) if drawn else w - right

        def x2(level: str) -> float:
            return second_left + (plot_right - second_left) * (bins.index(level) + 0.5) / len(bins)

        axis_left(parts, second_left, top, bottom, plot_right)
        parts.append(f'<line x1="{second_left}" y1="{y(1.0):.1f}" x2="{plot_right}" y2="{y(1.0):.1f}" stroke="{figures.GREY}" stroke-width="2.5" stroke-dasharray="8 8"/>')
        ticks_on(second_left)
        parts.append(f'<text class="axis" transform="translate({second_left - 124:.1f} {(top + bottom) / 2:.1f}) rotate(-90)" text-anchor="middle">Prior vs. SNVs</text>')
        for index, edge in enumerate(edges):
            edge_x = second_left + (plot_right - second_left) * index / len(bins)
            parts.append(f'<line x1="{edge_x:.1f}" y1="{bottom}" x2="{edge_x:.1f}" y2="{bottom + 10}" stroke="{figures.GREY}" stroke-width="3"/>')
            parts.append(f'<text class="tick" x="{edge_x:.1f}" y="{bottom + 44}" text-anchor="middle">{bp_text(edge)}</text>')
        parts.append(f'<text class="axis" x="{(second_left + plot_right) / 2:.1f}" y="{bottom + 96}" text-anchor="middle">SV length (bp)</text>')
        series: dict[str, list[tuple[str, float]]] = {}
        for row in lengths:
            series.setdefault(row["variant_class"], []).append((row["length_bin"], number(row["prior_enrichment"])))
        ends = []
        for name, colour in CLASS_COLOURS.items():
            if name not in series:
                continue
            marks = sorted(series[name], key=lambda entry: bins.index(entry[0]))
            if len(marks) > 1:
                parts.append('<polyline points="' + " ".join(f"{x2(level):.1f},{y(value):.1f}" for level, value in marks)
                             + f'" fill="none" stroke="{colour}" stroke-width="6" stroke-linejoin="round"/>')
            for level, value in marks:
                parts.append(f'<circle cx="{x2(level):.1f}" cy="{y(value):.1f}" r="9" fill="{colour}" stroke="#fff" stroke-width="3"/>')
            ends.append((x2(marks[-1][0]), y(marks[-1][1]), colour, CLASS_LABELS[name]))
        placed = figures.spread_labels([end[1] for end in ends], 34.0, top + 8, bottom - 4)
        for (end_x, end_y, colour, name), label_y in zip(ends, placed):
            parts.append(f'<polyline points="{end_x + 14:.1f},{end_y:.1f} {plot_right + 10:.1f},{label_y:.1f} {plot_right + 24:.1f},{label_y:.1f}" fill="none" stroke="{colour}" stroke-width="2.5"/>')
            parts.append(f'<text class="end" x="{plot_right + 32:.1f}" y="{label_y + 9:.1f}" style="fill:{colour}">{name}</text>')
    parts.append("</svg>")
    return "".join(parts)


def save(name: str, svg: str | None, width: float, height: float, out: Path) -> list[Path]:
    """The figure's SVG and, where Chrome is installed, its PNG; a figure without data removes its previous files."""
    made = []
    for suffix in (".svg", ".png", ".html"):
        (out / f"{name}{suffix}").unlink(missing_ok=True)
    if svg is None:
        return made
    (out / f"{name}.svg").write_text(svg, encoding="utf-8")
    made.append(out / f"{name}.svg")
    if Path(figures.CHROME).exists():
        figures.OUT = out.resolve()
        made.append(figures.render(name, figures.page(svg, width, height), width, height))
    return made


# ------------------------------------------------------------------ review figures (matplotlib)


REVIEW_FIGURES = ("aou_review_accuracy", "aou_review_gains", "aou_review_ancestry", "aou_review_enrichment", "aou_review_auc", "aou_review_loci")
"""The review figures, in aou_review.pdf's order: each method's partial r^2 per trait; each paired difference per trait
with the traits combined; the SV gains per ancestry group; the SV classes' heritability enrichment; a disease's paired
AUC differences; the panel windows' genetic variance by class."""
MM = 1 / 25.4
BLUE, VERMILLION, ORANGE, SKY, GREEN, PURPLE, YELLOW, BLACK, GREY = "#0072B2", "#D55E00", "#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#F0E442", "#000000", "#6E6E6E"
"""Okabe and Ito's colour-blind-safe palette, sim_gallery.py's, with its grey."""
REVIEW_METHODS = (
    # (summary prefix, label, colour)
    ("ldpred2", "LDpred2-auto", VERMILLION), ("snv_indel", "SV-PGS without SVs", SKY), ("with_sv", "SV-PGS with SVs", BLUE),
    ("stacked", "LDpred2-auto + SV-PGS", PURPLE), ("stacked_structural", "LDpred2-auto + SV part", GREEN),
)
REVIEW_PAIRS = (
    # (summary prefix, panel title, colour)
    ("sv_gain", "SVs in SV-PGS\nwith SVs − without", BLUE), ("stack_sv_gain", "SVs on LDpred2-auto\nLDpred2 + SV part − LDpred2", GREEN),
    ("with_sv_vs_ldpred2", "SV-PGS vs LDpred2-auto\nwith SVs − LDpred2", VERMILLION), ("stack_gain", "SV-PGS on LDpred2-auto\nLDpred2 + SV-PGS − LDpred2", PURPLE),
)
ANCESTRY_COLOURS = {"afr": ORANGE, "amr": GREEN, "eas": SKY, "eur": BLUE, "mid": PURPLE, "sas": VERMILLION}
"""All of Us's genetic ancestry groups, each always in its own colour; any other group is grey."""
LOCUS_KINDS = {"sv_known": "known SV", "sv_suspected": "suspected SV", "snv_control": "SNV control"}
REVIEW_NOTES = {
    "intervals": "95%: the report's family-clustered bootstrap percentiles (partial_r2.tsv, paired_differences.tsv); compare.tsv's own and "
                 f"comparison.tsv's estimate +/- {CONFIDENCE_QUANTILE:.2f} standard errors are influence-function intervals. A point without "
                 "an interval is drawn open.",
    "p_values": "Shown only where a trait's report reads one test fold; across several folds the error leaves out the folds' shared training.",
    "meta_analysis": "Fixed and random effects over the unflagged traits with a p-value: a run's own where it holds exactly those traits, else "
                     "each run's joint-bootstrap covariance with the runs as independent (optimistic: they share people); aou_meta.tsv's "
                     "covariance column says which.",
    "flags": f"{STATUS_MARK}: a quarantined or uncertified fit (quarantined.tsv, uncertified.tsv); left out of every meta-analysis made here.",
    "enrichment": "Heritability enrichment: an SV class's share of the with_sv model's posterior genetic variance over its share of the fitted "
                  f"records, with {pipeline.JACKKNIFE_BLOCKS}-block jackknife intervals; LD between records is left out of each record's variance.",
    "loci": "Each panel window's share of the with_sv model's posterior genetic variance by class (SNVs, short indels, each SV class); a record "
            f"with fewer than {MINIMUM_REPORTED_PARTICIPANTS} expected carriers enters no value.",
    "suppression": "Every input passed workspace_pipeline.check_exportable; suppressed cells stay blank and are never filled from another table.",
}
"""aou_review_notes.json: what the review figures' intervals and summaries do and do not hold."""


def journal_style(font_manager) -> dict:
    """sim_gallery.py's journal style: 7 pt type (Helvetica, else Arial), thin axes, no top or right spine."""
    names = {entry.name for entry in font_manager.fontManager.ttflist}
    font = next((name for name in ("Helvetica", "Arial", "Helvetica Neue") if name in names), "DejaVu Sans")
    return {
        "font.family": font, "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0, "legend.frameon": False,
        "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.default": "regular", "axes.titlelocation": "left",
    }


def log_axis(ax, values: list[float]) -> None:
    """A log x axis over the positive ``values``, ticked at the powers of ten it spans, with 2 and 5 times them over at
    most three decades and 1.5, 3 and 7 times them too within one (each written out, no minor labels)."""
    from matplotlib.ticker import FixedLocator, NullFormatter

    positive = [value for value in values if math.isfinite(value) and value > 0.0]
    low, high = min(positive), max(positive)
    steps = (1, 1.5, 2, 3, 5, 7) if high < 10.0 * low else (1, 2, 5) if high < 1e3 * low else (1,)
    ticks = [step * 10.0 ** power for power in range(math.floor(math.log10(low)) - 1, math.ceil(math.log10(high)) + 1) for step in steps]
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([tick for tick in ticks if low <= tick <= high] or [low, high]))
    ax.xaxis.set_major_formatter(lambda value, _position: f"{value:g}")
    ax.xaxis.set_minor_formatter(NullFormatter())


def trait_order(summary: list[dict[str, str]]) -> list[str]:
    """The traits by their with_sv partial r^2, largest first (the snv_indel arm's where with_sv has none)."""
    def key(row: dict[str, str]) -> float:
        value = number(row["with_sv_partial_r2"]) if math.isfinite(number(row["with_sv_partial_r2"])) else number(row["snv_indel_partial_r2"])
        return -value if math.isfinite(value) else 0.0

    return [row["trait"] for row in sorted(summary, key=key)]


def tick_labels(names: list[str], flagged: set[str]) -> list[str]:
    return [label(name) + (STATUS_MARK if name in flagged else "") for name in names]


def interval_point(ax, y: float, value: float, low: float, high: float, colour: str, marker: str = "o", floor: float = -math.inf) -> None:
    """One estimate with its interval (a bar, clipped at ``floor``), filled where it has one and open where not."""
    if not math.isfinite(value):
        return
    filled = math.isfinite(low) and math.isfinite(high)
    if filled:
        ax.hlines(y, max(low, floor), high, color=colour, linewidth=0.8, zorder=2)
    ax.scatter([value], [y], s=14, marker=marker, facecolor=colour if filled else "white", edgecolor=colour, linewidth=0.8, zorder=3)


def accuracy_review(plt, summary: list[dict[str, str]]):
    """Each method's held-out partial r^2 per trait (log scale) with its 95% interval."""
    rows = {row["trait"]: row for row in summary}
    names = [name for name in trait_order(summary) if any(number(rows[name][f"{prefix}_partial_r2"]) > 0.0 for prefix, *_rest in REVIEW_METHODS)]
    methods = [method for method in REVIEW_METHODS if any(number(rows[name][f"{method[0]}_partial_r2"]) > 0.0 for name in names)]
    if not names:
        return None
    values = [number(rows[name][f"{prefix}_{column}"]) for name in names for prefix, *_rest in methods for column in ("partial_r2", "ci_high")]
    positive = [value for value in values if value > 0.0]
    floor = min(positive) / 3.0
    fig, ax = plt.subplots(figsize=(120 * MM, (22 + 5.5 * len(names)) * MM), layout="constrained")
    for index, (prefix, method_label, colour) in enumerate(methods):
        offset = (index - (len(methods) - 1) / 2) * 0.8 / len(methods)
        for y, name in enumerate(names):
            row = rows[name]
            marker = "s" if prefix == "ldpred2" and row["ldpred2_source"] == "unpaired" else "o"
            interval_point(ax, y + offset, number(row[f"{prefix}_partial_r2"]), number(row[f"{prefix}_ci_low"]), number(row[f"{prefix}_ci_high"]), colour, marker, floor)
        ax.scatter([], [], s=14, color=colour, label=method_label)
    log_axis(ax, [floor, *positive])
    ax.set_xlim(left=floor)
    ax.set_yticks(range(len(names)), tick_labels(names, {name for name in names if rows[name]["status"]}))
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xlabel("Held-out partial r² (95% interval)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, handletextpad=0.2, columnspacing=1.0)
    return fig


def gains_review(plt, summary: list[dict[str, str]], meta: list[dict[str, str]]):
    """Each paired difference of REVIEW_PAIRS per trait with its 95% interval, then the traits combined (fixed and random
    effects, diamonds), one panel a pair."""
    rows = {row["trait"]: row for row in summary}
    names = trait_order(summary)
    pairs = {prefix: (candidate, baseline) for candidate, baseline, prefix in PAIRS}
    combined = {
        (prefix, entry["model"]): entry for prefix, *_rest in REVIEW_PAIRS for entry in meta
        if (entry["candidate"], entry["baseline"], entry["cell"]) == (*pairs[prefix], "all")
    }
    drawn = [pair for pair in REVIEW_PAIRS if any(math.isfinite(number(rows[name][pair[0]])) for name in names) or (pair[0], "fixed") in combined]
    if not drawn:
        return None
    labels = tick_labels(names, {name for name in names if rows[name]["status"]})
    models = [model for model in ("fixed", "random") if any((prefix, model) in combined for prefix, *_rest in drawn)]
    labels += [f"All traits ({model} effect{'s' if model == 'random' else ''})" for model in models]
    fig, axes = plt.subplots(1, len(drawn), sharey=True, figsize=(183 * MM, (24 + 5 * len(labels)) * MM), layout="constrained", squeeze=False)
    for ax, (prefix, title, colour) in zip(axes[0], drawn):
        ax.axvline(0.0, color=GREY, linewidth=0.6, linestyle=(0, (3, 2)))
        for y, name in enumerate(names):
            row = rows[name]
            interval_point(ax, y, number(row[prefix]), number(row[f"{prefix}_ci_low"]), number(row[f"{prefix}_ci_high"]), colour)
        for index, model in enumerate(models):
            entry = combined.get((prefix, model))
            if entry is not None:
                interval_point(ax, len(names) + index, number(entry["estimate"]), number(entry["ci_low"]), number(entry["ci_high"]), BLACK, "D")
        ax.set_title(title)
        ax.set_xlabel("Δ held-out partial r²")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
    axes[0][0].set_yticks(range(len(labels)), labels)
    axes[0][0].set_ylim(len(labels) - 0.5, -0.5)
    return fig



def ancestry_review(plt, ancestry: list[dict[str, str]], summary: list[dict[str, str]], meta: list[dict[str, str]]):
    """The SV gains (within SV-PGS, and on top of LDpred2-auto) per trait and ancestry group with 95% intervals, each
    trait's label with its heterogeneity p across the groups, then each group's traits combined (fixed effect)."""
    drawn = [pair for pair in REVIEW_PAIRS[:2] if any(math.isfinite(number(row[pair[0]])) for row in ancestry)]
    if not drawn:
        return None
    levels = sorted({row["level"] for row in ancestry})
    found = {(row["trait"], row["level"]): row for row in ancestry}
    rows = {row["trait"]: row for row in summary}
    names = [name for name in trait_order(summary) if any((name, level) in found for level in levels)]
    labels = []
    for name in names:
        p_value = number(rows[name]["ancestry_heterogeneity_p"])
        labels.append(label(name) + (STATUS_MARK if rows[name]["status"] else "") + (f"  (het. p = {p_value:.2g})" if math.isfinite(p_value) else ""))
    labels.append("All traits (fixed effect)")
    pairs = {prefix: (candidate, baseline) for candidate, baseline, prefix in PAIRS}
    fig, axes = plt.subplots(1, len(drawn), sharey=True, figsize=(150 * MM, (26 + 6 * len(labels)) * MM), layout="constrained", squeeze=False)
    for ax, (prefix, title, _colour) in zip(axes[0], drawn):
        ax.axvline(0.0, color=GREY, linewidth=0.6, linestyle=(0, (3, 2)))
        combined = {entry["level"]: entry for entry in meta if (entry["candidate"], entry["baseline"], entry["cell"], entry["model"]) == (*pairs[prefix], "ancestry", "fixed")}
        for index, level in enumerate(levels):
            offset = (index - (len(levels) - 1) / 2) * 0.8 / len(levels)
            colour = ANCESTRY_COLOURS.get(level, GREY)
            for y, name in enumerate(names):
                row = found.get((name, level))
                if row is not None:
                    interval_point(ax, y + offset, number(row[prefix]), number(row[f"{prefix}_ci_low"]), number(row[f"{prefix}_ci_high"]), colour)
            if level in combined:
                entry = combined[level]
                interval_point(ax, len(names) + offset, number(entry["estimate"]), number(entry["ci_low"]), number(entry["ci_high"]), colour, "D")
            if ax is axes[0][0]:
                ax.scatter([], [], s=14, color=colour, label=level.upper())
        ax.set_title(title)
        ax.set_xlabel("Δ held-out partial r²")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
    axes[0][0].set_yticks(range(len(labels)), labels)
    axes[0][0].set_ylim(len(labels) - 0.5, -0.5)
    fig.legend(loc="outside upper right", ncol=len(levels), handletextpad=0.2)
    return fig


def enrichment_review(plt, exports: Exports, summary: list[dict[str, str]]):
    """The with_sv model's heritability enrichment per trait: every SV together with its 95% jackknife interval (log
    scale), and each SV class's (log2, a dot where its interval excludes 1)."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    entries = {(row["trait"], row["variant_class"]): row for row in exports.tables["prior_enrichment"] if row["arm"] == "with_sv"}
    names = [name for name in trait_order(summary) if number(entries.get((name, "all_structural"), {}).get("heritability_enrichment")) > 0.0]
    if not names:
        return None
    classes = list(dict.fromkeys(name for _trait, name in entries if name not in ("snv", "all_structural")))
    flagged = {row["trait"] for row in summary if row["status"]}
    width = 183 if classes else 90
    fig, axes = plt.subplots(1, 2 if classes else 1, figsize=(width * MM, (26 + 5.5 * len(names)) * MM), layout="constrained", squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1.4]} if classes else None)
    ax = axes[0][0]
    ax.axvline(1.0, color=GREY, linewidth=0.6, linestyle=(0, (3, 2)))
    points = [number(entries[name, "all_structural"][column]) for name in names for column in ("heritability_enrichment", "heritability_enrichment_ci_low")]
    floor = min(value for value in points if value > 0.0) / 2.0
    for y, name in enumerate(names):
        row = entries[name, "all_structural"]
        interval_point(ax, y, number(row["heritability_enrichment"]), number(row["heritability_enrichment_ci_low"]), number(row["heritability_enrichment_ci_high"]), BLUE, floor=floor)
    log_axis(ax, [floor, 1.0, *(number(entries[name, "all_structural"]["heritability_enrichment_ci_high"]) for name in names), *points])
    ax.set_yticks(range(len(names)), tick_labels(names, flagged))
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xlabel("Heritability enrichment, all SVs (95% interval)")
    ax.set_title("SV share of genetic variance / SV share of records")
    if classes:
        ax = axes[0][1]
        matrix = np.array([[math.log2(value) if (value := number(entries.get((name, kind), {}).get("heritability_enrichment"))) > 0.0 else np.nan
                            for kind in classes] for name in names])
        limit = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
        colours = LinearSegmentedColormap.from_list("enrichment", [BLUE, "#FFFFFF", VERMILLION])
        colours.set_bad("#E6E6E6")
        image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=colours, norm=TwoSlopeNorm(0.0, -max(limit, 1e-9), max(limit, 1e-9)), interpolation="nearest")
        for y, name in enumerate(names):
            for x, kind in enumerate(classes):
                row = entries.get((name, kind), {})
                low, high = number(row.get("heritability_enrichment_ci_low")), number(row.get("heritability_enrichment_ci_high"))
                if (math.isfinite(low) and low > 1.0) or (math.isfinite(high) and high < 1.0):
                    ax.plot(x, y, marker="o", markersize=2.0, color=BLACK)
        ax.set_xticks(range(len(classes)), [CLASS_LABELS.get(kind, kind) for kind in classes], rotation=35, ha="right")
        ax.set_yticks(range(len(names)), tick_labels(names, flagged))
        ax.set_title("Each SV class (dot: interval excludes 1)")
        fig.colorbar(image, ax=ax, shrink=0.8, label="log2 heritability enrichment")
    return fig


def auc_review(plt, summary: list[dict[str, str]]):
    """A disease's paired AUC differences, with SVs less without: the genetic scores alone and the whole predictor."""
    rows = {row["trait"]: row for row in summary}
    names = [name for name in trait_order(summary) if any(math.isfinite(number(rows[name][metric])) for metric in ("delta_pgs_auc", "delta_auc"))]
    if not names:
        return None
    fig, ax = plt.subplots(figsize=(100 * MM, (22 + 5.5 * len(names)) * MM), layout="constrained")
    ax.axvline(0.0, color=GREY, linewidth=0.6, linestyle=(0, (3, 2)))
    for offset, (metric, metric_label, colour) in ((-0.15, ("delta_pgs_auc", "Genetic score alone", BLUE)), (0.15, ("delta_auc", "Whole predictor", GREY))):
        for y, name in enumerate(names):
            row = rows[name]
            interval_point(ax, y + offset, number(row[metric]), number(row[f"{metric}_ci_low"]), number(row[f"{metric}_ci_high"]), colour)
        ax.scatter([], [], s=14, color=colour, label=metric_label)
    ax.set_yticks(range(len(names)), tick_labels(names, {name for name in names if rows[name]["status"]}))
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xlabel("Δ AUC, SV-PGS with SVs − without (95% interval)")
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    return fig


def loci_review(plt, exports: Exports):
    """Each panel window's share of the with_sv model's genetic variance there, by class: SNVs, short indels and each SV
    class, the windows by their SV share."""
    windows: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in exports.tables["locus_variance"]:
        if row["arm"] == "with_sv":
            windows.setdefault((row["trait"], row["locus"]), {})[row["variant_class"]] = row
    windows = {key: classes for key, classes in windows.items() if math.isfinite(number(classes.get("all", {}).get("genetic_variance")))}
    if not windows:
        return None
    structural = list(dict.fromkeys(kind for classes in windows.values() for kind in classes if kind not in ("all", "snv", pipeline.SHORT_INDEL_CLASS)))
    order = [("snv", "#D9D9D9"), (pipeline.SHORT_INDEL_CLASS, "#A6A6A6")]
    order += [(kind, colour) for kind, colour in zip(structural, (BLUE, ORANGE, GREEN, VERMILLION, PURPLE, SKY, YELLOW, BLACK))]

    def share(classes: dict[str, dict[str, str]], kind: str) -> float:
        value = number(classes.get(kind, {}).get("window_share"))
        return value if math.isfinite(value) else 0.0

    keys = sorted(windows, key=lambda key: -sum(share(windows[key], kind) for kind in structural))
    fig, ax = plt.subplots(figsize=(150 * MM, (24 + 4.5 * len(keys)) * MM), layout="constrained")
    left = np.zeros(len(keys))
    for kind, colour in order:
        widths = np.array([share(windows[key], kind) for key in keys])
        ax.barh(range(len(keys)), widths, left=left, height=0.7, color=colour, edgecolor="white", linewidth=0.3, label=CLASS_LABELS.get(kind, kind))
        left += widths
    ax.set_yticks(range(len(keys)), [
        f"{locus} · {label(trait)} ({LOCUS_KINDS.get(windows[trait, locus]['all']['kind'], windows[trait, locus]['all']['kind'])})" for trait, locus in keys
    ])
    ax.set_ylim(len(keys) - 0.5, -0.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Share of the window's genetic variance, SV-PGS with SVs")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=min(len(order), 6), handlelength=1.0)
    return fig


def review_figures(summary: list[dict[str, str]], ancestry: list[dict[str, str]], meta: list[dict[str, str]], exports: Exports, out: Path) -> list[Path]:
    """REVIEW_FIGURES, each a PDF and a 600 dpi PNG, all of them in aou_review.pdf, and aou_review_notes.json; a figure
    without data is not drawn. Every previous review file is removed first, so none is left beside tables it does not
    show. Without matplotlib nothing is drawn, and the reason goes to stderr."""
    for name in (*REVIEW_FIGURES, "aou_review"):
        for suffix in (".pdf", ".png"):
            (out / f"{name}{suffix}").unlink(missing_ok=True)
    (out / "aou_review_notes.json").unlink(missing_ok=True)
    try:
        import matplotlib
    except ImportError:
        print("aou_results: matplotlib is not installed, so no review figure is drawn (uv run --with matplotlib).", file=sys.stderr)
        return []
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.backends.backend_pdf import PdfPages

    builders = {
        "aou_review_accuracy": lambda: accuracy_review(plt, summary), "aou_review_gains": lambda: gains_review(plt, summary, meta),
        "aou_review_ancestry": lambda: ancestry_review(plt, ancestry, summary, meta), "aou_review_enrichment": lambda: enrichment_review(plt, exports, summary),
        "aou_review_auc": lambda: auc_review(plt, summary), "aou_review_loci": lambda: loci_review(plt, exports),
    }
    made: list[Path] = []
    drawn: list[str] = []
    logging.getLogger("fontTools").setLevel(logging.ERROR)  # its notes on the system Helvetica's tables, harmless
    with plt.rc_context(journal_style(font_manager)):
        book = None
        try:
            for name in REVIEW_FIGURES:
                fig = builders[name]()
                if fig is None:
                    continue
                try:
                    fig.savefig(out / f"{name}.pdf")
                    fig.savefig(out / f"{name}.png", dpi=600)
                    if book is None:
                        book = PdfPages(out / "aou_review.pdf")
                    book.savefig(fig)
                finally:
                    plt.close(fig)
                drawn.append(name)
                made += [out / f"{name}.pdf", out / f"{name}.png"]
        finally:
            if book is not None:
                book.close()
                made.append(out / "aou_review.pdf")
    notes = out / "aou_review_notes.json"
    notes.write_text(json.dumps({**REVIEW_NOTES, "figures": drawn}, indent=2) + "\n", encoding="utf-8")
    return [*made, notes]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("exports", nargs="+", type=Path, help="Each run's directory of exported tables (and its compare.tsv and ldpred2.jsonl).")
    parser.add_argument("--ldpred2", nargs="*", type=Path, default=[], help="More LDpred2-auto aggregate lines (aou_ldpred2.py --out), by trait.")
    parser.add_argument("--region", default="", help="The region of compare.tsv and ldpred2.jsonl lines read (default: the whole genome's).")
    parser.add_argument("--trait", default="hypertension", help="The trait whose SV effects the poster's effects figure shows.")
    parser.add_argument("--out", type=Path, default=figures.OUT, help="The output directory.")
    args = parser.parse_args(argv)
    try:
        exports = load(args.exports, args.ldpred2, args.region)
    except ValueError as error:
        parser.error(str(error))
    summary, ancestry, meta, loci = summary_rows(exports), ancestry_rows(exports), meta_rows(exports), loci_rows(exports)
    args.out.mkdir(parents=True, exist_ok=True)
    made = [
        write_table(args.out / "aou_summary.tsv", SUMMARY_COLUMNS, summary), write_table(args.out / "aou_ancestry.tsv", ANCESTRY_COLUMNS, ancestry),
        write_table(args.out / "aou_meta.tsv", pipeline.META_ANALYSIS_COLUMNS, meta), write_table(args.out / "aou_loci.tsv", LOCI_COLUMNS, loci),
    ]
    made += save("fig_aou_diseases", diseases_figure(summary, meta), 15.40, 4.60, args.out)
    made += save("fig_aou_effects", effects_figure(exports.tables, args.trait), 15.40, 5.00, args.out)
    made += review_figures(summary, ancestry, meta, exports, args.out)
    for path in made:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
