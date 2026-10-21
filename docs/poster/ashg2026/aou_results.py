"""The All of Us results on the poster: one row per trait, the traits' combined SV gain, and the two All of Us figures,
from the aggregate tables an SV-PGS run exports.

Every input is a table the workspace pipeline's export step staged (``<run>/export``) after its export check
(``workspace_pipeline.check_exportable``): partial_r2.tsv, paired_differences.tsv, meta_analysis.tsv, trait_covariance.tsv
and heterogeneity.tsv (the held-out partial r^2 of each arm, of LDpred2-auto on the same people and of LDpred2-auto
stacked with SV-PGS, with family-clustered bootstrap intervals, overall and per ancestry group), comparison.tsv (a
disease's paired AUC difference), and prior_enrichment.tsv and prior_by_length.tsv (the learned prior by SV class and
length, and each class's share of the genetic variance). Nothing here reads a participant, and a suppressed cell stays
blank. Several export directories (runs fitted apart) are read together; a trait in two of them is refused, so no trait
is counted twice. An LDpred2-auto aggregate line (benchmarks/bench_real/aou_ldpred2.py's ``--out``) fills a trait whose
run had no paired LDpred2 scores, marked "unpaired", only where its held-out people (and a disease's cases) number the
run's: its partial r^2 is then on the same people and scale, but with no interval and no paired difference.

Outputs, in the output directory:
- ``aou_summary.tsv`` (SUMMARY_COLUMNS): one row per trait.
- ``aou_ancestry.tsv`` (ANCESTRY_COLUMNS): one row per trait and ancestry group shown.
- ``aou_meta.tsv`` (META_COLUMNS): each paired difference combined over the traits whose single test fold gives it a
  p-value, overall and per ancestry group. Where one run holds every such trait, its own meta-analysis is copied;
  traits from runs fitted apart are combined here (``held_out_comparison.meta_analysis``) with each run's traits'
  covariance from its own bootstrap resamples (trait_covariance.tsv) and the runs as independent, which their shared
  people make optimistic; the covariance column says which.
- ``fig_aou_diseases``: each trait's held-out partial r^2 for SV-PGS with and without SVs and for LDpred2-auto, and the
  paired gain from SVs (within SV-PGS, and stacked on top of LDpred2-auto) with its meta-analysis (the poster's
  ph_aou_diseases slot).
- ``fig_aou_effects``: for one trait (``--trait``, hypertension by default), the learned prior variance of each SV class
  relative to SNVs beside the posterior's, and the prior by SV length (the poster's ph_aou_hypertension slot).
Each figure is an SVG in figures.py's units (hundredths of an inch at its printed size) and, where figures.py's Chrome
is installed, its PNG; a figure without the data it needs is not drawn, and its previous files are removed. The default
output directory is the poster's figures directory, where build_pptx.py places the PNGs in their slots; git ignores
these outputs there (.gitignore), so no All of Us result enters the repository before it is released.

    python -m docs.poster.ashg2026.aou_results EXPORT_DIR [EXPORT_DIR ...] [--ldpred2 ldpred2.jsonl ...] [--trait NAME] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

import numpy as np

try:
    from docs.poster.ashg2026 import figures
except ImportError:  # run as a script from this directory, as figures.py is
    import figures  # type: ignore[no-redef]

from sv_pgs.held_out_comparison import meta_analysis

ARMS = ("snv_indel", "with_sv")
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
CONFIDENCE_QUANTILE = 1.959963984540054
"""The two-sided 95% normal quantile, for an interval combined here across runs."""
SUMMARY_COLUMNS = (
    "trait", "n", "cases",
    "snv_indel_partial_r2", "snv_indel_ci_low", "snv_indel_ci_high",
    "with_sv_partial_r2", "with_sv_ci_low", "with_sv_ci_high",
    "ldpred2_partial_r2", "ldpred2_ci_low", "ldpred2_ci_high", "ldpred2_source",
    "stacked_partial_r2", "stacked_ci_low", "stacked_ci_high",
    *(f"{prefix}{suffix}" for _candidate, _baseline, prefix in PAIRS for suffix in ("", "_ci_low", "_ci_high", "_p")),
    "sv_gain_relative", "delta_auc", "delta_auc_p", "sv_prior_enrichment", "sv_posterior_enrichment", "sv_posterior_enrichment_ci_low",
    "sv_posterior_enrichment_ci_high", "sv_variance_share", "sv_variance_share_ci_low", "sv_variance_share_ci_high", "sv_variant_share",
    "sv_heritability_enrichment", "sv_heritability_enrichment_ci_low", "sv_heritability_enrichment_ci_high", "ancestry_heterogeneity_p",
    "inference", "source",
)
"""``aou_summary.tsv``: per trait its held-out people (n, a disease's cases), each method's partial r^2 with its 95%
bootstrap interval (``stacked``: LDpred2-auto stacked with SV-PGS), each paired difference of PAIRS (candidate less
baseline on the same people and resamples) with its interval and two-sided p, a disease's paired AUC difference, the
with_sv model's prior and posterior enrichment of every structural record over SNVs, their share of its genetic
variance against their share of its records and the ratio (the heritability enrichment), the p of the SV gain's
difference between ancestry groups, the inference the run's single or several test folds allow, and the export
directory."""
META_COLUMNS = (
    "candidate", "baseline", "cell", "level", "model", "traits", "estimate", "standard_error", "ci_low", "ci_high", "p_value",
    "heterogeneity_p", "i_squared", "positive_traits", "covariance",
)
"""``aou_meta.tsv``: per pair, cell (all held-out people, each ancestry group) and model (fixed, random), the combined
difference over the named traits."""
ANCESTRY_COLUMNS = (
    "trait", "level", "n", "cases", *(f"{prefix}{suffix}" for _candidate, _baseline, prefix in PAIRS for suffix in ("", "_ci_low", "_ci_high", "_p")),
)
"""``aou_ancestry.tsv``: per trait and ancestry group shown, its held-out people and each paired difference of PAIRS."""
TRAIT_LABELS = {
    "hypertension": "Hypertension", "coronary_artery_disease": "CAD", "asthma": "Asthma", "hyperlipidemia": "Hyperlipidemia",
    "breast_cancer": "Breast cancer", "type2_diabetes": "T2D", "atrial_fibrillation": "AF", "hypothyroidism": "Hypothyroidism",
    "copd": "COPD", "depression": "Depression", "gout": "Gout", "cataract": "Cataract", "chronic_kidney_disease": "CKD",
    "psoriasis": "Psoriasis", "prostate_cancer": "Prostate cancer", "egfr_ckd_epi_2021": "eGFR", "ldl_cholesterol": "LDL",
    "body_mass_index": "BMI", "systolic_blood_pressure": "Systolic BP", "hemoglobin_a1c": "HbA1c", "combined": "All traits",
}
CLASS_LABELS = {
    "deletion": "Deletion", "insertion": "Insertion", "duplication": "Duplication", "insertion_mei": "Mobile element",
    "inversion_bnd_complex": "Inversion/BND", "str_vntr_repeat": "STR/VNTR", "other_complex_sv": "Complex",
    "copy_number": "Copy number", "inversion": "Inversion", "all_structural": "All SVs", "snv": "SNV",
}
CLASS_COLOURS = {"deletion": figures.NAVY, "insertion": figures.TEAL, "duplication": figures.AMBER, "insertion_mei": figures.PURPLE}
"""The length panel's series, each class always in its own colour; the other classes are in the class panel alone."""
METHODS = (
    # (method, summary prefix, key label, colour, filled marker, horizontal offset)
    (BASELINE, "ldpred2", "LDpred2-auto", figures.AMBER, True, -18.0),
    ("snv_indel", "snv_indel", "SV-PGS without SVs", figures.NAVY, False, 0.0),
    ("with_sv", "with_sv", "SV-PGS with SVs", figures.TEAL, True, 18.0),
)
"""The partial r^2 panel's methods; the stacked scores are in the summary and, as gains, in the gain panel."""
GAINS = (
    # (summary prefix, key label, colour, horizontal offset)
    ("sv_gain", "within SV-PGS", figures.TEAL, -12.0),
    ("stack_sv_gain", "on top of LDpred2-auto", figures.PURPLE, 12.0),
)
"""The gain panel's two measures of what SVs add, each per trait and combined over the traits: with_sv less snv_indel,
and LDpred2-auto stacked with SV-PGS's structural part less LDpred2-auto alone."""
TABLES = (
    "partial_r2", "paired_differences", "meta_analysis", "trait_covariance", "heterogeneity", "comparison", "prior_enrichment", "prior_by_length",
)


def number(text: object) -> float:
    """A table's value, or NaN where it is blank or suppressed."""
    try:
        return float(text)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan


def text(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}g}" if math.isfinite(value) else ""


def read_table(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def label(name: str) -> str:
    return TRAIT_LABELS.get(name, name.replace("_", " ").capitalize())


def load(directories: list[Path]) -> dict:
    """Every export directory's tables, each row tagged with its directory; a trait in two directories is refused."""
    tables: dict[str, list[dict[str, str]]] = {name: [] for name in TABLES}
    owner: dict[str, str] = {}
    for directory in directories:
        rows = {name: read_table(directory / f"{name}.tsv") for name in tables}
        for trait in {row["trait"] for row in rows["partial_r2"]}:
            if trait in owner:
                raise ValueError(f"{trait} is in both {owner[trait]} and {directory}; give each trait's run once.")
            owner[trait] = str(directory)
        for name, entries in rows.items():
            tables[name] += [{**row, "source": str(directory)} for row in entries]
    return tables


def unpaired_ldpred2(paths: list[Path]) -> dict[str, dict]:
    """The LDpred2-auto aggregate lines by trait (aou_ldpred2.py ``--out``)."""
    lines = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                lines[entry["trait"]] = entry
    return lines


def paired_columns(row: dict[str, str], paired: dict, trait: str, cell: str, level: str) -> None:
    """Each PAIRS difference of one trait's cell into ``row``: estimate, interval and p."""
    for candidate, baseline, prefix in PAIRS:
        entry = paired.get((trait, candidate, baseline, cell, level), {})
        row |= {prefix: entry.get("estimate", ""), f"{prefix}_ci_low": entry.get("ci_low", ""), f"{prefix}_ci_high": entry.get("ci_high", ""),
                f"{prefix}_p": entry.get("p_value", "")}


def summary_rows(tables: dict, ldpred2: dict[str, dict]) -> list[dict[str, str]]:
    """One row per trait (SUMMARY_COLUMNS), from the all-held-out-rows cells, in the order the runs list them. An
    LDpred2-auto aggregate line fills a trait without paired LDpred2 scores only when its held-out people (and a
    disease's cases) are the trait's as the run shows them, marked unpaired."""
    partial = {(row["trait"], row["method"]): row for row in tables["partial_r2"] if row["cell"] == "all"}
    paired = {(row["trait"], row["candidate"], row["baseline"], row["cell"], row["level"]): row for row in tables["paired_differences"]}
    auc = {row["trait"]: row for row in tables["comparison"] if row["cell"] == "all" and row["metric"] == "delta_auc"}
    enrichment = {row["trait"]: row for row in tables["prior_enrichment"] if row["arm"] == "with_sv" and row["variant_class"] == "all_structural"}
    ancestry = {row["trait"]: row for row in tables["heterogeneity"] if (row["candidate"], row["baseline"], row["cell"]) == ("with_sv", "snv_indel", "ancestry")}
    traits = list(dict.fromkeys(row["trait"] for row in tables["partial_r2"]))
    rows = []
    for trait in traits:
        reference = partial.get((trait, "with_sv")) or partial[(trait, "snv_indel")]
        row = {"trait": trait, "n": reference["n"], "cases": reference["cases"], "inference": reference["inference"], "source": reference["source"]}
        for method, prefix in (("snv_indel", "snv_indel"), ("with_sv", "with_sv"), (BASELINE, "ldpred2"), (STACKED, "stacked")):
            entry = partial.get((trait, method), {})
            row |= {f"{prefix}_partial_r2": entry.get("partial_r2", ""), f"{prefix}_ci_low": entry.get("ci_low", ""), f"{prefix}_ci_high": entry.get("ci_high", "")}
        row["ldpred2_source"] = "paired" if (trait, BASELINE) in partial else ""
        line = ldpred2.get(trait, {})
        same_people = str(line.get("held_out_rows", "")) == row["n"] and str(line.get("cases", "")) == row["cases"]
        if not row["ldpred2_partial_r2"] and line.get("pgs_partial_r2") is not None and same_people:
            row["ldpred2_partial_r2"], row["ldpred2_source"] = text(float(line["pgs_partial_r2"])), "unpaired"
        paired_columns(row, paired, trait, "all", "all")
        row["sv_gain_relative"] = paired.get((trait, "with_sv", "snv_indel", "all", "all"), {}).get("relative_estimate", "")
        row |= {"delta_auc": auc.get(trait, {}).get("estimate", ""), "delta_auc_p": auc.get(trait, {}).get("p_value", "")}
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
            "ancestry_heterogeneity_p": ancestry.get(trait, {}).get("p_value", ""),
        }
        rows.append(row)
    return rows


def ancestry_rows(tables: dict) -> list[dict[str, str]]:
    """One row per trait and ancestry group shown (ANCESTRY_COLUMNS)."""
    paired = {(row["trait"], row["candidate"], row["baseline"], row["cell"], row["level"]): row for row in tables["paired_differences"]}
    rows = []
    for entry in tables["partial_r2"]:
        if entry["cell"] == "ancestry" and entry["method"] == "with_sv":
            row = {"trait": entry["trait"], "level": entry["level"], "n": entry["n"], "cases": entry["cases"]}
            paired_columns(row, paired, entry["trait"], "ancestry", entry["level"])
            rows.append(row)
    return rows


def meta_rows(tables: dict) -> list[dict[str, str]]:
    """Each pair's combined difference per cell (META_COLUMNS): a run's own meta-analysis where it holds every tested
    trait; else the traits of the runs combined here, each run's traits with their joint-bootstrap covariance
    (trait_covariance.tsv) and the runs taken as independent of each other (their shared people make that optimistic);
    a run without its covariance table counts its traits as independent too, and the covariance column says which."""
    tested: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in tables["paired_differences"]:
        if row["p_value"] and number(row["standard_error"]) > 0.0:
            tested.setdefault((row["candidate"], row["baseline"], row["cell"], row["level"]), []).append(row)
    covariances = {
        (row["source"], row["candidate"], row["baseline"], row["cell"], row["level"], *sorted((row["trait"], row["other_trait"]))): number(row["covariance"])
        for row in tables["trait_covariance"]
    }
    rows = []
    for (candidate, baseline, cell, level), entries in tested.items():
        traits = sorted(entry["trait"] for entry in entries)
        if len(traits) < 2:
            continue
        own = [row for row in tables["meta_analysis"] if (row["candidate"], row["baseline"], row["cell"], row["level"]) == (candidate, baseline, cell, level)
               and sorted(row["traits"].split(";")) == traits]
        if own:
            rows += [{column: row.get(column, "") for column in META_COLUMNS} for row in own]
            continue
        estimates = np.array([number(entry["estimate"]) for entry in entries])
        covariance = np.diag([number(entry["standard_error"]) ** 2 for entry in entries])
        within = True
        for first, one in enumerate(entries):
            for second, other in enumerate(entries):
                if first != second and one["source"] == other["source"]:
                    value = covariances.get((one["source"], candidate, baseline, cell, level, *sorted((one["trait"], other["trait"]))))
                    within &= value is not None
                    covariance[first, second] = 0.0 if value is None else value
        result = meta_analysis(estimates, covariance)
        for model, combined in (("fixed", result.fixed), ("random", result.random)):
            rows.append({
                "candidate": candidate, "baseline": baseline, "cell": cell, "level": level, "model": model,
                "traits": ";".join(entry["trait"] for entry in entries), "estimate": text(combined.estimate), "standard_error": text(combined.standard_error),
                "ci_low": text(combined.estimate - CONFIDENCE_QUANTILE * combined.standard_error),
                "ci_high": text(combined.estimate + CONFIDENCE_QUANTILE * combined.standard_error),
                "p_value": text(math.erfc(abs(combined.z_score) / math.sqrt(2.0))), "heterogeneity_p": text(result.heterogeneity_p),
                "i_squared": text(result.i_squared), "positive_traits": str(result.positive),
                "covariance": "within_run_joint_bootstrap_runs_independent" if within else "independent_traits_assumed",
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


def trait_labels(parts: list[str], names: list[str], x, bottom: float) -> None:
    for name in names:
        parts.append(f'<text class="row" transform="translate({x(name) + 8:.1f} {bottom + 22:.1f}) rotate(-30)" text-anchor="end">{html.escape(label(name))}</text>')


def diseases_figure(summary: list[dict[str, str]], meta: list[dict[str, str]], width: float = 15.40, height: float = 4.60) -> str | None:
    """Two panels over one trait axis, the traits ordered by the with_sv partial r^2: above, each method's held-out
    partial r^2 (log scale) with its 95% bootstrap interval; below, what SVs add on the same held-out people (GAINS: to
    SV-PGS, and to LDpred2-auto when SV-PGS's structural part is stacked with it) with its interval, then the traits
    combined (fixed effect, a diamond) where a meta-analysis exists. An unpaired LDpred2 value is an open square without
    an interval."""
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
    key_y, top, middle, lower_top, bottom = 30.0, 58.0, 196.0, 240.0, h - 144.0
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
    trait_labels(parts, columns, x, bottom)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("exports", nargs="+", type=Path, help="Export directories of SV-PGS runs (their staged aggregate tables).")
    parser.add_argument("--ldpred2", nargs="*", type=Path, default=[], help="LDpred2-auto aggregate lines (aou_ldpred2.py --out) for traits without paired scores.")
    parser.add_argument("--trait", default="hypertension", help="The trait whose SV effects the effects figure shows.")
    parser.add_argument("--out", type=Path, default=figures.OUT, help="The output directory.")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    tables = load(args.exports)
    summary = summary_rows(tables, unpaired_ldpred2(args.ldpred2))
    meta = meta_rows(tables)
    made = [
        write_table(args.out / "aou_summary.tsv", SUMMARY_COLUMNS, summary), write_table(args.out / "aou_ancestry.tsv", ANCESTRY_COLUMNS, ancestry_rows(tables)),
        write_table(args.out / "aou_meta.tsv", META_COLUMNS, meta),
    ]
    made += save("fig_aou_diseases", diseases_figure(summary, meta), 15.40, 4.60, args.out)
    made += save("fig_aou_effects", effects_figure(tables, args.trait), 15.40, 5.00, args.out)
    for path in made:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
