"""The poster's All of Us summary reads the tables the workspace pipeline exports, and combines runs fitted apart."""
from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.bench_real.aou_compare import COLUMNS as COMPARE_COLUMNS
from docs.poster.ashg2026 import aou_results
from sv_pgs import workspace_pipeline
from sv_pgs.draw_laws import ProductMixtureDraws
from sv_pgs.held_out_comparison import meta_analysis


def _traits(generator: np.random.Generator, names: tuple[str, ...], rows: int = 1200, baseline: bool = True) -> tuple[list, np.ndarray, np.ndarray, list[str]]:
    """Quantitative traits on one cohort's rows, fold 0 held out, in two ancestry groups: with_sv adds a structural part
    snv_indel misses, LDpred2-auto sees less of the shared part, and each stack (``stacked_scores``) adds to it."""
    families = np.arange(rows, dtype=np.int64)
    folds = families % 2
    ancestry = ["afr" if row % 3 == 0 else "eur" for row in range(rows)]
    covariates = np.column_stack([np.ones(rows), generator.normal(size=(rows, 2))])
    held = folds == 0
    traits = []
    for name in names:
        small, structural = generator.normal(size=rows), generator.normal(size=rows)
        target = np.where(held, small + 0.6 * structural + 1.5 * generator.normal(size=rows), np.nan)
        snv_indel = small + generator.normal(size=rows)
        scores = {"snv_indel": snv_indel, "with_sv": snv_indel + 0.6 * structural}
        if baseline:
            scores["ldpred2_auto"] = small + 1.5 * generator.normal(size=rows)
        scores = {method: np.where(held, score, np.nan) for method, score in scores.items()}
        scores |= workspace_pipeline.stacked_scores(target, covariates, scores, np.where(held, 0.6 * structural, np.nan), families, folds, (0,), 7)
        traits.append(workspace_pipeline.HeldOutTrait(name, False, target, covariates, scores, (0,)))
    return traits, families, folds, ancestry


def _write(path: Path, lines) -> None:
    with path.open("w", newline="") as handle:
        csv.writer(handle, delimiter="\t").writerows(lines)


def _export(directory: Path, names: tuple[str, ...], seed: int, baseline: bool = True) -> list:
    """An export directory of the report's held-out tables for ``names``, each through the export check."""
    directory.mkdir(parents=True)
    traits, families, folds, ancestry = _traits(np.random.default_rng(seed), names, baseline=baseline)
    tables, _incomplete = workspace_pipeline.partial_r2_tables(traits, families, folds, ancestry, seed, 60, 1 << 26)
    for name, lines in tables.items():
        _write(directory / name, lines)
        workspace_pipeline.check_exportable(directory / name)
    return traits


def _prior_tables(directory: Path, trait: str) -> dict:
    """The effects step's prior tables for one with_sv model: SNVs, deletions growing with length, and insertions."""
    generator = np.random.default_rng(9)
    names = ["snv"] * 200 + ["deletion"] * 120 + ["insertion"] * 60
    lengths = np.r_[np.ones(200), np.repeat([80.0, 800.0, 8000.0], 40), np.repeat([70.0, 700.0], 30)]
    classes = np.array(names, dtype=object)
    legend = ["snv", "deletion", "insertion"]
    index = np.array([legend.index(name) for name in names])
    count = index.shape[0]
    law = ProductMixtureDraws(
        class_index=index, log_scale=0.4 * np.log(lengths), log_density=np.log(np.full((3, 4), 0.25)), log_variance_grid=np.linspace(-10.0, -4.0, 4),
        shift=np.zeros((1, count)), omega=np.zeros((1, count)), weights=np.ones(1), key=(1, 2), draw_count=4,
    )
    unit = np.ones(count)
    prior = workspace_pipeline.prior_moments(SimpleNamespace(posterior_draws=law), unit)
    effects = workspace_pipeline.VariantEffects(np.arange(count, dtype=np.int64), generator.normal(0.0, 0.01, count), np.full(count, 0.01), np.full(count, 0.5), np.full(count, 0.5))
    frequency = np.linspace(0.01, 0.5, count)
    lines = workspace_pipeline.effect_summaries(trait, "with_sv", classes, workspace_pipeline.length_bins(lengths), lengths, effects, unit, prior, frequency)
    headers = {"prior_by_length.tsv": workspace_pipeline.PRIOR_BY_LENGTH_COLUMNS, "prior_enrichment.tsv": workspace_pipeline.PRIOR_ENRICHMENT_COLUMNS}
    for name, header in headers.items():
        _write(directory / name, [header, *lines[name]])
        workspace_pipeline.check_exportable(directory / name)
    return lines


def _read(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_the_summary_has_one_row_per_trait_from_the_exported_tables(tmp_path: Path) -> None:
    first = tmp_path / "run_a" / "export"
    _export(first, ("hypertension", "asthma"), 3)
    _prior_tables(first, "hypertension")
    second = tmp_path / "run_b" / "export"
    _export(second, ("gout", "copd"), 4, baseline=False)
    ldpred2 = tmp_path / "ldpred2.jsonl"
    ldpred2.write_text("".join(json.dumps(line) + "\n" for line in (
        {"trait": "gout", "held_out_rows": 600, "variants": 1000, "pgs_partial_r2": 0.0123},
        {"trait": "copd", "held_out_rows": 599, "variants": 1000, "pgs_partial_r2": 0.0456},
    )))
    out = tmp_path / "poster"
    assert aou_results.main([str(first), str(second), "--ldpred2", str(ldpred2), "--out", str(out)]) == 0

    summary = {row["trait"]: row for row in _read(out / "aou_summary.tsv")}
    assert list(summary) == ["hypertension", "asthma", "gout", "copd"] and set(_read(out / "aou_summary.tsv")[0]) == set(aou_results.SUMMARY_COLUMNS)
    assert all(row["status"] == "" and row["partial_r2_interval"] == "bootstrap" for row in summary.values()) and _read(out / "aou_loci.tsv") == []
    partial = {(row["trait"], row["method"]): row for row in _read(first / "partial_r2.tsv") if row["cell"] == "all"}
    paired = {(row["trait"], row["candidate"], row["baseline"]): row for row in _read(first / "paired_differences.tsv") if row["cell"] == "all"}
    row = summary["hypertension"]
    assert row["n"] == partial["hypertension", "with_sv"]["n"] == "600"
    assert row["with_sv_partial_r2"] == partial["hypertension", "with_sv"]["partial_r2"] and row["ldpred2_ci_high"] == partial["hypertension", "ldpred2_auto"]["ci_high"]
    assert row["sv_gain"] == paired["hypertension", "with_sv", "snv_indel"]["estimate"] and row["sv_gain_p"] == paired["hypertension", "with_sv", "snv_indel"]["p_value"]
    assert row["with_sv_vs_ldpred2"] == paired["hypertension", "with_sv", "ldpred2_auto"]["estimate"] and row["ldpred2_source"] == "paired"
    assert float(row["sv_gain"]) > 0.0 and float(row["sv_prior_enrichment"]) > 0.0 and row["source"] == str(first)
    # LDpred2-auto stacked with SV-PGS, and with its structural part alone: each gains over LDpred2-auto on the same people.
    assert row["stacked_partial_r2"] == partial["hypertension", "ldpred2_auto+with_sv"]["partial_r2"]
    assert row["stack_gain"] == paired["hypertension", "ldpred2_auto+with_sv", "ldpred2_auto"]["estimate"] and float(row["stack_gain"]) > 0.0
    assert row["stack_sv_gain"] == paired["hypertension", "ldpred2_auto+structural", "ldpred2_auto"]["estimate"] and float(row["stack_sv_gain"]) > 0.0
    heterogeneity = {(entry["trait"], entry["candidate"], entry["baseline"], entry["cell"]): entry for entry in _read(first / "heterogeneity.tsv")}
    assert row["ancestry_heterogeneity_p"] == heterogeneity["hypertension", "with_sv", "snv_indel", "ancestry"]["p_value"] != ""
    # gout's run had no paired LDpred2 scores: its aggregate fills the partial r^2 alone, marked unpaired, and only
    # where its held-out people number the run's (copd's line counts other people).
    assert summary["gout"]["ldpred2_partial_r2"] == "0.0123" and summary["gout"]["ldpred2_source"] == "unpaired"
    assert summary["gout"]["ldpred2_ci_low"] == "" and summary["gout"]["with_sv_vs_ldpred2"] == "" and summary["asthma"]["sv_prior_enrichment"] == ""
    assert summary["copd"]["ldpred2_partial_r2"] == summary["copd"]["ldpred2_source"] == summary["copd"]["stack_gain"] == ""

    # Each trait's ancestry groups, with their paired differences.
    ancestry = {(entry["trait"], entry["level"]): entry for entry in _read(out / "aou_ancestry.tsv")}
    groups = {(entry["trait"], entry["level"]): entry for entry in _read(first / "paired_differences.tsv")
              if entry["cell"] == "ancestry" and (entry["candidate"], entry["baseline"]) == ("with_sv", "snv_indel")}
    assert set(ancestry) == {(trait, level) for trait in summary for level in ("afr", "eur")}
    assert ancestry["asthma", "afr"]["sv_gain"] == groups["asthma", "afr"]["estimate"] and ancestry["asthma", "afr"]["n"] == "200"

    # The SV gain over the four traits: two runs, each run's traits with their joint-bootstrap covariance, the runs apart.
    meta = {(row["candidate"], row["baseline"], row["cell"], row["level"], row["model"]): row for row in _read(out / "aou_meta.tsv")}
    fixed = meta["with_sv", "snv_indel", "all", "all", "fixed"]
    assert fixed["covariance"] == "within_run_joint_bootstrap_runs_independent" and fixed["traits"] == "hypertension;asthma;gout;copd"
    entries = [(directory, entry) for directory in (first, second) for entry in _read(directory / "paired_differences.tsv")
               if entry["cell"] == "all" and (entry["candidate"], entry["baseline"]) == ("with_sv", "snv_indel")]
    covariance = np.diag([float(entry["standard_error"]) ** 2 for _directory, entry in entries])
    for directory, block in ((first, (0, 1)), (second, (2, 3))):
        off = next(float(entry["covariance"]) for entry in _read(directory / "trait_covariance.tsv")
                   if (entry["candidate"], entry["baseline"], entry["cell"]) == ("with_sv", "snv_indel", "all") and entry["trait"] != entry["other_trait"])
        covariance[block[0], block[1]] = covariance[block[1], block[0]] = off
    expected = meta_analysis(np.array([float(entry["estimate"]) for _directory, entry in entries]), covariance)
    assert float(fixed["estimate"]) == pytest.approx(expected.fixed.estimate, rel=1e-5)
    assert float(fixed["standard_error"]) == pytest.approx(expected.fixed.standard_error, rel=1e-5)
    assert ("with_sv", "snv_indel", "ancestry", "afr", "fixed") in meta
    # The LDpred2 comparisons are in one run only: its own meta-analysis, with the traits' covariance, is copied.
    own = next(row for row in _read(first / "meta_analysis.tsv")
               if (row["candidate"], row["baseline"], row["cell"], row["model"]) == ("ldpred2_auto+structural", "ldpred2_auto", "all", "fixed"))
    copied = meta["ldpred2_auto+structural", "ldpred2_auto", "all", "all", "fixed"]
    assert copied["estimate"] == own["estimate"] and copied["covariance"] == "joint_bootstrap" and copied["positive_traits"] == "2"

    # Both figures are drawn as SVG; PNGs need figures.py's Chrome.
    diseases = (out / "fig_aou_diseases.svg").read_text()
    assert all(label in diseases for label in ("Hypertension", "Asthma", "Gout", "All traits", "LDpred2-auto", "on top of LDpred2-auto"))
    assert "Deletion" in (out / "fig_aou_effects.svg").read_text()


def test_a_trait_in_two_runs_is_refused_and_a_figure_without_data_is_removed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    _export(first, ("asthma",), 5)
    _export(second, ("asthma",), 6)
    out = tmp_path / "out"
    with pytest.raises(SystemExit):
        aou_results.main([str(first), str(second), "--out", str(out)])
    assert "asthma is in both" in capsys.readouterr().err and not out.exists()
    out.mkdir()
    (out / "fig_aou_effects.svg").write_text("<svg/>")
    assert aou_results.main([str(first), "--trait", "hypertension", "--out", str(out)]) == 0
    assert not (out / "fig_aou_effects.svg").exists() and (out / "fig_aou_diseases.svg").is_file()
    # One trait: no meta-analysis.
    assert _read(out / "aou_meta.tsv") == []



def _compare(directory: Path, lines) -> None:
    _write(directory / "compare.tsv", [COMPARE_COLUMNS, *lines])
    workspace_pipeline.check_exportable(directory / "compare.tsv")


def test_a_flagged_trait_leaves_the_meta_analysis_and_compare_tsv_fills_ldpred2_only_on_the_reports_people(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = tmp_path / "run" / "export"
    _export(run, ("hypertension", "asthma", "gout"), 7, baseline=False)
    _write(run / "uncertified.tsv", [workspace_pipeline.UNCERTIFIED_COLUMNS, ("gout", "with_sv (fold 0: not certified)")])
    partial = {(row["trait"], row["method"]): row for row in _read(run / "partial_r2.tsv") if row["cell"] == "all"}

    def arms(trait: str, scale: float = 1.0) -> list[tuple[str, ...]]:
        return [("", trait, arm, partial[trait, arm]["n"], "", "pgs_partial_r2", aou_results.text(float(partial[trait, arm]["partial_r2"]) * scale), "0.01", "", "")
                for arm in ("snv_indel", "with_sv")]

    # hypertension's compare.tsv arms are the report's; asthma's partial r^2 is another's; gout's LDpred2 is a region's.
    _compare(run, [
        *arms("hypertension"), ("", "hypertension", "ldpred2", "600", "", "pgs_partial_r2", "0.05", "0.01", "0.03", "0.07"),
        ("", "hypertension", "with_sv - ldpred2", "600", "", "delta_pgs_partial_r2", "0.02", "0.008", "0.004", "0.036"),
        ("", "hypertension", "with_sv - snv_indel", "600", "", "delta_pgs_partial_r2", "0.5", "0.1", "0.3", "0.7"),
        *arms("asthma", 1.1), ("", "asthma", "ldpred2", "600", "", "pgs_partial_r2", "0.05", "0.01", "0.03", "0.07"),
        ("chr22", "gout", "ldpred2", "600", "", "pgs_partial_r2", "0.04", "0.01", "0.02", "0.06"),
    ])
    (run / "ldpred2.jsonl").write_text("".join(json.dumps(line) + "\n" for line in (
        {"trait": "gout", "held_out_rows": 600, "variants": 10, "pgs_partial_r2": 0.02},
        {"region": "chr22", "trait": "gout", "held_out_rows": 600, "variants": 10, "pgs_partial_r2": 0.03},
    )))
    out = tmp_path / "out"
    assert aou_results.main([str(run), "--out", str(out)]) == 0
    assert "compare.tsv's arms of asthma" in capsys.readouterr().err
    summary = {row["trait"]: row for row in _read(out / "aou_summary.tsv")}
    paired = {(row["trait"], row["candidate"], row["baseline"]): row for row in _read(run / "paired_differences.tsv") if row["cell"] == "all"}
    row = summary["hypertension"]
    assert (row["ldpred2_partial_r2"], row["ldpred2_ci_low"], row["ldpred2_ci_high"], row["ldpred2_source"]) == ("0.05", "0.03", "0.07", "paired_compare")
    # Its one test fold gives compare.tsv's difference a p-value; the report's bootstrap SV gain is kept over compare.tsv's.
    assert (row["with_sv_vs_ldpred2"], row["with_sv_vs_ldpred2_ci_low"]) == ("0.02", "0.004")
    assert float(row["with_sv_vs_ldpred2_p"]) == pytest.approx(math.erfc(2.5 / math.sqrt(2.0)), rel=1e-5)
    assert row["sv_gain"] == paired["hypertension", "with_sv", "snv_indel"]["estimate"]
    assert summary["asthma"]["ldpred2_source"] == "" and summary["asthma"]["with_sv_vs_ldpred2"] == ""
    # gout is flagged, keeps its row, and takes the whole genome's LDpred2 line, not the region's.
    assert summary["gout"]["status"] == "with_sv uncertified" and (summary["gout"]["ldpred2_partial_r2"], summary["gout"]["ldpred2_source"]) == ("0.02", "unpaired")
    assert "Gout" + aou_results.STATUS_MARK in (out / "fig_aou_diseases.svg").read_text()

    # The SV gain over the unflagged traits only: not the run's own meta-analysis, which holds gout too.
    meta = {(entry["candidate"], entry["baseline"], entry["cell"], entry["level"], entry["model"]): entry for entry in _read(out / "aou_meta.tsv")}
    fixed = meta["with_sv", "snv_indel", "all", "all", "fixed"]
    assert fixed["traits"] == "hypertension;asthma" and fixed["covariance"] == "within_run_joint_bootstrap_runs_independent"
    entries = [paired[trait, "with_sv", "snv_indel"] for trait in ("hypertension", "asthma")]
    off = next(float(entry["covariance"]) for entry in _read(run / "trait_covariance.tsv")
               if (entry["candidate"], entry["baseline"], entry["cell"]) == ("with_sv", "snv_indel", "all") and {entry["trait"], entry["other_trait"]} == {"hypertension", "asthma"})
    covariance = np.array([[float(entries[0]["standard_error"]) ** 2, off], [off, float(entries[1]["standard_error"]) ** 2]])
    expected = meta_analysis(np.array([float(entry["estimate"]) for entry in entries]), covariance)
    assert float(fixed["estimate"]) == pytest.approx(expected.fixed.estimate, rel=1e-5)
    # One trait has a paired LDpred2 difference: nothing to combine.
    assert not any(key[1] == "ldpred2_auto" for key in meta)

    # The chr22 region: its compare.tsv rows lack the arms, so its LDpred2 is the region's ldpred2.jsonl line alone.
    assert aou_results.main([str(run), "--region", "chr22", "--out", str(out)]) == 0
    assert "compare.tsv's arms of gout" in capsys.readouterr().err
    summary = {row["trait"]: row for row in _read(out / "aou_summary.tsv")}
    assert summary["hypertension"]["ldpred2_source"] == "" and summary["gout"]["ldpred2_partial_r2"] == "0.03"


def test_an_input_the_export_rule_would_not_show_is_refused_whole_without_its_values(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = tmp_path / "base"
    _export(base, ("asthma",), 11)
    accuracy = workspace_pipeline.ACCURACY_COLUMNS

    def counted(directory: Path) -> None:
        lines = [line.split("\t") for line in (directory / "partial_r2.tsv").read_text().splitlines()]
        level = next(index for index, line in enumerate(lines) if line[2] == "ancestry")
        lines[level][4] = "17"
        _write(directory / "partial_r2.tsv", lines)

    def extra_column(directory: Path) -> None:
        lines = [line.split("\t") for line in (directory / "partial_r2.tsv").read_text().splitlines()]
        _write(directory / "partial_r2.tsv", [[*line, "research_id" if index == 0 else "8675309"] for index, line in enumerate(lines)])

    plants = {
        "a count below 21": counted,
        "control count below 21": lambda directory: _write(directory / "accuracy.tsv", [accuracy, ("asthma", "with_sv", "all", "all", "100", "90", "auc", "0.6173")]),
        "without its case count": lambda directory: _write(directory / "accuracy.tsv", [accuracy, ("asthma", "with_sv", "all", "all", "600", "", "auc", "0.6173")]),
        "its columns are not": extra_column,
        "refused by the export rule": lambda directory: (directory / "ldpred2.jsonl").write_text(json.dumps({"trait": "asthma", "held_out_rows": 17, "variants": 10, "pgs_partial_r2": 0.6173}) + "\n"),
        "keys are not": lambda directory: (directory / "ldpred2.jsonl").write_text(json.dumps({"trait": "asthma", "held_out_rows": 600, "variants": 10, "pgs_partial_r2": 0.6173, "research_ids": ["8675309"]}) + "\n"),
        "two different LDpred2 lines": lambda directory: (directory / "ldpred2.jsonl").write_text("".join(json.dumps({"trait": "asthma", "held_out_rows": 600, "variants": 10, "pgs_partial_r2": value}) + "\n" for value in (0.6173, 0.5))),
    }
    for reason, plant in plants.items():
        run = tmp_path / reason.replace(" ", "_")
        shutil.copytree(base, run)
        plant(run)
        out = tmp_path / f"{run.name}_out"
        with pytest.raises(SystemExit):
            aou_results.main([str(run), "--out", str(out)])
        error = capsys.readouterr().err.replace(str(run), "")
        assert reason in error and not out.exists(), reason
        assert not any(value in error for value in ("17", "0.6173", "8675309")), reason


def test_the_locus_table_splits_a_window_between_snvs_short_indels_and_svs(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _export(run, ("hypertension",), 8)
    loci = [{"name": "LPA", "kind": "sv_known", "chromosome": 6, "start": 1000, "end": 2000, "traits": ["hypertension"]}]
    positions = np.array([1100, 1200, 1300, 1400, 1500, 5000])
    classes = np.array(["snv", "short_indel", "deletion", "deletion", "snv", "snv"], dtype=object)
    second = np.array([0.1, 0.05, 0.2, 0.3, 0.02, 0.5])
    carriers = np.array([100.0, 100.0, 100.0, 100.0, 5.0, 100.0])
    lines = workspace_pipeline.locus_variance("hypertension", "with_sv", loci, np.full(6, 6), positions, classes, second, carriers)
    small = classes != "deletion"
    lines += workspace_pipeline.locus_variance("hypertension", "snv_indel", loci, np.full(4, 6), positions[small], classes[small], second[small], carriers[small])
    _write(run / "locus_variance.tsv", [workspace_pipeline.LOCUS_VARIANCE_COLUMNS, *lines])
    workspace_pipeline.check_exportable(run / "locus_variance.tsv")
    out = tmp_path / "out"
    assert aou_results.main([str(run), "--out", str(out)]) == 0
    (row,) = _read(out / "aou_loci.tsv")
    table = {(entry["arm"], entry["variant_class"]): entry for entry in _read(run / "locus_variance.tsv")}
    assert (row["trait"], row["locus"], row["kind"], row["top_sv_class"], row["rare_left_out"]) == ("hypertension", "LPA", "sv_known", "deletion", "1")
    # 0.65 in the window from its kept records: SNVs 0.1, short indels 0.05, deletions 0.5.
    assert float(row["snv_share"]) == pytest.approx(0.1 / 0.65, rel=1e-5) and float(row["short_indel_share"]) == pytest.approx(0.05 / 0.65, rel=1e-5)
    assert float(row["sv_share"]) == pytest.approx(0.5 / 0.65, rel=1e-5) == float(row["top_sv_class_share"])
    assert row["genome_share"] == table["with_sv", "all"]["genome_share"] and row["snv_indel_genome_share"] == table["snv_indel", "all"]["genome_share"]


def _figure_run(tmp_path: Path) -> Path:
    run = tmp_path / "run" / "export"
    _export(run, ("hypertension", "asthma"), 12)
    _prior_tables(run, "hypertension")
    return run


def test_without_matplotlib_the_review_figures_are_skipped_with_a_note(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    out = tmp_path / "out"
    out.mkdir()
    (out / "aou_review_gains.png").write_bytes(b"")
    assert aou_results.main([str(_figure_run(tmp_path)), "--out", str(out)]) == 0
    assert "matplotlib is not installed" in capsys.readouterr().err
    # A previous run's review figure is removed: none is left beside tables it does not show.
    assert (out / "aou_summary.tsv").is_file() and not list(out.glob("aou_review*"))


def test_the_review_figures_are_drawn_with_matplotlib(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    out = tmp_path / "out"
    assert aou_results.main([str(_figure_run(tmp_path)), "--out", str(out)]) == 0
    notes = json.loads((out / "aou_review_notes.json").read_text())
    assert notes["figures"] == ["aou_review_accuracy", "aou_review_gains", "aou_review_ancestry", "aou_review_enrichment"]
    assert all((out / f"{name}{suffix}").stat().st_size > 0 for name in notes["figures"] for suffix in (".pdf", ".png"))
    assert (out / "aou_review.pdf").is_file() and not (out / "aou_review_auc.pdf").exists() and not (out / "aou_review_loci.png").exists()

