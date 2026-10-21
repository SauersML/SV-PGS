"""The learned prior by variant class and SV length for a finished run whose effects step predates its prior tables:
``prior_by_length.tsv`` and ``prior_enrichment.tsv``, read from the run's saved all-rows models and its store's record
annotations with the effects step's own functions (``workspace_pipeline.variant_effects``, ``prior_moments``,
``effect_summaries``), so each table holds the lines that step writes now, with its suppression of 1 to 20 records.

Inputs: these files, relative to the run directory (the run config's ``run_directory``), and nothing else in it.
- ``fit/models/<arm>/<trait>/model.json`` and ``arrays.npz``, for both arms and every trait: each saved model
  (``artifact.load_model``), of which the all-rows model (``<trait>/all``; the one model of a single split) is read, as
  the effects step reads it. The ``<trait>.json`` records beside the folders are not read.
- ``direct_sv/_COMPLETE.json``: its summary's ``store``, the store the fit read, relative to the run directory
  (``store/store`` where the run added no direct row and the measurement step rewrote nothing), unless ``--store``
  names it.
- From that store, its variant table alone (``variant_table``): ``MANIFEST.json``, ``variants/`` (every chromosome's
  columns) and ``stats/half<h>/<chromosome>/sum_code`` and ``sum_code2`` (per-record sums over each half's samples,
  which the table's reader checks). ``dosage/``, ``samples/``, ``loci/`` and ``FEATURE_IDENTITY.json`` are never
  opened, so a store copied without them works.
- ``cohort/cohort.json``: the trait order (its ``trait_names``, nothing else in it), so the lines come in the effects
  step's order; without it, the traits' folder names in sorted order.
- ``score/_COMPLETE.json``, where it exists: the traits the score step quarantined, which have no lines, as in the
  effects step; without it, a model ``model_acceptance.require_scorable`` refuses is still left out.

Outputs: ``--out`` (default ``<run>/prior_tables``) gets the two tables, each through
``workspace_pipeline.check_exportable``; a table it refuses is removed and its rule printed. Nothing else is written,
and stdout names the tables, the models read and how many held a prior (every model of the full-data fit's default,
mean-field inference does; a dense law, from its EP inference, has none and adds no line), never a record. No step
runner, step key or run config is read, so a run made by older code works as long as this tree's
``artifact.load_model`` and ``DosageStore`` read its model and store. The tables' columns are this tree's: a run whose
effects step predates the posterior enrichment's interval in ``prior_by_length.tsv`` gets it here.

    python -m benchmarks.bench_real.prior_tables RUN_DIR [--out DIR] [--store STORE]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from sv_pgs import artifact
from sv_pgs.compute_budget import detect_compute_budget
from sv_pgs.dosage_store import (
    VARIANT_CLASS_LEGEND,
    VariantTable,
    _read_variant_table,
    read_manifest,
)
from sv_pgs.model_acceptance import require_scorable
from sv_pgs.workspace_pipeline import (
    ALL_ROWS_MODEL,
    ARMS,
    COMPLETE_FILE,
    PRIOR_BY_LENGTH_COLUMNS,
    PRIOR_ENRICHMENT_COLUMNS,
    check_exportable,
    effect_summaries,
    length_bins,
    prior_moments,
    variant_effects,
)

TABLES = {"prior_by_length.tsv": PRIOR_BY_LENGTH_COLUMNS, "prior_enrichment.tsv": PRIOR_ENRICHMENT_COLUMNS}


def store_path(run_directory: Path) -> Path:
    """The store the fit read: the direct_sv step's (``workspace_pipeline._final_store``)."""
    record = json.loads((run_directory / "direct_sv" / COMPLETE_FILE).read_text(encoding="utf-8"))
    return run_directory / record["summary"]["store"]


def variant_table(store: Path) -> VariantTable:
    """The store's variant table, read as ``DosageStore.open`` reads it (its manifest, ``variants/`` and every half's
    ``stats/`` sums, with the same checks) but without opening ``dosage/``: the genotype arrays need not be there."""
    manifest = read_manifest(store)
    return _read_variant_table(store, manifest, range(len(manifest["half_sample_counts"])))


def traits(run_directory: Path) -> list[str]:
    """The traits with a saved model, in the cohort's trait order, less those the score step quarantined."""
    folders = [run_directory / "fit" / "models" / arm for arm in ARMS]
    saved = sorted({path.name for folder in folders if folder.is_dir() for path in folder.iterdir() if path.is_dir()})
    cohort = run_directory / "cohort" / "cohort.json"
    order = [name for name in json.loads(cohort.read_text(encoding="utf-8"))["trait_names"] if name in saved] if cohort.is_file() else saved
    score = run_directory / "score" / COMPLETE_FILE
    quarantined = set(json.loads(score.read_text(encoding="utf-8"))["summary"].get("quarantined", ())) if score.is_file() else set()
    return [name for name in order if name not in quarantined]


def all_rows_model(fitted: Any, trait: str) -> int:
    """The index of the trait's released model in ``fitted``: ``<trait>/all``, or the one model of a single split."""
    names = list(fitted.model_names)
    if f"{trait}/{ALL_ROWS_MODEL}" in names:
        return names.index(f"{trait}/{ALL_ROWS_MODEL}")
    if len(names) == 1:
        return 0
    raise ValueError(f"the saved model of {trait} holds no all-rows model among {names}.")


def prior_lines(
    run_directory: Path, store: Path, working_bytes: int, load_model: Callable[[Path], Any] = artifact.load_model,
) -> tuple[dict[str, list[list[Any]]], list[str]]:
    """Each table's lines (its header first) over every arm and trait in the effects step's order, and a note per
    model read or left out (no value in it)."""
    lines: dict[str, list[list[Any]]] = {name: [list(columns)] for name, columns in TABLES.items()}
    notes = []
    names = traits(run_directory)
    table = variant_table(store)
    classes = np.asarray(VARIANT_CLASS_LEGEND, dtype=object)[np.asarray(table.variant_class)]
    lengths = np.asarray(table.annotations["sv_length"], dtype=np.float64)
    bins = length_bins(lengths)
    codes_per_unit = np.asarray(table.codes_per_unit, dtype=np.float64)
    for arm in ARMS:
        for trait in names:
            folder = run_directory / "fit" / "models" / arm / trait
            if not folder.is_dir():
                notes.append(f"{arm} {trait}: no saved model")
                continue
            fitted = load_model(folder)
            try:
                require_scorable(fitted)
            except ValueError as error:
                notes.append(f"{arm} {trait}: left out ({error})")
                continue
            scoring = fitted.scoring[all_rows_model(fitted, trait)]
            effects = variant_effects(scoring, codes_per_unit, working_bytes)
            rows = effects.store_rows
            unit = codes_per_unit[rows] / np.asarray(scoring.signed_scales, dtype=np.float64)
            prior = prior_moments(scoring, unit)
            found = effect_summaries(trait, arm, classes[rows], bins[rows], lengths[rows], effects, unit, prior)
            # The effects step keeps a model's lines as JSON before writing them; the same round trip writes the same text.
            found = json.loads(json.dumps({name: found[name] for name in TABLES}, default=str))
            for name in TABLES:
                lines[name].extend(found[name])
            notes.append(f"{arm} {trait}: " + ("prior read" if prior is not None else "its draw law is dense and holds no prior, so no lines"))
    return lines, notes


def write_tables(lines: dict[str, list[list[Any]]], out: Path) -> tuple[list[Path], dict[str, str]]:
    """Each table in ``out`` as the effects step writes it, kept only where the export check passes it."""
    out.mkdir(parents=True, exist_ok=True)
    written, refused = [], {}
    for name, rows in lines.items():
        path = out / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle, delimiter="\t").writerows(rows)
        try:
            check_exportable(path)
        except ValueError as error:
            path.unlink()
            refused[name] = str(error).rsplit(": ", 1)[-1]
        else:
            written.append(path)
    return written, refused


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_directory", type=Path, help="The finished run's directory (its fit, direct_sv, cohort and score steps).")
    parser.add_argument("--out", type=Path, help="Where the two tables go (default <run>/prior_tables).")
    parser.add_argument("--store", type=Path, help="The store the fit read (default the direct_sv step's).")
    args = parser.parse_args(argv)
    run_directory = args.run_directory.resolve()
    store = args.store or store_path(run_directory)
    print(f"store: {store} (its variant table only)")
    lines, notes = prior_lines(run_directory, store, detect_compute_budget().host_bytes // 2)
    written, refused = write_tables(lines, args.out or run_directory / "prior_tables")
    for note in notes:
        print(note)
    for path in written:
        print(f"wrote {path}")
    for name, rule in refused.items():
        print(f"not written, refused by the export check: {name} ({rule})")
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
