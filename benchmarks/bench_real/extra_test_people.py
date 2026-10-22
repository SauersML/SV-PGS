"""Extra held-out people for the matched SV-gain test (``sv_gain_meta``): both arms of each matched pairing run scored
on people of a later imputed callset who were in no fit, with their outcomes, covariates, families and ancestry.

The callset (``--callset``, a path pattern over ``{batch}`` and ``{chromosome}``) must carry the popped site list the
runs' store was built from. Each record a model uses is looked up by POS and INFO/ID and checked against the md5 of its
REF/ALT, the strata sidecar's keys (``workspace_pipeline.read_popped_sites``). Per model, the records the callset lacks
are counted and scored at their training mean. A found record's DS is coded as the store step codes it
(``store_converter._decode_records``): thousandths, the value-matched background removed with the bubble's K and m,
then 1/127-dosage codes. A model's genetic score is fast_scoring's mean score,
sum_j beta_j (s_ij - mu_j) / sigma_j over the signed codes s = code - 127. Only the records some model uses are read,
by index (``bcftools view -R``). Where a folder holds several models (cross-fitting), the trait's released model, the
last, is scored, since the new people are outside every fold.

``check`` scores one batch of a run's own imputed people this way and compares them with the run's score step. The
largest absolute difference of the genetic scores must be at rounding level.

The callset names people by DRAGEN sequencing ids. ``--crosswalk`` (research_id, sequencing_id) maps them to research
ids. A new person enters a trait's rows where the run's phenotype table has their outcome and the ancestry table has
their PCs. They are left out where they are already in the run's cohort or related (above second degree) to one of its
fit rows. Families come from the same kinship table as the cohort's, joined to the cohort's families where a new person
is related to someone in it. Everything runs inside the workspace; only counts and aggregates are printed, with counts
of 1 to 20 shown as <=20.

    python -m benchmarks.bench_real.extra_test_people score --callset PATTERN --batches 0 1 2 3 4 \\
        --crosswalk TSV --out DIR PAIR=SNV_RUN,SV_RUN [...]
    python -m benchmarks.bench_real.extra_test_people check RUN --batch BATCH
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OPS = "/mnt/disks/ops/svpgs"
CODE_CENTRE = 127.0
"""fast_scoring.SIGNED_CODE_OFFSET: the signed code is code - 127."""
CHUNK_ROWS = 4000
"""Records per extraction task: one ``bcftools view -R`` over their positions in one batch file."""


def shown(count: int) -> str:
    return "<=20" if 1 <= count <= 20 else str(count)


@dataclass(frozen=True)
class ModelWeights:
    """One saved model's mean score in store rows: weights beta / sigma and centres weights * mu (signed-code units)."""

    rows: np.ndarray
    weights: np.ndarray
    centres: np.ndarray
    sites_digest: str


def model_weights(folder: str | Path) -> ModelWeights:
    """The released (last) model of a saved fit folder (``artifact.save_model``), read from its arrays alone."""
    folder = Path(folder)
    metadata = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    index = len(metadata["model_names"]) - 1
    with np.load(folder / "arrays.npz", allow_pickle=False) as arrays:
        rows = np.asarray(arrays[f"scoring/{index}/store_rows"], dtype=np.int64)
        means = np.asarray(arrays[f"scoring/{index}/signed_means"], dtype=np.float64)
        scales = np.asarray(arrays[f"scoring/{index}/signed_scales"], dtype=np.float64)
        beta = np.asarray(arrays[f"scoring/{index}/coefficients"], dtype=np.float64)
    weights = beta / scales
    return ModelWeights(rows, weights, weights * means, str(metadata["provenance"]["sites_digest"]))


@dataclass(frozen=True)
class WeightTable:
    """Every model's weights and centres over the union of their store rows ([rows, models], 0 off a model's rows)."""

    rows: np.ndarray
    weights: np.ndarray
    centres: np.ndarray
    member: np.ndarray


def weight_table(models: Sequence[ModelWeights]) -> WeightTable:
    rows = np.unique(np.concatenate([model.rows for model in models]))
    weights = np.zeros((rows.shape[0], len(models)))
    centres = np.zeros_like(weights)
    member = np.zeros(weights.shape, dtype=bool)
    for column, model in enumerate(models):
        at = np.searchsorted(rows, model.rows)
        weights[at, column], centres[at, column], member[at, column] = model.weights, model.centres, True
    return WeightTable(rows, weights, centres, member)


def coded(dosage: np.ndarray, kept_paths: np.ndarray, carrying_paths: np.ndarray) -> np.ndarray:
    """DS [records, samples] as the store holds it: thousandths, the value-matched background removed, 8-bit codes."""
    from sv_pgs.dosage_store import MAXIMUM_DOSAGE_MILLI, encode_dosage_milli
    from sv_pgs.store_converter import value_matched_background

    milli = np.rint(np.asarray(dosage, dtype=np.float64) * 1000.0)
    if not np.all(np.isfinite(milli)) or np.any((milli < 0) | (milli > MAXIMUM_DOSAGE_MILLI)):
        raise ValueError("a DS is missing or outside [0, 2].")
    corrected = value_matched_background(milli.astype(np.uint16), kept_paths, carrying_paths)
    return encode_dosage_milli(corrected.dosage_milli)


def partial_scores(codes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """[models, samples]: each model's sum of w_j s_ij over these records (s = code - 127)."""
    return weights.T @ (codes.astype(np.float64) - CODE_CENTRE)


def finished_scores(partial: np.ndarray, table: WeightTable, found: np.ndarray) -> np.ndarray:
    """[models, samples] mean scores: a record not found sits at its training mean, so it adds neither term."""
    return partial - table.centres[found].sum(axis=0)[:, None]


# ---------------------------------------------------------------------------
# Inside the workspace: the store's site keys, the callset's DS, and the runs
# ---------------------------------------------------------------------------


def load_run(name: str):
    from sv_pgs.compute_budget import detect_compute_budget
    from sv_pgs.workspace_pipeline import WorkspaceConfig, _Run, workspace_bindings

    config = json.load(open(f"{OPS}/{name}/run_config.json", encoding="utf-8"))
    config["run_directory"], config["marker_directory"] = f"{OPS}/{name}/run", f"/tmp/extra_test_markers_{name}"
    path = f"/tmp/extra_test_{name}.json"
    json.dump(config, open(path, "w", encoding="utf-8"))
    return _Run(WorkspaceConfig.read(path), workspace_bindings(), detect_compute_budget(), {})


def store_layout(run) -> tuple[tuple[str, ...], np.ndarray, dict[str, str]]:
    """The final store's chromosomes, chromosome starts and per-chromosome sites md5, from its MANIFEST."""
    from sv_pgs.workspace_pipeline import _final_store

    manifest = json.loads((_final_store(run) / "MANIFEST.json").read_text(encoding="utf-8"))
    chromosomes = tuple(manifest["chromosomes"])
    starts = np.concatenate([[0], np.cumsum(manifest["record_counts"])]).astype(np.int64)
    return chromosomes, starts, dict(manifest["sites_md5"])


def chromosome_keys(task: tuple) -> dict[str, np.ndarray]:
    """The site keys and path counts of a chromosome's ``local`` records (strata order), checked against the store."""
    from sv_pgs.workspace_pipeline import path_counts, read_bubble_paths, read_strata_sites

    popped, chromosome, legend, local, record_count, sites_md5 = task
    strata = read_strata_sites(popped.strata_directory, chromosome, legend)
    if strata.positions.shape[0] != record_count or strata.sites_md5 != sites_md5:
        raise ValueError(f"{chromosome}: the strata sidecar is not the store's site list.")
    counts = path_counts(strata, read_bubble_paths(popped.bubble_split_path(chromosome)))
    return {
        "positions": strata.positions[local], "identifiers": np.array(strata.identifiers, dtype=object)[local],
        "digests": strata.refalt_digests[local], "kept": counts.kept[local], "carrying": counts.carrying[local],
    }


def extract_task(task: tuple) -> tuple:
    """One batch file's coded DS at one chunk of records, scored: (samples, found rows, partial [models, samples])."""
    from cyvcf2 import VCF

    from sv_pgs.store_converter import refalt_digest

    path, chromosome, rows, keys, weights = task
    wanted = {(int(position), str(identifier)): index for index, (position, identifier) in enumerate(zip(keys["positions"], keys["identifiers"]))}
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as handle:
        for position in np.unique(keys["positions"]):
            handle.write(f"{chromosome}\t{position}\n")
        regions = handle.name
    process = subprocess.Popen(["bcftools", "view", "--no-version", "-R", regions, "--regions-overlap", "0", "-Ou", path],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    reader = VCF(f"/proc/self/fd/{process.stdout.fileno()}")
    samples = tuple(reader.samples)
    dosage = np.zeros((len(wanted), len(samples)), dtype=np.float32)
    found = np.zeros(len(wanted), dtype=bool)
    try:
        for record in reader:
            index = wanted.get((record.POS, str(record.INFO.get("ID"))))
            if index is None or found[index]:
                continue
            if refalt_digest(record.REF, ",".join(record.ALT)) != int(keys["digests"][index]):
                raise ValueError(f"{path}: {chromosome}:{record.POS} has the site's ID but not its REF/ALT.")
            dosage[index] = record.format("DS")[:, 0]
            found[index] = True
    finally:
        reader.close()
        error = process.communicate()[1]
        os.unlink(regions)
    if process.returncode != 0:
        raise RuntimeError(f"bcftools view failed on {path}: {error.decode(errors='replace')[-500:]}")
    codes = coded(dosage[found], keys["kept"][found], keys["carrying"][found])
    return samples, rows[found], partial_scores(codes, weights[found])


def score_callset(pattern: str, batches: Sequence[str], run, table: WeightTable, workers: int, chromosomes: Sequence[str] | None = None):
    """Every batch's samples, each model's mean scores [models, samples] and found-record mask over ``table.rows``."""
    from sv_pgs.workspace_pipeline import strata_complexity_legend

    store_chromosomes, starts, sites_md5 = store_layout(run)
    wanted = [chromosome for chromosome in store_chromosomes if chromosomes is None or chromosome in chromosomes]
    popped = run.config.popped
    legend = strata_complexity_legend(popped.strata_directory, store_chromosomes)
    position = np.searchsorted(starts, table.rows, side="right") - 1
    begun = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pieces = {}
        for chromosome in wanted:
            index = store_chromosomes.index(chromosome)
            pieces[chromosome] = np.flatnonzero(position == index)
        key_tasks = [(popped, chromosome, legend, table.rows[pieces[chromosome]] - starts[store_chromosomes.index(chromosome)],
                      int(starts[store_chromosomes.index(chromosome) + 1] - starts[store_chromosomes.index(chromosome)]), sites_md5[chromosome])
                     for chromosome in wanted]
        keys = dict(zip(wanted, pool.map(chromosome_keys, key_tasks)))
        print(f"site keys of {len(wanted)} chromosomes in {time.time() - begun:.0f} s", flush=True)
        tasks, owners = [], []
        for batch in batches:
            for chromosome in wanted:
                members = pieces[chromosome]
                for start in range(0, members.shape[0], CHUNK_ROWS):
                    part = slice(start, start + CHUNK_ROWS)
                    tasks.append((pattern.format(batch=batch, chromosome=chromosome), chromosome, members[part],
                                  {name: values[part] for name, values in keys[chromosome].items()}, table.weights[members[part]]))
                    owners.append(batch)
        samples: dict[str, tuple[str, ...]] = {}
        partial: dict[str, np.ndarray] = {}
        found = {batch: np.zeros(table.rows.shape[0], dtype=bool) for batch in batches}
        for done, (owner, (names, rows, piece)) in enumerate(zip(owners, pool.map(extract_task, tasks, chunksize=1)), start=1):
            if samples.setdefault(owner, names) != names:
                raise ValueError(f"batch {owner}: the files list different samples.")
            partial[owner] = partial.get(owner, 0.0) + piece
            found[owner][rows] = True
            if done % 100 == 0 or done == len(tasks):
                print(f"extraction tasks {done}/{len(tasks)} in {time.time() - begun:.0f} s", flush=True)
    scores = {batch: finished_scores(partial[batch], table, found[batch]) for batch in batches}
    return samples, scores, found


def model_folder(run_name: str, arm: str, trait: str) -> Path:
    return Path(f"{OPS}/{run_name}/run/fit/models/{arm}/{trait}")


def report_found(label: str, table: WeightTable, found: dict[str, np.ndarray], names: Sequence[str]) -> None:
    for column, name in enumerate(names):
        members = table.member[:, column]
        lacking = [int((members & ~mask).sum()) for mask in found.values()]
        print(f"{label} {name}: model records {int(members.sum())}, not in the callset {max(lacking)} (largest over batches)")


# ---------------------------------------------------------------------------
# The new people's rows: outcome, covariates, family and ancestry
# ---------------------------------------------------------------------------


def aligned_covariates(names: Sequence[str], values: np.ndarray, wanted: Sequence[str]) -> np.ndarray:
    """``values`` [people, names] as the columns ``wanted``, in that order; a column the new people lack (a level none
    of them has) is 0."""
    out = np.zeros((values.shape[0], len(wanted)))
    for column, name in enumerate(wanted):
        if name in names:
            out[:, column] = values[:, list(names).index(name)]
    return out


def joined_families(old_ids: Sequence[str], old_families: np.ndarray, new_ids: Sequence[str], components: np.ndarray) -> np.ndarray:
    """A family label for each new person: the cohort family of anyone in their kinship component, else a new label
    above the cohort's. ``components`` labels ``old_ids`` then ``new_ids``."""
    count = len(old_ids)
    joined = {int(component): int(family) for component, family in zip(components[:count], old_families)}
    fresh: dict[int, int] = {}
    base = int(np.max(old_families)) + 1 if count else 0
    labels = []
    for component in components[count:].tolist():
        if component in joined:
            labels.append(joined[component])
        else:
            labels.append(fresh.setdefault(component, base + len(fresh)))
    return np.array(labels, dtype=np.int64)


def new_people_rows(run, trait_index: int, new_ids: Sequence[str]) -> dict[str, np.ndarray]:
    """One trait's new people (research ids) with an outcome and PCs, outside the cohort and unrelated to its fit rows."""
    from sv_pgs.cohort import AncestryPcs, ResearchId, build_cohort, kinship_components, read_kinship_pairs, read_predicted_ancestry
    from sv_pgs.workspace_pipeline import _read_cohort, _read_table, trait_table

    cohort = _read_cohort(run)
    name = cohort.trait_names[trait_index]
    phenotypes = run.output("phenotypes")
    metadata = json.loads((phenotypes / f"{name}.tsv.metadata.json").read_text(encoding="utf-8"))
    table = trait_table(_read_table(phenotypes / f"{name}.tsv"), metadata["covariate_columns"])
    ancestry = AncestryPcs.read(run.config.ancestry)
    labels = read_predicted_ancestry(run.config.ancestry)
    in_cohort = set(cohort.research_ids)
    pairs = read_kinship_pairs(run.config.relatedness, *run.config.relatedness_columns)
    fit_ids = {cohort.research_ids[row] for row in np.flatnonzero(cohort.fit_rows)}
    related_to_fit = {second.value if first.value in fit_ids else first.value for first, second, kinship in pairs
                      if kinship > 0.0884 and ((first.value in fit_ids) != (second.value in fit_ids))}
    have_pcs = set(ancestry.research_ids)
    counts = {"new": len(new_ids), "in cohort": sum(i in in_cohort for i in new_ids),
              "related to fit rows": sum(i in related_to_fit and i not in in_cohort for i in new_ids)}
    people = [i for i in new_ids if i not in in_cohort and i not in related_to_fit and i in have_pcs and i in table.targets]
    counts["with outcome and PCs"] = len(people)
    built = build_cohort([ResearchId(i) for i in people], ancestry, ["extra"] * len(people), ["imputed"] * len(people), {name: table})
    wanted = [column for column, used in zip(cohort.covariate_names, cohort.covariate_columns[trait_index]) if used]
    components = kinship_components(list(cohort.research_ids) + people, [p[0].value for p in pairs], [p[1].value for p in pairs], [p[2] for p in pairs])
    return {
        "ids": np.array(people, dtype=object), "target": built.targets[:, 0],
        "covariates": aligned_covariates(built.covariate_names, built.covariates, wanted),
        "covariate_names": np.array(wanted, dtype=object),
        "families": joined_families(cohort.research_ids, cohort.families, people, components),
        "ancestry": np.array([labels[i] for i in people], dtype=object), "counts": counts,
    }


def read_crosswalk(path: str | Path) -> dict[str, str]:
    """sequencing_id -> research_id."""
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["sequencing_id"]: row["research_id"] for row in csv.DictReader(handle, delimiter="\t")}


def score(args: argparse.Namespace) -> int:
    from sv_pgs.workspace_pipeline import ARMS

    pairs = []
    for spec in args.pairs:
        pair, arms = spec.split("=")
        snv_run, sv_run = arms.split(",")
        pairs.append((pair, {"snv_indel": snv_run, "with_sv": sv_run}))
    runs = {pair: load_run(pair) for pair, _arms in pairs}
    entries, models = [], []
    for pair, arms in pairs:
        from sv_pgs.workspace_pipeline import _read_cohort

        for trait in _read_cohort(runs[pair]).trait_names:
            for arm in ARMS:
                entries.append((pair, trait, arm))
                models.append(model_weights(model_folder(arms[arm], arm, trait)))
    digests = {model.sites_digest for model in models}
    if len(digests) != 1:
        raise ValueError(f"the models index {len(digests)} different store layouts; score each layout on its own.")
    table = weight_table(models)
    print(f"{len(models)} models over {table.rows.shape[0]} store records", flush=True)
    samples, scores, found = score_callset(args.callset, args.batches, runs[pairs[0][0]], table, args.workers, args.chromosomes)
    report_found("records", table, found, [f"{pair}/{trait}/{arm}" for pair, trait, arm in entries])
    crosswalk = read_crosswalk(args.crosswalk)
    sequencing = [name for batch in args.batches for name in samples[batch]]
    all_scores = np.concatenate([scores[batch] for batch in args.batches], axis=1)
    mapped = np.array([name in crosswalk for name in sequencing])
    research = np.array([crosswalk.get(name, "") for name in sequencing], dtype=object)
    print(f"callset people {shown(len(sequencing))}, with a research id {shown(int(mapped.sum()))}")
    out = Path(args.out)
    for pair, _arms in pairs:
        run = runs[pair]
        from sv_pgs.workspace_pipeline import _read_cohort

        for trait_index, trait in enumerate(_read_cohort(run).trait_names):
            rows = new_people_rows(run, trait_index, list(research[mapped]))
            place = {person: column for column, person in enumerate(research)}
            columns = np.array([place[person] for person in rows["ids"]], dtype=np.int64)
            arm_scores = {arm: all_scores[entries.index((pair, trait, arm)), columns] for arm in ARMS}
            (out / pair).mkdir(parents=True, exist_ok=True)
            np.savez(out / pair / f"{trait}.npz", ids=rows["ids"].astype(str), target=rows["target"], covariates=rows["covariates"],
                     covariate_names=rows["covariate_names"].astype(str), families=rows["families"], ancestry=rows["ancestry"].astype(str),
                     baseline=arm_scores["snv_indel"], candidate=arm_scores["with_sv"])
            print(f"{pair} {trait}: " + ", ".join(f"{key} {shown(value)}" for key, value in rows["counts"].items()))
    return 0


def check(args: argparse.Namespace) -> int:
    """One batch of a run's own imputed people scored here against the run's score step."""
    from sv_pgs.workspace_pipeline import ARMS, _read_cohort, _read_scores

    run = load_run(args.run)
    cohort, stored = _read_cohort(run), _read_scores(run)
    entries, models = [], []
    for trait_index, trait in enumerate(cohort.trait_names):
        for arm in ARMS:
            entries.append((trait_index, arm))
            models.append(model_weights(model_folder(args.run, arm, trait)))
    table = weight_table(models)
    half = run.config.imputed_halves[0]
    pattern = str(half.batch_path).replace("{batch}", args.batch)
    samples, scores, found = score_callset(pattern, ["x"], run, table, args.workers, args.chromosomes)
    report_found("records", table, found, [f"{cohort.trait_names[t]}/{arm}" for t, arm in entries])
    by_sequencing = read_crosswalk_columns(run)
    row_of = {person: row for row, person in enumerate(cohort.research_ids)}
    columns, rows = [], []
    for column, name in enumerate(samples["x"]):
        person = by_sequencing.get(name)
        if person in row_of:
            columns.append(column)
            rows.append(row_of[person])
    for model_index, (trait_index, arm) in enumerate(entries):
        reference = stored.genetic_mean[ARMS.index(arm), rows, cohort.all_rows_model(trait_index)]
        difference = np.abs(scores["x"][model_index, columns] - reference)
        print(f"check {cohort.trait_names[trait_index]}/{arm}: people {shown(len(rows))}, largest |difference| {difference.max():.3g}, "
              f"score SD {reference.std():.3g}")
    return 0


def read_crosswalk_columns(run) -> dict[str, str]:
    research, sequencing = run.config.crosswalk_columns
    with open(run.config.crosswalk, newline="", encoding="utf-8") as handle:
        return {row[sequencing]: row[research] for row in csv.DictReader(handle, delimiter="\t")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    scoring = commands.add_parser("score")
    scoring.add_argument("pairs", nargs="+", help="PAIR=SNV_RUN,SV_RUN: the run sv_gain_meta reads, then each arm's run.")
    scoring.add_argument("--callset", required=True, help="Path pattern over {batch} and {chromosome}.")
    scoring.add_argument("--batches", nargs="+", required=True)
    scoring.add_argument("--crosswalk", required=True)
    scoring.add_argument("--out", required=True)
    checking = commands.add_parser("check")
    checking.add_argument("run")
    checking.add_argument("--batch", required=True)
    for command in (scoring, checking):
        command.add_argument("--workers", type=int, default=os.cpu_count() or 1)
        command.add_argument("--chromosomes", nargs="+", default=None)
    args = parser.parse_args(argv)
    return score(args) if args.command == "score" else check(args)


if __name__ == "__main__":
    raise SystemExit(main())
