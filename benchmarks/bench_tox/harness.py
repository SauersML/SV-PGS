"""The polygenic cytotoxicity benchmark (PREREG.md): every panel record, the ``snv`` and ``snv_sv`` arms, the
pre-registered splits, covariates fitted in-fold, development compounds only, three methods on the same screened
variant set per compound and fold, held-out r² pooled over folds and compounds.

The screen is the one a polygenic pipeline uses when a fitter cannot take the genome: each compound's marginal
association on the training lines alone (its phenotype and every record projected on the fold's covariates, a
two-sided t test of the correlation), kept where Benjamini-Hochberg holds the false discovery rate at 1/(2K), K the
engine's draw count, the resolution every certificate in the project already uses. The screen's size is measured per
compound, fold and arm, never chosen; it is applied identically to every method; the ``snv`` arm screens among the
small variants only and ``snv_sv`` among every record. A compound whose screen keeps nothing has no predictor: its
methods return the training mean, scored as r² = 0 and counted.

Covariates, fitted on the training lines only: sex, the cytotoxicity batch (indicators), and the genotype principal
components of the training lines' GRM of random near-unlinked autosomal SNVs (``structure_grm``) that Patterson's
sequential Tracy-Widom test keeps at level 1/K (Patterson, Price and Reich 2006, PLoS Genet 2:e190).

Methods: ``top_variant`` and ``gblup_reml`` from bench-real's baselines, and SV-PGS by the small-n route with
mean-field fixed points (``fit_small_n``); each fit's wall time is recorded. Held-out r² is the squared correlation
of the genetic score with the test lines' phenotype residualized on the training-fitted covariate effects.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd
from scipy import stats

from benchmarks import svpgs_small_n as bench
from benchmarks.bench_real import baselines, harness as real_harness
from benchmarks.bench_tox import dataset, genotypes, splits
from benchmarks.seeds import seed_from_name
from sv_pgs.fit_model import DRAW_COUNT
from sv_pgs.small_n import fit_small_n

WORK = dataset.WORK
# Tracy-Widom (beta = 1) percentage points, Patterson, Price and Reich 2006, Table 1: P = 0.05, 0.01, 0.005, 0.001.
_TRACY_WIDOM_POINTS = ((0.05, 0.9793), (0.01, 2.0234), (0.005, 2.4224), (0.001, 3.2724))
_CHUNK = 20_000


def _tracy_widom_quantile(level: float) -> float:
    """The TW1 point at ``level``, interpolated linearly in log(P) between the tabulated points."""
    points = sorted(_TRACY_WIDOM_POINTS, key=lambda point: -point[0])
    logs = np.log([point[0] for point in points])
    values = [point[1] for point in points]
    return float(np.interp(np.log(level), logs[::-1], values[::-1]))


def structure_grm(train_rows: np.ndarray, sample_count: int) -> np.ndarray:
    """The GRM the PCs come from: the KING draw of random autosomal SNVs (``dataset.KING_MARKERS_PER_CHROMOSOME`` per
    chromosome, one per 65 kb on average: near-unlinked, as Patterson's test assumes), each standardized by its
    training-fold frequency, without the training fold's singletons (a singleton carries no information on any pair)
    or its monomorphic sites; over every line, so the test lines' scores project onto the training eigenvectors.

    Not GCTA's all-record GRM: on lococ/AFR its spectrum is nowhere near Wishart (25 million records, most rare, the
    diagonal 0.47 with LD everywhere), and Patterson's sequential test kept every axis (598 of 599)."""
    generator = np.random.default_rng(splits.seed_from_name("bench-tox/king"))
    blocks = []
    for chrom in genotypes.AUTOSOMES:
        table = genotypes.variant_table(chrom)
        snv_rows = np.flatnonzero(table["is_snv"].to_numpy())
        chosen = np.sort(generator.permutation(snv_rows)[: dataset.KING_MARKERS_PER_CHROMOSOME])
        blocks.append(genotypes.read_bed(chrom, chosen, sample_count).T)
    calls = np.concatenate(blocks, axis=1).astype(np.float64)
    count = calls[train_rows].sum(axis=0)
    kept = (count >= 2.0) & (count <= 2.0 * train_rows.shape[0] - 2.0)
    frequency = count[kept] / (2.0 * train_rows.shape[0])
    standardized = (calls[:, kept] - 2.0 * frequency) / np.sqrt(2.0 * frequency * (1.0 - frequency))
    return standardized @ standardized.T / kept.sum()


def significant_components(grm: np.ndarray, level: float) -> np.ndarray:
    """The training lines' principal components Patterson's test keeps: the leading eigenvalues tested one at a time
    against TW1 at ``level``, each on the eigenvalues remaining after the ones before it (Patterson et al. 2006,
    "Testing for structure")."""
    eigenvalues, eigenvectors = np.linalg.eigh(grm)
    eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
    point = _tracy_widom_quantile(level)
    kept = 0
    for start in range(eigenvalues.shape[0] - 1):
        remaining = eigenvalues[start:]
        remaining = remaining[remaining > 0.0]
        m = remaining.shape[0]
        if m < 3:
            break
        total, squares = float(np.sum(remaining)), float(np.sum(np.square(remaining)))
        effective = (m + 1.0) * total * total / ((m - 1.0) * squares - total * total)
        if not np.isfinite(effective) or effective <= 1.0:
            break
        normalized = m * float(remaining[0]) / total
        mean = (np.sqrt(effective - 1.0) + np.sqrt(m)) ** 2 / effective
        spread = (np.sqrt(effective - 1.0) + np.sqrt(m)) / effective * (1.0 / np.sqrt(effective - 1.0) + 1.0 / np.sqrt(m)) ** (1.0 / 3.0)
        if (normalized - mean) / spread <= point:
            break
        kept += 1
    return eigenvectors[:, :kept]


def fold_covariates(lines: pd.DataFrame, train_rows: np.ndarray, test_rows: np.ndarray, grm: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray, int]:
    """(training covariates, test covariates, PCs kept), without the intercept: sex, batch indicators (levels seen in
    training, the first dropped), and the training GRM's significant PCs, with the test lines' scores projected onto
    them (the GRM rows of the test lines against the training lines)."""
    sex = (lines["sex"].to_numpy(dtype=np.float64) - 1.0)[:, None]
    batches = sorted(lines["batch"].iloc[train_rows].unique())[1:]
    batch = np.column_stack([(lines["batch"] == level_name).to_numpy(dtype=np.float64) for level_name in batches]) if batches else np.zeros((len(lines), 0))
    training_grm = grm[np.ix_(train_rows, train_rows)]
    components = significant_components(training_grm, level)
    scores_train = components
    scores_test = grm[np.ix_(test_rows, train_rows)] @ components / np.maximum(np.linalg.eigvalsh(training_grm)[::-1][: components.shape[1]], np.finfo(np.float64).tiny) if components.shape[1] else np.zeros((len(test_rows), 0))
    train = np.column_stack([sex[train_rows], batch[train_rows], scores_train])
    test = np.column_stack([sex[test_rows], batch[test_rows], scores_test])
    return train, test, int(components.shape[1])


def _projector(design: np.ndarray):
    pseudo = np.linalg.pinv(design)
    return lambda values: values - design @ (pseudo @ values)


def _screen_chromosome(chrom: int, phenotypes: np.ndarray, phenotype_norm: np.ndarray, project, train_rows: np.ndarray, degrees: int, level: float, sample_count: int, compound_count: int):
    """One chromosome's part of ``screen_fold``: the records at or below ``level`` per compound, and the counts tested."""
    is_small = ~genotypes.variant_table(chrom)["is_sv"].to_numpy()
    held = [[] for _compound in range(compound_count)]
    tested = np.zeros(2, dtype=np.int64)
    for start, block in genotypes.read_chromosome_chunks(chrom, sample_count, _CHUNK):
        block = block[:, train_rows].T.astype(np.float64)
        if np.any(block < 0):
            raise ValueError("a missing call in the panel")
        projected = project(block)
        norms = np.sqrt(np.sum(np.square(projected), axis=0))
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation = (projected.T @ phenotypes) / (norms[:, None] * phenotype_norm[None, :])
        correlation = np.where(np.isfinite(correlation), correlation, 0.0)
        correlation = np.clip(correlation, -1.0 + np.finfo(np.float64).eps, 1.0 - np.finfo(np.float64).eps)
        t_values = correlation * np.sqrt(degrees / (1.0 - np.square(correlation)))
        p_values = 2.0 * stats.t.sf(np.abs(t_values), degrees)
        small_block = is_small[start : start + block.shape[1]]
        tested += (int(np.sum(small_block)), block.shape[1])
        rows, columns = np.nonzero(p_values <= level)
        for column in np.unique(columns):
            chosen = rows[columns == column]
            held[column].append(np.column_stack([
                np.full(chosen.shape[0], chrom), start + chosen, small_block[chosen].astype(np.int64), np.round(p_values[chosen, column] * 2.0**52).astype(np.int64),
            ]))
    return chrom, [np.concatenate(parts) if parts else np.zeros((0, 4), dtype=np.int64) for parts in held], tested


_SCREEN: dict = {}


def _screen_worker(chrom: int):
    return _screen_chromosome(chrom, **_SCREEN)


def screen_fold(values: pd.DataFrame, train_rows: np.ndarray, design: np.ndarray, level: float, sample_count: int, workers: int = 1) -> dict:
    """Per compound and arm, the rows (chromosome, bim row) BH keeps at ``level`` on the training lines: the t test of
    each record's correlation with the phenotype, both projected on ``design``, with n - k - 2 degrees of freedom.
    Only a p-value at or below the level can pass BH (its cutoff i q / m never exceeds q), so those alone are held.
    The chromosomes are screened in parallel over ``workers`` forked processes."""
    project = _projector(design)
    phenotypes = project(values.to_numpy(dtype=np.float64)[train_rows])
    phenotype_norm = np.sqrt(np.sum(np.square(phenotypes), axis=0))
    degrees = train_rows.shape[0] - design.shape[1] - 1
    _SCREEN.update(phenotypes=phenotypes, phenotype_norm=phenotype_norm, project=project, train_rows=train_rows, degrees=degrees, level=level,
                   sample_count=sample_count, compound_count=values.shape[1])
    held = {compound: [] for compound in values.columns}
    tested = np.zeros(2, dtype=np.int64)
    if workers > 1:
        with get_context("fork").Pool(workers) as pool:
            results = list(pool.imap_unordered(_screen_worker, list(genotypes.AUTOSOMES)))
    else:
        results = [_screen_worker(chrom) for chrom in genotypes.AUTOSOMES]
    _SCREEN.clear()
    for _chrom, parts, counted in sorted(results, key=lambda item: item[0]):
        tested += counted
        for compound, part in zip(values.columns, parts):
            held[compound].append(part)
    kept = {}
    for compound in values.columns:
        table = np.concatenate(held[compound]) if held[compound] else np.zeros((0, 4), dtype=np.int64)
        for arm, mask, count in (("snv", table[:, 2] == 1, int(tested[0])), ("snv_sv", np.ones(table.shape[0], dtype=bool), int(tested[1]))):
            candidates = table[mask]
            order = np.argsort(candidates[:, 3], kind="stable")
            p = candidates[order, 3].astype(np.float64) / 2.0**52
            passing = p <= level * (np.arange(1, p.shape[0] + 1) / max(count, 1))
            largest = int(np.max(np.flatnonzero(passing))) + 1 if np.any(passing) else 0
            kept[(arm, compound)] = candidates[order[:largest], :2]
    return kept


def _variants(chrom_rows: np.ndarray, tables: dict, sample_count: int, train_rows: np.ndarray) -> tuple[np.ndarray, real_harness.Variants]:
    """The screened records' dosages (lines x records) and a bench-real Variants of them."""
    blocks, frames = [], []
    for chrom in np.unique(chrom_rows[:, 0]):
        rows = chrom_rows[chrom_rows[:, 0] == chrom, 1]
        blocks.append(genotypes.read_bed(int(chrom), rows, sample_count).T)
        frames.append(tables[int(chrom)].iloc[rows])
    dosage = np.concatenate(blocks, axis=1) if blocks else np.zeros((sample_count, 0), dtype=np.int8)
    table = pd.concat(frames) if frames else genotypes.variant_table(22).iloc[:0]
    count = table.shape[0]
    variants = real_harness.Variants(
        position=table["position"].to_numpy(), end=table["end"].to_numpy(), distance_to_tss=np.zeros(count, dtype=np.int64),
        is_sv=table["is_sv"].to_numpy(), sv_type=table["sv_type"].to_numpy(dtype=str), sv_length=np.zeros(count, dtype=np.int64),
        allele_length_change=table["allele_length_change"].to_numpy(), train_allele_frequency=dosage[train_rows].mean(axis=0) / 2.0 if count else np.zeros(0),
        source=np.array(["panel"] * count),
    )
    return dosage, variants


class _Train:
    def __init__(self, genotypes_, phenotype, covariates, variants, gene_id):
        self.genotypes, self.phenotype, self.covariates, self.variants, self.gene_id = genotypes_, phenotype, covariates, variants, gene_id


def _working_bytes() -> int:
    """This worker's share of the task's memory: the runq allotment over the task's workers (``one_core_budget`` divides
    by the machine's threads, 48 on the runner against the task's 8, and refused every fit on a screen past 100,000
    records), less what the forked worker already holds."""
    allotment = os.environ.get("RUNQ_MEM_BYTES")
    if allotment is None:
        return bench._METHOD.one_core_budget().working_bytes
    share = int(allotment) // max(int(os.environ.get("RUNQ_CORES", "1")), 1) - bench._METHOD._resident_bytes()
    if share <= 0:
        raise MemoryError("a worker's share of the task's memory allotment is already spent by what it holds.")
    return share


def _svpgs(train: _Train, seed: int):
    genotypes_ = np.asarray(train.genotypes)
    if genotypes_.shape[1] == 0:
        return baselines.ZeroPredictor(train.phenotype.mean())
    fit = fit_small_n(
        codes=genotypes_.astype(np.uint8) * np.uint8(bench.CODES_PER_DOSAGE), covariates=bench._METHOD.bench_real_covariates(train),
        target=np.asarray(train.raw_phenotype, dtype=np.float64), variant_class=bench.classes_for_arm(train.variants, "full"), log_variance_offset=None,
        draw_count=DRAW_COUNT, working_bytes=_working_bytes(), seed=seed, inference="mean_field",
    )
    return bench.SmallNPredictor(scoring=fit.scoring, profile=fit.profile)


def _r2(prediction: np.ndarray, truth: np.ndarray) -> float:
    if not np.all(np.isfinite(prediction)) or np.ptp(prediction) == 0.0 or np.ptp(truth) == 0.0:
        return 0.0
    return float(np.corrcoef(prediction, truth)[0, 1] ** 2)


_STATE: dict = {}


def _init(state):
    _STATE.update(state)


def _one(job):
    """One (compound, arm) on the fold: the three methods' held-out r² and wall seconds, the screen's size."""
    arm, compound = job
    values, lines, train_rows, test_rows, design_train, design_test, kept, tables = (
        _STATE["values"], _STATE["lines"], _STATE["train_rows"], _STATE["test_rows"], _STATE["design_train"], _STATE["design_test"], _STATE["kept"], _STATE["tables"])
    chrom_rows = kept[(arm, compound)]
    dosage, variants = _variants(chrom_rows, tables, len(lines), train_rows)
    y = values[compound].to_numpy(dtype=np.float64)
    full_train = np.column_stack([np.ones(train_rows.shape[0]), design_train])
    full_test = np.column_stack([np.ones(test_rows.shape[0]), design_test])
    effects = np.linalg.pinv(full_train) @ y[train_rows]
    residual_train = y[train_rows] - full_train @ effects
    truth = y[test_rows] - full_test @ effects
    train = _Train(dosage[train_rows].astype(np.float64), residual_train, design_train, variants, compound)
    train.raw_phenotype = y[train_rows]
    test_genotypes = dosage[test_rows].astype(np.float64)
    record = {"compound": compound, "arm": arm, "screened": int(chrom_rows.shape[0]), "screened_sv": int(np.sum(variants.is_sv)) if chrom_rows.shape[0] else 0}
    seed = seed_from_name(compound)
    for name, method in (("top_variant", baselines.top_variant), ("gblup_reml", baselines.gblup_reml), ("svpgs_mean_field", lambda t: _svpgs(t, seed))):
        started = time.perf_counter()
        try:
            predictor = method(train)
            prediction = np.asarray(predictor.predict(test_genotypes), dtype=np.float64)
            status = "ok"
        except Exception as error:  # noqa: BLE001 - recorded, the run goes on
            prediction, status = np.full(test_rows.shape[0], np.nan), f"failed: {type(error).__name__}: {str(error)[:200]}"
        record[name] = {"r2": _r2(prediction, truth), "seconds": time.perf_counter() - started, "status": status}
        if name == "svpgs_mean_field" and status == "ok":
            record[name]["outer_criterion_met"] = bool(getattr(predictor, "profile", {}).get("outer_criterion_met"))
    return record


def run_fold(split: dict, workers: int, out_dir: pathlib.Path, level: float) -> None:
    lines = dataset.load_lines()
    values = dataset.load_values()
    index = {line: row for row, line in enumerate(lines["line"])}
    train_rows = np.array(sorted(index[line] for line in split["train"]))
    test_rows = np.array(sorted(index[line] for line in split["test"]))
    grm = structure_grm(train_rows, len(lines))
    design_train, design_test, components = fold_covariates(lines, train_rows, test_rows, grm, 1.0 / DRAW_COUNT)
    started = time.perf_counter()
    kept = screen_fold(values, train_rows, np.column_stack([np.ones(train_rows.shape[0]), design_train]), level, len(lines), workers)
    screen_seconds = time.perf_counter() - started
    tables = {chrom: genotypes.variant_table(chrom) for chrom in genotypes.AUTOSOMES}
    state = {"values": values, "lines": lines, "train_rows": train_rows, "test_rows": test_rows, "design_train": design_train, "design_test": design_test, "kept": kept, "tables": tables}
    jobs = [(arm, compound) for arm in ("snv", "snv_sv") for compound in values.columns]
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o700)
    name = split["name"].replace("/", "_")
    with open(out_dir / f"{name}.jsonl", "w") as handle, get_context("fork").Pool(workers, initializer=_init, initargs=(state,)) as pool:
        handle.write(json.dumps({"split": split["name"], "train": len(train_rows), "test": len(test_rows), "pcs": components, "screen_seconds": screen_seconds, "level": level}) + "\n")
        handle.flush()
        for record in pool.imap_unordered(_one, jobs, chunksize=1):
            record["split"] = split["name"]
            handle.write(json.dumps(record) + "\n")
            handle.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, help="a split name from splits.json, e.g. random5/fold0 or lococ/AFR")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("RUNQ_CORES", "1")))
    parser.add_argument("--out", default=str(WORK / "results"))
    arguments = parser.parse_args()
    splits = json.loads((WORK / "splits.json").read_text())
    (split,) = [each for each in splits if each["name"] == arguments.split]
    run_fold(split, arguments.workers, pathlib.Path(arguments.out), 0.5 / DRAW_COUNT)


if __name__ == "__main__":
    main()
