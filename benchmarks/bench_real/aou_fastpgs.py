"""Fast standard polygenic scores beside an All of Us run's fit, each fitted twice on the same people: once on the
snv_indel arm's records and once on the with_sv arm's, so the two fits differ only in the SVs.

The records are the run's: ``fit_filtered`` at its config's floors and site list on the direct_sv step's store. The
with_sv arm fits every kept record and the snv_indel arm the kept records under 50 bp (``arm_log_reliability``), so
the snv_indel records are a subset of the with_sv records and the arms share every SNV and indel. The people are the
run's too: each trait's evaluated model's training rows, and its held-out rows, where SV-PGS is scored. Three methods:

- LDpred2-auto, as aou_ldpred2.py fits it (ldpred2_auto_allvar.py's GWAS, band and settings), on every training row.
- lassosum2 (bigsnpr's snp_lassosum2 at its default grid: in 1.12.21, deltas 0.001, 0.01, 0.1 and 1 by 30 lambdas) on
  the GWAS of the training rows outside the tuning slice, with the band of every training row.
- Clumping and thresholding (``ct``): greedy clumping by p at r^2 above CLUMP_R2 within the band's 3 cM window, then
  each p-value threshold in THRESHOLDS, on the same GWAS as lassosum2.

The tuning slice is TUNING_FRACTION of the training families (``tuning_slice``), drawn once per trait and shared by
both arms. lassosum2's and C+T's parameters are each chosen by the largest partial r^2 on it, by the same rule in both
arms. Held-out rows never enter a fit or a choice.

One pass over the with_sv records computes both GWAS (all training rows, and the rows outside the tuning slice) and
both arms' bands (``gwas_and_bands``). The snv_indel arm's band is the with_sv band's correlations among its own
records. The projected dosages are the same, and its window is the same 3 cM in the same genetic positions, so the
restriction is exactly the band a pass over its records alone would write. The two arms' R sessions then run side by
side, each with half the cores. The codes of every record the arms read, over the trait's training and held-out
columns, are read from the store once (``read_codes``) and kept in memory: about 1.06 million records by 50,000
people is 53 GB.

Outputs, per trait:
- ``--out``: one JSON line per arm and method, aggregates only (the chosen parameter, the tuning and held-out partial
  r^2, a disease's held-out AUC, the variant counts, the held-out row count, never 1 to 20, and ``_cases_text``).
- ``--scores``: <trait>.npz, each held-out person's score by every method and arm, with the research ids, families,
  ancestry, target and covariates that fastpgs_meta.py's paired tests read. Participant-level: it stays in the
  workspace bucket beside the run's mirror.
- ``--units``: each trait's fitted effects, kept as they are made, so a preempted Spot task that starts again only
  reads the codes and scores (``Units``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import tempfile
import time
import zipfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import stats

from benchmarks.bench_sim.submissions import ldpred2_auto, ldpred2_auto_allvar, sbayesrc
from benchmarks.bench_sim.submissions.ldpred2_auto import WINDOW_CM
from benchmarks.bench_sim.submissions.ldpred2_auto_allvar import Adjuster
from benchmarks.bench_sim.submissions.sbayesrc import gemm_nt
from sv_pgs.all_of_us import MINIMUM_REPORTED_PARTICIPANTS
from sv_pgs.compute_budget import detect_compute_budget
from sv_pgs.config import TraitType
from sv_pgs.dosage_store import DosageStore
from sv_pgs.held_out_comparison import auc_components
from sv_pgs.workspace_pipeline import (
    WorkspaceConfig, _cases_text, _final_measurement_model, _final_store, _partial_r2, _Run, arm_log_reliability, fit_filtered,
    workspace_bindings,
)

try:
    from sv_pgs.workspace_pipeline import _fit_cohort as _cohort_of
except ImportError:  # a tree from before validation splits: its fit reads the cohort step's own split
    from sv_pgs.workspace_pipeline import _read_cohort as _cohort_of

ARMS = ("snv_indel", "with_sv")
METHODS = ("ldpred2_auto", "lassosum2", "ct")
THRESHOLDS = (5e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5, 1.0)
CLUMP_R2 = 0.1
TUNING_FRACTION = 0.2
CHUNK = ldpred2_auto_allvar.CHUNK
CHROMOSOME_SPACING_CM = 1e4
"""Each chromosome's genetic positions are offset by its number times this (aou_ldpred2.CHROMOSOME_SPACING_CM), so the
genome is one band and no window reaches another chromosome."""
MERGE_GAP = 64
"""Kept records this close in the store are read in one call (``read_codes``): a zstd store decodes a whole inner
chunk for any row of it, so separate reads of rows in one chunk would decode it again for each."""
MAX_SPAN = 4096
"""The most store rows one such read spans."""
READ_THREADS = 32
_SCRIPT = pathlib.Path(__file__).with_name("aou_fastpgs.R")


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_codes(store: DosageStore, records: np.ndarray, columns: np.ndarray, threads: int = READ_THREADS) -> np.ndarray:
    """Codes [records, columns] (uint8, dosage = code / 127) of ``records`` in the order given, over the ascending store
    ``columns``. Kept records closer than MERGE_GAP, or than the store's inner chunk rows where those are more, are read
    in one call, so a chunk is decoded once for all its kept records; a call spans at most MAX_SPAN rows, or four
    chunks. The calls run on ``threads`` threads, each on a run of neighbouring calls."""
    out = np.empty((records.shape[0], columns.shape[0]), dtype=np.uint8)
    order = np.argsort(records, kind="stable")
    ascending = records[order]
    gap = max([MERGE_GAP, *(array.layout.inner_rows for arrays in getattr(store, "_arrays", ()) for array in arrays)])
    longest = max(MAX_SPAN, 4 * gap)
    spans, first = [], 0
    for index in range(1, ascending.shape[0] + 1):
        if index == ascending.shape[0] or ascending[index] - ascending[index - 1] >= gap or ascending[index] - ascending[first] >= longest:
            spans.append((first, index))
            first = index

    def read(group: list[tuple[int, int]]) -> None:
        for first, last in group:
            start, stop = int(ascending[first]), int(ascending[last - 1]) + 1
            rows = order[first:last]
            if stop - start == last - first and np.all(np.diff(rows) == 1):
                store.read_codes(start, stop, columns, out=out[rows[0] : rows[0] + last - first])
            else:
                out[rows] = store.read_codes(start, stop, columns)[ascending[first:last] - start]

    groups = [[spans[index] for index in part] for part in np.array_split(np.arange(len(spans)), min(len(spans), threads * 8)) if part.size]
    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(read, groups))
    return out


class MemoryView:
    """The bench-sim training view (``codes``, ``covariates``, ``phenotype``, ``n_samples``) of in-memory codes: row r is
    row r of ``codes``, over its columns ``columns``."""

    def __init__(self, codes: np.ndarray, columns: np.ndarray, covariates: np.ndarray, phenotype: np.ndarray) -> None:
        self.all_codes, self.columns = codes, np.asarray(columns, dtype=np.int64)
        self.covariates, self.phenotype = covariates, phenotype
        self.n_samples = int(self.columns.shape[0])

    def codes(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        block = self.all_codes[rows[0] : rows[-1] + 1] if rows.size and rows[-1] - rows[0] + 1 == rows.size else self.all_codes[rows]
        return np.take(block, self.columns, axis=1)


def windows(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each record's band rows lo .. hi - 1: the records within WINDOW_CM of it (``ldpred2_auto_allvar.write_band``)."""
    return np.searchsorted(cm, cm - WINDOW_CM, side="left"), np.searchsorted(cm, cm + WINDOW_CM, side="right")


def gwas_and_bands(train, part, cm: np.ndarray, short: np.ndarray, folders: Mapping[str, pathlib.Path]) -> dict[str, dict[str, np.ndarray]]:
    """Both GWAS and both arms' bands in one pass over the with_sv records (rising in ``cm``): ``train`` is the view of
    every training row and ``part`` of those outside the tuning slice; ``short`` marks the snv_indel arm's records.

    Each arm's band goes to its folder's ld_values.bin as ``write_band`` writes it (column j holds rows lo[j] .. hi[j] - 1,
    float32, diagonal exactly 1). Returns per arm, in its own record order: the training means, the all-training GWAS
    (beta, se), the tuning-free GWAS (beta_part, se_part), lo, hi and the LD scores."""
    total = cm.shape[0]
    short = np.asarray(short, dtype=bool)
    lo, hi = windows(cm)
    short_lo, short_hi = windows(cm[short])
    adjuster, part_adjuster = Adjuster(train), Adjuster(part)
    means, beta, se, beta_part, se_part, scores = (np.empty(total) for _ in range(6))
    short_scores = np.empty(int(short.sum()))
    rank = np.cumsum(short) - 1
    cache: dict[int, np.ndarray] = {}
    with open(folders["with_sv"] / "ld_values.bin", "wb") as wide, open(folders["snv_indel"] / "ld_values.bin", "wb") as narrow:
        for c0 in range(0, total, CHUNK):
            c1 = min(c0 + CHUNK, total)
            r0, r1 = int(lo[c0]), int(hi[c1 - 1])
            for index in [key for key in cache if (key + 1) * CHUNK <= r0]:
                del cache[index]
            for index in range(r0 // CHUNK, (r1 - 1) // CHUNK + 1):
                if index not in cache:
                    rows = np.arange(index * CHUNK, min((index + 1) * CHUNK, total))
                    means[rows], beta[rows], se[rows], _, cache[index] = adjuster.block(rows)
                    _, beta_part[rows], se_part[rows], _, _ = part_adjuster.block(rows)
            columns = cache[c0 // CHUNK]
            # products[k, i - r0] = r(c0 + k, i) for i in [r0, r1)
            products = np.hstack([gemm_nt(columns, cache[index][max(r0 - index * CHUNK, 0) : r1 - index * CHUNK])
                                  for index in range(r0 // CHUNK, (r1 - 1) // CHUNK + 1)])
            products[np.arange(c1 - c0), np.arange(c0, c1) - r0] = 1.0
            position = np.arange(r0, r1)[None, :]
            inside = (position >= lo[c0:c1, None]) & (position < hi[c0:c1, None])
            wide.write(products[inside].astype("<f4").tobytes())
            squares = np.where(inside, products.astype(np.float64) ** 2, 0.0)
            scores[c0:c1] = squares.sum(axis=1)
            kept_columns, kept_rows = short[c0:c1], short[r0:r1]
            if kept_columns.any():
                narrow_inside = inside[kept_columns][:, kept_rows]
                found = narrow_inside.sum(axis=1)
                own = rank[c0:c1][kept_columns]
                if not np.array_equal(found, short_hi[own] - short_lo[own]):
                    raise AssertionError("the snv_indel band is not the with_sv band's restriction to its records.")
                narrow.write(products[kept_columns][:, kept_rows][narrow_inside].astype("<f4").tobytes())
                short_scores[own] = squares[kept_columns][:, kept_rows].sum(axis=1)
            del products, inside, squares
    found = {"means": means, "beta": beta, "se": se, "beta_part": beta_part, "se_part": se_part}
    return {
        "with_sv": {**found, "lo": lo, "hi": hi, "ld_scores": scores},
        "snv_indel": {**{name: values[short] for name, values in found.items()}, "lo": short_lo, "hi": short_hi, "ld_scores": short_scores},
    }


def p_values(beta: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Two-sided normal p of beta / se, 2 where it is not finite (a record with no variance among the rows)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.abs(beta / se)
    return np.where(np.isfinite(z), 2.0 * stats.norm.sf(z), 2.0)


def clump(p: np.ndarray, lo: np.ndarray, hi: np.ndarray, values: np.ndarray, r2: float = CLUMP_R2) -> np.ndarray:
    """Greedy clumping on a band (``gwas_and_bands``' layout in ``values``): in rising p, each record not yet removed is
    an index record and removes the records of its band column whose r^2 with it is above ``r2``. Returns the index mask."""
    offsets = np.r_[0, np.cumsum(hi - lo)]
    removed = np.zeros(p.shape[0], dtype=bool)
    index = np.zeros(p.shape[0], dtype=bool)
    for j in np.argsort(p, kind="stable").tolist():
        if removed[j]:
            continue
        index[j] = True
        column = np.asarray(values[offsets[j] : offsets[j + 1]], dtype=np.float64)
        removed[lo[j] : hi[j]] |= column * column > r2
    return index


def ct_effects(beta: np.ndarray, p: np.ndarray, index: np.ndarray) -> np.ndarray:
    """[records, THRESHOLDS]: each index record's beta where its p is at most the threshold, else 0."""
    return np.column_stack([np.where(index & (p <= threshold), beta, 0.0) for threshold in THRESHOLDS])


def tuning_slice(families: np.ndarray, seed: int) -> np.ndarray:
    """Per training person, whether their family is in the tuning slice: each family is drawn with probability
    TUNING_FRACTION by ``seed``, so relatives fall on one side and the slice is the same for both arms."""
    labels, inverse = np.unique(np.asarray(families), return_inverse=True)
    return np.random.default_rng(seed).random(labels.shape[0])[inverse] < TUNING_FRACTION


def _seed(seed: int, trait: str) -> list[int]:
    return [int(seed), int.from_bytes(hashlib.sha256(trait.encode()).digest()[:4], "little")]


def write_inputs(folder: pathlib.Path, arm: Mapping[str, np.ndarray], count: int, part_count: int, cores: int) -> None:
    """The R script's inputs beside the arm's band (aou_fastpgs.R). A record with no variance among a GWAS's rows (none
    at the run's frequency floor, which counts every store sample) would give R a NaN: its beta is 0 there, with se 1."""
    for beta, se, name in (("beta", "se", "sumstats.bin"), ("beta_part", "se_part", "sumstats_part.bin")):
        usable = np.isfinite(arm[beta]) & np.isfinite(arm[se]) & (arm[se] > 0)
        if not usable.all():
            _log(f"{folder.name} {name}: {int((~usable).sum())} records with no variance, beta 0")
        np.column_stack([np.where(usable, arm[beta], 0.0), np.where(usable, arm[se], 1.0)]).astype("<f8").tofile(folder / name)
    arm["lo"].astype("<i4").tofile(folder / "ld_lo.bin")
    arm["hi"].astype("<i4").tofile(folder / "ld_hi.bin")
    arm["ld_scores"].astype("<f8").tofile(folder / "ld_scores.bin")
    (folder / "meta.txt").write_text(f"{arm['lo'].shape[0]} {count} {part_count} {cores} {CHUNK}\n")


def r_outputs(folder: pathlib.Path, count: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """LDpred2-auto's effects (NaN where no chain passed the filter), lassosum2's [records, grid] and the grid's labels."""
    auto = np.fromfile(folder / "beta_auto.bin", dtype="<f8")
    lasso = np.fromfile(folder / "beta_lasso.bin", dtype="<f8")
    grid = [line.split("\t") for line in (folder / "lasso_grid.tsv").read_text().splitlines()[1:]]
    lasso = lasso.reshape(len(grid), count).T
    return auto, lasso, [f"delta={delta} lambda={float(value):.4g}" for value, delta, *_ in grid]


class Units:
    """Each trait's fitted effects, kept in ``directory`` (the ops mount: beside the run's mirror) as they are made, one
    NPZ named by the digest of what they were fitted from, written aside and renamed into place."""

    FILES = ("means", "effects", "labels")

    def __init__(self, directory: pathlib.Path, identity: Mapping[str, object]) -> None:
        self.directory, self.identity = pathlib.Path(directory), dict(identity)

    def path(self, **fit: object) -> pathlib.Path:
        content = json.dumps({**self.identity, **fit}, sort_keys=True)
        return self.directory / f"{hashlib.sha256(content.encode()).hexdigest()}.npz"

    def read(self, path: pathlib.Path) -> dict[str, np.ndarray] | None:
        try:
            with np.load(path, allow_pickle=False) as arrays:
                return {name: np.array(arrays[name]) for name in self.FILES}
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            return None

    def write(self, path: pathlib.Path, arrays: Mapping[str, np.ndarray]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        aside = path.with_name(f".{path.name}.{os.getpid()}")
        with open(aside, "wb") as handle:
            np.savez(handle, **{name: arrays[name] for name in self.FILES})
        os.replace(aside, path)


def code_digest() -> str:
    code = hashlib.sha256()
    for source in (pathlib.Path(__file__), _SCRIPT, *(pathlib.Path(module.__file__) for module in (ldpred2_auto_allvar, ldpred2_auto, sbayesrc))):
        code.update(source.read_bytes())
    return code.hexdigest()


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values)).hexdigest()


def fit_effects(cm: np.ndarray, short: np.ndarray, train: MemoryView, part: MemoryView, cores: int) -> dict[str, np.ndarray]:
    """Every candidate's effects over the with_sv records (0 on the SVs for the snv_indel arm), with the training means
    and one label per candidate, "arm|method|parameter"."""
    effects, labels = [], []
    with tempfile.TemporaryDirectory(prefix="fastpgs_", dir=os.environ.get("TMPDIR"), ignore_cleanup_errors=True) as directory:
        folders = {arm: pathlib.Path(directory) / arm for arm in ARMS}
        for folder in folders.values():
            folder.mkdir()
        arms = gwas_and_bands(train, part, cm, short, folders)
        _log(f"GWAS and bands written: {cm.shape[0]} with_sv records, {int(short.sum())} snv_indel")
        places = {"with_sv": np.arange(cm.shape[0]), "snv_indel": np.flatnonzero(short)}
        clumped = {}
        for arm in ARMS:
            found = arms[arm]
            p = p_values(found["beta_part"], found["se_part"])
            values = np.memmap(folders[arm] / "ld_values.bin", dtype="<f4", mode="r")
            clumped[arm] = ct_effects(found["beta_part"], p, clump(p, found["lo"], found["hi"], values))
            del values
            write_inputs(folders[arm], found, train.n_samples, part.n_samples, max(1, cores // len(ARMS)))
        _log("clumped: " + ", ".join(f"{arm} {int(np.any(clumped[arm] != 0, axis=1).sum())} index records at p <= 1" for arm in ARMS))
        single = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
        sessions = {arm: subprocess.Popen(["Rscript", str(_SCRIPT), str(folders[arm])], env=os.environ | single) for arm in ARMS}
        failed = [arm for arm, session in sessions.items() if session.wait() != 0]
        if failed:
            raise RuntimeError(f"aou_fastpgs.R failed for {failed}")
        _log("R sessions done")
        for arm in ARMS:
            auto, lasso, grid = r_outputs(folders[arm], places[arm].shape[0])
            for method, columns, names in (("ldpred2_auto", auto[:, None], ["auto"]), ("lassosum2", lasso, grid),
                                           ("ct", clumped[arm], [f"p<={threshold:g}" for threshold in THRESHOLDS])):
                wide = np.zeros((cm.shape[0], columns.shape[1]), dtype=np.float32)
                wide[places[arm]] = columns
                effects.append(wide)
                labels.extend(f"{arm}|{method}|{name}" for name in names)
        means = arms["with_sv"]["means"]
    return {"means": means, "effects": np.hstack(effects), "labels": np.array(labels)}


def candidate_scores(codes: np.ndarray, columns: np.ndarray, means: np.ndarray, effects: np.ndarray) -> np.ndarray:
    """[columns, candidates]: sum over the records of (dosage - training mean) x effect, for each of ``columns``."""
    total = np.zeros((columns.shape[0], effects.shape[1]))
    nonfinite = ~np.all(np.isfinite(effects), axis=0)
    clean = np.where(np.isfinite(effects), effects, 0.0).astype(np.float32)
    for start in range(0, codes.shape[0], CHUNK):
        stop = min(start + CHUNK, codes.shape[0])
        dosages = np.take(codes[start:stop], columns, axis=1).astype(np.float32) / np.float32(127.0) - means[start:stop, None].astype(np.float32)
        total += (dosages.T @ clean[start:stop]).astype(np.float64)
    total[:, nonfinite] = np.nan
    return total


def choose(labels: np.ndarray, tuning: np.ndarray, target: np.ndarray, covariates: np.ndarray) -> tuple[dict[tuple[str, str], int], np.ndarray]:
    """Per (arm, method), the candidate with the largest partial r^2 on the tuning rows (``tuning`` [rows, candidates]),
    and every candidate's tuning partial r^2 (NaN for a constant or non-finite score)."""
    r2 = np.array([_partial_r2(target, tuning[:, index], covariates) for index in range(tuning.shape[1])])
    chosen = {}
    for arm in ARMS:
        for method in METHODS:
            members = np.flatnonzero([label.startswith(f"{arm}|{method}|") for label in labels.tolist()])
            usable = members[np.isfinite(r2[members])]
            if usable.size:
                chosen[(arm, method)] = int(usable[np.argmax(r2[usable])])
    return chosen, r2


def _count(count: int) -> int | str:
    return count if count == 0 or count >= MINIMUM_REPORTED_PARTICIPANTS else f"<{MINIMUM_REPORTED_PARTICIPANTS}"


def evaluated_model(cohort, trait: int) -> tuple[int, np.ndarray]:
    """The trait's one model with held-out rows and those rows (observed targets only); a cross-fitted trait has
    several, which this baseline does not fit."""
    models = np.flatnonzero(np.asarray(cohort.model_traits) == trait)
    found = [(int(model), np.flatnonzero(cohort.held_out[:, model] & np.isfinite(cohort.targets[:, trait]))) for model in models]
    found = [(model, rows) for model, rows in found if rows.size]
    if len(found) != 1:
        raise ValueError(f"{cohort.trait_names[trait]}: {len(found)} models hold out rows; fastpgs fits one split.")
    return found[0]


def fit_trait(store: DosageStore, records: np.ndarray, cm: np.ndarray, short: np.ndarray, cohort, trait: int, seed: int,
              units: Units | None, cores: int) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    """Both arms' three methods for one trait: the aggregate lines and the held-out people's arrays (module docstring)."""
    name = cohort.trait_names[trait]
    binary = cohort.trait_types[trait] == TraitType.BINARY
    model, held = evaluated_model(cohort, trait)
    training = np.flatnonzero(cohort.training[:, model] & np.isfinite(cohort.targets[:, trait]))
    training = training[np.argsort(cohort.store_columns[training], kind="stable")]
    held = held[np.argsort(cohort.store_columns[held], kind="stable")]
    tuning = tuning_slice(cohort.families[training], _seed(seed, name))
    columns = np.union1d(cohort.store_columns[training], cohort.store_columns[held])
    where = {kind: np.searchsorted(columns, cohort.store_columns[rows]) for kind, rows in (("training", training), ("held", held))}
    own = np.flatnonzero(cohort.covariate_columns[trait])

    def view(rows: np.ndarray, places: np.ndarray, codes: np.ndarray) -> MemoryView:
        return MemoryView(codes, places, np.asarray(cohort.covariates[np.ix_(rows, own)], dtype=np.float64), np.asarray(cohort.targets[rows, trait], dtype=np.float64))

    _log(f"{name}: reading {records.shape[0]} records over {columns.shape[0]} people")
    codes = read_codes(store, records, columns)
    _log(f"{name}: codes read")
    path = None if units is None else units.path(
        trait=name, model=model, records=_digest(records), training=_digest(cohort.store_columns[training]), held=_digest(cohort.store_columns[held]),
        tuning=_digest(tuning),
    )
    fitted = None if path is None or units is None else units.read(path)
    if fitted is None:
        fitted = fit_effects(cm, short, view(training, where["training"], codes), view(training[~tuning], where["training"][~tuning], codes), cores)
        if path is not None and units is not None:
            units.write(path, fitted)
    labels = fitted["labels"]
    tuning_rows, scored = training[tuning], np.r_[where["training"][tuning], where["held"]]
    found = candidate_scores(codes, scored, fitted["means"], fitted["effects"])
    del codes
    tuning_scores, held_scores = found[: tuning_rows.shape[0]], found[tuning_rows.shape[0] :]
    chosen, tuning_r2 = choose(labels, tuning_scores, cohort.targets[tuning_rows, trait], np.asarray(cohort.covariates[np.ix_(tuning_rows, own)], dtype=np.float64))
    target, covariates = np.asarray(cohort.targets[held, trait], dtype=np.float64), np.asarray(cohort.covariates[np.ix_(held, own)], dtype=np.float64)
    scores = np.full((len(METHODS), len(ARMS), held.shape[0]), np.nan)
    cases = int(target.sum()) if binary else None
    lines = []
    for (arm, method), index in sorted(chosen.items()):
        scores[METHODS.index(method), ARMS.index(arm)] = held_scores[:, index]
    for arm in ARMS:
        for method in METHODS:
            index = chosen.get((arm, method))
            line: dict[str, object] = {
                "trait": name, "arm": arm, "method": method, "held_out_rows": _count(held.shape[0]),
                "variants": int(short.sum()) if arm == "snv_indel" else int(short.shape[0]),
                "parameter": None if index is None else labels[index].split("|", 2)[2],
                "nonzero_effects": None if index is None else int(np.count_nonzero(fitted["effects"][:, index])),
                # LDpred2-auto is fitted on the tuning rows too, so its tuning partial r^2 is in-sample and not shown
                "tuning_rows": _count(tuning_rows.shape[0]),
                "tuning_partial_r2": None if index is None or method == "ldpred2_auto" else float(tuning_r2[index]),
                "held_out_partial_r2": None if index is None else _partial_r2(target, held_scores[:, index], covariates),
            }
            if binary:
                line["cases"] = _cases_text(held.shape[0], cases)
                line["held_out_auc"] = None if index is None else auc_components(target, held_scores[:, index]).auc
            lines.append(line)
    arrays = {
        "trait": np.array(name), "binary": np.array(binary), "methods": np.array(METHODS), "arms": np.array(ARMS),
        "research_ids": np.array(cohort.research_ids)[held], "families": np.asarray(cohort.families)[held],
        "ancestry": np.array(cohort.ancestry)[held], "target": target, "covariates": covariates, "scores": scores, "seed": np.array(seed),
    }
    return lines, arrays


def arm_records(table, kept: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The with_sv records in rising genetic position (chromosomes offset by CHROMOSOME_SPACING_CM), their positions and
    which are the snv_indel arm's."""
    wide = np.flatnonzero(np.isfinite(arm_log_reliability(table, kept, "with_sv")))
    narrow = np.isfinite(arm_log_reliability(table, kept, "snv_indel"))
    if np.any(narrow & ~np.isfinite(arm_log_reliability(table, kept, "with_sv"))):
        raise AssertionError("a snv_indel record is not a with_sv record.")
    cm = np.asarray(table.chromosome, dtype=np.float64)[wide] * CHROMOSOME_SPACING_CM + np.asarray(table.genetic_position_cm, dtype=np.float64)[wide]
    if not np.all(np.isfinite(cm)):
        raise ValueError(f"{int(np.sum(~np.isfinite(cm)))} kept records have no genetic position.")
    order = np.argsort(cm, kind="stable")
    return wide[order], cm[order], narrow[wide[order]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="The run config the driver ran (its run directory holds the finished steps).")
    parser.add_argument("--out", required=True, help="The aggregate table to write (one JSON line per trait, arm and method).")
    parser.add_argument("--scores", required=True, type=pathlib.Path, help="The directory of each trait's held-out arrays (<trait>.npz): kept in the workspace.")
    parser.add_argument("--units", type=pathlib.Path, help="Where each trait's fitted effects are kept, and found again after a restart (Units).")
    args = parser.parse_args(argv)
    config = WorkspaceConfig.read(args.config)
    run = _Run(config, workspace_bindings(), detect_compute_budget(), {})
    cohort = _cohort_of(run)
    units = None if args.units is None else Units(args.units, {"format": "aou-fastpgs-v1", "code": code_digest(), "config": pathlib.Path(args.config).read_text()})
    measurement = run.bindings.load_measurement_model(_final_measurement_model(run))
    cores = os.cpu_count() or 1
    lines = []
    args.scores.mkdir(parents=True, exist_ok=True)
    with DosageStore.open(_final_store(run)) as store:
        table = store.variant_table
        filtered = (table, store.n_samples, measurement.log_reliability, config.fit_minimum_minor_allele_frequency, config.fit_minimum_r2, config.fit_variant_sites)
        region = getattr(config, "fit_region", None)
        kept = fit_filtered(*filtered, region) if region is not None else fit_filtered(*filtered)
        records, cm, short = arm_records(table, kept)
        _log(f"records: {records.shape[0]} with_sv, {int(short.sum())} snv_indel")
        for trait in range(len(cohort.trait_names)):
            found, arrays = fit_trait(store, records, cm, short, cohort, trait, int(config.seed), units, cores)
            for line in found:
                lines.append(json.dumps(line))
                print(lines[-1], flush=True)
            with open(args.scores / f"{cohort.trait_names[trait]}.npz", "wb") as handle:
                np.savez(handle, **arrays)
    pathlib.Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
