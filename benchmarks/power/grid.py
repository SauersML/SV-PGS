"""The power study's simulation grid: controlled architectures on bench-sim's real-LD cohort (1kGP haplotypes with
their imputed SVs, PREREG.md sections 1-2), both study arms fitted by the method on training sets of several sizes, and the
held-out comparison the workspace report makes, with null and positive controls.

Each grid point fixes the architecture that ``benchmarks/bench_sim/truth.py`` otherwise draws at random: a liability
threshold disease of liability heritability HERITABILITY on the simulated chromosome and prevalence PREVALENCE,
CAUSAL_VARIANTS causal records, a given fraction of them SVs and an SV effect size relative to small variants, Gaussian
effects whose per-SD variance follows FREQUENCY_ALPHA, no annotation or non-additive effects and homoscedastic noise. Points:

- ``null`` (legacy name): no causal SV. SVs retain their LD with causal small variants. This is a no-SV-causality
  control, NOT necessarily a null of equal predictive accuracy: tagging, representation and regularization can
  change performance. Its rejection rate is not automatically a type I error rate. Its length readout shows what
  the pipeline produces with no true length effect.
- ``sv_5pct``, ``sv_5pct_fold3``, ``sv_20pct``, ``sv_20pct_fold3``: power, by SV causal fraction and SV effect SD fold.
- ``length_positive``: as ``sv_20pct``, with an SV's log effect variance rising LENGTH_SLOPE per SD of log length.
  ``sv_20pct`` is its null: the same architecture with no length effect.
- ``deletion_positive``: as ``sv_20pct``, with a deletion's effect SD DELETION_FOLD times an insertion's.
- ``multi_feature_positive`` combines length, type and annotation-scale effects; ``sv_heavy_tail`` and
  ``sv_two_point`` stress the effect distribution. ``sparse_sv``/``polygenic_sv`` vary expected causal counts,
  while ``rare_disease`` and ``low_heritability`` probe outcome imbalance and weak signal.

Arms follow the workspace (``workspace_pipeline.ARMS``) plus the published-practice competitor: ``with_sv`` fits every
record; ``snv_indel`` is refitted without bench-sim's SV class (records >= 50 bp or symbolic), keeping its short indels
and short repeat indels; ``ldpred2_auto`` is LDpred2-auto on HapMap3-density SNVs, and ``ldpred2_allvar`` is
LDpred2-auto on every measured record of minor allele frequency at least 1%, SVs included. SV-PGS also retains rarer
variants, so this comparison does not hold the input variants fixed. One job is one arm of one (point, training size, replicate) task (``run``); the last of
the core arms to finish writes the task's comparison (a later ``ldpred2_allvar`` adds its own), and ``summarize`` compares
any task left without one and collects every finished task into ``summary.tsv``: per point and size, the mean SV-arm gain
in AUC and its SE over replicates, the share of replicates whose paired DeLong test favours the SV arm at one-sided 0.05,
its Monte Carlo uncertainty, and the positive controls' readouts. Arms and tasks are independent jobs; ``run-group``
fits one SV-PGS arm of every point at a training size and replicate in one job, the points sharing their training
codes and differing only in phenotype.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy import stats

from benchmarks.bench_sim import harness, truth
from benchmarks.bench_sim.records import measured_records
from sv_pgs.fast_scoring import SIGNED_CODE_OFFSET
from sv_pgs.held_out_comparison import paired_delta_auc
from benchmarks.power.uncertainty import rejection_summary

HERITABILITY = 0.1
PREVALENCE = 0.1
CAUSAL_VARIANTS = 200
FREQUENCY_ALPHA = -0.38
"""The frequency dependence of effect sizes, per-allele variance proportional to [2p(1 - p)]^alpha: the mean over 25 UK
Biobank traits (Schoech et al. 2019, Nat Commun 10:790). With causal records drawn uniformly over chr22's records, most
of them rare, it puts about 17% of h2 on variants of minor allele frequency below 1% and about 70% on those of at least
5%, close to whole-genome estimates; alpha = -1 (equal per-SD variance at every frequency) put about 64% below 1%."""
LENGTH_SLOPE = 0.5
DELETION_FOLD = 3.0
TRAIN_COUNTS = (5_000, 10_000, 20_000, 40_000)
MEASUREMENT_ARM = "beagle"
"""bench-sim's Beagle arm: the one imputed arm covering all 50,000 samples (the GLIMPSE2 arm covers only the 2,500-sample
calibration subset, SUBMIT.md). Simple sites are read-model calls and TR and SV records are imputed from the separate panel."""
LDPRED2 = Path(__file__).resolve().parents[1] / "bench_sim" / "submissions" / "ldpred2_auto.py"
LDPRED2_ALLVAR = LDPRED2.with_name("ldpred2_auto_allvar.py")
"""The published-practice competitor, LDpred2-auto on HapMap3-density SNVs, fitted on the same training samples."""
SMALL_ARM_DROPPED = ("SV",)
POINTS = {
    "null": {"sv_causal_fraction": 0.0, "sv_effect_fold": 1.0},
    "sv_5pct": {"sv_causal_fraction": 0.05, "sv_effect_fold": 1.0},
    "sv_5pct_fold3": {"sv_causal_fraction": 0.05, "sv_effect_fold": 3.0},
    "sv_20pct": {"sv_causal_fraction": 0.2, "sv_effect_fold": 1.0},
    "sv_20pct_fold3": {"sv_causal_fraction": 0.2, "sv_effect_fold": 3.0},
    "length_positive": {"sv_causal_fraction": 0.2, "sv_effect_fold": 1.0, "sv_log_length_slope": LENGTH_SLOPE},
    "deletion_positive": {"sv_causal_fraction": 0.2, "sv_effect_fold": 1.0, "sv_deletion_scale_fold": DELETION_FOLD},
    "multi_feature_positive": {"sv_causal_fraction": 0.2, "sv_effect_fold": 1.0, "sv_log_length_slope": LENGTH_SLOPE,
                               "sv_deletion_scale_fold": DELETION_FOLD, "truth": {"annotation_mode": "scale"}},
    "sv_heavy_tail": {"sv_causal_fraction": 0.2, "sv_effect_fold": 1.0,
                      "truth": {"sv_own_shape": True, "sv_shape": "student_t", "sv_shape_params": {"nu": 3.0}}},
    "sv_two_point": {"sv_causal_fraction": 0.2, "sv_effect_fold": 1.0,
                     "truth": {"sv_own_shape": True, "sv_shape": "two_point"}},
    "sparse_sv": {"sv_causal_fraction": 0.2, "sv_effect_fold": 1.0, "causal_variants": 50},
    "polygenic_sv": {"sv_causal_fraction": 0.2, "sv_effect_fold": 1.0, "causal_variants": 2000},
    "rare_disease": {"sv_causal_fraction": 0.2, "sv_effect_fold": 1.0, "truth": {"prevalence": 0.01}},
    "low_heritability": {"sv_causal_fraction": 0.2, "sv_effect_fold": 1.0, "truth": {"h2": 0.01}},
    # Where SVs should help: causal SVs drawn only from those that small-variant haplotypes tag (Beagle realized
    # r^2 >= 0.8), so imputed SVs are accurate, or only from common SVs. The SV class's prior still differs from the
    # small variants' (SVs are causal far more often per record), so a model that fits the SV directly can gain even
    # where SNVs tag it. Appended last: the seeds of the points above depend on their positions.
    "tagged_sv": {"sv_causal_fraction": 0.2, "sv_effect_fold": 1.0, "truth": {"sv_min_realized_r2": 0.8}},
    "tagged_sv_fold3": {"sv_causal_fraction": 0.2, "sv_effect_fold": 3.0, "truth": {"sv_min_realized_r2": 0.8}},
    "common_sv": {"sv_causal_fraction": 0.2, "sv_effect_fold": 1.0, "truth": {"sv_min_frequency": 0.05}},
}


def _odds(probability: float) -> float:
    return probability / (1.0 - probability)


def scenario_parameters(point: str, replicate: int, cohort: Path) -> dict:
    """The truth parameters of ``point``'s replicate on ``cohort``: per-record causal probabilities set so that, in
    expectation, CAUSAL_VARIANTS records are causal and the point's fraction of them are SVs."""
    setting = POINTS[point]
    classes = np.load(cohort / "variants.npz")["cls"]
    structural_count = int(np.sum(classes == truth.SV))
    small_count = int(classes.size - structural_count)
    if min(structural_count, small_count) == 0 or replicate < 0:
        raise ValueError("The grid needs both variant classes and a nonnegative replicate index")
    # the eligible SVs only, where the point restricts which SVs can be causal
    variants = np.load(cohort / "variants.npz")
    eligible_params = {key: setting.get("truth", {}).get(key, 0.0) for key in ("sv_min_frequency", "sv_min_realized_r2")}
    structural_count -= int(truth.structural_ineligible(eligible_params, cohort, variants).sum())
    fraction = setting["sv_causal_fraction"]
    causal_count = setting.get("causal_variants", CAUSAL_VARIANTS)
    small_probability = causal_count * (1.0 - fraction) / small_count
    structural_probability = causal_count * fraction / structural_count
    if not 0 < small_probability < 1 or not 0 <= structural_probability < 1:
        raise ValueError("Causal count exceeds the available records in a variant class")
    seeds = np.random.SeedSequence([list(POINTS).index(point), replicate]).generate_state(2)
    return {
        "seed": int(seeds[0]),
        "h2": HERITABILITY,
        "pi": small_probability,
        # The SVs' causal odds relative to the small variants' (0: no causal SV), and their effect SD fold.
        "sv_mode": "both",
        "sv_probability_fold": _odds(structural_probability) / _odds(small_probability),
        "sv_scale_fold": setting["sv_effect_fold"],
        "tr_mode": "none", "tr_scale_fold": 1.0, "tr_probability_fold": 1.0,
        "shape": "gaussian", "shape_params": {}, "sv_own_shape": False, "sv_shape": "gaussian", "sv_shape_params": {},
        # Per-SD variance proportional to [2p(1 - p)]^(1 + alpha) at FREQUENCY_ALPHA for every class, no plateau, no LD dependence.
        "alpha_snv": FREQUENCY_ALPHA, "alpha_sv": FREQUENCY_ALPHA, "plateau": 0.0, "frequency_source": "cohort", "ld_gamma": 0.0,
        "annotation_mode": "none", "annotation_seed": int(seeds[1]),
        "dominance": False, "tr_saturating": False, "link_c": 0.0, "hetero_kappa": 0.0, "t_noise": False,
        "beta_sex": 0.0, "beta_age": 0.0, "beta_batch": 0.0, "ancestry_means": [0.0] * 5,
        "binary": True, "prevalence": PREVALENCE, "effect_seed": int(seeds[1]),
        "sv_log_length_slope": setting.get("sv_log_length_slope", 0.0),
        "sv_deletion_scale_fold": setting.get("sv_deletion_scale_fold", 1.0),
        **setting.get("truth", {}),
    }


def structural_effect_readout(model, cohort: Path, target: Path, measurement: str = MEASUREMENT_ARM) -> dict:
    """The SV arm's fitted SV effects against what the positive controls set: Spearman's rho of |effect| per allele with
    log length, and the median |effect| of deletions over insertions. Descriptive, like the workspace's: records in LD.
    Every SV record's signed per-allele effect is written to ``target`` with its cohort record index, for the comparison
    with the simulated effects."""
    table = harness.public_variant_table(cohort, measurement)
    scoring = model.scoring
    rows = model.order[np.asarray(scoring.store_rows)]
    effect = np.asarray(scoring.coefficients) / np.asarray(scoring.signed_scales) * SIGNED_CODE_OFFSET
    magnitude = np.abs(effect)
    structural = np.asarray(table["cls"])[rows] == truth.SV
    log_length = np.asarray(table["log_sv_length"])[rows]
    change = np.asarray(table["len_change"])[rows]
    deletions, insertions = magnitude[structural & (change < 0)], magnitude[structural & (change > 0)]
    records = np.flatnonzero(measured_records(cohort))
    np.savez(target, record=records[rows[structural]], per_allele=effect[structural])
    return {
        "structural_records": int(structural.sum()),
        "length_rho": float(stats.spearmanr(log_length[structural], magnitude[structural]).statistic) if structural.sum() > 2 else math.nan,
        "deletion_over_insertion": float(np.median(deletions) / np.median(insertions)) if deletions.size and insertions.size and np.median(insertions) > 0 else math.nan,
    }


ARMS = {"snv_indel": (None, SMALL_ARM_DROPPED), "with_sv": (None, ()), "ldpred2_auto": (LDPRED2, ()), "ldpred2_allvar": (LDPRED2_ALLVAR, ())}
"""Each task's fits: (method, withheld classes). None is the submitted method, so both SV-PGS arms run it; each
LDpred2-auto arm selects its own variants (HapMap3-density SNVs; every class at minor allele frequency >= 1%)."""
CORE_ARMS = ("snv_indel", "with_sv", "ldpred2_auto")
"""The arms a task's comparison waits for; ``ldpred2_allvar`` joins it when it completes."""


def _scenario(cohort: Path, out: Path, point: str, replicate: int) -> Path:
    """The point's replicate scenario, built once and shared by every training size and arm."""
    scenario = out / "scenarios" / point / f"replicate{replicate}"
    scenario.parent.mkdir(parents=True, exist_ok=True)
    # Different tasks share a scenario. Serialize creation to prevent partial truth reads.
    with (scenario.parent / f".{scenario.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        parameters = scenario_parameters(point, replicate, cohort)
        if not (scenario / "scenario.json").is_file():
            truth.write_scenario(parameters, cohort, scenario)
        elif json.loads((scenario / "scenario.json").read_text())["params"] != parameters:
            raise ValueError("Existing scenario has different parameters; use a new output directory")
    return scenario


def _task(out: Path, point: str, train_count: int, replicate: int) -> Path:
    return out / "tasks" / point / f"n{train_count}" / f"replicate{replicate}"


def _completed(folder: Path) -> bool:
    attempt = folder / "attempt.json"
    return attempt.is_file() and json.loads(attempt.read_text())["status"] == "completed"


def _write_attempt(folder: Path, identity: dict, status: str, **details) -> None:
    temporary = folder / "attempt.partial.json"
    temporary.write_text(json.dumps(identity | dict(status=status) | details, indent=1))
    temporary.replace(folder / "attempt.json")


def run(method: Path, cohort: Path, out: Path, point: str, train_count: int, replicate: int, arm: str, cores: int,
        measurement: str = MEASUREMENT_ARM) -> dict | None:
    """One arm of one (point, training size, replicate) task, fitted on ``train_count`` training samples drawn by the
    replicate's seed and scored on bench-sim's test samples, every record observed through ``measurement`` (a bench-sim
    measurement arm; one per output directory). Arms run as separate jobs; the last one to finish writes the task's
    comparison. Failed fits are recorded as well as successful ones; duplicate writers are rejected."""
    scenario = _scenario(cohort, out, point, replicate)
    folder = _task(out, point, train_count, replicate) / arm
    folder.mkdir(parents=True, exist_ok=True)
    arm_method, dropped = ARMS[arm][0] or method, ARMS[arm][1]
    identity = dict(point=point, train_count=train_count, replicate=replicate, arm=arm, method=str(arm_method), measurement=measurement)

    with (folder / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if _completed(folder):
            raise FileExistsError("Completed arm already exists; use a new directory to rerun changed inputs")
        _write_attempt(folder, identity, "running")
        try:
            model = harness.run(arm_method, cohort, scenario, folder, cores, measurement, dropped, train_count=train_count, train_seed=replicate)
            if arm == "with_sv":
                readout = structural_effect_readout(model, cohort, folder / "sv_effects.npz", measurement)
                (folder / "sv_readout.json").write_text(json.dumps(readout, indent=1))
        except Exception as error:
            _write_attempt(folder, identity, "failed", error_type=type(error).__name__, error=str(error))
            raise
        _write_attempt(folder, identity, "completed")
    return compare(cohort, out, point, train_count, replicate)


def run_group(method: Path, cohort: Path, out: Path, points: list[str], train_count: int, replicate: int, arm: str, cores: int,
              measurement: str = MEASUREMENT_ARM) -> list:
    """One SV-PGS arm of several points' tasks at one training size and replicate, fitted together: the points share
    the training samples and their codes and differ only in phenotype, so the method's ``fit_phenotypes`` fits them
    as one group (one store, one set of LD blocks, every pass over the codes serving each model). Each point's arm is
    recorded as ``run`` records it, and a point whose arm another job holds or has completed is left out."""
    if ARMS[arm][0] is not None:
        raise ValueError(f"run-group fits the submitted method's arms, not {arm}")
    if not points or len(set(points)) != len(points) or any(point not in POINTS for point in points):
        raise ValueError("run-group requires distinct, known grid points")
    module = harness.load_method(method)
    dropped = ARMS[arm][1]
    with contextlib.ExitStack() as held:
        members = []
        for point in points:
            scenario = _scenario(cohort, out, point, replicate)
            folder = _task(out, point, train_count, replicate) / arm
            folder.mkdir(parents=True, exist_ok=True)
            lock = held.enter_context((folder / ".lock").open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if not _completed(folder):
                members.append((point, scenario, folder))
        if not members:
            return []
        identities = {point: dict(point=point, train_count=train_count, replicate=replicate, arm=arm, method=str(method),
                                  measurement=measurement, fitted_with=[member[0] for member in members]) for point, _scenario_path, _folder in members}

        for point, _scenario_path, folder in members:
            _write_attempt(folder, identities[point], "running")
        try:
            views = [harness.views(cohort, scenario, cores, measurement, dropped, train_count, replicate)
                     for _point, scenario, _folder in members]
            train, test = views[0]
            started = time.monotonic()
            outcomes = list(module.fit_phenotypes(train, np.column_stack([view[0].phenotype for view in views])))
            if len(outcomes) != len(members):
                raise ValueError(f"Grouped fit returned {len(outcomes)} outcomes for {len(members)} phenotypes")
        except Exception as error:
            for point, _scenario_path, folder in members:
                _write_attempt(folder, identities[point], "failed", error_type=type(error).__name__, error=str(error))
            raise
        seconds = time.monotonic() - started
        errors = []
        for (point, scenario, folder), outcome in zip(members, outcomes):
            if isinstance(outcome, Exception):
                _write_attempt(folder, identities[point], "failed", error_type=type(outcome).__name__, error=str(outcome))
                continue
            try:
                harness.write_prediction(outcome, test, folder, seconds, method=str(method), scenario=str(scenario), cores=cores,
                                         measurement=harness.ARMS[measurement][2], dropped_classes=list(dropped),
                                         train_count=int(train.n_samples), train_seed=replicate, models_fitted_together=len(members))
                if arm == "with_sv":
                    readout = structural_effect_readout(outcome, cohort, folder / "sv_effects.npz", measurement)
                    (folder / "sv_readout.json").write_text(json.dumps(readout, indent=1))
            except Exception as error:
                _write_attempt(folder, identities[point], "failed", error_type=type(error).__name__, error=str(error))
                error.add_note(f"grid point {point}, n={train_count}, replicate={replicate}, arm={arm}")
                errors.append(error)
                continue
            _write_attempt(folder, identities[point], "completed")
    comparisons = [compare(cohort, out, point, train_count, replicate) for point, _scenario_path, _folder in members]
    if errors:
        raise ExceptionGroup("Grouped fit finished with scoring or output failures", errors)
    return comparisons


def prebuild(method: Path, cohort: Path, out: Path, point: str, train_count: int, replicate: int, arm: str,
             measurement: str = MEASUREMENT_ARM) -> Path:
    """Build an SV-PGS arm's training store into the method's store cache once, before the fits that share it start: the
    store holds the training samples' codes and public fields and no phenotype, so every point with the same training
    size, replicate and arm reads the same one."""
    module = harness.load_method(method)
    if not os.environ.get(module.STORE_CACHE_VARIABLE):
        raise ValueError(f"prebuild writes into the store cache; set {module.STORE_CACHE_VARIABLE}")
    train, _test = harness.views(cohort, _scenario(cohort, out, point, replicate), 1, measurement, ARMS[arm][1], train_count, replicate)
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as work:
        path, _order = module.cached_store(train, Path(work))
        return path


def compare(cohort: Path, out: Path, point: str, train_count: int, replicate: int) -> dict | None:
    """The task's held-out comparison once the core arms have completed (None before): each arm's AUC, the SV arm's
    paired DeLong gain over the SNV/indel arm and over each LDpred2-auto arm, the truth summary and the SV effect
    readout. The all-variant LDpred2 arm's fields are added when it completes, before or after the core arms."""
    task = _task(out, point, train_count, replicate)
    if not all(_completed(task / arm) for arm in CORE_ARMS):
        return None
    allvar = _completed(task / "ldpred2_allvar")
    with (task / ".compare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (task / "result.json").exists():
            existing = json.loads((task / "result.json").read_text())
            if not allvar or "ldpred2_allvar_auc" in existing:
                return existing
        scenario = out / "scenarios" / point / f"replicate{replicate}"
        samples = np.load(cohort / "samples.npz")
        labels = np.load(scenario / "truth.npz")["phenotype"][np.flatnonzero(samples["is_test"])]
        arms = CORE_ARMS + (("ldpred2_allvar",) if allvar else ())
        scores = {arm: np.load(task / arm / "prediction.npz")["total"] for arm in arms}
        comparison = paired_delta_auc(labels, scores["snv_indel"], scores["with_sv"])
        against_ldpred2 = paired_delta_auc(labels, scores["ldpred2_auto"], scores["with_sv"])
        result = {
            "point": point, "train_count": train_count, "replicate": replicate,
            "snv_indel_auc": harness.auc(labels, scores["snv_indel"]), "with_sv_auc": harness.auc(labels, scores["with_sv"]),
            "ldpred2_auc": harness.auc(labels, scores["ldpred2_auto"]),
            "delta_auc": comparison.estimate, "z": comparison.z_score,
            "delta_auc_vs_ldpred2": against_ldpred2.estimate, "z_vs_ldpred2": against_ldpred2.z_score,
            "prediction": "genetic_only",
            "fit_seconds": {arm: json.loads((task / arm / "meta.json").read_text())["fit_seconds"] for arm in arms},
            "truth": json.loads((scenario / "scenario.json").read_text())["summary"],
            **json.loads((task / "with_sv" / "sv_readout.json").read_text()),
        }
        if allvar:
            against_allvar = paired_delta_auc(labels, scores["ldpred2_allvar"], scores["with_sv"])
            result |= {
                "ldpred2_allvar_auc": harness.auc(labels, scores["ldpred2_allvar"]),
                "delta_auc_vs_ldpred2_allvar": against_allvar.estimate, "z_vs_ldpred2_allvar": against_allvar.z_score,
            }
        temporary = task / "result.partial.json"
        temporary.write_text(json.dumps(result, indent=1))
        temporary.replace(task / "result.json")
        return result


SUMMARY_COLUMNS = (
    "point", "train_count", "replicates", "snv_indel_auc", "with_sv_auc", "mean_delta_auc", "delta_auc_se",
    "share_favouring_sv", "valid_comparisons", "invalid_comparisons", "rate_mcse", "rate_lower_95", "rate_upper_95",
    "length_rho", "deletion_over_insertion", "failed_tasks", "unfinished_tasks",
)


def summarize(out: Path, cohort: Path) -> Path:
    """Every finished task's result, per (point, training size): the replicates' means, the SE of the mean gain, and the
    share whose one-sided paired DeLong test favours the SV arm at 0.05. Conditions on completed tasks;
    invalid comparisons are counted explicitly and excluded from the rejection-rate denominator. A task whose arms all
    completed but whose comparison is missing (its last arm died between the two writes), or lacks a since-completed
    all-variant LDpred2 arm, is compared here first."""
    for folder in sorted((out / "tasks").glob("*/n*/replicate*")):
        compare(cohort, out, folder.parent.parent.name, int(folder.parent.name[1:]), int(folder.name.removeprefix("replicate")))
    results = [json.loads(path.read_text()) for path in sorted((out / "tasks").glob("*/*/*/result.json"))]
    attempts = [json.loads(path.read_text()) for path in sorted((out / "tasks").glob("*/*/*/*/attempt.json"))]
    def finite_mean(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        return float(values.mean()) if values.size else math.nan
    critical = stats.norm.isf(0.05)
    lines = [SUMMARY_COLUMNS]
    for point in POINTS:
        for train_count in sorted({result["train_count"] for result in results + attempts if result["point"] == point}):
            group = [result for result in results if result["point"] == point and result["train_count"] == train_count]
            completed = {result['replicate'] for result in group}
            incomplete_arms = [row for row in attempts if row['point'] == point and row['train_count'] == train_count and row['replicate'] not in completed]
            failed = {row['replicate'] for row in incomplete_arms if row['status'] == 'failed'}
            unfinished = {row['replicate'] for row in incomplete_arms} - failed
            deltas = np.array([result["delta_auc"] for result in group])
            valid = [result for result in group if np.isfinite(result["delta_auc"]) and np.isfinite(result["z"])]
            rate = rejection_summary(sum(result["z"] > critical for result in valid), len(valid))
            lines.append((
                point, train_count, len(group), f"{finite_mean([result['snv_indel_auc'] for result in group]):.4f}",
                f"{finite_mean([result['with_sv_auc'] for result in group]):.4f}", f"{finite_mean(deltas):.5f}",
                f"{deltas.std(ddof=1) / math.sqrt(deltas.size):.5f}" if deltas.size > 1 else "",
                f"{rate['rate']:.3f}", len(valid), len(group) - len(valid), f"{rate['mcse']:.4f}",
                f"{rate['lower']:.4f}", f"{rate['upper']:.4f}",
                f"{finite_mean([result['length_rho'] for result in group]):.3f}",
                f"{finite_mean([result['deletion_over_insertion'] for result in group]):.3f}",
                len(failed), len(unfinished),
            ))
    target = out / "summary.tsv"
    with target.open("w", newline="") as handle:
        csv.writer(handle, delimiter="\t").writerows(lines)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    task = commands.add_parser("run", help="one arm of one (point, training size, replicate) task")
    task.add_argument("--method", type=Path, required=True, help="a bench-sim submission, e.g. benchmarks/bench_sim/submissions/svpgs_full.py")
    task.add_argument("--cohort", type=Path, required=True, help="a bench-sim cohort chromosome directory")
    task.add_argument("--out", type=Path, required=True)
    task.add_argument("--point", choices=list(POINTS), required=True)
    task.add_argument("--train-count", type=int, choices=TRAIN_COUNTS, required=True)
    task.add_argument("--replicate", type=int, required=True)
    task.add_argument("--arm", choices=list(ARMS), required=True)
    task.add_argument("--cores", type=int, default=16)
    task.add_argument("--measurement", choices=list(harness.ARMS), default=MEASUREMENT_ARM)
    group = commands.add_parser("run-group", help="one SV-PGS arm of several points' tasks, fitted together")
    group.add_argument("--method", type=Path, required=True)
    group.add_argument("--cohort", type=Path, required=True)
    group.add_argument("--out", type=Path, required=True)
    group.add_argument("--points", required=True, help="comma-separated grid points")
    group.add_argument("--train-count", type=int, choices=TRAIN_COUNTS, required=True)
    group.add_argument("--replicate", type=int, required=True)
    group.add_argument("--arm", choices=[arm for arm, (method, _dropped) in ARMS.items() if method is None], required=True)
    group.add_argument("--cores", type=int, default=16)
    group.add_argument("--measurement", choices=list(harness.ARMS), default=MEASUREMENT_ARM)
    store = commands.add_parser("prebuild", help="build an SV-PGS arm's training store into the store cache")
    store.add_argument("--method", type=Path, required=True)
    store.add_argument("--cohort", type=Path, required=True)
    store.add_argument("--out", type=Path, required=True)
    store.add_argument("--point", choices=list(POINTS), required=True)
    store.add_argument("--train-count", type=int, choices=TRAIN_COUNTS, required=True)
    store.add_argument("--replicate", type=int, required=True)
    store.add_argument("--arm", choices=[arm for arm, (method, _dropped) in ARMS.items() if method is None], required=True)
    store.add_argument("--measurement", choices=list(harness.ARMS), default=MEASUREMENT_ARM)
    collect = commands.add_parser("summarize", help="compare finished tasks and collect them into summary.tsv")
    collect.add_argument("--out", type=Path, required=True)
    collect.add_argument("--cohort", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "run":
        print(json.dumps(run(arguments.method, arguments.cohort, arguments.out, arguments.point, arguments.train_count,
                             arguments.replicate, arguments.arm, arguments.cores, arguments.measurement)))
    elif arguments.command == "run-group":
        print(json.dumps(run_group(arguments.method, arguments.cohort, arguments.out, arguments.points.split(","), arguments.train_count,
                                   arguments.replicate, arguments.arm, arguments.cores, arguments.measurement)))
    elif arguments.command == "prebuild":
        print(prebuild(arguments.method, arguments.cohort, arguments.out, arguments.point, arguments.train_count,
                       arguments.replicate, arguments.arm, arguments.measurement))
    else:
        print(summarize(arguments.out, arguments.cohort))


if __name__ == "__main__":
    main()
