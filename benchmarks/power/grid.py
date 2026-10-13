"""The power study's simulation grid: controlled architectures on bench-sim's real-LD cohort (1kGP haplotypes with
their imputed SVs, PREREG.md sections 1-2), both study arms fitted by the method on training sets of several sizes, and the
held-out comparison the workspace report makes, with null and positive controls.

Each grid point fixes the architecture that ``benchmarks/bench_sim/truth.py`` otherwise draws at random: a liability
threshold disease of liability heritability HERITABILITY on the simulated chromosome and prevalence PREVALENCE,
CAUSAL_VARIANTS causal records, a given fraction of them SVs and an SV effect size relative to small variants, Gaussian
effects with equal per-SD variance, no annotation or non-additive effects and homoscedastic noise. Points:

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

Arms follow the workspace (``workspace_pipeline.ARMS``): ``with_sv`` fits every record; ``snv_indel`` is refitted without
bench-sim's SV class (records >= 50 bp or symbolic), keeping its short indels and short repeat indels.

One task is one (point, training size, replicate), both arms fitted (``run``); ``summarize`` collects every finished task
into ``summary.tsv``: per point and size, the mean SV-arm gain in AUC and its SE over replicates, the share of replicates
whose paired DeLong test favours the SV arm at one-sided 0.05, its Monte Carlo uncertainty, and the
positive controls' readouts. Tasks are independent, so the grid runs as one task each (bench-sim's runq, or Batch).
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
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
LENGTH_SLOPE = 0.5
DELETION_FOLD = 3.0
TRAIN_COUNTS = (5_000, 10_000, 20_000, 40_000)
MEASUREMENT_ARM = "glimpse2"
"""bench-sim's measurement mirroring the All of Us imputation (PREREG.md Amendment 1)."""
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
        # Equal per-SD variance at every frequency (alpha = -1), no plateau, no LD dependence.
        "alpha_snv": -1.0, "alpha_sv": -1.0, "plateau": 0.0, "frequency_source": "cohort", "ld_gamma": 0.0,
        "annotation_mode": "none", "annotation_seed": int(seeds[1]),
        "dominance": False, "tr_saturating": False, "link_c": 0.0, "hetero_kappa": 0.0, "t_noise": False,
        "beta_sex": 0.0, "beta_age": 0.0, "beta_batch": 0.0, "ancestry_means": [0.0] * 5,
        "binary": True, "prevalence": PREVALENCE, "effect_seed": int(seeds[1]),
        "sv_log_length_slope": setting.get("sv_log_length_slope", 0.0),
        "sv_deletion_scale_fold": setting.get("sv_deletion_scale_fold", 1.0),
        **setting.get("truth", {}),
    }


def structural_effect_readout(model, cohort: Path) -> dict:
    """The SV arm's fitted SV effects against what the positive controls set: Spearman's rho of |effect| per allele with
    log length, and the median |effect| of deletions over insertions. Descriptive, like the workspace's: records in LD."""
    table = harness.public_variant_table(cohort, MEASUREMENT_ARM)
    scoring = model.scoring
    rows = model.order[np.asarray(scoring.store_rows)]
    magnitude = np.abs(np.asarray(scoring.coefficients) / np.asarray(scoring.signed_scales)) * SIGNED_CODE_OFFSET
    structural = np.asarray(table["cls"])[rows] == truth.SV
    log_length = np.asarray(table["log_sv_length"])[rows]
    change = np.asarray(table["len_change"])[rows]
    deletions, insertions = magnitude[structural & (change < 0)], magnitude[structural & (change > 0)]
    return {
        "structural_records": int(structural.sum()),
        "length_rho": float(stats.spearmanr(log_length[structural], magnitude[structural]).statistic) if structural.sum() > 2 else math.nan,
        "deletion_over_insertion": float(np.median(deletions) / np.median(insertions)) if deletions.size and insertions.size and np.median(insertions) > 0 else math.nan,
    }


def _run_task(method: Path, cohort: Path, out: Path, point: str, train_count: int, replicate: int, cores: int) -> dict:
    """One task: the point's replicate scenario (built once and shared by every training size), both arms fitted on
    ``train_count`` training samples drawn by the replicate's seed, scored on bench-sim's test samples, compared."""
    scenario = out / "scenarios" / point / f"replicate{replicate}"
    scenario.parent.mkdir(parents=True, exist_ok=True)
    # Different sample-size tasks share a scenario. Serialize creation to prevent partial truth reads.
    with (scenario.parent / f".{scenario.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        parameters = scenario_parameters(point, replicate, cohort)
        if not (scenario / "scenario.json").is_file():
            truth.write_scenario(parameters, cohort, scenario)
        elif json.loads((scenario / "scenario.json").read_text())["params"] != parameters:
            raise ValueError("Existing scenario has different parameters; use a new output directory")
    task = out / "tasks" / point / f"n{train_count}" / f"replicate{replicate}"
    models = {}
    for arm, dropped in (("snv_indel", SMALL_ARM_DROPPED), ("with_sv", ())):
        models[arm] = harness.run(method, cohort, scenario, task / arm, cores, MEASUREMENT_ARM, dropped, train_count=train_count, train_seed=replicate)
    samples = np.load(cohort / "samples.npz")
    labels = np.load(scenario / "truth.npz")["phenotype"][np.flatnonzero(samples["is_test"])]
    small_score = np.load(task / "snv_indel" / "prediction.npz")["total"]
    sv_score = np.load(task / "with_sv" / "prediction.npz")["total"]
    comparison = paired_delta_auc(labels, small_score, sv_score)
    result = {
        "point": point, "train_count": train_count, "replicate": replicate,
        "snv_indel_auc": harness.auc(labels, small_score), "with_sv_auc": harness.auc(labels, sv_score),
        "delta_auc": comparison.estimate, "z": comparison.z_score, "prediction": "genetic_only",
        "truth": json.loads((scenario / "scenario.json").read_text())["summary"],
        **structural_effect_readout(models["with_sv"], cohort),
    }
    (task / "result.json").write_text(json.dumps(result, indent=1))
    return result


def run(method: Path, cohort: Path, out: Path, point: str, train_count: int, replicate: int, cores: int) -> dict:
    """Record failed fits as well as successful ones; reject duplicate task writers."""
    task = out / "tasks" / point / f"n{train_count}" / f"replicate{replicate}"
    task.mkdir(parents=True, exist_ok=True)
    identity = dict(point=point, train_count=train_count, replicate=replicate, method=str(method))

    def status(value, **details):
        temporary = task / "attempt.partial.json"
        temporary.write_text(json.dumps(identity | dict(status=value) | details, indent=1))
        temporary.replace(task / "attempt.json")

    with (task / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (task / "result.json").exists():
            raise FileExistsError("Completed task already exists; use a new directory to rerun changed inputs")
        status("running")
        try:
            result = _run_task(method, cohort, out, point, train_count, replicate, cores)
        except Exception as error:
            status("failed", error_type=type(error).__name__, error=str(error))
            raise
        status("completed")
        return result


SUMMARY_COLUMNS = (
    "point", "train_count", "replicates", "snv_indel_auc", "with_sv_auc", "mean_delta_auc", "delta_auc_se",
    "share_favouring_sv", "valid_comparisons", "invalid_comparisons", "rate_mcse", "rate_lower_95", "rate_upper_95",
    "length_rho", "deletion_over_insertion", "failed_tasks", "unfinished_tasks",
)


def summarize(out: Path) -> Path:
    """Every finished task's result, per (point, training size): the replicates' means, the SE of the mean gain, and the
    share whose one-sided paired DeLong test favours the SV arm at 0.05. Conditions on completed tasks;
    invalid comparisons are counted explicitly and excluded from the rejection-rate denominator."""
    results = [json.loads(path.read_text()) for path in sorted((out / "tasks").glob("*/*/*/result.json"))]
    attempts = [json.loads(path.read_text()) for path in sorted((out / "tasks").glob("*/*/*/attempt.json"))]
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
            incomplete = [row for row in attempts if row['point'] == point and row['train_count'] == train_count and row['replicate'] not in completed]
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
                sum(row['status'] == 'failed' for row in incomplete), sum(row['status'] != 'failed' for row in incomplete),
            ))
    target = out / "summary.tsv"
    with target.open("w", newline="") as handle:
        csv.writer(handle, delimiter="\t").writerows(lines)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    task = commands.add_parser("run", help="one (point, training size, replicate) task, both arms")
    task.add_argument("--method", type=Path, required=True, help="a bench-sim submission, e.g. benchmarks/bench_sim/submissions/svpgs_full.py")
    task.add_argument("--cohort", type=Path, required=True, help="a bench-sim cohort chromosome directory")
    task.add_argument("--out", type=Path, required=True)
    task.add_argument("--point", choices=list(POINTS), required=True)
    task.add_argument("--train-count", type=int, choices=TRAIN_COUNTS, required=True)
    task.add_argument("--replicate", type=int, required=True)
    task.add_argument("--cores", type=int, default=16)
    collect = commands.add_parser("summarize", help="collect every finished task into summary.tsv")
    collect.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "run":
        print(json.dumps(run(arguments.method, arguments.cohort, arguments.out, arguments.point, arguments.train_count, arguments.replicate, arguments.cores)))
    else:
        print(summarize(arguments.out))


if __name__ == "__main__":
    main()
