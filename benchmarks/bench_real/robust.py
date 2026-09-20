"""Robust scoring of bench-real's out-of-fold predictions (the fixes in critique/CRITIQUE_REAL.md, items 1-4 and 12).

It reads the harness's own outputs (``<results>/<method>/<design>/<tag>.{genes.tsv,truth.npy,<set>.predictions*.npy}``)
and changes nothing upstream.

Metrics, per gene and held-out superpopulation:
  r2        the squared Pearson correlation, as in report.py.
  r2_floor  r2 minus its null expectation, rescaled. For a prediction independent of the truth over n people,
            r2 ~ Beta(1/2, (n-2)/2) when either vector is spherically symmetric after centring, so E[r2] = 1/(n-1);
            r2_floor = (r2 - 1/(n-1)) / (1 - 1/(n-1)) is 0 in expectation under the null and 1 at r2 = 1.
  oos_r2    out-of-sample R2 = 1 - sum (y - yhat)^2 / sum (y - ybar)^2 over the held-out group. It also charges the
            prediction's scale and location, which r2 ignores.
A pooled value is the mean over genes of the mean over the five held-out groups; under loso every person is held out once.

Uncertainty is the pigeonhole bootstrap for a crossed genes x people design (Owen 2007, Ann. Appl. Stat. 1:541).
Each replicate independently resamples, with replacement:
  - families within every held-out group, using the 1kGP FamilyID (a family of one is a person); and
  - chromosomes, because genes on one chromosome share variants.
Predictions are held fixed, so this covers the sampling of test people and of genes, not of training sets.
Intervals are bootstrap percentiles at the conventional 95% level. A replicate count B gives an SE whose Monte Carlo
relative error is about 1/sqrt(2B); the summary reports it.

SV gain split: take f, the snv_sv prediction; m, the same prediction with the SV columns set to their training means
(the harness's predictions_without_sv); and s, the snv prediction. Then, exactly,
  r2(f) - r2(s) = [r2(f) - r2(m)]  (SV part)  +  [r2(m) - r2(s)]  (SNV refit).
The same holds for every metric and for any other SV feature set.

Per-gene SV tests: a one-sided Wald test of each gene's pooled gain and SV part, using the SE from the family bootstrap
(people only; the gene is fixed). Benjamini-Hochberg q values (Benjamini & Hochberg 1995) are computed over the genes of
one method, design and feature set.
"""
import argparse
import dataclasses
import hashlib
import json
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

GROUPS = ("AFR", "AMR", "EAS", "EUR", "SAS")
METRICS = ("r2", "r2_floor", "oos_r2")
MASKED = "/masked"
BASELINE_SET = "snv"
LEVEL = 0.95  # the conventional two-sided interval level; it also sets the conventional q <= 1 - LEVEL count


def seed_from_name(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")


@dataclasses.dataclass
class Arm:
    """One method's out-of-fold predictions for one design and feature set; feature_set ends in /masked for f-with-SVs-at-mean."""
    method: str
    design: str
    feature_set: str
    genes: pd.DataFrame
    prediction: np.ndarray
    truth: np.ndarray

    @property
    def key(self):
        return self.method, self.design, self.feature_set


def _append(arms: dict, key, genes: pd.DataFrame, prediction: np.ndarray, truth: np.ndarray):
    if key in arms:
        old = arms[key]
        genes = pd.concat([old.genes, genes], ignore_index=True)
        prediction, truth = np.vstack([old.prediction, prediction]), np.vstack([old.truth, truth])
    if genes["gene_id"].duplicated().any():
        raise ValueError(f"{key}: a gene appears in two result files")
    arms[key] = Arm(*key, genes=genes.reset_index(drop=True), prediction=prediction, truth=truth)


def load_arms(results_dirs, methods, designs):
    """Every (method, design, feature set) found under the result directories, plus a /masked arm per non-baseline set."""
    arms = {}
    for results in results_dirs:
        for method in methods:
            for design in designs:
                directory = pathlib.Path(results) / method / design
                for genes_file in sorted(directory.glob("*.genes.tsv")):
                    tag = genes_file.name.removesuffix(".genes.tsv")
                    genes = pd.read_csv(genes_file, sep="\t")[["gene_id", "chrom"]]
                    truth = np.load(directory / f"{tag}.truth.npy").astype(np.float64)
                    for path in sorted(directory.glob(f"{tag}.*.predictions.npy")):
                        feature_set = path.name.removeprefix(f"{tag}.").removesuffix(".predictions.npy")
                        _append(arms, (method, design, feature_set), genes, np.load(path).astype(np.float64), truth)
                        masked = directory / f"{tag}.{feature_set}.predictions_without_sv.npy"
                        if feature_set != BASELINE_SET and masked.exists():
                            _append(arms, (method, design, feature_set + MASKED), genes, np.load(masked).astype(np.float64), truth)
    return arms


def null_r2(count: int) -> float:
    """E[r2] for a prediction independent of the truth over count people."""
    return 1.0 / (count - 1)


def group_metrics(weights: np.ndarray, prediction: np.ndarray, truth: np.ndarray):
    """Metrics per replicate and gene. weights is (B, n) multiplicities; prediction and truth are (genes, n)."""
    count = prediction.shape[1]
    centred_prediction = prediction - prediction.mean(axis=1, keepdims=True)
    centred_truth = truth - truth.mean(axis=1, keepdims=True)
    total = weights.sum(axis=1, keepdims=True)
    mean_prediction = weights @ centred_prediction.T / total
    mean_truth = weights @ centred_truth.T / total
    covariance = weights @ (centred_prediction * centred_truth).T / total - mean_prediction * mean_truth
    prediction_variance = np.maximum(weights @ (centred_prediction ** 2).T / total - mean_prediction ** 2, 0.0)
    truth_variance = np.maximum(weights @ (centred_truth ** 2).T / total - mean_truth ** 2, 0.0)
    product = prediction_variance * truth_variance
    r2 = np.divide(covariance ** 2, product, out=np.zeros_like(covariance), where=product > 0)
    squared_error = weights @ ((truth - prediction) ** 2).T / total
    oos = 1.0 - np.divide(squared_error, truth_variance, out=np.full_like(squared_error, np.nan), where=truth_variance > 0)
    floor = null_r2(count)
    return {"r2": r2, "r2_floor": (r2 - floor) / (1.0 - floor), "oos_r2": oos}


def family_weights(families: np.ndarray, replicates: int, generator: np.random.Generator):
    """(replicates, n) multiplicities from resampling the group's families with replacement."""
    labels, family_of_person = np.unique(families, return_inverse=True)
    draws = generator.integers(0, len(labels), size=(replicates, len(labels)))
    counts = np.zeros((replicates, len(labels)))
    np.add.at(counts, (np.arange(replicates)[:, None], draws), 1.0)
    return counts[:, family_of_person]


@dataclasses.dataclass
class ArmStatistics:
    """Point values and bootstrap replicates of every metric, pooled over groups per gene, and pooled over genes per group."""
    genes: pd.DataFrame
    gene_point: dict   # metric -> (genes,)
    gene_boot: dict    # metric -> (B, genes) float32, people resampled, genes fixed
    group_point: dict  # metric -> (groups,)
    group_boot: dict   # metric -> (B, groups), people and chromosomes resampled


def arm_statistics(arms: dict, samples: pd.DataFrame, replicates: int, chunk: int, seed_name: str):
    """ArmStatistics for every arm, all from the same replicate draws so that any two arms can be paired."""
    group_index = {group: np.flatnonzero(samples["Superpopulation"].to_numpy() == group) for group in GROUPS}
    chromosomes = sorted({chrom for arm in arms.values() for chrom in arm.genes["chrom"]})
    chrom_code = {chrom: code for code, chrom in enumerate(chromosomes)}
    generator = np.random.default_rng(seed_from_name(seed_name))
    chrom_counts = np.zeros((replicates, len(chromosomes)))
    np.add.at(chrom_counts, (np.arange(replicates)[:, None], generator.integers(0, len(chromosomes), size=(replicates, len(chromosomes)))), 1.0)
    person_weights = {group: family_weights(samples["FamilyID"].to_numpy()[index], replicates, generator) for group, index in group_index.items()}
    result = {}
    for key, arm in arms.items():
        codes = arm.genes["chrom"].map(chrom_code).to_numpy()
        gene_boot = {metric: np.zeros((replicates, len(arm.genes)), dtype=np.float32) for metric in METRICS}
        gene_point = {metric: np.zeros(len(arm.genes)) for metric in METRICS}
        group_boot = {metric: np.zeros((replicates, len(GROUPS))) for metric in METRICS}
        group_point = {metric: np.zeros(len(GROUPS)) for metric in METRICS}
        for position, (group, index) in enumerate(group_index.items()):
            prediction, truth = arm.prediction[:, index], arm.truth[:, index]
            if not (np.isfinite(prediction).all() and np.isfinite(truth).all()):
                raise ValueError(f"{key}: missing out-of-fold values for {group}")
            point = group_metrics(np.ones((1, len(index))), prediction, truth)
            for metric in METRICS:
                gene_point[metric] += point[metric][0] / len(GROUPS)
                group_point[metric][position] = point[metric][0].mean()
            for start in range(0, replicates, chunk):
                rows = slice(start, min(start + chunk, replicates))
                values = group_metrics(person_weights[group][rows], prediction, truth)
                gene_weight = chrom_counts[rows][:, codes]
                for metric in METRICS:
                    gene_boot[metric][rows] += (values[metric] / len(GROUPS)).astype(np.float32)
                    group_boot[metric][rows, position] = (gene_weight * values[metric]).sum(axis=1) / gene_weight.sum(axis=1)
        result[key] = ArmStatistics(arm.genes, gene_point, gene_boot, group_point, group_boot)
    return result, chrom_counts, chrom_code


def pooled(point: np.ndarray, boot: np.ndarray, codes: np.ndarray, chrom_counts: np.ndarray):
    """The mean over genes and its crossed-bootstrap replicates."""
    gene_weight = chrom_counts[:, codes]
    return float(point.mean()), (gene_weight * boot).sum(axis=1) / gene_weight.sum(axis=1)


def summarize(point: float, boot: np.ndarray):
    tail = (1.0 - LEVEL) / 2.0
    return {"estimate": point, "se": float(np.std(boot, ddof=1)), "ci_low": float(np.quantile(boot, tail)), "ci_high": float(np.quantile(boot, 1.0 - tail)),
            "z": point / float(np.std(boot, ddof=1)) if np.std(boot) > 0 else np.nan}


def aligned(statistics: dict, keys):
    """Gene-aligned (point, boot) per key, on the genes every key has, plus the chromosome of each gene."""
    common = set.intersection(*(set(statistics[key].genes["gene_id"]) for key in keys))
    frames = []
    for key in keys:
        genes = statistics[key].genes
        order = np.flatnonzero(genes["gene_id"].isin(common).to_numpy())
        order = order[np.argsort(genes["gene_id"].to_numpy()[order])]
        frames.append(order)
    reference = statistics[keys[0]].genes.iloc[frames[0]]
    return frames, reference.reset_index(drop=True)


def contrast(statistics: dict, plus, minus, metric: str):
    """(genes, point per gene, boot per gene) of plus - minus on their common genes."""
    (plus_rows, minus_rows), genes = aligned(statistics, [plus, minus])
    point = statistics[plus].gene_point[metric][plus_rows] - statistics[minus].gene_point[metric][minus_rows]
    boot = statistics[plus].gene_boot[metric][:, plus_rows].astype(np.float64) - statistics[minus].gene_boot[metric][:, minus_rows]
    return genes, point, boot


def planned_contrasts(keys):
    """Name -> (plus, minus): SV gain and its split within each method, methods against each other, and random5 - loso."""
    present = set(keys)
    planned = {}
    for method, design, feature_set in sorted(present):
        if feature_set.endswith(MASKED) or feature_set == BASELINE_SET:
            continue
        baseline, masked = (method, design, BASELINE_SET), (method, design, feature_set + MASKED)
        if baseline in present:
            planned[f"{method}/{design}: {feature_set} - {BASELINE_SET}"] = ((method, design, feature_set), baseline)
        if masked in present:
            planned[f"{method}/{design}: {feature_set} SV part"] = ((method, design, feature_set), masked)
            if baseline in present:
                planned[f"{method}/{design}: {feature_set} SNV refit"] = (masked, baseline)
    methods = sorted({key[0] for key in present})
    for design in sorted({key[1] for key in present}):
        for feature_set in sorted({key[2] for key in present if not key[2].endswith(MASKED)}):
            for first in methods:
                for second in methods:
                    if first < second and (first, design, feature_set) in present and (second, design, feature_set) in present:
                        planned[f"{design}/{feature_set}: {first} - {second}"] = ((first, design, feature_set), (second, design, feature_set))
    for method, design, feature_set in sorted(present):
        if design == "loso" and (method, "random5", feature_set) in present:
            planned[f"{method}/{feature_set}: random5 - loso"] = ((method, "random5", feature_set), (method, "loso", feature_set))
    return planned


def gene_tests(statistics: dict, keys, metric: str):
    """Per-gene one-sided Wald tests of the SV gain and SV part, with BH q values, for every non-baseline feature set."""
    rows = []
    for method, design, feature_set in keys:
        if feature_set.endswith(MASKED) or feature_set == BASELINE_SET:
            continue
        full, masked, baseline = (method, design, feature_set), (method, design, feature_set + MASKED), (method, design, BASELINE_SET)
        if masked not in statistics or baseline not in statistics:
            continue
        (full_rows, masked_rows, baseline_rows), genes = aligned(statistics, [full, masked, baseline])
        table = genes.assign(method=method, design=design, feature_set=feature_set, metric=metric)
        for name, (plus, plus_rows, minus, minus_rows) in {"gain": (full, full_rows, baseline, baseline_rows),
                                                           "sv_part": (full, full_rows, masked, masked_rows),
                                                           "refit": (masked, masked_rows, baseline, baseline_rows)}.items():
            point = statistics[plus].gene_point[metric][plus_rows] - statistics[minus].gene_point[metric][minus_rows]
            boot = statistics[plus].gene_boot[metric][:, plus_rows].astype(np.float64) - statistics[minus].gene_boot[metric][:, minus_rows]
            se = boot.std(axis=0, ddof=1)
            table[name], table[f"{name}_se"] = point, se
            if name == "refit":
                continue
            z = np.divide(point, se, out=np.full_like(point, np.nan), where=se > 0)
            p = np.where(se > 0, stats.norm.sf(z), np.where(point == 0, 1.0, np.nan))
            table[f"{name}_p"] = p
            table[f"{name}_bootstrap_share_at_or_below_zero"] = (boot <= 0).mean(axis=0)
            q = np.full_like(p, np.nan)
            finite = np.isfinite(p)
            if finite.any():
                q[finite] = stats.false_discovery_control(p[finite], method="bh")
            table[f"{name}_q"] = q
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def sv_weight_share(arms: dict):
    """Per gene, var(f - m) / var(f) over everyone: the share of the SV-inclusive score carried by the SV columns."""
    rows = []
    for (method, design, feature_set), arm in arms.items():
        if feature_set.endswith(MASKED) or (method, design, feature_set + MASKED) not in arms:
            continue
        masked = arms[(method, design, feature_set + MASKED)]
        variance = arm.prediction.var(axis=1)
        share = np.divide((arm.prediction - masked.prediction).var(axis=1), variance, out=np.zeros_like(variance), where=variance > 0)
        rows.append(arm.genes.assign(method=method, design=design, feature_set=feature_set, sv_share_of_score_variance=share))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run(results_dirs, dataset_dir, methods, designs, replicates, chunk, out_dir):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(pathlib.Path(dataset_dir) / "samples.tsv", sep="\t")
    arms = load_arms(results_dirs, methods, designs)
    if not arms:
        raise ValueError("no results found")
    statistics, chrom_counts, chrom_code = arm_statistics(arms, samples, replicates, chunk, "bench-real/robust/bootstrap")

    per_gene = []
    for key, value in statistics.items():
        per_gene.append(value.genes.assign(method=key[0], design=key[1], feature_set=key[2], **{metric: value.gene_point[metric] for metric in METRICS}))
    pd.concat(per_gene, ignore_index=True).to_csv(out / "per_gene_pooled_over_groups.tsv.gz", sep="\t", index=False)

    arm_rows, group_rows = [], []
    for key, value in statistics.items():
        codes = value.genes["chrom"].map(chrom_code).to_numpy()
        for metric in METRICS:
            point, boot = pooled(value.gene_point[metric], value.gene_boot[metric].astype(np.float64), codes, chrom_counts)
            arm_rows.append({"method": key[0], "design": key[1], "feature_set": key[2], "metric": metric, "genes": len(value.genes), **summarize(point, boot)})
            for position, group in enumerate(GROUPS):
                group_rows.append({"method": key[0], "design": key[1], "feature_set": key[2], "metric": metric, "group": group,
                                   **summarize(value.group_point[metric][position], value.group_boot[metric][:, position])})
            if key[1] == "loso":
                others = [position for position, group in enumerate(GROUPS) if group != "AFR"]
                afr = GROUPS.index("AFR")
                ratio_point = value.group_point[metric][afr] / value.group_point[metric][others].mean()
                ratio_boot = value.group_boot[metric][:, afr] / value.group_boot[metric][:, others].mean(axis=1)
                group_rows.append({"method": key[0], "design": key[1], "feature_set": key[2], "metric": metric, "group": "AFR / mean(others)",
                                   **summarize(ratio_point, ratio_boot)})
    pd.DataFrame(arm_rows).to_csv(out / "pooled.tsv", sep="\t", index=False)
    pd.DataFrame(group_rows).to_csv(out / "per_group.tsv", sep="\t", index=False)

    contrast_rows, leave_rows = [], []
    for name, (plus, minus) in planned_contrasts(statistics).items():
        for metric in METRICS:
            genes, point, boot = contrast(statistics, plus, minus, metric)
            codes = genes["chrom"].map(chrom_code).to_numpy()
            estimate, replicate = pooled(point, boot, codes, chrom_counts)
            contrast_rows.append({"contrast": name, "metric": metric, "genes": len(genes), **summarize(estimate, replicate)})
            for chrom in sorted(set(genes["chrom"])):
                keep = genes["chrom"].to_numpy() != chrom
                leave_rows.append({"contrast": name, "metric": metric, "left_out": chrom, "genes": int(keep.sum()), "estimate": float(point[keep].mean())})
    pd.DataFrame(contrast_rows).to_csv(out / "contrasts.tsv", sep="\t", index=False)
    pd.DataFrame(leave_rows).to_csv(out / "leave_one_chromosome_out.tsv", sep="\t", index=False)

    tests = pd.concat([gene_tests(statistics, list(statistics), metric) for metric in METRICS], ignore_index=True)
    weights = sv_weight_share(arms)
    if len(tests) and len(weights):
        tests = tests.merge(weights, on=["gene_id", "chrom", "method", "design", "feature_set"], how="left")
    tests.to_csv(out / "sv_gene_tests.tsv.gz", sep="\t", index=False)

    level = 1.0 - LEVEL
    discoveries = {}
    if len(tests):
        for (method, design, feature_set, metric), group in tests.groupby(["method", "design", "feature_set", "metric"]):
            discoveries[f"{method}/{design}/{feature_set}/{metric}"] = {"genes": len(group), "gain_q_at_or_below": int((group["gain_q"] <= level).sum()),
                                                                        "sv_part_q_at_or_below": int((group["sv_part_q"] <= level).sum()),
                                                                        "untestable": int(group["gain_p"].isna().sum())}
    families = {group: int((samples.loc[samples["Superpopulation"] == group, "FamilyID"].value_counts() > 1).sum()) for group in GROUPS}
    summary = {"replicates": replicates, "monte_carlo_relative_error_of_se": 1.0 / np.sqrt(2.0 * replicates), "level": LEVEL,
               "q_threshold": level, "multi_member_families_per_group": families, "arms": [list(key) for key in statistics],
               "discoveries": discoveries, "bootstrap": "pigeonhole (Owen 2007): families within held-out groups x chromosomes",
               "note": "predictions are fixed: training-set sampling is not covered"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--designs", nargs="+", default=["loso", "random5"])
    parser.add_argument("--replicates", type=int, required=True, help="bootstrap replicates; the SE's Monte Carlo relative error is about 1/sqrt(2B)")
    parser.add_argument("--chunk", type=int, required=True, help="replicates per matrix product (memory only; results do not depend on it)")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    summary = run(arguments.results, arguments.dataset, arguments.methods, arguments.designs, arguments.replicates, arguments.chunk, arguments.out)
    print(json.dumps(summary["discoveries"], indent=1))


if __name__ == "__main__":
    main()
