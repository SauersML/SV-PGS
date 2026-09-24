"""bench-sim per-chromosome replicates: one independent trait per chromosome, each fitted alone.

A dev scenario's interval on chr22 resamples the test people only: the trait (its causal set, effects and noise) and
the fit stay fixed, so it misses trait-to-trait and fit-to-fit variation. Here every chromosome's cohort (cohort.py's
own seeds: the same people, groups, covariates and 40k/10k split; only the haplotype mosaic differs) carries its own
draw of a dev scenario: the dev scenario's parameters with the effect and annotation seeds drawn afresh per
chromosome, so the causal variants lie on that chromosome alone and their effects and the phenotype noise are
independent between chromosomes. chr22 keeps the dev scenario itself. Each method is fitted on each chromosome alone,
and the chromosomes are the replicates of the across-chromosome interval (a t interval on n_chrom - 1 degrees of
freedom), for each method's genetic r2 and for its paired difference from the reference method.

    python -m benchmarks.bench_sim.per_chrom scenarios --cohort <v7/cohort/chr19> --dev <v7/dev> --out <root>/chr19 --tags 000 005
    python -m benchmarks.bench_sim.per_chrom people --cohort <v7/cohort/chr19> --reference <v7/cohort/chr22>
    python -m benchmarks.bench_sim.per_chrom summary --cohorts <v7/cohort> --scenarios <root> --dev <v7/dev> \\
        --tags 000 005 --results SV-PGS=<dir> SBayesRC=<dir> --out summary.json

A result directory holds <dir>/<chrom>/scenario_<tag>/prediction.npz; the first --results entry is the reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from benchmarks.bench_sim.harness import covariate_matrix
from benchmarks.bench_sim.truth import write_scenario

REFERENCE_CHROMOSOME = "chr22"
"""The chromosome whose replicate is the dev scenario itself (the harness's v1 chromosome)."""
REPLICATE_SEEDS = ("effect_seed", "annotation_seed")
"""The seeds that draw a scenario's trait on its cohort: the causal set, effects, annotation terms and noise."""
PEOPLE_FIELDS = ("group", "group_weights", "proportions", "sex", "age", "batch", "is_test")
"""samples.npz fields drawn from cohort.py's public seed alone, so equal on every chromosome. The realized ancestry
proportions are each chromosome's own mosaic and differ."""
LEVEL = 0.95
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 2026
"""The within-chromosome interval: test people resampled as in the chr22 figure (lead/plot_human.py)."""


def replicate_params(params: dict, chrom: str) -> dict:
    """The dev scenario's parameters on ``chrom``: unchanged on the reference chromosome, else the replicate seeds
    replaced by sha256("<seed>:<chrom>") (the sealed seeds' construction), so each chromosome's trait is its own draw."""
    replicate = dict(params)
    if chrom != REFERENCE_CHROMOSOME:
        for key in REPLICATE_SEEDS:
            replicate[key] = int.from_bytes(hashlib.sha256(f"{params[key]}:{chrom}".encode()).digest()[:4], "little")
    return replicate


def scenario_path(scenarios: Path, dev: Path, chrom: str, tag: str) -> Path:
    return dev / f"scenario_{tag}" if chrom == REFERENCE_CHROMOSOME else scenarios / chrom / f"scenario_{tag}"


def build_scenarios(cohort: Path, dev: Path, out: Path, tags: list[str]) -> list[dict]:
    records = []
    for tag in tags:
        params = json.loads((dev / f"scenario_{tag}" / "scenario.json").read_text())["params"]
        records.append(write_scenario(replicate_params(params, cohort.name), cohort, out / f"scenario_{tag}"))
    return records


def people_differences(cohort: Path, reference: Path) -> list[str]:
    """The PEOPLE_FIELDS on which ``cohort``'s samples differ from ``reference``'s."""
    samples, expected = np.load(cohort / "samples.npz"), np.load(reference / "samples.npz")
    return [name for name in PEOPLE_FIELDS if not np.array_equal(samples[name], expected[name])]


def residualized(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    base = np.column_stack([np.ones(values.shape[0]), covariates])
    values = np.asarray(values, dtype=np.float64)
    return values - base @ np.linalg.lstsq(base, values, rcond=None)[0]


def replicate_accuracy(genetic_value: np.ndarray, prediction: np.ndarray, covariates: np.ndarray,
                       draws: list[np.ndarray]) -> dict:
    """Genetic r2 on one chromosome (harness.genetic_accuracy's: both residualized on [1, covariates] over the test
    people), with its interval over the test people resampled by ``draws``."""
    genetic, predicted = residualized(genetic_value, covariates), residualized(prediction, covariates)

    def r2(rows) -> float:
        return float(np.corrcoef(predicted[rows], genetic[rows])[0, 1] ** 2) if np.any(predicted[rows]) else 0.0

    low, high = np.quantile([r2(rows) for rows in draws], [(1.0 - LEVEL) / 2.0, (1.0 + LEVEL) / 2.0])
    return {"r2": r2(slice(None)), "low": float(low), "high": float(high)}


def across_interval(values) -> dict:
    """Mean over the chromosome replicates and its t interval on n - 1 degrees of freedom (none below two)."""
    values = np.asarray(list(values), dtype=np.float64)
    count = values.shape[0]
    mean = float(values.mean()) if count else float("nan")
    if count < 2:
        return {"mean": mean, "low": float("nan"), "high": float("nan"), "n": count}
    half = float(student_t.ppf((1.0 + LEVEL) / 2.0, count - 1) * values.std(ddof=1) / np.sqrt(count))
    return {"mean": mean, "low": mean - half, "high": mean + half, "n": count}


def summarize_methods(per_chromosome: dict[str, dict[str, float]], reference: str) -> dict:
    """Per method: the across-chromosome interval of its r2, and of its paired difference from ``reference`` over
    the chromosomes both have."""
    summary = {}
    for method, values in per_chromosome.items():
        shared = sorted(set(values) & set(per_chromosome.get(reference, {})))
        summary[method] = {
            "mean": across_interval(values.values()),
            "difference": across_interval(values[chrom] - per_chromosome[reference][chrom] for chrom in shared),
        }
    return summary


def summarize(cohorts: Path, scenarios: Path, dev: Path, tags: list[str], results: list[tuple[str, Path]]) -> dict:
    chromosomes = sorted({chrom.name for _, root in results if root.is_dir() for chrom in root.iterdir() if chrom.is_dir()},
                         key=lambda name: int(name.removeprefix("chr")))
    reference_method = results[0][0]
    output: dict = {"reference": reference_method, "level": LEVEL, "scenarios": {}}
    for tag in tags:
        per_chromosome: dict[str, dict[str, float]] = {name: {} for name, _ in results}
        detail: dict[str, dict] = {name: {} for name, _ in results}
        truths: dict[str, dict] = {}
        for chrom in chromosomes:
            scenario = scenario_path(scenarios, dev, chrom, tag)
            if not (scenario / "truth.npz").exists():
                continue
            is_test = np.load(cohorts / chrom / "samples.npz")["is_test"]
            test = np.flatnonzero(is_test)
            covariates = covariate_matrix(cohorts / chrom, "truth")[0][test]
            genetic_value = np.load(scenario / "truth.npz")["genetic_value"][test]
            record = json.loads((scenario / "scenario.json").read_text())
            truths[chrom] = {"h2": record["params"]["h2"], **record["summary"]}
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            draws = [rng.integers(0, test.size, test.size) for _ in range(BOOTSTRAP_DRAWS)]
            for name, root in results:
                prediction = root / chrom / f"scenario_{tag}" / "prediction.npz"
                if prediction.exists():
                    detail[name][chrom] = replicate_accuracy(genetic_value, np.load(prediction)["total"], covariates, draws)
                    per_chromosome[name][chrom] = detail[name][chrom]["r2"]
        summary = summarize_methods(per_chromosome, reference_method)
        output["scenarios"][tag] = {
            "truth": truths,
            "methods": {name: {"chromosomes": detail[name], **summary[name]} for name, _ in results},
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    scenarios_parser = sub.add_parser("scenarios")
    scenarios_parser.add_argument("--cohort", required=True)
    scenarios_parser.add_argument("--dev", required=True)
    scenarios_parser.add_argument("--out", required=True)
    scenarios_parser.add_argument("--tags", nargs="+", required=True)
    people_parser = sub.add_parser("people")
    people_parser.add_argument("--cohort", required=True)
    people_parser.add_argument("--reference", required=True)
    summary_parser = sub.add_parser("summary")
    summary_parser.add_argument("--cohorts", required=True)
    summary_parser.add_argument("--scenarios", required=True)
    summary_parser.add_argument("--dev", required=True)
    summary_parser.add_argument("--tags", nargs="+", required=True)
    summary_parser.add_argument("--results", nargs="+", required=True, help="label=directory, the reference first")
    summary_parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "scenarios":
        for tag, record in zip(args.tags, build_scenarios(Path(args.cohort), Path(args.dev), Path(args.out), args.tags)):
            print(tag, json.dumps(record["summary"]), flush=True)
    elif args.command == "people":
        differences = people_differences(Path(args.cohort), Path(args.reference))
        print(json.dumps({"cohort": args.cohort, "reference": args.reference, "differing_fields": differences}))
        if differences:
            raise SystemExit(1)
    else:
        results = [(label, Path(directory)) for label, directory in (entry.split("=", 1) for entry in args.results)]
        summary = summarize(Path(args.cohorts), Path(args.scenarios), Path(args.dev), args.tags, results)
        Path(args.out).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
