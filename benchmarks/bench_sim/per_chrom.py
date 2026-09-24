"""bench-sim per-chromosome replicates: one independent trait per chromosome, each fitted alone.

A dev scenario's interval on chr22 resamples the test people only: the trait (its causal set, effects and noise) and
the fit stay fixed, so it misses trait-to-trait and fit-to-fit variation. Here every chromosome's cohort (cohort.py's
own seeds: the same people, groups, covariates and 40k/10k split; only the haplotype mosaic differs) carries its own
draw of a dev scenario: the dev scenario's parameters with the effect and annotation seeds drawn afresh per
chromosome, so the causal variants lie on that chromosome alone and their effects and the phenotype noise are
independent between chromosomes. chr22 keeps the dev scenario itself. Each method is fitted on each chromosome alone.

The headline interval of a method's mean genetic r2 over chromosomes is a two-level bootstrap: the chromosomes resampled
with replacement, then the test people within each drawn chromosome, so it covers both the trait-and-fit variation
between chromosomes and the test sample within each. A method's difference from the reference method is computed on the
same drawn chromosomes and the same drawn people. The chromosome-only bootstrap (people fixed) is reported beside it.

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
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 2026
PEOPLE_CHUNK = 250
"""Person resamples evaluated per step (250 x 10,000 test people x 8 bytes = 20 MB per array)."""


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


def squared_correlation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Row-wise squared Pearson correlation of two [draws, people] arrays (0 where a row has no spread)."""
    first = first - first.mean(axis=1, keepdims=True)
    second = second - second.mean(axis=1, keepdims=True)
    denominator = (first * first).sum(axis=1) * (second * second).sum(axis=1)
    numerator = (first * second).sum(axis=1) ** 2
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def chromosome_accuracy(genetic_value: np.ndarray, predictions: dict[str, np.ndarray], covariates: np.ndarray,
                        seed) -> dict[str, dict]:
    """Each method's genetic r2 on one chromosome (harness.genetic_accuracy's: the prediction and the genetic value
    residualized on [1, covariates] over the test people) and its BOOTSTRAP_DRAWS person-resampled values ("pool"),
    every method on the same resampled people, with the pool's percentile interval."""
    genetic = residualized(genetic_value, covariates)
    predicted = {name: residualized(values, covariates) for name, values in predictions.items()}
    pools = {name: np.empty(BOOTSTRAP_DRAWS) for name in predicted}
    rng = np.random.default_rng(seed)
    size = genetic.shape[0]
    for first in range(0, BOOTSTRAP_DRAWS, PEOPLE_CHUNK):
        rows = rng.integers(0, size, (min(PEOPLE_CHUNK, BOOTSTRAP_DRAWS - first), size))
        drawn = genetic[rows]
        for name, values in predicted.items():
            pools[name][first:first + rows.shape[0]] = squared_correlation(values[rows], drawn)
    result = {}
    for name, values in predicted.items():
        low, high = np.quantile(pools[name], [(1.0 - LEVEL) / 2.0, (1.0 + LEVEL) / 2.0])
        result[name] = {"r2": float(squared_correlation(values[None], genetic[None])[0]), "low": float(low),
                        "high": float(high), "pool": pools[name]}
    return result


def bootstrap_interval(points: np.ndarray, pools: np.ndarray, seed) -> dict:
    """The mean over chromosomes of per-chromosome values ``points`` [n], with two percentile intervals over
    BOOTSTRAP_DRAWS draws of n chromosomes with replacement: "two_level" also takes, for each drawn chromosome, one of
    its person-resampled values (``pools`` [n, BOOTSTRAP_DRAWS], themselves independent person resamples);
    "chromosomes" keeps each chromosome's point. The same seed and chromosome set give the same draws, so intervals of
    differences built from paired pools are paired."""
    count = points.shape[0]
    if count == 0:
        return {"mean": float("nan"), "n": 0, "two_level": [float("nan")] * 2, "chromosomes": [float("nan")] * 2}
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, count, (BOOTSTRAP_DRAWS, count))
    person = rng.integers(0, pools.shape[1], (BOOTSTRAP_DRAWS, count))
    quantiles = [(1.0 - LEVEL) / 2.0, (1.0 + LEVEL) / 2.0]
    return {
        "mean": float(points.mean()),
        "n": count,
        "two_level": np.quantile(pools[chosen, person].mean(axis=1), quantiles).tolist(),
        "chromosomes": np.quantile(points[chosen].mean(axis=1), quantiles).tolist(),
    }


def summarize_methods(per_chromosome: dict[str, dict[str, dict]], reference: str, seed) -> dict:
    """Per method: the bootstrap intervals of its mean r2 over its chromosomes, and of its paired difference from
    ``reference`` over the chromosomes both have (the difference of their pools, drawn on the same people)."""
    summary = {}
    for method, entries in per_chromosome.items():
        chromosomes = sorted(entries)
        shared = [chrom for chrom in chromosomes if chrom in per_chromosome.get(reference, {})]
        base = per_chromosome.get(reference, {})
        summary[method] = {
            "mean": bootstrap_interval(np.array([entries[c]["r2"] for c in chromosomes]),
                                       np.array([entries[c]["pool"] for c in chromosomes]).reshape(len(chromosomes), BOOTSTRAP_DRAWS), seed),
            "difference": bootstrap_interval(np.array([entries[c]["r2"] - base[c]["r2"] for c in shared]),
                                             np.array([entries[c]["pool"] - base[c]["pool"] for c in shared]).reshape(len(shared), BOOTSTRAP_DRAWS), seed),
        }
    return summary


def summarize(cohorts: Path, scenarios: Path, dev: Path, tags: list[str], results: list[tuple[str, Path]]) -> dict:
    chromosomes = sorted({chrom.name for _, root in results if root.is_dir() for chrom in root.iterdir() if chrom.is_dir()},
                         key=lambda name: int(name.removeprefix("chr")))
    reference_method = results[0][0]
    output: dict = {"reference": reference_method, "level": LEVEL, "draws": BOOTSTRAP_DRAWS, "scenarios": {}}
    for tag in tags:
        per_chromosome: dict[str, dict[str, dict]] = {name: {} for name, _ in results}
        truths: dict[str, dict] = {}
        for chrom in chromosomes:
            scenario = scenario_path(scenarios, dev, chrom, tag)
            predictions = {name: root / chrom / f"scenario_{tag}" / "prediction.npz" for name, root in results}
            predictions = {name: path for name, path in predictions.items() if path.exists()}
            if not (scenario / "truth.npz").exists() or not predictions:
                continue
            test = np.flatnonzero(np.load(cohorts / chrom / "samples.npz")["is_test"])
            covariates = covariate_matrix(cohorts / chrom, "truth")[0][test]
            record = json.loads((scenario / "scenario.json").read_text())
            truths[chrom] = {"h2": record["params"]["h2"], **record["summary"]}
            accuracy = chromosome_accuracy(np.load(scenario / "truth.npz")["genetic_value"][test],
                                           {name: np.load(path)["total"] for name, path in predictions.items()},
                                           covariates, [BOOTSTRAP_SEED, int(chrom.removeprefix("chr"))])
            for name, entry in accuracy.items():
                per_chromosome[name][chrom] = entry
        summary = summarize_methods(per_chromosome, reference_method, BOOTSTRAP_SEED)
        output["scenarios"][tag] = {
            "truth": truths,
            "methods": {name: {"chromosomes": {chrom: {key: value for key, value in entry.items() if key != "pool"}
                                               for chrom, entry in per_chromosome[name].items()},
                               **summary[name]} for name, _ in results},
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
