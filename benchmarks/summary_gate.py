"""The summary route's accuracy gate on bench-sim: genetic_r2 (and the phenotype's incremental r2) of a method's
predictions against the exact route's reference, overall and within each ancestry group, with paired bootstrap 95%
intervals over the test people, and the two fits' wall times.

The reference per scenario is the first of ``--references`` (the lead's runs, by default svpgs_v5, svpgs_v4, svpgs_v2)
that holds a prediction. genetic_r2 is ``harness.genetic_accuracy``'s squared partial correlation with the simulated
genetic value beyond the covariates; within a group it is the same on the group's test people. The gate: no
difference whose whole interval lies below zero, overall or in any group. Evidence label [sim].
usage: python -m benchmarks.summary_gate --method-root <dir> --scenarios 000,001,005,008"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.bench_sim.harness import covariate_matrix, genetic_accuracy, incremental_r2

COHORT = Path("/scratch.global/sauer354/svpgs-team/bench-sim/v7/cohort/chr22")
SCENARIOS = Path("/scratch.global/sauer354/svpgs-team/bench-sim/v7/dev")
LEAD = Path("/scratch.global/sauer354/svpgs-team/lead/bench-sim-truth/results/truth")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--method-root", type=Path, required=True, help="the method's results: <root>/scenario_XXX/prediction.npz")
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--references", default="svpgs_v5,svpgs_v4,svpgs_v2")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args()
    samples = np.load(COHORT / "samples.npz")
    test = np.flatnonzero(samples["is_test"])
    covariates, _names = covariate_matrix(COHORT, "truth")
    covariates = covariates[test]
    groups = samples["group"][test]
    names = [str(name) for name in samples["group_names"]]
    rows = []
    for scenario in arguments.scenarios.split(","):
        truth = np.load(SCENARIOS / f"scenario_{scenario}" / "truth.npz")
        outcome = truth["phenotype"][test].astype(np.float64)
        genetic = truth["genetic_value"][test].astype(np.float64)
        reference = next((LEAD / name / f"scenario_{scenario}" for name in arguments.references.split(",")
                          if (LEAD / name / f"scenario_{scenario}" / "prediction.npz").exists()), None)
        mine = arguments.method_root / f"scenario_{scenario}"
        if reference is None or not (mine / "prediction.npz").exists():
            print(f"scenario {scenario}: missing {'reference' if reference is None else 'method'} prediction", flush=True)
            continue
        predictions = {"reference": np.load(reference / "prediction.npz")["total"].astype(np.float64),
                       "summary": np.load(mine / "prediction.npz")["total"].astype(np.float64)}
        seconds = {name: json.loads((path / "meta.json").read_text()).get("fit_seconds") for name, path in (("reference", reference), ("summary", mine))}
        generator = np.random.default_rng(0)
        subsets = {"all": np.arange(test.size), **{name: np.flatnonzero(groups == index) for index, name in enumerate(names)}}
        row = {"scenario": scenario, "reference": reference.parent.name, "fit_seconds": seconds, "groups": {}}
        for subset, members in subsets.items():
            if members.size <= covariates.shape[1] + 2:
                continue
            draws = [members[generator.integers(0, members.size, members.size)] for _ in range(arguments.resamples)]

            def measures(index: np.ndarray) -> dict:
                return {name: (genetic_accuracy(genetic[index], values[index], covariates[index])[0],
                               incremental_r2(outcome[index], values[index], covariates[index])[0]) for name, values in predictions.items()}

            point = measures(members)
            boot = [measures(index) for index in draws]
            genetic_differences = np.array([value["summary"][0] - value["reference"][0] for value in boot])
            phenotype_differences = np.array([value["summary"][1] - value["reference"][1] for value in boot])
            row["groups"][subset] = {
                "genetic_r2": {"summary": point["summary"][0], "reference": point["reference"][0], "difference": point["summary"][0] - point["reference"][0],
                               "interval": np.percentile(genetic_differences, [2.5, 97.5]).tolist()},
                "incremental_r2": {"summary": point["summary"][1], "reference": point["reference"][1], "difference": point["summary"][1] - point["reference"][1],
                                   "interval": np.percentile(phenotype_differences, [2.5, 97.5]).tolist()},
            }
            entry = row["groups"][subset]["genetic_r2"]
            print(
                f"scenario {scenario} {subset:>12}: genetic_r2 summary {entry['summary']:.4f} reference {entry['reference']:.4f} "
                f"d {entry['difference']:+.4f} [{entry['interval'][0]:+.4f}, {entry['interval'][1]:+.4f}]"
                f"{'  LOSS' if entry['interval'][1] < 0.0 else ''}", flush=True,
            )
        print(f"scenario {scenario}: fit seconds summary {seconds['summary']} against {row['reference']} {seconds['reference']}", flush=True)
        rows.append(row)
    if arguments.out is not None:
        arguments.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
