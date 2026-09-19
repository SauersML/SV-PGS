"""bench-sim: score every (method, scenario) prediction and report paired differences against a reference.

    python -m benchmarks.bench_sim.compare --cohort <cohort/chr22> --scenarios <dev or sealed dir> --results <results dir> \
        --reference ridge_inf_simple [--methods a b ...]

For each scenario the primary metric is the incremental R2 (quantitative) or the liability-scale R2 (binary).
Every scenario is reported, losses included; the summary is the mean paired difference with its SE across
scenarios, then descriptive breakdowns by the scenario's truth parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.bench_sim.harness import score

BREAKDOWNS = ("shape", "sv_mode", "tr_mode", "annotation_mode", "binary", "frequency_source")


def primary(metrics: dict) -> float:
    return metrics.get("liability_r2", metrics["incremental_r2"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--methods", nargs="*")
    args = parser.parse_args()
    cohort, scenarios, results = Path(args.cohort), Path(args.scenarios), Path(args.results)
    methods = args.methods or sorted(path.name for path in results.iterdir() if path.is_dir())
    table: dict[str, dict[str, dict]] = {}
    for method in methods:
        for prediction in sorted((results / method).glob("*/prediction.npz")):
            name = prediction.parent.name
            metrics_path = prediction.parent / "metrics.json"
            if metrics_path.exists() and metrics_path.stat().st_mtime >= prediction.stat().st_mtime:
                metrics = json.loads(metrics_path.read_text())
            else:
                metrics = score(cohort, scenarios / name, prediction)
                metrics_path.write_text(json.dumps(metrics, indent=1))
            table.setdefault(method, {})[name] = metrics
    reference = table[args.reference]
    report = {"reference": args.reference, "methods": {}}
    for method, rows in table.items():
        shared = sorted(set(rows) & set(reference))
        differences = np.array([primary(rows[name]) - primary(reference[name]) for name in shared])
        entry = {
            "scenarios": len(shared),
            "mean_primary": float(np.mean([primary(rows[name]) for name in shared])) if shared else None,
            "mean_difference": float(differences.mean()) if shared else None,
            "se_difference": float(differences.std(ddof=1) / np.sqrt(differences.size)) if differences.size > 1 else None,
            "wins": int((differences > 0).sum()),
            "losses": int((differences < 0).sum()),
            "per_scenario": {name: float(value) for name, value in zip(shared, differences)},
            "mean_calibration_slope": float(np.mean([rows[name]["calibration_slope"] for name in shared])) if shared else None,
        }
        credit = [(rows[name].get("structural_share_predicted"), rows[name]["structural_share_truth"]) for name in shared]
        credit = [(predicted, true) for predicted, true in credit if predicted is not None]
        if credit:
            entry["structural_share_error"] = float(np.mean([predicted - true for predicted, true in credit]))
        breakdown: dict = {}
        for key in BREAKDOWNS:
            groups: dict = {}
            for name, value in zip(shared, differences):
                params = json.loads((scenarios / name / "scenario.json").read_text())["params"]
                groups.setdefault(str(params[key]), []).append(float(value))
            breakdown[key] = {level: {"n": len(values), "mean": float(np.mean(values))} for level, values in groups.items()}
        entry["breakdown"] = breakdown
        report["methods"][method] = entry
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
