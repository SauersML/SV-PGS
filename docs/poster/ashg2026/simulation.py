"""The simulation study's numbers for the poster: every finished task of the power grid (benchmarks/power/grid.py, run on
MSI on bench-sim's 1000 Genomes chr22 cohort) collected into simulation_summary.json beside this file, which figures.py
and build_pptx.py read. Nothing here is from All of Us.

    uv run python docs/poster/ashg2026/simulation.py fetch     # copy the grid's results from MSI, then collect
    uv run python docs/poster/ashg2026/simulation.py collect   # collect the local copy again

The local copy lives outside the repository (LOCAL). Per grid point and training size the summary gives each method's
mean test AUC with its SE over replicates, the SV arm's mean AUC gain with a 95% interval, the share of replicates whose
one-sided paired DeLong test favours the SV arm at 0.05 with its Wilson interval, the truth's realized SV share of the
genetic variance, and the expected AUC of a PGS that captures the whole heritability (Daetwyler et al. 2008 accuracy,
Wray et al. 2010 AUC; benchmarks/power/expected.py). The SV effects section pairs each causal SV's simulated effect with
its fitted one at the largest training size, and bins every SV's fitted |effect| by length.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from scipy import stats

from benchmarks.power.expected import Architecture, auc_of, liability_r2
from benchmarks.power.grid import CAUSAL_VARIANTS, HERITABILITY, POINTS, PREVALENCE, TRAIN_COUNTS
from benchmarks.power.uncertainty import rejection_summary

HERE = Path(__file__).resolve().parent
SUMMARY = HERE / "simulation_summary.json"
GRID = "grid2"
"""The grid run on MSI the poster reads: alpha = -0.38 (benchmarks/power/grid.py, FREQUENCY_ALPHA)."""
LOCAL = Path.home() / "svpgs-bench" / f"poster-{GRID}"
REMOTE = "msi:/scratch.global/sauer354/svpgs-poster"
TEST_PEOPLE = 10_000
CHR22_MORGANS = 0.839
"""chr22's genetic length in the cohort's map (0 to 83.9 cM)."""
EFFECTIVE_POPULATION = 10_000
"""The human effective population size of Goddard (2009)'s effective-segment count."""
ONE_SIDED = stats.norm.isf(0.05)
LENGTH_EDGES_BP = (50, 100, 300, 1_000, 10_000, math.inf)
"""SV length bins: about 950, 380, 170, 130 and 30 imputed SVs on chr22 from shortest to longest."""
EFFECT_POINTS = ("sv_20pct", "length_positive")


def effective_segments() -> float:
    """Independent chromosome segments on chr22, Me = 2 Ne L / ln(4 Ne L) (Goddard 2009): about 1,600. The accuracy
    model uses min(Me, causal variants) (Daetwyler et al. 2010), so at 200 causal variants Me does not enter."""
    product = EFFECTIVE_POPULATION * CHR22_MORGANS
    return 2.0 * product / math.log(4.0 * product)


def setting(point: str) -> dict:
    """The point's heritability, prevalence and expected causal count (grid.py's defaults unless the point sets them)."""
    values = POINTS[point]
    truth = values.get("truth", {})
    return {
        "h2": truth.get("h2", HERITABILITY), "prevalence": truth.get("prevalence", PREVALENCE),
        "causal": values.get("causal_variants", CAUSAL_VARIANTS), "sv_causal_fraction": values["sv_causal_fraction"],
        "sv_effect_fold": values["sv_effect_fold"],
    }


def expected_auc(point: str, train_count: int) -> float:
    """AUC of a PGS from ``train_count`` people capturing the point's whole liability heritability."""
    values = setting(point)
    architecture = Architecture(
        name=point, prevalence=values["prevalence"], heritability=values["h2"], sv_share=0.0, tagged_by_small=0.0,
        sv_imputation_r2=1.0, causal_variants=values["causal"], effective_segments=effective_segments(),
        training_people=train_count, test_people=TEST_PEOPLE,
    )
    return auc_of(liability_r2(values["h2"], architecture), values["prevalence"])


def mean_se(values) -> list[float]:
    values = np.asarray([value for value in values if value is not None and math.isfinite(value)], dtype=float)
    if values.size == 0:
        return [math.nan, math.nan]
    return [float(values.mean()), float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else math.nan]


def interval(values) -> dict:
    values = [value for value in values if value is not None and math.isfinite(value)]
    mean, se = mean_se(values)
    critical = float(stats.t.ppf(0.975, len(values) - 1)) if len(values) > 1 else math.nan
    return {"mean": mean, "se": se, "lower": mean - critical * se, "upper": mean + critical * se,
            "replicates": len(values)}


def collect(local: Path = LOCAL, target: Path = SUMMARY) -> Path:
    results = [json.loads(path.read_text()) for path in sorted((local / "grid" / "tasks").glob("*/n*/replicate*/result.json"))]
    # each arm's attempt record (grid.py writes one per arm job), to count replicates lost to a failed fit
    attempts = [json.loads(path.read_text()) for path in sorted((local / "grid" / "tasks").glob("*/n*/replicate*/*/attempt.json"))]
    if not results and not attempts:
        raise SystemExit(f"no grid results or attempts under {local / 'grid' / 'tasks'}")
    coverage = {}
    for attempt in attempts:
        cell = coverage.setdefault(attempt["point"], {}).setdefault(str(attempt["train_count"]), {})
        counts = cell.setdefault(attempt["arm"], {"completed": 0, "failed": 0, "running": 0})
        counts[attempt["status"]] += 1
    points: dict = {}
    for point in POINTS:
        rows = [result for result in results if result["point"] == point]
        if not rows:
            continue
        by_size = {}
        for train_count in sorted({row["train_count"] for row in rows}):
            group = [row for row in rows if row["train_count"] == train_count]
            valid = [row for row in group if math.isfinite(row["delta_auc"]) and math.isfinite(row["z"])]
            power = rejection_summary(sum(row["z"] > ONE_SIDED for row in valid), len(valid))
            valid_ldpred2 = [row for row in group if "z_vs_ldpred2" in row and math.isfinite(row["z_vs_ldpred2"])]
            valid_allvar = [row for row in group if "z_vs_ldpred2_allvar" in row and math.isfinite(row["z_vs_ldpred2_allvar"])]
            failed = {row["replicate"] for row in attempts if row["point"] == point and row["train_count"] == train_count and row["status"] == "failed"}
            by_size[str(train_count)] = {
                "replicates": len(group),
                "failed_replicates": len(failed - {row["replicate"] for row in group}),
                "snv_indel_auc": mean_se(row["snv_indel_auc"] for row in group),
                "with_sv_auc": mean_se(row["with_sv_auc"] for row in group),
                "ldpred2_auc": mean_se(row.get("ldpred2_auc") for row in group),
                "delta_auc": interval(row["delta_auc"] for row in group),
                "delta_auc_vs_ldpred2": interval(row.get("delta_auc_vs_ldpred2") for row in group),
                "ldpred2_allvar_auc": mean_se(row.get("ldpred2_allvar_auc") for row in group),
                "ldpred2_allvar_replicates": len(valid_allvar),
                "auc_intervals": {key: interval(row.get(key) for row in group)
                                  for key in ("snv_indel_auc", "with_sv_auc", "ldpred2_auc", "ldpred2_allvar_auc")},
                "delta_auc_vs_ldpred2_allvar": interval(row.get("delta_auc_vs_ldpred2_allvar") for row in group),
                "power": power,
                "share_sv_beats_ldpred2": rejection_summary(sum(row["z_vs_ldpred2"] > ONE_SIDED for row in valid_ldpred2), len(valid_ldpred2)),
                "share_sv_beats_ldpred2_allvar": rejection_summary(sum(row["z_vs_ldpred2_allvar"] > ONE_SIDED for row in valid_allvar), len(valid_allvar)),
                "structural_share": mean_se(row["truth"]["structural_share"] for row in group),
                "causal_svs": mean_se(row["truth"]["causal_by_class"][3] for row in group),
                "causal_count": mean_se(row["truth"]["causal_count"] for row in group),
                "length_rho": mean_se(row.get("length_rho") for row in group),
                "deletion_over_insertion": mean_se(row.get("deletion_over_insertion") for row in group),
                "fit_seconds": {arm: mean_se(row["fit_seconds"][arm] for row in group if arm in row.get("fit_seconds", {}))[0]
                                for arm in ("snv_indel", "with_sv", "ldpred2_auto", "ldpred2_allvar")},
                "expected_auc": expected_auc(point, train_count),
            }
        points[point] = {"setting": setting(point), "sizes": by_size}
    summary = {
        "source": "benchmarks/power/grid.py on bench-sim v7 chr22 (Beagle arm), MSI",
        "tasks": len(results),
        "coverage": coverage,
        "mean_interval": "Student t 95% interval across completed simulation replicates",
        "train_counts": list(TRAIN_COUNTS),
        "test_people": TEST_PEOPLE,
        "effective_segments": effective_segments(),
        "points": points,
        "effects": effects(local),
    }
    temporary = target.with_suffix(".partial.json")
    temporary.write_text(json.dumps(summary, indent=1, allow_nan=True) + "\n")
    temporary.replace(target)
    return target


def effects(local: Path) -> dict:
    """At each effect point's largest training size with fitted effects: (simulated, fitted) per-allele effects of
    every causal SV, the simulated liability-scale effect put on the fit's scale (a per-allele change in case
    probability: phi(threshold) times the liability effect), and every SV's fitted |effect| by length bin."""
    completed = {point: [path for path in sorted((local / "grid" / "tasks" / point).glob("n*/replicate*/with_sv/sv_effects.npz"))
                         if json.loads((path.parent / "attempt.json").read_text())["status"] == "completed"]
                 for point in EFFECT_POINTS}
    if not any(completed.values()):
        return {}
    public = np.load(local / "public" / "sv_records.npz")
    records, log_length = public["record"], public["log_sv_length"]
    edges = np.log(np.asarray(LENGTH_EDGES_BP, dtype=float))
    bin_count = edges.size - 1
    imputed = log_length[public["measured"]]
    found = {}
    for point in EFFECT_POINTS:
        files = completed[point]
        if not files:
            continue
        largest = max(int(path.parts[-4][1:]) for path in files)
        files = [path for path in files if int(path.parts[-4][1:]) == largest]
        slope = stats.norm.pdf(stats.norm.isf(setting(point)["prevalence"]))
        pairs, bins = [], []
        for path in files:
            replicate = path.parts[-3]
            truth = np.load(local / "grid" / "scenarios" / point / replicate / "truth.npz")
            fitted = np.load(path)
            estimate = dict(zip(fitted["record"].tolist(), fitted["per_allele"].tolist()))
            for record, effect in zip(truth["causal"].tolist(), truth["per_allele"].tolist()):
                if record in estimate:
                    pairs.append([slope * effect, estimate[record]])
            lengths = log_length[np.searchsorted(records, fitted["record"])]
            which = np.clip(np.searchsorted(edges, lengths, side="right") - 1, 0, bin_count - 1)
            magnitude = np.abs(fitted["per_allele"])
            bins.append([float(magnitude[which == index].mean()) if np.any(which == index) else math.nan for index in range(bin_count)])
        bins = np.asarray(bins)
        bin_intervals = [interval(bins[:, index]) for index in range(bin_count)]
        median_lengths = []
        for low, high in zip(edges[:-1], edges[1:]):
            lengths = imputed[(imputed >= low) & (imputed < high)]
            median_lengths.append(float(np.median(np.exp(lengths))) if lengths.size else math.nan)
        simulated, fitted_values = (np.asarray(pairs).T if pairs else (np.array([]), np.array([])))
        found[point] = {
            "train_count": largest, "replicates": len(files), "pairs": pairs,
            "pearson": (float(np.corrcoef(simulated, fitted_values)[0, 1])
                        if len(pairs) > 2 and np.ptp(simulated) > 0 and np.ptp(fitted_values) > 0 else math.nan),
            "length_bins": {
                "lower_bp": list(LENGTH_EDGES_BP[:-1]), "upper_bp": list(LENGTH_EDGES_BP[1:]),
                "median_bp": median_lengths,
                "mean_abs": [entry["mean"] for entry in bin_intervals],
                "se": [entry["se"] for entry in bin_intervals],
                "intervals": bin_intervals,
            },
        }
    return found


def fetch(local: Path = LOCAL) -> None:
    """Copy the grid's results (result.json, the SV arm's SV effects) and the scenario truths from MSI, over the open
    SSH connection (ssh config host msi), plus the public SV record table."""
    local.mkdir(parents=True, exist_ok=True)
    rules = ["--include=*/", "--include=result.json", "--include=attempt.json", "--include=sv_effects.npz",
             "--include=truth.npz", "--include=scenario.json", "--exclude=*"]
    subprocess.run(["rsync", "-a", "--prune-empty-dirs", *rules, f"{REMOTE}/{GRID}/", str(local / "grid")], check=True)
    subprocess.run(["rsync", "-a", f"{REMOTE}/figs/", str(local / "public")], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("fetch", "collect"))
    parser.add_argument("--local", type=Path, default=LOCAL, help="the local copy of the grid's results")
    parser.add_argument("--summary", type=Path, default=SUMMARY, help="where the summary is written")
    arguments = parser.parse_args()
    if arguments.command == "fetch":
        fetch(arguments.local)
    print(collect(arguments.local, arguments.summary))


if __name__ == "__main__":
    main()
