"""Post hoc: the SV gain and SV-PGS against LDpred2-auto within each genetic ancestry group (eur, afr, amr) of the All of
Us held-out people, on the converged traits of svpgs_vs_ldpred2_app.py's SPEC. Per group and trait: the with_sv arm's
held-out partial r^2 less the snv_indel arm's (relative to the two arms' mean r^2, as the primary's g), and less
LDpred2-auto's (relative to LDpred2-auto's r^2), each with the joint family bootstrap's 95% interval and a one-sided p
(H1: the with_sv arm is more accurate); per group, the common relative gain g over its traits (sv_gain_meta's
common_relative), and the one-sided test that AFR's or AMR's g exceeds EUR's, from their difference over the same
joint resamples. A group-trait cell is analysed only with at least 100 held-out people and, for a disease, at least
40 cases and 40 controls; any count under 40 is printed as <40 and its cell has no estimate.

Runs inside the workspace (the app), where it reads each run's held-out scores; it prints aggregates only, as one JSON
line.

    python ancestry_app.py SV_GAIN_META.py LONGREAD_IDS SPEC.json [RESAMPLES]
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys

import numpy as np
from scipy import stats

SHOWN = 40
GROUPS = ("eur", "afr", "amr")
MINIMUM_PEOPLE = 100


def main() -> None:
    spec = importlib.util.spec_from_file_location("sv_gain_meta", sys.argv[1])
    meta = importlib.util.module_from_spec(spec)
    sys.modules["sv_gain_meta"] = meta
    spec.loader.exec_module(meta)
    with open(sys.argv[2], encoding="utf-8") as handle:
        rows = [line.rstrip("\n").split("\t") for line in handle if line.strip()]
    column = rows[0].index("research_id") if "research_id" in rows[0] else None
    excluded_ids = {row[column] for row in rows[1:]} if column is not None else {row[0] for row in rows}
    with open(sys.argv[3], encoding="utf-8") as handle:
        entries = json.load(handle)
    resamples = int(sys.argv[4]) if len(sys.argv) > 4 else meta.RESAMPLES

    found = []
    for entry in entries:
        runs = [meta.excluded(meta.read_run(name), excluded_ids)[0] for name in entry["runs"]]
        run = meta.merge_pairs(runs) if len(runs) > 1 else runs[0]
        with np.load(f"{meta.OPS}/{entry['fastpgs']}/fastpgs/{run['name']}.npz", allow_pickle=False) as arrays:
            methods, arms = arrays["methods"].tolist(), arrays["arms"].tolist()
            ldpred2 = np.asarray(arrays["scores"], dtype=np.float64)[methods.index("ldpred2_auto"), arms.index("snv_indel")]
            ldpred2_target = np.asarray(arrays["target"], dtype=np.float64)
            position = {str(value): row for row, value in enumerate(np.asarray(arrays["research_ids"]).astype(str))}
        keep = np.array([str(value) in position for value in run["ids"]], dtype=bool)
        other = np.array([position[str(value)] for value in np.asarray(run["ids"])[keep]], dtype=np.int64)
        if not np.allclose(ldpred2_target[other], np.asarray(run["target"], dtype=np.float64)[keep]) or keep.sum() < SHOWN:
            continue
        restricted = {key: np.asarray(run[key])[keep] for key in meta.PER_PERSON}
        found.append({"label": entry["label"], "binary": run["binary"], **restricted, "ldpred2": ldpred2[other]})

    people, clusters, ancestry, _ = meta.union(found)
    generator = np.random.default_rng(20261023)
    counts = meta.cluster_bootstrap_counts(np.zeros(int(clusters.max()) + 1, dtype=np.int64), resamples, generator)
    weights, unit = counts[:, clusters], np.ones((1, clusters.shape[0]))
    comparisons = {"sv_gain": ("baseline", True), "vs_ldpred2": ("ldpred2", False)}
    result = {"resamples": resamples, "groups": {}}
    draws_by_group: dict[str, dict[str, tuple[float, np.ndarray]]] = {}
    for group in GROUPS:
        cells = []
        pooled = {name: ([], [], [], []) for name in comparisons}
        for i, entry in enumerate(found):
            target = np.asarray(entry["target"], dtype=np.float64)
            rows = np.flatnonzero(ancestry[people[i]] == group)
            size = int(rows.shape[0])
            cases = int(np.sum(target[rows])) if entry["binary"] else None
            cell = {"trait": entry["label"], "people": meta.shown(size) if size else "0",
                    "cases": (meta.shown(cases) if cases else "0") if entry["binary"] else None}
            if size < MINIMUM_PEOPLE or (entry["binary"] and (cases < SHOWN or size - cases < SHOWN)):
                cell["analysed"] = False
                cells.append(cell)
                continue
            cell["analysed"] = True
            for name, (reference, mean_level) in comparisons.items():
                trait = meta.PairedTrait(entry["label"], entry["binary"], people[i], target,
                                         np.asarray(entry["covariates"], dtype=np.float64),
                                         np.asarray(entry[reference], dtype=np.float64), np.asarray(entry["candidate"], dtype=np.float64))
                gain, base = meta.gains(unit, trait, rows=rows)
                gain_draws, base_draws = meta.gains(weights, trait, rows=rows)
                level, level_draws = (base + gain / 2.0, base_draws + gain_draws / 2.0) if mean_level else (base, base_draws)
                relative = gain_draws / level_draws
                error = float(np.nanstd(gain_draws, ddof=1))
                cell[name] = {"reference_r2": float(base[0]), "with_sv_r2": float(base[0] + gain[0]),
                              "relative": float(gain[0] / level[0]), "lower": float(np.nanquantile(relative, 0.025)),
                              "upper": float(np.nanquantile(relative, 0.975)),
                              "p_one": float(stats.norm.sf(gain[0] / error)) if error > 0 else math.nan}
                for store, value in zip(pooled[name], (gain[0], base[0], gain_draws, base_draws)):
                    store.append(value)
            cells.append(cell)
        summary = {}
        for name, (estimates, bases, draws, base_draws) in pooled.items():
            if not estimates:
                continue
            common = meta.common_relative(np.array(estimates), np.array(bases), np.column_stack(draws), np.column_stack(base_draws))
            summary[name] = {key: float(common[key]) for key in ("estimate", "ci_low", "ci_high", "p_one_sided", "upper_95")}
            summary[name]["traits"] = len(estimates)
            # g on every resample, with common_relative's trait weights, for the between-group differences
            levels, level_draws = np.array(bases) + np.array(estimates) / 2.0, np.column_stack(base_draws) + np.column_stack(draws) / 2.0
            trait_weights = common["weight_share"] / levels
            draws_by_group.setdefault(name, {})[group] = (summary[name]["estimate"],
                                                          (np.column_stack(draws) @ trait_weights) / (level_draws @ trait_weights))
        result["groups"][group] = {"cells": cells, "common": summary}
    # one-sided test that a group's g exceeds EUR's: the difference over the same joint resamples
    result["difference_from_eur"] = {}
    for name, by_group in draws_by_group.items():
        for group in ("afr", "amr"):
            if group not in by_group or "eur" not in by_group:
                continue
            difference = by_group[group][0] - by_group["eur"][0]
            difference_draws = by_group[group][1] - by_group["eur"][1]
            error = float(np.nanstd(difference_draws, ddof=1))
            result["difference_from_eur"][f"{name}:{group}"] = {
                "difference": difference, "lower": difference - 1.959964 * error, "upper": difference + 1.959964 * error,
                "p_one": float(stats.norm.sf(difference / error)) if error > 0 else math.nan}
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
