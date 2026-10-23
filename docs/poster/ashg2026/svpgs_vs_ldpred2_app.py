"""SV-PGS against LDpred2-auto on the same All of Us held-out people: per trait, the held-out partial r^2 of SV-PGS
(each arm of the SV-gain primary's matched pairs, benchmarks/bench_real/sv_gain_meta.py) less LDpred2-auto's fitted
on the same run records' SNVs and indels (aou_fastpgs.py's ldpred2_auto, snv_indel arm), with 95% intervals from the
joint family bootstrap, and the common relative difference over the traits (sv_gain_meta's primary statistic). Converged
fits only: each trait enters an arm's comparison only where that arm's SV-PGS fit converged, as SPEC lists (svgain's
converged list, one JSON entry per trait: label, the pairing runs, the run whose aou_fastpgs.py job fitted LDpred2-auto,
and the converged arms).

Runs inside the workspace (the app), where it reads each run's held-out scores; it prints aggregates only, as one JSON
line, with any count under 40 masked. The long-read panel's members are dropped, as in the primary. svpgs_vs_ldpred2.py
draws the figure from that line.

    .venv/bin/python svpgs_vs_ldpred2_app.py SV_GAIN_META_PRIMARY.py LONGREAD_IDS SPEC.json > line.json
"""
from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np

SHOWN = 40


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
        spec = json.load(handle)
    found, notes = [], []
    for entry in spec:
        label, names, fastpgs_run = entry["label"], entry["runs"], entry["fastpgs"]
        runs = [meta.excluded(meta.read_run(name), excluded_ids)[0] for name in names]
        run = meta.merge_pairs(runs) if len(runs) > 1 else runs[0]
        with np.load(f"{meta.OPS}/{fastpgs_run}/fastpgs/{run['name']}.npz", allow_pickle=False) as arrays:
            methods, arms = arrays["methods"].tolist(), arrays["arms"].tolist()
            ldpred2 = np.asarray(arrays["scores"], dtype=np.float64)[methods.index("ldpred2_auto"), arms.index("snv_indel")]
            ldpred2_target = np.asarray(arrays["target"], dtype=np.float64)
            position = {str(value): row for row, value in enumerate(np.asarray(arrays["research_ids"]).astype(str))}
        keep = np.array([str(value) in position for value in run["ids"]], dtype=bool)
        other = np.array([position[str(value)] for value in np.asarray(run["ids"])[keep]], dtype=np.int64)
        agree = bool(np.allclose(ldpred2_target[other], np.asarray(run["target"], dtype=np.float64)[keep]))
        share = float(keep.mean())
        notes.append(f"{label}: {share:.3f} of SV-PGS held-out people scored by LDpred2-auto, outcomes agree {agree}")
        if not agree or keep.sum() < SHOWN:
            continue
        restricted = {key: np.asarray(run[key])[keep] for key in meta.PER_PERSON}
        found.append({"label": label, "binary": run["binary"], **restricted, "ldpred2": ldpred2[other], "arms": entry["arms"]})

    people, clusters, ancestry, _ = meta.union(found)
    result = {}
    for arm, column in (("with_sv", "candidate"), ("snv_indel", "baseline")):
        traits = [meta.PairedTrait(entry["label"], entry["binary"], people[i], np.asarray(entry["target"], dtype=np.float64),
                                   np.asarray(entry["covariates"], dtype=np.float64), entry["ldpred2"], np.asarray(entry[column], dtype=np.float64))
                  for i, entry in enumerate(found) if arm in entry["arms"]]
        analysed = meta.analyse(traits, clusters, ancestry, seed=20261023, permutations=0)
        common = analysed["common"]
        result[arm] = {
            "traits": analysed["traits"],
            "difference": analysed["estimate"].tolist(), "ci_low": analysed["ci_low"].tolist(), "ci_high": analysed["ci_high"].tolist(),
            "p_one_sided": analysed["p_one_sided"].tolist(), "ldpred2_r2": analysed["baseline_r2"].tolist(),
            "relative": analysed["relative"].tolist(), "relative_se": analysed["relative_standard_error"].tolist(),
            "common": {key: float(common[key]) for key in ("estimate", "ci_low", "ci_high", "p_one_sided", "lower_95", "upper_95")},
            "gls_absolute": {"estimate": float(analysed["absolute_pooled"]["fixed"].estimate),
                             "standard_error": float(analysed["absolute_pooled"]["fixed"].standard_error)},
        }
    result["held_out"] = [int(entry["ids"].shape[0]) if entry["ids"].shape[0] >= SHOWN else None for entry in found]
    result["notes"] = notes
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
