"""bench-tox's report from the harness's per-fold JSON lines (aggregates only): per design, method and arm, the mean
held-out r² over folds and development compounds, the SV gain (snv_sv minus snv, paired by compound and fold), the
screen's size, the fits' wall seconds, and how many fits met the outer criterion or failed. Standard errors are a bootstrap over
compounds of the fold-averaged r² (the compounds are correlated: PREREG.md gives their effective number, 13.2, so a
compound bootstrap is the honest unit)."""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

METHODS = ("top_variant", "gblup_reml", "svpgs_mean_field")
BOOTSTRAP = 2_000


def load(results: pathlib.Path) -> tuple[pd.DataFrame, list[dict]]:
    rows, headers = [], []
    for path in sorted(results.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if "compound" not in record:
                headers.append(record)
                continue
            for method in METHODS:
                if method in record:
                    rows.append({
                        "design": record["split"].split("/")[0], "split": record["split"], "compound": record["compound"], "arm": record["arm"],
                        "screened": record["screened"], "screened_sv": record["screened_sv"], "method": method, "r2": record[method]["r2"],
                        "seconds": record[method]["seconds"], "status": record[method]["status"], "outer_criterion_met": record[method].get("outer_criterion_met"),
                    })
    return pd.DataFrame(rows), headers


def summarize(table: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    out = []
    for (design, method, arm), part in table.groupby(["design", "method", "arm"]):
        by_compound = part.groupby("compound")["r2"].mean()
        values = by_compound.to_numpy()
        draws = generator.integers(0, values.shape[0], size=(BOOTSTRAP, values.shape[0]))
        out.append({
            "design": design, "method": method, "arm": arm, "compounds": int(values.shape[0]), "folds": int(part["split"].nunique()),
            "mean_r2": float(values.mean()), "se_bootstrap_compounds": float(values[draws].mean(axis=1).std(ddof=1)),
            "median_screened": float(part["screened"].median()), "max_screened": int(part["screened"].max()),
            "empty_screens": int((part["screened"] == 0).sum()), "median_seconds": float(part["seconds"].median()), "max_seconds": float(part["seconds"].max()),
            "failed": int((part["status"] != "ok").sum()), "outer_criterion_met": int(part["outer_criterion_met"].fillna(False).sum()) if method == "svpgs_mean_field" else None,
        })
    return pd.DataFrame(out)


def sv_gain(table: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    out = []
    wide = table.pivot_table(index=["design", "method", "compound", "split"], columns="arm", values="r2")
    if "snv" not in wide or "snv_sv" not in wide:
        return pd.DataFrame(out)
    wide = wide.dropna()
    wide["gain"] = wide["snv_sv"] - wide["snv"]
    for (design, method), part in wide.groupby(level=["design", "method"]):
        by_compound = part.groupby(level="compound")["gain"].mean().to_numpy()
        draws = generator.integers(0, by_compound.shape[0], size=(BOOTSTRAP, by_compound.shape[0]))
        out.append({"design": design, "method": method, "compounds": int(by_compound.shape[0]), "mean_gain": float(by_compound.mean()),
                    "se_bootstrap_compounds": float(by_compound[draws].mean(axis=1).std(ddof=1))})
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    table, headers = load(pathlib.Path(arguments.results))
    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(table)
    gain = sv_gain(table)
    summary.to_csv(out / "mean_r2.tsv", sep="\t", index=False)
    gain.to_csv(out / "sv_gain.tsv", sep="\t", index=False)
    table.groupby(["design", "method", "arm", "compound"])["r2"].mean().reset_index().to_csv(out / "per_compound_r2.tsv", sep="\t", index=False)
    pd.DataFrame(headers).to_csv(out / "folds.tsv", sep="\t", index=False)
    print("folds:", [(h["split"], h["train"], h["test"], h["pcs"], round(h["screen_seconds"])) for h in headers])
    print(summary.to_string(index=False))
    print(gain.to_string(index=False))


if __name__ == "__main__":
    main()
