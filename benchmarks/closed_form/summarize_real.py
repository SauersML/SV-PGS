"""Summarize validate_real.py output: predicted vs realized GBLUP r^2 per split, and the LD-only portability prediction.

usage: summarize_real.py REAL.json
"""
import json
import sys

import numpy as np
import pandas as pd


def main():
    frame = pd.DataFrame(json.load(open(sys.argv[1])))
    frame = frame[frame["reml_h2"] > 0]
    rows = []
    for split, group in frame.groupby("split"):
        expected, realized = group["expected_sample_r2"].to_numpy(), group["realized_r2"].to_numpy()
        slope = float(np.polyfit(expected, realized, 1)[0])
        rows.append({"split": split, "genes": len(group), "mean_expected_r2": expected.mean(), "mean_realized_r2": realized.mean(),
                     "se_realized": realized.std(ddof=1) / np.sqrt(len(group)), "slope_realized_on_expected": slope,
                     "correlation": float(np.corrcoef(expected, realized)[0, 1]), "mean_reml_h2": group["reml_h2"].mean(),
                     "mean_predicted_rho2": group["predicted_rho2"].mean()})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    loso = frame[frame["split"].str.startswith("loso/")]
    wide_p = loso.pivot_table(index="gene_id", columns="split", values="expected_sample_r2")
    wide_r = loso.pivot_table(index="gene_id", columns="split", values="realized_r2")
    if "loso/AFR" in wide_p:
        others = [column for column in wide_p.columns if column != "loso/AFR"]
        predicted_ratio = wide_p["loso/AFR"].sum() / wide_p[others].mean(axis=1).sum()
        realized_ratio = wide_r["loso/AFR"].sum() / wide_r[others].mean(axis=1).sum()
        print(f"AFR portability ratio (sum AFR / sum mean-other): predicted from LD alone {predicted_ratio:.3f}; realized {realized_ratio:.3f}")


if __name__ == "__main__":
    main()
