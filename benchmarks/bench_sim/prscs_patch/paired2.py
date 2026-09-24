"""Paired bootstrap 95% CIs (1,000 resamples of the test people) of method minus SV-PGS, bench-sim truth arm, on both the
phenotype incremental r2 and genetic_r2 (squared partial correlation with the simulated genetic value beyond the
covariates, main's harness.genetic_accuracy). The SV-PGS reference is the first of svpgs_v2, svpgs_main, svpgs_full that exists.
usage: paired2.py <scenarios,> <methods,>"""
import sys
from pathlib import Path

import numpy as np

from benchmarks.bench_sim.harness import covariate_matrix, incremental_r2

C = Path("/scratch.global/sauer354/svpgs-team/bench-sim/v7/cohort/chr22")
D = Path("/scratch.global/sauer354/svpgs-team/bench-sim/v7/dev")
MINE = Path("/scratch.global/sauer354/svpgs-team/agents/baselines-genome/results")
LEAD = Path("/scratch.global/sauer354/svpgs-team/lead/bench-sim-truth/results/truth")


def genetic_r2(genetic, prediction, base):
    """harness.genetic_accuracy's squared partial correlation (main f16de460)."""
    def residual(values):
        coefficients, *_ = np.linalg.lstsq(base, values, rcond=None)
        return values - base @ coefficients
    g, p = residual(genetic), residual(prediction)
    if not np.any(p):
        return 0.0
    return float((g @ p) ** 2 / ((g @ g) * (p @ p)))


samples = np.load(C / "samples.npz")
test = np.flatnonzero(samples["is_test"])
cov, _ = covariate_matrix(C, "truth")
cov = cov[test]
for scenario in sys.argv[1].split(","):
    truth = np.load(D / f"scenario_{scenario}" / "truth.npz")
    y, g = truth["phenotype"][test], np.asarray(truth["genetic_value"][test], dtype=np.float64)
    reference = next((LEAD / name / f"scenario_{scenario}" / "prediction.npz" for name in ("svpgs_v2", "svpgs_main", "svpgs_full")
                      if (LEAD / name / f"scenario_{scenario}" / "prediction.npz").exists()), None)
    if reference is None:
        print(f"scenario {scenario}: no SV-PGS prediction yet", flush=True)
        continue
    base_pred = np.load(reference)["total"].astype(np.float64)
    rng = np.random.default_rng(0)
    draws = [rng.integers(0, y.size, y.size) for _ in range(1000)]
    design = np.column_stack([np.ones(y.size), cov])
    ref_r2, ref_g = incremental_r2(y, base_pred, cov)[0], genetic_r2(g, base_pred, design)
    print(f"scenario {scenario} reference {reference.parent.parent.name}: r2 {ref_r2:.4f} genetic_r2 {ref_g:.4f}", flush=True)
    for method in sys.argv[2].split(","):
        path = MINE / method / f"scenario_{scenario}" / "prediction.npz"
        if not path.exists():
            continue
        pred = np.load(path)["total"].astype(np.float64)
        point = incremental_r2(y, pred, cov)[0] - ref_r2
        gpoint = genetic_r2(g, pred, design) - ref_g
        diffs, gdiffs = [], []
        for i in draws:
            diffs.append(incremental_r2(y[i], pred[i], cov[i])[0] - incremental_r2(y[i], base_pred[i], cov[i])[0])
            gdiffs.append(genetic_r2(g[i], pred[i], design[i]) - genetic_r2(g[i], base_pred[i], design[i]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        glo, ghi = np.percentile(gdiffs, [2.5, 97.5])
        # one row per (scenario, method) in results/paired_table/, collected into one table by collect_table.sh
        import json
        score = json.loads((MINE / method / f"scenario_{scenario}" / "score.json").read_text())
        rows_dir = MINE / "paired_table"
        rows_dir.mkdir(exist_ok=True)
        fields = [scenario, method, reference.parent.parent.name, f"{point + ref_r2:.6g}", f"{score['calibration_slope']:.6g}",
                  f"{ref_r2:.6g}", f"{point:.6g}", f"{lo:.6g}", f"{hi:.6g}", f"{gpoint + ref_g:.6g}", f"{ref_g:.6g}",
                  f"{gpoint:.6g}", f"{glo:.6g}", f"{ghi:.6g}", f"{score['oracle_r2']:.6g}"]
        (rows_dir / f"{scenario}__{method}.tsv").write_text("\t".join(fields) + "\n")
        print(f"scenario {scenario} {method}: r2 {point + ref_r2:.4f} d {point:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
              f"genetic_r2 {gpoint + ref_g:.4f} d {gpoint:+.4f} [{glo:+.4f}, {ghi:+.4f}]", flush=True)
