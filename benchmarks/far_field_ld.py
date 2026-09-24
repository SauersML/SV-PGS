"""The far field's structure on a bench-sim cohort: is the LD beyond the band chance (r^2 ~ 1/n) or systematic?

A random sample of ``--variants`` records along the chromosome is read on the training samples, standardized and
projected on the covariates (the harness's own: sex, age, batch and the PCs, with an intercept), exactly as Stage 0
projects its Grams. Then:

1. Excess r^2 by genomic distance: for each distance bin, the pairs' mean r^2 less the null 1/n' (n' = n - rank C),
   with its standard error, pooled and within each ancestry group (each group's own centring and projection). Chance
   LD has excess 0 at every distance; local-ancestry LD (admixture) leaves an excess at megabase distances that does
   not shrink with n.
2. Low rank: the cross-correlation between the sampled variants of the chromosome's two ends (farther apart than any
   band) against its null, a K x K matrix of independent N(0, 1/n') entries whose largest singular value is about
   (2 sqrt(K)) / sqrt(n') (Marchenko-Pastur); the top singular values over that edge, and the share of the excess
   squared Frobenius norm the top few carry.

Evidence label [sim]: bench-sim's simulated cohort. Diagnostic only; nothing here enters a fit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.bench_sim.harness import ARMS, covariate_matrix, measured_records

DISTANCE_EDGES = (0, 10_000, 100_000, 1_000_000, 5_000_000, 20_000_000, 1_000_000_000)
"""Distance bins, in base pairs: within LD blocks, the band's reach, and megabase scales where only local ancestry
(or a sweep) correlates variants."""


def projected(codes: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """Columns standardized (population sd) and projected on [1, covariates] (samples x variants)."""
    values = codes.astype(np.float64)
    values -= values.mean(axis=0)
    scale = values.std(axis=0)
    keep = scale > 0.0
    values = values[:, keep] / scale[keep]
    basis, _ = np.linalg.qr(np.column_stack([np.ones(values.shape[0]), covariates]))
    values -= basis @ (basis.T @ values)
    values /= np.linalg.norm(values, axis=0) / np.sqrt(values.shape[0])
    return values, keep


def excess_by_distance(values: np.ndarray, positions: np.ndarray, residual: float) -> list[dict]:
    correlation = (values.T @ values) / values.shape[0]
    squared = correlation * correlation
    upper = np.triu_indices(values.shape[1], 1)
    distance = np.abs(positions[:, None] - positions[None, :])[upper]
    pair_squares = squared[upper]
    rows = []
    for low, high in zip(DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:]):
        inside = (distance >= low) & (distance < high)
        count = int(inside.sum())
        if count < 2:
            continue
        chosen = pair_squares[inside]
        excess = float(chosen.mean() - 1.0 / residual)
        rows.append({
            "from_bp": low, "to_bp": high, "pairs": count, "excess_r2_times_n": excess * residual,
            "standard_error_times_n": float(chosen.std() / np.sqrt(count)) * residual,
        })
    return rows


def far_rank(values: np.ndarray, positions: np.ndarray, residual: float, ends: int) -> dict:
    order = np.argsort(positions)
    first, last = order[:ends], order[-ends:]
    gap = float(positions[last].min() - positions[first].max())
    cross = (values[:, first].T @ values[:, last]) / values.shape[0]
    singular = np.linalg.svd(cross, compute_uv=False)
    edge = 2.0 * np.sqrt(ends) / np.sqrt(residual)
    total = float(np.sum(cross * cross))
    null_total = ends * ends / residual
    excess = max(total - null_total, 0.0)
    shares = {}
    for top in (1, 2, 5, 10):
        above = float(np.sum(np.maximum(singular[:top] ** 2 - edge ** 2, 0.0)))
        shares[str(top)] = above / excess if excess > 0.0 else 0.0
    return {
        "ends": ends, "gap_bp": gap, "top_singular_over_null_edge": (singular[:10] / edge).tolist(),
        "frobenius_over_null": total / null_total, "excess_share_of_top": shares,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--arm", default="truth")
    parser.add_argument("--variants", type=int, default=4000)
    parser.add_argument("--ends", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    samples = np.load(arguments.cohort / "samples.npz")
    train = np.flatnonzero(~samples["is_test"])
    records = np.flatnonzero(measured_records(arguments.cohort))
    positions_all = np.load(arguments.cohort / "variants.npz")["pos"][records]
    generator = np.random.default_rng(arguments.seed)
    chosen = np.sort(generator.choice(records.size, size=min(arguments.variants, records.size), replace=False))
    observed = np.load(arguments.cohort / ARMS[arguments.arm][0], mmap_mode="r")
    codes = np.asarray(observed[records[chosen]])[:, train].T
    covariates, _names = covariate_matrix(arguments.cohort, arguments.arm)
    covariates = covariates[train]
    groups = samples["group"][train]
    names = [str(name) for name in samples["group_names"]]
    report = {"variants": int(chosen.size), "training_samples": int(train.size), "groups": {}}
    values, keep = projected(codes, covariates)
    positions = positions_all[chosen][keep].astype(np.float64)
    residual = float(values.shape[0] - np.linalg.matrix_rank(np.column_stack([np.ones(values.shape[0]), covariates])))
    report["pooled"] = {"by_distance": excess_by_distance(values, positions, residual), "far": far_rank(values, positions, residual, arguments.ends)}
    for index, name in enumerate(names):
        members = groups == index
        if members.sum() < 2 * covariates.shape[1] + 10:
            continue
        group_values, group_keep = projected(codes[members], covariates[members])
        group_positions = positions_all[chosen][group_keep].astype(np.float64)
        group_residual = float(group_values.shape[0] - np.linalg.matrix_rank(np.column_stack([np.ones(group_values.shape[0]), covariates[members]])))
        report["groups"][name] = {
            "samples": int(members.sum()), "by_distance": excess_by_distance(group_values, group_positions, group_residual),
            "far": far_rank(group_values, group_positions, group_residual, min(arguments.ends, group_values.shape[1] // 3)),
        }
    arguments.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
