"""Whether imputed SVs raise the held-out accuracy of standard PGS methods: the paired tests of aou_fastpgs.py's two
arms on every trait's held-out people, method by method.

Each trait's scores come from an aou_fastpgs.py job on its run (<run>/fastpgs/<trait>.npz in the ops bucket): every
method fitted on the snv_indel records and on every kept record, for the same training people with the same tuning
rule, and scored on the run's held-out people. Per method this prints:
- per trait and ancestry group (all, EUR, AFR): both arms' partial r^2 given the trait's covariates, the paired
  difference (with_sv less snv_indel) with its family bootstrap 95% interval, one-sided p (gain > 0) and one-sided upper 95% bound
  ("SVs add at most"), absolute and relative to the snv_indel arm; for a disease, both arms' AUC and their DeLong
  difference with family-clustered variance, each score residualized on the trait's covariates first (a raw score's AUC
  moves with the ancestry and age it carries, which the covariates hold);
- the combined test across traits, sv_gain_meta's pre-specified analysis (``analyse`` and ``report``), over the traits
  whose two arms differ. Where tuning chose the same model in both arms (one holding no SV), the scores are equal, the
  gain is exactly 0 with no variance, and the trait is listed as such: it has no z to combine.
Every method reads the same bootstrap resamples. A group enters with at least MINIMUM_GROUP_PEOPLE people and, for a
disease, more than 20 cases and controls. Aggregates only: every count of 1 to 39 is shown as <40.

``--screen`` is the wide screen's one pre-specified analysis (FASTPGS_SCREEN_PRESPEC.txt) in place of the above: per
trait the one-sided test of a with_sv gain for one method, and the Benjamini-Hochberg q over the traits that pass the
power threshold (``screen_report``). ``--exclude`` drops the long-read panel's members first, as sv_gain_meta does. It
is required: without it only the per-method report is printed, with --preliminary, and every line is marked PRELIMINARY.

    python -m benchmarks.bench_real.fastpgs_meta RUN [RUN ...] --exclude MANIFEST [--methods M ...] [--resamples N] [--permutations N]
    python -m benchmarks.bench_real.fastpgs_meta RUN [RUN ...] --screen ldpred2_auto --exclude MANIFEST
"""
from __future__ import annotations

import argparse
import math
import pathlib
from collections.abc import Sequence

import numpy as np
from scipy import stats

from benchmarks.bench_real.sv_gain_meta import (
    MINIMUM_GROUP_PEOPLE, OPS, PERMUTATIONS, PRELIMINARY, RESAMPLES, UPPER, PairedTrait, analyse, gains, report, shown, union,
)
from sv_pgs.held_out_comparison import auc_components, cluster_bootstrap_counts, paired_delta_auc
from sv_pgs.workspace_pipeline import _residualized

GROUPS = ("all", "eur", "afr")
ARMS = ("snv_indel", "with_sv")
SCREEN_CASES = 100
"""The screen's power threshold for a disease: at least this many held-out cases and as many held-out controls."""
SCREEN_PEOPLE = 1000
"""The screen's power threshold for a quantitative trait: at least this many held-out people."""
SCREEN_Q = 0.05


def read_run(name: str, root: str = OPS) -> list[dict[str, object]]:
    """Each trait an aou_fastpgs.py job left in the run: its held-out people's ids, families, ancestry, target,
    covariates and scores [methods, arms, people]."""
    traits = []
    for path in sorted(pathlib.Path(root, name, "fastpgs").glob("*.npz")):
        with np.load(path, allow_pickle=False) as arrays:
            if arrays["arms"].tolist() != list(ARMS):
                raise ValueError(f"{path}: arms {arrays['arms'].tolist()}")
            traits.append({
                "run": name, "name": str(arrays["trait"]), "binary": bool(arrays["binary"]), "methods": arrays["methods"].tolist(),
                "ids": np.asarray(arrays["research_ids"]).astype(object), "families": np.asarray(arrays["families"]),
                "ancestry": np.asarray(arrays["ancestry"]).astype(object), "target": np.asarray(arrays["target"], dtype=np.float64),
                "covariates": np.asarray(arrays["covariates"], dtype=np.float64), "scores": np.asarray(arrays["scores"], dtype=np.float64),
                "seed": int(arrays["seed"]),
            })
    if not traits:
        raise FileNotFoundError(f"{name}: no fastpgs scores under {root}/{name}/fastpgs")
    return traits


def paired(trait: dict[str, object], people: np.ndarray, method: str) -> PairedTrait | None:
    """The trait's two arms of ``method`` as sv_gain_meta pairs them, or None where an arm has no score (its fit failed)."""
    scores = trait["scores"][trait["methods"].index(method)]
    if not np.all(np.isfinite(scores)):
        return None
    return PairedTrait(trait["name"], trait["binary"], people, trait["target"], trait["covariates"], scores[0], scores[1])


def group_rows(trait: PairedTrait, ancestry: np.ndarray, group: str) -> np.ndarray | None:
    """The trait's people in ``group`` (None for all of them), or an empty array where the group is too small to show."""
    rows = None if group == "all" else np.flatnonzero(ancestry[trait.people] == group)
    target = trait.target if rows is None else trait.target[rows]
    cases = int(np.sum(target)) if trait.binary else None
    if target.shape[0] < MINIMUM_GROUP_PEOPLE or (cases is not None and min(cases, target.shape[0] - cases) <= 20):
        return np.zeros(0, dtype=np.int64)
    return rows


def cell(trait: PairedTrait, rows: np.ndarray | None, weights: np.ndarray, clusters: np.ndarray) -> dict[str, float]:
    """Both arms' partial r^2 and their paired difference on ``rows`` of the trait's people (None: all), with the family
    bootstrap's 95% interval, one-sided normal p (gain > 0) and one-sided upper 95% bound; for a disease, both arms' AUC
    and the DeLong difference with family-clustered variance and its one-sided p."""
    unit = np.ones((1, weights.shape[1]))
    gain, base = (values[0] for values in gains(unit, trait, rows=rows))
    gain_draws, base_draws = gains(weights, trait, rows=rows)
    error = float(np.nanstd(gain_draws, ddof=1))
    relative_error = float(np.nanstd(gain_draws / base_draws, ddof=1))
    found = {
        "n": trait.people.shape[0] if rows is None else rows.shape[0], "snv_indel": base, "with_sv": base + gain, "gain": gain,
        "ci_low": float(np.nanquantile(gain_draws, 0.025)), "ci_high": float(np.nanquantile(gain_draws, 0.975)),
        "p": float(stats.norm.sf(gain / error)) if error > 0 else math.nan, "upper": gain + UPPER * error,
        "relative": gain / base, "relative_upper": gain / base + UPPER * relative_error,
    }
    if trait.binary:
        keep = np.arange(trait.people.shape[0]) if rows is None else rows
        target = trait.target[keep]
        baseline, candidate = (_residualized(score[keep], trait.covariates[keep]) for score in (trait.baseline, trait.candidate))
        found["cases"] = int(target.sum())
        found["auc_snv_indel"] = auc_components(target, baseline).auc
        found["auc_with_sv"] = auc_components(target, candidate).auc
        delta = paired_delta_auc(target, baseline, candidate, clusters[trait.people[keep]])
        found.update(auc_gain=delta.estimate, auc_se=delta.standard_error, auc_upper=delta.estimate + UPPER * delta.standard_error,
                     auc_p=float(stats.norm.sf(delta.z_score)) if math.isfinite(delta.z_score) else math.nan)
    return found


def cell_line(name: str, group: str, found: dict[str, float]) -> str:
    text = (f"  {name}\t{group.upper()}\tn={shown(found['n'])}" + (f" cases={shown(found['cases'])}" if "cases" in found else "")
            + f"\tr2 snv_indel {found['snv_indel']:.5f} with_sv {found['with_sv']:.5f}\tgain {found['gain'] * 1e4:+.2f}e-4 "
            f"[{found['ci_low'] * 1e4:+.2f}, {found['ci_high'] * 1e4:+.2f}] one-sided p={found['p']:.3g} at most {found['upper'] * 1e4:+.2f}e-4 "
            f"({found['relative'] * 100:+.2f}%, at most {found['relative_upper'] * 100:+.2f}%)")
    if "auc_gain" in found:
        half = 1.959963984540054 * found["auc_se"]
        text += (f"\tAUC {found['auc_snv_indel']:.4f} -> {found['auc_with_sv']:.4f}, gain {found['auc_gain']:+.5f} "
                 f"[{found['auc_gain'] - half:+.5f}, {found['auc_gain'] + half:+.5f}] one-sided p={found['auc_p']:.3g} at most {found['auc_upper']:+.5f}")
    return text


def method_report(method: str, traits: Sequence[dict[str, object]], people: Sequence[np.ndarray], clusters: np.ndarray, labels: np.ndarray,
                  weights: np.ndarray, resamples: int, permutations: int) -> list[str]:
    """One method's per-trait cells and its combined test across traits."""
    pairs = [(trait, paired(trait, people[index], method)) for index, trait in enumerate(traits)]
    missing = [trait["name"] for trait, pair in pairs if pair is None]
    usable = [pair for _trait, pair in pairs if pair is not None]
    same = [pair.name for pair in usable if np.array_equal(pair.baseline, pair.candidate)]
    lines = [f"=== {method}: {len(usable)} traits" + (f"; no score in an arm for {', '.join(missing)}" if missing else "")
             + (f"; the same model in both arms (gain 0, not combined) for {', '.join(same)}" if same else "")]
    for pair in usable:
        if pair.name in same:
            lines.append(f"  {pair.name}\tALL\tthe same model in both arms: r2 {cell(pair, None, weights[:2], clusters)['snv_indel']:.5f} in each, gain 0")
            continue
        for group in GROUPS:
            rows = group_rows(pair, labels, group)
            if rows is not None and rows.shape[0] == 0:
                lines.append(f"  {pair.name}\t{group.upper()}\ttoo few people or cases to show")
                continue
            lines.append(cell_line(pair.name, group, cell(pair, rows, weights, clusters)))
    combined = [pair for pair in usable if pair.name not in same]
    if combined:
        result = analyse(combined, clusters, labels, int(traits[0]["seed"]), resamples, permutations)
        lines.extend(report(result, combined))
    return lines


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg q-values: q_(i) = min over j >= i of p_(j) m / j, capped at 1, in the input order."""
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p, kind="stable")
    ranked = p[order] * p.shape[0] / np.arange(1, p.shape[0] + 1)
    q = np.empty_like(p)
    q[order] = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    return q


def screen_admits(pair: PairedTrait) -> bool:
    """Whether a trait passes the screen's power threshold (SCREEN_CASES, SCREEN_PEOPLE), read from counts alone."""
    count = pair.target.shape[0]
    if pair.binary:
        cases = int(np.sum(pair.target))
        return min(cases, count - cases) >= SCREEN_CASES
    return count >= SCREEN_PEOPLE


def screen_report(method: str, traits: Sequence[dict[str, object]], people: Sequence[np.ndarray], weights: np.ndarray) -> list[str]:
    """The screen: per trait passing ``screen_admits``, the with_sv gain in held-out partial r^2 with its family bootstrap
    95% interval and SE, the one-sided normal p of gain / SE (H1: gain > 0) and the Benjamini-Hochberg q over those
    traits. Traits below the threshold, or with no score in an arm, are named without their counts."""
    pairs = [(trait["name"], paired(trait, people[index], method)) for index, trait in enumerate(traits)]
    missing = [name for name, pair in pairs if pair is None]
    below = [name for name, pair in pairs if pair is not None and not screen_admits(pair)]
    admitted = [pair for _name, pair in pairs if pair is not None and screen_admits(pair)]
    lines = [f"=== screen, {method}: {len(admitted)} traits in the family"
             + (f"; below the power threshold: {', '.join(below)}" if below else "")
             + (f"; no score in an arm: {', '.join(missing)}" if missing else "")]
    if not admitted:
        return lines
    unit = np.ones((1, weights.shape[1]))
    rows = []
    for pair in admitted:
        gain, base = (values[0] for values in gains(unit, pair))
        draws, _ = gains(weights, pair)
        error = float(np.nanstd(draws, ddof=1))
        z = gain / error if error > 0 else math.nan
        rows.append({"pair": pair, "gain": gain, "base": base, "low": float(np.nanquantile(draws, 0.025)), "high": float(np.nanquantile(draws, 0.975)),
                     "p": float(stats.norm.sf(z)) if math.isfinite(z) else 1.0})
    q = benjamini_hochberg([row["p"] for row in rows])
    for row, value in sorted(zip(rows, q), key=lambda item: item[0]["p"]):
        pair = row["pair"]
        cases = f" cases={shown(int(np.sum(pair.target)))}" if pair.binary else ""
        lines.append(f"  {pair.name}\tn={shown(pair.target.shape[0])}{cases}\tr2 snv_indel {row['base']:.5f} with_sv {row['base'] + row['gain']:.5f}"
                     f"\tgain {row['gain'] * 1e4:+.2f}e-4 [{row['low'] * 1e4:+.2f}, {row['high'] * 1e4:+.2f}] ({row['gain'] / row['base'] * 100:+.2f}%)"
                     f"\tone-sided p={row['p']:.3g} q={value:.3g}" + ("\tSV GAIN at q < 0.05" if value < SCREEN_Q else ""))
    hits = int(np.sum(q < SCREEN_Q))
    lines.append(f"  {hits} of {len(rows)} traits with a with_sv gain at BH q < {SCREEN_Q}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="Runs an aou_fastpgs.py job finished on.")
    parser.add_argument("--methods", nargs="+", help="Methods to test (default: every method the scores hold).")
    parser.add_argument("--root", default=OPS, help="The ops bucket's svpgs directory as this machine mounts it.")
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--screen", choices=("ldpred2_auto", "lassosum2", "ct"), help="Run the wide screen's analysis for this method only.")
    parser.add_argument("--exclude", help="A file of research ids to drop from every trait's held-out people (sv_gain_meta --exclude).")
    parser.add_argument("--preliminary", action="store_true", help="Report without --exclude, every line marked PRELIMINARY (not with --screen).")
    args = parser.parse_args(argv)
    if not args.exclude and (args.screen or not args.preliminary):
        parser.error("--exclude (the long-read panel's research ids) is required" + ("" if args.screen else "; --preliminary reports without it"))
    traits = [trait for name in args.runs for trait in read_run(name, args.root)]
    if args.exclude:
        with open(args.exclude, encoding="utf-8") as handle:
            rows = [line.rstrip("\n").split("\t") for line in handle if line.strip()]
        column = rows[0].index("research_id") if "research_id" in rows[0] else None
        ids = {row[column] for row in rows[1:]} if column is not None else {row[0] for row in rows}
        for index, trait in enumerate(traits):
            keep = np.array([str(value) not in ids for value in trait["ids"]], dtype=bool)
            print(f"{trait['name']}: {shown(int(np.sum(~keep)))} held-out people in {args.exclude} dropped")
            traits[index] = {**trait, **{key: np.asarray(trait[key])[keep] for key in ("ids", "families", "ancestry", "target", "covariates")},
                             "scores": np.asarray(trait["scores"])[:, :, keep]}
    people, clusters, labels, disagreements = union(traits)
    seed = traits[0]["seed"]
    weights = cluster_bootstrap_counts(np.zeros(int(clusters.max()) + 1, dtype=np.int64), args.resamples, np.random.default_rng(seed))[:, clusters]
    print(f"runs {' '.join(args.runs)}: traits {', '.join(trait['name'] for trait in traits)}; union of held-out people {shown(labels.shape[0])}, "
          f"clusters {shown(int(clusters.max()) + 1)}, ancestry labels differing between runs {shown(disagreements)}; "
          f"resamples {args.resamples}, permutations {args.permutations}")
    if args.screen:
        print("\n".join(screen_report(args.screen, traits, people, weights)), flush=True)
        return 0
    if not args.exclude:
        print(PRELIMINARY)
    for method in args.methods or traits[0]["methods"]:
        lines = method_report(method, traits, people, clusters, labels, weights, args.resamples, args.permutations)
        print("\n".join(lines if args.exclude else (f"PRELIMINARY {line}" for line in lines)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
