"""Within-ancestry permutation null for the SV gain of bench-real's top_variant baseline (critique/CRITIQUE_METHOD.md
item 5, EVALUATION.md's C-null).

Each replicate permutes people within every superpopulation and applies that one permutation to every panel SV column
of every gene, in training and test alike. That keeps each SV's allele frequency per ancestry and the SV-SV linkage, and
breaks the SVs' linkage to expression and to the SNVs. The snv_sv arm is then refitted and scored exactly as the harness
does, on the loso splits; the snv arm doesn't depend on the SV columns. Under loso a within-superpopulation permutation
never moves a person between training and test, so no training column changes its polymorphism, and no test person
changes which group scores them.

The null must run the benchmark's own estimator, or it calibrates something else. Both halves of that estimator are
projections onto the complement of the covariates, and neither commutes with a permutation of people:
  selection  baselines.top_variant picks the largest |corr(M x, y)| among the columns M leaves outside the covariate
             span, with M the residual maker of [1, C] over the training people. So the null projects too.
  scoring    report.py scores corr(R_T s, R_T y)^2 within the held-out group, with R_T the residual maker of
             [1, C_T] over that group's people. A score differing from the column by a covariate combination or a
             positive scale scores alike, so the winning column's own R_T is the arm's r^2.
top_variant stays exact and cheap under both. Writing sigma for the permutation of a replicate, <R x[sigma], R y> is
<x, (R y)[sigma^-1]> and ||R x[sigma]||^2 is ||x||^2 - ||Q[sigma^-1]' x||^2 with Q an orthonormal basis of the design:
one matrix product of the columns against the permuted residual responses, and one against the permuted rows of Q. The
columns themselves are never permuted or re-projected. The identity replicate therefore reproduces the harness's own
top_variant snv_sv predictions and the report's r^2 of them exactly; the test asserts it against the pipeline itself
for a permuted replicate too, and the run checks it against stored harness results.

One difference from the pipeline is deliberate: where report.py decides that a stored score's residual is float
rounding rather than a direction, it measures the residual against the score's norm, and the null measures it against
the column's. The two differ only for a column whose held-out residual is already at rounding level, which scores 0
either way.
"""
import argparse
import hashlib
import json
import pathlib
from multiprocessing import get_context

import numpy as np
import pandas as pd

from benchmarks.bench_real import baselines, harness, report

GROUPS = ("AFR", "AMR", "EAS", "EUR", "SAS")
DOUBLE_EPSILON = np.finfo(np.float64).eps
# Replicates per matrix product against the permuted rows of the design basis; it only trades memory for calls.
REPLICATE_BLOCK = 64


def seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")


def within_group_permutations(groups: np.ndarray, count: int, generator: np.random.Generator):
    """(count + 1, n) inverse permutations; row 0 is the identity. Each maps a person to someone in their own group."""
    inverse = np.tile(np.arange(len(groups)), (count + 1, 1))
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        for row in range(1, count + 1):
            inverse[row, members] = members[generator.permutation(len(members))]
    return inverse


def local_permutations(inverse: np.ndarray, people: np.ndarray, sample_count: int):
    """The rows of `inverse` restricted to `people` and renumbered to positions among them.

    A permutation that moved one of these people to someone outside them would not be a permutation of this split's
    training or held-out set, and the fit it defines is not the one this null means."""
    position = np.full(sample_count, -1, dtype=np.int64)
    position[people] = np.arange(len(people))
    local = position[inverse[:, people]]
    if (local < 0).any():
        raise ValueError("a permutation moves a person out of the split's own people: it is not within superpopulation")
    return local


def projected(columns: np.ndarray, basis: np.ndarray, response: np.ndarray, local: np.ndarray):
    """Per column j and replicate r: <R x_j[sigma_r], response> and ||R x_j[sigma_r]||^2, with R the residual maker
    of `basis` over these people and sigma_r the permutation that `local[r]` inverts.

    R is symmetric and idempotent, so the first is <x_j, response[local_r]> once `response` is already R response,
    and the second is ||x_j||^2 - ||basis[local_r]' x_j||^2. Both read the permutation through permuted rows, so the
    columns are multiplied as they are."""
    products = columns.T @ response[local.T]
    squared = (columns ** 2).sum(axis=0)
    norms = np.empty((columns.shape[1], local.shape[0]))
    for start in range(0, local.shape[0], REPLICATE_BLOCK):
        block = local[start:start + REPLICATE_BLOCK]
        # (replicates * rank, people) @ (people, columns): one matrix product for the whole block.
        rotated = np.ascontiguousarray(basis[block].transpose(0, 2, 1)).reshape(-1, basis.shape[0]) @ columns
        norms[:, start:start + len(block)] = (squared - (rotated.reshape(len(block), basis.shape[1], -1) ** 2).sum(axis=1)).T
    return products, np.maximum(norms, 0.0)


def selection_strength(products: np.ndarray, norms: np.ndarray, span_level: np.ndarray):
    """|corr(R x[sigma], R y)| up to the response's own norm, which is common to every column of a replicate, and
    -inf for a column the projection leaves inside the covariate span (baselines.top_variant's own test, whose
    threshold is the column's untouched norm and so is the same under every permutation)."""
    outside = norms > (span_level ** 2)[:, None]
    strength = np.full(products.shape, -np.inf)
    np.divide(np.abs(products), np.sqrt(norms, where=outside, out=np.ones_like(norms)), out=strength, where=outside)
    return strength


def best_column(strength: np.ndarray, columns: np.ndarray):
    """The column top_variant would pick per replicate, and its strength: the first largest among those outside the
    span, in column order, as np.argmax over the columns in order gives."""
    if len(columns) == 0:
        return np.full(strength.shape[1], -1), np.full(strength.shape[1], -np.inf)
    chosen = np.argmax(strength, axis=0)
    return columns[chosen], strength[chosen, np.arange(strength.shape[1])]


def held_out_r2(products: np.ndarray, norms: np.ndarray, truth_squared: float, constant: np.ndarray, column_squared: np.ndarray, tolerance: float):
    """report.py's r^2 of the column as a score: corr(R_T x[sigma], R_T y)^2, and 0 where R_T x is not a direction
    (a column constant over the group's people, or a residual at the rounding level of the projection)."""
    zero = constant[:, None] | (norms <= (tolerance ** 2) * column_squared[:, None])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(zero | (truth_squared == 0), 0.0, products ** 2 / (norms * truth_squared))


def gene_null(dataset, gene_row: int, inverse: np.ndarray):
    """Per loso split: (observed and null) r2 gain of top_variant snv_sv over snv, and whether an SV won the selection."""
    window = harness.load_gene_window(dataset, gene_row)
    replicates = inverse.shape[0]
    gains, wins = [], []
    for name in sorted(split for split in dataset.splits if split.startswith("loso/")):
        train_all, test_all, test_phenotype, test_index = harness.build_gene_task(dataset, window, dataset.splits[name])
        train, test = harness.subset(train_all, test_all, "snv_sv", name)
        train_index = np.array([dataset.sample_index[sample] for sample in dataset.splits[name]["train"]])
        genotypes, test_genotypes = np.asarray(train.genotypes, dtype=np.float64), np.asarray(test, dtype=np.float64)
        structural, small = np.flatnonzero(train.variants.is_sv), np.flatnonzero(~train.variants.is_sv)
        train_basis, _ = report.group_basis(train.covariates)
        test_basis, test_rank = report.group_basis(dataset.covariates[test_index])
        if len(structural) == 0 or len(test_index) <= test_rank:
            # No SV to permute, or a held-out group the report leaves unscored: no arm differs from the other.
            gains.append(np.zeros(replicates))
            wins.append(np.zeros(replicates, dtype=bool))
            continue
        # M y and R_T y, the responses both halves of the estimator see; the phenotype is already orthogonal to
        # [1, C] in training, and the saved truth to a covariate combination of the test people, but the estimator
        # projects and so does this.
        response = train.phenotype - train_basis @ (train_basis.T @ train.phenotype)
        truth = report.residual_on(test_basis, np.asarray(test_phenotype, dtype=np.float64)[None])[0]
        local_train = local_permutations(inverse, train_index, len(dataset.samples))
        local_test = local_permutations(inverse, test_index, len(dataset.samples))
        identity_train, identity_test = local_train[:1], local_test[:1]
        # baselines.top_variant's own span threshold, per column and untouched by a permutation of its rows.
        span_level = max(genotypes.shape[0], 1 + train.covariates.shape[1]) * baselines.DOUBLE_EPSILON * np.linalg.norm(genotypes, axis=0)
        small_products, small_norms = projected(genotypes[:, small], train_basis, response, identity_train)
        small_column, small_strength = best_column(selection_strength(small_products, small_norms, span_level[small]), small)
        structural_products, structural_norms = projected(genotypes[:, structural], train_basis, response, local_train)
        structural_column, structural_strength = best_column(selection_strength(structural_products, structural_norms, span_level[structural]), structural)
        # An SV wins when it is strictly stronger, or ties and sits at a lower column, which is where np.argmax over
        # every column in order would have left the tie.
        sv_wins = np.isfinite(structural_strength) & ((structural_strength > small_strength[0])
                                                      | ((structural_strength == small_strength[0]) & (structural_column < small_column[0])))
        truth_squared = float(truth @ truth)
        tolerance = DOUBLE_EPSILON * (1 + max(test_basis.shape))
        constant = np.ptp(test_genotypes, axis=0) == 0
        column_squared = (test_genotypes ** 2).sum(axis=0)
        snv_r2 = 0.0
        if np.isfinite(small_strength[0]):
            chosen = small_column[:1]
            snv_r2 = float(held_out_r2(*projected(test_genotypes[:, chosen], test_basis, truth, identity_test), truth_squared,
                                       constant[chosen], column_squared[chosen], tolerance)[0, 0])
        structural_r2 = held_out_r2(*projected(test_genotypes[:, structural], test_basis, truth, local_test), truth_squared,
                                    constant[structural], column_squared[structural], tolerance)
        won = structural_r2[np.searchsorted(structural, structural_column), np.arange(replicates)]
        gains.append(np.where(sv_wins, won - snv_r2, 0.0))
        wins.append(sv_wins)
    return window.gene_id, window.chrom, np.mean(gains, axis=0), np.sum(wins, axis=0)


_STATE = {}


def _init(dataset_dir, inverse):
    _STATE["dataset"] = harness.Dataset(dataset_dir)
    _STATE["inverse"] = inverse


def _work(gene_row):
    return gene_null(_STATE["dataset"], gene_row, _STATE["inverse"])


def harness_gain(dataset, directory: pathlib.Path):
    """The snv_sv minus snv r^2 gain of a stored harness run, under report.py's own rule, per gene.

    The report scores the dataset's expression; the saved truth differs from it by a covariate combination of the
    held-out people, which R_T removes, so it scores the same and needs no second reading of the dataset."""
    groups = dataset.samples["Superpopulation"].to_numpy()
    gains = {}
    for genes_file in directory.glob("*.genes.tsv"):
        tag = genes_file.name.removesuffix(".genes.tsv")
        table = pd.read_csv(genes_file, sep="\t")
        truth = np.load(directory / f"{tag}.truth.npy").astype(np.float64)
        arms = {arm: np.load(directory / f"{tag}.{arm}.predictions.npy").astype(np.float64) for arm in ("snv_sv", "snv")}
        for row, gene_id in enumerate(table["gene_id"]):
            values = []
            for group in GROUPS:
                held = np.flatnonzero((groups == group) & np.isfinite(truth[row]))
                basis, rank = report.group_basis(dataset.covariates[held]) if held.size else (None, 0)
                if held.size <= rank:
                    continue
                residual_truth = report.residual_on(basis, truth[row:row + 1, held])
                scored = {arm: report.partial_scores(report.residual_on(basis, values_of[row:row + 1, held]), residual_truth)[1][0]
                          for arm, values_of in arms.items()}
                values.append(scored["snv_sv"] - scored["snv"])
            if values:
                gains[gene_id] = float(np.mean(values))
    return gains


def run(dataset_dir, gene_prefix, permutations, workers, observed_dir, out_dir):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset = harness.Dataset(dataset_dir)
    inverse = within_group_permutations(dataset.samples["Superpopulation"].to_numpy(), permutations,
                                        np.random.default_rng(seed("bench-real/perm_null/top_variant")))
    rows = dataset.gene_rows([f"chr{number}" for number in range(1, 23)], gene_prefix)
    results = []
    with get_context("fork").Pool(workers, initializer=_init, initargs=(dataset_dir, inverse)) as pool:
        for result in pool.imap_unordered(_work, rows, chunksize=4):
            results.append(result)
    results.sort(key=lambda item: item[0])
    genes = pd.DataFrame({"gene_id": [item[0] for item in results], "chrom": [item[1] for item in results]})
    gains = np.stack([item[2] for item in results])
    wins = np.stack([item[3] for item in results])
    np.save(out / "gains.npy", gains)
    np.save(out / "sv_wins.npy", wins)
    genes.to_csv(out / "genes.tsv", sep="\t", index=False)
    pooled = gains.mean(axis=0)
    null = pooled[1:]
    per_gene_p = (1 + (gains[:, 1:] >= gains[:, :1]).sum(axis=1)) / (permutations + 1)
    genes.assign(observed_gain=gains[:, 0], null_mean=gains[:, 1:].mean(axis=1), permutation_p=per_gene_p,
                 observed_sv_wins=wins[:, 0], null_sv_wins_mean=wins[:, 1:].mean(axis=1)).to_csv(out / "per_gene.tsv", sep="\t", index=False)
    summary = {"genes": len(genes), "permutations": permutations, "observed_pooled_gain": float(pooled[0]),
               "null_pooled_gain_mean": float(null.mean()), "null_pooled_gain_sd": float(null.std(ddof=1)),
               "pooled_p_one_sided": float((1 + (null >= pooled[0]).sum()) / (permutations + 1)),
               "observed_gene_split_sv_wins": int(wins[:, 0].sum()), "null_gene_split_sv_wins_mean": float(wins[:, 1:].sum(axis=0).mean()),
               "null_gene_split_sv_wins_max": int(wins[:, 1:].sum(axis=0).max()),
               "genes_with_min_attainable_p": int((per_gene_p == 1 / (permutations + 1)).sum())}
    if observed_dir is not None:
        truth_gain = harness_gain(dataset, pathlib.Path(observed_dir) / "top_variant" / "loso")
        matched = [(gain, truth_gain[gene]) for gene, gain in zip(genes["gene_id"], gains[:, 0]) if gene in truth_gain]
        summary["identity_vs_harness_max_abs_difference"] = float(max(abs(a - b) for a, b in matched)) if matched else None
        summary["identity_vs_harness_genes"] = len(matched)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gene-prefix", type=int, required=True)
    parser.add_argument("--permutations", type=int, required=True, help="the one-sided p value's resolution is 1/(permutations + 1)")
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--observed", help="bench-real results directory, to check the identity permutation against the harness")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.dataset, arguments.gene_prefix, arguments.permutations, arguments.workers, arguments.observed, arguments.out), indent=1))


if __name__ == "__main__":
    main()
