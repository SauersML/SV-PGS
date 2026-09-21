"""Score out-of-fold predictions from the harness.

The held-out metric is the within-group partial r^2 (lead ruling, review-stats STATS_REVIEW.md §0). Within one held-out
superpopulation T, R_T is the residual on [1, C] (the MAGE covariates) fitted over T's own held-out people. Per gene:
  partial_correlation  corr(R_T s, R_T y), with y the dataset's expression and s the saved prediction (0 when
           R_T s = 0): the partial correlation of the score with the expression given [1, C_T], inside the target
           ancestry, with its sign. The harness's saved truth (y minus a training-OLS covariate fit) is not scored: in
           a held-out ancestry that fit is an extrapolation whose within-group spread dominated the truth, and every
           score shared it. The saved truth only marks who was held out, and it is checked against y
           (HeldOut.expression_for), so a mismatched --dataset stops the report.
  r2       its square, the squared partial correlation. It is the share of the expression's WITHIN-GROUP RESIDUAL
           variance the score explains, not the rise in the total-response R^2: that rise is r2 times the share of
           the variance [1, C_T] leaves, so r2 is an upper bound on it. It charges neither the score's sign (a score
           that is the exact negative of the expression scores 1) nor its scale (any multiple of a score scores
           alike). The headline is r2; the other two say what it leaves out.
  oos_r2   1 - |R_T (y - s)|^2 / |R_T y|^2: the squared-error skill of the score as it stands against the target
           group's own covariate fit, so it charges the sign and the scale that r2 ignores. It is negative for a
           score worse than that fit, and is never clipped. Its nuisance fit is made inside the target group, so it
           is a target-residualized statistic, not the R^2 of a deployable predictor whose covariate model was
           fitted in training: this benchmark measures conditional association, not prospective skill (see E03).
  null_r2  1 / (n_T - rank[1, C_T]), the expectation of r2 under the null model below.
  mismatched_r2  the standing negative control: gene i's expression against the score of the next gene of its chunk on
           another chromosome (review-stats). Its mean must sit at null_r2; above it is signal no gene owns.
R_T s is the same for any two scores that differ by one covariate combination, so raw and covariate-adjusted scores
score alike, and fits made before the harness adjusted scores (C2) re-score from their saved predictions. Under loso
that is exact. Under random5 a group's people come from five training fits, so a score adjusted fold by fold keeps a
small fold-to-fold covariate term; the saved (adjusted) prediction is scored as it is.

Paired differences between two arms are averaged over genes; their standard error is the delete-one-chromosome jackknife
(genes on one chromosome share variants and trans structure, chromosomes do not), and a gene-level SE is shown when only
one chromosome is scored. They hold the people fixed; robust.py's family x chromosome bootstrap, which re-fits R_T in
every replicate, covers the sampling of people too.

Headline: both designs hold every person out exactly once, so the five superpopulations are pooled into one test.
Per gene, r^2 and paired differences are averaged over the groups and covariances are summed over them; the result
is one estimate per method and arm with one chromosome-jackknife SE, reported under superpopulation "pooled". The
per-group rows follow as secondary detail (the African-ancestry drop under loso is the per-group result that matters).

SV credit, per method and feature set with SV columns: the harness also predicts with every SV column held at its
training mean, so the SV part of a prediction is prediction - prediction_without_sv. The credit is that part's
share of the within-group covariance <R_T y, R_T s>, summed over genes (an exact additive split for linear
predictors), and the r^2 lost when SVs are held at their means.
"""
import argparse
import functools
import itertools
import json
import pathlib

import numpy as np
import pandas as pd

SUPERPOPULATIONS = ("AFR", "AMR", "EAS", "EUR", "SAS")
POOLED = "pooled"
FEATURE_SETS = ("snv", "snv_sv", "snv_pgsv", "sv", "pgsv", "snv_matched", "hgsvc3", "snv_hgsvc3", "ont", "snv_ont",
                "sv_merged", "snv_sv_merged", "pgsv_merged", "snv_pgsv_merged", "hgsvc3_merged", "snv_hgsvc3_merged", "gatksv", "snv_sv_cn",
                "svimp", "snv_svimp", "ctyper", "snv_ctyper", "hprc2", "snv_hprc2")
JOINT_SETS = ("snv_sv", "snv_pgsv", "snv_hgsvc3", "snv_ont", "snv_sv_merged", "snv_pgsv_merged", "snv_hgsvc3_merged", "snv_sv_cn", "snv_svimp",
              "snv_ctyper", "snv_hprc2")
# Within a method: adding each SV source to SNVs; each other source, and each collapsed source, against the panel SVs or its
# uncollapsed self; and SVs alone against an equal number of matched SNVs (and against each other). HGSVC3 is the PanGenie
# arm of record; HGSVC2 PanGenie (pgsv) stays as a labelled legacy comparison.
WITHIN_METHOD_COMPARISONS = (("snv_sv", "snv"), ("snv_pgsv", "snv"), ("snv_hgsvc3", "snv"), ("snv_ont", "snv"), ("snv_sv_merged", "snv"),
                             ("snv_pgsv_merged", "snv"), ("snv_hgsvc3_merged", "snv"), ("snv_sv_cn", "snv"),
                             ("snv_hgsvc3", "snv_sv"), ("snv_ont", "snv_sv"), ("snv_sv_cn", "snv_sv"),
                             ("snv_sv_merged", "snv_sv"), ("snv_pgsv_merged", "snv_pgsv"), ("snv_hgsvc3_merged", "snv_hgsvc3"),
                             ("snv_svimp", "snv"), ("snv_svimp", "snv_sv"), ("snv_ctyper", "snv"), ("snv_ctyper", "snv_sv"), ("snv_hprc2", "snv"),
                             ("snv_hprc2", "snv_sv"), ("sv", "snv_matched"), ("pgsv", "snv_matched"), ("sv", "pgsv"))


def group_basis(covariates, weights=None):
    """An orthonormal basis of span[1, covariates] over one group's people (the rows of covariates), and its rank. With
    weights (one per person), of that design with each row scaled by sqrt(weight): the weighted within-group fit."""
    design = np.column_stack([np.ones(covariates.shape[0]), covariates])
    if weights is not None:
        design = design * np.sqrt(weights)[:, None]
    left, singular, _ = np.linalg.svd(design, full_matrices=False)
    # numpy.linalg.matrix_rank's default: a singular value counts when it exceeds the largest one times the larger
    # dimension times the float64 machine epsilon.
    rank = int(np.sum(singular > singular.max() * max(design.shape) * np.finfo(np.float64).eps))
    return left[:, :rank], rank


def residual_on(basis, values, weights=None):
    """Each row of values (rows x the group's people) minus its projection on the basis. A row in the span (a constant,
    or any covariate combination) has residual 0, and it is set so rather than left as float rounding noise, which would
    score as a random direction. A row is in the span when it is constant, or when its residual is within what rounding
    leaves of a row in the span: its storage precision (half an ulp per entry of a float32 prediction moves the row by at
    most eps32 / 2 of its norm) plus group_basis's rank tolerance for the float64 projection. With weights, the residual
    of the sqrt(weight)-scaled rows, so plain sums of products of two residuals are the weighted sums."""
    values = np.asarray(values)
    storage = np.finfo(values.dtype).eps if np.issubdtype(values.dtype, np.floating) else 0.0
    values = values.astype(np.float64)
    constant = np.ptp(values, axis=1) == 0
    if weights is not None:
        values = values * np.sqrt(weights)
    residual = values - (values @ basis) @ basis.T
    tolerance = storage + max(basis.shape) * np.finfo(np.float64).eps
    residual[constant | (np.linalg.norm(residual, axis=1) <= tolerance * np.linalg.norm(values, axis=1))] = 0.0
    return residual


def within_group_residual(values, covariates, weights=None):
    """R_T values: each row of values (rows x one group's people) minus its least-squares fit on [1, covariates] within
    the group, and the rank of [1, covariates]."""
    basis, rank = group_basis(covariates, weights)
    return residual_on(basis, values, weights), rank


def partial_scores(score, truth):
    """Per row of within-group residuals: the signed partial correlation corr(R s, R y) (0 when R s = 0), its square
    r2, and oos_r2 = 1 - |R y - R s|^2 / |R y|^2. Three estimands, not one: see this module's docstring."""
    product = np.sum(score * truth, axis=1)
    score_norm, truth_norm = np.sum(score ** 2, axis=1), np.sum(truth ** 2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.where(score_norm == 0, 0.0, product / np.sqrt(score_norm * truth_norm))
        return correlation, correlation ** 2, 1.0 - np.sum((truth - score) ** 2, axis=1) / truth_norm


class HeldOut:
    """The dataset side of the within-group rule: expression, covariates, superpopulations and each split's test people."""

    def __init__(self, dataset_dir):
        dataset_dir = pathlib.Path(dataset_dir)
        samples = pd.read_csv(dataset_dir / "samples.tsv", sep="\t")
        self.covariates = np.load(dataset_dir / "covariates.npy").astype(np.float64)
        self.expression = np.load(dataset_dir / "expression.npy", mmap_mode="r")
        self.gene_row = {gene: row for row, gene in enumerate(pd.read_csv(dataset_dir / "genes.tsv", sep="\t")["gene_id"])}
        index = {sample: position for position, sample in enumerate(samples["sample"])}
        self.tests = {split["name"]: np.array([index[sample] for sample in split["test"]], dtype=int)
                      for split in json.loads((dataset_dir / "splits.json").read_text())}
        self.groups = {group: np.flatnonzero(samples["Superpopulation"].to_numpy() == group) for group in SUPERPOPULATIONS}
        # (design, group, people, rank) of groups left unscored: [1, C_T] leaves their people no residual dimension.
        self.unscorable = set()

    def expression_for(self, genes: pd.DataFrame, truth: np.ndarray, design: str):
        """Each gene's expression y, NaN where the saved truth t is NaN (not held out), checked against t. Within every
        split's test people t is y minus a covariate combination, so R(t - y) = 0 up to rounding: t is stored in float32
        (at most half an ulp per entry, so |R(t - y)| <= eps32 |t| / 2) and both covariate fits are float64, far below
        that, so eps32 (|t| + |y|) bounds it."""
        missing = sorted(set(genes["gene_id"]) - set(self.gene_row))
        if missing:
            raise ValueError(f"{len(missing)} scored genes are not in the dataset, e.g. {missing[:3]}")
        expression = np.asarray(self.expression[[self.gene_row[gene] for gene in genes["gene_id"]]], dtype=np.float64)
        expression[~np.isfinite(truth)] = np.nan
        for name, test in self.tests.items():
            if not name.startswith(f"{design}/"):
                continue
            # Gene rows are checked in groups of equal missingness, each on its own held-out people. Checking only the
            # people every gene row has a finite truth for left every other cell unchecked, and under heterogeneous
            # missingness that intersection can be empty, which passed for "verified".
            finite = np.isfinite(truth[:, test])
            for pattern in np.unique(finite, axis=0):
                held, rows = test[pattern], np.flatnonzero((finite == pattern).all(axis=1))
                if held.size == 0 or rows.size == 0:
                    continue
                cells = np.ix_(rows, held)
                gap, _ = within_group_residual(truth[cells] - expression[cells], self.covariates[held])
                bound = np.finfo(np.float32).eps * (np.linalg.norm(truth[cells], axis=1) + np.linalg.norm(expression[cells], axis=1))
                if (np.linalg.norm(gap, axis=1) > bound).any():
                    raise ValueError(f"the saved truth of split {name} is not the dataset's expression minus a covariate fit: is --dataset this run's dataset?")
        return expression

    def raw_split_order(self, directory: pathlib.Path, tag: str, columns: int):
        """Which split each column of a raw-score array is, from the run's own record.

        The harness writes <tag>.raw_splits.json beside the arrays (since e448b16). A run from before that file
        records the same list in its <tag>.run.json ("splits"), and a merged run records its parts' lists in the
        order merge_splits stacked them. A layout with neither is refused by name, and so is one whose list does not
        have one entry per column: an array's split order is never guessed from a file name or a sort."""
        order = None
        raw_splits = directory / f"{tag}.raw_splits.json"
        record_path = directory / f"{tag}.run.json"
        if raw_splits.exists():
            order = json.loads(raw_splits.read_text())
        elif record_path.exists():
            record = json.loads(record_path.read_text())
            order = record.get("splits") or ([name for part in record["parts"] for name in part["splits"]] if "parts" in record else None)
        if order is None:
            raise ValueError(f"{directory}: {tag}'s raw scores have no split order. Write the splits, in the array's column order, "
                             f"to {raw_splits.name}; the run's own {record_path.name} lists them under 'splits'.")
        if len(order) != columns:
            raise ValueError(f"{directory}: {tag}'s split order names {len(order)} splits for {columns} raw-score columns")
        unknown = [name for name in order if name not in self.tests]
        if unknown:
            raise ValueError(f"{directory}: {tag}'s split order names splits the dataset does not have, e.g. {unknown[:3]}")
        return order

    def predictions(self, directory: pathlib.Path, tag: str, feature_set: str, masked: bool):
        """A saved prediction (with the SV columns held at their training means when masked). A fit whose raw score is
        constant over its split's people, train and test, predicts no differences, so its covariate-adjusted score is
        exactly 0; the harness's adjustment left float rounding noise there, which is set back to 0. Needs the raw
        scores (saved since e448b16); an earlier unadjusted prediction of such a fit is exactly constant already."""
        suffix = "_without_sv" if masked else ""
        predictions = np.load(directory / f"{tag}.{feature_set}.predictions{suffix}.npy")
        raw_path = directory / f"{tag}.{feature_set}.raw_scores{suffix}.npy"
        if raw_path.exists():
            raw = np.load(raw_path)
            for position, name in enumerate(self.raw_split_order(directory, tag, raw.shape[1])):
                block = raw[:, position]
                finite = np.isfinite(block)
                constant = finite.any(axis=1) & (np.where(finite, block, np.inf).min(axis=1) == np.where(finite, block, -np.inf).max(axis=1))
                cells = np.ix_(constant, self.tests[name])
                predictions[cells] = np.where(np.isfinite(predictions[cells]), 0.0, np.nan)
        return predictions

    def within_groups(self, design: str, expression: np.ndarray, *scores):
        """Per held-out group, and per set of genes held out on the same people: the gene rows, the people count, the
        rank of [1, C_T], R_T of the expression and R_T of each score."""
        for group, members in self.groups.items():
            held = np.isfinite(expression[:, members])
            for pattern in np.unique(held, axis=0):
                rows = np.flatnonzero((held == pattern).all(axis=1))
                people = members[pattern]
                if people.size == 0:
                    continue
                basis, rank = group_basis(self.covariates[people])
                if people.size <= rank:
                    self.unscorable.add((design, group, int(people.size), rank))
                    continue
                yield (group, rows, int(people.size), rank, residual_on(basis, expression[np.ix_(rows, people)]),
                       [residual_on(basis, score[np.ix_(rows, people)]) for score in scores])


@functools.lru_cache(maxsize=None)
def held_out(dataset_dir: pathlib.Path) -> HeldOut:
    return HeldOut(dataset_dir)


def per_gene_scores(results_dir: pathlib.Path, dataset_dir: pathlib.Path, method: str, design: str):
    """Per gene, feature set and held-out group: r2, oos_r2 and null_r2 under the within-group rule, and the group's
    held-out people count."""
    data, directory, frames = held_out(pathlib.Path(dataset_dir)), results_dir / method / design, []
    for genes_file in sorted(directory.glob("*.genes.tsv")):
        tag = genes_file.name.removesuffix(".genes.tsv")
        genes = pd.read_csv(genes_file, sep="\t")
        expression = data.expression_for(genes, np.load(directory / f"{tag}.truth.npy").astype(np.float64), design)
        for feature_set in FEATURE_SETS:
            if not (directory / f"{tag}.{feature_set}.predictions.npy").exists():
                continue
            predictions = data.predictions(directory, tag, feature_set, masked=False)
            for group, rows, people, rank, truth, (score,) in data.within_groups(design, expression, predictions):
                correlation, r2, oos_r2 = partial_scores(score, truth)
                partner = mismatched_partners(genes["chrom"].to_numpy()[rows])
                mismatched = np.where(partner >= 0, partial_scores(score[partner], truth)[1], np.nan)
                frames.append(pd.DataFrame({"gene_id": genes["gene_id"].to_numpy()[rows], "chrom": genes["chrom"].to_numpy()[rows], "method": method,
                                            "feature_set": feature_set, "design": design, "superpopulation": group,
                                            "partial_correlation": correlation, "r2": r2, "oos_r2": oos_r2,
                                            "null_r2": 1.0 / (people - rank), "mismatched_r2": mismatched, "people": people}))
    return pd.concat(frames, ignore_index=True)


def mismatched_partners(chromosomes: np.ndarray) -> np.ndarray:
    """The negative control's pairing (review-stats): each gene's partner is the next gene of its chunk, cyclically, on
    another chromosome (-1 when every gene shares one). The r2 of gene i's expression against its partner's score is
    signal that no gene owns, such as shared covariate or ancestry structure; under the within-group rule its mean sits
    at null_r2."""
    count = len(chromosomes)
    doubled = np.concatenate([chromosomes, chromosomes])
    # next_other[j]: the first position after j whose chromosome differs from j's, i.e. the end of j's run.
    next_other = np.full(2 * count, 2 * count)
    for position in range(2 * count - 2, -1, -1):
        next_other[position] = position + 1 if doubled[position + 1] != doubled[position] else next_other[position + 1]
    first = next_other[:count]
    return np.where(first < np.arange(count) + count, first % max(count, 1), -1)


def jackknife(differences: pd.Series, blocks: pd.Series):
    labels = blocks.unique()
    if len(labels) < 2:
        return float(differences.mean()), float(differences.std(ddof=1) / np.sqrt(len(differences))), "gene-level"
    estimates = np.array([differences[blocks != label].mean() for label in labels])
    count = len(labels)
    return float(differences.mean()), float(np.sqrt((count - 1) / count * ((estimates - estimates.mean()) ** 2).sum())), "chromosome jackknife"


def paired(scores: pd.DataFrame, arm_a, arm_b):
    """Paired differences between two arms, reported two ways when fits failed (--record-failures):
    complete-case (difference, se): only genes every compared arm completed, with failed_genes counted;
    intention-to-treat (itt_difference, itt_se): every gene, a failed group scored as the training-mean prediction
    (r^2 = 0), so an arm cannot gain by failing on hard genes. Without failures the two coincide."""
    rows = []
    key = ["gene_id", "chrom", "design", "superpopulation"]
    left = scores[(scores["method"] == arm_a[0]) & (scores["feature_set"] == arm_a[1])][key + ["r2"]]
    right = scores[(scores["method"] == arm_b[0]) & (scores["feature_set"] == arm_b[1])][key + ["r2"]]
    merged = left.merge(right, on=key, suffixes=("_a", "_b"))
    pooled = merged.groupby(["gene_id", "chrom", "design"], as_index=False)[["r2_a", "r2_b"]].agg(lambda values: values.mean(skipna=False)).assign(
        superpopulation=POOLED)
    pooled_itt = merged.fillna({"r2_a": 0.0, "r2_b": 0.0}).groupby(["gene_id", "chrom", "design"], as_index=False)[["r2_a", "r2_b"]].mean()
    itt = pd.concat([pooled_itt.assign(superpopulation=POOLED), merged.fillna({"r2_a": 0.0, "r2_b": 0.0})], ignore_index=True)
    itt_groups = dict(list(itt.groupby(["design", "superpopulation"], sort=False)))
    for (design, superpopulation), group in pd.concat([pooled, merged], ignore_index=True).groupby(["design", "superpopulation"], sort=False):
        failed = int(group[["r2_a", "r2_b"]].isna().any(axis=1).sum())
        complete = group.dropna(subset=["r2_a", "r2_b"])
        mean, error, kind = jackknife(complete["r2_a"] - complete["r2_b"], complete["chrom"])
        whole = itt_groups[(design, superpopulation)]
        itt_mean, itt_error, _ = jackknife(whole["r2_a"] - whole["r2_b"], whole["chrom"])
        rows.append({"arm_a": "/".join(arm_a), "arm_b": "/".join(arm_b), "design": design, "superpopulation": superpopulation, "genes": len(complete),
                     "mean_r2_a": complete["r2_a"].mean(), "mean_r2_b": complete["r2_b"].mean(), "difference": mean, "se": error, "se_kind": kind,
                     "failed_genes": failed, "itt_genes": len(whole), "itt_mean_r2_a": whole["r2_a"].mean(), "itt_mean_r2_b": whole["r2_b"].mean(),
                     "itt_difference": itt_mean, "itt_se": itt_error})
    return rows


def pooled_r2(scores: pd.DataFrame):
    """Per method, feature set and design: the mean over genes of each gene's r^2 (and its signed partial correlation,
    oos_r2, and the null r^2) averaged over the held-out groups, complete-case (failed genes dropped and counted) and
    intention-to-treat (a failed group scored as the training-mean prediction: r^2 = oos_r2 = 0)."""
    key = ["method", "feature_set", "design", "gene_id", "chrom"]
    per_gene = scores.groupby(key, as_index=False)[["partial_correlation", "r2", "oos_r2", "null_r2", "mismatched_r2"]].agg(
        lambda values: values.mean(skipna=False))
    per_gene_itt = scores.fillna({"r2": 0.0}).groupby(key, as_index=False)["r2"].mean()
    itt_groups = dict(list(per_gene_itt.groupby(["method", "feature_set", "design"])))
    rows = []
    for arm, group in per_gene.groupby(["method", "feature_set", "design"]):
        failed = int(group["r2"].isna().sum())
        complete = group.dropna(subset=["r2"])
        mean, error, kind = jackknife(complete["r2"], complete["chrom"])
        oos_mean, oos_error, _ = jackknife(complete["oos_r2"], complete["chrom"])
        correlation_mean, correlation_error, _ = jackknife(complete["partial_correlation"], complete["chrom"])
        whole = itt_groups[arm]
        itt_mean, itt_error, _ = jackknife(whole["r2"], whole["chrom"])
        rows.append(dict(zip(["method", "feature_set", "design"], arm), superpopulation=POOLED, genes=len(complete), mean_r2=mean, se=error, se_kind=kind,
                         null_r2=complete["null_r2"].mean(), mismatched_r2=complete["mismatched_r2"].mean(), mean_oos_r2=oos_mean, oos_se=oos_error,
                         mean_partial_correlation=correlation_mean, partial_correlation_se=correlation_error, failed_genes=failed, itt_genes=len(whole),
                         itt_mean_r2=itt_mean, itt_se=itt_error))
    return pd.DataFrame(rows)


def sv_credit(results_dir: pathlib.Path, dataset_dir: pathlib.Path, method: str, design: str):
    """Per gene, joint feature set and held-out group, under the within-group rule: the covariances <R_T y, R_T s> of the
    full prediction and of its SV part s - m (m: the SV columns held at their training means), and the r2 lost to m."""
    data, directory, frames = held_out(pathlib.Path(dataset_dir)), results_dir / method / design, []
    for genes_file in sorted(directory.glob("*.genes.tsv")):
        tag = genes_file.name.removesuffix(".genes.tsv")
        genes = pd.read_csv(genes_file, sep="\t")
        expression = data.expression_for(genes, np.load(directory / f"{tag}.truth.npy").astype(np.float64), design)
        for feature_set in JOINT_SETS:
            if not all((directory / f"{tag}.{feature_set}.{kind}.npy").exists() for kind in ("predictions", "predictions_without_sv")):
                continue
            full = data.predictions(directory, tag, feature_set, masked=False)
            masked = data.predictions(directory, tag, feature_set, masked=True)
            for group, rows, _, _, truth, (score, score_without, sv_part) in data.within_groups(design, expression, full, masked, full - masked):
                frames.append(pd.DataFrame({"gene_id": genes["gene_id"].to_numpy()[rows], "chrom": genes["chrom"].to_numpy()[rows], "method": method,
                                            "feature_set": feature_set, "design": design, "superpopulation": group,
                                            "covariance_full": np.sum(truth * score, axis=1), "covariance_sv": np.sum(truth * sv_part, axis=1),
                                            "r2_drop": partial_scores(score, truth)[1] - partial_scores(score_without, truth)[1]}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_sv_credit(credit: pd.DataFrame):
    # Genes with a failed fit in any group are dropped whole: a NaN covariance must not enter a sum as 0.
    failed = credit[["covariance_full", "covariance_sv"]].isna().any(axis=1)
    bad = set(map(tuple, credit.loc[failed, ["method", "feature_set", "design", "gene_id"]].to_numpy()))
    credit = credit[[tuple(row) not in bad for row in credit[["method", "feature_set", "design", "gene_id"]].to_numpy()]]
    pooled = credit.groupby(["method", "feature_set", "design", "gene_id", "chrom"], as_index=False).agg(
        covariance_full=("covariance_full", "sum"), covariance_sv=("covariance_sv", "sum"), r2_drop=("r2_drop", "mean")).assign(superpopulation=POOLED)
    rows = []
    for key, group in pd.concat([pooled, credit], ignore_index=True).groupby(["method", "feature_set", "design", "superpopulation"], sort=False):
        chromosomes = group["chrom"].unique()
        share = group["covariance_sv"].sum() / group["covariance_full"].sum()
        if len(chromosomes) > 1:
            leave_out = np.array([group.loc[group["chrom"] != label, "covariance_sv"].sum() / group.loc[group["chrom"] != label, "covariance_full"].sum() for label in chromosomes])
            share_se = float(np.sqrt((len(chromosomes) - 1) / len(chromosomes) * ((leave_out - leave_out.mean()) ** 2).sum()))
        else:
            share_se = float("nan")
        drop, drop_se, kind = jackknife(group["r2_drop"], group["chrom"])
        rows.append(dict(zip(["method", "feature_set", "design", "superpopulation"], key), genes=len(group), sv_share_of_covariance=share,
                         share_se=share_se, r2_drop=drop, r2_drop_se=drop_se, se_kind=kind))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True, help="one or more results directories; each (method, design, feature set) must come from one")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    results_dirs, dataset_dir = [pathlib.Path(path) for path in arguments.results], pathlib.Path(arguments.dataset)
    runs = [(results_dir, method, design) for results_dir in results_dirs for method in arguments.methods for design in ("random5", "loso")
            if (results_dir / method / design).exists()]
    scores = pd.concat([per_gene_scores(results_dir, dataset_dir, method, design).assign(results=str(results_dir)) for results_dir, method, design in runs],
                       ignore_index=True)
    sources = scores.groupby(["method", "design", "feature_set"])["results"].nunique()
    if (sources > 1).any():
        raise ValueError(f"an arm appears in more than one results directory: {list(sources[sources > 1].index)}")
    scores = scores.drop(columns="results")
    scores.to_csv(pathlib.Path(arguments.out) / "per_gene_r2.tsv.gz", sep="\t", index=False)
    present = set(zip(scores["method"], scores["feature_set"]))
    arms = [(method, feature_set) for method in arguments.methods for feature_set in FEATURE_SETS if (method, feature_set) in present]
    comparisons = []
    for method in arguments.methods:
        for feature_set, baseline in WITHIN_METHOD_COMPARISONS:
            if (method, feature_set) in present and (method, baseline) in present:
                comparisons += paired(scores, (method, feature_set), (method, baseline))
    for arm_a, arm_b in itertools.combinations(arms, 2):
        if arm_a[1] == arm_b[1] and arm_a[0] != arm_b[0]:
            comparisons += paired(scores, arm_a, arm_b)
    table = pd.DataFrame(comparisons)
    table.to_csv(pathlib.Path(arguments.out) / "paired_differences.tsv", sep="\t", index=False)
    credit = pd.concat([sv_credit(results_dir, dataset_dir, method, design) for results_dir, method, design in runs], ignore_index=True)
    if len(credit):
        summarize_sv_credit(credit).to_csv(pathlib.Path(arguments.out) / "sv_credit.tsv", sep="\t", index=False)
    summary = scores.groupby(["design", "superpopulation", "method", "feature_set"]).agg(mean=("r2", "mean"), count=("r2", "count"), null_r2=("null_r2", "mean"),
                                                                                       mismatched_r2=("mismatched_r2", "mean"),
                                                                                       mean_oos_r2=("oos_r2", "mean"), people=("people", "max"),
                                                                                       mean_partial_correlation=("partial_correlation", "mean")).reset_index()
    summary.to_csv(pathlib.Path(arguments.out) / "mean_r2.tsv", sep="\t", index=False)
    headline = pooled_r2(scores)
    headline.to_csv(pathlib.Path(arguments.out) / "pooled_r2.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.float_format", "{:.5f}".format):
        print("within-group partial r^2 given the covariates (R_T y, R_T s; STATS_REVIEW.md §0)")
        for design, group, people, rank in sorted(held_out(dataset_dir).unscorable):
            print(f"unscored: {design} {group}: {people} held-out people, [1, C] rank {rank}, no residual dimension")
        print("pooled held-out r^2 (headline)")
        print(headline.to_string(index=False))
        print("pooled paired differences (headline)")
        print(table[table["superpopulation"] == POOLED].to_string(index=False))
        print("per held-out group")
        print(summary.pivot_table(index=["design", "superpopulation"], columns=["method", "feature_set"], values="mean"))


if __name__ == "__main__":
    main()
