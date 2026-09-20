"""The genome-wide per-gene SV table: which genes gain from SVs, and whether one SV event or several carry the gain.

Inputs:
  - a results directory written by the harness with saved SV effects (<tag>.sv_coefficients.tsv.gz), for one method,
    design and joint feature set (e.g. mr_ash / loso / snv_sv);
  - robust.py's sv_gene_tests.tsv.gz for the same arm: each gene's SV gain, its split into the SV part and the SNV
    refit, and their family-bootstrap SEs, one-sided p values and Benjamini-Hochberg q values;
  - the dataset, for genotypes, gene structure and the segmental-duplication track.

Per gene, over the held-out groups of the design (each person held out once under loso and random5): SV j contributes
c_j = effect_j (x_j - training mean_j) to each held-out person's score, and the SV part is s = sum_j c_j. SVs that are one
event called more than once (genotype r^2 above DUPLICATE_R2 over all samples, or the same span by 50% reciprocal overlap,
the SV-callset matching criterion of Collins et al. 2020, Nature 581:444) are merged: an event's contribution is the sum
of its SVs'. Reported per gene:
  - events_for_90pct: how many events, largest share first, carry 90% of var(s), each event's share being its additive
    share cov(c_e, s) / var(s);
  - effective_events: (sum_e var(c_e))^2 / sum_e var(c_e)^2;
  - the leading event's leading SV: type, length, distance to the TSS, allele frequency, whether it overlaps the gene
    body or an exon, its largest r^2 with any panel SNV/indel of the window, the fraction of its span in segmental
    duplications (UCSC genomicSuperDups; a proxy for cross-mapping of short reads), and in how many splits it leads.
  - with --crossmap, lead_sv_crossmappable_partners: the number of genes inside the lead SV's span that are
    cross-mappable with the target gene (Saha & Battle 2018, F1000Research 7:1860; GENCODE v26 scores, matched by
    unversioned gene id), and lead_sv_max_crossmappability, the largest such score. Any positive score counts: that is
    the paper's own definition (at least one k-mer cross-maps between the two genes), and the pair file lists only such
    pairs. Reads from those genes can be counted as the target's, so an SV deleting or duplicating them can move the
    target's measured expression with no regulatory effect. The flag marks an artifact route, not an artifact: only
    requantification from uniquely mapping reads settles an individual gene.
  - class: "none" when the SV part is not significant (BH q above the conventional 0.05 FDR level); otherwise
    "single" when one event carries 90% of var(s), else "multi".
"""
import argparse
import pathlib
from multiprocessing import get_context

import numpy as np
import pandas as pd

from benchmarks.bench_real import harness

DUPLICATE_R2 = 0.8
RECIPROCAL_OVERLAP = 0.5
VARIANCE_SHARE = 0.9
FDR_LEVEL = 0.05


def overlap(first_start, first_end, second_start, second_end):
    return max(0, min(first_end, second_end) - max(first_start, second_start) + 1)


def same_span(first, second):
    """50% reciprocal overlap; for a point-like call (an insertion, whose span is shorter than its length), breakpoints
    closer than the shorter of the two lengths."""
    first_length, second_length = max(int(first["sv_length"]), 1), max(int(second["sv_length"]), 1)
    first_span, second_span = int(first["end"]) - int(first["pos"]) + 1, int(second["end"]) - int(second["pos"]) + 1
    if first_span < first_length or second_span < second_length:
        return abs(int(first["pos"]) - int(second["pos"])) <= min(first_length, second_length)
    shared = overlap(int(first["pos"]), int(first["end"]), int(second["pos"]), int(second["end"]))
    return shared >= RECIPROCAL_OVERLAP * first_span and shared >= RECIPROCAL_OVERLAP * second_span


def squared_correlation_matrix(genotypes):
    centred = genotypes - genotypes.mean(axis=0)
    norms = np.sqrt((centred ** 2).sum(axis=0))
    unit = np.divide(centred, norms, out=np.zeros_like(centred), where=norms > 0)
    return (unit.T @ unit) ** 2


def events(rows, table, genotypes):
    """Union-find over SVs that are one event: r^2 above DUPLICATE_R2, or the same span."""
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    r2 = squared_correlation_matrix(genotypes[:, rows].astype(np.float64))
    for first in range(len(rows)):
        for second in range(first + 1, len(rows)):
            if r2[first, second] > DUPLICATE_R2 or same_span(table.iloc[rows[first]], table.iloc[rows[second]]):
                parent[find(first)] = find(second)
    return np.array([find(index) for index in range(len(rows))])


class Context:
    dataset = None
    superdups = None
    effects = None


def initialize(dataset_dir, superdups_path, effects):
    Context.dataset = harness.Dataset(dataset_dir)
    columns = pd.read_csv(superdups_path, sep="\t", header=None, usecols=[1, 2, 3], names=["chrom", "start", "end"])
    Context.superdups = columns.assign(start=columns["start"] + 1)
    Context.effects = effects


def segmental_duplication_fraction(chrom, start, end):
    blocks = Context.superdups[(Context.superdups["chrom"] == chrom) & (Context.superdups["end"] >= start) & (Context.superdups["start"] <= end)]
    covered = np.zeros(end - start + 1, dtype=bool)
    for block_start, block_end in zip(blocks["start"], blocks["end"]):
        covered[max(block_start, start) - start:min(block_end, end) - start + 1] = True
    return float(covered.mean())


def gene_row(gene_id):
    dataset = Context.dataset
    effects = Context.effects.get_group(gene_id)
    window = harness.load_gene_window(dataset, int(np.flatnonzero(dataset.genes["gene_id"].to_numpy() == gene_id)[0]))
    table = window.table
    rows = np.unique(effects["window_row"].to_numpy())
    position_of = {row: index for index, row in enumerate(rows)}
    contribution_var, contribution_cov = {}, {}
    leaders = []
    per_split = []
    for split_name, split_effects in effects.groupby("split"):
        test_index = np.array([dataset.sample_index[sample] for sample in dataset.splits[split_name]["test"]])
        columns = split_effects["window_row"].to_numpy()
        contributions = (window.genotypes[np.ix_(test_index, columns)].astype(np.float64) - split_effects["train_mean"].to_numpy()) * split_effects["effect"].to_numpy()
        full = np.zeros((len(test_index), len(rows)))
        full[:, [position_of[row] for row in columns]] = contributions
        full -= full.mean(axis=0)
        per_split.append(full)
        sv_part = full.sum(axis=1)
        leaders.append(rows[int(np.argmax(full.T @ sv_part))])
    stacked = np.vstack(per_split)
    sv_part = stacked.sum(axis=1)
    sv_variance = float(sv_part @ sv_part)
    record = {"gene_id": gene_id, "chrom": window.chrom, "svs": len(rows), "sv_part_variance": sv_variance / len(sv_part)}
    if sv_variance == 0:
        return record | {"events": 0, "events_for_90pct": 0, "effective_events": 0.0}
    labels = events(rows, table, window.genotypes)
    event_ids = np.unique(labels)
    event_contributions = np.column_stack([stacked[:, labels == label].sum(axis=1) for label in event_ids])
    shares = event_contributions.T @ sv_part / sv_variance
    order = np.argsort(-shares)
    needed = int(np.searchsorted(np.cumsum(shares[order]), VARIANCE_SHARE) + 1)
    variances = (event_contributions ** 2).sum(axis=0)
    lead_event = event_ids[order[0]]
    members = np.flatnonzero(labels == lead_event)
    member_shares = stacked[:, members].T @ sv_part / sv_variance
    lead = rows[members[int(np.argmax(member_shares))]]
    sv = table.iloc[lead]
    start, end = int(sv["pos"]), max(int(sv["end"]), int(sv["pos"]))
    small = np.flatnonzero((table["source"] == "panel").to_numpy() & ~table["is_sv"].to_numpy(dtype=bool))
    dosage = window.genotypes[:, lead].astype(np.float64)
    centred_small = window.genotypes[:, small].astype(np.float64)
    centred_small -= centred_small.mean(axis=0)
    centred_dosage = dosage - dosage.mean()
    norms = np.sqrt((centred_small ** 2).sum(axis=0) * (centred_dosage @ centred_dosage))
    max_small_r2 = float(np.max(np.divide((centred_small.T @ centred_dosage) ** 2, norms ** 2, out=np.zeros(len(small)), where=norms > 0))) if len(small) else 0.0
    annotation = dataset.gene_annotation[gene_id]
    return record | {
        "events": len(event_ids), "events_for_90pct": needed if shares.sum() >= VARIANCE_SHARE else np.nan,
        "effective_events": float(variances.sum() ** 2 / (variances ** 2).sum()), "lead_event_share": float(shares[order[0]]),
        "lead_event_svs": len(members), "lead_sv_id": sv["id"], "lead_sv_source": sv["source"], "lead_sv_type": sv["sv_type"],
        "lead_sv_length": int(sv["sv_length"]), "lead_sv_distance_to_tss": start - window.tss, "lead_sv_allele_frequency": float(dosage.mean() / 2),
        "lead_sv_overlaps_gene_body": overlap(start, end, annotation["start"], annotation["end"]) > 0,
        "lead_sv_overlaps_exon": any(overlap(start, end, exon_start, exon_end) > 0 for exon_start, exon_end in annotation["exons"]),
        "lead_sv_max_r2_with_small_variant": max_small_r2, "lead_sv_segmental_duplication_fraction": segmental_duplication_fraction(window.chrom, start, end),
        "lead_sv_splits_leading": int(sum(row == lead for row in leaders)), "splits": len(leaders), "lead_sv_start": start, "lead_sv_end": end}


def cross_mappability_partners(path, targets):
    """target unversioned gene id -> {partner unversioned id: symmetric cross-mappability}, from the sorted pair file."""
    import gzip

    partners = {target: {} for target in targets}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            first, second, score = line.rstrip("\n").split("\t")
            first, second = first.split(".")[0], second.split(".")[0]
            if first in partners:
                partners[first][second] = float(score)
            if second in partners:
                partners[second][first] = float(score)
    return partners


def gene_bodies(gtf_path):
    import gzip

    rows = []
    with gzip.open(gtf_path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.split("\t", 9)
            if fields[2] == "gene":
                rows.append((fields[8].split('gene_id "', 1)[1].split('"', 1)[0].split(".")[0], fields[0], int(fields[3]), int(fields[4])))
    return pd.DataFrame(rows, columns=["gene", "chrom", "start", "end"]).drop_duplicates("gene").set_index("gene")


def add_cross_mappability(table, crossmap_path, gtf_path):
    targets = {gene_id.split(".")[0] for gene_id in table["gene_id"]}
    partners = cross_mappability_partners(crossmap_path, targets)
    bodies = gene_bodies(gtf_path)
    counts, strongest = [], []
    for _, row in table.iterrows():
        partner_scores = partners.get(row["gene_id"].split(".")[0], {})
        if pd.isna(row.get("lead_sv_id")) or not partner_scores:
            counts.append(0)
            strongest.append(0.0)
            continue
        start, end = int(row["lead_sv_start"]), int(row["lead_sv_end"])
        inside = [score for partner, score in partner_scores.items() if partner in bodies.index and bodies.loc[partner, "chrom"] == row["chrom"]
                  and bodies.loc[partner, "start"] <= end and bodies.loc[partner, "end"] >= start]
        counts.append(len(inside))
        strongest.append(max(inside, default=0.0))
    return table.assign(lead_sv_crossmappable_partners=counts, lead_sv_max_crossmappability=strongest)


def classify(table):
    significant = table["sv_part_q"] <= FDR_LEVEL
    return np.where(~significant, "none", np.where(table["events_for_90pct"] == 1, "single", "multi"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="the harness results directory holding the arm")
    parser.add_argument("--method", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--feature-set", default="snv_sv")
    parser.add_argument("--tests", required=True, help="robust.py sv_gene_tests.tsv.gz")
    parser.add_argument("--metric", default="r2")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--superdups", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--crossmap", help="Saha & Battle cross-mappability pair file (hg38, GENCODE v26)")
    parser.add_argument("--gtf", help="GENCODE GTF for partner gene bodies (with --crossmap)")
    arguments = parser.parse_args()
    if arguments.metric == "oos_r2" and arguments.design == "loso":
        raise ValueError("raw oos_r2 under loso is dominated by the held-out group's mean offset; use r2 or oos_r2_centred")
    directory = pathlib.Path(arguments.results) / arguments.method / arguments.design
    effects = pd.concat([pd.read_csv(path, sep="\t") for path in sorted(directory.glob("*.sv_coefficients.tsv.gz"))], ignore_index=True)
    effects = effects[(effects["feature_set"] == arguments.feature_set)]
    grouped = effects.groupby("gene_id")
    genes = sorted(grouped.groups)
    with get_context("fork").Pool(arguments.workers, initializer=initialize, initargs=(arguments.dataset, arguments.superdups, grouped)) as pool:
        decomposition = pd.DataFrame(list(pool.imap_unordered(gene_row, genes, chunksize=1)))
    tests = pd.read_csv(arguments.tests, sep="\t")
    tests = tests[(tests["method"] == arguments.method) & (tests["design"] == arguments.design) & (tests["feature_set"] == arguments.feature_set)
                  & (tests["metric"] == arguments.metric)]
    table = tests.merge(decomposition, on=["gene_id", "chrom"], how="left")
    if arguments.crossmap:
        table = add_cross_mappability(table, arguments.crossmap, arguments.gtf)
    table["class"] = classify(table)
    table.sort_values("gain", ascending=False).to_csv(arguments.out, sep="\t", index=False)
    print(table["class"].value_counts().to_string())


if __name__ == "__main__":
    main()
