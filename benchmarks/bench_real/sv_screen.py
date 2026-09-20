"""A genotype-only screen for bench-real: which genes' cis windows hold SV genotype variance no small variant tags.

It reads no expression data at all (no MAGE expression, no MAGE eQTL results, no benchmark scores): only the
dataset's variant tables and dosages of the 730 people, and the GENCODE v38 gene structure in gene_annotation.json.

Per SV (panel rows with is_sv, and PanGenie SVs): the allele frequency, the genotype variance v in the 730, the
maximum r² with any panel small variant (SNV/indel under 50 bp, the harness's ``snv`` set) whose interval overlaps
[pos - R, end + R], and the untagged variance u = v (1 - max r²), the part of the SV's genotype variance that its best
single small-variant proxy leaves unexplained. R is the harness's cis radius.

Per gene: over the SVs in the harness's own cis window (the SV interval overlaps TSS +/- R), the sums of u for panel
and for PanGenie SVs, the largest single u, SV counts, whether an SV overlaps the gene body or its merged exons, and
the leading SVs. Genes are ranked by the panel sum, ties broken by the sealed gene order.

Outputs: screen_<version>.tsv (every column, one row per gene in rank order), sv_ranked_<version>.tsv (a `gene_id`
column only, in rank order, for the harness's gene-list option), sv_proxies_<version>.tsv.gz (per SV), and SEALED.txt
with the sha256 of all three.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib

import numpy as np
import pandas as pd

from benchmarks.bench_real.harness import CIS_RADIUS_BP

SV_COLUMNS = ["chrom", "row", "id", "source", "pos", "end", "sv_type", "sv_length", "allele_frequency", "genotype_variance",
              "max_r2", "proxy_id", "proxies", "untagged_variance"]
LEADING_SVS = 3  # how many SVs each gene row lists; display only
FITTED_PREFIXES = (2000, 5000)  # the --gene-prefix values of bench-real's v2 runs (mr.ash; lead variant and GBLUP)


def standardized(block):
    """Rows centred and scaled to unit variance over the samples; constant rows become zero."""
    values = np.asarray(block, dtype=np.float64)
    centered = values - values.mean(axis=1, keepdims=True)
    scale = np.sqrt((centered ** 2).mean(axis=1))
    varying = scale > 0
    centered[varying] /= scale[varying, None]
    centered[~varying] = 0.0
    return centered


class SmallVariantWindow:
    """Standardized small-variant rows over a sliding index range, each row read from disk once as the range advances."""

    def __init__(self, dosage, rows):
        self.dosage, self.rows = dosage, rows
        self.start = self.stop = 0
        self.values = np.empty((0, dosage.shape[1]))

    def _read(self, start, stop):
        rows = self.rows[start:stop]
        if len(rows) == 0:
            return np.empty((0, self.dosage.shape[1]))
        first = int(rows.min())
        span = np.asarray(self.dosage[first:int(rows.max()) + 1])
        return standardized(span[rows - first])

    def window(self, start, stop):
        if start >= self.stop or stop <= self.start or start < self.start:
            self.values, self.start, self.stop = self._read(start, stop), start, stop
        else:
            self.values, self.start = self.values[start - self.start:], start
            if stop > self.stop:
                self.values = np.concatenate([self.values, self._read(self.stop, stop)])
                self.stop = stop
        return self.values[:stop - self.start]


def sv_proxies(chrom, table, dosage, radius=CIS_RADIUS_BP):
    """Per SV of one chromosome: frequency, genotype variance, best small-variant proxy and untagged variance."""
    sample_count = dosage.shape[1]
    positions, ends = table["pos"].to_numpy(), table["end"].to_numpy()
    small = np.flatnonzero((table["source"].to_numpy() == "panel") & ~table["is_sv"].to_numpy(dtype=bool))
    small = small[np.argsort(positions[small], kind="stable")]
    small_positions, small_ends = positions[small], ends[small]
    slack = int((small_ends - small_positions).max()) if len(small) else 0
    structural = np.flatnonzero(table["is_sv"].to_numpy(dtype=bool))
    structural = structural[np.argsort(positions[structural], kind="stable")]
    genotypes = np.asarray(dosage[structural], dtype=np.float64) if len(structural) else np.empty((0, sample_count))
    frequency = genotypes.mean(axis=1) / 2
    variance = genotypes.var(axis=1)
    scaled = standardized(genotypes)
    best = np.zeros(len(structural))
    proxy = np.full(len(structural), -1)
    proxies = np.zeros(len(structural), dtype=np.int64)
    window = SmallVariantWindow(dosage, small)
    batch_start = 0
    while batch_start < len(structural):
        batch_stop = int(np.searchsorted(positions[structural], positions[structural[batch_start]] + radius, side="left"))
        batch_stop = max(batch_stop, batch_start + 1)
        batch = np.arange(batch_start, batch_stop)
        lows = positions[structural[batch]] - radius
        highs = ends[structural[batch]] + radius
        first = np.searchsorted(small_positions, lows - slack, side="left")
        last = np.searchsorted(small_positions, highs, side="right")
        block_start, block_stop = int(first.min()), int(last.max())
        block = window.window(block_start, block_stop)
        correlation = scaled[batch] @ block.T / sample_count
        for offset, index in enumerate(batch):
            begin, finish = int(first[offset]), int(last[offset])
            covered = np.flatnonzero(small_ends[begin:finish] >= lows[offset])
            proxies[index] = len(covered)
            if len(covered) == 0 or variance[index] == 0:
                continue
            squared = correlation[offset, begin - block_start + covered] ** 2
            top = int(np.argmax(squared))
            if squared[top] > 0:
                best[index], proxy[index] = squared[top], small[begin + covered[top]]
        batch_start = batch_stop
    identifiers = table["id"].to_numpy(dtype=str)
    frame = pd.DataFrame({"chrom": chrom, "row": structural, "id": identifiers[structural],
                          "source": table["source"].to_numpy(dtype=str)[structural], "pos": positions[structural],
                          "end": ends[structural], "sv_type": table["sv_type"].to_numpy(dtype=str)[structural],
                          "sv_length": table["sv_length"].to_numpy()[structural], "allele_frequency": frequency,
                          "genotype_variance": variance, "max_r2": np.minimum(best, 1.0),
                          "proxy_id": np.where(proxy >= 0, identifiers[np.maximum(proxy, 0)], ""), "proxies": proxies})
    frame["untagged_variance"] = frame["genotype_variance"] * (1.0 - frame["max_r2"])
    return frame[SV_COLUMNS]


def window_members(sv_positions, sv_ends, tss, radius=CIS_RADIUS_BP):
    """The harness's cis-window rule: the variant interval overlaps TSS +/- radius."""
    return np.flatnonzero((sv_ends >= tss - radius) & (sv_positions <= tss + radius))


def _describe(frame):
    return ";".join(f"{row.sv_type}:{row.sv_length}:{row.untagged_variance:.4g}:{row.max_r2:.3f}:{row.allele_frequency:.3f}:{row.id}"
                    for row in frame.itertuples())


def gene_scores(genes, annotation, structural, radius=CIS_RADIUS_BP):
    """Per gene: untagged SV variance in its cis window, SV counts, and body and exon overlap."""
    records = []
    for chrom, chromosome_genes in genes.groupby("chrom", sort=False):
        on_chromosome = structural[structural["chrom"] == chrom].reset_index(drop=True)
        positions, ends = on_chromosome["pos"].to_numpy(), on_chromosome["end"].to_numpy()
        sources = on_chromosome["source"].to_numpy(dtype=str)
        untagged_all = on_chromosome["untagged_variance"].to_numpy()
        polymorphic = on_chromosome["genotype_variance"].to_numpy() > 0
        for gene in chromosome_genes.itertuples():
            members = window_members(positions, ends, int(gene.tss), radius)
            members = members[polymorphic[members]]
            structure = annotation[gene.gene_id]
            body = (ends[members] >= structure["start"]) & (positions[members] <= structure["end"])
            exons = np.array(structure["exons"], dtype=np.int64).reshape(-1, 2)
            in_exon = ((ends[members, None] >= exons[None, :, 0]) & (positions[members, None] <= exons[None, :, 1])).any(axis=1)
            record = {"gene_id": gene.gene_id, "chrom": chrom, "tss": int(gene.tss)}
            for source, name in (("panel", "panel"), ("pangenie", "pgsv")):
                mine = sources[members] == source
                untagged = untagged_all[members[mine]]
                record.update({f"U_{name}": float(untagged.sum()), f"max_u_{name}": float(untagged.max()) if len(untagged) else 0.0,
                               f"n_sv_{name}": int(mine.sum()), f"U_{name}_body": float(untagged[body[mine]].sum()),
                               f"body_overlap_{name}": bool(body[mine].any()), f"exon_overlap_{name}": bool(in_exon[mine].any()),
                               f"leading_{name}": _describe(on_chromosome.iloc[members[mine]].nlargest(LEADING_SVS, "untagged_variance"))})
            records.append(record)
    return pd.DataFrame(records)


def rank_genes(scores, gene_order):
    """Order by the panel sum of untagged variance, ties broken by the sealed gene order; mark the fitted prefixes."""
    order = {gene: index for index, gene in enumerate(gene_order)}
    ranked = scores.assign(gene_order_index=scores["gene_id"].map(order))
    ranked = ranked.sort_values(["U_panel", "gene_order_index"], ascending=[False, True], kind="stable").reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def _chromosome_proxies(arguments):
    dataset_dir, chrom, out_dir = arguments
    target = out_dir / f"{chrom}.sv_proxies.tsv.gz"
    if not target.exists():
        number = chrom.removeprefix("chr")
        table = pd.read_csv(dataset_dir / f"chr{number}.variants.tsv", sep="\t")
        if "source" not in table:
            table["source"] = "panel"
        dosage = np.load(dataset_dir / f"chr{number}.dosage.npy", mmap_mode="r")
        partial = target.with_name(f".{target.name}.partial")
        sv_proxies(chrom, table, dosage).to_csv(partial, sep="\t", index=False)
        partial.replace(target)
    return chrom


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--version", default="v1")
    arguments = parser.parse_args()
    dataset_dir, out_dir = pathlib.Path(arguments.dataset), pathlib.Path(arguments.out)
    proxies_dir = out_dir / f"sv_proxies_{arguments.version}"
    proxies_dir.mkdir(parents=True, exist_ok=True)
    genes = pd.read_csv(dataset_dir / "genes.tsv", sep="\t")
    chromosomes = sorted(genes["chrom"].unique(), key=lambda chrom: -(dataset_dir / f"chr{chrom.removeprefix('chr')}.dosage.npy").stat().st_size)
    with concurrent.futures.ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for chrom in pool.map(_chromosome_proxies, [(dataset_dir, chrom, proxies_dir) for chrom in chromosomes]):
            print("proxies done", chrom, flush=True)
    structural = pd.concat([pd.read_csv(proxies_dir / f"{chrom}.sv_proxies.tsv.gz", sep="\t", keep_default_na=False)
                            for chrom in chromosomes], ignore_index=True)
    annotation = json.loads((dataset_dir / "gene_annotation.json").read_text())
    gene_order = pd.read_csv(dataset_dir / "gene_order.tsv", sep="\t")["gene_id"].tolist()
    ranked = rank_genes(gene_scores(genes, annotation, structural), gene_order)
    for prefix in FITTED_PREFIXES:
        ranked[f"in_prefix_{prefix}"] = ranked["gene_order_index"] < prefix
    screen = out_dir / f"screen_{arguments.version}.tsv"
    ranked_list = out_dir / f"sv_ranked_{arguments.version}.tsv"
    table = out_dir / f"sv_proxies_{arguments.version}.tsv.gz"
    ranked.to_csv(screen, sep="\t", index=False)
    ranked[["gene_id"]].to_csv(ranked_list, sep="\t", index=False)
    structural.to_csv(table, sep="\t", index=False)
    (out_dir / "SEALED.txt").write_text("".join(f"{_sha256(path)}  {path.name}\n" for path in (screen, ranked_list, table)))
    print((out_dir / "SEALED.txt").read_text(), flush=True)


if __name__ == "__main__":
    main()
