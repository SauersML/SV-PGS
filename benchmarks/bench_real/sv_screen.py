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

Outputs: screen_<version>.tsv (every column, one row per gene in rank order, with a `confirm` flag),
sv_ranked_<version>.tsv (a `gene_id` column only, in rank order, without the confirmation genes, for the harness's
gene-list option), confirm_genes_<version>.tsv (the sealed confirmation genes, used exactly once for the final
confirmation; also written to the dataset's sealed_confirmation_genes.tsv, which the harness enforces and which is
never replaced), sv_proxies_<version>.tsv.gz (per SV), and SEALED.txt with the sha256 of each and of the
already-scored gene list the confirmation set was drawn against.
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

SV_COLUMNS = ["chrom", "row", "source", "pos", "end", "sv_type", "sv_length", "allele_frequency", "genotype_variance",
              "max_r2", "proxy_row", "proxy_pos", "proxies", "untagged_variance"]
TABLE_COLUMNS = {"pos": np.int64, "end": np.int64, "is_sv": bool, "source": "category", "sv_type": "category", "sv_length": np.int64}
LEADING_SVS = 3  # how many SVs each gene row lists; display only
FITTED_PREFIXES = (2000, 5000)  # the --gene-prefix values of bench-real's v2 runs (mr.ash; lead variant and GBLUP)


def standardized(block):
    """Rows centred and scaled to unit variance over the samples, as float64; constant rows become zero."""
    values = np.array(block, dtype=np.float64)
    values -= values.mean(axis=1, keepdims=True)
    scale = np.sqrt((values ** 2).mean(axis=1))
    varying = scale > 0
    values[varying] /= scale[varying, None]
    values[~varying] = 0.0
    return values


def read_standardized(dosage, rows):
    """Standardized dosage rows; ``rows`` ascending, read as one contiguous span."""
    if len(rows) == 0:
        return np.empty((0, dosage.shape[1]))
    first = int(rows[0])
    return standardized(np.asarray(dosage[first:int(rows[-1]) + 1])[rows - first])


def resident_bytes():
    """This process's anonymous resident memory (Linux /proc/self/status RssAnon, in kB). File-backed pages of the
    memory-mapped dosages are reclaimable and not counted, as the kernel's OOM accounting does not charge them."""
    for line in pathlib.Path("/proc/self/status").read_text().splitlines():
        if line.startswith("RssAnon:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/self/status has no RssAnon")


def worker_memory_bytes(workers):
    """One worker's memory: the task's usable memory split evenly over the workers.

    The usable memory is what is free now: MemAvailable capped by the memory cgroups' headroom and by the runner's
    per-task allotment. Only on a bare host, with neither a cgroup cap nor an allotment, is it shared with every other
    user, and then the task takes its cores-proportional share (the runners' default rule).
    """
    from sv_pgs.compute_budget import RUNQ_MEMORY_VARIABLE, _cgroup_memory_headroom_bytes, _usable_host_bytes
    usable = _usable_host_bytes()
    if RUNQ_MEMORY_VARIABLE not in os.environ and _cgroup_memory_headroom_bytes() is None:
        usable = usable * workers // os.cpu_count()
    return usable // workers


def _batches(positions, ends, radius):
    """SVs (sorted by position) in groups whose proxy windows are read together: consecutive SVs starting within
    ``radius`` of the group's first, each SV longer than ``radius`` in a group of its own."""
    long = (ends - positions) > radius
    start = 0
    while start < len(positions):
        if long[start]:
            yield np.array([start])
            start += 1
            continue
        stop = start + 1
        while stop < len(positions) and not long[stop] and positions[stop] < positions[start] + radius:
            stop += 1
        yield np.arange(start, stop)
        start = stop


def sv_proxies(chrom, table, dosage, radius=CIS_RADIUS_BP, chunk_rows=None, worker_bytes=None):
    """Per SV of one chromosome: frequency, genotype variance, best small-variant proxy and untagged variance.

    Small variants are streamed in chunks; without ``chunk_rows``, each chunk is sized so that its live arrays fit in
    ``worker_bytes`` less this process's resident set.
    """
    if (chunk_rows is None) == (worker_bytes is None):
        raise ValueError("give exactly one of chunk_rows and worker_bytes")
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
    del genotypes
    best = np.zeros(len(structural))
    proxy = np.full(len(structural), -1)
    proxies = np.zeros(len(structural), dtype=np.int64)
    # Live bytes per streamed small-variant row: the int8 span and its selection, the float64 standardized row, and
    # one float64 correlation per SV of the group.
    row_bytes = lambda group: sample_count * (2 * np.dtype(np.int8).itemsize + np.dtype(np.float64).itemsize) + group * np.dtype(np.float64).itemsize
    for batch in _batches(positions[structural], ends[structural], radius):
        lows = positions[structural[batch]] - radius
        highs = ends[structural[batch]] + radius
        first = np.searchsorted(small_positions, lows - slack, side="left")
        last = np.searchsorted(small_positions, highs, side="right")
        step = chunk_rows
        if step is None:
            headroom = worker_bytes - resident_bytes()
            if headroom < row_bytes(len(batch)):
                raise MemoryError(f"{chrom}: {headroom} bytes of headroom cannot hold one small-variant row")
            step = headroom // row_bytes(len(batch))
        for chunk_start in range(int(first.min()), int(last.max()), step):
            chunk_stop = min(chunk_start + step, int(last.max()))
            correlation = scaled[batch] @ read_standardized(dosage, small[chunk_start:chunk_stop]).T / sample_count
            for offset, index in enumerate(batch):
                begin, finish = max(int(first[offset]), chunk_start), min(int(last[offset]), chunk_stop)
                if begin >= finish:
                    continue
                covered = np.flatnonzero(small_ends[begin:finish] >= lows[offset])
                proxies[index] += len(covered)
                if len(covered) == 0 or variance[index] == 0:
                    continue
                squared = correlation[offset, begin - chunk_start + covered] ** 2
                top = int(np.argmax(squared))
                if squared[top] > best[index]:
                    best[index], proxy[index] = squared[top], small[begin + covered[top]]
    frame = pd.DataFrame({"chrom": chrom, "row": structural,
                          "source": np.asarray(table["source"].to_numpy(), dtype=object)[structural], "pos": positions[structural],
                          "end": ends[structural], "sv_type": np.asarray(table["sv_type"].to_numpy(), dtype=object)[structural],
                          "sv_length": table["sv_length"].to_numpy()[structural], "allele_frequency": frequency,
                          "genotype_variance": variance, "max_r2": np.minimum(best, 1.0),
                          "proxy_row": proxy, "proxy_pos": np.where(proxy >= 0, positions[np.maximum(proxy, 0)], -1), "proxies": proxies})
    frame["untagged_variance"] = frame["genotype_variance"] * (1.0 - frame["max_r2"])
    return frame[SV_COLUMNS]


def window_members(sv_positions, sv_ends, tss, radius=CIS_RADIUS_BP):
    """The harness's cis-window rule: the variant interval overlaps TSS +/- radius."""
    return np.flatnonzero((sv_ends >= tss - radius) & (sv_positions <= tss + radius))


def _describe(frame):
    return ";".join(f"{row.sv_type}:{row.sv_length}:{row.untagged_variance:.4g}:{row.max_r2:.3f}:{row.allele_frequency:.3f}:{row.chrom}:{row.pos}-{row.end}"
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


CONFIRM_SALT = "bench-real/confirm/"
CONFIRM_MODULUS = 4  # lead ruling: a quarter of the genes bench-real has never scored are sealed for confirmation
SEALED_GENES = "sealed_confirmation_genes.tsv"  # the file bench-real's harness enforces (harness.SEALED_GENES)


def confirmation_genes(gene_ids, scored_genes):
    """The sealed confirmation genes: never scored by any bench-real run, and int(sha256(salt + gene_id), 16) % modulus == 0."""
    development = set(scored_genes)
    return np.array([gene not in development and int(hashlib.sha256((CONFIRM_SALT + gene).encode()).hexdigest(), 16) % CONFIRM_MODULUS == 0
                     for gene in gene_ids])


def rank_genes(scores, gene_order):
    """Order by the panel sum of untagged variance, ties broken by the sealed gene order; mark the fitted prefixes."""
    order = {gene: index for index, gene in enumerate(gene_order)}
    ranked = scores.assign(gene_order_index=scores["gene_id"].map(order))
    ranked = ranked.sort_values(["U_panel", "gene_order_index"], ascending=[False, True], kind="stable").reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def _chromosome_proxies(arguments):
    dataset_dir, chrom, out_dir, worker_bytes = arguments
    target = out_dir / f"{chrom}.sv_proxies.tsv.gz"
    if not target.exists():
        number = chrom.removeprefix("chr")
        table = pd.read_csv(dataset_dir / f"chr{number}.variants.tsv", sep="\t", usecols=list(TABLE_COLUMNS), dtype=TABLE_COLUMNS)
        if "source" not in table:
            table["source"] = "panel"
        dosage = np.load(dataset_dir / f"chr{number}.dosage.npy", mmap_mode="r")
        partial = target.with_name(f".{target.name}.partial")
        sv_proxies(chrom, table, dosage, worker_bytes=worker_bytes).to_csv(partial, sep="\t", index=False, compression="gzip")
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
    parser.add_argument("--scored-genes", required=True, help="bench-real's reports/genes_already_scored.tsv (a gene_id column)")
    arguments = parser.parse_args()
    dataset_dir, out_dir = pathlib.Path(arguments.dataset), pathlib.Path(arguments.out)
    proxies_dir = out_dir / f"sv_proxies_{arguments.version}"
    proxies_dir.mkdir(parents=True, exist_ok=True)
    genes = pd.read_csv(dataset_dir / "genes.tsv", sep="\t")
    chromosomes = sorted(genes["chrom"].unique(), key=lambda chrom: -(dataset_dir / f"chr{chrom.removeprefix('chr')}.dosage.npy").stat().st_size)
    worker_bytes = worker_memory_bytes(arguments.workers)
    print("worker memory bytes", worker_bytes, flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for chrom in pool.map(_chromosome_proxies, [(dataset_dir, chrom, proxies_dir, worker_bytes) for chrom in chromosomes]):
            print("proxies done", chrom, flush=True)
    structural = pd.concat([pd.read_csv(proxies_dir / f"{chrom}.sv_proxies.tsv.gz", sep="\t", keep_default_na=False)
                            for chrom in chromosomes], ignore_index=True)
    annotation = json.loads((dataset_dir / "gene_annotation.json").read_text())
    gene_order = pd.read_csv(dataset_dir / "gene_order.tsv", sep="\t")["gene_id"].tolist()
    ranked = rank_genes(gene_scores(genes, annotation, structural), gene_order)
    for prefix in FITTED_PREFIXES:
        ranked[f"in_prefix_{prefix}"] = ranked["gene_order_index"] < prefix
    scored = pd.read_csv(arguments.scored_genes, sep="\t")["gene_id"]
    ranked["already_scored"] = ranked["gene_id"].isin(set(scored))
    ranked["confirm"] = confirmation_genes(ranked["gene_id"], scored)
    screen = out_dir / f"screen_{arguments.version}.tsv"
    ranked_list = out_dir / f"sv_ranked_{arguments.version}.tsv"
    table = out_dir / f"sv_proxies_{arguments.version}.tsv.gz"
    ranked.to_csv(screen, sep="\t", index=False)
    confirm_list = out_dir / f"confirm_genes_{arguments.version}.tsv"
    ranked.loc[~ranked["confirm"], ["gene_id"]].to_csv(ranked_list, sep="\t", index=False)
    ranked.loc[ranked["confirm"], ["gene_id"]].sort_values("gene_id").to_csv(confirm_list, sep="\t", index=False)
    sealed = dataset_dir / SEALED_GENES
    if sealed.exists() and sealed.read_bytes() != confirm_list.read_bytes():
        raise RuntimeError(f"{sealed} already holds a different confirmation set; it is never replaced")
    if not sealed.exists():
        partial = sealed.with_name(f".{sealed.name}.partial")
        partial.write_bytes(confirm_list.read_bytes())
        partial.replace(sealed)
    structural.to_csv(table, sep="\t", index=False)
    (out_dir / "SEALED.txt").write_text("".join(f"{_sha256(path)}  {path.name}\n" for path in (screen, ranked_list, confirm_list, table))
                                        + f"{_sha256(sealed)}  {sealed} (the harness copy of {confirm_list.name})\n"
                                        + f"{_sha256(pathlib.Path(arguments.scored_genes))}  {arguments.scored_genes} (input: genes already scored)\n")
    print((out_dir / "SEALED.txt").read_text(), flush=True)


if __name__ == "__main__":
    main()
