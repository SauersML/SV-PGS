"""bench-sim out-of-family truths (PREREG amendment 9): architectures outside SV-PGS's own model class.

critic-method (critique/CRITIQUE_METHOD.md, item 3) showed that the in-family truths of PREREG section 3 mostly share
SV-PGS's prior: Gaussian scale-mixture effect shapes, the [2p(1-p)]^(1+alpha) frequency law, LD and annotation smooths,
and class scales. Each family here breaks one of those assumptions. Section 3 is reused only for heritability,
covariates, noise and trait type.

- fixed_count: exactly K causal records, one per-SD magnitude, random signs.
- nonscale_heavy: Pareto magnitudes above a floor with random signs. Heavy-tailed, with density zero near 0, so it
  is not a Gaussian scale mixture (those have a density that decreases in |beta|).
- hidden_annotation: causal records only inside UCSC CpG islands and their 2 kb shores, an annotation methods never
  see.
- clustered_loci: whole loci (equal genetic-length segments) carry sign-concordant, equal per-allele effects.
- epistasis: products of centred genotype pairs carry a drawn share of the genetic variance.
- panel_absent: most causal records are ones the imputation panel lacks (outside the measured set), so no method
  can measure them.
- sv_gene_dosage: an SV's effect is its gene's effect times the SV's copy-number change over the gene's exons, with
  no class scale. A deletion loses its exonic fraction, a whole-gene duplication gains one copy, and any other SV that
  touches an exon disrupts it (minus one). A point-mass background covers the other classes.

    python -m benchmarks.bench_sim.truth_out --dir <cohort/chr22> --out <v7> --refseq <..> --cpg <..> --set dev|test
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.bench_sim import truth
from benchmarks.bench_sim.annotations import overlaps

FAMILIES = ("fixed_count", "nonscale_heavy", "hidden_annotation", "clustered_loci", "epistasis", "panel_absent", "sv_gene_dosage")
TR, SV = 2, 3
# Irizarry et al. 2009 (Nat Genet): CpG island shores are the 2 kb flanking an island.
SHORE_BASES = 2000
# Interaction partners are common records (cohort MAF >= 5%), so the interaction variance is not carried by a few
# carriers: a design choice of amendment 9.
EPISTASIS_MINIMUM_MAF = 0.05
READ_ROWS = 5000
DEV_SEED_BASE = 1000
IN_FAMILY_KEYS = (
    "pi", "sv_mode", "tr_mode", "sv_scale_fold", "tr_scale_fold", "sv_probability_fold", "tr_probability_fold", "shape",
    "shape_params", "sv_own_shape", "sv_shape", "sv_shape_params", "alpha_snv", "alpha_sv", "plateau",
    "frequency_source", "ld_gamma", "annotation_mode", "annotation_seed", "dominance", "tr_saturating", "link_c",
)


def log_uniform_integer(rng: np.random.Generator, low: int, high: int) -> int:
    return int(round(truth.log_uniform(rng, low, high)))


def draw_out_parameters(seed: int, family: str) -> dict:
    params = truth.draw_parameters(seed)
    for key in IN_FAMILY_KEYS:
        params.pop(key)
    params["family"] = family
    rng = np.random.default_rng([seed, 9])
    if family == "fixed_count":
        params["causal_count"] = log_uniform_integer(rng, 10, 300)
    elif family == "nonscale_heavy":
        params["pi"] = truth.log_uniform(rng, 1e-4, 3e-2)
        params["tail_index"] = float(rng.uniform(1.2, 2.5))
    elif family == "hidden_annotation":
        params["causal_count"] = log_uniform_integer(rng, 20, 500)
    elif family == "clustered_loci":
        params["locus_cm"] = truth.log_uniform(rng, 0.05, 0.5)
        params["locus_count"] = log_uniform_integer(rng, 5, 100)
        params["locus_causal_fraction"] = float(rng.uniform(0.01, 0.2))
    elif family == "epistasis":
        params["causal_count"] = log_uniform_integer(rng, 10, 300)
        params["pair_count"] = log_uniform_integer(rng, 5, 100)
        params["interaction_share"] = float(rng.uniform(0.3, 1.0))
    elif family == "panel_absent":
        params["causal_count"] = log_uniform_integer(rng, 20, 500)
        params["absent_share"] = float(rng.uniform(0.5, 0.9))
    elif family == "sv_gene_dosage":
        params["causal_count"] = log_uniform_integer(rng, 10, 300)
        params["gene_count"] = log_uniform_integer(rng, 3, 60)
        params["sv_share"] = float(rng.uniform(0.2, 0.8))
    else:
        raise ValueError(family)
    return params


def pareto_magnitudes(rng: np.random.Generator, count: int, tail_index: float) -> np.ndarray:
    """Pareto(tail_index) magnitudes with floor 1: P(|beta| > x) = x^-tail_index for x >= 1."""
    return (1.0 - rng.random(count)) ** (-1.0 / tail_index)


def record_spans(variants) -> tuple[np.ndarray, np.ndarray]:
    pos = variants["pos"]
    return pos - 1, np.maximum(variants["end"], pos + variants["ref_len"] - 1)


def cpg_regions(path: Path, chrom: str) -> tuple[np.ndarray, np.ndarray]:
    starts, ends = [], []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            fields = line.split("\t")
            if fields[1] == chrom:
                starts.append(int(fields[2]) - SHORE_BASES)
                ends.append(int(fields[3]) + SHORE_BASES)
    return np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)


def gene_models(path: Path, chrom: str) -> list[dict]:
    """RefSeq genes (by symbol) on chrom: span and the union of their transcripts' exons."""
    genes: dict[str, dict] = {}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if fields[2] != chrom:
                continue
            entry = genes.setdefault(fields[12], {"name": fields[12], "start": int(fields[4]), "end": int(fields[5]), "exons": []})
            entry["start"] = min(entry["start"], int(fields[4]))
            entry["end"] = max(entry["end"], int(fields[5]))
            starts = [int(value) for value in fields[9].rstrip(",").split(",")]
            ends = [int(value) for value in fields[10].rstrip(",").split(",")]
            entry["exons"].extend(zip(starts, ends))
    models = []
    for entry in genes.values():
        merged: list[list[int]] = []
        for start, end in sorted(entry["exons"]):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        entry["exons"] = merged
        entry["exonic_bases"] = sum(end - start for start, end in merged)
        models.append(entry)
    return sorted(models, key=lambda model: model["start"])


def copy_change(kind: str, span_start: int, span_end: int, gene: dict) -> float:
    """The SV's change to the gene's functional copies (amendment 9's rule)."""
    exonic = sum(max(0, min(span_end, end) - max(span_start, start)) for start, end in gene["exons"])
    if kind == "DEL":
        return -exonic / gene["exonic_bases"]
    if kind == "DUP":
        if span_start <= gene["start"] and span_end >= gene["end"]:
            return 1.0
        return -exonic / gene["exonic_bases"]
    return -1.0 if exonic > 0 else 0.0


def sv_gene_changes(variants, genes: list[dict]) -> dict[tuple[int, int], float]:
    """(record, gene index) -> copy change, for every SV record with a nonzero change."""
    cls = variants["cls"]
    span_start, span_end = record_spans(variants)
    gene_start = np.array([gene["start"] for gene in genes])
    gene_end = np.array([gene["end"] for gene in genes])
    changes = {}
    for row in np.flatnonzero(cls == SV):
        kind = str(variants["svtype"][row]) or ("DEL" if variants["len_change"][row] < 0 else "INS")
        for gene_index in np.flatnonzero((gene_start < span_end[row]) & (gene_end > span_start[row])):
            change = copy_change(kind, int(span_start[row]), int(span_end[row]), genes[gene_index])
            if change:
                changes[int(row), int(gene_index)] = change
    return changes


def additive(genotypes, rows: np.ndarray, weights: np.ndarray, per_sd: bool, structural: np.ndarray):
    """Sum over rows of weight x (centred genotype), weights per SD or per allele; also the structural rows' part."""
    order = np.argsort(rows)
    rows, weights = rows[order], weights[order]
    size = genotypes.shape[1]
    value, structural_value = np.zeros(size), np.zeros(size)
    per_allele = np.zeros(rows.size)
    for first in range(0, rows.size, READ_ROWS):
        chunk = rows[first:first + READ_ROWS]
        block = np.asarray(genotypes[chunk], dtype=np.float64)
        centred = block - block.mean(axis=1, keepdims=True)
        sd = block.std(axis=1)
        weight = weights[first:first + READ_ROWS]
        scale = np.divide(weight, sd, out=np.zeros_like(weight), where=sd > 0) if per_sd else np.where(sd > 0, weight, 0.0)
        per_allele[first:first + READ_ROWS] = scale
        value += scale @ centred
        members = structural[chunk]
        structural_value += scale[members] @ centred[members]
    return rows, per_allele, value, structural_value


def interactions(genotypes, pairs: np.ndarray, weights: np.ndarray, structural: np.ndarray):
    """Sum over pairs of weight x standardized(g_i) x standardized(g_j); a pair's term is split between its partners'
    classes for the structural share."""
    size = genotypes.shape[1]
    value, structural_value = np.zeros(size), np.zeros(size)
    rows = np.unique(pairs)
    block = np.asarray(genotypes[rows], dtype=np.float64)
    sd = block.std(axis=1, keepdims=True)
    standardized = np.divide(block - block.mean(axis=1, keepdims=True), sd, out=np.zeros_like(block), where=sd > 0)
    position = {int(row): index for index, row in enumerate(rows)}
    for (first, second), weight in zip(pairs, weights):
        term = weight * standardized[position[int(first)]] * standardized[position[int(second)]]
        value += term
        structural_value += term * (0.5 * structural[first] + 0.5 * structural[second])
    return value, structural_value


def unit_scale(value: np.ndarray) -> float:
    spread = value.std()
    return 1.0 / spread if spread > 0 else 0.0


def genetic_value(params: dict, root: Path, refseq: Path, cpg: Path, chrom: str) -> tuple[np.ndarray, dict]:
    variants = np.load(root / "variants.npz")
    cls = variants["cls"]
    structural = cls >= TR
    genotypes = np.load(root / "truth_G.npy", mmap_mode="r")
    n_var = cls.size
    rng = np.random.default_rng(params["effect_seed"])
    family = params["family"]
    components = []  # (weight on the unit-variance scale, rows, per-allele, value, structural value)
    interaction = None
    pair_count = 0

    def signs(count: int) -> np.ndarray:
        return rng.choice(np.array([-1.0, 1.0]), size=count)

    def point_mass(candidates: np.ndarray, count: int):
        rows = rng.choice(candidates, size=min(count, candidates.size), replace=False)
        return additive(genotypes, rows, signs(rows.size), True, structural)

    if family == "fixed_count":
        components.append((1.0, *point_mass(np.arange(n_var), params["causal_count"])))
    elif family == "nonscale_heavy":
        rows = np.flatnonzero(rng.random(n_var) < params["pi"])
        weights = signs(rows.size) * pareto_magnitudes(rng, rows.size, params["tail_index"])
        components.append((1.0, *additive(genotypes, rows, weights, True, structural)))
    elif family == "hidden_annotation":
        island_start, island_end = cpg_regions(cpg, chrom)
        span_start, span_end = record_spans(variants)
        candidates = np.flatnonzero(overlaps(island_start, island_end, span_start, span_end))
        rows = rng.choice(candidates, size=min(params["causal_count"], candidates.size), replace=False)
        components.append((1.0, *additive(genotypes, rows, rng.standard_normal(rows.size), True, structural)))
    elif family == "clustered_loci":
        cm = variants["cm"]
        locus = np.floor((cm - cm.min()) / params["locus_cm"]).astype(np.int64)
        chosen = rng.choice(np.unique(locus), size=min(params["locus_count"], np.unique(locus).size), replace=False)
        rows, weights = [], []
        for locus_id in chosen:
            members = np.flatnonzero(locus == locus_id)
            causal = members[rng.random(members.size) < params["locus_causal_fraction"]]
            if causal.size == 0:
                causal = members[rng.integers(members.size, size=1)]
            rows.append(causal)
            weights.append(np.full(causal.size, rng.standard_normal()))
        components.append((1.0, *additive(genotypes, np.concatenate(rows), np.concatenate(weights), False, structural)))
    elif family == "epistasis":
        annotations = np.load(root / "annotations.npz")
        frequency = annotations["ld_subsample_af"]
        common = np.flatnonzero(np.minimum(frequency, 1.0 - frequency) >= EPISTASIS_MINIMUM_MAF)
        partners = rng.choice(common, size=2 * min(params["pair_count"], common.size // 2), replace=False).reshape(-1, 2)
        interaction = interactions(genotypes, partners, signs(partners.shape[0]), structural)
        pair_count = int(partners.shape[0])
        components.append((np.sqrt(1.0 - params["interaction_share"]), *point_mass(np.arange(n_var), params["causal_count"])))
    elif family == "panel_absent":
        measured = np.load(root / "measured.npy")
        absent_count = int(round(params["absent_share"] * params["causal_count"]))
        absent = rng.choice(np.flatnonzero(~measured), size=min(absent_count, int((~measured).sum())), replace=False)
        present = rng.choice(np.flatnonzero(measured), size=params["causal_count"] - absent.size, replace=False)
        rows = np.concatenate([absent, present])
        components.append((1.0, *additive(genotypes, rows, signs(rows.size), True, structural)))
    elif family == "sv_gene_dosage":
        genes = gene_models(refseq, chrom)
        changes = sv_gene_changes(variants, genes)
        affected = sorted({gene_index for _, gene_index in changes})
        chosen = rng.choice(affected, size=min(params["gene_count"], len(affected)), replace=False)
        gene_effect = dict(zip(chosen.tolist(), rng.standard_normal(chosen.size)))
        per_record: dict[int, float] = {}
        for (row, gene_index), change in changes.items():
            if gene_index in gene_effect:
                per_record[row] = per_record.get(row, 0.0) + gene_effect[gene_index] * change
        sv_rows = np.array(sorted(per_record), dtype=np.int64)
        sv_weights = np.array([per_record[row] for row in sv_rows])
        components.append((np.sqrt(params["sv_share"]), *additive(genotypes, sv_rows, sv_weights, False, structural)))
        components.append((np.sqrt(1.0 - params["sv_share"]), *point_mass(np.flatnonzero(cls != SV), params["causal_count"])))
    else:
        raise ValueError(family)

    total = np.zeros(genotypes.shape[1])
    structural_total = np.zeros(genotypes.shape[1])
    scaled = []
    for weight, rows, per_allele, value, structural_value in components:
        factor = weight * unit_scale(value)
        total += factor * value
        structural_total += factor * structural_value
        scaled.append((rows, factor * per_allele))
    if interaction is not None:
        factor = np.sqrt(params["interaction_share"]) * unit_scale(interaction[0])
        total += factor * interaction[0]
        structural_total += factor * interaction[1]
    final = np.sqrt(params["h2"]) * unit_scale(total)
    rows = np.concatenate([rows for rows, _ in scaled])
    per_allele = np.concatenate([effects for _, effects in scaled]) * final
    order = np.argsort(rows)
    rows, per_allele = rows[order], per_allele[order]
    summary = {
        "family": family,
        "causal_count": int(np.unique(rows).size),
        "causal_by_class": np.bincount(cls[np.unique(rows)], minlength=4).tolist(),
        "structural_share": float(np.cov(structural_total, total)[0, 1] / np.var(total, ddof=1)),
    }
    if interaction is not None:
        summary["interaction_pairs"] = pair_count
    return final * total, {"summary": summary, "effects": {"causal": rows, "per_allele": per_allele}}


def build(seed: int, family: str, root: Path, out: Path, refseq: Path, cpg: Path, chrom: str) -> dict:
    params = draw_out_parameters(seed, family)
    value, record = genetic_value(params, root, refseq, cpg, chrom)
    phenotype = truth.phenotype(params, value, root)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "truth.npz", genetic_value=value, phenotype=phenotype, **record["effects"])
    (out / "scenario.json").write_text(json.dumps({"params": params, "summary": record["summary"]}, indent=1, sort_keys=True))
    return record


def sealed_out_seeds(master_hex: str, count: int) -> list[int]:
    """Out-of-family test seeds: a hash domain of their own, disjoint from truth.sealed_seeds."""
    return [int.from_bytes(hashlib.sha256(f"{master_hex}:out:{index}".encode()).digest()[:4], "little") for index in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="cohort chromosome directory")
    parser.add_argument("--out", required=True, help="bench-sim root; writes dev_out/ or sealed_out/")
    parser.add_argument("--refseq", required=True)
    parser.add_argument("--cpg", required=True)
    parser.add_argument("--chrom", default="chr22")
    parser.add_argument("--set", choices=("dev", "test"), required=True)
    parser.add_argument("--per-family", type=int, required=True)
    args = parser.parse_args()
    root, out = Path(args.dir), Path(args.out)
    count = args.per_family * len(FAMILIES)
    if args.set == "dev":
        seeds, target = [DEV_SEED_BASE + index for index in range(count)], out / "dev_out"
    else:
        seeds, target = sealed_out_seeds((out / "sealed" / "master_seed.txt").read_text().strip(), count), out / "sealed_out"
    for index, seed in enumerate(seeds):
        family = FAMILIES[index % len(FAMILIES)]
        record = build(seed, family, root, target / f"scenario_{index:03d}", Path(args.refseq), Path(args.cpg), args.chrom)
        print(index, family, json.dumps(record["summary"]) if args.set == "dev" else "built", flush=True)


if __name__ == "__main__":
    main()
