"""bench-sim step 1: a real-haplotype mosaic cohort from the public 1kGP phased SNV/INDEL/SV panel.

PREREG.md section 1. Writes, under <out>/<chrom>/:
  variants.npz     per kept record: pos, end, ref_len, alt_len, cm, cls (0 SNV, 1 INDEL, 2 TR, 3 SV),
                   svtype, ids, donor allele counts, superpopulation AFs of the donors
  donor_haps.u8    [n_var, 2 * n_donor] donor haplotypes (0/1), variant-major (kept for re-imputation)
  truth_G.npy      [n_var, N] cohort genotypes 0/1/2, variant-major
  truth_hapA.npy   [n_var, N] each member's first haplotype (0/1); truth_G - truth_hapA is the second
  samples.npz      cohort group, realized ancestry proportions, sex, age, batch, split
and <out>/founders.tsv with the donor / panel split.
"""

from __future__ import annotations

import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from cyvcf2 import VCF

PUBLIC_SEED = 20260919
SUPERPOPS = ("AFR", "AMR", "EAS", "EUR", "SAS")
# name: (1kGP superpopulation whose founder share sets the group's weight, mean ancestry over SUPERPOPS,
# admixture generations or 0). The weights are derived at build time from the public 1kGP founder composition
# (group_weights). The admixed groups' mean ancestry and admixture time follow Bryc et al. 2015 (AJHG) and
# Baharian et al. 2016 (PLoS Genet), coarsened.
GROUPS = {
    "EUR": ("EUR", (0.0, 0.0, 0.0, 1.0, 0.0), 0),
    "AFR_admixed": ("AFR", (0.80, 0.02, 0.0, 0.18, 0.0), 7),
    "AMR_admixed": ("AMR", (0.08, 0.50, 0.0, 0.42, 0.0), 13),
    "EAS": ("EAS", (0.0, 0.0, 1.0, 0.0, 0.0), 0),
    "SAS": ("SAS", (0.0, 0.0, 0.0, 0.0, 1.0), 0),
}
DIRICHLET_CONCENTRATION = 20.0
EFFECTIVE_POPULATION_SIZE = 20_000.0
DONOR_FRACTION = 0.6
MINIMUM_DONOR_MAC = 3
SV_MIN_LENGTH = 50


def read_founders(ped_path: Path) -> list[tuple[str, str]]:
    founders = []
    with open(ped_path) as handle:
        header = handle.readline().split()
        column = {name: index for index, name in enumerate(header)}
        for line in handle:
            fields = line.split()
            if fields[column["FatherID"]] == "0" and fields[column["MotherID"]] == "0":
                founders.append((fields[column["SampleID"]], fields[column["Superpopulation"]]))
    return founders


def split_founders(founders: list[tuple[str, str]]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    rng = np.random.default_rng(PUBLIC_SEED)
    donors: dict[str, list[str]] = {}
    panel: dict[str, list[str]] = {}
    for superpop in SUPERPOPS:
        members = sorted(sample for sample, pop in founders if pop == superpop)
        order = rng.permutation(len(members))
        cut = int(round(DONOR_FRACTION * len(members)))
        donors[superpop] = [members[index] for index in order[:cut]]
        panel[superpop] = [members[index] for index in order[cut:]]
    return donors, panel


def load_repeats(path: Path, chrom: str) -> tuple[np.ndarray, np.ndarray]:
    starts, ends = [], []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            fields = line.split("\t")
            if fields[1] == chrom:
                starts.append(int(fields[2]))
                ends.append(int(fields[3]))
    order = np.argsort(starts)
    return np.asarray(starts, dtype=np.int64)[order], np.asarray(ends, dtype=np.int64)[order]


def overlaps_interval(starts: np.ndarray, ends: np.ndarray, query_start: np.ndarray, query_end: np.ndarray) -> np.ndarray:
    """True where [query_start, query_end) overlaps any interval; intervals may nest, so use a running max end."""
    running_end = np.maximum.accumulate(ends)
    index = np.searchsorted(starts, query_end, side="left") - 1
    valid = index >= 0
    result = np.zeros(query_start.shape[0], dtype=bool)
    result[valid] = running_end[index[valid]] > query_start[valid]
    return result


def read_region(args: tuple[str, str, int, int, list[str]]) -> dict[str, np.ndarray]:
    vcf_path, chrom, start, end, samples = args
    vcf = VCF(vcf_path, samples=samples, threads=2)
    pos, stop, ref_len, alt_len, len_change, cm, symbolic, svtype, ids, haps = [], [], [], [], [], [], [], [], [], []
    refs, alts = [], []
    for record in vcf(f"{chrom}:{start}-{end}"):
        if record.POS < start or record.POS > end or len(record.ALT) != 1:
            continue
        alt = record.ALT[0]
        genotype = record.genotype.array()[:, :2]
        haps.append(genotype.astype(np.uint8).reshape(-1))
        pos.append(record.POS)
        is_symbolic = alt.startswith("<")
        symbolic.append(is_symbolic)
        info_end = record.INFO.get("END")
        svlen = record.INFO.get("SVLEN")
        stop.append(int(info_end) if info_end is not None else record.POS + len(record.REF) - 1)
        ref_len.append(len(record.REF))
        alt_len.append(abs(int(svlen)) if (is_symbolic and svlen is not None) else len(alt))
        if is_symbolic:
            signed = int(svlen) if svlen is not None else int(stop[-1] - record.POS)
            if str(record.INFO.get("SVTYPE")) == "DEL" and signed > 0:
                signed = -signed
        else:
            signed = len(alt) - len(record.REF)
        len_change.append(signed)
        cm.append(float(record.INFO.get("CM")))
        svtype.append(str(record.INFO.get("SVTYPE") or ""))
        ids.append(record.ID or f"{chrom}:{record.POS}:{record.REF[:10]}:{alt[:10]}")
        refs.append(record.REF)
        alts.append(alt)
    vcf.close()
    return {
        "pos": np.asarray(pos, dtype=np.int64),
        "end": np.asarray(stop, dtype=np.int64),
        "ref_len": np.asarray(ref_len, dtype=np.int64),
        "alt_len": np.asarray(alt_len, dtype=np.int64),
        "len_change": np.asarray(len_change, dtype=np.int64),
        "cm": np.asarray(cm, dtype=np.float64),
        "symbolic": np.asarray(symbolic, dtype=bool),
        "svtype": np.asarray(svtype, dtype=object),
        "ids": np.asarray(ids, dtype=object),
        "refs": np.asarray(refs, dtype=object),
        "alts": np.asarray(alts, dtype=object),
        "haps": np.vstack(haps) if haps else np.zeros((0, 2 * len(samples)), dtype=np.uint8),
    }


def group_weights(founders: list[tuple[str, str]]) -> np.ndarray:
    """Each group's weight: its superpopulation's share of the 1kGP founders (PREREG amendment 7)."""
    counts = np.array([sum(1 for _, superpop in founders if superpop == GROUPS[name][0]) for name in GROUPS], dtype=np.float64)
    return counts / counts.sum()


def draw_cohort(size: int, rng: np.random.Generator, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = list(GROUPS)
    group = rng.choice(len(names), size=size, p=weights)
    proportions = np.zeros((size, len(SUPERPOPS)))
    generations = np.zeros(size)
    for index, name in enumerate(names):
        _, mean, admixture_generations = GROUPS[name]
        members = np.flatnonzero(group == index)
        mean_array = np.asarray(mean)
        if admixture_generations:
            support = mean_array > 0
            draws = rng.dirichlet(DIRICHLET_CONCENTRATION * mean_array[support], size=members.size)
            proportions[np.ix_(members, np.flatnonzero(support))] = draws
        else:
            proportions[members] = mean_array
        generations[members] = admixture_generations
    return group, proportions, generations


def haplotype_path(
    cm: np.ndarray,
    proportions: np.ndarray,
    generations: float,
    donor_columns: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Breakpoints (variant indices), donor haplotype column and ancestry per segment, for one haplotype."""
    morgans_total = (cm[-1] - cm[0]) / 100.0
    ancestry_switch_points: list[float] = []
    if generations:
        count = rng.poisson(generations * morgans_total)
        ancestry_switch_points = sorted(rng.uniform(0.0, morgans_total, size=count).tolist())
    tract_edges = [0.0, *ancestry_switch_points, morgans_total]
    starts_m, donors, ancestries = [], [], []
    for left, right in zip(tract_edges[:-1], tract_edges[1:]):
        ancestry = int(rng.choice(len(SUPERPOPS), p=proportions))
        columns = donor_columns[ancestry]
        rate = 4.0 * EFFECTIVE_POPULATION_SIZE / columns.size
        count = rng.poisson(rate * (right - left))
        cuts = np.sort(rng.uniform(left, right, size=count))
        edges = np.concatenate([[left], cuts])
        for edge in edges:
            starts_m.append(edge)
            donors.append(int(columns[rng.integers(columns.size)]))
            ancestries.append(ancestry)
    start_index = np.searchsorted(cm - cm[0], 100.0 * np.asarray(starts_m), side="left")
    start_index[0] = 0
    return start_index.astype(np.int64), np.asarray(donors, dtype=np.int64), np.asarray(ancestries, dtype=np.int8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", required=True)
    parser.add_argument("--ped", required=True)
    parser.add_argument("--repeats", required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sample-batch", type=int, default=1024)
    args = parser.parse_args()

    out_root = Path(args.out)
    out = out_root / args.chrom
    out.mkdir(parents=True, exist_ok=True)
    founders = read_founders(Path(args.ped))
    donors, panel = split_founders(founders)
    with open(out_root / "founders.tsv", "w") as handle:
        for role, table in (("donor", donors), ("panel", panel)):
            for superpop in SUPERPOPS:
                for sample in table[superpop]:
                    handle.write(f"{sample}\t{superpop}\t{role}\n")
    donor_samples = [sample for superpop in SUPERPOPS for sample in donors[superpop]]
    panel_samples = [sample for superpop in SUPERPOPS for sample in panel[superpop]]
    superpop_of = {sample: pop for sample, pop in founders}
    vcf_samples = VCF(args.vcf).samples
    wanted = set(donor_samples) | set(panel_samples)
    missing = sorted(wanted - set(vcf_samples))
    if missing:
        raise SystemExit(f"{len(missing)} founders absent from the VCF, e.g. {missing[:3]}")
    # cyvcf2 returns the selected samples in VCF order.
    vcf_order = [sample for sample in vcf_samples if sample in wanted]
    donor_set = set(donor_samples)
    is_donor_sample = np.array([sample in donor_set for sample in vcf_order])
    superpop_index = {pop: index for index, pop in enumerate(SUPERPOPS)}
    sample_superpop = np.array([superpop_index[superpop_of[sample]] for sample in vcf_order])
    donor_superpop_vcf = sample_superpop[is_donor_sample]
    panel_order = [sample for sample, donor in zip(vcf_order, is_donor_sample) if not donor]
    with open(out / "panel_samples.txt", "w") as handle:
        handle.write("\n".join(panel_order) + "\n")

    hap_superpop = np.repeat(donor_superpop_vcf, 2)
    if not (out / "donor_haps.npy").exists():
        # Parallel region reads over equal-length chunks of the chromosome.
        vcf = VCF(args.vcf)
        length = dict(zip(vcf.seqnames, vcf.seqlens))[args.chrom]
        vcf.close()
        edges = np.linspace(1, length + 1, args.workers * 4 + 1).astype(np.int64)
        tasks = [(args.vcf, args.chrom, int(left), int(right) - 1, vcf_order) for left, right in zip(edges[:-1], edges[1:])]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            parts = list(pool.map(read_region, tasks))
        fields = {key: np.concatenate([part[key] for part in parts]) for key in parts[0] if key != "haps"}
        all_haps = np.vstack([part["haps"] for part in parts])
        hap_is_donor = np.repeat(is_donor_sample, 2)
        haps = np.ascontiguousarray(all_haps[:, hap_is_donor])
        panel_haps = np.ascontiguousarray(all_haps[:, ~hap_is_donor])
        del all_haps
        donor_ac = haps.sum(axis=1, dtype=np.int64)
        donor_an = haps.shape[1]
        keep = np.minimum(donor_ac, donor_an - donor_ac) >= MINIMUM_DONOR_MAC
        fields = {key: value[keep] for key, value in fields.items()}
        haps = haps[keep]
        panel_haps = panel_haps[keep]
        donor_ac = donor_ac[keep]
        print(f"{args.chrom}: {keep.size} biallelic records, {keep.sum()} kept at donor MAC >= {MINIMUM_DONOR_MAC}", flush=True)

        rep_start, rep_end = load_repeats(Path(args.repeats), args.chrom)
        is_sv = fields["symbolic"] | (np.abs(fields["len_change"]) >= SV_MIN_LENGTH)
        is_indel = ~is_sv & (fields["ref_len"] != fields["alt_len"])
        span_start = fields["pos"] - 1
        span_end = np.maximum(fields["end"], fields["pos"] + fields["ref_len"] - 1)
        in_repeat = overlaps_interval(rep_start, rep_end, span_start, span_end)
        cls = np.zeros(keep.sum(), dtype=np.int8)
        cls[is_indel] = 1
        cls[is_indel & in_repeat] = 2
        cls[is_sv] = 3
        superpop_af = np.zeros((len(SUPERPOPS), haps.shape[0]), dtype=np.float32)
        hap_superpop = np.repeat(donor_superpop_vcf, 2)
        for index in range(len(SUPERPOPS)):
            superpop_af[index] = haps[:, hap_superpop == index].mean(axis=1)
        np.savez(
            out / "variants.npz",
            pos=fields["pos"], end=fields["end"], ref_len=fields["ref_len"], alt_len=fields["alt_len"],
            len_change=fields["len_change"],
            cm=fields["cm"], cls=cls, svtype=fields["svtype"].astype(str), ids=fields["ids"].astype(str),
            refs=fields["refs"].astype(str), alts=fields["alts"].astype(str),
            donor_ac=donor_ac, donor_an=np.int64(donor_an), superpop_af=superpop_af, in_repeat=in_repeat,
        )
        np.save(out / "donor_haps.npy", haps)
        np.save(out / "panel_haps.npy", panel_haps)
        del panel_haps
        counts = {name: int(value) for name, value in zip(("SNV", "INDEL", "TR", "SV"), np.bincount(cls, minlength=4))}
        print(json.dumps(counts), flush=True)

    variants = np.load(out / "variants.npz")
    haps = np.load(out / "donor_haps.npy")
    fields = {"cm": variants["cm"]}

    # Cohort draws (public seed): groups, proportions, covariates, split.
    rng = np.random.default_rng(PUBLIC_SEED + int(args.chrom.lstrip("chr")))
    cohort_rng = np.random.default_rng(PUBLIC_SEED)
    weights = group_weights(founders)
    group, proportions, generations = draw_cohort(args.size, cohort_rng, weights)
    sex = cohort_rng.integers(0, 2, size=args.size).astype(np.int8)
    age = cohort_rng.uniform(18.0, 80.0, size=args.size)
    batch = cohort_rng.integers(0, 2, size=args.size).astype(np.int8)
    is_test = np.zeros(args.size, dtype=bool)
    for index in range(len(GROUPS)):
        members = np.flatnonzero(group == index)
        chosen = cohort_rng.choice(members, size=int(round(0.2 * members.size)), replace=False)
        is_test[chosen] = True

    cm = fields["cm"]
    donor_columns = {index: np.flatnonzero(hap_superpop == index) for index in range(len(SUPERPOPS))}
    n_var = haps.shape[0]
    # Built in RAM (n_var x N bytes) and written once: column-block writes into a network-FS memmap are
    # random I/O and far slower than the mosaic itself.
    truth = np.zeros((n_var, args.size), dtype=np.uint8)
    first_haplotype = np.zeros((n_var, args.size), dtype=np.uint8)
    realized = np.zeros((args.size, len(SUPERPOPS)), dtype=np.float32)
    donors_t = np.ascontiguousarray(haps.T)
    del haps
    morgan_span = cm - cm[0]
    for first in range(0, args.size, args.sample_batch):
        last = min(first + args.sample_batch, args.size)
        rows = np.zeros((last - first, n_var), dtype=np.uint8)
        first_rows = np.zeros((last - first, n_var), dtype=np.uint8)
        for offset, person in enumerate(range(first, last)):
            for haplotype_index in range(2):
                starts, donor_cols, ancestry = haplotype_path(cm, proportions[person], generations[person], donor_columns, rng)
                stops = np.append(starts[1:], n_var)
                for start, stop, column in zip(starts.tolist(), stops.tolist(), donor_cols.tolist()):
                    if stop > start:
                        rows[offset, start:stop] += donors_t[column, start:stop]
                seg_len = np.diff(np.append(morgan_span[starts], morgan_span[-1]))
                np.add.at(realized[person], ancestry, seg_len / 2.0)
                if haplotype_index == 0:
                    first_rows[offset] = rows[offset]
        truth[:, first:last] = rows.T
        first_haplotype[:, first:last] = first_rows.T
        print(f"cohort samples {last}/{args.size}", flush=True)
    np.save(out / "truth_G.npy", truth)
    np.save(out / "truth_hapA.npy", first_haplotype)
    del truth, first_haplotype
    realized /= realized.sum(axis=1, keepdims=True)
    np.savez(
        out / "samples.npz", group=group, group_names=np.asarray(list(GROUPS)), group_weights=weights, proportions=proportions,
        realized_proportions=realized, sex=sex, age=age, batch=batch, is_test=is_test,
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
