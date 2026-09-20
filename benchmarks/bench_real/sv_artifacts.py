"""Artifact checks for the genes whose SV columns carry held-out signal (critique/CRITIQUE_REAL.md items 6-7,
critique/CRITIQUE_METHOD.md items 4-5).

For each (method, design, gene) the SV part of the score is d = f - m: the snv_sv prediction minus the same prediction
with the SV columns at their training means. For a linear predictor d = sum_j beta_j (x_j - mean_j), and the driving SV
is the panel SV in the cis window whose dosage correlates most with d over all people. Checks on the driving SV:
  covariates   its correlation with every MAGE eQTL covariate (sex, 5 genotype PCs, 60 PEER factors), and the R2 of its
               dosage on all of them. A high R2 means residualizing expression on the covariates also removes SV signal,
               using test samples too, since MAGE fit PEER on all of them.
  expression   corr(expression, SV) before and after the full-sample covariate adjustment (a diagnostic, not a score).
  carriers     minor-allele carriers per held-out superpopulation and per random5 fold.
  influence    test-side leave-one-carrier-out: the gene's pooled r2 gain (f vs the snv prediction) recomputed with each
               minor-allele carrier removed from its held-out group; predictions are fixed, so this is not a refit.
  germline     in the 3,202-sample 1kGP phased panel: Mendelian errors in every complete trio, Hardy-Weinberg exact
               tests (Wigginton, Cutler & Abecasis 2005, AJHG 76:887) among pedigree founders within each
               superpopulation, and whether each MAGE minor-allele carrier's genotyped parents and children carry it.
               The panel was phased with pedigree information, so Mendelian consistency there is necessary but weak
               evidence; a culture-acquired CNV would still show as a carrier none of whose parents carry it.
  mappability  the target gene's Saha & Battle 2018 (F1000Research 7:1860) GENCODE v26 gene mappability, and every
               gene it cross-maps with (symmetric cross-mappability > 0) that the driving SV overlaps. Such a partner's
               copy number moves the reads that align to the target gene, which is an RNA-seq alignment artifact route.
"""
import argparse
import gzip
import json
import pathlib
import subprocess

import numpy as np
import pandas as pd

from benchmarks.bench_real import robust

CIS_RADIUS_BP = 1_000_000  # harness.CIS_RADIUS_BP; restated so this module does not import the harness's worker state


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    first, second = first - first.mean(), second - second.mean()
    denominator = np.sqrt((first @ first) * (second @ second))
    return float(first @ second / denominator) if denominator > 0 else 0.0


def residual(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    full = np.column_stack([np.ones(len(values)), design])
    coefficients, *_ = np.linalg.lstsq(full, values, rcond=None)
    return values - full @ coefficients


def hwe_exact(heterozygotes: int, homozygous_first: int, homozygous_second: int) -> float:
    """Two-sided exact Hardy-Weinberg p value (Wigginton, Cutler & Abecasis 2005)."""
    rare_homozygotes, common_homozygotes = sorted((homozygous_first, homozygous_second))
    individuals = heterozygotes + rare_homozygotes + common_homozygotes
    rare_copies = 2 * rare_homozygotes + heterozygotes
    if individuals == 0 or rare_copies == 0:
        return 1.0
    probabilities = np.zeros(rare_copies + 1)
    middle = int(rare_copies * (2 * individuals - rare_copies) / (2 * individuals))
    if (rare_copies - middle) % 2:
        middle += 1
    probabilities[middle] = 1.0
    rare_hom, common_hom = (rare_copies - middle) // 2, individuals - middle - (rare_copies - middle) // 2
    for het in range(middle, 1, -2):
        probabilities[het - 2] = probabilities[het] * het * (het - 1) / (4.0 * (rare_hom + 1) * (common_hom + 1))
        rare_hom, common_hom = rare_hom + 1, common_hom + 1
    rare_hom, common_hom = (rare_copies - middle) // 2, individuals - middle - (rare_copies - middle) // 2
    for het in range(middle, rare_copies - 1, 2):
        probabilities[het + 2] = probabilities[het] * 4.0 * rare_hom * common_hom / ((het + 2) * (het + 1))
        rare_hom, common_hom = rare_hom - 1, common_hom - 1
    probabilities /= probabilities.sum()
    observed = probabilities[heterozygotes]
    return float(min(1.0, probabilities[probabilities <= observed].sum()))


def mendelian_consistent(child: int, father: int, mother: int) -> bool:
    transmitted = {0: {0}, 1: {0, 1}, 2: {1}}
    return child in {a + b for a in transmitted[father] for b in transmitted[mother]}


def minor_carriers(dosage: np.ndarray) -> np.ndarray:
    """Carriers of the minor allele: non-reference if the alternate allele is the minor one, else non-homozygous-alternate."""
    return dosage > 0 if dosage.mean() <= 1.0 else dosage < 2


def panel_genotypes(panel_dir: pathlib.Path, chrom: str, position: int, variant_id: str):
    """(samples, dosages) of one panel record, or None if the record isn't found."""
    path = next(panel_dir.glob(f"*.{chrom}.filtered.*phased_panel.vcf.gz"))
    samples = subprocess.run(["bcftools", "query", "-l", str(path)], check=True, capture_output=True, text=True).stdout.split()
    lines = subprocess.run(["bcftools", "query", "-r", f"{chrom}:{position}-{position}", "-f", "%ID\t[%GT\t]\n", str(path)],
                           check=True, capture_output=True, text=True).stdout.splitlines()
    for line in lines:
        fields = line.rstrip("\t").split("\t")
        if fields[0] == variant_id:
            dosages = np.array([sum(allele == "1" for allele in call.replace("|", "/").split("/")) for call in fields[1:]])
            return samples, dosages
    return None


def germline_checks(samples, dosages, pedigree: pd.DataFrame, mage_carriers):
    genotype = dict(zip(samples, dosages))
    trios = pedigree[(pedigree["FatherID"] != "0") & (pedigree["MotherID"] != "0")]
    trios = trios[trios["SampleID"].isin(genotype) & trios["FatherID"].isin(genotype) & trios["MotherID"].isin(genotype)]
    errors = sum(not mendelian_consistent(genotype[c], genotype[f], genotype[m]) for c, f, m in zip(trios["SampleID"], trios["FatherID"], trios["MotherID"]))
    founders = pedigree[(pedigree["FatherID"] == "0") & (pedigree["MotherID"] == "0") & pedigree["SampleID"].isin(genotype)]
    hwe = {}
    for group, members in founders.groupby("Superpopulation"):
        values = np.array([genotype[sample] for sample in members["SampleID"]])
        hwe[group] = hwe_exact(int((values == 1).sum()), int((values == 0).sum()), int((values == 2).sum()))
    minor_alt = np.mean(dosages) <= 1.0
    carries = (lambda value: value > 0) if minor_alt else (lambda value: value < 2)
    parents = pedigree.set_index("SampleID")[["FatherID", "MotherID"]]
    with_parent, parent_carries, with_child, child_carries = 0, 0, 0, 0
    for carrier in mage_carriers:
        if carrier in parents.index:
            genotyped = [parent for parent in parents.loc[carrier] if parent != "0" and parent in genotype]
            if genotyped:
                with_parent += 1
                parent_carries += any(carries(genotype[parent]) for parent in genotyped)
        children = pedigree.loc[(pedigree["FatherID"] == carrier) | (pedigree["MotherID"] == carrier), "SampleID"]
        children = [child for child in children if child in genotype]
        if children:
            with_child += len(children)
            child_carries += sum(carries(genotype[child]) for child in children)
    return {"trios": int(len(trios)), "mendelian_errors": int(errors), "panel_allele_frequency": float(np.mean(dosages) / 2),
            **{f"hwe_p_{group}": value for group, value in hwe.items()},
            "carriers_with_a_genotyped_parent": with_parent, "of_which_a_parent_carries": int(parent_carries),
            "children_of_carriers": with_child, "of_which_carry": int(child_carries)}


def cross_mappable_partners(path: pathlib.Path, targets: set):
    """target unversioned gene id -> {partner unversioned id: symmetric cross-mappability}, streamed from the sorted file."""
    partners = {target: {} for target in targets}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            first, second, score = line.split("\t")
            first, second = first.split(".")[0], second.split(".")[0]
            if first in partners:
                partners[first][second] = float(score)
            if second in partners:
                partners[second][first] = float(score)
    return partners


def gencode_genes(path: pathlib.Path):
    rows = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.split("\t", 9)
            if fields[2] != "gene":
                continue
            attributes = fields[8]
            gene_id = attributes.split('gene_id "', 1)[1].split('"', 1)[0].split(".")[0]
            name = attributes.split('gene_name "', 1)[1].split('"', 1)[0] if 'gene_name "' in attributes else ""
            rows.append((gene_id, name, fields[0], int(fields[3]), int(fields[4])))
    return pd.DataFrame(rows, columns=["gene", "name", "chrom", "start", "end"]).drop_duplicates("gene").set_index("gene")


def run(results_dirs, dataset_dir, tests_path, methods, designs, panel_dir, pedigree_path, crossmap_dir, gencode_path, out_dir):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset_dir = pathlib.Path(dataset_dir)
    samples = pd.read_csv(dataset_dir / "samples.tsv", sep="\t")
    genes_table = pd.read_csv(dataset_dir / "genes.tsv", sep="\t")
    row_of_gene = {gene: row for row, gene in enumerate(genes_table["gene_id"])}
    expression = np.load(dataset_dir / "expression.npy", mmap_mode="r")
    annotation = json.loads((dataset_dir / "gene_annotation.json").read_text())
    covariates = np.load(dataset_dir / "covariates.npy")
    covariate_names = pd.read_csv(dataset_dir / "covariate_names.tsv", sep="\t")["covariate"].tolist()
    splits = json.loads((dataset_dir / "splits.json").read_text())
    fold = {sample: split["name"] for split in splits if split["name"].startswith("random5/") for sample in split["test"]}
    groups = samples["Superpopulation"].to_numpy()
    tests = pd.read_csv(tests_path, sep="\t")
    selected = tests[(tests["metric"] == "r2") & (tests["feature_set"] == "snv_sv") & (tests["sv_part_q"] <= 1 - robust.LEVEL)]
    selected = selected[selected["method"].isin(methods) & selected["design"].isin(designs)]
    arms = robust.load_arms(results_dirs, methods, designs)
    pedigree = pd.read_csv(pedigree_path, sep=r"\s+", dtype=str)
    mappability = pd.read_csv(pathlib.Path(crossmap_dir) / "hg38_gene_mappability.txt.gz", sep="\t", header=None, names=["gene", "mappability"])
    mappability["gene"] = mappability["gene"].str.split(".").str[0]
    mappability = mappability.drop_duplicates("gene").set_index("gene")["mappability"]
    targets = {gene.split(".")[0] for gene in selected["gene_id"]}
    partners = cross_mappable_partners(pathlib.Path(crossmap_dir) / "hg38_cross_mappability_strength_symmetric_mean_sorted.txt.gz", targets)
    gencode = gencode_genes(pathlib.Path(gencode_path))
    tables, panel_cache, rows, carrier_rows, influence_rows = {}, {}, [], [], []
    for (method, design, gene_id, chrom), _ in selected.groupby(["method", "design", "gene_id", "chrom"]):
        full, masked, snv = (arms[(method, design, name)] for name in ("snv_sv", "snv_sv" + robust.MASKED, "snv"))
        position = int(np.flatnonzero(full.genes["gene_id"].to_numpy() == gene_id)[0])
        snv_position = int(np.flatnonzero(snv.genes["gene_id"].to_numpy() == gene_id)[0])
        sv_part = full.prediction[position] - masked.prediction[position]
        if chrom not in tables:
            tables[chrom] = (pd.read_csv(dataset_dir / f"{chrom}.variants.tsv", sep="\t"), np.load(dataset_dir / f"{chrom}.dosage.npy", mmap_mode="r"))
        table, dosage = tables[chrom]
        tss = int(genes_table.loc[row_of_gene[gene_id], "tss"])
        window = np.flatnonzero((table["end"].to_numpy() >= tss - CIS_RADIUS_BP) & (table["pos"].to_numpy() <= tss + CIS_RADIUS_BP)
                                & table["is_sv"].to_numpy(dtype=bool) & (table.get("source", pd.Series("panel", index=table.index)).to_numpy() == "panel"))
        if len(window) == 0 or not np.any(sv_part != sv_part[0]):
            continue
        genotypes = np.asarray(dosage[window], dtype=np.float64)
        scores = np.array([abs(correlation(row, sv_part)) for row in genotypes])
        driver = int(np.argmax(scores))
        sv = genotypes[driver]
        record = table.iloc[window[driver]]
        carriers = minor_carriers(sv)
        body = annotation[gene_id]
        exons = np.array(body["exons"], dtype=np.int64).reshape(-1, 2)
        overlaps_body = int(record["end"]) >= int(body["start"]) and int(record["pos"]) <= int(body["end"])
        overlaps_exon = bool(np.any((exons[:, 1] >= int(record["pos"])) & (exons[:, 0] <= int(record["end"])))) if len(exons) else False
        row = {"method": method, "design": design, "gene_id": gene_id, "chrom": chrom, "sv_id": record["id"], "sv_type": record["sv_type"],
               "sv_start": int(record["pos"]), "sv_end": int(record["end"]), "sv_length": record["sv_length"], "allele_frequency": float(sv.mean() / 2),
               "abs_corr_with_sv_part": float(scores[driver]), "minor_allele_carriers": int(carriers.sum()),
               "sv_overlaps_gene_body": overlaps_body, "sv_overlaps_exon": overlaps_exon}
        covariate_corr = np.array([correlation(covariates[:, index], sv) for index in range(covariates.shape[1])])
        strongest = int(np.argmax(np.abs(covariate_corr)))
        fitted = sv - residual(sv, covariates)
        row.update({"sv_r2_on_covariates": float(np.var(fitted) / np.var(sv)), "strongest_covariate": covariate_names[strongest],
                    "strongest_covariate_corr": float(covariate_corr[strongest]),
                    "max_abs_corr_with_peer": float(np.abs([value for name, value in zip(covariate_names, covariate_corr) if name.startswith("PEER")]).max())})
        values = np.asarray(expression[row_of_gene[gene_id]], dtype=np.float64)
        row.update({"expression_sv_corr_raw": correlation(values, sv), "expression_sv_corr_adjusted": correlation(residual(values, covariates), sv),
                    "expression_sv_corr_both_adjusted": correlation(residual(values, covariates), residual(sv, covariates))})
        for group in robust.GROUPS:
            carrier_rows.append({"method": method, "design": design, "gene_id": gene_id, "sv_id": record["id"], "unit": group,
                                 "minor_allele_carriers": int(carriers[groups == group].sum()), "people": int((groups == group).sum())})
        for name in sorted(set(fold.values())):
            members = samples["sample"].map(fold).to_numpy() == name
            carrier_rows.append({"method": method, "design": design, "gene_id": gene_id, "sv_id": record["id"], "unit": name,
                                 "minor_allele_carriers": int(carriers[members].sum()), "people": int(members.sum())})
        truth = full.truth[position]

        def pooled_gain(drop=None):
            gains = []
            for group in robust.GROUPS:
                index = np.flatnonzero((groups == group) & (np.arange(len(groups)) != (-1 if drop is None else drop)))
                weights = np.ones((1, len(index)))
                with_sv = robust.group_metrics(weights, full.prediction[position:position + 1, index], truth[None, index])["r2"][0, 0]
                without = robust.group_metrics(weights, snv.prediction[snv_position:snv_position + 1, index], snv.truth[snv_position:snv_position + 1, index])["r2"][0, 0]
                gains.append(with_sv - without)
            return float(np.mean(gains))

        base = pooled_gain()
        dropped = [(pooled_gain(person), samples.loc[person, "sample"]) for person in np.flatnonzero(carriers)]
        low, low_sample = min(dropped) if dropped else (base, "")
        high, _ = max(dropped) if dropped else (base, "")
        row.update({"pooled_gain": base, "gain_min_leaving_one_carrier_out": low, "gain_max_leaving_one_carrier_out": high,
                    "most_influential_carrier": low_sample})
        influence_rows += [{"method": method, "design": design, "gene_id": gene_id, "sample": sample, "gain_without": value, "gain": base} for value, sample in dropped]
        key = (chrom, int(record["pos"]), record["id"])
        if key not in panel_cache:
            panel_cache[key] = panel_genotypes(pathlib.Path(panel_dir), chrom, int(record["pos"]), record["id"])
        if panel_cache[key] is not None:
            row.update(germline_checks(*panel_cache[key], pedigree, set(samples.loc[carriers, "sample"])))
        target = gene_id.split(".")[0]
        row["gene_mappability"] = float(mappability.get(target, np.nan))
        overlapping = []
        for partner, score in partners.get(target, {}).items():
            if partner in gencode.index:
                where = gencode.loc[partner]
                if where["chrom"] == chrom and where["end"] >= row["sv_start"] and where["start"] <= row["sv_end"]:
                    overlapping.append(f"{where['name'] or partner}:{score:g}")
        row.update({"cross_mappable_partners": len(partners.get(target, {})), "cross_mappable_partners_inside_sv": ";".join(overlapping),
                    "cross_mappability_flag": bool(overlapping)})
        rows.append(row)
    names = gencode["name"].to_dict()
    frame = pd.DataFrame(rows)
    if len(frame):
        frame.insert(3, "gene_name", [names.get(gene.split(".")[0], "") for gene in frame["gene_id"]])
    frame.to_csv(out / "sv_artifacts.tsv", sep="\t", index=False)
    pd.DataFrame(carrier_rows).to_csv(out / "carriers_by_group_and_fold.tsv", sep="\t", index=False)
    pd.DataFrame(influence_rows).to_csv(out / "leave_one_carrier_out.tsv.gz", sep="\t", index=False)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tests", required=True, help="sv_gene_tests.tsv.gz from robust.py")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--designs", nargs="+", default=["loso", "random5"])
    parser.add_argument("--panel", required=True)
    parser.add_argument("--pedigree", required=True)
    parser.add_argument("--crossmap", required=True)
    parser.add_argument("--gencode", required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    frame = run(arguments.results, arguments.dataset, arguments.tests, arguments.methods, arguments.designs, arguments.panel, arguments.pedigree,
                arguments.crossmap, arguments.gencode, arguments.out)
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
