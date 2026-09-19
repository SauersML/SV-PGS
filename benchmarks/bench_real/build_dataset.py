"""Build the per-chromosome genotype and expression arrays of the MAGE cis-expression benchmark.

Inputs (public): the MAGE v1.0 expression matrix and covariates (Zenodo 10535719), and the MAGE-sample subset of
the 1kGP 3,202-sample phased SNV/INDEL/SV panel (NYGC high coverage, 2022-04-22).

Outputs per chromosome, in <root>/dataset/:
  chr<c>.dosage.npy     int8 [variants x samples], alternate-allele counts, one row per ALT allele
  chr<c>.variants.tsv   one row per dosage row
  chr<c>.genes.tsv      genes whose TSS lies on the chromosome, with their expression row index
and once: samples.tsv, expression.npy (float64 [genes x samples], inverse-normal TMM), covariates.npy, genes.tsv.
"""
import argparse
import pathlib

import cyvcf2
import numpy as np
import pandas as pd

# A structural variant is an allele of at least 50 bp or a symbolic allele: the field-standard size definition
# (Mahmoud et al. 2019, Genome Biology 20:246; HGSVC, Ebert et al. 2021, Science 372:eabf7117).
STRUCTURAL_VARIANT_MIN_BP = 50

# MAGE v1.0 ships two swapped libraries (mccoy-lab/MAGE README, 2026-03-27): the library labelled HG00237
# (SRR19762530) is NA11919 and the library labelled NA11919 (SRR19762653) is HG00237.
SWAPPED_LIBRARIES = {"SRR19762530": ("HG00237", "NA11919"), "SRR19762653": ("NA11919", "HG00237")}


def read_expression(mage_dir: pathlib.Path):
    quants = mage_dir / "QTL_results/eQTL_results/expression_quants"
    expression = pd.read_csv(quants / "inverse_normal_TMM.filtered.TSS.MAGE.v1.0.bed.gz", sep="\t")
    expression = expression.rename(columns={expression.columns[0]: "chrom", expression.columns[1]: "start", expression.columns[2]: "end", expression.columns[3]: "gene_id"})
    covariates = pd.read_csv(quants / "eQTL_covariates.tab.gz", sep="\t", index_col=0)
    metadata = pd.read_csv(mage_dir / "sample_library_info/sample.metadata.MAGE.v1.0.txt", sep="\t")
    return expression, covariates, metadata


def fix_library_swap(expression: pd.DataFrame, covariates: pd.DataFrame, metadata: pd.DataFrame):
    """Repair the two mislabelled libraries. Returns the individuals to drop and a note.

    Both in the analysis set: swap their expression-derived columns. One in the set: its column holds another
    individual's expression; drop the labelled individual (MAGE v1.0 keeps SRR19762653, labelled NA11919 but from
    HG00237, while HG00237's own library SRR19762247 is also kept, so NA11919 has no true expression in the set).
    """
    present = [accession for accession in SWAPPED_LIBRARIES if accession in set(metadata["SRA_accession"])]
    if len(present) == 0:
        return expression, covariates, [], "no swapped library in the 731-library set"
    if len(present) == 2:
        first, second = SWAPPED_LIBRARIES[present[0]][0], SWAPPED_LIBRARIES[present[1]][0]
        expression[[first, second]] = expression[[second, first]].to_numpy()
        expression_derived = [name for name in covariates.index if name.startswith("PEER")]
        covariates.loc[expression_derived, [first, second]] = covariates.loc[expression_derived, [second, first]].to_numpy()
        return expression, covariates, [], f"swapped expression and PEER columns of {first} and {second}"
    labelled, true_individual = SWAPPED_LIBRARIES[present[0]]
    return expression, covariates, [labelled], f"dropped {labelled}: library {present[0]} labelled {labelled} is {true_individual}"


def variant_rows(record: cyvcf2.Variant, alleles: np.ndarray):
    """Yield (dosage, attributes) for each ALT allele of a record that is polymorphic in the samples."""
    reference_length = len(record.REF)
    sv_type = record.INFO.get("SVTYPE")
    sv_length_info = record.INFO.get("SVLEN")
    end = record.INFO.get("END")
    for allele_index, alternate in enumerate(record.ALT, start=1):
        dosage = (alleles == allele_index).sum(axis=1).astype(np.int8)
        allele_count = int(dosage.sum())
        if allele_count == 0 or allele_count == 2 * dosage.shape[0]:
            continue
        symbolic = alternate.startswith("<")
        length_change = 0 if symbolic else len(alternate) - reference_length
        if sv_length_info is not None:
            sv_length = abs(int(sv_length_info if np.isscalar(sv_length_info) else sv_length_info[allele_index - 1]))
        elif symbolic and end is not None:
            sv_length = int(end) - record.POS
        else:
            sv_length = abs(length_change)
        is_sv = symbolic or max(reference_length, 0 if symbolic else len(alternate)) >= STRUCTURAL_VARIANT_MIN_BP or abs(length_change) >= STRUCTURAL_VARIANT_MIN_BP
        yield dosage, (record.POS, int(end) if end is not None else record.POS + reference_length - 1, record.ID or ".", reference_length,
                       -1 if symbolic else len(alternate), symbolic, sv_type or ("INS" if length_change > 0 else "DEL" if length_change < 0 else "."),
                       sv_length if is_sv else 0, is_sv)


def read_rows(bcf_path: pathlib.Path, samples: list, structural_only: bool):
    reader = cyvcf2.VCF(str(bcf_path), samples=samples, gts012=False)
    if list(reader.samples) != samples:
        raise ValueError("sample order in the BCF differs from the requested order")
    dosages, attributes = [], []
    missing_records = 0
    for record in reader:
        alleles = record.genotype.array()[:, :2]
        if (alleles < 0).any():
            missing_records += 1
            continue
        for dosage, attribute in variant_rows(record, alleles):
            if structural_only and not attribute[-1]:
                continue
            dosages.append(dosage)
            attributes.append(attribute)
    return dosages, attributes, missing_records


def build_chromosome(root: pathlib.Path, samples: list, out_dir: pathlib.Path, chrom: str):
    """Panel rows (SNV/indel/SV), then PanGenie SV rows when that subset exists, tagged by source."""
    dosages, attributes, missing_records = read_rows(root / f"data/geno/chr{chrom}.bcf", samples, structural_only=False)
    sources = ["panel"] * len(dosages)
    pangenie_path = root / f"data/geno/pangenie_chr{chrom}.bcf"
    if pangenie_path.exists():
        pangenie_dosages, pangenie_attributes, pangenie_missing = read_rows(pangenie_path, samples, structural_only=True)
        dosages += pangenie_dosages
        attributes += pangenie_attributes
        sources += ["pangenie"] * len(pangenie_dosages)
        missing_records += pangenie_missing
    matrix = np.lib.format.open_memmap(out_dir / f"chr{chrom}.dosage.npy", mode="w+", dtype=np.int8, shape=(len(dosages), len(samples)))
    for row, dosage in enumerate(dosages):
        matrix[row] = dosage
    matrix.flush()
    table = pd.DataFrame(attributes, columns=["pos", "end", "id", "ref_len", "alt_len", "symbolic", "sv_type", "sv_length", "is_sv"])
    table["source"] = sources
    table.to_csv(out_dir / f"chr{chrom}.variants.tsv", sep="\t", index=False)
    return len(dosages), int(table["is_sv"].sum()), int((table["source"] == "pangenie").sum()), missing_records


def write_shared(root: pathlib.Path, out_dir: pathlib.Path):
    expression, covariates, metadata = read_expression(root / "data/mage")
    if metadata["sample_kgpID"].duplicated().any():
        raise ValueError("duplicate individuals in the 731-library metadata")
    expression, covariates, dropped, swap_note = fix_library_swap(expression, covariates, metadata)
    samples = sorted(set(metadata["sample_kgpID"]) - set(dropped))
    autosomal = expression[expression["chrom"].isin([f"chr{number}" for number in range(1, 23)])].reset_index(drop=True)
    np.save(out_dir / "expression.npy", autosomal[samples].to_numpy(dtype=np.float64))
    autosomal[["chrom", "start", "end", "gene_id"]].assign(tss=autosomal["start"] + 1).to_csv(out_dir / "genes.tsv", sep="\t", index=False)
    np.save(out_dir / "covariates.npy", np.vstack([(covariates.loc["sex", samples] == "XY").to_numpy(dtype=np.float64)] +
                                                  [covariates.loc[name, samples].to_numpy(dtype=np.float64) for name in covariates.index if name != "sex"]).T)
    pd.DataFrame({"covariate": list(covariates.index)}).to_csv(out_dir / "covariate_names.tsv", sep="\t", index=False)
    ped = pd.read_csv(root / "data/kgp/20130606_g1k_3202_samples_ped_population.txt", sep=" ")
    sample_table = pd.DataFrame({"sample": samples}).merge(ped, left_on="sample", right_on="SampleID", how="left")
    if sample_table["Superpopulation"].isna().any():
        raise ValueError("a MAGE sample is missing from the 1kGP pedigree table")
    sample_table[["sample", "FamilyID", "FatherID", "MotherID", "Sex", "Population", "Superpopulation"]].to_csv(out_dir / "samples.tsv", sep="\t", index=False)
    (out_dir / "BUILD_NOTES.txt").write_text(f"{swap_note}\nexcluded chrX (haploid in XY samples)\n")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--chromosomes", nargs="*", default=[])
    parser.add_argument("--shared", action="store_true", help="also write the chromosome-independent arrays (samples, expression, covariates, genes)")
    arguments = parser.parse_args()
    root = pathlib.Path(arguments.root)
    out_dir = root / "dataset"
    out_dir.mkdir(exist_ok=True)
    samples = write_shared(root, out_dir) if arguments.shared else pd.read_csv(out_dir / "samples.tsv", sep="\t")["sample"].tolist()
    for chrom in arguments.chromosomes:
        count, sv_count, pangenie_count, missing = build_chromosome(root, samples, out_dir, chrom)
        print(f"chr{chrom}: {count} dosage rows, {sv_count} SV rows ({pangenie_count} PanGenie), {missing} records with missing genotypes skipped", flush=True)


if __name__ == "__main__":
    main()
