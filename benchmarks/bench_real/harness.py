"""The bench-real harness: cis-expression prediction in MAGE with 1kGP SNV/indel and SV genotypes.

A method is a callable ``fit(train: TrainData) -> predictor``; the predictor has ``predict(genotypes) -> np.ndarray``.
The harness also records each prediction with the SV columns set to their training means, for SV credit.
Methods never see test phenotypes: the harness builds TrainData (genotypes, phenotype, variant annotations) and
passes only test genotypes to ``predict``. Test phenotypes are read only when scoring.

Phenotype: MAGE inverse-normal TMM expression, residualized on the MAGE eQTL covariates (sex, 5 genotype PCs,
60 PEER factors) by OLS fitted on the training samples only; test phenotypes are adjusted with the training fit.
Target gene: GENCODE v38 gene body, strand, merged exons and merged CDS (1-based closed, like POS/END).
Genotypes: alternate-allele counts 0/1/2 from the 1kGP phased panel; variants monomorphic in the training samples
are dropped. Window: the variant interval overlaps TSS +/- 1 Mb, the cis window of MAGE's own mapping and of
GTEx (GTEx Consortium 2020, Science 369:1318).
"""
import dataclasses
import hashlib
import importlib.util
import json
import subprocess
import os
import pathlib
import time

import numpy as np
import pandas as pd

CIS_RADIUS_BP = 1_000_000
FEATURE_SETS = ("snv", "snv_sv", "snv_pgsv", "sv", "pgsv", "snv_matched")
MATCHED_SEED = hashlib.sha256(b"bench-real/snv_matched").digest()


@dataclasses.dataclass(frozen=True)
class Variants:
    position: np.ndarray
    end: np.ndarray
    distance_to_tss: np.ndarray
    is_sv: np.ndarray
    sv_type: np.ndarray
    sv_length: np.ndarray
    allele_length_change: np.ndarray
    train_allele_frequency: np.ndarray
    source: np.ndarray


@dataclasses.dataclass(frozen=True)
class TrainData:
    gene_id: str
    chrom: str
    tss: int
    genotypes: np.ndarray
    phenotype: np.ndarray
    variants: Variants
    superpopulation: np.ndarray
    population: np.ndarray
    gene_start: int
    gene_end: int
    strand: str
    exons: np.ndarray
    coding_exons: np.ndarray


class Dataset:
    def __init__(self, dataset_dir):
        self.directory = pathlib.Path(dataset_dir)
        self.samples = pd.read_csv(self.directory / "samples.tsv", sep="\t")
        self.genes = pd.read_csv(self.directory / "genes.tsv", sep="\t")
        self.expression = np.load(self.directory / "expression.npy")
        self.covariates = np.load(self.directory / "covariates.npy")
        self.splits = {split["name"]: split for split in json.loads((self.directory / "splits.json").read_text())}
        self.gene_annotation = json.loads((self.directory / "gene_annotation.json").read_text())
        self.sample_index = {sample: index for index, sample in enumerate(self.samples["sample"])}
        self._chromosomes = {}

    def chromosome(self, chrom: str):
        """The variant table and memory-mapped dosages of one chromosome; only the latest one stays cached."""
        if chrom not in self._chromosomes:
            self._chromosomes.clear()
            number = chrom.removeprefix("chr")
            table = pd.read_csv(self.directory / f"chr{number}.variants.tsv", sep="\t")
            if "source" not in table:
                table["source"] = "panel"
            dosage = np.load(self.directory / f"chr{number}.dosage.npy", mmap_mode="r")
            self._chromosomes[chrom] = (table, dosage)
        return self._chromosomes[chrom]

    def gene_rows(self, chromosomes, gene_prefix=None, gene_list=None):
        """Genes on the chromosomes; with gene_prefix, only those among the first gene_prefix of gene_order.tsv; with
        gene_list (a TSV with a gene_id column, e.g. a frozen screened list), only the genes it names."""
        on_chromosomes = self.genes["chrom"].isin(chromosomes)
        if gene_prefix is not None:
            leading = set(pd.read_csv(self.directory / "gene_order.tsv", sep="\t")["gene_id"].head(gene_prefix))
            on_chromosomes &= self.genes["gene_id"].isin(leading)
        if gene_list is not None:
            named = set(pd.read_csv(gene_list, sep="\t")["gene_id"])
            unknown = named - set(self.genes["gene_id"])
            if unknown:
                raise ValueError(f"{len(unknown)} listed genes are not benchmark genes, e.g. {sorted(unknown)[:3]}")
            on_chromosomes &= self.genes["gene_id"].isin(named)
        return [int(index) for index in self.genes.index[on_chromosomes]]

    def cis_rows(self, chrom: str, tss: int):
        table, _ = self.chromosome(chrom)
        start, end = table["pos"].to_numpy(), table["end"].to_numpy()
        return np.flatnonzero((end >= tss - CIS_RADIUS_BP) & (start <= tss + CIS_RADIUS_BP))


def residualize(phenotype, covariates, train_index, test_index):
    design = np.column_stack([np.ones(len(phenotype)), covariates])
    coefficients, *_ = np.linalg.lstsq(design[train_index], phenotype[train_index], rcond=None)
    fitted = design @ coefficients
    return phenotype[train_index] - fitted[train_index], phenotype[test_index] - fitted[test_index]


@dataclasses.dataclass(frozen=True)
class GeneWindow:
    """A gene's cis-window genotypes for all samples, read once and sliced per split."""
    gene_row: int
    gene_id: str
    chrom: str
    tss: int
    genotypes: np.ndarray
    table: pd.DataFrame


def load_gene_window(dataset: Dataset, gene_row: int):
    gene = dataset.genes.iloc[gene_row]
    chrom, tss = gene["chrom"], int(gene["tss"])
    rows = dataset.cis_rows(chrom, tss)
    table, dosage = dataset.chromosome(chrom)
    return GeneWindow(gene_row=gene_row, gene_id=gene["gene_id"], chrom=chrom, tss=tss,
                      genotypes=np.asarray(dosage[rows], dtype=np.float32).T, table=table.iloc[rows].reset_index(drop=True))


def build_gene_task(dataset: Dataset, window: GeneWindow, split: dict):
    train_index = np.array([dataset.sample_index[sample] for sample in split["train"]])
    test_index = np.array([dataset.sample_index[sample] for sample in split["test"]])
    train_genotypes, test_genotypes = window.genotypes[train_index], window.genotypes[test_index]
    allele_count = train_genotypes.sum(axis=0)
    # A column that is constant in the training samples carries no information and has zero variance, which
    # breaks standardization: that is every sample 0 or 2, but also every sample heterozygous (seen in PanGenie
    # calls), so the test is the training variance, not the allele count.
    polymorphic = train_genotypes.var(axis=0) > 0
    train_genotypes, test_genotypes = train_genotypes[:, polymorphic], test_genotypes[:, polymorphic]
    selected = window.table[polymorphic]
    position, end = selected["pos"].to_numpy(), selected["end"].to_numpy()
    distance = np.where(position > window.tss, position - window.tss, np.where(end < window.tss, end - window.tss, 0))
    alternate_length = selected["alt_len"].to_numpy()
    variants = Variants(position=position, end=end, distance_to_tss=distance, is_sv=selected["is_sv"].to_numpy(dtype=bool),
                        sv_type=selected["sv_type"].to_numpy(dtype=str), sv_length=selected["sv_length"].to_numpy(),
                        allele_length_change=np.where(alternate_length < 0, 0, alternate_length - selected["ref_len"].to_numpy()),
                        train_allele_frequency=allele_count[polymorphic] / (2 * len(train_index)), source=selected["source"].to_numpy(dtype=str))
    train_phenotype, test_phenotype = residualize(dataset.expression[window.gene_row], dataset.covariates, train_index, test_index)
    samples = dataset.samples
    gene = dataset.gene_annotation[window.gene_id]
    train = TrainData(gene_id=window.gene_id, chrom=window.chrom, tss=window.tss, genotypes=train_genotypes, phenotype=train_phenotype, variants=variants,
                      superpopulation=samples["Superpopulation"].to_numpy()[train_index], population=samples["Population"].to_numpy()[train_index],
                      gene_start=gene["start"], gene_end=gene["end"], strand=gene["strand"],
                      exons=np.array(gene["exons"], dtype=np.int64).reshape(-1, 2), coding_exons=np.array(gene["coding_exons"], dtype=np.int64).reshape(-1, 2))
    return train, test_genotypes, test_phenotype, test_index


def matched_small_variants(variants: Variants, draw_key: str):
    """Panel SNVs/indels matched one-to-one to the panel SVs of the window, on training allele frequency and distance.

    Each SV, taken in a seeded random order, gets the nearest unused small variant in (logit MAF, log(1 + |distance
    to TSS| in bp)), measured by the Mahalanobis distance of the small variants' own covariance of those two
    coordinates, so neither unit dominates. The seed (sha256 of the sealed key sha256("bench-real/snv_matched") and
    the gene/split draw key) fixes the order in which SVs claim neighbours and breaks exact ties; with continuous
    coordinates most draws are the plain nearest neighbours. Each fold's draw is reproducible and method-independent.
    """
    small = np.flatnonzero((variants.source == "panel") & ~variants.is_sv)
    structural = np.flatnonzero((variants.source == "panel") & variants.is_sv)
    frequency = np.minimum(variants.train_allele_frequency, 1 - variants.train_allele_frequency)
    coordinates = np.column_stack([np.log(frequency / (1 - frequency)), np.log1p(np.abs(variants.distance_to_tss))])
    generator = np.random.default_rng(int.from_bytes(hashlib.sha256(MATCHED_SEED + draw_key.encode()).digest()[:8], "little"))
    candidates = small[generator.permutation(len(small))]
    whitening = np.linalg.cholesky(np.linalg.inv(np.cov(coordinates[candidates], rowvar=False)))
    candidate_points = coordinates[candidates] @ whitening
    available = np.ones(len(candidates), dtype=bool)
    chosen = []
    for target in structural[generator.permutation(len(structural))]:
        if not available.any():
            break
        distances = np.where(available, ((candidate_points - coordinates[target] @ whitening) ** 2).sum(axis=1), np.inf)
        best = int(np.argmin(distances))
        available[best] = False
        chosen.append(candidates[best])
    mask = np.zeros(len(variants.is_sv), dtype=bool)
    mask[chosen] = True
    return mask


def feature_mask(variants: Variants, feature_set: str, draw_key: str):
    """snv: panel SNVs/indels; snv_sv: all panel rows; snv_pgsv: panel SNVs/indels plus PanGenie SVs; sv: panel SVs;
    pgsv: PanGenie SVs; snv_matched: as many panel SNVs/indels as panel SVs, matched to them (matched_small_variants)."""
    panel = variants.source == "panel"
    small = panel & ~variants.is_sv
    pangenie_sv = (variants.source == "pangenie") & variants.is_sv
    if feature_set == "snv_matched":
        return matched_small_variants(variants, draw_key)
    return {"snv": small, "snv_sv": panel, "snv_pgsv": small | pangenie_sv, "sv": panel & variants.is_sv, "pgsv": pangenie_sv}[feature_set]


def subset(train: TrainData, test_genotypes: np.ndarray, feature_set: str, split_name: str):
    keep = feature_mask(train.variants, feature_set, f"{train.gene_id}/{split_name}")
    variants = Variants(**{field.name: getattr(train.variants, field.name)[keep] for field in dataclasses.fields(Variants)})
    return dataclasses.replace(train, genotypes=train.genotypes[:, keep], variants=variants), test_genotypes[:, keep]


def load_method(spec: str):
    """spec is '<path/to/file.py>:<callable>'."""
    path, name = spec.rsplit(":", 1)
    module_spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return getattr(module, name)


_WORKER = {}


def _init_worker(dataset_dir, method_spec, feature_sets):
    _WORKER["dataset"] = Dataset(dataset_dir)
    _WORKER["fit"] = load_method(method_spec)
    _WORKER["feature_sets"] = feature_sets


def _without_structural_variants(train: TrainData, test_genotypes: np.ndarray):
    """Test genotypes with every SV column set to its training mean, so a predictor's SV contribution drops out."""
    masked = test_genotypes.copy()
    masked[:, train.variants.is_sv] = train.genotypes[:, train.variants.is_sv].mean(axis=0)
    return masked


def _run_gene(arguments):
    gene_row, split_names = arguments
    dataset, fit = _WORKER["dataset"], _WORKER["fit"]
    window = load_gene_window(dataset, gene_row)
    results = []
    for split_name in split_names:
        train_all, test_all, test_phenotype, test_index = build_gene_task(dataset, window, dataset.splits[split_name])
        for feature_set in _WORKER["feature_sets"]:
            train, test_genotypes = subset(train_all, test_all, feature_set, split_name)
            started = time.process_time()
            predictor = fit(train)
            prediction = np.asarray(predictor.predict(test_genotypes), dtype=np.float64)
            seconds = time.process_time() - started
            without_sv = np.asarray(predictor.predict(_without_structural_variants(train, test_genotypes)), dtype=np.float64) if train.variants.is_sv.any() else prediction
            results.append((gene_row, split_name, feature_set, test_index, prediction, without_sv, test_phenotype, train.genotypes.shape[1], int(train.variants.is_sv.sum()), seconds))
    return results


def run(dataset_dir, method_spec, method_name, design, chromosomes, out_dir, workers, feature_sets=FEATURE_SETS, gene_prefix=None, gene_list=None):
    """Out-of-fold predictions of one method for every gene on the chromosomes, under one split design."""
    from multiprocessing import get_context

    dataset = Dataset(dataset_dir)
    split_names = [name for name in dataset.splits if name.startswith(design + "/")]
    gene_rows = dataset.gene_rows(chromosomes, gene_prefix, gene_list)
    sample_count = len(dataset.samples)
    predictions = {feature_set: np.full((len(gene_rows), sample_count), np.nan, dtype=np.float32) for feature_set in feature_sets}
    predictions_without_sv = {feature_set: np.full((len(gene_rows), sample_count), np.nan, dtype=np.float32) for feature_set in feature_sets}
    truth = np.full((len(gene_rows), sample_count), np.nan, dtype=np.float32)
    log = []
    position_of_row = {row: position for position, row in enumerate(gene_rows)}
    with get_context("fork").Pool(workers, initializer=_init_worker, initargs=(dataset_dir, method_spec, feature_sets)) as pool:
        for results in pool.imap_unordered(_run_gene, [(row, split_names) for row in gene_rows], chunksize=1):
            for gene_row, split_name, feature_set, test_index, prediction, without_sv, test_phenotype, variant_count, sv_count, seconds in results:
                position = position_of_row[gene_row]
                predictions[feature_set][position, test_index] = prediction
                predictions_without_sv[feature_set][position, test_index] = without_sv
                truth[position, test_index] = test_phenotype
                log.append((dataset.genes.iloc[gene_row]["gene_id"], split_name, feature_set, variant_count, sv_count, seconds))
    out = pathlib.Path(out_dir) / method_name / design
    out.mkdir(parents=True, exist_ok=True)
    tag = "_".join(chromosomes)
    method_file = pathlib.Path(method_spec.rsplit(":", 1)[0])
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=pathlib.Path(__file__).resolve().parent, capture_output=True, text=True, check=True).stdout.strip()
    (out / f"{tag}.run.json").write_text(json.dumps({
        "method": method_spec, "method_sha256": hashlib.sha256(method_file.read_bytes()).hexdigest(), "harness_commit": commit,
        "design": design, "chromosomes": list(chromosomes), "feature_sets": list(feature_sets), "gene_prefix": gene_prefix,
        "gene_list": str(gene_list) if gene_list is not None else None,
        "gene_list_sha256": hashlib.sha256(pathlib.Path(gene_list).read_bytes()).hexdigest() if gene_list is not None else None,
        "genes": len(gene_rows), "splits_sha256": (dataset.directory / "splits.sha256").read_text().strip()}, indent=1))
    for feature_set in feature_sets:
        np.save(out / f"{tag}.{feature_set}.predictions.npy", predictions[feature_set])
        np.save(out / f"{tag}.{feature_set}.predictions_without_sv.npy", predictions_without_sv[feature_set])
    np.save(out / f"{tag}.truth.npy", truth)
    dataset.genes.iloc[gene_rows].to_csv(out / f"{tag}.genes.tsv", sep="\t", index=False)
    pd.DataFrame(log, columns=["gene_id", "split", "feature_set", "variants", "sv_variants", "cpu_seconds"]).to_csv(out / f"{tag}.log.tsv", sep="\t", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True, help="path/to/file.py:callable")
    parser.add_argument("--name", required=True)
    parser.add_argument("--design", required=True, choices=["random5", "loso"])
    parser.add_argument("--chromosomes", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("RUNQ_CORES", "1")))
    parser.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS), choices=FEATURE_SETS)
    parser.add_argument("--gene-prefix", type=int, help="run only genes among the first N of the sealed gene_order.tsv")
    parser.add_argument("--genes", help="run only the genes a TSV with a gene_id column names (a frozen screened list)")
    arguments = parser.parse_args()
    run(arguments.dataset, arguments.method, arguments.name, arguments.design, arguments.chromosomes, arguments.out, arguments.workers,
        tuple(arguments.feature_sets), arguments.gene_prefix, arguments.genes)
