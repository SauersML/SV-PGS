"""The bench-real harness: cis-expression prediction in MAGE with 1kGP SNV/indel and SV genotypes.

A method is a callable ``fit(train: TrainData) -> predictor``; the predictor has ``predict(genotypes) -> np.ndarray``.
A method that pools hyperparameters across genes uses the batch contract instead:
``fit_batch(trains: Sequence[TrainData]) -> list[predictor]``, called once per split with every selected gene.
The most general contract is ``fit_views(views) -> {(gene_id, split, feature_set): predictor}`` (or an iterator of
such pairs), called once with a lazy mapping of every requested view, so a method can share work across overlapping
windows, folds and nested feature sets. The per-gene and batch contracts are its special cases.
The harness also records each prediction with the SV columns set to their training means, for SV credit.
Methods never see test phenotypes: the harness builds TrainData (genotypes, phenotype, variant annotations) and
passes only test genotypes to ``predict``. Test phenotypes are read only when scoring.

Phenotype: MAGE inverse-normal TMM expression, residualized on the MAGE eQTL covariates (sex, 5 genotype PCs,
60 PEER factors) by OLS fitted on the training samples only; test phenotypes are adjusted with the training fit.
Covariates are fixed effects for every method: TrainData.covariates carries the training rows (a predictor whose
predict accepts ``covariates`` receives the test rows), and every method's score is compared with the truth by one
rule, predict_for_truth: the score is residualized on [1, covariates] with training OLS coefficients, as the truth is.
That saved truth is the fit's target, not the held-out metric: report.py scores expression and score both residualized
on [1, covariates] within each held-out group (the within-group partial r^2, lead ruling).
MAGE computed the PEER factors from all 731 samples' expression, held-out samples included. That is unsupervised,
shared by every method, and not a per-method leak, but the held-out expression did shape one covariate set.
Target gene: GENCODE v38 gene body, strand, merged exons and merged CDS (1-based closed, like POS/END).
Genotypes: alternate-allele counts 0/1/2 from the 1kGP phased panel; variants monomorphic in the training samples
are dropped. Window: the variant interval overlaps TSS +/- 1 Mb, the cis window of MAGE's own mapping and of
GTEx (GTEx Consortium 2020, Science 369:1318).
"""
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import subprocess
import os
import pathlib
import time
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

CIS_RADIUS_BP = 1_000_000
SEALED_GENES = "sealed_confirmation_genes.tsv"
PARENT_DATASET = "parent_dataset.txt"
# Per-row measurement-quality columns; a source without one of them gets 1.0 there (a direct call).
MEASUREMENT_COLUMNS = ("reliability", "concordance", "called_r2")
# Recorded in run.json: how a score is compared with the truth (predict_for_truth). Results without it are "pre-C2".
PREDICTION_RULE = "C2: score residualized on [1, covariates] by training OLS, as the truth is"
FEATURE_SETS = ("snv", "snv_sv", "snv_pgsv", "sv", "pgsv", "snv_matched", "hgsvc3", "snv_hgsvc3", "ont", "snv_ont",
                "sv_merged", "snv_sv_merged", "pgsv_merged", "snv_pgsv_merged", "hgsvc3_merged", "snv_hgsvc3_merged", "gatksv", "snv_sv_cn",
                "svimp", "snv_svimp", "ctyper", "snv_ctyper", "hprc2", "snv_hprc2")
# Rows of these sources are SVs; each set is its source alone, or panel SNVs/indels plus it (lr-sv's derived datasets).
SOURCE_SETS = {"hgsvc3": "hgsvc3", "ont": "ont", "sv_merged": "panel_merged", "pgsv_merged": "pangenie_merged", "hgsvc3_merged": "hgsvc3_merged",
               "gatksv": "gatksv", "svimp": "svimp", "ctyper": "ctyper", "hprc2": "hprc2"}
JOINT_SOURCE_SETS = {"snv_hgsvc3": "hgsvc3", "snv_ont": "ont", "snv_sv_merged": "panel_merged", "snv_pgsv_merged": "pangenie_merged",
                     "snv_hgsvc3_merged": "hgsvc3_merged", "snv_sv_cn": "gatksv", "snv_svimp": "svimp", "snv_ctyper": "ctyper",
                     "snv_hprc2": "hprc2"}
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
    # Row of each variant in its gene window (load_gene_window's table), a stable key for saved coefficients.
    window_row: np.ndarray = None
    # Row of each variant in its chromosome's variant table: with source, it identifies one column across overlapping
    # gene windows, so a method can share work between genes (fit_views).
    chromosome_row: np.ndarray = None
    # Each column's measurement reliability: the expected squared correlation of the stored dosage with the true genotype.
    # 1 for direct calls; a derived dataset supplies it (a variants.tsv "reliability" column) where dosages were filled
    # or imputed, and an imputed overlay supplies its imputation r^2 estimate (svimp.npz "dr2"). A method that cannot
    # use it simply sees the dosages.
    reliability: np.ndarray = None
    # Each column's expected genotype concordance where dosages were filled (the share of people whose stored dosage
    # equals the called one; lr-sv's GATK-SV no-call fills). A concordance, not an r^2: 1 where absent.
    concordance: np.ndarray = None
    # The exact squared correlation of the stored dosage with the CALLED genotype under the fill mixture (lr-sv): an upper
    # bound on the r^2 with the true genotype, so not the reliability either. 1 where absent.
    # In reliability, concordance and called_r2 alike, NaN means "undefined" (e.g. called_r2 on a row whose called people
    # are all homozygous reference, where every carrier is a fill): no reported value, never a value of 0 or 1.
    called_r2: np.ndarray = None


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
    # The training rows of the covariates the phenotype was residualized on (sex, 5 genotype PCs, 60 PEER factors;
    # dataset/covariate_names.tsv), without the intercept. phenotype is already orthogonal to [1, covariates].
    covariates: np.ndarray = None


class _StackedDosage:
    """Rows of several memory-mapped (variants x samples) int8 arrays, in a merged row order, read by fancy indexing
    without concatenating the arrays: source_of[i] names the array holding merged row i, and row_of[i] its row there."""

    def __init__(self, parts, source_of, row_of):
        self.parts, self.source_of, self.row_of = parts, source_of, row_of
        self.shape = (len(source_of), parts[0].shape[1])

    def __getitem__(self, rows):
        rows = np.asarray(rows)
        out = np.empty((len(rows), self.shape[1]), dtype=self.parts[0].dtype)
        for index, part in enumerate(self.parts):
            chosen = np.flatnonzero(self.source_of[rows] == index)
            if len(chosen):
                out[chosen] = part[self.row_of[rows[chosen]]]
        return out


class Dataset:
    def __init__(self, dataset_dir, overlay_dir=None, rows_dirs=None, sample_subset=None):
        self.directory = pathlib.Path(dataset_dir)
        # Extra-rows overlays (lr-sv's SV sources, stored beside the parent instead of as full copies): each directory
        # holds chrN.variants.tsv and chrN.dosage.npy (int8, rows x the parent's samples, in samples.tsv order). Per
        # chromosome the table is the parent's rows, then each directory's rows in the given order, stably sorted by pos,
        # exactly as a full derived copy is built, so windows, row order and chromosome_row equal the full copy's.
        self.rows_dirs = [pathlib.Path(path) for path in (rows_dirs or [])]
        # A sample subset (e.g. the 260 people with ONT calls): every split's train and test are cut to it, and any
        # stored genotype read for a subset person must be a real value (the harness refuses negative placeholders).
        self.sample_subset = pathlib.Path(sample_subset) if sample_subset is not None else None
        # Imputed SV dosages (bench-sim's svimp): per chromosome, <overlay>/<chrom>.svimp.npz with rows (indices into the
        # chromosome's variant table, panel SV rows) and ds (float32 [rows x samples] in samples.tsv order). They enter a
        # gene window as extra columns with source "svimp", beside the called genotypes they impute.
        self.overlay_dir = pathlib.Path(overlay_dir) if overlay_dir is not None else None
        self.samples = pd.read_csv(self.directory / "samples.tsv", sep="\t")
        self.genes = pd.read_csv(self.directory / "genes.tsv", sep="\t")
        self.expression = np.load(self.directory / "expression.npy")
        self.covariates = np.load(self.directory / "covariates.npy")
        self.splits = {split["name"]: split for split in json.loads((self.directory / "splits.json").read_text())}
        if self.sample_subset is not None:
            keep = set(pd.read_csv(self.sample_subset, sep="\t", header=None)[0].astype(str)) - {"sample"}
            unknown = keep - set(self.samples["sample"])
            if unknown:
                raise ValueError(f"{len(unknown)} subset samples are not dataset samples, e.g. {sorted(unknown)[:3]}")
            restricted = {name: dict(split, train=[s for s in split["train"] if s in keep], test=[s for s in split["test"] if s in keep])
                          for name, split in self.splits.items()}
            self.splits = {name: split for name, split in restricted.items() if split["train"] and split["test"]}
        self.gene_annotation = json.loads((self.directory / "gene_annotation.json").read_text())
        self.sample_index = {sample: index for index, sample in enumerate(self.samples["sample"])}
        self._chromosomes = {}
        self._parent_position = None

    def overlay(self, chrom: str):
        """(rows, dosages) of the chromosome's imputed SV overlay, or None."""
        if self.overlay_dir is None:
            return None
        path = self.overlay_dir / f"{chrom}.svimp.npz"
        if not path.exists():
            return None
        if getattr(self, "_overlay_chrom", None) != chrom:
            data = np.load(path)
            if data["ds"].shape[1] != len(self.samples):
                raise ValueError(f"{path}: {data['ds'].shape[1]} samples, the dataset has {len(self.samples)}")
            if "samples" in data and list(data["samples"]) != list(self.samples["sample"]):
                raise ValueError(f"{path}: sample order differs from samples.tsv")
            self._overlay_chrom, self._overlay = chrom, (data["rows"].astype(np.int64), data["ds"])
            self._overlay_dr2 = data["dr2"].astype(np.float64) if "dr2" in data else None
        return self._overlay

    def overlay_reliability(self, chrom: str):
        """The overlay's per-row imputation r^2 estimate (Beagle DR2), or None."""
        return self._overlay_dr2 if self.overlay(chrom) is not None else None

    def chromosome(self, chrom: str):
        """The variant table and memory-mapped dosages of one chromosome; only the latest one stays cached."""
        if chrom not in self._chromosomes:
            self._chromosomes.clear()
            number = chrom.removeprefix("chr")
            table = pd.read_csv(self.directory / f"chr{number}.variants.tsv", sep="\t")
            if "source" not in table:
                table["source"] = "panel"
            dosage = np.load(self.directory / f"chr{number}.dosage.npy", mmap_mode="r")
            self._parent_position = None
            if self.rows_dirs:
                table, dosage = self._with_extra_rows(number, table, dosage)
            self._chromosomes[chrom] = (table, dosage)
        return self._chromosomes[chrom]

    def _with_extra_rows(self, number, parent_table, parent_dosage):
        tables, dosages = [parent_table], [parent_dosage]
        for directory in self.rows_dirs:
            path = directory / f"chr{number}.variants.tsv"
            if not path.exists():
                continue
            extra_dosage = np.load(directory / f"chr{number}.dosage.npy", mmap_mode="r")
            if extra_dosage.shape[1] != parent_dosage.shape[1]:
                raise ValueError(f"{directory}: {extra_dosage.shape[1]} samples, the parent has {parent_dosage.shape[1]}")
            extra = pd.read_csv(path, sep="\t")
            if len(extra) != extra_dosage.shape[0]:
                raise ValueError(f"{directory} chr{number}: {len(extra)} table rows but {extra_dosage.shape[0]} dosage rows")
            tables.append(extra)
            dosages.append(extra_dosage)
        measured = [column for column in MEASUREMENT_COLUMNS if any(column in part for part in tables)]
        tables = [part.reindex(columns=list(parent_table.columns) + [column for column in measured if column not in parent_table])
                  .assign(**{column: part[column] if column in part else 1.0 for column in measured}) for part in tables]
        source_of = np.concatenate([np.full(len(part), index) for index, part in enumerate(tables)])
        row_of = np.concatenate([np.arange(len(part)) for part in tables])
        merged = pd.concat(tables, ignore_index=True)
        order = np.argsort(merged["pos"].to_numpy(), kind="stable")
        self._parent_position = np.empty(len(parent_table), dtype=np.int64)
        self._parent_position[row_of[order][source_of[order] == 0]] = np.flatnonzero(source_of[order] == 0)
        return merged.iloc[order].reset_index(drop=True), _StackedDosage(dosages, source_of[order], row_of[order])

    def sealed_genes(self):
        """The sealed confirmation genes (lead ruling): scored once, only when the lead calls the confirmation.

        A derived dataset (a sample subset, or one with extra SV sources) names its parent in parent_dataset.txt and
        must carry the parent's sealed list byte for byte, so no derived copy can score a sealed gene."""
        path = self.directory / SEALED_GENES
        parent_file = self.directory / PARENT_DATASET
        if parent_file.exists():
            parent_sealed = pathlib.Path(parent_file.read_text().strip()) / SEALED_GENES
            if parent_sealed.exists() and (not path.exists() or path.read_bytes() != parent_sealed.read_bytes()):
                raise ValueError(f"{self.directory} does not carry its parent's sealed confirmation genes")
        return set(pd.read_csv(path, sep="\t")["gene_id"]) if path.exists() else set()

    def gene_rows(self, chromosomes, gene_prefix=None, gene_list=None, confirmation=False, gene_ranks=None):
        """Genes on the chromosomes; with gene_prefix, only those among the first gene_prefix of gene_order.tsv; with
        gene_list (a TSV with a gene_id column, e.g. a frozen screened list), only the genes it names; with gene_ranks
        (start, stop), only the list's rows start..stop-1, so a ranked list can be run top first in checkpointed chunks.

        The sealed confirmation genes are never scored unless confirmation is set, and then only they are. A gene list
        naming a sealed gene is an error rather than a silent drop, so a leak is caught where it starts."""
        sealed = self.sealed_genes()
        on_chromosomes = self.genes["chrom"].isin(chromosomes)
        on_chromosomes &= self.genes["gene_id"].isin(sealed) if confirmation else ~self.genes["gene_id"].isin(sealed)
        if gene_prefix is not None:
            leading = set(pd.read_csv(self.directory / "gene_order.tsv", sep="\t")["gene_id"].head(gene_prefix))
            on_chromosomes &= self.genes["gene_id"].isin(leading)
        if gene_list is not None:
            listed = pd.read_csv(gene_list, sep="\t")["gene_id"]
            if listed.duplicated().any():
                raise ValueError("the gene list names a gene twice")
            if gene_ranks is not None:
                listed = listed.iloc[gene_ranks[0]:gene_ranks[1]]
            named = set(listed)
            if not confirmation and named & sealed:
                raise ValueError(f"the gene list names {len(named & sealed)} sealed confirmation genes")
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
    chromosome_rows: np.ndarray = None


def load_gene_window(dataset: Dataset, gene_row: int):
    gene = dataset.genes.iloc[gene_row]
    chrom, tss = gene["chrom"], int(gene["tss"])
    rows = dataset.cis_rows(chrom, tss)
    table, dosage = dataset.chromosome(chrom)
    genotypes, window_table = np.asarray(dosage[rows], dtype=np.float32).T, table.iloc[rows].reset_index(drop=True)
    chromosome_rows = np.asarray(rows)
    overlay = dataset.overlay(chrom)
    if overlay is not None:
        imputed_rows, imputed = overlay
        if dataset._parent_position is not None:
            # svimp rows index the parent's table; with extra rows merged in, map them to merged positions.
            imputed_rows = dataset._parent_position[imputed_rows]
        present = np.flatnonzero(np.isin(imputed_rows, rows))
        if len(present):
            imputed_table = table.iloc[imputed_rows[present]].reset_index(drop=True).assign(source="svimp")
            if dataset.overlay_reliability(chrom) is not None:
                imputed_table["reliability"] = dataset.overlay_reliability(chrom)[present]
            if "reliability" in imputed_table and "reliability" not in window_table:
                window_table = window_table.assign(reliability=1.0)
            genotypes = np.hstack([genotypes, imputed[present].T.astype(np.float32)])
            window_table = pd.concat([window_table, imputed_table], ignore_index=True)
            chromosome_rows = np.concatenate([chromosome_rows, imputed_rows[present]])
    return GeneWindow(gene_row=gene_row, gene_id=gene["gene_id"], chrom=chrom, tss=tss, genotypes=genotypes, table=window_table,
                      chromosome_rows=chromosome_rows)


# The sign of a symbolic allele's length change by its SV type: deletions lose sequence, insertions and duplications
# gain it; inversions, breakends, complex and multi-allelic copy-number records have no single signed change.
LENGTH_CHANGE_SIGN = {"DEL": -1, "INS": 1, "DUP": 1}


def allele_lengths(table: pd.DataFrame):
    """(length, signed allele-length change) for every record, SV or not.

    Sequence-resolved alleles: change = len(ALT) - len(REF), and length = |change|, or the record's stored SV length when
    it has one (SVLEN). Symbolic alleles (alt_len -1): length is the stored SV length (SVLEN, else END - POS), and the
    change is signed by the SV type (LENGTH_CHANGE_SIGN). The 50 bp threshold lives only in is_sv, the reporting label:
    a length is never zeroed for being short, so a length-dependent prior sees a continuous length."""
    alternate, reference = table["alt_len"].to_numpy(), table["ref_len"].to_numpy()
    stored = table["sv_length"].to_numpy()
    symbolic = alternate < 0
    resolved_change = np.where(symbolic, 0, alternate - reference)
    length = np.where(symbolic | (stored > 0), stored, np.abs(resolved_change))
    base_type = np.array([str(value).split(":")[0] for value in table["sv_type"]])
    sign = np.array([LENGTH_CHANGE_SIGN.get(value, 0) for value in base_type])
    change = np.where(symbolic, sign * length, resolved_change)
    return length, change


def build_gene_task(dataset: Dataset, window: GeneWindow, split: dict):
    train_index = np.array([dataset.sample_index[sample] for sample in split["train"]])
    test_index = np.array([dataset.sample_index[sample] for sample in split["test"]])
    train_genotypes, test_genotypes = window.genotypes[train_index], window.genotypes[test_index]
    if (train_genotypes < 0).any() or (test_genotypes < 0).any():
        raise ValueError(f"{window.gene_id}: a negative placeholder genotype was read for a split sample (sample subset missing?)")
    allele_count = train_genotypes.sum(axis=0)
    # A column that is constant in the training samples carries no information and has zero variance, which
    # breaks standardization: that is every sample 0 or 2, but also every sample heterozygous (seen in PanGenie
    # calls), so the test is the training variance, not the allele count.
    polymorphic = train_genotypes.var(axis=0) > 0
    train_genotypes, test_genotypes = train_genotypes[:, polymorphic], test_genotypes[:, polymorphic]
    selected = window.table[polymorphic]
    position, end = selected["pos"].to_numpy(), selected["end"].to_numpy()
    distance = np.where(position > window.tss, position - window.tss, np.where(end < window.tss, end - window.tss, 0))
    length, length_change = allele_lengths(selected)
    variants = Variants(position=position, end=end, distance_to_tss=distance, is_sv=selected["is_sv"].to_numpy(dtype=bool),
                        sv_type=selected["sv_type"].to_numpy(dtype=str), sv_length=length, allele_length_change=length_change,
                        train_allele_frequency=allele_count[polymorphic] / (2 * len(train_index)), source=selected["source"].to_numpy(dtype=str),
                        window_row=np.flatnonzero(polymorphic),
                        chromosome_row=window.chromosome_rows[polymorphic] if window.chromosome_rows is not None else None,
                        reliability=selected["reliability"].to_numpy(dtype=np.float64) if "reliability" in selected else np.ones(len(selected)),
                        concordance=selected["concordance"].to_numpy(dtype=np.float64) if "concordance" in selected else np.ones(len(selected)),
                        called_r2=selected["called_r2"].to_numpy(dtype=np.float64) if "called_r2" in selected else np.ones(len(selected)))
    train_phenotype, test_phenotype = residualize(dataset.expression[window.gene_row], dataset.covariates, train_index, test_index)
    samples = dataset.samples
    gene = dataset.gene_annotation[window.gene_id]
    train = TrainData(gene_id=window.gene_id, chrom=window.chrom, tss=window.tss, genotypes=train_genotypes, phenotype=train_phenotype, variants=variants,
                      superpopulation=samples["Superpopulation"].to_numpy()[train_index], population=samples["Population"].to_numpy()[train_index],
                      gene_start=gene["start"], gene_end=gene["end"], strand=gene["strand"],
                      exons=np.array(gene["exons"], dtype=np.int64).reshape(-1, 2), coding_exons=np.array(gene["coding_exons"], dtype=np.int64).reshape(-1, 2),
                      covariates=np.asarray(dataset.covariates[train_index], dtype=np.float64))
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
    pgsv: PanGenie SVs; snv_matched: as many panel SNVs/indels as panel SVs, matched to them (matched_small_variants);
    hgsvc3 / ont: long-read SVs only (HGSVC3 PanGenie lifted to GRCh38; 1KG-ONT SVIM-asm), and snv_hgsvc3 / snv_ont:
    panel SNVs/indels plus those SVs. The long-read rows exist only in the derived datasets that carry them.
    *_merged: the same SV sources after truvari collapse, one row per collapsed site; gatksv / snv_sv_cn: the GATK-SV 1kGP
    callset, whose multi-allelic CNV rows carry copies above the lowest copy number (so train_allele_frequency on them
    is half a mean copy offset, not an allele frequency); ctyper / snv_ctyper: Ctyper paralog-specific copy numbers (one row per
    gene-family copy, placed at its locus); hprc2 / snv_hprc2: HPRC release-2 pangenome SV genotypes. See SOURCE_SETS and
    JOINT_SOURCE_SETS."""
    panel = variants.source == "panel"
    small = panel & ~variants.is_sv
    pangenie_sv = (variants.source == "pangenie") & variants.is_sv
    if feature_set == "snv_matched":
        return matched_small_variants(variants, draw_key)
    if feature_set in SOURCE_SETS:
        return (variants.source == SOURCE_SETS[feature_set]) & variants.is_sv
    if feature_set in JOINT_SOURCE_SETS:
        return small | ((variants.source == JOINT_SOURCE_SETS[feature_set]) & variants.is_sv)
    return {"snv": small, "snv_sv": panel, "snv_pgsv": small | pangenie_sv, "sv": panel & variants.is_sv, "pgsv": pangenie_sv}[feature_set]


def subset(train: TrainData, test_genotypes: np.ndarray, feature_set: str, split_name: str):
    keep = feature_mask(train.variants, feature_set, f"{train.gene_id}/{split_name}")
    variants = Variants(**{field.name: None if getattr(train.variants, field.name) is None else getattr(train.variants, field.name)[keep]
                           for field in dataclasses.fields(Variants)})
    return dataclasses.replace(train, genotypes=train.genotypes[:, keep], variants=variants), test_genotypes[:, keep]


def sv_coefficients(train: TrainData, predictor, gene_id: str, split_name: str, feature_set: str):
    """The SV columns' effects on the genotype scale, for a linear predictor (intercept + (x - center) / scale @ beta).

    With these and the training means, SV j's contribution to any person's score is beta_j (x_j - mean_j), so the SV
    part of the score can be decomposed without refitting. Returns None for a predictor that is not linear."""
    coefficients = getattr(predictor, "coefficients", None)
    if coefficients is None or not train.variants.is_sv.any() or train.variants.window_row is None:
        return None
    scale = getattr(predictor, "scale", None)
    effects = np.asarray(coefficients, dtype=np.float64) / (np.asarray(scale, dtype=np.float64) if scale is not None else 1.0)
    columns = np.flatnonzero(train.variants.is_sv)
    return pd.DataFrame({"gene_id": gene_id, "split": split_name, "feature_set": feature_set, "window_row": train.variants.window_row[columns],
                         "source": train.variants.source[columns], "sv_type": train.variants.sv_type[columns],
                         "train_mean": train.genotypes[:, columns].mean(axis=0), "effect": effects[columns]})


def load_method(spec: str):
    """spec is '<path/to/file.py>:<callable>'."""
    path, name = spec.rsplit(":", 1)
    module_spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return getattr(module, name)


_WORKER = {}


def _init_worker(dataset_dir, method_spec, feature_sets, overlay_dir=None, rows_dirs=None, sample_subset=None, record_failures=False):
    _WORKER["dataset"] = Dataset(dataset_dir, overlay_dir, rows_dirs, sample_subset)
    _WORKER["fit"] = load_method(method_spec)
    _WORKER["feature_sets"] = feature_sets
    _WORKER["record_failures"] = record_failures


def _without_structural_variants(train: TrainData, test_genotypes: np.ndarray):
    """Test genotypes with every SV column set to its training mean, so a predictor's SV contribution drops out."""
    masked = test_genotypes.copy()
    masked[:, train.variants.is_sv] = train.genotypes[:, train.variants.is_sv].mean(axis=0)
    return masked


def _call_predict(predictor, genotypes, covariates):
    """predictor.predict(genotypes), passing the samples' covariates too when the predictor accepts them."""
    if "covariates" in inspect.signature(predictor.predict).parameters:
        return np.asarray(predictor.predict(genotypes, covariates=covariates), dtype=np.float64)
    return np.asarray(predictor.predict(genotypes), dtype=np.float64)


def predict_for_truth(predictor, train: TrainData, test_genotypes: np.ndarray, test_covariates: np.ndarray, raw=None):
    """The held-out predictions scored against the truth, full and with the SV columns held at their training means.

    The truth is the test expression minus [1, C_test] a, with a the training OLS coefficients of expression on
    [1, C] (residualize). Under the fixed-effects model y = [1, C] g + X b + e, a estimates g + B b, where B is the
    training OLS of the genotypes on [1, C], so the truth's genetic part is (X_test - [1, C_test] B) b. The harness
    therefore treats every method's score the way it treats the truth: it fits the score on [1, C] over the training
    samples and removes [1, C_test] times those coefficients from the test score. For a linear score X b this gives
    exactly (X_test - [1, C_test] B) b, whatever b is; for a score already orthogonal to [1, C] in training it changes
    nothing. One rule for every method (review-mathbugs C2).

    With raw a dict, the method's unadjusted scores are also returned in it (train, test, and both with the SV columns
    held at their training means), so a fit can be re-scored under any other truth definition without refitting."""
    design_train = np.column_stack([np.ones(train.genotypes.shape[0]), train.covariates])
    design_test = np.column_stack([np.ones(test_genotypes.shape[0]), test_covariates])

    def adjusted(train_genotypes, genotypes, label):
        train_score, test_score = _call_predict(predictor, train_genotypes, train.covariates), _call_predict(predictor, genotypes, test_covariates)
        if raw is not None:
            raw[label] = (train_score, test_score)
        if np.ptp(train_score) == 0:
            # A constant's fit on [1, C] is the constant on the intercept alone; lstsq would leave float rounding noise,
            # which scores as a random direction instead of as no prediction.
            return test_score - train_score[0]
        coefficients, *_ = np.linalg.lstsq(design_train, train_score, rcond=None)
        return test_score - design_test @ coefficients

    prediction = adjusted(train.genotypes, test_genotypes, "full")
    if not train.variants.is_sv.any():
        if raw is not None:
            raw["without_sv"] = raw["full"]
        return prediction, prediction
    return prediction, adjusted(_without_structural_variants(train, train.genotypes), _without_structural_variants(train, test_genotypes), "without_sv")


def _train_index(dataset, split_name):
    return np.array([dataset.sample_index[sample] for sample in dataset.splits[split_name]["train"]])


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
            try:
                predictor = fit(train)
                raw = {}
                prediction, without_sv = predict_for_truth(predictor, train, test_genotypes, dataset.covariates[test_index], raw)
                coefficients, status = sv_coefficients(train, predictor, window.gene_id, split_name, feature_set), "ok"
            except Exception as error:
                # A failed fit is never replaced by a stand-in predictor. By default it stops the run; with
                # --record-failures it is kept as a failure: NaN predictions (unscored) and the error in the log.
                if not _WORKER["record_failures"]:
                    raise
                prediction = without_sv = np.full(len(test_index), np.nan)
                raw = None
                coefficients, status = None, f"failed: {type(error).__name__}: {str(error)[:300]}"
            seconds = time.process_time() - started
            results.append((gene_row, split_name, feature_set, test_index, prediction, without_sv, test_phenotype, train.genotypes.shape[1],
                            int(train.variants.is_sv.sum()), seconds, coefficients, status, raw, _train_index(dataset, split_name)))
    return results


def _per_gene_results(dataset_dir, method_spec, feature_sets, gene_rows, split_names, workers, overlay_dir=None, rows_dirs=None, sample_subset=None,
                      record_failures=False):
    from multiprocessing import get_context

    with get_context("fork").Pool(workers, initializer=_init_worker, initargs=(dataset_dir, method_spec, feature_sets, overlay_dir, rows_dirs, sample_subset, record_failures)) as pool:
        yield from pool.imap_unordered(_run_gene, [(row, split_names) for row in gene_rows], chunksize=1)


class _LazyTrains(Sequence):
    """The training data of every selected gene for one split and feature set, built on access so the batch never holds
    every gene's genotypes at once. Test genotypes and phenotypes are not reachable from it."""

    def __init__(self, dataset, gene_rows, split_name, feature_set):
        sealed = dataset.sealed_genes()
        if any(dataset.genes.iloc[row]["gene_id"] in sealed for row in gene_rows):
            raise ValueError("a batch may not contain a sealed confirmation gene")
        self.dataset, self.gene_rows, self.split_name, self.feature_set = dataset, gene_rows, split_name, feature_set

    def __len__(self):
        return len(self.gene_rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        train, _, _, _ = self._task(index)
        return train

    def _task(self, index):
        window = load_gene_window(self.dataset, self.gene_rows[index])
        train, test_genotypes, test_phenotype, test_index = build_gene_task(self.dataset, window, self.dataset.splits[self.split_name])
        train, test_genotypes = subset(train, test_genotypes, self.feature_set, self.split_name)
        return train, test_genotypes, test_phenotype, test_index


class _LazyViews(Mapping):
    """Every requested (gene_id, split, feature_set) view's training data, built on access; test data unreachable.

    Views are keyed in gene, then split, then feature-set order, and the latest gene's window stays loaded, so a method
    that walks one gene's views together reads its window once. Variants.chromosome_row (with source) identifies a
    column across overlapping windows, for sharing between genes."""

    def __init__(self, dataset, gene_rows, split_names, feature_sets):
        sealed = dataset.sealed_genes()
        if any(dataset.genes.iloc[row]["gene_id"] in sealed for row in gene_rows):
            raise ValueError("views may not contain a sealed confirmation gene")
        self.dataset = dataset
        self.row_of_gene = {dataset.genes.iloc[row]["gene_id"]: row for row in gene_rows}
        self.keys_in_order = [(dataset.genes.iloc[row]["gene_id"], split, feature_set) for row in gene_rows for split in split_names for feature_set in feature_sets]
        self.key_set = set(self.keys_in_order)
        self._window = None

    def __len__(self):
        return len(self.keys_in_order)

    def __iter__(self):
        return iter(self.keys_in_order)

    def __contains__(self, key):
        return key in self.key_set

    def __getitem__(self, key):
        return self._task(key)[0]

    def _task(self, key):
        if key not in self.key_set:
            raise KeyError(key)
        gene_id, split_name, feature_set = key
        if self._window is None or self._window.gene_id != gene_id:
            self._window = load_gene_window(self.dataset, self.row_of_gene[gene_id])
        train, test_genotypes, test_phenotype, test_index = build_gene_task(self.dataset, self._window, self.dataset.splits[split_name])
        train, test_genotypes = subset(train, test_genotypes, feature_set, split_name)
        return train, test_genotypes, test_phenotype, test_index


def _run_views(dataset, fit_views, gene_rows, split_names, feature_sets):
    """fit_views(views) returns a mapping {key: predictor}, or yields (key, predictor) pairs so predictors needn't all be
    held at once. Every requested key must come back exactly once."""
    views = _LazyViews(dataset, gene_rows, split_names, feature_sets)
    started = time.process_time()
    returned = fit_views(views)
    pairs = returned.items() if isinstance(returned, Mapping) else returned
    seen = set()
    for key, predictor in pairs:
        if key not in views or key in seen:
            raise ValueError(f"fit_views returned an unrequested or repeated view {key}")
        seen.add(key)
        seconds = (time.process_time() - started) / len(views)
        train, test_genotypes, test_phenotype, test_index = views._task(key)
        raw = {}
        prediction, without_sv = predict_for_truth(predictor, train, test_genotypes, dataset.covariates[test_index], raw)
        yield (views.row_of_gene[key[0]], key[1], key[2], test_index, prediction, without_sv, test_phenotype, train.genotypes.shape[1],
               int(train.variants.is_sv.sum()), seconds, sv_coefficients(train, predictor, key[0], key[1], key[2]), "ok", raw, _train_index(dataset, key[1]))
    if len(seen) != len(views):
        raise ValueError(f"fit_views returned {len(seen)} of {len(views)} requested views")


def _run_batch(dataset, fit_batch, gene_rows, split_names, feature_sets):
    for split_name in split_names:
        for feature_set in feature_sets:
            trains = _LazyTrains(dataset, gene_rows, split_name, feature_set)
            started = time.process_time()
            predictors = list(fit_batch(trains))
            seconds = (time.process_time() - started) / len(gene_rows)
            if len(predictors) != len(gene_rows):
                raise ValueError(f"fit_batch returned {len(predictors)} predictors for {len(gene_rows)} genes")
            for index, (gene_row, predictor) in enumerate(zip(gene_rows, predictors)):
                train, test_genotypes, test_phenotype, test_index = trains._task(index)
                raw = {}
                prediction, without_sv = predict_for_truth(predictor, train, test_genotypes, dataset.covariates[test_index], raw)
                yield (gene_row, split_name, feature_set, test_index, prediction, without_sv, test_phenotype, train.genotypes.shape[1],
                       int(train.variants.is_sv.sum()), seconds, sv_coefficients(train, predictor, train.gene_id, split_name, feature_set), "ok", raw,
                       _train_index(dataset, split_name))


def source_provenance():
    """What source this run is: the git commit of the checkout the harness lives in, or, when there is no checkout, a
    digest of the source files themselves.

    Resolved before any gene is fitted. Asking git after the fits meant a run from a source archive (no .git), or one
    where git is missing or fails, threw away every finished fit at the last step, hours of CPU for a provenance line.
    A run from an archive still records exactly what it ran: the sha256 over benchmarks/ and sv_pgs/, each file's path
    with its own digest, in path order."""
    directory = pathlib.Path(__file__).resolve().parent
    try:
        finished = subprocess.run(["git", "rev-parse", "HEAD"], cwd=directory, capture_output=True, text=True, check=True)
        return {"harness_commit": finished.stdout.strip(), "harness_source": "git", "harness_source_sha256": None}
    except (OSError, subprocess.SubprocessError):
        pass
    root = directory.parent.parent
    digest = hashlib.sha256()
    for package in ("benchmarks", "sv_pgs"):
        for path in sorted((root / package).rglob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return {"harness_commit": None, "harness_source": "no-git", "harness_source_sha256": digest.hexdigest()}


def run(dataset_dir, method_spec, method_name, design, chromosomes, out_dir, workers, feature_sets=FEATURE_SETS, gene_prefix=None, gene_list=None,
        confirmation=False, contract="gene", gene_ranks=None, overlay_dir=None, split_subset=None, note=None, rows_dirs=None, sample_subset=None,
        record_failures=False):
    """Out-of-fold predictions of one method for every gene on the chromosomes, under one split design.

    contract "gene": the method is fit(train) -> predictor, called per gene, split and feature set in worker processes.
    contract "batch": the method is fit_batch(trains) -> list of predictors, called once per split and feature set
    with a lazy sequence of every selected gene's TrainData, so it can pool hyperparameters across genes. It never
    sees a test phenotype, and it owns its own parallelism (RUNQ_CORES)."""
    # Before the fits: a provenance failure must never cost a finished run its results.
    provenance = source_provenance()
    method_file = pathlib.Path(method_spec.rsplit(":", 1)[0])
    method_sha256 = hashlib.sha256(method_file.read_bytes()).hexdigest()
    dataset = Dataset(dataset_dir, overlay_dir, rows_dirs, sample_subset)
    split_names = [name for name in dataset.splits if name.startswith(design + "/")]
    if split_subset is not None:
        unknown = set(split_subset) - set(split_names)
        if unknown:
            raise ValueError(f"splits not in design {design}: {sorted(unknown)}")
        split_names = [name for name in split_names if name in set(split_subset)]
    if gene_ranks is not None and gene_list is None:
        raise ValueError("gene ranks need a gene list")
    gene_rows = dataset.gene_rows(chromosomes, gene_prefix, gene_list, confirmation, gene_ranks)
    sample_count = len(dataset.samples)
    predictions = {feature_set: np.full((len(gene_rows), sample_count), np.nan, dtype=np.float32) for feature_set in feature_sets}
    predictions_without_sv = {feature_set: np.full((len(gene_rows), sample_count), np.nan, dtype=np.float32) for feature_set in feature_sets}
    # The raw (unadjusted) scores of every sample, per split: genes x splits x samples, so a fit can be re-scored under
    # another truth definition without refitting. NaN where a sample was not scored in that split (or the fit failed).
    raw_scores = {(feature_set, kind): np.full((len(gene_rows), len(split_names), sample_count), np.nan, dtype=np.float32)
                  for feature_set in feature_sets for kind in ("full", "without_sv")}
    split_position = {name: position for position, name in enumerate(split_names)}
    truth = np.full((len(gene_rows), sample_count), np.nan, dtype=np.float32)
    log = []
    position_of_row = {row: position for position, row in enumerate(gene_rows)}
    if contract == "batch":
        results = _run_batch(dataset, load_method(method_spec), gene_rows, split_names, feature_sets)
    elif contract == "views":
        results = _run_views(dataset, load_method(method_spec), gene_rows, split_names, feature_sets)
    else:
        results = (result for chunk in _per_gene_results(dataset_dir, method_spec, feature_sets, gene_rows, split_names, workers, overlay_dir, rows_dirs, sample_subset,
                                                            record_failures) for result in chunk)
    coefficient_tables = []
    out = pathlib.Path(out_dir) / method_name / design
    out.mkdir(parents=True, exist_ok=True)
    tag = ("_".join(chromosomes) + (f".ranks{gene_ranks[0]}-{gene_ranks[1]}" if gene_ranks is not None else "")
           + ("." + "+".join(name.split("/")[1] for name in split_names) if split_subset is not None else ""))
    # One line per fit as it lands (the tables below are written once every fit is in): a run's progress and each
    # fit's CPU time are readable while it runs, and a run that dies leaves its finished fits' record.
    progress = (out / f"{tag}.progress.jsonl").open("w")
    for (gene_row, split_name, feature_set, test_index, prediction, without_sv, test_phenotype, variant_count, sv_count, seconds, coefficients, status,
         raw, train_index) in results:
        if coefficients is not None:
            coefficient_tables.append(coefficients)
        position = position_of_row[gene_row]
        predictions[feature_set][position, test_index] = prediction
        predictions_without_sv[feature_set][position, test_index] = without_sv
        if raw is not None:
            for kind, (train_score, test_score) in raw.items():
                raw_scores[(feature_set, kind)][position, split_position[split_name], train_index] = train_score
                raw_scores[(feature_set, kind)][position, split_position[split_name], test_index] = test_score
        truth[position, test_index] = test_phenotype
        log.append((dataset.genes.iloc[gene_row]["gene_id"], split_name, feature_set, variant_count, sv_count, seconds, status,
                    bool(np.isfinite(prediction).all() and np.ptp(prediction) == 0)))
        progress.write(json.dumps({
            "gene_id": log[-1][0], "split": split_name, "feature_set": feature_set, "variants": int(variant_count), "sv_variants": int(sv_count),
            "cpu_seconds": float(seconds), "status": status, "finished": len(log),
        }) + "\n")
        progress.flush()
    progress.close()
    (out / f"{tag}.run.json").write_text(json.dumps({
        "method": method_spec, "method_sha256": method_sha256, **provenance,
        "design": design, "chromosomes": list(chromosomes), "feature_sets": list(feature_sets), "gene_prefix": gene_prefix,
        "gene_list": str(gene_list) if gene_list is not None else None, "gene_ranks": list(gene_ranks) if gene_ranks is not None else None,
        "confirmation": confirmation,
        "sealed_genes_sha256": hashlib.sha256((dataset.directory / SEALED_GENES).read_bytes()).hexdigest() if (dataset.directory / SEALED_GENES).exists() else None,
        "gene_list_sha256": hashlib.sha256(pathlib.Path(gene_list).read_bytes()).hexdigest() if gene_list is not None else None,
        "genes": len(gene_rows), "contract": contract, "splits": split_names, "prediction_rule": PREDICTION_RULE,
        "record_failures": record_failures, "failed_fits": sum(entry[6] != "ok" for entry in log),
        "constant_predictions": sum(bool(entry[7]) for entry in log),
        "rows_dirs": [{"dir": str(path), "provenance_sha256": hashlib.sha256((path / "PROVENANCE.json").read_bytes()).hexdigest()
                       if (path / "PROVENANCE.json").exists() else None} for path in dataset.rows_dirs],
        "sample_subset": str(sample_subset) if sample_subset is not None else None,
        "sample_subset_sha256": hashlib.sha256(pathlib.Path(sample_subset).read_bytes()).hexdigest() if sample_subset is not None else None,
        "note": json.loads(pathlib.Path(note).read_text()) if note is not None else None, "overlay": str(overlay_dir) if overlay_dir is not None else None,
        "overlay_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(pathlib.Path(overlay_dir).glob("*.svimp.npz"))
                           if path.name.split(".")[0] in chromosomes} if overlay_dir is not None else None, "splits_sha256": (dataset.directory / "splits.sha256").read_text().strip()}, indent=1))
    for feature_set in feature_sets:
        np.save(out / f"{tag}.{feature_set}.predictions.npy", predictions[feature_set])
        np.save(out / f"{tag}.{feature_set}.predictions_without_sv.npy", predictions_without_sv[feature_set])
        np.save(out / f"{tag}.{feature_set}.raw_scores.npy", raw_scores[(feature_set, "full")])
        np.save(out / f"{tag}.{feature_set}.raw_scores_without_sv.npy", raw_scores[(feature_set, "without_sv")])
    (out / f"{tag}.raw_splits.json").write_text(json.dumps(split_names))
    np.save(out / f"{tag}.truth.npy", truth)
    if coefficient_tables:
        pd.concat(coefficient_tables, ignore_index=True).to_csv(out / f"{tag}.sv_coefficients.tsv.gz", sep="\t", index=False)
    dataset.genes.iloc[gene_rows].to_csv(out / f"{tag}.genes.tsv", sep="\t", index=False)
    pd.DataFrame(log, columns=["gene_id", "split", "feature_set", "variants", "sv_variants", "cpu_seconds", "status", "constant_prediction"]).to_csv(
        out / f"{tag}.log.tsv", sep="\t", index=False)


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
    parser.add_argument("--confirmation", action="store_true", help="score only the sealed confirmation genes (only when the lead calls it)")
    parser.add_argument("--contract", choices=["gene", "batch", "views"], default="gene",
                        help="gene: fit(train); batch: fit_batch(trains) once per split; views: fit_views(views) once over every view")
    parser.add_argument("--gene-ranks", nargs=2, type=int, metavar=("START", "STOP"), help="with --genes, only the list's rows START..STOP-1")
    parser.add_argument("--overlay", help="directory of <chrom>.svimp.npz imputed SV dosages (feature sets svimp, snv_svimp)")
    parser.add_argument("--splits", nargs="+", help="only these splits of the design (e.g. loso/AFR), for per-split checkpoints")
    parser.add_argument("--note", help="a JSON file recorded verbatim in run.json (arm label, test status, rulings)")
    parser.add_argument("--rows", action="append", metavar="NAME=DIR", help="an extra-rows overlay (repeatable; merged in the given order)")
    parser.add_argument("--sample-subset", help="a file of sample ids, one per line: every split is cut to these people")
    parser.add_argument("--record-failures", action="store_true", help="keep going when a fit raises: the fit is recorded as failed (NaN, never a stand-in)")
    arguments = parser.parse_args()
    run(arguments.dataset, arguments.method, arguments.name, arguments.design, arguments.chromosomes, arguments.out, arguments.workers,
        tuple(arguments.feature_sets), arguments.gene_prefix, arguments.genes, arguments.confirmation, arguments.contract,
        tuple(arguments.gene_ranks) if arguments.gene_ranks else None, arguments.overlay, arguments.splits, arguments.note,
        [value.split("=", 1)[1] for value in arguments.rows] if arguments.rows else None, arguments.sample_subset, arguments.record_failures)
