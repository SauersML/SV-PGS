"""bench-sim submission: SBayesRC (Zheng et al. 2024, Nature Genetics), the biobank-scale summary-statistic method with
functional annotations, run through its own R package (github.com/zhilizheng/SBayesRC v0.2.6) at its defaults.

SBayesRC fits a five-component normal mixture (variances gamma * sigma^2, gamma = 0, 1e-3, 1e-2, 1e-1, 1) to the
marginal effects through a low-rank LD model: per LD block the correlation matrix R = U diag(lambda) U' keeps the
leading eigenvectors that explain 99.5% of its variance, and the likelihood is written on the rotated statistics
U'b (the block's "eigen" LD). With annotations each variant's mixture probabilities follow its annotation vector
through a hierarchical (probit-type, one per component step) model learned jointly in the Gibbs sampler; without them
every variant shares one set of probabilities (SBayesR on the low-rank LD). The package's automatic tuning of the
eigen-variance cutoff (0.995 / 0.99 / 0.95 / 0.9 on a pseudo-validation split of the summary statistics) runs as
published.

Inputs built here, in the package's own file formats (LDstep1-2 run GCTB on PLINK files of the reference people;
the eigen step LDstep3, the merge LDstep4 and sbayesrc() are the package's):

- Variants: every measured record of minor allele frequency at least 1% of every class (SNV, INDEL, TR, SV), as the
  published method uses its densest variant set (7.3M imputed common variants in the paper, rather than HapMap3);
  the LD blocks are 4 cM windows of genetic position, the width of the package's default block file (ref4cM_v37).
- Marginal statistics: the GWAS with covariates, by Frisch-Waugh-Lovell b_j = <P x_j, P y>/|P x_j|^2 with P the
  projection off [1, covariates], and se_j with n - k - 1 degrees of freedom.
- LD: the correlations of the same projected dosages of the training people, in-sample, so the LD matches the
  statistics it is paired with (see ldpred2_auto.py: the cohort's five ancestry groups make raw correlations carry
  the structure the PCs remove from the phenotype).
- Annotations (sbayesrc.py): an intercept, in_gene, in_exon, log_tss_distance, in_repeat, log_sv_length and the
  INDEL / TR / SV class indicators, the public per-record annotations the benchmark provides. sbayesrc_annotfree.py
  runs the same fit with no annotation file.

Quantitative traits (a binary one is fitted as its 0/1 values). Needs R with SBayesRC (module R/4.4.2-openblas-rocky8;
R_LIBS_USER listing the library SBayesRC is installed in, then the compete library).
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import tempfile

import numpy as np

MINOR_ALLELE_FLOOR = 0.01
"""The published SBayesRC variant set's minor allele frequency floor (common imputed variants, MAF > 1%)."""
BLOCK_CM = 4.0
"""Width of the package's default LD blocks (ref4cM_v37.pos: 4 cM windows)."""
ANNOTATIONS = ("in_gene", "in_exon", "log_tss_distance", "in_repeat", "log_sv_length")
CLASSES = ("INDEL", "TR", "SV")
"""Class indicators (codes 1, 2, 3); SNVs are the intercept's baseline."""
CHUNK = 4096
_SCRIPT = pathlib.Path(__file__).with_name("sbayesrc.R")


def _dosages(source, rows: np.ndarray) -> np.ndarray:
    return np.asarray(source.codes(rows), dtype=np.float32) / np.float32(127.0)


class Model:
    def __init__(self, rows: np.ndarray, means: np.ndarray, beta: np.ndarray, structural: np.ndarray) -> None:
        self.rows, self.means, self.beta, self.structural = rows, means, beta, structural

    def score(self, test):
        total = np.zeros(test.covariates.shape[0])
        structural = np.zeros_like(total)
        for start in range(0, self.rows.shape[0], CHUNK):
            block = slice(start, start + CHUNK)
            dosages = _dosages(test, self.rows[block]).astype(np.float64) - self.means[block][:, None]
            total += dosages.T @ self.beta[block]
            structural += dosages.T @ (self.beta[block] * self.structural[block])
        return {"total": total, "structural": structural}


def common_rows(train) -> np.ndarray:
    """Every record of minor allele frequency >= 1%, ordered by genetic position."""
    frequency = np.empty(train.n_variants)
    for start in range(0, train.n_variants, CHUNK):
        rows = np.arange(start, min(start + CHUNK, train.n_variants))
        frequency[rows] = _dosages(train, rows).mean(axis=1) / 2.0
    rows = np.flatnonzero(np.minimum(frequency, 1.0 - frequency) >= MINOR_ALLELE_FLOOR)
    cm = np.asarray(train.variants["cm"], dtype=np.float64)[rows]
    return rows[np.argsort(cm, kind="stable")]


def write_inputs(train, rows: np.ndarray, folder: pathlib.Path, annotated: bool) -> np.ndarray:
    """The GWAS (.ma), the per-block full LD in GCTB's .ldm.full format, ldm.info, and the annotation file.

    Returns the dosage means (the score centres)."""
    count = train.n_samples
    design = np.column_stack([np.ones(count), np.asarray(train.covariates, dtype=np.float64)])
    basis, _ = np.linalg.qr(design)
    basis32 = basis.astype(np.float32)
    phenotype = np.asarray(train.phenotype, dtype=np.float64)
    residual = phenotype - basis @ (basis.T @ phenotype)
    degrees = count - basis.shape[1] - 1
    variants = train.variants
    cm = np.asarray(variants["cm"], dtype=np.float64)[rows]
    position = np.asarray(variants["pos"])[rows]
    block_of = np.floor((cm - cm[0]) / BLOCK_CM).astype(np.int64)
    ld = folder / "ld"
    ld.mkdir()
    means = np.empty(rows.shape[0])
    names = np.array([f"v{row}" for row in rows])
    gwas = []
    info = []
    for number, block in enumerate(np.unique(block_of), start=1):
        members = np.flatnonzero(block_of == block)
        dosages = np.vstack([_dosages(train, rows[members[start : start + CHUNK]]) for start in range(0, members.shape[0], CHUNK)])
        means[members] = dosages.mean(axis=1, dtype=np.float64)
        frequency = means[members] / 2.0
        projected = dosages - (dosages @ basis32) @ basis32.T
        del dosages
        squares = np.einsum("ij,ij->i", projected, projected, dtype=np.float64)
        cross = projected.astype(np.float64) @ residual
        beta = cross / squares
        se = np.sqrt((residual @ residual - beta * cross) / (degrees * squares))
        scaled = projected / np.sqrt(squares).astype(np.float32)[:, None]
        del projected
        correlation = scaled @ scaled.T
        del scaled
        np.fill_diagonal(correlation, 1.0)
        correlation.astype("<f4").tofile(ld / f"b{number}.ldm.full.bin")
        with open(ld / f"b{number}.ldm.full.info", "w") as handle:
            handle.write("Chrom\tID\tGenPos\tPhysPos\tA1\tA2\tA1Freq\tN\n")
            for index, member in enumerate(members):
                handle.write(f"22\t{names[member]}\t{cm[member]:.6f}\t{position[member]}\tA\tC\t{frequency[index]:.8f}\t{count}\n")
        for index, member in enumerate(members):
            gwas.append(f"{names[member]}\tA\tC\t{frequency[index]:.8f}\t{beta[index]:.10g}\t{se[index]:.10g}\tNA\t{count}\n")
        info.append(f"{number}\t22\t{members[0]}\t{names[members[0]]}\t{members[-1]}\t{names[members[-1]]}\t{members.shape[0]}\n")
    with open(ld / "ldm.info", "w") as handle:
        handle.write("Block\tChrom\tStartSnpIdx\tStartSnpID\tEndSnpIdx\tEndSnpID\tNumSnps\n")
        handle.writelines(info)
    with open(folder / "gwas.ma", "w") as handle:
        handle.write("SNP\tA1\tA2\tfreq\tb\tse\tp\tN\n")
        handle.writelines(gwas)
    if annotated:
        cls = np.asarray(variants["cls"])[rows]
        columns = [np.asarray(variants[name], dtype=np.float64)[rows] for name in ANNOTATIONS]
        columns += [(cls == code).astype(np.float64) for code, _ in enumerate(CLASSES, start=1)]
        table = np.column_stack(columns)
        with open(folder / "annot.txt", "w") as handle:
            handle.write("\t".join(("SNP", "Intercept", *ANNOTATIONS, *CLASSES)) + "\n")
            for name, values in zip(names, table):
                handle.write(name + "\t1\t" + "\t".join(f"{value:.6g}" for value in values) + "\n")
    return means


def fit_sbayesrc(train, annotated: bool) -> Model:
    rows = common_rows(train)
    structural = (np.asarray(train.variants["cls"])[rows] >= 2).astype(np.float64)
    with tempfile.TemporaryDirectory(prefix="sbayesrc_", dir=os.environ.get("TMPDIR"),
                                     ignore_cleanup_errors=True) as directory:
        folder = pathlib.Path(directory)
        means = write_inputs(train, rows, folder, annotated)
        threads = {name: str(int(train.cores)) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
        subprocess.run(["Rscript", str(_SCRIPT), str(folder), "annot.txt" if annotated else ""], check=True,
                       env=os.environ | threads)
        effects = {}
        with open(folder / "sbrc.txt") as handle:
            next(handle)
            for line in handle:
                fields = line.split("\t")
                effects[fields[0]] = float(fields[2])
    # the output's A1 is the ALT allele ("A") the .ma declared, so BETA is per ALT dosage
    beta = np.array([effects[f"v{row}"] for row in rows])
    return Model(rows, means, beta, structural)


def fit(train) -> Model:
    return fit_sbayesrc(train, annotated=True)
