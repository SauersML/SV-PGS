"""Genome-wide totals from bench-real's two gene samples, by Horvitz-Thompson weighting.

Two samples of the population U of benchmark genes are scored:
  targeted  the top of the genotype-only ranked list (sv-screen), a fixed set C; every gene in it is observed, so it is
            a certainty stratum with weight 1;
  random    the first n genes of the sealed gene order, a uniform random sample P of size n from all |U| genes, so
            each gene enters P with probability n / |U|.
The total of a per-gene quantity y over U splits into the total over C, observed exactly, and the total over U \\ C,
estimated from the genes of P outside C with weight |U| / n:

    T = sum_{g in C} y_g + (|U| / n) sum_{g in P \\ C} y_g,

which is design-unbiased for sum_{g in U} y_g whatever C is, as long as C does not depend on which genes P drew
(Horvitz & Thompson 1952, JASA 47:663). Its standard error is the delete-one-chromosome jackknife of T, which covers
both the measurement noise in y and the random sample. Genes of C that were already scored (the development genes)
are counted like any other gene: y_g is a measurement, and the certainty stratum has no selection weight.
"""
import numpy as np
import pandas as pd


def horvitz_thompson_total(genes: pd.DataFrame, population_size: int, random_sample_size: int, value: str = "y"):
    """genes: one row per scored gene with columns gene_id, chrom, <value>, targeted (in C) and random (in P).

    Returns (total, standard error, jackknife kind)."""
    if not (genes["targeted"] | genes["random"]).all():
        raise ValueError("every row must belong to the targeted set, the random sample, or both")
    weight = np.where(genes["targeted"], 1.0, population_size / random_sample_size)
    contribution = weight * genes[value].to_numpy(dtype=np.float64)
    total = float(contribution.sum())
    chromosomes = genes["chrom"].to_numpy()
    labels = np.unique(chromosomes)
    if len(labels) < 2:
        return total, float("nan"), "none"
    leave_out = np.array([contribution[chromosomes != label].sum() for label in labels])
    count = len(labels)
    # The jackknife of a sum over chromosomes: each leave-out total is rescaled to the full genome by count / (count - 1).
    scaled = leave_out * count / (count - 1)
    error = np.sqrt((count - 1) / count * ((scaled - scaled.mean()) ** 2).sum())
    return total, float(error), "chromosome jackknife"
