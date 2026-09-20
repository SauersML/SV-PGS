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
(Horvitz & Thompson 1952, JASA 47:663). Genes of C that were already scored (the development genes) are counted like
any other gene: y_g is a measurement, and the certainty stratum has no selection weight.

Standard error. With z_g = y_g for g outside C and 0 inside it, the random term is (|U| / n) sum_{g in P} z_g, the
expansion estimator of a total under simple random sampling without replacement. Its design variance is estimated
without bias by |U|^2 (1 - n/|U|) s_z^2 / n, with s_z^2 the sample variance of z over P (Cochran 1977, Sampling
Techniques, 3rd ed., Theorem 2.2). The certainty stratum is a census and has no design variance. Measurement noise in
y (the sampling of test people, shared by every gene) enters through person-bootstrap replicates of y when given: the
variance of the replicate totals is added. Because s_z^2 already carries the independent part of the random-stratum
genes' own noise, the sum is conservative by at most that part.
"""
import numpy as np
import pandas as pd


def horvitz_thompson_total(genes: pd.DataFrame, population_size: int, random_sample_size: int, value: str = "y",
                           replicates: np.ndarray | None = None):
    """genes: one row per scored gene with columns gene_id, <value>, targeted (in C) and random (in P); every gene of P
    must be present. replicates: optional (B, rows) person-bootstrap replicates of <value>, aligned with the rows.

    Returns (total, standard error, kind)."""
    targeted = genes["targeted"].to_numpy(dtype=bool)
    random = genes["random"].to_numpy(dtype=bool)
    if not (targeted | random).all():
        raise ValueError("every row must belong to the targeted set, the random sample, or both")
    if int(random.sum()) != random_sample_size:
        raise ValueError(f"the random sample has {int(random.sum())} scored genes, not {random_sample_size}")
    values = genes[value].to_numpy(dtype=np.float64)
    weight = np.where(targeted, 1.0, population_size / random_sample_size)
    total = float(np.sum(weight * values))
    outside = np.where(targeted[random], 0.0, values[random])
    design_variance = 0.0
    if random_sample_size > 1:
        design_variance = (population_size ** 2 * (1.0 - random_sample_size / population_size)
                           * float(np.var(outside, ddof=1)) / random_sample_size)
    if replicates is None:
        return total, float(np.sqrt(design_variance)), "design (simple random sampling, finite-population corrected)"
    replicates = np.asarray(replicates, dtype=np.float64)
    if replicates.shape[1] != len(genes):
        raise ValueError("replicates must have one column per row of genes")
    measurement_variance = float(np.var(replicates @ weight, ddof=1))
    return total, float(np.sqrt(design_variance + measurement_variance)), "design + person bootstrap"
