"""The measurement model: from each stored imputed column to its true genotype.

Imputed SV and TR dosages behave as confident draws, not calibrated posterior
means: the slope of the true genotype on the dosage is about sqrt(r^2), and the
dosage keeps nearly the genotype's variance (bench-sim v7, public 1kGP haplotypes
re-imputed with Beagle 5.5 [semi-real]). Such a column is biased in two ways,
and this module estimates both corrections from calibration pairs: samples that
carry both a stored column D and a truth T of the same genotype, whose error is
independent of D. Nothing here is a shipped table. Every quantity is computed by
the pipeline from its own calibration pairs: inside the AoU workspace from its
truth samples, and on public benchmarks from their truth.

1. Recalibration scales. kappa_j = Cov(G, D_j) / Var(D_j), so that
   D*_j = mu_j + kappa_j (D_j - mu_j) satisfies Cov(G, D*) = Var(D*). For a
   truth with E[T | G] = G and error independent of D given G,
   Cov(T, D) = Cov(G, D), so one truth suffices. Each record's estimate is noisy
   (a rare record's few pairs can even fit exactly by chance), so records are
   pooled within a stratum: kappa(x) is a least-squares fit, linear in a
   caller-given design, over all the stratum's pairs, and each record's own
   slope is shrunk toward it by a normal-normal empirical Bayes whose
   between-record variance is chosen by marginal likelihood (0 when the records
   agree).
2. Conditional moments. With D* calibrated, the residual variance
   v_j = E[(G - D*_j)^2] = Var(G_j) - Var(D*_j) = V_j (lambda_j - kappa_j^2),
   with lambda = Var(G) / Var(D) fitted like kappa(x), enters the predictive
   variance, and r^2_j = Var(D*_j) / (Var(D*_j) + v_j) = kappa_j^2 / lambda_j
   gives the prior offset log r^2_j (coefficient 1,
   docs/design/math/scale_model.md section 1).
3. The leakage map, the "A-map" (scale_model.md sections 2-3). A draw-type
   column leaves information about G_k in nearby columns, so in a joint fit part
   of an SV's effect moves onto its tag SNPs, and a stacked truth half no longer
   shares coefficients with the imputed half. The linear predictor of G_k from
   the whole LD block,
       Xtilde_k = D*_k + sum_j C_jk (D*_j - mu_j),
   removes it. C_k is the Bayesian ridge fit of the calibrated residual
   T_k - D*_k on the block's standardized columns. One ridge ratio per block is
   chosen by marginal likelihood, and each target has its own noise variance.
   A ratio of 0 means the data show no leakage and gives back D*.
4. The engine's products for the mapped columns. Xtilde = X A with
   A = I + E, where E holds C in the target columns, so a block's centred Gram
   becomes A' G A and its cross-products A' X'y.

The calibration pairs are an explicit input (``CalibrationPairs``, keyed by
typed research IDs). They enter as per-record sufficient statistics, which the
store step accumulates in its one pass, plus the pair-level data of the LD blocks
to be mapped, so no [records, samples] array is ever needed. In the AoU workspace
they come from a truth source the user supplies: the long-read panel members'
hard calls are not part of the imputation deliverables. Where calibration pairs
are missing, for a record or for everything, ``fit_measurement_model`` degrades
loudly, never silently. Those records get no recalibration and no leakage
correction, and their reliability offsets come from the imputation's own reported
r^2, which the caller must then pass. The model's certificate, which the fit
certificate carries, records each of these facts with its record count, and they
are logged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields

import numpy as np

from sv_pgs._typing import F64Array, I64Array, NDArray
from sv_pgs.progress import log
from sv_pgs.sample_ids import ResearchId


@dataclass(frozen=True)
class CalibrationMoments:
    """Per-record moments over the calibration pairs observed in both D and T (1/n normalized).

    A record with no pairs has count 0 and zero moments. Moments of disjoint sets of
    pairs combine exactly with ``merge_calibration_moments``, so they can be
    accumulated in one pass over the store, chunk by chunk.
    """

    pair_counts: I64Array
    dosage_mean: F64Array
    truth_mean: F64Array
    dosage_variance: F64Array
    truth_variance: F64Array
    covariance: F64Array

    def subset(self, records: NDArray) -> CalibrationMoments:
        return CalibrationMoments(*(getattr(self, field.name)[records] for field in fields(self)))


def calibration_moments(dosage: NDArray, truth: NDArray) -> CalibrationMoments:
    """Moments of (D, T) per record, from arrays of shape [records, samples] with NaN for a missing value."""
    dosage_values = np.asarray(dosage, dtype=np.float64)
    truth_values = np.asarray(truth, dtype=np.float64)
    if dosage_values.ndim != 2 or dosage_values.shape != truth_values.shape:
        raise ValueError("calibration_moments needs dosage and truth of the same shape [records, samples].")
    observed = np.isfinite(dosage_values) & np.isfinite(truth_values)
    counts = observed.sum(axis=1)
    divisor = np.maximum(counts, 1)
    dosage_mean = np.where(observed, dosage_values, 0.0).sum(axis=1) / divisor
    truth_mean = np.where(observed, truth_values, 0.0).sum(axis=1) / divisor
    dosage_centred = np.where(observed, dosage_values - dosage_mean[:, None], 0.0)
    truth_centred = np.where(observed, truth_values - truth_mean[:, None], 0.0)
    return CalibrationMoments(
        pair_counts=counts.astype(np.int64),
        dosage_mean=dosage_mean,
        truth_mean=truth_mean,
        dosage_variance=(dosage_centred**2).sum(axis=1) / divisor,
        truth_variance=(truth_centred**2).sum(axis=1) / divisor,
        covariance=(dosage_centred * truth_centred).sum(axis=1) / divisor,
    )


def merge_calibration_moments(first: CalibrationMoments, second: CalibrationMoments) -> CalibrationMoments:
    """The moments of the union of two disjoint sets of pairs (Chan, Golub and LeVeque's pairwise update)."""
    if first.pair_counts.shape != second.pair_counts.shape:
        raise ValueError("merge_calibration_moments needs moments of the same records.")
    first_count = first.pair_counts.astype(np.float64)
    second_count = second.pair_counts.astype(np.float64)
    count = first_count + second_count
    second_share = np.divide(second_count, count, out=np.zeros_like(count), where=count > 0)
    cross_weight = first_count * second_share
    dosage_step = second.dosage_mean - first.dosage_mean
    truth_step = second.truth_mean - first.truth_mean

    def pooled(first_moment: F64Array, second_moment: F64Array, product: F64Array) -> F64Array:
        total = first_count * first_moment + second_count * second_moment + cross_weight * product
        return np.divide(total, count, out=np.zeros_like(count), where=count > 0)

    return CalibrationMoments(
        pair_counts=first.pair_counts + second.pair_counts,
        dosage_mean=first.dosage_mean + second_share * dosage_step,
        truth_mean=first.truth_mean + second_share * truth_step,
        dosage_variance=pooled(first.dosage_variance, second.dosage_variance, dosage_step**2),
        truth_variance=pooled(first.truth_variance, second.truth_variance, truth_step**2),
        covariance=pooled(first.covariance, second.covariance, dosage_step * truth_step),
    )


def concatenate_calibration_moments(parts: Sequence[CalibrationMoments]) -> CalibrationMoments:
    """The moments of consecutive record chunks, joined in record order."""
    return CalibrationMoments(*(np.concatenate([getattr(part, field.name) for part in parts]) for field in fields(CalibrationMoments)))


@dataclass(frozen=True)
class NormalPooling:
    """Empirical-Bayes pooling of estimates y_j ~ N(x_j' beta, tau^2 + s_j)."""

    coefficients: F64Array
    between_variance: float
    shrunk: F64Array


def _bisect_to_exhaustion(score, low: float, high: float) -> float:
    """Root of a score that is positive at ``low`` and negative at ``high``, to the last float."""
    while True:
        middle = (low + high) / 2
        if middle in (low, high):
            return middle
        if score(middle) > 0.0:
            low = middle
        else:
            high = middle


def pool_normal(estimates: NDArray, sampling_variances: NDArray, design: NDArray | None = None) -> NormalPooling:
    """Marginal maximum likelihood for the normal-normal model, and each estimate's posterior mean.

    ``design`` is [estimates, features] for the prior mean x_j' beta; the default is
    an intercept. Estimates with infinite sampling variance don't enter the fit
    and are shrunk all the way to their prior mean; estimates with zero sampling
    variance are exact, don't enter the fit either, and are kept. The between-record variance is
    the root of its profile score, found by bisection to the last float. It is
    exactly 0 when the score is not positive there, i.e. when the estimates
    scatter no more than their own sampling error.
    """
    values = np.asarray(estimates, dtype=np.float64)
    variances = np.asarray(sampling_variances, dtype=np.float64)
    features = np.ones((values.shape[0], 1)) if design is None else np.asarray(design, dtype=np.float64)
    if values.ndim != 1 or variances.shape != values.shape or features.shape[0] != values.shape[0]:
        raise ValueError("pool_normal needs one sampling variance and one design row per estimate.")
    if not np.all(variances >= 0.0):
        raise ValueError("pool_normal needs nonnegative sampling variances.")
    exact = variances == 0.0
    informative = np.isfinite(variances) & ~exact
    if informative.sum() < features.shape[1]:
        raise ValueError("pool_normal needs at least as many informative estimates as design features.")
    fit_values, fit_variances, fit_features = values[informative], variances[informative], features[informative]

    def coefficients_at(between: float) -> F64Array:
        weights = 1.0 / (between + fit_variances)
        gram = fit_features.T @ (weights[:, None] * fit_features)
        return np.linalg.solve(gram, fit_features.T @ (weights * fit_values))

    def score(between: float) -> float:
        weights = 1.0 / (between + fit_variances)
        residuals = fit_values - fit_features @ coefficients_at(between)
        return float(np.sum(weights**2 * residuals**2 - weights))

    between = 0.0
    if score(0.0) > 0.0:
        high = float(np.var(fit_values))
        while score(high) > 0.0:
            high *= 2
        between = _bisect_to_exhaustion(score, 0.0, high)
    coefficients = coefficients_at(between)
    prior_means = features @ coefficients
    shrinkage = np.where(informative, between / (between + np.where(informative, variances, 1.0)), 0.0)
    shrunk = prior_means + shrinkage * (values - prior_means)
    shrunk[exact] = values[exact]
    return NormalPooling(coefficients, between, shrunk)


@dataclass(frozen=True)
class PooledCalibration:
    """Per record: the pooled scale kappa_j and the variance ratio lambda_j = Var(G_j) / Var(D_j)."""

    scales: F64Array
    variance_ratios: F64Array


def pooled_calibration(
    moments: CalibrationMoments, cohort_dosage_variance: NDArray, strata: NDArray, design: NDArray | None = None
) -> PooledCalibration:
    """Each record's kappa and lambda, pooled within its stratum.

    Within a stratum both are linear in the record's design row x_j (default an
    intercept), fitted by least squares over all the stratum's pairs:
    kappa(x) from sum_j S_DT,j x_j = sum_j S_DD,j x_j x_j' beta, and lambda(x) from
    sum_j S_TT,j x_j = sum_j S_DD,j x_j x_j' gamma, with S the per-record sums of
    centred products. The pooled fits weigh each record by its dosage energy, so a
    record whose few pairs happen to fit exactly carries no more weight than its
    energy. Each record's own slope then enters a normal-normal empirical Bayes
    around kappa(x_j), with the model's sampling variance
    V_j (lambda(x_j) - kappa(x_j)^2) / S_DD,j, where V_j is the record's cohort
    dosage variance: a sparse record's own residual collapses to 0 by chance, the
    model's does not. The between-record variance is chosen by marginal likelihood.
    A stratum whose pooled model leaves no residual keeps every informative slope,
    which is then exact, and gives the others kappa(x_j).
    """
    labels = np.asarray(strata)
    cohort_variance = np.asarray(cohort_dosage_variance, dtype=np.float64)
    if labels.shape != moments.pair_counts.shape or cohort_variance.shape != labels.shape:
        raise ValueError("pooled_calibration needs one stratum label and one cohort variance per record.")
    features = np.ones((labels.shape[0], 1)) if design is None else np.asarray(design, dtype=np.float64)
    if features.shape[0] != labels.shape[0] or not np.all(np.isfinite(features)):
        raise ValueError("pooled_calibration needs one finite design row per record.")
    counts = moments.pair_counts.astype(np.float64)
    dosage_energy = counts * moments.dosage_variance
    scales = np.empty_like(cohort_variance)
    ratios = np.empty_like(cohort_variance)
    for stratum in np.unique(labels):
        members = labels == stratum
        rows, energy = features[members], dosage_energy[members]
        gram = rows.T @ (energy[:, None] * rows)
        if np.linalg.matrix_rank(gram) < rows.shape[1]:
            raise ValueError(f"stratum {stratum!r} has too little dosage variation to fit its design.")
        slope_prior = rows @ np.linalg.solve(gram, rows.T @ (counts[members] * moments.covariance[members]))
        ratio = rows @ np.linalg.solve(gram, rows.T @ (counts[members] * moments.truth_variance[members]))
        informative = energy > 0.0
        slopes = np.divide(counts[members] * moments.covariance[members], energy, out=np.zeros_like(energy), where=informative)
        residual_scale = cohort_variance[members] * np.maximum(ratio - slope_prior**2, 0.0)
        sampling = np.divide(residual_scale, energy, out=np.full_like(energy, np.inf), where=informative)
        uncertain = np.isfinite(sampling) & (sampling > 0.0)
        if uncertain.sum() >= rows.shape[1]:
            scales[members] = pool_normal(slopes, sampling, rows).shrunk
        else:
            # The pooled model leaves no residual: every informative slope is exact.
            scales[members] = np.where(informative, slopes, slope_prior)
        ratios[members] = ratio
    return PooledCalibration(scales, ratios)


def residual_variances(cohort_dosage_variance: NDArray, scales: NDArray, variance_ratios: NDArray) -> F64Array:
    """v_j = E[(G - D*_j)^2] = Var(G_j) - Var(D*_j) = V_j (lambda_j - kappa_j^2) for a calibrated D*.

    The residual is nonnegative; a pooled lambda below kappa^2 is projected back to 0.
    With a truth that is not exact, lambda includes the truth's error variance, so v
    upper-bounds the genotype's residual variance.
    """
    variance = np.asarray(cohort_dosage_variance, dtype=np.float64)
    kappa = np.asarray(scales, dtype=np.float64)
    ratios = np.asarray(variance_ratios, dtype=np.float64)
    if kappa.shape != variance.shape or ratios.shape != variance.shape:
        raise ValueError("residual_variances needs one variance, scale and ratio per record.")
    return variance * np.maximum(ratios - kappa**2, 0.0)


def log_reliability_offsets(dosage_variance: NDArray, scales: NDArray, residual_variance: NDArray) -> F64Array:
    """log r^2 = log(Var(D*) / (Var(D*) + v)) with Var(D*) = kappa^2 Var(D); the prior offset.

    ``dosage_variance`` is the stored column's variance in the fitted cohort. A column
    with no calibrated variance carries no signal, and its offset is -inf: the caller
    must drop it, never give it a finite offset.
    """
    calibrated_variance = np.asarray(scales, dtype=np.float64) ** 2 * np.asarray(dosage_variance, dtype=np.float64)
    residual = np.asarray(residual_variance, dtype=np.float64)
    if np.any(calibrated_variance < 0.0) or np.any(residual < 0.0):
        raise ValueError("log_reliability_offsets needs nonnegative variances.")
    return _log_signal_share(calibrated_variance, residual)


def _log_signal_share(signal: F64Array, residual: F64Array) -> F64Array:
    """log(signal / (signal + residual)), -inf where there is no signal."""
    share = np.full(np.broadcast(signal, residual).shape, -np.inf)
    present = signal > 0.0
    share[present] = np.log(signal[present]) - np.log(signal[present] + residual[present])
    return share


def pooled_log_reliability(
    scales: NDArray,
    residual_variance: NDArray,
    cohort_dosage_mean: NDArray,
    cohort_dosage_variance: NDArray,
    group_counts: NDArray,
) -> F64Array:
    """log r^2 of each record's recalibrated column over the whole fitted cohort, from its groups' models.

    The arrays are [records, groups], one column per ancestry group's model, and
    ``group_counts`` is the number of fitted samples in each group. With
    w_g = n_g / n, each group's recalibration keeps that group's mean, so D*'s
    cohort variance is sum_g w_g (kappa_g^2 V_g + (mu_g - mu)^2) and its residual
    variance is sum_g w_g v_g, and r^2 = Var(D*) / (Var(D*) + v). A group with no
    fitted samples contributes nothing. A record whose recalibrated column has no
    cohort variance gets -inf.
    """
    kappa = np.asarray(scales, dtype=np.float64)
    residual = np.asarray(residual_variance, dtype=np.float64)
    means = np.asarray(cohort_dosage_mean, dtype=np.float64)
    variance = np.asarray(cohort_dosage_variance, dtype=np.float64)
    counts = np.asarray(group_counts, dtype=np.float64)
    if kappa.ndim != 2 or any(array.shape != kappa.shape for array in (residual, means, variance)) or counts.shape != kappa.shape[1:]:
        raise ValueError("pooled_log_reliability needs [records, groups] arrays and one count per group.")
    if np.any(counts < 0.0) or not counts.sum() > 0.0:
        raise ValueError("pooled_log_reliability needs nonnegative group counts with at least one fitted sample.")
    weights = counts / counts.sum()
    fitted = weights > 0.0

    def weighted(values: F64Array) -> F64Array:
        return (np.where(fitted, values, 0.0) * weights).sum(axis=1)

    centre = weighted(means)
    calibrated_variance = weighted(kappa**2 * variance + (means - centre[:, None]) ** 2)
    return _log_signal_share(calibrated_variance, weighted(residual))


@dataclass(frozen=True)
class LeakageMap:
    """The linear predictor of each target's genotype from its LD block, as corrections to D*.

    ``coefficients`` is [block columns, targets] in per-allele units, so the mapped
    column is Xtilde_k = D*_k + sum_j coefficients[j, k] (D*_j - column_means[j]).
    ``ridge_ratio`` is the marginal-likelihood ratio: 0 when the pairs show no
    leakage, infinite when they are fitted exactly (the minimum-norm interpolant).
    """

    targets: I64Array
    column_means: F64Array
    coefficients: F64Array
    ridge_ratio: float


def fit_leakage_map(calibrated_block: NDArray, target_truth: NDArray, targets: NDArray) -> LeakageMap:
    """Fit the A-map of one LD block from its calibration pairs.

    ``calibrated_block`` is [pairs, columns]: the calibration samples' calibrated
    stored columns D* for every column of the block, complete (no missing values).
    ``target_truth`` is [pairs, targets]: the truth of the columns being mapped,
    whose indices in the block are ``targets``. The ridge prior is exchangeable in
    standardized units, c ~ N(0, rho sigma_k^2 I). Its ratio rho is shared by the
    block's targets and maximizes the marginal likelihood of the centred residuals,
    which live in the n - 1 directions orthogonal to the mean, with each target's
    sigma_k^2 profiled out in closed form. When the likelihood still rises as rho
    grows without bound, the map is its limit, the minimum-norm least-squares fit.
    A column with no calibration variation predicts nothing and gets coefficient 0,
    as does a target whose calibrated column already equals its truth on every pair.
    """
    block = np.asarray(calibrated_block, dtype=np.float64)
    truth = np.asarray(target_truth, dtype=np.float64)
    target_index = np.asarray(targets, dtype=np.int64)
    if block.ndim != 2 or truth.ndim != 2 or truth.shape != (block.shape[0], target_index.shape[0]):
        raise ValueError("fit_leakage_map needs a block [pairs, columns] and one truth column per target.")
    if not (np.all(np.isfinite(block)) and np.all(np.isfinite(truth))):
        raise ValueError("fit_leakage_map needs complete calibration pairs.")
    if np.any((target_index < 0) | (target_index >= block.shape[1])) or np.unique(target_index).size != target_index.size:
        raise ValueError("fit_leakage_map needs distinct target columns inside the block.")
    column_means = block.mean(axis=0)
    coefficients = np.zeros((block.shape[1], target_index.shape[0]))
    centred = block - column_means
    deviations = centred.std(axis=0)
    varying = deviations > 0.0
    residuals = (truth - truth.mean(axis=0)) - centred[:, target_index]
    leaking = np.any(residuals != 0.0, axis=0)
    if not (np.any(varying) and np.any(leaking)):
        return LeakageMap(target_index, column_means, coefficients, 0.0)
    standardized = centred[:, varying] / deviations[varying]
    left, singular, right_transposed = np.linalg.svd(standardized, full_matrices=False)
    retained = singular > singular[0] * np.finfo(np.float64).eps * max(standardized.shape)
    left, singular, right_transposed = left[:, retained], singular[retained], right_transposed[retained]
    eigenvalues = singular**2
    directions = block.shape[0] - 1
    target_count = int(leaking.sum())
    projected = left.T @ residuals[:, leaking]
    outside = np.sum((residuals[:, leaking] - left @ projected) ** 2, axis=0)

    def score(ratio: float) -> float:
        shrink = 1.0 + ratio * eigenvalues
        energy = np.sum(projected**2 / shrink[:, None], axis=0) + outside
        energy_slope = -np.sum(projected**2 * (eigenvalues / shrink**2)[:, None], axis=0)
        return float(-(directions * np.sum(energy_slope / energy) + target_count * np.sum(eigenvalues / shrink)) / 2)

    if not score(0.0) > 0.0:
        return LeakageMap(target_index, column_means, coefficients, 0.0)
    high = 1.0 / eigenvalues[0]
    while np.isfinite(high) and score(high) > 0.0:
        high *= 2
    ratio = _bisect_to_exhaustion(score, 0.0, high) if np.isfinite(high) else np.inf
    # rho s / (1 + rho s^2) written as s / (1 / rho + s^2), which is 1 / s at rho = infinity.
    gains = singular / (1.0 / ratio + eigenvalues)
    standardized_coefficients = right_transposed.T @ (gains[:, None] * projected)
    coefficients[np.ix_(varying, leaking)] = standardized_coefficients / deviations[varying][:, None]
    return LeakageMap(target_index, column_means, coefficients, ratio)


def apply_leakage_map(calibrated_block: NDArray, leakage: LeakageMap) -> F64Array:
    """The mapped columns for any samples' calibrated block [samples, columns]: targets corrected, others unchanged."""
    block = np.asarray(calibrated_block, dtype=np.float64)
    if block.ndim != 2 or block.shape[1] != leakage.column_means.shape[0]:
        raise ValueError("apply_leakage_map needs the block the map was fitted on.")
    mapped = block.copy()
    mapped[:, leakage.targets] += (block - leakage.column_means) @ leakage.coefficients
    return mapped


def leakage_transform(leakage: LeakageMap) -> F64Array:
    """A = I + E with E[:, targets] = coefficients, so the centred mapped block is X A."""
    transform = np.eye(leakage.column_means.shape[0])
    transform[:, leakage.targets] += leakage.coefficients
    return transform


def mapped_gram(gram: NDArray, leakage: LeakageMap) -> F64Array:
    """A' G A: the Stage 0 Gram of the mapped block from the Gram G of its centred calibrated columns."""
    transform = leakage_transform(leakage)
    matrix = np.asarray(gram, dtype=np.float64)
    if matrix.shape != (transform.shape[0], transform.shape[0]):
        raise ValueError("mapped_gram needs the Gram of the block the map was fitted on.")
    return transform.T @ matrix @ transform


@dataclass(frozen=True)
class LdBlock:
    """One LD block's records (store row indices) and which of them are mapped targets (indices into ``records``)."""

    records: I64Array
    targets: I64Array


@dataclass(frozen=True)
class BlockPairs:
    """One LD block's calibration pairs for its leakage map.

    ``dosage`` is [pairs, block records]: the stored (uncalibrated) columns of every
    record in the block, and ``truth`` is [pairs, targets] for the block's targets.
    Only pairs complete across the block and its targets are kept.
    """

    block: LdBlock
    dosage: F64Array
    truth: F64Array

    def __post_init__(self) -> None:
        records = np.asarray(self.block.records)
        targets = np.asarray(self.block.targets)
        dosage, truth = np.asarray(self.dosage), np.asarray(self.truth)
        if dosage.ndim != 2 or dosage.shape[1] != records.shape[0] or truth.shape != (dosage.shape[0], targets.shape[0]):
            raise ValueError("BlockPairs needs dosage [pairs, block records] and truth [pairs, targets].")
        if not (np.all(np.isfinite(dosage)) and np.all(np.isfinite(truth))):
            raise ValueError("BlockPairs needs pairs complete across the block and its targets.")


@dataclass(frozen=True)
class CalibrationPairs:
    """What the measurement model uses from the samples carrying both a stored column and a truth genotype.

    ``moments`` are the per-record moments over those samples (store record order),
    which the store step accumulates in its one pass with ``calibration_moments``
    and ``merge_calibration_moments``. ``blocks`` holds the pair-level data of the
    LD blocks that get a leakage map. The truth must be an orthogonal call whose
    error is independent of the stored column given the genotype, with E[T | G] = G:
    a long-read or other independent genotyping, never the imputation itself. A
    truth that misclassifies attenuates kappa by its own slope Cov(T, G) / Var(G),
    which no second moment of such truths can identify, so the truth source is the
    user's stated input.
    """

    sample_ids: tuple[ResearchId, ...]
    moments: CalibrationMoments
    blocks: tuple[BlockPairs, ...] = ()

    def __post_init__(self) -> None:
        if not all(isinstance(sample, ResearchId) for sample in self.sample_ids):
            raise TypeError("CalibrationPairs needs typed ResearchIds for its samples.")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("CalibrationPairs has a repeated research ID.")
        if np.any(self.moments.pair_counts > len(self.sample_ids)):
            raise ValueError("CalibrationPairs has a record with more pairs than calibration samples.")
        for pairs in self.blocks:
            if pairs.dosage.shape[0] > len(self.sample_ids):
                raise ValueError("CalibrationPairs has a block with more pairs than calibration samples.")


def calibration_pairs(
    sample_ids: Sequence[ResearchId], dosage: NDArray, truth: NDArray, blocks: Sequence[LdBlock] = ()
) -> CalibrationPairs:
    """Calibration pairs from dense [records, samples] arrays, NaN where missing (a benchmark's scale)."""
    dosage_values = np.asarray(dosage, dtype=np.float64)
    truth_values = np.asarray(truth, dtype=np.float64)
    if dosage_values.ndim != 2 or dosage_values.shape[1] != len(sample_ids):
        raise ValueError("calibration_pairs needs dosage and truth of shape [records, samples], one column per sample ID.")
    block_pairs = []
    for block in blocks:
        records = np.asarray(block.records, dtype=np.int64)
        target_records = records[np.asarray(block.targets, dtype=np.int64)]
        complete = np.all(np.isfinite(dosage_values[records]), axis=0) & np.all(np.isfinite(truth_values[target_records]), axis=0)
        block_pairs.append(BlockPairs(block, dosage_values[records][:, complete].T, truth_values[target_records][:, complete].T))
    return CalibrationPairs(tuple(sample_ids), calibration_moments(dosage_values, truth_values), tuple(block_pairs))


def fit_block_map(pairs: BlockPairs, scales: NDArray) -> LeakageMap:
    """One block's leakage map, from its pairs with each column recalibrated by its record's scale."""
    block_scales = np.asarray(scales, dtype=np.float64)[np.asarray(pairs.block.records, dtype=np.int64)]
    dosage = np.asarray(pairs.dosage, dtype=np.float64)
    means = dosage.mean(axis=0)
    return fit_leakage_map(means + block_scales * (dosage - means), pairs.truth, pairs.block.targets)


@dataclass(frozen=True)
class MeasurementModel:
    """What the fit uses for its columns, offsets and predictive variance, and what was and wasn't applied.

    ``residual_variance`` is v_j = E[(G - D*_j)^2], the predictive variance's
    measurement term. For a record without calibration pairs it is the value the
    reported r^2 implies for the unscaled column, Var(D) (1 - r^2) / r^2.
    """

    scales: F64Array
    residual_variance: F64Array
    log_reliability: F64Array
    leakage_maps: tuple[LeakageMap, ...]
    certificate: dict[str, object]


def fit_measurement_model(
    calibration: CalibrationPairs | None,
    cohort_dosage_variance: NDArray,
    strata: NDArray,
    design: NDArray | None = None,
    reported_reliability: NDArray | None = None,
) -> MeasurementModel:
    """The measurement model for every stored record, from calibration pairs where there are any.

    ``cohort_dosage_variance`` is each stored column's variance in the fitted cohort.
    ``strata`` and ``design`` define the pooling of the recalibration scales (see
    ``pooled_calibration``). ``reported_reliability`` is the imputation's own r^2
    per record (INFO or DR2). It is used only for records with fewer than 3
    calibration pairs, and is then required. Those records get no recalibration and
    their offsets come from the reported r^2; the certificate counts them and the
    fact is logged, never silent.
    """
    variance = np.asarray(cohort_dosage_variance, dtype=np.float64)
    labels = np.asarray(strata)
    if variance.ndim != 1 or np.any(variance < 0.0) or labels.shape != variance.shape:
        raise ValueError("fit_measurement_model needs one nonnegative cohort variance and one stratum per record.")
    calibrated = np.zeros(variance.shape, dtype=bool)
    if calibration is not None:
        if calibration.moments.pair_counts.shape != variance.shape:
            raise ValueError("the calibration pairs and the cohort variances need the same records.")
        calibrated = calibration.moments.pair_counts > 2
    scales = np.ones_like(variance)
    residual = np.empty_like(variance)
    offsets = np.empty_like(variance)
    uncalibrated = ~calibrated
    if np.any(uncalibrated):
        if reported_reliability is None:
            raise ValueError(
                f"{int(uncalibrated.sum())} records have fewer than 3 calibration pairs; their reliability offsets "
                "need the imputation's reported r^2."
            )
        reported = np.asarray(reported_reliability, dtype=np.float64)
        if reported.shape != variance.shape or np.any((reported < 0.0) | (reported > 1.0)):
            raise ValueError("reported_reliability needs one r^2 in [0, 1] per record.")
        with np.errstate(divide="ignore", invalid="ignore"):
            offsets[uncalibrated] = np.log(reported[uncalibrated])
            residual[uncalibrated] = variance[uncalibrated] * (1.0 - reported[uncalibrated]) / reported[uncalibrated]
    maps: list[LeakageMap] = []
    if calibration is not None and np.any(calibrated):
        moments = calibration.moments.subset(calibrated)
        feature_rows = None if design is None else np.asarray(design, dtype=np.float64)[calibrated]
        pooled = pooled_calibration(moments, variance[calibrated], labels[calibrated], feature_rows)
        scales[calibrated] = pooled.scales
        residual[calibrated] = residual_variances(variance[calibrated], pooled.scales, pooled.variance_ratios)
        offsets[calibrated] = log_reliability_offsets(variance[calibrated], scales[calibrated], residual[calibrated])
        maps = [fit_block_map(pairs, scales) for pairs in calibration.blocks]
    ratios = np.array([leakage.ridge_ratio for leakage in maps])
    certificate: dict[str, object] = {
        "calibration_samples": 0 if calibration is None else len(calibration.sample_ids),
        "calibrated_records": int(calibrated.sum()),
        "uncalibrated_records": int(uncalibrated.sum()),
        "recalibration": (
            f"applied to {int(calibrated.sum())} records: pooled per-record kappa from the calibration pairs; "
            f"not applied to {int(uncalibrated.sum())} records with fewer than 3 pairs"
        ),
        "leakage_correction": (
            f"applied to {len(maps)} LD blocks: no leakage found in {int(np.sum(ratios == 0.0))}, "
            f"fitted exactly (minimum-norm interpolant) in {int(np.sum(np.isinf(ratios)))}"
            if maps
            else "not applied: no LD block has calibration pairs"
        ),
        "reliability_source": (
            f"calibration pairs for {int(calibrated.sum())} records; the imputation's reported r^2 "
            f"(biased for draw-type columns) for {int(uncalibrated.sum())} records"
        ),
    }
    for key in ("recalibration", "leakage_correction", "reliability_source"):
        log(f"measurement model: {key}: {certificate[key]}")
    return MeasurementModel(scales, residual, offsets, tuple(maps), certificate)
