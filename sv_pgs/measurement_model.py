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
   D*_j = mu_j + kappa_j (D_j - mu_j) satisfies Cov(G, D*) = Var(D*). Since
   Cov(T, D) = Cov(G, D), one truth suffices. Each record's estimate is noisy,
   so records are pooled by a normal-normal empirical Bayes: the mean is a linear
   model in a caller-given design, and the between-record variance is chosen by
   marginal likelihood (0 when the records agree).
2. Conditional moments. With D* calibrated, the residual variance
   v_j = E[(G - D*_j)^2] enters the predictive variance, and
   r^2_j = Var(D*_j) / (Var(D*_j) + v_j) gives the prior offset log r^2_j
   (coefficient 1, docs/design/math/scale_model.md section 1).
3. The leakage map, the "A-map" (scale_model.md sections 2-3). A draw-type
   column leaves information about G_k in nearby columns, so in a joint fit part
   of an SV's effect moves onto its tag SNPs, and a stacked truth half no longer
   shares coefficients with the imputed half. The linear predictor of G_k from
   the whole LD block,
       Xtilde_k = D*_k + sum_j C_jk (D*_j - mu_j),
   removes it. C_k is the Bayesian ridge fit of the calibrated residual
   T_k - D*_k on the block's standardized columns. One ridge ratio per block is
   chosen by marginal likelihood, and each target has its own noise variance.
   A ratio of 0 means the data show no leakage and gives back D*. A block whose
   ratio is not identified (the calibration pairs can interpolate every target)
   returns no correction and says so.
4. The engine's products for the mapped columns. Xtilde = X A with
   A = I + E, where E holds C in the target columns, so a block's centred Gram
   becomes A' G A and its cross-products A' X'y.

The calibration pairs are an explicit input (``CalibrationPairs``, keyed by
typed research IDs). In the AoU workspace they come from a truth source the user
supplies: the long-read panel members' hard calls are not part of the imputation
deliverables. Without calibration pairs, ``fit_measurement_model`` degrades
loudly, never silently. It applies no recalibration and no leakage correction,
takes the reliability offsets from the imputation's own reported r^2, which the
caller must pass, and records each of these facts in the model's certificate,
which the fit certificate carries. It also logs them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from sv_pgs._typing import F64Array, I64Array, NDArray
from sv_pgs.progress import log
from sv_pgs.sample_ids import ResearchId


@dataclass(frozen=True)
class CalibrationMoments:
    """Per-record moments over the calibration pairs observed in both D and T (1/n normalized)."""

    pair_counts: I64Array
    dosage_mean: F64Array
    truth_mean: F64Array
    dosage_variance: F64Array
    truth_variance: F64Array
    covariance: F64Array


def calibration_moments(dosage: NDArray, truth: NDArray) -> CalibrationMoments:
    """Moments of (D, T) per record, from arrays of shape [records, samples] with NaN for a missing value."""
    dosage_values = np.asarray(dosage, dtype=np.float64)
    truth_values = np.asarray(truth, dtype=np.float64)
    if dosage_values.ndim != 2 or dosage_values.shape != truth_values.shape:
        raise ValueError("calibration_moments needs dosage and truth of the same shape [records, samples].")
    observed = np.isfinite(dosage_values) & np.isfinite(truth_values)
    counts = observed.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("every record needs at least one calibration pair observed in both D and T.")
    dosage_zeroed = np.where(observed, dosage_values, 0.0)
    truth_zeroed = np.where(observed, truth_values, 0.0)
    dosage_mean = dosage_zeroed.sum(axis=1) / counts
    truth_mean = truth_zeroed.sum(axis=1) / counts
    dosage_centred = np.where(observed, dosage_values - dosage_mean[:, None], 0.0)
    truth_centred = np.where(observed, truth_values - truth_mean[:, None], 0.0)
    return CalibrationMoments(
        pair_counts=counts.astype(np.int64),
        dosage_mean=dosage_mean,
        truth_mean=truth_mean,
        dosage_variance=(dosage_centred**2).sum(axis=1) / counts,
        truth_variance=(truth_centred**2).sum(axis=1) / counts,
        covariance=(dosage_centred * truth_centred).sum(axis=1) / counts,
    )


def record_scale_estimates(moments: CalibrationMoments) -> tuple[F64Array, F64Array]:
    """Least-squares slope kappa_hat = Cov(T, D) / Var(D) per record and its sampling variance.

    The sampling variance is the ordinary least-squares one, RSS / (n - 2) / S_DD. A
    record with no dosage variation, or fewer than 3 pairs, carries no information
    about its slope: its variance is infinite, and pooling gives it the stratum's mean.
    """
    counts = moments.pair_counts.astype(np.float64)
    informative = (moments.dosage_variance > 0.0) & (moments.pair_counts > 2)
    slopes = np.zeros_like(moments.covariance)
    variances = np.full_like(moments.covariance, np.inf)
    slopes[informative] = moments.covariance[informative] / moments.dosage_variance[informative]
    residual_sum_of_squares = counts * (
        moments.truth_variance - moments.covariance**2 / np.where(informative, moments.dosage_variance, 1.0)
    )
    variances[informative] = (
        residual_sum_of_squares[informative]
        / (counts[informative] - 2)
        / (counts[informative] * moments.dosage_variance[informative])
    )
    return slopes, variances


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


def recalibration_scales(moments: CalibrationMoments, strata: NDArray, design: NDArray | None = None) -> F64Array:
    """Pooled kappa per record: the empirical-Bayes posterior mean of its slope within its stratum.

    ``strata`` labels each record's pooling stratum, e.g. variant class x ancestry
    group. ``design`` is [records, features] for the stratum's prior mean, e.g. an
    intercept with the record's logit frequency and imputation information. A
    stratum whose informative records are fewer than its design's features is
    refused rather than guessed.
    """
    slopes, variances = record_scale_estimates(moments)
    labels = np.asarray(strata)
    if labels.shape != slopes.shape:
        raise ValueError("recalibration_scales needs one stratum label per record.")
    features = np.ones((slopes.shape[0], 1)) if design is None else np.asarray(design, dtype=np.float64)
    pooled = np.empty_like(slopes)
    for stratum in np.unique(labels):
        members = labels == stratum
        pooled[members] = pool_normal(slopes[members], variances[members], features[members]).shrunk
    return pooled


def residual_variances(dosage: NDArray, truth: NDArray, scales: NDArray) -> F64Array:
    """v_j = mean over pairs of ((T - Tbar) - kappa_j (D - Dbar))^2, nonnegative by construction.

    This is Var(G - D*) plus the truth's own error variance, so it upper-bounds the
    genotype's residual variance whenever the truth is not exact.
    """
    dosage_values = np.asarray(dosage, dtype=np.float64)
    truth_values = np.asarray(truth, dtype=np.float64)
    kappa = np.asarray(scales, dtype=np.float64)
    observed = np.isfinite(dosage_values) & np.isfinite(truth_values)
    counts = observed.sum(axis=1)
    if kappa.shape != (dosage_values.shape[0],) or np.any(counts == 0):
        raise ValueError("residual_variances needs one scale per record and a calibration pair in every record.")
    dosage_mean = np.where(observed, dosage_values, 0.0).sum(axis=1) / counts
    truth_mean = np.where(observed, truth_values, 0.0).sum(axis=1) / counts
    residual = (truth_values - truth_mean[:, None]) - kappa[:, None] * (dosage_values - dosage_mean[:, None])
    return (np.where(observed, residual, 0.0) ** 2).sum(axis=1) / counts


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
    with np.errstate(divide="ignore"):
        return np.log(calibrated_variance) - np.log(calibrated_variance + residual)


@dataclass(frozen=True)
class LeakageMap:
    """The linear predictor of each target's genotype from its LD block, as corrections to D*.

    ``coefficients`` is [block columns, targets] in per-allele units, so the mapped
    column is Xtilde_k = D*_k + sum_j coefficients[j, k] (D*_j - column_means[j]).
    """

    targets: I64Array
    column_means: F64Array
    coefficients: F64Array
    ridge_ratio: float
    identified: bool
    log_evidence_gain: float


def _unmapped(targets: I64Array, column_means: F64Array) -> LeakageMap:
    return LeakageMap(targets, column_means, np.zeros((column_means.shape[0], targets.shape[0])), 0.0, True, 0.0)


def fit_leakage_map(calibrated_block: NDArray, target_truth: NDArray, targets: NDArray) -> LeakageMap:
    """Fit the A-map of one LD block from its calibration pairs.

    ``calibrated_block`` is [pairs, columns]: the calibration samples' calibrated
    stored columns D* for every column of the block, complete (no missing values).
    ``target_truth`` is [pairs, targets]: the truth of the columns being mapped,
    whose indices in the block are ``targets``. The ridge prior is exchangeable in
    standardized units, c ~ N(0, rho sigma_k^2 I). Its ratio rho is shared by the
    block's targets and is the maximizer of the profile marginal likelihood, with
    each target's sigma_k^2 profiled out in closed form. A column with no
    calibration variation predicts nothing and gets coefficient 0.
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
    pair_count = block.shape[0]
    column_means = block.mean(axis=0)
    centred = block - column_means
    deviations = centred.std(axis=0)
    varying = deviations > 0.0
    residuals = (truth - truth.mean(axis=0)) - centred[:, target_index]
    if not np.any(varying):
        return _unmapped(target_index, column_means)
    standardized = centred[:, varying] / deviations[varying]
    left, singular, right_transposed = np.linalg.svd(standardized, full_matrices=False)
    retained = singular > singular[0] * np.finfo(np.float64).eps * max(standardized.shape)
    left, singular, right_transposed = left[:, retained], singular[retained], right_transposed[retained]
    eigenvalues = singular**2
    projected = left.T @ residuals
    outside = np.sum((residuals - left @ projected) ** 2, axis=0)

    def residual_energy(ratio: float) -> F64Array:
        return np.sum(projected**2 / (1.0 + ratio * eigenvalues)[:, None], axis=0) + outside

    def profile(ratio: float) -> float:
        return float(-(pair_count * np.sum(np.log(residual_energy(ratio))) + truth.shape[1] * np.sum(np.log1p(ratio * eigenvalues))) / 2)

    def score(ratio: float) -> float:
        shrink = 1.0 + ratio * eigenvalues
        energy_slope = -np.sum(projected**2 * (eigenvalues / shrink**2)[:, None], axis=0)
        return float(-(pair_count * np.sum(energy_slope / residual_energy(ratio)) + truth.shape[1] * np.sum(eigenvalues / shrink)) / 2)

    if not score(0.0) > 0.0:
        return _unmapped(target_index, column_means)
    high = 1.0 / eigenvalues[0]
    while score(high) > 0.0:
        high *= 2
        if not np.isfinite(high):
            unmapped = _unmapped(target_index, column_means)
            return LeakageMap(unmapped.targets, unmapped.column_means, unmapped.coefficients, np.inf, False, 0.0)
    ratio = _bisect_to_exhaustion(score, 0.0, high)
    standardized_coefficients = right_transposed.T @ ((ratio * singular / (1.0 + ratio * eigenvalues))[:, None] * projected)
    coefficients = np.zeros((block.shape[1], target_index.shape[0]))
    coefficients[varying] = standardized_coefficients / deviations[varying][:, None]
    return LeakageMap(target_index, column_means, coefficients, ratio, True, profile(ratio) - profile(0.0))


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
class CalibrationPairs:
    """Samples carrying both a stored column and a truth genotype, [records, samples] in store record order.

    ``dosage`` is the samples' stored (uncalibrated) column and ``truth`` their truth
    genotype for the same records, NaN where either is missing. The truth's error
    must be independent of the stored column: a long-read or other orthogonal call,
    never the imputation itself.
    """

    sample_ids: tuple[ResearchId, ...]
    dosage: F64Array
    truth: F64Array

    def __post_init__(self) -> None:
        if not all(isinstance(sample, ResearchId) for sample in self.sample_ids):
            raise TypeError("CalibrationPairs needs typed ResearchIds for its samples.")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("CalibrationPairs has a repeated research ID.")
        dosage = np.asarray(self.dosage)
        if dosage.ndim != 2 or dosage.shape != np.asarray(self.truth).shape or dosage.shape[1] != len(self.sample_ids):
            raise ValueError("CalibrationPairs needs dosage and truth of shape [records, samples], one column per sample ID.")


@dataclass(frozen=True)
class LdBlock:
    """One LD block's records (store row indices) and which of them are mapped targets (indices into ``records``)."""

    records: I64Array
    targets: I64Array


@dataclass(frozen=True)
class MeasurementModel:
    """What the fit uses for its columns, offsets and predictive variance, and what was and wasn't applied."""

    scales: F64Array
    residual_variance: F64Array | None
    log_reliability: F64Array
    leakage_maps: tuple[LeakageMap, ...]
    certificate: dict[str, object]


def fit_measurement_model(
    calibration: CalibrationPairs | None,
    cohort_dosage_variance: NDArray,
    strata: NDArray,
    design: NDArray | None = None,
    blocks: Sequence[LdBlock] = (),
    reported_reliability: NDArray | None = None,
) -> MeasurementModel:
    """The measurement model for every stored record, from calibration pairs when there are any.

    ``cohort_dosage_variance`` is each stored column's variance in the fitted cohort.
    ``strata`` and ``design`` define the pooling of the recalibration scales (see
    ``recalibration_scales``). ``blocks`` lists the LD blocks whose imperfect columns
    get a leakage map. ``reported_reliability`` is the imputation's own r^2 per
    record (INFO or DR2). It is used only when there are no calibration pairs, and
    is then required.
    """
    variance = np.asarray(cohort_dosage_variance, dtype=np.float64)
    if calibration is None:
        if reported_reliability is None:
            raise ValueError("without calibration pairs the reliability offsets need the imputation's reported r^2.")
        reported = np.asarray(reported_reliability, dtype=np.float64)
        if reported.shape != variance.shape or np.any((reported < 0.0) | (reported > 1.0)):
            raise ValueError("reported_reliability needs one r^2 in [0, 1] per record.")
        certificate: dict[str, object] = {
            "calibration_pairs": 0,
            "recalibration": "not applied: no truth genotypes were supplied",
            "leakage_correction": "not applied: no truth genotypes were supplied",
            "reliability_source": "the imputation's reported r^2 (biased for draw-type columns)",
        }
        for key in ("recalibration", "leakage_correction", "reliability_source"):
            log(f"measurement model: {key}: {certificate[key]}")
        with np.errstate(divide="ignore"):
            offsets = np.log(reported)
        return MeasurementModel(np.ones_like(variance), None, offsets, (), certificate)
    moments = calibration_moments(calibration.dosage, calibration.truth)
    if moments.pair_counts.shape != variance.shape:
        raise ValueError("the calibration pairs and the cohort variances need the same records.")
    scales = recalibration_scales(moments, strata, design)
    residual = residual_variances(calibration.dosage, calibration.truth, scales)
    offsets = log_reliability_offsets(variance, scales, residual)
    calibrated = moments.dosage_mean[:, None] + scales[:, None] * (np.asarray(calibration.dosage, dtype=np.float64) - moments.dosage_mean[:, None])
    maps: list[LeakageMap] = []
    unidentified = 0
    for block in blocks:
        block_rows = np.asarray(block.records, dtype=np.int64)
        target_rows = block_rows[np.asarray(block.targets, dtype=np.int64)]
        complete = np.all(np.isfinite(calibrated[block_rows]), axis=0) & np.all(np.isfinite(np.asarray(calibration.truth)[target_rows]), axis=0)
        leakage = fit_leakage_map(
            calibrated[block_rows][:, complete].T, np.asarray(calibration.truth, dtype=np.float64)[target_rows][:, complete].T, block.targets
        )
        unidentified += int(not leakage.identified)
        maps.append(leakage)
    certificate = {
        "calibration_pairs": len(calibration.sample_ids),
        "recalibration": "applied: pooled per-record kappa from the calibration pairs",
        "leakage_correction": f"applied to {len(maps)} LD blocks; not identified (left unmapped) in {unidentified}",
        "reliability_source": "calibration pairs",
    }
    log(f"measurement model: {certificate['calibration_pairs']} calibration pairs; {certificate['leakage_correction']}")
    return MeasurementModel(scales, residual, offsets, tuple(maps), certificate)
