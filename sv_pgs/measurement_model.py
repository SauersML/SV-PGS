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
   between-record variance is the energy-weighted moment estimate (0 when the
   records agree). This is the linear calibration; its monotone counterpart, for
   a column whose E[G | D] is curved, is imputation_reliability's isotonic shape.
   Both regress a truth on D, which the truth's own error does not attenuate: it
   attenuates the correlation of D with the truth, not the regression on D. Both
   also give Cov(G, D*) = Var(D*), which is one orthogonality condition and not
   calibration at every dosage; only the monotone map delivers E[G | D*] = D*.
   mu_j is the column's own mean, so D* carries the dosage's mean rather than the
   genotype's; every column is centred before it is fitted, so only kappa_j
   reaches the fit.
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
   T_k - D*_k on the block's standardized columns, from the pairs'
   cross-covariances and the cohort's block covariance (A = Sigma_D^-1 Sigma_DG
   with Sigma_D from the cohort and Sigma_DG from the truth). One ridge ratio per
   block is chosen by marginal likelihood, and each target has its own noise
   variance. A ratio of 0 means the data show no leakage and gives back D*.
4. The engine's products for the mapped columns. Xtilde = X A with
   A = I + E, where E holds C in the target columns, so a block's centred Gram
   becomes A' G A and its cross-products A' X'y.

The calibration pairs are an explicit input (``CalibrationPairs``, keyed by
typed research IDs). They enter as per-record sufficient statistics, computed
chunk by chunk of records, plus the pair-level data of the LD blocks to be
mapped, so no [records, samples] array of the whole store is ever needed. In the AoU workspace
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
from dataclasses import dataclass, field, fields, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from sv_pgs._typing import F64Array, I64Array, NDArray
from sv_pgs.progress import log
from sv_pgs.sample_ids import ResearchId
from sv_pgs.store_converter import _uniform_cubic_weights


@dataclass(frozen=True)
class CalibrationMoments:
    """Per-record central moments over the calibration pairs observed in both D and T (1/n normalized).

    With w = D - Dbar and u = T - Tbar over a record's pairs: the variances and
    covariance, and the fourth-order products E[w^4], E[w^3 u] and E[w^2 u^2]
    that give each slope its heteroscedasticity-robust sampling variance. A record
    with no pairs has count 0 and zero moments. Each record's moments come from all
    its pairs at once, so a pass over the store may split the records into chunks
    (``concatenate_calibration_moments``) but not the pairs.
    """

    pair_counts: I64Array
    dosage_mean: F64Array
    truth_mean: F64Array
    dosage_variance: F64Array
    truth_variance: F64Array
    covariance: F64Array
    dosage_fourth: F64Array
    dosage_cubed_truth: F64Array
    dosage_squared_truth_squared: F64Array

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
    dosage_squared = dosage_centred**2
    return CalibrationMoments(
        pair_counts=counts.astype(np.int64),
        dosage_mean=dosage_mean,
        truth_mean=truth_mean,
        dosage_variance=dosage_squared.sum(axis=1) / divisor,
        truth_variance=(truth_centred**2).sum(axis=1) / divisor,
        covariance=(dosage_centred * truth_centred).sum(axis=1) / divisor,
        dosage_fourth=(dosage_squared**2).sum(axis=1) / divisor,
        dosage_cubed_truth=(dosage_squared * dosage_centred * truth_centred).sum(axis=1) / divisor,
        dosage_squared_truth_squared=(dosage_squared * truth_centred**2).sum(axis=1) / divisor,
    )


def concatenate_calibration_moments(parts: Sequence[CalibrationMoments]) -> CalibrationMoments:
    """The moments of consecutive record chunks, joined in record order."""
    return CalibrationMoments(*(np.concatenate([getattr(part, field.name) for part in parts]) for field in fields(CalibrationMoments)))


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


def octave_spline_basis(log2_values: NDArray) -> F64Array:
    """Cubic B-splines uniform in log2 of a positive feature, one knot per octave over the data's range.

    The one-per-octave spacing is the convention of the store's SV-context kernel
    (docs/design/STORE.md), and the basis polynomials are its ``_uniform_cubic_weights``.
    The columns sum to 1 on the range, so the basis spans the intercept.
    """
    values = np.asarray(log2_values, dtype=np.float64)
    low = float(np.floor(values.min()))
    segments = max(int(np.ceil(values.max())) - int(low), 1)
    position = values - low
    segment = np.minimum(np.floor(position), segments - 1).astype(np.int64)
    weights = _uniform_cubic_weights(position - segment)
    # A spline of degree d has segments + d basis functions, d + 1 of them nonzero on each segment.
    basis = np.zeros((values.shape[0], segments + len(weights) - 1))
    rows = np.arange(values.shape[0])
    for shift, weight in enumerate(weights):
        basis[rows, segment + shift] = weight
    return basis


@dataclass(frozen=True)
class CalibrationCurve:
    """kappa(x) = x' beta and lambda(x) = x' gamma per stratum, fitted over the stratum's truth pairs."""

    strata: tuple[object, ...]
    kappa_coefficients: F64Array
    ratio_coefficients: F64Array

    def predict(self, strata: NDArray, design: NDArray) -> tuple[F64Array, F64Array]:
        """(kappa, lambda) at each row; a stratum the curve has no fit for gives NaN."""
        labels = np.asarray(strata)
        rows = np.asarray(design, dtype=np.float64)
        kappa = np.full(labels.shape[0], np.nan)
        ratio = np.full(labels.shape[0], np.nan)
        for index, stratum in enumerate(self.strata):
            members = labels == stratum
            kappa[members] = rows[members] @ self.kappa_coefficients[index]
            ratio[members] = rows[members] @ self.ratio_coefficients[index]
        return kappa, ratio


def fit_calibration_curve(moments: CalibrationMoments, strata: NDArray, design: NDArray) -> CalibrationCurve:
    """Energy-weighted least squares of kappa and lambda on the design, per stratum (see ``pooled_calibration``).

    The coefficients are the minimum-norm solution, so the fitted values are unique
    even for a rank-deficient design. The design may hold smooth bases of record
    features (e.g. B-splines in log length and logit frequency, a segdup fraction),
    so a record without truth pairs gets kappa(x) and lambda(x), hence
    r^2(x) = kappa(x)^2 / lambda(x), from the records that have them.
    """
    labels = np.asarray(strata)
    rows = np.asarray(design, dtype=np.float64)
    counts = moments.pair_counts.astype(np.float64)
    energy = counts * moments.dosage_variance
    kept, betas, gammas = [], [], []
    for stratum in np.unique(labels):
        members = labels == stratum
        if not energy[members].sum() > 0.0:
            continue
        gram = rows[members].T @ (energy[members, None] * rows[members])
        betas.append(np.linalg.lstsq(gram, rows[members].T @ (counts[members] * moments.covariance[members]), rcond=None)[0])
        gammas.append(np.linalg.lstsq(gram, rows[members].T @ (counts[members] * moments.truth_variance[members]), rcond=None)[0])
        kept.append(stratum.item() if hasattr(stratum, "item") else stratum)
    width = rows.shape[1]
    return CalibrationCurve(tuple(kept), np.array(betas).reshape(-1, width), np.array(gammas).reshape(-1, width))


@dataclass(frozen=True)
class PooledCalibration:
    """Per record: the pooled scale kappa_j, the variance ratio lambda_j = Var(G_j) / Var(D_j),
    and the between-record variance tau^2 of its stratum."""

    scales: F64Array
    variance_ratios: F64Array
    between_variances: F64Array


def pooled_calibration(
    moments: CalibrationMoments, cohort_dosage_variance: NDArray, strata: NDArray, design: NDArray | None = None
) -> PooledCalibration:
    """Each record's kappa and lambda, pooled within its stratum.

    Within a stratum both are linear in the record's design row x_j (default an
    intercept), fitted by least squares over all the stratum's pairs:
    kappa(x) from sum_j S_DT,j x_j = sum_j S_DD,j x_j x_j' beta, and lambda(x) from
    sum_j S_TT,j x_j = sum_j S_DD,j x_j x_j' gamma, with S the per-record sums of
    centred products. Each record's own slope kappa_hat_j = S_DT,j / S_DD,j then
    has the heteroscedasticity-robust sampling variance
    s_j = sum_i w_i^2 e_i^2 / S_DD,j^2 with the residual e = u - kappa(x_j) w taken
    at the pooled slope: a draw-type column's residual grows with |w|, which a
    homoscedastic variance misses, and a sparse record's own fitted residual
    collapses to 0 by chance, while its residual at the pooled slope does not (one
    carrier at a tiny dosage gives s_j = 1 / w_i^2). Its posterior mean under
    kappa_j ~ N(kappa(x_j), tau^2) is kappa(x_j) + tau^2 / (tau^2 + s_j) (kappa_hat_j - kappa(x_j)).

    Everything is weighed by dosage energy S_DD, the metric in which a scale error
    costs: sum_j V_j (kappa_j - kappa_j^true)^2. So tau^2 is the energy-weighted
    moment estimate, E[S_DD,j (kappa_hat_j - kappa(x_j))^2] = S_DD,j (tau^2 + s_j)
    summed over the stratum (the pooled fit treated as known), projected to 0 when
    negative. A normal likelihood would let a sparse record whose one carrier sits at
    a tiny dosage (slope 127, a heavy-tailed outlier) inflate tau^2; its energy is
    tiny, so here it cannot. Cov(G, D) >= 0 for any mixture of posterior means and
    draws, so a negative posterior mean is projected to 0: that column carries no
    signal. A stratum whose pooled model leaves no residual keeps every informative
    slope, which is then exact.
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
    between_variances = np.empty_like(cohort_variance)
    for stratum in np.unique(labels):
        members = labels == stratum
        rows, energy = features[members], dosage_energy[members]
        if not energy.sum() > 0.0:
            raise ValueError(f"stratum {stratum!r} has no dosage variation among its calibration pairs.")
        curve = fit_calibration_curve(moments.subset(members), labels[members], rows)
        prior, ratio = curve.predict(labels[members], rows)
        cross_energy = counts[members] * moments.covariance[members]
        informative = energy > 0.0
        slopes = np.divide(cross_energy, energy, out=np.zeros_like(energy), where=informative)
        # S_DD s_j: the sandwich sum_i w_i^2 e_i^2 / S_DD with e = u - kappa(x_j) w.
        residual = np.maximum(
            np.divide(
                moments.dosage_squared_truth_squared[members]
                - 2.0 * prior * moments.dosage_cubed_truth[members]
                + prior**2 * moments.dosage_fourth[members],
                moments.dosage_variance[members],
                out=np.zeros_like(energy),
                where=informative,
            ),
            0.0,
        )
        excess = np.sum(np.divide((cross_energy - prior * energy) ** 2, energy, out=np.zeros_like(energy), where=informative))
        between = max(float((excess - residual[informative].sum()) / energy.sum()), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            weight = np.where(residual > 0.0, between * energy / (between * energy + residual), 1.0)
        weight = np.where(informative, weight, 0.0)
        scales[members] = np.maximum(prior + weight * (slopes - prior), 0.0)
        ratios[members] = ratio
        between_variances[members] = between
    return PooledCalibration(scales, ratios, between_variances)


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
    ``ridge_ratio`` is the marginal-likelihood ratio t: 0 when the pairs show no
    leakage, infinite at the least-squares limit. ``fits_pairs_exactly`` marks a
    ratio stopped where a target's residual on the pairs reaches 0. ``records`` are
    the block's store rows (set by ``fit_block_map``), so a map is self-contained;
    ``targets`` index into them.
    """

    targets: I64Array
    column_means: F64Array
    coefficients: F64Array
    ridge_ratio: float
    fits_pairs_exactly: bool = False
    records: I64Array = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


def fit_leakage_map(cohort_covariance: NDArray, calibrated_pairs: NDArray, target_truth: NDArray, targets: NDArray) -> LeakageMap:
    """Fit the A-map of one LD block: A = Sigma_D^-1 Sigma_DG (scale_model.md section 2).

    ``cohort_covariance`` is [columns, columns], the covariance of the block's
    calibrated columns D* in the fitted cohort: Sigma_D, which the cohort knows
    precisely. ``calibrated_pairs`` is [pairs, columns], the calibration samples'
    D* for every column of the block, complete, and ``target_truth`` is
    [pairs, targets], the truth of the columns being mapped, whose indices in the
    block are ``targets``. The pairs supply only what needs the truth: for each
    target, the cross-covariances s_k = X' R_k / n of the standardized columns
    with the calibrated residual R_k = T_k - D*_k, and its energy u_k = R_k' R_k / n.
    The regression R_k = X c_k + e_k, e_k of variance sigma_k^2, then has the
    likelihood of (s_k, u_k) with X'X / n replaced by the cohort correlation C, and
    the ridge prior c_k ~ N(0, (t / n) sigma_k^2 I), exchangeable in standardized
    units, gives the posterior mean c_k = (C + I / t)^-1 s_k and the profiled
    residual sigma_k^2(t) = u_k - s_k' (C + I / t)^-1 s_k. The ratio t is shared by
    the block's targets and maximizes the marginal likelihood of the n - 1 centred
    directions: t = 0 when the pairs show no leakage, the least-squares limit
    C^-1 s_k when the likelihood rises without bound in t, and, when the pairs'
    own correlation differs enough from the cohort's that a residual reaches 0
    first, the ratio where it does (the pairs are then fitted exactly). A column
    with no cohort variation gets coefficient 0, as does a target whose calibrated
    column already equals its truth on every pair.
    """
    covariance = np.asarray(cohort_covariance, dtype=np.float64)
    block = np.asarray(calibrated_pairs, dtype=np.float64)
    truth = np.asarray(target_truth, dtype=np.float64)
    target_index = np.asarray(targets, dtype=np.int64)
    columns = block.shape[1] if block.ndim == 2 else -1
    if covariance.shape != (columns, columns) or truth.ndim != 2 or truth.shape != (block.shape[0], target_index.shape[0]):
        raise ValueError("fit_leakage_map needs the cohort covariance [columns, columns], pairs [pairs, columns] and one truth column per target.")
    if not (np.all(np.isfinite(covariance)) and np.all(np.isfinite(block)) and np.all(np.isfinite(truth))):
        raise ValueError("fit_leakage_map needs a finite cohort covariance and complete calibration pairs.")
    if np.any((target_index < 0) | (target_index >= columns)) or np.unique(target_index).size != target_index.size:
        raise ValueError("fit_leakage_map needs distinct target columns inside the block.")
    pair_count = block.shape[0]
    column_means = block.mean(axis=0)
    coefficients = np.zeros((columns, target_index.shape[0]))
    centred = block - column_means
    residuals = (truth - truth.mean(axis=0)) - centred[:, target_index]
    deviations = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    varying = deviations > 0.0
    leaking = np.any(residuals != 0.0, axis=0)
    if not (np.any(varying) and np.any(leaking)):
        return LeakageMap(target_index, column_means, coefficients, 0.0)
    scale = deviations[varying]
    correlation = covariance[np.ix_(varying, varying)] / np.outer(scale, scale)
    eigenvalues, vectors = np.linalg.eigh(correlation)
    retained = eigenvalues > eigenvalues[-1] * np.finfo(np.float64).eps * correlation.shape[0]
    eigenvalues, vectors = eigenvalues[retained], vectors[:, retained]
    projected = vectors.T @ ((centred[:, varying] / scale).T @ residuals[:, leaking] / pair_count)
    energy = np.sum(residuals[:, leaking] ** 2, axis=0) / pair_count
    directions = pair_count - 1
    target_count = projected.shape[1]

    def gain(ratio: float) -> F64Array:
        """t / (1 + t lambda), which is 1 / lambda at t = infinity."""
        return 1.0 / eigenvalues if np.isinf(ratio) else ratio / (1.0 + ratio * eigenvalues)

    def residual_variance(ratio: float) -> F64Array:
        return energy - np.sum(projected**2 * gain(ratio)[:, None], axis=0)

    def score(ratio: float) -> float:
        shrink = 1.0 + ratio * eigenvalues
        explained_slope = np.sum(projected**2 / (shrink**2)[:, None], axis=0)
        return float((directions * np.sum(explained_slope / residual_variance(ratio)) - target_count * np.sum(eigenvalues / shrink)) / 2)

    if not score(0.0) > 0.0:
        return LeakageMap(target_index, column_means, coefficients, 0.0)
    # Every residual stays positive up to the limit: infinity, or the ratio where the
    # first residual reaches 0 when the pairs' correlation departs from the cohort's.
    limit = np.inf
    if not np.min(residual_variance(np.inf)) > 0.0:
        reach = 1.0 / eigenvalues[-1]
        while np.min(residual_variance(reach)) > 0.0:
            reach *= 2
        limit = _bisect_to_exhaustion(lambda value: float(np.min(residual_variance(value))), 0.0, reach)
    low, high = 0.0, 1.0 / eigenvalues[-1]
    while high < limit and score(high) > 0.0:
        low, high = high, high * 2
    ratio = limit if high >= limit else _bisect_to_exhaustion(score, low, high)
    standardized = vectors @ (projected * gain(ratio)[:, None])
    coefficients[np.ix_(varying, leaking)] = standardized / scale[:, None]
    return LeakageMap(target_index, column_means, coefficients, ratio, bool(np.isfinite(limit) and ratio == limit))


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
    """One block's records (store row indices), its mapped targets and its absorbed records (indices into ``records``).

    An LD block maps its imperfect columns onto the linear predictor of their true
    genotypes. A fusion block pairs an imputed record (the target) with a direct
    call of the same event from the same people's reads, e.g. a GATK-SV copy
    number (absorbed): the target becomes E_lin[G | D*, direct call], and the
    absorbed column, whose information the target now carries, leaves the fit
    (offset -inf, prior variance exactly 0) while staying an input of the map.
    """

    records: I64Array
    targets: I64Array
    absorbed: I64Array = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def __post_init__(self) -> None:
        targets, absorbed = np.asarray(self.targets), np.asarray(self.absorbed)
        if np.intersect1d(targets, absorbed).size:
            raise ValueError("a block's absorbed records cannot also be its targets.")


@dataclass(frozen=True)
class BlockPairs:
    """One LD block's calibration pairs for its leakage map.

    ``dosage`` is [pairs, block records]: the stored (uncalibrated) columns of every
    record in the block, and ``truth`` is [pairs, targets] for the block's targets.
    Only pairs complete across the block and its targets are kept.
    ``cohort_covariance`` is [block records, block records], the stored columns'
    covariance in the fitted cohort (Stage 0's block Gram over the sample count).
    """

    block: LdBlock
    dosage: F64Array
    truth: F64Array
    cohort_covariance: F64Array

    def __post_init__(self) -> None:
        records = np.asarray(self.block.records)
        targets = np.asarray(self.block.targets)
        dosage, truth = np.asarray(self.dosage), np.asarray(self.truth)
        if dosage.ndim != 2 or dosage.shape[1] != records.shape[0] or truth.shape != (dosage.shape[0], targets.shape[0]):
            raise ValueError("BlockPairs needs dosage [pairs, block records] and truth [pairs, targets].")
        if not (np.all(np.isfinite(dosage)) and np.all(np.isfinite(truth))):
            raise ValueError("BlockPairs needs pairs complete across the block and its targets.")
        if np.asarray(self.cohort_covariance).shape != (records.shape[0], records.shape[0]):
            raise ValueError("BlockPairs needs the cohort covariance of the block's records.")


@dataclass(frozen=True)
class CalibrationPairs:
    """What the measurement model uses from the samples carrying both a stored column and a truth genotype.

    ``moments`` are the per-record moments over those samples (store record order),
    from ``calibration_moments`` on record chunks joined by
    ``concatenate_calibration_moments``. ``blocks`` holds the pair-level data of the
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
    sample_ids: Sequence[ResearchId],
    dosage: NDArray,
    truth: NDArray,
    blocks: Sequence[LdBlock] = (),
    block_covariances: Sequence[NDArray] = (),
) -> CalibrationPairs:
    """Calibration pairs from dense [records, samples] arrays, NaN where missing (a benchmark's scale).

    ``block_covariances`` holds each block's cohort covariance of its stored columns.
    """
    dosage_values = np.asarray(dosage, dtype=np.float64)
    truth_values = np.asarray(truth, dtype=np.float64)
    if dosage_values.ndim != 2 or dosage_values.shape[1] != len(sample_ids):
        raise ValueError("calibration_pairs needs dosage and truth of shape [records, samples], one column per sample ID.")
    if len(block_covariances) != len(blocks):
        raise ValueError("calibration_pairs needs one cohort covariance per block.")
    block_pairs = []
    for block, block_covariance in zip(blocks, block_covariances):
        records = np.asarray(block.records, dtype=np.int64)
        target_records = records[np.asarray(block.targets, dtype=np.int64)]
        complete = np.all(np.isfinite(dosage_values[records]), axis=0) & np.all(np.isfinite(truth_values[target_records]), axis=0)
        block_pairs.append(
            BlockPairs(block, dosage_values[records][:, complete].T, truth_values[target_records][:, complete].T, np.asarray(block_covariance, dtype=np.float64))
        )
    return CalibrationPairs(tuple(sample_ids), calibration_moments(dosage_values, truth_values), tuple(block_pairs))


def _calibrated_block_covariance(pairs: BlockPairs, scales: NDArray) -> F64Array:
    block_scales = np.asarray(scales, dtype=np.float64)[np.asarray(pairs.block.records, dtype=np.int64)]
    return block_scales[:, None] * np.asarray(pairs.cohort_covariance, dtype=np.float64) * block_scales[None, :]


def fit_block_map(pairs: BlockPairs, scales: NDArray) -> LeakageMap:
    """One block's leakage map, with each column recalibrated by its record's scale, pairs and cohort alike."""
    block_scales = np.asarray(scales, dtype=np.float64)[np.asarray(pairs.block.records, dtype=np.int64)]
    dosage = np.asarray(pairs.dosage, dtype=np.float64)
    means = dosage.mean(axis=0)
    leakage = fit_leakage_map(_calibrated_block_covariance(pairs, scales), means + block_scales * (dosage - means), pairs.truth, pairs.block.targets)
    return replace(leakage, records=np.asarray(pairs.block.records, dtype=np.int64))


def engine_blocks(
    block_starts: NDArray, block_stops: NDArray, target_rows: NDArray, absorbed_rows: NDArray, absorbing_rows: NDArray
) -> tuple[LdBlock, ...]:
    """One LdBlock per engine LD block [start, stop) that holds a target, so every engine block has one A.

    ``target_rows`` are the store rows to map (imputed SV/TR records); ``absorbed_rows[i]``
    is a direct-call row fused into target ``absorbing_rows[i]``. The engine's sources
    read each block's own records, so a fused pair must lie inside one engine block:
    the store keeps them in one unbreakable group, and a pair that straddles a
    boundary is refused here rather than silently split.
    """
    starts = np.asarray(block_starts, dtype=np.int64)
    stops = np.asarray(block_stops, dtype=np.int64)
    targets = np.unique(np.asarray(target_rows, dtype=np.int64))
    absorbed = np.asarray(absorbed_rows, dtype=np.int64)
    absorbing = np.asarray(absorbing_rows, dtype=np.int64)
    if starts.shape != stops.shape or np.any(stops <= starts) or np.any(starts[1:] < stops[:-1]):
        raise ValueError("engine_blocks needs ordered, disjoint, non-empty blocks.")
    if absorbed.shape != absorbing.shape or not np.all(np.isin(absorbing, targets)):
        raise ValueError("every absorbed row needs the target row it is fused into.")

    def block_of(rows: I64Array) -> I64Array:
        index = np.searchsorted(stops, rows, side="right")
        inside = (index < starts.shape[0]) & (starts[np.minimum(index, starts.shape[0] - 1)] <= rows)
        if not np.all(inside):
            raise ValueError("a target or absorbed row lies outside every engine block.")
        return index

    if np.any(block_of(absorbed) != block_of(absorbing)):
        raise ValueError("a fused pair straddles an engine block boundary; keep its rows in one unbreakable group.")
    target_blocks = block_of(targets)
    absorbed_blocks = block_of(absorbed)
    blocks = []
    for index in np.unique(target_blocks).tolist():
        records = np.arange(starts[index], stops[index], dtype=np.int64)
        blocks.append(
            LdBlock(records, targets[target_blocks == index] - starts[index], absorbed[absorbed_blocks == index] - starts[index])
        )
    return tuple(blocks)


def mapped_signal_variances(pairs: BlockPairs, scales: NDArray, leakage: LeakageMap) -> F64Array:
    """Var(Xtilde_k) = a_k' Sigma_D* a_k in the fitted cohort, for each mapped target k (a_k: column k of A)."""
    transform = leakage_transform(leakage)[:, leakage.targets]
    return np.einsum("jk,jl,lk->k", transform, _calibrated_block_covariance(pairs, scales), transform)


@dataclass(frozen=True)
class MeasurementModel:
    """What the fit uses for its columns, offsets and predictive variance, and what was and wasn't applied.

    ``residual_variance`` is v_j = E[(G - D*_j)^2] per record (not per person), the
    predictive variance's measurement term. For a record without calibration pairs it is the value the
    reported r^2 implies for the unscaled column, Var(D) (1 - r^2) / r^2.
    """

    scales: F64Array
    residual_variance: F64Array
    log_reliability: F64Array
    leakage_maps: tuple[LeakageMap, ...]
    certificate: dict[str, object]

    def _arrays(self) -> dict[str, NDArray]:
        maps = self.leakage_maps
        return {
            "scales": self.scales,
            "residual_variance": self.residual_variance,
            "log_reliability": self.log_reliability,
            "map_records": np.concatenate([leakage.records for leakage in maps] or [np.zeros(0, np.int64)]),
            "map_record_counts": np.array([leakage.records.shape[0] for leakage in maps], dtype=np.int64),
            "map_targets": np.concatenate([leakage.targets for leakage in maps] or [np.zeros(0, np.int64)]),
            "map_target_counts": np.array([leakage.targets.shape[0] for leakage in maps], dtype=np.int64),
            "map_column_means": np.concatenate([leakage.column_means for leakage in maps] or [np.zeros(0)]),
            "map_coefficients": np.concatenate([leakage.coefficients.ravel() for leakage in maps] or [np.zeros(0)]),
            "map_ridge_ratios": np.array([leakage.ridge_ratio for leakage in maps], dtype=np.float64),
            "map_fits_pairs_exactly": np.array([leakage.fits_pairs_exactly for leakage in maps], dtype=bool),
            "certificate": np.array(json.dumps(self.certificate, sort_keys=True)),
        }

    def digest(self) -> str:
        """sha256 over every array and the certificate, in a fixed order: the artifact's provenance of the model."""
        hasher = hashlib.sha256()
        for name, values in sorted(self._arrays().items()):
            hasher.update(name.encode())
            hasher.update(np.ascontiguousarray(values).tobytes())
        return hasher.hexdigest()

    def save(self, path: str | Path) -> None:
        """Write the model as one npz (no pickled objects), readable by ``MeasurementModel.load``."""
        np.savez(Path(path), **self._arrays())

    @classmethod
    def load(cls, path: str | Path) -> MeasurementModel:
        with np.load(Path(path), allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        record_ends = np.cumsum(arrays["map_record_counts"])
        target_ends = np.cumsum(arrays["map_target_counts"])
        maps = []
        coefficient_start = 0
        for index in range(arrays["map_record_counts"].shape[0]):
            records = arrays["map_records"][record_ends[index] - arrays["map_record_counts"][index] : record_ends[index]]
            targets = arrays["map_targets"][target_ends[index] - arrays["map_target_counts"][index] : target_ends[index]]
            means = arrays["map_column_means"][record_ends[index] - records.shape[0] : record_ends[index]]
            size = records.shape[0] * targets.shape[0]
            coefficients = arrays["map_coefficients"][coefficient_start : coefficient_start + size].reshape(records.shape[0], targets.shape[0])
            coefficient_start += size
            maps.append(
                LeakageMap(targets, means, coefficients, float(arrays["map_ridge_ratios"][index]),
                           bool(arrays["map_fits_pairs_exactly"][index]), records)
            )
        return cls(arrays["scales"], arrays["residual_variance"], arrays["log_reliability"], tuple(maps),
                   json.loads(str(arrays["certificate"])))


def fit_measurement_model(
    calibration: CalibrationPairs | None,
    cohort_dosage_variance: NDArray,
    strata: NDArray,
    reported_reliability: NDArray,
    features: NDArray | None = None,
) -> MeasurementModel:
    """The measurement model for every stored record, from calibration pairs where there are any.

    ``cohort_dosage_variance`` is each stored column's variance in the fitted cohort,
    ``strata`` each record's pooling stratum (e.g. its variant class), and
    ``reported_reliability`` the imputation's own r^2 per record (INFO or DR2).
    Within a stratum kappa and lambda are pooled on the design [1, r^2_reported, log V]
    (``pooled_calibration``): on bench-sim v7 [semi-real, ~200 pairs per ancestry
    group] it cut the energy-weighted kappa error of TR and SV columns from
    0.11-0.17 (intercept only) to 0.08-0.10, against 0.14-0.22 for sqrt(DR2).
    A record with fewer than 3 calibration pairs, or in a stratum where no pair
    varies, gets no recalibration and its offset from the reported r^2; a column
    with no cohort variation carries no signal and gets offset -inf. The certificate counts both, and they are
    logged, never silent.
    """
    variance = np.asarray(cohort_dosage_variance, dtype=np.float64)
    labels = np.asarray(strata)
    reported = np.asarray(reported_reliability, dtype=np.float64)
    if variance.ndim != 1 or np.any(variance < 0.0) or labels.shape != variance.shape:
        raise ValueError("fit_measurement_model needs one nonnegative cohort variance and one stratum per record.")
    if reported.shape != variance.shape:
        raise ValueError("reported_reliability needs one r^2 per record.")
    undefined = ~((reported >= 0.0) & (reported <= 1.0))
    if np.any(undefined):
        # NaN passes a range test; an undefined r^2 must be replaced (e.g. by a reliability curve), never logged.
        raise ValueError(f"reported_reliability needs one r^2 in [0, 1] per record; {int(undefined.sum())} are outside it or undefined.")
    varying = variance > 0.0
    calibrated = np.zeros(variance.shape, dtype=bool)
    if calibration is not None:
        if calibration.moments.pair_counts.shape != variance.shape:
            raise ValueError("the calibration pairs and the cohort variances need the same records.")
        calibrated = (calibration.moments.pair_counts > 2) & varying
        energy = calibration.moments.pair_counts * calibration.moments.dosage_variance
        for stratum in np.unique(labels[calibrated]):
            members = calibrated & (labels == stratum)
            if not energy[members].sum() > 0.0:
                # No calibration pair varies anywhere in the stratum: it has nothing to pool.
                calibrated[members] = False
    uncalibrated = ~calibrated & varying
    scales = np.ones_like(variance)
    residual = np.zeros_like(variance)
    offsets = np.full_like(variance, -np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        offsets[uncalibrated] = np.log(reported[uncalibrated])
        residual[uncalibrated] = variance[uncalibrated] * (1.0 - reported[uncalibrated]) / reported[uncalibrated]
    maps: list[LeakageMap] = []
    absorbed = np.zeros(variance.shape, dtype=bool)
    extra = np.zeros((variance.shape[0], 0)) if features is None else np.asarray(features, dtype=np.float64)
    if extra.ndim != 2 or extra.shape[0] != variance.shape[0] or not np.all(np.isfinite(extra)):
        raise ValueError("features needs one finite row per record.")
    predicted = np.zeros(variance.shape, dtype=bool)
    if calibration is not None and np.any(calibrated):
        moments = calibration.moments.subset(calibrated)
        full_design = np.column_stack([np.ones(variance.shape[0]), reported, np.log(np.where(varying, variance, 1.0)), extra])
        design = full_design[calibrated]
        pooled = pooled_calibration(moments, variance[calibrated], labels[calibrated], design)
        if extra.shape[1]:
            # A record without its own pairs takes kappa(x) and lambda(x) from the records that have them.
            curve = fit_calibration_curve(moments, labels[calibrated], design)
            kappa, ratio = curve.predict(labels, full_design)
            predicted = uncalibrated & np.isfinite(kappa) & (kappa > 0.0) & (ratio > 0.0)
            scales[predicted] = kappa[predicted]
            residual[predicted] = variance[predicted] * np.maximum(ratio[predicted] - kappa[predicted] ** 2, 0.0)
            offsets[predicted] = _log_signal_share(kappa[predicted] ** 2 * variance[predicted], residual[predicted])
        scales[calibrated] = pooled.scales
        residual[calibrated] = residual_variances(variance[calibrated], pooled.scales, pooled.variance_ratios)
        offsets[calibrated] = log_reliability_offsets(variance[calibrated], scales[calibrated], residual[calibrated])
        genotype_variance = np.where(calibrated, 0.0, residual + scales**2 * variance)
        genotype_variance[calibrated] = pooled.variance_ratios * variance[calibrated]
        for pairs in calibration.blocks:
            leakage = fit_block_map(pairs, scales)
            maps.append(leakage)
            # A mapped target's signal is its mapped column's variance; the genotype variance is unchanged.
            rows = np.asarray(pairs.block.records, dtype=np.int64)
            target_rows = rows[np.asarray(pairs.block.targets, dtype=np.int64)]
            signal = mapped_signal_variances(pairs, scales, leakage)
            residual[target_rows] = np.maximum(genotype_variance[target_rows] - signal, 0.0)
            offsets[target_rows] = _log_signal_share(signal, residual[target_rows])
            absorbed[rows[np.asarray(pairs.block.absorbed, dtype=np.int64)]] = True
        offsets[absorbed] = -np.inf
    ratios = np.array([leakage.ridge_ratio for leakage in maps])
    certificate: dict[str, object] = {
        "calibration_samples": 0 if calibration is None else len(calibration.sample_ids),
        "calibrated_records": int(calibrated.sum()),
        "uncalibrated_records": int(uncalibrated.sum()),
        "records_without_cohort_variation": int((~varying).sum()),
        "direct_calls_fused": int(absorbed.sum()),
        "records_from_the_calibration_curve": int(predicted.sum()),
        "recalibration": (
            f"applied to {int(calibrated.sum())} records: kappa pooled by stratum on [1, reported r^2, log V] "
            f"from the calibration pairs; not applied to {int(uncalibrated.sum())} records with fewer than 3 pairs "
            "or no varying pair in their stratum"
        ),
        "leakage_correction": (
            f"applied to {len(maps)} LD blocks: no leakage found in {int(np.sum(ratios == 0.0))}, "
            f"at the least-squares limit in {int(np.sum(np.isinf(ratios)))}, "
            f"stopped where the pairs are fitted exactly in {sum(leakage.fits_pairs_exactly for leakage in maps)}"
            if maps
            else "not applied: no LD block has calibration pairs"
        ),
        "reliability_source": (
            f"calibration pairs for {int(calibrated.sum())} records; the imputation's reported r^2 "
            f"(biased for draw-type columns) for {int(uncalibrated.sum())} records; "
            f"-inf (no cohort variation) for {int((~varying).sum())} records"
        ),
    }
    for key in ("recalibration", "leakage_correction", "reliability_source", "direct_calls_fused"):
        log(f"measurement model: {key}: {certificate[key]}")
    return MeasurementModel(scales, residual, offsets, tuple(maps), certificate)


def within_group_moments(group_moments: Sequence[CalibrationMoments]) -> CalibrationMoments:
    """Each record's moments within its groups, pooled: sum_a n_a M_a / sum_a n_a for every central moment.

    Every group keeps its own mean (the recalibration is per group), so the pooled
    regression is the within-group one; the means are the pooled means, reported
    only for completeness.
    """
    counts = np.stack([moments.pair_counts for moments in group_moments]).astype(np.float64)
    total = counts.sum(axis=0)

    def pooled(name: str) -> F64Array:
        values = np.stack([getattr(moments, name) for moments in group_moments])
        return np.divide((counts * values).sum(axis=0), total, out=np.zeros_like(total), where=total > 0)

    return CalibrationMoments(
        pair_counts=total.astype(np.int64),
        **{field.name: pooled(field.name) for field in fields(CalibrationMoments) if field.name != "pair_counts"},
    )


def fit_ancestry_measurement_models(
    calibrations: Sequence[CalibrationPairs | None],
    cohort_dosage_variance: NDArray,
    strata: NDArray,
    reported_reliability: NDArray,
) -> tuple[MeasurementModel, ...]:
    """One model per ancestry group, each group's kappa pooled toward the record's all-group kappa by EB.

    ``calibrations[a]`` holds group a's truth pairs (None when it has none) and
    ``cohort_dosage_variance`` is [records, groups]. A group's own ~n_a pairs give
    each record a noisy slope, so kappa_{j,a} ~ N(kappa_j, tau^2): kappa_j is the
    record's pooled within-group kappa (``pooled_calibration`` on
    ``within_group_moments``, the design [1, reported r^2, log V] per stratum), and
    tau^2, the between-ancestry variance of a stratum, is the energy-weighted moment
    estimate from the groups' slopes and their robust sampling variances at kappa_j.
    Each group keeps its own lambda (its own genotype variance), so v and the offsets
    follow from its pooled kappa. A group without pairs gets kappa_j itself.
    """
    variance = np.asarray(cohort_dosage_variance, dtype=np.float64)
    labels = np.asarray(strata)
    reported = np.asarray(reported_reliability, dtype=np.float64)
    if variance.ndim != 2 or variance.shape[1] != len(calibrations):
        raise ValueError("fit_ancestry_measurement_models needs cohort variances [records, groups], one column per group.")
    models = [fit_measurement_model(calibration, variance[:, group], labels, reported) for group, calibration in enumerate(calibrations)]
    present = [group for group, calibration in enumerate(calibrations) if calibration is not None]
    if not present:
        return tuple(models)
    group_moments = [calibrations[group].moments for group in present]  # type: ignore[union-attr]
    combined = within_group_moments(group_moments)
    counts = np.stack([moments.pair_counts for moments in group_moments])
    combined_variance = np.divide((counts * variance[:, present].T).sum(axis=0), counts.sum(axis=0),
                                  out=variance[:, present].mean(axis=1), where=counts.sum(axis=0) > 0)
    calibrated = (combined.pair_counts > 2) & (combined.dosage_variance > 0.0) & (combined_variance > 0.0)
    if not np.any(calibrated):
        return tuple(models)
    design = np.column_stack([np.ones(int(calibrated.sum())), reported[calibrated], np.log(combined_variance[calibrated])])
    shared = np.ones_like(reported)
    shared_ratio = np.ones_like(reported)
    combined_fit = pooled_calibration(combined.subset(calibrated), combined_variance[calibrated], labels[calibrated], design)
    shared[calibrated], shared_ratio[calibrated] = combined_fit.scales, combined_fit.variance_ratios
    pooled_models = list(models)
    between_by_stratum: dict[str, float] = {}
    for stratum in np.unique(labels[calibrated]):
        members = calibrated & (labels == stratum)
        slopes, energies, residuals = [], [], []
        for moments in group_moments:
            energy = moments.pair_counts[members] * moments.dosage_variance[members]
            informative = energy > 0.0
            prior = shared[members]
            slopes.append(np.divide(moments.pair_counts[members] * moments.covariance[members], energy,
                                    out=prior.copy(), where=informative))
            residuals.append(np.maximum(np.divide(
                moments.dosage_squared_truth_squared[members] - 2.0 * prior * moments.dosage_cubed_truth[members]
                + prior**2 * moments.dosage_fourth[members],
                moments.dosage_variance[members], out=np.zeros_like(energy), where=informative), 0.0))
            energies.append(energy)
        slope, energy, residual = np.stack(slopes), np.stack(energies), np.stack(residuals)
        excess = np.sum(energy * (slope - shared[members]) ** 2)
        between = max(float((excess - residual[energy > 0].sum()) / energy.sum()), 0.0) if energy.sum() > 0 else 0.0
        between_by_stratum[str(stratum)] = between
        with np.errstate(divide="ignore", invalid="ignore"):
            weight = np.where(residual > 0.0, between * energy / (between * energy + residual), 1.0)
        weight = np.where(energy > 0.0, weight, 0.0)
        for position, group in enumerate(present):
            rows = np.flatnonzero(members)
            model = pooled_models[group]
            kappa = np.maximum(shared[members] + weight[position] * (slope[position] - shared[members]), 0.0)
            # The group's own genotype variance where its model calibrated the record, else lambda_j V_a.
            own_fit = (group_moments[position].pair_counts[rows] > 2) & (group_moments[position].dosage_variance[rows] > 0.0)
            genotype_variance = np.where(
                own_fit,
                model.residual_variance[rows] + model.scales[rows] ** 2 * variance[rows, group],
                shared_ratio[rows] * variance[rows, group],
            )
            scales = model.scales.copy()
            residual_variance = model.residual_variance.copy()
            offsets = model.log_reliability.copy()
            scales[rows] = kappa
            residual_variance[rows] = np.maximum(genotype_variance - kappa**2 * variance[rows, group], 0.0)
            offsets[rows] = _log_signal_share(kappa**2 * variance[rows, group], residual_variance[rows])
            pooled_models[group] = MeasurementModel(scales, residual_variance, offsets, model.leakage_maps, model.certificate)
    for group in range(len(calibrations)):
        model = pooled_models[group]
        if group not in present:
            # No pairs of its own: kappa_j and lambda_j borrowed from the other groups.
            borrowed = calibrated & (variance[:, group] > 0.0)
            scales, residual_variance, offsets = model.scales.copy(), model.residual_variance.copy(), model.log_reliability.copy()
            scales[borrowed] = shared[borrowed]
            residual_variance[borrowed] = variance[borrowed, group] * np.maximum(shared_ratio[borrowed] - shared[borrowed] ** 2, 0.0)
            offsets[borrowed] = _log_signal_share(shared[borrowed] ** 2 * variance[borrowed, group], residual_variance[borrowed])
            pooled_models[group] = MeasurementModel(scales, residual_variance, offsets, model.leakage_maps, model.certificate)
        certificate = dict(pooled_models[group].certificate)
        certificate["ancestry_pooling"] = {
            "records": int(calibrated.sum()),
            "groups_with_pairs": len(present),
            "between_ancestry_variance_by_stratum": between_by_stratum,
        }
        pooled_models[group] = replace(pooled_models[group], certificate=certificate)
    log(f"measurement model: ancestry pooling over {len(present)} groups; between-ancestry kappa variance by stratum {between_by_stratum}")
    return tuple(pooled_models)


def pooled_measurement_model(
    models: Sequence[MeasurementModel],
    cohort_dosage_mean: NDArray,
    cohort_dosage_variance: NDArray,
    group_counts: NDArray,
    blocks: Sequence[BlockPairs] = (),
) -> MeasurementModel:
    """The one model the fit uses over store records, from one model per ancestry group.

    Each group's kappa is applied in the store, so the fit's columns are D* and the
    pooled ``scales`` are 1. ``cohort_dosage_mean`` and ``cohort_dosage_variance``
    are [records, groups], the stored (uncalibrated) columns' moments in each
    group's fitted samples, and ``group_counts`` the fitted samples per group. Each
    recalibration keeps its group's mean, so D*'s cohort variance is
    sum_g w_g (kappa_g^2 V_g + (mu_g - mu)^2) and its residual variance
    sum_g w_g v_g (w_g = n_g / n); the offsets are ``pooled_log_reliability``.
    The maps are fitted here, once over the pooled cohort, not per group: a linear
    predictor is not an average of per-group predictors. Each of ``blocks`` holds
    the truth pairs of every group with their stored D* columns (already
    recalibrated, so the map applies no scale) and the pooled cohort's covariance
    of those D* columns. A mapped target's offset is its mapped column's share of
    the pooled genotype variance; an absorbed direct call gets -inf.
    """
    if not models:
        raise ValueError("pooled_measurement_model needs one model per ancestry group.")
    scales = np.column_stack([model.scales for model in models])
    residuals = np.column_stack([model.residual_variance for model in models])
    means = np.asarray(cohort_dosage_mean, dtype=np.float64)
    variances = np.asarray(cohort_dosage_variance, dtype=np.float64)
    counts = np.asarray(group_counts, dtype=np.float64)
    if means.shape != scales.shape or variances.shape != scales.shape or counts.shape != (scales.shape[1],):
        raise ValueError("pooled_measurement_model needs [records, groups] moments and one count per model.")
    offsets = pooled_log_reliability(scales, residuals, means, variances, counts)
    weights = counts / counts.sum()
    fitted = weights > 0.0

    def weighted(values: F64Array) -> F64Array:
        return (np.where(fitted, values, 0.0) * weights).sum(axis=1)

    centre = weighted(means)
    signal = weighted(scales**2 * variances + (means - centre[:, None]) ** 2)
    residual = weighted(residuals)
    genotype_variance = signal + residual
    ones = np.ones(scales.shape[0])
    maps: list[LeakageMap] = []
    absorbed = np.zeros(scales.shape[0], dtype=bool)
    for pairs in blocks:
        leakage = fit_block_map(pairs, ones)
        maps.append(leakage)
        rows = np.asarray(pairs.block.records, dtype=np.int64)
        target_rows = rows[np.asarray(pairs.block.targets, dtype=np.int64)]
        mapped = mapped_signal_variances(pairs, ones, leakage)
        residual[target_rows] = np.maximum(genotype_variance[target_rows] - mapped, 0.0)
        offsets[target_rows] = _log_signal_share(mapped, residual[target_rows])
        absorbed[rows[np.asarray(pairs.block.absorbed, dtype=np.int64)]] = True
    offsets[absorbed] = -np.inf
    ratios = np.array([leakage.ridge_ratio for leakage in maps])
    certificate: dict[str, object] = {
        "groups": [model.certificate for model in models],
        "group_counts": [int(count) for count in counts],
        "scales": "kappa applied per ancestry group in the store; the fit's columns are D*",
        "reliability_source": "each group's model, pooled over the fitted cohort's groups",
        "leakage_correction": (
            f"applied to {len(maps)} blocks over the pooled cohort: no leakage found in {int(np.sum(ratios == 0.0))}"
            if maps
            else "not applied: no block has calibration pairs"
        ),
        "direct_calls_fused": int(absorbed.sum()),
    }
    for key in ("leakage_correction", "direct_calls_fused"):
        log(f"pooled measurement model: {key}: {certificate[key]}")
    return MeasurementModel(ones, residual, offsets, tuple(maps), certificate)
