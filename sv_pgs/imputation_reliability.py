"""Imputation reliability of a stored genotype column, and the unit contract that carries it to the prior.

r^2 = corr^2(D, G) between a stored column D and the true genotype G is fixed from truth outside the fit (or taken
from the imputation's own report). How it reaches the prior depends on what D is, which ``ColumnMeasurement`` states
per record: the prior is on raw effects (per ALT allele, per copy), and its standardized form has variance
kappa^2 Var(D) tau^2 = r^2 Var(G) tau^2 with kappa = Cov(G, D) / Var(D) the calibration slope. For a calibrated
conditional mean (kappa = 1) the reliability is already in Var(D) and enters only the true genotype's variance
Var(G) = Var(D) / r^2, the frequency function's argument; a prior offset of log r^2 on top of the per-unit
log Var(D) counted it twice (review F21). r^2, a squared correlation, is invariant to affine maps of D: the codec's
scale, centring and the recalibration below leave it alone. A non-affine map does change it, so the shape below is
part of the column the reliability is measured on, never applied after it. The pieces here:

- The triad estimator. A long-read truth T = G + e carries its own error, which
  attenuates corr(D, T) below corr(D, G), so corr^2(D, T) understates r^2. Two
  truths with errors independent of each other and of D identify it exactly:
  r^2 = r(D,T1) r(D,T2) / r(T1,T2).
- The per-record reliability model. It predicts r^2 from sites-only features
  on the logit scale and is fitted once on truth; the fit reads log r^2 through
  ``ColumnMeasurement``.
- The monotone calibration curve. Where the imputed dosage is not a calibrated
  posterior mean (confident-draw SV/TR dosages, deflated multi-path alleles),
  E[G | D] is not D, and D* = scale h(D) restores it. The shape h is the
  isotonic regression of a truth on D, which estimates the regression function
  E[T | D]; a truth's own error attenuates its correlation with D, never that
  regression function, so a truth on the genotype's scale identifies E[G | D]
  outright and the scale is 1 (measurement_model.py's linear kappa is the same
  statement for the linear map). The scale is there for a truth on an unknown
  scale, E[T | G] = lambda G: then E[T | D] = lambda E[G | D] and the scale is
  1 / lambda, which the triad correlation identifies without knowing lambda.
- ``ColumnMeasurement``: raw unit, encoding scale, calibration slope and
  reliability with its source, from store metadata to the prior's offset and
  frequency argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sv_pgs._typing import F64Array, NDArray


def checked_log_reliability(values: NDArray, source: str) -> F64Array:
    """The records' log reliabilities log r^2, against the one contract they all meet.

    Every source of a record's reliability ends here: this module's fitted model,
    ``measurement_model.log_reliability_offsets`` on truth pairs, a store's reported
    imputation r^2, and a caller's explicit offsets. The contract is r^2 in [0, 1],
    so the value is at most 0, and -inf is the record whose stored column carries
    no information about its genotype: its standardized prior variance
    r^2 Var(G) tau^2 is 0, its effect is exactly zero, and the fit leaves it out.

    Anything else is corrupt metadata rather than a measurement, and raises here
    instead of reaching the prior. A value above 0 is no squared correlation; a nan would be dropped
    by candidate selection, which tests offsets for finiteness, so a record with
    unreadable metadata would pass for one that was measured and found empty.
    """
    offsets = np.asarray(values, dtype=np.float64)
    invalid = ~(offsets <= 0.0)
    if invalid.any():
        first = int(np.flatnonzero(invalid)[0])
        raise ValueError(
            f"{source}: every record's log reliability must be <= 0, a log r^2 with r^2 in [0, 1]; "
            f"record {first} has {offsets[first]} ({int(invalid.sum())} of {offsets.size} records)"
        )
    return offsets


def triad_squared_correlation(dosage: NDArray, truth_a: NDArray, truth_b: NDArray) -> float:
    """r^2 = corr^2(D, G) from two truths whose errors are mutually independent.

    Samples missing in any of the three arrays are dropped. Every way the
    identity can return something that is not a squared correlation raises
    instead, because the caller's use of the value is a prior variance factor
    and its logarithm is the prior's offset:

    - a constant column over the shared samples has no correlation, and the
      ratio is NaN, not a reliability of nan;
    - two truths that do not share signal make the ratio a quotient of noise;
    - a ratio outside [0, 1] is not a squared correlation. It says this sample
      contradicts the identity's assumption of independent truth errors, or is
      too small to resolve the ratio. Clipping it would hand the prior a
      reliability of exactly 1 or 0, which the record has not earned; it is for
      the caller to pool the record or drop it.
    """
    values = np.vstack([np.asarray(dosage, float), np.asarray(truth_a, float), np.asarray(truth_b, float)])
    observed = np.isfinite(values).all(axis=0)
    shared = int(observed.sum())
    if shared < 3:
        raise ValueError("the triad needs at least 3 samples observed in all three columns")
    present = values[:, observed]
    for name, spread in zip(("the dosage", "the first truth", "the second truth"), present.var(axis=1)):
        if not spread > 0.0:
            raise ValueError(f"{name} is constant over the {shared} shared samples, so it has no correlation")
    correlation = np.corrcoef(present)
    truth_truth = correlation[1, 2]
    if not truth_truth > 0.0:
        raise ValueError(f"the two truths are not positively correlated (r = {truth_truth:.4f})")
    squared = float(correlation[0, 1] * correlation[0, 2] / truth_truth)
    if not 0.0 <= squared <= 1.0:
        raise ValueError(
            f"the triad gives r^2 = {squared:.4f}, which is not a squared correlation: r(D,T1) = "
            f"{correlation[0, 1]:.4f}, r(D,T2) = {correlation[0, 2]:.4f} and r(T1,T2) = {truth_truth:.4f} "
            f"over {shared} shared samples contradict the identity's independent truth errors"
        )
    return squared


def _pool_adjacent_violators(values: F64Array, weights: F64Array) -> F64Array:
    """Weighted non-decreasing least-squares fit of values, in their given order."""
    block_value: list[float] = []
    block_weight: list[float] = []
    block_length: list[int] = []
    for value, weight in zip(values, weights):
        block_value.append(float(value))
        block_weight.append(float(weight))
        block_length.append(1)
        while len(block_value) > 1 and block_value[-2] > block_value[-1]:
            merged_weight = block_weight[-2] + block_weight[-1]
            merged_value = (block_value[-2] * block_weight[-2] + block_value[-1] * block_weight[-1]) / merged_weight
            merged_length = block_length[-2] + block_length[-1]
            del block_value[-1], block_weight[-1], block_length[-1]
            block_value[-1], block_weight[-1], block_length[-1] = merged_value, merged_weight, merged_length
    return np.repeat(np.asarray(block_value), block_length)


@dataclass(frozen=True)
class CalibrationCurve:
    """Monotone recalibration D* = scale h(D) for one stratum.

    The shape h is a free monotone function of the stored column, so a shift of
    the column is already in h and the map needs no centre of its own. A centre
    would be a second one: the shape's own mean is the truth's, lambda E[G], not
    the genotype's, and using it for both sends the genotype [0, 1, 2] measured
    by a doubled truth to [1, 2, 3] instead of back to [0, 1, 2].
    """

    stratum: str
    version: str
    knots_dosage: F64Array
    knots_expectation: F64Array
    scale: float

    def shape(self, dosage: NDArray) -> F64Array:
        return np.interp(np.asarray(dosage, float), self.knots_dosage, self.knots_expectation)

    def apply(self, dosage: NDArray) -> F64Array:
        return self.scale * self.shape(dosage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stratum": self.stratum,
            "version": self.version,
            "knots_dosage": self.knots_dosage.tolist(),
            "knots_expectation": self.knots_expectation.tolist(),
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> CalibrationCurve:
        return cls(
            stratum=str(record["stratum"]),
            version=str(record["version"]),
            knots_dosage=np.asarray(record["knots_dosage"], float),
            knots_expectation=np.asarray(record["knots_expectation"], float),
            scale=float(record["scale"]),
        )


def fit_calibration_shape(dosage: NDArray, truth: NDArray, stratum: str, version: str) -> CalibrationCurve:
    """Isotonic E[T | D] on the distinct dosage values, with unit scale.

    Unit scale is the whole map for a truth on the genotype's scale, where
    E[T | D] is E[G | D] already: the truth's error attenuates corr(D, T), not
    E[T | D]. With E[T | G] = lambda G the shape carries lambda; set the scale
    with calibrated_scale before applying.
    """
    dosage_values = np.asarray(dosage, float).ravel()
    truth_values = np.asarray(truth, float).ravel()
    observed = np.isfinite(dosage_values) & np.isfinite(truth_values)
    knots, inverse, counts = np.unique(dosage_values[observed], return_inverse=True, return_counts=True)
    if knots.size < 2:
        raise ValueError(f"stratum {stratum}: the calibration shape needs at least 2 distinct dosage values")
    truth_sums = np.bincount(inverse, weights=truth_values[observed], minlength=knots.size)
    expectation = _pool_adjacent_violators(truth_sums / counts, counts.astype(float))
    return CalibrationCurve(stratum, version, knots, expectation, 1.0)


def calibrated_scale(shape_truth_correlation: float, genotype_sd: float, shape_sd: float) -> float:
    """Scale b = r(h, G) sd(G) / sd(h) for D* = b h(D).

    b is the least-squares slope of G on the shaped column, Cov(G, h) / Var(h),
    so D* satisfies the linear identity Cov(G, D*) = Var(D*) for any shape. That
    identity is one orthogonality condition, not calibration at every dosage:
    the linear recalibration of a column whose E[G | D] is curved satisfies it
    and still misses E[G | D] by O(1) in the tails. Calibration at every dosage,
    E[G | D*] = D*, follows when h is the conditional mean E[T | D] of a truth
    with E[T | G] = lambda G, since then h = lambda E[G | D] and b = 1 / lambda.

    r(h, G) is the triad correlation of the shaped column with the true
    genotype, which is free of lambda, never the slope of one truth on h: that
    slope is lambda b, carrying the very scale b has to remove. A truth's noise
    does not enter either estimate, because it attenuates a correlation of D
    with the truth, not a regression on D.
    """
    if not (0.0 < shape_truth_correlation <= 1.0 and genotype_sd > 0.0 and shape_sd > 0.0):
        raise ValueError("calibrated_scale needs 0 < r <= 1 and positive standard deviations")
    return shape_truth_correlation * genotype_sd / shape_sd


@dataclass(frozen=True)
class ReliabilityModel:
    """Per-record r^2 = sigmoid(intercept + x'coefficients), fitted once on truth."""

    version: str
    feature_names: tuple[str, ...]
    intercept: float
    coefficients: F64Array

    def _linear_predictor(self, features: NDArray) -> F64Array:
        design = np.asarray(features, float)
        if design.ndim != 2 or design.shape[1] != len(self.feature_names):
            raise ValueError(f"expected features of shape (records, {len(self.feature_names)}), got {design.shape}")
        return self.intercept + design @ self.coefficients

    def predict_r2(self, features: NDArray) -> F64Array:
        return np.exp(self.log_reliability_offset(features))

    def log_reliability_offset(self, features: NDArray) -> F64Array:
        """log r^2 = -log(1 + exp(-eta)), the prior-variance offset; finite for every finite eta."""
        return -np.logaddexp(0.0, -self._linear_predictor(features))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "feature_names": list(self.feature_names),
            "intercept": self.intercept,
            "coefficients": self.coefficients.tolist(),
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> ReliabilityModel:
        coefficients = np.asarray(record["coefficients"], float)
        feature_names = tuple(str(name) for name in record["feature_names"])
        if coefficients.shape != (len(feature_names),):
            raise ValueError("reliability model coefficients do not match its feature names")
        return cls(
            version=str(record["version"]),
            feature_names=feature_names,
            intercept=float(record["intercept"]),
            coefficients=coefficients,
        )


@dataclass(frozen=True)
class ColumnMeasurement:
    """The unit and measurement contract of the prior's columns: what one stored column D_j measures, in which unit.

    The prior is stated in raw units first. The effect b_j of one unit of the true genotype G_j (one ALT allele of a
    dosage record, one copy of a copy-number record: ``raw_unit``) has prior variance tau_j^2 = e^(t + a_j), a draw t of
    the class's mixing density and a_j the learned log-variance terms (the frequency function h(log Var G_j), the
    annotation smooths). The fit sees the column standardized, x_j = (D_j - mean) / sd(D_j), with D_j = code /
    ``codes_per_unit`` in raw units. Where the conditional mean of G_j given D_j is linear with slope
    kappa_j = Cov(G_j, D_j) / Var(D_j) (``calibration_slope``), E[y | D] carries kappa_j b_j per unit of D_j, so the
    coefficient on x_j is beta_j = kappa_j sd(D_j) b_j and

        Var(beta_j) = kappa_j^2 Var(D_j) tau_j^2 = Corr(G_j, D_j)^2 Var(G_j) tau_j^2

    (the two forms agree through r_j^2 = Corr(G, D)^2 = kappa_j^2 Var(D_j) / Var(G_j)). So the prior's offset, with
    coefficient exactly 1, is log(kappa_j^2 Var(D_j)), and the true genotype's variance, the argument of the frequency
    function, is Var(G_j) = kappa_j^2 Var(D_j) / r_j^2.

    A calibrated conditional mean, D = E[G | O] for the evidence O (a GLIMPSE2 or Beagle DS, the measurement step's
    recalibrated D*, a hard call measured exactly), has Cov(G, D) = Var(D), so kappa = 1: its reduced variance already
    carries the lost information, r^2 = Var(D) / Var(G), and multiplying the prior by r^2 again would shrink a poorly
    measured column twice (Var(beta) = r^2 Var(D) tau^2 = r^4 Var(G) tau^2; review F21). The reliability instead enters
    through Var(G), the frequency function's argument. A deliberately uncalibrated column (a confident draw, whose
    variance stays near Var(G)) has kappa^2 = r^2 Var(G) / Var(D) < 1 and takes it here, measured from truth pairs
    (``measurement_model.fit_measurement_model``'s scales), never guessed from its r^2.

    Measured on bench-real's Beagle-imputed SV overlay (chr16, 637 varying records, the panel's calls as truth [real
    genotypes]): the energy-weighted kappa of the stored DS is 0.87 (median 0.81), and among records with true
    r^2 > 0.1 the target log(Corr^2 Var G) regressed on log Var(D) and log DR2 has coefficients 0.97 and -1.55: the
    calibrated-mean offset log Var(D) tracks it (RMS error 1.07 in log variance), the doubly counted log DR2 + log Var(D)
    does not (1.44). Var(D) / DR2 estimates Var(G) with median log error -0.03 (RMS 1.21).

    ``log_reliability`` is log r_j^2 (``checked_log_reliability``: at most 0, -inf where the column carries no
    information), and ``reliability_source`` says where it came from (a store's reported imputation r^2, truth pairs,
    or none: measured exactly), recorded with the fit.
    """

    raw_unit: NDArray
    codes_per_unit: F64Array
    calibration_slope: F64Array
    log_reliability: F64Array
    reliability_source: str

    @classmethod
    def build(
        cls, record_count: int, *, log_reliability: NDArray | None = None, codes_per_unit: NDArray | None = None,
        calibration_slope: NDArray | None = None, raw_unit: NDArray | None = None, reliability_source: str | None = None,
    ) -> ColumnMeasurement:
        """The contract of ``record_count`` stored columns. Absent inputs are the store's defaults and say so: no
        reliability is measurement without error (r^2 = 1), no encoding scale is the store's 127 codes per ALT
        allele, no calibration slope is a calibrated conditional mean (kappa = 1)."""
        from sv_pgs.dosage_store import CODES_PER_DOSAGE  # noqa: PLC0415 - the store imports nothing from here

        def per_record(values: NDArray | None, default: float, name: str) -> F64Array:
            array = np.full(record_count, default) if values is None else np.asarray(values, dtype=np.float64)
            if array.shape != (record_count,):
                raise ValueError(f"{name} needs one value per record ({record_count}), got shape {array.shape}")
            return array

        reliability = per_record(log_reliability, 0.0, "log_reliability")
        checked_log_reliability(reliability, reliability_source or "the column measurement's log reliability")
        units = per_record(codes_per_unit, float(CODES_PER_DOSAGE), "codes_per_unit")
        slope = per_record(calibration_slope, 1.0, "calibration_slope")
        if not (np.all(units > 0.0) and np.all(np.isfinite(units))):
            raise ValueError("every record needs a positive, finite encoding scale (codes per raw unit)")
        if not (np.all(slope > 0.0) and np.all(np.isfinite(slope))):
            raise ValueError("every calibration slope must be positive and finite: a column whose conditional mean does not rise with it is not a measurement of its genotype")
        labels = np.full(record_count, "ALT allele", dtype=object) if raw_unit is None else np.asarray(raw_unit, dtype=object)
        if labels.shape != (record_count,):
            raise ValueError("raw_unit needs one label per record")
        source = reliability_source or ("none: every record measured exactly" if log_reliability is None else "the caller's log reliability")
        return cls(raw_unit=labels, codes_per_unit=units, calibration_slope=slope, log_reliability=reliability, reliability_source=source)

    def standardized_terms(self, members: NDArray, code_scales: NDArray) -> tuple[F64Array, F64Array]:
        """(offset, log genotype variance) of the prior's ``members`` (record indices) from their training code SDs.

        The offset is log(kappa_j^2 Var(D_j)) less its largest over the members: a constant shift of every member's
        log prior variance, which the class densities' common location carries, so every offset stays at or below 0.
        The log genotype variance log Var(G_j) = log(kappa_j^2 Var(D_j)) - log r_j^2 is the frequency function's
        argument (``annotation_design.frequency_annotation``)."""
        rows = np.asarray(members, dtype=np.int64)
        scales = np.asarray(code_scales, dtype=np.float64) / self.codes_per_unit[rows]
        if scales.shape != rows.shape or not np.all(scales > 0.0):
            raise ValueError("every member needs a positive training spread")
        log_signal = 2.0 * (np.log(self.calibration_slope[rows]) + np.log(scales))
        offset = log_signal - float(np.max(log_signal)) if rows.size else log_signal
        return offset, log_signal - self.log_reliability[rows]
