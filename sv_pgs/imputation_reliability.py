"""Imputation reliability of a stored genotype column, for the r^2-scaled prior.

SPEC scales each variant's prior variance by r^2 = corr^2(D, G) between its
stored column D and the true genotype G, with r^2 fixed from long-read truth
outside the fit. The prior it offsets is the one on the coefficient per SD of
the stored column, so log r^2 enters log u_j with coefficient exactly 1
(docs/design/math/scale_model.md section 1), and r^2, a squared correlation, is
invariant to affine maps of D: the codec's scale, centring and the recalibration
below leave the offset alone. A non-affine map does change it, so the shape
below is part of the column the reliability is measured on, never applied after
it. Three pieces live here:

- The triad estimator. A long-read truth T = G + e carries its own error, which
  attenuates corr(D, T) below corr(D, G), so corr^2(D, T) understates r^2. Two
  truths with errors independent of each other and of D identify it exactly:
  r^2 = r(D,T1) r(D,T2) / r(T1,T2).
- The per-record reliability model. It predicts r^2 from sites-only features
  on the logit scale and is fitted once on truth; the fit supplies the prior
  offset log r^2, whose coefficient is exactly 1 by derivation (the prior on the
  true-genotype effect maps to the stored column through r^2).
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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sv_pgs._typing import F64Array, NDArray


def checked_log_reliability(values: NDArray, source: str) -> F64Array:
    """The records' prior log-variance offsets log r^2, against the one contract they all meet.

    Every source of a record's reliability ends here: this module's fitted model,
    ``measurement_model.log_reliability_offsets`` on truth pairs, a store's reported
    imputation r^2, and a caller's explicit offsets. The contract is r^2 in [0, 1],
    so the offset is at most 0, and -inf is the record whose stored column carries
    no information about its genotype: its prior variance is r^2 x (its class's) = 0,
    its effect is exactly zero, and the fit leaves it out.

    Anything else is corrupt metadata rather than a measurement, and raises here
    instead of reaching the prior. An offset above 0 would inflate a record's prior
    variance above its class's, which no reliability can do; a nan would be dropped
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
