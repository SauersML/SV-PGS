"""Imputation reliability of a stored genotype column, for the r^2-scaled prior.

SPEC scales each variant's prior variance by r^2 = corr^2(D, G) between its
stored column D and the true genotype G, with r^2 fixed from long-read truth
outside the fit. Three pieces live here:

- The triad estimator. A long-read truth T = G + e carries its own error, so
  corr^2(D, T) understates r^2. Two truths with errors independent of each
  other and of D identify it exactly: r^2 = r(D,T1) r(D,T2) / r(T1,T2).
- The per-record reliability model. It predicts r^2 from sites-only features
  on the logit scale and is fitted once on truth; the fit supplies the prior
  offset log r^2 (entering with an EB-learned coefficient centred at 1).
- The monotone calibration curve. Where the imputed dosage is not a calibrated
  posterior mean (confident-draw SV/TR dosages, deflated multi-path alleles),
  D* = centre + scale (h(D) - centre) restores E[G | D*] = D*. Its shape h is
  fitted on truth by isotonic regression; its scale comes from the triad r,
  because a noisy truth identifies the shape of E[G | D] but not its scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sv_pgs._typing import F64Array, NDArray


def triad_squared_correlation(dosage: NDArray, truth_a: NDArray, truth_b: NDArray) -> float:
    """r^2 = corr^2(D, G) from two truths whose errors are mutually independent.

    Samples missing in any of the three arrays are dropped. Fails loudly when
    the two truths do not share signal, since the identity then has no value.
    """
    values = np.vstack([np.asarray(dosage, float), np.asarray(truth_a, float), np.asarray(truth_b, float)])
    observed = np.isfinite(values).all(axis=0)
    if observed.sum() < 3:
        raise ValueError("the triad needs at least 3 samples observed in all three columns")
    correlation = np.corrcoef(values[:, observed])
    truth_truth = correlation[1, 2]
    if not truth_truth > 0.0:
        raise ValueError(f"the two truths are not positively correlated (r = {truth_truth:.4f})")
    return float(correlation[0, 1] * correlation[0, 2] / truth_truth)


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
    """Monotone recalibration D* = centre + scale (h(D) - centre) for one stratum."""

    stratum: str
    version: str
    knots_dosage: F64Array
    knots_expectation: F64Array
    centre: float
    scale: float

    def shape(self, dosage: NDArray) -> F64Array:
        return np.interp(np.asarray(dosage, float), self.knots_dosage, self.knots_expectation)

    def apply(self, dosage: NDArray) -> F64Array:
        return self.centre + self.scale * (self.shape(dosage) - self.centre)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stratum": self.stratum,
            "version": self.version,
            "knots_dosage": self.knots_dosage.tolist(),
            "knots_expectation": self.knots_expectation.tolist(),
            "centre": self.centre,
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> CalibrationCurve:
        return cls(
            stratum=str(record["stratum"]),
            version=str(record["version"]),
            knots_dosage=np.asarray(record["knots_dosage"], float),
            knots_expectation=np.asarray(record["knots_expectation"], float),
            centre=float(record["centre"]),
            scale=float(record["scale"]),
        )


def fit_calibration_shape(dosage: NDArray, truth: NDArray, stratum: str, version: str) -> CalibrationCurve:
    """Isotonic E[T | D] on the distinct dosage values, with unit scale.

    The scale is left at 1 here: with a noisy truth T = lambda G + e the shape is
    identified but lambda is not. Set it with calibrated_scale before applying.
    """
    dosage_values = np.asarray(dosage, float).ravel()
    truth_values = np.asarray(truth, float).ravel()
    observed = np.isfinite(dosage_values) & np.isfinite(truth_values)
    knots, inverse, counts = np.unique(dosage_values[observed], return_inverse=True, return_counts=True)
    if knots.size < 2:
        raise ValueError(f"stratum {stratum}: the calibration shape needs at least 2 distinct dosage values")
    truth_sums = np.bincount(inverse, weights=truth_values[observed], minlength=knots.size)
    expectation = _pool_adjacent_violators(truth_sums / counts, counts.astype(float))
    centre = float(np.sum(expectation * counts) / counts.sum())
    return CalibrationCurve(stratum, version, knots, expectation, centre, 1.0)


def calibrated_scale(shape_truth_correlation: float, genotype_sd: float, shape_sd: float) -> float:
    """Scale b with Cov(G, D*) = Var(D*) for D* = centre + b (h - centre).

    b = r(h, G) sd(G) / sd(h), where r(h, G) is the triad correlation of the
    shaped column with the true genotype, never a regression slope on one noisy
    truth, which is attenuated by that truth's error.
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
    minimum_r2: float

    def predict_r2(self, features: NDArray) -> F64Array:
        design = np.asarray(features, float)
        if design.ndim != 2 or design.shape[1] != len(self.feature_names):
            raise ValueError(f"expected features of shape (records, {len(self.feature_names)}), got {design.shape}")
        linear_predictor = self.intercept + design @ self.coefficients
        r2 = 1.0 / (1.0 + np.exp(-linear_predictor))
        return np.maximum(r2, self.minimum_r2)

    def log_reliability_offset(self, features: NDArray) -> F64Array:
        """log r^2, the prior-variance offset that enters with an EB coefficient centred at 1."""
        return np.log(self.predict_r2(features))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "feature_names": list(self.feature_names),
            "intercept": self.intercept,
            "coefficients": self.coefficients.tolist(),
            "minimum_r2": self.minimum_r2,
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
            minimum_r2=float(record["minimum_r2"]),
        )
