"""Separation of a binary trait's labels by its covariates, and the logistic likelihood's curvature bound.

A binary trait's likelihood is prod_i sigmoid(sign_i eta_i), with sign_i = 2 y_i - 1 and
eta_i = c_i' alpha + x_i' beta: the covariates alpha under a flat prior, the effects beta under
the variant-side prior. With alpha flat, the posterior is improper when the covariates alone
(quasi-)separate the labels (for example a sex indicator on a sex-limited disease), and a fit would
drift without bound. ``assert_no_separation`` detects it exactly with a linear program before any fit.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from sv_pgs._typing import F64Array

_EPSILON = float(np.finfo(np.float64).eps)
_LOG_SIGMOID_CURVATURE_BOUND = 0.25
"""max over w of -(log sigmoid)''(w) = sigmoid(w) sigmoid(-w), attained at w = 0."""


class SeparatedLabels(ValueError):
    """The covariates alone (quasi-)separate the labels, so the flat-prior posterior is improper."""


def _signs(labels: F64Array) -> F64Array:
    outcome = np.asarray(labels, dtype=np.float64)
    if not np.all((outcome == 0.0) | (outcome == 1.0)):
        raise ValueError("labels must be 0/1.")
    return 2.0 * outcome - 1.0


def separating_direction(covariates: F64Array, labels: F64Array) -> F64Array | None:
    """A direction alpha != 0 with sign_i c_i' alpha >= 0 for every sample, or None where there is none.

    A linear program maximizes sum_i sign_i c_i' alpha over those alpha in the unit box. The
    objective is homogeneous, so the box only normalizes it. The maximum is zero exactly when no
    (quasi-)separating direction exists; the test allows the program's rounding, n eps ||C||.
    """
    design = np.asarray(covariates, dtype=np.float64)
    signed = _signs(labels)[:, None] * design
    if signed.shape[1] == 0:
        return None
    result = linprog(
        c=-signed.sum(axis=0),
        A_ub=-signed,
        b_ub=np.zeros(signed.shape[0]),
        bounds=[(-1.0, 1.0)] * signed.shape[1],
        method="highs",
    )
    if result.status != 0:
        raise ArithmeticError(f"the separation linear program did not solve: {result.message}")
    rounding = signed.shape[0] * _EPSILON * float(np.abs(signed).sum())
    return np.asarray(result.x, dtype=np.float64) if -result.fun > rounding else None


def assert_no_separation(covariates: F64Array, labels: F64Array) -> None:
    """Raise SeparatedLabels if some direction alpha != 0 has sign_i c_i' alpha >= 0 for every sample
    (``separating_direction``)."""
    direction = separating_direction(covariates, labels)
    if direction is not None:
        raise SeparatedLabels(
            f"the covariates (quasi-)separate the labels along direction {direction.tolist()};"
            " restrict the cohort to the at-risk group or drop the separating covariate."
        )
