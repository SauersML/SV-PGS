"""The cross-validated lasso the mean-field oracle starts from (``small_n.lasso_start``), by pathwise coordinate descent.

Minimizes (1/(2n)) ||y - X b||^2 + lambda ||b||_1 on a geometric path of penalties from lambda_max = ||X'y||_inf / n down
to ``PATH_RATIO`` of it in ``PATH_LENGTH`` steps (glmnet's defaults, Friedman, Hastie and Tibshirani 2010, J. Stat.
Softw. 33:1), each solution warm-starting the next. At each penalty only the columns the sequential strong rule keeps,
|x_j'r| / n >= 2 lambda - lambda_prev (Tibshirani et al. 2012, JRSS-B 74:245), are swept; after convergence every
column's KKT condition |x_j'r| / n <= lambda is checked and any violator joins the sweep, so each solution is the lasso to glmnet's
stopping rule. The penalty is the 10-fold cross-validated minimum of the held-out squared error; the
columns are the projected standardized design, so no column is rescaled.

scikit-learn's LassoCV sweeps every column at every penalty: minutes per gene at p = 25k, n = 534; this is seconds.
"""

from __future__ import annotations

import numba
import numpy as np

from sv_pgs._typing import F64Array

PATH_LENGTH = 100
"""Penalties on the path: glmnet's default nlambda."""
PATH_RATIO = 0.01
"""The path's smallest penalty over its largest where n < p: glmnet's default lambda.min.ratio."""
FOLDS = 10
"""Cross-validation folds: cv.glmnet's default nfolds, the one the mr.ash workflow's lasso start uses."""
THRESHOLD = 1e-7
"""glmnet's default thresh: coordinate descent stops when the largest objective change of any coefficient update in a
pass, (||x_j||^2 / n) dbeta_j^2, is below this share of the null deviance per sample, y'y / n."""


@numba.njit(cache=True)
def _sweep_active(x, squares, beta, residual, active, penalty, sample_count, threshold):
    """Coordinate descent over ``active`` until a pass's largest objective change, (||x_j||^2 / n) dbeta_j^2, is at
    most ``threshold`` (glmnet's rule)."""
    count = x.shape[0]
    while True:
        largest = 0.0
        for index in active:
            norm = squares[index]
            if norm <= 0.0:
                continue
            correlation = 0.0
            for sample in range(count):
                correlation += x[sample, index] * residual[sample]
            correlation = correlation / sample_count + norm / sample_count * beta[index]
            if correlation > penalty:
                new = (correlation - penalty) / (norm / sample_count)
            elif correlation < -penalty:
                new = (correlation + penalty) / (norm / sample_count)
            else:
                new = 0.0
            step = new - beta[index]
            if step != 0.0:
                for sample in range(count):
                    residual[sample] -= x[sample, index] * step
                beta[index] = new
                change = norm / sample_count * step * step
                if change > largest:
                    largest = change
        if largest <= threshold:
            return


def lasso_path(x: F64Array, y: F64Array, penalties: F64Array) -> F64Array:
    """(len(penalties), p) lasso solutions along decreasing ``penalties``, warm-started, strong-rule screened and
    KKT-checked."""
    x = np.asfortranarray(x, dtype=np.float64)
    count = float(x.shape[0])
    squares = np.einsum("ij,ij->j", x, x)
    beta = np.zeros(x.shape[1])
    residual = np.array(y, dtype=np.float64, copy=True)
    threshold = THRESHOLD * float(residual @ residual) / count
    solutions = np.zeros((penalties.shape[0], x.shape[1]))
    previous = float(np.max(np.abs(x.T @ residual)) / count)
    for step, penalty in enumerate(penalties):
        gradient = np.abs(x.T @ residual) / count
        active = np.flatnonzero((gradient >= 2.0 * penalty - previous) | (beta != 0.0)).astype(np.int64)
        while True:
            _sweep_active(x, squares, beta, residual, active, float(penalty), count, threshold)
            gradient = np.abs(x.T @ residual) / count
            violators = np.setdiff1d(np.flatnonzero(gradient > penalty), active)
            if violators.size == 0:
                break
            active = np.union1d(active, violators).astype(np.int64)
        solutions[step] = beta
        previous = float(penalty)
    return solutions


def cross_validated_lasso(x: F64Array, y: F64Array, seed: int) -> F64Array:
    """The lasso at the penalty of least ``FOLDS``-fold cross-validated squared error, refitted on every row."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    count = x.shape[0]
    largest = float(np.max(np.abs(x.T @ y)) / count)
    if largest <= 0.0:
        return np.zeros(x.shape[1])
    penalties = largest * np.geomspace(1.0, PATH_RATIO, PATH_LENGTH)
    folds = np.random.default_rng(seed).permutation(count) % FOLDS
    error = np.zeros(PATH_LENGTH)
    for fold in range(FOLDS):
        held = folds == fold
        solutions = lasso_path(x[~held], y[~held], penalties)
        error += np.sum(np.square(y[held][None, :] - solutions @ x[held].T), axis=1)
    best = int(np.argmin(error))
    return lasso_path(x, y, penalties[: best + 1])[best]
