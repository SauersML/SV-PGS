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
FLOAT32_ROUNDINGS = 3
"""The float32 roundings in one term x_ij r_i of the screened product besides its sum's: x_ij's, r_i's and the product's."""
FOLDS = 10
"""Cross-validation folds: cv.glmnet's default nfolds, the one the mr.ash workflow's lasso start uses."""
DEVIANCE_CHANGE = 1e-5
"""glmnet's fdev: the path stops where a penalty raises the explained deviance by less than this share of it."""
DEVIANCE_MAX = 0.999
"""glmnet's devmax: the path stops where the explained share of the null deviance exceeds this."""
THRESHOLD = 1e-7
"""glmnet's default thresh: coordinate descent stops when the largest objective change of any coefficient update in a
pass, (||x_j||^2 / n) dbeta_j^2, is below this share of the null deviance per sample, y'y / n."""


# fastmath lets the compiler reassociate the column dot products and residual updates into SIMD lanes; the start is
# refined by the mean-field fixed point, so their rounding order does not matter.
@numba.njit(cache=True, fastmath=True)
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
    """(m, p) lasso solutions along the first m of decreasing ``penalties``, warm-started, strong-rule screened and
    KKT-checked; the path stops early by glmnet's rules (``DEVIANCE_CHANGE``, ``DEVIANCE_MAX``): where p > n the
    path's tail approaches interpolation, where the sweeps cost most and the cross-validated error has turned."""
    x = np.asfortranarray(x, dtype=np.float64)
    count = float(x.shape[0])
    squares = np.einsum("ij,ij->j", x, x)
    beta = np.zeros(x.shape[1])
    residual = np.array(y, dtype=np.float64, copy=True)
    null_deviance = float(residual @ residual)
    threshold = THRESHOLD * null_deviance / count
    solutions = np.zeros((penalties.shape[0], x.shape[1]))
    explained = 0.0
    # |X'r| / n: the one p x n product each penalty needs. The KKT check's product after penalty k is at the residual
    # the strong rule reads at penalty k + 1, so it is kept rather than formed again (the products were 93% of a
    # path's time [real, ENSG00000138468.16: 10.1 s of products against 0.8 s of sweeps]).
    # The product runs on a float32 copy of X (half the bytes it streams); its error on column j is at most
    # (n + 3) eps32 ||x_j|| ||r|| (x_ij and r_i each round once to float32, their product once, the n-term sum by n eps32,
    # and Cauchy-Schwarz), so a column can violate KKT only where its float32 value is within that bound of the penalty,
    # and those few are formed exactly.
    single = np.asfortranarray(x, dtype=np.float32)
    column_norms = np.sqrt(squares)
    rounding = float(np.finfo(np.float32).eps) * (count + FLOAT32_ROUNDINGS)

    def screened() -> tuple[np.ndarray, np.ndarray]:
        approximate = np.abs(single.T @ residual.astype(np.float32)).astype(np.float64) / count
        return approximate, rounding * column_norms * float(np.linalg.norm(residual)) / count

    gradient, bound = screened()
    previous = float(np.max(gradient + bound))
    for step, penalty in enumerate(penalties):
        active = np.flatnonzero((gradient >= 2.0 * penalty - previous) | (beta != 0.0)).astype(np.int64)
        while True:
            _sweep_active(x, squares, beta, residual, active, float(penalty), count, threshold)
            gradient, bound = screened()
            candidates = np.setdiff1d(np.flatnonzero(gradient + bound > penalty), active)
            if candidates.size:
                exact = np.abs(x[:, candidates].T @ residual) / count
                gradient[candidates] = exact
                bound[candidates] = 0.0
            violators = candidates[gradient[candidates] > penalty] if candidates.size else candidates
            if violators.size == 0:
                break
            active = np.union1d(active, violators).astype(np.int64)
        solutions[step] = beta
        previous = float(penalty)
        if null_deviance <= 0.0:
            return solutions[: step + 1]
        now = 1.0 - float(residual @ residual) / null_deviance
        if step > 0 and (now > DEVIANCE_MAX or now - explained < DEVIANCE_CHANGE * now):
            return solutions[: step + 1]
        explained = now
    return solutions


def cross_validated_lasso(x: F64Array, y: F64Array, seed: int) -> F64Array:
    """The whole data's lasso at the penalty of least ``FOLDS``-fold cross-validated squared error."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    count = x.shape[0]
    largest = float(np.max(np.abs(x.T @ y)) / count)
    if largest <= 0.0:
        return np.zeros(x.shape[1])
    # The whole data's path sets the penalties (cv.glmnet: its early stop truncates them); each fold is fitted on
    # them, and a fold whose own path stopped earlier predicts with its last solution beyond it.
    full = lasso_path(x, y, largest * np.geomspace(1.0, PATH_RATIO, PATH_LENGTH))
    penalties = largest * np.geomspace(1.0, PATH_RATIO, PATH_LENGTH)[: full.shape[0]]
    folds = np.random.default_rng(seed).permutation(count) % FOLDS
    error = np.zeros(penalties.shape[0])
    for fold in range(FOLDS):
        held = folds == fold
        solutions = lasso_path(x[~held], y[~held], penalties)
        if solutions.shape[0] < penalties.shape[0]:
            solutions = np.concatenate([solutions, np.repeat(solutions[-1:], penalties.shape[0] - solutions.shape[0], axis=0)])
        error += np.sum(np.square(y[held][None, :] - solutions @ x[held].T), axis=1)
    return full[int(np.argmin(error))]
