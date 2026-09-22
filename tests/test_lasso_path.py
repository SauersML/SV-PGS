"""The pathwise lasso (strong rules, KKT checks, warm starts) against scikit-learn's coordinate descent."""
import numpy as np
from sklearn.linear_model import Lasso

from sv_pgs.lasso_path import PATH_RATIO, cross_validated_lasso, lasso_path


def _data(seed: int, n: int, p: int):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    x[:, 1::2] = 0.8 * x[:, 0::2][:, : p // 2] + 0.6 * x[:, 1::2]
    x -= x.mean(axis=0)
    x /= x.std(axis=0)
    beta = np.zeros(p)
    beta[rng.choice(p, 8, replace=False)] = rng.normal(0.0, 1.0, 8)
    return x, x @ beta + rng.normal(size=n)


def test_every_solution_on_the_path_is_the_lasso() -> None:
    x, y = _data(1, 120, 400)
    largest = float(np.max(np.abs(x.T @ y)) / x.shape[0])
    penalties = largest * np.geomspace(1.0, PATH_RATIO, 25)
    solutions = lasso_path(x, y, penalties)
    count = x.shape[0]
    null = float(y @ y) / (2.0 * count)

    def objective(beta, penalty):
        residual = y - x @ beta
        return float(residual @ residual) / (2.0 * count) + penalty * float(np.sum(np.abs(beta)))

    for penalty, solution in zip(penalties[::6], solutions[::6]):
        reference = Lasso(alpha=penalty, fit_intercept=False, tol=1e-12, max_iter=100000).fit(x, y).coef_
        # glmnet's stopping rule (THRESHOLD of the null deviance per coordinate move): the objective is the lasso's
        # to a small share of the null objective, and the support is the lasso's where its coefficients are resolved.
        assert objective(solution, penalty) - objective(reference, penalty) <= 1e-5 * null
        resolved = np.abs(reference) > 1e-2
        assert np.all(solution[resolved] != 0.0)
    # the first penalty is lambda_max: every coefficient is zero there, to the soft threshold's rounding
    assert np.max(np.abs(solutions[0])) <= 1e-12


def test_the_cross_validated_lasso_is_deterministic_and_sparse() -> None:
    x, y = _data(2, 150, 600)
    first, second = cross_validated_lasso(x, y, 7), cross_validated_lasso(x, y, 7)
    np.testing.assert_array_equal(first, second)
    assert 0 < int(np.sum(first != 0.0)) < x.shape[1]
