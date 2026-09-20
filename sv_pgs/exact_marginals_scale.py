"""Exact marginal posterior variances and sample leverages through the n x n dual, with a rounding certificate.

Model as in marginal_variances: the metric is folded into the columns, Xt = (I - H) W^1/2 X, and the Gaussian
EP sites give q(beta) with precision A = Xt'Xt + diag(Pi). EP needs diag(A^-1) at every refresh; logistic EP
also needs the sample leverages h_i = [Xt A^-1 Xt']_ii.

Split the sites into the bulk S (Pi_j > 0, D_j = 1 / Pi_j) and the resolved set L (every Pi_j <= 0, plus any
sites the caller names). With K = I + Xt_S D_S Xt_S' (n x n, K >= I), U = Xt_L and core = Pi_L + U' K^-1 U,

    diag(A^-1)_j = D_j - D_j^2 (q_j - c_j core^-1 c_j'),   q_j = xt_j' K^-1 xt_j,   c_j = xt_j' K^-1 U   (j in S)
    diag(A^-1)_L = diag(core^-1)
    h_i = 1 - [K^-1]_ii + d_i core^-1 d_i',   d_i = e_i' K^-1 U.

These are marginal_variances' identity 1 with K^-1 applied exactly: one Cholesky K = RR' replaces the window
and its far-field equivalent, so nothing is estimated and no probe certificate is needed. With u = R^-1 x every
quantity above is an inner product of forward solves. Two passes over the columns (form K and gather U, then
forward-solve every column) cost n^2 p flops each, the factor n^3/3, and diag(K^-1) another n^3/3 when the
leverages are asked for. `exact_dual_cost` gives the counts.

Certificate. The algebra is exact, so the bound covers floating point only. Let u be the unit roundoff and
gamma_m = m u / (1 - m u) (Higham 2002, Accuracy and Stability of Numerical Algorithms, section 3.1).
- Formation: the computed K differs from the true one by at most gamma_m sum_j D_j ||xt_j||^2 in the 2-norm,
  with m the accumulation depth. Each entry is a sum of m products (Higham Thm 3.5), and the elementwise bound
  sum_j D_j |xt_j||xt_j|' is PSD, so its 2-norm is at most its trace.
- Factor: the computed factor satisfies RR' = K_hat + E with |E| <= gamma_{n+1} |R||R'| (Thm 10.3), so
  ||E||_2 <= gamma_{n+1} ||R||_F^2. Together ||RR' - K||_2 <= delta.
- Inner products: K >= I gives ||K^-1|| <= 1 and ||(RR')^-1|| <= 1 / (1 - delta), so every pair of columns has
  |x_a'(RR')^-1 x_b - x_a'K^-1 x_b| <= ||x_a|| ||x_b|| delta / (1 - delta).
- Solves: each forward solve is backward stable, (R + F) u_hat = x with |F| <= gamma_n |R| (Thm 8.5), so
  u_hat = R^-1 x + e with ||e|| <= beta / (1 - beta) ||u_hat||, where
  beta = gamma_n ||R||_F / (sqrt(1 - delta) - gamma_n ||R||_F). Each inner product carries a further relative
  gamma_n.
That bounds every q, c, core and d entry. The core terms go through the same argument on the small factor of
core, with core >= its certified smallest eigenvalue. A core that isn't certifiably positive definite, or any
bound that can't be established, raises NotCertified; nothing is guessed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

UNIT_ROUNDOFF = np.finfo(np.float64).eps / 2
# Roundings in one product D_j xt_ij xt_kj as formed: sqrt(D_j), two scalings and the product, plus the
# addition of the block's product into K (Higham 2002 section 3.1).
FORMATION_ROUNDINGS = 5
# Operations in D (1 - D v): the product D v, the difference and the final scaling.
VARIANCE_OPERATIONS = 3

Blocks = Callable[[], Iterable[tuple[NDArray[np.int64], Any]]]


class NotCertified(ArithmeticError):
    """The rounding bound cannot be established, so the exact route refuses rather than guess."""


@dataclass(frozen=True)
class ExactMarginals:
    variances: NDArray[np.float64]
    variance_bound: NDArray[np.float64]
    leverages: NDArray[np.float64] | None
    leverage_bound: NDArray[np.float64] | None
    resolved: NDArray[np.int64]
    factor_bound: float


def _gamma(count: int) -> float:
    product = count * UNIT_ROUNDOFF
    if product >= 1:
        raise NotCertified(f"gamma_{count} is undefined: {count} u >= 1")
    return product / (1 - product)


def _backend(array_module: Any) -> tuple[Callable, Callable, Callable]:
    if array_module is np:
        return (lambda matrix: scipy.linalg.cholesky(matrix, lower=True, overwrite_a=True, check_finite=False),
                lambda factor, right: scipy.linalg.solve_triangular(factor, right, lower=True, check_finite=False),
                lambda values: values)
    import cupyx.scipy.linalg

    return (lambda matrix: array_module.linalg.cholesky(matrix),
            lambda factor, right: cupyx.scipy.linalg.solve_triangular(factor, right, lower=True),
            array_module.asnumpy)


@dataclass(frozen=True)
class _FactorBounds:
    """Pair error |fl(u_a'u_b) - x_a' M^-1 x_b| <= ||x_a|| ||x_b|| rho + ||u_a|| ||u_b|| sigma for a factor of M."""
    rho: float
    sigma: float
    inverse_norm: float

    def pair(self, x_a: NDArray[np.float64], x_b: NDArray[np.float64], u_a: NDArray[np.float64], u_b: NDArray[np.float64]) -> NDArray[np.float64]:
        return x_a * x_b * self.rho + u_a * u_b * self.sigma


def _factor_bounds(delta: float, frobenius_squared: float, dimension: int, smallest_eigenvalue: float) -> _FactorBounds:
    """Bounds for RR' = M + E, ||E||_2 <= delta, M >= smallest_eigenvalue I, given ||R||_F^2."""
    margin = smallest_eigenvalue - delta
    if margin <= 0:
        raise NotCertified(f"the factor's error {delta:.3g} reaches the smallest eigenvalue {smallest_eigenvalue:.3g}")
    rho = delta / (smallest_eigenvalue * margin)
    solve_error = _gamma(dimension) * np.sqrt(frobenius_squared)
    denominator = np.sqrt(margin) - solve_error
    if denominator <= 0:
        raise NotCertified("forward solves cannot be bounded: the factor is too ill-conditioned for float64")
    beta = solve_error / denominator
    if beta >= 1:
        raise NotCertified("forward-solve error bound exceeds the solution")
    growth = beta / (1 - beta)
    return _FactorBounds(rho=rho, sigma=growth * (1 + 1 / (1 - beta)) + _gamma(dimension), inverse_norm=1 / margin)


def _upper(values: NDArray[np.float64], terms: int) -> NDArray[np.float64]:
    """Upper bound on a sum of `terms` non-negative floating-point terms from its computed value."""
    return values * (1 + _gamma(terms))


def exact_marginals(blocks: Blocks, precision: NDArray[np.float64], sample_count: int, *,
                    resolved: NDArray[np.int64] | None = None, leverages: bool = False, array_module: Any = np,
                    identity_block: int | None = None) -> ExactMarginals:
    """diag(A^-1) (and, if asked, the sample leverages) with a certified rounding bound.

    blocks() must return a fresh iterable of (column indices, Xt[:, columns]) on every call: the columns are
    read twice. `resolved` names extra sites to eliminate through the core (non-positive sites always are).
    """
    xp = array_module
    cholesky, forward, to_host = _backend(xp)
    precision = np.asarray(precision, dtype=np.float64)
    variant_count = precision.size
    resolved_mask = precision <= 0
    if resolved is not None:
        resolved_mask[np.asarray(resolved, dtype=np.int64)] = True
    bulk_variance = np.zeros(variant_count)
    bulk_variance[~resolved_mask] = 1 / precision[~resolved_mask]

    kernel = xp.eye(sample_count, dtype=xp.float64)
    column_norm = np.zeros(variant_count)
    load, depth = 0.0, 0
    resolved_columns: list[NDArray[np.int64]] = []
    resolved_values: list[Any] = []
    for columns, block in blocks():
        columns = np.asarray(columns, dtype=np.int64)
        block = xp.asarray(block, dtype=xp.float64)
        norms = _upper(to_host((block * block).sum(axis=0)), sample_count + 1)
        column_norm[columns] = np.sqrt(norms)
        bulk = ~resolved_mask[columns]
        if bulk.any():
            variance = bulk_variance[columns][bulk]
            scaled = block[:, xp.asarray(np.flatnonzero(bulk))] * xp.asarray(np.sqrt(variance))
            kernel += scaled @ scaled.T
            load += float((variance * norms[bulk]).sum())
            depth += int(bulk.sum()) + 1
        if not bulk.all():
            resolved_columns.append(columns[~bulk])
            resolved_values.append(block[:, xp.asarray(np.flatnonzero(~bulk))])
    resolved_index = np.concatenate(resolved_columns) if resolved_columns else np.zeros(0, dtype=np.int64)
    if not np.array_equal(np.sort(resolved_index), np.flatnonzero(resolved_mask)):
        raise ValueError("blocks() must cover every column exactly once")

    factor = cholesky(kernel)
    frobenius_squared = float(_upper(np.array([to_host((factor * factor).sum())]), sample_count * sample_count)[0])
    delta = (_gamma(depth + FORMATION_ROUNDINGS) * load * (1 + _gamma(variant_count + sample_count))
             + _gamma(sample_count + 1) * frobenius_squared)
    kernel_bounds = _factor_bounds(delta, frobenius_squared, sample_count, 1.0)

    resolved_count = resolved_index.size
    if resolved_count:
        resolved_block = xp.concatenate(resolved_values, axis=1)
        solved_resolved = forward(factor, resolved_block)
        resolved_x = column_norm[resolved_index]
        resolved_u = np.sqrt(_upper(to_host((solved_resolved * solved_resolved).sum(axis=0)), sample_count))
        gram = to_host(solved_resolved.T @ solved_resolved)
        core = gram + np.diag(precision[resolved_index])
        core_error = kernel_bounds.pair(resolved_x[:, None], resolved_x[None, :], resolved_u[:, None], resolved_u[None, :])
        core_error = core_error + UNIT_ROUNDOFF * np.abs(core)
        core_error_norm = float(np.sqrt((core_error * core_error).sum()))
        eigenvalues = np.linalg.eigvalsh(core)
        core_frobenius = float(np.sqrt((core * core).sum()))
        computed_smallest = float(eigenvalues[0]) - _gamma(resolved_count * resolved_count) * core_frobenius
        true_smallest = computed_smallest - core_error_norm
        if computed_smallest <= 0 or true_smallest <= 0:
            raise NotCertified(f"core is not certifiably positive definite (smallest eigenvalue {eigenvalues[0]:.3g}, error {core_error_norm:.3g})")
        core_factor = np.linalg.cholesky(core)
        core_frobenius_squared = float(_upper(np.array([(core_factor * core_factor).sum()]), resolved_count * resolved_count)[0])
        core_delta = _gamma(resolved_count + 1) * core_frobenius_squared
        core_bounds = _factor_bounds(core_delta, core_frobenius_squared, resolved_count, computed_smallest)
        true_core_inverse_norm = 1 / true_smallest
    else:
        solved_resolved = None

    def through_core(cross: NDArray[np.float64], cross_error: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """t = c core^-1 c' per row of c, and its bound, from the computed rows and their elementwise errors."""
        solved = scipy.linalg.solve_triangular(core_factor, cross.T, lower=True, check_finite=False)
        term = (solved * solved).sum(axis=0)
        cross_norm = np.sqrt(_upper((cross * cross).sum(axis=1), resolved_count))
        solved_norm = np.sqrt(_upper(term, resolved_count))
        error_norm = np.sqrt((cross_error * cross_error).sum(axis=1))
        rounding = core_bounds.pair(cross_norm, cross_norm, solved_norm, solved_norm)
        exact_norm = cross_norm + error_norm
        propagated = (core_bounds.inverse_norm * (cross_norm + exact_norm) * error_norm
                      + exact_norm ** 2 * core_bounds.inverse_norm * true_core_inverse_norm * core_error_norm)
        return term, rounding + propagated

    variances = np.zeros(variant_count)
    variance_bound = np.zeros(variant_count)
    for columns, block in blocks():
        columns = np.asarray(columns, dtype=np.int64)
        bulk = ~resolved_mask[columns]
        if not bulk.any():
            continue
        chosen = columns[bulk]
        solved = forward(factor, xp.asarray(block, dtype=xp.float64)[:, xp.asarray(np.flatnonzero(bulk))])
        quadratic = to_host((solved * solved).sum(axis=0))
        solved_norm = np.sqrt(_upper(quadratic, sample_count))
        chosen_x = column_norm[chosen]
        quadratic_error = kernel_bounds.pair(chosen_x, chosen_x, solved_norm, solved_norm)
        if resolved_count:
            cross = to_host(solved.T @ solved_resolved)
            cross_error = kernel_bounds.pair(chosen_x[:, None], resolved_x[None, :], solved_norm[:, None], resolved_u[None, :])
            term, term_error = through_core(cross, cross_error)
            quadratic = quadratic - term
            quadratic_error = quadratic_error + term_error
        variance = bulk_variance[chosen]
        values = variance * (1 - variance * quadratic)
        variances[chosen] = values
        variance_bound[chosen] = (variance ** 2 * quadratic_error
                                  + _gamma(VARIANCE_OPERATIONS) * (variance + variance ** 2 * np.abs(quadratic)))

    if resolved_count:
        unit = np.eye(resolved_count)
        term, term_error = through_core(unit, np.zeros_like(unit))
        variances[resolved_index] = term
        variance_bound[resolved_index] = term_error

    leverage_values = leverage_bound = None
    if leverages:
        leverage_values = np.zeros(sample_count)
        leverage_bound = np.zeros(sample_count)
        width = identity_block or sample_count
        for start in range(0, sample_count, width):
            stop = min(start + width, sample_count)
            unit_columns = xp.zeros((sample_count - start, stop - start), dtype=xp.float64)
            unit_columns[xp.arange(stop - start), xp.arange(stop - start)] = 1
            solved = forward(factor[start:, start:], unit_columns)
            diagonal = to_host((solved * solved).sum(axis=0))
            solved_norm = np.sqrt(_upper(diagonal, sample_count))
            ones = np.ones(stop - start)
            error = kernel_bounds.pair(ones, ones, solved_norm, solved_norm)
            value = 1 - diagonal
            if resolved_count:
                cross = to_host(solved.T @ solved_resolved[start:])
                cross_error = kernel_bounds.pair(ones[:, None], resolved_x[None, :], solved_norm[:, None], resolved_u[None, :])
                term, term_error = through_core(cross, cross_error)
                value = value + term
                error = error + term_error
            leverage_values[start:stop] = value
            leverage_bound[start:stop] = error + _gamma(2) * (1 + diagonal + np.abs(value))
    return ExactMarginals(variances=variances, variance_bound=variance_bound, leverages=leverage_values,
                          leverage_bound=leverage_bound, resolved=np.sort(resolved_index), factor_bound=delta)


def exact_dual_cost(sample_count: int, variant_count: int, resolved_count: int = 0, *, leverages: bool = False) -> dict[str, float]:
    """Flops, fp64 bytes resident and column passes of one exact refresh (Golub & Van Loan 2013, sections 3.1, 4.2)."""
    n, p = float(sample_count), float(variant_count)
    flops = {
        "formation": n * n * p,
        "factor": n ** 3 / 3,
        "forward_solves": n * n * p,
        "resolved": n * n * resolved_count + n * p * resolved_count,
        "diagonal_of_inverse": n ** 3 / 3 if leverages else 0.0,
    }
    return {**flops, "total_flops": sum(flops.values()), "resident_bytes": np.dtype(np.float64).itemsize * n * n,
            "column_passes": 2.0}
