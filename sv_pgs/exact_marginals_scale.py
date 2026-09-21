"""Exact marginal posterior variances and the bulk kernel's diagonal through the n x n dual, with a rounding certificate.

Model as in marginal_variances: the metric is folded into the columns, Xt = (I - H) W^1/2 X, and the Gaussian
EP sites give q(beta) with precision A = Xt'Xt + diag(Pi). EP needs diag(A^-1) at every refresh; logistic EP
also needs the bulk kernel's diagonal diag(K^-1) for the sample leverages.

Split the sites into the bulk S (Pi_j > 0, D_j = 1 / Pi_j) and the resolved set L (every Pi_j <= 0, plus any
sites the caller names). With K = I + Xt_S D_S Xt_S' (n x n, K >= I), U = Xt_L and core = Pi_L + U' K^-1 U,

    diag(A^-1)_j = D_j - D_j^2 (q_j - c_j core^-1 c_j'),   q_j = xt_j' K^-1 xt_j,   c_j = xt_j' K^-1 U   (j in S)
    diag(A^-1)_L = diag(core^-1).

The bulk diagonal [K^-1]_ii is returned alone, in binary-ep's convention: the resolved sites' term and the
covariate leverage are added once, by dual_solve.SampleDiagonal.predictor_variance, and adding them here too
would count them twice.

These are marginal_variances' identity 1 with K^-1 applied exactly: one Cholesky K = RR' replaces the window
and its far-field equivalent, so nothing is estimated and no probe certificate is needed. With u = R^-1 x every
quantity above is an inner product of forward solves, run on the array module's device. Two passes over the
columns (form K and gather U, then forward-solve every column) cost n^2 p flops each, the factor n^3/3, and
diag(K^-1) another n^3 when it is asked for, in blocks of identity columns, so no second n x n array is held.
`exact_dual_cost` gives the counts.

Certificate. The algebra is exact, so the bound covers floating point only. Let u be the unit roundoff and
gamma_m = m u / (1 - m u) (Higham 2002, Accuracy and Stability of Numerical Algorithms, section 3.1).
- Formation: the computed K differs from the true one by at most gamma_m sum_j D_j ||xt_j||^2 in the 2-norm,
  with m the accumulation depth. Each entry is a sum of m products (Higham Thm 3.5), and the elementwise bound
  sum_j D_j |xt_j||xt_j|' is PSD, so its 2-norm is at most its trace.
- Factor: the computed factor satisfies RR' = K_hat + E with |E| <= gamma_{n+1} |R||R'| (Thm 10.3), so
  ||E||_2 <= gamma_{n+1} || |R| ||_2^2 <= gamma_{n+1} min(||R||_F^2, || |R| ||_1 || |R| ||_inf). The 1- and
  inf-norms cost n^2 to read and, for K near I plus a small trace, are far below tr(K) = ||R||_F^2.
  Together ||RR' - K||_2 <= delta.
- Inner products: K >= I gives ||K^-1|| <= 1 and ||(RR')^-1|| <= 1 / (1 - delta), so every pair of columns has
  |x_a'(RR')^-1 x_b - x_a'K^-1 x_b| <= ||x_a|| ||x_b|| delta / (1 - delta). Since (RR')^-1 - K^-1 =
  -(RR')^-1 E K^-1, it is also at most ||u_a|| ||u_b|| delta (1 + delta) / (1 - delta), with u = R^-1 x; the
  smaller of the two is taken. The second removes a factor 1 + D_j ||xt_j||^2 on data-dominated sites, where
  D_j (1 - D_j q_j) cancels.
- Solves: each forward solve is backward stable, (R + F) u_hat = x with |F| <= gamma_n |R| (Thm 8.5), so
  u_hat = R^-1 x + e with ||e|| <= beta / (1 - beta) ||u_hat||, where
  beta = gamma_n || |R| ||_2 / (sqrt(1 - delta) - gamma_n || |R| ||_2). Each inner product carries a further
  relative gamma_n.
The bound expressions themselves are evaluated in float64, so each is exact to a relative few u of a bound
that lies orders of magnitude above the error it bounds.
That bounds every q, c, core and diag(K^-1) entry. The core terms go through the same argument on the small factor of
core, with its smallest eigenvalue bounded below from that factor and its computed inverse. A core that isn't
certifiably positive definite, or any bound that can't be established, raises NotCertified; nothing is guessed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

UNIT_ROUNDOFF = np.finfo(np.float64).eps / 2
# Roundings in one term D_j xt_ij xt_kj beyond the accumulation depth, which already counts the product and every
# addition: D_j = 1 / Pi_j (its error enters each factor as a square root, so once), sqrt(D_j) (in both factors)
# and the two scalings (Higham 2002 Lemma 3.1).
FORMATION_ROUNDINGS = 5
# Roundings in D (1 - D v): D = 1 / Pi (which enters twice), the product D v, the difference and the scaling.
VARIANCE_OPERATIONS = 5

Blocks = Callable[[], Iterable[tuple[NDArray[np.int64], Any]]]


class NotCertified(ArithmeticError):
    """The rounding bound cannot be established, so the exact route refuses rather than guess."""


@dataclass(frozen=True)
class KernelParts:
    """The factored bulk kernel, for callers that keep it (DualGaussian.kernel_factor): `lower` is R
    with RR' = K_S, `resolved_solves` is K_S^-1 Xt_L (n x |L|) and `resolved_core` is Pi_L + Xt_L' K_S^-1 Xt_L,
    both with the resolved sites in ExactMarginals.resolved order. `lower` and `resolved_solves` stay on the array
    module's device; the core is on the host."""
    lower: Any
    resolved_solves: Any
    resolved_core: NDArray[np.float64]


@dataclass(frozen=True)
class ExactMarginals:
    variances: NDArray[np.float64]
    variance_bound: NDArray[np.float64]
    bulk_diagonal: NDArray[np.float64] | None
    bulk_diagonal_bound: NDArray[np.float64] | None
    resolved: NDArray[np.int64]
    factor_bound: float
    kernel: KernelParts | None = None


def _gamma(count: int) -> float:
    product = count * UNIT_ROUNDOFF
    if product >= 1:
        raise NotCertified(f"gamma_{count} is undefined: {count} u >= 1")
    return product / (1 - product)


class _Backend:
    """The kernel's one n x n array, on the host or the device: K = I + sum of SYRK updates into its lower
    triangle, in place, then factored in place (LAPACK / cuSOLVER potrf, lower). Fortran order throughout, so
    the triangular solves read the factor without copying it; the strict upper triangle is never written and
    stays exactly zero."""

    def __init__(self, array_module: Any) -> None:
        self.xp = array_module
        self.on_host = array_module is np

    def identity(self, size: int) -> Any:
        if self.on_host:
            return np.eye(size, order="F")
        kernel = self.xp.zeros((size, size), dtype=self.xp.float64, order="F")
        self.xp.fill_diagonal(kernel, 1)
        return kernel

    def accumulate(self, kernel: Any, scaled: Any) -> Any:
        """kernel += scaled scaled' on the lower triangle."""
        if self.on_host:
            return scipy.linalg.blas.dsyrk(1.0, scaled, beta=1.0, c=kernel, lower=1, overwrite_c=1)
        import cupy.cublas

        cupy.cublas.syrk("N", scaled, out=kernel, alpha=1.0, beta=1.0, lower=True)
        return kernel

    def cholesky(self, kernel: Any) -> Any:
        if self.on_host:
            factor, info = scipy.linalg.lapack.dpotrf(kernel, lower=1, clean=1, overwrite_a=1)
        else:
            from cupy.cuda import device
            from cupy_backends.cuda.libs import cublas, cusolver

            size = kernel.shape[0]
            handle = device.get_cusolver_handle()
            status = self.xp.empty(1, dtype=np.int32)
            buffer_size = cusolver.dpotrf_bufferSize(handle, cublas.CUBLAS_FILL_MODE_LOWER, size, kernel.data.ptr, size)
            workspace = self.xp.empty(buffer_size, dtype=self.xp.float64)
            cusolver.dpotrf(handle, cublas.CUBLAS_FILL_MODE_LOWER, size, kernel.data.ptr, size,
                            workspace.data.ptr, buffer_size, status.data.ptr)
            factor, info = kernel, int(status.get()[0])
        if info != 0:
            raise NotCertified("the kernel K >= I lost positive definiteness in float64")
        return factor

    def forward(self, factor: Any, right: Any) -> Any:
        if self.on_host:
            return scipy.linalg.solve_triangular(factor, right, lower=True, check_finite=False)
        import cupyx.scipy.linalg

        return cupyx.scipy.linalg.solve_triangular(factor, right, lower=True)

    def backward(self, factor: Any, right: Any) -> Any:
        """R'^-1 right."""
        if self.on_host:
            return scipy.linalg.solve_triangular(factor, right, lower=True, trans="T", check_finite=False)
        import cupyx.scipy.linalg

        return cupyx.scipy.linalg.solve_triangular(factor, right, lower=True, trans="T")

    def to_host(self, values: Any) -> Any:
        return values if self.on_host else self.xp.asnumpy(values)


@dataclass(frozen=True)
class _FactorBounds:
    """Pair error |fl(u_a'u_b) - x_a' M^-1 x_b| for a computed factor of M, from the columns' norms ||x|| and the
    computed solves' norms ||u||: min(||x_a|| ||x_b|| rho, ||u_a|| ||u_b|| solved_rho) + ||u_a|| ||u_b|| sigma."""
    rho: float
    solved_rho: float
    sigma: float
    inverse_norm: float

    def pair(self, x_a: NDArray[np.float64], x_b: NDArray[np.float64], u_a: NDArray[np.float64], u_b: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.minimum(x_a * x_b * self.rho, u_a * u_b * self.solved_rho) + u_a * u_b * self.sigma


def _factor_bounds(delta: float, factor_norm_squared: float, dimension: int, smallest_eigenvalue: float) -> _FactorBounds:
    """Bounds for RR' = M + E, ||E||_2 <= delta, M >= smallest_eigenvalue I, given an upper bound on || |R| ||_2^2."""
    margin = smallest_eigenvalue - delta
    if margin <= 0:
        raise NotCertified(f"the factor's error {delta:.3g} reaches the smallest eigenvalue {smallest_eigenvalue:.3g}")
    rho = delta / (smallest_eigenvalue * margin)
    solve_error = _gamma(dimension) * np.sqrt(factor_norm_squared)
    denominator = np.sqrt(margin) - solve_error
    if denominator <= 0:
        raise NotCertified("forward solves cannot be bounded: the factor is too ill-conditioned for float64")
    beta = solve_error / denominator
    if beta >= 1:
        raise NotCertified("forward-solve error bound exceeds the solution")
    growth = beta / (1 - beta)
    # (RR')^-1 - M^-1 = -(RR')^-1 E M^-1, with ||(RR')^-1 x|| <= ||R^-1|| ||R^-1 x||, ||M^-1 (RR')|| <= 1 + delta / lambda,
    # ||R^-1||^2 <= 1 / margin and ||R^-1 x|| <= ||u_hat|| / (1 - beta).
    solved_rho = delta * (1 + delta / smallest_eigenvalue) / (margin * (1 - beta) ** 2)
    return _FactorBounds(rho=rho, solved_rho=solved_rho, sigma=growth * (1 + 1 / (1 - beta)) + _gamma(dimension),
                         inverse_norm=1 / margin)


def _upper(values: NDArray[np.float64], terms: int) -> NDArray[np.float64]:
    """Upper bound on a sum of `terms` non-negative floating-point terms from its computed value."""
    return values * (1 + _gamma(terms))


def _factor_norm_squared(factor: Any, array_module: Any, to_host: Callable, width: int) -> float:
    """Upper bound on || |R| ||_2^2 for a lower-triangular R: min(||R||_F^2, || |R| ||_1 || |R| ||_inf) (Higham 2002
    section 6.3), read in row blocks of `width` so no second n x n array is formed."""
    size = factor.shape[0]
    row_largest, frobenius_squared = 0.0, 0.0
    column_sums = array_module.zeros(size, dtype=array_module.float64)
    for start in range(0, size, width):
        stop = min(start + width, size)
        magnitude = array_module.abs(factor[start:stop, :stop])
        row_largest = max(row_largest, float(to_host(magnitude.sum(axis=1).max())))
        column_sums[:stop] += magnitude.sum(axis=0)
        frobenius_squared += float(to_host((magnitude * magnitude).sum()))
    column_largest = float(to_host(column_sums.max()))
    absolute = _upper(np.array([row_largest]), size)[0] * _upper(np.array([column_largest]), size)[0]
    return float(min(_upper(np.array([frobenius_squared]), size * size)[0], absolute))


def exact_marginals(blocks: Blocks, precision: NDArray[np.float64], sample_count: int, *,
                    resolved: NDArray[np.int64] | None = None, bulk_diagonal: bool = False, array_module: Any = np,
                    identity_block: int | None = None, keep_kernel: bool = False) -> ExactMarginals:
    """diag(A^-1) (and, if asked, the bulk diagonal diag(K^-1)) with a certified rounding bound.

    blocks() must return a fresh iterable of (column indices, Xt[:, columns]) on every call: the columns are
    read twice. `resolved` names extra sites to eliminate through the core (non-positive sites always are).
    `keep_kernel` also returns the factored kernel (KernelParts), at one more triangular solve of the resolved
    columns (n^2 |L|); the factor then stays alive with the result.
    """
    xp = array_module
    backend = _Backend(xp)
    forward, to_host = backend.forward, backend.to_host
    precision = np.asarray(precision, dtype=np.float64)
    variant_count = precision.size
    resolved_mask = precision <= 0
    if resolved is not None:
        resolved_mask[np.asarray(resolved, dtype=np.int64)] = True
    bulk_variance = np.zeros(variant_count)
    bulk_variance[~resolved_mask] = 1 / precision[~resolved_mask]

    kernel = backend.identity(sample_count)
    column_norm = np.zeros(variant_count)
    coverage = np.zeros(variant_count, dtype=np.int64)
    load, depth, widest = 0.0, 0, 0
    resolved_columns: list[NDArray[np.int64]] = []
    resolved_values: list[Any] = []
    for columns, block in blocks():
        columns = np.asarray(columns, dtype=np.int64)
        np.add.at(coverage, columns, 1)
        widest = max(widest, columns.size)
        block = xp.asarray(block, dtype=xp.float64)
        norms = _upper(to_host((block * block).sum(axis=0)), sample_count + 1)
        column_norm[columns] = np.sqrt(norms)
        bulk = ~resolved_mask[columns]
        if bulk.any():
            variance = bulk_variance[columns][bulk]
            scaled = block[:, xp.asarray(np.flatnonzero(bulk))] * xp.asarray(np.sqrt(variance))
            kernel = backend.accumulate(kernel, scaled)
            load += float((variance * norms[bulk]).sum())
            depth += int(bulk.sum()) + 1
        if not bulk.all():
            resolved_columns.append(columns[~bulk])
            resolved_values.append(block[:, xp.asarray(np.flatnonzero(~bulk))])
    resolved_index = np.concatenate(resolved_columns) if resolved_columns else np.zeros(0, dtype=np.int64)
    if not np.all(coverage == 1):
        raise ValueError("blocks() must cover every column exactly once")

    # Blocks of n x width temporaries, as wide as the widest column block the caller streams (its own working size).
    width = identity_block or widest
    factor = backend.cholesky(kernel)
    del kernel
    factor_norm_squared = _factor_norm_squared(factor, xp, to_host, width)
    delta = (_gamma(depth + FORMATION_ROUNDINGS) * load * (1 + _gamma(variant_count + sample_count))
             + _gamma(sample_count + 1) * factor_norm_squared)
    kernel_bounds = _factor_bounds(delta, factor_norm_squared, sample_count, 1.0)

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
        core_error_norm = float(np.sqrt(_upper(np.array([(core_error * core_error).sum()]), resolved_count * resolved_count)[0]))
        try:
            core_factor = np.linalg.cholesky(core)
        except np.linalg.LinAlgError as error:
            raise NotCertified("core is not positive definite in float64") from error
        core_norm_squared = _factor_norm_squared(core_factor, np, np.asarray, resolved_count)
        core_delta = _gamma(resolved_count + 1) * core_norm_squared
        # lambda_min(core) from its own factor, not an eigensolver's unstated error constant: the computed inverse
        # factor Z has L Z = I - G with ||G||_2 <= gamma_r || |L| ||_2 ||Z||_F (Thm 8.5, column by column), so
        # sigma_min(L) >= (1 - ||G||) / ||Z||_F, and core >= sigma_min(L)^2 - core_delta (Thm 10.3).
        inverse_factor = scipy.linalg.solve_triangular(core_factor, np.eye(resolved_count), lower=True, check_finite=False)
        inverse_frobenius = float(np.sqrt(_upper(np.array([(inverse_factor * inverse_factor).sum()]), resolved_count * resolved_count)[0]))
        residual = _gamma(resolved_count) * np.sqrt(core_norm_squared) * inverse_frobenius
        if residual >= 1:
            raise NotCertified("the core's inverse factor cannot be bounded: the core is too ill-conditioned for float64")
        computed_smallest = ((1 - residual) / inverse_frobenius) ** 2 - core_delta
        true_smallest = computed_smallest - core_error_norm
        if computed_smallest <= 0 or true_smallest <= 0:
            raise NotCertified(f"core is not certifiably positive definite (smallest eigenvalue >= {computed_smallest:.3g}, error {core_error_norm:.3g})")
        core_bounds = _factor_bounds(core_delta, core_norm_squared, resolved_count, computed_smallest)
        true_core_inverse_norm = 1 / true_smallest
    else:
        solved_resolved = None

    def through_core(cross: NDArray[np.float64], cross_error: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """t = c core^-1 c' per row of c, and its bound, from the computed rows and their elementwise errors."""
        solved = scipy.linalg.solve_triangular(core_factor, cross.T, lower=True, check_finite=False)
        term = (solved * solved).sum(axis=0)
        cross_norm = np.sqrt(_upper((cross * cross).sum(axis=1), resolved_count))
        solved_norm = np.sqrt(_upper(term, resolved_count))
        error_norm = np.sqrt(_upper((cross_error * cross_error).sum(axis=1), resolved_count))
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
            quadratic_error = quadratic_error + term_error + UNIT_ROUNDOFF * (np.abs(quadratic) + np.abs(term))
            quadratic = quadratic - term
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

    diagonal_values = diagonal_bound = None
    if bulk_diagonal:
        # [K^-1]_ii = ||R^-1 e_i||^2, from blocks of identity columns solved against the whole factor: a trailing
        # sub-factor would save two thirds of the n^3 flops but is a strided view the solvers copy (up to n^2).
        diagonal_values = np.zeros(sample_count)
        diagonal_bound = np.zeros(sample_count)
        for start in range(0, sample_count, width):
            stop = min(start + width, sample_count)
            unit_columns = xp.zeros((sample_count, stop - start), dtype=xp.float64, order="F")
            unit_columns[xp.arange(start, stop), xp.arange(stop - start)] = 1
            solved = forward(factor, unit_columns)
            diagonal = to_host((solved * solved).sum(axis=0))
            solved_norm = np.sqrt(_upper(diagonal, sample_count))
            ones = np.ones(stop - start)
            diagonal_values[start:stop] = diagonal
            diagonal_bound[start:stop] = kernel_bounds.pair(ones, ones, solved_norm, solved_norm)
    kept = None
    if keep_kernel:
        order = np.argsort(resolved_index)
        if resolved_count:
            kept = KernelParts(lower=factor, resolved_solves=backend.backward(factor, solved_resolved)[:, xp.asarray(order)],
                               resolved_core=core[np.ix_(order, order)])
        else:
            kept = KernelParts(lower=factor, resolved_solves=xp.zeros((sample_count, 0)), resolved_core=np.zeros((0, 0)))
    return ExactMarginals(variances=variances, variance_bound=variance_bound, bulk_diagonal=diagonal_values,
                          bulk_diagonal_bound=diagonal_bound, resolved=np.sort(resolved_index), factor_bound=delta,
                          kernel=kept)


def exact_dual_cost(sample_count: int, variant_count: int, resolved_count: int = 0, *, bulk_diagonal: bool = False) -> dict[str, float]:
    """Flops, fp64 bytes resident and column passes of one exact refresh (Golub & Van Loan 2013, sections 3.1, 4.2)."""
    n, p = float(sample_count), float(variant_count)
    flops = {
        "formation": n * n * p,
        "factor": n ** 3 / 3,
        "forward_solves": n * n * p,
        "resolved": n * n * resolved_count + n * p * resolved_count,
        "diagonal_of_inverse": n ** 3 if bulk_diagonal else 0.0,
    }
    return {**flops, "total_flops": sum(flops.values()), "resident_bytes": np.dtype(np.float64).itemsize * n * n,
            "column_passes": 2.0}
