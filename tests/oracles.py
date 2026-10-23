"""Code only the tests call: the direct formulas the package's fast routes are checked against, and harnesses that
run one package stage at fixed parameters.

Test infrastructure, kept out of the package: a route that agrees with one of these has its algebra right.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular

from sv_pgs.code_products import DIGIT_BITS
from sv_pgs.dual_solve import DualGaussian, DualModels, DualTile, DualTileSource, PassCount, _checked_bounds, _host, sequential_map_reduce
from sv_pgs.marginal_variances import BlockCertificate, BulkSolve, _certificate
from sv_pgs.phenotype_measurement import OccasionModelFit, Occasions, _Model, _result, _State
from sv_pgs.scale_mixture_ep import (
    Cavity,
    CurvatureCorrection,
    GaussianPosterior,
    MixtureHyperparameters,
    ScaleMixturePrior,
    _line_log_integrals,
    _lines,
    _log_sum_exp,
    _row_chunks,
    _total_curvature_columns,
    class_log_density,
    log_scale,
)


def far_field_trace(bulk_trace: float, bulk_square_trace: float, sample_count: int, eigenvalues: NDArray[np.float64]) -> float:
    """omega_F: the root of f(w) = w - omega_S - (omega_S2 / n) sum lambda / (1 + w lambda).

    f is increasing and concave, and f(omega_S) <= 0. So Newton from omega_S never overshoots and
    increases monotonically. It stops when an iterate no longer increases in floating point.
    """
    weight = bulk_square_trace / sample_count
    current = bulk_trace
    while True:
        denominator = 1.0 + current * eigenvalues
        value = current - bulk_trace - weight * float(np.sum(eigenvalues / denominator))
        slope = 1.0 + weight * float(np.sum(np.square(eigenvalues / denominator)))
        candidate = current - value / slope
        if not candidate > current:
            return current
        current = candidate


@dataclass(frozen=True)
class KernelFactor:
    """The bulk kernel K_S = I + Xt_S D_S Xt_S' factored densely: for shapes whose n x n fits in memory.

    ``lower`` is its Cholesky factor, ``resolved_solves`` Z_L = K_S^-1 Xt_L, and ``resolved_core``
    core = Pi_L + Xt_L' Z_L. Then Q = K^-1 = K_S^-1 - Z_L core^-1 Z_L' (Woodbury, valid for any core > 0).
    """

    lower: NDArray[np.float64]
    resolved_solves: NDArray[np.float64]
    resolved_core: NDArray[np.float64]

    @cached_property
    def core_inverse(self) -> NDArray[np.float64]:
        """core^-1, formed once per factor and shared by every block's call."""
        return np.linalg.inv(self.resolved_core) if self.resolved_core.shape[0] else np.zeros((0, 0))


def exact_block_information(factor: KernelFactor, bulk_variance: NDArray[np.float64], columns: NDArray[np.float64]) -> NDArray[np.float64]:
    """D_j - Sigma_jj for one block's variants, exactly: D_j^2 (xt_j' K_S^-1 xt_j - c_j core^-1 c_j').

    ``columns`` is the block's Xt_b (n x |b|) and ``bulk_variance`` its D_b (zero on resolved sites, whose rows
    the caller takes from core^-1). The cost is n^2 |b| for the triangular solve plus n |b| |L|.
    """
    whitened = solve_triangular(factor.lower, columns, lower=True)
    quadratic = np.sum(np.square(whitened), axis=0)
    cross = columns.T @ factor.resolved_solves
    spikes = np.sum((cross @ factor.core_inverse) * cross, axis=1) if cross.shape[1] else 0.0
    return np.square(bulk_variance) * (quadratic - spikes)


def _design_block(gaussian: DualGaussian, model: int, start: int, stop: int, tile: DualTile) -> Any:
    """Xt_b = (I - H) W^1/2 X_b of one block for model m, dense (n x p_b)."""
    return gaussian._state["models"].design_to_sample(tile.columns(np.arange(stop - start)), gaussian.array_module.full(stop - start, model))


def kernel_factor(gaussian: DualGaussian, model: int) -> KernelFactor:
    """Model m's bulk kernel K_S = I + Xt_S D_S Xt_S' factored densely at the gaussian's last iterate's sites (host).

    One read forms K_S (n^2 p flops), and its Cholesky factor gives Z_L = K_S^-1 Xt_L and core = Pi_L + Xt_L'Z_L
    (Pi_L of any sign). Raises LinAlgError when the core, hence A, is not positive definite.
    """
    array_module = gaussian.array_module
    state = gaussian._state
    variances = state["models"].variances[:, model]
    kernel = array_module.eye(gaussian.source.sample_count)
    for start, stop, tile in gaussian.source.blocks():
        block = _design_block(gaussian, model, start, stop, tile)
        kernel += (block * variances[start:stop][None, :]) @ block.T
    kernel = _host(kernel)
    lower = np.linalg.cholesky(0.5 * (kernel + kernel.T))
    block = state["blocks"].get(model)
    if block is None:
        return KernelFactor(lower=lower, resolved_solves=np.zeros((lower.shape[0], 0)), resolved_core=np.zeros((0, 0)))
    design = _host(block.design)
    resolved_solves = np.linalg.solve(lower.T, np.linalg.solve(lower, design))
    core = np.diag(_host(block.precision)) + design.T @ resolved_solves
    core = 0.5 * (core + core.T)
    np.linalg.cholesky(core)
    return KernelFactor(lower=lower, resolved_solves=resolved_solves, resolved_core=core)


def exact_marginals(gaussian: DualGaussian, model: int, factor: KernelFactor) -> NDArray[np.float64]:
    """diag(A_m^-1) at the gaussian's last iterate's sites (host, (p,)), exactly, from ``kernel_factor``'s factor.

    Bulk sites take D_j - ``exact_block_information``, resolved sites diag(core^-1) (A^-1_LL = core^-1, Schur's
    complement of A's bulk block). The only error is float64 rounding.
    """
    variances = _host(gaussian._state["models"].variances[:, model])
    marginals = np.empty(gaussian.source.variant_count)
    for start, stop, tile in gaussian.source.blocks():
        block_variances = variances[start:stop]
        information = exact_block_information(factor, block_variances, _host(_design_block(gaussian, model, start, stop, tile)))
        marginals[start:stop] = block_variances - information
    resolved = gaussian._resolved[model]
    if resolved.size:
        marginals[resolved] = np.diag(np.linalg.inv(factor.resolved_core))
    return marginals


def exact_bulk_diagonal(factor: KernelFactor) -> NDArray[np.float64]:
    """diag(K_S^-1), exactly: the column sums of squares of L^-1, with K_S = L L'. The cost is n^3 / 3.

    This is the bulk diagonal only: the resolved sites' correction (Z_L core^-1 Z_L')_ii and the covariate leverage
    are its callers'.
    """
    inverse_lower = solve_triangular(factor.lower, np.eye(factor.lower.shape[0]), lower=True)
    return np.sum(np.square(inverse_lower), axis=0)


def block_trace_certificate(
    variances: NDArray[np.float64],
    blocks: tuple[NDArray[np.int64], ...],
    probes: NDArray[np.float64],
    covariance_probes: NDArray[np.float64],
    tolerance: "float | NDArray[np.float64]",
    level: float,
    skewness: "NDArray[np.float64] | None" = None,
) -> BlockCertificate:
    """Test each block's tr(Sigma_bb) against Rademacher probes z (p x k) and Sigma z (from the solver).

    For each probe, z_b' (Sigma z)_b is an unbiased estimate of tr(Sigma_bb). The block's relative error is
    (mean - sum_b variances) / sum_b variances, and its standard error is the probes' own spread over sqrt(k),
    with intervals as in ``_certificate``.
    """
    per_probe = [np.sum(probes[members] * covariance_probes[members], axis=0) for members in blocks]
    estimate = np.array([float(np.sum(variances[members])) for members in blocks])
    return _certificate(estimate, per_probe, tolerance, level, skewness)


def covariance_products(
    solve: BulkSolve, probes: NDArray[np.float64], back_products: NDArray[np.float64], resolved_coupling: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Sigma v for variant-side vectors v (p x k), exactly, from two solver products.

    With u = Xt (D_S v) (forward pass), t = Z_L' u (sample-side, |L| x k) and
    w = K_S^-1 u - Z_L core^-1 (t - v_L), the solver returns ``back_products`` = Xt' w and
    ``resolved_coupling`` = t. Block elimination (identity 1) gives
        (Sigma v)_S = D v - D (Xt' w),   (Sigma v)_L = core^-1 (v_L - t).
    """
    is_resolved = np.zeros(solve.site_precision.shape[0], dtype=bool)
    is_resolved[solve.resolved] = True
    bulk_variance = np.where(is_resolved, 0.0, 1.0 / np.where(is_resolved, 1.0, solve.site_precision))
    products = bulk_variance[:, None] * (probes - back_products)
    products[solve.resolved] = np.linalg.solve(solve.resolved_core, probes[solve.resolved] - resolved_coupling)
    return products


def reassociation_bound(block_images: Sequence[Any], array_module: Any) -> Any:
    """Elementwise bound on how far two summation orders of ``block_images`` can differ in fp64.

    Each order is within ``gamma_(B-1) M`` of the exact sum, ``M = sum_b |image_b|``. The bound is
    evaluated in fp64 as ``2 gamma_(B+1) M~``: the computed M~ is at least ``(1 - gamma_(B-1)) M``
    and the bound's own arithmetic rounds by a relative ``4u`` or less, while
    ``gamma_(B+1) / gamma_(B-1) >= 1 + 2 / (B - 1)`` covers both for any ``B`` below ``u^-1/2``.
    """
    count = len(block_images)
    if count <= 1:
        return array_module.zeros_like(block_images[0]) if count else 0.0
    unit = np.finfo(np.float64).eps / 2.0
    if count * count * unit >= 1.0:
        raise ValueError(f"{count} blocks are too many for the fp64 reassociation bound")
    gamma = (count + 1) * unit / (1.0 - (count + 1) * unit)
    magnitude = array_module.zeros_like(block_images[0])
    for image in block_images:
        magnitude += array_module.abs(image)
    return 2.0 * gamma * magnitude


def fit_at_exponent(occasions: Occasions, exponent: float, working_bytes: int) -> OccasionModelFit:
    """The per-occasion model's fit at one Box-Cox exponent (``fit_occasion_model`` without the exponent search)."""
    model = _Model(occasions, exponent, working_bytes)
    return _result(model, *model.fit(model.start()))


def level_posterior_at(
    occasions: Occasions, exponent: float, fixed_effects: NDArray[np.float64], level_variance: float, prior: ScaleMixturePrior,
    hyperparameters: MixtureHyperparameters, working_bytes: int,
) -> OccasionModelFit:
    """Every person's exact level posterior at given model parameters, learning nothing: the levels of people
    outside the fit, and the exactness checks. ``prior`` supplies the lattice of ``hyperparameters``."""
    model = _Model(occasions, exponent, working_bytes)
    state = _State(np.asarray(fixed_effects, dtype=np.float64), float(level_variance), prior, hyperparameters)
    expectation = model.expectation(state, louis=False)
    return _result(model, state, expectation, expectation.log_likelihood)


INDEPENDENT_EFFECTS = CurvatureCorrection()
"""No EP correction: independent effects (normal means), whose cavities do not move with the prior, so C = 0."""


def quadrature_majorant_ratio(
    prior: ScaleMixturePrior, hyperparameters: MixtureHyperparameters, cavity: Cavity, working_bytes: int
) -> float:
    """sum_j M_j / Z_j with M_j = max(Z_j, sum_k pi_k |L_j(t_k + i pi/2)|), for ``spacing_bound``.

    At t + i pi/2 the variance is i v, so log|L| = -log(1 + q^2)/4 + h^2 v q / (2 (1 + q^2)), at every node with its
    own v (``_kernel_terms``).
    """
    log_density = class_log_density(prior, hyperparameters.coefficients)
    scales = log_scale(prior, hyperparameters.coefficients)
    total = 0.0
    for class_position, class_rows in enumerate(prior.class_rows):
        for rows in _row_chunks(class_rows, prior.grid_size, working_bytes):
            with np.errstate(over="ignore"):
                variance = np.exp(scales[rows][:, None] + prior.log_variance_grid[None, :])
            ratio = variance * cavity.precision[rows][:, None]
            shift_square = np.square(cavity.shift[rows])[:, None]
            log_real = log_density[class_position] - 0.5 * np.log1p(ratio) + 0.5 * shift_square * variance / (1.0 + ratio)
            log_strip = log_density[class_position] - 0.25 * np.log1p(ratio * ratio) + 0.5 * shift_square * variance * ratio / (1.0 + ratio * ratio)
            total += float(np.sum(np.exp(np.maximum(_log_sum_exp(log_strip, axis=1) - _log_sum_exp(log_real, axis=1), 0.0))))
    return total


def diagonal_posterior(variance: NDArray[np.float64]) -> GaussianPosterior:
    """The posterior of independent effects (orthogonal design, normal means): Sigma = diag(variance). There the
    cavities do not move with the prior, so B equals the fixed-cavity curvature."""
    column = np.asarray(variance, dtype=np.float64)[:, None]
    return GaussianPosterior(solve=lambda right, _relative_tolerance: column * right, variance_jvp=lambda weights: -np.square(column) * weights, exact=True)


def _total_curvature(
    prior: ScaleMixturePrior, coefficients: NDArray[np.float64], cavity: Cavity, posterior: GaussianPosterior, working_bytes: int, relative_tolerance: float
) -> NDArray[np.float64]:
    """B in x: M' B_z M (``_total_curvature_columns`` on every direction)."""
    directions = prior.coefficient_map
    total = directions.T @ _total_curvature_columns(prior, coefficients, cavity, posterior, working_bytes, relative_tolerance, directions)
    return 0.5 * (total + total.T)


def _line(
    prior: ScaleMixturePrior, log_smoothing: NDArray[np.float64], origin: NDArray[np.float64], direction: NDArray[np.float64], cavity: Cavity, working_bytes: int,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """``_lines`` for one direction: the penalized objective F(x + t b) - P(x + t b) as a function of the steps t."""
    evaluate = _lines(prior, log_smoothing, origin, direction[:, None], cavity, working_bytes)
    return lambda steps: evaluate({0: np.asarray(steps, dtype=np.float64)})[0]


def _line_log_integral(
    prior: ScaleMixturePrior, log_smoothing: NDArray[np.float64], origin: NDArray[np.float64], direction: NDArray[np.float64], value: float, cavity: Cavity, working_bytes: int, share: float
) -> float:
    """``_line_log_integrals`` for one direction."""
    return float(_line_log_integrals(prior, log_smoothing, origin, direction[:, None], value, cavity, working_bytes, share)[0])


def rounded_operand(values: Any, relative_error: float, array_module: Any) -> Any:
    """The operand the int8 digit split multiplies: the fewest digits meeting the normwise bound.

    m digits keep each entry within 2^-(7m - 2) of its column's maximum and leave zeros zero, so a
    column's rounding is at most sqrt(nnz) 2^-(7m - 2) max|v|. m is the least count meeting
    relative_error * ||v|| for every column; once 2^-(7m - 2) reaches float64's resolution the
    operand is exact.
    """
    if relative_error <= 0.0:
        return values
    magnitude = array_module.max(array_module.abs(values), axis=0)
    norm = array_module.linalg.norm(values, axis=0)
    support = array_module.sum(values != 0.0, axis=0)
    needed = array_module.sqrt(support) * magnitude / array_module.maximum(relative_error * norm, np.finfo(np.float64).tiny)
    exponent = float(array_module.max(array_module.log2(array_module.maximum(needed, 1.0))))
    digits = int(np.ceil((exponent + 2.0) / DIGIT_BITS))
    if DIGIT_BITS * digits - 2 >= np.finfo(np.float64).nmant:
        return values
    scale = array_module.exp2((DIGIT_BITS * digits - 2) - array_module.ceil(array_module.log2(array_module.where(magnitude > 0, magnitude, 1.0))))
    return array_module.rint(values * scale[None, :]) / scale[None, :]


class DenseDualTile:
    """A block held as dense standardized columns; its products round the operand as the int8 split does."""

    def __init__(self, values: Any, array_module: Any) -> None:
        self.values = values
        self.array_module = array_module

    def sample_operand(self, left: Any, relative_error: float) -> Any:
        return rounded_operand(left, relative_error, self.array_module)

    def rmatmat(self, left: Any) -> Any:
        return self.values.T @ left

    def matmat(self, right: Any) -> Any:
        return self.values @ right

    def accumulate_matmat(self, right: Any, image: Any, relative_error: float) -> None:
        image += self.values @ rounded_operand(right, relative_error, self.array_module)

    def weighted_column_squares(self, weights: Any) -> Any:
        return (self.values * self.values).T @ weights

    def columns(self, local: Any) -> Any:
        return self.values[:, local]


class DenseDualSource:
    """Dense standardized genotypes (n x p) cut into contiguous blocks."""

    def __init__(self, genotypes: Any, block_bounds: list[tuple[int, int]], array_module: Any = np) -> None:
        self.array_module = array_module
        self.genotypes = array_module.asarray(genotypes, dtype=array_module.float64)
        self.sample_count, self.variant_count = (int(extent) for extent in self.genotypes.shape)
        self.block_bounds = _checked_bounds(block_bounds, self.variant_count)

    def blocks(self) -> Iterator[tuple[int, int, DualTile]]:
        for start, stop in self.block_bounds:
            yield start, stop, DenseDualTile(self.genotypes[:, start:stop], self.array_module)

    def map_reduce(self, work: Callable[..., None], shared: dict, rows: dict, image_shape: tuple[int, ...]) -> Any:
        return sequential_map_reduce(self, work, shared, rows, image_shape)


def mean_from_dual(source: DualTileSource, models: DualModels, prior_mean: Any, duals: Any, count: PassCount) -> Any:
    """mu = m + D Xt' z per model, one read."""
    array_module = source.array_module
    left = models.sample_to_design(duals, array_module.arange(models.model_count))
    mean = prior_mean.copy()
    for start, stop, tile in source.blocks():
        mean[start:stop] += models.variances[start:stop] * tile.rmatmat(left)
    count.note(int(duals.shape[1]), 0.0, "mean")
    return mean


def cavities(posterior_mean: NDArray[np.float64], posterior_variance: NDArray[np.float64], site_precision: NDArray[np.float64], site_shift: NDArray[np.float64]) -> Cavity:
    """Divide each site out of q's marginal."""
    marginal_precision = 1.0 / posterior_variance
    return Cavity(precision=marginal_precision - site_precision, shift=posterior_mean * marginal_precision - site_shift)
