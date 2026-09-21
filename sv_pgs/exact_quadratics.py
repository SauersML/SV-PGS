"""Exact bulk quadratics for the LD blocks that the marginal-variance certificate flags.

marginal_variances replaces K_S^-1 on a block's window by a deterministic equivalent (its identity 2), and
block_information_certificate flags the blocks where that misses. For a flagged block b this module computes,
from the dual solver,

    M_b = Xt_b' Q Xt_b,   Q = K_S^-1 - Z_L core^-1 Z_L'   (every resolved site, not only the window's),

on b's bulk sites. Identity 1 then gives Sigma_jj = D_j - D_j^2 M_jj and the information D_j - Sigma_jj =
D_j^2 M_jj exactly, with no window, far-field or spike-expectation term.

The block enters as r directions u_k = Xt_b c_k over its bulk columns: the columns themselves (C = I on the
nonzero ones; a zero column has M_j. = 0 exactly), or the eigenvectors of the bulk Gram R_b whose eigenvalues
exceed its float64 resolution |b| eps lambda_max, whichever is fewer. Writing xt_j = sum_k u_k C_jk + d_j
(d_j = 0 for the columns; ||d_j||^2 <= sum over the dropped directions of lambda_k V_jk^2, plus the cut, for
the eigenvectors), each computed entry M_hat_ij = sum_k (xt_i' w_hat_k) C_jk has a certified error bound:

- the K_S solve: w_hat_k = y_hat_k - Z_hat core_hat^-1 g_hat_k, y_hat_k = K_S^-1 (u_k - r_k), so
  |xt_i' K_S^-1 r_k| <= ||xt_i|| ||r_k|| (K_S >= I);
- the resolved block, from the refresh's Z_hat (exact residuals R_L = Xt_L - K_S Z_hat) and core_hat = L L':
  with h_i = Z_L'xt_i and g_k = Z_L'u_k, ||h_i - h_hat_i|| <= ||R_L|| ||xt_i|| and ||g_k - g_hat_k|| <=
  ||R_L|| ||u_k||, and L' core^-1 L lies within delta / (1 - delta) of I (dual_solve.core_bounds). So
  |h_hat' core_hat^-1 g_hat - h' core^-1 g| <= a_i b_k + c_i e_k + kappa c_i z_k with a_i = ||R_L|| ||xt_i|| /
  sqrt(lambda_min), b_k = ||L^-1 g_hat_k||, c_i = ||L^-1 h_hat_i|| + a_i, e_k = ||R_L|| ||u_k|| / sqrt(lambda_min),
  z_k = b_k + e_k and kappa = delta / (1 - delta);
- the dropped part: |xt_i' Q d_j| <= ||Q|| ||xt_i|| ||d_j||, with ||Q|| <= 1 + ||L^-1 Z_L'||^2 / (1 - delta).

The solve starts at a relative residual equal to the requested relative error (the columns' own scale, where
M_jj <= ||xt_j||^2) and is tightened once, by the shortfall of the part it controls. When the resolved block's
own terms exceed the request, tightening the solve cannot help, and the result says so (``met`` False); the
refresh's Z_L is then what needs tightening.

Cost: r CG columns per block, batched over the blocks of one call, plus one image and one back-product read.
The products are (sum of the blocks' bulk sites) x (sum of their directions), so a call takes a few blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from sv_pgs.dual_solve import DualGaussian, core_bounds
from sv_pgs.marginal_variances import BlockGrams

EPS = float(np.finfo(np.float64).eps)


def _host(values) -> np.ndarray:
    return values.get() if hasattr(values, "get") else np.asarray(values)


@dataclass(frozen=True)
class BlockQuadratics:
    """One block's exact bulk quadratics.

    ``quadratic`` is M_b = Xt_b' Q Xt_b over the block's variants (|b| x |b|, zero on resolved sites), and
    ``error`` bounds each entry's distance from the exact value. A bulk site's variance is D_j - D_j^2 M_jj and
    its information D_j^2 M_jj. ``columns`` counts the K_S solve's columns for this block, and ``met`` says
    whether every diagonal entry's bound is within the requested relative error of its value.
    """

    block: int
    sites: NDArray[np.int64]
    quadratic: NDArray[np.float64]
    error: NDArray[np.float64]
    columns: int
    met: bool


@dataclass(frozen=True)
class _Plan:
    block: int
    members: NDArray[np.int64]
    bulk: NDArray[np.int64]
    coefficients: NDArray[np.float64]
    dropped: NDArray[np.float64]


def _storage_unit(stored) -> float:
    """The unit roundoff of a Gram stored coarser than float64 (zero for float64 storage)."""
    dtype = np.asarray(stored).dtype
    return float(np.finfo(dtype).eps) / 2 if np.issubdtype(dtype, np.floating) and np.finfo(dtype).eps > EPS else 0.0


def _plan(
    block: int, members: NDArray[np.int64], is_bulk: NDArray[np.bool_], gram: NDArray[np.float64], squares: NDArray[np.float64], storage_unit: float
) -> _Plan:
    """The block's directions: its nonzero bulk columns, or R_b's resolvable eigenvectors when fewer.

    R_b's resolution is the eigensolver's |b| eps lambda_max plus, when the Gram is stored coarser than float64
    (Stage 0's float32 LdGramStore, unit roundoff ``storage_unit``), its storage error: each entry is rounded once,
    so the stored Gram is within storage_unit ||R_b||_F of the design's in the 2-norm, and by Weyl so is every
    eigenvalue, and ||d_j||^2 = (P R_b P)_jj by at most that more than the stored Gram's.
    """
    bulk = np.flatnonzero(is_bulk[members])
    live = squares[members[bulk]] > 0.0
    bulk_gram = gram[np.ix_(bulk, bulk)]
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (bulk_gram + bulk_gram.T)) if bulk.size else (np.zeros(0), np.zeros((0, 0)))
    cut = bulk.size * EPS * float(eigenvalues[-1]) + storage_unit * float(np.linalg.norm(bulk_gram)) if bulk.size else 0.0
    kept = eigenvalues > cut
    if int(np.sum(kept)) < int(np.sum(live)):
        dropped = np.sqrt(np.square(eigenvectors[:, ~kept]) @ np.maximum(eigenvalues[~kept], 0.0) + cut)
        return _Plan(block, members, bulk, eigenvectors[:, kept], dropped)
    return _Plan(block, members, bulk, np.eye(bulk.size)[:, live], np.zeros(bulk.size))


def exact_block_quadratics(
    gaussian: DualGaussian, model: int, grams: BlockGrams, blocks: Sequence[int], relative_error: float
) -> list[BlockQuadratics]:
    """M_b for each of ``blocks`` of ``model`` at the last iterate's sites, each diagonal entry certified to
    ``relative_error`` of its value when the resolved block allows it (module docstring).

    ``grams`` are the model's metric Grams (marginal_variances.BlockGrams, which may be Stage 0's shared float32
    arrays with the model's ``scale``); they choose the directions and bound the dropped part only.
    The norms come from the solver: ||xt_j||^2 from its column squares and ||u_k|| from the image.
    """
    if not relative_error > 0.0:
        raise ValueError("relative_error must be positive")
    array_module = gaussian.array_module
    solve = gaussian.bulk_solves[model]
    variant_count = solve.site_precision.shape[0]
    is_bulk = np.ones(variant_count, dtype=bool)
    is_bulk[solve.resolved] = False
    squares = _host(gaussian.unit_squares)[:, model] / float(gaussian.noise_variance[model])
    plans = [_plan(block, grams.blocks[block], is_bulk, grams.within_block(block), squares, _storage_unit(grams.within[block])) for block in blocks]
    sites = np.concatenate([plan.members[plan.bulk] for plan in plans]) if plans else np.zeros(0, dtype=np.int64)
    order = np.argsort(sites, kind="stable")
    widths = [int(plan.coefficients.shape[1]) for plan in plans]
    row_offsets = np.concatenate([[0], np.cumsum([plan.bulk.size for plan in plans])]).astype(np.int64)
    column_offsets = np.concatenate([[0], np.cumsum(widths)]).astype(np.int64)
    coefficients = np.zeros((sites.size, int(column_offsets[-1])))
    for index, plan in enumerate(plans):
        coefficients[row_offsets[index] : row_offsets[index + 1], column_offsets[index] : column_offsets[index + 1]] = plan.coefficients
    norms = np.sqrt(squares[sites])
    if not coefficients.shape[1]:
        # nothing bulk and polymorphic: every M_b is exactly zero
        return [BlockQuadratics(plan.block, plan.members, np.zeros((plan.members.size,) * 2), np.zeros((plan.members.size,) * 2), 0, True) for plan in plans]

    resolved = gaussian.resolved_block(model)
    if resolved is not None:
        lowest, residual_norm, delta = core_bounds(array_module, resolved)
        whitened_duals = _host(array_module.linalg.solve(resolved.factor, resolved.duals.T))
        duals_norm = float(np.sqrt(max(float(np.linalg.eigvalsh(whitened_duals @ whitened_duals.T)[-1]), 0.0)))
        root_lowest = float(np.sqrt(lowest))
    tolerance = relative_error
    refined = False
    while True:
        try:
            products, coupling, residuals, image_norms = gaussian.block_products(model, sites[order], coefficients[order], sites[order], tolerance)
        except ValueError:
            if not refined:
                raise
            break
        products = _host(products)[np.argsort(order, kind="stable")]
        residual_norms = _host(residuals) * _host(image_norms)
        image_norms = _host(image_norms)
        whitened_coupling = _host(array_module.linalg.solve(resolved.factor, coupling)) if resolved is not None else np.zeros((0, sites.size))
        results = []
        shortfall = np.inf
        for index, plan in enumerate(plans):
            rows = slice(int(row_offsets[index]), int(row_offsets[index + 1]))
            columns = slice(int(column_offsets[index]), int(column_offsets[index + 1]))
            weights = np.abs(plan.coefficients)
            row_norms = norms[rows]
            estimate = products[rows, columns] @ plan.coefficients.T
            solver = np.outer(row_norms, residual_norms[columns]) @ weights.T
            scaled = np.outer(row_norms, tolerance * image_norms[columns]) @ weights.T
            fixed = np.zeros_like(estimate)
            if resolved is not None:
                if not delta < 1.0:
                    fixed = np.full_like(estimate, np.inf)
                else:
                    kappa = delta / (1.0 - delta)
                    loadings = whitened_coupling[:, columns]
                    projected = np.linalg.norm(loadings @ plan.coefficients.T, axis=0) + duals_norm * plan.dropped
                    row_error = residual_norm * row_norms / root_lowest
                    row_whitened = projected + row_error
                    direction_whitened = np.linalg.norm(loadings, axis=0)
                    direction_error = residual_norm * image_norms[columns] / root_lowest
                    terms = np.outer(row_error, direction_whitened) + np.outer(row_whitened, direction_error)
                    terms += kappa * np.outer(row_whitened, direction_whitened + direction_error)
                    operator = 1.0 + (duals_norm + residual_norm / root_lowest) ** 2 / (1.0 - delta)
                    fixed = terms @ weights.T + operator * np.outer(row_norms, plan.dropped)
            else:
                fixed = np.outer(row_norms, plan.dropped)
            error = solver + fixed
            diagonal = np.diag(estimate)
            live = row_norms > 0.0
            allowed = relative_error * np.abs(diagonal) / (1.0 + relative_error)
            met = bool(np.all(np.diag(error)[live] <= allowed[live]))
            if not met:
                # the factor on the solve's residuals that fits each site's solver part into the room its fixed
                # part leaves; a site with no room is the resolved block's to fix, not the solve's
                room = allowed[live] - np.diag(fixed)[live]
                needed = np.diag(scaled)[live]
                fits = (room > 0.0) & (needed > 0.0)
                if np.any(fits):
                    shortfall = min(shortfall, float(np.min(room[fits] / needed[fits])))
            quadratic = np.zeros((plan.members.size, plan.members.size))
            bound = np.zeros_like(quadratic)
            quadratic[np.ix_(plan.bulk, plan.bulk)] = estimate
            bound[np.ix_(plan.bulk, plan.bulk)] = error
            results.append(BlockQuadratics(plan.block, plan.members, quadratic, bound, widths[index], met))
        if all(result.met for result in results) or refined or not shortfall < 1.0:
            return results
        tolerance *= shortfall
        refined = True
    return results
