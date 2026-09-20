"""Closed-form fold updates for the Gaussian surrogate of an EP-EB fit, and an honest nested plan for fold fits.

A benchmark fits each target once per fold, on training sets that overlap in most samples. Two kinds of closed form
apply, and they differ in what they may be used for:

- **At fixed sites, exact.** With the EP sites (tau_j, nu_j), the noise sigma^2 and a flat prior on the covariate
  coefficients held fixed, q(beta) is a Gaussian linear model, and removing any block R of rows is exact algebra
  (``leave_blocks_out``). Through the n x n projected precision P (below), every block's leave-out prediction, its
  predictive variance and the leave-out posterior mean follow from one factorization of the full-sample kernel.
- **Not an honest fold fit.** Sites and hyperparameters fitted on all samples have seen every block's targets, so the
  fixed-site downdate of a full-sample fit is NOT the fold's fit: the rows' influence through the EP sites and the EB
  hyperparameters stays in it. It is a diagnostic only (approximate leave-out, ALO), never a benchmark result.

A fold's honest fit uses its training rows alone. ``nested_fold_plan`` orders the folds' fits so that each one is
warm-started from a fit on a SUBSET of its own training rows (divide and conquer over the held-out blocks). Every fit
in the chain leading to a fold's result sees only rows that fold trains on, so no held-out target reaches it by any
route, and the warm start only moves where the fold's own certified fit starts.

The identities (flat covariate prior; site prior beta ~ N(m, D), m = nu / tau, D = diag(1 / tau); u = y - G m):

- V = sigma^2 I + G D G',  P = V^-1 - V^-1 C (C' V^-1 C)^-1 C' V^-1 (the covariate-projected precision).
- The fitted values are y - sigma^2 P u, and the posterior mean is mu = m + D G' P u.
- Leaving out a block R: its residual is (P_RR)^-1 (P u)_R, its predictive covariance (including noise) is (P_RR)^-1,
  and the leave-out projected precision on the rest F is P_FF - P_FR (P_RR)^-1 P_RF, so mu^(-R) = m + D G_F' P^(F) u_F.

The kernel form needs every tau_j > 0. With non-positive sites (EP's are unclipped), the same quantities come from the
joint precision of (alpha, beta) and a Woodbury downdate of its rows (``_primal_leave_blocks_out``), exact whenever the
joint precision is positive definite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import linalg

from sv_pgs._typing import F64Array, I64Array


@dataclass(frozen=True)
class FixedSites:
    """EP site precisions tau (p,), site shifts nu (p,) and the noise variance sigma^2, held fixed."""

    precision: F64Array
    shift: F64Array
    noise: float


@dataclass(frozen=True)
class BlockLeaveOut:
    """One block's leave-out quantities at fixed sites: the predictive mean and variance of y on the block's rows (in
    the block's order, noise included) and the posterior mean of beta from the other rows."""

    rows: I64Array
    prediction: F64Array
    variance: F64Array
    mean: F64Array


def _symmetric_solve(matrix: F64Array, right: F64Array) -> F64Array:
    return linalg.solve(matrix, right, assume_a="sym")


def _kernel_leave_blocks_out(genotypes: F64Array, covariates: F64Array, target: F64Array, sites: FixedSites,
                             blocks: Sequence[I64Array]) -> list[BlockLeaveOut]:
    """The n x n form: one kernel V, its covariate-projected precision P, then every block from P's blocks."""
    variance = 1.0 / sites.precision
    prior_mean = sites.shift * variance
    residual = target - genotypes @ prior_mean
    kernel = sites.noise * np.eye(genotypes.shape[0]) + (genotypes * variance) @ genotypes.T
    inverse = linalg.inv(kernel, check_finite=False)
    inverse = 0.5 * (inverse + inverse.T)
    left = inverse @ covariates
    projected = inverse - left @ _symmetric_solve(covariates.T @ left, left.T)
    weighted = projected @ residual
    results = []
    everything = np.arange(genotypes.shape[0])
    for block in blocks:
        rows = np.asarray(block, dtype=np.int64)
        rest = np.setdiff1d(everything, rows, assume_unique=True)
        block_precision = projected[np.ix_(rows, rows)]
        leave_out_residual = _symmetric_solve(block_precision, weighted[rows])
        predictive = linalg.inv(block_precision, check_finite=False)
        cross = projected[np.ix_(rest, rows)]
        rest_precision = projected[np.ix_(rest, rest)] - cross @ _symmetric_solve(block_precision, cross.T)
        mean = prior_mean + variance * (genotypes[rest].T @ (rest_precision @ residual[rest]))
        results.append(BlockLeaveOut(rows=rows, prediction=target[rows] - leave_out_residual,
                                     variance=np.diag(predictive).copy(), mean=mean))
    return results


def _primal_leave_blocks_out(genotypes: F64Array, covariates: F64Array, target: F64Array, sites: FixedSites,
                             blocks: Sequence[I64Array]) -> list[BlockLeaveOut]:
    """The (k + p)-dimensional form: the joint precision of (alpha, beta), downdated by each block's rows (Woodbury)."""
    design = np.hstack([covariates, genotypes])
    covariate_count = covariates.shape[1]
    prior_precision = np.concatenate([np.zeros(covariate_count), sites.precision])
    precision = design.T @ design / sites.noise + np.diag(prior_precision)
    information = design.T @ target / sites.noise + np.concatenate([np.zeros(covariate_count), sites.shift])
    factor = linalg.cho_factor(precision, check_finite=False)
    results = []
    for block in blocks:
        rows = np.asarray(block, dtype=np.int64)
        block_design = design[rows]
        # Lambda_F = Lambda - Z_R' Z_R / sigma^2, h_F = h - Z_R' y_R / sigma^2 (training rows only).
        solved = linalg.cho_solve(factor, block_design.T, check_finite=False)
        capacitance = sites.noise * np.eye(rows.shape[0]) - block_design @ solved
        downdated = information - block_design.T @ target[rows] / sites.noise
        base = linalg.cho_solve(factor, downdated, check_finite=False)
        coefficients = base + solved @ _symmetric_solve(capacitance, block_design @ base)
        leave_out_covariance = block_design @ solved + block_design @ solved @ _symmetric_solve(capacitance, block_design @ solved)
        results.append(BlockLeaveOut(rows=rows, prediction=block_design @ coefficients,
                                     variance=sites.noise + np.diag(leave_out_covariance),
                                     mean=coefficients[covariate_count:]))
    return results


def leave_blocks_out(genotypes: F64Array, covariates: F64Array, target: F64Array, sites: FixedSites,
                     blocks: Sequence[I64Array]) -> list[BlockLeaveOut]:
    """Each block's exact leave-out prediction, predictive variance and posterior mean at FIXED sites.

    ``genotypes`` (n x p) and ``covariates`` (n x k, intercept included, flat prior) are the design; ``blocks`` are
    disjoint row sets. The kernel form is used when every site precision is positive and p exceeds n, else the joint
    form. At sites fitted on all rows this is approximate leave-out (a diagnostic), not an honest fold fit.
    """
    genotypes = np.asarray(genotypes, dtype=np.float64)
    covariates = np.asarray(covariates, dtype=np.float64).reshape(genotypes.shape[0], -1)
    target = np.asarray(target, dtype=np.float64)
    if np.all(sites.precision > 0.0) and genotypes.shape[1] > genotypes.shape[0]:
        return _kernel_leave_blocks_out(genotypes, covariates, target, sites, blocks)
    return _primal_leave_blocks_out(genotypes, covariates, target, sites, blocks)


def rescale_sites(precision: F64Array, shift: F64Array, old_scale: F64Array, new_scale: F64Array) -> tuple[F64Array, F64Array]:
    """The same Gaussian site on raw coefficients, expressed on another fit's standardized coefficients.

    A standardized coefficient is b_j = beta_j * s_j for the column scale s_j. A site N(b; nu / tau, 1 / tau) on the
    old scale is, with r = s_new / s_old, N(b'; r nu / tau, r^2 / tau) on the new one: tau' = tau / r^2 and
    nu' = nu / r. Carried states keep their meaning across folds whose training SDs differ.
    """
    ratio = np.asarray(new_scale, dtype=np.float64) / np.asarray(old_scale, dtype=np.float64)
    return np.asarray(precision, dtype=np.float64) / np.square(ratio), np.asarray(shift, dtype=np.float64) / ratio


@dataclass(frozen=True)
class PlanNode:
    """One fit in a nested plan: its training rows, the folds whose results descend from it, and its parent's index
    in the plan (``None`` for the root, which fits the rows every fold trains on, possibly none)."""

    rows: I64Array
    folds: tuple[int, ...]
    parent: int | None


def nested_fold_plan(held_out: Sequence[I64Array], sample_count: int) -> tuple[PlanNode, ...]:
    """The divide-and-conquer order of the folds' fits.

    A node for a set S of folds fits the rows every fold in S trains on (all rows minus the union of S's held-out
    blocks). Its children split S in two and add rows, so each child's rows contain its parent's, and a leaf (one fold)
    fits exactly that fold's training rows. Every node is listed after its parent.
    """
    blocks = [np.asarray(block, dtype=np.int64) for block in held_out]
    everything = np.arange(int(sample_count))
    nodes: list[PlanNode] = []

    def rows_for(folds: Sequence[int]) -> I64Array:
        excluded = np.concatenate([blocks[fold] for fold in folds]) if folds else np.zeros(0, dtype=np.int64)
        return np.setdiff1d(everything, excluded)

    def add(folds: tuple[int, ...], parent: int | None) -> None:
        nodes.append(PlanNode(rows=rows_for(folds), folds=folds, parent=parent))
        index = len(nodes) - 1
        if len(folds) > 1:
            half = len(folds) // 2
            add(folds[:half], index)
            add(folds[half:], index)

    add(tuple(range(len(blocks))), None)
    return tuple(nodes)


def check_honest(plan: Sequence[PlanNode], held_out: Sequence[I64Array], sample_count: int) -> None:
    """Raise unless every node's rows avoid the held-out block of every fold below it, each child contains its parent's
    rows, and each leaf fits exactly its fold's training rows (every sample outside its held-out block)."""
    blocks = [np.asarray(block, dtype=np.int64) for block in held_out]
    everything = np.arange(int(sample_count))
    for index, node in enumerate(plan):
        for fold in node.folds:
            if np.intersect1d(node.rows, blocks[fold]).size:
                raise AssertionError(f"node {index} fits rows held out by fold {fold}")
        if node.parent is not None and np.setdiff1d(plan[node.parent].rows, node.rows).size:
            raise AssertionError(f"node {index} drops rows its parent fitted")
        if len(node.folds) == 1 and not np.array_equal(node.rows, np.setdiff1d(everything, blocks[node.folds[0]])):
            raise AssertionError(f"leaf {index} does not fit exactly its fold's training rows")


def run_nested(plan: Sequence[PlanNode], fit: Callable[[I64Array, object | None], object]) -> dict[int, object]:
    """Fit the plan in order, each node warm-started from its parent's result; return each fold's leaf result.

    ``fit(rows, warm)`` sees only the node's rows and the parent's result (``None`` at the root, and for a root that has
    no rows its children start cold). By ``check_honest`` those rows never include a descendant fold's held-out block.
    """
    results: list[object | None] = []
    leaves: dict[int, object] = {}
    for node in plan:
        warm = results[node.parent] if node.parent is not None else None
        result = fit(node.rows, warm) if node.rows.size else warm
        results.append(result)
        if len(node.folds) == 1:
            leaves[node.folds[0]] = result
    return leaves
