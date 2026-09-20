"""Stage 2's EP linear response by block flexible GCRO-DR: every direction in one block Krylov space, a read-free
block-local preconditioner, and a subspace recycled across outer steps (lane speed-recycle).

The total curvature B needs the cavity response dP to every coefficient direction (``scale_mixture_ep.
_total_curvature_columns``, b_products.md): the fixed point of an affine map whose linear part is

    L = (I - diag(w) (Sigma o Sigma)) (diag(l) Sigma diag(r) + diag(d)),

with per-variant vectors w, l, r, d. It applies one p x p operator to each of the r direction columns, so
(I - L) dP = c is r systems with one operator. GMRES on the flattened p r vector builds one scalar polynomial for all r
columns. Block GMRES (O'Leary 1980; Vital 1990) builds the block Krylov space K_j(I - L, c), of dimension up to j r,
from the same j applications. An application of I - L is one posterior solve (Sigma on its columns), which is where the
store reads are, and its r columns ride the same reads (compute_floor.md section 2).

Three more things cut applications without changing what is accepted (the caller's true-residual test):
- **Flexible right preconditioning** (Saad 1993) by the exact inverse of I - L_loc, L with Sigma replaced by the block
  diagonal of the window covariance that ``marginal_variances`` forms (``local_response``). It is read-free, factored
  once per block per fixed point and applied to all columns; the directions a flexible cycle keeps are the ones it
  applied the true operator to, so the preconditioner changes speed only.
- **Deflated restarts** (GCRO-DR: Parks, de Sturler, Mackey, Johnson & Maiti 2006; its flexible form, Carvalho, Gratton,
  Lago & Vasseur 2011). At each restart the harmonic Ritz vectors of the slowest part are kept as U with C = (I - L) U
  orthonormal, so a restarted cycle keeps what the last one learned. It holds one block (the right-hand side's rank):
  refreshing it for a new operator is one application, the cost of one block step.
- **Recycling across outer steps.** U stays in the caller's ``RecycledSpace``. The next outer step's operator differs
  only through the moved sites and hyperparameters, so its image C = (I - L') U, one application, starts that solve
  with the slow directions already in hand; so does the next solve at a tightened inner tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eig, lu_factor, lu_solve

from sv_pgs.marginal_variances import BlockGrams, BulkSolve, _block_terms, _prepare

_EPSILON = float(np.finfo(np.float64).eps)
_ITEM_BYTES = np.dtype(np.float64).itemsize

F64 = NDArray[np.float64]


@dataclass
class RecycledSpace:
    """U (p x k): the search directions kept from the last solve (empty until one ran); owned by the caller per model."""

    vectors: F64 | None = None


@dataclass(frozen=True)
class KrylovSolve:
    """The solution, its true residual norm ||right - A X||_F (measured once, at the end), and what it cost."""

    solution: F64
    residual_norm: float
    target: float
    applications: int
    applied_columns: int
    cycles: int


def _svqb(values: F64) -> tuple[F64, F64]:
    """One SVQB pass: (Q, T) with values ~ Q T, Q'Q ~ I, dropping directions below float64 resolution."""
    if values.shape[1] == 0:
        return values[:, :0], np.zeros((0, 0))
    gram = values.T @ values
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
    largest = float(eigenvalues[-1])
    resolvable = eigenvalues > largest * values.shape[0] * _EPSILON
    if not largest > 0.0 or not bool(np.any(resolvable)):
        return values[:, :0], np.zeros((0, values.shape[1]))
    basis = values @ (eigenvectors[:, resolvable] / np.sqrt(eigenvalues[resolvable]))
    return basis, basis.T @ values


def _orthonormal(values: F64) -> tuple[F64, F64]:
    """(Q, T): values = Q T with Q orthonormal, SVQB twice (the second pass restores orthonormality to rounding)."""
    first, triangle = _svqb(values)
    second, correction = _svqb(first)
    return second, correction @ triangle


def _assemble(blocks: dict, heights: list[int], widths: list[int]) -> F64:
    rows, columns = np.concatenate([[0], np.cumsum(heights)]), np.concatenate([[0], np.cumsum(widths)])
    matrix = np.zeros((int(rows[-1]), int(columns[-1])))
    for (row, column), block in blocks.items():
        if row < len(heights) and column < len(widths):
            matrix[rows[row] : rows[row + 1], columns[column] : columns[column + 1]] = block
    return matrix


def _deflate(search: F64, image_basis: F64, arnoldi: F64, keep: int) -> tuple[F64, F64] | None:
    """The harmonic Ritz space of a flexible cycle: (U, C) with A U = C, C orthonormal, spanning the ``keep`` smallest
    harmonic Ritz values |theta| (one more where that completes a conjugate pair).

    ``search`` W~ holds the directions the operator was applied to and ``image_basis`` W_bar the orthonormal basis of
    their images, A W~ = W_bar G with G = ``arnoldi``; the harmonic Ritz pairs solve G'G y = theta G'(W_bar'W~) y.
    """
    if keep <= 0 or search.shape[1] == 0:
        return None
    values, vectors = eig(arnoldi.T @ arnoldi, arnoldi.T @ (image_basis.T @ search))
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return None
    order = finite[np.argsort(np.abs(values[finite]), kind="stable")]
    chosen = list(order[:keep])
    last = values[chosen[-1]]
    if last.imag != 0.0:
        chosen.extend([index for index in order[keep:] if values[index] == np.conj(last)][:1])
    coefficients = vectors[:, chosen]
    real = np.column_stack([coefficients.real, coefficients.imag[:, np.any(coefficients.imag != 0.0, axis=0)]])
    basis, _triangle = _orthonormal(real)
    if basis.shape[1] == 0:
        return None
    image, triangle = _orthonormal(image_basis @ (arnoldi @ basis))
    if image.shape[1] == 0:
        return None
    return (search @ basis) @ np.linalg.pinv(triangle), image


def block_gcro_dr(
    apply: Callable[[F64], F64],
    right: F64,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
    working_bytes: int,
    application_limit: int,
    start: F64 | None = None,
    precondition: Callable[[F64], F64] | None = None,
    recycled: RecycledSpace | None = None,
) -> KrylovSolve:
    """Block flexible GCRO-DR (module docstring) for A X = right, A = ``apply`` a linear map on p x c blocks.

    It iterates until its recurrence residual is at most max(rtol ||right||_F, atol), then measures the true residual
    once and returns it: whether that meets the caller's test (and what to do if not, e.g. tighten the products) is the
    caller's. The Arnoldi cycle is the longest that ``working_bytes`` holds (its bases and search directions, the
    recycled pair, and the solution, residual and right-hand side). Past ``application_limit`` applications without
    converging it raises FloatingPointError.
    """
    right = np.asarray(right, dtype=np.float64)
    size, width = right.shape
    target = max(float(absolute_tolerance), float(relative_tolerance) * float(np.linalg.norm(right)))
    counts = {"applications": 0, "columns": 0, "cycles": 0}

    def operator(values: F64) -> F64:
        counts["applications"] += 1
        counts["columns"] += int(values.shape[1])
        return np.asarray(apply(values), dtype=np.float64)

    if start is None or not bool(np.any(start != 0.0)):
        solution = np.zeros_like(right)
        residual = right.copy()
    else:
        solution = np.array(start, dtype=np.float64, copy=True)
        residual = right - operator(solution)
    # Whether ``residual`` is the true right - A X, or the recurrence's: only a true one is returned.
    residual_is_true = True
    kept: tuple[F64, F64] | None = None
    if recycled is not None and recycled.vectors is not None and recycled.vectors.shape[0] == size and recycled.vectors.shape[1]:
        image, triangle = _orthonormal(operator(recycled.vectors))
        if image.shape[1]:
            kept = (recycled.vectors @ np.linalg.pinv(triangle), image)
            coefficients = image.T @ residual
            solution += kept[0] @ coefficients
            residual -= image @ coefficients
            residual_is_true = False
    columns_available = int(working_bytes) // (_ITEM_BYTES * size)
    while float(np.linalg.norm(residual)) > target:
        if counts["applications"] >= application_limit:
            raise FloatingPointError(f"the block linear response did not converge within {application_limit} applications")
        counts["cycles"] += 1
        first, coordinates = _orthonormal(residual)
        block = int(first.shape[1])
        recycle_width = 0 if kept is None else int(kept[1].shape[1])
        steps = max(1, (columns_available - 2 * max(recycle_width, block) - 3 * width - block) // (2 * block))
        steps = min(steps, application_limit - counts["applications"])
        bases, searches, couplings = [first], [], []
        heights, widths = [block], []
        hessenberg: dict[tuple[int, int], F64] = {}
        best = None
        for step in range(steps):
            direction = bases[step] if precondition is None else np.asarray(precondition(bases[step]), dtype=np.float64)
            image = operator(direction)
            searches.append(direction)
            widths.append(int(direction.shape[1]))
            if kept is not None:
                coupling = np.zeros((kept[1].shape[1], direction.shape[1]))
                for _pass in range(2):
                    part = kept[1].T @ image
                    image = image - kept[1] @ part
                    coupling += part
                couplings.append(coupling)
            for _pass in range(2):
                for index, basis in enumerate(bases):
                    part = basis.T @ image
                    image = image - basis @ part
                    hessenberg[(index, step)] = hessenberg.get((index, step), 0.0) + part
            new_basis, new_block = _orthonormal(image)
            hessenberg[(step + 1, step)] = new_block
            bases.append(new_basis)
            heights.append(int(new_basis.shape[1]))
            arnoldi = _assemble(hessenberg, heights, widths)
            right_side = np.zeros((arnoldi.shape[0], width))
            right_side[:block] = coordinates
            weights = np.linalg.lstsq(arnoldi, right_side, rcond=None)[0]
            remainder = right_side - arnoldi @ weights
            best = (weights, remainder, arnoldi, list(couplings))
            if float(np.linalg.norm(remainder)) <= target or new_basis.shape[1] == 0:
                break
        weights, remainder, arnoldi, used_couplings = best
        search = np.hstack(searches)
        solution += search @ weights
        if kept is not None:
            coupled = np.hstack(used_couplings)
            solution -= kept[0] @ (coupled @ weights)
        image_basis = np.hstack(bases)
        residual = image_basis @ remainder
        residual_is_true = False
        # The cycle's harmonic Ritz space becomes the kept pair, and the residual's component in its image is solved.
        if kept is None:
            deflated = _deflate(search, image_basis, arnoldi, block)
        else:
            full = np.block([[np.eye(kept[1].shape[1]), coupled], [np.zeros((arnoldi.shape[0], kept[1].shape[1])), arnoldi]])
            deflated = _deflate(np.hstack([kept[0], search]), np.hstack([kept[1], image_basis]), full, block)
        if deflated is not None:
            kept = deflated
            coefficients = kept[1].T @ residual
            solution += kept[0] @ coefficients
            residual -= kept[1] @ coefficients
    if recycled is not None and kept is not None:
        recycled.vectors = kept[0]
    true_residual = residual if residual_is_true else right - operator(solution)
    return KrylovSolve(solution, float(np.linalg.norm(true_residual)), target, counts["applications"], counts["columns"], counts["cycles"])


def local_response(solve: BulkSolve, grams: BlockGrams) -> Callable[[F64, F64, F64, F64], Callable[[F64], F64]]:
    """The block-local preconditioner of the EP linear response: (left, right, diagonal, weight) -> V -> M^-1 V with

        M = I - (I - diag(weight) S2_loc) (diag(left) Sigma_loc diag(right) + diag(diagonal)),

    Sigma_loc the block diagonal of the window covariance ``marginal_variances`` forms (each block's Sigma_bb, bulk and
    resolved rows alike) and S2_loc = Sigma_loc o Sigma_loc. It costs no read: the window algebra is formed once here,
    each block's p_b x p_b M_b is LU-factored once per call, and M^-1 applies to any number of columns.
    """
    variant_count = solve.site_precision.shape[0]
    covered = np.zeros(variant_count, dtype=bool)
    for members in grams.blocks:
        covered[members] = True
    if not bool(np.all(covered)):
        raise ValueError("the blocks must cover every variant")
    # The window algebra is formed at the first call, once per fixed point, not for posteriors B never asks about.
    covariances: list[F64] = []

    def build(left: F64, right: F64, diagonal: F64, weight: F64) -> Callable[[F64], F64]:
        if not covariances:
            cross, bulk_variance, core_inverse, _is_resolved = _prepare(solve, grams)
            covariances.extend(_block_terms(solve, grams, cross, bulk_variance, core_inverse, block).covariance for block in range(len(grams.blocks)))
        factors = []
        for members, covariance in zip(grams.blocks, covariances):
            response = left[members, None] * covariance * right[None, members] + np.diag(diagonal[members])
            variance_map = np.eye(members.shape[0]) - weight[members, None] * np.square(covariance)
            factors.append((members, lu_factor(np.eye(members.shape[0]) - variance_map @ response)))

        def inverse(values: F64) -> F64:
            values = np.asarray(values, dtype=np.float64)
            result = np.empty_like(values)
            for members, factor in factors:
                result[members] = lu_solve(factor, values[members])
            return result

        return inverse

    return build
