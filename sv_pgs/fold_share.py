"""Exact work sharing between related fits: folds that share most samples, nested feature sets, overlapping windows.

A benchmark fits each target about ten times (leave-one-group-out and random folds), on training sets that overlap
in most samples, and on nested column sets (small variants, then small variants plus SVs). Three identities let those
fits share work without changing any result beyond rounding:

- **Folds.** At weights common to the folds, a fold's kernel is a submatrix of the full-sample kernel, so one n²·p
  formation serves every fold. Its Cholesky factor can also be shared: with the held-out block contiguous, the rows
  above it are the full factor's, and the rows below are one QR of a positive rank-|held out| update, stable with no
  subtraction. That pays only when the held-out block is small against n (``sharing_cost`` counts it).
- **Nested columns.** Adding k columns with non-negative weights adds W Wᵀ to the kernel. The factor of K + W Wᵀ is
  one QR of [L, W]ᵀ; signed weights (negative EP sites) go through the capacitance matrix, and every solve carries a
  residual computed from the kernel itself, so a result is reported only with its measured error.
- **Overlapping windows.** Columns that belong to the same set of windows form a segment. A window's kernel is the
  sum of its segments' Grams, each computed once, so a column shared by many windows is multiplied once. With
  non-negative weights every term is positive semi-definite, so the sums suffer no cancellation.

Nothing here chooses a number: tolerances are the rounding bounds of the operations themselves, measured on the data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Mapping, Sequence

import numpy as np
from scipy import linalg

_EPS = np.finfo(np.float64).eps


def _positive_diagonal(triangular: np.ndarray) -> np.ndarray:
    """A triangular factor with the signs of its diagonal made positive (QR leaves them arbitrary)."""
    signs = np.sign(np.diag(triangular))
    signs[signs == 0] = 1
    return triangular * signs[:, None]


def cholesky_of_gram(left: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor of left leftᵀ from one QR of leftᵀ: stable, and never forms the product."""
    upper = linalg.qr(left.T, mode="r", check_finite=False)[0][: left.shape[0]]
    return _positive_diagonal(upper).T


@dataclass(frozen=True)
class FoldOrder:
    """Samples ordered as [always-trained, block 1, ..., block m]; ``bounds[k]`` is block k's slice in that order."""

    order: np.ndarray
    bounds: tuple[slice, ...]

    @classmethod
    def from_held_out(cls, held_out: Sequence[np.ndarray], sample_count: int) -> FoldOrder:
        blocks = [np.asarray(block, dtype=np.int64) for block in held_out]
        seen = np.zeros(sample_count, dtype=bool)
        for block in blocks:
            if np.any(seen[block]):
                raise ValueError("held-out blocks must be disjoint.")
            seen[block] = True
        order = [np.flatnonzero(~seen)]
        bounds = []
        start = order[0].shape[0]
        for block in blocks:
            order.append(np.sort(block))
            bounds.append(slice(start, start + block.shape[0]))
            start += block.shape[0]
        return cls(np.concatenate(order), tuple(bounds))

    def training_order(self, fold: int) -> np.ndarray:
        """The training samples of ``fold``, in the order its factor uses."""
        block = self.bounds[fold]
        return np.concatenate([self.order[: block.start], self.order[block.stop:]])


def fold_factors(full_factor: np.ndarray, folds: FoldOrder) -> Iterator[tuple[int, np.ndarray]]:
    """Each fold's training-kernel Cholesky factor from the full kernel's factor, in ``folds.order``.

    With L the factor of the full kernel and B, H, A the rows before, inside and after the held-out block, the
    training kernel is [[L_BB L_BBᵀ, ·], [L_AB L_BBᵀ, L_AB L_ABᵀ + L_AH L_AHᵀ + L_AA L_AAᵀ]], so its factor keeps L_BB and
    L_AB and replaces the trailing block by the factor of L_AH L_AHᵀ + L_AA L_AAᵀ, one QR of [L_AH, L_AA]ᵀ.
    """
    for fold, block in enumerate(folds.bounds):
        before, after = slice(0, block.start), slice(block.stop, full_factor.shape[0])
        size = full_factor.shape[0] - (block.stop - block.start)
        factor = np.zeros((size, size))
        factor[: block.start, : block.start] = full_factor[before, before]
        if block.stop < full_factor.shape[0]:
            factor[block.start:, : block.start] = full_factor[after, before]
            trailing = np.hstack([full_factor[after, block], full_factor[after, after]])
            factor[block.start:, block.start:] = cholesky_of_gram(trailing)
        yield fold, factor


def logdet_from_factor(factor: np.ndarray) -> float:
    return float(2.0 * np.sum(np.log(np.diag(factor))))


def add_nonnegative_columns(factor: np.ndarray, columns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Factor of L Lᵀ + U diag(c) Uᵀ for c ≥ 0: one QR of [L, U diag(√c)]ᵀ, exact up to QR rounding."""
    weights = np.asarray(weights, dtype=np.float64)
    if np.any(weights < 0):
        raise ValueError("add_nonnegative_columns needs non-negative weights; use SignedColumnUpdate.")
    return cholesky_of_gram(np.hstack([factor, columns * np.sqrt(weights)]))


@dataclass(frozen=True)
class CertifiedSolve:
    solution: np.ndarray
    residual_norm: float
    rounding_bound: float

    @property
    def certified(self) -> bool:
        return self.residual_norm <= self.rounding_bound


class SignedColumnUpdate:
    """Solves and log-determinant of K + U diag(c) Uᵀ with signed c, through the capacitance matrix.

    With G = Uᵀ K⁻¹ U, the update's determinant is det K · det(I + diag(c) G), and a solve is
    x = K⁻¹b − K⁻¹U (diag(c)⁻¹ + G)⁻¹ Uᵀ K⁻¹ b written without dividing by c: (I + diag(c) G)⁻¹ diag(c).
    Every solve returns the residual of the updated kernel applied to x, with the rounding bound of that product,
    so a caller can tell a certified solve from a cancellation-damaged one.
    """

    def __init__(self, factor: np.ndarray, columns: np.ndarray, weights: np.ndarray):
        self.factor = factor
        self.columns = columns
        self.weights = np.asarray(weights, dtype=np.float64)
        self._kernel_inverse_columns = linalg.cho_solve((factor, True), columns, check_finite=False)
        gram = columns.T @ self._kernel_inverse_columns
        self._capacitance = np.eye(columns.shape[1]) + self.weights[:, None] * gram
        self._capacitance_lu = linalg.lu_factor(self._capacitance, check_finite=False)

    def logdet(self) -> tuple[float, float]:
        """(sign, log |det|) of the updated kernel."""
        sign, magnitude = np.linalg.slogdet(self._capacitance)
        return float(sign), logdet_from_factor(self.factor) + float(magnitude)

    def _weighted(self, projected: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return weights[:, None] * projected if projected.ndim == 2 else weights * projected

    def apply(self, vector: np.ndarray) -> np.ndarray:
        """(L Lᵀ + U diag(c) Uᵀ) vector, straight from the pieces."""
        return self.factor @ (self.factor.T @ vector) + self.columns @ self._weighted(self.columns.T @ vector, self.weights)

    def solve(self, right_hand_side: np.ndarray) -> CertifiedSolve:
        base = linalg.cho_solve((self.factor, True), right_hand_side, check_finite=False)
        correction = linalg.lu_solve(self._capacitance_lu, self._weighted(self.columns.T @ base, self.weights), check_finite=False)
        solution = base - self._kernel_inverse_columns @ correction
        residual = self.apply(solution) - right_hand_side
        magnitude = np.abs(self.factor) @ (np.abs(self.factor.T) @ np.abs(solution)) + np.abs(self.columns) @ self._weighted(
            np.abs(self.columns.T) @ np.abs(solution), np.abs(self.weights))
        # γ_m = m·ε/(1 − m·ε) bounds the rounding of an m-term product (Higham 2002, Lemma 3.1); m = factor rows + columns.
        depth = self.factor.shape[0] + self.columns.shape[1]
        gamma = depth * _EPS / (1 - depth * _EPS)
        bound = float(gamma * np.linalg.norm(magnitude + np.abs(right_hand_side)))
        return CertifiedSolve(solution, float(np.linalg.norm(residual)), bound)


@dataclass(frozen=True)
class Segment:
    columns: np.ndarray
    windows: tuple[int, ...]


def window_segments(window_columns: Sequence[np.ndarray]) -> list[Segment]:
    """Group columns by the exact set of windows that contain them; each group's Gram is computed once."""
    membership: dict[int, list[int]] = {}
    for window, columns in enumerate(window_columns):
        for column in np.asarray(columns, dtype=np.int64).tolist():
            membership.setdefault(column, []).append(window)
    groups: dict[tuple[int, ...], list[int]] = {}
    for column in sorted(membership):
        groups.setdefault(tuple(membership[column]), []).append(column)
    segments = [Segment(np.asarray(columns, dtype=np.int64), windows) for windows, columns in groups.items()]
    segments.sort(key=lambda segment: (segment.windows[0], int(segment.columns[0])))
    return segments


def window_kernels(
    column_block: Callable[[np.ndarray], np.ndarray],
    weights: np.ndarray,
    window_columns: Sequence[np.ndarray],
    sample_count: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Each window's kernel X_W diag(w_W) X_Wᵀ, from segment Grams shared by every window that holds them.

    ``column_block(columns)`` returns the [samples, len(columns)] genotype block. Windows are emitted as soon as their
    last segment is added, so only the windows still open are held in memory. Weights must be non-negative, which
    makes every summand positive semi-definite and the sums free of cancellation.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if np.any(weights < 0):
        raise ValueError("window_kernels sums positive semi-definite segment Grams; weights must be non-negative.")
    segments = window_segments(window_columns)
    remaining = np.zeros(len(window_columns), dtype=np.int64)
    for segment in segments:
        remaining[list(segment.windows)] += 1
    open_kernels: dict[int, np.ndarray] = {}
    for window in np.flatnonzero(remaining == 0).tolist():
        yield window, np.zeros((sample_count, sample_count))
    for segment in segments:
        block = column_block(segment.columns)
        gram = (block * weights[segment.columns]) @ block.T
        for window in segment.windows:
            if window in open_kernels:
                open_kernels[window] += gram
            else:
                open_kernels[window] = gram.copy()
            remaining[window] -= 1
            if remaining[window] == 0:
                yield window, open_kernels.pop(window)


def transfer_site_state(
    source_ids: np.ndarray,
    source_state: Mapping[str, np.ndarray],
    target_ids: np.ndarray,
    initial_state: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Per-variant state (e.g. EP site precisions and shifts) carried from a related fit, by variant identity.

    Variants the source fitted keep its state; the rest keep ``initial_state`` (the target's own start, e.g. the
    prior's moment-matched sites). The target's fit still iterates to its own certified fixed point; this only
    changes where it starts.
    """
    source_index = {identifier: row for row, identifier in enumerate(np.asarray(source_ids).tolist())}
    rows = np.array([source_index.get(identifier, -1) for identifier in np.asarray(target_ids).tolist()], dtype=np.int64)
    carried = rows >= 0
    state = {}
    for name, initial in initial_state.items():
        values = np.array(initial, dtype=np.float64, copy=True)
        values[carried] = np.asarray(source_state[name], dtype=np.float64)[rows[carried]]
        state[name] = values
    return state


def common_weight_fold_kernels(full_kernel: np.ndarray, training_sets: Sequence[np.ndarray]) -> Iterator[tuple[int, np.ndarray]]:
    """Each fold's kernel at weights shared by every fold: a submatrix of the full-sample kernel, formed once.

    X_T diag(w) X_Tᵀ is the (T, T) block of X diag(w) Xᵀ, so one n²·p formation replaces Σ_f n_T²·p. This holds only
    while the weights are common to the folds (the prior's variances before any fold's own sites, a fixed state being
    scored, or a baseline); a fold's own EP sites give it its own weights, and then only warm starts carry over.
    """
    for fold, training in enumerate(training_sets):
        rows = np.asarray(training, dtype=np.int64)
        yield fold, full_kernel[np.ix_(rows, rows)]


@dataclass(frozen=True)
class SharingCost:
    """Leading-order multiply-add counts, direct against shared, for one kernel formation per fit."""

    window_direct: float
    window_shared: float
    fold_kernel_direct: float
    fold_kernel_shared: float
    fold_factor_direct: float
    fold_factor_shared: float
    nested_direct: float
    nested_shared: float

    def ratios(self) -> dict[str, float]:
        return {"windows": self.window_direct / self.window_shared,
                "fold_kernels_common_weights": self.fold_kernel_direct / self.fold_kernel_shared,
                "fold_factors": self.fold_factor_direct / self.fold_factor_shared,
                "nested_columns_common_weights": self.nested_direct / self.nested_shared}


def sharing_cost(
    sample_count: int,
    held_out_sizes: Sequence[int],
    window_sizes: Sequence[int],
    unique_columns: int,
    segment_window_memberships: int,
    added_columns: Sequence[int],
) -> SharingCost:
    """Multiply-add counts of each sharing identity for one benchmark (n samples, the given folds and windows).

    - windows: n² per column of every window directly; n² per unique column plus n² per (segment, window) sum shared;
    - fold kernels at common weights: Σ_f n_T² per column directly; n² per column once shared;
    - fold factors: Σ_f n_T³/3 directly; shared, n³/3 plus one Householder QR of the (a + h) × a trailing block per
      fold, 2(a + h)a² − 2a³/3 (Golub and Van Loan). With held-out blocks near a fifth of n this exceeds the direct
      count, so for such designs a fresh factor of the (free) submatrix is the right route;
    - nested columns at common weights: a refit forms n²(p + k) again; the shared route adds n²·k.
    """
    n = float(sample_count)
    mean_window = float(np.mean(window_sizes)) if len(window_sizes) else 0.0
    window_direct = n * n * float(np.sum(window_sizes))
    window_shared = n * n * (float(unique_columns) + float(segment_window_memberships))
    training_sizes = [n - h for h in held_out_sizes]
    fold_kernel_direct = sum(t * t for t in training_sizes) * mean_window
    fold_kernel_shared = n * n * mean_window
    fold_factor_direct = sum(t ** 3 / 3.0 for t in training_sizes)
    fold_factor_shared = n ** 3 / 3.0
    trailing = n
    for h in held_out_sizes:
        trailing -= h
        if trailing > 0:
            fold_factor_shared += 2.0 * (trailing + h) * trailing ** 2 - 2.0 * trailing ** 3 / 3.0
    nested_direct = sum(n * n * (mean_window + k) for k in added_columns)
    nested_shared = sum(n * n * float(k) for k in added_columns)
    return SharingCost(window_direct, window_shared, fold_kernel_direct, fold_kernel_shared, fold_factor_direct,
                       fold_factor_shared, nested_direct, nested_shared)
