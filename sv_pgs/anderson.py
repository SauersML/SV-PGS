"""Anderson(m) acceleration for fixed-point iterations.

Given a map T : R^d -> R^d, classical Anderson(m) extrapolates the next
iterate from the last m residuals r_i = T(x_i) - x_i, gaining local
Krylov-style speedup. Each caller guards the proposal against its own
objective.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sv_pgs._typing import F64Array


_CONDITION_LIMIT = 1.0e12


@dataclass(slots=True)
class AndersonState:
    """Rolling history for Anderson(m). Newest entry is index 0."""

    memory_depth: int
    iterates: list[F64Array] = field(default_factory=list)
    map_values: list[F64Array] = field(default_factory=list)
    residuals: list[F64Array] = field(default_factory=list)
    fallback_count: int = 0

    def reset(self) -> None:
        self.iterates.clear()
        self.map_values.clear()
        self.residuals.clear()
        self.fallback_count = 0


def _push(history: list[F64Array], value: F64Array, depth: int) -> None:
    history.insert(0, value)
    if len(history) > depth:
        history.pop()


def anderson_step(
    state: AndersonState,
    *,
    x_current: F64Array,
    map_value: F64Array,
    regularization: float = 1e-10,
) -> F64Array:
    """Return the proposed accelerated iterate; update state in place.

    On the first call (empty history) returns ``map_value`` (plain step).
    """
    x_flat = np.asarray(x_current, dtype=np.float64).ravel()
    t_flat = np.asarray(map_value, dtype=np.float64).ravel()
    residual = t_flat - x_flat

    if not state.residuals:
        _push(state.iterates, x_flat, state.memory_depth)
        _push(state.map_values, t_flat, state.memory_depth)
        _push(state.residuals, residual, state.memory_depth)
        return t_flat.copy()

    # Build difference matrices using existing history (before pushing new).
    # Columns: newest difference first.
    prev_residuals = state.residuals
    prev_map_values = state.map_values
    num_columns = len(prev_residuals)
    # Build the difference matrices by stacking [current, *prev] and taking
    # the per-row diff along the history axis. This replaces a Python loop
    # over history depth with a single C-level np.diff.
    if num_columns == 1:
        delta_residuals = (residual - prev_residuals[0]).reshape(-1, 1)
        delta_map_values = (t_flat - prev_map_values[0]).reshape(-1, 1)
    else:
        stacked_residuals = np.empty((x_flat.size, num_columns + 1), dtype=np.float64)
        stacked_map_values = np.empty((x_flat.size, num_columns + 1), dtype=np.float64)
        stacked_residuals[:, 0] = residual
        stacked_map_values[:, 0] = t_flat
        # np.column_stack would allocate twice; copy directly into the buffer.
        for column_index, (prev_r, prev_t) in enumerate(
            zip(prev_residuals, prev_map_values, strict=True), start=1
        ):
            stacked_residuals[:, column_index] = prev_r
            stacked_map_values[:, column_index] = prev_t
        # We want column k = stacked[:, k] - stacked[:, k+1] (negated np.diff).
        delta_residuals = -np.diff(stacked_residuals, axis=1)
        delta_map_values = -np.diff(stacked_map_values, axis=1)

    proposal = t_flat.copy()
    use_fallback = False
    try:
        # Tikhonov-regularized normal equations via QR for conditioning check.
        # cond of dR is what matters; use SVD for the conditioning estimate.
        singular_values = np.linalg.svd(delta_residuals, compute_uv=False)
        if singular_values.size == 0 or singular_values[0] == 0.0:
            use_fallback = True
        else:
            smallest = singular_values[-1]
            condition_number = (
                np.inf if smallest <= 0.0 else singular_values[0] / smallest
            )
            if not np.isfinite(condition_number) or condition_number > _CONDITION_LIMIT:
                use_fallback = True
        if not use_fallback:
            gram = delta_residuals.T @ delta_residuals
            gram_regularized = gram + regularization * np.eye(num_columns)
            rhs = delta_residuals.T @ residual
            gamma = np.linalg.solve(gram_regularized, rhs)
            if not np.all(np.isfinite(gamma)):
                use_fallback = True
            else:
                proposal = t_flat - delta_map_values @ gamma
                if not np.all(np.isfinite(proposal)):
                    use_fallback = True
                    proposal = t_flat.copy()
    except np.linalg.LinAlgError:
        use_fallback = True
        proposal = t_flat.copy()

    if use_fallback:
        state.fallback_count += 1

    _push(state.iterates, x_flat, state.memory_depth)
    _push(state.map_values, t_flat, state.memory_depth)
    _push(state.residuals, residual, state.memory_depth)
    return proposal
