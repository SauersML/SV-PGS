"""Safeguarded Anderson(m) acceleration for fixed-point iterations.

Given a map T : R^d -> R^d, classical Anderson(m) extrapolates the next
iterate from the last m residuals r_i = T(x_i) - x_i. With a monotone
safeguard (Henderson-Varadhan) it preserves the ascent property of EM
while gaining local Krylov-style speedup.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

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
    delta_residuals = np.empty((x_flat.size, num_columns), dtype=np.float64)
    delta_map_values = np.empty((x_flat.size, num_columns), dtype=np.float64)
    # newest column: residual - prev_residuals[0]
    delta_residuals[:, 0] = residual - prev_residuals[0]
    delta_map_values[:, 0] = t_flat - prev_map_values[0]
    for column_index in range(1, num_columns):
        delta_residuals[:, column_index] = (
            prev_residuals[column_index - 1] - prev_residuals[column_index]
        )
        delta_map_values[:, column_index] = (
            prev_map_values[column_index - 1] - prev_map_values[column_index]
        )

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


def safeguarded_anderson(
    *,
    initial_iterate: F64Array,
    fixed_point_map: Callable[[F64Array], F64Array],
    objective: Callable[[F64Array], float],
    memory_depth: int = 5,
    max_iterations: int = 200,
    tolerance: float = 1e-6,
    regularization: float = 1e-10,
    safeguard_slack: float = 0.0,
    relative_safeguard_slack: float = 0.0,
    nonmonotone_window: int = 1,
    damping_fractions: Sequence[float] = (1.0, 0.5, 0.25),
) -> tuple[F64Array, list[float], bool]:
    """Run safeguarded Anderson(m).

    Returns ``(best_iterate, objective_history, converged)``. The history
    contains the objective value at each accepted iterate (including the
    initial one).

    The safeguard is non-monotone (Grippo-Lampariello-Lucidi style): a
    candidate is accepted if its objective is at least
    ``max(last_k_objectives) - safeguard_slack - relative_safeguard_slack *
    |reference|``. This admits short-term objective dips that Anderson
    typically produces while still rejecting genuine divergence. Setting
    ``nonmonotone_window=1`` and ``relative_safeguard_slack=0`` recovers the
    strict monotone safeguard.
    """
    original_shape = np.asarray(initial_iterate).shape
    current = np.asarray(initial_iterate, dtype=np.float64).ravel().copy()

    def _call_map(vector: F64Array) -> F64Array:
        result = fixed_point_map(vector.reshape(original_shape))
        return np.asarray(result, dtype=np.float64).ravel()

    def _call_objective(vector: F64Array) -> float:
        return float(objective(vector.reshape(original_shape)))

    state = AndersonState(memory_depth=memory_depth)
    history: list[float] = [_call_objective(current)]
    converged = False

    for _ in range(max_iterations):
        mapped = _call_map(current)
        residual_norm = float(np.linalg.norm(mapped - current))
        if residual_norm < tolerance * max(1.0, float(np.linalg.norm(current))):
            current = mapped
            history.append(_call_objective(current))
            converged = True
            break

        plain_objective = _call_objective(mapped)
        accelerated = anderson_step(
            state,
            x_current=current,
            map_value=mapped,
            regularization=regularization,
        )

        accepted = mapped
        accepted_objective = plain_objective
        # Non-monotone reference: best objective over the last window.
        # Anderson candidates often dip transiently below the plain step's
        # objective before recovering super-linearly; a strict monotone
        # safeguard rejects every such candidate. Using max-over-window
        # plus a relative slack admits useful Anderson steps while still
        # catching genuine divergence (which produces unbounded drops).
        window = max(1, int(nonmonotone_window))
        recent_objectives = history[-window:] if history else []
        if recent_objectives:
            reference_objective = max(recent_objectives)
        else:
            reference_objective = plain_objective
        # Plain step is always at least as good as the worst-case fallback.
        reference_objective = max(reference_objective, plain_objective)
        relative_tolerance = float(relative_safeguard_slack) * abs(reference_objective)
        acceptance_threshold = reference_objective - float(safeguard_slack) - relative_tolerance
        for damping in damping_fractions:
            candidate = damping * accelerated + (1.0 - damping) * mapped
            if not np.all(np.isfinite(candidate)):
                continue
            candidate_objective = _call_objective(candidate)
            if candidate_objective >= acceptance_threshold:
                accepted = candidate
                accepted_objective = candidate_objective
                break

        current = accepted
        history.append(accepted_objective)

    return current.reshape(original_shape), history, converged
