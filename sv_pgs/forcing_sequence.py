"""Dembo-Eisenstat-Steihaug forcing-sequence helpers for inexact Newton / CG.

Inexact Newton solves H_k s_k = -g_k approximately, with residual
``‖H_k s_k + g_k‖ ≤ η_k ‖g_k‖``. The Dembo-Eisenstat-Steihaug theorem
relates the forcing sequence ``{η_k}`` to the outer convergence order:

* Constant ``η_k = η < 1`` gives q-linear convergence at rate ``η``.
* ``η_k → 0`` gives q-superlinear convergence.
* ``η_k = O(‖g_k‖^q)`` with ``q ∈ (0, 1]`` gives q-order ``1 + q``
  (quadratic when ``q = 1``).

The standard "safe" choice used here is
``η_k = min(η_max, ‖g_k‖^q)`` with ``η_max = 0.5`` and ``q = 0.5``.
"""

from __future__ import annotations

import numpy as np


def forcing_tolerance(
    *,
    gradient_norm: float,
    eta_max: float = 0.5,
    eta_min: float = 1e-12,
    exponent: float = 0.5,
    safeguard_against_oversolve: bool = True,
) -> float:
    """Dembo-Eisenstat-Steihaug forcing-sequence tolerance.

    Returns ``η_k = clip(‖g_k‖^exponent, eta_min, eta_max)``.

    Parameters
    ----------
    gradient_norm:
        Current outer gradient norm ``‖g_k‖`` (non-negative).
    eta_max:
        Upper clip; must lie strictly in ``(0, 1)`` for DES convergence.
    eta_min:
        Lower clip to keep the inner system from being solved to
        machine precision when the outer iteration is far from the
        minimiser. Must satisfy ``0 < eta_min ≤ eta_max``.
    exponent:
        DES exponent ``q ∈ (0, 1]`` controlling outer order ``1 + q``.
    safeguard_against_oversolve:
        If ``True``, also clip below by ``sqrt(eps_float64)`` so that
        the inner solver does not chase below the noise floor.

    Returns
    -------
    float
        The forcing tolerance ``η_k`` as a float64 scalar.
    """
    g = float(gradient_norm)
    if not np.isfinite(g):
        raise ValueError("gradient_norm must be finite")
    if g < 0.0:
        raise ValueError("gradient_norm must be non-negative")
    eta_max_f = float(eta_max)
    eta_min_f = float(eta_min)
    q = float(exponent)
    if not (0.0 < eta_max_f < 1.0):
        raise ValueError("eta_max must lie in (0, 1)")
    if not (0.0 < eta_min_f <= eta_max_f):
        raise ValueError("eta_min must satisfy 0 < eta_min <= eta_max")
    if not (0.0 < q <= 1.0):
        raise ValueError("exponent must lie in (0, 1]")

    # Compute ‖g‖^q in float64. For g == 0 this is 0 and gets clipped to eta_min.
    raw = np.float64(g) ** np.float64(q)
    floor = eta_min_f
    if safeguard_against_oversolve:
        floor = max(floor, float(np.sqrt(np.finfo(np.float64).eps)))
    return float(np.clip(raw, floor, eta_max_f))


def relaxed_iteration_cap(
    *,
    forcing_tolerance_value: float,
    base_cap: int = 256,
    minimum_cap: int = 32,
    log_tol_floor: float = -12.0,
) -> int:
    """CG iteration budget consistent with the forcing tolerance.

    Heuristic: solving a moderately conditioned, preconditioned linear
    system to relative residual ``η`` requires roughly
    ``log(1/η) / log(1 / (1 - 2/sqrt(κ)))`` CG iterations. As a smooth
    proxy we set

        cap ≈ minimum_cap
            + (base_cap - minimum_cap) * (log10(1/η) / |log_tol_floor|)

    clipped into ``[minimum_cap, base_cap]``.

    Parameters
    ----------
    forcing_tolerance_value:
        Target relative residual ``η_k`` (must be in ``(0, 1)``).
    base_cap:
        Maximum iteration budget (returned when ``η`` is at or below
        ``10**log_tol_floor``).
    minimum_cap:
        Lower bound on the iteration budget.
    log_tol_floor:
        Negative log10 anchor; ``η = 10**log_tol_floor`` saturates the
        cap at ``base_cap``.

    Returns
    -------
    int
        Iteration cap in ``[minimum_cap, base_cap]``.
    """
    eta = float(forcing_tolerance_value)
    if not np.isfinite(eta) or eta <= 0.0 or eta >= 1.0:
        raise ValueError("forcing_tolerance_value must lie in (0, 1)")
    base = int(base_cap)
    minimum = int(minimum_cap)
    if minimum < 1:
        raise ValueError("minimum_cap must be >= 1")
    if base < minimum:
        raise ValueError("base_cap must be >= minimum_cap")
    floor = float(log_tol_floor)
    if not (floor < 0.0):
        raise ValueError("log_tol_floor must be negative")

    log_inv_eta = float(np.log10(1.0 / eta))  # > 0 since eta < 1
    fraction = log_inv_eta / abs(floor)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    raw = minimum + (base - minimum) * fraction
    cap = int(np.floor(raw + 1e-12))
    if cap < minimum:
        cap = minimum
    if cap > base:
        cap = base
    return cap
