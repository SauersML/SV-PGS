"""Tests for the Dembo-Eisenstat-Steihaug forcing-sequence helpers."""

from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.forcing_sequence import forcing_tolerance, relaxed_iteration_cap


def test_forcing_tolerance_monotone_in_gradient_norm() -> None:
    grads = np.geomspace(1e-10, 0.9, num=50)
    etas = np.array(
        [forcing_tolerance(gradient_norm=float(g)) for g in grads]
    )
    diffs = np.diff(etas)
    assert np.all(diffs >= -1e-15), "forcing tolerance must be non-decreasing in ‖g‖"


def test_forcing_tolerance_clipping() -> None:
    eta_min = 1e-10
    eta_max = 0.3
    for g in [0.0, 1e-30, 1e-15, 1e-6, 1e-2, 0.5, 1.0, 1e3, 1e9]:
        eta = forcing_tolerance(
            gradient_norm=g,
            eta_max=eta_max,
            eta_min=eta_min,
            exponent=0.5,
            safeguard_against_oversolve=False,
        )
        assert eta_min - 1e-18 <= eta <= eta_max + 1e-18

    # With the oversolve safeguard, the floor is at least sqrt(eps).
    sqrt_eps = float(np.sqrt(np.finfo(np.float64).eps))
    eta_zero = forcing_tolerance(
        gradient_norm=0.0,
        eta_max=0.5,
        eta_min=1e-20,
        safeguard_against_oversolve=True,
    )
    assert eta_zero >= sqrt_eps - 1e-18


def test_forcing_tolerance_asymptotic_order_half() -> None:
    # Pick gradient norms whose sqrt lies strictly inside (eta_min, eta_max)
    # so neither clip is active.
    for g in [1e-8, 1e-6, 1e-4, 1e-2, 0.1]:
        eta = forcing_tolerance(
            gradient_norm=g,
            eta_max=0.5,
            eta_min=1e-20,
            exponent=0.5,
            safeguard_against_oversolve=False,
        )
        assert eta == pytest.approx(np.sqrt(g), rel=1e-12, abs=1e-15)


def test_relaxed_iteration_cap_monotone_in_tolerance() -> None:
    tols = np.geomspace(0.49, 1e-14, num=80)
    caps = np.array(
        [relaxed_iteration_cap(forcing_tolerance_value=float(t)) for t in tols]
    )
    # As tolerance shrinks (index increases), cap must not decrease.
    diffs = np.diff(caps)
    assert np.all(diffs >= 0), "smaller forcing tolerance must yield >= iteration cap"
    assert caps[0] >= 32 and caps[-1] <= 256


def test_relaxed_iteration_cap_bounds() -> None:
    assert relaxed_iteration_cap(forcing_tolerance_value=0.4999) >= 32
    assert relaxed_iteration_cap(forcing_tolerance_value=1e-30) == 256
    assert relaxed_iteration_cap(
        forcing_tolerance_value=0.5, base_cap=128, minimum_cap=16
    ) >= 16


def test_des_superlinear_demonstration() -> None:
    """End-to-end inexact-Newton demo with DES forcing sequence.

    We minimise ``f(x) = 0.5 x^T A x`` (so the gradient is ``g_k = A x_k``
    and the Hessian is the constant SPD matrix ``A``) using an inexact
    Newton step computed by k iterations of CG, where k comes from
    ``relaxed_iteration_cap(forcing_tolerance(‖g_k‖))``.

    With a well-conditioned ``A`` and the DES sqrt-forcing sequence, the
    outer iterates should converge at q-order at least ~1.4 over a short
    sequence of iterations.
    """
    rng = np.random.default_rng(0)
    n = 60
    # Build a well-conditioned SPD matrix A (eigenvalues in [1, 10]).
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigvals = np.linspace(1.0, 10.0, n)
    A = (Q * eigvals) @ Q.T
    A = 0.5 * (A + A.T)

    def cg(A_mat: np.ndarray, b: np.ndarray, *, max_iter: int, tol: float) -> np.ndarray:
        x = np.zeros_like(b)
        r = b - A_mat @ x
        p = r.copy()
        rs_old = float(r @ r)
        b_norm = float(np.linalg.norm(b))
        if b_norm == 0.0:
            return x
        for _ in range(max_iter):
            Ap = A_mat @ p
            alpha = rs_old / float(p @ Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = float(r @ r)
            if np.sqrt(rs_new) <= tol * b_norm:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    # Start close enough that the DES sqrt-forcing is interior (η = sqrt(‖g‖)
    # rather than clipped at eta_max), so we observe the superlinear regime
    # rather than the initial q-linear phase.
    x = 1e-2 * rng.standard_normal(n)
    grad_norms: list[float] = []
    grad_norms.append(float(np.linalg.norm(A @ x)))

    for _ in range(5):
        g = A @ x
        gn = float(np.linalg.norm(g))
        eta = forcing_tolerance(
            gradient_norm=gn,
            eta_max=0.5,
            eta_min=1e-14,
            exponent=0.5,
            safeguard_against_oversolve=False,
        )
        k = relaxed_iteration_cap(
            forcing_tolerance_value=eta,
            base_cap=n,  # full-rank cap; CG exits early via tol anyway.
            minimum_cap=2,
            log_tol_floor=-12.0,
        )
        s = cg(A, -g, max_iter=k, tol=eta)
        x = x + s
        grad_norms.append(float(np.linalg.norm(A @ x)))

    grad_norms_arr = np.asarray(grad_norms, dtype=np.float64)
    # Sanity: must actually be converging.
    assert grad_norms_arr[-1] < grad_norms_arr[0] * 1e-6

    # Estimate the empirical q-order from the last few consecutive ratios:
    #   ‖g_{k+1}‖ ≈ C ‖g_k‖^p
    # log r_{k+1} ≈ p log r_k + log C.
    logs = np.log(grad_norms_arr + 1e-300)
    # Use the last 4 transitions where convergence dominates.
    x_log = logs[-5:-1]
    y_log = logs[-4:]
    # Linear regression slope of y on x.
    x_centered = x_log - x_log.mean()
    y_centered = y_log - y_log.mean()
    slope = float((x_centered @ y_centered) / (x_centered @ x_centered))

    assert slope >= 1.4, f"empirical q-order {slope:.3f} below 1.4"
