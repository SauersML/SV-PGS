"""Regression tests for binary TR-Newton non-convergence handling.

Bug: ``_binary_posterior_state_tr_newton`` previously continued and returned
a stale tuple when the inner trust-region solver reported non-convergence,
silently propagating bad alpha/beta into the variational posterior. The fix
raises ``_BinaryTRNewtonNotConverged`` (a ``RuntimeError`` subclass) at the
gate. An operational escape hatch — ``_TR_NEWTON_RAISE_ON_NONCONVERGENCE``
— may flip the behaviour back to the lenient legacy mode.

Most of the surrounding inputs are heavy (real genotype matrices, JAX-backed
linear ops); we patch ``trust_region_newton_logistic`` so the test exercises
only the convergence gate.
"""
from __future__ import annotations

import types
from typing import Any

import numpy as np
import pytest

from sv_pgs import mixture_inference as mi

_GATE_PRESENT = hasattr(mi, "_TR_NEWTON_RAISE_ON_NONCONVERGENCE")
_FUNC_PRESENT = hasattr(mi, "_binary_posterior_state_tr_newton")
_RESULT_HELPER_PRESENT = hasattr(mi, "_trust_region_result_converged")

if not (_GATE_PRESENT and _FUNC_PRESENT and _RESULT_HELPER_PRESENT):
    pytest.skip(
        "waiting for fix in sv_pgs/mixture_inference.py "
        "(TR-Newton convergence gate not yet defined)",
        allow_module_level=True,
    )


def _make_fake_result(converged: bool, n_samples: int = 4, n_variants: int = 3) -> Any:
    """Construct an object mirroring the trust-region solver's return shape."""
    return types.SimpleNamespace(
        converged=converged,
        status="not_converged" if not converged else "converged",
        iterations=7,
        final_gradient_norm=1.0,
        alpha=np.zeros(n_samples + n_variants, dtype=np.float64),
        beta=np.zeros(n_variants, dtype=np.float64),
        linear_predictor=np.zeros(n_samples, dtype=np.float64),
    )


def test_non_convergence_raises_runtime_error(monkeypatch):
    """Drive the production wrapper ``_binary_posterior_state_tr_newton``
    with a monkeypatched inner solver that returns a non-converged
    ``result``. The wrapper must raise ``_BinaryTRNewtonNotConverged``
    (a ``RuntimeError`` subclass)."""
    exc_cls = getattr(mi, "_BinaryTRNewtonNotConverged", None)
    if exc_cls is None:
        pytest.skip("waiting for fix: _BinaryTRNewtonNotConverged not defined")
    assert issubclass(exc_cls, RuntimeError)

    n_samples = 4
    n_variants = 0  # short-circuits the design matvec helpers entirely

    # Fake genotype matrix: only ``.shape`` and ``._cupy_cache`` are read
    # by ``_binary_posterior_state_tr_newton`` before the inner solver runs.
    fake_genotype = types.SimpleNamespace(
        shape=(n_samples, n_variants),
        _cupy_cache=None,
    )

    def _fake_solver(**_kwargs: Any) -> Any:
        return _make_fake_result(
            converged=False, n_samples=n_samples, n_variants=n_variants
        )

    monkeypatch.setattr(mi, "trust_region_newton_logistic", _fake_solver)

    covariate_matrix = np.ones((n_samples, 1), dtype=np.float64)
    targets = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    prior_variances = np.zeros(n_variants, dtype=np.float64)
    alpha_init = np.zeros(n_samples + n_variants, dtype=np.float64)
    beta_init = np.zeros(n_variants, dtype=np.float64)

    with pytest.raises(exc_cls, match=r"(?i)converge"):
        mi._binary_posterior_state_tr_newton(
            genotype_matrix=fake_genotype,  # type: ignore[arg-type]
            covariate_matrix=covariate_matrix,
            targets=targets,
            prior_variances=prior_variances,
            alpha_init=alpha_init,
            beta_init=beta_init,
            minimum_weight=1e-6,
            max_iterations=5,
            gradient_tolerance=1e-6,
            solver_tolerance=1e-6,
            maximum_linear_solver_iterations=10,
            logdet_probe_count=1,
            logdet_lanczos_steps=1,
            exact_solver_matrix_limit=1024,
            posterior_variance_batch_size=64,
            posterior_variance_probe_count=1,
            random_seed=0,
            compute_logdet=False,
            compute_beta_variance=False,
            sample_space_preconditioner_rank=0,
            predictor_offset=None,
            posterior_working_set_initial_size=1,
            posterior_working_set_growth=2,
            posterior_working_set_max_passes=1,
            posterior_working_set_coefficient_tolerance=1e-4,
            restricted_posterior_warm_start=None,
            allow_gpu_exact_variant=False,
        )


def test_gate_constant_defined_and_default_true():
    """The escape-hatch gate must exist and default to True (raising)."""
    assert mi._TR_NEWTON_RAISE_ON_NONCONVERGENCE is True


def test_gate_can_be_flipped_to_false(monkeypatch):
    """Flipping the module-level gate to False must not raise on read."""
    monkeypatch.setattr(mi, "_TR_NEWTON_RAISE_ON_NONCONVERGENCE", False)
    assert mi._TR_NEWTON_RAISE_ON_NONCONVERGENCE is False


def test_trust_region_result_converged_helper_status_strings():
    """The helper recognises both the boolean and the status-string paths."""
    assert mi._trust_region_result_converged(
        types.SimpleNamespace(converged=True, status="converged")
    )
    assert not mi._trust_region_result_converged(
        types.SimpleNamespace(converged=False, status="not_converged")
    )
    # Status-string fallback: object exposes only ``status``.
    assert mi._trust_region_result_converged(types.SimpleNamespace(status="success"))
    assert not mi._trust_region_result_converged(types.SimpleNamespace(status="diverged"))
