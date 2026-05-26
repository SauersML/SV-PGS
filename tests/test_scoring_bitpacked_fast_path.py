"""Tests for the scoring (`decision_components`) bitpacked fast-path.

The fast path in :func:`sv_pgs.model._bitpacked_scoring_fast_path` should:
  * Return ``None`` when the raw genotype matrix is NOT a
    ``BitpackedDeviceMatrix`` (so the caller falls back to legacy
    ``iter_column_batches`` streaming).
  * Be wired into :meth:`BayesianPGS.decision_components` ahead of the
    legacy path.

These tests deliberately exercise the CPU/no-CuPy branch — they assert the
fallback contract and the wiring. A GPU-only test (gated on CuPy + bitpacked
loader availability) exercises the single-kernel device path.
"""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs import model as _model


def test_fast_path_returns_none_for_dense_matrix():
    """Dense (non-bitpacked) inputs must return None so the caller falls back."""
    from sv_pgs.genotype import as_raw_genotype_matrix

    raw_values = np.array(
        [
            [0.0, 1.0, 2.0],
            [2.0, 0.0, 1.0],
            [1.0, 2.0, 0.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    raw = as_raw_genotype_matrix(raw_values)
    variant_indices = np.array([0, 2], dtype=np.int32)
    means = np.array([0.75, 1.25], dtype=np.float32)
    scales = np.array([0.83, 0.83], dtype=np.float32)
    coefficients = np.array([0.5, -0.25], dtype=np.float32)

    result = _model._bitpacked_scoring_fast_path(
        raw_genotypes=raw,
        variant_indices=variant_indices,
        means=means,
        scales=scales,
        coefficients=coefficients,
    )
    assert result is None


def test_decision_components_invokes_fast_path_helper(monkeypatch):
    """``decision_components`` must call ``_bitpacked_scoring_fast_path``
    ahead of the legacy streaming helper. We monkeypatch both to record
    invocation order on a synthetic ``BayesianPGS`` whose ``state`` is a
    minimal mock — bypassing the FittedState dataclass schema."""
    from types import SimpleNamespace
    from sv_pgs.config import ModelConfig, TraitType
    from sv_pgs.model import BayesianPGS

    n_samples = 6
    nz_indices = np.array([0, 2], dtype=np.int32)
    nz_coef = np.array([0.5, -0.25], dtype=np.float32)
    nz_means = np.array([1.0, 1.0], dtype=np.float32)
    nz_scales = np.array([1.0, 1.0], dtype=np.float32)
    alpha = np.array([0.0], dtype=np.float32)

    state = SimpleNamespace(
        nonzero_coefficient_indices=nz_indices,
        nonzero_coefficients=nz_coef,
        nonzero_means=nz_means,
        nonzero_scales=nz_scales,
        full_coefficients=np.zeros(4, dtype=np.float32),
        fit_result=SimpleNamespace(alpha=alpha),
    )
    model_instance = BayesianPGS(config=ModelConfig(trait_type=TraitType.QUANTITATIVE))
    model_instance.state = state

    call_log: list[str] = []
    sentinel = np.full(n_samples, 7.5, dtype=np.float32)

    def _fake_fast_path(**kwargs):
        call_log.append("fast")
        assert kwargs["variant_indices"].shape[0] == 2
        return sentinel

    def _spy_legacy(*args, **kwargs):
        call_log.append("legacy")
        return np.zeros(n_samples, dtype=np.float32)

    monkeypatch.setattr(_model, "_bitpacked_scoring_fast_path", _fake_fast_path)
    monkeypatch.setattr(_model, "_raw_standardized_subset_matvec", _spy_legacy)
    raw = np.zeros((n_samples, 4), dtype=np.float32)
    covariates = np.zeros((n_samples, 0), dtype=np.float32)
    genetic, _ = model_instance.decision_components(raw, covariates)
    assert call_log == ["fast"]  # legacy NOT invoked when fast path returns a value
    np.testing.assert_array_equal(genetic, sentinel)


def test_decision_components_falls_back_when_fast_path_returns_none(monkeypatch):
    """When the fast path returns ``None``, the legacy streaming helper is used."""
    from types import SimpleNamespace
    from sv_pgs.config import ModelConfig, TraitType
    from sv_pgs.model import BayesianPGS

    n_samples = 6
    state = SimpleNamespace(
        nonzero_coefficient_indices=np.array([0, 2], dtype=np.int32),
        nonzero_coefficients=np.array([0.5, -0.25], dtype=np.float32),
        nonzero_means=np.array([1.0, 1.0], dtype=np.float32),
        nonzero_scales=np.array([1.0, 1.0], dtype=np.float32),
        full_coefficients=np.zeros(4, dtype=np.float32),
        fit_result=SimpleNamespace(alpha=np.array([0.0], dtype=np.float32)),
    )
    model_instance = BayesianPGS(config=ModelConfig(trait_type=TraitType.QUANTITATIVE))
    model_instance.state = state

    legacy_calls: list[int] = []

    def _fake_fast_path(**kwargs):
        return None

    def _spy_legacy(*args, **kwargs):
        legacy_calls.append(int(kwargs["variant_indices"].shape[0]))
        return np.zeros(n_samples, dtype=np.float32)

    monkeypatch.setattr(_model, "_bitpacked_scoring_fast_path", _fake_fast_path)
    monkeypatch.setattr(_model, "_raw_standardized_subset_matvec", _spy_legacy)

    raw = np.zeros((n_samples, 4), dtype=np.float32)
    covariates = np.zeros((n_samples, 0), dtype=np.float32)
    genetic, _ = model_instance.decision_components(raw, covariates)
    assert legacy_calls == [2]
    assert genetic.shape == (n_samples,)


def test_signal_handler_includes_sigtstp():
    """``_install_graceful_shutdown_handlers`` must wire SIGTSTP alongside
    SIGTERM/SIGHUP, so AoU workbench Ctrl-Z + parent-exit unwinds gracefully
    instead of dying with rc=148 mid-iteration."""
    import signal
    from pathlib import Path

    if not hasattr(signal, "SIGTSTP"):
        pytest.skip("SIGTSTP not available on this platform")

    # Read the file directly: importing sv_pgs.cli pulls in aou_runner +
    # pandas which is an optional dependency. The wiring is a literal tuple
    # in the source, so a text-level check is the right granularity here.
    cli_path = Path(__file__).resolve().parent.parent / "sv_pgs" / "cli.py"
    source = cli_path.read_text(encoding="utf-8")
    assert '"SIGTSTP"' in source, (
        "Expected SIGTSTP in the signal-handler wiring tuple — without it, "
        "Ctrl-Z + parent-exit on the AoU workbench bypasses atexit/finally."
    )
