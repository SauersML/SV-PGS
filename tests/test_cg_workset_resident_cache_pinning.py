"""Pin behavior of the wave-5713d85 CG working-set resident GPU cache.

These adversarial regression tests fence the helper pair
``_try_install_cg_workset_resident_cache`` /
``_release_cg_workset_resident_cache`` and the install/try/finally pattern
inside ``_solve_sample_space_rhs_gpu`` against silent regressions.

The tests avoid real CuPy, BigQuery, network, GPU, or PLINK BED entirely:
the genotype matrix is a duck-typed stub exposing only the fields the
helpers touch (``_cupy_cache``, ``raw``, ``shape``,
``try_materialize_gpu_subset``, ``try_materialize_gpu``), and ``cupy`` is
either ``None`` or a minimal ``SimpleNamespace`` shim modeled on
``tests/test_gpu_materialization_budget_bugfixes.py`` /
``tests/test_gpu_memory_hygiene.py``.
"""
from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

import sv_pgs.mixture_inference as mi


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubGenotypeMatrix:
    """Duck-typed stand-in for ``StandardizedGenotypeMatrix``.

    The helpers only read ``_cupy_cache``, ``raw``, ``shape``, and the two
    materialization methods; they never inspect anything else, so this is
    sufficient to exercise install/release paths.
    """

    def __init__(
        self,
        *,
        n_rows: int = 32,
        n_cols: int = 16,
        raw: Any = object(),
        cupy_cache: Any = None,
        materialize_result: bool = True,
        materialize_side_effect: BaseException | None = None,
        provide_subset_method: bool = True,
    ) -> None:
        self._cupy_cache: Any = cupy_cache
        self.raw = raw
        self.shape: tuple[int, int] = (int(n_rows), int(n_cols))
        self._materialize_result = materialize_result
        self._materialize_side_effect = materialize_side_effect
        self.materialize_calls = 0

        def _try_materialize_gpu() -> bool:
            self.materialize_calls += 1
            if self._materialize_side_effect is not None:
                raise self._materialize_side_effect
            if self._materialize_result:
                # Mimic real behavior: install a sentinel cache object.
                self._cupy_cache = object()
            return self._materialize_result

        self.try_materialize_gpu = _try_materialize_gpu

        if provide_subset_method:
            # The install helper only checks the attribute exists; it never
            # actually calls this. A noop returning None is enough.
            self.try_materialize_gpu_subset = lambda *a, **kw: None
        # When provide_subset_method is False, no attribute is set →
        # ``getattr(..., "try_materialize_gpu_subset", None)`` returns None.


class _FakeDevice:
    def __init__(self) -> None:
        self.sync_calls = 0

    def synchronize(self) -> None:
        self.sync_calls += 1


class _FakePool:
    def __init__(self) -> None:
        self.free_all_blocks_calls = 0

    def free_bytes(self) -> int:
        return 0

    def free_all_blocks(self) -> None:
        self.free_all_blocks_calls += 1


def _make_fake_cupy(*, free_bytes: int = 10 * 10**9, total_bytes: int = 16 * 10**9) -> Any:
    pool = _FakePool()
    device = _FakeDevice()

    fake_runtime = types.SimpleNamespace(memGetInfo=lambda: (free_bytes, total_bytes))
    fake_cuda = types.SimpleNamespace(runtime=fake_runtime, Device=lambda: device)

    fake_cupy = types.SimpleNamespace(
        cuda=fake_cuda,
        get_default_memory_pool=lambda: pool,
        get_default_pinned_memory_pool=lambda: pool,
    )
    # Expose the underlying recorders so tests can assert on call counts.
    fake_cupy._pool = pool  # type: ignore[attr-defined]
    fake_cupy._device = device  # type: ignore[attr-defined]
    return fake_cupy


# ---------------------------------------------------------------------------
# 1. cupy is None → install is a no-op, release is safe
# ---------------------------------------------------------------------------


def test_install_short_circuits_when_cupy_is_none() -> None:
    gm = _StubGenotypeMatrix()
    result = mi._try_install_cg_workset_resident_cache(gm, cupy=None)
    assert result is False
    assert gm._cupy_cache is None
    assert gm.materialize_calls == 0


def test_release_is_safe_when_cupy_is_none_and_no_cache() -> None:
    gm = _StubGenotypeMatrix()
    # Must not raise.
    mi._release_cg_workset_resident_cache(gm, cupy=None)
    assert gm._cupy_cache is None


# ---------------------------------------------------------------------------
# 2. Genotype matrix without ``try_materialize_gpu`` → short-circuit
# ---------------------------------------------------------------------------


def test_install_short_circuits_when_materialize_method_missing() -> None:
    # The install helper gates on the method it actually invokes,
    # ``try_materialize_gpu`` (not the never-called ``try_materialize_gpu_subset``);
    # when it is absent the helper must short-circuit without materializing.
    gm = _StubGenotypeMatrix()
    delattr(gm, "try_materialize_gpu")
    fake_cupy = _make_fake_cupy()
    result = mi._try_install_cg_workset_resident_cache(gm, cupy=fake_cupy)
    assert result is False
    assert gm._cupy_cache is None
    assert gm.materialize_calls == 0


# ---------------------------------------------------------------------------
# 3. Budget exceeded → install refuses
# ---------------------------------------------------------------------------


def test_install_refuses_when_materialize_declines(monkeypatch: pytest.MonkeyPatch) -> None:
    # try_materialize_gpu is the SINGLE gate now (the premature budget pre-check
    # was removed because it over-reserved fixed staging and rejected matrices
    # the adaptive-staging plan accepts). When the matrix doesn't fit, the plan
    # makes try_materialize_gpu return False (cheaply, no allocation) and the
    # install refuses.
    gm = _StubGenotypeMatrix(n_rows=1000, n_cols=1000, materialize_result=False)
    fake_cupy = _make_fake_cupy()

    result = mi._try_install_cg_workset_resident_cache(gm, cupy=fake_cupy)
    assert result is False
    assert gm._cupy_cache is None
    # It DID consult the real gate (vs the old pre-check that refused first).
    assert gm.materialize_calls == 1


def test_install_refuses_when_matrix_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    # A degenerate (0-column) working set can't be materialized; the install
    # bails before touching try_materialize_gpu.
    gm = _StubGenotypeMatrix(n_rows=100, n_cols=0)
    fake_cupy = _make_fake_cupy()
    result = mi._try_install_cg_workset_resident_cache(gm, cupy=fake_cupy)
    assert result is False
    assert gm.materialize_calls == 0


def test_install_does_not_consult_budget_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """The premature budget pre-check is gone — try_materialize_gpu (which runs
    the adaptive-staging plan + cupy-pool release) is the single source of truth.
    A broken budget helper must therefore NOT affect the install at all."""
    gm = _StubGenotypeMatrix()  # materialize_result=True
    fake_cupy = _make_fake_cupy()

    def _must_not_be_called(*a, **kw):
        raise AssertionError("budget pre-check must no longer be consulted")

    monkeypatch.setattr(mi, "_call_gpu_materialization_budget_bytes", _must_not_be_called)
    result = mi._try_install_cg_workset_resident_cache(gm, cupy=fake_cupy)
    assert result is True
    assert gm.materialize_calls == 1


# ---------------------------------------------------------------------------
# 4. OOM rollback → install returns False, does NOT propagate
# ---------------------------------------------------------------------------


def test_install_swallows_oom_from_try_materialize_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gm = _StubGenotypeMatrix(
        materialize_side_effect=MemoryError("simulated GPU OOM"),
    )
    fake_cupy = _make_fake_cupy()
    monkeypatch.setattr(
        mi,
        "_call_gpu_materialization_budget_bytes",
        lambda cupy, **kwargs: 10 * 10**9,  # generous; install should proceed
    )
    # Spy on _release_cupy_cached_memory so we can verify it gets called on
    # the OOM rollback path (best-effort pool drain).
    release_calls: list[Any] = []
    monkeypatch.setattr(
        mi,
        "_release_cupy_cached_memory",
        lambda cupy: release_calls.append(cupy),
    )

    # Must NOT propagate MemoryError.
    result = mi._try_install_cg_workset_resident_cache(gm, cupy=fake_cupy)
    assert result is False
    assert gm._cupy_cache is None
    assert gm.materialize_calls == 1
    # OOM rollback must drain the pool exactly once.
    assert release_calls == [fake_cupy]


def test_install_swallows_runtimeerror_from_try_materialize_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gm = _StubGenotypeMatrix(
        materialize_side_effect=RuntimeError("cudaErrorMemoryAllocation"),
    )
    fake_cupy = _make_fake_cupy()
    monkeypatch.setattr(
        mi,
        "_call_gpu_materialization_budget_bytes",
        lambda cupy, **kwargs: 10 * 10**9,
    )
    monkeypatch.setattr(mi, "_release_cupy_cached_memory", lambda cupy: None)

    result = mi._try_install_cg_workset_resident_cache(gm, cupy=fake_cupy)
    assert result is False


# ---------------------------------------------------------------------------
# Successful install (sanity)
# ---------------------------------------------------------------------------


def test_install_succeeds_when_budget_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    gm = _StubGenotypeMatrix(n_rows=32, n_cols=16)
    fake_cupy = _make_fake_cupy()
    monkeypatch.setattr(
        mi,
        "_call_gpu_materialization_budget_bytes",
        lambda cupy, **kwargs: 10 * 10**9,
    )
    result = mi._try_install_cg_workset_resident_cache(gm, cupy=fake_cupy)
    assert result is True
    assert gm._cupy_cache is not None
    assert gm.materialize_calls == 1


# ---------------------------------------------------------------------------
# 5. Cache release on exception (via install→solve→release try/finally)
# ---------------------------------------------------------------------------


def test_solve_releases_cache_when_solve_body_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gm = _StubGenotypeMatrix(n_rows=32, n_cols=16)
    fake_cupy = _make_fake_cupy()

    # Make ``import cupy as cp`` inside _solve_sample_space_rhs_gpu succeed.
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    # Force ``_try_import_cupy`` to return our fake (must be non-None for
    # the solve path to proceed past the guard).
    monkeypatch.setattr(mi, "_try_import_cupy", lambda: fake_cupy)

    monkeypatch.setattr(
        mi,
        "_call_gpu_materialization_budget_bytes",
        lambda cupy, **kwargs: 10 * 10**9,
    )
    monkeypatch.setattr(mi, "_release_cupy_cached_memory", lambda cupy: None)

    # Force the solve body to blow up so we can verify the finally block
    # still releases the cache.
    def _raise(**kwargs):
        raise RuntimeError("simulated solve failure")

    monkeypatch.setattr(
        mi,
        "_solve_sample_space_rhs_gpu_with_optional_workset_cache",
        _raise,
    )

    with pytest.raises(RuntimeError, match="simulated solve failure"):
        mi._solve_sample_space_rhs_gpu(
            genotype_matrix=gm,
            prior_variances=np.ones(16),
            diagonal_noise=np.ones(32),
            right_hand_side=np.zeros(32),
            initial_guess=None,
            tolerance=1e-6,
            max_iterations=10,
            preconditioner=lambda v: v,
            batch_size=8,
        )

    # The install ran (materialize_calls == 1) and the finally released it.
    assert gm.materialize_calls == 1
    assert gm._cupy_cache is None
    # Device sync must have been called on release.
    assert fake_cupy._device.sync_calls >= 1


# ---------------------------------------------------------------------------
# 6. Idempotent release
# ---------------------------------------------------------------------------


def test_release_is_idempotent_when_no_cache_installed() -> None:
    gm = _StubGenotypeMatrix(cupy_cache=None)
    fake_cupy = _make_fake_cupy()
    # Two back-to-back calls must not raise and must not toggle anything.
    mi._release_cg_workset_resident_cache(gm, cupy=fake_cupy)
    mi._release_cg_workset_resident_cache(gm, cupy=fake_cupy)
    assert gm._cupy_cache is None
    # Because ``_cupy_cache is None`` short-circuits release before
    # touching the device, no sync should happen.
    assert fake_cupy._device.sync_calls == 0


def test_release_is_idempotent_after_actual_release() -> None:
    sentinel = object()
    gm = _StubGenotypeMatrix(cupy_cache=sentinel)
    fake_cupy = _make_fake_cupy()
    mi._release_cg_workset_resident_cache(gm, cupy=fake_cupy)
    assert gm._cupy_cache is None
    first_sync = fake_cupy._device.sync_calls
    # Second call: must NOT raise AttributeError; must NOT re-sync (early
    # exit on ``_cupy_cache is None``).
    mi._release_cg_workset_resident_cache(gm, cupy=fake_cupy)
    assert gm._cupy_cache is None
    assert fake_cupy._device.sync_calls == first_sync


# ---------------------------------------------------------------------------
# 7. Already-installed re-entry → outer owner protected
# ---------------------------------------------------------------------------


def test_install_refuses_when_cache_already_installed() -> None:
    """If an outer caller already installed _cupy_cache, the inner install
    must NOT replace it and must return False so the outer owner keeps
    ownership of the release lifecycle."""
    outer_cache = object()
    gm = _StubGenotypeMatrix(cupy_cache=outer_cache)
    fake_cupy = _make_fake_cupy()
    result = mi._try_install_cg_workset_resident_cache(gm, cupy=fake_cupy)
    assert result is False
    # The outer caller's cache MUST be left intact.
    assert gm._cupy_cache is outer_cache
    assert gm.materialize_calls == 0


def test_solve_does_not_release_outer_owned_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``_cupy_cache`` is already populated on entry, the solve must
    not install a fresh cache and must NOT release the outer one in its
    finally block — only the outer owner releases."""
    outer_cache = object()
    gm = _StubGenotypeMatrix(cupy_cache=outer_cache)
    fake_cupy = _make_fake_cupy()

    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setattr(mi, "_try_import_cupy", lambda: fake_cupy)
    monkeypatch.setattr(
        mi,
        "_call_gpu_materialization_budget_bytes",
        lambda cupy, **kwargs: 10 * 10**9,
    )
    monkeypatch.setattr(mi, "_release_cupy_cached_memory", lambda cupy: None)

    sentinel_return = np.zeros((32, 1))

    def _fake_solve(**kwargs):
        # The solve sees the outer cache untouched.
        assert kwargs["genotype_matrix"]._cupy_cache is outer_cache
        return sentinel_return

    monkeypatch.setattr(
        mi,
        "_solve_sample_space_rhs_gpu_with_optional_workset_cache",
        _fake_solve,
    )

    result = mi._solve_sample_space_rhs_gpu(
        genotype_matrix=gm,
        prior_variances=np.ones(16),
        diagonal_noise=np.ones(32),
        right_hand_side=np.zeros(32),
        initial_guess=None,
        tolerance=1e-6,
        max_iterations=10,
        preconditioner=lambda v: v,
        batch_size=8,
    )
    assert result is sentinel_return
    # No materialization attempt (cache was already present).
    assert gm.materialize_calls == 0
    # Crucially, the outer cache is still in place after the solve returns.
    assert gm._cupy_cache is outer_cache


# ---------------------------------------------------------------------------
# 8. Concurrent (sequential) CG solves: no GPU memory leak across solves
# ---------------------------------------------------------------------------


def test_back_to_back_solves_release_cache_between_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two sequential ``_solve_sample_space_rhs_gpu`` invocations must each
    install AND release their own cache; solve 2 must observe a clean
    (``_cupy_cache is None``) state on entry."""
    gm = _StubGenotypeMatrix(n_rows=32, n_cols=16)
    fake_cupy = _make_fake_cupy()

    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setattr(mi, "_try_import_cupy", lambda: fake_cupy)
    monkeypatch.setattr(
        mi,
        "_call_gpu_materialization_budget_bytes",
        lambda cupy, **kwargs: 10 * 10**9,
    )
    monkeypatch.setattr(mi, "_release_cupy_cached_memory", lambda cupy: None)

    entry_cache_states: list[Any] = []
    body_cache_states: list[Any] = []

    def _fake_solve(**kwargs):
        # Confirm install ran: cache is non-None inside the body.
        body_cache_states.append(kwargs["genotype_matrix"]._cupy_cache)
        return np.zeros((32, 1))

    monkeypatch.setattr(
        mi,
        "_solve_sample_space_rhs_gpu_with_optional_workset_cache",
        _fake_solve,
    )

    for _ in range(2):
        # Snapshot the cache state at entry to each solve.
        entry_cache_states.append(gm._cupy_cache)
        mi._solve_sample_space_rhs_gpu(
            genotype_matrix=gm,
            prior_variances=np.ones(16),
            diagonal_noise=np.ones(32),
            right_hand_side=np.zeros(32),
            initial_guess=None,
            tolerance=1e-6,
            max_iterations=10,
            preconditioner=lambda v: v,
            batch_size=8,
        )

    # Both solves entered with a clean cache and saw an installed cache
    # inside the body. The cache was released between solves — solve 2's
    # entry state is None, not "still installed from solve 1".
    assert entry_cache_states == [None, None]
    assert all(state is not None for state in body_cache_states)
    assert gm._cupy_cache is None
    # Each solve materialized exactly once → no leak across solves.
    assert gm.materialize_calls == 2
    # Each solve synchronized the device on release.
    assert fake_cupy._device.sync_calls >= 2


# ---------------------------------------------------------------------------
# Misc: ``raw is None`` → install short-circuits (caller has no streaming
# source either, but the helper guards against it).
# ---------------------------------------------------------------------------


def test_install_short_circuits_when_raw_is_none() -> None:
    gm = _StubGenotypeMatrix(raw=None)
    fake_cupy = _make_fake_cupy()
    result = mi._try_install_cg_workset_resident_cache(gm, cupy=fake_cupy)
    assert result is False
    assert gm.materialize_calls == 0


# ---------------------------------------------------------------------------
# Degenerate shapes guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_rows,n_cols",
    [(0, 16), (16, 0), (0, 0)],
)
def test_install_short_circuits_on_degenerate_shape(
    n_rows: int, n_cols: int
) -> None:
    gm = _StubGenotypeMatrix(n_rows=n_rows, n_cols=n_cols)
    fake_cupy = _make_fake_cupy()
    result = mi._try_install_cg_workset_resident_cache(gm, cupy=fake_cupy)
    assert result is False
    assert gm.materialize_calls == 0
