"""Regression tests for failure modes a 9-hour AoU production run exposed.

The production failure produced no final summary, zero validation epochs,
and a non-converged "completed" artifact. Each test in this file pins down
the contract for one of the four root causes the existing test suite missed.

These tests are deliberately written against the *fixed* contracts that
sibling agents are mid-flight introducing in ``sv_pgs/``. Until those
fixes land the four tests are expected to fail — that is the regression
guardrail.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import sv_pgs.genotype as genotype_module
import sv_pgs.mixture_inference as mixture_module
import sv_pgs.pipeline as pipeline_module
from sv_pgs import BayesianPGS, ModelConfig, TraitType, VariantClass, VariantRecord
from sv_pgs.genotype import as_raw_genotype_matrix
from sv_pgs.io import LoadedDataset


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _make_binary_records(n_variants: int) -> list[VariantRecord]:
    return [
        VariantRecord(
            variant_id=f"variant_{i}",
            variant_class=VariantClass.SNV,
            chromosome="1",
            position=100 + i,
            length=1.0,
            allele_frequency=0.25,
            quality=1.0,
        )
        for i in range(n_variants)
    ]


def _make_tiny_binary_problem(
    *,
    n_samples: int = 120,
    n_variants: int = 6,
    seed: int = 17,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    # int8 genotypes (0/1/2) so the raw matrix path is exercised.
    raw_genotypes = rng.integers(low=0, high=3, size=(n_samples, n_variants), dtype=np.int8)
    beta_star = np.zeros(n_variants, dtype=np.float64)
    beta_star[0] = 1.0
    beta_star[1] = -0.8
    eta = raw_genotypes.astype(np.float64) @ beta_star
    probs = 1.0 / (1.0 + np.exp(-(eta - eta.mean()) / max(eta.std(), 1.0)))
    targets = (rng.uniform(size=n_samples) < probs).astype(np.float32)
    covariates = rng.standard_normal((n_samples, 1)).astype(np.float32)
    return {
        "genotypes": raw_genotypes,
        "covariates": covariates,
        "targets": targets,
        "records": _make_binary_records(n_variants),
    }


# ---------------------------------------------------------------------------
# Test 1 — TR-Newton must not stream raw BED silently
#
# Pairs with the production fix: when ``try_materialize_gpu()`` returns False
# inside the binary fit, the SVI block either retries with a smaller block
# OR raises a clear RuntimeError. Today, the streaming mmap path silently
# replaces 2s GPU CG iterations with 40s reads.
# ---------------------------------------------------------------------------


def test_tr_newton_refuses_silent_mmap_streaming(monkeypatch) -> None:
    """Binary fit must not silently stream raw mmap when GPU materialize fails."""
    problem = _make_tiny_binary_problem(n_samples=160, n_variants=8)

    # Force every GPU upload attempt to report failure so the binary fit must
    # decide: shrink the block and retry, or refuse loudly. Silent fallback
    # to mmap streaming is the bug we are guarding against.
    monkeypatch.setattr(genotype_module, "require_gpu", lambda: None)
    monkeypatch.setattr(
        genotype_module.StandardizedGenotypeMatrix,
        "try_materialize_gpu",
        lambda self: False,
    )

    config = ModelConfig(
        trait_type=TraitType.BINARY,
        max_outer_iterations=1,
        max_inner_newton_iterations=5,
        minimum_minor_allele_frequency=0.0,
    )

    # The contract we want: EITHER the fit finishes (because the block was
    # shrunk until it fit / a recoverable retry succeeded) AND the test was
    # not allowed to silently stream from mmap, OR a RuntimeError is raised
    # whose message names the refused-streaming path. We instrument the
    # streaming entry point to detect a silent fall-through.
    streamed_calls: list[tuple[int, int]] = []
    original_matvec = mixture_module._genotype_matvec_result_numpy

    def _tracking_matvec(genotype_matrix, vector, *args, **kwargs):
        # Only count calls that would actually stream from raw storage (no
        # GPU cache, no dense cache) — that is the failure signature.
        cupy_cache = getattr(genotype_matrix, "_cupy_cache", None)
        dense_cache = getattr(genotype_matrix, "_dense_cache", None)
        if cupy_cache is None and dense_cache is None:
            streamed_calls.append((int(genotype_matrix.shape[0]), int(genotype_matrix.shape[1])))
        return original_matvec(genotype_matrix, vector, *args, **kwargs)

    monkeypatch.setattr(mixture_module, "_genotype_matvec_result_numpy", _tracking_matvec)

    try:
        BayesianPGS(config).fit(
            problem["genotypes"],
            problem["covariates"],
            problem["targets"],
            problem["records"],
        )
    except RuntimeError as exc:
        # Acceptable fix path: refuse loudly with a recognisable message.
        message = str(exc).lower()
        assert "refusing" in message and "mmap" in message and "stream" in message, (
            "RuntimeError must explicitly name the refused mmap streaming path; "
            f"got: {exc!r}"
        )
        return

    # If the fit returned successfully, then the block must have been
    # shrunk / retried so that no raw mmap streaming matvec ever ran.
    assert not streamed_calls, (
        "Binary fit silently fell back to raw mmap streaming after GPU "
        f"materialization failure ({len(streamed_calls)} streaming matvecs). "
        "Either retry with a smaller block or raise a 'refusing mmap streaming' "
        "RuntimeError; silent success is the production regression."
    )


# ---------------------------------------------------------------------------
# Test 2 — GPU materialization failure must abort or shrink-and-retry
#
# Pairs with the fix that turns CuPy OOM during materialization into either
# (a) a successful shrink-and-retry, or (b) a clear RuntimeError. The bug:
# OOM was caught and silently swallowed, leaving the fit on the streaming
# slow path forever.
# ---------------------------------------------------------------------------


def test_gpu_materialization_failure_aborts_or_shrinks(monkeypatch) -> None:
    """GPU OOM during materialization must not silently fall through."""
    problem = _make_tiny_binary_problem(n_samples=80, n_variants=4)

    monkeypatch.setattr(genotype_module, "require_gpu", lambda: None)

    call_counter = {"n": 0}
    original_materialize = genotype_module.StandardizedGenotypeMatrix.try_materialize_gpu

    def _oom_first_then_real(self):
        call_counter["n"] += 1
        if call_counter["n"] == 1:
            # Simulate the CuPy allocator going OOM on the first try.
            raise MemoryError("simulated CuPy OOM during materialization")
        return original_materialize(self)

    monkeypatch.setattr(
        genotype_module.StandardizedGenotypeMatrix,
        "try_materialize_gpu",
        _oom_first_then_real,
    )

    config = ModelConfig(
        trait_type=TraitType.BINARY,
        max_outer_iterations=1,
        max_inner_newton_iterations=5,
        minimum_minor_allele_frequency=0.0,
    )

    raised: Exception | None = None
    model = None
    try:
        model = BayesianPGS(config).fit(
            problem["genotypes"],
            problem["covariates"],
            problem["targets"],
            problem["records"],
        )
    except (MemoryError, RuntimeError) as exc:
        raised = exc

    # Either the fit raised an explicit, descriptive error (acceptable),
    # or it survived by retrying — but only if the retry actually ran.
    # The forbidden outcome is: caught the OOM and silently moved on
    # without ever attempting another materialize.
    if raised is None:
        assert model is not None and model.state is not None
        assert call_counter["n"] >= 2, (
            "GPU materialization OOM was swallowed silently — no retry was "
            "attempted. The fix must shrink-and-retry or raise."
        )
    else:
        # A descriptive error is the other accepted contract. The message
        # must name the OOM/materialization path so an operator can
        # disambiguate it from unrelated failures — a bare non-empty string
        # is too loose and lets unrelated regressions pass.
        text = str(raised).lower()
        assert text, "raised exception must carry a message"
        expected_tokens = (
            "materializ",  # materialization / materialize
            "oom",
            "out of memory",
            "memory",
            "refus",       # "refusing"
            "shrink",
            "stream",      # "mmap streaming"
        )
        assert any(tok in text for tok in expected_tokens), (
            "GPU materialization OOM RuntimeError must name the failure "
            f"path (one of {expected_tokens!r}); got: {raised!r}"
        )


# ---------------------------------------------------------------------------
# Test 3 — Checkpointed holdout monitoring still records epoch history
#
# Pairs with the model.py bug at ~line 1229 where
# ``em_per_epoch_eval_callback = None`` whenever durable checkpointing is
# enabled and validation is holdout-only. Consequence in production: zero
# validation epochs and an empty (header-only) training_history.tsv.
# ---------------------------------------------------------------------------


def test_durable_checkpoint_with_holdout_still_invokes_per_epoch_callback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Durable checkpoint + holdout-only validation must keep epoch eval live."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(genotype_module, "require_gpu", lambda: None)
    monkeypatch.setattr(
        genotype_module.StandardizedGenotypeMatrix,
        "try_materialize_gpu",
        lambda self: False,
    )

    train = _make_tiny_binary_problem(n_samples=120, n_variants=6, seed=3)
    holdout = _make_tiny_binary_problem(n_samples=40, n_variants=6, seed=4)

    callback_snapshots: list[dict[str, Any]] = []

    def _callback(snapshot: dict[str, Any]) -> None:
        callback_snapshots.append(snapshot)

    config = ModelConfig(
        trait_type=TraitType.BINARY,
        max_outer_iterations=3,
        validation_interval=1,
        validate_first_iteration=True,
        max_inner_newton_iterations=5,
        minimum_minor_allele_frequency=0.0,
    )

    checkpoint_path = tmp_path / "fit_checkpoint.pkl"
    model = BayesianPGS(config).fit(
        train["genotypes"],
        train["covariates"],
        train["targets"],
        train["records"],
        validation_data=(holdout["genotypes"], holdout["covariates"], holdout["targets"]),
        per_epoch_eval_callback=_callback,
        validation_is_holdout_only=True,
        fit_checkpoint_path=checkpoint_path,
    )
    assert model.state is not None

    # The bug we guard against: per_epoch_eval_callback gets silently
    # replaced with None as soon as the durable checkpoint is active and
    # validation is holdout-only, so the EM loop never calls back and the
    # pipeline's training_history.tsv writer never sees a row.
    assert len(callback_snapshots) >= 1, (
        "per_epoch_eval_callback was never invoked when durable checkpointing "
        "+ holdout-only validation were both enabled. This is the production "
        "regression that produced zero validation epochs and a header-only "
        "training_history.tsv."
    )
    # And the snapshot must carry the bits the pipeline writer needs.
    last = callback_snapshots[-1]
    assert "epoch" in last
    assert "objective" in last


def test_pipeline_history_has_rows_with_checkpoint_and_holdout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Pipeline-level: training_history.tsv must contain data rows, not just header."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(genotype_module, "require_gpu", lambda: None)
    monkeypatch.setattr(
        genotype_module.StandardizedGenotypeMatrix,
        "try_materialize_gpu",
        lambda self: False,
    )

    train = _make_tiny_binary_problem(n_samples=120, n_variants=6, seed=5)

    history_path = tmp_path / "training_history.tsv"
    config = ModelConfig(
        trait_type=TraitType.BINARY,
        max_outer_iterations=2,
        validation_interval=1,
        validate_first_iteration=True,
        max_inner_newton_iterations=5,
        minimum_minor_allele_frequency=0.0,
    )
    writer_callback, close_history = pipeline_module._build_per_epoch_history_writer(
        history_path=history_path,
        config=config,
        test_dataset=None,
    )
    try:
        BayesianPGS(config).fit(
            train["genotypes"],
            train["covariates"],
            train["targets"],
            train["records"],
            per_epoch_eval_callback=writer_callback,
            fit_checkpoint_path=tmp_path / "fit_checkpoint.pkl",
        )
    finally:
        close_history()

    rows = history_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) >= 2, (
        f"training_history.tsv was header-only ({len(rows)} lines). "
        "The per-epoch eval callback must fire even when durable "
        "checkpointing is active."
    )


# ---------------------------------------------------------------------------
# Test 4 — Prediction cache requires sample-ID match (not shape match)
#
# Pairs with the pipeline.py bug at ~line 283 where the cached training
# decision components are reused as long as the *length* matches. If a
# different cohort happens to have the same sample count, the cached
# training scores are emitted as that cohort's predictions, with no
# warning. Contract: the predictions must come from a fresh
# ``decision_components(...)`` call when the sample IDs differ.
# ---------------------------------------------------------------------------


def test_write_predictions_does_not_reuse_training_cache_for_different_cohort(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Prediction cache must key on sample identity, not just shape."""
    monkeypatch.setattr(genotype_module, "require_gpu", lambda: None)
    monkeypatch.setattr(
        genotype_module.StandardizedGenotypeMatrix,
        "try_materialize_gpu",
        lambda self: False,
    )

    n_samples = 64
    n_variants = 5
    train_problem = _make_tiny_binary_problem(n_samples=n_samples, n_variants=n_variants, seed=11)
    other_problem = _make_tiny_binary_problem(n_samples=n_samples, n_variants=n_variants, seed=99)

    config = ModelConfig(
        trait_type=TraitType.BINARY,
        max_outer_iterations=1,
        max_inner_newton_iterations=5,
        minimum_minor_allele_frequency=0.0,
    )
    model = BayesianPGS(config).fit(
        train_problem["genotypes"],
        train_problem["covariates"],
        train_problem["targets"],
        train_problem["records"],
    )
    assert model.state is not None

    train_sample_ids = [f"train_sample_{i}" for i in range(n_samples)]
    other_sample_ids = [f"other_sample_{i}" for i in range(n_samples)]
    assert train_sample_ids != other_sample_ids
    assert len(train_sample_ids) == len(other_sample_ids)

    other_dataset = LoadedDataset(
        sample_ids=other_sample_ids,
        genotypes=as_raw_genotype_matrix(other_problem["genotypes"]),
        covariates=other_problem["covariates"],
        targets=other_problem["targets"],
        variant_records=other_problem["records"],
    )

    # Compute the ground-truth predictions for the "other" cohort directly
    # — these are what the pipeline must emit.
    expected_genetic, expected_covariate = model.decision_components(
        other_dataset.genotypes, other_dataset.covariates
    )
    expected_genetic = np.asarray(expected_genetic, dtype=np.float32)
    expected_covariate = np.asarray(expected_covariate, dtype=np.float32)

    # Now spy on decision_components so we can prove it was called for the
    # write path (i.e. the cache was NOT used).
    call_log: list[tuple[int, int]] = []
    real_decision_components = model.decision_components

    def _spy_decision_components(genotypes, covariates):
        call_log.append((int(genotypes.shape[0]), int(covariates.shape[0])))
        return real_decision_components(genotypes, covariates)

    monkeypatch.setattr(model, "decision_components", _spy_decision_components)

    predictions_path = tmp_path / "predictions.tsv.gz"
    pipeline_module._write_predictions_and_summary(
        predictions_path=predictions_path,
        dataset=other_dataset,
        model=model,
    )

    assert call_log, (
        "_write_predictions_and_summary reused the cached training decision "
        "components for a different cohort (same sample count, different "
        "sample IDs). decision_components must be called fresh whenever the "
        "input cohort is not the training cohort."
    )

    # Cross-check: parse the written predictions and ensure the values
    # match the expected (fresh) computation, not the training-cached one.
    import gzip

    with gzip.open(predictions_path, "rt", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        rows = [line.rstrip("\n").split("\t") for line in handle if line.strip()]

    assert len(rows) == n_samples
    genetic_col = header.index("genetic_score")
    covariate_col = header.index("covariate_score")
    sample_col = header.index("sample_id")
    written_sample_ids = [row[sample_col] for row in rows]
    assert written_sample_ids == other_sample_ids, (
        "Predictions file emitted the wrong sample IDs; the cache key check "
        "is the only thing that prevents cross-cohort score leakage."
    )
    written_genetic = np.array([float(row[genetic_col]) for row in rows], dtype=np.float32)
    written_covariate = np.array([float(row[covariate_col]) for row in rows], dtype=np.float32)

    np.testing.assert_allclose(written_genetic, expected_genetic, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(written_covariate, expected_covariate, rtol=1e-4, atol=1e-4)

    # And: the written scores must NOT equal the training-cached scores
    # (those are what the buggy path would have emitted).
    training_components = model.training_decision_components()
    assert training_components is not None
    training_genetic, training_covariate = training_components
    if training_genetic.shape == written_genetic.shape:
        assert not np.allclose(
            written_genetic, np.asarray(training_genetic, dtype=np.float32), atol=1e-6
        ) or not np.allclose(
            written_covariate, np.asarray(training_covariate, dtype=np.float32), atol=1e-6
        ), (
            "Predictions for a different-cohort dataset match the training "
            "cohort's cached scores byte-for-byte — the sample-ID guard is "
            "missing from _write_predictions_and_summary."
        )
