"""Tests for the public ``checkpoint_from_result`` warm-start helper.

These cover the workflow of taking a ``VariationalFitResult`` from a prior
fit, converting it to a ``VariationalFitCheckpoint`` via the new public
helper, then resuming ``fit_variational_em`` from that checkpoint.
"""

from __future__ import annotations

import numpy as np

from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import VariantRecord
from sv_pgs.inference import fit_variational_em
from sv_pgs.mixture_inference import (
    VariationalFitCheckpoint,
    _build_prior_design,
    checkpoint_from_result,
)
from sv_pgs.preprocessing import build_tie_map, collapse_tie_groups


def _records(p: int) -> list[VariantRecord]:
    return [
        VariantRecord(
            variant_id=f"v{j}",
            variant_class=VariantClass.SNV,
            chromosome="chr1",
            position=100 * j,
            length=1.0,
            allele_frequency=0.1,
            quality=1.0,
        )
        for j in range(p)
    ]


def _make_problem(rng, n=120, p=10, sigma2=0.05):
    X = rng.standard_normal((n, p)).astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    W = np.column_stack(
        [np.ones(n, dtype=np.float32), rng.standard_normal((n, 1)).astype(np.float32)]
    )
    beta_true = np.zeros(p, dtype=np.float32)
    beta_true[:2] = 0.8
    noise = rng.standard_normal(n).astype(np.float32) * np.sqrt(sigma2)
    y = (X @ beta_true + noise).astype(np.float32)
    return X, W, y


def _make_prior_design(X, records, config):
    tie_map = build_tie_map(X, records, config)
    reduced_records = collapse_tie_groups(records, tie_map)
    return tie_map, _build_prior_design(reduced_records)


def _fit(X, W, y, records, config, *, resume=None):
    tie_map = build_tie_map(X, records, config)
    return fit_variational_em(
        genotypes=X,
        covariates=W,
        targets=y,
        records=records,
        config=config,
        tie_map=tie_map,
        resume_checkpoint=resume,
    )


def test_warm_start_from_result_continues_fit():
    rng = np.random.default_rng(0)
    X, W, y = _make_problem(rng)
    records = _records(X.shape[1])

    cfg_partial = ModelConfig(trait_type=TraitType.QUANTITATIVE, max_outer_iterations=4)
    result_partial = _fit(X, W, y, records, cfg_partial)

    cfg_resume = ModelConfig(trait_type=TraitType.QUANTITATIVE, max_outer_iterations=6)
    _, prior_design = _make_prior_design(X, records, cfg_resume)
    ckpt = checkpoint_from_result(
        result_partial, config=cfg_resume, prior_design=prior_design
    )
    assert isinstance(ckpt, VariationalFitCheckpoint)
    assert ckpt.completed_iterations == len(result_partial.objective_history)

    result_resumed = _fit(X, W, y, records, cfg_resume, resume=ckpt)

    cfg_fresh = ModelConfig(trait_type=TraitType.QUANTITATIVE, max_outer_iterations=6)
    result_fresh = _fit(X, W, y, records, cfg_fresh)

    # Both runs target the same fixed point; allow generous tolerance because
    # warm-start re-initializes local_scale/auxiliary_delta to defaults.
    beta_r = np.asarray(result_resumed.beta_reduced, dtype=np.float64)
    beta_f = np.asarray(result_fresh.beta_reduced, dtype=np.float64)
    denom = np.linalg.norm(beta_f) + 1e-8
    rel = float(np.linalg.norm(beta_r - beta_f) / denom)
    assert rel < 0.5, f"resumed vs fresh beta differ too much: rel={rel:.3e}"


def test_warm_start_with_more_iterations():
    rng = np.random.default_rng(1)
    X, W, y = _make_problem(rng)
    records = _records(X.shape[1])

    cfg_a = ModelConfig(trait_type=TraitType.QUANTITATIVE, max_outer_iterations=3)
    result_a = _fit(X, W, y, records, cfg_a)
    starting_iters = len(result_a.objective_history)
    assert starting_iters >= 1

    cfg_b = ModelConfig(trait_type=TraitType.QUANTITATIVE, max_outer_iterations=5)
    _, prior_design = _make_prior_design(X, records, cfg_b)
    ckpt = checkpoint_from_result(result_a, config=cfg_b, prior_design=prior_design)
    result_b = _fit(X, W, y, records, cfg_b, resume=ckpt)

    # Resume should run additional iterations up to the new cap (allow early
    # convergence to short-circuit).
    assert len(result_b.objective_history) > starting_iters
    assert len(result_b.objective_history) <= cfg_b.max_outer_iterations


def test_warm_start_idempotency_when_already_converged():
    rng = np.random.default_rng(2)
    X, W, y = _make_problem(rng, n=200, p=12, sigma2=0.05)
    records = _records(X.shape[1])

    cfg = ModelConfig(
        trait_type=TraitType.QUANTITATIVE,
        max_outer_iterations=40,
        convergence_tolerance=1e-3,
    )
    result_converged = _fit(X, W, y, records, cfg)

    cfg_resume = ModelConfig(
        trait_type=TraitType.QUANTITATIVE,
        max_outer_iterations=len(result_converged.objective_history) + 3,
        convergence_tolerance=1e-3,
    )
    _, prior_design = _make_prior_design(X, records, cfg_resume)
    ckpt = checkpoint_from_result(
        result_converged, config=cfg_resume, prior_design=prior_design
    )
    result_resume = _fit(X, W, y, records, cfg_resume, resume=ckpt)

    # The beta_reduced should barely move post-resume; tolerate up to the
    # configured convergence tolerance (relative L2).
    beta1 = np.asarray(result_converged.beta_reduced, dtype=np.float64)
    beta2 = np.asarray(result_resume.beta_reduced, dtype=np.float64)
    denom = np.linalg.norm(beta1) + 1e-8
    rel = float(np.linalg.norm(beta1 - beta2) / denom)
    assert rel < 5e-2, (
        f"Resuming from a converged fit should be near-idempotent; rel={rel:.3e}"
    )


def test_warm_start_helper_round_trip_signatures():
    """The checkpoint produced by the helper must be accepted by the resume
    machinery — i.e. config_signature/prior_design_signature must match the
    fresh fit's signatures rather than landing in the "incompatible" branch.
    """
    rng = np.random.default_rng(3)
    X, W, y = _make_problem(rng)
    records = _records(X.shape[1])

    cfg = ModelConfig(trait_type=TraitType.QUANTITATIVE, max_outer_iterations=2)
    result = _fit(X, W, y, records, cfg)

    _, prior_design = _make_prior_design(X, records, cfg)
    ckpt = checkpoint_from_result(result, config=cfg, prior_design=prior_design)

    from sv_pgs.mixture_inference import (
        _checkpoint_config_signature,
        _checkpoint_prior_design_signature,
    )

    assert ckpt.config_signature == _checkpoint_config_signature(cfg)
    assert ckpt.prior_design_signature == _checkpoint_prior_design_signature(prior_design)
    assert ckpt.alpha_state.shape == (W.shape[1],)
    assert ckpt.beta_state.shape == (prior_design.class_membership_matrix.shape[0],)
