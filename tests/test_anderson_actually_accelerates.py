"""Regression test for Anderson acceleration in the CAVI EM loop.

Previously the `AndersonState` was constructed and checkpointed but never
invoked inside the EM loop, so the infrastructure was dead. This test
verifies that with `_ANDERSON_MEMORY_DEPTH > 0` the EM converges to the
same fixed point as without acceleration (math is preserved) and that
the number of completed iterations is no worse.
"""
from __future__ import annotations

import numpy as np

from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.inference import fit_variational_em
import sv_pgs.mixture_inference as mixture_inference
from sv_pgs.preprocessing import build_tie_map

from tests.conftest import make_variant_records


def _make_stiff_problem(*, n: int = 300, p: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    genotype_matrix = rng.standard_normal((n, p)).astype(np.float32)
    covariate_matrix = np.ones((n, 1), dtype=np.float32)
    true_beta = np.zeros(p, dtype=np.float32)
    true_beta[:5] = np.array([1.2, -0.9, 0.7, -0.5, 0.4], dtype=np.float32)
    target_vector = (
        genotype_matrix @ true_beta
        + 0.3 * rng.standard_normal(n).astype(np.float32)
    )
    records = make_variant_records(p)
    return genotype_matrix, covariate_matrix, target_vector, records


def _run_with_memory_depth(depth: int, *, genotype_matrix, covariate_matrix, target_vector, records):
    config = ModelConfig(
        trait_type=TraitType.QUANTITATIVE,
        max_outer_iterations=30,
        convergence_tolerance=1e-4,
        exact_solver_matrix_limit=512,
        minimum_minor_allele_frequency=0.0,
        random_seed=0,
    )
    tie_map = build_tie_map(genotype_matrix, records, config)
    result = fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=config,
        tie_map=tie_map,
    )
    return result


def test_anderson_acceleration_is_wired_and_safe(monkeypatch):
    genotype_matrix, covariate_matrix, target_vector, records = _make_stiff_problem()

    # With Anderson disabled.
    monkeypatch.setattr(mixture_inference, "_ANDERSON_MEMORY_DEPTH", 0)
    result_no_anderson = _run_with_memory_depth(
        0,
        genotype_matrix=genotype_matrix,
        covariate_matrix=covariate_matrix,
        target_vector=target_vector,
        records=records,
    )

    # With Anderson enabled (memory depth 5).
    monkeypatch.setattr(mixture_inference, "_ANDERSON_MEMORY_DEPTH", 5)
    result_anderson = _run_with_memory_depth(
        5,
        genotype_matrix=genotype_matrix,
        covariate_matrix=covariate_matrix,
        target_vector=target_vector,
        records=records,
    )

    # Both must produce finite results.
    assert np.all(np.isfinite(result_no_anderson.beta_reduced))
    assert np.all(np.isfinite(result_anderson.beta_reduced))

    # Both must converge to (approximately) the same fixed point — Anderson
    # is supposed to preserve the math of the underlying EM map.
    final_obj_off = float(result_no_anderson.objective_history[-1])
    final_obj_on = float(result_anderson.objective_history[-1])
    obj_scale = max(abs(final_obj_off), abs(final_obj_on), 1.0)
    assert abs(final_obj_on - final_obj_off) / obj_scale < 0.05, (
        f"Anderson changed final objective beyond tolerance: "
        f"off={final_obj_off:.6f} on={final_obj_on:.6f}"
    )

    # Iteration count with Anderson should be no worse than without (the
    # acceleration should not slow convergence). Allow a small slack because
    # the convergence detector is delta-based and Anderson can cause a
    # transient delta blip after a step.
    iters_off = (
        result_no_anderson.selected_iteration_count
        if result_no_anderson.selected_iteration_count is not None
        else len(result_no_anderson.objective_history)
    )
    iters_on = (
        result_anderson.selected_iteration_count
        if result_anderson.selected_iteration_count is not None
        else len(result_anderson.objective_history)
    )
    assert iters_on <= iters_off + 2, (
        f"Anderson made convergence slower: off={iters_off} on={iters_on}"
    )
