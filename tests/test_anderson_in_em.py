"""Anderson acceleration on the outer CAVI EM map.

The Anderson wrapper sits on top of the slowly-changing hyperparameter
slice. The safeguard checks finiteness + bounds, falling back to the
plain CAVI step when violated; this test verifies that
  * enabling Anderson does not destabilize the fit (objective agrees
    with the plain CAVI run to within tolerance), and
  * Anderson does not slow down convergence (iteration count to a
    fixed tolerance is at most the count without Anderson).
"""
from __future__ import annotations

import numpy as np

from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.inference import fit_variational_em
from sv_pgs.preprocessing import build_tie_map

from tests.conftest import make_variant_records


def _make_small_quantitative_problem(rng: np.random.Generator):
    sample_count, variant_count = 120, 16
    genotype_matrix = rng.standard_normal((sample_count, variant_count)).astype(np.float32)
    covariate_matrix = np.column_stack(
        [np.ones(sample_count), rng.standard_normal((sample_count, 2))]
    ).astype(np.float32)
    true_beta = np.zeros(variant_count, dtype=np.float32)
    true_beta[:4] = np.array([1.2, -0.8, 0.6, -0.4], dtype=np.float32)
    target_vector = (
        genotype_matrix @ true_beta
        + rng.standard_normal(sample_count).astype(np.float32) * 0.5
    )
    records = make_variant_records(variant_count)
    return genotype_matrix, covariate_matrix, target_vector, records


def _run(config: ModelConfig, problem):
    genotype_matrix, covariate_matrix, target_vector, records = problem
    return fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=config,
        tie_map=build_tie_map(genotype_matrix, records, config),
    )


def _config(*, anderson_acceleration: bool, max_iters: int = 30) -> ModelConfig:
    return ModelConfig(
        trait_type=TraitType.QUANTITATIVE,
        max_outer_iterations=max_iters,
        convergence_tolerance=1e-4,
        stochastic_variational_updates=False,
        anderson_acceleration=anderson_acceleration,
        anderson_memory_depth=5,
    )


def test_anderson_acceleration_preserves_objective_on_small_quantitative_problem():
    rng = np.random.default_rng(0)
    problem = _make_small_quantitative_problem(rng)

    result_off = _run(_config(anderson_acceleration=False), problem)
    result_on = _run(_config(anderson_acceleration=True), problem)

    assert result_off.objective_history
    assert result_on.objective_history

    # The outer CAVI iterates can oscillate in the late tail (the inner
    # collapsed posterior is solved exactly each step but hyperparameter
    # updates can overshoot). Compare best-objective across the run; the
    # safeguard guarantees Anderson never strictly degrades the achievable
    # objective.
    best_off = max(result_off.objective_history)
    best_on = max(result_on.objective_history)
    tolerance = max(1.0, abs(best_off)) * 1e-2
    assert best_on >= best_off - tolerance, (
        f"Anderson best objective {best_on} below baseline {best_off}"
    )


def test_anderson_does_not_slow_outer_em_convergence():
    rng = np.random.default_rng(1)
    problem = _make_small_quantitative_problem(rng)

    result_off = _run(_config(anderson_acceleration=False), problem)
    result_on = _run(_config(anderson_acceleration=True), problem)

    iterations_off = len(result_off.objective_history)
    iterations_on = len(result_on.objective_history)
    # Anderson should never require materially MORE outer iterations than the
    # plain CAVI baseline under the safeguard; allow a 1-iteration slack to
    # absorb harmless reordering of the early seeding step.
    assert iterations_on <= iterations_off + 1, (
        f"Anderson took {iterations_on} iterations vs baseline {iterations_off}"
    )
