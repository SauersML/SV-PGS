"""The compiled host objective (``engine_kernels.objective_statistics_host``) against the numpy reference
(``scale_mixture_ep._data_objective_numpy``): the same value, gradient and Hessian in z = (eta, theta)."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.scale_mixture_ep import (
    AnnotationGroup, Cavity, _data_objective, _data_objective_numpy, initial_hyperparameters, scale_mixture_prior,
)

_WORKING_BYTES = 1 << 20


def _problem(seed: int, classes: int, annotated: bool):
    generator = np.random.default_rng(seed)
    variant_count = 300
    nodes = np.linspace(np.log(1e-5), np.log(1.0), 12)
    design = generator.uniform(-1.0, 1.0, (variant_count, 2 if annotated else 0))
    groups = (AnnotationGroup(columns=np.array([0, 1]), penalty=np.eye(2)),) if annotated else ()
    prior = scale_mixture_prior(
        class_index=generator.integers(0, classes, variant_count).astype(np.int64),
        log_variance_offset=np.log(generator.uniform(0.2, 1.0, variant_count)), annotation_design=design, annotation_groups=groups,
        nodes=nodes, floor=nodes[0] - 1.0, top=nodes[-1],
    )
    # proper cavities of both signs of precision (a mean-field site's can be negative) and shifts from small to large
    precision = generator.uniform(-0.5, 400.0, variant_count)
    shift = generator.normal(0.0, 1.0, variant_count) * np.sqrt(np.abs(precision) + 1.0) * generator.uniform(0.0, 4.0, variant_count)
    coefficients = initial_hyperparameters(prior).coefficients + generator.normal(0.0, 0.3, prior.coefficient_size)
    return prior, Cavity(precision=precision, shift=shift), coefficients


@pytest.mark.parametrize("classes, annotated", [(1, False), (3, False), (2, True)])
def test_the_compiled_host_objective_is_the_numpy_one(classes: int, annotated: bool) -> None:
    prior, cavity, coefficients = _problem(7 + classes, classes, annotated)
    compiled = _data_objective(prior, coefficients, cavity, _WORKING_BYTES)
    reference = _data_objective_numpy(prior, coefficients, cavity, _WORKING_BYTES)
    assert compiled.value == pytest.approx(reference.value, rel=1e-12, abs=1e-9)
    np.testing.assert_allclose(compiled.gradient, reference.gradient, rtol=1e-9, atol=1e-9 * (1.0 + np.max(np.abs(reference.gradient))))
    np.testing.assert_allclose(compiled.hessian, reference.hessian, rtol=1e-9, atol=1e-9 * (1.0 + np.max(np.abs(reference.hessian))))
    # the gradient's pass leaves the Hessian zero and the gradient unchanged
    gradient_only = _data_objective(prior, coefficients, cavity, _WORKING_BYTES, hessian_too=False)
    np.testing.assert_array_equal(gradient_only.gradient, compiled.gradient)
    assert not np.any(gradient_only.hessian)


def test_chunks_do_not_change_the_objective() -> None:
    prior, cavity, coefficients = _problem(3, 2, True)
    whole = _data_objective(prior, coefficients, cavity, _WORKING_BYTES)
    # a budget of a few rows a chunk
    pieces = _data_objective(prior, coefficients, cavity, 2 * 8 * (prior.grid_size + 4) * 7)
    assert pieces.value == pytest.approx(whole.value, rel=1e-13)
    np.testing.assert_allclose(pieces.hessian, whole.hessian, rtol=1e-11, atol=1e-11 * np.max(np.abs(whole.hessian)))
