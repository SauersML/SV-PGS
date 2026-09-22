"""The prior's annotation design from a variant table's annotation columns."""

import numpy as np
import pytest

from sv_pgs.annotation_design import annotation_design
from sv_pgs.scale_mixture_ep import class_log_density, initial_hyperparameters, scale_mixture_prior


def _table(rng: np.random.Generator, count: int):
    classes = rng.choice(3, size=count).astype(np.int64)
    annotations = {
        "quality": rng.uniform(0.2, 1.0, count),
        "in_gene": (rng.random(count) < 0.3).astype(np.float64),
        "log_length": np.where(classes > 0, rng.normal(3.0, 1.0, count), np.nan),
        "distance": rng.exponential(1.0, count),
        "constant": np.ones(count),
        "kind": rng.choice(3, size=count).astype(np.int32),
    }
    legends = {"kind": ("a", "b", "c")}
    return classes, annotations, legends


def test_each_annotation_becomes_one_group_of_a_full_rank_class_centred_design() -> None:
    rng = np.random.default_rng(11)
    classes, annotations, legends = _table(rng, 400)
    result = annotation_design(annotations, legends, class_index=classes, exclude=("quality",))
    assert result.design.shape == (400, len(result.names))
    # the reliability column and the constant one leave no column; the rest are there
    assert not any(name.startswith("quality") or name.startswith("constant") for name in result.names)
    assert "in_gene" in result.names
    assert "log_length:linear" in result.names and "log_length:missing" in result.names
    assert any(name.startswith("distance:hinge@") for name in result.names)
    assert sum(name.startswith("kind=") for name in result.names) == 2
    # one group per annotation, covering every column once, with a penalty per column
    covered = np.concatenate([group.columns for group in result.groups])
    assert np.array_equal(np.sort(covered), np.arange(result.design.shape[1]))
    assert len(result.groups) == 4
    for group in result.groups:
        assert group.penalty.shape == (group.columns.shape[0],) * 2
    # a smoothing spline's linear term and missing indicator are free; its hinges are penalised
    for group in result.groups:
        for column, weight in zip(group.columns, np.diag(group.penalty)):
            name = result.names[column]
            assert weight == (1.0 if ("hinge" in name or ":" not in name) else 0.0), name
    # the prior accepts it: the class-centred design has full column rank, and theta shifts log u_j through it
    nodes = np.linspace(np.log(1e-5), np.log(0.5), 24)
    prior = scale_mixture_prior(
        class_index=classes, log_variance_offset=np.log(annotations["quality"]), annotation_design=result.design,
        annotation_groups=result.groups, nodes=nodes, floor=nodes[0] - 1.0, top=nodes[-1],
    )
    assert prior.scale_design.shape == result.design.shape
    assert len(prior.smoothing_blocks) >= len(result.groups)
    start = initial_hyperparameters(prior)
    assert np.all(np.isfinite(class_log_density(prior, start.coefficients)))


def test_a_column_constant_within_every_class_and_a_dependent_column_are_dropped() -> None:
    rng = np.random.default_rng(3)
    classes = np.repeat(np.arange(2), 50)
    values = rng.normal(size=100)
    annotations = {"per_class": classes.astype(np.float64), "x": values, "twice_x": 2.0 * values + 1.0}
    result = annotation_design(annotations, {}, class_index=classes)
    assert not any(name.startswith("per_class") for name in result.names)
    # x and 2x + 1 share every column after centring: the second annotation's linear term is dependent and dropped,
    # and so are its hinges (the same knots on the same standardized values)
    assert sum(name.startswith("twice_x") for name in result.names) == 0
    assert sum(name.startswith("x") for name in result.names) >= 1


def test_an_annotation_with_the_wrong_length_is_refused() -> None:
    with pytest.raises(ValueError, match="one value per variant"):
        annotation_design({"x": np.zeros(3)}, {}, class_index=np.zeros(4, dtype=np.int64))
