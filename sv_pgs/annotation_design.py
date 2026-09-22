"""The prior's annotation design from a variant table's annotation columns.

Each annotation becomes one ``AnnotationGroup`` of the scale-mixture prior (``scale_mixture_ep.scale_mixture_prior``):
columns of the class-centred design d_j that enter log u_j = o_j + d_j' theta, and a penalty whose weight the
empirical Bayes learns with the rest of the hyperparameters. Nothing here is chosen by hand: a factor is its
indicators, a continuous annotation is a smoothing spline in its truncated-power form (a linear term and cubic
hinges at its interior quartile knots, the same basis ``prior_design`` compiles), whose penalty is the standard
mixed-model one, the hinge coefficients' squared norm with the linear term free (Ruppert, Wand and Carroll,
Semiparametric Regression, section 3.5), and a missing value is the column's mean with an indicator that says so.

The design is class-centred (the class densities carry each class's location) and then rank-screened by the same
deterministic Gram-Schmidt screen ``prior_design`` uses, so the prior's full-column-rank check holds: an annotation
constant within every class, or a hinge two nearby knots make dependent, leaves no column. Reliability columns are
offsets, not annotations, and the caller names them to leave out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from sv_pgs._typing import F64Array, I64Array
from sv_pgs.prior_design import _continuous_spline_knots
from sv_pgs.scale_mixture_ep import AnnotationGroup

_EPSILON = float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class AnnotationDesign:
    """The prior's annotation inputs: the design [variants, columns], its groups and a name per column."""

    design: F64Array
    groups: tuple[AnnotationGroup, ...]
    names: tuple[str, ...]


def _is_indicator(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    return finite.shape[0] > 0 and bool(np.all((finite == 0.0) | (finite == 1.0)))


def _factor_columns(name: str, codes: np.ndarray, legend: Sequence[str]) -> tuple[list[F64Array], list[str]]:
    """Indicators of every level but the most frequent one (the reference; the class densities carry the constant)."""
    counts = np.bincount(codes.astype(np.int64), minlength=len(legend))
    reference = int(np.argmax(counts))
    columns, names = [], []
    for level in range(len(legend)):
        if level == reference or counts[level] == 0:
            continue
        columns.append((codes == level).astype(np.float64))
        names.append(f"{name}={legend[level]}")
    return columns, names


def _continuous_columns(name: str, values: np.ndarray) -> tuple[list[F64Array], list[str], list[float]]:
    """The truncated-power smoothing-spline basis of a continuous annotation, with a missing indicator where it has
    non-finite values; returns the columns, their names and their penalty diagonal (0 free, 1 penalised)."""
    finite = np.isfinite(values)
    if not finite.any():
        return [], [], []
    filled = np.where(finite, values, values[finite].mean())
    mean, scale = float(filled.mean()), float(filled.std())
    if scale <= _EPSILON * max(float(np.max(np.abs(filled))), 1.0):
        return [], [], []
    standardized = (filled - mean) / scale
    columns, names, penalty = [standardized], [f"{name}:linear"], [0.0]
    for knot in _continuous_spline_knots(standardized[finite]):
        columns.append(np.maximum(standardized - knot, 0.0) ** 3)
        names.append(f"{name}:hinge@{knot:.4g}")
        penalty.append(1.0)
    if not finite.all():
        columns.append((~finite).astype(np.float64))
        names.append(f"{name}:missing")
        penalty.append(0.0)
    return columns, names, penalty


def _class_centred(column: F64Array, class_rows: Sequence[I64Array]) -> F64Array:
    centred = column.copy()
    for rows in class_rows:
        centred[rows] -= centred[rows].mean()
    return centred


def annotation_design(
    annotations: Mapping[str, np.ndarray],
    annotation_legends: Mapping[str, Sequence[str]],
    *,
    class_index: I64Array,
    exclude: Sequence[str] = (),
) -> AnnotationDesign:
    """The prior's annotation design over the rows of ``annotations`` (already restricted to the prior's members),
    one group per annotation, rank-screened after class centring; ``exclude`` names the columns that are not
    annotations (a reliability column is the prior's offset)."""
    classes = np.asarray(class_index, dtype=np.int64)
    order = np.argsort(classes, kind="stable")
    class_rows = tuple(np.split(order, np.cumsum(np.bincount(classes))[:-1]))
    orthonormal: list[F64Array] = []
    kept_columns: list[F64Array] = []
    kept_names: list[str] = []
    groups: list[AnnotationGroup] = []

    def independent(column: F64Array) -> bool:
        centred = _class_centred(column, class_rows)
        norm = float(np.linalg.norm(centred))
        if norm <= _EPSILON * centred.shape[0] * max(float(np.max(np.abs(column))), 1.0):
            return False
        residual = centred / norm
        for _ in range(2):
            for basis in orthonormal:
                residual -= basis * float(basis @ residual)
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= max(centred.shape[0], len(orthonormal) + 1) * _EPSILON:
            return False
        orthonormal.append(residual / residual_norm)
        return True

    for name in sorted(annotations):
        if name in exclude:
            continue
        raw = annotations[name]
        if name in annotation_legends:
            columns, names = _factor_columns(name, np.asarray(raw), annotation_legends[name])
            penalty = [1.0] * len(columns)
        else:
            values = np.asarray(raw, dtype=np.float64)
            if values.shape[0] != classes.shape[0]:
                raise ValueError(f"annotation {name!r} needs one value per variant")
            if _is_indicator(values):
                columns, names, penalty = [np.where(np.isfinite(values), values, 0.0)], [name], [1.0]
            else:
                columns, names, penalty = _continuous_columns(name, values)
        members, member_penalty = [], []
        for column, column_name, weight in zip(columns, names, penalty):
            if independent(column):
                members.append(len(kept_columns))
                member_penalty.append(weight)
                kept_columns.append(column)
                kept_names.append(column_name)
        if members:
            groups.append(AnnotationGroup(columns=np.asarray(members, dtype=np.int64), penalty=np.diag(member_penalty)))
    design = np.column_stack(kept_columns) if kept_columns else np.zeros((classes.shape[0], 0))
    return AnnotationDesign(design=design, groups=tuple(groups), names=tuple(kept_names))
