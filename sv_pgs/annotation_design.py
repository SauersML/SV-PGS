"""The prior's annotation design from a variant table's annotation columns.

Each annotation becomes one ``AnnotationGroup`` of the scale-mixture prior (``scale_mixture_ep.scale_mixture_prior``):
columns of the class-centred design d_j that enter log u_j = o_j + d_j' theta, and a penalty whose weight the
empirical Bayes learns with the rest of the hyperparameters. Nothing here is chosen by hand: a factor is its
indicators, a continuous annotation is a smoothing spline in its truncated-power form (a linear term and cubic
hinges at its interior quartile knots, the same basis ``prior_design`` compiles), whose penalty is the standard
mixed-model one, the hinge coefficients' squared norm with the linear term free (Ruppert, Wand and Carroll,
Semiparametric Regression, section 3.5), and a missing value is the column's mean with an indicator that says so.

The design is class-centred (the class densities carry each class's location) and rank-screened twice: the
deterministic Gram-Schmidt screen ``prior_design`` uses, then the prior's own rank test by a column-pivoted QR, so
an annotation constant within every class, or a hinge two nearby knots or a nearly-all-zero column make dependent,
leaves no column. Reliability columns are
offsets, not annotations, and the caller names them to leave out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import scipy.linalg

from sv_pgs._typing import F64Array, I64Array
from sv_pgs.prior_design import _continuous_spline_knots
from sv_pgs.scale_mixture_ep import AnnotationGroup

_EPSILON = float(np.finfo(np.float64).eps)


COLUMN_VARIANCE = "log_column_variance"
"""The annotation every route adds from the training genotypes themselves: each column's log training variance."""


def column_variance_annotation(scales: np.ndarray) -> dict[str, np.ndarray]:
    """{COLUMN_VARIANCE: 2 log s_j} for the members' training standard deviations s_j (any constant unit cancels in the
    class centring). Its free linear coefficient c in log u_j spans the frequency-dependent architectures: c = 0 is one
    prior on standardized effects, c = 1 one prior on effects per unit of the stored value (per allele for a dosage,
    mr.ash's scale), and the empirical Bayes learns c with the smooth's curvature, from the training data alone. On the
    bench-real genes where SV-PGS trailed mr.ash most, c = 1 raised both the training ELBO and the held-out r2
    (ENSG00000124613.9 0.21 -> 0.52, ENSG00000237248.5 0.13 -> 0.44, loso/AFR [real])."""
    scales = np.asarray(scales, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return {COLUMN_VARIANCE: np.where(scales > 0.0, 2.0 * np.log(np.where(scales > 0.0, scales, 1.0)), np.nan)}


def per_unit_offset(scales: np.ndarray) -> F64Array:
    """Each member's log prior-variance baseline at the per-unit architecture: 2 log s_j less its largest (so every
    offset stays at or below 0, as a reliability's does), one prior per unit of the stored value. The prior starts
    there and the column-variance annotation's free coefficient moves it toward one prior on standardized effects
    where the data say so: the empirical Bayes searches locally, and on the bench-real genes where SV-PGS trailed
    mr.ash most the per-unit baseline held a higher training ELBO the standardized start never reached."""
    scales = np.asarray(scales, dtype=np.float64)
    positive = scales > 0.0
    log_variance = np.where(positive, 2.0 * np.log(np.where(positive, scales, 1.0)), 0.0)
    return np.where(positive, log_variance - float(np.max(log_variance[positive])), 0.0) if positive.any() else np.zeros_like(scales)


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
    return _full_rank(design, tuple(groups), tuple(kept_names), class_rows)


def _full_rank(design: F64Array, groups: tuple[AnnotationGroup, ...], names: tuple[str, ...], class_rows) -> AnnotationDesign:
    """The columns the prior's own rank test accepts (``scale_mixture_prior``: the class-centred Gram's least eigenvalue
    above eps p max(its largest, 1)), by a column-pivoted QR of the class-centred design: a column is kept while its
    pivot's squared diagonal passes that test, in the pivot order, and the groups are re-indexed over what remains."""
    if design.shape[1] == 0:
        return AnnotationDesign(design=design, groups=groups, names=names)
    centred = np.column_stack([_class_centred(design[:, column], class_rows) for column in range(design.shape[1])])
    _q, triangle, pivots = scipy.linalg.qr(centred, mode="economic", pivoting=True)
    squares = np.square(np.abs(np.diag(triangle)))
    threshold = _EPSILON * centred.shape[0] * max(float(squares[0]), 1.0)
    passing = np.flatnonzero(squares <= threshold)
    order = list(pivots[: passing[0]] if passing.size else pivots)
    # The QR diagonal bounds the singular values only up to the column count; the prior's exact test decides.
    while order:
        gram = centred[:, order].T @ centred[:, order]
        eigenvalues = np.linalg.eigvalsh(gram)
        if eigenvalues[0] > _EPSILON * centred.shape[0] * max(float(eigenvalues[-1]), 1.0):
            break
        order.pop()
    kept = np.sort(np.asarray(order, dtype=np.int64))
    position = {column: index for index, column in enumerate(kept)}
    new_groups = []
    for group in groups:
        members = [index for index, column in enumerate(group.columns) if int(column) in position]
        if members:
            new_groups.append(AnnotationGroup(
                columns=np.array([position[int(group.columns[index])] for index in members], dtype=np.int64),
                penalty=np.asarray(group.penalty)[np.ix_(members, members)],
            ))
    return AnnotationDesign(design=design[:, kept], groups=tuple(new_groups), names=tuple(names[column] for column in kept))
