"""The prior's annotation design from a variant table's annotation columns.

Each annotation becomes one named ``AnnotationGroup`` of the scale-mixture prior (``scale_mixture_ep.scale_mixture_prior``):
columns of the class-centred design d_j that enter the raw-unit log prior variance a_j = d_j' theta (added to the
unit offset, ``imputation_reliability.ColumnMeasurement``), and a penalty whose weight the empirical Bayes learns with
the rest of the hyperparameters. Nothing here is chosen by hand:

- A continuous annotation f(x) is a cubic spline on a local-support B-spline basis over its observed range, penalized
  by its roughness as a function, the integral of f''(x)^2 over that range (the O'Sullivan penalty; Wand and Ormerod,
  "On semiparametric regression with O'Sullivan penalized splines", 2008), the basis and its exact penalty built by
  gamfit (``gamfit.basis.bspline_basis``, ``smoothness_penalty``). The penalty's null space, the
  linear functions, is handled explicitly: the constant is the class densities' (class centring removes it) and the
  linear term is a free column; the penalized part is the basis rotated onto the penalty's eigenvectors, so its
  penalty is diagonal there and is the same function-space penalty, transformed with the basis (a change of basis
  that drops the penalty's null directions changes no function's roughness). x is the annotation standardized over
  its observed values; the knots are its interior quartiles (``prior_design._continuous_spline_knots``), where the
  spline space refines toward the full smoothing spline as knots are added, and the penalty, not the knot count,
  sets the smoothness.
- A factor (a legend-coded annotation, or a numeric one whose observed values are all 0 or 1) is a level per
  distinct observed value, with an exchangeable Gaussian prior on the levels about their mean: the level indicators
  times the closed-form orthonormal sum-to-zero (Helmert) basis, penalty identity, so the prior does not depend on
  which level comes first and no dense level-by-level change of basis is formed (the rows are gathered from it).
- A missing value is its own state, built apart from the numeric basis and kept whenever it is identified: the
  numeric columns are 0 on a missing row, a continuous annotation gets a free missing-state column, and a factor gets
  a missing level. So an annotation constant where observed but missing elsewhere keeps its missing state (its
  observed level is the missing state's complement after class centring), and a binary annotation's unknown stays
  apart from its false (review F19).

The design is class-centred (the class densities carry each class's location) and rank-screened once, in the order
the columns are built, against the class-centred Gram the prior itself tests (``scale_mixture_prior``: its least
eigenvalue above eps p max(its largest, 1)): a column is dropped where it adds nothing resolvable to the ones before
it, so an annotation constant within every class, a spline direction that too few distinct values support, or a
nearly-all-zero column leaves nothing. A dropped column of a penalized group restricts that group's function space,
and its penalty becomes the restriction of the quadratic form. Reliability columns are the unit contract's, not
annotations, and the caller names them to leave out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse
from gamfit.basis import bspline_basis, smoothness_penalty

from sv_pgs._typing import F64Array, I64Array
from sv_pgs.prior_design import _continuous_spline_knots
from sv_pgs.scale_mixture_ep import AnnotationGroup, _sum_to_zero_rows

_EPSILON = float(np.finfo(np.float64).eps)
# A cubic spline: f'' continuous and piecewise linear, the lowest degree whose roughness integral f''^2 is finite and
# whose penalty null space is exactly the linear functions.
_SPLINE_DEGREE = 3
# The roughness is the integral of the squared second derivative: its null space, the linear functions, is the
# smooth's free fixed part (the constant going to the class densities).
_ROUGHNESS_DERIVATIVE = 2


FREQUENCY = "log_genotype_variance"
"""The frequency function's annotation: each member's log true-genotype variance log Var(G_j), from the unit contract
(``imputation_reliability.ColumnMeasurement.standardized_terms``)."""


def frequency_annotation(log_genotype_variance: np.ndarray) -> dict[str, np.ndarray]:
    """{FREQUENCY: log Var(G_j)}: the frequency function h(log Var G_j), a learned smooth of the raw-unit log prior
    variance, as its own named group. Its linear coefficient c spans the alpha models of frequency-dependent
    architecture, Var(b_j) proportional to Var(G_j)^c: c = 0 is one prior per raw unit (per allele for a dosage,
    mr.ash's scale), c = -1 one prior on true-genotype-standardized effects (whose standardized-dosage variance is then
    r^2 tau^2, the SPEC's r^2-scaled prior), and its curvature departs from both where the data say so; the empirical
    Bayes learns it from the training data alone. On the bench-real genes where SV-PGS trailed mr.ash most, the
    per-unit end raised both the training ELBO and the held-out r2 (ENSG00000124613.9 0.21 -> 0.52,
    ENSG00000237248.5 0.13 -> 0.44, loso/AFR [real]), which is why the unit offset sits there and this term moves away
    from it. Any constant unit cancels in the class centring."""
    return {FREQUENCY: np.asarray(log_genotype_variance, dtype=np.float64)}


@dataclass(frozen=True)
class AnnotationDesign:
    """The prior's annotation inputs: the design [variants, columns], its groups and a name per column."""

    design: F64Array
    groups: tuple[AnnotationGroup, ...]
    names: tuple[str, ...]


@dataclass
class _Candidate:
    """One annotation's candidate columns (dense, one row per variant), their names and the group's penalty."""

    name: str
    columns: list[F64Array]
    names: list[str]
    penalty: F64Array


def _is_indicator(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    return finite.shape[0] > 0 and bool(np.all((finite == 0.0) | (finite == 1.0)))


def _factor_candidate(name: str, level_of_row: I64Array, labels: Sequence[str]) -> _Candidate:
    """Level contrasts of a factor: row j's level indicator times the orthonormal sum-to-zero basis, gathered by level.

    With levels l ~ N(0, (I - 11'/L) / lambda) (exchangeable about their mean, which the class densities carry), the
    coordinates c = H' l of the Helmert basis H are N(0, I / lambda) and row j's contribution is H[level_j] c."""
    present, level = np.unique(level_of_row, return_inverse=True)
    if present.shape[0] < 2:
        return _Candidate(name, [], [], np.zeros((0, 0)))
    contrasts = _sum_to_zero_rows(level, present.shape[0])
    names = [f"{name}:contrast({labels[int(present[position + 1])]})" for position in range(present.shape[0] - 1)]
    return _Candidate(name, [np.ascontiguousarray(contrasts[:, position]) for position in range(contrasts.shape[1])], names, np.eye(contrasts.shape[1]))


def spline_basis(standardized: F64Array) -> tuple[F64Array, F64Array, F64Array]:
    """(knots, B, S) of a cubic B-spline on ``standardized``'s range with its interior quartile knots: the basis at
    those points (n x K, local support: at most 4 nonzero per row) and the exact roughness penalty S_ik = integral of
    B_i''(x) B_k''(x) dx (K x K), both from gamfit (``gamfit.basis``)."""
    low, high = float(np.min(standardized)), float(np.max(standardized))
    interior = np.asarray(_continuous_spline_knots(standardized), dtype=np.float64)
    knots = np.concatenate([np.full(_SPLINE_DEGREE + 1, low), interior, np.full(_SPLINE_DEGREE + 1, high)])
    basis = np.asarray(bspline_basis(np.clip(standardized, low, high), knots, degree=_SPLINE_DEGREE), dtype=np.float64)
    penalty, _null_basis = smoothness_penalty(knots, degree=_SPLINE_DEGREE, order=_ROUGHNESS_DERIVATIVE)
    return knots, basis, np.asarray(penalty, dtype=np.float64)


def _continuous_candidate(name: str, values: F64Array) -> _Candidate:
    """A continuous annotation's free linear term, its penalized spline directions (the basis on the penalty's range
    space, penalty diagonal there) and, where it has missing values, a free missing-state column; every numeric column
    is 0 on a missing row, and the missing state is built whatever the observed values are."""
    observed = np.isfinite(values)
    columns: list[F64Array] = []
    names: list[str] = []
    penalty: list[float] = []
    finite = values[observed]
    if np.unique(finite).shape[0] >= 2:
        mean, scale = float(finite.mean()), float(finite.std())
        standardized = (finite - mean) / scale
        linear = np.zeros(values.shape[0])
        linear[observed] = standardized
        columns.append(linear)
        names.append(f"{name}:linear")
        penalty.append(0.0)
        _knots, basis, roughness = spline_basis(standardized)
        eigenvalues, eigenvectors = np.linalg.eigh(roughness)
        # The penalty's null space is the linear functions (rank K - 2); its range directions carry the roughness.
        rough = eigenvalues > _EPSILON * eigenvalues.shape[0] * max(float(eigenvalues[-1]), np.finfo(np.float64).tiny)
        rotated = basis @ eigenvectors[:, rough]
        for position in range(rotated.shape[1]):
            column = np.zeros(values.shape[0])
            column[observed] = rotated[:, position]
            columns.append(column)
            names.append(f"{name}:spline{position + 1}")
        penalty.extend(eigenvalues[rough].tolist())
    if not observed.all():
        columns.append((~observed).astype(np.float64))
        names.append(f"{name}:missing")
        penalty.append(0.0)
    return _Candidate(name, columns, names, np.diag(penalty))


def _candidate(name: str, raw: np.ndarray, legend: Sequence[str] | None, variant_count: int) -> _Candidate:
    if legend is not None:
        codes = np.asarray(raw).astype(np.int64)
        if codes.shape != (variant_count,):
            raise ValueError(f"annotation {name!r} needs one value per variant")
        missing = (codes < 0) | (codes >= len(legend))
        # The missing state is one more level, after every legend level.
        return _factor_candidate(name, np.where(missing, len(legend), codes), [*legend, "missing"])
    values = np.asarray(raw, dtype=np.float64)
    if values.shape != (variant_count,):
        raise ValueError(f"annotation {name!r} needs one value per variant")
    if _is_indicator(values):
        missing = ~np.isfinite(values)
        return _factor_candidate(name, np.where(missing, 2, np.nan_to_num(values)).astype(np.int64), ["0", "1", "missing"])
    return _continuous_candidate(name, values)


def _class_sums(columns: F64Array, classes: I64Array, class_count: int) -> F64Array:
    """Each class's column sums (classes x columns): the sparse class incidence times the columns."""
    incidence = sparse.csr_matrix((np.ones(classes.shape[0]), (classes, np.arange(classes.shape[0]))), shape=(class_count, classes.shape[0]))
    return np.asarray(incidence @ columns)


def annotation_design(
    annotations: Mapping[str, np.ndarray],
    annotation_legends: Mapping[str, Sequence[str]],
    *,
    class_index: I64Array,
    exclude: Sequence[str] = (),
) -> AnnotationDesign:
    """The prior's annotation design over the rows of ``annotations`` (already restricted to the prior's members),
    one named group per annotation in name order, rank-screened after class centring; ``exclude`` names the columns
    that are not annotations (a reliability column is the unit contract's)."""
    classes = np.asarray(class_index, dtype=np.int64)
    variant_count = classes.shape[0]
    class_count = int(classes.max()) + 1 if variant_count else 0
    candidates = [
        _candidate(name, annotations[name], annotation_legends.get(name), variant_count)
        for name in sorted(annotations) if name not in exclude
    ]
    flat = [(position, index) for position, candidate in enumerate(candidates) for index in range(len(candidate.columns))]
    if not flat:
        return AnnotationDesign(design=np.zeros((variant_count, 0)), groups=(), names=())
    design = np.column_stack([candidates[position].columns[index] for position, index in flat])
    counts = np.bincount(classes, minlength=class_count).astype(np.float64)
    centred = design - (_class_sums(design, classes, class_count) / np.maximum(counts, 1.0)[:, None])[classes]
    kept = _screened(centred)
    groups = []
    names = []
    for new_position, column in enumerate(kept):
        names.append(candidates[flat[column][0]].names[flat[column][1]])
    for position, candidate in enumerate(candidates):
        members = [(new_position, flat[column][1]) for new_position, column in enumerate(kept) if flat[column][0] == position]
        if members:
            local = np.array([index for _new, index in members], dtype=np.int64)
            groups.append(AnnotationGroup(
                columns=np.array([new for new, _index in members], dtype=np.int64),
                penalty=candidate.penalty[np.ix_(local, local)],
                name=candidate.name,
            ))
    return AnnotationDesign(design=design[:, kept], groups=tuple(groups), names=tuple(names))


def _screened(centred: F64Array) -> I64Array:
    """The columns of the class-centred design kept in order: a column enters when the Gram of the kept columns with
    it passes the prior's exact test (``scale_mixture_prior``: its least eigenvalue above eps p max(largest, 1)),
    with the largest eigenvalue of the whole candidate Gram in the bound, which is at least that of every subset's, so
    the kept design passes the prior's test. A column is dropped only for what it adds to the ones before it: a test
    run once on the whole kept set and failed by removing the latest columns dropped whole later annotations for an
    earlier one's near-dependence (every log_tss_distance column on ENSG00000187605.16 [real], for the SV length
    smooths' few rows)."""
    count, size = centred.shape
    gram = centred.T @ centred
    largest = float(np.linalg.eigvalsh(gram)[-1]) if size else 0.0
    bound = _EPSILON * count * max(largest, 1.0)
    kept: list[int] = []
    for column in range(size):
        trial = [*kept, column]
        if float(np.linalg.eigvalsh(gram[np.ix_(trial, trial)])[0]) > bound:
            kept.append(column)
    return np.asarray(kept, dtype=np.int64)
