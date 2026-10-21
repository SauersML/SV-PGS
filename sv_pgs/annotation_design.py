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
  linear term is its own column, with a zero entry in the group's penalty (the prior gives the group's penalty null
  space a ridge of its own learned weight, ``scale_mixture_prior``); the penalized part is the basis rotated onto the penalty's eigenvectors, so its
  penalty is diagonal there and is the same function-space penalty, transformed with the basis (a change of basis
  that drops the penalty's null directions changes no function's roughness). x is the annotation standardized over
  its observed values; the knots are its interior quartiles (``_continuous_spline_knots``), where the
  spline space refines toward the full smoothing spline as knots are added, and the penalty, not the knot count,
  sets the smoothness.
- A factor (a legend-coded annotation, or a numeric one whose observed values are all 0 or 1) is a level per
  distinct observed value, with an exchangeable Gaussian prior on the levels about their mean: the level indicators
  times the closed-form orthonormal sum-to-zero (Helmert) basis, penalty identity, so the prior does not depend on
  which level comes first and no dense level-by-level change of basis is formed (the rows are gathered from it).
- A missing value is its own state, built apart from the numeric basis and kept whenever it is identified: the
  numeric columns are 0 on a missing row, a continuous annotation gets a missing-state column, and a factor gets
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

Hierarchical class smooths (``by_class``, off unless the caller names an annotation there; ``BY_CLASS_ANNOTATIONS``
for the run's length-by-type prior). Without them an annotation is one function shared by every variant class: a
deletion's and an insertion's log prior variance bend with log length alike. An annotation named in ``by_class``
keeps that shared smooth and gains a second group, ``<name>:by_class``: each variant class bends it by a deviation
function in the same function space (the same linear term, spline directions and missing state), the deviations
exchangeable about their mean, with their own learned roughness weight and their own ridge on their linear parts
(the group's penalty null space, ``scale_mixture_prior``). Its columns are the annotation's columns times the
orthonormal sum-to-zero (Helmert) basis over the classes whose values of them vary (``_sum_to_zero_rows``, as a
factor's levels are coded), and its penalty is the annotation's on every contrast (P kron I), which is the sum over
classes of each deviation's roughness, so relabelling the classes moves neither the column space nor the prior. This
is the global smoother with group-level smoothers of one shared wiggliness (model "GS" of Pedersen, Miller, Simpson
and Ross, "Hierarchical generalized additive models in ecology: an introduction with mgcv", PeerJ 2019) in its
sum-to-zero form: with independent deviations about a shared curve, their mean is itself a shared curve, and moving
it there leaves exchangeable deviations that sum to zero, an exact reparametrization whose shared curve's prior
variance is the sum of the two parts, which the shared smooth's own learned weight covers. A class whose few members
resolve little of its deviation is pulled to the shared curve by the deviations' learned penalty, and a class whose
values are constant (the SNVs, all at length 1) has no deviation: class centring leaves its curve nothing to move.

The deviations are identified against everything before them. Each class's column of a function is the shared
column on that class's rows, so the shared smooth is their sum and only contrasts can be added, and class centring
removes each class's constant, which its class density carries. A class whose rows support only part of the function
space (copy numbers span longer lengths than insertions) leaves contrast directions that no row resolves or that only
re-express a column before them; a screen in column order would fix those by dropping whichever contrast came last, a
reference class the class order picks. Instead those directions are found as one subspace (``_unresolved_directions``:
the right singular vectors of the deviations' residual on every column before them, at the prior's rank bound) and
held at their smoothest given the rest (``_smoothest_complement``: the complement P-orthogonal to them, which for the
roughness alone is their conditional mean, so the restriction is the exact marginal prior of what the rows resolve, and
which keeps a slope deviation of a class that spans part of the range a straight line, charged no roughness for rows it
never reaches), repeated on what that keeps until the screen's own test passes at the same bound
(``_below_the_bound``), so the final screen keeps the group whole. Both are defined in the contrasts' own coordinates,
where a class relabelling is a rotation that P commutes with, so the restriction picks no class; the group's penalty is
the restriction of its quadratic form.
The deviation groups follow every annotation's own columns, so an interaction enters only for what no main effect
carries, and at their lambda = infinity edges the prior is the one without them: the nested empirical Bayes
(``full_data_fit.fit_full_data``, ``scale_mixture_ep.embed_hyperparameters``) starts there and releases a deviation
group only where its evidence rises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse
from gamfit.basis import bspline_basis, smoothness_penalty

from sv_pgs._typing import F64Array, I64Array, NDArray
from sv_pgs.scale_mixture_ep import AnnotationGroup, _sum_to_zero_rows

_EPSILON = float(np.finfo(np.float64).eps)
# A cubic spline: f'' continuous and piecewise linear, the lowest degree whose roughness integral f''^2 is finite and
# whose penalty null space is exactly the linear functions.
_SPLINE_DEGREE = 3
# The roughness is the integral of the squared second derivative: its null space, the linear functions, is the
# smooth's fixed part (the constant going to the class densities).
_ROUGHNESS_DERIVATIVE = 2


FREQUENCY = "log_genotype_variance"
LOG_SCALE_ANNOTATIONS = ("sv_length",)
"""Annotations that enter the prior on their log scale: a record's length (bp, at least 1) is a learned smooth of log
length, which spans 1 bp to megabases evenly. On raw base pairs the smooth was one
cubic over [1, the longest SV], effectively linear across every SV a cohort has, so a length effect fitted between 50 bp
and 5 kb gave a 2 Mb record hundreds of times its log-variance shift."""
"""The frequency function's annotation: each member's log true-genotype variance log Var(G_j), from the unit contract
(``imputation_reliability.ColumnMeasurement.standardized_terms``)."""
BY_CLASS = ":by_class"
"""The suffix of an annotation's class-deviation group (``annotation_design``'s ``by_class``)."""
BY_CLASS_ANNOTATIONS = ("sv_length",)
"""The annotations the hierarchical length-by-type prior (``stage2_wiring``'s ``length_by_class``) gives class
deviations: a record's log length, whose effect on the prior variance may differ between deletions, duplications,
insertions and the other structural classes."""


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


LD_SCORE = "log_ld_score"
"""The LD function's annotation (``ld_score_annotation``)."""


def ld_score_annotation(ld_scores: np.ndarray) -> dict[str, np.ndarray]:
    """{LD_SCORE: log l_j}: the level of LD around each member as a learned smooth of its log prior variance, its own
    named group (``stage2_wiring``'s ``ld_score_prior``). l_j = sum_i R_ij^2 over the member's Stage 0 block, R the
    training rows' correlations after the covariates (``genotype_statistics.LdGramStore.ld_scores``, which Stage 0
    computes with every block and nothing read before). A joint regression on tag columns spreads one causal signal
    over every column in LD with it, so a column in high LD carries less effect variance each, the premise of the LDAK
    weighting (Speed, Hemani, Johnson and Balding, "Improved heritability estimation from genome-wide SNPs", AJHG 2012),
    and the level of LD predicts per-SNP heritability beyond frequency across complex traits (the baseline-LD model's
    LLD annotation, lower LD with more heritability per SNP: Gazal et al., "Linkage disequilibrium-dependent
    architecture of human complex traits shows action of negative selection", Nat Genet 2017). The smooth's linear
    coefficient c spans the LDAK-like power family Var(b_j) proportional to l_j^c, c = 0 the prior without it, and its
    curvature departs from it where the data say so; the nested empirical Bayes enters it only where the evidence
    rises. The chance part of the sum, about the block's width over the training rows per member (0.4-1.1 at the
    whole-genome fits' ~1e6 columns in ~55 blocks over 1.7e4-4.4e4 rows [real]), is left in, a near-constant shift
    within a block where the sums reach tens. Stage 0 keeps one column per exact tie, so the sum counts a tie group
    once and every member of it takes its column's score. A member whose column the covariates explain (R_jj = 0, no
    LD at all) has no finite log and takes the smooth's missing state."""
    with np.errstate(divide="ignore"):
        return {LD_SCORE: np.log(np.asarray(ld_scores, dtype=np.float64))}


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


def _continuous_spline_knots(standardized_values: NDArray) -> tuple[float, ...]:
    """Distinct quartile knots strictly inside the data range; the rank screen drops any hinge a
    nearby knot makes numerically dependent."""
    candidate_knots = np.quantile(standardized_values, [0.25, 0.5, 0.75])
    minimum_value = float(np.min(standardized_values))
    maximum_value = float(np.max(standardized_values))
    knot_values: list[float] = []
    for knot_value in candidate_knots:
        knot_float = float(knot_value)
        if minimum_value < knot_float < maximum_value and (not knot_values or knot_float != knot_values[-1]):
            knot_values.append(knot_float)
    return tuple(knot_values)


def spline_basis(standardized: F64Array) -> tuple[F64Array, F64Array, F64Array]:
    """(knots, B, S) of a cubic B-spline on ``standardized``'s range with interior knots at the quartiles of its
    distinct values: the basis at those points (n x K, local support: at most 4 nonzero per row) and the exact roughness
    penalty S_ik = integral of B_i''(x) B_k''(x) dx (K x K), both from gamfit (``gamfit.basis``). The knots follow the
    distinct values, not the rows: a point mass (the millions of SNVs at length 1) put every row quartile at the range's
    minimum and left the spline with no interior knot at all."""
    low, high = float(np.min(standardized)), float(np.max(standardized))
    interior = np.asarray(_continuous_spline_knots(np.unique(standardized)), dtype=np.float64)
    knots = np.concatenate([np.full(_SPLINE_DEGREE + 1, low), interior, np.full(_SPLINE_DEGREE + 1, high)])
    basis = np.asarray(bspline_basis(np.clip(standardized, low, high), knots, degree=_SPLINE_DEGREE), dtype=np.float64)
    penalty, _null_basis = smoothness_penalty(knots, degree=_SPLINE_DEGREE, order=_ROUGHNESS_DERIVATIVE)
    return knots, basis, np.asarray(penalty, dtype=np.float64)


def _continuous_candidate(name: str, values: F64Array) -> _Candidate:
    """A continuous annotation's linear term, its penalized spline directions (the basis on the penalty's range
    space, penalty diagonal there) and, where it has missing values, a missing-state column; every numeric column
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


def _on_its_scale(name: str, values: np.ndarray) -> np.ndarray:
    """An annotation's values on the scale its smooth is learned on: log for LOG_SCALE_ANNOTATIONS, as stored otherwise."""
    if name not in LOG_SCALE_ANNOTATIONS:
        return values
    lengths = np.asarray(values, dtype=np.float64)
    observed = np.isfinite(lengths)
    if np.any(lengths[observed] <= 0.0):
        raise ValueError(f"annotation {name!r} is a length, so every observed value must be positive")
    return np.where(observed, np.log(np.where(observed, lengths, 1.0)), np.nan)


def annotation_design(
    annotations: Mapping[str, np.ndarray],
    annotation_legends: Mapping[str, Sequence[str]],
    *,
    class_index: I64Array,
    exclude: Sequence[str] = (),
    by_class: Sequence[str] = (),
    class_labels: Sequence[str] | None = None,
) -> AnnotationDesign:
    """The prior's annotation design over the rows of ``annotations`` (already restricted to the prior's members),
    one named group per annotation in name order, rank-screened after class centring; ``exclude`` names the columns
    that are not annotations (a reliability column is the unit contract's). ``by_class`` names the annotations whose
    smooth each variant class bends (the module docstring's hierarchical class smooths): each adds a group
    ``<name>:by_class`` after every annotation's own, its contrasts named by ``class_labels`` (one per class index;
    the indices where None). A name there that is no annotation, or whose values vary within fewer than two classes,
    adds nothing, so the snv_indel arm's one class leaves the prior as it was."""
    classes = np.asarray(class_index, dtype=np.int64)
    variant_count = classes.shape[0]
    class_count = int(classes.max()) + 1 if variant_count else 0
    candidates = [
        _candidate(name, _on_its_scale(name, annotations[name]), annotation_legends.get(name), variant_count)
        for name in sorted(annotations) if name not in exclude
    ]
    flat = [(position, index) for position, candidate in enumerate(candidates) for index in range(len(candidate.columns))]
    if not flat:
        return AnnotationDesign(design=np.zeros((variant_count, 0)), groups=(), names=())
    design = np.column_stack([candidates[position].columns[index] for position, index in flat])
    counts = np.bincount(classes, minlength=class_count).astype(np.float64)
    centred = design - (_class_sums(design, classes, class_count) / np.maximum(counts, 1.0)[:, None])[classes]
    labels = [str(position) for position in range(class_count)] if class_labels is None else [str(label) for label in class_labels]
    if len(labels) != class_count:
        raise ValueError("class_labels needs one label per class index")
    deviations = [
        deviation for deviation in (
            _class_deviation_candidate(candidate, classes, class_count, labels) for candidate in candidates if candidate.name in by_class
        ) if deviation.columns
    ]
    # The rank bound of the whole candidate design where class deviations join it (``_restricted_deviations``), else
    # the screen's own.
    bound = None
    if deviations:
        deviations, bound = _restricted_deviations(deviations, centred, classes, class_count, counts)
        added = [(len(candidates) + position, index) for position, deviation in enumerate(deviations) for index in range(len(deviation.columns))]
        candidates = [*candidates, *deviations]
        if added:
            block = np.column_stack([candidates[position].columns[index] for position, index in added])
            design = np.column_stack([design, block])
            centred = np.column_stack([centred, block - (_class_sums(block, classes, class_count) / np.maximum(counts, 1.0)[:, None])[classes]])
            flat = [*flat, *added]
    kept = _screened(centred, bound)
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


def smooth_shift_record(
    scale_design: F64Array, groups: Sequence[AnnotationGroup], theta: F64Array, name: str, class_index: I64Array, lengths: F64Array,
    class_labels: Sequence[str],
) -> str:
    """For a fit's log, the learned shift of each class's log prior variance by an annotation's smooth and its class
    deviations (the groups ``name`` and ``name:by_class``), d_j' theta over their columns of the prior's class-centred
    ``scale_design``, averaged over each class's members in each decade [10^k, 10^(k+1)) of ``lengths`` (bp), for the
    classes whose members reach two decades. Each class's shift has mean zero over its members (its class density
    carries the level), so the record is the curve's shape within the class; empty where neither group is in the prior.
    A member with no length (the smooth's missing state) is in no decade. Members are averaged, never counted, and no
    participant enters it."""
    selected = [group.columns for group in groups if group.name in (name, f"{name}{BY_CLASS}")]
    if not selected:
        return ""
    columns = np.concatenate(selected)
    shift = scale_design[:, columns] @ np.asarray(theta, dtype=np.float64)[columns]
    values = np.asarray(lengths, dtype=np.float64)
    observed = np.isfinite(values)
    decade = np.zeros(values.shape[0], dtype=np.int64)
    decade[observed] = np.floor(np.log10(values[observed])).astype(np.int64)
    parts = []
    for position, label in enumerate(class_labels):
        rows = (np.asarray(class_index) == position) & observed
        present = np.unique(decade[rows])
        if present.shape[0] < 2:
            continue
        means = ", ".join(f"1e{int(value)}: {float(shift[rows & (decade == value)].mean()):+.3g}" for value in present)
        parts.append(f"{label} [{means}]")
    return "; ".join(parts)


def _class_deviation_candidate(candidate: _Candidate, classes: I64Array, class_count: int, labels: Sequence[str]) -> _Candidate:
    """The class deviations of ``candidate``'s function, unrestricted: each of its columns on the rows of every class
    whose values of its columns vary, times the orthonormal sum-to-zero basis over those classes (contrast m sets class
    m + 1 against the ones before it, ``_sum_to_zero_rows``), column (k, m) at k (C - 1) + m, and the function's
    penalty on every contrast, P kron I_(C-1): with deviations D = E H' (H the basis), sum_c d_c' P d_c = tr(E' P E)
    since H'H = I. Empty where fewer than two classes vary: one class's curve is the shared curve itself."""
    name = f"{candidate.name}{BY_CLASS}"
    if not candidate.columns:
        return _Candidate(name, [], [], np.zeros((0, 0)))
    columns = np.column_stack(candidate.columns)
    varying = [
        position for position in range(class_count)
        if np.any(classes == position) and bool(np.any(np.ptp(columns[classes == position], axis=0) > 0.0))
    ]
    if len(varying) < 2:
        return _Candidate(name, [], [], np.zeros((0, 0)))
    level = np.full(class_count, -1, dtype=np.int64)
    level[varying] = np.arange(len(varying))
    member = level[classes]
    contrasts = np.where((member >= 0)[:, None], _sum_to_zero_rows(np.maximum(member, 0), len(varying)), 0.0)
    suffixes = [column_name[len(candidate.name) + 1 :] for column_name in candidate.names]
    return _Candidate(
        name,
        [np.ascontiguousarray(columns[:, function] * contrasts[:, contrast]) for function in range(columns.shape[1]) for contrast in range(contrasts.shape[1])],
        [f"{name}({labels[varying[contrast + 1]]}):{suffixes[function]}" for function in range(columns.shape[1]) for contrast in range(contrasts.shape[1])],
        np.kron(candidate.penalty, np.eye(contrasts.shape[1])),
    )


def _restricted_deviations(
    deviations: Sequence[_Candidate], centred: F64Array, classes: I64Array, class_count: int, counts: F64Array,
) -> tuple[list[_Candidate], float]:
    """Each deviation group restricted to a complement of its contrast directions that add nothing resolvable to every
    column before it, the annotations' own (as the screen keeps them) and the earlier deviation groups'
    (``_unresolved_directions``, at the prior's rank bound over the whole candidate design), the complement that holds
    them at their smoothest (``_smoothest_complement``); unchanged, names and all, where every direction adds something.
    A restricted group's columns are its contrasts on an orthonormal basis W of that complement, named by position, and
    its penalty W' P W, the restriction of the quadratic form; a group with nothing resolvable is dropped. With the
    bound, which the final screen then takes (``_screened``).

    The complement is not the resolved directions' own span: it holds the unresolved part at its smoothest, so a kept
    direction can carry a little of an unresolved one, and the design with it can fall to the bound where the residual
    alone stood above it. The restriction is therefore repeated on what it kept until the residual and then the
    screen's own test pass (``_below_the_bound``, the Gram of every column before the group with it above the bound);
    each pass removes at least one direction, so it ends. The design then passes the final screen whole: its main
    columns are screened at the same bound as here, and every subset of the kept deviations passes by interlacing, so
    no column order picks a class."""
    blocks = [np.column_stack(deviation.columns) for deviation in deviations]
    centred_blocks = [block - (_class_sums(block, classes, class_count) / np.maximum(counts, 1.0)[:, None])[classes] for block in blocks]
    stacked = np.column_stack(centred_blocks)
    cross = centred.T @ stacked
    gram = np.block([[centred.T @ centred, cross], [cross.T, stacked.T @ stacked]])
    bound = _EPSILON * centred.shape[0] * max(float(np.linalg.eigvalsh(gram)[-1]), 1.0)
    previous = centred[:, _screened(centred, bound)]
    restricted: list[_Candidate] = []
    for deviation, block, centred_block in zip(deviations, blocks, centred_blocks):
        directions = np.eye(block.shape[1])
        while directions.shape[1]:
            candidate = centred_block @ directions
            unresolved = _unresolved_directions(previous, candidate, bound)
            if not unresolved.shape[1]:
                unresolved = _below_the_bound(previous, candidate, bound)
            if not unresolved.shape[1]:
                break
            directions = directions @ _smoothest_complement(unresolved, directions.T @ deviation.penalty @ directions)
        if directions.shape[1] < block.shape[1]:
            block, centred_block = block @ directions, centred_block @ directions
            deviation = _Candidate(
                deviation.name,
                [np.ascontiguousarray(block[:, position]) for position in range(block.shape[1])],
                [f"{deviation.name}:direction{position + 1}" for position in range(block.shape[1])],
                directions.T @ deviation.penalty @ directions,
            )
        if deviation.columns:
            restricted.append(deviation)
            previous = np.column_stack([previous, centred_block])
    return restricted, bound


def _unresolved_directions(previous: F64Array, block: F64Array, bound: float) -> F64Array:
    """V, an orthonormal basis (columns) of the coefficient directions of the class-centred ``block`` whose columns add
    at most ``bound`` to the span of ``previous``: the residual's right singular vectors whose squared singular value,
    the residual Gram's eigenvalue, is not above it (none: zero columns). A rotation of the block's coordinates (a
    relabelling of the classes, for contrasts) rotates V with it. The singular vectors of the residual itself, not the
    eigenvectors of its Gram, whose squared condition number left the subspace of a class's barely resolved directions
    (a narrow class's cubic, at 1e-4 of the largest singular value) to 1e-7."""
    residual = block
    if previous.shape[1]:
        basis, _triangle = np.linalg.qr(previous)
        residual = block - basis @ (basis.T @ block)
    # The reduced factorization's right factor is square only where the rows are at least the columns.
    _left, singular, right = np.linalg.svd(residual, full_matrices=residual.shape[0] < residual.shape[1])
    resolved = np.zeros(right.shape[0], dtype=bool)
    resolved[: singular.shape[0]] = np.square(singular) > bound
    return right[~resolved].T


def _below_the_bound(previous: F64Array, block: F64Array, bound: float) -> F64Array:
    """V, an orthonormal basis of the coefficient directions of the class-centred ``block`` along which the Gram of
    ``previous`` with the block reaches ``bound`` or below, the screen's own test, where ``previous``'s Gram is above it
    (the screen kept it at that bound): with G - bI positive definite on ``previous``, it is positive definite on both
    exactly where the Schur complement S = R'R - bI - Z' diag(b / (s_i^2 - b)) Z is, R the block's residual on
    ``previous``, s_i the singular values of its triangular factor and Z the block's coordinates on their left vectors
    (so S is block'block - bI less the projected part's (A - bI)^-1 term); V spans the eigenvectors with eigenvalue at or
    below zero. Read after ``_unresolved_directions``, which has the singular vectors' accuracy for the plainly
    unresolved directions, it finds only those at the bound's edge."""
    if previous.shape[1]:
        basis, triangle = np.linalg.qr(previous)
        projected = basis.T @ block
        residual = block - basis @ projected
        left, singular, _right = np.linalg.svd(triangle)
        coordinates = left.T @ projected
        complement = residual.T @ residual - coordinates.T @ ((bound / (np.square(singular) - bound))[:, None] * coordinates)
    else:
        complement = block.T @ block
    eigenvalues, eigenvectors = np.linalg.eigh(complement - bound * np.eye(block.shape[1]))
    return eigenvectors[:, eigenvalues <= 0.0]


def _smoothest_complement(unresolved: F64Array, penalty: F64Array) -> F64Array:
    """W, an orthonormal basis of the deviation directions P-orthogonal to the ``unresolved`` ones, and orthogonal to
    those of them P does not see. Restricting to it holds each unresolved direction at its smoothest given the resolved
    ones (the minimizer of the roughness over the unresolved part: for P alone, their conditional mean, so the
    restriction is the exact marginal prior of what the rows resolve), and the ones the roughness cannot see at zero.
    A deviation linear on its class's rows stays linear, charged no roughness for the rows it never reaches; the
    Euclidean complement bent it there, which the roughness weight then penalized. Both constraint sets are
    independent (P's range is orthogonal to its null space) and their count is the unresolved dimension, so W has the
    rest, and no unresolved direction lies in its span (one would have u'Pu = 0, so lie in P's null space, where W is
    orthogonal to it). The null test is the prior's own (``scale_mixture_prior``'s group penalty eigenvalues), relative
    to the penalty's largest eigenvalue, never to the unresolved directions' own: where every one of them lies in P's
    null space (a missing state's contrast) their roughness is roundoff, which a test relative to itself took for
    roughness, constraining a spline direction in place of the missing state."""
    seen = unresolved.T @ penalty @ unresolved
    eigenvalues, eigenvectors = np.linalg.eigh(seen)
    scale = max(float(np.linalg.eigvalsh(penalty)[-1]), np.finfo(np.float64).tiny)
    rough = eigenvalues > _EPSILON * penalty.shape[0] * scale
    constraints = np.column_stack([penalty @ unresolved @ eigenvectors[:, rough], unresolved @ eigenvectors[:, ~rough]])
    basis, _triangle = np.linalg.qr(constraints, mode="complete")
    return basis[:, constraints.shape[1] :]


def _screened(centred: F64Array, bound: float | None = None) -> I64Array:
    """The columns of the class-centred design kept in order: a column enters when the Gram of the kept columns with
    it passes the prior's exact test (``scale_mixture_prior``: its least eigenvalue above eps p max(largest, 1)),
    with the largest eigenvalue of the whole candidate Gram in the bound, which is at least that of every subset's, so
    the kept design passes the prior's test. A column is dropped only for what it adds to the ones before it: a test
    run once on the whole kept set and failed by removing the latest columns dropped whole later annotations for an
    earlier one's near-dependence (every log_tss_distance column on ENSG00000187605.16 [real], for the SV length
    smooths' few rows). A given ``bound`` (the class deviations', ``_restricted_deviations``: the whole candidate
    design's, at least the kept design's) stands in for its own."""
    count, size = centred.shape
    gram = centred.T @ centred
    if bound is None:
        largest = float(np.linalg.eigvalsh(gram)[-1]) if size else 0.0
        bound = _EPSILON * count * max(largest, 1.0)
    kept: list[int] = []
    for column in range(size):
        trial = [*kept, column]
        if float(np.linalg.eigvalsh(gram[np.ix_(trial, trial)])[0]) > bound:
            kept.append(column)
    return np.asarray(kept, dtype=np.int64)
