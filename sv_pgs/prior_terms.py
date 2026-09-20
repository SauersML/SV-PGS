"""The prior's scale-model terms: what the engine's ``annotation_design`` and offset are made of.

The engine (``scale_mixture_ep``) takes a per-variant offset o_j with coefficient
exactly 1, an annotation design d_j, and groups of design columns that share one
learned penalty weight. This module builds those from measured quantities, so that
each term of the scale model

    log u_j = o_j + d_j' theta,   o_j = log r2_j + log H^loc_j,

is structurally present and switched off by its own weight when the data do not
support it. Nothing here is a choice: a weight at infinity leaves exactly the
term's penalty null space, which is the simpler model it generalizes, and a term
whose null space is empty disappears.

Offset (``measurement_offset``). Both parts carry coefficient 1 by derivation, so
neither is fitted:
- log r2_j maps the prior on the true genotype's effect to the stored column
  (scale_model.md §1);
- log H^loc_j is the per-allele to per-SD unit conversion, with H^loc the column's
  genotype variance in the analysis sample (§5). Var(D*)/r2 equals Var(G) only for
  the recalibrated column, where Cov(G, D*) = Var(D*); a raw draw-like DS has
  Var(DS) near Var(G), so dividing it by r2 overstates H^loc by about 1/r2 and
  turns the offset's effective coefficient into -S. ``local_heterozygosity``
  therefore takes a ``Recalibration``, the output of ``linear_recalibration``, and
  never a variance; ``heterozygosity(p)`` is the 2p(1-p) route.

Frequency (``frequency_design``). Selection at drift equilibrium makes the per-SD
prior variance vary with the frequency at which that equilibrium holds, H^sel,
which is ancestry-resolved: p^sel = w p_AFR + (1-w) p_pooled with w learned
(novel-evoprior.md §3, scale_model.md §5). The term is a smooth in log H^sel whose
penalty is the integrated squared second derivative, so its null space is
{1, log H}: at an infinite weight the term is exactly the power law H^(1+S) that
BayesS and LDAK fit, with S no longer a hand-set constant but the null direction's
profiled coefficient, and at a finite weight the plateau at low frequency and the
saturation ceiling at common frequency are learned rather than assumed. The
saturating map v = u H s/(1 + kappa H s) itself, which changes the density's shape
and not only its scale, needs the engine to carry kappa and n; this term is its
learned residual, which novel-evoprior.md §8 keeps in that model too.

SV context (``sv_context_design``). One learned kernel f_j = sum_k H_k w(log d_jk)
over the SV alleles of other bubbles, from ``store_converter.sv_kernel_features``
(scale_model.md §8). Classes share one kernel plus deviations, w_c = w_0 + delta_c,
with sum-to-zero deviations, so an infinite deviation weight pools the classes
exactly and an infinite roughness weight leaves the kernel's null space, a length
tensor of {1, log dist} x {1, log len}. Overlap is its own column per class: being
nested inside an SV allele is a different relation from being near one.

Shape (``shape_functionals``, ``shape_score``). Every term above moves the
density's location. Whether an annotation changes its shape (say, raising the tail
without moving typical effects) is decided by the score test of scale_model.md §6,
never assumed: the functionals are the degree-2 and degree-3 polynomials in t
orthogonalized against {1, t} under the fitted weights, so they are exactly the
directions a scale term cannot reach. The nested generalization, log g(t | d) =
log g_0(t - m(d)) + sum_k d_k delta_k(t) with each delta_k penalized and
constrained to that complement, returns scale-only at an infinite weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sv_pgs._typing import F64Array, I64Array
from sv_pgs.dosage_store import MAXIMUM_DOSAGE_MILLI
from sv_pgs.store_converter import Recalibration, SvKernelFeatures, _uniform_cubic_weights

_EPSILON = float(np.finfo(np.float64).eps)
# A cubic B-spline's second derivative is linear, so the penalty's integrand is quadratic on each
# knot interval and two Gauss-Legendre nodes integrate it exactly.
_PENALTY_NODES = 2
# A uniform cubic B-spline is nonzero on four knot intervals.
_CUBIC_SUPPORT = 4


@dataclass(frozen=True)
class PenaltyGroup:
    """One learned weight: the term's columns it acts on and the square-root factor R of its penalty.

    The penalty value is ||R c||^2 over those columns. A group whose factor has fewer
    rows than columns leaves its null space unpenalized; the engine profiles those
    directions as fixed effects.
    """

    name: str
    columns: I64Array
    factor: F64Array

    @property
    def matrix(self) -> F64Array:
        return self.factor.T @ self.factor


@dataclass(frozen=True)
class TermDesign:
    """One scale-model term: its design columns and the learned weights over them.

    The engine gives each design column exactly one weight, so the groups partition
    the term's columns. ``names`` labels the columns. The directions no group
    penalizes are the model the term reduces to when every weight is infinite; a
    term whose groups are all full rank is switched off exactly at that limit.
    """

    design: F64Array
    groups: tuple[PenaltyGroup, ...]
    names: tuple[str, ...]
    knots: F64Array | None = None
    transform: F64Array | None = None

    def __post_init__(self) -> None:
        if self.design.ndim != 2:
            raise ValueError("a term's design must be two-dimensional")
        width = self.design.shape[1]
        if len(self.names) != width:
            raise ValueError("a term needs one name per design column")
        covered = np.concatenate([group.columns for group in self.groups]) if self.groups else np.zeros(0, np.int64)
        if not np.array_equal(np.sort(covered), np.arange(width)):
            raise ValueError("a term's groups must cover every design column exactly once")
        for group in self.groups:
            if group.factor.ndim != 2 or group.factor.shape[1] != group.columns.shape[0]:
                raise ValueError("every penalty factor must have one column per column of its group")

    def penalty(self) -> F64Array:
        """The term's total penalty matrix at unit weights, over all its columns."""
        width = self.design.shape[1]
        total = np.zeros((width, width))
        for group in self.groups:
            total[np.ix_(group.columns, group.columns)] += group.matrix
        return total

    @property
    def null_basis(self) -> F64Array:
        """An orthonormal basis of the directions no weight penalizes: the infinite-weight model."""
        width = self.design.shape[1]
        rows = [np.zeros((0, width))]
        for group in self.groups:
            embedded = np.zeros((group.factor.shape[0], width))
            embedded[:, group.columns] = group.factor
            rows.append(embedded)
        stacked = np.vstack(rows)
        _left, singular, right = np.linalg.svd(stacked, full_matrices=True)
        padded = np.zeros(right.shape[0])
        padded[: singular.shape[0]] = singular
        tolerance = _EPSILON * max(stacked.shape) * (float(padded.max()) if padded.size else 0.0)
        return right[padded <= tolerance].T


def _whole(name: str, factor: F64Array) -> PenaltyGroup:
    """A group over every column of a term."""
    return PenaltyGroup(name=name, columns=np.arange(factor.shape[1], dtype=np.int64), factor=factor)


def _empty_term(variant_count: int) -> TermDesign:
    """A term with no columns: what a variable that never varies within a class contributes."""
    return TermDesign(design=np.zeros((variant_count, 0)), groups=(), names=())


def local_heterozygosity(*, recalibrated: Recalibration, reliability: F64Array) -> F64Array:
    """H^loc = Var(D*)/r2, the genotype variance the stored column measures (scale_model.md §5).

    ``recalibrated`` is a block from ``linear_recalibration``, records by analysis
    samples; ``reliability`` is each record's r2. The identity Var(D*) = r2 Var(G)
    holds because the recalibration makes Cov(G, D*) = Var(D*), so it holds for
    D* only: a raw DS block is refused by type rather than silently overstating
    H^loc by about 1/r2 (review-mathbugs F6).
    """
    if not isinstance(recalibrated, Recalibration):
        raise TypeError("local_heterozygosity needs the recalibrated block from linear_recalibration, not a raw dosage or variance")
    milli = np.asarray(recalibrated.dosage_milli)
    r_squared = np.asarray(reliability, dtype=np.float64)
    if milli.ndim != 2 or r_squared.shape != (milli.shape[0],):
        raise ValueError("the recalibrated block must be (records, samples) with one reliability per record")
    if np.any(~np.isfinite(r_squared)) or np.any(r_squared <= 0.0) or np.any(r_squared > 1.0):
        raise ValueError("every reliability must be finite and lie in (0, 1]")
    dosage = milli.astype(np.float64) / (MAXIMUM_DOSAGE_MILLI / 2.0)
    variance = dosage.var(axis=1)
    if np.any(variance <= 0.0):
        raise ValueError("every recalibrated column must vary over the analysis samples")
    return variance / r_squared


def measurement_offset(*, reliability: F64Array, local_variance: F64Array) -> F64Array:
    """o_j = log r2_j + log H^loc_j, the part of log u_j whose coefficient is 1 by derivation."""
    r_squared = np.asarray(reliability, dtype=np.float64)
    heterozygosity = np.asarray(local_variance, dtype=np.float64)
    if r_squared.shape != heterozygosity.shape:
        raise ValueError("the reliabilities and local variances must have one value per variant")
    if np.any(~np.isfinite(r_squared)) or np.any(r_squared <= 0.0) or np.any(r_squared > 1.0):
        raise ValueError("every reliability must be finite and lie in (0, 1]")
    if np.any(~np.isfinite(heterozygosity)) or np.any(heterozygosity <= 0.0):
        raise ValueError("every local genotype variance must be finite and positive")
    return np.log(r_squared) + np.log(heterozygosity)


def ancestry_mixed_frequency(*, group_frequencies: F64Array, weights: F64Array) -> F64Array:
    """p^sel = sum_g w_g p_g, the frequency at which selection-drift equilibrium is read.

    ``weights`` are learned by profiling the evidence over the simplex; this
    function is the design at one point of that profile, so the search never
    enters the engine's linear coefficients (scale_model.md §5).
    """
    frequencies = np.asarray(group_frequencies, dtype=np.float64)
    mixture = np.asarray(weights, dtype=np.float64)
    if frequencies.ndim != 2 or mixture.shape != (frequencies.shape[1],):
        raise ValueError("group frequencies must be (variants, groups) with one weight per group")
    if np.any(~np.isfinite(frequencies)) or np.any(frequencies < 0.0) or np.any(frequencies > 1.0):
        raise ValueError("every group frequency must be finite and lie in [0, 1]")
    if np.any(~np.isfinite(mixture)) or np.any(mixture < 0.0):
        raise ValueError("the mixture weights must be finite and non-negative")
    total = float(mixture.sum())
    if total <= 0.0:
        raise ValueError("the mixture weights must not all be zero")
    return frequencies @ (mixture / total)


def heterozygosity(frequency: F64Array) -> F64Array:
    """H = 2p(1-p), a biallelic locus's genotype variance in per-allele units."""
    values = np.asarray(frequency, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("every frequency must be finite and lie in [0, 1]")
    return 2.0 * values * (1.0 - values)


def _knots(values: F64Array, spacing: float) -> F64Array:
    """Uniform cubic B-spline knots covering the values, with the three extra knots each end needs."""
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("the knot spacing must be finite and positive")
    lowest, highest = float(values.min()), float(values.max())
    interval_count = max(1, int(np.ceil((highest - lowest) / spacing)))
    interior = lowest + spacing * np.arange(interval_count + 1)
    below = lowest - spacing * np.arange(_CUBIC_SUPPORT - 1, 0, -1)
    above = interior[-1] + spacing * np.arange(1, _CUBIC_SUPPORT)
    return np.concatenate([below, interior, above])


def _cubic_weights(offset: F64Array) -> F64Array:
    """The four nonzero uniform cubic B-splines at offset u in [0, 1] of their knot interval.

    The store's own segment polynomials (``store_converter._uniform_cubic_weights``),
    so a smooth and the SV-context features are one basis.
    """
    return np.stack(_uniform_cubic_weights(offset), axis=-1)


def _cubic_curvatures(offset: F64Array, spacing: float) -> F64Array:
    """Their second derivatives at the same offsets: the exact d2/dx2 of each local cubic.

    The four cubics of ``_cubic_weights`` differentiate twice to (1-u, 3u-2, 1-3u, u)/h^2,
    which sum to zero, as a partition of unity must.
    """
    return np.stack(
        [1.0 - offset, 3.0 * offset - 2.0, 1.0 - 3.0 * offset, offset], axis=-1
    ) / (spacing * spacing)


def _cubic_design(values: F64Array, knots: F64Array) -> F64Array:
    """The uniform cubic B-spline basis at ``values``: (n, len(knots) - 4).

    Interval i of the covered range [knots[3], knots[-4]] carries the four
    functions i..i+3, so the basis is a partition of unity there and the last
    interval's functions reach the final column exactly.
    """
    spacing = float(knots[1] - knots[0])
    basis_count = knots.shape[0] - _CUBIC_SUPPORT
    interval_count = basis_count - (_CUBIC_SUPPORT - 1)
    scaled = (np.asarray(values, dtype=np.float64) - knots[_CUBIC_SUPPORT - 1]) / spacing
    interval = np.clip(np.floor(scaled).astype(np.int64), 0, interval_count - 1)
    offset = scaled - interval
    design = np.zeros((scaled.shape[0], basis_count))
    rows = np.repeat(np.arange(scaled.shape[0]), _CUBIC_SUPPORT)
    columns = (interval[:, None] + np.arange(_CUBIC_SUPPORT)[None, :]).ravel()
    design[rows, columns] = _cubic_weights(offset).ravel()
    return design


def _second_derivative_penalty(knots: F64Array) -> F64Array:
    """The exact integral of the squared second derivative over the covered range.

    On a knot interval the second derivative is linear, so the integrand is
    quadratic and two Gauss-Legendre nodes integrate it exactly.
    """
    spacing = float(knots[1] - knots[0])
    basis_count = knots.shape[0] - _CUBIC_SUPPORT
    interval_count = basis_count - (_CUBIC_SUPPORT - 1)
    offsets, quadrature_weights = np.polynomial.legendre.leggauss(_PENALTY_NODES)
    offsets = 0.5 * (offsets + 1.0)
    quadrature_weights = 0.5 * spacing * quadrature_weights
    curvatures = _cubic_curvatures(offsets, spacing)
    penalty = np.zeros((basis_count, basis_count))
    for interval in range(interval_count):
        columns = interval + np.arange(_CUBIC_SUPPORT)
        for node, node_weight in zip(curvatures, quadrature_weights):
            local = np.zeros(basis_count)
            local[columns] = node
            penalty += node_weight * np.outer(local, local)
    return penalty


def _class_centred(design: F64Array, class_index: I64Array | None) -> F64Array:
    """The design as the engine will see it: each column centred within its class."""
    if class_index is None:
        return design - design.mean(axis=0, keepdims=True)
    classes = np.asarray(class_index, dtype=np.int64)
    centred = design.copy()
    for value in np.unique(classes):
        rows = classes == value
        centred[rows] -= centred[rows].mean(axis=0, keepdims=True)
    return centred


def _rank_floor(centred: F64Array, largest_square: float) -> float:
    """The smallest squared singular value the engine's rank test accepts for this design."""
    return _EPSILON * centred.shape[0] * max(largest_square, 1.0)


def _greville(knots: F64Array) -> F64Array:
    """The abscissae whose B-spline combination is the identity, sum_m xi_m B_m(x) = x."""
    degree = _CUBIC_SUPPORT - 1
    windows = np.lib.stride_tricks.sliding_window_view(knots[1:-1], degree)
    return windows.mean(axis=1)[: knots.shape[0] - _CUBIC_SUPPORT]


def _identified_transform(basis: F64Array, knots: F64Array, class_index: I64Array | None) -> F64Array:
    """A basis of the directions this term keeps: the penalty's linear null direction, then the rest.

    The penalty annihilates exactly the constant and the linear function, whose
    coefficient vectors are 1 and the Greville abscissae. The constant is the class
    level, so it is dropped; the linear direction is kept exactly, so that at an
    infinite weight the term is exactly a power law rather than nearly one. The
    remaining directions are kept where the class levels do not already carry them,
    which is the rank the engine requires of the centred design.
    """
    size = basis.shape[1]
    constant = np.ones(size) / np.sqrt(size)
    greville = _greville(knots)
    linear = greville - constant * (constant @ greville)
    linear_norm = float(np.linalg.norm(linear))
    if linear_norm <= _EPSILON * size * float(np.linalg.norm(greville)):
        raise ValueError("the knots carry no linear direction: the spacing is degenerate")
    linear /= linear_norm
    complement = np.eye(size) - np.outer(constant, constant) - np.outer(linear, linear)
    vectors, singular, _rows = np.linalg.svd(complement)
    range_directions = vectors[:, singular > _EPSILON * size * float(singular[0])]

    # The engine tests the whole centred design's rank, so the kept directions are made
    # orthogonal in that design: the linear one first, then the residual's own directions.
    centred_linear = _class_centred(basis @ linear[:, None], class_index)[:, 0]
    centred_range = _class_centred(basis @ range_directions, class_index)
    linear_square = float(centred_linear @ centred_linear)
    projection = (centred_linear @ centred_range) / linear_square if linear_square > 0.0 else np.zeros(centred_range.shape[1])
    residual = centred_range - np.outer(centred_linear, projection)
    _left, residual_singular, residual_rows = np.linalg.svd(residual, full_matrices=False)
    largest_square = max(linear_square, float(residual_singular[0] ** 2) if residual_singular.size else 0.0)
    floor = _rank_floor(centred_range, largest_square)
    if linear_square <= floor:
        raise ValueError("the smooth carries no direction the class levels do not: it cannot be identified")
    kept = residual_singular**2 > floor
    rotated = (range_directions - np.outer(linear, projection)) @ residual_rows[kept].T
    return np.column_stack([linear, rotated])


def _factor(penalty: F64Array) -> F64Array:
    """A square-root factor R of a symmetric positive-semidefinite penalty, with R'R = it."""
    eigenvalues, eigenvectors = np.linalg.eigh(penalty)
    largest = float(eigenvalues[-1]) if eigenvalues.size else 0.0
    kept = eigenvalues > _EPSILON * eigenvalues.shape[0] * max(largest, np.finfo(np.float64).tiny)
    return np.sqrt(eigenvalues[kept])[:, None] * eigenvectors[:, kept].T


def smooth_term(values: F64Array, spacing: float, *, name: str, class_index: I64Array | None = None) -> TermDesign:
    """A penalized cubic-spline smooth in ``values``, identified against the class levels.

    The penalty is the exact integrated squared second derivative, so an infinite
    weight leaves exactly the linear direction and the term becomes a power law in
    the exponentiated variable. ``class_index`` is the class each variant belongs
    to, which the engine centres the design within; passing it keeps the term
    identified against exactly those levels.
    """
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 1:
        raise ValueError("a smooth takes one value per variant")
    if np.any(~np.isfinite(points)):
        raise ValueError("every value of a smooth must be finite")
    knots = _knots(points, spacing)
    basis = _cubic_design(points, knots)
    transform = _identified_transform(basis, knots, class_index)
    design = basis @ transform
    penalty = transform.T @ _second_derivative_penalty(knots) @ transform
    names = tuple(f"{name}::basis_{position}" for position in range(design.shape[1]))
    return TermDesign(design=design, groups=(_whole(f"{name} roughness", _factor(penalty)),), names=names, knots=knots, transform=transform)


def switchable_smooth(values: F64Array, spacing: float, *, name: str, class_index: I64Array | None = None) -> TermDesign:
    """A smooth in ``values`` that its own weights switch off exactly.

    The same cubic-spline basis as ``smooth_term``, split into two groups:
    - the linear direction, alone, with a ridge weight;
    - the curvature directions, with the exact integrated squared second derivative,
      which is full rank on them because its null space is the linear functions.
    Every direction is penalized, so at two infinite weights the term is exactly 0
    and the model it extends is recovered unchanged. At an infinite curvature weight
    alone it is exactly linear in ``values``. A variable that never varies within a
    class carries nothing the class levels do not, and gives an empty term.
    """
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 1:
        raise ValueError("a smooth takes one value per variant")
    if np.any(~np.isfinite(points)):
        raise ValueError("every value of a smooth must be finite")
    centred = _class_centred(points[:, None], class_index)[:, 0]
    if not float(centred @ centred) > _rank_floor(centred[:, None], float(points @ points)):
        return _empty_term(points.shape[0])
    knots = _knots(points, spacing)
    basis = _cubic_design(points, knots)
    transform = _identified_transform(basis, knots, class_index)
    design = basis @ transform
    penalty = transform.T @ _second_derivative_penalty(knots) @ transform
    curvature = np.arange(1, design.shape[1], dtype=np.int64)
    groups = [PenaltyGroup(name=f"{name} linear", columns=np.array([0], dtype=np.int64), factor=np.eye(1))]
    if curvature.size:
        groups.append(PenaltyGroup(name=f"{name} curvature", columns=curvature, factor=_factor(penalty[np.ix_(curvature, curvature)])))
    names = (f"{name}::linear",) + tuple(f"{name}::curvature_{position}" for position in range(curvature.size))
    return TermDesign(design=design, groups=tuple(groups), names=names, knots=knots, transform=transform)


def frequency_design(*, selection_frequency: F64Array, spacing: float, class_index: I64Array | None = None) -> TermDesign:
    """The frequency term: a smooth in log H^sel whose infinite-weight limit is H^(1+S).

    ``selection_frequency`` is p^sel from ``ancestry_mixed_frequency``; variants
    with no variation in the selection frequency carry no information about it and
    are rejected rather than floored.
    """
    frequencies = np.asarray(selection_frequency, dtype=np.float64)
    selection_heterozygosity = heterozygosity(frequencies)
    if np.any(selection_heterozygosity <= 0.0):
        raise ValueError("every selection frequency must be polymorphic: H = 2p(1-p) > 0")
    return smooth_term(np.log(selection_heterozygosity), spacing, name="frequency", class_index=class_index)


def smooth_values(term: TermDesign, values: F64Array) -> F64Array:
    """The same fitted smooth evaluated at new points, on the term's own knots and transform."""
    if term.knots is None or term.transform is None:
        raise ValueError("only a spline smooth can be evaluated at new points")
    return _cubic_design(np.asarray(values, dtype=np.float64), term.knots) @ term.transform


def _class_contrasts(class_count: int) -> F64Array:
    """Orthonormal sum-to-zero contrasts over classes: the deviations delta_c."""
    centred = np.eye(class_count) - np.full((class_count, class_count), 1.0 / class_count)
    basis, _singular, _rows = np.linalg.svd(centred)
    return basis[:, : class_count - 1]


def sv_context_design(features: SvKernelFeatures) -> TermDesign:
    """The SV-context kernel: one pooled kernel plus sum-to-zero class deviations.

    The pooled columns sum a class's features over classes, so their coefficient is
    the kernel every class shares; the deviation columns are the same features
    against orthonormal sum-to-zero contrasts. Two weights are learned: the
    kernel's roughness, whose null space is the {1, log dist} x {1, log len}
    tensor, and the deviations' size, whose infinite limit pools the classes
    exactly.
    """
    overlap = np.asarray(features.overlap, dtype=np.float64)
    distance = np.asarray(features.distance, dtype=np.float64)
    if overlap.ndim != 2 or distance.ndim != overlap.ndim + 2 or distance.shape[:2] != overlap.shape:
        raise ValueError("the kernel features must be (variants, classes) and (variants, classes, bases, length)")
    variant_count, class_count = overlap.shape
    basis_count, length_count = distance.shape[2], distance.shape[3]
    contrasts = _class_contrasts(class_count) if class_count > 1 else np.zeros((1, 0))
    flat = distance.reshape(variant_count, class_count, basis_count * length_count)
    pooled_distance = flat.sum(axis=1)
    pooled_overlap = overlap.sum(axis=1, keepdims=True)
    deviation_distance = np.einsum("vcf,cd->vdf", flat, contrasts).reshape(variant_count, -1)
    deviation_overlap = overlap @ contrasts
    design = np.hstack([pooled_distance, pooled_overlap, deviation_distance, deviation_overlap])
    pooled_size = pooled_distance.shape[1] + 1
    deviation_size = deviation_distance.shape[1] + deviation_overlap.shape[1]
    # Roughness in the distance direction: the exact integral of the kernel's squared second
    # derivative in x = log(1 + gap), per length term, the same measure as the frequency smooth's.
    # ``sv_kernel_features`` puts basis function m on the knots h (m - 3, ..., m + 1), covering
    # [0, (bases - 3) h]: the convention of ``_cubic_design``, so ``_second_derivative_penalty``
    # is exact on these knots (review-mathbugs F5). Its null space is {1, x} per length term.
    knots = features.spacing * (np.arange(basis_count + _CUBIC_SUPPORT, dtype=np.float64) - (_CUBIC_SUPPORT - 1))
    roughness_block = _factor(np.kron(_second_derivative_penalty(knots), np.eye(length_count)))
    distance_width = pooled_distance.shape[1]
    names = (
        tuple(f"sv_kernel::pooled::basis_{position // length_count}::length_{position % length_count}" for position in range(distance_width))
        + ("sv_kernel::pooled::overlap",)
        + tuple(f"sv_kernel::deviation_{position}" for position in range(deviation_size))
    )
    groups = [
        PenaltyGroup(name="sv kernel roughness", columns=np.arange(distance_width, dtype=np.int64), factor=roughness_block),
        # Nesting inside an SV allele is its own relation with its own weight, so it can be switched off.
        PenaltyGroup(name="sv kernel overlap", columns=np.array([distance_width], dtype=np.int64), factor=np.eye(1)),
    ]
    if deviation_size:
        groups.append(PenaltyGroup(
            name="sv kernel class deviations",
            columns=pooled_size + np.arange(deviation_size, dtype=np.int64),
            factor=np.eye(deviation_size),
        ))
    return TermDesign(design=design, groups=tuple(groups), names=names)


@dataclass(frozen=True)
class StackedDesign:
    """Terms laid side by side in the form the engine takes, and the way back to each term's own columns.

    ``groups`` is one (columns, penalty matrix) per learned weight, partitioning the
    design's columns as ``AnnotationGroup`` requires. ``loading`` maps the stacked
    coefficients to the terms' original columns: original = loading @ stacked.
    """

    design: F64Array
    groups: tuple[tuple[I64Array, F64Array], ...]
    group_names: tuple[str, ...]
    names: tuple[str, ...]
    loading: F64Array


def _split_by_penalty(factor: F64Array) -> tuple[F64Array, F64Array]:
    """Orthonormal bases of a factor's null space and of its complement, in the group's coefficients."""
    width = factor.shape[1]
    if factor.shape[0] == 0:
        return np.eye(width), np.zeros((width, 0))
    _left, singular, right = np.linalg.svd(factor, full_matrices=True)
    padded = np.zeros(width)
    padded[: singular.shape[0]] = singular
    penalized = padded > _EPSILON * max(factor.shape) * float(padded.max())
    return right[~penalized].T, right[penalized].T


def stack_terms(*terms: TermDesign, class_index: I64Array | None = None) -> StackedDesign:
    """Lay terms side by side, keeping in order the directions the engine can identify.

    The engine centres the design within each class and requires full column rank.
    A term's raw columns need not have it: a basis function beyond the data's range
    is a zero column, and two terms can share a direction. So each group keeps the
    directions whose class-centred columns are not already spanned by what was kept
    before it, taking its penalty's null space first: when the null directions are
    identified they survive exactly, and the infinite-weight limit (a power law, a
    linear dose response) is unchanged. A group with nothing identified is dropped
    along with its weight.
    """
    if not terms:
        raise ValueError("stacking needs at least one term")
    rows = {term.design.shape[0] for term in terms}
    if len(rows) != 1:
        raise ValueError("every term must have one row per variant")
    variant_count = rows.pop()
    total_width = sum(term.design.shape[1] for term in terms)
    stacked_centred = _class_centred(np.hstack([term.design for term in terms]), class_index) if total_width else np.zeros((variant_count, 0))
    floor = _rank_floor(stacked_centred, float(np.sum(stacked_centred * stacked_centred)))
    kept_centred = np.zeros((variant_count, 0))
    selected_groups: list[tuple[int, PenaltyGroup, F64Array, F64Array]] = []
    offset = 0
    for term in terms:
        for group in term.groups:
            block = term.design[:, group.columns]
            centred = _class_centred(block, class_index)
            chosen: list[F64Array] = []
            for directions in _split_by_penalty(group.factor):
                if directions.shape[1] == 0:
                    continue
                candidate = centred @ directions
                if kept_centred.shape[1]:
                    orthonormal, _triangle = np.linalg.qr(kept_centred)
                    candidate = candidate - orthonormal @ (orthonormal.T @ candidate)
                _left, singular, right = np.linalg.svd(candidate, full_matrices=False)
                keep = singular**2 > floor
                if np.any(keep):
                    selected = directions @ right[keep].T
                    chosen.append(selected)
                    kept_centred = np.hstack([kept_centred, centred @ selected])
            if not chosen:
                continue
            selected_groups.append((offset, group, block, np.hstack(chosen)))
        offset += term.design.shape[1]

    # The sequential test bounds each new direction's residual, not the smallest eigenvalue of the
    # whole Gram, which coupling to earlier columns can push lower. So finish with the engine's own
    # criterion, dropping the column the smallest eigenvector loads most on until it holds.
    while selected_groups:
        centred = _class_centred(np.hstack([block @ transform for _o, _g, block, transform in selected_groups]), class_index)
        eigenvalues, eigenvectors = np.linalg.eigh(centred.T @ centred)
        if eigenvalues[0] > _EPSILON * variant_count * max(float(eigenvalues[-1]), 1.0):
            break
        weakest = int(np.argmax(np.abs(eigenvectors[:, 0])))
        for position, (term_offset, group, block, transform) in enumerate(selected_groups):
            if weakest < transform.shape[1]:
                remaining = np.delete(transform, weakest, axis=1)
                if remaining.shape[1]:
                    selected_groups[position] = (term_offset, group, block, remaining)
                else:
                    del selected_groups[position]
                break
            weakest -= transform.shape[1]

    columns: list[F64Array] = []
    groups: list[tuple[I64Array, F64Array]] = []
    group_names: list[str] = []
    names: list[str] = []
    kept_width = sum(transform.shape[1] for _o, _g, _b, transform in selected_groups)
    loading = np.zeros((total_width, kept_width))
    start = 0
    for term_offset, group, block, transform in selected_groups:
        width = transform.shape[1]
        kept = start + np.arange(width, dtype=np.int64)
        columns.append(block @ transform)
        groups.append((kept, transform.T @ group.matrix @ transform))
        group_names.append(group.name)
        names.extend(f"{group.name}::{position}" for position in range(width))
        loading[np.ix_(term_offset + group.columns, kept)] = transform
        start += width
    design = np.hstack(columns) if columns else np.zeros((variant_count, 0))
    return StackedDesign(design=design, groups=tuple(groups), group_names=tuple(group_names), names=tuple(names), loading=loading)


def shape_functionals(*, nodes: F64Array, log_weights: F64Array) -> F64Array:
    """The shape directions a scale term cannot reach: t^2 and t^3 orthogonalized under the fitted weights.

    Normalization and translation of the density are the directions {1, t}; a
    scale term moves the density along them. What remains after projecting them
    out is what an annotation could change about the density's shape
    (scale_model.md §6).
    """
    points = np.asarray(nodes, dtype=np.float64)
    log_mass = np.asarray(log_weights, dtype=np.float64)
    if points.ndim != 1 or log_mass.shape != points.shape:
        raise ValueError("the functionals need one log weight per lattice node")
    weights = np.exp(log_mass - log_mass.max())
    weights /= weights.sum()
    raised = np.stack([points**2, points**3], axis=1)
    reachable = np.stack([np.ones_like(points), points], axis=1)
    weighted = reachable * weights[:, None]
    gram = reachable.T @ weighted
    coefficients = np.linalg.solve(gram, weighted.T @ raised)
    return raised - reachable @ coefficients


def shape_score(*, responsibilities: F64Array, functionals: F64Array, annotation: F64Array, log_weights: F64Array) -> F64Array:
    """The per-variant score of an annotation-dependent shape, at the fitted scale-only model.

    ``responsibilities`` are the posterior weights q_jk the M-step already forms,
    and the score of variant j is a_j (sum_k q_jk phi_k - sum_k w_k phi_k): how far
    its posterior sits from the fitted density along the shape directions.
    """
    posterior = np.asarray(responsibilities, dtype=np.float64)
    shape = np.asarray(functionals, dtype=np.float64)
    annotated = np.asarray(annotation, dtype=np.float64)
    log_mass = np.asarray(log_weights, dtype=np.float64)
    if posterior.ndim != 2 or shape.shape[0] != posterior.shape[1] or annotated.shape != (posterior.shape[0],):
        raise ValueError("the score needs one responsibility row per variant and one functional row per node")
    if log_mass.shape != (posterior.shape[1],):
        raise ValueError("the score needs one log weight per lattice node")
    weights = np.exp(log_mass - log_mass.max())
    weights /= weights.sum()
    return annotated[:, None] * (posterior @ shape - weights @ shape)
