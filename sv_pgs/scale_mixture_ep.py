"""The variant side of the one model's EP-EB fit, shared by Stage 1 and Stage 2.

Every stage approximates the posterior of the effects by a Gaussian q(beta)
with precision (data precision) + diag(tau) and one Gaussian site
exp(-tau_j beta_j^2 / 2 + nu_j beta_j) per effect. The stages differ only in
how they compute q's means and marginal variances (LD blocks in Stage 1, the
exact full-data operator in Stage 2); everything that involves the prior is
here.

Prior. Each effect is a continuous Gaussian scale mixture,

    beta_j ~ integral g_c(t) N(0, u_j e^t) dt,   log u_j = o_j + d_j' theta,

with no point mass at zero.
- g_c, class c's mixing density over t = log s, is a continuous function
  learned nonparametrically, held as its log values eta_ck at the nodes of a
  uniform lattice over the real line; integrals over t are trapezoid sums, so
  node k carries pi_ck = softmax(eta_c)_k.
- The classes share one density shape: eta_c = eta_bar + delta_c. Roughness,
  the integral of eta'''(t)^2 (``ROUGHNESS_ORDERS``), is penalized in its
  lattice form lambda h^-5 ||D3 eta||^2, so lambda does not depend on the
  spacing h: one learned weight on eta_bar and one per class on delta_c. The
  penalty's null space is the quadratics, so as a weight grows log g tends to
  a concave quadratic, a log-normal mixing density proper on the real line;
  lower orders tend to an exponential in t, whose mass depends on where the
  lattice ends. Linear log-tails carry no penalty, so heavy tails stay free.
- A deviation's null-space part, its location and width offsets (the Legendre
  P1 and P2 coefficients over the kernel range), has a Gaussian pooling prior
  with mean zero and a learned precision: eta_bar carries the common location,
  so there is no separate class level, and a class's scale is its deviation's
  location. The kernel range [floor, top] comes from the data, so neither
  offset depends on the lattice's spacing or extent.
- eta_bar's own null space (the log-normal limit's location and width) is
  profiled (the Schur-complement form of the evidence below): integrating it
  under a flat prior diverges where the likelihood tends to a positive
  constant, and makes the evidence grid-dependent.
- Every weight lives in [0, infinity] with exact edges: at infinity the
  block's penalized directions are zero, at zero the block is absent.
- The lattice is a quadrature rule: its floor, top, spacing and extent are
  derived from the data and a tolerance (``kernel_floor``, ``kernel_top``,
  ``spacing_bound``, ``tail_mass``). Below the floor every kernel is flat to
  that tolerance, so those nodes carry the density's mass with no kernel
  evaluation: an exact effective zero, not a point mass.
- o_j = log r2_j is the measurement offset, with coefficient exactly 1 by
  derivation (the prior is on the true genotype's effect).
- d_j is the annotation row centred within its class; theta splits into
  groups, each with a penalty matrix and a learned weight.

EP. The cavity of site j is N(beta; h_j / P_j, 1 / P_j) written through its
precision P_j = 1 / Sigma_jj - tau_j and shift h_j = mu_j / Sigma_jj - nu_j.
At node k the tilted integrand integrates to

    Z_jk = (1 + v P)^(-1/2) exp(h^2 v / (2 (1 + v P))),   v = u_j e^t_k,

and given t_k it is Gaussian with mean h c_k and variance c_k = v / (1 + v P).
Sites are mean-matched and never clipped: a site precision may be negative
while the posterior precision stays positive definite, because clipping
breaks the identity that makes the fixed-cavity gradient the EP evidence's
gradient.

Empirical Bayes. At fixed cavities, the EP marginal likelihood's dependence on
the prior is sum_j log Z_j. The coefficients x (eta_bar, the deviations,
theta) maximize it minus the penalties, by Newton on the observed Hessian
where it is negative definite and on its spectrum's magnitudes elsewhere (the
objective is not concave in log g). The penalty weights lambda maximize the
Laplace marginal likelihood of x,

    V(rho) = F(x_rho) + 1/2 log|S_rho|_+ - 1/2 log|-H(x_rho)| + 1/2 log|N'(-H(x_rho))N|,

with rho = log lambda, x_rho the penalized maximizer, H its observed Hessian
and N a basis of any directions no block penalizes (an annotation smooth's
linear part, or a direction left bare by a weight at zero), which are
profiled as fixed effects: integrating them under a flat prior diverges
where the likelihood tends to a positive constant. Its gradient is exact:
dx/drho through the observed Hessian, and the change of H with x through the
third derivatives of log Z. Across a stage's outer loop the curvature is the
EP-re-solved one; at fixed cavities it is H.

Every derivative is computed in z = (eta_1, ..., eta_C, theta), where each variant's log Z_j depends on its
class's eta_c and its own log u_j, and carried to x by the linear map M.

Noise. A quantitative trait's residual variance takes the MacKay form of its
type-II ML stationarity, sigma^2 = RSS / (n - k - gamma) with
gamma = p - sum_j tau_j Sigma_jj, the covariates' flat prior removing k.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator, Sequence

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import erfcx, logsumexp

from sv_pgs._typing import F64Array, I64Array

_EPSILON = float(np.finfo(np.float64).eps)
# Half of double precision: the resolution of a quantity whose square is compared at eps.
_HALF_PRECISION = _EPSILON**0.5
# The mixing density's roughness penalty: the integral of its squared third derivative, with a learned weight.
# Third order is derived: its null space, the quadratics (a normal density in log s), is the only lambda =
# infinity limit that is proper on the real line, and that null space is profiled. The first-plus-second
# order form (no null space in sum-to-zero coordinates) is the alternative the benchmark's held-out log
# predictive density adjudicates (lead ruling); any tuple of orders is supported.
ROUGHNESS_ORDERS = (3,)


@dataclass(frozen=True)
class AnnotationGroup:
    """Annotation columns that share one smoothing weight, and their penalty matrix."""

    columns: I64Array
    penalty: F64Array


@dataclass(frozen=True)
class SmoothingBlock:
    """One learned penalty weight: its coordinates in x and the square-root factor R of its matrix R'R there.

    Blocks with the same coordinates add up to one total penalty on them. Penalty
    values are ||R x||^2, never x'(R'R)x, which rounding can make negative when
    x is large in the null space.
    """

    name: str
    coordinates: I64Array
    factor: F64Array

    @property
    def matrix(self) -> F64Array:
        return self.factor.T @ self.factor


@dataclass(frozen=True)
class ScaleMixturePrior:
    """Everything about the prior except the fitted hyperparameters. Build it with ``scale_mixture_prior``.

    ``scale_design`` is the class-centred annotation design (p, F), over theta;
    ``coefficient_map`` is M, from x to z.
    """

    class_index: I64Array
    class_count: int
    class_rows: tuple[I64Array, ...]
    log_variance_offset: F64Array
    scale_design: F64Array
    annotation_groups: tuple[AnnotationGroup, ...]
    log_variance_grid: F64Array
    kernel_floor: float
    kernel_top: float
    coefficient_map: F64Array
    smoothing_blocks: tuple[SmoothingBlock, ...]
    pooled_size: int
    null_basis: F64Array

    @property
    def variant_count(self) -> int:
        return int(self.class_index.shape[0])

    @property
    def grid_size(self) -> int:
        return int(self.log_variance_grid.shape[0])

    @property
    def scale_size(self) -> int:
        return int(self.scale_design.shape[1])

    @property
    def density_size(self) -> int:
        return self.class_count * self.grid_size

    @property
    def coefficient_size(self) -> int:
        return int(self.coefficient_map.shape[1])


@dataclass(frozen=True)
class MixtureHyperparameters:
    """x, the packed coefficients, and rho, one log penalty weight per ``prior.smoothing_blocks``."""

    coefficients: F64Array
    log_smoothing: F64Array


@dataclass(frozen=True)
class Cavity:
    precision: F64Array
    shift: F64Array


@dataclass(frozen=True)
class TiltedMoments:
    log_normalizer: F64Array
    mean: F64Array
    variance: F64Array


@dataclass(frozen=True)
class HyperStep:
    """The result of one empirical-Bayes step at fixed cavities.

    ``start_decrement`` is 1/2 g'|H|^-1 g of the penalized objective at the
    hyperparameters the step started from (what these cavities still ask of
    them), and ``evidence_gain`` is V at the returned weights minus V at the
    starting ones, both in nats: the outer loop's certificate.
    ``newton_decrement`` is the same decrement at the returned
    hyperparameters, and ``smoothing_gradient`` the largest |dV/drho| over
    weights not held at a bound of their range.
    """

    hyperparameters: MixtureHyperparameters
    penalized_objective: float
    evidence: float
    newton_decrement: float
    smoothing_gradient: float
    start_decrement: float
    evidence_gain: float


# ------------------------------------------------------------------ the lattice


def _sum_to_zero_basis(size: int) -> F64Array:
    """Orthonormal basis of the vectors that sum to zero."""
    centred = np.eye(size) - np.full((size, size), 1.0 / size)
    basis, _singular, _rows = np.linalg.svd(centred)
    return basis[:, : size - 1]


def roughness_factor(size: int, spacing: float, order: int) -> F64Array:
    """R with ||R eta||^2 = h^-(2 order - 1) ||D_order eta||^2, the lattice form of the integral of eta^(order)(t)^2."""
    return np.diff(np.eye(size), n=order, axis=0) / spacing ** (order - 0.5)


def kernel_floor(single_precision: F64Array, single_shift: F64Array, log_scale_values: F64Array, tolerance: float) -> float:
    """t_min: below it every kernel is flat, and replacing it by 1 moves sum_j log Z_j by at most ``tolerance``.

    |log L_j| <= v |h^2 - P| / 2 + (v P)^2 / 4 with v = u_j s, so sum_j of the bound is
    A1 s + A2 s^2; its root at ``tolerance`` is taken in the stable form.
    """
    scale = np.exp(log_scale_values)
    linear = 0.5 * float(np.sum(scale * np.abs(np.square(single_shift) - single_precision)))
    quadratic = 0.25 * float(np.sum(np.square(scale * single_precision)))
    return float(np.log(2.0 * tolerance / (linear + np.sqrt(linear * linear + 4.0 * quadratic * tolerance))))


def kernel_top(single_precision: F64Array, single_shift: F64Array, log_scale_values: F64Array, floor: float) -> float:
    """The largest kernel mode, log((h^2 - P) / (P^2 u)); every kernel decreases beyond it. The floor when none has a mode."""
    signal = np.square(single_shift) > single_precision
    if not np.any(signal):
        return floor
    precision = single_precision[signal]
    modes = np.log((np.square(single_shift[signal]) - precision) / np.square(precision)) - log_scale_values[signal]
    return float(max(floor, np.max(modes)))


def spacing_bound(majorant_ratio_sum: float, tolerance: float) -> float:
    """h <= pi^2 / ln(1 + 2 sum_j M_j / Z_j / tolerance): the trapezoid error on the strip |Im t| < pi/2."""
    return float(np.pi**2 / np.log1p(2.0 * majorant_ratio_sum / tolerance))


def derived_lattice(single_precision: F64Array, single_shift: F64Array, log_scale_values: F64Array, tolerance: float) -> tuple[F64Array, float, float]:
    """The start lattice (nodes, floor, top) from each effect's own likelihood exp(-P beta^2 / 2 + h beta).

    The floor and top are ``kernel_floor`` and ``kernel_top`` at ``tolerance``; the spacing is
    ``spacing_bound`` with every M_j / Z_j at its minimum of one (a fit refines it against the fitted
    density by ``halved_lattice``); the lattice reaches past the kernel range by the range's own width on
    each side, which the start density's tails (``initial_hyperparameters``) leave empty to double precision.
    """
    floor = kernel_floor(single_precision, single_shift, log_scale_values, tolerance)
    top = kernel_top(single_precision, single_shift, log_scale_values, floor)
    spacing = spacing_bound(float(np.asarray(single_precision).shape[0]), tolerance)
    width = max(top - floor, spacing)
    return np.arange(floor - width, top + width + spacing, spacing), floor, top


def tail_mass(end_value: float, outward_slope: float, curvature: float) -> float:
    """The integral over tau >= 0 of exp(end_value + outward_slope tau + curvature tau^2 / 2); inf when improper."""
    if curvature < 0.0:
        root = np.sqrt(-2.0 * curvature)
        return float(np.exp(end_value) * np.sqrt(np.pi / (-2.0 * curvature)) * erfcx(-outward_slope / root))
    if curvature == 0.0 and outward_slope < 0.0:
        return float(np.exp(end_value) / -outward_slope)
    return float("inf")


def _legendre_functionals(nodes: F64Array, floor: float, top: float, degree_count: int) -> F64Array:
    """(degree_count, K): node weights of a function's Legendre P1..P_degree_count coefficients over [floor, top].

    These are the polynomial parts a roughness penalty of order degree_count + 1 cannot see, beyond the
    constant that normalization removes. Trapezoid weights over the nodes inside the range (at least three
    nearest its centre when it is narrower) keep them independent of the spacing and of the lattice's extent.
    """
    distance = np.abs(nodes - 0.5 * (floor + top))
    inside = (nodes >= floor) & (nodes <= top)
    if int(inside.sum()) < 3:
        inside = distance <= np.sort(distance)[2]
    covered = nodes[inside]
    centre = 0.5 * (covered[0] + covered[-1])
    half = 0.5 * (covered[-1] - covered[0])
    standardized = (covered - centre) / half
    weights = np.full(covered.shape[0], covered[1] - covered[0])
    weights[[0, -1]] *= 0.5
    functionals = np.zeros((degree_count, nodes.shape[0]))
    for degree in range(1, degree_count + 1):
        polynomial = np.polynomial.legendre.Legendre.basis(degree)(standardized)
        functionals[degree - 1, inside] = (2 * degree + 1) / (2.0 * half) * weights * polynomial
    return functionals


def scale_mixture_prior(
    *,
    class_index: I64Array,
    log_variance_offset: F64Array,
    annotation_design: F64Array,
    annotation_groups: Sequence[AnnotationGroup],
    nodes: F64Array,
    floor: float,
    top: float,
) -> ScaleMixturePrior:
    """Validate and centre the prior's inputs and lay out x; every class in 0..C-1 must have a member.

    ``nodes`` is the uniform lattice in t; nodes below ``floor`` carry mass with a
    flat kernel, and [floor, top] is the kernel range. x is laid out as (eta_bar in
    sum-to-zero coordinates, each class's deviation delta_c in the same
    coordinates when there are several classes, theta).
    """
    classes = np.asarray(class_index, dtype=np.int64)
    class_count = int(classes.max()) + 1
    class_sizes = np.bincount(classes, minlength=class_count)
    if np.any(class_sizes == 0):
        raise ValueError("every class index below the largest must have a member")
    design = np.array(annotation_design, dtype=np.float64, copy=True)
    if design.shape[0] != classes.shape[0]:
        raise ValueError("the annotation design needs one row per variant")
    order = np.argsort(classes, kind="stable")
    class_rows = tuple(np.split(order, np.cumsum(class_sizes)[:-1]))
    for rows in class_rows:
        design[rows] -= design[rows].mean(axis=0)
    covered = np.concatenate([group.columns for group in annotation_groups]) if annotation_groups else np.zeros(0, np.int64)
    if not np.array_equal(np.sort(covered), np.arange(design.shape[1])):
        raise ValueError("annotation groups must cover every design column exactly once")
    if design.shape[1]:
        eigenvalues = np.linalg.eigvalsh(design.T @ design)
        if eigenvalues[0] <= _EPSILON * design.shape[0] * max(float(eigenvalues[-1]), 1.0):
            raise ValueError(
                "the class-centred annotation design must have full column rank: a smooth basis must "
                "drop its constant (the class densities carry it)"
            )
    lattice = np.asarray(nodes, dtype=np.float64)
    spacing = np.diff(lattice)
    if lattice.shape[0] < 5 or spacing[0] <= 0.0 or not np.allclose(spacing, spacing[0], rtol=_HALF_PRECISION, atol=0.0):
        raise ValueError("the lattice must be at least five evenly spaced increasing nodes")
    grid_size = lattice.shape[0]
    basis = _sum_to_zero_basis(grid_size)
    factors = {order: roughness_factor(grid_size, float(spacing[0]), order) @ basis for order in ROUGHNESS_ORDERS}
    null_functionals = _legendre_functionals(lattice, floor, top, min(ROUGHNESS_ORDERS) - 1) @ basis
    pooled_size = grid_size - 1
    deviation_count = class_count if class_count > 1 else 0
    deviation_size = deviation_count * pooled_size
    coefficient_size = pooled_size + deviation_size + design.shape[1]
    coefficient_map = np.zeros((class_count * grid_size + design.shape[1], coefficient_size))
    for class_position in range(class_count):
        rows = slice(class_position * grid_size, (class_position + 1) * grid_size)
        coefficient_map[rows, :pooled_size] = basis
        if deviation_count:
            start = pooled_size + class_position * pooled_size
            coefficient_map[rows, start : start + pooled_size] = basis
    coefficient_map[class_count * grid_size :, pooled_size + deviation_size :] = np.eye(design.shape[1])
    pooled = np.arange(pooled_size)
    blocks = [SmoothingBlock(f"pooled roughness order {order}", pooled, factor) for order, factor in factors.items()]
    for class_position in range(deviation_count):
        deviation = pooled_size + class_position * pooled_size + np.arange(pooled_size)
        blocks.extend(SmoothingBlock(f"class {class_position} deviation roughness order {order}", deviation, factor) for order, factor in factors.items())
    if deviation_count and null_functionals.shape[0]:
        # A deviation's polynomial part below the penalty orders (its location and width, for order 3) has a
        # Gaussian pooling prior with mean zero and one learned precision: eta_bar carries the common part.
        blocks.append(SmoothingBlock(
            "deviation polynomial part", np.arange(pooled_size, pooled_size + deviation_size), np.kron(np.eye(deviation_count), null_functionals)
        ))
    annotation_start = pooled_size + deviation_size
    for position, group in enumerate(annotation_groups):
        eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(group.penalty, dtype=np.float64))
        kept = eigenvalues > _EPSILON * eigenvalues.shape[0] * max(float(eigenvalues[-1]), np.finfo(np.float64).tiny)
        blocks.append(SmoothingBlock(
            f"annotation group {position}",
            annotation_start + np.asarray(group.columns, dtype=np.int64),
            np.sqrt(eigenvalues[kept])[:, None] * eigenvectors[:, kept].T,
        ))
    total = np.zeros((coefficient_size, coefficient_size))
    for block in blocks:
        total[np.ix_(block.coordinates, block.coordinates)] += block.matrix
    null_eigenvalues, null_vectors = np.linalg.eigh(total)
    null_basis = null_vectors[:, null_eigenvalues <= _EPSILON * coefficient_size * float(null_eigenvalues[-1])]
    return ScaleMixturePrior(
        class_index=classes,
        class_count=class_count,
        class_rows=class_rows,
        log_variance_offset=np.asarray(log_variance_offset, dtype=np.float64),
        scale_design=design,
        annotation_groups=tuple(annotation_groups),
        log_variance_grid=lattice,
        kernel_floor=float(floor),
        kernel_top=float(top),
        coefficient_map=coefficient_map,
        smoothing_blocks=tuple(blocks),
        pooled_size=pooled_size,
        null_basis=null_basis,
    )


def initial_hyperparameters(prior: ScaleMixturePrior) -> MixtureHyperparameters:
    """A start, not a prior: every class at the log-normal centred on the lattice whose density at the lattice's
    ends is eps of its peak; no deviation or annotation effect; unit penalty weights."""
    nodes = prior.log_variance_grid
    centre = 0.5 * (nodes[0] + nodes[-1])
    width = 0.5 * (nodes[-1] - nodes[0]) / np.sqrt(2.0 * np.log(1.0 / _EPSILON))
    quadratic = -0.5 * np.square((nodes - centre) / width)
    coefficients = np.zeros(prior.coefficient_size)
    coefficients[: prior.pooled_size] = prior.coefficient_map[: prior.grid_size, : prior.pooled_size].T @ (quadratic - quadratic.mean())
    return MixtureHyperparameters(coefficients=coefficients, log_smoothing=np.zeros(len(prior.smoothing_blocks)))


def _density_and_scale(prior: ScaleMixturePrior, coefficients: F64Array) -> tuple[F64Array, F64Array]:
    """z = M x split into the class log densities (C x K, unnormalized) and the scale coefficients (L,)."""
    values = prior.coefficient_map @ coefficients
    return values[: prior.density_size].reshape(prior.class_count, prior.grid_size), values[prior.density_size :]


def class_log_density(prior: ScaleMixturePrior, coefficients: F64Array) -> F64Array:
    """log pi_ck = eta_ck - log sum_m e^eta_cm: the lattice mass of node k (C x K); the uniform weight h cancels."""
    log_weights, _scale = _density_and_scale(prior, coefficients)
    return log_weights - logsumexp(log_weights, axis=1, keepdims=True)


def log_scale(prior: ScaleMixturePrior, coefficients: F64Array) -> F64Array:
    """log u_j = o_j + d_j' theta."""
    _density, scale_coefficients = _density_and_scale(prior, coefficients)
    return prior.log_variance_offset + prior.scale_design @ scale_coefficients


def halved_lattice(prior: ScaleMixturePrior, hyperparameters: MixtureHyperparameters) -> tuple[ScaleMixturePrior, MixtureHyperparameters]:
    """The same model on the lattice with half the spacing; each class's eta at the new midpoints from the cubic through the old nodes."""
    nodes = prior.log_variance_grid
    finer = np.linspace(nodes[0], nodes[-1], 2 * nodes.shape[0] - 1)
    log_density, scale_coefficients = _density_and_scale(prior, hyperparameters.coefficients)
    finer_density = CubicSpline(nodes, log_density.T)(finer).T
    refined = scale_mixture_prior(
        class_index=prior.class_index,
        log_variance_offset=prior.log_variance_offset,
        annotation_design=prior.scale_design,
        annotation_groups=prior.annotation_groups,
        nodes=finer,
        floor=prior.kernel_floor,
        top=prior.kernel_top,
    )
    normalized = finer_density - finer_density.mean(axis=1, keepdims=True)
    target = np.concatenate([normalized.ravel(), scale_coefficients])
    coefficients = np.linalg.lstsq(refined.coefficient_map, target, rcond=None)[0]
    return refined, MixtureHyperparameters(coefficients=coefficients, log_smoothing=hyperparameters.log_smoothing.copy())


def prior_second_moment(prior: ScaleMixturePrior, hyperparameters: MixtureHyperparameters) -> F64Array:
    """E[beta_j^2] under the prior: u_j sum_k pi_ck e^t_k."""
    log_density = class_log_density(prior, hyperparameters.coefficients)
    log_mean_variance = logsumexp(log_density + prior.log_variance_grid[None, :], axis=1)
    return np.exp(log_scale(prior, hyperparameters.coefficients) + log_mean_variance[prior.class_index])


def _row_chunks(rows: I64Array, grid_size: int, working_bytes: int) -> Iterator[I64Array]:
    """Pieces of ``rows`` whose (rows x K) intermediates, about 16 of them, fit ``working_bytes``."""
    if working_bytes <= 0:
        raise ValueError("working_bytes must be positive")
    chunk = max(1, int(working_bytes) // (16 * 8 * max(grid_size, 1)))
    for start in range(0, rows.shape[0], chunk):
        yield rows[start : start + chunk]


# ------------------------------------------------------------------ the tilted moments


@dataclass(frozen=True)
class _Components:
    """Per variant and node: responsibilities w, conditional variances c and the derivatives of log Z_jk in eta = log u_j."""

    log_normalizer: F64Array
    responsibility: F64Array
    conditional_variance: F64Array
    first: F64Array
    second: F64Array
    third: F64Array


def _kernel_terms(
    log_density: F64Array, log_scale_rows: F64Array, grid: F64Array, floor: float, precision: F64Array, shift: F64Array
) -> tuple[F64Array, F64Array, F64Array, F64Array]:
    """(variance v, q = vP, r = 1/(1+q), log pi_k + log Z_jk) at every node; v = 0 below ``floor``."""
    variance = np.where(grid[None, :] >= floor, np.exp(log_scale_rows[:, None] + grid[None, :]), 0.0)
    ratio = variance * precision[:, None]
    if np.any(ratio <= -1.0):
        raise FloatingPointError("a cavity is improper on the lattice: 1 + v P <= 0")
    retained = 1.0 / (1.0 + ratio)
    log_component = log_density - 0.5 * np.log1p(ratio) + 0.5 * np.square(shift)[:, None] * variance * retained
    return variance, ratio, retained, log_component


def _log_normalizers(
    log_density: F64Array, log_scale_rows: F64Array, grid: F64Array, floor: float, precision: F64Array, shift: F64Array
) -> F64Array:
    return logsumexp(_kernel_terms(log_density, log_scale_rows, grid, floor, precision, shift)[3], axis=1)


def _components(
    log_density: F64Array, log_scale_rows: F64Array, grid: F64Array, floor: float, precision: F64Array, shift: F64Array
) -> _Components:
    """With q = vP, r = 1/(1+q) and a = h^2 v r:  d log Z_k / d eta = (a - q) r / 2,
    its eta-derivative a r (2r - 1)/2 - q r^2/2, and the next a r (6r^2 - 6r + 1)/2 - q r^2 (2r - 1)/2.
    Nodes below ``floor`` have a flat kernel: v = 0 there."""
    variance, ratio, retained, log_component = _kernel_terms(log_density, log_scale_rows, grid, floor, precision, shift)
    signal = np.square(shift)[:, None] * variance * retained
    log_normalizer = logsumexp(log_component, axis=1)
    responsibility = np.exp(log_component - log_normalizer[:, None])
    ratio_retained = ratio * retained
    return _Components(
        log_normalizer=log_normalizer,
        responsibility=responsibility,
        conditional_variance=variance * retained,
        first=0.5 * retained * (signal - ratio),
        second=0.5 * signal * retained * (2.0 * retained - 1.0) - 0.5 * ratio_retained * retained,
        third=0.5 * signal * retained * (6.0 * retained * retained - 6.0 * retained + 1.0)
        - 0.5 * ratio_retained * retained * (2.0 * retained - 1.0),
    )


def _class_terms(
    prior: ScaleMixturePrior, coefficients: F64Array, cavity: Cavity, working_bytes: int
) -> Iterator[tuple[int, I64Array, _Components]]:
    """(class, rows, components) over every chunk of every class."""
    log_density = class_log_density(prior, coefficients)
    scales = log_scale(prior, coefficients)
    for class_position, class_rows in enumerate(prior.class_rows):
        for rows in _row_chunks(class_rows, prior.grid_size, working_bytes):
            yield class_position, rows, _components(
                log_density[class_position], scales[rows], prior.log_variance_grid, prior.kernel_floor, cavity.precision[rows], cavity.shift[rows]
            )


def tilted_moments(
    prior: ScaleMixturePrior, hyperparameters: MixtureHyperparameters, cavity: Cavity, working_bytes: int
) -> TiltedMoments:
    """log Z_j and the exact tilted mean and variance of every effect."""
    log_normalizer = np.empty(prior.variant_count)
    mean = np.empty(prior.variant_count)
    variance = np.empty(prior.variant_count)
    for _class, rows, terms in _class_terms(prior, hyperparameters.coefficients, cavity, working_bytes):
        shift = cavity.shift[rows]
        conditional = terms.conditional_variance
        first_moment = np.sum(terms.responsibility * conditional, axis=1)
        # Var = E_w[c] + h^2 Var_w(c): both terms are non-negative, so nothing cancels.
        spread = np.sum(terms.responsibility * np.square(conditional - first_moment[:, None]), axis=1)
        log_normalizer[rows] = terms.log_normalizer
        mean[rows] = shift * first_moment
        variance[rows] = first_moment + np.square(shift) * spread
    return TiltedMoments(log_normalizer=log_normalizer, mean=mean, variance=variance)


def quadrature_majorant_ratio(
    prior: ScaleMixturePrior, hyperparameters: MixtureHyperparameters, cavity: Cavity, working_bytes: int
) -> float:
    """sum_j M_j / Z_j with M_j = max(Z_j, sum_k pi_k |L_j(t_k + i pi/2)|), for ``spacing_bound``.

    At t + i pi/2 the variance is i v, so log|L| = -log(1 + q^2)/4 + h^2 v q / (2 (1 + q^2)); flat nodes stay at 1.
    """
    log_density = class_log_density(prior, hyperparameters.coefficients)
    scales = log_scale(prior, hyperparameters.coefficients)
    kernel = prior.log_variance_grid[None, :] >= prior.kernel_floor
    total = 0.0
    for class_position, class_rows in enumerate(prior.class_rows):
        for rows in _row_chunks(class_rows, prior.grid_size, working_bytes):
            variance = np.where(kernel, np.exp(scales[rows][:, None] + prior.log_variance_grid[None, :]), 0.0)
            ratio = variance * cavity.precision[rows][:, None]
            shift_square = np.square(cavity.shift[rows])[:, None]
            log_real = log_density[class_position] - 0.5 * np.log1p(ratio) + 0.5 * shift_square * variance / (1.0 + ratio)
            log_strip = log_density[class_position] - 0.25 * np.log1p(ratio * ratio) + 0.5 * shift_square * variance * ratio / (1.0 + ratio * ratio)
            total += float(np.sum(np.exp(np.maximum(logsumexp(log_strip, axis=1) - logsumexp(log_real, axis=1), 0.0))))
    return total


def cavities(posterior_mean: F64Array, posterior_variance: F64Array, site_precision: F64Array, site_shift: F64Array) -> Cavity:
    """Divide each site out of q's marginal."""
    marginal_precision = 1.0 / posterior_variance
    return Cavity(precision=marginal_precision - site_precision, shift=posterior_mean * marginal_precision - site_shift)


def site_targets(moments: TiltedMoments, cavity: Cavity) -> tuple[F64Array, F64Array]:
    """Mean-matched sites from the exact tilted moments: q's marginal becomes N(tilted mean, tilted variance)."""
    precision = 1.0 / moments.variance - cavity.precision
    return precision, moments.mean / moments.variance - cavity.shift


def moment_matched_prior_sites(prior: ScaleMixturePrior, hyperparameters: MixtureHyperparameters) -> tuple[F64Array, F64Array]:
    """The sites of the Gaussian with the prior's own variance: q starts at the prior."""
    return 1.0 / prior_second_moment(prior, hyperparameters), np.zeros(prior.variant_count)


def noise_variance(
    *, residual_sum_of_squares: float, sample_count: int, covariate_count: int, site_precision: F64Array, posterior_variance: F64Array
) -> float:
    """sigma^2 = RSS / (n - k - gamma), gamma = p - sum_j tau_j Sigma_jj (the effective number of effects)."""
    effective_effects = float(site_precision.shape[0] - np.sum(site_precision * posterior_variance))
    return float(residual_sum_of_squares) / (sample_count - covariate_count - effective_effects)


# ------------------------------------------------------------------ the hyper objective


@dataclass(frozen=True)
class _Objective:
    """sum_j log Z_j at fixed cavities, its gradient and Hessian in z, and the sum of |log Z_j| (its rounding scale)."""

    value: float
    gradient: F64Array
    hessian: F64Array
    magnitude: float


def _data_objective(prior: ScaleMixturePrior, coefficients: F64Array, cavity: Cavity, working_bytes: int) -> _Objective:
    """For variant j of class c, with responsibilities w_j, component derivatives g_jk and their mean gbar_j, in z:
    d/deta_c = w_j - pi_c and d/d(scale) = gbar_j d_j (d_j the variant's scale-design row);
    d2/deta_c2 = diag(w_j) - w_j w_j' - (diag pi_c - pi_c pi_c'), d2/deta_ck d(scale) = w_jk (g_jk - gbar_j) d_j,
    and d2/d(scale)2 = (Var_w(g_j) + E_w[dg_j/deta]) d_j d_j'."""
    grid_size = prior.grid_size
    scale_span = slice(prior.density_size, prior.density_size + prior.scale_size)
    dimension = prior.density_size + prior.scale_size
    gradient = np.zeros(dimension)
    hessian = np.zeros((dimension, dimension))
    value = magnitude = 0.0
    density = np.exp(class_log_density(prior, coefficients))
    responsibility_sum = np.zeros((prior.class_count, grid_size))
    responsibility_outer = np.zeros((prior.class_count, grid_size, grid_size))
    cross = np.zeros((prior.class_count, grid_size, prior.scale_size))
    for class_position, rows, terms in _class_terms(prior, coefficients, cavity, working_bytes):
        responsibility = terms.responsibility
        design = prior.scale_design[rows]
        mean_first = np.sum(responsibility * terms.first, axis=1)
        centred_first = terms.first - mean_first[:, None]
        curvature = np.sum(responsibility * (np.square(centred_first) + terms.second), axis=1)
        value += float(np.sum(terms.log_normalizer))
        magnitude += float(np.sum(np.abs(terms.log_normalizer)))
        responsibility_sum[class_position] += responsibility.sum(axis=0)
        responsibility_outer[class_position] += responsibility.T @ responsibility
        cross[class_position] += (responsibility * centred_first).T @ design
        gradient[scale_span] += design.T @ mean_first
        hessian[scale_span, scale_span] += design.T @ (curvature[:, None] * design)
    for class_position, class_rows in enumerate(prior.class_rows):
        size = float(class_rows.shape[0])
        class_density = density[class_position]
        span = slice(class_position * grid_size, (class_position + 1) * grid_size)
        gradient[span] = responsibility_sum[class_position] - size * class_density
        hessian[span, span] = (
            np.diag(responsibility_sum[class_position])
            - responsibility_outer[class_position]
            - size * (np.diag(class_density) - np.outer(class_density, class_density))
        )
        hessian[span, scale_span] = cross[class_position]
        hessian[scale_span, span] = cross[class_position].T
    return _Objective(value=value, gradient=gradient, hessian=hessian, magnitude=magnitude)


def _data_value(prior: ScaleMixturePrior, coefficients: F64Array, cavity: Cavity, working_bytes: int) -> float:
    """sum_j log Z_j alone: a trial point's evaluation."""
    log_density = class_log_density(prior, coefficients)
    scales = log_scale(prior, coefficients)
    total = 0.0
    for class_position, class_rows in enumerate(prior.class_rows):
        for rows in _row_chunks(class_rows, prior.grid_size, working_bytes):
            total += float(np.sum(_log_normalizers(
                log_density[class_position], scales[rows], prior.log_variance_grid, prior.kernel_floor, cavity.precision[rows], cavity.shift[rows]
            )))
    return total


def _penalty_matrix(prior: ScaleMixturePrior, log_smoothing: F64Array) -> F64Array:
    """S_rho over x: each block's matrix times its weight."""
    penalty = np.zeros((prior.coefficient_size, prior.coefficient_size))
    for block, log_weight in zip(prior.smoothing_blocks, log_smoothing):
        penalty[np.ix_(block.coordinates, block.coordinates)] += np.exp(log_weight) * block.matrix
    return penalty


def _penalty_value(prior: ScaleMixturePrior, log_smoothing: F64Array, coefficients: F64Array) -> tuple[float, F64Array]:
    """1/2 sum_i lambda_i ||R_i x_i||^2 and its gradient, from the square-root factors."""
    value = 0.0
    gradient = np.zeros_like(coefficients)
    for block, log_weight in zip(prior.smoothing_blocks, log_smoothing):
        residual = block.factor @ coefficients[block.coordinates]
        value += 0.5 * float(np.exp(log_weight)) * float(residual @ residual)
        gradient[block.coordinates] += np.exp(log_weight) * (block.factor.T @ residual)
    return value, gradient


def _penalized(
    prior: ScaleMixturePrior, objective: _Objective, log_smoothing: F64Array, penalty: F64Array, coefficients: F64Array
) -> tuple[float, F64Array, F64Array]:
    """The penalized objective, its gradient and Hessian in x."""
    mapping = prior.coefficient_map
    penalty_value, penalty_gradient = _penalty_value(prior, log_smoothing, coefficients)
    return (
        objective.value - penalty_value,
        mapping.T @ objective.gradient - penalty_gradient,
        mapping.T @ objective.hessian @ mapping - penalty,
    )


def _ascent_direction(negative_hessian: F64Array, gradient: F64Array) -> F64Array:
    """The Newton direction on -H with every eigenvalue replaced by its magnitude (an ascent direction where -H is indefinite).

    Eigenvalues below eps times the largest are raised to that floor, the rounding level of the spectrum.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (negative_hessian + negative_hessian.T))
    magnitudes = np.maximum(np.abs(eigenvalues), _EPSILON * float(np.max(np.abs(eigenvalues))))
    return eigenvectors @ ((eigenvectors.T @ gradient) / magnitudes)


def _trust_region_step(negative_hessian: F64Array, gradient: F64Array, radius: float) -> F64Array:
    """The maximizer of g's - s'(-H)s/2 over ||s|| <= radius (More and Sorensen), from -H's eigendecomposition.

    s(mu) = (-H + mu I)^-1 g with mu >= max(0, -lambda_min) and ||s(mu)|| = radius unless the Newton
    step of a positive definite -H already fits; ||s(mu)|| falls monotonically in mu, so mu is bisected
    to double precision between its bounds.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (negative_hessian + negative_hessian.T))
    components = eigenvectors.T @ gradient
    if eigenvalues[0] > 0.0:
        newton = components / eigenvalues
        if float(np.linalg.norm(newton)) <= radius:
            return eigenvectors @ newton
    lower = max(0.0, -float(eigenvalues[0]))
    upper = lower + float(np.linalg.norm(gradient)) / radius
    while upper - lower > _EPSILON * upper:
        middle = 0.5 * (lower + upper)
        if middle in (lower, upper):
            break
        if float(np.linalg.norm(components / (eigenvalues + middle))) > radius:
            lower = middle
        else:
            upper = middle
    return eigenvectors @ (components / (eigenvalues + upper))


def _maximize_coefficients(
    prior: ScaleMixturePrior, log_smoothing: F64Array, start: F64Array, cavity: Cavity, working_bytes: int, tolerance: float
) -> tuple[F64Array, _Objective]:
    """Trust-region Newton ascent of the penalized objective at fixed penalty weights.

    The objective is not concave in log g, so each step maximizes the quadratic model inside a radius
    (More-Sorensen); a trial costs one value-only pass, and the radius follows the ratio of actual to
    predicted gain (Nocedal and Wright, Algorithm 4.1). It stops at a strict maximum, where -H is positive
    definite and the Newton step's predicted gain is below ``tolerance`` nats (the resolution the caller
    certifies, and at least the objective's rounding level); a saddle's negative curvature is followed by
    the trust-region step instead. It also stops when no step longer than half of double precision raises
    the objective.
    """
    penalty = _penalty_matrix(prior, log_smoothing)
    coefficients = np.array(start, dtype=np.float64, copy=True)
    objective = _data_objective(prior, coefficients, cavity, working_bytes)
    value, gradient, hessian = _penalized(prior, objective, log_smoothing, penalty, coefficients)
    radius = float(np.linalg.norm(_ascent_direction(-hessian, gradient)))
    while True:
        rounding = _EPSILON * (objective.magnitude + abs(value))
        definite = float(np.linalg.eigvalsh(0.5 * (hessian + hessian.T))[-1]) < 0.0
        if definite and 0.5 * float(gradient @ _ascent_direction(-hessian, gradient)) <= max(tolerance, rounding):
            return coefficients, objective
        if radius <= _HALF_PRECISION * (1.0 + float(np.linalg.norm(coefficients))):
            return coefficients, objective
        step = _trust_region_step(-hessian, gradient, radius)
        predicted = float(gradient @ step) + 0.5 * float(step @ hessian @ step)
        candidate = coefficients + step
        candidate_value = _data_value(prior, candidate, cavity, working_bytes) - _penalty_value(prior, log_smoothing, candidate)[0]
        actual = candidate_value - value
        ratio = actual / predicted if predicted > 0.0 else -np.inf
        step_norm = float(np.linalg.norm(step))
        if not np.isfinite(ratio) or ratio < 0.25:
            radius = 0.25 * step_norm
        elif ratio > 0.75 and step_norm >= radius * (1.0 - _HALF_PRECISION):
            radius = 2.0 * radius
        if np.isfinite(candidate_value) and actual > 0.0:
            coefficients = candidate
            objective = _data_objective(prior, coefficients, cavity, working_bytes)
            value, gradient, hessian = _penalized(prior, objective, log_smoothing, penalty, coefficients)



# ------------------------------------------------------- the Laplace evidence for the weights


def _curvature_trace_gradient(
    prior: ScaleMixturePrior, coefficients: F64Array, cavity: Cavity, weight: F64Array, working_bytes: int
) -> F64Array:
    """d tr(W (-H(x))) / dx for a fixed symmetric W: the third derivatives of sum_j log Z_j contracted with W.

    With W = (-H)^-1 it is d log|-H| / dx. Computed in z with Sigma_z = M W M' and carried back by M'. Variant j
    depends on z through a = (eta_c, e_j), e_j = log u_j, with
    log Z_j = LSE(eta_c + L_j(e)) - LSE(eta_c). With w = softmax(eta_c + L),
    Q = diag(w) - w w', M_j = J_j Sigma_z J_j' and N = [I, L'] M_j [I, L']', the
    gradient of tr(M_j Hess) is
        in eta:  Q [diag N - 2 N w + M_ee L'']
        in e:    L'.Q[...] + 2 (M_pe + L' M_ee).Q L'' + M_ee w.L'''.
    The -LSE(eta_c) term adds p_c Q_pi (diag M_pp - 2 M_pp pi) in eta (its Hessian enters -H with a plus sign).
    """
    mapping = prior.coefficient_map
    covariance_z = mapping @ weight @ mapping.T
    grid_size = prior.grid_size
    scale_span = slice(prior.density_size, prior.density_size + prior.scale_size)
    scale_covariance = covariance_z[scale_span, scale_span]
    gradient_z = np.zeros(covariance_z.shape[0])
    density_gradient = np.zeros((prior.class_count, grid_size))
    for class_position, rows, terms in _class_terms(prior, coefficients, cavity, working_bytes):
        span = slice(class_position * grid_size, (class_position + 1) * grid_size)
        density_density = covariance_z[span, span]
        density_scale = covariance_z[span, scale_span]
        weights = terms.responsibility
        first, second, third = terms.first, terms.second, terms.third
        design = prior.scale_design[rows]
        scale_scale = np.sum((design @ scale_covariance) * design, axis=1)
        density_row = design @ density_scale.T
        first_dot_weights = np.sum(first * weights, axis=1)
        density_row_dot_weights = np.sum(density_row * weights, axis=1)
        diagonal = np.diag(density_density)[None, :] + 2.0 * density_row * first + scale_scale[:, None] * np.square(first)
        applied = (
            weights @ density_density
            + density_row * first_dot_weights[:, None]
            + first * density_row_dot_weights[:, None]
            + scale_scale[:, None] * first * first_dot_weights[:, None]
        )
        inner = diagonal - 2.0 * applied + scale_scale[:, None] * second
        weighted_inner = weights * (inner - np.sum(weights * inner, axis=1)[:, None])
        weighted_second = weights * (second - np.sum(weights * second, axis=1)[:, None])
        scale_gradient = (
            np.sum(first * weighted_inner, axis=1)
            + 2.0 * np.sum((density_row + first * scale_scale[:, None]) * weighted_second, axis=1)
            + scale_scale * np.sum(weights * third, axis=1)
        )
        density_gradient[class_position] -= weighted_inner.sum(axis=0)
        gradient_z[scale_span] -= design.T @ scale_gradient
    density = np.exp(class_log_density(prior, coefficients))
    for class_position, class_rows in enumerate(prior.class_rows):
        span = slice(class_position * grid_size, (class_position + 1) * grid_size)
        density_density = covariance_z[span, span]
        class_density = density[class_position]
        prior_inner = np.diag(density_density) - 2.0 * density_density @ class_density
        gradient_z[span] = density_gradient[class_position] + class_rows.shape[0] * class_density * (prior_inner - float(class_density @ prior_inner))
    return mapping.T @ gradient_z


@dataclass(frozen=True)
class _Evidence:
    """V and dV/drho at x_rho; ``responses`` is dx_rho/drho (coefficients x weights), the first-order predictor of x."""

    value: float
    gradient: F64Array
    coefficients: F64Array
    responses: F64Array
    penalized_value: float
    newton_decrement: float
    magnitude: float


def _range_projector(matrix: F64Array) -> tuple[F64Array, F64Array]:
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    kept = eigenvalues > _EPSILON * matrix.shape[0] * max(float(eigenvalues[-1]), np.finfo(np.float64).tiny)
    return eigenvalues[kept], eigenvectors[:, kept]


def _log_pseudo_determinant(matrix: F64Array) -> float:
    eigenvalues, _vectors = _range_projector(matrix)
    return float(np.sum(np.log(eigenvalues)))


def _pseudo_inverse_trace(total: F64Array, part: F64Array) -> float:
    """tr(S^+ S_i): the number of the total penalty's dimensions that S_i accounts for."""
    eigenvalues, eigenvectors = _range_projector(total)
    return float(np.sum((eigenvectors.T @ part @ eigenvectors).diagonal() / eigenvalues))


def _cholesky_log_determinant_and_inverse(matrix: F64Array) -> tuple[float, F64Array]:
    factor = np.linalg.cholesky(matrix)
    inverse = np.linalg.solve(factor.T, np.linalg.solve(factor, np.eye(factor.shape[0])))
    return 2.0 * float(np.sum(np.log(np.diag(factor)))), inverse


def _penalty_groups(prior: ScaleMixturePrior) -> list[I64Array]:
    """The coordinates of each connected set of overlapping penalty blocks: log|S|_+ factors over them."""
    groups: list[set[int]] = []
    for block in prior.smoothing_blocks:
        coordinates = set(int(coordinate) for coordinate in block.coordinates)
        touching = [group for group in groups if group & coordinates]
        for group in touching:
            groups.remove(group)
            coordinates |= group
        groups.append(coordinates)
    return [np.array(sorted(group), dtype=np.int64) for group in groups]


def _evidence(
    prior: ScaleMixturePrior, log_smoothing: F64Array, start: F64Array, cavity: Cavity, working_bytes: int, tolerance: float
) -> _Evidence | None:
    """V(rho) and its exact gradient, with x_rho re-maximized from ``start``; None when x_rho is not a strict maximum.

    The Laplace curvature is the observed -H of the objective maximized. The null
    space N is profiled: -1/2 log|-H| + 1/2 log|N'(-H)N| keeps only the integrated
    directions. Both terms move with x_rho, through dx/drho_i = -(-H)^-1 lambda_i S_i x
    and the third derivatives of log Z (``_curvature_trace_gradient``).
    """
    penalty = _penalty_matrix(prior, log_smoothing)
    coefficients, objective = _maximize_coefficients(prior, log_smoothing, start, cavity, working_bytes, tolerance)
    value, gradient, hessian = _penalized(prior, objective, log_smoothing, penalty, coefficients)
    null_basis = prior.null_basis
    try:
        log_determinant, covariance = _cholesky_log_determinant_and_inverse(-hessian)
        null_log_determinant, null_inverse = _cholesky_log_determinant_and_inverse(null_basis.T @ -hessian @ null_basis)
    except np.linalg.LinAlgError:
        return None
    newton_decrement = 0.5 * float(gradient @ covariance @ gradient)
    penalty_log_determinant = sum(_log_pseudo_determinant(penalty[np.ix_(group, group)]) for group in _penalty_groups(prior))
    evidence_value = value + 0.5 * penalty_log_determinant - 0.5 * log_determinant + 0.5 * null_log_determinant
    # W = (-H)^-1 - N (N'(-H)N)^-1 N' carries both determinants' dependence on x.
    weight = covariance - null_basis @ null_inverse @ null_basis.T
    curvature_gradient = _curvature_trace_gradient(prior, coefficients, cavity, weight, working_bytes)
    evidence_gradient = np.empty(len(prior.smoothing_blocks))
    responses = np.empty((coefficients.shape[0], len(prior.smoothing_blocks)))
    group_of = {int(coordinate): group for group in _penalty_groups(prior) for coordinate in group}
    for position, (block, log_weight) in enumerate(zip(prior.smoothing_blocks, log_smoothing)):
        lambda_weight = float(np.exp(log_weight))
        coordinates = block.coordinates
        group = group_of[int(coordinates[0])]
        embedded = np.zeros((group.shape[0], group.shape[0]))
        inside = np.searchsorted(group, coordinates)
        embedded[np.ix_(inside, inside)] = block.matrix
        residual = block.factor @ coefficients[coordinates]
        pull = np.zeros_like(coefficients)
        pull[coordinates] = lambda_weight * (block.factor.T @ residual)
        # dx/drho_i = -(-H)^-1 pull, so -1/2 d tr through x is +1/2 grad . (-H)^-1 pull; N' S_i N = 0.
        responses[:, position] = -(covariance @ pull)
        evidence_gradient[position] = (
            -0.5 * lambda_weight * float(residual @ residual)
            + 0.5 * lambda_weight * _pseudo_inverse_trace(penalty[np.ix_(group, group)], embedded)
            - 0.5 * lambda_weight * float(np.sum(weight[np.ix_(coordinates, coordinates)] * block.matrix))
            + 0.5 * float(curvature_gradient @ (covariance @ pull))
        )
    return _Evidence(
        value=evidence_value,
        gradient=evidence_gradient,
        coefficients=coefficients,
        responses=responses,
        penalized_value=value,
        newton_decrement=newton_decrement,
        magnitude=objective.magnitude + abs(evidence_value),
    )


def _smoothing_bounds(prior: ScaleMixturePrior, objective: _Objective) -> list[tuple[float, float]]:
    """The range of rho that double precision resolves, per weight.

    With D the data curvature of the weight's block and s the smallest nonzero
    eigenvalue of S_i: below lambda s = sqrt(eps) ||D|| the weakest penalized
    direction of -H has a condition number past 1/sqrt(eps), and above
    lambda s = ||D|| / sqrt(eps) every penalized coefficient is below sqrt(eps)
    of its unpenalized size, so the fit is the null-space fit to half of
    double precision.
    """
    data = -(prior.coefficient_map.T @ objective.hessian @ prior.coefficient_map)
    bounds = []
    for block in prior.smoothing_blocks:
        block_data = data[np.ix_(block.coordinates, block.coordinates)]
        data_norm = max(float(np.max(np.abs(np.linalg.eigvalsh(0.5 * (block_data + block_data.T))))), np.finfo(np.float64).tiny)
        penalty_eigenvalues, _vectors = _range_projector(block.matrix)
        bounds.append((
            float(np.log(_HALF_PRECISION * data_norm / penalty_eigenvalues[0])),
            float(np.log(data_norm / (_HALF_PRECISION * penalty_eigenvalues[0]))),
        ))
    return bounds


def _restricted_prior(prior: ScaleMixturePrior, infinite: frozenset[int], zero: frozenset[int]) -> tuple[ScaleMixturePrior, F64Array]:
    """The prior with the weights in ``infinite`` at lambda = infinity and those in ``zero`` at lambda = 0, and the
    basis K of the allowed x.

    At lambda_i = infinity block i's penalized directions are exactly zero, so x = K z with K an orthonormal
    basis of every such block's null space; at lambda_i = 0 block i is absent. The other blocks act on z
    through their factors times K, and any direction none of them penalizes is profiled.
    """
    constraints = [np.zeros((0, prior.coefficient_size))]
    for position in sorted(infinite):
        block = prior.smoothing_blocks[position]
        embedded = np.zeros((block.factor.shape[0], prior.coefficient_size))
        embedded[:, block.coordinates] = block.factor
        constraints.append(embedded)
    constraint = np.vstack(constraints)
    if constraint.shape[0]:
        _left, singular, right = np.linalg.svd(constraint)
        rank = int(np.sum(singular > _EPSILON * max(constraint.shape) * float(singular[0])))
        allowed = right[rank:].T
    else:
        allowed = np.eye(prior.coefficient_size)
    blocks = []
    for position, block in enumerate(prior.smoothing_blocks):
        if position in infinite or position in zero:
            continue
        embedded = np.zeros((block.factor.shape[0], prior.coefficient_size))
        embedded[:, block.coordinates] = block.factor
        blocks.append(SmoothingBlock(block.name, np.arange(allowed.shape[1]), embedded @ allowed))
    total = sum((block.matrix for block in blocks), np.zeros((allowed.shape[1], allowed.shape[1])))
    null_eigenvalues, null_vectors = np.linalg.eigh(total)
    scale = max(float(null_eigenvalues[-1]) if null_eigenvalues.size else 0.0, np.finfo(np.float64).tiny)
    null_basis = null_vectors[:, null_eigenvalues <= _EPSILON * allowed.shape[1] * scale]
    view = replace(prior, coefficient_map=prior.coefficient_map @ allowed, smoothing_blocks=tuple(blocks), null_basis=null_basis)
    return view, allowed


def _ascend_evidence(
    prior: ScaleMixturePrior,
    start_weights: F64Array,
    start: _Evidence,
    cavity: Cavity,
    working_bytes: int,
    lower: F64Array,
    upper: F64Array,
    tolerance: float,
    flat_start: F64Array,
) -> tuple[F64Array, _Evidence]:
    """Trust-region quasi-Newton ascent of V(rho) inside [lower, upper], from a certified evidence ``start``.

    The model is V's gradient with a BFGS approximation of -V's Hessian; each trial maximizes it inside a
    radius, projected onto the bounds, and the radius follows the ratio of actual to predicted gain
    (Nocedal and Wright, Algorithm 4.1). A trial's x starts from the first-order predictor
    x_rho + (dx/drho) delta; a trial whose inner answer is not a certified maximum counts as V = -infinity.
    It stops when the model's predicted gain falls to ``tolerance`` nats, or the radius to half of double
    precision.
    """
    weights = np.clip(start_weights, lower, upper)
    current = start
    hessian = np.eye(weights.shape[0])
    radius = float(np.linalg.norm(current.gradient))
    while True:
        gradient = current.gradient
        free = ~(((weights <= lower) & (gradient < 0.0)) | ((weights >= upper) & (gradient > 0.0)))
        if not np.any(free):
            return weights, current
        free_hessian = hessian[np.ix_(free, free)]
        if 0.5 * float(gradient[free] @ np.linalg.solve(free_hessian, gradient[free])) <= max(tolerance, _EPSILON * current.magnitude):
            return weights, current
        if radius <= _HALF_PRECISION * (1.0 + float(np.max(np.abs(weights)))):
            return weights, current
        step = np.zeros_like(weights)
        step[free] = _trust_region_step(free_hessian, gradient[free], radius)
        trial_weights = np.clip(weights + step, lower, upper)
        step = trial_weights - weights
        predicted = float(gradient @ step) - 0.5 * float(step @ hessian @ step)
        trial = _certified_evidence(prior, trial_weights, current.coefficients + current.responses @ step, cavity, working_bytes, tolerance, flat_start)
        actual = -np.inf if trial is None else trial.value - current.value
        ratio = actual / predicted if predicted > 0.0 else -np.inf
        step_norm = float(np.linalg.norm(step))
        if not np.isfinite(ratio) or ratio < 0.25:
            radius = 0.25 * step_norm
        elif ratio > 0.75 and step_norm >= radius * (1.0 - _HALF_PRECISION):
            radius = 2.0 * radius
        if trial is None or actual <= 0.0:
            continue
        gradient_change = gradient - trial.gradient
        curvature = float(step @ gradient_change)
        if curvature > 0.0:
            image = hessian @ step
            hessian = hessian - np.outer(image, image) / float(step @ image) + np.outer(gradient_change, gradient_change) / curvature
        weights, current = trial_weights, trial


def _certified_evidence(
    prior: ScaleMixturePrior, log_smoothing: F64Array, start: F64Array, cavity: Cavity, working_bytes: int, tolerance: float, flat_start: F64Array
) -> _Evidence | None:
    """V at the certified inner maximum from ``start``, or from the flat start when that is not a certified
    maximum (the two structural starts; lead ruling); None when neither is."""
    warm = _evidence(prior, log_smoothing, start, cavity, working_bytes, tolerance)
    return warm if warm is not None else _evidence(prior, log_smoothing, flat_start, cavity, working_bytes, tolerance)


def _best_certified(
    prior: ScaleMixturePrior, log_smoothing: F64Array, starts: Sequence[F64Array], cavity: Cavity, working_bytes: int, tolerance: float
) -> _Evidence | None:
    """The certified inner maximum with the highest V over the given starts."""
    candidates = [_evidence(prior, log_smoothing, start, cavity, working_bytes, tolerance) for start in starts]
    certified = [candidate for candidate in candidates if candidate is not None]
    return max(certified, key=lambda candidate: candidate.value) if certified else None


def _maximize_evidence(
    prior: ScaleMixturePrior,
    start_weights: F64Array,
    start_coefficients: F64Array,
    cavity: Cavity,
    working_bytes: int,
    bounds: list[tuple[float, float]],
    tolerance: float,
) -> tuple[F64Array, F64Array, _Evidence, _Evidence]:
    """Maximize V over every weight in [0, infinity], with exact edges at both ends.

    A weight whose ascent reaches the top of its resolvable range (where its penalized directions are
    frozen to half of double precision) moves to lambda = infinity exactly: x is confined to that block's
    null space. One that reaches the bottom (where its penalty is lost in the data curvature's rounding)
    moves to lambda = 0 exactly: the block is dropped. An edge weight is released when V at the end of
    its range falls toward the edge. Returns the log weights (+inf and -inf at the edges), x in full
    coordinates, V there, and V at the start.
    """
    lower = np.array([bound[0] for bound in bounds])
    upper = np.array([bound[1] for bound in bounds])
    infinite = frozenset(int(position) for position in np.flatnonzero(start_weights == np.inf))
    zero = frozenset(int(position) for position in np.flatnonzero(start_weights == -np.inf))
    weights = np.where(start_weights == np.inf, upper, np.where(start_weights == -np.inf, lower, start_weights))
    coefficients = np.array(start_coefficients, dtype=np.float64, copy=True)
    flat = initial_hyperparameters(prior).coefficients
    start = None
    while True:
        edges = infinite | zero
        finite = np.array([position for position in range(len(bounds)) if position not in edges], dtype=np.int64)
        view, allowed = _restricted_prior(prior, infinite, zero)
        finite_start = np.clip(weights[finite], lower[finite], upper[finite])
        evidence = _best_certified(view, finite_start, [allowed.T @ coefficients, allowed.T @ flat], cavity, working_bytes, tolerance)
        if evidence is None:
            raise FloatingPointError("neither structural start reaches a certified maximum at the starting penalty weights")
        if start is None:
            start = evidence
        finite_weights, evidence = _ascend_evidence(
            view, finite_start, evidence, cavity, working_bytes, lower[finite], upper[finite], tolerance, allowed.T @ flat
        )
        # Refit at the chosen weights from the best certified answer the search saw and from the flat start.
        evidence = _best_certified(view, finite_weights, [evidence.coefficients, allowed.T @ flat], cavity, working_bytes, tolerance) or evidence
        weights[finite] = finite_weights
        coefficients = allowed @ evidence.coefficients
        to_infinity = {int(finite[index]) for index in np.flatnonzero((finite_weights >= upper[finite]) & (evidence.gradient >= 0.0))}
        to_zero = {int(finite[index]) for index in np.flatnonzero((finite_weights <= lower[finite]) & (evidence.gradient <= 0.0))}
        released = set()
        for position in sorted(edges):
            loose_infinite, loose_zero = infinite - {position}, zero - {position}
            loose_view, loose_allowed = _restricted_prior(prior, loose_infinite, loose_zero)
            loose_finite = [index for index in range(len(bounds)) if index not in loose_infinite | loose_zero]
            trial_weights = weights.copy()
            at_infinity = position in infinite
            trial_weights[position] = upper[position] if at_infinity else lower[position]
            trial = _certified_evidence(
                loose_view, trial_weights[loose_finite], loose_allowed.T @ coefficients, cavity, working_bytes, tolerance, loose_allowed.T @ flat
            )
            if trial is None:
                continue
            slope = trial.gradient[loose_finite.index(position)]
            if (at_infinity and slope < 0.0) or (not at_infinity and slope > 0.0):
                released.add(position)
        if not to_infinity and not to_zero and not released:
            log_smoothing = weights.copy()
            log_smoothing[sorted(infinite)] = np.inf
            log_smoothing[sorted(zero)] = -np.inf
            return log_smoothing, coefficients, evidence, start
        infinite = frozenset((infinite | to_infinity) - released)
        zero = frozenset((zero | to_zero) - released)


def hyper_step(
    prior: ScaleMixturePrior, hyperparameters: MixtureHyperparameters, cavity: Cavity, working_bytes: int, tolerance: float
) -> HyperStep:
    """Maximize V over every penalty weight in [0, infinity], with x at the penalized maximum for each, to ``tolerance``
    nats: the resolution the fit certifies (1/(2K) for a scorer with K posterior draws)."""
    start_objective = _data_objective(prior, hyperparameters.coefficients, cavity, working_bytes)
    infinite = frozenset(int(position) for position in np.flatnonzero(hyperparameters.log_smoothing == np.inf))
    zero = frozenset(int(position) for position in np.flatnonzero(hyperparameters.log_smoothing == -np.inf))
    view, allowed = _restricted_prior(prior, infinite, zero)
    finite = np.isfinite(hyperparameters.log_smoothing)
    start_coefficients = allowed.T @ hyperparameters.coefficients
    _value, start_gradient, start_hessian = _penalized(
        view, _data_objective(view, start_coefficients, cavity, working_bytes), hyperparameters.log_smoothing[finite],
        _penalty_matrix(view, hyperparameters.log_smoothing[finite]), start_coefficients,
    )
    start_decrement = 0.5 * float(start_gradient @ _ascent_direction(-start_hessian, start_gradient))
    bounds = _smoothing_bounds(prior, start_objective)
    log_smoothing, coefficients, evidence, start_evidence = _maximize_evidence(
        prior, hyperparameters.log_smoothing, hyperparameters.coefficients, cavity, working_bytes, bounds, tolerance
    )
    interior = np.array([np.isfinite(weight) and bound[0] < weight < bound[1] for weight, bound in zip(log_smoothing, bounds)])
    interior_gradient = evidence.gradient[interior[np.isfinite(log_smoothing)]]
    return HyperStep(
        hyperparameters=MixtureHyperparameters(coefficients=coefficients, log_smoothing=log_smoothing),
        penalized_objective=evidence.penalized_value,
        evidence=evidence.value,
        newton_decrement=evidence.newton_decrement,
        smoothing_gradient=float(np.max(np.abs(interior_gradient))) if interior_gradient.size else 0.0,
        start_decrement=start_decrement,
        evidence_gain=evidence.value - start_evidence.value,
    )
