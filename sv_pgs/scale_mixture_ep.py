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
  the integral of eta'''(t)^2 (``ROUGHNESS_ORDER``), is penalized in its
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
from typing import Callable, Iterator, Sequence

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.special import erfcx, logsumexp

from sv_pgs._typing import F64Array, I64Array

_EPSILON = float(np.finfo(np.float64).eps)
# Half of double precision: the resolution of a quantity whose square is compared at eps.
_HALF_PRECISION = _EPSILON**0.5
# The mixing density's roughness penalty: the integral of its squared derivative of this order, with a learned
# weight. Third order is derived: its null space, the quadratics (a normal density in log s), is the only lambda =
# infinity limit that is proper on the real line and does not move with the range (<= 1e-4 nats when the range
# doubles or quadruples; the first- and second-order limits move by 4-100 nats: math-density, lead ruling).
ROUGHNESS_ORDER = 3
# The (rows x K) float64 arrays alive at once in a chunk: the ten of ``_Components``, the seven directional-derivative
# arrays of ``_directional_derivatives`` and three expression temporaries.
_ROW_INTERMEDIATES = 20
# QUADPACK's relative accuracy is bounded below by 50 eps (scipy.integrate.quad raises under it).
_QUADPACK_RELATIVE_FLOOR = 50.0 * _EPSILON


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
    if lattice.shape[0] <= ROUGHNESS_ORDER or spacing[0] <= 0.0 or not np.allclose(spacing, spacing[0], rtol=_HALF_PRECISION, atol=0.0):
        raise ValueError("the lattice must be evenly spaced increasing nodes, more than the roughness order")
    grid_size = lattice.shape[0]
    basis = _sum_to_zero_basis(grid_size)
    roughness = roughness_factor(grid_size, float(spacing[0]), ROUGHNESS_ORDER) @ basis
    null_functionals = _legendre_functionals(lattice, floor, top, ROUGHNESS_ORDER - 1) @ basis
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
    blocks = [SmoothingBlock("pooled roughness", pooled, roughness)]
    for class_position in range(deviation_count):
        deviation = pooled_size + class_position * pooled_size + np.arange(pooled_size)
        blocks.append(SmoothingBlock(f"class {class_position} deviation roughness", deviation, roughness))
    if deviation_count:
        # A deviation's location and width (its part the penalty cannot see) have a Gaussian pooling prior with
        # mean zero and one learned precision: eta_bar carries the common location and width.
        blocks.append(SmoothingBlock(
            "deviation location and width", np.arange(pooled_size, pooled_size + deviation_size), np.kron(np.eye(deviation_count), null_functionals)
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


def relattice(
    prior: ScaleMixturePrior, hyperparameters: MixtureHyperparameters, nodes: F64Array, floor: float, top: float
) -> tuple[ScaleMixturePrior, MixtureHyperparameters]:
    """The same model on new nodes and a new kernel range: each class's eta from the cubic through the old
    nodes inside the old lattice, and along its end slopes outside it, so the log-tails stay linear (which the
    third-order penalty leaves free) rather than following a cubic's extrapolation."""
    old_nodes = prior.log_variance_grid
    log_density, scale_coefficients = _density_and_scale(prior, hyperparameters.coefficients)
    spline = CubicSpline(old_nodes, log_density.T)
    new_nodes = np.asarray(nodes, dtype=np.float64)
    inside = np.clip(new_nodes, old_nodes[0], old_nodes[-1])
    slopes = np.where(new_nodes < old_nodes[0], spline(old_nodes[0], 1)[:, None], spline(old_nodes[-1], 1)[:, None])
    new_density = spline(inside).T + slopes * (new_nodes - inside)[None, :]
    moved = scale_mixture_prior(
        class_index=prior.class_index,
        log_variance_offset=prior.log_variance_offset,
        annotation_design=prior.scale_design,
        annotation_groups=prior.annotation_groups,
        nodes=new_nodes,
        floor=floor,
        top=top,
    )
    normalized = new_density - new_density.mean(axis=1, keepdims=True)
    target = np.concatenate([normalized.ravel(), scale_coefficients])
    coefficients = np.linalg.lstsq(moved.coefficient_map, target, rcond=None)[0]
    return moved, MixtureHyperparameters(coefficients=coefficients, log_smoothing=hyperparameters.log_smoothing.copy())


def halved_lattice(prior: ScaleMixturePrior, hyperparameters: MixtureHyperparameters) -> tuple[ScaleMixturePrior, MixtureHyperparameters]:
    """The same model on the lattice with half the spacing over the same extent and kernel range."""
    nodes = prior.log_variance_grid
    return relattice(prior, hyperparameters, np.linspace(nodes[0], nodes[-1], 2 * nodes.shape[0] - 1), prior.kernel_floor, prior.kernel_top)


def prior_second_moment(prior: ScaleMixturePrior, hyperparameters: MixtureHyperparameters) -> F64Array:
    """E[beta_j^2] under the prior: u_j sum_k pi_ck e^t_k."""
    log_density = class_log_density(prior, hyperparameters.coefficients)
    log_mean_variance = logsumexp(log_density + prior.log_variance_grid[None, :], axis=1)
    return np.exp(log_scale(prior, hyperparameters.coefficients) + log_mean_variance[prior.class_index])


def _row_chunks(rows: I64Array, grid_size: int, working_bytes: int) -> Iterator[I64Array]:
    """Pieces of ``rows`` whose (rows x K) float64 intermediates (``_ROW_INTERMEDIATES`` of them) fit ``working_bytes``."""
    if working_bytes <= 0:
        raise ValueError("working_bytes must be positive")
    chunk = max(1, int(working_bytes) // (_ROW_INTERMEDIATES * np.dtype(np.float64).itemsize * max(grid_size, 1)))
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
    fourth: F64Array


def _kernel_terms(
    log_density: F64Array, log_scale_rows: F64Array, grid: F64Array, floor: float, precision: F64Array, shift: F64Array
) -> tuple[F64Array, F64Array, F64Array, F64Array, F64Array]:
    """(v r = v/(1 + vP), r = 1/(1 + vP), qr = vP/(1 + vP), log pi_k + log Z_jk, h^2 v r) at every node; v = 0 below
    ``floor``. Written so that an overflowing v (a node far past every effect's scale) gives its limits
    v r = 1/P, r = 0, qr = 1 and a component of weight zero, not inf * 0."""
    # An overflowing v and a flat node's v = 0 are both meant: the reciprocal forms below take their limits exactly.
    with np.errstate(over="ignore", divide="ignore"):
        variance = np.where(grid[None, :] >= floor, np.exp(log_scale_rows[:, None] + grid[None, :]), 0.0)
        column_precision = precision[:, None]
        ratio = variance * column_precision
        if np.any(ratio <= -1.0):
            raise FloatingPointError("a cavity is improper on the lattice: 1 + v P <= 0")
        retained = 1.0 / (1.0 + ratio)
        ratio_retained = 1.0 / (1.0 + 1.0 / ratio)
        conditional = 1.0 / (1.0 / variance + column_precision)
    signal = np.square(shift)[:, None] * conditional
    log_component = log_density - 0.5 * np.log1p(ratio) + 0.5 * signal
    return conditional, retained, ratio_retained, log_component, signal


def _log_normalizers(
    log_density: F64Array, log_scale_rows: F64Array, grid: F64Array, floor: float, precision: F64Array, shift: F64Array
) -> F64Array:
    return logsumexp(_kernel_terms(log_density, log_scale_rows, grid, floor, precision, shift)[3], axis=1)


def _components(
    log_density: F64Array, log_scale_rows: F64Array, grid: F64Array, floor: float, precision: F64Array, shift: F64Array
) -> _Components:
    """With q = vP, r = 1/(1+q) and a = h^2 v r (so dr/deta = -r(1 - r) and da/deta = a r), each derivative of
    log Z_k in eta = log u is a A_n(r) - B_n(r): d1 = (a - q) r / 2, then A_(n+1) = r A_n - r(1 - r) A_n' and
    B_(n+1) = -r(1 - r) B_n', giving A_2 = r^2 - r/2, B_2 = r(1 - r)/2, A_3 = 3r^3 - 3r^2 + r/2,
    B_3 = r(1 - r)(2r - 1)/2, and A_4 = r A_3 - r(1 - r)(9r^2 - 6r + 1/2), B_4 = -r(1 - r)(-3r^2 + 3r - 1/2).
    Nodes below ``floor`` have a flat kernel: v = 0 there."""
    conditional, retained, ratio_retained, log_component, signal = _kernel_terms(log_density, log_scale_rows, grid, floor, precision, shift)
    log_normalizer = logsumexp(log_component, axis=1)
    responsibility = np.exp(log_component - log_normalizer[:, None])
    return _Components(
        log_normalizer=log_normalizer,
        responsibility=responsibility,
        conditional_variance=conditional,
        first=0.5 * (retained * signal - ratio_retained),
        second=0.5 * signal * retained * (2.0 * retained - 1.0) - 0.5 * ratio_retained * retained,
        third=0.5 * signal * retained * (6.0 * retained * retained - 6.0 * retained + 1.0)
        - 0.5 * ratio_retained * retained * (2.0 * retained - 1.0),
        fourth=signal * (
            retained * (3.0 * retained**3 - 3.0 * retained**2 + 0.5 * retained)
            - retained * (1.0 - retained) * (9.0 * retained**2 - 6.0 * retained + 0.5)
        )
        + retained * (1.0 - retained) * (-3.0 * retained**2 + 3.0 * retained - 0.5),
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



@dataclass(frozen=True)
class GaussianPosterior:
    """q's linear responses at the EP fixed point, for the total curvature B: ``solve(R)`` is Sigma R and
    ``variance_jvp(W)`` is -(Sigma o Sigma) W, both (p x r). Stage 2 answers them with extra right-hand sides of its
    solve and with ``marginal_variances.variance_jvp``."""

    solve: Callable[[F64Array], F64Array]
    variance_jvp: Callable[[F64Array], F64Array]


PosteriorAt = Callable[[ScaleMixturePrior, F64Array], GaussianPosterior]
"""q's responses at the EP fixed point for a prior's coefficients: fixed at Stage 2's fixed point, exact for normal means."""


def normal_means_posterior(cavity: Cavity, working_bytes: int) -> PosteriorAt:
    """For independent effects q's marginal is the tilted law itself, so Sigma is the diagonal of tilted variances."""

    def posterior_at(prior: ScaleMixturePrior, coefficients: F64Array) -> GaussianPosterior:
        hyperparameters = MixtureHyperparameters(coefficients=coefficients, log_smoothing=np.zeros(len(prior.smoothing_blocks)))
        return diagonal_posterior(tilted_moments(prior, hyperparameters, cavity, working_bytes).variance)

    return posterior_at


def diagonal_posterior(variance: F64Array) -> GaussianPosterior:
    """The posterior of independent effects (orthogonal design, normal means): Sigma = diag(variance). There the
    cavities do not move with the prior, so B equals the fixed-cavity curvature."""
    column = np.asarray(variance, dtype=np.float64)[:, None]
    return GaussianPosterior(solve=lambda right: column * right, variance_jvp=lambda weights: -np.square(column) * weights)


@dataclass(frozen=True)
class _VariantDerivatives:
    """Per variant at its cavity: the tilted mean m, variance v and second moment s2, their derivatives in the
    cavity shift (v_h) and precision (m_P, v_P), and in z: per node of the class density (m_eta, s2_eta, p x K)
    and in log u (m_logu, s2_logu), which the scale-design row carries to the scale coefficients."""

    mean: F64Array
    variance: F64Array
    second: F64Array
    variance_by_shift: F64Array
    mean_by_precision: F64Array
    variance_by_precision: F64Array
    mean_by_density: F64Array
    second_by_density: F64Array
    mean_by_log_scale: F64Array
    second_by_log_scale: F64Array


def _variant_derivatives(prior: ScaleMixturePrior, coefficients: F64Array, cavity: Cavity, working_bytes: int) -> _VariantDerivatives:
    """Given node k the tilted law is N(mu_k, c_k) with mu_k = h c_k; with weights w and ell_k = -(c_k + mu_k^2)/2
    (d log Z_k / dP), f_k = d log Z_k / d eta and dc_k / d eta = c_k r_k (r_k = 1/(1 + v_k P)):
        v_h = E[(mu - m)^3] + 3 E[c (mu - m)],   m_P = Cov(ell, mu) - E[mu c],   s2_P = Cov(ell, c + mu^2) - E[c^2 + 2 mu^2 c],
        m_eta = w (mu - m),   s2_eta = w (c + mu^2 - s2),
        m_logu = Cov(f, mu) + E[mu r],   s2_logu = Cov(f, c + mu^2) + E[(c + 2 mu^2) r]
    (the eta derivatives are the same in normalized and unnormalized coordinates, since sum_k w_k (mu_k - m) = 0)."""
    variant_count, grid_size = prior.variant_count, prior.grid_size
    fields = {name: np.empty(variant_count) for name in (
        "mean", "variance", "second", "variance_by_shift", "mean_by_precision", "variance_by_precision", "mean_by_log_scale", "second_by_log_scale",
    )}
    mean_by_density = np.empty((variant_count, grid_size))
    second_by_density = np.empty((variant_count, grid_size))
    scales = log_scale(prior, coefficients)
    for _class, rows, terms in _class_terms(prior, coefficients, cavity, working_bytes):
        weights = terms.responsibility
        conditional = terms.conditional_variance
        retained = _kernel_terms(
            class_log_density(prior, coefficients)[_class], scales[rows], prior.log_variance_grid, prior.kernel_floor, cavity.precision[rows], cavity.shift[rows]
        )[1]
        centre = cavity.shift[rows][:, None] * conditional

        def expectation(values: F64Array) -> F64Array:
            return np.sum(weights * values, axis=1)

        def covariance(left: F64Array, right: F64Array) -> F64Array:
            return expectation(left * right) - expectation(left) * expectation(right)

        mean = expectation(centre)
        raw_second = conditional + centre * centre
        second = expectation(raw_second)
        deviation = centre - mean[:, None]
        slope = -0.5 * raw_second
        mean_by_precision = covariance(slope, centre) - expectation(centre * conditional)
        second_by_precision = covariance(slope, raw_second) - expectation(conditional * conditional + 2.0 * centre * centre * conditional)
        fields["mean"][rows] = mean
        fields["second"][rows] = second
        fields["variance"][rows] = second - mean * mean
        fields["variance_by_shift"][rows] = expectation(deviation**3) + 3.0 * expectation(conditional * deviation)
        fields["mean_by_precision"][rows] = mean_by_precision
        fields["variance_by_precision"][rows] = second_by_precision - 2.0 * mean * mean_by_precision
        fields["mean_by_log_scale"][rows] = covariance(terms.first, centre) + expectation(centre * retained)
        fields["second_by_log_scale"][rows] = covariance(terms.first, raw_second) + expectation((conditional + 2.0 * centre * centre) * retained)
        mean_by_density[rows] = weights * deviation
        second_by_density[rows] = weights * (raw_second - second[:, None])
    return _VariantDerivatives(mean_by_density=mean_by_density, second_by_density=second_by_density, **fields)


def _through_z(prior: ScaleMixturePrior, by_density: F64Array, by_log_scale: F64Array, directions_z: F64Array) -> F64Array:
    """(p x r): each variant's derivative in z applied to the directions (C K + L) x r."""
    grid_size = prior.grid_size
    density_part = directions_z[: prior.density_size].reshape(prior.class_count, grid_size, -1)
    return np.einsum("jk,jkr->jr", by_density, density_part[prior.class_index]) + by_log_scale[:, None] * (
        prior.scale_design @ directions_z[prior.density_size :]
    )


def _through_z_transposed(prior: ScaleMixturePrior, by_density: F64Array, by_log_scale: F64Array, values: F64Array) -> F64Array:
    """(C K + L) x r: the transpose of ``_through_z`` applied to (p x r) values."""
    grid_size = prior.grid_size
    result = np.zeros((prior.density_size + prior.scale_size, values.shape[1]))
    for class_position, rows in enumerate(prior.class_rows):
        result[class_position * grid_size : (class_position + 1) * grid_size] = by_density[rows].T @ values[rows]
    result[prior.density_size :] = prior.scale_design.T @ (by_log_scale[:, None] * values)
    return result


def _total_curvature(
    prior: ScaleMixturePrior, coefficients: F64Array, cavity: Cavity, posterior: GaussianPosterior, working_bytes: int, relative_tolerance: float
) -> F64Array:
    """B = -d2 log Z_EP / dx2 with EP re-solved, in x: M' B_z M (speed-ep, B_PRODUCTS.md), without re-solving EP.

    B_z E = A E - m_x' dh + s2_x' dP / 2 (d grad_x log Z_j / dh_j = m_x and d grad_x log Z_j / dP_j = -s2_x / 2, with
    B the negative derivative), where the cavity response (dh, dP) to a direction E solves the linear response of the
    EP fixed point:
        (Q + diag tau) dm = (m + m_P / v) dP + m_x E / v          (``posterior.solve``)
        dh = (dm - m_P dP - m_x E) / v
        v^2 dP = -(Sigma o Sigma)_off (dv / v^2 + dP),  dv = v_h dh + v_P dP + v_x E   (``posterior.variance_jvp``)
    dP is the fixed point of that affine map, found by GMRES on all directions at once to ``relative_tolerance``;
    Krylov is exact within its dimension, which bounds the work.
    """
    derivatives = _variant_derivatives(prior, coefficients, cavity, working_bytes)
    directions = prior.coefficient_map
    mean_by_z = _through_z(prior, derivatives.mean_by_density, derivatives.mean_by_log_scale, directions)
    variance_by_z = _through_z(prior, derivatives.second_by_density, derivatives.second_by_log_scale, directions) - 2.0 * derivatives.mean[:, None] * mean_by_z
    variance = derivatives.variance[:, None]

    def through(precision_step: F64Array) -> tuple[F64Array, F64Array]:
        mean_step = posterior.solve((derivatives.mean + derivatives.mean_by_precision / derivatives.variance)[:, None] * precision_step + mean_by_z / variance)
        shift_step = (mean_step - derivatives.mean_by_precision[:, None] * precision_step - mean_by_z) / variance
        variance_step = derivatives.variance_by_shift[:, None] * shift_step + derivatives.variance_by_precision[:, None] * precision_step + variance_by_z
        response = variance_step / variance**2 + precision_step
        return shift_step, response + posterior.variance_jvp(response) / variance**2, response

    shape = mean_by_z.shape
    _shift, offset, start_response = through(np.zeros(shape))
    # The map's last step cancels the diagonal of Sigma o Sigma against v^2: its value is known only to eps times the
    # terms that cancel, which is where GMRES's residual can stop.
    rounding = _EPSILON * float(np.linalg.norm(start_response))

    def linear_part(vector: F64Array) -> F64Array:
        precision_step = vector.reshape(shape)
        return (precision_step - (through(precision_step)[1] - offset)).ravel()

    size = int(np.prod(shape))
    operator = LinearOperator((size, size), matvec=linear_part, dtype=np.float64)
    solution, information = gmres(operator, offset.ravel(), rtol=relative_tolerance, atol=rounding, restart=size, maxiter=size)
    if information != 0:
        raise FloatingPointError(f"the EP fixed point's linear response did not converge (gmres information {information})")
    precision_step = solution.reshape(shape)
    shift_step, _next, _response = through(precision_step)
    fixed_cavity = -_data_objective(prior, coefficients, cavity, working_bytes).hessian
    total_z = fixed_cavity @ directions - _through_z_transposed(prior, derivatives.mean_by_density, derivatives.mean_by_log_scale, shift_step) + 0.5 * (
        _through_z_transposed(prior, derivatives.second_by_density, derivatives.second_by_log_scale, precision_step)
    )
    total = directions.T @ total_z
    return 0.5 * (total + total.T)


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
    # newton_decrement is 1/2 g'(B + S)^-1 g: what the fit's certificate records (math-epeb: the fixed-cavity form
    # understates the remaining gain by up to 100x).


def _directional_derivatives(
    prior: ScaleMixturePrior, coefficients: F64Array, cavity: Cavity, directions: F64Array, working_bytes: int
) -> tuple[F64Array, F64Array]:
    """The third and fourth derivatives of sum_j log Z_j along each column of ``directions`` (in x), exactly.

    Along x + s b, variant j's natural parameters are theta_k(s) = eta_ck + s b_ck + L_jk(e_j + s b_e) with
    b_e = d_j' b_theta, and log Z_j = LSE(theta) - LSE(eta_c). With the node distribution w and
    theta' = b_c + L' b_e, theta'' = L'' b_e^2, theta''' = L''' b_e^3, theta'''' = L'''' b_e^4, the derivatives of a
    log-sum-exp are joint cumulants under w:
        d3 = k3(theta') + 3 cov(theta', theta'') + E theta''',
        d4 = k4(theta') + 6 k(theta', theta', theta'') + 3 var(theta'') + 4 cov(theta', theta''') + E theta''''.
    The -LSE(eta_c) term subtracts k3 and k4 of b_c under pi_c, once per variant of the class.
    """
    mapping = prior.coefficient_map
    directions_z = mapping @ directions
    grid_size = prior.grid_size
    scale_span = slice(prior.density_size, prior.density_size + prior.scale_size)
    third = np.zeros(directions.shape[1])
    fourth = np.zeros(directions.shape[1])
    density = np.exp(class_log_density(prior, coefficients))
    for class_position, rows, terms in _class_terms(prior, coefficients, cavity, working_bytes):
        weights = terms.responsibility
        design = prior.scale_design[rows]
        span = slice(class_position * grid_size, (class_position + 1) * grid_size)
        for column in range(directions.shape[1]):
            density_step = directions_z[span, column][None, :]
            scale_step = (design @ directions_z[scale_span, column])[:, None]
            first = density_step + terms.first * scale_step
            second = terms.second * scale_step**2
            third_term = terms.third * scale_step**3
            fourth_term = terms.fourth * scale_step**4
            centred = first - np.sum(weights * first, axis=1, keepdims=True)
            centred_second = second - np.sum(weights * second, axis=1, keepdims=True)
            centred_third = third_term - np.sum(weights * third_term, axis=1, keepdims=True)
            variance = np.sum(weights * centred**2, axis=1)
            third[column] += float(np.sum(
                np.sum(weights * centred**3, axis=1) + 3.0 * np.sum(weights * centred * centred_second, axis=1) + np.sum(weights * third_term, axis=1)
            ))
            fourth[column] += float(np.sum(
                np.sum(weights * centred**4, axis=1) - 3.0 * variance**2
                + 6.0 * np.sum(weights * centred**2 * centred_second, axis=1)
                + 3.0 * np.sum(weights * centred_second**2, axis=1)
                + 4.0 * np.sum(weights * centred * centred_third, axis=1)
                + np.sum(weights * fourth_term, axis=1)
            ))
    for class_position, class_rows in enumerate(prior.class_rows):
        span = slice(class_position * grid_size, (class_position + 1) * grid_size)
        class_density = density[class_position]
        for column in range(directions.shape[1]):
            step = directions_z[span, column]
            centred = step - float(class_density @ step)
            variance = float(class_density @ centred**2)
            third[column] -= class_rows.shape[0] * float(class_density @ centred**3)
            fourth[column] -= class_rows.shape[0] * (float(class_density @ centred**4) - 3.0 * variance**2)
    return third, fourth


def _laplace_corrections(
    prior: ScaleMixturePrior, log_smoothing: F64Array, evidence: _Evidence, cavity: Cavity, posterior_at: PosteriorAt, working_bytes: int, tolerance: float
) -> tuple[F64Array, F64Array, F64Array]:
    """Per integrated direction, the correction from the Laplace term to the exact one-dimensional integral, the
    Tierney-Kadane term that decided it, and the standardized directions themselves (columns, in x).

    The integrated directions are the eigenvectors of -H's Schur complement on the complement of the profiled null
    space, each moved with the null coordinates' first-order response and scaled to unit curvature. Along a
    standardized direction the Tierney-Kadane O(1) term is k4/8 + 5 k3^2/24; where its magnitude exceeds
    ``tolerance`` the Laplace term is replaced by an exact quadrature of the integrand along that line, and the
    correction is the log of their ratio (its limit covers a density collapsing to a point). Elsewhere it is 0.
    """
    penalty = _penalty_matrix(prior, log_smoothing)
    objective = _data_objective(prior, evidence.coefficients, cavity, working_bytes)
    value, _gradient, hessian = _penalized(prior, objective, log_smoothing, penalty, evidence.coefficients)
    negative = -hessian
    null_basis = prior.null_basis
    complement = np.linalg.svd(np.eye(negative.shape[0]) - null_basis @ null_basis.T)[0][:, : negative.shape[0] - null_basis.shape[1]]
    response = np.eye(negative.shape[0])
    if null_basis.shape[1]:
        response = response - null_basis @ np.linalg.solve(null_basis.T @ negative @ null_basis, null_basis.T @ negative)
    moved = response @ complement
    schur = moved.T @ negative @ moved
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (schur + schur.T))
    directions = moved @ eigenvectors / np.sqrt(eigenvalues)[None, :]
    third, fourth = _directional_derivatives(prior, evidence.coefficients, cavity, directions, working_bytes)
    terms = fourth / 8.0 + 5.0 * third**2 / 24.0
    corrections = np.zeros(directions.shape[1])
    for index in np.flatnonzero(np.abs(terms) > tolerance):
        direction = directions[:, index]

        def integrand(step: float) -> float:
            point = evidence.coefficients + step * direction
            return float(np.exp(
                _data_value(prior, point, cavity, working_bytes) - _penalty_value(prior, log_smoothing, point)[0] - value
            ))

        integral, error, _information, *message = quad(
            integrand, -np.inf, np.inf, epsabs=0.0, epsrel=max(tolerance, _QUADPACK_RELATIVE_FLOOR), full_output=True
        )
        # The log of the integral is what enters V: accept QUADPACK's answer when its own error estimate resolves that
        # log to the tolerance, or to half of double precision when rounding is what stopped it.
        if message and error > max(tolerance, _HALF_PRECISION) * abs(integral):
            raise FloatingPointError(f"the exact integral along a direction did not converge: {message[0]}")
        corrections[index] = float(np.log(integral) - 0.5 * np.log(2.0 * np.pi))
    return corrections, terms, directions


def _corrected_value(
    prior: ScaleMixturePrior, log_smoothing: F64Array, evidence: _Evidence, cavity: Cavity, posterior_at: PosteriorAt, working_bytes: int, tolerance: float
) -> float:
    """V with its per-direction Laplace terms replaced by exact one-dimensional integrals where they fail: the value
    basins and edges are compared by (lead ruling)."""
    corrections, _terms, _directions = _laplace_corrections(prior, log_smoothing, evidence, cavity, posterior_at, working_bytes, tolerance)
    return evidence.value + float(np.sum(corrections))


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
    prior: ScaleMixturePrior, log_smoothing: F64Array, start: F64Array, cavity: Cavity, posterior_at: PosteriorAt, working_bytes: int, tolerance: float
) -> _Evidence | None:
    """V(rho) with the total curvature B, and the fixed-cavity gradient that steers the search; x_rho re-maximized
    from ``start``. None when x_rho is not a strict maximum of the objective V integrates, i.e. B + S is not
    positive definite there (lead ruling: such a point is never accepted, and its V never reported).

    V = F + 1/2 log|S|_+ - 1/2 log|B + S| + 1/2 log|N'(B + S)N|: B = -d2 log Z_EP / dx2 with EP re-solved
    (``_total_curvature``), the second-order approximation of the actual marginal likelihood; the null space N is
    profiled. The gradient is that of the fixed-cavity form (-H in place of B), whose terms move with x_rho through
    dx/drho_i = -(-H)^-1 lambda_i S_i x and the third derivatives of log Z (``_curvature_trace_gradient``); it only
    chooses the search direction, and every step is accepted on V itself.
    """
    penalty = _penalty_matrix(prior, log_smoothing)
    null_basis = prior.null_basis
    # V's determinant terms move with x at first order: an inexact x-hat with decrement d moves V by up to
    # sqrt(c'(-H)^-1 c) sqrt(2 d) / 2, c their x-gradient. The inner tolerance tightens until that is below ``tolerance``.
    inner_tolerance = tolerance
    coefficients = np.array(start, dtype=np.float64, copy=True)
    while True:
        coefficients, objective = _maximize_coefficients(prior, log_smoothing, coefficients, cavity, working_bytes, inner_tolerance)
        value, gradient, hessian = _penalized(prior, objective, log_smoothing, penalty, coefficients)
        try:
            log_determinant, covariance = _cholesky_log_determinant_and_inverse(-hessian)
            null_log_determinant, null_inverse = _cholesky_log_determinant_and_inverse(null_basis.T @ -hessian @ null_basis)
        except np.linalg.LinAlgError:
            return None
        newton_decrement = 0.5 * float(gradient @ covariance @ gradient)
        weight = covariance - null_basis @ null_inverse @ null_basis.T
        curvature_gradient = _curvature_trace_gradient(prior, coefficients, cavity, weight, working_bytes)
        sensitivity = max(float(curvature_gradient @ covariance @ curvature_gradient), np.finfo(np.float64).tiny)
        rounding = _EPSILON * (objective.magnitude + abs(value))
        if 0.5 * np.sqrt(sensitivity * 2.0 * newton_decrement) <= tolerance or newton_decrement <= rounding:
            break
        inner_tolerance = 2.0 * tolerance * tolerance / sensitivity
    penalty_log_determinant = sum(_log_pseudo_determinant(penalty[np.ix_(group, group)]) for group in _penalty_groups(prior))
    # A relative error e in B moves log|B + S| by at most D e for well-scaled B + S: the linear response is solved to
    # the evidence tolerance over the dimension, and never past what double precision resolves.
    total = _total_curvature(
        prior, coefficients, cavity, posterior_at(prior, coefficients), working_bytes, max(tolerance / coefficients.shape[0], _EPSILON)
    ) + penalty
    try:
        total_log_determinant, total_covariance = _cholesky_log_determinant_and_inverse(total)
        total_null_log_determinant, _total_null_inverse = _cholesky_log_determinant_and_inverse(null_basis.T @ total @ null_basis)
    except np.linalg.LinAlgError:
        return None
    evidence_value = value + 0.5 * penalty_log_determinant - 0.5 * total_log_determinant + 0.5 * total_null_log_determinant
    # W = (-H)^-1 - N (N'(-H)N)^-1 N' carries both determinants' dependence on x (computed above).
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
        newton_decrement=0.5 * float(gradient @ total_covariance @ gradient),
        magnitude=objective.magnitude + abs(evidence_value),
    )


def _smoothing_bounds(prior: ScaleMixturePrior, objective: _Objective) -> list[tuple[float, float]]:
    """The range of rho over which a fit at that weight is resolved in double precision, per weight.

    With D the data curvature of the weight's block and s_min, s_max the extreme nonzero eigenvalues of S_i:
    below lambda s_min = sqrt(eps) ||D|| the weakest penalized direction of -H has a condition number past
    1/sqrt(eps), and above lambda s_max = ||D|| / sqrt(eps) the penalty swamps the data's curvature past
    1/sqrt(eps), so a fit there loses half of double precision. The edges beyond (lambda = 0 and infinity) are
    evaluated exactly instead.
    """
    data = -(prior.coefficient_map.T @ objective.hessian @ prior.coefficient_map)
    bounds = []
    for block in prior.smoothing_blocks:
        block_data = data[np.ix_(block.coordinates, block.coordinates)]
        data_norm = max(float(np.max(np.abs(np.linalg.eigvalsh(0.5 * (block_data + block_data.T))))), np.finfo(np.float64).tiny)
        penalty_eigenvalues, _vectors = _range_projector(block.matrix)
        bounds.append((
            float(np.log(_HALF_PRECISION * data_norm / penalty_eigenvalues[0])),
            float(np.log(data_norm / (_HALF_PRECISION * penalty_eigenvalues[-1]))),
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
    cavity: Cavity, posterior_at: PosteriorAt,
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
        trial = _certified_evidence(prior, trial_weights, current.coefficients + current.responses @ step, cavity, posterior_at, working_bytes, tolerance, flat_start)
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
    prior: ScaleMixturePrior, log_smoothing: F64Array, start: F64Array, cavity: Cavity, posterior_at: PosteriorAt, working_bytes: int, tolerance: float, flat_start: F64Array
) -> _Evidence | None:
    """V at the certified inner maximum from ``start``, or from the flat start when that is not a certified
    maximum (the two structural starts; lead ruling); None when neither is."""
    warm = _evidence(prior, log_smoothing, start, cavity, posterior_at, working_bytes, tolerance)
    return warm if warm is not None else _evidence(prior, log_smoothing, flat_start, cavity, posterior_at, working_bytes, tolerance)


def _best_certified(
    prior: ScaleMixturePrior, log_smoothing: F64Array, starts: Sequence[F64Array], cavity: Cavity, posterior_at: PosteriorAt, working_bytes: int, tolerance: float
) -> _Evidence | None:
    """The certified inner maximum with the highest corrected V over the given starts; distinct basins are compared
    by ``_corrected_value``, and a start that lands in an already-found basin adds nothing."""
    certified: list[_Evidence] = []
    for start in starts:
        candidate = _evidence(prior, log_smoothing, start, cavity, posterior_at, working_bytes, tolerance)
        if candidate is None:
            continue
        scale = 1.0 + float(np.max(np.abs(candidate.coefficients)))
        if all(float(np.max(np.abs(candidate.coefficients - other.coefficients))) > _HALF_PRECISION * scale for other in certified):
            certified.append(candidate)
    if len(certified) <= 1:
        return certified[0] if certified else None
    return max(certified, key=lambda candidate: _corrected_value(prior, log_smoothing, candidate, cavity, posterior_at, working_bytes, tolerance))


def _log_normal_start(prior: ScaleMixturePrior, coefficients: F64Array, cavity: Cavity, working_bytes: int) -> F64Array:
    """The best log-normal shared density over a grid, with no class deviation and theta kept: the global start for
    the density's null model (lambda = infinity), whose objective is not concave in its location and log width.

    Widths run from the spacing, doubling, up to the kernel range's width; a width's locations are spaced by that
    width across the range, which a bump of that width resolves. The lattice mass of a quadratic eta carries no
    roughness penalty, so the grid compares data values alone; the trust-region refit then polishes the winner.
    """
    nodes = prior.log_variance_grid
    spacing = float(nodes[1] - nodes[0])
    extent = max(prior.kernel_top - prior.kernel_floor, spacing)
    basis = prior.coefficient_map[: prior.grid_size, : prior.pooled_size]
    base = np.array(coefficients, dtype=np.float64, copy=True)
    base[: prior.coefficient_size - prior.scale_size] = 0.0
    best, best_value = base, -np.inf
    width = spacing
    while width <= extent * (1.0 + _HALF_PRECISION):
        for centre in np.arange(prior.kernel_floor, prior.kernel_top + width, width):
            log_density = -0.5 * np.square((nodes - centre) / width)
            candidate = base.copy()
            candidate[: prior.pooled_size] = basis.T @ (log_density - log_density.mean())
            value = _data_value(prior, candidate, cavity, working_bytes)
            if value > best_value:
                best, best_value = candidate, value
        width *= 2.0
    return best


def _edge_evidence(
    prior: ScaleMixturePrior,
    weights: F64Array,
    infinite: frozenset[int],
    zero: frozenset[int],
    starts: Sequence[F64Array],
    cavity: Cavity, posterior_at: PosteriorAt,
    working_bytes: int,
    tolerance: float,
) -> tuple[F64Array, _Evidence] | None:
    """The best certified evidence of the model with ``infinite`` at lambda = infinity and ``zero`` at lambda = 0,
    the other weights at ``weights``; with its restriction basis."""
    view, allowed = _restricted_prior(prior, infinite, zero)
    finite = [position for position in range(weights.shape[0]) if position not in infinite | zero]
    evidence = _best_certified(view, weights[finite], [allowed.T @ start for start in starts], cavity, posterior_at, working_bytes, tolerance)
    return None if evidence is None else (allowed, evidence)


def _maximize_evidence(
    prior: ScaleMixturePrior,
    start_weights: F64Array,
    start_coefficients: F64Array,
    cavity: Cavity, posterior_at: PosteriorAt,
    working_bytes: int,
    bounds: list[tuple[float, float]],
    tolerance: float,
) -> tuple[F64Array, F64Array, _Evidence, _Evidence]:
    """Maximize V over every weight in [0, infinity]: the interior by the trust-region ascent inside the resolvable
    range, and each edge evaluated exactly (lambda = infinity by confining x to the block's null space, lambda = 0 by
    dropping the block), never by fitting at an extreme weight. A weight at the end of its range whose gradient
    points past it moves to the edge when the edge's V is higher; an edge weight moves back to the end of its range
    when V is higher there. Every V is the best certified maximum over the warm, flat and global log-normal starts.
    Returns the log weights (+inf and -inf at the edges), x in full coordinates, V there, and V at the start.
    """
    lower = np.array([bound[0] for bound in bounds])
    upper = np.array([bound[1] for bound in bounds])
    infinite = frozenset(int(position) for position in np.flatnonzero(start_weights == np.inf))
    zero = frozenset(int(position) for position in np.flatnonzero(start_weights == -np.inf))
    weights = np.where(start_weights == np.inf, upper, np.where(start_weights == -np.inf, lower, np.clip(start_weights, lower, upper)))
    flat = initial_hyperparameters(prior).coefficients
    log_normal = _log_normal_start(prior, start_coefficients, cavity, working_bytes)
    coefficients = np.array(start_coefficients, dtype=np.float64, copy=True)
    first = _edge_evidence(prior, weights, infinite, zero, [coefficients, flat, log_normal], cavity, posterior_at, working_bytes, tolerance)
    if first is None:
        raise FloatingPointError("no structural start reaches a certified maximum at the starting penalty weights")
    start = first[1]
    best_corrected = -np.inf
    while True:
        edges = infinite | zero
        finite = np.array([position for position in range(len(bounds)) if position not in edges], dtype=np.int64)
        view, allowed = _restricted_prior(prior, infinite, zero)
        entry = _edge_evidence(prior, weights, infinite, zero, [coefficients, flat, log_normal], cavity, posterior_at, working_bytes, tolerance)
        if entry is None:
            raise FloatingPointError("no structural start reaches a certified maximum at the current penalty weights")
        finite_weights, evidence = _ascend_evidence(
            view, weights[finite], entry[1], cavity, posterior_at, working_bytes, lower[finite], upper[finite], tolerance, allowed.T @ flat
        )
        refit = _best_certified(view, finite_weights, [evidence.coefficients, allowed.T @ flat, allowed.T @ log_normal], cavity, posterior_at, working_bytes, tolerance)
        if refit is not None and float(np.max(np.abs(refit.coefficients - evidence.coefficients))) > _HALF_PRECISION * (
            1.0 + float(np.max(np.abs(evidence.coefficients)))
        ):
            # Another basin wins at these weights: its weights are not optimized yet, so ascend again from it, as long
            # as each switch raises the best corrected V seen (so switches cannot cycle).
            refit_value = _corrected_value(view, finite_weights, refit, cavity, posterior_at, working_bytes, tolerance)
            if refit_value > best_corrected + tolerance:
                best_corrected = refit_value
                weights[finite] = finite_weights
                coefficients = allowed @ refit.coefficients
                continue
        weights[finite] = finite_weights
        coefficients = allowed @ evidence.coefficients
        current_value = evidence.value
        moved = False
        pushing = [
            (int(finite[index]), "infinite")
            for index in np.flatnonzero((finite_weights >= upper[finite]) & (evidence.gradient >= 0.0))
        ] + [(int(finite[index]), "zero") for index in np.flatnonzero((finite_weights <= lower[finite]) & (evidence.gradient <= 0.0))]
        for position, edge in pushing:
            trial_infinite = infinite | {position} if edge == "infinite" else infinite
            trial_zero = zero | {position} if edge == "zero" else zero
            trial = _edge_evidence(prior, weights, trial_infinite, trial_zero, [coefficients, flat, log_normal], cavity, posterior_at, working_bytes, tolerance)
            if trial is not None and trial[1].value > current_value + tolerance:
                infinite, zero, moved = frozenset(trial_infinite), frozenset(trial_zero), True
                coefficients = trial[0] @ trial[1].coefficients
                break
        if not moved:
            for position in sorted(edges):
                trial_infinite, trial_zero = infinite - {position}, zero - {position}
                trial_weights = weights.copy()
                trial_weights[position] = upper[position] if position in infinite else lower[position]
                trial = _edge_evidence(prior, trial_weights, trial_infinite, trial_zero, [coefficients, flat, log_normal], cavity, posterior_at, working_bytes, tolerance)
                if trial is not None and trial[1].value > current_value + tolerance:
                    infinite, zero, weights, moved = frozenset(trial_infinite), frozenset(trial_zero), trial_weights, True
                    coefficients = trial[0] @ trial[1].coefficients
                    break
        if not moved:
            log_smoothing = weights.copy()
            log_smoothing[sorted(infinite)] = np.inf
            log_smoothing[sorted(zero)] = -np.inf
            return log_smoothing, coefficients, evidence, start


def _stationarity_check(
    view: ScaleMixturePrior,
    weights: F64Array,
    evidence: _Evidence,
    interior: F64Array,
    cavity: Cavity,
    posterior_at: PosteriorAt,
    working_bytes: int,
    tolerance: float,
) -> tuple[F64Array, F64Array]:
    """The B-evidence's own gradient in each interior weight, by one central difference, and its curvature.

    The step balances the difference's truncation against V's rounding: with V known to eps times its magnitude and
    its log-weight curvature c (a three-point estimate at a unit step, the natural scale of a log weight, which the
    difference then refines), the optimal central step is (3 eps |V| / c)^(1/3).
    """
    gradient = np.zeros(weights.shape[0])
    curvature = np.ones(weights.shape[0])
    rounding = _EPSILON * evidence.magnitude
    for position in np.flatnonzero(interior):
        unit = np.zeros(weights.shape[0])
        unit[position] = 1.0
        wide = [_evidence(view, weights + side * unit, evidence.coefficients, cavity, posterior_at, working_bytes, tolerance) for side in (-1.0, 1.0)]
        if all(side is not None for side in wide):
            curvature[position] = max(abs(wide[0].value - 2.0 * evidence.value + wide[1].value), rounding)
        step = (3.0 * rounding / curvature[position]) ** (1.0 / 3.0)
        sides = [_evidence(view, weights + side * step * unit, evidence.coefficients, cavity, posterior_at, working_bytes, tolerance) for side in (-1.0, 1.0)]
        if any(side is None for side in sides):
            raise FloatingPointError("the B-evidence has no certified maximum next to the fitted penalty weights")
        gradient[position] = (sides[1].value - sides[0].value) / (2.0 * step)
    return gradient, curvature


def hyper_step(
    prior: ScaleMixturePrior, hyperparameters: MixtureHyperparameters, cavity: Cavity, posterior_at: PosteriorAt, working_bytes: int, tolerance: float
) -> HyperStep:
    """Maximize the B-evidence over every penalty weight in [0, infinity], with x at the penalized maximum for each, to
    ``tolerance`` nats: the resolution the fit certifies (1/(2K) for a scorer with K posterior draws).

    The search steers by the fixed-cavity gradient and accepts on V (with B). At the end, the B-evidence's own
    stationarity is checked by one central difference per interior weight (lead ruling); while the gain a Newton step
    on that difference predicts exceeds ``tolerance``, the search continues along it, accepting only steps that raise V.
    """
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
        prior, hyperparameters.log_smoothing, hyperparameters.coefficients, cavity, posterior_at, working_bytes, bounds, tolerance
    )
    final_infinite = frozenset(int(position) for position in np.flatnonzero(log_smoothing == np.inf))
    final_zero = frozenset(int(position) for position in np.flatnonzero(log_smoothing == -np.inf))
    final_view, final_allowed = _restricted_prior(prior, final_infinite, final_zero)
    finite_final = np.isfinite(log_smoothing)
    weights = log_smoothing[finite_final]
    lower = np.array([bound[0] for bound in bounds])[finite_final]
    upper = np.array([bound[1] for bound in bounds])[finite_final]
    evidence = replace(evidence, coefficients=final_allowed.T @ coefficients)
    while True:
        interior = (weights > lower) & (weights < upper)
        check, curvature = _stationarity_check(final_view, weights, evidence, interior, cavity, posterior_at, working_bytes, tolerance)
        if 0.5 * float(np.sum(check * check / curvature)) <= tolerance:
            break
        direction = check / curvature
        step_length, moved = 1.0, None
        while step_length * float(np.max(np.abs(direction))) > _HALF_PRECISION * (1.0 + float(np.max(np.abs(weights)))):
            trial_weights = np.clip(weights + step_length * direction, lower, upper)
            trial = _certified_evidence(final_view, trial_weights, evidence.coefficients, cavity, posterior_at, working_bytes, tolerance, final_allowed.T @ initial_hyperparameters(prior).coefficients)
            if trial is not None and trial.value > evidence.value:
                moved = (trial_weights, trial)
                break
            step_length *= 0.5
        if moved is None:
            break
        weights, evidence = moved
    log_smoothing = log_smoothing.copy()
    log_smoothing[finite_final] = weights
    return HyperStep(
        hyperparameters=MixtureHyperparameters(coefficients=final_allowed @ evidence.coefficients, log_smoothing=log_smoothing),
        penalized_objective=evidence.penalized_value,
        evidence=evidence.value,
        newton_decrement=evidence.newton_decrement,
        smoothing_gradient=float(np.max(np.abs(check))) if check.size else 0.0,
        start_decrement=start_decrement,
        evidence_gain=evidence.value - start_evidence.value,
    )
