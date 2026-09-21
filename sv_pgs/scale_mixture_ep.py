"""The variant side of the one model's EP-EB fit.

Stage 2 approximates the posterior of the effects by a Gaussian q(beta)
with precision (data precision) + diag(tau) and one Gaussian site
exp(-tau_j beta_j^2 / 2 + nu_j beta_j) per effect, and computes q's means and
marginal variances on the exact full-data operator; everything that involves
the prior is here.

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
- Every weight lives in (0, infinity], with one exact edge: at infinity the
  block's penalized directions are zero. There is no lambda = 0 edge (lead
  ruling): dropping a block profiles its directions under a flat prior, a
  different and improper model whose value bounds every proper-prior V from
  above, while V itself falls without bound as lambda -> 0 (slope r_i / 2 in
  log lambda). The low end of a weight's range is its resolvable bound.
- The lattice is a quadrature rule: its floor, top, spacing and extent are
  derived from the data and a tolerance (``kernel_floor``, ``kernel_top``,
  ``spacing_bound``, ``tail_mass``). Below the floor every kernel is flat to
  that tolerance (log Z's error there is bounded), but each node keeps its own
  variance v = u e^t in the kernel, so a density with its mass below the floor
  is a near-zero effect, never a point mass at zero (review-mathbugs N1).
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
linear part), which are
profiled as fixed effects: integrating them under a flat prior diverges
where the likelihood tends to a positive constant. Its gradient is exact:
dx/drho through the observed Hessian, and the change of H with x through the
third derivatives of log Z. Across the outer loop the curvature is the
EP-re-solved one; at fixed cavities it is H.

Outer loop (``fit_hyperparameters``). At an EP fixed point the EP evidence's
gradient in x is the fixed-cavity one, and its negative Hessian is the total
curvature B + S, with the cavities re-solved (``_total_curvature``). The loop
ascends one function, V(rho), with x its inner solve: the weights and x move
jointly to the maximum of V's local model (``hyper_step`` on the anchored
objective), accepted on V's realized gain above the tolerance, and x alone by
Newton on B + S at fixed weights, globalized by the natural monotonicity
test. It never moves by the fixed-cavity maximizer: that EP-EM step solves
with A + S, and it overshoots where (A + S)^-1 (B + S) exceeds 2.

Every derivative is computed in z = (eta_1, ..., eta_C, theta), where each variant's log Z_j depends on its
class's eta_c and its own log u_j, and carried to x by the linear map M.

Noise. A quantitative trait's residual variance takes the stationarity form of its type-II ML with q held,
sigma^2' = (RSS + sigma^2 gamma) / (n - k), with gamma = p - sum_j tau_j Sigma_jj and the covariates' flat prior
removing k. Its fixed point is the MacKay form RSS / (n - k - gamma), which is undefined where gamma >= n - k.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
from dataclasses import dataclass, replace
from types import ModuleType
from typing import Callable, Iterator, Sequence

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.linalg import solve_triangular
from scipy.optimize import brentq
from scipy.special import erfcx

from sv_pgs import engine_kernels
from sv_pgs._typing import F64Array, I64Array
from sv_pgs.krylov_recycle import block_gcro_dr

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
# QUADPACK's QK15I (dqk15i, Piessens et al. 1983): the 15-point Kronrod nodes on [0, 1) and their weights, with the
# embedded 7-point Gauss rule's weights at the same nodes (zero where a node is Kronrod's alone), and its error
# estimate's scale and power: err = asc min(1, (200 |K - G| / asc)^1.5).
_KRONROD_NODES = np.array([
    0.991455371120812639206854697526329, 0.949107912342758524526189684047851, 0.864864423359769072789712788640926,
    0.741531185599394439863864773280788, 0.586087235467691130294144845693013, 0.405845151377397166906606412076961,
    0.207784955007898467600689403773245, 0.0,
])
_KRONROD_WEIGHTS = np.array([
    0.022935322010529224963732008058970, 0.063092092629978553290700663189204, 0.104790010322250183839876322541518,
    0.140653259715525918745189590510238, 0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
    0.204432940075298892414161999234649, 0.209482141084727828012999174891714,
])
_GAUSS_WEIGHTS = np.array([
    0.0, 0.129484966168869693270611432679082, 0.0, 0.279705391489276667901467771423780, 0.0,
    0.381830050505118944950369775488975, 0.0, 0.417959183673469387755102040816327,
])
_QUADPACK_ERROR_SCALE = 200.0
_QUADPACK_ERROR_POWER = 1.5


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

    ``scale_design`` is the class-centred annotation design (p, F), over theta, followed by the uncentred level columns
    of ``offset_groups`` when there are any (see ``scale_mixture_prior``); ``coefficient_map`` is M, from x to z.
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
    offset_groups: I64Array | None = None
    # The EP evidence's local model about the fixed point whose cavities the objective uses (``_Anchor``); set only
    # inside ``hyper_step``, which searches that model.
    anchor: _Anchor | None = None

    @property
    def level_size(self) -> int:
        """The number of level coordinates (G - 1 for G offset groups), the last columns of ``scale_design``."""
        return 0 if self.offset_groups is None else int(np.unique(self.offset_groups).shape[0]) - 1

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
    weights not held at an edge or a bound of their range, from V's analytic
    gradient; ``stationarity_steps`` are the curvature's difference steps,
    ``stationarity_errors`` the gradient's error bounds, and ``stationarity_gain`` the weights' remaining gain
    (``_stationarity``: their Newton decrement, plus what a fold lets them climb; at most the tolerance when the
    step certified), with its parts ``stationarity_floor`` (the gain with exact correction slopes),
    ``stationarity_decrement`` (1/2 g'K^-1 g) and ``stationarity_fold`` (the folded weights'). ``tighten(budget)``
    re-checks the stationarity at the returned weights with the bound tightened to ``budget``, with no new search
    (None where that check finds a better side or no finite bound).
    """

    hyperparameters: MixtureHyperparameters
    penalized_objective: float
    evidence: float
    newton_decrement: float
    smoothing_gradient: float
    start_decrement: float
    evidence_gain: float
    stationarity_steps: F64Array
    stationarity_errors: F64Array
    stationarity_gain: float
    stationarity_floor: float = np.inf
    stationarity_decrement: float = np.inf
    stationarity_fold: float = np.inf
    tighten: Callable[[float], "HyperStep | None"] | None = None
    # ``resolve(budget)``: this step with V at its returned weights certified to ``budget``, where a decision needs it.
    resolve: Callable[[float], "HyperStep | None"] | None = None
    # The certified error of ``evidence`` (``_Evidence.error``): the outer loop's predicted gain carries it.
    evidence_error: float = 0.0


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
    # Where no effect clears the noise the kernel range is empty (floor = top), and the lattice still needs more
    # nodes than the roughness order to carry a density: half the order in spacings on each side gives order + 1.
    width = max(top - floor, 0.5 * ROUGHNESS_ORDER * spacing)
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
    offset_groups: I64Array | None = None,
) -> ScaleMixturePrior:
    """Validate and centre the prior's inputs and lay out x; every class in 0..C-1 must have a member.

    ``offset_groups`` (one group index per variant, e.g. its gene) gives each group a learned level: a shift l_g of every
    one of its variants' log prior variance, the same whatever their class. The levels sum to zero over the groups,
    unweighted (the class densities' locations carry the common location, and each class's location absorbs its own
    composition-weighted mean level, which is identified: do not re-centre the levels by weight), and have a Gaussian
    prior with one learned precision (ridge I on sum-to-zero coordinates); they enter log u_j uncentred, as the last G - 1 scale coefficients. They are identified
    against the classes' locations when the groups and classes connect: a common shift of the levels is the only
    direction they share, and sum-to-zero removes it.

    ``nodes`` is the uniform lattice in t; below ``floor`` the kernel is flat to the lattice's tolerance (each node
    still takes its own variance), and [floor, top] is the kernel range. x is laid out as (eta_bar in
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
    annotation_size = design.shape[1]
    groups = None
    if offset_groups is not None:
        groups = np.asarray(offset_groups, dtype=np.int64)
        labels, group_of_row = np.unique(groups, return_inverse=True)
        if groups.shape != classes.shape or labels.shape[0] < 2:
            raise ValueError("offset_groups needs one group per variant and at least two groups")
        indicator = np.zeros((classes.shape[0], labels.shape[0]))
        indicator[np.arange(classes.shape[0]), group_of_row] = 1.0
        # Uncentred: every variant of group g shifts by l_g exactly, whatever its class.
        levels = indicator @ _sum_to_zero_basis(labels.shape[0])
        # The levels must reach no class location (which the class densities carry) and no annotation: full column
        # rank of [annotations | levels | class indicators], i.e. the groups connect the classes.
        class_indicator = np.zeros((classes.shape[0], class_count))
        class_indicator[np.arange(classes.shape[0]), classes] = 1.0
        combined = np.column_stack([design, levels, class_indicator])
        eigenvalues = np.linalg.eigvalsh(combined.T @ combined)
        if eigenvalues[0] <= _EPSILON * combined.shape[0] * max(float(eigenvalues[-1]), 1.0):
            raise ValueError("the offset groups' levels are not identified: the groups must connect the classes")
        design = np.column_stack([design, levels])
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
    level_size = design.shape[1] - annotation_size
    if level_size:
        blocks.append(SmoothingBlock("offset group levels", annotation_start + annotation_size + np.arange(level_size), np.eye(level_size)))
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
        offset_groups=groups,
    )


def initial_hyperparameters(prior: ScaleMixturePrior, mean_variance: float | None = None) -> MixtureHyperparameters:
    """A start, not a prior: every class at one log-normal on the lattice, of the width whose density at the lattice's
    ends is eps of its peak when it is centred; no deviation or annotation effect; every penalty weight at its
    lambda = infinity edge, where that log-normal is the certified fit itself (the search releases a block inward
    where V rises; a unit weight was an arbitrary start, deep in the near-collapse regime on real genes).

    It is centred on the lattice unless ``mean_variance`` is given: the start's mean of e^t over the lattice, so that
    every prior variance E[beta_j^2] starts at mean_variance u_j (``moment_start``). The centre then moves along the
    lattice until the lattice mean equals it (the mean rises with the centre), or stops at the lattice's end nearer a
    target beyond its reach. The start only moves the fit's first iterate, never its certified answer.
    """
    nodes = prior.log_variance_grid
    width = 0.5 * (nodes[-1] - nodes[0]) / np.sqrt(2.0 * np.log(1.0 / _EPSILON))

    def log_mean(centre: float) -> float:
        quadratic = -0.5 * np.square((nodes - centre) / width)
        return float(_log_sum_exp(quadratic + nodes, axis=0) - _log_sum_exp(quadratic, axis=0))

    centre = 0.5 * (nodes[0] + nodes[-1])
    if mean_variance is not None:
        if not mean_variance > 0.0:
            raise ValueError("mean_variance must be positive")
        target = float(np.log(mean_variance))
        if log_mean(float(nodes[0])) >= target:
            centre = float(nodes[0])
        elif log_mean(float(nodes[-1])) <= target:
            centre = float(nodes[-1])
        else:
            centre = float(brentq(lambda value: log_mean(value) - target, float(nodes[0]), float(nodes[-1])))
    quadratic = -0.5 * np.square((nodes - centre) / width)
    coefficients = np.zeros(prior.coefficient_size)
    coefficients[: prior.pooled_size] = prior.coefficient_map[: prior.grid_size, : prior.pooled_size].T @ (quadratic - quadratic.mean())
    return MixtureHyperparameters(coefficients=coefficients, log_smoothing=np.full(len(prior.smoothing_blocks), np.inf))


def _log_sum_exp(values: F64Array, axis: int, keepdims: bool = False) -> F64Array:
    """log sum exp over ``axis``, shifted by the largest term (-inf where every term is -inf). scipy's logsumexp
    computes the same, but its array-API dispatch costs ~0.2 ms a call, which the line integrals' ~10^5 evaluations
    of the data value per hyper step turned into most of its time [sim-only: 131 of 192 s]."""
    largest = np.max(values, axis=axis, keepdims=True)
    shift = np.where(np.isfinite(largest), largest, 0.0)
    shifted = values - shift
    np.exp(shifted, out=shifted)
    with np.errstate(divide="ignore"):
        total = np.log(np.sum(shifted, axis=axis, keepdims=True)) + shift
    return total if keepdims else np.squeeze(total, axis=axis)


@dataclass(frozen=True)
class MomentStart:
    """Where EB starts on a trait: its phenotypic variance split into genetic and noise parts.

    ``heritability`` is the share of y's residual variance (after the covariates) the start gives the genotypes,
    ``genetic_variance`` and ``noise`` its two parts per sample, and ``mean_variance`` the start's mean of e^t, so
    that every E[beta_j^2] starts at mean_variance u_j (``initial_hyperparameters``); ``resolution`` is the moment
    estimate's standard error under no signal.
    """

    heritability: float
    genetic_variance: float
    noise: float
    mean_variance: float
    resolution: float


def moment_start(
    *, target_square: float, residual_dimension: float, score_square: float, gram_trace: float, weighted_diagonal: float, weighted_square: float, gram_square: float
) -> MomentStart:
    """The EB start from Haseman-Elston moments: the genetic variance can never exceed the phenotypic.

    With y (after the covariates, in r = n - k dimensions) = X beta + e, beta_j independent with variance c u_j and
    e ~ N(0, sigma^2 I), and G = X'X, the two moments are exact for a fixed design:
        E[y'y]       = sigma^2 r      + c sum_j u_j G_jj           (``target_square``, ``weighted_diagonal``)
        E[||X'y||^2] = sigma^2 tr G   + c sum_j u_j ||G e_j||^2    (``score_square``, ``gram_trace``, ``weighted_square``).
    The first is held exactly, V_y = y'y / r = sigma^2 + c sum_j u_j G_jj / r, so the split always satisfies the
    variance bound; the second then gives the heritability h^2 = c sum_j u_j G_jj / y'y. Under no signal ||X'y||^2
    has variance 2 sigma^4 ||G||_F^2 (``gram_square``), which makes h^2's standard error the resolution s: the start
    stays inside [s, 1 - s], at one resolution from either end the data cannot tell apart from it. Where the moments
    cannot place h^2 at all (s >= 1/2, or no curvature between the two moments), it is 1/2: the minimax point of the
    feasible interval [0, 1], the start whose largest distance to any heritability the data allow is least.
    """
    total = target_square / residual_dimension
    denominator = target_square * weighted_square / weighted_diagonal - total * gram_trace
    if denominator > 0.0:
        resolution = float(np.sqrt(2.0 * gram_square) * total / denominator)
        estimate = (score_square - total * gram_trace) / denominator
    else:
        resolution, estimate = np.inf, 0.5
    heritability = 0.5 if resolution >= 0.5 else float(np.clip(estimate, resolution, 1.0 - resolution))
    return MomentStart(
        heritability=heritability,
        genetic_variance=heritability * total,
        noise=(1.0 - heritability) * total,
        mean_variance=heritability * target_square / weighted_diagonal,
        resolution=resolution,
    )


def _density_and_scale(prior: ScaleMixturePrior, coefficients: F64Array) -> tuple[F64Array, F64Array]:
    """z = M x split into the class log densities (C x K, unnormalized) and the scale coefficients (L,)."""
    values = prior.coefficient_map @ coefficients
    return values[: prior.density_size].reshape(prior.class_count, prior.grid_size), values[prior.density_size :]


def class_log_density(prior: ScaleMixturePrior, coefficients: F64Array) -> F64Array:
    """log pi_ck = eta_ck - log sum_m e^eta_cm: the lattice mass of node k (C x K); the uniform weight h cancels."""
    log_weights, _scale = _density_and_scale(prior, coefficients)
    return log_weights - _log_sum_exp(log_weights, axis=1, keepdims=True)


def log_scale(prior: ScaleMixturePrior, coefficients: F64Array) -> F64Array:
    """log u_j = o_j + d_j' theta."""
    _density, scale_coefficients = _density_and_scale(prior, coefficients)
    return prior.log_variance_offset + prior.scale_design @ scale_coefficients


def relattice(
    prior: ScaleMixturePrior, hyperparameters: MixtureHyperparameters, nodes: F64Array, floor: float, top: float
) -> tuple[ScaleMixturePrior, MixtureHyperparameters]:
    """The same model on new nodes and a new kernel range: each class's eta from the model's own continuous log g
    inside the old lattice, and along its end slopes outside it, so the log-tails stay linear (which the
    third-order penalty leaves free) rather than following a polynomial's extrapolation.

    The model holds log g by its nodal values, and its roughness is the integral of the m-th derivative squared
    (m = ``ROUGHNESS_ORDER``); the continuous function it implies between the nodes is the one of least roughness
    through them, the natural spline of degree 2m - 1, whose derivatives m to 2m - 2 vanish at both ends
    (Schoenberg; Wahba 1990, Section 1.3). So a finer lattice holds the same density, and only the quadrature
    changes.
    """
    old_nodes = prior.log_variance_grid
    log_density, scale_coefficients = _density_and_scale(prior, hyperparameters.coefficients)
    # One end condition per class: the spline is vector-valued over the classes (review-mathbugs L-0).
    natural = [(order, np.zeros(log_density.shape[0])) for order in range(ROUGHNESS_ORDER, 2 * ROUGHNESS_ORDER - 1)]
    spline = make_interp_spline(old_nodes, log_density.T, k=2 * ROUGHNESS_ORDER - 1, bc_type=(natural, natural), axis=0)
    new_nodes = np.asarray(nodes, dtype=np.float64)
    inside = np.clip(new_nodes, old_nodes[0], old_nodes[-1])
    slopes = np.where(new_nodes < old_nodes[0], spline(old_nodes[0], nu=1)[:, None], spline(old_nodes[-1], nu=1)[:, None])
    new_density = spline(inside).T + slopes * (new_nodes - inside)[None, :]
    moved = scale_mixture_prior(
        class_index=prior.class_index,
        log_variance_offset=prior.log_variance_offset,
        annotation_design=prior.scale_design[:, : prior.scale_size - prior.level_size],
        annotation_groups=prior.annotation_groups,
        offset_groups=prior.offset_groups,
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
    log_mean_variance = _log_sum_exp(log_density + prior.log_variance_grid[None, :], axis=1)
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
    """Per variant and node: responsibilities w, conditional variances c and the derivatives of log Z_jk in eta = log u_j;
    ``rounding`` bounds each log Z_j's rounding (``_KernelRows.components``)."""

    log_normalizer: F64Array
    responsibility: F64Array
    conditional_variance: F64Array
    first: F64Array
    second: F64Array
    third: F64Array
    fourth: F64Array
    rounding: F64Array


def _kernel_terms(
    log_density: F64Array, log_scale_rows: F64Array, grid: F64Array, precision: F64Array, shift: F64Array
) -> tuple[F64Array, F64Array, F64Array, F64Array, F64Array]:
    """(v r = v/(1 + vP), r = 1/(1 + vP), qr = vP/(1 + vP), log pi_k + log Z_jk, h^2 v r) at every node, with the
    node's own variance v = u e^t (below the kernel floor too: the floor only bounds log Z's error there, and a flat
    kernel's v = 0 would make those nodes a point mass at zero, which the model does not have; review-mathbugs N1).
    Written so that an overflowing v (a node far past every effect's scale) gives its limits v r = 1/P, r = 0,
    qr = 1 and a component of weight zero, not inf * 0; an underflowing v gives v r = 0 and r = 1 exactly."""
    with np.errstate(over="ignore", divide="ignore"):
        variance = np.exp(log_scale_rows[:, None] + grid[None, :])
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
    log_density: F64Array, log_scale_rows: F64Array, grid: F64Array, precision: F64Array, shift: F64Array
) -> F64Array:
    return _log_sum_exp(_kernel_terms(log_density, log_scale_rows, grid, precision, shift)[3], axis=1)


class _KernelRows:
    """One chunk's kernel at a cavity and its log scales: every per-node quantity of ``_components`` except the class
    density's part, which enters log Z_jk = log pi_ck + L_jk additively (L the kernel's log with a flat density).

    So log Z_j = LSE_k(L_jk + log pi_ck) is one product of exp(L - max_k L) with exp(log pi - max log pi) over the
    nodes (a GEMV of positive terms: exact to K eps in relative terms), with a row whose product falls to where
    subnormal terms could matter taken exactly, as ``_line`` takes its fixed rows. The derivatives in eta = log u are
    formed on first use."""

    def __init__(self, log_scale_rows: F64Array, grid: F64Array, precision: F64Array, shift: F64Array) -> None:
        self.conditional, self.retained, self._ratio_retained, self.kernel, self._signal = _kernel_terms(
            np.zeros(grid.shape[0]), log_scale_rows, grid, precision, shift
        )
        # For the rounding bound of ``components``, per row: the largest size of a node term's kernel pieces,
        # log(1 + vP)/2 + h^2 c/2 (both non-negative), and the kernel's range over the nodes, both over the nodes
        # of weight above zero (a node past the overflow limit, retained 0, has an infinite piece and weight zero).
        with np.errstate(divide="ignore", invalid="ignore"):
            pieces = -0.5 * np.log(self.retained) + 0.5 * self._signal
            finite = np.isfinite(self.kernel)
            self._pieces_max = np.max(np.where(finite, pieces, -np.inf), axis=1)
            self._kernel_range = np.max(np.where(finite, self.kernel, -np.inf), axis=1) - np.min(np.where(finite, self.kernel, np.inf), axis=1)
        # Shifted by each row's largest term, as ``_log_sum_exp`` shifts (by 0 where that term is not finite).
        peak = np.max(self.kernel, axis=1)
        self.peak = np.where(np.isfinite(peak), peak, 0.0)
        with np.errstate(invalid="ignore"):
            self.exponentials = np.exp(self.kernel - self.peak[:, None])
        self._derivatives: tuple[F64Array, F64Array, F64Array, F64Array] | None = None
        # Held from one pass to the next (``_kernel_chunks``): read-only, so no caller can change a later pass's rows.
        for held in (self.conditional, self.retained, self.kernel, self.exponentials, self._pieces_max, self._kernel_range):
            held.setflags(write=False)

    def derivatives(self) -> tuple[F64Array, F64Array, F64Array, F64Array]:
        """(first, second, third, fourth): see ``_components``."""
        if self._derivatives is None:
            retained, ratio_retained, signal = self.retained, self._ratio_retained, self._signal
            self._derivatives = (
                0.5 * (retained * signal - ratio_retained),
                0.5 * signal * retained * (2.0 * retained - 1.0) - 0.5 * ratio_retained * retained,
                0.5 * signal * retained * (6.0 * retained * retained - 6.0 * retained + 1.0)
                - 0.5 * ratio_retained * retained * (2.0 * retained - 1.0),
                signal * (
                    retained * (3.0 * retained**3 - 3.0 * retained**2 + 0.5 * retained)
                    - retained * (1.0 - retained) * (9.0 * retained**2 - 6.0 * retained + 0.5)
                )
                + retained * (1.0 - retained) * (-3.0 * retained**2 + 3.0 * retained - 0.5),
            )
            for held in self._derivatives:
                held.setflags(write=False)
        return self._derivatives

    def normalizers(self, log_density: F64Array) -> tuple[F64Array, F64Array, F64Array, F64Array]:
        """(log Z_j, exp(log pi - max log pi), the products, the rows taken exactly)."""
        density_peak = float(np.max(log_density))
        scaled = np.exp(log_density - density_peak)
        products = self.exponentials @ scaled
        # Rows at subnormal level, and rows with a non-finite term (a kernel at its overflow limit), are taken exactly.
        lost = ~np.isfinite(products) | (products < np.finfo(np.float64).tiny / _EPSILON)
        with np.errstate(divide="ignore"):
            log_normalizer = np.log(products) + self.peak + density_peak
        if np.any(lost):
            log_normalizer[lost] = _log_sum_exp(self.kernel[lost] + log_density[None, :], axis=1)
        return log_normalizer, scaled, products, lost

    def components(self, log_density: F64Array) -> _Components:
        log_normalizer, scaled, products, lost = self.normalizers(log_density)
        responsibility = self.exponentials * scaled[None, :] / np.where(lost, 1.0, products)[:, None]
        if np.any(lost):
            responsibility[lost] = np.exp(self.kernel[lost] + log_density[None, :] - log_normalizer[lost, None])
        first, second, third, fourth = self.derivatives()
        # log Z_j's rounding (the verification harness's bound, tests/test_engine_verification._objective_rounding):
        # each node's term x_k = log pi_k - log(1 + vP)/2 + h^2 c/2 rounds by about four ulps of its pieces' sizes,
        # the log-sum-exp adds eps times sum_k w_k (1 + |x_k - max|) relative to its sum, and its maximum's own ulp.
        # |log Z_j| alone understates this wherever the terms are large and cancel. Each responsibility-weighted
        # mean is bounded by its largest term (the weights sum to one), which the kernel rows hold per row, so the
        # bound costs a pass over the rows, not the nodes: |x_k - max| <= the kernel's range plus the density's.
        finite_density = log_density[np.isfinite(log_density)]
        density_range = float(np.max(finite_density) - np.min(finite_density)) if finite_density.size else 0.0
        density_size = float(np.max(np.abs(finite_density))) if finite_density.size else 0.0
        rounding = _EPSILON * (np.abs(log_normalizer) + 1.0 + self._kernel_range + density_range + 4.0 * (density_size + self._pieces_max))
        return _Components(
            log_normalizer=log_normalizer, responsibility=responsibility, conditional_variance=self.conditional,
            first=first, second=second, third=third, fourth=fourth, rounding=rounding,
        )


def _components(
    log_density: F64Array, log_scale_rows: F64Array, grid: F64Array, precision: F64Array, shift: F64Array
) -> _Components:
    """With q = vP, r = 1/(1+q) and a = h^2 v r (so dr/deta = -r(1 - r) and da/deta = a r), each derivative of
    log Z_k in eta = log u is a A_n(r) - B_n(r): d1 = (a - q) r / 2, then A_(n+1) = r A_n - r(1 - r) A_n' and
    B_(n+1) = -r(1 - r) B_n', giving A_2 = r^2 - r/2, B_2 = r(1 - r)/2, A_3 = 3r^3 - 3r^2 + r/2,
    B_3 = r(1 - r)(2r - 1)/2, and A_4 = r A_3 - r(1 - r)(9r^2 - 6r + 1/2), B_4 = -r(1 - r)(-3r^2 + 3r - 1/2).
    Every node takes its own variance v = u e^t (``_kernel_terms``)."""
    return _KernelRows(log_scale_rows, grid, precision, shift).components(log_density)


# One hyper step's caches, opened by ``hyper_step`` and ended with it (context-local, so no two calls share one): the
# kernel rows held between its passes (``_kernel_chunks``) and its exact repeats (``_repeated``). Outside a hyper step
# nothing is held.
_STEP_CACHE: contextvars.ContextVar[dict | None] = contextvars.ContextVar("scale_mixture_ep_step_cache", default=None)


def _step_scoped(function: Callable) -> Callable:
    """``function`` with a ``_STEP_CACHE`` for the call's duration: its own where none is open, the open one otherwise
    (a hyper step inside the outer loop shares the loop's, so its exact repeats at an unchanged fixed point, the
    release trials of one polished state's successive hyper steps among them, are answered once)."""

    @functools.wraps(function)
    def scoped(*arguments, **keywords):
        if _STEP_CACHE.get() is not None:
            return function(*arguments, **keywords)
        token = _STEP_CACHE.set({})
        try:
            return function(*arguments, **keywords)
        finally:
            _STEP_CACHE.reset(token)

    return scoped


# Each held row set is ten p x K arrays: the six ``_KernelRows`` forms and the four derivatives once formed (its two
# rounding-bound rows are p-vectors). The outer
# loop holds its own cache across its iterations (``fit_hyperparameters`` is step-scoped too): every evaluation at one
# fixed point's cavity (its state, its Newton model, the gradient at a trial that becomes the next state, its hyper
# steps) shares the kernel rows and the exact repeats, which the cavity key keeps exact.
_HELD_ARRAYS_PER_ROW_SET = 10


def _kernel_chunks(
    prior: ScaleMixturePrior, scales: F64Array, cavity: Cavity, working_bytes: int
) -> Iterator[tuple[int, I64Array, _KernelRows]]:
    """(class, rows, kernel rows) over every chunk of every class. Within a hyper step, while the cavity, the log scales,
    the lattice and the chunks are the ones they were formed for (a hyper step holds its cavity, and without an
    annotation design the log scales too), later passes reuse the rows: held only where each class is one chunk and the
    held rows fit in ``working_bytes`` beside a pass's own (twice their size), so holding them stays inside the budget."""
    cache = _STEP_CACHE.get()
    held = None if cache is None else cache.get("rows")
    if (
        held is not None and held[0] is prior.class_rows and held[1] is prior.log_variance_grid and held[2] == working_bytes
        and np.array_equal(held[3], scales) and np.array_equal(held[4], cavity.precision) and np.array_equal(held[5], cavity.shift)
    ):
        yield from held[6]
        return
    chunks = [list(_row_chunks(class_rows, prior.grid_size, working_bytes)) for class_rows in prior.class_rows]
    if all(len(pieces) <= 1 for pieces in chunks):
        formed = [
            (class_position, rows, _KernelRows(scales[rows], prior.log_variance_grid, cavity.precision[rows], cavity.shift[rows]))
            for class_position, pieces in enumerate(chunks) for rows in pieces
        ]
        held_bytes = _HELD_ARRAYS_PER_ROW_SET * sum(kernel_rows.kernel.nbytes for _class, _rows, kernel_rows in formed)
        if cache is not None and 2 * held_bytes <= working_bytes:
            cache["rows"] = (
                prior.class_rows, prior.log_variance_grid, working_bytes, scales.copy(), cavity.precision.copy(), cavity.shift.copy(), formed,
            )
        yield from formed
        return
    for class_position, pieces in enumerate(chunks):
        for rows in pieces:
            yield class_position, rows, _KernelRows(scales[rows], prior.log_variance_grid, cavity.precision[rows], cavity.shift[rows])


def _class_terms(
    prior: ScaleMixturePrior, coefficients: F64Array, cavity: Cavity, working_bytes: int
) -> Iterator[tuple[int, I64Array, _Components]]:
    """(class, rows, components) over every chunk of every class."""
    log_density = class_log_density(prior, coefficients)
    scales = log_scale(prior, coefficients)
    for class_position, rows, kernel_rows in _kernel_chunks(prior, scales, cavity, working_bytes):
        yield class_position, rows, kernel_rows.components(log_density[class_position])


def tilted_moments(
    prior: ScaleMixturePrior, hyperparameters: MixtureHyperparameters, cavity: Cavity, working_bytes: int,
    array_module: ModuleType = np,
) -> TiltedMoments:
    """log Z_j and the exact tilted mean and variance of every effect; with ``array_module`` CuPy, by the fused
    device kernel (``engine_kernels``) within ``working_bytes`` of device memory."""
    if array_module is not np:
        on_device = engine_kernels.tilted_moments(
            array_module, prior.class_index, class_log_density(prior, hyperparameters.coefficients),
            log_scale(prior, hyperparameters.coefficients), prior.log_variance_grid, cavity.precision, cavity.shift, working_bytes,
        )
        log_normalizer, mean, variance = (array_module.asnumpy(values) for values in on_device)
        return TiltedMoments(log_normalizer=log_normalizer, mean=mean, variance=variance)
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


def tilted_cumulants(
    prior: ScaleMixturePrior, hyperparameters: MixtureHyperparameters, cavity: Cavity, working_bytes: int
) -> tuple[F64Array, F64Array]:
    """The third and fourth cumulants of every effect's tilted law, exact.

    Given node k the tilted law is N(h c_k, c_k), with responsibility w_k, so its central moments come from the
    components' own: with d_k = h c_k - m, m2 = E_w[d^2 + c], m3 = E_w[d^3 + 3 d c] and m4 = E_w[d^4 + 6 d^2 c + 3 c^2],
    each a weighted sum of terms whose signs are known, and kappa3 = m3, kappa4 = m4 - 3 m2^2.
    """
    third = np.empty(prior.variant_count)
    fourth = np.empty(prior.variant_count)
    for _class, rows, terms in _class_terms(prior, hyperparameters.coefficients, cavity, working_bytes):
        shift = cavity.shift[rows][:, None]
        conditional = terms.conditional_variance
        weights = terms.responsibility
        component_mean = shift * conditional
        mean = np.sum(weights * component_mean, axis=1)[:, None]
        offset = component_mean - mean
        second = np.sum(weights * (np.square(offset) + conditional), axis=1)
        third[rows] = np.sum(weights * (offset ** 3 + 3.0 * offset * conditional), axis=1)
        fourth[rows] = np.sum(weights * (offset ** 4 + 6.0 * np.square(offset) * conditional + 3.0 * np.square(conditional)), axis=1) - 3.0 * np.square(second)
    return third, fourth


def quadrature_majorant_ratio(
    prior: ScaleMixturePrior, hyperparameters: MixtureHyperparameters, cavity: Cavity, working_bytes: int
) -> float:
    """sum_j M_j / Z_j with M_j = max(Z_j, sum_k pi_k |L_j(t_k + i pi/2)|), for ``spacing_bound``.

    At t + i pi/2 the variance is i v, so log|L| = -log(1 + q^2)/4 + h^2 v q / (2 (1 + q^2)), at every node with its
    own v (``_kernel_terms``).
    """
    log_density = class_log_density(prior, hyperparameters.coefficients)
    scales = log_scale(prior, hyperparameters.coefficients)
    total = 0.0
    for class_position, class_rows in enumerate(prior.class_rows):
        for rows in _row_chunks(class_rows, prior.grid_size, working_bytes):
            with np.errstate(over="ignore"):
                variance = np.exp(scales[rows][:, None] + prior.log_variance_grid[None, :])
            ratio = variance * cavity.precision[rows][:, None]
            shift_square = np.square(cavity.shift[rows])[:, None]
            log_real = log_density[class_position] - 0.5 * np.log1p(ratio) + 0.5 * shift_square * variance / (1.0 + ratio)
            log_strip = log_density[class_position] - 0.25 * np.log1p(ratio * ratio) + 0.5 * shift_square * variance * ratio / (1.0 + ratio * ratio)
            total += float(np.sum(np.exp(np.maximum(_log_sum_exp(log_strip, axis=1) - _log_sum_exp(log_real, axis=1), 0.0))))
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
    *, residual_sum_of_squares: float, sample_count: int, covariate_count: int, site_precision: F64Array, posterior_variance: F64Array, noise: float
) -> float:
    """sigma^2' = (RSS + tr(Xt Sigma Xt')) / (n - k): the evidence's stationarity in sigma^2 with q held, always positive.

    q's precision is Xt'Xt / sigma^2 + diag tau, so tr(Xt Sigma Xt') = sigma^2 gamma with gamma = p - sum_j tau_j
    Sigma_jj, the effective number of effects. Its fixed point is the MacKay form RSS / (n - k - gamma), which has no
    solution where gamma >= n - k (p far above n); this form is defined everywhere.
    """
    effective_effects = float(site_precision.shape[0] - np.sum(site_precision * posterior_variance))
    return (float(residual_sum_of_squares) + float(noise) * effective_effects) / (sample_count - covariate_count)


def noise_gain(new_noise: float, old_noise: float, sample_count: int, covariate_count: int) -> float:
    """The evidence gain of moving sigma^2 from ``old_noise`` to its stationary value ``new_noise`` with q held:
    1/2 (n - k) (r - 1 - log r), r = new / old. It is never negative, since r - 1 >= log r."""
    ratio = float(new_noise) / float(old_noise)
    return 0.5 * (sample_count - covariate_count) * (ratio - 1.0 - float(np.log(ratio)))


# ------------------------------------------------------------------ the hyper objective


@dataclass(frozen=True)
class _Objective:
    """sum_j log Z_j at fixed cavities, its gradient and Hessian in z, the sum of |log Z_j| (its scale), and ``rounding``,
    a bound on |value - sum_j log Z_j| in double precision: each term's own (``_Components.rounding``) plus the
    recursive summation's, at most (p - 1) eps of the terms' sizes (Higham, Accuracy and Stability of Numerical
    Algorithms, 2nd ed., Lemma 3.1)."""

    value: float
    gradient: F64Array
    hessian: F64Array
    magnitude: float
    rounding: float


def _data_objective(
    prior: ScaleMixturePrior, coefficients: F64Array, cavity: Cavity, working_bytes: int, array_module: ModuleType = np, hessian_too: bool = True
) -> _Objective:
    """For variant j of class c, with responsibilities w_j, component derivatives g_jk and their mean gbar_j, in z:
    d/deta_c = w_j - pi_c and d/d(scale) = gbar_j d_j (d_j the variant's scale-design row);
    d2/deta_c2 = diag(w_j) - w_j w_j' - (diag pi_c - pi_c pi_c'), d2/deta_ck d(scale) = w_jk (g_jk - gbar_j) d_j,
    and d2/d(scale)2 = (Var_w(g_j) + E_w[dg_j/deta]) d_j d_j'. With ``array_module`` CuPy, by the fused device kernel.
    With ``hessian_too`` false the Hessian is left zero (a gradient's pass: the p x K^2 responsibility products are the
    pass's bulk), on the host path."""
    if array_module is not np:
        value, gradient, hessian, magnitude = engine_kernels.objective_statistics(
            array_module, prior.class_rows, class_log_density(prior, coefficients), log_scale(prior, coefficients),
            prior.log_variance_grid, cavity.precision, cavity.shift, prior.scale_design, working_bytes,
        )
        # The device kernel returns the terms' sizes, not their pieces: the bound is the summation lemma's on K-term
        # log-sum-exps and p terms, (K + 1 + p) eps of the sizes.
        return _Objective(
            value=value, gradient=gradient, hessian=hessian, magnitude=magnitude,
            rounding=(prior.grid_size + 1 + prior.variant_count) * _EPSILON * magnitude,
        )
    grid_size = prior.grid_size
    scale_span = slice(prior.density_size, prior.density_size + prior.scale_size)
    dimension = prior.density_size + prior.scale_size
    gradient = np.zeros(dimension)
    hessian = np.zeros((dimension, dimension))
    value = magnitude = rounding = 0.0
    density = np.exp(class_log_density(prior, coefficients))
    responsibility_sum = np.zeros((prior.class_count, grid_size))
    responsibility_outer = np.zeros((prior.class_count, grid_size, grid_size))
    cross = np.zeros((prior.class_count, grid_size, prior.scale_size))
    for class_position, rows, terms in _class_terms(prior, coefficients, cavity, working_bytes):
        responsibility = terms.responsibility
        value += float(np.sum(terms.log_normalizer))
        magnitude += float(np.sum(np.abs(terms.log_normalizer)))
        rounding += float(np.sum(terms.rounding))
        responsibility_sum[class_position] += responsibility.sum(axis=0)
        if hessian_too:
            responsibility_outer[class_position] += responsibility.T @ responsibility
        if prior.scale_size:
            # Without an annotation design the scale terms are empty products.
            design = prior.scale_design[rows]
            mean_first = np.sum(responsibility * terms.first, axis=1)
            gradient[scale_span] += design.T @ mean_first
            if hessian_too:
                centred_first = terms.first - mean_first[:, None]
                curvature = np.sum(responsibility * (np.square(centred_first) + terms.second), axis=1)
                cross[class_position] += (responsibility * centred_first).T @ design
                hessian[scale_span, scale_span] += design.T @ (curvature[:, None] * design)
    for class_position, class_rows in enumerate(prior.class_rows):
        size = float(class_rows.shape[0])
        class_density = density[class_position]
        span = slice(class_position * grid_size, (class_position + 1) * grid_size)
        gradient[span] = responsibility_sum[class_position] - size * class_density
        if not hessian_too:
            continue
        hessian[span, span] = (
            np.diag(responsibility_sum[class_position])
            - responsibility_outer[class_position]
            - size * (np.diag(class_density) - np.outer(class_density, class_density))
        )
        hessian[span, scale_span] = cross[class_position]
        hessian[scale_span, span] = cross[class_position].T
    return _Objective(
        value=value, gradient=gradient, hessian=hessian, magnitude=magnitude,
        rounding=rounding + max(prior.variant_count - 1, 0) * _EPSILON * magnitude,
    )


def _data_value(
    prior: ScaleMixturePrior, coefficients: F64Array, cavity: Cavity, working_bytes: int, array_module: ModuleType = np
) -> float:
    """sum_j log Z_j alone: a trial point's evaluation."""
    log_density = class_log_density(prior, coefficients)
    scales = log_scale(prior, coefficients)
    if array_module is not np:
        log_normalizer, _mean, _variance = engine_kernels.tilted_moments(
            array_module, prior.class_index, log_density, scales, prior.log_variance_grid, cavity.precision, cavity.shift, working_bytes,
        )
        return float(log_normalizer.sum())
    total = 0.0
    for class_position, _rows, kernel_rows in _kernel_chunks(prior, scales, cavity, working_bytes):
        total += float(np.sum(kernel_rows.normalizers(log_density[class_position])[0]))
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
    """The penalized objective, its gradient and Hessian in x, less the local model's C term where the prior has an
    anchor (``_Anchor``; ``penalty`` stays S alone)."""
    mapping = prior.coefficient_map
    penalty_value, penalty_gradient = _penalty_value(prior, log_smoothing, coefficients)
    anchor_value, anchor_gradient = _anchor_value(prior, coefficients)
    hessian = mapping.T @ objective.hessian @ mapping - penalty
    if prior.anchor is not None:
        hessian = hessian - prior.anchor.matrix
    return objective.value - penalty_value - anchor_value, mapping.T @ objective.gradient - penalty_gradient - anchor_gradient, hessian


def _resolved_spectrum(spectrum: tuple[F64Array, F64Array]) -> tuple[F64Array, F64Array]:
    """The spectrum with every eigenvalue within the eigendecomposition's rounding of zero (eps times the dimension
    times the largest magnitude) raised to that floor: such a direction is flat to double precision, neither an
    ascent to follow nor a negative curvature to leave by (on a weak-data verification case the smallest eigenvalue
    of -H sat at +-1e-14 and flipped sign between iterates, and the trust-region step walked 2-10 units along it)."""
    eigenvalues, eigenvectors = spectrum
    floor = _EPSILON * eigenvalues.shape[0] * max(float(np.max(np.abs(eigenvalues))), np.finfo(np.float64).tiny)
    return np.where(np.abs(eigenvalues) <= floor, floor, eigenvalues), eigenvectors


def _spectrum(negative_hessian: F64Array) -> tuple[F64Array, F64Array]:
    """-H's eigendecomposition (of its symmetric part), shared by every step taken at one point."""
    return np.linalg.eigh(0.5 * (negative_hessian + negative_hessian.T))


def _ascent_direction(negative_hessian: F64Array, gradient: F64Array, spectrum: tuple[F64Array, F64Array] | None = None) -> F64Array:
    """The Newton direction on -H with every eigenvalue replaced by its magnitude (an ascent direction where -H is indefinite).

    Eigenvalues below eps times the largest are raised to that floor, the rounding level of the spectrum.
    """
    eigenvalues, eigenvectors = _spectrum(negative_hessian) if spectrum is None else spectrum
    magnitudes = np.maximum(np.abs(eigenvalues), _EPSILON * float(np.max(np.abs(eigenvalues))))
    return eigenvectors @ ((eigenvectors.T @ gradient) / magnitudes)


def _trust_region_step(
    negative_hessian: F64Array, gradient: F64Array, radius: float, spectrum: tuple[F64Array, F64Array] | None = None
) -> F64Array:
    """The maximizer of g's - s'(-H)s/2 over ||s|| <= radius (More and Sorensen), from -H's eigendecomposition.

    s(mu) = (-H + mu I)^-1 g with mu >= max(0, -lambda_min) and ||s(mu)|| = radius unless the Newton
    step of a positive definite -H already fits; ||s(mu)|| falls monotonically in mu, so mu is bisected
    to double precision between its bounds.
    """
    eigenvalues, eigenvectors = _spectrum(negative_hessian) if spectrum is None else spectrum
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
    shifted = eigenvalues + upper
    # More and Sorensen's hard case: -H is not positive definite, and no mu > -lambda_min reaches the boundary, because
    # g has no part on -H's lowest eigenspace (or one below mu's resolution, where mu = -lambda_min exactly and the
    # shifted eigenvalue there is 0) and the rest of the step at mu = -lambda_min lies inside the radius. The
    # maximizer then keeps the rest at mu = -lambda_min and reaches the boundary along the lowest eigenvector, on g's
    # side where g has a part there (either side ascends equally where it has none).
    at_lower = eigenvalues + lower
    lowest = ~(at_lower > 0.0)
    rest = np.where(lowest, 0.0, components / np.where(lowest, 1.0, at_lower))
    unresolved = not np.all(shifted > 0.0)
    orthogonal = not np.any(components[lowest] != 0.0) and float(rest @ rest) <= radius * radius
    if eigenvalues[0] > 0.0 or not (unresolved or orthogonal):
        return eigenvectors @ (components / shifted)
    reach = float(np.sqrt(max(radius * radius - float(rest @ rest), 0.0)))
    first = int(np.flatnonzero(lowest)[0])
    rest[first] = reach if components[first] >= 0.0 else -reach
    return eigenvectors @ rest


def _maximize_coefficients(
    prior: ScaleMixturePrior, log_smoothing: F64Array, start: F64Array, cavity: Cavity, working_bytes: int, tolerance: float
) -> tuple[F64Array, _Objective]:
    """Trust-region Newton ascent of the penalized objective at fixed penalty weights.

    The objective is not concave in log g, so each step maximizes the quadratic model inside a radius
    (More-Sorensen); a trial costs one value-only pass, and the radius follows the ratio of actual to
    predicted gain (Nocedal and Wright, Algorithm 4.1). It stops at a strict maximum, where -H is positive
    definite and the Newton step's predicted gain is below ``tolerance`` nats (the resolution the caller
    certifies); a saddle's negative curvature is followed by the trust-region step instead. It also stops when no
    step longer than half of double precision raises the objective. Where the model predicts no gain above the
    objective's rounding inside the radius, a value-only trial would compare two values at their rounding (on
    gene 1's release trials, whose penalized directions carry a data curvature 1e-9 of the penalty's scale, the
    loop once accepted rounding-level gains and doubled its radius without end): such a step is judged on the
    gradient instead, taken where its decrement in the current metric falls, as the outer loop judges its inner
    steps, so x reaches a stationary point to the gradient's resolution when the tolerance asks for it (the
    engine's verification harness asks with tolerance 0: the value's rounding is not the gradient's).
    """
    penalty = _penalty_matrix(prior, log_smoothing)
    coefficients = np.array(start, dtype=np.float64, copy=True)
    objective = _data_objective(prior, coefficients, cavity, working_bytes)
    value, gradient, hessian = _penalized(prior, objective, log_smoothing, penalty, coefficients)
    # -H changes only when a step is accepted: its one eigendecomposition serves the test, the ascent direction and
    # every trust-region trial at that point; a direction flat to rounding counts as flat (``_resolved_spectrum``).
    spectrum = _resolved_spectrum(_spectrum(-hessian))
    ascent = _ascent_direction(-hessian, gradient, spectrum)
    # The first radius is the Cauchy step's length on |-H|, ||g||^3 / g'|-H|g (as the outer loop's, ``_cauchy_radius``),
    # never Newton's: on a direction the data barely curve Newton's step divides a gradient at rounding by a curvature
    # at rounding (gene 1's classes without a variant: 710 units along their unidentified deviation, at an unchanged V).
    components = spectrum[1].T @ gradient
    curvature = float(np.sum(np.abs(spectrum[0]) * np.square(components)))
    radius = float(np.linalg.norm(gradient)) ** 3 / curvature if curvature > 0.0 else 0.0
    while True:
        # The objective's own rounding (its terms' and their summation's) plus the penalized value's arithmetic.
        rounding = objective.rounding + _EPSILON * abs(value)
        definite = float(spectrum[0][0]) > 0.0
        if definite and 0.5 * float(gradient @ ascent) <= tolerance:
            return coefficients, objective
        if radius <= _HALF_PRECISION * (1.0 + float(np.linalg.norm(coefficients))):
            return coefficients, objective
        step = _trust_region_step(-hessian, gradient, radius, spectrum)
        predicted = float(gradient @ step) + 0.5 * float(step @ hessian @ step)
        if predicted <= rounding:
            # The value cannot resolve this step; the gradient can. It is judged as the outer loop judges its inner
            # steps: taken where the decrement in the current metric falls (a strict maximum), or where the trapezoid
            # gain of the two gradients is positive (a saddle), and the maximization ends otherwise. A step below x's
            # own resolution cannot be told from the point it leaves (the decrements of both sit at their rounding),
            # so it ends the maximization too.
            if float(np.linalg.norm(step)) <= _HALF_PRECISION * (1.0 + float(np.linalg.norm(coefficients))):
                return coefficients, objective
            candidate = coefficients + step
            candidate_objective = _data_objective(prior, candidate, cavity, working_bytes)
            candidate_value, candidate_gradient, candidate_hessian = _penalized(prior, candidate_objective, log_smoothing, penalty, candidate)
            if definite:
                resolved = 0.5 * float(candidate_gradient @ _ascent_direction(-hessian, candidate_gradient, spectrum)) < 0.5 * float(gradient @ ascent)
            else:
                resolved = 0.5 * float((gradient + candidate_gradient) @ step) > 0.0
            if not (resolved and np.isfinite(candidate_value)):
                return coefficients, objective
            coefficients, objective = candidate, candidate_objective
            value, gradient, hessian = candidate_value, candidate_gradient, candidate_hessian
            spectrum = _resolved_spectrum(_spectrum(-hessian))
            ascent = _ascent_direction(-hessian, gradient, spectrum)
            continue
        candidate = coefficients + step
        candidate_value = (
            _data_value(prior, candidate, cavity, working_bytes) - _penalty_value(prior, log_smoothing, candidate)[0] - _anchor_value(prior, candidate)[0]
        )
        actual = candidate_value - value
        ratio = actual / predicted if predicted > 0.0 else -np.inf
        step_norm = float(np.linalg.norm(step))
        # A gain at the objective's rounding is not a gain: the step is refused, and a refused step shrinks the radius
        # whatever its ratio (a step refused at an unchanged radius would be proposed again without end).
        accepted = bool(np.isfinite(candidate_value)) and actual > rounding
        if not accepted or ratio < 0.25:
            radius = 0.25 * step_norm
        elif ratio > 0.75 and step_norm >= radius * (1.0 - _HALF_PRECISION):
            radius = 2.0 * radius
        if accepted:
            coefficients = candidate
            objective = _data_objective(prior, coefficients, cavity, working_bytes)
            if not np.any(np.abs(objective.gradient) > _EPSILON * objective.magnitude):
                # The data no longer see x: every tilted moment has saturated (the density is a point mass on the
                # lattice to rounding). What the model still gains beyond here is the anchor's quadratic alone, a
                # local model of the fixed point's response taken out of its range (at a released weight of e^-22
                # on gene 1 its indefinite C sent x to 1e153 over 6,600 accepted steps), so the maximization ends
                # here; the point certifies nothing (``_evidence_once``).
                return coefficients, objective
            value, gradient, hessian = _penalized(prior, objective, log_smoothing, penalty, coefficients)
            spectrum = _resolved_spectrum(_spectrum(-hessian))
            ascent = _ascent_direction(-hessian, gradient, spectrum)


@dataclass(frozen=True)
class GaussianPosterior:
    """q's linear responses at the fixed point, for the total curvature B: ``solve(R, e)`` is Sigma R, each column
    to relative error e in the posterior metric, and ``variance_jvp(W)`` is -(Sigma o Sigma) W, both (p x r). Stage 2
    answers them with extra right-hand sides of its solve and with ``marginal_variances.variance_jvp``.

    ``linear_response(left, right, diagonal, weight, B)``, when a posterior can give it, is the exact solution X of
    (I - (I - diag(weight) (Sigma o Sigma)) (diag(left) Sigma diag(right) + diag(diagonal))) X = B: the linear response
    ``_total_curvature`` otherwise finds by GMRES (speed-smalln: the small-n route factors this p x p matrix once).

    ``cavity_response(m_x E, s2_x E)``, for a fixed point that is not EP's (the mean-field fixed point, whose cavity
    is the pseudo-likelihood: ``mean_field``), is that fixed point's own response (dh, dP) of every variant's cavity
    to the directions' fixed-cavity moment changes (both p x r); ``_total_curvature_columns`` then takes B from it
    directly and never asks ``solve`` or ``variance_jvp``, which such a posterior leaves None."""

    solve: Callable[[F64Array, float], F64Array] | None = None
    variance_jvp: Callable[[F64Array], F64Array] | None = None
    linear_response: Callable[[F64Array, F64Array, F64Array, F64Array, F64Array], F64Array] | None = None
    # ``local_response(left, right, diagonal, weight)``: V -> M^-1 V for the same matrix with Sigma replaced by its
    # block-local part (read-free), the preconditioner of the Krylov route (``krylov_recycle``, lane speed-recycle).
    local_response: Callable[[F64Array, F64Array, F64Array, F64Array], Callable[[F64Array], F64Array]] | None = None
    cavity_response: Callable[[F64Array, F64Array], tuple[F64Array, F64Array]] | None = None
    # Whether the responses are exact to rounding whatever tolerance they are asked for (a dense factor, or
    # independent effects): B's response is then charged its rounding, not the request.
    exact: bool = False

    def __post_init__(self) -> None:
        if self.cavity_response is None and (self.solve is None or self.variance_jvp is None):
            raise ValueError("a GaussianPosterior answers by solve and variance_jvp, or by cavity_response.")


class NoCertifiedProgress(FloatingPointError):
    """The outer loop's step shrank to double precision with every trial refused, where no uncertified fit can be
    returned honestly (the weights have no evaluated step, or B + S is indefinite there)."""


class LinearResponseError(RuntimeError):
    """B's linear response could not be solved to its tolerance. Not a FloatingPointError: the outer loop reads those
    as "no certified maximum here" and steps on; without B the fit cannot be certified at all."""


class CurvatureCorrection:
    """C = B - A at an EP fixed point: the total curvature's EP-response part, in the full prior's x coordinates.

    V evaluates B + S at the x_rho its search re-maximizes, where EP has not been re-solved; the correction is held
    at the fixed point x_k and A moves with x, so B + S is taken as A(x_rho) + S + C. At x_rho = x_k that is B + S
    exactly, which is where the certificate is taken; elsewhere it is B to first order in x_rho - x_k, as A is. A
    view's coordinates are x = K x_view with M_view = M K, so the correction there is K' C K. For independent effects
    (normal means) the cavities do not move with the prior, so C = 0 (``INDEPENDENT_EFFECTS``).

    Either given whole (``coefficient_map`` and ``matrix``), or formed only where it is asked (``columns``, from
    ``curvature_correction``): B's linear response is solved for the directions of the views asked for, starting
    with the free coefficients of the current edges, and extended by the new directions when a view releases an
    edge (lead ruling: the response on d_free directions, not all D).
    """

    def __init__(
        self,
        coefficient_map: F64Array | None = None,
        matrix: F64Array | None = None,
        *,
        mapping: F64Array | None = None,
        columns: Callable[[F64Array], F64Array] | None = None,
        fixed_curvature: F64Array | None = None,
        resolutions: list[float] | None = None,
    ) -> None:
        self.coefficient_map = coefficient_map
        # The relative residuals B's response columns were solved to (``columns`` appends them); none for a given C.
        self._resolutions = [] if resolutions is None else resolutions
        self.matrix = matrix
        self._mapping = mapping
        self._columns = columns
        self._fixed = fixed_curvature
        # An orthonormal basis Q (x coordinates) of the directions solved so far, and B_z M Q.
        self._basis: F64Array | None = None
        self._total: F64Array | None = None

    @property
    def resolution(self) -> float:
        """The largest relative residual B's response has been solved to so far (0 for a correction given whole)."""
        return max(self._resolutions, default=0.0)

    @property
    def solved_directions(self) -> int:
        """How many directions B's linear response has been solved for (lazy corrections only)."""
        return 0 if self._basis is None else int(self._basis.shape[1])

    def on(self, coefficient_map: F64Array) -> F64Array:
        size = int(coefficient_map.shape[1])
        if self._columns is not None:
            return self._lazy(coefficient_map)
        if self.matrix is None or self.coefficient_map is None:
            return np.zeros((size, size))
        basis = np.linalg.lstsq(self.coefficient_map, coefficient_map, rcond=None)[0]
        return basis.T @ self.matrix @ basis

    def _lazy(self, coefficient_map: F64Array) -> F64Array:
        mapping, columns, fixed = self._mapping, self._columns, self._fixed
        assert mapping is not None and columns is not None and fixed is not None
        view = np.linalg.lstsq(mapping, coefficient_map, rcond=None)[0]
        if view.shape[1] == 0:
            return np.zeros((0, 0))
        if self._basis is None:
            missing = view
        else:
            missing = view - self._basis @ (self._basis.T @ view)
        # A direction is new when the view has a part outside the solved span beyond the span's own rounding.
        values, vectors = np.linalg.eigh(missing.T @ missing)
        new = values > _EPSILON * view.shape[0] * max(float(np.max(np.linalg.eigvalsh(view.T @ view))), np.finfo(np.float64).tiny)
        if np.any(new):
            added = missing @ vectors[:, new] / np.sqrt(values[new])
            if self._basis is not None:
                added -= self._basis @ (self._basis.T @ added)
            added = np.linalg.qr(added)[0]
            total = columns(mapping @ added)
            self._basis = added if self._basis is None else np.column_stack([self._basis, added])
            self._total = total if self._total is None else np.column_stack([self._total, total])
        assert self._basis is not None and self._total is not None
        total = coefficient_map.T @ (self._total @ (self._basis.T @ view))
        return 0.5 * (total + total.T) - coefficient_map.T @ fixed @ coefficient_map


INDEPENDENT_EFFECTS = CurvatureCorrection()


@dataclass(frozen=True)
class _Anchor:
    """The EP evidence's local model about the fixed point x_k where ``correction`` (C = B - A) was solved: EP
    stationarity makes E and the fixed-cavity F_k share their gradient at x_k, with Hessians -B and -A, so
    E(x) = F_k(x) - 1/2 (x - x_k)'C(x - x_k) + const + O(|x - x_k|^3) (theory-ep). The objective V integrates is
    that model minus the penalty, whose curvature A + C + S is the one V's determinant takes: its maximizer, its
    responses and its line integrals then belong to the same integrand. ``center`` is z_k = M x_k, the same in every
    view; ``matrix``, ``linear`` and ``constant`` are this prior's K'CK, K'C x_k and x_k'C x_k (x = K x_view), so the
    term is -(1/2 x'(K'CK)x - (K'C x_k)'x + 1/2 x_k'C x_k)."""

    correction: CurvatureCorrection
    center: F64Array
    matrix: F64Array
    linear: F64Array
    constant: float


def _anchored(prior: ScaleMixturePrior, correction: CurvatureCorrection, center: F64Array) -> ScaleMixturePrior:
    """``prior`` with the local model about z_k = ``center`` in its own coordinates (``_Anchor``): C on the prior's
    directions and on x_k's, from the correction's own lazy solve."""
    size = prior.coefficient_size
    blocks = correction.on(np.column_stack([prior.coefficient_map, center]))
    return replace(prior, anchor=_Anchor(correction, center, blocks[:size, :size], blocks[:size, size], float(blocks[size, size])))


def _anchor_value(prior: ScaleMixturePrior, coefficients: F64Array) -> tuple[float, F64Array]:
    """1/2 (x - x_k)'C(x - x_k) and its gradient in the prior's coordinates (zero without an anchor)."""
    anchor = prior.anchor
    if anchor is None:
        return 0.0, np.zeros_like(coefficients)
    moved = anchor.matrix @ coefficients
    return 0.5 * float(coefficients @ moved) - float(anchor.linear @ coefficients) + 0.5 * anchor.constant, moved - anchor.linear


def curvature_correction(
    prior: ScaleMixturePrior, coefficients: F64Array, cavity: Cavity, posterior: GaussianPosterior, working_bytes: int, tolerance: float
) -> CurvatureCorrection:
    """C = B - A at the EP fixed point with q's responses ``posterior``, solved lazily for the directions asked
    (``CurvatureCorrection``): one linear-response solve per outer step on the free coefficients, and one more per
    released edge.

    B to the tolerance over the dimension (a relative error e in B moves log|B + S| by at most D e for well-scaled
    B + S), never past what double precision resolves."""
    relative_tolerance = max(tolerance / coefficients.shape[0], _EPSILON)
    fixed = -_data_objective(prior, coefficients, cavity, working_bytes).hessian
    achieved: list[float] = []
    return CurvatureCorrection(
        mapping=prior.coefficient_map,
        columns=lambda directions: _total_curvature_columns(prior, coefficients, cavity, posterior, working_bytes, relative_tolerance, directions, achieved),
        fixed_curvature=0.5 * (fixed + fixed.T),
        resolutions=achieved,
    )


def diagonal_posterior(variance: F64Array) -> GaussianPosterior:
    """The posterior of independent effects (orthogonal design, normal means): Sigma = diag(variance). There the
    cavities do not move with the prior, so B equals the fixed-cavity curvature."""
    column = np.asarray(variance, dtype=np.float64)[:, None]
    return GaussianPosterior(solve=lambda right, _relative_tolerance: column * right, variance_jvp=lambda weights: -np.square(column) * weights, exact=True)


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
    log_density = class_log_density(prior, coefficients)
    for _class, rows, kernel_rows in _kernel_chunks(prior, scales, cavity, working_bytes):
        terms = kernel_rows.components(log_density[_class])
        weights = terms.responsibility
        conditional = terms.conditional_variance
        retained = kernel_rows.retained
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
    """B in x: M' B_z M (``_total_curvature_columns`` on every direction)."""
    directions = prior.coefficient_map
    total = directions.T @ _total_curvature_columns(prior, coefficients, cavity, posterior, working_bytes, relative_tolerance, directions)
    return 0.5 * (total + total.T)


def _total_curvature_columns(
    prior: ScaleMixturePrior,
    coefficients: F64Array,
    cavity: Cavity,
    posterior: GaussianPosterior,
    working_bytes: int,
    relative_tolerance: float,
    directions: F64Array,
    achieved: list[float] | None = None,
) -> F64Array:
    """B_z E for the given z-space directions E (columns): B = -d2 log Z_EP / dz2 with EP re-solved (speed-ep,
    B_PRODUCTS.md), without re-solving EP; B in x on a view is (M K)' B_z (M K).

    B_z E = A E - m_x' dh + s2_x' dP / 2 (d grad_x log Z_j / dh_j = m_x and d grad_x log Z_j / dP_j = -s2_x / 2, with
    B the negative derivative), where the cavity response (dh, dP) to a direction E solves the linear response of the
    EP fixed point:
        (Q + diag tau) dm = (m + m_P / v) dP + m_x E / v          (``posterior.solve``)
        dh = (dm - m_P dP - m_x E) / v
        v^2 dP = -(Sigma o Sigma)_off (dv / v^2 + dP),  dv = v_h dh + v_P dP + v_x E   (``posterior.variance_jvp``)
    dP is the fixed point of that affine map, found by GMRES on all directions at once to ``relative_tolerance``, with
    the restart length that fits ``working_bytes``. ``achieved``, when given, receives the relative residual the
    solve reached (its rounding where the posterior is ``exact``).
    """
    derivatives = _variant_derivatives(prior, coefficients, cavity, working_bytes)
    mean_by_z = _through_z(prior, derivatives.mean_by_density, derivatives.mean_by_log_scale, directions)
    variance_by_z = _through_z(prior, derivatives.second_by_density, derivatives.second_by_log_scale, directions) - 2.0 * derivatives.mean[:, None] * mean_by_z
    if posterior.cavity_response is not None:
        # The fixed point's own cavity response (a mean-field fixed point: ``mean_field``), exact by construction.
        shift_step, precision_step = posterior.cavity_response(mean_by_z, variance_by_z)
        if achieved is not None:
            achieved.append(_EPSILON if posterior.exact else relative_tolerance)
        return _total_from_response(prior, coefficients, cavity, derivatives, directions, shift_step, precision_step, working_bytes)
    solve, variance_jvp = posterior.solve, posterior.variance_jvp
    assert solve is not None and variance_jvp is not None
    # A variant whose tilted law is a point mass at zero (all its prior mass where v = u e^t underflows to 0) does not
    # respond: its mean and every derivative are zero for every cavity, so dh = dP = 0 exactly (the limit of the map,
    # whose 1/v factors are 0/0 there). Its rows are identity rows of the fixed point, with zero offset (speed-smalln).
    live = derivatives.variance > 0.0
    inverse = np.where(live, 1.0 / np.where(live, derivatives.variance, 1.0), 0.0)
    inverse_column = inverse[:, None]
    live_column = live[:, None].astype(np.float64)

    def through(precision_step: F64Array, inner: float, affine: bool = True) -> tuple[F64Array, F64Array, F64Array]:
        # The map is affine in dP. With ``affine`` False its constant (the E terms) is dropped, which leaves its linear
        # part exactly: GMRES applies that, so each solve inside is accurate relative to what it applies, not to the
        # constant, as a difference of two solves would be (speed-recycle a7752ae).
        # A scalar zero, so the linear part applies to any number of columns (a block Krylov step's).
        mean_constant = mean_by_z if affine else 0.0
        mean_step = solve(
            (derivatives.mean + derivatives.mean_by_precision * inverse)[:, None] * precision_step + mean_constant * inverse_column, inner
        )
        shift_step = (mean_step - derivatives.mean_by_precision[:, None] * precision_step - mean_constant) * inverse_column
        variance_step = derivatives.variance_by_shift[:, None] * shift_step + derivatives.variance_by_precision[:, None] * precision_step
        if affine:
            variance_step = variance_step + variance_by_z
        response = variance_step * inverse_column**2 + live_column * precision_step
        return shift_step, response + variance_jvp(response) * inverse_column**2, response

    shape = mean_by_z.shape
    # ``through`` is affine in dP, with linear part (I - diag(1/v^2) (Sigma o Sigma)) R and R = diag(v_h / v^3) Sigma
    # diag(m + m_P / v) + diag(1 + v_P / v^2 - v_h m_P / v^3) (a point mass's rows the identity's).
    left = derivatives.variance_by_shift * inverse**3
    gain = derivatives.mean + derivatives.mean_by_precision * inverse
    diagonal = live + derivatives.variance_by_precision * inverse**2 - derivatives.variance_by_shift * derivatives.mean_by_precision * inverse**3
    weight = inverse**2
    if posterior.linear_response is not None:
        # The posterior solves its fixed point exactly.
        _shift, offset, _start = through(np.zeros(shape), relative_tolerance)
        try:
            precision_step = posterior.linear_response(left, gain, diagonal, weight, offset)
            if achieved is not None:
                achieved.append(_EPSILON if posterior.exact else relative_tolerance)
        except np.linalg.LinAlgError as error:
            # I - L singular: the EP fixed point is not locally stable, and its linear response does not exist.
            raise LinearResponseError(f"the EP fixed point's linear response is singular: {error}") from error
        return _total_from_response(prior, coefficients, cavity, derivatives, directions, through(precision_step, relative_tolerance)[0], precision_step, working_bytes)
    # The linear part applies one p x p operator to every direction column, so the solve is block Krylov over the
    # columns (``krylov_recycle.block_gcro_dr``, speed-recycle): each application serves them all, restarts keep the
    # slowest harmonic Ritz space, and ``local_response`` (read-free) preconditions it; it stops where a cycle no longer
    # lowers its residual. Each product solves the posterior only to ``inner``, so the operator itself errs,
    # and the Krylov residual estimate can sit far below the true one (inexact Krylov: Simoncini and Szyld, SIAM J. Sci.
    # Comput. 25, 2003): half the tolerance goes to the Krylov solve, half to the products. The true residual is
    # measured once the solve meets its own tolerance; while it exceeds the tolerance, the inner solves tighten by the
    # measured excess (the operator's own amplification of their error) and the solve continues from where it stopped.
    precondition = None if posterior.local_response is None else posterior.local_response(left, gain, diagonal, weight)
    inner = relative_tolerance
    solution = np.zeros(shape)
    floor_residual = np.inf
    while True:
        try:
            _shift, offset, start_response = through(np.zeros(shape), inner)
        except ValueError as error:
            if inner >= relative_tolerance:
                raise
            # The solver cannot reach the tightened accuracy in float64: the response is not resolvable here.
            raise LinearResponseError(f"the EP fixed point's linear response cannot be resolved: {error}") from error
        # The map's last step cancels the diagonal of Sigma o Sigma against v^2: its value is known only to eps times
        # the terms that cancel, which is where the residual can stop.
        rounding = _EPSILON * float(np.linalg.norm(start_response))

        def linear_part(values: F64Array, inner: float = inner) -> F64Array:
            return values - through(values, inner, affine=False)[1]

        try:
            result = block_gcro_dr(
                linear_part, offset, relative_tolerance=0.5 * relative_tolerance, absolute_tolerance=rounding, working_bytes=working_bytes,
                start=solution, precondition=precondition,
            )
        except (FloatingPointError, ValueError) as error:
            raise LinearResponseError(f"the EP fixed point's linear response did not converge: {error}") from error
        solution = result.solution
        target = max(relative_tolerance * float(np.linalg.norm(offset)), rounding)
        residual = result.residual_norm
        if residual <= target:
            if achieved is not None:
                achieved.append(max(residual, rounding) / max(float(np.linalg.norm(offset)), np.finfo(np.float64).tiny))
            break
        # The residual is measured with products at ``inner``, so it carries their error: the inner solves tighten by
        # the measured excess each round, which ends where they reach float64's attainable accuracy rather than on one
        # noisy comparison (speed-recycle: 3.61 then 4.26 against 3.34). That end is either the solver refusing (above)
        # or, where it returns a certificate above the request instead (speed-krylov cf364bd), float64's unit roundoff:
        # no solve delivers a request below it, so the solves stay there, and the Krylov solve continues from its true
        # residual for as long as each such round lowers it.
        tightened = inner * 0.5 * target / residual
        if tightened < _EPSILON:
            if not residual < floor_residual:
                raise LinearResponseError(
                    f"the EP fixed point's linear response cannot be resolved in float64: its true residual {residual:.3e} stays above "
                    f"{target:.3e} with the posterior solves at float64's unit roundoff"
                )
            floor_residual, tightened = residual, _EPSILON
        inner = tightened
    return _total_from_response(prior, coefficients, cavity, derivatives, directions, through(solution, inner)[0], solution, working_bytes)


def _total_from_response(
    prior: ScaleMixturePrior, coefficients: F64Array, cavity: Cavity, derivatives: _VariantDerivatives, directions: F64Array,
    shift_step: F64Array, precision_step: F64Array, working_bytes: int,
) -> F64Array:
    """B_z E from the cavity response (dh, dP) to each direction E (``_total_curvature_columns``)."""
    fixed_cavity = -_data_objective(prior, coefficients, cavity, working_bytes).hessian
    return fixed_cavity @ directions - _through_z_transposed(prior, derivatives.mean_by_density, derivatives.mean_by_log_scale, shift_step) + 0.5 * (
        _through_z_transposed(prior, derivatives.second_by_density, derivatives.second_by_log_scale, precision_step)
    )


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
    """V and dV/drho at x_rho; ``responses`` is dx_rho/drho (coefficients x weights), the first-order predictor of x.

    ``laplace_value`` is the Laplace form, whose exact rho-gradient ``gradient`` is; ``value`` is what every
    comparison uses: the Laplace form from ``_evidence``, and after ``_corrected`` the Tierney-Kadane-certified V.
    ``error`` bounds |value - V| by the sum of its sources' certified bounds: the inner maximizer, the determinant's
    rounding and B's linear response (``_evidence``), and after ``_corrected`` the Tierney-Kadane remainder too.
    """

    value: float
    laplace_value: float
    error: float
    gradient: F64Array
    coefficients: F64Array
    responses: F64Array
    penalized_value: float
    newton_decrement: float
    magnitude: float
    # -H at x_rho and the inner maximizer's own decrement 1/2 g'(-H)^-1 g there: x_rho lies within sqrt(2 d) of the
    # maximum it approximates in that metric, which is how two starts are recognized as one basin.
    precision: F64Array
    inner_decrement: float
    # Per weight, the two rho-dependent parts of dV/drho_i: the effective degrees of freedom
    # lambda_i tr((B + S)^-1 S_i) and the penalty's size lambda_i ||R_i x||^2.
    effective_degrees: F64Array
    penalty_sizes: F64Array
    # newton_decrement is 1/2 g'(B + S)^-1 g: what the fit's certificate records (math-epeb: the fixed-cavity form
    # understates the remaining gain by up to 100x).
    # After ``_corrected``: the standardized directions whose Laplace terms it replaced by line integrals, and the
    # share of the tolerance each was resolved to (their slopes keep the count, ``_correction_slopes``).
    replaced_directions: F64Array | None = None
    replaced_share: float = 0.0
    # The corrections' own part of ``error`` (``_corrected``: the directions left to the Laplace term and the
    # integrals' resolution), so a later ``_corrected`` at another tolerance starts from the Laplace error.
    correction_remainder: float = 0.0


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

    Without an annotation design b_e = 0, so theta' = b_c and d3, d4 are the third cumulant and the fourth cumulant
    of b_c under w_j: every direction at once from the raw moments R_r = w_j' (b_c - b_c[m])^r (one GEMM per power),
    taken about each variant's modal node m, where they do not cancel (the variants grouped by that node).
    """
    mapping = prior.coefficient_map
    directions_z = mapping @ directions
    grid_size = prior.grid_size
    scale_span = slice(prior.density_size, prior.density_size + prior.scale_size)
    third = np.zeros(directions.shape[1])
    fourth = np.zeros(directions.shape[1])
    density = np.exp(class_log_density(prior, coefficients))
    if not prior.scale_size:
        for class_position, _rows, terms in _class_terms(prior, coefficients, cavity, working_bytes):
            steps = directions_z[class_position * grid_size : (class_position + 1) * grid_size]
            weights = terms.responsibility
            modes = np.argmax(weights, axis=1)
            for node in np.unique(modes):
                offset = steps - steps[node][None, :]
                node_weights = weights[modes == node]
                mean = node_weights @ offset
                second = node_weights @ np.square(offset)
                cubed = node_weights @ offset**3
                central_second = second - mean * mean
                third += np.sum(cubed - 3.0 * mean * second + 2.0 * mean**3, axis=0)
                fourth += np.sum(
                    node_weights @ offset**4 - 4.0 * mean * cubed + 6.0 * mean * mean * second - 3.0 * mean**4 - 3.0 * central_second**2, axis=0
                )
    for class_position, rows, terms in (() if not prior.scale_size else _class_terms(prior, coefficients, cavity, working_bytes)):
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


def _line(
    prior: ScaleMixturePrior, log_smoothing: F64Array, origin: F64Array, direction: F64Array, cavity: Cavity, working_bytes: int
) -> Callable[[F64Array], F64Array]:
    """The penalized objective F(x + t b) - P(x + t b) as a function of the steps t, each call one pass over the
    variants for all its steps.

    z = M x is affine in t, so every step's class log densities and log scales follow from those of x and b, and P
    is exactly quadratic in t. A variant whose log scale does not move along b keeps its kernel row L_jk, so its
    log Z_j(t) = LSE_k(L_jk + log pi_ck(t)) is one product of exp(L - max) with exp(log pi(t) - max) over the nodes
    for all its steps (a GEMM of positive terms: exact to K eps in relative terms), with the rows whose product
    falls to where subnormal terms could matter taken exactly. The other variants' kernels are evaluated per step."""
    density, _scale = _density_and_scale(prior, origin)
    density_step, scale_step = _density_and_scale(prior, direction)
    scales = log_scale(prior, origin)
    scale_slope = prior.scale_design @ scale_step
    penalty, penalty_gradient = _penalty_value(prior, log_smoothing, origin)
    penalty_slope = float(penalty_gradient @ direction)
    penalty_curvature = float(direction @ _penalty_matrix(prior, log_smoothing) @ direction)
    if prior.anchor is not None:
        # The local model's C term is quadratic in t too (``_Anchor``).
        anchor_value, anchor_gradient = _anchor_value(prior, origin)
        penalty += anchor_value
        penalty_slope += float(anchor_gradient @ direction)
        penalty_curvature += float(direction @ prior.anchor.matrix @ direction)
    fixed_rows, moving_rows, kernels = [], [], []
    for class_position, class_rows in enumerate(prior.class_rows):
        still = class_rows[scale_slope[class_rows] == 0.0]
        fixed_rows.append(still)
        moving_rows.append(class_rows[scale_slope[class_rows] != 0.0])
        rows_kernels = []
        for rows in _row_chunks(still, prior.grid_size, working_bytes):
            # L_jk: the kernel's log with a flat class density (its log pi part enters per step).
            row_kernel = _kernel_terms(
                np.zeros(prior.grid_size), scales[rows], prior.log_variance_grid, cavity.precision[rows], cavity.shift[rows]
            )[3]
            peak = np.max(row_kernel, axis=1)
            rows_kernels.append((rows, peak, np.exp(row_kernel - peak[:, None]), row_kernel))
        kernels.append(rows_kernels)

    def values(steps: F64Array) -> F64Array:
        count = steps.shape[0]
        log_weights = density[None] + steps[:, None, None] * density_step[None]
        log_density = log_weights - _log_sum_exp(log_weights, axis=2, keepdims=True)
        total = -(penalty + steps * penalty_slope + 0.5 * np.square(steps) * penalty_curvature)
        for class_position in range(prior.class_count):
            class_density = log_density[:, class_position]
            density_peak = np.max(class_density, axis=1)
            scaled = np.exp(class_density - density_peak[:, None]).T
            for rows, peak, exponentials, row_kernel in kernels[class_position]:
                products = exponentials @ scaled
                # Past tiny / eps a product's subnormal terms could matter at double precision: taken exactly there.
                lost = products < np.finfo(np.float64).tiny / _EPSILON
                with np.errstate(divide="ignore"):
                    normalizers = np.log(products) + peak[:, None] + density_peak[None, :]
                if np.any(lost):
                    row_index, step_index = np.nonzero(lost)
                    normalizers[row_index, step_index] = _log_sum_exp(row_kernel[row_index] + class_density[step_index], axis=1)
                total += normalizers.sum(axis=0)
            moving = moving_rows[class_position]
            for rows in _row_chunks(moving, prior.grid_size * count, working_bytes):
                size = rows.shape[0] * count
                normalizers = _log_normalizers(
                    np.broadcast_to(class_density[None], (rows.shape[0], count, prior.grid_size)).reshape(size, prior.grid_size),
                    (scales[rows][:, None] + scale_slope[rows][:, None] * steps[None, :]).reshape(size),
                    prior.log_variance_grid, np.repeat(cavity.precision[rows], count), np.repeat(cavity.shift[rows], count),
                )
                total += normalizers.reshape(rows.shape[0], count).sum(axis=0)
        return total

    return values


def _line_log_integrals(
    prior: ScaleMixturePrior, log_smoothing: F64Array, origin: F64Array, directions: F64Array, value: float, cavity: Cavity, working_bytes: int, share: float
) -> F64Array:
    """For each standardized direction b (a column, unit curvature at the maximum x): the log of the line integral
    of exp(F - P - value) over its Laplace term sqrt(2 pi), each to ``share`` in its log.

    QUADPACK's rule for an infinite range, vectorized: QAGI folds both half-lines onto u in (0, 1] by
    t = (1 - u) / u, and each interval takes QK15I's 15-point Kronrod rule with its embedded 7-point Gauss rule and
    QUADPACK's own error estimate (Piessens et al. 1983). Every round evaluates the nodes of every open interval of
    every direction in one pass over the variants per direction (``_line``), bisects the intervals whose error is
    over their share of the target by width, and stops a direction once its errors sum to ``share`` of its integral
    (or to half of double precision when rounding is what stops it). It raises when an interval can no longer be
    halved, as QUADPACK reports a failure. (QUADPACK's epsilon extrapolation is not used: the integrands are
    analytic, and bisection alone meets the tolerance.)

    Gauss-Hermite rules were tried first and refused: along the replaced directions the integrand falls off a cliff
    on one side, and consecutive rules agreed to the share at values up to 560 shares from the integral in over a
    tenth of the cases [sim-only, e2e fastline diagnostic], so no agreement of fixed rules certifies it here.
    """
    count = directions.shape[1]
    lines = [_line(prior, log_smoothing, origin, directions[:, column], cavity, working_bytes) for column in range(count)]
    tolerance = max(share, _HALF_PRECISION)
    nodes = np.concatenate([-_KRONROD_NODES[:-1], _KRONROD_NODES[::-1]])
    kronrod = np.concatenate([_KRONROD_WEIGHTS[:-1], _KRONROD_WEIGHTS[::-1]])
    gauss = np.concatenate([_GAUSS_WEIGHTS[:-1], _GAUSS_WEIGHTS[::-1]])
    intervals = [(np.array([0.0]), np.array([1.0])) for _column in range(count)]
    done_value, done_error = np.zeros(count), np.zeros(count)
    logs = np.full(count, np.nan)
    open_ = list(range(count))
    while open_:
        still_open = []
        for column in open_:
            lows, highs = intervals[column]
            centres, halves = 0.5 * (lows + highs), 0.5 * (highs - lows)
            points = centres[:, None] + halves[:, None] * nodes[None, :]
            steps = (1.0 - points) / points
            both = lines[column](np.concatenate([steps.ravel(), -steps.ravel()])) - value
            with np.errstate(over="ignore", invalid="ignore"):
                folded = (np.exp(both[: steps.size]) + np.exp(both[steps.size :])).reshape(steps.shape) / np.square(points)
            if not np.all(np.isfinite(folded)):
                # The objective along this line rises past the maximum's value by more than double precision holds:
                # x is not its maximum along it (an anchored model's indefinite C at a freed weight), and every error
                # estimate below would be nan, which bisection would take for "unsettled" without end.
                raise FloatingPointError("the exact integral along a direction is not finite: the objective rises past its value at x")
            kronrod_value = folded @ kronrod
            gauss_value = folded @ gauss
            # QUADPACK's error estimate (dqk15i): |K - G| scaled against the integrand's mean absolute deviation.
            deviation = np.abs(folded - 0.5 * kronrod_value[:, None]) @ kronrod
            error = np.abs(kronrod_value - gauss_value) * halves
            scale = deviation * halves
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(scale > 0.0, np.minimum(1.0, (_QUADPACK_ERROR_SCALE * error / np.where(scale > 0.0, scale, 1.0)) ** _QUADPACK_ERROR_POWER), 1.0)
            error = np.where((scale > 0.0) & (error > 0.0), scale * ratio, error)
            error = np.maximum(error, _QUADPACK_RELATIVE_FLOOR * np.abs(folded) @ kronrod * halves)
            estimates = kronrod_value * halves
            total = done_value[column] + float(np.sum(estimates))
            target = tolerance * total
            if done_error[column] + float(np.sum(error)) <= target:
                logs[column] = float(np.log(total) - 0.5 * np.log(2.0 * np.pi))
                continue
            settled = error <= target * (highs - lows)
            done_value[column] += float(np.sum(estimates[settled]))
            done_error[column] += float(np.sum(error[settled]))
            lows, highs = lows[~settled], highs[~settled]
            if np.any((highs - lows) <= _EPSILON * np.maximum(highs, _EPSILON)):
                raise FloatingPointError("the exact integral along a direction did not converge: an interval cannot be halved further")
            middles = 0.5 * (lows + highs)
            intervals[column] = (np.concatenate([lows, middles]), np.concatenate([middles, highs]))
            if 2 * 2 * lows.shape[0] * nodes.shape[0] * prior.variant_count * np.dtype(np.float64).itemsize > working_bytes:
                # The next round's pass (its normalizers, one per variant and step, for both half-lines) would not
                # fit the working budget: the integrand is not resolved to the share within what the budget can
                # evaluate, and bisection alone would grow the open intervals without end.
                raise FloatingPointError("the exact integral along a direction did not converge within the working budget")
            still_open.append(column)
        open_ = still_open
    return logs


def _line_log_integral(
    prior: ScaleMixturePrior, log_smoothing: F64Array, origin: F64Array, direction: F64Array, value: float, cavity: Cavity, working_bytes: int, share: float
) -> float:
    """``_line_log_integrals`` for one direction."""
    return float(_line_log_integrals(prior, log_smoothing, origin, direction[:, None], value, cavity, working_bytes, share)[0])


# One hyper step's exact repeats are answered once: the structural starts of each basin search land in one basin, and
# the edge comparisons re-evaluate the models they came from (gene 1: 9 of 31 corrections repeated an earlier one bit
# for bit [real, tk-closedform]). Keyed by the bytes of every input; held in the step's ``_STEP_CACHE``, so they end
# with the step, and dropped when the cavity changes within it.


def _digest(*values: object) -> bytes:
    hasher = hashlib.blake2b()
    for value in values:
        if isinstance(value, np.ndarray):
            hasher.update(repr((value.dtype.str, value.shape)).encode())
            hasher.update(np.ascontiguousarray(value).tobytes())
        else:
            hasher.update(repr(value).encode())
    return hasher.digest()


def _prior_digest(prior: ScaleMixturePrior) -> tuple[object, ...]:
    """What a view's evidence depends on besides x and rho: its map, blocks, null space and lattice."""
    blocks = tuple(value for block in prior.smoothing_blocks for value in (block.coordinates, block.factor))
    return (
        prior.coefficient_map, prior.null_basis, prior.log_variance_grid, prior.kernel_floor, prior.class_index, prior.log_variance_offset, prior.scale_design,
        prior.pooled_size, prior.offset_groups, *blocks,
    )


def _repeated(name: str, cavity: Cavity, inputs: tuple[object, ...], compute: Callable[[], object], held: object | None = None) -> object:
    """compute() once per distinct ``inputs`` at this cavity within a hyper step (and, when given, this ``held``
    object: compared by identity); outside a hyper step, compute() every time."""
    cache = _STEP_CACHE.get()
    if cache is None:
        return compute()
    repeats = cache.setdefault("repeats", {})
    scope = _digest(cavity.precision, cavity.shift)
    if repeats.get("cavity") != scope:
        repeats.clear()
        repeats["cavity"] = scope
    if held is not None and repeats.get(("held", name)) is not held:
        for key in [key for key in repeats if isinstance(key, tuple) and key[0] == name]:
            del repeats[key]
        repeats[("held", name)] = held
    key = (name, _digest(*inputs))
    if key not in repeats:
        repeats[key] = compute()
    return repeats[key]


def _laplace_corrections(
    prior: ScaleMixturePrior, log_smoothing: F64Array, evidence: _Evidence, cavity: Cavity, correction: CurvatureCorrection, working_bytes: int, tolerance: float
) -> tuple[F64Array, F64Array, F64Array]:
    """``_laplace_corrections_once``, each distinct input once per cavity (and per correction, held by identity). The
    corrections depend on the evidence's -H and responses, which come from the correction, so those are keyed too:
    equal coefficients can come with a different -H once a lazy correction has grown (e2e review)."""
    return _repeated(
        "corrections", cavity,
        (*_prior_digest(prior), log_smoothing, evidence.coefficients, evidence.precision, evidence.responses, working_bytes, tolerance),
        lambda: _laplace_corrections_once(prior, log_smoothing, evidence, cavity, correction, working_bytes, tolerance), held=correction,
    )


def _laplace_corrections_once(
    prior: ScaleMixturePrior, log_smoothing: F64Array, evidence: _Evidence, cavity: Cavity, correction: CurvatureCorrection, working_bytes: int, tolerance: float
) -> tuple[F64Array, F64Array, F64Array]:
    """Per integrated direction, the correction from the Laplace term to the exact one-dimensional integral, the
    Tierney-Kadane term that decided it, and the standardized directions themselves (columns, in x).

    The integrated directions are the eigenvectors of -H's Schur complement on the complement of the profiled null
    space, each moved with the null coordinates' first-order response and scaled to unit curvature. Along a
    standardized direction the Tierney-Kadane O(1) term is k4/8 + 5 k3^2/24. V is certified to ``tolerance`` in
    total: half of it bounds the directions left to the Laplace term (the largest terms are replaced until the
    remaining ones sum to at most tolerance / 2), and half bounds the quadratures of the replaced ones (each to
    tolerance / (2 m) in its log, m of them). A replaced direction's correction is the log of the exact line
    integral's ratio to the Laplace term (its limit covers a density collapsing to a point); elsewhere it is 0.
    """
    standardized = _standardized(prior, log_smoothing, evidence.coefficients, cavity, working_bytes)
    if standardized is None:
        raise FloatingPointError("the Schur complement of -H is not positive definite at x")
    value, terms, directions = standardized
    corrections = np.zeros(directions.shape[1])
    order = np.argsort(-np.abs(terms))
    # Remaining sums from the smallest term up: the replaced set is the shortest prefix of ``order`` whose
    # complement sums to at most tolerance / 2.
    remaining = np.concatenate([np.cumsum(np.abs(terms[order])[::-1])[::-1], [0.0]])
    replaced = order[: int(np.argmax(remaining <= 0.5 * tolerance))]
    share = 0.5 * tolerance / max(replaced.shape[0], 1)
    if replaced.shape[0]:
        corrections[replaced] = _line_log_integrals(prior, log_smoothing, evidence.coefficients, directions[:, replaced], value, cavity, working_bytes, share)
    return corrections, terms, directions


def _standardized(
    prior: ScaleMixturePrior, log_smoothing: F64Array, coefficients: F64Array, cavity: Cavity, working_bytes: int
) -> tuple[float, F64Array, F64Array] | None:
    """(penalized value, Tierney-Kadane terms, standardized directions) at x: the eigenvectors of -H's Schur
    complement on the complement of the profiled null space, each moved with the null coordinates' first-order
    response and scaled to unit curvature, and each one's O(1) term k4/8 + 5 k3^2/24 from exact third and fourth
    derivatives. None where the Schur complement is not positive definite (x is not inside a basin)."""
    penalty = _penalty_matrix(prior, log_smoothing)
    objective = _data_objective(prior, coefficients, cavity, working_bytes)
    value, _gradient, hessian = _penalized(prior, objective, log_smoothing, penalty, coefficients)
    negative = -hessian
    null_basis = prior.null_basis
    complement = np.linalg.svd(np.eye(negative.shape[0]) - null_basis @ null_basis.T)[0][:, : negative.shape[0] - null_basis.shape[1]]
    response = np.eye(negative.shape[0])
    if null_basis.shape[1]:
        response = response - null_basis @ np.linalg.solve(null_basis.T @ negative @ null_basis, null_basis.T @ negative)
    moved = response @ complement
    schur = moved.T @ negative @ moved
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (schur + schur.T))
    if eigenvalues.size and eigenvalues[0] <= 0.0:
        return None
    # An eigenvalue below eps times the largest is rounding, raised to that floor as in ``_ascent_direction``.
    if eigenvalues.size:
        eigenvalues = np.maximum(eigenvalues, _EPSILON * float(np.max(np.abs(eigenvalues))))
    directions = moved @ eigenvectors / np.sqrt(eigenvalues)[None, :]
    third, fourth = _directional_derivatives(prior, coefficients, cavity, directions, working_bytes)
    return value, fourth / 8.0 + 5.0 * third**2 / 24.0, directions


def _correction_value(
    prior: ScaleMixturePrior, log_smoothing: F64Array, coefficients: F64Array, cavity: Cavity, working_bytes: int, count: int, share: float
) -> float | None:
    """The corrections' sum at x with the ``count`` largest terms replaced (``_corrected``'s rule at a fixed count),
    each line integral resolved to ``share`` in its log; None where x is not inside a basin or an integral fails."""
    standardized = _standardized(prior, log_smoothing, coefficients, cavity, working_bytes)
    if standardized is None:
        return None
    value, terms, directions = standardized
    if count == 0:
        return 0.0
    chosen = np.argsort(-np.abs(terms))[:count]
    try:
        return float(np.sum(_line_log_integrals(prior, log_smoothing, coefficients, directions[:, chosen], value, cavity, working_bytes, share)))
    except FloatingPointError:
        return None


def _coarse_targets(evidence: _Evidence, interior: F64Array, tolerance: float) -> F64Array:
    """The correction slopes' error per weight before the curvature is known: with s bounding |V''| (MODEL.md S4) and
    n interior weights, E^2 / (2 s) = tolerance / (4 n) leaves the error a quarter of the tolerance's gain at that
    curvature (``_stationarity`` tightens it once the difference curvature is in)."""
    count = max(int(np.count_nonzero(interior)), 1)
    scale = np.maximum(0.5 * (evidence.effective_degrees + evidence.penalty_sizes), _EPSILON * evidence.magnitude)
    return np.sqrt(0.5 * tolerance * scale / count)


def _correction_slopes(
    view: ScaleMixturePrior, weights: F64Array, evidence: _Evidence, interior: F64Array, cavity: Cavity, working_bytes: int, targets: F64Array
) -> tuple[F64Array, F64Array, F64Array]:
    """The rho-slopes of V's corrections at a corrected evidence, their error estimates, and their second
    differences, per interior weight (zero elsewhere).

    The corrections are the sum of the replaced directions' line-integral ratios to their Laplace terms, along the
    Schur complement's standardized eigenvectors, which rotate with rho: holding them fixed gave the slope's wrong
    sign where V was then climbing the other way [sim-only, e2e: +0.25 held against +0.04 V's own]. So the slope is
    V's corrections' own, by central differences in rho_i with x on its first-order path x + h dx/drho_i (the
    corrections are smooth there, and no inner maximum is re-solved, so no fold can intervene; x's second-order
    error cancels in the central difference), at the same count of replaced directions. Each correction is resolved
    to a share e / m of its log (m integrals) with e set so the difference errs by about ``targets`` (the slope error
    the certificate can carry, ``_stationarity``), never below the rounding of the line values (eps times the
    objective's magnitude), and at the step h = (3 e / s)^(1/3) that balances truncation h^2 s / 6 against e / h (s the scale of
    V's derivatives, MODEL.md S4); the differences at h and h / 2 must agree within their two errors, and the step
    halves otherwise (a ranking switch of the replaced directions). Where a side leaves the basin (the Schur
    complement is not positive definite there), the one-sided difference on the other side is used, with its larger
    truncation h s / 2. The error estimate is a posteriori (the h / h/2 agreement); it is not a bound.
    """
    count_weights = weights.shape[0]
    slopes = np.zeros(count_weights)
    errors = np.zeros(count_weights)
    second = np.zeros(count_weights)
    count = 0 if evidence.replaced_directions is None else int(evidence.replaced_directions.shape[1])
    if count == 0:
        return slopes, errors, second
    scale = np.maximum(0.5 * (evidence.effective_degrees + evidence.penalty_sizes), _EPSILON * evidence.magnitude)
    target = np.asarray(targets, dtype=np.float64)
    limit = _HALF_PRECISION * (1.0 + float(np.max(np.abs(weights), initial=0.0)))
    for position in np.flatnonzero(interior):
        unit = np.zeros(count_weights)
        unit[position] = 1.0
        # No integral is resolved past the rounding of the line values it integrates: each exponent is known to eps
        # times the objective's magnitude, and QUADPACK's own floor is 50 eps.
        floor = count * max(_QUADPACK_RELATIVE_FLOOR, _EPSILON * evidence.magnitude)
        accuracy = max(float((2.0 * target[position] / 3.0 ** (2.0 / 3.0)) ** 1.5 / scale[position] ** 0.5), floor)
        share = accuracy / count
        step = max(float((3.0 * accuracy / scale[position]) ** (1.0 / 3.0)), limit)
        centre = _correction_value(view, weights, evidence.coefficients, cavity, working_bytes, count, share)
        if centre is None:
            slopes[position], errors[position] = 0.0, np.inf
            continue
        while True:
            values = {}
            for multiple in (-1.0, -0.5, 0.5, 1.0):
                length = multiple * step
                values[multiple] = _correction_value(
                    view, weights + length * unit, evidence.coefficients + length * evidence.responses[:, position], cavity, working_bytes, count, share
                )
            if all(values[multiple] is not None for multiple in values):
                differences = [(values[1.0] - values[-1.0]) / (2.0 * step), (values[0.5] - values[-0.5]) / step]
                bounds = [step**2 * scale[position] / 6.0 + accuracy / step, (0.5 * step) ** 2 * scale[position] / 6.0 + 2.0 * accuracy / step]
                curvature = (values[1.0] - 2.0 * centre + values[-1.0]) / step**2
            else:
                side = 1.0 if values[1.0] is not None and values[0.5] is not None else -1.0
                if values[side] is None or values[0.5 * side] is None:
                    differences = None
                else:
                    differences = [(values[side] - centre) / (side * step), (values[0.5 * side] - centre) / (0.5 * side * step)]
                    bounds = [step * scale[position] / 2.0 + 2.0 * accuracy / step, 0.25 * step * scale[position] + 4.0 * accuracy / step]
                    curvature = (values[side] - 2.0 * values[0.5 * side] + centre) / (0.5 * step) ** 2
            if differences is not None and abs(differences[0] - differences[1]) <= bounds[0] + bounds[1]:
                slopes[position] = differences[1]
                errors[position] = bounds[1] + abs(differences[0] - differences[1])
                second[position] = curvature
                break
            if step <= limit:
                slopes[position], errors[position] = 0.0, np.inf
                break
            step = max(0.5 * step, limit)
    return slopes, errors, second


def _corrected(
    prior: ScaleMixturePrior, log_smoothing: F64Array, evidence: _Evidence | None, cavity: Cavity, correction: CurvatureCorrection, working_bytes: int, tolerance: float
) -> _Evidence | None:
    """``evidence`` with its value certified to ``tolerance``: the Laplace terms replaced by exact one-dimensional
    integrals along every standardized direction whose Tierney-Kadane term exceeds the tolerance (lead ruling for
    basins, and for every comparison of V: where that term is large the Laplace value is not V to the tolerance).

    It matters most at a fold of the inner maximum, where the data's negative curvature nearly cancels the penalty:
    there -1/2 log|B + S| rises without bound while the integral stays finite, so the Laplace value draws the search
    to the fold [sim-only: 9e10 TK term and a 10-nat correction where the neighbouring basin was 3.6 nats better].
    None when the evidence is None or a line integral cannot be certified.
    """
    if evidence is None:
        return None
    try:
        corrections, _terms, _directions = _laplace_corrections(prior, log_smoothing, evidence, cavity, correction, working_bytes, tolerance)
    except FloatingPointError:
        return None
    # The replaced set is _laplace_corrections' own: the largest terms until the rest sum to at most tolerance / 2.
    order = np.argsort(-np.abs(_terms))
    remaining = np.concatenate([np.cumsum(np.abs(_terms[order])[::-1])[::-1], [0.0]])
    replaced = int(np.argmax(remaining <= 0.5 * tolerance))
    share = 0.5 * tolerance / max(replaced, 1)
    remainder = float(remaining[replaced]) + replaced * max(share, _HALF_PRECISION)
    return replace(
        evidence, value=evidence.laplace_value + float(np.sum(corrections)), error=evidence.error - evidence.correction_remainder + remainder,
        correction_remainder=remainder,
        replaced_directions=_directions[:, order[:replaced]], replaced_share=share,
    )


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


@dataclass(frozen=True)
class _Profiled:
    """A positive definite M with the null space N profiled, from one Cholesky factor in the basis [N, C]:
    ``schur_log_determinant`` = log|M| - log|N'MN|, ``weight`` W = M^-1 - N (N'MN)^-1 N', ``inverse`` M^-1, and
    ``rounding``, a bound on the Schur log-determinant's rounding error.

    With Q'MQ = L L' (Q = [N, C]), the Schur complement's factor is L's trailing block, and L^-1's trailing rows
    G give W = Q G'G Q': the N block cancels by structure, never by subtracting two inverses. So a direction of N
    that the data barely curve (the profiled location and width where the density's mass has no kernel to see it)
    costs nothing: rounding reaches the complement only through M_CN M_NN^-1 M_NC, at second order.
    The computed factor is exact for M + E with |E_ij| <= (n + 1) eps sqrt(M_ii M_jj) in the basis it is computed in
    (Demmel 1989, componentwise, since each row of L has norm sqrt(M_ii)), which moves the Schur log-determinant by
    |tr(W E)| <= (n + 1) eps sum_ij |W_ij| sqrt(M_ii M_jj): that is ``rounding``. It follows the scaled condition
    number, so penalties of disparate scale on different coordinates cost nothing.
    """

    schur_log_determinant: float
    weight: F64Array
    inverse: F64Array
    rounding: float


def _null_complement(null_basis: F64Array) -> F64Array:
    """An orthonormal basis of the complement of the orthonormal ``null_basis`` that keeps every coordinate N does
    not touch as its own unit vector, so the rotation mixes only N's support and the Cholesky factor keeps the
    coordinates' own scales (what the componentwise rounding bound needs)."""
    size = null_basis.shape[0]
    support = np.flatnonzero(np.any(null_basis != 0.0, axis=1))
    inside = null_basis[support]
    local = np.linalg.svd(np.eye(support.shape[0]) - inside @ inside.T)[0][:, : support.shape[0] - null_basis.shape[1]]
    outside = np.setdiff1d(np.arange(size), support)
    complement = np.zeros((size, size - null_basis.shape[1]))
    complement[support, : local.shape[1]] = local
    complement[outside, local.shape[1] + np.arange(outside.shape[0])] = 1.0
    return complement


def _profiled_factor(matrix: F64Array, null_basis: F64Array, complement: F64Array) -> _Profiled:
    """``_Profiled`` for M; raises LinAlgError when M is not resolvably positive definite: an eigenvalue within the
    eigendecomposition's rounding of zero (eps times the dimension times the largest magnitude) is a direction flat
    to double precision, whose sign the factorization cannot resolve (on a weak-data verification case it sat at
    +-1e-14 and V came out as a value or None by its coin toss), and a point with one is not a strict maximum."""
    basis = np.hstack([null_basis, complement])
    rotated = basis.T @ matrix @ basis
    symmetric = 0.5 * (rotated + rotated.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if float(eigenvalues[0]) <= _EPSILON * eigenvalues.shape[0] * max(float(np.max(np.abs(eigenvalues))), np.finfo(np.float64).tiny):
        raise np.linalg.LinAlgError("the profiled curvature is not resolvably positive definite: a direction is flat to rounding")
    factor = np.linalg.cholesky(symmetric)
    inverse_factor = solve_triangular(factor, np.eye(factor.shape[0]), lower=True)
    profiled = null_basis.shape[1]
    trailing = inverse_factor[profiled:]
    rotated_weight = trailing.T @ trailing
    weight = basis @ rotated_weight @ basis.T
    root = np.sqrt(np.diag(rotated))
    return _Profiled(
        schur_log_determinant=2.0 * float(np.sum(np.log(np.diag(factor)[profiled:]))),
        weight=weight,
        inverse=basis @ (inverse_factor.T @ inverse_factor) @ basis.T,
        rounding=_EPSILON * (matrix.shape[0] + 1) * float(np.sum(np.abs(rotated_weight) * np.outer(root, root))),
    )


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
    prior: ScaleMixturePrior,
    log_smoothing: F64Array,
    start: F64Array,
    cavity: Cavity,
    correction: CurvatureCorrection,
    working_bytes: int,
    tolerance: float,
    maximize: bool = True,
) -> _Evidence | None:
    """``_evidence_once``, each distinct input once per cavity (and per correction, held by identity: one per hyper step)."""
    return _repeated(
        "evidence", cavity, (*_prior_digest(prior), log_smoothing, start, working_bytes, tolerance, maximize),
        lambda: _evidence_once(prior, log_smoothing, start, cavity, correction, working_bytes, tolerance, maximize), held=correction,
    )


def _evidence_once(
    prior: ScaleMixturePrior, log_smoothing: F64Array, start: F64Array, cavity: Cavity, correction: CurvatureCorrection, working_bytes: int, tolerance: float, maximize: bool = True
) -> _Evidence | None:
    """V(rho) with the total curvature B, and its exact rho-gradient; x_rho re-maximized from ``start``. None when x_rho is not a strict maximum of the objective V integrates, i.e. B + S is not
    positive definite there (lead ruling: such a point is never accepted, and its V never reported).

    With ``maximize`` false, V's formula is taken at ``start`` itself (an outer state's own x, ``_outer_state``): its
    error then carries x's own decrement there, as the inner maximizer's share does.

    V = F + 1/2 log|S|_+ - 1/2 log|B + S| + 1/2 log|N'(B + S)N|: B = -d2 log Z_EP / dx2 with EP re-solved, the
    second-order approximation of the actual marginal likelihood, taken as A + ``correction`` (``CurvatureCorrection``);
    the null space N is profiled. The gradient is V's own (C held, as V holds it): its determinant terms move with
    rho through S and through x_rho, dx/drho_i = -(-H)^-1 lambda_i S_i x, by the third derivatives of log Z
    contracted with W_B (``_curvature_trace_gradient``); x_rho itself maximizes F - P (less the local model's C term
    where the prior has an anchor, ``_Anchor``, which makes -H = B + S), so F moves only through the penalty (the
    envelope). Every step is still accepted on V itself.
    """
    penalty = _penalty_matrix(prior, log_smoothing)
    null_basis = prior.null_basis
    complement = _null_complement(null_basis)
    # V's determinant terms move with x at first order: an inexact x-hat with decrement d moves V by up to
    # sqrt(c'(-H)^-1 c) sqrt(2 d) / 2, c their x-gradient. The inner tolerance tightens until that is below ``tolerance``.
    inner_tolerance = tolerance
    coefficients = np.array(start, dtype=np.float64, copy=True)
    previous = None
    while True:
        if maximize:
            coefficients, objective = _maximize_coefficients(prior, log_smoothing, coefficients, cavity, working_bytes, inner_tolerance)
        else:
            objective = _data_objective(prior, coefficients, cavity, working_bytes)
        value, gradient, hessian = _penalized(prior, objective, log_smoothing, penalty, coefficients)
        try:
            fixed = _profiled_factor(-hessian, null_basis, complement)
        except np.linalg.LinAlgError:
            return None
        covariance, weight = fixed.inverse, fixed.weight
        newton_decrement = 0.5 * float(gradient @ covariance @ gradient)
        curvature_gradient = _curvature_trace_gradient(prior, coefficients, cavity, weight, working_bytes)
        sensitivity = max(float(curvature_gradient @ covariance @ curvature_gradient), np.finfo(np.float64).tiny)
        rounding = objective.rounding + _EPSILON * abs(value)
        # x-hat's error moves the determinant terms by at most this at first order, and F itself by at most the
        # decrement (the quadratic model's own gain); together, the inner maximizer's share of V's error.
        inner_error = 0.5 * float(np.sqrt(sensitivity * 2.0 * newton_decrement)) + newton_decrement
        if not maximize or 0.5 * np.sqrt(sensitivity * 2.0 * newton_decrement) <= tolerance or newton_decrement <= rounding:
            break
        if previous is not None and np.array_equal(coefficients, previous):
            # The maximizer resolves x no further (its own rounding stop): V is as accurate as double precision gives.
            break
        previous = coefficients
        inner_tolerance = 2.0 * tolerance * tolerance / sensitivity
    penalty_log_determinant = sum(_log_pseudo_determinant(penalty[np.ix_(group, group)]) for group in _penalty_groups(prior))
    # B + S at x_rho: the fixed-cavity A + S there plus the EP-response part held at the fixed point; a relative residual
    # e of that response moves 1/2 log|B + S| by at most D e / 2, charged below at the residual the solve reached
    # (``CurvatureCorrection.resolution``: its rounding where the posterior is exact, 0 for a correction given whole).
    if prior.anchor is not None:
        # With the local model's anchor, -H is already A + C + S (``_Anchor``): C enters once, and x_rho's own factor
        # is the determinant's.
        profiled_total = fixed
    else:
        try:
            profiled_total = _profiled_factor(-hessian + correction.on(prior.coefficient_map), null_basis, complement)
        except np.linalg.LinAlgError:
            return None
    # V is certified to ``tolerance`` only where its determinant is resolved (``_Profiled.rounding``); where that
    # bound exceeds the tolerance (B + S near-singular off the profiled space), the point is not a certified maximum.
    if tolerance > 0.0 and 0.5 * profiled_total.rounding > tolerance:
        return None
    total_covariance = profiled_total.inverse
    evidence_value = value + 0.5 * penalty_log_determinant - 0.5 * profiled_total.schur_log_determinant
    # V's own rho-gradient: W_B = (B + S)^-1 - N (N'(B + S)N)^-1 N' carries both determinants' dependence on rho,
    # through S directly and through x_rho in A(x_rho) (C is held), with dx/drho_i = -(-H)^-1 lambda_i S_i x.
    total_weight = profiled_total.weight
    # The gradient's trace term is one pass over the sites' third derivatives; a view with no finite weight (every
    # block at its edge) has no rho-gradient to take.
    total_curvature_gradient = (
        _curvature_trace_gradient(prior, coefficients, cavity, total_weight, working_bytes) if prior.smoothing_blocks else np.zeros(coefficients.shape[0])
    )
    evidence_gradient = np.empty(len(prior.smoothing_blocks))
    effective_degrees = np.empty(len(prior.smoothing_blocks))
    penalty_sizes = np.empty(len(prior.smoothing_blocks))
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
        # N' S_i = 0, so tr((B + S)^-1 S_i) = tr(W_B S_i), which the profiled factor resolves.
        effective_degrees[position] = lambda_weight * float(np.sum(profiled_total.weight[np.ix_(coordinates, coordinates)] * block.matrix))
        penalty_sizes[position] = lambda_weight * float(residual @ residual)
        evidence_gradient[position] = (
            -0.5 * lambda_weight * float(residual @ residual)
            + 0.5 * lambda_weight * _pseudo_inverse_trace(penalty[np.ix_(group, group)], embedded)
            - 0.5 * lambda_weight * float(np.sum(total_weight[np.ix_(coordinates, coordinates)] * block.matrix))
            + 0.5 * float(total_curvature_gradient @ (covariance @ pull))
        )
    return _Evidence(
        value=evidence_value,
        laplace_value=evidence_value,
        # The inner maximizer, the determinant's rounding, and B's response at the residual its solve reached.
        error=inner_error + 0.5 * profiled_total.rounding + 0.5 * coefficients.shape[0] * correction.resolution,
        gradient=evidence_gradient,
        coefficients=coefficients,
        responses=responses,
        effective_degrees=effective_degrees,
        penalty_sizes=penalty_sizes,
        penalized_value=value,
        newton_decrement=0.5 * float(gradient @ total_covariance @ gradient),
        magnitude=objective.magnitude + abs(evidence_value),
        precision=-hessian,
        inner_decrement=newton_decrement,
    )


def _smoothing_bounds(prior: ScaleMixturePrior, objective: _Objective) -> list[tuple[float, float]]:
    """The range of rho over which a fit at that weight is resolved in double precision, per weight.

    With D the data curvature of the weight's block and s_min, s_max the extreme nonzero eigenvalues of S_i:
    below lambda s_min = sqrt(eps) ||D|| the weakest penalized direction of -H has a condition number past
    1/sqrt(eps), and above lambda s_max = ||D|| / sqrt(eps) the penalty swamps the data's curvature past
    1/sqrt(eps), so a fit there loses half of double precision. Beyond the upper end the lambda = infinity edge is
    evaluated exactly; below the lower end V only falls (with slope r_i / 2 in rho), so the lower end is the range's
    end, and there is no lambda = 0 edge.
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


def _restricted_prior(prior: ScaleMixturePrior, infinite: frozenset[int]) -> tuple[ScaleMixturePrior, F64Array]:
    """The prior with the weights in ``infinite`` at lambda = infinity, and the basis K of the allowed x.

    At lambda_i = infinity block i's penalized directions are exactly zero, so x = K z with K an orthonormal
    basis of every such block's null space. The other blocks act on z
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
        if position in infinite:
            continue
        embedded = np.zeros((block.factor.shape[0], prior.coefficient_size))
        embedded[:, block.coordinates] = block.factor
        blocks.append(SmoothingBlock(block.name, np.arange(allowed.shape[1]), embedded @ allowed))
    total = sum((block.matrix for block in blocks), np.zeros((allowed.shape[1], allowed.shape[1])))
    null_eigenvalues, null_vectors = np.linalg.eigh(total)
    scale = max(float(null_eigenvalues[-1]) if null_eigenvalues.size else 0.0, np.finfo(np.float64).tiny)
    null_basis = null_vectors[:, null_eigenvalues <= _EPSILON * allowed.shape[1] * scale]
    view = replace(prior, coefficient_map=prior.coefficient_map @ allowed, smoothing_blocks=tuple(blocks), null_basis=null_basis, anchor=None)
    if prior.anchor is not None:
        # The same local model, in the view's coordinates (x_k itself, which may lie outside the view).
        view = _anchored(view, prior.anchor.correction, prior.anchor.center)
    return view, allowed


def _ascend_evidence(
    prior: ScaleMixturePrior,
    start_weights: F64Array,
    start: _Evidence,
    cavity: Cavity, correction: CurvatureCorrection,
    working_bytes: int,
    lower: F64Array,
    upper: F64Array,
    tolerance: float,
    flat_start: F64Array,
) -> tuple[F64Array, _Evidence]:
    """Trust-region quasi-Newton ascent of V(rho) inside [lower, upper], from a certified evidence ``start``.

    The model is the Laplace V's gradient with a BFGS approximation of -V's Hessian; each trial maximizes it inside a
    radius, projected onto the bounds, and the radius follows the ratio of actual to predicted gain
    (Nocedal and Wright, Algorithm 4.1). A trial's x starts from the first-order predictor
    x_rho + (dx/drho) delta; a trial whose inner answer is not a certified maximum counts as V = -infinity.
    It stops when the model's predicted gain falls to ``tolerance`` nats, or the radius to half of double
    precision.
    """
    weights = np.clip(start_weights, lower, upper)
    current = start
    # The Laplace gradient proposes the steps; every acceptance is on the certified V (lead ruling A), and the final
    # stationarity check takes V's own gradient, corrections included (``_stationarity``).
    gradient = current.gradient
    hessian = np.eye(weights.shape[0])
    radius = float(np.linalg.norm(gradient))
    while True:
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
        trial = _certified_evidence(prior, trial_weights, current.coefficients + current.responses @ step, cavity, correction, working_bytes, tolerance, flat_start)
        actual = -np.inf if trial is None else trial.value - current.value
        ratio = actual / predicted if predicted > 0.0 else -np.inf
        step_norm = float(np.linalg.norm(step))
        if not np.isfinite(ratio) or ratio < 0.25:
            radius = 0.25 * step_norm
        elif ratio > 0.75 and step_norm >= radius * (1.0 - _HALF_PRECISION):
            radius = 2.0 * radius
        if trial is None or actual <= 0.0:
            continue
        trial_gradient = trial.gradient
        gradient_change = gradient - trial_gradient
        curvature = float(step @ gradient_change)
        if curvature > 0.0:
            image = hessian @ step
            hessian = hessian - np.outer(image, image) / float(step @ image) + np.outer(gradient_change, gradient_change) / curvature
        weights, current, gradient = trial_weights, trial, trial_gradient


def _certified_evidence(
    prior: ScaleMixturePrior, log_smoothing: F64Array, start: F64Array, cavity: Cavity, correction: CurvatureCorrection, working_bytes: int, tolerance: float, flat_start: F64Array
) -> _Evidence | None:
    """V at the certified inner maximum from ``start``, or from the flat start when that is not a certified
    maximum (the two structural starts; lead ruling); None when neither is."""
    warm = _evidence(prior, log_smoothing, start, cavity, correction, working_bytes, tolerance)
    chosen = warm if warm is not None else _evidence(prior, log_smoothing, flat_start, cavity, correction, working_bytes, tolerance)
    return _corrected(prior, log_smoothing, chosen, cavity, correction, working_bytes, tolerance)


def _same_basin(first: _Evidence, second: _Evidence) -> bool:
    """Whether two certified inner maxima are one: each point lies within sqrt(2 d) of its maximum in the -H metric
    (d its inner decrement), so one maximum is within the sum of the radii of both points; their V must then agree
    to within their certified errors."""
    step = first.coefficients - second.coefficients
    radius = np.sqrt(2.0 * first.inner_decrement) + np.sqrt(2.0 * second.inner_decrement)
    distance = np.sqrt(max(float(step @ first.precision @ step), 0.0))
    return distance <= radius and abs(first.laplace_value - second.laplace_value) <= first.error + second.error


def _best_certified(
    prior: ScaleMixturePrior, log_smoothing: F64Array, starts: Sequence[F64Array], cavity: Cavity, correction: CurvatureCorrection, working_bytes: int, tolerance: float,
    screen: float | None = None,
) -> _Evidence | None:
    """The certified inner maximum with the highest corrected V over the given starts; distinct basins are compared
    by their certified V (``_corrected``), and a start that lands in an already-found basin adds nothing.

    Two starts found one basin when their points are within the inner maximizer's own radii of each other,
    sqrt(2 d_a) + sqrt(2 d_b) in the -H metric, and their V agree to within their certified errors; the first
    start's candidate stands for the basin, which is then corrected once.

    With ``screen``, the corrections are taken only where some basin's Laplace V is above it, and None is returned
    otherwise: the search's screen for a release trial (``_maximize_evidence``), never a certificate."""
    certified: list[_Evidence] = []
    for start in starts:
        candidate = _evidence(prior, log_smoothing, start, cavity, correction, working_bytes, tolerance)
        if candidate is None:
            continue
        for other in certified:
            if _same_basin(candidate, other):
                # The first start's candidate stands for the basin: the starts are ordered warm, flat, log-normal,
                # and the warm one is the closest to the state the outer loop moves from (on gene 1 [real] a
                # profile approaching its supremum along a ray, the log-normal start's candidate lay far along it,
                # and the joint trial to it had no certified state at its own fixed point).
                break
        else:
            certified.append(candidate)
    if screen is not None and not any(candidate.value > screen for candidate in certified):
        return None
    corrected = [
        evidence
        for evidence in (_corrected(prior, log_smoothing, candidate, cavity, correction, working_bytes, tolerance) for candidate in certified)
        if evidence is not None
    ]
    return max(corrected, key=lambda evidence: evidence.value) if corrected else None


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
    starts: Sequence[F64Array],
    cavity: Cavity, correction: CurvatureCorrection,
    working_bytes: int,
    tolerance: float,
    screen: float | None = None,
) -> tuple[F64Array, _Evidence] | None:
    """The best certified evidence of the model with ``infinite`` at lambda = infinity, the other weights at
    ``weights``; with its restriction basis. ``screen`` as in ``_best_certified``."""
    view, allowed = _restricted_prior(prior, infinite)
    finite = [position for position in range(weights.shape[0]) if position not in infinite]
    evidence = _best_certified(view, weights[finite], [allowed.T @ start for start in starts], cavity, correction, working_bytes, tolerance, screen)
    return None if evidence is None else (allowed, evidence)


def _maximize_evidence(
    prior: ScaleMixturePrior,
    start_weights: F64Array,
    start_coefficients: F64Array,
    cavity: Cavity, correction: CurvatureCorrection,
    working_bytes: int,
    bounds: list[tuple[float, float]],
    tolerance: float,
    held_edges: frozenset[int] = frozenset(),
) -> tuple[F64Array, F64Array, _Evidence, _Evidence]:
    """Maximize V over every weight in (0, infinity]: the interior by the trust-region ascent inside the resolvable
    range, and the lambda = infinity edge evaluated exactly (x confined to the block's null space), never by fitting
    at an extreme weight. Once the interior ascent converges, every finite weight is compared with its infinity edge
    (lead ruling), and the best edge that raises V past the tolerance is taken; an edge weight is released to the
    centre of its resolvable range when the certified V is higher there. There is no lambda = 0 edge (see the module
    docstring), so a start weight of -inf means its range's lower end, and one past its upper end the edge. Every V is the best certified maximum over the warm, flat and
    global log-normal starts. Returns the log weights (+inf at an edge), x in full coordinates, V there, and V at
    the start.
    """
    lower = np.array([bound[0] for bound in bounds])
    upper = np.array([bound[1] for bound in bounds])
    # A start weight past its range's resolvable upper end (a fit on another lattice, whose range differs) is the
    # lambda = infinity edge itself: V is not resolved there, and the edge is its limit.
    infinite = frozenset(int(position) for position in np.flatnonzero((start_weights == np.inf) | (start_weights > upper)))
    weights = np.where(np.isin(np.arange(start_weights.shape[0]), list(infinite)), upper, np.clip(start_weights, lower, upper))
    flat = initial_hyperparameters(prior).coefficients
    log_normal = _log_normal_start(prior, start_coefficients, cavity, working_bytes)
    coefficients = np.array(start_coefficients, dtype=np.float64, copy=True)
    first = _edge_evidence(prior, weights, infinite, [coefficients, flat, log_normal], cavity, correction, working_bytes, tolerance)
    if first is None:
        # The warm weights (a fit on another lattice, or at a fold of their basin) can have no certified maximum:
        # the search then starts from the canonical start's weights, every edge released.
        infinite = frozenset()
        weights = 0.5 * (lower + upper)
        first = _edge_evidence(prior, weights, infinite, [coefficients, flat, log_normal], cavity, correction, working_bytes, tolerance)
    if first is None:
        raise FloatingPointError("no structural start reaches a certified maximum at the starting penalty weights")
    start = first[1]
    best_corrected = -np.inf
    # Blocks released from their edge below, once each per search: a block the ascent then returns to its edge is
    # not tried again; ``held_edges`` (``hyper_step``) are never released.
    released_once: set[int] = set(held_edges)
    while True:
        edges = infinite
        finite = np.array([position for position in range(len(bounds)) if position not in edges], dtype=np.int64)
        view, allowed = _restricted_prior(prior, infinite)
        entry = _edge_evidence(prior, weights, infinite, [coefficients, flat, log_normal], cavity, correction, working_bytes, tolerance)
        if entry is None:
            raise FloatingPointError("no structural start reaches a certified maximum at the current penalty weights")
        finite_weights, evidence = _ascend_evidence(
            view, weights[finite], entry[1], cavity, correction, working_bytes, lower[finite], upper[finite], tolerance, allowed.T @ flat
        )
        refit = _best_certified(view, finite_weights, [evidence.coefficients, allowed.T @ flat, allowed.T @ log_normal], cavity, correction, working_bytes, tolerance)
        if refit is not None and float(np.max(np.abs(refit.coefficients - evidence.coefficients))) > _HALF_PRECISION * (
            1.0 + float(np.max(np.abs(evidence.coefficients)))
        ):
            # Another basin wins at these weights: its weights are not optimized yet, so ascend again from it, as long
            # as each switch raises the best corrected V seen (so switches cannot cycle).
            if refit.value > best_corrected + tolerance:
                best_corrected = refit.value
                weights[finite] = finite_weights
                coefficients = allowed @ refit.coefficients
                continue
        weights[finite] = finite_weights
        coefficients = allowed @ evidence.coefficients
        current_value = evidence.value
        moved = False
        # Every finite weight is compared with its infinity edge, not only one the gradient points to: the certified V
        # can prefer an edge the Laplace gradient does not see. The best edge that raises V past the tolerance is taken.
        best_edge = None
        for position in (int(index) for index in finite):
            trial_infinite = infinite | {position}
            trial = _edge_evidence(prior, weights, trial_infinite, [coefficients, flat, log_normal], cavity, correction, working_bytes, tolerance)
            if trial is not None and trial[1].value > current_value + tolerance and (best_edge is None or trial[1].value > best_edge[1][1].value):
                best_edge = (trial_infinite, trial)
        if best_edge is not None:
            infinite, moved = frozenset(best_edge[0]), True
            coefficients = best_edge[1][0] @ best_edge[1][1].coefficients
        if not moved:
            # An edge weight is released where the certified V at the centre of its resolvable range is above the
            # edge's by more than the tolerance. The centre is the weight at which the penalty's geometric-mean
            # eigenvalue matches the data's curvature (the log-midpoint of the two half-precision ends): the one
            # scale-free point where the two are commensurate. The range's upper end cannot release anything: the
            # penalty swamps the data there past half precision, so V is the edge's to the tolerance by
            # construction and its slope is O(e^-rho) (on the mean-field test problem V's interior maximum sits
            # mid-range, 2.7 nats above the edge on one class, with a slope of -0.0004 +- 0.06 at the upper end).
            # The Laplace form screens the centre (``_best_certified``): the Tierney-Kadane terms lowered V on every
            # problem measured (gene 1's flat interior by 1-66 nats, the test problem's optimum by 0.4), so the
            # corrections are taken only where the Laplace V is itself above the edge. The ascent from the centre
            # then searches the interior.
            for position in sorted(edges):
                if position in released_once:
                    continue
                trial_infinite = infinite - {position}
                trial_weights = weights.copy()
                trial_weights[position] = 0.5 * (lower[position] + upper[position])
                trial = _edge_evidence(
                    prior, trial_weights, trial_infinite, [coefficients, flat, log_normal], cavity, correction, working_bytes, tolerance,
                    screen=current_value + tolerance,
                )
                if trial is not None and trial[1].value > current_value + tolerance:
                    released_once.add(position)
                    infinite, weights, moved = frozenset(trial_infinite), trial_weights, True
                    coefficients = trial[0] @ trial[1].coefficients
                    break
        if not moved:
            log_smoothing = weights.copy()
            log_smoothing[sorted(infinite)] = np.inf
            return log_smoothing, coefficients, evidence, start


@dataclass(frozen=True)
class _Stationarity:
    """The B-evidence's stationarity in the weights at a certified maximum x_rho (``_stationarity``).

    ``gradient`` is V's rho-gradient (the Laplace part's, exact, and the corrections' own slopes,
    ``_correction_slopes``) with its error ``error``; ``curvature`` the matrix -d2V/drho2 (forward differences of the
    Laplace gradient plus the corrections' second differences on the diagonal), of steps
    ``steps``; ``folds`` is, per weight, the distance within which its basin ends on the side its gradient climbs to
    (zero where it does not end); ``gain`` the remaining gain; ``better`` a side whose certified V is above the base's
    by more than the tolerance, both at their certified bounds (weights and evidence), when one was found.
    """

    gradient: F64Array
    error: F64Array
    curvature: F64Array
    steps: F64Array
    folds: F64Array
    gain: float
    better: tuple[F64Array, _Evidence] | None
    # The gain's parts (theory-ep's split): ``floor`` with exact correction slopes (E at its fixed part), ``decrement``
    # 1/2 g'K^-1 g on the open weights, ``fold`` the folded weights' (|g| + E) h; the rest of ``gain`` is E's.
    floor: float = np.inf
    decrement: float = np.inf
    fold: float = np.inf


def _full_gradient(
    view: ScaleMixturePrior, weights: F64Array, evidence: _Evidence, interior: F64Array, cavity: Cavity, working_bytes: int, targets: F64Array
) -> tuple[F64Array, F64Array, F64Array]:
    """The certified V's rho-gradient, its error, and the corrections' second differences: the Laplace part's
    gradient (exact at x_rho) plus the corrections' own slopes to ``targets`` (``_correction_slopes``)."""
    correction_slopes, correction_errors, correction_second = _correction_slopes(view, weights, evidence, interior, cavity, working_bytes, targets)
    return evidence.gradient + correction_slopes, correction_errors + _laplace_gradient_error(evidence), correction_second


def _laplace_gradient_error(evidence: _Evidence) -> F64Array:
    """The Laplace gradient's error per weight, which no slope target lowers: x_rho's own error moves it by its x-slope
    over x's error, sqrt(2 d) in the -H metric, bounded by the gradient's scale s_i times that radius (the same
    logistic bound as V's derivatives), plus the rounding of the objective."""
    radius = float(np.sqrt(2.0 * max(evidence.inner_decrement, 0.0)))
    return 0.5 * (evidence.effective_degrees + evidence.penalty_sizes) * radius + _EPSILON * evidence.magnitude


def _stationarity(
    view: ScaleMixturePrior,
    weights: F64Array,
    evidence: _Evidence,
    interior: F64Array,
    cavity: Cavity,
    correction: CurvatureCorrection,
    working_bytes: int,
    tolerance: float,
    budget: float = np.inf,
) -> _Stationarity:
    """The weights' Newton decrement at a certified maximum, from V's analytic gradient (lead ruling: in place of
    central differences of V, which cannot certify where the base's inner basin ends within their step).

    ``budget`` is the gain the certificate leaves the weights (``HyperStep.tighten``): where the bound exceeds it only
    through the correction slopes' error, the slopes are resolved again to the error that meets it (theory-ep), for
    as long as every slope's error falls. Where it exceeds it even with exact slopes (``floor``), the slopes are not
    what stops the certificate. Infinite, the bound is reported as measured.

    The gradient is the Laplace part's (exact) plus the corrections' own slopes (``_correction_slopes``). The curvature
    is -dg/drho of the Laplace part by one forward difference per interior weight, taken on the side the gradient
    climbs (where a fold would be), plus the corrections' second differences on the diagonal (their mixed second
    differences are left out of the local model). With the Laplace gradient's error E and V's third derivative in
    rho_i bounded by s_i = (edf_i + lambda_i ||R_i x||^2) / 2 (the logistic bound of MODEL.md S4), a difference of
    step h errs by h s / 2 + 2 E / h, least at h = 2 sqrt(E / s). Where that side has no certified maximum within h, the basin ends
    there: the difference is taken on the other side, and the gain along that weight is at most (|g| + E) h, the
    most V can climb before the fold. Elsewhere the gain is the most the Newton decrement 1/2 (g + d)'K^-1 (g + d)
    reaches over the gradient's error box |d| <= E on the other interior weights, 1/2 g'K^-1 g + |K^-1 g|'E +
    1/2 E'|K^-1|E (theory-ep), with K the symmetrized difference matrix; an indefinite K has no decrement (infinite gain),
    as an indefinite B + S has none for x. A side whose certified V beats the base's by the tolerance, both at their
    certified bounds, is returned as ``better`` at once: the base is then not the maximum, and the search resumes
    from it.

    This is a Newton decrement on the local model (K is a difference estimate), with an indefinite model refused: the
    same standard as the x-certificate, not a global curvature bound over the step.
    """
    count = weights.shape[0]
    # The gradient is taken at x_rho resolved to double precision, so x's own error barely enters it.
    refined = _corrected(view, weights, _evidence(view, weights, evidence.coefficients, cavity, correction, working_bytes, 0.0), cavity, correction, working_bytes, tolerance)
    base = evidence if refined is None else refined
    # The corrections' slopes first to the error the curvature bound s allows (cheap), for the gradient's sign; once
    # the difference curvature K is in, again where K asks for less: E^2 / (2 K) = tolerance / (4 n).
    coarse = _coarse_targets(base, interior, tolerance)
    gradient, error, correction_second = _full_gradient(view, weights, base, interior, cavity, working_bytes, coarse)
    scale = np.maximum(0.5 * (base.effective_degrees + base.penalty_sizes), _EPSILON * base.magnitude)
    laplace_error = scale * float(np.sqrt(2.0 * max(base.inner_decrement, 0.0))) + _EPSILON * base.magnitude
    limit = _HALF_PRECISION * (1.0 + float(np.max(np.abs(weights), initial=0.0)))
    curvature = np.zeros((count, count))
    steps = np.zeros(count)
    folds = np.zeros(count)
    for position in np.flatnonzero(interior):
        unit = np.zeros(count)
        unit[position] = 1.0
        climb = 1.0 if gradient[position] >= 0.0 else -1.0
        step = max(2.0 * float(np.sqrt(laplace_error[position] / scale[position])), limit)
        column = None
        while column is None:
            for side in (climb, -climb):
                trial_weights = weights + side * step * unit
                trial = _corrected(
                    view, trial_weights,
                    _evidence(view, trial_weights, base.coefficients + side * step * base.responses[:, position], cavity, correction, working_bytes, 0.0),
                    cavity, correction, working_bytes, tolerance,
                )
                if trial is not None and trial.value - trial.error > evidence.value + evidence.error + tolerance:
                    return _Stationarity(gradient, error, curvature, steps, folds, np.inf, (trial_weights, trial))
                if trial is None:
                    if side == climb:
                        folds[position] = step
                    continue
                column = -(trial.gradient - base.gradient) / (side * step)
                break
            if column is None:
                if step <= limit:
                    raise FloatingPointError("the B-evidence has no certified maximum in one basin on either side of a fitted penalty weight")
                folds[position] = 0.0
                step = max(0.5 * step, limit)
        curvature[:, position] = column
        steps[position] = step
    inside = np.flatnonzero(interior)
    interior_count = max(inside.shape[0], 1)
    laplace_curvature = np.abs(np.diag(curvature))
    fine = np.sqrt(0.5 * tolerance * np.maximum(laplace_curvature, _EPSILON * scale) / interior_count)
    tighter = interior & (fine < coarse)
    if np.any(tighter):
        refined_gradient, refined_error, refined_second = _full_gradient(view, weights, base, tighter, cavity, working_bytes, np.minimum(coarse, fine))
        gradient = np.where(tighter, refined_gradient, gradient)
        error = np.where(tighter, refined_error, error)
        correction_second = np.where(tighter, refined_second, correction_second)
    for position in inside:
        curvature[position, position] -= correction_second[position]
    curvature[np.ix_(inside, inside)] = 0.5 * (curvature[np.ix_(inside, inside)] + curvature[np.ix_(inside, inside)].T)
    folded = interior & (folds > 0.0)
    open_ = np.flatnonzero(interior & ~folded)

    def terms(gradient: F64Array, fixed: F64Array, slopes: F64Array, curvature: F64Array) -> tuple[float, float, float, float, float]:
        """(alpha, beta, gamma, decrement, fold): the gain bound alpha + beta t + gamma t^2 for the gradient's error
        E = F + t E_c (theory-ep), 1/2 g'K^-1 g, and the folded weights' part at t = 1. On the open weights the bound
        is the most 1/2 (g + d)'K^-1 (g + d) reaches over the box |d| <= E, with the signed g:
        1/2 g'K^-1 g + |K^-1 g|'E + 1/2 E'|K^-1|E (|.| elementwise; K^-1's off-diagonals can be negative, where
        1/2 (|g| + E)'K^-1 (|g| + E) would understate it). On the folded weights it is (|g| + E) h. Infinite where a
        slope is unresolved or K is not positive definite (an indefinite model has no decrement, as an indefinite
        B + S has none for x)."""
        if not all(np.all(np.isfinite(values[interior])) for values in (gradient, fixed, slopes)):
            return np.inf, 0.0, 0.0, np.inf, np.inf
        alpha = float(np.sum((np.abs(gradient) + fixed)[folded] * folds[folded]))
        beta = float(np.sum(slopes[folded] * folds[folded]))
        fold = alpha + beta
        gamma = decrement = 0.0
        if open_.shape[0]:
            try:
                factor = np.linalg.cholesky(curvature[np.ix_(open_, open_)])
            except np.linalg.LinAlgError:
                return np.inf, 0.0, 0.0, np.inf, fold
            root = solve_triangular(factor, np.eye(open_.shape[0]), lower=True)
            inverse = root.T @ root
            absolute = np.abs(inverse)
            reach = np.abs(inverse @ gradient[open_])
            floor, error = fixed[open_], slopes[open_]
            decrement = 0.5 * float(gradient[open_] @ inverse @ gradient[open_])
            alpha += decrement + float(reach @ floor) + 0.5 * float(floor @ absolute @ floor)
            beta += float(reach @ error) + float(floor @ absolute @ error)
            gamma = 0.5 * float(error @ absolute @ error)
        return alpha, beta, gamma, decrement, fold

    # E's part no slope target lowers (x_rho's own error and the rounding), and the slopes' part above it.
    fixed = np.where(interior, _laplace_gradient_error(base), 0.0)
    alpha, beta, gamma, decrement, fold = terms(gradient, fixed, np.maximum(error - fixed, 0.0), curvature)
    gain = alpha + beta + gamma
    while np.isfinite(gain) and alpha <= budget < gain:
        # gain(t) rises in the slopes' error scale t (alpha, beta, gamma >= 0), from at most the budget at t = 0 to
        # above it at t = 1: its one root in [0, 1) is the error that meets the budget.
        room = budget - alpha
        scale_down = 2.0 * room / (beta + float(np.sqrt(beta * beta + 4.0 * gamma * room)))
        slopes = np.maximum(error - fixed, 0.0)
        tighter = interior & (slopes > 0.0)
        resolved = _full_gradient(view, weights, base, tighter, cavity, working_bytes, scale_down * slopes)
        new_gradient, new_error, new_second = (np.where(tighter, new, old) for new, old in zip(resolved, (gradient, error, correction_second)))
        if not np.all(new_error[tighter] < error[tighter]):
            # A slope resolves no finer (its rounding floor, or its differences no longer agree): the bound stands.
            break
        for position in np.flatnonzero(tighter):
            curvature[position, position] += correction_second[position] - new_second[position]
        gradient, error, correction_second = new_gradient, new_error, new_second
        alpha, beta, gamma, decrement, fold = terms(gradient, fixed, np.maximum(error - fixed, 0.0), curvature)
        gain = alpha + beta + gamma
    return _Stationarity(gradient, error, curvature, steps, folds, gain, None, floor=alpha, decrement=decrement, fold=fold)


@_step_scoped
def hyper_step(
    prior: ScaleMixturePrior, hyperparameters: MixtureHyperparameters, cavity: Cavity, correction: CurvatureCorrection, working_bytes: int, tolerance: float,
    held_edges: frozenset[int] = frozenset(),
) -> HyperStep:
    """Maximize the B-evidence over every penalty weight in [0, infinity], with x at the penalized maximum for each, to
    ``tolerance`` nats: the resolution the fit certifies (1/(2K) for a scorer with K posterior draws).

    The search steers by V's own Laplace gradient and accepts on the certified V (with B). At the end, V's
    stationarity is checked from its analytic gradient, corrections included, and a forward difference of that
    gradient per interior weight (``_stationarity``): while the weights' Newton decrement exceeds ``tolerance``, the
    search takes the Newton step, accepting steps whose gain's certified lower bound exceeds ``tolerance`` (so every
    such move gains a resolved amount: V is bounded above), and, where none does (the band: a real gain within the
    tolerance), the full Newton step on a certified rise, kept only when the next check's decrement falls (natural
    monotonicity, so Newton's local convergence ends it); a weight whose basin ends on its climbing side
    within the difference's step contributes the most V can climb before the fold.

    The returned step's ``tighten`` re-checks its stationarity at the final weights with the bound tightened to a
    budget (``_stationarity``), with no new search.

    ``held_edges`` are blocks the search leaves at their lambda = infinity edge: those the outer loop released on
    this model's proposal and returned (``fit_hyperparameters``: x polished at the freed weights certified below
    the state it left), so the local model's account of that interior was measured and refuted at its own fixed
    point, which stands above what the model predicts here.

    The search is over the EP evidence's local model about the fixed point at ``hyperparameters.coefficients``, where
    ``correction`` was solved (``_Anchor``): x_rho maximizes F - 1/2 (x - x_k)'C(x - x_k) - P, whose curvature
    A + C + S is the one V's determinant takes, so V is the Laplace evidence of one integrand (theory-ep (a)).
    """
    prior = _anchored(prior, correction, prior.coefficient_map @ hyperparameters.coefficients)
    start_objective = _data_objective(prior, hyperparameters.coefficients, cavity, working_bytes)
    bounds = _smoothing_bounds(prior, start_objective)
    # There is no lambda = 0 edge: a -inf start weight means its range's lower end.
    lowest = np.array([bound[0] for bound in bounds])
    hyperparameters = replace(
        hyperparameters, log_smoothing=np.where(hyperparameters.log_smoothing == -np.inf, lowest, hyperparameters.log_smoothing)
    )
    infinite = frozenset(int(position) for position in np.flatnonzero(hyperparameters.log_smoothing == np.inf))
    view, allowed = _restricted_prior(prior, infinite)
    finite = np.isfinite(hyperparameters.log_smoothing)
    start_coefficients = allowed.T @ hyperparameters.coefficients
    _value, start_gradient, start_hessian = _penalized(
        view, _data_objective(view, start_coefficients, cavity, working_bytes), hyperparameters.log_smoothing[finite],
        _penalty_matrix(view, hyperparameters.log_smoothing[finite]), start_coefficients,
    )
    start_decrement = 0.5 * float(start_gradient @ _ascent_direction(-start_hessian, start_gradient))
    log_smoothing, coefficients, evidence, start_evidence = _maximize_evidence(
        prior, hyperparameters.log_smoothing, hyperparameters.coefficients, cavity, correction, working_bytes, bounds, tolerance, held_edges
    )
    final_infinite = frozenset(int(position) for position in np.flatnonzero(log_smoothing == np.inf))
    final_view, final_allowed = _restricted_prior(prior, final_infinite)
    finite_final = np.isfinite(log_smoothing)
    weights = log_smoothing[finite_final]
    lower = np.array([bound[0] for bound in bounds])[finite_final]
    upper = np.array([bound[1] for bound in bounds])[finite_final]
    evidence = replace(evidence, coefficients=final_allowed.T @ coefficients)
    # The weights, evidence and check before a band move (below), until the check after it has judged it.
    band: tuple[F64Array, _Evidence, _Stationarity] | None = None
    while True:
        interior = (weights > lower) & (weights < upper)
        check = _stationarity(final_view, weights, evidence, interior, cavity, correction, working_bytes, tolerance)
        if band is not None:
            if check.better is None and not check.gain < band[2].gain:
                # The band move did not lower the weights' Newton decrement (natural monotonicity): it is undone, and
                # the search ends at the weights before it.
                weights, evidence, check = band
                break
            band = None
        moved = check.better
        if moved is None and check.gain <= tolerance:
            break
        if moved is None:
            # Newton on the difference curvature over the weights whose basin does not end on their climbing side
            # (on its magnitude where it is indefinite), accepted only when the certified V rises by more than the tolerance.
            open_ = interior & ~(check.folds > 0.0)
            direction = np.zeros_like(weights)
            if np.any(open_):
                eigenvalues, eigenvectors = np.linalg.eigh(check.curvature[np.ix_(open_, open_)])
                magnitudes = np.maximum(np.abs(eigenvalues), _EPSILON * float(np.max(np.abs(eigenvalues))))
                direction[open_] = eigenvectors @ ((eigenvectors.T @ check.gradient[open_]) / magnitudes)
            step_length = 1.0
            while step_length * float(np.max(np.abs(direction), initial=0.0)) > _HALF_PRECISION * (1.0 + float(np.max(np.abs(weights)))):
                trial_weights = np.clip(weights + step_length * direction, lower, upper)
                # The first-order predictor can carry x into a worse inner basin than the base's own: the trial is the
                # best certified maximum over the predictor, the base's x and the flat start.
                trial = _best_certified(
                    final_view, trial_weights,
                    [evidence.coefficients + evidence.responses @ (trial_weights - weights), evidence.coefficients,
                     final_allowed.T @ initial_hyperparameters(prior).coefficients],
                    cavity, correction, working_bytes, tolerance,
                )
                # The gain's lower bound (both values at their certified bounds, as ``_stationarity``'s better side) must
                # exceed the tolerance: a rise within the certificate's resolution is not resolved, and only a resolved
                # gain per move bounds the number of moves by V's range, so the search ends.
                if trial is not None and trial.value - trial.error > evidence.value + evidence.error + tolerance:
                    moved = (trial_weights, trial)
                    break
                # The band (theory-ep): where the remaining gain is real but within the tolerance, no move resolves it.
                # The full Newton step is then taken on a certified rise, and kept only when the next check's decrement
                # falls (natural monotonicity): Newton's local convergence ends it, not the tolerance's count.
                if step_length == 1.0 and trial is not None and trial.value - trial.error > evidence.value + evidence.error:
                    moved = (trial_weights, trial)
                    band = (weights, evidence, check)
                    break
                step_length *= 0.5
        if moved is None:
            break
        weights, evidence = moved
        if check.better is None:
            continue
        # A certifiably better side is another basin: the search resumes from it, edges and structural starts included.
        # It keeps the move when the resumed search ends lower (a basin chosen by its corrected V), so V rises at every
        # pass and the loop ends.
        resumed_smoothing = log_smoothing.copy()
        resumed_smoothing[finite_final] = weights
        resumed = _maximize_evidence(prior, resumed_smoothing, final_allowed @ evidence.coefficients, cavity, correction, working_bytes, bounds, tolerance, held_edges)
        if resumed[2].value > evidence.value:
            log_smoothing, coefficients, evidence = resumed[0], resumed[1], resumed[2]
            final_infinite = frozenset(int(position) for position in np.flatnonzero(log_smoothing == np.inf))
            final_view, final_allowed = _restricted_prior(prior, final_infinite)
            finite_final = np.isfinite(log_smoothing)
            weights = log_smoothing[finite_final]
            lower = np.array([bound[0] for bound in bounds])[finite_final]
            upper = np.array([bound[1] for bound in bounds])[finite_final]
            evidence = replace(evidence, coefficients=final_allowed.T @ coefficients)
    log_smoothing = log_smoothing.copy()
    log_smoothing[finite_final] = weights

    def checked(check: _Stationarity, evidence: _Evidence = evidence) -> HyperStep:
        """The step at ``evidence`` (V at the returned weights, as resolved so far) with the stationarity ``check``;
        its ``tighten`` and ``resolve`` compose, each keeping what the other resolved."""

        def tighten(budget: float) -> HyperStep | None:
            """This step with its stationarity bound tightened to ``budget`` at its final weights; None where the
            check then finds a better side or no finite bound (the search, not the bound, is what is left there)."""
            tightened = _stationarity(final_view, weights, evidence, interior, cavity, correction, working_bytes, tolerance, budget)
            return checked(tightened, evidence) if tightened.better is None and np.isfinite(tightened.gain) else None

        def resolve(budget: float) -> HyperStep | None:
            """This step with V at its returned weights certified to ``budget``: x re-maximized to the inner tolerance
            that budget asks (the determinant's first-order move under x's remaining decrement), then more of the
            Tierney-Kadane directions integrated exactly; the stationarity check as it stands. None where V has no
            certified value there."""
            refined = _evidence(final_view, weights, evidence.coefficients, cavity, correction, working_bytes, budget)
            resolved = _corrected(final_view, weights, refined, cavity, correction, working_bytes, budget)
            return None if resolved is None else checked(check, resolved)

        return HyperStep(
            hyperparameters=MixtureHyperparameters(coefficients=final_allowed @ evidence.coefficients, log_smoothing=log_smoothing),
            penalized_objective=evidence.penalized_value,
            evidence=evidence.value,
            newton_decrement=evidence.newton_decrement,
            smoothing_gradient=float(np.max(np.abs(check.gradient[interior]), initial=0.0)),
            start_decrement=start_decrement,
            evidence_gain=evidence.value - start_evidence.value,
            stationarity_steps=check.steps,
            stationarity_errors=check.error,
            stationarity_gain=check.gain,
            stationarity_floor=check.floor,
            stationarity_decrement=check.decrement,
            stationarity_fold=check.fold,
            tighten=tighten,
            resolve=resolve,
            evidence_error=evidence.error,
        )

    return checked(check)


# ------------------------------------------------------------------ the outer loop


@dataclass(frozen=True)
class FixedPoint:
    """A model's certified EP fixed point at a prior's hyperparameters: each variant's cavity, q's linear responses
    there (valid until the next fixed point is solved), q's mean, ``precision_norm(d)`` = d' Sigma^-1 d in q's
    posterior metric, and the effective number of effects p_eff = p - sum_j tau_j Sigma_jj.

    Where one fixed point holds several independently scored models sharing x (the pooled arm's genes),
    ``precision_norm`` returns one move per model and ``effective_effects`` their p_eff: the prediction check then
    holds each model to its own budget, so no model uses another's (fit-api P1)."""

    cavity: Cavity
    posterior: GaussianPosterior
    mean: F64Array
    precision_norm: Callable[[F64Array], float | F64Array]
    effective_effects: float | F64Array


FixedPoints = Callable[[Sequence[MixtureHyperparameters]], Sequence["FixedPoint | None"]]
"""Each model's certified EP fixed point at its hyperparameters, warm from the previous call; None for a model where
none exists (EP reaches no proper cavities there)."""


@dataclass(frozen=True)
class OuterFit:
    """One model's certified empirical Bayes.

    ``remaining_gain`` is what the last check still found, in nats: the gain V's local model predicts from the last
    state to the weights' maximum, the weights' remaining stationarity gain beyond it, and that prediction's measured
    error (``fit_hyperparameters``); it is at most the tolerance. ``newton_decrement`` is x's own decrement at the
    last state, 1/2 g'(B + S)^-1 g. ``prediction_move`` is the posterior-mean move of the certifying step in q's
    posterior metric, against ``prediction_tolerance`` = 1 / K = 2 tolerance, i.e. KL(q || q') = move / 2 <= 1 / (2K)
    nats (K = 1 / (2 tolerance) draws; in evidence units, as every other certificate, so it stays defined where p_eff
    collapses). ``iterations`` counts accepted steps, ``halvings`` the trials refused (by the test, or for having no
    EP fixed point), and ``unresolved`` those of them that had no EP fixed point at all, so a loop that keeps refusing
    near its answer is visible in the certificate.

    ``fixed_point_term_measured`` is False while the decisions charge the fixed points' own error along the steps as
    zero (theory-ep: the oracles' perturbation probes are not wired yet); callers treat such a fit as uncertified
    even where ``certified`` holds.
    """

    hyperparameters: MixtureHyperparameters
    step: HyperStep
    newton_decrement: float
    remaining_gain: float
    prediction_move: float
    prediction_tolerance: float
    iterations: int
    halvings: int
    unresolved: int
    # The remaining gain at every outer evaluation, in order: the outer rate.
    history: tuple[float, ...]
    # False when the loop stopped at double precision without certifying (``remaining_gain`` over the tolerance, or the
    # prediction move unchecked): an honest fit, which callers count and report as uncertified, never as certified.
    certified: bool = True
    fixed_point_term_measured: bool = False


@dataclass(frozen=True)
class _NewtonB:
    """The EP evidence's quadratic model at x with fixed weights, in the weights' restricted coordinates: the
    fixed-cavity gradient g (the EP evidence's own at a fixed point) and B + S's spectrum."""

    view: ScaleMixturePrior
    allowed: F64Array
    log_smoothing: F64Array
    origin: F64Array
    gradient: F64Array
    total: F64Array
    eigenvectors: F64Array
    eigenvalues: F64Array
    definite: bool
    decrement: float


def _penalized_gradient(prior: ScaleMixturePrior, weights: F64Array, coefficients: F64Array, cavity: Cavity, working_bytes: int) -> F64Array:
    """The fixed-cavity gradient of the penalized objective in x: at an EP fixed point, the EP evidence's."""
    objective = _data_objective(prior, coefficients, cavity, working_bytes, hessian_too=False)
    return _penalized(prior, objective, weights, _penalty_matrix(prior, weights), coefficients)[1]


def _newton_b(
    prior: ScaleMixturePrior, log_smoothing: F64Array, coefficients: F64Array, point: FixedPoint, correction: CurvatureCorrection, working_bytes: int
) -> _NewtonB:
    """The quadratic model of the EP evidence at x with the weights ``log_smoothing`` (edges included): g, and the
    total curvature B + S, never the fixed-cavity one. ``decrement`` is 1/2 g'(B + S)^-1 g where B + S is positive
    definite (a strict maximum's certificate), and infinite elsewhere (a point that is not a maximum certifies
    nothing, however small g is)."""
    infinite = frozenset(int(position) for position in np.flatnonzero(log_smoothing == np.inf))
    view, allowed = _restricted_prior(prior, infinite)
    weights = log_smoothing[np.isfinite(log_smoothing)]
    origin = allowed.T @ coefficients
    objective = _data_objective(view, origin, point.cavity, working_bytes)
    _value, gradient, hessian = _penalized(view, objective, weights, _penalty_matrix(view, weights), origin)
    # At the fixed point the correction makes A + S exactly B + S.
    total = -hessian + correction.on(view.coefficient_map)
    total = 0.5 * (total + total.T)
    eigenvalues, eigenvectors = np.linalg.eigh(total)
    definite = bool(eigenvalues[0] > 0.0)
    components = eigenvectors.T @ gradient
    return _NewtonB(
        view=view, allowed=allowed, log_smoothing=log_smoothing, origin=origin, gradient=gradient, total=total, eigenvectors=eigenvectors,
        eigenvalues=eigenvalues, definite=definite, decrement=0.5 * float(np.sum(components * components / eigenvalues)) if definite else np.inf,
    )


def _trial(newton: _NewtonB, step: F64Array) -> MixtureHyperparameters:
    return MixtureHyperparameters(coefficients=newton.allowed @ (newton.origin + step), log_smoothing=newton.log_smoothing)


def _proposal(newton: _NewtonB, radius: float) -> F64Array:
    """The step: the maximizer of the quadratic model inside ``radius`` (More and Sorensen): Newton's (B + S)^-1 g
    where B + S is positive definite and that step fits, the boundary maximizer otherwise, which follows B + S's
    negative curvature out of a saddle. The radius binds the definite case too: Newton's step divides each
    direction's gradient by its own curvature, and on a direction the data barely curve (the log-normal family at
    the lambda = infinity edge, curvature 3e-5 against a gradient of 0.02 on the mean-field test problem) it is
    hundreds of units long, which the acceptance test cannot refuse: it lands where the density has collapsed onto
    one lattice node, every derivative is at rounding, and the vanished gradient reads as a maximum."""
    return _trust_region_step(newton.total, newton.gradient, radius, spectrum=(newton.eigenvalues, newton.eigenvectors))


def _cauchy_radius(newton: _NewtonB) -> float:
    """The first trust radius: the Cauchy step's length on |B + S|, ||g||^3 / g'|B + S| g, the model's steepest-ascent
    maximizer. Newton's step on |B + S| divides each direction's gradient by its own curvature, which the rounding
    floor on a direction the data barely curve makes astronomically long (review-mathbugs: |x| 1.3e4 on the first
    trial of a real gene, which put a class's density below the lattice); the Cauchy step weighs the gradient by
    all the curvature it sees."""
    components = newton.eigenvectors.T @ newton.gradient
    curvature = float(np.sum(np.abs(newton.eigenvalues) * np.square(components)))
    norm = float(np.linalg.norm(newton.gradient))
    return norm**3 / curvature if curvature > 0.0 else 0.0


@dataclass(frozen=True)
class _State:
    """V's formula at an outer state (x_j, rho_j) and its fixed point, with x_j itself, not re-maximized (theory-ep
    (b)): W_j = E(x_j) - P(x_j) + 1/2 log|S|_+ - 1/2 log|B_j + S|_N + T_j, which V(rho_j) is within ``error`` of
    (x_j's own decrement and its first-order move of the determinant and the corrections, and their rounding:
    ``_evidence``'s and ``_corrected``'s error at x_j). E's value is not needed: ``value`` is W_j with the
    fixed-cavity F_j(x_j) in E's place (the units ``hyper_step``'s V takes at this fixed point), ``rest`` is
    W_j - F_j(x_j), and E's change between two states is the path integral of ``gradient`` (the EP evidence's
    x-gradient, the fixed-cavity one at a fixed point), corrected by B's change along the step (``fixed_curvature``
    A_j and ``correction`` C_j, in full x). ``decrement`` is x's own Newton decrement delta_j; ``polished`` marks a
    state whose x inner steps have taken to its maximum at rho_j to the certificate's resolution; ``tail`` is the
    remaining gain the inner steps' own geometric contraction bounds where they end (``fit_hyperparameters``:
    a profile that approaches its supremum along a ray, E_inf - c e^{-t/tau}, gives Newton steps of one length
    with gains falling by a fixed ratio r, and its remaining gain is the last gain times r / (1 - r), twice the
    Newton decrement there; zero where the decrement is the bound)."""

    value: float
    rest: float
    gradient: F64Array
    fixed_curvature: F64Array
    correction: CurvatureCorrection
    error: float
    decrement: float
    polished: bool = False
    tail: float = 0.0


def _outer_state(
    prior: ScaleMixturePrior, hyperparameters: MixtureHyperparameters, point: FixedPoint, correction: CurvatureCorrection, working_bytes: int, tolerance: float
) -> _State | None:
    """``_State`` at ``hyperparameters`` = (x_j, rho_j) and its fixed point, where ``correction`` was solved; None where
    B_j + S has no certified determinant at x_j (x_j is not inside a basin of V's integrand)."""
    data = _data_objective(prior, hyperparameters.coefficients, point.cavity, working_bytes)
    # As in ``hyper_step``: a -inf weight means its range's lower end.
    lowest = np.array([bound[0] for bound in _smoothing_bounds(prior, data)])
    log_smoothing = np.where(hyperparameters.log_smoothing == -np.inf, lowest, hyperparameters.log_smoothing)
    infinite = frozenset(int(position) for position in np.flatnonzero(log_smoothing == np.inf))
    # Restricted first, anchored second: the anchor asks C on the view's own directions (at an all-edge state two per
    # class, not the lattice's K per class), and the correction's lazy solve keeps them for the hyper step's full
    # anchor. The order is exact: z_k is the same in every view, and ``_restricted_prior`` re-anchors a restricted
    # view with the same correction and center.
    view, allowed = _restricted_prior(prior, infinite)
    view = _anchored(view, correction, prior.coefficient_map @ hyperparameters.coefficients)
    weights = log_smoothing[np.isfinite(log_smoothing)]
    evidence = _evidence(view, weights, allowed.T @ hyperparameters.coefficients, point.cavity, correction, working_bytes, tolerance, maximize=False)
    corrected = None if evidence is None else _corrected(view, weights, evidence, point.cavity, correction, working_bytes, tolerance)
    if corrected is None:
        return None
    mapping = prior.coefficient_map
    fixed = -(mapping.T @ data.hessian @ mapping)
    return _State(
        value=corrected.value, rest=corrected.value - data.value, gradient=mapping.T @ data.gradient, fixed_curvature=0.5 * (fixed + fixed.T),
        correction=correction, error=corrected.error, decrement=evidence.inner_decrement,
    )


def _path_gain(prior: ScaleMixturePrior, start: _State, end: _State, move: F64Array) -> tuple[float, float]:
    """(G, eps): V's realized gain from ``start`` to ``end`` over the step ``move`` = x_end - x_start, and its
    resolution (theory-ep (b)). E's change is the trapezoid rule of its path integral with its Euler-Maclaurin end
    correction, (g_start + g_end)'s / 2 + s'(B_end - B_start)s / 12 (exact where E is quartic along s), whose plain
    rule's error |s'(B_end - B_start)s| / 12 bounds the corrected one's; the penalty and the determinant terms
    change in closed form (the states' ``rest``), and eps adds both states' own errors. The fixed points' gradient
    error along s (their perturbation probes) is not charged yet (``OuterFit.fixed_point_term_measured``)."""
    if not np.any(move):
        return end.rest - start.rest, start.error + end.error
    along = (prior.coefficient_map @ move)[:, None]

    def curvature(state: _State) -> float:
        return float(move @ state.fixed_curvature @ move) + float(state.correction.on(along)[0, 0])

    end_correction = (curvature(end) - curvature(start)) / 12.0
    gain = 0.5 * float((start.gradient + end.gradient) @ move) + end_correction + end.rest - start.rest
    return gain, abs(end_correction) + start.error + end.error


@dataclass(frozen=True)
class _OuterTrial:
    """A pending outer trial: the joint step to ``hyperparameters`` at ``fraction`` of its segment (``newton`` None;
    ``predicted`` the model's gain pi_k for the whole step, ``realized`` the previous, longer halving's realized
    gain), or an inner x step at the state's weights (``newton``, its ``proposal`` and trust ``radius``, and whether
    it ``polishes`` x after a refused joint trial or leaves a saddle). ``remaining`` and ``certifying`` are the
    state's certificate."""

    hyperparameters: MixtureHyperparameters
    step: HyperStep | None
    remaining: float
    certifying: bool
    fraction: float = 1.0
    predicted: float = np.nan
    realized: float = -np.inf
    newton: _NewtonB | None = None
    proposal: F64Array | None = None
    radius: float = np.nan
    polishes: bool = False
    # A joint trial from a state where V's formula has no certified value (x_k outside every basin of its integrand at
    # rho_k): it enters one, accepted where the trial's state is certified, and uncounted (V at x_k is unmeasured).
    enters: bool = False
    # The tolerance the weights were searched to when this trial was planned (``weight_tolerances``).
    weights_tolerance: float = np.nan


@_step_scoped
def fit_hyperparameters(
    prior: ScaleMixturePrior, starts: Sequence[MixtureHyperparameters], fixed_points: FixedPoints, working_bytes: int, tolerance: float
) -> list[OuterFit]:
    """Every model's empirical Bayes at its EP fixed point, certified to ``tolerance`` nats (lead ruling: Newton-B,
    never plain EP-EM), as the ascent of one function: V(rho), the certified EP-Laplace evidence of the weights,
    with x its inner solve (theory-ep).

    At an EP fixed point x_k the EP evidence's x-gradient is the fixed-cavity g and its negative Hessian the total
    curvature B = A + C, so its local model there is F_k - 1/2 (x - x_k)'C(x - x_k), which ``hyper_step`` searches
    (``_Anchor``): its V is V's own model at x_k. The fixed-cavity maximizer (the EM step) solves with A + S instead:
    to first order it maps the error e to (I - (A + S)^-1 (B + S)) e, which diverges wherever that pencil has an
    eigenvalue above 2, and where A + S is indefinite the fixed-cavity objective has no maximum near x at all.

    Each outer iteration, at the state (x_k, rho_k) and its fixed point:
    - ``hyper_step`` maximizes the model over the weights, to rho-hat and the model's own maximizer x there; the model
      predicts V to gain pi_k = V_k(rho-hat) - W_k, with W_k V's formula at the state itself (``_outer_state``).
    - The joint trial (rho-hat, x) is solved for its fixed point and its own B, and V's realized gain along the step
      measured (``_path_gain``). It is accepted when that gain less its resolution exceeds ``tolerance``: V(rho) then
      rises by more than the fit's resolution at every accepted joint step, so there are at most
      (V* - V(rho_0)) / tolerance of them. A refused joint trial is halved along its segment at rho-hat (where that
      keeps the edges) while each halving raises the realized gain. From a state where V's formula has no certified
      value (x_k outside every basin of its integrand at rho_k, as at an arbitrary start) the joint trial enters one:
      it is accepted where the trial's own state is certified, uncounted, and every accepted joint step lands in a
      certified state, so only an inner step can leave one.
    - Otherwise x takes inner steps at rho_k. An inner step ascends L_rho, not V (a function of the weights alone),
      so it is accepted on any certified rise: the natural monotonicity test where B + S is positive definite
      (Deuflhard, Newton Methods for Nonlinear Problems, 2004, Section 3.1.4: at the trial's fixed point
      g'(B + S)^-1 g, in the step's own B + S, must fall), the trapezoid rule of the gradient's path integral where
      it is not (the step then maximizes the model inside a radius, More and Sorensen, starting at the Cauchy step's
      length on |B + S|, ``_cauchy_radius``, and doubling when an accepted step reached it); a refused one halves.
      After a refused joint trial they polish x at rho_k until a step halves to double precision, and the state is
      planned once more; where ``hyper_step`` finds no certified maximum (a saddle), one accepted inner step leads to
      a new plan.
    A model is certified when its state is inside a basin, the predicted gain plus the weights' remaining stationarity
    gain and the prediction's measured error (the model's realized remainder at its last whole joint trial and both
    values' own errors) is at most ``tolerance``, and the joint trial then moves q's mean by at most 1 / K in q's
    posterior metric, KL(q || q') <= 1 / (2K) nats (MODEL.md: the certificate includes the prediction change; in
    evidence units, lead ruling via speed-smalln), at the trial's own EP fixed point. With K = 1 / (2 tolerance)
    posterior draws that is the scorer's own Monte Carlo resolution, as for the EP fixed point. A trial that fails
    the check is taken as an ordinary joint trial. Where the weights' stationarity bound is what stops the
    certificate, it is tightened to the share the rest leaves it (``HyperStep.tighten``). A polished state whose
    joint trial is refused again, or an inner step that halves to double precision at a saddle's weights with an
    evaluated step, is returned uncertified with its measured remaining gain.

    On return the oracle's state is each model's returned fixed point (it is solved there once more where the last
    trial was elsewhere).
    """
    count = len(starts)
    hyperparameters = list(starts)
    points = list(fixed_points(hyperparameters))
    if any(point is None for point in points):
        raise FloatingPointError("a starting point has no certified EP fixed point")

    def solve_state(trial: MixtureHyperparameters, point: FixedPoint) -> tuple[CurvatureCorrection, _State | None]:
        correction = curvature_correction(prior, trial.coefficients, point.cavity, point.posterior, working_bytes, tolerance)
        return correction, _outer_state(prior, trial, point, correction, working_bytes, tolerance)

    solved = [solve_state(start, point) for start, point in zip(hyperparameters, points)]
    corrections = [correction for correction, _state in solved]
    states = [state for _correction, state in solved]
    fits: list[OuterFit | None] = [None] * count
    pending: list[_OuterTrial | None] = [None] * count
    radii: list[float | None] = [None] * count
    iterations, halvings, unresolved = [0] * count, [0] * count, [0] * count
    # |G - pi| at the model's last whole joint trial: the local model's measured error (unknown before the first).
    remainders = [np.inf] * count
    # A release (the step frees an edge weight) is taken as its weights' move first, at x_k and its fixed point
    # (the fixed point does not depend on the weights), and x then follows by inner steps at the freed weights; the
    # joint step stands where x's certified V at the freed weights, once polished, is above the state's it left by
    # more than the tolerance and both errors. ``anchors`` holds that state (and the step's predicted gain) until
    # then, and ``refused_releases`` the blocks a model was returned from, which its next hyper steps hold at their
    # edge (``hyper_step``'s ``held_edges``): the interior there was measured at its own fixed point and lost.
    anchors: list[tuple[MixtureHyperparameters, FixedPoint, CurvatureCorrection, _State, float] | None] = [None] * count
    refused_releases: list[set[frozenset[int]]] = [set() for _model in range(count)]
    # Each model's last evaluated hyper step: the certificate an uncertified return at the family's boundary reports.
    last_steps: list[HyperStep | None] = [None] * count
    # The realized gains of each model's consecutive accepted inner steps at its current weights (``_State.tail``).
    inner_gains: list[list[float]] = [[] for _model in range(count)]

    def tail_of(model: int) -> float:
        """The remaining gain the last two inner gains' contraction bounds: g r / (1 - r) with r their ratio, where
        two gains are measured and the ratio is below one (the geometric tail of a profile approaching its
        supremum along a ray); infinite where the gains do not contract, zero where none is measured yet."""
        gains = inner_gains[model]
        if len(gains) < 2:
            return 0.0
        previous, last = gains[-2], gains[-1]
        if last <= 0.0:
            return 0.0
        if previous <= 0.0:
            return np.inf
        rate = last / previous
        return last * rate / (1.0 - rate) if rate < 1.0 else np.inf
    # The tolerance each model's weights are searched to: the fit's, until a decision finds their remaining gain is what
    # stops the certificate and their bound cannot be tightened to the share the rest leaves (``decide``).
    weight_tolerances = [tolerance] * count
    # Whether the oracle's last solve for the model was elsewhere than the point it returns.
    displaced = [False] * count
    histories: list[list[float]] = [[] for _model in range(count)]

    def settle_release(model: int) -> None:
        """x polished at the freed weights: the release stands where its certified V is above the state's it left by
        more than the tolerance and both errors (its realized gain measured against the step's prediction);
        otherwise the model returns to that state and this release is not planned again."""
        anchor = anchors[model]
        assert anchor is not None
        anchor_hyperparameters, anchor_point, anchor_correction, anchor_state, predicted = anchor
        polished_state = states[model]
        realized = -np.inf if polished_state is None else polished_state.value - anchor_state.value
        resolution = np.inf if polished_state is None else polished_state.error + anchor_state.error
        remainders[model] = abs(realized - predicted) if np.isfinite(realized) else np.inf
        anchors[model] = None
        if polished_state is not None and realized - resolution > tolerance:
            states[model] = replace(polished_state, polished=True)
            return
        refused_releases[model].add(frozenset(
            int(position) for position in np.flatnonzero(np.isfinite(hyperparameters[model].log_smoothing) & ~np.isfinite(anchor_hyperparameters.log_smoothing))
        ))
        hyperparameters[model], points[model], corrections[model], states[model] = anchor_hyperparameters, anchor_point, anchor_correction, anchor_state
        displaced[model], radii[model] = True, None

    def inner(model: int, step: HyperStep | None, remaining: float, polishes: bool) -> _OuterTrial | None:
        newton = _newton_b(prior, hyperparameters[model].log_smoothing, hyperparameters[model].coefficients, points[model], corrections[model], working_bytes)
        tail = tail_of(model)
        if polishes and newton.definite and max(newton.decrement, tail) <= tolerance and states[model] is not None:
            # x is at its maximum at rho_k to the certificate's resolution: the model predicts less gain than the
            # tolerance, and so does the inner steps' own contraction (``_State.tail``), so the state is planned
            # once more (a release in flight is settled here). A polish that only ends on a step below x's own
            # resolution walks a flat ray to the family's boundary: on gene 1 [real] the width -> 0 ray, 2.8 units
            # a step with 1e-4 to 1e-7 nats each, until the density collapsed.
            states[model] = replace(states[model], tail=tail)
            if anchors[model] is not None:
                settle_release(model)
            else:
                states[model] = replace(states[model], polished=True)
            return None
        radius = radii[model]
        if radius is None:
            radius = _cauchy_radius(newton)
            radii[model] = radius
        proposal = _proposal(newton, radius)
        return _OuterTrial(_trial(newton, proposal), step, remaining, False, newton=newton, proposal=proposal, radius=radius, polishes=polishes)

    def decide(model: int, state: _State, step: HyperStep) -> tuple[_State, HyperStep, float, float]:
        """The certificate at the state and its step: (state, step, predicted, remaining), each resolvable piece
        tightened where they, not the decision's fixed part, stop it.

        remaining = predicted + fixed + S, with fixed the model's measured remainder at its last whole joint trial
        and S the resolvable pieces: the state's certified V error, the step's, and the weights' remaining gain. Where
        predicted + fixed < tolerance < remaining, every piece is scaled by theta = M / S, M = tolerance - predicted -
        fixed (theory-ep's per-decision tightening: no fixed shares): each V is re-certified to theta times its
        error (``_outer_state`` and ``HyperStep.resolve``: more directions integrated exactly) and the weights' bound
        is tightened to theta times their gain (``HyperStep.tighten``), and the certificate is read again. A piece
        that cannot reach its share stays as measured, so a decision fails only on what no resolution removes."""
        fixed = remainders[model] + state.tail
        predicted = step.evidence - state.value
        pieces = (state.error, step.evidence_error, step.stationarity_gain)
        resolvable = float(sum(pieces))
        remaining = predicted + fixed + resolvable
        if not (predicted + fixed < tolerance < remaining) or not 0.0 < resolvable < np.inf:
            return state, step, predicted, remaining
        theta = (tolerance - predicted - fixed) / resolvable
        if state.error > 0.0:
            resolved_state = _outer_state(prior, hyperparameters[model], points[model], corrections[model], working_bytes, theta * state.error)
            if resolved_state is not None:
                state = replace(resolved_state, polished=state.polished, tail=state.tail)
        if step.evidence_error > 0.0 and step.resolve is not None:
            resolved_step = step.resolve(theta * step.evidence_error)
            if resolved_step is not None:
                step = resolved_step
        if step.stationarity_gain > 0.0 and step.tighten is not None:
            share = theta * step.stationarity_gain
            tightened = step.tighten(share)
            if tightened is not None:
                step = tightened
            if step.stationarity_gain > share:
                # The weights' bound cannot be tightened to their share at these weights: the next search moves them
                # further, to the resolution their share asks (a continued search, warm from these weights).
                weight_tolerances[model] = min(weight_tolerances[model], share)
        predicted = step.evidence - state.value
        return state, step, predicted, predicted + fixed + state.error + step.evidence_error + step.stationarity_gain

    def plan(model: int) -> _OuterTrial | None:
        state = states[model]
        planned = weight_tolerances[model]
        # A polish's geometric sequence of gains belongs to one plan.
        inner_gains[model] = []
        # A release is judged from a polished state (below), so an unpolished state's hyper step holds every edge:
        # its release trials (three maximizations per block at the range's centre) would be discarded.
        held = frozenset().union(*refused_releases[model])
        if state is not None and not state.polished:
            held = held | frozenset(int(position) for position in np.flatnonzero(hyperparameters[model].log_smoothing == np.inf))
        try:
            step = hyper_step(prior, hyperparameters[model], points[model].cavity, corrections[model], working_bytes, planned, held_edges=held)
        except FloatingPointError:
            # V's model has no certified maximum here (an indefinite iterate): the weights wait, and x leaves the saddle.
            step = None
        if step is None:
            histories[model].append(np.inf)
            return inner(model, step, np.inf, False)
        last_steps[model] = step
        if state is None:
            histories[model].append(np.inf)
            return _OuterTrial(step.hyperparameters, step, np.inf, False, enters=True)
        state, step, predicted, remaining = decide(model, state, step)
        states[model] = state
        histories[model].append(float(remaining))
        entry = _OuterTrial(step.hyperparameters, step, remaining, remaining <= tolerance, predicted=predicted, weights_tolerance=planned)
        released = frozenset(
            int(position) for position in np.flatnonzero(np.isfinite(step.hyperparameters.log_smoothing) & ~np.isfinite(hyperparameters[model].log_smoothing))
        )
        if not released or entry.certifying:
            return entry
        assert not (released & frozenset().union(*refused_releases[model]))
        if not state.polished:
            # A release is judged against a polished state (x at its maximum at these weights, the model's own
            # account of them reliable there), so x polishes here first; the hyper step from the polished state
            # proposes the release again where it still holds.
            return inner(model, step, remaining, True)
        # The joint step to a freed edge cannot be tested as one move: its x-part is the whole distance from the
        # edge's density to the interior's (|move| 43 on the mean-field test problem), where the path integral's
        # end correction alone is 10 nats, and a halving cannot keep the weights at their edge. So the trial is
        # taken at its own fixed point without a gain test, x is polished there by inner steps (accepted on the
        # model as always), and the polished certified V against the state's here decides the release (``anchors``).
        anchors[model] = (hyperparameters[model], points[model], corrections[model], state, predicted)
        return entry

    def uncertified(model: int, entry: _OuterTrial, step: HyperStep, newton_decrement: float) -> None:
        fits[model] = OuterFit(
            hyperparameters=hyperparameters[model], step=step, newton_decrement=newton_decrement, remaining_gain=entry.remaining,
            prediction_move=np.inf, prediction_tolerance=2.0 * tolerance, iterations=iterations[model], halvings=halvings[model],
            unresolved=unresolved[model], history=tuple(histories[model]), certified=False,
        )
        pending[model] = None

    while True:
        for model in range(count):
            if fits[model] is None and pending[model] is None:
                pending[model] = plan(model)
        if all(fit is not None for fit in fits):
            if any(displaced):
                points = list(fixed_points(hyperparameters))
            return [fit for fit in fits if fit is not None]
        trials = [hyperparameters[model] if entry is None else entry.hyperparameters for model, entry in enumerate(pending)]
        trial_points = list(fixed_points(trials))
        for model, entry in enumerate(pending):
            if entry is None:
                # An oracle that refuses refuses every model at once; a model without a trial keeps its point.
                if trial_points[model] is not None:
                    points[model] = trial_points[model]
                displaced[model] = False
                continue
            trial, trial_point = trials[model], trial_points[model]
            displaced[model] = True
            if entry.newton is None and entry.enters:
                # The joint trial from a state outside every basin: accepted where the trial's state is certified.
                assert entry.step is not None
                if trial_point is None:
                    unresolved[model] += 1
                else:
                    trial_correction, trial_state = solve_state(trial, trial_point)
                    if trial_state is not None:
                        hyperparameters[model], points[model], corrections[model], states[model] = trial, trial_point, trial_correction, trial_state
                        displaced[model], pending[model] = False, None
                        iterations[model] += 1
                        continue
                halvings[model] += 1
                pending[model] = inner(model, entry.step, entry.remaining, False)
                continue
            if entry.newton is None and anchors[model] is not None:
                # A release in flight: the trial stands provisionally at its own fixed point, and x polishes there.
                assert entry.step is not None
                if trial_point is None:
                    unresolved[model] += 1
                    anchor_hyperparameters, anchor_point, anchor_correction, anchor_state, _predicted = anchors[model]
                    refused_releases[model].add(frozenset(
                        int(position) for position in np.flatnonzero(np.isfinite(trial.log_smoothing) & ~np.isfinite(anchor_hyperparameters.log_smoothing))
                    ))
                    hyperparameters[model], points[model], corrections[model], states[model] = anchor_hyperparameters, anchor_point, anchor_correction, anchor_state
                    anchors[model], radii[model], pending[model] = None, None, None
                    continue
                trial_correction, trial_state = solve_state(trial, trial_point)
                hyperparameters[model], points[model], corrections[model], states[model] = trial, trial_point, trial_correction, trial_state
                displaced[model], radii[model] = False, None
                iterations[model] += 1
                pending[model] = inner(model, entry.step, entry.remaining, True)
                continue
            if entry.newton is None:
                # The joint trial.
                state = states[model]
                assert state is not None and entry.step is not None
                if trial_point is not None and entry.certifying and entry.fraction == 1.0:
                    current = points[model]
                    moves = np.atleast_1d(np.asarray(current.precision_norm(trial_point.mean - current.mean), dtype=np.float64))
                    allowed = np.full(moves.shape[0], 2.0 * tolerance)
                    # Reported as the most-used share of a block's budget, in that block's units.
                    with np.errstate(divide="ignore", invalid="ignore"):
                        shares = np.where(allowed > 0.0, moves / allowed, np.where(moves > 0.0, np.inf, 0.0))
                    worst = int(np.argmax(shares))
                    if bool(np.all(moves <= allowed)):
                        # The oracle's state is the trial's certified fixed point: the fit returns the trial, so its
                        # hyperparameters and its fixed point are one model (review-mathbugs E1). The certificate covers
                        # the move: V's remaining gain at the state, and q's mean moved by at most 1 / K in its metric.
                        hyperparameters[model], points[model], displaced[model] = trial, trial_point, False
                        fits[model] = OuterFit(
                            hyperparameters=trial, step=entry.step, newton_decrement=state.decrement, remaining_gain=entry.remaining,
                            prediction_move=float(moves[worst]), prediction_tolerance=float(allowed[worst]), iterations=iterations[model],
                            halvings=halvings[model], unresolved=unresolved[model], history=tuple(histories[model]),
                        )
                        pending[model] = None
                        continue
                gain = -np.inf
                previous_remainder = remainders[model]
                if trial_point is None:
                    # No EP fixed point at the trial: refused, like a trial the test rejects, and counted.
                    unresolved[model] += 1
                else:
                    trial_correction, trial_state = solve_state(trial, trial_point)
                    if trial_state is not None:
                        gain, resolution = _path_gain(prior, state, trial_state, trial.coefficients - hyperparameters[model].coefficients)
                        if entry.fraction == 1.0:
                            remainders[model] = abs(gain - entry.predicted)
                        if gain - resolution > tolerance:
                            hyperparameters[model], points[model], corrections[model], states[model] = trial, trial_point, trial_correction, trial_state
                            displaced[model], pending[model] = False, None
                            iterations[model] += 1
                            continue
                halvings[model] += 1
                target = entry.step.hyperparameters
                segment = target.coefficients - hyperparameters[model].coefficients
                fraction = 0.5 * entry.fraction
                same_edges = np.array_equal(target.log_smoothing == np.inf, hyperparameters[model].log_smoothing == np.inf)
                longer = fraction * float(np.linalg.norm(segment)) > _HALF_PRECISION * (1.0 + float(np.max(np.abs(hyperparameters[model].coefficients))))
                if state.polished and remainders[model] < previous_remainder:
                    # This whole trial re-measured the model's remainder below the one the plan's certificate carried
                    # (on the mean-field test problem the first whole trial, from the uncertified start, left 0.106
                    # nats of remainder that the polished state's own zero-length trial measured at 1e-6; on gene 1
                    # a 0.0247 the certifying trial measured at 2e-7): the certificate is read again with the fresh
                    # remainder, before any halving. A remainder that did not fall leaves the certificate as it was,
                    # so this replans at most once per measured decrease.
                    pending[model] = None
                elif same_edges and longer and (np.isneginf(gain) or gain > entry.realized):
                    halved = MixtureHyperparameters(coefficients=hyperparameters[model].coefficients + fraction * segment, log_smoothing=target.log_smoothing)
                    pending[model] = replace(entry, hyperparameters=halved, certifying=False, fraction=fraction, realized=max(gain, entry.realized))
                elif state.polished and weight_tolerances[model] < entry.weights_tolerance:
                    # The decision tightened the weights' tolerance since this plan: the polished state is planned once
                    # more, with the weights searched to what their share asks.
                    pending[model] = None
                elif state.polished:
                    # x is at its maximum at rho_k to double precision and the joint step still resolves no gain.
                    uncertified(model, entry, entry.step, state.decrement)
                else:
                    pending[model] = inner(model, entry.step, entry.remaining, True)
                continue
            # An inner step at the state's weights. A proposal below double precision's resolution of x cannot be told
            # from the point it leaves: the decrements of both sit at their rounding, where the monotonicity test is a
            # coin toss (a polish that accepted 3,000 such steps on the mean-field test problem), so it is never
            # accepted, and x counts as at its maximum to double precision.
            newton, proposal = entry.newton, entry.proposal
            assert proposal is not None
            length = float(np.linalg.norm(proposal))
            resolved = length > _HALF_PRECISION * (1.0 + float(np.max(np.abs(newton.origin))))
            if trial_point is None:
                accepted = False
                unresolved[model] += 1
            elif not resolved:
                accepted = False
            else:
                gradient = _penalized_gradient(
                    newton.view, newton.log_smoothing[np.isfinite(newton.log_smoothing)], newton.origin + proposal, trial_point.cavity, working_bytes,
                )
                if not np.any(gradient):
                    # The data no longer see x at the trial: every derivative is at rounding, the density has collapsed
                    # onto a lattice node (the mixing density's width -> 0 boundary: on gene 1 [real] with the
                    # mean-field fixed point solved to its Newton decrement, V's supremum lies there, Newton's steps
                    # along the ray being 2.8 units each with the gradient and the curvature falling by the same
                    # factor). x is at its maximum to double precision from here on, and V has no Laplace value there
                    # (the fixed-cavity curvature is singular), so the trial is the fit's state and the fit returns
                    # uncertified with its last evaluated step; the boundary model is the open work (MODEL.md).
                    hyperparameters[model], points[model] = trial, trial_point
                    displaced[model] = False
                    iterations[model] += 1
                    last_step = last_steps[model]
                    if last_step is None:
                        raise NoCertifiedProgress("the Newton-B step reaches the mixing density's boundary before any hyper step certified a value")
                    uncertified(model, replace(entry, remaining=np.inf), last_step, newton.decrement)
                    continue
                # The step is an ascent where the trapezoid rule of the two fixed points' gradients along it is
                # positive (the path integral of E's gradient to first order). A decrement test in the origin's
                # metric refused every step inside the radius on the mean-field test problem: with B + S's
                # eigenvalues 2e-5 and 1e-2 the decrement is the weak direction's g^2 / lambda, which no step shorter
                # than Newton's thousand units lowers, so the loop halved to x's resolution and read a maximum where
                # the state's own error was 9.6 nats.
                accepted = 0.5 * float((newton.gradient + gradient) @ proposal) > 0.0
            if accepted:
                trial_correction, trial_state = solve_state(trial, trial_point)
                previous_state = states[model]
                if previous_state is not None and trial_state is not None:
                    inner_gain, _inner_resolution = _path_gain(prior, previous_state, trial_state, trial.coefficients - hyperparameters[model].coefficients)
                    inner_gains[model].append(float(inner_gain))
                else:
                    inner_gains[model] = []
                hyperparameters[model], points[model], corrections[model], states[model] = trial, trial_point, trial_correction, trial_state
                displaced[model] = False
                iterations[model] += 1
                # A step that reached the radius widens it; one that stopped short (Newton's, fitting) keeps it.
                radii[model] = 2.0 * entry.radius if length >= entry.radius * (1.0 - _HALF_PRECISION) else entry.radius
                # A polishing step continues at the same weights; one that left a saddle leads to a new plan.
                pending[model] = inner(model, entry.step, entry.remaining, True) if entry.polishes else None
                continue
            halvings[model] += 1
            inner_gains[model] = []
            if not resolved:
                if entry.polishes and anchors[model] is not None:
                    settle_release(model)
                    pending[model] = None
                    continue
                if entry.polishes and states[model] is not None:
                    # x is at its maximum at rho_k to double precision: the state is planned once more.
                    states[model] = replace(states[model], polished=True)
                    pending[model] = None
                    continue
                if newton.definite and entry.step is not None:
                    # x is at its maximum to double precision (no trial can lower a decrement at its rounding) and what
                    # stops the certificate is not x: the fit is returned uncertified with its measured remaining gain.
                    uncertified(model, entry, entry.step, newton.decrement)
                    continue
                raise NoCertifiedProgress(
                    "the Newton-B step makes no certified progress at the EP fixed point "
                    + ("(B + S is indefinite there)" if not newton.definite else "(the weights have no evaluated step)")
                )
            radius = 0.5 * length
            radii[model] = radius
            shorter = _proposal(newton, radius)
            pending[model] = replace(entry, hyperparameters=_trial(newton, shorter), proposal=shorter, radius=radius, fraction=0.5 * entry.fraction)
