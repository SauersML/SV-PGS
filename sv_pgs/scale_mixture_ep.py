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
linear part), which are
profiled as fixed effects: integrating them under a flat prior diverges
where the likelihood tends to a positive constant. Its gradient is exact:
dx/drho through the observed Hessian, and the change of H with x through the
third derivatives of log Z. Across the outer loop the curvature is the
EP-re-solved one; at fixed cavities it is H.

Outer loop (``fit_hyperparameters``). At an EP fixed point the EP evidence's
gradient in x is the fixed-cavity one, and its negative Hessian is the total
curvature B + S, with the cavities re-solved (``_total_curvature``). So x moves
by Newton on B + S, globalized by the natural monotonicity test. It never moves
by the fixed-cavity maximizer: that EP-EM step solves with A + S, and it
overshoots where (A + S)^-1 (B + S) exceeds 2.

Every derivative is computed in z = (eta_1, ..., eta_C, theta), where each variant's log Z_j depends on its
class's eta_c and its own log u_j, and carried to x by the linear map M.

Noise. A quantitative trait's residual variance takes the stationarity form of its type-II ML with q held,
sigma^2' = (RSS + sigma^2 gamma) / (n - k), with gamma = p - sum_j tau_j Sigma_jj and the covariates' flat prior
removing k. Its fixed point is the MacKay form RSS / (n - k - gamma), which is undefined where gamma >= n - k.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterator, Sequence

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import make_interp_spline
from scipy.linalg import solve_triangular
from scipy.optimize import brentq
from scipy.special import erfcx

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
    weights not held at an edge or a bound of their range, by central
    differences of the B-evidence; ``stationarity_steps`` and
    ``stationarity_errors`` are each difference's step and error bound, and ``stationarity_gain`` the certified
    upper bound 1/2 sum (|c| + E)^2 / s on the gain a Newton step on them could still find (at most the tolerance).
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


def initial_hyperparameters(prior: ScaleMixturePrior, mean_variance: float | None = None) -> MixtureHyperparameters:
    """A start, not a prior: every class at one log-normal on the lattice, of the width whose density at the lattice's
    ends is eps of its peak when it is centred; no deviation or annotation effect; unit penalty weights.

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
    return MixtureHyperparameters(coefficients=coefficients, log_smoothing=np.zeros(len(prior.smoothing_blocks)))


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
    natural = [(order, 0.0) for order in range(ROUGHNESS_ORDER, 2 * ROUGHNESS_ORDER - 1)]
    spline = make_interp_spline(old_nodes, log_density.T, k=2 * ROUGHNESS_ORDER - 1, bc_type=(natural, natural), axis=0)
    new_nodes = np.asarray(nodes, dtype=np.float64)
    inside = np.clip(new_nodes, old_nodes[0], old_nodes[-1])
    slopes = np.where(new_nodes < old_nodes[0], spline(old_nodes[0], nu=1)[:, None], spline(old_nodes[-1], nu=1)[:, None])
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
    return _log_sum_exp(_kernel_terms(log_density, log_scale_rows, grid, floor, precision, shift)[3], axis=1)


def _components(
    log_density: F64Array, log_scale_rows: F64Array, grid: F64Array, floor: float, precision: F64Array, shift: F64Array
) -> _Components:
    """With q = vP, r = 1/(1+q) and a = h^2 v r (so dr/deta = -r(1 - r) and da/deta = a r), each derivative of
    log Z_k in eta = log u is a A_n(r) - B_n(r): d1 = (a - q) r / 2, then A_(n+1) = r A_n - r(1 - r) A_n' and
    B_(n+1) = -r(1 - r) B_n', giving A_2 = r^2 - r/2, B_2 = r(1 - r)/2, A_3 = 3r^3 - 3r^2 + r/2,
    B_3 = r(1 - r)(2r - 1)/2, and A_4 = r A_3 - r(1 - r)(9r^2 - 6r + 1/2), B_4 = -r(1 - r)(-3r^2 + 3r - 1/2).
    Nodes below ``floor`` have a flat kernel: v = 0 there."""
    conditional, retained, ratio_retained, log_component, signal = _kernel_terms(log_density, log_scale_rows, grid, floor, precision, shift)
    log_normalizer = _log_sum_exp(log_component, axis=1)
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
    """q's linear responses at the EP fixed point, for the total curvature B: ``solve(R, e)`` is Sigma R, each column
    to relative error e in the posterior metric, and ``variance_jvp(W)`` is -(Sigma o Sigma) W, both (p x r). Stage 2
    answers them with extra right-hand sides of its solve and with ``marginal_variances.variance_jvp``.

    ``linear_response(left, right, diagonal, weight, B)``, when a posterior can give it, is the exact solution X of
    (I - (I - diag(weight) (Sigma o Sigma)) (diag(left) Sigma diag(right) + diag(diagonal))) X = B: the linear response
    ``_total_curvature`` otherwise finds by GMRES (speed-smalln: the small-n route factors this p x p matrix once)."""

    solve: Callable[[F64Array, float], F64Array]
    variance_jvp: Callable[[F64Array], F64Array]
    linear_response: Callable[[F64Array, F64Array, F64Array, F64Array, F64Array], F64Array] | None = None
    # ``local_response(left, right, diagonal, weight)``: V -> M^-1 V for the same matrix with Sigma replaced by its
    # block-local part (read-free), the preconditioner of the Krylov route (``krylov_recycle``, lane speed-recycle).
    local_response: Callable[[F64Array, F64Array, F64Array, F64Array], Callable[[F64Array], F64Array]] | None = None


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
    ) -> None:
        self.coefficient_map = coefficient_map
        self.matrix = matrix
        self._mapping = mapping
        self._columns = columns
        self._fixed = fixed_curvature
        # An orthonormal basis Q (x coordinates) of the directions solved so far, and B_z M Q.
        self._basis: F64Array | None = None
        self._total: F64Array | None = None

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
    return CurvatureCorrection(
        mapping=prior.coefficient_map,
        columns=lambda directions: _total_curvature_columns(prior, coefficients, cavity, posterior, working_bytes, relative_tolerance, directions),
        fixed_curvature=0.5 * (fixed + fixed.T),
    )


def diagonal_posterior(variance: F64Array) -> GaussianPosterior:
    """The posterior of independent effects (orthogonal design, normal means): Sigma = diag(variance). There the
    cavities do not move with the prior, so B equals the fixed-cavity curvature."""
    column = np.asarray(variance, dtype=np.float64)[:, None]
    return GaussianPosterior(solve=lambda right, _relative_tolerance: column * right, variance_jvp=lambda weights: -np.square(column) * weights)


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
    the restart length that fits ``working_bytes``.
    """
    derivatives = _variant_derivatives(prior, coefficients, cavity, working_bytes)
    mean_by_z = _through_z(prior, derivatives.mean_by_density, derivatives.mean_by_log_scale, directions)
    variance_by_z = _through_z(prior, derivatives.second_by_density, derivatives.second_by_log_scale, directions) - 2.0 * derivatives.mean[:, None] * mean_by_z
    variance = derivatives.variance[:, None]

    def through(precision_step: F64Array, inner: float, affine: bool = True) -> tuple[F64Array, F64Array, F64Array]:
        # The map is affine in dP. With ``affine`` False its constant (the E terms) is dropped, which leaves its linear
        # part exactly: GMRES applies that, so each solve inside is accurate relative to what it applies, not to the
        # constant, as a difference of two solves would be (speed-recycle a7752ae).
        # A scalar zero, so the linear part applies to any number of columns (a block Krylov step's).
        mean_constant = mean_by_z if affine else 0.0
        mean_step = posterior.solve(
            (derivatives.mean + derivatives.mean_by_precision / derivatives.variance)[:, None] * precision_step + mean_constant / variance, inner
        )
        shift_step = (mean_step - derivatives.mean_by_precision[:, None] * precision_step - mean_constant) / variance
        variance_step = derivatives.variance_by_shift[:, None] * shift_step + derivatives.variance_by_precision[:, None] * precision_step
        if affine:
            variance_step = variance_step + variance_by_z
        response = variance_step / variance**2 + precision_step
        return shift_step, response + posterior.variance_jvp(response) / variance**2, response

    shape = mean_by_z.shape
    # ``through`` is affine in dP, with linear part (I - diag(1/v^2) (Sigma o Sigma)) R and R = diag(v_h / v^3) Sigma
    # diag(m + m_P / v) + diag(1 + v_P / v^2 - v_h m_P / v^3).
    tilted = derivatives.variance
    left = derivatives.variance_by_shift / tilted**3
    gain = derivatives.mean + derivatives.mean_by_precision / tilted
    diagonal = 1.0 + derivatives.variance_by_precision / tilted**2 - derivatives.variance_by_shift * derivatives.mean_by_precision / tilted**3
    weight = 1.0 / tilted**2
    if posterior.linear_response is not None:
        # The posterior solves its fixed point exactly.
        _shift, offset, _start = through(np.zeros(shape), relative_tolerance)
        precision_step = posterior.linear_response(left, gain, diagonal, weight, offset)
        return _total_from_response(prior, coefficients, cavity, derivatives, directions, through(precision_step, relative_tolerance)[0], precision_step, working_bytes)
    size = int(np.prod(shape))
    # The linear part applies one p x p operator to every direction column, so the solve is block Krylov over the
    # columns (``krylov_recycle.block_gcro_dr``): each application serves them all, restarts keep the slowest harmonic
    # Ritz space, and ``local_response`` (read-free) preconditions it. At most ``size`` applications, the flattened
    # GMRES's own cap. Carrying the kept space to the next outer step measured no gain once preconditioned (it costs one
    # application per solve [sim-only, speed-recycle]), so each solve starts without it.
    # Each product solves the posterior only to ``inner``, so the operator itself errs, and the Krylov residual
    # estimate can sit far below the true one (inexact Krylov: Simoncini and Szyld, SIAM J. Sci. Comput. 25, 2003):
    # half the tolerance goes to the Krylov solve, half to the products. The true residual is measured once the solve
    # stops; while it exceeds the tolerance, the inner solves tighten by the measured excess (the operator's own
    # amplification of their error) and the solve continues from where it stopped.
    precondition = None if posterior.local_response is None else posterior.local_response(left, gain, diagonal, weight)
    inner = relative_tolerance
    solution = np.zeros(shape)
    previous = np.inf
    while True:
        try:
            _shift, offset, start_response = through(np.zeros(shape), inner)
        except ValueError as error:
            if inner >= relative_tolerance:
                raise
            # The solver cannot reach the tightened accuracy in float64: the response is not resolvable here.
            raise LinearResponseError(f"the EP fixed point's linear response cannot be resolved: {error}") from error
        # The map's last step cancels the diagonal of Sigma o Sigma against v^2: its value is known only to eps times
        # the terms that cancel, which is where GMRES's residual can stop.
        rounding = _EPSILON * float(np.linalg.norm(start_response))

        def linear_part(values: F64Array, inner: float = inner) -> F64Array:
            return values - through(values, inner, affine=False)[1]

        try:
            result = block_gcro_dr(
                linear_part, offset, relative_tolerance=0.5 * relative_tolerance, absolute_tolerance=rounding, working_bytes=working_bytes,
                application_limit=size, start=solution, precondition=precondition,
            )
        except FloatingPointError as error:
            raise LinearResponseError(f"the EP fixed point's linear response did not converge: {error}") from error
        solution = result.solution
        target = max(relative_tolerance * float(np.linalg.norm(offset)), rounding)
        residual = result.residual_norm
        if residual <= target:
            break
        if residual >= previous:
            raise LinearResponseError(f"the EP fixed point's linear response did not converge (true residual {residual:.3e} against {target:.3e})")
        previous = residual
        inner *= 0.5 * target / residual
    precision_step = solution
    return _total_from_response(prior, coefficients, cavity, derivatives, directions, through(precision_step, inner)[0], precision_step, working_bytes)


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


def _line(
    prior: ScaleMixturePrior, log_smoothing: F64Array, origin: F64Array, direction: F64Array, cavity: Cavity, working_bytes: int
) -> Callable[[F64Array], F64Array]:
    """The penalized objective F(x + t b) - P(x + t b) as a function of the steps t, each call one pass over the
    variants for all its steps.

    z = M x is affine in t, so every step's class log densities and log scales follow from those of x and b, and P
    is exactly quadratic in t: only the log normalizers are evaluated per step."""
    density, _scale = _density_and_scale(prior, origin)
    density_step, scale_step = _density_and_scale(prior, direction)
    scales = log_scale(prior, origin)
    scale_slope = prior.scale_design @ scale_step
    penalty, penalty_gradient = _penalty_value(prior, log_smoothing, origin)
    penalty_slope = float(penalty_gradient @ direction)
    penalty_curvature = float(direction @ _penalty_matrix(prior, log_smoothing) @ direction)

    def values(steps: F64Array) -> F64Array:
        count = steps.shape[0]
        log_weights = density[None] + steps[:, None, None] * density_step[None]
        log_density = log_weights - _log_sum_exp(log_weights, axis=2, keepdims=True)
        total = -(penalty + steps * penalty_slope + 0.5 * np.square(steps) * penalty_curvature)
        for class_position, class_rows in enumerate(prior.class_rows):
            for rows in _row_chunks(class_rows, prior.grid_size * count, working_bytes):
                size = rows.shape[0] * count
                normalizers = _log_normalizers(
                    np.broadcast_to(log_density[None, :, class_position], (rows.shape[0], count, prior.grid_size)).reshape(size, prior.grid_size),
                    (scales[rows][:, None] + scale_slope[rows][:, None] * steps[None, :]).reshape(size),
                    prior.log_variance_grid, prior.kernel_floor, np.repeat(cavity.precision[rows], count), np.repeat(cavity.shift[rows], count),
                )
                total += normalizers.reshape(rows.shape[0], count).sum(axis=0)
        return total

    return values


def _line_log_integral(
    prior: ScaleMixturePrior, log_smoothing: F64Array, origin: F64Array, direction: F64Array, value: float, cavity: Cavity, working_bytes: int, share: float
) -> float:
    """log of the line integral of exp(F - P - value) along a standardized direction b (unit curvature at the
    maximum x), over its Laplace term sqrt(2 pi), to ``share`` in its log, by QUADPACK's adaptive rule over the
    whole line; its own error estimate must resolve the log to the share, or to half of double precision when
    rounding is what stopped it.

    Gauss-Hermite rules were tried first and refused: along the replaced directions the integrand falls off a cliff
    on one side, and consecutive rules agreed to the share at values up to 560 shares from the integral in over a
    tenth of the cases [sim-only, e2e fastline diagnostic], so no agreement of fixed rules certifies it here.
    """
    line = _line(prior, log_smoothing, origin, direction, cavity, working_bytes)

    def integrand(step: float) -> float:
        return float(np.exp(line(np.array([step]))[0] - value))

    integral, error, _information, *message = quad(
        integrand, -np.inf, np.inf, epsabs=0.0, epsrel=max(share, _QUADPACK_RELATIVE_FLOOR), full_output=True
    )
    if message and error > max(share, _HALF_PRECISION) * abs(integral):
        raise FloatingPointError(f"the exact integral along a direction did not converge: {message[0]}")
    return float(np.log(integral) - 0.5 * np.log(2.0 * np.pi))


def _laplace_corrections(
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
    # -H passed its Cholesky test, so the Schur complement is positive definite; an eigenvalue below eps times the
    # largest is rounding, raised to that floor as in ``_ascent_direction``.
    if eigenvalues.size:
        eigenvalues = np.maximum(eigenvalues, _EPSILON * float(np.max(np.abs(eigenvalues))))
    directions = moved @ eigenvectors / np.sqrt(eigenvalues)[None, :]
    third, fourth = _directional_derivatives(prior, evidence.coefficients, cavity, directions, working_bytes)
    terms = fourth / 8.0 + 5.0 * third**2 / 24.0
    corrections = np.zeros(directions.shape[1])
    order = np.argsort(-np.abs(terms))
    # Remaining sums from the smallest term up: the replaced set is the shortest prefix of ``order`` whose
    # complement sums to at most tolerance / 2.
    remaining = np.concatenate([np.cumsum(np.abs(terms[order])[::-1])[::-1], [0.0]])
    replaced = order[: int(np.argmax(remaining <= 0.5 * tolerance))]
    share = 0.5 * tolerance / max(replaced.shape[0], 1)
    for index in replaced:
        corrections[index] = _line_log_integral(prior, log_smoothing, evidence.coefficients, directions[:, index], value, cavity, working_bytes, share)
    return corrections, terms, directions


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
    return replace(evidence, value=evidence.laplace_value + float(np.sum(corrections)), error=evidence.error + remainder)


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
    """``_Profiled`` for M; raises LinAlgError when M is not positive definite."""
    basis = np.hstack([null_basis, complement])
    rotated = basis.T @ matrix @ basis
    factor = np.linalg.cholesky(0.5 * (rotated + rotated.T))
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
    prior: ScaleMixturePrior, log_smoothing: F64Array, start: F64Array, cavity: Cavity, correction: CurvatureCorrection, working_bytes: int, tolerance: float
) -> _Evidence | None:
    """V(rho) with the total curvature B, and the fixed-cavity gradient that steers the search; x_rho re-maximized
    from ``start``. None when x_rho is not a strict maximum of the objective V integrates, i.e. B + S is not
    positive definite there (lead ruling: such a point is never accepted, and its V never reported).

    V = F + 1/2 log|S|_+ - 1/2 log|B + S| + 1/2 log|N'(B + S)N|: B = -d2 log Z_EP / dx2 with EP re-solved, the
    second-order approximation of the actual marginal likelihood, taken as A + ``correction`` (``CurvatureCorrection``);
    the null space N is profiled. The gradient is that of the fixed-cavity form (-H in place of B), whose terms move with x_rho through
    dx/drho_i = -(-H)^-1 lambda_i S_i x and the third derivatives of log Z (``_curvature_trace_gradient``); it only
    chooses the search direction, and every step is accepted on V itself.
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
        coefficients, objective = _maximize_coefficients(prior, log_smoothing, coefficients, cavity, working_bytes, inner_tolerance)
        value, gradient, hessian = _penalized(prior, objective, log_smoothing, penalty, coefficients)
        try:
            fixed = _profiled_factor(-hessian, null_basis, complement)
        except np.linalg.LinAlgError:
            return None
        covariance, weight = fixed.inverse, fixed.weight
        newton_decrement = 0.5 * float(gradient @ covariance @ gradient)
        curvature_gradient = _curvature_trace_gradient(prior, coefficients, cavity, weight, working_bytes)
        sensitivity = max(float(curvature_gradient @ covariance @ curvature_gradient), np.finfo(np.float64).tiny)
        rounding = _EPSILON * (objective.magnitude + abs(value))
        # x-hat's error moves the determinant terms by at most this at first order, and F itself by at most the
        # decrement (the quadratic model's own gain); together, the inner maximizer's share of V's error.
        inner_error = 0.5 * float(np.sqrt(sensitivity * 2.0 * newton_decrement)) + newton_decrement
        if 0.5 * np.sqrt(sensitivity * 2.0 * newton_decrement) <= tolerance or newton_decrement <= rounding:
            break
        if previous is not None and np.array_equal(coefficients, previous):
            # The maximizer resolves x no further (its own rounding stop): V is as accurate as double precision gives.
            break
        previous = coefficients
        inner_tolerance = 2.0 * tolerance * tolerance / sensitivity
    penalty_log_determinant = sum(_log_pseudo_determinant(penalty[np.ix_(group, group)]) for group in _penalty_groups(prior))
    # B + S at x_rho: the fixed-cavity A + S there plus the EP-response part held at the fixed point; its bound below
    # takes the response as solved to a relative residual that moves 1/2 log|B + S| by at most D / 2 times it.
    response_tolerance = max(tolerance / coefficients.shape[0], _EPSILON)
    total = -hessian + correction.on(prior.coefficient_map)
    try:
        profiled_total = _profiled_factor(total, null_basis, complement)
    except np.linalg.LinAlgError:
        return None
    # V is certified to ``tolerance`` only where its determinant is resolved (``_Profiled.rounding``); where that
    # bound exceeds the tolerance (B + S near-singular off the profiled space), the point is not a certified maximum.
    if tolerance > 0.0 and 0.5 * profiled_total.rounding > tolerance:
        return None
    total_covariance = profiled_total.inverse
    evidence_value = value + 0.5 * penalty_log_determinant - 0.5 * profiled_total.schur_log_determinant
    # W = (-H)^-1 - N (N'(-H)N)^-1 N' carries both determinants' dependence on x (computed above).
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
            - 0.5 * lambda_weight * float(np.sum(weight[np.ix_(coordinates, coordinates)] * block.matrix))
            + 0.5 * float(curvature_gradient @ (covariance @ pull))
        )
    return _Evidence(
        value=evidence_value,
        laplace_value=evidence_value,
        # The inner maximizer, the determinant's rounding, and B solved to a relative residual that moves
        # 1/2 log|B + S| by at most D / 2 times it.
        error=inner_error + 0.5 * profiled_total.rounding + 0.5 * coefficients.shape[0] * response_tolerance,
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
    view = replace(prior, coefficient_map=prior.coefficient_map @ allowed, smoothing_blocks=tuple(blocks), null_basis=null_basis)
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
        gradient_change = gradient - trial.gradient
        curvature = float(step @ gradient_change)
        if curvature > 0.0:
            image = hessian @ step
            hessian = hessian - np.outer(image, image) / float(step @ image) + np.outer(gradient_change, gradient_change) / curvature
        weights, current = trial_weights, trial


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


def _same_basin(first: _Evidence, second: _Evidence) -> bool:
    """Whether two certified inner maxima are one: each point lies within sqrt(2 d) of its maximum in the -H metric
    (d its inner decrement), so one maximum is within the sum of the radii of both points; their V must then agree
    to within their certified errors."""
    step = first.coefficients - second.coefficients
    radius = np.sqrt(2.0 * first.inner_decrement) + np.sqrt(2.0 * second.inner_decrement)
    distance = np.sqrt(max(float(step @ first.precision @ step), 0.0))
    return distance <= radius and abs(first.laplace_value - second.laplace_value) <= first.error + second.error


def _best_certified(
    prior: ScaleMixturePrior, log_smoothing: F64Array, starts: Sequence[F64Array], cavity: Cavity, correction: CurvatureCorrection, working_bytes: int, tolerance: float
) -> _Evidence | None:
    """The certified inner maximum with the highest corrected V over the given starts; distinct basins are compared
    by their certified V (``_corrected``), and a start that lands in an already-found basin adds nothing.

    Two starts found one basin when their points are within the inner maximizer's own radii of each other,
    sqrt(2 d_a) + sqrt(2 d_b) in the -H metric, and their V agree to within their certified errors; the one with the
    smaller error stands for the basin, which is then corrected once."""
    certified: list[_Evidence] = []
    for start in starts:
        candidate = _evidence(prior, log_smoothing, start, cavity, correction, working_bytes, tolerance)
        if candidate is None:
            continue
        for position, other in enumerate(certified):
            if _same_basin(candidate, other):
                if candidate.error < other.error:
                    certified[position] = candidate
                break
        else:
            certified.append(candidate)
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
) -> tuple[F64Array, _Evidence] | None:
    """The best certified evidence of the model with ``infinite`` at lambda = infinity, the other weights at
    ``weights``; with its restriction basis."""
    view, allowed = _restricted_prior(prior, infinite)
    finite = [position for position in range(weights.shape[0]) if position not in infinite]
    evidence = _best_certified(view, weights[finite], [allowed.T @ start for start in starts], cavity, correction, working_bytes, tolerance)
    return None if evidence is None else (allowed, evidence)


def _maximize_evidence(
    prior: ScaleMixturePrior,
    start_weights: F64Array,
    start_coefficients: F64Array,
    cavity: Cavity, correction: CurvatureCorrection,
    working_bytes: int,
    bounds: list[tuple[float, float]],
    tolerance: float,
) -> tuple[F64Array, F64Array, _Evidence, _Evidence]:
    """Maximize V over every weight in (0, infinity]: the interior by the trust-region ascent inside the resolvable
    range, and the lambda = infinity edge evaluated exactly (x confined to the block's null space), never by fitting
    at an extreme weight. Once the interior ascent converges, every finite weight is compared with its infinity edge
    (lead ruling), and the best edge that raises V past the tolerance is taken; an edge weight moves back to the
    upper end of its range when V is higher there. There is no lambda = 0 edge (see the module docstring), so a
    start weight of -inf means its range's lower end. Every V is the best certified maximum over the warm, flat and
    global log-normal starts. Returns the log weights (+inf at an edge), x in full coordinates, V there, and V at
    the start.
    """
    lower = np.array([bound[0] for bound in bounds])
    upper = np.array([bound[1] for bound in bounds])
    infinite = frozenset(int(position) for position in np.flatnonzero(start_weights == np.inf))
    weights = np.where(start_weights == np.inf, upper, np.clip(start_weights, lower, upper))
    flat = initial_hyperparameters(prior).coefficients
    log_normal = _log_normal_start(prior, start_coefficients, cavity, working_bytes)
    coefficients = np.array(start_coefficients, dtype=np.float64, copy=True)
    first = _edge_evidence(prior, weights, infinite, [coefficients, flat, log_normal], cavity, correction, working_bytes, tolerance)
    if first is None:
        raise FloatingPointError("no structural start reaches a certified maximum at the starting penalty weights")
    start = first[1]
    best_corrected = -np.inf
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
            for position in sorted(edges):
                trial_infinite = infinite - {position}
                trial_weights = weights.copy()
                trial_weights[position] = upper[position]
                trial = _edge_evidence(prior, trial_weights, trial_infinite, [coefficients, flat, log_normal], cavity, correction, working_bytes, tolerance)
                if trial is not None and trial[1].value > current_value + tolerance:
                    infinite, weights, moved = frozenset(trial_infinite), trial_weights, True
                    coefficients = trial[0] @ trial[1].coefficients
                    break
        if not moved:
            log_smoothing = weights.copy()
            log_smoothing[sorted(infinite)] = np.inf
            return log_smoothing, coefficients, evidence, start


def _stationarity_check(
    view: ScaleMixturePrior,
    weights: F64Array,
    evidence: _Evidence,
    interior: F64Array,
    cavity: Cavity,
    correction: CurvatureCorrection,
    working_bytes: int,
    tolerance: float,
) -> tuple[F64Array, F64Array, F64Array, F64Array, tuple[F64Array, _Evidence] | None]:
    """The B-evidence's own gradient in each interior weight by one central difference: (gradient, curvature scale,
    step, error bound, better). ``better`` is a side whose certified V is above the base's by more than the
    tolerance, both taken at their certified bounds (its weights and evidence), found at once: the base is then not the maximum, only a point where its
    own inner maximum is ending (a fold), and the search resumes from the better side instead of certifying.

    Per eigen-direction of its block, V depends on rho_i through terms log(1 + e^(rho + a)) / 2 and
    b sigma(rho + a) / 2; the first derivatives are sigma / 2 and b sigma' / 2, and every higher derivative of the
    logistic is bounded by sigma itself. So s_i = (edf_i + lambda_i ||R_i x||^2) / 2 bounds |V''| and |V'''| in rho_i.

    With each side's V certified to e (``_evidence``'s tolerance, never below V's rounding), the central difference
    errs by at most E = h^2 s / 6 + e / h, least at h = (3 e / s)^(1/3), where E = (3^(2/3) / 2) s^(1/3) e^(2/3). The
    caller certifies the gain's upper bound 1/2 sum (|c| + E)^2 / s against ``tolerance``; e is set so that the error
    alone takes a quarter of it over the n interior weights, 1/2 E^2 / s = tolerance / (4 n):
    e = (2 tolerance / (n 3^(4/3)))^(3/4) s^(1/4). That plans the step; the bound recorded, and the one the h/2
    agreement below is tested against, use each side's own certified error (``_Evidence.error``: the inner maximizer,
    the determinant's rounding, B's linear response and the Tierney-Kadane remainder, summed), which can exceed e.

    The inner maxima of the penalized objective are not unique, so each side restarts from the base's x (not from
    the first-order predictor, which along a direction the data barely curve extrapolates far past the basin). The
    difference at h/2 must agree with the one at h within their two error bounds; otherwise a side reached another
    inner maximum (V there is a different function of rho), and the step halves, as it does while a side has no
    certified maximum.
    """
    gradient = np.zeros(weights.shape[0])
    rounding = _EPSILON * evidence.magnitude
    scale = np.maximum(0.5 * (evidence.effective_degrees + evidence.penalty_sizes), rounding)
    steps = np.zeros(weights.shape[0])
    errors = np.zeros(weights.shape[0])
    limit = _HALF_PRECISION * (1.0 + float(np.max(np.abs(weights), initial=0.0)))
    count = max(int(np.count_nonzero(interior)), 1)
    for position in np.flatnonzero(interior):
        unit = np.zeros(weights.shape[0])
        unit[position] = 1.0
        accuracy = max((2.0 * tolerance / (count * 3.0 ** (4.0 / 3.0))) ** 0.75 * scale[position] ** 0.25, rounding)
        step = (3.0 * accuracy / scale[position]) ** (1.0 / 3.0)

        def bound(length: float, sides: list[_Evidence]) -> float:
            # Truncation, and each side's own certified error over the difference's 2h.
            return length * length * scale[position] / 6.0 + (sides[0].error + sides[1].error) / (2.0 * length)

        while True:
            if step <= limit:
                raise FloatingPointError("the B-evidence has no certified maximum in one basin on both sides of a fitted penalty weight")
            quotients, bounds = [], []
            for length in (step, 0.5 * step):
                both = [
                    _corrected(
                        view, weights + side * length * unit,
                        _evidence(view, weights + side * length * unit, evidence.coefficients, cavity, correction, working_bytes, accuracy),
                        cavity, correction, working_bytes, accuracy,
                    )
                    for side in (-1.0, 1.0)
                ]
                for side, candidate in zip((-1.0, 1.0), both):
                    # Certified above: the side's lower bound beats the base's upper bound by the tolerance, so rounding
                    # cannot send the search back and forth between two basins.
                    if candidate is not None and candidate.value - candidate.error > evidence.value + evidence.error + tolerance:
                        return gradient, scale, steps, errors, (weights + side * length * unit, candidate)
                if any(side is None for side in both):
                    break
                quotients.append((both[1].value - both[0].value) / (2.0 * length))
                bounds.append(bound(length, both))
            if len(quotients) == 2 and abs(quotients[0] - quotients[1]) <= bounds[0] + bounds[1]:
                break
            step *= 0.5
        gradient[position] = quotients[0]
        steps[position] = step
        errors[position] = bounds[0]
    return gradient, scale, steps, errors, None


def hyper_step(
    prior: ScaleMixturePrior, hyperparameters: MixtureHyperparameters, cavity: Cavity, correction: CurvatureCorrection, working_bytes: int, tolerance: float
) -> HyperStep:
    """Maximize the B-evidence over every penalty weight in [0, infinity], with x at the penalized maximum for each, to
    ``tolerance`` nats: the resolution the fit certifies (1/(2K) for a scorer with K posterior draws).

    The search steers by the fixed-cavity gradient and accepts on V (with B). At the end, the B-evidence's own
    stationarity is checked by one central difference per interior weight (lead ruling); while the gain a Newton step
    on that difference predicts exceeds ``tolerance``, the search continues along it, accepting only steps that raise V.
    """
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
        prior, hyperparameters.log_smoothing, hyperparameters.coefficients, cavity, correction, working_bytes, bounds, tolerance
    )
    final_infinite = frozenset(int(position) for position in np.flatnonzero(log_smoothing == np.inf))
    final_view, final_allowed = _restricted_prior(prior, final_infinite)
    finite_final = np.isfinite(log_smoothing)
    weights = log_smoothing[finite_final]
    lower = np.array([bound[0] for bound in bounds])[finite_final]
    upper = np.array([bound[1] for bound in bounds])[finite_final]
    evidence = replace(evidence, coefficients=final_allowed.T @ coefficients)
    while True:
        interior = (weights > lower) & (weights < upper)
        check, curvature, check_steps, check_errors, better = _stationarity_check(
            final_view, weights, evidence, interior, cavity, correction, working_bytes, tolerance
        )
        gain_bound = 0.5 * float(np.sum(np.square(np.abs(check) + check_errors) / curvature))
        if better is None and gain_bound <= tolerance:
            break
        direction = check / curvature
        step_length, moved = 1.0, better
        while moved is None and step_length * float(np.max(np.abs(direction))) > _HALF_PRECISION * (1.0 + float(np.max(np.abs(weights)))):
            trial_weights = np.clip(weights + step_length * direction, lower, upper)
            trial = _certified_evidence(
                final_view, trial_weights, evidence.coefficients, cavity, correction, working_bytes,
                tolerance, final_allowed.T @ initial_hyperparameters(prior).coefficients,
            )
            if trial is not None and trial.value > evidence.value:
                moved = (trial_weights, trial)
                break
            step_length *= 0.5
        if moved is None:
            break
        # The search resumes from the moved point, edges and structural starts included. It keeps the move when the
        # resumed search ends lower (a basin chosen by its corrected V), so V rises at every pass and the loop ends.
        weights, evidence = moved
        resumed_smoothing = log_smoothing.copy()
        resumed_smoothing[finite_final] = weights
        resumed = _maximize_evidence(prior, resumed_smoothing, final_allowed @ evidence.coefficients, cavity, correction, working_bytes, bounds, tolerance)
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
    return HyperStep(
        hyperparameters=MixtureHyperparameters(coefficients=final_allowed @ evidence.coefficients, log_smoothing=log_smoothing),
        penalized_objective=evidence.penalized_value,
        evidence=evidence.value,
        newton_decrement=evidence.newton_decrement,
        smoothing_gradient=float(np.max(np.abs(check))) if check.size else 0.0,
        start_decrement=start_decrement,
        evidence_gain=evidence.value - start_evidence.value,
        stationarity_steps=check_steps,
        stationarity_errors=check_errors,
        stationarity_gain=gain_bound,
    )


# ------------------------------------------------------------------ the outer loop


@dataclass(frozen=True)
class FixedPoint:
    """A model's certified EP fixed point at a prior's hyperparameters: each variant's cavity, q's linear responses
    there (valid until the next fixed point is solved), q's mean, ``precision_norm(d)`` = d' Sigma^-1 d in q's
    posterior metric, and the effective number of effects p_eff = p - sum_j tau_j Sigma_jj."""

    cavity: Cavity
    posterior: GaussianPosterior
    mean: F64Array
    precision_norm: Callable[[F64Array], float]
    effective_effects: float


FixedPoints = Callable[[Sequence[MixtureHyperparameters]], Sequence["FixedPoint | None"]]
"""Each model's certified EP fixed point at its hyperparameters, warm from the previous call; None for a model where
none exists (EP reaches no proper cavities there)."""


@dataclass(frozen=True)
class OuterFit:
    """One model's certified empirical Bayes.

    ``remaining_gain`` is what the last check still found, in nats: ``newton_decrement``, 1/2 g'|B + S|^-1 g at the
    returned coefficients, plus the B-evidence gain the weights still had (``step.evidence_gain``); it is at most the
    tolerance. ``prediction_move`` is the posterior-mean move of the certifying Newton step in q's posterior metric,
    against ``prediction_tolerance`` = p_eff / K (K = 1 / (2 tolerance) draws). ``iterations`` counts accepted steps,
    ``halvings`` the trials refused (by the test, or for having no EP fixed point), and ``unresolved`` those of them
    that had no EP fixed point at all, so a loop that keeps refusing near its answer is visible in the certificate.
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
    # The decrement plus the weights' remaining gain at every outer evaluation, in order: the outer rate.
    history: tuple[float, ...]


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
    objective = _data_objective(prior, coefficients, cavity, working_bytes)
    return _penalized(prior, objective, weights, _penalty_matrix(prior, weights), coefficients)[1]


def _metric_decrement(newton: _NewtonB, gradient: F64Array) -> float:
    """1/2 g'(B + S)^-1 g in the step's own metric (B + S positive definite)."""
    components = newton.eigenvectors.T @ gradient
    return 0.5 * float(np.sum(components * components / newton.eigenvalues))


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
    """The step: Newton's (B + S)^-1 g where B + S is positive definite, else the maximizer of the quadratic model
    inside ``radius`` (More and Sorensen), which follows B + S's negative curvature out of a saddle."""
    if newton.definite:
        return newton.eigenvectors @ ((newton.eigenvectors.T @ newton.gradient) / newton.eigenvalues)
    return _trust_region_step(newton.total, newton.gradient, radius)


def fit_hyperparameters(
    prior: ScaleMixturePrior, starts: Sequence[MixtureHyperparameters], fixed_points: FixedPoints, working_bytes: int, tolerance: float
) -> list[OuterFit]:
    """Every model's empirical Bayes at its EP fixed point, certified to ``tolerance`` nats (lead ruling: Newton-B,
    never plain EP-EM).

    At an EP fixed point the EP evidence's x-gradient is the fixed-cavity g, and its negative Hessian is the total
    curvature B + S. Each outer step sets the weights by ``hyper_step`` at the current fixed point, then moves x on
    the quadratic model (g, B + S). The fixed-cavity maximizer (the EM step) solves with A + S instead: to first
    order it maps the error e to (I - (A + S)^-1 (B + S)) e, which diverges wherever that pencil has an eigenvalue
    above 2, and where A + S is indefinite the fixed-cavity objective has no maximum near x at all. On real LD at
    genome scale B + S itself is indefinite at the true prior (speed-floor [semi-real]), so the trust region below
    is the production case, not an edge case.

    - Where B + S is positive definite the step is Newton's, accepted by the natural monotonicity test (Deuflhard,
      Newton Methods for Nonlinear Problems, 2004, Section 3.1.4): at the trial's fixed point g'(B + S)^-1 g, taken
      with the step's own B + S, must fall, and otherwise the step halves. It needs no evidence value, which a
      full-data fixed point does not give, and it is invariant to x's coordinates.
    - Where B + S is indefinite (as at the true prior on real LD: speed-floor [semi-real]) the step maximizes the
      model inside a radius (More and Sorensen), and is accepted when the evidence rises along it. With no evidence
      value, the rise is the trapezoid rule of the path integral of the gradient, (g_x + g_trial)' s / 2, exact for
      a quadratic. A refused trial halves the radius; an accepted one that reached it doubles it. The radius starts
      at the length of the step on |B + S|.
    The loop stops when, for every model, B + S is positive definite, the Newton decrement plus the weights'
    remaining gain is at most ``tolerance`` (a saddle is never certified), and the Newton step then moves q's mean
    by at most p_eff / K in q's posterior metric (MODEL.md: the certificate includes the prediction change), taken
    at the step's own EP fixed point, not on the quadratic model. With K = 1 / (2 tolerance) posterior draws that is
    the scorer's own Monte Carlo resolution, as for the EP fixed point. Where the data barely identify a direction
    (the profiled null space at small n) the evidence can be flat to the tolerance while predictions still move; a
    step that fails the check is taken as an ordinary Newton trial. B's EP-response part is solved once per
    outer iterate (``curvature_correction``) and serves both the weights' evidence and the x step; where B + S has
    no certified maximum the weights wait while x steps.

    Each call of ``fixed_points`` passes every model's current hyperparameters, so on return its state is each
    model's certified fixed point.
    """
    count = len(starts)
    hyperparameters = list(starts)
    points = list(fixed_points(hyperparameters))
    if any(point is None for point in points):
        raise FloatingPointError("a starting point has no certified EP fixed point")
    fits: list[OuterFit | None] = [None] * count
    # (the model, its hyper step, the trial step, the radius, whether it certifies, the fraction of Newton's step)
    pending: list[tuple[_NewtonB, HyperStep | None, F64Array, float, bool, float] | None] = [None] * count
    radii: list[float | None] = [None] * count
    iterations, halvings, unresolved = [0] * count, [0] * count, [0] * count
    steps_taken: list[tuple[HyperStep, float] | None] = [None] * count
    histories: list[list[float]] = [[] for _model in range(count)]
    while True:
        for model in range(count):
            if fits[model] is not None or pending[model] is not None:
                continue
            point = points[model]
            correction = curvature_correction(prior, hyperparameters[model].coefficients, point.cavity, point.posterior, working_bytes, tolerance)
            try:
                step = hyper_step(prior, hyperparameters[model], point.cavity, correction, working_bytes, tolerance)
            except FloatingPointError:
                # B + S has no certified maximum here (an indefinite iterate): the weights wait, and x leaves the saddle.
                step = None
            log_smoothing = hyperparameters[model].log_smoothing if step is None else step.hyperparameters.log_smoothing
            newton = _newton_b(prior, log_smoothing, hyperparameters[model].coefficients, point, correction, working_bytes)
            remaining = newton.decrement + (np.inf if step is None else step.evidence_gain)
            histories[model].append(float(remaining))
            certifying = step is not None and remaining <= tolerance
            if certifying:
                steps_taken[model] = (step, remaining)
            radius = radii[model]
            if radius is None:
                magnitudes = np.maximum(np.abs(newton.eigenvalues), _EPSILON * float(np.max(np.abs(newton.eigenvalues))))
                radius = float(np.linalg.norm((newton.eigenvectors.T @ newton.gradient) / magnitudes))
            pending[model] = (newton, step, _proposal(newton, radius), radius, certifying, 1.0)
        if all(fit is not None for fit in fits):
            return [fit for fit in fits if fit is not None]
        trials = [hyperparameters[model] if entry is None else _trial(entry[0], entry[2]) for model, entry in enumerate(pending)]
        trial_points = list(fixed_points(trials))
        for model, entry in enumerate(pending):
            if entry is None:
                # An oracle that refuses refuses every model at once; a model without a trial keeps its point.
                if trial_points[model] is not None:
                    points[model] = trial_points[model]
                continue
            newton, step, proposal, radius, certifying, fraction = entry
            trial_point = trial_points[model]
            if trial_point is not None and certifying:
                current = points[model]
                # A shortened certifying step moves q's mean by its fraction, to first order: the full step's move is
                # its own over fraction^2 in the squared metric.
                move = current.precision_norm(trial_point.mean - current.mean) / (fraction * fraction)
                allowed_move = 2.0 * tolerance * current.effective_effects
                if move <= allowed_move:
                    certified_step, remaining = steps_taken[model]
                    fits[model] = OuterFit(
                        hyperparameters=hyperparameters[model], step=certified_step, newton_decrement=newton.decrement, remaining_gain=remaining,
                        prediction_move=move, prediction_tolerance=allowed_move, iterations=iterations[model], halvings=halvings[model],
                        unresolved=unresolved[model], history=tuple(histories[model]),
                    )
                    pending[model] = None
                    continue
            if trial_point is None:
                # No EP fixed point at the trial: refused, like a trial the test rejects, and counted.
                accepted = False
                unresolved[model] += 1
            else:
                gradient = _penalized_gradient(
                    newton.view, newton.log_smoothing[np.isfinite(newton.log_smoothing)], newton.origin + proposal, trial_point.cavity, working_bytes,
                )
                if newton.definite:
                    accepted = _metric_decrement(newton, gradient) < newton.decrement
                else:
                    accepted = 0.5 * float((newton.gradient + gradient) @ proposal) > 0.0
            length = float(np.linalg.norm(proposal))
            if accepted:
                hyperparameters[model], points[model], pending[model] = trials[model], trial_points[model], None
                iterations[model] += 1
                if not newton.definite:
                    radii[model] = 2.0 * radius if length >= radius * (1.0 - _HALF_PRECISION) else radius
                continue
            halvings[model] += 1
            if length <= _HALF_PRECISION * (1.0 + float(np.max(np.abs(newton.origin)))):
                raise FloatingPointError("the Newton-B step makes no certified progress at the EP fixed point")
            # A certifying step refused for having no fixed point stays certifying at half the length; one whose
            # move was too large, or an ordinary one, becomes an ordinary shorter trial.
            keep = certifying and trial_point is None
            if newton.definite:
                pending[model] = (newton, step, 0.5 * proposal, radius, keep, 0.5 * fraction)
            else:
                radius = 0.5 * length
                radii[model] = radius
                pending[model] = (newton, step, _proposal(newton, radius), radius, keep, 0.5 * fraction)
