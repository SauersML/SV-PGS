"""Dense reference for SV-PGS's EP-EB fixed point, for exactness tests of the fast stages.

This module is test infrastructure: it shares no code with the fitting stages,
so a stage that agrees with it has the model right, not just its own algebra.
It is dense (p x p inverses), so it is only for problems of a few hundred variants.

Model. The data enter as one Gaussian factor exp(-1/2 βᵀΛβ + ℓᵀβ): for a
quantitative trait in LD space Λ = κ nR and ℓ = κ X̃ᵀr; for a binary trait's
working model Λ = w̄ nR and ℓ is the working score. Each effect has the prior

    β_j ~ ∫ N(0, u_j e^t) g_c(j)(t) dt,   log u_j = o_j + d̃_jᵀθ,

a continuous Gaussian scale mixture over the log variance t, with no point mass at zero.
- g_c = exp(η + δ_c) / ∫exp(η + δ_c) over a range [t_lo, t_hi] is class c's mixing density,
  a continuous function of t learned nonparametrically: a shared log-density η plus a class
  deviation δ_c. Each carries the roughness ∫(φ‴)² dt with a learned weight
  (ROUGHNESS_ORDER, derived). Its null space is the quadratics, the normal density in log s:
  η's is profiled, and δ_c's (each class's location and width offset) is pooled across
  classes by a learned precision. A class with little signal leans on η. The null space's
  width → 0 and location → −∞ ends are models in their own right (a Gaussian effect prior;
  every effect null) and compete by V.
- The range is derived from the data (derived_log_variance_range): below it every kernel is
  flat to within the tolerance, above it every kernel decreases. fit_reference also certifies
  that doubling it moves the fit by less than the tolerance.
- o_j is the fixed measurement offset log r²_j: the prior is on the effect of the true
  genotype, so the observed column's prior variance carries r²_j with coefficient 1.
- d̃_j is the annotation row centred within its class. The class means are absorbed by g_c's
  location, so there is no separate class level to confound with it. θ is split into
  groups, each with a penalty matrix S_g (the identity for a discrete annotation, second
  differences for a smooth one) and a learned weight λ_g.

Computation, none of which is part of the model. η and δ_c are Chebyshev series on the
range, and every t-integral is composite Gauss–Legendre with an a-posteriori certificate.
fit_reference doubles the nodes per panel until every log Z_j is certified to tolerance/(p + 1),
doubles the degree until the evidence and the posterior moments change by less than the
tolerance, and doubles the range until that too moves them by less than the tolerance
(raising RangeNotConverged otherwise). So the fit does not depend on the grid or the range.

Inference (type-II maximum likelihood under EP).
- EP with one Gaussian site per effect, unclipped, with the tilted moments computed exactly
  on the quadrature: Newton on the moment-matching equations, falling back to the
  Opper–Winther double loop (a provably convergent minimization of the EP free energy)
  whenever Newton cannot lower the residual. Every cavity precision must be positive: the
  continuous prior has mass at every variance, so its tilted integral diverges otherwise.
- At fixed penalty weights, the coefficients x = (c, θ) maximize log Z_EP(x) − ½xᵀS_λx, by
  Newton with the total curvature B = −∇² log Z_EP (EP re-solved; the fixed-cavity curvature
  corrected by the cavity response, from implicit differentiation of the EP equations).
- The penalty weights maximize the Laplace evidence
      V(λ) = log Z_EP(x̂) − ½x̂ᵀS_λx̂ + ½log|RᵀS_λR| − ½log|Kᵀ(B + S_λ)K| + ½log|Nᵀ(B + S_λ)N|
  directly, with its exact gradient (which carries dx̂/dλ through D_xB). K is the allowed
  basis, R its penalized directions and N its unpenalized ones (η's null space, a smooth
  annotation's linear effect), which are profiled: a flat prior's integral over them is
  improper and diverges where the likelihood goes flat. Profiling gives the Schur form
  above (the profile's curvature), not the conditional block RᵀHR.
- Each weight ranges over (0, ∞]. V → −∞ as λ → 0 (the prior on the penalized directions
  becomes flat). At ∞ the term lies exactly in its penalty's null space (a quadratic
  log-density, no class offset, a discrete annotation's coefficient 0, a smooth annotation's
  linear effect), evaluated in closed form by that restriction, never by fitting at a huge
  λ. V is compared at ∞ and inside for every weight, and a weight sits at ∞ when V is highest
  there and the one-sided derivative of V in 1/λ is ≤ 0 (the score test, EdgeScore).

The fixed point is where EP, x and λ are all stationary (λ at its KKT point). It does not
depend on the path to it, which is what lets a stage be compared with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import chebyshev, legendre
from scipy import integrate
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr, logsumexp
from threadpoolctl import threadpool_limits

SITE_TOLERANCE = 1e-12
# Where rounding stops Newton: a solution this close is converged (the gates are 1e-6).
SITE_RESIDUAL_FLOOR = 1e-10
EDGE_SCORE_TOLERANCE = 1e-9
INITIAL_DEGREE = 8
COEFFICIENT_STEP_LIMIT = 2.0
MAXIMUM_RANGE_DOUBLINGS = 6
# The roughness penalty's order, derived rather than chosen (lead ruling after measurement):
# its null space, the quadratics in t (the normal density in log s), is the λ = ∞ limit of each
# log-density and must be a proper, range-invariant density. Measured: the log-normal limit
# moves by ≤ 1e-4 nats when the range is doubled or quadrupled, while order 2's power law moves
# by 4–53 and order 1's flat density by up to 100 (math-density). Under the profiled (Schur)
# evidence the fit is grid-invariant (prior lane: predictions within 0.46% for h from 0.5 to 0.1).
ROUGHNESS_ORDER = 3


def _chebyshev_columns(x: np.ndarray, degree: int, derivative: int = 0) -> np.ndarray:
    """T_1..T_degree at x (or their x-derivatives); T_0 is dropped because the softmax ignores it."""
    columns = np.empty((np.shape(x)[0], degree), dtype=np.result_type(x, np.float64))
    for order in range(1, degree + 1):
        series = np.zeros(order + 1)
        series[order] = 1.0
        columns[:, order - 1] = chebyshev.chebval(x, chebyshev.chebder(series, derivative))
    return columns


# Every kernel L_j(t) = (1 + ue^tP)^(-1/2) exp(½h²ue^t/(1 + ue^tP)) is analytic in the strip
# |Im t| < π/2, where Re(ue^t) ≥ 0 keeps |1 + ue^tP| ≥ 1 (its singularities sit at Im t = ±π).
# Each Gauss–Legendre panel's Bernstein ellipse is taken with that semi-minor axis. The
# panel length only trades panels against nodes per panel; the certificate decides accuracy.
STRIP_HALF_WIDTH = 0.5 * np.pi
MAXIMUM_PANEL_LENGTH = 2.0
INITIAL_NODES_PER_PANEL = 8


@dataclass(frozen=True)
class MixingQuadrature:
    """A continuous log-density φ(t) = Σ_m c_m T_m(x(t)) on [lower, upper], where the density
    lives, with its roughness matrix ∫(φ‴)² dt and a composite Gauss–Legendre rule on equal
    panels of length ≤ MAXIMUM_PANEL_LENGTH."""

    lower: float
    upper: float
    degree: int
    nodes_per_panel: int
    panel_edges: np.ndarray = field(init=False)
    nodes: np.ndarray = field(init=False)
    log_weights: np.ndarray = field(init=False)
    design: np.ndarray = field(init=False)
    roughness: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        half_length = 0.5 * (self.upper - self.lower)
        panel_count = int(np.ceil((self.upper - self.lower) / MAXIMUM_PANEL_LENGTH))
        edges = np.linspace(self.lower, self.upper, panel_count + 1)
        panel_half = 0.5 * (edges[1] - edges[0])
        standard_nodes, standard_weights = legendre.leggauss(self.nodes_per_panel)
        nodes = (0.5 * (edges[:-1] + edges[1:])[:, None] + panel_half * standard_nodes[None, :]).ravel()
        # ∫(φ⁽ᵐ⁾)² dt with m = ROUGHNESS_ORDER: a polynomial of degree 2(degree − m), so `degree`
        # Gauss–Legendre nodes integrate it exactly.
        roughness_nodes, roughness_weights = legendre.leggauss(max(self.degree, 2))
        derivative = _chebyshev_columns(roughness_nodes, self.degree, ROUGHNESS_ORDER) / half_length**ROUGHNESS_ORDER
        object.__setattr__(self, "panel_edges", edges)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "log_weights", np.tile(np.log(standard_weights * panel_half), panel_count))
        object.__setattr__(self, "design", _chebyshev_columns(self.standard_coordinate(nodes), self.degree))
        object.__setattr__(self, "roughness", derivative.T @ ((roughness_weights * half_length)[:, None] * derivative))

    @property
    def node_count(self) -> int:
        return int(self.nodes.shape[0])

    def standard_coordinate(self, log_variance):
        return (log_variance - self.lower) / (0.5 * (self.upper - self.lower)) - 1.0

    def log_density_at(self, coefficients: np.ndarray, log_variance: np.ndarray) -> np.ndarray:
        """φ(t) (unnormalized), with t held inside the range: only to carry a fit to another range."""
        inside = np.clip(log_variance, self.lower, self.upper)
        return _chebyshev_columns(self.standard_coordinate(inside), self.degree) @ coefficients


def mixing_quadrature(lower: float, upper: float, degree: int, nodes_per_panel: int = INITIAL_NODES_PER_PANEL) -> MixingQuadrature:
    return MixingQuadrature(lower=lower, upper=upper, degree=degree, nodes_per_panel=nodes_per_panel)


def difference_penalty(size: int, order: int) -> np.ndarray:
    difference = np.diff(np.eye(size), n=order, axis=0)
    return difference.T @ difference


def second_difference_penalty(size: int) -> np.ndarray:
    return difference_penalty(size, 2)


@dataclass(frozen=True)
class AnnotationGroup:
    columns: np.ndarray
    penalty: np.ndarray


@dataclass(frozen=True)
class ReferenceDesign:
    """Everything about the prior that the data do not fit."""

    class_index: np.ndarray
    log_variance_offset: np.ndarray
    annotation_design: np.ndarray
    annotation_groups: tuple[AnnotationGroup, ...]
    centred_design: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        centred = np.array(self.annotation_design, dtype=np.float64, copy=True)
        for class_position in range(self.class_count):
            members = self.class_index == class_position
            centred[members] -= centred[members].mean(axis=0)
        if centred.shape[1] and np.linalg.matrix_rank(centred) < centred.shape[1]:
            raise ValueError(
                "The class-centred annotation design must have full column rank: a smooth basis must "
                "drop its constant (the class means are the mixing density's location)."
            )
        object.__setattr__(self, "centred_design", centred)

    @property
    def class_count(self) -> int:
        return int(self.class_index.max()) + 1

    @property
    def feature_count(self) -> int:
        return int(self.annotation_design.shape[1])


@dataclass(frozen=True)
class ReferencePrior:
    """The design, the representation of the mixing densities, and ``tolerance``: the nats to which
    the fit is certified (its caller's resolution, 1/(2K) for a scorer with K posterior draws, as
    for the engine). The representation, the λ search's stationarity and the floor of every
    comparison of V are all certified to it."""

    design: ReferenceDesign
    mixing: MixingQuadrature
    tolerance: float

    @property
    def class_index(self) -> np.ndarray:
        return self.design.class_index

    @property
    def log_variance_offset(self) -> np.ndarray:
        return self.design.log_variance_offset

    @property
    def centred_design(self) -> np.ndarray:
        return self.design.centred_design

    @property
    def annotation_groups(self) -> tuple[AnnotationGroup, ...]:
        return self.design.annotation_groups

    @property
    def class_count(self) -> int:
        return self.design.class_count

    @property
    def feature_count(self) -> int:
        return self.design.feature_count

    @property
    def degree(self) -> int:
        return self.mixing.degree

    @property
    def coefficient_count(self) -> int:
        """The parameters: the shared log-density η, each class's deviation δ_c, and θ."""
        return (self.class_count + 1) * self.degree + self.feature_count

    @property
    def null_coefficient_count(self) -> int:
        """How many of T_1, T_2, ... span the roughness's null space (the quadratics: 2)."""
        return min(ROUGHNESS_ORDER - 1, self.degree)

    @property
    def class_coefficient_count(self) -> int:
        """Each class's log-density coefficients η + δ_c, and θ."""
        return self.class_count * self.degree + self.feature_count

    @property
    def pooling(self) -> np.ndarray:
        """d(η + δ_c for every c, θ)/d(η, δ, θ): the linear map from parameters to class coefficients."""
        degree = self.degree
        linear_map = np.zeros((self.class_coefficient_count, self.coefficient_count))
        for class_position in range(self.class_count):
            rows = slice(class_position * degree, (class_position + 1) * degree)
            linear_map[rows, :degree] = np.eye(degree)
            linear_map[rows, (class_position + 1) * degree : (class_position + 2) * degree] = np.eye(degree)
        linear_map[self.class_count * degree :, (self.class_count + 1) * degree :] = np.eye(self.feature_count)
        return linear_map


@dataclass(frozen=True)
class ReferenceHyperparameters:
    """The shared log-density η (M), the class deviations δ (C x M), θ, and the penalty weights.

    Class c's log mixing density has Chebyshev coefficients η + δ_c. The weights follow
    _penalty_terms: η's slope and curvature roughness, each δ_c's, then the annotation
    groups. Each lies in [0, ∞]; ∞ puts the term in its null space.
    """

    shared_coefficients: np.ndarray
    deviation_coefficients: np.ndarray
    annotation_coefficients: np.ndarray
    penalty_weights: np.ndarray

    @property
    def mixing_coefficients(self) -> np.ndarray:
        return self.shared_coefficients[None, :] + self.deviation_coefficients


@dataclass(frozen=True)
class ReferenceFit:
    prior: ReferencePrior
    posterior_mean: np.ndarray
    posterior_variance: np.ndarray
    hyperparameters: ReferenceHyperparameters
    evidence: float
    ep_log_evidence: float
    mixing_density: np.ndarray
    sites: "SiteState"
    cavity_precision: np.ndarray
    cavity_shift: np.ndarray
    newton_decrement: float


def coefficient_vector(hyperparameters: ReferenceHyperparameters) -> np.ndarray:
    return np.concatenate(
        [hyperparameters.shared_coefficients, hyperparameters.deviation_coefficients.ravel(), hyperparameters.annotation_coefficients]
    )


def _unpack(prior: ReferencePrior, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(class log-density coefficients η + δ_c, C x M; θ) from the parameter vector."""
    degree = prior.degree
    shared = vector[:degree]
    deviations = vector[degree : (prior.class_count + 1) * degree].reshape(prior.class_count, degree)
    return shared[None, :] + deviations, vector[(prior.class_count + 1) * degree :]


def _weights(hyperparameters: ReferenceHyperparameters) -> np.ndarray:
    return np.array(hyperparameters.penalty_weights, dtype=np.float64)


def _hyperparameters(prior: ReferencePrior, vector: np.ndarray, weights: np.ndarray) -> ReferenceHyperparameters:
    degree = prior.degree
    return ReferenceHyperparameters(
        shared_coefficients=vector[:degree].copy(),
        deviation_coefficients=vector[degree : (prior.class_count + 1) * degree].reshape(prior.class_count, degree).copy(),
        annotation_coefficients=vector[(prior.class_count + 1) * degree :].copy(),
        penalty_weights=np.array(weights, dtype=np.float64),
    )


def log_mixing_density(prior: ReferencePrior, mixing_coefficients: np.ndarray) -> np.ndarray:
    """log(quadrature weight x normalized density) at each node, per class."""
    log_weights = prior.mixing.log_weights[None, :] + mixing_coefficients @ prior.mixing.design.T
    return log_weights - logsumexp(log_weights, axis=1, keepdims=True)


def mixing_density(prior: ReferencePrior, mixing_coefficients: np.ndarray) -> np.ndarray:
    return np.exp(log_mixing_density(prior, mixing_coefficients))


def _log_component_normalizer(variance, precision, shift_square):
    """log of ∫N(β; 0, v) exp(−½Pβ² + hβ) dβ: −½ log(1 + vP) + ½ h² v / (1 + vP)."""
    relative = 1.0 + variance * precision
    return -0.5 * np.log(relative) + 0.5 * shift_square * variance / relative


def tilted_terms(
    prior: ReferencePrior,
    mixing_coefficients: np.ndarray,
    annotation_coefficients: np.ndarray,
    cavity_precision: np.ndarray,
    cavity_shift: np.ndarray,
) -> dict[str, np.ndarray]:
    """log Z_j, the tilted mean and variance, the node responsibilities, and scale derivatives.

    At node k the tilted density integrates to π_jk (1 + v_jk P_j)^(-1/2) exp(1/2 h_j² c_jk),
    with v_jk = u_j e^t_k and c_jk = v_jk / (1 + v_jk P_j); given k it is Gaussian with
    mean h_j c_jk and variance c_jk. With q = vP and r = 1/(1 + q), the node's log-factor
    has η-derivatives (η = log u_j)
        g   = ½ r (h² v r − q),
        g'  = ½ h² v r³ (1 − q) − ½ q r²,
        g'' = ½ h² v r⁴ (1 − 4q + q²) − ½ q r³ (1 − q).
    """
    log_scale = prior.log_variance_offset + prior.centred_design @ annotation_coefficients
    variance = np.exp(log_scale[:, None] + prior.mixing.nodes[None, :])
    precision = cavity_precision[:, None]
    shift_square = np.square(cavity_shift)[:, None]
    relative = 1.0 + variance * precision
    conditional_variance = variance / relative
    log_density = log_mixing_density(prior, mixing_coefficients)[prior.class_index]
    log_component = log_density + _log_component_normalizer(variance, precision, shift_square)
    log_normalizer = logsumexp(log_component, axis=1)
    responsibility = np.exp(log_component - log_normalizer[:, None])
    tilted_mean = cavity_shift * np.sum(responsibility * conditional_variance, axis=1)
    tilted_second_moment = np.sum(responsibility * (conditional_variance + shift_square * np.square(conditional_variance)), axis=1)
    retained = 1.0 / relative
    ratio = variance * precision
    derivative = 0.5 * retained * (shift_square * variance * retained - ratio)
    curvature = 0.5 * shift_square * variance * retained**3 * (1.0 - ratio) - 0.5 * ratio * retained**2
    third = 0.5 * shift_square * variance * retained**4 * (1.0 - 4.0 * ratio + ratio**2) - 0.5 * ratio * retained**3 * (1.0 - ratio)
    scale_derivative = np.sum(responsibility * derivative, axis=1)
    return {
        "log_normalizer": log_normalizer,
        "responsibility": responsibility,
        "tilted_mean": tilted_mean,
        "tilted_variance": tilted_second_moment - np.square(tilted_mean),
        "scale_derivative": scale_derivative,
        "component_scale_derivative": derivative,
        "component_scale_curvature": curvature,
        "component_scale_third": third,
        "scale_curvature": np.sum(responsibility * (np.square(derivative - scale_derivative[:, None]) + curvature), axis=1),
    }


def cavity_log_marginal(prior, vector, cavity_precision, cavity_shift) -> tuple[float, np.ndarray]:
    """f = Σ_j log Z_j at fixed cavities, and its gradient in the parameters (η, δ, θ)."""
    mixing_coefficients, annotation_coefficients = _unpack(prior, vector)
    terms = tilted_terms(prior, mixing_coefficients, annotation_coefficients, cavity_precision, cavity_shift)
    density = mixing_density(prior, mixing_coefficients)
    # ∂/∂(log weight at node k) of Σ_{j in c} log Σ_k π_ck Z_jk is Σ_{j in c} (w_jk − π_ck).
    responsibility_sums = np.zeros((prior.class_count, prior.mixing.node_count))
    np.add.at(responsibility_sums, prior.class_index, terms["responsibility"])
    class_sizes = np.bincount(prior.class_index, minlength=prior.class_count).astype(np.float64)
    mixing_gradient = (responsibility_sums - class_sizes[:, None] * density) @ prior.mixing.design
    annotation_gradient = prior.centred_design.T @ terms["scale_derivative"]
    class_gradient = np.concatenate([mixing_gradient.ravel(), annotation_gradient])
    return float(np.sum(terms["log_normalizer"])), prior.pooling.T @ class_gradient


def cavity_log_marginal_hessian(prior, vector, cavity_precision, cavity_shift) -> np.ndarray:
    """∇²f in the parameters (η, δ, θ), assembled in the class coefficients (η + δ_c, θ).

    For variant j of class c, with node responsibilities w_j, node derivatives g_jk and
    their mean ḡ_j: ∂²/∂φ² = diag(w_j) − w_jw_jᵀ − (diag π_c − π_cπ_cᵀ),
    ∂²/∂η∂φ_k = w_jk (g_jk − ḡ_j), and ∂²/∂η² = Var_w(g_j) + E_w[∂g_j/∂η].
    """
    mixing_coefficients, annotation_coefficients = _unpack(prior, vector)
    terms = tilted_terms(prior, mixing_coefficients, annotation_coefficients, cavity_precision, cavity_shift)
    density = mixing_density(prior, mixing_coefficients)
    design_nodes = prior.mixing.design
    block = prior.degree
    mixing_size = prior.class_count * block
    hessian = np.zeros((prior.class_coefficient_count, prior.class_coefficient_count))
    responsibility = terms["responsibility"]
    covariance_weight = responsibility * (terms["component_scale_derivative"] - terms["scale_derivative"][:, None])
    design = prior.centred_design
    for class_position in range(prior.class_count):
        members = prior.class_index == class_position
        member_weights = responsibility[members]
        data_part = np.diag(member_weights.sum(axis=0)) - member_weights.T @ member_weights
        prior_part = members.sum() * (np.diag(density[class_position]) - np.outer(density[class_position], density[class_position]))
        span = slice(class_position * block, (class_position + 1) * block)
        hessian[span, span] = design_nodes.T @ (data_part - prior_part) @ design_nodes
        cross = design_nodes.T @ covariance_weight[members].T @ design[members]
        hessian[span, mixing_size:] = cross
        hessian[mixing_size:, span] = cross.T
    hessian[mixing_size:, mixing_size:] = design.T @ (terms["scale_curvature"][:, None] * design)
    pooling = prior.pooling
    return pooling.T @ hessian @ pooling


def _penalty_terms(prior: ReferencePrior) -> list[tuple[np.ndarray, np.ndarray]]:
    """(columns, matrix) of every penalty term, in the order of the weights.

    - η: its roughness ∫(η‴)² over all its coefficients. The null space, T_1 and T_2 (the
      shared density's location and width), is unpenalized and so profiled.
    - Each δ_c: its roughness on its coefficients T_3.. beyond that null space.
    - The T_1 and T_2 coordinates of every δ_c together (each class's location and width
      offset from η), under one learned precision: the classes are pooled around η.
    - The annotation groups.

    Every term acts on its own columns, so each block has one penalty.
    """
    degree = prior.degree
    null = prior.null_coefficient_count
    terms = [(np.arange(degree), prior.mixing.roughness)]
    for class_position in range(prior.class_count):
        start = (class_position + 1) * degree
        terms.append((np.arange(start + null, start + degree), prior.mixing.roughness[null:, null:]))
    level_columns = np.concatenate([np.arange((position + 1) * degree, (position + 1) * degree + null) for position in range(prior.class_count)])
    terms.append((level_columns, np.eye(level_columns.size)))
    offset = (prior.class_count + 1) * degree
    terms += [(offset + group.columns, group.penalty) for group in prior.annotation_groups]
    return terms


def penalty_count(prior: ReferencePrior) -> int:
    return prior.class_count + 2 + len(prior.annotation_groups)


def _embedded(prior: ReferencePrior, columns: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    full = np.zeros((prior.coefficient_count, prior.coefficient_count))
    full[np.ix_(columns, columns)] = matrix
    return full


def _null_space(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return eigenvectors[:, eigenvalues <= 1e-10 * max(float(eigenvalues.max()), 1.0)]


def _blocks(prior: ReferencePrior) -> list[tuple[np.ndarray, list[int]]]:
    """The coefficient blocks and the positions of the penalty terms acting on each."""
    blocks: list[tuple[np.ndarray, list[int]]] = []
    for position, (columns, _matrix) in enumerate(_penalty_terms(prior)):
        for block_columns, positions in blocks:
            if np.array_equal(block_columns, columns):
                positions.append(position)
                break
        else:
            blocks.append((columns, [position]))
    return [(columns, positions) for columns, positions in blocks if columns.size]


def _penalizing(weight: float) -> bool:
    """A term penalizes through S_λ when its weight is finite and positive (∞ acts through the basis)."""
    return bool(0.0 < weight < np.inf)


def allowed_basis(prior: ReferencePrior, weights: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the coefficients allowed when every infinite weight's term is in its null space."""
    terms = _penalty_terms(prior)
    columns_used = np.zeros(prior.coefficient_count, dtype=bool)
    pieces = []
    for columns, positions in _blocks(prior):
        infinite = [terms[position][1] for position in positions if np.isinf(weights[position])]
        basis = _null_space(sum(infinite)) if infinite else np.eye(columns.size)
        embedded = np.zeros((prior.coefficient_count, basis.shape[1]))
        embedded[columns] = basis
        pieces.append(embedded)
        columns_used[columns] = True
    pieces.append(np.eye(prior.coefficient_count)[:, ~columns_used])
    return np.hstack(pieces)


def integrated_basis(prior: ReferencePrior, weights: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """The directions the Laplace evidence integrates: the penalized ones within the allowed basis.

    Every unpenalized direction (a smooth annotation's linear effect) is profiled: maximized,
    not integrated. Under a flat prior its integral is improper, diverges wherever the
    likelihood goes flat along it, and leaves models with different flat dimensions
    incomparable.
    """
    if basis.shape[1] == 0:
        return basis
    unweighted = np.zeros((basis.shape[1], basis.shape[1]))
    for (columns, matrix), weight in zip(_penalty_terms(prior), weights):
        if _penalizing(weight) and columns.size:
            unweighted += basis.T @ _embedded(prior, columns, matrix) @ basis
    eigenvalues, eigenvectors = np.linalg.eigh(unweighted)
    return basis @ eigenvectors[:, eigenvalues > 1e-10 * max(float(np.max(eigenvalues, initial=0.0)), 1.0)]


def penalty_matrix(prior: ReferencePrior, weights: np.ndarray) -> np.ndarray:
    """S_λ from the finite weights (an infinite weight acts through allowed_basis instead)."""
    penalty = np.zeros((prior.coefficient_count, prior.coefficient_count))
    for (columns, matrix), weight in zip(_penalty_terms(prior), weights):
        if _penalizing(weight):
            penalty[np.ix_(columns, columns)] += weight * matrix
    return penalty


def _penalty_log_determinant(prior, weights, basis) -> tuple[float, int, np.ndarray]:
    """log|BᵀS_λB|₊, its rank, and its pseudo-inverse, with the rank taken from the unweighted terms."""
    restricted = basis.T @ penalty_matrix(prior, weights) @ basis
    if basis.shape[1] == 0:
        return 0.0, 0, np.zeros((0, 0))
    unweighted = sum(
        basis.T @ _embedded(prior, columns, matrix) @ basis
        for (columns, matrix), weight in zip(_penalty_terms(prior), weights)
        if _penalizing(weight)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(unweighted)
    penalized = eigenvectors[:, eigenvalues > 1e-10 * max(float(eigenvalues.max()), 1.0)]
    reduced = penalized.T @ restricted @ penalized
    factor = cho_factor(reduced)
    log_determinant = 2.0 * float(np.sum(np.log(np.diag(factor[0]))))
    pseudo_inverse = penalized @ cho_solve(factor, penalized.T)
    return log_determinant, penalized.shape[1], pseudo_inverse


def tilted_power_moments(prior, vector, cavity_precision, cavity_shift) -> dict[str, np.ndarray]:
    """log Z_j and the first four moments of each variant's tilted density prior_j(β) exp(−½Pβ² + hβ).

    Given node k the tilted density is N(h c_jk, c_jk), c_jk = v_jk / (1 + v_jk P_j).
    """
    mixing_coefficients, annotation_coefficients = _unpack(prior, vector)
    log_scale = prior.log_variance_offset + prior.centred_design @ annotation_coefficients
    variance = np.exp(log_scale[:, None] + prior.mixing.nodes[None, :])
    precision = cavity_precision[:, None]
    shift = cavity_shift[:, None]
    conditional_variance = variance / (1.0 + variance * precision)
    log_density = log_mixing_density(prior, mixing_coefficients)[prior.class_index]
    log_component = log_density + _log_component_normalizer(variance, precision, np.square(shift))
    log_normalizer = logsumexp(log_component, axis=1)
    weights = np.exp(log_component - log_normalizer[:, None])
    mean = shift * conditional_variance
    return {
        "log_normalizer": log_normalizer,
        "first": np.sum(weights * mean, axis=1),
        "second": np.sum(weights * (conditional_variance + mean**2), axis=1),
        "third": np.sum(weights * (mean**3 + 3.0 * mean * conditional_variance), axis=1),
        "fourth": np.sum(weights * (mean**4 + 6.0 * mean**2 * conditional_variance + 3.0 * conditional_variance**2), axis=1),
    }


def tilted_moment_gradients(prior, vector, cavity_precision, cavity_shift) -> tuple[np.ndarray, np.ndarray]:
    """∂E_r[β]/∂x and ∂E_r[β²]/∂x (p x dim) at fixed cavities.

    Node k's tilted component is N(h c_k, c_k). The responsibilities move as
    ∂w_k/∂(log weight l) = w_k(δ_kl − w_l) and ∂w_k/∂η = w_k(g_k − ḡ), and ∂c_k/∂η = c_k r_k
    with r_k = 1/(1 + v_kP).
    """
    mixing_coefficients, annotation_coefficients = _unpack(prior, vector)
    terms = tilted_terms(prior, mixing_coefficients, annotation_coefficients, cavity_precision, cavity_shift)
    log_scale = prior.log_variance_offset + prior.centred_design @ annotation_coefficients
    variance = np.exp(log_scale[:, None] + prior.mixing.nodes[None, :])
    retained = 1.0 / (1.0 + variance * cavity_precision[:, None])
    conditional_variance = variance * retained
    shift = cavity_shift[:, None]
    weights = terms["responsibility"]
    centred_derivative = terms["component_scale_derivative"] - terms["scale_derivative"][:, None]
    block = prior.degree
    mixing_size = prior.class_count * block
    gradients = []
    for node_moment, node_derivative in (
        (shift * conditional_variance, shift * conditional_variance * retained),
        (conditional_variance + shift**2 * conditional_variance**2, conditional_variance * retained + 2.0 * shift**2 * conditional_variance**2 * retained),
    ):
        moment = np.sum(weights * node_moment, axis=1)
        gradient = np.zeros((cavity_precision.shape[0], prior.class_coefficient_count))
        weight_part = weights * (node_moment - moment[:, None]) @ prior.mixing.design
        for class_position in range(prior.class_count):
            members = prior.class_index == class_position
            gradient[np.ix_(members, np.arange(class_position * block, (class_position + 1) * block))] = weight_part[members]
        scale_part = np.sum(weights * (centred_derivative * node_moment + node_derivative), axis=1)
        gradient[:, mixing_size:] = scale_part[:, None] * prior.centred_design
        gradients.append(gradient @ prior.pooling)
    return gradients[0], gradients[1]


@dataclass(frozen=True)
class SiteState:
    """An EP solution: the sites (τ, ν), the Gaussian approximation, the cavities, and log Z_EP."""

    site_precision: np.ndarray
    site_shift: np.ndarray
    covariance: np.ndarray
    posterior_mean: np.ndarray
    cavity_precision: np.ndarray
    cavity_shift: np.ndarray
    moments: dict
    residual: float
    log_evidence: float

    @property
    def posterior_variance(self) -> np.ndarray:
        return np.diag(self.covariance).copy()


class EPFailure(ArithmeticError):
    """EP found no stationary point from this start (a trial point far outside where the fit lives)."""


class NoInteriorOptimum(ArithmeticError):
    """At these penalty weights log Z_EP − ½xᵀS_λx has no strict interior maximum: the coefficients
    run along a ray (a density collapsing below resolution, or two atoms at the range's ends)."""


def site_state(prior, vector, likelihood_precision, linear_term, site_precision, site_shift) -> SiteState | None:
    """The Gaussian approximation Λ + diag τ, ℓ + ν at these sites, its cavities and tilted moments,
    the scaled moment-matching residual, and log Z_EP = log Z_q + log Z_r − log Z_s.

    None outside the domain: Λ + diag τ not positive definite, or a cavity precision ≤ 0
    (the continuous prior has mass at every variance, so the tilted integral then diverges).
    """
    if not (np.all(np.isfinite(site_precision)) and np.all(np.isfinite(site_shift)) and np.all(np.isfinite(vector))):
        return None
    try:
        factor = cho_factor(likelihood_precision + np.diag(site_precision))
    except np.linalg.LinAlgError:
        return None
    covariance = cho_solve(factor, np.eye(linear_term.shape[0]))
    covariance = 0.5 * (covariance + covariance.T)
    shift = linear_term + site_shift
    mean = covariance @ shift
    variance = np.diag(covariance)
    cavity_precision = 1.0 / variance - site_precision
    if not np.all(cavity_precision > 0.0):
        return None
    cavity_shift = mean / variance - site_shift
    with np.errstate(over="ignore", invalid="ignore"):
        moments = tilted_power_moments(prior, vector, cavity_precision, cavity_shift)
    if not all(np.all(np.isfinite(value)) for value in moments.values()):
        return None
    residual = max(
        float(np.max(np.abs(mean - moments["first"]) / np.sqrt(variance))),
        float(np.max(np.abs(variance + mean**2 - moments["second"]) / (variance + mean**2))),
    )
    log_evidence = (
        0.5 * float(shift @ mean)
        - float(np.sum(np.log(np.diag(factor[0]))))
        + float(np.sum(moments["log_normalizer"]))
        - float(np.sum(0.5 * mean**2 / variance + 0.5 * np.log(variance)))
    )
    return SiteState(
        site_precision=np.array(site_precision, dtype=np.float64),
        site_shift=np.array(site_shift, dtype=np.float64),
        covariance=covariance,
        posterior_mean=mean,
        cavity_precision=cavity_precision,
        cavity_shift=cavity_shift,
        moments=moments,
        residual=residual,
        log_evidence=float(log_evidence),
    )


def site_jacobian(state: SiteState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """dR/ds of the moment residual R = (m − E_r[β], Σ_jj + m² − E_r[β²]) in s = (ν, τ), and
    the cavity derivatives dP/ds and dh/ds.

    With m = Σ(ℓ + ν), Σ = (Λ + diag τ)⁻¹, P_j = 1/Σ_jj − τ_j and h_j = m_j/Σ_jj − ν_j:
    ∂m/∂ν = Σ, ∂m_j/∂τ_k = −Σ_jk m_k, ∂Σ_jj/∂τ_k = −Σ_jk²; the tilted moments move with the
    cavity as ∂E[β^a]/∂h = Cov(β^a, β) and ∂E[β^a]/∂P = −½ Cov(β^a, β²).
    """
    covariance = state.covariance
    mean = state.posterior_mean
    variance = np.diag(covariance)
    moments = state.moments
    size = mean.shape[0]
    identity = np.eye(size)
    squared = covariance**2
    precision_by_precision = squared / variance[:, None] ** 2 - identity
    shift_by_shift = covariance / variance[:, None] - identity
    shift_by_precision = -covariance * mean[None, :] / variance[:, None] + mean[:, None] * squared / variance[:, None] ** 2
    first_by_shift = moments["second"] - moments["first"] ** 2
    first_by_precision = -0.5 * (moments["third"] - moments["first"] * moments["second"])
    second_by_shift = moments["third"] - moments["second"] * moments["first"]
    second_by_precision = -0.5 * (moments["fourth"] - moments["second"] ** 2)
    jacobian = np.empty((2 * size, 2 * size))
    jacobian[:size, :size] = covariance - first_by_shift[:, None] * shift_by_shift
    jacobian[:size, size:] = -covariance * mean[None, :] - first_by_precision[:, None] * precision_by_precision - first_by_shift[:, None] * shift_by_precision
    jacobian[size:, :size] = 2.0 * mean[:, None] * covariance - second_by_shift[:, None] * shift_by_shift
    jacobian[size:, size:] = (
        -squared
        - 2.0 * mean[:, None] * covariance * mean[None, :]
        - second_by_precision[:, None] * precision_by_precision
        - second_by_shift[:, None] * shift_by_precision
    )
    cavity_precision_by_sites = np.hstack([np.zeros((size, size)), precision_by_precision])
    cavity_shift_by_sites = np.hstack([shift_by_shift, shift_by_precision])
    return jacobian, cavity_precision_by_sites, cavity_shift_by_sites


def _double_loop_objective(prior, vector, likelihood_precision, linear_term, marginal_precision, marginal_shift, site_precision, site_shift):
    """Φ(τ, ν) = log Z_q(τ, ν) + log Z_r(P_s − τ, h_s − ν), its gradient and Hessian in (ν, τ).

    Z_q = ∫exp(−½βᵀ(Λ + diag τ)β + (ℓ + ν)ᵀβ) dβ is the Gaussian approximation and Z_r the
    product of tilted normalizers at the cavities; the statistics are (β_j, −½β_j²).
    Returns None outside the domain.
    """
    cavity_precision = marginal_precision - site_precision
    if not np.all(cavity_precision > 0.0):
        return None
    try:
        factor = cho_factor(likelihood_precision + np.diag(site_precision))
    except np.linalg.LinAlgError:
        return None
    covariance = cho_solve(factor, np.eye(linear_term.shape[0]))
    covariance = 0.5 * (covariance + covariance.T)
    shift = linear_term + site_shift
    mean = covariance @ shift
    with np.errstate(over="ignore", invalid="ignore"):
        moments = tilted_power_moments(prior, vector, cavity_precision, marginal_shift - site_shift)
    if not all(np.all(np.isfinite(value)) for value in moments.values()):
        return None
    value = 0.5 * float(shift @ mean) - float(np.sum(np.log(np.diag(factor[0])))) + float(np.sum(moments["log_normalizer"]))
    second = np.diag(covariance) + mean**2
    gradient = np.concatenate([mean - moments["first"], -0.5 * (second - moments["second"])])
    size = linear_term.shape[0]
    hessian = np.empty((2 * size, 2 * size))
    hessian[:size, :size] = covariance + np.diag(moments["second"] - moments["first"] ** 2)
    cross = -covariance * mean[None, :] + np.diag(-0.5 * (moments["third"] - moments["first"] * moments["second"]))
    hessian[:size, size:] = cross
    hessian[size:, :size] = cross.T
    hessian[size:, size:] = 0.5 * covariance**2 + np.outer(mean, mean) * covariance + np.diag(0.25 * (moments["fourth"] - moments["second"] ** 2))
    return value, gradient, hessian, mean, covariance


def double_loop_sites(prior, vector, likelihood_precision, linear_term, start: SiteState, target_residual: float = SITE_TOLERANCE) -> SiteState:
    """EP by the Opper–Winther double loop, which provably reaches a stationary point, run until
    the moment residual is below ``target_residual``.

    The EP (expectation-consistent) free energy is −log Z_q − log Z_r + log Z_s, with Z_s
    the normalizer of the diagonal Gaussian with natural parameters (P_s, h_s). The outer
    loop fixes (P_s, h_s) at the current posterior marginals, which bounds the concave part
    linearly; the inner problem, the minimum over the sites of the convex Φ = log Z_q +
    log Z_r, is solved by Newton. Each outer step lowers the free energy.
    """
    site_precision = np.array(start.site_precision, dtype=np.float64, copy=True)
    site_shift = np.array(start.site_shift, dtype=np.float64, copy=True)
    variance = np.diag(start.covariance)
    marginal_precision = 1.0 / variance
    marginal_shift = start.posterior_mean / variance
    size = linear_term.shape[0]
    for _outer in range(100_000):
        evaluation = _double_loop_objective(prior, vector, likelihood_precision, linear_term, marginal_precision, marginal_shift, site_precision, site_shift)
        if evaluation is None:
            raise EPFailure("The double loop's start lies outside the EP domain.")
        for _newton in range(100):
            value, gradient, hessian, mean, covariance = evaluation
            step = -np.linalg.solve(hessian, gradient)
            decrease = float(gradient @ step)
            if -decrease < 1e-12 * (1.0 + abs(value)):
                break
            for _halving in range(80):
                candidate = _double_loop_objective(
                    prior, vector, likelihood_precision, linear_term, marginal_precision, marginal_shift,
                    site_precision + step[size:], site_shift + step[:size],
                )
                if candidate is not None and candidate[0] <= value + 1e-4 * decrease:
                    break
                step *= 0.5
                decrease *= 0.5
            else:
                break
            site_precision = site_precision + step[size:]
            site_shift = site_shift + step[:size]
            evaluation = candidate
        _value, _gradient, _hessian, mean, covariance = evaluation
        new_precision = 1.0 / np.diag(covariance)
        new_shift = mean * new_precision
        change = max(
            float(np.max(np.abs(new_precision / marginal_precision - 1.0))),
            float(np.max(np.abs(new_shift - marginal_shift) / np.sqrt(new_precision))),
        )
        marginal_precision, marginal_shift = new_precision, new_shift
        state = site_state(prior, vector, likelihood_precision, linear_term, site_precision, site_shift)
        if state is None:
            raise EPFailure("The double loop left the EP domain.")
        # At its stationary point the marginals stop moving and the moments match.
        if state.residual < target_residual or change < SITE_TOLERANCE:
            return state
    raise EPFailure("The EP double loop did not converge.")


def _sequential_sweep(prior, vector, likelihood_precision, linear_term, state: SiteState) -> SiteState | None:
    """One sequential EP sweep: each site in turn set to match its variant's tilted moments at its
    current cavity, the posterior updated by a rank-one step. The update is halved while it would
    leave Λ + diag τ indefinite or any cavity improper (a numerical guard: it never moves the
    fixed point). Sequential updates settle strongly coupled (LD-tied) sites where a
    parallel sweep or a Newton step on all sites at once is only valid in a tiny region. None if
    the swept sites leave the domain."""
    mixing_coefficients, annotation_coefficients = _unpack(prior, vector)
    log_scale = prior.log_variance_offset + prior.centred_design @ annotation_coefficients
    log_density = log_mixing_density(prior, mixing_coefficients)[prior.class_index]
    site_precision = state.site_precision.copy()
    site_shift = state.site_shift.copy()
    covariance = state.covariance.copy()
    mean = state.posterior_mean.copy()
    for variant in range(linear_term.shape[0]):
        marginal = covariance[variant, variant]
        cavity_precision = 1.0 / marginal - site_precision[variant]
        cavity_shift = mean[variant] / marginal - site_shift[variant]
        if not cavity_precision > 0.0:
            return None
        variance = np.exp(log_scale[variant] + prior.mixing.nodes)
        conditional = variance / (1.0 + variance * cavity_precision)
        log_component = log_density[variant] + _log_component_normalizer(variance, cavity_precision, cavity_shift**2)
        weights = np.exp(log_component - logsumexp(log_component))
        tilted_mean = cavity_shift * float(weights @ conditional)
        tilted_variance = float(weights @ (conditional + (cavity_shift * conditional) ** 2)) - tilted_mean**2
        precision_step = 1.0 / tilted_variance - cavity_precision - site_precision[variant]
        shift_step = tilted_mean / tilted_variance - cavity_shift - site_shift[variant]
        column = covariance[:, variant].copy()
        diagonal = np.diag(covariance)
        for _halving in range(60):
            denominator = 1.0 + precision_step * marginal
            if denominator > 0.0:
                # Every cavity after the rank-one step: 1/Σ'_kk − τ'_k, with Σ'_kk = Σ_kk − Σ_kj²Δτ/(1 + Δτ Σ_jj).
                updated = diagonal - column**2 * (precision_step / denominator)
                sites_after = site_precision.copy()
                sites_after[variant] += precision_step
                if np.all(updated > 0.0) and np.all(1.0 / updated - sites_after > 0.0):
                    break
            precision_step *= 0.5
            shift_step *= 0.5
        else:
            return None
        mean = mean + column * (shift_step - precision_step * mean[variant]) / denominator
        covariance = covariance - np.outer(column, column) * (precision_step / denominator)
        site_precision[variant] += precision_step
        site_shift[variant] += shift_step
    return site_state(prior, vector, likelihood_precision, linear_term, site_precision, site_shift)


def solve_sites(prior, vector, likelihood_precision, linear_term, start: SiteState) -> SiteState:
    """EP at fixed hyperparameters: Newton on the moment-matching equations R(s) = 0 from the start.

    Where a Newton step (halved as needed) cannot lower the residual, the start is too far for
    the linearization, so a sequential EP sweep approaches if it lowers the residual; if not,
    the Opper–Winther double loop (provably convergent, and slow) runs until the residual has
    halved, and Newton resumes. The answer is the
    EP stationary point with residual ≤ SITE_TOLERANCE whichever path reached it.
    """
    state = site_state(prior, vector, likelihood_precision, linear_term, start.site_precision, start.site_shift)
    if state is None:
        # The start is outside the domain for this prior; the prior's own sites always lie inside.
        state = prior_sites(prior, vector, likelihood_precision, linear_term)
    size = linear_term.shape[0]
    for _iteration in range(1000):
        if state.residual < SITE_TOLERANCE:
            return state
        jacobian, _precision_by_sites, _shift_by_sites = site_jacobian(state)
        residual_vector = np.concatenate(
            [state.posterior_mean - state.moments["first"], np.diag(state.covariance) + state.posterior_mean**2 - state.moments["second"]]
        )
        step = -np.linalg.solve(jacobian, residual_vector)
        accepted = None
        for _halving in range(40):
            candidate = site_state(prior, vector, likelihood_precision, linear_term, state.site_precision + step[size:], state.site_shift + step[:size])
            if candidate is not None and candidate.residual < state.residual:
                accepted = candidate
                break
            step *= 0.5
        if accepted is not None:
            state = accepted
            continue
        # Newton cannot go below rounding; that floor is the converged solution.
        if state.residual < SITE_RESIDUAL_FLOOR:
            return state
        swept = _sequential_sweep(prior, vector, likelihood_precision, linear_term, state)
        if swept is not None and swept.residual < state.residual:
            state = swept
            continue
        # The double loop only has to bring the sites to where Newton works again: halve the residual.
        advanced = double_loop_sites(prior, vector, likelihood_precision, linear_term, state, 0.5 * state.residual)
        if not advanced.residual < state.residual:
            raise EPFailure(f"EP stalled at moment residual {state.residual:.2e}.")
        state = advanced
    raise EPFailure("EP Newton did not converge.")


def prior_sites(prior: ReferencePrior, vector, likelihood_precision, linear_term) -> SiteState:
    """Moment-matched sites of the prior itself: τ_j = 1/E_prior[β_j²], ν = 0.

    Always inside the EP domain: with every τ_j > 0, Λ + diag τ is positive definite for any
    positive semidefinite Λ, including real LD's singular ones (tied columns), and every cavity
    precision 1/Σ_jj − τ_j is the Schur complement of Λ + diag τ at j, positive as well.
    """
    mixing_coefficients, annotation_coefficients = _unpack(prior, vector)
    log_scale = prior.log_variance_offset + prior.centred_design @ annotation_coefficients
    log_mean_variance = logsumexp(log_mixing_density(prior, mixing_coefficients) + prior.mixing.nodes[None, :], axis=1)
    state = site_state(
        prior,
        vector,
        likelihood_precision,
        linear_term,
        np.exp(-(log_scale + log_mean_variance[prior.class_index])),
        np.zeros(linear_term.shape[0]),
    )
    if state is None:
        raise EPFailure("The prior's own sites leave the EP domain: the tilted moments overflow at this prior.")
    return state


def initial_sites(prior: ReferencePrior, vector, likelihood_precision, linear_term, site_precision) -> SiteState:
    """The EP solution reached from sites with the given precision and no shift."""
    site_precision = np.array(site_precision, dtype=np.float64)
    start = site_state(prior, vector, likelihood_precision, linear_term, site_precision, np.zeros(linear_term.shape[0]))
    return solve_sites(prior, vector, likelihood_precision, linear_term, start if start is not None else prior_sites(prior, vector, likelihood_precision, linear_term))


def total_curvature(prior, vector, state: SiteState) -> np.ndarray:
    """B = −∇² log Z_EP(x) with EP re-solved: the fixed-cavity A corrected by the cavity response.

    ∇ log Z_EP = ∇_x f at the solution's cavities (the EP stationarity), so
    B = A − [∂E_r[β]/∂x]ᵀ dh/dx + ½[∂E_r[β²]/∂x]ᵀ dP/dx, since ∂(∇_x log Z_j)/∂h_j = ∂E_r[β]/∂x
    and ∂(∇_x log Z_j)/∂P_j = −½ ∂E_r[β²]/∂x. The cavity response comes from implicit
    differentiation of R(s, x) = 0: ds/dx = (dR/ds)⁻¹ [∂E_r[β]/∂x; ∂E_r[β²]/∂x].
    """
    fixed_cavity = -cavity_log_marginal_hessian(prior, vector, state.cavity_precision, state.cavity_shift)
    jacobian, precision_by_sites, shift_by_sites = site_jacobian(state)
    first, second = tilted_moment_gradients(prior, vector, state.cavity_precision, state.cavity_shift)
    site_response = np.linalg.solve(jacobian, np.vstack([first, second]))
    curvature = fixed_cavity - first.T @ (shift_by_sites @ site_response) + 0.5 * second.T @ (precision_by_sites @ site_response)
    return 0.5 * (curvature + curvature.T)


def _local_maximum(prior, weights, start, state: SiteState, likelihood_precision, linear_term) -> tuple[np.ndarray, SiteState, float]:
    """A certified local maximum of log Z_EP(x) − ½xᵀS_λx over the allowed coefficients, by
    Newton with the total curvature: the returned point is stationary and B + S_λ is positive
    definite there. The objective is not concave, so wherever the curvature is indefinite the
    step escapes along the most negative direction, and a stationary point that is not a
    maximum is never returned (NoInteriorOptimum)."""
    basis = allowed_basis(prior, weights)
    penalty = penalty_matrix(prior, weights)
    vector = basis @ (basis.T @ start)
    state = solve_sites(prior, vector, likelihood_precision, linear_term, state)

    def objective(point_state, point):
        return point_state.log_evidence - 0.5 * float(point @ penalty @ point)

    value = objective(state, vector)
    if basis.shape[1] == 0:
        return vector, state, value

    def line_search(direction, gain):
        """The first of a halving sequence of steps along ``direction`` that raises the objective by
        ≥ 1e-4·gain·scale (Armijo); a full step that is accepted is then doubled, inside the trust
        region, while each doubling raises the objective further. Along a ridge whose curvature
        falls away (the objective far from quadratic, as along a density's weakly identified
        location or width) the Newton step is too short by a large factor, and without the doubling
        it converges only linearly."""
        scale = 1.0
        for _halving in range(60):
            candidate = basis @ (basis.T @ vector + scale * direction)
            try:
                candidate_state = solve_sites(prior, candidate, likelihood_precision, linear_term, state)
            except EPFailure:
                scale *= 0.5
                continue
            candidate_value = objective(candidate_state, candidate)
            if candidate_value >= value + 1e-4 * scale * gain:
                break
            scale *= 0.5
        else:
            return None
        best = (candidate, candidate_state, candidate_value)
        # Doubling stays inside the trust region, so a genuine ray is still walked a radius at a time.
        while scale >= 1.0 and 2.0 * scale * float(np.max(np.abs(basis @ direction))) <= COEFFICIENT_STEP_LIMIT:
            scale *= 2.0
            longer = basis @ (basis.T @ vector + scale * direction)
            try:
                longer_state = solve_sites(prior, longer, likelihood_precision, linear_term, best[1])
            except EPFailure:
                break
            longer_value = objective(longer_state, longer)
            if not longer_value > best[2]:
                break
            best = (longer, longer_state, longer_value)
        return best

    for _iteration in range(100):
        _unused, gradient = cavity_log_marginal(prior, vector, state.cavity_precision, state.cavity_shift)
        restricted_gradient = basis.T @ (gradient - penalty @ vector)
        hessian = basis.T @ (total_curvature(prior, vector, state) + penalty) @ basis
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        floor = 1e-8 * float(np.max(np.abs(eigenvalues)))
        # Definite at the eigenvalues' own rounding scale (eigh's error is about ε‖H‖): a large λ
        # makes H ill-conditioned without making its small eigenvalues any less positive.
        definite = bool(eigenvalues.min() > 64.0 * np.finfo(np.float64).eps * float(np.max(np.abs(eigenvalues))))
        # Modified Newton: negative curvature is taken with its magnitude, so the step ascends.
        step = eigenvectors @ ((eigenvectors.T @ restricted_gradient) / np.maximum(np.abs(eigenvalues), floor))
        # A trust region on the coefficients (a numerical safeguard; it does not move x̂).
        step *= min(1.0, COEFFICIENT_STEP_LIMIT / max(float(np.max(np.abs(basis @ step))), 1e-300))
        decrement = 0.5 * float(restricted_gradient @ step)
        if decrement < 1e-13 * (1.0 + abs(value)):
            if not definite:
                # Stationary but not a maximum: escape along the most negative curvature.
                escape = eigenvectors[:, 0] * COEFFICIENT_STEP_LIMIT / max(float(np.max(np.abs(basis @ eigenvectors[:, 0]))), 1e-300)
                moved = line_search(escape, 0.0) or line_search(-escape, 0.0)
                if moved is None:
                    raise NoInteriorOptimum("A stationary point with indefinite curvature and no ascent along it.")
                vector, state, value = moved
                continue
            if decrement < 1e-20 * (1.0 + abs(value)):
                return vector, state, value
            # Quadratic regime: judge full Newton steps by the gradient, not the rounded value.
            # A step that does not at least halve it has reached the gradient's rounding floor.
            candidate = basis @ (basis.T @ vector + step)
            candidate_state = solve_sites(prior, candidate, likelihood_precision, linear_term, state)
            _unused, candidate_gradient = cavity_log_marginal(prior, candidate, candidate_state.cavity_precision, candidate_state.cavity_shift)
            if not np.max(np.abs(basis.T @ (candidate_gradient - penalty @ candidate))) < 0.5 * np.max(np.abs(restricted_gradient)):
                return vector, state, value
            vector, state = candidate, candidate_state
            value = objective(state, vector)
            continue
        moved = line_search(step, 2.0 * decrement)
        if moved is None:
            if definite:
                return vector, state, value
            raise NoInteriorOptimum("No ascent from a point with indefinite curvature.")
        vector, state, value = moved
    raise NoInteriorOptimum("The coefficient maximization did not settle: the coefficients run along a ray.")


def _normal_coefficients(prior: ReferencePrior, location: float, width: float) -> np.ndarray:
    """T_1 and T_2 coefficients of log g = −(t − location)²/(2 width²) (up to a constant), t = t_lo + a(x + 1)."""
    half_length = 0.5 * (prior.mixing.upper - prior.mixing.lower)
    return np.array([-half_length * (half_length + prior.mixing.lower - location) / width**2, -(half_length**2) / (4.0 * width**2)])


def _normal_start(prior, weights, start, state: SiteState, likelihood_precision, linear_term) -> tuple[np.ndarray, SiteState] | None:
    """With η in its null space (its weight at ∞), the best normal density in log s over a grid of
    (location, width), the rest of the start kept: the null-space objective is not concave, so a
    local maximization from one start can stop in the wrong basin (math-density measured 137.6
    against the global 141.7).

    The grid points are ranked by the fixed-cavity objective at the start's cavities (the EP-EM
    M-step objective: no EP solve per point), and EP is solved at the best one only; a start only
    chooses the basin, whose maximum is certified afterwards. Locations are the panel centres;
    widths double from √2, the narrowest feature any kernel resolves in t (math-density K), to
    the range's length.
    """
    if not np.isinf(weights[0]) or prior.degree < 2:
        return None
    mixing = prior.mixing
    penalty = penalty_matrix(prior, weights)
    length = mixing.upper - mixing.lower
    widths = np.sqrt(2.0) * 2.0 ** np.arange(max(int(np.ceil(np.log2(length / np.sqrt(2.0)))), 0) + 1)
    best = None
    for location in 0.5 * (mixing.panel_edges[:-1] + mixing.panel_edges[1:]):
        for width in widths:
            candidate = np.array(start, dtype=np.float64, copy=True)
            candidate[: prior.degree] = 0.0
            candidate[: min(2, prior.degree)] = _normal_coefficients(prior, location, width)[: min(2, prior.degree)]
            value = cavity_log_marginal(prior, candidate, state.cavity_precision, state.cavity_shift)[0] - 0.5 * float(candidate @ penalty @ candidate)
            if best is None or value > best[1]:
                best = (candidate, value)
    if best is None:
        return None
    try:
        return best[0], solve_sites(prior, best[0], likelihood_precision, linear_term, state)
    except EPFailure:
        return None


def certified_maxima(prior, weights, start, state: SiteState, likelihood_precision, linear_term, structural: bool = True) -> list[tuple[np.ndarray, SiteState]]:
    """The distinct certified local maxima of log Z_EP(x) − ½xᵀS_λx reached from the starts.

    With ``structural`` (every fresh evaluation), the starts are the warm one given; the flat one
    (every log-density 0, θ = 0, the prior's own sites); and, when η is in its null space, the
    best normal density on a (location, width) grid. The objective is not concave, and a warm
    start can sit in a basin (a density collapsing toward zero variance) that the others avoid.
    Without it (a small step in λ from a certified point), the warm start alone continues its
    basin. Two maxima are one basin when their quadratic-model separation ½ΔxᵀHΔx is within
    √ε of the objective's scale: a maximum located from its stationarity is resolved in x only to √ε.
    """
    starts = [(start, state)]
    if structural:
        flat = np.zeros(prior.coefficient_count)
        starts.append((flat, prior_sites(prior, flat, likelihood_precision, linear_term)))
        normal = _normal_start(prior, weights, start, state, likelihood_precision, linear_term)
        if normal is not None:
            starts.append(normal)
    penalty = penalty_matrix(prior, weights)
    found: list[tuple[np.ndarray, SiteState, float]] = []
    for candidate_start, candidate_state in starts:
        try:
            vector, reached, value = _local_maximum(prior, weights, candidate_start, candidate_state, likelihood_precision, linear_term)
        except (NoInteriorOptimum, EPFailure):
            continue
        hessian = total_curvature(prior, vector, reached) + penalty
        duplicate = any(
            0.5 * float((vector - other) @ hessian @ (vector - other)) <= np.sqrt(np.finfo(np.float64).eps) * (1.0 + abs(value))
            for other, _state, _value in found
        )
        if not duplicate:
            found.append((vector, reached, value))
    if not found:
        raise NoInteriorOptimum("No start reaches a certified maximum.")
    return [(vector, reached) for vector, reached, _value in found]


def maximize_penalized_evidence(prior, weights, start, state: SiteState, likelihood_precision, linear_term, structural: bool = True) -> tuple[np.ndarray, SiteState]:
    """x̂: the certified maximum (of those certified_maxima reaches) whose evidence is highest, the
    basins compared by the Tierney–Kadane-corrected V (``laplace_evidence``)."""
    point = laplace_evidence(prior, weights, start, state, likelihood_precision, linear_term, with_gradient=False, structural=structural)
    return point.coefficients, point.state


def _curvature_derivative(prior, vector, direction, state, likelihood_precision, linear_term) -> np.ndarray:
    """D_x B[direction], by central differences of the total curvature, EP re-solved at each side."""
    step = 1e-4 / max(float(np.max(np.abs(direction))), 1e-300)
    sides = []
    for sign in (1.0, -1.0):
        moved = vector + sign * step * direction
        sides.append(total_curvature(prior, moved, solve_sites(prior, moved, likelihood_precision, linear_term, state)))
    return (sides[0] - sides[1]) / (2.0 * step)


@dataclass(frozen=True)
class EvidencePoint:
    """V (the Laplace evidence) at some weights, x̂ and the EP solution there, and dV/dλ_i for
    every finite weight (nan otherwise), with the prior and weights it was evaluated at."""

    value: float
    coefficients: np.ndarray
    state: SiteState
    weight_gradient: np.ndarray
    prior: "ReferencePrior"
    weights: np.ndarray
    # The standardized directions and their Tierney–Kadane terms, computed once per point.
    corrections: dict = field(default_factory=dict, compare=False, repr=False)


def profiled_basis(prior: ReferencePrior, weights: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """The allowed directions no finite penalty acts on (η's null space, a smooth annotation's
    linear effect): the complement of integrated_basis within the allowed basis."""
    integrated = integrated_basis(prior, weights, basis)
    remainder = basis - integrated @ (integrated.T @ basis)
    left, singular, _right = np.linalg.svd(remainder, full_matrices=False)
    return left[:, singular > 1e-8]


def _log_determinant(matrix: np.ndarray) -> float:
    if matrix.shape[0] == 0:
        return 0.0
    try:
        factor = cho_factor(matrix)
    except np.linalg.LinAlgError as error:
        raise NoInteriorOptimum("log Z_EP − ½xᵀS_λx is not strictly concave at its stationary point.") from error
    return 2.0 * float(np.sum(np.log(np.diag(factor[0]))))


def evidence_at(prior, weights, coefficients, state, likelihood_precision, linear_term, with_gradient: bool = True) -> EvidencePoint:
    """V(λ) = log Z_EP(x̂) − ½x̂ᵀS_λx̂ + ½log|RᵀS_λR| − ½log|Kᵀ(B + S_λ)K| + ½log|Nᵀ(B + S_λ)N| at a
    certified maximum x̂ (``coefficients``, with its EP solution ``state``).

    B is the total curvature, K the allowed basis, R its penalized directions and N its
    profiled (unpenalized) ones. Profiling N gives the Schur form
    −½log|K ᵀHK| + ½log|NᵀHN| = −½log of H's Schur complement on N, the curvature of the
    profile in the integrated directions, not the conditional block RᵀHR (which is not the
    profile and fails numerically).
    dV/dλ_i = −½x̂ᵀS_ix̂ + ½tr(S_λ⁺S_i) − ½tr(H_KK⁻¹Kᵀ(S_i + D_xB[dx̂])K) + ½tr(H_NN⁻¹NᵀD_xB[dx̂]N),
    with dx̂/dλ_i = −K H_KK⁻¹KᵀS_ix̂ from the stationarity of x̂ (NᵀS_iN = 0).
    """
    weights = np.asarray(weights, dtype=np.float64)
    basis = allowed_basis(prior, weights)
    integrated = integrated_basis(prior, weights, basis)
    profiled = profiled_basis(prior, weights, basis)
    penalty = penalty_matrix(prior, weights)
    if basis.shape[1] == 0:
        return EvidencePoint(
            value=state.log_evidence, coefficients=coefficients, state=state, weight_gradient=np.full(weights.shape[0], np.nan),
            prior=prior, weights=weights,
        )
    hessian = total_curvature(prior, coefficients, state) + penalty
    kept_hessian = basis.T @ hessian @ basis
    profiled_hessian = profiled.T @ hessian @ profiled
    log_determinant_penalty, _rank, penalty_inverse = _penalty_log_determinant(prior, weights, integrated)
    evidence = (
        state.log_evidence
        - 0.5 * float(coefficients @ penalty @ coefficients)
        + 0.5 * log_determinant_penalty
        - 0.5 * _log_determinant(kept_hessian)
        + 0.5 * _log_determinant(profiled_hessian)
    )
    gradient = np.full(weights.shape[0], np.nan)
    if with_gradient and integrated.shape[1]:
        kept_inverse = np.linalg.inv(kept_hessian)
        profiled_inverse = np.linalg.inv(profiled_hessian) if profiled.shape[1] else np.zeros((0, 0))
        for position, ((columns, matrix), weight) in enumerate(zip(_penalty_terms(prior), weights)):
            if not np.isfinite(weight) or columns.size == 0:
                continue
            term = _embedded(prior, columns, matrix)
            coefficient_derivative = -basis @ (kept_inverse @ (basis.T @ (term @ coefficients)))
            curvature_derivative = _curvature_derivative(prior, coefficients, coefficient_derivative, state, likelihood_precision, linear_term)
            gradient[position] = (
                -0.5 * float(coefficients @ term @ coefficients)
                + 0.5 * float(np.sum(penalty_inverse * (integrated.T @ term @ integrated)))
                - 0.5 * float(np.sum(kept_inverse * (basis.T @ (term + curvature_derivative) @ basis)))
                + 0.5 * float(np.sum(profiled_inverse * (profiled.T @ curvature_derivative @ profiled)))
            )
    return EvidencePoint(value=float(evidence), coefficients=coefficients, state=state, weight_gradient=gradient, prior=prior, weights=weights)


def laplace_evidence(prior, weights, start, state, likelihood_precision, linear_term, with_gradient: bool = True, structural: bool = True) -> EvidencePoint:
    """V at the certified maximum with the highest evidence: the Laplace value, with distinct
    basins compared by their Tierney–Kadane-corrected V (``prefer``)."""
    weights = np.asarray(weights, dtype=np.float64)
    maxima = certified_maxima(prior, weights, start, state, likelihood_precision, linear_term, structural)
    best = None
    for coefficients, reached in maxima:
        point = evidence_at(prior, weights, coefficients, reached, likelihood_precision, linear_term, with_gradient=False)
        if best is None or prefer(point, best, likelihood_precision, linear_term):
            best = point
    if with_gradient:
        best = evidence_at(prior, weights, best.coefficients, best.state, likelihood_precision, linear_term, with_gradient=True)
    return best


def laplace_corrections(point: EvidencePoint, likelihood_precision, linear_term, tolerance: float) -> float:
    """The Tierney–Kadane correction to V at ``point``, certified to ``tolerance`` nats (lead ruling).

    As in the engine, it is taken on the fixed-cavity objective f(x) = Σ_j log Z_j(x) − ½xᵀS_λx at
    the point's cavities (the EP-EM surrogate whose curvature is A + S): the correction measures
    the integrand's departure from its own Gaussian, and needs no EP solution away from x̂.
    The integrated directions are the eigenvectors of the Schur complement of A + S on the
    penalized directions R, each moved with the profiled coordinates' first-order response and
    scaled to unit curvature. Along such a direction d, f(x̂ + sd) − f(x̂) has second derivative −1
    at 0 and O(1) Laplace error κ4/8 + 5κ3²/24, with κ3 and κ4 its third and fourth derivatives
    (central differences of dᵀA(x̂ + sd)d at the step ε^¼, which balances A's rounding against
    truncation). The directions with the largest terms are replaced by the log of the exact line
    integral over its Laplace value √(2π) until the terms left sum to at most tolerance/2, each
    integral taken to relative tolerance/(2m) over the m replaced.
    """
    prior, weights = point.prior, point.weights
    basis = allowed_basis(prior, weights)
    integrated = integrated_basis(prior, weights, basis)
    if integrated.shape[1] == 0:
        return 0.0
    profiled = profiled_basis(prior, weights, basis)
    penalty = penalty_matrix(prior, weights)
    coefficients, cavity_precision, cavity_shift = point.coefficients, point.state.cavity_precision, point.state.cavity_shift

    def curvature_at(vector):
        return penalty - cavity_log_marginal_hessian(prior, vector, cavity_precision, cavity_shift)

    def objective(vector):
        return cavity_log_marginal(prior, vector, cavity_precision, cavity_shift)[0] - 0.5 * float(vector @ penalty @ vector)

    base = objective(coefficients)
    if "terms" not in point.corrections:
        hessian = curvature_at(coefficients)
        response = np.eye(prior.coefficient_count)
        if profiled.shape[1]:
            response = response - profiled @ np.linalg.solve(profiled.T @ hessian @ profiled, profiled.T @ hessian)
        moved = response @ integrated
        eigenvalues, eigenvectors = np.linalg.eigh(moved.T @ hessian @ moved)
        if not np.all(eigenvalues > 0.0):
            raise UncertifiedCorrection("The fixed-cavity curvature is not positive definite at the point.")
        directions = moved @ eigenvectors / np.sqrt(eigenvalues)[None, :]
        step = np.finfo(np.float64).eps ** 0.25
        terms = np.empty(directions.shape[1])
        for index in range(directions.shape[1]):
            direction = directions[:, index]
            forward = float(direction @ curvature_at(coefficients + step * direction) @ direction)
            backward = float(direction @ curvature_at(coefficients - step * direction) @ direction)
            centre = float(direction @ hessian @ direction)
            third = -(forward - backward) / (2.0 * step)
            fourth = -(forward - 2.0 * centre + backward) / step**2
            terms[index] = fourth / 8.0 + 5.0 * third**2 / 24.0
        point.corrections.update(terms=terms, directions=directions, integrals={})
    terms, directions, integrals = point.corrections["terms"], point.corrections["directions"], point.corrections["integrals"]
    order = np.argsort(-np.abs(terms))
    remaining = np.concatenate([np.cumsum(np.abs(terms[order])[::-1])[::-1], [0.0]])
    replaced = order[: int(np.argmax(remaining <= 0.5 * tolerance))]
    share = 0.5 * tolerance / max(replaced.shape[0], 1)
    correction = 0.0
    for index in replaced:
        # A line integral already taken to a relative accuracy at least this fine is reused.
        if index not in integrals or integrals[index][1] > share:
            direction = directions[:, index]
            with np.errstate(over="ignore", invalid="ignore"):
                integral, _error = integrate.quad(
                    lambda at: float(np.exp(objective(coefficients + at * direction) - base)), -np.inf, np.inf, epsabs=0.0, epsrel=share
                )
            if not (np.isfinite(integral) and integral > 0.0):
                raise UncertifiedCorrection("A line integral of the fixed-cavity objective is not finite.")
            integrals[index] = (float(np.log(integral)) - 0.5 * float(np.log(2.0 * np.pi)), share)
        correction += integrals[index][0]
    return correction


class UncertifiedCorrection(ArithmeticError):
    """A Tierney–Kadane correction could not be certified (an indefinite curvature, or a line integral that is not finite)."""


def prefer(first: EvidencePoint, second: "EvidencePoint | float", likelihood_precision, linear_term) -> bool:
    """Whether ``first`` has the higher Tierney–Kadane-corrected evidence than ``second`` (a point,
    or an exact evidence such as the null model's 0).

    Each corrected value is certified to a tolerance t, so their difference to 2t; t starts at
    a quarter of the Laplace gap and halves until the corrected gap exceeds 2t, or t reaches the
    fit's certified tolerance, where the two are equal and the Laplace order stands.
    A point with no integrated directions (a one-node density) is exact. Where a correction cannot
    be certified, the Laplace order stands too.
    """
    exact = not isinstance(second, EvidencePoint)
    gap = first.value - (second if exact else second.value)
    floor = first.prior.tolerance
    # Values within the fit's certified tolerance are equal: the Laplace order stands.
    if abs(gap) <= floor:
        return gap > 0.0
    tolerance = 0.25 * abs(gap)
    while True:
        try:
            corrected = gap + laplace_corrections(first, likelihood_precision, linear_term, tolerance)
            if not exact:
                corrected -= laplace_corrections(second, likelihood_precision, linear_term, tolerance)
        except UncertifiedCorrection:
            # No certified correction exists at this tolerance: the Laplace order is all there is.
            return gap > 0.0
        if abs(corrected) > 2.0 * tolerance or tolerance <= floor:
            return corrected > 0.0 or (abs(corrected) <= 2.0 * tolerance and gap > 0.0)
        tolerance = max(0.5 * tolerance, floor)


@dataclass(frozen=True)
class EdgeScore:
    """At λ_i = ∞: dV/d(1/λ_i) = ½(q − d + c).

    q = GᵀE⁻¹G is the squared score; d = tr(E⁻¹B_UU) − tr(E⁻¹H_UK H_KK⁻¹H_KU) is the information
    on the released directions given every kept one; c = −tr(H_KK⁻¹KᵀD_xB[v]K) +
    tr(H_NN⁻¹NᵀD_xB[v]N) is the change of the profiled curvature as x̂ moves along v =
    dx̂/d(1/λ_i). c vanishes for a Gaussian likelihood, where this is Tipping–Faul.
    """

    squared_score: float
    information: float
    curvature_change: float

    @property
    def derivative(self) -> float:
        return 0.5 * (self.squared_score - self.information + self.curvature_change)

    @property
    def holds(self) -> bool:
        scale = abs(self.squared_score) + abs(self.information) + abs(self.curvature_change) + 1.0
        return 2.0 * self.derivative <= EDGE_SCORE_TOLERANCE * scale


def edge_score(prior, weights, position, point: EvidencePoint, likelihood_precision, linear_term) -> EdgeScore:
    """The score test for releasing weight ``position`` from ∞, in closed form at 1/λ → 0.

    With x = Kz + Uy (K the current allowed basis, U the directions the release adds) and
    y ~ N(0, τE⁻¹), τ = 1/λ_i, E = UᵀS_iU, expanding the Laplace evidence (Schur form, see
    laplace_evidence) in τ gives V(τ) = V(0) + ½τ(q − d + c) + O(τ²).
    - The max term gains ½τGᵀE⁻¹G, G = Uᵀ∇log Z_EP, since ŷ = τE⁻¹G.
    - The r·log τ and log|E| parts of log|S_λ| and log|H_(K,U)| cancel; the Schur complement
      of the U block in log|H_(K,U)| gives −tr(E⁻¹B_UU) + tr(E⁻¹H_UK H_KK⁻¹H_KU). No other term
      acts on U (each block has one penalty), so log|S_λ| adds nothing more.
    - log|H_KK(x̂(τ))| and log|H_NN(x̂(τ))| move along v = (U − K H_KK⁻¹H_KU)E⁻¹G, every kept
      coordinate re-optimizing; N ⊂ K is profiled and H = B + S_λ with the current finite terms.
    """
    weights = np.asarray(weights, dtype=np.float64)
    released = weights.copy()
    released[position] = 1.0
    basis = allowed_basis(prior, weights)
    released_basis = allowed_basis(prior, released)
    complement = released_basis - basis @ (basis.T @ released_basis)
    left, _singular, _right = np.linalg.svd(complement, full_matrices=False)
    added = left[:, : released_basis.shape[1] - basis.shape[1]]
    columns, matrix = _penalty_terms(prior)[position]
    information_inverse = np.linalg.inv(added.T @ _embedded(prior, columns, matrix) @ added)
    coefficients, state = point.coefficients, point.state
    _unused, data_gradient = cavity_log_marginal(prior, coefficients, state.cavity_precision, state.cavity_shift)
    curvature = total_curvature(prior, coefficients, state)
    hessian = curvature + penalty_matrix(prior, weights)
    profiled = profiled_basis(prior, weights, basis)
    kept_hessian = basis.T @ hessian @ basis
    coupling = added.T @ hessian @ basis
    released_gradient = added.T @ data_gradient
    released_step = information_inverse @ released_gradient
    direction = added @ released_step - basis @ np.linalg.solve(kept_hessian, coupling.T @ released_step)
    curvature_derivative = _curvature_derivative(prior, coefficients, direction, state, likelihood_precision, linear_term)
    profiled_change = (
        float(np.trace(np.linalg.solve(profiled.T @ hessian @ profiled, profiled.T @ curvature_derivative @ profiled))) if profiled.shape[1] else 0.0
    )
    return EdgeScore(
        squared_score=float(released_gradient @ released_step),
        information=float(
            np.trace(information_inverse @ (added.T @ curvature @ added))
            - np.trace(information_inverse @ coupling @ np.linalg.solve(kept_hessian, coupling.T))
        ),
        curvature_change=-float(np.trace(np.linalg.solve(kept_hessian, basis.T @ curvature_derivative @ basis))) + profiled_change,
    )


def boundary_conditions(prior, weights, point, likelihood_precision, linear_term) -> list[tuple[int, EdgeScore]]:
    """The score test at every weight sitting at ∞."""
    active = _active_terms(prior)
    return [
        (int(position), edge_score(prior, weights, position, point, likelihood_precision, linear_term))
        for position in np.flatnonzero(np.isinf(weights) & active)
    ]


def _active_terms(prior: ReferencePrior) -> np.ndarray:
    """The penalty terms whose weights are learned: those acting on some coefficient (a one-node
    representation has none), and the class deviations only when there are two classes or more.

    With one class, η + δ_0 is all the likelihood sees, so δ_0 is not identified: its weights have
    no evidence of their own (V is flat along them) and they sit at ∞, δ_0 = 0, with η carrying
    the class's density.
    """
    degree = prior.degree
    deviation_columns = np.arange(degree, (prior.class_count + 1) * degree)
    return np.array(
        [
            columns.size > 0 and (prior.class_count > 1 or not np.isin(columns, deviation_columns).all())
            for columns, _matrix in _penalty_terms(prior)
        ]
    )


def _evaluate(prior, weights, start_point, likelihood_precision, linear_term, structural=True):
    """V and x̂ at these weights, or None where log Z_EP − ½xᵀS_λx has no certified maximum or EP fails."""
    try:
        return laplace_evidence(prior, weights, start_point.coefficients, start_point.state, likelihood_precision, linear_term, structural=structural)
    except (NoInteriorOptimum, EPFailure):
        return None


def _at_resolution_limit(prior, weights, positions, point: EvidencePoint) -> np.ndarray:
    """For each term, whether λ·s_min ≥ ‖D‖/√ε: its smallest penalty eigenvalue s_min times λ
    exceeds the data's curvature D = RᵀBR on the term's range by 1/√ε, so in double precision the
    term's directions are already pinned to its null space and V cannot resolve λ from ∞."""
    curvature = total_curvature(prior, point.coefficients, point.state)
    terms = _penalty_terms(prior)
    limit = np.zeros(len(positions), dtype=bool)
    for index, position in enumerate(positions):
        columns, matrix = terms[position]
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        penalized = eigenvalues > 1e-10 * max(float(eigenvalues.max()), 1.0)
        embedded = np.zeros((prior.coefficient_count, int(penalized.sum())))
        embedded[columns] = eigenvectors[:, penalized]
        data = float(np.linalg.norm(embedded.T @ curvature @ embedded, 2))
        limit[index] = weights[position] * float(eigenvalues[penalized].min()) * np.sqrt(np.finfo(np.float64).eps) >= data
    return limit


def _log_weight_metric(prior, weights, free, point, likelihood_precision, linear_term) -> np.ndarray:
    """The inverse of |∂²V/∂(log λ)²| over the free weights, by central differences of the exact
    gradient (the step ε^⅓ balances the gradient's rounding against truncation); a
    steepest-ascent metric where a neighbour has no certified maximum."""
    step = np.finfo(np.float64).eps ** (1.0 / 3.0)
    log_gradient = weights[free] * point.weight_gradient[free]
    hessian = np.empty((free.size, free.size))
    for column, position in enumerate(free):
        gradients = []
        for sign in (1.0, -1.0):
            moved = weights.copy()
            moved[position] = weights[position] * np.exp(sign * step)
            moved_point = _evaluate(prior, moved, point, likelihood_precision, linear_term, structural=False)
            if moved_point is None:
                return np.eye(free.size) / max(float(np.max(np.abs(log_gradient))), 1.0)
            gradients.append(moved[free] * moved_point.weight_gradient[free])
        hessian[:, column] = (gradients[0] - gradients[1]) / (2.0 * step)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (hessian + hessian.T))
    floor = np.finfo(np.float64).eps * max(float(np.max(np.abs(eigenvalues))), np.finfo(np.float64).tiny)
    return eigenvectors @ np.diag(1.0 / np.maximum(np.abs(eigenvalues), floor)) @ eigenvectors.T


def _ascend(prior, weights, point, likelihood_precision, linear_term):
    """Quasi-Newton ascent of V in log λ over the finite weights (BFGS from a central-difference
    Hessian), moving a weight to ∞, in closed form, wherever V is highest there and the score
    test holds. Weights with no certified maximum in x are not candidates.

    It is stationary when the quasi-Newton step's predicted gain ½gᵀH⁻¹g, the Newton decrement in
    log λ, is within the fit's certified tolerance (the engine's rule): V is not certified more
    finely, so a smaller gain is not a resolved improvement. A step is taken only if V rises
    by at least a 1e-4 share of its predicted first-order gain (Armijo; Nocedal and Wright, Alg. 3.1).
    """
    active = _active_terms(prior)
    inverse_hessian = None
    free_before = None
    for _iteration in range(500):
        free = np.flatnonzero(np.isfinite(weights) & active)
        if free.size == 0:
            return weights, point
        log_gradient = weights[free] * point.weight_gradient[free]
        if inverse_hessian is None or free_before is None or not np.array_equal(free, free_before):
            inverse_hessian = _log_weight_metric(prior, weights, free, point, likelihood_precision, linear_term)
            free_before = free
        predicted = 0.5 * float(log_gradient @ inverse_hessian @ log_gradient)
        stationary = predicted <= prior.tolerance
        # V can have separate maxima inside and at ∞, so once the ascent is stationary every finite
        # weight's edge is compared (in closed form, by the corrected V). Before that, a weight
        # rising past where double precision still resolves its term from the edge is compared too.
        candidates = free if stationary else free[(log_gradient > 0.0) & _at_resolution_limit(prior, weights, free, point)]
        best = None
        for position in candidates:
            candidate = weights.copy()
            candidate[position] = np.inf
            candidate_point = _evaluate(prior, candidate, point, likelihood_precision, linear_term)
            if candidate_point is None or not edge_score(prior, candidate, position, candidate_point, likelihood_precision, linear_term).holds:
                continue
            if prefer(candidate_point, best[1] if best is not None else point, likelihood_precision, linear_term):
                best = (candidate, candidate_point)
        if best is not None:
            weights, point = best
            inverse_hessian = None
            continue
        if stationary:
            return weights, point
        ascent = inverse_hessian @ log_gradient
        # A trust region in log λ of one e-fold (a numerical safeguard; it does not move λ̂).
        ascent *= min(1.0, 1.0 / float(np.max(np.abs(ascent))))
        for _halving in range(60):
            candidate = weights.copy()
            candidate[free] = weights[free] * np.exp(ascent)
            candidate_point = _evaluate(prior, candidate, point, likelihood_precision, linear_term, structural=False)
            if candidate_point is not None and candidate_point.value >= point.value + 1e-4 * float(ascent @ log_gradient):
                break
            ascent *= 0.5
        else:
            return weights, point
        new_gradient = candidate[free] * candidate_point.weight_gradient[free]
        # BFGS on −V: s = Δlog λ, y = −Δ(dV/dlog λ); keep the update only when it stays positive definite.
        displacement = np.log(candidate[free]) - np.log(weights[free])
        change = -(new_gradient - log_gradient)
        curvature = float(displacement @ change)
        if curvature > np.finfo(np.float64).eps * float(np.linalg.norm(displacement) * np.linalg.norm(change)):
            rho = 1.0 / curvature
            identity = np.eye(free.size)
            inverse_hessian = (identity - rho * np.outer(displacement, change)) @ inverse_hessian @ (
                identity - rho * np.outer(change, displacement)
            ) + rho * np.outer(displacement, displacement)
        weights, point = candidate, candidate_point
    raise RuntimeError("The penalty-weight ascent did not converge.")


def _release(prior, weights, position, point, likelihood_precision, linear_term):
    """Move weight ``position`` from ∞ to a finite value where V is higher: λ is walked down from
    where its term's prior precision is 1/√ε of the data's curvature on it (the resolution limit,
    so the fit starts at the edge's own solution) by decades, each fit warm-started from the last,
    and the first λ whose V the corrected comparison prefers to the edge's is taken. The ascent
    from there finds the interior optimum; the walk only finds a finite start."""
    columns, matrix = _penalty_terms(prior)[position]
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    penalized = eigenvalues > 1e-10 * max(float(eigenvalues.max()), 1.0)
    embedded = np.zeros((prior.coefficient_count, int(penalized.sum())))
    embedded[columns] = eigenvectors[:, penalized]
    data = float(np.linalg.norm(embedded.T @ total_curvature(prior, point.coefficients, point.state) @ embedded, 2))
    smallest = float(eigenvalues[penalized].min())
    current = point
    level = np.log(max(data, np.finfo(np.float64).tiny) / smallest) - 0.5 * np.log(np.finfo(np.float64).eps)
    # From the resolution limit down to where the prior precision is ε of the data's curvature.
    for log_weight in np.arange(level, level + np.log(np.finfo(np.float64).eps) - np.log(10.0), -np.log(10.0)):
        candidate = weights.copy()
        candidate[position] = float(np.exp(log_weight))
        candidate_point = _evaluate(prior, candidate, current, likelihood_precision, linear_term, structural=False)
        if candidate_point is None:
            continue
        current = candidate_point
        if prefer(candidate_point, point, likelihood_precision, linear_term):
            return candidate, candidate_point
    return None


def maximize_evidence(prior, weights, start, state, likelihood_precision, linear_term) -> tuple[np.ndarray, EvidencePoint]:
    """λ̂ = argmax V over (0, ∞] for every weight, with x̂ and the EP solution there."""
    weights = np.array(weights, dtype=np.float64)
    weights[~_active_terms(prior)] = np.inf
    point = laplace_evidence(prior, weights, start, state, likelihood_precision, linear_term)
    for _round in range(100):
        weights, point = _ascend(prior, weights, point, likelihood_precision, linear_term)
        releases = []
        for position, score in boundary_conditions(prior, weights, point, likelihood_precision, linear_term):
            if not score.holds:
                released = _release(prior, weights, position, point, likelihood_precision, linear_term)
                if released is not None:
                    releases.append(released)
        if not releases:
            return weights, point
        weights, point = releases[0]
        for release in releases[1:]:
            if prefer(release[1], point, likelihood_precision, linear_term):
                weights, point = release
    raise RuntimeError("The penalty-weight search did not settle.")


def newton_decrement(prior, weights, coefficients, state: SiteState) -> float:
    """½ gᵀ(B + S_λ)⁻¹g of log Z_EP − ½xᵀS_λx over the allowed coefficients (B the total curvature):
    what a Newton step would still gain, in nats."""
    basis = allowed_basis(prior, weights)
    penalty = penalty_matrix(prior, weights)
    _unused, gradient = cavity_log_marginal(prior, coefficients, state.cavity_precision, state.cavity_shift)
    restricted_gradient = basis.T @ (gradient - penalty @ coefficients)
    hessian = basis.T @ (total_curvature(prior, coefficients, state) + penalty) @ basis
    return float(0.5 * restricted_gradient @ np.linalg.solve(hessian, restricted_gradient))


def derived_log_variance_range(design: ReferenceDesign, annotation_coefficients, cavity_precision, cavity_shift, tolerance) -> tuple[float, float]:
    """The range of log variance the evidence can see, from the kernels L_j(t) (math-density K, L, U).

    - Lower end: log L_j(t) = −½log(1 + vP) + ½h²v/(1 + vP), v = u_je^t, lies within
      b_j = ½v|h² − P| + ¼v²P² of its v → 0 limit, so choosing e^{t_lo} with Σ_j b_j = tolerance
      makes every density below t_lo act as "effect below resolution" to within the tolerance.
      The root is taken in its stable form s = 2·tol/(A1 + √(A1² + 4·A2·tol)).
    - Upper end: past its mode v* = (h² − P)/P² (when h² > P) every L_j decreases, so t_hi is
      the largest mode, or where the least precise kernel starts to fall when none has a mode.
    """
    log_scale = design.log_variance_offset + design.centred_design @ annotation_coefficients
    scale = np.exp(log_scale)
    shift_square = np.square(cavity_shift)
    linear = 0.5 * float(np.sum(scale * np.abs(shift_square - cavity_precision)))
    quadratic = 0.25 * float(np.sum(np.square(scale * cavity_precision)))
    lower = float(np.log(2.0 * tolerance / (linear + np.sqrt(linear**2 + 4.0 * quadratic * tolerance))))
    has_mode = shift_square > cavity_precision
    if np.any(has_mode):
        upper = float(np.max(np.log((shift_square[has_mode] - cavity_precision[has_mode]) / cavity_precision[has_mode] ** 2) - log_scale[has_mode]))
    else:
        upper = float(np.max(-np.log(cavity_precision * scale)))
    return lower, max(upper, lower + MAXIMUM_PANEL_LENGTH)


def _project(fit: ReferenceFit, prior: ReferencePrior) -> np.ndarray:
    """The fitted parameters expressed on another range or degree: η and each δ_c evaluated at
    the new nodes and refitted by least squares, the constant absorbed (a warm start only)."""
    old = fit.prior.mixing
    new = prior.mixing
    design = np.column_stack([np.ones(new.node_count), new.design])

    def refit(coefficients):
        return np.linalg.lstsq(design, old.log_density_at(coefficients, new.nodes), rcond=None)[0][1:]

    hyperparameters = fit.hyperparameters
    return np.concatenate(
        [refit(hyperparameters.shared_coefficients)]
        + [refit(deviation) for deviation in hyperparameters.deviation_coefficients]
        + [hyperparameters.annotation_coefficients]
    )


def fit_fixed_point(prior, likelihood_precision, linear_term, start, weights, state: SiteState) -> ReferenceFit:
    """The EP-EB fixed point at one representation, by nested exact optimization: λ̂ maximizes V,
    x̂ maximizes log Z_EP − ½xᵀS_λx at λ̂, and EP is solved at x̂. It is the joint fixed point
    (EP stationary, x stationary, λ at its KKT point) whatever path reached it."""
    weights, point = maximize_evidence(prior, weights, start, state, likelihood_precision, linear_term)
    # Refit from the best inner solution: when the λ search's own path left the coefficients in a
    # worse basin, the restart finds the better one and the search runs again from there.
    for _restart in range(10):
        restarted_weights, restarted = maximize_evidence(prior, weights, point.coefficients, point.state, likelihood_precision, linear_term)
        if not prefer(restarted, point, likelihood_precision, linear_term):
            break
        weights, point = restarted_weights, restarted
    hyperparameters = _hyperparameters(prior, point.coefficients, weights)
    return ReferenceFit(
        prior=prior,
        posterior_mean=point.state.posterior_mean,
        posterior_variance=point.state.posterior_variance,
        hyperparameters=hyperparameters,
        evidence=point.value,
        ep_log_evidence=point.state.log_evidence,
        mixing_density=mixing_density(prior, hyperparameters.mixing_coefficients),
        sites=point.state,
        cavity_precision=point.state.cavity_precision,
        cavity_shift=point.state.cavity_shift,
        newton_decrement=newton_decrement(prior, weights, point.coefficients, point.state),
    )


def fits_agree(first: ReferenceFit, second: ReferenceFit, tolerance: float) -> bool:
    """Evidence within tolerance (nats), and posterior means and variances within tolerance relative."""
    mean_scale = float(np.max(np.abs(second.posterior_mean)))
    return (
        abs(first.evidence - second.evidence) <= tolerance
        and float(np.max(np.abs(first.posterior_mean - second.posterior_mean))) <= tolerance * mean_scale
        and float(np.max(np.abs(first.posterior_variance / second.posterior_variance - 1.0))) <= tolerance
    )


def quadrature_error_bound(prior, hyperparameters, cavity_precision, cavity_shift) -> float:
    """A certified bound on max_j |error of log Z_j| from the composite Gauss–Legendre rule.

    On a panel of half-length a, n Gauss–Legendre nodes integrate a function analytic in
    the Bernstein ellipse E_ρ, with |f| ≤ M there, to within a(64/15)Mρ^(−2n)/(ρ² − 1)
    (Trefethen, Approximation Theory and Approximation Practice, Thm 19.3). Here E_ρ has
    semi-minor axis STRIP_HALF_WIDTH in t, so ρ = β + √(1 + β²) with β = STRIP_HALF_WIDTH/a.
    The bound is applied to ∫e^φ L_j dt and to ∫e^φ dt, whose relative errors add in log Z_j;
    M is the maximum of |f| over the ellipse, sampled densely.
    """
    mixing = prior.mixing
    panel_half = 0.5 * (mixing.panel_edges[1] - mixing.panel_edges[0])
    reach = STRIP_HALF_WIDTH / panel_half
    rho = reach + np.sqrt(1.0 + reach**2)
    log_factor = float(np.log(panel_half * 64.0 / 15.0 / (rho**2 - 1.0)) - 2.0 * mixing.nodes_per_panel * np.log(rho))
    angles = np.linspace(0.0, 2.0 * np.pi, 257)[:-1]
    ellipse = 0.5 * (rho * np.exp(1j * angles) + np.exp(-1j * angles) / rho)
    centres = 0.5 * (mixing.panel_edges[:-1] + mixing.panel_edges[1:])
    complex_nodes = (centres[:, None] + panel_half * ellipse[None, :]).ravel()
    complex_design = _chebyshev_columns(mixing.standard_coordinate(complex_nodes), mixing.degree)
    log_scale = prior.log_variance_offset + prior.centred_design @ hyperparameters.annotation_coefficients
    shift_square = np.square(cavity_shift)
    worst = 0.0
    for class_position in range(prior.class_count):
        coefficients = hyperparameters.mixing_coefficients[class_position]
        members = np.flatnonzero(prior.class_index == class_position)
        log_density_nodes = mixing.log_weights + mixing.design @ coefficients
        log_density_ellipse = (complex_design @ coefficients).reshape(centres.shape[0], -1)
        normalizer_error = float(logsumexp(np.max(log_density_ellipse.real, axis=1)) + log_factor - logsumexp(log_density_nodes))
        variance_nodes = np.exp(log_scale[members, None] + mixing.nodes[None, :])
        log_kernel_nodes = _log_component_normalizer(variance_nodes, cavity_precision[members, None], shift_square[members, None])
        exact = logsumexp(log_density_nodes[None, :] + log_kernel_nodes, axis=1)
        variance_ellipse = np.exp(log_scale[members, None] + complex_nodes[None, :])
        relative = 1.0 + variance_ellipse * cavity_precision[members, None]
        log_kernel_ellipse = -0.5 * np.log(relative) + 0.5 * shift_square[members, None] * variance_ellipse / relative
        panel_maxima = np.max(
            (log_density_ellipse.ravel()[None, :] + log_kernel_ellipse).real.reshape(members.size, centres.shape[0], -1), axis=2
        )
        mixture_error = logsumexp(panel_maxima, axis=1) + log_factor - exact
        worst = max(worst, float(np.max(np.exp(mixture_error))) + float(np.exp(normalizer_error)))
    return worst


class NoResolvableSignal(ArithmeticError):
    """The null model (every effect exactly zero, V = 0) has more evidence than every continuous-prior fit."""


def point_prior(design: ReferenceDesign, level: float, tolerance: float) -> ReferencePrior:
    """The width → 0 end of the roughness's null space: every class's density is one point at log
    variance ``level``, so β_j ~ N(0, u_j e^level), a Gaussian effect prior, still continuous in β."""
    return ReferencePrior(design=design, mixing=MixingQuadrature(lower=level - 1.0, upper=level + 1.0, degree=0, nodes_per_panel=1), tolerance=tolerance)


def single_variance_fit(design: ReferenceDesign, likelihood_precision, linear_term, tolerance: float) -> ReferenceFit:
    """The best Gaussian-effect model: V at the one-point density, maximized over its level.

    It is the limit of the continuous model as η's width goes to 0 along the roughness's null
    space: the density's shape then carries no information, so its penalized directions
    cancel in V. The level is searched over the window the marginal estimates span (from the
    smallest sampling variance to the largest squared estimate), widened by its length at each end.
    """
    precision = np.diag(likelihood_precision)
    scale = np.exp(design.log_variance_offset)
    window = np.log([float(np.min(1.0 / (precision * scale))), float(np.max(np.square(linear_term / precision) / scale))])
    lower, upper = float(np.min(window)), float(np.max(window))
    length = max(upper - lower, MAXIMUM_PANEL_LENGTH)

    def fit_at(level: float) -> ReferenceFit:
        prior = point_prior(design, level, tolerance)
        start = np.zeros(prior.coefficient_count)
        sites = prior_sites(prior, start, likelihood_precision, linear_term)
        return fit_fixed_point(prior, likelihood_precision, linear_term, start, np.ones(penalty_count(prior)), sites)

    # The level is located to Δm with ½|V''|Δm² ≤ tolerance, V'' measured at the located maximum; the
    # search tightens until the measured curvature confirms it.
    resolution = np.sqrt(tolerance)
    while True:
        search = minimize_scalar(
            lambda level: -fit_at(level).evidence, bounds=(lower - length, upper + length), method="bounded", options={"xatol": resolution}
        )
        best = fit_at(float(search.x))
        sides = [fit_at(float(search.x) + sign * resolution).evidence for sign in (1.0, -1.0)]
        curvature = abs(sides[0] + sides[1] - 2.0 * best.evidence) / resolution**2
        if 0.5 * curvature * resolution**2 <= tolerance:
            return best
        resolution = np.sqrt(2.0 * tolerance / curvature)


class RangeNotConverged(ArithmeticError):
    """Doubling the range kept moving the fit by more than the tolerance: the fit depends on the range."""


def fit_reference(design: ReferenceDesign, likelihood_precision, linear_term, tolerance: float) -> ReferenceFit:
    """The converged EP-EB fixed point, certified against its representation.

    - The range covers derived_log_variance_range at the fit's own cavities and θ (extended
      when the fit moves it), starting from the likelihood's cavities.
    - The nodes per panel are doubled until the certified quadrature error of every log Z_j
      is ≤ tolerance/(p + 1), and the degree until the fit changes by less than the tolerance.
    - Range invariance: the range is doubled (by half its length at each end, the degree
      doubled with it) until that moves the fit by less than the tolerance; if it never does,
      RangeNotConverged is raised.
    - The null space's Gaussian-effect and null ends compete by V.
    """
    # The matrices here are small, and threaded BLAS makes each small LAPACK call orders of magnitude slower.
    with threadpool_limits(limits=1):
        candidates = []
        try:
            candidates.append(_fit_reference(design, likelihood_precision, linear_term, tolerance))
        except NoInteriorOptimum:
            pass
        # The null space's width → 0 end (a Gaussian effect prior) and its location → −∞ end
        # (the null model, V = 0) are models too, compared by V.
        candidates.append(single_variance_fit(design, likelihood_precision, linear_term, tolerance))
        if not candidates:
            raise NoInteriorOptimum("No penalty weights give an interior optimum.")
        best = candidates[0]
        for candidate in candidates[1:]:
            if prefer(_as_point(candidate), _as_point(best), likelihood_precision, linear_term):
                best = candidate
        if not prefer(_as_point(best), 0.0, likelihood_precision, linear_term):
            raise NoResolvableSignal(f"The null model's evidence 0 is not below the best fit's {best.evidence:.6g}.")
        return best


def _as_point(fit: "ReferenceFit") -> EvidencePoint:
    return EvidencePoint(
        value=fit.evidence,
        coefficients=coefficient_vector(fit.hyperparameters),
        state=fit.sites,
        weight_gradient=np.full(penalty_count(fit.prior), np.nan),
        prior=fit.prior,
        weights=_weights(fit.hyperparameters),
    )


def _fit_reference(design: ReferenceDesign, likelihood_precision, linear_term, tolerance: float) -> ReferenceFit:
    variant_count = linear_term.shape[0]
    lower, upper = derived_log_variance_range(
        design, np.zeros(design.feature_count), np.diag(likelihood_precision).copy(), linear_term.copy(), tolerance
    )
    degree = INITIAL_DEGREE
    nodes_per_panel = INITIAL_NODES_PER_PANEL
    prior = ReferencePrior(design=design, mixing=mixing_quadrature(lower, upper, degree, nodes_per_panel), tolerance=tolerance)
    start = np.zeros(prior.coefficient_count)
    fit = _fit_from(prior, likelihood_precision, linear_term, start, None, prior_sites(prior, start, likelihood_precision, linear_term))
    previous = None
    doublings = 0
    for _refinement in range(60):
        derived_lower, derived_upper = derived_log_variance_range(
            design, fit.hyperparameters.annotation_coefficients, fit.cavity_precision, fit.cavity_shift, tolerance
        )
        if derived_lower < lower or derived_upper > upper:
            lower, upper = min(lower, derived_lower), max(upper, derived_upper)
            previous = None
        elif quadrature_error_bound(fit.prior, fit.hyperparameters, fit.cavity_precision, fit.cavity_shift) > tolerance / (variant_count + 1):
            nodes_per_panel *= 2
            previous = None
        elif previous is None or not fits_agree(previous, fit, tolerance):
            previous = fit
            degree *= 2
        else:
            length = upper - lower
            wide = ReferencePrior(design=design, mixing=mixing_quadrature(lower - 0.5 * length, upper + 0.5 * length, 2 * degree, nodes_per_panel), tolerance=tolerance)
            wide_fit = _fit_from(wide, likelihood_precision, linear_term, _project(fit, wide), _weights(fit.hyperparameters), fit.sites)
            if fits_agree(fit, wide_fit, tolerance):
                return fit
            doublings += 1
            if doublings > MAXIMUM_RANGE_DOUBLINGS:
                raise RangeNotConverged(
                    f"Doubling the range {doublings} times still moved the evidence by {abs(wide_fit.evidence - fit.evidence):.3g} nats."
                )
            lower, upper, degree, fit, previous = wide.mixing.lower, wide.mixing.upper, 2 * degree, wide_fit, None
            continue
        prior = ReferencePrior(design=design, mixing=mixing_quadrature(lower, upper, degree, nodes_per_panel), tolerance=tolerance)
        fit = _fit_from(prior, likelihood_precision, linear_term, _project(fit, prior), _weights(fit.hyperparameters), fit.sites)
    raise AssertionError("The reference's range and resolution did not converge.")


def _fit_from(prior, likelihood_precision, linear_term, start, weights, sites) -> ReferenceFit:
    """The fixed point from a start: the carried penalty weights first (a refinement's warm start),
    then weights at successive decades of the unit prior precision, strongest smoothing first,
    from the flat start. The weights only start the search, whose answer does not depend on them."""
    attempts = [] if weights is None else [(start, weights, sites)]
    flat = np.zeros(prior.coefficient_count)
    flat_sites = prior_sites(prior, flat, likelihood_precision, linear_term)
    attempts += [(flat, np.full(penalty_count(prior), 10.0**exponent), flat_sites) for exponent in range(0, 7, 2)]
    for attempt_start, attempt_weights, attempt_sites in attempts:
        try:
            return fit_fixed_point(prior, likelihood_precision, linear_term, attempt_start, attempt_weights, attempt_sites)
        except (NoInteriorOptimum, EPFailure):
            continue
    raise NoInteriorOptimum("No starting penalty weights give an interior optimum.")
