"""Dense reference for SV-PGS's EP-EB fixed point, for exactness tests of the fast stages.

This module is test infrastructure: it shares no code with the fitting stages,
so a stage that agrees with it has the model right, not just its own algebra.
It is dense (p x p inverses), so it is only for problems of a few hundred variants.

Model. The data enter as one Gaussian factor exp(-1/2 βᵀΛβ + ℓᵀβ): for a
quantitative trait in LD space Λ = κ nR and ℓ = κ X̃ᵀr; for a binary trait's
working model Λ = w̄ nR and ℓ is the working score. Each effect has the prior

    β_j ~ ∫ N(0, u_j e^t) g_c(j)(t) dt,   log u_j = o_j + d̃_jᵀθ,

a continuous Gaussian scale mixture over the log variance t, with no point mass at zero.
- g_c = exp(φ_c) / ∫exp(φ_c) is class c's mixing density, a continuous function of t
  learned nonparametrically. Its roughness ∫(φ_c''')² dt carries a learned weight λ_c.
  Third order is the one order whose null space is a proper model on the whole line:
  λ_c = ∞ leaves φ_c quadratic, so g_c is normal in log s (a normal–log-normal scale
  mixture). The null spaces of first and second order, flat and power-law log-densities,
  have no normalizable member on ℝ, so their limits depend on where the range is cut.
  A linear log-tail (a power law in s, as in TPB) is free at every λ_c; only bends cost.
- o_j is the fixed measurement offset log r²_j: the prior is on the effect of the true
  genotype, so the observed column's prior variance carries r²_j with coefficient 1.
- d̃_j is the annotation row centred within its class. The class means are absorbed by
  g_c's location, which is learned; there is no separate class level to confound with it.
  θ is split into groups, each with a penalty matrix S_g (the identity for a discrete
  annotation, second differences for a smooth one) and a learned weight λ_g.

Computation, none of which is part of the model. φ_c is represented on a range
[t_lo, t_hi] of log variance in a Chebyshev basis and extended beyond it by its
quadratic Taylor polynomial, which adds no roughness; every t-integral is Gauss–Legendre.
fit_reference extends the range until the mass the extension puts outside it moves
every evidence term by less than the tolerance, and doubles the basis degree (with a
node count from the analyticity of the integrand) until the evidence and the posterior
moments change by less than the tolerance. So the fit does not depend on the grid.

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
      V(λ) = log Z_EP(x̂) − ½x̂ᵀS_λx̂ + ½log|S_λ|₊ − ½log|B(x̂) + S_λ| + ½(dim − rank) log 2π
  directly, with its exact gradient (which carries dx̂/dλ through D_xB).
- Each weight ranges over (0, ∞]. V → −∞ as λ → 0 (the prior on the penalized directions
  becomes flat), and V extends continuously to λ = ∞, where the term lies exactly in its
  penalty's null space: g_c normal in log s, a discrete annotation's coefficient 0, a
  smooth annotation's effect linear. V is compared at ∞ and inside for every weight, and a
  weight sits at ∞ when V is highest there and the one-sided derivative of V in 1/λ is ≤ 0
  (the score test, EdgeScore).

The fixed point is where EP, x and λ are all stationary (λ at its KKT point). It does not
depend on the path to it, which is what lets a stage be compared with it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import chebyshev, legendre
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp

# The reference's own error budget: 100x below the relative 1e-6 at which stages are gated.
QUADRATURE_TOLERANCE = 1e-8
SITE_TOLERANCE = 1e-12
COEFFICIENT_TOLERANCE = 1e-10
EVIDENCE_GRADIENT_TOLERANCE = 1e-9
EDGE_SCORE_TOLERANCE = 1e-9
INITIAL_DEGREE = 8
COEFFICIENT_STEP_LIMIT = 2.0
# The roughness penalty's order, derived rather than chosen: its null space (polynomials of
# degree ROUGHNESS_ORDER − 1 in log s) is the λ = ∞ limit of g, which must be a proper
# density on the whole line. Degree 0 and 1 (flat, power law) are never normalizable and
# degree 3 only when its cubic term vanishes; degree 2 with a negative leading term is the
# normal density in log s. So the order is 3.
ROUGHNESS_ORDER = 3
LOG_TWO_PI = float(np.log(2.0 * np.pi))


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
    """A continuous log-density φ(t) = Σ_m c_m T_m(x(t)) on [lower, upper], and a composite
    Gauss–Legendre rule on equal panels of length ≤ MAXIMUM_PANEL_LENGTH."""

    lower: float
    upper: float
    degree: int
    nodes_per_panel: int
    panel_edges: np.ndarray = field(init=False)
    nodes: np.ndarray = field(init=False)
    log_weights: np.ndarray = field(init=False)
    design: np.ndarray = field(init=False)
    roughness: np.ndarray = field(init=False)
    boundary_jets: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        half_length = 0.5 * (self.upper - self.lower)
        panel_count = int(np.ceil((self.upper - self.lower) / MAXIMUM_PANEL_LENGTH))
        edges = np.linspace(self.lower, self.upper, panel_count + 1)
        panel_half = 0.5 * (edges[1] - edges[0])
        standard_nodes, standard_weights = legendre.leggauss(self.nodes_per_panel)
        nodes = (0.5 * (edges[:-1] + edges[1:])[:, None] + panel_half * standard_nodes[None, :]).ravel()
        # ∫(φ⁽ᵐ⁾)² dt with m = ROUGHNESS_ORDER: the integrand is a polynomial of degree
        # 2(degree − m), so `degree` Gauss–Legendre nodes integrate it exactly.
        roughness_nodes, roughness_weights = legendre.leggauss(max(self.degree, 2))
        derivative = _chebyshev_columns(roughness_nodes, self.degree, ROUGHNESS_ORDER) / half_length**ROUGHNESS_ORDER
        ends = np.array([-1.0, 1.0])
        object.__setattr__(self, "panel_edges", edges)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "log_weights", np.tile(np.log(standard_weights * panel_half), panel_count))
        object.__setattr__(self, "design", _chebyshev_columns(self.standard_coordinate(nodes), self.degree))
        object.__setattr__(self, "roughness", derivative.T @ ((roughness_weights * half_length)[:, None] * derivative))
        # The t-derivatives of φ below the penalty's order at (lower end, upper end): the
        # Taylor polynomial they give continues φ past the range with no roughness.
        object.__setattr__(
            self,
            "boundary_jets",
            np.stack([_chebyshev_columns(ends, self.degree, order) / half_length**order for order in range(ROUGHNESS_ORDER)]),
        )

    @property
    def node_count(self) -> int:
        return int(self.nodes.shape[0])

    def standard_coordinate(self, log_variance):
        return (log_variance - self.lower) / (0.5 * (self.upper - self.lower)) - 1.0

    def log_density_at(self, coefficients: np.ndarray, log_variance: np.ndarray) -> np.ndarray:
        """φ(t) inside the range, and its Taylor continuation outside (unnormalized)."""
        inside = np.clip(log_variance, self.lower, self.upper)
        value = _chebyshev_columns(self.standard_coordinate(inside), self.degree) @ coefficients
        jets = self.boundary_jets @ coefficients
        for end, edge in ((0, self.lower), (1, self.upper)):
            outside = (log_variance < self.lower) if end == 0 else (log_variance > self.upper)
            offset = log_variance[outside] - edge
            value[outside] = sum(jets[order, end] * offset**order / math.factorial(order) for order in range(ROUGHNESS_ORDER))
        return value


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
    """The design together with the representation of the mixing densities."""

    design: ReferenceDesign
    mixing: MixingQuadrature

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
    _penalty_terms: η's roughness, each δ_c's roughness, the δ_c null-space precision μ,
    then the annotation groups. Each lies in (0, ∞]; ∞ puts the term in its null space.
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

    - The roughness of η, the shared log-density.
    - The roughness of each δ_c, on its coefficients beyond T_1 and T_2: those two span the
      roughness's null space, a class's shift and width relative to η.
    - The shift and width coordinates of every δ_c together, under one learned precision μ,
      so a class moves away from η in location or width only as far as the classes do.
    - The annotation groups.

    Every term acts on its own columns, so each block has one penalty.
    """
    degree = prior.degree
    terms = [(np.arange(degree), prior.mixing.roughness)]
    for class_position in range(prior.class_count):
        start = (class_position + 1) * degree
        terms.append((np.arange(start + 2, start + degree), prior.mixing.roughness[2:, 2:]))
    level_columns = np.concatenate(
        [np.arange((position + 1) * degree, (position + 1) * degree + 2) for position in range(prior.class_count)]
    )
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


def allowed_basis(prior: ReferencePrior, weights: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the coefficients allowed when every infinite weight's term is in its null space."""
    columns_used = np.zeros(prior.coefficient_count, dtype=bool)
    pieces = []
    for (columns, matrix), weight in zip(_penalty_terms(prior), weights):
        basis = _null_space(matrix) if np.isinf(weight) else np.eye(columns.size)
        embedded = np.zeros((prior.coefficient_count, basis.shape[1]))
        embedded[columns] = basis
        pieces.append(embedded)
        columns_used[columns] = True
    pieces.append(np.eye(prior.coefficient_count)[:, ~columns_used])
    return np.hstack(pieces)


def penalty_matrix(prior: ReferencePrior, weights: np.ndarray) -> np.ndarray:
    """S_λ from the finite weights (an infinite weight acts through allowed_basis instead)."""
    penalty = np.zeros((prior.coefficient_count, prior.coefficient_count))
    for (columns, matrix), weight in zip(_penalty_terms(prior), weights):
        if np.isfinite(weight):
            penalty[np.ix_(columns, columns)] += weight * matrix
    return penalty


def _penalty_log_determinant(prior, weights, basis) -> tuple[float, int, np.ndarray]:
    """log|BᵀS_λB|₊, its rank, and its pseudo-inverse, with the rank taken from the unweighted terms."""
    restricted = basis.T @ penalty_matrix(prior, weights) @ basis
    unweighted = sum(
        basis.T @ _embedded(prior, columns, matrix) @ basis
        for (columns, matrix), weight in zip(_penalty_terms(prior), weights)
        if np.isfinite(weight)
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


def double_loop_sites(prior, vector, likelihood_precision, linear_term, start: SiteState) -> SiteState:
    """EP by the Opper–Winther double loop, which provably reaches a stationary point.

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
        if change < 1e-8:
            state = site_state(prior, vector, likelihood_precision, linear_term, site_precision, site_shift)
            if state is None:
                raise EPFailure("The double loop left the EP domain.")
            return state
    raise EPFailure("The EP double loop did not converge.")


def solve_sites(prior, vector, likelihood_precision, linear_term, start: SiteState) -> SiteState:
    """EP at fixed hyperparameters: Newton on the moment-matching equations R(s) = 0 from the start.

    Whenever a Newton step cannot lower the residual, the Opper–Winther double loop
    (provably convergent) takes over and Newton polishes its result, so the answer is the
    EP stationary point with residual ≤ SITE_TOLERANCE whichever path reached it.
    """
    state = site_state(prior, vector, likelihood_precision, linear_term, start.site_precision, start.site_shift)
    if state is None:
        state = double_loop_sites(prior, vector, likelihood_precision, linear_term, start)
    size = linear_term.shape[0]
    fallback_used = False
    for _iteration in range(100):
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
        if accepted is None:
            if fallback_used:
                raise EPFailure(f"EP stalled at moment residual {state.residual:.2e}.")
            state = double_loop_sites(prior, vector, likelihood_precision, linear_term, state)
            fallback_used = True
            continue
        state = accepted
    raise EPFailure("EP Newton did not converge.")


def initial_sites(prior: ReferencePrior, vector, likelihood_precision, linear_term, site_precision) -> SiteState:
    """The EP solution reached from sites with the given precision and no shift."""
    site_precision = np.array(site_precision, dtype=np.float64)
    start = site_state(prior, vector, likelihood_precision, linear_term, site_precision, np.zeros(linear_term.shape[0]))
    return solve_sites(prior, vector, likelihood_precision, linear_term, start)


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


def maximize_penalized_evidence(prior, weights, start, state: SiteState, likelihood_precision, linear_term) -> tuple[np.ndarray, SiteState]:
    """x̂ = argmax log Z_EP(x) − ½xᵀS_λx over the allowed coefficients, by Newton with the total curvature."""
    basis = allowed_basis(prior, weights)
    penalty = penalty_matrix(prior, weights)
    vector = basis @ (basis.T @ start)
    state = solve_sites(prior, vector, likelihood_precision, linear_term, state)

    def objective(point_state, point):
        return point_state.log_evidence - 0.5 * float(point @ penalty @ point)

    value = objective(state, vector)
    for _iteration in range(200):
        _unused, gradient = cavity_log_marginal(prior, vector, state.cavity_precision, state.cavity_shift)
        restricted_gradient = basis.T @ (gradient - penalty @ vector)
        hessian = basis.T @ (total_curvature(prior, vector, state) + penalty) @ basis
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        floor = 1e-8 * float(np.max(np.abs(eigenvalues)))
        # Modified Newton: negative curvature is taken with its magnitude, so the step ascends.
        step = eigenvectors @ ((eigenvectors.T @ restricted_gradient) / np.maximum(np.abs(eigenvalues), floor))
        # A trust region on the coefficients (a numerical safeguard; it does not move x̂).
        step *= min(1.0, COEFFICIENT_STEP_LIMIT / max(float(np.max(np.abs(basis @ step))), 1e-300))
        decrement = 0.5 * float(restricted_gradient @ step)
        if decrement < 1e-13 * (1.0 + abs(value)):
            if np.all(eigenvalues > floor) and decrement < 1e-20 * (1.0 + abs(value)):
                return vector, state
            # Quadratic regime: judge full Newton steps by the gradient, not the rounded value.
            candidate = basis @ (basis.T @ vector + step)
            candidate_state = solve_sites(prior, candidate, likelihood_precision, linear_term, state)
            _unused, candidate_gradient = cavity_log_marginal(prior, candidate, candidate_state.cavity_precision, candidate_state.cavity_shift)
            if not np.max(np.abs(basis.T @ (candidate_gradient - penalty @ candidate))) < np.max(np.abs(restricted_gradient)):
                return vector, state
            vector, state = candidate, candidate_state
            value = objective(state, vector)
            continue
        scale = 1.0
        for _halving in range(60):
            candidate = basis @ (basis.T @ vector + scale * step)
            try:
                candidate_state = solve_sites(prior, candidate, likelihood_precision, linear_term, state)
            except EPFailure:
                scale *= 0.5
                continue
            candidate_value = objective(candidate_state, candidate)
            if candidate_value >= value + 1e-4 * scale * 2.0 * decrement:
                break
            scale *= 0.5
        else:
            return vector, state
        vector, state, value = candidate, candidate_state, candidate_value
    raise AssertionError("The coefficient maximization did not converge.")


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
    """V at some weights, x̂ and the EP solution there, and dV/dλ_i for every finite weight (nan otherwise)."""

    value: float
    coefficients: np.ndarray
    state: SiteState
    weight_gradient: np.ndarray


def laplace_evidence(prior, weights, start, state, likelihood_precision, linear_term, with_gradient: bool = True) -> EvidencePoint:
    """V(λ) = log Z_EP(x̂) − ½x̂ᵀS_λx̂ + ½log|S_λ|₊ − ½log|B + S_λ| + ½(dim − rank) log 2π, B the total curvature.

    dV/dλ_i = −½x̂ᵀS_ix̂ + ½tr(S_λ⁺S_i) − ½tr((B + S_λ)⁻¹(S_i + D_xB[dx̂/dλ_i])), with
    dx̂/dλ_i = −(B + S_λ)⁻¹S_ix̂ from the stationarity of x̂.
    """
    weights = np.asarray(weights, dtype=np.float64)
    basis = allowed_basis(prior, weights)
    penalty = penalty_matrix(prior, weights)
    coefficients, state = maximize_penalized_evidence(prior, weights, start, state, likelihood_precision, linear_term)
    curvature = total_curvature(prior, coefficients, state)
    restricted_hessian = basis.T @ (curvature + penalty) @ basis
    try:
        factor = cho_factor(restricted_hessian)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "The Laplace evidence is undefined: log Z_EP − ½xᵀS_λx is not strictly concave at x̂. A direction the "
            "penalty leaves free is unidentified, e.g. a mixing density collapsing to a single variance "
            "(the width of the normal-in-log-s null space going to 0)."
        ) from error
    log_determinant_hessian = 2.0 * float(np.sum(np.log(np.diag(factor[0]))))
    log_determinant_penalty, rank, penalty_inverse = _penalty_log_determinant(prior, weights, basis)
    evidence = (
        state.log_evidence
        - 0.5 * float(coefficients @ penalty @ coefficients)
        + 0.5 * log_determinant_penalty
        - 0.5 * log_determinant_hessian
        + 0.5 * (basis.shape[1] - rank) * LOG_TWO_PI
    )
    gradient = np.full(weights.shape[0], np.nan)
    if with_gradient:
        hessian_inverse = cho_solve(factor, np.eye(basis.shape[1]))
        for position, ((columns, matrix), weight) in enumerate(zip(_penalty_terms(prior), weights)):
            if not np.isfinite(weight):
                continue
            term = _embedded(prior, columns, matrix)
            restricted_term = basis.T @ term @ basis
            coefficient_derivative = -basis @ (hessian_inverse @ (basis.T @ (term @ coefficients)))
            curvature_derivative = _curvature_derivative(prior, coefficients, coefficient_derivative, state, likelihood_precision, linear_term)
            gradient[position] = (
                -0.5 * float(coefficients @ term @ coefficients)
                + 0.5 * float(np.sum(penalty_inverse * restricted_term))
                - 0.5 * float(np.sum(hessian_inverse * (restricted_term + basis.T @ curvature_derivative @ basis)))
            )
    return EvidencePoint(value=float(evidence), coefficients=coefficients, state=state, weight_gradient=gradient)


@dataclass(frozen=True)
class EdgeScore:
    """At λ_i = ∞: dV/d(1/λ_i) = ½(q − d + c).

    q = GᵀE⁻¹G is the squared score and d = tr(E⁻¹B_{U|K}) the information on the released
    directions; c = −tr(H_KK⁻¹ Kᵀ D_xB[v] K) is the change of the kept curvature as x̂ moves
    along v = dx̂/d(1/λ_i). c vanishes for a Gaussian likelihood, where this is Tipping–Faul.
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
    """The score test for releasing weight ``position`` from ∞.

    With x = Kz + Uy (K the current allowed basis, U the directions the release adds) and
    y ~ N(0, τE⁻¹), τ = 1/λ, E = UᵀS_iU, expanding the Laplace evidence in τ gives
        V(τ) = V(0) + ½τ(GᵀE⁻¹G − tr(E⁻¹B_{U|K}) − tr(H_KK⁻¹ KᵀD_xB[v]K)) + O(τ²),
    where G = Uᵀ∇log Z_EP, B_{U|K} = B_UU − B_UK H_KK⁻¹ B_KU is the curvature in the released
    directions given the rest, and v = (U − K H_KK⁻¹B_KU) E⁻¹G is dx̂/dτ at τ = 0 (the released
    coordinates move by τE⁻¹G, the kept ones re-optimize). No other term penalizes a released
    direction (each block has one penalty), so S_λ enters only through H_KK.
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
    information_matrix = added.T @ _embedded(prior, columns, matrix) @ added
    coefficients, state = point.coefficients, point.state
    _unused, gradient = cavity_log_marginal(prior, coefficients, state.cavity_precision, state.cavity_shift)
    curvature = total_curvature(prior, coefficients, state)
    kept_hessian = basis.T @ (curvature + penalty_matrix(prior, weights)) @ basis
    coupling = added.T @ curvature @ basis
    conditional = added.T @ curvature @ added - coupling @ np.linalg.solve(kept_hessian, coupling.T)
    released_gradient = added.T @ gradient
    released_step = np.linalg.solve(information_matrix, released_gradient)
    direction = added @ released_step - basis @ np.linalg.solve(kept_hessian, coupling.T @ released_step)
    curvature_derivative = _curvature_derivative(prior, coefficients, direction, state, likelihood_precision, linear_term)
    return EdgeScore(
        squared_score=float(released_gradient @ released_step),
        information=float(np.trace(np.linalg.solve(information_matrix, conditional))),
        curvature_change=-float(np.trace(np.linalg.solve(kept_hessian, basis.T @ curvature_derivative @ basis))),
    )


def _ascend(prior, weights, point, likelihood_precision, linear_term):
    """Quasi-Newton ascent of V in log λ over the finite weights (BFGS from a central-difference
    Hessian), taking a weight to ∞ exactly wherever V is highest there."""

    def evaluate(candidate_weights, start_point, with_gradient=True):
        return laplace_evidence(
            prior, candidate_weights, start_point.coefficients, start_point.state, likelihood_precision, linear_term, with_gradient
        )

    inverse_hessian = None
    free_before = None
    for _iteration in range(500):
        free = np.flatnonzero(np.isfinite(weights))
        if free.size == 0:
            return weights, point
        best = None
        # V can have separate maxima inside and at ∞, so every finite weight's edge is compared.
        for position in free:
            candidate = weights.copy()
            candidate[position] = np.inf
            candidate_point = evaluate(candidate, point)
            if candidate_point.value >= point.value and (best is None or candidate_point.value > best[1].value):
                if edge_score(prior, candidate, position, candidate_point, likelihood_precision, linear_term).holds:
                    best = (candidate, candidate_point)
        if best is not None:
            weights, point = best
            inverse_hessian = None
            continue
        log_gradient = weights[free] * point.weight_gradient[free]
        if float(np.max(np.abs(log_gradient))) < EVIDENCE_GRADIENT_TOLERANCE:
            return weights, point
        if inverse_hessian is None or free_before is None or not np.array_equal(free, free_before):
            step = 1e-4
            hessian = np.empty((free.size, free.size))
            for column, position in enumerate(free):
                gradients = []
                for sign in (1.0, -1.0):
                    moved = weights.copy()
                    moved[position] = weights[position] * np.exp(sign * step)
                    gradients.append(moved[free] * evaluate(moved, point).weight_gradient[free])
                hessian[:, column] = (gradients[0] - gradients[1]) / (2.0 * step)
            eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (hessian + hessian.T))
            floor = 1e-10 * max(float(np.max(np.abs(eigenvalues))), 1e-12)
            inverse_hessian = eigenvectors @ np.diag(1.0 / np.maximum(np.abs(eigenvalues), floor)) @ eigenvectors.T
            free_before = free
        ascent = inverse_hessian @ log_gradient
        ascent *= min(1.0, 2.0 / float(np.max(np.abs(ascent))))
        for _halving in range(60):
            candidate = weights.copy()
            candidate[free] = weights[free] * np.exp(ascent)
            candidate_point = evaluate(candidate, point)
            if candidate_point.value >= point.value - 1e-13 * abs(point.value):
                break
            ascent *= 0.5
        else:
            return weights, point
        new_gradient = candidate[free] * candidate_point.weight_gradient[free]
        # BFGS on −V: s = Δlog λ, y = −Δ(dV/dlog λ); keep the update only when it stays positive definite.
        displacement = np.log(candidate[free]) - np.log(weights[free])
        change = -(new_gradient - log_gradient)
        curvature = float(displacement @ change)
        if curvature > 1e-12 * float(np.linalg.norm(displacement) * np.linalg.norm(change)):
            rho = 1.0 / curvature
            identity = np.eye(free.size)
            inverse_hessian = (identity - rho * np.outer(displacement, change)) @ inverse_hessian @ (
                identity - rho * np.outer(change, displacement)
            ) + rho * np.outer(displacement, displacement)
        weights, point = candidate, candidate_point
    raise RuntimeError("The penalty-weight ascent did not converge.")


def _release(prior, weights, position, point, score, likelihood_precision, linear_term):
    """Move weight ``position`` from ∞ to the finite value, among decades below the data scale, where V is highest."""
    columns, matrix = _penalty_terms(prior)[position]
    # The weight at which the prior precision matches the data's curvature on the term.
    scale = np.log(max(abs(score.information), 1e-300)) - np.log(np.trace(matrix) / columns.size)
    best = None
    for shift in np.arange(12.0, -12.5, -2.0):
        candidate = weights.copy()
        candidate[position] = np.exp(scale + shift)
        candidate_point = laplace_evidence(prior, candidate, point.coefficients, point.state, likelihood_precision, linear_term)
        if candidate_point.value > point.value and (best is None or candidate_point.value > best[1].value):
            best = (candidate, candidate_point)
    return best


def maximize_evidence(prior, weights, start, state, likelihood_precision, linear_term) -> tuple[np.ndarray, EvidencePoint]:
    """λ̂ = argmax V over (0, ∞] for every weight, with x̂ and the EP solution there."""
    weights = np.array(weights, dtype=np.float64)
    point = laplace_evidence(prior, weights, start, state, likelihood_precision, linear_term)
    for _round in range(100):
        weights, point = _ascend(prior, weights, point, likelihood_precision, linear_term)
        releases = []
        for position in np.flatnonzero(np.isinf(weights)):
            score = edge_score(prior, weights, position, point, likelihood_precision, linear_term)
            if not score.holds:
                released = _release(prior, weights, position, point, score, likelihood_precision, linear_term)
                if released is not None:
                    releases.append(released)
        if not releases:
            return weights, point
        weights, point = max(releases, key=lambda release: release[1].value)
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


def initial_log_variance_range(design: ReferenceDesign, likelihood_precision: np.ndarray, linear_term: np.ndarray) -> tuple[float, float]:
    """The window of log variance the data resolve, on the offset-free scale: from the
    smallest marginal sampling variance to the largest squared marginal estimate."""
    precision_diagonal = np.diag(likelihood_precision)
    scale = np.exp(design.log_variance_offset)
    ends = np.log(
        [
            float(np.min(1.0 / precision_diagonal)) / float(np.max(scale)),
            float(np.max(np.square(linear_term / precision_diagonal))) / float(np.min(scale)),
        ]
    )
    return float(np.min(ends)), float(np.max(ends))


def _log_tail_integral(value: float, slope: float, curvature: float) -> float:
    """log ∫_0^∞ exp(value + slope·Δ + ½ curvature·Δ²) dΔ; +∞ when the integrand does not decay."""
    if curvature < 0.0:
        precision = -curvature
        return value + 0.5 * slope**2 / precision + 0.5 * np.log(2.0 * np.pi / precision) + float(log_ndtr(slope / np.sqrt(precision)))
    if curvature == 0.0 and slope < 0.0:
        return value - float(np.log(-slope))
    return np.inf


def tail_excess(prior, hyperparameters, cavity_precision, cavity_shift, tolerance) -> tuple[bool, bool]:
    """Whether the mass the quadratic extension of φ_c puts below / above the range is too large.

    With δ = tolerance / (p + 1), a tail is negligible when its normalized mass m ≤ δ and,
    for every variant of the class, m · max_tail Z_j(t) / Z_j ≤ δ: moving it changes the
    normalizer by ≤ δ and each log Z_j by ≤ δ, so the evidence by ≤ tolerance. Z_j(t) is
    unimodal in t (maximal at v* = (h² − P)/P² when h² > P), which gives its tail maximum.
    """
    log_bound = float(np.log(tolerance / (prior.class_index.shape[0] + 1)))
    terms = tilted_terms(prior, hyperparameters.mixing_coefficients, hyperparameters.annotation_coefficients, cavity_precision, cavity_shift)
    log_scale = prior.log_variance_offset + prior.centred_design @ hyperparameters.annotation_coefficients
    shift_square = np.square(cavity_shift)
    has_mode = shift_square > cavity_precision
    mode = np.where(has_mode, np.log(np.where(has_mode, shift_square - cavity_precision, 1.0) / np.square(cavity_precision)) - log_scale, -np.inf)
    excess = [False, False]
    for class_position in range(prior.class_count):
        coefficients = hyperparameters.mixing_coefficients[class_position]
        log_normalizer = float(logsumexp(prior.mixing.log_weights + prior.mixing.design @ coefficients))
        jets = prior.mixing.boundary_jets @ coefficients
        members = prior.class_index == class_position
        for end, (edge, direction) in enumerate(((prior.mixing.lower, -1.0), (prior.mixing.upper, 1.0))):
            curvature = jets[2, end] if jets.shape[0] > 2 else 0.0
            log_mass = _log_tail_integral(jets[0, end], direction * jets[1, end], curvature) - log_normalizer
            if end == 0:
                where = np.minimum(edge, mode[members])
            else:
                where = np.maximum(edge, mode[members])
            log_peak = np.where(
                np.isfinite(where),
                _log_component_normalizer(np.exp(log_scale[members] + np.where(np.isfinite(where), where, 0.0)), cavity_precision[members], shift_square[members]),
                0.0,
            )
            ratio = float(np.max(log_peak - terms["log_normalizer"][members]))
            if log_mass > log_bound or log_mass + max(ratio, 0.0) > log_bound:
                excess[end] = True
    return excess[0], excess[1]


def _project(fit: ReferenceFit, prior: ReferencePrior) -> np.ndarray:
    """The fitted parameters expressed on another range or degree: η and each δ_c evaluated
    (with their Taylor continuations) at the new nodes and refitted by least squares, the
    constant absorbed."""
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


def fit_reference(design: ReferenceDesign, likelihood_precision, linear_term, tolerance: float = QUADRATURE_TOLERANCE) -> ReferenceFit:
    """The converged EP-EB fixed point.

    The range is extended until both tails are negligible (tail_excess), the nodes per panel
    are doubled until the certified quadrature error of every log Z_j is ≤ tolerance/(p + 1),
    and the degree is doubled until the fit changes by less than the tolerance (fits_agree).
    """
    variant_count = linear_term.shape[0]
    lower, upper = initial_log_variance_range(design, likelihood_precision, linear_term)
    degree = INITIAL_DEGREE
    nodes_per_panel = INITIAL_NODES_PER_PANEL
    prior = ReferencePrior(design=design, mixing=mixing_quadrature(lower, upper, degree, nodes_per_panel))
    typical_variance = np.exp(design.log_variance_offset + 0.5 * (lower + upper))
    start = np.zeros(prior.coefficient_count)
    fit = fit_fixed_point(
        prior,
        likelihood_precision,
        linear_term,
        start,
        np.ones(penalty_count(prior)),
        initial_sites(prior, start, likelihood_precision, linear_term, 1.0 / typical_variance),
    )
    previous = None
    for _refinement in range(40):
        lower_excess, upper_excess = tail_excess(fit.prior, fit.hyperparameters, fit.cavity_precision, fit.cavity_shift, tolerance)
        if lower_excess or upper_excess:
            length = upper - lower
            lower -= 0.5 * length * lower_excess
            upper += 0.5 * length * upper_excess
            previous = None
        elif quadrature_error_bound(fit.prior, fit.hyperparameters, fit.cavity_precision, fit.cavity_shift) > tolerance / (variant_count + 1):
            nodes_per_panel *= 2
            previous = None
        elif previous is not None and fits_agree(previous, fit, tolerance):
            return fit
        else:
            previous = fit
            degree *= 2
        prior = ReferencePrior(design=design, mixing=mixing_quadrature(lower, upper, degree, nodes_per_panel))
        fit = fit_fixed_point(
            prior,
            likelihood_precision,
            linear_term,
            _project(fit, prior),
            _weights(fit.hyperparameters),
            fit.sites,
        )
    raise AssertionError("The reference's range and resolution did not converge.")
