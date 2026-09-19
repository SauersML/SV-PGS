"""Dense reference for SV-PGS's EP-EB fixed point, for exactness tests of the fast stages.

This module is test infrastructure: it shares no code with the fitting stages,
so a stage that agrees with it has the model right, not just its own algebra.
It is dense (one p x p inverse per sweep), so it is only for problems of a few
hundred variants.

Model. The data enter as one Gaussian factor exp(-1/2 βᵀΛβ + ℓᵀβ): for a
quantitative trait in LD space Λ = κ nR and ℓ = κ X̃ᵀr; for a binary trait's
working model Λ = w̄ nR and ℓ is the working score. Each effect has the prior

    β_j ~ Σ_k π_c(j),k N(0, u_j s_k),   log u_j = o_j + d̃_jᵀθ,

a continuous Gaussian scale mixture with no point mass at zero.
- s_k is a log-even grid of variances whose support is set by the data (see
  data_driven_variance_grid).
- π_c = softmax(φ_c) is class c's mixing density, learned nonparametrically. φ_c
  is kept sum-to-zero, and its roughness φ_cᵀSφ_c (squared second differences
  in log s) is penalized with a learned weight λ_c.
- o_j is the fixed measurement offset log r²_j: the prior is on the effect of
  the true genotype, so the observed column's prior variance carries r²_j with
  coefficient exactly 1.
- d̃_j is the annotation row centred within its class; the class means are the
  mixing density's location. θ is split into groups, each with a penalty
  matrix S_g (the identity for a discrete annotation, second differences for a
  smooth one) and a learned weight λ_g.

Inference.
- EP with one Gaussian site per effect, clipped (site precision ≥ 0) and
  mean-matched, with exact tilted moments on the grid.
- (φ, θ) maximize the penalized EP cavity marginal Σ_j log Z_j with the
  cavities held fixed (type-II ML under EP).
- Each penalty weight λ solves the Laplace marginal-likelihood stationarity
  condition (the MacKay / Fellner–Schall fixed point),
  λ = (rank S − λ tr(H⁻¹S)) / (x̂ᵀSx̂), with H the negative Hessian of the
  penalized objective.

The fixed point is where the sites, (φ, θ) and λ all stop moving. It does not
depend on damping or update order, which is what lets a stage be compared with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

# Data-driven grid resolution, as in ash: the smallest variance sits two decades
# below the finest marginal standard error squared, the largest at 4x the largest
# squared marginal estimate, and adjacent variances differ by 2^(1/2).
GRID_LOG_SPACING = 0.5 * np.log(2.0)
GRID_LOWER_FACTOR = 1e-2
GRID_UPPER_FACTOR = 4.0


def data_driven_variance_grid(likelihood_precision: np.ndarray, linear_term: np.ndarray, log_variance_offset: np.ndarray):
    """Log-even variances covering every effect the data resolve (on the offset-free scale)."""
    precision_diagonal = np.diag(likelihood_precision)
    marginal_estimate = linear_term / precision_diagonal
    offset_scale = np.exp(log_variance_offset)
    lower = GRID_LOWER_FACTOR * float(np.min(1.0 / precision_diagonal)) / float(np.max(offset_scale))
    upper = GRID_UPPER_FACTOR * float(np.max(np.square(marginal_estimate))) / float(np.min(offset_scale))
    upper = max(upper, 10.0 * lower)
    point_count = int(np.ceil((np.log(upper) - np.log(lower)) / GRID_LOG_SPACING)) + 1
    return np.linspace(np.log(lower), np.log(upper), point_count)


def second_difference_penalty(size: int) -> np.ndarray:
    difference = np.diff(np.eye(size), n=2, axis=0)
    return difference.T @ difference


def sum_to_zero_basis(size: int) -> np.ndarray:
    """Orthonormal basis of the vectors summing to zero (the softmax is invariant to a constant)."""
    centred = np.eye(size) - np.full((size, size), 1.0 / size)
    basis, _singular, _rows = np.linalg.svd(centred)
    return basis[:, : size - 1]


@dataclass(frozen=True)
class AnnotationGroup:
    columns: np.ndarray
    penalty: np.ndarray


@dataclass(frozen=True)
class ReferencePrior:
    """Everything about the prior except the fitted hyperparameters."""

    class_index: np.ndarray
    log_variance_offset: np.ndarray
    annotation_design: np.ndarray
    annotation_groups: tuple[AnnotationGroup, ...]
    log_variance_grid: np.ndarray
    centred_design: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        centred = np.array(self.annotation_design, dtype=np.float64, copy=True)
        for class_position in range(self.class_count):
            members = self.class_index == class_position
            centred[members] -= centred[members].mean(axis=0)
        object.__setattr__(self, "centred_design", centred)

    @property
    def class_count(self) -> int:
        return int(self.class_index.max()) + 1

    @property
    def grid_size(self) -> int:
        return int(self.log_variance_grid.shape[0])

    @property
    def feature_count(self) -> int:
        return int(self.annotation_design.shape[1])


@dataclass(frozen=True)
class ReferenceHyperparameters:
    """ψ (C x (K-1)), the sum-to-zero coordinates of the log mixing weights; θ; the penalty weights."""

    mixing_coordinates: np.ndarray
    annotation_coefficients: np.ndarray
    mixing_penalty: np.ndarray
    annotation_penalty: np.ndarray


@dataclass(frozen=True)
class ReferenceFit:
    posterior_mean: np.ndarray
    posterior_variance: np.ndarray
    hyperparameters: ReferenceHyperparameters
    mixing_density: np.ndarray
    site_precision: np.ndarray
    site_shift: np.ndarray
    cavity_precision: np.ndarray
    cavity_shift: np.ndarray
    newton_decrement: float
    outer_iterations: int


def mixing_density(prior: ReferencePrior, mixing_coordinates: np.ndarray) -> np.ndarray:
    log_weights = mixing_coordinates @ sum_to_zero_basis(prior.grid_size).T
    return np.exp(log_weights - logsumexp(log_weights, axis=1, keepdims=True))


def _pack(prior: ReferencePrior, mixing_coordinates: np.ndarray, annotation_coefficients: np.ndarray) -> np.ndarray:
    return np.concatenate([mixing_coordinates.ravel(), annotation_coefficients])


def _unpack(prior: ReferencePrior, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mixing_size = prior.class_count * (prior.grid_size - 1)
    return vector[:mixing_size].reshape(prior.class_count, prior.grid_size - 1), vector[mixing_size:]


def tilted_terms(
    prior: ReferencePrior,
    mixing_coordinates: np.ndarray,
    annotation_coefficients: np.ndarray,
    cavity_precision: np.ndarray,
    cavity_shift: np.ndarray,
) -> dict[str, np.ndarray]:
    """log Z_j, the tilted mean and variance, the responsibilities, and the scale derivative.

    Component k of the tilted density integrates to π_ck (1 + v_jk P_j)^(-1/2)
    exp(1/2 h_j² c_jk), with v_jk = u_j s_k and c_jk = v_jk / (1 + v_jk P_j); given
    k it is Gaussian with mean h_j c_jk and variance c_jk.
    """
    log_scale = prior.log_variance_offset + prior.centred_design @ annotation_coefficients
    variance = np.exp(log_scale[:, None] + prior.log_variance_grid[None, :])
    precision = cavity_precision[:, None]
    shift_square = np.square(cavity_shift)[:, None]
    relative = 1.0 + variance * precision
    conditional_variance = variance / relative
    log_density = np.log(mixing_density(prior, mixing_coordinates))[prior.class_index]
    log_component = log_density - 0.5 * np.log(relative) + 0.5 * shift_square * conditional_variance
    log_normalizer = logsumexp(log_component, axis=1)
    responsibility = np.exp(log_component - log_normalizer[:, None])
    tilted_mean = cavity_shift * np.sum(responsibility * conditional_variance, axis=1)
    tilted_second_moment = np.sum(responsibility * (conditional_variance + shift_square * np.square(conditional_variance)), axis=1)
    component_scale_derivative = variance * (-0.5 * precision / relative + 0.5 * shift_square / np.square(relative))
    return {
        "log_normalizer": log_normalizer,
        "responsibility": responsibility,
        "tilted_mean": tilted_mean,
        "tilted_variance": tilted_second_moment - np.square(tilted_mean),
        "scale_derivative": np.sum(responsibility * component_scale_derivative, axis=1),
    }


def _mixing_penalty_matrix(prior: ReferencePrior) -> np.ndarray:
    basis = sum_to_zero_basis(prior.grid_size)
    return basis.T @ second_difference_penalty(prior.grid_size) @ basis


def penalized_objective(
    prior: ReferencePrior,
    hyperparameters: ReferenceHyperparameters,
    vector: np.ndarray,
    cavity_precision: np.ndarray,
    cavity_shift: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Σ_j log Z_j minus the roughness and annotation penalties, at fixed cavities and penalty weights."""
    mixing_coordinates, annotation_coefficients = _unpack(prior, vector)
    terms = tilted_terms(prior, mixing_coordinates, annotation_coefficients, cavity_precision, cavity_shift)
    density = mixing_density(prior, mixing_coordinates)
    basis = sum_to_zero_basis(prior.grid_size)
    mixing_penalty_matrix = _mixing_penalty_matrix(prior)
    value = float(np.sum(terms["log_normalizer"]))
    # ∂/∂φ_ck of Σ_j log Σ_k π_ck Z_jk is Σ_{j in c} (w_jk - π_ck); ψ enters through φ = Bψ.
    responsibility_sums = np.zeros((prior.class_count, prior.grid_size))
    np.add.at(responsibility_sums, prior.class_index, terms["responsibility"])
    class_sizes = np.bincount(prior.class_index, minlength=prior.class_count).astype(np.float64)
    mixing_gradient = (responsibility_sums - class_sizes[:, None] * density) @ basis
    for class_position in range(prior.class_count):
        coordinates = mixing_coordinates[class_position]
        value -= 0.5 * hyperparameters.mixing_penalty[class_position] * float(coordinates @ mixing_penalty_matrix @ coordinates)
        mixing_gradient[class_position] -= hyperparameters.mixing_penalty[class_position] * (mixing_penalty_matrix @ coordinates)
    annotation_gradient = prior.centred_design.T @ terms["scale_derivative"]
    for group_position, group in enumerate(prior.annotation_groups):
        coefficients = annotation_coefficients[group.columns]
        weight = hyperparameters.annotation_penalty[group_position]
        value -= 0.5 * weight * float(coefficients @ group.penalty @ coefficients)
        annotation_gradient[group.columns] -= weight * (group.penalty @ coefficients)
    return value, np.concatenate([mixing_gradient.ravel(), annotation_gradient])


def _numerical_hessian(function, vector: np.ndarray) -> np.ndarray:
    step = 1e-5
    hessian = np.empty((vector.shape[0], vector.shape[0]))
    for coordinate in range(vector.shape[0]):
        forward = vector.copy()
        backward = vector.copy()
        forward[coordinate] += step
        backward[coordinate] -= step
        hessian[:, coordinate] = (function(forward)[1] - function(backward)[1]) / (2.0 * step)
    return 0.5 * (hessian + hessian.T)


def maximize_coefficients(prior, hyperparameters, start, cavity_precision, cavity_shift) -> np.ndarray:
    def objective(vector):
        return penalized_objective(prior, hyperparameters, vector, cavity_precision, cavity_shift)

    result = minimize(
        lambda vector: tuple(-part for part in objective(vector)),
        start,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 20_000, "ftol": 1e-16, "gtol": 1e-12},
    )
    vector = np.asarray(result.x, dtype=np.float64)
    for _newton_step in range(12):
        _value, gradient = objective(vector)
        step = np.linalg.solve(-_numerical_hessian(objective, vector), gradient)
        vector = vector + step
        if float(np.max(np.abs(step))) < 1e-12:
            break
    return vector


def update_penalties(prior, hyperparameters, vector, cavity_precision, cavity_shift) -> ReferenceHyperparameters:
    """One MacKay / Fellner–Schall step for every penalty weight at the current optimum."""

    def objective(candidate):
        return penalized_objective(prior, hyperparameters, candidate, cavity_precision, cavity_shift)

    inverse = np.linalg.inv(-_numerical_hessian(objective, vector))
    mixing_coordinates, annotation_coefficients = _unpack(prior, vector)
    mixing_penalty_matrix = _mixing_penalty_matrix(prior)
    mixing_rank = np.linalg.matrix_rank(mixing_penalty_matrix)
    block = prior.grid_size - 1
    mixing_penalty = hyperparameters.mixing_penalty.copy()
    for class_position in range(prior.class_count):
        span = slice(class_position * block, (class_position + 1) * block)
        coordinates = mixing_coordinates[class_position]
        trace = float(np.trace(inverse[span, span] @ mixing_penalty_matrix))
        roughness = float(coordinates @ mixing_penalty_matrix @ coordinates)
        mixing_penalty[class_position] = max(mixing_rank - mixing_penalty[class_position] * trace, 1e-12) / max(roughness, 1e-300)
    annotation_penalty = hyperparameters.annotation_penalty.copy()
    offset = prior.class_count * block
    for group_position, group in enumerate(prior.annotation_groups):
        indices = offset + group.columns
        coefficients = annotation_coefficients[group.columns]
        trace = float(np.trace(inverse[np.ix_(indices, indices)] @ group.penalty))
        rank = np.linalg.matrix_rank(group.penalty)
        size = float(coefficients @ group.penalty @ coefficients)
        annotation_penalty[group_position] = max(rank - annotation_penalty[group_position] * trace, 1e-12) / max(size, 1e-300)
    return ReferenceHyperparameters(
        mixing_coordinates=mixing_coordinates.copy(),
        annotation_coefficients=annotation_coefficients.copy(),
        mixing_penalty=mixing_penalty,
        annotation_penalty=annotation_penalty,
    )


def newton_decrement(prior, hyperparameters, cavity_precision, cavity_shift) -> float:
    """1/2 gᵀ(-H)⁻¹g of the penalized objective in (ψ, θ): what a Newton step would still gain."""
    vector = _pack(prior, hyperparameters.mixing_coordinates, hyperparameters.annotation_coefficients)

    def objective(candidate):
        return penalized_objective(prior, hyperparameters, candidate, cavity_precision, cavity_shift)

    _value, gradient = objective(vector)
    return float(0.5 * gradient @ np.linalg.solve(-_numerical_hessian(objective, vector), gradient))


def site_targets(log_scale, log_density, log_variance_grid, cavity_precision, cavity_shift):
    """Clipped, mean-matched site parameters from the exact tilted moments.

    ``log_scale`` is log u_j and ``log_density`` the log mixing weights of each
    variant's class, for the variants whose cavities are given.
    """
    variance = np.exp(log_scale[:, None] + log_variance_grid[None, :])
    precision = cavity_precision[:, None]
    shift_square = np.square(cavity_shift)[:, None]
    relative = 1.0 + variance * precision
    conditional_variance = variance / relative
    log_component = log_density - 0.5 * np.log(relative) + 0.5 * shift_square * conditional_variance
    responsibility = np.exp(log_component - logsumexp(log_component, axis=1, keepdims=True))
    tilted_mean = cavity_shift * np.sum(responsibility * conditional_variance, axis=1)
    tilted_variance = (
        np.sum(responsibility * (conditional_variance + shift_square * np.square(conditional_variance)), axis=1)
        - np.square(tilted_mean)
    )
    target_precision = np.maximum(1.0 / tilted_variance - cavity_precision, 0.0)
    return target_precision, tilted_mean * (cavity_precision + target_precision) - cavity_shift


def run_sites(prior, hyperparameters, likelihood_precision, linear_term, site_precision, site_shift, damping, tolerance, maximum_sweeps):
    """Damped sequential EP at fixed hyperparameters until the sites stop moving.

    Sequential (one site at a time, rank-one updates of the posterior) rather
    than parallel: parallel EP can cycle when many effects share one signal, and
    the reference must converge wherever a fixed point exists. Every sweep
    restarts from a fresh inverse so rounding does not accumulate.
    """
    site_precision = np.array(site_precision, dtype=np.float64, copy=True)
    site_shift = np.array(site_shift, dtype=np.float64, copy=True)
    log_scale = prior.log_variance_offset + prior.centred_design @ hyperparameters.annotation_coefficients
    log_density = np.log(mixing_density(prior, hyperparameters.mixing_coordinates))[prior.class_index]
    for _sweep in range(maximum_sweeps):
        covariance = np.linalg.inv(likelihood_precision + np.diag(site_precision))
        covariance = 0.5 * (covariance + covariance.T)
        posterior_mean = covariance @ (linear_term + site_shift)
        largest_change = 0.0
        for variant in range(linear_term.shape[0]):
            marginal_variance = covariance[variant, variant]
            cavity_precision = np.array([max(1.0 / marginal_variance - site_precision[variant], 0.0)])
            cavity_shift = np.array([posterior_mean[variant] / marginal_variance - site_shift[variant]])
            target_precision, target_shift = site_targets(
                log_scale[variant : variant + 1],
                log_density[variant : variant + 1],
                prior.log_variance_grid,
                cavity_precision,
                cavity_shift,
            )
            precision_step = damping * (float(target_precision[0]) - site_precision[variant])
            shift_step = damping * (float(target_shift[0]) - site_shift[variant])
            largest_change = max(
                largest_change,
                abs(precision_step) / (abs(site_precision[variant]) + 1.0),
                abs(shift_step) / (abs(site_shift[variant]) + 1.0),
            )
            column = covariance[:, variant].copy()
            denominator = 1.0 + precision_step * marginal_variance
            posterior_mean = posterior_mean + column * (shift_step - precision_step * posterior_mean[variant]) / denominator
            covariance = covariance - np.outer(column, column) * (precision_step / denominator)
            site_precision[variant] += precision_step
            site_shift[variant] += shift_step
        if largest_change < tolerance:
            break
    covariance = np.linalg.inv(likelihood_precision + np.diag(site_precision))
    covariance = 0.5 * (covariance + covariance.T)
    posterior_mean = covariance @ (linear_term + site_shift)
    posterior_variance = np.diag(covariance).copy()
    marginal_precision = 1.0 / posterior_variance
    cavity_precision = np.maximum(marginal_precision - site_precision, 0.0)
    cavity_shift = posterior_mean * marginal_precision - site_shift
    return site_precision, site_shift, posterior_mean, posterior_variance, cavity_precision, cavity_shift


def initial_hyperparameters(prior: ReferencePrior) -> ReferenceHyperparameters:
    """A flat mixing density (every grid variance equally likely), no annotation effect, unit penalties."""
    return ReferenceHyperparameters(
        mixing_coordinates=np.zeros((prior.class_count, prior.grid_size - 1)),
        annotation_coefficients=np.zeros(prior.feature_count),
        mixing_penalty=np.ones(prior.class_count),
        annotation_penalty=np.ones(len(prior.annotation_groups)),
    )


def fit_reference(
    prior: ReferencePrior,
    likelihood_precision: np.ndarray,
    linear_term: np.ndarray,
    damping: float = 0.3,
    site_tolerance: float = 1e-12,
    hyperparameter_tolerance: float = 1e-9,
    maximum_outer_iterations: int = 300,
) -> ReferenceFit:
    """The EP-EB fixed point: EP at fixed hyperparameters, the exact (ψ, θ) maximizer, one penalty step; repeated.

    The outer map is EM-like (linear convergence), so its residual in
    (ψ, θ, log λ) is Anderson-mixed; that does not move the fixed point. The
    tolerance sits above the finite-difference Hessian's noise floor (~1e-10);
    gates compare at 1e-6.
    """
    hyperparameters = initial_hyperparameters(prior)
    typical_variance = np.exp(prior.log_variance_offset + np.mean(prior.log_variance_grid))
    site_precision = 1.0 / typical_variance
    site_shift = np.zeros(linear_term.shape[0])
    history_points: list[np.ndarray] = []
    history_residuals: list[np.ndarray] = []

    def state_vector(current: ReferenceHyperparameters) -> np.ndarray:
        return np.concatenate(
            [
                _pack(prior, current.mixing_coordinates, current.annotation_coefficients),
                np.log(current.mixing_penalty),
                np.log(current.annotation_penalty),
            ]
        )

    def from_state(vector: np.ndarray) -> ReferenceHyperparameters:
        coefficient_size = prior.class_count * (prior.grid_size - 1) + prior.feature_count
        mixing_coordinates, annotation_coefficients = _unpack(prior, vector[:coefficient_size])
        return ReferenceHyperparameters(
            mixing_coordinates=mixing_coordinates.copy(),
            annotation_coefficients=annotation_coefficients.copy(),
            mixing_penalty=np.exp(vector[coefficient_size : coefficient_size + prior.class_count]),
            annotation_penalty=np.exp(vector[coefficient_size + prior.class_count :]),
        )

    for outer_iteration in range(1, maximum_outer_iterations + 1):
        site_precision, site_shift, posterior_mean, posterior_variance, cavity_precision, cavity_shift = run_sites(
            prior, hyperparameters, likelihood_precision, linear_term, site_precision, site_shift, damping, site_tolerance, 20_000
        )
        coefficients = maximize_coefficients(
            prior,
            hyperparameters,
            _pack(prior, hyperparameters.mixing_coordinates, hyperparameters.annotation_coefficients),
            cavity_precision,
            cavity_shift,
        )
        mapped_hyperparameters = update_penalties(prior, hyperparameters, coefficients, cavity_precision, cavity_shift)
        current = state_vector(hyperparameters)
        mapped = state_vector(mapped_hyperparameters)
        residual = mapped - current
        if float(np.max(np.abs(residual))) < hyperparameter_tolerance:
            hyperparameters = mapped_hyperparameters
            break
        history_points.append(current)
        history_residuals.append(residual)
        del history_points[:-5]
        del history_residuals[:-5]
        proposal = mapped
        if len(history_residuals) >= 2:
            residual_differences = np.diff(np.array(history_residuals), axis=0).T
            point_differences = np.diff(np.array(history_points), axis=0).T
            mixing, *_ = np.linalg.lstsq(residual_differences, residual, rcond=None)
            proposal = mapped - (point_differences + residual_differences) @ mixing
            if not (np.all(np.isfinite(proposal)) and float(np.max(np.abs(proposal - mapped))) <= 2.0):
                history_points.clear()
                history_residuals.clear()
                proposal = mapped
        hyperparameters = from_state(proposal)
    site_precision, site_shift, posterior_mean, posterior_variance, cavity_precision, cavity_shift = run_sites(
        prior, hyperparameters, likelihood_precision, linear_term, site_precision, site_shift, damping, site_tolerance, 20_000
    )
    return ReferenceFit(
        posterior_mean=posterior_mean,
        posterior_variance=posterior_variance,
        hyperparameters=hyperparameters,
        mixing_density=mixing_density(prior, hyperparameters.mixing_coordinates),
        site_precision=site_precision,
        site_shift=site_shift,
        cavity_precision=cavity_precision,
        cavity_shift=cavity_shift,
        newton_decrement=newton_decrement(prior, hyperparameters, cavity_precision, cavity_shift),
        outer_iterations=outer_iteration,
    )


def assert_converged(fit: ReferenceFit, maximum_outer_iterations: int = 300) -> None:
    if fit.outer_iterations >= maximum_outer_iterations:
        raise AssertionError(f"The reference did not converge in {maximum_outer_iterations} outer iterations.")
