"""Dense reference for SV-PGS's EP-EB fixed point, for exactness tests of the fast stages.

This module is test infrastructure: it shares no code with the fitting stages,
so a stage that agrees with it has the model right, not just its own algebra.
It is dense (one p x p inverse per sweep), so it is only for problems of a few
hundred variants.

Model. The data enter as one Gaussian factor exp(-1/2 βᵀΛβ + ℓᵀβ): for a
quantitative trait in LD space Λ = κ nR and ℓ = κ X̃ᵀr; for a binary trait's
working model Λ = w̄ nR and ℓ is the working score. Each effect has the prior

    β_j | λ_j ~ N(0, v(u_j λ_j)),   log u_j = level + o_j + d_jᵀθ,
    λ_j ~ BetaPrime(a, b_c(j)),

with v(x) = x, or v(x) = x s² / (x + s²) for the regularized slab of scale s.
BetaPrime is represented on a fixed even grid in t = log λ with the quadrature
masses a t - (a + b) log(1 + e^t) - log B(a, b) + log Δt (the grid is part of
the model definition, so a stage and this reference integrate the same prior).
θ ~ N(θ₀, diag(1/P)); the level is flat; the log tail shapes x_c = log b_c get
the pooling penalty -1/2 Σ_c (x_c - x̄)² / s_b² with the mean x̄ free, and stay
in [log 0.1, log 10].

Inference. EP with one Gaussian site per effect, clipped (site precision ≥ 0)
and mean-matched (the posterior mean equals the tilted mean exactly), with the
exact tilted moments on the grid. The hyperparameters maximize the EP cavity
marginal Σ_j log Z_j plus their priors with the cavities held fixed (type-II
maximum likelihood under EP). The fixed point is reached when neither the sites
nor the hyperparameters move; it does not depend on the damping or the order
of the updates, which is what lets a stage and this reference be compared.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, digamma, logsumexp

GRID_LOWER_TAIL_LOG_MASS = 20.0
GRID_UPPER_LOG_LOCAL_SCALE = 50.0
GRID_STEP = 0.5
MINIMUM_SHAPE_B = 0.1
MAXIMUM_SHAPE_B = 10.0


@dataclass(frozen=True)
class ReferencePrior:
    """Everything about the prior except the fitted hyperparameters."""

    class_index: np.ndarray
    log_variance_offset: np.ndarray
    annotation_design: np.ndarray
    annotation_prior_mean: np.ndarray
    annotation_prior_precision: np.ndarray
    shape_a: float
    shape_b_pooling_variance: float
    slab_scale: float | None

    @property
    def class_count(self) -> int:
        return int(self.class_index.max()) + 1


@dataclass(frozen=True)
class ReferenceHyperparameters:
    log_variance_level: float
    annotation_coefficients: np.ndarray
    shape_b: np.ndarray

    def vector(self) -> np.ndarray:
        return np.concatenate([[self.log_variance_level], self.annotation_coefficients, np.log(self.shape_b)])


@dataclass(frozen=True)
class ReferenceFit:
    posterior_mean: np.ndarray
    posterior_variance: np.ndarray
    hyperparameters: ReferenceHyperparameters
    site_precision: np.ndarray
    site_shift: np.ndarray
    cavity_precision: np.ndarray
    cavity_shift: np.ndarray
    newton_decrement: float
    outer_iterations: int


def log_local_scale_grid(shape_a: float) -> np.ndarray:
    lower = -GRID_LOWER_TAIL_LOG_MASS / shape_a
    point_count = int(np.ceil((GRID_UPPER_LOG_LOCAL_SCALE - lower) / GRID_STEP)) + 1
    return np.linspace(lower, GRID_UPPER_LOG_LOCAL_SCALE, point_count)


def log_prior_masses(shape_a: float, shape_b: np.ndarray) -> np.ndarray:
    """Quadrature masses of BetaPrime(a, b_c) at the grid points, one row per class."""
    log_local_scale = log_local_scale_grid(shape_a)
    log_one_plus = np.logaddexp(0.0, log_local_scale)
    step = log_local_scale[1] - log_local_scale[0]
    return (
        shape_a * log_local_scale[None, :]
        - (shape_a + shape_b[:, None]) * log_one_plus[None, :]
        - betaln(shape_a, shape_b)[:, None]
        + np.log(step)
    )


def log_prior_variance(prior: ReferencePrior, hyperparameters: ReferenceHyperparameters) -> np.ndarray:
    return (
        hyperparameters.log_variance_level
        + prior.log_variance_offset
        + prior.annotation_design @ hyperparameters.annotation_coefficients
    )


def _component_variances(prior: ReferencePrior, log_scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Component variances v_jk and their elasticities ∂ log v_jk / ∂ log u_j."""
    raw_variance = np.exp(log_scale)[:, None] * np.exp(log_local_scale_grid(prior.shape_a))[None, :]
    if prior.slab_scale is None:
        return raw_variance, np.ones_like(raw_variance)
    slab_square = prior.slab_scale**2
    return raw_variance * slab_square / (raw_variance + slab_square), slab_square / (raw_variance + slab_square)


def tilted_terms(
    prior: ReferencePrior,
    hyperparameters: ReferenceHyperparameters,
    cavity_precision: np.ndarray,
    cavity_shift: np.ndarray,
) -> dict[str, np.ndarray]:
    """log Z_j, the tilted mean and variance, and the pieces of their hyperparameter derivatives.

    The tilted density is Σ_k m_ck N(β; 0, v_jk) exp(-1/2 P_j β² + h_j β). Its
    component k integrates to m_ck (1 + v_jk P_j)^(-1/2) exp(1/2 h_j² c_jk) with
    c_jk = v_jk / (1 + v_jk P_j), and is Gaussian with mean h_j c_jk and variance c_jk.
    """
    variance, elasticity = _component_variances(prior, log_prior_variance(prior, hyperparameters))
    precision = cavity_precision[:, None]
    shift_square = np.square(cavity_shift)[:, None]
    conditional_variance = variance / (1.0 + variance * precision)
    log_mass = log_prior_masses(prior.shape_a, hyperparameters.shape_b)[prior.class_index]
    log_component = log_mass - 0.5 * np.log1p(variance * precision) + 0.5 * shift_square * conditional_variance
    log_normalizer = logsumexp(log_component, axis=1)
    weight = np.exp(log_component - log_normalizer[:, None])
    mean_conditional_variance = np.sum(weight * conditional_variance, axis=1)
    tilted_mean = cavity_shift * mean_conditional_variance
    tilted_second_moment = np.sum(weight * (conditional_variance + shift_square * np.square(conditional_variance)), axis=1)
    # ∂ log component / ∂ log u = v ε [-1/2 P / (1 + vP) + 1/2 h² / (1 + vP)²].
    component_scale_derivative = (
        variance
        * elasticity
        * (-0.5 * precision / (1.0 + variance * precision) + 0.5 * shift_square / np.square(1.0 + variance * precision))
    )
    return {
        "log_normalizer": log_normalizer,
        "weight": weight,
        "tilted_mean": tilted_mean,
        "tilted_variance": tilted_second_moment - np.square(tilted_mean),
        "scale_derivative": np.sum(weight * component_scale_derivative, axis=1),
        "mean_log_one_plus": weight @ np.logaddexp(0.0, log_local_scale_grid(prior.shape_a)),
    }


def _unpack(prior: ReferencePrior, vector: np.ndarray) -> ReferenceHyperparameters:
    feature_count = prior.annotation_design.shape[1]
    return ReferenceHyperparameters(
        log_variance_level=float(vector[0]),
        annotation_coefficients=np.asarray(vector[1 : 1 + feature_count], dtype=np.float64),
        shape_b=np.exp(np.asarray(vector[1 + feature_count :], dtype=np.float64)),
    )


def hyperparameter_objective(
    prior: ReferencePrior, vector: np.ndarray, cavity_precision: np.ndarray, cavity_shift: np.ndarray
) -> tuple[float, np.ndarray]:
    """Σ_j log Z_j plus the hyperparameter priors at fixed cavities, and its gradient in (level, θ, log b)."""
    hyperparameters = _unpack(prior, vector)
    terms = tilted_terms(prior, hyperparameters, cavity_precision, cavity_shift)
    coefficient_deviation = hyperparameters.annotation_coefficients - prior.annotation_prior_mean
    log_shape_b = np.log(hyperparameters.shape_b)
    centered_log_shape_b = log_shape_b - log_shape_b.mean()
    value = (
        float(np.sum(terms["log_normalizer"]))
        - 0.5 * float(np.sum(prior.annotation_prior_precision * np.square(coefficient_deviation)))
        - 0.5 * float(np.sum(np.square(centered_log_shape_b))) / prior.shape_b_pooling_variance
    )
    scale_derivative = terms["scale_derivative"]
    shape_b = hyperparameters.shape_b
    # ∂ log m_ck / ∂ b_c = -log(1 + λ_k) + ψ(a + b_c) - ψ(b_c).
    per_variant_shape_derivative = (
        -terms["mean_log_one_plus"] + (digamma(prior.shape_a + shape_b) - digamma(shape_b))[prior.class_index]
    )
    shape_gradient = shape_b * np.bincount(
        prior.class_index, weights=per_variant_shape_derivative, minlength=prior.class_count
    ) - centered_log_shape_b / prior.shape_b_pooling_variance
    gradient = np.concatenate(
        [
            [float(np.sum(scale_derivative))],
            prior.annotation_design.T @ scale_derivative - prior.annotation_prior_precision * coefficient_deviation,
            shape_gradient,
        ]
    )
    return value, gradient


def _numerical_hessian(prior: ReferencePrior, vector: np.ndarray, cavity_precision, cavity_shift) -> np.ndarray:
    step = 1e-5
    dimension = vector.shape[0]
    hessian = np.empty((dimension, dimension))
    for coordinate in range(dimension):
        forward = vector.copy()
        backward = vector.copy()
        forward[coordinate] += step
        backward[coordinate] -= step
        hessian[:, coordinate] = (
            hyperparameter_objective(prior, forward, cavity_precision, cavity_shift)[1]
            - hyperparameter_objective(prior, backward, cavity_precision, cavity_shift)[1]
        ) / (2.0 * step)
    return 0.5 * (hessian + hessian.T)


def _free_coordinates(prior: ReferencePrior, vector: np.ndarray) -> np.ndarray:
    """Coordinates not pinned at a tail-shape bound."""
    feature_count = prior.annotation_design.shape[1]
    free = np.ones(vector.shape[0], dtype=bool)
    log_shape_b = vector[1 + feature_count :]
    pinned = (log_shape_b <= np.log(MINIMUM_SHAPE_B) + 1e-12) | (log_shape_b >= np.log(MAXIMUM_SHAPE_B) - 1e-12)
    free[1 + feature_count :] = ~pinned
    return free


def newton_decrement(prior: ReferencePrior, vector: np.ndarray, cavity_precision, cavity_shift) -> float:
    """1/2 gᵀ(-H)⁻¹g over the free coordinates: the objective gain a Newton step would still make."""
    free = _free_coordinates(prior, vector)
    _value, gradient = hyperparameter_objective(prior, vector, cavity_precision, cavity_shift)
    negative_hessian = -_numerical_hessian(prior, vector, cavity_precision, cavity_shift)[np.ix_(free, free)]
    return float(0.5 * gradient[free] @ np.linalg.solve(negative_hessian, gradient[free]))


def maximize_hyperparameters(
    prior: ReferencePrior, start: np.ndarray, cavity_precision: np.ndarray, cavity_shift: np.ndarray
) -> np.ndarray:
    feature_count = prior.annotation_design.shape[1]
    bounds = [(None, None)] * (1 + feature_count) + [(np.log(MINIMUM_SHAPE_B), np.log(MAXIMUM_SHAPE_B))] * prior.class_count
    result = minimize(
        lambda vector: tuple(-part for part in hyperparameter_objective(prior, vector, cavity_precision, cavity_shift)),
        start,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 10_000, "ftol": 1e-16, "gtol": 1e-12},
    )
    vector = np.asarray(result.x, dtype=np.float64)
    # Polish with Newton steps on the free coordinates (the finite-difference
    # Hessian is accurate to ~1e-10, so steps stop shrinking around there).
    for _newton_step in range(12):
        free = _free_coordinates(prior, vector)
        _value, gradient = hyperparameter_objective(prior, vector, cavity_precision, cavity_shift)
        negative_hessian = -_numerical_hessian(prior, vector, cavity_precision, cavity_shift)[np.ix_(free, free)]
        step = np.linalg.solve(negative_hessian, gradient[free])
        vector[free] += step
        vector[1 + feature_count :] = np.clip(vector[1 + feature_count :], np.log(MINIMUM_SHAPE_B), np.log(MAXIMUM_SHAPE_B))
        if float(np.max(np.abs(step))) < 1e-12:
            break
    return vector


def posterior(likelihood_precision: np.ndarray, linear_term: np.ndarray, site_precision, site_shift):
    covariance = np.linalg.inv(likelihood_precision + np.diag(site_precision))
    covariance = 0.5 * (covariance + covariance.T)
    return covariance @ (linear_term + site_shift), np.diag(covariance).copy()


def cavities(posterior_mean, posterior_variance, site_precision, site_shift):
    marginal_precision = 1.0 / posterior_variance
    cavity_precision = np.maximum(marginal_precision - site_precision, 0.0)
    return cavity_precision, posterior_mean * marginal_precision - site_shift


def site_targets(prior, hyperparameters, cavity_precision, cavity_shift):
    """Clipped, mean-matched site parameters from the exact tilted moments."""
    terms = tilted_terms(prior, hyperparameters, cavity_precision, cavity_shift)
    target_precision = np.maximum(1.0 / terms["tilted_variance"] - cavity_precision, 0.0)
    return target_precision, terms["tilted_mean"] * (cavity_precision + target_precision) - cavity_shift


def run_sites(
    prior: ReferencePrior,
    hyperparameters: ReferenceHyperparameters,
    likelihood_precision: np.ndarray,
    linear_term: np.ndarray,
    site_precision: np.ndarray,
    site_shift: np.ndarray,
    damping: float,
    tolerance: float,
    maximum_sweeps: int,
):
    """Damped sequential EP at fixed hyperparameters until the sites stop moving.

    Sequential (one site at a time, rank-one updates of the posterior) rather
    than parallel: parallel EP can cycle when many effects share one signal, and
    the reference must converge wherever a fixed point exists. Every sweep
    restarts from a fresh inverse so rounding does not accumulate.
    """
    site_precision = site_precision.copy()
    site_shift = site_shift.copy()
    for _sweep in range(maximum_sweeps):
        covariance = np.linalg.inv(likelihood_precision + np.diag(site_precision))
        covariance = 0.5 * (covariance + covariance.T)
        posterior_mean = covariance @ (linear_term + site_shift)
        largest_change = 0.0
        for variant in range(linear_term.shape[0]):
            marginal_variance = covariance[variant, variant]
            cavity_precision = np.array([max(1.0 / marginal_variance - site_precision[variant], 0.0)])
            cavity_shift = np.array([posterior_mean[variant] / marginal_variance - site_shift[variant]])
            single = ReferencePrior(
                class_index=prior.class_index[variant : variant + 1],
                log_variance_offset=prior.log_variance_offset[variant : variant + 1],
                annotation_design=prior.annotation_design[variant : variant + 1],
                annotation_prior_mean=prior.annotation_prior_mean,
                annotation_prior_precision=prior.annotation_prior_precision,
                shape_a=prior.shape_a,
                shape_b_pooling_variance=prior.shape_b_pooling_variance,
                slab_scale=prior.slab_scale,
            )
            target_precision, target_shift = site_targets(single, hyperparameters, cavity_precision, cavity_shift)
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
    posterior_mean, posterior_variance = posterior(likelihood_precision, linear_term, site_precision, site_shift)
    cavity_precision, cavity_shift = cavities(posterior_mean, posterior_variance, site_precision, site_shift)
    return site_precision, site_shift, posterior_mean, posterior_variance, cavity_precision, cavity_shift


def fit_reference(
    prior: ReferencePrior,
    start: ReferenceHyperparameters,
    likelihood_precision: np.ndarray,
    linear_term: np.ndarray,
    damping: float = 0.3,
    site_tolerance: float = 1e-12,
    hyperparameter_tolerance: float = 1e-10,
    maximum_outer_iterations: int = 200,
) -> ReferenceFit:
    """The EP-EB fixed point: EP to convergence at fixed hyperparameters, then the exact type-II step, repeated.

    The hyperparameter tolerance sits above the Newton polish's noise floor
    (~1e-10 from the finite-difference Hessian); gates compare at 1e-6.
    """
    variant_count = linear_term.shape[0]
    vector = start.vector()
    site_precision = 1.0 / np.exp(log_prior_variance(prior, start))
    site_shift = np.zeros(variant_count)
    # The outer map (EP to convergence, then the exact hyperparameter maximizer) is
    # EM-like and converges linearly; Anderson mixing of its residual keeps the
    # fixed point and reaches 1e-12 in tens of iterations instead of hundreds.
    history_points: list[np.ndarray] = []
    history_residuals: list[np.ndarray] = []
    anderson_memory = 5
    for outer_iteration in range(1, maximum_outer_iterations + 1):
        hyperparameters = _unpack(prior, vector)
        site_precision, site_shift, posterior_mean, posterior_variance, cavity_precision, cavity_shift = run_sites(
            prior, hyperparameters, likelihood_precision, linear_term, site_precision, site_shift, damping, site_tolerance, 20_000
        )
        mapped = maximize_hyperparameters(prior, vector, cavity_precision, cavity_shift)
        residual = mapped - vector
        if float(np.max(np.abs(residual))) < hyperparameter_tolerance:
            vector = mapped
            break
        history_points.append(vector.copy())
        history_residuals.append(residual.copy())
        del history_points[:-anderson_memory]
        del history_residuals[:-anderson_memory]
        proposal = mapped
        if len(history_residuals) >= 2:
            residual_differences = np.diff(np.array(history_residuals), axis=0).T
            point_differences = np.diff(np.array(history_points), axis=0).T
            mixing, *_ = np.linalg.lstsq(residual_differences, residual, rcond=None)
            proposal = mapped - (point_differences + residual_differences) @ mixing
            feature_count = prior.annotation_design.shape[1]
            proposal[1 + feature_count :] = np.clip(
                proposal[1 + feature_count :], np.log(MINIMUM_SHAPE_B), np.log(MAXIMUM_SHAPE_B)
            )
            if not (np.all(np.isfinite(proposal)) and float(np.max(np.abs(proposal - mapped))) <= 2.0):
                history_points.clear()
                history_residuals.clear()
                proposal = mapped
        vector = proposal
    hyperparameters = _unpack(prior, vector)
    site_precision, site_shift, posterior_mean, posterior_variance, cavity_precision, cavity_shift = run_sites(
        prior, hyperparameters, likelihood_precision, linear_term, site_precision, site_shift, damping, site_tolerance, 20_000
    )
    return ReferenceFit(
        posterior_mean=posterior_mean,
        posterior_variance=posterior_variance,
        hyperparameters=hyperparameters,
        site_precision=site_precision,
        site_shift=site_shift,
        cavity_precision=cavity_precision,
        cavity_shift=cavity_shift,
        newton_decrement=newton_decrement(prior, vector, cavity_precision, cavity_shift),
        outer_iterations=outer_iteration,
    )


def assert_converged(fit: ReferenceFit, maximum_outer_iterations: int = 200) -> None:
    if fit.outer_iterations >= maximum_outer_iterations:
        raise AssertionError(f"The reference did not converge in {maximum_outer_iterations} outer iterations.")
