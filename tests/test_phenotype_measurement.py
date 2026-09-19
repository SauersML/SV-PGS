"""phenotype_measurement: the exact level quadrature against brute force and closed forms, Louis' information,
the evidence, and gross errors. The simulations check the algebra ([sim-only])."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm

from sv_pgs.phenotype_measurement import (
    Occasions,
    _density_prior,
    _log_modulus_bound,
    box_cox,
    fit_at_exponent,
    level_posterior,
    level_posterior_at,
    person_tolerance,
)
from sv_pgs.scale_mixture_ep import MixtureHyperparameters, ScaleMixturePrior, class_log_density
from tests.phenotype_bounds import rounding_gamma, sampling_bound, variance_component_standard_errors

WORKING_BYTES = 1 << 28
# Below every node's own share: exp of this is eps^2, so an absent node moves no moment at double precision.
_ABSENT = -2.0 * np.log(1.0 / np.finfo(np.float64).eps)


def _lattice(floor_variance: float, top_variance: float, count: int, occasions: int) -> ScaleMixturePrior:
    nodes = np.linspace(np.log(floor_variance), np.log(top_variance), count)
    return _density_prior(nodes, float(nodes[-1]), occasions)


def _density(prior: ScaleMixturePrior, masses: dict[int, float]) -> MixtureHyperparameters:
    """Hyperparameters whose lattice masses are ``masses`` (node -> mass) and absent elsewhere."""
    log_density = np.full(prior.grid_size, _ABSENT)
    for node, mass in masses.items():
        log_density[node] = np.log(mass)
    mapping = prior.coefficient_map[: prior.grid_size]
    coefficients = np.linalg.lstsq(mapping, log_density - log_density.mean(), rcond=None)[0]
    return MixtureHyperparameters(coefficients=coefficients, log_smoothing=np.zeros(len(prior.smoothing_blocks)))


def _node(prior: ScaleMixturePrior, variance: float) -> int:
    return int(np.argmin(np.abs(prior.log_variance_grid - np.log(variance))))


def _brute(residuals: np.ndarray, level_variance: float, masses: np.ndarray, variances: np.ndarray):
    """(log L, E[T], E[T^2]) by adaptive quadrature, and its own estimate of each moment's error."""

    def integrand(level: float, power: int) -> float:
        log_occasions = [logsumexp(np.log(masses) + norm.logpdf(residual - level, scale=np.sqrt(variances))) for residual in residuals]
        return level**power * float(np.exp(norm.logpdf(level, scale=np.sqrt(level_variance)) + np.sum(log_occasions)))

    results = [_integral(integrand, power, residuals) for power in (0, 1, 2)]
    (total, first, second), (total_error, first_error, second_error) = zip(*results)
    mean = first / total
    return (
        np.log(total), mean, second / total,
        np.array([total_error / total, (first_error + abs(mean) * total_error) / total, (second_error + second / total * total_error) / total]),
    )


def _moment_errors(tolerance: float, mean, second):
    """Bounds on the errors of E[T], E[T^2] and Var(T) when L, the integral of |T| F and that of T^2 F each have
    relative error at most ``tolerance``: E|T| <= |E T| + sd, and a ratio (I + d) / (L + e) moves by at most
    tolerance (|I| / L + |I| / L) / (1 - tolerance)."""
    scale = abs(mean) + np.sqrt(second - mean * mean)
    first_error = 2.0 * tolerance / (1.0 - tolerance) * scale
    second_error = 2.0 * tolerance / (1.0 - tolerance) * second
    return first_error, second_error, second_error + (2.0 * abs(mean) + first_error) * first_error


def test_box_cox_is_increasing_with_an_exact_log_jacobian():
    values = np.geomspace(0.01, 1000.0, 200)
    for exponent in (-0.5, 0.0, 0.5, 1.0, 2.0):
        transformed, log_jacobian = box_cox(values, exponent)
        assert np.all(np.diff(transformed) > 0.0)
        # h'(y) = y^(lambda - 1): one log, one product and the power's own two roundings.
        np.testing.assert_allclose(log_jacobian, np.log(np.power(values, exponent - 1.0)), rtol=0.0,
                                   atol=2.0 * rounding_gamma(4) * np.max(np.abs(log_jacobian)))
    np.testing.assert_array_equal(box_cox(values, 0.0)[0], np.log(values))


@pytest.mark.parametrize(
    "residuals",
    [
        [1.3, 48.0],  # one gross reading
        [3.0, 3.0],  # a duplicated pair: the integrand spikes at the shared value
        [0.4, -1.1, 2.0, 3.0, 3.0, 55.0],  # six occasions with a duplicate and a gross reading
    ],
)
def test_the_certified_quadrature_matches_brute_force(residuals):
    prior = _lattice(1.0 / 12.0, 5000.0, 40, 2)
    hyperparameters = _density(prior, {_node(prior, 1.0): 0.9, _node(prior, 1.0 / 12.0): 0.05, _node(prior, 2500.0): 0.05})
    masses = np.exp(class_log_density(prior, hyperparameters.coefficients)[0])
    variances = np.exp(prior.log_variance_grid)
    level_variance, tolerance = 4.0, person_tolerance(50_000)
    residual_array = np.array([residuals])
    posterior = level_posterior(residual_array, level_variance, np.log(masses), variances, tolerance, None, None, True, WORKING_BYTES)
    log_likelihood, mean, second, brute_error = _brute(residual_array[0], level_variance, masses, variances)
    first_error, second_error, _variance_error = _moment_errors(tolerance, mean, second)
    # The certificate bounds L's relative error by the tolerance; the brute force adds its own error estimate.
    assert abs(posterior.log_likelihood[0] - log_likelihood) <= -np.log1p(-tolerance) + brute_error[0]
    assert abs(posterior.level_mean[0] - mean) <= first_error + brute_error[1]
    assert abs(posterior.level_second_moment[0] - second) <= second_error + brute_error[2]


@pytest.mark.parametrize("residuals", [[0.7], [1.3, 48.0], [3.0, 3.0], [0.4, -1.1, 2.0, 3.0, 3.0, 55.0]])
def test_the_closed_form_modulus_bound_covers_the_mass_weighted_modulus(residuals):
    prior = _lattice(1.0 / 12.0, 5000.0, 12, 2)
    hyperparameters = _density(prior, {_node(prior, 1.0): 0.9, _node(prior, 1.0 / 12.0): 0.05, _node(prior, 2500.0): 0.05})
    masses = np.exp(class_log_density(prior, hyperparameters.coefficients)[0])
    variances = np.exp(prior.log_variance_grid)
    level_variance, residual_array = 4.0, np.array([residuals])
    for half_width in (0.05, 0.3, 1.0):
        bound = _log_modulus_bound(residual_array, level_variance, np.log(masses), variances, np.array([half_width]))[0]
        widened = masses * np.exp(0.5 * half_width**2 / variances)

        def integrand(level: float, power: int) -> float:
            occasions = np.prod([widened @ norm.pdf(residual - level, scale=np.sqrt(variances)) for residual in residuals])
            prior_factor = norm.pdf(level, scale=np.sqrt(level_variance)) * np.exp(0.5 * half_width**2 / level_variance)
            return (abs(level) + half_width) ** power * prior_factor * occasions

        for power in (0, 1, 2):
            value, error = _integral(integrand, power, residual_array[0])
            # Hoelder and the power mean are equalities for one occasion, so the bound is then the modulus itself.
            if len(residuals) == 1:
                assert bound[power] == pytest.approx(np.log(value), abs=error / value + 2.0 * rounding_gamma(8 * prior.grid_size))
            else:
                assert bound[power] >= np.log(value - error)


def _integral(integrand, power: int, residuals: np.ndarray) -> tuple[float, float]:
    """The integral over the line by adaptive quadrature, split at the readings where the integrand may spike,
    and its own error estimate."""
    low, high = float(residuals.min()), float(residuals.max())
    pieces = [integrate.quad(integrand, -np.inf, low, args=(power,), limit=1000, epsabs=0.0, epsrel=1e-12),
              integrate.quad(integrand, high, np.inf, args=(power,), limit=1000, epsabs=0.0, epsrel=1e-12)]
    if high > low:
        interior = sorted(set(residuals.tolist()) - {low, high})
        pieces.append(integrate.quad(integrand, low, high, args=(power,), points=interior or None, limit=1000, epsabs=0.0, epsrel=1e-12))
    values, errors = zip(*pieces)
    return float(sum(values)), float(sum(errors))


def test_a_one_point_noise_density_gives_the_dense_henderson_solution_and_evidence():
    generator = np.random.default_rng(2)
    counts = generator.integers(1, 6, 40)
    person_index = np.repeat(np.arange(40), counts)
    values = np.round(50.0 + generator.normal(0.0, 2.0, 40)[person_index] + generator.normal(0.0, 1.0, person_index.shape[0]), 1)
    design = np.ones((values.shape[0], 1))
    occasions = Occasions(person_index=person_index, values=values, design=design)
    level_variance = 4.0
    fixed_effects = np.array([np.mean(values) - 1.0])
    prior = _lattice(0.1**2 / 12.0, 100.0, 50, values.shape[0])
    node = _node(prior, 1.0)
    noise = float(np.exp(prior.log_variance_grid[node]))
    fit = level_posterior_at(occasions, 1.0, fixed_effects, level_variance, prior, _density(prior, {node: 1.0}), WORKING_BYTES)

    residuals = values - 1.0 - fixed_effects[0]
    precision = 1.0 / level_variance + counts / noise
    mean = np.bincount(person_index, weights=residuals) / noise / precision
    first_error, _second_error, variance_error = _moment_errors(person_tolerance(40), mean, np.square(mean) + 1.0 / precision)
    np.testing.assert_array_less(np.abs(fit.level_mean - mean), first_error)
    np.testing.assert_array_less(np.abs(fit.level_posterior_variance - 1.0 / precision), variance_error)
    evidence = sum(
        multivariate_normal(mean=np.zeros(int(count)), cov=noise * np.eye(int(count)) + level_variance).logpdf(residuals[person_index == person])
        for person, count in enumerate(counts)
    )
    # Each person's L to relative error 1 - e^(-1/(2n)), so the sum to 1/2 nat.
    assert abs(fit.log_likelihood - evidence) <= 0.5 + 2.0 * rounding_gamma(8) * abs(evidence)


def test_the_likelihood_is_a_density_over_the_reading_at_the_log_transform():
    prior = _lattice(0.001, 4.0, 40, 1)
    hyperparameters = _density(prior, {_node(prior, 0.04): 0.8, _node(prior, 1.0): 0.2})
    masses = np.exp(class_log_density(prior, hyperparameters.coefficients)[0])
    variances = np.exp(prior.log_variance_grid)
    level_variance, centre, tolerance = 0.25, float(np.log(80.0)), person_tolerance(1)

    def density(log_reading: float) -> float:
        posterior = level_posterior(np.array([[log_reading - centre]]), level_variance, np.log(masses), variances, tolerance, None, None, False, WORKING_BYTES)
        return float(np.exp(posterior.log_likelihood[0]))  # the density of log y; its Jacobian dy = y d(log y) cancels

    total, error = integrate.quad(density, -np.inf, np.inf)
    assert abs(total - 1.0) <= tolerance + error


def test_louis_information_is_the_curvature_of_the_exact_log_likelihood():
    generator = np.random.default_rng(6)
    residuals = np.column_stack([generator.normal(0.0, 2.0, 30)] * 3) + generator.standard_t(3, (30, 3))
    prior = _lattice(0.05, 400.0, 12, 90)
    base = _density(prior, {node: 1.0 / prior.grid_size for node in range(prior.grid_size)}).coefficients
    variances = np.exp(prior.log_variance_grid)
    tolerance = person_tolerance(30)
    mapping = prior.coefficient_map[: prior.grid_size]

    def log_likelihood(coefficients: np.ndarray) -> float:
        log_masses = class_log_density(prior, coefficients)[0]
        return float(level_posterior(residuals, 4.0, log_masses, variances, tolerance, steps, centres, False, WORKING_BYTES).log_likelihood.sum())

    log_masses = class_log_density(prior, base)[0]
    # Half the admissible steps stay certified at the nearby densities of the differences, so every likelihood
    # below is one trapezoid rule on the same centred nodes, a finite mixture, for which Louis' identity is exact.
    first = level_posterior(residuals, 4.0, log_masses, variances, tolerance, None, None, False, WORKING_BYTES)
    steps, centres = 0.5 * first.admissible_step, first.level_mean
    posterior = level_posterior(residuals, 4.0, log_masses, variances, tolerance, steps, centres, True, WORKING_BYTES)
    masses = np.exp(log_masses)
    louis = mapping.T @ (posterior.counts.sum() * (np.diag(masses) - np.outer(masses, masses)) - posterior.missing_information) @ mapping

    def central(step: float) -> np.ndarray:
        size = base.shape[0]
        hessian = np.empty((size, size))
        for row in range(size):
            for column in range(size):
                shift_row, shift_column = np.eye(size)[row] * step, np.eye(size)[column] * step
                hessian[row, column] = (
                    log_likelihood(base + shift_row + shift_column) - log_likelihood(base + shift_row - shift_column)
                    - log_likelihood(base - shift_row + shift_column) + log_likelihood(base - shift_row - shift_column)
                ) / (4.0 * step * step)
        return hessian

    # Richardson: a central difference's error is O(h^2), so |D(h) - D(h/2)| 4/3 estimates D(h/2)'s; each of its
    # four log-likelihoods rounds by at most gamma_n sum_i |l_i|, with the 4 (K + J) operations of a node's sum
    # (an exp, a log, an add and a product per term), and the difference divides that by h^2.
    step = float(np.finfo(np.float64).eps ** 0.25)
    coarse, fine = central(step), central(0.5 * step)
    rounding = rounding_gamma(4 * (prior.grid_size + residuals.shape[1])) * float(np.abs(posterior.log_likelihood).sum())
    estimate = 4.0 / 3.0 * np.abs(coarse - fine) + 4.0 * rounding / (4.0 * (0.5 * step) ** 2)
    np.testing.assert_array_less(np.abs(-louis - fine), estimate + 2.0 * rounding_gamma(16) * np.abs(louis))


def _simulated(persons: int, level_variance: float, noise: float, seed: int, gross_share: float = 0.0):
    """Readings at 0.1 resolution around an age trend, and a copy with a share of them grossly wrong: half with
    an extra digit (x 10), half in mmol/L recorded as mg/dL (/ 18.016, glucose's molar mass over 10)."""
    generator = np.random.default_rng(seed)
    counts = generator.integers(1, 6, persons)
    person_index = np.repeat(np.arange(persons), counts)
    ages = generator.uniform(20.0, 80.0, person_index.shape[0])
    levels = generator.normal(0.0, np.sqrt(level_variance), persons)
    values = np.round(50.0 + 0.05 * (ages - 50.0) + levels[person_index] + generator.normal(0.0, np.sqrt(noise), person_index.shape[0]), 1)
    gross = generator.random(values.shape[0]) < gross_share
    factor = np.where(generator.random(values.shape[0]) < 0.5, 10.0, 1.0 / 18.016)
    design = np.column_stack([np.ones_like(ages), ages - ages.mean()])
    return person_index, values, np.where(gross, np.round(values * factor, 1), values), design, gross


def test_the_fit_recovers_gaussian_data_within_sampling_error():
    level_variance, noise = 4.0, 1.0
    person_index, values, _gross_values, design, _gross = _simulated(300, level_variance, noise, seed=3)
    fit = fit_at_exponent(Occasions(person_index=person_index, values=values, design=design), 1.0, WORKING_BYTES)
    # Maximum likelihood is efficient, so the Henderson III moment estimators' standard errors bound its own.
    between_error, within_error = variance_component_standard_errors(np.bincount(person_index), level_variance, noise)
    assert abs(fit.level_variance - level_variance) < sampling_bound(between_error)
    assert abs(fit.noise_second_moment - noise) < sampling_bound(within_error)


def test_gross_errors_are_downweighted_by_the_learned_density():
    # At one transform, so every fit's levels share a scale.
    person_index, values, gross_values, design, gross = _simulated(300, 4.0, 1.0, seed=4, gross_share=0.04)
    clean = fit_at_exponent(Occasions(person_index=person_index, values=values, design=design), 1.0, WORKING_BYTES)
    occasions = Occasions(person_index=person_index, values=gross_values, design=design)
    contaminated = fit_at_exponent(occasions, 1.0, WORKING_BYTES)
    affected = np.unique(person_index[gross])
    assert affected.shape[0] > 0

    # The oracle drops the gross readings and keeps the clean fit: a person left with none is at the prior.
    kept = ~gross
    remaining = np.unique(person_index[kept])
    oracle_mean, oracle_sd = np.zeros(person_index.max() + 1), np.full(person_index.max() + 1, np.sqrt(clean.level_variance))
    oracle = level_posterior_at(
        Occasions(person_index=np.searchsorted(remaining, person_index[kept]), values=values[kept], design=design[kept]),
        1.0, clean.fixed_effects, clean.level_variance, clean.prior, clean.hyperparameters, WORKING_BYTES,
    )
    oracle_mean[remaining], oracle_sd[remaining] = oracle.level_mean, np.sqrt(oracle.level_posterior_variance)
    # Each gross reading moves its person's level by less than one posterior sd from the oracle's ...
    np.testing.assert_array_less(np.abs(contaminated.level_mean[affected] - oracle_mean[affected]), oracle_sd[affected])
    # ... where Gaussian noise at the clean noise variance moves every one of them by more.
    gaussian = level_posterior_at(
        occasions, 1.0, clean.fixed_effects, clean.level_variance, contaminated.prior,
        _density(contaminated.prior, {_node(contaminated.prior, clean.noise_second_moment): 1.0}), WORKING_BYTES,
    )
    np.testing.assert_array_less(oracle_sd[affected], np.abs(gaussian.level_mean[affected] - oracle_mean[affected]))
