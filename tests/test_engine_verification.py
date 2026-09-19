"""Independent verification of the variant-side EP-EB engine across randomized regimes (lane verify-engine).

Every reference here avoids the engine's own formulas: direct quadrature of the tilted integrals, finite
differences of values (never of the engine's derivatives), the EP evidence log Z_EP itself with EP re-solved
at every point, and exact integrals of the evidence's integrand. Each certificate the engine returns (the
kernel floor, the lattice spacing, the Tierney-Kadane-corrected V) is checked against the true error it claims
to bound. Every number here is a math check [sim-only].
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp

from sv_pgs.scale_mixture_ep import (
    _HALF_PRECISION,
    AnnotationGroup,
    Cavity,
    GaussianPosterior,
    MixtureHyperparameters,
    _corrected,
    _data_objective,
    _data_value,
    _evidence,
    _penalty_matrix,
    _penalty_value,
    _restricted_prior,
    _smoothing_bounds,
    _total_curvature,
    cavities,
    class_log_density,
    hyper_step,
    initial_hyperparameters,
    kernel_floor,
    kernel_top,
    log_scale,
    normal_means_posterior,
    quadrature_majorant_ratio,
    scale_mixture_prior,
    site_targets,
    spacing_bound,
    tilted_moments,
)

_EPSILON = float(np.finfo(np.float64).eps)
_WORKING_BYTES = 1 << 22
# Scenario inputs, not model constants: the tolerance a lattice is built to, and the evidence resolution a fit
# certifies for a scorer with 64 posterior draws (1/(2K) nats).
_LATTICE_TOLERANCE = 1e-3
_EVIDENCE_TOLERANCE = 1.0 / 128.0
_KINDS = ("normal_means", "weak", "strong", "mixed")
_SEEDS = (101, 202, 303)


def _cavity(generator: np.random.Generator, kind: str, variant_count: int) -> Cavity:
    """Normal-means cavities: exp(-P b^2 / 2 + h b), the likelihood of one effect seen through P units of data."""
    if kind == "normal_means":
        precision = generator.uniform(50.0, 400.0, variant_count)
        effect = np.where(generator.random(variant_count) < 0.3, generator.normal(0.0, 0.25, variant_count), 0.0)
    elif kind == "weak":
        precision = generator.uniform(1.0, 10.0, variant_count)
        effect = np.where(generator.random(variant_count) < 0.3, generator.normal(0.0, 0.25, variant_count), 0.0)
    elif kind == "strong":
        precision = generator.uniform(1e3, 1e4, variant_count)
        effect = np.where(generator.random(variant_count) < 0.3, generator.normal(0.0, 0.3, variant_count), 0.0)
    else:
        precision = np.where(generator.random(variant_count) < 0.5, generator.uniform(1.0, 10.0, variant_count), generator.uniform(1e3, 1e4, variant_count))
        effect = np.where(generator.random(variant_count) < 0.3, generator.normal(0.0, 0.3, variant_count), 0.0)
    shift = precision * (effect + generator.standard_normal(variant_count) / np.sqrt(precision))
    return Cavity(precision=precision, shift=shift)


def _classes(generator: np.random.Generator, class_count: int, variant_count: int, weak_class: bool) -> np.ndarray:
    """Class labels with every class present; with ``weak_class`` the last class has two members."""
    labels = generator.integers(0, max(class_count - int(weak_class), 1), variant_count)
    labels[: class_count] = np.arange(class_count)
    if weak_class and class_count > 1:
        labels[labels == class_count - 1] = 0
        labels[[class_count - 1, class_count]] = class_count - 1
    return labels.astype(np.int64)


def _prior(generator: np.random.Generator, variant_count: int, nodes: np.ndarray, floor: float, top: float, *,
           class_count: int, annotated: bool, weak_class: bool = False):
    class_index = _classes(generator, class_count, variant_count, weak_class)
    offset = np.log(generator.uniform(0.2, 1.0, variant_count))
    if annotated:
        position = generator.uniform(-1.0, 1.0, variant_count)
        design = np.column_stack([position])
        groups = (AnnotationGroup(columns=np.array([0]), penalty=np.eye(1)),)
    else:
        design = np.zeros((variant_count, 0))
        groups = ()
    return scale_mixture_prior(
        class_index=class_index, log_variance_offset=offset, annotation_design=design, annotation_groups=groups,
        nodes=nodes, floor=floor, top=top,
    )


def _random_coefficients(generator: np.random.Generator, prior) -> np.ndarray:
    return initial_hyperparameters(prior).coefficients + 0.5 * generator.standard_normal(prior.coefficient_size)


def _hyperparameters(prior, coefficients: np.ndarray) -> MixtureHyperparameters:
    return MixtureHyperparameters(coefficients=coefficients, log_smoothing=np.zeros(len(prior.smoothing_blocks)))


# ---------------------------------------------------------------- 1. tilted moments against direct quadrature


def _node_integrals(variance: float, precision: float, shift: float) -> tuple[float, float, float, float]:
    """log of the integral of N(b; 0, v) exp(-P b^2 / 2 + h b), its normalized mean and variance, and the reference's
    relative error (QUADPACK's estimates, the moments scaled by the integrand's width), by adaptive quadrature around
    the integrand's own peak: no closed form is used."""
    precision_total = 1.0 / variance + precision
    peak = shift / precision_total
    width = 1.0 / np.sqrt(precision_total)

    def log_integrand(beta: float) -> float:
        return -0.5 * beta * beta / variance - 0.5 * np.log(2.0 * np.pi * variance) - 0.5 * precision * beta * beta + shift * beta

    peak_value = log_integrand(peak)
    # Beyond 40 widths the Gaussian integrand is below exp(-800) of its peak: nothing representable is dropped.
    lower, upper = peak - 40.0 * width, peak + 40.0 * width
    options = dict(epsabs=0.0, epsrel=1e-13, limit=400, points=[peak])
    zeroth, zeroth_error = quad(lambda beta: np.exp(log_integrand(beta) - peak_value), lower, upper, **options)
    first, first_error = quad(lambda beta: (beta - peak) * np.exp(log_integrand(beta) - peak_value), lower, upper, **options)
    second, second_error = quad(lambda beta: (beta - peak) ** 2 * np.exp(log_integrand(beta) - peak_value), lower, upper, **options)
    centred_mean = first / zeroth
    relative_error = max(zeroth_error / zeroth, first_error / (zeroth * width), second_error / (zeroth * width * width))
    return peak_value + np.log(zeroth), peak + centred_mean, second / zeroth - centred_mean**2, relative_error


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("kind", _KINDS)
def test_tilted_moments_match_direct_quadrature_in_every_regime(seed, kind):
    generator = np.random.default_rng(seed)
    variant_count = 5
    nodes = np.linspace(np.log(1e-5), np.log(1.0), 12)
    prior = _prior(generator, variant_count, nodes, nodes[0] - 1.0, nodes[-1], class_count=1 + seed % 3, annotated=bool(seed % 2))
    cavity = _cavity(generator, kind, variant_count)
    hyperparameters = _hyperparameters(prior, _random_coefficients(generator, prior))
    moments = tilted_moments(prior, hyperparameters, cavity, _WORKING_BYTES)
    log_density = class_log_density(prior, hyperparameters.coefficients)
    scales = log_scale(prior, hyperparameters.coefficients)
    for variant in range(variant_count):
        log_weights = log_density[prior.class_index[variant]] - logsumexp(log_density[prior.class_index[variant]])
        rows = [_node_integrals(float(np.exp(scales[variant] + node)), float(cavity.precision[variant]), float(cavity.shift[variant])) for node in nodes]
        log_node, node_mean, node_variance, node_error = (np.array(column) for column in zip(*rows))
        log_normalizer = float(logsumexp(log_weights + log_node))
        responsibility = np.exp(log_weights + log_node - log_normalizer)
        mean = float(np.sum(responsibility * node_mean))
        variance = float(np.sum(responsibility * (node_variance + np.square(node_mean - mean))))
        # The reference's own error (QUADPACK's estimates, carried through the mixture) plus double precision's
        # rounding of the K-term sums.
        reference_error = float(np.sum(responsibility * node_error))
        summation = 4.0 * nodes.shape[0] * _EPSILON
        assert abs(moments.log_normalizer[variant] - log_normalizer) <= reference_error + summation * (1.0 + abs(log_normalizer))
        scale = np.sqrt(variance)
        assert abs(moments.mean[variant] - mean) <= (reference_error + summation) * (abs(mean) + scale)
        assert abs(moments.variance[variant] - variance) <= 2.0 * (reference_error + summation) * (variance + mean * mean)


# ---------------------------------------------------------------- 2. the lattice certificates


def _flat_kernel_error(precision: np.ndarray, shift: np.ndarray, log_scale_values: np.ndarray, node: float) -> float:
    """sum_j |log L_j| at node t: L_j = Z_j(v) / Z_j(0), the Gaussian integral's ratio, by direct quadrature."""
    total = 0.0
    for precision_j, shift_j, scale_j in zip(precision, shift, log_scale_values):
        variance = float(np.exp(scale_j + node))
        log_node, _mean, _variance, _error = _node_integrals(variance, float(precision_j), float(shift_j))
        # Z_j(0) = 1: the flat kernel.
        total += abs(log_node)
    return total


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("kind", _KINDS)
def test_the_kernel_floor_bounds_the_flat_kernel_error_at_and_below_it(seed, kind):
    generator = np.random.default_rng(seed)
    variant_count = 8
    cavity = _cavity(generator, kind, variant_count)
    offset = np.log(generator.uniform(0.2, 1.0, variant_count))
    floor = kernel_floor(cavity.precision, cavity.shift, offset, _LATTICE_TOLERANCE)
    for depth in (0.0, 1.0, 5.0):
        error = _flat_kernel_error(cavity.precision, cavity.shift, offset, floor - depth)
        # The certificate: replacing every kernel at or below the floor by 1 moves sum_j log Z_j by at most the
        # tolerance (plus the quadrature reference's own relative accuracy).
        assert error <= _LATTICE_TOLERANCE * (1.0 + 1e-6)


def _smooth_log_density(nodes: np.ndarray, floor: float, top: float, bump_width: float) -> np.ndarray:
    """A continuous two-bump log density over t, spanning the kernel range, evaluated at ``nodes``; each bump's
    standard deviation in t is ``bump_width``."""
    width = max(top - floor, 1.0)
    first = -0.5 * np.square((nodes - (floor + 0.3 * width)) / bump_width)
    second = -0.5 * np.square((nodes - (floor + 0.8 * width)) / bump_width) + np.log(0.4)
    return np.logaddexp(first, second)


def _lattice_sum(cavity: Cavity, offset: np.ndarray, spacing: float, floor: float, top: float, bump_width: float) -> tuple[float, float]:
    """sum_j log Z_j on the lattice of ``spacing`` over the density's support, and sum_j M_j / Z_j there."""
    width = max(top - floor, 1.0)
    # The two-bump density is below exp(-800) of its peak beyond 40 bump widths from either end of the range.
    start, stop = floor - 40.0 * bump_width, top + 40.0 * bump_width
    nodes = np.arange(start, stop + spacing, spacing)
    prior = scale_mixture_prior(
        class_index=np.zeros(cavity.precision.shape[0], np.int64), log_variance_offset=offset,
        annotation_design=np.zeros((cavity.precision.shape[0], 0)), annotation_groups=(), nodes=nodes, floor=floor, top=top,
    )
    log_density = _smooth_log_density(nodes, floor, top, bump_width)
    basis = prior.coefficient_map[: prior.grid_size, : prior.pooled_size]
    coefficients = basis.T @ (log_density - log_density.mean())
    hyperparameters = _hyperparameters(prior, coefficients)
    total = float(np.sum(tilted_moments(prior, hyperparameters, cavity, _WORKING_BYTES).log_normalizer))
    return total, quadrature_majorant_ratio(prior, hyperparameters, cavity, _WORKING_BYTES)


def _certified_spacing_error(seed: int, kind: str, bump_width: float) -> float:
    generator = np.random.default_rng(seed)
    variant_count = 12
    cavity = _cavity(generator, kind, variant_count)
    offset = np.log(generator.uniform(0.2, 1.0, variant_count))
    floor = kernel_floor(cavity.precision, cavity.shift, offset, _LATTICE_TOLERANCE)
    top = kernel_top(cavity.precision, cavity.shift, offset, floor)
    # The spacing the certificate allows at the lattice it is evaluated on (every M_j / Z_j >= 1 starts it).
    spacing = spacing_bound(float(variant_count), _LATTICE_TOLERANCE)
    total, majorant = _lattice_sum(cavity, offset, spacing, floor, top, bump_width)
    while spacing > spacing_bound(majorant, _LATTICE_TOLERANCE):
        spacing = spacing_bound(majorant, _LATTICE_TOLERANCE)
        total, majorant = _lattice_sum(cavity, offset, spacing, floor, top, bump_width)
    # The reference: the same continuous density on a lattice sixteen times finer.
    reference, _majorant = _lattice_sum(cavity, offset, spacing / 16.0, floor, top, bump_width)
    return abs(total - reference)


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("kind", _KINDS)
def test_the_derived_spacing_certifies_the_trapezoid_error_for_a_smooth_density(seed, kind):
    """A density smooth on the strip's scale (bumps two units wide in t, beyond the strip's pi/2)."""
    assert _certified_spacing_error(seed, kind, bump_width=2.0) <= _LATTICE_TOLERANCE


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("kind", _KINDS)
def test_the_derived_spacing_certifies_the_trapezoid_error_for_a_sharp_density(seed, kind):
    """A density sharper than the strip (bumps 0.3 wide in t): the majorant must account for the density's own
    growth off the real axis, not only the kernel's."""
    assert _certified_spacing_error(seed, kind, bump_width=0.3) <= _LATTICE_TOLERANCE


# ---------------------------------------------------------------- 3. the objective's derivatives by differences of values


def _central_difference(function, point: np.ndarray, direction: np.ndarray, step: float) -> float:
    return (function(point + step * direction) - function(point - step * direction)) / (2.0 * step)


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("kind", _KINDS)
def test_the_objective_gradient_and_curvature_match_differences_of_its_value(seed, kind):
    generator = np.random.default_rng(seed)
    variant_count = 20
    nodes = np.linspace(np.log(1e-5), np.log(1.0), 10)
    prior = _prior(generator, variant_count, nodes, nodes[0] - 1.0, nodes[-1], class_count=1 + seed % 3, annotated=True, weak_class=seed % 2 == 0)
    cavity = _cavity(generator, kind, variant_count)
    point = _random_coefficients(generator, prior)
    mapping = prior.coefficient_map
    objective = _data_objective(prior, point, cavity, _WORKING_BYTES)
    gradient = mapping.T @ objective.gradient
    hessian = mapping.T @ objective.hessian @ mapping

    def value(coefficients):
        return _data_value(prior, coefficients, cavity, _WORKING_BYTES)

    def analytic_gradient(coefficients):
        return mapping.T @ _data_objective(prior, coefficients, cavity, _WORKING_BYTES).gradient

    # Each log Z_j is a log-sum-exp over the K nodes: K roundings of terms of size |log Z_j| at most.
    rounding = prior.grid_size * _EPSILON * objective.magnitude
    gradient_rounding = prior.grid_size * _EPSILON * (float(np.sum(np.abs(objective.gradient))) + variant_count) * float(np.linalg.norm(mapping, 2))
    for _draw in range(4):
        direction = generator.standard_normal(point.shape[0])
        direction /= np.linalg.norm(direction)
        # A central difference at s errs by T + r (truncation T = O(s^2), rounding |r| <= R / s); at s/2 by T/4 + r'
        # (|r'| <= 2R/s). Richardson: T/4 = (coarse - fine - r + r') / 3, so the fine difference errs by at most
        # |coarse - fine| / 3 + 3R/s; |coarse - fine| + 3R/s also covers the O(s^4) remainder. The step is 64 times
        # the balance point of truncation and rounding, so that the difference measures truncation.
        step = (3.0 * rounding / max(abs(float(direction @ hessian @ direction)), 1.0)) ** (1.0 / 3.0) * 64.0
        coarse = _central_difference(value, point, direction, step)
        fine = _central_difference(value, point, direction, 0.5 * step)
        assert abs(float(gradient @ direction) - fine) <= abs(coarse - fine) + 3.0 * rounding / step

        def directional(coefficients):
            return float(analytic_gradient(coefficients) @ direction)

        coarse_second = _central_difference(directional, point, direction, step)
        fine_second = _central_difference(directional, point, direction, 0.5 * step)
        second_bound = abs(coarse_second - fine_second) + 3.0 * gradient_rounding / step
        assert abs(float(direction @ hessian @ direction) - fine_second) <= second_bound


# ---------------------------------------------------------------- 4. EP on a correlated design: the evidence identities


def _ar1_genotypes(generator: np.random.Generator, variant_count: int, sample_count: int, correlation: float) -> np.ndarray:
    latent = generator.standard_normal((sample_count, variant_count))
    for column in range(1, variant_count):
        latent[:, column] = correlation * latent[:, column - 1] + np.sqrt(1.0 - correlation**2) * latent[:, column]
    return latent


def _gaussian_likelihood(generator: np.random.Generator, genotypes: np.ndarray):
    """Standardized genotypes, sparse effects and unit noise: the likelihood exp(-b' X'X b / 2 + (X'y)' b)."""
    genotypes = (genotypes - genotypes.mean(axis=0)) / genotypes.std(axis=0)
    variant_count = genotypes.shape[1]
    effects = np.where(generator.random(variant_count) < 0.3, generator.normal(0.0, 0.2, variant_count), 0.0)
    targets = genotypes @ effects + generator.standard_normal(genotypes.shape[0])
    return genotypes.T @ genotypes, genotypes.T @ targets


def _expectation_propagation(prior, coefficients, likelihood_precision, linear_term, sites=None):
    """Parallel EP on exp(-b' Lambda b / 2 + l' b) times the prior, to its fixed point. A step that would leave the
    posterior precision indefinite or a cavity improper is halved (a numerical safeguard of this reference, not
    part of the engine). Returns (site precision, site shift, covariance, cavity, tilted moments)."""
    variant_count = linear_term.shape[0]
    hyperparameters = _hyperparameters(prior, coefficients)
    if sites is None:
        site_precision = np.full(variant_count, 1.0)
        site_shift = np.zeros(variant_count)
    else:
        site_precision, site_shift = (np.array(part, copy=True) for part in sites)
    for _sweep in range(50000):
        covariance = np.linalg.inv(likelihood_precision + np.diag(site_precision))
        mean = covariance @ (linear_term + site_shift)
        cavity = cavities(mean, np.diag(covariance).copy(), site_precision, site_shift)
        moments = tilted_moments(prior, hyperparameters, cavity, _WORKING_BYTES)
        target_precision, target_shift = site_targets(moments, cavity)
        change = max(float(np.max(np.abs(target_precision - site_precision) / (1.0 + np.abs(site_precision)))),
                     float(np.max(np.abs(target_shift - site_shift) / (1.0 + np.abs(site_shift)))))
        if change <= 16.0 * _EPSILON:
            return site_precision, site_shift, covariance, cavity, moments
        fraction = 0.5
        while True:
            trial_precision = site_precision + fraction * (target_precision - site_precision)
            try:
                np.linalg.cholesky(likelihood_precision + np.diag(trial_precision))
                trial_covariance = np.linalg.inv(likelihood_precision + np.diag(trial_precision))
                if np.all(1.0 / np.diag(trial_covariance) - trial_precision > 0.0):
                    break
            except np.linalg.LinAlgError:
                pass
            fraction *= 0.5
        site_precision = trial_precision
        site_shift = site_shift + fraction * (target_shift - site_shift)
    raise AssertionError("the reference EP did not reach its fixed point")


def _log_ep_evidence(likelihood_precision, linear_term, site_precision, site_shift, cavity, moments) -> float:
    """log Z_EP up to its x-free likelihood constant: log of the integral of exp(-b' Lambda b / 2 + l' b) times the
    sites, plus sum_j (log Z_j - log of the integral of cavity_j times site_j)."""
    precision = likelihood_precision + np.diag(site_precision)
    factor = np.linalg.cholesky(precision)
    shift = linear_term + site_shift
    solved = np.linalg.solve(precision, shift)
    gaussian = 0.5 * shift.shape[0] * np.log(2.0 * np.pi) - float(np.sum(np.log(np.diag(factor)))) + 0.5 * float(shift @ solved)
    marginal_precision = cavity.precision + site_precision
    marginal_shift = cavity.shift + site_shift
    site_normalizers = 0.5 * np.log(2.0 * np.pi / marginal_precision) + 0.5 * np.square(marginal_shift) / marginal_precision
    return gaussian + float(np.sum(moments.log_normalizer - site_normalizers))


def _check_ep_evidence_derivatives(generator: np.random.Generator, genotypes: np.ndarray) -> None:
    """The fixed-cavity gradient and the total curvature B against differences of log Z_EP, EP re-solved at every
    point, along random directions."""
    variant_count = genotypes.shape[1]
    likelihood_precision, linear_term = _gaussian_likelihood(generator, genotypes)
    nodes = np.linspace(np.log(1e-4), np.log(0.3), 8)
    prior = _prior(generator, variant_count, nodes, nodes[0] - 1.0, nodes[-1], class_count=1 + int(generator.integers(2)), annotated=False)
    coefficients = initial_hyperparameters(prior).coefficients + 0.3 * generator.standard_normal(prior.coefficient_size)
    site_precision, site_shift, covariance, cavity, moments = _expectation_propagation(prior, coefficients, likelihood_precision, linear_term)
    mapping = prior.coefficient_map
    fixed_cavity_gradient = mapping.T @ _data_objective(prior, coefficients, cavity, _WORKING_BYTES).gradient
    posterior = GaussianPosterior(
        solve=lambda right: covariance @ right,
        variance_jvp=lambda weights: -np.einsum("jk,kr,kj->jr", covariance, weights, covariance),
    )
    total_curvature = _total_curvature(prior, coefficients, cavity, posterior, _WORKING_BYTES, 1e-13)

    def evidence(point):
        solved = _expectation_propagation(prior, point, likelihood_precision, linear_term, (site_precision, site_shift))
        return _log_ep_evidence(likelihood_precision, linear_term, solved[0], solved[1], solved[3], solved[4])

    base = evidence(coefficients)
    # log Z_EP's rounding: its Gaussian part solves with Lambda + diag(tau) (backward stable, so eps times its
    # condition number relatively), and each log Z_j rounds K terms.
    condition = float(np.linalg.cond(likelihood_precision + np.diag(site_precision)))
    rounding = _EPSILON * (condition * abs(base) + prior.grid_size * float(np.sum(np.abs(moments.log_normalizer))))
    for _draw in range(2):
        direction = generator.standard_normal(coefficients.shape[0])
        direction /= np.linalg.norm(direction)
        step = rounding ** (1.0 / 3.0)
        coarse = _central_difference(evidence, coefficients, direction, step)
        fine = _central_difference(evidence, coefficients, direction, 0.5 * step)
        # As in the objective's test: the fine difference errs by at most |coarse - fine| + 3R/s.
        assert abs(float(fixed_cavity_gradient @ direction) - fine) <= abs(coarse - fine) + 3.0 * rounding / step
        second_step = rounding ** 0.25
        plus, minus = evidence(coefficients + second_step * direction), evidence(coefficients - second_step * direction)
        half_plus, half_minus = evidence(coefficients + 0.5 * second_step * direction), evidence(coefficients - 0.5 * second_step * direction)
        second = (plus - 2.0 * base + minus) / second_step**2
        half_second = (half_plus - 2.0 * base + half_minus) / (0.25 * second_step**2)
        # A second difference at s rounds by at most 4R/s^2, at s/2 by 16R/s^2; Richardson as for the first
        # difference gives |coarse - fine| + (16 + 20/3) R/s^2 at most, within 24 R/s^2.
        second_bound = abs(second - half_second) + 24.0 * rounding / second_step**2
        assert abs(-float(direction @ total_curvature @ direction) - half_second) <= second_bound


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("correlation", (0.0, 0.7, 0.95))
def test_the_fixed_cavity_gradient_and_the_total_curvature_are_the_ep_evidence_derivatives(seed, correlation):
    generator = np.random.default_rng(seed)
    _check_ep_evidence_derivatives(generator, _ar1_genotypes(generator, 10, 300, correlation))


def test_the_ep_evidence_derivatives_on_public_1kgp_windows():
    """Real LD: windows of consecutive common biallelic SNVs from the public 1kGP high-coverage phased panel (EBI),
    as an .npz of dosage matrices named by ``SV_PGS_PUBLIC_WINDOWS``; skipped where no such file is given
    (``scripts`` in the verify-engine lane builds it on MSI). The traits are simulated [semi-real]."""
    import os

    path = os.environ.get("SV_PGS_PUBLIC_WINDOWS")
    if not path:
        pytest.skip("no public 1kGP windows given (SV_PGS_PUBLIC_WINDOWS)")
    with np.load(path) as windows:
        for position, name in enumerate(sorted(windows.files)):
            _check_ep_evidence_derivatives(np.random.default_rng(position), np.asarray(windows[name], dtype=np.float64))


# ---------------------------------------------------------------- 5. V's formula, recomputed independently


def _log_pseudo_determinant(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    kept = eigenvalues > _EPSILON * matrix.shape[0] * max(float(eigenvalues[-1]), 1.0)
    return float(np.sum(np.log(eigenvalues[kept]))), eigenvectors[:, ~kept]


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("kind", ("normal_means", "strong", "mixed"))
def test_the_evidence_is_the_profiled_laplace_value_at_a_stationary_point(seed, kind):
    generator = np.random.default_rng(seed)
    variant_count = 60
    nodes = np.linspace(np.log(1e-5), np.log(1.0), 10)
    prior = _prior(generator, variant_count, nodes, nodes[0] - 1.0, nodes[-1], class_count=1 + seed % 2, annotated=True)
    cavity = _cavity(generator, kind, variant_count)
    log_smoothing = generator.uniform(-1.0, 2.0, len(prior.smoothing_blocks))
    posterior = normal_means_posterior(cavity, _WORKING_BYTES)
    evidence = _evidence(prior, log_smoothing, initial_hyperparameters(prior).coefficients, cavity, posterior, _WORKING_BYTES, 0.0)
    if evidence is None:
        pytest.skip("no certified maximum at these weights: the engine refuses V there, which is its contract")
    point = evidence.coefficients
    mapping = prior.coefficient_map
    objective = _data_objective(prior, point, cavity, _WORKING_BYTES)
    penalty = _penalty_matrix(prior, log_smoothing)
    penalty_value, penalty_gradient = _penalty_value(prior, log_smoothing, point)
    gradient = mapping.T @ objective.gradient - penalty_gradient
    negative = -(mapping.T @ objective.hessian @ mapping) + penalty
    # Stationary: the penalized gradient is at the maximizer's certified resolution (its Newton decrement at the
    # objective's rounding).
    rounding = _EPSILON * (objective.magnitude + abs(objective.value - penalty_value))
    assert 0.5 * float(gradient @ np.linalg.solve(negative, gradient)) <= 64.0 * rounding
    # V recomputed from its definition, independently of the engine's profiled factor: for independent effects B is
    # the fixed-cavity curvature, and the null space N is the penalty's.
    log_penalty, null_basis = _log_pseudo_determinant(penalty)
    sign, log_total = np.linalg.slogdet(negative)
    assert sign > 0.0
    log_null = np.linalg.slogdet(null_basis.T @ negative @ null_basis)[1] if null_basis.shape[1] else 0.0
    reference = objective.value - penalty_value + 0.5 * log_penalty - 0.5 * log_total + 0.5 * log_null
    condition = float(np.linalg.cond(negative))
    assert abs(evidence.laplace_value - reference) <= 16.0 * negative.shape[0] * _EPSILON * condition * (1.0 + abs(reference))
    # The certificate: V at the evidence tolerance (an inexact inner maximum) is V at the stationary point to it.
    certified = _evidence(prior, log_smoothing, initial_hyperparameters(prior).coefficients, cavity, posterior, _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    if certified is not None:
        assert abs(certified.laplace_value - reference) <= _EVIDENCE_TOLERANCE


# ---------------------------------------------------------------- 6. the corrected V against the exact integral


def _profile(prior, log_smoothing, cavity, point, null_basis):
    """Maximize the penalized objective over the null directions from ``point``: Newton on the null block with
    backtracking (a step is kept only when it raises the objective), to double precision."""
    mapping = prior.coefficient_map
    penalty = _penalty_matrix(prior, log_smoothing)

    def penalized(coefficients):
        return _data_value(prior, coefficients, cavity, _WORKING_BYTES) - _penalty_value(prior, log_smoothing, coefficients)[0]

    current = np.array(point, copy=True)
    value = penalized(current)
    for _iteration in range(400):
        objective = _data_objective(prior, current, cavity, _WORKING_BYTES)
        gradient = mapping.T @ objective.gradient - penalty @ current
        hessian = mapping.T @ objective.hessian @ mapping - penalty
        null_gradient = null_basis.T @ gradient
        null_hessian = null_basis.T @ hessian @ null_basis
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (null_hessian + null_hessian.T))
        # Newton on the magnitudes of the curvature: an ascent direction wherever the block is not concave.
        step = eigenvectors @ ((eigenvectors.T @ null_gradient) / np.maximum(np.abs(eigenvalues), _EPSILON * float(np.max(np.abs(eigenvalues)))))
        length = 1.0
        while length * float(np.max(np.abs(step))) > _EPSILON * (1.0 + float(np.max(np.abs(current)))):
            trial = current + length * (null_basis @ step)
            trial_value = penalized(trial)
            if np.isfinite(trial_value) and trial_value > value:
                current, value = trial, trial_value
                break
            length *= 0.5
        else:
            return current, value
    return current, value


@pytest.mark.slow
@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("kind", ("normal_means", "weak"))
def test_the_corrected_evidence_matches_the_exact_integral_over_the_penalized_directions(seed, kind):
    """Two penalized directions (one class, five nodes: third-order roughness leaves a two-dimensional null space
    that is profiled), so the exact integral of exp(profiled objective) over them is a 2-D quadrature."""
    generator = np.random.default_rng(seed)
    variant_count = 40
    nodes = np.linspace(np.log(1e-4), np.log(1.0), 5)
    prior = _prior(generator, variant_count, nodes, nodes[0] - 1.0, nodes[-1], class_count=1, annotated=False)
    cavity = _cavity(generator, kind, variant_count)
    # Weights from near-singular B + S (small lambda over weak data) to a stiff penalty.
    log_smoothing = np.array([generator.uniform(-4.0, 2.0)])
    posterior = normal_means_posterior(cavity, _WORKING_BYTES)
    laplace = _evidence(prior, log_smoothing, initial_hyperparameters(prior).coefficients, cavity, posterior, _WORKING_BYTES, 0.0)
    if laplace is None:
        pytest.skip("no certified maximum at this weight")
    corrected = _corrected(prior, log_smoothing, laplace, cavity, posterior, _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    penalty = _penalty_matrix(prior, log_smoothing)
    log_penalty, null_basis = _log_pseudo_determinant(penalty)
    range_basis = np.linalg.svd(np.eye(penalty.shape[0]) - null_basis @ null_basis.T)[0][:, : penalty.shape[0] - null_basis.shape[1]]
    centre, peak = _profile(prior, log_smoothing, cavity, laplace.coefficients, null_basis)
    # Standardize the range coordinates by the profiled curvature at the maximum.
    mapping = prior.coefficient_map
    objective = _data_objective(prior, centre, cavity, _WORKING_BYTES)
    negative = -(mapping.T @ objective.hessian @ mapping) + penalty
    response = np.eye(negative.shape[0]) - null_basis @ np.linalg.solve(null_basis.T @ negative @ null_basis, null_basis.T @ negative)
    moved = response @ range_basis
    schur = moved.T @ negative @ moved
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (schur + schur.T))
    directions = range_basis @ eigenvectors / np.sqrt(eigenvalues)[None, :]
    # The exact 2-D integral of exp(profile - peak) in the standardized coordinates, on a Gauss-Hermite grid
    # widened until the integral stops changing to the tolerance's share.
    previous = None
    for order in (24, 48, 96):
        abscissae, weights = np.polynomial.hermite_e.hermegauss(order)
        total = 0.0
        for first, first_weight in zip(abscissae, weights):
            for second, second_weight in zip(abscissae, weights):
                start = centre + first * directions[:, 0] + second * directions[:, 1]
                _point, value = _profile(prior, log_smoothing, cavity, start, null_basis)
                standard_normal = -0.5 * (first * first + second * second)
                total += first_weight * second_weight * np.exp(value - peak - standard_normal)
        log_integral = float(np.log(total))
        if previous is not None and abs(log_integral - previous) <= 0.25 * _EVIDENCE_TOLERANCE:
            break
        previous = log_integral
    else:
        pytest.fail("the reference quadrature did not converge: the harness cannot judge this case")
    # The exact profiled evidence: F + 1/2 log|S|_+ + log of the integral over the range - (r/2) log(2 pi), where the
    # standardized coordinates carry the Schur determinant.
    exact = peak + 0.5 * log_penalty + log_integral - 0.5 * float(np.sum(np.log(eigenvalues))) - 0.5 * eigenvalues.shape[0] * np.log(2.0 * np.pi)
    assert corrected is not None
    assert abs(corrected.value - exact) <= _EVIDENCE_TOLERANCE, (
        f"corrected V {corrected.value:.6f} vs exact {exact:.6f} (Laplace {laplace.laplace_value:.6f})"
    )


# ---------------------------------------------------------------- 7. the penalty weights' edges


def _edge_problem(seed: int, *, annotation_effect: float = 0.0):
    """One class, ten nodes and one annotation column whose true effect on log u is ``annotation_effect``."""
    generator = np.random.default_rng(seed)
    variant_count = 80
    nodes = np.linspace(np.log(1e-5), np.log(1.0), 10)
    position = generator.uniform(-1.0, 1.0, variant_count)
    offset = np.log(generator.uniform(0.2, 1.0, variant_count))
    prior = scale_mixture_prior(
        class_index=np.zeros(variant_count, np.int64), log_variance_offset=offset, annotation_design=position[:, None],
        annotation_groups=(AnnotationGroup(columns=np.array([0]), penalty=np.eye(1)),), nodes=nodes, floor=nodes[0] - 1.0, top=nodes[-1],
    )
    precision = generator.uniform(50.0, 400.0, variant_count)
    effect = np.where(generator.random(variant_count) < 0.4, generator.normal(0.0, 0.25, variant_count), 0.0) * np.exp(0.5 * annotation_effect * position)
    shift = precision * (effect + generator.standard_normal(variant_count) / np.sqrt(precision))
    return prior, Cavity(precision=precision, shift=shift)


def _laplace_path(prior, cavity, block: int, rhos: np.ndarray, start: np.ndarray):
    """The Laplace V (tolerance 0: at the stationary point) along rho_block, the other weight at 0, each fit warm
    started from the previous one; with each value's rounding bound (dimension times eps times the condition number
    of B + S, relative)."""
    posterior = normal_means_posterior(cavity, _WORKING_BYTES)
    values, roundings, point = [], [], start
    for rho in rhos:
        weights = np.zeros(len(prior.smoothing_blocks))
        weights[block] = rho
        evidence = _evidence(prior, weights, point, cavity, posterior, _WORKING_BYTES, 0.0)
        assert evidence is not None
        objective = _data_objective(prior, evidence.coefficients, cavity, _WORKING_BYTES)
        negative = -(prior.coefficient_map.T @ objective.hessian @ prior.coefficient_map) + _penalty_matrix(prior, weights)
        values.append(evidence.laplace_value)
        roundings.append(16.0 * negative.shape[0] * _EPSILON * float(np.linalg.cond(negative)) * (1.0 + abs(evidence.laplace_value)))
        point = evidence.coefficients
    return np.array(values), np.array(roundings), point


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("block", (0, 1))
def test_the_infinity_edge_is_the_limit_of_the_evidence(seed, block):
    """V(rho) -> V(lambda = infinity) as rho -> infinity, with V - V_inf = a e^-rho + O(e^-2rho) (V is smooth in
    1/lambda there): Richardson's extrapolation from rho and rho + 2 lands on the edge's V to within its difference
    from the extrapolation one step earlier (whose remainder is e^4 times larger)."""
    prior, cavity = _edge_problem(seed)
    posterior = normal_means_posterior(cavity, _WORKING_BYTES)
    base = _evidence(prior, np.zeros(2), initial_hyperparameters(prior).coefficients, cavity, posterior, _WORKING_BYTES, 0.0)
    assert base is not None
    upper = _smoothing_bounds(prior, _data_objective(prior, base.coefficients, cavity, _WORKING_BYTES))[block][1]
    # Where the penalty's largest eigenvalue meets the data's curvature (half of double precision below the upper bound),
    # then two, four and six units stiffer.
    balance = upper + np.log(_HALF_PRECISION)
    step = 2.0
    rhos = balance + step * np.arange(1.0, 4.0)
    values, roundings, point = _laplace_path(prior, cavity, block, rhos, base.coefficients)
    view, allowed = _restricted_prior(prior, frozenset({block}), frozenset())
    edge = _evidence(view, np.zeros(1), allowed.T @ point, cavity, posterior, _WORKING_BYTES, 0.0)
    assert edge is not None
    near = values[2] + (values[2] - values[1]) / np.expm1(step)
    far = values[1] + (values[1] - values[0]) / np.expm1(step)
    rounding = float(np.sum(roundings)) * (1.0 + 2.0 / np.expm1(step))
    assert abs(near - edge.laplace_value) <= abs(near - far) + rounding, (values, edge.laplace_value)


@pytest.mark.parametrize("seed", _SEEDS)
def test_every_interior_evidence_is_below_the_zero_edge(seed):
    """At lambda_i = 0 block i is dropped and its directions profiled (``_restricted_prior``): V_0 is the profile,
    and every proper-prior V(rho) is at most it. For the Laplace forms: V(rho) - V_0 = [max of the penalized
    objective - max of the unpenalized] + 1/2 log|lambda S (B~ + lambda S)^-1| over the block's directions, both
    terms <= 0 (B~ the profiled data curvature there). So V falls without bound toward rho = -infinity (slope
    r_i / 2) while V_0 stays above every interior value [sim-only]."""
    prior, cavity = _edge_problem(seed)
    posterior = normal_means_posterior(cavity, _WORKING_BYTES)
    base = _evidence(prior, np.zeros(2), initial_hyperparameters(prior).coefficients, cavity, posterior, _WORKING_BYTES, 0.0)
    assert base is not None
    lower, upper = _smoothing_bounds(prior, _data_objective(prior, base.coefficients, cavity, _WORKING_BYTES))[1]
    rhos = np.linspace(lower, upper, 9)
    values, roundings, _point = _laplace_path(prior, cavity, 1, rhos, base.coefficients)
    view, allowed = _restricted_prior(prior, frozenset(), frozenset({1}))
    zero_edge = _evidence(view, np.zeros(1), allowed.T @ base.coefficients, cavity, posterior, _WORKING_BYTES, 0.0)
    assert zero_edge is not None
    assert np.all(values <= zero_edge.laplace_value + roundings), (values, zero_edge.laplace_value)


@pytest.mark.slow
@pytest.mark.parametrize("seed", _SEEDS)
def test_a_null_annotation_is_not_left_unpenalized(seed):
    """An annotation with no effect on the variances: the fitted weight should not sit at the zero edge, where the
    annotation's coefficient is its unpenalized maximum. By the bound above, V_0 exceeds every interior V by the
    annotation's Occam term, so a certified zero edge wins the edge comparison whenever that term passes the
    tolerance [sim-only]."""
    prior, cavity = _edge_problem(seed, annotation_effect=0.0)
    step = hyper_step(prior, initial_hyperparameters(prior), cavity, normal_means_posterior(cavity, _WORKING_BYTES), _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    assert step.hyperparameters.log_smoothing[1] != -np.inf, step.hyperparameters.log_smoothing


@pytest.mark.slow
@pytest.mark.parametrize("seed", _SEEDS)
def test_the_stationarity_certificate_bounds_the_gain_of_nearby_weights(seed):
    """``stationarity_gain`` is claimed to bound the gain a Newton step on the weights could still find. Scan each
    interior weight over a neighbourhood in rho (0.1, 0.3 and 1 either side: a Newton step's reach where V is flat in
    rho) and compare the best certified V with the fitted one: both are certified to the tolerance, so the gain may
    exceed the claimed bound by at most two tolerances."""
    prior, cavity = _edge_problem(seed, annotation_effect=1.0)
    posterior = normal_means_posterior(cavity, _WORKING_BYTES)
    step = hyper_step(prior, initial_hyperparameters(prior), cavity, posterior, _WORKING_BYTES, _EVIDENCE_TOLERANCE)
    fitted = step.hyperparameters
    infinite = frozenset(int(position) for position in np.flatnonzero(fitted.log_smoothing == np.inf))
    zero = frozenset(int(position) for position in np.flatnonzero(fitted.log_smoothing == -np.inf))
    view, allowed = _restricted_prior(prior, infinite, zero)
    weights = fitted.log_smoothing[np.isfinite(fitted.log_smoothing)]
    if weights.shape[0] == 0:
        pytest.skip("every weight at an edge: no interior weight to scan")
    base = _corrected(
        view, weights, _evidence(view, weights, allowed.T @ fitted.coefficients, cavity, posterior, _WORKING_BYTES, 0.0), cavity, posterior,
        _WORKING_BYTES, _EVIDENCE_TOLERANCE,
    )
    assert base is not None
    best = base.value
    gains = {}
    for position in range(weights.shape[0]):
        for distance in (-1.0, -0.3, -0.1, 0.1, 0.3, 1.0):
            moved_weights = weights.copy()
            moved_weights[position] += distance
            moved = _corrected(
                view, moved_weights, _evidence(view, moved_weights, base.coefficients, cavity, posterior, _WORKING_BYTES, 0.0), cavity, posterior,
                _WORKING_BYTES, _EVIDENCE_TOLERANCE,
            )
            if moved is not None:
                gains[(position, distance)] = moved.value - base.value
                best = max(best, moved.value)
    assert best - base.value <= step.stationarity_gain + 2.0 * _EVIDENCE_TOLERANCE, (gains, step.stationarity_gain, fitted.log_smoothing)
