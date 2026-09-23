"""The collapsed scale sampler (sv_pgs/scale_sampler.py) against exact enumeration of every scale configuration, the
audit's identical-proxy example, the identities its empirical Bayes rests on, and one empirical-Bayes recovery on an
own simulation. Every sampled answer is compared within its own measured Monte Carlo error. Synthetic data only."""

import itertools
import math

import numpy as np
import pytest

from sv_pgs import scale_sampler as sampler
from sv_pgs.scale_mixture_ep import AnnotationGroup, _components, scale_mixture_prior

# The audit's identical-proxy example: the exact posterior mean of each of two identical columns.
_IDENTICAL_PROXY_MEAN = 0.742691
# Monte Carlo agreement: a sampled estimate within this many of its own standard errors of the exact value (a
# two-sided normal tail of 6e-5 per comparison).
_STANDARD_ERRORS = 4.0


def _brute_force(precision, shift, variances, log_weights):
    """Every configuration's posterior weight and Gaussian moments by dense algebra: (log Z, mean, covariance,
    node marginals, E[beta^2 / v], E[beta^4 / v^2]) of prod_j p_j(beta_j) exp(-beta' Lambda beta / 2 + h' beta)."""
    size, nodes = variances.shape
    logs, means, seconds, fourths, states = [], [], [], [], []
    for state in itertools.product(range(nodes), repeat=size):
        diagonal = variances[np.arange(size), state]
        covariance = np.linalg.inv(precision + np.diag(1.0 / diagonal))
        mean = covariance @ shift
        sign, log_det = np.linalg.slogdet(np.eye(size) + np.diag(diagonal) @ precision)
        assert sign > 0
        logs.append(float(np.sum(log_weights[np.arange(size), state])) - 0.5 * log_det + 0.5 * float(shift @ mean))
        means.append(mean)
        seconds.append(covariance + np.outer(mean, mean))
        variance = np.diag(covariance)
        fourths.append((3.0 * variance**2 + 6.0 * variance * mean**2 + mean**4) / diagonal**2)
        states.append(state)
    logs = np.array(logs)
    weights = np.exp(logs - logs.max())
    total = weights.sum()
    weights /= total
    mean = np.einsum("c,cj->j", weights, np.array(means))
    second = np.einsum("c,cjk->jk", weights, np.array(seconds))
    marginals = np.zeros((size, nodes))
    scaled = np.zeros((size, nodes))
    fourth = np.zeros(size)
    for weight, state, moment, extra in zip(weights, states, seconds, fourths):
        for member in range(size):
            marginals[member, state[member]] += weight
            scaled[member, state[member]] += weight * moment[member, member] / variances[member, state[member]]
        fourth += weight * extra
    return float(logs.max() + np.log(total)), mean, second - np.outer(mean, mean), marginals, scaled, fourth


def _lattice_prior(rng, size, nodes):
    grid = np.linspace(-6.0, 1.0, nodes)
    log_weights = rng.normal(size=(size, nodes))
    log_weights -= np.log(np.exp(log_weights).sum(axis=1, keepdims=True))
    return sampler.NodePrior.build(log_weights, rng.normal(scale=0.3, size=size), grid)


def _correlated_cavity(rng, size, samples, correlation):
    """(Lambda, h) of a Gaussian likelihood X'X, X'y over columns sharing a common factor of this correlation."""
    common = rng.standard_normal(samples)
    columns = np.sqrt(correlation) * common[:, None] + np.sqrt(1.0 - correlation) * rng.standard_normal((samples, size))
    target = columns[:, 0] * 0.4 + rng.standard_normal(samples)
    return columns.T @ columns, columns.T @ target


def test_the_rank_one_algebra_is_the_dense_conditional():
    rng = np.random.default_rng(0)
    size = 5
    root = rng.standard_normal((8, size))
    precision = root.T @ root
    shift = rng.standard_normal(size)
    variances = np.exp(rng.uniform(-12.0, 2.0, (size, 1)))
    gram, field, log_likelihood, ok = sampler._cluster_start(precision, shift, variances, np.zeros(size, dtype=np.int64))
    assert ok
    diagonal = np.diag(variances[:, 0])
    np.testing.assert_allclose(gram, precision @ np.linalg.inv(np.eye(size) + diagonal @ precision), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(field, np.linalg.solve(np.eye(size) + precision @ diagonal, shift), rtol=1e-10)
    covariance = np.linalg.inv(precision + np.linalg.inv(diagonal))
    np.testing.assert_allclose(diagonal @ field, covariance @ shift, rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(diagonal - diagonal @ gram @ diagonal, covariance, rtol=1e-8, atol=1e-18)
    expected = -0.5 * np.linalg.slogdet(np.eye(size) + diagonal @ precision)[1] + 0.5 * shift @ covariance @ shift
    np.testing.assert_allclose(log_likelihood, expected, rtol=1e-12)


@pytest.mark.parametrize("indefinite", [False, True])
def test_enumeration_is_the_exact_tilted_law(indefinite):
    rng = np.random.default_rng(1)
    size, nodes = 3, 4
    prior = _lattice_prior(rng, size, nodes)
    precision, shift = _correlated_cavity(rng, size, 30, 0.8)
    if indefinite:
        # A negative cavity direction the prior's largest variance still bounds: Lambda + D^-1 stays positive definite.
        precision = precision - (np.linalg.eigvalsh(precision)[0] + 0.1) * np.eye(size)
        assert np.linalg.eigvalsh(precision)[0] < 0.0 < np.linalg.eigvalsh(precision + np.diag(1.0 / prior.variances.max(axis=1)))[0]
    log_z, mean, covariance = sampler.exact_tilted_moments(prior, precision, shift)
    expected = _brute_force(precision, shift, prior.variances, prior.log_weights)
    np.testing.assert_allclose(log_z, expected[0], rtol=1e-12)
    np.testing.assert_allclose(mean, expected[1], rtol=1e-9, atol=1e-13)
    np.testing.assert_allclose(covariance, expected[2], rtol=1e-8, atol=1e-13)


def test_a_cluster_of_one_is_eps_single_site_tilted_law():
    rng = np.random.default_rng(2)
    nodes = 9
    prior = _lattice_prior(rng, 1, nodes)
    precision, shift = np.array([[37.0]]), np.array([4.2])
    log_z, mean, covariance = sampler.exact_tilted_moments(prior, precision, shift)
    terms = _components(prior.log_weights[0], prior.log_scale, prior.grid, precision[0], shift)
    conditional = terms.conditional_variance[0]
    weights = terms.responsibility[0]
    np.testing.assert_allclose(log_z, terms.log_normalizer[0], rtol=1e-12)
    np.testing.assert_allclose(mean[0], shift[0] * weights @ conditional, rtol=1e-12)
    first = weights @ conditional
    np.testing.assert_allclose(covariance[0, 0], first + shift[0] ** 2 * weights @ (conditional - first) ** 2, rtol=1e-10)


def _identical_proxy_prior(members=2):
    """Two nodes, variances 1e-5 and 1 with prior weights 0.95 and 0.05, for every member."""
    return sampler.NodePrior.build(np.tile(np.log([0.95, 0.05]), (members, 1)), np.zeros(members), np.log([1e-5, 1.0]))


def _identical_proxy_data():
    x = np.repeat([1.0, -1.0], 50)
    return x, 1.5 * x


def test_the_identical_proxy_example_is_exact_and_symmetric_by_enumeration():
    x, y = _identical_proxy_data()
    columns = np.column_stack([x, x])
    _log_z, mean, _covariance = sampler.exact_tilted_moments(_identical_proxy_prior(), columns.T @ columns, columns.T @ y)
    np.testing.assert_allclose(mean, [_IDENTICAL_PROXY_MEAN, _IDENTICAL_PROXY_MEAN], atol=5e-7)
    assert mean[0] == mean[1]


def test_the_local_chain_reaches_the_identical_proxy_answer_from_a_one_sided_start():
    """The chain starts with the whole signal on the first column; the scale exchange moves it to its twin in one step,
    so the Rao-Blackwellized means agree with the exact ones and with each other within their Monte Carlo errors."""
    x, y = _identical_proxy_data()
    columns = np.column_stack([x, x])
    prior = _identical_proxy_prior()
    generator = np.array([12345], dtype=np.uint64)
    state = np.array([1, 0], dtype=np.int64)
    mean, _second, mc_variance, _occupancy, sweeps, resolved, ok = sampler._cluster_chain(
        columns.T @ columns, columns.T @ y, prior.log_weights, prior.variances, prior.log_scale, prior.spacing, state, generator, 64
    )
    assert ok and resolved and sweeps >= 128
    error = np.sqrt(mc_variance)
    assert np.all(np.abs(mean - _IDENTICAL_PROXY_MEAN) <= _STANDARD_ERRORS * error + 1e-12)
    assert abs(mean[0] - mean[1]) <= _STANDARD_ERRORS * np.sqrt(2.0) * error.max() + 1e-12


def test_sampled_clusters_agree_with_enumeration_within_their_monte_carlo_error():
    rng = np.random.default_rng(3)
    nodes = 9
    sizes = (3, 3, 4, 2, 1)
    prior = _lattice_prior(rng, sum(sizes), nodes)
    clusters, precisions, shifts = [], [], []
    start = 0
    for size, correlation in zip(sizes, (0.99, 0.5, 0.9, 0.0, 0.0)):
        clusters.append(np.arange(start, start + size))
        precision, shift = _correlated_cavity(rng, size, 60, correlation)
        precisions.append(precision)
        shifts.append(shift)
        start += size
    draw_count = 16
    moments = sampler.cluster_tilted_moments(prior, clusters, precisions, shifts, draw_count, seed=7)
    assert moments.proper.all()
    # K^(m-1) <= m draw_count: the clusters of one and two are enumerated, the rest sampled.
    np.testing.assert_array_equal(moments.exact, [False, False, False, True, True])
    for index, members in enumerate(clusters):
        mean, covariance = moments.cluster(index)
        log_z, exact_mean, exact_covariance = sampler.exact_tilted_moments(prior.rows(members), precisions[index], shifts[index])
        start, stop = moments.member_offsets[index], moments.member_offsets[index + 1]
        error = moments.mean_error[start:stop]
        if moments.exact[index]:
            np.testing.assert_allclose(mean, exact_mean, rtol=1e-10, atol=1e-13)
            np.testing.assert_allclose(covariance, exact_covariance, rtol=1e-9, atol=1e-13)
            np.testing.assert_allclose(moments.log_normalizer[index], log_z, rtol=1e-12)
            continue
        assert moments.resolved[index]
        assert np.all(np.abs(mean - exact_mean) <= _STANDARD_ERRORS * error)
        # Every mean resolved to draw_count effective draws: its error is at most its spread over sqrt(draw_count).
        assert np.all(error <= np.sqrt(np.diag(covariance) / draw_count) * (1.0 + 1e-12))
        assert abs(moments.log_normalizer[index] - log_z) <= _STANDARD_ERRORS * moments.log_normalizer_error[index] + 1e-12
        # The covariance is a sampled average of exact Gaussian covariances: within a few spreads of the exact one.
        np.testing.assert_allclose(covariance, exact_covariance, atol=_STANDARD_ERRORS * float(np.max(np.diag(exact_covariance))) / np.sqrt(draw_count))


def test_the_local_engine_is_reproducible_whatever_the_threads():
    import numba

    rng = np.random.default_rng(4)
    prior = _lattice_prior(rng, 12, 9)
    clusters = [np.arange(0, 4), np.arange(4, 8), np.arange(8, 12)]
    cavities = [_correlated_cavity(rng, 4, 40, 0.9) for _ in clusters]
    arguments = (prior, clusters, [cavity[0] for cavity in cavities], [cavity[1] for cavity in cavities], 8)
    threads = numba.get_num_threads()
    first = sampler.cluster_tilted_moments(*arguments, seed=11)
    numba.set_num_threads(1)
    try:
        second = sampler.cluster_tilted_moments(*arguments, seed=11)
    finally:
        numba.set_num_threads(threads)
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.state, second.state)


def _global_problem(rng, samples, size, nodes):
    columns = rng.standard_normal((samples, size))
    columns[:, 1] = columns[:, 0] + 0.1 * rng.standard_normal(samples)
    covariates = np.ones((samples, 1))
    target = 0.5 * columns[:, 0] + rng.standard_normal(samples) + 3.0
    basis = np.linalg.qr(covariates)[0]
    design = sampler.SampleSideDesign.build(columns, target, basis, np.arange(size))
    return design, _lattice_prior(rng, size, nodes)


def test_the_global_sampler_matches_enumeration():
    rng = np.random.default_rng(5)
    design, prior = _global_problem(rng, 40, 4, 5)
    noise = 0.9
    precision = design.columns.T @ design.columns / noise
    shift = design.columns.T @ design.target / noise
    expected = _brute_force(precision, shift, prior.variances, prior.log_weights)
    posterior = sampler.sample_posterior(design, prior, noise, draw_count=64, seed=3)
    assert posterior.status == "resolved"
    error = np.sqrt(sampler.batch_mc_variance(posterior.batch_means))
    assert np.all(np.abs(posterior.mean - expected[1]) <= _STANDARD_ERRORS * error)
    # The occupancies are Rao-Blackwellized averages too: within a few percent of the exact marginals.
    np.testing.assert_allclose(posterior.occupancy, expected[3], atol=0.03)


def test_the_panel_updates_keep_the_kernel_inverse():
    rng = np.random.default_rng(6)
    design, prior = _global_problem(rng, 30, 12, 5)
    generator = np.random.default_rng(1)
    chain = sampler._Chain(design, sampler._prior_draws(prior, generator), generator)
    statistics = chain.run(prior, 0.7, 3)
    weights = np.bincount(design.group, weights=prior.variances[np.arange(design.member_count), chain.state], minlength=design.columns.shape[1])
    kernel = 0.7 * np.eye(design.dimension) + (design.columns * weights[None, :]) @ design.columns.T
    np.testing.assert_allclose(chain.inverse, np.linalg.inv(kernel), rtol=1e-9, atol=1e-11)
    assert statistics.sweeps == 3


@pytest.mark.parametrize("tied", [False, True])
def test_the_global_sampler_reproduces_the_identical_proxy_example(tied):
    """Two identical columns, as two groups (a singular 2 x 2 Gram) or as one exact-tie group with two members: the
    posterior means are the exact (0.742691, 0.742691) and equal, within their Monte Carlo errors."""
    x, y = _identical_proxy_data()
    carriers = x[:, None] if tied else np.column_stack([x, x])
    group = np.array([0, 0]) if tied else np.array([0, 1])
    design = sampler.SampleSideDesign.build(carriers, y, np.zeros((100, 0)), group)
    assert list(design.partner) == [1, 0]
    posterior = sampler.sample_posterior(design, _identical_proxy_prior(), 1.0, draw_count=64, seed=9)
    assert posterior.status == "resolved"
    error = np.sqrt(sampler.batch_mc_variance(posterior.batch_means))
    assert np.all(np.abs(posterior.mean - _IDENTICAL_PROXY_MEAN) <= _STANDARD_ERRORS * error)
    difference = np.array([values[:, 0] - values[:, 1] for values in posterior.batch_means])
    spread = math.sqrt(float(np.mean(difference.var(axis=1, ddof=1) / difference.shape[1])) / len(posterior.batch_means))
    assert abs(posterior.mean[0] - posterior.mean[1]) <= _STANDARD_ERRORS * spread + 1e-12


def _annotated_prior(size, nodes):
    annotation = np.linspace(-1.0, 1.0, size)[:, None]
    return scale_mixture_prior(
        class_index=np.zeros(size, dtype=np.int64), log_variance_offset=np.zeros(size), annotation_design=annotation,
        annotation_groups=(AnnotationGroup(columns=np.array([0]), penalty=np.eye(1)),), nodes=np.linspace(-5.0, 1.0, nodes), floor=-4.0, top=0.0,
    )


def _exact_log_evidence(prior, coefficients, precision, shift):
    node = sampler.NodePrior.of(prior, coefficients)
    return _brute_force(precision, shift, node.variances, node.log_weights)


@pytest.mark.parametrize("orthogonal", [True, False])
def test_the_fisher_score_and_louis_information_are_the_evidences_derivatives(orthogonal):
    """log p(y | x) is log Z of the data's cavity under the prior: its gradient in x is the complete-data score's
    posterior mean exactly (Fisher), and its Hessian is Louis' observed information, exactly where the posterior
    factorizes over members (orthogonal columns) and up to the members' score covariance otherwise."""
    rng = np.random.default_rng(8)
    size, nodes = 3, 5
    prior = _annotated_prior(size, nodes)
    coefficients = rng.normal(scale=0.5, size=prior.coefficient_size)
    if orthogonal:
        precision = np.diag([30.0, 12.0, 50.0])
    else:
        precision, _ = _correlated_cavity(rng, size, 40, 0.7)
    shift = np.array([6.0, -2.0, 9.0])
    exact = _exact_log_evidence(prior, coefficients, precision, shift)
    mapping = prior.coefficient_map
    gradient = mapping.T @ sampler.complete_data_score(prior, coefficients, exact[3], exact[4])
    hessian = mapping.T @ sampler.observed_information(prior, coefficients, exact[3], exact[4], exact[5]) @ mapping
    step = 1e-5
    numeric = np.empty(prior.coefficient_size)
    numeric_hessian = np.empty((prior.coefficient_size, prior.coefficient_size))
    for index in range(prior.coefficient_size):
        move = np.zeros(prior.coefficient_size)
        move[index] = step
        upper = _exact_log_evidence(prior, coefficients + move, precision, shift)
        lower = _exact_log_evidence(prior, coefficients - move, precision, shift)
        numeric[index] = (upper[0] - lower[0]) / (2.0 * step)
        numeric_hessian[:, index] = (
            mapping.T @ sampler.complete_data_score(prior, coefficients + move, upper[3], upper[4])
            - mapping.T @ sampler.complete_data_score(prior, coefficients - move, lower[3], lower[4])
        ) / (2.0 * step)
    np.testing.assert_allclose(gradient, numeric, rtol=1e-6, atol=1e-7)
    if orthogonal:
        np.testing.assert_allclose(hessian, numeric_hessian, rtol=1e-5, atol=1e-6)
    else:
        # The members' score covariance left out is a small part: the metric stays within 25% of the exact Hessian.
        assert np.linalg.norm(hessian - numeric_hessian) <= 0.25 * np.linalg.norm(numeric_hessian)


def test_empirical_bayes_recovers_the_noise_and_the_annotation_slope():
    """[own-sim] Effects drawn from a known annotated prior on independent genotype columns: the empirical Bayes on the
    marginal likelihood recovers the noise variance and the annotation's log-variance slope within their sampling
    spread, and the fit's posterior is resolved."""
    rng = np.random.default_rng(10)
    samples, size = 500, 80
    frequencies = rng.uniform(0.1, 0.5, size)
    codes = (rng.binomial(2, frequencies, (samples, size)) * 127).astype(np.uint8)
    annotation = rng.standard_normal(size)
    slope = 1.5
    effects = rng.standard_normal(size) * np.sqrt(0.02 * np.exp(slope * (annotation - annotation.mean())))
    standardized = (codes.astype(np.float64) - codes.mean(axis=0)) / codes.std(axis=0)
    target = standardized @ effects + rng.standard_normal(samples)
    fit = sampler.fit_scale_sampler(
        codes=codes, covariates=np.ones((samples, 1)), target=target, variant_class=np.zeros(size, dtype=np.int64), log_variance_offset=None,
        draw_count=16, working_bytes=1 << 30, seed=4, annotations={"context": annotation},
    )
    assert fit.empirical_bayes.status == "resolved"
    assert fit.posterior.status == "resolved"
    assert abs(fit.empirical_bayes.noise - 1.0) <= 0.2
    prior = fit.prior
    names = [block.name for block in prior.smoothing_blocks]
    assert any(name.startswith("annotation") for name in names)
    scales = sampler.log_scale(prior, fit.empirical_bayes.hyperparameters.coefficients)
    fitted_slope = np.polyfit(annotation, scales, 1)[0]
    assert abs(fitted_slope - slope) <= 0.75
    truth = standardized @ effects
    prediction = standardized @ fit.coefficients
    assert np.corrcoef(truth, prediction)[0, 1] ** 2 > 0.5
