"""The small-n route (sv_pgs/small_n.py): its kernel-form algebra against brute-force dense inverses, its Stage 0
against the definitions, and one end-to-end fit on synthetic genotypes. Synthetic data only."""

import numpy as np
import pytest

from sv_pgs.config import VariantClass
from sv_pgs.fast_scoring import SIGNED_CODE_OFFSET
from sv_pgs.small_n import _DensePosterior, _Design, _Kernel, _new_profile, dense_statistics, fit_small_n

_ROUNDING = 1e-9


def _design(rng, samples, variants):
    return np.asfortranarray(rng.standard_normal((samples, variants)))


def _dense_inverse(design, precision):
    return np.linalg.inv(design.T @ design + np.diag(precision))


@pytest.mark.parametrize("negative", [0, 3])
def test_kernel_solve_variances_and_jvp_match_the_dense_inverse(negative):
    rng = np.random.default_rng(1 + negative)
    design = _design(rng, 12, 30)
    precision = rng.uniform(0.5, 2.0, 30)
    # Negative sites on columns the data resolve: A stays positive definite.
    precision[:negative] = -0.05
    inverse = _dense_inverse(design, precision)
    assert np.all(np.linalg.eigvalsh(design.T @ design + np.diag(precision)) > 0.0)
    kernel = _Kernel(_Design.dense(design), precision)
    right = rng.standard_normal((30, 4))
    np.testing.assert_allclose(kernel.solve(right), inverse @ right, rtol=_ROUNDING, atol=_ROUNDING)
    np.testing.assert_allclose(kernel.variances(), np.diag(inverse), rtol=_ROUNDING, atol=_ROUNDING)
    weights = rng.standard_normal((30, 5))
    noise = 1.7
    expected = -(noise * noise) * (inverse * inverse) @ weights
    for jvp_bytes in (10**9, 0):  # formed Sigma o Sigma, and the factor form
        posterior = _DensePosterior(kernel, noise, jvp_bytes, _new_profile())
        np.testing.assert_allclose(posterior.variance_jvp(weights), expected, rtol=_ROUNDING, atol=_ROUNDING)
        np.testing.assert_allclose(posterior.solve(right, 0.0), noise * inverse @ right, rtol=_ROUNDING, atol=_ROUNDING)


def test_the_cavity_precision_keeps_its_digits_where_the_data_barely_inform_a_column():
    rng = np.random.default_rng(9)
    design = _design(rng, 10, 6)
    design[:, 0] *= 1e-7  # a column whose data information q ~ 1e-14 of its site precision
    precision = rng.uniform(0.5, 2.0, 6)
    precision[0] = 1e3
    _variances, removed, cavity = _Kernel(_Design.dense(design), precision).cavity()
    # Reference, no cancellation: the cavity precision of column j is x_j' (I + X_-j T_-j^-1 X_-j')^-1 x_j.
    for column in range(6):
        others = np.delete(np.arange(6), column)
        kernel = np.eye(10) + design[:, others] @ np.diag(1.0 / precision[others]) @ design[:, others].T
        expected = float(design[:, column] @ np.linalg.solve(kernel, design[:, column]))
        np.testing.assert_allclose(cavity[column], expected, rtol=1e-9)
        np.testing.assert_allclose(removed[column], expected / (expected + precision[column]), rtol=1e-9)
    # The naive 1/z - t has no digits left for column 0.
    naive = 1.0 / _Kernel(_Design.dense(design), precision).variances()[0] - precision[0]
    assert abs(naive - cavity[0]) > 1e-6 * abs(cavity[0])


def test_a_column_spanned_by_negative_site_columns_has_a_small_negative_cavity():
    """x3 = x1 + x2 (a doubleton, the sum of two singleton columns) with slightly negative sites on x1 and x2: x3's
    cavity precision is negative and proportional to those sites, never positive however small they get. The refresh
    must therefore accept it whenever the tilted law is proper (1 + v_max P > 0), not ask P > 0."""
    rng = np.random.default_rng(12)
    design = _design(rng, 15, 6)
    design[:, 2] = design[:, 0] + design[:, 1]
    for scale in (1e-3, 1e-6, 1e-9):
        precision = rng.uniform(0.5, 2.0, 6)
        precision[[0, 1]] = -scale
        _variances, _removed, cavity = _Kernel(_Design.dense(design), precision).cavity()
        others = np.array([0, 1, 3, 4, 5])
        # Reference (T invertible here): x3' (I + X_-3 T_-3^-1 X_-3')^-1 x3.
        kernel = np.eye(15) + design[:, others] @ np.diag(1.0 / precision[others]) @ design[:, others].T
        expected = float(design[:, 2] @ np.linalg.solve(kernel, design[:, 2]))
        assert expected < 0.0 and cavity[2] < 0.0
        np.testing.assert_allclose(cavity[2], expected, rtol=1e-6, atol=10 * _ROUNDING * scale)
        assert abs(cavity[2]) <= 10.0 * scale


def test_the_carrier_design_gives_the_projected_design_quantities():
    """Stage 0's G (minor-allele codes over their SD) with the covariates projected out equals the projected
    standardized design in every kernel quantity, including negative sites."""
    rng = np.random.default_rng(21)
    samples, variants = 60, 90
    frequency = rng.uniform(0.005, 0.6, variants)
    dosage = rng.binomial(2, frequency, size=(samples, variants))
    dosage[:, 0] = 0
    dosage[3, 0] = 1  # a singleton
    codes = (dosage * 127).astype(np.uint8)
    statistics = dense_statistics(codes, np.column_stack([np.ones(samples), rng.standard_normal(samples)]), rng.standard_normal(samples))
    projected = _Design.dense(statistics.projected)
    count = statistics.design.variant_count
    precision = rng.uniform(0.5, 3.0, count)
    precision[:2] = -0.01
    carrier_kernel, projected_kernel = _Kernel(statistics.design, precision), _Kernel(projected, precision)
    right = rng.standard_normal((count, 3))
    np.testing.assert_allclose(carrier_kernel.solve(right), projected_kernel.solve(right), rtol=1e-9, atol=1e-11)
    for carrier_part, projected_part in zip(carrier_kernel.cavity(), projected_kernel.cavity()):
        np.testing.assert_allclose(carrier_part, projected_part, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(carrier_kernel.covariance(), projected_kernel.covariance(), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(statistics.design.column_squares(), np.einsum("ij,ij->j", statistics.projected, statistics.projected), rtol=1e-10)
    # X~'s covariate loading, against the definition.
    # Every member, in its oriented coordinates (its standardized column times its sign, its group's column).
    signed = codes[:, statistics.active_rows].astype(np.float64) - SIGNED_CODE_OFFSET
    standardized = statistics.signs * (signed - statistics.means) / statistics.scales
    np.testing.assert_allclose(statistics.loading, statistics.covariates.T @ standardized, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("negative", [0, 2])
def test_the_exact_linear_response_solves_the_curvature_fixed_point(negative):
    rng = np.random.default_rng(31 + negative)
    design = _design(rng, 10, 25)
    precision = rng.uniform(0.5, 2.0, 25)
    precision[:negative] = -0.05
    noise = 0.8
    sigma = noise * _dense_inverse(design, precision)
    left, right, diagonal, weight = (rng.standard_normal(25) for _ in range(4))
    weight = np.abs(weight)
    rhs = rng.standard_normal((25, 4))
    matrix = np.eye(25) - (np.eye(25) - weight[:, None] * (sigma * sigma)) @ (left[:, None] * sigma * right[None, :] + np.diag(diagonal))
    posterior = _DensePosterior(_Kernel(_Design.dense(design), precision), noise, 10**9, _new_profile())
    np.testing.assert_allclose(posterior.linear_response(left, right, diagonal, weight, rhs), np.linalg.solve(matrix, rhs), rtol=1e-8, atol=1e-10)
    # The lazy correction asks again for new directions at the same fixed point: the factor is reused.
    more = rng.standard_normal((25, 2))
    np.testing.assert_allclose(posterior.linear_response(left, right, diagonal, weight, more), np.linalg.solve(matrix, more), rtol=1e-8, atol=1e-10)
    assert posterior.profile["response_factorizations"] == 1 and posterior.profile["responses"] == 2
    # The factor is LAPACK's in place on the matrix's own buffer (Fortran order), not a copy.
    assert posterior._response_factor[0].flags.f_contiguous


def test_the_total_curvature_by_the_exact_response_equals_gmres():
    from sv_pgs.scale_mixture_ep import Cavity, GaussianPosterior, _total_curvature, derived_lattice, initial_hyperparameters, scale_mixture_prior

    rng = np.random.default_rng(41)
    samples, variants = 30, 40
    design = _design(rng, samples, variants)
    precision = rng.uniform(0.5, 2.0, variants)
    precision[0] = -0.02
    noise = 1.3
    posterior = _DensePosterior(_Kernel(_Design.dense(design), precision), noise, 10**9, _new_profile())
    single_precision = np.einsum("ij,ij->j", design, design) / noise
    single_shift = rng.standard_normal(variants) * 2.0
    nodes, floor, top = derived_lattice(single_precision, single_shift, np.zeros(variants), 1.0 / 128)
    prior = scale_mixture_prior(
        class_index=np.arange(variants) % 2, log_variance_offset=np.zeros(variants), annotation_design=np.zeros((variants, 0)),
        annotation_groups=(), nodes=nodes, floor=floor, top=top,
    )
    coefficients = initial_hyperparameters(prior).coefficients
    variances, _removed, cavity_precision = posterior.kernel.cavity()
    cavity = Cavity(precision=cavity_precision / noise, shift=rng.standard_normal(variants))
    exact = _total_curvature(prior, coefficients, cavity, posterior.gaussian_posterior(), 10**9, 1e-10)
    iterative = _total_curvature(prior, coefficients, cavity, GaussianPosterior(solve=posterior.solve, variance_jvp=posterior.variance_jvp), 10**9, 1e-10)
    np.testing.assert_allclose(exact, iterative, rtol=1e-6, atol=1e-7 * float(np.max(np.abs(iterative))))


def test_kernel_refuses_an_indefinite_precision():
    rng = np.random.default_rng(7)
    design = _design(rng, 5, 20)
    precision = np.ones(20)
    precision[0] = -50.0
    with pytest.raises(np.linalg.LinAlgError):
        _Kernel(_Design.dense(design), precision)


def test_draws_have_the_posterior_covariance():
    rng = np.random.default_rng(3)
    design = _design(rng, 6, 8)
    precision = rng.uniform(0.5, 2.0, 8)
    precision[0] = -0.1
    inverse = _dense_inverse(design, precision)
    draws = _Kernel(_Design.dense(design), precision).draws(np.random.default_rng(11), 200_000)
    empirical = draws @ draws.T / draws.shape[1]
    # Monte Carlo error of a covariance entry: sqrt((S_ii S_jj + S_ij^2) / K) <= sqrt(2 / K) max S_ii.
    bound = 5.0 * np.sqrt(2.0 / draws.shape[1]) * float(np.max(np.diag(inverse)))
    assert float(np.max(np.abs(empirical - inverse))) <= bound


def test_stage0_standardizes_and_merges_exact_ties():
    rng = np.random.default_rng(5)
    samples = 40
    dosage = rng.integers(0, 3, size=(samples, 6))
    dosage[:, 2] = dosage[:, 0]          # a copy
    dosage[:, 3] = 2 - dosage[:, 0]      # a negated copy
    dosage[:, 4] = 1                     # every sample heterozygous: no variance
    carrier = rng.integers(0, 2, size=samples)
    dosage[:, 5] = carrier               # 0/1 ...
    extra = 2 * carrier                  # ... and its 0/2 multiple: an exact tie of r = 1
    dosage = np.column_stack([dosage, extra])
    codes = (dosage * 127).astype(np.uint8)
    covariates = np.ones((samples, 1))
    target = rng.standard_normal(samples)
    statistics = dense_statistics(codes, covariates, target)
    np.testing.assert_array_equal(statistics.active_rows, [0, 1, 2, 3, 5, 6])
    signed = codes[:, statistics.active_rows].astype(np.float64) - SIGNED_CODE_OFFSET
    np.testing.assert_allclose(statistics.means, signed.mean(axis=0), rtol=1e-14)
    np.testing.assert_allclose(statistics.scales, signed.std(axis=0), rtol=1e-14)
    tie_map = statistics.tie_map
    # Active positions: 0 -> column 0, 1 -> 1, 2 -> 2 (copy of 0), 3 -> 3 (negated 0), 4 -> 5, 5 -> 6 (tie of 5).
    np.testing.assert_array_equal(tie_map.kept_indices, [0, 1, 4])
    np.testing.assert_array_equal(tie_map.original_to_reduced, [0, 1, 0, 0, 2, 2])
    signs = {int(group.representative_index): dict(zip(group.member_indices.tolist(), group.signs.tolist())) for group in tie_map.reduced_to_group}
    assert signs[0] == {0: 1.0, 2: 1.0, 3: -1.0}
    assert signs[4] == {4: 1.0, 5: 1.0}
    # (I - H_C) X~ with the intercept over every member (review-mathbugs T1), oriented: each member's column is its
    # group representative's centred standardized column, and its sign carries the negation.
    np.testing.assert_array_equal(statistics.signs, [1.0, 1.0, 1.0, -1.0, 1.0, 1.0])
    representatives = signed[:, [0, 1, 4]]
    standardized = (representatives - representatives.mean(axis=0)) / representatives.std(axis=0)
    np.testing.assert_allclose(statistics.projected, standardized[:, [0, 1, 0, 0, 2, 2]], atol=1e-12)
    assert statistics.design.variant_count == 6 and statistics.design.group_count == 3
    np.testing.assert_allclose(statistics.projected_target, target - target.mean(), atol=1e-12)


@pytest.mark.slow  # the whole outer loop on 120 columns: about a minute on one core
def test_the_small_n_fit_certifies_and_scores():
    """Machinery only (own simulation): the outer loop certifies, and the scoring model carries the fit."""
    rng = np.random.default_rng(17)
    samples, variants = 150, 120
    frequency = rng.uniform(0.05, 0.5, variants)
    dosage = rng.binomial(2, frequency, size=(samples, variants))
    effects = np.zeros(variants)
    effects[[10, 50, 90]] = [0.8, -0.6, 0.5]
    standardized = (dosage - dosage.mean(axis=0)) / dosage.std(axis=0)
    target = standardized @ effects + rng.standard_normal(samples)
    codes = (dosage * 127).astype(np.uint8)
    classes = np.full(variants, list(VariantClass).index(VariantClass.SNV), dtype=np.uint8)
    fit = fit_small_n(
        codes=codes, covariates=np.ones((samples, 1)), target=target, variant_class=classes, log_variance_offset=None,
        draw_count=64, working_bytes=2 * 10**9, seed=0,
    )
    assert fit.certificate.remaining_gain[0] <= 0.5 / 64
    assert fit.certificate.prediction_move[0] <= fit.certificate.prediction_tolerance[0]
    assert np.all(np.isfinite(fit.scoring.coefficients)) and fit.scoring.posterior_draws.shape == (variants, 64)
    assert np.all(np.isfinite(fit.scoring.posterior_draws)) and fit.noise_variance > 0.0
    np.testing.assert_array_equal(fit.scoring.store_rows, np.arange(variants))


def _engine_problem(seed, samples, variants, noise):
    """A small dense problem with the engine's lattice prior at its start hyperparameters: (design, target, prior,
    hyperparameters, tilted, largest prior variance)."""
    from sv_pgs.scale_mixture_ep import Cavity, derived_lattice, initial_hyperparameters, log_scale, scale_mixture_prior, tilted_cumulants, tilted_moments

    rng = np.random.default_rng(seed)
    design = _design(rng, samples, variants)
    effects = np.zeros(variants)
    effects[:2] = [1.2, -0.8]
    target = design @ effects + np.sqrt(noise) * rng.standard_normal(samples)
    nodes, floor, top = derived_lattice(np.einsum("ij,ij->j", design, design) / noise, design.T @ target / noise, np.zeros(variants), 1.0 / 128)
    prior = scale_mixture_prior(
        class_index=np.zeros(variants, dtype=np.int64), log_variance_offset=np.zeros(variants), annotation_design=np.zeros((variants, 0)),
        annotation_groups=(), nodes=nodes, floor=floor, top=top,
    )
    hyperparameters = initial_hyperparameters(prior)

    def tilted(cavity_precision, cavity_shift):
        cavity = Cavity(precision=cavity_precision, shift=cavity_shift)
        moments = tilted_moments(prior, hyperparameters, cavity, 10**8)
        third, fourth = tilted_cumulants(prior, hyperparameters, cavity, 10**8)
        return moments.log_normalizer, moments.mean, moments.variance, third, fourth

    largest = np.exp(log_scale(prior, hyperparameters.coefficients) + prior.log_variance_grid[-1])
    return design, target, prior, hyperparameters, tilted, largest


def test_the_double_loop_reaches_the_reference_double_loops_stationary_point(monkeypatch):
    """small_n's matrix-free double loop against ``tests/ep_eb_reference.double_loop_sites`` itself (dense Cholesky,
    exact 2p x 2p Newton), run on the engine's tilted moments: the same EP stationary point."""
    import tests.ep_eb_reference as reference
    from sv_pgs.scale_mixture_ep import moment_matched_prior_sites
    from sv_pgs.small_n import double_loop_sites

    noise = 0.6
    design, target, prior, hyperparameters, tilted, largest = _engine_problem(51, 14, 9, noise)

    def power_moments(_prior, _vector, cavity_precision, cavity_shift):
        log_normalizer, mean, variance, third, fourth = tilted(cavity_precision, cavity_shift)
        return {
            "log_normalizer": log_normalizer, "first": mean, "second": variance + mean**2, "third": third + 3.0 * mean * variance + mean**3,
            "fourth": fourth + 3.0 * variance**2 + 4.0 * mean * third + 6.0 * mean**2 * variance + mean**4,
        }

    monkeypatch.setattr(reference, "tilted_power_moments", power_moments)
    precision, shift = moment_matched_prior_sites(prior, hyperparameters)
    likelihood_precision, linear_term = design.T @ design / noise, design.T @ target / noise
    expected = reference.double_loop_sites(None, None, likelihood_precision, linear_term, reference.site_state(None, None, likelihood_precision, linear_term, precision, shift))
    profile = _new_profile()
    # A draw count whose tolerance resolves the stationary point as far as the reference's own stopping rule does.
    got_precision, got_shift = double_loop_sites(_Design.dense(design), noise, design.T @ target, precision, shift, tilted, largest, 2**30, 10**9, profile)
    np.testing.assert_allclose(got_precision, expected.site_precision, rtol=1e-6)
    np.testing.assert_allclose(got_shift, expected.site_shift, rtol=1e-6, atol=1e-8 * float(np.max(np.abs(expected.site_shift))))
    assert profile["double_loop_outer"] >= 1


def test_the_frozen_passes_fall_back_to_the_double_loop_instead_of_refusing(monkeypatch):
    """Where no damped pass keeps the precision positive definite (every trial refused here), EP falls back to the
    double loop and reaches its fixed point: the refresh after it matches every site's moments."""
    from sv_pgs.scale_mixture_ep import Cavity
    from sv_pgs.small_n import _DenseFixedPoints, small_n_prior, small_n_start

    rng = np.random.default_rng(61)
    samples, variants = 40, 30
    dosage = rng.binomial(2, rng.uniform(0.1, 0.5, variants), size=(samples, variants))
    target = (dosage[:, 0] - dosage[:, 0].mean()) + rng.standard_normal(samples)
    statistics = dense_statistics((dosage * 127).astype(np.uint8), np.ones((samples, 1)), target)
    prior = small_n_prior(statistics, np.zeros(variants, dtype=np.uint8), np.zeros(variants), 64)
    start, start_noise, _moment = small_n_start(statistics, prior)
    oracle = _DenseFixedPoints(statistics, prior, start, start_noise, 2**30, 10**9)
    variances, frozen = oracle._refresh(start)
    cavity = Cavity(precision=frozen, shift=oracle.mean / variances - oracle.site_shift)
    target_precision, target_shift = oracle._targets(start, cavity)
    iterate = oracle._iterate

    def refuse_trials(precision, shift):
        if not (np.array_equal(precision, oracle.site_precision) and np.array_equal(shift, oracle.site_shift)):
            raise np.linalg.LinAlgError("trial refused")
        iterate(precision, shift)

    monkeypatch.setattr(oracle, "_iterate", refuse_trials)
    oracle._frozen_passes(start, frozen, target_precision, target_shift)
    assert oracle.profile["double_loops"] == 1
    monkeypatch.setattr(oracle, "_iterate", iterate)
    variances, frozen = oracle._refresh(start)
    moments_mean = oracle._targets(start, Cavity(precision=frozen, shift=oracle.mean / variances - oracle.site_shift))
    np.testing.assert_allclose(moments_mean[0], oracle.site_precision, rtol=1e-6)
    np.testing.assert_allclose(moments_mean[1], oracle.site_shift, rtol=1e-6, atol=1e-8 * float(np.max(np.abs(oracle.site_shift))))


def test_the_kernel_log_determinant_is_the_precisions():
    rng = np.random.default_rng(71)
    design = _design(rng, 7, 12)
    precision = rng.uniform(0.5, 2.0, 12)
    precision[:2] = -0.05
    np.testing.assert_allclose(_Kernel(_Design.dense(design), precision).log_determinant(), np.linalg.slogdet(design.T @ design + np.diag(precision))[1], rtol=1e-12)


def test_a_trial_whose_prior_second_moment_overflows_is_refused_not_raised(monkeypatch):
    """At a far trial the moment-matched precision 1/E[beta^2] underflows to 0; with dependent columns the precision is
    then singular. The fallback refuses the trial (NoFixedPoint: the outer loop halves), never a LinAlgError."""
    from sv_pgs import small_n
    from sv_pgs.full_data_fit import NoFixedPoint
    from sv_pgs.small_n import _DenseFixedPoints, small_n_prior, small_n_start

    rng = np.random.default_rng(81)
    samples, variants = 12, 40
    dosage = rng.binomial(2, rng.uniform(0.2, 0.5, variants), size=(samples, variants))
    statistics = dense_statistics((dosage * 127).astype(np.uint8), np.ones((samples, 1)), rng.standard_normal(samples))
    prior = small_n_prior(statistics, np.zeros(variants, dtype=np.uint8), np.zeros(variants), 64)
    start, start_noise, _moment = small_n_start(statistics, prior)
    oracle = _DenseFixedPoints(statistics, prior, start, start_noise, 64, 10**9)
    count = statistics.design.variant_count
    oracle.site_precision = np.full(count, -1.0)  # current sites outside the domain
    monkeypatch.setattr(small_n, "moment_matched_prior_sites", lambda _prior, _hyperparameters: (np.zeros(count), np.zeros(count)))
    with pytest.raises(NoFixedPoint):
        oracle._double_loop(start)
    with pytest.raises(ValueError):
        small_n.double_loop_sites(statistics.design, start_noise, statistics.design.back(statistics.target), np.zeros(count), np.zeros(count),
                                  oracle._tilted(start), oracle._largest_variances(start), 64, 10**9, _new_profile())



def _tied_design(rng, samples, groups, members_of):
    """A dense projected design over ``groups`` group columns and the members' ``members_of`` indicator, the explicit
    member-level matrix (duplicated columns) and the member-aggregating ``_Design`` of the same model."""
    group_columns = _design(rng, samples, groups)
    members_of = np.asarray(members_of, dtype=np.int64)
    return group_columns[:, members_of], _Design(group_columns, np.zeros((samples, 0)), members=members_of)


@pytest.mark.parametrize("negative", [False, True])
def test_tie_members_keep_their_own_sites_exactly(negative):
    """review-mathbugs T1: every tie member is its own effect with its own site; duplicates are aggregated only in the
    kernel. Against the explicit member-level A' = X_m'X_m + diag t (duplicated columns): solve, marginal variances,
    cavities, covariance, the variance JVP and the exact linear response, with units of equal sites and a right-hand
    side that is not constant within them."""
    rng = np.random.default_rng(91)
    members_of = [0, 0, 0, 1, 2, 2, 3, 4, 4, 4, 5]
    explicit, tied = _tied_design(rng, 9, 6, members_of)
    count = len(members_of)
    precision = rng.uniform(0.5, 2.0, count)
    precision[[1, 2]] = precision[0]      # a unit: three members of group 0 with one site
    precision[[8, 9]] = precision[7]      # and two of group 4 with one site
    if negative:
        precision[3] = -0.05              # a singleton group's negative site
        precision[6] = -0.04
    inverse = _dense_inverse(explicit, precision)
    kernel = _Kernel(tied, precision)
    right = rng.standard_normal((count, 3))
    np.testing.assert_allclose(kernel.solve(right), inverse @ right, rtol=1e-9, atol=1e-10)
    variances, removed, cavity = kernel.cavity()
    np.testing.assert_allclose(variances, np.diag(inverse), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(removed, 1.0 - precision * np.diag(inverse), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(cavity, 1.0 / np.diag(inverse) - precision, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(kernel.covariance(), inverse, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(kernel.log_determinant(), np.linalg.slogdet(explicit.T @ explicit + np.diag(precision))[1], rtol=1e-12)
    assert kernel.units().size == count - 4  # two units of three and two members
    noise = 1.3
    sigma = noise * inverse
    weights = rng.standard_normal((count, 4))
    posterior = _DensePosterior(kernel, noise, 10**9, _new_profile())
    np.testing.assert_allclose(posterior.variance_jvp(weights), -(sigma * sigma) @ weights, rtol=1e-9, atol=1e-11)
    left, gain, diagonal, weight = (rng.standard_normal(count) for _ in range(4))
    for vector in (left, gain, diagonal, weight):
        vector[[1, 2]] = vector[0]
        vector[[8, 9]] = vector[7]
    weight = np.abs(weight)
    rhs = rng.standard_normal((count, 3))  # not constant within the units
    matrix = np.eye(count) - (np.eye(count) - weight[:, None] * (sigma * sigma)) @ (left[:, None] * sigma * gain[None, :] + np.diag(diagonal))
    np.testing.assert_allclose(posterior.linear_response(left, gain, diagonal, weight, rhs), np.linalg.solve(matrix, rhs), rtol=1e-8, atol=1e-10)
    # Response coefficients that differ within a unit split it: still exact.
    left[2] += 0.3
    matrix = np.eye(count) - (np.eye(count) - weight[:, None] * (sigma * sigma)) @ (left[:, None] * sigma * gain[None, :] + np.diag(diagonal))
    np.testing.assert_allclose(posterior.linear_response(left, gain, diagonal, weight, rhs), np.linalg.solve(matrix, rhs), rtol=1e-8, atol=1e-10)


def test_two_non_positive_sites_in_one_tie_group_are_not_positive_definite():
    rng = np.random.default_rng(92)
    _explicit, tied = _tied_design(rng, 8, 4, [0, 0, 1, 2, 3])
    precision = np.array([-0.01, -0.02, 1.0, 1.0, 1.0])
    with pytest.raises(np.linalg.LinAlgError):
        _Kernel(tied, precision)


def test_a_mixed_class_tie_group_reaches_the_reference_ep_fixed_point(monkeypatch):
    """An SV tied with two SNVs (review-mathbugs T1 on real genes 3 and 4): each member keeps its own class prior and
    site. small_n's double loop over the members against ``tests/ep_eb_reference.double_loop_sites`` (dense, exact
    Newton) on the explicit member-level design with the duplicated columns, both on the engine's tilted moments:
    the same EP stationary point, member by member."""
    import tests.ep_eb_reference as reference
    from sv_pgs.scale_mixture_ep import Cavity, derived_lattice, initial_hyperparameters, log_scale, moment_matched_prior_sites, scale_mixture_prior
    from sv_pgs.scale_mixture_ep import tilted_cumulants, tilted_moments
    from sv_pgs.small_n import double_loop_sites

    rng = np.random.default_rng(93)
    noise = 0.7
    members_of = [0, 0, 0, 1, 2, 3, 4]
    explicit, tied = _tied_design(rng, 16, 5, members_of)
    target = explicit[:, 0] * 1.1 + explicit[:, 4] * -0.7 + np.sqrt(noise) * rng.standard_normal(16)
    count = len(members_of)
    class_index = np.array([1, 0, 0, 0, 0, 1, 0])  # member 0 (the SV) tied with members 1 and 2 (SNVs)
    nodes, floor, top = derived_lattice(np.einsum("ij,ij->j", explicit, explicit) / noise, explicit.T @ target / noise, np.zeros(count), 1.0 / 128)
    prior = scale_mixture_prior(
        class_index=class_index, log_variance_offset=np.zeros(count), annotation_design=np.zeros((count, 0)), annotation_groups=(),
        nodes=nodes, floor=floor, top=top,
    )
    hyperparameters = initial_hyperparameters(prior)
    # Give the SV class a different density, so the tied members' priors differ.
    coefficients = hyperparameters.coefficients.copy()
    coefficients[prior.pooled_size : 2 * prior.pooled_size] = 0.3 * rng.standard_normal(prior.pooled_size)
    hyperparameters = type(hyperparameters)(coefficients=coefficients, log_smoothing=hyperparameters.log_smoothing)

    def tilted(cavity_precision, cavity_shift):
        cavity = Cavity(precision=cavity_precision, shift=cavity_shift)
        moments = tilted_moments(prior, hyperparameters, cavity, 10**8)
        third, fourth = tilted_cumulants(prior, hyperparameters, cavity, 10**8)
        return moments.log_normalizer, moments.mean, moments.variance, third, fourth

    def power_moments(_prior, _vector, cavity_precision, cavity_shift):
        log_normalizer, mean, variance, third, fourth = tilted(cavity_precision, cavity_shift)
        return {
            "log_normalizer": log_normalizer, "first": mean, "second": variance + mean**2, "third": third + 3.0 * mean * variance + mean**3,
            "fourth": fourth + 3.0 * variance**2 + 4.0 * mean * third + 6.0 * mean**2 * variance + mean**4,
        }

    monkeypatch.setattr(reference, "tilted_power_moments", power_moments)
    largest = np.exp(log_scale(prior, hyperparameters.coefficients) + prior.log_variance_grid[-1])
    precision, shift = moment_matched_prior_sites(prior, hyperparameters)
    likelihood_precision, linear_term = explicit.T @ explicit / noise, explicit.T @ target / noise
    expected = reference.double_loop_sites(None, None, likelihood_precision, linear_term, reference.site_state(None, None, likelihood_precision, linear_term, precision, shift))
    got_precision, got_shift = double_loop_sites(tied, noise, tied.back(target), precision, shift, tilted, largest, 2**30, 10**9, _new_profile())
    np.testing.assert_allclose(got_precision, expected.site_precision, rtol=1e-6)
    np.testing.assert_allclose(got_shift, expected.site_shift, rtol=1e-6, atol=1e-8 * float(np.max(np.abs(expected.site_shift))))
    # The tied members' sites differ: each keeps its own prior (no merged column, no b / M split).
    assert not np.isclose(got_precision[0], got_precision[1])


def test_the_prior_and_scoring_are_per_member():
    """An insertion tied with SNVs keeps its INSERTION class in the prior, and the scoring model has one coefficient
    per active member, no representative-only rows."""
    from sv_pgs.small_n import small_n_prior

    rng = np.random.default_rng(94)
    samples = 60
    dosage = rng.binomial(2, rng.uniform(0.1, 0.4, 8), size=(samples, 8))
    dosage[:, 5] = dosage[:, 2]   # an insertion (column 5) tied with an SNV (column 2)
    dosage[:, 6] = dosage[:, 2]   # and another SNV
    statistics = dense_statistics((dosage * 127).astype(np.uint8), np.ones((samples, 1)), rng.standard_normal(samples))
    classes = np.full(8, list(VariantClass).index(VariantClass.SNV), dtype=np.uint8)
    classes[5] = list(VariantClass).index(VariantClass.INSERTION)
    prior = small_n_prior(statistics, classes, np.zeros(8), 64)
    assert prior.variant_count == 8 and statistics.design.group_count == 6
    member_classes = np.unique(classes, return_inverse=True)[1]
    np.testing.assert_array_equal(prior.class_index, member_classes)
    assert prior.class_index[5] != prior.class_index[2]
