"""The dual Stage 2 E-step against dense linear algebra.

Every tolerance is a rounding bound: a quantity computed through a system of condition number
kappa in float64 carries relative error <= dimension * kappa * eps.
"""

from __future__ import annotations

import numpy as np

from sv_pgs import dual_solve

EPS = np.finfo(np.float64).eps
MODEL_COUNT = 4


def _problem(seed: int, sample_count: int = 80, variant_count: int = 180, block: int = 30):
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((sample_count, variant_count))
    for start in range(0, variant_count, block):
        stop = min(start + block, variant_count)
        raw[:, start:stop] = np.cumsum(raw[:, start:stop], axis=1) / np.sqrt(np.arange(1, stop - start + 1))
    genotypes = (raw - raw.mean(0)) / raw.std(0)
    bounds = [(start, min(start + block, variant_count)) for start in range(0, variant_count, block)]
    covariates = np.column_stack([np.ones(sample_count), rng.standard_normal((sample_count, 2))])
    masks = np.ones((sample_count, MODEL_COUNT))
    masks[rng.permutation(sample_count)[: sample_count // 4], 1] = 0.0
    masks[rng.permutation(sample_count)[: sample_count // 4], 2] = 0.0
    noise = np.array([0.7, 0.9, 1.3, 1.0])
    weights = masks / noise[None, :]
    probability = 1.0 / (1.0 + np.exp(-rng.normal(-1.0, 1.0, sample_count)))
    weights[:, 3] = masks[:, 3] * probability * (1.0 - probability)
    variances = np.exp(rng.normal(-6.0, 2.0, (variant_count, MODEL_COUNT)))
    variances *= 0.4 / variances.sum(axis=0, keepdims=True)
    variances[rng.integers(0, variant_count, 3), 0] = 0.05
    prior_mean = rng.normal(0.0, 1.0, (variant_count, MODEL_COUNT)) * np.sqrt(variances)
    response = rng.standard_normal((sample_count, MODEL_COUNT))
    return genotypes, bounds, covariates, weights, variances, prior_mean, response


def _dense(genotypes, covariates, weights, variances, model):
    root = np.sqrt(weights[:, model])
    weighted_covariates = root[:, None] * covariates
    projector = np.eye(genotypes.shape[0]) - weighted_covariates @ np.linalg.pinv(weighted_covariates)
    design = projector @ (root[:, None] * genotypes)
    precision = design.T @ design + np.diag(1.0 / variances[:, model])
    operator = np.eye(genotypes.shape[0]) + design @ (variances[:, model][:, None] * design.T)
    return projector, design, precision, operator


def _setup(seed: int):
    genotypes, bounds, covariates, weights, variances, prior_mean, response = _problem(seed)
    source = dual_solve.DenseDualSource(genotypes, bounds)
    models = dual_solve.DualModels(weights, variances, covariates)
    return genotypes, source, models, covariates, weights, variances, prior_mean, response


def _solve_bound(right):
    """A bound float64 attains for the well-conditioned test systems: half its digits."""
    return np.linalg.norm(right, axis=0) * np.sqrt(EPS)


def _exact_solve(source, models, right, count, deflation=None):
    bound = _solve_bound(right)
    return dual_solve.certified_block_cg(source, models, right, np.zeros_like(right), np.arange(right.shape[1]), bound, count, deflation=deflation)


def test_the_dual_mean_is_the_posterior_mean_of_every_model() -> None:
    genotypes, source, models, covariates, weights, variances, prior_mean, response = _setup(1)
    right = dual_solve.mean_right_hand_side(models, response, genotypes @ prior_mean)
    count = dual_solve.PassCount()
    result = _exact_solve(source, models, right, count)
    mean = dual_solve.mean_from_dual(source, models, prior_mean, result.solution, count)
    for model in range(MODEL_COUNT):
        projector, design, precision, _operator = _dense(genotypes, covariates, weights, variances, model)
        root = np.sqrt(weights[:, model])
        exact = np.linalg.solve(precision, design.T @ (projector @ (root * response[:, model])) + prior_mean[:, model] / variances[:, model])
        error = mean[:, model] - exact
        assert np.sqrt(float(error @ precision @ error)) <= float(result.residual_norm[model]) * (1.0 + np.linalg.cond(precision) * EPS * genotypes.shape[1])


def test_the_energy_error_equals_the_dual_residual_identity() -> None:
    genotypes, _source, _models, covariates, weights, variances, prior_mean, response = _setup(2)
    rng = np.random.default_rng(7)
    for model in range(MODEL_COUNT):
        projector, design, precision, operator = _dense(genotypes, covariates, weights, variances, model)
        root = np.sqrt(weights[:, model])
        right = projector @ (root * (response[:, model] - genotypes @ prior_mean[:, model]))
        exact_dual = np.linalg.solve(operator, right)
        iterate = exact_dual + rng.standard_normal(exact_dual.shape) * np.linalg.norm(exact_dual) / np.sqrt(exact_dual.size)
        residual = right - operator @ iterate
        mean_error = variances[:, model] * (design.T @ (iterate - exact_dual))
        energy = float(mean_error @ precision @ mean_error)
        identity = float(residual @ residual - residual @ np.linalg.solve(operator, residual))
        assert abs(energy - identity) <= np.linalg.cond(precision) * genotypes.shape[1] * EPS * max(energy, identity)
        assert energy <= float(residual @ residual) * (1.0 + np.linalg.cond(operator) * genotypes.shape[0] * EPS)


def test_held_out_prediction_error_is_bounded_by_the_dual_residual() -> None:
    genotypes, _source, _models, covariates, weights, variances, prior_mean, response = _setup(10)
    rng = np.random.default_rng(13)
    for model in (1, 2):
        held_out = weights[:, model] == 0
        projector, design, _precision, operator = _dense(genotypes, covariates, weights, variances, model)
        root = np.sqrt(weights[:, model])
        right = projector @ (root * (response[:, model] - genotypes @ prior_mean[:, model]))
        exact_dual = np.linalg.solve(operator, right)
        kernel = genotypes[held_out] @ (variances[:, model][:, None] * genotypes[held_out].T)
        top = float(np.linalg.eigvalsh(kernel)[-1])
        iterate = exact_dual + rng.standard_normal(exact_dual.shape) * np.linalg.norm(exact_dual) / np.sqrt(exact_dual.size)
        residual = right - operator @ iterate
        held_out_error = genotypes[held_out] @ (variances[:, model] * (design.T @ (iterate - exact_dual)))
        assert float(held_out_error @ held_out_error) <= 0.25 * top * float(residual @ residual) * (1.0 + np.linalg.cond(operator) * genotypes.shape[0] * EPS)


def test_a_fold_mask_is_the_posterior_of_its_training_rows() -> None:
    genotypes, source, models, covariates, weights, variances, prior_mean, response = _setup(3)
    right = dual_solve.mean_right_hand_side(models, response, genotypes @ prior_mean)
    count = dual_solve.PassCount()
    result = _exact_solve(source, models, right, count)
    mean = dual_solve.mean_from_dual(source, models, prior_mean, result.solution, count)
    rows = weights[:, 1] > 0
    projector = np.eye(rows.sum()) - covariates[rows] @ np.linalg.pinv(covariates[rows])
    root = np.sqrt(weights[rows, 1])
    design = projector @ (root[:, None] * genotypes[rows])
    precision = design.T @ design + np.diag(1.0 / variances[:, 1])
    exact = np.linalg.solve(precision, design.T @ (projector @ (root * response[rows, 1])) + prior_mean[:, 1] / variances[:, 1])
    error = mean[:, 1] - exact
    assert np.sqrt(float(error @ precision @ error)) <= float(result.residual_norm[1]) * (1.0 + np.linalg.cond(precision) * EPS * genotypes.shape[1])


def test_matheron_draws_have_the_posterior_covariance_exactly() -> None:
    genotypes, _source, _models, covariates, weights, variances, _prior_mean, _response = _setup(4)
    for model in range(MODEL_COUNT):
        _projector, design, precision, operator = _dense(genotypes, covariates, weights, variances, model)
        root_variance = np.sqrt(variances[:, model])
        inverse_operator = np.linalg.inv(operator)
        prior_part = np.diag(root_variance) - variances[:, model][:, None] * (design.T @ inverse_operator @ design) * root_variance[None, :]
        sample_part = variances[:, model][:, None] * (design.T @ inverse_operator)
        covariance = prior_part @ prior_part.T + sample_part @ sample_part.T
        target = np.linalg.inv(precision)
        tolerance = np.linalg.cond(precision) * np.linalg.cond(operator) * genotypes.shape[1] * EPS * np.linalg.norm(target)
        assert np.linalg.norm(covariance - target) <= tolerance


def test_the_fused_final_pass_gives_weights_scores_certificates_and_design_products() -> None:
    genotypes, source, models, covariates, weights, variances, prior_mean, response = _setup(5)
    rng = np.random.default_rng(11)
    right = dual_solve.mean_right_hand_side(models, response, genotypes @ prior_mean)
    draw_models = np.repeat(np.arange(MODEL_COUNT), 3)
    prior_noise = rng.standard_normal((genotypes.shape[1], draw_models.size))
    sample_noise = rng.standard_normal((genotypes.shape[0], draw_models.size))
    count = dual_solve.PassCount()
    draw_right = dual_solve.draw_right_hand_side(source, models, draw_models, prior_noise, sample_noise, count)
    all_right = np.column_stack([right, draw_right])
    column_models = np.concatenate([np.arange(MODEL_COUNT), draw_models])
    bound = _solve_bound(all_right)
    result = dual_solve.certified_block_cg(source, models, all_right, np.zeros_like(all_right), column_models, bound, count)
    score_rows = np.flatnonzero(weights[:, 1] == 0)
    draw_columns = np.arange(MODEL_COUNT, MODEL_COUNT + draw_models.size)
    fused_weights, scores, residual_norm, design_products = dual_solve.fused_final_pass(
        source, models, prior_mean, result.solution, column_models, all_right, draw_columns, prior_noise, score_rows, count,
        design_columns=np.arange(MODEL_COUNT),
    )
    mean = dual_solve.mean_from_dual(source, models, prior_mean, result.solution[:, :MODEL_COUNT], count)
    np.testing.assert_allclose(fused_weights[:, :MODEL_COUNT], mean, rtol=0.0, atol=genotypes.shape[0] * EPS * np.abs(mean).max())
    for column, model in zip(draw_columns, draw_models):
        expected = mean[:, model] + np.sqrt(variances[:, model]) * prior_noise[:, column - MODEL_COUNT] + variances[:, model] * _design_times(genotypes, covariates, weights, model, result.solution[:, column])
        np.testing.assert_allclose(fused_weights[:, column], expected, rtol=0.0, atol=genotypes.shape[0] * EPS * np.abs(expected).max())
    reference_scores = genotypes[score_rows] @ fused_weights
    np.testing.assert_allclose(scores, reference_scores, rtol=0.0, atol=genotypes.shape[1] * EPS * np.abs(reference_scores).max())
    assert np.all(residual_norm <= bound * (1.0 + genotypes.shape[0] * EPS))
    for model in range(MODEL_COUNT):
        expected = _design_times(genotypes, covariates, weights, model, result.solution[:, model])
        np.testing.assert_allclose(design_products[:, model], expected, rtol=0.0, atol=genotypes.shape[0] * EPS * np.abs(expected).max())


def _design_times(genotypes, covariates, weights, model, dual):
    _projector, design, _precision, _operator = _dense(genotypes, covariates, weights, np.ones((genotypes.shape[1], MODEL_COUNT)), model)
    return design.T @ dual


def test_relaxed_operand_error_keeps_the_exact_certificate() -> None:
    genotypes, source, models, covariates, weights, variances, prior_mean, response = _setup(6)
    right = dual_solve.mean_right_hand_side(models, response, genotypes @ prior_mean)
    bound = _solve_bound(right)
    count = dual_solve.PassCount()
    deflation, _resolved = dual_solve.spike_deflation(source, models, count)
    result = dual_solve.certified_block_cg(source, models, right, np.zeros_like(right), np.arange(MODEL_COUNT), bound, count, deflation=deflation)
    assert np.all(result.residual_norm <= bound)
    assert result.relative_errors[0] == 0.0
    assert any(relative_error > 0.0 for relative_error in result.relative_errors[1:])
    undeflated = dual_solve.certified_block_cg(source, models, right, np.zeros_like(right), np.arange(MODEL_COUNT), bound, dual_solve.PassCount())
    assert all(relative_error == 0.0 for relative_error in undeflated.relative_errors)
    operators = [_dense(genotypes, covariates, weights, variances, model)[3] for model in range(MODEL_COUNT)]
    exact = right - np.column_stack([operators[model] @ result.solution[:, model] for model in range(MODEL_COUNT)])
    # Two float64 evaluations of b - S z differ by the rounding of S z: n eps ||S|| ||z|| per column.
    rounding = np.array([genotypes.shape[0] * EPS * np.linalg.norm(operators[model], 2) * np.linalg.norm(result.solution[:, model]) for model in range(MODEL_COUNT)])
    assert np.all(np.abs(np.linalg.norm(exact, axis=0) - result.residual_norm) <= rounding)


def test_rounded_operand_meets_its_normwise_bound_and_keeps_zeros() -> None:
    rng = np.random.default_rng(21)
    values = rng.standard_normal((500, 6)) * np.exp(rng.normal(0.0, 3.0, 6))[None, :]
    values[rng.random(values.shape) < 0.3] = 0.0
    for relative_error in (1e-1, 1e-3, 1e-6):
        rounded = dual_solve.rounded_operand(values, relative_error, np)
        assert np.all(np.linalg.norm(rounded - values, axis=0) <= relative_error * np.linalg.norm(values, axis=0))
        assert np.all(rounded[values == 0.0] == 0.0)
    np.testing.assert_array_equal(dual_solve.rounded_operand(values, 0.0, np), values)


def test_spike_deflation_keeps_the_certificate_and_removes_the_spikes_from_cg() -> None:
    genotypes, source, models, covariates, weights, variances, prior_mean, response = _setup(9)
    right = dual_solve.mean_right_hand_side(models, response, genotypes @ prior_mean)
    plain = _exact_solve(source, models, right, dual_solve.PassCount())
    count = dual_solve.PassCount()
    deflation, resolved = dual_solve.spike_deflation(source, models, count)
    deflated = _exact_solve(source, models, right, count, deflation)
    for model in range(MODEL_COUNT):
        _projector, design, _precision, operator = _dense(genotypes, covariates, weights, variances, model)
        spikes = variances[:, model] * np.sum(design * design, axis=0)
        chosen = np.zeros(spikes.shape[0], dtype=bool)
        while True:
            updated = spikes > 1.0 + np.sum(spikes[~chosen]) / genotypes.shape[0]
            if np.array_equal(updated, chosen):
                break
            chosen = updated
        assert resolved[model] == int(chosen.sum())
        error = deflated.solution[:, model] - np.linalg.solve(operator, right[:, model])
        assert float(error @ operator @ error) <= float(deflated.residual_norm[model]) ** 2 * (1.0 + np.linalg.cond(operator) * genotypes.shape[0] * EPS)
    assert deflated.iterations <= plain.iterations


def test_negative_sites_leave_exactly_that_many_negative_dual_eigenvalues() -> None:
    genotypes, _source, _models, covariates, weights, variances, _prior_mean, _response = _setup(8)
    _projector, design, _precision, _operator = _dense(genotypes, covariates, weights, variances, 0)
    site_precision = 1.0 / variances[:, 0]
    negative = np.argsort(variances[:, 0])[-2:]
    data = design.T @ design
    # Negative precisions that keep A positive definite: with A0 the precision without those sites
    # and B = (A0^-1)_JJ, A0 + diag(pi_J) stays PD iff 1 + pi lambda_max(B) > 0.
    site_precision[negative] = 0.0
    block = np.linalg.inv(data + np.diag(site_precision))[np.ix_(negative, negative)]
    site_precision[negative] = -0.5 / np.linalg.eigvalsh(block)[-1]
    assert np.linalg.eigvalsh(data + np.diag(site_precision))[0] > 0.0
    kernel = np.eye(genotypes.shape[0]) + design @ np.diag(1.0 / site_precision) @ design.T
    assert int(np.sum(np.linalg.eigvalsh(kernel) < 0.0)) == negative.size


def test_the_refresh_pass_applies_the_new_sites_in_the_same_read() -> None:
    genotypes, source, models, covariates, weights, variances, _prior_mean, _response = _setup(12)
    rng = np.random.default_rng(17)
    duals = rng.standard_normal((genotypes.shape[0], MODEL_COUNT))
    new_variances = variances * np.exp(rng.normal(0.0, 0.2, variances.shape))
    new_means = rng.standard_normal(variances.shape) * np.sqrt(new_variances)
    seen = {}

    def block_update(start, stop, products):
        seen[start] = products
        return new_variances[start:stop], new_means[start:stop]

    count = dual_solve.PassCount()
    applied, prior_image = dual_solve.refresh_pass(source, models, duals, np.arange(MODEL_COUNT), block_update, count)
    assert count.passes == 1
    for model in range(MODEL_COUNT):
        _projector, design, _precision, operator = _dense(genotypes, covariates, weights, new_variances, model)
        expected = operator @ duals[:, model]
        np.testing.assert_allclose(applied[:, model], expected, rtol=0.0, atol=np.linalg.cond(operator) * genotypes.shape[0] * EPS * np.abs(expected).max())
        lagged = np.concatenate([seen[start][:, model] for start, _stop in source.block_bounds])
        np.testing.assert_allclose(lagged, design.T @ duals[:, model], rtol=0.0, atol=genotypes.shape[0] * EPS * np.abs(design.T @ duals[:, model]).max())
    expected_prior = genotypes @ new_means
    np.testing.assert_allclose(prior_image, expected_prior, rtol=0.0, atol=genotypes.shape[1] * EPS * np.abs(expected_prior).max())
    np.testing.assert_array_equal(models.variances, new_variances)


def test_a_bound_below_float64_resolution_is_refused() -> None:
    genotypes, source, models, _covariates, _weights, _variances, prior_mean, response = _setup(14)
    right = dual_solve.mean_right_hand_side(models, response, genotypes @ prior_mean)
    count = dual_solve.PassCount()
    first = _exact_solve(source, models, right, count)
    try:
        dual_solve.certified_block_cg(source, models, right, first.solution, np.arange(MODEL_COUNT), np.zeros(MODEL_COUNT), count,
                                      operator_scale=first.operator_scale)
    except ValueError as error:
        assert "below the accuracy" in str(error)
    else:
        raise AssertionError("a zero bound must be refused")


def test_a_column_budget_keeps_the_largest_spikes() -> None:
    genotypes, source, models, covariates, weights, variances, prior_mean, response = _setup(16)
    _deflation, unlimited = dual_solve.spike_deflation(source, models, dual_solve.PassCount())
    budget = sum(unlimited.values()) // 2
    deflation, limited = dual_solve.spike_deflation(source, models, dual_solve.PassCount(), column_budget=budget)
    assert sum(limited.values()) == budget
    spikes = np.column_stack([
        variances[:, model] * np.sum(_dense(genotypes, covariates, weights, variances, model)[1] ** 2, axis=0) for model in range(MODEL_COUNT)
    ])
    kept = np.concatenate([spikes[deflation.indices[model], model] for model in range(MODEL_COUNT)])
    right = dual_solve.mean_right_hand_side(models, response, genotypes @ prior_mean)
    result = _exact_solve(source, models, right, dual_solve.PassCount(), deflation)
    assert np.all(result.residual_norm <= _solve_bound(right))
    assert kept.min() >= np.sort(spikes[spikes > 1.0].ravel())[::-1][budget - 1] * (1.0 - genotypes.shape[0] * EPS)


def test_the_split_eliminates_negative_sites_exactly_in_mean_and_draws() -> None:
    for descending in (False, True):
        _check_split(descending)


def _check_split(descending: bool) -> None:
    """The split with its resolved sites listed ascending or descending: column k must pair with site k."""
    genotypes, bounds, covariates, weights, variances, prior_mean, response = _problem(15)
    rng = np.random.default_rng(23)
    model = 0
    projector, design, _precision, _operator = _dense(genotypes, covariates, weights, variances, model)
    data = design.T @ design
    site_precision = 1.0 / variances[:, model]
    # The two largest site variances, in a fixed order (argsort's order among ties varies by platform).
    negative = np.sort(np.argsort(variances[:, model], kind="stable")[-2:])
    if descending:
        negative = negative[::-1]
    site_precision[negative] = 0.0
    block = np.linalg.inv(data + np.diag(site_precision))[np.ix_(negative, negative)]
    site_precision[negative] = -0.5 / np.linalg.eigvalsh(block)[-1]
    precision = data + np.diag(site_precision)
    assert np.linalg.eigvalsh(precision)[0] > 0.0
    bulk_mean = prior_mean[:, model].copy()
    bulk_mean[negative] = 0.0
    shift = site_precision * bulk_mean
    shift[negative] = rng.standard_normal(negative.size)
    root = np.sqrt(weights[:, model])
    exact_mean = np.linalg.solve(precision, design.T @ (projector @ (root * response[:, model])) + shift)
    bulk_variance = variances[:, [model]].copy()
    bulk_variance[negative] = 0.0
    source = dual_solve.DenseDualSource(genotypes, bounds)
    bulk = dual_solve.DualModels(weights[:, [model]], bulk_variance, covariates)
    resolved = dual_solve.ResolvedSites({0: negative}, {0: site_precision[negative]}, {0: shift[negative]})
    designs = dual_solve.resolved_design(source, bulk, resolved)
    right = dual_solve.mean_right_hand_side(bulk, response[:, [model]], genotypes @ bulk_mean[:, None])
    count = dual_solve.PassCount()
    split = dual_solve.split_mean(source, bulk, resolved, designs, right, float(np.sqrt(EPS)), count)
    mean = dual_solve.mean_from_dual(source, bulk, bulk_mean[:, None], split.mean_duals, count)[:, 0]
    mean[negative] = split.resolved_mean[0]
    error = mean - exact_mean
    scale = float(np.sqrt(exact_mean @ precision @ exact_mean))
    assert np.sqrt(float(error @ precision @ error)) <= np.linalg.cond(precision) * np.sqrt(EPS) * scale
    # The draw map, extracted with unit noise in (eps_L, e1, e2), must have covariance A^-1.
    resolved_count, variant_count, sample_count = negative.size, genotypes.shape[1], genotypes.shape[0]
    draw_count = resolved_count + variant_count + sample_count
    resolved_noise = np.zeros((resolved_count, draw_count))
    resolved_noise[:, :resolved_count] = np.eye(resolved_count)
    prior_noise = np.zeros((variant_count, draw_count))
    prior_noise[:, resolved_count : resolved_count + variant_count] = np.eye(variant_count)
    sample_noise = np.zeros((sample_count, draw_count))
    sample_noise[:, resolved_count + variant_count :] = np.eye(sample_count)
    draw_models = np.zeros(draw_count, dtype=np.int64)
    draw_right = dual_solve.draw_right_hand_side(source, bulk, draw_models, prior_noise, sample_noise, count)
    perturbation = dual_solve.certified_block_cg(source, bulk, draw_right, np.zeros_like(draw_right), draw_models, _solve_bound(draw_right), count)
    draw_duals, resolved_draws = dual_solve.split_draw_duals(split, draw_models, perturbation.solution, {0: resolved_noise}, np)
    duals = np.column_stack([split.mean_duals, draw_duals])
    column_models = np.concatenate([[0], draw_models])
    fused_weights, _scores, _norms, _design = dual_solve.fused_final_pass(
        source, bulk, bulk_mean[:, None], duals, column_models, np.zeros_like(duals), np.arange(1, draw_count + 1), prior_noise, np.zeros(0, dtype=np.int64), count,
    )
    fused_weights[negative, 0] = split.resolved_mean[0]
    fused_weights[negative, 1:] = resolved_draws[0]
    draw_map = fused_weights[:, 1:] - fused_weights[:, [0]]
    target = np.linalg.inv(precision)
    assert np.linalg.norm(draw_map @ draw_map.T - target) <= np.linalg.cond(precision) * np.sqrt(EPS) * np.linalg.norm(target)


class _CodeTileSource:
    """A GenotypeBlockSource over code_products.CodeBlockTile tiles, as the store source streams them."""

    def __init__(self, codes, means, scales, bounds, array_module, workspace_bytes):
        from sv_pgs.code_products import CodeBlockTile

        self.array_module = array_module
        self.sample_count = int(codes.shape[1])
        self.block_variant_indices = [np.arange(start, stop) for start, stop in bounds]
        self._tiles = [
            CodeBlockTile(array_module.asarray(codes[start:stop]), means[start:stop], scales[start:stop], array_module, workspace_bytes)
            for start, stop in bounds
        ]

    def iter_tiles(self):
        yield from enumerate(self._tiles)


def _coded_problem(seed: int):
    genotypes, bounds, covariates, weights, variances, prior_mean, response = _problem(seed)
    rng = np.random.default_rng(seed)
    codes = np.clip(np.rint(rng.normal(0.0, 40.0, (genotypes.shape[1], genotypes.shape[0]))), -127, 127).astype(np.int8)
    means = codes.astype(np.float64).mean(axis=1)
    scales = codes.astype(np.float64).std(axis=1)
    standardized = (codes.astype(np.float64).T - means[None, :]) / scales[None, :]
    return standardized, codes, means, scales, bounds, covariates, weights, variances, prior_mean, response


def test_code_block_tiles_stream_through_the_same_certified_solve() -> None:
    standardized, codes, means, scales, bounds, covariates, weights, variances, prior_mean, response = _coded_problem(41)
    dense = dual_solve.DenseDualSource(standardized, bounds)
    streamed = dual_solve.StreamedDualSource(_CodeTileSource(codes, means, scales, bounds, np, 1 << 26))
    solutions = []
    for source in (dense, streamed):
        models = dual_solve.DualModels(weights, variances, covariates)
        right = dual_solve.mean_right_hand_side(models, response, standardized @ prior_mean)
        count = dual_solve.PassCount()
        deflation, _resolved = dual_solve.spike_deflation(source, models, count)
        bound = _solve_bound(right)
        result = dual_solve.certified_block_cg(source, models, right, np.zeros_like(right), np.arange(MODEL_COUNT), bound, count, deflation=deflation)
        assert np.all(result.residual_norm <= bound)
        solutions.append((result.solution, bound))
    (dense_solution, bound), (streamed_solution, _bound) = solutions
    assert np.all(np.linalg.norm(dense_solution - streamed_solution, axis=0) <= 2.0 * bound * (1.0 + standardized.shape[0] * EPS))
