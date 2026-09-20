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


def test_a_bound_below_float64_resolution_is_met_at_its_floor() -> None:
    genotypes, source, models, _covariates, _weights, _variances, prior_mean, response = _setup(14)
    right = dual_solve.mean_right_hand_side(models, response, genotypes @ prior_mean)
    count = dual_solve.PassCount()
    first = _exact_solve(source, models, right, count)
    result = dual_solve.certified_block_cg(source, models, right, first.solution, np.arange(MODEL_COUNT), np.zeros(MODEL_COUNT), count,
                                           operator_scale=first.operator_scale)
    # A zero bound becomes float64's floor, the rounding of one exact product S z, and the exact residual meets it.
    floor = (source.sample_count + source.variant_count) * EPS * result.operator_scale * np.linalg.norm(result.solution, axis=0)
    np.testing.assert_allclose(result.residual_bound, floor, rtol=4 * EPS, atol=0.0)
    assert np.all(result.residual_bound > 0.0)
    assert np.all(result.residual_norm <= result.residual_bound)


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


def test_resolved_design_pairs_each_column_with_its_site_in_the_given_order() -> None:
    genotypes, bounds, covariates, weights, variances, _prior_mean, _response = _problem(15)
    source = dual_solve.DenseDualSource(genotypes, bounds)
    models = dual_solve.DualModels(weights, variances, covariates)
    chosen = np.array([124, 28, 85, 3])
    sites = dual_solve.ResolvedSites({0: chosen}, {0: np.ones(chosen.size)}, {0: np.zeros(chosen.size)})
    designs = dual_solve.resolved_design(source, models, sites)
    _projector, design, _precision, _operator = _dense(genotypes, covariates, weights, variances, 0)
    np.testing.assert_allclose(designs[0], design[:, chosen], rtol=0.0, atol=genotypes.shape[0] * EPS * np.abs(design).max())
    repeated = dual_solve.ResolvedSites({0: np.array([3, 3])}, {0: np.ones(2)}, {0: np.zeros(2)})
    try:
        dual_solve.resolved_design(source, models, repeated)
    except ValueError as error:
        assert "distinct" in str(error)
    else:
        raise AssertionError("repeated indices must be refused")

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


def _gaussian_problem(seed: int):
    """Quantitative models on fold masks, with strong sites and, in model 0, two negative sites (A stays PD)."""
    genotypes, bounds, covariates, weights, variances, prior_mean, response = _problem(seed)
    rng = np.random.default_rng(seed + 100)
    training = (weights > 0).astype(np.float64)
    noise = np.array([0.7, 0.9, 1.3, 1.1])
    precision = 1.0 / variances
    negative = np.sort(np.argsort(variances[:, 0], kind="stable")[-2:])
    root = np.sqrt(training[:, 0] / noise[0])
    weighted_covariates = root[:, None] * covariates
    design = (root[:, None] * genotypes) - weighted_covariates @ np.linalg.lstsq(weighted_covariates, root[:, None] * genotypes, rcond=None)[0]
    data = design.T @ design
    precision[negative, 0] = 0.0
    block = np.linalg.inv(data + np.diag(precision[:, 0]))[np.ix_(negative, negative)]
    precision[negative, 0] = -0.5 / np.linalg.eigvalsh(block)[-1]
    shift = rng.standard_normal(precision.shape) * np.sqrt(np.abs(precision))
    offsets = rng.standard_normal(response.shape) * 0.1
    return genotypes, bounds, covariates, training, noise, precision, shift, response, offsets, negative


def _grams(bounds, adjacent):
    """Window blocks equal to the source's blocks; the maps read only the blocks and whether cross-Grams exist."""
    from sv_pgs.marginal_variances import BlockGrams

    blocks = tuple(np.arange(start, stop) for start, stop in bounds)
    within = tuple(np.zeros((block.size, block.size)) for block in blocks)
    next_cross = tuple(np.zeros((blocks[index].size, blocks[index + 1].size)) for index in range(len(blocks) - 1)) if adjacent else ()
    return BlockGrams(blocks=blocks, within=within, next_cross=next_cross)


def _dense_gaussian(genotypes, covariates, training, noise, precision, shift, response, offsets, model):
    weights = training[:, model] / noise[model]
    root = np.sqrt(weights)
    weighted_covariates = root[:, None] * covariates
    projector = np.eye(genotypes.shape[0]) - weighted_covariates @ np.linalg.pinv(weighted_covariates)
    design = projector @ (root[:, None] * genotypes)
    posterior_precision = design.T @ design + np.diag(precision[:, model])
    mean = np.linalg.solve(posterior_precision, design.T @ (projector @ (root * (response[:, model] - offsets[:, model]))) + shift[:, model])
    remainder = response[:, model] - offsets[:, model] - genotypes @ mean
    alpha = np.linalg.solve(covariates.T @ (weights[:, None] * covariates), covariates.T @ (weights * remainder))
    residual = remainder - covariates @ alpha
    return posterior_precision, mean, alpha, float(np.sum(training[:, model] * residual * residual)), design


def test_the_dual_gaussian_is_the_dense_posterior_with_strong_and_negative_sites() -> None:
    genotypes, bounds, covariates, training, noise, precision, shift, response, offsets, negative = _gaussian_problem(51)
    source = dual_solve.DenseDualSource(genotypes, bounds)
    gaussian = dual_solve.DualGaussian(source=source, training=training, targets=response, offsets=offsets, covariates=covariates, grams=_grams(bounds, True), probe_count=4, seed=3)
    scales = np.array([np.sqrt(float(_dense_gaussian(genotypes, covariates, training, noise, precision, shift, response, offsets, model)[1]
                                     @ _dense_gaussian(genotypes, covariates, training, noise, precision, shift, response, offsets, model)[0]
                                     @ _dense_gaussian(genotypes, covariates, training, noise, precision, shift, response, offsets, model)[1]))
                       for model in range(MODEL_COUNT)])
    error_bound = np.sqrt(EPS) * scales
    certificate = gaussian.iterate(site_precision=precision, site_shift=shift, noise_variance=noise, error_bound=error_bound, probe_residual_ratio=np.sqrt(EPS))
    assert np.all(certificate.error_bound <= error_bound)
    assert set(negative) <= set(gaussian.bulk_solves[0].resolved)
    rss = gaussian.residual_sum_of_squares()
    for model in range(MODEL_COUNT):
        posterior_precision, mean, alpha, residual_sum, _design = _dense_gaussian(genotypes, covariates, training, noise, precision, shift, response, offsets, model)
        error = gaussian.mean[:, model] - mean
        rounding = np.linalg.cond(posterior_precision) * genotypes.shape[1] * EPS * scales[model]
        assert np.sqrt(float(error @ posterior_precision @ error)) <= float(certificate.error_bound[model]) + rounding
        np.testing.assert_allclose(gaussian.alpha[:, model], alpha, rtol=0.0, atol=np.sqrt(EPS) * np.abs(alpha).max() * np.linalg.cond(posterior_precision))
        assert abs(rss[model] - residual_sum) <= np.sqrt(EPS) * residual_sum * np.linalg.cond(posterior_precision)


def test_the_dual_gaussian_refresh_quantities_match_the_dense_bulk_operator() -> None:
    genotypes, bounds, covariates, training, noise, precision, shift, response, offsets, _negative = _gaussian_problem(52)
    halves = [piece for start, stop in bounds for piece in ((start, (start + stop) // 2), ((start + stop) // 2, stop))]
    for window_bounds, adjacent in ((bounds, True), (halves, False), (halves, True)):
        _check_refresh_quantities(genotypes, bounds, covariates, training, noise, precision, shift, response, offsets, window_bounds, adjacent)


def _check_refresh_quantities(genotypes, bounds, covariates, training, noise, precision, shift, response, offsets, window_bounds, adjacent) -> None:
    source = dual_solve.DenseDualSource(genotypes, bounds)
    ratio = np.sqrt(EPS)
    grams = _grams(window_bounds, adjacent)
    gaussian = dual_solve.DualGaussian(source=source, training=training, targets=response, offsets=offsets, covariates=covariates, grams=grams, probe_count=3, seed=4)
    gaussian.iterate(site_precision=precision, site_shift=shift, noise_variance=noise, error_bound=np.full(MODEL_COUNT, np.sqrt(EPS)), probe_residual_ratio=ratio)
    for model, solve in enumerate(gaussian.bulk_solves):
        _precision, _mean, _alpha, _rss, design = _dense_gaussian(genotypes, covariates, training, noise, precision, shift, response, offsets, model)
        bulk = np.setdiff1d(np.arange(genotypes.shape[1]), solve.resolved)
        bulk_operator = np.eye(genotypes.shape[0]) + design[:, bulk] @ (design[:, bulk] / precision[bulk, model][None, :]).T
        inverse = np.linalg.inv(bulk_operator)
        conditioning = np.linalg.cond(bulk_operator)
        if solve.resolved.size:
            design_resolved = design[:, solve.resolved]
            core = np.diag(precision[solve.resolved, model]) + design_resolved.T @ inverse @ design_resolved
            np.testing.assert_allclose(solve.resolved_core, core, rtol=0.0, atol=ratio * conditioning * np.abs(core).max() * genotypes.shape[0])
            cross = design.T @ inverse @ design_resolved
            from sv_pgs.marginal_variances import BulkSolve, window_cross

            reference = window_cross(BulkSolve(solve.site_precision, solve.resolved, solve.resolved_core, cross, solve.bulk_trace,
                                               solve.bulk_square_trace, solve.kernel_square_trace, solve.sample_count), grams)
            for block in range(len(grams.blocks)):
                np.testing.assert_array_equal(solve.resolved_cross.positions[block], reference.positions[block])
                np.testing.assert_allclose(solve.resolved_cross.values[block], reference.values[block], rtol=0.0,
                                           atol=ratio * conditioning * np.abs(cross).max() * genotypes.shape[0])
        probes = gaussian.probes[:, gaussian.probe_models == model]
        count = float(training[:, model].sum())
        exact_trace = float(np.sum(probes * (inverse @ probes))) / (probes.shape[1] * count)
        # Each probe's quadratic form is off by at most ||z|| ||r|| <= ratio ||z||^2 (S_S >= I).
        assert abs(solve.bulk_trace - exact_trace) <= ratio * float(np.sum(probes * probes)) / (probes.shape[1] * count) * (1.0 + conditioning * genotypes.shape[0] * EPS)


def test_the_dual_gaussian_draws_have_the_posterior_covariance() -> None:
    genotypes, bounds, covariates, training, noise, precision, shift, response, offsets, _negative = _gaussian_problem(53)
    source = dual_solve.DenseDualSource(genotypes, bounds)
    gaussian = dual_solve.DualGaussian(source=source, training=training, targets=response, offsets=offsets, covariates=covariates, grams=_grams(bounds, True), probe_count=2, seed=5)
    gaussian.iterate(site_precision=precision, site_shift=shift, noise_variance=noise, error_bound=np.full(MODEL_COUNT, np.sqrt(EPS)), probe_residual_ratio=np.sqrt(EPS))
    model = 0
    resolved_count = gaussian.bulk_solves[model].resolved.size
    variant_count, sample_count = genotypes.shape[1], genotypes.shape[0]
    draw_count = resolved_count + variant_count + sample_count
    resolved_noise = {other: np.zeros((gaussian.bulk_solves[other].resolved.size, 0)) for other in range(MODEL_COUNT) if other != model and gaussian.bulk_solves[other].resolved.size}
    resolved_noise[model] = np.hstack([np.eye(resolved_count), np.zeros((resolved_count, variant_count + sample_count))])
    prior_noise = np.hstack([np.zeros((variant_count, resolved_count)), np.eye(variant_count), np.zeros((variant_count, sample_count))])
    sample_noise = np.hstack([np.zeros((sample_count, resolved_count + variant_count)), np.eye(sample_count)])
    draw_models = np.zeros(draw_count, dtype=np.int64)
    bound = np.linalg.norm(sample_noise, axis=0) * np.sqrt(EPS) + np.sqrt(EPS)
    draws, _norms = gaussian.draws_from_noise(prior_noise=prior_noise, sample_noise=sample_noise, resolved_noise=resolved_noise, draw_models=draw_models, error_bound=bound)
    draw_map = draws - gaussian.mean[:, [model]]
    posterior_precision = _dense_gaussian(genotypes, covariates, training, noise, precision, shift, response, offsets, model)[0]
    target = np.linalg.inv(posterior_precision)
    assert np.linalg.norm(draw_map @ draw_map.T - target) <= np.linalg.cond(posterior_precision) * np.sqrt(EPS) * np.linalg.norm(target)


def test_posterior_solve_is_the_dense_inverse_with_negative_sites() -> None:
    genotypes, bounds, covariates, training, noise, precision, shift, response, offsets, negative = _gaussian_problem(57)
    source = dual_solve.DenseDualSource(genotypes, bounds)
    gaussian = dual_solve.DualGaussian(source=source, training=training, targets=response, offsets=offsets, covariates=covariates, grams=_grams(bounds, True), probe_count=2, seed=7)
    gaussian.iterate(site_precision=precision, site_shift=shift, noise_variance=noise, error_bound=np.full(MODEL_COUNT, np.sqrt(EPS)), probe_residual_ratio=np.sqrt(EPS))
    rng = np.random.default_rng(58)
    right = rng.standard_normal((genotypes.shape[1], 5))
    for model in (0, 1):
        posterior_precision = _dense_gaussian(genotypes, covariates, training, noise, precision, shift, response, offsets, model)[0]
        exact = np.linalg.solve(posterior_precision, right)
        scale = np.sqrt(np.einsum("pc,pq,qc->c", exact, posterior_precision, exact))
        bound = np.sqrt(EPS) * scale
        solved, certificate = gaussian.posterior_solve(right, model, bound)
        assert np.all(certificate <= bound)
        error = solved - exact
        energy = np.sqrt(np.einsum("pc,pq,qc->c", error, posterior_precision, error))
        assert np.all(energy <= certificate + np.linalg.cond(posterior_precision) * genotypes.shape[1] * EPS * scale)
    assert set(negative) <= set(gaussian.bulk_solves[0].resolved)


def test_solves_sharing_z_l_refine_it_only_where_it_limits_the_certificate() -> None:
    # A Krylov loop around posterior_solve asks the same relative accuracy of ever smaller right-hand sides. Each
    # is the first one scaled, so every certificate scales with it: once Z_L suffices for the first, it suffices
    # for all, and refining Z_L on every miss (the bulk's included) would compound it down to float64's floor.
    genotypes, bounds, covariates, training, noise, precision, shift, response, offsets, _negative = _gaussian_problem(57)
    source = dual_solve.DenseDualSource(genotypes, bounds)
    gaussian = dual_solve.DualGaussian(source=source, training=training, targets=response, offsets=offsets, covariates=covariates, grams=_grams(bounds, True), probe_count=2, seed=7)
    gaussian.iterate(site_precision=precision, site_shift=shift, noise_variance=noise, error_bound=np.full(MODEL_COUNT, np.sqrt(EPS)), probe_residual_ratio=np.sqrt(EPS))
    posterior_precision = _dense_gaussian(genotypes, covariates, training, noise, precision, shift, response, offsets, 0)[0]
    right = np.random.default_rng(59).standard_normal((genotypes.shape[1], 3))
    exact = np.linalg.solve(posterior_precision, right)
    relative = np.sqrt(np.sqrt(EPS))
    bound = relative * np.sqrt(np.einsum("pc,pq,qc->c", exact, posterior_precision, exact))

    def refinements() -> int:
        return sum(1 for label, _columns, _error in gaussian.count.records if label.startswith("posterior-resolved"))

    gaussian.posterior_solve(right, 0, bound)
    after_first = refinements()
    for power in range(1, 60):
        scale = 0.5**power
        solved, certificate = gaussian.posterior_solve(scale * right, 0, scale * bound)
        assert np.all(certificate <= scale * bound)
        error = solved - scale * exact
        rounding = np.linalg.cond(posterior_precision) * genotypes.shape[1] * EPS * scale * bound / relative
        assert np.all(np.sqrt(np.einsum("pc,pq,qc->c", error, posterior_precision, error)) <= certificate + rounding)
    assert refinements() == after_first


def test_a_posterior_bound_below_float64_resolution_returns_the_floors_certificate() -> None:
    genotypes, bounds, covariates, training, noise, precision, shift, response, offsets, _negative = _gaussian_problem(57)
    source = dual_solve.DenseDualSource(genotypes, bounds)
    gaussian = dual_solve.DualGaussian(source=source, training=training, targets=response, offsets=offsets, covariates=covariates, grams=_grams(bounds, True), probe_count=2, seed=7)
    gaussian.iterate(site_precision=precision, site_shift=shift, noise_variance=noise, error_bound=np.full(MODEL_COUNT, np.sqrt(EPS)), probe_residual_ratio=np.sqrt(EPS))
    posterior_precision = _dense_gaussian(genotypes, covariates, training, noise, precision, shift, response, offsets, 0)[0]
    right = np.random.default_rng(60).standard_normal((genotypes.shape[1], 2))
    exact = np.linalg.solve(posterior_precision, right)
    solved, certificate = gaussian.posterior_solve(right, 0, np.zeros(2))
    # No solve certifies a zero error; the answer is float64's best, with the certificate it has.
    assert np.all(np.isfinite(certificate)) and np.all(certificate > 0.0)
    scale = np.sqrt(np.einsum("pc,pq,qc->c", exact, posterior_precision, exact))
    assert np.all(certificate <= np.sqrt(EPS) * scale)
    error = solved - exact
    assert np.all(np.sqrt(np.einsum("pc,pq,qc->c", error, posterior_precision, error)) <= certificate + np.linalg.cond(posterior_precision) * genotypes.shape[1] * EPS * scale)


def test_the_recursive_share_minimizes_the_cycles_digit_passes() -> None:
    for log_ratio, rate, scale, samples in ((np.log(1e6), 0.3, 50.0, 60000), (np.log(10.0), 1.2, 5.0, 3000), (0.5, 0.05, 400.0, 100000)):
        share = dual_solve.recursive_share(log_ratio, rate, scale, samples)
        assert 0.0 < share < 1.0

        def cost(value):
            iterations = (log_ratio + np.log(1.0 / value)) / rate
            inverse_error = 2 * scale * iterations * np.sqrt(np.exp(log_ratio) * value) / (1.0 - value)
            return iterations * max(0.0, np.log(np.sqrt(samples) * inverse_error))

        grid = np.linspace(share / 2, (1.0 + share) / 2, 2001)
        best = grid[np.argmin([cost(value) for value in grid])]
        # The stationary point is the grid's minimizer to within the grid's spacing.
        assert abs(best - share) <= grid[1] - grid[0]


def test_information_solve_gives_the_bulk_back_products_and_coupling() -> None:
    genotypes, bounds, covariates, training, noise, precision, shift, response, offsets, _negative = _gaussian_problem(59)
    source = dual_solve.DenseDualSource(genotypes, bounds)
    gaussian = dual_solve.DualGaussian(source=source, training=training, targets=response, offsets=offsets, covariates=covariates, grams=_grams(bounds, True), probe_count=2, seed=8)
    gaussian.iterate(site_precision=precision, site_shift=shift, noise_variance=noise, error_bound=np.full(MODEL_COUNT, np.sqrt(EPS)), probe_residual_ratio=np.sqrt(EPS))
    rng = np.random.default_rng(60)
    probes = rng.choice(np.array([-1.0, 1.0]), size=(genotypes.shape[1], 3))
    for model in (0, 2):
        resolved = gaussian.bulk_solves[model].resolved
        _posterior, _mean, _alpha, _rss, design = _dense_gaussian(genotypes, covariates, training, noise, precision, shift, response, offsets, model)
        bulk = np.setdiff1d(np.arange(genotypes.shape[1]), resolved)
        variances = np.zeros(genotypes.shape[1])
        variances[bulk] = 1.0 / precision[bulk, model]
        kernel = np.eye(genotypes.shape[0]) + (design * variances[None, :]) @ design.T
        image = design @ (variances[:, None] * probes)
        inverse_image = np.linalg.solve(kernel, image)
        design_resolved = design[:, resolved]
        resolved_duals = np.linalg.solve(kernel, design_resolved)
        core = np.diag(precision[resolved, model]) + design_resolved.T @ resolved_duals
        coupling = resolved_duals.T @ image
        w = inverse_image - resolved_duals @ np.linalg.solve(core, coupling - probes[resolved]) if resolved.size else inverse_image
        expected = design.T @ w
        tolerance = np.sqrt(EPS)
        back_products, resolved_coupling, residual_norm = gaussian.information_solve(probes, model, tolerance)
        assert np.all(residual_norm <= tolerance)
        conditioning = np.linalg.cond(kernel) * (np.linalg.cond(core) if resolved.size else 1.0)
        np.testing.assert_allclose(back_products, expected, rtol=0.0, atol=conditioning * np.sqrt(EPS) * np.abs(expected).max() * genotypes.shape[0])
        if resolved.size:
            np.testing.assert_allclose(resolved_coupling, coupling, rtol=0.0, atol=conditioning * np.sqrt(EPS) * np.abs(coupling).max() * genotypes.shape[0])
