"""Stage 2 exact polish: exact E-step, certified fixed point, equality with the full fit.

Every reference here is a dense solve written independently of the block
Gauss-Seidel code: the joint (alpha, beta) normal equations for a Gaussian
trait, Newton's method on the penalized logistic likelihood for a binary one,
and explicit inverses for the posterior variances.
"""
from __future__ import annotations

from dataclasses import replace as dataclass_replace

import numpy as np
import pytest
from scipy.special import expit

from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.exact_polish import (
    DenseGenotypeBlockSource,
    PolishModel,
    PolishStart,
    _BatchedModelState,
    _gauss_seidel_pass,
    _initial_linear_predictor,
    polish_to_fixed_point,
)
from sv_pgs.inference import fit_variational_em
from sv_pgs.mixture_inference import (
    _as_standardized_genotype_matrix,
    _binary_expected_polya_gamma_weights,
    _build_prior_design,
    _effective_prior_variances,
    _fit_collapsed_posterior,
    _metadata_baseline_scales_from_coefficients,
    _propose_cavi_hyperparameters,
    _scale_model_penalty,
)
from sv_pgs.preprocessing import build_tie_map

from tests.conftest import make_variant_records


def _records(variant_count: int):
    snv_count = variant_count // 2
    records = make_variant_records(snv_count, VariantClass.SNV) + make_variant_records(
        variant_count - snv_count,
        VariantClass.DELETION_SHORT,
    )
    return [
        dataclass_replace(record, variant_id=f"variant_{variant_index}", position=variant_index * 100)
        for variant_index, record in enumerate(records)
    ]


def _simulate(*, sample_count: int, variant_count: int, trait_type: TraitType, seed: int):
    """Standardized genotypes with AR(1) LD that runs across every block boundary."""
    generator = np.random.default_rng(seed)
    latent = generator.standard_normal((sample_count, variant_count))
    for column in range(1, variant_count):
        latent[:, column] = 0.7 * latent[:, column - 1] + np.sqrt(1.0 - 0.49) * latent[:, column]
    dosage = (latent > 0.6).astype(np.float64) + (latent > 1.4).astype(np.float64)
    dosage += 0.05 * generator.standard_normal(dosage.shape)
    genotypes = (dosage - dosage.mean(axis=0)) / dosage.std(axis=0)
    covariates = np.column_stack([np.ones(sample_count), generator.standard_normal(sample_count)])
    effects = np.zeros(variant_count)
    causal = generator.choice(variant_count, size=max(variant_count // 20, 3), replace=False)
    effects[causal] = generator.standard_normal(causal.size) * 0.35
    effects += generator.standard_normal(variant_count) * 0.02
    genetic = genotypes @ effects
    genetic *= 0.6 / np.std(genetic)
    if trait_type == TraitType.BINARY:
        targets = (generator.random(sample_count) < expit(-0.8 + 0.3 * covariates[:, 1] + genetic)).astype(np.float64)
    else:
        targets = 1.5 + 0.3 * covariates[:, 1] + genetic + generator.standard_normal(sample_count)
    return genotypes, covariates, targets


def _blocks(variant_count: int, block_size: int) -> list[np.ndarray]:
    return [np.arange(start, min(start + block_size, variant_count)) for start in range(0, variant_count, block_size)]


def _config(trait_type: TraitType, *, max_outer_iterations: int, convergence_tolerance: float) -> ModelConfig:
    return ModelConfig(
        trait_type=trait_type,
        max_outer_iterations=max_outer_iterations,
        convergence_tolerance=convergence_tolerance,
        exact_solver_matrix_limit=4096,
        minimum_minor_allele_frequency=0.0,
        beta_variance_update_interval=1,
        stochastic_variational_updates=False,
        binary_inner_tolerance=1e-12,
        max_inner_newton_iterations=200,
        random_seed=0,
    )


def _dense_mode(*, genotypes, covariates, targets, offset, weights_mask, prior_precision, noise_variance, trait_type):
    """Exact (alpha, beta) maximizer of the penalized likelihood with a flat alpha prior."""
    design = np.column_stack([covariates, genotypes])
    penalty = np.concatenate([np.zeros(covariates.shape[1]), prior_precision])
    if trait_type == TraitType.QUANTITATIVE:
        weight = weights_mask / noise_variance
        normal_matrix = design.T @ (weight[:, None] * design) + np.diag(penalty)
        solution = np.linalg.solve(normal_matrix, design.T @ (weight * (targets - offset)))
        return solution[: covariates.shape[1]], solution[covariates.shape[1]:]
    solution = np.zeros(design.shape[1])
    for _newton_index in range(200):
        linear_predictor = offset + design @ solution
        probability = expit(linear_predictor)
        gradient = design.T @ (weights_mask * (targets - probability)) - penalty * solution
        hessian = design.T @ ((weights_mask * probability * (1.0 - probability))[:, None] * design) + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        solution += step
        if np.max(np.abs(step)) < 1e-14:
            break
    return solution[: covariates.shape[1]], solution[covariates.shape[1]:]


def _dense_restricted_precision(*, genotypes, covariates, weights, prior_precision):
    """X' P_W X + diag(P) with the restricted-posterior projector's covariate ridge
    (trace(C'WC) / k * 1e-6, as _restricted_precision_projector)."""
    weighted_covariates = weights[:, None] * covariates
    cross = genotypes.T @ weighted_covariates
    covariate_gram = covariates.T @ weighted_covariates
    ridge = max(float(np.trace(covariate_gram)) / covariates.shape[1] * 1e-6, 1e-8)
    return (
        genotypes.T @ (weights[:, None] * genotypes)
        - cross @ np.linalg.solve(covariate_gram + ridge * np.eye(covariates.shape[1]), cross.T)
        + np.diag(prior_precision)
    )


def _block_inverse(precision: np.ndarray, blocks: list[np.ndarray]) -> np.ndarray:
    inverse = np.zeros_like(precision)
    for block in blocks:
        inverse[np.ix_(block, block)] = np.linalg.inv(precision[np.ix_(block, block)])
    return inverse


def _start_from_prior(covariates, targets, variant_count: int, trait_type: TraitType) -> PolishStart:
    prior_design = _build_prior_design(_records(variant_count))
    class_count = prior_design.class_membership_matrix.shape[1]
    return PolishStart(
        alpha=np.zeros(covariates.shape[1]),
        beta=np.zeros(variant_count),
        global_scale=0.05,
        scale_model_coefficients=np.zeros(prior_design.design_matrix.shape[1]),
        tpb_shape_a_vector=np.ones(class_count),
        tpb_shape_b_vector=np.ones(class_count),
        local_scale=np.ones(variant_count),
        auxiliary_delta=np.ones(variant_count),
        sigma_error2=float(np.var(targets)) if trait_type == TraitType.QUANTITATIVE else 1.0,
    )


@pytest.mark.parametrize("trait_type", [TraitType.QUANTITATIVE, TraitType.BINARY])
@pytest.mark.parametrize(("sample_count", "variant_count"), [(400, 120), (150, 240)])
def test_block_gauss_seidel_reaches_the_exact_joint_posterior(trait_type, sample_count, variant_count):
    """At fixed hyperparameters the sweeps reach the exact joint mode, every probe
    solve reaches A^-1 z, and the variance estimate is the control-variate
    Hutchinson estimate of the full-covariance diagonal (fold mask included)."""
    genotypes, covariates, targets = _simulate(
        sample_count=sample_count,
        variant_count=variant_count,
        trait_type=trait_type,
        seed=3,
    )
    blocks = _blocks(variant_count, 25)
    generator = np.random.default_rng(11)
    prior_precision = np.exp(generator.uniform(np.log(20.0), np.log(2000.0), variant_count))
    noise_variance = 0.8
    sample_masks = np.ones((2, sample_count))
    sample_masks[1, : sample_count // 5] = 0.0
    config = _config(trait_type, max_outer_iterations=5, convergence_tolerance=1e-6)
    models = [
        PolishModel(
            config=config,
            targets=targets,
            sample_mask_index=mask_index,
            predictor_offset=np.full(sample_count, 0.1),
            start=_start_from_prior(covariates, targets, variant_count, trait_type),
        )
        for mask_index in range(2)
    ]
    source = DenseGenotypeBlockSource(genotypes, blocks)
    state = _BatchedModelState(
        models=models,
        covariates=covariates,
        sample_masks=sample_masks,
        variant_count=variant_count,
        probe_count=8,
    )
    state.prior_precision[:] = prior_precision[:, None]
    state.noise_variance[:] = noise_variance
    state.linear_predictor = _initial_linear_predictor(source, state)
    exact_control_variate = np.full(2, trait_type == TraitType.BINARY)
    for _pass_index in range(300):
        moments = _gauss_seidel_pass(source=source, state=state, exact_binary_variance=exact_control_variate)
        if np.all(moments.gradient_relative_norm < 1e-12) and np.all(moments.maximum_probe_residual < 1e-12):
            break
    assert np.all(moments.gradient_relative_norm < 1e-12)
    assert np.all(moments.maximum_probe_residual < 1e-12)
    for mask_index in range(2):
        mask = sample_masks[mask_index]
        expected_alpha, expected_beta = _dense_mode(
            genotypes=genotypes,
            covariates=covariates,
            targets=targets,
            offset=0.1,
            weights_mask=mask,
            prior_precision=prior_precision,
            noise_variance=noise_variance,
            trait_type=trait_type,
        )
        np.testing.assert_allclose(
            state.beta[:, mask_index], expected_beta, rtol=0.0, atol=1e-9 * np.max(np.abs(expected_beta))
        )
        np.testing.assert_allclose(
            state.alpha[:, mask_index], expected_alpha, rtol=0.0, atol=1e-9 * np.max(np.abs(expected_alpha))
        )
        if trait_type == TraitType.QUANTITATIVE:
            weights = mask / noise_variance
        else:
            weights = mask * _binary_expected_polya_gamma_weights(
                linear_predictor=0.1 + covariates @ expected_alpha + genotypes @ expected_beta,
                minimum_weight=config.polya_gamma_minimum_weight,
            )
        precision = _dense_restricted_precision(
            genotypes=genotypes,
            covariates=covariates,
            weights=weights,
            prior_precision=prior_precision,
        )
        probe_solutions = np.linalg.solve(precision, state.probes)
        np.testing.assert_allclose(
            state.probe_solution[:, state.probe_columns(mask_index)],
            probe_solutions,
            rtol=0.0,
            atol=1e-9 * np.max(np.abs(probe_solutions)),
        )
        control_variate = _block_inverse(precision, blocks)
        expected_estimate = np.diag(control_variate) + np.mean(
            state.probes * (probe_solutions - control_variate @ state.probes),
            axis=1,
        )
        np.testing.assert_allclose(moments.start_variance[:, mask_index], expected_estimate, rtol=1e-6)


def _early_checkpoint(genotypes, covariates, targets, records, config, *, checkpoint_iteration: int):
    """The current individual-level CAVI fit's state after checkpoint_iteration iterations."""
    captured = {}

    def keep_checkpoint(checkpoint):
        if checkpoint.completed_iterations == checkpoint_iteration:
            captured["checkpoint"] = checkpoint

    tie_map = build_tie_map(genotypes, records, config)
    assert tie_map.kept_indices.shape[0] == genotypes.shape[1]
    fit_variational_em(
        genotypes=genotypes,
        covariates=covariates,
        targets=targets,
        records=records,
        config=dataclass_replace(config, max_outer_iterations=checkpoint_iteration),
        tie_map=tie_map,
        checkpoint_callback=keep_checkpoint,
    )
    return captured["checkpoint"]


def _full_fit_map(polished, genotypes, covariates, targets, records, config):
    """One step of the full fit's own CAVI map at the polish result.

    The E-step is _fit_collapsed_posterior (the routine every individual-level
    EM iteration calls, exact dense routes here, full-covariance variances);
    the M-step is the shared CAVI proposal.
    """
    prior_design = _build_prior_design(records)
    posterior = _fit_collapsed_posterior(
        genotype_matrix=_as_standardized_genotype_matrix(genotypes),
        covariate_matrix=covariates,
        targets=targets,
        reduced_prior_variances=polished.prior_variances,
        sigma_error2=polished.sigma_error2,
        alpha_init=polished.alpha,
        beta_init=polished.beta,
        trait_type=config.trait_type,
        config=config,
        compute_logdet=False,
        compute_beta_variance=True,
    )
    baseline = (
        polished.global_scale
        * _metadata_baseline_scales_from_coefficients(
            polished.scale_model_coefficients,
            prior_design.design_matrix,
            config,
        )
    ) ** 2
    proposal = _propose_cavi_hyperparameters(
        reduced_second_moment=posterior.beta * posterior.beta + posterior.beta_variance,
        baseline_reduced_prior_variances=baseline,
        local_shape_a=prior_design.class_membership_matrix @ polished.tpb_shape_a_vector,
        local_shape_b=prior_design.class_membership_matrix @ polished.tpb_shape_b_vector,
        auxiliary_delta=polished.auxiliary_delta,
        global_scale=polished.global_scale,
        scale_model_coefficients=polished.scale_model_coefficients,
        tpb_shape_a_vector=polished.tpb_shape_a_vector,
        tpb_shape_b_vector=polished.tpb_shape_b_vector,
        update_scale_and_shapes=True,
        prior_design=prior_design,
        scale_penalty=_scale_model_penalty(prior_design.feature_names, config),
        config=config,
    )
    next_baseline = (
        proposal.global_scale
        * _metadata_baseline_scales_from_coefficients(
            proposal.scale_model_coefficients,
            prior_design.design_matrix,
            config,
        )
    ) ** 2
    return posterior, proposal, _effective_prior_variances(
        baseline_prior_variances=next_baseline,
        local_scale=proposal.local_scale,
        config=config,
    )


def _polish_one(genotypes, covariates, targets, records, config, blocks, start):
    [polished] = polish_to_fixed_point(
        source=DenseGenotypeBlockSource(genotypes, blocks),
        covariates=covariates,
        sample_masks=np.ones((1, genotypes.shape[0])),
        models=[
            PolishModel(
                config=config,
                targets=targets,
                sample_mask_index=0,
                predictor_offset=np.zeros(genotypes.shape[0]),
                start=start,
            )
        ],
        prior_design=_build_prior_design(records),
    )
    return polished


@pytest.mark.parametrize("trait_type", [TraitType.QUANTITATIVE, TraitType.BINARY])
@pytest.mark.parametrize(("sample_count", "variant_count"), [(500, 80), (160, 220)])
def test_polish_from_an_early_checkpoint_lands_on_the_fixed_point_of_the_full_fit_map(
    trait_type,
    sample_count,
    variant_count,
):
    """One LD block (the variance estimator is then exact): from the full fit's
    3-iteration state, Stage 2 certifies a state that the full fit's own E-step
    and M-step map to itself, for p<n and p>n."""
    genotypes, covariates, targets = _simulate(
        sample_count=sample_count,
        variant_count=variant_count,
        trait_type=trait_type,
        seed=5,
    )
    records = _records(variant_count)
    config = _config(trait_type, max_outer_iterations=600, convergence_tolerance=1e-8)
    checkpoint = _early_checkpoint(genotypes, covariates, targets, records, config, checkpoint_iteration=3)
    polished = _polish_one(
        genotypes,
        covariates,
        targets,
        records,
        config,
        [np.arange(variant_count)],
        PolishStart.from_checkpoint(checkpoint),
    )
    assert polished.certificate.certified, polished.certificate
    assert polished.certificate.variance_relative_standard_error < 1e-6
    posterior, proposal, next_prior_variances = _full_fit_map(polished, genotypes, covariates, targets, records, config)
    beta_scale = float(np.max(np.abs(posterior.beta)))
    np.testing.assert_allclose(polished.beta, posterior.beta, rtol=0.0, atol=1e-5 * beta_scale)
    np.testing.assert_allclose(polished.beta_variance, posterior.beta_variance, rtol=1e-5)
    np.testing.assert_allclose(next_prior_variances, polished.prior_variances, rtol=1e-4)
    np.testing.assert_allclose(proposal.global_scale, polished.global_scale, rtol=1e-4)
    np.testing.assert_allclose(proposal.tpb_shape_a_vector, polished.tpb_shape_a_vector, rtol=1e-4)
    np.testing.assert_allclose(proposal.tpb_shape_b_vector, polished.tpb_shape_b_vector, rtol=1e-4)
    if trait_type == TraitType.QUANTITATIVE:
        np.testing.assert_allclose(posterior.sigma_error2, polished.sigma_error2, rtol=1e-5)


@pytest.mark.parametrize("trait_type", [TraitType.QUANTITATIVE, TraitType.BINARY])
def test_polish_across_ld_blocks_matches_the_full_fit_map_within_the_probe_error(trait_type):
    """Several LD blocks with LD across every boundary: the certified mean is the
    exact E-step mean at the certified prior, and the Hutchinson variances agree
    with the exact full-covariance variances within the reported standard error."""
    sample_count, variant_count = 400, 150
    genotypes, covariates, targets = _simulate(
        sample_count=sample_count,
        variant_count=variant_count,
        trait_type=trait_type,
        seed=5,
    )
    records = _records(variant_count)
    config = _config(trait_type, max_outer_iterations=600, convergence_tolerance=1e-6)
    checkpoint = _early_checkpoint(genotypes, covariates, targets, records, config, checkpoint_iteration=3)
    polished = _polish_one(
        genotypes,
        covariates,
        targets,
        records,
        config,
        _blocks(variant_count, 30),
        PolishStart.from_checkpoint(checkpoint),
    )
    assert polished.certificate.certified, polished.certificate
    posterior, _proposal, _next_prior_variances = _full_fit_map(polished, genotypes, covariates, targets, records, config)
    beta_scale = float(np.max(np.abs(posterior.beta)))
    np.testing.assert_allclose(polished.beta, posterior.beta, rtol=0.0, atol=1e-5 * beta_scale)
    exact_leverage = float(np.sum(posterior.beta_variance / polished.prior_variances))
    estimated_leverage = float(np.sum(polished.beta_variance / polished.prior_variances))
    standard_error = polished.certificate.variance_relative_standard_error
    assert 0.0 < standard_error < 0.05
    assert abs(estimated_leverage - exact_leverage) / exact_leverage < 4.0 * standard_error


def test_batched_models_and_fold_masks_match_separate_fits():
    """One pass serving a Gaussian full-cohort model, a Gaussian fold model and a
    binary model returns what each model gets alone, and a fold model equals the
    same model fitted on its training rows only."""
    sample_count, variant_count = 300, 90
    genotypes, covariates, quantitative_targets = _simulate(
        sample_count=sample_count,
        variant_count=variant_count,
        trait_type=TraitType.QUANTITATIVE,
        seed=7,
    )
    _genotypes, _covariates, binary_targets = _simulate(
        sample_count=sample_count,
        variant_count=variant_count,
        trait_type=TraitType.BINARY,
        seed=7,
    )
    records = _records(variant_count)
    blocks = _blocks(variant_count, 30)
    sample_masks = np.ones((2, sample_count))
    sample_masks[1, :60] = 0.0
    quantitative_config = _config(TraitType.QUANTITATIVE, max_outer_iterations=600, convergence_tolerance=1e-7)
    binary_config = _config(TraitType.BINARY, max_outer_iterations=600, convergence_tolerance=1e-7)
    quantitative_start = PolishStart.from_checkpoint(
        _early_checkpoint(genotypes, covariates, quantitative_targets, records, quantitative_config, checkpoint_iteration=3)
    )
    binary_start = PolishStart.from_checkpoint(
        _early_checkpoint(genotypes, covariates, binary_targets, records, binary_config, checkpoint_iteration=3)
    )
    specifications = [
        (quantitative_config, quantitative_targets, 0, quantitative_start),
        (quantitative_config, quantitative_targets, 1, quantitative_start),
        (binary_config, binary_targets, 0, binary_start),
    ]
    source = DenseGenotypeBlockSource(genotypes, blocks)
    prior_design = _build_prior_design(records)

    def polish_models(model_specifications, *, model_source, model_covariates, model_masks):
        return polish_to_fixed_point(
            source=model_source,
            covariates=model_covariates,
            sample_masks=model_masks,
            models=[
                PolishModel(
                    config=config,
                    targets=targets,
                    sample_mask_index=mask_index,
                    predictor_offset=np.zeros(targets.shape[0]),
                    start=start,
                )
                for config, targets, mask_index, start in model_specifications
            ],
            prior_design=prior_design,
        )

    batched = polish_models(specifications, model_source=source, model_covariates=covariates, model_masks=sample_masks)
    for specification, batched_fit in zip(specifications, batched):
        [separate_fit] = polish_models(
            [specification],
            model_source=source,
            model_covariates=covariates,
            model_masks=sample_masks,
        )
        assert batched_fit.certificate.certified and separate_fit.certificate.certified
        np.testing.assert_allclose(
            batched_fit.beta, separate_fit.beta, rtol=0.0, atol=1e-6 * np.max(np.abs(separate_fit.beta))
        )
        np.testing.assert_allclose(batched_fit.prior_variances, separate_fit.prior_variances, rtol=1e-5)
    training_rows = sample_masks[1] > 0.0
    [subset_fit] = polish_models(
        [(quantitative_config, quantitative_targets[training_rows], 0, quantitative_start)],
        model_source=DenseGenotypeBlockSource(genotypes[training_rows], blocks),
        model_covariates=covariates[training_rows],
        model_masks=np.ones((1, int(training_rows.sum()))),
    )
    assert subset_fit.certificate.certified
    np.testing.assert_allclose(batched[1].beta, subset_fit.beta, rtol=0.0, atol=1e-6 * np.max(np.abs(subset_fit.beta)))
    np.testing.assert_allclose(batched[1].sigma_error2, subset_fit.sigma_error2, rtol=1e-6)
