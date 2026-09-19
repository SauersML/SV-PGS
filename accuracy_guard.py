"""Accuracy guard: math-core's reproducer (refvb.simulate) for current SV-PGS vs Stage 1.

Quantitative: n=600/p=1500 seeds 0,1 and n=1000/p=2000 seed 0 (h2 = 0.4, 30-variant AR(0.9) blocks),
held-out corr^2 with the true genetic value. Stage 1 sees only the within-block training LD
(its block-diagonal approximation; at p > n this ignores cross-block sampling correlation).
Binary: n=1500, p=1500, logistic, held-out AUC and corr^2 with the true logit.
"""
import sys
import time

import numpy as np
from scipy.special import expit

sys.path.insert(0, ".")
sys.path.insert(0, sys.argv[1])
import refvb
from sv_pgs import ld_space_fit
from sv_pgs.compute_budget import detect_compute_budget
from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import VariantRecord
from sv_pgs.inference import fit_variational_em
from sv_pgs.preprocessing import build_tie_map

BLOCK = 30
BUDGET = detect_compute_budget()


def records_for(cls):
    return [VariantRecord(variant_id=f"v{i}", variant_class=VariantClass.DELETION_SHORT if c else VariantClass.SNV,
                          chromosome="chr1", position=i * 100, length=500.0 if c else 1.0, allele_frequency=0.2,
                          quality=1.0, training_support=40 if c else None) for i, c in enumerate(cls)]


def stage1_inputs(X, W):
    coefficients, *_ = np.linalg.lstsq(W, X, rcond=None)
    projected = X - W @ coefficients
    n, p = X.shape
    boundaries = np.arange(0, p + BLOCK, BLOCK)
    boundaries[-1] = p
    boundaries = np.unique(boundaries).astype(np.int64)
    blocks = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        block = projected[:, start:stop].T @ projected[:, start:stop] / n
        blocks.append(0.5 * (block + block.T))
    return projected, ld_space_fit.InMemoryLDBlocks(block_boundaries=boundaries, correlation_blocks=tuple(blocks))


def hypermodel_for(cls):
    sv = cls.astype(np.float64)
    return ld_space_fit.LDPriorHypermodel(
        annotation_design=(sv - sv.mean())[:, None], annotation_prior_mean=np.zeros(1), annotation_prior_precision=np.ones(1),
        log_variance_offset=np.zeros(cls.shape[0]), variant_class_index=cls.astype(np.int64), class_names=("snv", "sv"),
        shape_a=0.5, initial_shape_b=np.array([0.5, 0.4]), shape_b_pooling_variance=0.25)


def quantitative(seed, n_train, p):
    data = refvb.simulate(n_train, 2000, p, seed=seed, polygenic_frac=0.0, h2=0.4)
    rows = []
    records = records_for(data["cls"])
    config = ModelConfig(trait_type=TraitType.QUANTITATIVE, max_outer_iterations=40, random_seed=0, stochastic_variational_updates=False)
    genotypes = data["X"].astype(np.float32)
    tie_map = build_tie_map(genotypes, records, config)
    start = time.time()
    result = fit_variational_em(genotypes=genotypes, covariates=data["W"].astype(np.float32), targets=data["y"].astype(np.float32),
                                records=records, config=config, tie_map=tie_map)
    beta = np.zeros(p)
    beta[tie_map.kept_indices] = result.beta_reduced
    rows.append(("current_sv_pgs", np.corrcoef(data["Xte"] @ beta, data["gte"])[0, 1] ** 2, time.time() - start, ""))
    projected, ld = stage1_inputs(data["X"], data["W"])
    coefficients, *_ = np.linalg.lstsq(data["W"], data["y"], rcond=None)
    residual = data["y"] - data["W"] @ coefficients
    statistics = ld_space_fit.QuantitativeTraitStatistics(sample_count=n_train, covariate_count=data["W"].shape[1],
                                                          score=projected.T @ residual, residual_sum_of_squares=float(residual @ residual))
    for scheme in ("expectation_propagation", "coherent_vb", "plug_in"):
        start = time.time()
        (fit,) = ld_space_fit.fit_ld_space(ld, [statistics], hypermodel_for(data["cls"]), BUDGET, scheme_name=scheme)
        rows.append((f"stage1_{scheme}", np.corrcoef(data["Xte"] @ fit.posterior_mean, data["gte"])[0, 1] ** 2, time.time() - start,
                     f"passes={fit.passes} converged={fit.converged} b={np.round(fit.shape_b, 3)}"))
    for name, corr2, seconds, extra in rows:
        print(f"QUANT seed={seed} n={n_train} p={p} {name:34s} corr2={corr2:.4f} t={seconds:.1f}s {extra}", flush=True)


def logistic_offset_fit(W, y, offset):
    alpha = np.zeros(W.shape[1])
    for _ in range(100):
        prob = expit(W @ alpha + offset)
        step = np.linalg.solve(W.T @ (W * (prob * (1 - prob))[:, None]), W.T @ (y - prob))
        alpha += step
        if np.max(np.abs(step)) < 1e-12:
            break
    return alpha


def auc(labels, scores):
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.shape[0])
    ranks[order] = np.arange(1, scores.shape[0] + 1)
    pos = labels == 1
    npos = int(pos.sum())
    return (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * (labels.shape[0] - npos))


def binary(seed, n_train, p):
    data = refvb.simulate(n_train, 5000, p, seed=seed, polygenic_frac=0.0, h2=0.4)
    rng = np.random.default_rng(seed + 100)
    genetic_train = data["X"] @ data["beta"]
    scale = 0.8 / np.std(genetic_train)
    logit_train = -1.4 + scale * genetic_train
    logit_test = -1.4 + scale * data["gte"]
    y = (rng.uniform(size=n_train) < expit(logit_train)).astype(np.float64)
    y_test = (rng.uniform(size=logit_test.shape[0]) < expit(logit_test)).astype(np.float64)
    W, Wte = data["W"], data["Wte"]
    rows = []
    records = records_for(data["cls"])
    config = ModelConfig(trait_type=TraitType.BINARY, max_outer_iterations=40, random_seed=0, stochastic_variational_updates=False)
    genotypes = data["X"].astype(np.float32)
    tie_map = build_tie_map(genotypes, records, config)
    start = time.time()
    result = fit_variational_em(genotypes=genotypes, covariates=W.astype(np.float32), targets=y.astype(np.float32),
                                records=records, config=config, tie_map=tie_map)
    beta = np.zeros(p)
    beta[tie_map.kept_indices] = result.beta_reduced
    rows.append(("current_sv_pgs", beta, time.time() - start, ""))
    projected, ld = stage1_inputs(data["X"], W)
    hyper = hypermodel_for(data["cls"])
    start = time.time()
    alpha = logistic_offset_fit(W, y, np.zeros(n_train))
    prob = expit(W @ alpha)
    first_stats = ld_space_fit.binary_statistics_at(residual_score=projected.T @ (y - prob), gram_times_expansion=np.zeros(p),
                                                    fitted_probability=prob, covariate_count=W.shape[1])
    (first,) = ld_space_fit.fit_ld_space(ld, [first_stats], hyper, BUDGET)
    rows.append(("stage1_ep_mean_weight", first.posterior_mean, time.time() - start, f"passes={first.passes} converged={first.converged}"))
    offset = projected @ first.posterior_mean
    alpha = logistic_offset_fit(W, y, offset)
    prob = expit(W @ alpha + offset)
    refreshed_stats = ld_space_fit.binary_statistics_at(residual_score=projected.T @ (y - prob),
                                                        gram_times_expansion=projected.T @ (projected @ first.posterior_mean),
                                                        fitted_probability=prob, covariate_count=W.shape[1])
    (refreshed,) = ld_space_fit.fit_ld_space(ld, [refreshed_stats], hyper, BUDGET, warm_starts=[first])
    rows.append(("stage1_ep_one_refresh", refreshed.posterior_mean, time.time() - start, f"passes={refreshed.passes} converged={refreshed.converged}"))
    for name, effect, seconds, extra in rows:
        fitted_alpha = logistic_offset_fit(W, y, data["X"] @ effect)
        score = Wte @ fitted_alpha + data["Xte"] @ effect
        corr2 = np.corrcoef(data["Xte"] @ effect, data["gte"])[0, 1] ** 2
        print(f"BINARY seed={seed} n={n_train} p={p} {name:34s} auc={auc(y_test, score):.4f} corr2={corr2:.4f} t={seconds:.1f}s {extra}", flush=True)


for seed, n_train, p in [(0, 600, 1500), (1, 600, 1500), (0, 1000, 2000)]:
    quantitative(seed, n_train, p)
for seed in (0, 1):
    binary(seed, 1500, 1500)
