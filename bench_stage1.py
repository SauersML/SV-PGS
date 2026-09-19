"""Stage 1 wall-clock benchmark at biobank scale on synthetic LD (MSI; not a test).

LD: a pool of AR(1) Toeplitz correlation blocks R_jk = rho^|j-k| (widths 2500-4096,
rho 0.90-0.99) reused cyclically along the genome, served as float32 like the
store's Grams. Dense block linear algebra costs the same whatever the LD values.
Trait: sparse causal effects (1 per 2000 variants), h2 = 0.3, n = 100k; the score is
drawn from its exact sampling distribution g = n R beta + sqrt(n sigma2) R^(1/2) eps,
with R beta and R^(1/2) eps from AR(1) filters in O(p).

usage: bench_stage1.py VARIANTS SCHEME PASSES [TRAITS]
  PASSES = 0 runs to convergence; PASSES = k stops after k passes (per-pass timing).
"""
from __future__ import annotations

import resource
import sys
import time

import numpy as np
from scipy.signal import lfilter

from sv_pgs import ld_space_fit
from sv_pgs.compute_budget import detect_compute_budget

SAMPLE_COUNT = 100_000
HERITABILITY = 0.3
POOL_WIDTHS = (2500, 2900, 3300, 3600, 3800, 4000, 4096, 3100)
POOL_CORRELATIONS = (0.90, 0.95, 0.97, 0.99, 0.93, 0.98, 0.96, 0.94)


class CyclicToeplitzLD:
    def __init__(self, variant_count: int) -> None:
        self.pool = []
        self.pool_scores = []
        for width, correlation in zip(POOL_WIDTHS, POOL_CORRELATIONS, strict=True):
            lags = np.abs(np.arange(width)[:, None] - np.arange(width)[None, :])
            block = (correlation ** lags).astype(np.float32)
            self.pool.append(block)
            self.pool_scores.append(np.einsum("ij,ij->j", block.astype(np.float64), block.astype(np.float64)))
        widths, pool_index = [], []
        total = 0
        position = 0
        while total < variant_count:
            entry = position % len(POOL_WIDTHS)
            width = min(POOL_WIDTHS[entry], variant_count - total)
            if width < POOL_WIDTHS[entry]:
                entry = self._pool_entry_for_width(width)
            widths.append(width)
            pool_index.append(entry)
            total += width
            position += 1
        self.pool_index = np.asarray(pool_index)
        self._boundaries = np.concatenate([[0], np.cumsum(widths)]).astype(np.int64)

    def _pool_entry_for_width(self, width: int) -> int:
        lags = np.abs(np.arange(width)[:, None] - np.arange(width)[None, :])
        block = (0.95 ** lags).astype(np.float32)
        self.pool.append(block)
        self.pool_scores.append(np.einsum("ij,ij->j", block.astype(np.float64), block.astype(np.float64)))
        return len(self.pool) - 1

    @property
    def block_boundaries(self):
        return self._boundaries

    def correlation_block(self, block_index: int):
        return self.pool[self.pool_index[block_index]]

    def correlation(self, block_index: int) -> float:
        entry = self.pool_index[block_index]
        return POOL_CORRELATIONS[entry] if entry < len(POOL_CORRELATIONS) else 0.95

    def ld_diagonal(self):
        return np.ones(int(self._boundaries[-1]))

    def ld_scores(self):
        return np.concatenate([self.pool_scores[entry] for entry in self.pool_index])


def ar_filter_times(values: np.ndarray, correlation: float) -> np.ndarray:
    """R v for R_jk = rho^|j-k|: forward plus backward exponential filters minus v."""
    forward = lfilter([1.0], [1.0, -correlation], values)
    backward = lfilter([1.0], [1.0, -correlation], values[::-1])[::-1]
    return forward + backward - values


def ar_square_root_times(noise: np.ndarray, correlation: float) -> np.ndarray:
    """A draw with covariance R: the stationary AR(1) process driven by `noise`."""
    innovation = noise.copy()
    innovation[1:] *= np.sqrt(1.0 - correlation**2)
    return lfilter([1.0], [1.0, -correlation], innovation)


def simulate(variant_count: int, trait_count: int, rng: np.random.Generator):
    ld = CyclicToeplitzLD(variant_count)
    boundaries = ld.block_boundaries
    class_index = (rng.uniform(size=variant_count) < 0.02).astype(np.int64)
    statistics = []
    for _trait in range(trait_count):
        effects = np.zeros(variant_count)
        causal = rng.choice(variant_count, max(variant_count // 2000, 10), replace=False)
        effects[causal] = rng.standard_normal(causal.shape[0]) * np.where(class_index[causal] == 1, 2.0, 1.0)
        genetic_variance = 0.0
        correlated_effects = np.empty(variant_count)
        noise_draw = np.empty(variant_count)
        for block_index in range(boundaries.shape[0] - 1):
            block = slice(int(boundaries[block_index]), int(boundaries[block_index + 1]))
            correlation = ld.correlation(block_index)
            correlated_effects[block] = ar_filter_times(effects[block], correlation)
            genetic_variance += float(effects[block] @ correlated_effects[block])
            noise_draw[block] = ar_square_root_times(rng.standard_normal(block.stop - block.start), correlation)
        scale = np.sqrt(HERITABILITY / genetic_variance)
        noise_variance = 1.0 - HERITABILITY
        score = SAMPLE_COUNT * scale * correlated_effects + np.sqrt(SAMPLE_COUNT * noise_variance) * noise_draw
        statistics.append(
            ld_space_fit.QuantitativeTraitStatistics(
                sample_count=SAMPLE_COUNT, covariate_count=12, score=score, residual_sum_of_squares=float(SAMPLE_COUNT)
            )
        )
    sv_indicator = class_index.astype(np.float64)
    annotation = rng.standard_normal(variant_count)
    hypermodel = ld_space_fit.LDPriorHypermodel(
        annotation_design=np.column_stack([sv_indicator - sv_indicator.mean(), annotation - annotation.mean()]),
        annotation_prior_mean=np.zeros(2),
        annotation_prior_precision=np.array([1.0, 4.0]),
        log_variance_offset=np.log(rng.uniform(0.3, 1.0, size=variant_count)),
        variant_class_index=class_index,
        class_names=("snv", "sv"),
        shape_a=0.5,
        initial_shape_b=np.array([0.5, 0.4]),
        shape_b_pooling_variance=0.25,
    )
    return ld, statistics, hypermodel


def main() -> None:
    variant_count, scheme_name, pass_limit = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    trait_count = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    budget = detect_compute_budget()
    rng = np.random.default_rng(0)
    start = time.perf_counter()
    ld, statistics, hypermodel = simulate(variant_count, trait_count, rng)
    blocks = ld.block_boundaries.shape[0] - 1
    widths = np.diff(ld.block_boundaries)
    print(
        f"BENCH setup variants={variant_count} blocks={blocks} mean_width={widths.mean():.0f} max_width={widths.max()} "
        f"sum_width3={float(np.sum(widths.astype(np.float64) ** 3)):.3e} traits={trait_count} simulate_s={time.perf_counter() - start:.1f} "
        f"device={budget.describe()}",
        flush=True,
    )
    if pass_limit > 0:
        ld_space_fit._MAXIMUM_PASSES = pass_limit
    start = time.perf_counter()
    fits = ld_space_fit.fit_ld_space(ld, statistics, hypermodel, budget, scheme_name=scheme_name)
    seconds = time.perf_counter() - start
    peak_gigabytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    for fit in fits:
        print(
            f"BENCH result variants={variant_count} scheme={scheme_name} traits={trait_count} passes={fit.passes} converged={fit.converged} "
            f"fit_s={seconds:.1f} per_pass_s={seconds / max(fit.passes, 1):.1f} peak_rss_gb={peak_gigabytes:.1f} "
            f"sigma2={1 / fit.likelihood_precision:.4f} level={fit.log_variance_level:.3f} b={np.round(fit.shape_b, 3)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
