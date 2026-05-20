"""Regression test: SVI beta_variance_state must NOT carry a stale EMA of
block-local solves on no-refresh epochs.

When ``refresh_beta_variance=False`` the per-block solve only sees its own
block's prior, so blending ``(1-step)*stale + step*fresh`` produces a biased
estimate that over-shrinks sigma_e^2 through the leverage correction. The
fix snaps ``beta_variance_state`` to the current ``reduced_prior_variances``
on no-refresh epochs (i.e. variance equals the prior in expectation under
the variational prior) and only blends on real refresh epochs.
"""

from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.mixture_inference import (
    VariationalFitCheckpoint,
    fit_variational_em,
)
import sv_pgs.mixture_inference as mixture_inference
from sv_pgs.preprocessing import build_tie_map

from tests.conftest import make_variant_records


@pytest.fixture
def random_generator() -> np.random.Generator:
    return np.random.default_rng(0xBEEF)


def test_beta_variance_state_snaps_to_prior_on_no_refresh_epoch(
    random_generator: np.random.Generator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # p < n so the small-n quantitative override does NOT force refresh.
    sample_count, variant_count = 64, 12
    genotype_matrix = random_generator.standard_normal(
        (sample_count, variant_count)
    ).astype(np.float32)
    covariate_matrix = np.column_stack(
        [
            np.ones(sample_count, dtype=np.float32),
            random_generator.standard_normal(sample_count).astype(np.float32),
        ]
    )
    true_coefficients = np.zeros(variant_count, dtype=np.float32)
    true_coefficients[:2] = np.array([0.9, -0.7], dtype=np.float32)
    target_vector = (
        genotype_matrix @ true_coefficients
        + 0.3 * random_generator.standard_normal(sample_count).astype(np.float32)
    )
    records = make_variant_records(variant_count)
    # max_outer_iterations=2 with beta_variance_update_interval=2 means:
    #   epoch 0: (0+1) % 2 == 1  -> refresh_beta_variance = False
    #   epoch 1: (1+1) % 2 == 0  -> refresh_beta_variance = True
    # final_posterior_refinement=False so force_final_refresh path does not
    # short-circuit epoch 0.
    config = ModelConfig(
        trait_type=TraitType.QUANTITATIVE,
        max_outer_iterations=2,
        update_hyperparameters=False,
        stochastic_variational_updates=True,
        stochastic_min_variant_count=1,
        stochastic_variant_batch_size=4,
        beta_variance_update_interval=2,
        final_posterior_refinement=False,
    )

    # Force a checkpoint callback to fire after every stochastic block so we
    # can capture beta_variance_state at the END of epoch 0's last block —
    # i.e. AFTER all block-local updates but BEFORE the post-epoch line
    # that recomputes beta_variance_state from second moments.
    monkeypatch.setattr(
        mixture_inference,
        "_should_checkpoint_stochastic_block",
        lambda **_: True,
    )

    captured: list[VariationalFitCheckpoint] = []

    def _capture(ckpt: VariationalFitCheckpoint) -> None:
        captured.append(ckpt)

    result = fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=config,
        tie_map=build_tie_map(genotype_matrix, records, config),
        checkpoint_callback=_capture,
    )

    # Collect checkpoints belonging to epoch 0 (completed_iterations == 0
    # means the epoch has not yet finalized — i.e. mid-epoch snapshots from
    # the stochastic block loop of epoch 0).
    epoch0_checkpoints = [
        ckpt
        for ckpt in captured
        if ckpt.completed_iterations == 0 and ckpt.beta_variance_state is not None
    ]
    assert epoch0_checkpoints, "expected at least one mid-epoch checkpoint from epoch 0"

    last_epoch0 = epoch0_checkpoints[-1]
    assert last_epoch0.beta_variance_state is not None
    assert last_epoch0.epoch_reduced_prior_variances is not None

    prior_variances = np.asarray(
        last_epoch0.epoch_reduced_prior_variances, dtype=np.float64
    )
    observed_variance = np.asarray(
        last_epoch0.beta_variance_state, dtype=np.float64
    )

    # The fix: on a no-refresh epoch, beta_variance_state must equal the
    # current reduced_prior_variances (floored at 1e-8), NOT a stale EMA
    # of block-local solves.
    expected = np.maximum(prior_variances, 1e-8)
    np.testing.assert_allclose(observed_variance, expected, rtol=1e-10, atol=1e-12)

    # And sigma_e^2 must stay in a reasonable range — not collapsed to
    # near-zero by a fake leverage correction driven by biased variances.
    assert result.sigma_error2 > 1e-3, (
        f"sigma_e^2 collapsed unreasonably: {result.sigma_error2}"
    )
    assert result.sigma_error2 < 100.0, (
        f"sigma_e^2 blew up unreasonably: {result.sigma_error2}"
    )
