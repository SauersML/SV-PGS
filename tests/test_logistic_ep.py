"""Separation of a binary trait's labels by its covariates."""

from __future__ import annotations

import numpy as np
import pytest

from sv_pgs import logistic_ep


def test_separating_covariates_are_refused() -> None:
    generator = np.random.default_rng(7)
    sample_count = 50
    score = generator.standard_normal(sample_count)
    complete = np.column_stack([np.ones(sample_count), score])
    with pytest.raises(logistic_ep.SeparatedLabels):
        logistic_ep.assert_no_separation(complete, (score > 0.0).astype(np.float64))
    # Quasi-separation: every sample with the indicator is a case, as with a sex-limited disease.
    indicator = (np.arange(sample_count) % 2).astype(np.float64)
    labels = np.where(indicator == 1.0, 1.0, (generator.random(sample_count) < 0.3).astype(np.float64))
    with pytest.raises(logistic_ep.SeparatedLabels):
        logistic_ep.assert_no_separation(np.column_stack([np.ones(sample_count), indicator]), labels)
    mixed = (generator.random(sample_count) < 0.4).astype(np.float64)
    logistic_ep.assert_no_separation(complete, mixed)
