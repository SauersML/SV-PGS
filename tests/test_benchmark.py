from __future__ import annotations

import numpy as np

from sv_pgs.benchmark import _top_tail_enrichment
from sv_pgs.config import TraitType


def test_quantitative_top_tail_enrichment_uses_standardized_shift():
    scores = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    targets = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)

    enrichment = _top_tail_enrichment(
        scores=scores,
        targets=targets,
        fraction=0.2,
        trait_type=TraitType.QUANTITATIVE,
    )

    expected_value = (np.mean([2.0]) - np.mean(targets)) / np.std(targets)
    assert np.isclose(enrichment, expected_value)


def test_binary_top_tail_enrichment_remains_prevalence_ratio():
    scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    targets = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)

    enrichment = _top_tail_enrichment(
        scores=scores,
        targets=targets,
        fraction=0.25,
        trait_type=TraitType.BINARY,
    )

    assert np.isclose(enrichment, 2.0)
