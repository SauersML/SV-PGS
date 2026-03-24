from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.config import VariantClass
from sv_pgs.data import VariantRecord


@pytest.fixture
def random_generator() -> np.random.Generator:
    return np.random.default_rng(42)


def make_variant_records(
    variant_count: int,
    variant_class: VariantClass = VariantClass.SNV,
    chromosome: str = "chr1",
) -> list[VariantRecord]:
    return [
        VariantRecord(
            variant_id="variant_" + str(variant_index),
            variant_class=variant_class,
            chromosome=chromosome,
            position=variant_index * 100,
            length=1.0,
            allele_frequency=0.1,
            quality=1.0,
        )
        for variant_index in range(variant_count)
    ]
