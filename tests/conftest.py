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
) -> list[VariantRecord]:
    return [
        VariantRecord(
            variant_id="variant_" + str(variant_index),
            variant_class=variant_class,
            chromosome="chr1",
            position=variant_index * 100,
            length=50.0 + variant_index,
            allele_frequency=min(0.45, 0.05 + 0.01 * variant_index),
            quality=0.95,
            is_repeat=False,
            is_copy_number=variant_class in {
                VariantClass.DUPLICATION_SHORT,
                VariantClass.DUPLICATION_LONG,
            },
        )
        for variant_index in range(variant_count)
    ]
