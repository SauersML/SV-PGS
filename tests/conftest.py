from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.config import VariantClass
from sv_pgs.data import GraphEdges, VariantRecord


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
            length_bin="short",
            chromosome="chr1",
            position=variant_index * 100,
            quality=1.0,
        )
        for variant_index in range(variant_count)
    ]


def empty_graph(variant_count: int) -> GraphEdges:
    return GraphEdges(
        src=np.array([], dtype=np.int32),
        dst=np.array([], dtype=np.int32),
        sign=np.array([], dtype=np.float32),
        weight=np.array([], dtype=np.float32),
        block_ids=np.arange(variant_count, dtype=np.int32),
    )
