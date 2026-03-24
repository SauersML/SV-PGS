from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from sv_pgs.config import VariantClass


@dataclass(slots=True)
class VariantRecord:
    variant_id: str
    variant_class: VariantClass
    length_bin: str
    chromosome: str
    position: int
    quality: float = 1.0
    cluster_id: str | None = None


@dataclass(slots=True)
class PreparedArrays:
    genotypes: np.ndarray
    covariates: np.ndarray
    targets: np.ndarray
    means: np.ndarray
    scales: np.ndarray


@dataclass(slots=True)
class TieGroup:
    representative_index: int
    member_indices: np.ndarray
    signs: np.ndarray


@dataclass(slots=True)
class TieMap:
    kept_indices: np.ndarray
    original_to_reduced: np.ndarray
    reduced_to_group: list[TieGroup]

    def expand_coefficients(self, reduced_beta: np.ndarray) -> np.ndarray:
        full = np.zeros(self.original_to_reduced.shape[0], dtype=np.float32)
        for reduced_index, group in enumerate(self.reduced_to_group):
            full[group.member_indices] = reduced_beta[reduced_index] * group.signs
        return full


@dataclass(slots=True)
class GraphEdges:
    src: np.ndarray
    dst: np.ndarray
    sign: np.ndarray
    weight: np.ndarray
    block_ids: np.ndarray


def _coerce_variant_class(value: Any) -> VariantClass:
    if isinstance(value, VariantClass):
        return value
    return VariantClass(str(value))


def normalize_variant_records(records: Sequence[VariantRecord | dict[str, Any]]) -> list[VariantRecord]:
    normalized: list[VariantRecord] = []
    for record in records:
        if isinstance(record, VariantRecord):
            normalized.append(record)
            continue
        normalized.append(
            VariantRecord(
                variant_id=str(record["variant_id"]),
                variant_class=_coerce_variant_class(record["variant_class"]),
                length_bin=str(record.get("length_bin", "na")),
                chromosome=str(record["chromosome"]),
                position=int(record["position"]),
                quality=float(record.get("quality", 1.0)),
                cluster_id=None if record.get("cluster_id") in (None, "") else str(record["cluster_id"]),
            )
        )
    return normalized
