from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from sv_pgs.config import VariantClass


@dataclass(slots=True)
class VariantRecord:
    variant_id: str
    variant_class: VariantClass
    chromosome: str
    position: int
    length: float = 1.0
    allele_frequency: float = 0.01
    quality: float = 1.0
    is_repeat: bool = False
    is_copy_number: bool = False
    prior_class_members: tuple[VariantClass, ...] = ()
    prior_class_membership: tuple[float, ...] = ()


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
        expanded_coefficients = np.zeros(self.original_to_reduced.shape[0], dtype=np.float32)
        for reduced_index, tie_group in enumerate(self.reduced_to_group):
            group_size = max(int(tie_group.member_indices.shape[0]), 1)
            neutral_coefficient = reduced_beta[reduced_index] / float(group_size)
            expanded_coefficients[tie_group.member_indices] = neutral_coefficient * tie_group.signs
        return expanded_coefficients


def _coerce_variant_class(value: Any) -> VariantClass:
    if isinstance(value, VariantClass):
        return value
    return VariantClass(str(value))


def normalize_variant_records(records: Sequence[VariantRecord | dict[str, Any]]) -> list[VariantRecord]:
    normalized_records: list[VariantRecord] = []
    for record in records:
        if isinstance(record, VariantRecord):
            normalized_records.append(record)
            continue
        normalized_records.append(
            VariantRecord(
                variant_id=str(record["variant_id"]),
                variant_class=_coerce_variant_class(record["variant_class"]),
                chromosome=str(record["chromosome"]),
                position=int(record["position"]),
                length=float(record.get("length", 1.0)),
                allele_frequency=float(record.get("allele_frequency", 0.01)),
                quality=float(record.get("quality", 1.0)),
                is_repeat=bool(record.get("is_repeat", False)),
                is_copy_number=bool(record.get("is_copy_number", False)),
                prior_class_members=tuple(
                    VariantClass(member_value) for member_value in record.get("prior_class_members", ())
                ),
                prior_class_membership=tuple(float(member_weight) for member_weight in record.get("prior_class_membership", ())),
            )
        )
    return normalized_records
