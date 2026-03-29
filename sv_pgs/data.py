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
    training_support: int | None = None
    is_repeat: bool = False
    is_copy_number: bool = False
    prior_class_members: tuple[VariantClass, ...] = ()
    prior_class_membership: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.prior_class_members and not self.prior_class_membership:
            self.prior_class_members = (self.variant_class,)
            self.prior_class_membership = (1.0,)
            return
        if len(self.prior_class_members) != len(self.prior_class_membership):
            raise ValueError("prior_class_members and prior_class_membership must have the same length.")
        if not self.prior_class_members:
            raise ValueError("prior_class_members cannot be empty when prior_class_membership is provided.")


@dataclass(slots=True)
class VariantStatistics:
    """Pre-computed per-variant statistics from streaming over raw genotypes (2 passes)."""
    means: np.ndarray
    scales: np.ndarray
    allele_frequencies: np.ndarray
    support_counts: np.ndarray  # int32, non-zero dosage count per variant


@dataclass(slots=True)
class PreparedArrays:
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

    def expand_coefficients(
        self,
        reduced_beta: np.ndarray,
        group_weights: Sequence[np.ndarray],
    ) -> np.ndarray:
        expanded_coefficients = np.zeros(self.original_to_reduced.shape[0], dtype=np.float32)
        for reduced_index, tie_group in enumerate(self.reduced_to_group):
            group_weight_vector = np.asarray(group_weights[reduced_index], dtype=np.float32)
            expanded_coefficients[tie_group.member_indices] = (
                reduced_beta[reduced_index] * group_weight_vector * tie_group.signs
            )
        return expanded_coefficients


def _coerce_variant_class(value: Any) -> VariantClass:
    if isinstance(value, VariantClass):
        return value
    return VariantClass(str(value))


def normalize_variant_records(records: Sequence[VariantRecord | dict[str, Any]]) -> list[VariantRecord]:
    normalized_records: list[VariantRecord] = []
    for record in records:
        if isinstance(record, VariantRecord):
            normalized_records.append(
                VariantRecord(
                    variant_id=record.variant_id,
                    variant_class=record.variant_class,
                    chromosome=record.chromosome,
                    position=record.position,
                    length=record.length,
                    allele_frequency=record.allele_frequency,
                    quality=record.quality,
                    training_support=record.training_support,
                    is_repeat=record.is_repeat,
                    is_copy_number=record.is_copy_number,
                    prior_class_members=record.prior_class_members,
                    prior_class_membership=record.prior_class_membership,
                )
            )
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
                training_support=(
                    None
                    if record.get("training_support") is None
                    else int(record["training_support"])
                ),
                is_repeat=bool(record.get("is_repeat", False)),
                is_copy_number=bool(record.get("is_copy_number", False)),
                prior_class_members=tuple(
                    VariantClass(member_value) for member_value in record.get("prior_class_members", ())
                ),
                prior_class_membership=tuple(float(member_weight) for member_weight in record.get("prior_class_membership", ())),
            )
        )
    return normalized_records
