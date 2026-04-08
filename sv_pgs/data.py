from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from sv_pgs.config import VariantClass


NESTED_PATH_DELIMITER = ">"
RESERVED_PRIOR_CONTINUOUS_FEATURE_NAMES = frozenset(
    {
        "log_length",
        "logit_allele_frequency",
        "quality",
        "log_training_support",
    }
)
RESERVED_PRIOR_FACTOR_FEATURE_NAMES = frozenset(
    {
        "repeat_indicator",
        "copy_number_indicator",
        "maf_bucket",
    }
)
RESERVED_PRIOR_FEATURE_NAMES = frozenset(
    set(RESERVED_PRIOR_CONTINUOUS_FEATURE_NAMES) | set(RESERVED_PRIOR_FACTOR_FEATURE_NAMES)
)


def _validate_prior_feature_name(feature_name: str, field_name: str) -> None:
    if not feature_name:
        raise ValueError(field_name + " keys cannot be empty.")
    if "::" in feature_name:
        raise ValueError(field_name + " keys cannot contain '::'.")
    if feature_name in RESERVED_PRIOR_FEATURE_NAMES:
        raise ValueError(
            field_name + " keys cannot override built-in features: " + feature_name
        )


def _normalize_nested_path(path_value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(path_value, str):
        path_parts = tuple(path_value.split(NESTED_PATH_DELIMITER))
    else:
        path_parts = tuple(str(path_part) for path_part in path_value)
    if not path_parts:
        raise ValueError(field_name + " paths cannot be empty.")
    normalized_parts: list[str] = []
    for path_part in path_parts:
        part_value = str(path_part).strip()
        if not part_value:
            raise ValueError(field_name + " paths cannot contain empty levels.")
        if "::" in part_value:
            raise ValueError(field_name + " paths cannot contain '::'.")
        if NESTED_PATH_DELIMITER in part_value:
            raise ValueError(field_name + f" paths cannot contain '{NESTED_PATH_DELIMITER}'.")
        normalized_parts.append(part_value)
    return tuple(normalized_parts)


def _encode_nested_path(path_value: Any, field_name: str) -> str:
    return NESTED_PATH_DELIMITER.join(_normalize_nested_path(path_value, field_name))


@dataclass(slots=True)
class VariantRecord:
    """Metadata for a single genetic variant.

    Carries all the non-genotype information about a variant that the model
    uses to set its prior: what type of variant it is, how long it is,
    how common it is, whether it overlaps a repeat region, etc.
    """
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
    prior_binary_features: dict[str, bool] = field(default_factory=dict)
    prior_continuous_features: dict[str, float] = field(default_factory=dict)
    prior_categorical_features: dict[str, str] = field(default_factory=dict)
    prior_membership_features: dict[str, dict[str, float]] = field(default_factory=dict)
    prior_nested_features: dict[str, tuple[str, ...]] = field(default_factory=dict)
    prior_nested_membership_features: dict[str, dict[str, float]] = field(default_factory=dict)
    prior_class_members: tuple[VariantClass, ...] = ()
    prior_class_membership: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        self.prior_binary_features = {
            str(feature_name): bool(feature_value)
            for feature_name, feature_value in self.prior_binary_features.items()
        }
        for feature_name in self.prior_binary_features:
            _validate_prior_feature_name(feature_name, "prior_binary_features")
        self.prior_continuous_features = {
            str(feature_name): float(feature_value)
            for feature_name, feature_value in self.prior_continuous_features.items()
        }
        for feature_name, feature_value in self.prior_continuous_features.items():
            _validate_prior_feature_name(feature_name, "prior_continuous_features")
            if not np.isfinite(feature_value):
                raise ValueError("prior_continuous_features values must be finite.")
        self.prior_categorical_features = {
            str(feature_name): str(feature_value)
            for feature_name, feature_value in self.prior_categorical_features.items()
        }
        for feature_name, feature_value in self.prior_categorical_features.items():
            _validate_prior_feature_name(feature_name, "prior_categorical_features")
            if not feature_value:
                raise ValueError("prior_categorical_features values cannot be empty.")
            if "::" in feature_value:
                raise ValueError("prior_categorical_features values cannot contain '::'.")
        self.prior_membership_features = {
            str(feature_name): {
                str(level_name): float(level_weight)
                for level_name, level_weight in feature_memberships.items()
            }
            for feature_name, feature_memberships in self.prior_membership_features.items()
        }
        for feature_name, feature_memberships in self.prior_membership_features.items():
            _validate_prior_feature_name(feature_name, "prior_membership_features")
            for level_name, level_weight in feature_memberships.items():
                if not level_name:
                    raise ValueError("prior_membership_features level names cannot be empty.")
                if "::" in level_name:
                    raise ValueError("prior_membership_features level names cannot contain '::'.")
                if not np.isfinite(level_weight):
                    raise ValueError("prior_membership_features weights must be finite.")
                if level_weight < 0.0:
                    raise ValueError("prior_membership_features weights must be non-negative.")
        self.prior_nested_features = {
            str(feature_name): _normalize_nested_path(feature_value, "prior_nested_features")
            for feature_name, feature_value in self.prior_nested_features.items()
        }
        for feature_name in self.prior_nested_features:
            _validate_prior_feature_name(feature_name, "prior_nested_features")
        self.prior_nested_membership_features = {
            str(feature_name): {
                _encode_nested_path(path_name, "prior_nested_membership_features"): float(path_weight)
                for path_name, path_weight in feature_memberships.items()
            }
            for feature_name, feature_memberships in self.prior_nested_membership_features.items()
        }
        for feature_name, feature_memberships in self.prior_nested_membership_features.items():
            _validate_prior_feature_name(feature_name, "prior_nested_membership_features")
            for path_weight in feature_memberships.values():
                if not np.isfinite(path_weight):
                    raise ValueError("prior_nested_membership_features weights must be finite.")
                if path_weight < 0.0:
                    raise ValueError("prior_nested_membership_features weights must be non-negative.")
        custom_feature_name_sets = {
            "prior_binary_features": set(self.prior_binary_features),
            "prior_continuous_features": set(self.prior_continuous_features),
            "prior_categorical_features": set(self.prior_categorical_features),
            "prior_membership_features": set(self.prior_membership_features),
            "prior_nested_features": set(self.prior_nested_features),
            "prior_nested_membership_features": set(self.prior_nested_membership_features),
        }
        for left_field_name, left_feature_names in custom_feature_name_sets.items():
            for right_field_name, right_feature_names in custom_feature_name_sets.items():
                if left_field_name >= right_field_name:
                    continue
                overlapping_feature_names = sorted(left_feature_names & right_feature_names)
                if overlapping_feature_names:
                    raise ValueError(
                        "Custom prior feature names must be unique across annotation families: "
                        + ", ".join(overlapping_feature_names)
                    )
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
    """Pre-computed per-variant statistics from a single streaming pass."""
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
    support_counts: np.ndarray


@dataclass(slots=True)
class TieGroup:
    """A group of variants with identical (or exactly negated) genotype columns.

    representative_index: the variant we keep in the reduced model
    member_indices: all variants in the group (including the representative)
    signs: +1 if a member's column matches the representative, -1 if negated
    """
    representative_index: int
    member_indices: np.ndarray
    signs: np.ndarray


@dataclass(slots=True)
class TieMap:
    """Maps between the full variant set and the de-duplicated (reduced) set.

    Many variants can have identical genotype patterns across all samples
    (especially in LD or when multiple callers detect the same event).
    The tie map lets us fit the model on unique columns only, then expand
    the results back to all variants.

    kept_indices: which original variants were kept as representatives
    original_to_reduced: for each original variant, its reduced-space index (-1 if not active)
    reduced_to_group: for each reduced variant, the full group of tied members
    """
    kept_indices: np.ndarray
    original_to_reduced: np.ndarray
    reduced_to_group: list[TieGroup]

    def expand_coefficients(
        self,
        reduced_beta: np.ndarray,
        group_weights: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Distribute each group's single fitted effect back to all its members.

        Each member gets: beta_member = beta_group * weight_member * sign_member
        where weights are proportional to prior variance and signs handle negation.
        """
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


def normalize_variant_record(record: VariantRecord | dict[str, Any]) -> VariantRecord:
    if isinstance(record, VariantRecord):
        return VariantRecord(
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
            prior_binary_features=dict(record.prior_binary_features),
            prior_continuous_features=dict(record.prior_continuous_features),
            prior_categorical_features=dict(record.prior_categorical_features),
            prior_membership_features={
                feature_name: dict(feature_memberships)
                for feature_name, feature_memberships in record.prior_membership_features.items()
            },
            prior_nested_features=dict(record.prior_nested_features),
            prior_nested_membership_features={
                feature_name: dict(feature_memberships)
                for feature_name, feature_memberships in record.prior_nested_membership_features.items()
            },
            prior_class_members=record.prior_class_members,
            prior_class_membership=record.prior_class_membership,
        )
    return VariantRecord(
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
        prior_binary_features={
            str(feature_name): bool(feature_value)
            for feature_name, feature_value in record.get("prior_binary_features", {}).items()
        },
        prior_continuous_features={
            str(feature_name): float(feature_value)
            for feature_name, feature_value in record.get("prior_continuous_features", {}).items()
        },
        prior_categorical_features={
            str(feature_name): str(feature_value)
            for feature_name, feature_value in record.get("prior_categorical_features", {}).items()
        },
        prior_membership_features={
            str(feature_name): {
                str(level_name): float(level_weight)
                for level_name, level_weight in feature_memberships.items()
            }
            for feature_name, feature_memberships in record.get("prior_membership_features", {}).items()
        },
        prior_nested_features={
            str(feature_name): _normalize_nested_path(feature_value, "prior_nested_features")
            for feature_name, feature_value in record.get("prior_nested_features", {}).items()
        },
        prior_nested_membership_features={
            str(feature_name): {
                _encode_nested_path(path_name, "prior_nested_membership_features"): float(path_weight)
                for path_name, path_weight in feature_memberships.items()
            }
            for feature_name, feature_memberships in record.get("prior_nested_membership_features", {}).items()
        },
        prior_class_members=tuple(
            VariantClass(member_value) for member_value in record.get("prior_class_members", ())
        ),
        prior_class_membership=tuple(float(member_weight) for member_weight in record.get("prior_class_membership", ())),
    )


def normalize_variant_records(records: Sequence[VariantRecord | dict[str, Any]]) -> list[VariantRecord]:
    return [normalize_variant_record(record) for record in records]
