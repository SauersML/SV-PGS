from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from sv_pgs._typing import F32Array, I32Array
from sv_pgs.config import VariantClass


NESTED_PATH_DELIMITER = ">"

# Class-membership weights are mixture proportions: they must sum to one, up to
# floating-point slack from the collapse/normalization paths that produce them.
CLASS_MEMBERSHIP_SUM_TOLERANCE = 1e-6

_TRUE_BOOLEAN_TOKENS = frozenset({"true", "t", "yes", "1"})
_FALSE_BOOLEAN_TOKENS = frozenset({"false", "f", "no", "0"})


def _parse_boolean(value: Any, field_name: str) -> bool:
    """Parse a strict boolean.

    Accepts real booleans, the integers/floats 0 and 1, and the case-insensitive
    tokens "true"/"t"/"yes"/"1" and "false"/"f"/"no"/"0". Anything else raises.
    Generic truthiness is never used: the strings "false" and "0" are False, not
    True, so binary annotations ingested from TSV/CSV cannot silently invert.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric_value = float(value)
        if numeric_value == 0.0:
            return False
        if numeric_value == 1.0:
            return True
        raise ValueError(field_name + " must be a boolean, 0, or 1; got " + repr(value) + ".")
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_BOOLEAN_TOKENS:
            return True
        if token in _FALSE_BOOLEAN_TOKENS:
            return False
        raise ValueError(field_name + " could not be parsed as a boolean: " + repr(value) + ".")
    raise ValueError(field_name + " could not be parsed as a boolean: " + repr(value) + ".")


def _validate_class_membership(
    prior_class_members: Sequence[VariantClass],
    prior_class_membership: Sequence[float],
) -> None:
    if len(prior_class_members) != len(prior_class_membership):
        raise ValueError("prior_class_members and prior_class_membership must have the same length.")
    if not prior_class_members:
        raise ValueError("prior_class_members cannot be empty when prior_class_membership is provided.")
    duplicate_members = sorted(
        {
            _class_key(member_class)
            for member_index, member_class in enumerate(prior_class_members)
            if _class_key(member_class) in {_class_key(other) for other in prior_class_members[:member_index]}
        }
    )
    if duplicate_members:
        raise ValueError("prior_class_members cannot contain duplicates: " + ", ".join(duplicate_members))
    membership_sum = 0.0
    for member_weight in prior_class_membership:
        weight_value = float(member_weight)
        if not np.isfinite(weight_value):
            raise ValueError("prior_class_membership weights must be finite.")
        if weight_value < 0.0:
            raise ValueError("prior_class_membership weights must be non-negative.")
        membership_sum += weight_value
    if membership_sum <= 0.0:
        raise ValueError("prior_class_membership weights must sum to a positive value.")
    if abs(membership_sum - 1.0) > CLASS_MEMBERSHIP_SUM_TOLERANCE:
        raise ValueError(
            "prior_class_membership weights are mixture proportions and must sum to 1.0"
            f" (within {CLASS_MEMBERSHIP_SUM_TOLERANCE:g}); got {membership_sum!r}."
        )


def _class_key(member_class: Any) -> str:
    return member_class.value if isinstance(member_class, VariantClass) else str(member_class)


def _validate_prior_feature_name(feature_name: str, field_name: str) -> None:
    if not feature_name:
        raise ValueError(field_name + " keys cannot be empty.")
    if "::" in feature_name:
        raise ValueError(field_name + " keys cannot contain '::'.")


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

    ``prior_class_members`` / ``prior_class_membership`` are a soft assignment of
    the variant to variant classes. The weights are mixture proportions: they must
    be finite, non-negative, and sum to 1.0 (within ``CLASS_MEMBERSHIP_SUM_TOLERANCE``).
    They are validated, never auto-normalized, and a class may appear at most once.
    When both are omitted the variant is assigned to its own ``variant_class`` with
    weight 1.0.

    Boolean fields (``is_repeat``, ``is_copy_number``, ``prior_binary_features``
    values) are parsed strictly by :func:`_parse_boolean`, not by truthiness.
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
        # Fast path: at AoU scale we construct ~3.4M VariantRecord objects with
        # no per-variant annotations (no variant_metadata file supplied), so
        # every feature dict is the default-empty one. Skipping the
        # normalize/validate work in that case turns a ~60s post-init pass into
        # a ~5s one for the default --variants snp+sv pipeline.
        if not (
            self.prior_binary_features
            or self.prior_continuous_features
            or self.prior_categorical_features
            or self.prior_membership_features
            or self.prior_nested_features
            or self.prior_nested_membership_features
        ):
            if not self.prior_class_members and not self.prior_class_membership:
                self.prior_class_members = (self.variant_class,)
                self.prior_class_membership = (1.0,)
            else:
                _validate_class_membership(self.prior_class_members, self.prior_class_membership)
            self.is_repeat = _parse_boolean(self.is_repeat, "is_repeat")
            self.is_copy_number = _parse_boolean(self.is_copy_number, "is_copy_number")
            return

        self.is_repeat = _parse_boolean(self.is_repeat, "is_repeat")
        self.is_copy_number = _parse_boolean(self.is_copy_number, "is_copy_number")
        self.prior_binary_features = {
            str(feature_name): _parse_boolean(feature_value, "prior_binary_features")
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
        normalized_categorical_features: dict[str, str] = {
            str(feature_name): str(feature_value)
            for feature_name, feature_value in self.prior_categorical_features.items()
        }
        self.prior_categorical_features = normalized_categorical_features
        for feature_name, categorical_value in normalized_categorical_features.items():
            _validate_prior_feature_name(feature_name, "prior_categorical_features")
            if not categorical_value:
                raise ValueError("prior_categorical_features values cannot be empty.")
            if "::" in categorical_value:
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
        _validate_class_membership(self.prior_class_members, self.prior_class_membership)


@dataclass(slots=True)
class VariantStatistics:
    """Pre-computed per-variant statistics from a single streaming pass."""
    means: F32Array
    scales: F32Array
    allele_frequencies: F32Array
    support_counts: I32Array  # non-zero dosage count per variant


@dataclass(slots=True)
class PreparedArrays:
    covariates: F32Array
    targets: F32Array
    means: F32Array
    scales: F32Array
    support_counts: I32Array


@dataclass(slots=True)
class TieGroup:
    """A group of variants with identical (or exactly negated) genotype columns.

    representative_index: the variant we keep in the reduced model
    member_indices: all variants in the group (including the representative)
    signs: +1 if a member's column matches the representative, -1 if negated
    """
    representative_index: int
    member_indices: I32Array
    signs: F32Array


@dataclass(slots=True)
class TieMap:
    """Maps between the full variant set and the de-duplicated (reduced) set.

    Many variants can have identical genotype patterns across all samples
    (especially in LD or when multiple callers detect the same event).
    The tie map lets us fit the model on unique columns only, then expand
    the results back to all variants.

    kept_indices: which original variants were kept as representatives
    original_to_reduced: for each original variant, its reduced-space index (-1 if not active)
    reduced_to_group: for each reduced variant, the full group of tied members.
        Empty when kept_indices/original_to_reduced encode a compact no-ties map.
    """
    kept_indices: I32Array
    original_to_reduced: I32Array
    reduced_to_group: list[TieGroup]

    def expand_coefficients(
        self,
        reduced_beta: F32Array,
        group_weights: Sequence[F32Array],
    ) -> F32Array:
        """Distribute each group's single fitted effect back to all its members.

        Each member gets: beta_member = beta_group * weight_member * sign_member
        where weights are proportional to prior variance and signs handle negation.
        """
        reduced_beta_array = np.asarray(reduced_beta, dtype=np.float32)
        if not self.reduced_to_group:
            expanded_coefficients = np.zeros(self.original_to_reduced.shape[0], dtype=np.float32)
            if self.kept_indices.shape[0] != reduced_beta_array.shape[0]:
                raise ValueError("reduced_beta must align with compact tie-map representatives.")
            # self.kept_indices is already int32 (dataclass invariant via callers);
            # use it directly to avoid an unnecessary np.asarray copy.
            expanded_coefficients[self.kept_indices] = reduced_beta_array
            return expanded_coefficients
        expanded_coefficients = np.zeros(self.original_to_reduced.shape[0], dtype=np.float32)
        for reduced_index, tie_group in enumerate(self.reduced_to_group):
            group_weight_vector = np.asarray(group_weights[reduced_index], dtype=np.float32)
            expanded_coefficients[tie_group.member_indices] = (
                reduced_beta_array[reduced_index] * group_weight_vector * tie_group.signs
            )
        return expanded_coefficients


def _coerce_variant_class(value: Any) -> VariantClass:
    if isinstance(value, VariantClass):
        return value
    return VariantClass(str(value))


def normalize_variant_record(record: VariantRecord | dict[str, Any]) -> VariantRecord:
    if isinstance(record, VariantRecord):
        return record
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
        is_repeat=_parse_boolean(record.get("is_repeat", False), "is_repeat"),
        is_copy_number=_parse_boolean(record.get("is_copy_number", False), "is_copy_number"),
        prior_binary_features={
            str(feature_name): _parse_boolean(feature_value, "prior_binary_features")
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
