"""The prior design: per-variant features of the effect prior's scale.

Built once from the reduced (tie-collapsed) variant records: class membership,
continuous annotations as spline bases, factor and nested-factor annotations,
and the baseline scales, prior variances and hyperprior penalty derived from
the scale-model coefficients.
"""
from __future__ import annotations

from dataclasses import dataclass, replace as dataclass_replace
from typing import Any, Iterable, Mapping, Sequence, cast

import numpy as np

from sv_pgs._typing import F64Array, NDArray
from sv_pgs.config import ModelConfig, VariantClass
from sv_pgs.data import NESTED_PATH_DELIMITER, TieMap, VariantRecord


@dataclass(slots=True)
class PriorDesign:
    """Describes what we know about each variant *before* seeing the outcome data.

    Each variant's "allowed effect size range" depends on its metadata:
    variant type (SNV vs deletion vs duplication ...), length, whether it
    overlaps a repeat region, etc.  This dataclass holds the matrices that
    encode those relationships so the model can learn how metadata
    predicts effect magnitude.
    """
    design_matrix: NDArray           # each row = one variant's metadata features
    feature_names: list[str]            # human-readable names for each feature column
    feature_specs: tuple[ScaleModelFeatureSpec, ...]  # compiled feature descriptors for fast reuse
    class_membership_matrix: NDArray # which variant class(es) each variant belongs to
    inverse_class_lookup: dict[int, VariantClass]  # column index -> VariantClass enum


@dataclass(slots=True)
class ScaleModelFeatureSpec:
    kind: str
    center_value: float | None = None
    rms_scale: float | None = None
    variant_class: VariantClass | None = None
    source_name: str | None = None
    level_name: str | None = None
    nested_depth: int | None = None
    basis_index: int | None = None
    basis_kind: str | None = None
    standardize_mean: float | None = None
    standardize_scale: float | None = None
    knot_values: tuple[float, ...] = ()


@dataclass(slots=True)
class _PriorAnnotationTables:
    continuous_values_by_source: dict[str, NDArray]
    factor_weights_by_source: dict[str, dict[str, NDArray]]
    nested_weights_by_source: dict[str, dict[int, dict[str, NDArray]]]


@dataclass(slots=True)
class _PriorAnnotationFeatureNames:
    continuous: set[str]
    binary: set[str]
    categorical: set[str]
    membership: set[str]
    nested: set[str]
    nested_membership: set[str]


# Build the metadata design matrix for the prior scale model.
#
# The scale hypermodel is schema-driven:
#   - class effects
#   - pooled categorical / multi-membership effects
#   - nested categorical node effects
#   - smooth bases for continuous features
#   - class-varying interactions for factor and smooth terms
def _build_prior_design(records: Sequence[VariantRecord]) -> PriorDesign:
    unique_classes = sorted(
        {
            prior_class
            for record in records
            for prior_class in record.prior_class_members
        },
        key=lambda variant_class: variant_class.value,
    )
    class_lookup = {
        variant_class: class_index
        for class_index, variant_class in enumerate(unique_classes)
    }
    inverse_class_lookup = {
        class_index: variant_class
        for variant_class, class_index in class_lookup.items()
    }
    class_membership_matrix = np.zeros((len(records), len(unique_classes)), dtype=np.float64)
    for record_index, record in enumerate(records):
        for prior_class, prior_weight in zip(record.prior_class_members, record.prior_class_membership, strict=True):
            # VariantRecord rejects duplicate members; accumulate anyway so that a
            # record built around validation cannot silently lose mixture mass.
            class_membership_matrix[record_index, class_lookup[prior_class]] += prior_weight

    annotation_tables = _prior_annotation_tables(records)
    class_membership_by_class = {
        variant_class: class_membership_matrix[:, class_index]
        for class_index, variant_class in enumerate(unique_classes)
    }
    feature_specs = _compile_prior_feature_specs(
        annotation_tables=annotation_tables,
        class_membership_by_class=class_membership_by_class,
    )
    feature_names = [_feature_name_from_spec(feature_spec) for feature_spec in feature_specs]
    design_matrix = _design_matrix_for_feature_specs(
        records=records,
        feature_specs=feature_specs,
        annotation_tables=annotation_tables,
        class_membership_by_class=class_membership_by_class,
    )
    return PriorDesign(
        design_matrix=design_matrix,
        feature_names=feature_names,
        feature_specs=feature_specs,
        class_membership_matrix=class_membership_matrix,
        inverse_class_lookup=inverse_class_lookup,
    )


_UNIT_ROUNDOFF = float(np.finfo(np.float64).eps) / 2.0


def _summation_error_bound(term_count: int) -> float:
    """Higham's gamma_n = n u / (1 - n u): the relative error bound of an fp64 sum of n terms."""
    return term_count * _UNIT_ROUNDOFF / (1.0 - term_count * _UNIT_ROUNDOFF)


def _compile_prior_feature_specs(
    annotation_tables: _PriorAnnotationTables,
    class_membership_by_class: dict[VariantClass, NDArray],
) -> tuple[ScaleModelFeatureSpec, ...]:
    feature_specs: list[ScaleModelFeatureSpec] = []
    orthonormal_columns: list[NDArray] = []
    main_effect_specs: list[ScaleModelFeatureSpec] = []
    interaction_specs: list[ScaleModelFeatureSpec] = []
    def append_if_independent(feature_spec: ScaleModelFeatureSpec) -> None:
        feature_column = _column_for_feature_spec(
            feature_spec=feature_spec,
            annotation_tables=annotation_tables,
            class_membership_by_class=class_membership_by_class,
        )
        center_value = float(np.mean(feature_column))
        centered_column = np.asarray(feature_column, dtype=np.float64) - center_value
        rms_scale = float(np.sqrt(np.mean(centered_column * centered_column)))
        # A constant column centres to the rounding of its mean, at most gamma_n of its magnitude.
        if not np.isfinite(rms_scale) or rms_scale <= _summation_error_bound(centered_column.shape[0]) * float(
            np.max(np.abs(feature_column))
        ):
            return
        standardized_column = centered_column / rms_scale
        residual_column = standardized_column.copy()
        # Re-orthogonalize once. This is only a deterministic rank-revealing
        # screen: the emitted columns retain their interpretable reference-coded
        # values rather than becoming order-dependent orthogonal contrasts.
        for _reorthogonalization_pass in range(2):
            for orthonormal_column in orthonormal_columns:
                residual_column -= orthonormal_column * float(orthonormal_column @ residual_column)
        standardized_norm = float(np.linalg.norm(standardized_column))
        residual_norm = float(np.linalg.norm(residual_column))
        # The numerical-rank tolerance max(m, n) eps ||column|| (Golub and Van Loan, 5.4.1).
        rank_tolerance = max(standardized_column.shape[0], len(orthonormal_columns) + 1) * float(np.finfo(np.float64).eps)
        if residual_norm <= rank_tolerance * standardized_norm:
            return
        feature_specs.append(
            dataclass_replace(
                feature_spec,
                center_value=center_value,
                rms_scale=rms_scale,
            )
        )
        orthonormal_columns.append(np.asarray(residual_column / residual_norm, dtype=np.float64))

    encoded_variant_classes = _levels_to_encode(
        {
            variant_class: class_membership_by_class[variant_class]
            for variant_class in sorted(class_membership_by_class, key=lambda class_value: class_value.value)
        },
        parent_weights=np.ones_like(next(iter(class_membership_by_class.values()))),
    ) if class_membership_by_class else ()
    for variant_class in encoded_variant_classes:
        main_effect_specs.append(ScaleModelFeatureSpec(kind="type_offset", variant_class=variant_class))

    for source_name in sorted(annotation_tables.factor_weights_by_source):
        level_weights_by_name = annotation_tables.factor_weights_by_source[source_name]
        if not level_weights_by_name:
            continue
        for level_name in _levels_to_encode(
            level_weights_by_name,
            parent_weights=np.ones_like(next(iter(level_weights_by_name.values()))),
        ):
            main_effect_specs.append(
                ScaleModelFeatureSpec(
                    kind="factor_level",
                    source_name=source_name,
                    level_name=level_name,
                )
            )
            for variant_class in encoded_variant_classes:
                interaction_specs.append(
                    ScaleModelFeatureSpec(
                        kind="factor_interaction",
                        variant_class=variant_class,
                        source_name=source_name,
                        level_name=level_name,
                    )
                )

    for source_name in sorted(annotation_tables.nested_weights_by_source):
        nested_weights_by_depth = annotation_tables.nested_weights_by_source[source_name]
        for nested_depth in sorted(nested_weights_by_depth):
            for level_name in _nested_levels_to_encode(nested_weights_by_depth, nested_depth):
                main_effect_specs.append(
                    ScaleModelFeatureSpec(
                        kind="nested_level",
                        source_name=source_name,
                        level_name=level_name,
                        nested_depth=nested_depth,
                    )
                )
                for variant_class in encoded_variant_classes:
                    interaction_specs.append(
                        ScaleModelFeatureSpec(
                            kind="nested_interaction",
                            variant_class=variant_class,
                            source_name=source_name,
                            level_name=level_name,
                            nested_depth=nested_depth,
                        )
                    )

    for source_name in sorted(annotation_tables.continuous_values_by_source):
        continuous_values = annotation_tables.continuous_values_by_source[source_name]
        for base_feature_spec in _continuous_spline_feature_specs(source_name, continuous_values):
            main_effect_specs.append(base_feature_spec)
            for variant_class in encoded_variant_classes:
                interaction_specs.append(
                    ScaleModelFeatureSpec(
                        kind="continuous_spline_interaction",
                        variant_class=variant_class,
                        source_name=base_feature_spec.source_name,
                        basis_index=base_feature_spec.basis_index,
                        basis_kind=base_feature_spec.basis_kind,
                        standardize_mean=base_feature_spec.standardize_mean,
                        standardize_scale=base_feature_spec.standardize_scale,
                        knot_values=base_feature_spec.knot_values,
                    )
                )

    # Compile every main effect before any interaction. When annotations are
    # confounded on a particular dataset, the rank screen therefore preserves
    # the hierarchy instead of retaining an interaction at the expense of its
    # marginal annotation effect. The same screen drops the interactions of a
    # class too small to identify them, so no member count is needed.
    for feature_spec in (*main_effect_specs, *interaction_specs):
        append_if_independent(feature_spec)

    return tuple(feature_specs)


def _design_matrix_for_feature_specs(
    records: Sequence[VariantRecord],
    feature_specs: Sequence[ScaleModelFeatureSpec],
    annotation_tables: _PriorAnnotationTables,
    class_membership_by_class: dict[VariantClass, NDArray],
) -> NDArray:
    if len(feature_specs) == 0:
        return np.zeros((len(records), 0), dtype=np.float64)
    design_columns = [
        _standardize_design_column(
            _column_for_feature_spec(
                feature_spec=feature_spec,
                annotation_tables=annotation_tables,
                class_membership_by_class=class_membership_by_class,
            ),
            feature_spec=feature_spec,
        )
        for feature_spec in feature_specs
    ]
    return np.column_stack(design_columns).astype(np.float64)


def _column_for_feature_spec(
    feature_spec: ScaleModelFeatureSpec,
    annotation_tables: _PriorAnnotationTables,
    class_membership_by_class: dict[VariantClass, NDArray],
) -> NDArray:
    if feature_spec.kind == "type_offset":
        if feature_spec.variant_class is None:
            raise ValueError("Scale-model feature spec is missing variant_class.")
        return np.asarray(class_membership_by_class[feature_spec.variant_class], dtype=np.float64)

    if feature_spec.kind in {"factor_level", "factor_interaction"}:
        if feature_spec.source_name is None or feature_spec.level_name is None:
            raise ValueError("Factor scale-model feature spec is missing source_name or level_name.")
        if feature_spec.source_name not in annotation_tables.factor_weights_by_source:
            raise ValueError("Unknown factor scale-model feature: " + str(feature_spec.source_name))
        if feature_spec.level_name not in annotation_tables.factor_weights_by_source[feature_spec.source_name]:
            raise ValueError("Unknown factor scale-model level: " + str(feature_spec.level_name))
        feature_column = np.asarray(
            annotation_tables.factor_weights_by_source[feature_spec.source_name][feature_spec.level_name],
            dtype=np.float64,
        )
        if feature_spec.kind == "factor_level":
            return feature_column
        if feature_spec.variant_class is None:
            raise ValueError("Scale-model interaction feature spec is missing variant_class.")
        return feature_column * class_membership_by_class[feature_spec.variant_class]

    if feature_spec.kind in {"nested_level", "nested_interaction"}:
        if feature_spec.source_name is None or feature_spec.level_name is None or feature_spec.nested_depth is None:
            raise ValueError("Nested scale-model feature spec is missing source_name, level_name, or nested_depth.")
        if feature_spec.source_name not in annotation_tables.nested_weights_by_source:
            raise ValueError("Unknown nested scale-model feature: " + str(feature_spec.source_name))
        if feature_spec.nested_depth not in annotation_tables.nested_weights_by_source[feature_spec.source_name]:
            raise ValueError("Unknown nested depth for scale-model feature: " + str(feature_spec.nested_depth))
        nested_weights = annotation_tables.nested_weights_by_source[feature_spec.source_name][feature_spec.nested_depth]
        if feature_spec.level_name not in nested_weights:
            raise ValueError("Unknown nested scale-model level: " + str(feature_spec.level_name))
        feature_column = np.asarray(nested_weights[feature_spec.level_name], dtype=np.float64)
        if feature_spec.kind == "nested_level":
            return feature_column
        if feature_spec.variant_class is None:
            raise ValueError("Scale-model interaction feature spec is missing variant_class.")
        return feature_column * class_membership_by_class[feature_spec.variant_class]

    if feature_spec.kind in {"continuous_spline", "continuous_spline_interaction"}:
        if feature_spec.source_name is None:
            raise ValueError("Continuous scale-model feature spec is missing source_name.")
        if feature_spec.source_name not in annotation_tables.continuous_values_by_source:
            raise ValueError("Unknown continuous scale-model feature: " + str(feature_spec.source_name))
        feature_column = _continuous_spline_basis_column(
            raw_values=annotation_tables.continuous_values_by_source[feature_spec.source_name],
            feature_spec=feature_spec,
        )
        if feature_spec.kind == "continuous_spline":
            return feature_column
        if feature_spec.variant_class is None:
            raise ValueError("Scale-model interaction feature spec is missing variant_class.")
        return feature_column * class_membership_by_class[feature_spec.variant_class]

    raise ValueError("Unsupported scale-model feature kind: " + feature_spec.kind)


def _feature_name_from_spec(feature_spec: ScaleModelFeatureSpec) -> str:
    if feature_spec.kind == "type_offset":
        if feature_spec.variant_class is None:
            raise ValueError("Class effect scale-model feature spec is missing variant_class.")
        return "type_offset::" + feature_spec.variant_class.value
    if feature_spec.kind == "factor_level":
        return "factor_level::" + str(feature_spec.source_name) + "::" + str(feature_spec.level_name)
    if feature_spec.kind == "factor_interaction":
        return (
            "factor_interaction::"
            + str(feature_spec.source_name)
            + "::"
            + str(feature_spec.level_name)
            + "::"
            + str(feature_spec.variant_class.value if feature_spec.variant_class is not None else "")
        )
    if feature_spec.kind == "nested_level":
        return (
            "nested_level::"
            + str(feature_spec.source_name)
            + "::"
            + str(feature_spec.nested_depth)
            + "::"
            + str(feature_spec.level_name)
        )
    if feature_spec.kind == "nested_interaction":
        return (
            "nested_interaction::"
            + str(feature_spec.source_name)
            + "::"
            + str(feature_spec.nested_depth)
            + "::"
            + str(feature_spec.level_name)
            + "::"
            + str(feature_spec.variant_class.value if feature_spec.variant_class is not None else "")
        )
    if feature_spec.kind == "continuous_spline":
        return "continuous_spline::" + str(feature_spec.source_name) + "::basis_" + str(feature_spec.basis_index)
    if feature_spec.kind == "continuous_spline_interaction":
        return (
            "continuous_spline_interaction::"
            + str(feature_spec.source_name)
            + "::"
            + str(feature_spec.variant_class.value if feature_spec.variant_class is not None else "")
            + "::basis_"
            + str(feature_spec.basis_index)
        )
    raise ValueError("Unsupported scale-model feature kind: " + feature_spec.kind)


def _class_membership_weight(record: VariantRecord, variant_class: VariantClass) -> float:
    for member_class, member_weight in zip(record.prior_class_members, record.prior_class_membership, strict=True):
        if member_class == variant_class:
            return float(member_weight)
    return 0.0


def _class_membership_by_class(
    records: Sequence[VariantRecord],
    variant_classes: Iterable[VariantClass],
) -> dict[VariantClass, NDArray]:
    return {
        variant_class: np.asarray(
            [_class_membership_weight(record, variant_class) for record in records],
            dtype=np.float64,
        )
        for variant_class in variant_classes
    }


def _prior_annotation_tables(records: Sequence[VariantRecord]) -> _PriorAnnotationTables:
    feature_names = _prior_annotation_feature_names(records)
    return _PriorAnnotationTables(
        continuous_values_by_source=_continuous_prior_annotation_values(records, feature_names.continuous),
        factor_weights_by_source=_factor_prior_annotation_weights(
            records,
            binary_feature_names=feature_names.binary,
            categorical_feature_names=feature_names.categorical,
            membership_feature_names=feature_names.membership,
        ),
        nested_weights_by_source=_nested_prior_annotation_weights(
            records,
            nested_feature_names=feature_names.nested | feature_names.nested_membership,
        ),
    )


def _prior_annotation_feature_names(records: Sequence[VariantRecord]) -> _PriorAnnotationFeatureNames:
    feature_names = _PriorAnnotationFeatureNames(
        continuous=set(),
        binary=set(),
        categorical=set(),
        membership=set(),
        nested=set(),
        nested_membership=set(),
    )
    for record in records:
        feature_names.continuous.update(record.prior_continuous_features)
        feature_names.binary.update(record.prior_binary_features)
        feature_names.categorical.update(record.prior_categorical_features)
        feature_names.membership.update(record.prior_membership_features)
        feature_names.nested.update(record.prior_nested_features)
        feature_names.nested_membership.update(record.prior_nested_membership_features)
    return feature_names


def _continuous_prior_annotation_values(
    records: Sequence[VariantRecord],
    feature_names: set[str],
) -> dict[str, NDArray]:
    feature_values_by_name = {}
    for feature_name in sorted(feature_names):
        feature_values_by_name[feature_name] = np.asarray(
            [
                record.prior_continuous_features.get(feature_name, 0.0)
                for record in records
            ],
            dtype=np.float64,
        )
    return feature_values_by_name


def _factor_prior_annotation_weights(
    records: Sequence[VariantRecord],
    *,
    binary_feature_names: set[str],
    categorical_feature_names: set[str],
    membership_feature_names: set[str],
) -> dict[str, dict[str, NDArray]]:
    factor_weights_by_source: dict[str, dict[str, NDArray]] = {}
    for feature_name in sorted(binary_feature_names):
        true_values = np.asarray(
            [float(record.prior_binary_features.get(feature_name, False)) for record in records],
            dtype=np.float64,
        )
        factor_weights_by_source[feature_name] = {
            "false": 1.0 - true_values,
            "true": true_values,
        }

    for feature_name in sorted(categorical_feature_names):
        levels = sorted(
            {
                record.prior_categorical_features[feature_name]
                for record in records
                if feature_name in record.prior_categorical_features
            }
        )
        factor_weights_by_source[feature_name] = {
            level_name: np.asarray(
                [
                    float(record.prior_categorical_features.get(feature_name) == level_name)
                    for record in records
                ],
                dtype=np.float64,
            )
            for level_name in levels
        }

    for feature_name in sorted(membership_feature_names):
        levels = sorted(
            {
                level_name
                for record in records
                for level_name in record.prior_membership_features.get(feature_name, {})
            }
        )
        factor_weights_by_source[feature_name] = {
            level_name: np.asarray(
                [
                    record.prior_membership_features.get(feature_name, {}).get(level_name, 0.0)
                    for record in records
                ],
                dtype=np.float64,
            )
            for level_name in levels
        }
    return factor_weights_by_source


def _nested_prior_annotation_weights(
    records: Sequence[VariantRecord],
    *,
    nested_feature_names: set[str],
) -> dict[str, dict[int, dict[str, NDArray]]]:
    nested_weights_by_source: dict[str, dict[int, dict[str, NDArray]]] = {}
    for feature_name in sorted(nested_feature_names):
        source_nested_weights: dict[int, dict[str, NDArray]] = {}
        for record_index, record in enumerate(records):
            path_weights: dict[str, float] = {}
            if feature_name in record.prior_nested_features:
                path_weights[NESTED_PATH_DELIMITER.join(record.prior_nested_features[feature_name])] = 1.0
            for path_name, path_weight in record.prior_nested_membership_features.get(feature_name, {}).items():
                path_weights[path_name] = path_weights.get(path_name, 0.0) + float(path_weight)
            for path_name, path_weight in path_weights.items():
                if path_weight <= 0.0:
                    continue
                path_parts = tuple(path_name.split(NESTED_PATH_DELIMITER))
                for nested_depth in range(len(path_parts)):
                    nested_level_name = NESTED_PATH_DELIMITER.join(path_parts[: nested_depth + 1])
                    source_nested_weights.setdefault(nested_depth, {}).setdefault(
                        nested_level_name,
                        np.zeros(len(records), dtype=np.float64),
                    )[record_index] += path_weight
        nested_weights_by_source[feature_name] = source_nested_weights
    return nested_weights_by_source


def _levels_to_encode(
    level_weights_by_name: Mapping[Any, NDArray],
    *,
    parent_weights: NDArray,
) -> tuple[Any, ...]:
    """Return a full-rank reference coding for one sibling group.

    When the supplied levels exhaust their parent, the most prevalent level
    is the deterministic reference and is omitted. If some parent membership
    is unassigned (missing annotations or a terminal node), that unassigned
    mass is already an implicit reference, so every explicit level is kept.
    """
    if not level_weights_by_name:
        return ()
    sorted_levels = tuple(
        sorted(
            level_weights_by_name,
            key=lambda level_name: (
                level_name.value if isinstance(level_name, VariantClass) else str(level_name)
            ),
        )
    )
    represented_weights = np.sum(
        np.vstack([np.asarray(level_weights_by_name[level_name], dtype=np.float64) for level_name in sorted_levels]),
        axis=0,
    )
    implicit_reference = np.asarray(parent_weights, dtype=np.float64) - represented_weights
    centered_implicit_reference = implicit_reference - float(np.mean(implicit_reference))
    # Levels that exhaust the parent leave only the rounding of the sum and the mean.
    exhaustion_bound = _summation_error_bound(len(sorted_levels) + implicit_reference.shape[0]) * float(
        np.max(np.abs(parent_weights))
    )
    if np.max(np.abs(centered_implicit_reference)) > exhaustion_bound:
        return sorted_levels
    reference_level = max(
        sorted_levels,
        key=lambda level_name: float(np.sum(level_weights_by_name[level_name])),
    )
    return tuple(level_name for level_name in sorted_levels if level_name != reference_level)


def _nested_levels_to_encode(
    nested_weights_by_depth: dict[int, dict[str, NDArray]],
    nested_depth: int,
) -> tuple[str, ...]:
    level_weights_by_name = nested_weights_by_depth[nested_depth]
    levels_by_parent: dict[str | None, dict[str, NDArray]] = {}
    for level_name in sorted(level_weights_by_name):
        parent_name = None if nested_depth == 0 else level_name.rsplit(NESTED_PATH_DELIMITER, maxsplit=1)[0]
        levels_by_parent.setdefault(parent_name, {})[level_name] = level_weights_by_name[level_name]

    encoded_levels: list[str] = []
    for parent_name in sorted(levels_by_parent, key=lambda value: "" if value is None else value):
        sibling_weights = levels_by_parent[parent_name]
        if parent_name is None:
            parent_weights = np.ones_like(next(iter(sibling_weights.values())))
        else:
            parent_weights = nested_weights_by_depth[nested_depth - 1][parent_name]
        encoded_levels.extend(
            _levels_to_encode(
                sibling_weights,
                parent_weights=parent_weights,
            )
        )
    return tuple(encoded_levels)


def _continuous_spline_feature_specs(
    source_name: str,
    raw_values: NDArray,
) -> tuple[ScaleModelFeatureSpec, ...]:
    mean_value = float(np.mean(raw_values))
    scale_value = float(np.std(raw_values))
    if scale_value <= _summation_error_bound(np.asarray(raw_values).shape[0]) * float(np.max(np.abs(raw_values))):
        return ()
    standardized_values = (np.asarray(raw_values, dtype=np.float64) - mean_value) / scale_value
    feature_specs = [
        ScaleModelFeatureSpec(
            kind="continuous_spline",
            source_name=source_name,
            basis_index=0,
            basis_kind="linear",
            standardize_mean=mean_value,
            standardize_scale=scale_value,
        )
    ]
    for basis_index, knot_value in enumerate(_continuous_spline_knots(standardized_values), start=1):
        feature_specs.append(
            ScaleModelFeatureSpec(
                kind="continuous_spline",
                source_name=source_name,
                basis_index=basis_index,
                basis_kind="cubic_hinge",
                standardize_mean=mean_value,
                standardize_scale=scale_value,
                knot_values=(float(knot_value),),
            )
        )
    return tuple(feature_specs)


def _continuous_spline_knots(standardized_values: NDArray) -> tuple[float, ...]:
    """Distinct quartile knots strictly inside the data range; the rank screen drops any hinge a
    nearby knot makes numerically dependent."""
    candidate_knots = np.quantile(standardized_values, [0.25, 0.5, 0.75])
    minimum_value = float(np.min(standardized_values))
    maximum_value = float(np.max(standardized_values))
    knot_values: list[float] = []
    for knot_value in candidate_knots:
        knot_float = float(knot_value)
        if minimum_value < knot_float < maximum_value and (not knot_values or knot_float != knot_values[-1]):
            knot_values.append(knot_float)
    return tuple(knot_values)


def _continuous_spline_basis_column(
    raw_values: NDArray,
    feature_spec: ScaleModelFeatureSpec,
) -> NDArray:
    if feature_spec.standardize_mean is None or feature_spec.standardize_scale is None:
        raise ValueError("Continuous spline feature spec is missing standardization parameters.")
    standardized_values = (
        np.asarray(raw_values, dtype=np.float64) - float(feature_spec.standardize_mean)
    ) / float(feature_spec.standardize_scale)
    if feature_spec.basis_kind == "linear":
        return standardized_values
    if feature_spec.basis_kind == "cubic_hinge":
        if len(feature_spec.knot_values) != 1:
            raise ValueError("Cubic hinge spline feature spec must store exactly one knot.")
        return np.maximum(standardized_values - float(feature_spec.knot_values[0]), 0.0) ** 3
    raise ValueError("Unsupported continuous spline basis kind: " + str(feature_spec.basis_kind))


def _standardize_design_column(
    values: NDArray,
    *,
    feature_spec: ScaleModelFeatureSpec,
) -> NDArray:
    if feature_spec.center_value is None or feature_spec.rms_scale is None:
        raise ValueError("Scale-model feature spec is missing its fitted centering or RMS scale.")
    if not np.isfinite(feature_spec.center_value):
        raise ValueError("Scale-model feature centering value must be finite.")
    if not np.isfinite(feature_spec.rms_scale) or feature_spec.rms_scale <= 0.0:
        raise ValueError("Scale-model feature RMS scale must be finite and positive.")
    return np.asarray(
        (np.asarray(values, dtype=np.float64) - feature_spec.center_value) / feature_spec.rms_scale,
        dtype=np.float64,
    )


def _metadata_baseline_scales_from_coefficients(
    scale_model_coefficients: NDArray,
    design_matrix: NDArray,
    config: ModelConfig,
) -> NDArray:
    if design_matrix.shape[1] == 0:
        return np.ones(design_matrix.shape[0], dtype=np.float64)
    linear_prediction = design_matrix @ scale_model_coefficients
    return _metadata_baseline_scales_from_log_predictions(
        log_predictions=linear_prediction,
        config=config,
    )


def _metadata_baseline_scales_from_log_predictions(
    *,
    log_predictions: NDArray,
    config: ModelConfig,
) -> NDArray:
    bounded_log_scales = np.clip(
        np.asarray(log_predictions, dtype=np.float64),
        np.log(config.prior_scale_floor),
        np.log(config.prior_scale_ceiling),
    )
    return np.exp(bounded_log_scales).astype(np.float64)


def _support_bounded_member_log_scale_predictions(
    *,
    member_design_matrix: NDArray,
    reduced_design_matrix: NDArray,
    scale_model_coefficients: NDArray,
) -> NDArray:
    member_design = np.asarray(member_design_matrix, dtype=np.float64)
    reduced_design = np.asarray(reduced_design_matrix, dtype=np.float64)
    coefficients = np.asarray(scale_model_coefficients, dtype=np.float64)
    if member_design.ndim != 2 or reduced_design.ndim != 2:
        raise ValueError("Member and reduced scale-model designs must be two-dimensional.")
    if reduced_design.shape[0] == 0:
        raise ValueError("The reduced scale-model design must contain at least one fitted row.")
    if member_design.shape[1] != reduced_design.shape[1] or coefficients.shape != (reduced_design.shape[1],):
        raise ValueError("Scale-model designs and coefficients must share one feature dimension.")

    member_predictions = member_design @ coefficients
    reduced_predictions = reduced_design @ coefficients
    return np.asarray(
        np.clip(
            member_predictions,
            float(np.min(reduced_predictions)),
            float(np.max(reduced_predictions)),
        ),
        dtype=np.float64,
    )


def _effective_prior_variances(
    baseline_prior_variances: NDArray,
    local_scale: NDArray,
    config: ModelConfig,
) -> NDArray:
    return np.asarray(
        np.maximum(
            baseline_prior_variances * np.maximum(local_scale, config.local_scale_floor),
            1e-8,
        ),
        dtype=np.float64,
    )


def _scale_state_reduced_prior_variances(
    *,
    global_scale: float,
    scale_model_coefficients: NDArray,
    local_scale: NDArray,
    design_matrix: NDArray,
    config: ModelConfig,
) -> NDArray:
    """Return the identifiable prior operator induced by the scale state.

    ``global_scale`` and ``local_scale`` can move in opposite directions
    without changing the Gaussian prior seen by the posterior solve. Outer
    convergence therefore tracks their effective product instead of treating
    that harmless reparameterization as hyperparameter movement.
    """
    metadata_baseline_scales = _metadata_baseline_scales_from_coefficients(
        scale_model_coefficients,
        design_matrix,
        config,
    )
    return _effective_prior_variances(
        baseline_prior_variances=(float(global_scale) * metadata_baseline_scales) ** 2,
        local_scale=local_scale,
        config=config,
    )


def _scale_model_penalty(
    feature_names: Sequence[str],
    config: ModelConfig,
) -> NDArray:
    penalty_values = np.full(len(feature_names), config.scale_model_ridge_penalty, dtype=np.float64)
    for feature_index, feature_name in enumerate(feature_names):
        if feature_name.startswith("type_offset::"):
            penalty_values[feature_index] = config.type_offset_penalty
    return penalty_values


def _member_prior_variances_from_reduced_state(
    member_records: Sequence[VariantRecord],
    tie_map: TieMap,
    scale_model_coefficients: NDArray,
    scale_model_feature_specs: Sequence[ScaleModelFeatureSpec],
    reduced_design_matrix: NDArray,
    global_scale: float,
    local_scale: NDArray,
    config: ModelConfig,
) -> NDArray:
    reduced_design = np.asarray(reduced_design_matrix, dtype=np.float64)
    reduced_local_scale = np.asarray(local_scale, dtype=np.float64)
    expected_reduced_shape = (reduced_local_scale.shape[0], len(scale_model_feature_specs))
    if reduced_design.shape != expected_reduced_shape:
        raise ValueError(
            "reduced_design_matrix must align with the reduced local-scale state and feature specs."
        )
    feature_variant_classes = {
        feature_spec.variant_class
        for feature_spec in scale_model_feature_specs
        if feature_spec.variant_class is not None
    }
    member_design_matrix = _design_matrix_for_feature_specs(
        records=member_records,
        feature_specs=scale_model_feature_specs,
        annotation_tables=_prior_annotation_tables(member_records),
        class_membership_by_class=_class_membership_by_class(member_records, feature_variant_classes),
    )
    coefficients = np.asarray(scale_model_coefficients, dtype=np.float64)
    # A mixed exact-tie record can carry fractional memberships even though its
    # individual members are pure levels. Reapplying a standardized contrast to
    # those members can extrapolate the fitted scalar log scale far outside the
    # support seen by the reduced model (for example, fitted predictions in
    # [-0.43, +0.43] becoming +2.14 for one pure member). Neither the genotype
    # likelihood nor the reduced prior fit identifies that extrapolation. Bound
    # the allocation coordinate to the convex hull of fitted reduced log-scale
    # predictions. Unlike per-feature clipping, this is invariant to any
    # invertible recoding of the full-rank prior design.
    supported_member_log_scales = _support_bounded_member_log_scale_predictions(
        member_design_matrix=member_design_matrix,
        reduced_design_matrix=reduced_design,
        scale_model_coefficients=coefficients,
    )
    member_baseline_scales = _metadata_baseline_scales_from_log_predictions(
        log_predictions=supported_member_log_scales,
        config=config,
    )
    member_baseline_prior_variances = (float(global_scale) * member_baseline_scales) ** 2
    member_local_scale = _expand_group_values_to_members(
        reduced_values=reduced_local_scale,
        tie_map=tie_map,
    )
    provisional_member_variances = _effective_prior_variances(
        baseline_prior_variances=member_baseline_prior_variances,
        local_scale=member_local_scale,
        config=config,
    )
    reduced_prior_variances = _scale_state_reduced_prior_variances(
        global_scale=global_scale,
        scale_model_coefficients=coefficients,
        local_scale=reduced_local_scale,
        design_matrix=reduced_design,
        config=config,
    )
    return _normalize_member_variances_to_reduced_groups(
        provisional_member_variances=provisional_member_variances,
        reduced_prior_variances=reduced_prior_variances,
        tie_map=tie_map,
    )


def _normalize_member_variances_to_reduced_groups(
    *,
    provisional_member_variances: NDArray,
    reduced_prior_variances: NDArray,
    tie_map: TieMap,
) -> NDArray:
    member_variances = np.asarray(provisional_member_variances, dtype=np.float64)
    reduced_variances = np.asarray(reduced_prior_variances, dtype=np.float64)
    if member_variances.shape != tie_map.original_to_reduced.shape:
        raise ValueError("provisional_member_variances must align with the tie-map member space.")
    if reduced_variances.shape != (tie_map.kept_indices.shape[0],):
        raise ValueError("reduced_prior_variances must align with the tie-map reduced space.")

    normalized_variances = np.zeros_like(member_variances)
    if not tie_map.reduced_to_group:
        normalized_variances[tie_map.kept_indices] = reduced_variances
        return normalized_variances
    if len(tie_map.reduced_to_group) != reduced_variances.shape[0]:
        raise ValueError("tie-map groups must align with reduced_prior_variances.")

    for reduced_index, tie_group in enumerate(tie_map.reduced_to_group):
        group_member_indices = np.asarray(tie_group.member_indices, dtype=np.int32)
        group_variances = member_variances[group_member_indices]
        group_total = float(np.sum(group_variances))
        if not np.isfinite(group_total) or group_total <= 0.0:
            raise FloatingPointError("Exact-tie member prior variances must have a finite positive sum.")
        normalized_variances[group_member_indices] = (
            group_variances * (float(reduced_variances[reduced_index]) / group_total)
        )
    return normalized_variances


def _tie_map_is_identity(tie_map: TieMap, *, member_count: int) -> bool:
    if (
        tie_map.kept_indices.shape != (int(member_count),)
        or tie_map.original_to_reduced.shape != (int(member_count),)
    ):
        return False
    if not np.array_equal(tie_map.kept_indices, np.arange(int(member_count), dtype=np.int32)):
        return False
    if not np.array_equal(tie_map.original_to_reduced, np.arange(int(member_count), dtype=np.int32)):
        return False
    return True


def _expand_group_values_to_members(
    reduced_values: NDArray,
    tie_map: TieMap,
) -> NDArray:
    reduced_array = np.asarray(reduced_values, dtype=np.float64)
    if not tie_map.reduced_to_group:
        member_values = np.zeros(tie_map.original_to_reduced.shape[0], dtype=np.float64)
        if tie_map.kept_indices.shape[0] != reduced_array.shape[0]:
            raise ValueError("reduced_values must align with compact tie-map representatives.")
        member_values[np.asarray(tie_map.kept_indices, dtype=np.int32)] = reduced_array
        return member_values
    member_values = np.zeros(tie_map.original_to_reduced.shape[0], dtype=np.float64)
    for reduced_index, tie_group in enumerate(tie_map.reduced_to_group):
        member_values[tie_group.member_indices] = float(reduced_values[reduced_index])
    return member_values


def collapse_tie_groups(
    records: Sequence[VariantRecord],
    tie_map: TieMap,
) -> list[VariantRecord]:
    """Create one merged record per tie group for use in the reduced model."""
    if not tie_map.reduced_to_group:
        kept_indices = np.asarray(tie_map.kept_indices, dtype=np.int32)
        if kept_indices.shape[0] == len(records) and np.array_equal(kept_indices, np.arange(len(records), dtype=np.int32)):
            return cast(list[VariantRecord], records) if isinstance(records, list) else list(records)
        return [records[int(record_index)] for record_index in kept_indices]
    if len(tie_map.reduced_to_group) == len(records):
        record_indices = np.arange(len(records), dtype=np.int32)
        if (
            np.array_equal(tie_map.kept_indices, record_indices)
            and np.array_equal(tie_map.original_to_reduced, record_indices)
        ):
            return cast(list[VariantRecord], records) if isinstance(records, list) else list(records)

    collapsed_records: list[VariantRecord] = []
    for tie_group in tie_map.reduced_to_group:
        member_indices = tie_group.member_indices
        if member_indices.shape[0] == 1:
            collapsed_records.append(records[int(member_indices[0])])
            continue
        member_records = [records[int(member_index)] for member_index in member_indices]
        unique_variant_classes, class_membership = _class_membership(member_records)
        latent_variant_class = unique_variant_classes[0] if len(unique_variant_classes) == 1 else VariantClass.OTHER_COMPLEX_SV
        support_values = [
            float(member_record.training_support)
            for member_record in member_records
            if member_record.training_support is not None
        ]
        continuous_feature_names = sorted(
            {
                feature_name
                for member_record in member_records
                for feature_name in member_record.prior_continuous_features
            }
        )
        binary_feature_names = sorted(
            {
                feature_name
                for member_record in member_records
                for feature_name in member_record.prior_binary_features
            }
        )
        categorical_feature_names = sorted(
            {
                feature_name
                for member_record in member_records
                for feature_name in member_record.prior_categorical_features
            }
        )
        nested_feature_names = sorted(
            {
                feature_name
                for member_record in member_records
                for feature_name in member_record.prior_nested_features
            }
        )
        collapsed_binary_features: dict[str, bool] = {}
        collapsed_categorical_features: dict[str, str] = {}
        collapsed_membership_features = _average_weighted_feature_dicts(
            [
                member_record.prior_membership_features
                for member_record in member_records
            ]
        )
        collapsed_nested_features: dict[str, tuple[str, ...]] = {}
        collapsed_nested_membership_features = _average_weighted_feature_dicts(
            [
                member_record.prior_nested_membership_features
                for member_record in member_records
            ]
        )

        for feature_name in binary_feature_names:
            true_frequency = float(
                np.mean(
                    [
                        float(member_record.prior_binary_features.get(feature_name, False))
                        for member_record in member_records
                    ]
                )
            )
            if np.isclose(true_frequency, 0.0):
                collapsed_binary_features[feature_name] = False
                continue
            if np.isclose(true_frequency, 1.0):
                collapsed_binary_features[feature_name] = True
                continue
            collapsed_membership_features[feature_name] = {
                "false": 1.0 - true_frequency,
                "true": true_frequency,
            }

        for feature_name in categorical_feature_names:
            feature_values = [
                member_record.prior_categorical_features.get(feature_name)
                for member_record in member_records
            ]
            observed_feature_values = [feature_value for feature_value in feature_values if feature_value is not None]
            if not observed_feature_values:
                continue
            if len(set(observed_feature_values)) == 1 and len(observed_feature_values) == len(member_records):
                collapsed_categorical_features[feature_name] = observed_feature_values[0]
                continue
            collapsed_membership_features[feature_name] = {
                feature_value: float(np.mean([value == feature_value for value in feature_values]))
                for feature_value in sorted(set(observed_feature_values))
            }

        for feature_name in nested_feature_names:
            feature_paths = [
                member_record.prior_nested_features.get(feature_name)
                for member_record in member_records
            ]
            observed_feature_paths = [feature_path for feature_path in feature_paths if feature_path is not None]
            if not observed_feature_paths:
                continue
            if len(set(observed_feature_paths)) == 1 and len(observed_feature_paths) == len(member_records):
                collapsed_nested_features[feature_name] = observed_feature_paths[0]
                continue
            collapsed_nested_membership_features[feature_name] = {
                ">".join(feature_path): float(np.mean([path_value == feature_path for path_value in feature_paths]))
                for feature_path in sorted(set(observed_feature_paths))
            }

        collapsed_records.append(
            VariantRecord(
                variant_id=member_records[0].variant_id,
                variant_class=latent_variant_class,
                chromosome=member_records[0].chromosome,
                position=int(np.min([member_record.position for member_record in member_records])),
                length=float(np.mean([member_record.length for member_record in member_records])),
                allele_frequency=float(np.mean([member_record.allele_frequency for member_record in member_records])),
                quality=float(np.mean([member_record.quality for member_record in member_records])),
                training_support=None if not support_values else int(np.round(np.mean(support_values))),
                is_repeat=any(member_record.is_repeat for member_record in member_records),
                is_copy_number=any(member_record.is_copy_number for member_record in member_records),
                prior_binary_features=collapsed_binary_features,
                prior_continuous_features={
                    feature_name: float(
                        np.mean(
                            [
                                member_record.prior_continuous_features.get(feature_name, 0.0)
                                for member_record in member_records
                            ]
                        )
                    )
                    for feature_name in continuous_feature_names
                },
                prior_categorical_features=collapsed_categorical_features,
                prior_membership_features=collapsed_membership_features,
                prior_nested_features=collapsed_nested_features,
                prior_nested_membership_features=collapsed_nested_membership_features,
                prior_class_members=tuple(unique_variant_classes),
                prior_class_membership=tuple(class_membership.tolist()),
            )
        )
    return collapsed_records


def _average_weighted_feature_dicts(
    weighted_feature_dicts: Sequence[dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    averaged_features: dict[str, dict[str, float]] = {}
    feature_names = sorted(
        {
            feature_name
            for feature_dict in weighted_feature_dicts
            for feature_name in feature_dict
        }
    )
    for feature_name in feature_names:
        level_names = sorted(
            {
                level_name
                for feature_dict in weighted_feature_dicts
                for level_name in feature_dict.get(feature_name, {})
            }
        )
        averaged_levels = {
            level_name: float(
                np.mean(
                    [
                        feature_dict.get(feature_name, {}).get(level_name, 0.0)
                        for feature_dict in weighted_feature_dicts
                    ]
                )
            )
            for level_name in level_names
        }
        nonzero_levels = {
            level_name: level_weight
            for level_name, level_weight in averaged_levels.items()
            if level_weight > 0.0
        }
        if nonzero_levels:
            averaged_features[feature_name] = nonzero_levels
    return averaged_features


def _class_membership(member_records: Sequence[VariantRecord]) -> tuple[list[VariantClass], F64Array]:
    class_counts: dict[VariantClass, int] = {}
    for member_record in member_records:
        class_counts[member_record.variant_class] = class_counts.get(member_record.variant_class, 0) + 1
    unique_variant_classes = sorted(class_counts, key=lambda variant_class: variant_class.value)
    class_weights = np.asarray(
        [class_counts[variant_class] for variant_class in unique_variant_classes],
        dtype=np.float64,
    )
    class_weights /= np.sum(class_weights)
    return unique_variant_classes, class_weights
