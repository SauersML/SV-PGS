"""A tie group's prior record: the representative's position, pooled class membership and averaged features."""
from __future__ import annotations

import numpy as np

from sv_pgs.config import VariantClass
from sv_pgs.data import VariantRecord
from sv_pgs.prior_design import collapse_tie_groups
from sv_pgs.tie_map import _compact_identity_tie_map, tie_map_from_groups


def test_without_ties_the_records_are_reused() -> None:
    variant_records = [
        VariantRecord("variant_0", VariantClass.SNV, "1", 100),
        VariantRecord("variant_1", VariantClass.DELETION, "1", 101),
    ]

    assert collapse_tie_groups(variant_records, _compact_identity_tie_map(2)) is variant_records


def test_a_mixed_class_group_gets_the_symmetric_latent_class() -> None:
    variant_records = [
        VariantRecord("variant_0", VariantClass.SNV, "1", 100, length=1.0, allele_frequency=0.10, quality=1.0),
        VariantRecord("variant_1", VariantClass.DELETION, "1", 101, length=500.0, allele_frequency=0.02, quality=0.8),
        VariantRecord("variant_2", VariantClass.SNV, "1", 200, length=1.0, allele_frequency=0.10, quality=0.9),
    ]
    tie_map = tie_map_from_groups(3, {0: [(0, 1.0), (1, 1.0)], 2: [(2, 1.0)]})

    collapsed_records = collapse_tie_groups(variant_records, tie_map)

    assert collapsed_records[0].variant_class == VariantClass.OTHER_COMPLEX_SV
    assert collapsed_records[0].position == 100
    assert collapsed_records[0].is_repeat is False
    assert collapsed_records[0].prior_class_members == (VariantClass.DELETION, VariantClass.SNV)
    np.testing.assert_array_equal(collapsed_records[0].prior_class_membership, [0.5, 0.5])
    assert collapsed_records[1] is variant_records[2]


def test_a_group_averages_support_and_annotations() -> None:
    variant_records = [
        VariantRecord(
            "variant_0",
            VariantClass.DELETION,
            "1",
            100,
            training_support=6,
            prior_binary_features={"coding_annotation": True},
            prior_continuous_features={"sv_length_score": 1.0},
            prior_categorical_features={"functional_state": "lof"},
            prior_nested_features={"gene_context": ("protein_coding", "exon")},
        ),
        VariantRecord(
            "variant_1",
            VariantClass.DELETION,
            "1",
            101,
            training_support=8,
            prior_binary_features={"coding_annotation": False},
            prior_continuous_features={"sv_length_score": 3.0},
            prior_categorical_features={"functional_state": "missense"},
            prior_nested_features={"gene_context": ("protein_coding", "intron")},
        ),
        VariantRecord("variant_2", VariantClass.SNV, "1", 102),
    ]
    tie_map = tie_map_from_groups(3, {0: [(0, 1.0), (1, 1.0)], 2: [(2, 1.0)]})

    collapsed_records = collapse_tie_groups(variant_records, tie_map)

    assert collapsed_records[0].training_support == 7
    assert collapsed_records[0].prior_binary_features == {}
    assert collapsed_records[0].prior_continuous_features == {"sv_length_score": 2.0}
    assert collapsed_records[0].prior_membership_features["coding_annotation"] == {"false": 0.5, "true": 0.5}
    assert collapsed_records[0].prior_membership_features["functional_state"] == {"lof": 0.5, "missense": 0.5}
    assert collapsed_records[0].prior_nested_membership_features["gene_context"] == {
        "protein_coding>exon": 0.5,
        "protein_coding>intron": 0.5,
    }
    assert collapsed_records[1] is variant_records[2]
