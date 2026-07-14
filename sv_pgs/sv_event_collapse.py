"""Collapse near-duplicate SV calls that describe the same biological event.

Many SV callers emit multiple, slightly-shifted calls describing the
same underlying event — same chromosome, same SV type, breakpoints
agreeing to within tens of base pairs, and similar carrier counts.
Treating each of these as an independent feature inflates ``p_active``
and slows the variational fit without adding information.

This module provides a pure, deterministic clustering routine that
groups such redundant calls and picks one representative per cluster
based on support count. It is the "speed mode" collapse — it relies on
breakpoint+SVTYPE+support similarity only and never inspects per-sample
genotypes. A future "exact mode" can verify genotype-vector equality
within a cluster before collapsing.

The routine deliberately favors correctness and determinism over raw
speed: within each (chromosome, SV-class) group it does a sorted sweep
with a small local window. The asymptotic complexity is therefore
``O(N log N + N * W)`` where ``W`` is ``_LOCAL_WINDOW`` (currently 50).
"""

from __future__ import annotations

from collections import defaultdict

from sv_pgs.config import STRUCTURAL_VARIANT_CLASSES, VariantClass
from sv_pgs.data import VariantRecord


_LOCAL_WINDOW = 50

_STRUCTURAL_CLASS_SET = frozenset(STRUCTURAL_VARIANT_CLASSES)


def _is_structural(variant_class: VariantClass) -> bool:
    """Return True if a variant class is an SV (as opposed to SNV / small indel)."""
    return variant_class in _STRUCTURAL_CLASS_SET


def _record_span(record: VariantRecord) -> tuple[int, int]:
    """Closed start / open end coordinates for a variant.

    ``length`` is stored as a float on :class:`VariantRecord`; we coerce
    to ``int`` and clamp to at least one base so that single-bp events
    (insertions, breakend signals) still yield a well-defined span.
    """
    start_position = int(record.position)
    span_length = max(1, int(round(float(record.length))))
    return start_position, start_position + span_length


def _reciprocal_overlap(
    span_a: tuple[int, int],
    span_b: tuple[int, int],
) -> float:
    """Reciprocal overlap fraction in ``[0.0, 1.0]``.

    Defined as ``intersection / max(len_a, len_b)`` — equivalent to
    requiring that the intersection covers at least the threshold
    fraction of *both* spans (the classical "reciprocal overlap"
    convention used by SV truth-set comparisons).
    """
    start_a, end_a = span_a
    start_b, end_b = span_b
    intersection_length = max(0, min(end_a, end_b) - max(start_a, start_b))
    if intersection_length <= 0:
        return 0.0
    length_a = end_a - start_a
    length_b = end_b - start_b
    longer_length = max(length_a, length_b)
    if longer_length <= 0:
        return 0.0
    return intersection_length / float(longer_length)


def _support_value(record: VariantRecord) -> int:
    """Support count for similarity comparison.

    Falls back to ``0`` when ``training_support`` is absent so that
    records loaded without per-variant support still cluster purely on
    breakpoint geometry.
    """
    support = record.training_support
    if support is None:
        return 0
    return int(support)


def _support_counts_similar(
    support_a: int,
    support_b: int,
    *,
    ratio_tolerance: float,
) -> bool:
    """True if two non-negative support counts differ by at most ``ratio_tolerance``.

    Comparison is symmetric: ``|a - b| <= ratio_tolerance * max(a, b)``.
    When both counts are zero (e.g. support unavailable on both
    records) we treat them as similar.
    """
    if support_a == support_b:
        return True
    larger = max(support_a, support_b)
    if larger <= 0:
        return True
    return abs(support_a - support_b) <= ratio_tolerance * larger


def cluster_sv_events(
    variant_records: list[VariantRecord],
    *,
    reciprocal_overlap_threshold: float = 0.9,
    breakpoint_slop_bp: int = 50,
    support_count_ratio_tolerance: float = 0.2,
) -> dict[int, list[int]]:
    """Map representative-variant-index -> [member-variant-indices].

    Non-clustered variants map to themselves. Only structural variants
    are eligible for clustering; SNV / small-indel records always map
    to singleton clusters keyed on their own index. The output is
    deterministic given a sorted input order.

    Two SV records are considered the same event when:

    1. They share a chromosome and ``VariantClass`` (the SVTYPE proxy).
    2. Their start positions are within ``breakpoint_slop_bp`` of each
       other.
    3. Their spans satisfy reciprocal overlap >= ``reciprocal_overlap_threshold``.
    4. Their support counts agree to within ``support_count_ratio_tolerance``.

    The representative of each cluster is the member with the highest
    support count; ties are broken by smallest original index so the
    mapping is fully determined by the input.
    """
    if reciprocal_overlap_threshold < 0.0 or reciprocal_overlap_threshold > 1.0:
        raise ValueError("reciprocal_overlap_threshold must be in [0, 1].")
    if breakpoint_slop_bp < 0:
        raise ValueError("breakpoint_slop_bp must be non-negative.")
    if support_count_ratio_tolerance < 0.0:
        raise ValueError("support_count_ratio_tolerance must be non-negative.")

    cluster_members: dict[int, list[int]] = {
        record_index: [record_index] for record_index in range(len(variant_records))
    }

    # Group structural-variant indices by (chromosome, SV class).
    group_to_indices: dict[tuple[str, VariantClass], list[int]] = defaultdict(list)
    for record_index, record in enumerate(variant_records):
        if not _is_structural(record.variant_class):
            continue
        group_to_indices[(record.chromosome, record.variant_class)].append(record_index)

    for group_indices in group_to_indices.values():
        if len(group_indices) < 2:
            continue
        # Sort by (start, end, original index) for determinism.
        sorted_indices = sorted(
            group_indices,
            key=lambda record_index: (
                int(variant_records[record_index].position),
                _record_span(variant_records[record_index])[1],
                record_index,
            ),
        )

        # Union-find over positions in `sorted_indices`.
        parent: list[int] = list(range(len(sorted_indices)))

        def find_root(node_index: int) -> int:
            root = node_index
            while parent[root] != root:
                root = parent[root]
            # Path compression.
            while parent[node_index] != root:
                parent[node_index], node_index = root, parent[node_index]
            return root

        def union(left_index: int, right_index: int) -> None:
            left_root = find_root(left_index)
            right_root = find_root(right_index)
            if left_root == right_root:
                return
            # Attach the larger label onto the smaller to keep the root
            # index minimal — purely cosmetic for determinism.
            if left_root < right_root:
                parent[right_root] = left_root
            else:
                parent[left_root] = right_root

        for local_index, sorted_record_index in enumerate(sorted_indices):
            record = variant_records[sorted_record_index]
            record_span = _record_span(record)
            record_support = _support_value(record)
            window_stop = min(len(sorted_indices), local_index + 1 + _LOCAL_WINDOW)
            for neighbor_local_index in range(local_index + 1, window_stop):
                neighbor_record_index = sorted_indices[neighbor_local_index]
                neighbor_record = variant_records[neighbor_record_index]
                if abs(int(neighbor_record.position) - int(record.position)) > breakpoint_slop_bp:
                    # Sorted by start — once we exceed the slop, all
                    # further neighbors in the window also exceed it.
                    break
                neighbor_span = _record_span(neighbor_record)
                if _reciprocal_overlap(record_span, neighbor_span) < reciprocal_overlap_threshold:
                    continue
                if not _support_counts_similar(
                    record_support,
                    _support_value(neighbor_record),
                    ratio_tolerance=support_count_ratio_tolerance,
                ):
                    continue
                union(local_index, neighbor_local_index)

        # Materialize clusters: group local indices by their root, then
        # pick the representative as the member with the highest
        # support count (ties: smallest original index).
        root_to_members: dict[int, list[int]] = defaultdict(list)
        for local_index in range(len(sorted_indices)):
            root_to_members[find_root(local_index)].append(sorted_indices[local_index])

        for member_indices in root_to_members.values():
            if len(member_indices) < 2:
                continue
            representative_index = max(
                member_indices,
                key=lambda record_index: (
                    _support_value(variant_records[record_index]),
                    -record_index,
                ),
            )
            sorted_members = sorted(member_indices)
            # Reset all member singletons we pre-seeded above, then
            # install the merged cluster on the representative.
            for member_index in sorted_members:
                if member_index != representative_index:
                    cluster_members.pop(member_index, None)
            cluster_members[representative_index] = sorted_members

    return cluster_members


def select_representative_indices(
    cluster_map: dict[int, list[int]],
) -> list[int]:
    """Convenience: sorted list of representative indices from a cluster map."""
    return sorted(cluster_map.keys())


__all__ = [
    "cluster_sv_events",
    "select_representative_indices",
]
