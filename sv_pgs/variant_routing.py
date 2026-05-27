"""Per-variant routing classifier.

Decides at load time, for each variant, whether the variant should be carried
in the dense genotype representation, the sparse (carrier-list) representation,
or collapsed into a single per-event representative.

The classifier is pure, stateless, and deterministic: same inputs -> same
output, no I/O, no global state, no randomness. It runs once on the
materialised ``list[VariantRecord]`` plus the per-variant carrier-support
count vector.

Decision rules (see ``classify_variants`` for details):

* Structural-ish variants — copy-number, deletion / duplication / mobile-element
  / inversion classes — go to SPARSE when their carrier count is at or below
  the threshold; otherwise DENSE.
* Repeat-flagged variants (``is_repeat``) at or below the threshold also go to
  SPARSE: the repeat genotypes are noisy *and* rare, so carrier-list storage
  is both cheaper and a stronger statistical signal.
* Everything else (SNVs, common SVs, common repeats) goes to DENSE.

Event collapse is intentionally out of scope here — it lives in its own pass
(``swarm/p4-event-collapse``). The ``collapsed_representative_for`` field is
always returned empty by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from sv_pgs._typing import NDArray
from sv_pgs.data import VariantRecord

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from collections.abc import Sequence


# Variant-class string prefixes that we treat as "structural" for the purpose
# of routing rare carriers into the sparse representation. We match by prefix
# against the underlying ``VariantClass`` string value (e.g. ``deletion_short``,
# ``deletion_long``, ``duplication_short``, ``insertion_mei``, ``inversion_bnd_complex``),
# which keeps the rule stable as new sub-classes are added to the enum.
_STRUCTURAL_PREFIXES: tuple[str, ...] = (
    "deletion",
    "duplication",
    "mobile_element",
    "insertion_mei",  # mobile-element insertion in the current enum spelling
    "inversion",
)


@dataclass(slots=True)
class RoutingDecision:
    """Result of routing each variant to dense / sparse / collapsed storage.

    Attributes
    ----------
    dense_local_indices:
        ``int32`` indices into the original ``variant_records`` list for variants
        that should be carried in the dense representation.
    sparse_local_indices:
        ``int32`` indices into the original ``variant_records`` list for variants
        that should be carried in the sparse (carrier-list) representation.
    collapsed_representative_for:
        Map ``event_id -> representative variant index``. Always empty here;
        populated by the separate event-collapse pass.
    rationale_counts:
        Bookkeeping of how many variants matched each routing rule. Useful
        both for tests and for logging at pipeline startup.
    """

    dense_local_indices: NDArray
    sparse_local_indices: NDArray
    collapsed_representative_for: dict[int, int] = field(default_factory=dict)
    rationale_counts: dict[str, int] = field(default_factory=dict)


def _is_structural_class(variant_class_value: str) -> bool:
    """Return True if ``variant_class_value`` names a structural variant class.

    Matches by prefix against ``_STRUCTURAL_PREFIXES`` so that both the bare
    spec tokens (``"deletion"``, ``"duplication"``, ``"mobile_element"``,
    ``"inversion"``) and the live ``VariantClass`` enum values
    (``"deletion_short"``, ``"inversion_bnd_complex"``, etc.) route correctly.
    """
    value = variant_class_value.lower()
    return any(value.startswith(prefix) for prefix in _STRUCTURAL_PREFIXES)


def classify_variants(
    variant_records: "Sequence[VariantRecord]",
    support_counts: NDArray,
    n_samples: int,
    *,
    sparse_carrier_threshold: int | None = None,
) -> RoutingDecision:
    """Classify each variant into dense or sparse storage.

    Parameters
    ----------
    variant_records:
        The materialised list of ``VariantRecord`` objects, one per variant.
    support_counts:
        ``(n_variants,)`` integer array of carrier counts (number of samples
        carrying at least one alt allele) per variant. Must align positionally
        with ``variant_records``.
    n_samples:
        Total number of samples in the cohort. Used only to derive the default
        carrier threshold.
    sparse_carrier_threshold:
        Optional override for the carrier-count threshold below which a
        structural-ish or repeat-flagged variant is routed to sparse storage.
        Defaults to ``n_samples // 64``.

    Returns
    -------
    RoutingDecision
        Indices of dense / sparse variants and a per-rule rationale count.
    """
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative.")

    support_array = np.ascontiguousarray(support_counts)
    if support_array.ndim != 1:
        raise ValueError("support_counts must be a 1-D array.")
    if support_array.shape[0] != len(variant_records):
        raise ValueError(
            "support_counts length does not match variant_records length: "
            f"{support_array.shape[0]} vs {len(variant_records)}."
        )

    threshold = (
        int(sparse_carrier_threshold)
        if sparse_carrier_threshold is not None
        else n_samples // 64
    )

    dense_indices: list[int] = []
    sparse_indices: list[int] = []

    n_dense_snv_like = 0
    n_dense_common_structural = 0
    n_dense_common_repeat = 0
    n_sparse_rare_structural = 0
    n_sparse_rare_repeat = 0

    for variant_index, record in enumerate(variant_records):
        carrier_count = int(support_array[variant_index])
        # ``variant_class`` is a ``VariantClass(str, Enum)``, so its ``.value`` and
        # its string form are the underlying token (e.g. ``"deletion_short"``).
        class_value = str(getattr(record.variant_class, "value", record.variant_class))
        is_structural = bool(record.is_copy_number) or _is_structural_class(class_value)
        is_repeat = bool(record.is_repeat)

        if is_structural and carrier_count <= threshold:
            sparse_indices.append(variant_index)
            n_sparse_rare_structural += 1
        elif is_repeat and carrier_count <= threshold:
            sparse_indices.append(variant_index)
            n_sparse_rare_repeat += 1
        else:
            dense_indices.append(variant_index)
            if is_structural:
                n_dense_common_structural += 1
            elif is_repeat:
                n_dense_common_repeat += 1
            else:
                n_dense_snv_like += 1

    rationale_counts: dict[str, int] = {
        "threshold": threshold,
        "n_variants": len(variant_records),
        "dense_snv_like": n_dense_snv_like,
        "dense_common_structural": n_dense_common_structural,
        "dense_common_repeat": n_dense_common_repeat,
        "sparse_rare_structural": n_sparse_rare_structural,
        "sparse_rare_repeat": n_sparse_rare_repeat,
        "dense_total": len(dense_indices),
        "sparse_total": len(sparse_indices),
        "collapsed_total": 0,
    }

    return RoutingDecision(
        dense_local_indices=np.asarray(dense_indices, dtype=np.int32),
        sparse_local_indices=np.asarray(sparse_indices, dtype=np.int32),
        collapsed_representative_for={},
        rationale_counts=rationale_counts,
    )
