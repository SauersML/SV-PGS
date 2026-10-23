from __future__ import annotations

from dataclasses import dataclass

from sv_pgs._typing import F32Array, I32Array


@dataclass(slots=True)
class TieMap:
    """Maps between the full variant set and the de-duplicated (reduced) set.

    Many variants can have identical genotype patterns across all samples
    (especially in LD or when multiple callers detect the same event).
    The tie map lets us fit the model on unique columns only, then expand
    the results back to all variants.

    kept_indices: which original variants were kept as representatives
    original_to_reduced: for each original variant, its reduced-space index
    signs: each original variant's sign, +1 for a copy of its group's representative and -1 for a negated copy
    """
    kept_indices: I32Array
    original_to_reduced: I32Array
    signs: F32Array
