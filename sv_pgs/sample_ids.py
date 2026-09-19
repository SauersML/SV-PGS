"""Sample identifiers typed by namespace.

The long-read truth half names its samples by All of Us research ID; the imputed half
by DRAGEN sequencing name. The two namespaces share strings by chance (distinct people
whose research ID and sequencing name are spelled the same), so a person is matched
across them only through the CDR crosswalk or by genotype (KING), never by name.
Comparing an ID of one namespace with an ID of the other, or with a bare string, is a
programming error and raises TypeError; that includes a set or dict holding both kinds
whose strings collide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast


@dataclass(frozen=True, slots=True, eq=False)
class _NamespacedId:
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value:
            raise ValueError(f"a {type(self).__name__} must be a non-empty string.")

    def _same_namespace_value(self, other: object) -> str:
        if type(other) is not type(self):
            raise TypeError(
                f"compared a {type(self).__name__} with a {type(other).__name__}; "
                "sample IDs of different namespaces are matched only through the crosswalk."
            )
        return cast(_NamespacedId, other).value

    def __eq__(self, other: object) -> bool:
        return self.value == self._same_namespace_value(other)

    def __lt__(self, other: object) -> bool:
        return self.value < self._same_namespace_value(other)

    def __hash__(self) -> int:
        return hash(self.value)


class ResearchId(_NamespacedId):
    """An All of Us research ID (OMOP person_id): phenotypes, covariates, the long-read half."""

    __slots__ = ()


class SequencingId(_NamespacedId):
    """A DRAGEN sequencing sample name: the imputed half's VCF columns."""

    __slots__ = ()
