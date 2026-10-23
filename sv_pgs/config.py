from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TraitType(str, Enum):
    QUANTITATIVE = "quantitative"
    BINARY = "binary"


class VariantClass(str, Enum):
    SNV = "snv"
    DELETION = "deletion"
    INSERTION = "insertion"
    DUPLICATION = "duplication"
    INSERTION_MEI = "insertion_mei"
    INVERSION_BND_COMPLEX = "inversion_bnd_complex"
    STR_VNTR_REPEAT = "str_vntr_repeat"
    OTHER_COMPLEX_SV = "other_complex_sv"
    # Integer copy number from FORMAT/CN (multi-allelic CNVs), not a GT dosage.
    COPY_NUMBER = "copy_number"
    INVERSION = "inversion"


@dataclass(slots=True)
class ModelConfig:
    """Trait likelihood and variant filters used during fitting."""

    trait_type: TraitType = TraitType.BINARY
    minimum_scale: float = 1e-6                    # variants with std < this are treated as monomorphic
    minimum_minor_allele_frequency: float = 1e-2

    def __post_init__(self) -> None:
        if self.minimum_scale <= 0.0:
            raise ValueError("minimum_scale must be positive.")
        if not 0.0 <= self.minimum_minor_allele_frequency < 0.5:
            raise ValueError("minimum_minor_allele_frequency must lie in [0.0, 0.5).")
