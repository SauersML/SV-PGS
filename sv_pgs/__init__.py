"""SV-PGS: polygenic scores with structural variants from one empirical-Bayes model."""

from __future__ import annotations

from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import VariantRecord

__all__ = [
    "ModelConfig",
    "TraitType",
    "VariantClass",
    "VariantRecord",
]
