from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from sv_pgs.config import ModelConfig, VariantClass
from sv_pgs.data import PreparedArrays, TieGroup, TieMap, VariantRecord


@dataclass(slots=True)
class Preprocessor:
    means: np.ndarray
    scales: np.ndarray

    def transform(self, genotypes: np.ndarray) -> np.ndarray:
        x = np.asarray(genotypes, dtype=np.float32)
        missing = np.isnan(x)
        if missing.any():
            x = x.copy()
            x[missing] = np.take(self.means, np.where(missing)[1])
        return (x - self.means) / self.scales


def fit_preprocessor(genotypes: np.ndarray, covariates: np.ndarray, targets: np.ndarray, config: ModelConfig) -> PreparedArrays:
    x = np.asarray(genotypes, dtype=np.float32)
    c = np.asarray(covariates, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32).reshape(-1)
    if x.ndim != 2:
        raise ValueError("genotypes must be 2D.")
    if c.ndim != 2:
        raise ValueError("covariates must be 2D.")
    if x.shape[0] != c.shape[0] or x.shape[0] != y.shape[0]:
        raise ValueError("genotypes, covariates, and targets must share sample dimension.")

    means = np.nanmean(x, axis=0, dtype=np.float64).astype(np.float32)
    means = np.where(np.isnan(means), 0.0, means)

    x_filled = x.copy()
    missing = np.isnan(x_filled)
    if missing.any():
        x_filled[missing] = np.take(means, np.where(missing)[1])

    centered = x_filled - means
    scales = np.sqrt(np.mean(centered * centered, axis=0, dtype=np.float64)).astype(np.float32)
    scales = np.where(scales < config.min_scale, 1.0, scales)
    standardized = centered / scales

    return PreparedArrays(genotypes=standardized, covariates=c, targets=y, means=means, scales=scales)


def build_tie_map(genotypes: np.ndarray, records: Sequence[VariantRecord]) -> TieMap:
    if genotypes.ndim != 2:
        raise ValueError("genotypes must be 2D.")
    if genotypes.shape[1] != len(records):
        raise ValueError("genotypes and records length mismatch.")

    signatures: dict[tuple[VariantClass, tuple[int, ...]], int] = {}
    anti_signatures: dict[tuple[VariantClass, tuple[int, ...]], int] = {}
    groups: list[TieGroup] = []
    kept: list[int] = []
    original_to_reduced = np.full(genotypes.shape[1], -1, dtype=np.int32)

    rounded = np.rint(genotypes * 1_000_000.0).astype(np.int64, copy=False)
    for variant_index, record in enumerate(records):
        signature = tuple(rounded[:, variant_index].tolist())
        key = (record.variant_class, signature)
        anti_key = (record.variant_class, tuple((-rounded[:, variant_index]).tolist()))

        if key in signatures:
            reduced_index = signatures[key]
            group = groups[reduced_index]
            group.member_indices = np.append(group.member_indices, variant_index)
            group.signs = np.append(group.signs, 1.0)
            original_to_reduced[variant_index] = reduced_index
            continue

        if key in anti_signatures:
            reduced_index = anti_signatures[key]
            group = groups[reduced_index]
            group.member_indices = np.append(group.member_indices, variant_index)
            group.signs = np.append(group.signs, -1.0)
            original_to_reduced[variant_index] = reduced_index
            continue

        reduced_index = len(groups)
        groups.append(
            TieGroup(
                representative_index=variant_index,
                member_indices=np.asarray([variant_index], dtype=np.int32),
                signs=np.asarray([1.0], dtype=np.float32),
            )
        )
        kept.append(variant_index)
        signatures[key] = reduced_index
        anti_signatures[anti_key] = reduced_index
        original_to_reduced[variant_index] = reduced_index

    return TieMap(
        kept_indices=np.asarray(kept, dtype=np.int32),
        original_to_reduced=original_to_reduced,
        reduced_to_group=groups,
    )
