"""Synthetic svpgs-store v1 generator: 1kGP haplotype mosaics with imputation-realistic DS noise.

Genotypes (validation-bench REPORT P1, Tier 1).  Each sample's two haplotypes are mosaics of public
1kGP founder haplotypes (design-credit's ``src_chr*.npz``: NYGC 3202 panel, founders, MAF >= 1%):
HAPGEN-style copying with donor segments of mean 2 cM and local-ancestry tracts from an
8-generation admixture clock.  The cohort's groups are EUR, African American (0.8 AFR / 0.2 EUR),
Hispanic (0.5 AMR / 0.35 EUR / 0.15 AFR), EAS and SAS, weighted by the source panel's founder
count in each group's anchor ancestry (EUR, AFR, AMR, EAS, SAS), with per-person proportions drawn
from Dirichlet(15 q).  Each tile is one source chromosome with a fresh mosaic;
tiles cycle through the source chromosomes to reach any record count, spread over 22 synthetic
chromosomes in proportion to their hg38 lengths.

Records (target-format REPORT §1.2).  Source variants become single-path records, or the paths of
multi-path bubbles.  Every N_PATHS_TOTAL complexity bin holds an equal share of records, for every
class and frequency (a design input chosen so that every bin is exercised; no production counts are
used).  A haplotype
carries at most one path of a bubble: the first source variant in the run that it carries.  Some
bubbles also carry a nested atomic record, spread over several of their paths.

Dosage noise (tx-glimpse-math REPORT §1).
- Each path's haplotype posterior pi is one of four kinds:
  - informed: the truth;
  - uninformed: the donor ancestry's allele frequency;
  - a confident error;
  - softened by delta ~ 10**U(-5, -2).
- The per-path r2 target is the median single-record dosage r2 that bench-sim measured for the
  record's class and frequency on public data (its v7 cohort of 1kGP haplotypes, weighted by 1kGP
  founder shares, re-imputed with Beagle 5.5 against a disjoint public panel, with read-model calls at
  simple sites; v7/results_calibration_beagle_5000.json).  The confident-error rate is
  solved per record to hit it.  Read evidence sits at single-path SNVs and at single-path INDELs
  outside tandem repeats (the production pipeline's configuration: read likelihoods only at simple
  sites).  There the reads inform every haplotype, so none is uninformed and the error is all
  confident; elsewhere half of the error mass is uninformed (a design midpoint between mean-like and
  draw-like dosages).
- GLIMPSE2's err-imp then mixes in the floor: h = eps + (1 - 2 eps) pi, with eps = 1e-3.  At
  read-evidence sites pop's clamp sets eps to 1e-5.
- pop keeps each sample's top 10 paths of a bubble and odds-normalizes them.  This deflates split
  mass and zeroes the paths it drops, so the r2 loss with complexity emerges from pop's arithmetic
  rather than being imposed.
- DS is rounded to 3 decimals and stored as the uint8 code.

The per-half sidecar holds the addendum-A1 integer sums.  sum_var uses the unrounded
product-form variance of the popped haplotype probabilities.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence, get_args

import numpy as np

from sv_pgs._typing import F32Array, F64Array, I64Array, NDArray, U8Array
from sv_pgs.config import VariantClass
from sv_pgs.copy_number import allele_count_decode
from sv_pgs.dosage_store import (
    DEFAULT_INNER_CHUNK_ROWS,
    MAXIMUM_DOSAGE_MILLI,
    VARIANT_CLASS_LEGEND,
    Codec,
    CodeArrayLayout,
    CodeShardWriter,
    code_sums,
    create_code_array,
    create_column,
    dosage_array_directory,
    encode_dosage_milli,
    open_column,
    shard_rows_for,
    sites_md5,
    statistic_column_directory,
    variant_column_directory,
    write_column,
    write_manifest,
    write_variant_ids,
)
from sv_pgs.variant_typing import normalize_variant_token, structural_variant_class_from_token

HG38_AUTOSOME_MEGABASES = (
    248.96, 242.19, 198.30, 190.21, 181.54, 170.81, 159.35, 145.14, 138.39, 133.80, 135.09,
    133.28, 114.36, 107.04, 101.99, 90.34, 83.26, 80.37, 58.62, 64.44, 46.71, 50.82,
)
ANCESTRIES = ("EUR", "AFR", "AMR", "EAS", "SAS")
# (anchor ancestry, mean ancestry proportions); a group's weight is the source panel's founder
# count in its anchor ancestry.
COHORT_GROUPS: tuple[tuple[str, Mapping[str, float]], ...] = (
    ("EUR", {"EUR": 1.0}),
    ("AFR", {"AFR": 0.8, "EUR": 0.2}),
    ("AMR", {"AMR": 0.5, "EUR": 0.35, "AFR": 0.15}),
    ("EAS", {"EAS": 1.0}),
    ("SAS", {"SAS": 1.0}),
)
ANCESTRY_DIRICHLET_CONCENTRATION = 15.0
DONOR_SEGMENT_MEAN_CM = 2.0
ADMIXTURE_GENERATIONS = 8
TILE_GAP_BP = 1_000_000
TILE_GAP_CM = 50.0
ERR_IMP = 1e-3
POP_CLAMP = 1e-5
POP_KEPT_PATHS = 10
CLASS_LEGEND = ("SNV", "INDEL", "SV")
SV_CONTEXT_LEGEND = ("not_sv", "tandem_repeat", "outside_tandem_repeat")
NESTED_RECORD_PROBABILITY = 0.3
MAXIMUM_NESTED_PATHS = 5
SOFT_POSTERIOR_FRACTION = 0.1
SOFT_DELTA_LOG10_RANGE = (-5.0, -2.0)
NOISE_CLASSES = ("SNV", "INDEL", "SV_outTR", "SV_TR")
# Median single-record dosage r2 per noise class over the MAF bins (0, 0.001], (0.001, 0.01],
# (0.01, 0.05], (0.05, 0.5]: bench-sim's v7 cohort (1kGP founder-share weights), Beagle 5.5 arm, on
# public 1kGP haplotypes, all groups (v7/results_calibration_beagle_5000.json; SV_outTR is its SV
# class, SV_TR its TR class, and SNV and INDEL are its 30x read-model calls at simple sites).
MAF_BIN_EDGES = (0.001, 0.01, 0.05)
SINGLE_PATH_R2 = {
    "SNV": (0.8329993319973283, 0.9543537402545237, 0.9917454154040785, 0.9979868123448445),
    "INDEL": (0.726935085825916, 0.9306600470124883, 0.9865243070521974, 0.9967590549910882),
    "SV_outTR": (0.0022485654501617743, 0.1738066716577618, 0.6865351451998423, 0.868419151645502),
    "SV_TR": (0.06496154204265829, 0.30317890563719757, 0.6548260520832868, 0.9006202645418034),
}
# Half B is a second imputation pipeline whose r2 target sits this far below half A's for every class: a
# stated design input with no measured source, so the store carries two reliability classes.
PIPELINE_R2_LOSS = {"A": 0.0, "B": 0.01}
R2_BETA_CONCENTRATION = 20.0
COMPLEXITY_BIN_PATHS = ((1, 1), (2, 5), (6, 10), (11, 20), (21, 60))
RECORD_SINGLE = 0
RECORD_PATH = 1
RECORD_NESTED = 2
STATISTIC_DTYPES = {
    "sum_ds": np.uint64,
    "sum_ds2": np.uint64,
    "sum_var": np.int64,
    "n_het": np.uint32,
    "n_homalt": np.uint32,
    "n_ge05": np.uint32,
    "max_ds": np.uint16,
    "ds_mode_milli": np.uint16,
    "n_at_mode": np.uint32,
    "ds_min_milli": np.uint16,
    "code_mode": np.uint8,
    "n_off_mode_code": np.uint32,
    "sum_code": np.uint64,
    "sum_code2": np.uint64,
}


@dataclass(frozen=True)
class HaplotypeSource:
    """Phased founder haplotypes of public reference chromosomes, concatenated.

    ``haplotypes`` is [variants, founder haplotypes] of ALT indicators; ``chromosome_starts``
    bounds each source chromosome; ``class_codes`` index ``CLASS_LEGEND`` and ``variant_classes``
    ``tuple(VariantClass)`` (target-format REPORT §2.5 mapping).
    """

    positions: I64Array
    genetic_map_cm: F64Array
    reference_lengths: I64Array
    alternate_lengths: I64Array
    class_codes: NDArray
    variant_classes: NDArray
    tandem_repeat: NDArray
    chromosome_starts: I64Array
    haplotypes: U8Array
    haplotype_ancestry: NDArray
    ancestry_frequencies: F32Array
    pooled_frequencies: F64Array

    @classmethod
    def load(cls, paths: Sequence[Path]) -> HaplotypeSource:
        archives = [np.load(path) for path in paths]
        superpopulation = archives[0]["superpop"].astype(str)
        founder_count = int(archives[0]["n_samples"])
        for path, archive in zip(paths, archives):
            if int(archive["n_samples"]) != founder_count or not np.array_equal(archive["superpop"].astype(str), superpopulation):
                raise ValueError(f"{path} lists different founders from {paths[0]}.")
        unknown = sorted(set(superpopulation.tolist()) - set(ANCESTRIES))
        if unknown:
            raise ValueError(f"founders from superpopulations {unknown} are outside {ANCESTRIES}.")
        haplotypes = np.concatenate(
            [np.unpackbits(archive["packed_haps"], axis=1)[:, : 2 * founder_count] for archive in archives]
        )
        ancestry = np.repeat(np.array([ANCESTRIES.index(name) for name in superpopulation], dtype=np.int8), 2)
        missing = [name for index, name in enumerate(ANCESTRIES) if not np.any(ancestry == index)]
        if missing:
            raise ValueError(f"the source has no founder haplotypes for {missing}.")
        kinds = np.concatenate([archive["kinds"].astype(str) for archive in archives])
        tandem_repeat = np.concatenate([archive["in_tr"].astype(bool) for archive in archives])
        reference_lengths = np.concatenate([archive["ref_len"].astype(np.int64) for archive in archives])
        alternate_lengths = np.concatenate([archive["alt_len"].astype(np.int64) for archive in archives])
        return cls(
            positions=np.concatenate([archive["positions"].astype(np.int64) for archive in archives]),
            genetic_map_cm=np.concatenate([archive["cm"].astype(np.float64) for archive in archives]),
            reference_lengths=reference_lengths,
            alternate_lengths=alternate_lengths,
            class_codes=np.select([kinds == "SNV", kinds == "INDEL"], [0, 1], default=2).astype(np.uint8),
            variant_classes=_variant_classes(kinds, reference_lengths, alternate_lengths, tandem_repeat),
            tandem_repeat=tandem_repeat,
            chromosome_starts=np.concatenate([[0], np.cumsum([archive["positions"].shape[0] for archive in archives])]),
            haplotypes=haplotypes,
            haplotype_ancestry=ancestry,
            ancestry_frequencies=np.stack(
                [haplotypes[:, ancestry == index].mean(axis=1) for index in range(len(ANCESTRIES))], axis=1
            ).astype(np.float32),
            pooled_frequencies=haplotypes.mean(axis=1, dtype=np.float64),
        )

    @property
    def variant_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def chromosome_count(self) -> int:
        return int(self.chromosome_starts.shape[0]) - 1

    def tile_range(self, tile: int) -> tuple[int, int]:
        """Source variants of a tile: tiles cycle through the source chromosomes."""
        chromosome = tile % self.chromosome_count
        return int(self.chromosome_starts[chromosome]), int(self.chromosome_starts[chromosome + 1])


def _variant_classes(kinds: NDArray, reference_lengths: NDArray, alternate_lengths: NDArray, tandem_repeat: NDArray) -> NDArray:
    """SV-PGS classes of 1kGP records: SNVs directly; an INDEL is a deletion, an insertion or,
    with equal allele lengths, a complex event, from its allele lengths, the class its SV
    counterpart has; SVs in tandem repeats are str_vntr_repeat (target-format §2.5); every
    other SV is typed from its kind token by the shared ``variant_typing`` rule."""
    order = {variant_class: index for index, variant_class in enumerate(VariantClass)}
    classes = np.empty(kinds.shape[0], dtype=np.uint8)
    for kind in np.unique(kinds).tolist():
        members = kinds == kind
        if kind == "SNV":
            classes[members] = order[VariantClass.SNV]
            continue
        if kind == "INDEL":
            loss = reference_lengths > alternate_lengths
            gain = reference_lengths < alternate_lengths
            classes[members & loss] = order[VariantClass.DELETION]
            classes[members & gain] = order[VariantClass.INSERTION]
            classes[members & ~loss & ~gain] = order[VariantClass.OTHER_COMPLEX_SV]
            continue
        classes[members] = order[structural_variant_class_from_token(normalize_variant_token(kind))]
        classes[members & tandem_repeat] = order[VariantClass.STR_VNTR_REPEAT]
    return classes


# ---------------------------------------------------------------------------
# Record layout and per-record noise parameters
# ---------------------------------------------------------------------------


def _maf_bins(frequencies: F64Array) -> I64Array:
    minor = np.minimum(frequencies, 1.0 - frequencies)
    # Right-closed bins (a, b], as the cited measurement bins them.
    return np.minimum(np.searchsorted(MAF_BIN_EDGES, minor, side="left"), len(MAF_BIN_EDGES)).astype(np.int64)


def _bubble_start_cumulative() -> NDArray:
    """Cumulative P(unit complexity bin) for the unit starting at a variant.

    Every bin holds an equal share of records.  A bubble of mean size K_b consumes K_b records,
    so starting it with probability proportional to 1 / K_b gives each bin the same share.
    """
    weights = 1.0 / np.array([(low + high) / 2 for low, high in COMPLEXITY_BIN_PATHS])
    return np.cumsum(weights / weights.sum())


@dataclass(frozen=True)
class ChromosomeLayout:
    """Per-record structure of one synthetic chromosome.

    ``bubble_start`` is the record index of the bubble's first path (a single-path record's own
    index); ``nested_path_mask`` has bit k set when a nested record rides on the bubble's k-th path.
    """

    record_kind: NDArray
    tile: I64Array
    source_index: I64Array
    bubble_start: I64Array
    bubble_paths: I64Array
    nested_path_mask: NDArray
    has_read_evidence: NDArray
    noise_class: NDArray

    @property
    def record_count(self) -> int:
        return int(self.record_kind.shape[0])


def _tile_units(
    source: HaplotypeSource, tile: int, rng: np.random.Generator
) -> tuple[I64Array, I64Array, NDArray, NDArray]:
    """Partition one tile's source variants into units: (start, paths, nested mask, has nested)."""
    first, last = source.tile_range(tile)
    count = last - first
    cumulative = _bubble_start_cumulative()
    complexity_bin = (rng.random(count)[:, None] > cumulative[None, :]).sum(axis=1)
    complexity_bin = np.minimum(complexity_bin, len(COMPLEXITY_BIN_PATHS) - 1)
    lows = np.array([low for low, _ in COMPLEXITY_BIN_PATHS])[complexity_bin]
    highs = np.array([high for _, high in COMPLEXITY_BIN_PATHS])[complexity_bin]
    sizes = np.minimum(rng.integers(lows, highs + 1), count - np.arange(count))
    starts = []
    cursor = 0
    size_list = sizes.tolist()
    while cursor < count:
        starts.append(cursor)
        cursor += size_list[cursor]
    unit_offsets = np.array(starts, dtype=np.int64)
    unit_starts = first + unit_offsets
    unit_paths = sizes[unit_offsets].astype(np.int64)
    has_nested = (unit_paths >= 3) & (rng.random(unit_starts.size) < NESTED_RECORD_PROBABILITY)
    nested_masks = np.zeros(unit_starts.size, dtype=np.uint64)
    nested_units = np.flatnonzero(has_nested)
    if nested_units.size:
        width = int(COMPLEXITY_BIN_PATHS[-1][1])
        keys = rng.random((nested_units.size, width))
        keys[np.arange(width)[None, :] >= unit_paths[nested_units][:, None]] = np.inf
        ranks = np.argsort(np.argsort(keys, axis=1), axis=1)
        carried_counts = rng.integers(2, np.minimum(unit_paths[nested_units] - 1, MAXIMUM_NESTED_PATHS) + 1)
        carried = ranks < carried_counts[:, None]
        bits = np.left_shift(np.uint64(1), np.arange(width, dtype=np.uint64))
        nested_masks[nested_units] = (carried * bits[None, :]).sum(axis=1, dtype=np.uint64)
    return unit_starts, unit_paths, nested_masks, has_nested


def _popcount(masks: NDArray) -> I64Array:
    return np.unpackbits(masks.astype("<u8").view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1).astype(np.int64)


def lay_out_chromosome(
    source: HaplotypeSource,
    record_count: int,
    shard_rows: int,
    rng: np.random.Generator,
) -> ChromosomeLayout:
    """Tile the source into single-path records and bubbles until ``record_count`` records.

    A nested record takes the position of its bubble's last path, so rows stay in position order.
    A bubble cut by a shard boundary or by the chromosome end is split into single-path records
    (its nested record becomes one more single-path record of the last path's variant), so every
    shard is generated independently.
    """
    pieces: list[dict[str, NDArray]] = []
    produced = 0
    tile = 0
    while produced < record_count:
        unit_starts, unit_paths, nested_masks, has_nested = _tile_units(source, tile, rng)
        records_per_unit = unit_paths + has_nested
        unit_first_record = produced + np.concatenate([[0], np.cumsum(records_per_unit)[:-1]])
        unit_of_record = np.repeat(np.arange(unit_starts.size), records_per_unit)
        offset = np.arange(unit_of_record.size) - np.repeat(unit_first_record - produced, records_per_unit)
        paths = unit_paths[unit_of_record]
        nested = offset == paths
        kind = np.where(paths == 1, RECORD_SINGLE, np.where(nested, RECORD_NESTED, RECORD_PATH)).astype(np.uint8)
        masks = np.where(nested, nested_masks[unit_of_record], np.uint64(0)).astype(np.uint64)
        source_index = unit_starts[unit_of_record] + np.minimum(offset, paths - 1)
        pieces.append(
            {
                "kind": kind,
                "tile": np.full(kind.size, tile, dtype=np.int64),
                "source": source_index,
                "bubble_start": unit_first_record[unit_of_record],
                "bubble_records": records_per_unit[unit_of_record],
                "paths": paths,
                "mask": masks,
            }
        )
        produced += kind.size
        tile += 1
    merged = {name: np.concatenate([piece[name] for piece in pieces])[:record_count] for name in pieces[0]}
    bubble_start = merged["bubble_start"]
    bubble_stop = bubble_start + merged["bubble_records"]
    broken = (bubble_stop > record_count) | (bubble_start // shard_rows != (bubble_stop - 1) // shard_rows)
    rows = np.flatnonzero(broken & (merged["paths"] > 1))
    merged["kind"][rows] = RECORD_SINGLE
    merged["bubble_start"][rows] = rows
    merged["paths"][rows] = 1
    merged["mask"][rows] = 0
    classes = source.class_codes[merged["source"]]
    sv_code = CLASS_LEGEND.index("SV")
    noise_class = np.where(
        classes == sv_code,
        np.where(source.tandem_repeat[merged["source"]], NOISE_CLASSES.index("SV_TR"), NOISE_CLASSES.index("SV_outTR")),
        classes,
    ).astype(np.uint8)
    simple_site = (classes == CLASS_LEGEND.index("SNV")) | (
        (classes == CLASS_LEGEND.index("INDEL")) & ~source.tandem_repeat[merged["source"]]
    )
    has_read_evidence = (merged["kind"] == RECORD_SINGLE) & simple_site
    return ChromosomeLayout(
        record_kind=merged["kind"],
        tile=merged["tile"],
        source_index=merged["source"],
        bubble_start=merged["bubble_start"],
        bubble_paths=merged["paths"],
        nested_path_mask=merged["mask"],
        has_read_evidence=has_read_evidence,
        noise_class=noise_class,
    )


_SOFT_DELTA_MEAN = (10 ** SOFT_DELTA_LOG10_RANGE[1] - 10 ** SOFT_DELTA_LOG10_RANGE[0]) / (
    (SOFT_DELTA_LOG10_RANGE[1] - SOFT_DELTA_LOG10_RANGE[0]) * np.log(10)
)
_SOFT_DELTA_SECOND_MOMENT = (10 ** (2 * SOFT_DELTA_LOG10_RANGE[1]) - 10 ** (2 * SOFT_DELTA_LOG10_RANGE[0])) / (
    2 * (SOFT_DELTA_LOG10_RANGE[1] - SOFT_DELTA_LOG10_RANGE[0]) * np.log(10)
)


def _flip_rates(error_rate: F64Array, frequency: F64Array, ceiling: F64Array) -> tuple[F64Array, F64Array]:
    """Balanced confident errors: carriers lose c/(2f), non-carriers gain c/(2(1 - f))."""
    return np.minimum(error_rate / (2 * frequency), ceiling), np.minimum(error_rate / (2 * (1 - frequency)), ceiling)


def haplotype_r2(
    uninformed: F64Array,
    error_rate: F64Array,
    soft: F64Array,
    frequency: F64Array,
) -> F64Array:
    """Exact r2 between a haplotype's posterior pi and its allele under the four-part mixture.

    The err-imp floor is affine in pi, so it leaves r2 unchanged.
    """
    ceiling = 1 - uninformed - soft
    carrier_flip, noncarrier_flip = _flip_rates(error_rate, frequency, ceiling)
    informed_carrier = 1 - uninformed - carrier_flip - soft
    carrier_mean = uninformed * frequency + soft * (1 - _SOFT_DELTA_MEAN) + informed_carrier
    carrier_square = (
        uninformed * frequency**2 + soft * (1 - 2 * _SOFT_DELTA_MEAN + _SOFT_DELTA_SECOND_MOMENT) + informed_carrier
    )
    noncarrier_mean = uninformed * frequency + noncarrier_flip + soft * _SOFT_DELTA_MEAN
    noncarrier_square = uninformed * frequency**2 + noncarrier_flip + soft * _SOFT_DELTA_SECOND_MOMENT
    mean = frequency * carrier_mean + (1 - frequency) * noncarrier_mean
    variance = frequency * carrier_square + (1 - frequency) * noncarrier_square - mean**2
    covariance = frequency * (1 - frequency) * (carrier_mean - noncarrier_mean)
    return covariance**2 / (variance * frequency * (1 - frequency))


def solve_error_rate(target: F64Array, uninformed: F64Array, soft: F64Array, frequency: F64Array) -> F64Array:
    """Bisect, to float64 precision, the confident-error rate that brings haplotype_r2 down to
    ``target`` (0 if already below)."""
    low = np.zeros_like(target)
    high = 2 * np.minimum(frequency, 1 - frequency) * (1 - uninformed - soft)
    high = np.where(haplotype_r2(uninformed, low, soft, frequency) > target, high, 0.0)
    while True:
        middle = 0.5 * (low + high)
        # Stop where the interval is within one rounding of its upper end or has no float inside.
        open_interval = (middle > low) & (middle < high) & (high - low > np.finfo(np.float64).eps * high)
        if not open_interval.any():
            return middle
        too_accurate = haplotype_r2(uninformed, middle, soft, frequency) > target
        low = np.where(too_accurate, middle, low)
        high = np.where(too_accurate, high, middle)


@dataclass(frozen=True)
class NoiseParameters:
    """Per-record haplotype posterior mixture weights for one pipeline."""

    uninformed: F32Array
    carrier_flip: F32Array
    noncarrier_flip: F32Array
    soft: F32Array
    floor: F32Array

    def rows(self, records: slice) -> NoiseParameters:
        return NoiseParameters(
            uninformed=self.uninformed[records],
            carrier_flip=self.carrier_flip[records],
            noncarrier_flip=self.noncarrier_flip[records],
            soft=self.soft[records],
            floor=self.floor[records],
        )


def noise_parameters(
    layout: ChromosomeLayout,
    source: HaplotypeSource,
    pipeline: str,
    rng: np.random.Generator,
) -> NoiseParameters:
    # The founder panel resolves frequencies in steps of one haplotype.
    resolution = 1.0 / source.haplotypes.shape[1]
    frequency = np.clip(source.pooled_frequencies[layout.source_index], resolution, 1 - resolution)
    maf_bin = _maf_bins(frequency)
    noise_class = layout.noise_class
    target_mean = np.array([SINGLE_PATH_R2[name] for name in NOISE_CLASSES])[noise_class, maf_bin]
    target = rng.beta(target_mean * R2_BETA_CONCENTRATION, (1 - target_mean) * R2_BETA_CONCENTRATION)
    # r2 is a squared correlation: the pipeline loss can only take it down to 0.
    target = np.clip(target - PIPELINE_R2_LOSS[pipeline], 0.0, 1.0)
    uninformed = np.where(layout.has_read_evidence, 0.0, 0.5 * (1 - target))
    soft = np.minimum(SOFT_POSTERIOR_FRACTION * np.minimum(1.0, 10 * np.minimum(frequency, 1 - frequency)), 1 - uninformed)
    error_rate = solve_error_rate(target, uninformed, soft, frequency)
    carrier_flip, noncarrier_flip = _flip_rates(error_rate, frequency, 1 - uninformed - soft)
    floor = np.where(layout.has_read_evidence, POP_CLAMP, ERR_IMP)
    return NoiseParameters(
        uninformed=uninformed.astype(np.float32),
        carrier_flip=carrier_flip.astype(np.float32),
        noncarrier_flip=noncarrier_flip.astype(np.float32),
        soft=soft.astype(np.float32),
        floor=floor.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Mosaic haplotypes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cohort:
    """Per-haplotype ancestry proportions (2 haplotypes per sample, samples in store order)."""

    haplotype_proportions: F64Array

    @property
    def haplotype_count(self) -> int:
        return int(self.haplotype_proportions.shape[0])


def draw_cohort(sample_count: int, source: HaplotypeSource, rng: np.random.Generator) -> Cohort:
    founder_haplotypes = np.bincount(source.haplotype_ancestry, minlength=len(ANCESTRIES))
    group_weights = np.array([founder_haplotypes[ANCESTRIES.index(anchor)] for anchor, _ in COHORT_GROUPS], dtype=np.float64)
    groups = rng.choice(len(COHORT_GROUPS), size=sample_count, p=group_weights / group_weights.sum())
    proportions = np.zeros((sample_count, len(ANCESTRIES)))
    for group_index, (_, base) in enumerate(COHORT_GROUPS):
        members = np.flatnonzero(groups == group_index)
        base_vector = np.array([base.get(name, 0.0) for name in ANCESTRIES])
        if np.count_nonzero(base_vector) == 1:
            proportions[members] = base_vector
            continue
        present = base_vector > 0
        draws = rng.dirichlet(ANCESTRY_DIRICHLET_CONCENTRATION * base_vector[present], size=members.size)
        proportions[np.ix_(members, np.flatnonzero(present))] = draws
    return Cohort(haplotype_proportions=np.repeat(proportions, 2, axis=0))


@dataclass(frozen=True)
class TileMosaic:
    """Donor segments of every cohort haplotype over one tile, in source-variant coordinates.

    ``segment_starts[h, k]`` is the first source variant of haplotype h's k-th segment (column 0 is
    the tile's first variant; padding is past the tile), ``donors`` the source haplotype copied and
    ``ancestries`` its ancestry index.
    """

    segment_starts: NDArray
    donors: NDArray
    ancestries: NDArray


def draw_tile_mosaic(source: HaplotypeSource, tile: int, cohort: Cohort, rng: np.random.Generator) -> TileMosaic:
    """Superpose donor switches (1 per 2 cM) and ancestry switches (T per Morgan); redraw ancestry at the latter."""
    first, last = source.tile_range(tile)
    genetic_map = source.genetic_map_cm[first:last]
    span = float(genetic_map[-1] - genetic_map[0])
    ancestry_rate = ADMIXTURE_GENERATIONS / 100.0
    total_rate = 1.0 / DONOR_SEGMENT_MEAN_CM + ancestry_rate
    haplotype_count = cohort.haplotype_count
    # A Poisson process on [0, span): a Poisson count, then that many sorted uniform positions.
    # Columns past a haplotype's count sit at infinity, past the tile.
    counts = rng.poisson(span * total_rate, size=haplotype_count)
    breakpoint_columns = int(counts.max(initial=0))
    positions = rng.random((haplotype_count, breakpoint_columns)) * span
    positions[np.arange(breakpoint_columns)[None, :] >= counts[:, None]] = np.inf
    breakpoints = np.sort(positions, axis=1)
    cumulative = np.cumsum(cohort.haplotype_proportions, axis=1)
    columns = breakpoint_columns + 1
    candidate = np.zeros((haplotype_count, columns), dtype=np.int8)
    uniforms = rng.random((haplotype_count, columns))
    for ancestry in range(len(ANCESTRIES) - 1):
        candidate += uniforms > cumulative[:, ancestry : ancestry + 1]
    switches = np.ones((haplotype_count, columns), dtype=bool)
    switches[:, 1:] = rng.random((haplotype_count, breakpoint_columns)) < ancestry_rate / total_rate
    last_switch = np.maximum.accumulate(np.where(switches, np.arange(columns)[None, :], 0), axis=1)
    ancestries = np.take_along_axis(candidate, last_switch, axis=1)
    pool_members = [np.flatnonzero(source.haplotype_ancestry == index) for index in range(len(ANCESTRIES))]
    pool_sizes = np.array([members.size for members in pool_members])
    pool_offsets = np.concatenate([[0], np.cumsum(pool_sizes)[:-1]])
    pooled = np.concatenate(pool_members)
    picks = (rng.random((haplotype_count, columns)) * pool_sizes[ancestries]).astype(np.int64)
    donors = pooled[pool_offsets[ancestries] + picks].astype(np.int32)
    starts = np.empty((haplotype_count, columns + 1), dtype=np.int64)
    starts[:, 0] = first
    starts[:, 1:-1] = first + np.searchsorted(genetic_map, genetic_map[0] + breakpoints)
    starts[:, -1] = last + 1
    return TileMosaic(segment_starts=starts, donors=donors, ancestries=ancestries)


def mosaic_truth(source: HaplotypeSource, mosaic: TileMosaic, source_start: int, source_stop: int) -> tuple[U8Array, NDArray]:
    """True alleles [source_stop - source_start, haplotypes] and each haplotype's ancestry at the start."""
    segment = (mosaic.segment_starts <= source_start).sum(axis=1) - 1
    rows = np.arange(mosaic.segment_starts.shape[0])
    donor = mosaic.donors[rows, segment]
    truth = np.take(source.haplotypes[source_start:source_stop], donor, axis=1)
    next_start = mosaic.segment_starts[rows, segment + 1]
    while True:
        split = np.flatnonzero(next_start < source_stop)
        if split.size == 0:
            return truth, mosaic.ancestries[rows, segment]
        segment[split] += 1
        boundary = next_start[split]
        replacement = mosaic.donors[split, segment[split]]
        offsets = np.arange(source_stop - source_start)[:, None]
        after = offsets >= (boundary - source_start)[None, :]
        switched = source.haplotypes[source_start + offsets, replacement[None, :]]
        truth[:, split] = np.where(after, switched, truth[:, split])
        next_start[split] = mosaic.segment_starts[split, segment[split] + 1]


# ---------------------------------------------------------------------------
# Posteriors, pop, and per-half statistics
# ---------------------------------------------------------------------------


def haplotype_posteriors(
    truth: U8Array,
    parameters: NoiseParameters,
    ancestry_frequency: F32Array,
    haplotype_ancestry: NDArray,
    rng: np.random.Generator,
) -> F32Array:
    """GLIMPSE2 haplotype posteriors h = eps + (1 - 2 eps) pi, clamped as pop does.

    One uniform per haplotype picks its kind by thresholds (uninformed below u, then confident
    error, then softened, else informed) and also sets the softening delta, so each entry costs
    a single draw.  ``parameters`` hold one row per truth row; ``ancestry_frequency`` is
    [rows, ancestries].
    """
    column_count = truth.shape[1]
    draw = rng.random(truth.shape, dtype=np.float32)
    uninformed = parameters.uninformed[:, None]
    flip_upper = np.where(truth, parameters.carrier_flip[:, None], parameters.noncarrier_flip[:, None])
    flip_upper += uninformed
    posterior = truth.astype(np.float32)
    flat_posterior = posterior.reshape(-1)
    flat_draw = draw.reshape(-1)
    flipped = np.flatnonzero(((draw >= uninformed) & (draw < flip_upper)).reshape(-1))
    flat_posterior[flipped] = 1.0 - flat_posterior[flipped]
    softened = np.flatnonzero(((draw >= flip_upper) & (draw < flip_upper + parameters.soft[:, None])).reshape(-1))
    fraction = (flat_draw[softened] - flip_upper.reshape(-1)[softened]) / parameters.soft[softened // column_count]
    delta = np.power(
        np.float32(10.0),
        SOFT_DELTA_LOG10_RANGE[0] + (SOFT_DELTA_LOG10_RANGE[1] - SOFT_DELTA_LOG10_RANGE[0]) * fraction,
    )
    flat_posterior[softened] = np.abs(flat_posterior[softened] - delta)
    uninformed_entries = np.flatnonzero((draw < uninformed).reshape(-1))
    flat_posterior[uninformed_entries] = ancestry_frequency[
        uninformed_entries // column_count, haplotype_ancestry[uninformed_entries % column_count]
    ]
    floor = parameters.floor[:, None]
    posterior *= 1.0 - 2.0 * floor
    posterior += floor
    np.clip(posterior, POP_CLAMP, 1.0 - POP_CLAMP, out=posterior)
    return posterior


def pop_bubble(path_posteriors: F32Array, nested_masks: Sequence[int]) -> F32Array:
    """pop-glimpse2 rule A for one bubble: per sample keep the top-10 paths by p0 + p1, then
    q = w / (1 + sum w) per haplotype with w = p / (1 - p).  Returns haplotype probabilities of the
    path records followed by one row per nested record (the sum over its carried paths).
    """
    path_count = path_posteriors.shape[0]
    odds = path_posteriors / (1.0 - path_posteriors)
    if path_count > POP_KEPT_PATHS:
        score = path_posteriors[:, 0::2] + path_posteriors[:, 1::2]
        order = np.argsort(-score, axis=0, kind="stable")
        dropped = np.zeros(score.shape, dtype=bool)
        np.put_along_axis(dropped, order[POP_KEPT_PATHS:], True, axis=0)
        odds[:, 0::2][dropped] = 0.0
        odds[:, 1::2][dropped] = 0.0
    kept = odds / (1.0 + odds.sum(axis=0, keepdims=True))
    nested_rows = [
        kept[[path for path in range(path_count) if mask >> path & 1]].sum(axis=0) for mask in nested_masks
    ]
    return np.vstack([kept, *nested_rows]) if nested_rows else kept


def _row_modes(values: NDArray, bins: int) -> tuple[I64Array, I64Array]:
    """Most frequent value of each row (the smallest on ties) and its count, from one bincount."""
    row_count = values.shape[0]
    offsets = (np.arange(row_count, dtype=np.int64) * bins)[:, None]
    histogram = np.bincount((values + offsets).reshape(-1), minlength=row_count * bins).reshape(row_count, bins)
    modes = histogram.argmax(axis=1)
    return modes, histogram[np.arange(row_count), modes]


def half_statistics(haplotype_probabilities: F32Array) -> tuple[U8Array, dict[str, NDArray]]:
    """Codes and addendum-A1 sums for records x (2 * samples) popped haplotype probabilities.

    ``sum_var`` sums the product-form variance p0(1 - p0) + p1(1 - p1) over samples before its
    single rounding to millionths.
    """
    first = haplotype_probabilities[:, 0::2]
    second = haplotype_probabilities[:, 1::2]
    dosage_milli = np.rint((first + second) * 1000.0).astype(np.int32)
    codes = encode_dosage_milli(dosage_milli)
    probability_sums = np.add.reduce(haplotype_probabilities, axis=1, dtype=np.float64)
    probability_squares = np.einsum("ij,ij->i", haplotype_probabilities, haplotype_probabilities, dtype=np.float64)
    alt_calls = (first > 0.5).view(np.int8) + (second > 0.5).view(np.int8)
    ds_modes, ds_mode_counts = _row_modes(dosage_milli, MAXIMUM_DOSAGE_MILLI + 1)
    code_modes, code_mode_counts = _row_modes(codes, 256)
    code_sum, code_square_sum = code_sums(codes)
    statistics = {
        "sum_ds": np.add.reduce(dosage_milli, axis=1, dtype=np.int64),
        "sum_ds2": np.add.reduce(dosage_milli * dosage_milli, axis=1, dtype=np.int64),
        "sum_var": np.rint((probability_sums - probability_squares) * 1e6).astype(np.int64),
        "n_het": np.count_nonzero(alt_calls == 1, axis=1),
        "n_homalt": np.count_nonzero(alt_calls == 2, axis=1),
        "n_ge05": np.count_nonzero(dosage_milli >= 500, axis=1),
        "max_ds": dosage_milli.max(axis=1),
        "ds_mode_milli": ds_modes,
        "n_at_mode": ds_mode_counts,
        "ds_min_milli": dosage_milli.min(axis=1),
        "code_mode": code_modes,
        "n_off_mode_code": codes.shape[1] - code_mode_counts,
        "sum_code": code_sum,
        "sum_code2": code_square_sum,
    }
    return codes, statistics


# ---------------------------------------------------------------------------
# Store generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationPlan:
    root: Path
    source: HaplotypeSource
    cohort: Cohort
    half_sample_counts: tuple[int, ...]
    half_pipelines: tuple[str, ...]
    chromosomes: tuple[str, ...]
    layouts: tuple[ChromosomeLayout, ...]
    noise: tuple[dict[str, NoiseParameters], ...]
    code_layouts: tuple[tuple[CodeArrayLayout, ...], ...]
    seed: int
    shard_rows: int
    block_records: int

    @property
    def half_haplotype_starts(self) -> I64Array:
        return np.concatenate([[0], 2 * np.cumsum(self.half_sample_counts)]).astype(np.int64)


def _generator(seed: int, *keys: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=tuple(keys)))


def block_boundaries(layout: ChromosomeLayout, record_start: int, record_stop: int, block_records: int) -> list[tuple[int, int]]:
    """Split [record_start, record_stop) into blocks of about ``block_records`` that keep bubbles and tiles whole."""
    indices = np.arange(record_start, record_stop)
    opens_unit = layout.bubble_start[record_start:record_stop] == indices
    opens_tile = np.ones(indices.size, dtype=bool)
    opens_tile[1:] = layout.tile[record_start + 1 : record_stop] != layout.tile[record_start : record_stop - 1]
    blocks = []
    cursor = 0
    while cursor < indices.size:
        stop = min(indices.size, cursor + block_records)
        tile_change = np.flatnonzero(opens_tile[cursor + 1 : stop])
        if tile_change.size:
            stop = cursor + 1 + int(tile_change[0])
        while stop < indices.size and not opens_unit[stop]:
            stop -= 1
        if stop == cursor:
            stop = cursor + 1 + int(np.argmax(opens_unit[cursor + 1 :])) if np.any(opens_unit[cursor + 1 :]) else indices.size
        blocks.append((record_start + cursor, record_start + stop))
        cursor = stop
    return blocks


def _bubbles(layout: ChromosomeLayout, record_start: int, record_stop: int) -> list[tuple[int, int, list[int]]]:
    """(first record, path count, nested masks) of every multi-path bubble inside a block."""
    bubbles = []
    kinds = layout.record_kind[record_start:record_stop]
    for first in np.flatnonzero((kinds == RECORD_PATH) & (layout.bubble_start[record_start:record_stop] == np.arange(record_start, record_stop))):
        first_record = record_start + int(first)
        path_count = int(layout.bubble_paths[first_record])
        nested_record = first_record + path_count
        masks = (
            [int(layout.nested_path_mask[nested_record])]
            if nested_record < record_stop and layout.record_kind[nested_record] == RECORD_NESTED
            else []
        )
        bubbles.append((first_record, path_count, masks))
    return bubbles


def generate_block(
    plan: GenerationPlan,
    chromosome_index: int,
    record_start: int,
    record_stop: int,
    mosaic: TileMosaic,
) -> tuple[U8Array, list[F32Array]]:
    """Truth genotypes [records, haplotypes] and per-half popped haplotype probabilities of a block."""
    layout = plan.layouts[chromosome_index]
    source = plan.source
    source_index = layout.source_index[record_start:record_stop]
    source_start = int(source_index.min())
    source_truth, haplotype_ancestry = mosaic_truth(source, mosaic, source_start, int(source_index.max()) + 1)
    truth = source_truth[source_index - source_start]
    bubbles = _bubbles(layout, record_start, record_stop)
    for first_record, path_count, masks in bubbles:
        paths = truth[first_record - record_start : first_record - record_start + path_count]
        carried_earlier = np.zeros(paths.shape[1], dtype=np.uint8)
        for path in range(path_count):
            exclusive = paths[path] & (1 - carried_earlier)
            carried_earlier |= paths[path]
            paths[path] = exclusive
        for nested_offset, mask in enumerate(masks):
            carried = [path for path in range(path_count) if mask >> path & 1]
            truth[first_record - record_start + path_count + nested_offset] = np.bitwise_or.reduce(paths[carried], axis=0)
    rng = _generator(plan.seed, 2, chromosome_index, record_start)
    ancestry_frequency = source.ancestry_frequencies[source_index]
    starts = plan.half_haplotype_starts
    records = slice(record_start, record_stop)
    popped_by_half = []
    for half_position, pipeline in enumerate(plan.half_pipelines):
        columns = slice(int(starts[half_position]), int(starts[half_position + 1]))
        probabilities = haplotype_posteriors(
            truth[:, columns],
            plan.noise[chromosome_index][pipeline].rows(records),
            ancestry_frequency,
            haplotype_ancestry[columns],
            rng,
        )
        for first_record, path_count, masks in bubbles:
            local_first = first_record - record_start
            popped = pop_bubble(probabilities[local_first : local_first + path_count], masks)
            probabilities[local_first : local_first + popped.shape[0]] = popped
        popped_by_half.append(probabilities)
    return truth, popped_by_half


def _generate_shard(plan: GenerationPlan, chromosome_index: int, shard_index: int) -> None:
    chromosome = plan.chromosomes[chromosome_index]
    layout = plan.layouts[chromosome_index]
    record_start = shard_index * plan.shard_rows
    record_stop = min(layout.record_count, record_start + plan.shard_rows)
    writers = [
        CodeShardWriter(dosage_array_directory(plan.root, half_index, chromosome), code_layouts[chromosome_index], shard_index)
        for half_index, code_layouts in enumerate(plan.code_layouts)
    ]
    statistic_columns = [
        {
            name: open_column(statistic_column_directory(plan.root, half_index, chromosome, name), writable=True)[0]
            for name in STATISTIC_DTYPES
        }
        for half_index in range(len(plan.half_sample_counts))
    ]
    mosaic_tile = -1
    for block_start, block_stop in block_boundaries(layout, record_start, record_stop, plan.block_records):
        tile = int(layout.tile[block_start])
        if tile != mosaic_tile:
            mosaic = draw_tile_mosaic(plan.source, tile, plan.cohort, _generator(plan.seed, 1, chromosome_index, tile))
            mosaic_tile = tile
        _, popped_by_half = generate_block(plan, chromosome_index, block_start, block_stop, mosaic)
        for writer, columns, probabilities in zip(writers, statistic_columns, popped_by_half):
            codes, statistics = half_statistics(probabilities)
            writer.write_rows(codes)
            for name, values in statistics.items():
                columns[name][block_start:block_stop] = values
    for writer in writers:
        writer.close()
    for columns in statistic_columns:
        for column in columns.values():
            column.flush()


_WORKER_PLAN: list[GenerationPlan] = []


def _run_shard(task: tuple[int, int]) -> tuple[int, int, float]:
    started = time.perf_counter()
    _generate_shard(_WORKER_PLAN[0], *task)
    return task[0], task[1], time.perf_counter() - started


def _record_counts(total_records: int, chromosome_count: int) -> list[int]:
    lengths = np.array(HG38_AUTOSOME_MEGABASES[:chromosome_count])
    counts = np.floor(total_records * lengths / lengths.sum()).astype(np.int64)
    counts[: total_records - int(counts.sum())] += 1
    return [int(count) for count in counts]


def plan_store(
    root: Path,
    source: HaplotypeSource,
    *,
    half_sample_counts: Sequence[int],
    half_pipelines: Sequence[str],
    total_records: int,
    chromosome_count: int,
    seed: int,
    block_records: int,
    codec: Codec,
    shard_rows: int | None = None,
    inner_rows: int = DEFAULT_INNER_CHUNK_ROWS,
    parallel_writers: int = 1,
) -> GenerationPlan:
    """Draw the cohort, lay out every chromosome, solve noise parameters and write all metadata.

    Each (chromosome, shard) is one generation task, and a bubble a shard boundary cuts becomes
    single-path records, so ``shard_rows`` defaults to ``dosage_store.shard_rows_for``: the fewest
    shards that still give ``parallel_writers`` tasks.
    """
    if len(half_sample_counts) != len(half_pipelines) or not set(half_pipelines) <= set(PIPELINE_R2_LOSS):
        raise ValueError(f"each half needs a pipeline in {sorted(PIPELINE_R2_LOSS)}.")
    chromosomes = tuple(f"chr{index + 1}" for index in range(chromosome_count))
    record_counts = _record_counts(total_records, chromosome_count)
    if shard_rows is None:
        shard_rows = shard_rows_for(
            record_counts, inner_rows=inner_rows, parallel_writers=parallel_writers, arrays_per_count=len(half_sample_counts)
        )
    cohort = draw_cohort(int(sum(half_sample_counts)), source, _generator(seed, 0))
    layouts = []
    noise = []
    digests = []
    for chromosome_index, (chromosome, record_count) in enumerate(zip(chromosomes, record_counts)):
        layout_rng = _generator(seed, 3, chromosome_index)
        layout = lay_out_chromosome(source, record_count, shard_rows, layout_rng)
        layouts.append(layout)
        noise.append({pipeline: noise_parameters(layout, source, pipeline, layout_rng) for pipeline in sorted(set(half_pipelines))})
        digests.append(_write_variant_table(root, chromosome, layout, source))
        for half_index in range(len(half_sample_counts)):
            for name, dtype in STATISTIC_DTYPES.items():
                create_column(statistic_column_directory(root, half_index, chromosome, name), dtype, record_count).flush()
    code_layouts = tuple(
        tuple(
            create_code_array(
                dosage_array_directory(root, half_index, chromosome),
                record_count,
                sample_count,
                codec=codec,
                shard_rows=shard_rows,
                inner_rows=inner_rows,
            )
            for chromosome, record_count in zip(chromosomes, record_counts)
        )
        for half_index, sample_count in enumerate(half_sample_counts)
    )
    write_manifest(
        root,
        chromosomes=chromosomes,
        record_counts=record_counts,
        half_sample_counts=half_sample_counts,
        chromosome_sites_md5=digests,
        attributes={
            "synthetic": {
                "generator": "sv_pgs.synthetic_store",
                "seed": seed,
                "half_pipelines": list(half_pipelines),
                "codec": codec,
                "source_variants": source.variant_count,
                "source_haplotypes": int(source.haplotypes.shape[1]),
            }
        },
    )
    return GenerationPlan(
        root=root,
        source=source,
        cohort=cohort,
        half_sample_counts=tuple(int(count) for count in half_sample_counts),
        half_pipelines=tuple(half_pipelines),
        chromosomes=chromosomes,
        layouts=tuple(layouts),
        noise=tuple(noise),
        code_layouts=code_layouts,
        seed=seed,
        shard_rows=shard_rows,
        block_records=block_records,
    )


def _write_variant_table(root: Path, chromosome: str, layout: ChromosomeLayout, source: HaplotypeSource) -> str:
    """Write the chromosome's variant columns and ids; returns its sites md5.

    Tiles are laid end to end, each shifted past the previous one by its span plus a gap, in
    both bp and cM.
    """
    tile_count = int(layout.tile[-1]) + 1
    first_rows = np.array([source.tile_range(tile)[0] for tile in range(tile_count)])
    last_rows = np.array([source.tile_range(tile)[1] - 1 for tile in range(tile_count)])
    span_bp = source.positions[last_rows] - source.positions[first_rows] + TILE_GAP_BP
    span_cm = source.genetic_map_cm[last_rows] - source.genetic_map_cm[first_rows] + TILE_GAP_CM
    offset_bp = np.concatenate([[0], np.cumsum(span_bp)[:-1]])
    offset_cm = np.concatenate([[0.0], np.cumsum(span_cm)[:-1]])
    source_index = layout.source_index
    tile_first = first_rows[layout.tile]
    positions = offset_bp[layout.tile] + source.positions[source_index] - source.positions[tile_first] + 1
    if int(positions.max()) >= 2**31:
        raise ValueError(f"{chromosome} spans {int(positions.max())} bp, past int32 positions; use more chromosomes.")
    genetic_map = offset_cm[layout.tile] + source.genetic_map_cm[source_index] - source.genetic_map_cm[tile_first]
    context = np.zeros(layout.record_count, dtype=np.uint8)
    context[layout.noise_class == NOISE_CLASSES.index("SV_TR")] = SV_CONTEXT_LEGEND.index("tandem_repeat")
    context[layout.noise_class == NOISE_CLASSES.index("SV_outTR")] = SV_CONTEXT_LEGEND.index("outside_tandem_repeat")
    carried_paths = np.where(layout.record_kind == RECORD_NESTED, _popcount(layout.nested_path_mask), 1)
    reference_lengths = source.reference_lengths[source_index].astype(np.int32)
    alternate_lengths = source.alternate_lengths[source_index].astype(np.int32)
    # Every synthetic record is an ALT count: its value is code / 127.
    codes_per_unit, value_origin = allele_count_decode(positions.shape[0])
    columns: dict[str, tuple[NDArray, dict[str, Any]]] = {
        "pos": (positions.astype(np.int32), {}),
        "ref_len": (reference_lengths, {}),
        "alt_len": (alternate_lengths, {}),
        "cm": (genetic_map, {}),
        "variant_class": (source.variant_classes[source_index], {"legend": VARIANT_CLASS_LEGEND}),
        "codes_per_unit": (codes_per_unit, {}),
        "value_origin": (value_origin.astype(np.int16), {}),
        "group_first": (layout.bubble_start, {}),
        "class": (source.class_codes[source_index], {"legend": list(CLASS_LEGEND)}),
        "sv_ctx": (context, {"legend": list(SV_CONTEXT_LEGEND)}),
        "n_paths": (carried_paths.astype(np.uint16), {}),
        "n_paths_total": (layout.bubble_paths.astype(np.uint16), {}),
        "topk_truncated": (layout.bubble_paths > POP_KEPT_PATHS, {}),
        "has_pl": (layout.has_read_evidence, {}),
        "panel_af": (source.pooled_frequencies[source_index].astype(np.float32), {}),
    }
    for name, (values, attributes) in columns.items():
        write_column(variant_column_directory(root, chromosome, name), values, attributes)
    write_variant_ids(
        root,
        chromosome,
        [
            f"{chromosome}-{position}-allele{record}-{length}"
            for record, (position, length) in enumerate(zip(positions.tolist(), alternate_lengths.tolist()))
        ],
    )
    return sites_md5(positions, reference_lengths, alternate_lengths)


def generate_store(plan: GenerationPlan, *, workers: int) -> list[tuple[int, int, float]]:
    """Generate every shard of every chromosome; returns (chromosome, shard, seconds) per task."""
    tasks = [
        (chromosome_index, shard_index)
        for chromosome_index, layout in enumerate(plan.layouts)
        for shard_index in range(-(-layout.record_count // plan.shard_rows))
    ]
    _WORKER_PLAN.append(plan)
    try:
        if workers == 1:
            return [_run_shard(task) for task in tasks]
        with multiprocessing.get_context("fork").Pool(processes=workers) as pool:
            return list(pool.imap_unordered(_run_shard, tasks))
    finally:
        _WORKER_PLAN.clear()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic svpgs-store v1 from 1kGP haplotype mosaics.")
    parser.add_argument("--source", type=Path, nargs="+", required=True, help="design-credit src_chr*.npz haplotype sources")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--half-samples", type=int, nargs="+", required=True)
    parser.add_argument("--half-pipelines", nargs="+", required=True, choices=sorted(PIPELINE_R2_LOSS))
    parser.add_argument("--records", type=int, required=True)
    parser.add_argument("--chromosomes", type=int, default=len(HG38_AUTOSOME_MEGABASES))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--block-records", type=int, required=True)
    parser.add_argument("--codec", choices=get_args(Codec), required=True)
    parser.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    arguments = parser.parse_args(argv)
    started = time.perf_counter()
    source = HaplotypeSource.load(arguments.source)
    plan = plan_store(
        arguments.out,
        source,
        half_sample_counts=arguments.half_samples,
        half_pipelines=arguments.half_pipelines,
        total_records=arguments.records,
        chromosome_count=arguments.chromosomes,
        seed=arguments.seed,
        block_records=arguments.block_records,
        codec=arguments.codec,
        parallel_writers=arguments.workers,
    )
    planned = time.perf_counter()
    timings = generate_store(plan, workers=arguments.workers)
    finished = time.perf_counter()
    summary = {
        "records": arguments.records,
        "samples": int(sum(arguments.half_samples)),
        "plan_seconds": round(planned - started, 1),
        "generate_seconds": round(finished - planned, 1),
        "shards": len(timings),
        "mean_shard_seconds": round(float(np.mean([seconds for _, _, seconds in timings])), 2),
        "workers": arguments.workers,
    }
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
