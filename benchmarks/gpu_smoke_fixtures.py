"""The small store fixtures shared by the Stage 0 tests and the standalone GPU smoke command.

These are the tests' mosaic codes and dosage-store writer, kept here so a staged runtime can build the same
problem without importing pytest. The writer's encoding, metadata, shard sizes and write boundaries are unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sv_pgs.dosage_store import (
    CODES_PER_DOSAGE, VARIANT_CLASS_LEGEND, CodeShardWriter, create_code_array, dosage_array_directory,
    encode_dosage_milli, sites_md5, statistic_column_directory, variant_column_directory, write_column,
    write_manifest, write_variant_ids,
)

SHARD_ROWS = 128
INNER_ROWS = 16


def mosaic_codes(
    rng: np.random.Generator,
    samples: int,
    variants: int,
    founders: int = 10,
    hotspot_spacing: int = 60,
    regular_hotspots: bool = False,
) -> NDArray[np.uint8]:
    """Variant-major 8-bit dosage codes with block-structured LD.

    Each haplotype copies one of ``founders`` founder haplotypes and switches founder
    rarely inside LD blocks and often at hotspots, every ``hotspot_spacing`` variants
    (exactly, or on average). Dosages get imputation-like noise; a few variants are
    monomorphic.
    """
    frequency = rng.uniform(0.02, 0.5, size=variants)
    founder_alleles = rng.random((founders, variants)) < frequency
    switch_rate = np.full(variants, 0.004)
    if regular_hotspots:
        switch_rate[hotspot_spacing::hotspot_spacing] = 0.9
    else:
        switch_rate[rng.random(variants) < 1.0 / hotspot_spacing] = 0.6
    haplotypes = 2 * samples
    switches = rng.random((haplotypes, variants)) < switch_rate
    switches[:, 0] = True
    segment = np.cumsum(switches, axis=1) - 1
    segment_founder = rng.integers(0, founders, size=(haplotypes, variants))
    founder = np.take_along_axis(segment_founder, segment, axis=1)
    alleles = founder_alleles[founder, np.arange(variants)[None, :]].astype(np.float64)
    dosage = alleles[0::2] + alleles[1::2]
    noisy = np.clip(dosage + rng.normal(0.0, 0.08, size=dosage.shape), 0.0, 2.0)
    codes = np.rint(noisy * 127.0).astype(np.uint8).T.copy()
    codes[rng.choice(variants, size=max(1, variants // 100), replace=False)] = 3
    return codes



def _write_store(root: Path, milli_by_half: list[dict[str, np.ndarray]], codec: str = "raw") -> None:
    chromosomes = list(milli_by_half[0])
    record_counts = [milli_by_half[0][chromosome].shape[0] for chromosome in chromosomes]
    digests = []
    for chromosome, record_count in zip(chromosomes, record_counts):
        positions = np.arange(1, record_count + 1, dtype=np.int64) * 100
        lengths = np.ones(record_count, dtype=np.int32)
        columns = {
            "pos": (positions, {}),
            "ref_len": (lengths, {}),
            "alt_len": (lengths, {}),
            "cm": (positions / 1e6, {}),
            "variant_class": (np.zeros(record_count, dtype=np.uint8), {"legend": VARIANT_CLASS_LEGEND}),
            "codes_per_unit": (np.full(record_count, CODES_PER_DOSAGE, dtype=np.uint8), {}),
            "value_origin": (np.zeros(record_count, dtype=np.int16), {}),
            "group_first": (np.arange(record_count, dtype=np.int64), {}),
            "class": (np.zeros(record_count, dtype=np.uint8), {"legend": ["SNV", "INDEL", "SV"]}),
            "has_pl": (np.arange(record_count) % 3 == 0, {}),
            "panel_af": (np.full(record_count, 0.25, dtype=np.float32), {}),
        }
        for name, (values, attributes) in columns.items():
            write_column(variant_column_directory(root, chromosome, name), values, attributes)
        write_variant_ids(root, chromosome, [f"{chromosome}-{position}-allele0-1" for position in positions])
        digests.append(sites_md5(positions, lengths, lengths))
    write_manifest(
        root,
        chromosomes=chromosomes,
        record_counts=record_counts,
        half_sample_counts=[half[chromosomes[0]].shape[1] for half in milli_by_half],
        chromosome_sites_md5=digests,
    )
    for half_index, half in enumerate(milli_by_half):
        for chromosome, milli in half.items():
            codes = encode_dosage_milli(milli)
            directory = dosage_array_directory(root, half_index, chromosome)
            layout = create_code_array(
                directory, codes.shape[0], codes.shape[1], codec=codec, shard_rows=SHARD_ROWS, inner_rows=INNER_ROWS
            )
            for shard_index in range(layout.shard_count):
                with CodeShardWriter(directory, layout, shard_index) as writer:
                    shard_rows = codes[shard_index * SHARD_ROWS : (shard_index + 1) * SHARD_ROWS]
                    for start in range(0, shard_rows.shape[0], 37):
                        writer.write_rows(shard_rows[start : start + 37])
            wide = codes.astype(np.int64)
            write_column(statistic_column_directory(root, half_index, chromosome, "sum_code"), wide.sum(axis=1).astype(np.uint64))
            write_column(
                statistic_column_directory(root, half_index, chromosome, "sum_code2"), (wide * wide).sum(axis=1).astype(np.uint64)
            )
