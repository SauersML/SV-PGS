"""Integer copy number as a store column: its code scale, its origin and its values.

A multi-copy CNV (PDXDC1, KANSL1, WASH3P) is not a biallelic ALT count. Its
genotype is an integer copy number with more than three states, which fails
Hardy-Weinberg and Mendelian checks as a 0/1/2 dosage. Such a record
(``VariantClass.COPY_NUMBER``) is stored as its copy number measured from the
record's modal copy number, value = CN - modal CN, through the store's
per-record affine decode

    value = code / codes_per_unit + value_origin.

An ALT-count record has ``codes_per_unit = CODES_PER_DOSAGE`` (127) and
``value_origin = 0``, today's encoding. A copy-number record has
``codes_per_unit = floor(MAXIMUM_CODE / max CN)``, the largest scale at which
its highest copy number still fits a code, so every integer copy number is
exact and a recalibrated D* keeps 1 / codes_per_unit of a copy of resolution;
its ``value_origin`` is -modal CN. The codes stay uint8 in [0, MAXIMUM_CODE].

Every stage that works on standardized columns (Stage 0, Stage 2, scoring) is
affine-invariant per column and needs neither number; only code that reads
values in their own units (a frequency, a scale floor, the measurement model's
moments) decodes them here.
"""

from __future__ import annotations

import numpy as np

from sv_pgs._typing import F64Array, I64Array, NDArray, U8Array
from sv_pgs.dosage_store import CODES_PER_DOSAGE, MAXIMUM_CODE


def modal_copy_numbers(copy_numbers: NDArray, called: NDArray) -> I64Array:
    """Each record's most frequent called copy number, the smallest among ties.

    ``copy_numbers`` and ``called`` are [records, samples]; a record with no call
    has no mode and is refused.
    """
    values = np.asarray(copy_numbers)
    observed = np.asarray(called, dtype=bool)
    if values.dtype.kind not in "iu" or values.ndim != 2 or observed.shape != values.shape:
        raise ValueError("modal_copy_numbers needs integer copy numbers [records, samples] and a call mask of that shape.")
    if not np.all(observed.any(axis=1)):
        raise ValueError("every copy-number record needs at least one called sample.")
    if values.size and (int(values[observed].min()) < 0 or int(values[observed].max()) > MAXIMUM_CODE):
        raise ValueError(f"copy numbers must lie in [0, {MAXIMUM_CODE}].")
    counts = np.zeros((values.shape[0], MAXIMUM_CODE + 1), dtype=np.int64)
    rows = np.broadcast_to(np.arange(values.shape[0])[:, None], values.shape)
    np.add.at(counts, (rows[observed], values[observed].astype(np.int64)), 1)
    return np.argmax(counts, axis=1).astype(np.int64)


def copy_number_codes_per_unit(maximum_copy_numbers: NDArray) -> U8Array:
    """floor(MAXIMUM_CODE / max CN) codes per copy: every integer copy number up to the maximum is exact."""
    maximum = np.asarray(maximum_copy_numbers, dtype=np.int64)
    if np.any(maximum < 0) or np.any(maximum > MAXIMUM_CODE):
        raise ValueError(f"a copy-number record's maximum copy number must lie in [0, {MAXIMUM_CODE}].")
    return (MAXIMUM_CODE // np.maximum(maximum, 1)).astype(np.uint8)


def encode_copy_numbers(copy_numbers: NDArray, codes_per_unit: NDArray) -> U8Array:
    """Codes CN * codes_per_unit per record, [records, samples]; exact for every integer copy number."""
    values = np.asarray(copy_numbers)
    scale = np.asarray(codes_per_unit, dtype=np.int64)
    if values.dtype.kind not in "iu" or values.ndim != 2 or scale.shape != (values.shape[0],):
        raise ValueError("encode_copy_numbers needs integer copy numbers [records, samples] and one scale per record.")
    codes = values.astype(np.int64) * scale[:, None]
    if codes.size and (int(codes.min()) < 0 or int(codes.max()) > MAXIMUM_CODE):
        raise ValueError(f"a copy number times its record's codes per unit must lie in [0, {MAXIMUM_CODE}].")
    return codes.astype(np.uint8)


def allele_count_decode(record_count: int) -> tuple[U8Array, I64Array]:
    """The (codes_per_unit, value_origin) of ALT-count records: DS = code / 127."""
    return np.full(record_count, CODES_PER_DOSAGE, dtype=np.uint8), np.zeros(record_count, dtype=np.int64)


def decode_values(codes: NDArray, codes_per_unit: NDArray, value_origin: NDArray) -> F64Array:
    """value = code / codes_per_unit + value_origin, [records, samples]: an ALT dosage, or CN - modal CN."""
    stored = np.asarray(codes)
    scale = np.asarray(codes_per_unit, dtype=np.float64)
    origin = np.asarray(value_origin, dtype=np.float64)
    if stored.ndim != 2 or scale.shape != (stored.shape[0],) or origin.shape != scale.shape:
        raise ValueError("decode_values needs codes [records, samples] and one scale and origin per record.")
    if not np.all(scale > 0):
        raise ValueError("every record needs a positive codes_per_unit.")
    if stored.size and int(stored.max()) > MAXIMUM_CODE:
        raise ValueError(f"codes above {MAXIMUM_CODE} are the missing fill value, never a stored value.")
    return stored.astype(np.float64) / scale[:, None] + origin[:, None]
