from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

PLINK1_MAGIC = b"\x6c\x1b\x01"
PLINK1_HEADER_SIZE = 3
PLINK_MISSING_INT8 = np.int8(-127)

_DECODE_LOOKUP_A1 = np.array([2, PLINK_MISSING_INT8, 1, 0], dtype=np.int8)
_DECODE_LOOKUP_A2 = np.array([0, PLINK_MISSING_INT8, 1, 2], dtype=np.int8)
_ENCODE_LOOKUP_A1 = np.array([0b11, 0b10, 0b00], dtype=np.uint8)


def to_bed(
    bed_path: str | Path,
    values: np.ndarray,
    properties: dict[str, list[Any]],
) -> None:
    resolved_path = Path(bed_path)
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("BED matrix must be 2D.")

    sample_count, variant_count = matrix.shape
    bytes_per_variant = _bytes_per_variant(sample_count)

    with resolved_path.open("wb") as bed_handle:
        bed_handle.write(PLINK1_MAGIC)
        for variant_index in range(variant_count):
            packed = _encode_variant(matrix[:, variant_index], bytes_per_variant=bytes_per_variant)
            bed_handle.write(packed)

    sample_ids = [str(sample_id) for sample_id in properties.get("iid", [])]
    family_ids = [str(family_id) for family_id in properties.get("fid", sample_ids)]
    if len(sample_ids) != sample_count or len(family_ids) != sample_count:
        raise ValueError("Sample metadata must match BED row count.")
    fam_path = resolved_path.with_suffix(".fam")
    fam_path.write_text(
        "".join(
            f"{family_id} {sample_id} 0 0 0 -9\n"
            for family_id, sample_id in zip(family_ids, sample_ids, strict=True)
        ),
        encoding="utf-8",
    )

    variant_ids = [str(variant_id) for variant_id in properties.get("sid", [])]
    chromosomes = [str(chromosome) for chromosome in properties.get("chromosome", ["1"] * variant_count)]
    positions = [int(position) for position in properties.get("bp_position", range(1, variant_count + 1))]
    allele_one = [str(value) for value in properties.get("allele_1", ["A"] * variant_count)]
    allele_two = [str(value) for value in properties.get("allele_2", ["C"] * variant_count)]
    if not (
        len(variant_ids)
        == len(chromosomes)
        == len(positions)
        == len(allele_one)
        == len(allele_two)
        == variant_count
    ):
        raise ValueError("Variant metadata must match BED column count.")
    bim_path = resolved_path.with_suffix(".bim")
    bim_path.write_text(
        "".join(
            f"{chromosome} {variant_id} 0 {position} {allele_left} {allele_right}\n"
            for chromosome, variant_id, position, allele_left, allele_right in zip(
                chromosomes,
                variant_ids,
                positions,
                allele_one,
                allele_two,
                strict=True,
            )
        ),
        encoding="utf-8",
    )


@dataclass(slots=True)
class open_bed:
    path: str | Path
    iid_count: int
    sid_count: int
    properties: dict[str, Any] | None = None
    count_A1: bool = True
    num_threads: int | None = None

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.iid_count < 1 or self.sid_count < 1:
            raise ValueError("iid_count and sid_count must be positive.")
        with Path(self.path).open("rb") as bed_handle:
            header = bed_handle.read(PLINK1_HEADER_SIZE)
        if header != PLINK1_MAGIC:
            raise ValueError(
                "Invalid PLINK 1 .bed header: expected "
                + PLINK1_MAGIC.hex(" ")
                + ", got "
                + header.hex(" ")
            )
        expected_size = PLINK1_HEADER_SIZE + self.sid_count * _bytes_per_variant(self.iid_count)
        actual_size = Path(self.path).stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                "PLINK 1 .bed size does not match iid_count/sid_count: "
                + f"expected {expected_size} bytes, found {actual_size}"
            )

    def read(
        self,
        index=None,
        dtype: str | np.dtype = "float32",
        order: Literal["F", "C"] = "F",
        num_threads: int | None = None,
    ) -> np.ndarray:
        del num_threads
        sample_index, variant_index = _split_index(index)
        resolved_samples = _normalize_index(sample_index, self.iid_count)
        resolved_variants = _normalize_index(variant_index, self.sid_count)
        raw_i8 = self._read_int8_matrix(resolved_variants)
        raw_i8 = raw_i8[resolved_samples, :]

        resolved_dtype = np.dtype(dtype)
        if resolved_dtype == np.dtype(np.int8):
            result = np.asarray(raw_i8, dtype=np.int8)
        else:
            result = raw_i8.astype(resolved_dtype, copy=False)
            result[raw_i8 == PLINK_MISSING_INT8] = np.nan
        if order == "F":
            return np.asfortranarray(result)
        return np.ascontiguousarray(result)

    def _read_int8_matrix(self, variant_index: slice | np.ndarray) -> np.ndarray:
        bytes_per_variant = _bytes_per_variant(self.iid_count)
        if isinstance(variant_index, slice):
            variant_count = max(variant_index.stop - variant_index.start, 0)
            if variant_count == 0:
                return np.empty((self.iid_count, 0), dtype=np.int8)
            with Path(self.path).open("rb") as bed_handle:
                bed_handle.seek(PLINK1_HEADER_SIZE + variant_index.start * bytes_per_variant)
                payload = bed_handle.read(variant_count * bytes_per_variant)
            return _decode_payload(
                payload,
                iid_count=self.iid_count,
                variant_count=variant_count,
                bytes_per_variant=bytes_per_variant,
                count_a1=self.count_A1,
            )

        result = np.empty((self.iid_count, variant_index.shape[0]), dtype=np.int8)
        with Path(self.path).open("rb") as bed_handle:
            for output_index, bed_variant_index in enumerate(variant_index):
                bed_handle.seek(PLINK1_HEADER_SIZE + int(bed_variant_index) * bytes_per_variant)
                payload = bed_handle.read(bytes_per_variant)
                result[:, output_index] = _decode_payload(
                    payload,
                    iid_count=self.iid_count,
                    variant_count=1,
                    bytes_per_variant=bytes_per_variant,
                    count_a1=self.count_A1,
                )[:, 0]
        return result


def _bytes_per_variant(sample_count: int) -> int:
    return (int(sample_count) + 3) // 4


def _split_index(index: Any) -> tuple[Any, Any]:
    if index is None:
        return None, None
    if not isinstance(index, tuple) or len(index) != 2:
        raise ValueError("index must be a (sample_index, variant_index) tuple.")
    return index


def _normalize_index(index: Any, limit: int) -> slice | np.ndarray:
    if index is None:
        return slice(0, limit, 1)
    if isinstance(index, slice):
        start, stop, step = index.indices(limit)
        if step == 1:
            return slice(start, stop, 1)
        return np.arange(start, stop, step, dtype=np.intp)
    if np.isscalar(index):
        value = int(np.asarray(index, dtype=np.intp).item())
        if value < 0:
            value += limit
        if value < 0 or value >= limit:
            raise IndexError("PLINK index out of bounds.")
        return np.asarray([value], dtype=np.intp)

    values = np.asarray(index)
    if values.dtype == np.bool_:
        if values.ndim != 1 or values.shape[0] != limit:
            raise ValueError("Boolean PLINK index mask must be 1D and match the axis length.")
        return np.flatnonzero(values).astype(np.intp, copy=False)
    values = np.asarray(values, dtype=np.intp)
    if values.ndim != 1:
        raise ValueError("PLINK indices must be 1D.")
    normalized = values.copy()
    normalized[normalized < 0] += limit
    if np.any(normalized < 0) or np.any(normalized >= limit):
        raise IndexError("PLINK index out of bounds.")
    return np.ascontiguousarray(normalized, dtype=np.intp)


def _decode_payload(
    payload: bytes,
    *,
    iid_count: int,
    variant_count: int,
    bytes_per_variant: int,
    count_a1: bool,
) -> np.ndarray:
    raw_bytes = np.frombuffer(payload, dtype=np.uint8)
    if raw_bytes.size != variant_count * bytes_per_variant:
        raise ValueError("Unexpected PLINK 1 .bed payload length.")
    packed = raw_bytes.reshape(variant_count, bytes_per_variant)
    codes = np.empty((variant_count, bytes_per_variant * 4), dtype=np.uint8)
    codes[:, 0::4] = packed & 0b11
    codes[:, 1::4] = (packed >> 2) & 0b11
    codes[:, 2::4] = (packed >> 4) & 0b11
    codes[:, 3::4] = (packed >> 6) & 0b11
    lookup = _DECODE_LOOKUP_A1 if count_a1 else _DECODE_LOOKUP_A2
    return lookup[codes][:, :iid_count].T


def _encode_variant(column: np.ndarray, *, bytes_per_variant: int) -> bytes:
    observed = ~np.isnan(column)
    rounded = np.rint(column[observed])
    if not np.all(np.isclose(column[observed], rounded)):
        raise ValueError("PLINK BED supports only hardcall dosages 0, 1, 2, or NaN.")
    rounded_i8 = rounded.astype(np.int8, copy=False)
    if np.any((rounded_i8 < 0) | (rounded_i8 > 2)):
        raise ValueError("PLINK BED supports only hardcall dosages 0, 1, 2, or NaN.")

    codes = np.zeros(bytes_per_variant * 4, dtype=np.uint8)
    sample_codes = codes[: column.shape[0]]
    sample_codes[~observed] = 0b01
    sample_codes[observed] = _ENCODE_LOOKUP_A1[rounded_i8]
    packed = (
        codes[0::4]
        | (codes[1::4] << 2)
        | (codes[2::4] << 4)
        | (codes[3::4] << 6)
    )
    return packed.tobytes()
