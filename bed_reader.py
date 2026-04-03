from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def to_bed(
    bed_path: str | Path,
    values: np.ndarray,
    properties: dict[str, list[Any]],
) -> None:
    resolved_path = Path(bed_path)
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("BED matrix must be 2D.")

    with resolved_path.open("wb") as bed_handle:
        np.save(bed_handle, matrix)

    sample_ids = [str(sample_id) for sample_id in properties.get("iid", [])]
    family_ids = [str(family_id) for family_id in properties.get("fid", sample_ids)]
    if len(sample_ids) != matrix.shape[0] or len(family_ids) != matrix.shape[0]:
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
    chromosomes = [str(chromosome) for chromosome in properties.get("chromosome", ["1"] * matrix.shape[1])]
    positions = [int(position) for position in properties.get("bp_position", range(1, matrix.shape[1] + 1))]
    allele_one = [str(value) for value in properties.get("allele_1", ["A"] * matrix.shape[1])]
    allele_two = [str(value) for value in properties.get("allele_2", ["C"] * matrix.shape[1])]
    if not (
        len(variant_ids)
        == len(chromosomes)
        == len(positions)
        == len(allele_one)
        == len(allele_two)
        == matrix.shape[1]
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
    iid_count: int | None = None
    sid_count: int | None = None
    properties: dict[str, Any] | None = None
    skip_format_check: bool = True
    num_threads: int | None = None
    _matrix: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        with self.path.open("rb") as bed_handle:
            self._matrix = np.asarray(np.load(bed_handle), dtype=np.float32)
        if self._matrix.ndim != 2:
            raise ValueError("BED payload must be 2D.")

    def read(
        self,
        index=None,
        dtype: str | np.dtype = "float32",
        order: str = "F",
        num_threads: int | None = None,
    ) -> np.ndarray:
        del num_threads
        matrix = self._matrix
        if index is not None:
            sample_index, variant_index = index
            matrix = matrix[sample_index, :]
            matrix = matrix[:, variant_index]
        if dtype == "int8" or np.dtype(dtype) == np.dtype(np.int8):
            int_matrix = np.rint(matrix)
            int_matrix = np.where(np.isnan(matrix), -127, int_matrix)
            result = np.asarray(int_matrix, dtype=np.int8)
        else:
            result = np.asarray(matrix, dtype=dtype)
        if order == "F":
            return np.asfortranarray(result)
        return np.asarray(result, order=order)
