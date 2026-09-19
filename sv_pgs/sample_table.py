"""Delimited sample tables: sniffing, row iteration, and the per-sample target and covariates.

Also the small text helpers shared with the pipeline's writers: opening
optionally gzipped text files and the canonical float formatting.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import gzip
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

import numpy as np

from sv_pgs._typing import F32Array


@dataclass(slots=True)
class _SampleTable:
    sample_ids: list[str]
    covariates: F32Array
    targets: F32Array


@dataclass(slots=True)
class _DelimitedTableSpec:
    path: Path
    delimiter: str
    columns: tuple[str, ...]
    column_index_by_name: dict[str, int]


def _open_text_file(path: Path, mode: Literal["rt", "wt"], *, newline: str | None = None) -> Any:
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", newline=newline)
    return path.open(mode.replace("t", ""), encoding="utf-8", newline=newline)


def _build_sample_table(
    table_spec: _DelimitedTableSpec,
    sample_id_column: str,
    target_column: str,
    covariate_columns: Sequence[str],
    available_sample_ids: Sequence[str],
) -> tuple[_SampleTable, int, int]:
    _require_columns(
        available_columns=table_spec.columns,
        required_columns=(sample_id_column, target_column, *covariate_columns),
        context="sample table",
    )
    sample_id_index = table_spec.column_index_by_name[sample_id_column]
    target_index = table_spec.column_index_by_name[target_column]
    covariate_indices = tuple(table_spec.column_index_by_name[column_name] for column_name in covariate_columns)

    # Parse sample table into a dict keyed by sample_id so we can emit
    # rows in VCF order (available_sample_ids) rather than file order.
    # This guarantees the returned _SampleTable is already aligned with
    # the genotype matrices — no reindexing needed downstream.
    parsed: dict[str, tuple[float, list[float]]] = {}
    available_sample_id_set = set(available_sample_ids)
    total_rows = 0
    unmatched_rows = 0

    for row_values in _iter_delimited_row_values(table_spec):
        total_rows += 1
        sample_id = row_values[sample_id_index].strip()
        if not sample_id:
            raise ValueError("Encountered blank sample identifier in sample table.")
        if sample_id not in available_sample_id_set:
            unmatched_rows += 1
            continue
        if sample_id in parsed:
            raise ValueError("Duplicate sample identifier in sample table: " + sample_id)
        try:
            target_value = float(row_values[target_index])
            covariate_values = [float(row_values[column_index]) for column_index in covariate_indices]
        except (ValueError, TypeError):
            unmatched_rows += 1
            continue
        if np.isnan(target_value) or any(np.isnan(covariate_value) for covariate_value in covariate_values):
            unmatched_rows += 1
            continue
        parsed[sample_id] = (target_value, covariate_values)

    # Emit rows in VCF sample order so genotype matrices need no reindexing.
    sample_ids: list[str] = []
    targets: list[float] = []
    covariates: list[list[float]] = []
    for vcf_sample_id in available_sample_ids:
        if vcf_sample_id in parsed:
            target_value, covariate_values = parsed[vcf_sample_id]
            sample_ids.append(vcf_sample_id)
            targets.append(target_value)
            covariates.append(covariate_values)

    covariate_matrix = np.asarray(covariates, dtype=np.float32)
    if covariate_matrix.ndim != 2:
        covariate_matrix = covariate_matrix.reshape(len(sample_ids), len(covariate_columns))
    if not sample_ids:
        raise ValueError(
            "Sample table contains no rows that overlap the genotype source using column: " + sample_id_column
        )
    return (
        _SampleTable(
            sample_ids=sample_ids,
            covariates=covariate_matrix,
            targets=np.asarray(targets, dtype=np.float32),
        ),
        total_rows,
        unmatched_rows,
    )


def _inspect_delimited_table(path: str | Path) -> _DelimitedTableSpec:
    resolved_path = Path(path)
    with _open_text_file(resolved_path, "rt", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = _infer_delimiter(sample)
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError("Table has no header row: " + str(resolved_path))
        columns = tuple(str(field_name) for field_name in reader.fieldnames)
    return _DelimitedTableSpec(
        path=resolved_path,
        delimiter=delimiter,
        columns=columns,
        column_index_by_name={column_name: column_index for column_index, column_name in enumerate(columns)},
    )


def _iter_delimited_rows(table_spec: _DelimitedTableSpec) -> Iterator[dict[str, str]]:
    columns = table_spec.columns
    for row_values in _iter_delimited_row_values(table_spec):
        yield {
            column_name: row_values[column_index]
            for column_index, column_name in enumerate(columns)
        }


def _iter_delimited_row_values(table_spec: _DelimitedTableSpec) -> Iterator[list[str]]:
    with _open_text_file(table_spec.path, "rt", newline="") as handle:
        reader = csv.reader(handle, delimiter=table_spec.delimiter)
        next(reader, None)
        expected_width = len(table_spec.columns)
        for row in reader:
            normalized_row = ["" if value is None else str(value) for value in row[:expected_width]]
            if len(normalized_row) < expected_width:
                normalized_row.extend([""] * (expected_width - len(normalized_row)))
            yield normalized_row


def _infer_delimiter(sample: str) -> str:
    tab_count = sample.count("\t")
    comma_count = sample.count(",")
    if tab_count == 0 and comma_count == 0:
        raise ValueError("Expected a tab-delimited or comma-delimited file with a header row.")
    return "\t" if tab_count >= comma_count else ","


def _require_columns(
    available_columns: Sequence[str],
    required_columns: Sequence[str],
    context: str,
) -> None:
    available_column_set = set(available_columns)
    missing_columns = [column_name for column_name in required_columns if column_name not in available_column_set]
    if missing_columns:
        raise ValueError(
            context
            + " is missing required columns: "
            + ", ".join(missing_columns)
        )


def _format_float(value: float) -> str:
    return format(value, ".8g")


def _coerce_float(value: object) -> float:
    return float(str(value))
