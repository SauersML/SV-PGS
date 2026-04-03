from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


class _InfoMap(dict[str, object]):
    def get(self, key: str, default=None):
        return super().get(key, default)


@dataclass(slots=True)
class _Record:
    CHROM: str
    POS: int
    ID: str | None
    REF: str
    ALT: list[str]
    QUAL: float | None
    FILTER: str
    INFO: _InfoMap
    _sample_genotypes: list[str]

    @property
    def gt_types(self) -> np.ndarray:
        encoded_types = []
        for genotype in self._sample_genotypes:
            normalized = genotype.split(":", 1)[0]
            if normalized in {"./.", ".|."}:
                encoded_types.append(2)
                continue
            allele_tokens = normalized.replace("|", "/").split("/")
            if any(allele_token == "." for allele_token in allele_tokens):
                encoded_types.append(2)
                continue
            allele_values = [int(allele_token) for allele_token in allele_tokens]
            if all(allele_value == 0 for allele_value in allele_values):
                encoded_types.append(0)
            elif all(allele_value == allele_values[0] for allele_value in allele_values) and allele_values[0] > 0:
                encoded_types.append(3)
            else:
                encoded_types.append(1)
        return np.asarray(encoded_types, dtype=np.int8)

    @property
    def is_snp(self) -> bool:
        return len(self.REF) == 1 and all(len(alternate) == 1 and not alternate.startswith("<") for alternate in self.ALT)

    @property
    def is_sv(self) -> bool:
        return any(alternate.startswith("<") for alternate in self.ALT) or self.INFO.get("SVTYPE") is not None

    @property
    def is_indel(self) -> bool:
        return not self.is_sv and not self.is_snp

    @property
    def end(self) -> int | None:
        end_value = self.INFO.get("END")
        if end_value is None:
            return None
        return int(end_value)


class VCF:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.samples: list[str] = []
        self._records: list[_Record] = []
        self._parse()

    def _parse(self) -> None:
        with self.path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("##"):
                    continue
                if line.startswith("#CHROM"):
                    fields = line.split("\t")
                    self.samples = fields[9:]
                    continue
                fields = line.split("\t")
                self._records.append(
                    _Record(
                        CHROM=fields[0],
                        POS=int(fields[1]),
                        ID=None if fields[2] == "." else fields[2],
                        REF=fields[3],
                        ALT=fields[4].split(","),
                        QUAL=None if fields[5] == "." else float(fields[5]),
                        FILTER=fields[6],
                        INFO=_parse_info_field(fields[7]),
                        _sample_genotypes=fields[9:],
                    )
                )

    def __iter__(self) -> Iterator[_Record]:
        return iter(self._records)

    def set_threads(self, thread_count: int) -> None:
        del thread_count

    def close(self) -> None:
        return None


def _parse_info_field(info_field: str) -> _InfoMap:
    info_map: _InfoMap = _InfoMap()
    if info_field in {"", "."}:
        return info_map
    for entry in info_field.split(";"):
        if "=" not in entry:
            info_map[entry] = True
            continue
        key, raw_value = entry.split("=", 1)
        values = raw_value.split(",")
        parsed_values = [_parse_scalar(value) for value in values]
        info_map[key] = parsed_values[0] if len(parsed_values) == 1 else tuple(parsed_values)
    return info_map


def _parse_scalar(value: str) -> object:
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value
