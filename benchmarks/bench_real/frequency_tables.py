"""The posterior's frequency dependence by variant class for a finished run whose effects step predates it:
``effect_by_frequency.tsv`` and ``frequency_alpha.tsv`` (``workspace_pipeline.frequency_dependence``), read from the
effects step's per-record tables alone, so it runs where the run's store does not fit in memory.

Inputs: each ``<trait>.<arm>.tsv.gz`` named (the effects step's per-record table of one all-rows model, kept in the
workspace), of which only the columns variant_class, length, mean_value, effect and posterior_sd are read, in their
store order (the jackknife's blocks). A record's counted allele frequency is its mean value over two, its mean dosage
over the model's training rows, where the effects step reads the store's sum over every sample; a copy number's
(FREQUENCY_EXCLUDED_CLASSES) is never read. The posterior second moment of one allele's effect is effect^2 +
posterior_sd^2. The tables carry no prior (its columns blank), which the per-record tables do not hold: the prior's
frequency columns come from ``prior_tables.py`` or the effects step itself.

Outputs: ``--out`` gets the two tables, every named model's lines in the order named, each through
``workspace_pipeline.check_exportable``; a table it refuses is removed and its rule printed. stdout names the tables
and the models read, never a record.

    python -m benchmarks.bench_real.frequency_tables EFFECTS.tsv.gz [EFFECTS.tsv.gz ...] --out DIR
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sv_pgs.workspace_pipeline import (
    EFFECT_BY_FREQUENCY_COLUMNS,
    FREQUENCY_ALPHA_COLUMNS,
    check_exportable,
    frequency_dependence,
)

TABLES = {"effect_by_frequency.tsv": EFFECT_BY_FREQUENCY_COLUMNS, "frequency_alpha.tsv": FREQUENCY_ALPHA_COLUMNS}
COLUMNS = ("variant_class", "length", "mean_value", "effect", "posterior_sd")
"""The per-record table's columns read; every other is left unread."""


def model_name(path: Path) -> tuple[str, str]:
    """(trait, arm) of a per-record table named ``<trait>.<arm>.tsv.gz``."""
    trait, arm = path.name.removesuffix(".tsv.gz").rsplit(".", 1)
    return trait, arm


def frequency_lines(paths: list[Path]) -> dict[str, list[list[Any]]]:
    """Each table's lines (its header first) over the models of ``paths``, in their order."""
    lines: dict[str, list[list[Any]]] = {name: [list(columns)] for name, columns in TABLES.items()}
    for path in paths:
        trait, arm = model_name(path)
        records = pd.read_csv(path, sep="\t", usecols=list(COLUMNS), dtype={"variant_class": object})
        effect, spread = records["effect"].to_numpy(np.float64), records["posterior_sd"].to_numpy(np.float64)
        found = frequency_dependence(
            trait, arm, records["variant_class"].to_numpy(object), records["length"].to_numpy(np.float64),
            records["mean_value"].to_numpy(np.float64) / 2.0, effect * effect + spread * spread, None,
        )
        # As the effects step keeps a model's lines: a JSON round trip, so the text matches its tables'.
        found = json.loads(json.dumps(found, default=str))
        for name in TABLES:
            lines[name].extend(found[name])
    return lines


def write_tables(lines: dict[str, list[list[Any]]], out: Path) -> tuple[list[Path], dict[str, str]]:
    """Each table in ``out``, kept only where the export check passes it."""
    out.mkdir(parents=True, exist_ok=True)
    written, refused = [], {}
    for name, rows in lines.items():
        path = out / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle, delimiter="\t").writerows(rows)
        try:
            check_exportable(path)
        except ValueError as error:
            path.unlink()
            refused[name] = str(error).rsplit(": ", 1)[-1]
        else:
            written.append(path)
    return written, refused


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tables", type=Path, nargs="+", help="The effects step's per-record tables, <trait>.<arm>.tsv.gz.")
    parser.add_argument("--out", type=Path, required=True, help="Where the two tables go.")
    args = parser.parse_args(argv)
    written, refused = write_tables(frequency_lines(args.tables), args.out)
    for path in args.tables:
        print("read {} {}".format(*model_name(path)))
    for path in written:
        print(f"wrote {path}")
    for name, rule in refused.items():
        print(f"not written, refused by the export check: {name} ({rule})")
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
