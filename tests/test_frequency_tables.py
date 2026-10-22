"""benchmarks/bench_real/frequency_tables.py: the frequency tables from the effects step's per-record tables alone."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.bench_real import frequency_tables
from sv_pgs import workspace_pipeline


def _records(generator: np.random.Generator) -> pd.DataFrame:
    """A per-record table as the effects step writes it: 2,000 SNVs and 300 deletions of 50-499 bp at minor allele
    frequencies 1-50%, 25 copy numbers, store order mixing them, every column of the step's table present."""
    count = 2325
    classes = np.array(["snv"] * 2000 + ["deletion"] * 300 + ["copy_number"] * 25, dtype=object)
    frequency = generator.uniform(0.01, 0.5, count)
    frequency[::3] = 1.0 - frequency[::3]
    heterozygosity = 2.0 * frequency * (1.0 - frequency)
    effect = generator.normal(0.0, 1.0, count) * heterozygosity ** np.where(classes == "snv", -0.2, -0.35)
    order = generator.permutation(count)
    return pd.DataFrame({
        "chromosome": np.ones(count, dtype=np.int64), "position": np.arange(count), "id": [f"v{index}" for index in range(count)],
        "variant_class": classes[order], "length": np.where(classes == "snv", 1.0, np.where(classes == "deletion", 200.0, 5000.0))[order],
        "row_source": "imputed", "mean_value": np.where(classes == "copy_number", 2.3, 2.0 * frequency)[order], "effect": effect[order],
        "posterior_sd": np.full(count, 0.05), "likelihood": "gaussian", "probability_positive": 0.5, "probability_negative": 0.5,
    })


def test_the_tables_are_the_effects_step_s_lines_from_the_per_record_columns(tmp_path: Path) -> None:
    generator = np.random.default_rng(5)
    records = {arm: _records(generator) for arm in ("snv_indel", "with_sv")}
    paths = []
    for arm, table in records.items():
        path = tmp_path / f"ldl_cholesterol.{arm}.tsv.gz"
        table.to_csv(path, sep="\t", index=False, float_format="%.6g", compression={"method": "gzip", "compresslevel": 1})
        paths.append(path)
    assert frequency_tables.main([str(path) for path in paths] + ["--out", str(tmp_path / "out")]) == 0

    with (tmp_path / "out" / "frequency_alpha.tsv").open() as handle:
        written = list(csv.reader(handle, delimiter="\t"))
    assert written[0] == list(workspace_pipeline.FREQUENCY_ALPHA_COLUMNS)
    expected = []
    for path in paths:
        # the table as written (six significant digits), read back as the script reads it
        table = pd.read_csv(path, sep="\t")
        second = table["effect"].to_numpy() ** 2 + table["posterior_sd"].to_numpy() ** 2
        lines = workspace_pipeline.frequency_dependence(
            "ldl_cholesterol", path.name.split(".")[1], table["variant_class"].to_numpy(object), table["length"].to_numpy(np.float64),
            table["mean_value"].to_numpy(np.float64) / 2.0, second, None,
        )
        expected.extend([str(value) for value in line] for line in lines["frequency_alpha.tsv"])
    assert written[1:] == expected
    rows = [dict(zip(written[0], line)) for line in written[1:]]
    # the copy numbers are left out; no prior is read; the deletions' alpha lies below the SNVs'
    assert {row["variant_class"] for row in rows} == {"snv", "deletion", "all_structural"}
    assert all(row["prior_alpha"] == "" for row in rows)
    deletion = next(row for row in rows if row["arm"] == "with_sv" and row["variant_class"] == "deletion")
    assert float(deletion["alpha_minus_snv"]) < 0.0 and float(deletion["posterior_alpha_ci_low"]) < float(deletion["posterior_alpha"])
    workspace_pipeline.check_exportable(tmp_path / "out" / "effect_by_frequency.tsv")
