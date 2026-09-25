"""The bench-sim SBayesRC submission's annotation file: columns constant on the fitted records are left out."""

import importlib.util
from pathlib import Path

import numpy as np

SUBMISSION = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_sim" / "submissions" / "sbayesrc.py"
_spec = importlib.util.spec_from_file_location("sbayesrc_submission", SUBMISSION)
sbayesrc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sbayesrc)


def _variants(rng: np.random.Generator, n_var: int) -> dict:
    cls = rng.choice(4, size=n_var, p=(0.6, 0.2, 0.1, 0.1)).astype(np.int64)
    return {
        "cls": cls,
        "in_gene": (rng.random(n_var) < 0.3).astype(np.float64),
        "in_exon": (rng.random(n_var) < 0.1).astype(np.float64),
        "in_repeat": (rng.random(n_var) < 0.2).astype(np.float64),
        "log_tss_distance": rng.normal(9.0, 2.0, n_var),
        "log_sv_length": np.where(cls == 3, rng.normal(5.0, 1.0, n_var), 0.0),
    }


def test_every_annotation_is_kept_when_each_varies() -> None:
    variants = _variants(np.random.default_rng(0), 400)
    header, table = sbayesrc.annotation_table(variants, np.arange(400))
    assert header == (*sbayesrc.ANNOTATIONS, *sbayesrc.CLASSES)
    assert table.shape == (400, len(header))
    np.testing.assert_array_equal(table[:, header.index("SV")], (variants["cls"] == 3).astype(np.float64))


def test_columns_constant_without_the_structural_classes_are_left_out() -> None:
    variants = _variants(np.random.default_rng(1), 400)
    rows = np.flatnonzero(variants["cls"] <= 1)
    header, table = sbayesrc.annotation_table(variants, rows)
    assert header == ("in_gene", "in_exon", "log_tss_distance", "in_repeat", "INDEL")
    assert np.all(np.ptp(table, axis=0) > 0)
    np.testing.assert_array_equal(table[:, header.index("log_tss_distance")], variants["log_tss_distance"][rows])
