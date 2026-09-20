"""Training-side leave-one-carrier-out refits, on the synthetic dataset of the permutation-null test."""

import numpy as np
import pandas as pd

from benchmarks.bench_real import harness, loco_refit
from tests.test_bench_real_perm_null import direct_gain, synthetic_dataset

EPSILON = np.finfo(np.float64).eps


def test_refits_drop_each_carrier_from_training_only(tmp_path):
    synthetic_dataset(tmp_path)
    dataset = harness.Dataset(tmp_path)
    table, dosage = dataset.chromosome("chr1")
    sv_row = int(np.flatnonzero(table["is_sv"].to_numpy())[1])
    artifacts = pd.DataFrame([{"method": "top_variant", "design": "loso", "gene_id": "G0", "gene_name": "G0", "chrom": "chr1",
                               "sv_id": table["id"].iloc[sv_row], "minor_allele_carriers": 1}])
    artifacts.to_csv(tmp_path / "artifacts.tsv", sep="\t", index=False)
    spec = "benchmarks/bench_real/baselines.py:top_variant"
    loco_refit.run(tmp_path, tmp_path / "artifacts.tsv", spec, "top_variant", "loso", deadline=float("inf"), out_dir=tmp_path / "out")
    rows = pd.read_csv(tmp_path / "out" / "loco_refit_top_variant_loso.tsv", sep="\t")
    sv = np.asarray(dosage[sv_row], dtype=np.float64)
    carriers = dataset.samples["sample"].to_numpy()[sv > 0 if sv.mean() <= 1 else sv < 2]
    assert sorted(rows["carrier"]) == sorted(carriers)
    assert np.allclose(rows["gain"], direct_gain(dataset, 0), rtol=0, atol=1e4 * EPSILON)
    carrier = rows["carrier"].iloc[0]
    for split in dataset.splits.values():
        split["train"] = [sample for sample in split["train"] if sample != carrier]
    assert abs(rows["gain_without_carrier"].iloc[0] - direct_gain(dataset, 0)) < 1e4 * EPSILON
