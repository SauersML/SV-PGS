"""Checks of bench-real's robust scoring (benchmarks/bench_real/robust.py) on synthetic inputs only."""

import json

import numpy as np
import pandas as pd
from scipy import stats

from benchmarks.bench_real import robust

EPSILON = np.finfo(np.float64).eps
FALSE_ALARM = 1e-6  # the chance a correct implementation fails a Monte Carlo check


def test_integer_weights_equal_the_metrics_of_the_duplicated_sample():
    generator = np.random.default_rng(0)
    prediction, truth = generator.normal(size=(3, 9)), generator.normal(size=(3, 9))
    weights = generator.integers(0, 3, size=(4, 9)).astype(np.float64)
    weights[:, 0] += 1.0
    weighted = robust.group_metrics(weights, prediction, truth)
    for replicate in range(weights.shape[0]):
        repeat = weights[replicate].astype(int)
        expanded = robust.group_metrics(np.ones((1, repeat.sum())), np.repeat(prediction, repeat, axis=1), np.repeat(truth, repeat, axis=1))
        for metric in ("r2", "oos_r2"):
            assert np.allclose(weighted[metric][replicate], expanded[metric][0], rtol=0, atol=1e3 * EPSILON)


def test_out_of_sample_r2_is_one_minus_sse_over_sst():
    generator = np.random.default_rng(1)
    prediction, truth = generator.normal(size=(2, 30)), generator.normal(size=(2, 30))
    value = robust.group_metrics(np.ones((1, 30)), prediction, truth)["oos_r2"][0]
    direct = 1 - ((truth - prediction) ** 2).sum(axis=1) / ((truth - truth.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    assert np.allclose(value, direct, rtol=0, atol=1e2 * EPSILON)


def test_centred_out_of_sample_r2_ignores_a_group_mean_offset_and_matches_its_slope_form():
    generator = np.random.default_rng(6)
    truth = generator.normal(size=(1, 40))
    prediction = 0.7 * truth + generator.normal(size=(1, 40))
    base = robust.group_metrics(np.ones((1, 40)), prediction, truth)
    shifted = robust.group_metrics(np.ones((1, 40)), prediction + 5.0, truth)
    assert np.allclose(base["oos_r2_centred"], shifted["oos_r2_centred"], rtol=0, atol=1e3 * EPSILON)
    assert shifted["oos_r2"][0, 0] < base["oos_r2"][0, 0]
    p, t = prediction[0] - prediction.mean(), truth[0] - truth.mean()
    r = (p @ t) / np.sqrt((p @ p) * (t @ t))
    scale = np.sqrt((p @ p) / (t @ t))
    assert np.isclose(base["oos_r2_centred"][0, 0], 2 * r * scale - scale ** 2, rtol=0, atol=1e3 * EPSILON)


def test_the_floor_corrected_r2_is_unbiased_for_an_independent_prediction():
    generator = np.random.default_rng(2)
    count, draws = 20, 20000
    values = robust.group_metrics(np.ones((1, count)), generator.normal(size=(draws, count)), generator.normal(size=(draws, count)))
    a, b = 0.5, (count - 2) / 2
    r2_sd = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    bound = stats.norm.isf(FALSE_ALARM) * r2_sd / np.sqrt(draws)
    assert abs(values["r2"][0].mean() - robust.null_r2(count)) < bound
    assert abs(values["r2_floor"][0].mean()) < bound / (1 - robust.null_r2(count))


def test_family_weights_keep_every_family_whole():
    families = np.array(["a", "a", "b", "c", "c", "c", "d"])
    weights = robust.family_weights(families, 50, np.random.default_rng(3))
    for label in np.unique(families):
        members = weights[:, families == label]
        assert np.all(members == members[:, :1])
    counts = np.stack([weights[:, np.flatnonzero(families == label)[0]] for label in np.unique(families)], axis=1)
    assert np.all(counts.sum(axis=1) == len(np.unique(families)))


def synthetic_results(tmp_path):
    """Two methods under loso; gene 0 carries a planted SV effect; the other genes get no SV weight."""
    generator = np.random.default_rng(4)
    groups = np.repeat(robust.GROUPS, 24)
    samples = pd.DataFrame({"sample": [f"S{index}" for index in range(len(groups))], "Superpopulation": groups})
    samples["FamilyID"] = samples["sample"]
    samples.loc[1, "FamilyID"] = samples.loc[0, "FamilyID"]
    samples.to_csv(tmp_path / "samples.tsv", sep="\t", index=False)
    gene_count = 12
    genes = pd.DataFrame({"gene_id": [f"G{index}" for index in range(gene_count)], "chrom": [f"chr{1 + index % 3}" for index in range(gene_count)]})
    sv = generator.binomial(2, 0.3, size=len(samples)).astype(np.float64)
    truth = generator.normal(size=(gene_count, len(samples)))
    truth[0] += 3 * sv
    for method in ("alpha", "beta"):
        directory = tmp_path / "results" / method / "loso"
        directory.mkdir(parents=True)
        snv = truth + generator.normal(scale=3, size=truth.shape)
        full, masked = snv.copy(), snv.copy()
        full[0] = snv[0] + 3 * sv
        genes.to_csv(directory / "all.genes.tsv", sep="\t", index=False)
        np.save(directory / "all.truth.npy", truth.astype(np.float32))
        np.save(directory / "all.snv.predictions.npy", snv.astype(np.float32))
        np.save(directory / "all.snv.predictions_without_sv.npy", snv.astype(np.float32))
        np.save(directory / "all.snv_sv.predictions.npy", full.astype(np.float32))
        np.save(directory / "all.snv_sv.predictions_without_sv.npy", masked.astype(np.float32))
    return tmp_path


def test_end_to_end_split_is_exact_and_finds_only_the_planted_gene(tmp_path):
    root = synthetic_results(tmp_path)
    summary = robust.run([root / "results"], root, ["alpha", "beta"], ["loso"], replicates=200, chunk=64, out_dir=root / "out")
    tests = pd.read_csv(root / "out" / "sv_gene_tests.tsv.gz", sep="\t")
    for _, row in tests.iterrows():
        assert abs(row["gain"] - (row["sv_part"] + row["refit"])) < 1e2 * EPSILON
    r2 = tests[(tests["metric"] == "r2") & (tests["method"] == "alpha")].set_index("gene_id")
    assert r2.loc["G0", "sv_part_q"] < 1 - robust.LEVEL
    assert (r2.drop(index="G0")["gain_p"] == 1.0).all()
    assert summary["discoveries"]["alpha/loso/snv_sv/r2"]["gain_q_at_or_below"] == 1
    contrasts = pd.read_csv(root / "out" / "contrasts.tsv", sep="\t")
    assert {"alpha/loso: snv_sv - snv", "alpha/loso: snv_sv SV part", "alpha/loso: snv_sv SNV refit", "loso/snv: alpha - beta"} <= set(contrasts["contrast"])
    refit = contrasts[(contrasts["contrast"] == "alpha/loso: snv_sv SNV refit") & (contrasts["metric"] == "r2")].iloc[0]
    assert refit["estimate"] == 0 and refit["se"] == 0
    assert json.loads((root / "out" / "summary.json").read_text())["replicates"] == 200
    leave = pd.read_csv(root / "out" / "leave_one_chromosome_out.tsv", sep="\t")
    assert set(leave["left_out"]) == {"chr1", "chr2", "chr3"}
    groups = pd.read_csv(root / "out" / "per_group_contrasts.tsv", sep="\t")
    assert set(groups["group"]) == set(robust.GROUPS) and "alpha/loso: snv_sv SV part" in set(groups["contrast"])


def test_pooled_point_is_the_gene_mean_and_constant_statistics_have_no_spread():
    codes = np.array([0, 0, 1, 2])
    counts = np.random.default_rng(5).integers(0, 3, size=(30, 3)).astype(np.float64)
    counts[counts.sum(axis=1) == 0, 0] = 1
    counts[:, 0] = np.maximum(counts[:, 0], 1)
    point, boot = robust.pooled(np.full(4, 0.25), np.full((30, 4), 0.25), codes, counts)
    assert point == 0.25 and np.allclose(boot, 0.25, rtol=0, atol=EPSILON)
