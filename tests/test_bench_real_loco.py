import itertools

import numpy as np
import pandas as pd

from benchmarks.bench_real import loco, report


def test_weighted_jackknife_with_equal_blocks_is_the_delete_one_group_jackknife():
    generator = np.random.default_rng(0)
    values = pd.Series(generator.normal(size=24))
    blocks = pd.Series(np.repeat([f"chr{index}" for index in range(6)], 4))
    leave_out = np.array([values[blocks != label].mean() for label in blocks.unique()])
    ordinary = np.sqrt(5 / 6 * ((leave_out - leave_out.mean()) ** 2).sum())
    mean, error, kind = report.jackknife(values, blocks)
    assert kind == "chromosome jackknife" and np.isclose(mean, values.mean()) and np.isclose(error, ordinary, rtol=1e-12)


def test_weighted_jackknife_of_a_mean_uses_each_block_mean_as_its_pseudo_value():
    generator = np.random.default_rng(1)
    sizes = np.array([1, 3, 7, 12, 30])
    blocks = pd.Series(np.repeat([f"chr{index}" for index in range(len(sizes))], sizes))
    values = pd.Series(generator.normal(size=sizes.sum()) + blocks.str[3:].astype(int))
    total, mean = sizes.sum(), values.mean()
    block_means = values.groupby(blocks, sort=True).mean().to_numpy()
    closed_form = np.sqrt((sizes / (total - sizes) * (block_means - mean) ** 2).sum() / len(sizes))
    leave_out = np.array([values[blocks != label].mean() for label in sorted(blocks.unique())])
    centre, error = report.weighted_jackknife(mean, leave_out, sizes)
    assert np.isclose(centre, mean, rtol=1e-12) and np.isclose(error, closed_form, rtol=1e-12)
    assert np.isclose(report.jackknife(values, blocks)[1], closed_form, rtol=1e-12)


def test_gene_bootstrap_se_is_the_exact_bootstrap_variance_of_the_mean():
    values = pd.Series([0.1, 0.4, 0.25])
    resampled = [np.mean(draw) for draw in itertools.product(values, repeat=len(values))]
    result = loco.interval(values, pd.Series(["chr1", "chr2", "chr3"]))
    assert np.isclose(result["boot_se"], np.std(resampled), rtol=1e-12)
    assert np.isclose(result["loco_hi"] - result["mean"], loco.Z * result["loco_se"])


def test_gain_is_paired_per_gene_and_a_gene_failed_in_any_group_is_dropped():
    rows = []
    generator = np.random.default_rng(2)
    for gene in range(6):
        for group in report.SUPERPOPULATIONS:
            for feature_set in ("snv", "snv_sv"):
                r2 = np.nan if (gene == 5 and group == "EAS" and feature_set == "snv_sv") else generator.uniform()
                rows.append({"gene_id": f"g{gene}", "chrom": f"chr{gene % 3 + 1}", "method": "m", "feature_set": feature_set, "design": "loso",
                             "superpopulation": group, "r2": r2})
    scores = pd.DataFrame(rows)
    genes = loco.paired_genes(scores, "snv", "snv_sv")
    summary = loco.summarize(genes, "snv", "snv_sv").set_index(["superpopulation", "quantity"])
    kept = scores[scores["gene_id"] != "g5"].pivot_table(index=["gene_id", "superpopulation"], columns="feature_set", values="r2")
    expected = (kept["snv_sv"] - kept["snv"]).groupby(level="gene_id").mean().mean()
    assert summary.loc[(report.POOLED, "gain"), "genes"] == 5
    assert np.isclose(summary.loc[(report.POOLED, "gain"), "mean"], expected, rtol=1e-12)
    assert np.isclose(summary.loc[(report.POOLED, "snv_sv"), "mean"] - summary.loc[(report.POOLED, "snv"), "mean"], expected, rtol=1e-12)


def test_chromosome_bootstrap_resamples_whole_chromosomes_and_pairs_the_arms():
    generator = np.random.default_rng(3)
    blocks = pd.Series(np.repeat(["chr1", "chr2"], [3, 5]))
    values = pd.Series(np.r_[np.full(3, 0.1), np.full(5, 0.3)])
    draws = loco.cluster_bootstrap(values, blocks)
    # Two chromosomes: a draw holds chr1 twice, chr2 twice, or one of each (probability 1/2).
    outcomes = {0.1: 0.25, 0.3: 0.25, values.mean(): 0.5}
    for outcome, probability in outcomes.items():
        share = np.isclose(draws, outcome).mean()
        assert abs(share - probability) < 4 * np.sqrt(probability * (1 - probability) / loco.DRAWS)
    assert np.isclose(draws, np.array(list(outcomes))[:, None]).any(axis=0).all()
    baseline, joint = pd.Series(generator.uniform(size=8)), pd.Series(generator.uniform(size=8))
    assert np.allclose(loco.cluster_bootstrap(joint - baseline, blocks), loco.cluster_bootstrap(joint, blocks) - loco.cluster_bootstrap(baseline, blocks))
    result = loco.interval(values, blocks)
    assert result["cluster_lo"] <= result["mean"] <= result["cluster_hi"]
