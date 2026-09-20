"""The permutation null's identity replicate reproduces the harness's own top_variant fit, on a synthetic dataset."""

import json

import numpy as np
import pandas as pd

from benchmarks.bench_real import baselines, harness, perm_null, robust, splits

EPSILON = np.finfo(np.float64).eps


def synthetic_dataset(root):
    generator = np.random.default_rng(0)
    groups = np.repeat(perm_null.GROUPS, 12)
    samples = pd.DataFrame({"sample": [f"S{index:02d}" for index in range(len(groups))], "Superpopulation": groups})
    samples["FamilyID"], samples["Population"] = samples["sample"], samples["Superpopulation"]
    samples.to_csv(root / "samples.tsv", sep="\t", index=False)
    variant_count = 40
    positions = np.sort(generator.choice(np.arange(1, 50_000), size=variant_count, replace=False))
    is_sv = np.zeros(variant_count, dtype=bool)
    is_sv[::5] = True
    table = pd.DataFrame({"pos": positions, "end": positions, "id": [f"v{index}" for index in range(variant_count)], "ref_len": 1,
                          "alt_len": np.where(is_sv, 60, 1), "symbolic": False, "sv_type": np.where(is_sv, "INS", "."),
                          "sv_length": np.where(is_sv, 60, 0), "is_sv": is_sv, "source": "panel"})
    table.to_csv(root / "chr1.variants.tsv", sep="\t", index=False)
    dosage = generator.binomial(2, generator.uniform(0.1, 0.5, size=(variant_count, 1)), size=(variant_count, len(samples))).astype(np.int8)
    np.save(root / "chr1.dosage.npy", dosage)
    genes = pd.DataFrame({"gene_id": ["G0", "G1"], "chrom": ["chr1", "chr1"], "tss": [20_000, 30_000]})
    genes.to_csv(root / "genes.tsv", sep="\t", index=False)
    expression = generator.normal(size=(2, len(samples)))
    expression[0] += 1.5 * dosage[5]
    np.save(root / "expression.npy", expression)
    np.save(root / "covariates.npy", generator.normal(size=(len(samples), 2)))
    annotation = {gene: {"start": 1, "end": 2, "strand": "+", "exons": [[1, 2]], "coding_exons": []} for gene in genes["gene_id"]}
    (root / "gene_annotation.json").write_text(json.dumps(annotation))
    everyone = set(samples["sample"])
    design = splits.leave_one_superpopulation_out(samples)
    for split in design:
        split["train"] = sorted(everyone - set(split["test"]))
    (root / "splits.json").write_text(json.dumps(design))
    return samples


def direct_gain(dataset, gene_row):
    window = harness.load_gene_window(dataset, gene_row)
    groups = dataset.samples["Superpopulation"].to_numpy()
    gains = []
    for name in sorted(split for split in dataset.splits if split.startswith("loso/")):
        train_all, test_all, test_phenotype, test_index = harness.build_gene_task(dataset, window, dataset.splits[name])
        values = {}
        for feature_set in ("snv", "snv_sv"):
            train, test = harness.subset(train_all, test_all, feature_set, name)
            prediction = baselines.top_variant(train).predict(test)
            values[feature_set] = robust.group_metrics(np.ones((1, len(test_index))), prediction[None], test_phenotype[None])["r2"][0, 0]
        gains.append(values["snv_sv"] - values["snv"])
    return float(np.mean(gains))


def test_identity_replicate_matches_the_harness_and_null_permutations_stay_within_groups(tmp_path):
    samples = synthetic_dataset(tmp_path)
    dataset = harness.Dataset(tmp_path)
    groups = samples["Superpopulation"].to_numpy()
    inverse = perm_null.within_group_permutations(groups, 20, np.random.default_rng(1))
    assert np.array_equal(inverse[0], np.arange(len(groups)))
    assert all(np.array_equal(groups[row], groups) for row in inverse)
    assert all(np.array_equal(np.sort(row), np.arange(len(groups))) for row in inverse)
    for gene_row in (0, 1):
        _, _, gains, _ = perm_null.gene_null(dataset, gene_row, inverse)
        assert abs(gains[0] - direct_gain(dataset, gene_row)) < 1e4 * EPSILON
        assert gains.shape == (21,)
