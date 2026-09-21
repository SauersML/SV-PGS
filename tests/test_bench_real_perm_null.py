"""The permutation null runs the benchmark's own fit and score, on a synthetic dataset.

The identity replicate must reproduce baselines.top_variant plus harness.predict_for_truth plus report.py's
within-group partial r^2; so must a permuted replicate, against the same pipeline fed the permuted SV columns.
"""

import dataclasses
import json

import numpy as np
import pandas as pd

from benchmarks.bench_real import baselines, harness, perm_null, report, splits

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


def permuted_rows(values, local_inverse_row):
    """The rows of `values` under the permutation sigma that `local_inverse_row` inverts: x[sigma]."""
    forward = np.empty_like(local_inverse_row)
    forward[local_inverse_row] = np.arange(len(local_inverse_row))
    return values[forward]


def direct_gain(dataset, gene_row, replicate=None, inverse=None):
    """The gain the benchmark itself computes: top_variant fitted on the training data, scored by
    harness.predict_for_truth, and read by report.py's within-group partial r^2. With a replicate, every SV column
    of both arms is first permuted by that replicate's own permutation of the split's people."""
    window = harness.load_gene_window(dataset, gene_row)
    gains = []
    for name in sorted(split for split in dataset.splits if split.startswith("loso/")):
        train_all, test_all, test_phenotype, test_index = harness.build_gene_task(dataset, window, dataset.splits[name])
        train_index = np.array([dataset.sample_index[sample] for sample in dataset.splits[name]["train"]])
        basis, _ = report.group_basis(dataset.covariates[test_index])
        truth = report.residual_on(basis, np.asarray(test_phenotype, dtype=np.float64)[None])
        values = {}
        for feature_set in ("snv", "snv_sv"):
            train, test = harness.subset(train_all, test_all, feature_set, name)
            if replicate is not None:
                is_sv = train.variants.is_sv
                genotypes, test_genotypes = train.genotypes.copy(), np.asarray(test, dtype=np.float64).copy()
                genotypes[:, is_sv] = permuted_rows(genotypes, perm_null.local_permutations(inverse, train_index, len(dataset.samples))[replicate])[:, is_sv]
                test_genotypes[:, is_sv] = permuted_rows(test_genotypes, perm_null.local_permutations(inverse, test_index, len(dataset.samples))[replicate])[:, is_sv]
                train, test = dataclasses.replace(train, genotypes=genotypes), test_genotypes
            predictor = baselines.top_variant(train)
            score, _ = harness.predict_for_truth(predictor, train, test, dataset.covariates[test_index])
            values[feature_set] = report.partial_scores(report.residual_on(basis, score[None]), truth)[1][0]
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


def test_a_permuted_replicate_is_the_pipeline_fed_the_permuted_sv_columns(tmp_path):
    """The null's whole speed rests on reading a permutation through permuted responses and permuted basis rows
    instead of permuting and re-projecting the columns. Each replicate must still be the fit and the score the
    benchmark would produce from the permuted genotypes themselves."""
    synthetic_dataset(tmp_path)
    dataset = harness.Dataset(tmp_path)
    inverse = perm_null.within_group_permutations(dataset.samples["Superpopulation"].to_numpy(), 6, np.random.default_rng(2))
    for gene_row in (0, 1):
        _, _, gains, wins = perm_null.gene_null(dataset, gene_row, inverse)
        for replicate in range(inverse.shape[0]):
            expected = direct_gain(dataset, gene_row, replicate=replicate, inverse=inverse)
            assert abs(gains[replicate] - expected) < 1e4 * EPSILON, (gene_row, replicate, gains[replicate], expected)
        assert wins.max() <= 5 and wins.shape == (7,)
