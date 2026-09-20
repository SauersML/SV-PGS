"""Leakage and bookkeeping checks of the bench-real harness on synthetic inputs (no MAGE or 1kGP data)."""

import numpy as np
import pandas as pd

from benchmarks.bench_real import baselines, harness, splits

EPSILON = np.finfo(np.float64).eps


def synthetic_variants(is_sv, source):
    count = len(is_sv)
    return harness.Variants(position=np.arange(count), end=np.arange(count), distance_to_tss=np.arange(count), is_sv=np.asarray(is_sv),
                            sv_type=np.array(["."] * count), sv_length=np.zeros(count), allele_length_change=np.zeros(count),
                            train_allele_frequency=np.full(count, 0.5), source=np.asarray(source))


def test_residualization_never_reads_test_phenotypes():
    generator = np.random.default_rng(0)
    phenotype, covariates = generator.normal(size=40), generator.normal(size=(40, 3))
    train_index, test_index = np.arange(30), np.arange(30, 40)
    train_first, test_first = harness.residualize(phenotype, covariates, train_index, test_index)
    changed = phenotype.copy()
    changed[test_index] += generator.normal(size=10) * 1e3
    train_second, test_second = harness.residualize(changed, covariates, train_index, test_index)
    assert np.array_equal(train_first, train_second)
    # The training fit is identical, so the adjusted test values differ by the change up to one rounding per subtraction.
    assert np.allclose(test_second - test_first, changed[test_index] - phenotype[test_index], rtol=0, atol=4 * EPSILON * np.abs(changed).max())


def test_feature_sets_select_the_documented_columns():
    variants = synthetic_variants([False, True, False, True], ["panel", "panel", "pangenie", "pangenie"])
    assert list(harness.feature_mask(variants, "snv", "g/s")) == [True, False, False, False]
    assert list(harness.feature_mask(variants, "snv_sv", "g/s")) == [True, True, False, False]
    assert list(harness.feature_mask(variants, "snv_pgsv", "g/s")) == [True, False, False, True]


def test_sv_masking_removes_exactly_the_sv_part_of_a_linear_prediction():
    generator = np.random.default_rng(1)
    genotypes = generator.binomial(2, 0.3, size=(50, 6)).astype(np.float64)
    test = generator.binomial(2, 0.3, size=(20, 6)).astype(np.float64)
    is_sv = np.array([False, True, False, False, True, False])
    train = harness.TrainData(gene_id="g", chrom="chr1", tss=0, genotypes=genotypes, phenotype=generator.normal(size=50),
                              variants=synthetic_variants(is_sv, ["panel"] * 6), superpopulation=np.array(["EUR"] * 50), population=np.array(["CEU"] * 50),
                              gene_start=0, gene_end=0, strand="+", exons=np.zeros((0, 2), dtype=np.int64), coding_exons=np.zeros((0, 2), dtype=np.int64))
    coefficients = generator.normal(size=6)
    predictor = baselines.LinearPredictor(0.3, coefficients)
    sv_part = predictor.predict(test) - predictor.predict(harness._without_structural_variants(train, test))
    expected = (test[:, is_sv] - genotypes[:, is_sv].mean(axis=0)) @ coefficients[is_sv]
    assert np.allclose(sv_part, expected, rtol=0, atol=64 * EPSILON * np.abs(test).max() * np.abs(coefficients).sum())


def test_random_folds_keep_families_together_and_balance_superpopulations():
    family_size = 3
    families = [f"family{index // family_size}" for index in range(60)]
    samples = pd.DataFrame({"sample": [f"sample{index}" for index in range(60)], "FamilyID": families,
                            "Superpopulation": ["AFR" if index < 30 else "EUR" for index in range(60)]})
    folds = splits.random_folds(samples)
    fold_of = {sample: position for position, fold in enumerate(folds) for sample in fold["test"]}
    assert len(fold_of) == 60
    assert samples.assign(fold=samples["sample"].map(fold_of)).groupby("FamilyID")["fold"].nunique().max() == 1
    counts = samples.assign(fold=samples["sample"].map(fold_of)).groupby(["Superpopulation", "fold"]).size()
    # Greedy placement of whole families keeps fold sizes within one family of each other.
    assert counts.max() - counts.min() <= family_size


def test_leave_one_superpopulation_out_partitions_the_samples():
    samples = pd.DataFrame({"sample": [f"s{index}" for index in range(10)], "FamilyID": [f"f{index}" for index in range(10)],
                            "Superpopulation": ["AFR", "AMR", "EAS", "EUR", "SAS"] * 2})
    held_out = [fold["test"] for fold in splits.leave_one_superpopulation_out(samples)]
    assert sorted(sum(held_out, [])) == sorted(samples["sample"])
    assert all(len(set(samples.set_index("sample").loc[group, "Superpopulation"])) == 1 for group in held_out)


def test_merged_intervals_are_the_disjoint_union():
    from benchmarks.bench_real import build_dataset

    assert build_dataset.merged([(10, 20), (5, 12), (22, 30), (21, 21), (40, 41)]) == [[5, 30], [40, 41]]


def test_training_constant_columns_are_dropped_even_when_heterozygous():
    samples = pd.DataFrame({"sample": ["a", "b", "c", "d"], "Superpopulation": ["EUR"] * 4, "Population": ["CEU"] * 4})
    table = pd.DataFrame({"pos": [1, 2, 3], "end": [1, 2, 3], "is_sv": [False, True, False], "sv_type": ["."] * 3, "sv_length": [0, 60, 0],
                          "alt_len": [1, 61, 1], "ref_len": [1, 1, 1], "source": ["panel", "pangenie", "panel"]})
    # Column 0 varies; column 1 is heterozygous in every training sample; column 2 is homozygous in every one.
    genotypes = np.array([[0, 1, 2], [1, 1, 2], [2, 1, 2], [1, 0, 1]], dtype=np.float32)
    window = harness.GeneWindow(gene_row=0, gene_id="g", chrom="chr1", tss=2, genotypes=genotypes, table=table)

    class FakeDataset:
        sample_index = {"a": 0, "b": 1, "c": 2, "d": 3}
        expression = np.array([[0.1, 0.4, -0.2, 0.3]])
        covariates = np.zeros((4, 0))
        gene_annotation = {"g": {"start": 1, "end": 3, "strand": "+", "exons": [], "coding_exons": []}}

    FakeDataset.samples = samples
    train, test, _, _ = harness.build_gene_task(FakeDataset, window, {"train": ["a", "b", "c"], "test": ["d"]})
    assert train.genotypes.shape == (3, 1) and test.shape == (1, 1)
    assert list(train.variants.position) == [1]


def test_run_end_to_end_on_a_tiny_synthetic_dataset(tmp_path):
    import hashlib
    import json

    generator = np.random.default_rng(3)
    sample_count, variant_count = 24, 8
    samples = pd.DataFrame({"sample": [f"s{index}" for index in range(sample_count)], "FamilyID": [f"f{index}" for index in range(sample_count)],
                            "FatherID": 0, "MotherID": 0, "Sex": 1, "Population": "CEU",
                            "Superpopulation": ["AFR", "EUR"] * (sample_count // 2)})
    samples.to_csv(tmp_path / "samples.tsv", sep="\t", index=False)
    pd.DataFrame({"chrom": ["chr1"], "start": [99], "end": [100], "gene_id": ["g1"], "tss": [100]}).to_csv(tmp_path / "genes.tsv", sep="\t", index=False)
    dosage = generator.binomial(2, 0.4, size=(variant_count, sample_count)).astype(np.int8)
    np.save(tmp_path / "chr1.dosage.npy", dosage)
    np.save(tmp_path / "expression.npy", (dosage[0] - dosage[3] + generator.normal(size=sample_count))[None, :].astype(np.float64))
    np.save(tmp_path / "covariates.npy", np.zeros((sample_count, 0)))
    is_sv = np.arange(variant_count) % 4 == 3
    pd.DataFrame({"pos": 100 + np.arange(variant_count), "end": 100 + np.arange(variant_count), "id": ".", "ref_len": 1, "alt_len": np.where(is_sv, 61, 1),
                  "symbolic": False, "sv_type": np.where(is_sv, "INS", "."), "sv_length": np.where(is_sv, 60, 0), "is_sv": is_sv,
                  "source": "panel"}).to_csv(tmp_path / "chr1.variants.tsv", sep="\t", index=False)
    split_list = [{"name": "loso/AFR", "test": list(samples["sample"][samples["Superpopulation"] == "AFR"]), "train": list(samples["sample"][samples["Superpopulation"] == "EUR"])},
                  {"name": "loso/EUR", "test": list(samples["sample"][samples["Superpopulation"] == "EUR"]), "train": list(samples["sample"][samples["Superpopulation"] == "AFR"])}]
    (tmp_path / "splits.json").write_text(json.dumps(split_list))
    (tmp_path / "splits.sha256").write_text("synthetic\n")
    (tmp_path / "gene_annotation.json").write_text(json.dumps({"g1": {"start": 90, "end": 110, "strand": "+", "exons": [[95, 105]], "coding_exons": []}}))
    method = f"{harness.__file__.rsplit('/', 1)[0]}/baselines.py:top_variant"
    (tmp_path / "screened.tsv").write_text("gene_id\tscore\ng1\t3.2\n")
    harness.run(tmp_path, method, "top_variant", "loso", ["chr1"], tmp_path / "results", 1, ("snv", "snv_sv"), gene_list=tmp_path / "screened.tsv")
    out = tmp_path / "results" / "top_variant" / "loso"
    record = json.loads((out / "chr1.run.json").read_text())
    assert record["genes"] == 1 and record["gene_prefix"] is None and record["splits_sha256"] == "synthetic"
    assert record["gene_list_sha256"] == hashlib.sha256((tmp_path / "screened.tsv").read_bytes()).hexdigest()
    truth = np.load(out / "chr1.truth.npy")
    for feature_set in ("snv", "snv_sv"):
        predictions = np.load(out / f"chr1.{feature_set}.predictions.npy")
        assert np.isfinite(predictions).all() and np.isfinite(truth).all()
    assert np.array_equal(np.load(out / "chr1.snv.predictions.npy"), np.load(out / "chr1.snv.predictions_without_sv.npy"))


def test_pooled_paired_difference_averages_each_gene_over_the_held_out_groups():
    from benchmarks.bench_real import report

    generator = np.random.default_rng(4)
    rows = []
    for gene in range(12):
        for group in report.SUPERPOPULATIONS:
            for method in ("a", "b"):
                rows.append({"gene_id": f"g{gene}", "chrom": f"chr{gene % 3 + 1}", "design": "loso", "superpopulation": group,
                             "method": method, "feature_set": "snv", "r2": generator.uniform()})
    scores = pd.DataFrame(rows)
    table = pd.DataFrame(report.paired(scores, ("a", "snv"), ("b", "snv")))
    pooled = table[table["superpopulation"] == report.POOLED].iloc[0]
    wide = scores.pivot_table(index=["gene_id", "superpopulation"], columns="method", values="r2")
    expected = (wide["a"] - wide["b"]).groupby(level="gene_id").mean().mean()
    assert pooled["genes"] == 12 and np.isclose(pooled["difference"], expected, rtol=0, atol=16 * EPSILON)
    assert set(table["superpopulation"]) == {report.POOLED, *report.SUPERPOPULATIONS}


def matched_fixture(seed):
    generator = np.random.default_rng(seed)
    count = 400
    is_sv = np.zeros(count, dtype=bool)
    is_sv[generator.choice(count, 12, replace=False)] = True
    source = np.where(np.arange(count) % 7 == 0, "pangenie", "panel")
    return harness.Variants(position=np.arange(count), end=np.arange(count), distance_to_tss=generator.integers(-10**6, 10**6, count),
                            is_sv=is_sv, sv_type=np.array(["."] * count), sv_length=np.zeros(count), allele_length_change=np.zeros(count),
                            train_allele_frequency=generator.uniform(0.01, 0.99, count), source=source)


def test_matched_small_variants_draw_one_panel_small_variant_per_panel_sv_reproducibly():
    variants = matched_fixture(5)
    mask = harness.feature_mask(variants, "snv_matched", "g/loso/AFR")
    panel_sv = (variants.source == "panel") & variants.is_sv
    assert mask.sum() == panel_sv.sum()
    assert not (mask & variants.is_sv).any() and (variants.source[mask] == "panel").all()
    assert np.array_equal(mask, harness.feature_mask(variants, "snv_matched", "g/loso/AFR"))


def test_matched_small_variant_is_the_nearest_on_frequency_and_distance():
    count = 6
    variants = harness.Variants(position=np.arange(count), end=np.arange(count), distance_to_tss=np.array([100, 100, 100_000, 5_000, 100, 90]),
                                is_sv=np.array([True, False, False, False, False, False]), sv_type=np.array(["."] * count),
                                sv_length=np.zeros(count), allele_length_change=np.zeros(count),
                                train_allele_frequency=np.array([0.2, 0.45, 0.2, 0.2, 0.05, 0.21]), source=np.array(["panel"] * count))
    assert list(np.flatnonzero(harness.feature_mask(variants, "snv_matched", "g/s"))) == [5]


def test_sv_only_feature_sets_select_the_documented_columns():
    variants = synthetic_variants([False, True, False, True], ["panel", "panel", "pangenie", "pangenie"])
    assert list(harness.feature_mask(variants, "sv", "g/s")) == [False, True, False, False]
    assert list(harness.feature_mask(variants, "pgsv", "g/s")) == [False, False, False, True]


def test_sealed_confirmation_genes_are_never_scored_outside_the_confirmation(tmp_path):
    pd.DataFrame({"chrom": ["chr1"] * 3, "start": [1, 2, 3], "end": [2, 3, 4], "gene_id": ["g1", "g2", "g3"], "tss": [2, 3, 4]}).to_csv(
        tmp_path / "genes.tsv", sep="\t", index=False)
    pd.DataFrame({"gene_id": ["g2"]}).to_csv(tmp_path / harness.SEALED_GENES, sep="\t", index=False)
    dataset = harness.Dataset.__new__(harness.Dataset)
    dataset.directory, dataset.genes = tmp_path, pd.read_csv(tmp_path / "genes.tsv", sep="\t")
    assert dataset.gene_rows(["chr1"]) == [0, 2]
    assert dataset.gene_rows(["chr1"], confirmation=True) == [1]
    pd.DataFrame({"gene_id": ["g1", "g2"]}).to_csv(tmp_path / "list.tsv", sep="\t", index=False)
    try:
        dataset.gene_rows(["chr1"], gene_list=tmp_path / "list.tsv")
    except ValueError as error:
        assert "sealed" in str(error)
    else:
        raise AssertionError("a gene list naming a sealed gene must be refused")


def test_long_read_feature_sets_select_their_source():
    variants = synthetic_variants([False, True, True, True, False], ["panel", "panel", "hgsvc3", "ont", "ont"])
    assert list(harness.feature_mask(variants, "hgsvc3", "g/s")) == [False, False, True, False, False]
    assert list(harness.feature_mask(variants, "snv_hgsvc3", "g/s")) == [True, False, True, False, False]
    assert list(harness.feature_mask(variants, "snv_ont", "g/s")) == [True, False, False, True, False]


def test_a_derived_dataset_must_carry_its_parents_sealed_genes(tmp_path):
    parent, child = tmp_path / "parent", tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    pd.DataFrame({"gene_id": ["g2"]}).to_csv(parent / harness.SEALED_GENES, sep="\t", index=False)
    (child / harness.PARENT_DATASET).write_text(str(parent) + "\n")
    dataset = harness.Dataset.__new__(harness.Dataset)
    dataset.directory = child
    try:
        dataset.sealed_genes()
    except ValueError as error:
        assert "sealed" in str(error)
    else:
        raise AssertionError("a derived dataset without the parent's sealed list must be refused")
    (child / harness.SEALED_GENES).write_bytes((parent / harness.SEALED_GENES).read_bytes())
    assert dataset.sealed_genes() == {"g2"}
