"""Leakage and bookkeeping checks of the bench-real harness on synthetic inputs (no MAGE or 1kGP data)."""

import pathlib

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


def test_the_cis_index_returns_exactly_the_overlapping_rows_including_the_spanning_ones(tmp_path):
    """The window's rows, against the definition read straight off the two columns, on a table with records long
    enough to span a whole window and with positions out of order."""
    generator = np.random.default_rng(31)
    count = 400
    position = generator.integers(1, 6_000_000, size=count)
    length = np.where(generator.random(count) < 0.1, generator.integers(1, 4_000_000, size=count), generator.integers(0, 300, size=count))
    table = pd.DataFrame({"pos": position, "end": position + length, "is_sv": length >= 50, "sv_type": ".", "sv_length": length,
                          "alt_len": 1, "ref_len": 1, "source": "panel"})
    table.to_csv(tmp_path / "chr7.variants.tsv", sep="\t", index=False)
    np.save(tmp_path / "chr7.dosage.npy", np.zeros((count, 2), dtype=np.int8))

    class FakeDataset(harness.Dataset):
        def __init__(self):
            self.directory, self.rows_dirs, self.overlay_dir = tmp_path, [], None
            self._chromosomes, self._cis_indexes, self._parent_position = {}, {}, None

    dataset = FakeDataset()
    for tss in (0, 500_000, 3_000_000, 5_999_999, *generator.integers(1, 6_000_000, size=20)):
        rows = dataset.cis_rows("chr7", int(tss))
        expected = np.flatnonzero((table["end"].to_numpy() >= tss - harness.CIS_RADIUS_BP) & (table["pos"].to_numpy() <= tss + harness.CIS_RADIUS_BP))
        assert np.array_equal(rows, expected)
    # A record that spans the window entirely is in it, and a start-position search alone would have missed it.
    spanning = np.flatnonzero((table["pos"].to_numpy() < 2_000_000 - harness.CIS_RADIUS_BP) & (table["end"].to_numpy() > 2_000_000 + harness.CIS_RADIUS_BP))
    assert len(spanning) and set(spanning) <= set(dataset.cis_rows("chr7", 2_000_000).tolist())


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


def tiny_dataset(tmp_path):
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
    return tmp_path


def test_run_end_to_end_on_a_tiny_synthetic_dataset(tmp_path):
    import hashlib
    import json

    tiny_dataset(tmp_path)
    method = f"{harness.__file__.rsplit('/', 1)[0]}/baselines.py:top_variant"
    (tmp_path / "screened.tsv").write_text("gene_id\tscore\ng1\t3.2\n")
    (tmp_path / "note.json").write_text(json.dumps({"label": "arm X", "tests": "green"}))
    harness.run(tmp_path, method, "top_variant", "loso", ["chr1"], tmp_path / "results", 1, ("snv", "snv_sv"), gene_list=tmp_path / "screened.tsv",
                note=tmp_path / "note.json")
    out = tmp_path / "results" / "top_variant" / "loso"
    record = json.loads((out / "chr1.run.json").read_text())
    assert record["genes"] == 1 and record["gene_prefix"] is None and record["splits_sha256"] == "synthetic"
    assert record["gene_list_sha256"] == hashlib.sha256((tmp_path / "screened.tsv").read_bytes()).hexdigest()
    assert record["note"] == {"label": "arm X", "tests": "green"}
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


def test_batch_contract_matches_the_per_gene_contract_for_a_per_gene_method(tmp_path):
    tiny_dataset(tmp_path)
    baselines_path = f"{harness.__file__.rsplit('/', 1)[0]}/baselines.py"
    (tmp_path / "batch_method.py").write_text(
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('baselines_for_batch', {baselines_path!r})\n"
        "baselines = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(baselines)\n\n\n"
        "def fit_batch(trains):\n"
        "    return [baselines.top_variant(train) for train in trains]\n")
    (tmp_path / "views_method.py").write_text(
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('baselines_for_views', {baselines_path!r})\n"
        "baselines = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(baselines)\n\n\n"
        "def fit_views(views):\n"
        "    for key in views:\n"
        "        yield key, baselines.top_variant(views[key])\n")
    harness.run(tmp_path, f"{baselines_path}:top_variant", "gene", "loso", ["chr1"], tmp_path / "results", 1, ("snv", "snv_sv"))
    harness.run(tmp_path, f"{tmp_path}/batch_method.py:fit_batch", "batch", "loso", ["chr1"], tmp_path / "results", 1, ("snv", "snv_sv"), contract="batch")
    harness.run(tmp_path, f"{tmp_path}/views_method.py:fit_views", "views", "loso", ["chr1"], tmp_path / "results", 1, ("snv", "snv_sv"), contract="views")
    for feature_set in ("snv", "snv_sv"):
        for kind in ("predictions", "predictions_without_sv"):
            reference = np.load(tmp_path / f"results/gene/loso/chr1.{feature_set}.{kind}.npy")
            for contract in ("batch", "views"):
                assert np.array_equal(reference, np.load(tmp_path / f"results/{contract}/loso/chr1.{feature_set}.{kind}.npy"))


def test_a_batch_refuses_a_sealed_gene(tmp_path):
    tiny_dataset(tmp_path)
    pd.DataFrame({"gene_id": ["g1"]}).to_csv(tmp_path / harness.SEALED_GENES, sep="\t", index=False)
    dataset = harness.Dataset(tmp_path)
    try:
        harness._LazyTrains(dataset, [0], "loso/AFR", "snv")
    except ValueError as error:
        assert "sealed" in str(error)
    else:
        raise AssertionError("a batch containing a sealed gene must be refused")


def test_gene_ranks_take_a_slice_of_the_list_in_list_order(tmp_path):
    pd.DataFrame({"chrom": ["chr1"] * 4, "start": [1, 2, 3, 4], "end": [2, 3, 4, 5], "gene_id": ["g1", "g2", "g3", "g4"], "tss": [2, 3, 4, 5]}).to_csv(
        tmp_path / "genes.tsv", sep="\t", index=False)
    pd.DataFrame({"gene_id": ["g3", "g1", "g4", "g2"]}).to_csv(tmp_path / "ranked.tsv", sep="\t", index=False)
    dataset = harness.Dataset.__new__(harness.Dataset)
    dataset.directory, dataset.genes = tmp_path, pd.read_csv(tmp_path / "genes.tsv", sep="\t")
    assert dataset.gene_rows(["chr1"], gene_list=tmp_path / "ranked.tsv", gene_ranks=(0, 2)) == [0, 2]
    assert dataset.gene_rows(["chr1"], gene_list=tmp_path / "ranked.tsv", gene_ranks=(2, 4)) == [1, 3]


def test_horvitz_thompson_total_is_exactly_unbiased_over_every_random_sample():
    import itertools

    from benchmarks.bench_real import genome_total

    values = np.array([0.3, -0.1, 0.7, 0.2, 0.05, 0.4, -0.2])
    targeted = {1, 4}
    population, sample_size = len(values), 3
    estimates = []
    for sample in itertools.combinations(range(population), sample_size):
        scored = sorted(targeted | set(sample))
        genes = pd.DataFrame({"gene_id": [f"g{index}" for index in scored], "chrom": [f"chr{index % 3 + 1}" for index in scored],
                              "y": values[scored], "targeted": [index in targeted for index in scored], "random": [index in sample for index in scored]})
        estimates.append(genome_total.horvitz_thompson_total(genes, population, sample_size)[0])
    assert np.isclose(np.mean(estimates), values.sum(), rtol=0, atol=64 * EPSILON)


def test_saved_sv_effects_reproduce_the_sv_part_of_the_prediction(tmp_path):
    tiny_dataset(tmp_path)
    method = f"{harness.__file__.rsplit('/', 1)[0]}/baselines.py:top_variant"
    harness.run(tmp_path, method, "top_variant", "loso", ["chr1"], tmp_path / "results", 1, ("snv_sv",))
    out = tmp_path / "results" / "top_variant" / "loso"
    effects = pd.read_csv(out / "chr1.sv_coefficients.tsv.gz", sep="\t")
    dataset = harness.Dataset(tmp_path)
    window = harness.load_gene_window(dataset, 0)
    full = np.load(out / "chr1.snv_sv.predictions.npy")[0].astype(np.float64)
    without = np.load(out / "chr1.snv_sv.predictions_without_sv.npy")[0].astype(np.float64)
    for split_name, rows in effects.groupby("split"):
        test_index = np.array([dataset.sample_index[sample] for sample in dataset.splits[split_name]["test"]])
        sv_part = (window.genotypes[np.ix_(test_index, rows["window_row"].to_numpy())] - rows["train_mean"].to_numpy()) @ rows["effect"].to_numpy()
        # The saved predictions are float32, so the comparison allows one float32 rounding of each prediction.
        assert np.allclose(sv_part, (full - without)[test_index], rtol=0, atol=4 * np.finfo(np.float32).eps * max(np.abs(full).max(), 1.0))


def test_duplicate_sv_calls_merge_into_one_event():
    from benchmarks.bench_real import sv_gene_table

    table = pd.DataFrame({"pos": [100, 5000, 105, 9000], "end": [2100, 5000, 2080, 9500], "sv_length": [2000, 300, 1975, 500]})
    generator = np.random.default_rng(6)
    base = generator.binomial(2, 0.3, size=40).astype(np.float64)
    genotypes = np.column_stack([base, generator.binomial(2, 0.3, size=40), generator.binomial(2, 0.3, size=40), base]).astype(np.float64)
    labels = sv_gene_table.events(np.arange(4), table, genotypes)
    # 0 and 2 share a span (reciprocal overlap); 0 and 3 share genotypes (r^2 = 1); 1 stands alone.
    assert labels[0] == labels[2] == labels[3] != labels[1]


def dense_least_squares(train):
    """A linear predictor with an effect on every column (the minimum-norm least-squares fit): the gene-row test
    needs nonzero SV effects, which the lead-variant baseline does not give."""
    from benchmarks.bench_real import baselines

    genotypes = np.asarray(train.genotypes, dtype=np.float64)
    center = genotypes.mean(axis=0)
    coefficients = np.linalg.lstsq(genotypes - center, train.phenotype - train.phenotype.mean(), rcond=None)[0]
    return baselines.LinearPredictor(train.phenotype.mean(), coefficients, center=center, scale=np.ones(genotypes.shape[1]))


def test_gene_row_decomposes_the_sv_part_of_a_real_harness_run(tmp_path):
    from benchmarks.bench_real import sv_gene_table

    tiny_dataset(tmp_path)
    method = f"{__file__}:dense_least_squares"
    harness.run(tmp_path, method, "dense", "loso", ["chr1"], tmp_path / "results", 1, ("snv_sv",))
    effects = pd.read_csv(tmp_path / "results/dense/loso/chr1.sv_coefficients.tsv.gz", sep="\t")
    superdups = tmp_path / "superdups.txt.gz"
    pd.DataFrame([[0, "chr1", 50, 150]]).to_csv(superdups, sep="\t", header=False, index=False)
    sv_gene_table.initialize(tmp_path, superdups, effects.groupby("gene_id"))
    record = sv_gene_table.gene_row("g1")
    assert record["svs"] == 2 and record["events"] in (1, 2)
    assert record["effective_events"] >= 1 and 0 <= record["lead_sv_max_r2_with_small_variant"] <= 1
    assert record["lead_sv_segmental_duplication_fraction"] == 1.0


def test_collapsed_and_copy_number_sets_select_their_source():
    variants = synthetic_variants([False, True, True, True, True], ["panel", "panel", "panel_merged", "gatksv", "hgsvc3_merged"])
    assert list(harness.feature_mask(variants, "snv_sv_merged", "g/s")) == [True, False, True, False, False]
    assert list(harness.feature_mask(variants, "snv_sv_cn", "g/s")) == [True, False, False, True, False]
    assert list(harness.feature_mask(variants, "hgsvc3_merged", "g/s")) == [False, False, False, False, True]


def test_cross_mappable_partner_inside_the_lead_sv_is_flagged(tmp_path):
    import gzip

    from benchmarks.bench_real import sv_gene_table

    with gzip.open(tmp_path / "crossmap.txt.gz", "wt") as handle:
        handle.write("ENSG1.4\tENSG2.1\t12.5\nENSG1.4\tENSG3.2\t3.0\n")
    with gzip.open(tmp_path / "genes.gtf.gz", "wt") as handle:
        handle.write('chr1\tX\tgene\t1000\t2000\t.\t+\t.\tgene_id "ENSG2.7";\n')
        handle.write('chr1\tX\tgene\t90000\t91000\t.\t+\t.\tgene_id "ENSG3.1";\n')
    table = pd.DataFrame({"gene_id": ["ENSG1.9"], "chrom": ["chr1"], "lead_sv_id": ["sv"], "lead_sv_start": [500], "lead_sv_end": [5000]})
    flagged = sv_gene_table.add_cross_mappability(table, tmp_path / "crossmap.txt.gz", tmp_path / "genes.gtf.gz")
    assert flagged["lead_sv_crossmappable_partners"].tolist() == [1] and flagged["lead_sv_max_crossmappability"].tolist() == [12.5]


def test_allele_lengths_are_signed_and_never_zeroed_below_the_sv_threshold():
    table = pd.DataFrame({"alt_len": [-1, -1, -1, 1, 50, 10, 1], "ref_len": [1, 1, 1, 50, 1, 1, 1],
                          "sv_length": [5000, 3000, 700, 49, 49, 0, 0], "sv_type": ["DEL", "DUP", "INV", "DEL", "INS", "INS", "."]})
    length, change = harness.allele_lengths(table)
    assert length.tolist() == [5000, 3000, 700, 49, 49, 9, 0]
    assert change.tolist() == [-5000, 3000, 0, -49, 49, 9, 0]


def test_imputed_sv_overlay_adds_columns_beside_the_called_ones(tmp_path):
    tiny_dataset(tmp_path)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    dosage = np.load(tmp_path / "chr1.dosage.npy")
    imputed = (dosage[[3, 7]] * 0.9 + 0.05).astype(np.float32)
    np.savez(overlay / "chr1.svimp.npz", rows=np.array([3, 7]), ds=imputed, dr2=np.array([0.8, 0.4]))
    dataset = harness.Dataset(tmp_path, overlay)
    window = harness.load_gene_window(dataset, 0)
    assert window.genotypes.shape[1] == dosage.shape[0] + 2 and list(window.table["source"].iloc[-2:]) == ["svimp", "svimp"]
    assert np.array_equal(window.genotypes[:, -2:], imputed.T)
    train, test, _, _ = harness.build_gene_task(dataset, window, dataset.splits["loso/AFR"])
    joint, _ = harness.subset(train, test, "snv_svimp", "loso/AFR")
    called, _ = harness.subset(train, test, "snv_sv", "loso/AFR")
    assert joint.variants.is_sv.sum() == called.variants.is_sv.sum() == 2
    assert (joint.variants.source[joint.variants.is_sv] == "svimp").all()
    assert joint.variants.reliability[joint.variants.is_sv].tolist() == [0.8, 0.4]
    assert (joint.variants.reliability[~joint.variants.is_sv] == 1.0).all() and (called.variants.reliability == 1.0).all()
    assert (joint.variants.concordance == 1.0).all() and (joint.variants.called_r2 == 1.0).all()


def test_views_refuse_sealed_genes_and_missing_or_extra_views(tmp_path):
    tiny_dataset(tmp_path)
    dataset = harness.Dataset(tmp_path)
    views = harness._LazyViews(dataset, [0], ["loso/AFR", "loso/EUR"], ["snv"])
    assert list(views) == [("g1", "loso/AFR", "snv"), ("g1", "loso/EUR", "snv")]
    assert views[("g1", "loso/AFR", "snv")].variants.chromosome_row is not None

    class Constant:
        def predict(self, genotypes):
            return np.zeros(genotypes.shape[0])

    try:
        list(harness._run_views(dataset, lambda views: {("g1", "loso/AFR", "snv"): Constant()}, [0], ["loso/AFR", "loso/EUR"], ["snv"]))
    except ValueError as error:
        assert "1 of 2" in str(error)
    else:
        raise AssertionError("a missing view must be refused")
    pd.DataFrame({"gene_id": ["g1"]}).to_csv(tmp_path / harness.SEALED_GENES, sep="\t", index=False)
    try:
        harness._LazyViews(harness.Dataset(tmp_path), [0], ["loso/AFR"], ["snv"])
    except ValueError as error:
        assert "sealed" in str(error)
    else:
        raise AssertionError("views containing a sealed gene must be refused")


def _every_sample(values, targeted, sample_size):
    import itertools

    from benchmarks.bench_real import genome_total

    results = []
    for sample in itertools.combinations(range(len(values)), sample_size):
        scored = sorted(targeted | set(sample))
        genes = pd.DataFrame({"gene_id": [f"g{index}" for index in scored], "chrom": [f"chr{index % 3 + 1}" for index in scored],
                              "y": values[scored], "targeted": [index in targeted for index in scored], "random": [index in sample for index in scored]})
        results.append(genome_total.horvitz_thompson_total(genes, len(values), sample_size))
    return np.array([total for total, _, _ in results]), np.array([error for _, error, _ in results])


def test_horvitz_thompson_variance_estimate_is_exactly_unbiased_over_every_random_sample():
    values = np.array([0.3, -0.1, 0.7, 0.2, 0.05, 0.4, -0.2, 0.9])
    totals, errors = _every_sample(values, {1, 4}, 4)
    # Every sample is equally likely, so the design variance is the population variance of the estimates.
    assert np.isclose(np.mean(errors ** 2), np.var(totals), rtol=0, atol=64 * EPSILON)


def test_horvitz_thompson_census_of_the_signal_has_no_design_error():
    # All of y sits in the targeted stratum, on one chromosome: the total is known exactly from every sample.
    values = np.array([0.0, 0.8, 0.0, 0.0, 0.6, 0.0, 0.0])
    totals, errors = _every_sample(values, {1, 4}, 3)
    assert np.allclose(totals, values.sum(), rtol=0, atol=64 * EPSILON)
    assert np.allclose(errors, 0.0, rtol=0, atol=64 * EPSILON)


def test_per_split_runs_merge_to_the_full_design_run(tmp_path):
    from benchmarks.bench_real import merge_splits

    tiny_dataset(tmp_path)
    method = f"{harness.__file__.rsplit('/', 1)[0]}/baselines.py:top_variant"
    harness.run(tmp_path, method, "full", "loso", ["chr1"], tmp_path / "results", 1, ("snv", "snv_sv"))
    for split in ("loso/AFR", "loso/EUR"):
        harness.run(tmp_path, method, "parts", "loso", ["chr1"], tmp_path / "results", 1, ("snv", "snv_sv"), split_subset=[split])
    merge_splits.merge(tmp_path / "results/parts/loso", "chr1")
    for name in ("snv.predictions", "snv_sv.predictions_without_sv", "truth"):
        assert np.array_equal(np.load(tmp_path / f"results/full/loso/chr1.{name}.npy"), np.load(tmp_path / f"results/parts/loso/chr1.merged.{name}.npy"),
                              equal_nan=True)


def test_ctyper_and_hprc2_sets_select_their_source():
    variants = synthetic_variants([False, True, True, True], ["panel", "panel", "ctyper", "hprc2"])
    assert list(harness.feature_mask(variants, "snv_ctyper", "g/s")) == [True, False, True, False]
    assert list(harness.feature_mask(variants, "snv_hprc2", "g/s")) == [True, False, False, True]


def test_undefined_quality_values_pass_through_as_nan(tmp_path):
    tiny_dataset(tmp_path)
    table = pd.read_csv(tmp_path / "chr1.variants.tsv", sep="\t")
    table["called_r2"] = np.where(table["is_sv"], np.nan, 1.0)
    table.to_csv(tmp_path / "chr1.variants.tsv", sep="\t", index=False)
    dataset = harness.Dataset(tmp_path)
    train, _, _, _ = harness.build_gene_task(dataset, harness.load_gene_window(dataset, 0), dataset.splits["loso/AFR"])
    assert np.isnan(train.variants.called_r2[train.variants.is_sv]).all()
    assert (train.variants.called_r2[~train.variants.is_sv] == 1.0).all()


def test_extra_rows_overlay_equals_the_full_derived_copy(tmp_path):
    import shutil

    (tmp_path / "parent").mkdir()
    parent = tiny_dataset(tmp_path / "parent")
    generator = np.random.default_rng(8)
    extra_table = pd.DataFrame({"pos": [100, 103, 106], "end": [100, 160, 106], "id": ["x1", "x2", "x3"], "ref_len": 1, "alt_len": -1, "symbolic": True,
                                "sv_type": ["DEL", "DUP", "CNV"], "sv_length": [50, 57, 900], "is_sv": True, "source": "gatksv",
                                "concordance": [0.99, 0.95, 1.0], "called_r2": [0.98, np.nan, 1.0]})
    extra_dosage = generator.integers(0, 3, size=(3, 24)).astype(np.int8)
    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()
    extra_table.to_csv(rows_dir / "chr1.variants.tsv", sep="\t", index=False)
    np.save(rows_dir / "chr1.dosage.npy", extra_dosage)
    # The full copy, built by the stated rule: parent rows then extra rows, measurement columns filled with 1.0, stable sort by pos.
    full = tmp_path / "full"
    shutil.copytree(parent, full)
    parent_table = pd.read_csv(parent / "chr1.variants.tsv", sep="\t")
    merged = pd.concat([parent_table.assign(concordance=1.0, called_r2=1.0), extra_table], ignore_index=True)
    order = np.argsort(merged["pos"].to_numpy(), kind="stable")
    merged.iloc[order].to_csv(full / "chr1.variants.tsv", sep="\t", index=False)
    np.save(full / "chr1.dosage.npy", np.vstack([np.load(parent / "chr1.dosage.npy"), extra_dosage])[order])
    overlaid = harness.load_gene_window(harness.Dataset(parent, rows_dirs=[rows_dir]), 0)
    copied = harness.load_gene_window(harness.Dataset(full), 0)
    assert np.array_equal(overlaid.genotypes, copied.genotypes) and np.array_equal(overlaid.chromosome_rows, copied.chromosome_rows)
    pd.testing.assert_frame_equal(overlaid.table, copied.table, check_dtype=False)


def test_sample_subset_cuts_every_split_and_refuses_placeholders(tmp_path):
    tiny_dataset(tmp_path)
    subset = tmp_path / "subset.txt"
    subset.write_text("\n".join(f"s{index}" for index in range(12)) + "\n")
    dataset = harness.Dataset(tmp_path, sample_subset=subset)
    assert all(set(split["train"]) | set(split["test"]) <= {f"s{index}" for index in range(12)} for split in dataset.splits.values())
    window = harness.load_gene_window(dataset, 0)
    train, _, _, _ = harness.build_gene_task(dataset, window, dataset.splits["loso/AFR"])
    assert len(train.phenotype) == 6
    dosage = np.load(tmp_path / "chr1.dosage.npy")
    dosage[:, 0] = -1
    np.save(tmp_path / "chr1.dosage.npy", dosage)
    try:
        harness.build_gene_task(dataset, harness.load_gene_window(harness.Dataset(tmp_path, sample_subset=subset), 0), dataset.splits["loso/AFR"])
    except ValueError as error:
        assert "placeholder" in str(error)
    else:
        raise AssertionError("a negative placeholder read for a split sample must be refused")


def test_scores_are_residualized_on_the_covariates_like_the_truth():
    generator = np.random.default_rng(9)
    train_count, test_count, variant_count, covariate_count = 80, 30, 6, 3
    covariates = generator.normal(size=(train_count + test_count, covariate_count))
    genotypes = (generator.binomial(2, 0.4, size=(train_count + test_count, variant_count)) + covariates[:, :1] > 1).astype(np.float64)
    effects = generator.normal(size=variant_count)
    is_sv = np.array([False, True, False, False, True, False])
    train = harness.TrainData(gene_id="g", chrom="chr1", tss=0, genotypes=genotypes[:train_count], phenotype=np.zeros(train_count),
                              variants=synthetic_variants(is_sv, ["panel"] * variant_count), superpopulation=np.array(["EUR"] * train_count),
                              population=np.array(["CEU"] * train_count), gene_start=0, gene_end=0, strand="+", exons=np.zeros((0, 2), dtype=np.int64),
                              coding_exons=np.zeros((0, 2), dtype=np.int64), covariates=covariates[:train_count])
    prediction, without_sv = harness.predict_for_truth(baselines.LinearPredictor(0.7, effects), train, genotypes[train_count:], covariates[train_count:])
    design_train = np.column_stack([np.ones(train_count), covariates[:train_count]])
    design_test = np.column_stack([np.ones(test_count), covariates[train_count:]])
    fitted, *_ = np.linalg.lstsq(design_train, genotypes[:train_count], rcond=None)
    expected = (genotypes[train_count:] - design_test @ fitted) @ effects
    tolerance = 1e3 * EPSILON * np.abs(genotypes).max() * np.abs(effects).sum()
    assert np.allclose(prediction, expected, rtol=0, atol=tolerance)
    assert np.allclose(prediction - without_sv, (genotypes[train_count:, is_sv] - design_test @ fitted[:, is_sv]) @ effects[is_sv], rtol=0, atol=tolerance)

    class Projected:
        """A predictor that already projects its genotypes on the covariates: the harness rule leaves it unchanged."""

        def predict(self, genotypes, covariates):
            design = np.column_stack([np.ones(len(genotypes)), covariates])
            return (genotypes - design @ fitted) @ effects

    again, _ = harness.predict_for_truth(Projected(), train, genotypes[train_count:], covariates[train_count:])
    assert np.allclose(again, expected, rtol=0, atol=tolerance)


def test_covariate_projected_genotypes_are_orthogonal_to_the_covariates():
    generator = np.random.default_rng(10)
    covariates = generator.normal(size=(50, 4))
    genotypes = generator.binomial(2, 0.3, size=(50, 7)).astype(np.float64)

    class Train:
        pass

    train = Train()
    train.covariates = covariates
    projected = baselines.covariate_projected(train, genotypes)
    design = np.column_stack([np.ones(50), covariates])
    assert np.abs(design.T @ projected).max() <= 1e3 * EPSILON * np.abs(genotypes).sum()


FAILING_METHOD = (
    "def fit(train):\n"
    "    if train.variants.is_sv.any():\n"
    "        raise RuntimeError('kernel failed to compile')\n"
    "    import numpy as np\n"
    "    class Predictor:\n"
    "        def predict(self, genotypes):\n"
    "            return genotypes[:, 0].astype(float)\n"
    "    return Predictor()\n")


def test_a_fit_failure_raises_by_default(tmp_path):
    tiny_dataset(tmp_path)
    (tmp_path / "failing.py").write_text(FAILING_METHOD)
    try:
        harness.run(tmp_path, f"{tmp_path}/failing.py:fit", "failing", "loso", ["chr1"], tmp_path / "results", 1, ("snv", "snv_sv"))
    except RuntimeError as error:
        assert "compile" in str(error)
    else:
        raise AssertionError("a fit that raises must stop the run, never be replaced by a stand-in predictor")


def test_recorded_failures_are_nan_logged_and_never_scored(tmp_path):
    import json

    from benchmarks.bench_real import report

    tiny_dataset(tmp_path)
    (tmp_path / "failing.py").write_text(FAILING_METHOD)
    harness.run(tmp_path, f"{tmp_path}/failing.py:fit", "failing", "loso", ["chr1"], tmp_path / "results", 1, ("snv", "snv_sv"), record_failures=True)
    out = tmp_path / "results/failing/loso"
    assert np.isnan(np.load(out / "chr1.snv_sv.predictions.npy")).all()
    assert np.isfinite(np.load(out / "chr1.snv.predictions.npy")).all()
    log = pd.read_csv(out / "chr1.log.tsv", sep="\t")
    assert (log.loc[log["feature_set"] == "snv_sv", "status"].str.startswith("failed: RuntimeError")).all()
    assert (log.loc[log["feature_set"] == "snv", "status"] == "ok").all()
    assert json.loads((out / "chr1.run.json").read_text())["failed_fits"] == 2
    scores = report.per_gene_scores(tmp_path / "results", tmp_path, "failing", "loso")
    table = pd.DataFrame(report.paired(scores, ("failing", "snv_sv"), ("failing", "snv")))
    pooled = table[table["superpopulation"] == report.POOLED].iloc[0]
    assert pooled["genes"] == 0 and pooled["failed_genes"] == 1
    # Intention-to-treat scores the failed arm as the training-mean prediction (r^2 = 0) on every gene.
    assert pooled["itt_genes"] == 1 and pooled["itt_mean_r2_a"] == 0.0 and np.isclose(pooled["itt_difference"], -pooled["itt_mean_r2_b"])


def test_raw_scores_reproduce_the_scored_predictions_and_merge_along_splits(tmp_path):
    import json

    from benchmarks.bench_real import merge_splits

    tiny_dataset(tmp_path)
    method = f"{harness.__file__.rsplit('/', 1)[0]}/baselines.py:top_variant"
    harness.run(tmp_path, method, "full", "loso", ["chr1"], tmp_path / "results", 1, ("snv_sv",))
    out = tmp_path / "results/full/loso"
    raw = np.load(out / "chr1.snv_sv.raw_scores.npy").astype(np.float64)
    splits_order = json.loads((out / "chr1.raw_splits.json").read_text())
    predictions = np.load(out / "chr1.snv_sv.predictions.npy").astype(np.float64)
    dataset = harness.Dataset(tmp_path)
    for position, split_name in enumerate(splits_order):
        train_index = np.array([dataset.sample_index[s] for s in dataset.splits[split_name]["train"]])
        test_index = np.array([dataset.sample_index[s] for s in dataset.splits[split_name]["test"]])
        design = np.column_stack([np.ones(len(dataset.samples)), dataset.covariates])
        coefficients, *_ = np.linalg.lstsq(design[train_index], raw[0, position, train_index], rcond=None)
        rescored = raw[0, position, test_index] - design[test_index] @ coefficients
        assert np.allclose(rescored, predictions[0, test_index], rtol=0, atol=64 * np.finfo(np.float32).eps * max(np.abs(raw).max(), 1.0))
    for split in ("loso/AFR", "loso/EUR"):
        harness.run(tmp_path, method, "parts", "loso", ["chr1"], tmp_path / "results", 1, ("snv_sv",), split_subset=[split])
    merge_splits.merge(tmp_path / "results/parts/loso", "chr1")
    merged = np.load(tmp_path / "results/parts/loso/chr1.merged.snv_sv.raw_scores.npy")
    assert merged.shape[1] == 2 and json.loads((tmp_path / "results/parts/loso/chr1.merged.raw_splits.json").read_text()) == ["loso/AFR", "loso/EUR"]


def within_group_fixture(tmp_path, per_group=40, covariate_count=3, gene_count=3):
    """A loso results directory written the way the harness writes it: the truth is y minus the training OLS fit on
    [1, C], stored in float32; the predictions are a genetic score plus a different covariate combination per split."""
    import json

    generator = np.random.default_rng(11)
    groups = np.repeat(["AFR", "EUR"], per_group)
    count = len(groups)
    samples = pd.DataFrame({"sample": [f"s{index}" for index in range(count)], "Superpopulation": groups})
    samples.to_csv(tmp_path / "samples.tsv", sep="\t", index=False)
    covariates = generator.normal(size=(count, covariate_count)) + (groups == "AFR")[:, None] * 3.0
    genetic = generator.normal(size=(gene_count, count))
    expression = genetic + (covariates @ generator.normal(size=(covariate_count, gene_count))).T * 2 + generator.normal(size=(gene_count, count))
    np.save(tmp_path / "covariates.npy", covariates)
    np.save(tmp_path / "expression.npy", expression)
    genes = pd.DataFrame({"chrom": ["chr1", "chr2", "chr3"][:gene_count], "gene_id": [f"g{index}" for index in range(gene_count)]})
    genes.to_csv(tmp_path / "genes.tsv", sep="\t", index=False)
    split_list = [{"name": f"loso/{group}", "test": list(samples["sample"][groups == group]), "train": list(samples["sample"][groups != group])}
                  for group in ("AFR", "EUR")]
    (tmp_path / "splits.json").write_text(json.dumps(split_list))
    out = tmp_path / "results/m/loso"
    out.mkdir(parents=True)
    truth, predictions = np.full((gene_count, count), np.nan, dtype=np.float32), np.full((gene_count, count), np.nan)
    design = np.column_stack([np.ones(count), covariates])
    for group in ("AFR", "EUR"):
        test, train = np.flatnonzero(groups == group), np.flatnonzero(groups != group)
        for gene in range(gene_count):
            truth[gene, test] = harness.residualize(expression[gene], covariates, train, test)[1]
            predictions[gene, test] = genetic[gene, test] + generator.normal(size=count)[test] + design[test] @ generator.normal(size=covariate_count + 1) * 5
    genes.to_csv(out / "chr.genes.tsv", sep="\t", index=False)
    np.save(out / "chr.truth.npy", truth)
    np.save(out / "chr.snv.predictions.npy", predictions)
    return samples, covariates, expression, predictions


def closed_form_partial_r2(prediction, expression, covariates):
    design = np.column_stack([np.ones(len(prediction)), covariates])
    residual = lambda values: values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
    return np.corrcoef(residual(prediction), residual(expression))[0, 1] ** 2


def test_within_group_partial_r2_matches_its_closed_form_and_ignores_covariate_terms(tmp_path):
    from benchmarks.bench_real import report

    report.held_out.cache_clear()
    samples, covariates, expression, predictions = within_group_fixture(tmp_path)
    scores = report.per_gene_scores(tmp_path / "results", tmp_path, "m", "loso")
    assert len(scores) == 6 and set(scores["superpopulation"]) == {"AFR", "EUR"}
    for row in scores.itertuples():
        members = np.flatnonzero(samples["Superpopulation"].to_numpy() == row.superpopulation)
        gene = int(row.gene_id[1:])
        expected = closed_form_partial_r2(predictions[gene, members], expression[gene, members], covariates[members])
        assert np.isclose(row.r2, expected, rtol=0, atol=1e3 * EPSILON)
        # The exact null expectation: people minus the rank of [1, C_T].
        assert row.null_r2 == 1.0 / (40 - 4) and row.people == 40
        assert np.isfinite(row.mismatched_r2)
    # Any covariate combination added to the score, or taken from the truth, leaves the metric unchanged.
    shifted = predictions + (np.column_stack([np.ones(len(covariates)), covariates]) @ np.arange(1.0, 5.0))[None, :]
    np.save(tmp_path / "results/m/loso/chr.snv.predictions.npy", shifted)
    again = report.per_gene_scores(tmp_path / "results", tmp_path, "m", "loso")
    assert np.allclose(again["r2"], scores["r2"], rtol=0, atol=1e3 * EPSILON)


def flaky_top_variant(train):
    """A method that records every gene it fits and refuses g2 while a sentinel file is there, so a run can be made
    to die after some fits have finished. Its own file's digest, which the run identity uses, never changes."""
    import os

    root = pathlib.Path(os.environ["BENCH_REAL_TEST_ROOT"])
    with (root / "fitted.log").open("a") as log:
        log.write(train.gene_id + "\n")
    if (root / "fail").exists() and train.gene_id == "g2":
        raise RuntimeError("the worker died")
    return baselines.top_variant(train)


def two_gene_dataset(tmp_path):
    import json

    tiny_dataset(tmp_path)
    genes = pd.read_csv(tmp_path / "genes.tsv", sep="\t")
    pd.concat([genes, genes.assign(gene_id="g2", tss=104)], ignore_index=True).to_csv(tmp_path / "genes.tsv", sep="\t", index=False)
    expression = np.load(tmp_path / "expression.npy")
    np.save(tmp_path / "expression.npy", np.vstack([expression, np.roll(expression, 5, axis=1)]))
    annotation = json.loads((tmp_path / "gene_annotation.json").read_text())
    annotation["g2"] = annotation["g1"]
    (tmp_path / "gene_annotation.json").write_text(json.dumps(annotation))


def test_a_run_that_dies_keeps_its_finished_fits_and_the_same_run_again_fits_only_what_is_missing(tmp_path, monkeypatch):
    """Fits used to live in the parent process's memory until the last one arrived, so anything that stopped a run
    threw away every finished fit. Each one is now a file of its own, and the same run started again carries them."""
    import json

    two_gene_dataset(tmp_path)
    monkeypatch.setenv("BENCH_REAL_TEST_ROOT", str(tmp_path))
    (tmp_path / "fail").touch()
    method = f"{__file__}:flaky_top_variant"
    arguments = (tmp_path, method, "m", "loso", ["chr1"], tmp_path / "results", 1, ("snv", "snv_sv"))
    with pytest.raises(RuntimeError, match="the worker died"):
        harness.run(*arguments)
    out = tmp_path / "results/m/loso"
    parts = out / "chr1.parts"
    # g1 finished: two splits by two feature sets. g2 died on its first fit and left none.
    assert sorted(path.name.split(".", 1)[1] for path in parts.glob("*.npz")) == ["0.snv.npz", "0.snv_sv.npz", "1.snv.npz", "1.snv_sv.npz"]
    assert len((out / "chr1.progress.jsonl").read_text().splitlines()) == 4
    assert (tmp_path / "fitted.log").read_text().split() == ["g1"] * 4 + ["g2"]
    # A different run may not take over this one's output directory, whichever of the two records it by itself.
    (out / "chr1.run.json").unlink()
    with pytest.raises(ValueError, match="holds the finished fits of run"):
        harness.run(tmp_path, method, "m", "loso", ["chr1"], tmp_path / "results", 1, ("snv",))
    (tmp_path / "fail").unlink()
    (tmp_path / "fitted.log").unlink()
    harness.run(*arguments)
    assert (tmp_path / "fitted.log").read_text().split() == ["g2"] * 4
    assert not parts.exists()
    assert len((out / "chr1.progress.jsonl").read_text().splitlines()) == 8
    assert len(pd.read_csv(out / "chr1.log.tsv", sep="\t")) == 8
    assert json.loads((out / "chr1.run.json").read_text())["failed_fits"] == 0
    # What the run wrote is what a run that never died writes.
    harness.run(tmp_path, method, "clean", "loso", ["chr1"], tmp_path / "results", 1, ("snv", "snv_sv"))
    clean = tmp_path / "results/clean/loso"
    for name in ("snv.predictions", "snv_sv.predictions", "snv_sv.predictions_without_sv", "snv_sv.raw_scores", "truth"):
        assert np.array_equal(np.load(out / f"chr1.{name}.npy"), np.load(clean / f"chr1.{name}.npy"), equal_nan=True)
    assert (pd.read_csv(out / "chr1.sv_coefficients.tsv.gz", sep="\t") == pd.read_csv(clean / "chr1.sv_coefficients.tsv.gz", sep="\t")).all().all()


def test_a_run_of_a_different_identity_may_not_overwrite_an_existing_run(tmp_path):
    """The output path and tag name a chromosome, not a run: the same tag serves a different method file, inference,
    dataset, gene set or feature sets. The run's own identity is written before the first fit and checked against it."""
    import json

    tiny_dataset(tmp_path)
    method = f"{harness.__file__.rsplit('/', 1)[0]}/baselines.py:top_variant"
    harness.run(tmp_path, method, "m", "loso", ["chr1"], tmp_path / "results", 1, ("snv",))
    record = json.loads((tmp_path / "results/m/loso/chr1.run.json").read_text())
    assert len(record["run_id"]) == 64 and len(record["gene_ids_sha256"]) == 64
    # The same run again is the same identity, and writes the same outputs.
    harness.run(tmp_path, method, "m", "loso", ["chr1"], tmp_path / "results", 1, ("snv",))
    assert json.loads((tmp_path / "results/m/loso/chr1.run.json").read_text())["run_id"] == record["run_id"]
    for different in (dict(feature_sets=("snv", "snv_sv")), dict(record_failures=True)):
        with pytest.raises(ValueError, match="is the record of run"):
            harness.run(tmp_path, method, "m", "loso", ["chr1"], tmp_path / "results", 1, **different)


def test_the_report_reads_several_results_roots_with_different_genes_and_refuses_a_gene_scored_twice(tmp_path, monkeypatch):
    """One method in gene-range chunks, one directory each, beside a comparator that scored other genes: the roots
    score different gene sets, and one root may hold no genes of a method at all. A gene scored twice for one arm is
    refused, whichever root or tag holds the copy."""
    import sys

    from benchmarks.bench_real import report

    report.held_out.cache_clear()
    within_group_fixture(tmp_path, gene_count=3)
    source = tmp_path / "results/m/loso"
    genes = pd.read_csv(source / "chr.genes.tsv", sep="\t")
    truth, predictions = np.load(source / "chr.truth.npy"), np.load(source / "chr.snv.predictions.npy")
    roots = []
    for chunk, rows in enumerate(([0, 1], [2])):
        root = tmp_path / f"chunk{chunk}"
        out = root / "m/loso"
        out.mkdir(parents=True)
        genes.iloc[rows].to_csv(out / "chr.genes.tsv", sep="\t", index=False)
        np.save(out / "chr.truth.npy", truth[rows])
        np.save(out / "chr.snv.predictions.npy", predictions[rows])
        (root / "other/loso").mkdir(parents=True)  # a method this root holds no genes of
        roots.append(str(root))
    out = pathlib.Path(roots[0]) / "other/loso"
    genes.iloc[[0]].to_csv(out / "chr.genes.tsv", sep="\t", index=False)
    np.save(out / "chr.truth.npy", truth[[0]])
    np.save(out / "chr.snv.predictions.npy", predictions[[0]])
    empty = report.per_gene_scores(pathlib.Path(roots[1]), tmp_path, "other", "loso")
    assert list(empty.columns) == list(report.SCORE_COLUMNS) and len(empty) == 0
    destination = tmp_path / "report"
    destination.mkdir()
    arguments = ["report.py", "--results", *roots, "--dataset", str(tmp_path), "--methods", "m", "other", "--out", str(destination)]
    monkeypatch.setattr(sys, "argv", arguments)
    report.main()
    scored = pd.read_csv(destination / "per_gene_r2.tsv.gz", sep="\t")
    assert sorted(scored.loc[scored["method"] == "m", "gene_id"].unique()) == ["g0", "g1", "g2"]
    assert sorted(scored.loc[scored["method"] == "other", "gene_id"].unique()) == ["g0"]
    # The same gene in a second root for the same arm is the double count that must stop the report.
    duplicate = tmp_path / "chunk1/m/loso"
    genes.iloc[[0]].to_csv(duplicate / "chr2.genes.tsv", sep="\t", index=False)
    np.save(duplicate / "chr2.truth.npy", truth[[0]])
    np.save(duplicate / "chr2.snv.predictions.npy", predictions[[0]])
    with pytest.raises(ValueError, match="scored more than once"):
        report.main()


def test_the_analytic_null_floor_is_the_mean_r2_of_a_score_uniform_in_the_residual_subspace():
    """The null the floor 1 / (n_T - rank[1, C_T]) belongs to, checked where it holds: a score whose within-group
    residual points uniformly in the residual subspace, independent of the expression. r2 is then the squared cosine
    of two uniform directions, Beta(1/2, (d-1)/2), with mean 1/d and variance 2(d-1) / (d^2 (d+2)). Math only."""
    from benchmarks.bench_real import report

    generator = np.random.default_rng(21)
    people, covariate_count, draws = 40, 3, 20_000
    basis, rank = report.group_basis(generator.normal(size=(people, covariate_count)))
    dimension = people - rank
    truth = report.residual_on(basis, generator.normal(size=(1, people)))
    scores = report.residual_on(basis, generator.normal(size=(draws, people)))
    _, r2, _ = report.partial_scores(scores, np.repeat(truth, draws, axis=0))
    error = np.sqrt(2 * (dimension - 1) / (dimension ** 2 * (dimension + 2)) / draws)
    assert abs(r2.mean() - 1.0 / dimension) < 4 * error
    assert np.isclose(r2.var(), 2 * (dimension - 1) / (dimension ** 2 * (dimension + 2)), rtol=0.1)


def test_the_signed_partial_correlation_and_the_squared_error_skill_sit_beside_the_squared_one(tmp_path):
    """r2 is the square of a partial correlation: it charges a score neither its sign nor its scale. The report
    names and writes all three, so a table cannot be read as if one were the other."""
    from benchmarks.bench_real import report

    report.held_out.cache_clear()
    _, _, _, predictions = within_group_fixture(tmp_path)
    scores = report.per_gene_scores(tmp_path / "results", tmp_path, "m", "loso")
    assert np.allclose(scores["partial_correlation"] ** 2, scores["r2"], rtol=0, atol=16 * EPSILON)
    for multiple, flips in ((-1.0, True), (3.0, False)):
        np.save(tmp_path / "results/m/loso/chr.snv.predictions.npy", predictions * multiple)
        again = report.per_gene_scores(tmp_path / "results", tmp_path, "m", "loso")
        assert np.allclose(again["r2"], scores["r2"], rtol=0, atol=1e3 * EPSILON)
        assert np.allclose(again["partial_correlation"], scores["partial_correlation"] * (-1 if flips else 1), rtol=0, atol=1e3 * EPSILON)
        # The squared-error skill does charge both, and is never clipped at zero.
        assert not np.allclose(again["oos_r2"], scores["oos_r2"]) and (again["oos_r2"] < 0).any()
    pooled = report.pooled_r2(again)
    assert np.isclose(pooled["mean_partial_correlation"].iloc[0], again.groupby("gene_id")["partial_correlation"].mean().mean(), rtol=0, atol=1e3 * EPSILON)


def test_raw_scores_score_from_the_run_record_when_the_split_file_is_missing(tmp_path):
    """An older results layout kept its raw scores without a <tag>.raw_splits.json. The run's own record lists the
    same splits in the same order, so the same command scores the old and the current layout alike; a layout with
    neither is refused by name, and a column count that does not match the list is refused too."""
    import json

    from benchmarks.bench_real import report

    report.held_out.cache_clear()
    within_group_fixture(tmp_path)
    out = tmp_path / "results/m/loso"
    order = ["loso/AFR", "loso/EUR"]
    raw = np.arange(3 * 2 * 80, dtype=np.float32).reshape(3, 2, 80)
    raw[0] = 2.5  # gene 0's fit is constant over its people, train and test: it predicts no differences at all
    np.save(out / "chr.snv.raw_scores.npy", raw)
    (out / "chr.raw_splits.json").write_text(json.dumps(order))
    data = report.held_out(tmp_path)
    expected = data.predictions(out, "chr", "snv", masked=False)
    assert (expected[0] == 0.0).all()
    (out / "chr.raw_splits.json").unlink()
    for record in ({"splits": order}, {"merged_from": ["a", "b"], "parts": [{"splits": order[:1]}, {"splits": order[1:]}]}):
        (out / "chr.run.json").write_text(json.dumps(record))
        assert np.array_equal(data.predictions(out, "chr", "snv", masked=False), expected, equal_nan=True)
    (out / "chr.run.json").write_text(json.dumps({"splits": order[:1]}))
    with pytest.raises(ValueError, match="names 1 splits for 2 raw-score columns"):
        data.predictions(out, "chr", "snv", masked=False)
    (out / "chr.run.json").unlink()
    with pytest.raises(ValueError, match="raw scores have no split order"):
        data.predictions(out, "chr", "snv", masked=False)


def test_a_foreign_dataset_is_refused_when_no_person_is_held_out_for_every_gene(tmp_path):
    """The truth is checked against the dataset's expression gene group by gene group, on each group's own people.
    Checking only the people every gene row has a finite truth for left the rest unchecked, and when missingness
    differs enough from gene to gene that intersection is empty, which passed as "verified"."""
    from benchmarks.bench_real import report

    report.held_out.cache_clear()
    _, _, expression, _ = within_group_fixture(tmp_path)
    out = tmp_path / "results/m/loso"
    truth = np.load(out / "chr.truth.npy")
    for group_start in (0, 40):
        for gene, people in enumerate(np.array_split(np.arange(group_start, group_start + 40), 3)):
            truth[gene, people] = np.nan
    np.save(out / "chr.truth.npy", truth)
    assert not np.isfinite(truth).all(axis=0).any()
    scores = report.per_gene_scores(tmp_path / "results", tmp_path, "m", "loso")
    assert np.isfinite(scores["r2"]).all()
    report.held_out.cache_clear()
    np.save(tmp_path / "expression.npy", expression[::-1])
    with pytest.raises(ValueError, match="not the dataset's expression"):
        report.per_gene_scores(tmp_path / "results", tmp_path, "m", "loso")


def test_a_covariate_only_score_scores_zero_and_a_foreign_dataset_is_refused(tmp_path):
    from benchmarks.bench_real import report

    report.held_out.cache_clear()
    _, covariates, expression, predictions = within_group_fixture(tmp_path)
    covariate_only = np.where(np.isfinite(predictions), (np.column_stack([np.ones(len(covariates)), covariates]) @ np.array([0.3, 1.0, -2.0, 0.5]))[None, :], np.nan)
    np.save(tmp_path / "results/m/loso/chr.snv.predictions.npy", covariate_only)
    scores = report.per_gene_scores(tmp_path / "results", tmp_path, "m", "loso")
    assert (scores["r2"] == 0.0).all() and (scores["oos_r2"] == 0.0).all()
    report.held_out.cache_clear()
    np.save(tmp_path / "expression.npy", expression[::-1])
    try:
        report.per_gene_scores(tmp_path / "results", tmp_path, "m", "loso")
    except ValueError as error:
        assert "not the dataset's expression" in str(error)
    else:
        raise AssertionError("a truth that is not the dataset's expression minus a covariate fit must stop the report")


def test_a_constant_fit_is_scored_as_no_prediction():
    from benchmarks.bench_real import report

    generator = np.random.default_rng(12)
    train_count, test_count = 60, 25
    covariates = generator.normal(size=(train_count + test_count, 4))
    genotypes = generator.binomial(2, 0.4, size=(train_count + test_count, 3)).astype(np.float64)
    train = harness.TrainData(gene_id="g", chrom="chr1", tss=0, genotypes=genotypes[:train_count], phenotype=np.zeros(train_count),
                              variants=synthetic_variants([False, True, False], ["panel"] * 3), superpopulation=np.array(["EUR"] * train_count),
                              population=np.array(["CEU"] * train_count), gene_start=0, gene_end=0, strand="+", exons=np.zeros((0, 2), dtype=np.int64),
                              coding_exons=np.zeros((0, 2), dtype=np.int64), covariates=covariates[:train_count])
    prediction, without_sv = harness.predict_for_truth(baselines.ZeroPredictor(0.37), train, genotypes[train_count:], covariates[train_count:])
    # Exactly zero: least squares of a constant on [1, C] leaves rounding noise, which would score as a random direction.
    assert (prediction == 0).all() and (without_sv == 0).all()
    residual, rank = report.within_group_residual(np.full((1, test_count), 0.37), covariates[train_count:])
    assert rank == 5 and (residual == 0).all()


def test_mismatched_partners_pair_each_gene_with_the_next_gene_on_another_chromosome():
    from benchmarks.bench_real import report

    assert list(report.mismatched_partners(np.array(["chr1", "chr1", "chr2", "chr3", "chr3"]))) == [2, 2, 3, 0, 0]
    assert list(report.mismatched_partners(np.array(["chr1", "chr1"]))) == [-1, -1]


def test_source_provenance_digests_the_sources_when_git_fails(monkeypatch):
    """The commit is resolved before the fits, and a checkout-less run records a source digest instead of raising."""
    import subprocess

    def raising(error):
        def run(*args, **kwargs):
            raise error

        return run

    digests = []
    for error in (FileNotFoundError("git"), subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])):
        monkeypatch.setattr(harness.subprocess, "run", raising(error))
        record = harness.source_provenance()
        assert record["harness_commit"] is None and record["harness_source"] == "no-git"
        digests.append(record["harness_source_sha256"])
    assert len(digests[0]) == 64 and digests[0] == digests[1]


def test_a_run_whose_git_fails_still_writes_every_fit(tmp_path, monkeypatch):
    import json

    tiny_dataset(tmp_path)
    method = f"{harness.__file__.rsplit('/', 1)[0]}/baselines.py:top_variant"

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(harness.subprocess, "run", no_git)
    harness.run(tmp_path, method, "top_variant", "loso", ["chr1"], tmp_path / "results", 1, ("snv",))
    out = tmp_path / "results" / "top_variant" / "loso"
    record = json.loads((out / "chr1.run.json").read_text())
    assert record["harness_commit"] is None and record["harness_source"] == "no-git" and len(record["harness_source_sha256"]) == 64
    assert record["genes"] == 1 and len(record["method_sha256"]) == 64
    assert np.isfinite(np.load(out / "chr1.snv.predictions.npy")).any()
    assert pd.read_csv(out / "chr1.log.tsv", sep="\t")["status"].tolist() == ["ok", "ok"]
