"""bench-sim per-chromosome replicates: the replicate scenarios, the people check and the across-chromosome summary."""

from __future__ import annotations

import json
import shutil

import numpy as np

from benchmarks.bench_sim import harness, per_chrom, truth

EPSILON = float(np.finfo(np.float64).eps)


def tiny_cohort(root, rng: np.random.Generator, n_var: int = 1500, size: int = 400) -> None:
    """Every file genetic_value, phenotype and the summary read, on one small chromosome."""
    root.mkdir(parents=True)
    cls = rng.choice(4, size=n_var, p=[0.6, 0.15, 0.15, 0.1]).astype(np.int8)
    frequency = rng.uniform(0.02, 0.5, size=n_var)
    np.savez(root / "variants.npz", cls=cls, len_change=np.where(cls >= 1, rng.integers(-500, 500, size=n_var), 0),
             superpop_af=np.tile(frequency.astype(np.float32), (5, 1)), donor_an=np.int64(3000))
    np.save(root / "truth_G.npy", rng.binomial(2, frequency[:, None], size=(n_var, size)).astype(np.uint8))
    np.savez(root / "annotations.npz", in_gene=rng.random(n_var) < 0.3, in_exon=rng.random(n_var) < 0.05,
             in_repeat=cls == 2, log_tss_distance=rng.uniform(0.0, 12.0, n_var), log_sv_length=np.where(cls >= 1, rng.uniform(1.0, 10.0, n_var), 0.0),
             ld_score=rng.uniform(1.0, 50.0, n_var), repeat_locus=np.where(cls == 2, rng.integers(0, 40, n_var), -1))
    is_test = np.zeros(size, dtype=bool)
    is_test[rng.choice(size, size // 5, replace=False)] = True
    np.savez(root / "samples.npz", group=rng.integers(0, 5, size), group_weights=np.full(5, 0.2), proportions=rng.dirichlet(np.ones(5), size),
             realized_proportions=rng.dirichlet(np.ones(5), size), sex=rng.integers(0, 2, size), age=rng.uniform(18, 80, size),
             batch=rng.integers(0, 2, size), is_test=is_test)
    np.savez(root / "pcs_truth.npz", pcs=rng.standard_normal((size, 3)))


def dev_params() -> dict:
    params = truth.draw_parameters(0)
    # Enough causal records on a 1,500-record chromosome.
    params.update(pi=0.05, binary=False)
    return params


def test_replicate_params_reseed_the_trait_off_the_reference_chromosome() -> None:
    params = dev_params()
    assert per_chrom.replicate_params(params, "chr22") == params
    replicates = {chrom: per_chrom.replicate_params(params, chrom) for chrom in ("chr19", "chr20")}
    for chrom, replicate in replicates.items():
        assert replicate == per_chrom.replicate_params(params, chrom)
        assert {key for key in params if replicate[key] != params[key]} == set(per_chrom.REPLICATE_SEEDS)
    assert all(replicates["chr19"][key] != replicates["chr20"][key] for key in per_chrom.REPLICATE_SEEDS)


def test_replicate_scenarios_draw_their_own_trait(tmp_path) -> None:
    cohorts = tmp_path / "cohort"
    tiny_cohort(cohorts / "chr22", np.random.default_rng(31))
    shutil.copytree(cohorts / "chr22", cohorts / "chr19")
    dev = tmp_path / "dev"
    reference = truth.write_scenario(dev_params(), cohorts / "chr22", dev / "scenario_000")
    scenarios = tmp_path / "scenarios"
    for chrom in ("chr22", "chr19"):
        per_chrom.build_scenarios(cohorts / chrom, dev, scenarios / chrom, ["000"])
    same, other = (np.load(scenarios / chrom / "scenario_000" / "truth.npz") for chrom in ("chr22", "chr19"))
    expected = np.load(dev / "scenario_000" / "truth.npz")
    # On the reference chromosome the replicate is the dev scenario itself.
    assert all(np.array_equal(same[name], expected[name]) for name in expected.files)
    assert per_chrom.scenario_path(scenarios, dev, "chr22", "000") == dev / "scenario_000"
    # Elsewhere, on identical genotypes, only the trait's draw differs; the drawn h2 still holds.
    record = json.loads((scenarios / "chr19" / "scenario_000" / "scenario.json").read_text())
    assert {key for key in record["params"] if record["params"][key] != reference["params"][key]} == set(per_chrom.REPLICATE_SEEDS)
    assert not np.array_equal(other["causal"], expected["causal"])
    assert abs(other["genetic_value"].var() - record["params"]["h2"]) <= 1e3 * EPSILON


def test_people_differences_name_the_fields_that_differ(tmp_path) -> None:
    tiny_cohort(tmp_path / "chr22", np.random.default_rng(32))
    shutil.copytree(tmp_path / "chr22", tmp_path / "chr21")
    samples = dict(np.load(tmp_path / "chr21" / "samples.npz"))
    samples["realized_proportions"] = samples["realized_proportions"][::-1]
    np.savez(tmp_path / "chr21" / "samples.npz", **samples)
    assert per_chrom.people_differences(tmp_path / "chr21", tmp_path / "chr22") == []
    samples["is_test"] = ~samples["is_test"]
    np.savez(tmp_path / "chr21" / "samples.npz", **samples)
    assert per_chrom.people_differences(tmp_path / "chr21", tmp_path / "chr22") == ["is_test"]


def test_bootstrap_levels_and_pairing() -> None:
    points = np.array([0.1, 0.2, 0.3, 0.4])
    # Pools with no person-level spread: the two-level interval is the chromosome-only one.
    flat = np.repeat(points[:, None], per_chrom.BOOTSTRAP_DRAWS, axis=1)
    interval = per_chrom.bootstrap_interval(points, flat, 7)
    assert interval["n"] == 4 and abs(interval["mean"] - 0.25) <= 4 * EPSILON
    assert np.allclose(interval["two_level"], interval["chromosomes"], rtol=0, atol=4 * EPSILON)
    assert interval["chromosomes"][0] < 0.25 < interval["chromosomes"][1]
    # Person-level spread widens the two-level interval, on the same chromosome draws.
    rng = np.random.default_rng(34)
    spread = flat + 0.05 * rng.standard_normal(flat.shape)
    wide = per_chrom.bootstrap_interval(points, spread, 7)
    assert wide["chromosomes"] == interval["chromosomes"]
    assert wide["two_level"][1] - wide["two_level"][0] > interval["two_level"][1] - interval["two_level"][0]
    # A method that is the reference plus a constant on every drawn person differs by exactly that constant.
    names = ("chr1", "chr2", "chr3", "chr4")
    reference = {chrom: {"r2": value, "pool": spread[index]} for index, (chrom, value) in enumerate(zip(names, points))}
    shifted = {chrom: {"r2": entry["r2"] + 0.05, "pool": entry["pool"] + 0.05} for chrom, entry in reference.items()}
    summary = per_chrom.summarize_methods({"ref": reference, "shifted": shifted}, "ref", 7)
    for level in ("two_level", "chromosomes"):
        assert np.allclose(summary["shifted"]["difference"][level], 0.05, rtol=0, atol=16 * EPSILON)
        assert summary["ref"]["difference"][level] == [0.0, 0.0]
    # A method with no fits yet has an empty summary.
    empty = per_chrom.summarize_methods({"ref": reference, "none": {}}, "ref", 7)["none"]
    assert empty["mean"]["n"] == 0 and empty["difference"]["n"] == 0


def test_chromosome_accuracy_is_the_harness_r2_on_shared_people() -> None:
    rng = np.random.default_rng(35)
    size = 500
    covariates = rng.standard_normal((size, 3))
    value = rng.standard_normal(size) + covariates @ np.array([0.3, -0.2, 0.1])
    prediction = value + rng.standard_normal(size)
    accuracy = per_chrom.chromosome_accuracy(value, {"a": prediction, "b": prediction.copy()}, covariates, 3)
    assert abs(accuracy["a"]["r2"] - harness.genetic_accuracy(value, prediction, covariates)[0]) <= 1e6 * EPSILON
    assert accuracy["a"]["low"] <= accuracy["a"]["r2"] <= accuracy["a"]["high"]
    # Every method is scored on the same resampled people.
    assert np.array_equal(accuracy["a"]["pool"], accuracy["b"]["pool"])


def test_summary_pairs_each_method_with_the_reference_on_shared_chromosomes(tmp_path) -> None:
    rng = np.random.default_rng(33)
    cohorts, dev, scenarios, results = tmp_path / "cohort", tmp_path / "dev", tmp_path / "scenarios", tmp_path / "results"
    chromosomes = ("chr20", "chr21", "chr22")
    for chrom in chromosomes:
        tiny_cohort(cohorts / chrom, rng)
    truth.write_scenario(dev_params(), cohorts / "chr22", dev / "scenario_000")
    for chrom in ("chr20", "chr21"):
        per_chrom.build_scenarios(cohorts / chrom, dev, scenarios / chrom, ["000"])
    # The other method has no chr21 fit: its difference uses chr20 and chr22 only.
    for method, noise, fitted in (("ref", 0.5, chromosomes), ("other", 2.0, ("chr20", "chr22"))):
        for chrom in fitted:
            value = np.load(per_chrom.scenario_path(scenarios, dev, chrom, "000") / "truth.npz")["genetic_value"]
            test = np.load(cohorts / chrom / "samples.npz")["is_test"]
            target = results / method / chrom / "scenario_000"
            target.mkdir(parents=True)
            np.savez(target / "prediction.npz", total=value[test] + noise * value.std() * rng.standard_normal(int(test.sum())))
    summary = per_chrom.summarize(cohorts, scenarios, dev, ["000"], [("ref", results / "ref"), ("other", results / "other")])
    methods = summary["scenarios"]["000"]["methods"]
    assert summary["reference"] == "ref" and set(summary["scenarios"]["000"]["truth"]) == set(chromosomes)
    for method in ("ref", "other"):
        for chrom, entry in methods[method]["chromosomes"].items():
            test = np.load(cohorts / chrom / "samples.npz")["is_test"]
            covariates = harness.covariate_matrix(cohorts / chrom, "truth")[0][test]
            value = np.load(per_chrom.scenario_path(scenarios, dev, chrom, "000") / "truth.npz")["genetic_value"][test]
            prediction = np.load(results / method / chrom / "scenario_000" / "prediction.npz")["total"]
            assert abs(entry["r2"] - harness.genetic_accuracy(value, prediction, covariates)[0]) <= 1e6 * EPSILON
            assert entry["low"] <= entry["r2"] <= entry["high"]
    r2 = {method: {chrom: entry["r2"] for chrom, entry in methods[method]["chromosomes"].items()} for method in methods}
    assert abs(methods["other"]["mean"]["mean"] - np.mean(list(r2["other"].values()))) <= 4 * EPSILON and methods["other"]["mean"]["n"] == 2
    assert methods["other"]["difference"]["n"] == 2 and abs(
        methods["other"]["difference"]["mean"] - np.mean([r2["other"][chrom] - r2["ref"][chrom] for chrom in ("chr20", "chr22")])) <= 4 * EPSILON
    assert methods["ref"]["difference"]["two_level"] == [0.0, 0.0] and methods["ref"]["difference"]["n"] == 3
    for method in methods:
        low, high = methods[method]["mean"]["two_level"]
        assert low <= methods[method]["mean"]["mean"] <= high
