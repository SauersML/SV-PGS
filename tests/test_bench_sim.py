"""bench-sim: the benchmark's data-free building blocks.

Statistical checks use fixed seeds; their bounds are Monte Carlo standard errors times the normal quantile of
FALSE_FAILURE_PROBABILITY, so a correct implementation fails with at most that probability over seeds.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys

import numpy as np
from scipy.stats import norm

from benchmarks.bench_sim import baselines, cohort, harness, measurement, measurement_beagle, measurement_readcn, measurement_truthhalf, records, truth, truth_out
from benchmarks.bench_sim.annotations import merged_intervals, overlaps
from sv_pgs.dosage_store import encode_dosage_milli

FALSE_FAILURE_PROBABILITY = 1e-9
Z_BOUND = float(norm.isf(FALSE_FAILURE_PROBABILITY / 2))
EPSILON = float(np.finfo(np.float64).eps)


def test_dosage_codes_match_the_store_encoder() -> None:
    milli = np.arange(2001)
    assert np.array_equal(measurement.encode_milli(milli), encode_dosage_milli(milli))


def test_group_weights_are_the_founder_composition() -> None:
    founders = [(f"s{index}", superpop) for index, superpop in enumerate(["AFR"] * 7 + ["AMR"] * 3 + ["EAS"] * 5 + ["EUR"] * 4 + ["SAS"] * 1)]
    weights = cohort.group_weights(founders)
    expected = {"EUR": 4, "AFR_admixed": 7, "AMR_admixed": 3, "EAS": 5, "SAS": 1}
    assert np.allclose(weights, [expected[name] / 20 for name in cohort.GROUPS], rtol=0, atol=len(cohort.GROUPS) * EPSILON)


def test_cohort_proportions_average_to_the_group_means() -> None:
    rng = np.random.default_rng(1)
    size = 200_000
    weights = np.full(len(cohort.GROUPS), 1.0 / len(cohort.GROUPS))
    group, proportions, generations = cohort.draw_cohort(size, rng, weights)
    for index, (name, (_, mean, admixture)) in enumerate(cohort.GROUPS.items()):
        members = proportions[group == index]
        assert np.allclose(members.sum(axis=1), 1.0, atol=len(cohort.SUPERPOPS) * EPSILON)
        standard_error = members.std(axis=0) / np.sqrt(members.shape[0])
        assert np.all(np.abs(members.mean(axis=0) - np.asarray(mean)) <= Z_BOUND * standard_error + EPSILON), name
        assert np.all(generations[group == index] == admixture)


def test_haplotype_paths_copy_only_from_the_tract_ancestry() -> None:
    rng = np.random.default_rng(2)
    cm = np.linspace(0.0, 70.0, 20_000)
    donor_columns = {index: np.arange(100 * index, 100 * index + 100) for index in range(len(cohort.SUPERPOPS))}
    for proportions, generations in (((1.0, 0, 0, 0, 0), 0.0), ((0.5, 0, 0, 0.5, 0), 7.0)):
        starts, donors, ancestries = cohort.haplotype_path(cm, np.asarray(proportions), generations, donor_columns, rng)
        assert starts[0] == 0 and np.all(np.diff(starts) >= 0)
        assert set(np.unique(ancestries)) <= set(np.flatnonzero(np.asarray(proportions) > 0))
        for donor, ancestry in zip(donors, ancestries):
            assert donor in donor_columns[int(ancestry)]


def test_interval_overlap_handles_nested_intervals() -> None:
    starts = np.array([10, 12, 40])
    ends = np.array([30, 14, 50])
    query_start = np.array([0, 15, 29, 30, 45, 55])
    query_end = np.array([10, 16, 31, 40, 46, 60])
    expected = np.array([False, True, True, False, True, False])
    assert np.array_equal(cohort.overlaps_interval(starts, ends, query_start, query_end), expected)
    assert np.array_equal(overlaps(starts, ends, query_start, query_end), expected)


def test_repeat_intervals_merge_overlaps(tmp_path) -> None:
    path = tmp_path / "repeats.txt.gz"
    with gzip.open(path, "wt") as handle:
        for start, end in ((100, 200), (150, 250), (300, 310), (10, 20)):
            handle.write(f"0\tchr22\t{start}\t{end}\tx\n")
        handle.write("0\tchr21\t5\t9\tx\n")
    merged_start, merged_end = merged_intervals(path, "chr22")
    assert merged_start.tolist() == [10, 100, 300]
    assert merged_end.tolist() == [20, 250, 310]


def test_hom_ref_calls_follow_the_gvcf_block_rule() -> None:
    rng = np.random.default_rng(3)
    genotype = np.zeros(200_000, dtype=np.intp)
    pl = measurement.simulated_pl(genotype, measurement.BASE_ERROR[0], rng)
    called_hom_ref = pl[:, 0] == 0
    assert called_hom_ref.any()
    block = pl[called_hom_ref]
    assert block[:, 1].max() <= measurement.GQ_CAP // 5
    assert block[:, 2].max() <= measurement.HOMREF_HOM_ALT_CAP
    assert np.array_equal(block[:, 2], np.minimum(2 * block[:, 1], measurement.HOMREF_HOM_ALT_CAP))
    assert np.all(pl.min(axis=1) == 0)


def test_variant_calls_keep_their_likelihoods() -> None:
    rng = np.random.default_rng(4)
    genotype = np.ones(50_000, dtype=np.intp)
    pl = measurement.simulated_pl(genotype, measurement.BASE_ERROR[0], rng)
    assert (pl[:, 1] == 0).any()
    assert np.all(pl <= measurement.PL_CAP)


def test_pl_text_round_trips() -> None:
    rng = np.random.default_rng(5)
    pl = rng.integers(0, measurement.PL_CAP + 1, size=(1_000, 3))
    text = measurement.pl_text(pl).tobytes().decode()
    fields = text.rstrip("\t").split("\t")
    parsed = np.array([[int(value) for value in field.split(",")] for field in fields])
    assert np.array_equal(parsed, pl)


def test_shape_families_have_unit_variance() -> None:
    rng = np.random.default_rng(6)
    count = 400_000
    cases = {
        "gaussian": {}, "laplace": {}, "two_point": {}, "sparse_mixture": {},
        "variance_grid": {"grid_weights": [0.5, 0.3, 0.2]}, "lognormal_scale": {"tau": 0.5},
        "student_t": {"nu": 8.0},
    }
    for shape, params in cases.items():
        draws = truth.unit_variance_draws(rng, shape, count, params)
        centred = draws - draws.mean()
        variance = float(np.mean(centred**2))
        fourth = float(np.mean(centred**4))
        standard_error = np.sqrt(max(fourth - variance**2, 0.0) / count)
        assert abs(variance - 1.0) <= Z_BOUND * standard_error + EPSILON, shape


def test_scenario_parameters_are_deterministic_and_in_range() -> None:
    for seed in range(200):
        params = truth.draw_parameters(seed)
        assert params == truth.draw_parameters(seed)
        assert 0.01 <= params["h2"] <= 0.2
        assert 1e-4 <= params["pi"] <= 3e-2
        assert -1.0 <= params["alpha_snv"] <= 0.0
        assert abs(params["alpha_sv"] - params["alpha_snv"]) <= 0.5
        assert params["shape"] in truth.SHAPES and params["sv_shape"] in truth.SHAPES
        assert params["sv_mode"] in truth.MODES and params["annotation_mode"] in truth.MODES
        assert 0.01 <= params["prevalence"] <= 0.3


def test_sealed_seeds_are_deterministic_and_distinct() -> None:
    master = "ab" * 32
    seeds = truth.sealed_seeds(master, 64)
    assert seeds == truth.sealed_seeds(master, 64)
    assert len(set(seeds)) == 64
    assert truth.sealed_seeds("cd" * 32, 64) != seeds


def test_random_spline_is_centred() -> None:
    rng = np.random.default_rng(7)
    values = rng.gamma(2.0, 3.0, size=50_000)
    curve = truth.random_spline(rng, values, 0.4)
    assert abs(curve.mean()) <= values.size * EPSILON * np.abs(curve).max()
    assert curve.std() > 0


def test_incremental_r2_of_an_exact_prediction() -> None:
    rng = np.random.default_rng(8)
    covariates = rng.standard_normal((5_000, 3))
    covariate_part = covariates @ np.array([0.5, -0.2, 0.1])
    prediction = rng.standard_normal(5_000)
    design = np.column_stack([np.ones(5_000), covariates])
    prediction -= design @ np.linalg.lstsq(design, prediction, rcond=None)[0]
    outcome = covariate_part + prediction
    gain, slope = harness.incremental_r2(outcome, prediction, covariates)
    base = covariate_part - covariate_part.mean()
    total = outcome - outcome.mean()
    expected = 1.0 - (base @ base) / (total @ total)
    # Backward-stable least squares: error of order cond(design) * eps, accumulated over n terms.
    tolerance = np.linalg.cond(np.column_stack([design, prediction])) * EPSILON * outcome.size
    assert abs(gain - expected) <= tolerance
    assert abs(slope - 1.0) <= tolerance


def test_auc_matches_pairwise_counting() -> None:
    rng = np.random.default_rng(9)
    outcome = (rng.random(400) < 0.3).astype(float)
    score = rng.standard_normal(400) + outcome
    positives, negatives = score[outcome > 0.5], score[outcome < 0.5]
    expected = float((positives[:, None] > negatives[None, :]).mean())
    assert abs(harness.auc(outcome, score) - expected) <= positives.size * negatives.size * EPSILON


def test_beagle_alleles_are_unique_per_record() -> None:
    assert measurement_beagle.beagle_allele("<INS>", 7) == "<INS:v7>"
    assert measurement_beagle.beagle_allele("<INS>", 7) != measurement_beagle.beagle_allele("<INS>", 8)
    assert measurement_beagle.beagle_allele("ACGT", 7) == "ACGT"


def test_phased_calls_keep_true_phase_when_the_call_is_right() -> None:
    rng = np.random.default_rng(10)
    first = rng.integers(0, 2, size=10_000)
    second = rng.integers(0, 2, size=10_000)
    genotype = first + second
    index = measurement_beagle.phased_call_index(genotype.astype(np.uint8), genotype, first, rng)
    assert np.array_equal(index // 2, first) and np.array_equal(index % 2, second)
    miscalled = np.where(genotype == 1, 2, 1).astype(np.uint8)
    index = measurement_beagle.phased_call_index(miscalled, genotype, first, rng)
    assert np.array_equal(index // 2 + index % 2, miscalled)


def test_batch_block_and_realized_r2_match_direct_computation() -> None:
    rng = np.random.default_rng(11)
    truth = rng.integers(0, 3, size=(50_000, 40)).astype(np.uint8)
    rows = np.sort(rng.choice(truth.shape[0], size=700, replace=False))
    assert np.array_equal(measurement.batch_block(truth, rows, 5, 30), truth[rows, 5:30])
    noise = rng.integers(0, 40, size=truth.shape)
    observed = np.clip(truth.astype(np.int64) * measurement.CODES_PER_DOSAGE + noise, 0, 254).astype(np.uint8)
    r2 = measurement.realized_r2(truth, observed)
    for row in rows[:20]:
        if truth[row].std() > 0:
            expected = np.corrcoef(truth[row], observed[row])[0, 1] ** 2
            # r2 = cross^2 / (ss_truth * ss_observed): four length-n sums, each with relative error <= n * eps.
            assert abs(r2[row] - expected) <= 4 * truth.shape[1] * EPSILON * max(expected, 1.0)


def test_ridge_inf_baseline_matches_a_direct_solve(tmp_path) -> None:
    """Build a tiny cohort's kernel files the way gpu_kernels does, run the baseline, and check its prediction
    against (K + lambda I)^-1 y solved directly, with lambda from the Haseman-Elston h2 computed directly."""
    rng = np.random.default_rng(12)
    size, n_var = 300, 400
    is_test = np.zeros(size, dtype=bool)
    is_test[rng.choice(size, size=60, replace=False)] = True
    train, test = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    cls = rng.choice(4, size=n_var, p=[0.6, 0.2, 0.15, 0.05]).astype(np.int8)
    genotype = rng.binomial(2, rng.uniform(0.05, 0.5, size=(n_var, 1)), size=(n_var, size)).astype(np.uint8)
    observed = genotype * np.uint8(measurement.CODES_PER_DOSAGE)
    np.save(tmp_path / "observed_beagle.npy", observed)
    np.savez(tmp_path / "samples.npz", is_test=is_test, sex=rng.integers(0, 2, size), age=rng.uniform(18, 80, size),
             batch=rng.integers(0, 2, size))
    np.savez(tmp_path / "variants.npz", cls=cls)
    np.save(tmp_path / "measured.npy", np.ones(n_var, dtype=bool))
    dosage = observed.astype(np.float64) / measurement.CODES_PER_DOSAGE
    standardized = (dosage - dosage[:, train].mean(axis=1, keepdims=True)) / dosage[:, train].std(axis=1, keepdims=True)
    counts = {}
    kernels = {}
    for name, members in (("simple", cls <= 1), ("structural", cls >= 2)):
        kernels[name] = standardized[members].T @ standardized[members] / members.sum()
        counts[name] = int(members.sum())
        np.save(tmp_path / f"kernel_{name}_beagle.npy", kernels[name].astype(np.float32))
    (tmp_path / "kernel_counts_beagle.json").write_text(json.dumps(counts))
    weight = counts["structural"] / n_var
    combined = (1 - weight) * kernels["simple"] + weight * kernels["structural"]
    np.savez(tmp_path / "pcs_beagle.npz", pcs=rng.standard_normal((size, 10)))
    scenario = tmp_path / "scenario_000"
    scenario.mkdir()
    effects = rng.standard_normal(n_var) * 0.05
    phenotype = effects @ standardized + rng.standard_normal(size)
    np.savez(scenario / "truth.npz", phenotype=phenotype, causal=np.arange(n_var), per_allele=effects)
    shared = baselines.load_shared(tmp_path, "beagle")
    baselines.baselines(shared, scenario, tmp_path / "results", "beagle")

    residual, scale = baselines.residualized(phenotype[train], shared["covariates"][train])
    design = np.column_stack([np.ones(train.size), shared["covariates"][train]])
    projector = np.eye(train.size) - design @ np.linalg.solve(design.T @ design, design.T)
    for name, matrix in (("simple", kernels["simple"]), ("all", combined)):
        train_kernel = projector @ matrix[np.ix_(train, train)] @ projector
        off_diagonal = ~np.eye(train.size, dtype=bool)
        cross_products = np.outer(residual, residual)[off_diagonal]
        heritability = float(np.clip((train_kernel[off_diagonal] @ cross_products) / (train_kernel[off_diagonal] @ train_kernel[off_diagonal]), 0.0, 1.0))
        meta = json.loads((tmp_path / "results" / f"ridge_inf_{name}" / "scenario_000" / "meta.json").read_text())
        # Kernels round-trip through float32 files: each entry carries relative error eps32 / 2, and every
        # quantity here is a sum of at most n^2 such entries, so its relative error is at most n * eps32.
        float32_tolerance = float(np.finfo(np.float32).eps) * train.size
        assert abs(meta["h2_used"] - heritability) <= float32_tolerance
        prediction = np.load(tmp_path / "results" / f"ridge_inf_{name}" / "scenario_000" / "prediction.npz")["total"]
        if heritability == 0.0:
            assert np.all(prediction == 0.0)
            continue
        ridge = (1 - heritability) / heritability
        system = train_kernel + ridge * np.eye(train.size)
        expected = scale * matrix[np.ix_(test, train)] @ projector @ np.linalg.solve(system, residual)
        # A relative perturbation delta of the kernel moves the solve by at most cond(system) * delta.
        assert np.max(np.abs(prediction - expected)) <= float32_tolerance * np.linalg.cond(system) * np.max(np.abs(expected))


def test_measured_records_follow_the_panel_allele_count_rule(tmp_path) -> None:
    panel = np.zeros((6, 10), dtype=np.uint8)
    panel[1, :1] = 1
    panel[2, :2] = 1
    panel[3, :9] = 1
    panel[4, :8] = 1
    panel[5, :5] = 1
    np.save(tmp_path / "panel_haps.npy", panel)
    measured = records.measured_records(tmp_path)
    assert measured.tolist() == [False, False, True, False, True, True]
    assert np.array_equal(np.load(tmp_path / "measured.npy"), measured)


def test_projection_matches_the_explicit_projector() -> None:
    rng = np.random.default_rng(13)
    size, test_size = 700, 90
    factor = rng.standard_normal((size, 40))
    kernel = factor @ factor.T / 40
    cross = rng.standard_normal((test_size, size))
    design = np.column_stack([np.ones(size), rng.standard_normal((size, 4))])
    projector = np.eye(size) - design @ np.linalg.solve(design.T @ design, design.T)
    expected_kernel, expected_cross = projector @ kernel @ projector, cross @ projector
    kernel_scale, cross_scale = np.max(np.abs(kernel)), np.max(np.abs(cross))
    baselines.project_out_covariates(kernel, cross, design)
    # P K P has four terms (K, QQ'K, KQQ', QQ'KQQ'), each an entry-wise sum of at most size^2 products of
    # orthonormal-basis entries (|q| <= 1) with kernel entries: rounding at most size * eps * max|K| per term,
    # in both this computation and the explicit one.
    assert np.max(np.abs(kernel - expected_kernel)) <= 2 * 4 * size * EPSILON * kernel_scale * size
    assert np.max(np.abs(cross - expected_cross)) <= 2 * 2 * size * EPSILON * cross_scale * size


def test_truth_half_takes_a_fixed_share_of_each_groups_training_samples(tmp_path, monkeypatch) -> None:
    rng = np.random.default_rng(14)
    size, n_var = 1_000, 30
    group = rng.integers(0, 5, size=size)
    is_test = rng.random(size) < 0.2
    np.savez(tmp_path / "samples.npz", group=group, is_test=is_test)
    flags = measurement_truthhalf.truth_half(np.load(tmp_path / "samples.npz"))
    assert not (flags & is_test).any()
    for index in range(5):
        members = (group == index) & ~is_test
        assert flags[members].sum() == int(round(measurement_truthhalf.TRUTH_FRACTION * members.sum()))
    truth_genotype = rng.integers(0, 3, size=(n_var, size)).astype(np.uint8)
    observed = rng.integers(0, 255, size=(n_var, size)).astype(np.uint8)
    measured = rng.random(n_var) < 0.8
    np.save(tmp_path / "truth_G.npy", truth_genotype)
    np.save(tmp_path / "observed_beagle.npy", observed)
    np.save(tmp_path / "measured.npy", measured)
    monkeypatch.setattr(sys, "argv", ["measurement_truthhalf", "--dir", str(tmp_path)])
    measurement_truthhalf.main()
    arm = np.load(tmp_path / "observed_beagle_truth.npy")
    saved_flags = np.load(tmp_path / "truth_half.npy")
    assert np.array_equal(saved_flags, flags)
    expected = observed.copy()
    expected[np.ix_(measured, flags)] = truth_genotype[np.ix_(measured, flags)] * measurement.CODES_PER_DOSAGE
    assert np.array_equal(arm, expected)


def test_harness_imports_without_cyvcf2() -> None:
    """A submission's environment needs only numpy and scipy: the harness must not pull in the measurement arms."""
    script = ("import sys, builtins; real = builtins.__import__\n"
              "def guard(name, *args, **kwargs):\n"
              "    if name.split('.')[0] == 'cyvcf2': raise ImportError('cyvcf2 blocked')\n"
              "    return real(name, *args, **kwargs)\n"
              "builtins.__import__ = guard\n"
              "import benchmarks.bench_sim.harness, benchmarks.bench_sim.compare\n")
    subprocess.run([sys.executable, "-c", script], check=True)


def out_of_family_cohort(tmp_path):
    """A tiny chromosome: 400 records (SNV, INDEL, TR, SV) over 3 cM, 600 samples, one gene with two exons, one CpG island."""
    rng = np.random.default_rng(15)
    n_var, size = 400, 600
    pos = np.sort(rng.choice(np.arange(1_000, 3_000_000), size=n_var, replace=False))
    cls = rng.choice(4, size=n_var, p=[0.6, 0.15, 0.15, 0.1]).astype(np.int8)
    svtype = np.where(cls == 3, rng.choice(["DEL", "DUP", "INS"], size=n_var), "")
    end = np.where(cls == 3, pos + rng.integers(100, 50_000, size=n_var), pos)
    # Two SVs in the gene's exons: a deletion across part of the first exon, an insertion inside the second.
    for target, kind, span in ((520_000, "DEL", 40_000), (850_000, "INS", 0)):
        row = int(np.argmin(np.abs(pos - target)))
        pos[row], cls[row], svtype[row], end[row] = target, 3, kind, target + span
    np.savez(tmp_path / "variants.npz", pos=pos, end=end, ref_len=np.ones(n_var, dtype=np.int64), cls=cls, cm=np.linspace(0.0, 3.0, n_var),
             svtype=svtype, len_change=np.where(cls == 3, -(end - pos), 0))
    frequency = rng.uniform(0.01, 0.5, size=n_var)
    genotype = rng.binomial(2, frequency[:, None], size=(n_var, size)).astype(np.uint8)
    np.save(tmp_path / "truth_G.npy", genotype)
    np.savez(tmp_path / "annotations.npz", ld_subsample_af=genotype.mean(axis=1) / 2.0)
    np.save(tmp_path / "measured.npy", rng.random(n_var) < 0.8)
    np.savez(tmp_path / "samples.npz", age=rng.uniform(18, 80, size), sex=rng.integers(0, 2, size), batch=rng.integers(0, 2, size),
             realized_proportions=rng.dirichlet(np.ones(5), size=size))
    with gzip.open(tmp_path / "refseq.txt.gz", "wt") as handle:
        handle.write("0\tNM_1\tchr22\t+\t500000\t900000\t500000\t900000\t2\t500000,800000,\t600000,900000,\t0\tGENE1\n")
    with gzip.open(tmp_path / "cpg.txt.gz", "wt") as handle:
        handle.write("0\tchr22\t1000000\t1200000\tCpG: 1\n")
    return genotype, cls


def test_out_of_family_copy_change_rule() -> None:
    gene = {"start": 100, "end": 400, "exons": [[100, 200], [300, 400]], "exonic_bases": 200}
    assert truth_out.copy_change("DEL", 150, 250, gene) == -0.25
    assert truth_out.copy_change("DUP", 50, 450, gene) == 1.0
    assert truth_out.copy_change("DUP", 150, 250, gene) == -0.25
    assert truth_out.copy_change("INS", 310, 312, gene) == -1.0
    assert truth_out.copy_change("INS", 220, 222, gene) == 0.0


def test_pareto_magnitudes_have_a_floor_and_the_drawn_tail() -> None:
    rng = np.random.default_rng(16)
    count, tail = 400_000, 1.8
    magnitude = truth_out.pareto_magnitudes(rng, count, tail)
    assert magnitude.min() >= 1.0
    # P(|beta| > 2) = 2^-tail; binomial standard error.
    probability = 2.0 ** -tail
    assert abs((magnitude > 2.0).mean() - probability) <= Z_BOUND * np.sqrt(probability * (1 - probability) / count)


def test_out_of_family_truths(tmp_path) -> None:
    genotype, cls = out_of_family_cohort(tmp_path)
    variants = np.load(tmp_path / "variants.npz")
    for seed, family in enumerate(truth_out.FAMILIES):
        out = tmp_path / f"scenario_{family}"
        truth_out.build(seed, family, tmp_path, out, tmp_path / "refseq.txt.gz", tmp_path / "cpg.txt.gz", "chr22")
        record = json.loads((out / "scenario.json").read_text())
        stored = np.load(out / "truth.npz")
        assert record["params"]["family"] == family and "shape" not in record["params"]
        value = stored["genetic_value"]
        # The genetic value is scaled to the drawn h2 exactly (population variance), up to rounding.
        assert abs(value.var() - record["params"]["h2"]) <= 1e3 * EPSILON
        rows, effects = stored["causal"], stored["per_allele"]
        centred = genotype[rows].astype(np.float64) - genotype[rows].mean(axis=1, keepdims=True)
        additive = effects @ centred
        if family != "epistasis":
            assert np.max(np.abs(additive - value)) <= 1e3 * EPSILON * np.max(np.abs(value))
        if family == "fixed_count":
            assert rows.size == record["params"]["causal_count"]
            per_sd = np.abs(effects) * genotype[rows].std(axis=1)
            assert np.allclose(per_sd, per_sd[0], rtol=1e3 * EPSILON)
        if family == "hidden_annotation":
            span_start, span_end = truth_out.record_spans(variants)
            inside = (span_end[rows] > 1_000_000 - truth_out.SHORE_BASES) & (span_start[rows] < 1_200_000 + truth_out.SHORE_BASES)
            assert inside.all()
        if family == "clustered_loci":
            locus = np.floor(variants["cm"][rows] / record["params"]["locus_cm"]).astype(int)
            for locus_id in np.unique(locus):
                assert np.unique(effects[locus == locus_id]).size == 1
        if family == "sv_gene_dosage":
            changes = truth_out.sv_gene_changes(variants, truth_out.gene_models(tmp_path / "refseq.txt.gz", "chr22"))
            sv_rows = np.array(sorted(row for row, _ in changes))
            ratio = effects[np.searchsorted(rows, sv_rows)] / np.array([changes[row, 0] for row in sv_rows])
            assert np.allclose(ratio, ratio[0], rtol=1e3 * EPSILON)  # one gene: every SV effect is its copy change x one gene effect
        if family == "panel_absent":
            measured = np.load(tmp_path / "measured.npy")
            assert (~measured[rows]).sum() >= 1
        again = tmp_path / f"again_{family}"
        truth_out.build(seed, family, tmp_path, again, tmp_path / "refseq.txt.gz", tmp_path / "cpg.txt.gz", "chr22")
        assert np.array_equal(np.load(again / "truth.npz")["phenotype"], stored["phenotype"])


def test_out_of_family_sealed_seeds_are_disjoint_from_the_in_family_ones() -> None:
    master = "ef" * 32
    assert not set(truth_out.sealed_out_seeds(master, 64)) & set(truth.sealed_seeds(master, 64))


def test_read_depth_likelihood_matches_the_negative_binomial() -> None:
    from scipy.stats import nbinom
    reads = np.array([0.0, 3.0, 17.0, 40.0])
    mean, size = np.array([2.5, 3.0, 20.0, 35.0]), 7.0
    expected = nbinom.logpmf(reads, size, size / (size + mean))
    got = measurement_readcn.negative_binomial_log_likelihood(reads, mean, size)
    assert np.max(np.abs(got - expected)) <= 1e3 * EPSILON * np.max(np.abs(expected))


def test_read_depth_calls_resolve_unique_cnvs_and_blur_paralogous_ones() -> None:
    rng = np.random.default_rng(17)
    samples = 4_000
    genotype = rng.integers(0, 3, size=(2, samples)).astype(np.float64)
    # 50 kb at 30x is about 5,000 reads per copy. With phi = 1e-3 and an exact depth scale (tau = 0) the relative
    # read-count sd is about sqrt(phi + 1/mean) = 0.03, while adjacent copy numbers differ by at least 25%: calls are
    # exact with overwhelming probability.
    params = {"sigma_s": 0.1, "phi": 1e-3, "identity": 0.98, "tau": 0.0}
    long_unique = measurement_readcn.simulate(genotype, np.array([True, False]), np.array([50_000.0, 50_000.0]),
                                              np.array([0, 0]), params, rng)
    assert long_unique.shape == (2, samples, 3) and np.all(long_unique.min(axis=-1) == 0)
    called = np.argmin(long_unique, axis=-1)
    assert np.array_equal(called, genotype)
    paralogous = measurement_readcn.simulate(genotype, np.array([True, False]), np.array([50_000.0, 50_000.0]),
                                             np.array([6, 6]), params, rng)
    assert (np.argmin(paralogous, axis=-1) == genotype).mean() < (called == genotype).mean()


def test_genetic_accuracy_is_the_partial_correlation_and_the_genetic_scale_slope():
    from benchmarks.bench_sim.harness import genetic_accuracy

    rng = np.random.default_rng(3)
    covariates = rng.standard_normal((4000, 3))
    genetic = rng.standard_normal(4000) + covariates @ np.array([0.5, -0.2, 0.1])
    prediction = 0.5 * genetic + rng.standard_normal(4000)
    r2, slope = genetic_accuracy(genetic, prediction, covariates)
    base = np.column_stack([np.ones(4000), covariates])
    project = lambda values: values - base @ np.linalg.lstsq(base, values, rcond=None)[0]
    g, p = project(genetic), project(prediction)
    assert np.isclose(r2, np.corrcoef(g, p)[0, 1] ** 2)
    assert np.isclose(slope, (g @ p) / (p @ p))
    # A prediction shrunk by any factor keeps its r2 and scales its slope by the inverse.
    shrunk_r2, shrunk_slope = genetic_accuracy(genetic, 0.1 * prediction, covariates)
    assert np.isclose(shrunk_r2, r2) and np.isclose(shrunk_slope, 10.0 * slope)


def test_drop_classes_withholds_exactly_the_named_classes() -> None:
    table = {"cls": np.array([0, 3, 1, 2, 3, 0]), "pos": np.arange(6) * 10, "class_names": np.array(["SNV", "INDEL", "TR", "SV"])}
    records = np.array([2, 5, 7, 11, 13, 17])
    kept, view = harness.drop_classes(records, table, ("TR", "SV"))
    assert kept.tolist() == [2, 7, 17]
    assert view["cls"].tolist() == [0, 1, 0] and view["pos"].tolist() == [0, 20, 50]
    assert view["class_names"].tolist() == table["class_names"].tolist()
    kept, _ = harness.drop_classes(records, table, ("SV",))
    assert kept.tolist() == [2, 7, 11, 17]
    kept, view = harness.drop_classes(records, table, ())
    assert kept.tolist() == records.tolist() and view["pos"].tolist() == table["pos"].tolist()
