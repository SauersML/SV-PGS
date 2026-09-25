"""The bench-sim SV-PGS submission (full-data route) and the truth measurement arm, on a small synthetic cohort."""

from pathlib import Path

import numpy as np

from benchmarks.bench_sim import harness, measurement_truth

SUBMISSION = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_sim" / "submissions" / "svpgs_full.py"


def _cohort(rng: np.random.Generator, n_var: int, n_samples: int):
    frequency = rng.uniform(0.05, 0.5, n_var)
    genotype = (rng.uniform(size=(n_var, n_samples)) < frequency[:, None]).astype(np.uint8) + (rng.uniform(size=(n_var, n_samples)) < frequency[:, None]).astype(np.uint8)
    observed = genotype * np.uint8(127)
    cls = rng.choice(4, size=n_var, p=(0.7, 0.1, 0.1, 0.1)).astype(np.int64)
    len_change = np.where(cls == 0, 0, rng.choice([-50, 0, 50], size=n_var))
    variants = {
        "pos": np.sort(rng.choice(10**6, size=n_var, replace=False)).astype(np.int64),
        "cm": np.linspace(0.0, 1.0, n_var),
        "cls": cls,
        "len_change": len_change,
        "ref_len": np.ones(n_var, dtype=np.int64),
        "alt_len": np.ones(n_var, dtype=np.int64),
        "imputation_info": np.where(cls >= 2, 0.8, np.nan),
        "in_gene": (rng.random(n_var) < 0.3).astype(np.float64),
        "in_exon": (rng.random(n_var) < 0.1).astype(np.float64),
        "in_repeat": (rng.random(n_var) < 0.2).astype(np.float64),
        "repeat_locus": np.where(cls == 2, np.arange(n_var) // 8, -1),
        "log_tss_distance": rng.normal(9.0, 2.0, n_var),
        "log_sv_length": np.where(cls == 3, rng.normal(5.0, 1.0, n_var), 0.0),
        "class_names": np.array(["SNV", "INDEL", "TR", "SV"]),
    }
    return observed, variants


def test_the_submission_fits_and_scores_a_small_cohort(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    rng = np.random.default_rng(3)
    n_var, n_train, n_test = 60, 240, 80
    observed, variants = _cohort(rng, n_var, n_train + n_test)
    effects = np.zeros(n_var)
    effects[rng.choice(n_var, 6, replace=False)] = rng.normal(0.0, 0.4, 6)
    genetic = effects @ (observed / 127.0)
    covariates = rng.normal(size=(n_train + n_test, 3))
    phenotype = genetic + covariates @ np.array([0.3, -0.2, 0.1]) + rng.normal(0.0, 1.0, n_train + n_test)
    columns = np.arange(n_train + n_test)
    records = np.arange(n_var)
    reads = harness.ReadEvidence(rows=np.zeros(0, dtype=np.int64), _pl=np.zeros((0, 0, 3), dtype=np.uint8), _columns=columns)
    train = harness.TrainData(
        variants=variants, covariates=covariates[:n_train], covariate_names=("a", "b", "c"), phenotype=phenotype[:n_train],
        trait_type="quantitative", prevalence=None, cores=1, truth_half=np.zeros(n_train, dtype=bool), reads=reads,
        _observed=observed, _columns=columns[:n_train], _records=records,
    )
    test = harness.ScoreData(
        variants=variants, covariates=covariates[n_train:], covariate_names=("a", "b", "c"), reads=reads,
        _observed=observed, _columns=columns[n_train:], _records=records,
    )
    module = harness.load_method(SUBMISSION)
    model = module.fit(train)
    prediction = model.score(test)
    assert set(prediction) == {"total", "structural"}
    assert prediction["total"].shape == (n_test,) and np.all(np.isfinite(prediction["total"]))
    assert np.all(np.isfinite(prediction["structural"]))
    assert model.profile["rows"] > 0 and model.profile["fit_seconds"] > 0
    # the store came and went under TMPDIR
    assert not any(path.name.startswith("svpgs_full_") for path in tmp_path.iterdir())
    # the structural part is the score of the TR and SV records alone: on an SNV-only submission it is zero
    if not np.any(np.asarray(variants["cls"])[model.order[model.scoring.store_rows]] >= 2):
        assert np.all(prediction["structural"] == 0.0)


def test_the_truth_arm_is_the_true_genotypes_in_code_units(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    truth = rng.integers(0, 3, size=(50, 12)).astype(np.uint8)
    np.save(tmp_path / "truth_G.npy", truth)
    measurement_truth.write_truth_arm(tmp_path)
    observed = np.load(tmp_path / "observed_truth.npy")
    assert observed.dtype == np.uint8 and np.array_equal(observed, truth * np.uint8(127))
    imputation = np.load(tmp_path / "imputation_truth.npz")
    assert np.all(imputation["info"] == 1.0) and np.all(imputation["realized_r2"] == 1.0)
    assert harness.ARMS["truth"] == ("observed_truth.npy", "imputation_truth.npz", "true genotypes")


def test_the_store_cache_builds_once_and_is_reused(tmp_path: Path, monkeypatch) -> None:
    rng = np.random.default_rng(9)
    observed, variants = _cohort(rng, 40, 60)
    columns = np.arange(60)
    reads = harness.ReadEvidence(rows=np.zeros(0, dtype=np.int64), _pl=np.zeros((0, 0, 3), dtype=np.uint8), _columns=columns)
    train = harness.TrainData(
        variants=variants, covariates=np.zeros((60, 1)), covariate_names=("a",), phenotype=np.zeros(60), trait_type="quantitative",
        prevalence=None, cores=1, truth_half=np.zeros(60, dtype=bool), reads=reads, _observed=observed, _columns=columns, _records=np.arange(40),
    )
    module = harness.load_method(SUBMISSION)
    monkeypatch.setenv(module.STORE_CACHE_VARIABLE, str(tmp_path / "cache"))
    (tmp_path / "cache").mkdir()
    built = []
    original = module.build_store
    monkeypatch.setattr(module, "build_store", lambda train, work: built.append(1) or original(train, work))
    for attempt in range(2):
        work = tmp_path / f"work{attempt}"
        work.mkdir()
        path, order = module.cached_store(train, work)
        with module.DosageStore.open(path) as store:
            assert np.array_equal(store.read_codes(0, store.n_variants), observed[order])
    assert built == [1]
    # another arm's codes are another store
    other = observed.copy()
    other[0] = 254 - other[0]
    train_other = harness.TrainData(**{**{name: getattr(train, name) for name in train.__dataclass_fields__}, "_observed": other})
    work = tmp_path / "work_other"
    work.mkdir()
    module.cached_store(train_other, work)
    assert len(list((tmp_path / "cache").iterdir())) == 2
