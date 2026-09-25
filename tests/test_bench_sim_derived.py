"""The bench-sim prototype arms whose columns are fixed combinations of records (TR locus length, REF state)."""

from pathlib import Path

import numpy as np

from benchmarks.bench_sim import harness
from benchmarks.bench_sim.submissions.svpgs_full_derived import derived_values
from benchmarks.bench_sim.submissions.svpgs_full_refstate import reference_columns
from benchmarks.bench_sim.submissions.svpgs_full_trlocus import locus_columns

SUBMISSIONS = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_sim" / "submissions"


def _cohort(rng: np.random.Generator, n_var: int, n_samples: int):
    frequency = rng.uniform(0.05, 0.5, n_var)
    genotype = (rng.uniform(size=(n_var, n_samples)) < frequency[:, None]).astype(np.uint8) + (rng.uniform(size=(n_var, n_samples)) < frequency[:, None]).astype(np.uint8)
    observed = genotype * np.uint8(127)
    cls = rng.choice(4, size=n_var, p=(0.6, 0.1, 0.2, 0.1)).astype(np.int64)
    len_change = np.where(cls == 0, 0, rng.choice([-6, -2, 3, 8], size=n_var))
    # Pairs of records share a position (split multiallelic sites); runs of four share a repeat locus.
    positions = np.repeat(np.sort(rng.choice(10**6, size=(n_var + 1) // 2, replace=False)), 2)[:n_var].astype(np.int64)
    locus = np.where(rng.random(n_var // 4 + 1) < 0.5, np.arange(n_var // 4 + 1), -1).repeat(4)[:n_var]
    variants = {
        "pos": positions,
        "cm": np.linspace(0.0, 1.0, n_var),
        "cls": cls,
        "len_change": len_change,
        "ref_len": np.ones(n_var, dtype=np.int64),
        "alt_len": np.ones(n_var, dtype=np.int64),
        "imputation_info": np.full(n_var, np.nan),
        "in_gene": (rng.random(n_var) < 0.3).astype(np.float64),
        "in_exon": (rng.random(n_var) < 0.1).astype(np.float64),
        "in_repeat": (locus >= 0).astype(np.float64),
        "repeat_locus": locus,
        "log_tss_distance": rng.normal(9.0, 2.0, n_var),
        "log_sv_length": np.where(cls == 3, rng.normal(5.0, 1.0, n_var), 0.0),
        "class_names": np.array(["SNV", "INDEL", "TR", "SV"]),
    }
    return observed, variants


def _data(observed, variants, columns):
    reads = harness.ReadEvidence(rows=np.zeros(0, dtype=np.int64), _pl=np.zeros((0, 0, 3), dtype=np.uint8), _columns=columns)
    return harness.ScoreData(
        variants=variants, covariates=np.zeros((columns.shape[0], 1)), covariate_names=("a",), reads=reads,
        _observed=observed, _columns=columns, _records=np.arange(observed.shape[0]),
    )


def test_the_derived_columns_are_their_records_combinations() -> None:
    rng = np.random.default_rng(1)
    observed, variants = _cohort(rng, 40, 30)
    data = _data(observed, variants, np.arange(30))
    dosage = observed / 127.0
    loci = locus_columns(variants)
    members = (variants["repeat_locus"] >= 0) & (variants["len_change"] != 0) & (variants["cls"] == 2)
    np.testing.assert_array_equal(np.sort(loci.replaced), np.flatnonzero(members))
    for row, locus in enumerate(np.unique(variants["repeat_locus"][members])):
        in_locus = members & (variants["repeat_locus"] == locus)
        np.testing.assert_allclose(derived_values(data, loci)[row], variants["len_change"][in_locus] @ dosage[in_locus])
    reference = reference_columns(variants)
    assert reference.replaced.shape == (0,)
    sites = [position for position in np.unique(variants["pos"])
             if (variants["pos"] == position).sum() > 1 and not np.any(variants["cls"][variants["pos"] == position] == 2)]
    values = derived_values(data, reference)
    assert values.shape[0] == len(sites)
    for row, position in enumerate(sites):
        np.testing.assert_allclose(values[row], 2.0 - dosage[variants["pos"] == position].sum(axis=0))


def test_the_derived_arms_fit_and_score_a_small_cohort(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    rng = np.random.default_rng(3)
    n_var, n_train, n_test = 60, 240, 80
    observed, variants = _cohort(rng, n_var, n_train + n_test)
    effects = np.zeros(n_var)
    effects[rng.choice(n_var, 6, replace=False)] = rng.normal(0.0, 0.4, 6)
    covariates = rng.normal(size=(n_train + n_test, 3))
    phenotype = effects @ (observed / 127.0) + covariates @ np.array([0.3, -0.2, 0.1]) + rng.normal(0.0, 1.0, n_train + n_test)
    columns = np.arange(n_train + n_test)
    reads = harness.ReadEvidence(rows=np.zeros(0, dtype=np.int64), _pl=np.zeros((0, 0, 3), dtype=np.uint8), _columns=columns)
    train = harness.TrainData(
        variants=variants, covariates=covariates[:n_train], covariate_names=("a", "b", "c"), phenotype=phenotype[:n_train],
        trait_type="quantitative", prevalence=None, cores=1, truth_half=np.zeros(n_train, dtype=bool), reads=reads,
        _observed=observed, _columns=columns[:n_train], _records=np.arange(n_var),
    )
    test = harness.ScoreData(
        variants=variants, covariates=covariates[n_train:], covariate_names=("a", "b", "c"), reads=reads,
        _observed=observed, _columns=columns[n_train:], _records=np.arange(n_var),
    )
    for name in ("svpgs_full_trlocus.py", "svpgs_full_refstate.py"):
        model = harness.load_method(SUBMISSIONS / name).fit(train)
        prediction = model.score(test)
        assert prediction["total"].shape == (n_test,) and np.all(np.isfinite(prediction["total"]))
        assert np.all(np.isfinite(prediction["structural"]))
        assert model.profile["derived_columns"] > 0
    assert not any(path.name.startswith("svpgs_full_") for path in tmp_path.iterdir())
