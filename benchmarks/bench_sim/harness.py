"""bench-sim harness: runs a submitted method on one scenario and scores it (PREREG.md section 5).

Submission contract (a Python file passed with --method):

    def fit(train: TrainData) -> model
    model.score(test: ScoreData) -> np.ndarray [n_test]  or  {"total": [n_test], "structural": [n_test]}

"structural" (optional) is the part of the prediction carried by TR+SV columns; the harness uses it for
SV credit. Methods only ever see observed codes, the public variant table, covariates, PCs and the
training phenotype. Truth lives outside their reach.

    python harness.py run --method m.py --cohort <cohort/chr22> --scenario <dev/scenario_003> --out <results/...>
    python harness.py score --cohort ... --scenario ... --prediction <results/.../prediction.npz>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import resource
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import norm

from benchmarks.bench_sim.records import measured_records

CODES_PER_DOSAGE = 127
# Measurement arms: observed codes and their per-record quality, plus the label every result carries.
ARMS = {
    "glimpse2": ("observed.npy", "imputation.npz", "GLIMPSE2-imputed"),
    "beagle": ("observed_beagle.npy", "imputation_beagle.npz", "Beagle-imputed"),
    # novel-measure's Rao-Blackwellised TR/SV columns built from the Beagle arm's phased simple-site input and
    # the panel only; SNV/INDEL rows are the Beagle arm's. Assembled by bench-sim after an input audit.
    "beagle_rb": ("observed_beagle_rb.npy", "imputation_beagle_rb.npz", "Beagle-imputed + RB structural columns (novel-measure)"),
    # A flagged fifth of the training samples observed at their true genotypes (PREREG amendment 8).
    "beagle_truthhalf": ("observed_beagle_truth.npy", "imputation_beagle.npz", "Beagle-imputed with a true-genotype training half"),
}
TRUTH_HALF_ARMS = {"beagle_truthhalf": "truth_half.npy"}
VARIANT_FIELDS = ("pos", "cm", "cls", "len_change", "ref_len", "alt_len")
ANNOTATION_FIELDS = ("in_gene", "in_exon", "log_tss_distance", "in_repeat", "log_sv_length")


@dataclass
class TrainData:
    """Training view. codes(rows) returns observed uint8 codes (dosage = code / 127) for training samples."""
    variants: dict
    covariates: np.ndarray
    covariate_names: tuple
    phenotype: np.ndarray
    trait_type: str
    prevalence: float | None
    cores: int
    truth_half: np.ndarray
    _observed: np.ndarray = field(repr=False)
    _columns: np.ndarray = field(repr=False)
    _records: np.ndarray = field(repr=False)

    @property
    def n_variants(self) -> int:
        return int(self._records.size)

    @property
    def n_samples(self) -> int:
        return int(self._columns.size)

    def codes(self, rows) -> np.ndarray:
        return np.asarray(self._observed[self._records[rows]])[..., self._columns]


@dataclass
class ScoreData:
    variants: dict
    covariates: np.ndarray
    covariate_names: tuple
    _observed: np.ndarray = field(repr=False)
    _columns: np.ndarray = field(repr=False)
    _records: np.ndarray = field(repr=False)

    @property
    def n_variants(self) -> int:
        return int(self._records.size)

    @property
    def n_samples(self) -> int:
        return int(self._columns.size)

    def codes(self, rows) -> np.ndarray:
        return np.asarray(self._observed[self._records[rows]])[..., self._columns]


def public_variant_table(cohort: Path, arm: str) -> dict:
    variants = np.load(cohort / "variants.npz")
    annotations = np.load(cohort / "annotations.npz")
    imputation = np.load(cohort / ARMS[arm][1])
    records = np.flatnonzero(measured_records(cohort))
    table = {name: variants[name][records] for name in VARIANT_FIELDS}
    table.update({name: annotations[name][records] for name in ANNOTATION_FIELDS})
    table["imputation_info"] = imputation["info"][records]
    table["class_names"] = np.array(["SNV", "INDEL", "TR", "SV"])
    return table


def covariate_matrix(cohort: Path, arm: str) -> tuple[np.ndarray, tuple]:
    samples = np.load(cohort / "samples.npz")
    pcs = np.load(cohort / f"pcs_{arm}.npz")["pcs"]
    age = samples["age"]
    matrix = np.column_stack([samples["sex"], (age - age.mean()) / age.std(), samples["batch"], pcs])
    names = ("sex", "age", "batch", *[f"pc{index + 1}" for index in range(pcs.shape[1])])
    return matrix, names


def load_method(path: Path):
    spec = importlib.util.spec_from_file_location("submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(method: Path, cohort: Path, scenario: Path, out: Path, cores: int, arm: str) -> None:
    samples = np.load(cohort / "samples.npz")
    is_test = samples["is_test"]
    train_columns, test_columns = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    observed = np.load(cohort / ARMS[arm][0], mmap_mode="r")
    records = np.flatnonzero(measured_records(cohort))
    table = public_variant_table(cohort, arm)
    covariates, names = covariate_matrix(cohort, arm)
    params = json.loads((scenario / "scenario.json").read_text())["params"]
    phenotype = np.load(scenario / "truth.npz")["phenotype"]
    train = TrainData(
        variants=table, covariates=covariates[train_columns], covariate_names=names, phenotype=phenotype[train_columns].copy(),
        trait_type="binary" if params["binary"] else "quantitative",
        prevalence=params["prevalence"] if params["binary"] else None, cores=cores,
        truth_half=(np.load(cohort / TRUTH_HALF_ARMS[arm])[train_columns] if arm in TRUTH_HALF_ARMS
                    else np.zeros(train_columns.size, dtype=bool)),
        _observed=observed, _columns=train_columns, _records=records,
    )
    test = ScoreData(variants=table, covariates=covariates[test_columns], covariate_names=names,
                     _observed=observed, _columns=test_columns, _records=records)
    del phenotype
    module = load_method(method)
    started = time.time()
    model = module.fit(train)
    fitted = time.time()
    prediction = model.score(test)
    finished = time.time()
    if not isinstance(prediction, dict):
        prediction = {"total": prediction}
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "prediction.npz", **{key: np.asarray(value, dtype=np.float64) for key, value in prediction.items()})
    meta = {
        "method": str(method), "scenario": str(scenario), "cores": cores, "measurement": ARMS[arm][2],
        "fit_seconds": fitted - started, "score_seconds": finished - fitted,
        "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def incremental_r2(outcome: np.ndarray, prediction: np.ndarray, covariates: np.ndarray) -> tuple[float, float]:
    """R2 gain of adding the prediction to an OLS on covariates (test samples), and the prediction's slope."""
    base = np.column_stack([np.ones(outcome.size), covariates])

    def fit(design: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients, *_ = np.linalg.lstsq(design, outcome, rcond=None)
        residual = outcome - design @ coefficients
        centred = outcome - outcome.mean()
        return 1.0 - residual @ residual / (centred @ centred), coefficients

    r2_base, _ = fit(base)
    r2_full, coefficients = fit(np.column_stack([base, prediction]))
    return r2_full - r2_base, float(coefficients[-1])


def auc(outcome: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score)
    ranks = np.empty(score.size)
    ranks[order] = np.arange(1, score.size + 1)
    positives = outcome > 0.5
    count_positive = positives.sum()
    count_negative = score.size - count_positive
    return float((ranks[positives].sum() - count_positive * (count_positive + 1) / 2) / (count_positive * count_negative))


def score(cohort: Path, scenario: Path, prediction_path: Path, arm: str) -> dict:
    samples = np.load(cohort / "samples.npz")
    is_test = samples["is_test"]
    test_columns = np.flatnonzero(is_test)
    covariates, _ = covariate_matrix(cohort, arm)
    record = json.loads((scenario / "scenario.json").read_text())
    params = record["params"]
    truth = np.load(scenario / "truth.npz")
    outcome = truth["phenotype"][test_columns]
    prediction = np.load(prediction_path)
    total = prediction["total"]
    result: dict = {"scenario": scenario.name, "measurement": ARMS[arm][2]}
    gain, slope = incremental_r2(outcome, total, covariates[test_columns])
    result["incremental_r2"] = gain
    result["calibration_slope"] = slope
    if params["binary"]:
        prevalence = params["prevalence"]
        height = norm.pdf(norm.ppf(1.0 - prevalence))
        result["liability_r2"] = gain * prevalence * (1.0 - prevalence) / height**2
        result["auc"] = auc(outcome, total)
    groups = samples["group"][test_columns]
    names = samples["group_names"]
    for index, name in enumerate(names):
        members = groups == index
        if members.sum() > covariates.shape[1] + 2:
            result[f"r2_{name}"] = incremental_r2(outcome[members], total[members], covariates[test_columns][members])[0]
    if "structural" in prediction.files and np.var(total) > 0:
        result["structural_share_predicted"] = float(np.cov(prediction["structural"], total)[0, 1] / np.var(total, ddof=1))
    result["structural_share_truth"] = record["summary"]["structural_share"]
    result["oracle_r2"] = incremental_r2(outcome, truth["genetic_value"][test_columns], covariates[test_columns])[0]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--method", required=True)
    run_parser.add_argument("--cohort", required=True)
    run_parser.add_argument("--scenario", required=True)
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--cores", type=int, default=16)
    run_parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--cohort", required=True)
    score_parser.add_argument("--scenario", required=True)
    score_parser.add_argument("--prediction", required=True)
    score_parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(Path(args.method), Path(args.cohort), Path(args.scenario), Path(args.out), args.cores, args.arm)
    else:
        print(json.dumps(score(Path(args.cohort), Path(args.scenario), Path(args.prediction), args.arm), indent=1))


if __name__ == "__main__":
    main()
