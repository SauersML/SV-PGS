"""Grouped against separate fits: the wall time and store passes of three traits on two training sets, six models.

On one synthetic store (``biobank_scale.generate``: block-structured LD, hard calls, one simulated trait and two
covariates), three quantitative traits share the simulated genetic value with their own scales and noise, and each is
fitted on two training sets (all samples but the first fifth, all but the last). ``separate`` fits each of the six
models in its own ``stage2_wiring.fit_models`` call, as the workspace did before models were grouped; ``grouped``
fits all six in one call, which fits the three models of each training set together (``stage2_wiring.model_groups``).
A store pass is the store's every record read once: the rows the store's arrays read or map
(``dosage_store.CodeArray.read_rows_into`` and ``view_rows``) over the record count. The report gives each arm's seconds and passes, both
by phase (``PHASES``), and how closely the grouped models' coefficients follow the separate ones', beside how closely
the separate fits with another seed follow them (the Monte Carlo spread of the fit itself). Nothing here is an
accuracy claim: the traits are this script's own simulation [own-sim], and the run measures time and reads only.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from benchmarks.biobank_scale import generate
from sv_pgs import dual_solve, full_data_fit, stage2_wiring
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import TraitType
from sv_pgs.dosage_store import CodeArray, DosageStore
from sv_pgs.progress import log

TRAITS = 3
"""Traits fitted on each training set: the store's simulated trait and two more on its genetic value."""


def _models(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(covariates with the intercept, targets [n, 6], training [n, 6]), in trait-major model order."""
    trait = np.load(root / "trait.npz")
    genetic, covariates = np.asarray(trait["genetic"]), np.asarray(trait["covariates"])
    samples = genetic.shape[0]
    generator = np.random.default_rng(len(genetic))
    traits = np.column_stack([
        trait["phenotype"],
        0.8 * genetic + 0.3 * covariates[:, 1] + 0.6 * generator.standard_normal(samples),
        -0.5 * genetic + 0.9 * generator.standard_normal(samples),
    ])
    fifth = samples // 5
    held_out = [np.arange(samples) < fifth, np.arange(samples) >= samples - fifth]
    training = np.column_stack([~rows for _trait in range(TRAITS) for rows in held_out])
    return np.column_stack([np.ones(samples), covariates]), np.repeat(traits, len(held_out), axis=1), training


def _fit(store: DosageStore, covariates: np.ndarray, targets: np.ndarray, training: np.ndarray, work: Path, budget: ComputeBudget, draw_count: int, seed: int):
    work.mkdir(parents=True, exist_ok=True)
    try:
        return stage2_wiring.fit_models(
            store=store, store_columns=np.arange(store.n_samples, dtype=np.int64), covariates=covariates,
            covariate_columns=np.ones((training.shape[1], covariates.shape[1]), dtype=bool), targets=np.where(training, targets, 0.0),
            training=training, trait_types=[TraitType.QUANTITATIVE] * training.shape[1], log_variance_offset=None, budget=budget,
            work_dir=work, seed=seed, draw_count=draw_count,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


PHASES = {
    "stage0": (stage2_wiring, "compute_genotype_statistics"),
    "sweeps": (full_data_fit._FullDataMeanField, "_sweep"),
    "exact_fields": (full_data_fit._GramMeanField, "_exact_fields"),
    "dual_iterate": (dual_solve.DualGaussian, "iterate"),
    "posterior_solves": (dual_solve.DualGaussian, "posterior_solve"),
    "response_images": (full_data_fit._FullDataMeanField, "_image"),
    "response_backs": (full_data_fit._FullDataMeanField, "_back"),
    "mixture": (full_data_fit, "_mode_mixture"),
}
"""Where a fit reads the store, by these functions on the call stack, outermost first ("mixture/sweeps": the sweeps of
the mode mixture's components): Stage 0, the mean-field sweeps (on the sample route; the band route's sweep the band,
not the store), the band route's exact fields, the dual solver's refactoring at new sites (``iterate``, every model of
a fit at once) and its posterior solves (one model's, for the fixed points' responses), the responses' images X c and
back-products X'u, and the mode mixture (``full_data_fit._mode_mixture``); every other read (the band's build, the far
field's columns, the column squares) is "other"."""


def measure(root: Path, budget: ComputeBudget, draw_count: int) -> dict:
    covariates, targets, training = _models(root)
    rows_read: dict[str, int] = {}
    seconds: dict[str, float] = {}
    phase: list[str] = []
    read_rows_into, view_rows = CodeArray.read_rows_into, CodeArray.view_rows
    originals = {name: getattr(owner, attribute) for name, (owner, attribute) in PHASES.items()}

    def counted(self, row_start, row_stop, out):
        path = "/".join(phase) or "other"
        rows_read[path] = rows_read.get(path, 0) + row_stop - row_start
        return read_rows_into(self, row_start, row_stop, out)

    def viewed(self, row_start, row_stop):
        path = "/".join(phase) or "other"
        rows_read[path] = rows_read.get(path, 0) + row_stop - row_start
        return view_rows(self, row_start, row_stop)

    def attributed(name):
        def call(*arguments, **keywords):
            phase.append(name)
            started = time.perf_counter()
            try:
                return originals[name](*arguments, **keywords)
            finally:
                seconds[name] = seconds.get(name, 0.0) + time.perf_counter() - started
                phase.pop()
        return call

    CodeArray.read_rows_into, CodeArray.view_rows = counted, viewed
    for name, (owner, attribute) in PHASES.items():
        setattr(owner, attribute, attributed(name))
    report: dict = {}
    scorings = {}
    try:
        with DosageStore.open(root / "store") as store:
            for arm in ("separate", "grouped", "reseeded"):
                rows_read.clear()
                seconds.clear()
                started = time.perf_counter()
                if arm == "grouped":
                    fits = [_fit(store, covariates, targets, training, root / "work", budget, draw_count, 1)]
                else:
                    # Separate fits, and again with another seed: the Monte Carlo spread a grouped fit is measured against.
                    seed = 1 if arm == "separate" else 2
                    fits = [_fit(store, covariates, targets[:, [model]], training[:, [model]], root / "work", budget, draw_count, seed) for model in range(targets.shape[1])]
                report[arm] = {
                    "seconds": time.perf_counter() - started, "store_passes": sum(rows_read.values()) / store.n_variants,
                    "passes_by_phase": {name: count / store.n_variants for name, count in rows_read.items()},
                    "seconds_by_phase": dict(seconds),
                    "outer_criterion_met": [bool(met) for fit in fits for met in np.asarray(fit.certificate.outer_criterion_met)],
                }
                scorings[arm] = [scoring for fit in fits for scoring in fit.scoring]
                log(f"grouped fit: {arm}: {json.dumps(report[arm])}")
    finally:
        CodeArray.read_rows_into, CodeArray.view_rows = read_rows_into, view_rows
        for name, (owner, attribute) in PHASES.items():
            setattr(owner, attribute, originals[name])
    for name, other in (("coefficient_correlation", "grouped"), ("reseeded_correlation", "reseeded")):
        report[name] = [float(np.corrcoef(separate.coefficients, model.coefficients)[0, 1]) for separate, model in zip(scorings["separate"], scorings[other])]
    report["speedup"] = report["separate"]["seconds"] / report["grouped"]["seconds"]
    report["pass_ratio"] = report["separate"]["store_passes"] / report["grouped"]["store_passes"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--variants", type=int, required=True)
    parser.add_argument("--causal", type=int, required=True)
    parser.add_argument("--heritability", type=float, required=True)
    parser.add_argument("--founders", type=int, required=True)
    parser.add_argument("--segment", type=int, nargs=2, required=True)
    parser.add_argument("--flip", type=float, required=True)
    parser.add_argument("--draw-count", type=int, required=True)
    parser.add_argument("--host-bytes", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--route", choices=("timed", "samples", "gram"), default="timed")
    parser.add_argument("--seed", type=int, default=20261013)
    arguments = parser.parse_args()
    if not (arguments.root / "store").is_dir():
        generate(arguments.root, arguments.samples, arguments.variants, arguments.causal, arguments.heritability, arguments.founders,
                 tuple(arguments.segment), arguments.flip, arguments.seed)
    budget = ComputeBudget(
        device_kind="cpu", device_ids=(), device_names=(), device_bytes=(), device_compute_capabilities=(), host_bytes=arguments.host_bytes,
        cpu_threads=arguments.threads,
    )
    if arguments.route != "timed":
        # The route pinned (``stage2_wiring.pass_costs``: the cheaper measured pass chooses it), so both are measured.
        costs = (0.0, np.inf) if arguments.route == "samples" else (np.inf, 0.0)
        stage2_wiring.pass_costs = lambda band, source, array_module: costs
    report = {"route": arguments.route, **measure(arguments.root, budget, arguments.draw_count)}
    (arguments.root / f"grouped_fit.{arguments.route}.json").write_text(json.dumps(report, indent=1))
    log(f"grouped fit: {json.dumps(report)}")


if __name__ == "__main__":
    main()
