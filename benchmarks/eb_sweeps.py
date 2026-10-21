"""Empirical-Bayes work on one GPU: the mean-field sweeps a fit spends, by phase, for a before/after comparison.

    python -m benchmarks.eb_sweeps run --out report.json [--label before] [sizes]
    python -m benchmarks.eb_sweeps compare before.json after.json

``run`` writes a synthetic store (``biobank_scale.generate``: block-structured LD, hard calls) and fits quantitative
traits on its simulated genetic value (``--traits``: one by default, as every whole-genome fit group holds one model
[real]; up to three, one group on the same training rows, as the chr22 test fits' groups of three [real]) on the first
four fifths of the samples, with ``stage2_wiring.fit_models`` on the machine's device, as the workspace fits a group.
The store's causal effects are
drawn per allele, so their standardized sizes grow with the allele frequency and the frequency annotation's groups
(``annotation_design.frequency_annotation``) carry signal: the annotated stage has groups to enter, as the
whole-genome fits' 30-36 annotation columns do, which the repository's CPU fixtures never exercise.

Every sweep (one pass of coordinate ascent over a model's members: ``_FullDataMeanField._sweep``, a model's share of a
device's batched pass ``_FullDataMeanField._panel_sweeps``, or the band's ``_GramMeanField._sweep``) is attributed to
its phase: an EB stage's oracle calls (the stage named by
``full_data_fit.fit_full_data``: prior, base, annotated or continued), split into the first call's solve, carried
solves and cold solves, or the mode mixture (``full_data_fit._mode_mixture``), or anything else. The report has each
phase's sweeps and seconds, each EB run's calls, and each model's fitted state (hyperparameters, noise, certificate
terms, the scoring model's effects and its held-out partial r^2 on the last fifth), so ``compare`` can say how much work
a change saved and whether it reached the same fixed point: the effects' correlation and relative distance and the
held-out r^2 of each model before and after, beside the hyperparameters' largest change.

The run happens in a subprocess with a deadline (``--deadline-seconds``, 600 by default, generation and cold
compilation included); a timeout fails the command. Nothing here is an accuracy claim: the traits are this script's own
simulation [own-sim], and the run measures work and agreement only. The observers count calls and copy results; they
never change a numerical result. To compare two revisions, run this file from both source trees (copy it into the
older one) on the same machine type and sizes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEADLINE_SECONDS = 600
SAMPLES = 12_000
VARIANTS = 20_000
CAUSAL = 1_000
HERITABILITY = 0.4
FOUNDERS = 8
SEGMENT = (20, 60)
FLIP = 0.05
SEED = 20261020
HELD_OUT_FRACTION = 5
"""The last 1 / HELD_OUT_FRACTION of the samples are held out of every model's training rows and score its r^2."""


class _Ledger:
    """Sweeps and seconds by phase, the EB runs' calls, from wrappers around the engine's own methods."""

    def __init__(self) -> None:
        self.phase: list[str] = ["other"]
        self.sweeps: dict[str, int] = {}
        self.sweep_seconds: dict[str, float] = {}
        self.phase_seconds: dict[str, float] = {}
        self.runs: list[dict] = []
        self.run: dict | None = None
        self.role: str | None = None
        self.cold = False

    def label(self) -> str:
        phase = self.phase[-1]
        if phase.startswith("eb:") and self.role is not None:
            return f"{phase}:{'cold' if self.cold else self.role}"
        return phase

    def install(self, stack) -> None:
        from sv_pgs import full_data_fit
        from sv_pgs.full_data_fit import _FullDataMeanField, _GramMeanField

        ledger = self

        def patch(owner, name, wrapper) -> None:
            original = owner.__dict__[name]
            stack.callback(setattr, owner, name, original)
            setattr(owner, name, wrapper(original))

        def sweep_of(original):
            def sweep(self, model, hyperparameters):
                started = time.perf_counter()
                try:
                    return original(self, model, hyperparameters)
                finally:
                    label = ledger.label()
                    ledger.sweeps[label] = ledger.sweeps.get(label, 0) + 1
                    ledger.sweep_seconds[label] = ledger.sweep_seconds.get(label, 0.0) + time.perf_counter() - started
                    if ledger.run is not None and ledger.phase[-1].startswith("eb:"):
                        ledger.run["sweeps"] += 1
            return sweep

        for owner in (_FullDataMeanField, _GramMeanField):
            if "_sweep" in owner.__dict__:
                patch(owner, "_sweep", sweep_of)

        def panel_of(original):
            # A device's batched pass sweeps every requested model at once (never through ``_sweep``): one sweep each.
            def panel_sweeps(self, requests, kernels=None):
                started = time.perf_counter()
                swept = original(self, requests, kernels)
                if swept is not None:
                    label = ledger.label()
                    ledger.sweeps[label] = ledger.sweeps.get(label, 0) + len(swept)
                    ledger.sweep_seconds[label] = ledger.sweep_seconds.get(label, 0.0) + time.perf_counter() - started
                    if ledger.run is not None and ledger.phase[-1].startswith("eb:"):
                        ledger.run["sweeps"] += len(swept)
                return swept
            return panel_sweeps

        patch(_FullDataMeanField, "_panel_sweeps", panel_of)

        def cold_of(original):
            def cold_state(self, model):
                ledger.cold = True
                return original(self, model)
            return cold_state

        patch(_FullDataMeanField, "_cold_state", cold_of)

        def solve_of(original):
            def solve(self, model, hyperparameters, pass_budget=None):
                try:
                    return original(self, model, hyperparameters, pass_budget)
                finally:
                    ledger.cold = False
            return solve

        def solves_of(original):
            def solves(self, models, hyperparameters, budgets):
                try:
                    return original(self, models, hyperparameters, budgets)
                finally:
                    ledger.cold = False
            return solves

        for owner in (_FullDataMeanField, _GramMeanField):
            if "_solve_model" in owner.__dict__:
                patch(owner, "_solve_model", solve_of)
            if "_solve_models" in owner.__dict__:
                patch(owner, "_solve_models", solves_of)

        def call_of(original):
            def call(self, hyperparameters):
                run = ledger.run
                if run is None or run["oracle"] != id(self):
                    return original(self, hyperparameters)
                ledger.role = "first" if not run["calls"] else "carried"
                run["calls"] += 1
                try:
                    return original(self, hyperparameters)
                finally:
                    ledger.role = None
            return call

        patch(_FullDataMeanField, "__call__", call_of)

        def eb_of(original):
            def fit_hyperparameters(prior, starts, oracle, working_bytes, tolerance):
                # The stage is ``fit_full_data``'s own name for the oracle it built (its local ``solve``'s argument),
                # a frame or more up (``_timed_hyperparameters`` wraps the call); EP's warm start has none.
                frame, stage = sys._getframe(1), "ep"
                while frame is not None:
                    if frame.f_code.co_name == "solve" and isinstance(frame.f_locals.get("stage"), str):
                        stage = frame.f_locals["stage"]
                        break
                    frame = frame.f_back
                outer, ledger.run = ledger.run, {"stage": stage, "oracle": id(getattr(oracle, "_oracle", oracle)), "models": len(starts), "calls": 0, "sweeps": 0}
                ledger.phase.append(f"eb:{stage}")
                started = time.perf_counter()
                try:
                    fits = original(prior, starts, oracle, working_bytes, tolerance)
                    ledger.run.update(iterations=[int(fit.iterations) for fit in fits], halvings=[int(fit.halvings) for fit in fits],
                                      certified=[bool(fit.certified) for fit in fits])
                    return fits
                finally:
                    ledger.run["seconds"] = time.perf_counter() - started
                    ledger.phase_seconds[f"eb:{stage}"] = ledger.phase_seconds.get(f"eb:{stage}", 0.0) + ledger.run["seconds"]
                    ledger.runs.append({key: value for key, value in ledger.run.items() if key != "oracle"})
                    ledger.phase.pop()
                    ledger.run = outer
            return fit_hyperparameters

        patch(full_data_fit, "fit_hyperparameters", eb_of)

        def mixture_of(original):
            def mixture(*arguments, **keywords):
                ledger.phase.append("mixture")
                started = time.perf_counter()
                try:
                    return original(*arguments, **keywords)
                finally:
                    ledger.phase_seconds["mixture"] = ledger.phase_seconds.get("mixture", 0.0) + time.perf_counter() - started
                    ledger.phase.pop()
            return mixture

        patch(full_data_fit, "_mode_mixture", mixture_of)

    def report(self) -> dict:
        return {
            "sweeps": dict(sorted(self.sweeps.items())), "total_sweeps": sum(self.sweeps.values()),
            "sweep_seconds": {key: round(value, 3) for key, value in sorted(self.sweep_seconds.items())},
            "phase_seconds": {key: round(value, 3) for key, value in sorted(self.phase_seconds.items())}, "eb_runs": self.runs,
        }


def _traits(root: Path, count: int = 1):
    """(covariates with the intercept, targets [n, count], training [n, count], held-out rows): the store's simulated
    trait and up to two more on its genetic value (``grouped_fit``'s), all trained on the first four fifths of the
    samples."""
    import numpy as np

    trait = np.load(root / "trait.npz")
    genetic, covariates = np.asarray(trait["genetic"]), np.asarray(trait["covariates"])
    samples = genetic.shape[0]
    generator = np.random.default_rng(samples)
    targets = np.column_stack([
        trait["phenotype"],
        0.8 * genetic + 0.3 * covariates[:, 1] + 0.6 * generator.standard_normal(samples),
        -0.5 * genetic + 0.9 * generator.standard_normal(samples),
    ])[:, :count]
    held_out = np.arange(samples) >= samples - samples // HELD_OUT_FRACTION
    return np.column_stack([np.ones(samples), covariates]), targets, np.repeat(~held_out[:, None], targets.shape[1], axis=1), held_out


def _worker(arguments) -> dict:
    import contextlib

    import numpy as np

    from benchmarks.bench_sim.submissions.svpgs_full import _array_module, task_budget
    from benchmarks.biobank_scale import generate
    from sv_pgs import stage2_wiring
    from sv_pgs.artifact import StoreCodeBlocks
    from sv_pgs.config import TraitType
    from sv_pgs.dosage_store import DosageStore
    from sv_pgs.eb_monitor import partial_r2
    from sv_pgs.fast_scoring import ScoringPlan, score_genetic
    from sv_pgs.scale_mixture_ep import device_scope

    root = Path(arguments.root)
    started = time.perf_counter()
    if not (root / "store").is_dir():
        generate(root, arguments.samples, arguments.variants, arguments.causal, arguments.heritability, arguments.founders,
                 tuple(arguments.segment), arguments.flip, arguments.seed)
    generated = time.perf_counter() - started
    covariates, targets, training, held_out = _traits(root, arguments.traits)
    budget = task_budget()
    ledger = _Ledger()
    work = root / "work"
    work.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with contextlib.ExitStack() as stack:
        ledger.install(stack)
        if arguments.route != "timed":
            # The route pinned (``stage2_wiring.pass_costs``: the cheaper measured pass chooses it otherwise).
            costs = (0.0, float("inf")) if arguments.route == "samples" else (float("inf"), 0.0)
            stack.callback(setattr, stage2_wiring, "pass_costs", stage2_wiring.pass_costs)
            stage2_wiring.pass_costs = lambda band, source, array_module: costs
        with DosageStore.open(root / "store") as store, device_scope(_array_module(budget)):
            fitted = stage2_wiring.fit_models(
                store=store, store_columns=np.arange(store.n_samples, dtype=np.int64), covariates=covariates,
                covariate_columns=np.ones((targets.shape[1], covariates.shape[1]), dtype=bool), targets=np.where(training, targets, 0.0),
                training=training, trait_types=[TraitType.QUANTITATIVE] * targets.shape[1], log_variance_offset=None, budget=budget,
                work_dir=work, seed=arguments.seed, draw_count=arguments.draw_count,
            )
            fit_seconds = time.perf_counter() - started
            scores = score_genetic(StoreCodeBlocks(store, budget), ScoringPlan.from_models(fitted.scoring), budget, draws="none").means
    shutil.rmtree(work, ignore_errors=True)
    certificate = fitted.certificate
    models = []
    for model, scoring in enumerate(fitted.scoring):
        hyperparameters = fitted.hyperparameters[model]
        models.append({
            "noise": float(fitted.noise_variance[model]),
            "coefficients": np.asarray(hyperparameters.coefficients, dtype=float).tolist(),
            "log_smoothing": np.atleast_1d(np.asarray(hyperparameters.log_smoothing, dtype=float)).tolist(),
            "outer_criterion_met": bool(np.asarray(certificate.outer_criterion_met)[model]),
            "outer_iterations": int(np.asarray(certificate.outer_iterations)[model]),
            "mixture_components": int(np.asarray(certificate.mixture_components)[model]),
            "store_rows": np.asarray(scoring.store_rows, dtype=np.int64).tolist(),
            "effects": np.asarray(scoring.coefficients, dtype=float).tolist(),
            "held_out_partial_r2": partial_r2(targets[held_out, model], scores[held_out, model], covariates[held_out]),
        })
    return {
        "label": arguments.label, "device": budget.device_kind, "device_names": list(budget.device_names), "draw_count": arguments.draw_count,
        "sizes": {name: getattr(arguments, name) for name in ("samples", "variants", "causal", "heritability", "founders", "flip", "seed", "route", "traits")},
        "seconds": {"generate": round(generated, 3), "fit": round(fit_seconds, 3)}, **ledger.report(), "models": models,
    }


def _compare(before: dict, after: dict) -> list[str]:
    import numpy as np

    lines = [f"{before['label']} -> {after['label']} ({before['device']}, draw count {before['draw_count']})"]
    if before["sizes"] != after["sizes"]:
        lines.append(f"WARNING sizes differ: {before['sizes']} vs {after['sizes']}")
    for phase in sorted(set(before["sweeps"]) | set(after["sweeps"])):
        lines.append(f"  sweeps {phase}: {before['sweeps'].get(phase, 0)} -> {after['sweeps'].get(phase, 0)}")
    ratio = before["total_sweeps"] / max(after["total_sweeps"], 1)
    lines.append(f"  total sweeps {before['total_sweeps']} -> {after['total_sweeps']} ({ratio:.2f}x fewer)")
    lines.append(f"  fit seconds {before['seconds']['fit']:.0f} -> {after['seconds']['fit']:.0f} ({before['seconds']['fit'] / after['seconds']['fit']:.2f}x)")
    for model, (left, right) in enumerate(zip(before["models"], after["models"], strict=True)):
        same_rows = left["store_rows"] == right["store_rows"]
        effects = np.asarray(left["effects"]), np.asarray(right["effects"])
        correlation = float(np.corrcoef(*effects)[0, 1]) if same_rows else float("nan")
        distance = float(np.linalg.norm(effects[1] - effects[0]) / max(np.linalg.norm(effects[0]), 1e-300)) if same_rows else float("nan")
        hyper = max(abs(a - b) for a, b in zip(left["coefficients"] + left["log_smoothing"], right["coefficients"] + right["log_smoothing"]))
        lines.append(
            f"  model {model}: effects correlation {correlation:.6f}, relative distance {distance:.3g}; held-out partial r2 "
            f"{left['held_out_partial_r2']:.4f} -> {right['held_out_partial_r2']:.4f}; noise {left['noise']:.5g} -> {right['noise']:.5g}; "
            f"largest hyperparameter change {hyper:.3g}; outer criterion {left['outer_criterion_met']} -> {right['outer_criterion_met']}; "
            f"mixture components {left['mixture_components']} -> {right['mixture_components']}"
        )
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--label", default="run")
    run.add_argument("--root", type=Path, help="a directory to keep the store in (a temporary one by default)")
    run.add_argument("--samples", type=int, default=SAMPLES)
    run.add_argument("--variants", type=int, default=VARIANTS)
    run.add_argument("--causal", type=int, default=CAUSAL)
    run.add_argument("--heritability", type=float, default=HERITABILITY)
    run.add_argument("--founders", type=int, default=FOUNDERS)
    run.add_argument("--segment", type=int, nargs=2, default=list(SEGMENT))
    run.add_argument("--flip", type=float, default=FLIP)
    run.add_argument("--seed", type=int, default=SEED)
    run.add_argument("--traits", type=int, choices=(1, 2, 3), default=1, help="models in the one fit group")
    run.add_argument("--draw-count", type=int, help="the fit's K (fit_model.DRAW_COUNT, the workspace's, by default)")
    run.add_argument("--route", choices=("timed", "samples", "gram"), default="timed", help="the fit's route: chosen by its measured pass costs, or pinned")
    run.add_argument("--deadline-seconds", type=float, default=DEADLINE_SECONDS)
    run.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    compare = commands.add_parser("compare")
    compare.add_argument("before", type=Path)
    compare.add_argument("after", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "run" and arguments.draw_count is None:
        from sv_pgs.fit_model import DRAW_COUNT

        arguments.draw_count = DRAW_COUNT
    if arguments.command == "compare":
        print("\n".join(_compare(json.loads(arguments.before.read_text()), json.loads(arguments.after.read_text()))))
        return 0
    if arguments.worker:
        arguments.out.write_text(json.dumps(_worker(arguments)))
        return 0
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="svpgs-eb-sweeps-") as scratch:
        root = arguments.root or Path(scratch)
        command = [sys.executable, "-m", "benchmarks.eb_sweeps", *(argv if argv is not None else sys.argv[1:]), "--worker", "--root", str(root)]
        environment = dict(os.environ, PYTHONUNBUFFERED="1")
        try:
            completed = subprocess.run(command, env=environment, timeout=arguments.deadline_seconds, check=False)
        except subprocess.TimeoutExpired:
            print(f"FAIL eb_sweeps deadline {time.perf_counter() - started:.0f} s exceeded {arguments.deadline_seconds:g} s", flush=True)
            return 1
    if completed.returncode != 0:
        print(f"FAIL eb_sweeps worker exit {completed.returncode}", flush=True)
        return 1
    report = json.loads(arguments.out.read_text())
    print(f"PASS eb_sweeps {time.perf_counter() - started:.0f} s: {report['total_sweeps']} sweeps {report['sweeps']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
