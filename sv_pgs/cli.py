from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

from sv_pgs.all_of_us import (
    available_disease_names,
    available_measurement_names,
    prepare_all_of_us_disease_sample_table,
    prepare_all_of_us_measurement_census,
    prepare_all_of_us_measurement_sample_table,
)
from sv_pgs.artifact import write_predictions
from sv_pgs.compute_budget import detect_compute_budget
from sv_pgs.fit_model import write_model
from sv_pgs.progress import QUIET_STACK_SECONDS, sample_quiet_stacks
from sv_pgs.workspace_pipeline import ARMS, STEP_NAMES, WorkspaceConfig, run_pipeline, workspace_bindings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sv-pgs")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "list-all-of-us-diseases",
        help="List built-in All of Us disease presets.",
    )

    aou_parser = subparsers.add_parser(
        "prepare-all-of-us-disease",
        help="Query All of Us EHR condition data for a built-in disease phenotype and write a pre-fit sample table.",
    )
    aou_parser.add_argument(
        "--disease",
        required=True,
        metavar="DISEASE",
        help="Built-in disease phenotype or alias. See list-all-of-us-diseases for canonical names.",
    )
    aou_parser.add_argument("--output", required=True, help="Output TSV path for the prepared sample table.")

    subparsers.add_parser(
        "list-all-of-us-traits",
        help="List built-in All of Us quantitative traits (EHR labs and physical measurements).",
    )

    aou_trait_parser = subparsers.add_parser(
        "prepare-all-of-us-trait",
        help=(
            "Query the All of Us measurement table for a built-in quantitative trait and write a pre-fit "
            "sample table (target = per-person empirical BLUP)."
        ),
    )
    aou_trait_parser.add_argument(
        "--trait",
        required=True,
        metavar="TRAIT",
        help="Built-in trait or alias. See list-all-of-us-traits for canonical names.",
    )
    aou_trait_parser.add_argument("--output", required=True, help="Output TSV path for the prepared sample table.")

    census_parser = subparsers.add_parser(
        "census-all-of-us-traits",
        help=(
            "Count participants and rows per trait, matched concept and unit label across every built-in "
            "trait (counts of 1-20 participants suppressed): the first query to run on a new CDR."
        ),
    )
    census_parser.add_argument("--output", required=True, help="Output TSV path for the census.")

    fit_parser = subparsers.add_parser(
        "fit",
        help="Fit every (trait, training set) model of a cohort file on a dosage store and save the fitted model.",
    )
    fit_parser.add_argument("store", help="Dosage store directory.")
    fit_parser.add_argument(
        "cohort",
        help=(
            "NPZ with research_ids [n], store_columns [n], covariates [n, k] without the intercept, covariate_names [k], "
            "covariate_columns [m, k] (bool: each model's own covariates), targets [n, m], training [n, m] (bool), "
            "model_names [m] and trait_types [m]."
        ),
    )
    fit_parser.add_argument("model", help="New fitted model directory; never overwritten.")

    score_parser = subparsers.add_parser(
        "score",
        help="Score store samples with a fitted model and write their genetic scores and posterior predictive.",
    )
    score_parser.add_argument("model", help="Fitted model directory.")
    score_parser.add_argument("store", help="Dosage store directory with the model's variant layout.")
    score_parser.add_argument(
        "people",
        help="NPZ with sample_indices [n] (store columns) and covariates [n, k] in the model's covariate order, without the intercept.",
    )
    score_parser.add_argument("output", help="Output NPZ of model_names and the prediction arrays [n, models]; never overwritten.")

    workspace_parser = subparsers.add_parser(
        "workspace-run",
        help=(
            "Run the in-workspace pipeline from its JSON config, resuming "
            "from the run directory's checkpoints. Nothing is sent anywhere; outputs stay in the run directory."
        ),
    )
    workspace_parser.add_argument("--config", required=True, help="The run config JSON supplied inside the workspace.")
    workspace_parser.add_argument("--through", choices=STEP_NAMES, default=None, help="Stop after this step.")
    workspace_parser.add_argument(
        "--store-shard", type=int, default=None,
        help="Convert only the config's chromosome at this index in the store step, one task of a fan-out (run_pipeline).",
    )
    workspace_parser.add_argument(
        "--fit-arm", action="append", choices=ARMS, default=None,
        help=(
            "Fit only this arm in the fit step (repeatable), one machine's part of the arms, then score and summarize "
            "it while the store is on this disk (run_pipeline)."
        ),
    )
    workspace_parser.add_argument(
        "--finish", action="store_true",
        help=(
            "Finish the steps from the fit on from the arms' parts in the run directory (run-mirror adopt brings the other "
            "arm's), making no step before the fit, fitting nothing and reading no store (run_pipeline)."
        ),
    )

    mirror_parser = subparsers.add_parser(
        "run-mirror",
        help=(
            "Copy a workspace run's durable state to its gs:// mirror (push) or from it into a new run directory "
            "(restore), which carries the run from one Batch job to the next (sv_pgs/run_mirror.py)."
        ),
    )
    mirror_parser.add_argument("action", choices=("push", "restore", "adopt"))
    mirror_parser.add_argument("--config", required=True, help="The run config JSON, whose run_directory is mirrored.")
    mirror_parser.add_argument("--mirror", required=True, help="The mirror, gs://<workspace bucket>/<prefix>.")
    mirror_parser.add_argument(
        "--base", action="append", default=None,
        help=(
            "Another run's mirror, gs://<bucket>/<prefix>, that carries this run's completed steps before the fit: "
            "restored from where this run's mirror holds none of a step, and not pushed again where it holds one whole "
            "(repeatable; the first that holds a step serves it, then the bases this run's mirror and each base name). "
            "It is only read."
        ),
    )
    mirror_parser.add_argument(
        "--fit-arm", action="append", choices=ARMS, default=None,
        help="The arms whose restart units this job restores and pushes (repeatable; every arm's when omitted): a fit job "
             "given --fit-arm, so jobs fitting other arms of the run at once never touch each other's units.",
    )
    mirror_parser.add_argument(
        "--finishing", action="store_true",
        help="restore: only what a finishing run reads (workspace-run --finish): no store and no fit checkpoint.",
    )
    mirror_parser.add_argument(
        "--partner", default=None,
        help=(
            "adopt: the mirror of the run that fitted the other arm, gs://<bucket>/<prefix>, whose arm's parts are copied "
            "into this run's partials; prints 'partner<TAB>adopted' when every arm's parts are then here, else "
            "'partner<TAB>waiting' (the partner's job has not done its parts, or it finished the run)."
        ),
    )

    subparsers.add_parser(
        "version",
        help="Print sv-pgs package version and git commit sha.",
    )

    return parser


def _resolve_version_info() -> tuple[str, str]:
    """The installed package version and the source checkout's commit.

    Either is "unknown" when it does not exist: a source tree that was never
    installed has no package metadata, and an installed wheel has no git
    checkout (or no git on PATH) beside it.
    """
    try:
        pkg_ver = package_version("sv-pgs")
    except PackageNotFoundError:
        pkg_ver = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )
    except OSError:
        return pkg_ver, "unknown"
    git_sha = result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else "unknown"
    return pkg_ver, git_sha


def _main_impl(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "version":
        pkg_ver, git_sha = _resolve_version_info()
        print(f"sv-pgs {pkg_ver} commit {git_sha}")
        return 0

    if args.command == "fit":
        write_model(args.store, args.cohort, args.model, detect_compute_budget())
        print("model\t" + str(args.model))
        return 0

    if args.command == "score":
        write_predictions(args.model, args.store, args.people, args.output, detect_compute_budget())
        print("predictions\t" + str(args.output))
        return 0

    if args.command == "list-all-of-us-diseases":
        for disease_name in available_disease_names():
            print(disease_name)
        return 0

    if args.command == "list-all-of-us-traits":
        for trait_name in available_measurement_names():
            print(trait_name)
        return 0

    if args.command == "workspace-run":
        if os.environ.get(QUIET_STACK_SECONDS):
            sample_quiet_stacks(float(os.environ[QUIET_STACK_SECONDS]))
        summaries = run_pipeline(
            WorkspaceConfig.read(args.config), workspace_bindings(), detect_compute_budget(), through=args.through, store_shard=args.store_shard,
            fit_arms=args.fit_arm, finish=args.finish,
        )
        for step in summaries:
            print(f"{step}\tcomplete")
        return 0

    if args.command == "run-mirror":
        from sv_pgs.run_mirror import Base, GcsRemote, adopt, bases_of, mounted, push, restore

        run_directory, remote = WorkspaceConfig.read(args.config).run_directory, GcsRemote(args.mirror)
        if args.action == "adopt":
            if args.partner is None or args.partner.rstrip("/") == args.mirror.rstrip("/"):
                raise SystemExit("run-mirror adopt needs --partner, the other arm's run mirror.")
            print("partner\t" + ("adopted" if adopt(run_directory, GcsRemote(args.partner)) else "waiting"))
            return 0
        bases = bases_of(args.mirror, remote, args.base or (), lambda url: Base(GcsRemote(url), url, mounted(url)))
        if args.action == "push":
            push(run_directory, remote, bases=bases, arms=args.fit_arm)
        else:
            restore(run_directory, remote, mounted(args.mirror), bases=bases, arms=args.fit_arm, finishing=args.finishing)
        return 0

    if args.command == "census-all-of-us-traits":
        print("census\t" + str(prepare_all_of_us_measurement_census(Path(args.output))))
        return 0

    if args.command == "prepare-all-of-us-trait":
        prepared_outputs = prepare_all_of_us_measurement_sample_table(
            trait=args.trait,
            output_path=Path(args.output),
        )
        print("sample_table\t" + str(prepared_outputs.sample_table_path))
        print("sql\t" + str(prepared_outputs.sql_path))
        print("metadata\t" + str(prepared_outputs.metadata_path))
        return 0

    if args.command == "prepare-all-of-us-disease":
        prepared_outputs = prepare_all_of_us_disease_sample_table(
            disease=args.disease,
            output_path=Path(args.output),
        )
        print("sample_table\t" + str(prepared_outputs.sample_table_path))
        print("sql\t" + str(prepared_outputs.sql_path))
        print("metadata\t" + str(prepared_outputs.metadata_path))
        return 0

    raise AssertionError(f"unhandled command {args.command!r}")


def main(argv: list[str] | None = None) -> int:
    # The package's progress records (fit times, store reads, checkpoint reuse) at INFO, on stderr, which a Batch job
    # keeps in its logs; other libraries' only from WARNING.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stderr)
    logging.getLogger("sv_pgs").setLevel(logging.INFO)
    try:
        return _main_impl(argv)
    except KeyboardInterrupt as exc:
        detail = str(exc).strip()
        if detail:
            sys.stderr.write(f"[sv-pgs] interrupted: {detail}\n")
        else:
            sys.stderr.write("[sv-pgs] interrupted\n")
        sys.stderr.flush()
        return 130


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
