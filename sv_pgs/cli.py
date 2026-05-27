from __future__ import annotations

import argparse
import faulthandler
import io
import os
import platform
import sys
from pathlib import Path
from typing import Iterable

from sv_pgs.all_of_us import (
    AllOfUsDiseaseRequest,
    available_disease_names,
    prepare_all_of_us_disease_sample_table,
    resolve_disease_definition,
)
from sv_pgs.aou_runner import _normalize_variants_choice, run_all_of_us, run_all_of_us_all_diseases
from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.evaluate import evaluate_all_of_us
from sv_pgs.io import load_dataset_from_files
from sv_pgs.pipeline import run_training_pipeline
from sv_pgs.progress import gpu_memory_snapshot, jax_runtime_snapshot, log, log_autotune_banner, nvidia_smi_snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sv-pgs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    disease_list_parser = subparsers.add_parser(
        "list-all-of-us-diseases",
        help="List built-in All of Us disease presets.",
    )
    disease_list_parser.set_defaults(command="list-all-of-us-diseases")

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

    aou_run_parser = subparsers.add_parser(
        "run-all-of-us",
        help="Full AoU pipeline: download VCFs, prepare phenotype, merge PCs, and fit one unified genome-wide Bayesian model.",
    )
    aou_run_parser.add_argument(
        "--disease",
        default=None,
        help=(
            "Disease name (e.g. hypertension, type2_diabetes). Pass 'all' or "
            "'top20' to loop over every built-in disease. Mutually exclusive "
            "with --all-diseases."
        ),
    )
    aou_run_parser.add_argument(
        "--all-diseases",
        action="store_true",
        help=(
            "Run the AoU pipeline sequentially over every built-in disease, "
            "writing per-disease outputs under <output-dir>/<canonical_name>_results/."
        ),
    )
    aou_run_parser.add_argument("--chromosomes", default="1-22", help="Chromosome range (default: 1-22).")
    aou_run_parser.add_argument("--output-dir", required=True, help="Base output directory.")
    aou_run_parser.add_argument(
        "--variant-metadata",
        help="Optional CSV or TSV keyed by variant_id. Every non-reserved column is used as a prior annotation.",
    )
    aou_run_parser.add_argument("--n-pcs", type=int, default=10, help="Number of genomic PCs to include (default: 10).")
    aou_run_parser.add_argument("--random-seed", type=int, default=0)
    aou_run_parser.add_argument(
        "--max-parallel-gpus",
        type=int,
        default=None,
        help=(
            "For --all-diseases / --disease all, explicitly run multiple "
            "diseases concurrently by pinning each subprocess to one GPU. "
            "Default runs one disease at a time with all visible GPUs available "
            "to the fit."
        ),
    )
    aou_run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Parse flags, run preflight, print the planned actions, then exit "
            "0 without touching disk (beyond reading the phenotype table) or "
            "network. Useful for sanity-checking a run before launching the "
            "multi-hour pipeline."
        ),
    )
    aou_run_parser.add_argument(
        "--variants",
        # Default is the joint model — array SNPs typically add the most
        # explained variance to a PGS and SVs sharpen tagged-region effects;
        # restricting to either alone is an opt-out, not the common case.
        default="snp+sv",
        # "snv" is accepted as an alias for "snp" since SNVs and SNPs are
        # used interchangeably in much of the polygenic-score literature.
        # _normalize_variants_choice() canonicalizes it back to "snp" so the
        # rest of the pipeline only ever sees the three canonical choices.
        choices=("sv", "snp", "snv", "snp+sv", "snv+sv", "sv+snp", "sv+snv"),
        help=(
            "Genotype sources for the model. 'snp+sv' (default) is the joint "
            "model: AoU microarray PLINK SNPs (447k samples, ~700k variants) "
            "PLUS AoU srWGS SV VCFs (97k samples, ~1.7M variants), intersected "
            "to the SV cohort. 'sv' restricts to SVs only. 'snp' (alias 'snv') "
            "restricts to microarray SNPs only. The +-separated forms accept "
            "either ordering."
        ),
    )

    run_parser = subparsers.add_parser("run", help="Load genotype files, fit the Bayesian model, and write outputs.")
    run_parser.add_argument("--genotypes", required=True, help="Path to a VCF/BCF file or PLINK 1 .bed file.")
    run_parser.add_argument(
        "--genotype-format",
        default="auto",
        choices=("auto", "vcf", "plink1"),
        help="Input genotype format. Default infers from the path.",
    )
    run_parser.add_argument(
        "--sample-table",
        required=True,
        help="CSV or TSV with an identifier column plus target and covariates.",
    )
    run_parser.add_argument(
        "--sample-id-column",
        default="auto",
        help="Sample identifier column in the sample table. Default auto-detects sample_id, research_id, or person_id.",
    )
    run_parser.add_argument("--target-column", required=True, help="Target column in the sample table.")
    run_parser.add_argument(
        "--covariate-column",
        action="append",
        default=[],
        help="Covariate column in the sample table. Repeat for multiple covariates.",
    )
    run_parser.add_argument(
        "--variant-metadata",
        help=(
            "Optional CSV or TSV keyed by variant_id. Every non-reserved column is inferred "
            "as a binary, numeric, categorical, weighted-membership, or nested annotation."
        ),
    )
    run_parser.add_argument("--output-dir", required=True, help="Directory for artifact and result tables.")
    run_parser.add_argument("--max-outer-iterations", type=int, default=20)
    run_parser.add_argument("--random-seed", type=int, default=0)
    run_parser.add_argument(
        "--marginal-screen-min-abs-z",
        type=float,
        default=0.0,
        help=(
            "Univariate |z| pre-screen threshold (residualized on covariates). "
            "0.0 disables."
        ),
    )
    run_parser.add_argument(
        "--allow-nonconverged-export",
        action="store_true",
        help=(
            "Suppress the audit-trail marker emitted when exporting a fit "
            "that reports converged=False. The artifact is ALWAYS written "
            "regardless of this flag — when converged=False a clear WARNING "
            "is logged with the four final_*_change deltas "
            "(parameter/predictor/objective/hyperparameter). Setting this "
            "flag acknowledges the non-convergence so the warning is not "
            "tagged as an unacknowledged override. Callers that need strict "
            "gating should introspect fit_result.converged directly."
        ),
    )

    eval_parser = subparsers.add_parser(
        "evaluate-all-of-us",
        help="Quasi-holdout evaluation using ICD code stratification and survey self-report.",
    )
    eval_parser.add_argument("--output-dir", required=True, help="Output directory from a completed run-all-of-us.")
    eval_parser.add_argument("--disease", required=True, help="Disease name (must match the training run).")

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run preflight + bitpacked smoke + bench quick mode and print a status table.",
    )
    doctor_parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache_dir for preflight; defaults to a tmpdir.",
    )

    subparsers.add_parser(
        "version",
        help="Print sv-pgs package version and git commit sha.",
    )

    return parser


def _resolve_version_info() -> tuple[str, str]:
    try:
        from importlib.metadata import version as _pkg_version

        pkg_ver = _pkg_version("sv-pgs")
    except Exception:
        pkg_ver = "unknown"
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        git_sha = result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else "unknown"
    except Exception:
        git_sha = "unknown"
    return pkg_ver, git_sha


def _install_graceful_shutdown_handlers() -> None:
    """Convert SIGTERM/SIGHUP into KeyboardInterrupt so atexit + finally + GPU
    teardown run instead of dying mid-write.

    Default Python behavior: SIGINT (^C) raises KeyboardInterrupt which unwinds
    cleanly (atexit fires, finally blocks run, ThreadPoolExecutor.shutdown is
    called, CuPy contexts get released). SIGTERM/SIGHUP just kill the
    interpreter with no cleanup, which leaves partial cache writes, orphaned
    decode threads pinning the GPU, and tmp files lying around. Re-raise them
    as KeyboardInterrupt so they unwind the same way.

    Called once at startup of any long-running ``sv-pgs`` command. Without
    this, ``run.sh``'s SIGTERM-on-cleanup tears down the bash side cleanly
    but the Python child still skips its cache writes.
    """
    import signal

    def _term_to_interrupt(signum: int, _frame: object) -> None:
        try:
            signame = signal.Signals(signum).name
        except (ValueError, AttributeError):
            signame = str(signum)
        sys.stderr.write(
            f"\n[sv-pgs] received {signame}; unwinding for graceful shutdown "
            f"(atexit + GPU teardown + cache flush)\n"
        )
        sys.stderr.flush()
        # Re-install the default disposition so a SECOND signal hard-kills
        # if the graceful unwind ends up hanging on a syscall.
        signal.signal(signum, signal.SIG_DFL)
        raise KeyboardInterrupt(f"signal {signame}")

    # SIGTSTP is included because the AoU workbench Ctrl-Z's a backgrounded
    # ``sv-pgs run`` and exits the parent shell, which delivers SIGTSTP (rc=148)
    # to the Python child. Default SIGTSTP suspends the process and on parent
    # exit it is reaped without unwinding atexit / finally / GPU teardown,
    # leaving partial cache writes behind. Convert it to KeyboardInterrupt so
    # the EM loop's per-iter checkpoint callback persists the in-flight state
    # and the active-matrix cache publishes atomically before exit.
    for sig_name in ("SIGTERM", "SIGHUP", "SIGTSTP"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _term_to_interrupt)
        except (OSError, ValueError):
            # Not the main thread, or platform doesn't support — accept silently.
            pass


def _main_impl(argv: list[str] | None = None) -> int:
    _install_graceful_shutdown_handlers()
    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except io.UnsupportedOperation:
        pass
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        pkg_ver, git_sha = _resolve_version_info()
        print(f"sv-pgs {pkg_ver} commit {git_sha}")
        return 0

    if args.command == "list-all-of-us-diseases":
        for disease_name in available_disease_names():
            print(disease_name)
        return 0

    if args.command == "prepare-all-of-us-disease":
        prepared_outputs = prepare_all_of_us_disease_sample_table(
            request=AllOfUsDiseaseRequest(disease=args.disease),
            output_path=Path(args.output),
        )
        print("sample_table\t" + str(prepared_outputs.sample_table_path))
        print("sql\t" + str(prepared_outputs.sql_path))
        print("metadata\t" + str(prepared_outputs.metadata_path))
        return 0

    if args.command == "run-all-of-us":
        chromosome_text = args.chromosomes
        if "-" in chromosome_text:
            low, high = chromosome_text.split("-", 1)
            chromosomes = list(range(int(low), int(high) + 1))
        elif "," in chromosome_text:
            chromosomes = [int(chromosome.strip()) for chromosome in chromosome_text.split(",")]
        else:
            chromosomes = [int(chromosome_text)]

        disease_value = args.disease
        normalized_disease = disease_value.strip().lower() if isinstance(disease_value, str) else None
        wants_all = bool(args.all_diseases) or normalized_disease in {"all", "top20"}
        if wants_all and args.all_diseases and disease_value is not None and normalized_disease not in {"all", "top20"}:
            raise ValueError("--all-diseases is mutually exclusive with --disease")
        if disease_value is None and not wants_all:
            raise ValueError("Either --disease or --all-diseases is required.")
        if getattr(args, "dry_run", False):
            return _run_dry_run(
                disease=disease_value,
                wants_all=wants_all,
                chromosomes=chromosomes,
                output_dir=args.output_dir,
                variants=args.variants,
                n_pcs=args.n_pcs,
                random_seed=args.random_seed,
            )
        if wants_all:
            # Propagate the sweep's non-zero exit code so CI surfaces any
            # per-disease subprocess failure instead of silently swallowing it.
            return int(run_all_of_us_all_diseases(
                chromosomes=chromosomes,
                output_base=args.output_dir,
                variant_metadata_path=args.variant_metadata,
                n_pcs=args.n_pcs,
                random_seed=args.random_seed,
                variants=args.variants,
                max_parallel_gpus=args.max_parallel_gpus,
            ) or 0)
        assert disease_value is not None
        run_all_of_us(
            disease=disease_value,
            chromosomes=chromosomes,
            output_base=args.output_dir,
            variant_metadata_path=args.variant_metadata,
            n_pcs=args.n_pcs,
            random_seed=args.random_seed,
            variants=args.variants,
        )
        return 0

    if args.command == "doctor":
        return _run_doctor(cache_dir_arg=getattr(args, "cache_dir", None))

    if args.command == "evaluate-all-of-us":
        evaluate_all_of_us(
            output_dir=Path(args.output_dir),
            disease=args.disease,
        )
        return 0

    if args.command != "run":
        raise ValueError("Unsupported command: " + str(args.command))

    try:
        with open("/proc/meminfo") as meminfo_file:
            for line in meminfo_file:
                if line.startswith("MemTotal:"):
                    total_gb = int(line.split()[1]) / 1024 / 1024
                    break
            else:
                total_gb = -1
        mem_info = f"total_ram={total_gb:.1f} GB"
    except OSError:
        mem_info = "total_ram=unknown"

    log(f"=== CLI RUN START ===  pid={os.getpid()}  {mem_info}  cpu_count={os.cpu_count()}  platform={platform.platform()}")
    log(f"jax runtime: {jax_runtime_snapshot()}")
    log(f"gpu memory: {gpu_memory_snapshot()}")
    log(f"nvidia-smi: {nvidia_smi_snapshot()}")
    log_autotune_banner()
    log(f"genotypes={args.genotypes} sample_table={args.sample_table} output_dir={args.output_dir}")
    log(f"genotype_format={args.genotype_format} sample_id_column={args.sample_id_column} target_column={args.target_column}")
    log(
        "covariates="
        + f"{list(args.covariate_column)}  max_outer_iter={args.max_outer_iterations}"
        + f"  seed={args.random_seed}"
    )

    config = ModelConfig(
        max_outer_iterations=args.max_outer_iterations,
        random_seed=args.random_seed,
        marginal_screen_min_abs_z=args.marginal_screen_min_abs_z,
        allow_nonconverged_export=args.allow_nonconverged_export,
    )
    dataset = load_dataset_from_files(
        genotype_path=args.genotypes,
        genotype_format=args.genotype_format,
        sample_table_path=args.sample_table,
        sample_id_column=args.sample_id_column,
        target_column=args.target_column,
        covariate_columns=args.covariate_column,
        variant_metadata_path=args.variant_metadata,
        config=config,
    )
    log(f"dataset loaded: samples={len(dataset.sample_ids)} variants={dataset.genotypes.shape[1]}")
    inferred_trait_type = _infer_trait_type(dataset.targets)
    log(f"inferred trait type: {inferred_trait_type.value}")
    config.trait_type = inferred_trait_type
    pipeline_outputs = run_training_pipeline(
        dataset=dataset,
        config=config,
        output_dir=Path(args.output_dir),
    )

    log("=== CLI RUN DONE ===")
    print("artifact_dir\t" + str(pipeline_outputs.artifact_dir))
    print("summary\t" + str(pipeline_outputs.summary_path))
    print("predictions\t" + str(pipeline_outputs.predictions_path))
    print("coefficients\t" + str(pipeline_outputs.coefficients_path))
    return 0


def main(argv: list[str] | None = None) -> int:
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


def _run_dry_run(
    *,
    disease: str | None,
    wants_all: bool,
    chromosomes: list[int],
    output_dir: str,
    variants: str,
    n_pcs: int,
    random_seed: int,
) -> int:
    """Print the planned actions of a run-all-of-us invocation without touching disk/network."""

    from sv_pgs.preflight import check_aou_preflight, log_preflight

    canonical_variants = _normalize_variants_choice(variants)
    if canonical_variants == "snp":
        cohort_source = "AoU microarray PLINK"
        sample_count_str = "~447,278 (pre-intersection)"
        download_action = "download/stage microarray PLINK trio (~194 GB)"
    elif canonical_variants == "sv":
        cohort_source = "AoU srWGS SV VCFs"
        sample_count_str = "~97,000 (pre-intersection)"
        download_action = "download/stage srWGS SV VCFs (~1.7M variants)"
    else:
        cohort_source = "AoU microarray PLINK + srWGS SV VCFs (joint)"
        sample_count_str = "~97,000 (intersected to SV cohort)"
        download_action = (
            "download/stage microarray PLINK trio (~194 GB) + srWGS SV VCFs"
        )

    out_base = Path(output_dir)
    if wants_all:
        disease_label = "ALL (sweep)"
        resolved_output_dir = str(out_base / "<canonical_name>_results")
    else:
        assert disease is not None
        disease_def = resolve_disease_definition(disease)
        disease_label = disease_def.canonical_name
        resolved_output_dir = str(out_base)
    cache_dir_path = out_base.parent / ".sv_pgs_cache" if not wants_all else out_base / ".sv_pgs_cache"

    # Preflight against an EXISTING tmpdir so we don't create the user's output dir.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="sv_pgs_dryrun_preflight_") as tmp:
        report = check_aou_preflight(
            cache_dir=Path(tmp),
            require_gpu=False,
        )
        log_preflight(report)

    print()
    print("=== dry run ===")
    print(f"disease:        {disease_label}")
    print(f"variants:       {canonical_variants}")
    print(f"chromosomes:    {chromosomes[0]}-{chromosomes[-1]}" if len(chromosomes) > 1 else f"chromosomes:    {chromosomes[0]}")
    print(f"cohort source:  {cohort_source}")
    print(f"sample count:   {sample_count_str}")
    print(f"n_pcs:          {n_pcs}")
    print(f"random_seed:    {random_seed}")
    print(f"output dir:     {resolved_output_dir}")
    print(f"cache dir:      {cache_dir_path}")
    print("actions that WOULD run:")
    print(f"  - {download_action}")
    print("  - intersect with phenotype table -> ~N samples")
    print("  - variant statistics via bitpacked GPU streaming")
    print("  - marginal pre-screen at |z|>=1.5")
    print("  - tie map collapse")
    print("  - variational EM (max_iter=20)")
    print("  - score test cohort")
    print("  - write artifacts")
    return 0


def _run_doctor(*, cache_dir_arg: str | None) -> int:
    """Print a status table of env / GPU / CuPy / smoke checks.

    Exit 0 if no FAIL entries, else 1. WARN entries never trigger exit 1.
    """

    import contextlib
    import io as _io

    lines: list[tuple[str, str]] = []

    def add(status: str, message: str) -> None:
        lines.append((status, message))

    # --- env vars (warn-only) ---
    for var in ("CDR_STORAGE_PATH", "GOOGLE_PROJECT", "WORKSPACE_BUCKET"):
        val = os.environ.get(var)
        if val:
            add("OK", f"env: {var}={val}")
        else:
            add("WARN", f"env: {var} unset")
    prealloc = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")
    if prealloc == "false":
        add("OK", "env: XLA_PYTHON_CLIENT_PREALLOCATE=false")
    else:
        add("WARN", f"env: XLA_PYTHON_CLIENT_PREALLOCATE={prealloc or 'unset'} (recommend 'false')")

    # --- cache dir (informational + writability check) ---
    if cache_dir_arg:
        cache_path = Path(cache_dir_arg).expanduser()
        if cache_path.exists():
            if os.access(cache_path, os.W_OK):
                add("OK", f"cache_dir: {cache_path} (writable)")
            else:
                add("FAIL", f"cache_dir: {cache_path} exists but is not writable")
        else:
            try:
                cache_path.mkdir(parents=True, exist_ok=True)
                add("OK", f"cache_dir: {cache_path} (created)")
            except OSError as exc:
                add("FAIL", f"cache_dir: {cache_path} cannot be created ({exc})")
    else:
        add("WARN", "cache_dir: not specified (per-run caches will go to default)")

    # --- GPU probe ---
    try:
        from sv_pgs.bitpacked.launch import gpu_arch

        arch = gpu_arch()
        if arch.family != "unknown":
            add("OK", f"GPU: {arch.name or 'GPU'} sm_{arch.sm} family={arch.family}")
        else:
            add("WARN", "GPU: no GPU detected")
    except Exception as exc:  # noqa: BLE001
        add("WARN", f"GPU: probe raised {type(exc).__name__}: {exc}")

    # --- CuPy + NVRTC probe ---
    try:
        from sv_pgs.preflight import _probe_cupy_nvrtc

        nvrtc_err = _probe_cupy_nvrtc()
        if nvrtc_err is None:
            try:
                import cupy as _cp  # type: ignore[import-not-found]

                cupy_ver = getattr(_cp, "__version__", "?")
            except Exception:  # noqa: BLE001
                cupy_ver = "?"
            add("OK", f"CuPy {cupy_ver}, NVRTC compile+launch OK")
        else:
            add("FAIL", f"CuPy/NVRTC: {nvrtc_err}")
    except Exception as exc:  # noqa: BLE001
        add("FAIL", f"CuPy/NVRTC: probe raised {type(exc).__name__}: {exc}")

    # --- smoke check ---
    stderr_buf = _io.StringIO()
    try:
        from sv_pgs.bitpacked.smoke import main as smoke_main

        with contextlib.redirect_stderr(stderr_buf):
            rc = smoke_main()
        if rc == 0:
            add("OK", "smoke: bitpacked end-to-end OK")
        else:
            tail = (stderr_buf.getvalue().strip().splitlines() or [""])[-1]
            add("FAIL", f"smoke: {tail or f'exit={rc}'}")
    except Exception as exc:  # noqa: BLE001
        add("FAIL", f"smoke: {type(exc).__name__}: {exc}")

    n_ok = sum(1 for s, _ in lines if s == "OK")
    n_warn = sum(1 for s, _ in lines if s == "WARN")
    n_fail = sum(1 for s, _ in lines if s == "FAIL")
    for status, message in lines:
        print(f"[{status}] {message}")
    print(f"=== sv-pgs doctor: {n_ok} OK, {n_warn} WARN, {n_fail} FAIL ===")
    return 1 if n_fail > 0 else 0


def _infer_trait_type(targets: Iterable[float]) -> TraitType:
    unique_targets = sorted({float(value) for value in targets})
    if all(target_value in {0.0, 1.0} for target_value in unique_targets):
        return TraitType.BINARY
    return TraitType.QUANTITATIVE


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
