from __future__ import annotations

import argparse
import faulthandler
import io
import os
import platform
import sys
from pathlib import Path

from sv_pgs._jax import require_full_gpu_runtime
from sv_pgs.aou_runner import run_all_of_us
from sv_pgs.all_of_us import AllOfUsDiseaseRequest, available_disease_names, prepare_all_of_us_disease_sample_table
from sv_pgs.config import InferenceBackend, ModelConfig, TraitType
from sv_pgs.io import load_dataset_from_files, run_training_pipeline
from sv_pgs.progress import gpu_memory_snapshot, jax_runtime_snapshot, log, nvidia_smi_snapshot


def _add_basil_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = ModelConfig()
    parser.add_argument("--basil-l1-ratio", type=float, default=defaults.basil_l1_ratio)
    parser.add_argument("--basil-lambda-min-ratio", type=float, default=defaults.basil_lambda_min_ratio)
    parser.add_argument("--basil-n-lambdas", type=int, default=defaults.basil_n_lambdas)
    parser.add_argument("--basil-strong-set-initial-size", type=int, default=defaults.basil_strong_set_initial_size)
    parser.add_argument("--basil-strong-set-growth", type=int, default=defaults.basil_strong_set_growth)
    parser.add_argument("--basil-batch-size", type=int, default=defaults.basil_batch_size)
    parser.add_argument("--basil-max-screening-passes", type=int, default=defaults.basil_max_screening_passes)
    parser.add_argument("--basil-irls-max-iterations", type=int, default=defaults.basil_irls_max_iterations)
    parser.add_argument(
        "--basil-coordinate-descent-max-epochs",
        type=int,
        default=defaults.basil_coordinate_descent_max_epochs,
    )
    parser.add_argument(
        "--basil-coordinate-descent-tolerance",
        type=float,
        default=defaults.basil_coordinate_descent_tolerance,
    )
    parser.add_argument("--basil-kkt-tolerance", type=float, default=defaults.basil_kkt_tolerance)


def _basil_config_from_args(args: argparse.Namespace) -> dict[str, int | float]:
    return {
        "basil_l1_ratio": float(args.basil_l1_ratio),
        "basil_lambda_min_ratio": float(args.basil_lambda_min_ratio),
        "basil_n_lambdas": int(args.basil_n_lambdas),
        "basil_strong_set_initial_size": int(args.basil_strong_set_initial_size),
        "basil_strong_set_growth": int(args.basil_strong_set_growth),
        "basil_batch_size": int(args.basil_batch_size),
        "basil_max_screening_passes": int(args.basil_max_screening_passes),
        "basil_irls_max_iterations": int(args.basil_irls_max_iterations),
        "basil_coordinate_descent_max_epochs": int(args.basil_coordinate_descent_max_epochs),
        "basil_coordinate_descent_tolerance": float(args.basil_coordinate_descent_tolerance),
        "basil_kkt_tolerance": float(args.basil_kkt_tolerance),
    }


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
        help="Full AoU pipeline: download VCFs, prepare phenotype, merge PCs, and fit one unified genome-wide model.",
    )
    aou_run_parser.add_argument("--disease", required=True, help="Disease name (e.g. hypertension, type2_diabetes).")
    aou_run_parser.add_argument("--chromosomes", default="1-22", help="Chromosome range (default: 1-22).")
    aou_run_parser.add_argument("--output-dir", required=True, help="Base output directory.")
    aou_run_parser.add_argument("--n-pcs", type=int, default=10, help="Number of genomic PCs to include (default: 10).")
    aou_run_parser.add_argument("--max-outer-iterations", type=int, default=30)
    aou_run_parser.add_argument("--random-seed", type=int, default=0)
    aou_run_parser.add_argument(
        "--inference-backend",
        default=InferenceBackend.BASIL.value,
        choices=tuple(backend.value for backend in InferenceBackend),
        help="Inference backend to use for fitting.",
    )
    _add_basil_arguments(aou_run_parser)

    run_parser = subparsers.add_parser("run", help="Load genotype files, fit the model, and write outputs.")
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
        help="Optional CSV or TSV keyed by variant_id with VariantRecord fields.",
    )
    run_parser.add_argument("--output-dir", required=True, help="Directory for artifact and result tables.")
    run_parser.add_argument("--max-outer-iterations", type=int, default=30)
    run_parser.add_argument("--random-seed", type=int, default=0)
    run_parser.add_argument(
        "--inference-backend",
        default=InferenceBackend.BASIL.value,
        choices=tuple(backend.value for backend in InferenceBackend),
        help="Inference backend to use for fitting.",
    )
    _add_basil_arguments(run_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except io.UnsupportedOperation:
        pass  # stderr has no fileno (e.g. pytest capture)
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-all-of-us-diseases":
        for disease_name in available_disease_names():
            print(disease_name)
        return 0

    if args.command == "prepare-all-of-us-disease":
        prepared_outputs = prepare_all_of_us_disease_sample_table(
            request=AllOfUsDiseaseRequest(
                disease=args.disease,
            ),
            output_path=Path(args.output),
        )
        print("sample_table\t" + str(prepared_outputs.sample_table_path))
        print("sql\t" + str(prepared_outputs.sql_path))
        print("metadata\t" + str(prepared_outputs.metadata_path))
        return 0

    if args.command == "run-all-of-us":
        # Parse chromosome range: "1-22" -> [1,2,...,22], "22" -> [22], "1,5,22" -> [1,5,22]
        chr_str = args.chromosomes
        if "-" in chr_str:
            lo, hi = chr_str.split("-", 1)
            chromosomes = list(range(int(lo), int(hi) + 1))
        elif "," in chr_str:
            chromosomes = [int(c.strip()) for c in chr_str.split(",")]
        else:
            chromosomes = [int(chr_str)]
        run_all_of_us(
            disease=args.disease,
            chromosomes=chromosomes,
            output_base=args.output_dir,
            n_pcs=args.n_pcs,
            max_outer_iterations=args.max_outer_iterations,
            random_seed=args.random_seed,
            inference_backend=InferenceBackend(args.inference_backend),
            **_basil_config_from_args(args),
        )
        return 0

    if args.command != "run":
        raise ValueError("Unsupported command: " + str(args.command))

    try:
        with open("/proc/meminfo") as f:
            for line in f:
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
    requested_backend = InferenceBackend(args.inference_backend)
    if requested_backend == InferenceBackend.VARIATIONAL_BAYES:
        require_full_gpu_runtime()
    else:
        log(f"skipping GPU-runtime enforcement for backend={requested_backend.value}")
    log(f"genotypes={args.genotypes} sample_table={args.sample_table} output_dir={args.output_dir}")
    log(f"genotype_format={args.genotype_format} sample_id_column={args.sample_id_column} target_column={args.target_column}")
    log(f"covariates={list(args.covariate_column)}  max_outer_iter={args.max_outer_iterations}  seed={args.random_seed}")
    config = ModelConfig(
        max_outer_iterations=args.max_outer_iterations,
        random_seed=args.random_seed,
        inference_backend=requested_backend,
        **_basil_config_from_args(args),
    )
    if requested_backend == InferenceBackend.BASIL:
        log(
            "basil controls: "
            + f"l1_ratio={config.basil_l1_ratio} n_lambdas={config.basil_n_lambdas} "
            + f"lambda_min_ratio={config.basil_lambda_min_ratio} strong_init={config.basil_strong_set_initial_size} "
            + f"strong_growth={config.basil_strong_set_growth} batch={config.basil_batch_size}"
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


def _infer_trait_type(targets) -> TraitType:
    unique_targets = sorted({float(value) for value in targets})
    if all(target_value in {0.0, 1.0} for target_value in unique_targets):
        return TraitType.BINARY
    return TraitType.QUANTITATIVE


if __name__ == "__main__":
    raise SystemExit(main())
