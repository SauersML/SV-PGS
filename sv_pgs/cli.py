from __future__ import annotations

import argparse
from pathlib import Path

from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.io import load_dataset_from_files, run_training_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sv-pgs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Load genotype files, fit the model, and write outputs.")
    run_parser.add_argument("--genotypes", required=True, help="Path to a VCF/BCF file or PLINK 1 .bed file.")
    run_parser.add_argument(
        "--genotype-format",
        default="auto",
        choices=("auto", "vcf", "plink1"),
        help="Input genotype format. Default infers from the path.",
    )
    run_parser.add_argument("--sample-table", required=True, help="CSV or TSV with sample_id, target, and covariates.")
    run_parser.add_argument("--sample-id-column", default="sample_id", help="Sample identifier column in the sample table.")
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
    run_parser.add_argument("--minimum-structural-variant-carriers", type=int, default=5)
    run_parser.add_argument("--random-seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "run":
        raise ValueError("Unsupported command: " + str(args.command))

    dataset = load_dataset_from_files(
        genotype_path=args.genotypes,
        genotype_format=args.genotype_format,
        sample_table_path=args.sample_table,
        sample_id_column=args.sample_id_column,
        target_column=args.target_column,
        covariate_columns=args.covariate_column,
        variant_metadata_path=args.variant_metadata,
    )
    outputs = run_training_pipeline(
        dataset=dataset,
        config=ModelConfig(
            trait_type=_infer_trait_type(dataset.targets),
            max_outer_iterations=args.max_outer_iterations,
            minimum_structural_variant_carriers=args.minimum_structural_variant_carriers,
            random_seed=args.random_seed,
        ),
        output_dir=Path(args.output_dir),
    )

    print("artifact_dir\t" + str(outputs.artifact_dir))
    print("summary\t" + str(outputs.summary_path))
    print("predictions\t" + str(outputs.predictions_path))
    print("coefficients\t" + str(outputs.coefficients_path))
    return 0


def _infer_trait_type(targets) -> TraitType:
    unique_targets = sorted({float(value) for value in targets})
    if all(target_value in {0.0, 1.0} for target_value in unique_targets):
        return TraitType.BINARY
    return TraitType.QUANTITATIVE


if __name__ == "__main__":
    raise SystemExit(main())
