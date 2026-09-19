from __future__ import annotations

import argparse
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

from sv_pgs.all_of_us import (
    AllOfUsDiseaseRequest,
    available_disease_names,
    available_measurement_names,
    prepare_all_of_us_disease_sample_table,
    prepare_all_of_us_measurement_census,
    prepare_all_of_us_measurement_sample_table,
)


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
            "trait (cells under 20 participants suppressed): the first query to run on a new CDR."
        ),
    )
    census_parser.add_argument("--output", required=True, help="Output TSV path for the census.")

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

    if args.command == "list-all-of-us-diseases":
        for disease_name in available_disease_names():
            print(disease_name)
        return 0

    if args.command == "list-all-of-us-traits":
        for trait_name in available_measurement_names():
            print(trait_name)
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
            request=AllOfUsDiseaseRequest(disease=args.disease),
            output_path=Path(args.output),
        )
        print("sample_table\t" + str(prepared_outputs.sample_table_path))
        print("sql\t" + str(prepared_outputs.sql_path))
        print("metadata\t" + str(prepared_outputs.metadata_path))
        return 0

    raise AssertionError(f"unhandled command {args.command!r}")


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


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
