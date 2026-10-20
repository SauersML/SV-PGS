#!/bin/bash
# The LDpred2-auto baseline on the run a Batch job's driver left (run_step.sh's AFTER_SCRIPT, with RUN and OPS set): R
# with bigsnpr from the packed environment in the ops bucket (svpgs/reference/renv.tar.gz, conda-forge's r-bigsnpr
# packed with conda-pack: the job has no internet), then aou_ldpred2.py, whose per-trait aggregates go to
# $OPS/ldpred2.jsonl.
set -euo pipefail
rm -rf "$RUN/renv" && mkdir -p "$RUN/renv"
tar -xzf "$OPS/../reference/renv.tar.gz" -C "$RUN/renv"
# conda-unpack's shebang asks for a python on PATH, and the Batch image has none: the packed environment's own runs it
"$RUN/renv/bin/python" "$RUN/renv/bin/conda-unpack"
cd "$RUN/source"
PATH="$RUN/renv/bin:$PATH" "$RUN/venv/bin/python" -m benchmarks.bench_real.aou_ldpred2 --config "$OPS/run_config.json" --out "$OPS/ldpred2.jsonl"
