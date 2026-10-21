#!/bin/bash
# The LDpred2-auto baseline on the run a Batch job's driver left (run_step.sh's AFTER_SCRIPT, with RUN and OPS set): R
# with bigsnpr from the packed environment in the ops bucket (svpgs/reference/renv.tar.gz, conda-forge's r-bigsnpr
# packed with conda-pack: the job has no internet), then aou_ldpred2.py, whose per-trait aggregates go to
# $OPS/ldpred2.jsonl and each held-out person's score to $OPS/ldpred2_held_out.npz and, with each model's training
# digest, to $OPS/baselines/ldpred2_auto.npz (a region's under regions/<name>; participant-level: they stay in the
# workspace bucket, beside the run's mirror, which does not carry them: the next job's report reads the latter with the
# run config's baseline_directory set to that directory). Then aou_compare.py compares the arms and LDpred2 on the same
# held-out people ($OPS/compare.tsv, aggregates), where the driver scored the arms. A run of fit_regions has both region
# by region.
set -euo pipefail
rm -rf "$RUN/renv" && mkdir -p "$RUN/renv"
tar -xzf "$OPS/../reference/renv.tar.gz" -C "$RUN/renv"
# conda-unpack's shebang asks for a python on PATH, and the Batch image has none: the packed environment's own runs it
"$RUN/renv/bin/python" "$RUN/renv/bin/conda-unpack"
cd "$RUN/source"
PATH="$RUN/renv/bin:$PATH" "$RUN/venv/bin/python" -m benchmarks.bench_real.aou_ldpred2 --config "$OPS/run_config.json" --out "$OPS/ldpred2.jsonl" \
  --scores "$OPS/ldpred2_held_out.npz" --baselines "$OPS/baselines"
"$RUN/venv/bin/python" -m benchmarks.bench_real.aou_compare --config "$OPS/run_config.json" --scores "$OPS/ldpred2_held_out.npz" --out "$OPS/compare.tsv"
