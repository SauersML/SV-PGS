#!/bin/bash
# The fast standard PGS methods of aou_fastpgs.py, each fitted on the snv_indel and the with_sv records, on the run a
# Batch job's driver left (run_step.sh's AFTER_SCRIPT, with RUN and OPS set). R with bigsnpr comes from the packed
# environment in the ops bucket, unpacked as aou_ldpred2.sh unpacks it. Per trait, arm and method the aggregates go to
# $OPS/fastpgs.jsonl, each held-out person's scores to $OPS/fastpgs/<trait>.npz and each trait's fitted effects to
# $OPS/fastpgs-units. The scores are participant-level: they stay in the workspace bucket, where fastpgs_meta.py reads them.
set -euo pipefail
rm -rf "$RUN/renv" && mkdir -p "$RUN/renv"
tar -xzf "$OPS/../reference/renv.tar.gz" -C "$RUN/renv"
# conda-unpack's shebang asks for a python on PATH, and the Batch image has none: the packed environment's own runs it
"$RUN/renv/bin/python" "$RUN/renv/bin/conda-unpack"
cd "$RUN/source"
PATH="$RUN/renv/bin:$PATH" "$RUN/venv/bin/python" -m benchmarks.bench_real.aou_fastpgs --config "$OPS/run_config.json" --out "$OPS/fastpgs.jsonl" \
  --scores "$OPS/fastpgs" --units "$OPS/fastpgs-units"
