#!/bin/bash
# Google Batch job for benchmarks/eb_sweeps.py: one synthetic EB fit on one A100, its sweeps by phase written as JSON.
# Stage the revision being measured first (launcher/workspace/stage.sh archives HEAD, so uncommitted edits are NOT in
# that source); for a before/after pair, stage both revisions (copy benchmarks/eb_sweeps.py and this file into the
# older one) and submit once per RUN_ID with the same sizes. No genotype bucket is needed. The VM rebuilds the locked
# GPU runtime offline from env.tar exactly as benchmarks/gpu_smoke.sh does.
#
# PROJECT_ID=... REGION=... OPS_BUCKET=... RUN_ID=... SERVICE_ACCOUNT=... \
# NETWORK=... SUBNETWORK=... LABEL=before bash benchmarks/eb_sweeps.sh submit
#
# GPU_COUNT defaults to 1 (a2-highgpu-1g). JOB_ID, MACHINE_TYPE, GPU_TYPE, RUN_DISK_GB, LABEL and EB_SWEEPS_ARGS (extra
# arguments for "run", e.g. "--samples 20000") may be overridden. The Python command has its own deadline (600 s by
# default, generation included); JOB_SECONDS bounds the whole task. No task retries hide a failure.
# Use "spec" instead of "submit" to print the JSON without submitting. The report and the log go to
# svpgs/${RUN_ID}/eb_sweeps/${JOB_ID}.json and .log in the ops bucket; compare two reports with
#   python -m benchmarks.eb_sweeps compare before.json after.json
set -euo pipefail

sweeps_task() {
    mkdir -p "$RUN/tools" "$RUN/tmp"
    tar -xf "$OPS/env.tar" -C "$RUN"
    mv "$RUN/bin/uv" "$RUN/tools/uv"
    chmod +x "$RUN/tools/uv" "$RUN"/python/*/bin/*
    cp -r "$OPS/source" "$RUN/source"
    UV_OFFLINE=1 UV_CACHE_DIR="$RUN/uv-cache" UV_PYTHON_INSTALL_DIR="$RUN/python" \
      UV_PROJECT_ENVIRONMENT="$RUN/venv" UV_LINK_MODE=copy \
      "$RUN/tools/uv" sync --project "$RUN/source" --frozen --python 3.12 --extra gpu
    export LD_LIBRARY_PATH="$(ls -d "$RUN"/venv/lib/python3.12/site-packages/nvidia/*/lib | paste -sd:)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export TMPDIR="$RUN/tmp" PYTHONUNBUFFERED=1
    cd "$RUN/source"
    # shellcheck disable=SC2086 # EB_SWEEPS_ARGS is a list of arguments
    "$RUN/venv/bin/python" -m benchmarks.eb_sweeps run --out "$RUN/eb_sweeps.json" --label "${LABEL:-run}" ${EB_SWEEPS_ARGS:-}
}

if [[ "${1:-}" == task ]]; then
    : "${OPS:?staged source/ and env.tar directory}" "${RUN:?task-local writable directory}" "${JOB:?job id}"
    # The workspace cannot read Cloud Logging, so the log is copied beside the staged source every minute and at the end.
    mkdir -p "$RUN" "$OPS/eb_sweeps"
    log="$OPS/eb_sweeps/$JOB.log"
    ( while sleep 60; do cp "$RUN/eb_sweeps.log" "$log" 2>/dev/null; done ) &
    copier=$!
    status=0
    sweeps_task 2>&1 | tee "$RUN/eb_sweeps.log" || status=${PIPESTATUS[0]}
    kill "$copier" 2>/dev/null
    echo "eb_sweeps task exit $status" >> "$RUN/eb_sweeps.log"
    cp "$RUN/eb_sweeps.log" "$log"
    if [[ -f "$RUN/eb_sweeps.json" ]]; then cp "$RUN/eb_sweeps.json" "$OPS/eb_sweeps/$JOB.json"; fi
    exit "$status"
fi

case "${1:-}" in
    spec|submit) ;;
    *) echo "usage: $0 {spec|submit}; see this script's header for staging and variables" >&2; exit 2 ;;
esac
: "${PROJECT_ID:?}" "${REGION:?}" "${OPS_BUCKET:?bucket name, without gs://}" "${RUN_ID:?staged run id}"
: "${SERVICE_ACCOUNT:?}" "${NETWORK:?}" "${SUBNETWORK:?}"
export PROJECT_ID REGION OPS_BUCKET RUN_ID SERVICE_ACCOUNT NETWORK SUBNETWORK
export GPU_COUNT="${GPU_COUNT:-1}"
case "$GPU_COUNT" in 1|2) ;; *) echo 'GPU_COUNT must be 1 or 2' >&2; exit 2 ;; esac
export MACHINE_TYPE="${MACHINE_TYPE:-a2-highgpu-${GPU_COUNT}g}" GPU_TYPE="${GPU_TYPE:-nvidia-tesla-a100}"
export RUN_DISK_GB="${RUN_DISK_GB:-100}" JOB_SECONDS="${JOB_SECONDS:-2400}" LABEL="${LABEL:-run}" EB_SWEEPS_ARGS="${EB_SWEEPS_ARGS:-}"
export JOB_ID="${JOB_ID:-svpgs-eb-sweeps-$(date +%Y%m%d%H%M%S)}"
spec=$(mktemp)
trap 'rm -f "$spec"' EXIT
python3 - > "$spec" <<'PY'
import json
import os

e = os.environ
ops = f"/mnt/disks/ops/svpgs/{e['RUN_ID']}"
# Task values are structured environment entries: no shell expansion of project, bucket or run names into code.
task = {
    "runnables": [{"script": {"text": 'bash "$OPS/source/benchmarks/eb_sweeps.sh" task'}}],
    "environment": {"variables": {"OPS": ops, "RUN": "/mnt/disks/run", "JOB": e["JOB_ID"], "LABEL": e["LABEL"], "EB_SWEEPS_ARGS": e["EB_SWEEPS_ARGS"]}},
    "maxRetryCount": 0,
    "maxRunDuration": f"{int(e['JOB_SECONDS'])}s",
    "computeResource": {"cpuMilli": 8000, "memoryMib": 65536},
    "volumes": [
        {"gcs": {"remotePath": e["OPS_BUCKET"]}, "mountPath": "/mnt/disks/ops", "mountOptions": ["--implicit-dirs"]},
        {"deviceName": "run", "mountPath": "/mnt/disks/run", "mountOptions": "rw,async"},
    ],
}
spec = {
    "taskGroups": [{"taskCount": 1, "taskSpec": task}],
    "allocationPolicy": {
        "instances": [{"installGpuDrivers": True, "policy": {
            "machineType": e["MACHINE_TYPE"], "provisioningModel": "STANDARD",
            "accelerators": [{"type": e["GPU_TYPE"], "count": int(e["GPU_COUNT"])}],
            "disks": [{"deviceName": "run", "newDisk": {"type": "pd-balanced", "sizeGb": int(e["RUN_DISK_GB"])}}],
        }}],
        "location": {"allowedLocations": [f"regions/{e['REGION']}"]},
        "network": {"networkInterfaces": [{
            "network": f"projects/{e['PROJECT_ID']}/global/networks/{e['NETWORK']}",
            "subnetwork": f"projects/{e['PROJECT_ID']}/regions/{e['REGION']}/subnetworks/{e['SUBNETWORK']}",
            "noExternalIpAddress": True,
        }]},
        "serviceAccount": {"email": e["SERVICE_ACCOUNT"]},
    },
    "logsPolicy": {"destination": "CLOUD_LOGGING"},
}
print(json.dumps(spec, indent=2))
PY
if [[ "$1" == spec ]]; then
    cat "$spec"
else
    gcloud batch jobs submit "$JOB_ID" --project="$PROJECT_ID" --location="$REGION" --config="$spec"
fi
