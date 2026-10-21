#!/bin/bash
# A synthetic-only Batch canary from the same staged source/ and env.tar as gpu_smoke.sh.
# This file is a launcher to review and run later; generating a spec does not contact a service.
# Stage the revision first: stage.sh archives HEAD, so an uncommitted canary is not in that archive.
#
# PROJECT_ID=... REGION=... OPS_BUCKET=... RUN_ID=... SERVICE_ACCOUNT=... \
# NETWORK=... SUBNETWORK=... bash benchmarks/wg_canary.sh spec
# Replace spec with submit only when ready to launch. GPU_COUNT=0 (default) selects n2d-standard-32 Spot;
# GPU_COUNT=1 selects one A100; GPU_COUNT=2 exercises sharded reads on two A100s. No retry hides a failure.
# Only logs, profiles and aggregate benchmark metrics are copied to ops; generated inputs and pipeline tables stay
# on the task disk. The runner has no remote input and refuses network connections. The shell alone stages the
# offline runtime and copies diagnostics to the mounted ops directory, as gpu_smoke.sh does.
#
# COLUMNS=1060000 SAMPLES=512 CHROMOSOMES=22 TRAITS=total_bilirubin are the default scenario. TRAITS is a space-separated
# list of canonical quantitative trait names; DISEASES optionally adds canonical Bernoulli traits. RECORDS defaults to COLUMNS; RECORDS=126700000 covers the full pre-filter
# record axis too and needs a larger disk and deadline. No cheap-runtime claim is made for that configuration.
# HOST_GIB / DEVICE_GIB override the documented mixed column/sample budget. DEADLINE_SECONDS bounds computation;
# JOB_SECONDS includes offline environment preparation. All wall-clock timestamps use America/Los_Angeles.
set -euo pipefail
export TZ=America/Los_Angeles

canary_task() {
    mkdir -p "$RUN/tools" "$RUN/tmp"
    tar -xf "$OPS/env.tar" -C "$RUN"
    mv "$RUN/bin/uv" "$RUN/tools/uv"
    chmod +x "$RUN/tools/uv" "$RUN"/python/*/bin/*
    cp -r "$OPS/source" "$RUN/source"
    # A round's staging (RUN_ID=_rounds/<round>, launcher/workspace/round.py) lists its files' digests: check the copy.
    if [[ -f "$OPS/source.sha256" ]]; then (cd "$RUN/source" && sha256sum --quiet --strict -c "$OPS/source.sha256"); fi
    local extra=()
    if (( GPU_COUNT > 0 )); then extra=(--extra gpu); fi
    UV_OFFLINE=1 UV_CACHE_DIR="$RUN/uv-cache" UV_PYTHON_INSTALL_DIR="$RUN/python" \
      UV_PROJECT_ENVIRONMENT="$RUN/venv" UV_LINK_MODE=copy \
      "$RUN/tools/uv" sync --project "$RUN/source" --frozen --python 3.12 "${extra[@]}"
    if (( GPU_COUNT > 0 )); then
        export LD_LIBRARY_PATH="$(ls -d "$RUN"/venv/lib/python3.12/site-packages/nvidia/*/lib | paste -sd:)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    export TMPDIR="$RUN/tmp" PYTHONUNBUFFERED=1
    cd "$RUN/source"
    local args=(--out "$RUN/canary" --columns "$COLUMNS" --records "$RECORDS" --samples "$SAMPLES"
                --chromosomes "$CHROMOSOMES" --seed "$SEED" --deadline-seconds "$DEADLINE_SECONDS")
    if (( GPU_COUNT > 0 )); then args+=(--device cuda --gpus "$GPU_COUNT"); else args+=(--device cpu); fi
    if [[ -n "${HOST_GIB:-}" ]]; then args+=(--host-gib "$HOST_GIB"); fi
    if [[ -n "${DEVICE_GIB:-}" ]]; then args+=(--device-gib "$DEVICE_GIB"); fi
    local traits=() diseases=()
    read -r -a traits <<< "$TRAITS"
    read -r -a diseases <<< "$DISEASES"
    "$RUN/venv/bin/python" -m benchmarks.wg_canary "${args[@]}" --traits "${traits[@]}" --diseases "${diseases[@]}"
}

copy_diagnostics() {
    local file
    for file in "$RUN/task.log" "$RUN/canary/progress.log" "$RUN/canary/profile.txt" "$RUN/canary/result.json" "$RUN/canary/"*.metrics.json; do
        if [[ -f "$file" ]]; then cp "$file" "$OPS/canary/$JOB/$(basename "$file")"; fi
    done
}

if [[ "${1:-}" == task ]]; then
    : "${OPS:?}" "${RUN:?}" "${JOB:?}" "${GPU_COUNT:?}"
    mkdir -p "$RUN" "$OPS/canary/$JOB"
    ( while sleep 60; do copy_diagnostics || true; done ) &
    copier=$!
    trap 'kill "$copier" 2>/dev/null || true; copy_diagnostics || true' EXIT
    status=0
    canary_task 2>&1 | tee "$RUN/task.log" || status=${PIPESTATUS[0]}
    echo "canary task exit $status" | tee -a "$RUN/task.log"
    exit "$status"
fi

case "${1:-}" in spec|submit) ;; *) echo "usage: $0 {spec|submit}" >&2; exit 2 ;; esac
: "${PROJECT_ID:?}" "${REGION:?}" "${OPS_BUCKET:?bucket name without scheme}" "${RUN_ID:?staged run id}"
: "${SERVICE_ACCOUNT:?}" "${NETWORK:?}" "${SUBNETWORK:?}"
export PROJECT_ID REGION OPS_BUCKET RUN_ID SERVICE_ACCOUNT NETWORK SUBNETWORK
export GPU_COUNT="${GPU_COUNT:-0}"
case "$GPU_COUNT" in 0|1|2) ;; *) echo 'GPU_COUNT must be 0, 1 or 2' >&2; exit 2 ;; esac
if (( GPU_COUNT == 0 )); then
    export MACHINE_TYPE="${MACHINE_TYPE:-n2d-standard-32}" DEADLINE_SECONDS="${DEADLINE_SECONDS:-28800}"
else
    export MACHINE_TYPE="${MACHINE_TYPE:-a2-highgpu-${GPU_COUNT}g}" DEADLINE_SECONDS="${DEADLINE_SECONDS:-10800}"
fi
export JOB_SECONDS="${JOB_SECONDS:-$((DEADLINE_SECONDS + 1800))}" RUN_DISK_GB="${RUN_DISK_GB:-200}"
export JOB_ID="${JOB_ID:-svpgs-wg-canary-$(date +%Y%m%d%H%M%S)}"
export COLUMNS="${COLUMNS:-1060000}" RECORDS="${RECORDS:-${COLUMNS:-1060000}}" SAMPLES="${SAMPLES:-512}"
export CHROMOSOMES="${CHROMOSOMES:-22}" SEED="${SEED:-20261020}" TRAITS="${TRAITS:-total_bilirubin}"
export DISEASES="${DISEASES:-}"
export HOST_GIB="${HOST_GIB:-}" DEVICE_GIB="${DEVICE_GIB:-}"
spec=$(mktemp)
trap 'rm -f "$spec"' EXIT
python3 - > "$spec" <<'PY'
import json
import os

e = os.environ
gpus = int(e["GPU_COUNT"])
variables = {name: e[name] for name in (
    "GPU_COUNT", "COLUMNS", "RECORDS", "SAMPLES", "CHROMOSOMES", "SEED", "TRAITS", "DISEASES", "HOST_GIB", "DEVICE_GIB", "DEADLINE_SECONDS",
)}
variables.update(OPS=f"/mnt/disks/ops/svpgs/{e['RUN_ID']}", RUN="/mnt/disks/run", JOB=e["JOB_ID"], TZ="America/Los_Angeles")
policy = {
    "machineType": e["MACHINE_TYPE"], "provisioningModel": "SPOT",
    "disks": [{"deviceName": "run", "newDisk": {"type": "pd-balanced", "sizeGb": int(e["RUN_DISK_GB"])}}],
}
if gpus:
    policy["accelerators"] = [{"type": "nvidia-tesla-a100", "count": gpus}]
print(json.dumps({
    "taskGroups": [{"taskCount": 1, "taskSpec": {
        "runnables": [{"script": {"text": 'bash "$OPS/source/benchmarks/wg_canary.sh" task'}}],
        "environment": {"variables": variables}, "maxRetryCount": 0, "maxRunDuration": f"{int(e['JOB_SECONDS'])}s",
        "computeResource": {"cpuMilli": 12000 * gpus if gpus else 32000, "memoryMib": 65536 * gpus if gpus else 98304},
        "volumes": [
            {"gcs": {"remotePath": e["OPS_BUCKET"]}, "mountPath": "/mnt/disks/ops", "mountOptions": ["--implicit-dirs"]},
            {"deviceName": "run", "mountPath": "/mnt/disks/run", "mountOptions": "rw,async"},
        ],
    }}],
    "allocationPolicy": {
        "instances": [{"installGpuDrivers": bool(gpus), "policy": policy}],
        "location": {"allowedLocations": [f"regions/{e['REGION']}"]},
        "network": {"networkInterfaces": [{
            "network": f"projects/{e['PROJECT_ID']}/global/networks/{e['NETWORK']}",
            "subnetwork": f"projects/{e['PROJECT_ID']}/regions/{e['REGION']}/subnetworks/{e['SUBNETWORK']}",
            "noExternalIpAddress": True,
        }]},
        "serviceAccount": {"email": e["SERVICE_ACCOUNT"]},
    },
    "logsPolicy": {"destination": "CLOUD_LOGGING"},
}, indent=2))
PY
if [[ "$1" == spec ]]; then
    cat "$spec"
else
    gcloud batch jobs submit "$JOB_ID" --project="$PROJECT_ID" --location="$REGION" --config="$spec"
fi
