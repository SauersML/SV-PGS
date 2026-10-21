#!/bin/bash
# Google Batch smoke job from launcher/workspace/stage.sh's source/ and env.tar.
# Stage the revision being checked first: stage.sh archives HEAD, so uncommitted edits are NOT in that source.
# No genotype bucket is needed. The VM copies source locally and rebuilds the locked GPU runtime offline from
# env.tar (bin/uv, python/, uv-cache/), exactly as launcher/workspace/run_step.sh.template does, without pytest.
#
# PROJECT_ID=... REGION=... OPS_BUCKET=... RUN_ID=... SERVICE_ACCOUNT=... \
# NETWORK=... SUBNETWORK=... bash benchmarks/gpu_smoke.sh submit
#
# GPU_COUNT defaults to 2 (a2-highgpu-2g); use GPU_COUNT=1 for a2-highgpu-1g.
# STEPS (default "smoke bench tests") picks the task's commands, run in the order given: smoke is
# benchmarks/gpu_smoke.py, bench is benchmarks/sweep_bench.py (old against new seconds per mean-field sweep on one GPU),
# stage0 is benchmarks/stage0_bench.py (old against new Stage 0 phases on one GPU), tests the CUDA test files.
# GPU_COUNT=1 STEPS=bench measures the sweep alone on one A100, GPU_COUNT=1 STEPS=stage0 Stage 0 alone.
# JOB_ID, MACHINE_TYPE, GPU_TYPE and RUN_DISK_GB may be overridden. JOB_SECONDS includes VM task environment
# preparation; the smoke command itself has a separate 300-second deadline, the benchmark 600 seconds, the Stage 0
# benchmark STAGE0_BENCH_SECONDS (default 900), the CUDA test files PYTEST_SECONDS (default 1200). No task retries hide
# a failure.
# Use "spec" instead of "submit" to print the JSON without submitting. Logs go to Cloud Logging and to
# svpgs/${RUN_ID}/smoke/${JOB_ID}.log in the ops bucket (mounted read-write for it); the task fails where any of its
# commands fails, and FAIL lines identify failed checks (including a missing CUDA device).
set -euo pipefail

smoke_task() {
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
    # The smoke command first, the mean-field sweep's benchmark (its own ten-minute deadline), then every CUDA test
    # file (env.tar's cache holds the dev group, pytest included): the CUDA tests skip on a laptop and on CI, so this
    # job is where they run. Any of them failing fails the task. The Stage 0 benchmark (stage0: Stage 0 and the steps up
    # to the EB, old against new on one GPU, FAIL where a pair disagrees) runs only where STEPS names it.
    local status=0 step
    for step in ${STEPS:-smoke bench tests}; do
        case "$step" in
            smoke) "$RUN/venv/bin/python" -m benchmarks.gpu_smoke || status=$? ;;
            bench) "$RUN/venv/bin/python" -m benchmarks.sweep_bench || status=$? ;;
            stage0) timeout "${STAGE0_BENCH_SECONDS:-900}" "$RUN/venv/bin/python" -m benchmarks.stage0_bench || status=$? ;;
            tests) timeout "${PYTEST_SECONDS:-1200}" "$RUN/venv/bin/python" -m pytest -q -p no:cacheprovider tests/*_cuda.py \
                     tests/test_scale_mixture_ep.py -k "cuda or device or chunk" || status=$? ;;
        esac
    done
    return "$status"
}

if [[ "${1:-}" == task ]]; then
    : "${OPS:?staged source/ and env.tar directory}" "${RUN:?task-local writable directory}" "${JOB:?job id}"
    # The workspace cannot read Cloud Logging (PERMISSION_DENIED for its service account), so the task's output also
    # goes to smoke/<job>.log beside the staged source: copied every minute while it runs and once at the end.
    mkdir -p "$RUN" "$OPS/smoke"
    log="$OPS/smoke/$JOB.log"
    ( while sleep 60; do cp "$RUN/smoke.log" "$log" 2>/dev/null; done ) &
    copier=$!
    status=0
    smoke_task 2>&1 | tee "$RUN/smoke.log" || status=${PIPESTATUS[0]}
    kill "$copier" 2>/dev/null
    echo "smoke task exit $status" >> "$RUN/smoke.log"
    cp "$RUN/smoke.log" "$log"
    exit "$status"
fi

case "${1:-}" in
    spec|submit) ;;
    *) echo "usage: $0 {spec|submit}; see this script's header for staging and variables" >&2; exit 2 ;;
esac
: "${PROJECT_ID:?}" "${REGION:?}" "${OPS_BUCKET:?bucket name, without gs://}" "${RUN_ID:?staged run id}"
: "${SERVICE_ACCOUNT:?}" "${NETWORK:?}" "${SUBNETWORK:?}"
export PROJECT_ID REGION OPS_BUCKET RUN_ID SERVICE_ACCOUNT NETWORK SUBNETWORK
export GPU_COUNT="${GPU_COUNT:-2}"
case "$GPU_COUNT" in 1|2) ;; *) echo 'GPU_COUNT must be 1 or 2' >&2; exit 2 ;; esac
export STEPS="${STEPS:-smoke bench tests}"
for step in $STEPS; do
    case "$step" in smoke|bench|stage0|tests) ;; *) echo "STEPS takes smoke, bench, stage0 and tests, not $step" >&2; exit 2 ;; esac
done
export MACHINE_TYPE="${MACHINE_TYPE:-a2-highgpu-${GPU_COUNT}g}" GPU_TYPE="${GPU_TYPE:-nvidia-tesla-a100}"
export RUN_DISK_GB="${RUN_DISK_GB:-100}" JOB_SECONDS="${JOB_SECONDS:-3600}"
export JOB_ID="${JOB_ID:-svpgs-gpu-smoke-$(date +%Y%m%d%H%M%S)}"
spec=$(mktemp)
trap 'rm -f "$spec"' EXIT
python3 - > "$spec" <<'PY'
import json
import os

e = os.environ
ops = f"/mnt/disks/ops/svpgs/{e['RUN_ID']}"
# Task values are structured environment entries: no shell expansion of project, bucket or run names into code.
task = {
    "runnables": [{"script": {"text": 'bash "$OPS/source/benchmarks/gpu_smoke.sh" task'}}],
    "environment": {"variables": {"OPS": ops, "RUN": "/mnt/disks/run", "JOB": e["JOB_ID"], "STEPS": e["STEPS"]}},
    "maxRetryCount": 0,
    "maxRunDuration": f"{int(e['JOB_SECONDS'])}s",
    "computeResource": {"cpuMilli": 2000, "memoryMib": 8192},
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
