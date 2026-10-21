#!/bin/bash
# Submit one pipeline job to Google Batch, from a terminal inside the All of Us workspace (a Workbench app or VM in the
# workspace project, as the workspace's identity). Never launched by an agent: launching is the user's decision.
#
#   submit.sh store      # steps samples .. direct_sv, on CPUs
#   submit.sh fit        # steps fit .. export, on GPUs
#
# PROVISIONING_MODEL is SPOT or STANDARD for either job. Batch places the job in any zone of REGION with capacity, and has
# no fallback from one model to the other (it uses a job's first instance policy only). A task is retried, up to
# MAX_RETRIES times (0 by default), only when its VM went away under it: a Spot preemption (exit 50001), a VM rebooted or
# recreated during the task (50003, 50006: what host maintenance does to a GPU VM, which cannot live-migrate; Batch
# documents no code of its own for maintenance), or the SIGTERM of a VM shutting down, which run_step.sh's trap exits 75
# on. Any other exit fails the task at once: the task's own failures (1, or the driver's status, such as an OOM kill's
# 137), a VM that stopped reporting or answering (50002, 50004, which a task exhausting its VM's memory can cause) and an
# attempt over its maxRunDuration (50005). Batch takes one lifecycle policy, and RETRY_TASK fails every exit code it does
# not name (the job templates' lifecyclePolicies). A retry resumes from what the mirror holds, so each preemption redoes
# the job's start (environment, restore, digest) and the work since the last durable push. Spot at a fraction p of the
# STANDARD price is the cheaper while preemptions per hour of runtime stay below (1 - p) / (hours each one redoes).
# A store job with TASK_COUNT > 1 (THROUGH=store) is a fan-out: task i converts the run config's chromosome i on its
# own VM (workspace-run --store-shard), and the next job, finding every chromosome in the mirror, writes the store.
#
# Each job reserves one slot in the study's immutable $3,000 envelope before submission.
# Batch limits task runtime; separately billed resources use the plan's non-task reserve.
set -euo pipefail
job="${1:-}"
case "$job" in
  store|fit) spec="$job" ;;
  *) echo "usage: submit.sh store|fit" >&2; exit 2 ;;
esac
here="$(cd "$(dirname "$0")" && pwd)"
: "${PROJECT_ID:?}" "${REGION:?}" "${OPS_BUCKET:?}" "${RUN_ID:?}" "${HOURLY_USD:?}" "${BUDGET_SLOT:?}"
: "${THROUGH:?}"
STUDY_BUDGET="${STUDY_BUDGET:-$here/study_budget.template.json}"
MAX_RETRIES="${MAX_RETRIES:-0}"
TASK_COUNT="${TASK_COUNT:-1}"
if [[ ! "$TASK_COUNT" =~ ^[1-9][0-9]*$ ]] || { [ "$spec" = fit ] && [ "$TASK_COUNT" != 1 ]; }; then
  echo "TASK_COUNT is the store job's number of chromosome tasks, one VM each; the fit job has one" >&2; exit 2
fi
# STORE_SHARDS (optional): the run config's chromosome indices the store job's tasks convert, one per task, such as the
# chromosomes an earlier fan-out did not finish.
STORE_SHARDS="${STORE_SHARDS:-}"
if [ -n "$STORE_SHARDS" ]; then
  read -r -a listed <<< "$STORE_SHARDS"
  if [ "$spec" = fit ] || [[ ! "$STORE_SHARDS" =~ ^[0-9]+( [0-9]+)*$ ]] || [ "${#listed[@]}" != "$TASK_COUNT" ]; then
    echo "STORE_SHARDS lists a store job's chromosome indices, one per task (TASK_COUNT of them)" >&2; exit 2
  fi
fi
# FIT_ARMS (optional): the arms a fit job fits, such as "with_sv", when each arm has its own job.
FIT_ARMS="${FIT_ARMS:-}"
if [ -n "$FIT_ARMS" ] && { [ "$spec" != fit ] || [[ ! "$FIT_ARMS" =~ ^(snv_indel|with_sv)( (snv_indel|with_sv))?$ ]]; }; then
  echo "FIT_ARMS lists a fit job's arms, snv_indel and/or with_sv" >&2; exit 2
fi
# AFTER_SCRIPT (optional): a script of the staged source tree the task runs once the driver has finished, such as
# benchmarks/bench_real/aou_ldpred2.sh (the LDpred2 baseline on the run's own cohort, store and split).
AFTER_SCRIPT="${AFTER_SCRIPT:-}"
if [ -n "$AFTER_SCRIPT" ] && [[ ! "$AFTER_SCRIPT" =~ ^[A-Za-z0-9_][A-Za-z0-9_./-]*\.sh$ || "$AFTER_SCRIPT" == *..* ]]; then
  echo "AFTER_SCRIPT is a .sh path inside the staged source tree" >&2; exit 2
fi
# SHARED_RUNTIME (optional): 1 to take the job's environment prebuilt from the ops bucket when a job has published one
# for the staged uv.lock, and to publish it otherwise (run_step.sh.template, launcher/workspace/runtime_pack.py).
SHARED_RUNTIME="${SHARED_RUNTIME:-}"
[[ "$SHARED_RUNTIME" =~ ^1?$ ]] || { echo "SHARED_RUNTIME is 1 or empty" >&2; exit 2; }
# BASE_RUN (optional): another run of this ops bucket whose mirror carries this run's completed steps before the fit
# (samples, store, phenotypes, cohort, measurement, direct_sv) where this run's mirror holds none of a step, in place of
# a copy of its 0.5-1.5 TB store chain: run-mirror --base (sv_pgs/run_mirror.py), which the staged source must have.
# The base is only read, and must be kept while this run is: this run's mirror names it under run/.bases/ from its first
# push on, and the run's later jobs read it from there, as they do the bases the base names.
BASE_RUN="${BASE_RUN:-}"
if [ -n "$BASE_RUN" ] && { [[ ! "$BASE_RUN" =~ ^[a-z][a-z0-9-]{0,25}$ ]] || [ "$BASE_RUN" = "${RUN_ID:?}" ]; }; then
  echo "BASE_RUN is another run's RUN_ID in the ops bucket" >&2; exit 2
fi
# STAGED_SOURCE (optional): the directory under svpgs/ in the ops bucket whose source/ and env.tar the job builds from,
# such as a round's one staging (launcher/workspace/round.py: _rounds/<round>); empty, the run's own svpgs/$RUN_ID.
STAGED_SOURCE="${STAGED_SOURCE:-}"
if [ -n "$STAGED_SOURCE" ] && [[ ! "$STAGED_SOURCE" =~ ^[a-z_][a-z0-9_-]*(/[a-z0-9][a-z0-9_-]*)*$ ]]; then
  echo "STAGED_SOURCE is a directory under svpgs/ in the ops bucket, such as _rounds/r7" >&2; exit 2
fi
export TASK_COUNT STORE_SHARDS FIT_ARMS AFTER_SCRIPT SHARED_RUNTIME BASE_RUN STAGED_SOURCE
if [[ ! "${RUN_DISK_GB:?}" =~ ^[1-9][0-9]*$ ]]; then
  echo "RUN_DISK_GB is the size in GB of the new disk each job works on (the run's stores, with room to write the next)" >&2; exit 2
fi
# RUN_DISK_TYPE (optional, pd-balanced by default): that disk's type. pd-balanced moves 140 MiB/s plus 0.28 MiB/s per GiB,
# 700 MiB/s at 2000 GiB (1,200 at most on 16 or more vCPUs): the AoU whole-genome restore wrote the 547 GB store to it in
# 17 minutes and the store digest read it back in 12. local-ssd is the VM's local NVMe SSDs, 375 GiB each, RUN_DISK_GB in
# all: a multiple of 375 that the machine type takes (2, 4 or 8 devices on a2-highgpu-2g); four or more read 2,650 MiB/s
# and write 1,400 on a 24-vCPU A2, at a lower price per GiB than pd-balanced. Their data does not outlive the VM, as no
# job's new disk does: a retry or the next job starts from the mirror either way.
RUN_DISK_TYPE="${RUN_DISK_TYPE:-pd-balanced}"
case "$RUN_DISK_TYPE" in
  pd-balanced|pd-ssd) ;;
  local-ssd) (( RUN_DISK_GB % 375 == 0 )) || { echo "a local-ssd RUN_DISK_GB is a multiple of 375, one local SSD each" >&2; exit 2; } ;;
  *) echo "RUN_DISK_TYPE is pd-balanced, pd-ssd or local-ssd" >&2; exit 2 ;;
esac
export RUN_DISK_TYPE
if [[ ! "$RUN_ID" =~ ^[a-z][a-z0-9-]{0,25}$ ]]; then
  echo "RUN_ID must be a lowercase study ID, at most 26 characters; keep it across restarts" >&2; exit 2
fi
MAX_RUN_SECONDS="$(python3 "$here/budget_guard.py" duration --plan "$STUDY_BUDGET" --slot "$BUDGET_SLOT" --hourly-usd "$HOURLY_USD" --retries "$MAX_RETRIES" --tasks "$TASK_COUNT")"
export MAX_RUN_SECONDS MAX_RETRIES
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
# Both jobs mount the workspace's CDR bucket read-only, CDR_STORAGE_PATH (gs://<bucket>/<release>), billed to the
# workspace project: the store job reads the CDR's GATK-SV call set (run config direct_sv) there, and every job keys each
# step by its inputs' contents, the call set's included, before it skips a completed step.
: "${CDR_STORAGE_PATH:?}" "${GOOGLE_PROJECT:?}"
if [[ ! "$CDR_STORAGE_PATH" =~ ^gs://[a-z0-9._-]+(/[A-Za-z0-9._-]+)*/?$ ]]; then
  echo "CDR_STORAGE_PATH must be the workspace's gs:// CDR path" >&2; exit 2
fi
CDR_BUCKET="${CDR_STORAGE_PATH#gs://}"
export CDR_BUCKET="${CDR_BUCKET%/}"
# Both jobs also mount, read-only, the bucket of the imputed batches and their strata sidecar (IMPUTED_BUCKET, at
# /mnt/disks/imputed) and the panel's bucket of bubble.split sites (PANEL_BUCKET, at /mnt/disks/panel): bucket names only.
: "${IMPUTED_BUCKET:?}" "${PANEL_BUCKET:?}"
for bucket in "$IMPUTED_BUCKET" "$PANEL_BUCKET"; do
  [[ "$bucket" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || { echo "IMPUTED_BUCKET and PANEL_BUCKET are bucket names, without gs://" >&2; exit 2; }
done
export IMPUTED_BUCKET PANEL_BUCKET
# The slot's task script, run_step.sh in the slot's launcher directory.
python3 "$here/budget_guard.py" render --template "$here/run_step.sh.template" --output "$stage/run_step.sh" \
  --variables RUN_ID THROUGH WORKSPACE_CDR GOOGLE_PROJECT UV_EXTRAS OPS_BUCKET CDR_BUCKET IMPUTED_BUCKET PANEL_BUCKET TASK_COUNT STORE_SHARDS FIT_ARMS AFTER_SCRIPT SHARED_RUNTIME BASE_RUN \
  STAGED_SOURCE
variables=(OPS_BUCKET RUN_ID MAX_RETRIES MAX_RUN_SECONDS MACHINE_TYPE PROVISIONING_MODEL PROJECT_ID RUN_DISK_GB RUN_DISK_TYPE REGION NETWORK SUBNETWORK SERVICE_ACCOUNT)
if [ "$spec" = fit ]; then variables+=(GPU_TYPE GPU_COUNT); else variables+=(TASK_COUNT); fi
variables+=(CDR_BUCKET IMPUTED_BUCKET PANEL_BUCKET GOOGLE_PROJECT)
python3 "$here/budget_guard.py" render --template "$here/job_${spec}.json.template" --output "$stage/job.json" --variables "${variables[@]}"
# Each submitted job keeps its own task script and Batch log, so another submission cannot change them underneath a
# running job.
python3 - "$stage/job.json" "$BUDGET_SLOT" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
spec = json.loads(path.read_text())
script = spec["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]
script["text"] = script["text"].replace("/launcher/run_step.sh", f"/launcher/{sys.argv[2]}/run_step.sh")
spec["logsPolicy"]["logsPath"] = spec["logsPolicy"]["logsPath"].replace("/launcher/batch.log", f"/launcher/{sys.argv[2]}/batch.log")
path.write_text(json.dumps(spec))
PY
python3 "$here/budget_guard.py" reserve --plan "$STUDY_BUDGET" --slot "$BUDGET_SLOT" --hourly-usd "$HOURLY_USD" --retries "$MAX_RETRIES" \
  --tasks "$TASK_COUNT" --spec "$stage/job.json" --ledger "gs://${OPS_BUCKET}/svpgs/${RUN_ID}/budget"
gcloud storage cp "$stage/run_step.sh" "gs://${OPS_BUCKET}/svpgs/${RUN_ID}/launcher/${BUDGET_SLOT}/run_step.sh" --if-generation-match=0
echo "submitting ${job} in reserved slot ${BUDGET_SLOT}: ${TASK_COUNT} task(s), ${MAX_RUN_SECONDS} seconds per attempt, at most $((MAX_RETRIES + 1)) attempts"
gcloud batch jobs submit "svpgs-${RUN_ID}-${BUDGET_SLOT}" --project="${PROJECT_ID}" --location="${REGION}" --config="$stage/job.json"
