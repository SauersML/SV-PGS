#!/bin/bash
# Submit one pipeline job to Google Batch, from a terminal inside the All of Us workspace (a Workbench app or VM in the
# workspace project, as the workspace's identity). Never launched by an agent: launching is the user's decision.
#
#   submit.sh store      # steps samples .. direct_sv, on CPUs
#   submit.sh fit        # steps fit .. export, on GPUs
#
# PROVISIONING_MODEL is SPOT or STANDARD for either job. Batch places the job in any zone of REGION with capacity.
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
export TASK_COUNT STORE_SHARDS
if [[ ! "${RUN_DISK_GB:?}" =~ ^[1-9][0-9]*$ ]]; then
  echo "RUN_DISK_GB is the size in GB of the new disk each job works on (the run's stores, with room to write the next)" >&2; exit 2
fi
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
  --variables RUN_ID THROUGH WORKSPACE_CDR GOOGLE_PROJECT UV_EXTRAS OPS_BUCKET CDR_BUCKET IMPUTED_BUCKET PANEL_BUCKET TASK_COUNT STORE_SHARDS
variables=(OPS_BUCKET RUN_ID MAX_RETRIES MAX_RUN_SECONDS MACHINE_TYPE PROVISIONING_MODEL PROJECT_ID RUN_DISK_GB REGION NETWORK SUBNETWORK SERVICE_ACCOUNT)
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
