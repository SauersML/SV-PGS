#!/bin/bash
# Submit one pipeline job to Google Batch, from a terminal inside the All of Us workspace (a Workbench app or VM in the
# workspace project, as the workspace's identity). Never launched by an agent: launching is the user's decision.
#
#   submit.sh store   # steps samples .. measurement, on spot CPUs
#   submit.sh fit     # steps fit .. export, on GPUs
#
# Each job reserves one slot in the study's immutable $3,000 envelope before submission.
# Batch limits task runtime; separately billed resources use the plan's non-task reserve.
set -euo pipefail
job="${1:-}"
case "$job" in
  store|fit) ;;
  *) echo "usage: submit.sh store|fit" >&2; exit 2 ;;
esac
here="$(cd "$(dirname "$0")" && pwd)"
: "${PROJECT_ID:?}" "${REGION:?}" "${OPS_BUCKET:?}" "${RUN_ID:?}" "${HOURLY_USD:?}" "${THROUGH:?}" "${BUDGET_SLOT:?}"
STUDY_BUDGET="${STUDY_BUDGET:-$here/study_budget.template.json}"
MAX_RETRIES="${MAX_RETRIES:-0}"
if [[ ! "${RUN_DISK_GB:?}" =~ ^[1-9][0-9]*$ ]]; then
  echo "RUN_DISK_GB is the size in GB of the new disk each job works on (the run's stores, with room to write the next)" >&2; exit 2
fi
if [[ ! "$RUN_ID" =~ ^[a-z][a-z0-9-]{0,25}$ ]]; then
  echo "RUN_ID must be a lowercase study ID, at most 26 characters; keep it across restarts" >&2; exit 2
fi
MAX_RUN_SECONDS="$(python3 "$here/budget_guard.py" duration --plan "$STUDY_BUDGET" --slot "$BUDGET_SLOT" --hourly-usd "$HOURLY_USD" --retries "$MAX_RETRIES")"
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
python3 "$here/budget_guard.py" render --template "$here/run_step.sh.template" --output "$stage/run_step.sh" \
  --variables RUN_ID THROUGH WORKSPACE_CDR GOOGLE_PROJECT UV_EXTRAS OPS_BUCKET CDR_BUCKET
variables=(OPS_BUCKET RUN_ID MAX_RETRIES MAX_RUN_SECONDS MACHINE_TYPE PROJECT_ID ZONE RUN_DISK_GB REGION NETWORK SUBNETWORK SERVICE_ACCOUNT)
if [ "$job" = fit ]; then variables+=(PROVISIONING_MODEL GPU_TYPE GPU_COUNT); fi
variables+=(CDR_BUCKET GOOGLE_PROJECT)
python3 "$here/budget_guard.py" render --template "$here/job_${job}.json.template" --output "$stage/job.json" --variables "${variables[@]}"
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
  --spec "$stage/job.json" --ledger "gs://${OPS_BUCKET}/svpgs/${RUN_ID}/budget"
gcloud storage cp "$stage/run_step.sh" "gs://${OPS_BUCKET}/svpgs/${RUN_ID}/launcher/${BUDGET_SLOT}/run_step.sh" --if-generation-match=0
echo "submitting ${job} in reserved slot ${BUDGET_SLOT}: ${MAX_RUN_SECONDS} seconds per attempt, at most $((MAX_RETRIES + 1)) attempts"
gcloud batch jobs submit "svpgs-${RUN_ID}-${BUDGET_SLOT}" --project="${PROJECT_ID}" --location="${REGION}" --config="$stage/job.json"
