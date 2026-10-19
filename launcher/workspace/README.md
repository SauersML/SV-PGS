# In-workspace launcher: Google Batch

**No agent launches anything here.** Launching is the user's decision. Every value that names the workspace, its project, network, service account, buckets or disks is a `${PLACEHOLDER}`, and those values stay out of this public repository. The pipeline is `sv-pgs workspace-run` (docs/design/WORKSPACE_PIPELINE.md).

Jobs go straight to Google Batch: `submit.sh` runs `gcloud batch jobs submit` from a terminal inside the workspace (a Workbench app or VM in the workspace project, as the workspace's identity). There is no WDL, Cromwell or Dataproc step. Each job runs on the workspace network with no external IP.

| File | What it is |
|---|---|
| `submit.sh` | Fills a job template and `run_step.sh.template` from the environment, copies `run_step.sh` to the ops bucket, and submits the job with its spend ceiling |
| `run_config.template.json` | The pipeline's run config. Every key is required. `truth_calls` is a path pattern or an explicit `null`. `imputation_reference_ids` is the file of the imputation reference panel's research IDs, or an explicit `null`, under which truth calls give no calibration pair. `direct_sv.gatksv_calls` is the CDR's GATK-SV call set, a path pattern over `{chromosome}` under `/mnt/disks/cdr` (the CPU job's read-only mount of `CDR_STORAGE_PATH`; check the release's file names there), or `null` for no direct rows, and every other `direct_sv` key must stay `null`. `panel` is the pre-registered panel test, frozen before any phenotype is read: `primary_fold` and each disease's weight (regenerate with `python -m benchmarks.power.expected --panel-weights ...`, EVALUATION.md); it writes `panel.tsv`. `export` lists the report files the user approves for staging |
| `job_store.json.template` | The CPU job: steps samples through direct_sv (`THROUGH=direct_sv`), on spot CPUs with a new run disk |
| `job_fit.json.template` | The GPU job: steps fit through export |
| `run_step.sh.template` | The task script: builds the locked environment offline with uv from the staged files, restores the run from its mirror, then runs `sv-pgs workspace-run --through ${THROUGH}`, pushing the run back every 15 minutes and at exit |

## The run's home: the workspace bucket

Batch cannot write to an existing persistent disk ("All options except writing are supported", [Batch storage](https://docs.cloud.google.com/batch/docs/create-run-job-storage)), so no disk carries a run from one job to the next. Each job instead gets a new `pd-balanced` disk of `RUN_DISK_GB`, and the run lives in `gs://${OPS_BUCKET}/svpgs/${RUN_ID}/run/` (`sv-pgs run-mirror`, `sv_pgs/run_mirror.py`):
- **Restored before the driver starts:** every completed step, and the fit step's checkpoints and saved models. Of the store copies (the store, measurement and direct_sv steps each write one), only the last completed one is downloaded; each earlier one is linked to its mirrored copy on the `/mnt/disks/ops` mount, since only a step made again reads it.
- **Pushed while the driver runs:** a step once it completes, the fit's checkpoints as they appear, and everything once more when the driver exits.
- **Sizing:** the mirror holds participant-level data, so `OPS_BUCKET` must be the workspace's own bucket. `RUN_DISK_GB` must hold every step's output, including the store step's, measurement step's and direct_sv step's copies of the store.
- **Cost:** a job's start copies the mirror to its disk, with one store copy rather than three once those steps are done. A preemption loses at most 15 minutes of fit progress and any unfinished step other than the fit. The mirror's storage comes out of the $750 reserve.


## Study budget

`study_budget.template.json` allocates $2,250 to jobs and reserves $750 for storage, queries and other overhead, within
the user's $3,000 total: `prepare` for the store job, `pilot` and `train-1` to `train-6` for fit jobs, `evaluate` for
remaining scoring and report work. Customize the allocations before the first submission, keeping their sum plus the
reserve at or below $3,000; `STUDY_BUDGET` can point to that customized file.

`submit.sh` requires `BUDGET_SLOT` and a current whole-VM `HOURLY_USD` quote, and sets the task timeout from that slot,
including every retry and a startup allowance (default: no automatic retries). The first submission writes an immutable
plan to the ops bucket, and each slot is reserved with a GCS generation precondition before its job is submitted, so
no slot is spent twice. Keep the same `RUN_ID` throughout the study. A failed submission retains its reservation.

The first fit job is the run itself: steps are keyed by their inputs, and the fit checkpoints each completed fold, so a
job that reaches its slot's timeout is continued by the next slot's job from the mirror, losing at most 15 minutes. The
diseases fitted are the run config's `diseases`: changing that list later re-runs the cohort step and every fit, so
choose it before the first fit job. Per-fold timings are in each arm/trait's JSON record (`completed_fold_timings`).

These are submission and task-runtime controls, **not an account-wide billing cap**: Batch task timeouts do not bound VM
provisioning or bucket charges ([Batch task timeouts](https://docs.cloud.google.com/batch/docs/set-timeouts)). Check
actual spending against the $750 reserve.

## Staging (once, from a Workbench app with internet access)

`OPS_BUCKET=... RUN_ID=... launcher/workspace/stage.sh run_config.json` stages all of the following from a clean checkout at the commit to run; it refuses a run config with a `${PLACEHOLDER}` left or a template key missing.

Under `gs://${OPS_BUCKET}/svpgs/${RUN_ID}/`:
- `source/`: this repository at the commit to run (`git archive`).
- `bin/uv`: the uv binary (`cp "$(command -v uv)"`).
- `python/`: CPython 3.12, `UV_PYTHON_INSTALL_DIR=python uv python install 3.12`.
- `uv-cache/`: every locked wheel, `UV_CACHE_DIR=uv-cache UV_PYTHON_INSTALL_DIR=python uv sync --frozen --extra gpu --python 3.12` in the source tree. The CPU job's `UV_EXTRAS` is empty and uses a subset.
- `run_config.json`: the filled `run_config.template.json`; its `direct_sv.gatksv_calls` reads the CDR's GATK-SV VCFs through the CPU job's `/mnt/disks/cdr` mount.

The staging machine and the Batch VMs must be the same platform (Linux x86-64). The inputs the config points at are read where they sit inside the workspace, through a mounted bucket.

## Placeholders

Exported before `submit.sh`:
- `PROJECT_ID`, `REGION`, `ZONE`, `NETWORK`, `SUBNETWORK`, `SERVICE_ACCOUNT`;
- `OPS_BUCKET` (the workspace bucket), `RUN_ID`, `RUN_DISK_GB` (each job's new disk, in GB);
- `WORKSPACE_CDR`, `GOOGLE_PROJECT`, and `CDR_STORAGE_PATH` (the workspace's own `gs://` CDR path, which both jobs mount read-only at `/mnt/disks/cdr`, billed to `GOOGLE_PROJECT`: the store job reads the GATK-SV call set there, and every job hashes each step's inputs, the call set's included, to key the steps it skips);
- `MACHINE_TYPE`, `PROVISIONING_MODEL`, `GPU_TYPE`, `GPU_COUNT`, `THROUGH`, `UV_EXTRAS` (`--extra gpu` for the fit job);
- `BUDGET_SLOT`, `HOURLY_USD`; optional `STUDY_BUDGET` and `MAX_RETRIES` (default 0).

Every one is the user's choice.

## Observing a run

- The only signal is empty marker objects under `svpgs/${RUN_ID}/markers/`, named `<UTC>_<step>_<event>` by the driver and `<UTC>_task_<event>` by the task script.
- Logs go to `svpgs/${RUN_ID}/launcher/<slot>/batch.log` in the ops bucket (`logsPolicy` PATH), never to a service outside the workspace. The driver logs each fit's hours, the store reads and checkpoint reuse.
- There is no notification, webhook, email or upload step, and none may be added.

## Restarts

- A spot preemption (Batch exit code 50001) or the task script's exit 75 retries the task, up to `MAX_RETRIES`. The retry restores the run from its mirror. The driver then resumes from the mirrored checkpoints: the fit per completed model group and then per (arm, trait), and every other step from its start.
- Any other exit fails the job and leaves the run directory as it was, for the user to inspect inside the workspace.
