# In-workspace launcher: Google Batch

**No agent launches anything here.** Launching is the user's decision. Every value that names the workspace, its project, network, service account, buckets or disks is a `${PLACEHOLDER}`, and those values stay out of this public repository. The pipeline is `sv-pgs workspace-run` (docs/design/WORKSPACE_PIPELINE.md).

Jobs go straight to Google Batch: `submit.sh` runs `gcloud batch jobs submit` from a terminal inside the workspace (a Workbench app or VM in the workspace project, as the workspace's identity). There is no WDL, Cromwell or Dataproc step. Each job runs on the workspace network with no external IP.

| File | What it is |
|---|---|
| `submit.sh` | Fills a job template and `run_step.sh.template` from the environment, copies `run_step.sh` to the ops bucket, and submits the job with its spend ceiling |
| `run_config.template.json` | The pipeline's run config. Every key is required. `truth_calls` is a path pattern or an explicit `null`. `export` lists the report files the user approves for staging |
| `job_store.json.template` | The CPU job: steps samples through measurement, on spot CPUs with a persistent run disk |
| `job_fit.json.template` | The GPU job: steps fit through export |
| `run_step.sh.template` | The task script: builds the locked environment offline with uv from the staged files, then runs `sv-pgs workspace-run --through ${THROUGH}` |

## Study budget

`study_budget.template.json` allocates $2,250 to jobs and reserves $750 for storage,
queries and other overhead, within the user's $3,000 total. Five folds and the final
full-data fit remain enabled. No runtime or total-cost claim has been established
at biobank scale; measure a full-genome pilot before spending the training slots.

`submit.sh` requires `BUDGET_SLOT` and a current whole-VM `HOURLY_USD` quote. It
calculates a task timeout from that slot, including every retry and a startup
allowance. The default is no automatic retries. The six `train-1` through `train-6`
slots allow bounded successive jobs to resume the same run on its persistent disk.
Use `prepare` for conversion, `pilot` for a bounded benchmark, and `evaluate` for
remaining scoring/report work. Customize allocations before the first submission,
keeping their sum plus the reserve at or below $3,000. `STUDY_BUDGET` can point to
that customized JSON file.

The first submission writes an immutable plan to the workspace ops bucket. Each
slot is reserved with a GCS generation precondition before its job is submitted;
a second launcher cannot reuse it. Keep the same `RUN_ID` throughout the study.
Changing the study ID or submitting jobs outside this launcher bypasses this
ledger. A failed submission retains its reservation and must be inspected; it is
never automatically refunded or resubmitted under a fresh job name.

These are submission and task-runtime controls, **not an account-wide billing
cap**. Batch task timeouts do not bound VM provisioning/cleanup or stop persistent
disk and bucket charges. Check actual spending and storage retention against the
$750 reserve. Do not delete the budget ledger or reuse a spent slot. Source:
[Batch task timeouts](https://docs.cloud.google.com/batch/docs/set-timeouts).

The engine checkpoints each completed fold atomically, preserving its seeds,
posterior draw law and convergence certificate. Checkpoints are bound to the code,
store and full fitting problem. After an interruption only the unfinished fold
restarts. Per-fold timing records are retained in each completed arm/trait's JSON
record as `completed_fold_timings`; unfinished groups retain them in the checkpoint
metadata. Checkpointing does not relax convergence or remove variants.

## Staging (once, from a Workbench app with internet access)

Under `gs://${OPS_BUCKET}/svpgs/${RUN_ID}/`:
- `source/`: this repository at the commit to run (`git archive`).
- `bin/uv`: the uv binary (`cp "$(command -v uv)"`).
- `python/`: CPython 3.12, `UV_PYTHON_INSTALL_DIR=python uv python install 3.12`.
- `uv-cache/`: every locked wheel, `UV_CACHE_DIR=uv-cache UV_PYTHON_INSTALL_DIR=python uv sync --frozen --no-dev --extra gpu --python 3.12` in the source tree (the CPU job's `UV_EXTRAS` is empty and uses a subset).
- `run_config.json`: the filled `run_config.template.json`.

The staging machine and the Batch VMs must be the same platform (Linux x86-64). The inputs the config points at are read where they sit inside the workspace, through a mounted bucket or the run disk.

## Placeholders

Exported before `submit.sh`:
- `PROJECT_ID`, `REGION`, `ZONE`, `NETWORK`, `SUBNETWORK`, `SERVICE_ACCOUNT`;
- `OPS_BUCKET`, `RUN_ID`, `RUN_DISK`;
- `WORKSPACE_CDR`, `GOOGLE_PROJECT`;
- `MACHINE_TYPE`, `PROVISIONING_MODEL`, `GPU_TYPE`, `GPU_COUNT`, `THROUGH`, `UV_EXTRAS` (`--extra gpu` for the fit job);
- `BUDGET_SLOT`, `HOURLY_USD`; optional `STUDY_BUDGET` and `MAX_RETRIES` (default 0).

Every one is the user's choice.

## Observing a run

- The only signal is empty marker objects under `svpgs/${RUN_ID}/markers/`, named `<UTC>_<step>_<event>` by the driver and `<UTC>_task_<event>` by the task script.
- Logs go to the run disk (`logsPolicy` PATH), never to a service outside the workspace. The engine logs completed-fold runtimes and checkpoint reuse.
- There is no notification, webhook, email or upload step, and none may be added.

## Restarts

- A spot preemption (Batch exit code 50001) or the task script's exit 75 retries the task, up to `MAX_RETRIES`. The driver resumes from the run disk's checkpoints: the store per chromosome, the fit per completed fold and then per (arm, trait).
- Any other exit fails the job and leaves the run directory as it was, for the user to inspect inside the workspace.
