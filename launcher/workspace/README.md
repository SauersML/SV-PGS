# In-workspace launcher: templates only

**Nothing here has ever been launched, and no agent launches it.** Launching is the user's decision. Every value that names the workspace, its project, network, service account, buckets or disks is a `${PLACEHOLDER}`, and those values stay out of this public repository. The pipeline is `sv-pgs workspace-run` (docs/design/WORKSPACE_PIPELINE.md).

The path is the in-perimeter pattern of COMPUTE.md, "Cloud":
1. The laptop cannot reach the workspace project's Compute, Batch or Storage APIs (VPC-SC).
2. A zero-worker Workbench Dataproc cluster's initialization action (`init.sh.template`) runs inside the perimeter as the workspace service account and submits one Google Batch job.
3. The job runs on the workspace network, with no external IP.

| File | What it is |
|---|---|
| `run_config.template.json` | The pipeline's run config. Every key is required. `truth_calls` is a path pattern or an explicit `null`. `export` lists the report files the user approves for staging |
| `init.sh.template` | The Dataproc initialization action: copies the filled job spec from the ops bucket and submits it with `gcloud batch jobs submit` |
| `job_store.json.template` | The CPU job: steps samples through store (or through measurement), on spot CPUs with a persistent run disk |
| `job_fit.json.template` | The GPU job: steps fit through export, on 8 GPUs |
| `run_step.sh.template` | The task script: installs the staged wheelhouse into a venv on the run disk, then runs `sv-pgs workspace-run --through ${THROUGH}` |

**Filling.** `envsubst < X.template > X` with every placeholder exported. The placeholders:
- `PROJECT_ID`, `REGION`, `ZONE`, `NETWORK`, `SUBNETWORK`, `SERVICE_ACCOUNT`;
- `OPS_BUCKET`, `RUN_ID`, `RUN_DISK`;
- `WORKSPACE_CDR`, `GOOGLE_PROJECT`;
- `MACHINE_TYPE`, `GPU_TYPE`, `GPU_COUNT`, `THROUGH`.

Every one is the user's choice. The machine types the resource plan derives are in WORKSPACE_PIPELINE.md, "Resource plan".

**What the user stages** in the ops bucket under `svpgs/${RUN_ID}/`:
- `launcher/` (the filled job spec and `run_step.sh`);
- `wheels/`, sv-pgs and its dependencies as wheels, because the VM has no external IP;
- `run_config.json`.

The inputs the config points at are read where they sit inside the workspace, through a mounted bucket or the run disk.

**Observing a run.**
- The only signal is empty marker objects under `svpgs/${RUN_ID}/markers/`, named `<UTC>_<step>_<event>` by the driver and `<UTC>_task_<event>` by the task script.
- Logs go to the run disk (`logsPolicy` PATH), never to a service outside the workspace.
- There is no notification, webhook, email or upload step, and none may be added: job status goes only to files inside the workspace.

**Restarts.**
- A spot preemption (Batch exit code 50001) or the task script's exit 75 retries the task. The driver resumes from the run disk's checkpoints.
- Any other exit fails the job and leaves the run directory as it was, for the user to inspect inside the workspace.
