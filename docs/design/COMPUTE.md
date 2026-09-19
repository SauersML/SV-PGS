# Compute

## Cost model
- **Workload:** n = 100k, p = 17M, 105 models (21 traits × 5 folds).
- **Where it comes from:** measured on MSI unless marked; COST_MODEL numbers from the speed lane.
- **What dominates:** passes over the store. One uint8 pass is about 1.7 TB uncompressed.

**Kernels (measured, exact):**
- **int8 → int32 Gram:** 39 TOPS on V100, 242–277 on A100, 1.4–1.6 POPS on H100.
- **int8 digit-split products X_b R with fp64 recombination** (`code_products.py`): 320 GB/s of codes on V100 at one right-hand side.
  - cuBLAS int8 GEMMs run only as TN, with dimensions and offsets that are multiples of 4, so tiles are zero-padded.
- **CPU (Milan):** an exact int16-madd Gram runs at 85 GOPS/core.

**I/O (measured):**
- Cold /scratch.global reads: 0.53 GB/s.
- Warm store iteration with ring prefetch: 7–8 GB/s; zstd-backed: 8–9.5 GB/s.
- Host → GPU pinned: 24 GB/s per GPU.
- Parallel `preadv` is used; mmap page faults ran at ~51 MB/s.

**Codec (design):**
- Quantization levels are chosen per variant by rate–distortion, for ≤ 0.1% r² loss.
- rANS for MAF < 5%, fixed-rate bitpacking above that.
- Expected 1–2 bits per code, i.e. 0.21–0.43 TB at 100k × 17M. That fits 8×H200 HBM, and 8×A100-40 at ≤ 1.5 bits per code.
- Rare columns are sparse in storage only: sparse arithmetic loses to dense tensor cores at ~1,800 right-hand sides.

**Stage 2** needs ~150 passes, each serving all 105 models plus 16 Hutchinson probes (~1,800 right-hand sides) [est]:

| Configuration | Stage 2 total | Verdict |
|---|---|---|
| 8×A100-40 spot | ~20–25 min [est] | primary |
| 8×H200 spot | ~8–10 min [est] | when spot capacity exists (unverified) |
| CPU ultramem VM (~3.8 TB RAM) | ~110 h [est] | compute-bound: Stage 0 or staging only |

**Stage 1 is the bottleneck.**
- The earlier implementation measured 39–55 µs per variant per model per sweep on H100, about 200× its flop bound.
- It needs batched block Cholesky and site updates across blocks and models on the GPU, with no per-block Python. The target is ≤ 1 µs, i.e. about 1 h for all models on 8×A100.
- An earlier branch, e2854ca in the wip history, has a measured fp64-vs-fp32 Cholesky policy (Jacobi-scaled fp32 factors with fp64 refinement) and multi-GPU block dispatch. Carry it into the new Stage 1.

## Cloud: the dedicated SV-PGS workspace
- **Platform:** a Verily Workbench (AoU Researcher Workbench 2.0) workspace in us-central1.
- **Quota** (read via the Service Usage API, which is reachable from outside the perimeter):
  - A100-40GB: 16 on-demand, 64 spot; L4 32/32; V100 8/32; T4 16/16;
  - spot H100 and H200: 128 by quota, capacity unverified;
  - m1/m3 ultramem VMs (~3.8 TB RAM).
- **Launch path (proven with synthetic data):**
  1. The laptop cannot call compute, Batch or storage APIs on the workspace project, because of VPC-SC.
  2. A single-node Workbench Dataproc cluster's initialization action runs inside the perimeter and submits Google Batch jobs, on the workspace network with no external IP, under the workspace service account.
  3. Progress is visible only as empty marker objects whose names carry non-sensitive facts: event, timing, machine, exit class.
- **Proven so far:**
  - spot CPU + local SSD, with checkpoint and resume on retry;
  - spot T4 with Batch-installed drivers.
- **Not yet tested:** a real preemption, H100/H200, multi-disk local SSD, and container mounts.
- **Identity:** always the dedicated SV-PGS user identity, never the other project identity.
  - The laptop's gcloud application-default credentials belong to the other identity, so `wb auth login --mode=APP_DEFAULT_CREDENTIALS` must never be used; re-login in the browser as the SV-PGS identity.
  - A `wb` create call can print a 401 and still succeed, so check `wb resource list` before retrying.
- **Observability:** read job status and marker filenames only, never object contents. Never encode participant-derived values in names.

## MSI (development compute; public and synthetic data only)
- **Direct nodes:** the two directly accessible nodes (acl42, acn112) are shared and heavily booked by other sessions' timing lanes.
  - Book explicit core slices with the other sessions before running.
  - Pin with `taskset` and `nice`, keep TMPDIR on /scratch.global, never /tmp (RAM-backed).
  - Timing benchmarks on shared cores are meaningless.
- **Slurm etiquette:**
  - Batch replicates inside tasks.
  - At most 2 concurrent jobs per agent.
  - Keep account-wide pending jobs under 90: backfill considers only the top 100 pending per user, so more starves every session's new jobs.
  - **Never mass-hold or mass-cancel array tasks.**
- **Status at handoff:** the account's default Slurm association submit counter underflowed. It reads MaxSubmitJobs=5000(4294967036), that is −260 wrapped to uint32, so every normal `sbatch` fails with AssocMaxSubmitJobLimit.
  - **Root cause, from Slurm source:** a bug in Slurm 25.05.9 (MSI agate).
    - In `src/slurmctld/acct_policy.c` `_adjust_limit_usage()`, the association `ACCT_POLICY_REM_SUBMIT` branch checks only `if (used_submit_jobs)`, not `>= job_cnt`. For a pending array job_cnt is the whole task count, so one over-removal wraps the counter. The QOS branch clamps correctly.
    - Upstream fix: SchedMD commit c9f89343b143 (ticket 24379), first released in 25.11.3 and not backported to 25.05.
  - **Why it stays stuck:** new submits are refused while every finishing job decrements further. It went −201 → −260 over the session.
  - **Our trigger:** mass hold and cancel of about 1,000 pending array tasks, plus about 40 `scontrol update partition=` calls on pending arrays. The single over-removing call needs root-only slurmctld debug2 logs to identify.
  - **Reset:** only an administrator can do it. A slurmctld restart or reconfigure runs `_restore_job_accounting()` in read_config.c, which clears and recounts usage. The permanent fix is Slurm ≥ 25.11.3. A support request was filed; any follow-up is sent by the user, never by an agent.
  - Until the reset, the `interactive` and `interactive-gpu` partitions still accept jobs, at most one running job per user each, shared across all sessions of the account.
- **Task runner (runq), the workaround while the counter is wrapped:**
  - one long `interactive-gpu` allocation (2×A40, 48 cores, 24 h, which submits its own successor) runs small tasks from file queues;
  - tasks go to `/scratch.global/<user>/svpgs-team/runq/{gpu,cpu}/pending/` as `<lane>__<name>__c<cores>[__g<gpus>].sh`, at most 16 cores each and at most 2 queued per lane;
  - the runner pins each task (`taskset`, `nice 10`), sets the thread counts, `CUDA_VISIBLE_DEVICES` and a private TMPDIR, and records each exit in `logs/records.jsonl`;
  - `touch <queue>/STOP` drains it;
  - the sources are `svpgs-team/runq_bin/` (runner.py and the two sbatch launchers);
  - a watcher (`runq_bin/slurm_watch.sh`, one core, nice 19) writes `runq/SLURM_RESTORED` once the counter is reset;
  - no task may call `sbatch` itself.
- **Environments** (uv, Python 3.12, synced `--locked` from `svpgs-team/env-src`, a detached checkout of origin/main):
  - `svpgs-team/venv-cpu`: the dev and sim groups (msprime included);
  - `svpgs-team/venv-gpu`: the same plus the `gpu` extra (cupy-cuda12x, jax[cuda12]). Export `LD_LIBRARY_PATH` from `venv-gpu/lib/python3.12/site-packages/nvidia/*/lib` before importing cupy or jax.
  - Both are built with `--no-install-project`, so `sv_pgs` is never installed in them. Every run uses its own worktree as cwd (`python -m pytest`) or sets `PYTHONPATH` to it, and logs `sv_pgs.__file__` and the commit. A run without either fails at import instead of silently using another commit.
  - To refresh after a lock change: update `env-src` to origin/main, then rerun `uv sync --locked --no-install-project --group dev --group sim [--extra gpu]` with `UV_PROJECT_ENVIRONMENT` set to the venv.
  - The older `main/.venv` belongs to the shared clone, which another session depends on. It predates zstandard and duckdb, so it cannot collect the store tests. Never sync or install into it.
