# Compute

## Cost model
- **Workload:** n = 100k, p = 17M, 105 models (21 traits × 5 folds).
- **Where it comes from:** measured on MSI unless marked [machinery]; the derivations and the per-stage gap are in [math/compute_floor.md](math/compute_floor.md).
- **What dominates:** passes over the store. One uint8 pass is about 1.7 TB uncompressed.

**Kernels (measured, exact).** The exact int8 path's achieved rate on the A40 is in compute_floor.md §9.3: the raw GEMM runs at 68–73% of spec in the pass shape.
- **int8 → int32 Gram:** 39 TOPS on V100, 242–277 on A100, 1.4–1.6 POPS on H100.
- **int8 digit-split products X_b R with fp64 recombination** (`code_products.py`): 320 GB/s of codes on V100 at one right-hand side.
  - cuBLAS int8 GEMMs run only as TN, with dimensions and offsets that are multiples of 4, so tiles are zero-padded.
- **CPU (Milan):** an exact int16-madd Gram runs at 85 GOPS/core.

**I/O (measured):**
- Cold /scratch.global reads: 0.53 GB/s.
- Warm store iteration with ring prefetch: 7–8 GB/s; zstd-backed: 8–9.5 GB/s.
- Host → GPU pinned: 24 GB/s per GPU.
- Parallel `preadv` is used; mmap page faults ran at ~51 MB/s.

**Codec (status):**
- The store is zstd today. Host-side zstd decode runs at about 0.65 GB/s per core, about 2,600 core-seconds per full read at N·P = 1.7e12, so no host-decoded format reaches the floor [machinery].
- A lossless GPU-decodable prototype (a per-row dictionary, fixed-rate slots and exceptions, with k chosen by exact byte count) gave 2.22 bits per code against zstd's 1.62 and decoded at 186 GB/s on an A40, on the synthetic s1M_100k store [machinery: synthetic store]. Real imputed DS may differ; its size is not measured, and an AoU-side aggregate of it was withdrawn under the AoU rule.
- The earlier design target of 1–2 bits per code (rate–distortion levels, rANS below 5% MAF) is [est].
- Rare columns are sparse in storage only: sparse arithmetic loses to dense tensor cores at ~1,800 right-hand sides [sim-only: synthetic store; timing].

**The floor and the gap** (derived and measured in [math/compute_floor.md](math/compute_floor.md)). All 105 models, n = 10⁵, p = 1.7·10⁷:
- **Floor:** ~47 store passes after the one cold staging read.
  - 105 right-hand sides per fit pass: no Hutchinson probes, since tr(ΛΣ) = p − Σ_j τ_jΣ_jj gives p_eff exactly.
  - 3 int8 operand digits, from the certificate's tolerance.
  - Sample-side solver state.
  - Stage 0 fused into the staging read; in-cohort scoring riding the final pass.
- **Floor totals:** about **6 min on 8×A100-40** and 4 min on 8×H100 (both staging-bound), ~36 min on one A40, and 1.5–2.6 h on one V100 or T4 (variance-bound).
- **CPU ultramem VM (~3.8 TB RAM, ~2.3 TF fp64 [est]):** compute-bound for the fit, so it's for staging and Stage 0 only.
- **Each additional disease** adds right-hand-side columns, not passes. Below the pass's compute-bound column count R\* = F_int8/(2·L·codes/s), it costs no pass time.

**Gap today** (itemized in compute_floor.md §9):

| Item | Current or planned | Floor | Factor |
|---|---|---|---|
| Stage 1 | 39–55 µs per variant·model·sweep (H100), ~800 GPU-h | dropped provisionally on cost (compute_floor.md §3) | removed |
| Columns per fit pass | 1,785 (16 probes per model) | 105 | 17× |
| Solver state | ~6 variant-side fp64 host arrays, 1.46 TB at R = 1,785 (exceeds a 680 GB host) | sample-side n×R, 1.4 GB | infeasible → feasible |
| Posterior draws | separate from-zero block-CG, R = 6,720 × ~25 passes | recycled through the last outer steps, with a block control variate | ~40× |
| Stage 2 passes | ~131 (cold corrections per outer step) | ~42 (inexact, warm-started) | ~3× |
| Operand digits | 6 | 3 | 2× |
| Block-variance refreshes | a dense fp64 factor per model per refresh (1,030 s per A100 for 105 models) | on demand, TF32 with refinement, resolved-set factors | 8–50× [est] |
| Exact path around the int8 GEMM (A40, measured) | matmat and rmatmat 4–10× over their raw GEMM | ≈ the raw GEMM | partly closed by speed-io, bit-identical |

End to end: about 1,000× today, and 10–40× once Stage 1 is removed.

**Stage 1 is dropped, provisionally (lead's decision, 2026-09-19), on the cost argument alone.** One Stage 1 sweep's variance refresh costs 75–600 Stage 2 pass-equivalents (compute_floor.md §3).
- The outer-convergence measurement first cited for this decision ("no slow direction", 1–3 outer steps) was withdrawn: it linearized at the true prior, which is a saddle of the genome-scaled penalized evidence (compute_floor.md §10.1–10.3).
- The decision is revisited only if the corrected measurement at the pooled fixed point shows production needs more outer steps × passes per step than a Stage 1 sweep costs.
- The outer step is Newton with the total curvature B and a trust region. The fixed-cavity curvature A + S is indefinite in every measured configuration, so plain EP-EM is ill-posed.
- Stage 2 uses only certified marginals. Block-Jacobi variances put the top ~1% of cavity precisions 28–58% off, so they need cross-block correction (compute_floor.md §10.4).
- The variance refreshes still need e2854ca's fp64-vs-fp32 Cholesky policy (Jacobi-scaled fp32 factors with fp64 refinement; archive tags `build-stage1` and `wip-wt-build-store`) and its multi-GPU block dispatch.
- Evidence for the two bullets above: at the true prior of the pooled chr22 problem, A + S was indefinite in every measured configuration [semi-real: public 1kGP-haplotype chr22 LD with simulated effects]. On bench-sim's v7 chr22 LD, block-Jacobi variances put the top 1% of cavity precisions 54% off at block cap 1024 and 25% at cap 4096, and 1.5% at cap 4096 with the second-order term [semi-real: compute_floor.md §10.4].

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
- **Slurm status:** the account's default Slurm association submit counter underflowed. It reads MaxSubmitJobs=5000(4294967036), that is −260 wrapped to uint32, so every normal `sbatch` fails with AssocMaxSubmitJobLimit.
  - **Root cause, from Slurm source:** a bug in Slurm 25.05.9 (MSI agate).
    - In `src/slurmctld/acct_policy.c` `_adjust_limit_usage()`, the association `ACCT_POLICY_REM_SUBMIT` branch checks only `if (used_submit_jobs)`, not `>= job_cnt`. For a pending array job_cnt is the whole task count, so one over-removal wraps the counter. The QOS branch clamps correctly.
    - Upstream fix: SchedMD commit c9f89343b143 (ticket 24379), first released in 25.11.3 and not backported to 25.05.
  - **Why it stays stuck:** new submits are refused while every finishing job decrements further. It went −201 → −260 over the session.
  - **Our trigger:** mass hold and cancel of about 1,000 pending array tasks, plus about 40 `scontrol update partition=` calls on pending arrays. The single over-removing call needs root-only slurmctld debug2 logs to identify.
  - **Reset:** only an administrator can do it. A slurmctld restart or reconfigure runs `_restore_job_accounting()` in read_config.c, which clears and recounts usage. The permanent fix is Slurm ≥ 25.11.3. Whether to pursue a reset is the user's decision: the user decides and acts, and no agent contacts MSI or anyone else about it.
  - Until the reset, the `interactive` and `interactive-gpu` partitions still accept jobs, at most one running job per user each, shared across all sessions of the account.
- **Task runners (runq), the workaround while the counter is wrapped.** Two long allocations run small tasks from file queues under `/scratch.global/<user>/svpgs-team/runq/`:
  - the GPU runner, one `interactive-gpu` job (2×A40, 48 cores), serves `gpu/` and `cpu/`;
  - the CPU runner, one `interactive` job on a 124-core node, serves `cpu-node/`, with node-local ext4 TMPDIR;
  - each submits its own successor before its walltime ends.
  - **Task names:** `<lane>__<name>__c<cores>[__m<GB>][__g<gpus>].sh`, written as `.tmp` then moved in; at most 16 cores each, at most 2 queued per lane, and one GPU task per lane while others wait. Only a task whose name has `__g<n>` gets a GPU.
  - **Memory budgets** (`__m<GB>`, or a cores-proportional default, exported as `RUNQ_MEM_BYTES`) take effect only on successor runners. The live runners reject names with `__m`.
  - **Cancel** only by `touch <queue>/cancel/<task>`, never by deleting from `pending/`. Successor runners kill a cancelled task's whole session and never assign a busy GPU. Until the CPU runner's successor is live, never cancel a RUNNING task there: the old code can crash on that path and take every task with it.
  - Inside tasks use `timeout --foreground`, never a plain `timeout`, which escapes the cancel.
  - The runner pins each task (`taskset`, `nice 10`), sets the thread counts, `CUDA_VISIBLE_DEVICES` and a private TMPDIR, and records each exit in `logs/records.jsonl`; `touch <queue>/STOP` drains it.
  - Sources: `svpgs-team/runq_bin/` (runner.py and the sbatch launchers). A watcher (`runq_bin/slurm_watch.sh`, one core, nice 19) writes `runq/SLURM_RESTORED` once the counter is reset. No task may call `sbatch` or `srun` itself.
- **Landing gate:** the full suite on MSI (runq `cpu-node`) on the exact tip that lands, under a memory ulimit so allocation bugs fail, plus the GPU tests on runq `gpu` when CUDA code changes. The merge-queue lane stacks the ready branches and fast-forwards main only on green. GitHub CI runs on pushes to main as a secondary signal; a failure there blocks nothing unless it reproduces on MSI.
- **Data sources on MSI:** public data only, and never from a Google Cloud Storage or AoU bucket (user rule, 2026-09-19): EBI/IGSR, Zenodo, NCBI, UCSC, AWS open data and tool authors' sites.
- **Environments** (uv, Python 3.12, synced `--locked` from `svpgs-team/env-src`, a detached checkout of origin/main):
  - `svpgs-team/venv-cpu`: the dev and sim groups (msprime included);
  - `svpgs-team/venv-gpu`: the same plus the `gpu` extra (cupy-cuda12x and the CUDA 12 library wheels it loads). Export `LD_LIBRARY_PATH` from `venv-gpu/lib/python3.12/site-packages/nvidia/*/lib` before importing cupy.
  - Both are built with `--no-install-project`, so `sv_pgs` is never installed in them. Every run uses its own worktree as cwd (`python -m pytest`) or sets `PYTHONPATH` to it, and logs `sv_pgs.__file__` and the commit. A run without either fails at import instead of silently using another commit.
  - To refresh after a lock change: update `env-src` to origin/main, then rerun `uv sync --locked --no-install-project --group dev --group sim [--extra gpu]` with `UV_PROJECT_ENVIRONMENT` set to the venv.
  - The older `main/.venv` belongs to the shared clone, which another session depends on. It predates zstandard and duckdb, so it cannot collect the store tests. Never sync or install into it.
