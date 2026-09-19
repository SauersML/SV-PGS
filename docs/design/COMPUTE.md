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
- **Status at handoff:** the account's default Slurm association submit counter underflowed (reported as MaxSubmitJobs=5000(≈4.29e9)), so every normal `sbatch` fails with AssocMaxSubmitJobLimit.
  - It needs an administrator reset, and a support request was filed.
  - Until the reset, the `interactive` and `interactive-gpu` partitions still accept jobs, at most one running job per user each, shared across all sessions of the account.
- **Imports:** script runs must set PYTHONPATH to their worktree and log `sv_pgs.__file__` and the commit. The shared clone's editable install otherwise imports whatever commit that clone has checked out.
- **Environment:** simulation studies need the uv `sim` dependency group (msprime).
