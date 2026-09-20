# Handoff: current state (2026-09-19)

This is the single entry point for the project's state. The model is in [MODEL.md](MODEL.md), the rulings in [DECISIONS.md](DECISIONS.md), compute and the landing gate in [COMPUTE.md](COMPUTE.md), and evaluation and benchmark results in [EVALUATION.md](EVALUATION.md). Evidence tags are defined in MODEL.md.

## On main (`8432a67`)
- **Store:** `dosage_store`, `store_converter` (per-half manifest, typed sample IDs), `store_block_source` (the streamed reader), and `synthetic_store` (public 1kGP haplotype mosaics, with reliability targets from bench-sim's v7 cohort).
- **Stage 0:** `genotype_statistics` (adjacent-block Grams, rank-deficient covariate projection) and `code_products` (code-domain products with derived digit counts).
- **Engine:** `scale_mixture_ep` — variant-side EP-EB with the D3-penalized mixing density, the λ step, the certified V, and the stationarity certificate.
- **Stage 2 solves:** `dual_solve` (DualGaussian, the negative-site split, information_solve) and `marginal_variances` (leave-block-out marginals, the information certificate, variance_jvp).
- **Scoring:** `fast_scoring` (trapezoid-rule predictive, Student-t).
- **Measurement and prior:** `imputation_reliability`, `sv_fusion`, `gatksv_source`, `prior_design`, `variant_typing` (merged classes), `sv_prior_features`, `external_annotations`, `hyperprior_pooling`.
- **Cohort and phenotypes:** `cohort`, `sample_ids`, `sample_crosswalk`, `all_of_us`, `phenotype_measurement` (the per-occasion measurement model with a learned noise density), `held_out_comparison`.
- **Benchmarks:** `benchmarks/bench_real` and `benchmarks/bench_sim` (v7 cohort from the public 1kGP founder composition, PREREG amendment 7).
- **Guards:** `tests/ep_eb_reference.py` (the EP-EB oracle) and `tests/test_no_arbitrary_constants.py`.
- **Still on main but being deleted:** `anderson.py` (lane/engine-anderson).
- **Not on main:** the certified outer loop, the Stage 2 driver (`full_data_fit.py`), logistic EP, and so any end-to-end fit. No SV-PGS accuracy number exists yet.

## Process
- **Landing:** lane branch → READY line in LANDQ → land-train runs the full MSI suite on the exact tip (runq cpu-node, under a memory ulimit, plus GPU tests if CUDA code changed) → fast-forward of main. GitHub CI runs on main pushes only, as a secondary signal (COMPUTE.md).
- **Coordination** lives outside the repo, in the team folder (`~/svpgs-team/`): TEAM_RULES.md (binding), LANDQ.md, FIXLOG.md, and each lane's STATUS.md.
- **Compute:** MSI only, through the runq task runners; the laptop does git and reading only. The Slurm submit counter is still wrapped, so normal `sbatch` fails (COMPUTE.md).
- **Evidence:** accuracy claims come only from bench-real, bench-sim, or later AoU held-out data inside the workspace. A lane's own simulations check math only.

## In flight, by lane
Queued in LANDQ (READY, not yet landed):
- **aou-audit** (URGENT): `lane/aou-audit-pipeline` replaces the untraced `PIPELINE_R2_LOSS` in `synthetic_store`.
- **deslop-hygiene** (URGENT): `lane/deslop-hygiene-replicates`, a memory-sizing fix for the phenotype replicate test.
- **e2e:** `lane/engine-anderson` deletes `anderson.py`.
- **bench-sim:** `lane/bench-sim-commit7` (sealed v7 commitments) and `lane/bench-sim-truthhalf` (the beagle_truthhalf arm, PREREG amendment 8).
- **speed-floor:** `lane/speed-floor-pooled`, compute_floor.md §10 re-measured on bench-sim v7.
- **ablate:** `lane/ablate-prereg`, the pre-registered term ablations (benchmarks/ABLATION_PLAN.md).
- **binary-ep:** `lane/binary-ep-logistic`, certified sample-side logistic EP (`logistic_ep.py`).
- **docs-sync:** this documentation pass.

Working:
- **e2e:** `wip/engine-driver`, the certified outer loop (Newton-B plus a trust region) and the Stage 2 driver; `lane/engine-fdbound` (a flaky-test fix); the EM regression test awaits the v7 fixture.
- **e2e-scale:** runs the driver at chr22 scale on bench-sim's public v7 store and profiles it against the floor [machinery].
- **oracle:** `fit_reference` fails on real-LD windows; being fixed.
- **verify-engine, verify-stage2, bug-engine, bug-stage2:** randomized verification and bug review of the engine, `dual_solve` and `marginal_variances`.
- **novel-inference:** the control-variate certificate, exact quadratics, the derived cavity tolerance, the b±2 window.
- **speed-floor, speed-krylov:** the learned-λ outer rate on the engine at chr22; batched multi-disease solves.
- **gpu-engine, multi-gpu, codec:** fused GPU kernels in the engine; multi-GPU sharding of Stage 0 and Stage 2; a GPU-decodable store codec.
- **measure-path:** D* recalibration and the A-map for draw-like imputed dosages (MODEL.md §2).
- **prior-terms:** the frequency, pooling, SV-context and shape terms.
- **fit-api:** the public fit/score API, the model artifact and the CLI.
- **pheno-disease, deslop-hygiene:** the disease channel model; the DE/sinh level transform and the R2 lattice extent.
- **workspace-pipeline:** the in-workspace pipeline driver, tested on synthetic data only.
- **bench-real:** the svfunction and portable arms; evoprior pending; waits for the engine entry point.
- **bench-sim:** the v7 Beagle arm, kernels, dev baselines; the GLIMPSE2 v7 subset at low priority.
- **novel-measure:** the `beagle_rb` arm, waiting on v7 Beagle output.
- **bug-recent:** guards main's CI and reviews each landing.
- **land-train:** runs the queue. **env:** successor runq runners.

## Next, in order
1. Land the queue.
2. The certified outer loop and Stage 2 driver (e2e), then the logistic EP hook.
3. The end-to-end fit: synthetic first, then chr22 on bench-sim v7 (e2e-scale).
4. Score SV-PGS on both benchmarks, bench-real and bench-sim's sealed test, then the pre-registered ablations.
5. The in-workspace pipeline, which needs the user's decisions below.

## The user's pending decisions
Agents never act on these and never contact anyone about them; the user decides and acts.
1. **Git history.** AoU-derived values remain in the public git history and in the `archive/2026-09-19/*` tags. Whether to rewrite history, delete tags, or leave them is the user's decision. A purge of the orphaned commit `ba1b71e` (blob `0b20f6b750f8b7899a860045a5fe6a4d5e2f20c0`) from GitHub's caches is also the user's call.
   - Also in main's history: the message of commit `302419a` (block_information_certificate, novel-inference) cites chr22 variance and cavity numbers measured on bench-sim's first chr22 cohort. That cohort was withdrawn for its AoU-derived group weights. The results were deleted locally and on MSI and reported to aou-audit; the message itself is left alone, as the lead ruled.
2. **Attach the All of Us Controlled Tier data collection (v9)** to the SV-PGS workspace. It holds the phenotypes, person tables, GATK-SV VCFs, and the only DRAGEN-to-research-ID crosswalk.
3. **Approve the store pull:** build the 8-bit stores inside the imputation workspace and copy only the stores into the SV-PGS workspace, in-perimeter. Agents never move data between workspaces without this approval.
4. **The MSI Slurm counter reset** (COMPUTE.md): whether to pursue it.

## Branches and archive
- GitHub holds `main` plus the lane branches above (`lane/*`, `wip/*`).
- Pre-restart branches are kept as `archive/2026-09-19/*` tags; recover one with `git checkout -b <name> archive/2026-09-19/<tag>`. The old fitting path is `archive/2026-09-19/old-path-final`.
