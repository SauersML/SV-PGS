# Handoff: current state (2026-09-21, main `eef5f00`)

This is the single entry point for the project's state. The model is in [MODEL.md](MODEL.md), the rulings in [DECISIONS.md](DECISIONS.md), compute and the landing gate in [COMPUTE.md](COMPUTE.md), and evaluation and benchmark results in [EVALUATION.md](EVALUATION.md). Evidence tags are defined in MODEL.md.

## On main
- **Store:** `dosage_store`, `store_converter` (per-half manifest, typed sample IDs), `store_block_source` (the streamed reader; rowdict stores decoded on the device), `rowdict_codec`, `resident_codes`, `copy_number` (integer copy number as a first-class column) and `synthetic_store` (public 1kGP haplotype mosaics, with reliability targets from bench-sim's v7 cohort).
- **Stage 0:** `genotype_statistics` (adjacent-block Grams, rank-deficient covariate projection, exact ties merged) and `code_products` (code-domain products with derived digit counts).
- **Engine:** `scale_mixture_ep` — variant-side EP-EB with the D3-penalized mixing density, the λ step, the certified V with the total curvature B (`CurvatureCorrection`; the linear response by block GCRO-DR, `krylov_recycle`), the stationarity certificate from V's analytic gradient, the trust-region outer loop that accepts only a resolved gain, offset groups (gene-owned levels), and the fused fp64 device kernels (`engine_kernels`, every node with its own variance).
- **Stage 2:** `dual_solve` (DualGaussian, the negative-site split, information_solve, the float64 floor as a reachable target, exact marginals for small n), `marginal_variances` (leave-block-out marginals, the information certificate, exact bulk marginals for flagged blocks from `exact_quadratics`, variance_jvp), `exact_marginals_scale`, `fold_share` and `fold_update`.
- **The Stage 2 driver and the fit API:** `full_data_fit` (store → Stage 0 → Stage 2 → the fitted model, one trait × fold per model), `stage2_wiring`, `tie_members`, `fit_model` (`fit`, `write_model`), `artifact` (the model artifact with covariate columns, fit counts and the offset digest) and the `fit` CLI command.
- **Small n and pooled fits:** `small_n` (the dense n × n route for cis windows: Stage 0 dense and EP-EB with exact algebra, the double loop, the KL certificate) and `pooled_fit` (one prior across genes, x frozen after the training genes, per-gene levels and certificates); `benchmarks/svpgs_method.py` and `benchmarks/svpgs_small_n.py` are SV-PGS's entries into bench-real and bench-sim.
- **Scoring:** `fast_scoring` (trapezoid-rule predictive, Student-t). **Binary traits:** `logistic_ep`.
- **Measurement and prior:** `imputation_reliability`, `measurement_model` (ancestry-pooled κ, the smooth reliability curve), `sv_fusion`, `gatksv_source`, `gatksv_store_rows`, `prior_design`, `variant_typing`, `sv_prior_features`, `external_annotations`, `hyperprior_pooling`.
- **Cohort, phenotypes and the workspace:** `cohort`, `sample_ids`, `sample_crosswalk`, `all_of_us`, `phenotype_measurement`, `held_out_comparison`, `workspace_pipeline` (the in-workspace driver, tested on synthetic data only) and the `workspace-run` CLI command.
- **Benchmarks:** `benchmarks/bench_real` (MAGE/1kGP expression; the within-group partial r² metric) and `benchmarks/bench_sim` (v7 cohort), plus the closed-form, tox, yeast and mouse designs.
- **Guards:** `tests/ep_eb_reference.py` (the EP-EB oracle), `tests/test_engine_verification.py` (the independent engine harness) and `tests/test_no_arbitrary_constants.py`.

## Where SV-PGS stands against mr.ash [real, bench-real chr22 gene 1 ENSG00000172404.5, loso/AFR, n = 534, p = 23,635 SNVs, 1 thread]
| arm | CPU-s | held-out r² | fit |
|---|---|---|---|
| SV-PGS, mean-field fixed points (`fit_small_n(inference="mean_field")`, main `eef5f00`) | 1,825 | 0.0291 | certified |
| numba mr.ash (`bench_real/baselines.py`) | 6 | 0.0308 | |
| SV-PGS, EP fixed points (run branch `1c888d9`, before the merge) | 3,296 | 0.0129 | certified |

Accuracy under SV-PGS's own learned prior is within 6% of mr.ash with the mean-field fixed points (EP was at 42%); the inference decision (definition of done, item 3) waits on the 20-gene loso/AFR run (`svpgs_mean_field` through the harness, within-group partial r² beside `mr_ashr_init`) and the EP arm of the same gene on the merged engine. The mean-field inference itself costs 26 CPU-s of the 1,825 (124 sweeps 13.5 s, 29 linear responses 8.4 s); **the hyper step is 94% (3 calls, 1,715 s)**: each certified-V evaluation is `_standardized` (the Tierney–Kadane cumulants of D ≈ 314 directions, 0.66 s) plus ~18 replaced line integrals at 0.21 s each (QAGI on the folded range, 60–120 kernel evaluations of p × K per line, the direction's scale part moving every kernel row), and the search makes 60–150 such evaluations per step (three starts at every stage, edges, the stationarity slopes' four extra evaluations per interior weight). That is the speed gap to the definition of done (item 5: CPU per gene at most numba mr.ash's): the floor for a weights search on a handful of penalty weights is tens of evidence evaluations, and each evaluation's floor is one p × K pass plus the integrals of the few directions whose Tierney–Kadane term exceeds their share (the terms: maximum ~10³, median 10⁻¹³ on this gene). The levers, in order: a cheap bound that skips the cumulants of directions that cannot exceed their share; fewer evaluations in the search; a quadrature that spends its points where the integrand is not Gaussian.

The 20-gene pooled runs of the run branch never produced a fit record (the outer loop's non-terminating cycle and its uncertifiable certificate, both fixed on main at `eef5f00`: an unresolved inner step is never accepted, and the certificate's resolvable pieces are tightened per decision). The definition of done (TEAM_RULES) is: never refuses, exact where it claims exactness, the inference chosen by measurement, pooled loso r² at least mr.ash's, CPU per gene at most numba mr.ash's, one code path. Items 1–3 are in reach; 4 waits on the 20-gene measurement; 5 is the hyper step above.
## Process
- **Landing:** lane branch → READY line in LANDQ → the full MSI suite on the exact tip (runq, `-m "not slow"` under a memory share, plus the GPU tests if CUDA code changed) → fast-forward of main. GitHub CI runs on main pushes only, as a secondary signal (COMPUTE.md).
- **Coordination** lives outside the repo, in the team folder (`~/svpgs-team/`): TEAM_RULES.md (binding), LANDQ.md, FIXLOG.md, and each lane's STATUS.md.
- **Compute:** MSI only, through the runq task runners; the laptop does git and reading only. The Slurm submit counter is still wrapped, so normal `sbatch` fails (COMPUTE.md).
- **Evidence:** accuracy claims come only from bench-real, bench-sim, or later AoU held-out data inside the workspace. A lane's own simulations check math only.

## Next, in order
1. The 20-gene loso/AFR mean-field run and the EP arm of gene 1 on main, both against mr.ash: the inference decision, then the loser is deleted (definition of done, items 3 and 6).
2. The hyper step to its floor (above), then the 20-gene pooled run against mr_ashr_init.
3. The engine's remaining certificate work: the fixed-point term δ_fp and the C-variation slope (theory-ep), the five normal-means cases where a block's weight ends far down its range at a near-flat direction of −H (strict expected failures in tests/test_scale_mixture_ep.py), the pooled oracle on `lane/fit-api-pooled-targets` 0227198.
4. Score SV-PGS on bench-sim's sealed test and the pre-registered ablations.
5. The in-workspace pipeline, which needs the user's decisions below.

## The user's pending decisions
Agents never act on these and never contact anyone about them; the user decides and acts.
1. **Git history.** AoU-derived values remain in the public git history and in the `archive/2026-09-19/*` tags. Whether to rewrite history, delete tags, or leave them is the user's decision. A purge of the orphaned commit `ba1b71e` (blob `0b20f6b750f8b7899a860045a5fe6a4d5e2f20c0`) from GitHub's caches is also the user's call.
   - Also in main's history: the message of commit `302419a` (block_information_certificate, novel-inference) cites chr22 variance and cavity numbers measured on bench-sim's first chr22 cohort. That cohort was withdrawn for its AoU-derived group weights. The results were deleted locally and on MSI and reported to aou-audit; the message itself is left alone, as the lead ruled.
2. **Attach the All of Us Controlled Tier data collection (v9)** to the SV-PGS workspace. It holds the phenotypes, person tables, GATK-SV VCFs, and the only DRAGEN-to-research-ID crosswalk.
3. **Approve the store pull:** build the 8-bit stores inside the imputation workspace and copy only the stores into the SV-PGS workspace, in-perimeter. Agents never move data between workspaces without this approval.
4. **The MSI Slurm counter reset** (COMPUTE.md): whether to pursue it.

## Branches and archive
- GitHub holds `main` plus the lane branches (`lane/*`, `run/*`, `wip/*`). The run branches `run/svpgs-working` and `run/svpgs-bench-1` and the engine lanes through `lane/engine-krylov-single-v` are merged into main; `lane/fit-api-pooled-targets` keeps one commit (0227198, the pooled oracle) that waits on engine interfaces.
- Pre-restart branches are kept as `archive/2026-09-19/*` tags; recover one with `git checkout -b <name> archive/2026-09-19/<tag>`. The old fitting path is `archive/2026-09-19/old-path-final`.
