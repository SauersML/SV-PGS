# Handoff: state at pause (2026-09-19) and how to resume

## On main, all passing full-suite CI
- **New-path modules:**
  - `dosage_store.py` (8-bit store, halves, zstd, ring reads) and `store_converter.py` (background removal, recalibration hook, TR loci, SV context, long-read half);
  - `synthetic_store.py`;
  - `genotype_statistics.py` and `code_products.py` (Stage 0 and exact int8 products);
  - `exact_polish.py` (the Stage 2 E-step, block CG per model);
  - `fast_scoring.py` (one-pass scoring with posterior draws);
  - `imputation_reliability.py` (triad r², reliability model, linear D*);
  - `prior_design.py` and annotation-coefficient pooling across traits;
  - `external_annotations.py` (Bai 2026 SV/VNTR, Pan-UKB SNV, record and locus maps);
  - `gatksv_source.py`, `gatksv_store_rows.py`, `sv_fusion.py` (the general two-source measurement model) and `sample_crosswalk.py`;
  - `pleiotropy_layer.py` (not yet wired);
  - `held_out_comparison.py` (the evaluation tests and gates);
  - `compute_budget.py`, `tie_map.py`, `sample_table.py`;
  - phenotypes in `all_of_us.py`, trimmed to the 21-trait panel.
- **SPEC.md** carries the rulings: one model; CPU/GPU first-class; certification of approximate stages; EP-EB with a certificate; the r² prior; the TR length column; SV context in every prior; the learned mixing density; no hand-chosen priors.
- **The old path is still present** and is scheduled for deletion (below). Cutover step C0 (dead code) and most of C1 (module moves) are done.

## What remains, in order
1. **Stage 1 (EP-EB LD-space warm start) is not on main.**
   - The newest work is branch `build/ep-oracle`: a dense EP-EB reference, the nonparametric scale-mixture oracle, learned difference penalties, log-space mixing weights and an analytic Hessian. Finish it, land the oracle in `tests/`, then land Stage 1 gated against it.
   - Gates: rel ≤ 1e-6 to the dense reference; within ~2% of Gibbs; calibration slope ≥ 0.85; stable across sweeps 20–150.
   - Fix the implementation overhead: it needs batched block Cholesky and site updates across blocks and models (see COMPUTE.md).
   - Fold in the fp64/fp32 Cholesky policy and multi-GPU dispatch from e2854ca.
   - Settle the prior family: the learned mixing density vs TPB and BayesR on the reliability, TR-locus and multi-trait scenarios.
2. **Wire the full path end to end:** store → Stage 0 → Stage 1 → Stage 2 → score. Run it on synthetic data first; the harness is branch `lane/e2e`.
3. **Cutover:**
   - C2 flips the default entry point;
   - C3 deletes the marginal screen;
   - C4 deletes the old GIG core, PG-IRLS, TR-Newton, SVI, elbo and friends;
   - C5 deletes the old scorer, C6 the old genotype backends, C7 JAX, C8 config, CLI and docs.
   - The inventory is in the design history; recompute it against main before deleting.
4. **Data-side inputs from the imputation team (aggregates only):**
   - the r̂ model coefficients: target corr²(stored D, G), triad-corrected, with smooth terms in AF, log N_PATHS_TOTAL, log size and rsq_ds;
   - the E[G|DS] table, which gives the per-stratum κ for D*;
   - the has_pl mapping;
   - the TR-locus size histogram.
5. **In-workspace pieces:**
   - the fusion's E[B | SL] no-call fill (it needs the GATK-SV SL field, checked in the VCF header inside the workspace);
   - service-half gates S1–S3 when that half arrives.
6. **Measurements to rerun when compute is back:**
   - the evaluation validity set (QT, Q0, Q3, QC1, QS_pop, B0) and the red-team sweep;
   - credit identifiability;
   - the post-hoc TR read-evidence update, on public HPRC v2 with 1kGP 30× CRAM slices, scored cross-truth;
   - SV-context and external-annotation ablations (the B − A prior gain);
   - TR mutation-rate features;
   - the family-history liability targets for the diseases.
7. **Pilot:**
   - P1: the store converter on one chromosome of the real imputed data inside the imputation workspace, with counts-only QC.
   - P2: chr22 with two quantitative traits, SNV vs SNV+SV, on one spot VM.
   - Both run through the in-perimeter launcher (COMPUTE.md). Validate on synthetic data first.

## Steps only the user can take
1. **Attach the All of Us Controlled Tier data collection (v9)** to the SV-PGS workspace: Resources → Data from Catalog → All of Us Controlled Tier, then pick the version and its genomics bucket. This supplies phenotypes, person tables, the GATK-SV VCFs and the ID crosswalk.
2. **Confirm the ID-map source.** Imputed samples are keyed by sequencing IDs; the model needs research IDs.
3. **Approve the data pull.**
   - The plan: build the 8-bit stores inside the imputation workspace, then pull only the stores into the SV-PGS workspace with an in-perimeter copy.
   - The imputed store is about 0.45 TB for 50k samples, against about 13.6 TB of raw BCFs. The long-read half is about 0.1–0.2 TB.
   - The copy is same-region, so there is no egress. No permission changes are needed.
   - Agents never move data between workspaces without this approval.

## Branches
- The `wip/*` branches preserve unpushed worktree snapshots from MSI. All were reviewed against main: their valuable content is already there, or obsolete by ruling.
- Branches with commits not yet on main, to review before deleting:
  - `build/ep-oracle`: the live Stage 1 work.
  - `lane/e2e` and `lane/e2e-nodamp`: the end-to-end synthetic harness.
  - `build/stage2`, `build/store*`, `build/stage0`, `build/stage1`, `fast-arch`, `build/fast-scoring`, `build/phenotypes`, `lane/r2prior`: mostly superseded by main. Check each with `git cherry origin/main <branch>`.

## MSI at pause
- All SV-PGS jobs and processes are stopped. No MSI files were deleted: stores, simulation outputs, worktrees and venvs remain.
- The account's Slurm submit counter had underflowed (see COMPUTE.md). Check `scontrol show assoc_mgr users=<user> flags=assoc` before submitting.
