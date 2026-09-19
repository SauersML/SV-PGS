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
  - `held_out_comparison.py` (the evaluation tests and gates);
  - `compute_budget.py`, `tie_map.py`;
  - `cohort.py` (the shared covariates, multi-trait targets and kinship-grouped folds);
  - phenotypes in `all_of_us.py`, trimmed to the 21-trait panel.
- **SPEC.md** carries the rulings: one model; CPU/GPU first-class; certification of approximate stages; EP-EB with a certificate; the r² prior; the TR length column; SV context in every prior; the learned mixing density; no hand-chosen priors.
- **The old path is deleted** (cutover C0–C8, CUTOVER.md); it is recoverable from tag `archive/2026-09-19/old-path-final`.

## What remains, in order
1. **Stage 1 is dropped** (lead ruling, 2026-09-19; MODEL.md §5, COMPUTE.md). The production outer map has no slow direction, and the certified Newton-B loop needs 0.7–2.6 outer steps [semi-real: speed-floor, bench-sim chr22 real-haplotype LD]. The pipeline is Stage 0, then Stage 2, then scoring.
   - **The dense EP-EB reference is on main:** `tests/ep_eb_reference.py`, checked by `tests/test_ep_eb_reference.py` (see MODEL.md §3–4). It remains the exactness oracle for the engine.
   - Settle the prior family: the learned mixing density vs TPB and BayesR on the reliability, TR-locus and multi-trait scenarios.
2. **Wire the full path end to end:** store → Stage 0 → Stage 2 (`full_data_fit.py`) → score. It runs from the prior, with the engine's Newton-B outer loop and certified marginals, on synthetic data first (`tests/test_full_data_fit.py`).
3. **Cutover: done** (C0–C8, `21cdec3`…`3d63745`). The step-by-step record, and the old tests whose intent the engine's tests carry, are in [CUTOVER.md](CUTOVER.md); the CDR input locations the old runner documented are in [CDR_LAYOUT.md](CDR_LAYOUT.md). The pre-cutover tree is tag `archive/2026-09-19/old-path-final`.
4. **Reliability inputs are computed only inside the AoU workspace, by the pipeline itself.** Nothing AoU-derived is delivered to SV-PGS outside the workspace, and no AoU-derived number is requested from the imputation team (user rule, 2026-09-19; the imputation team confirmed). The pipeline fits and uses, in-workspace, from the long-read truth rows:
   - the r̂ model: target corr²(stored D, G), triad-corrected, with smooth terms in AF, log N_PATHS_TOTAL, log size and rsq_ds;
   - the per-stratum, per-ancestry E[G|DS] calibration, which gives κ for D*;
   - the has_pl mapping and the TR-locus size distribution;
   - the A-map's per-block Σ_DG (scale_model.md §3).
5. **In-workspace pieces:**
   - the fusion's E[B | SL] no-call fill (it needs the GATK-SV SL field, checked in the VCF header inside the workspace);
   - service-half gates S1–S3 when that half arrives.
   - the phenotype rules that are not standards (PHENOTYPES.md): the disease evidence rules, the measurement windows and plausible ranges, the treatment corrections, the analysis scale and the covariate forms. Each needs its learned model, validated on synthetic OMOP first, before the case definitions are frozen (EVALUATION.md).
6. **Measurements to rerun when compute is back:**
   - every `[sim-only]` result a ruling rests on (MODEL.md, DECISIONS.md, EVALUATION.md), re-measured on the neutral benchmarks bench-real and bench-sim. Until then those rulings are provisional;
   - the evaluation validity set (QT, Q0, Q3, QC1, QS_pop, B0) and the red-team sweep;
   - credit identifiability;
   - the post-hoc TR read-evidence update, on public HPRC v2 with 1kGP 30× CRAM slices, scored cross-truth;
   - SV-context and external-annotation ablations (the B − A prior gain);
   - TR mutation-rate features;
   - the family-history liability targets for the diseases.
7. **Pilot:**
   - P1: the store converter on one chromosome of the real imputed data inside the imputation workspace, with QC read inside the workspace only; no value leaves it.
   - P2: chr22 with two quantitative traits, SNV vs SNV+SV, on one spot VM.
   - Both run through the in-perimeter launcher (COMPUTE.md). Validate on synthetic data first.

## Steps only the user can take
1. **Attach the All of Us Controlled Tier data collection (v9)** to the SV-PGS workspace: Resources → Data from Catalog → All of Us Controlled Tier, then pick the version and its genomics bucket. This supplies phenotypes, person tables, the GATK-SV VCFs and the ID crosswalk.
2. **Confirm the ID-map source.** Imputed samples are keyed by sequencing IDs; the model needs research IDs.
3. **Approve the data pull.**
   - The plan: build the 8-bit stores inside the imputation workspace, then pull only the stores into the SV-PGS workspace with an in-perimeter copy.
   - The imputed store is about 0.45 TB for 50k samples (8-bit codes); the long-read half is smaller.
   - The copy is same-region, so there is no egress. No permission changes are needed.
   - Agents never move data between workspaces without this approval.

## Branches
- GitHub has only `main`. Every other branch, local-only branch and dirty worktree state was checked against main and then deleted. Each one is kept as an `archive/2026-09-19/*` tag (33 tags), so any of them can be recovered with `git checkout -b <name> archive/2026-09-19/<tag>`.
- Tags worth reviving:
  - `build-ep-oracle`: the Stage 1 reference;
  - `build-stage1` and `wip-wt-build-store`: e2854ca;
  - `lane-e2e` and `lane-e2e-nodamp`: the end-to-end harness;
  - `local-lane-cutover-sim`: the unfinished msprime dependency group;
  - `dirty-*`: uncommitted worktree state.
- The rest hold only obsolete or superseded work, for example the old Stage 1 files `ld_space_fit.py` and `fast_fit.py`.

## MSI at pause
- All SV-PGS jobs and processes are stopped; nothing is queued.
- **Scratch cleanup, about 345 GB freed:**
  - kept: the main clone and its `.venv`; the stores `s1M_100k` (94 GB) and `mini`; `build-lead/synth` (127 GB); `lit-review`; the Descent olean cache; public-data caches; result dirs;
  - deleted by an interrupted cleanup: the agent script and log dirs, `venv-cpu-fast`, `venv-fast`, the Descent clone and `runq_bin`;
  - the venvs and the task runner were rebuilt on resume (`venv-cpu`, `venv-gpu`, `runq_bin`; see COMPUTE.md);
  - re-clone Descent from d86c2669 if it's needed.
- **Slurm:** the account's submit counter is still wrapped at −260 (see COMPUTE.md for the root cause). Check `scontrol show assoc_mgr users=<user> flags=assoc` before submitting. Until an admin reset, only the `interactive` partitions work.
