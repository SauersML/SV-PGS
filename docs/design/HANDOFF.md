# Handoff: current state (2026-10-14)

This is the entry point for the project's state. The model is in [MODEL.md](MODEL.md), the AoU run in [WORKSPACE_PIPELINE.md](WORKSPACE_PIPELINE.md) and [launcher/workspace/README.md](../../launcher/workspace/README.md), what the routes do not yet meet in [CAPABILITIES.md](CAPABILITIES.md), and the rulings in [DECISIONS.md](DECISIONS.md).

**"Certified"** means a fit met its numerical stopping criterion on the computed objective: remaining gain at most 1/(2K) nats on V, the Laplace evidence with the leading Tierney–Kadane term along each standardized direction. That term has no remainder bound, so a certified V is not within 1/(2K) of the exact evidence (CAPABILITIES.md). The artifact records it as `outer_criterion_met`; `model_acceptance` quarantines a model without it.

## The study
SV-inclusive Bayesian PGS for 15 diseases (hypertension first) in about 100k multi-ancestry All of Us participants, imputation already done: the `snv_indel` arm (records under 50 bp, no copy number) against `with_sv` (everything), paired on the same folds and seeds, with effects by SV type and length. Compute is under $3,000, as Google Batch jobs the user submits (`launcher/workspace/submit.sh`); agents never launch anything in the workspace.

## On main
- **Workspace pipeline** (`sv_pgs/workspace_pipeline.py`): samples, store, phenotypes, cohort, measurement, direct_sv, fit, score, effects, report, export. Each step is keyed by its inputs, upstream keys and code; a finished step is never redone. Inputs on a mounted bucket are keyed by object generation (`content_identity`).
- **Batch launcher** (`launcher/workspace/`): each job runs on a new disk, so the run lives in the workspace bucket (`sv_pgs/run_mirror.py`: restored before the driver, pushed every 15 minutes and at exit; superseded store copies are linked to the mount, not downloaded). The budget guard reserves each slot once.
- **Fit** (`stage2_wiring.fit_models`): models sharing training rows, covariates and likelihood are fitted together on one Stage 0. Stage 2's reads are sharded by LD block over every GPU of the budget. The mean-field fixed point runs on the samples or, where a pass measures cheaper, on Stage 0's LD band with the exact far field (a round that lowers the exact ELBO is redone under a proximal minorizer). A band-route model the gate would quarantine is refitted on the sample route.
- **Acceptance:** a quarantined model is never scored or reported. Its trait leaves every table and is listed in `report/quarantined.tsv`; the other traits are reported.

## Next, in order
1. **Re-stage the uv cache** with the dev group (README "Staging": `uv sync --frozen --extra gpu`, no `--no-dev`).
2. **`submit.sh gpucheck`** on two A100s (a `gpucheck-*` slot). No CUDA path has ever run: the CUDA tests and a small end-to-end fit run there first. A failure is fixed and checked again in the next slot.
3. **The pilot** (`pilot` slot): a full-genome fit of one trait, to measure the hours per arm, the store passes, and how often a fit certifies at this scale before the training slots are spent.
4. The training slots, then `evaluate`.

## Open engine work
- **Certification at the width → 0 boundary.** On some real genes the evidence has no interior maximum along the mixing density's width direction (B + S negative there to the boundary, a ridge prior in the limit), and the fit returns uncertified there (`tests/test_mean_field.py::test_the_mean_field_fit_certifies`, strict xfail). How often this happens at biobank scale is unmeasured; the pilot measures it.
- **The walk outside every basin.** Where an empirical Bayes ends uncertified and the lattice check extends the lattice (the fitted density's mass above its top), the refit can walk through states with no certified value, planning a hyper step (its stationarity's evidence maximizations) at each: minutes per step on `tests/test_bench_sim_derived.py`'s 240-person annotated cohort (marked slow), where the route timing picks decides whether the fit takes 50 s or more than 10 minutes.
- **A value comparable across fixed points.** `settle_release` and `_path_gain` compare V on one continuous branch of fixed points, not across two (V omits the cavity-dependent Gaussian part of the evidence).
- **The certificate's remaining terms:** the fixed-point term and the C-variation slope (theory-ep), and the normal-means cases pinned as strict expected failures in `tests/test_scale_mixture_ep.py`.
- **Not wired, by design so far:** the external-GWAS annotations (`external_annotations`), the SV-context prior features (`sv_prior_features`), and the full-data EP route (`inference="ep"`, bench-sim's `svpgs_ep` submission), each waiting on a measured held-out gain.

## The user's pending decisions
Agents never act on these; the user decides and acts.
1. **Git history.** AoU-derived values remain in the public git history and in the `archive/2026-09-19/*` tags. Whether to rewrite history, delete the tags, or leave them is the user's decision, as is a purge of the orphaned commit `ba1b71e` (blob `0b20f6b750f8b7899a860045a5fe6a4d5e2f20c0`) from GitHub's caches. Commit `302419a`'s message also cites chr22 numbers from bench-sim's withdrawn first cohort.
2. **Attach the All of Us Controlled Tier data collection (v9)** to the SV-PGS workspace: the phenotypes, person tables, GATK-SV VCFs and the DRAGEN-to-research-ID crosswalk.
3. **Approve the store pull:** build the stores inside the imputation workspace and copy only the stores into the SV-PGS workspace, in-perimeter.

## Process
- Everything is on `main`, committed and pushed as it lands; there are no other branches. The pre-restart branches are tags (`archive/2026-09-19/*`; the old fitting path is `archive/2026-09-19/old-path-final`).
- The code is Linux-only: tests run in a Linux VM (`-m "not slow"`), and GitHub CI on main pushes is a secondary signal.
- Accuracy claims come only from bench-real, bench-sim, or AoU held-out data inside the workspace.
