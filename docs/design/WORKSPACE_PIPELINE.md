# The in-workspace pipeline

- **Code:** `sv_pgs/workspace_pipeline.py`, run as `sv-pgs workspace-run --config RUN_CONFIG.json [--through STEP]`.
- **Test:** `tests/test_workspace_pipeline.py`, a synthetic dry run of every step on a synthetic store and synthetic OMOP.
- **Launcher:** `launcher/workspace/`, templates only. Launching is the user's decision.

**Data class.** Everything the pipeline reads or writes is participant-derived and stays in the AoU workspace (STORE.md, "Data class"):
- Nothing is ever sent anywhere: no notifications, webhooks, uploads or messages.
- The `export` step only copies the report files the user named in the config into `<run>/export`, after checking that every count is 0 or at least 21. Moving them out of the workspace is the user's own action.
- Progress is visible only as empty marker files named `<UTC>_<step>_<event>`, where the event is `started`, `skipped`, `completed` or `failed-<exception class>`. Names never carry a value computed from data.

## Inputs

These are formats; the paths come from a run config supplied inside the workspace (template: `launcher/workspace/run_config.template.json`). Every key is required.

| Input | Format the reader expects | Checked |
|---|---|---|
| Popped imputed batches | One BCF/VCF per chromosome per batch, GT:DS:GP, INFO/ID (atomic ID), INFO/CM, INFO/SVLEN, INFO/INFO (the imputation's r², read as the measurement model's fallback, sample-weighted over batches). Batches of a half in batch order; the same samples on every chromosome | G2: POS, md5(REF\tALT) and INFO/ID of every record against the sidecar. G4: DS/GP consistency. INFO/INFO present and in [0, 1] |
| Strata sidecar v2 | `chrK.strata.tsv.gz` (header line starting `#`) with idx, pos, id, refalt_md5, ref_len, alt_len, n_paths, n_paths_total, cx; `_done/chrK.json` with `sites_md5` and `ids_md5` | The contig is used only once `_done` exists. The popped records' md5 over `CHROM\tPOS\tREF\tALT\n` and over `INFO/ID\n` must equal the sidecar's |
| bubble.split | One biallelic record per path, a bubble's paths consecutive at its POS; INFO/ID lists the path's atomic IDs split on `:` (`,` read as `:`) | For every multi-path record: one bubble carries it, with the sidecar's n_paths_total paths, n_paths of which carry its ID |
| Tandem repeats | GIAB v3.6 AllTandemRepeatsandHomopolymers_slop5 BED | Sorted, non-overlapping (`tr_loci`) |
| Truth calls | Long-read hard calls (GT) on the popped site list, one file per chromosome, samples named by research ID. **Or an explicit `null`** | The same lockstep gate as a batch |
| Crosswalk | Research ID ↔ DRAGEN sequencing ID, one to one | `SampleCrosswalk` |
| Ancestry | The CDR ancestry predictions TSV: `research_id`, `ancestry_pred`, `pca_features` | Every store sample has a label and PCs |
| Relatedness | The CDR KING pairs TSV (research IDs, kinship) | Finite kinship |
| CDR | BigQuery through `all_of_us` (`WORKSPACE_CDR`, `GOOGLE_PROJECT`) | The phenotype fingerprints and the CDR name enter the phenotypes step's key |

**The truth rows are not among the imputation's deliverables.** The panel was an input to the imputation, and only its sites-only BCF is the strata key. The user points `truth_calls` at wherever the long-read calls sit inside the workspace. With `null` the run degrades explicitly, never silently:
- there is no long-read half and there are no calibration pairs;
- the measurement model runs in its no-truth form (κ = 1, offsets from the reported r²);
- the store keeps the background-corrected DS;
- the samples, measurement and certificate records say so.

## Steps

| Step | Reads | Writes | Restart grain |
|---|---|---|---|
| samples | batch headers, crosswalk, ancestry, relatedness | typed half manifests, cohort rows (`resolve_cohort_rows`), ancestry groups, calibration pairs, trait-agnostic folds (`kinship_folds`, strata half × ancestry) | step |
| phenotypes | the CDR | one sample table per disease and trait (`all_of_us`) | step |
| cohort | samples, phenotypes | each trait's own covariates beside the structure columns every trait shares (below), the target matrix, every (trait, fold) training and held-out mask | step |
| store | samples, the genotype inputs | the dosage store: typed per-half manifests, background-corrected codes, no-call fill, variant columns, TR loci, and the calibration pairs' codes | each batch's decode; each chromosome |
| measurement | store, cohort | per-group κ, residual variances and reliability offsets (`measurement_model`); the fit's prior offset, pooled over the fit rows' ancestry groups (`pooled_log_reliability`); the store rewritten with D* where pairs exist | step |
| fit | cohort, measurement, final store | every (trait, fold) model in one `fit_model.fit` call, each projecting out its own trait's columns (`covariate_columns`), saved by `artifact.save_model` | step |
| score | fit, cohort | every model's predictions of the cohort from one store read (`artifact.predict`), and each person's own-fold held-out prediction | step |
| report | score, cohort | held-out accuracy per trait by fold, ancestry and half; counts of 1 to 20 suppressed | step |
| export | report | the approved report files in `<run>/export` | step |

**Keys and restarts.**
- A step's key digests the config it reads (small tables by content, batch files by size), its upstream keys, and the source of the modules and functions it runs.
- A step is complete when `<run>/<step>/_COMPLETE.json` holds its key. A rerun skips it, resumes `<run>/<step>.partial` from its sub-checkpoints, or sets a stale directory aside as `<step>.superseded-<key>`.
- The only deletions are the store step's decoded batch files, once their chromosome is written, and the fit's work directory once the model is saved.
- A phenotype change re-runs phenotypes onward and keeps samples and the store.

**Where the steps run.** HANDOFF.md plans to build the store inside the imputation workspace and copy only the store into the SV-PGS workspace.
- `--through store` builds it where the BCFs are.
- A run directory that already holds a completed `samples` and `store` resumes from them after the copy, as long as the config and code keys match.

## Resource plan

The derivation starts from COMPUTE.md's floor at its design workload (n = 10⁵, p = 1.7·10⁷, 105 models) and the SV_PRS quota (COMPUTE.md, "Cloud"). The in-workspace sizes are measured there and are not reproduced here.

| Step | Bound by | Machine (quota) | Derivation |
|---|---|---|---|
| samples, phenotypes, cohort, report, export | BigQuery and small tables | n2-standard-8, on demand | Negligible compute |
| store | Reading and decoding GT:DS:GP, one parse per genotype, independent per batch file; then one zstd write per genotype | n2d-standard-224 spot (n2d 5,000 / preemptible 10,000 CPUs), a persistent run disk | Measured per core [sim-only: one synthetic 20,000-record × 250-sample BCF, one pinned MSI core, median of 3]: `decode_batch` 3.3·10⁶ genotypes/s, `assemble_half` with zstd 6.3·10⁷ genotypes/s. At n·p = 1.7·10¹² that is 140 core-hours to decode and 7.5 to assemble, ≈ 40 min on 224 threads if they scale. The input read is b bytes per genotype over the mount's bandwidth (b = 4.1 for the synthetic BCF), which is measured in-workspace. **Gap:** the decode is bound by cyvcf2's per-record overhead (≈ 75 µs a record at 250 samples), not by BGZF inflate. |
| measurement | One read of the store (per-group moments over the fitted rows), plus one read and one write of the imputed halves for D* | the store's VM | Two store passes. The moments could ride the store step's assembly; that is the gap to remove |
| fit + score | Store passes on GPUs | a2-highgpu-8g, 8×A100-40 (16 on demand / 64 spot); or a3-highgpu-8g, 8×H100 spot (128 by quota, capacity unverified) | Floor ≈ 6 min on 8×A100-40 (staging-bound). Today's gap, 10–40× with Stage 1 dropped, gives ≈ 1–4 h [derived: COMPUTE.md floor × gap]. Scoring rides one extra pass |
| ultramem (m1-ultramem-160, ~3.8 TB) | — | only if the calibration moments or Stage 0 need the whole store in RAM | COMPUTE.md: for staging and Stage 0 only |

- **Disk.** The raw store is one byte per genotype, 1.7 TB at the design workload, and zstd writes less. Each chromosome's decoded batches (its records × n bytes) exist only until that chromosome is written. The D* rewrite needs a second store's worth of space.
- **Preemption.** Put the run directory on a persistent disk attached by name, so the per-batch and per-chromosome sub-checkpoints survive a spot preemption. Batch retries a preempted task; exit code 50001 is spot preemption. On local SSD a preemption loses the partial step, and only completed steps copied off the VM survive.

## Open items, stated in the step summaries

1. **The disease target is `target`.** pheno-disease's latent-onset model replaces the 0/1 rule path with a reliability-weighted target, and its tables will list every EHR participant.
2. **The fit's prior offset is pooled** over the fit rows' ancestry groups until the fit takes one per group. It is measure-path's `pooled_log_reliability`, log r² of the stacked D*: Var(D*) = Σ_g w_g (κ_g² V_g + (μ_g − μ)²) over Var(D*) + Σ_g w_g v_g. The per-group residual variances also feed scoring later. No LD-block pairs are passed, so the A-map is not built.3. **The fit needs `covariate_columns`** (fit-api), and the engine a per-model F on each model's own columns. A run whose `fit_model.fit` lacks any keyword in `FIT_KEYWORDS` is refused before its first step.
4. **The store lacks** the SV-context features (STORE.md: the storage plan is not yet written by the converter), and `tr_motif_len` and `r2_locus` in the loci table. A non-SNV record whose core overlaps a GIAB repeat interval is `str_vntr_repeat`.
5. **Batch `SECURED.ok` files are not read.** The lockstep gate checks every record of every batch.
6. **Each person's `target_reliability`** (their precision) is not passed to the fit until it takes `target_weights`.

## Covariates (lead ruling, 2026-09-19: each trait its own)

No covariate matrix is shared across traits.
- **Shared by every trait, over the cohort rows:** the intercept, the pipeline-half indicators, the genotype-source indicator and the genetic PCs. A long-read row takes the reference half (`cohort.pipeline_half_levels`), so the two indicator sets stay distinct.
- **Each trait's own columns:** its table's metadata `covariate_columns`, named `<trait>:<column>`. For a disease those are age at the end of observation, its square, its product with female sex, log(1 + pre-landmark condition dates) and sex at birth. For a quantitative trait they are mean age at measurement, its square, its product with female sex and sex at birth. Sex at birth becomes indicators over the trait's own rows.
- **Off the rows a trait observes,** its columns are 0. None of its models weights those rows, and none predicts them.
- **The fit gets the union matrix and a per-model mask.** One call keeps one Stage 0 pass and cross-trait hyperparameter pooling: X'(I − H_t)X = X'X − X'C_t (C_t'C_t)^+ C_t'X needs only the union's cross-products.
- **No rank is required.** A trait's rows can take a column's rank away (a sex-restricted disease), and each model projects onto its own columns' span (`dual_solve.covariate_whitener`).
- Every genotyped cohort row stays a row. A person no table lists has no target and is scored but never fitted.
