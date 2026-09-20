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

## Direct read-based SV/CNV channel (specified; not yet built)

**Why.** SVs imputed from SNVs keep little of the real SV signal: median r² 0.33, and 0.00–0.77 at the CNVs that drive the signal [real: bench-sim, as reported by the lead, 2026-09-20]. So the pipeline must carry a second, direct measurement of the same people's SVs and CNVs, read from their short reads, beside the imputed long-read-panel DS. measure-path's fusion combines the two (MODEL.md §2: B = α_B + ρ_B·G + e_B).

**Status.** This section is a specification with placeholders. The driver reads none of it yet. The step lands with its synthetic tests:
- a synthetic GATK-SV VCF like `tests/test_gatksv_source.py`'s, carrying CNV/MULTIALLELIC and biallelic records, no-calls and FILTER values;
- the same synthetic cohort the dry run uses;
- no AoU data, and no workspace action by an agent.

### Inputs (software and configuration; every path a placeholder)

| Source | Product | Software and configuration | Fields read | Samples |
|---|---|---|---|---|
| B, primary | The CDR's srWGS structural-variant call set, one VCF per chromosome under `${CDR_STORAGE_PATH}/wgs/short_read/structural_variants/vcf/full/` (CDR_LAYOUT.md; the file pattern is checked against the attached release) | GATK-SV joint calling, release version as the CDR documents | INFO SVTYPE, SVLEN, END; FILTER; FORMAT GT, CN, RD_CN, and SL and GL where the header declares them | research IDs, joined to an imputed half through the crosswalk and to the long-read half by research ID, its own namespace (`gatksv_source`) |
| C, optional third source | Per-sample read-depth CNV calls, if the attached CDR carries them (DRAGEN CNV VCFs, for example) | the caller and version the CDR documents; format only: FORMAT CN per segment | segment CN | research IDs |
| Targeted loci, optional | Copy-number calls at segmental-duplication genes GATK-SV genotypes poorly, from targeted callers run on the CRAMs inside the workspace | open-source targeted callers, configured per locus; a lead decision on cost | per-locus CN | research IDs |

- **Filters:** `gatksv_source.GatksvSource`'s FILTER policy applies. It keeps PASS, plus MULTIALLELIC on copy-number records, and drops breakends, multi-ALT non-CN records and copy numbers beyond the code range, each counted by reason.
- **Which products exist** in the attached CDR is read inside the workspace from its documentation and VCF headers, never by an agent.
- **Questions for the imputation peer** (software and configuration only): whether any srWGS SV/CNV product fed the imputation, the DRAGEN version and CNV-caller configuration of the srWGS CRAMs, and the GATK-SV release version. No imputation session was reachable on 2026-09-20; the questions go through the lead.

Run-config fragment, a template (the driver doesn't read it yet):

```json
"direct_sv": {
  "gatksv_calls": "${CDR_STORAGE_PATH}/wgs/short_read/structural_variants/vcf/full/${GATKSV_FILE_PATTERN}",
  "read_depth_cnv_calls": null,
  "targeted_copy_numbers": null,
  "truth_copy_numbers": null
}
```

Each optional source is an explicit `null`, and a null source is stated in the step summary and the certificate.

### Conversion into store columns (measure-path-cn encoding)

A new step, `direct_sv`, sits between `measurement` and `fit`. It needs the imputed D* and each imputed record's r²_A, and it writes the final store the fit reads. Per chromosome:
1. **Read and align.** `GatksvSource` reads the chromosome. `GatksvBlock.aligned_to_store_samples` puts its research-ID samples in store column order.
   - An imputed half's columns come from `sample_crosswalk.source_columns_for_store_samples`.
   - The long-read half is named by research ID, the call set's own namespace, so its columns are looked up by research ID. That is not a name match across namespaces.
   - A store sample the call set lacks is a no-call.
2. **Every kept GATK-SV record becomes a store row of its own** (`gatksv_store_rows` → `GatksvRows`), with its codes, `codes_per_unit` and `value_origin` (measure-path-cn).
   - `FusedRows` is not used. Its truth-free calibration assumes classical error on DS, which draw-type DS violates (measure-path).
   - **Copy-number records** (SVTYPE=CNV or FILTER MULTIALLELIC) are class COPY_NUMBER: value = CN − modal CN, with `codes_per_unit` = ⌊254 / max CN⌋ and `value_origin` = −modal CN (`copy_number.modal_copy_numbers`, `copy_number_codes_per_unit`, `encode_copy_numbers`).
   - **Biallelic DEL, DUP, INS and CPX records** are ALT counts: `codes_per_unit` 127, `value_origin` 0.
3. **Pairs.** `sv_fusion.candidate_pairs(imputed SV sites, gatksv_sites(block))` pairs direct records with the imputed SV records of the same chromosome. The pairs are resolved one to one by |corr(DS, B)| over the fitted cohort.
4. **No-calls** are never zero. `GatksvRows` fills each with E_lin[B | DS] from the best-paired imputed DS, or with the record's observed mean where no candidate pairs. E[B | SL] replaces that where the record carries SL.
5. **Rows and annotations.**
   - The GATK-SV-only rows go into the chromosome in coordinate order, after the popped records at the same POS.
   - `group_first` merges them with overlapping bubbles and TR loci (`unbreakable_group_first`).
   - A `row_source` annotation (popped, direct) records each row's origin, and `sv_length` comes from |SVLEN| or the END span.
   - The store's sites md5 covers the merged list, and its MANIFEST records the call set's release and the FILTER policy.
6. **Arms (EVALUATION.md).** Arm A drops every SV, CN and direct row through its −inf offset. Arm C keeps them, with each fused pair's map. C-null permutes them within ancestry.

### How the channel enters the measurement model

The fusion is measure-path's `measurement_model` (lane/measure-path-model f043b26): a map fitted on the long-read truth pairs, not a truth-free calibration.
- **Each accepted pair is a two-row block.** It is `LdBlock(records=[imputed_row, direct_row], targets=[0], absorbed=[1])`, passed in `CalibrationPairs.blocks` as a `BlockPairs` with:
  - `dosage`: [truth pairs, 2], the stored values of both rows, decoded (`copy_number.decode_values`);
  - `truth`: [truth pairs, 1], the event's truth ALT count, or truth CN − modal CN for a CN row;
  - `cohort_covariance`: 2 × 2, from the fitted cohort's sums over both rows.
  The truth pairs are the long-read panel members, with B already no-call-free.
- **What the model returns.** It fits E_lin[G | D*, B] for the imputed row. It sets the direct row's offset to −inf, so the fit drops that column exactly while the map still reads it. The fused target gets its own offset and residual variance, and the certificate counts `direct_calls_fused`.
- **Persisted.** The fusion maps (targets, column means, coefficients) are saved with the LD maps. e2e applies each map A in Stage 0, Stage 2 and scoring.
  - **Open (fit-api/e2e):** `fit_model.fit` takes no map argument yet, so the step can't land before it does.
- **No truth pairs: no fusion.** Both rows stay separate columns, the step summary and the certificate say so, and the run continues. The truth-free path (`sv_fusion.calibrate_two_sources`, anchor error models, false-positive rates) assumes classical error on DS, which draw-type DS violates, so the driver never uses it.
- **Copy-number truth.** A CN pair needs a truth copy number for the long-read panel members (`truth_copy_numbers`, the long-read call set's CN at those loci if it carries one).
- **A direct row with no truth,** CN or not, takes reported r² = 1: the hard-call set's own claim of exactness, with no better number and no genotype likelihoods (measure-path). The certificate counts it under `uncalibrated_records`, with the reported r² as its source.
  - If the release carries GL for CN records, the principled value is 1 − mean_i Var(G | reads_i) / Var(posterior means), and measure-path adds that estimator. The user checks for GL in the VCF header inside the workspace.
- **The third source C** (RD_CN, or a separate read-depth call set) stays gated. measure-path hasn't validated the three-source identity (MODEL.md §2), and it isn't on their queue.
- **Moments in value units.** Every moment the measurement model sees, calibration and fitted-cohort alike, is computed on decoded values (code / codes_per_unit + value_origin). A CN row is in copies and an ALT-count row in dosage.
- **Offsets and scoring.** The fit's `log_variance_offset` covers the merged store's rows. `pooled_log_reliability` must keep a direct row's −inf, since it is −inf in every group. Scoring, Stage 0 and Stage 2 are otherwise affine-invariant per column and need no change (`copy_number` module docstring).
- **Measured** [semi-real: bench-sim v7, simulated read-depth DEL/DUP channel, reported by measure-path]: fused r² 0.77–0.80, against 0.69–0.73 for D* alone and 0.41–0.48 for the direct call alone. That is 98–99% of the in-sample linear oracle.

**Restart.** The step's key covers the call-set files by size, the crosswalk, the measurement step's key and its code. Each chromosome is a sub-checkpoint, as in the store step.
