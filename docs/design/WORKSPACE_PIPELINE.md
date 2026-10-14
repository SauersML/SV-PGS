# The in-workspace pipeline

- **Code:** `sv_pgs/workspace_pipeline.py`, run as `sv-pgs workspace-run --config RUN_CONFIG.json [--through STEP]`.
- **Test:** `tests/test_workspace_pipeline.py`, a synthetic dry run of every step on a synthetic store and synthetic OMOP.
- **Launcher:** `launcher/workspace/`: `submit.sh` submits each job straight to Google Batch from a terminal inside the workspace, after reserving a slot in the study's budget ledger (launcher/workspace/README.md). Launching is the user's decision.

**Data class.** Everything the pipeline reads or writes is participant-derived and stays in the AoU workspace (STORE.md, "Data class"):
- Nothing is ever sent anywhere: no notifications, webhooks, uploads or messages.
- The `export` step only copies the report files the user named in the config into `<run>/export`, after checking that every count is 0 or at least 21. Moving them out of the workspace is the user's own action.
- Progress is visible only as empty marker files named `<UTC>_<step>_<event>`, where the event is `started`, `skipped`, `completed` or `failed-<exception class>`. Names never carry a value computed from data.

## Inputs

These are formats; the paths come from a run config supplied inside the workspace (template: `launcher/workspace/run_config.template.json`). Every key is required.

| Input | Format the reader expects | Checked |
|---|---|---|
| Popped imputed batches | One BCF/VCF per chromosome per batch, GT:DS:GP, INFO/ID (atomic ID), INFO/CM, INFO/SVLEN, INFO/INFO (the imputation's r², read as the measurement model's fallback, sample-weighted over batches). Batches of a half in batch order; the same samples on every chromosome. **Or plain imputed VCF/BCFs** (below) | G2: POS, md5(REF\tALT) and INFO/ID of every record against the sidecar, and each file's md5 over its `CHROM\tPOS\tREF\tALT\n` lines against the sidecar's. G4: DS/GP consistency. INFO/INFO present and in [0, 1] |
| Strata sidecar v2 | `null` for plain input. `chrK.strata.tsv.gz` (header line starting `#`) with idx, pos, id, refalt_md5, ref_len, alt_len, n_paths, n_paths_total, cx; `_done/chrK.json` with `sites_md5` and `ids_md5` | The contig is used only once `_done` exists. The popped records' md5 over `CHROM\tPOS\tREF\tALT\n` and over `INFO/ID\n` must equal the sidecar's |
| bubble.split | `null` together with `strata_directory` for plain input. One biallelic record per path, a bubble's paths consecutive at its POS; INFO/ID lists the path's atomic IDs split on `:` (`,` read as `:`) | For every multi-path record: one bubble carries it, with the sidecar's n_paths_total paths, n_paths of which carry its ID |
| Genetic map | `genetic_map`: a path pattern over `{chromosome}` to a PLINK-format map (chromosome, ID, cM, bp): the public Beagle GRCh38 maps, plink.GRCh38.map.zip, e.g. `no_chr_in_chrom_field/plink.{chromosome}.GRCh38.map`. **Or `null`**, as it is for popped input | Read only for a plain chromosome whose records lack INFO/CM. The chromosome field names the store chromosome as `chr21` or `21` (the zip ships both); bp strictly increasing, cM finite and non-decreasing |
| Tandem repeats | GIAB v3.6 AllTandemRepeatsandHomopolymers_slop5 BED | Sorted, non-overlapping (`tr_loci`) |
| Genes | The public GENCODE GTF (gencode.v38.basic.annotation.gtf.gz, EBI): gene, transcript and exon features, 1-based closed | Each record's core span gives the store's `in_gene`, `in_exon` and `log_tss_distance` (`gene_overlap`); a chromosome the GTF lacks has none |
| Truth calls | Long-read hard calls (GT) on the imputed site list, one file per chromosome, samples named by research ID. **Or an explicit `null`** | The same per-record lockstep gate as a batch, its records named as the batches' are |
| Imputation reference IDs | `imputation_reference_ids`: one column headed `research_id`, everyone whose genome was in the imputation's reference panel. **Or an explicit `null`**, which it must be without truth calls | No blank or repeated ID. A listed person's imputed row is a leave-in imputation, so the calibration pairs leave out every listed person and everyone closer than second degree (KING, the folds' bound) to one (`calibrating_pairs`). With truth calls and `null` the overlap is unchecked, so no pair is used: the measurement model takes its no-truth form, and the samples and measurement summaries and the certificate say why |
| Crosswalk | Research ID ↔ DRAGEN sequencing ID, one to one | `SampleCrosswalk` |
| Ancestry | The CDR ancestry predictions TSV: `research_id`, `ancestry_pred`, `pca_features` | Every store sample has a label and PCs |
| Relatedness | The CDR KING pairs TSV (research IDs, kinship) | Finite kinship |
| CDR | BigQuery through `all_of_us` (`WORKSPACE_CDR`, `GOOGLE_PROJECT`) | The phenotype fingerprints and the CDR name enter the phenotypes step's key |

**The truth rows are not among the imputation's deliverables.** The panel was an input to the imputation, and only its sites-only BCF is the strata key. The user points `truth_calls` at wherever the long-read calls sit inside the workspace. With `null` the run degrades explicitly, never silently:
- there is no long-read half and there are no calibration pairs;
- the measurement model runs in its no-truth form (κ = 1, offsets from the reported r²);
- the store keeps the background-corrected DS;
- the samples, measurement and certificate records say so.

### Plain imputed VCF/BCFs

With `strata_directory` and `bubble_split` both `null`, each batch file is one imputation tool's standard output (GLIMPSE2, Beagle 5, Minimac4 or IMPUTE5), one per chromosome per batch, with the same `imputed_halves` structure. `_plain_chromosome_sites` reads it; everything after the site list (classes, cores, loci, genes, decoding, assembly, the variant table) is the popped path's code.
- **Config:** the template is the popped form. For plain input set `strata_directory` and `bubble_split` to `null`, and `genetic_map` to the map pattern where the batches carry no INFO/CM.
- **Site list:** the first half's first batch's records: POS, REF, ALT, ID, INFO/SVTYPE, INFO/SVLEN, INFO/END. Every batch of every half holds the same records in the same order: each record's POS, REF/ALT and identifier are gated as it is decoded (G2), and each file's md5 over its site lines (`PlainExpectedSites.site_line`, all of those fields) must equal the first batch's.
- **Records:** biallelic only; a record with more than one ALT is an error naming the file and position (split with `bcftools norm -m-any`). Records are in POS order, on the store chromosome (CHROM `chr21` or `21`).
- **Identifier:** the ID column, or `CHROM:POS:REF:ALT` where it is `.`. Unique within a chromosome; a duplicate is an error. Truth calls name their records by the same rule.
- **Paths:** each record is its own single path (bubble −1, K = 1) with no imputation floor (m = 0), so `value_matched_background` removes nothing (q = 0). K = m = 1 would zero DS of 0.001 and 0.002, which encode to code 0 either way but would count as background. Same-POS sets and TR loci still group in `group_first`.
- **Genetic position:** INFO/CM where every record of the chromosome carries it, otherwise interpolated from `genetic_map`: linear in bp between map positions and, beyond the map's ends, at the end segment's rate. A chromosome with neither is an error naming `genetic_map` and the public map; no rate of cM per Mb is ever assumed. The store summary names each chromosome's source.
- **Reported r²:** the header declares exactly one of INFO (GLIMPSE2, IMPUTE5), DR2 (Beagle 5) or R2 (Minimac4); none or several is an error listing what it found. Every record's value must be present and nonnegative. Minimac4's R2 is a ratio of dosage variances whose estimate can exceed 1. A value above 1 enters the reported r² as 1, the bound of the quantity it estimates, before the batches are averaged. The store summary counts them (`reported_r2_above_one`), and the MANIFEST's `reported_r2` gate reads `TRUNCATED` with the count and the largest value.
- **FORMAT:** DS is required, in [0, 2]. Where a file's header declares GP, G4 checks it against DS to one unit of the written precision: 0.01 where every DS and GP of the record lies on that grid (Beagle writes 2 decimals), otherwise 0.001. Where it does not, the `G4_format` gate reads `NOT_CHECKED` with the number of batch files that wrote no GP.
- **Annotations:** there is no sidecar, so no n_paths, n_paths_total or cx. The store carries tr_locus, sv_length and the gene columns, and Stage 2's annotation design reads whichever columns the store has.
- **Symbolic ALTs** (`<DEL>`, `<INS>`, ...) are typed from SVTYPE, SVLEN and END. Their core is [POS, END) where END passes POS, and otherwise the two bases around the padding base. A record typed `copy_number` (`<CNV>`) is refused, since its DS is an ALT count.
- **MANIFEST gates:** `strata_sites_md5` and `strata_ids_md5` read `NOT_APPLICABLE: plain input has no strata sidecar`. The store step's key digests the mode, the batch files' sizes and the genetic maps' contents.

## Steps

| Step | Reads | Writes | Restart grain |
|---|---|---|---|
| samples | batch headers, crosswalk, ancestry, relatedness, imputation reference IDs | typed half manifests, cohort rows (`resolve_cohort_rows`), ancestry groups, calibration pairs and the ones the measurement model uses (`calibrating_pairs`), trait-agnostic folds (`kinship_folds`, strata half × ancestry) | step |
| store | the samples step's half manifests, groups and calibration pairs, the genotype inputs | the dosage store: typed per-half manifests, background-corrected codes, no-call fill, variant columns, TR loci, and the calibration pairs' codes; the store summary counts the records the 8-bit codes made monomorphic (`encoding_flattened`) | each batch's decode; each chromosome |
| phenotypes | the CDR | one sample table per disease and trait (`all_of_us`) | step |
| cohort | samples, phenotypes | each trait's own covariates beside the structure columns every trait shares (below), the target matrix, every (trait, fold) training and held-out mask | step |
| measurement | store, cohort | per-group κ, residual variances and reliability offsets (`measurement_model`); the one model the fit reads, pooled over the fit rows' ancestry groups (`pooled_measurement_model`); the store rewritten with D* where pairs exist | step |
| direct_sv | samples, cohort, measurement, the GATK-SV call set (`direct_sv`) | the final store the fit reads: every kept GATK-SV record a row of its own beside the imputed records, its no-calls filled, and the measurement model over the merged rows (below); with `direct_sv` or its call set null, the measurement step's store and model unchanged | each chromosome |
| fit | cohort, measurement, final store | both arms of every trait (below): per (arm, trait) one `fit_model.fit` call holding the trait's fold models and its all-rows model, each projecting out its own trait's columns (`covariate_columns`), saved by `artifact.save_model` | each (arm, trait) |
| score | fit, cohort | every model of both arms scored for the cohort from one store read (`artifact.predict_models`), with the structural part of each trait's all-rows `with_sv` model (`artifact.model_part`) in the same read, and each person's own-fold held-out linear predictor, genetic score and predictive mean | step |
| effects | fit, score, cohort, final store | each trait's all-rows model in both arms: every fitted record's effect per allele (or per copy), posterior SD and P(effect > 0) (`effects/<trait>.<arm>.tsv.gz`, workspace only), summaries by class and length (`effects_by_class.tsv`, `effect_length_correlation.tsv`), and the structural records' share of the `with_sv` score's variance (`genetic_variance.tsv`) | step |
| report | score, cohort | held-out accuracy of both arms (`accuracy.tsv`) and their paired comparison (`comparison.tsv`) per trait, by fold, ancestry and half; with the config's frozen `panel`, the pre-registered test (`panel.tsv`, EVALUATION.md (a) and (f)): each trait's one-sided paired test on the primary fold, Hommel-adjusted, and the panel's w'z / sqrt(w'Rw); counts of 1 to 20 suppressed | step |
| export | report, effects | the approved report and effects summaries in `<run>/export` | step |

**Arms.** Every trait is fitted twice on the same people, folds and seeds (`workspace_pipeline.ARMS`):
- `snv_indel`: every record shorter than 50 bp (SNVs, short indels, short repeat-length changes), and no copy-number record and no direct read-based row (`direct_sv`). The other records get a −inf log reliability, which the fit drops exactly.
- `with_sv`: every record.

The arms differ by exactly the structural variation of the study's definition (≥ 50 bp, by the store's `sv_length`, plus copy number). Both arms of a trait share its seed, so their probes and posterior draws are common random numbers.

**Models.** Per trait, one model per fold (trained without that fold and scored on it) and one model on every observed row (`<trait>/all`). The all-rows model is the released score and the effects step's source; the fold models only measure held-out accuracy (SPEC: the fit itself never holds rows out). A run of T traits fits 2 · T · (folds + 1) models.

**Likelihood.** With `disease_likelihood: "gaussian_working"`, the template's value, every model is fitted on the Gaussian likelihood (`FIT_TRAIT_TYPE`), a disease's 0/1 status included: a linear probability working model. A disease's coefficients are then per-allele changes in the probability of recorded status, not log odds, and its raw fitted values (linear predictor, predictive mean) are not probabilities and can leave [0, 1]. `disease_likelihood: "bernoulli"` fits diseases on the logistic likelihood's Polya-Gamma bound (`binary_likelihood`) instead; the template does not select it, for cost: its weights move with its sites, so every sweep re-reads the whole store, hundreds of store passes per model. The effects tables name each model's likelihood and effect scale (`likelihood`, `effect_scale`).

**LD route.** Every model's likelihood is the full data's: Stage 2 reads the samples through `DualGaussian` (`stage2_wiring._fit_one`). For a quantitative mean-field model, Stage 0's banded Gram also accelerates the sweeps where a pass over it is measured cheaper than a pass over the samples on this store and device (`gram_space.pass_costs`). Those sweeps hold the far field the band omits fixed, and the exact fields are read from the samples again until that far field stops moving the fixed point (`full_data_fit._GramMeanField`), so the band changes the cost, not the target. The choice depends on measured timing, so two runs on different hardware can take different paths; either path ends at a fixed point of the full-data problem, to the solve's tolerance, though not necessarily the same one where the mean-field objective has several. The summary-only band, which treats the far field as noise (`GramGaussian`), is no route's; a model whose certificate reports a summary-only far-field scale (`far_field_scale` nonzero) is not scored. A full-data far-field diagnostic can be nonzero without indicating truncation.

**Report.**
- **Frozen calibration.** A disease's held-out probabilities come from a logistic calibration of each arm's linear predictor, an intercept and a slope (`score/calibrators.json`), fitted on reserved development families and frozen before evaluation. The template reserves 10% of each outer development fold (`calibration_fraction=0.1`); neither their scores nor labels depend on the outer test labels. Other-fold out-of-fold predictions are not independent calibration data, because those models trained on the evaluated fold.
- `accuracy.tsv` (trait, arm, cell, level, n, cases, metric, value). A disease gets the AUC of its linear predictor (`auc`) and the AUC of its genetic score alone (`pgs_auc`), and with frozen calibration the log loss and Brier score of its calibrated probability (`log_loss`, `brier`) plus diagnostic calibration intercept and slope. Each metric reports the number of people actually evaluated; proper scores include single-class folds, whereas AUC requires both classes. A quantitative trait gets the r² of its linear predictor and the partial r² of its genetic score given the trait's covariates.
- `comparison.tsv` (trait, cell, level, n, cases, metric, estimate, standard_error, z, p_value, inference), with positive values favouring `with_sv`, per cell of all held-out rows and per ancestry group. A disease gets the DeLong paired delta AUC of the linear predictors and of the genetic scores; with frozen calibration, the family-clustered paired gains in log loss and Brier score (`log_likelihood_gain`, `brier_reduction`). The legacy test-recalibrated statistic is excluded from this report. A quantitative trait gets the family-clustered delta partial r² of the genetic scores (`held_out_comparison`).
- Every value is each fold's own model on its own fold, weighted by the fold's rows: never one pooled correlation of the folds' differently scaled predictors. The comparison's standard error adds the folds' in quadrature, conditional on the fitted models; it leaves out the cross-fold pair term (their shared training rows), which needs each arm's posterior covariance. So a comparison over more than one fold leaves z and p empty, with inference `descriptive_cross_fit`.
- A cell shows values only with at least 21 rows and, for a disease, at least 21 cases and 21 controls. A case count is written only where neither it nor the controls it implies is 1 to 20.

**Effects summaries** count records, never participants. `effects_by_class.tsv` gives, per (trait, arm, class, length bin), the records, how many have a positive and a negative posterior mean, the posterior-expected counts of positive and negative effects (Σ P(β > 0), Σ P(β < 0)), and the mean and median absolute effect and the root-mean-square effect per unit. Length bins are `<50` and then decades from 50 bp. `effect_length_correlation.tsv` gives Spearman's ρ between log length and, separately, the absolute posterior mean and the posterior SD, per structural class and over every structural record; it carries no p-value, since LD and the prior's length features make the records dependent (`interpretation` reads `descriptive_prior_and_LD_dependent`). Every moment is under the law the model's draws come from: the fitted product approximation on the mean-field route (`fast_scoring`).

**Structural share of the score's variance.** A sum of per-record squared effects is not the score's variance under LD: Var(Xβ) = β′Σβ. `genetic_variance.tsv` (trait, n, quantity, value) decomposes it from the scores themselves, over each trait's observed rows: the variance of the all-rows `with_sv` genetic score, of its structural part (the same model on its structural records alone, scored in the score step's read), of the small part (the rest), the covariance term 2 Cov(structural, small) that carries their LD, and the structural share Cov(structural, total) / Var(total). Every value aggregates the trait's n rows; a trait observed on 1 to 20 rows shows neither its count nor a value.

**Keys and restarts.**
- A step's key digests the config it reads (small tables by content, batch files by size), its upstream keys, and the source of the modules and functions it runs.
- A step is complete when `<run>/<step>/_COMPLETE.json` holds its key. A rerun skips it, resumes `<run>/<step>.partial` from its sub-checkpoints, or sets a stale directory aside as `<step>.superseded-<key>`.
- The only deletions are the store step's decoded batch files, once their chromosome is written, the direct_sv step's blocks of rows, once their chromosome is written, and the fit's work directory once the model is saved.
- A phenotype change re-runs phenotypes onward and keeps samples and the store.

**Where the steps run.** HANDOFF.md plans to build the store inside the imputation workspace and copy only the store into the SV-PGS workspace.
- `--through store` builds it where the BCFs are.
- A run directory that already holds a completed `samples` and `store` resumes from them after the copy, as long as the config and code keys match.

## Interpreting the outputs

- **The disease target** is the 0/1 `target`: ever-recorded EHR status, a case on evidence at any age up to the end of observation (prevalent and incident alike) against a control with none (PHENOTYPES.md). It is not incident risk over a horizon.
- **Scale.** On the working likelihood (above) a disease coefficient is a per-allele (per-copy) change in the probability of recorded status, not a log odds ratio, and raw fitted values are not probabilities; only the frozen calibration's outputs are. The effects tables' `effect_scale` says which scale a model's effects are on.
- **Signs.** Each effect is per ALT allele, so its sign follows which allele the input calls ALT. The prior is symmetric in sign, so a class's balance of positive and negative effects (`effects_by_class.tsv`) is what the prior predicts whatever the biology: weak evidence either way.
- **Not causal.** A coefficient is the joint fit's share of the signal among correlated records under the prior, not a causal effect; under LD a large effect can stand for a neighbour.
- **What a `with_sv` gain means.** An imputed SV dosage is a function of the nearby SNVs and indels the imputation read, so it carries no genetic information the `snv_indel` arm's data lack. A gain means the SV columns represent the haplotypes better in a linear model. New information needs SVs called from reads (the long-read half's hard calls, or the direct channel's GATK-SV rows below).
- **The baseline** is `snv_indel`: SNVs, short indels and short repeat-length changes (every record under 50 bp), not SNVs alone.
- **Folds share training rows** (with K folds, any two training sets share (K − 2)/(K − 1) of each), so their held-out errors are correlated. Fold-combined comparisons are descriptive: their standard error leaves that covariance out, and z and p are left empty. `test_recalibrated_log_likelihood_gain` recalibrates on the rows it scores, so it compares discrimination and shape, not probabilities.
- **Certificate terms and acceptance.** `outer_criterion_met` and the other terms say the outer loop met its numerical stopping criterion on the computed objective, a directional Laplace approximation with leading-order corrections (CAPABILITIES.md); the acceptance decision a model carries checks those terms. Neither bounds the error of the evidence or of the posterior, and posterior SDs and P(effect > 0) are under the law the draws come from (Effects summaries).
- **Encoding.** A record whose only nonzero dosages are 0.003 or less is stored monomorphic and is in neither arm; the store summary counts these (`encoding_flattened`).

## Resource plan

The derivation starts from COMPUTE.md's floor at its design workload (n = 10⁵, p = 1.7·10⁷, 105 models) and the SV_PRS quota (COMPUTE.md, "Cloud"). The in-workspace sizes are measured there and are not reproduced here.

| Step | Bound by | Machine (quota) | Derivation |
|---|---|---|---|
| samples, phenotypes, cohort, report, export | BigQuery and small tables | n2-standard-8, on demand | Negligible compute |
| store | Reading and decoding GT:DS:GP, one parse per genotype, independent per batch file; then one zstd write per genotype | n2d-standard-224 spot (n2d 5,000 / preemptible 10,000 CPUs), a persistent run disk | Measured per core [sim-only: one synthetic 20,000-record × 250-sample BCF, one pinned MSI core, median of 3]: `decode_batch` 3.3·10⁶ genotypes/s, `assemble_half` with zstd 6.3·10⁷ genotypes/s. At n·p = 1.7·10¹² that is 140 core-hours to decode and 7.5 to assemble, ≈ 40 min on 224 threads if they scale. The input read is b bytes per genotype over the mount's bandwidth (b = 4.1 for the synthetic BCF), which is measured in-workspace. **Gap:** the decode is bound by cyvcf2's per-record overhead (≈ 75 µs a record at 250 samples), not by BGZF inflate. |
| measurement | One read of the store (per-group moments over the fitted rows), plus one read and one write of the imputed halves for D* | the store's VM | Two store passes. The moments could ride the store step's assembly; that is the gap to remove |
| direct_sv | Parsing the call set's VCF (every FORMAT field of every sample), then one read and one write of the store | the store's VM, one process per chromosome | R·N / 1.2·10¹⁰ core-hours to parse R records of N samples at the store's measured GT:DS:GP rate, 8.4–35 at R = 10⁶ and N = 10⁵–4.1·10⁵, up to 31–127 with GATK-SV's longer FORMAT [est; R and N assumed, not read from the CDR], plus about 7.5 to rewrite the store at the design workload ("Direct read-based SV/CNV channel", Cost) |
| fit + score | Store passes on GPUs | a2-highgpu-8g, 8×A100-40 (16 on demand / 64 spot); or a3-highgpu-8g, 8×H100 spot (128 by quota, capacity unverified) | Floor ≈ 6 min on 8×A100-40 (staging-bound). Today's gap, 10–40× with Stage 1 dropped, gives ≈ 1–4 h [derived: COMPUTE.md floor × gap]. Scoring rides one extra pass |
| ultramem (m1-ultramem-160, ~3.8 TB) | — | only if the calibration moments or Stage 0 need the whole store in RAM | COMPUTE.md: for staging and Stage 0 only |

- **Disk.** The raw store is one byte per genotype, 1.7 TB at the design workload, and zstd writes less. Each chromosome's decoded batches (its records × n bytes) exist only until that chromosome is written. The D* rewrite needs a second store's worth of space.
- **Preemption.** Put the run directory on a persistent disk attached by name, so the per-batch and per-chromosome sub-checkpoints survive a spot preemption. Batch retries a preempted task; exit code 50001 is spot preemption. On local SSD a preemption loses the partial step, and only completed steps copied off the VM survive.

## Open items, stated in the step summaries

1. **The disease target is `target`.** pheno-disease's latent-onset model replaces the 0/1 rule path with a reliability-weighted target, and its tables will list every EHR participant.
2. **The fit gets one measurement model, pooled** over the fit rows' ancestry groups, until it takes one per group. This is measure-path's `pooled_measurement_model`, saved as `measurement/measurement_model.npz` and read back with `MeasurementModel.load`. Its log reliability is the stacked D*'s r²: Var(D*) = Σ_g w_g (κ_g² V_g + (μ_g − μ)²), over Var(D*) + Σ_g w_g v_g. No LD-block pairs are passed yet, so the A-map is not built.
3. **Every model projects out its own columns (`covariate_columns`, fit-api).** The engine still needs a per-model F on each model's own columns (e2e). The fit step passes one `fit_model.FitRequest`, whose own constructor checks every field, so a run that cannot build the fitter's input fails where it builds it and there is no keyword list to drift from the fitter's signature.
4. **The store lacks** the SV-context features (STORE.md: the storage plan is not yet written by the converter), and `tr_motif_len` and `r2_locus` in the loci table. A non-SNV record whose core overlaps a GIAB repeat interval is `str_vntr_repeat`.
5. **Batch `SECURED.ok` files are not read.** The lockstep gate checks every record of every batch.
6. **`target_variance` is None,** so every target is taken as measured exactly. Each person's Gaussian-site variance comes once deslop-hygiene exports it and e2e's heteroscedastic noise exists (review-stats §7). A (target, target_reliability) pair is never passed as an outcome.

## Covariates (lead ruling, 2026-09-19: each trait its own)

No covariate matrix is shared across traits.
- **Shared by every trait, over the cohort rows:** the intercept, the pipeline-half indicators, the genotype-source indicator and the genetic PCs. A long-read row takes the reference half (`cohort.pipeline_half_levels`), so the two indicator sets stay distinct.
- **Each trait's own columns:** its table's metadata `covariate_columns`, named `<trait>:<column>`. For a disease those are age at the end of observation, its square, its product with female sex, log(1 + pre-landmark condition dates) and sex at birth. For a quantitative trait they are mean age at measurement, its square, its product with female sex and sex at birth. Sex at birth becomes indicators over the trait's own rows.
- **Off the rows a trait observes,** its columns are 0. None of its models weights those rows, and none predicts them.
- **The fit gets the union matrix and a per-model mask.** One call keeps one Stage 0 pass and cross-trait hyperparameter pooling: X'(I − H_t)X = X'X − X'C_t (C_t'C_t)^+ C_t'X needs only the union's cross-products.
- **No rank is required.** A trait's rows can take a column's rank away (a sex-restricted disease), and each model projects onto its own columns' span (`dual_solve.covariate_whitener`).
- Every genotyped cohort row stays a row. A person no table lists has no target and is scored but never fitted.

## Diseases under the disease measurement model (specified; lands after lane/pheno-disease-model)

pheno-disease's disease model (docs/design/math/disease_measurement.md) replaces 0/1 case status with a latent onset read through the EHR. The pipeline changes as follows:

- **The target.** Each person gets a Gaussian site on the liability scale: precision-mean s_i (`liability_score`) and precision R_i (`target_reliability`), which can be ≤ 0 and is never clipped.
  - Until the fit takes the site, a disease is a QUANTITATIVE fit on `target` = s/mean R, unweighted, with `target_variance` None. That is the unbiased engine target, and fit-api agrees.
  - A binary fit would refuse these targets, since they aren't 0/1.
  - fit-api has proposed `target_precision` (signed) to e2e for the site proper.
- **Fold-strict disease fits** (lead ruling, EVALUATION.md claim (a)). The disease model's parameters are fitted on each training fold only, and the PGS of model (disease, k) is trained on the sites of that training-fold fit. So:
  - The phenotypes step reads the samples step's frozen folds; it runs after `samples` and keys on it.
  - Per disease there is one CDR query and one disease-model fit per training fold. Every other fold's persons are scored under that fold's parameters.
  - A disease's targets differ by fold. The cohort's target matrix is [rows, models], not [rows, traits], and the fit already takes it per model.
- **Kept in the workspace** for the exact held-out score: each disease's per-person records (`PersonRecord`), and each training fold's fitted parameters and knots. The TSV alone isn't enough.
- **The report's disease score** is each held-out person's exact log-likelihood of their own records under their fold's training-fold fit, with the arm's held-out linear predictor f_i as a liability shift: `disease_posterior_at(persons, knots, parameters, working_bytes, liability_shift=f)`.
  - The gain is ℓ_i(η_i + f_i) − ℓ_i(η_i + f_0,i), in nats, where f_0 is the same fold's fit on the trait's covariates and PCs alone. It is averaged over folds with weights n_f/n.
  - The exact ℓ_i is used, never the site's quadratic, since a negative R makes the quadratic unbounded.
  - `onset_probability` is display-only and enters no score.
- **Needed from other lanes before this lands:**
  - pheno-disease (agreed, being built): `prepare_all_of_us_disease_folds(disease, output_dir, fold_of_person, *, client=None)`, with loaders `load_disease_fold(output_dir, k) -> (knots, parameters)` and `load_disease_records(output_dir)`.
    - It runs the queries once and fits once per training fold, writing each fold's sites TSV, metadata and parameters, and every person's records.
    - EHR persons without genotypes enter every training-fold fit (pheno-disease's call: θ_M is phenotype-only). Fold k's fit never sees a record of a fold-k person.
    - The K fits are independent, so the driver runs them in parallel within the step's budget. The per-fit time is to be measured on a synthetic 10⁵ cohort [sim-only].
  - fit-api (agreed, pending e2e): f_0 as `predict`'s `linear_predictor_0` = [1, C] `null_alpha`. That is the covariate-only optimum under the same likelihood, zero off each model's covariate_columns, stored in the FittedModel.
    - **Interim,** while the target is the unweighted Gaussian one: the driver computes f_0 as least squares of the target on [1, C_t] over each model's training rows. That is exactly what the engine would return, and the report labels it as the interim form.
  - novel-inference: the disease pair term. Until then the variance is conservative (EVALUATION.md, Pending).

## Direct read-based SV/CNV channel (built for the GATK-SV call set; the other sources are not)

**Why.** SVs imputed from SNVs keep little of the real SV signal: median r² 0.33, and 0.00–0.77 at the CNVs that drive the signal [real: bench-sim, as reported by the lead, 2026-09-20]. So the pipeline must carry a second, direct measurement of the same people's SVs and CNVs, read from their short reads, beside the imputed long-read-panel DS. measure-path's fusion combines the two (MODEL.md §2: B = α_B + ρ_B·G + e_B).

**Status.** The `direct_sv` step (`workspace_pipeline._direct_sv_step`), between `measurement` and `fit`, reads the GATK-SV call set and writes the store and the measurement model the fit reads. It has run only on synthetic inputs: a synthetic GATK-SV VCF with a multi-allelic copy-number record, biallelic records, no-calls and failing FILTER values, on the dry run's cohort (`tests/test_workspace_pipeline.py::test_a_gatksv_call_set_adds_direct_rows_that_only_with_sv_fits`, `test_a_null_gatksv_call_set_adds_no_row_and_says_so`, `test_direct_sv_refuses_a_source_it_does_not_read`; `tests/test_gatksv_source.py`, `tests/test_gatksv_store_rows.py`). It has not run on AoU data. Not built: the optional sources (C, Ctyper, cn_estimator, truth copy numbers), the E[B | SL] fill, and the fusion maps (below).

### Inputs (software and configuration; every path a placeholder)

| Source | Product | Software and configuration | Fields read | Samples |
|---|---|---|---|---|
| B, primary (read) | The CDR's srWGS structural-variant call set, one VCF per chromosome: `${CDR_STORAGE_PATH}/wgs/short_read/structural_variants/vcf/full/AoU_srWGS_SV.v8.{chromosome}.vcf.gz` (CDR_LAYOUT.md; the user checks the pattern against the attached release) | GATK-SV joint calling, release version as the CDR documents | INFO SVTYPE, SVLEN, END; FILTER; FORMAT GT, CN with its CNQ, and RD_CN with its RD_GQ where CN is missing. SL is not read. The public 1kGP freeze V3's CNV records carry GT, GQ, RD_CN, RD_GQ, PE_GT, PE_GQ, SR_GT, SR_GQ, EV, CN and CNQ, with no GL or PL (measure-path); the user checks the attached release's header inside the workspace | research IDs, joined to an imputed half through the crosswalk and to the long-read half by research ID, its own namespace (`call_set_columns`) |
| C, optional third source (not read) | Per-sample read-depth CNV calls, if the attached CDR carries them (DRAGEN CNV VCFs, for example) | the caller and version the CDR documents; format only: FORMAT CN per segment | segment CN | research IDs |
| Ctyper, optional (not read) | Paralog-specific integer copy numbers per pangenome-allele subgroup, aggregated per gene: 3,351 CNV genes plus 273 medically relevant genes. It gives no per-call confidence. | Ctyper (Nature Genetics 2025; github.com/ChaissonLab/Ctyper) with its database (Zenodo 13381931), which the user stages into the workspace; run from CRAM by the user | per sample, per gene or subgroup: integer CN | research IDs |
| cn_estimator, optional (not read) | Fractional GC-corrected normalized copy number per sample and region (4 decimals), `${sample}.cn.txt.gz`. Its per-locus clustering into integer states, with a silhouette score per locus rather than per person, is not used | cn_estimator (github.com/pgarg-tools/cn_estimator, MIT), from CRAM plus a BED of regions (`${CN_REGIONS_BED}`); run by the user | per sample, per region: fractional CN | research IDs |

- **Filters:** `gatksv_source.GatksvSource`'s FILTER policy applies. It keeps PASS, plus MULTIALLELIC on copy-number records, and drops breakends, multi-ALT non-CN records and copy numbers beyond the code range, each counted by reason (the summary's `skipped_records`).
- **Which products exist** in the attached CDR is read inside the workspace from its documentation and VCF headers, never by an agent.
- **Questions for the imputation peer** (software and configuration only): whether any srWGS SV/CNV product fed the imputation, the DRAGEN version and CNV-caller configuration of the srWGS CRAMs, and the GATK-SV release version. No imputation session was reachable on 2026-09-20; the questions go through the lead.

The run config's `direct_sv` key is required (`launcher/workspace/run_config.template.json`):

```json
"direct_sv": {
  "gatksv_calls": "${CDR_STORAGE_PATH}/wgs/short_read/structural_variants/vcf/full/AoU_srWGS_SV.v8.{chromosome}.vcf.gz",
  "read_depth_cnv_calls": null,
  "ctyper_copy_numbers": null,
  "cn_estimator_copy_numbers": null,
  "truth_copy_numbers": null
}
```

- `gatksv_calls` is a path pattern over `{chromosome}`, or null.
- Every other key must be null. A non-null value is refused when the config is read ("not implemented"), never ignored.
- `direct_sv: null` or `gatksv_calls: null` makes the step a stated no-op: no row is added, the measurement step's store and measurement model pass through unchanged, and the step's summary and `certificate.json` name which key was null. The certificate lists each optional source as not read.
- The job must be able to open the files. The Batch job templates mount only the ops bucket and the run disk, so the pattern names a mounted copy of the CDR bucket or a copy on the run disk.

**What each optional caller costs to run,** at the published rates (both are the user's decision, and both run on the user's own compute):
- **Ctyper:** about 1.5 CPU-hours and ~20 GB RAM per genome from CRAM (published), so n genomes cost 1.5·n CPU-hours [derived from the published rate].
- **cn_estimator:** about $0.01 per sample (published).

**Benchmark caveat.** Ctyper's database includes HPRC, HGSVC and 1kGP-overlapping samples. Any public benchmark of it must leave those samples out.

### Conversion into store columns (built; measure-path-cn encoding)

The step reads the measurement step's store (D* where it recalibrated) and its pooled measurement model, and writes `direct_sv/store` with `direct_sv/measurement_model.npz`. Each chromosome runs in a process of its own with its share of the host (`_direct_chromosome`):
1. **Read and align.** `GatksvSource` reads the chromosome's VCF in blocks of records that hold half the process's host share. Every record's CHROM must name the store chromosome (`chr21` or `21`).
   - `GatksvBlock.aligned_to_store_samples` puts the call set's samples in store column order (`call_set_columns`). An imputed half's columns come through the crosswalk (`sample_crosswalk.source_columns_for_store_samples`). The long-read half is named by research ID, the call set's own namespace, so its columns are looked up by research ID; no name is compared across namespaces.
   - A store sample the call set lacks is a no-call.
2. **Every kept GATK-SV record that some store sample is called at becomes a store row of its own** (`gatksv_store_rows.gatksv_block_rows`). A record no store sample is called at carries nothing and is dropped (`uncalled_records`).
   - **Copy-number records** (SVTYPE=CNV or FILTER MULTIALLELIC) are class COPY_NUMBER: value = CN − modal called CN, with `codes_per_unit` = ⌊254 / max called CN⌋ and `value_origin` = −modal CN (`copy_number`). Each called copy number keeps the phred quality of the field it came from: CNQ for CN, RD_GQ for RD_CN.
   - **Biallelic DEL, DUP, INS, INV and CPX records** are ALT counts: `codes_per_unit` 127, `value_origin` 0.
   - A direct row keeps its GATK-SV class and joins no tandem-repeat locus.
3. **Pairs.** The imputed records a direct row may pair with are the site list's non-SNV records whose REF and ALT are both sequence (`imputed_sv_sites`). The store keeps no REF/ALT, so the step reads the site list again from the first batch, as the store step read it. A symbolic imputed record (plain input) pairs with nothing.
   - `sv_fusion.candidate_pairs(imputed sites, gatksv_sites(block))` pairs each block's records with them. An imputed record smaller than `MINIMUM_SIZE_RATIO` times the block's smallest record cannot pair and is not read.
   - The evidence is the Fisher z of corr(DS, B) over the record's called samples, from the stored codes over every store sample.
   - Pairs with z ≥ 5 are resolved one to one in decreasing z over the whole chromosome (`one_to_one`), so an imputed record pairs with at most one direct row across blocks.
4. **No-calls** are never zero. Each is filled with E_lin[B | DS] from the candidate imputed DS with the largest z and a defined slope, or with the record's observed mean where there is none. A fill outside the stored range is clipped to it and counted (`clipped_entries`). E[B | SL] is not built.
5. **Rows and annotations.**
   - Each direct row goes into the chromosome in coordinate order, after the imputed records at its POS. A call set out of POS order is refused.
   - `group_first` keeps unbroken (`unbreakable_group_first`): the imputed rows' own groups, every same-POS set of the merged rows, and each one-to-one pair. So an engine block never splits a pair.
   - `row_source` (legend `imputed`, `direct`) records each row's origin. As a store annotation it enters the prior design like the others, and the fit's empirical Bayes learns its weight.
   - `sv_length` is |SVLEN|, or the END span where SVLEN is absent. The gene columns come from the record's core: [POS, END) where END passes POS, else the two bases around POS (`store_core_spans`). The genetic position is interpolated linearly in bp between the imputed records, and beyond them at the end segment's rate. The imputed input's own annotations (`n_paths`, `n_paths_total`, `cx`) are missing on a direct row.
   - The IDs are the call set's. `ref_len` and `alt_len` are the lengths of the REF and ALT text.
   - The sites md5 and the feature identity cover the merged list. The MANIFEST's `direct_calls` records the caller, each chromosome's file name and size, the FILTER policy and the row count. The files' contents enter the step's key.
6. **Arms.** `structural_records` counts every direct row as structural, whatever its length. A GATK-SV insertion of unknown length writes SVLEN −1, for example. So `snv_indel` gives every direct row a −inf log reliability, and `with_sv` fits them at their offsets. The effects tables carry each record's `row_source` beside its class and length. The C-null arm (EVALUATION.md) is not built.

### How the channel enters the measurement model (built: the no-truth form, no fusion)

- **No fusion.** `fit_model.fit` takes no leakage map, so no map is fitted. Each pair stays two columns, kept in one unbreakable group. The certificate says `direct_calls_fused: 0`.
- **Precondition for any fusion: truth pairs.** These are long-read panel members with an imputed D*, a GATK-SV call and a long-read truth at the imputed site. The template sets `truth_calls: null`, so the study's config cannot fuse whatever the code does.
- **The separate-rows form is sound meanwhile.** The span of {D*, B} contains E_lin[G | D*, B], so the joint regression can reach the fused predictor, and `group_first` keeps each pair in one block, so the engine sees their correlation exactly. Predictions are not biased. Its costs:
  - **Efficiency:** the pair's split of the effect is learned from y, not fixed by truth pairs, so a weak effect's split is unresolved and the sparse prior tends to put it on one column.
  - **Prior:** one event gets two r²-scaled priors, about (r²_A + r²_B) τ² rather than τ², which the empirical Bayes τ² partly absorbs.
  - **Interpretation:** an event shows two effects in the effects tables, and neither column's P(effect > 0) is the event's.
  - **Mean field:** the default route's product approximation understates a strongly correlated pair's posterior variance and can favour one column.
- **What fusion would gain:** genotype r², not measured PGS accuracy: bench-sim's fused 0.77–0.80 against 0.69–0.73 for D* alone (Measured, below), on paired events only: roughly 1–10 kb DEL, DUP and INS. A copy-number row never pairs (no imputed record is a copy number), so the CNVs gain nothing from fusion; they already enter as columns of their own.
- **Two routes, neither built:**
  - **(B) in the store, the one to build first if `truth_calls` becomes available.** A pooled pair map E_lin[T | D*, B] (`fit_leakage_map` on the two columns) writes the fused value into the imputed record's row: from D* and B where B is called, from D* alone (κ_A) where it is not, which a column map cannot express. The target gets its mapped share as its offset and the direct row −inf. No fit change is needed.
  - **(A) in the fit, the spec's A = I + E.** `FitRequest` takes the maps; Stage 0 forms A'GA; the Stage 2 operator applies X A; scoring and effects fold A into the coefficients. It is needed only for the block-wide leakage map or for artifacts that score new genomes from D* and B.
- **Each direct row's reliability** is the offset the fit uses. Every direct row is uncalibrated: no truth pair is formed for a GATK-SV row. Its reported r² goes through `fit_measurement_model`'s no-truth path in each ancestry group: offset log r², and −inf where the column does not vary over the group's fitted rows. That path uses the group's fitted-row moments in the row's own units (copies for a copy number). `pooled_measurement_model` then pools the groups, as the measurement step pools the imputed records. The reported r² (`gatksv_store_rows.reported_reliability`, over the store's called samples) is:
  - **a hard-call ALT count:** 1, the call set's own claim of exactness, with no genotype likelihoods to give a better number;
  - **a copy number:** the CNQ bound. Each called sample's miscall probability is ε_i = 10^(−Q_i/10), with Q_i the phred quality of its copy number, and a miscall moves the copy number by at least one copy. That gives r² ≤ 1 − Σ_i ε_i / (n · Var(called CN)), an upper bound;
  - **0, so a −inf offset that leaves the row out:**
    - a row whose called samples all share one value (the 0/0 case, whose variation would come only from fills; lr-sv's reliability curve is not in the code, so none is used);
    - a copy number whose bound is at or below 0;
    - a copy number with a called sample whose copy number has no quality.

  It never defaults to 1. The summary and certificate count the rows under each rule (`reliability_rules`, `excluded_without_reliability`) and state the rule in full (`reliability`).
- **The merged model** (`merged_measurement_model`) holds the measurement step's pooled model at the imputed rows and the direct rows' pooled model at theirs. Its certificate keeps both. It is saved as `direct_sv/measurement_model.npz`, and `_final_measurement_model` names it for the fit; `_final_store` names the store.
- **Specified, not built:**
  - **The truth-pair fusion maps.** Each block goes to `pooled_measurement_model(models, means, variances, counts, blocks)` as a `BlockPairs`, and the maps are fitted once on the pooled model, never per group. They fit E_lin[G | D*, B] for the imputed row and give the absorbed direct row −inf, so that pooling keeps a direct row's −inf in every group. This needs truth pairs for direct rows and a map argument in `fit_model.fit` (fit-api/e2e). The truth-free path (`sv_fusion.calibrate_two_sources`) assumes classical error on DS, which draw-type DS violates, so it is never used.
  - **Copy-number truth** (`truth_copy_numbers`) for a CN pair.
  - **Ctyper and cn_estimator rows.** These would be COPY_NUMBER rows over a gene or BED region. A Ctyper row's no-truth r² would come from the published accuracy (ρ = 0.996 against HPRC assemblies, 99.1% CN agreement leave-one-out [published: Ctyper, Nature Genetics 2025]; worse in subtelomeric and sex-chromosome matrices), and its conversion into an r² is measure-path's. A cn_estimator row would be calibrated on truth pairs.
  - **The third source C** (RD_CN, or a separate read-depth call set). measure-path hasn't validated the three-source identity (MODEL.md §2).
- **Measured** [semi-real: bench-sim v7, simulated read-depth DEL/DUP channel, reported by measure-path]: fused r² 0.77–0.80, against 0.69–0.73 for D* alone and 0.41–0.48 for the direct call alone. That is 98–99% of the in-sample linear oracle. The workspace run does not fuse yet, so this gain is not what it measures.

**Restart.** The step's key covers:
- the call-set files by size and content (`genotype_file_fingerprint`), the codec and the genome build;
- the samples (crosswalk), cohort, store and measurement steps' keys;
- its code: its functions and `module_closure` of `gatksv_store_rows`, `store_converter` and `measurement_model`.

Each chromosome is a sub-checkpoint. Its blocks of rows are deleted once the chromosome is written.

**Cost** [est]. The parse is R·N genotypes for R kept and dropped records and N call-set samples, and every FORMAT field of every sample is parsed. At the store step's measured GT:DS:GP rate, 3.3·10⁶ genotypes/s per core, that is R·N / 1.2·10¹⁰ core-hours:
- 8.4 core-hours at R = 10⁶ and N = 10⁵;
- 35 core-hours at N = 4.1·10⁵.

R and N are not in these docs; both figures are assumptions to check against the attached release. A GATK-SV record carries more FORMAT fields than GT:DS:GP (11 on the 1kGP freeze's CNV records, against 3), so the parse can be slower by up to that ratio: 31 to 127 core-hours.

The merge reads and writes the store once more. At `assemble_half`'s measured 6.3·10⁷ genotypes/s, that is about 7.5 core-hours at the design workload.

The chromosomes run in parallel, at most 22 processes, so the wall time is the largest chromosome's. With chr1 holding about 8% of the records [est], that is 0.7 to 11 h for the range above. The VM is billed whole, so a machine of about 32 vCPUs fits this step better than the store's.

The step needs one more store's worth of disk: `direct_sv/store`, beside the store step's (and the measurement step's, where it recalibrated). The added rows alone are R_kept·n bytes before zstd: 100 GB at R = 10⁶, n = 10⁵ [est]. A chromosome's blocks of rows are held uncompressed until it is written.
