# bench-real: MAGE cis-expression prediction with 1kGP SNV/indel and SV genotypes

The real-data benchmark that no method lane controls (scratchpad EVIDENCE_RULE). Public data only; nothing from All of Us.

## Sources (all public; nothing from All of Us)
| Data | Version and location | Integrity |
|---|---|---|
| Expression, covariates, sample metadata | MAGE v1.0 (Taylor et al. 2024, Nature 631:610), Zenodo record 10535719, `MAGE.v1.0.data.zip`, CC-BY-4.0. 15 members are extracted by HTTP range reads, one request per member (`fetch_mage.py`). | zip md5 9b32d1e24aa883b3dc57359598420203; per-member CRC32 and sha256 in `data/mage/PROVENANCE.json` |
| Genotypes | 1kGP 3,202-sample phased SNV/INDEL/SV panel, NYGC high coverage, `20220422_3202_phased_SNV_INDEL_SV` (Byrska-Bishop et al. 2022, Cell 185:3426), from the EBI FTP over http | every file md5-checked against `20220804_manifest.txt`; `public/kgp_phased_panel_20220422/md5_checked.txt` |
| Target-gene structure | GENCODE v38 comprehensive annotation (MAGE's gene set), `gencode.v38.annotation.gtf.gz`, EBI FTP | md5 checked against the release MD5SUMS |
| Pedigree, populations | `20130606_g1k_3202_samples_ped_population.txt`, from the same FTP directory | — |
| Second SV source | HGSVC2 PanGenie genotypes of the 3,202 samples (Ebert et al. 2021), `20201217_pangenie_merged_bi_nosnvs.vcf.gz`: short-read genotyping of long-read-discovered SVs | — |

On MSI the benchmark root is `/scratch.global/sauer354/svpgs-team/bench-real/`, and the shared panel copy is `/scratch.global/sauer354/svpgs-team/public/kgp_phased_panel_20220422/`.

## Samples
- **Individuals:** 730 of the 731 MAGE lymphoblastoid-line individuals, one library each: AFR 196, AMR 113, EAS 141, EUR 141, SAS 139, across 26 populations.
- **PanGenie records** with a missing genotype in any of the 730 are skipped (199 on chr22); `build_dataset.py` prints the count.
- **Library swap:** MAGE v1.0 carries a swap, per the mccoy-lab README of 2026-03-27. SRR19762530 is labelled HG00237 but is NA11919, and SRR19762653 is the reverse.
  - Only SRR19762653 is in the 731-library analysis set, and HG00237's own library SRR19762247 is in it too. So the column labelled NA11919 holds HG00237's expression, and NA11919 has none.
  - `build_dataset.py` drops NA11919 and records this in `dataset/BUILD_NOTES.txt`. (If both swapped libraries were present, it would swap their expression and PEER columns instead.)

## Phenotype
- **Expression:** inverse-normal TMM expression, as used for MAGE eQTL mapping (GENCODE v38 genes that passed MAGE expression filtering). Autosomes only; chrX is excluded because it's haploid in XY samples.
- **Covariates:** MAGE's own eQTL covariates: sex, 5 genotype PCs, 60 PEER factors.
- **Adjustment:** the covariates are regressed out by OLS fitted on the training samples only, and test expression is adjusted with that training fit.
- **Caveat:** MAGE computed the PEER factors and PCs on all samples. That use is unsupervised (expression-only or genotype-only) and identical for every method.

## Genotypes and variant annotation
- **Dosages:** alternate-allele counts 0/1/2, one row per ALT allele. A site is kept if it's polymorphic among the 731, and each fold then drops sites that are monomorphic in its training set.
- **SV definition:** a symbolic ALT, or an allele of at least 50 bp. That's the field-standard size definition (Mahmoud et al. 2019; HGSVC).
- **Annotations passed to methods:** position, end, signed distance to the TSS (0 when an SV overlaps it), is_sv, SV type, SV length, allele-length change, and training allele frequency.

## Window
A variant is included if its interval overlaps TSS ± 1 Mb, the cis window of MAGE's own mapping and of GTEx (GTEx Consortium 2020).
- **This is an external standard, not derived from the data.** A data-derived cut-off (e.g. where pooled excess χ² reaches zero) needs a significance threshold or a random-walk argmax, and neither is well posed.
- **What methods do instead:** they get distance to the TSS as an annotation, and can learn how signal decays.

## Splits (sealed; `dataset/splits.json` and its sha256 in `splits.sha256`)
- **random5:** 5 folds. Whole 1kGP families stay in one fold, and each superpopulation is spread evenly over the folds. The seed comes from sha256("bench-real/random5").
- **loso:** leave one superpopulation out (5 splits). Train on four superpopulations, test on the fifth. This is the portability test.

## Harness contract
- **The method:** `fit(train: TrainData) -> predictor`, where `predictor.predict(test_genotypes) -> ndarray`.
- **What TrainData holds:** training genotypes, the adjusted phenotype, variant annotations, superpopulation and population labels, and the target gene's GENCODE v38 body, strand, merged exons and merged CDS (1-based closed, like POS/END).
- **Sealing:** test phenotypes never reach method code; the harness reads them only to score.
- **Batch contract:** a method that pools hyperparameters across genes (as SV-PGS does across traits) is `fit_batch(trains) -> list[predictor]`, run with `--contract batch`. It's called once per split and feature set, with a lazy sequence of every selected gene's TrainData. The sequence exposes no test data and refuses any sealed gene. Scoring and run.json are unchanged.
- **Views contract:** `fit_views(views) -> {(gene_id, split, feature_set): predictor}`, run with `--contract views`. It may instead yield pairs, so predictors needn't all be held at once. It's called once with a lazy mapping of every requested view: training data only, and it refuses sealed genes. `Variants.chromosome_row` (with `source`) identifies a column across overlapping windows, so a method can share Grams across genes, folds and nested feature sets. The per-gene and batch contracts are special cases, and a test shows identical predictions under all three for a non-sharing method.
- **Confirmation genes:** `dataset/sealed_confirmation_genes.tsv` is never scored except under `--confirmation`, and only when the lead calls it.
  - A `--genes` list naming a sealed gene is refused.
  - A derived dataset (parent_dataset.txt) must carry its parent's sealed list byte for byte.
- **Feature sets:** `snv` (panel SNVs and indels under 50 bp), `snv_sv` (all panel rows), and `snv_pgsv` (panel SNVs/indels plus PanGenie SVs).
- **Submitting:** a method lane sends a file and callable (`path.py:callable`), and bench-real runs it on the sealed splits. Lanes don't run the benchmark themselves.
- **Costly methods:** they run on a sealed random gene sample, a prefix of `dataset/gene_order.tsv` (a seeded permutation of all genes), via `--gene-prefix N`.
- **Reliability:** each column carries `Variants.reliability`, the expected r² of its stored dosage with the true genotype. It's 1 for direct calls; a derived dataset's variants.tsv `reliability` column gives it where dosages were filled (for GATK-SV no-calls, a smooth function of log length per type fitted to the measured GT/RD_CN agreement, from lr-sv), and an overlay's `dr2` gives it for imputed SVs. A method that can't use it just sees the dosages.
- **SV effects:** for linear predictors, each run also writes `<tag>.sv_coefficients.tsv.gz`: every SV column's effect on the genotype scale, its training mean, and its row in the gene window, per gene, split and feature set. So the SV part of any score decomposes into single SVs without refitting.
- **Run record:** each run writes `<chromosomes>.run.json` before its first fit: the method spec and its file's sha256, the harness source, design, feature sets, gene prefix N, gene count, the scored genes' own digest and the splits' sha256. The source is resolved before the first fit, so no provenance failure can cost a finished run its results: `harness_commit` in a git checkout, otherwise `harness_source` "no-git" with `harness_source_sha256`, a digest over the `benchmarks/` and `sv_pgs/` sources.
- **Run identity (`run_id`):** the sha256 of that record without the note and the outcome, i.e. of everything a prediction depends on: method file, harness source, dataset and splits, the gene set itself, feature sets, extra genotype sources, sample subset and failure policy. The output path and the tag name a chromosome, not a run, so a run whose record has a different identity is refused rather than allowed to overwrite the other run's outputs; use another `--out`.
- **Durable fits:** every finished fit is written to its own file under `<tag>.parts/` as it lands (a temporary name renamed into place, so a file under its own name is complete), and the run's arrays are built from them at the end, one feature set at a time. A run that dies keeps its finished fits; the same run started again carries them and fits only what is missing, which for the per-gene contract is whole genes. The pooled `batch` and `views` contracts are called with every gene at once, so they refit and rewrite their parts. The parts are removed once the run's own outputs are written.
- **Output:** out-of-fold predictions for every gene and sample, per design and feature set, plus per-fit CPU seconds and variant counts.

## Scoring
- **r²:** per gene and superpopulation, the squared Pearson correlation of prediction with adjusted held-out expression; 0 for a constant prediction.
- **Pairs:** paired differences between arms, averaged over genes.
- **SE:** the delete-one-chromosome jackknife, weighted for unequal chromosome sizes (Busing et al. 1999), since genes on one chromosome share variants. loco.py gives SNV vs SNV+SV means and gains with these intervals beside the gene-level bootstrap. On a single chromosome the SE is gene-level and labelled as such.

## Baselines
- **top_variant:** the lead marginal variant, fitted by OLS.
- **gblup_reml:** REML h², with the GLS intercept.
- **mr.ash:** the published R package (mr.ash.alpha) run as itself; the former Python port was deleted (identical SNV and SNV+SV predictions).
- **The engine's current design** is added when it lands on main.

The baselines' math checks are in `tests/test_bench_real_baselines.py`: REML optimality against direct REML, dual/primal BLUP identity, sparse recovery, and lead-variant choice.

## Leakage rules
- MAGE's published eQTL, fine-mapping and colocalization results were computed on all 731 samples, including every test fold. **They must never be used as features, priors or gene filters.**
- Gene strata may only come from training-fold quantities, e.g. GBLUP-REML h² on the training fold.

## Limits (what this benchmark is and is not)
- **Small n** (about 585 training samples in random5, 535–618 in loso) and a sparse cis architecture. It's closer to fine-mapping than to polygenic traits, so it tests prior shape, SV credit and portability in the large-effect, few-causal regime. It says little about the polygenic tail.
- **Genotypes are direct high-coverage calls,** not imputed. There's no imputation-reliability channel, so the r² offset and fusion terms of the model aren't exercised. The PanGenie arm (short-read genotyping of long-read SVs) is the nearest public proxy.
- **Tandem repeats aren't annotated** in the panel, so the TR signed-length term isn't tested.
- **Lymphoblastoid-line expression** is a molecular phenotype. Its architecture (strong cis, larger SV enrichment) differs from complex traits.

## Targeted runs on the screened ranked list
- **The list:** a genotype-only ranked list (sv-screen's sv_ranked_v2.tsv) runs top first, in checkpointed chunks, via `--genes <list> --gene-ranks START STOP`. run.json records the list's sha256 and the rank range.
- **Strata:**
  - development: every gene bench-real had already scored when the ordering was chosen (genes_already_scored.tsv, 5,430 genes);
  - held out: ranked genes never scored before.
  - Only the held-out stratum, and later the sealed confirmation set, supports discovery claims.
- **Genome-wide totals** combine the targeted set (a certainty stratum, weight 1) with the random gene-order prefix (weight |U| / n on its genes outside the targeted set), by Horvitz–Thompson weighting (genome_total.py). The standard error is the design standard error of the random stratum (simple random sampling without replacement, finite-population corrected; the targeted stratum is a census with no design error), plus, when person-bootstrap replicates are given, the measurement variance of the total. A chromosome jackknife is not used here: it would charge the census stratum's between-chromosome heterogeneity (the SV gain is concentrated on chr16) as sampling error.

## Which open design questions it can answer
**Can answer, on real biology:**
- **Do SVs add held-out accuracy?** Compare `snv_sv` with `snv` per method. It also shows whether a method that gives SVs their own prior class extracts more of that value than GBLUP or mr.ash.
- **Does the SV gain transfer to an unseen ancestry?** Compare the gain under loso with the gain under random5, per superpopulation.
- **Prior shape:** a learned continuous mixing density against GBLUP (Gaussian), mr.ash (a discrete scale-mixture grid) and the lead variant (the sparse extreme), where cis effects are large and few.
- **Annotation-driven scale:** distance to TSS, SV type and length, and training allele frequency (the frequency term), all learned from real effects.
- **SV measurement:** short-read panel SVs against PanGenie long-read-panel SVs, on the same people.

**Cannot answer:**
- the imputation-reliability channel (the r² offset, D* and GATK-SV fusion), because genotypes here are direct calls;
- the polygenic tail and genome-wide architecture;
- tandem-repeat length terms (no TR annotation);
- binary traits;
- behaviour at n in the tens of thousands.

## Tier 2 (not built): complex traits from public summary statistics
- **Design:**
  - train on Pan-UKB EUR summary statistics, with the Bai et al. 2026 UK Biobank SV/VNTR releases as the SV arm;
  - evaluate in an independent cohort's summary statistics (FinnGen, or BBJ for cross-ancestry) with the summary-statistic R² estimate (wᵀẑ_t/√N_t)² / (wᵀR_t w);
  - use target-ancestry reference LD from 1kGP/HGSVC.
- **What stops it for SVs:** no public independent cohort has SV association statistics. SV weights would need target SV z-scores imputed from SNP z-scores through reference LD, which makes the result depend on the reference panel. Tier 2 would therefore test the SNV-level prior structure (learned density, frequency term) on real complex traits, not SV credit.
- **Other caveats:** reference-LD mismatch, and the estimator's bias under that mismatch.

## Running it
bcftools must be on PATH. From the repository root:

    python benchmarks/bench_real/fetch_mage.py <root>/data/mage
    benchmarks/bench_real/fetch_genotypes.sh <root> <public-dir> <threads>
    python benchmarks/bench_real/build_dataset.py --root <root> --shared --gene-annotation
    python benchmarks/bench_real/build_dataset.py --root <root> --chromosomes <c>   # one per autosome, in parallel
    python benchmarks/bench_real/splits.py <root>/dataset
    python benchmarks/bench_real/harness.py --dataset <root>/dataset --method benchmarks/bench_real/baselines.py:gblup_reml         --name gblup_reml --design loso --chromosomes chr22 --out <root>/results --workers <n>
    python benchmarks/bench_real/report.py --results <root>/results --dataset <root>/dataset --methods gblup_reml top_variant --out <report dir>

Set OMP_NUM_THREADS=1 (and the OpenBLAS, MKL and numba equivalents) when running several workers.

The math checks are `tests/test_bench_real_baselines.py`, in the regular suite, on synthetic data only.
