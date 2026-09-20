# bench-sim pre-registration v1 (frozen before any submission)

This is the neutral semi-synthetic benchmark (EVIDENCE_RULE.md 2b). It was written without reading any method lane's theory or prototypes. Changes after freezing are versioned amendments at the end of this file, dated before any sealed test run.

## 1. Genotypes: real haplotypes
- **Source.**
  - 1kGP high-coverage phased panel `20220422_3202_phased_SNV_INDEL_SV` (EBI, public), with its GATK-SV symbolic SV records.
  - v1 uses chr22. The same scripts extend to chr19–21, then genome-wide.
  - Only founders are used: samples with neither parent in the 3,202 set, per `20130606_g1k_3202_samples_ped_population.txt`.
- **Donor/panel split.** Within each superpopulation, founders are split 60/40 into donors and an imputation panel, using the public seed 20260919. Cohort haplotypes are copied only from donors, and imputation uses only the panel. That keeps imputation from being unrealistically perfect.
- **Target cohort: N = 50,000,** with group weights withdrawn by Amendment 6: they had been derived from a published All of Us statistic, which the user's rule of 2026-09-19 excludes. bench-sim re-specifies them from a non-AoU basis in a later amendment.

  | group | weight | mean ancestry | per-person proportions | admixture time T (generations) |
  |---|---|---|---|---|
  | EUR | withdrawn (A6) | EUR 1.0 | — | — |
  | AFR-admixed | withdrawn (A6) | AFR 0.80, EUR 0.18, AMR 0.02 | Dirichlet(20·mean) | 7 |
  | AMR-admixed | withdrawn (A6) | AMR 0.50, EUR 0.42, AFR 0.08 | Dirichlet(20·mean) | 13 |
  | EAS | withdrawn (A6) | EAS 1.0 | — | — |
  | SAS | withdrawn (A6) | SAS 1.0 | — | — |

  The withdrawn weights had solved mean-ancestry equations exactly. The admixed-group means and T follow Bryc et al. 2015 (AJHG) and Baharian et al. 2016, coarsened.
- **Mosaic haplotypes.**
  - Local ancestry switches as a Poisson process in genetic distance at rate T per Morgan. Each new tract's ancestry is drawn from the person's proportions.
  - Within a tract, Li–Stephens copying from that ancestry's donor haplotypes switches donor at rate ρ_a = 4·N_e/K_a per Morgan, with N_e = 20,000 (the IMPUTE2 recommended value) and K_a the donor haplotype count.
  - Genetic map: Beagle's PLINK GRCh38 map, linearly interpolated.
  - No mutation or miscopy is added: every allele in the cohort is a real 1kGP allele on a real haplotype background.
- **Variants kept:** records with donor MAC ≥ 3.
- **Classes:**
  - SV: symbolic ALT, or |len(REF) − len(ALT)| ≥ 50;
  - TR: a non-SV indel overlapping a UCSC simpleRepeat interval;
  - INDEL: other indels;
  - SNV.

## 2. Measurement: real imputation
- **SNV and INDEL** are observed exactly as unphased genotypes, standing in for 30× WGS calls.
- **TR and SV** are masked in the cohort and imputed with Beagle 5 (gp=true). The target is the cohort's observed SNV/INDEL genotypes, the reference is the disjoint panel's phased haplotypes, and the map is the PLINK GRCh38 map. The cohort is imputed in batches of 5,000 samples.
- **Stored value:** round(DS × 1000) through the repo's `dosage_store.encode_dosage_milli`. Exact genotypes map to the exact codes 0, 127 and 254.
- **What methods receive:** Beagle's per-variant DR2, as the measurement-quality annotation. The realized r²(DS, truth) is recorded for reporting only, never given to methods.

## 3. Truth family (phenotypes)
Each scenario is generated from its own seed. The draws are independent unless stated.
1. **Heritability** of chr22, on the quantitative or liability scale: h² ~ LogUniform(0.01, 0.2).
2. **Base causal probability:** π ~ LogUniform(1e-4, 3e-2) per variant.
3. **SV enrichment mode** ∈ {none, scale, probability, both}, uniform.
   - For scale: effect SD × LogUniform(1.5, 5).
   - For probability: π × LogUniform(2, 30).
   - TR gets its own independent mode and folds.
4. **Effect shape family**, uniform over:
   - (a) Gaussian;
   - (b) Student t, ν ~ U(2.2, 8);
   - (c) Laplace;
   - (d) a 3-point variance grid {1, 10, 100}·c with Dirichlet(1,1,1) weights;
   - (e) two-point magnitude with a random sign;
   - (f) 0.99·N(0,1) + 0.01·N(0,100);
   - (g) a log-normal scale, β | τ ~ N(0, exp(N(0, τ²))) with τ ~ U(0.5, 2);
   - (h) directional: Laplace magnitudes whose sign follows a class direction with probability U(0.5, 0.9). For SV DELs overlapping exons the direction is negative; elsewhere it is random per scenario.

   With probability 0.5 the SV class draws its own family independently, a shape difference.
5. **Frequency dependence.**
   - Per-SD variance ∝ [2p(1−p)]^(1+α_c), with α_SNV ~ U(−1, 0), where −1 means no frequency dependence.
   - α_SV = α_SNV + δ, with δ ~ U(−0.5, 0.5) with probability 0.5, else 0. TR shares α_SV.
   - With probability 0.5, p is floored at p0 ~ LogUniform(1e-3, 2e-2), a plateau.
   - p is the cohort allele frequency, the AFR-donor frequency or the EUR-donor frequency, chosen uniformly.
6. **LD dependence:** variance ∝ ℓ_j^γ, where ℓ_j is the truth LD score within ±1 cM from a 5,000-sample subsample. γ = 0 with probability 0.5, else U(−0.5, 0).
7. **Annotations.**
   - The annotations: in a gene (RefSeq transcript span), exon overlap, log distance to the nearest TSS, TR membership, and log SV length.
   - Mode ∈ {absent, scale, probability, both}, uniform:
     - scale: log variance += Σ f_k(x_k);
     - probability: logit π += Σ h_k(x_k).
   - Binary annotations get coefficients N(0, 0.5²) for scale and N(0, 0.7²) for probability.
   - Continuous annotations get a random natural cubic spline: 4 knots at data quantiles, coefficients N(0, 0.4²), centred.
8. **Non-additivity:**
   - with probability 0.3, dominance for causal SVs overlapping exons: d = a·U(−0.5, 1);
   - with probability 0.3, TR loci (TR records within one simpleRepeat interval) act through a saturating curve of their summed length change, c·tanh(L/L0) with L0 = the locus's median |L| over carriers;
   - with probability 0.2, a mild global link: g → g + c·(g² − E g²), with c ~ U(0, 0.15)/sd(g).
9. **Noise.** The residual variance ∝ exp(κ·z_i), where z is standardized age and κ ~ U(0, 0.5). With probability 0.3 the noise is Student t with ν = 5.
10. **Covariates and confounding:**
    - sex ~ Bernoulli(0.5), with effect N(0, 0.3²) in phenotypic SDs;
    - age ~ U(18, 80), with effect N(0, 0.2²);
    - a batch indicator ~ Bernoulli(0.5), with effect N(0, 0.1²);
    - an ancestry-correlated environmental mean Σ_a c_a·q_ia, with c_a ~ N(0, 0.3²) and q the true proportions. Methods get genetic PCs, computed by the harness from observed genotypes, never q.
11. **Trait type:** quantitative with probability 0.6. Otherwise binary by liability threshold, with prevalence K ~ LogUniform(0.01, 0.3).

   Scaling: Var(g) is set so that h² holds on the liability or quantitative scale, before covariates are added.

## 4. Scenario sets and sealing
- **Dev:** 24 scenarios, seeds 0–23, public along with their full truth. They're for development and smoke tests.
- **Test:** 64 scenarios from a sealed master seed. The seed, the test truths and the test phenotypes live only in `bench-sim/sealed/` (MSI, mode 700).
  - Method lanes must never read that directory.
  - The SHA-256 commitments of the master seed and of the test-scenario parameter file are published in `COMMITMENTS.txt` before any submission.
- **Samples:** a fixed 80/20 train/test split, stratified by group, with the public seed 20260919. It is the same for every scenario.

## 5. Harness and evaluation
- **Inputs to a method:**
  - training samples' observed codes: uint8, variant-major memmap;
  - the variant table: chrom, pos, ref/alt length, class, cM, the annotations of §3.7, and Beagle DR2 for imputed classes;
  - covariates: sex, age, batch;
  - 10 genetic PCs from observed SNVs, computed on training samples and projected onto test samples;
  - training phenotypes and the trait type (with prevalence for binary traits).
- **At scoring,** the method sees the test samples' codes, covariates and PCs, and nothing else.
- **Compute:** one runq task per (method, scenario), with 16 cores; GPU use is declared and recorded. Wall time and peak RSS are recorded. The same limits apply to every method.
- **Metrics on test samples:**
  - incremental R² over covariates and PCs;
  - for binary traits, liability-scale R² (Lee et al. 2012) and AUC;
  - the calibration slope of y on the prediction, adjusted for covariates and PCs;
  - R² per group;
  - SV credit: the predicted genetic variance share from TR+SV columns vs the truth share.
- **Comparisons:**
  - per scenario, the paired difference against the reference arm, then the mean paired difference with its SE across scenarios, reporting every scenario, losses included;
  - descriptive breakdowns by shape family, enrichment mode, trait type and frequency form.
- **Baselines (neutral):**
  - `oracle_true`: the true genetic value, an upper bound;
  - `oracle_observed`: the true additive β on observed codes, the imputation-limited ceiling;
  - `ridge_inf`: infinitesimal BLUP on standardized observed columns, λ = M(1−ĥ²)/ĥ², with ĥ² from Haseman–Elston on the training set, solved by CG. It runs in two arms: SNV+INDEL columns only, and all columns.

## Amendment 1 (2026-09-19, before any submission): measurement mirrors aou2
Section 2 is replaced so that the measurement process follows the aou2 imputation, per process facts relayed from imputation-4c (no AoU data):
- **Imputation tool:** GLIMPSE2 v2.0.0 static binaries, with --err-imp 1e-3. Chunking comes from GLIMPSE2_chunk --sequential on the reference with the b38 GLIMPSE maps. Other settings stay at defaults until the exact aou2 flags arrive; any change is logged here.
- **Target input:**
  - PLs only at simple sites, the benchmark's SNV and non-TR indel records.
  - PLs come from a read model on the true genotypes: gamma-Poisson depth with mean 30 and shape 8, base error 0.005 for SNVs and 0.01 for indels.
  - A hom-ref call gets the gVCF block rule, PL = (0, trunc(GQ/5), min(30, 2·trunc(GQ/5))), at most (0, 19, 30). Other calls keep their PLs, capped at 999.
  - TR and SV records carry no PL, so they are imputed from haplotype matching alone.
- **Observed values:** every record, SNVs included, is observed as its GLIMPSE2 DS, stored as the uint8 code. Nothing passes through exactly.
- **Quality annotation:** methods receive GLIMPSE2's per-record INFO score, the sample-weighted mean over batches, in place of Beagle DR2.
- **Known limitation:** the public panel has 2,072 haplotypes (40% of the 1kGP founders), which is smaller than the production long-read panel. Benchmark imputation is therefore less accurate than aou2's, especially for rare variants. The public panel's SVs are short-read GATK-SV calls, not long-read calls.

## Amendment 2 (2026-09-19, before any submission): implementation details
- **Genetic positions:** mosaics use the panel's own INFO/CM per record, not the PLINK map. GLIMPSE2 uses its b38 maps.
- **Frequency floor:** when a scenario has no plateau, p is still floored at 1/(2·n_donor_haplotypes). That keeps variants absent from the chosen ancestry from getting exactly zero prior variance.
- **The truth LD score is used only to generate truths.** It is not in the public variant table; methods compute LD themselves.
- **Public variant table:** pos, cm, cls, len_change, ref_len, alt_len, in_gene, in_exon, log_tss_distance, in_repeat, log_sv_length, and the GLIMPSE2 imputation_info.
- **Commitments:** the sealed test parameter-file hash is published in COMMITMENTS.txt before any submission runs.

## Amendment 3 (2026-09-19, before any submission): definitions fixed while landing the code
- **LD score:** ℓ_j = 1 + Σ_{k≠j} r²_jk over reference records within ±1 cM, using raw r² and counting the record itself once. The log is defined without a floor.
- **Truth file key:** the phenotype is stored as `phenotype`, not `y`, since the repo forbids single-letter names. The submission API exposes `train.phenotype`.
- **ridge_inf:** the Haseman–Elston h² is truncated to its parameter space [0, 1]. At h² = 0 the prediction is zero; at h² = 1 the solve is the pseudo-inverse.
- **Per-group R²:** it is reported for every group with more samples than regression parameters + 2.

## Amendment 4 (2026-09-19, before any submission): measurement arms (lead ruling C)
- **Arm "glimpse2"** (label GLIMPSE2-imputed): Amendment 1 exactly, run on a calibration subset, the first 2,500 cohort samples. Its group mix is iid from the cohort's weights.
- **Arm "beagle"** (label Beagle-imputed): all 50,000 samples.
  - Simple sites are observed as read-model calls, the minimum-PL genotype from the Amendment 1 read model.
  - Targets are written phased: a correct call keeps the member's true phase, and a miscalled heterozygote gets a random phase.
  - TR and SV records are imputed by Beagle 5.5 (27Feb25.75f) from the disjoint panel, with defaults and the PLINK GRCh38 map. Symbolic ALTs carry the record ID so Beagle sees them as distinct markers.
  - The arm has no statistical phasing error, so it is mildly optimistic.
- **Every result records its arm label.** Headline claims about draw-like columns wait for a GLIMPSE2 cohort if GLIMPSE2 DS proves draw-like (κ ≈ √r²).
- **Donor/panel disjointness is verified:** 1,554 donor and 1,036 panel founders, with an intersection of 0. Cryptic relatedness between founders is not removed; that is a known limitation.

## Amendment 5 (2026-09-19, before any submission): the measured record set
GLIMPSE2 drops records that are monomorphic in its reference panel: 32,285 chr22 records, all monomorphic among the 1,036 panel founders. In aou2 the imputed callset's records are exactly the panel's records with allele count ≥ 2 (imputation-4c, process fact). So:
- **The measured records** are those with panel minor allele count ≥ 2, in both arms. Every method, the harness's variant table, the kernels, oracle_observed and the calibration see only these.
- **Truths are unchanged.** Causal variants outside the measured set still contribute to the genetic value and to oracle_true, but no method can see them. That is the realistic cost of variants missing from the imputation panel.

## Amendment 6 (2026-09-19, before any submission): AoU-related values withdrawn
The user ruled on 2026-09-19: "never ever use any AoU related data outside of permit and don't download any data from Google or aou buckets ever and don't use real panel". So this document no longer reproduces any AoU-related value:
- **Target-cohort group weights:** they were derived from a published All of Us mean-ancestry statistic, and are withdrawn. bench-sim re-specifies them from a non-AoU basis, such as the public 1kGP superpopulation composition or a stated design choice, and rebuilds the cohort in a later amendment. Until then, results from the existing cohort are labelled "built under withdrawn weights".
- **The production panel's size** is no longer quoted.
- **Kept:** pure process facts about the production imputation, because they are software configuration, not AoU data: tool versions and flags, the hom-ref PL block rule, PLs only at simple sites, and the panel's allele-count ≥ 2 record filter.

## Amendment 7 (2026-09-19, before any submission): group weights from the public 1kGP founder composition
The cohort's group weights no longer come from any All of Us source. They are derived at build time (cohort.group_weights) as each group's superpopulation share of the 2,590 1kGP founders in the public 3,202-sample ped table: EUR 525, AFR 686, AMR 353, EAS 512, SAS 514, so the EUR, AFR-admixed, AMR-admixed, EAS and SAS groups get weights 525/2590, 686/2590, 353/2590, 512/2590 and 514/2590.
- The admixed groups keep their non-AoU mean ancestry and admixture times (Bryc et al. 2015; Baharian et al. 2016).
- The group-weight table in section 1 is superseded.
- The cohort, annotations, dev and sealed truths, Beagle arm and kernels are rebuilt under these weights in `bench-sim/v7/`, with new commitments in COMMITMENTS.txt. The sealed master seed is unchanged.
- Results from the earlier cohort, including the GLIMPSE2 calibration tables, are labelled "built under withdrawn weights".

## Amendment 8 (2026-09-19, before any submission): the true-genotype training half
- **Arm "beagle_truthhalf"** (label: Beagle-imputed with a true-genotype training half) is the Beagle arm in which a flagged subset of the training samples is observed at its true genotypes (code = 127·G) on every measured record, as a long-read truth half would be.
- **The subset:** 20% of each group's training samples, chosen with the public seed (measurement_truthhalf.truth_half). Test samples are never in it.
- **The 20% is a benchmark design choice, fixed here;** it isn't taken from any production cohort. Methods see the flags as `train.truth_half`; every other arm reports them all False.
- **Purpose:** it lets the ablation plan (benchmarks/ABLATION_PLAN.md, term t7) test measurement-model terms that learn from a truth subset.

## Amendment 9 (2026-09-20, before any sealed run): out-of-family truths
critic-method (critique/CRITIQUE_METHOD.md, item 3) found that section 3 mostly sits inside SV-PGS's own model class:
- six of its eight effect shapes are Gaussian scale mixtures;
- its frequency law, LD power, annotation splines and SV enrichment modes mirror SV-PGS's prior terms.

A second, sealed set of truths is added, each outside that class (benchmarks/bench_sim/truth_out.py). Heritability, covariates, noise and trait type are still drawn as in section 3; none of section 3's effect terms apply.

| family | the architecture | the assumption it breaks |
|---|---|---|
| fixed_count | exactly K ~ LogUniform(10, 300) causal records, one per-SD magnitude, random signs | Bernoulli causals under a continuous prior |
| nonscale_heavy | causal probability π ~ LogUniform(1e-4, 3e-2); per-SD magnitudes Pareto with tail index ~ U(1.2, 2.5) above a floor, random signs | Gaussian scale-mixture shapes (this density is zero near 0) |
| hidden_annotation | K ~ LogUniform(20, 500) causal records, all inside UCSC CpG islands plus 2 kb shores (Irizarry et al. 2009), N(0, 1) per SD; the annotation is never given to methods | the given annotations explaining the enrichment |
| clustered_loci | loci are equal genetic-length segments of LogUniform(0.05, 0.5) cM; LogUniform(5, 100) causal loci; a U(0.01, 0.2) fraction of each locus's records share one per-allele effect (same sign and size) | independent effects across variants |
| epistasis | products of standardized genotype pairs (LogUniform(5, 100) pairs of records with cohort MAF ≥ 5%, random signs) carry a U(0.3, 1) share of the genetic variance; a fixed_count additive background carries the rest | additivity |
| panel_absent | K ~ LogUniform(20, 500) causal records, a U(0.5, 0.9) share of them outside the measured set (panel MAC < 2), random-sign unit per-SD effects | the causal variants being measured at all |
| sv_gene_dosage | for LogUniform(3, 60) RefSeq genes touched by SVs, a gene effect N(0, 1); an SV's per-allele effect is the gene effect times its copy change; a fixed_count background on the other classes carries a U(0.2, 0.8)-complement share | SV effects set by a class scale |

The sv_gene_dosage copy-change rule, summed over genes:
- a deletion: minus the fraction of the gene's exonic bases it removes;
- a duplication containing the whole gene: +1; a partial duplication: minus its exonic fraction;
- any other SV that touches an exon: −1.

**Sets:**
- dev_out/: 14 public scenarios, 2 per family, seeds 1000–1013.
- sealed_out/: 56 sealed scenarios, 8 per family. Seeds are sha256("<master>:out:<i>"), a hash domain disjoint from section 4's. Their parameter hash goes into COMMITMENTS.txt before any sealed run.

**Reporting:** every result is reported separately for in-family (sealed/) and out-of-family (sealed_out/) scenarios. The out-of-family report also breaks down by family.
- The epistasis family's oracle_observed uses the additive part only.
- In panel_absent, most causal effect is unmeasurable by design.

**Reference arms:** MegaPRS and SBayesRC are added alongside ridge_inf and oracle_observed, because they also model frequency, LD and annotations. They're built by the compete lane at package defaults, frozen before the sealed run.

## Amendment 10 (2026-09-20, before any sealed run): a read-depth copy-number channel
- **Arm "beagle_readcn"** (label: Beagle-imputed + read-depth CN likelihoods) is the Beagle arm plus per-sample read-depth genotype likelihoods on every measured DEL and DUP record, simulated from the truth (benchmarks/bench_sim/measurement_readcn.py). It lets methods test fusing imputed DS with direct read evidence.
- **Generating model:**
  - The copy number is c = 2 − g for a DEL and 2 + g for a DUP.
  - Reads over the span follow R ~ NB(mean s_i·30·L/150·(c + 2P)/(2 + 2P), size 1/φ), for 30× depth and 150 bp reads.
  - The sample depth scale is s_i ~ LogNormal(0, σ_s).
  - P counts the segmental-duplication partners (UCSC hg38 genomicSuperDups) overlapping the span with fracMatch ≥ identity. Their reads dilute the signal.
- **The caller** assumes unique sequence (P = 0) and a depth scale with error exp(N(0, τ²)). It emits PL for g = 0, 1, 2, as `train.reads` / `test.reads`.
- **Parameter ranges:** σ_s ~ U(0.05, 0.25), φ ~ LogUniform(0.01, 0.2), identity ~ U(0.97, 0.995), τ ~ U(0, 0.1).
- **Two draws:**
  - a public dev draw (seed 20260919·10) serves dev scenarios;
  - a sealed draw, sha256("<master>:readcn"), stored under sealed/, serves sealed scenarios.
  - The sealed parameter file's hash goes into COMMITMENTS.txt.
