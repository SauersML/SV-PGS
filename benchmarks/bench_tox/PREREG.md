# bench-tox pre-registration (frozen before any method is scored)

## Data
- **Source:** NIEHS-NCATS-UNC DREAM Toxicogenetics Challenge cytotoxicity (Synapse syn1761567).
  - Cite: Eduati et al. 2015, Nat Biotechnol 33:933; Abdo et al. 2015, Environ Health Perspect 123:458.
  - Used under the challenge's data use terms, which the user accepted and confirmed allow research use with citation.
- **Restricted:** the per-line values stay in a mode-700 directory on MSI. No per-line value, table or derived dataset enters any repository; only aggregate results do.
- **Lines:** the DREAM training and test sets are pooled, 884 lines × 106 compounds with no missing values. The DREAM split is not used.
  - 793 lines are in the 1kGP 30× phased SNV/INDEL/SV panel (20220422), and those form the analysis set: EUR 290, EAS 203, AFR 193, AMR 107.
- **Values:** the released ComBat-normalized cytotoxicity values, one trait per compound.

## Sealing
- **Rule:** a compound is a confirmation trait when sha256("bench-tox/confirm/" + NCGC id) ≡ 0 (mod 4) (`seal.py`). That gives 34 confirmation and 72 development compounds; the confirmation list's sha256 begins d154565847dc7915.
- **Use:** confirmation compounds are scored once, at the lead's call. No development analysis reads them, including heritability, method choice and term adoption.

## Genotypes and feature sets
- **Genotypes:** every panel record for the 793 lines, with no frequency or quality threshold (SPEC 1fca1cf). Only training-constant columns are dropped, per fold.
- **Feature sets:**
  - `snv`: SNVs and indels under 50 bp;
  - `snv_sv`: every panel record;
  - `sv`: SVs only;
  - `snv_matched`: bench-real's matched-SNV control;
  - `snv_svimp`: bench-sim's imputed-SV overlay, when it covers the genome.

## Splits (`splits.py`; sha256 recorded in DATA_CARD.md when built)
- **random5:** relatedness clusters kept whole. A cluster is a 1kGP pedigree family joined with any pair that KING-robust calls third degree or closer (bench-real `relatedness.py`). Clusters are spread evenly within each continental group, with the seed from "bench-tox/random5".
- **lococ:** leave one continental group out: EUR, EAS, AFR and AMR.
- **Covariates, fitted in-fold on training lines only:** sex, cytotoxicity batch, and genotype PCs. The number of PCs comes from Patterson's Tracy–Widom test (Patterson et al. 2006) at level 1/K with the engine's K, on the training GRM.

## Methods (the same inputs for all)
1. **SV-PGS** (fit-api's callable): full, no-SV-terms and no-annotations.
2. **BayesR,** individual-level (GCTB 2.5.4, published defaults): the polygenic standard.
3. **mr.ashr** at the mr-ash-workflow settings (Lasso init, 2,000 iterations): genome-wide where memory allows, otherwise in the cis-candidate analysis only (stated per run).
4. **top_variant.**
5. **GBLUP:** a labelled polygenic reference only.

Wall time and peak RSS are recorded for every fit.

## Metrics
- **Accuracy:** out-of-sample R² (the training-mean baseline, plus critic-real's centred form) and r², pooled over folds and development compounds.
- **Headline:** the pooled SV gain Δ = r²(snv_sv) − r²(snv) for SV-PGS, and SV-PGS against BayesR. SEs come from a bootstrap over lines, relatedness clusters crossed with chromosome blocks.
- **SV credit:** the SV share of signal, as a covariance split.
- **Per compound:** results are reported with BH-FDR and are secondary.

## Cis-candidate analysis (pre-specified)
- **Candidates:** CNVs (DEL/DUP) whose span overlaps an exon of a gene in the fixed list.
- **Gene list:** Reactome R-HSA-211859 (Biological oxidations) ∪ GO:0006805 (xenobiotic metabolic process) ∪ GO:0042910 (xenobiotic transmembrane transporter activity). The list is fetched from the Reactome and EBI QuickGO downloads, with versions recorded.
- **Test:** each candidate is scored against the within-ancestry permutation null (999 permutations of the CNV column within continental group, shared across compounds), as critic-real did on bench-real.
- **Reported:** candidates and compounds at BH q ≤ 0.05, among development compounds only.

## Power analysis (closed form, `power.py`; [real] inputs from the data, the rest as stated)
- **Sizes:** n = 793. random5 trains on about 634 lines and tests on about 159; lococ trains on 503–686.
- **Null floor for one test fold:** E[r²] = 0.0063, SD 0.0089.
- **Expected genome-wide polygenic r²** (Daetwyler et al. 2008) at n = 634, for h² from 0.25 to 0.8 (the published range for these traits) and M_e from 6·10⁴ to 1.5·10⁵: **0.0003–0.0067**.
- **Pooled precision:** the 72 development compounds are strongly correlated; their effective number (Σλ)²/Σλ² is **13.2** [real]. The pooled SE of a mean r² is about 0.0089/√(13.2·5) ≈ **0.0011**.
- **Reading:** genome-wide polygenic signal sits between about 0.2 and 6 SE of that SE. The SV share of a polygenic gain is **not detectable** here.
- **Large-effect loci** have 80% power at n = 793:
  - genome-wide (α = 5·10⁻⁸): a variant explaining ≥ 4.8% of variance;
  - candidate-set level (α = 5·10⁻⁵ to 5·10⁻⁴): ≥ 2.3–2.9%.
- **Where SVs can show:** that's where SV gains are measurable, through single large CNVs.
- **Context:** the DREAM challenge found genotype-based prediction of individual responses only modestly better than random (Eduati et al. 2015).
- **Phenotype structure [real]:** in the development compounds' variance, continental group explains a median 0.6%, population 1.8%, batch 0.9% and sex 0.1%. So ancestry confounding of the phenotype is small.
- **Empirical per-compound heritability:** HE regression and GREML on training folds with in-fold PCs, SE reported. It follows once the genome-wide GRM is built.
