# Evaluation: how an SV gain is claimed

Code: `sv_pgs/held_out_comparison.py` (`cross_fit_delta_r2`, `panel_z`, `power_weights`, `influence_correlation`, `size_gate`, `null_cost_gate`), tested on synthetic data.

## Arms
All arms share the same samples, folds, covariates and inference; only the columns and prior inputs differ.

| Arm | Columns | Prior inputs |
|---|---|---|
| A (SNV) | SNVs and indels | class, r̂², SNV-derived annotations and external SNV z² only; no SV information |
| B (SV prior) | A's columns | A's inputs, plus SV-context features and external SV z² |
| C (SV-PGS) | A's columns + SV alleles + TR length columns + fused GATK-SV rows | B's inputs |
| C-null | C with the SV and TR-length columns permuted within ancestry | the same EB as C |
| B-null | B with the SV features permuted among LD × MAF × gene-density-matched blocks | the same EB as B |
| C-capacity | C with the SV columns replaced by MAF-matched random SNVs | the same EB as C |

- **Prior gain** = B − A. It can exist even when SNVs tag every SV perfectly.
- **Column gain** = C − B. Pre-registered forecast: ≤ ~5e-4 R² per typical trait.
  - The forecast comes from the exact identity Δ_j = r²_j(1 − ρ²_j), where ρ² is how predictable an SV column is from local SNVs.
  - Imputed SVs measured ρ² ≈ 0.95–0.98 on public data.
- **Total gain** = C − A.

## Claims and their tests
α = 0.05 is split 0.025 / 0.0125 / 0.0125 across (a) / (b) / (c).

- **(a) SV-PGS predicts better.** The test is a one-sided, fold-stratified paired ΔR² (Δ log-loss for binary traits) of C against A:
  - each fold's predictor is scored on its own fold, then the folds are averaged;
  - the variance is the family-block influence variance plus the closed-form cross-fit pair term;
  - C must also beat C-null by the same test.
  - Cross-fitted scores are never pooled into one correlation. With adaptive fits under an equal-accuracy null, the pooled z had SD ≈ 10 and rejected 29–57% of the time.
  - The nested/encompassing test is never a primary. It rejects even when the arms are equally accurate, because z_nested = 2·z_equal at matched β.
- **(b) SVs carry signal beyond SNVs.**
  - Model-X residual tests: a per-SV GCM scan combined by ACAT, plus a cross-fitted distilled CRT, the two combined by ACAT.
  - Shuffling SV columns is not a null for (b), because it breaks SV–SNV LD.
- **(c) SVs supply prior information.** B vs A, calibrated by B-null refits.
- **(d) Causal or portable, not stratification.**
  - Within-sibship versions of (a) and (b), and per-ancestry tests combined by Stouffer, with Cochran's Q.
  - Negative controls: pipeline half as the phenotype, and a PC-only phenotype.
  - Structure-driven gains are labelled "total", and within-sibship or PC-conditioned gains "direct".
- **(e) Locus claims.**
  - Conditional tests of the SV given its local SNVs; SuSiE with in-sample LD.
  - Gated on a calibration check in the long-read truth subset: any nonzero tag coefficient blocks PIP claims.
- **(f) Across traits.** The pre-registered panel test below, plus Hommel-adjusted per-trait tests.

## The pre-registered panel (21 traits)
- **11 quantitative:** height, BMI, SBP, heart rate, MCV, platelets, WBC, eGFR (CKD-EPI 2021), total bilirubin, LDL (pre-statin), HbA1c (non-diabetic).
- **10 diseases:** T2D, atrial fibrillation, hypothyroidism, COPD, depression, gout, cataract, CKD (lab-defined), psoriasis, prostate cancer.
- **Statistic:** z = wᵀz / √(wᵀRw).
  - w ∝ E[z] from a power model. The weights are non-negative and use no SV biology, only h², n, K, misclassification and repeatability. Correlation-inverse weights would have made most diseases negative.
  - R is the correlation of per-trait family-block influences.
- **Frozen** before any test phenotype is touched: weights, trait list, case definitions, code hash and α plan.
- **Power** (conservative: calibration slope 1, full pair term): panel z 5.2 at 50k and 9.9 at 100k.
- **The diseases-only sub-panel** is secondary: 0.14 / 0.31 power. Family-history liability targets are the planned lift.
- Trait choice uses power and phenotype quality only, never known SV biology.

## Folds, resampling, reporting
- **Folds:** one trait-agnostic 5-fold assignment, frozen before any phenotype is built.
  - Kinship components (KING > 0.0884) and duplicates stay within one fold. Folds are built over the union of the imputed and long-read halves.
  - Stratified by imputation half × ancestry.
  - Every method uses the same folds. Baselines (LDpred2, SBayesRC, PRS-CS, BayesR, GBLUP) run on in-sample Stage 0 LD, so no external panel penalizes them, and they get their own SV-inclusive arms.
- **Resampling unit:** the family.
- **Reporting:** per trait × ancestry and per imputation half, in cells n ≥ 21.
  - A, B, C and the three differences, each with a CI;
  - a half × increment interaction test;
  - negative controls and the α ledger.
  - Per-ancestry results are a headline dimension. In simulation the SV gain was 2–4× larger in African-ancestry groups.

## Simulation gates (must pass before any real claim)
- **G13:** the primary test rejects ≤ 7% of ≥ 200 null traits at one-sided 0.05. That covers the SV null against both A and C-null, and the equal-accuracy twin null.
- **G13b:** C-capacity against A passes the same size gate.
- **G13c:** with SV effects switched off, C costs ≤ 0.5% relative R². Measured: −1e-4 to −2e-4.
- **Red-team gates RT1–RT8:** each attack must keep the null rejection rate within 2 SE of the clean null, and the fake ΔR² within 2 SE of 0.
  - RT1: imputation-half-specific quality with a phenotype shift by half.
  - RT2: fine-structure environment with ancestry-differentiated SVs as a proxy.
  - RT3: relatedness leakage across folds.
  - RT4: leaked in-cohort external GWAS.
  - RT5: SV features proxying LD-dependent architecture.
  - RT6: the imputation panel missing ancestry-private haplotypes.
  - RT7: EB regularization differing between arms, i.e. a non-certified fit.
  - RT8: multiplicity across traits and arms.

## Pending
- The binary pair term, as the IRLS-linearized form. Until then binary claims use the conservative variance.
- The operator (Hutchinson) form of the pair term at p ≈ 17M.
- The claim (b) and (c) machinery in the repo; prototypes exist in the simulation code.
- The validity simulations and the red-team sweep. They were running on MSI and were stopped; resubmit them when compute is back.
