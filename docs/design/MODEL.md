# The one model

SV-PGS fits one Bayesian model to every variant. Speed comes from exact computation in the right order, never from a different model for some variants (SPEC).

**Evidence tags** (scratchpad EVIDENCE_RULE):
- `[sim-only]`: a lane's own simulation. It checks the math and the code, but it is not evidence of an accuracy gain.
- `[semi-real]`: real public haplotypes or real imputation, with simulated effects and phenotypes.
- `[real]`: measured on real data with nothing simulated: real phenotypes for accuracy claims, real genotype data for genotype-accuracy claims.
- `[provenance unknown]`: no traceable source.

Untagged numbers are derivations, definitions or targets.

## 1. Likelihood

- **Quantitative traits:** y = Cα + Σ_j D_j β_j + ε, with ε ~ N(0, σ²_e).
- **Binary traits:** a logistic likelihood on the same linear predictor. A continuous liability target (age of onset, family history) is used where it exists, so the trait gets a Gaussian working likelihood and shares the Gram with other traits.
- **C, the covariates:**
  - sex, age terms, genetic PCs;
  - a pipeline-half indicator (the two imputation halves) and a cohort indicator (imputed vs long-read-called rows).
- **Covariate handling:** they are projected out exactly (Frisch–Waugh–Lovell) before any LD summary is formed. Block-diagonal LD with ancestry structure left in overfit 400× in simulation [semi-real: design-credit, 1kGP-based genotypes].
- **D_j, the stored genotype column:** the dosage after two corrections, standardized by its empirical SD in the training fold (never √(2pq), which up-weights ancestry-differentiated variants by (1+F)):
  - value-matched removal of the imputation background (STORE.md);
  - where a validated curve exists, a per-stratum, per-ancestry linear recalibration D* = μ + κ(DS − μ). The recalibration's shape added ≤ 0.012 r², so only the scale is kept [real: pilot50 imputation vs long-read truth].
- **A tandem-repeat (TR) locus** is one signed-length column Z_i = Σ_a Δlen_a · D_ia, summed over every record in the locus, including indels under 50 bp.
  - That is the reference-invariant total length, E[ΣL | data].
  - TR loci have no per-allele columns. Equivalently, the allele effects get the rank-one prior β_a = Δlen_a · θ, so every allele dosage still enters.
  - Measured: +4–12% R² where TRs carry signal, ≤ 0.8% cost where they don't. Per-allele TR columns were neutral or negative [sim-only: design-trlocus coalescent simulator].
- **Non-TR multi-allelic bubbles** keep allele columns, with the REF state given its own column (exchangeable allele states), so predictions don't depend on which allele GRCh38 carries. This is accuracy-neutral [provenance unknown] and removes the reference dependence.

## 2. Measurement model

- **The column is a measurement of the true genotype G_j.** Imputed SV and TR dosages behave like confident posterior draws (κ ≈ √r² per stratum), not calibrated posterior means [real: pilot50 truth slopes]. So INFO and dosage-based r² cannot rank SV reliability.
- **Reliability:** r²_j = corr²(D_j, G_j). It is truth-calibrated and triad-corrected: r(D,T1)·r(D,T2)/r(T1,T2) over two independent long-read truths. A per-record model predicts it from site features, and the model ships as a coefficient table (`sv_pgs/imputation_reliability.py`).
  - Locus-level r² is predicted the same way for Z.
  - The r² estimate never enters the Gram. Adding E[x²] corrections shrinks every effect by r² (measured −0.2 to −3.2% R² [sim-only: design-trlocus]).
- **GATK-SV short-read calls are a second measurement B of the same G:** B = α_B + ρ_B·G + e_B. α_B is the false-positive intercept (class precision 0.90–0.97 [real: AoU GATK-SV genotype filter]).
  - **Recalibrate first:** D* with triad-derived κ, within ancestry.
  - **Per-locus reliability:** for calibrated (Berkson) strata, from the per-locus V_A/V_G. Elsewhere, a mean anchor shrunk toward the stratum truth value, with weight σ²_s/(σ²_s + v + ω²_s), where ω²_s is the anchor's excess error measured on truth loci.
  - **Where GATK-SV per-evidence read-depth genotypes exist** (DEL/DUP ≥ ~1 kb), the three-source identity r²_A = C_AB·C_AC/(C_BC·V_A) replaces the anchor, once validated. Paired-end and split-read genotypes equal the final call (r = 1.0 [real: 1kGP GATK-SV freeze V3, chr22]), so they cannot be a third source.
  - **The fused column** is the per-missingness-group best linear predictor. GATK-SV no-calls are informative (a quality-score band), so inside the workspace they are filled with E[B | SL].
  - **Acceptance (met in simulation):** gap to the oracle ≤ 0.001 with known r²_A, and ≤ 0.01 (MCAR) / ≤ 0.025 (MNAR) with estimated r²_A. The old closure's VNTR gap was −0.34 [sim-only: sim F, per-locus r² spread from pilot50].
- **The ~12k long-read panel members are stacked as an extra store half** of hard calls on the same sites, with their own cohort covariate. Measured: +7–8% R² at 100k, +18–19% at 50k (a Daetwyler projection the simulation agreed with), and truth rows performed like imputed rows for the same people [semi-real: 1kGP-based genotypes with emulated imputation]. Folds are kinship-grouped across the union of both halves.

## 3. Prior

- **Effect prior:** β_j | class c ~ ∫ N(0, u_j · s) g_c(s) ds.
  - The mixing density g_c is learned nonparametrically by empirical Bayes for each class: a dense log-scale grid with a learned smoothness penalty, over a data-driven scale range.
  - There is no point mass at zero; effects are continuous, so the prior is continuous.
  - TPB and BayesR-like shapes are special cases. A learned mixing density exists because a fixed-shape TPB lost to BayesR by 0.025–0.10 R² in fifteen Gibbs scenarios [sim-only: design-reliability founder mosaics]. The E6 scenarios showed the opposite [sim-only: design-trlocus].
- **Scale model:** log u_j = level_c + log r̂²_j + d_jᵀθ.
  - The r² coefficient is exactly 1 by derivation: the prior on the true-genotype effect maps to the observed column through r². Free EB could not identify it [sim-only: design-reliability].
  - level_c and θ get hierarchical EB priors whose mean and variance are learned across classes and across traits (`da928bd`). Cross-trait pooling matters: a single trait's EB recovered only 1–9% of the true-prior gain, against 25–115% for the true prior at n = 10–20k [sim-only: idea-svprior founder mosaics].
- **d_j, the same design for every variant** (SNVs included, SPEC 8a5a936). Continuous entries are smooths with learned smoothness, never bins; discrete entries stay discrete.
  - variant type, length and repeat status (SPEC);
  - SV context:
    - the K = 3 nearest SV loci, each with its distance and locus diversity H_locus = 1 − Σ f_a², plus class and length;
    - H_locus-weighted SV density within ±50 kb;
    - TR-locus membership and locus properties;
    - tagging strength to SV/TR columns (max and summed r²), and ρ²_j = R²(SV column ~ local SNVs);
  - external association z² per source, with an absent indicator and a per-source EB weight, symmetric between SNVs and SVs:
    - Bai et al. 2026 UK Biobank SV and VNTR releases;
    - the Pan-UKB SNV release as the SNV side;
    - a disease with no release uses its quantitative proxy's z²;
  - candidates pending measurement: TR mutation-rate features (Ewens θ̂, the σ_d² tagging ceiling) and symmetric functional annotations.
- **No hand-chosen prior anywhere.** Numerical safeguards (trust region, warm-up, Anderson) are not priors.

## 4. Inference: EP-EB

- **Method:** type-II maximum likelihood under an expectation-propagation approximation, with exact one-dimensional tilted moments. It matched exact Gibbs within ±1.5%. The old GIG mean-field fit was 23–36% worse, and its plug-in fixed point 28–42% worse [sim-only: theory-inference E6 on design-trlocus scenarios].
- **Hyper step:**
  - a moment start, then the MacKay/Fellner–Schall fixed point with Anderson acceleration, inside a trust region;
  - a warm-up before the first hyper step.
  - Plain EM converges at rate ≥ 1 − edf/p, about 0.99 at production scale. From the defaults its reported SV/SNV enrichment was 1.65 whatever the truth [sim-only: gam-eval reproducer].
- **Certificate:** the Newton decrement of the hyper objective (in nats) together with the relative prediction change ‖XΔμ‖/‖Xμ‖. Parallel EP leaves a few sites in limit cycles, so the per-site maximum is not a certificate. The certificate is recorded in the artifact, and a fit without it is not accepted.
- **Binary traits:** logistic EP with an exact 32-point Gauss–Hermite predictive. Probit was rejected: VB-probit lost 0.012–0.020 AUC, and EP-probit only tied logistic [sim-only: theory-inference].
- **Predictive variance:** K = 64 exact posterior draws by perturb-and-solve, riding Stage 2's passes and scored in the same single read.

## 5. Pipeline

- **Stage 0: one phenotype-independent genotype pass** (`genotype_statistics.py`, `code_products.py`).
  - Exact int8 → int32 LD-block Grams on the FWL-projected columns, per-variant sums, and X'Y for every trait × fold.
  - The TR length columns as an exact sparse map of stored codes.
  - The tagging and ρ² features.
  - The candidate set by the information rule N·Var(D_j)·r̂²_j·τ²_c ≥ c. It is variance-based, so copy-number rows are kept, and in exact Bayes it is a compute knob only.
- **Stage 1: the LD-space EP-EB warm start**, one per trait × fold. It is not on main yet. The newest work, the dense EP-EB reference, is tag `archive/2026-09-19/build-ep-oracle` (see HANDOFF.md).
- **Stage 2: full-data certification** (`exact_polish.py`).
  - Block-Jacobi PCG on the FWL-projected system, which needed 17–28 passes where block Gauss–Seidel needed over 40 [sim-only: synthetic store].
  - Control-variate Hutchinson estimates of diag(Σ), using the block inverse as the control variate.
  - Posterior draws.
  - This stage carries the real weight: a block-diagonal Stage 1 alone was 2.7× off at p/n = 20 [semi-real: design-credit].
- **Scoring** (`fast_scoring.py`): every trait × fold model and its posterior draws in one read of the store. It is exact to 1e-13; an H100 does 100k × 17.3M in about 100 s [sim-only: synthetic store; timing].

## 6. Built but not wired
- `pleiotropy_layer.py`, the joint multi-trait EP-EB layer: continuous learned multiplier density, class rates by type-II ML. It gets wired in when Stage 1 lands, then its gain is re-measured. It measured +1.4% on a weaker base, and +7.3% under high trait overlap [sim-only: design-multitrait].

## 7. Measured and rejected (do not re-propose without new evidence)
A rejection resting only on a lane's own simulation (`[sim-only]`) is provisional. It stands until it is re-measured on the neutral benchmarks: bench-real (real held-out data) or bench-sim (real haplotypes, a misspecified truth family, sealed seeds). A `[sim-only]` rejection is not grounds to refuse a new measurement there.
- A multi-ancestry deviation prior β_{j,a} = β_j + δ_{j,a}: −7 to −46% non-EUR accuracy [sim-only: idea-ancestry msprime].
- Refining SV dosages from information already in the imputed files (locus kernels, tag-SNV shrinkage, boosted trees, monotone recalibration for SVs): no PGS gain [real r² on pilot50; sim-only PGS impact: design-genorefine].
- Sibling-allele coupling (GR-2) beyond the locus model: no gain [real r² on pilot50; sim-only PGS impact: design-genorefine].
- Fine-structure signal beyond PCs carried by SVs: ΔR² −0.00002 ± 0.00003 [sim-only: idea-ancestry msprime].
- Per-class free tails estimated from each class's own columns: they cost more than they returned [sim-only: idea-bigcausal, at n·h²/p ≈ 4.2 against production's ≈ 0.009].
- Sparse arithmetic kernels for rare columns in Stage 2: they lose to dense tensor cores at ~1,800 right-hand sides [sim-only: synthetic store; timing]. Rare variants are sparse in storage only.
- gamfit for the r² calibration model: it failed REML certification, and only tied the linear model [real: pilot50 truth tables].
