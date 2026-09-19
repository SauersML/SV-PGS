# Design decisions, 2026-09-18/19

Each entry gives the ruling and the measurement or identity behind it. Simulations ran on MSI with public 1kGP/HPRC data or synthetic data. No AoU data was used.

Measurements carry the evidence tags defined in MODEL.md: `[sim-only]`, `[semi-real]`, `[real]` and `[provenance unknown]`. Under the evidence rule, a `[sim-only]` result checks the math and the code, but it is not evidence of an accuracy gain. Rulings resting on it stand until the neutral benchmarks (bench-real, bench-sim) re-measure them.

## Why the old fit failed
- **The inference, not the prior family.**
  - On nine TR-locus scenarios (n_train 40k), exact Gibbs with the TPB prior was +2–7% over BayesR, and EP-EB was within ±1.5% of Gibbs [sim-only: theory-inference E6 on design-trlocus's coalescent simulator].
  - The shipped GIG mean-field fit was 23–36% worse, and its plug-in fixed point 28–42% worse at convergence [sim-only: the same scenarios].
  - Stopping at 20 iterations beat the converged answer. The iteration cap had been acting as regularization [sim-only].
- **The enrichment estimate came from the start point, not the data.** Complete-data EM converges at rate ≥ 1 − edf/p. At M/n = 20, EM from the config defaults reported an SV/SNV prior-variance ratio of 1.65–1.66 whether the truth was 1× or 5×, and lost 14–24% held-out accuracy. MacKay/Fellner–Schall + Anderson reached the ML fixed point in 22–105 iterations [sim-only: gam-eval reproducer]. So no pre-cutover AoU SV-enrichment or SV-credit number is evidence.
- **Other defects found and fixed on the old path:**
  - convergence declared on rejected steps;
  - the prior variance substituted for the posterior variance;
  - binary posterior variances taken from Pólya-Gamma weights (2.5× too small at 4% prevalence [sim-only]);
  - a GIG moment floor bug.

## Model
- **One model, one path; approximate stages are warm starts, certified on full data (SPEC).** Block mean field shifted β by 16% on a toy [sim-only]. Block-diagonal LD alone over-predicted 2.7× at p/n = 20 [semi-real: design-credit, 1kGP-based genotypes].
- **Inference is EP-EB,** with a MacKay/EFS + Anderson hyper step, a trust region, warm-up, and a Newton-decrement + prediction-change certificate. Plain EM is never used.
- **Binary traits use a logistic link.** VB-probit lost 0.012–0.020 AUC, and EP-probit only tied logistic [sim-only: theory-inference].
- **No hand-chosen priors; the effect prior's mixing density is learned (SPEC 9c57144).**
  - The trigger: design-reliability's fifteen Gibbs scenarios had BayesR beating fixed-shape TPB by 0.025–0.10 R² [sim-only: founder mosaics]. The E6 scenarios showed the opposite [sim-only: design-trlocus]. A learned continuous mixing density nests both.
  - Hand-set constants were removed: TPB shapes, slab width, class-offset scale.
  - Continuous quantities get continuous priors, with no point mass at zero.
- **The r² prior offset has coefficient 1 by derivation.** Free EB could not identify it (range −0.29 to +0.69). With the offset, the full reliability prior gained +0.002–0.019 R² [sim-only: design-reliability].
- **Every variant's prior depends on SV context (SPEC 8a5a936).**
  - Why: a SNV's prior used to ignore whether it sits in an SV locus. Poorly imputed SVs (VNTR allele r² ≈ 0.3) reach the phenotype mostly through tag SNVs, which a generic prior over-shrinks.
  - With the true SV-informed prior, simulated R² rose 25–115% at n = 10–20k. EB from one trait alone recovered only 1–9% [sim-only: idea-svprior founder mosaics], so the annotation coefficients are pooled hierarchically across traits.
- **TR loci are one signed-length column each.** +4–12% where TRs carry signal, ≤ 0.8% cost otherwise; per-allele columns were neutral or negative [sim-only: design-trlocus].
- **Standardize by empirical SD, never √(2pq).** HWE scaling gives no shrinkage for draw-like SV columns, and over-weights ancestry-differentiated variants by (1+F).
- **Var(G | data) stays out of the Gram.** E[x²] corrections cost 0.2–3.2% R², and the SV share fell to about 0 [sim-only: design-trlocus].

## Data
- **Imputation is final; GLIMPSE2 is never re-run.** Genotype improvements must be post-hoc updates of the delivered output.
- **Background removal is value-matched,** because the modal floor mis-corrects 20–26% of non-carriers at multi-path records [sim-only: a simulation of the pop algorithm].
- **D* is a scale-only recalibration.** The isotonic shape added ≤ 0.012 r², and even that was optimistic [real: pilot50 imputation vs long-read truth].
- **GATK-SV fusion uses a general measurement model.**
  - Imputed SV dosages are draw-like, not Berkson [real: pilot50 truth slopes]. The calibrated-A estimator gave GATK-SV zero weight: VNTR fused r² 0.43 against an oracle of 0.76 [sim-only: sim F].
  - The rewrite meets its acceptance targets [sim-only: sim F].
  - Copy number is read from FORMAT/CN, and breakends are dropped.
- **Stack the 12k long-read panel members.** +7–8% R² at 100k and +18–19% at 50k, a Daetwyler projection the simulation agreed with; truth rows performed like imputed rows [semi-real: 1kGP-based genotypes with emulated imputation]. A truth-scale hyperprior would count the same data twice.
- **External annotations enter as per-source z² with learned weights.**
  - The Bai 2026 SV/VNTR releases, with Pan-UKB as the SNV side (the fastGWA SNV release lacks biochemistry and downloads 57× slower [real]).
  - An RSS likelihood was rejected: there is no public SV–SNV LD, the panel representations differ, and the binary phenotypes are mismatched.
  - Licences are no obstacle.
- **The panel is 21 traits,** about half diseases, chosen by power and phenotype quality and never by SV biology. Near-duplicate traits are dropped, and CKD is defined from lab values.

## Where the SV gain can come from
- **Imputed SV columns carry little beyond local SNVs.** ΔR²(column) = r²(1 − ρ²), with ρ² ≈ 0.95–0.98 on public data [real: pilot50 imputation vs long-read truth], so the column gain is ≤ ~5e-4 per trait. Imputing more samples does not raise the SV-specific ceiling.
- **The levers:**
  - SV-informed priors on nearby SNVs, the prior route;
  - new read evidence at TR and SV alleles, via a post-hoc TR-specialized likelihood update. The measurement is pending; k-mer genotyping did worse than imputation inside TRs [real: the imputation team's SV_FINDINGS test C];
  - reading the GATK-SV calls correctly, which adds copy number and SVs over 10 kb.
- **Portability is a headline dimension.** The SV gain was 2–4× larger in African-ancestry groups (e.g. +12% EUR, +39% AFR-ancestry, +20% AMR in the most polygenic scenario). A per-ancestry deviation prior lost 7–46%, so training stays pooled [sim-only: idea-ancestry msprime].

## Evaluation
- **The primary test is a fold-stratified paired test with a cross-fit pair term.**
  - The nested test fires on a 3% difference in regularization between arms, even when the SV model predicts worse [sim-only: design-traits].
  - A naive pooled cross-fit variance gave a 12–13% Type I rate, and 29–57% with adaptive fits [sim-only: theory-evaluation].
- **The headline is one pre-registered panel test with non-negative power weights.** Per-trait power is weak at 50k [sim-only: a power model].

## Engineering
- **The new path replaces the old one, and superseded code is deleted.** About 74% of sv_pgs (48,616 of 65,304 lines at `c2a9443`) is slated for removal, in ordered commits C0–C8 ([CUTOVER.md](CUTOVER.md)).
- **Continuous integration:** every validated increment lands on main through full-suite CI, sharded three ways.
- **Compute:** development runs on MSI with public or synthetic data. Training runs in the dedicated AoU workspace through its in-perimeter launch path.
- **All agents run on one model; helper subagents on other models are not used.**
- **No external contact** (email, support tickets, forums) is made without an explicit, per-message approval.
