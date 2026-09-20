# Design decisions, 2026-09-18/19

Each entry gives the ruling and the measurement or identity behind it. Simulations ran on MSI with public 1kGP/HPRC data or synthetic data. No AoU participant data was used outside the AoU workspace. Results measured inside it are tagged [in-workspace], and their values are not reproduced here (user rule, 2026-09-19).

Measurements carry the evidence tags defined in MODEL.md: `[sim-only]`, `[semi-real]`, `[real]`, `[machinery]`, `[est]`, `[in-workspace]` and `[provenance unknown]`. Under the evidence rule, a `[sim-only]` result checks the math and the code, but it is not evidence of an accuracy gain. Rulings resting on it stand until the neutral benchmarks (bench-real, bench-sim) re-measure them.

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
- **No joint multi-trait effect model.** The user: "we dont need multi-trait tbh". The unwired `pleiotropy_layer.py` (shared per-variant multiplier across traits) is deleted; it is recoverable from tag `archive/2026-09-19/old-path-final`. Cross-trait pooling of the prior's hyperparameters (level_c, θ) stays.
- **Binary traits use a logistic link.** VB-probit lost 0.012–0.020 AUC, and EP-probit only tied logistic [sim-only: theory-inference].
- **No hand-chosen priors; the effect prior's mixing density is learned (SPEC 9c57144).**
  - The trigger: design-reliability's fifteen Gibbs scenarios had BayesR beating fixed-shape TPB by 0.025–0.10 R² [sim-only: founder mosaics]. The E6 scenarios showed the opposite [sim-only: design-trlocus]. A learned continuous mixing density nests both.
  - Hand-set constants were removed: TPB shapes, slab width, class-offset scale.
  - Continuous quantities get continuous priors, with no point mass at zero.
- **The mixing density's roughness penalty is D3, with its null space profiled in the Schur-form evidence; D1 + D2 stays supported until the benchmarks confirm** (lead ruling, correcting 1dd33e4; [math/prior_sweep_a.md](math/prior_sweep_a.md)). These are numerical-property checks, not accuracy claims.
  - **The evidence form.** V = F + ½ log|S|₊ − ½ log|B + S| + ½ log|Nᵀ(B + S)N|, with N the penalty's null space: the null-space coordinates are profiled, and the Schur complement is integrated.
  - **The integrated flat-prior form is banned.** Integrating the null space under a flat prior drops the last term, and that form is the grid-dependent one.
    - Setup: exact normal means, a BayesR-type truth, 1,000 variants, one replicate, D3, spacings h = 0.5 / 0.25 / 0.125 / 0.1 [sim-only: prior sweep A].
    - Its level rose with the grid size (3449.8 / 3454.2 / 3470.0 / 3485.3), and its optimum jumped to λ → 0 (log λ −18.1, −19.7) at h ≤ 0.125.
    - It also diverges along the null space's collapse ray to a point mass at zero effect (3e302 on a weak 100-variant class).
  - **The Schur form is grid-convergent on the same case.** Optimum log λ 3.55 / 2.04 / 1.03 / 0.70, along a flat ridge (within 0.8 nats over log λ ∈ [−3, 5]); V at the optimum 3448.96 / 3449.79 / 3449.95 / 3449.99. Predictions changed by ≤ 0.46% from h = 0.5, and min eig(B + S) reaches 1e-12–1e-13 only as λ → 0.
  - **Not the conditional form either.** Integrating only the penalty's range at fixed null coordinates (−½ log|Qᵀ(B + S)Q|) drove λ to its bound in one run (ΔLPD −4.06 vs −0.71 nats per 1,000 variants). 1dd33e4 had wrongly attributed that result to profiling.
  - **D1 + D2 has no null space in sum-to-zero coordinates, so all three forms agree.** Its log λ went −1.04/3.08 → −1.03/3.12 → −1.01/3.15 across h = 0.5 / 0.25 / 0.125. The held-out log predictive moved ≤ 0.05 nats per 1,000 variants under refinement, and ≤ 0.5 at a ×100 wider range.
  - **Final ruling: D3 alone; D1 + D2 is dropped.** All three orders are grid-invariant under the profiled B + S evidence, but only D3's λ = ∞ limit (the log-normal) is range-invariant: ≤ 1e-4 nats when the range doubles or quadruples, while the D1 and D2 limits move by 4–100 nats (math-density) — a SPEC 131b205 violation for them. The engine's `ROUGHNESS_ORDER = 3` is the one constant.
  - **Penalty values are computed as |R x|² from square-root factors.** xᵀSx went negative in floating point along null directions (penalized objective 5687 against a log-likelihood of 173).
- **Basins are compared by the Tierney–Kadane-corrected evidence, not raw Laplace** (lead ruling, from the engine's measurement).
  - At identical penalty weights, the warm and flat starts reached two certified maxima (−H positive definite): V = 154.66 (|∇V| = 0.33) and V = 158.33 (|∇V| = 290). The second is a class deviation collapsing in width, with −H near-singular along that direction; its V is higher only through −½ log|−H| [sim-only: engine tests, 150 variants, 2 classes, normal means].
  - The rule: per eigen-direction of −H, the O(1) Laplace correction (1/8)κ₄ − (5/24)κ₃² from the standardized third and fourth derivatives. Where it exceeds 1/(2K), that direction's integral is an exact one-dimensional quadrature, finite under the deviation's proper pooling prior, and its width → 0 limit is the Gaussian-prior boundary model. The per-direction diagnostics go in the certificate.
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
- **D* is a scale-only recalibration.** The isotonic shape was not supported [in-workspace: measured in-workspace; value not reproduced here].
- **GATK-SV fusion uses a general measurement model.**
  - Imputed SV dosages are draw-like, not Berkson [semi-real: bench-sim public 1kGP re-imputation with GLIMPSE2 and Beagle]. The calibrated-A estimator gave GATK-SV zero weight: VNTR fused r² 0.43 against an oracle of 0.76 [sim-only: sim F].
  - The rewrite meets its acceptance targets [sim-only: sim F].
  - Copy number is read from FORMAT/CN, and breakends are dropped.
- **Stack the long-read panel members.** A Daetwyler projection, which the simulation agreed with, favours stacking (its projected gains depend on the in-workspace panel size and are not reproduced here); truth rows performed like imputed rows [semi-real: 1kGP-based genotypes with emulated imputation]. A truth-scale hyperprior would count the same data twice.
- **External annotations enter as per-source z² with learned weights.**
  - The Bai 2026 SV/VNTR releases, with Pan-UKB as the SNV side (the fastGWA SNV release lacks biochemistry and downloads 57× slower [real]).
  - An RSS likelihood was rejected: there is no public SV–SNV LD, the panel representations differ, and the binary phenotypes are mismatched.
  - Licences are no obstacle.
- **The panel is 21 traits,** about half diseases, chosen by power and phenotype quality and never by SV biology. Near-duplicate traits are dropped, and CKD is defined from lab values.

## Where the SV gain can come from
- **Imputed SV columns carry little beyond local SNVs.** ΔR²(column) = r²(1 − ρ²), with ρ² [in-workspace: measured in-workspace; value not reproduced here]; where ρ² is near 1, the column gain is small. Imputing more samples does not raise the SV-specific ceiling.
- **The levers:**
  - SV-informed priors on nearby SNVs, the prior route;
  - new read evidence at TR and SV alleles, via a post-hoc TR-specialized likelihood update. The measurement is pending; k-mer genotyping inside TRs was evaluated in-workspace (result not reproduced here);
  - reading the GATK-SV calls correctly, which adds copy number and SVs over 10 kb.
- **Portability is a headline dimension.** The SV gain was 2–4× larger in African-ancestry groups (e.g. +12% EUR, +39% AFR-ancestry, +20% AMR in the most polygenic scenario). A per-ancestry deviation prior lost 7–46%, so training stays pooled [sim-only: idea-ancestry msprime].

## Evaluation
- **The primary test is a fold-stratified paired test with a cross-fit pair term.**
  - The nested test fires on a 3% difference in regularization between arms, even when the SV model predicts worse [sim-only: design-traits].
  - A naive pooled cross-fit variance gave a 12–13% Type I rate, and 29–57% with adaptive fits [sim-only: theory-evaluation].
- **The headline is one pre-registered panel test with non-negative power weights.** Per-trait power is weak at 50k [sim-only: a power model].

## Engineering
- **The new path replaces the old one, and superseded code is deleted.** About 74% of sv_pgs (48,616 of 65,304 lines at `c2a9443`) is slated for removal, in ordered commits C0–C8 ([CUTOVER.md](CUTOVER.md)).
- **Landing:** every increment lands through the merge queue. The gate is the full suite on MSI (runq `cpu-node`) on the exact tip, under a memory ulimit, plus GPU tests when CUDA code changes. GitHub CI runs on main pushes as a secondary signal only (lead, 2026-09-19, after the user asked why GitHub CI was used when MSI is available).
- **Compute:** development runs on MSI with public or synthetic data. Training runs in the dedicated AoU workspace through its in-perimeter launch path.
- **All agents run on one model; helper subagents on other models are not used.**
- **No external contact, ever** (user order): no agent contacts anything outside this machine or sends anything to a human, for any reason. That bans email, web forms, tickets, posts, relays and file drops, gists, webhooks, notification services, third-party uploads, GitHub mentions, review requests, issues and PRs, and any repo other than SauersML's own. Pushing lane branches to SauersML/SV-PGS for the merge queue is the only outward write. Any step that would need outside contact is the user's: the user decides and acts.

## Later rulings, 2026-09-19
- **Stage 1 is dropped, provisionally, on cost** (compute_floor.md §3): one Stage 1 sweep costs 75–600 Stage 2 pass-equivalents [est]. The outer-convergence measurement first cited for it was withdrawn (compute_floor.md §10.1–10.3). The pipeline is Stage 0, Stage 2 from the prior, scoring.
- **The outer step is Newton on the total curvature B, with a trust region** (superseding the MacKay/EFS + Anderson hyper step; `anderson.py` is being deleted). Plain EP-EM is never used: its fixed-cavity M-step is ill-posed wherever A + S is indefinite, as it was at the true prior on chr22 LD [semi-real: public 1kGP-haplotype LD with simulated effects]. The certificate requires B + S positive definite, a Newton decrement within 1/(2K), and a bounded prediction change (MODEL.md §4).
- **Stage 2 uses certified marginal variances only** (`marginal_variances.py`), never block-Jacobi inverses and never stochastic variances in site updates. Block-Jacobi cavities were off by p99 25–54% on real chr22 LD [semi-real: bench-sim v7].
- **The mixing density is a continuous function; any grid is a converged quadrature; the range is data-derived** (SPEC 131b205), with D3 roughness (above).
- **No hard bins of continuous quantities.** DEL/DUP short/long classes and the small-indel class are merged; length enters as a learned smooth of log length, and 50 bp survives only as a reporting label. The SV-context window and K-nearest features became one learned distance kernel.
- **No variant is filtered by rarity** (SPEC); a column leaves computation only under a derived bound on its contribution. **Accelerator routes are CuPy, CPU routes NumPy/SciPy; JAX is not used** (SPEC).
- **No arbitrary constants** (user rule): each is derived, learned, measured at runtime, or its feature is deleted. `tests/test_no_arbitrary_constants.py` guards the package, with a registry of justified values and a strict list of pending ones.
- **Draw-like imputed SV/TR dosages are the measurement model** [semi-real: bench-sim v7, Beagle 5.5]: per-stratum D* recalibration, the A-map leakage correction (scale_model.md §3), and predictive variance from the calibrated Var(G|D) are all required.
- **Quantitative traits use an exact per-occasion measurement model** (`phenotype_measurement.py`): a learned heavy-tailed noise density and a learned Box–Cox transform replace the plausible-range and log-scale constants. It gained +0.110 ± 0.022 nats per reading for SBP and +0.191 ± 0.022 for pulse on held-out NHANES 2017–2020 [real, NHANES replicates: 60-second within-exam replicates, predictive density, not genetic signal]. Traits with no replicates use the transformed reading with reliability 1; lab-criterion disease rules keep their ranges until the disease model lands.
- **The AoU rule** (user, 2026-09-19): no AoU-related data outside the permitted environment, aggregates included; no download from any Google or AoU bucket; the AoU long-read panel is never used outside it. Public 1kGP/HGSVC panels are allowed for the benchmarks. bench-sim's cohort weights were rebuilt from the public 1kGP founder composition (PREREG amendment 7); results on the earlier cohort are labelled "withdrawn weights".
- **Evidence rule:** a lane's own simulation checks math only; accuracy claims come from bench-real, bench-sim, or in-workspace held-out data.
