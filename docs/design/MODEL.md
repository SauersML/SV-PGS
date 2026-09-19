# The one model

SV-PGS fits one Bayesian model to every variant. Speed comes from exact computation in the right order, never from a different model for some variants (SPEC).

**Evidence tags** (the evidence rule: a lane's own simulation checks math, never accuracy; accuracy claims come from the neutral benchmarks bench-real and bench-sim, or from in-workspace held-out data):
- `[sim-only]`: a lane's own simulation. It checks the math and the code, but it is not evidence of an accuracy gain.
- `[semi-real]`: real public haplotypes or real imputation, with simulated effects and phenotypes.
- `[real]`: measured on real data with nothing simulated: real phenotypes for accuracy claims, real genotype data for genotype-accuracy claims.
- `[machinery]`: exactness, pass counts, iterations or time, measured against an exact reference. Never an accuracy claim.
- `[est]`: an estimate from a derivation or cost model, not a measurement.
- `[provenance unknown]`: no traceable source.
- `[in-workspace]`: measured inside the AoU workspace. Its value is never reproduced outside it (user rule, 2026-09-19: no AoU-related data outside the permitted environment).

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
  - where a validated curve exists, a per-stratum, per-ancestry linear recalibration D* = μ + κ(DS − μ). Only the scale is kept; the shape term was not supported [in-workspace: measured in-workspace; value not reproduced here].
- **A tandem-repeat (TR) locus** is one signed-length column Z_i = Σ_a Δlen_a · D_ia, summed over every record in the locus, including indels under 50 bp.
  - That is the reference-invariant total length, E[ΣL | data].
  - TR loci have no per-allele columns. Equivalently, the allele effects get the rank-one prior β_a = Δlen_a · θ, so every allele dosage still enters.
  - Measured: +4–12% R² where TRs carry signal, ≤ 0.8% cost where they don't. Per-allele TR columns were neutral or negative [sim-only: design-trlocus coalescent simulator].
- **Non-TR multi-allelic bubbles** keep allele columns, with the REF state given its own column (exchangeable allele states), so predictions don't depend on which allele GRCh38 carries. This is accuracy-neutral [provenance unknown] and removes the reference dependence.

## 2. Measurement model

- **The column is a measurement of the true genotype G_j.** Imputed SV and TR dosages behave like confident posterior draws (κ ≈ √r² per stratum, and Var(DS)/Var(G) far above r²), not calibrated posterior means. On bench-sim's v7 cohort (public 1kGP haplotypes re-imputed with Beagle 5.5, 5,000 samples), SVs at MAF 1–5% had κ 0.80 against √r² 0.83 and a variance ratio of 0.89 against r² 0.69; TRs at MAF 1–5% had κ 0.81 against √r² 0.81. Only below MAF 1% does κ move partway toward 1 [semi-real: bench-sim v7, Beagle 5.5; GLIMPSE2 v2.0.0 showed the same pattern on bench-sim's earlier cohort, and its v7 check is pending]. So INFO and dosage-based r² cannot rank SV reliability, and the D* recalibration and the A-map (scale_model.md §3) are required.
- **Reliability:** r²_j = corr²(D_j, G_j). It is truth-calibrated and triad-corrected: r(D,T1)·r(D,T2)/r(T1,T2) over two independent long-read truths. A per-record model predicts it from site features (`sv_pgs/imputation_reliability.py`). The pipeline fits it inside the AoU workspace from the long-read truth rows, and its coefficients never leave the workspace.
  - Locus-level r² is predicted the same way for Z.
  - The r² estimate never enters the Gram. Adding E[x²] corrections shrinks every effect by r² (measured −0.2 to −3.2% R² [sim-only: design-trlocus]).
- **GATK-SV short-read calls are a second measurement B of the same G:** B = α_B + ρ_B·G + e_B. α_B is the false-positive intercept (its class precision is estimated inside the workspace [in-workspace: measured in-workspace; value not reproduced here]).
  - **Recalibrate first:** D* with triad-derived κ, within ancestry.
  - **Per-locus reliability:** for calibrated (Berkson) strata, from the per-locus V_A/V_G. Elsewhere, a mean anchor shrunk toward the stratum truth value, with weight σ²_s/(σ²_s + v + ω²_s), where ω²_s is the anchor's excess error measured on truth loci.
  - **Where a GATK-SV record carries its own read-depth genotype evidence** (decided per record by whether that evidence is present, never by a length cutoff), the three-source identity r²_A = C_AB·C_AC/(C_BC·V_A) replaces the anchor, once validated. Paired-end and split-read genotypes equal the final call (r = 1.0 [real: 1kGP GATK-SV freeze V3, chr22]), so they cannot be a third source.
  - **The fused column:** where GATK-SV calls, the best linear predictor from both sources. Where GATK-SV is a no-call, the recalibrated imputed dosage with the population slope κ_A = r²_A/ρ_A, not a slope fitted within the no-call group. Under strong genotype-dependent no-calls κ_A sits up to about 20% MSE above the within-group oracle (draw-like A, p = 0.2, 50% carrier no-calls: 0.206 vs 0.162), while the estimated group predictor was often far worse (p = 0.05, 27% carrier no-calls: 0.54 vs 0.23) [sim-only: bug-store exact MNAR check]. GATK-SV no-calls are informative (a quality-score band), so inside the workspace they are filled with E[B | SL].
  - **Acceptance (met in simulation):** gap to the oracle ≤ 0.001 with known r²_A, and ≤ 0.01 (MCAR) / ≤ 0.025 (MNAR) with estimated r²_A. The old closure's VNTR gap was −0.34 [sim-only: sim F].
- **The long-read panel members are stacked as an extra store half** of hard calls on the same sites, with their own cohort covariate. A Daetwyler projection, which the simulation agreed with, favours stacking, and truth rows performed like imputed rows for the same people [semi-real: 1kGP-based genotypes with emulated imputation]. The projected gains depend on the in-workspace panel size and are not reproduced here. Folds are kinship-grouped across the union of both halves.

## 3. Prior

- **Effect prior:** β_j | class c ~ ∫ N(0, u_j · s) g_c(s) ds.
  - The mixing density g_c is learned nonparametrically by empirical Bayes for each class. It is a continuous function of t = log s, log g_c = η + δ_c: a shared shape η plus a class deviation δ_c.
    - Each carries a learned weight on its third-order roughness ∫(f‴)² dt. The deviations' location and width share one learned precision.
    - The third order is derived: its null space, the normal density in log s, is the only proper λ = ∞ limit. The flat and power-law limits of lower orders are not normalizable, so they depend on where the range is cut.
    - Grids are only quadrature (composite Gauss–Legendre, certified a posteriori), over a range extended until the tails are negligible. No grid constant is set by hand (SPEC 131b205).
  - There is no point mass at zero; effects are continuous, so the prior is continuous.
  - TPB and BayesR-like shapes are special cases. A learned mixing density exists because a fixed-shape TPB lost to BayesR by 0.025–0.10 R² in fifteen Gibbs scenarios [sim-only: design-reliability founder mosaics]. The E6 scenarios showed the opposite [sim-only: design-trlocus].
- **Scale model:** log u_j = level_c + log r̂²_j + d_jᵀθ.
  - The r² coefficient is exactly 1 by derivation: the prior on the true-genotype effect maps to the observed column through r². Free EB could not identify it [sim-only: design-reliability].
  - level_c and θ get hierarchical EB priors whose mean and variance are learned across classes and across traits (`da928bd`). Cross-trait pooling matters: a single trait's EB recovered only 1–9% of the true-prior gain, against 25–115% for the true prior at n = 10–20k [sim-only: idea-svprior founder mosaics].
- **d_j, the same design for every variant** (SNVs included, SPEC 8a5a936). Continuous entries are smooths with learned smoothness, never bins; discrete entries stay discrete.
  - variant type, length and repeat status (SPEC). Classes carry no length bins: a deletion, an insertion or a duplication is one class at every size from 1 bp up, small indels included, and length enters only here, as a learned smooth of log length. The 50 bp SV definition only labels reported results;
  - SV context:
    - one learned distance kernel over every other SV allele of the chromosome, f_j = Σ_k H_k · w_c(log(1 + gap_jk)) per class, with w_c a smooth of learned smoothness (a cubic B-spline basis in log(1 + gap)) and a log-length term; nesting in an SV is its own per-class term. No window, K or frequency cutoff;
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
  - the coefficients maximize log Z_EP − ½xᵀS_λx by Newton with the total curvature B = −∇² log Z_EP, with EP re-solved (docs/design/math/ep_eb.md §1.4);
  - the penalty weights maximize the Laplace evidence with B directly: an exact gradient, and each weight compared at λ = ∞ (exactly, in its penalty's null space) and inside. The λ = ∞ score test is ½(q − d + c), with c = −tr(H_KK⁻¹KᵀD_xB[v]K) the curvature change that the Gaussian (Tipping–Faul) form omits. Fellner–Schall is not used: it creeps toward infinite optima and is not the Laplace maximizer for a non-Gaussian likelihood;
  - EP is unclipped, Newton on the moment equations with the Opper–Winther double loop as the fallback (the dense reference, `tests/ep_eb_reference.py`);
  - a warm-up before the first hyper step.
  - Plain EM converges at rate ≥ 1 − edf/p, about 0.99 at production scale. From the defaults its reported SV/SNV enrichment was 1.65 whatever the truth [sim-only: gam-eval reproducer].
- **The production engine** (`sv_pgs/scale_mixture_ep.py`) holds everything that involves the prior, for a stage that supplies q's means and marginal variances:
  - g on a uniform lattice in t (nodal log g, roughness λh⁻⁵‖Δ³η‖² from square-root factors); the lattice's floor (flat-kernel bound), top (largest kernel mode), spacing (complex-strip trapezoid bound) and tails come from the data and a tolerance (math-density's rules);
  - the ruled layout: η shared, δ_c per class with its own roughness weight, one Gaussian pooling precision on the deviations' location and width, no class level; η's null space profiled;
  - exact tilted moments, unclipped mean-matched sites, cavities, the MacKay/REML noise update;
  - the fixed-cavity objective with its exact gradient and Hessian, a spectrum-shifted Newton M-step, and the Laplace evidence with the observed curvature and its exact gradient (third derivatives of log Z).
  - The λ step: every weight in (0, ∞] with one exact edge, λ = ∞ (the block's penalized directions removed; V's limit there, verified by Richardson extrapolation), released when V at the upper end of the resolvable range is higher. There is no λ = 0 edge (lead ruling, replacing the earlier one): dropping a block profiles its directions under a flat prior, an improper model whose value bounds every proper-prior V(ρ) from above, while V ~ (r_i/2) log λ → −∞, so comparing the two let a null annotation go unpenalized. The lower end of each weight's range is its resolvable bound, and a −inf start weight means that bound. The evidence is always the proper-prior V; the penalty's null space keeps its flat prior at every λ; a trust region in ρ and in x (value-only trial passes, stopping at 1/(2K) nats); the structural starts (warm, flat, and the best log-normal shared density over a grid, the global start for the density's null model), keeping the certified maximum (B + S positive definite) with the highest corrected V, and a refit from the best certified solution.
  - The B-evidence (lead ruling (b)): V uses the EP-re-solved curvature B (speed-ep's closed form, `_total_curvature`: a linear response solved by GMRES through q's `solve` and `variance_jvp`), so B enters V and the certificate; the fixed-cavity Hessian only shapes the ρ search direction, and a step is accepted only if V rises. B matches EP-re-solved differences of the evidence gradient to 1e-9 [machinery: dense EP].
  - **V is the Tierney–Kadane-certified evidence in every comparison** (acceptance, edges, basins, the stationarity check); the Laplace form's exact fixed-cavity gradient only steers.
    - Per standardized Schur direction the O(1) term is κ4/8 + 5κ3²/24, from exact third and fourth derivatives. The largest terms are replaced by exact QUADPACK line integrals until the rest sum to at most tol/2, and each integral is resolved to tol/(2m) in its log, so V is certified to tol in total.
    - Why every comparison, not only basins: the Laplace V rises without bound at a fold of the inner maximum, where the data's negative curvature nearly cancels the penalty. There the search was drawn to a point with TK term 9e10, whose corrected V was 10 nats lower than its Laplace V and 3.6 nats below the neighbouring basin's [sim-only: engine test problem].
  - The determinants come from one Cholesky factor in the basis [N, C], with C keeping every coordinate outside N's support: the Schur log-determinant is the trailing diagonal, and W = (B+S)⁻¹ − N(N'(B+S)N)⁻¹N' is L⁻¹'s trailing rows squared, never a difference of inverses. V is not certified where Demmel's componentwise rounding bound (n+1)ε Σ|W_ij|√(M_ii M_jj) exceeds the tolerance.
  - The final stationarity check: one central difference of V per interior weight. Each side's V is certified to e = (2 tol/(n·3^{4/3}))^{3/4} s^{1/4}, with s = ½(edf + λ‖Rx‖²) bounding |V''| and |V'''|, at the optimal step h = (3e/s)^{1/3}, so the error E = h²s/6 + e/h takes a quarter of the tolerance. The sides restart from the base's x, and the h/2 difference must agree within both bounds, which detects a switch of inner maximum. The certificate is ½Σ(|c| + E)²/s ≤ tol (`stationarity_gain`); while it fails, the search moves along the difference and resumes, edges included.
  - Once the interior ascent converges, every finite weight is compared with both of its edges, and the best edge that raises V past the tolerance is taken.
- **The outer loop** (in flight on branch `wip/engine-driver`, not yet on main): Newton on the EP evidence with the total curvature B and a trust region, from a certified EP fixed point at every step.
  - Plain EP-EM is never used. Its fixed-cavity M-step is ill-posed wherever A + S is indefinite, which happens on real LD: at the true prior of a pooled chr22 problem, B + S had negative eigenvalues in every measured configuration, so the true prior is a saddle, not a fixed point [semi-real: public 1kGP-haplotype chr22 LD with simulated effects; first measured on bench-sim's withdrawn-weights cohort, re-checked on the v7 cohort by the engine's real-LD regression test].
  - The earlier outer-rate measurement (a spectrum "with no slow direction", 1–3 accelerated outer steps, "plain EP-EM diverges in 7 of 16") was withdrawn: it linearized at the true prior on a non-invariant subspace (compute_floor.md §10). The production outer rate is unmeasured; it will be measured on the engine at chr22 scale with learned λ.
- **Certificate:** a fit is accepted only if
  - B + S is positive definite at the returned point (certified Cholesky, not a sign test);
  - the Newton decrement of the hyper objective is within the tolerance 1/(2K) nats, with B recomputed at the returned point;
  - the certifying Newton step moves the posterior mean by at most p_eff/K in its own posterior metric, measured at the step's EP fixed point. The evidence alone can be flat along a direction where predictions still move (a 3.3e-6 eigenvalue along the profiled null space moved predictions up to 73% within its local radius [sim-only: engine test, p = 40, n = 60]).
  - Trials where EP has no fixed point are counted as unresolved and never pass. Parallel EP leaves a few sites in limit cycles, so the per-site maximum is not a certificate. The certificate is recorded in the artifact, and a fit without it is not accepted.
- **Binary traits:** logistic EP with a posterior predictive computed by the trapezoid rule, with its step and truncation derived a priori from the integrand's strip of analyticity so it is exact to fp64 for any predictor variance. Probit was rejected: VB-probit lost 0.012–0.020 AUC, and EP-probit only tied logistic [sim-only: theory-inference].
- **Predictive variance:** K exact posterior draws by perturb-and-solve (exact for a positive-definite global precision, with non-positive sites split out; dual_solve.md), riding Stage 2's passes and scored in the same single read. Credible intervals use Student-t_K quantiles, which are exact for any K. K itself is a registered pending constant, to be derived from the Monte Carlo error target of every reported quantity.

## 5. Pipeline

- **Stage 0: one phenotype-independent genotype pass** (`genotype_statistics.py`, `code_products.py`).
  - Exact int8 → int32 LD-block Grams on the FWL-projected columns, per-variant sums, and X'Y for every trait × fold.
  - The TR length columns as an exact sparse map of stored codes.
  - The tagging and ρ² features.
  - The candidate set by the information rule N·Var(D_j)·r̂²_j·τ²_c ≥ c. It is variance-based, so copy-number rows are kept, and in exact Bayes it is a compute knob only.
- **Stage 1 is dropped, provisionally** (lead, 2026-09-19), on cost alone: one Stage 1 sweep's variance refresh costs 75–600 Stage 2 pass-equivalents [est: compute_floor.md §3]. Stage 2 starts from the prior itself: moment-matched sites τ_j = 1/E_prior[β_j²], ν = 0, a zero mean, the start density, and the covariate-only residual variance as the noise. The Stage 1 slot interface is being removed in the engine driver branch.
  - The decision is revisited only if the outer rate measured at the fitted maximum shows production needs more outer steps × passes than a Stage 1 sweep costs.
- **Stage 2: full-data certification.**
  - **The Gaussian E-step is `dual_solve.py`** (DualGaussian): the solves in dual form, sample-side state, every model, fold, probe and draw in one pass. Non-positive sites and spikes are eliminated exactly, with a Schur certificate; the operand digits come from each call's error budget (dual_solve.md). At n = 30,000 × 483,944 variants on an A40, a cold iterate for 12 models took 149–180 s, 6–9 reads and 3–6 CG iterations, and 192 draws took 16–21 s [machinery: semi-real genotypes, simulated sites]. The older block-Jacobi PCG in `exact_polish.py` needed 17–28 passes [sim-only: synthetic store].
  - **Marginal variances come from `marginal_variances.py`**, never from block-Jacobi inverses (those are conditional on the other blocks' effects, which moves the EP fixed point) and never from stochastic estimates in the site updates.
    - The leave-block-out marginals: exact elimination of the resolved sites plus a neighbour-window Woodbury, with a deterministic equivalent only for the far field. Every block carries an information certificate, which a failing block must pass before the fit may use its cavities.
    - Against the dense inverse with LD across cuts: max per-variant relative variance error 0.5–3.5%, against 16–58% for block-Jacobi [machinery: dense inverse].
    - On real chr22 LD (bench-sim, 12k variants × 50k people, 10 PCs projected), the implied cavity-precision error was median 2.7% / p99 23% at block cap 1024 and 0.33% / 6.4% at cap 4096, against 4.2% / 34% and 2.2% / 20% for block-Jacobi. The information certificate's interval covered the true block error in every block of every case [semi-real: public 1kGP-haplotype LD; 10 PCs only, not the production covariate set; to be re-measured on bench-sim v7 with the production covariates before these numbers are relied on]. At cap 1024 the window form is not yet accurate enough to certify EP; an exact repair for flagged blocks is in progress.
    - `variance_jvp` gives the variance map's derivative −diag(Σ diag(w) Σ) for the curvature products: resolved rows exact (1e-13); bulk entries mean relative error 0.2–0.6% (99th percentile 1–4%); block sums within 0.7% [machinery: dense derivative].
  - This stage carries the real weight: a block-diagonal fit alone was 2.7× off at p/n = 20 [semi-real: design-credit].
- **Scoring** (`fast_scoring.py`): every trait × fold model and its posterior draws in one read of the store. It is exact to 1e-13; an H100 does 100k × 17.3M in about 100 s [machinery: synthetic store]. The binary predictive uses an a-priori trapezoid rule with no iteration.

## 6. Measured and rejected (do not re-propose without new evidence)
A rejection resting only on a lane's own simulation (`[sim-only]`) is provisional. It stands until it is re-measured on the neutral benchmarks: bench-real (real held-out data) or bench-sim (real haplotypes, a misspecified truth family, sealed seeds). A `[sim-only]` rejection is not grounds to refuse a new measurement there.
- A multi-ancestry deviation prior β_{j,a} = β_j + δ_{j,a}: −7 to −46% non-EUR accuracy [sim-only: idea-ancestry msprime].
- Refining SV dosages from information already in the imputed files (locus kernels, tag-SNV shrinkage, boosted trees, monotone recalibration for SVs): no PGS gain [r² in-workspace, measured in-workspace; value not reproduced here; sim-only PGS impact: design-genorefine].
- Sibling-allele coupling (GR-2) beyond the locus model: no gain [r² in-workspace, measured in-workspace; value not reproduced here; sim-only PGS impact: design-genorefine].
- Fine-structure signal beyond PCs carried by SVs: ΔR² −0.00002 ± 0.00003 [sim-only: idea-ancestry msprime].
- Per-class free tails estimated from each class's own columns: they cost more than they returned [sim-only: idea-bigcausal, at n·h²/p ≈ 4.2 against production's ≈ 0.009].
- Sparse arithmetic kernels for rare columns in Stage 2: they lose to dense tensor cores at ~1,800 right-hand sides [sim-only: synthetic store; timing]. Rare variants are sparse in storage only.
- gamfit for the r² calibration model: it failed REML certification, and only tied the linear model [in-workspace: measured in-workspace; value not reproduced here].

## 7. Related work and evidence

External results that bear on the model. The numbers here are the authors' own, not our measurements.
- **EP-family inference over mean-field VB.** At proportional p/n, naive mean-field VB (the CAVI behind mr.ash) gets the log-normalizer wrong and is overconfident ([Qiu 2023](https://arxiv.org/abs/2310.09931)). The TAP free energy has a local minimizer with consistent posterior marginals, which AMP can find ([Celentano, Fan, Lin & Mei 2023](https://arxiv.org/abs/2311.08442)). EB estimation of g through the mean-field surrogate is consistent only when that objective has a dominant optimizer ([Mukherjee, Sen & Sen 2023](https://arxiv.org/abs/2309.16843)). EP, from which VAMP is derived, is in the TAP/AMP family (§4).
- **gVAMP, the closest method:** VAMP whole-genome regression with an EM-learned spike-plus-Gaussian-grid prior, at biobank scale ([bioRxiv 10.1101/2023.09.14.557703](https://doi.org/10.1101/2023.09.14.557703); [code](https://github.com/medical-genomics-group/gVAMP)). By its authors' account it relies on damping, LD pruning and a training-R² stopping rule with no convergence guarantee, and its prior is a discrete grid with no annotation or SV dependence. SV-PGS accepts a fit only with its certificate (§4), learns a continuous mixing density, makes each scale depend on annotations and SV context (§3), and prunes no variant (SPEC). A head-to-head on the benchmarks is pending.
- **The deconvolution start:** for correlated Gaussian sequences, g estimated by the composite (independence) marginal likelihood is nearly minimax, at an effective sample size set by the correlation's spectral radius ([Han & Zhang 2026](https://arxiv.org/abs/2607.03596)). This is the warrant for starting g from a deconvolution of the marginal estimates under the independence approximation.
- **SV enrichment:** MiXeR-SV, from GWAS summary statistics with an SV-specific effect variance and a 1KG-ONT LD reference, finds SVs significantly enriched in 31 of 105 traits, median 14.8× (6.6–61.5×), and 3.4–32% of variant heritability in those traits; 59–62% of common SVs have r² ≥ 0.5 with some reference SNP ([Nguyen et al. 2026, preprint](https://pmc.ncbi.nlm.nih.gov/articles/PMC13484855/); [code](https://github.com/precimed/mixer_sv)). External support for a class-specific prior scale (§3); it builds no score.
- **Imputation loss:** from a 482-haplotype long-read assembly panel imputed into UK Biobank, common SVs (MAF ≥ 1%) impute at median r² 0.78, and only 958 of 17,335 SV–trait associations are unlikely to be driven by nearby small variants ([Bai et al. 2026, ImputeSV](https://doi.org/10.1038/s41588-026-02612-z)). External support for treating an imputed dosage as a measurement with its own reliability (§2).
- **Portability:**
  - ANCHOR: in admixed UK Biobank participants, effect sizes in African- and European-ancestry segments correlate at 0.98 ± 0.07 across 53 quantitative traits, so the portability loss is tagging, not effect change ([Hu et al. 2025](https://doi.org/10.1038/s41588-024-02035-8)).
  - Cis-expression effects are nearly shared between European and Yoruba LCLs (genetic correlation ≈ 0.95), and causal allele-frequency differences drive the loss: portability falls by more than 32% when the causal variant is common in the training ancestry but rare in the target ([Saitou, Dahl, Wang & Liu 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11639078/)).
  - Together they imply that a score on causal variants ports better than one on tags, and that a rare causal SV still loses power across ancestries through its frequency alone.
