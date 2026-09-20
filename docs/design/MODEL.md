# The one model

SV-PGS fits one Bayesian model to every variant. Speed comes from exact computation in the right order, never from a different model for some variants (SPEC).

**Evidence tags** (scratchpad EVIDENCE_RULE):
- `[sim-only]`: a lane's own simulation. It checks the math and the code, but it is not evidence of an accuracy gain.
- `[semi-real]`: real public haplotypes or real imputation, with simulated effects and phenotypes.
- `[real]`: measured on real data with nothing simulated: real phenotypes for accuracy claims, real genotype data for genotype-accuracy claims.
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

- **The column is a measurement of the true genotype G_j.** Imputed SV and TR dosages behave like confident posterior draws (κ ≈ √r² per stratum), not calibrated posterior means [semi-real: bench-sim, public 1kGP haplotypes re-imputed with real GLIMPSE2 v2.0.0 and Beagle 5.5]. So INFO and dosage-based r² cannot rank SV reliability.
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
  - At small p the evidence pools a sharply bimodal g to the log-normal λ = ∞ edge, by design: its roughness is not supported by the data. On the oracle's ×1000 v7 windows (80 variants, truth 0.97 N(0,1) + 0.03 N(log 100, 1) in t, roughness J = 2420) the engine certifies λ = ∞ at log Z_EP 23.586, the oracle's certified V(∞) 23.592, 0.2 nats below the truth's own log Z_EP, and every finite-λ basin also tops out there [semi-real]. Such a g should become identifiable at genome scale; bench-sim's out-of-family `fixed_count` and `nonscale_heavy` scenarios test that.
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
  - the coefficients maximize log Z_EP − ½xᵀS_λx by Newton with the total curvature B = −∇² log Z_EP, with EP re-solved (docs/design/math/ep_eb.md §1.4). The outer loop (`fit_hyperparameters`) sets the weights at each certified EP fixed point, then moves x by (B + S)⁻¹g, globalized by Deuflhard's natural monotonicity test (g'(B + S)⁻¹g at the trial's fixed point must fall, else the step halves), and by a trust region where B + S is indefinite; it certifies only where B + S is positive definite, and only when the certifying step also moves q's mean by at most p_eff/K in q's posterior metric. It is never plain EP-EM (x to the fixed-cavity maximizer): that step solves with A + S, maps the error by I − (A + S)⁻¹(B + S), diverges where the pencil exceeds 2, and has no maximum to move to where A + S is indefinite (lead ruling, 2026-09-19). At the true prior on real LD scaled to genome size, B + S has negative eigenvalues [semi-real: speed-floor], so the trust region is the production case. (speed-floor's earlier pencil [0.89, 5.4] and 7-of-16 divergence were withdrawn: they dropped the directions where A + S ≤ 0.);
  - the penalty weights maximize the Laplace evidence with B directly: an exact gradient, and each weight compared at λ = ∞ (exactly, in its penalty's null space) and inside. The λ = ∞ score test is ½(q − d + c), with c = −tr(H_KK⁻¹KᵀD_xB[v]K) the curvature change that the Gaussian (Tipping–Faul) form omits. Fellner–Schall is not used: it creeps toward infinite optima and is not the Laplace maximizer for a non-Gaussian likelihood;
  - EP is unclipped, Newton on the moment equations with the Opper–Winther double loop as the fallback (the dense reference, `tests/ep_eb_reference.py`);
  - a warm-up before the first hyper step.
  - Plain EM converges at rate ≥ 1 − edf/p, about 0.99 at production scale. From the defaults its reported SV/SNV enrichment was 1.65 whatever the truth [sim-only: gam-eval reproducer].
- **The production engine** (`sv_pgs/scale_mixture_ep.py`) holds everything that involves the prior, for Stage 2, which supplies q's means, marginal variances and linear responses:
  - g on a uniform lattice in t (nodal log g, roughness λh⁻⁵‖Δ³η‖² from square-root factors); the lattice's floor (flat-kernel bound), top (largest kernel mode), spacing (complex-strip trapezoid bound) and tails come from the data and a tolerance (math-density's rules);
  - the ruled layout: η shared, δ_c per class with its own roughness weight, one Gaussian pooling precision on the deviations' location and width, no class level; η's null space profiled;
  - exact tilted moments, unclipped mean-matched sites, cavities, the MacKay/REML noise update;
  - the fixed-cavity objective with its exact gradient and Hessian, a spectrum-shifted Newton M-step, and the Laplace evidence with the observed curvature and its exact gradient (third derivatives of log Z).
  - The λ step: every weight in (0, ∞] with one exact edge, λ = ∞ (the block's penalized directions removed; V's limit there, verified by Richardson extrapolation), released when V at the upper end of the resolvable range is higher. There is no λ = 0 edge (lead ruling, replacing the earlier one): dropping a block profiles its directions under a flat prior, an improper model whose value bounds every proper-prior V(ρ) from above, while V ~ (r_i/2) log λ → −∞, so comparing the two let a null annotation go unpenalized. The lower end of each weight's range is its resolvable bound, and a −inf start weight means that bound. The evidence is always the proper-prior V; the penalty's null space keeps its flat prior at every λ; a trust region in ρ and in x (value-only trial passes, stopping at 1/(2K) nats); the structural starts (warm, flat, and the best log-normal shared density over a grid, the global start for the density's null model), keeping the certified maximum (B + S positive definite) with the highest corrected V, and a refit from the best certified solution.
  - The B-evidence (lead ruling (b)): V uses the EP-re-solved curvature B (speed-ep's closed form, `_total_curvature_columns`: a linear response solved through q's `solve` and `variance_jvp`, exactly by the posterior's `linear_response` where it can factor it, as the small-n route does, and otherwise by GMRES on the map's linear part), so B enters V and the certificate. The response is solved only for the directions the current views ask for: the free coefficients of the current edges, extended when an edge is released (`CurvatureCorrection`). V's ρ-gradient is its own, W_B in the trace terms, and a step is accepted only if the corrected V rises. B matches EP-re-solved differences of the evidence gradient to 1e-9 [machinery: dense EP].
  - **V is the Tierney–Kadane-certified evidence in every comparison** (acceptance, line searches, edges, basins, the stationarity check).
    - Per standardized Schur direction the O(1) term is κ4/8 + 5κ3²/24, from exact third and fourth derivatives. The largest terms are replaced by exact QUADPACK line integrals until the rest sum to at most tol/2, and each integral is resolved to tol/(2m) in its log, so V is certified to tol in total.
    - Why every comparison, not only basins: the Laplace V rises without bound at a fold of the inner maximum, where the data's negative curvature nearly cancels the penalty. There the search was drawn to a point with TK term 9e10, whose corrected V was 10 nats lower than its Laplace V and 3.6 nats below the neighbouring basin's [sim-only: engine test problem].
  - The determinants come from one Cholesky factor in the basis [N, C], with C keeping every coordinate outside N's support: the Schur log-determinant is the trailing diagonal, and W = (B+S)⁻¹ − N(N'(B+S)N)⁻¹N' is L⁻¹'s trailing rows squared, never a difference of inverses. V is not certified where Demmel's componentwise rounding bound (n+1)ε Σ|W_ij|√(M_ii M_jj) exceeds the tolerance.
  - The final stationarity check (`_stationarity`), from V's analytic ρ-gradient: the Laplace part's, and the corrections' with each replaced direction held (their rotation with ρ couples a line to the others, the mixed terms the product of line integrals already leaves out), with its error bound E. The curvature K is a forward difference of that gradient per interior weight, on the side the gradient climbs, at h = 2√(E/s), which balances hs/2 against 2E/h with s = ½(edf + λ‖Rx‖²) the logistic bound on V's third derivative. Where that side has no certified maximum within h, the basin ends there (a fold): the difference is taken on the other side, and that weight's gain is at most (|g| + E)h, what V can climb before the fold. The remaining gain (`stationarity_gain`) is that plus the Newton decrement ½ rᵀK⁻¹r, r = |g| + E, over the other interior weights; an indefinite K has none. **This ρ-certificate is a Newton decrement on the local model, with an indefinite model refused: the same standard as the x-certificate, and not a global curvature bound over the step.** (A certified lower bound on −V'' over the step from s would need |g| ≲ 10⁻³, since s overstates |V''| about thirtyfold here [sim-only].) While it exceeds the tolerance the search takes the Newton step, accepted only when the corrected V rises; a certifiably better side (another basin) resumes the search there, edges included. The outer certificate adds this remaining gain to the x decrement and the weights' realized gain.
  - Once the interior ascent converges, every finite weight is compared with both of its edges, and the best edge that raises V past the tolerance is taken.
- **Certificate:** the Newton decrement of the hyper objective (in nats) together with the relative prediction change ‖XΔμ‖/‖Xμ‖. Parallel EP leaves a few sites in limit cycles, so the per-site maximum is not a certificate. The certificate is recorded in the artifact, and a fit without it is not accepted.
- **Binary traits:** logistic EP with a posterior predictive computed by the trapezoid rule, with its step and truncation derived a priori from the integrand's strip of analyticity so it is exact to fp64 for any predictor variance. Probit was rejected: VB-probit lost 0.012–0.020 AUC, and EP-probit only tied logistic [sim-only: theory-inference].
- **Predictive variance:** K = 64 exact posterior draws by perturb-and-solve, riding Stage 2's passes and scored in the same single read.

## 5. Pipeline

- **Stage 0: one phenotype-independent genotype pass** (`genotype_statistics.py`, `code_products.py`).
  - Exact int8 → int32 LD-block Grams on the FWL-projected columns, per-variant sums, and X'Y for every trait × fold.
  - The TR length columns as an exact sparse map of stored codes.
  - The tagging and ρ² features.
  - The candidate set by the information rule N·Var(D_j)·r̂²_j·τ²_c ≥ c. It is variance-based, so copy-number rows are kept, and in exact Bayes it is a compute knob only.
- **No Stage 1, provisionally** (lead ruling, 2026-09-19), on cost: the measured Stage 1 costs ~800 GPU-h (COMPUTE.md). The outer-contraction measurement first cited for it ("no slow direction, 1–3 outer steps") was withdrawn; speed-floor is re-measuring at the pooled fixed point.
- **Stage 2: full-data EP-EB** (`full_data_fit.py` over `dual_solve.py` and `marginal_variances.py`), one per trait × fold, from the prior itself: moment-matched sites τ_j = 1/E_prior[β_j²], ν = 0, the start density, and the covariate-only residual variance as the noise.
  - Every cavity comes from the certified marginals (`marginal_variances.py`), refreshed at the sites it is used with. Block-Jacobi variances may appear only as a preconditioner: their cavity-precision errors were median 2.6–4.9% and p99 28–58% at production, and the second-order series diverges at denser signal [semi-real: speed-floor, bench-sim chr22 real-haplotype LD, simulated effects].
  - The EP fixed point is certified at a refresh: the undamped update's move of the mean, Σ(δν − δτ∘μ), is at most p_eff/K in the posterior metric, and the noise update's evidence gain is at most 1/(2K). Between refreshes, mean-only EP runs with the cavity precisions frozen.
  - The outer loop is the engine's Newton-B loop (§4); the full fit is certified when every model's Newton-B decrement plus its weights' remaining gain is at most 1/(2K).
- **The previous Stage 2 E-step** (`exact_polish.py`).
  - Block-Jacobi PCG on the FWL-projected system, which needed 17–28 passes where block Gauss–Seidel needed over 40 [sim-only: synthetic store].
  - Control-variate Hutchinson estimates of diag(Σ), using the block inverse as the control variate. These are being replaced by `marginal_variances.py`, the leave-block-out marginals:
    - Block-Jacobi inverses are variances conditional on the other blocks' effects, which biases the EP fixed point.
    - The replacement is exact elimination of the resolved sites plus a neighbour-window Woodbury, with a deterministic equivalent only for the far field. Every block carries a probe certificate.
    - Measured against the dense inverse, with LD across cuts: max per-variant relative error 0.5–3.5%, below the equivalent's scale ‖K_S⁻¹‖_F/tr K_S⁻¹. Block-Jacobi was off by 16–58% [machinery: dense inverse].
    - `variance_jvp` gives the variance map's derivative −diag(Σ diag(w) Σ) for the hyperparameter curvature products:
      - resolved rows are exact (1e-13);
      - bulk entries have mean relative error 0.2–0.6% (99th percentile 1–4%);
      - block sums are within 0.7% [machinery: dense derivative].
      - Distant pairs enter through the block sandwich diag(Σ_bb R_b Σ_bb). Per-variant norms overstated them 3–10×, because LD partners absorb the chance coupling.
  - Posterior draws.
  - A block-diagonal fit alone was 2.7× off at p/n = 20 [semi-real: design-credit]; the full-data stage carries the real weight.
- **Scoring** (`fast_scoring.py`): every trait × fold model and its posterior draws in one read of the store. It is exact to 1e-13; an H100 does 100k × 17.3M in about 100 s [sim-only: synthetic store; timing].

## 6. Measured and rejected (do not re-propose without new evidence)
A rejection resting only on a lane's own simulation (`[sim-only]`) is provisional. It stands until it is re-measured on the neutral benchmarks: bench-real (real held-out data) or bench-sim (real haplotypes, a misspecified truth family, sealed seeds). A `[sim-only]` rejection is not grounds to refuse a new measurement there.
- A multi-ancestry deviation prior β_{j,a} = β_j + δ_{j,a}: −7 to −46% non-EUR accuracy [sim-only: idea-ancestry msprime].
- Refining SV dosages from information already in the imputed files (locus kernels, tag-SNV shrinkage, boosted trees, monotone recalibration for SVs): no PGS gain [r² in-workspace, measured in-workspace; value not reproduced here; sim-only PGS impact: design-genorefine].
- Sibling-allele coupling (GR-2) beyond the locus model: no gain [r² in-workspace, measured in-workspace; value not reproduced here; sim-only PGS impact: design-genorefine].
- Fine-structure signal beyond PCs carried by SVs: ΔR² −0.00002 ± 0.00003 [sim-only: idea-ancestry msprime].
- Per-class free tails estimated from each class's own columns: they cost more than they returned [sim-only: idea-bigcausal, at n·h²/p ≈ 4.2 against production's ≈ 0.009].
- Sparse arithmetic kernels for rare columns in Stage 2: they lose to dense tensor cores at ~1,800 right-hand sides [sim-only: synthetic store; timing]. Rare variants are sparse in storage only.
- gamfit for the r² calibration model: it failed REML certification, and only tied the linear model [in-workspace: measured in-workspace; value not reproduced here].
