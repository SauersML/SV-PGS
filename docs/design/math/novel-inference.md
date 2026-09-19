# EDB-EP: environment-decoupled block EP for the Stage 1 and Stage 2 fits, from first principles

Lane novel-inference, 2026-09-19.
- It derives how the one model's EP-EB fit should be computed at production scale (n ≈ 5·10⁴–10⁵, p ≈ 1.7·10⁷, 105 models), and measures the derivation against exact EP.
- Landed code: `sv_pgs/marginal_variances.py` (Stage 2's marginal variances, their derivative and their certificate).
- The prototype, raw results and the check scripts are in `/projects/standard/hsiehph/sauer354/svpgs-team/novel-inference/` (proto/, and the JSON-lines outputs under /scratch.global/sauer354/svpgs-team/novel-inference/).
- Related: ep_eb.md (the fixed point and its certificate), compute_floor.md (the pass and memory floors), b_products.md (the hyperparameter curvature that consumes `variance_jvp`).

**Evidence class.** Machinery only (EVIDENCE_RULE.md): distances to the exact EP fixed point computed by two independent dense references, pass counts, and identities checked against dense linear algebra. The method changes how the model is computed, not the model, so it makes no accuracy claim.

**Tags.**
- [proved]: an exact identity with a proof here.
- [checked]: verified numerically by a script in the prototype directory.
- [bound]: a proved inequality under stated assumptions.
- [measured]: see "Measured" at the end.
- [conjecture]: not proved.

## 0. The problem and its cost model

- **Model.** After FWL: y = Xβ + ε with ε ~ N(0, σ²I). X is n × p, with columns standardized. The prior is β_j ~ ∫ N(0, u_j s) g_c(s) ds, with g continuous (SPEC 131b205).
- **Sites.** Expectation propagation replaces each prior by a Gaussian site exp(−½Π_jβ_j² + h_jβ_j). Write D = Π⁻¹ and m = Dh.
- **Scale:** n ≈ 5·10⁴–10⁵, p ≈ 1.7·10⁷, and M ≈ 10² models (21 traits × 5 folds), each with its probes and 64 draws.
- **Cost currency.** The unit is a **pass**: one sweep of the int8 store, about 0.45 TB compressed at n = 5·10⁴. It is I/O-bound at roughly 100–250 s.
  - Flops per pass are 2·n·p·(columns). At 10⁴ columns that is 1.7·10¹⁶, about 20–140 s on an H100 or V100 tensor core, so compute and I/O are comparable.
  - Every design is scored in passes, plus the **state** it must keep across passes.

## 1. Lower bounds

**1a. Passes [bound].** Let an algorithm see X only through block products, i.e. passes that return XV and XᵀU for chosen V and U. At fixed sites, its iterate after k passes lies in a (block) Krylov space of the operator.
- For SPD operators whose spectrum fills [λ_min, λ_max], the Chebyshev/Nemirovski–Yudin bound applies: no k-step Krylov method beats a worst-case energy-norm reduction of 2((√κ−1)/(√κ+1))^k.
- The bound is tight for continuous spectra such as Marchenko–Pastur. So

  passes ≥ ½·√κ·ln(2/ε),

  where κ is the condition number of the best preconditioned operator available from "free" information: Stage 0 statistics plus O(1) passes.
- The design problem is therefore: **pick the space and the preconditioner that minimize κ.** §5 shows that the right choice makes κ independent of p.

**1b. The accuracy we need [proved].**
- For the posterior mean under squared loss,

  E[(xᵀβ − xᵀμ̂)² | y] = E[(xᵀ(β−μ))² | y] + (xᵀ(μ−μ̂))²,

  where μ is the exact mean. So computational error δ = μ̂ − μ adds ‖X_test δ‖² to the held-out loss.
- With ε = ‖X_test δ‖/‖X_test μ‖ in centred test norm, the unit-normalized predictor moves by at most 2ε. So |Δcorr| ≤ 2ε, and |ΔR²| ≤ 4ε|r| + 4ε², where r = corr(X_test μ, g).
- The evaluation's resolution is the SE of the paired ΔR² (EVALUATION.md), about 10⁻³ at n_test ≈ 10⁴. So ε_target = SE(ΔR²)/(4|r|) is derived, not chosen, and is typically 10⁻³–10⁻⁴.
- Solving further wastes passes.

## 2. Stage 1: the environment

**Theorem 1 (the block-marginal identity) [proved].** For any partition of the variants into blocks,

  p(β_b | y) ∝ p(β_b) · E_{β_{−b} ~ prior}[ N(y; X_bβ_b + X_{−b}β_{−b}, σ²I) ].

*Proof.* Bayes' rule, integrating β_{−b} out of the joint; the prior factorizes across blocks. ∎

So the posterior of block b is exactly its own prior times a likelihood in which every other block's effects are integrated out **under their prior**.

**Lemma 1 (the environment is Gaussian) [proved; the CLT condition is explicit].** Given X_{−b}, the vector g_{−b} = X_{−b}β_{−b} is a sum of independent vectors x_kβ_k, with covariance K_{−b} = X_{−b} M_{−b} X_{−b}ᵀ, where M = diag(E[β_k²]).
- By Lindeberg's theorem it is Gaussian when no single term dominates: max_k n·E[β_k²] ≪ tr(K_{−b})/n · n.
- A variant with n·E[β_k²] comparable to the bulk level violates this. It is **resolved** and must be modelled explicitly (§5, the set L).

**Theorem 2 (the deterministic equivalent of the environment) [bound; checked in `test_deterministic_equivalent_of_the_block_data_precision`].**
- Assume the sample rows of (X_b, X_{−b}) are i.i.d., with no population LD between blocks.
- For any PSD Ψ independent of X_b,

  X_bᵀ Ψ X_b = (tr Ψ / n) · R_b · (1 + O_p(‖Ψ‖_F / tr Ψ)),  with R_b = X_bᵀX_b.

  The relative error is O_p(n^{−1/2}).
- Apply this with Ψ = σ²I + K_{−b} to the Stage 0 statistic:

  X_bᵀy | β_b ~ N(R_bβ_b, (σ² + c_{−b}) R_b),  c_{−b} = tr(K_{−b})/n = Σ_{k∉b} E[β_k²].

- **So the LD-space likelihood of block b has noise σ² + c_{−b}, not σ².**
- Note that σ² + c_{−b} = Var(y) − c_b: the phenotypic variance minus the block's own prior genetic variance. c_{−b} = Σ_{k∉b} u_k ∫ s g(s) ds is a function of the hyperparameters, so it's learned along with them, never chosen.

**Corollary (why the block-diagonal Stage 1 failed) [measured].**
- The current plan's Stage 1 uses R_b/σ². It overweights each block's data by (σ² + c_{−b})/σ² ≈ 1/(1−h²).
- That is exactly the double counting: each block explains the other blocks' genetic signal as if it were its own.
- The reported "2.7× off at p/n = 20" matches 1/(1−h²) at h² ≈ 0.63.
- The prediction concerns machinery, not accuracy: the environment Stage 1 lands closer to the exact full-data EP fixed point, measured as the held-out-predictor distance ‖X_test(μ̂ − μ_EP)‖/‖X_test μ_EP‖. "Measured" below gives that distance.
- **Evidence rule:** R² and calibration against the simulation truth, from this lane's own simulator, are labelled `sim-only`. They are never adoption evidence. The adoption claim of this lane is purely computational: the same fixed point in fewer passes, with less state and no stochastic site variances. That is checkable against the dense exact EP reference without any accuracy claim.

**Theorem 3 (what LD space cannot know) [bound; checked in `check_mp.py`].**
- The full-data block conditional (§3) uses X_bᵀQ_{−b}(·), with Q_{−b} = (σ²I + K_{−b})⁻¹, and not X_bᵀ(·)/(σ²+c).
- Its effective noise is σ²_eff = n / tr(Q_{−b}). If K_{−b}/c follows Marchenko–Pastur with ratio γ = n/M_e, where M_e is the effective number of weighted independent markers, then

  tr(Q)/n = m(−σ²/c) / c,
  m(−t) = [√((1−γ+t)² + 4γt) − (1−γ+t)] / (2γt).

- By Jensen, σ²_eff ≤ σ² + c. The LD-space estimator's information ratio is

  ρ = σ²_eff/(σ²+c) = 1 / ((1+t)·m(−t)),  t = σ²/c.

  For example, ρ = 0.84 at γ = 0.8 and h² = 0.5.
- **Stage 1 with the environment is the Bayes estimator given block summaries. Stage 2 recovers the remaining factor 1/ρ of information, and nothing more is available.**
- γ is estimated from the kernel's second spectral moment: γ̂ = tr((K/c)²)/n − 1, which is computable from probes.

## 3. Stage 2: Schur decoupling

**Theorem 4 (the exact block conditional) [proved; checked in `test_schur_block_conditional_reproduces_the_exact_mean_and_variance`].** Under Gaussian sites, with Q_{−b} = (σ²I + X_{−b}D_{−b}X_{−b}ᵀ)⁻¹ and α = (σ²I + XDXᵀ)⁻¹(y − Xm):

  Σ_bb = (Π_b + Λ_b)⁻¹,  μ_b = Σ_bb (h_b + ℓ_b),
  Λ_b = X_bᵀQ_{−b}X_b,  ℓ_b = X_bᵀQ_{−b}(y − X_{−b}m_{−b}) = X_bᵀα + Λ_b μ_b.

*Proof.*
- The first two equations are the Schur complement of the joint precision after marginalizing β_{−b}, with the Gaussian sites playing the role of their prior; this is Theorem 1 with Gaussian sites.
- For the last identity: μ = m + DXᵀα (Woodbury), so ℓ_b = Σ_bb⁻¹μ_b − h_b = Π_b(m_b + D_bX_bᵀα) − h_b + Λ_bμ_b = X_bᵀα + Λ_bμ_b. ∎

ℓ_b depends only on the other blocks' sites. So **given the single n-vector α, every block's exact conditional data term is known.**

**Theorem 5 (the variances need one scalar per block) [bound; checked; superseded in use by §5 and §5b].** Its ω_{−b} formula applies the deterministic equivalent to the full, spiked Q. v1 used it and was replaced: the measured v3 applies the equivalent only to the far field of the bulk kernel, with resolved sites and neighbour blocks exact.
- By Theorem 2, Λ_b = ω_{−b} R_b (1 + O_p(n^{−1/2})) with ω_{−b} = tr(Q_{−b})/n.
- By Woodbury, Q_{−b} = Q + QX_b(D_b⁻¹ − X_bᵀQX_b)⁻¹X_bᵀQ. Applying Theorem 2 inside it gives

  ω_{−b} = ω + (ω₂/n)·tr((D_b⁻¹ − ωR_b)⁻¹R_b),  ω = tr(Q)/n,  ω₂ = tr(Q²)/n.

- ω and ω₂ are two scalars per model. Hutchinson estimates them with relative SD √(2/(n·probes)), about 0.2% at n = 5·10⁴ with 8 probes.
- **The per-variant marginal variances are therefore deterministic, and Monte Carlo noise never enters a site update.** This settles lit-ep's objection C1 without their fallback of "fixed block-inverse variances", which omits the environment factor ω_{−b}.

**The algorithm (EDB-EP v3, as measured), per model.**
1. **Stage 1 (no pass).** Block EP against (R_b/(σ²+c_{−b}), X_bᵀy/(σ²+c_{−b})), with c_{−b} = Σ_{k∉b} E[β_k²] from the current prior. This is Theorem 2's environment likelihood.
2. **Refresh (passes).**
   - Split solve (§5): Z = K_S⁻¹[y − X_S m_S, X_L, probes], by CG on the bulk kernel.
     - The mean and X_L columns go to relative residual min(10⁻³, 0.1 × the last outer change): the forcing term, §8c.
     - The probes go to 10⁻³, fixed and warm-started.
   - core = Π_L + X_LᵀZ_L; μ_L = core⁻¹(h_L + X_LᵀZ_y); α = Z_y − Z_Lμ_L.
   - One final pass forms Xᵀ[α, Z_L]. That gives the bulk means μ_S = m_S + D_S X_Sᵀα and the cross products C = XᵀZ_L.
   - ω_S and ω_S2 come from the probe columns.
3. **Local step (no pass).** For every block:
   - Λ_b = X_bᵀK_{−b}⁻¹X_b by the direct leave-out form (§5b): the window b±1 exact, the far field by ω_F, and the resolved sites outside b through C and core.
   - ℓ_b = X_bᵀα + Λ_bμ_b (Theorem 4).
   - Then run block EP to convergence against (Λ_b, ℓ_b).
4. **Relaxed site update** (§8d), then repeat from step 2 until the held-out-predictor change falls below ε_target (§1b).

The hyperparameter step belongs inside step 3 (§6). The measured runs hold hyperparameters at their true values, which isolates the machinery.

**Fixed points [proved].**
- At every refresh the Gaussian means are exact, because α is exact.
- The block-local EP solves exactly the block conditional of Theorem 4, with Λ_b known up to O(n^{−1/2}).
- So a fixed point of EDB-EP is the EP fixed point of the full model, with marginal variances perturbed by O(n^{−1/2}) relative. Its distance from the exact EP fixed point is under "Measured".

**Convergence [conjecture, measured].**
- The local EP is small and dense; in production it runs sequential and damped per block, so it has no limit cycles.
- The refresh loop is a nonlinear block-Jacobi on the EP equations. Its coupling is the chance correlation between blocks, whose operator norm relative to the in-block scale is ‖M⁻¹(K − M)‖ ≤ h_S(2√γ + γ) (§5).
- A contraction factor of order (EP site sensitivity) × h_S(2√γ+γ) is predicted. The measured refresh counts are under "Measured".

**Why this is the right split [design argument].**
- Every slow nonlinear iteration happens inside the pass-free local step: EP site sweeps, the M-step for g (EM-rate), the smoothing weights, and σ².
- Passes only transport a weakly varying linear correction (α, ω, ω₂).
- The current plan does the reverse: every EP iteration costs a full PCG, plus stochastic variances.

## 5. The solve: resolved variants in primal, the bulk in dual

**Lemma 5a (what a dual residual controls) [proved].** Let α̂ solve Kα = r₀ with residual r = r₀ − Kα̂, and set μ̂ = m + DXᵀα̂. Then the primal residual is

  Aμ̂ − b = −Xᵀr/σ²,

and the error is μ̂ − μ = −DXᵀK⁻¹r = −A⁻¹Xᵀr/σ².

*Proof.* Π(μ̂ − m) = Xᵀα̂ gives Πμ̂ − h = Xᵀα̂. So Aμ̂ − b = Xᵀ(Xμ̂ − y + σ²α̂)/σ², and Xμ̂ − y + σ²α̂ = Kα̂ − (y − Xm) = −r. The error formula is the push-through identity. ∎

- **Consequence.** A small relative dual residual does not give a small prediction error when D has large entries, i.e. strong effects. The error direction DXᵀK⁻¹r is amplified by the large D_k.
- Measured [checked, E2 `linear`]: at p/n = 2 with 23 strong sites, a dual relative residual of 10⁻⁶ left a held-out-predictor error of 3·10⁻². The primal system at the same residual left 10⁻⁶.
- An environment preconditioner M = aI + X_L D_L X_Lᵀ doesn't cure this; it changes the Krylov space, not the metric.

**Theorem 6 (exact elimination) [proved; checked in `test_split_solve_is_the_exact_mean_and_kernel_inverse`].**
- **The split.** The resolved set is L = {k : n·D_k ≥ a} ∪ {k : Π_k ≤ 0}. Non-positive sites would make K indefinite, by Sylvester's inertia (speed-krylov). The first part, with a = σ² + Σ_{k∉L} D_k, is the variants whose rank-one spike rises above the bulk level. This is a fixed point of a derived rule, not a tuning constant.
- **Integrate the bulk S out in dual:** y | β_L ~ N(X_Lβ_L + X_S m_S, K_S), with K_S = σ²I + X_S D_S X_Sᵀ.
- **Solve** Z = K_S⁻¹[y − X_S m_S, X_L, probes], with 1 + |L| + probes columns in the same passes. Then:

  core = Π_L + X_LᵀZ_L,  μ_L = core⁻¹(h_L + X_LᵀZ_y),
  α = b̃ − Z_L core⁻¹ X_Lᵀ b̃,  b̃ = Z_y − Z_L m_L,  Qz = Z_z − Z_L core⁻¹ X_LᵀZ_z,
  μ_S = m_S + D_S X_Sᵀα.

- **Why it is conditioned.** Errors in the bulk solve reach the mean only through D_S, and every bulk D_k < a/n. The strong effects are solved exactly in an |L| × |L| system.
- **Theorem 6' (conditioning of the bulk) [bound; corrected after `check_mp.py`].**
  - K_S = σ²I + c_S·(K_S − σ²I)/c_S, and the normalized bulk kernel N = X_S D_S X_Sᵀ/c_S has an exact zero eigenvalue for every covariate FWL projects out, the intercept included. So λ_min(N) = 0, exactly, and

    κ(K_S) = (σ² + c_S λ_max(N)) / σ² = (1 + h_S(λ_max(N) − 1)) / (1 − h_S),  h_S = c_S/(σ² + c_S).

  - A first version used the Marchenko–Pastur lower edge (1−√γ)² and was **wrong**: measured κ exceeded it (3.91 vs 3.61). With λ_min = 0 the formula matches the unlinked cases (3.91 vs 3.92; 2.82 vs 2.84).
  - λ_max(N) is p-free in the sense that matters: it is set by n/M_e (M_e = effective independent markers) and by LD, not by p. Under MP with ratio γ, λ_max = (1+√γ)². **With LD, MP with the second-moment γ̂ under-predicts λ_max** (measured κ 4.40 vs 3.83 at ρ = 0.9, and 8.08 vs 5.23 at h² = 0.7). So λ_max is not taken from a formula. The first CG iterations' Ritz values give it, and the pass count ½√κ·ln(2/ε) is then certified a posteriori.
  - **Measured comparison, stated honestly:** the primal block-Jacobi κ was 4.27, 2.97 and 5.08 where the dual's was 3.91, 2.82 and 4.40. For a single mean solve the two iteration counts are comparable. EDB-EP's gains are elsewhere:
    - n-dimensional state;
    - exact marginals without per-variant Monte Carlo;
    - all nonlinear work in the pass-free local step.
- **Lemma 5b (the leave-block-out map is exact) [proved; checked in `test_leave_block_out_map_is_exact`].** With G_b = X_bᵀQX_b and Q = K⁻¹, the Woodbury add-back of block b gives

  Λ_b = X_bᵀQ_{−b}X_b = G_b + G_b(Π_b − G_b)⁻¹G_b = G_b(Π_b − G_b)⁻¹Π_b.

- **The block data precision in EDB-EP v2:**

  G_b ≈ ω_S R_b − (X_bᵀZ_L) core⁻¹ (Z_LᵀX_b),  ω_S = tr(K_S⁻¹)/n,

  - The deterministic equivalent is used only on the bulk K_S⁻¹, where Theorem 2 holds: K_S has no spikes.
  - The resolved spikes enter exactly, through the cross products X_bᵀZ_L. Those come from the refresh's extra pass Xᵀ[α, Z_L].
  - This replaces the ω_{−b} formula of Theorem 5, which applied the equivalent to the full, spiked Q.

## 5b. The local data precision, formed without inverting Π_b − G_b

- **The first version failed.** It formed Λ_b = G_b(Π_b − G_b)⁻¹Π_b from an approximate G_b.
  - The map is exact given the exact G_b. But for a strong or resolved site j inside block b, the exact G_jj saturates at Π_j. An approximate G_b can cross that ceiling, which makes Π_b − G_b indefinite.
  - Measured [E2, p/n = 2, 23 resolved sites]: the fixed-point error was 0.47 or NaN.
- **The direct form:**

  Λ_b = X_bᵀK_{−b}⁻¹X_b = X_bᵀK_{S,−b}⁻¹X_b − C_{b,L∖b} core_{L∖b}⁻¹ C_{b,L∖b}ᵀ,
  X_bᵀK_{S,−b}⁻¹X_b ≈ ω_F R_b − ω_F² R_{bN} D_N^{½} (I + ω_F B_N)⁻¹ D_N^{½} R_{Nb}.

  - N is the neighbouring blocks, handled exactly.
  - ω_F is the far-field trace, the unique root from marginal_variances.py over the window b ∪ N.
  - The spike term keeps only the resolved sites outside block b; block b's own sites are what its local EP updates.
- **Why it's well-posed.** Nothing is inverted except I + ω_F B_N (≻ I) and core_{L∖b} (≻ 0 whenever q is proper).
- **Approximations made:** the deterministic equivalent on the far field, and K_S in place of K_{S,−b} in the spike cross terms. The latter differs by block b's own bulk share, of order 1/(number of blocks).
- **The landed module doesn't have this problem.** `sv_pgs/marginal_variances.py` forms marginals as D − D²q + D²·spikes, plus core⁻¹ for the resolved sites. It never inverts Π_b − G_b.

## 6. Hyperparameters inside EDB-EP

- **Noise [proved].** y − Xμ = σ²α and XΣXᵀ = σ²(I − σ²Q). Both are push-through identities. So the EB update

  σ²_new = (‖y − Xμ‖² + tr XΣXᵀ)/n = σ⁴‖α‖²/n + σ²(1 − σ²ω)

  is **exact** given α and ω: no pass, no per-variant quantity.
- **Mixing density, scale model and smoothing weights.** At fixed cavities these are the math-density and lit-eb problems: concave in the quadrature weights of the continuous g, the scale model by Newton with the location pin, and λ by the Tipping–Faul one-dimensional rule.
  - In EDB-EP the cavities are block-local and exact given (α, ω), so the M-step runs inside the local step at no pass cost, to full convergence.
  - When the hyperparameters change the prior globally, the local step also updates c_{−b} (Stage 1 form) and keeps the Stage 2 correction (Λ_b − R_b/(σ²+c_{−b}), ℓ_b − X_bᵀy/(σ²+c_{−b})) frozen until the next refresh. That transports the between-block correction, not the whole data term [design; to be measured].

## 7. Traits, folds, draws, memory

- **All models share passes.** A fold is a sample mask in the same n-space: K_f a = σ²a + mask_f ∘ X D X ᵀ (mask_f ∘ a).
- **Draws by Matheron's rule [proved]:** β* = μ + D^{1/2}z₁ − DXᵀK⁻¹(X D^{1/2}z₁ + σz₂) is an exact draw of q(β). It needs one dual solve per draw, and its state is n-dimensional.
- **Memory**, at n = 5·10⁴ and M = 105 models × (1 mean + 8 probes + 64 draws) = 7,665 columns:
  - dual CG state: about 4 vectors × n × columns × 4 B ≈ 6 GB. It fits a 16 GB GPU next to an int8 tile (n × 4096 = 200 MB) and its U_b (4096 × 7665 × 4 B = 126 MB).
  - the primal equivalent: 4 × p × columns × 4 B ≈ 2 TB, streamed per pass.

## 8. Precision and the GPU mapping

- **Inputs are exact.** Stored codes are integers 0–255, exact in bf16 (up to 256) and fp16 (up to 2048).
  - Standardization is a rank-one correction: X_stdᵀa = S⁻¹(X_codeᵀa − μ(1ᵀa)) and X_std b = X_code(S⁻¹b) − 1(μᵀS⁻¹b), applied in fp64 to p- or n-vectors.
  - So the GEMMs run on raw codes with fp32 accumulation.
- **Accumulation.** Split the inner dimension into chunks of about 1,024, accumulate each chunk in fp32 and sum the chunks in fp64. The per-product relative error is then ≲ √1024·u₃₂ ≈ 2·10⁻⁶.
- **CG in fp32 attains** a relative residual of about κ·(that error) ≈ 10⁻⁵ at κ ≤ 5, below ε_target.
- **A certificate pass** with fp64 chunk sums re-evaluates the residual exactly.
- **fp32 is safe here because κ is p-free and small.** The primal block-Jacobi system is what needed the fp64 Cholesky policy.

## 8b. The local step's cost, and a truncation rule derived from the approximation's own error

- **The cost.** The local step factors (Π_b + ω_{−b}R_b) for every block, model and local EP iteration. With dense blocks of size m that is p·m² per model per iteration. At m ≈ 2,000 that exceeds a pass.
- **R_b is phenotype-independent,** so Stage 0 can store its eigendecomposition R_b = V_b S_b V_bᵀ once.

**Truncation bound [bound; checked in `check_lowrank.py`].**
- Keep the eigenpairs with ω λ_k > δ·min_j Π_j, and drop the rest; call the dropped part E.
- Then 0 ⪯ ωE ⪯ δ Π_b, so

  (1+δ)⁻¹ (Π_b + ωR_b) ⪯ Π_b + ωR_b^trunc ⪯ Π_b + ωR_b,

- and every marginal variance changes by a relative amount of at most δ. Woodbury on the kept rank r_b costs m·r_b² per factorization and m·r_b per diagonal.
- **The derived tolerance is δ = n^{−1/2}.** That is the relative error the deterministic equivalent Λ_b ≈ ω_{−b}R_b already carries (Theorem 2), so a smaller δ buys nothing.
- The kept rank is set by the LD spectrum and n, not by hand. `check_lowrank.py` measures r_b on AR(1) blocks, and the bound held in every case:
  - m = 500, n = 5,000: r_b = 48–66, max variance error 0.3–0.4%;
  - m = 2,000: r_b = 733 at n = 5,000, but 1,857 at n = 20,000.
- **The compression fades as n grows,** because R_b's eigenvalues scale with n while min Π does not. At production n it saves little for large dense blocks. There the cost floor is speed-floor's p_b·|L_b|² route, not truncation.

## 8c. Inexact refreshes (a forcing term)

- **The refresh needs only limited accuracy.** It feeds the local step through ℓ_b = X_bᵀα + Λ_bμ_b. An error δα moves ℓ_b by X_bᵀδα.
- **The rule.** Solving the mean column to a relative residual 0.1 × (the last outer change) keeps the refresh error an order of magnitude below the change it drives. This is an Eisenstat–Walker forcing term, so the outer iteration converges at the same rate while spending passes in proportion to progress.
- **Probes** enter only through ω and ω₂. They are held fixed and warm-started across refreshes, and need a relative residual of 1e-3, well below their own Monte Carlo SD (§3).

## 8d. Outer relaxation

- The refresh is a block-Jacobi step: every block moves at once against the same residual.
- When the between-block coupling is strong (small p/n, large per-variant signal), the blocks overshoot together, the same mechanism that makes parallel EP cycle.
- A relaxed step, sites ← sites + η(local fixed point − sites), leaves the fixed point unchanged. η is halved when the held-out-predictor change grows and restored otherwise.
- The coupling bound h_S(2√γ+γ) of §5 predicts η = 1 for p/n ≫ 1, the production regime.

## 9. Certificate

1. The dual residual ‖r − Kα‖/‖r‖ from the fp64 certificate pass. This bounds the Gaussian-mean error in the energy norm, hence the prediction norm (§1b).
2. The block-local EP moment residuals, which are exact and local, plus the number of clipped sites.
3. **Variance validity.** Compare the deterministic variances with one control-variate Hutchinson estimate (K probes, one solve set). Agreement within Monte Carlo error certifies Theorem 5's O(n^{−1/2}).
4. The hyper certificate as in lit-ep and math-density: Newton decrement and KKT sign at boundaries.

## 10. Assumptions and failure modes

- **No population LD between blocks** is assumed in Theorem 2. Residual long-range LD enters as a structured part of K_{−b}; the exact dual solve still handles it, and only the variance DE degrades. Test: the §9 variance-validity check.
- **Relatedness.** With sample covariance Φ the DE becomes tr(QΦ)/n, still one scalar. AoU folds are kinship-grouped.
- **Large effects** break the CLT in Stage 1. They are the resolved set L; in Stage 1 they need the cross-Gram X_bᵀX_L (p × |L|, one pass) to be exact. Stage 2 is exact regardless.
- **Binary traits.** The working Gaussian (IRLS weights W) turns σ²I into W⁻¹ and ω into tr(W^{1/2}QW^{1/2})/n: still a scalar per model.

## 11. Relation to prior work, and what is new

**Known, and used as tools:**
- the dual/kernel form (BLUP; BOLT-LMM's CG on the infinitesimal kernel);
- Woodbury and Schur identities;
- Marchenko–Pastur, and deterministic equivalents;
- total-variance noise in summary-statistic likelihoods (RSS);
- Matheron's rule;
- Eisenstat–Walker forcing.

**New here, as far as the literature lanes report:**
1. **The exact block-marginal identity as an LD-space likelihood** with noise σ² + c_{−b}, learned with the hyperparameters. The information ratio ρ(γ, t) says exactly what block summaries lose.
2. **Resolved/bulk elimination** (Theorem 6 and marginal_variances identity 1).
   - Strong and improper sites are exact in an |L| × |L| core, and the bulk sits in a dual kernel whose mean errors are damped by D_S < a/n (Lemma 5a).
   - Lemma 5a itself: a dual residual doesn't certify held-out error when strong sites are present.
3. **Leave-block-out data precision without inverting Π_b − G_b** (§5b). The neighbour window is exact and only the far field uses the equivalent. That makes it valid with LD across cuts.
4. **Marginal variances and their derivative from sample-side state only.** The far field of the derivative goes through the block sandwich diag(Σ_bb R_b Σ_bb), which accounts for LD partners absorbing chance coupling. Each block carries a probe certificate at family-wise level 1/B. This is `sv_pgs/marginal_variances.py`.
5. **The split:** every slow nonlinear iteration runs in a pass-free local step, and passes only refresh (α, C, ω).
6. **The exact σ² update** from (α, ω).

**What the lit-ep lane reports for prior work:**
- the recommended "decoupled scheme" freezes block-inverse variances, which omit the other blocks' signal (measured here: a 1–3% fixed-point bias);
- gVAMP needs rotational invariance and damping.

**Measured failures of my own earlier versions**, recorded under "Measured":
- v1: scalar ω_{−b} variances and an unsplit dual;
- v2: an indefinite local map;
- the first Theorem 6 κ bound;
- per-variant far-field norms.

## Measured (machinery; n = 1,000 unless noted; AR(1) haplotypes, blocks of 100)

The exact fixed point comes from block-sequential exact EP (exact block conditionals) and parallel damped exact EP; they agree to 4e-8 to 6e-7 on polygenic rows. Distances are ‖X_test(μ̂ − μ_EP)‖/‖X_test μ_EP‖.

**Stage 2, polygenic, 3 reps.**
- The current plan, block-Jacobi PCG plus Hutchinson diag(Σ):
  - with 16 probes it stops at 1.7–3.3e-2;
  - with 64 probes it stops at 1.3e-2 (p/n = 2).
- The decoupled block-Jacobi driver stops at 1.6–3.3e-2.
- Exact variances converge, but need 66–85 passes to 1e-2 and 91–145 to 1e-3, and they cost O(p³).
- EDB-EP v3 with the neighbour window:

  | LD | p/n | passes to 1e-2 | passes to 1e-3 | final distance |
  |---|---|---|---|---|
  | blocks independent | 2 | 26 | — | 1.6–1.9e-3 |
  | across every cut (ρ = 0.97) | 2 | 38 | 52 | 0.9–1.5e-3 |
  | across every cut (ρ = 0.97) | 10 | 11 | 17–26 | 3.8–4.8e-4 |

  - That floor is the deterministic equivalent's O(n^{−½}) error; its n-scaling is pending.
- Without the window, under LD across cuts, the floor rises from 4.8e-4 to 1.8e-3.

**Stage 1.** The distance to the exact fixed point is 0.31–0.34, 0.24 and 0.18–0.20 for the environment likelihood at p/n = 2, 10 and 20. The block-diagonal fit is at 0.51–0.60, 0.78–0.89 and 0.93–1.04.

**Marginal variances,** against the dense inverse with LD across cuts (n = 1,000–3,000, p = 600–3,000):
- the window form: max per-variant relative error 0.5–3.5%, always below ‖K_S⁻¹‖_F/tr K_S⁻¹;
- the own block alone: 9–16%;
- block-Jacobi: 16–58%.
- The per-block trace certificate flagged no block for the window form, and every block for block-Jacobi.

**variance_jvp,** against the dense derivative:
- resolved rows: 1e-13;
- bulk entries: mean 0.2–0.6%;
- block sums: within 0.7%.
- The block sandwich was needed: per-variant norms overstated the far field 2.6–11×.

**Negative results, kept:**
- v1 (scalar leave-out variances, unsplit dual): 0.4–1.2e-2 at p/n = 2.
- v2 (Λ_b = G(Π − G)⁻¹Π from an approximate G): indefinite, 0.26–0.47 or NaN.
- The first κ bound: FWL forces λ_min = 0, and LD raises λ_max above the Marchenko–Pastur edge.
- For a single mean solve, the dual and the primal block-Jacobi κ are comparable (3.9 vs 4.3; 4.4 vs 5.1). EDB-EP's gains are the state, the exact marginals and the pass-free local step.
