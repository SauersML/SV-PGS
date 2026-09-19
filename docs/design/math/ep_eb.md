# EP-EB: the objective, its fixed point, convergence and the certificate

This note derives the theory of the EP-EB fit in MODEL.md §3–5 and checks every claim numerically.
- **What was checked against:** the dense reference in tag `archive/2026-09-19/build-ep-oracle` (`tests/ep_eb_reference.py`), on small problems (p = 25–40, n = 300–500, grid K = 30–39).
- **Script:** `checks.py` (lane math-epeb), run on MSI. Each section ends with what was measured.
  - The final version is at `/projects/standard/hsiehph/sauer354/svpgs-team/math-epeb/checks6.py`; run `python checks6.py <check>` next to the archived `ep_eb_reference.py`.
  - It is not in the repo because it imports the archived reference.
- **Related:** the mixing-density geometry (range, penalty order, the density's own λ → ∞ limit, the location pin) is in `mixing_density.md`. This note covers what couples the density to EP.

## 0. Notation

**Data factor.** exp(−½βᵀΛβ + ℓᵀβ) times e^c.
- A quantitative trait has Λ = XᵀX/σ², ℓ = Xᵀy/σ², c = −(n/2) log 2πσ² − yᵀy/2σ².
- In LD space, Λ = κnR.

**Prior.**
- p_j(β) = Σ_k π_{c(j)k} N(β; 0, u_j s_k), with π_c = softmax(Bψ_c) and log u_j = o_j + d̃_jᵀθ.
- The hyperparameters with penalties are x = (ψ, θ), penalized by ½ Σ_i λ_i xᵀS_i x. S_λ = Σ_i λ_i S_i.
- **The class level is not a separate parameter.** It is g_c's location on the shared grid, which is identified because d̃ is centred within each class.
  - An explicit level_c would give F a flat direction (level_c + a with g_c shifted by −a), so it needs a location pin; see mixing_density.md.
- **The rest enter as follows:** the smoothing weights λ through V (§1.4), and σ² through c, Λ and ℓ.

**Sites.** t_j(β) = exp(−½τ_jβ² + ν_jβ). The global approximation is q ∝ exp(−½βᵀ(Λ+T)β + (ℓ+ν)ᵀβ), with Σ = (Λ+T)⁻¹ and μ = Σ(ℓ+ν).

**Per-variant quantities.**
- Marginal natural parameters: (m_j, n_j) = (1/Σ_jj, μ_j/Σ_jj). Cavity: (P_j, h_j) = (m_j − τ_j, n_j − ν_j).
- Cavity-tilted normalizer: Z̃_j(P, h; x) = ∫ p_j(β) e^{−½Pβ² + hβ} dβ = Σ_k π_k (1 + v_k P)^{−1/2} exp(½h² v_k/(1 + v_k P)), with v_k = u_j s_k.
- Tilted mean m̃_j = ∂_h log Z̃_j, tilted second moment ẽ_j = −2 ∂_P log Z̃_j, tilted variance Ṽ_j. Marginal mean μ_j and marginal second moment e_j = Σ_jj + μ_j².
- Moment mismatches: a_j = m̃_j − μ_j and δ_j = ẽ_j − e_j.

**Other symbols.**
- Φ(P, h) = log ∫ e^{−½βᵀPβ + hᵀβ} dβ, and φ is its scalar version.
- G(x; c) = Σ_j log Z̃_j(P_j, h_j; x) − ½ xᵀS_λx: the penalized hyper objective at fixed cavities c (the reference's `penalized_objective`).
- A = −∇²_xx G: the fixed-cavity curvature (the reference's `penalized_hessian`, negated).

## 1. The objective

### 1.1 The EP evidence
Replace each p_j by C_j t_j, with C_j chosen so that the site reproduces the tilted normalizer: ∫ C_j t_j · cavity_j = ∫ p_j · cavity_j, i.e. log C_j = log Z̃_j − φ(m_j, n_j). Integrating the data factor against Π_j C_j t_j gives

  **F(τ, ν; x, σ²) = c + Φ(Λ+T, ℓ+ν) + Σ_j [log Z̃_j(P_j, h_j; x) − φ(m_j, n_j)].**

Everything the fit does is stationarity of F, or of its penalized and Laplace-integrated forms:
- **For x:** F_pen = F − ½xᵀS_λx.
- **For λ:** the Laplace evidence V(λ) = F_pen(x̂_λ) + ½ log|S_λ|₊ − ½ log|H_λ|, where H_λ is the curvature of F_pen at x̂_λ (§1.4).

### 1.2 Lemma 1 (the site differential)
At fixed (x, σ²),

  dF = Σ_j [ −a_j dν_j + ½ δ_j dτ_j + a_j dn_j − ½ δ_j dm_j ].

*Proof.* The three pieces are:
- dΦ = μᵀdν − ½ Σ_j e_j dτ_j;
- d log Z̃_j = m̃_j dh_j − ½ ẽ_j dP_j, with dh = dn − dν and dP = dm − dτ;
- dφ(m_j, n_j) = μ_j dn_j − ½ e_j dm_j.
Collecting terms gives the result. ∎

The same algebra holds when (m_j, n_j) move for any other reason at fixed sites (σ², Λ). The site terms then contribute Σ_j [a_j dn_j − ½ δ_j dm_j].

**Consequences.**
- **(i)** ∇_{τ,ν} F = 0 at every EP fixed point where each site matches both its mean and its variance.
- **(ii) Clipping.** The reference holds a site's precision at 0 when the tilted variance exceeds the cavity variance (Ṽ_j > 1/P_j); only the mean is matched there. Then a = 0 and δ_j > 0 on the clipped set C. So ∇_ν F = 0 but

  ∂F/∂τ_i = ½ δ_i − ½ Σ_{j∈C} δ_j Σ_ji²/Σ_jj²   (using ∂m_j/∂τ_i = Σ_ji²/Σ_jj²),

  which is non-zero in every direction that correlates with a clipped site.

*Measured* (AR(1) LD, r = 0.6 and 0.9; 3–4 clipped sites):
- |∂F/∂ν| ≤ 4e-8, at the finite-difference floor;
- |∂F/∂τ| up to 5.7e-5 and 3.8e-3, matching the formula to 2e-8 and 9e-9.

### 1.3 Theorem 1 (the joint fixed point is a stationary point)
Let x ↦ s(x) be an EP fixed point that is locally unique and differentiable (I − ∂T/∂s is non-singular). Let F_pen(x) = F(s(x); x) − ½xᵀS_λx. Then

  ∇F_pen(x) = ∇_x G(x; c(s(x))) + (∇_s F)ᵀ ds/dx.

Two cases follow:
- **No clipped site:** ∇_s F = 0 by Lemma 1(i), so ∇F_pen equals the fixed-cavity gradient. A point where the sites are moment-matched and x maximizes G at those cavities is then a stationary point of F_pen, and conversely.
- **Clipped sites:** the extra term (∇_τ F)ᵀ dτ/dx remains. The EP-EB fixed point is then a zero of the hyper residual r(x) = ∇_x G(x; c(x)). It is still deterministic and a valid comparison target, but it is not a stationary point of the EP evidence.

*Measured:*
- With no clipped site, dF_pen/dx (EP re-converged per coordinate) equals the fixed-cavity gradient to 6e-9 relative.
- With 1–4 clipped sites the relative error is 1.1e-4 to 4.8e-4. After adding the predicted adjoint term (∇_τF)ᵀdτ/dx it falls to 1e-10 to 1.2e-8.

**The noise variance.**
- Lemma 1 and ∂Φ/∂Λ = −½(Σ + μμᵀ), ∂Φ/∂ℓ = μ give, at fixed unclipped sites,

  ∂F/∂σ² = [‖y − Xμ‖² + tr(XᵀXΣ) − nσ²]/(2σ⁴).

- So **σ² = (‖y − Xμ‖² + tr(XᵀXΣ))/n is the exact stationarity condition of the EP evidence**, not an EM heuristic, whenever no site is clipped.
- With clipping, the fixed-site derivative gains −½ Σ_{j∈C} δ_j ∂m_j/∂σ², where ∂m_j/∂σ² = −(ΣXᵀXΣ)_jj/(σ⁴Σ_jj²).

*Measured.* The formula predicts the fixed-site derivative to seven digits (0.3543293 vs 0.3543293). At the EM fixed point:
- with 3 clipped sites, the stationary σ² lies 0.0035 away (0.31%);
- with 1 clipped site, 4e-5 away.

**What fails at a parallel-EP limit cycle.**
- A cycle is not a fixed point, so a ≠ 0 and δ ≠ 0 at every iterate, and ∇_s F ≠ 0 by Lemma 1.
- The hyper step and the certificate are then computed from a gradient that is not dF/dx. The error is (∇_sF)ᵀ ds/dx along the orbit.
- Averaging sites over the cycle doesn't help: the average is not a fixed point either.
- A fit taken from a cycling EP state can't be certified, whatever the per-site change looks like.

**Remedies for clipping, in order of preference.**
1. **Allow negative site precisions** whenever Λ + T stays positive definite: check by Cholesky, and take a per-site step that keeps the cavity proper. In an orthogonal design this is always safe, since Λ_jj + τ_j = 1/Var_exact > 0.
2. **Keep clipping but correct the hyper gradient by the adjoint.** Solve w = (I − ∂T/∂s)⁻ᵀ ∇_s F; the correction is wᵀ ∂T/∂x. That is one transposed linear-response solve per outer step, costing about a few EP sweeps.
3. **Record the bias.** Put the clipped count and ‖adjoint‖_{A⁻¹} in the certificate.

### 1.4 The penalty weights: which curvature
- **The right Laplace curvature.** The Laplace evidence for λ integrates exp(F(s(x); x)) against the Gaussian penalty prior, so its curvature is the total one, B = −∇²F(s(x); x). It is not the fixed-cavity A.
- **The two differ by the cavity response,** B = A − (∂²G/∂x∂c)(dc/dx), obtained by differentiating r(x).
- **What the reference does.** Its Fellner–Schall step λ_i ← λ_i [tr(S_λ⁺S_i) − tr(H⁻¹S_i)]/(x̂ᵀS_ix̂) uses H = A + S_λ. That is stationarity of a different Laplace approximation.
- **What exact stationarity also needs.** The derivative of log|H| through x̂ (Wood & Fasiolo 2017), which Fellner–Schall drops.

*Measured:*
- In a fitted problem, the eigenvalues of A⁻¹B lie in [0.82, 1.05]. EP-EM has no EM-style ordering B ⪯ A: the cavity response can add curvature as well as remove it.
- In the λ runaway case (§2.3), the creep ratio computed with B is 1.73–2.06, against 1.45–1.65 with A.
- **So the choice matters, but it does not cause the runaway.**
- For exactness, the reference and the engine should both use B. In the engine, a product Bv is one warm EP re-solve (a directional difference of the total gradient). Traces use Hutchinson, and the Newton step uses CG.

## 2. Convergence

### 2.1 The inner EP loop
- **The local rule.** Let T be the undamped parallel update and J_T its Jacobian at the fixed point. The damped update s ← s + ω(T(s) − s) converges locally iff |1 − ω(1 − λ)| < 1 for every eigenvalue λ of J_T, that is

  ω < ω_max = min_λ 2 Re(1 − λ)/|1 − λ|²   (it needs Re λ < 1).

- **The optimal damping** ω* minimizes max_λ |1 − ω(1 − λ)|.
- **How to get it without hand-setting.** Estimate the spectrum's extremes by a power iteration on the linearized map (a few extra sweeps), so no damping constant is set by hand. The reference's damping 0.3 is such a hand-set constant.

*Measured* (AR(1) r = 0.6; 10 columns at pairwise r = 0.99; 16 at r = 0.999; one strong effect in the block):
- The spectral radius of J_T is 0.07, 0.04 and 0.13.
- ω_max is predicted at 1.88, 1.92 and 1.97. At 1.1 ω_max the iteration diverged; at 0.9 ω_max it converged at the predicted rate (0.80 predicted; 0.81, 0.82 and 0.96 observed).
- ω* is 1.00–1.05, so undamped parallel EP was near-optimal in all three.
- **No design here produced a limit cycle.** The cycles reported at production scale must therefore come from much larger blocks, or from nonlinear, far-from-fixed-point dynamics such as clip switching. One far start with r = 0.95 took 2,914 sweeps despite a local spectral radius of 0.54. The local rule still sets the damping near convergence.

### 2.2 The outer (EP-EM) loop
- **The map.** M(x) maximizes G(·; c(x)).
- **Its Jacobian.** Differentiating ∇_x G(M(x); c(x)) = 0 gives

  **J_M = I − A⁻¹B′**, with B′ = −dr/dx the Jacobian of the hyper residual. B′ = B, symmetric, when no site is clipped.

- **Local convergence** holds iff every eigenvalue λ of A⁻¹B′ has |1 − λ| < 1. For a real spectrum that means λ ∈ (0, 2).
- **Relaxation.** x ← x + ω(M(x) − x) has eigenvalues 1 − ωλ, with the best ω = 2/(λ_min + λ_max).
- **Contrast with exact EM.** Exact EM has 0 ⪯ B ⪯ A (missing information), so λ ∈ (0, 1] and the rate is 1 − λ_min. MODEL's "rate ≥ 1 − edf/p ≈ 0.99" is the λ_min ≈ 0.01 regime. EP-EM has no such ordering (§1.4).
- **Anderson(m)** is equivalent to GMRES on the linearized fixed point (Walker & Ni 2011). It does not move the fixed point.
- **Newton on F_pen with B** converges quadratically. Its cost is products Bv, each one warm EP re-solve.
- **Double loop.** Double-loop EP (Heskes & Zoeter 2002; Opper & Winther 2005) guarantees the inner EP converges to a stationary point of the EP energy, at several times the sweeps of a sequential EP that converges. It gives no joint guarantee with hyperparameters, because the joint problem is a saddle (lit-ep).
- **The recommended structure:** sequential or damped-parallel EP to the §3.2 tolerance, then the hyper step with the §2.4 acceptance rule. The double loop is a fallback only when the power-iteration estimate says the sweep map is expansive.

*Measured (one clipped problem, dimension 39):*
- The eigenvalues of A⁻¹B′ lie in [0.82, 1.05], imaginary parts < 4e-12, and B′ is asymmetric by 1.1e-4 because of the 3 clipped sites.
- Plain iteration took 11 steps, Anderson 8.
- The plain steps contract by 0.157–0.187 per step (0.182 from t = 4 on). The predicted rate is max|1 − λ| = 0.182.

**Clipping is generic at EB fixed points.** A search over 30 seeds × 2 smoothing strengths (AR r = 0.8, p = 30, n = 300) found no fitted fixed point without a clipped site. So:
- The "unclipped" conditions of Theorem 1 and §2.4 essentially never hold for the scale-mixture prior as it is.
- The remedies of §1.3 are needed, not optional.

### 2.3 Penalty weights at a boundary: the creep, and the exact test
- **The creep.** Take penalty i with the others held fixed. Let N = null(S_i) and R its range. Let x̂_∞ be the optimum with x_R = 0, g̃ = ∇G there, Ã = −∇²G with λ_i = 0 (other penalties included), and S_o the other penalties. As λ_i → ∞:
  - x̂_R ≈ S_i⁺ g̃_R/λ_i;
  - tr(S_λ⁺S_i) − tr(H⁻¹S_i) ≈ [tr(S_i⁺Ã_{R|N}) − tr(S_i⁺(S_o)_{R|N})]/λ_i², with X_{R|N} the Schur complement on R after N is profiled out.
- **So Fellner–Schall multiplies λ_i by a constant each step:**

  λ_i^{new}/λ_i → ρ = [tr(S_i⁺Ã_{R|N}) − tr(S_i⁺(S_o)_{R|N})] / (g̃_Rᵀ S_i⁺ g̃_R),

  This ratio describes Fellner–Schall's own step, which drops the dependence of log|H| on x̂.
- **The exact test (corrected by the oracle lane).** Write τ = 1/λ_i. For a non-Gaussian likelihood the one-sided derivative at λ_i = ∞ has a third term that Fellner–Schall and the Gaussian Tipping–Faul form both lack:

  dV/dτ|₀ = ½(q − d + c),

  - q = GᵀE⁻¹G is the squared score in the released directions U;
  - d = tr(E⁻¹B_{U|K}) is its information, with E = UᵀS_iU;
  - c = −tr(H_KK⁻¹ Kᵀ D_xB[v] K) is how log|H_KK| moves at first order as x̂ moves with τ, where v = dx̂/dτ at τ = 0.

  λ_i = ∞ is a local optimum iff q − d + c ≤ 0. The full statement and derivation are in the oracle's STAGE1_GATES.md §2; it was checked against Richardson-extrapolated −λ²dV/dλ for four term types.
  - The q − d part is the variance-component score test (squared score against its information), and it equals ρ ≥ 1 above only when c = 0. That holds when B does not depend on x, i.e. for a Gaussian or locally quadratic model.
  - Otherwise c can flip the sign: the oracle measured c = −1.75 against q − d = +1.04.
  - When the test holds, set x_R = 0 exactly rather than creeping.
- **The evidence at the boundary.** The λ-dependent log-determinants cancel, giving

  V(∞) = J_∞(x̂_∞) + ½ log|(S_o)_{N_bN_b}| − ½ log|Ã_{NN}|,

  where N_b is the null space inside the penalized block. Compare it with interior stationary points.
- **Caution: the flat null-space integral can diverge.**
  - This V(∞) integrates the null directions that no other penalty covers under a flat prior (the −½ log|Ã_NN| term).
  - With D3 alone, the null space {1, t, t²} contains a ray along which log g concentrates on the lower grid end. There the likelihood tends to the null model's, a positive constant, so Ã_NN → 0 and V(∞) → +∞. The prior lane observed 3e302 on a weak 100-variant class. D2 alone has the same ray, from its tilt.
  - **Profiling the null space does not fix it.** The prior lane checked this numerically: exact normal means, D3, the null coordinates maximized and the log-det taken over the penalized subspace only. The profiled evidence then ran to λ → ∞ (the log-normal limit) on a BayesR truth, a 1k-variant class: ΔLPD −4.06 nats per 1k against −0.71 for integrated D3, and MSE +1.5% against −0.8%. The profiled null-space parameters, fitted freely, make switching off the penalized part look best.
  - **What works is a penalty with no uncovered null space.** First plus second differences on the sum-to-zero coordinates, as in the reference: S₁ is positive definite there, since the constant is removed.
    - At λ₂ = ∞ the tilt then keeps the proper Gaussian prior λ₁S₁, which is the ½ log|(S_o)_{N_bN_b}| term above, and V(∞) is finite. This is the case measured in the profile above.
    - At λ₁ = ∞ the limit is the flat density on the compact range, also proper.
    - Measured by the prior lane: D1+D2's learned λ is grid-invariant (log λ −1.04/3.08 → −1.01/3.15 across spacings 0.5 to 0.125), while D3's is not (3.33 / 1.35 / −18.4). D1+D2 also beat D3 on the snvtr TR class (ΔLPD −7.0 against −11.8).
  - Penalty values must be computed as |Rx|², never xᵀSx, which went negative at large null-space x in the prior lane's runs.
- **The same test applies at the other boundaries** (for hyperprior_pooling the per-trait model is quadratic in θ_t, so c = 0 and the q − d form is exact within that approximation):
  - the λ → 0 end, where FS's numerator is ≤ 0, which the likelihood's indefinite Hessian allows (lit-ep);
  - hyperprior_pooling's ω² → 0 (full pooling): a squared score ≤ information at ω² = 0 means the coordinate is shared exactly. That replaces the 1e-6 per-step shrink floor, and the FS ratio there shrinks ω² geometrically without ever reaching 0.

*Measured* (p = 40, K = 30; second-difference weight λ₂; AR(1) r = 0.6):
- **From λ₂ = e^8:** FS grows λ₂ by 1.40, 1.42, 1.62, 1.63, 1.62 per step. The formula gives ρ = 1.61 at the same iterates. The reference's absorbing rule then jumps λ₂ to its e^15 ceiling.
- **From λ₂ = 1:** the same problem sends λ₂ to the e^−15 floor, with a non-positive numerator.
- **The fixed point therefore depends on the start.** The evidence profile below decides which end is the optimum.
- The ρ prediction concerns Fellner–Schall's dynamics. The profile below evaluates V_A directly, with x̂ re-maximized at each λ₂, so it includes the c term and its conclusion stands.

*Measured: the fixed-cavity Laplace evidence V_A(λ₂)* (λ₁ = e^1.38; x̂ re-maximized at each λ₂; two cavity sets):

| log λ₂ | −15 | −5 | 0 | 2.5 | 5 | 7.5 | 10 | 15 | 20 | V(∞), closed form |
|---|---|---|---|---|---|---|---|---|---|---|
| cavities from λ₂ = e^−15 | 6.04668 | 6.04666 | **6.04613** | 6.05409 | 6.10746 | 6.19375 | 6.21699 | 6.21946 | 6.21948 | **6.21948** |
| cavities from λ₂ = e^10 | 6.21156 | 6.21155 | **6.21099** | 6.21887 | 6.27291 | 6.36021 | 6.38367 | 6.38617 | 6.38619 | **6.38619** |

What the profile shows:
- **V is bimodal in λ₂.** It has a plateau maximum at λ₂ → 0, a minimum near λ₂ = 1, and it rises to its global maximum at λ₂ = ∞, higher by 0.173 and 0.175 nats.
- **The reference's default start (λ = 1) sits at the minimum.** Fellner–Schall from there descends to the wrong boundary.
- **The closed-form V(∞) matches the profile's limit to 1e-7.** Beyond λ₂ ≈ e^21 the profile carries rounding noise of about 1e-4, which is the cancellation the formula avoids.

**The rule:** evaluate V at both boundaries, in closed form, and at any interior stationary point, and take the maximum. Here that is the power-law null space of S₂, with λ₁ penalizing its slope. It is not what either Fellner–Schall start gives without the absorbing jump.

### 2.4 "Accept only if the evidence rises": conditions for monotone ascent
The EM-type step has no minorization, so log Z_EP can fall. Take the step d = M(x_t) − x_t, backtrack along it, and accept the first point where F_pen (with EP re-converged) does not fall. Monotone ascent to a stationary point of F_pen holds under four conditions:
1. **EP is converged** at x_t to the §3.2 tolerance, so the fixed-cavity gradient equals ∇F_pen up to the §3.2 error.
2. **No site is clipped.** Otherwise ∇F_pen differs from ∇G by the adjoint term, and d need not be an ascent direction.
3. **The EP fixed point stays on one branch.** EP is warm-started along the segment, and its sweep map has spectral radius < 1. That keeps s(x) locally unique and C¹; a jump to another EP fixed point breaks continuity.
4. **d is gradient-related:** ∇Gᵀd ≥ c‖∇G‖‖d‖, which a Newton step with a positive-definite model Hessian provides.

Then:
- ∇F_penᵀd = ∇Gᵀd > 0 unless x_t is stationary, so the backtracking terminates.
- F_pen is bounded above: the mixture weights are bounded, and log Z̃_j → −∞ as u_j → ∞.
- By Zoutendijk's theorem the limit points are stationary.

If condition 2 fails, the rule converges to where no ascent along ∇G exists. That point is neither the EP-EB fixed point nor a stationary point of F_pen, so accept-if-rises must be paired with remedy 1 or 2 of §1.3.

*Measured* (AR r = 0.8, p = 25; two starts, and both fixed points had 2 clipped sites):
- The accepted values were monotone: +146 and +424 nats over 4–5 outer steps.
- The rule then **stalled**: no step size down to 1e-6 raised log Z_EP.
- At the stall, the fixed-cavity residual is 1.1e-3 and 2.6e-4. The adjoint-corrected total gradient is 6.4e-3 and 1.9e-3.
- So it stopped at neither the EP-EB fixed point nor a stationary point of the EP evidence, as condition 2 predicts.

*Measured, the contrast* (seed 317 with strong smoothing, the only unclipped path found in 30 seeds): accept-if-rises was monotone and converged in 4 outer steps, to a residual of 1.9e-7.

## 3. The certificate

### 3.1 The Newton decrement and the prediction change
- **The ratio bounds.** Let g be the total gradient: the fixed-cavity gradient at converged cavities (§1.3). Near x*, the true gap is F_pen(x*) − F_pen(x) ≈ ½gᵀB⁻¹g, while the reference's decrement is δ_A = ½gᵀA⁻¹g. Then

  δ_A/λ_max(A⁻¹B) ≤ gap ≤ δ_A/λ_min(A⁻¹B).

  Along the slowest mode, gap/δ_A → 1/λ_min. When the spectrum lies in (0, 1], as in EM, that equals 1/(1 − ρ), where ρ is the outer contraction.
- **What that means at production rates.** At ρ = 0.99 the fixed-cavity decrement understates the remaining gain by up to 100×. **The certificate must use δ_B = ½gᵀB⁻¹g, or δ_A/(1 − ρ̂)**, with ρ̂ the observed contraction.
- **The prediction change.** J_M = I − A⁻¹B is self-adjoint in the A inner product, which gives the exact bound

  ‖x_t − x*‖_A ≤ ‖x_{t+1} − x_t‖_A / λ_min(A⁻¹B).

  Likewise ‖X(μ_t − μ*)‖ ≈ ‖X(μ_{t+1} − μ_t)‖/(1 − ρ). A small step therefore certifies nothing unless it is divided by 1 − ρ̂.

*Measured* (the plain outer iterates of the clipped study in §2.2; ρ = 0.182 from the spectrum):
- The prediction bound ‖μ_t − μ*‖_Λ ≈ ‖μ_{t+1} − μ_t‖_Λ/(1 − ρ) holds to five digits: 2.0389e-7 against 2.0389e-7 at t = 9, and 1.0185e-3 against 1.0181e-3 at t = 4. The raw step understates the distance by the factor 1/(1 − ρ) = 1.22.
- **At a clipped fixed point the gap check fails, as it should.** The fixed point is not a maximizer of F_pen, and the measured "gap" F_pen(x*) − F_pen(x_t) turned negative from t = 2 on (−2.5e-3, −5.5e-4, …). That is the §1.3 non-stationarity made visible.

*Measured at the unclipped fixed point* (seed 317, strong smoothing, dimension 29):
- B is symmetric to 5.8e-14, as Theorem 1 predicts without clipping.
- The spectrum of A⁻¹B is [0.988, 1.024], so the bound interval for gap/δ_A is [0.977, 1.013].

| outer step | true gap | δ_A | δ_B |
|---|---|---|---|
| 1 | 1.1340e-3 | 1.1217e-3 | 1.1334e-3 |
| 2 | 1.60442e-7 | 1.58778e-7 | 1.60442e-7 |
| 3 | 2.81837e-11 | 2.80124e-11 | 2.81830e-11 |

- δ_B equals the true gap to 4–6 digits, while δ_A is 1% off, inside the predicted interval.
- In this well-conditioned case A ≈ B. At production rates, where λ_min(A⁻¹B) ≈ 0.01, the same bound allows δ_A to be off by 100×.

### 3.2 Gradient error at incomplete EP convergence
- **The error.** Let s be the sites after the last sweep and s* the fixed point at x. Then s − s* ≈ −(I − J)⁻¹(T(s) − s), and the fixed-cavity gradient is off by

  e_g = ∇_xG(x; c(s)) − ∇_xG(x; c(s*)) ≈ (∂²G/∂x∂c)(∂c/∂s)(s − s*),  so ‖e_g‖_{A⁻¹} ≤ ‖A^{−1/2}G_xc c_s‖ ‖(I − J)⁻¹‖ ‖T(s) − s‖.

- **An estimator that needs no Jacobians.** For a linearly convergent sweep with ratio ρ_s (successive gradient changes),

  g_t − g* ≈ (g_t − g_{t−1}) ρ_s/(1 − ρ_s).

- **How it sets the EP tolerance.** The certified decrement satisfies √(2δ*) ≤ √(2δ̃) + ‖e_g‖_{A⁻¹}. So EP has converged enough once the tail estimate is below the remaining budget √(2ε) − √(2δ̃), where ε is the certificate tolerance (§3.3). No separate EP tolerance is needed.

*Measured* (40 sequential sweeps from a cold start; the error against the converged fixed-cavity gradient in the A⁻¹ norm, or the Euclidean norm where A is indefinite away from the M-step optimum):

| case | sweep | true error | tail estimate |
|---|---|---|---|
| r = 0.6, smooth prior | 7 | 3.95e-3 | 3.97e-3 |
| | 23 | 5.15e-8 | 5.15e-8 |
| | 39 | 5.83e-13 | 6.08e-13 |
| r = 0.9, smooth prior | 7 | 5.78e-3 | 5.88e-3 |
| | 39 | 1.23e-12 | 1.27e-12 |
| r = 0.9, random prior, clipped | 7 | 2.97e-3 | 4.59e-3 |
| | 31 | 9.28e-8 | 9.49e-8 |

The estimate is within about 4% of the true error once the sweep ratio has settled (ρ_s ≈ 0.5–0.65 here). Early on it is within a factor of 2.

### 3.3 Tolerances derived from the scorer's K draws
- **The principle.** The scorer represents posterior uncertainty with K posterior draws (MODEL §4: K = 64). Its Monte Carlo error on any posterior functional f is sd_post(f)/√K. Numerical error below that is invisible downstream, so each tolerance is set to it.
- **Mean solve (Stage 2 CG).** Require ‖X(μ_k − μ*)‖² ≤ tr(XΣXᵀ)/K. Since ‖Xe‖² = σ² eᵀΛe ≤ σ²‖e‖²_{Λ+T} and tr(XΣXᵀ) = σ² tr(ΛΣ), it suffices that

  ‖e_k‖²_{Λ+T} ≤ p_eff/K,  with p_eff = tr(ΛΣ),

  the effective number of parameters that Stage 2's Hutchinson estimates already give. The A-norm error is estimated from the CG coefficients (Hestenes–Stiefel / Strakoš–Tichý).
- **Hyperparameters.** Require the error to be at most 1/√K posterior standard deviations: √(2δ_B) ≤ 1/√K, i.e. **δ_B ≤ 1/(2K)** nats. That is 0.0078 nats at K = 64.
- **EP sites:** from the §3.2 budget. **σ²:** the same rule on its own curvature.

### 3.4 The recommended certificate
Record all of:
1. the EP moment residuals and the clipped count, with the adjoint size if the count is > 0;
2. δ_B, or δ_A/(1 − ρ̂) with ρ̂ recorded, against 1/(2K);
3. the prediction-error bound ‖X(μ_{t+1} − μ_t)‖/(1 − ρ̂) against √(tr(XΣXᵀ)/K);
4. for each boundary penalty, the score-test statistic with the right sign (§2.3);
5. the EP gradient-error estimate (§3.2) inside the budget.

A per-site maximum change is not a certificate.

## 4. Exactness

### 4.1 Theorem 2 (orthogonal design)
Let Λ be diagonal. Then, for any sites with Λ + T ≻ 0:
- **(i)** every cavity equals the data factor: P_j = Λ_jj and h_j = ℓ_j;
- **(ii)** F ≡ Σ_j log Z̃_j(Λ_jj, ℓ_j; x) + c, the exact log evidence. The Φ and φ terms cancel term by term, so F doesn't depend on the sites at all;
- **(iii)** the EP-EB hyperparameters are the exact penalized type-II ML ones, and G ≡ F − c in x, so A = B.
  - The outer step reaches the (ψ, θ) fixed point in one step at fixed λ; any further outer iterations come from λ alone.
  - The fixed-cavity Fellner–Schall rule is then the correct Laplace rule.
- **(iv)** EP's marginal means are the exact posterior means. Its variances are exact wherever Var_exact ≤ 1/Λ_jj. Elsewhere the clipped reference returns 1/Λ_jj, while unclipped EP (negative τ_j) returns the exact value.

**What the orthogonal test can and cannot check.** It validates the M-step and the tilted moments. It cannot detect errors in the cavity coupling (A vs B, the adjoint), which need a correlated design and the §1.3 checks.

*Measured* (p = 30, orthogonal):
- F equals the exact evidence for random sites to 2.6e-14;
- cavities equal the data to 1.1e-12;
- means are exact to 8e-16 and variances to 1.7e-16;
- the outer residuals were 3.8, then 3.0e-14, then 0.

### 4.2 The grid as a quadrature of the continuous density
- **(a) On the grid model itself,** the tilted moments are finite sums and exact to rounding.
- **(b) For a continuous density g in t = log s,** π_k ∝ g(t_k)Δ is the trapezoidal rule for ∫ g(t) Z_{P,h}(t) dt.
  - Z(t) = (1 + e^t uP)^{−1/2} exp(½h² e^t u/(1 + e^t uP)) is analytic in the strip |Im t| < π; its singularity is where e^t uP = −1.
  - So for g analytic in a strip of half-width a_g, the error is O(M(a) e^{−2πa/Δ}) for every a < min(π, a_g) (Trefethen & Weideman 2014).
  - The essential singularity at Im t = π grows with z² = h²/P, so strongly associated variants have a smaller usable a.
  - The moments are integrals of the same kind and converge at the same rate.
- **(c) A point mass** (a BayesR-type truth) sitting between grid points cannot be represented exactly. Its error shrinks like Δ².
- **(d) Grid invariance of the smoothness prior.**
  - Σ(Δ¹φ)² ≈ Δ∫φ′² and Σ(Δ²φ)² ≈ Δ³∫φ″², so the same continuous prior needs λ₁ ∝ Δ^{−1} and λ₂ ∝ Δ^{−3}.
  - Because λ is learned, a converged fit should rescale these itself, and a refinement test compares fits, not λ values.
  - This holds only when the fitted density is smooth on the scale of Δ. With weak smoothing, the density uses the finer grid's extra freedom, and refinement changes the model, not just its quadrature. That is why SPEC requires the refinement test with λ learned.

*Measured* (P = 400, z² = 9; worst relative error in the normalizer, mean and variance, over grid phases):

| density width w (in log s) | Δ = 1 | Δ = ½ log 2 | Δ = 0.2 | Δ = 0.1 |
|---|---|---|---|---|
| 1.0 | 4.9e-4 | 7.5e-16 | 1.5e-15 | 1.5e-15 |
| 0.5 | 4.0e-2 | 2.8e-12 | 2.1e-15 | 2.1e-15 |
| 0.25 | 0.80 | 1.2e-4 | 9.3e-13 | 1.3e-15 |

A point mass midway between grid points (Δ = ½ log 2) is at best 2.4% off, with a 0.39/0.61 split to its neighbours.

**Two consequences:**
- The reference's Δ = ½ log 2 is converged to machine precision for mixing features wider than about 0.5 in log-variance.
- A sweep that pits the learned density against a BayesR truth must either put the truth's variances on grid points, or refine Δ until §4.2 converges. Otherwise BayesR-at-truth gains a representational edge of about 1% in the tilted moments.

*Measured (refinement at fixed λ)* (p = 20; 31 → 61 points; λ₁ ×2, λ₂ ×8; λ = (1, 1)):
- predictions moved by 1.7e-3 relative (Λ-norm);
- one posterior variance moved by up to 5.8%;
- θ moved by 0.11.

At this weak smoothing, the grid is not merely a quadrature.

## 5. Stage 1 vs Stage 2

### 5.1 Where the gap comes from
- **Sufficiency.** For a Gaussian likelihood, (XᵀX, Xᵀy, yᵀy, n) are sufficient. F, every EP update and every hyper step depend on the data only through Λ, ℓ and c.
- **So a Stage 1 with the full FWL-projected Gram is Stage 2 exactly.** All of the Stage 1/Stage 2 difference comes from three sources:
  - the block truncation E = Λ − Λ_block;
  - for binary traits, the working-Gaussian Λ;
  - how σ² and κ are handled.

### 5.2 The warm start, and when it is in the right basin
- **First-order shift.** x₂* − x₁* ≈ B₂⁻¹ g₂(x₁*), where g₂(x₁*) is Stage 2's own fixed-cavity gradient at the Stage 1 hyperparameters. It costs nothing extra: it is the first Stage 2 hyper step.
- **Distance in posterior units.** δ_B(x₁*) = ½ g₂ᵀB₂⁻¹g₂ is half the squared distance in posterior standard deviations of the hyperparameters.
- **Basin guarantee.** Suppose the Stage 2 outer map is a contraction with factor ρ on a ball B(x₁*, r), and its first step satisfies ‖M₂(x₁*) − x₁*‖ ≤ (1 − ρ)r. Then Stage 2 stays in the ball and converges to the unique fixed point there (Banach). ρ is read from the spectrum of A⁻¹B′ at x₁* (§2.2).
  - The Newton-form alternative is Kantorovich: with β = ‖B₂⁻¹‖, L the Lipschitz constant of B₂ and η = ‖B₂⁻¹g₂‖, βLη ≤ ½ suffices.

*Measured* (40 variants, AR(1) r = 0.8, four LD blocks of 10; the off-block part is 42% of ‖Λ‖_F):
- Stage 1 took 7 outer steps. Stage 2 took 15 from cold and 7 from Stage 1.
- At the warm start, the A⁻¹B′ spectrum is [0.98, 1.30], a contraction with ρ ≤ 0.30, so the start is in the basin.
- The hyperparameter shift is 1.72. The one-step Newton prediction leaves 0.60 of it, so the map is noticeably nonlinear at this LD.

*Measured, three seeds* (the relative prediction difference is in the Λ norm against the Stage 2 posterior mean):

| seed | Stage 1 means vs Stage 2 | Stage 1 hyperparameters, full-LD posterior, vs Stage 2 | Stage 2 start δ_A (nats) | outer steps: Stage 1 / Stage 2 cold / Stage 2 warm |
|---|---|---|---|---|
| 71 | 62% | 21% | 0.176 | 7 / 15 / 7 |
| 72 | 47% | 2.0% | 0.038 | 10 / 13 / 10 |
| 73 | 44% | 2.5% | 0.211 | 10 / 8 / 17 |

What the table shows:
- **The block-diagonal posterior itself is far off:** 44–62%. Its hyperparameters are close: about 0.3–0.65 posterior standard deviations, since √(2δ_A) = 0.28–0.65 and A⁻¹B is near 1 here.
- **Put the Stage 1 hyperparameters under the full-LD posterior** and the prediction is within 2–21% of Stage 2. So the full-data posterior, Stage 2, carries the weight, consistent with MODEL §5 (2.7× off at p/n = 20).
- **The warm start is in the basin but doesn't always save steps.** Seed 73 took 17 warm steps against 8 cold, because Anderson's path differs.
- **So the warm start's value is the certified basin, not a guaranteed speed-up.**

### 5.3 The decoupled Stage 2 mean problem: convexity with our prior sites
- **Setup.** Freeze the site precisions T and the marginal variances D (diagonal), with cavity precisions P_j = 1/D_jj − τ_j > 0. Let ζ_j(h) = log Z̃_j(P_j, h). It is **convex for every prior p_j**, being a log-Laplace transform, with ζ_j″ = Ṽ_j. Let ζ_j* be its conjugate.

**Theorem 3.** The mean-matching conditions for all sites, together with μ = Σ(ℓ + ν), are exactly the stationarity conditions of

  E(μ) = ½ μᵀ(Λ − diag P)μ − ℓᵀμ + Σ_j ζ_j*(μ_j),  with  ∇²E = Λ + diag(1/Ṽ_j − P_j).

*Proof.*
- The site shift is ν = (Λ+T)μ − ℓ and the cavity shift is h_j = μ_j/D_jj − ν_j.
- Mean matching, m̃_j(h_j) = μ_j, is the same as h_j = ζ_j*′(μ_j).
- Substituting gives [(Λ + T − D⁻¹)μ − ℓ]_j + ζ_j*′(μ_j) = 0, and Λ + T − D⁻¹ = Λ − diag P. ∎

Consequences:
- **(i)** E is strictly convex wherever every tilted variance is below its cavity variance (Ṽ_j < 1/P_j).
  - That holds automatically for log-concave priors (Brascamp–Lieb).
  - For our scale mixture it holds everywhere except at the variants whose moment-matched site precision would be negative: the clipped set.
- **(ii)** At an unclipped EP fixed point with D = diag Σ, ∇²E = Λ + T = Σ⁻¹ ≻ 0.
- **(iii)** At clipped sites the diagonal falls by P_j − 1/Ṽ_j > 0. E is then convex only if Λ absorbs it.
- **The mean-only iteration** (solve for μ at fixed ν, then match each site's mean) has Jacobian

  J = diag(Ṽ/D − 1)(D⁻¹Σ − I)  in ν.

  It is zero when the frozen variances equal the tilted ones, and ρ(J) ≤ max_j |Ṽ_j/D_jj − 1| · max(λ_max(C) − 1, 1 − λ_min(C)), where C is the posterior correlation matrix.
- **So the frozen variances need relative accuracy below 1/(λ_max(C) − 1).** Stochastic variances with error ε diverge once ε(λ_max(C) − 1) ≳ 1. This is the derived form of the Seeger & Nickisch (2011) observation reported by lit-ep, and the reason Hutchinson variances must stay out of site updates.

*Measured* (frozen T at the EP fixed point; D the exact diagonal of Σ times a log-normal relative error of size ε):

| case | clipped | min eig ∇²E | min(1/Ṽ − P) | ε | predicted ρ(J) | observed rate |
|---|---|---|---|---|---|---|
| r = 0.9, smooth | 0 | 120.6 | +91.9 | 0 | 1e-14 | 1 step |
| | | | | 0.01 | 0.0028 | converged in 4 |
| | | | | 0.05 | 0.012 | converged in 5 |
| r = 0.9, random | 4 | 22.1 | −26.4 | 0 | 0.097 | 0.098 |
| | | | | 0.05 | 0.149 | 0.134 |
| | | | | 0.2 | 0.94 | diverged |
| r = 0.97, random | 2 | 5.7 | −3.0 | 0.01 | 0.054 | 0.056 |
| | | | | 0.05 | 0.060 | 0.053 |

What the rows show:
- With clipped sites, ∇²E loses diagonal (min(1/Ṽ − P) < 0), but Λ absorbs it and E stays convex.
- With exact variances and no clipping, the mean-only iteration converges in one step, as J = 0 predicts.
- Clipped sites make Ṽ ≠ D even with exact variances, so J ≠ 0.
- A 20% variance error diverged.

*Measured on near-duplicate blocks* (16 columns at r = 0.999, and 24 at r = 0.9999; smooth prior; no clipped site):
- The **posterior** correlation matrix C of (Λ+T)⁻¹ had λ_max = 1.11 and 1.10. The sites' prior precision decorrelates the posterior even when the genotypes are near-collinear.
- The LD factor that matters is therefore the posterior one. It is large only for duplicated variants with small site precision, i.e. strongly associated ones.
- Predicted and observed rates stayed ≤ 0.006, with the product bound holding in every row.

**A second, sharper condition.** With frozen D̃_j = D_jj(1 + ε_j), the cavity precision is P̃_j = (P_j + τ_j)/(1 + ε_j) − τ_j. It stays proper iff

  ε_j < P_j/τ_j,

the ratio of data precision to site (prior) precision.
- Heavily shrunk variants have P_j ≪ τ_j, which is most variants under a sparse prior. For them even a few percent of overestimated variance makes the cavity improper.
- That happened at ε = 3–10% on the duplicate blocks, and at ε = 20% on two of the three AR designs.
- So stochastic variances in the site updates fail first through this condition, not through the rate.

## 6. What this changes
**Oracle (the dense reference).**
1. Choose λ by the §2.3 test and the V(∞) comparison instead of the absorbing ceiling. V can be bimodal with its minimum at the default start, so compare V at both ends, in closed form, with any interior optimum.
2. Use the total curvature B in the λ step (§1.4).
3. Either allow negative site precisions under a positive-definite guard, or add the adjoint correction. Clipping occurred at every fitted fixed point in 60 tries, so without one of these the target is not a stationary point of the EP evidence (§1.3, §2.2).
4. Derive the damping (§2.1) rather than fixing it at 0.3.
5. Report the certificate of §3.4.

**Engine (production).**
1. Share the objective F and use Theorem 1 as the hyper-gradient identity; σ² comes from the closed form of §1.3.
2. Take products Bv by warm EP re-solves, both for Newton–CG outer steps and for the λ traces.
3. Keep Hutchinson variances out of site updates. In the decoupled scheme, each frozen variance must satisfy ε_j < P_j/τ_j, or the cavity turns improper. Also refresh whenever max|Ṽ/D − 1|(λ_max(C) − 1) approaches 1, where C is the posterior correlation (§5.3).
4. Stop on the K-draw tolerances (§3.3).

**Prior sweep.** Put the BayesR truth on grid points, or refine Δ (§4.2).

## 7. Verification

| Claim | Check | Result |
|---|---|---|
| Lemma 1 and its clipping form | `sites` | ∂F/∂ν ≤ 4e-8; ∂F/∂τ matches the prediction to 2e-8 |
| Theorem 1, the hyper-gradient identity | `hypergrad` | 6e-9 unclipped; the adjoint closes the clipped gap to 1e-10 |
| σ² stationarity | `noise` | clipping term predicted to 7 digits |
| Parallel damping bound | `parallel` | predicted ω_max separates convergence from divergence; rates match |
| Theorem 2 (orthogonal) | `orthogonal` | exact to 1e-12 or better; one outer step |
| Outer Jacobian and relaxation | `outer`, `outer_unclipped` | contraction 0.182 as predicted; no unclipped fixed point in 60 fits |
| λ creep ratio | `lambda` | 1.61 predicted vs 1.62 observed |
| Evidence profile in λ₂ | `vprofile` | bimodal; global maximum at ∞; the closed form is exact to 1e-7 |
| Accept-if-rises | `accept`, `accept_unclipped` | stalls when clipped; converges when unclipped |
| Certificate bounds | `outer`, `certificate_unclipped` | the prediction bound holds to 5 digits; δ_B equals the gap to 4–6 digits |
| Incomplete-EP gradient error | `incomplete` | the tail estimate is within 4% over 11 decades |
| Grid quadrature and refinement | `quadrature`, `grid` | table in §4.2 |
| Stage 1 vs Stage 2 | `stages`, `stages_seeds` | the full-LD posterior carries the weight |
| Decoupled convexity and rate | `decoupled`, `decoupled_duplicates` | the Jacobian rates match; ε < P/τ is the binding condition |

References:
- Minka 2001 (UAI);
- Heskes & Zoeter 2002 (UAI);
- Opper & Winther 2005 (JMLR 6:2177);
- Seeger 2005 (tech. report, EP for exponential families);
- Seeger & Nickisch 2011 (AISTATS);
- Wood & Fasiolo 2017 (Biometrics 73:1071);
- Walker & Ni 2011 (SIAM J. Numer. Anal. 49:1715);
- Trefethen & Weideman 2014 (SIAM Review 56:385);
- Bach 2010 (Electron. J. Statist. 4:384).

These are standard results cited from memory; lit-ep's notes hold the verified entries.
