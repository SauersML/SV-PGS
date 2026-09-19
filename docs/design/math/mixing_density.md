# The learned mixing density: range, quadrature, penalty order, λ = ∞, pins and the M-step

This note derives the mathematics of the continuous mixing density g in MODEL.md §3 (SPEC 131b205) and checks every claim numerically.
- **Scripts** (lane math-density, run on MSI acl42): `mdcore.py`, `check_range_quadrature.py`, `check_certified_quadrature.py`, `check_penalty_order.py`, `check_opg.py`, `check_pooled_curvature.py`, `check_concavity_pooling.py`, in `/scratch.global/sauer354/svpgs-team/math-density/`. They are not in the repo; each section ends with what was measured.
- **Related:** `ep_eb.md` (how the density couples to EP, the certificate), `scale_model.md` (the scale model and its pins), `prior_sweep_a.md` (the family comparison).

## 0. Notation

- **Scale.** t = log s. Variant j's EP cavity is exp(−½P_jβ² + h_jβ); its prior variance at scale s is v = u_j e^t. Write z_j² = h_j²/P_j and c = v/(1 + vP_j).
- **Kernel.** The component likelihood at t is L_j(t) = (1 + vP_j)^{−1/2} exp(½h_j² c). It is the tilted normalizer given t, normalized so that L_j(−∞) = 1.
- **Density.** g(t) = e^{η(t)}/∫e^η on the real line. The class's data term is ℓ(η) = Σ_j log Z_j with Z_j = ∫ g L_j dt.
- **Roughness.** J_m(η) = ∫(η^{(m)})² dt, with penalized objective F = ℓ − ½λJ_m.
- **Lattice.** Nodes t_k = t_0 + kh. The lattice form of J_m is h^{−(2m−1)}‖Δ^m η‖², because Δ^m η ≈ h^m η^{(m)}.
- **Responsibilities.** r_jk = w_k L_jk/Z_j, w = softmax(η), R = Σ_j r_j, and G = R − nw is the η-gradient of ℓ. C(p) = diag p − ppᵀ.

## 1. The kernel

- **(K1) Mode and monotonicity.** ∂_v log L = ½(h² − P − vP²)/(1 + vP)².
  - L_j peaks at t*_j = log((h_j² − P_j)/(P_j² u_j)) when z_j² > 1. It is strictly increasing below the peak and strictly decreasing above it.
  - When z_j² ≤ 1 it decreases everywhere.
- **(K2) Width.** At the mode, ∂_t² log L_j = −½((z²−1)/z²)² ∈ (−½, 0).
  - So one variant's likelihood is never narrower than √2 in t. Resolution finer than that comes only from many variants, at the logarithmic rate of deconvolution.
- **(K3) Analyticity.** L_j is analytic in |Im t| < π. Its singularity is at t = −log(u_jP_j) + iπ, where 1 + vP = 0.
- **(K4) The π/2 strip.** On |Im t| ≤ π/2, the value 1 + vP has real part ≥ 1, so |1 + vP| ≥ 1, Re c ≤ 1/P and |c| ≤ 1/P.
  - Hence |L_j| ≤ e^{½z_j²} there, and the moment factors hc and c + h²c² are bounded by |h|/P and 1/P + h²/P².
- **Measured** (400 simulated variants):
  - mode formula vs dense grid: error ≤ the grid step (5e-4);
  - decreasing beyond the mode for every variant;
  - curvature at the mode: relative error 8e-6;
  - on the π/2 strip: max of Re(½h²c) − ½z² is 6e-14, and min |1 + vP| = 1.000.

## 2. The range

**(R1) Lower end, an effective zero.**
- For v = u_j e^t, integrating |∂_w log L| ≤ ½(|h² − P| + wP²) over [0, v] gives |log L_j(t)| ≤ b_j(t) = ½v|h_j² − P_j| + ¼v²P_j².
- **Rule:** choose s_min with Σ_j b_j(u_j s_min) = tol_lo, i.e. s_min = 2·tol_lo/(A₁ + √(A₁² + 4A₂·tol_lo)), with A₁ = ½Σ_j u_j|h_j² − P_j| and A₂ = ¼Σ_j (u_jP_j)².
  - Use this stable form. The textbook root cancels catastrophically, and was 2.6% off in the test.
- **Below t_min = log s_min, set log L_j ≡ 0.** Z_j is a mixture, so perturbing some of its component likelihoods by factors within e^{±b_j} moves Z_j by a factor within e^{±b_j}. Hence Σ_j |Δ log Z_j| ≤ tol_lo.
  - The tilted mean from components below t_min is |h|c ≤ |h|u_j s_min.
- **The density's mass below t_min is kept.** For a sparse class most of g lives there. On the lattice these are nodes with log L = 0 exactly, so no kernel evaluations are needed.
- **Other published rules:**
  - The rule u e^t(P + h²) ≤ 2·tol/p is also valid, but looser.
  - The first-order term alone (lit-eb) is not a bound.
- **Measured:** sup_{t ≤ t_min} |log L_j| equals the bound to 1e-12 (the bound is tight). Summed over variants it is 1.0000026e-6 against tol_lo = 1e-6.

**(R2) Upper end.**
- t_max = max_j t*_j. Beyond it every L_j decreases (K1), so moving mass inward raises every Z_j: the NPMLE has no mass beyond t_max (Lindsay).
- A smooth g still has a tail there, and it is part of the model. Keep lattice nodes until the fitted mass M₊ beyond the last node satisfies M₊·(D(t_end) + 2n) ≤ tol_hi, with D(t) = Σ_j L_j(t)/Z_j − n.
  - Dropping mass M₊ and renormalizing moves Σ_j log Z_j by at most that amount.
- **D(t_hi) < 0 is not sufficient.** It is the NPMLE's condition, and the smooth tail's mass still enters the evidence.
- **Measured:** every L_j is non-increasing on [t_max, t_max + 20].

**(R3) Lower tail of the lattice.**
- Extend the lattice below t_min until the fitted mass M₋ beyond the last node satisfies M₋·|D(−∞)| ≤ tol_lo′, with D(−∞) = Σ_j 1/Z_j − n.
  - That is the first-order change in Σ_j log Z_j from moving that mass, and L ≡ 1 there.

**Location of the cut.** With R1–R3 the fit no longer depends on where the lattice is cut, to tolerance. Under D3 the data-free continuation costs nothing (§4).

## 3. The quadrature certificate

- **(Q1) The theorem.** Take the trapezoid rule on the real line with spacing h, and f analytic and integrable on |Im t| < a. Then |hΣ_k f(t_k) − ∫f| ≤ 2M(a)/(e^{2πa/h} − 1), with M(a) = sup_{|y|<a} ∫|f(t + iy)| dt (Trefethen & Weideman 2014, Thm 5.1).
- **(Q2) Applying it to f = gL_j.** Take a = π/2 (K4).
  - log ∫|f(t + iy)| dt is convex in y (Hardy), and f(t̄) = f̄(t). So M_j = max(Z_j, ∫|gL_j|(t + iπ/2) dt).
  - That is one extra lattice sum, with the kernel evaluated at complex t.
- **(Q3) The spacing rule.**
  - The relative error in Z_j is at most 2(M_j/Z_j)/(e^{π²/h} − 1).
  - Σ_j |Δ log Z_j| ≤ tol_q when h ≤ π²/ln(1 + 2Σ_j(M_j/Z_j)/tol_q).
  - The same holds for the tilted moments, with M bounded as in K4.
  - **The M_j/Z_j factor is not optional.** It reached about 2·10³ in the test. The p-only rule h ≤ π²/ln(2p/tol) predicted 5e-9 at h = 0.5, and the actual error was 6e-6.
- **(Q4) Where Q1 applies.** It needs log g analytic on the strip, hence smooth on the whole line.
  - A piecewise representation (a Chebyshev series on a range with a polynomial continuation outside) jumps in η^{(m)} at the junction, which drops the rate to algebraic. Keep log g smooth across the whole lattice, or give each piece its own certified rule.
- **(Q5) Gauss–Legendre over one long range is not certified by the kernel alone.**
  - The Bernstein parameter from K3 (ρ = b + √(1 + b²), b = 2π/L at the centre) sets the asymptotic rate only.
  - The growth of |e^η e^{½h²c}| on the ellipse sets the constant. Observed errors exceeded the ρ-based a posteriori estimate by up to 10⁷.
  - Composite Gauss–Legendre is certified: panels of length ℓ, each with its ellipse at semi-minor axis π/2 in t (ρ = π/ℓ + √(1 + π²/ℓ²)), and M sampled on that ellipse.
- **(Q6) Stopping.** With Q3 no refinement loop is needed for the quadrature: compute the certificate at the fit and lower h if it fails.
  - Halving comparisons (err(h/2) ≈ err(h)²) hold only in the geometric regime.
  - The representation error of η is a separate question (§4).
- **Measured:**
  - Trapezoid, g a three-component normal mixture in t, 60 variants: the actual error was ≤ 0.87× the Q3 bound at every h in {1.5, 1.2, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3}. Max relative errors were 3e-1, 1e-1, 4e-2, 6e-3, 1.4e-4, 6e-6, 4e-8 and 6e-12.
  - Composite GL: the bound held for ℓ = 2 and 4 at every node count; ℓ = 2 with 12 nodes per panel gave 9e-12. For ℓ = 1 the sampled-ellipse bound was exceeded by up to 3× at 4–8 nodes.
  - One GL rule on a 32-unit range, with a degree-14 Chebyshev log g: 5e-7 at 64 nodes, 4e-10 at 96, 3e-14 at 128. The ρ-based estimate failed at every level.

## 4. Penalty order: D3

**(P1) Null spaces and λ = ∞ limits.** J_m's null space is the polynomials of degree < m in t.
- **D1:** constants, i.e. a uniform log g on whatever range the lattice spans.
- **D2:** linear. e^{bt} is integrable at neither end of ℝ, so the limit exists only on a truncated range, as a truncated power law that depends on the range ends.
- **D3:** quadratic. For negative curvature this is the log-normal N(t; μ, σ²), proper on ℝ and independent of the range. D3 is the lowest order with a proper limit, and that limit is the maximum-entropy density for given E[t] and Var[t].
- **D1+D2:** λ₁ = ∞ gives the uniform density on the range, whatever λ₂ is.

**(P2) Power-law tails stay reachable at finite λ under D3.** A linear piece has η''' = 0, so its only cost is the transition. The zero-cost continuation of the fit into a data-free region is quadratic.

**(P3) Lattice scaling.** J₃ ≈ h⁻⁵‖Δ³η‖², since Δ³η ≈ h³η'''. λ is then a continuous-model quantity.

**Measured, refinement** (check_refine.py, check_refine_capped.py; 200 normal means, P ~ logN(log 10⁴, 0.5), u ~ logN(0, 0.3)).
- **Evidence:** the ruled form, V = F + ½log|S|₊ − ½log|B + S| + ½log|NᵀHN|, with the observed curvature and the null space profiled.
- **Search:** λ̂ by a coarse scan plus golden section on log λ, coordinate-wise for two weights.
- **Table:** log λ̂ at h = 0.8 / 0.4 / 0.2 / 0.1, and the largest prediction change ‖XΔμ‖/‖Xμ‖ between neighbouring h:

| truth | D2 | D1+D2 (λ₁, λ₂) | D3 |
|---|---|---|---|
| BayesR (atoms 1e-9/1e-4/1e-3/1e-2, p = .9/.06/.03/.01) | 2.49 / 2.48 / 2.47 / 2.47; ≤ 7e-5 | (−1.24, 2.36) … (−1.23, 2.33); ≤ 1.9e-4 | 18.7 / 17.1 / 10.3 / 10.1 (uncapped) and 20.9 / 16.7 / λ=∞ / 9.3 (capped), along a ridge flat within 0.15 nats, V(∞) = 141.737; ≤ 8.5e-4 |
| log-normal (σ_t = 1.2) | −1.88 / −1.94 / −1.96 / −1.96; ≤ 2.1e-4 | (−5.38, −1.42) … (−5.41, −1.52); ≤ 2.3e-4 | see below |
| bimodal (80% at 1e-7, 20% at 2e-3) | 0.03 / −0.04 / −0.05 / −0.05; ≤ 4.4e-4 | (−0.82, −7.97) … (−0.84, −7.97); ≤ 3.7e-4 | −0.47 / −0.47 / −0.47 / −0.46 (capped scan); ≤ 7.5e-5 |

- **D3 on the log-normal truth** (the fitted log-normal has σ = 0.83 in t; capped scan):
  - At h = 0.8 and 0.4 the interior V exceeds V(∞) by 2.2 and 0.45 nats, and the excess vanishes as h shrinks. That is consistent with quadrature error for this narrow density. The Q3 certificate was not evaluated in these runs.
  - At h = 0.2 the evidence selects λ = ∞: V(∞) = 217.710 against 217.709 in the interior.
  - **At h = 0.1 my harness is not reliable for this truth.** The uncapped and capped searches disagree: V = 218.12 at λ = e^{3.8}, and V = 226.36 at λ = e^{6.9}, against V(∞) = 217.71. Predictions still moved by only 9e-4 from h = 0.2.
  - Reading: on a log-normal truth D3's λ is weakly identified between about e³ and ∞. Across h the choice moved predictions by ≤ 0.8%. The h = 0.1 evidence values need a better-conditioned implementation before they can be trusted.
- **Numerical limit.** Direct fits at λ ≳ e¹⁵ lose all precision, because the condition number of λh⁻⁵P exceeds double precision. One scan at h = 0.1 returned V = 2842 against ≈ 317. So the λ scan is capped at λ‖P‖ ≤ 10¹², and λ = ∞ is evaluated in closed form (§5) instead.
- **The one reported non-invariance was the evidence form.** prior's D3 on 1k BayesR variants had λ̂ collapse to e^−18.4 at h = 0.125. That came from the flat-prior integrated evidence, whose level grows with K (3449.8 → 3485.3). Under the profiled form, prior measured log λ̂ 3.55 / 2.04 / 1.03 / 0.70 at h = 0.5 / 0.25 / 0.125 / 0.1, with V converging (3448.96 → 3449.99) and predictions within 0.46%.

**Measured, range invariance of the λ = ∞ limit** (check_range_null.py; global fits; h = 0.2).
- **Ranges:** base [t_min − 12, t_max + 6], then doubled and quadrupled about the same centre.
- **Table:** log-likelihood of each λ = ∞ model, base / ×2 / ×4:

| truth | D3 log-normal | D2 power law | D1 uniform |
|---|---|---|---|
| BayesR | 141.73729 / 141.73722 / 141.73722 | 136.8 / 133.1 / 130.9 | 109.4 / 65.1 / 37.1 |
| log-normal | 217.71003 at all three | 153.7 / 117.2 / 100.3 | 153.7 / 91.9 / 46.2 |
| bimodal | 315.4548 / 315.4460 / 315.4460 | 307.0 / 293.8 / 285.0 | 296.1 / 245.8 / 211.7 |

- D3's limit moves by ≤ 1e-4 nats. The exception is bimodal from base to ×2, where the base range cuts the σ = 3.6 tail at μ − 8σ.
- D2's and D1's limits move by 4–100 nats.
- **Ruling (lead):** D3 on SPEC 131b205 grounds, because range invariance separates it from D1 and D2.

## 5. The λ = ∞ boundary

**(B1) The expansion.**
- Setup, for one penalty J with the others fixed:
  - τ = 1/λ; N is J's null space and R its range, in sum-to-zero coordinates;
  - η* is the maximizer of ℓ, minus the other penalties, over N;
  - g = ∇ℓ(η*), so g ⟂ N; A = −∇²ℓ(η*) (the curvature used in the evidence, §7);
  - Ã = A_RR − A_RN A_NN⁻¹ A_NR, and J⁺ is the pseudo-inverse on R.
- As τ → 0, V(λ) = V(∞) + (τ/2)·T + O(τ²), with T = gᵀJ⁺g − tr(J⁺Ã).
  - The fit gain is ½gᵀ(A + λJ)⁻¹g = (τ/2)gᵀJ⁺g + O(τ²).
  - The log-determinant is log|A + λJ| = log|A_NN| + log|λJ|₊ + τ·tr(J⁺Ã) + O(τ²).
- **With other penalties S_o present** (math-epeb): T = gᵀJ⁺g − tr(J⁺[(A + S_o)~ − (S_o)~]), where ~ is the Schur complement with respect to N.
- **V(∞) depends on the evidence convention.** With N profiled (fixed effects, the lead's ruling, since the flat-prior integral over N diverges), V(∞) is the restricted fit's value plus the other penalties' terms. T is the same under either convention.

**(B2) The test.**
- λ = ∞ is a first-order (KKT) optimum iff T ≤ 0. This is the variance-component score test at the boundary.
- lit-ep's Σ_k(q_k² − d_k) is the same statistic, with q = J⁺^{1/2}g and d_k the eigenvalues of J⁺^{1/2}ÃJ⁺^{1/2}. Measured: the two forms agree to ≤ 2e-7 relative on D2 and D3 (check_boundary.py).

**(B3) Why Fellner–Schall creeps.**
- Near ∞, λ_new/λ → ρ = tr(J⁺Ã)/(gᵀJ⁺g). FS runs to ∞ iff ρ > 1 ⇔ T < 0, at a constant log ρ per step, and never arrives.
- The oracle's runaway, +0.6 to +1.0 in log λ per step, means ρ ≈ 1.8–2.7, i.e. T ≈ −(0.45 to 0.63)·gᵀJ⁺g: that class's optimum was λ = ∞.
- math-epeb measured ρ = 1.607 predicted against 1.62 observed on the reference.
- **The expansion itself, measured on D2** (check_boundary.py, h = 0.3):
  - λ(V − V∞) = 32528 at λ = e¹⁴, against T/2 = 32556 (log-normal truth), and 7320 against 7327 (bimodal).
  - The FS ratio is 0.02440 against a predicted 0.02431.
- **For D3 the asymptotic regime starts at λ ≫ ‖J⁺A‖ ≈ 10⁷–10⁸,** which direct fits can't reach (§4). V(∞) and T are therefore computed in closed form.

**(B4) The λ = ∞ fit must be found globally.** The objective restricted to the null space is not concave (C2).
- A local Newton on the BayesR truth stopped at log-likelihood 137.6, with curvature −0.007: a nearly flat quadratic filling the range. The global log-normal fit reached 141.74.
- For D3 the null model is two-dimensional (μ, log σ), so use a grid plus a local polish.
- **Degenerate cases:** the proper family needs curvature < 0. Its supremum is attained in the interior or at the effective-zero limit μ → −∞, never at σ → ∞, which the null limit dominates.

**(B5) The procedure.**
- Evaluate V in closed form at every boundary face: λ_i = ∞ by B1, and λ_i → 0 by §7.
- Find the interior stationary points by Newton on log λ with the exact gradient, from more than one start.
- Take the largest. FS is not a maximizer: on the reference it descended from λ = 1 to the wrong boundary, while the global maximum was at λ = ∞, 0.17 nats higher (math-epeb).

## 6. Identifiability and pins

- **(I1) Level vs location.**
  - Prior variance is u_j e^t with log u_j = level_c + …. The map (level_c + a, g(t) → g(t + a)) leaves every Z_j unchanged, and J_m is translation invariant, so it is an exact flat direction. On a lattice a shift by k nodes with level + kh is exact.
  - **Pin with a functional the data identify:** E_g[s] = 1, the mean prior variance, or no separate level at all (the reference family: g_c carries its own location, with d̃ centred within class).
  - **Never pin E_g[log s].** The lower tail below resolution is unidentified (R1), so mass there can meet any E[log s] while the level drifts.
    - math-scale's profile check (20k normal means) measured the log-likelihood drop at level −1.5 … +1.0:
      - with the E[log s] pin: 0.00 / 0.00 / 0.03 / 0.44 / 3.3 / 18.9, with the maximum at the range edge;
      - with the E[s] pin: 442 / 221 / 54 / 0 / 3.9 / 7.4, a sharp maximum at the truth.
  - For either pin, the Jacobian of the section along the orbit is constant: ∂_a E_{g(·+a)}[t] = −1 and ∂_a log E_{g(·+a)}[s] = −1. So the Laplace evidence changes by a λ-independent constant.
- **(I2) Pooled classes.** log g_c = η̄ + δ_c − log Z_c. The trait term was dropped, per the user.
  - (η̄ + q, δ_c − q for every c), with q in null(J₃) (linear and quadratic in t), leaves both the likelihood and both penalties unchanged, so it is exact and flat.
  - It is removed by any one of:
    - a pin Σ_c Π_N δ_c = 0;
    - tilt-free and width-free δ_c;
    - a proper N(0, τ²) prior on δ_c's null-space part with τ² learned (the lead's ruling, §7).
  - **Measured (C4, C5):**
    - The two pins Σ_c Π_N δ_c = 0 and Π_N δ_1 = 0 give the same fit, with F equal to 1e-7.
    - Across five (λ̄, λ_δ) pairs, the LAML difference between those two pins was 20.06–21.04 nats. Exact arithmetic predicts a constant (det(GᵀHG) = pdet(H)·det(GᵀU_R)², a factor that depends only on the pin geometry). I did not isolate the cause of the spread. It may be the same issue as the unclean LAML-gradient check in §7.
    - A lattice shift of 4 nodes, with level + 4h, reproduces the log-likelihood to 2e-12 relative (765.99092801 in both).

## 7. Derivatives, the evidence and its curvature

**Pooled objective.**
- η_c = B(ψ̄ + ψ_c) in sum-to-zero coordinates, where B is an orthonormal basis of 1^⊥. Write M_c for the design that picks (ψ̄, ψ_c).
- **Gradient:** Σ_c M_cᵀBᵀ(R_c − n_c w_c) − Sθ.
- **Hessian:** Σ_c M_cᵀBᵀ[diag(R_c − n_c w_c) − Σ_j r_j r_jᵀ + n_c w_c w_cᵀ]BM_c − S.
- **Third derivative contracted with e:** Σ_j κ₃(r_j)[e] − n_c κ₃(w_c)[e], where κ₃(p)[e] = diag(q) − qpᵀ − pqᵀ and q = p∘(e − pᵀe). This is the categorical third cumulant.
- **Exact LAML gradient in ρ_i = log λ_i:**
  ∂V/∂ρ_i = −½λ_i θ̂ᵀS_iθ̂ + ½λ_i tr(S⁺S_i) − ½tr(H⁻¹[λ_iS_i − 𝒯[dθ]]),
  with dθ = −H⁻¹λ_iS_iθ̂ and 𝒯 the contracted third derivative. Fellner–Schall drops the 𝒯 term.
- **Measured** (check_concavity_pooling.py; three classes of 120, 40 and 15 variants; D3 on both terms):
  - Against finite differences, the gradient agrees to 1.2e-9 relative, the Hessian to 1.0e-10 and the third-derivative contraction to 3.6e-5.
  - The flat directions of I2 are exact: |vᵀHv| is 8e-12 against a Hessian scale of 2.4e5.
  - My own check of the full LAML gradient against finite differences was not clean: −1.44 against −1.20. I did not isolate the cause (log-determinant precision in the pinned pooled H, or my harness). The engine's implementation of this gradient matches finite differences to 1e-5 (e2e's report).

**Which curvature goes in −½log|H|.**
- The observed data information and the OPG (empirical Fisher, Σ_j(r_j − w)(r_j − w)ᵀ) differ exactly by the penalty's pull:
  −∇²ℓ = OPG − diag(G) + Gwᵀ + wGᵀ, where G = R − nw, and BᵀG = λSψ̂ at the fit.
  - Verified to 1e-16.
  - The two coincide at any unpenalized stationary point.
- **Ruling (lead):** the evidence uses the observed B + S. η̄'s null space is profiled, and δ_c's null-space part gets a proper N(0, τ²) with τ² learned and integrated.
- **Measured** (check_pooled_curvature.py):
  - Setup: two classes of 120 + 30 and 360 + 90 variants; λ_d scanned from e⁸ down to e⁻¹²; τ² ∈ {e⁻¹, e²}.
  - Under that ruling V is smooth. It falls steadily as λ_d → 0, and H stays positive definite. The earlier spike, where the observed curvature cancelled the prior's along the collapse direction, does not occur.
  - Both curvatures give the same λ̂_d on a 0.5-step scan: log 8.5 at n = 150; log 3 at n = 450 for both τ². V_obs − V_OPG is about 0.4–0.6 nats near the optimum.
  - A coarse 2.5-step grid had split them at n = 150 (log 8 vs 3). That reflects how flat V is there: the prediction change between those two λ was 1.6%.
- **Small classes (a real weakness of the Laplace evidence).** As λ → 0, V → F + Σ over informed directions of ½log(λμ/o). The number of informed directions is about the number of NPMLE atoms minus one, so for a class of about 25 variants V approaches −∞ only very slowly. In a 25-variant check it was still rising at λ = e⁻¹⁰. Class pooling is the remedy.

## 8. Concavity and the M-step

- **(C1) F is not concave in η.**
  - ∇²ℓ = Σ_j C(r_j) − nC(w), and a responsibility vector can be more spread out than w.
  - For K = 2, take w = (1−ε, ε) and data that make r = (½, ½). Then ∂²ℓ/∂φ² = ¼ − ε(1 − ε) > 0 when ε < 0.146. Measured: at ε = 0.05 the finite difference is 0.20250002, against the formula's 0.2025.
- **(C2) The positive curvature can lie in the penalty's null space** (the D3 quadratics, or the D2 tilt), where no λ removes it. Measured: on the BayesR truth the D3 null-space fit has at least two local maxima. A local Newton stopped at log-likelihood 137.6, against the global 141.74 (§5 B4).
- **(C3) No convex reformulation.** In w-coordinates ℓ is concave (the log of a linear function), but the roughness of log w is not convex in w, so no jointly convex form keeps a log-smoothness penalty.
- **(C4) The EM minorizer.** Q(η | η⁰) = Σ_k R⁰_k η_k − n·logsumexp(η) − ½λJ(η) has Hessian −nC(w) − λJ ≺ 0 on 1^⊥, so each step is strictly concave and gives monotone ascent, at a linear rate.
- **The M-step to implement: safeguarded Newton.**
  - Use the observed Hessian when −∇²F ≻ 0 (checked by Cholesky). Otherwise use a spectrum-shifted Hessian (|eigenvalues|) or the minorizer's.
  - Armijo backtracking on F.
  - This is quadratic near a strict local maximum.
- **(C5) On average the objective is concave.** Under the model, E[∇²ℓ] = −Σ_j Cov(E[e | y_j]) ≼ 0, so non-concavity is a finite-sample effect.

## 9. The recipe

1. At fixed cavities: compute t*_j, t_min (R1) and t_max (R2).
2. Choose h from Q3, with M_j from one complex line sum. Extend the lattice past t_min and t_max until the tail criteria R2 and R3 hold.
3. Represent log g as nodal η on the lattice, with J₃ = h^{−5}‖Δ³η‖² (§4). Use the location pin E_g[s] = 1 or no level (I1). Handle the pooled deviations as in I2 and §7.
4. M-step: safeguarded Newton (§8).
5. λ: Newton on log λ with the exact gradient (§7), plus the closed-form boundary values (§5). Take the largest.
6. Tolerance: tol = tol_lo + tol_lo′ + tol_hi + tol_q, where tol is the certificate's tolerance in nats. Any split works; the fit is within tol whatever the split.

## 10. The same rules for a vector kernel

- For a shared scale over a vector of T cavities, K_j(t) = N(m_j; 0, e^tU_j + V_j).
- With V^{−1/2}UV^{−1/2} = QΛQᵀ and y = QᵀV^{−1/2}m,
  log K_j(t) = log K_j(−∞) + Σ_k[−½log(1 + e^tλ_k) + ½y_k² e^tλ_k/(1 + e^tλ_k)].
  That is a sum of scalar kernels with uP → λ_k and z² → y_k², so K1–K4, R1–R3 and Q1–Q3 carry over with sums over k:
  - singularities at −log λ_k + iπ;
  - |K_j/K_j(−∞)| ≤ e^{½‖y‖²} on |Im t| ≤ π/2;
  - b_j = Σ_k[½e^tλ_k|y_k² − 1| + ¼e^{2t}λ_k²];
  - t_max = max_k log((y_k² − 1)₊/λ_k).
- The same boundary statistic (§5) replaces floors on Gaussian penalty weights, e.g. ν = freedom/quadratic in hyperprior pooling. The Gaussian evidence's expansion in 1/ν is exact to first order, and ν = ∞ exactly when T ≤ 0.

## 11. A related integral: the phenotype occasion model

For deslop-hygiene; lead ruling: plain certified trapezoid.

**The integral.** L = ∫N(T; 0, τ²) Π_{j=1..J} f(r_j − T) dT, with f(e) = Σ_k π_k N(e; 0, s_k), floored at the rounding variance.

**The mass-weighted strip bound.**
- |N(e + iy; 0, s)| = N(e; 0, s)·e^{y²/(2s)} exactly. The triangle inequality inside each mixture therefore gives
  M(b) ≤ ∫ N(x; 0, τ²) e^{b²/(2τ²)} Π_j Σ_k π_k N(r_j − x; 0, s_k) e^{b²/(2s_k)} dx.
  - Each occasion is weighted by its own responsibilities.
  - It costs one real-line lattice sum.
- With it, h = max_b 2πb/ln(1 + 2M(b)/(εL̂)).

**The split does not pay.** The split (narrow components by a heat-kernel expansion) has a certified remainder:
  |∫N(T; c, s)H − Σ_{m<M}(s/2)^m H^{(2m)}(c)/m!| ≤ (2M)!·s^M·M₁(b)/(π·2^M·M!·b^{2M+1}),
  from |Ĥ(ω)| ≤ M₁e^{−b|ω|}.
- It needs s ≪ σ_H², where σ_H² is H's own scale, and that scale shrinks like s_W/J.
- Terms with two or more narrow factors are bounded by Σ_{j<l} C_jl·max(rest), with C_jl = Σπ_kπ_m N(r_j − r_l; 0, s_k + s_m). Duplicated values trigger exact evaluation.

**Tolerance.** |ΔL_i|/L_i ≤ 1 − e^{−1/(2n)} ≈ 1/(2n) per person resolves ½ nat over n people.

**Measured** (check_occasion_split.py, against exact enumeration of all K^J assignments):
- **Setup:** K = 7 components (1/12·4^k); τ² = 100; tol 1e-5.
- **Plain certified trapezoid:** errors 1.6e-8 (two occasions, one gross reading), 5.8e-7 (a duplicated pair) and 1.4e-7 (six occasions with a duplicate and a gross reading). All three are within tolerance.
- **The split:**
  - gross case: within tolerance at the three smallest splits (errors 2.6e-8 to 4.4e-6, 723 to 188 nodes), but the certified bound exceeded L at each, so no split could be certified;
  - duplicated pair: 3.6e-5–6e-3;
  - six occasions: 3.4%–100%.
- **The certified bounds covered every error, 18/18,** but were loose at 0.07–270× L.
