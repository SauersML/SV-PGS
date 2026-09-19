# Products with the total curvature B, without EP re-solves

B is the Hessian of the EP evidence in the hyperparameters, with EP re-converged: the "observed" curvature the λ evidence and the certificate use (ep_eb.md §1.4, §3). It has a closed form, so no product with B needs an EP re-solve. This note gives the form, its production cost, the certified rule for how many directions the λ step needs, and the variance-refresh trigger. Sign convention: A and B are Hessians (negative definite); ep_eb.md's curvatures are their negatives.

## 1. The closed form
At an unclipped EP fixed point, for each variant j, write:
- (h_j, P_j) for the cavity's natural shift and precision;
- m_j = μ_j and v_j = Σ_jj for the tilted mean and variance, and s2_j = m_j² + v_j;
- m_x, v_x, s2_x for their explicit derivatives in the hyperparameters at a fixed cavity;
- m_P = ∂m/∂P, v_h = ∂v/∂h, v_P = ∂v/∂P, and ∂m/∂h = v.

For directions E (D × r) the fixed point moves by (δm, δP), which solve

    (i)  (Q + diag τ) δm = (m + m_P/v) ∘ δP + (m_x E)/v                        mean channel
    (ii) δv = (Σ∘Σ)(δv/v² + δP),  δv = v_h δh + v_P δP + v_x E,                variance channel
         δh = (δm − m_P δP − m_x E)/v,

and

    B E = A E + m_xᵀ δh − ½ s2_xᵀ δP.

**Derivation.**
1. Differentiate the fixed-point conditions μ = m(h, P; x) and Σ_jj = v(h, P; x), with Σ = (Q + diag τ)⁻¹, τ = 1/v − P and ν = m/v − h.
2. Substituting the local relations makes (i) collapse to the posterior-precision operator.
3. In (ii) the diagonal of Σ∘Σ cancels δv_j, so a cavity precision responds only to other variants' site changes, through squared posterior correlations.
4. B E is then the total derivative of the fixed-cavity gradient g = Σ_j ∂_x ζ_j, with ∂²ζ/∂x∂h = m_x and ∂²ζ/∂x∂P = −½ s2_x.

**Special cases.**
- For an orthogonal design Σ is diagonal and B = A (Theorem 2).
- The mean channel alone gives B = A + Cᵀ(Σ − diag Σ)C with C = diag(1/v) m_x. Its Woodbury form, with A' = A − Cᵀdiag(v)C, needs a single solve with (Σ⁻¹ + C A'⁻¹Cᵀ).

**The per-variant derivatives are closed forms over the lattice nodes.**
- Per node: responsibilities r_k, σ_k = q_k/(1+q_k), μ_k = (h/P)σ_k and c_k = σ_k/P. Below, E and Cov are taken under r.
- Moments and their cavity derivatives:
  - v_h = E[(μ−m)³] + 3E[c(μ−m)];
  - m_P = Cov(ℓ, μ) − E[μc], with ℓ_k = −(c_k + μ_k²)/2;
  - v_P = s2_P − 2m m_P.
- Class-density derivatives: ∂m/∂η_l = r_l(μ_l − m) and ∂s2/∂η_l = r_l(c_l + μ_l² − s2).
- Scale derivatives, with f_k = ∂L_k/∂log u:
  - ∂m/∂log u = Cov(f, μ) + E[μ(1−σ)];
  - ∂s2/∂log u = Cov(f, c + μ²) + E[(c + 2μ²)(1−σ)].

**Checked** (dense EP, p = 30, D = 11, AR(1) LD r = 0–0.95, every case with negative sites), against central differences with EP re-solved to a 2e-15 moment residual:
- the closed form matches to 0.9e-10–4.7e-10 relative, the finite-difference floor, in 9 cases;
- the mean channel alone leaves 0.2–19% of B, so both channels are needed;
- each per-variant derivative matches central differences to ≤ 2.2e-9.

## 2. Production cost
- **Variant-local terms:** one fused pass over variants × nodes, the same pass as the M-step statistics.
- **Mean channel:** a solve with the posterior precision. r directions are r extra right-hand sides of the Stage 2 PCG.
- **Variance channel:** through the marginal-variance map's JVP, variance_jvp(w) = −diag(Σ diag(w) Σ): δP = u + variance_jvp(u)/v², with u = δv/v² + δP.
- **Solve:** one stacked GMRES over δP for all r directions. Each iteration is one posterior solve and one JVP on r columns.
  - [measured, dense] 12–24 iterations for all 11 directions at a 1e-13 residual, with the product equal to the EP-re-solved B to 1.0e-10–4.6e-10.
  - The alternative, one warm EP re-solve per direction, needed 33–173 full EP sweeps (each a posterior refresh including the variances), and one case did not converge.
- **Certificate and x step:** δ_B = ½gᵀ(−B)⁻¹g needs one extra right-hand side, via the Woodbury form, with the variance channel as a defect correction.

## 3. How many directions the λ step needs: a certified rule
- **Setup.** Write E = B − A' = Φᵀ𝓛⁻¹Ψ, where 𝓛 is the linear-response operator of (i)–(ii) and Φ, Ψ are variant-local. Let H0 = −A' + S_λ (exact), H = −B + S_λ = H0 − E and Z = H0^{-1/2} E H0^{-1/2}.
- **Directions and approximation.** Take directions V_r as the top right singular vectors of Ψ̂. Set Z_r = Z − (I−P)Z(I−P), with P = V_rV_rᵀ; it is exact except on the unexplored complement, and costs r products.
- **Complement bound**, with Jacobi scaling T² = |diag 𝓛|:

      ‖(I−P)Z(I−P)‖ ≤ ‖Φ̂_T(I−P)‖ · κ_T · ‖Ψ̂_T(I−P)‖,  κ_T = ‖(T⁻¹𝓛T⁻¹)⁻¹‖.

  With R = (I − Z_r)⁻¹ this bounds the log|H| error by −(D − r) log(1 − ω̄‖R‖), and each tr(H⁻¹S_i) error by ω̄‖R‖² tr(H0⁻¹S_i)/(1 − ω̄‖R‖).
- **Stop** at the first r where every bound is below its tolerance: 1/(2K) nats for log|H|, and the λ-gradient tolerance from the K-draw budget (§3.3). If no r < D meets them, use all D.
- **κ in production:** it comes from the EP fixed point's stability certificate, since 𝓛 is its linearization in (δm, δP).
- **[measured, dense]** At test points where both curvatures are positive definite:
  - the rule used 6–7 of 40 directions;
  - the bound was at or above the exact complement norm at every r;
  - the actual errors were 4–6 orders below the tolerance;
  - Jacobi scaling took κ from ~1e6 to 2–11.

  Limitation: the test points are not fixed points, and in 9 of 12 cases no test point was positive definite.

## 4. Variance-refresh trigger
Frozen marginals v_frz (the decoupled Stage 2 scheme) are refreshed only when one of these holds; none uses a chosen constant.
1. **Properness.** With v predicted by the JVP from τ_frz to τ, a frozen cavity stays proper iff its relative error e_j satisfies e_j < P_j/τ_j for τ_j > 0, or e_j > −P_j/|τ_j| for τ_j < 0. The check runs inside every EP sweep.
2. **Bias vs progress.** Staleness shifts the hyper-gradient by δg = Φᵀ𝓛⁻¹[0; −δ], with δ = v_frz − Σ_jj. That is (ii) with right-hand side −δ: one linear-response solve. Refresh when ½δgᵀ(−B)⁻¹δg ≥ δ_B.
3. **Certification** requires δ_B + ½δgᵀ(−B)⁻¹δg ≤ 1/(2K).

A dense comparison against always and never refreshing is not yet conclusive: the test EP itself hits improper cavities from the prior start.
