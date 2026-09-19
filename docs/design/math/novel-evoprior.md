# The evolutionary effect prior (novel-evoprior)

**Status:**
- the theory is complete, and every identity is checked numerically;
- all numbers here are **sim-only** (EVIDENCE_RULE): they check the math and code, not accuracy;
- adoption is pending bench-real and bench-sim.

**Relation to scale_model.md §5.** That document already uses the frequency-conditioned prior derived here, with its split between the column's H^loc and the selection-equilibrium H^sel. This document gives:
- the exactness proof;
- the closed form of the frequency exponent;
- the results for SV length, TR loci, traits and the SFS;
- identifiability;
- the forward-simulation checks.

Prototype code: kept with the novel-evoprior lane, not in the repo; its fit/score entry point is submitted to the benchmarks.

## 1. Model

- **A1: mutation.** A new mutation of class c carries a pleiotropic effect a ∈ ℝⁿ on the n trait axes under selection. a | s ~ N(0, sIₙ), with s ~ g_mut,c(· | L) a continuous density; L is the variant's length.
- **A2: selection.** Gaussian stabilizing selection with the mean at the optimum, so an allele of derived frequency x drifts by M(x) = −(S/4N)x(1−x)(1−2x), with S = κ‖a‖² and κ = 2N/V_s.
- **A3: drift.** Wright–Fisher in the diffusion limit, with sites in linkage equilibrium.
- **A4: equilibrium.** Constant N.
- **A5: focal traits.** β_t = σ_t ℓ_tᵀa, with ‖ℓ_t‖ = 1.

## 2. Theorem 1: the folded SFS is exact

The scale density ψ(x) = e^{S x(1−x)} is symmetric. With Σ(x) = ∫₀ˣψ, the Green function of the absorbed diffusion from p = 1/2N gives the unfolded density:

t(x) = θ (Σ(1) − Σ(x)) / (Σ(1) ψ(x) x(1−x)).

At S = 0 this is θ/x. Symmetry gives Σ(1) − Σ(1−m) = Σ(m), so

**t(m) + t(1−m) = θ e^{−S m(1−m)} / (m(1−m))**, exactly, for every minor-allele frequency m.

No strong-selection approximation is needed. The result needs ψ to be symmetric, so a directional component breaks it.

## 3. Theorem 2: the frequency-conditioned prior

Condition the segregating-site intensity on m and complete the square in a (H = 2m(1−m)):

- **b_t | s, H ~ N(0, σ_t² H s/(1 + κHs)),**
- **s | H ~ g_mut(s)(1 + κHs)^{−n/2}/Z(H),**
- **Z(H) = ∫ g_mut(s)(1+κHs)^{−n/2} ds.**

This is again a Gaussian scale mixture, i.e. within the SPEC family, but both its variance map and its mixing density depend on H.
- **The ceiling:** v < σ_t²/κ for every H and s. Frequency conditioning truncates the small-effect bulk, while the large-effect tail stays pinned at the ceiling. So a scale-only f(log H) is misspecified whenever effects reach κHs ≳ 1.
- **Anisotropy** (K = diag(κ_i)): the tilt is Π_i (1+κ_i Hs)^{−1/2}, and the variance map uses the focal axis's κ.

## 4. Consequences

**4.1 The frequency exponent in closed form.**
- In the many-axes limit (n → ∞, κ → 0, λ = nκ/2 fixed), the tilt becomes e^{−λHs} and v → σ²Hs. Then
  S(H) := d log E[v | H]/d log H − 1 = −λH · Var_H(s)/E_H(s) ≤ 0.
- If g_mut is regularly varying with tail index α ∈ (0,1) (g ∼ s^{−1−α}), then E[v | H] ∝ H^α, so S → α − 1.
- So S is not a free constant. It is set by the tail of the mutational effect distribution. Heavier tails, as expected for large SVs, push S toward −1.

**4.2 SV length as a subordinator.**
- If a variant of length L hits functional elements as a Poisson process with iid element variances, then s(L) is a subordinator X_L with Laplace exponent Φ_c(t) = d_c t + ∫(1 − e^{−ts}) ν_c(ds).
- One continuous Lévy measure ν_c generates g_mut,c(· | L) for every length:
  - E[s | L] ∝ L;
  - the relative dispersion falls as L^{−1/2}.
- A scale-only length smooth, which makes g(· | L) a scaled copy of one law, is exact **iff** X is strictly stable (Φ ∝ t^γ). The present length smooth is therefore a power-law special case. Learning ν_c instead is a later candidate.

**4.3 TR loci: the diversity-deficit tilt.**
- For a multi-allelic length locus with an additive effect θ per unit length, W̄^{2N} = exp(−κθ² Var_ℓ(x)).
- Wright's formula, which is exact for parent-independent mutation and an approximation for stepwise mutation, gives
  π(θ | x) ∝ g_mut(θ) e^{−κθ²Var_ℓ(x)} / Z_μ(θ),  Z_μ(θ) = E_{π₀^μ}[e^{−κθ²Var_ℓ}].
- When the neutral Var_ℓ concentrates near V₀(μ), the tilt is e^{−κθ²(Var_ℓ − V₀(μ))}. Length diversity below what the locus's mutation rate predicts is evidence of a larger per-unit-length effect.
- This gives TR mutation-rate features a derived role: they predict V₀(μ).

**4.4 Pooling across traits.**
- Under A5, (b_1..b_T) | s, H ~ N(0, (Hs/(1+κHs)) D_σ R D_σ), with R = LᵀL constant across variants.
- All traits share g_mut, κ and n, and differ only in σ_t². Each variant has one shared continuous scale.
- So the mixing density is pooled across traits by the model itself. Anisotropy enters as learned, penalized trait deviations.

**4.5 The SFS channel.**
- Relative to a neutral class with the same ascertainment, a class's frequency spectrum is n_c(H)/n_0(H) = (θ_c/θ_0) · Z_c(H). Z_c is a generalized Stieltjes transform of g_mut,c (the n = 2 case is the Stieltjes transform itself), and it is injective.
- So the frequencies of all variants in a class, associated or not, are a likelihood for that class's effect prior. That information grows with the class size, not with the number of detectable effects, which matters for rare SV classes.
- **Confounders:**
  - non-equilibrium demography;
  - class-dependent discovery ascertainment;
  - trait-independent selection;
  - linked selection.
- Not proposed until its robustness is measured on public SV spectra.

## 5. Theorem 5: identifiability

- **Claim:** given the laws of v at a continuum of H, (f, κ, n, g_mut) are identified up to the one-parameter group (c, κ, g) → (λc, λκ, g(·λ)λ). A single location pin breaks that group; it is the same direction as the current design's level-vs-g-location confounding. Use E_g[s] = 1, or no free level; ∫ log s g = 0 is ill-conditioned at finite data because g's sub-noise part is unidentified (math-scale).
- **Proof sketch:**
  1. The essential supremum of v_H gives c(H)/κ.
  2. In W = κHs the tilt cancels in density ratios across H, so ℓ(u) = log g(e^u/κ) is identified up to a linear term.
  3. The log-slopes of p_H(w) at w → 0 and w → ∞ give that linear term and n.
  4. The pin gives κ.
- **At finite data** the well-identified combination is λ = nκ/2, which governs the unsaturated bulk. κ separately is identified only through the saturated tail. Measured on data from the model's own law (sim-only; κ = 200, n = 6, λ = 600):
  - over 6 fits at p = 30k and 120k, λ̂ was within ±10%;
  - κ̂ ranged over 128–264 and n̂ over 4.1–10.0, along the λ ridge, with no narrowing from 30k to 120k.
  - So λ is identified at these sizes and κ is not. The engine should report λ and treat κ, n as one weakly identified direction; predictions depend on λ.

## 6. Quadrature and range (SPEC 131b205)

- **Lower end:** moving mass from v < v_lo to v_lo changes the log evidence by at most (v_lo/2σ²) Σ_j |z_j² − 1|. So v_lo = 2σ²·tol/Σ_j |z_j² − 1|.
- **Upper end:** each variant's likelihood decreases in v beyond b̂_j², so the unpenalized optimum puts no mass above v_hi = max_j b̂_j².
- **The log-s grid** covers the image of [v_lo, v_hi], and is widened until its edge mass is below tol.

## 7. Checks (all sim-only)

| Claim | Check | Result |
|---|---|---|
| Theorem 1, the exact folded SFS | forward WF at N = 1000, S = 0/30/150, lattice sums | within 1.6% at S = 0; within 2–4% at S = 30; within 11% in the far tail at S = 150 (density about e^{−11}), which is the O(s) WF-vs-diffusion correction at s = 0.04 |
| Theorem 2, the conditional Gaussian law | Monte Carlo | weight and variance within 1%, kurtosis 3 ± 0.05 |
| Theorem 2 in a forward pleiotropic WF population | E[a₁² \| s, m] by decile of predicted variance | every decile within 15% |
| §4.1, S = α − 1 | quadrature | slope within 0.05 of α |
| §4.5, class SFS = generalized Stieltjes transform | forward WF, SNV class with a mixture DFE | within 4/√E + 5% in every populated bin |
| §4.3, Wright tilt (exact case: parent-independent recurrent mutation) | forward WF, 40k loci, 4Nμ = 0.5, S = 0 and 20 | within 4√E + 3% in every bin |
| Class-assembled Hessian = full Hessian | JAX | exact to 1e-8 |
| Estimator recovery | data from M2′, p = 30k and 120k, 3 seeds each | λ within ±10%; κ, n spread up to 2× along the λ ridge, not shrinking with p |

## 8. What would change in the one model

- **The derived prior:**
  - replace "s ~ g_c" by "s | H ~ g_mut,c(s)(1 + κHs)^{−n/2}/Z_c(H)";
  - replace "v = u s" by "v = u H s/(1 + κHs)";
  - add two continuous global parameters, κ and n.
- **Nothing is hand-chosen:** everything is learned by type-II ML. The learned residual smooth f_c(log H) is kept, with a penalty null space of power laws, so the current design is nested.
- **Engine cost:** per variant it is still a Gaussian scale mixture on the quadrature, with log-weights and variances from `prior_components()`. The per-variant normalizer Z_c(H_j) costs O(pK) per hyper step, the same order as the current M-step.
