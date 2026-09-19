# Sweep A: the learned mixing density on exact normal means

This note checks the effect prior of MODEL.md §3 where every posterior is exact: an orthogonal design, so each variant's posterior is one-dimensional and there is no EP error.
- **What it is for:** the math. It checks that the continuous learned density nests the fixed families, that the fit is invariant to its quadrature (grid spacing and range), and that each candidate model term nests its limits.
- **What it is not:** evidence of accuracy. Per the evidence rule, every accuracy number here is `[sim-only]`. Which prior terms go live is decided on bench-real and bench-sim.
- **Code:** self-contained numpy/scipy, in the prior lane's MSI directory `/projects/standard/hsiehph/sauer354/svpgs-team/prior/`:
  - `nm_core.py`: exact posteriors, held-out log predictive;
  - `nm_learned.py`: the candidate and the NPMLE;
  - `nm_families.py`: TPB, BayesR, true priors, pseudo-true fits;
  - `nm_scenarios.py`, `sweep_a.py`, `aggregate.py`.
  It shares no code with `sv_pgs` or with the EP reference.

## 1. Setup

- **Measurement model:** β̂_j = r_j β_j + e_j, e_j ~ N(0, s²), s² = (1 − h²)/n. A held-out replicate β̂′_j uses the same effect and fresh noise.
- **Metrics:**
  - the population R² of the score Σ D_j m_j, with m_j = E[r_j β_j | β̂];
  - the calibration slope and the posterior-mean MSE;
  - the held-out log predictive Σ_j log p(β̂′_j | β̂_j), which is exact given the fitted prior.
  All are reported as paired per-replicate differences against the true prior.
- **Fits:**
  - **The true prior.** Atoms are exact; continuous parts are exact CDF cell masses on a 0.02 log-variance grid.
  - **The candidate.** log g(t), t = log s, carried on a uniform grid over a data-derived range, outside which no variant's likelihood distinguishes variances to 1%. The roughness penalties are continuum integrals ∫(δ^(m))² dt, so a weight means the same thing at every spacing. Penalty weights maximize the Laplace evidence directly.
  - **The NPMLE on the same grid:** a tuning-free benchmark; its estimate is discrete.
  - **TPB and BayesR,** each fit by exact EB and at population-optimal ("pseudo-true") parameters.
- **Truths:** BayesR-like atoms, TPB(½, 3/2), spike-and-slab, sparse plus polygenic, t3, and Gaussian. Also reliability offsets, an SNV + TR class pair, small classes of 200 and 2,000 variants, the pooling arms, a frequency arm and a shape-vs-scale arm.

## 2. The null space of an order-m penalty is improper under a flat prior

- **The identity.** A D_m roughness penalty leaves log g ∈ span{1, t, …, t^{m−1}} unpenalized. For m ≥ 2 that span contains a collapse ray. Along log g = −c(t − t_lo)² with c → ∞ (m = 3), or along a slope → ∞ (m = 2), the density concentrates on the grid's lower end: V ≈ 0, a point mass at zero effect.
- **Consequence.** Along the ray the likelihood tends to the null model's, a positive constant, not to 0. So the integral over the null-space coefficients under a flat prior diverges, and the Laplace/REML evidence goes to +∞: the curvature in that direction vanishes and −½ log det rewards it.
- **Where it bites.** It is the rule, not a corner case, for weak classes such as rare SV types, low-r² TR loci and small classes.
- **The D1 + D2 pair has no null space** in sum-to-zero coordinates, so its evidence is proper.
- **Measured** (snvtr TR class: 100 variants, r² 0.2–0.6, t3 truth) [sim-only]:
  - D3's integrated evidence reached 3e302, with 100% of the density on the bottom node;
  - posterior-mean MSE +348% vs the true prior, calibration slope 4528;
  - even at moderate λ the integrated evidence prefers the collapsed fit, 293.8 vs 293.0 smooth.
  - Part of that blow-up was the rounding fault of §3. With it fixed, D3 still loses there: ΔLPD −11.8 nats per 1,000 variants, against −7.0 for D1+D2.
- **The fix is to profile the null space in the Schur form** (ruling; ep_eb.md point 3): V = F + ½ log|S|₊ − ½ log|B + S| + ½ log|Nᵀ(B + S)N|. The collapse ray then stays finite: "the whole class below resolution" becomes an ordinary candidate that competes on evidence.
- **A conditional form is not that fix.** Integrating only the penalty's range at fixed null coordinates (−½ log|Qᵀ(B + S)Q|) drove λ to its bound on the BayesR truth: ΔLPD −4.06 vs −0.71 per 1,000 variants [sim-only]. An earlier version of this note wrongly called that result "profiling".

## 3. Penalty values must be exact squares

- Along a null-space direction the coefficients grow large, and xᵀSx computed from S = RᵀR can come out negative in floating point.
- Observed: the penalized objective was 5687 against a log-likelihood of 173.
- **The rule:** carry each penalty as its square-root factor R, and compute the value as |R x|². The gradient and Hessian may use S.
- The same rounding fault had produced a spurious λ → ∞ mode when the range was widened. With |R x|², t3 gives log λ = 1.93 at both ×1 and ×100 range.

## 4. Grid and range invariance: the evidence form decides it

**Setup:** BayesR-type truth, 1,000 variants, one replicate, D3. The evidence is profiled along log λ ∈ [−20, 5], and the density is refit at each form's optimum [sim-only; numerical-property check].

| Evidence form | h = 0.5 | 0.25 | 0.125 | 0.1 |
|---|---|---|---|---|
| Integrated (flat prior on the null space): optimum log λ | 3.33 | 1.35 | −18.1 | −19.7 |
| Integrated: V at the optimum | 3449.8 | 3454.2 | 3470.0 | 3485.3 |
| Schur (null space profiled): optimum log λ | 3.55 | 2.04 | 1.03 | 0.70 |
| Schur: V at the optimum | 3448.96 | 3449.79 | 3449.95 | 3449.99 |
| Schur: ΔLPD vs the true prior, nats per 1,000 variants | −0.80 | −0.35 | −0.19 | −0.14 |
| Schur: ‖m_h − m_0.5‖ / ‖m_0.5‖ | — | 0.31% | 0.43% | 0.46% |

- **The integrated form is grid-dependent.** Its level grows with the number of grid points, because a flat-prior integral over the null space has no limit under refinement. So its optimum runs to λ → 0 on fine grids. That form is banned.
- **The Schur form converges.** Its optimum drifts along a flat ridge (within 0.8 nats over log λ ∈ [−3, 5] at h = 0.125), its value converges, and the predictions settle. min eig(B + S) falls to 1e-12–1e-13 only as λ → 0.
- **D1 + D2 has no null space, so its evidence forms coincide.** Smoke measurements, one replicate:

| Truth | log λ₁ / log λ₂ at h = 0.5 | 0.25 | 0.125 | ×100 range |
|---|---|---|---|---|
| BayesR | −1.04 / 3.08 | −1.03 / 3.12 | −1.01 / 3.15 | −1.32 / 2.49 |
| t3 | −3.64 / 0.17 | −3.63 / 0.15 | −3.61 / 0.14 | −4.13 / 0.27 |

  Its held-out log predictive moved ≤ 0.05 nats per 1,000 variants under refinement, and ≤ 0.5 at ×100 range.

## 5. Pooling: a level-only pool cannot carry a heavier tail

- **Scenario:** an SV class shares the SNV class's bulk and large-effect scale, but has 30× the fraction of large effects (0.15 vs 0.005).
- **A level-only pool** gives every class the same shape, moved along log s. Smoke results (one replicate, a 200-variant SV class) [sim-only]:
  - fitted large-mode mass 0.072 against a true 0.15;
  - class held-out log predictive −17 nats per 1,000 variants vs the true prior, and MSE +9%.
- **log g_c = log g_0 + δ_c with a learned λ_c** recovered 0.138, and its class held-out log predictive was within replicate noise of the true prior's. So the class deviation must stay free in shape, not only in location.
- The candidate configurations (level-only, a shared or per-class deviation weight, the tilt penalized or free, a D3 deviation) go to the engine lane as benchmark candidates. They are not adopted on this evidence.
