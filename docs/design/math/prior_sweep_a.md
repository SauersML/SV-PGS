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
- **Profiling the null space is not a fix.** Maximizing over the null-space coefficients instead of integrating them drives λ → ∞, the log-normal limit: ΔLPD −4.06 vs −0.71 per 1,000 variants on the BayesR truth [sim-only].

## 3. Penalty values must be exact squares

- Along a null-space direction the coefficients grow large, and xᵀSx computed from S = RᵀR can come out negative in floating point.
- Observed: the penalized objective was 5687 against a log-likelihood of 173.
- **The rule:** carry each penalty as its square-root factor R, and compute the value as |R x|². The gradient and Hessian may use S.
- The same rounding fault had produced a spurious λ → ∞ mode when the range was widened. With |R x|², t3 gives log λ = 1.93 at both ×1 and ×100 range.

## 4. Grid and range invariance of the chosen λ

*Smoke measurements: one replicate, 1,000-variant class, integrated evidence* [sim-only]:

| Truth | Penalty | log λ at spacing 0.5 | 0.25 | 0.125 | ×100 range |
|---|---|---|---|---|---|
| BayesR | D1+D2 | −1.04 / 3.08 | −1.03 / 3.12 | −1.01 / 3.15 | −1.32 / 2.49 |
| BayesR | D2 | 3.21 | 3.23 | 3.26 | 2.97 |
| BayesR | D3 | 3.33 | 1.35 | −18.4 | 3.58 |
| t3 | D1+D2 | −3.64 / 0.17 | −3.63 / 0.15 | −3.61 / 0.14 | −4.13 / 0.27 |
| t3 | D2 | 0.18 | 0.17 | 0.17 | 0.21 |
| t3 | D3 | 1.93 | 1.91 | −7.17 | 1.93 |

- **D1+D2 and D2 are invariant under refinement.** Held-out log predictive moved ≤ 0.05 nats per 1,000 variants.
- **Range sensitivity is small:** D1+D2 moved ≤ 0.5 nats per 1,000 variants at ×100 range, and D2 by 0.3.
- **D3 flips to a λ → 0 mode at the finest spacing.** That happens once the NPMLE support has ≤ 3 points: the unpenalized quadratic then absorbs the Occam factor of the supported components.
- The full sweep (5,000-variant classes, 10–20 replicates) replaces this table when it completes.

## 5. Pooling: a level-only pool cannot carry a heavier tail

- **Scenario:** an SV class shares the SNV class's bulk and large-effect scale, but has 30× the fraction of large effects (0.15 vs 0.005).
- **A level-only pool** gives every class the same shape, moved along log s. Smoke results (one replicate, a 200-variant SV class) [sim-only]:
  - fitted large-mode mass 0.072 against a true 0.15;
  - class held-out log predictive −17 nats per 1,000 variants vs the true prior, and MSE +9%.
- **log g_c = log g_0 + δ_c with a learned λ_c** recovered 0.138, and its class held-out log predictive was within replicate noise of the true prior's. So the class deviation must stay free in shape, not only in location.
- The candidate configurations (level-only, a shared or per-class deviation weight, the tilt penalized or free, a D3 deviation) go to the engine lane as benchmark candidates. They are not adopted on this evidence.
