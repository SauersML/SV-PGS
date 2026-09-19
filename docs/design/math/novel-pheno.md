# Phenotype measurement theory: from EHR records to the genetic likelihood

**Status (2026-09-19):**
- **Checked numerically (§10):**
  - the Tweedie identities and the closed-form disease likelihood;
  - Theorems 2, 3, 3′ and 5;
  - the calibration consequence of Theorem 1;
  - recovery of the disease-model parameters.
- **Derived but not yet checked:** the rest of Theorem 1, Theorem 4, the separation theorem, the shared-Gram bound and the identifiability of the transform.
- Evidence: all simulated numbers are **sim-only** under the evidence rule; they check the algebra. The one real-data check (§10, NHANES) tests two measurement components on within-exam replicates, not EHR occasions and not genetics.
- Adoption: the models below are a *specification*, gated on the in-workspace validation of §8, which needs the user's authorization. One change is correct by derivation alone (§5.2).
- Code: the prototype and its tests live with the novel-pheno lane's scratch files (`proto/`).

It replaces the 40 hand-set phenotype values listed in PHENOTYPES.md, "Rules that are not standards".

## 0. Setup

Person $i$ has genotypes $x_i$, non-genetic covariates $c_i$, and a latent trait
$L_i = c_i^\top\gamma + g_i + u_i$, with polygenic value $g_i = x_i^\top\beta$, non-genetic person effect $u_i$, and latent residual $v_i = g_i + u_i$, where $\sigma_L^2 = \sigma_g^2 + \sigma_u^2$.
The EHR produces the record $D_i$ (occasions, codes, drugs, procedures, labs, visit dates) through a measurement process with parameters $\theta_M$.

| | Assumption | Where it can fail |
|---|---|---|
| A1 | Non-differential measurement: $D_i \perp x_i \mid (L_i, c_i)$ | engagement genetics (§4.6) |
| A2 | Gaussian latent residual: $v_i \mid c_i \sim N(0,\sigma_L^2)$ (polygenic CLT for $g$) | heavy-tailed person effects |
| A3 | Small per-variant effects: $O(\beta_j^2)$ negligible | monogenic loci |
| A4 | Sequential ignorability: the treatment decision at occasion $j$ depends only on the record up to $j$ | decisions on unrecorded information (§3.5) |

## 1. The interface: one function per person

$$\ell_i(\eta) = \log \int p(D_i \mid L)\, N(L;\, \eta,\, \sigma_u^2)\, dL, \qquad \eta_i = c_i^\top\gamma + g_i.$$

The genetic likelihood is $\prod_i e^{\ell_i(\eta_i)}$. Every phenotyping choice either models $\ell_i$ or approximates it.

**Tweedie identities.**
$\ell_i'(\eta) = E[L-\eta \mid D_i]/\sigma_u^2$ and $-\ell_i''(\eta) = (1 - \mathrm{Var}(L\mid D_i)/\sigma_u^2)/\sigma_u^2$.
So the score is the posterior mean of the latent residual, and the curvature is the reliability $R_i = 1 - \mathrm{Var}(v_i\mid D_i)/\sigma_v^2$.

**Theorem 1 (optimal target).** Under A1–A2, among all targets $f(D_i)$ the genetic signal-to-noise $\mathrm{SNR}(f) = \mathrm{Cov}(f,g)^2/(\mathrm{Var} f\,\mathrm{Var}\, g)$ is maximized by $f^* = E[v_i \mid D_i]$. For every $f$,
$$\frac{\mathrm{SNR}(f)}{\mathrm{SNR}(f^*)} = \mathrm{corr}^2(f, f^*), \qquad \mathrm{corr}(f, v) = \mathrm{corr}(f, f^*)\,\mathrm{corr}(f^*, v).$$

*Proof.*
1. A1 gives $E[f\mid g,v] = E[f\mid v]$.
2. A2 gives $E[g\mid v] = (\sigma_g^2/\sigma_L^2)v$, so $\mathrm{Cov}(f,g) = (\sigma_g^2/\sigma_L^2)\,\mathrm{Cov}(f,v)$.
3. The tower rule gives $\mathrm{Cov}(f,v) = \mathrm{Cov}(f, f^*)$.
4. Apply Cauchy–Schwarz. ∎

**Consequences.**
- Every hand rule is some $f(D)$, and its efficiency, as a fraction of the effective sample size, is $\mathrm{corr}^2(f,f^*)$.
- That can be computed inside the workspace **without genotypes**; the result is used there and never leaves it.
- A rule that uses only $n_{\rm used}$ people has relative efficiency $n_{\rm used}\,\mathrm{corr}^2_{\rm used}(f,v)/(n\,\mathrm{corr}^2(f^*,v))$.

**Theorem 2 (every target estimates the same direction).** Under A1–A3, the regression of any $f(D)$ on standardized genotypes has coefficients $c_f\beta + O(\beta^2)$, with $c_f = E[\partial_v E[f\mid v]]\,\sigma_g^2/\sigma_L^2$ (Stein's lemma; Brillinger 1982).
- A phenotype definition changes only the efficiency $c_f^2/\mathrm{Var} f$, never the relative effects.
- That is why a Gaussian working likelihood with the shared Stage 0 Gram is valid for non-Gaussian targets.
- Outcome-dependent selection and A1 violations are the exceptions (§3.6, §4.6).

**The shared Gram is first-order efficient.**
- The target is $t_i = f^*_i/\bar R$, with precision $h_i \propto R_i$ and $h_i \perp x_i$ to first order.
- Then $X^\top HX = \bar h X^\top X + E$ with $E[E]=0$ and relative spectral size $O(\mathrm{CV}(h)\sqrt{p/n})$.
- So $\hat\beta = (\bar hX^\top X)^{-1}X^\top H z$ has the weighted estimator's variance up to $1+O(\mathrm{CV}\sqrt{p/n})$.
- By contrast, unweighted person means have efficiency $1/(E[h]E[1/h]) \le 1$.
- Stage 2's exact weighted PCG removes the $O(\sqrt{p_{\rm block}/n})$ error left in Stage 1.

**The current constructions are special cases.**
- With Gaussian occasions, $f^*$ is today's BLUP and $R_i$ its reliability.
- With perfect coding and dense visits, $f^*$ is today's age-of-onset liability target: $t(\text{onset})$ for a case, $-\phi(t)/\Phi(t)$ at censoring for a control.
- For a binary status, $\bar R = \phi(t)^2/(K(1-K))$, the Dempster–Lerner factor.

**Exact path.** For diseases, $\ell_i$ is closed form (§4.3). It can be a one-dimensional EP site, as logistic EP is. By Theorem 2 that gains efficiency only.

## 2. Separation theorem

Under A2, the phenotype-only marginal likelihood $\prod_i\int p(D_i\mid L)N(L; c_i^\top\gamma,\sigma_L^2)dL$ does not depend on how $\sigma_L^2$ splits into $\sigma_g^2 + \sigma_u^2$.
- So $\theta_M$ is learned from phenotypes alone, on **every participant with EHR data**, genotyped or not, before any genetics.
- $f^*$ uses $\sigma_u^2 = \sigma_L^2$, the first order in $h^2$. The error from that is $O(h^2\,\mathrm{CV}(R))$ in efficiency and zero in direction.

## 3. Quantitative traits: the per-occasion model

$$z_{ij} = h(y_{ij}) = f_{{\rm age},s}(a_{ij}) + \textstyle\sum_w f_w(\Delta t_{ijw}) + A_{ij}(\bar\tau + t_i) + T_i + e_{ij}.$$

**3.1 Transform** (replaces the per-trait log-vs-linear choice).
- $\log h'(y) = b_0 + b_1\log y + \delta(\log y)$, with $\int(\delta'')^2$ penalized by a weight learned by LAML.
- The penalty's null space is Box–Cox ($h'\propto y^{\lambda-1}$, $\lambda = b_1+1$, learned exactly). The Jacobian enters the likelihood.
- *Identifiability:* with ≥2 occasions for some people, under A2 and level-independent noise, $h$ is identified up to an affine map, because a non-affine reparametrization $k$ makes $\mathrm{Var}(k(z_1)-k(z_2)\mid m) \approx k'(m)^2\mathrm{Var}(z_1-z_2)$ depend on the level $m$.
- A Gaussian polygenic latent plus level-independent noise is what picks out the additive (genetic) scale.

**3.2 Occasion noise** (replaces the plausible ranges).
- $e \sim \int N(0,s)\,g_e(s)\,ds$ with a learned continuous mixing density: the effect prior's machinery (SPEC 131b205).
- The range runs from the rounding variance of the recorded precision (resolution²/12) to the largest squared within-person deviation.
- *Identifiability:* $\varphi_d = \varphi_e^2$ for within-person differences, and $\varphi_e > 0$ for any normal scale mixture, so $g_e$ is identified.
- A gross error is downweighted by $E[1/s\mid r]$ instead of being cut at a hand-set bound.
- An optional derived extension adds components for unit confusion by known conversion factors (spec values), which recovers such values instead of discarding them.

**3.3 Event-time perturbations** (replace the acute-care, pregnancy and clinical windows).
- Transient events: $f_w(\Delta t)$ is a smooth of the signed time since events of class $w$ (the code lists are kept), with learned smoothness. $f_w \to 0$ far from events defines the long-run level, and the largest lag comes from a boundary test.
- Chronic states (cirrhosis, kidney replacement, diabetes for HbA1c, hematologic malignancy) get a persistent shift with a learned mean and heterogeneity.

**3.4 Age** (replaces minimum ages 18 and 20). A smooth $f_{{\rm age},s}$ by sex, plus an age-varying noise scale. Occasions before adulthood count, with the precision the data give them.

**3.5 Treatment** (replaces LDL/0.7, SBP+15 and dropping treated BMI and heart rate).
- The term is $A_{ij}(\bar\tau + t_i)$, with $t_i\sim N(0,\sigma_\tau^2)$ and a ramp in the time since initiation.

**Theorem 3 (no regression to the mean).** Under A4,
$p(A_i \mid y_i, T_i, t_i) = p(A_i \mid \text{recorded } y)$, which is free of $(T,t,\theta)$ and so drops out of the joint likelihood (Rubin's ignorability).
- Joint ML over all occasions is therefore consistent.
- Pre/post comparisons, and targets built from untreated occasions only, are biased, because the triggering reading was selected for high noise.
- Treated-only people keep a target with reliability $R = \sigma_T^2/(\sigma_T^2+\sigma_\tau^2+\sigma_e^2/k)$.

**Theorem 3′ (the decision is part of the measurement process).** If the decision can also depend on the unrecorded level, A4 fails.
- Model the decision jointly: after each untreated occasion, $P(\text{start}) = \sigma(\alpha + b_y(y_{ij}-\mu) + b_T T_i)$, with $T_i$ integrated. The factor involves $T_i$, so it stays in the likelihood.
- $(b_y, b_T)$ are identified because $y$ varies within a person over time while $T$ is fixed.
- The joint ML is then consistent under either mechanism or both. In production the decision function is a learned smooth in $y$.
- Check T1 (§10) shows that each simpler estimator is unbiased under only one mechanism:
  - the ignorable joint ML when the decision is on the record;
  - a Mundlak term (the person's treated fraction) or within-person contrasts when it is on the level.

**3.6 HbA1c and diabetes.** Excluding people with diabetes selects on the trait's upper tail. That preserves the direction to first order but loses efficiency and calibration. Include them with the state and treatment terms instead.

**3.7 Output.** $f^*_i = E[T_i\mid D_i]$ and $R_i$, with the engine target $t_i = f^*_i/\bar R$. Under Gaussian occasions with no events and no treatment, this is exactly today's BLUP.

## 4. Diseases: the latent-onset per-visit channel model

**4.1 Onset.** Onset by age $a$ happens iff $L > t_s(a) = \Phi^{-1}(e^{-H_s(a)})$, so $P(\text{onset}\le a\mid\eta) = \Phi(\eta + q(a))$ with $q = -t$.
- $H_s$ is a monotone cumulative hazard: a piecewise-constant log-hazard with a second-difference penalty (whose null space is Gompertz) and a sex deviation, both with smoothness learned by LAML.
- $H$ saturates, so the lifetime risk is below 1.

**4.2 Channels.**
- At each visit (a distinct date with any record), channel $k$ is recorded with a person-level probability $p_{ik}\sim\mathrm{Beta}(\mu_{1k},\kappa_{1k})$ after onset and $\mathrm{Beta}(\mu_{0k},\kappa_{0k})$ before it.
- The channels are diagnoses, the case drug, procedures, lab occasions meeting the cited KDIGO/ADA standard, control-exclusion codes and drugs, and competing-class codes.
- Competing conditions (e.g. type 1 diabetes for type 2 diabetes) are a third latent class.

**4.3 Closed form.** The onset falls in one of the $V+1$ gaps between visit ages:
$$\ell_i(\eta) = \log\sum_{m=0}^{V} E_{im}\,\big[\Phi(\eta+q(a_{m+1})) - \Phi(\eta+q(a_m))\big],$$
where $E_{im}$ is the product over channels of beta-binomial terms for the pre-onset and post-onset runs.
- The cost is $O(KV)$ per person with prefix sums.
- Gap 0 is a prevalent case, so left truncation at EHR entry is exact.
- Given a gap, $v$ is a truncated standard normal, which makes $f^*_i$ (a mixture of truncated means) and $R_i = 1-\mathrm{Var}(v\mid D_i)$ closed form. So are all gradients.

**4.4 Identifiability.**
- Per state, ≥3 visits identify a two-component binomial mixture (Teicher: $n\ge 2k-1$), and varying visit counts identify each beta-binomial $(\mu,\kappa)$ through two moments.
- The onset distribution comes from the change-point locations, and the covariates from the shift they cause in onset.
- "≥2 diagnosis dates" is a threshold decision on the statistic the likelihood uses optimally.

**4.5 The hand rules.**

| Rule | Replacement |
|---|---|
| 2 diagnosis dates; 2 levothyroxine dates; excluding one unsupported date | channel rates and the onset posterior |
| EHR-depth floor (5 dates by entry + 365 days) | visits as exposures ($R_i \approx 0$ when sparse) |
| minimum control ages (COPD 40, prostate cancer 50) | $t_s(a)$ (young people without disease carry score ≈ 0) |
| minimum case age | an age-varying false-positive rate |
| ambiguous codes | a competing class |
| KDIGO/ADA thresholds | kept, as the cited definition of the lab channel |

**4.6 Visits and engagement.**
- **(a) Informative visits.** Conditioning on visit ages is exact when visits are ancillary for $L$. A post-onset rise in the visit rate ($e^\psi$) needs the visit-intensity factor: a Poisson process with gamma frailty, which integrates in closed form to a negative-binomial ratio per gap. *Not yet in the prototype.*
- **(b) Engagement genetics.** A target that depends on the visit statistics $V$ beyond disease carries their genetic component at first order.

**Theorem 4 (orthogonal target).** $f^*_\perp = f^* - E[f^*\mid V]$ removes all first-order genetic contamination through $V$, at relative efficiency $\mathrm{corr}^2(f^*_\perp, f^*)$. The current log1p(EHR-depth) covariate is a crude version of it. Whether to orthogonalize is decided by the in-workspace measurement V-E.

## 5. Covariates

**5.1 Age and sex.** They enter the measurement models as smooths with learned smoothness, replacing age, age² and age × female at the person level.

**5.2 Theorem 5 (no descendants of the outcome).** For $T = \beta x + u$ and $K = aT + v$, regressing $T$ on $(x, K)$ gives the coefficient $\beta\,\sigma_v^2/(a^2\sigma_u^2+\sigma_v^2)$ on $x$.
- So `log_occasion_count` is to be dropped: the occasion count is a descendant of the trait, and its role as precision is already in $R_i$. The contamination this could admit is second order for quantitative targets.
- log1p(EHR depth) stays for diseases until V-E, as a crude Theorem 4.

**5.3 Genetic covariates.** PCs and the cohort and pipeline indicators stay in the genetic model's FWL projection (MODEL.md §1).

## 6. Engine interface

- Per person and trait: $(t_i, R_i)$, and for diseases, optionally, the gap representation of $\ell_i$.
- The measurement fits run in-workspace on phenotypes only.
- Cost: $O(\sum_i K V_i)$ per likelihood evaluation for diseases, and $O(\sum_i J_i G Q)$ for occasions.

## 7. The 40 values and their replacements

| Values | Replacement |
|---|---|
| MIN_DISEASE_OCCURRENCES; case_medication_minimum_dates; the one-date exclusion | channel rates and the onset posterior (§4) |
| MIN_PRE_LANDMARK_CONDITION_DATES; EHR_DEPTH_LANDMARK_DAYS | visits as exposures; orthogonalization if measured (§4.6) |
| minimum_control_age_years; minimum_case_age_years | the onset hazard; an age-varying false-positive rate |
| plausible_range | the scale-mixture occasion noise (§3.2) |
| ACUTE_CARE_WINDOW_DAYS; the pregnancy windows; the clinical window days | event-time smooths and chronic shifts (§3.3) |
| ADULT_AGE_YEARS; height 20 | an age smooth and age-varying noise (§3.4) |
| LDL/0.7; SBP+15; excluding treated BMI and HR | the random treatment effect (§3.5) |
| log_scale; log_offset | the learned transform (§3.1) |
| age², age × female, log(occasion count), log1p(EHR depth) | smooths; descendants dropped (§5) |
| Henderson III moments | the joint ML (its Gaussian special case) |

## 8. Validation inside the workspace

It runs inside the AoU workspace only, with the user's authorization, and its results are used there; no value is reported outside the workspace.

| Test | What it reports |
|---|---|
| V-A | Channel rates, κ and hazard curves; cross-tabulation of the rule classification against the posterior case probability |
| V-B | corr²(f_rule, f*) for every current target (Theorem 1, no genotypes) |
| V-C | Posterior predictive checks of the coded-visit counts |
| V-D, decisive | Held-out PGS R² paired over folds, f* vs current targets; z-scores at known loci (LDL: LDLR, APOB, PCSK9; T2D: TCF7L2) |
| V-E | Association of each target's PGS with visit count (the orthogonalization decision) |
| V-F | τ̄ and σ_τ against external averages (CTT's ~30% LDL reduction, ~15 mmHg SBP), as sanity checks |
| V-G | The transform, the noise mixture and the event-time curves |

## 9. Failure modes and related work

- **Failure modes:** A1 through engagement (§4.6b); non-ignorable treatment (§3.5); informative visits (§4.6a); competing risks and death (censoring assumed non-informative).
- **Misspecified channels** make $f^*$ suboptimal but still direction-consistent (Theorem 2).
- **Related work:** PheNorm and MAP (latent EHR phenotypes from code counts normalized by utilization); LT-FH and LT-FH++ (conditional liability means); BLUP of repeated measures; Box–Cox random-effects models; Rubin's ignorability.
- **Novelty is not established:** no literature search was run in this lane.

## 10. Checks
**Unit tests.** `proto/test_proto.py`, 7/7 pass on MSI (venv-cpu):
- the stable log interval probability;
- the gap probabilities sum to 1;
- the score and reliability equal the first and second η-derivatives of ℓ_i (Richardson-bounded finite differences);
- the penalized gradient is exact;
- the treated-only reliability has its closed form;
- the Box–Cox Jacobian;
- the predictive density integrates to 1.

**Theorem 2, direction (B1, sim-only).** A rule target (case iff liability > its 95th percentile) regressed on 200 standardized genotypes with n = 200,000:
- The correlation of the estimates with the true effects is 0.9937, against 0.9950 predicted from sampling noise alone.
- The slope is 0.0995, against the Stein slope φ(t/σ_L)/σ_L = 0.0960. The 3.6% scale gap is the O(β²) term from rare standardized genotypes (A3).

**Theorem 5 (C1, sim-only).** The coefficient adjusted for a descendant of the trait is 0.1085 ± 0.0009, against the formula's 0.108 (unadjusted truth 0.3).

**Theorems 3 and 3′ (T1, T3, sim-only).** True τ̄ = −0.8.
- T1 used deterministic decisions: 40 × 3,000 people, 6 occasions.
- T3 used logistic decisions: 20 × 3,000 people.

| Decision depends on | Joint ML, ignorable | Mundlak | Pre/post | Joint ML with the decision model |
|---|---|---|---|---|
| T1: the recorded value (threshold) | **−0.801 ± 0.004** | −1.089 | −1.121 | — |
| T1: the unrecorded level (threshold) | −0.169 | **−0.793 ± 0.006** | **−0.793** | — |
| T3: the recorded value (logistic 1.5) | **−0.797 ± 0.004** | −0.966 | — | **−0.800 ± 0.004** |
| T3: the unrecorded level (logistic 1.5) | −0.628 | **−0.801 ± 0.005** | — | **−0.801 ± 0.005** |
| T3: both (1.0, 1.0) | −0.700 | −0.907 | — | **−0.802 ± 0.004** |

- Only the joint decision model is unbiased under every mechanism.
- It recovers the decision coefficients: (1.48, 0.03), (0.00, 1.50) and (0.98, 1.03) against (1.5, 0), (0, 1.5) and (1, 1).
- Its quadrature has converged: order 40 vs 80 on the same data changes τ̂ by ≤ 0.001, a fifth of the SE.

**T2.** The treated-only reliability equals the closed form to machine precision for k = 1, 3, 6: 0.6329, 0.7979, 0.8535.

**The real-data check, NHANES 2017–March 2020 (P_BPXO + P_DEMO; SHA256SUMS with the data).**
- *What it is:* three oscillometric readings 60 s apart; train on odd SEQN, test on even SEQN.
- *The metric:* the held-out mean log predictive density of each reading given the other two, on the original scale, as paired differences against the current rules (A0: linear, Gaussian, plausible range, age ≥ 18). The mean model is identical across arms.
- **This is [real] for the measurement component only: within-exam noise, not EHR occasions, and not genetics.**

| Arm | SBP (n_test = 4,003) | Pulse (n_test = 3,649) |
|---|---|---|
| A0b log scale, Gaussian | +0.006 ± 0.012 | +0.030 ± 0.009 |
| A1 learned Box–Cox, Gaussian | +0.006 ± 0.011 (λ̂ = 0.03) | +0.030 ± 0.008 (λ̂ = 0.07) |
| A2 learned Box–Cox, learned scale-mixture noise | **+0.110 ± 0.022** (λ̂ = −0.31) | **+0.191 ± 0.022** (λ̂ = −0.16) |
| A3 as A2, no range or age exclusion | **+0.109 ± 0.022** | **+0.181 ± 0.022** |

- The gain comes from the learned noise density, not from the transform.
- Fitting without the plausible-range and minimum-age exclusions (A3) loses nothing measurable against A2.
- The current plausible ranges bind on only 5 of about 24,000 SBP readings and none for pulse in these protocol-QC'd data. EHR data will stress them far more; V-G tests that.

**The disease model, V2–V5:**

- **V2a, the closed form against the Stieltjes integral.** For 40 people, the gap-sum ℓ_i(η) was compared with the integral over onset age, whose record counts come from an independent search on the visit ages. The maximum error falls from 0.049 to 0.012 to 0.0015 as the age grid refines from 2,000 to 8,000 to 32,000 cells. It converges to the closed form at at least first order in the cell width.
- **V2b, Tweedie.** The closed-form score and reliability match E[v | D] and 1 − Var(v | D), computed by brute-force quadrature over v, to 3.8e-4 and 4.0e-4. That is the resolution of the v-grid (spacing 5e-4).
- **V2c, the gradient.** The exact penalized gradient matches central differences to a relative 6e-9.
- **V3, the perfect-coding limit.** With per-visit false-positive rate 1e-9, true-positive rate 1 − 1e-9 and dense visits, the score converges to the current age-of-onset liability target.
  - Cases: t(first record), with the maximum error 0.016, 0.0030 and 0.00065 at visit spacings of 0.5, 0.1 and 0.02 years, i.e. linear in the spacing.
  - Controls: −φ(t)/Φ(t) at the last visit, exact to 1e-11.
  - A first run drew onsets beyond the age where its capped hazard saturates, where the model puts no onset mass, so those cases could not match. The rerun uses a hazard that stays positive over every simulated onset age.
- **V5, recovery by marginal likelihood (sim-only; checks identifiability).** 3,000 simulated people, three channels, λ learned by LAML, the Hessian positive definite at the optimum.
  - The false-positive rates come out as 0.0096, 0.0197 and 0.0049 (truth 0.01, 0.02, 0.005), and the true-positive rates as 0.501, 0.394 and 0.191 (truth 0.5, 0.4, 0.2).
  - The concentrations are 19–40 (truth 20–50) before onset and 3.2–6.4 (truth 3–5) after it.
  - The sex shift is 0.262 (truth 0.3).
  - The maximum cumulative-incidence error is 0.043 with 13 knots and 0.025 with 25.
  - **The knot refinement has not converged yet:** from 13 to 25 knots the largest per-person target change is 0.25, against a target SD of 1.42. A 25 → 49 check is still owed before the hazard grid can be called converged.
- **V4, the current rules on the same simulated data (sim-only, not evidence).**
  - The current rules keep 2,399 of 3,000 people with corr(target, v) = 0.553, an effective N of 734.
  - The fitted score target keeps all 3,000 with corr = 0.699, an effective N of 1,466.
  - As Theorem 1 requires of a calibrated posterior mean, corr²(f*, v) = 0.489 matches R̄ = 0.494.
- **Not built in `proto/`:** the visit-intensity factor (§4.6a), the competing class, the event-time smooths, the smooth transform deviation δ, and a nonparametric continuous noise density (the prototype uses a converged NPMLE grid).

