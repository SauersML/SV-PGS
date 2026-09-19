# The scale model: derivation and checks

Scope: the per-variant prior scale of MODEL.md §3,

$$\beta_j \mid c \sim \int N(0,\,u_j s)\,g_c(s)\,ds,\qquad \log u_j = \ell_c + \log r^2_j + f(d_j),$$

derived from the generative model rather than asserted. Every claim below has a numerical check. The scripts are in lane math-scale's scratch directory, with copies on MSI under `/scratch.global/sauer354/svpgs-team/math-scale/`, and their outputs are quoted.

**Evidence.**
- Every number from a check here is `[sim-only]`: it verifies an identity or a derivation, not an accuracy gain (MODEL.md, evidence tags).
- Literature numbers are marked as coming via lit-pgs or lit-pool; their sources and full-text or abstract tags are in those lanes' notes.

**Related.**
- [ep_eb.md](ep_eb.md): the EP-EB objective and certificate.
- [novel-measure.md](novel-measure.md): measurement theory, Berkson and the Rao-Blackwellised column.
- mixing_density.md: range, penalty order, the λ→∞ limit.

**Notation.**
- $G_j$ is the true genotype (allele count, or copy number, or for a TR locus the total length $L$).
- $D_j$ is the stored column: DS after background removal, and D* where a κ exists.
- $\tilde x$ is $x$ standardized to unit variance. $\gamma_j$ is the true effect per SD of $G_j$, and $b_j$ the coefficient per SD of $D_j$.
- $r_j=\mathrm{corr}(D_j,G_j)$. A column is **calibrated (Berkson)** when $\mathrm{Cov}(G_j,D_j)=\mathrm{Var}(D_j)$, i.e. the linear regression of $G_j$ on $D_j$ has slope 1.

## 1. The r² offset has coefficient exactly 1, marginally

**Claim.** Suppose the measurement is non-differential: $D_j \perp y \mid G$. Then, for any column construction, the marginal standardized coefficient is $b_j=r_j\gamma_j$. So $\mathrm{Var}(b_j)=r_j^2\,\mathrm{Var}(\gamma_j)$, and $\log r_j^2$ enters $\log u_j$ with coefficient 1.

**Proof.** Take $y=\gamma_j\tilde G_j+\varepsilon$ with $\varepsilon\perp D_j$. Then
$$b_j=\frac{\mathrm{Cov}(y,D_j)}{\mathrm{sd}(D_j)}=\gamma_j\frac{\mathrm{Cov}(G_j,D_j)}{\mathrm{sd}(G_j)\,\mathrm{sd}(D_j)}=\gamma_j r_j.\ \square$$

What this means for our column construction:
- **Affine maps are free.** μ, κ and the empirical-SD standardization drop out, because $r_j$ is invariant to affine maps of $D_j$. The D* recalibration, $\mu+\kappa(DS-\mu)$, changes the per-allele slope, not $b_j$.
- **Non-affine maps are not.** Background removal and any monotone shape $h$ change $r_j$. So the reliability model must target corr²(**stored** D, G), after background removal and D*. HANDOFF item 4 already specifies this, and it is necessary, not a convention.
- **Empirical-SD standardization** adds only $O(n^{-1/2})$ noise.
- **An equivalent parametrization without the offset.** Standardize a *calibrated* column by $\mathrm{sd}(G)=\mathrm{sd}(D)/r$ instead of $\mathrm{sd}(D)$. Then the coefficient is $\gamma$ itself, and the $\log r^2$ offset disappears. novel-measure's Result 4‴ checks this: $\mathrm{sd}(G)=\sqrt{\mathrm{Var}(X)+\overline{\mathrm{Var}(G\mid M)}}$ matches the offset form to 4 decimals in 60 paired replicates. The choice between the two forms is bookkeeping; mixing them is an error.

**Check (`check_r2.py` §1a, n = 2·10⁶ per cell).**
- Setup: 9 settings of allele frequency p ∈ {0.02, 0.1, 0.3} and read noise, each with 4 column types:
  - classical $D=G+e$;
  - exact posterior mean;
  - a posterior draw;
  - a draw recalibrated with `calibrated_scale`.
- $b/(r\gamma)$ = 1.000 ± 0.01 in all 36 cells. The largest deviation, 0.977–0.992, comes where r² < 0.06 and the Monte Carlo error is largest.
- The **per-allele** slope differs by column type:
  - it is 1.00 for posterior means and recalibrated draws;
  - it equals r² for classical error;
  - it equals r for raw draws, which is κ = √r², matching MODEL §2.

## 2. In the joint model the offset is exact only for joint posterior means

The prior is placed on the coefficients of the **joint** regression of y on all columns. With the linear predictor of G from D,
$$\mathbb E_{\rm lin}[\tilde G\mid \tilde D]=\tilde D A,\qquad A=\Sigma_D^{-1}\Sigma_{DG},$$
the D-space coefficients are $b=A\gamma$.

**Claim.** $A=\mathrm{diag}(r)$ exactly, so the independent prior $\mathrm{Var}(b_j)=r_j^2\mathrm{Var}(\gamma_j)$ is exact, **iff** every column's error is uncorrelated with every other column. That holds when all columns are posterior means of their genotypes given one common data set: $D_j=\mathbb E[G_j\mid\mathcal F]$ with all $D_k$ $\mathcal F$-measurable. Then $G_j-D_j\perp D_k$ for all $k$ (tower property), so $\Sigma_{DG}=\Sigma_D$ in per-allele units.

The same result, in its general non-linear form, is novel-measure's Result 2 in `scratchpad/team/novel-measure/THEORY.md`: the joint regression on $X=\mathbb E[G\mid M]$ is conditionally unbiased for the true effects. Those numbers and these agree.

**Leakage otherwise.** For a draw-type SV column with a clean tag SNP (haplotype correlation ρ), the tag carries information about $G_{SV}$ that $D^*_{SV}$ lacks:
- $\mathrm{Cov}(D_{SNP},G_{SV}) > \mathrm{Cov}(D_{SNP},D^*_{SV})$;
- so $A_{SNP,SV}\neq0$, and part of the SV effect moves onto the SNP.

**Check (`check_r2.py` §1b).**
- Setup: a two-locus haplotype model; the SV is causal and the SNP genotype is exact; n = 2·10⁶.
- The formula for $A$ matches the fitted joint regression to ±0.003 in every row.

| haplotype ρ | SV column | r² | SV's own coefficient / (rγ) | leaked to SNP / γ |
|---|---|---|---|---|
| 0.5 | classical error | 0.34 | 0.82 | +0.36 |
| 0.5 | posterior mean, own data only | 0.40 | 0.83 | +0.33 |
| 0.5 | **posterior mean, joint data** | 0.51 | **1.00** | **−0.003** |
| 0.5 | draw, joint data, recalibrated | 0.26 | 0.69 | +0.32 |
| 0.8 | classical error | 0.34 | 0.45 | +0.67 |
| 0.8 | posterior mean, own data only | 0.39 | 0.47 | +0.65 |
| 0.8 | **posterior mean, joint data** | 0.73 | **1.00** | **−0.001** |
| 0.8 | draw, joint data, recalibrated | 0.53 | 0.35 | +0.59 |
| 0.95 | draw, joint data, recalibrated | 0.84 | 0.12 | +0.85 |

**Consequences for SV-PGS.** Imputed SV/TR dosages behave as confident draws with κ ≈ √r² (MODEL §2; bench-sim's public re-imputation). So we are in the leaking rows.
- **Single-half prediction:** unaffected to first order. The column space is unchanged, and only the prior's parametrization differs. This is consistent with MODEL §7 ("tag-SNV shrinkage: no PGS gain").
- **SV credit and enrichment:** biased. The class level $\ell_{SV}$ and every SV-specific θ are estimated from $b$, not $\gamma$, so they are attenuated by the leakage, and the SNP levels are inflated by the same amount. The goal "SVs must win" cannot be judged from D-space levels.
- **Stacked halves:** mis-specified; see §3.

## 3. Stacked long-read half

The long-read panel members enter as hard-call rows on the same sites, with a cohort covariate that is projected out. After the covariate is projected out, half h has weight $w_h$ (its row share), genotype variance $V_h$ and reliability $r_h^2$.

**Marginal.** If every half is calibrated,
$$r^2_{\rm eff}=\frac{\sum_h w_h V_h r_h^2}{\sum_h w_h V_h},$$
and the per-allele slopes are equal across halves, so one shared coefficient is exact.

For an uncalibrated classical half:
- the slopes differ by the factor $r_h^2$;
- with equal $V_h$, $r^2_{\rm eff}=1/\sum_h (w_h/r_h^2)$, dominated by the worst half.

**Check (`check_r2.py` §1c, 80/20 split).**
- **Calibrated half:** $r^2_{\rm eff}$ = 0.2278 and 0.3227 against the formula's 0.2278 and 0.3227; the slope ratio between halves is 1.02 and 1.00.
- **Classical half:** $r^2_{\rm eff}$ = 0.1557 and 0.4221 against the classical formula's 0.1555 and 0.4222; the slope ratio is 0.129 and 0.367, equal to $r^2$ (0.128 and 0.369).

So **D\* recalibration is required for stacking**, not optional. The HANDOFF item for the κ table is a stacking prerequisite.

**Joint, the stronger condition.** Marginal calibration equalizes only the *marginal* slopes. The imputed half's joint coefficients are $A\gamma$, while the hard-call half's are $\gamma$ (there $D=G$), so a shared coefficient fits neither half.

**Check (`check_r2.py` §1d):** per-allele SV and SNP coefficients.

| ρ | model | SV: imputed half | SV: hard-call half | SV: stacked | SNP: stacked |
|---|---|---|---|---|---|
| 0.5 | shared coefficient | 0.69 | 1.00 | 0.81 | +0.25 |
| 0.5 | A-mapped imputed half | 1.01 | 1.00 | 1.00 | +0.00 |
| 0.8 | shared coefficient | 0.33 | 1.01 | 0.53 | +0.47 |
| 0.8 | A-mapped imputed half | 0.98 | 1.01 | 1.00 | +0.00 |
| 0.95 | shared coefficient | 0.12 | 0.99 | 0.32 | +0.67 |
| 0.95 | A-mapped imputed half | 0.98 | 0.99 | 0.99 | +0.01 |

**Derived fix, the "A-map".** In the imputed half only, replace each LD block's columns by $\tilde D A$, the linear Berkson predictor of the true genotypes. Then:
- both halves share $\gamma$ exactly;
- the independent prior with the $\log r^2$ offset becomes exact;
- scoring an imputed person with $\tilde D A\gamma$ equals $\tilde D b$, the optimal D-space predictor.

**Estimating A needs only per-block covariances, computed inside the AoU workspace.**
- For $j\neq k$, $\mathrm{Cov}(D_j,T_k)=\mathrm{Cov}(D_j,G_k)$ exactly for any truth $T_k=G_k+e_k$ whose error is independent of the imputed column. So one long-read truth suffices; no triad is needed off the diagonal.
- The diagonal is $\mathrm{Var}(D^*_j)$ by calibration.
- This is a per-LD-block covariance. The pipeline computes it inside the AoU workspace, and it never leaves the workspace (HANDOFF item 4).
- It is not a re-proposal of the MODEL §7 item. That item measured prediction without stacking, where the A-map is a reparametrization. Here the A-map changes the stacked likelihood.
- **The A-map is the linear projection of novel-measure's Rao-Blackwellised column $X=\mathbb E[G\mid M]$.** It is exact when $X$ is linear in the stored columns. The full $X$ goes further in two cases (novel-measure, sim-only):
  - haplotype-nonlinear tag information: imputed-row r² 0.195 → 0.358 under a single-draw imputer;
  - large-effect loci, where the exact per-row mixture likelihood replaces the Gaussian working likelihood (variance ratio 1.25 at $\beta^2\mathrm{Var}(G)\approx0.3\sigma^2$).
  See docs/design/math/novel-measure.md.

## 4. Estimated r̂²

The offset uses $\log\hat r^2_j$ from the reliability model.
- If the model is a Berkson predictor on the log scale, $\log r_j^2=\log\hat r^2_j+\eta_j$ with $\eta_j\perp\hat r^2_j$, then the coefficient 1 stays correct and $\eta_j$ is absorbed into $g_c$ as extra spread.
- The target should therefore be $\mathbb E[\log r^2\mid\text{features}]$. $\log\mathbb E[r^2\mid\cdot]$ differs from it by a feature-dependent Jensen gap.
- The spread from $\eta$ is class-specific: SV r² predictions are noisier than SNV ones. This is one more reason the density needs class deviations (§8).

**Code notes, for the engine lane:**
- The docstring of `sv_pgs/imputation_reliability.py` still says the offset enters "with an EB-learned coefficient centred at 1". MODEL.md §3 and §1 above say exactly 1. The docstring is stale.
- `minimum_r2` is a floor on the prediction. It must stay a numerical guard below any truth-supported value, never a prior.

## 5. Frequency dependence: derived from stabilizing selection

**Status on main:** `prior_design.py` has no frequency term; AF enters only through $\log\hat r^2$. On per-SD columns that is the $S=-1$ model (per-SD variance constant across frequency). Every published estimate rejects it:
- $S\approx-0.38$ across 25 UKB traits (Schoech 2019);
- median $S$ of −0.37 (BayesS 2018) and −0.58 (SBayesS 2021);
- LDAK uses −0.25.

(These are from lit-pgs, `scratchpad/team/lit-pgs/REPORT.md`; not re-read here.)

**Model (novel-pleio / novel-evoprior candidate; verified below).**
- Mutations have per-allele effect vectors $a\in\mathbb R^n$ on the $n$ trait axes under selection, with $a\mid s\sim N(0,sI_n)$ and $s\sim g$.
- Stabilizing selection sits at the optimum, with scaled strength $\kappa$, so the diffusion equilibrium density of a segregating site at frequency $m$ is
$$p(s,a,m)\ \propto\ g(s)\,N(a;0,sI_n)\,\frac{e^{-\kappa\lVert a\rVert^2 m(1-m)}}{m(1-m)}.$$
- Write $H=2m(1-m)$ and complete the square in $a$. The focal axis then has
$$a_1\mid s,H\sim N\!\Big(0,\ \frac{s}{1+\kappa Hs}\Big),\qquad s\mid H\ \propto\ g(s)\,(1+\kappa Hs)^{-n/2}\ \Big/\ Z(H).$$
- With the per-SD conversion $\gamma=a_1\,\mathrm{sd}(G)$ and the r² map of §1, the **derived prior** for a stored column is
$$\beta_j\sim\int N\!\Big(0,\ u_j\,H^{\rm loc}_j\,\frac{s}{1+\kappa_c H^{\rm sel}_j s}\Big)\,\frac{g_c(s)\,(1+\kappa_c H^{\rm sel}_j s)^{-n_c/2}}{Z_c(H^{\rm sel}_j)}\,ds,\qquad \log u_j=\ell_c+\log r_j^2+f(d_j).$$
- **Two frequencies with different roles.**
  - $H^{\rm loc}_j$ is the genotype variance of the column in the analysis sample, after FWL projection. It enters with coefficient exactly 1, because it is the per-allele → per-SD unit conversion. Use $\mathrm{Var}(D^*_j)/\hat r^2_j$, or $2\hat p(1-\hat p)$ with $\hat p=\overline{D^*}/2$, never $\mathrm{Var}(D_j)$. Using $\log\mathrm{Var}(D)$ turns the offset's effective coefficient into $-S$.
  - $H^{\rm sel}_j$ is the frequency at which selection–drift equilibrium holds.
    - EUR effect variance tracks African MAF (weight 0.96; Rossen…Price, a medRxiv preprint, via lit-pgs). So $H^{\rm sel}$ should be ancestry-resolved: $p^{\rm sel}=w\,p_{\rm AFR}+(1-w)\,p_{\rm pooled}$, with $w$ learned by 1-D type-II ML from in-cohort ancestry-group frequencies.
    - Those are aggregates, and they are available for SNVs and SVs alike, so the term is SV-fair.
    - When $H^{\rm sel}=H^{\rm loc}$, the whole term collapses to the familiar $H^{1+S}$ form, with $S$ now a function of $H$.

**What the derived form explains.** The illustrative check below uses $g$ log-normal with median 0.05, log-SD 1.5, and $\kappa=20$.
- **The plateau.** For $\kappa Hs\ll1$ the per-allele variance is flat in $H$, so the local $S$ is 0. This is the plateau at low frequency that lit-pgs reports below $T\approx0.006$.
- **The common range.** As $H$ grows, the large-$s$ components hit the per-SD ceiling $1/\kappa$ first. The effective exponent $S_{\rm eff}(H)=d\log\mathrm{Var}(a_1\mid H)/d\log H$ falls smoothly toward −1.

$S_{\rm eff}(H)$:

| $H$ | $n=1$ | $n=4$ | $n=16$ |
|---|---|---|---|
| 1e-4 | −0.00 | −0.01 | −0.02 |
| 2e-3 | −0.04 | −0.08 | −0.16 |
| 7e-3 | −0.12 | −0.18 | −0.30 |
| 0.03 | −0.25 | −0.33 | −0.45 |
| 0.1 | −0.41 | −0.48 | −0.58 |
| 0.5 | −0.53 | −0.58 | −0.66 |

  So SBayesS's single $S$ is an average over $H$ and over the effect-size distribution, and its range −0.37 to −0.58 is what this family produces at common frequencies.
- **Frequency changes shape, not only scale.** The ceiling truncates the upper tail, and the reweighting by $(1+\kappa Hs)^{-n/2}$ moves mass between components. A scale-only smooth $f(\log H)$ is therefore misspecified in the saturation regime. The derived term replaces it.

**Parameters, all learned, none hand-set:**
- $\kappa_c\ge0$, the selection strength per unit effect², learned through λ = nκ/2 (see the ridge below). $\kappa=2N/V_s$ depends on the trait's $V_s$ in trait-SD units, so it is pooled hierarchically across traits: $\log\kappa_{c,t}\sim N(\overline{\log\kappa}_c,\omega^2_\kappa)$. lit-pool reports that SBayesS's $S$ is nearly common across traits (79% of 155 traits in [−0.7, −0.5]), so expect small $\omega^2_\kappa$.
- $n_c>0$, the effective number of trait axes under selection, per class. It differs by annotation: coding −0.74 vs TSS −0.36 in SBayesS, via lit-pool.
  - For SVs, gene disruption means larger $n$ and a heavier $g$. A rare SV is 841× more likely than a rare SNV to be strongly deleterious (Abel 2020, via lit-pgs), which predicts a more negative $S$.
  - Length dependence enters as $n_c(\log\mathrm{len})$ and $g_c$ deviations, both smooth with learned smoothness.
- $w$, the ancestry mixing of $H^{\rm sel}$.

**κ and n sit on a ridge; parametrize by (log λ, log n) with λ = nκ/2.** The bulk of the density ($\kappa Hs\ll1$) sees only $(1+\kappa Hs)^{-n/2}\approx e^{-\lambda Hs}$. κ alone is informed only by the saturated tail.
- **Check with g known** (`check_kappa_n.py`: true κ = 200, n = 6, 6 replicates, 2-D type-II ML):

| M | κ̂ | n̂ | λ̂ (true 600) | corr(log κ̂, log n̂) |
|---|---|---|---|---|
| 30k | 145–247 | 4.35–9.22 | 537–666 | −0.99 |
| 120k | 181–224 | 5.07–6.64 | 568–617 | −0.98 |

  So even with g known the ridge is present. It narrows with M, and λ is identified far more sharply than either κ or n.
- **With a nonparametric g**, novel-evoprior reports that κ̂ and n̂ spread up to 2× along the ridge with no narrowing from 30k to 120k, while λ̂ stays within ±10% (their prototype; not re-run here).
- **Practical rule:**
  - learn log λ (well identified) and log n (a weak direction), in that parametrization;
  - pool log n across classes and traits;
  - report n and κ only with their ridge uncertainty.
- The κ̂ recovery in check (ii) held n = 1 and g fixed, so it overstates how well κ alone is identified.

**Confounded direction and its pin.** $(s,\kappa,\ell)\to(cs,\kappa/c,\ell-\log c)$ leaves the prior unchanged; the weights are invariant too. The mean-variance pin $\mathbb E_g[s]=1$ of §9 removes it. novel-evoprior derived the same direction independently (THEORY.md, Theorem 5).

**Checks (`check_selection.py`):**
- **(i) The closed form is exact given the diffusion joint.** Importance sampling (4·10⁶ draws, κ = 20) matches Var($a_1\mid H$) and its kurtosis in every $H$ bin:

| $n$ | $H$ bin | Var, importance-sampled | Var, closed form | kurtosis, sampled | kurtosis, closed form |
|---|---|---|---|---|---|
| 1 | [0.001, 0.002) | 0.1445 | 0.1453 | 18.3 | 18.7 |
| 1 | [0.01, 0.02) | 0.1168 | 0.1166 | 11.9 | 11.6 |
| 1 | [0.1, 0.12) | 0.0657 | 0.0658 | 6.56 | 6.49 |
| 1 | [0.45, 0.5) | 0.03222 | 0.03221 | 4.63 | 4.63 |
| 4 | [0.001, 0.002) | 0.1413 | 0.1406 | 17.4 | 17.9 |
| 4 | [0.01, 0.02) | 0.1008 | 0.1005 | 10.8 | 10.6 |
| 4 | [0.1, 0.12) | 0.0487 | 0.0485 | 6.46 | 6.41 |
| 4 | [0.45, 0.5) | 0.02169 | 0.02173 | 4.91 | 4.90 |

  The diffusion-vs-Wright–Fisher comparison is novel-evoprior's (proto/test_theory.py).
- **(ii) $Z(H)$ is required.** Type-II ML for κ on normal-means data (M = 40k, 6 replicates per value):
  - with $Z(H)$: κ̂ = 5.07 ± 0.26, 20.08 ± 0.48 and 80.3 ± 2.2 for true κ = 5, 20, 80;
  - without $Z(H)$: κ̂ collapses to the search bound (0.10) every time.
  - Leaving out the per-$H$ normalizer adds $-\sum_j\log Z(H_j)$ to the evidence, which is minimized at κ → 0.
- **(iii) The M-step is jointly concave in per-allele exponential-family coordinates.** With $\eta=\log g$ on the grid, the M-step objective is
$$\sum_j\Big[q_j^\top\eta-\tfrac{\kappa}{2}H_j\,\mathbb E[a_j^2]-A_{H_j}(\eta,\kappa)\Big],\qquad A_H=\log\sum_k e^{\eta_k}(1+\kappa Hs_k)^{-1/2}.$$
  $A_H$ is a log-partition function, hence convex, so the objective is jointly concave in $(\eta,\kappa)$.
  - Numerically: over 50 random points, the largest non-null Hessian eigenvalue divided by the most negative is −3.2·10⁻⁵, so every non-null eigenvalue is negative.
  - The κ stationarity is moment matching: $\sum_jH_j\mathbb E_{\rm post}[a_j^2]=\sum_jH_j\mathbb E_{\rm prior}[a^2\mid H_j]$.
  - Concavity is established in these coordinates only (nothing was checked in the variance-map coordinates), so the engine should do its update in them.

**Identification against the r̂² offset.** The offset is fixed at 1 and $H^{\rm loc}$ is fixed at 1, so κ, $n$ and $w$ are identified from the within-class variation of $H^{\rm sel}$ and from the ceiling's shape signature.
- An AF-dependent miscalibration of r̂², $\log\hat r^2=\log r^2+\delta(p)$, is still absorbed by the frequency term. The reliability model's AF calibration on truth is what separates them.
- **Check (`check_frequency.py`).** Normal means with M = 20k and n = 50k, a fixed log-normal mixing family (used only for this identifiability demonstration), true $1+S=0.5$, 12 replicates, and r² depending on AF with slope b:

| b | $\mathrm{corr}(\log\hat r^2,\log H)$ | offset derived: $\widehat{1+S}$ (emp. SD) | offset coefficient free: $\widehat{1+S}$ (emp. SD) |
|---|---|---|---|
| 0 | 0.00 | 0.512 (0.024) | 0.513 (0.024) |
| 0.6 | 0.72 | 0.507 (0.026) | 0.508 (0.031) |
| 1.2 | 0.86 | 0.507 (0.023) | 0.504 (0.042) |
| 0, with miscalibration $\delta=0.15\log\mathrm{MAF}$ | 0.59 | 0.335 (0.010) | 0.340 (0.017) |
| 1.2, with miscalibration | 0.91 | 0.333 (0.024) | 0.347 (0.058) |

  - Freeing the offset's coefficient costs precision: the SD inflates by 1.8× at ρ = 0.86, and $1/\sqrt{1-\rho^2}=1.96$.
  - It also buys no protection against a miscalibrated r̂²-AF curve. Both fits absorb the $0.15\log\mathrm{MAF}$ error as a −0.17 shift in $1+S$.
  - So keep the offset derived, and certify the reliability model's AF calibration on truth. That calibration is the only thing separating selection from r̂² error.

**Exactness conditions:**
- diffusion equilibrium;
- the trait mean at the optimum (a directional component breaks the $m\leftrightarrow1-m$ symmetry and the closed form);
- Gaussian stabilizing selection;
- additive effects;
- isotropic mutational pleiotropy. The anisotropic version, $K=\mathrm{diag}(\kappa_i)$, gives weights $\prod_i(1+\kappa_iHs)^{-1/2}$, with the focal axis's own κ in the variance map.

A residual smooth in $\log H^{\rm sel}$ is kept only as a diagnostic, through the score test of §6 in the $\log H$ direction, never as a default term.

## 6. Shape vs scale

Write $t=\log(u s)$ for a variant's log prior variance.
- **Scale-only means a translation family:** $h(t\mid d)=h_0(t-m(d))$, with $m(d)=\ell_c+\log r^2+f(d)$.
- It is exactly right iff annotations act multiplicatively on every effect scale at once.
- It is wrong when an annotation changes the *weights* of the density, for example raising the tail probability without moving typical effects.

**Score test from a fitted scale-only model.** Perturb the shape of the annotated variants ($a_j=1$): $w_k\to w_k e^{\epsilon^\top\phi_k}/Z$. Here $\phi$ holds degree-2 and degree-3 polynomials in $t$, orthogonal under the fitted weights $w$ to $\{1,t\}$ (the normalization and translation directions). Let $q_{jk}$ be variant $j$'s posterior responsibility for grid point $k$, the same quantity the EB M-step uses. The per-variant score at $\epsilon=0$ is
$$U_j=a_j\Big(\sum_k q_{jk}\phi_k-\sum_k w_k\phi_k\Big).$$
The Rao statistic uses the observed information $\mathcal I$ of all parameters $(\epsilon,\eta,\theta)$ at the scale-only fit:
$$T=S^\top\mathcal I^{-1}S,\qquad S=\Big(\textstyle\sum_jU_j,\ \nabla_\eta,\ \nabla_\theta\Big)\ \ (\nabla_\eta,\nabla_\theta\approx0\text{ at the fit}),\qquad T\sim\chi^2_{\dim\phi}\ \text{under }H_0.$$
- The efficient information per annotated variant, $[(\mathcal I^{-1})_{\epsilon\epsilon}]^{-1}/\sum_ja_j$, is what sets the power. It is small, because one $\hat\beta_j$ says little about its own latent $t_j$: normal-means deconvolution is severely ill-posed.
- **Calibration needs the test restricted to identified directions.** The part of $g$ below the noise floor (variances ≪ the sampling variance) carries essentially no Fisher information, and a fixed-df score test that includes those nuisance directions is ill-conditioned. The practical test therefore:
  - uses only the identified subspace: shape directions $\phi$ and nuisance directions whose information clears the certificate tolerance;
  - or, equivalently in production, is the variance-component score test of $\lambda_\delta=\infty$ in the penalized model below, whose penalty removes the unidentified directions;
  - is calibrated by a parametric bootstrap from the fitted scale-only model.
- With LD, sum the score contributions within LD blocks before forming outer products (block jackknife), since EP responsibilities are dependent within blocks.

**Principled generalization if the test fires.** Let
$$\log g(t\mid d)=\log g_0(t-m(d))+\sum_k d_k\,\delta_k(t),$$
with each $\delta_k$ a second-difference-penalized deviation constrained orthogonal to $\{1,t\}$ under $g_0$.
- λ_k is learned; $\lambda_k\to\infty$ returns scale-only *exactly*, so the test and the model agree.
- The orthogonality keeps $m(d)$ and the location pin (§9) identified.

**Checks (`check_shape_known.py`, `check_shape.py`, `check_shape_param.py`).**
- Setup: normal means with unit noise and 30% annotated variants, where $t$ follows a two-component normal mixture on the log-variance scale (0.9 at −3, 0.1 at +1, SD 0.7).
  - H0: a translation by +1;
  - H1a: tail weight 0.1 → 0.3;
  - H1b: tail only, 0.1 → 0.05 + 0.05 at +3.
- **Power in the simple case** (nuisance known except the translation, which is projected out; 4·10⁵ draws per group):

| truth | best translation θ* | NCP per annotated variant | annotated variants for 80% power (α = 0.05, df 2) |
|---|---|---|---|
| H0 | 1.011 | 6.6·10⁻⁶ (≈ 0) | none, as it should be |
| H1a | 1.139 | 9.0·10⁻³ | ≈ 1,070 |
| H1b | 1.094 | 1.08·10⁻² | ≈ 900 |

  A shape change of this size is detectable with about 10³ annotated variants when the null density is known. Real annotation classes have 10⁴–10⁶ variants.
- **Fitted-nuisance forms fail without the identified-subspace restriction.** Even with the correctly specified two-component $g$:
  - The observed-information Rao statistic gave values from −2.6 to 17.4 in 11 H0 replicates.
  - The OPG (BHHH) form gave values from −1388 to 2728 in 27 H0 replicates.
  - Negative values mean the information matrices are numerically singular: the lower component (variance 0.05, noise 1) is nearly unidentified. This is the same sub-noise tail that forces the $\mathbb E_g[s]=1$ pin (§9).
  - With a flexible unpenalized spline $g$, the per-variant OPG-regression form rejected 50–67% under H0.
- **Cost of the wrong shape model** (posterior-mean MSE excess over the oracle prior, flexible $g$, M = 3000, 6 replicates):
  - H0: scale-only 1.07%, separate densities 1.72%. Pooling wins when the truth is a translation.
  - H1a: scale-only 1.23%, separate 0.73%.
  - The nested deviation with learned λ is built to take the better of the two; prior sweep B measures it.

## 7. The TR signed-length column

The locus column is $Z_i=\sum_a\Delta_a D_{ia}$ with $\Delta_a=\mathrm{len}_a-\mathrm{len}_{\rm ref}$. The true total length is $L_i=\sum_a\Delta_a G_{ia}$.
- **Implied prior.** The rank-one allele prior $\beta_a=\Delta_a\theta$ gives $\mathrm{Cov}(\beta_a,\beta_b)=\Delta_a\Delta_b\mathrm{Var}(\theta)$. In standardized form $\tilde\theta=\theta\,\mathrm{sd}(L)$, and §1 gives $b_Z=r_Z\tilde\theta$. So
$$\log u_Z=\ell_{TR}+\log r_Z^2+(1+S_{TR})\log\mathrm{Var}(L)+f(d).$$
  - For TRs, $\mathrm{Var}(L)$ is largely set by the mutation rate, so its slope mixes selection with mutation–selection balance; the pending Ewens θ̂ feature separates them.
  - The selection map of §5 carries over with $H\to\mathrm{Var}(L)$ only heuristically. Stepwise length mutation has no biallelic diffusion, so the closed form is a candidate for TRs, not a derivation.
- **Reference invariance.** Changing the reference shifts every $\Delta_a$ by a constant $c$, so $Z\to Z+c\sum_aD_{ia}=Z+2c$, a constant removed by centering. This requires $\sum_a D_{ia}=2$ over *all* alleles, including REF. Records dropped from a locus, or a background-removal floor applied to only some alleles, break it.
- **Locus reliability.** In general
$$r_Z^2=\frac{(\Delta^\top\Sigma_{DG}\Delta)^2}{(\Delta^\top\Sigma_D\Delta)(\Delta^\top\Sigma_G\Delta)}.$$
  - **Posterior-mean columns** (Berkson, $\Sigma_{DG}=\Sigma_D$) give $L=Z+\sum_a\Delta_a e_a$ with the error uncorrelated with $Z$, so this reduces to $r_Z^2=\Delta^\top\Sigma_D\Delta/\Delta^\top\Sigma_G\Delta$.
  - **Draw-type columns** have $\Sigma_{DG}=\Sigma_{\rm pm}$ and $\Sigma_D\approx\Sigma_G$, so $r^2_{Z,\rm draw}=(r^2_{Z,\rm pm})^2$. This is the locus analogue of κ = √r².
  - All three covariances are computed inside the AoU workspace and never leave it. $\Sigma_{DG}$ needs one long-read truth per person, with no triad off the diagonal (§3).
  - **Why the length column works:** imputation errors between alleles of similar length have opposite signs and nearly equal $\Delta$, so they cancel in $L$. That makes $r_Z^2$ far larger than the per-allele r², the mechanism behind the measured +4–12%.
- **Calibrating Z for stacking.** Summing per-record D\* values with allele-specific κ_a does not give a calibrated Z. Z must be recalibrated at the locus: $Z^*=\mu+\kappa_Z(Z-\mu)$, with $\kappa_Z=\Delta^\top\Sigma_{DG}\Delta/\Delta^\top\Sigma_D\Delta$. It is affine, so $r_Z^2$ is unchanged, but it is required before the TR column shares a coefficient with the long-read half (§3).

**Check (`check_tr.py`).**
- Setup: 10 alleles, repeat counts 8–17 at a 4-bp unit, confusion with neighbouring lengths at error rate ε, n = 10⁶.

| ε | mean per-allele r² | r²_Z, posterior-mean columns | Berkson formula | r²_Z, draws | (r²_Z,pm)² |
|---|---|---|---|---|---|
| 0.1 | 0.729 | 0.880 | 0.880 | 0.773 | 0.774 |
| 0.3 | 0.384 | 0.691 | 0.691 | 0.477 | 0.477 |
| 0.5 | 0.180 | 0.545 | 0.544 | 0.297 | 0.297 |

- corr(Z, Z shifted by a reference change) = 1.000000 in every case.
- $b_Z/(r_Z\tilde\theta)$ = 0.994–1.011.

## 8. The SV-context kernel

The lead ruled the form $f_j=\sum_kH_k\,w(\log\mathrm{dist}_{jk})$ over all SV loci $k$, with $w$ a smooth whose smoothness is learned. It replaces the ±50 kb window density and the K = 3 nearest SVs. The rules below are implemented in `store_converter.sv_kernel_features` (deslop-store, lane/deslop-store 214f197).

**Distance.**
- Tagging is broken by recombination between the variant and the nearest SV breakpoint. So $d_{jk}=\max(0,s_k-e_j,s_j-e_k)$ on 0-based half-open spans, entered as $x_{jk}=\log(d_{jk}+1)$; the +1 is the 1-bp coordinate resolution.
- Overlap (nesting inside an SV allele) is a different relation from adjacency. It gets its own per-class feature, $\sum_{k\in c,\ \rm overlap}H_k$.
- With CIPOS/CIEND breakpoints, use the expected feature over the confidence interval.
- The variant's own locus is excluded (same bubble, or the same TR/SV locus).

**Weight.** Use the per-allele record's genotype variance $H_k=2f_k(1-f_k)$, the same $H$ as the frequency term, not $1-\sum f^2$.
- Each allele record is its own column.
- If the mechanism is tagging, the derived weight is the SV's own prior variance, $\propto H_k^{\rm loc}\,s/(1+\kappa H_k^{\rm sel}s)$. $H_k$ is its $\kappa\to0$ limit; a block weighted by $H_k\log H_k$ tests the first-order correction.

**Range.**
- Under LD decay $\mathbb E\,r^2(d)\approx1/(1+\rho d)$, the H-weighted tagging mass per octave of $d$ is constant for $d\gg1/\rho$. So there is no natural cutoff before LD reaches the sampling floor $1/n$.
- $x_{\max}=\min\{\log(1+\text{chromosome length}),\ \log d^*\}$, where $\mathbb E\,r^2(d^*)=1/n$ on the empirical SV–variant LD curve. With n ≈ 60k, $d^*$ usually exceeds the chromosome.
- Truncation is then exact rather than approximate: B-splines have compact support in $x$, so a locus outside a basis function's support contributes exactly 0.

**Basis.**
- Cubic B-splines uniform in $x$, the scale-free choice under that decay. Start at one knot per octave ($h=\log2$) and halve until the log-evidence changes by less than the certificate tolerance. Knot count is a resolution knob checked by convergence; the smoothness is learned.
- Store at the finest resolution used: coarser dyadic B-splines are exact linear combinations of finer ones (knot insertion).
- Class blocks carry $w_c=w_0+\delta_c$ (pooled, learned penalty). Length enters as a tensor product with a B-spline basis in $\log\mathrm{len}_k$; the minimum is $\{1,\log\mathrm{len}_k\}$, its null-space part.

**Binning bound, if binning is used for far bases.**
- For uniform cubic B-splines, $\max|B'|=1/(2h)$, and a locus moved by $b/2$ at distance $d$ shifts $x$ by at most $b/(2d)$.
- The relative feature error is therefore at most $(3/8)(b/d)/h$.
- Matching it to the store's quantization step $q=1/254$ gives $b/d\le\varepsilon=(8/3)\,q\,h$, about 0.73% of distance at $h=\log2$.

**Identifiability with the other scale-model terms.**
1. The far bases are regional SV density and nearly constant across variants. Centre every kernel feature within the target variant's class; otherwise they are collinear with $\ell_c$.
2. Self-exclusion keeps the kernel separate from the variant's own frequency term and class level.
3. **The leakage trap.** Without the A-map (§3), an SNV's D-space prior variance carries the additive leakage term $\sum_kA_{jk}^2u_k$ from nearby imperfect SV columns, and the kernel would absorb it in the wrong (multiplicative) form. With the A-map, leakage lives in the measurement model and the kernel captures only genuine proximity enrichment. The tagging-strength features in $d_j$ (max and summed r², ρ²_j) then compete with the kernel's short-range bases; they are identified, since r² is not a function of distance, and their added value is a measurement.
4. The kernel is a scale (translation) term. Whether proximity changes shape is for the §6 test to decide.
5. Storage, if it binds: keep the class-summed far bases ($w_0$), plus class blocks only where $n\,\mathrm{Var}_j(F^{(c)}_{jm})\,\hat\tau^2$ clears the certificate tolerance, with $\hat\tau^2$ from `hyperprior_pooling`. That rule is computable on one chromosome.

## 9. Pooling across classes and traits

**The hierarchical form:**
- **per-trait intercept $\ell_t$:** unpooled, since heritability and polygenicity differ by trait;
- **class offsets and annotation coefficients, relative to it:** $\theta_t\sim N(\bar\theta,\Omega)$, which is `hyperprior_pooling.py`, with learned $\Omega$ and penalty weights $\nu_k$;
- **the density:** $\log g_{c,t}=\eta_0+\delta_c+\delta_t\,(+\delta_{ct})$, each deviation with its own second-difference penalty and learned λ.

**Identifiability:**
1. **Level vs. translation of the density.**
   - For *any* smooth density, raising $\ell_{c,t}$ by ε while shifting $g_{c,t}$ by −ε leaves the likelihood unchanged to first order.
   - For a log-quadratic $\eta$ it is exact: a tilt $b\,t$ is a shift by $b\tau^2$.
   - **Pin every density's mean variance, $\mathbb E_{g_{c,t}}[s]=1$, not its mean log-variance.** Then $\ell_{c,t}=\log$ (mean prior variance), the per-variant heritability scale, and $g$ carries only shape.
     - The left tail of $\log s$ (effects far below noise) is not identified by the data. So a pin on $\mathbb E[\log s]$ can be met by moving unidentified mass while the level drifts.
     - Check (`check_pooling_profile.py`: M = 20k, class density = true η plus a degree-5 polynomial deformation, level profiled). Log-likelihood drop from the maximum at level −1.5 / −1.0 / −0.5 / 0 (true) / +0.5 / +1.0:
       - no pin: 0.00 / 0.13 / 0.16 / 0.13 / 0.08 / 0.07. Flat, so not identified.
       - $\mathbb E[\log s]$ pinned: 0.00 / 0.00 / 0.03 / 0.44 / 3.3 / 18.9. The maximum is at the range edge, so still not identified.
       - $\mathbb E[s]$ pinned: 442 / 221 / 54 / **0** / 3.9 / 7.4. A sharp maximum at the truth.
     - The same pin breaks the $(s,\kappa,\ell)$ direction of §5.
     - Equivalently, and this is what the oracle reference does, drop $\ell_c$ and let $g_c$ carry its own location, with a translation-invariant roughness penalty. It is the same family; only the Laplace evidence differs.
   - **The exact tilt case** (`check_pooling.py` (a)): with log-quadratic $\eta$ (τ = 1.3), tilting by $bt$ and moving the level by $-b\tau^2$ changes the log-likelihood by at most 2·10⁻⁵ for $b$ = 0.2, 0.5, 1.0. The tilt alone costs 5.2, 50 and 278.
2. **Constants** in the δ's are absorbed by normalization. Impose sum-to-zero across classes and across traits, so the δ's are identified even when λ→0.
3. **The per-trait intercept vs. class offsets:** a baseline class, or sum-to-zero over classes.
4. **θ vs. δ:** θ acts on within-class variation between variants, and δ on the distribution. They are identified provided the annotation design has no class-constant column.

**If cross-trait pooling is ruled out.** The pleiotropy layer is being deleted per the user's "no multi-trait", and pooling of hyperparameters across traits may follow. Every rule above still holds with $t$ fixed: class pooling, the pin, and sum-to-zero over classes.

**Code note: hand-set penalties (on main at c2a9443).** `prior_design.py:_scale_model_penalty` penalizes the scale-model coefficients with config constants `scale_model_ridge_penalty = 1.0` and `type_offset_penalty = 2.0`. `config.py` also carries `tpb_hierarchical_prior_variance = 1.0`. These are hand-chosen priors, which SPEC forbids. The conforming form, with learned $\nu_k$ and $\Omega$, is `hyperprior_pooling.py`. deslop-fit has claimed prior_design.py.

## 10. Summary of checks

| Claim | Check | Result |
|---|---|---|
| Marginal $b=r\gamma$ for any column | `check_r2.py` §1a | 36/36 cells at 1.00 ± 0.01 |
| Joint exactness only for joint posterior means | §1b | leak −0.003 (joint PM) vs +0.32…+0.85 (draws) |
| $A=\Sigma_D^{-1}\Sigma_{DG}$ predicts the leak | §1b | ±0.003 |
| Stacked $r^2_{\rm eff}$ formula (calibrated / classical) | §1c | exact to 4 decimals |
| A-map fixes stacked credit | §1d | SV 0.53 → 1.00, SNP +0.47 → 0.00 at ρ = 0.8 |
| TR locus r², draws squared, reference invariance | `check_tr.py` | exact |
| Stabilizing-selection map exact given the diffusion joint | `check_selection.py` (i) | variance to 0.5%, kurtosis to 3% |
| $S_{\rm eff}(H)$: plateau and literature range | (ii) | 0 → −0.53…−0.66 |
| $Z(H)$ required for EB | (iii) | κ̂ recovered with Z; collapses to the bound without |
| M-step concave in per-allele exponential-family coordinates | (iv) | all non-null eigenvalues < 0 |
| Offset derived vs free; r̂² AF miscalibration | `check_frequency.py` | same bias, 1.8× SD when free |
| Tilt ≡ shift for log-quadratic η | `check_pooling.py` (a) | Δloglik ≤ 2e-5 |
| The pin must be $\mathbb E_g[s]=1$ | `check_pooling_profile.py` | only the mean-variance pin identifies the level |
| Shape score test: simple-case power | `check_shape_known.py` | ≈ 900–1,070 annotated variants for 80% power; H0 NCP ≈ 0 |
| Fitted-nuisance shape test | `check_shape_param.py` | ill-conditioned unless restricted to identified directions |

## 11. Ranked recommendations
Ordered by expected gain × confidence. Each is a derived model term plus the measurement that decides it.
1. **Frequency term from stabilizing selection (§5).**
   - The form: $H^{\rm loc}$ with coefficient 1; the tilted mixing density $g_c(s)(1+\kappa_cH^{\rm sel}s)^{-n_c/2}/Z_c$ with variance map $s/(1+\kappa_cH^{\rm sel}s)$; κ, $n$ and ancestry mixing $w$ learned; the M-step in per-allele coordinates.
   - Today's code is $S=-1$, rejected by every published estimate.
   - Decided by: prior sweep A (the frequency arms), with paired R², the SV-vs-SNV level bias and κ̂/n̂ recovery.
2. **The density pin $\mathbb E_g[s]=1$** (or the oracle's no-level parametrization) for every pooled density, with sum-to-zero δ's. This is identifiability, not an option.
3. **Remove the hand-set scale-model penalties** in `prior_design.py` and `config.py` in favour of learned ν and Ω (`hyperprior_pooling.py`). SPEC compliance.
4. **The A-map for the stacked imputed half (§3),** and SV credit judged in γ-space.
   - Decided by: a stacked simulation with draw-type SV columns and tag SNPs, measuring imputed-half held-out R² and the ℓ_SV bias.
   - Needs Σ_DG, computed inside the workspace from one long-read truth (HANDOFF item 4). Consistent with novel-measure's THEORY.md Results 1–2.
5. **Locus-level κ_Z for TR columns before stacking,** and the general $r_Z^2$ formula for the TR offset (§7).
6. **The SV-context kernel (§8),** now implemented in the store: centring within class, self-exclusion, and a measured comparison against the tagging-strength features.
7. **Shape vs scale (§6).** Nested deviations δ_k orthogonal to {1, t} with learned λ; SBayesRC's mixture-weight annotations give +14% over SBayesR (lit-pgs).
   - Decided by: prior sweep B (H0 / H1a / H1b), with the variance-component score test as the diagnostic.
8. **Reliability-model hygiene:**
   - target $\mathbb E[\log r^2\mid\text{features}]$;
   - certify its AF calibration on truth, since §5 shows a miscalibration shifts $1+S$ one for one;
   - fix the stale "EB coefficient centred at 1" docstring.
