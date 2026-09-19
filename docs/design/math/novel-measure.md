# Measurement theory: imputed SV genotypes with an internal truth panel

**Status (2026-09-19):**
- Theory: every numbered result below has a numerical check (19/19 pass), and math-scale's two-locus check independently confirms Results 2, 3 and 4″.
- Numbers: every "measured" number here comes from our own simulator and is **sim-only** under the evidence rule. It checks the algebra, not accuracy.
- Adoption: the Rao–Blackwellised column (§7) is a *candidate*, gated on the neutral benchmarks (bench-sim arm `beagle_rb`: Beagle re-imputation of public 1kGP SVs, against the same arm's DS).
- Code: the prototype, its tests and the harness contract live with the novel-measure lane's scratch files.

## 0. Setup

For person $i$ and variant $j$:
- $G_{ij}$ is the true genotype, or allele count; for a tandem-repeat locus it is the length functional $Z$.
- $M_i$ is everything we measure about person $i$:
  - sequenced SNV genotypes $S_i$, with their phase;
  - the imputer's output at SV records, $(DS_{ij}, GP_{ij})$;
  - GATK-SV calls $B_{ij}$;
  - for the long-read panel $P$, the truth $T_{ij}$;
  - and, as reference data shared by everyone, the panel's phased haplotypes $\mathcal H$ with their SV alleles.

The trait follows $y_i = c_i^\top\alpha + G_i^\top\beta + \varepsilon_i$, with $\varepsilon_i \sim N(0,\sigma^2)$.

**Assumption N (non-differential measurement):** $p(y \mid G, M) = p(y \mid G)$. This holds for imputation, read calling and long-read calling, none of which use $y$.

Define the calibrated conditional mean $X_{ij} = E[G_{ij} \mid M_i]$, the conditional covariance $V_i = \mathrm{Cov}(G_i \mid M_i)$, and the Berkson residual $U_i = G_i - X_i$, which satisfies $E[U_i \mid M_i] = 0$.

For any statistic $h$ of $M$, the reliability is $r^2(h) = \mathrm{Var}(E[G \mid h])/\mathrm{Var}(G)$. For a calibrated column, $r^2(h) = \mathrm{corr}^2(h, G)$.

## 1. The exact training likelihood (question a)

By N,
$p(y_i \mid M_i, \beta) = \int N(y_i;\, c_i^\top\alpha + g^\top\beta,\, \sigma^2)\, p(g \mid M_i)\, dg,$
with exact moments
$E[y_i \mid M_i] = c_i^\top\alpha + X_i^\top\beta$ and $\mathrm{Var}(y_i \mid M_i) = \sigma^2 + \beta^\top V_i\beta.$

The cumulants of order $r \ge 3$ of $G_i^\top\beta \mid M_i$ are $O(\|\beta\|^r)$. So

$$\log p(y_i \mid M_i,\beta) = \log N\big(y_i;\, c_i^\top\alpha + X_i^\top\beta,\ \sigma^2 + \beta^\top V_i\beta\big) + O(\beta^3). \tag{1}$$

- **Result 1 (sufficiency).** To first order in the effects, the likelihood depends on $M$ only through $X$. Here $V$ enters at second order: the relative size of that term is $\beta^\top V\beta/\sigma^2 \le h^2_{\rm lost}/(1-h^2)$, where $h^2_{\rm lost} = \sum_j \beta_j^2 E[V_{jj}]$ is the genetic variance hidden by measurement error. For the polygenic bulk this is negligible. For a single large-effect locus it is not; see §8.
- **Consequence.** The optimal regressor for every row is its own calibrated conditional mean $X_i$: $T_i$ for panel rows, and $E[G \mid M_i]$ for imputed rows.

## 2. Joint Berkson unbiasedness and the credit leak

**Result 2 (Berkson).** $y = C\alpha + X\beta + (U\beta + \varepsilon)$ with $E[U\beta + \varepsilon \mid M] = 0$. So the joint regression of $y$ on all columns of $X$ is conditionally unbiased for the true-genotype effects. This holds whatever the LD, and it includes the split of credit between an SV and its tags.

It has two conditions:
- every column is the conditional mean given the same information set $M_i$;
- N holds.

**Result 3 (leak).** Let a column be $D = X + W$, with $W$ independent of $(X, U, \varepsilon)$ given $M$; a posterior draw is the case $\mathrm{Var}(W) = E\,\mathrm{Var}(G \mid M)$. Split the SV's conditional mean into its linear tag projection and the rest, $X = a^\top S + R$. Then

$$\beta_D \to \beta_{SV}(1-L), \qquad \gamma_S \to \gamma_S + \beta_{SV}\, a\, L, \qquad L = \frac{\mathrm{Var}(W)}{\mathrm{Var}(R) + \mathrm{Var}(W)}.$$

The fraction $L$ of the SV's effect moves to its tags.
- **A linear recalibration $D^* = \mu + \kappa(D-\mu)$ does not change $L$;** it only rescales the SV coefficient by $1/\kappa$ (verified to $10^{-8}$).
- **When the SV is well tagged linearly** ($R$ small), almost all of its credit leaks. The no-stack arm in the sweep shows this: the SV credit share is about 0.01 against a true 0.5.

For a single-draw imputer, $\mathrm{Var}(W) = (1-r_X^2)\mathrm{Var}(G)$ and $\mathrm{Var}(R) = (1-\rho^2)\, r_X^2\,\mathrm{Var}(G)$, where $\rho^2$ is the share of $X$ that its tags explain linearly. So
$$L = \frac{1-r_X^2}{(1-r_X^2) + (1-\rho^2)\,r_X^2}.$$

| $\kappa = r_X^2$ | $\rho^2 = 0.5$ | $\rho^2 = 0.8$ | $\rho^2 = 0.95$ |
|---|---|---|---|
| 0.9 | 0.18 | 0.36 | 0.69 |
| 0.8 | 0.33 | 0.56 | 0.83 |
| 0.6 | 0.57 | 0.77 | 0.93 |

The better an SV is tagged, the more of its credit a draw-like column hands to the tags.

## 3. Span invariance: why refining dosages from the imputed files gave no gain

**Result 4.** Suppose all rows of the design come from one measurement process. For any invertible $A$, the design $\tilde Z = ZA$ gives the same Gaussian likelihood under the reparametrisation $\tilde\beta = A^{-1}\beta$. So:
- **(a) Per-column rescaling is an exact no-op.** Under empirical-SD standardisation, $\kappa$-recalibration changes nothing, provided the prior scale uses $\log r^2$ consistently; the prediction difference was $10^{-14}$.
- **(b) Any linear re-mix of the SV column with measured columns changes the fit only through the prior.** That covers tag-SNV shrinkage and linear locus kernels.

A refinement can therefore help only by:
1. extracting information that is non-linear in the measurements, or
2. adding information not already among the columns.

MODEL.md §7's null result ("refining SV dosages from information already in the imputed files: no PGS gain") is what Result 4 predicts for linear, in-span refinements. The draw analysis in §5 shows where the non-linear information lies.

**Result 4′ (the $r^2$ offset is exactly 1).** The standardised coefficient of a column is $\theta = b\sqrt{r^2_{\rm col}}$, with $b$ the per-SD effect of $G$. It holds for both kinds of column:
- a calibrated column: $\mathrm{Var}(X) = r^2\mathrm{Var}(G)$;
- a draw: $\mathrm{Var}(D) = \mathrm{Var}(G)$ and $\mathrm{Cov}(D, G) = r_X^2\mathrm{Var}(G)$, so $\theta = b\, r_X^2 = b\sqrt{r^2(D)}$.

So $\log u_j \ni \log r^2_{\rm col}$ with coefficient 1, as MODEL.md derives; verified for both.

**Result 4″ (the stacked column's reliability).** Take a column whose panel rows are $T$ and whose imputed rows are calibrated with reliability $r^2$. After group centring,

$$r^2_{\rm col} = \frac{n_P V_P + n_I\, r^2 V_I}{n_P V_P + n_I V_I}, \qquad \text{which is } a + (1-a)\,r^2 \text{ when } V_P = V_I,\ a = n_P/n.$$

This, not the imputed-half $r^2$, is the offset the stacked column's prior needs.

**Result 4‴ (a calibrated column carries its own reliability).** Scale a calibrated column by the SD of $G$ instead of its own SD, with
$$\widehat{SD}(G)^2 = \mathrm{Var}(X) + \overline{\mathrm{Var}(G \mid M)},$$
the law of total variance, using the row-level conditional variances. Then:
- the coefficient is exactly the per-SD-of-G effect $b$, by Result 2;
- the column's variance is $r^2_{\rm col}$;
- the prior $b \sim g$ needs no reliability offset.

This is algebraically equivalent to empirical-SD scaling with the $\log r^2_{\rm col}$ offset when the offset is right. It needs no $\hat r^2$ model, and it can't be broken by a mis-estimated or degenerate $\hat r^2$. In the sweep it equals the offset form to 4 decimals in every paired replicate (RESULTS.md).

## 4. Row heterogeneity: where calibration matters (the stacked half)

**Result 5.** Suppose a column's rows come from different processes: panel rows $T$ and imputed rows $h(M)$, or different ancestries.
- A transform applied to one row group is not a column operation, so Result 4 no longer protects it.
- A single coefficient fitted over both groups converges to

$$\hat\beta \to \beta\,\frac{n_P V_P + n_I\,\mathrm{Cov}(h, G)}{n_P V_P + n_I\,\mathrm{Var}(h)}.$$

  It is unbiased iff $\mathrm{Cov}(h, G) = \mathrm{Var}(h)$ on the imputed rows, i.e. the imputed rows are calibrated in $G$ units.
- For a draw-like $h = D$, $\hat\beta \to \beta\,(n_P + n_I r_X^2)/(n_P + n_I)$ (verified: 0.311 against a predicted 0.309, with the truth 0.4).

So a stratum-level $\kappa$ removes the unit mismatch only on average over loci. Per-locus calibration removes it exactly; the calibrated conditional mean removes it and also restores the lost information (§5).

## 5. Information accounting and the storage question (question c)

**Result 6 (the information order).** For any statistic $h$ of $M$, $\mathrm{Cov}(h, G) = \mathrm{Cov}(h, X)$. So, by Cauchy–Schwarz,

$$\frac{\mathrm{Cov}(h, G)^2}{\mathrm{Var}(h)} \le \mathrm{Var}(X),$$

with equality iff $h$ is affine in $X$.
- The per-row Fisher information about $\beta$ is maximised by $X$ and by nothing else.
- Replacing the imputer's output with $X = E[D \mid M]$ is literally a Rao–Blackwellisation. $D$ is an unbiased report of $G$ given $M$ for draw-type imputers.
- The data-processing chain is $r^2(\text{8-bit }DS) \le r^2(DS) \le r^2(GP) \le r^2(X)$.

**Result 7 (the imputer's output mechanism fixes the gap).**
- **Tempered posterior** ($GP \propto P^\tau$): $r^2(GP) = r^2(X)$ exactly, since the map is invertible. $r^2(DS) < r^2(GP)$, but only slightly. Measured: 0.329, 0.332, 0.336; the small GP–X gap is the binned estimator's resolution.
- **One posterior draw:** $GP$ is one-hot, so $r^2(DS) = r^2(GP) = r_X^4$ and $\kappa = r_X^2 = \sqrt{r^2(DS)}$. bench-sim's calibration found this signature [semi-real] for TR and SV records under real GLIMPSE2 and Beagle re-imputation of public 1kGP haplotypes. Measured here [sim-only]: $r_X^2 = 0.334$, $r^2(D) = 0.111$ against $0.1115$, $\kappa = 0.333$.
- **An average of $k$ draws:** $r^2 = r_X^4/(r_X^2 + (1-r_X^2)/k)$ and $\kappa = r_X^2/(r_X^2 + (1-r_X^2)/k)$ (verified for $k = 2, 4$).

A measured $\kappa \approx \sqrt{r^2}$ therefore identifies the regime. Of the mechanisms simulated, only the single draw gives it (κ against $\sqrt{r^2}$: 0.454 against 0.455). The others don't:
- a 4-draw average: 0.77 against 0.59;
- a tempered posterior: 0.87 against 0.65;
- a confidently wrong imputer: 0.79 against 0.59;
- a calibrated mean: 1.00 against 0.67.

In the draw regime, the stored imputed SV dosages carry about $r^2(DS) = r_X^4$ of the genotype information, while about $r_X^2 = \kappa$ is available from $M$. At $\kappa = 0.8$ that's $r^2$ 0.64 against 0.80: a quarter more information per SV row. No function of $(DS, GP)$ can recover it (Result 6 applied to $h = GP$). It needs $M$ itself: the person's phased tag haplotypes and the panel's haplotypes.

**Result 8 (8-bit storage).** The code step is $\Delta = 1/127$, so the rounding variance is at most $\Delta^2/12 = 5.2\times10^{-6}$, and only on rows with non-zero value.
- **Draw-like output** ($DS \in \{0, 1, 2\}$): the store is exact.
- **Continuous columns:** the fixed $DS \cdot 127$ code costs a relative $\le 2\times10^{-3}$ of $r^2$ wherever $r^2 \ge 0.1$ (measured worst: $1.8\times10^{-3}$, at $p = 10^{-4}$).
- **Rare, nearly uninformative continuous records** ($r^2 \approx 10^{-3}$): carriers' values sit near $\Delta$, and the fixed scale loses up to about half of an already tiny $r^2$.
- **Fix:** a per-record code scale $s_j = 254/\max_i X_{ij}$. It is exactly invariant by Result 4a, since a per-column scale is a no-op after standardisation.
  - Its loss is $\le 10^{-3}$ wherever $r^2 \ge 0.1$, $\le 1.7\times10^{-2}$ at every level tested, and never worse than the fixed scale.
  - The int8 kernels are unchanged; only $s_j$ is stored with the record's sums.
- **So storage is not where information is lost.** The imputer's output mechanism (Result 7) is.

## 6. The optimal target score (questions b and d)

**Result 9.** $\beta$ is independent of the target's genotype given the training data, and $y_t$ is linear in $G_t$. So, exactly,

$$E[y_t \mid M_t, \mathcal D] = c_t^\top\hat\alpha + \sum_j E[\beta_j \mid \mathcal D]\; E[G_{tj} \mid M_t].$$

Under squared loss, the Bayes score is linear in the target's own calibrated conditional means, with posterior-mean effects in true-genotype units.

- **(i) Same measurement in training and target:** every column construction is valid, and the best is the one with the most information, $X_t$.
- **(ii) Different measurement in the target:** e.g. a long-read-genotyped person, another ancestry, or another imputation. The coefficients have to be in $G$ units, and the target supplies its own $E[G \mid M_t]$.
  - A model trained on draw-like columns has leaked SV credit to the tags (Result 3). It under-weights SVs wherever the tag–SV relation differs from training: truth-genotyped targets, and ancestries with different LD. Portability of SV effects therefore hinges on the Berkson condition in training.
- **(iii) Predictive variance:** $\mathrm{Var}(y_t \mid M_t, \mathcal D) = \sigma^2 + \beta^\top V_t\beta + \mathrm{Var}_{\beta\mid\mathcal D}(X_t^\top\beta)$.
  - The middle term is missing from the current predictive, which uses posterior draws of $\beta$ only.
  - It differs by person with imputation quality, which tracks ancestry. Without it, 90% intervals under-cover, most in the poorly imputed group (0.893 and 0.855); with it, both groups are at 0.900 (numerical check T8).
  - $V_{t,jj} = E[G^2 \mid M] - X^2$ is available from the same store pass.
  - For binary traits, it widens the Gauss–Hermite predictive and so changes calibrated probabilities, though not AUC.

## 7. Optimal use of the internal validation panel (question e)

The data are $(y, T, M)$ on $P$ and $(y, M)$ on $I$. With measurement parameters $\phi$ (the calibration of $X$) and effects $\beta$:

$$L(\beta, \phi) = \prod_{P} p(y_i \mid T_i;\beta)\, p(T_i \mid M_i;\phi)\ \times\ \prod_{I} \int p(y_i \mid g;\beta)\, p(g \mid M_i;\phi)\, dg .$$

**Result 10.** Two facts follow:
- The information about $\phi$ in the $I$ rows' $y$ is $O(\beta^2)$, so $\phi$'s posterior is determined by the panel pairs $(T, M)$ to leading order.
- Given $T$, a panel row's $M$ carries nothing about $y$ (by N). So panel rows enter the effect fit through $T$ alone, which the stacked half already does.

**Result 11 (plug-in is exact where it matters).** The imputed-row mean function is $X_i(\phi)^\top\beta$. Whenever $X_i$ is linear in the unknown calibration quantities, $E_\phi[X_i(\phi)] = X_i(E[\phi \mid P])$. So plugging in the posterior mean is exact to first order, and the posterior variance of $\phi$ adds only to $V$ (second order). Two cases:
- the linear $D^*$ map;
- the Rao–Blackwellised column $X_i = x(k_{i1}) + x(k_{i2})$, which is linear in the unknown haplotype-carriage function $x(\cdot)$.

For the latter, the correct plug-in is the posterior mean of $x(k)$ given the panel haplotypes. A copying-kernel (Nadaraya–Watson) estimate with its bandwidth chosen by leave-one-out panel likelihood approximates it. No cross-fitting is needed, because $\phi$ is learned from genotypes, never from $y$. Panel rows' own values must be leave-one-out wherever they are used to estimate reliability or a combination map.

**The proposal: a Rao–Blackwellised SV column.** For every SV or TR record, the imputed rows get
$$X_{ij} = \hat E\big[G_{ij} \mid \text{phased local SNV haplotypes of } i,\ \mathcal H\big],$$
the posterior-mean carriage of each of the person's two haplotypes under a copying model fitted to the panel's haplotypes with its bandwidth learned by leave-one-out likelihood. It is combined with the imputer's GP through a per-record linear map fitted on the panel's leave-one-out values, which recovers any read evidence the imputer had.
- Panel rows keep $T$.
- The column is jointly Berkson (Result 2). It is scaled by $\widehat{SD}(G)$ with no reliability offset (Result 4‴); the equivalent offset form uses Result 4″ with the leave-one-out reliability.
- It is stored with a per-record 8-bit scale (Result 8).

## 8. Large-effect loci (a correction to the working likelihood)

When $\beta_j^2 V_{ij}$ is comparable to $\sigma^2$, as for a single VNTR that explains a large share of a biomarker's variance, (1) is not enough. The exact per-row likelihood is a three-component mixture, $\sum_g P(G = g \mid M_i)\, N(y_i; \dots + g\beta_j, \sigma^2)$.
- It is heteroscedastic, and it is more efficient than OLS on $X$: the variance ratio was 1.25 at $\beta^2\mathrm{Var}(G) \approx 0.3\sigma^2$.
- The ratio goes to 1 for small effects (T6).

For those few loci, the principled term is the exact mixture likelihood, with the EP site for that coordinate given the mixture tilted distribution. The polygenic bulk keeps the Gaussian working likelihood.

## 9. Comparison with the current design

| Current element | Verdict |
|---|---|
| $\log r^2$ offset with coefficient 1 | Exact for both calibrated and draw-like columns (4′). For the stacked column, the offset must be the stacked reliability (4″). |
| Linear $D^*$ with a stratum $\kappa$ | A no-op on a single-process column (4a). In the stacked column it fixes the unit mismatch only on average (5). It cannot restore the information lost to draw noise, or stop the credit leak (3). |
| Stacked long-read half | Optimal use of $T$ (10). |
| Imputed rows as $D^*$ | Sub-optimal wherever $\kappa \approx \sqrt{r^2}$: the rows carry $r_X^4$ of the information where $r_X^2$ is available (7), and leak SV credit to tags (3). |
| "No gain from DS refinement" (§7) | Predicted by 4b for in-span refinements, and by 6 for any function of $(DS, GP)$. The information is in $M$ (haplotypes plus panel), not in the files. |
| Predictive variance from $\beta$ draws only | Missing $\beta^\top V_t\beta$ (9-iii). |
| 8-bit fixed-scale store | Exact for draw-like DS. Loss is $\le 2\times10^{-3}$ for continuous columns with $r^2 \ge 0.1$. A per-record scale brings the worst case to $1.7\times10^{-2}$ (at $r^2 \approx 10^{-3}$) and never does worse (8). |

## 10. Candidate change and how it is decided

**Candidate: a Rao–Blackwellised column for the imputed half of every SV and TR record.** It is the copying-kernel posterior mean given the person's phased local SNV haplotypes and the long-read panel's haplotypes.
- The kernel's mismatch penalty, distance decay and prior weight are learned by leave-one-out likelihood on the panel; there is no hand-set window.
- An optional evidence-ridge recalibration with the imputer's GP is fitted on panel members' leave-one-out values. It is within-population only, because the map does not transfer across ancestries when the imputer is mis-specified.
- The column is scaled by $\widehat{SD}(G)$ with no reliability offset (4‴), stored with a per-record code scale (8), and scored with the $\beta^\top V\beta$ predictive term (9-iii).

**Decided by the neutral benchmarks.** bench-sim's `beagle_rb` arm builds this column from public 1kGP panel haplotypes and the cohort's phased calls only, and scores it against the Beagle arm [semi-real]. Any use on All of Us data is the user's decision. The column's own diagnostic, per stratum on a truth panel, compares three leave-one-out reliabilities:
- $r^2(DS, T)$;
- $r^2(\text{cohort DS averaged by phased local haplotype}, T)$;
- $r^2(\text{RB}, T)$.

| Outcome | Meaning | Action |
|---|---|---|
| RB ≈ haplotype-averaged DS ≫ DS | Monte Carlo draw noise | RB, or the cheaper average |
| RB > average ≈ DS | Systematic imputer error | RB |
| All about equal | Calibrated imputer | Keep DS |

**Sim-only magnitudes**, paired over 20 replicates against the current design (stratum κ, D*, stacked half, log r² offset):

| Imputer | ΔR², imputed targets | SV effect slope against truth, current → RB (truth-genotyped reference) |
|---|---|---|
| One draw (κ 0.45) | +0.0121 ± 0.0009 | 0.53 → 0.81 (0.91) |
| 4-draw average | +0.0085 ± 0.0007 | |
| Confidently wrong | +0.0030–0.0039 | |
| Tempered | +0.0008 | |
| Calibrated | −0.0005 | |

**Theory-exact items that need no benchmark to be correct:**
1. If the current D* design stays, the stacked-column offset is $r^2_{\rm col}$ from 4″, not the imputed half's $\hat r^2$.
2. Add $\beta^\top V_t\beta$ to the predictive variance.
3. Any per-locus κ must be EB-shrunk; raw per-locus κ̂ was unstable.
4. Use the exact mixture likelihood at loci where $\hat\beta^2\,\mathrm{Var}(G\mid M)$ is comparable to $\sigma^2$.
