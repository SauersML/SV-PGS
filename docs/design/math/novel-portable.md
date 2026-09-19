# Portability from the causal model: the ancestry-optimal score and why SVs port

**Status (2026-09-19):**
- **Theory:** every numbered result has a numerical check (C1–C4, table in §6); all pass.
- **Evidence class:**
  - §7 is a *real-mechanism* measurement: real 1kGP haplotypes and GATK-SV calls, with no phenotype and no simulated LD.
  - The phenotype studies in the lane's scratch files are sim-only under the evidence rule. They check the algebra and the code, not accuracy.
- **Adoption:** accuracy claims wait for bench-real (MAGE, leave-one-superpopulation-out), which has the lane's `fit_per_class` / `fit_shared_level` adapters, and for bench-sim.
- **Code:** the prototype (exact blocked Gibbs in primal full-Gram and dual Woodbury forms, with a continuous per-class mixing density fit by Monte Carlo EM) lives with the novel-portable lane's scratch files.

## 0. Setup

**Generative model.**
- Ancestries $a \in \mathcal A$. For a person of ancestry $a$, the true genotypes $G\in\mathbb R^p$ are allele counts at every variant (SNVs, SVs, TR loci), with within-ancestry mean $m_a$ and covariance (LD) $\Sigma_a$.
- The phenotype is $y = (G-m_a)^\top\beta + c_a + e$, where $e\sim N(0,\sigma^2)$.
- **Shared causal effects:** $\beta$ is per allele of the *true* genotype and is the same in every ancestry. That is the biological hypothesis under test; ancestry intercepts $c_a$ absorb mean differences.
- **Observed data:** $O$, the typed SNVs plus imputed SV and TR dosages. The imputation is a function of the person's local haplotypes and a reference panel.
- **Training data:** $\mathcal D$, individual-level, from a mix $t_a$ of ancestries. The target population mix is $m_a$ (fraction per ancestry, overloading $m$ only in §4).

**Loss:** squared prediction error in a target person of ancestry $a$, equivalently $R^2_a = 1 - \mathrm{MSE}_a/\mathrm{Var}_a(y)$.

## 1. The Bayes-optimal score has ancestry-invariant weights

**Theorem 1.** For a target of ancestry $a$ with observations $O$,
$$\hat y^\star_a(O) = E[y\mid O,\mathcal D,a] = \mu^\top\big(E_a[G\mid O]-m_a\big)+c_a,\qquad \mu = E[\beta\mid\mathcal D].$$
The weights $\mu$ are the same for every ancestry. Everything ancestry-specific sits in the target's genotype predictor $E_a[G\mid O]$ and the intercept.

*Proof.*
- Given $\mathcal D$, the effects $\beta$ are independent of a new person's genotypes and noise, whose distribution depends only on ancestry.
- Hence $E[G^\top\beta\mid O,\mathcal D]=E[G\mid O]^\top E[\beta\mid \mathcal D]$. ∎

**Risk decomposition.** Write $\hat G = E_a[G\mid O]$, $V_a = E_a\,\mathrm{Cov}(G\mid O)$ (the genotype uncertainty left after observation), $C_a = \mathrm{Cov}_a(\hat G) = \Sigma_a - V_a$, and $\Sigma_{\rm post}=\mathrm{Cov}(\beta\mid\mathcal D)$. Then:
- **For fixed true $\beta$:**
$$\mathrm{MSE}_a(\beta) = \sigma^2 + \underbrace{\beta^\top V_a\beta}_{\text{genotype loss}} + \underbrace{(\beta-\mu)^\top C_a(\beta-\mu)}_{\text{estimation loss}} .$$
- **Averaged over the posterior (Bayes risk):**
$$\mathrm{MSE}_a = \sigma^2 + \mu^\top V_a\mu + \mathrm{tr}(\Sigma_{\rm post}\Sigma_a).$$

*Proof.* $y-\hat y = (G-\hat G)^\top\beta + \hat G_c^\top(\beta-\mu)+e$. Given $O$ and $\mathcal D$ the three terms are uncorrelated, because $G-\hat G$ has conditional mean 0 and is independent of $\beta$. Take expectations. ∎

**Consequences.**
- *Portability is estimation error made visible by the target's LD.* Training data pin down $\beta$ well along directions where the training Gram is large, and poorly along contrasts among variants that are nearly collinear in training. $\Sigma_{\rm post}$ is large exactly there. A target whose LD separates those variants ($\Sigma_a$ has variance along the contrast) pays $\mathrm{tr}(\Sigma_{\rm post}\Sigma_a)$ for it.
- *No re-weighting of $\mu$ by target ancestry can reduce the Bayes risk.* $\mu$ minimizes $E[(\beta-w)^\top M(\beta-w)\mid\mathcal D]$ for every positive semidefinite $M$ at once, so it is optimal for all targets simultaneously.

## 2. Which variant is causal: the split inside a tagging cluster

Take $k$ variants whose training columns are identical, so they are in perfect LD in all training people. The likelihood depends on their effects only through the sum $S=\mathbf 1^\top\beta$.

**Theorem 2a (one-causal cluster).**
- Suppose exactly one of them, $c$, is causal, with effect $b$, and the prior is $P(c=j)=\pi_j$ independent of $b$. Then the posterior over $c$ equals $\pi$ (the data cannot distinguish them), and $\mu = E[b\mid\mathcal D]\,\pi$.
- Let $R_a$ be the target correlation among the $k$ variants (standardized columns). The cluster's Bayes risk in ancestry $a$ is
$$\mathrm{Var}(b\mid\mathcal D)\,\pi^\top R_a\pi + E[b^2\mid\mathcal D]\,\big(1-\pi^\top R_a\pi\big).$$
- Among all splits $w$, the expected loss per unit $b^2$, $1-2w^\top R_a\pi + w^\top R_a w$, is minimized at $w=\pi$ for *every* $R_a$.
- The second term is the *causal-identity loss*. It is zero in the training ancestry ($R=\mathbf 1\mathbf 1^\top$), and grows as the target's LD breaks the cluster: $1-\pi^\top R_a\pi$ ranges from 0 up to $1-\sum_j\pi_j^2$ when $R_a=I$.
- It vanishes only when $\pi$ is concentrated, i.e. the causal variant is known. It is therefore reduced by:
  - training data from ancestries where the cluster's LD is broken (these make the likelihood distinguish the variants);
  - a prior that puts the right probability on the variant that really is causal.

**Theorem 2b (how a continuous prior splits a large effect).**
- The variants have independent priors $p_j$. Then $E[\beta_j\mid S]$ is the prior's conditional mean given the sum.
  - **Gaussian** $p_j=N(0,u_j)$: $E[\beta_j\mid S]/S = u_j/\sum_k u_k$ exactly, for every $S$.
  - **Subexponential, regularly varying tails** $p_j(x)\sim K u_j^{\alpha/2}|x|^{-(1+\alpha)}$ (scale family $p_j(x)=u_j^{-1/2}p_0(x/\sqrt{u_j})$):
$$E[\beta_j\mid S]/S\;\to\; \frac{p_j(S)}{\sum_k p_k(S)} = \frac{u_j^{\alpha/2}}{\sum_k u_k^{\alpha/2}}\quad (|S|\to\infty).$$
  This is the single-big-jump principle: given a large sum, one component carries it, with probability proportional to its tail density at $S$.
- For the true one-causal model with a common slab and per-variant causal probability $\pi_j$, the split is proportional to the causal odds $\pi_j/(1-\pi_j)$.
- **The regime that statement covers.** With a light-tailed (Gaussian) slab and $|S|$ far beyond the slab scale, configurations with several causal variants take over and the split drifts toward equal sharing. The odds split is the regime where single-causal configurations dominate; the exact mixture is computed in check C3.

**Corollary 2c (SV causal enrichment is shape, not scale).**
- Suppose SVs are more often causal than SNVs, but the size of a causal effect has the same distribution. That is a difference in the *weight* of the prior's upper tail, a change of shape.
- A class scale offset $u_{\rm SV}/u_{\rm SNV}$ moves the split of a large effect only by the power $\alpha/2$ of that ratio. Matching the true odds ratio $\rho$ needs $u$-ratio $\rho^{2/\alpha}$, which then mis-states the SV effect-size distribution everywhere else, since every SV effect is inflated.
- A class-specific mixing density $g_c$ carries both correctly: a heavier upper tail for SVs at the same location.
- So the portable (target-optimal) split requires per-class densities, or class-dependent tail weight. A shared density with class scale shifts is not enough. Any pooling of densities across classes, as proposed by the prior and math-density lanes, must keep class-specific *tail weight*.
- This agrees with lit-pgs's finding that the leading methods let annotations change mixture weights (SBayesRC, GMRM). §2 gives the reason in terms of portability.

## 3. The omitted causal SV: the exact SNP-tagging loss

Take a causal SV $c$ (standardized effect $b$) and a typed SNV tag set $T$ around it.
- For ancestry $a$, the correlations are $r_a=\mathrm{Corr}_a(G_T,G_c)$ and $R_a=\mathrm{Corr}_a(G_T)$. The target-optimal linear tag weights are $w_a^\star = R_a^{-1}r_a$.
- **SNP-only model.** With within-ancestry centering and large $n$, it converges on the tags to the training-mix projection $w_{\rm tr}=(\sum_a t_aR_a)^{-1}\sum_a t_a r_a$.

**Theorem 3 (tagging loss).** Per unit $b^2$, the target loss of the SNP-only model in ancestry $a$ is
$$\ell^{\rm SNP}_a = \underbrace{\big(1-r_a^\top R_a^{-1}r_a\big)}_{\text{best linear tagging loss in }a} + \underbrace{(w_{\rm tr}-w_a^\star)^\top R_a(w_{\rm tr}-w_a^\star)}_{\text{LD-mismatch } M_a}.$$
*Proof.* Pythagoras in $L^2_a$: $G_c-w^\top G_T = (G_c-w_a^{\star\top}G_T)+(w^\star_a-w)^\top G_T$. The first part is orthogonal to $G_T$ by the normal equations. ∎

$M_a=0$ for the ancestry that dominates training, and it is positive wherever the LD pattern differs.

**SV-inclusive model.**
- **Sufficiency condition.** Let $D_c$ be the imputed dosage, and suppose it is calibrated and *sufficient in every ancestry*: $D_c=E_a[G_c\mid H]$, with $H$ the local haplotypes, of which the tags are a function. Then $E_a[G_c\mid D_c,G_T]=D_c$ for every $a$.
  - The target-optimal weights are $(1,0)$ in every ancestry, so the mismatch vanishes identically.
  - The loss is the imputation loss $\ell^{\rm SV}_a = 1-r^2_{D,a}$, where $r^2_{D,a}=\mathrm{Corr}_a^2(D_c,G_c)$.
- **Gain per causal SV in ancestry $a$:**
$$\Delta_a = b^2\Big[\big(r^2_{D,a}-\rho^2_{T,a}\big) + M_a\Big],\qquad \rho^2_{T,a}=r_a^\top R_a^{-1}r_a .$$
  - *Haplotype term:* $r^2_{D,a}\ge\rho^2_{T,a}$ whenever the panel represents ancestry $a$, because $E_a[G_c\mid H]$ dominates every linear function of $G_T$ in $L^2_a$.
  - *Mismatch term:* $M_a$ is zero in the training-majority ancestry and positive elsewhere.
- **Prediction:** SVs add accuracy in every ancestry, and *more* in ancestries under-represented in training. So SV-inclusive scores port better.
- **When the gain can be negative:** only if $r^2_{D,a} < \rho^2_{T,a}-M_a$, i.e. the panel misses ancestry $a$'s haplotypes. Sufficiency then also fails, and part of the mismatch returns.
- **Corollary 3a (one tag, closed form).** With a single tag whose correlation with the causal SV is $r_{\rm tr}$ in training and $r_a$ in the target:
- $w_{\rm tr}=r_{\rm tr}$ and $w_a^\star=r_a$, so $M_a=(r_{\rm tr}-r_a)^2$ and $\ell^{\rm SNP}_a = 1-2r_ar_{\rm tr}+r_{\rm tr}^2$.
- The SNP-only locus's portability ratio is
$$\frac{R^2_a}{R^2_{\rm tr}}=\frac{2r_a r_{\rm tr}-r_{\rm tr}^2}{r_{\rm tr}^2}= \frac{2r_a}{r_{\rm tr}}-1 .$$
  It is linear in the LD ratio, and becomes negative (the locus *hurts*) once $r_a<r_{\rm tr}/2$.
- With a sufficient dosage of reliability $r^2_{D,a}$, the SV-inclusive locus's ratio is $r^2_{D,a}/r^2_{D,{\rm tr}}$, independent of the tag LD.
- LD decays faster in African-ancestry haplotypes, so $r_a<r_{\rm tr}$ is typical for EUR-trained tags scored in AFR. Every such locus is a place where the SNP score leaks accuracy and the SV score does not.

**Imperfect sufficiency.** If the imputed dosage is noisy but calibrated (a posterior draw, $\kappa<1$), the tags keep residual information about $G_c-D_c$. That residual regression is ancestry-specific, so a mismatch term of the same form, on the residual, reappears. Calibrating the dosage within each ancestry ($D^*=\mu+\kappa(DS-\mu)$, per ancestry as in MODEL.md §1) removes the calibration part but not the insufficiency.

## 4. When one model is optimal for every ancestry

**Theorem 4.**
- **(a) Well specified.** If the model contains the causal variants and its prior is right, $\mu=E[\beta\mid\mathcal D]$ is Bayes-optimal in every ancestry at once (Theorem 1).
  - Importance-weighting the likelihood to the target mix cannot help.
  - It costs precision: the weighted posterior behaves like one with effective sample size $n_{\rm eff}=(\sum_i w_i)^2/\sum_i w_i^2<n$, so the estimation loss grows.
- **(b) Misspecified** (causal variant omitted). The best single linear predictor for a target mix $m$ is
$$w_m=\Big(\sum_a m_aR_a\Big)^{-1}\sum_a m_a r_a,$$
  which is exactly the least-squares fit with training people weighted by $m_a/t_a$.
  - One weight vector is optimal for all ancestries iff $r_a=R_a w$ for a common $w$. That holds iff the omitted variant has the *same linear regression on the tags in every ancestry*, which generic LD differences violate.
- **(c) What the evidence should weight.** The ordinary, unweighted evidence of the most complete causal model:
  - The target mix should enter the fit only through the scoring-time genotype predictor $E_a[G\mid O]$.
  - The value of reweighting is a direct measure of residual misspecification from omitted causal variants.
  - It should help a SNP-only model in under-represented ancestries (repairing part of $M_a$), cost it in the majority ancestry, and help an SV-inclusive model less.

## 5. What this asks of the one model
1. **Keep the weights ancestry-invariant; make the genotype predictor ancestry-specific.** The score is $\mu^\top E_a[G\mid O]$. For SVs and TRs that means per-ancestry calibrated dosages at scoring (MODEL.md already recalibrates κ per ancestry). Standardization uses the training scale, never the target's.
2. **Per-class densities must keep class-specific tail weight** (Corollary 2c). A shared density with class scale offsets alone cannot give the portable split.
3. **No target-mix reweighting of the evidence** (Theorem 4a, c).
4. **Imputation sufficiency is the SV-specific portability lever.** The gain has two parts, $(r^2_{D,a}-\rho^2_{T,a})$ and $M_a$, and both require the panel to represent ancestry $a$.
5. **Report per-ancestry accuracy and the portability ratio $R^2_a/R^2_{\rm EUR}$** as SV-credit metrics. The theory predicts SV credit is largest outside the training-majority ancestry.

## 6. Numerical checks
| Result | Check | Outcome |
|---|---|---|
| Theorem 1 risk identity | C1: conjugate model with noisy target genotypes, 4,000 Monte Carlo replicates | realised MSE 1.435 ± 0.033 vs predicted Bayes risk 1.462 |
| Theorem 2a split and risk | C2: closed form vs 400k draws | 0.06088 vs 0.06085. Per unit b², the split w = π has loss 0.318, against 0.416 for an equal split and 0.460 for the lead variant only |
| Theorem 2b, u-ratio 4 | C3: log-space quadrature | Gaussian 0.800 at every S. t₃ goes 0.73 → 0.888 as S = 2 → 128 (limit 8/9). Cauchy 0.6667 (2/3). Spike-and-slab 0.816 at S ≤ 1, which is the exact mixture with its both-causal term (odds alone: 0.826) |
| Theorem 3 decomposition | C4: msprime out-of-Africa, 77 SVs, 100 nearest tags | identity error ≤ 0.008, from a 1e-3 ridge |

## 7. Theorem 3 on real haplotypes (real-mechanism)
**Setup.**
- 112 common (MAF ≥ 1%) GATK-SV SVs from the 1kGP high-coverage phased panel, in 64 windows of 250 kb on chr21 and chr22.
- A kNN haplotype-matching imputer on the 64 nearest typed sites, with a panel of ⅓ of each superpopulation's unrelated samples (861). Everyone scored is held out of the panel.
- Each SV is assumed causal, with a shared per-allele effect. The table gives the median fraction of its genetic signal each route loses.

| Route | EUR | AFR | EAS | SAS | AMR |
|---|---|---|---|---|---|
| EUR-trained best single SNV/indel tag | 0.187 (in-sample) | 0.562 | 0.207 | 0.203 | 0.265 |
| EUR+AFR+EAS-trained GCV-ridge tag model (in-sample for those three) | 0.169 | 0.190 | 0.132 | 0.183 | 0.196 |
| imputed SV dosage, raw | 0.192 | 0.285 | 0.170 | 0.188 | 0.193 |
| imputed SV dosage, per-ancestry calibrated | 0.174 | 0.282 | 0.156 | 0.163 | 0.177 |

**Per-SV paired gain of the calibrated imputed dosage** (median, with the share of SVs where it is positive):
- **vs the EUR-trained tag:** EUR −0.001 (38%), AFR +0.085 (83%), EAS +0.007 (73%), SAS +0.009 (74%), AMR +0.020 (78%).
- **vs the multi-ancestry tag model:** it ties or loses in the training ancestries (AFR −0.025, EAS −0.001, EUR −0.003) and wins in the ancestries absent from training (SAS +0.010, AMR +0.006).

**Single-tag loss by quartile of the LD difference |r_EUR − r_a|:**
- AFR: 0.07, 0.38, 0.89, 1.00
- EAS: 0.03, 0.08, 0.52, 1.00
- SAS: 0.02, 0.16, 0.35, 0.96
- AMR: 0.02, 0.14, 0.40, 0.99

This is Corollary 3a's monotone dependence.

**Reading.**
- Everything Theorem 3 predicts appears on real LD:
  - no SV advantage in the training ancestry;
  - an advantage wherever the LD differs;
  - the advantage removed when the SNP model trains on that ancestry;
  - but kept where training lacks the ancestry.
- Per-ancestry calibration of the dosage, i.e. using E_a[G|D] as the target predictor, cuts imputation loss by 0.003–0.025.
- The imputer here (r² ≈ 0.72–0.84) is far weaker than GLIMPSE2 with the production long-read panel, so the haplotype term is understated.

## 8. An inference pitfall found on the way
In LD space, a block-diagonal or banded fit ignores the cross-Gram between unlinked blocks. Chance correlation between them, O(1/√n) per pair and summed over many pairs, then double-counts signal. The LD-space residual yᵀy − 2βᵀXᵀy + Σ_b β_bᵀG_bbβ_b is biased low.
- **Measured:**
  - with 8 independent blocks, n = 3,000 and p/n = 0.27, the EM noise variance went to 0.05 against a truth of 1;
  - at p/n ≈ 2.8 the fit collapsed into interpolation.
- **With the exact full Gram:** f = Gβ kept current per block and RSS = yᵀy − 2βᵀXᵀy + βᵀf gave σ² = 0.998 on the first case and 0.588 (truth 0.6) at p/n = 1.5.
- **So:** the noise variance must come from a full-data residual, never from block-diagonal statistics.
