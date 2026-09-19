# SV genotype function: what the model should regress on, from first principles

Derived from a molecular model of how a variant changes a phenotype, and from a measurement model of how imputation reports the genotype. Every identity and bound is checked numerically; the checks and the prototype live in the novel-svfunction lane directory (`proto/verify_identities.py`, checks V1–V8; `proto/submission.py` is the benchmark entry point).

## 0. The answer in brief

1. **Cis-additivity (Theorem 1).** When a variant acts in cis (it changes the output of the molecular unit on its own chromosome), the trait is, to first order in the molecular effect, a sum over haplotypes of a per-allele function.
   - Every locus is then additive in allele counts, for every variant class.
   - Nonadditivity appears only at second order, through the curvature of the map from molecular output to phenotype.
   - The dominance ratio is d/a = −½ (H''/H') Δ: it grows linearly with the molecular effect Δ.
   - Consequences:
     - dominance is negligible for the polygenic SNV bulk, and can matter only for large-effect variants, meaning whole-gene dosage SVs;
     - all second-order nonadditivity of a single-pathway trait (within-locus dominance and cross-locus epistasis) is the quadratic term of **one learned link** of the additive index, y = h(A). That puts dominance at d_j = −½ (h''/h'²) a_j², one parameter per trait, not one per variant.
2. **Exact Bayes predictor under imputation (Theorem 2).** For any haplotype-additive truth, E[F(G) | data] is linear in the calibrated allele dosages, whatever the per-allele function is.
   - So any nonlinear function of an allele's quantity (repeat length, copy number) must be applied per allele before taking the expectation. That means columns Σ_a b(q_a) D_a, never b(Σ_a q_a D_a), whose plug-in bias is ½ b'' Var(q | data).
   - Any function of a biallelic genotype is exactly linear in (DS, GP1).
3. **Ordered-quantity loci (Theorem 3).** At a tandem-repeat or copy-number locus, alleles carry an ordered physical quantity q, and the truth is Σ_h φ(q_h). The optimal representation is φ as a continuous function of q:
   - the prior is a Gaussian-process/spline prior with roughness ∫ φ''(q)² dq and a class-level learned weight λ;
   - the linear part keeps the existing effect prior.
   - Its null space is {1, q}. At λ → ∞ it is **exactly** the current signed-length column Z = Σ_a q_a D_a; at λ → 0 it is the per-allele columns.
   - The current design's asymptotic loss is exactly 2 Var_f(φ_⊥), the allele-frequency-weighted variance of φ's non-linear part.
   - This explains both MODEL.md §7 results: per-allele columns pay about A − 1 parameters of estimation noise for a small 2 Var_f(φ_⊥), and the linear column pools all alleles into one parameter.
   - GATK-SV copy-number loci are already linear in total CN, which is the λ = ∞ case. Their non-linear part is the curvature of the phenotype map in total dosage, and the same smooth basis applies.
4. **Mechanism sharing (Theorem 5).** To first order, the per-allele effect of any variant that changes the dosage or function of a molecular unit m is c_vm θ_m, where:
   - θ_m is the effect of one extra functional copy of m;
   - c_vm is fixed by the variant's molecular consequence: −1 for a whole-unit deletion or a disruption, +ρ for a whole-unit duplication (ρ = 1 + O(curvature)), and so on.
   - So all variants hitting the same gene or element share one effect. The prior is β_v = Σ_m c_vm θ_m + ε_v: an aggregate column U_m = Σ_v c_vm D_v per unit, with its own learned prior variance, beside the variant columns.
   - The TR signed-length column is the special case where the unit is the repeat and c = Δlen. The sharing helps exactly when several low-information variants hit one unit, and costs nothing when they don't (the prior variance of θ goes to zero).
5. **Dominance information (Proposition 4).** After projecting out the allele count, the heterozygote column has variance (2pq)², against 2pq for the additive column.
   - So dominance carries 2pq times less information than the additive effect, and is estimable only for common variants.
   - It needs GP1, which the 8-bit dosage store does not keep.

## 1. Setup and notation

- **Individuals and loci:** individual i, locus j, haplotypes h ∈ {1, 2}. Allele a ∈ 𝒜_j sits on haplotype h with indicator x_{ihja}; the allele count is n_{ija} = Σ_h x_{ihja}.
- **Stored column:** D_{ija} ≈ E[n_{ija} | data_i], after background removal and the per-stratum linear recalibration D* (MODEL.md §1). "Calibrated" means D_{ija} = E[n_{ija} | data_i].
- **Molecular units** m = 1..M: genes, enhancers, insulators, repeat tracts. Haplotype h of individual i has output e_{ihm} (expression, functional protein amount, activity), and the unit's total is E_{im} = Σ_h e_{ihm}.
- **Trait:** y_i = H(E_{i1}, …, E_{iM}) + ε_i, where H is smooth and includes all trans and network effects.
- **Cis action:** a variant is cis-acting if allele a on haplotype h changes only that haplotype's outputs, log e_{ihm} = log e⁰_m + Σ_j δ_{mj}(a_{ihj}).

## 2. The molecular model implies cis-additivity

**Theorem 1 (first-order haplotype additivity).** Let H be twice continuously differentiable around the population-mean outputs Ē. Then
$$y_i = \bar y + \sum_j \sum_a \varphi_j(a)\, n_{ija} + R_i + \varepsilon_i,\qquad \varphi_j(a) = \sum_m \partial_m H(\bar E)\,\Delta_{mj}(a),$$
- The haplotype output change Δ_{mj}(a) is defined in the proof.
- The remainder R_i is a quadratic form in the output deviations E_{im} − Ē_m, with coefficients ½ ∂²_{mm'} H. It carries all dominance and epistasis.

*Proof.* Take a single variant j acting on unit m. Its haplotype output is e_{ihm} = e⁰_m exp(δ x_{ih}), with x_{ih} ∈ {0, 1}. Since x² = x, this equals exactly e⁰_m (1 + (e^δ − 1) x_{ih}). So E_{im} = E⁰_m + Δ g_{ij}, with Δ = e⁰_m (e^δ − 1) and g the allele count. That is linear in g with no approximation: nonlinearity within a haplotype is absorbed into Δ.

With many cis variants on the same haplotype, the exponent is a sum over them. Expand exp(Σ_j δ_j x_j) to first order: the cross terms are products δ_j δ_k x_j x_k of molecular effects, and so second order. Taylor-expand H around Ē. The first-order term is Σ_m ∂_m H (E_{im} − Ē_m) = Σ_j Σ_h Σ_m ∂_m H Δ_{mj}(a_{ihj}) + const, and summing over h gives the allele-count form. Everything left over, including the within-haplotype cross terms, is quadratic in the output deviations. ∎

**Corollary 1a (dominance grows with the molecular effect).** For one biallelic variant acting on one unit, F(g) = H(E⁰ + Δ g). The dominance deviation and the additive effect are
$$d = F(1) - \tfrac12\big(F(0)+F(2)\big) = -\tfrac12 H''(\xi)\,\Delta^2,\qquad a = \tfrac12\big(F(2)-F(0)\big) = H'(\zeta)\,\Delta,$$
for some ξ, ζ ∈ [E⁰, E⁰ + 2Δ]. The d formula is the mean-value form of the second difference, and a is the mean-value theorem. Hence
$$\frac{d}{a} = -\frac12\,\frac{H''(\xi)}{H'(\zeta)}\,\Delta,\qquad \Big|\frac{d}{a}\Big| \le \frac12\,\frac{\sup|H''|}{\inf|H'|}\,|\Delta| .$$
Verified: check V2a (the limit −½ (H''/H') Δ) and check V2b (the bound, with no violations over 60 maps × step sizes).

**Corollary 1b (where dominance lives).** At a locus under Hardy–Weinberg equilibrium (HWE), V_A = 2pq [a + d(q − p)]² and V_D = (2pq d)². So
$$\frac{V_D}{V_A} = \frac{2pq\,(d/a)^2}{\big(1+(d/a)(q-p)\big)^2} \approx \tfrac14\,2pq\,\kappa_H^2\,\Delta^2,\qquad \kappa_H = H''/H'.$$
- **SNVs:** a regulatory SNV typically changes expression by a few percent, Δ ≪ E⁰, so V_D/V_A is second order and negligible.
- **Whole-gene deletions:** these remove a full copy, Δ = −e⁰ = −E⁰/2, so d/a = ¼ κ_H E⁰ to leading order, which is O(1) wherever the map saturates.
  - Under a Michaelis–Menten-type map H(E) = E/(E + K), κ_H = −2/(E + K), so d/a → −E⁰/(2(E⁰ + K)).
  - That is additivity when E⁰ ≪ K, and substantial recessivity of the deletion when E⁰ ≫ K.
  - At a full-copy Δ the leading-order value is not accurate. The exact mean-value form gives complete recessivity, d/a → −1, when E⁰ ≫ K, since F(0) ≈ F(1) ≈ 1 and F(2) = 0.
- **So nonadditivity is an SV phenomenon** in this model: it lives with large molecular effects, which in the genome are copy-number changes of whole functional units.

**Corollary 1c (one link carries all second-order nonadditivity).** Suppose the trait depends on the outputs through one pathway index, H(E) = h(Σ_m w_m E_m). Then y = h(A) + O(3), with the additive index A_i = Σ_j Σ_a φ_j(a) n_{ija} / h'.
- Expanding to second order, y ≈ h'A + ½ h'' (A − Ā)². The single quadratic term contains:
  - every locus's dominance, d_j = −½ (h''/h'²) a_j² (verified: check V7, averaged over the background index);
  - every pair's multiplicative epistasis, h'' a_j a_k / h'².
- All dominance of a single-pathway trait therefore shares **one sign and one constant**, the relative curvature of the trait scale. So a per-variant dominance effect with its own prior variance is the wrong parametrization: it spends one parameter per variant on a quantity that the theory ties to a_j² with a single constant.
- The right parametrization is a smooth monotone link h, learned by type-II ML, with roughness ∫ h''² and the identity as its null space (λ = ∞ gives the additive model).
- A trait with P pathways gives a P-index model; P is small for the traits we fit.
- **What the link means in practice:** the most common source of h'' is the scale on which the trait was recorded (a raw biomarker vs its logarithm). The learned link is the trait's natural genetic scale, learned rather than chosen by hand.

**The measurement correction.** Under imputation, E[h(A) | data] = h(E[A | data]) + ½ h'' Var(A | data) + O(3), with Var(A | data) = Σ_j a_j² Var(g_j | data) for loci measured independently. The correction is largest in individuals who carry poorly imputed, large-effect variants, which is typically SVs. It uses the per-person genotype variance, which again needs GP (see §5).

## 3. The Bayes predictor under imputation

**Theorem 2 (linearity).** Let F be haplotype-additive, F(G) = Σ_j Σ_a φ_j(a) n_{ja}. Then for every individual,
$$\mathbb E[F(G)\mid \text{data}] = \sum_j \sum_a \varphi_j(a)\,\mathbb E[n_{ja}\mid\text{data}],$$
which is exactly linear in the calibrated allele dosages for any φ.

For a biallelic locus with an arbitrary genotype function F(g) = F(0) + a′ g + d 1[g = 1], the predictor is E[F | data] = F(0) + a′ DS + d GP1: exactly linear in (DS, GP1). *Proof:* linearity of expectation. ∎

**Consequence for columns.** A nonlinear function b of an allele quantity must be applied per allele, before taking the expectation: the column is Σ_a b(q_a) D_a.
- The plug-in alternative b(Σ_a q_a D_a / 2), using the expected length, is biased by E[b(q)] − b(E[q]) = ½ b''(E q) Var(q | data) + O(E|q − Eq|³). Verified: check V4.
- For tandem repeats, imputation confuses adjacent lengths, so Var(q | data) is large, and so is this bias.

**Calibration.** Imputed SV and TR dosages behave like confident draws (κ ≈ √r²), so D_{ja} ≠ E[n_{ja} | data]. The stored D* recalibrates each record linearly to E[n | DS]. Every column in this document is linear in the per-record D*, so each column inherits the recalibration exactly. Summing recalibrated records per locus is the correct construction; recalibrating the sum is not.

## 4. Ordered-quantity loci: tandem repeats and multi-allelic copy number

**Setup.** At locus j the alleles carry quantities q_a: the repeat-length difference from REF in repeat units, or the copy-number change ΔCN. By Theorem 1 the truth is Σ_h φ(q_{a_h}) with φ(0) = 0; the constant is absorbed, since Σ_a n_a = 2.

**Prior.** Split φ(q) = θ q + φ_⊥(q):
- θ keeps the model's existing continuous scale-mixture prior;
- φ_⊥ gets the Gaussian prior with density proportional to exp(−∫ φ_⊥''(q)² dq / (2τ²)), where τ² = 1/λ is learned per class by type-II ML and pooled across all loci of the class.
- The penalty's null space is {1, q}, so φ_⊥ carries only the non-linear part and θ stays identified.

**Representation (exact, finite).** A locus has finitely many distinct allele quantities t_1 < … < t_A. Among all functions with given values at those points, the natural cubic interpolant minimizes ∫ φ''². So the prior on the allele values is the Gaussian with precision Ω/τ², where Ω = Q R⁻¹ Qᵀ is the natural-spline roughness matrix.
- Checks: φᵀΩφ equals ∫ s''² for the natural cubic interpolant (V6a), and Ω1 = Ωq = 0 (V6b).
- Diagonalize Ω = Σ_k e_k u_k u_kᵀ, with k running over the A − 2 non-null eigenvectors. Then φ = θ q + Σ_k ω_k u_k / √e_k with ω_k ~ N(0, τ²).
- So every locus gets the signed column Z_j = Σ_a q_a D_{ja}, plus A − 2 columns W_{jk} = Σ_a (u_k(a)/√e_k) D_{ja} that all share one class-level variance τ².
- No grid, knots or basis size is chosen. The basis is fixed by the locus's own allele set, and the smoothness is learned. That meets SPEC 131b205 (a continuous function, with a learned smoothness).

**Theorem 3 (nesting and exact loss).**
1. **λ → ∞ (τ² → 0):** the W columns drop out, and the model is exactly the current design's signed-length column.
2. **λ → 0:** φ is free at every allele, which is the per-allele column model. The ridge on each allele coefficient then comes from the scale prior on θ alone; the "linear + exchangeable" hybrid adds an exchangeable prior on the deviations instead.
3. **With infinite samples,** the R² that the linear column loses relative to the true φ is exactly
$$\mathcal L_{\text{lin}} = 2\,\mathrm{Var}_f(\varphi_\perp),\qquad \varphi_\perp = \varphi - \mathbb E_f\varphi - \theta^*(q - \mathbb E_f q),\quad \theta^* = \frac{\mathrm{Cov}_f(\varphi,q)}{\mathrm{Var}_f(q)},$$
   with f the allele frequencies under HWE (verified: check V3, 200 random loci, maximum relative error 10⁻⁹).
4. **With n samples,** a model with effective degrees of freedom df pays about df · σ²_e / n of prediction variance per locus under a flat prior (less with shrinkage):
   - the linear column pays df = 1;
   - per-allele columns pay df ≈ A − 1;
   - the smooth model pays df_λ ∈ [1, A − 1], chosen by the evidence to balance 2 Var_f(φ_⊥) against the variance cost.

**Why MODEL.md §7 came out as it did.**
- **Per-allele TR columns were neutral or negative:** 2 Var_f(φ_⊥) was small next to (A − 2) σ²_e / n. Most TR loci in the panel have A between 3 and 15 alleles, and the estimation cost of about A − 1 parameters per locus cancelled a small non-linear gain.
- **The linear column gained 4–12%:** it has the variance of one parameter and captures 2 Var_f(φ) − 2 Var_f(φ_⊥).
- **Sibling-allele coupling (GR-2) beyond the locus model brought no gain:** once a locus is summarized by its signed length, the only coupling left between sibling alleles is φ_⊥. The only principled coupling left is smoothness in q, and its gain is bounded by 2 Var_f(φ_⊥).

**Robustness to adjacent-length misassignment.** Suppose imputation reports a neighbour allele a′ with probability ε_adj. Then the per-person error in a column is ε_adj |φ(q_a) − φ(q_{a′})| per haplotype.
- For a smooth φ this is about ε_adj |φ′| |q_a − q_{a′}|: the error is small, and it shrinks with the smoothness.
- For free per-allele coefficients, the estimated differences β̂_a − β̂_{a′} also carry the estimation noise of both coefficients. So the per-allele model is hit twice: once by misassignment and once by noise. The smooth model is hit once.

**Units.** The penalty ∫ φ''(q)² dq is not invariant to rescaling q, so with a class-level λ the unit matters. The mechanism acts per repeat copy (binding motifs, spacing) or per gene copy, so q is taken in repeat units (Δlen / motif length) and in copies (ΔCN).
- The alternative is locus-relative units q / SD_f(q). Under an exchangeable-locus prior this is also defensible, and the evidence can choose between the two per class. Nothing is set by hand either way.

**Copy-number loci.** GATK-SV multi-allelic CNVs are already stored as the integer total copy number, class `COPY_NUMBER` (`gatksv_store_rows.py`). That is the λ = ∞ representation, linear in CN, which Theorem 1 says is correct to first order.
- A total-CN column carries no per-haplotype split. So its non-linear part F(CN_total) − linear is not a cis per-allele function: it is the curvature of H in the unit's total output, E = e⁰ CN_total. This is Corollary 1b with a very large Δ, since copy numbers range over many copies.
- The derived extension is the same smooth basis applied to the total, W_k = b_k(CN), with the class-level λ learned. The CN is a hard call whose error is handled by the two-source fusion, so the column is b(CN_call), not an expectation.
- Imputed non-TR multi-allelic bubbles keep exchangeable allele columns (MODEL.md §1), and that is right unless their alleles differ only by the copy count of one sequence. Alleles with different inserted sequences have no order for a smooth prior to use.

**Prior work in this repository.** The earlier design-trlocus study (its lane report, §1.2; not in the repository) derived Theorem 2's linearity lemma first.
- It tested a non-linear length basis built from hinge functions at the 75th and 90th length quantiles plus a quadratic, each with its own EB variance offset. That basis gained +1.4–2.3% where the truth was non-linear and cost 0–2% elsewhere.
- The representation here differs in two ways that follow from the derivation:
  - it has no knots chosen by quantile: the basis is the natural-spline eigenbasis of the locus's own allele set;
  - it adds exactly one hyperparameter per class, λ, whose λ = ∞ limit is the linear column.
- So this representation should keep the gain without the cost. The benchmarks answer that question (§10).

## 5. Dominance: identifiability and information

**Proposition 4.** Under HWE, the part of the heterozygote indicator 1[g = 1] not explained by a linear regression on g has variance
$$\mathrm{Var}(1[g{=}1]) - \frac{\mathrm{Cov}(g,1[g{=}1])^2}{\mathrm{Var}(g)} = 2pq(1-2pq) - \frac{(2pq(q-p))^2}{2pq} = (2pq)^2,$$
using (q − p)² = 1 − 4pq. Verified: check V1, 199 frequencies, maximum error 10⁻¹⁰.

- **Information ratio:** a dominance coefficient therefore carries 2pq times less information than the additive coefficient. That ratio is 0.02 at p = 0.01 and 0.5 at p = 0.5.
- **Expected R² gain** from an EB-shrunk dominance coefficient with prior variance τ_d², per locus, is (v τ_d²)² n / (σ² + n v τ_d²), where v = (2pq)² r²_het.
- **When it is worth estimating:** only when n (2pq)² r²_het τ_d² ≳ σ², meaning common variants with large dominance. By Corollary 1b, that means common variants with large molecular effects: common whole-gene copy-number polymorphisms.
- **The store:** the 8-bit dosage store keeps DS only, and GP1 is not recoverable from DS without a model. Per-variant dominance columns, and the measurement correction of §2, would need GP1 stored for the few common copy-number SVs. The link of Corollary 1c needs no GP1 for its main term; only its variance correction does.

## 6. Mechanism sharing: one dosage effect per molecular unit

**Theorem 5 (shared dosage effect).** From Theorem 1 at first order, the per-allele effect of variant v is β_v = Σ_m ∂_m H · Δ_{mv}. Write Δ_{mv} = e⁰_m c_{vm}, where c_{vm} is the change in functional copies of unit m on the carrying haplotype. With θ_m = ∂_m H e⁰_m, the effect of one extra functional copy of m,
$$\beta_v = \sum_m c_{vm}\,\theta_m + \varepsilon_v ,$$
where ε_v collects the variant's own effects that are not dosage effects. The molecular consequence fixes c_{vm}:

| variant | c_vm | why |
|---|---|---|
| deletion containing all of unit m | −1 | one copy fewer |
| duplication containing all of unit m | +ρ | ρ = 1 if the extra copy is functional and H is locally linear; ρ = 1 + O(κ_H e⁰) otherwise |
| SV (DEL, INS, INV, MEI) breaking the coding sequence or unit | −1 | loss of function of that copy (NMD) |
| LoF SNV or indel (stop-gain, frameshift, essential splice) | −1 | the same consequence class as a disruption |
| variant not changing m's copies or function | 0 | no dosage effect; only ε_v |

**The two structural predictions.**
1. A deletion and a duplication of the same gene have effects of equal size and opposite sign, to first order.
2. Every loss-of-function event in a gene has the same effect, whether it is an SV, a SNV or an indel.

Neither statement is available to a per-variant prior, however well its scale is modelled.

**Representation.**
- Add one aggregate column per unit, U_m = Σ_v c_{vm} D_v, with coefficient θ_m under the model's continuous scale-mixture prior. Its scale model uses unit-level annotations, such as dosage sensitivity, and is symmetric between SNV-LoF and SV contributions.
- Keep every variant column with its own prior for ε_v. The two variances are learned by type-II ML, and ρ by one-dimensional profile ML. ρ is a derived quantity whose value under linear dosage is 1, so it is learned, not set.
- The TR signed-length column is the same construction: the unit is the repeat tract, and c_{va} = q_a.

**Identifiability.** With a single variant per unit, θ_m and ε_v enter only through their sum. The pair then behaves like a single Gaussian with variance τ_θ² + τ_ε², which is harmless. The split is identified by units with two or more variants (the EB pools across units), and by the sign constraint between DEL and DUP.

**Bayes gain (closed form).** Take K variants hitting one unit, with sampling variances s_k² = σ² / (n v_k) and v_k = 2p_k q_k r_k². Compare the shared prior, with covariance τ_θ² 11ᵀ + τ_ε² I, against an independent prior with the same marginal variance. The prediction risk of each is:
$$\text{shared: } \sum_k v_k\big[(\Sigma^{-1}+S^{-1})^{-1}\big]_{kk},\qquad \text{independent: } \sum_k v_k\,\frac{(\tau_\theta^2+\tau_\varepsilon^2)\,s_k^2}{\tau_\theta^2+\tau_\varepsilon^2+s_k^2}.$$
(Verified: check V5, closed form against 200,000-draw Monte Carlo, relative error below 10⁻².)
- The gain is largest when every s_k² ≫ τ²: that is rare, poorly imputed variants, each individually uninformative, which pool into the unit's θ.
- The gain vanishes when K = 1 or τ_θ² → 0.

**Corollary 5a (the scale of θ_m follows from selection on dosage; a derived offset, not a feature).** Suppose fitness is Gaussian stabilizing selection on the T traits, w(z) ∝ exp(−Σ_t z_t² / (2V_s)), with z in units of phenotypic SD.
- A heterozygous copy change of unit m shifts trait t by θ_{m,t}. To first order its selection coefficient is s_het,m = Σ_t θ_{m,t}² / (2V_s): the mean fitness loss of a shift δ under Gaussian stabilizing selection is δ²/(2V_s) when V_s ≫ 1.
- For exchangeable traits, therefore, E[θ_{m,t}²] = 2V_s s_het,m / T. On the log scale:
$$\log \tau^2_{\theta,m} = \text{level}_t + \log s_{\text{het},m},$$
  with the coefficient on log s_het **exactly 1**, just as the r² offset's coefficient is exactly 1 in MODEL.md §3. Only the level is learned.
- **Estimating s_het:** at mutation–selection balance for a strongly selected heterozygous effect, the unit's aggregate LoF allele frequency is q_m ≈ μ_m / s_het,m. So ŝ_het,m = μ̂_m / q̂_m, computed from the LoF and SV allele counts of the unit. That is a quantity counted in SNVs and SVs alike, and so symmetric between them.
- **Frequency flattening:** combining the two relations, the unit's expected contribution to trait variance, 2 q_m θ_m² ∝ μ_m, does not depend on s_het. The per-variant consequence β² ∝ 1/p is the α = −1 end of the learned frequency term (1 + α_c) log 2pq already in the scale model.
- The theory therefore predicts α ≈ −1 for dosage-altering variants and a larger α for weakly selected regulatory SNVs. That is a measurable, SV-specific prediction.
- **Caveat:** the derivation assumes all selection acts through the modelled traits. Selection through unmodelled traits rescales the level, which is learned, and leaves the coefficient of 1 unchanged only if that other selection is proportional across units. The measurement below tests this.

**Beyond genes.** Enhancers, insulators and other elements are molecular units too. A deletion and a duplication of the same enhancer share its dosage effect with opposite sign. Deleting a CTCF site and a disruptive SNV in its core motif share the loss-of-function effect.

This is the SV-specific content: SNVs almost never change the copy number of a functional unit, and SVs do it wholesale. So the sharing structure is how SVs contribute information that no SNV model can reproduce.

## 7. The other classes

- **Inversions:** balanced, so c = 0 for genes inside them. For genes broken at a breakpoint, c = −1, sharing with the gene's other LoF events.
  - Recombination suppression makes the inversion allele a near-perfect tag of the haplotype inside it. The effects of the variants carried on that haplotype are then an LD question, which the likelihood handles.
  - Nothing beyond the additive column plus breakpoint sharing is implied.
- **Mobile-element insertions:** additive (cis). An exonic insertion is a disruption, c = −1. An intronic or regulatory insertion has an unknown sign, so it contributes only through the scale model (distance to exon or TSS, as smooths).
- **TAD-boundary disruption:** rewires regulation with no fixed sign, so it enters the scale model, not the sharing model.

## 7b. When the current columns are exactly optimal, and their exact loss otherwise

**Theorem 6.**
- **(a) Biallelic SV, additive column.** The per-locus Bayes predictor is a′ DS + d GP1 (Theorem 2). The best predictor that is linear in DS loses exactly
$$\mathcal L_{\text{add}} = d^2\,\mathrm{Var}\big(\mathrm{GP1} - \Pi_{\mathrm{DS}}\mathrm{GP1}\big),$$
  where Π_DS is the population linear projection on (1, DS).
  - With perfect genotypes this is d² (2pq)² = V_D (Proposition 4). With imputation it is smaller, and it vanishes as the posterior becomes uninformative.
  - The additive column is exactly optimal if and only if d = 0. By Corollary 1a that means H is affine on [E⁰, E⁰ + 2Δ].
  - Cross-locus epistasis adds the second-order link terms of Corollary 1c. Their loss is Var of the quadratic form ½ h''(A − Ā)² after projection onto the additive span.
- **(b) TR locus, signed-length column.** For true genotypes the loss is 2 Var_f(φ_⊥) (Theorem 3), so the column is exactly optimal if and only if φ is affine on the alleles with positive frequency.
  - Under imputation, with reported allele dosages D, the loss relative to the Bayes predictor E[Σ_h φ(q_h) | D] is Var(Σ_a φ_⊥(q_a) D_a − Π_Z(·)), the part of the per-allele φ_⊥-column not explained linearly by Z.
  - Misassignment to adjacent lengths averages φ_⊥ over neighbours, and this reduces the loss. A smooth φ_⊥ loses the least to misassignment, so the loss that survives imputation is concentrated in the smooth, large-scale shape of φ_⊥. That is exactly what the smooth basis captures.
- **(c) Gene dosage.** Independent per-variant priors with the correct marginal variances lose exactly the Bayes-risk gap of Theorem 5's closed form. That gap is zero if and only if each unit has at most one dosage-altering variant or τ_θ² = 0.

## 8. The optimal representation, class by class, inside the one model

| class | columns | prior |
|---|---|---|
| SNV / indel | additive D | the existing scale-mixture prior; LoF alleles also load −1 on their gene's U_m |
| biallelic DEL / DUP | additive D; also load ∓1 (DUP: +ρ) on each fully contained unit's U_m, and −1 on each broken unit | existing prior for ε_v; U_m gets θ_m |
| INS / MEI / INV | additive D; load −1 on the U_m of each unit they break | as above |
| TR locus | Z = Σ q_a D_a plus W_k = Σ (u_k(a)/√e_k) D_a | θ: existing prior; W: one learned class variance τ² (= 1/λ) |
| GATK-SV copy number | CN (already stored) plus W_k = b_k(CN_call) | as TR |
| per trait | a smooth monotone link h of the fitted additive index, identity null space | learned roughness; its variance correction needs GP |

**What stays the same:**
- no new inference machinery: every item is extra columns with Gaussian or scale-mixture priors, so it fits EP-EB unchanged;
- no hand-chosen constants: λ, τ_θ², τ_ε² and ρ are learned, and c_{vm} is derived from annotation.

**Costs:**
- **Columns:** Σ_j (A_j − 2) extra columns for TR loci, one per unit with two or more dosage-altering variants, and none for the link.
- **Store:** TR and mCNV columns come from the per-record dosages already in the store. U_m is a sparse sum of stored columns. GP1 is needed only for per-variant dominance, which the theory says to replace with the link.

## 9. What the theory predicts, to be measured

1. **Smooth TR function vs the current linear column:**
   - it matches the linear column when φ is linear (the evidence sends λ → ∞);
   - it gains when φ saturates, has a threshold or has an optimum;
   - it should never lose to per-allele columns, and should beat them most when misassignment is high.
2. **Gene-dosage sharing:**
   - it gains for rare, multi-variant units;
   - there is no gain and no loss with one variant per unit or no sharing, since τ_θ² → 0;
   - it stays robust when ρ ≠ 1, because ρ is learned.
3. **Per-variant dominance columns** should gain little or nothing. The single link should capture most of the nonadditive variance, and its second-order coefficient should come out at ½ h''.
4. **SV credit:** a misrepresented SV function leaks credit to tag SNVs, so the right representation should also raise the SV share of the prediction toward its true value.

## 10. Evidence status

- **The theory's predictions hold in this lane's own simulators, which are sim-only.** Those simulators encode the mechanisms assumed above, so under EVIDENCE_RULE.md they check the math and are not evidence of gain.
  - **TR loci:** the smooth φ matched the signed-length column exactly when φ was linear (λ → ∞, difference 0.0000 ± 0.0000). It beat per-allele columns in every scenario, and beat "linear + exchangeable deviations" wherever φ was non-linear. Its gains grew with 2 Var_f(φ_⊥): up to +0.062 R² for an optimum-shaped response at h²_TR = 0.2, and about 0 for near-linear saturating responses at low h².
  - **Gene-dosage sharing:** +0.007 to +0.008 R² for rare or common multi-variant genes, about 0 with one variant per gene, and −0.0002 ± 0.0001 under a null with no sharing. The learned ρ recovered 0.45 ± 0.19 against a true 0.5.
  - **Dominance:** per-variant dominance columns gained nothing (−0.0001) at any curvature. The single link gained +0.003 and +0.012 at nonadditive shares of 3% and 12%, and its quadratic coefficient had the predicted sign and scale, ½ E[h''] attenuated by the error in Â.
- **Public data:** in gnomAD-SV v2.1 (PASS sites), 305 genes carry two or more dosage-altering SVs at AF ≥ 0.001, and 102 carry both a loss and a gain. Those are the units where Theorem 5 applies; LoF SNVs add more.
- **Adoption waits for bench-real and bench-sim.** bench-real is 1kGP/MAGE cis-expression with real SVs and TRs, a direct real test of Theorems 3 and 5. The build is unchanged until those report paired held-out differences.
