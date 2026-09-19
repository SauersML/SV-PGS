# The Stage 2 solves in dual form: every model, fold, probe and draw in one pass

This note derives `sv_pgs/dual_solve.py`: the Gaussian E-step of Stage 2 for all 21 traits × (K folds + the full-data fit), their probes and their posterior draws. It covers:
- the certificate;
- the treatment of spikes and non-positive sites;
- the precision policy;
- the measured pass counts.

**Status tags:**
- **[proved]:** an identity proved here.
- **[checked]:** verified numerically by a test in `tests/test_dual_solve.py` (CPU) or `tests/test_dual_solve_cuda.py` (CUDA).
- **[measured]:** numbers from the lane's runs.

All measured numbers are machinery only: passes, iterations, digits and time, never accuracy (EVIDENCE_RULE). They come from the synthetic s1M_100k store, whose genotypes are real 1kGP haplotype mosaics, so they are [semi-real] genotypes with simulated sites.

## 1. One operator per model, and one pass for all of them [proved; checked]

A model is a trait on a training set. It has:
- likelihood weights w (n,) that are zero on held-out rows: 1/σ² for a quantitative trait, and the logistic curvature or EP likelihood-site precision for a binary one;
- Gaussian site variances D = 1/Π ≥ 0 on the bulk;
- the covariates C profiled out in its own weighted metric, H = W^½C(C'WC)⁻¹C'W^½.

With Xt = (I − H)W^½X,

    A = Xt'Xt + diag(Π),    S = I + Xt D Xt'  (every eigenvalue ≥ 1),
    μ = m + D Xt' z,  S z = b,  b = (I − H)W^½(ỹ − X m),

by the push-through identity A⁻¹Xt' = D Xt' S⁻¹.

**One read serves every column of every model:** T = W^½(I − H)V; per tile, U = X_b'T, U ∘= D_b[:, model], Y += X_b U; then S V = V + (I − H)W^½Y.
- A binary model is a per-column weight, and a fold is a mask. Neither needs a Gram or a factor, so the primal plan's block-Jacobi inverses (M·p·p_b·8 B, 58 TB for 105 models at p = 17M, p_b = 4,096) are not needed.
- The Krylov state is n × columns, against the primal's p × columns on the host (1.46 TB at 1,785 columns).

## 2. The certificate is the dual residual [proved; checked]

For any iterate with r = b − S ẑ and μ̂ = m + D Xt'ẑ:

    ‖μ̂ − μ‖²_A = r'(I − S⁻¹) r ≤ ‖r‖².

*Proof.* e = −D Xt'S⁻¹r and Xt (D A D) Xt' = G S with G = S − I, so e'Ae = r'S⁻¹G r = r'(I − S⁻¹)r. ∎

So ep_eb.md §3.3's target ‖e‖²_A ≤ p_eff/K holds exactly when ‖r‖² ≤ p_eff/K. The residual the solve already has is the certificate, with no energy-norm estimate.

**Rows outside the training metric** (held-out folds, target people). With P = X_out D X_out':

    ‖X_out(μ̂ − μ)‖² ≤ ¼ λ_max(P) ‖r‖²,

since ‖G^½S⁻¹r‖² = r'(S⁻¹ − S⁻²)r ≤ max_{λ≥1}(λ − 1)/λ² ‖r‖² = ‖r‖²/4. Once the spikes are eliminated, P is the bulk's.

## 3. Spikes, non-positive sites and the exact split

**Spikes.**
- Variant k adds the spike D_k‖xt_k‖² to G. Its bulk level is tr(G_bulk)/n_train.
- The resolved spikes are the fixed point of L = {k : D_k‖xt_k‖² > 1 + tr(G_{not L})/n_train}.
- ‖xt_k‖² = x_k'Wx_k − (x_k'WC)(C'WC)⁻¹(C'Wx_k) for every model comes from one read.
- Without elimination, spikes set the iteration count. [measured] At n = 30,000 an undeflated solve needed 68 CG iterations even at a loose bound (§6).

**Non-positive site precisions** (the no-clipping ruling) cannot enter S.
- With q of them and A ≻ 0, Sylvester's law of inertia on [[Π, Xt'],[Xt, −I]] gives S exactly q negative eigenvalues, so CG on S is invalid [proved; checked].

**The split [proved; checked].** With the bulk operator S_S (D = 0 on L) and Z = S_S⁻¹[b, Xt_L]:

    core = Π_L + Xt_L'Z_L      (the Schur complement of A's bulk block: positive definite exactly when A is)
    μ_L = core⁻¹(h_L + Xt_L'z_b),   z = z_b − Z_L μ_L,   μ_S = m_S + D_S Xt_S'z.

- The core's Cholesky is therefore the global positive-definite check.
- The columns of Xt_L are kept in the order of the sites they pair with.
  - A mismatch between block order and the given order was a real bug, exposed on AVX-512 runners by argsort's tie order; fixed in 26edb38.

**The certificate with the split [proved; checked].**
- Let r_S = b − Xt_Lμ̂_L − S_Sẑ and r_L = Xt_L'ẑ + h_L − Π_Lμ̂_L. Schur's complement of A's bulk block gives, exactly,

      ‖μ̂ − μ‖²_A = r_S'(I − S_S⁻¹) r_S + (r_L + Z_L'r_S)' core⁻¹ (r_L + Z_L'r_S),

  with the exact Z_L and core.
- **r_S costs no pass:** r_S = r_b − R_L μ̂_L, from the exact residuals of the solve's own columns.
- **The inexact terms are bounded:**
  - the computed Ẑ_L differs from Z_L by S_S⁻¹R_L, so Z_L'r_S moves by at most ‖R_L‖_F‖r_S‖;
  - core differs from the computed one by Z_L'R_L, whose relative size δ = ‖ĉore⁻¹‖₂‖Xt_L‖_F‖R_L‖_F must be below 1, and then core⁻¹ ⪯ ĉore⁻¹/(1 − δ).
- `DualGaussian.iterate` tightens an open model's columns by its measured shortfall until every bound holds.

**Draws exact for a positive-definite A [proved; checked].**
- β_L* = μ_L + core^{−½}ε_L is taken from β_L's marginal.
- The bulk is then drawn from its conditional by Matheron on S_S, with dual z_e − Z_L(β_L* − μ_L), where z_e = S_S⁻¹(e₂ − Xt_S D_S^½ e₁).
- The test extracts the draw map from unit noise and checks MM' = A⁻¹ against the dense posterior with negative sites.
- D^½ is taken only on the bulk.

## 4. Precision: the fewest exact int8 digits the residual allows [proved bound; measured]

**Exactness.**
- The codes are int8, and each read's float operand is written as m balanced base-128 digits (`code_products`).
- Every digit GEMM is exact in int32, so the only error is the operand's rounding. The tile takes the least m meeting a normwise bound per column, ‖δ_k‖₂ ≤ ε‖L_k‖₂, with exact zeros kept.

**The bound.**
- Iteration k's product error F_k moves the true residual by at most ‖F_k‖‖r_k‖, since S ⪰ I and the directions are orthonormal.
- ‖F_k‖ ≤ λ_max(S)(ε_left + ε_right).
- Keeping the remaining drift below half the bound, over a residual contracting by ρ, gives each of the two operands

      ε = (bound/2)(1 − ρ) / (2 λ_max ‖r_k‖).

- λ_max is the largest Ritz value seen. With none known, the first iteration runs exact. The returned residual is always exact.

**Only on spike-free operators [measured].**
- On an operator that still has its spikes, finite-precision CG re-converges its outlying Ritz values, and relaxed products lengthened that by 3 iterations at n = 3,500 and by 22 at n = 30,000.
- On a deflated or split operator they cost none. So the relaxation applies only there.

**Digits used [measured]:**
- 2 per CG pass at n = 3,500;
- at n = 30,000: 3–4 at a relative bound of 0.1 and 0.01, and 3–5 at 0.001, out of 8 (fp64-equivalent).
- **A40 pass time** (n = 62k, 4,096-variant tiles, proto tiles): 2 digits instead of 6 was 1.9× faster at 126 columns and 2.4× at 6,720.

## 5. Refresh with no pass of its own [proved; checked]

`refresh_pass` makes one read, and for every tile:
1. U_b = Xt_b'z for every column: the previous solve's products, which EDB-EP's block-local step consumes;
2. (D_b, m_b) = block_update(b, U_b) writes block b's new sites;
3. Y += X_b(D_b U_b) and X m += X_b m_b are accumulated under the new sites.

After the read, S_new z and X m_new are exact. So a refresh costs its warm CG passes plus that one read.

## 6. Pass counts [measured]

Models: 3 quantitative traits and 1 binary trait (logistic curvature weights at 10% prevalence), 24 models in all, each trait with its full-data fit and 5 folds. Sites: a heavy-tailed bulk plus 20 strong variants per trait.
- The target is ep_eb §3.3 at K = 64, with p_eff exact from the dense dual at n = 3,500.
- At n = 30,000 the target is a relative bound on ‖r‖/‖b‖.

**n = 3,500 × 56,294 variants (chr20–22), CPU:**

| scheme | CG iterations | passes |
|---|---|---|
| dual, exact, cold, no deflation | 23 | 24 |
| dual, relaxed + deflated, cold | **5** | 8, including 2 setup reads |
| refresh (sites moved 10%), warm, deflated | **1** | 5 |
| folds warm-started from the masked full-data dual | **2** (cold: 5) | 6 |
| 1,536 draws (64 per model), relaxed + deflated | 4 | 7 |
| primal block-Jacobi PCG (exact_polish) at the target | 8 | 9 |

**n = 30,000 × 483,944 variants (chr1–7), A40.**
- The deflation used a column budget from free device memory (~5,800 of the largest spikes), beside the resident codes.

| scheme | bound 0.1 | 0.01 | 0.001 |
|---|---|---|---|
| cold, exact, not deflated | 68 | | |
| cold, relaxed + deflated | **7** | 11 | 15 |
| warm refresh, deflated | **1** | 4 | 9 |
| folds warm from the full-data dual | 3 | 7 | 11 |

## 7. What it replaces
- **exact_polish's block-Jacobi factors and `exact_curvature` Grams:** not needed.
- **The p × C Krylov state:** n × C instead.
- **Hutchinson variances in site updates:** none. Probes feed only the leave-block-out marginals' scalars and certificates.
- **The 400-iteration cap:** replaced by the certificate.
- **The Stage 2 tail:** means, draws, certificate and scores in 3–4 reads before; one fused read now (`fused_final_pass`).
