# The compute floor: minimal sufficient computation for the certified fit, and the gap to it

Lane speed-floor, 2026-09-19. It covers the whole pipeline for 21 traits × 5 folds (binary diseases included): stored codes to certified posteriors, draws and held-out scores.
- §1–§8 derive the floor; §9 measures the current and planned pipeline against it, stage by stage.
- Related: ep_eb.md (the certificate and its tolerances, which set the accuracy the floor must reach) and COMPUTE.md (hardware, launch paths, MSI).

**Ruling (lead, 2026-09-19):** build and optimize the Stage-2-only path first. Stage 1 is optional until the production outer contraction ρ and the real partition's cut coupling are measured (§9.4, items 1–2).


**Scope.** The whole pipeline, 21 traits × F = 5 folds (M = 105 models, binary diseases included), from the stored codes to certified posteriors, draws and held-out scores.

**Tags.**
- [proved]: an identity with a proof here.
- [checked]: verified numerically by `check_identities.py` or `bench_pass.py`, run on MSI through runq. The scripts and their JSON output are in `/projects/standard/hsiehph/sauer354/svpgs-team/speed-floor/`.
- [spec]: a vendor peak rate.
- [est]: an estimate with its basis stated.

**Workload W.** n = 10⁵ samples (the plan; today 50k + 12k), p = 1.7·10⁷ columns, M = 105, K_d posterior draws per model (64 today).
- The store holds b bits per code: 8-bit uint8 gives B = n·p = 1.70 TB, and the codec gives b = 1–2, i.e. B = 0.21–0.43 TB.
- The unit of work is 2·n·p = 3.4·10¹² multiply-adds per right-hand-side column per pass.

## 1. What the certified answer needs, statistic by statistic

| Needed | For | Data access it needs | Phenotype-independent? |
|---|---|---|---|
| Per-block integer Grams S_g, sums u_g, n_g per sample group g | block preconditioner and block variances; the Gram of every fold mask is an exact integer sum over groups | one pass, 2·n·p·p̄_b int8 ops | **yes: once for all traits and folds** |
| Column moments, LD partition, tie map, LD and SV-context features | standardization, blocks, prior design | the same pass | yes |
| Xᵀy for every trait × fold | Stage 2 right-hand side | the same pass, T·F extra columns | no, but it rides the same pass |
| Posterior mean μ_m at the EP fixed point | prediction | Krylov passes | no |
| Marginal variances Σ_jj per model (the site cavities) | EP site updates | block factors, no pass (§6) | no, but the fold Gram is shared |
| Hyperparameters x_m (G(x; c) and its gradient and fixed-cavity Hessian A) | EB | **none**: per-variant cavities only, O(p·Q) | — |
| Total-curvature products Bv (Newton outer steps, the δ_B certificate) | exact λ step and certificate (ep_eb.md §1.4, §3.1) | extra right-hand sides in the same passes (the mean channel), plus block-local variance work (the variance channel); speed-ep's closed form needs no separate EP re-solve | — |
| σ²_m = (‖y − Xμ‖² + tr(ΛΣ))/n | noise | ‖y − Xμ‖² falls out of the pass; **tr(ΛΣ) = p − Σ_j τ_jΣ_jj exactly** [proved, checked: gap 1.7e-13] | — |
| Certificate: δ_A/(1 − ρ̂), prediction bound, EP residuals, boundary score tests | acceptance | **none** beyond the fit's own passes (ep_eb.md §3; ρ̂ is the observed contraction) | — |
| K_d posterior draws | predictive variance | K_d extra columns in the final passes | no |
| Held-out scores X_ho μ_m and X_ho β^(k) | evaluation | **none**: the held-out rows are in the store, and the final pass's X·(·) over all rows produces them | — |

**Proof of the trace identity.** tr(ΛΣ) = tr((A − T)A⁻¹) = p − tr(TΣ) = p − Σ_j τ_jΣ_jj, with A = Λ + T and T diagonal. ∎
- So p_eff, which the σ² update and every K-draw tolerance need, comes from the variances the sites already require.
- **No Hutchinson probe is needed for σ² or for the tolerances.**

## 2. The pass: bytes and operations it cannot avoid
- **One read serves one Krylov iteration for every model.** This is exact_polish's linearity trick: with X·P known, a read forms Xᵀ(W∘(XP)) block by block and accumulates X·(M⁻¹AP).
- **Minimal bytes per pass.** The pass reads B, the store at the fastest tier holding it. The solver state must be sample-side (n × R), or resident on the device, so that it adds no traffic (§5).
- **Minimal operations per pass:** 2·n·p·R·L int8 multiply-adds.
  - R is the number of columns: the models' unconverged directions only.
  - L is the number of int8 digits the operand needs (§4).
- **The pass is compute-bound once R > R\*:**

  R\* = F_int8 / (2·L·codes_per_second).

| Device (per GPU) | int8 TOPS [spec] | host link | R\* at 8-bit, host-streamed | R\* at codec b = 1.5, host-streamed | R\* at codec, HBM-resident |
|---|---|---|---|---|---|
| A40 (MSI) | 299 | PCIe4, 24.4 GB/s [measured] | 6,100/L | 1,150/L | 40/L (multi-GPU only) |
| T4 | 130 | PCIe3, ~12 GB/s [est] | 5,400/L | 1,000/L | 38/L (does not fit) |
| V100-16 | no IMMA; DP4A 39 measured | PCIe3, ~12 GB/s | 1,600/L | 300/L | — |
| A100-40 | 624 | PCIe4, ~24 GB/s | 13,000/L | 2,400/L | 38/L |
| H100-80 | 1,979 | PCIe5, ~50 GB/s [est] | 19,800/L | 3,700/L | 55/L |

**The consequence for fitting many diseases.**
- A host-streamed pass is I/O-bound up to thousands of columns. So in that regime an extra disease (F·r more columns) costs **zero** pass time, and only adds state.
- With the codec, or resident codes, the pass turns compute-bound at R ≈ 10²–10³. From there each column costs 2·n·p·L operations, so **R and L are the levers.**

## 3. The floor on passes, stage by stage
**Solver.** For a Krylov method with preconditioned condition number κ, the passes per solve to relative error ε are at least ½√κ·ln(2/ε) (novel-inference THEORY §1a). The stages:

- **Stage 0 needs no pass of its own.**
  - Its statistics (§1, row 1) come from the staging read, the unavoidable cold copy from GCS or NFS to the fastest tier. That read is bandwidth-bound, and the int8 Gram (2·n·p·p̄_b = 1.4·10¹⁶ ops at p̄_b = 4,096) fits under it on any listed GPU: 2.8 s at peak on 8×A100, or 6.7 s at the measured 242–277 TOPS, against a staging read of minutes.
  - Floor: **0 extra passes**; the staging read is counted once.
- **Stage 1 is not necessary as a separate stage.**
  - Everything Stage 1 computes is a Stage 2 outer step with the operator truncated to its blocks. Its hyperparameter work, the M-step, is already pass-free at Stage 2 cavities (§1).
  - A Stage 1 sweep costs one variance refresh per model, (2/3)·p·p_b² fp operations (§6). On an A100 that equals about 600 Stage 2 passes in time: 1.9·10¹⁴ fp64-TC flops against a 1.07·10¹⁵ int8-op pass, at 19.5 against 624 T/s.
  - Even with TF32 factors (128 s per sweep on an A100, about 75 passes), 40 Stage 1 sweeps cost ~3,000 pass-equivalents, which is more than every Stage 2 pass of the floor (47) put together.
  - The warm start was measured to be in the right basin, but not faster (ep_eb.md §5.2: 7/15/7, 10/13/10, 10/8/17 outer steps).
  - Floor: **0**. Start Stage 2 from the prior, with loose forcing terms in the early outer steps (inexact Newton).
- **Stage 2 fit.** The first solve costs I₁ = ½√κ ln(‖μ‖/tol) passes.
  - Each later outer step warm-starts from the previous μ and solves only to a forcing tolerance η_o ~ ρ (Eisenstat–Walker; novel-inference §8c), in about ½√κ ln(1/ρ) passes.
  - With block-Jacobi κ (17–28 iterations measured for a full solve) and N_o = 5 outer steps at ρ ≈ 0.5, P_fit ≈ 26 + 4·4 = **42** [est; ρ at production scale is unmeasured, and this term scales as log(δ₀/δ\*)/log(1/ρ)].
- **Draws.** The draws' right-hand sides need the final sites, but they warm-start through the last outer steps like the mean does (Krylov recycling). So they add about one forcing-level solve, **~5 passes**, instead of a from-zero block-CG.
  - K_d itself should come from the needed accuracy of the predictive variance, and a control variate (below) cuts it.
- **Scoring.** In-cohort held-out scoring rides the final pass (§1): **0 passes**. Only a new target cohort needs one scoring pass.

**Floor: P ≈ 42 + 5 ≈ 47 data passes after the staging read.**

**The draw control variate [proposed].**
- The predictive variance is x_tᵀΣx_t = x_tᵀΣ_blk x_t + x_tᵀ(Σ − Σ_blk)x_t.
- The first term is exact from the block factors that §6 builds anyway. Only the second, off-block term needs Monte Carlo.
- The draw count K_d then scales with Var(correction)/Var(total) instead of with the total. Its size needs measuring (speed-krylov).

## 4. The precision floor: how many int8 digits L
- **The operand error.** The digit split's quantization error is ≤ 2^−(7L−3) of each column's maximum. Codes are exact integers.
- **The accuracy needed.** The fit must meet the certificate's mean tolerance ‖e‖²_{Λ+T} ≤ p_eff/K_d (ep_eb.md §3.3). As a relative error that is ε_rel = √(p_eff/K_d)/‖μ‖_{Λ+T} ≈ 10⁻¹–10⁻² [est: p_eff ~ 10³–10⁴, ‖μ‖²_{Λ+T} ≈ n·R²/σ² ~ 10⁴–10⁵].
- **What CG can reach.** Finite-precision CG attains a relative residual of about κ times the matvec error. So the matvec error must be ≲ ε_rel/κ ≈ 10⁻⁴ at κ ~ 10², i.e. 14 bits.
- **So L = ⌈(14 + 3)/7⌉ = 3.** The current OPERAND_DIGITS = 6 targets a 10⁻⁷ gradient tolerance that the certificate never asks for, which makes it **2× the floor** in the compute-bound regime.
- **The route depends on the device**, and should be chosen at runtime by querying it:
  - A40, A100, H100 and T4: int8 at L = 3.
  - V100 has no IMMA (DP4A measured at 39 TOPS). There a two-term fp16 split (22 bits, fp32 accumulate) at 125 TF is 6.4× the int8 route.
  - A certificate pass evaluates the final residual in fp64.

## 5. The memory floor: the solver state must be sample-side
- **Per state array:** p·R·8 B in variant-space, against n·R·8 B in sample-space. At R = 1,785 that is **243 GB against 1.4 GB**.
- **What the current design pays.** Block-CG keeps about 6 variant-side arrays (direction, operator·direction, preconditioned·operator, residual, preconditioned residual, solution) on the host in fp64. That is 1.46 TB at R = 1,785 and 5.5 TB at the draw width R = 6,720.
  - Each pass moves about 3 of them over PCIe (~0.7 TB) and touches all of them in host memory (~1.5–3 TB).
  - Against the codec store (0.3 TB), **the state, not the codes, sets the pass time**, and 1.46 TB does not fit the 680 GB host of an a2-highgpu-8g [spec].
- **The floor:**
  - Sample-space Krylov state, as in novel-inference's dual and Matheron draws, at n·R·4–8 B: 0.3 GB at R = 105 and 21 GB at R = 6,720. It fits beside a streamed int8 tile, even on a 16 GB GPU at fp32 with the draws chunked.
  - Or variant-space state at R = 105 (14 GB per array), resident in aggregate device memory.
- **Per-model variant vectors** (μ, τ, ν, D): 4·p·M·4 B = 29 GB in fp32. That's fine on the host, streamed per block with the codes.

## 6. Per-model work that isn't a pass: sites, M-step and variance factors
- **EP sites and tilted moments:** O(p·Q) per model per sweep, with Q quadrature nodes (~30–60, certified by math-density's rule).
  - That is 1.4·10¹² flops per sweep for all 105 models, about 0.1 s on one A100, if fused and streamed without materializing the p × Q × M = 290 GB tensor.
- **The M-step:** G, ∇G and A at fixed cavities cost O(p·(Q + d_θ)) per model. They need no pass.
- **Block variances, the only heavy non-pass work.**
  - Exact block factors cost (2/3)·p·p_b² per model per refresh: 1.9·10¹⁴ at p_b = 4,096, so 2.0·10¹⁶ for M = 105.
  - Per refresh that is 1,030 s on one A100 in fp64-TC, or 128 s with TF32 plus fp64 refinement (certified by the Jacobi-scaled condition number, as in e2854ca's policy), and 16 s on 8×A100. On an A40 it is 34,000 s in fp64 or 267 s in TF32.
  - **Its floor is well below dense-per-model:**
    - **(a) Refresh only when needed.** A refresh is due when the change in D could move a cavity by more than the certificate's EP budget (ep_eb.md §3.2, §5.3). So not every outer step: about N_o/2 [est].
    - **(b) Second order in the off-block part E** [proved; checked]. diag((A_blk + E)⁻¹) − diag(A_blk⁻¹) has **no first-order term**, because A_blk⁻¹ is block-diagonal and E has zero diagonal blocks. It equals diag(A_blk⁻¹EA_blk⁻¹EA_blk⁻¹) + O(E³).
      - Measured: the first-order term is exactly 0.
      - Halving E cuts the block-variance error by 4.0–4.7×, and after the second-order correction the residual falls by a further 8×.
      - **But on AR(1) blocks cut at arbitrary points, the block error at full coupling was 38–55%.** Block variances are accurate only when the partition cuts at low LD.
      - The online partitioner's cut weights bound ‖A_blk^{−1/2}EA_blk^{−1/2}‖, so this bound can be certified per block.
      - Where it fails, **overlapping blocks** (the marginals of an enlarged block, whose error decays geometrically with the overlap for LD that decays with distance) or the explicit second-order correction restore accuracy. Their cost is local, O(p_b²·overlap).
    - **(c) Sharing across diseases.** The Gram Λ_b of a fold is shared by every Gaussian trait on that fold, and by every binary trait through its mean curvature. Only the diagonal site part T_m is per model.
      - Where the site precisions of the unresolved bulk dominate (τ_j ≫ Λ_jj), Σ_jj ≈ 1/(τ_j + Λ_jj − s_j), where s_j is the conditional-precision correction. That correction is dominated by the few resolved (lightly shrunk) variants in the block.
      - So an exact factor is needed only on the resolved set L_b, costing p_b·|L_b|² per block, with the bulk diagonal. This is novel-inference's resolved/bulk split, applied to the variances.
      - Alternatively, a per-fold eigenbasis of Λ_b (computed once for all 21 traits) makes each model's diagonal a Woodbury update. That pays off only if the Gram's numerical rank r_b ≪ p_b; unmeasured.
- **Binary traits need no per-model weighted Gram.** The operator uses W_m only as an n-vector between X and Xᵀ, which is free in the pass.
  - The preconditioner and variances use the mask's mean curvature on the shared fold Gram.
  - The error is (W_m − w̄)'s correlation with x_j² plus O(CV(w)/√n) noise. It enters only the frozen variances, and the ε_j criterion bounds it.
  - A cavity computed directly as data precision minus the Schur correction, rather than as 1/Σ_jj − τ_j, never goes improper, which removes the cancellation behind math-epeb's ε_j < P_j/τ_j failure.

## 7. Many diseases and folds: what is shared and what grows
**Once for everything:** the staging read and its Stage 0 statistics (§1, rows 1–2), including every fold's Gram by exact integer downdate (train(f) = total − held-out group), and the per-fold block eigen or factor structure of §6(c).

**What each additional disease adds:**
- F more columns per pass. Zero time while the pass is I/O-bound (R < R\*), 2·n·p·L·F operations per pass beyond that.
- F·n·8 B of sample-side state (4 MB).
- F·O(p·Q) site work per sweep.
- Its share of the block-variance refreshes: the dominant marginal cost, ~2·10¹⁴ fp per fold model per refresh at dense p_b = 4,096, but only p_b·|L_b|² with the resolved-set rule.
- **It does not add passes** unless it becomes the slowest model to certify: converged models drop their columns, and the pass count is the maximum over models.

**Folds of one trait:**
- They share every byte, the Gram (by downdate) and, through their shared true prior, the hyperparameter basin.
- **Fit the full-data model for each trait first** (T = 21 columns, cheap). Then start its 5 fold models from its μ and x. Cross-fitting still holds, because each fold's fixed point is computed on its own training rows.
- That cuts the fold models' outer steps to about 1–2 [est], and their first-solve passes to the forcing level.

## 8. The floor, per hardware (all 105 models, codec b = 1.5, R_fit = 105, L = 3, P = 47, K_d = 8 with the control variate, variance refreshes N_ref = 3 dense in TF32 with refinement)

| Hardware | Staging read (cold, 0.32 TB) | 47 passes | Variance refreshes | Draw columns | Floor total |
|---|---|---|---|---|---|
| 1×A40 (MSI) | 10 min, /scratch.global cold at 0.53 GB/s [measured] | I/O-bound 13 s/pass (compute 5.0 s at the measured 213 TOPS) → 10 min | 3 × 267 s = 13 min | 5 compute-bound passes at 29 s → 2.4 min | **~36 min** |
| 1×T4 | 3–5 min, GCS 1–2 GB/s [est] | 27 s/pass → 21 min | fp32 3 × 2,470 s = 2 h | 5 × 66 s → 5.5 min | **~2.6 h** (variance-bound) |
| 1×V100-16 | 3–5 min | 27 s/pass → 21 min | fp32 3 × 1,270 s = 64 min | fp16 split, 5 × 46 s → 4 min | **~1.5 h** (variance-bound) |
| 8×A100-40 | 3–5 min | 1.7 s/pass (8 × 24 GB/s PCIe) → 80 s | 3 × 16 s = 48 s | +10 s | **~6 min** (staging-bound) |
| 8×H100-80 (codes resident in 640 GB) | 3–5 min | compute-bound 0.07 s/pass → 3 s | 3 × 5 s = 15 s | +2 s | **~4 min** (staging-bound) |

**The same accounting for the current plan** (COMPUTE.md, exact_polish, the Stage 1 head):
- **Stage 1:** 39–55 µs per variant·model·sweep × 40 sweeps, about **800 GPU-hours** (~100 h on 8 GPUs, measured implementation).
- **Stage 2:** R_fit = 1,785 at L = 6 over ~131 passes, plus draws at R = 6,720 over ~25 passes from zero. That's about 8·10¹⁸ int8 ops: **27 min at peak on 8×A100, ~4 h at the ~11% of peak the exact path achieved in the A40 bench** (~1–1.5 h after speed-io's kernel fix). Its variant-side host state (1.46–5.5 TB) doesn't fit the VM.
- **Plus** a separate Stage 0 pass and a separate scoring pass.
- **Against the floor:** about **1,000×** end to end today, dominated by Stage 1. Take Stage 1 out and the gap is **~10–40×** on 8×A100 (§9 itemizes it).

**Checked against a real pass [measured, A40].** The counting model predicts the raw int8 GEMM time exactly (2.42 ms predicted and measured at R = 105) once the achieved rate is used: 68–73% of spec for the pass GEMM shape, 41% for the square Gram. Use η ≈ 0.7 × spec for R\* in the pass shape. The measured table is in §9.3.

**What the floor rests on**, the assumptions a lane must verify before a gap is closed:
1. **The production outer contraction ρ.** P_fit scales as ½·log(δ₀/δ\*)/log(1/ρ). ρ = 0.99 would need ~160 outer steps, where 0.5 needs ~3.
2. **Block cut coupling.** It must be small enough for second-order block variances (§6b), or overlap is needed.
3. **The K_d control variate's variance reduction** (§3).
4. **The GCS staging bandwidth**, which is the floor's dominant term on 8-GPU VMs.

## 9. The gap: current and planned pipeline against the floor

**Sources.**
- "Current" means main at 378d0fd (exact_polish, code_products, genotype_statistics, fast_scoring), COMPUTE.md, and team/lane-speed/COST_MODEL.md.
- "Measured" rows are from `bench_pass.py` on the MSI A40 through runq (`__g1`, an exclusive GPU): see the measured section below (A40, n = 10⁵, p_b = 4,096; raw JSON `bench_pass.json` in the directory above).

**Workload:** n = 10⁵, p = 1.7·10⁷, 105 models (21 traits × 5 folds), store codec at b = 1.5 (0.32 TB) unless noted.

### 9.1 Stage by stage
Factors are in the dominant resource on 8×A100-40, the primary AoU configuration.

| # | Stage / item | Current or planned | Floor | Gap factor | Cause | Owner |
|---|---|---|---|---|---|---|
| 1 | **Stage 1** (LD-space EP-EB warm start) | 39–55 µs per variant·model·sweep measured (H100) × 40 sweeps: **~800 GPU-h** | not a separate stage. Its hyperparameter work is pass-free at Stage 2 cavities, and its variance work is the Stage 2 refresh (row 7) | **~10³–10⁴×** (implementation); the stage itself is removable | per-block Python and fp64 factors at every sweep; the whole stage duplicates Stage 2's work at a block-truncated operator | e2e, novel-inference (design); speed-ep |
| 2 | Stage 2 columns per fit pass | 105 × (1 + 16 probes) = 1,785 | 105 (+ ≤ 1 pooled certificate probe per model) | **17×** columns | Hutchinson diagonals are used in the site updates, which ep_eb.md §5.3 bans; p_eff instead comes from tr(ΛΣ) = p − Στ_jΣ_jj [checked] | speed-krylov |
| 3 | Stage 2 fit passes | ~131 = 5 outer × (1 + 17–28), with each outer step's correction solved from zero | ~42: first solve plus inexact warm-started outer steps (forcing η ~ ρ) | **~3×** passes | fixed tolerance, and cold restarts per outer step | speed-krylov |
| 4 | Operand precision | OPERAND_DIGITS = 6 (42 bits; targets 1e-7) | L = 3 (18 bits; the certificate needs ~14) | **2×** compute (×6.4 more on V100 via the fp16 split) | a hand-set constant, not derived from the certificate tolerance | speed-krylov, speed-io |
| 5 | Variant-side solver state | ~6 host fp64 p×R arrays: **1.46 TB** at R = 1,785 and 5.5 TB at R = 6,720; ~0.7 TB PCIe and ~1.5–3 TB host traffic per pass | sample-side state n×R (1.4 GB), or R = 105 resident (14 GB per array) | **infeasible → feasible**; 5–10× bytes per pass | the primal block-CG keeps its Krylov vectors in variant space on the host | speed-krylov, novel-inference |
| 6 | Posterior draws | 64 per model, a separate from-zero block-CG: R = 6,720 × ~25 passes (3.4e18 int8 ops at L = 6) | recycled through the last outer steps: ~5 passes, with K_d reduced by the x'Σ_blk x control variate | **5×** passes × (64/K_d′) columns; ~40× at K_d′ = 8 | a separate solve with no control variate | speed-krylov |
| 7 | Block-variance refreshes | a dense per-model block factor at each refresh: (2/3)·p·p_b² = 1.9e14 per model, 2.0e16 per refresh in fp64 (1,030 s/A100) | refresh only when needed (≈ N_o/2); TF32 + fp64 refinement (8×); resolved-set factors p_b·\|L_b\|²; shared per-fold structure | **8–50×** [est] | model-specific p_b³ at fp64, every outer step | speed-ep |
| 8 | Block-variance accuracy | block-Jacobi diagonal, with a Hutchinson correction | second order in the off-block E [checked: first-order term = 0]; certified per block from the cut coupling; overlap or the explicit 2nd-order term where it fails | correctness (38–55% error at full AR(1) coupling with arbitrary cuts) | cut LD isn't certified | speed-ep |
| 9 | Binary `exact_curvature` models | weighted_gram, 6 digits, per model per refresh: 8.4e16 int8 ops ≈ 134 s/A100/model/refresh | 0: the shared mask mean-curvature Gram; W_m is free in the pass | **∞** for flagged models | the exact per-model weighted Gram | speed-krylov |
| 10 | Stage 0 | a separate pass, plus fp32 width² Grams stored (~280 GB per mask) | fused into the staging read; int8 Grams recomputed in-pass (3–7 s on 8×A100) or int32 upper triangles | 1 pass, and ~280 GB per mask of storage | a separate stage; full fp32 squares | speed-io |
| 11 | Scoring (in-cohort evaluation) | a separate one-read pass | 0: the final pass's X·(·) over all rows gives X_ho μ and X_ho β^(k) | 1 pass | a separate stage | e2e |
| 12 | Code layout in matmat | transposes p_b × chunk of codes on every call (TN-only int8) | a variant-contiguous copy held once, or a layout-aware GEMM | ~34% of matmat (speed-io: 1.6 of 4.7 ms per p_b = 2,048 block at K = 105; 83 GB/s via cupy, 252 GB/s with a tiled kernel) | TN-only cuBLAS int8 | speed-io |
| 13 | Folds | 105 models started from the prior together | full-data model per trait first, then folds warm-started from its μ and x | outer steps per fold model ~5 → ~1–2 [est] | no cross-fold warm start | speed-krylov, e2e |
| 14 | Staging read (cold) | not in the plan's cost model | 0.32 TB at the GCS rate (3–5 min at 1–2 GB/s [est]); **the floor's dominant term on 8-GPU VMs** | — | — | speed-io |

### 9.2 End to end, 8×A100-40 (all 105 models)

| | Current or planned | Floor |
|---|---|---|
| Stage 1 | ~100 h (measured implementation) | 0 |
| Stage 2 fit | 4.8e18 int8 ops: 16 min at peak, **~2.4 h at the measured implementation efficiency** (11% of int8 spec, A40 bench; speed-io's fix lifts it ~3.8× at K = 105); host state 1.46 TB (does not fit) | 47 passes × 1.7 s (PCIe-bound) = 80 s |
| Draws | 3.4e18 int8 ops: 11 min at peak, ~1.7 h at the measured efficiency | ~10 s |
| Variances | 5 refreshes × 1,030 s / 8 = 11 min (fp64-TC) | 3 × 16 s = 48 s (TF32 + refinement) |
| Stage 0, scoring | 2 extra passes | 0 |
| Staging | — | 3–5 min |
| **Total** | **~100 h** (Stage 1) + **~4 h** Stage 2 at the measured efficiency (~1–1.5 h after speed-io's kernel fix) | **~6 min** |

**Total gap:** about **1,000×** today; about **10–40×** once Stage 1 is gone (~1–4 h against ~6 min). The largest remaining items are rows 2, 5, 6, 3 and 7, in that order by bytes and ops.

### 9.3 Measured on the MSI A40 (runq `__g1`, exclusive GPU; n = 10⁵, one LD block p_b = 4,096; median of 3)
The counting model predicts the raw kernels; the gap is in the path around them.

| Kernel | R | Time | Achieved | Counting model |
|---|---|---|---|---|
| H2D pinned, 410 MB block | — | 16.8 ms | 24.4 GB/s | = COST_MODEL's 24.4 GB/s |
| Exact int8 Gram (S·Sᵀ, int32) | — | 27.0 ms | 124 TOPS (41% of 299 spec) | Stage 0 Gram at p = 1.7·10⁷ on one A40: 112 s |
| Raw int8 GEMM, matmat shape, 6 digits | 105 / 420 / 1,785 | 2.42 / 9.47 / 43.1 ms | 213 / 218 / 204 TOPS (68–73%) | 2·n·p_b·R·6 at 213 TOPS predicts 2.42 ms at R = 105: **exact** |
| CodeBlockTile.matmat (the full exact path) | 105 / 1,785 | 19.9 / 173.5 ms | 26 / 51 TOPS effective | **8.2× / 4.0× over its own raw GEMM** |
| CodeBlockTile.rmatmat | 105 / 1,785 | 23.8 / 347 ms | 22 / 25 TOPS effective | **9.8× / 8.1× over the raw GEMM** (speed-io: the per-block digit re-split, now fixed) |
| Dense fp16 (tensor) | 1,785 | 20.7 ms | 71 TFLOPS | a two-term fp16 split is ~35 T effective, below int8 L = 3 (68 T): int8 wins on the A40 |
| Dense fp32 (cupy default) | 1,785 | 58.4 ms | 25 TFLOPS | the variance factors' TF32/fp32 rate (§6) |
| Dense fp64 | 1,785 | 2.93 s | 0.50 TFLOPS | fp64 block factors on the A40 are infeasible (34,000 s per refresh) |
| weighted_gram (binary exact curvature) | — | 0.73 s per block | 28 TOPS effective | 4,150 blocks → **~50 min per model per refresh** on the A40 (row 9) |

**One full pass at p = 1.7·10⁷ on one A40, from these numbers:**
- The current path at R = 1,785 and L = 6: (173.5 + 347) ms × 4,150 blocks = **2,160 s**.
- The current path at R = 105: 181 s. speed-io's bit-identical fix measured 287 → 52 s at p_b = 2,048.
- The floor at R = 105 and L = 3: max(5.0 s compute at the measured 213 TOPS, 13 s streaming the codec over PCIe) = **13 s**.
- So the per-pass gap on the A40 is **~165×** = 17× (R) × 2× (L) × ~5× (the exact path's overhead over raw GEMM), capped by the codec I/O at the floor.

### 9.4 What the gap depends on (verify before closing)
1. **The production outer contraction ρ,** which sets rows 1 and 3. At ρ = 0.99 the floor needs ~160 outer steps. Then Stage 1's pass-free steps could be worth keeping, but reimplemented at the row-7 floor.
2. **Cut coupling of the real partition,** row 8. Measure ‖A_blk^{-1/2}EA_blk^{-1/2}‖ on 1kGP chr22 with the online partitioner's cuts.
3. **The variance reduction of the draw control variate,** row 6.
4. **GCS staging throughput** on a2-highgpu-8g and a3, row 14.
