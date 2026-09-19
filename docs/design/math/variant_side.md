# The variant side of the EP-EB fit: its floor, measured gap and removals

This note covers everything the fit does per variant besides store passes, for 21 traits × K folds × p ≈ 1.7e7 variants:
- the tilted moments of the continuous scale mixture and the EP site updates;
- the M-step at fixed cavities: the mixing density, the scale model, the D3 smoothing weights;
- the certificate quantities;
- the variance-refresh trigger.

B-products are in `b_products.md`. This note uses the same curvature convention, A = −∇² and B = −∇².

**Status tags:**
- **[proved]:** derived here.
- **[machinery]:** a measured computational quantity (time, passes, error against an exact reference, bound validity). None is an accuracy claim.
- **[est]:** an estimate from peak rates or extrapolation.
- Test problems are synthetic Stage-2-like cavities (n = 1e5, standardized columns, mostly-null z with a sparse tail) or dense EP problems with AR(1) LD, so every number is also sim-only as evidence. None bears on held-out accuracy.
- The scripts are on MSI in `svpgs-team/speed-ep/proto/`, and the kernels are prototypes, not production code. The engine (`sv_pgs/scale_mixture_ep.py`) owns integration.

## 1. The floor per pass [proved]
- **Inputs.** The cavity reduces each variant to three numbers: a_j = log(u_jP_j), b_j = h_j²/P_j and its class. Take q = e^{a_j + t_k} (one multiply with precomputed e^{t_k}) and σ = q/(1+q).
- **Kernel.** The kernel at node k is L_jk = ½(bσ − log1p q), and given the node the effect is Gaussian. So every tilted moment is closed form per node, and the node sum is the only irreducible work. It is the converged quadrature of a continuous density (math-density's certified spacing and range).
- **One pass** is Σ_j Q_j (variant, node) pairs, with Q_j the nodes where the kernel isn't flat. It costs about 12 flops and 2 transcendentals per pair.
- **Deviation form.** Sums run over expm1(L_jk), the change from the flat kernel. So flat nodes contribute exactly zero, and nothing cancels against the prior's own mass.
  - The M-step Hessian's η block is written as diag(ΣD) − DᵀD − (ΣD)πᵀ − π(ΣD)ᵀ with D = r − π, so the textbook n·π terms never appear.
  - The tilted variance, (E_r[σ] + b Var_r[σ])/P, is accumulated by a weighted Welford update.
- **Overflow guard.** L ≤ L_max = ½(b − 1 − ln b) for b > 1 (the maximum over all real node positions). When Σπe^L could exceed the fp32 range, a derived threshold, the kernel switches per variant to a max-shifted form.

**Passes per hyper step** at fixed cavities [proved]:
- The objective Σ_j log Σ_k π_ck e^{L_jk(a_j)} depends on z through every variant, so each Newton iterate needs a fresh pass; there is no pass-free exact update.
- **Floor:** 1 + n_N passes, where n_N is the Newton count from a warm start (quadratic, so a few), plus at most one corrector pass per λ trial with the implicit-function predictor x̂_{ρ+δ} ≈ x̂_ρ − B_pen⁻¹ λS x̂_ρ δ.
- The boundary test and the 1-D evidence maximization per smoothing weight work on D × D statistics and need no pass.

**Noise.** σ² = RSS/(n − k − γ) with γ = tr(QΣ) = p − Σ_j τ_j Σ_jj.
- RSS comes from the Stage 2 solver's own products; γ comes from the marginal variances the sites already need.
- So the noise update costs no extra pass.

## 2. Fused kernels against the engine's code [machinery; A40 46 GB, exclusive GPU via runq]
**Tilted moments** (`fast_tilted.py`): fp32 arithmetic, fp64 outputs, and a per-variant forward error bound on log Z_j.

| p | active nodes | fused kernel | naive cupy port of the engine's numpy (fp64) | engine numpy (fp64, 1 core) |
|---|---|---|---|---|
| 1e6 | 82 | 0.70–0.82 ms | 118 ms | 7.8 s |
| 1.7e7 | 105 | 12.7–13.2 ms, 1.4e11 pairs/s | out of memory (p × K fp64 temporaries) | ~132 s [est, linear from 2e5] |

- The fused kernel is ~170× faster than the naive GPU port, and ~10⁴× faster than one core.
- It runs about 5× over its fp32 compute floor [est ~2.6 ms for 1.8e9 pairs], because log1pf and expm1f are in software and the inputs are fp64.

**Accuracy** against the engine's fp64 code at p = 2e5 [machinery]:
- log Z_j: max abs error 4.6e-5; Σ_j|error| = 0.013 nats; Σ_j log Z_j differs by 1.3e-3 of 47,494.
- Tilted mean: relative error median 1.0e-7, max 8.2e-7. Tilted variance: median 9.2e-8, max 7.5e-7.
- **The certified bound** had 0 violations in 200,000 variants. Σ bound = 1.05 nats against the actual 0.013, so it is ~80× conservative.
- **Consequence for fp32:** at p = 1.7e7 that bound (~90 nats [est]) cannot certify the evidence value at 1/(2K) = 0.0078 nats.
  - The value must be evaluated in fp64, or accepted and rejected on value differences.
  - Gradients and Hessians don't have this problem (M-step accuracy, below).
- **A bug found and fixed:** the first run overflowed expm1f for strong variants (b up to ~1e4), and the overflow guard in §1 removed it.

**M-step statistics** (`fast_mstep.py`): value, gradient and Hessian in z; the same kernel structure plus fp64 GEMMs; chunk rows from the device's free memory.

| p | K | L | fused kernel + fp64 GEMMs | engine `_data_objective` (numpy, 1 core) |
|---|---|---|---|---|
| 6e4 | 74 | 20 | 16.0 ms | 429 ms |
| 1e6 | 93 | 20 | 198 ms | — |
| 1.7e7 | 113 | 20 | 4.29 s | ~122 s [est] |

- **Accuracy at p = 6e4 [machinery]:** gradient 4.4e-8 and Hessian 5.2e-8 relative; value 3.5e-4 absolute out of 3.3e4.
- **What dominates on the A40:** the fp64 GEMMs. DᵀD alone is 4.3e11 flop, ~0.75 s at 0.58 TF. D is fp32, so fp64 accumulation only avoids adding error to D's own ~1e-7.
- **Precision policy [proved as a policy; est for the savings]:**
  - Where fp64 runs at ≥ ½ the fp32 rate (V100, A100, H100), use fp64 throughout.
  - Where it's slow (A40, T4, L4), intermediate Newton iterations use an fp32 GEMM Hessian (inexact Newton converges the same), and the final certificate uses fp64. That's ~12 ms [est] instead of ~0.75 s per pass on the A40.

## 3. Passes one hyper step makes [machinery; engine ff714ff, p = 20k, 3 classes, a 6-column smooth, cold start]
**One `_maximize_coefficients` call** made 109 Newton iterations and 381 full passes (gradient + Hessian each), in 54.7 s on one core.
- 231 passes (61%) were rejected halvings that still computed a gradient and Hessian.
- 40 were a final halving loop that ran until the step equalled zero at rounding level.
- The predicted gain first fell below 1/(2K) at iteration 74; later iterations only moved toward eps·Σ|log Z_j|.

**A whole hyper step** made more than 846 full passes and was unfinished after 2 minutes. The earlier engine 20f81e7 made 1,365 in 3 minutes, also unfinished.

**Exact removals, on the engine's own iterates** (same path, same maximizer at the certificate tolerance):

| accounting | full passes | value-only passes |
|---|---|---|
| ff714ff as written | 381 | 0 |
| rejected trials value-only | 110 | 231 (+ 40 futile) |
| + stop at predicted gain ≤ 1/(2K) | 75 | 143 |

**Not an exact removal.** Cubic-regularized steps with a measured coefficient reached a certified local maximum (Newton decrement 5.7e-4) 0.85 nats below the engine's, 295 away in coefficient norm.
- The fixed-cavity objective is non-concave in log g, so the step rule changes which maximum is found.
- The lead's ruling since then: fit from two structural starts, keep the certified maximum with the higher corrected evidence, and refit from it. The engine records the local status.

**Adopted by the engine** (5f83226, 738a66a):
- value-only trial passes (`_data_value`);
- a stop at the caller's tolerance;
- the x predictor across λ trials;
- a trust region instead of halving.

## 4. The outer loop
**Dense EP [machinery; p = 30, K = 12]:**
- EP-EM (EP to convergence, then maximization at fixed cavities) reached δ_B ≤ 1e-10 in 3–4 outer iterations, and the certificate in 1–2.
- Unglobalized B-Newton took 9–13 iterations; at these sizes eig(A⁻¹B) ∈ [0.83, 1.7].

**At genome scale** [machinery, semi-real: speed-floor's measurement on real chr22 LD, extrapolated to genome scale by the chromosome-to-window LD-score ratio; `svpgs-team/speed-floor/rho_genome.json` on MSI]:
- At production signal, the outer pencil (B + S, A + S) has λ ∈ [0.89, 5.41], and mostly B ⪰ A.
- The fixed-cavity step's contraction factor, ρ = max|1 − λ|, reaches 4.41. It is ≥ 1 in 3 of the 10 production rows at the unit penalty weight.
- speed-floor reports that plain EP-EM diverges in 7 of its 16 configurations.
- **So the outer step must use B,** via b_products.md's closed form, globalized. The dense result above, where EP-EM converged, does not carry over to scale.

**Block-variance cavities** [machinery, semi-real, speed-floor, `cut_coupling3.json`]:
- Block-local marginal variances give cavity-precision errors with p99 of 28–58% at production block caps 1024–4096. The second-order cross-block term brings cap 4096 to p99 2%.
- B's variance channel therefore takes the cross-block marginal-variance map, `sv_pgs/marginal_variances.py` `variance_jvp` with its window and far-field terms, not the block-local (Σ_b∘Σ_b).

## 5. Variance-refresh trigger [proved; the demo is inconclusive]
Frozen marginals v_frz are refreshed only when one of these holds; no constant is chosen.
1. **Properness,** checked inside every EP sweep. The JVP predicts v from τ_frz to τ. The frozen cavity stays proper iff its relative error e_j < P_j/τ_j for τ_j > 0, or e_j > −P_j/|τ_j| for τ_j < 0.
2. **Bias vs progress.** Staleness shifts the hyper-gradient by δg = Φᵀ𝓛⁻¹[0; −δ], with δ = v_frz − Σ_jj: one linear-response solve (b_products.md). Refresh when ½δgᵀB⁻¹δg ≥ δ_B.
3. **Certification** requires δ_B + ½δgᵀB⁻¹δg ≤ 1/(2K).

**Dense demo:**
- At AR r = 0.6, refreshing every outer iteration certified at iteration 2, with 4 refreshes.
- The trigger re-fired at every sweep. Frozen errors reached 11% at p/n = 0.075, and a full site step moved τ far enough for the predicted cavity to go improper.
- At r = 0.95 exact EP itself failed.
- A conclusive test needs the production EP's derived damping, and the p/n ≫ 1 regime where per-sweep variance changes are small.

## 6. Totals [est]
- **Per model per outer iteration:** EP sweeps (1–3 near the fixed point) plus 1 + n_N + n_λ + 1 M-step passes, about 6–8 passes; ~30–40 passes per certified fit.
- **All 105 models:** ~35 passes × (~13 ms tilted + ~12 ms fp32 M-step) on one A40 ≈ 90 s. That is small next to the store side (compute_floor.md, dual_solve.md).
- **Not reducible below this:** the node window where the kernel is neither flat nor asymptotic, ~40–60 of the ~100 nodes. Below and above it, per-class prefix and suffix sums with a certified remainder could replace the node sum, a ~1.5× saving [est].
- **What models share:** they don't share kernel values, since each model's cavities differ. Batching models in one launch reads the design S and lattice constants once, and Sᵀdiag(curv)S becomes one batched GEMM.
