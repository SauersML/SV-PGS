# SV-PGS design documents

These documents describe SV-PGS: the one model, its dosage store, its phenotypes, its evaluation and its compute plan, plus the rulings behind each choice. The old fitting path was deleted in cutover steps C2–C8 (CUTOVER.md).

| Document | What it holds |
|---|---|
| [HANDOFF.md](HANDOFF.md) | The current-state entry point: what's on main, what's in flight, what's next, and the decisions waiting on the user |
| [CAPABILITIES.md](CAPABILITIES.md) | The requirements of SPEC.md and MODEL.md the public routes do not meet yet, what they do instead, and the test that pins each behaviour |
| [MODEL.md](MODEL.md) | The one generative model, its prior, its inference and certificate, the Stage 0 → Stage 2 pipeline, and the evidence tags |
| [STORE.md](STORE.md) | The 8-bit dosage store: arrays, halves, sidecar, loci, external annotations, gates |
| [EVALUATION.md](EVALUATION.md) | How an SV gain is claimed: arms, tests, the trait panel, simulation gates, red-team checks |
| [PHENOTYPES.md](PHENOTYPES.md) | The 21-trait All of Us panel as built: disease and trait rules, targets, which values are standards, and the rules still to be replaced |
| [COMPUTE.md](COMPUTE.md) | Cost model, cloud launch path, MSI task runners, the landing gate |
| [CUTOVER.md](CUTOVER.md) | The ordered deletion of the old path (C2–C8, done): files, tests, blockers, verification; a historical record |
| [CDR_LAYOUT.md](CDR_LAYOUT.md) | Where the All of Us CDR inputs live |
| [DECISIONS.md](DECISIONS.md) | Dated log of the design rulings and the measurements behind them |
| [math/scale_model.md](math/scale_model.md) | Derivation of the prior scale: the r² offset, joint leakage and the A-map, stacking, the frequency term from stabilizing selection, shape vs scale, the TR column, the SV-context kernel, the pooling pin |
| [math/novel-evoprior.md](math/novel-evoprior.md) | The effect prior derived from mutation–selection–drift: exact folded SFS, the frequency-conditioned scale mixture and its ceiling, the S exponent in closed form, SV length as a subordinator, the TR diversity-deficit tilt, trait pooling, the SFS channel, identifiability (sim-only checks; adoption pending bench) |
| [math/prior_sweep_a.md](math/prior_sweep_a.md) | Exact normal-means checks of the learned mixing density: the order-m null space is improper under a flat prior (collapse to V ≈ 0), penalty values as exact squares, grid and range invariance of the chosen λ, and a level-only pool losing a heavier tail (sim-only; adoption on bench) |
| [math/compute_floor.md](math/compute_floor.md) | The minimal sufficient compute for all traits × folds (passes, bytes, FLOPs by precision, memory, per-hardware floors) and the measured stage-by-stage gap of the current pipeline to it |
| [math/dual_solve.md](math/dual_solve.md) | The Stage 2 solves in dual form (sv_pgs/dual_solve.py): one pass for every model, fold, probe and draw; the dual-residual certificate and its held-out bound; spikes, the exact split of non-positive sites and its Schur certificate; relaxed int8 digits; the refresh read; measured pass counts (machinery only) |
| [math/ep_eb.md](math/ep_eb.md) | The EP-EB objective, its joint fixed point, convergence of the inner and outer loops, and the certificate |
| [math/mixing_density.md](math/mixing_density.md) | The continuous mixing density: data-derived range, certified quadrature, penalty order D3, the λ = ∞ limit, pins, the M-step |
| [math/b_products.md](math/b_products.md) | Closed-form products with the total curvature B, without EP re-solves, and the certified direction count for log det terms |
| [math/variant_side.md](math/variant_side.md) | The variant-side floor and gap: fused tilted-moment and M-step kernels, pass counts, the refresh trigger (machinery) |
| [math/novel-measure.md](math/novel-measure.md) | Measurement theory for imputed SVs with an internal truth panel: calibrated means, credit leakage, stacking, the Rao–Blackwellised column (sim-only; adoption on bench-sim) |
| [math/novel-svfunction.md](math/novel-svfunction.md) | What the model should regress on: cis-additivity, the smooth TR length function, gene-dosage sharing (sim-only; adoption on bench-real) |
| [math/novel-portable.md](math/novel-portable.md) | The ancestry-optimal score from the causal model and why SVs port (real-haplotype mechanism measurement; adoption on the benchmarks) |
| [math/novel-pheno.md](math/novel-pheno.md) | Phenotype measurement theory: the per-occasion model, treatment, disease evidence (NHANES check labelled [real, NHANES replicates]) |
| [math/novel-inference.md](math/novel-inference.md) | EDB-EP: the exact block-marginal identity and the environment likelihood for Stage 1; Schur decoupling, resolved/bulk elimination and the leave-block-out data precision for Stage 2; certified marginal variances and their derivative (sv_pgs/marginal_variances.py); machinery measurements against exact EP |

**Standing rules** (also in SPEC.md):
- one model and one path;
- every component is a derived term of the model, backed by a measured paired held-out gain or a proven identity;
- no prior is chosen by hand;
- continuous quantities get continuous priors;
- approximate stages are warm starts, accepted only after exact full-data certification;
- no arbitrary constants (guarded by `tests/test_no_arbitrary_constants.py`);
- no AoU-related data outside the AoU workspace;
- accuracy claims only from the neutral benchmarks or in-workspace held-out data.
