# SV-PGS design documents

These documents describe SV-PGS: the one model, its dosage store, its phenotypes, its evaluation and its compute plan, plus the rulings behind each choice. The old fitting path is being deleted step by step (CUTOVER.md).

| Document | What it holds |
|---|---|
| [HANDOFF.md](HANDOFF.md) | Where everything stands, and exactly what to do next when work resumes |
| [MODEL.md](MODEL.md) | The one generative model, its prior, its inference, and the Stage 0/1/2 pipeline |
| [STORE.md](STORE.md) | The 8-bit dosage store: arrays, halves, sidecar, loci, external annotations, gates |
| [EVALUATION.md](EVALUATION.md) | How an SV gain is claimed: arms, tests, the trait panel, simulation gates, red-team checks |
| [PHENOTYPES.md](PHENOTYPES.md) | The 21-trait All of Us panel as built: disease and trait rules, targets, which values are standards, and the rules still to be replaced |
| [COMPUTE.md](COMPUTE.md) | Cost model, cloud launch path, MSI notes |
| [CUTOVER.md](CUTOVER.md) | The ordered deletion of the old path (C2–C8): files, tests, blockers, verification |
| [DECISIONS.md](DECISIONS.md) | Dated log of the design rulings and the measurements behind them |
| [math/scale_model.md](math/scale_model.md) | Derivation of the prior scale: the r² offset, joint leakage and the A-map, stacking, the frequency term from stabilizing selection, shape vs scale, the TR column, the SV-context kernel, the pooling pin |
| [math/novel-evoprior.md](math/novel-evoprior.md) | The effect prior derived from mutation–selection–drift: exact folded SFS, the frequency-conditioned scale mixture and its ceiling, the S exponent in closed form, SV length as a subordinator, the TR diversity-deficit tilt, trait pooling, the SFS channel, identifiability (sim-only checks; adoption pending bench) |
| [math/prior_sweep_a.md](math/prior_sweep_a.md) | Exact normal-means checks of the learned mixing density: the order-m null space is improper under a flat prior (collapse to V ≈ 0), penalty values as exact squares, grid and range invariance of the chosen λ, and a level-only pool losing a heavier tail (sim-only; adoption on bench) |
| [math/compute_floor.md](math/compute_floor.md) | The minimal sufficient compute for all traits × folds (passes, bytes, FLOPs by precision, memory, per-hardware floors) and the measured stage-by-stage gap of the current pipeline to it |
| [math/novel-inference.md](math/novel-inference.md) | EDB-EP: the exact block-marginal identity and the environment likelihood for Stage 1; Schur decoupling, resolved/bulk elimination and the leave-block-out data precision for Stage 2; certified marginal variances and their derivative (sv_pgs/marginal_variances.py); machinery measurements against exact EP |

**Standing rules** (also in SPEC.md):
- one model and one path;
- every component is a derived term of the model, backed by a measured paired held-out gain or a proven identity;
- no prior is chosen by hand;
- continuous quantities get continuous priors;
- approximate stages are warm starts, accepted only after exact full-data certification.
