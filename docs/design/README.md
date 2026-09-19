# SV-PGS design documents

These documents describe the new SV-PGS path: the one model, its dosage store, its evaluation and its compute plan, plus the rulings behind each choice. The top-level README still describes the pre-cutover path; the new path replaces it (see HANDOFF.md, "Cutover").

| Document | What it holds |
|---|---|
| [HANDOFF.md](HANDOFF.md) | Where everything stands, and exactly what to do next when work resumes |
| [MODEL.md](MODEL.md) | The one generative model, its prior, its inference, and the Stage 0/1/2 pipeline |
| [STORE.md](STORE.md) | The 8-bit dosage store: arrays, halves, sidecar, loci, external annotations, gates |
| [EVALUATION.md](EVALUATION.md) | How an SV gain is claimed: arms, tests, the trait panel, simulation gates, red-team checks |
| [COMPUTE.md](COMPUTE.md) | Cost model, cloud launch path, MSI notes |
| [DECISIONS.md](DECISIONS.md) | Dated log of the design rulings and the measurements behind them |

**Standing rules** (also in SPEC.md):
- one model and one path;
- every component is a derived term of the model, backed by a measured paired held-out gain or a proven identity;
- no prior is chosen by hand;
- continuous quantities get continuous priors;
- approximate stages are warm starts, accepted only after exact full-data certification.
