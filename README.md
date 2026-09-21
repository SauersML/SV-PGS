# SV-PGS

Bayesian polygenic scores in which structural variants, tandem repeats and SNVs enter one generative model. The target data are the All of Us imputed genomes with a long-read-called reference panel.

## The model
The full model is in [docs/design/MODEL.md](docs/design/MODEL.md); the rulings behind it are in [DECISIONS.md](docs/design/DECISIONS.md). This is the model the code is built to; [docs/design/CAPABILITIES.md](docs/design/CAPABILITIES.md) lists the parts the public routes do not reach yet.
- **Likelihood.**
  - A Gaussian likelihood for quantitative traits.
  - A logistic likelihood for diseases, or a Gaussian working likelihood on a liability target where one exists.
  - Covariates are projected out exactly before anything else is computed.
- **Measurement.** A stored dosage is a measurement of the true genotype, with a truth-calibrated reliability r². GATK-SV short-read calls are a second measurement of the same genotype, and the two are fused.
- **Prior.** Each effect is a continuous Gaussian scale mixture.
  - Its mixing density is learned nonparametrically by empirical Bayes for each variant class, with no point mass.
  - Each variant's scale depends on r² and on its annotations, including its structural-variant context, through smooth functions with learned smoothness.
  - No prior is chosen by hand.
- **Inference.** Empirical Bayes: type-II maximum likelihood of the prior's hyperparameters at a fixed point of the approximate posterior. Two fixed points are implemented, expectation propagation and mean-field coordinate ascent; which one the project keeps is still being measured. Every fit carries its convergence certificate, whether or not it certified.
- **Pipeline:**
  1. an 8-bit dosage store;
  2. Stage 0, one phenotype-independent genotype pass;
  3. Stage 2, EP-EB on the full data from the prior, with certified marginals and a Newton-B outer loop;
  4. one-pass scoring with posterior draws.

## Status
[docs/design/HANDOFF.md](docs/design/HANDOFF.md) has the current state and the ordered next steps; [docs/design/CAPABILITIES.md](docs/design/CAPABILITIES.md) has the requirements the public routes do not meet yet, each with the test that pins today's behaviour.
- **On main:** the dosage store and its streamed reader, Stage 0, the EP-EB engine (`scale_mixture_ep.py`, with its certified outer loop), the Stage 2 solves (`dual_solve.py`) and certified marginal variances (`marginal_variances.py`), the Stage 2 driver (`full_data_fit.py`), the small-n dense route (`small_n.py`) and the pooled fit (`pooled_fit.py`), the fit/score API and its model artifact (`fit_model.py`, `artifact.py`, `cli.py`), scoring, the reliability and fusion models, the prior design, the cohort builder with typed sample IDs, the phenotypes and their measurement model, the evaluation tests, and two neutral benchmarks (`benchmarks/bench_real`, `benchmarks/bench_sim`) with SV-PGS's own method entry (`benchmarks/svpgs_method.py`).
- **Deleted:** the old fitting path, in cutover steps C2–C8 ([CUTOVER.md](docs/design/CUTOVER.md)); it is recoverable from tag `archive/2026-09-19/old-path-final`.

## Install
```bash
uv sync                 # CPU
uv sync --extra gpu     # plus the CUDA 12 GPU libraries
```

## Supported platform
Linux on CPython 3.12: the MSI suite runs 3.12.13 and the CI workflow runs 3.12, and nothing else has ever been run. `sv_pgs` reads `/proc` and `os.sched_getaffinity` for its memory and thread budgets (`compute_budget`, `dosage_store`, `progress`), which no other operating system provides, so an older Python or another platform is not a claim this project can make. `pyproject.toml` says the same.

## All of Us phenotypes
The 21-trait panel, 11 quantitative traits and 10 diseases, is built from the workspace's OMOP CDR by BigQuery. It runs inside the workspace, and only counts of at least 21 participants may leave it. [docs/design/PHENOTYPES.md](docs/design/PHENOTYPES.md) has the definitions.

```bash
uv run sv-pgs census-all-of-us-traits --output trait_census.tsv
uv run sv-pgs list-all-of-us-traits
uv run sv-pgs prepare-all-of-us-trait --trait ldl_cholesterol --output ldl.samples.tsv
uv run sv-pgs list-all-of-us-diseases
uv run sv-pgs prepare-all-of-us-disease --disease type2_diabetes --output t2d.samples.tsv
```

Each prepared table has two sidecars:
- the exact SQL it ran;
- a metadata file with the query parameters, exclusion counts, variance components, and a fingerprint of the phenotype definition.

## Synthetic data
`python -m sv_pgs.synthetic_store` writes a synthetic dosage store from public 1000 Genomes haplotype mosaics, for tests and simulation studies.

## Rules
[SPEC.md](SPEC.md) holds the binding rules:
- one model and one path;
- CPU and GPU both first-class;
- every component a derived term of the model, backed by a measured gain or a proven identity;
- no hand-chosen priors or constants.

## License
AGPL-3.0-or-later.
