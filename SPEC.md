# SV-PGS

- Each type of variant will have a different prior on its effect size.
- Very rare SVs will be filtered.
- Single letter variable names are not allowed anywhere for any reason.
- Dead code is not allowed.
- Duplicated code is not allowed.
- Unnecessary conditionals should be avoided.
- Conditional imports are never allowed.
- Never silently fall back if a dependency is missing. Crash immediately.
- Never silently swallow errors with bare `except Exception: pass`. If something fails, let it fail loud.
- Only use UV, never pip.
- The same math runs on a large-RAM CPU node, a single GPU, or several GPUs. With several GPUs, all visible CUDA devices share the resident genotype work (sharded by LD block or column). The device decision is logged at fit start, and a GPU that is exposed but unusable is an error, not a silent fallback.
- Use JAX for iterative accelerator routes. Direct CPU posterior solves use
  NumPy/SciPy and must not initialize JAX dtypes or projectors.
- Do not restrict or cap the number of variants included arbitrarily.
- Do not restrict or cap the number of samples included.
- The variant-class-specific prior structure is the core differentiator of this tool. Every inference path must use metadata-driven prior variances (variant type, length, repeat status) and per-variant local shrinkage. The effect prior is a continuous Gaussian scale mixture whose mixing density is learned nonparametrically by empirical Bayes for each variant class, over a data-driven scale range with a learned smoothness penalty. It has no point mass at zero, and fixed-shape families such as TPB or a BayesR grid are special cases of it.
- No prior is ever chosen by hand. Every prior parameter is learned by empirical Bayes, pooled hierarchically across classes and traits, or else fixed by the model's own derivation. The prior matches the kind of quantity it covers: a continuous quantity gets a continuous prior, and a continuous annotation enters as a smooth function with learned smoothness, never as bins. A generic LASSO/elastic net that applies the same penalty to all variants is not acceptable as a primary inference backend.
- Every variant's prior variance, SNVs and indels included, depends on its structural-variant context: whether it lies in a tandem-repeat or SV locus, how strongly it tags a nearby SV, and the local SV landscape. The weights are learned by empirical Bayes. A SNV inside an SV locus never gets the same prior as a SNV far from any SV just because both are SNVs.
- We have a single, best path for users. Options and choices must be absent unless absolutely necessary.
- `# noqa` bypasses are never allowed. Fix the underlying issue or just ignore warning instead of silencing the linter.
- No holdout splits or cross-validation to do the fit itself (fine for evaluation or testing fit). The Bayesian prior is the regularizer — all samples train the model.
- No unnecessary environment variables.
- No hardcoded GPU sizes or device-specific constants. All GPU memory budgets, block sizes, and solver limits must be derived from the actual device memory at runtime. Code must work correctly on any NVIDIA GPU (T4, A100, H100, etc.).
- One model, one inference pass. Every variant goes through the same Bayesian model with the same prior structure. No two-stage pipelines, no "background" models for some variants and "exact" models for others, no treating variant subsets differently at the algorithmic level. Computational shortcuts (working sets, stochastic blocks) are optimizations that must produce the same result as the full joint model — they are not license to use a different model for different variants. Approximate stages (LD-space summaries, linearized likelihoods) are warm starts only: a fit is accepted only after it passes exact full-data gradient certification against the one model.
- The inference is empirical Bayes: hyperparameters are type-II maximum likelihood under an expectation-propagation approximation with exact one-dimensional tilted moments. A fit is accepted only with its convergence certificate (the Newton decrement of the hyperparameter objective, in nats), which is recorded in the artifact.
- Each variant's prior variance scales with its imputation accuracy r², which is fixed from long-read truth outside the fit.
- A tandem-repeat locus enters the model as one signed-length column, the sum of length change times dosage over every record in the locus. Equivalently, each allele's effect has a rank-one prior along its length change, so every allele dosage still enters the model.
