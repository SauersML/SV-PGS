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
- No multi-GPU support.
- Fully and unconditionally use JAX.
- Do not restrict or cap the number of variants included, for any reason.
- Do not restrict or cap the number of samples included, for any reason.
- The variant-class-specific prior structure is the core differentiator of this tool. Every inference path must use metadata-driven prior variances (variant type, length, repeat status) and per-variant local shrinkage (TPB). A generic LASSO/elastic net that applies the same penalty to all variants is not acceptable as a primary inference backend.
- Working-set screening and KKT certification are optimization techniques, not approximations. Any screening must be provably exact (the subset solution equals the full solution when KKT is satisfied). Never silently drop variants from the model.
- Never allocate a dense copy of a genotype matrix that can stay as a zero-copy mmap. Reorder the small side (sample table, covariates, targets) instead of the large side (genotype matrices).
