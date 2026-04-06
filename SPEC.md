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
