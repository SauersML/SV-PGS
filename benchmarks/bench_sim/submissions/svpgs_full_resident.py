"""bench-sim submission: ``svpgs_full`` itself, under its own name, so a rerun on a scenario it already scored is kept
beside the first (the regression check of Stage 2's device-resident codes and cached panel couplings)."""
from benchmarks.bench_sim.submissions.svpgs_full import Model, fit  # noqa: F401
