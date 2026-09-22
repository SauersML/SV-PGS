"""bench-sim submission: SV-PGS's full-data fit with Stage 2's EP fixed point in place of mean field (``svpgs_full``
otherwise): EP's site approximation keeps the posterior's correlations, which mean field drops."""
from sv_pgs import stage2_wiring

stage2_wiring.INFERENCE = "ep"
from benchmarks.bench_sim.submissions.svpgs_full import Model, fit  # noqa: E402

__all__ = ["Model", "fit"]
