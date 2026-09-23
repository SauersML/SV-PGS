"""bench-sim submission: SBayesRC without annotations (every variant shares one set of mixture probabilities: SBayesR on
the low-rank eigen LD). Everything else is sbayesrc.py's: the same variants, blocks, statistics, LD and package defaults."""
from __future__ import annotations

from benchmarks.bench_sim.submissions.sbayesrc import Model, fit_sbayesrc

__all__ = ["Model", "fit"]


def fit(train) -> Model:
    return fit_sbayesrc(train, annotated=False)
