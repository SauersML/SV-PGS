"""Regression cases on real bench-real genes (MAGE / 1kGP; public data): pooled start fixed points that refused.

On svpgs-pooled-run's 20 chr22 genes (18cc5df, before the per-gene double-loop fallback) four of the five loso snv
start fixed points refused, "no damped EP pass keeps the precision positive definite": AMR at gene 13, EUR at gene 6,
EAS at gene 7 and SAS at gene 15 (AFR ran past an hour). Each group's start fixed
point must now be reached. Each case builds the group's 20 genes as the batch arm does and asks the oracle for the
start's fixed point: about 20 minutes on 2 cores, so the cases run only where the data are and when asked
(SVPGS_REAL_REGRESSION=1, BENCH_REAL_DATASET, SVPGS_POOLED_GENES: the gene list, one gene_id per line after a header).
"""

import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_DATASET = os.environ.get("BENCH_REAL_DATASET", "")
_GENES = os.environ.get("SVPGS_POOLED_GENES", "")
pytestmark = pytest.mark.skipif(
    os.environ.get("SVPGS_REAL_REGRESSION") != "1" or not Path(_DATASET).is_dir() or not Path(_GENES).is_file(),
    reason="real-gene regression cases run on request where bench-real's dataset is",
)


def _method():
    spec = importlib.util.spec_from_file_location("svpgs_method_regression", Path(__file__).resolve().parents[1] / "benchmarks" / "svpgs_method.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
@pytest.mark.parametrize(("split", "refused_gene"), [("loso/AMR", 13), ("loso/EUR", 6), ("loso/EAS", 7), ("loso/SAS", 15), ("loso/AFR", None)])
def test_the_pooled_start_fixed_point_is_reached_where_it_refused(split, refused_gene, monkeypatch):
    from benchmarks.bench_real import harness
    from sv_pgs import fit_model
    from sv_pgs.pooled_fit import GeneData, _PooledFixedPoints, _gene_rows, _held_bytes, _pooled_start, pooled_prior
    from sv_pgs.small_n import dense_statistics

    method = _method()
    dataset = harness.Dataset(_DATASET)
    wanted = set(pd.read_csv(_GENES, sep="\t")["gene_id"])
    rows = [row for row in dataset.gene_rows(["chr22"]) if dataset.genes.iloc[row]["gene_id"] in wanted]
    trains = harness._LazyTrains(dataset, rows, split, "snv")
    genes = []
    for index in range(len(rows)):
        train = trains[index]
        genotypes = np.asarray(train.genotypes)
        genes.append(GeneData(
            codes=genotypes.astype(np.uint8) * np.uint8(127), covariates=method.bench_real_covariates(train),
            target=np.asarray(train.phenotype, dtype=np.float64), variant_class=method.bench_real_classes_for_arm(train.variants, "full"),
        ))
    statistics = [dense_statistics(gene.codes, gene.covariates, gene.target) for gene in genes]
    offsets = [np.zeros(gene.codes.shape[1]) for gene in genes]
    residual = np.array([float(gene.projected_target @ gene.projected_target) / (gene.sample_count - gene.covariates.shape[1]) for gene in statistics])
    draw_count = fit_model.DRAW_COUNT
    prior = pooled_prior(statistics, [gene.variant_class for gene in genes], offsets, residual, draw_count)
    start, noise = _pooled_start(statistics, prior, _gene_rows(statistics))
    working = int(os.environ.get("RUNQ_MEM_BYTES", str(8 << 30))) - sum(_held_bytes(gene) + _held_bytes(gene.design) for gene in statistics)
    oracle = _PooledFixedPoints(statistics, prior, start, noise, draw_count, working // 2)
    # Every double loop's EC free energy must fall at every outer step, to rounding (lead: the rwAMR decrease assertion).
    from sv_pgs import small_n

    traces: list[list[float]] = []
    original = small_n.double_loop_sites

    def traced(*arguments, **keywords):
        trace: list[float] = []
        traces.append(trace)
        return original(*arguments, **(keywords | {"trace": trace}))

    monkeypatch.setattr(small_n, "double_loop_sites", traced)
    wall, cpu = time.perf_counter(), time.process_time()
    (point,) = oracle([start])
    for trace in traces:
        values = np.asarray(trace, dtype=np.float64)
        rounding = np.finfo(np.float64).eps * np.maximum(np.abs(values[1:]), 1.0) * values.shape[0]
        assert np.all(values[1:] <= values[:-1] + rounding), f"the double loop's free energy rose: {values.tolist()}"
    record = {
        "split": split, "genes": len(genes), "members": [int(rows.stop - rows.start) for rows in oracle.rows], "wall_s": time.perf_counter() - wall,
        "cpu_s": time.process_time() - cpu, "gene_cpu_seconds": oracle.gene_cpu_seconds.tolist(), "tilted_seconds": oracle.profile["tilted_seconds"],
        "gene_refreshes": [int(gene.profile["refreshes"]) for gene in oracle.genes], "double_loops": oracle.profile["double_loops"], "double_loop_traces": len(traces),
        "refusals": oracle.refusals, "reached": point is not None,
    }
    if os.environ.get("SVPGS_PROFILE_OUT"):
        with open(os.environ["SVPGS_PROFILE_OUT"], "a") as handle:
            handle.write(json.dumps(record) + "\n")
    assert point is not None, f"refused: {oracle.refusals}"
    assert oracle.mean_move <= 1.0 and oracle.noise_gain <= 0.5 / draw_count
    assert refused_gene is None or refused_gene < len(genes)
