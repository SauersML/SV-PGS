"""EM-VAMP against SV-PGS and the ridge on the gene-scale polygenic simulation (prototype; the head is lead/poly_sim.py). Polygenic failure at gene scale (own simulation on real genotypes; mechanism only): 5% of a gene window's variants
causal with normal effects, h2 = 0.3, trained on 80% of the non-African people and scored on the other 20% (same
ancestries). SV-PGS as fitted (``fit_small_n``), SV-PGS with the density coefficients re-fitted at fixed finite
smoothing weights (block ascent of the penalized ELBO), empirical-Bayes ridge, and the true genetic value's ceiling:
r2 and calibration slope. Usage: poly_sim.py <gene> <replicate>."""
import sys, json
from dataclasses import replace
import numpy as np
from scipy.optimize import minimize_scalar
gene, replicate = sys.argv[1], int(sys.argv[2])
sys.argv = [sys.argv[0], "-", "-", "1"]
from benchmarks.bench_real import harness
from benchmarks import svpgs_small_n as bench
from sv_pgs import small_n
from sv_pgs import scale_mixture_ep as engine
from sv_pgs.mean_field import MeanFieldFixedPoints
dataset = harness.Dataset("/scratch.global/sauer354/svpgs-team/bench-real/dataset")
chrom = dataset.genes.loc[dataset.genes.gene_id == gene, "chrom"].iat[0]
row = [v for v in dataset.gene_rows([chrom]) if dataset.genes.iloc[v]["gene_id"] == gene][0]
train_all, test_all, _tp, _ti = harness.build_gene_task(dataset, harness.load_gene_window(dataset, row), dataset.splits["loso/AFR"])
train, test = harness.subset(train_all, test_all, "snv", "loso/AFR")
G = np.asarray(train.genotypes, dtype=np.float64)
rng = np.random.default_rng(100 * replicate + 3)
keep = G.std(axis=0) > 0
G = G[:, keep]
p = G.shape[1]
causal = rng.random(p) < 0.05
Z = (G - G.mean(axis=0)) / G.std(axis=0)
beta = np.where(causal, rng.normal(size=p), 0.0)
genetic = Z @ beta
genetic *= np.sqrt(0.3 / np.var(genetic))
y = genetic + rng.normal(0, np.sqrt(0.7), G.shape[0])
held = rng.random(G.shape[0]) < 0.2
rows = ~held
def score(prediction):
    t = genetic[held]
    r2 = float(np.corrcoef(prediction, t)[0, 1] ** 2) if np.std(prediction) > 0 else 0.0
    slope = float(np.cov(y[held], prediction)[0, 1] / np.var(prediction, ddof=1)) if np.std(prediction) > 0 else float("nan")
    return round(r2, 4), round(slope, 3)
out = {"gene": gene, "replicate": replicate, "p": p, "causal": int(causal.sum())}
# --- EM-VAMP on the same simulation (standardized training design, intercept projected)
sys.path.insert(0, __import__("os").path.dirname(__file__))
from vamp_core import vamp
import time
mu, sd = Z[rows].mean(axis=0), Z[rows].std(axis=0)
Xt = (Z[rows] - mu) / sd
yt = y[rows] - y[rows].mean()
started = time.time()
result = vamp(Xt, yt)
Xh = (Z[held] - mu) / sd
out["vamp"] = score(Xh @ result["mean"])
out["vamp_lmmse"] = score(Xh @ result["lmmse_mean"])
out["vamp_iterations"] = result["iterations"]; out["vamp_last_change"] = result["last_change"]
out["vamp_seconds"] = round(time.time() - started, 1)
print("VAMP", json.dumps(out), flush=True)
