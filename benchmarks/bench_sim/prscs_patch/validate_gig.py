"""Checks gigrnd.gigrnd_vec against PRS-CS's scalar gigrnd.gigrnd: at several (p, a, b), 1e5 draws of each, the
two-sample KS test between them and the one-sample KS test of each against scipy's exact GIG distribution
(geninvgauss(p, sqrt(a b)) scaled by sqrt(b / a)). Also times both over 232,217 draws at PRS-CS-like parameters."""
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, "/scratch.global/sauer354/svpgs-team/agents/baselines-genome/soft/PRScs-master")
import gigrnd  # noqa: E402

np.random.seed(7)
DRAWS = 100_000
cases = [(0.5, 2.0, 1e-6), (0.5, 2.0, 1.0), (0.5, 1e-3, 50.0), (0.5, 20.0, 1e3), (0.5, 0.3, 4e4), (-0.7, 1.0, 1.0),
         (0.0, 1.0, 1.0), (2.5, 0.1, 0.1)]
for p, a, b in cases:
    scalar = np.array([gigrnd.gigrnd(p, a, b) for _ in range(DRAWS)])
    vector = gigrnd.gigrnd_vec(p, np.full(DRAWS, a), np.full(DRAWS, b))
    exact = stats.geninvgauss(p, np.sqrt(a * b), scale=np.sqrt(b / a)).cdf
    two = stats.ks_2samp(scalar, vector)
    print(f"p={p} a={a} b={b}: two-sample KS D={two.statistic:.4f} P={two.pvalue:.3f} | vs exact: "
          f"scalar P={stats.kstest(scalar, exact).pvalue:.3f} vector P={stats.kstest(vector, exact).pvalue:.3f} | "
          f"means {scalar.mean():.4g} {vector.mean():.4g}", flush=True)
# PRS-CS-like parameters: p = a - 0.5 with a = 1, a = 2 delta, b = n beta^2 / sigma over 232,217 variants
count = 232_217
delta = np.random.gamma(1.5, 1.0, count)
bb = 40_000 * (np.random.standard_normal(count) * 1e-3) ** 2
started = time.time(); gigrnd.gigrnd_vec(0.5, 2 * delta, bb); vector_seconds = time.time() - started
started = time.time()
for jj in range(20_000):
    gigrnd.gigrnd(0.5, 2 * delta[jj], bb[jj])
scalar_seconds = (time.time() - started) * count / 20_000
print(f"{count} draws: vector {vector_seconds:.2f} s, scalar loop {scalar_seconds:.1f} s (extrapolated)")
