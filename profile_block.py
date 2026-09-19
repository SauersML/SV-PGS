import time
import numpy as np
from scipy.linalg.lapack import dpotrf, dpotrs, dtrtri
import scipy, numpy
from sv_pgs import ld_space_fit as L
np.show_config() if False else None
print(numpy.__version__, scipy.__version__)
try:
    from threadpoolctl import threadpool_info
    print([ (i["internal_api"], i["num_threads"], i.get("filepath","")[-60:]) for i in threadpool_info()])
except ImportError:
    print("no threadpoolctl")
width = 3333
lags = np.abs(np.arange(width)[:, None] - np.arange(width)[None, :])
R = 0.95 ** lags
rng = np.random.default_rng(0)
precision = rng.uniform(1e5, 1e8, size=width)
linear = rng.standard_normal(width)
def t(label, fn, reps=3):
    fn(); start = time.perf_counter()
    for _ in range(reps): out = fn()
    print(f"{label}: {(time.perf_counter()-start)/reps*1e3:.1f} ms", flush=True); return out
system = 1.5e5 * R; system[np.diag_indices_from(system)] += precision
t("copy+scale", lambda: 1.5e5 * R)
t("dpotrf", lambda: dpotrf(system.T.copy(order="F"), lower=1, clean=1, overwrite_a=1))
factor, _ = dpotrf(system.T.copy(order="F"), lower=1, clean=1)
t("dpotrs", lambda: dpotrs(factor, linear, lower=1))
t("dtrtri", lambda: dtrtri(factor.copy(order="F"), lower=1, overwrite_c=1))
inv, _ = dtrtri(factor, lower=1)
t("einsum", lambda: np.einsum("ij,ij->j", inv, inv))
t("square-sum", lambda: np.sum(inv * inv, axis=0))
t("dgemm", lambda: system @ system)
host = L._HostBackend()
t("block_posterior", lambda: host.block_posterior(R, 1.5e5, precision, linear))
grid = L._log_local_scale_grid(0.5, np.array([0.5, 0.4]))
cls = np.zeros(width, dtype=np.int64)
t("tilted", lambda: L._tilted_weights(rng.uniform(1e4, 1e5, width), rng.standard_normal(width) * 100, np.full(width, 1e-8), grid.class_log_prior_mass[cls], grid.local_scale))
