import os, time
import numpy as np
from scipy.linalg.lapack import dpotrf, dtrtri, spotrf, strtri
from threadpoolctl import threadpool_limits, threadpool_info
print([(i["internal_api"], i["num_threads"], i.get("architecture"), i.get("filepath", "")[-45:]) for i in threadpool_info()], flush=True)
width = 4096
lags = np.abs(np.arange(width)[:, None] - np.arange(width)[None, :])
R = 0.95 ** lags
system = 1.5e5 * R + np.diag(np.random.default_rng(0).uniform(1e5, 1e8, width))
def t(label, fn, flops, reps=3):
    fn(); start = time.perf_counter()
    for _ in range(reps): fn()
    seconds = (time.perf_counter() - start) / reps
    print(f"  {label}: {seconds*1e3:.0f} ms  {flops/seconds/1e9:.0f} GFLOP/s", flush=True)
cube = width ** 3
for threads in (1, 8, 16, 32):
    with threadpool_limits(limits=threads):
        print(f"threads={threads}", flush=True)
        t("numpy dgemm", lambda: system @ system, 2 * cube)
        t("scipy dpotrf", lambda: dpotrf(system, lower=1, clean=1), cube / 3)
        t("numpy cholesky", lambda: np.linalg.cholesky(system), cube / 3)
        factor = np.asfortranarray(np.linalg.cholesky(system))
        t("scipy dtrtri", lambda: dtrtri(factor, lower=1), cube / 3)
        single = system.astype(np.float32)
        t("scipy spotrf", lambda: spotrf(single, lower=1, clean=1), cube / 3)
        sfactor = np.asfortranarray(np.linalg.cholesky(single))
        t("scipy strtri", lambda: strtri(sfactor, lower=1), cube / 3)
