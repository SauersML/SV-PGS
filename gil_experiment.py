import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.linalg.lapack import dpotrf, dtrtri
from threadpoolctl import threadpool_limits

width = 1500
lags = np.abs(np.arange(width)[:, None] - np.arange(width)[None, :])
matrices = [0.95 ** lags + np.eye(width) * (1 + index) for index in range(16)]

def scipy_work(matrix):
    factor, _ = dpotrf(matrix, lower=1, clean=1)
    inverse, _ = dtrtri(factor, lower=1)
    return float(np.einsum("ij,ij->j", inverse, inverse).sum())

def numpy_work(matrix):
    factor = np.linalg.cholesky(matrix)
    return float(np.linalg.inv(factor).sum())

with threadpool_limits(limits=1):
    for label, work in (("scipy potrf+trtri", scipy_work), ("numpy cholesky+inv", numpy_work)):
        for workers in (1, 8):
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=workers) as executor:
                list(executor.map(work, matrices))
            print(f"{label} workers={workers}: {time.perf_counter() - start:.2f}s", flush=True)
