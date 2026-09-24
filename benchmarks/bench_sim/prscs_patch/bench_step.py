"""Times each operation of PRS-CS's GPU block step on synthetic SPD blocks of 11k and 25k variants (float64), to find
where the ~1.3 s per iteration goes on an H100: the diagonal add, cupy.linalg.cholesky (copy + potrf + zeroing the
other triangle), a bare in-place cusolver potrf, the two triangular solves and the quadratic form."""
import time

import cupy as cp
import numpy as np
from cupy.cuda import cusolver, device
from cupyx.scipy import linalg as cplinalg


def timed(label, fn, repeat=3):
    fn(); cp.cuda.Device().synchronize()
    started = time.time()
    for _ in range(repeat):
        out = fn()
    cp.cuda.Device().synchronize()
    print(f"  {label}: {(time.time() - started) / repeat * 1000:.1f} ms", flush=True)
    return out


def potrf_inplace(a):
    """cusolver dpotrf on a C-ordered symmetric matrix in place (C order = column-major transpose; symmetric, so the
    'upper' factor of the column-major view is the lower factor of the row-major matrix)."""
    handle = device.get_cusolver_handle()
    n = a.shape[0]
    work = cusolver.dpotrf_bufferSize(handle, 1, n, a.data.ptr, n)  # 1 = CUBLAS_FILL_MODE_UPPER
    workspace = cp.empty(work, dtype=cp.float64)
    info = cp.empty(1, dtype=cp.int32)
    cusolver.dpotrf(handle, 1, n, a.data.ptr, n, workspace.data.ptr, work, info.data.ptr)
    return info


for size in (11000, 25000):
    print(f"block of {size}", flush=True)
    x = cp.random.standard_normal((size, 2 * size), dtype=cp.float64) / np.sqrt(2 * size)
    ld = x @ x.T
    del x
    psi = cp.random.uniform(0.01, 1.0, size)
    beta_mrg = cp.random.standard_normal((size, 1)) * 1e-3
    noise = cp.random.standard_normal((size, 1))
    dinvt = timed("ld + diag(1/psi)", lambda: ld + cp.diag(1.0 / psi))
    lower = timed("cupy.linalg.cholesky", lambda: cp.linalg.cholesky(dinvt))
    work = dinvt.copy()
    timed("bare dpotrf in place (on a copy made outside the timer)", lambda: potrf_inplace(dinvt.copy()), repeat=2)
    timed("copy of the block", lambda: dinvt.copy())
    tmp = timed("solve L x = b", lambda: cplinalg.solve_triangular(lower, beta_mrg, lower=True))
    beta = timed("solve L' x = tmp", lambda: cplinalg.solve_triangular(lower, tmp + 1e-3 * noise, lower=True, trans='T'))
    timed("quadratic form", lambda: cp.dot(cp.dot(beta.T, dinvt), beta))
    flops = size ** 3 / 3
    del ld, dinvt, lower, work
    cp.get_default_memory_pool().free_all_blocks()
    print(f"  (potrf flops {flops:.2e})", flush=True)
