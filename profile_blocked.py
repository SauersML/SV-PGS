"""Blocked Cholesky / inverse-diagonal built on GEMM vs OpenBLAS potrf/trtri (dedicated node)."""
import time
import numpy as np
from scipy.linalg.lapack import dpotrf, dtrtri
from scipy.linalg.blas import dsyrk, dgemm, dtrsm
from threadpoolctl import threadpool_limits

width = 4096
lags = np.abs(np.arange(width)[:, None] - np.arange(width)[None, :])
R = 0.95 ** lags
rng = np.random.default_rng(0)
system = 1.5e5 * R + np.diag(rng.uniform(1e5, 1e8, width))
reference_factor = np.linalg.cholesky(system)
reference_diagonal = np.diag(np.linalg.inv(system))
cube = width ** 3


def timed(label, fn, flops, reps=3):
    fn()
    start = time.perf_counter()
    for _ in range(reps):
        out = fn()
    seconds = (time.perf_counter() - start) / reps
    print(f"  {label}: {seconds * 1e3:.0f} ms  {flops / seconds / 1e9:.0f} GFLOP/s", flush=True)
    return out


def blocked_cholesky(matrix, panel):
    """Right-looking: factor the diagonal panel, solve the panel below with its inverse, GEMM-update the trailing matrix."""
    work = np.array(matrix, order="C", copy=True)
    size = work.shape[0]
    for start in range(0, size, panel):
        stop = min(start + panel, size)
        diagonal_factor, status = dpotrf(work[start:stop, start:stop], lower=1, clean=1)
        assert status == 0
        work[start:stop, start:stop] = diagonal_factor
        if stop < size:
            inverse_factor, status = dtrtri(diagonal_factor, lower=1)
            below = work[stop:, start:stop] @ inverse_factor.T
            work[stop:, start:stop] = below
            work[stop:, stop:] -= below @ below.T
    return np.tril(work)


def blocked_inverse_diagonal(factor, panel):
    """diag(A^-1) = column norms of L^-1; L^-1 by block forward substitution with GEMMs."""
    size = factor.shape[0]
    inverse = np.zeros_like(factor)
    for start in range(0, size, panel):
        stop = min(start + panel, size)
        diagonal_inverse, status = dtrtri(np.asfortranarray(factor[start:stop, start:stop]), lower=1)
        inverse[start:stop, start:stop] = diagonal_inverse
        if start > 0:
            # (L^-1)[s:e, :s] = -L_ss^-1 L[s:e, :s] (L^-1)[:s, :s]
            inverse[start:stop, :start] = -diagonal_inverse @ (factor[start:stop, :start] @ inverse[:start, :start])
    return np.einsum("ij,ij->j", inverse, inverse)


for threads in (16, 32):
    with threadpool_limits(limits=threads):
        print(f"threads={threads}", flush=True)
        timed("openblas dpotrf", lambda: dpotrf(system, lower=1, clean=1), cube / 3)
        for panel in (256, 512, 1024):
            factor = timed(f"blocked cholesky panel={panel}", lambda: blocked_cholesky(system, panel), cube / 3)
            print(f"    max rel err vs numpy {np.max(np.abs(factor - reference_factor)) / np.max(np.abs(reference_factor)):.1e}")
        fortran_factor = np.asfortranarray(reference_factor)
        timed("openblas dtrtri", lambda: dtrtri(fortran_factor, lower=1), cube / 3)
        for panel in (256, 512, 1024):
            diagonal = timed(f"blocked inverse diagonal panel={panel}", lambda: blocked_inverse_diagonal(reference_factor, panel), cube / 3)
            print(f"    max rel err {np.max(np.abs(diagonal / reference_diagonal - 1)):.1e}")
        timed("numpy dgemm", lambda: system @ system, 2 * cube)
        timed("scipy dsyrk", lambda: dsyrk(1.0, system, lower=1), cube)
