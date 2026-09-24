"""Validates PRS-CS's GPU block step against its CPU step on one real block (the first 4 cM block of the all-variant
set of scenario_000's training people, built by prscs_auto_allvar's own code): two chains of PRS-CS's sampler (the
loop of mcmc_gtb.mcmc) from the same seed, one with block_step_cpu and one with block_step_gpu, every draw made on the
host. Reports the largest relative differences of beta, psi, sigma and phi after each of 5 iterations, and the GPU
parse symmetrization against the CPU one. Also times one GPU block step at this block's size."""
import os
import sys
import time
from pathlib import Path

import numpy as np

A = Path("/scratch.global/sauer354/svpgs-team/agents/baselines-genome")
sys.path.insert(0, str(A / "code"))
sys.path.insert(0, str(A / "soft" / "PRScs-master"))
os.environ["PRSCS_DEVICE"] = "gpu"
import cupy as cp  # noqa: E402
from scipy import linalg  # noqa: E402

import gigrnd  # noqa: E402
import mcmc_gtb  # noqa: E402
from benchmarks.bench_sim.harness import (ARMS, TrainData, covariate_matrix, measured_records,  # noqa: E402
                                          public_variant_table)
from benchmarks.bench_sim.submissions.ldpred2_auto_allvar import Adjuster  # noqa: E402
from benchmarks.bench_sim.submissions.sbayesrc import BLOCK_CM  # noqa: E402

C = Path("/scratch.global/sauer354/svpgs-team/bench-sim/v7/cohort/chr22")
S = Path("/scratch.global/sauer354/svpgs-team/bench-sim/v7/dev/scenario_000")
samples = np.load(C / "samples.npz")
train_columns = np.flatnonzero(~samples["is_test"])
covariates, names = covariate_matrix(C, "truth")
train = TrainData(variants=public_variant_table(C, "truth"), covariates=covariates[train_columns], covariate_names=names,
                  phenotype=np.load(S / "truth.npz")["phenotype"][train_columns].copy(), trait_type="quantitative",
                  prevalence=None, cores=16, truth_half=np.zeros(train_columns.size, dtype=bool), reads=None,
                  _observed=np.load(C / ARMS["truth"][0], mmap_mode="r"), _columns=train_columns,
                  _records=np.flatnonzero(measured_records(C)))
# the first 4 cM block of common records (sbayesrc.common_rows restricted to the chromosome's start)
cm = np.asarray(train.variants["cm"], dtype=np.float64)
order = np.argsort(cm, kind="stable")
head = order[cm[order] < cm[order[0]] + BLOCK_CM]
frequency = np.asarray(train.codes(head), dtype=np.float32).mean(axis=1) / 127.0 / 2.0
rows = head[np.minimum(frequency, 1 - frequency) >= 0.01]
adjuster = Adjuster(train)
parts = [adjuster.block(rows[i:i + 4096]) for i in range(0, rows.size, 4096)]
beta = np.concatenate([p[1] for p in parts]); se = np.concatenate([p[2] for p in parts])
unit = np.vstack([p[4] for p in parts]).astype(np.float64)
ld = unit @ unit.T
np.fill_diagonal(ld, 1.0)
n = train.n_samples
print(f"block of {rows.size} variants, n = {n}", flush=True)

# the parse step's symmetrization, CPU (scipy gesdd) against GPU (cusolver gesvd)
_, s, v = linalg.svd(ld)
cpu_sym = (ld + np.dot(v.T, np.dot(np.diag(s), v))) / 2
block = cp.asarray(ld)
_, gs, gv = cp.linalg.svd(block)
gpu_sym = cp.asnumpy((block + cp.dot(gv.T, gs[:, None] * gv)) / 2)
print("parse symmetrization: max |gpu - cpu| / max |cpu| =", np.abs(gpu_sym - cpu_sym).max() / np.abs(cpu_sym).max())
# the same symmetrization by the symmetric eigendecomposition: for symmetric A = Q L Q', V' diag(s) V = Q |L| Q'
cp.cuda.Device().synchronize(); started = time.time()
_, gs, gv = cp.linalg.svd(block); cp.cuda.Device().synchronize(); svd_seconds = time.time() - started
started = time.time()
w, q = cp.linalg.eigh(block)
eig_sym = cp.asnumpy((block + cp.dot(q * cp.abs(w)[None, :], q.T)) / 2); eig_seconds = time.time() - started
print("parse symmetrization by eigh: max |gpu eigh - cpu svd| / max |cpu| =", np.abs(eig_sym - cpu_sym).max() / np.abs(cpu_sym).max(),
      f"| GPU svd {svd_seconds:.1f} s, GPU eigh {eig_seconds:.1f} s at {rows.size} variants", flush=True)
del gs, gv, w, q
ld = cpu_sym

beta_mrg = (beta / (se * np.sqrt(n)))[:, None]
p = rows.size
a, b = 1, 0.5


def chain(step, seed, iterations):
    random = np.random.RandomState(seed)
    ld_device = cp.asarray(ld) if step is mcmc_gtb.block_step_gpu else ld
    beta_c = np.zeros((p, 1)); psi = np.ones((p, 1)); sigma = 1.0; phi = 1.0
    states = []
    for _ in range(iterations):
        noise = random.randn(p, 1)
        beta_c[:], quad = step(ld_device, psi, beta_mrg, noise, sigma, n)
        err = max(n/2.0*(1.0-2.0*sum(beta_c*beta_mrg)+quad), n/2.0*sum(beta_c**2/psi))
        sigma = 1.0/random.gamma((n+p)/2.0, 1.0/err)
        delta = random.gamma(a+b, 1.0/(psi+phi))
        np.random.seed(random.randint(2**31))  # gigrnd draws from numpy's global generator
        for jj in range(p):
            psi[jj] = gigrnd.gigrnd(a-0.5, 2.0*delta[jj], n*beta_c[jj]**2/sigma)
        psi[psi > 1] = 1.0
        w = random.gamma(1.0, 1.0/(phi+1.0))
        phi = random.gamma(p*b+0.5, 1.0/(sum(delta)+w))
        states.append((beta_c.copy(), psi.copy(), float(np.squeeze(sigma)), float(np.squeeze(phi))))
    return states


cpu, gpu = chain(mcmc_gtb.block_step_cpu, 1, 5), chain(mcmc_gtb.block_step_gpu, 1, 5)
worst = 0.0
for itr, (c, g) in enumerate(zip(cpu, gpu), start=1):
    rel = [np.abs(g[0] - c[0]).max() / np.abs(c[0]).max(), np.abs(g[1] - c[1]).max() / np.abs(c[1]).max(),
           abs(g[2] - c[2]) / abs(c[2]), abs(g[3] - c[3]) / abs(c[3])]
    worst = max(worst, *rel)
    print(f"iteration {itr}: max relative difference beta {rel[0]:.2e} psi {rel[1]:.2e} sigma {rel[2]:.2e} phi {rel[3]:.2e}")
print("GPU step equals CPU step to rtol 1e-10 over 5 iterations:", worst < 1e-10, flush=True)

psi = np.ones((p, 1)); noise = np.random.randn(p, 1); ld_device = cp.asarray(ld)
for label, step, matrix in (("gpu", mcmc_gtb.block_step_gpu, ld_device), ("cpu", mcmc_gtb.block_step_cpu, ld)):
    step(matrix, psi, beta_mrg, noise, 1.0, n)
    cp.cuda.Device().synchronize()
    started = time.time()
    for _ in range(3):
        step(matrix, psi, beta_mrg, noise, 1.0, n)
    cp.cuda.Device().synchronize()
    print(f"{label} block step at {p} variants: {(time.time() - started) / 3:.3f} s", flush=True)
started = time.time()
for jj in range(p):
    gigrnd.gigrnd(a-0.5, 2.0*1.0, n*beta_mrg[jj, 0]**2)
print(f"gigrnd over {p} variants: {time.time() - started:.3f} s")
