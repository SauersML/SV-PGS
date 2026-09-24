"""Validates mcmc_gtb.mcmc's GPU paths against its CPU path: the same PRS-CS sampler (scalar GIG draws, so all three
share one random stream) on synthetic correlation blocks of several sizes, 5 iterations from seed 1, run on the CPU,
on one GPU, and on every visible GPU; reports the largest relative differences of the posterior means of beta and psi
and of sigma and phi. Run as: validate_multi.py <mode> <out.npz> with mode cpu or gpu (PRSCS_DEVICE and
CUDA_VISIBLE_DEVICES set by the caller); compare.py-style comparison at the end when all three files exist."""
import os
import sys

import numpy as np

sys.path.insert(0, "/scratch.global/sauer354/svpgs-team/agents/baselines-genome/soft/PRScs-master")
import mcmc_gtb  # noqa: E402

mode, out = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(11)
sizes = [3000, 1200, 2500, 800, 1800]
blocks = []
for size in sizes:
    x = rng.standard_normal((size, 3 * size)) / np.sqrt(3 * size)
    corr = x @ x.T
    d = np.sqrt(np.diag(corr))
    blocks.append(corr / d[:, None] / d[None, :])
p = sum(sizes)
n = 40000
beta_true = np.where(rng.random(p) < 0.01, rng.standard_normal(p) * 0.02, 0.0)
mrg = np.concatenate([b @ beta_true[s:s + len(b)] for b, s in zip(blocks, np.cumsum([0] + sizes[:-1]))])
mrg = mrg + rng.standard_normal(p) / np.sqrt(n)
sst = {"SNP": [f"s{i}" for i in range(p)], "BETA": list(mrg), "MAF": [0.2] * p, "BP": list(range(p)),
       "A1": ["A"] * p, "A2": ["C"] * p}
beta, psi, sigma, phi = mcmc_gtb.mcmc(1, 0.5, None, sst, n, [b.copy() for b in blocks], sizes, 5, 0, 1, 22,
                                      os.path.join(os.environ.get("TMPDIR", "/tmp"), f"val_{mode}"), "TRUE", "FALSE",
                                      "FALSE", 1)
np.savez(out, beta=beta, psi=psi, sigma=np.squeeze(sigma), phi=np.squeeze(phi))
print(mode, "devices", os.environ.get("CUDA_VISIBLE_DEVICES"), "saved", out, flush=True)
