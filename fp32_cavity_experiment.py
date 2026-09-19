"""fp32 block posterior with the cancellation-free cavity precision vs fp64 LAPACK."""
import time
import numpy as np
from scipy.linalg.lapack import dpotrf, dpotrs, dtrtri, spotrf, spotrs, strtri

rng = np.random.default_rng(0)
width = 3000
lags = np.abs(np.arange(width)[:, None] - np.arange(width)[None, :])
correlation = 0.97 ** lags
# near-duplicate pairs (r ~ 0.9995) sprinkled through the block
for anchor in rng.choice(width - 1, 40, replace=False):
    partner = anchor + 1
    correlation[anchor, partner] = correlation[partner, anchor] = 0.9995
eigenvalues, vectors = np.linalg.eigh(correlation)
correlation = (vectors * np.maximum(eigenvalues, 1e-6)) @ vectors.T
correlation = 0.5 * (correlation + correlation.T)
scale = 1.0e5
precision = rng.uniform(3e6, 3e7, size=width)
signals = rng.choice(width, 30, replace=False)
precision[signals] = 0.0
precision[signals[:10]] = rng.uniform(1.0, 100.0, size=10)
linear = rng.standard_normal(width) * 300.0

def fp64():
    system = scale * correlation
    system[np.diag_indices(width)] += precision
    factor, info = dpotrf(system, lower=1, clean=1)
    mean, _ = dpotrs(factor, linear, lower=1)
    inverse, _ = dtrtri(factor, lower=1)
    variance = np.einsum("ij,ij->j", inverse, inverse)
    return mean, variance, 1.0 / variance - precision

def fp32():
    data_diagonal = scale * np.diag(correlation)
    diagonal = data_diagonal + precision
    root = 1.0 / np.sqrt(diagonal)
    scaled = (scale * correlation) * root[:, None] * root[None, :]
    scaled[np.diag_indices(width)] = 1.0
    factor, info = spotrf(scaled.astype(np.float32), lower=1, clean=1)
    assert info == 0, info
    inverse, info = strtri(factor, lower=1)
    factor64 = factor.astype(np.float64)
    inverse64 = inverse.astype(np.float64)
    factor_diagonal = np.diag(factor64)
    below_factor = np.tril(factor64, -1)
    below_inverse = np.tril(inverse64, -1)
    excess = np.einsum("ij,ij->i", below_factor, below_factor) / factor_diagonal**2 + np.einsum("ij,ij->j", below_inverse, below_inverse)
    variance = root**2 * (1.0 + excess)
    cavity = (data_diagonal - precision * excess) / (1.0 + excess)
    # mean: fp32 solve of the scaled system, two fp64 refinement steps
    system64 = scale * correlation
    system64[np.diag_indices(width)] += precision
    mean = root * spotrs(factor, (root * linear).astype(np.float32), lower=1)[0].astype(np.float64)
    for _ in range(2):
        residual = linear - system64 @ mean
        mean = mean + root * spotrs(factor, (root * residual).astype(np.float32), lower=1)[0].astype(np.float64)
    return mean, variance, cavity

reference = fp64()
candidate = fp32()
for label, exact, approximate in zip(("mean", "variance", "cavity precision"), reference, candidate):
    relative = np.abs(approximate - exact) / np.maximum(np.abs(exact), 1e-300)
    print(f"{label}: max rel err {relative.max():.2e}, median {np.median(relative):.2e}, at signals {relative[signals].max():.2e}")
naive_cavity = 1.0 / candidate[1] - precision
relative = np.abs(naive_cavity - reference[2]) / np.abs(reference[2])
print(f"naive fp32 cavity (1/variance - precision): max rel err {relative.max():.2e}, negative {np.sum(naive_cavity < 0)}")
for fn, label in ((fp64, "fp64"), (fp32, "fp32")):
    start = time.perf_counter(); fn(); fn(); print(label, f"{(time.perf_counter() - start) / 2:.2f}s")
