"""EM-VAMP prototype (Rangan-Schniter-Fletcher VAMP with EM on the prior's mixing weights and the noise), dense and
exact through the SVD of the standardized design; own code for a mechanism check before the genome oracle.
Prior: beta_j ~ sum_k pi_k N(0, s_k^2 u_j) on a log-variance lattice (a point mass at 0 included), pi learned by EM."""
import numpy as np


def vamp(X, y, offsets=None, lattice=None, iterations=500, damping=0.7, tolerance=1e-7, verbose=False):
    n, p = X.shape
    u = np.ones(p) if offsets is None else np.exp(offsets)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s2 = s ** 2
    Uy = U.T @ y
    Xy = X.T @ y
    noise = float(y @ y) / n
    if lattice is None:
        # from far below one variant's resolvable effect to the whole phenotype variance on one variant
        top = float(y @ y) / n
        lattice = np.concatenate([[0.0], top * np.logspace(-8, 0, 25)])
    K = lattice.size
    weights = np.full(K, 1.0 / K)
    gamma1 = 1.0 / (np.var(y) / 1.0) * 1e-2 * n  # vague first message
    r1 = np.zeros(p)
    x1_old = np.zeros(p)
    history = []

    def denoise(r, gamma, weights):
        v = lattice[None, :] * u[:, None]                      # p x K prior variances
        total = v + 1.0 / gamma
        log_like = -0.5 * (np.log(total) + r[:, None] ** 2 / total)
        log_post = np.log(np.maximum(weights, 1e-300))[None, :] + log_like
        log_post -= log_post.max(axis=1, keepdims=True)
        resp = np.exp(log_post)
        resp /= resp.sum(axis=1, keepdims=True)
        shrink = v / total
        means_k = shrink * r[:, None]
        vars_k = v / (1.0 + gamma * v)
        mean = np.sum(resp * means_k, axis=1)
        second = np.sum(resp * (vars_k + means_k ** 2), axis=1)
        return mean, np.maximum(second - mean ** 2, 0.0), resp

    x1 = np.zeros(p)
    for it in range(iterations):
        mean1, var1, resp = denoise(r1, gamma1, weights)
        x1 = damping * mean1 + (1 - damping) * x1_old if it else mean1
        a1 = max(float(np.mean(var1)), 1e-300)
        eta1 = 1.0 / a1
        gamma2 = max(eta1 - gamma1, 1e-12)
        r2 = (eta1 * mean1 - gamma1 * r1) / gamma2
        # LMMSE: A = X'X / noise + gamma2 I
        d = s2 / noise + gamma2
        rhs = Xy / noise + gamma2 * r2
        Vr = Vt @ rhs
        x2 = Vt.T @ (Vr / d) + (rhs - Vt.T @ Vr) / gamma2
        trace = float(np.sum(1.0 / d) + (p - s.size) / gamma2)
        a2 = trace / p
        eta2 = 1.0 / a2
        gamma1_new = max(eta2 - gamma2, 1e-12)
        r1_new = (eta2 * x2 - gamma2 * r2) / gamma1_new
        r1 = damping * r1_new + (1 - damping) * r1
        gamma1 = damping * gamma1_new + (1 - damping) * gamma1
        # EM: mixing weights from the denoiser's responsibilities, the noise from the LMMSE posterior
        weights = resp.mean(axis=0)
        residual = y - X @ x2
        noise = float(residual @ residual + noise * (p - gamma2 * trace)) / n
        noise = max(noise, 1e-12)
        change = float(np.linalg.norm(x1 - x1_old) / max(np.linalg.norm(x1), 1e-300))
        x1_old = x1
        history.append(change)
        if verbose and it % 20 == 0:
            print(it, change, gamma1, gamma2, noise, flush=True)
        if it > 10 and change < tolerance:
            break
    return {"mean": mean1, "lmmse_mean": x2, "noise": noise, "weights": weights, "iterations": it + 1, "last_change": history[-1],
            "gamma1": gamma1, "gamma2": gamma2}
