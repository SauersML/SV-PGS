"""Grid tilted moments (density rule on an even log-lambda grid) vs quadrature of the untruncated BetaPrime prior."""
import numpy as np
from scipy import integrate
from scipy.special import betaln, logsumexp

def grid_moments(m, v, u, a, b, lower, upper, step):
    t = np.arange(lower, upper + 0.5 * step, step)
    logp = a * t - (a + b) * np.logaddexp(0, t)
    logp -= logsumexp(logp)
    slab = u * np.exp(t)
    tot = v + slab
    lw = logp - 0.5 * np.log(2 * np.pi * tot) - 0.5 * m * m / tot
    lz = logsumexp(lw)
    w = np.exp(lw - lz)
    s = slab / tot
    e = m * np.sum(w * s)
    var = v * np.sum(w * s) + m * m * np.sum(w * (s - np.sum(w * s)) ** 2)
    return lz, e, var

def quad_moments(m, v, u, a, b):
    def logprior(t):
        return a * t - (a + b) * np.logaddexp(0, t) - betaln(a, b)
    def f(t, k):
        slab = u * np.exp(t); tot = v + slab; s = slab / tot
        d = np.exp(logprior(t) - 0.5 * np.log(2 * np.pi * tot) - 0.5 * m * m / tot)
        return d * [1.0, m * s, m * m * s * s + v * s][k]
    pts = [np.log(v / u), np.log(max(m * m, v) / u)]
    rng = (-200.0, 200.0)
    vals = []
    for k in range(3):
        val = 0.0
        edges = sorted([rng[0], *pts, rng[1]])
        for lo, hi in zip(edges[:-1], edges[1:]):
            val += integrate.quad(lambda t: f(t, k), lo, hi, limit=4000, epsabs=0, epsrel=1e-13)[0]
        vals.append(val)
    z, e1, e2 = vals
    return np.log(z), e1 / z, e2 / z - (e1 / z) ** 2

cases = [(0.0, 1e-5, 1e-7, 0.5, 0.5), (0.02, 1e-5, 1e-7, 0.5, 0.5), (0.05, 1e-5, 1e-6, 0.5, 0.3), (0.003, 1e-5, 1e-8, 0.5, 2.0),
         (1.0, 0.5, 0.2, 0.5, 0.8), (0.1, 1e-5, 1e-9, 0.5, 0.1), (0.2, 1e-5, 1e-8, 1.0, 0.5), (0.01, 1e-3, 1e-10, 0.5, 0.4)]
for lower_mass in (10.0, 20.0):
    for step in (0.5, 0.75, 1.0, 1.25):
        worst = [0.0, 0.0, 0.0]
        for m, v, u, a, b in cases:
            g = grid_moments(m, v, u, a, b, -lower_mass / a, 40.0, step)
            q = quad_moments(m, v, u, a, b)
            worst[0] = max(worst[0], abs(g[0] - q[0]))
            worst[1] = max(worst[1], abs(g[1] - q[1]) / max(abs(q[1]), 1e-3 * np.sqrt(v)))
            worst[2] = max(worst[2], abs(g[2] / q[2] - 1))
        print(f"lower=-{lower_mass}/a step={step}: max |dlogZ|={worst[0]:.2e} max rel dE={worst[1]:.2e} max rel dV={worst[2]:.2e}")
