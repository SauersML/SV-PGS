#!/usr/bin/env python

"""
Random variate generator for the generalized inverse Gaussian distribution.
Reference: L Devroye. Random variate generation for the generalized inverse Gaussian distribution.
           Statistics and Computing, 24(2):239-246, 2014.

"""


import math
from numpy import random


def psi(x, alpha, lam):
    f = -alpha*(math.cosh(x)-1.0)-lam*(math.exp(x)-x-1.0)
    return f


def dpsi(x, alpha, lam):
    f = -alpha*math.sinh(x)-lam*(math.exp(x)-1.0)
    return f


def g(x, sd, td, f1, f2):
    if (x >= -sd) and (x <= td):
        f = 1.0
    elif x > td:
        f = f1
    elif x < -sd:
        f = f2

    return f


def gigrnd(p, a, b):
    # setup -- sample from the two-parameter version gig(lam,omega)
    p = float(p); a = float(a); b = float(b)
    lam = p
    omega = math.sqrt(a*b)

    if lam < 0:
        lam = -lam
        swap = True
    else:
        swap = False

    alpha = math.sqrt(math.pow(omega,2)+math.pow(lam,2))-lam

    # find t
    x = -psi(1.0, alpha, lam)
    if (x >= 0.5) and (x <= 2.0):
        t = 1.0
    elif x > 2.0:
        if (alpha == 0) and (lam == 0):
            t = 1.0
        else:
            t = math.sqrt(2.0/(alpha+lam))
    elif x < 0.5:
        if (alpha == 0) and (lam == 0):
            t = 1.0
        else:
            t = math.log(4.0/(alpha+2.0*lam))

    # find s
    x = -psi(-1.0, alpha, lam)
    if (x >= 0.5) and (x <= 2.0):
        s = 1.0
    elif x > 2.0:
        if (alpha == 0) and (lam == 0):
            s = 1.0
        else:
            s = math.sqrt(4.0/(alpha*math.cosh(1)+lam))
    elif x < 0.5:
        if (alpha == 0) and (lam == 0):
            s = 1.0
        elif alpha == 0:
            s = 1.0/lam
        elif lam == 0:
            s = math.log(1.0+1.0/alpha+math.sqrt(1.0/math.pow(alpha,2)+2.0/alpha))
        else:
            s = min(1.0/lam, math.log(1.0+1.0/alpha+math.sqrt(1.0/math.pow(alpha,2)+2.0/alpha)))

    # find auxiliary parameters
    eta = -psi(t, alpha, lam)
    zeta = -dpsi(t, alpha, lam)
    theta = -psi(-s, alpha, lam)
    xi = dpsi(-s, alpha, lam)

    p = 1.0/xi
    r = 1.0/zeta

    td = t-r*eta
    sd = s-p*theta
    q = td+sd

    # random variate generation
    while True:
        U = random.random()
        V = random.random()
        W = random.random()
        if U < q/(p+q+r):
            rnd = -sd+q*V
        elif U < (q+r)/(p+q+r):
            rnd = td-r*math.log(V)
        else:
            rnd = -sd+p*math.log(V)

        f1 = math.exp(-eta-zeta*(rnd-t))
        f2 = math.exp(-theta+xi*(rnd+s))
        if W*g(rnd, sd, td, f1, f2) <= math.exp(psi(rnd, alpha, lam)):
            break

    # transform back to the three-parameter version gig(p,a,b)
    rnd = math.exp(rnd)*(lam/omega+math.sqrt(1.0+math.pow(lam,2)/math.pow(omega,2)))
    if swap:
        rnd = 1.0/rnd

    rnd = rnd/math.sqrt(a/b)
    return rnd


def gigrnd_vec(p, a, b):
    """gigrnd over arrays (baselines-genome addition): the same Devroye (2014) setup and rejection scheme, element by
    element, with the draws of every element still rejected repeated until each has accepted. Each output is a draw from
    the same GIG(p, a, b) as gigrnd's; the random stream differs from the scalar loop's."""
    import numpy as np
    p, a, b = np.broadcast_arrays(np.asarray(p, dtype=np.float64), np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    shape = p.shape
    p, a, b = p.ravel(), a.ravel(), b.ravel()
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        lam = p.copy()
        omega = np.sqrt(a*b)
        swap = lam < 0
        lam = np.abs(lam)
        alpha = np.sqrt(omega**2+lam**2)-lam

        def psi_v(x, al, la):
            return -al*(np.cosh(x)-1.0)-la*(np.exp(x)-x-1.0)

        def dpsi_v(x, al, la):
            return -al*np.sinh(x)-la*(np.exp(x)-1.0)

        both0 = (alpha == 0) & (lam == 0)
        x = -psi_v(1.0, alpha, lam)
        t = np.where((x >= 0.5) & (x <= 2.0), 1.0,
                     np.where(x > 2.0, np.where(both0, 1.0, np.sqrt(2.0/(alpha+lam))),
                              np.where(both0, 1.0, np.log(4.0/(alpha+2.0*lam)))))
        x = -psi_v(-1.0, alpha, lam)
        log_term = np.log(1.0+1.0/alpha+np.sqrt(1.0/alpha**2+2.0/alpha))
        s_low = np.where(both0, 1.0, np.where(alpha == 0, 1.0/lam, np.where(lam == 0, log_term, np.minimum(1.0/lam, log_term))))
        s = np.where((x >= 0.5) & (x <= 2.0), 1.0,
                     np.where(x > 2.0, np.where(both0, 1.0, np.sqrt(4.0/(alpha*math.cosh(1)+lam))), s_low))

        eta = -psi_v(t, alpha, lam)
        zeta = -dpsi_v(t, alpha, lam)
        theta = -psi_v(-s, alpha, lam)
        xi = dpsi_v(-s, alpha, lam)
        pp = 1.0/xi
        r = 1.0/zeta
        td = t-r*eta
        sd = s-pp*theta
        q = td+sd

        out = np.empty(p.size)
        pending = np.arange(p.size)
        while pending.size:
            U = random.random(pending.size)
            V = random.random(pending.size)
            W = random.random(pending.size)
            qk, rk, pk, sdk, tdk = q[pending], r[pending], pp[pending], sd[pending], td[pending]
            total = pk+qk+rk
            rnd = np.where(U < qk/total, -sdk+qk*V, np.where(U < (qk+rk)/total, tdk-rk*np.log(V), -sdk+pk*np.log(V)))
            f1 = np.exp(-eta[pending]-zeta[pending]*(rnd-t[pending]))
            f2 = np.exp(-theta[pending]+xi[pending]*(rnd+s[pending]))
            gval = np.where((rnd >= -sdk) & (rnd <= tdk), 1.0, np.where(rnd > tdk, f1, f2))
            accept = W*gval <= np.exp(psi_v(rnd, alpha[pending], lam[pending]))
            out[pending[accept]] = rnd[accept]
            pending = pending[~accept]

        rnd = np.exp(out)*(lam/omega+np.sqrt(1.0+lam**2/omega**2))
        rnd = np.where(swap, 1.0/rnd, rnd)
        rnd = rnd/np.sqrt(a/b)
    return rnd.reshape(shape)
