"""Probe states at fixed mean photon number <n>."""

from __future__ import annotations

import numpy as np
import qutip as qt
from scipy.optimize import brentq


def coherent(n_target, N):
    return qt.coherent(N, np.sqrt(n_target))


def sqz_vac(n_target, N, theta=0.0):
    """S(r e^{i theta})|0>, r = arcsinh(sqrt(n)). theta=0 anti-squeezes x."""
    r = np.arcsinh(np.sqrt(n_target))
    return qt.squeeze(N, -r * np.exp(1j * theta)) * qt.basis(N, 0)


def cat(n_target, N):
    """Even cat |alpha> + |-alpha> (alpha real, along x), <n> = n_target."""
    f = lambda al: al**2 * np.tanh(al**2) - n_target
    alpha = brentq(f, 1e-4, np.sqrt(n_target) + 2.0)
    return (qt.coherent(N, alpha) + qt.coherent(N, -alpha)).unit()


def sqz_cat(n_target, N, split, theta=0.0):
    """S(r e^{i theta}) (|alpha> + |-alpha>): squeezing gets `split` of the
    photon budget (sinh^2 r = split * n_target), the cat displacement is
    solved numerically so the total state has <n> = n_target."""
    r = np.arcsinh(np.sqrt(split * n_target))
    S = qt.squeeze(N, -r * np.exp(1j * theta))
    n_op = qt.num(N)

    def mean_n(alpha):
        psi = S * (qt.coherent(N, alpha) + qt.coherent(N, -alpha)).unit()
        return qt.expect(n_op, psi.unit()) - n_target

    if mean_n(1e-6) > 0:          # all-squeezing already meets/exceeds budget
        alpha = 0.0
        psi = S * qt.basis(N, 0)
    else:
        alpha = brentq(mean_n, 1e-6, np.sqrt(n_target) + 2.0)
        psi = S * (qt.coherent(N, alpha) + qt.coherent(N, -alpha)).unit()
    return psi.unit()


def fock(n_target, N):
    return qt.basis(N, int(round(n_target)))


def make_probe(kind, n_target, N, **kw):
    return {"coherent": coherent, "sqz_vac": sqz_vac, "cat": cat,
            "sqz_cat": sqz_cat, "fock": fock}[kind](n_target, N, **kw)


def var_x(psi):
    N = psi.shape[0]
    a = qt.destroy(N)
    x = a + a.dag()
    return float(np.real(qt.expect(x * x, psi) - qt.expect(x, psi) ** 2))


def mean_n(psi):
    N = psi.shape[0]
    return float(np.real(qt.expect(qt.num(N), psi)))


def basis_size(n_target, K, var_x_est=None, base=40, cap=420):
    """Adaptive Fock cutoff: the shear inflates Var(p) by K^2 Var(x)."""
    vx = var_x_est if var_x_est is not None else 4 * n_target + 2
    need = base + 12 * n_target + int(3.0 * vx * (1 + K**2))
    return int(min(cap, need))
