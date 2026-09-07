r"""Probe-state optimisation at fixed mean photon number.

Two routines:

* :func:`optimal_no_backaction_state` -- the *exact* optimum for pure
  displacement sensing (no radiation pressure, no loss).  Maximising
  :math:`\mathrm{Var}(x)` at fixed :math:`\langle n\rangle` is a two-term
  Lagrangian eigenproblem, so it is solved by a one-dimensional root find on
  the multiplier rather than by gradient descent.  The answer is the squeezed
  vacuum with :math:`\sinh^2 r = \bar n`, giving
  :math:`\mathcal{F} = 2(2\bar n + 1 + 2\sqrt{\bar n(\bar n+1)})`.

* :func:`optimize_state_for_channel` -- numerical maximisation of the QFI
  through an arbitrary :class:`~ligo_backaction.channels.ChannelSpec`, used to
  ask what the best probe is *with* back-action and loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize

from .channels import ChannelSpec
from .operators import number, x_op
from .qfi import qfi_lambda
from .states import make_state, mean_photon_number, normalize

__all__ = ["optimal_no_backaction_state", "optimize_state_for_channel", "OptimizationResult"]


def _lagrange_state(N: int, mu: float) -> np.ndarray:
    r"""Top eigenvector of :math:`\hat x^2 - \mu\,\hat n`."""
    x = x_op(N)
    M = np.real(x @ x - mu * number(N))
    w, V = np.linalg.eigh(M)
    return V[:, -1].astype(complex)


def optimal_no_backaction_state(N: int, nbar: float, mu_max: float = 200.0) -> np.ndarray:
    r"""State maximising :math:`4\,\mathrm{Var}(x)` subject to :math:`\langle n\rangle = \bar n`.

    Solved as :math:`\max_\psi \langle \hat x^2 - \mu \hat n\rangle` with the
    multiplier :math:`\mu` tuned so the photon-number constraint is met.  This
    is the "state optimised without back-action" reference probe.
    """
    if nbar <= 0:
        out = np.zeros(N, dtype=complex)
        out[0] = 1.0
        return out

    def f(mu):
        return mean_photon_number(_lagrange_state(N, mu)) - nbar

    lo, hi = 1e-6, 1.0
    while f(hi) > 0 and hi < mu_max:
        hi *= 2.0
    if f(hi) > 0:  # pragma: no cover - only for absurd nbar
        raise ValueError(f"could not bracket the multiplier for nbar={nbar}")
    while f(lo) < 0 and lo > 1e-12:
        lo /= 4.0
    mu = brentq(f, lo, hi, xtol=1e-14, rtol=1e-14)
    return normalize(_lagrange_state(N, mu))


@dataclass
class OptimizationResult:
    state: np.ndarray
    qfi: float
    nbar: float
    success: bool
    n_iter: int


def optimize_state_for_channel(
    N: int,
    nbar: float,
    spec: ChannelSpec,
    seeds: tuple[str, ...] = ("squeezed", "cat_even", "coherent", "fock"),
    extra_seeds: tuple[np.ndarray, ...] = (),
    n_params: int | None = None,
    penalty: float = 50.0,
    maxiter: int = 400,
    real_only: bool = True,
) -> OptimizationResult:
    r"""Maximise :math:`\mathcal{F}_\lambda` through ``spec`` at fixed :math:`\bar n`.

    The state is parametrised by its (real, by default) Fock amplitudes; the
    photon-number constraint is imposed by a quadratic penalty
    :math:`P(\langle n\rangle - \bar n)^2` on the normalised state.  Several
    physically motivated seeds are tried and the best result is returned.

    ``real_only`` is the right choice whenever the channel and the signal
    generator are real in the Fock basis (which holds for ``phi_sig = 0``);
    set it to ``False`` for a general search.

    ``n_params`` restricts the search to the lowest Fock levels while the channel
    is still evaluated in the full space ``N``.  This decouples the size of the
    search space from the cutoff needed for numerical accuracy, which matters
    because the shear demands large ``N``.  ``extra_seeds`` accepts explicit kets
    (e.g. the Gaussian optimum for this channel) alongside the named families.

    This is a *local* search from several seeds -- it reports the best local
    optimum found, not a certified global one.
    """
    m = N if n_params is None else min(int(n_params), N)
    n_vec = np.arange(N, dtype=float)

    def unpack(v):
        psi = np.zeros(N, dtype=complex)
        if real_only:
            psi[:m] = v
        else:
            psi[:m] = v[:m] + 1j * v[m:]
        return normalize(psi)

    def cost(v):
        psi = unpack(v)
        nb = float(np.sum(n_vec * np.abs(psi) ** 2))
        return -qfi_lambda(psi, spec) + penalty * (nb - nbar) ** 2

    best: OptimizationResult | None = None
    candidates: list[np.ndarray] = []
    for seed in seeds:
        try:
            candidates.append(make_state(seed, N, nbar))
        except Exception:
            continue
    candidates.extend(np.asarray(psi, dtype=complex) for psi in extra_seeds)
    for psi0 in candidates:
        head = psi0[:m]
        v0 = np.real(head) if real_only else np.concatenate([np.real(head), np.imag(head)])
        res = minimize(cost, v0, method="L-BFGS-B", options={"maxiter": maxiter})
        # The seed itself is a legitimate candidate: it satisfies the photon-number
        # constraint exactly, whereas the penalised optimum only satisfies it
        # approximately, so the search must never report something worse.
        for psi, nit, ok in ((unpack(res.x), int(res.nit), bool(res.success)), (psi0, 0, True)):
            cand = OptimizationResult(
                state=psi,
                qfi=qfi_lambda(psi, spec),
                nbar=mean_photon_number(psi),
                success=ok,
                n_iter=nit,
            )
            if best is None or cand.qfi > best.qfi:
                best = cand
    assert best is not None
    return best
