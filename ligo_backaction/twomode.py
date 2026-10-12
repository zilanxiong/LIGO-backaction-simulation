r"""Two-frequency-mode sandbox: when does the per-frequency description break?

The waveform-estimation literature treats each sideband frequency
:math:`\Omega` as an independent single-mode channel and adds the resulting
QFIs.  That is exact for Gaussian states in the linearised two-photon
formalism, because distinct sidebands neither share correlations nor talk to
each other.  Two things can break it:

1. **Cross-frequency back-action.**  The radiation-pressure force is quadratic
   in the total field, so a two-sideband field produces a beat term
   :math:`\propto x_1 x_2` at :math:`\Omega_1-\Omega_2`.  Linearising per
   frequency drops it.  Here it is retained with strength ``kappa_cross``.
2. **Cross-frequency correlations in the probe.**  A broadband non-Gaussian
   source (a "frequency-multiplexed preparation") generally entangles the
   sideband modes; a per-frequency description assumes a product state.

This module builds the joint two-mode channel and compares the exact joint
QFI matrix against the per-frequency (reduced-state) prediction.  Everything is
a deliberately minimal toy model -- it is meant to *locate* the breakdown, not
to model a real interferometer's broadband response.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from .channels import loss_map
from .operators import annihilation, as_density_matrix, displace_op, x_op
from .states import normalize

__all__ = [
    "TwoModeSpec",
    "product_state",
    "two_mode_squeezed_vacuum",
    "entangled_cat",
    "run_two_mode_channel",
    "qfi_matrix",
    "joint_qfi_matrix",
    "per_frequency_qfi",
    "additivity_defect",
]


def _op1(op: np.ndarray, N: int) -> np.ndarray:
    return np.kron(op, np.eye(N, dtype=complex))


def _op2(op: np.ndarray, N: int) -> np.ndarray:
    return np.kron(np.eye(N, dtype=complex), op)


@dataclass
class TwoModeSpec:
    """Joint channel for two sideband modes.

    ``kappa_cross`` is the beat term that the per-frequency description drops;
    setting it to zero makes the *dynamics* factorise (correlations can still
    come from the input state).
    """

    kappa1: float = 1.0
    kappa2: float = 1.0
    kappa_cross: float = 0.0
    eta: float = 1.0
    ordering: str = "BA1"

    def hamiltonian(self, N: int) -> np.ndarray:
        r""":math:`H = \tfrac{\kappa_1}{2}x_1^2 + \tfrac{\kappa_2}{2}x_2^2 + \kappa_c x_1 x_2`."""
        x1, x2 = _op1(x_op(N), N), _op2(x_op(N), N)
        return 0.5 * self.kappa1 * (x1 @ x1) + 0.5 * self.kappa2 * (x2 @ x2) + self.kappa_cross * (x1 @ x2)

    def backaction_unitary(self, N: int) -> np.ndarray:
        return expm(-1j * self.hamiltonian(N))


def product_state(psi1: np.ndarray, psi2: np.ndarray) -> np.ndarray:
    return np.kron(psi1, psi2)


def two_mode_squeezed_vacuum(N: int, r: float) -> np.ndarray:
    r""":math:`\exp[r(a_1^\dagger a_2^\dagger - a_1 a_2)]|0,0\rangle` -- Gaussian and correlated."""
    a1, a2 = _op1(annihilation(N), N), _op2(annihilation(N), N)
    gen = r * (a1.conj().T @ a2.conj().T - a1 @ a2)
    psi = np.zeros(N * N, dtype=complex)
    psi[0] = 1.0
    return normalize(expm(gen) @ psi)


def entangled_cat(N: int, alpha: float) -> np.ndarray:
    r""":math:`\propto |\alpha,\alpha\rangle + |-\alpha,-\alpha\rangle` -- non-Gaussian and correlated."""
    vac = np.zeros(N, dtype=complex)
    vac[0] = 1.0
    plus = displace_op(N, alpha) @ vac
    minus = displace_op(N, -alpha) @ vac
    return normalize(np.kron(plus, plus) + np.kron(minus, minus))


def run_two_mode_channel(state: np.ndarray, spec: TwoModeSpec, N: int, lams=(0.0, 0.0)):
    r"""Apply back-action, the two signal displacements, and loss on both modes.

    Returns ``(rho_out, [drho/dlam1, drho/dlam2])``.  The signal generators
    :math:`x_1, x_2` commute with the back-action Hamiltonian (which is built
    only from :math:`x`), so the derivatives are exact commutators.
    """
    psi = np.asarray(state)
    rho = as_density_matrix(psi)
    x1, x2 = _op1(x_op(N), N), _op2(x_op(N), N)
    U_ba = spec.backaction_unitary(N)
    U_sig = expm(1j * (lams[0] * x1 + lams[1] * x2))

    if spec.ordering == "BA1":
        rho = U_sig @ (U_ba @ rho @ U_ba.conj().T) @ U_sig.conj().T
    elif spec.ordering == "BA2":
        rho = U_ba @ (U_sig @ rho @ U_sig.conj().T) @ U_ba.conj().T
    elif spec.ordering == "none":
        rho = U_sig @ rho @ U_sig.conj().T
    else:
        raise ValueError(f"unsupported ordering {spec.ordering!r}")

    drhos = [1j * (G @ rho - rho @ G) for G in (x1, x2)]

    if spec.eta != 1.0:
        single = loss_map(N, spec.eta)
        # Loss acts independently on each mode; apply it as a map on the joint
        # density matrix by reshaping to (N, N, N, N) and contracting per mode.
        def both_modes(m):
            return _apply_single_mode_map(_apply_single_mode_map(m, single, N, 0), single, N, 1)

        rho = both_modes(rho)
        drhos = [both_modes(d) for d in drhos]
    return rho, drhos


def _apply_single_mode_map(rho: np.ndarray, single_map, N: int, mode: int) -> np.ndarray:
    """Apply a single-mode CPTP map (given as a function on N x N matrices) to one mode."""
    t = rho.reshape(N, N, N, N)  # indices (i1, i2, j1, j2)
    if mode == 0:
        t = np.einsum("iajb->abij", t)  # spectator first
    else:
        t = np.einsum("aibj->abij", t)
    flat = t.reshape(N * N, N, N)
    out = np.stack([single_map(block) for block in flat]).reshape(N, N, N, N)
    if mode == 0:
        out = np.einsum("abij->iajb", out)
    else:
        out = np.einsum("abij->aibj", out)
    return out.reshape(N * N, N * N)


def qfi_matrix(rho: np.ndarray, drhos, eig_tol: float = 1e-10) -> np.ndarray:
    r"""Multi-parameter SLD QFI matrix.

    .. math::
        \mathcal{F}_{ij} = 2\sum_{k,l}
        \frac{\mathrm{Re}\left[(\partial_i\rho)_{kl}(\partial_j\rho)_{lk}\right]}{p_k+p_l}.
    """
    w, V = np.linalg.eigh(0.5 * (rho + rho.conj().T))
    w = np.clip(w, 0.0, None)
    scale = max(w.max(), 1e-300)
    denom = w[:, None] + w[None, :]
    mask = denom > eig_tol * scale
    inv = np.zeros_like(denom)
    inv[mask] = 1.0 / denom[mask]
    Ds = [V.conj().T @ (0.5 * (d + d.conj().T)) @ V for d in drhos]
    n = len(Ds)
    F = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            F[i, j] = 2.0 * float(np.real(np.sum(Ds[i] * Ds[j].T * inv)))
    return F


def joint_qfi_matrix(state, spec: TwoModeSpec, N: int, lams=(0.0, 0.0)) -> np.ndarray:
    rho, drhos = run_two_mode_channel(state, spec, N, lams)
    return qfi_matrix(rho, drhos)


def per_frequency_qfi(state, spec: TwoModeSpec, N: int, lams=(0.0, 0.0)) -> np.ndarray:
    r"""The independent-frequency prediction.

    Each mode is traced out of the *input*, propagated through its own
    single-frequency channel (``kappa_cross`` discarded), and its QFI computed.
    This is exactly what "treat every sideband as an independent channel" means.
    """
    from .channels import ChannelSpec
    from .qfi import qfi_lambda

    rho = as_density_matrix(np.asarray(state)).reshape(N, N, N, N)
    rho1 = np.einsum("iaja->ij", rho)
    rho2 = np.einsum("aiaj->ij", rho)
    out = []
    for red, kappa, lam in ((rho1, spec.kappa1, lams[0]), (rho2, spec.kappa2, lams[1])):
        sub = ChannelSpec(kappa=kappa, lam=lam, ordering=spec.ordering, eta_out=spec.eta)
        out.append(qfi_lambda(red, sub))
    return np.array(out)


def additivity_defect(state, spec: TwoModeSpec, N: int, lams=(0.0, 0.0)) -> dict:
    r"""Compare the joint QFI matrix with the per-frequency prediction.

    Returns a dict with the joint matrix, the per-frequency diagonal, the
    absolute and relative defect of the *summed* QFI
    :math:`\mathrm{tr}\mathcal{F}`, and the largest off-diagonal element
    (which the independent description predicts to be zero).
    """
    F = joint_qfi_matrix(state, spec, N, lams)
    per = per_frequency_qfi(state, spec, N, lams)
    tr_joint = float(np.trace(F))
    tr_per = float(np.sum(per))
    return {
        "F_joint": F,
        "per_frequency": per,
        "trace_joint": tr_joint,
        "trace_per_frequency": tr_per,
        "abs_defect": tr_joint - tr_per,
        "rel_defect": (tr_joint - tr_per) / tr_per if tr_per else np.nan,
        "max_offdiag": float(np.max(np.abs(F - np.diag(np.diag(F))))),
    }
