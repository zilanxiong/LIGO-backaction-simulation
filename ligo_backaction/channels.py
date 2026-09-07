r"""Channels as pure functions ``(state, params) -> rho_out``.

A channel is an ordered list of :class:`Step` objects.  Exactly one step is the
signal (the estimated parameter); this lets us differentiate the output state
*exactly* rather than by finite differences, because for a pipeline

.. math::
    \rho_{\rm out}(\lambda) = \Lambda_{\rm post}\!\left(
        U(\lambda)\, \Lambda_{\rm pre}(\rho_{\rm in})\, U^\dagger(\lambda)\right),
    \qquad U(\lambda) = e^{i\lambda G},

linearity of :math:`\Lambda_{\rm post}` gives

.. math::
    \partial_\lambda \rho_{\rm out} = \Lambda_{\rm post}\!\left(
        i\,[G,\, U \Lambda_{\rm pre}(\rho_{\rm in}) U^\dagger]\right).

For the "BA3" ordering the signal and back-action generators are exponentiated
together and the derivative falls back to a central finite difference.

Channel orderings
-----------------
``BA1``  back-action then signal        (radiation pressure -> displacement)
``BA2``  signal then back-action        (displacement -> radiation pressure)
``BA3``  simultaneous                   (the physical case)
``none`` signal only, no radiation pressure (reference)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
from scipy.special import gammaln

from .operators import (
    annihilation,
    as_density_matrix,
    backaction_unitary,
    joint_unitary,
    quad_op,
    signal_unitary,
)

__all__ = [
    "Step",
    "ChannelSpec",
    "ORDERINGS",
    "loss_map",
    "dephasing_map",
    "effective_support",
    "unitary_map",
    "build_steps",
    "run_channel",
    "output_and_derivative",
]

Map = Callable[[np.ndarray], np.ndarray]


@dataclass
class Step:
    """One CPTP map in the pipeline."""

    name: str
    apply: Map
    is_signal: bool = False
    generator: np.ndarray | None = None  # G, for the exact-derivative path
    fused_signal: bool = False  # signal is baked into a non-commuting exponential


def unitary_map(U: np.ndarray) -> Map:
    Ud = U.conj().T

    def _apply(rho: np.ndarray) -> np.ndarray:
        return U @ rho @ Ud

    return _apply


@lru_cache(maxsize=64)
def _loss_weights(N: int, eta: float) -> np.ndarray:
    r"""Coefficients :math:`c_k(m) = \eta^{m/2}\sqrt{(1-\eta)^k \binom{m+k}{k}}`.

    Returned as an ``(N, N)`` array ``w[k, m]``; ``K_k|m+k\rangle = c_k(m)|m\rangle``
    for the Kraus operators :math:`K_k = \sqrt{(1-\eta)^k/k!}\,\eta^{\hat n/2} a^k`.
    """
    k = np.arange(N)[:, None]
    m = np.arange(N)[None, :]
    log_binom = gammaln(m + k + 1.0) - gammaln(k + 1.0) - gammaln(m + 1.0)
    log_w = 0.5 * (m * np.log(eta) + k * np.log1p(-eta) + log_binom)
    return np.exp(log_w)


def loss_map(N: int, eta: float) -> Map:
    r"""Bosonic loss (beamsplitter with vacuum) of power transmissivity ``eta``.

    Kraus operators :math:`K_k = \sqrt{(1-\eta)^k/k!}\ \eta^{\hat n/2} a^k`, applied
    in the Fock basis as

    .. math::
        \Lambda(\rho)_{mn} = \sum_k c_k(m)\,c_k(n)\,\rho_{m+k,\,n+k},

    which is :math:`O(N^3)` rather than the :math:`O(N^4)` of explicit Kraus
    matrix products.  Summing ``k`` over the whole truncated space makes the map
    exactly trace preserving there.
    """
    eta = float(eta)
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must lie in [0, 1]")
    if eta == 1.0:
        return lambda rho: rho
    if eta == 0.0:
        def _to_vacuum(rho: np.ndarray) -> np.ndarray:
            out = np.zeros_like(rho)
            out[0, 0] = np.trace(rho)
            return out

        return _to_vacuum

    w = _loss_weights(N, eta)

    def _apply(rho: np.ndarray) -> np.ndarray:
        out = np.zeros_like(rho, dtype=complex)
        # Only Kraus orders up to the highest occupied level contribute; cutting
        # the sum there turns the map from O(N^3) into O(support * N^2), which
        # matters because large kappa forces large cutoffs.
        k_max = effective_support(rho)
        for k in range(k_max + 1):
            size = N - k
            wk = w[k, :size]
            out[:size, :size] += rho[k:, k:] * (wk[:, None] * wk[None, :])
        return out

    return _apply


def effective_support(*matrices: np.ndarray, rtol: float = 1e-15) -> int:
    """Highest Fock index carrying non-negligible weight in any of ``matrices``."""
    idx = 0
    for m in matrices:
        rows = np.abs(m).max(axis=1)
        scale = rows.max()
        if scale <= 0:
            continue
        nz = np.nonzero(rows > rtol * scale)[0]
        if nz.size:
            idx = max(idx, int(nz[-1]))
    return idx


def dephasing_map(N: int, sigma: float) -> Map:
    r"""Gaussian phase noise: a random rotation :math:`e^{-i\phi \hat n}` with
    :math:`\phi \sim \mathcal{N}(0,\sigma^2)`.

    Averaging exactly gives :math:`\rho_{mn} \to \rho_{mn}\,e^{-\sigma^2 (m-n)^2/2}`.
    This is a non-Gaussian channel (it is not a Gaussian map in the
    covariance-matrix sense), so it lives only in the Fock track.
    """
    sigma = float(sigma)
    if sigma == 0.0:
        return lambda rho: rho
    idx = np.arange(N)
    mask = np.exp(-0.5 * sigma**2 * (idx[:, None] - idx[None, :]) ** 2)

    def _apply(rho: np.ndarray) -> np.ndarray:
        return rho * mask

    return _apply


ORDERINGS = ("none", "BA1", "BA2", "BA3")


@dataclass
class ChannelSpec:
    """Full specification of a single-frequency channel.

    The pipeline is assembled in this fixed physical order::

        input  ->  phase noise (in)  ->  injection loss  ->
                   [ radiation pressure / signal, per `ordering` ]  ->
                   phase noise (out) ->  detection loss  ->  output

    Attributes
    ----------
    kappa : opto-mechanical coupling at this sideband frequency.
    lam : signal displacement amplitude :math:`\\lambda` (the estimated parameter's
        operating point; QFI is evaluated here).
    phi_sig : signal quadrature angle.  ``0`` is the physical free-mass case,
        for which the signal generator commutes with the back-action generator.
    ordering : one of :data:`ORDERINGS`.
    eta_in, eta_out : injection / detection efficiencies.
    sigma_in, sigma_out : phase-noise standard deviations before / after the
        opto-mechanical interaction.
    """

    kappa: float = 1.0
    lam: float = 0.0
    phi_sig: float = 0.0
    ordering: str = "BA1"
    eta_in: float = 1.0
    eta_out: float = 1.0
    sigma_in: float = 0.0
    sigma_out: float = 0.0

    def __post_init__(self):
        if self.ordering not in ORDERINGS:
            raise ValueError(f"ordering must be one of {ORDERINGS}, got {self.ordering!r}")


def _core_steps(N: int, spec: ChannelSpec) -> list[Step]:
    G = quad_op(N, spec.phi_sig)
    ba = Step("backaction", unitary_map(backaction_unitary(N, spec.kappa)))
    sig = Step(
        "signal",
        unitary_map(signal_unitary(N, spec.lam, spec.phi_sig)),
        is_signal=True,
        generator=G,
    )
    if spec.ordering == "none":
        return [sig]
    if spec.ordering == "BA1":  # radiation pressure -> displacement sensing
        return [ba, sig]
    if spec.ordering == "BA2":  # displacement sensing -> radiation pressure
        return [sig, ba]
    # BA3: simultaneous
    return [
        Step(
            "signal+backaction",
            unitary_map(joint_unitary(N, spec.kappa, spec.lam, spec.phi_sig)),
            is_signal=True,
            generator=G,
            fused_signal=True,
        )
    ]


def build_steps(N: int, spec: ChannelSpec) -> list[Step]:
    """Assemble the ordered list of maps for ``spec`` in dimension ``N``."""
    steps: list[Step] = []
    if spec.sigma_in:
        steps.append(Step("phase_noise_in", dephasing_map(N, spec.sigma_in)))
    if spec.eta_in != 1.0:
        steps.append(Step("loss_in", loss_map(N, spec.eta_in)))
    steps.extend(_core_steps(N, spec))
    if spec.sigma_out:
        steps.append(Step("phase_noise_out", dephasing_map(N, spec.sigma_out)))
    if spec.eta_out != 1.0:
        steps.append(Step("loss_out", loss_map(N, spec.eta_out)))
    return steps


def run_channel(state: np.ndarray, spec: ChannelSpec) -> np.ndarray:
    """Pure function ``(state, params) -> rho_out``.

    ``state`` may be a ket or a density matrix; the output is always a density
    matrix.
    """
    rho = as_density_matrix(state)
    N = rho.shape[0]
    for step in build_steps(N, spec):
        rho = step.apply(rho)
    return rho


def output_and_derivative(state: np.ndarray, spec: ChannelSpec, dlam: float = 1e-5):
    r"""Return :math:`(\rho_{\rm out}, \partial_\lambda \rho_{\rm out})`.

    Uses the exact commutator expression when the signal is a separate unitary,
    and a central finite difference (step ``dlam``) for the fused "BA3" step.
    """
    rho_in = as_density_matrix(state)
    N = rho_in.shape[0]
    steps = build_steps(N, spec)
    sig_index = next(i for i, s in enumerate(steps) if s.is_signal)
    fused = steps[sig_index].fused_signal

    if fused and not _generators_commute(N, spec):
        rho = run_channel(state, spec)
        plus = run_channel(state, ChannelSpec(**{**spec.__dict__, "lam": spec.lam + dlam}))
        minus = run_channel(state, ChannelSpec(**{**spec.__dict__, "lam": spec.lam - dlam}))
        return rho, (plus - minus) / (2 * dlam)

    rho = rho_in
    for step in steps[:sig_index]:
        rho = step.apply(rho)
    rho = steps[sig_index].apply(rho)
    G = steps[sig_index].generator
    drho = 1j * (G @ rho - rho @ G)
    for step in steps[sig_index + 1 :]:
        rho = step.apply(rho)
        drho = step.apply(drho)
    return rho, drho


def _generators_commute(N: int, spec: ChannelSpec) -> bool:
    """The BA and signal generators commute iff the signal is in the p quadrature."""
    return abs(np.sin(spec.phi_sig)) < 1e-14 or spec.kappa == 0.0
