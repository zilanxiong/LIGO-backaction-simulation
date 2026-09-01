r"""Interferometer calibration: :math:`\kappa(\Omega)` and :math:`h_{\rm SQL}(\Omega)`.

Formulae follow Kimble, Levin, Matsko, Thorne & Vyatchanin,
*Conversion of conventional gravitational-wave interferometers into QND
interferometers*, Phys. Rev. D **65**, 022002 (2001), for a free-mass
(no signal recycling) interferometer:

.. math::
    \kappa(\Omega) = \frac{2 (I_0/I_{\rm SQL})\, \gamma^4}
                          {\Omega^2 (\gamma^2 + \Omega^2)}, \qquad
    I_{\rm SQL} = \frac{m L^2 \gamma^4}{4 \omega_0}, \qquad
    h_{\rm SQL}(\Omega) = \sqrt{\frac{8\hbar}{m \Omega^2 L^2}} .

With :math:`I_0 = I_{\rm SQL}` one has :math:`\kappa(\gamma) = 1`, the
"SQL-touching" operating point.

This file is a copy of ``ligo_backaction/ifo.py`` from the Gaussian
verification track, so that ``backaction-qfi`` stays self-contained, plus the
strain conversion and the inverse :math:`\kappa \to f` used by the frequency
sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HBAR = 1.054571817e-34  # J s
C_LIGHT = 299792458.0  # m/s

__all__ = ["IFOParams", "ALIGO", "ALIGO_O4", "kappa_of_omega", "h_sql", "lambda_from_h",
           "qfi_h_from_qfi_lambda", "qfi_h_from_qfi_epsilon_a", "strain_uncertainty",
           "f_at_kappa", "EPS_A_PER_LAMBDA"]


@dataclass(frozen=True)
class IFOParams:
    """Parameters of a free-mass power-recycled Michelson interferometer.

    Attributes
    ----------
    mass : reduced mirror mass (kg).  For a four-mirror Michelson with mirrors
        of mass ``M`` the reduced mass entering the differential arm mode is
        ``M/4``; aLIGO has ``M = 40 kg`` so ``mass = 10 kg``.
    length : arm length (m).
    wavelength : carrier wavelength (m).
    gamma : arm-cavity half bandwidth (rad/s).
    power_ratio : circulating power in units of ``I_SQL``.
    """

    mass: float = 10.0
    length: float = 4000.0
    wavelength: float = 1064e-9
    gamma: float = 2 * np.pi * 500.0
    power_ratio: float = 1.0
    name: str = "aLIGO-like free-mass"

    @property
    def omega0(self) -> float:
        """Carrier angular frequency (rad/s)."""
        return 2 * np.pi * C_LIGHT / self.wavelength

    @property
    def I_sql(self) -> float:
        """Circulating power at which :math:`\\kappa(\\gamma) = 1` (W)."""
        return self.mass * self.length**2 * self.gamma**4 / (4 * self.omega0)

    @property
    def power(self) -> float:
        """Circulating power (W)."""
        return self.power_ratio * self.I_sql

    def kappa(self, f_hz):
        """Opto-mechanical coupling :math:`\\kappa` at signal frequency ``f_hz`` (Hz)."""
        return kappa_of_omega(2 * np.pi * np.asarray(f_hz, dtype=float), self)

    def h_sql(self, f_hz):
        """Standard quantum limit strain :math:`h_{\\rm SQL}` at ``f_hz`` (Hz)."""
        return h_sql(2 * np.pi * np.asarray(f_hz, dtype=float), self)


    @classmethod
    def from_arm_cavity(cls, *, mirror_mass, length, wavelength, t_itm,
                        power_w, name):
        """Build a preset from the numbers a detector paper actually quotes.

        The arm-cavity half bandwidth follows from the ITM transmissivity: the
        free spectral range ``c/2L`` divided by the finesse ``2 pi / T`` is the
        full width ``c T / (4 pi L)``, so the half width in Hz is
        ``c T / (8 pi L)`` and ``gamma = c T / (4 L)`` in rad/s.  The reduced
        mass of the differential arm mode is ``mirror_mass / 4``.  ``power_w`` is the circulating power
        in one arm; it is converted to ``power_ratio = power_w / I_SQL``.
        """
        gamma = C_LIGHT * t_itm / (4.0 * length)
        base = cls(mass=mirror_mass / 4.0, length=length, wavelength=wavelength,
                   gamma=gamma, power_ratio=1.0, name=name)
        return cls(mass=base.mass, length=base.length, wavelength=base.wavelength,
                   gamma=gamma, power_ratio=power_w / base.I_sql, name=name)


ALIGO = IFOParams()

#: Advanced LIGO as built, in the free-mass approximation.  Arm length and test
#: mass from the aLIGO reference design; ITM transmissivity 1.48 % gives a
#: 44 Hz arm-cavity pole; 350 kW circulating is representative of the O4 runs
#: (design is 750 kW).  Signal recycling is *not* modelled -- this is the
#: free-mass Kimble relation with aLIGO's numbers in it, which is the right
#: order of magnitude for kappa but not a substitute for the full
#: signal-recycled response.
ALIGO_O4 = IFOParams.from_arm_cavity(
    mirror_mass=40.0,
    length=3994.5,
    wavelength=1064e-9,
    t_itm=0.0148,
    power_w=350e3,
    name="aLIGO O4-like (free-mass)",
)


def kappa_of_omega(omega, params: IFOParams = ALIGO):
    r""":math:`\kappa(\Omega) = 2 (I_0/I_{\rm SQL})\gamma^4 / [\Omega^2(\gamma^2+\Omega^2)]`."""
    omega = np.asarray(omega, dtype=float)
    g = params.gamma
    return 2.0 * params.power_ratio * g**4 / (omega**2 * (g**2 + omega**2))


def h_sql(omega, params: IFOParams = ALIGO):
    r""":math:`h_{\rm SQL}(\Omega) = \sqrt{8\hbar/(m\Omega^2 L^2)}`."""
    omega = np.asarray(omega, dtype=float)
    return np.sqrt(8.0 * HBAR / (params.mass * omega**2 * params.length**2))


def lambda_from_h(h, kappa, h_sql_value):
    r"""Dimensionless displacement :math:`\lambda = \sqrt{\kappa}\, h/h_{\rm SQL}`."""
    return np.sqrt(kappa) * h / h_sql_value


def qfi_h_from_qfi_lambda(qfi_lambda, kappa, h_sql_value):
    r"""Convert :math:`\mathcal{F}_\lambda` to strain units.

    Because :math:`\lambda = \sqrt{\kappa}\,h/h_{\rm SQL}` is linear in ``h``,

    .. math:: \mathcal{F}_h = \frac{\kappa}{h_{\rm SQL}^2}\,\mathcal{F}_\lambda .

    The Cramer--Rao bound on the strain is then
    :math:`\sigma_h \ge 1/\sqrt{\mathcal{F}_h}`; for vacuum input
    (:math:`\mathcal{F}_\lambda = 2`) this gives
    :math:`\sigma_h \ge h_{\rm SQL}/\sqrt{2\kappa}`, i.e. exactly the noise of the
    frequency-dependent ("variational") homodyne readout.
    """
    return kappa * np.asarray(qfi_lambda, dtype=float) / h_sql_value**2


# ---------------------------------------------------------------------------
# Added for the Fock-basis study
# ---------------------------------------------------------------------------

#: ``epsilon_a`` drives the generator ``sqrt(2) x`` over ``t_final``, while the
#: dimensionless displacement ``lambda`` is referred to ``x`` itself, so
#: ``F_epsilon_a = 2 t_final^2 F_lambda``.  Same constant as
#: :data:`gaussian_reference.EPS_A_PER_LAMBDA`.
EPS_A_PER_LAMBDA = 2.0


def qfi_h_from_qfi_epsilon_a(qfi_eps_a, kappa, h_sql_value, t_final=1.0):
    r"""Strain QFI from the QFI of the channel's own sensing parameter.

    .. math:: \mathcal{F}_h = \frac{\kappa}{h_{\rm SQL}^2}\,\mathcal{F}_\lambda
              = \frac{\kappa\,\mathcal{F}_{\epsilon_a}}
                     {2\,t_{\rm final}^2\,h_{\rm SQL}^2}
    """
    qfi_lambda = np.asarray(qfi_eps_a, dtype=float) / (EPS_A_PER_LAMBDA * t_final**2)
    return qfi_h_from_qfi_lambda(qfi_lambda, kappa, h_sql_value)


def strain_uncertainty(qfi_eps_a, kappa, h_sql_value, t_final=1.0):
    r"""Cramer--Rao bound on the strain, :math:`\sigma_h = 1/\sqrt{\mathcal{F}_h}`.

    In units of :math:`h_{\rm SQL}` this is
    :math:`\sigma_h/h_{\rm SQL} = \sqrt{2/(\kappa\,\mathcal{F}_{\epsilon_a})}`
    at ``t_final = 1``, so a vacuum probe with no loss and no back-action
    (:math:`\mathcal{F}_{\epsilon_a} = 4`) sits at
    :math:`1/\sqrt{2\kappa}` -- the free-mass SQL crossing at
    :math:`\kappa = 1/2`.

    Returns ``(sigma_h, sigma_h / h_sql)``.
    """
    fh = qfi_h_from_qfi_epsilon_a(qfi_eps_a, kappa, h_sql_value, t_final)
    sigma = 1.0 / np.sqrt(fh)
    return sigma, sigma / h_sql_value


def f_at_kappa(kappa, params: IFOParams = ALIGO):
    r"""Invert :math:`\kappa(\Omega)`: the signal frequency (Hz) at which the
    coupling equals ``kappa``.

    With :math:`u = (\Omega/\gamma)^2` the definition rearranges to
    :math:`u^2 + u - 2R/\kappa = 0`, so
    :math:`u = \tfrac12(-1 + \sqrt{1 + 8R/\kappa})`.  Monotonic, so the
    solution is unique.
    """
    kappa = np.asarray(kappa, dtype=float)
    # kappa -> 0 is f -> infinity; matplotlib's secondary axis probes the
    # endpoints of its transform, so return the limit rather than warning.
    with np.errstate(divide="ignore", invalid="ignore"):
        u = 0.5 * (-1.0 + np.sqrt(1.0 + 8.0 * params.power_ratio / kappa))
    return params.gamma * np.sqrt(u) / (2 * np.pi)
