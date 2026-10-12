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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HBAR = 1.054571817e-34  # J s
C_LIGHT = 299792458.0  # m/s

__all__ = ["IFOParams", "ALIGO", "kappa_of_omega", "h_sql", "lambda_from_h", "qfi_h_from_qfi_lambda"]


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


ALIGO = IFOParams()


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
