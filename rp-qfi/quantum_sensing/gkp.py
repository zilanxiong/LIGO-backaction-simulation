"""
GKP (Gottesman-Kitaev-Preskill) state construction.

Provides discrete position-space generation, Fock basis coefficients,
and superposed squeezed state representations.

Functions taken from Xanadu AI.
"""

import numpy as np
from scipy import integrate
from scipy.special import factorial, eval_hermite


def discrete_q(Nq, qmax):
    """Returns a discretization of q space.

    Args:
        qmax (float): limits for the q space [-qmax, qmax]
        Nq (float): number of q points in the array
            (rounds down if not integer)

    Returns:
        q (array): Nq evenly spaced position values from [-qmax, qmax].
    """
    q = np.linspace(-qmax, qmax, Nq)
    return q


def gkp_s_values(smax):
    """Returns an array of indices to perform the summation over s
    for the superposition of squeezed states in a GKP state.

    Args:
        smax (int): limit for the summation [-smax, smax]

    Returns:
        s (array): integer indices for the summation [-smax, smax].
    """
    smax = np.ceil(smax)
    s = np.arange(-smax, smax+1, 1)
    return s


def gkp(D, mu=0, sd=7, N=200):
    """Returns a discretized q space and a normalized logical (0 or 1)
    GKP state. Default is to return logical 0 (mu=0).

    Args:
        D (float): width of the Gaussian envelope and squeezed states
            in the superposition
        mu (0 or 1): logical value for the GKP state
        sd (float): number of standard deviations for q space
        N (int): number of points per peak in the GKP comb

    Returns:
        q (array): position values
        gkp (array): array of GKP wavefunction values.
    """
    # Setting the total range for position space
    qmax = max(sd/D, sd*D)

    # Initiating indices to sum over the comb peaks in each state.
    smax = int(np.ceil(qmax/(2*np.sqrt(np.pi))))
    s = gkp_s_values(smax)

    # Getting a discretization of position space
    Nq = N*(2*smax+1)
    q = discrete_q(Nq, qmax)

    # Initiating empty arrays to be filled in
    gkp_wfn = np.zeros(len(q))

    # Looping over all peaks in the superposition
    for i in range(len(s)):
        gkp_wfn = gkp_wfn + np.exp(-((D**2)*np.pi*(2*s[i]+mu)**2)/2.0) *\
            np.exp(-((q-(2*s[i]+mu)*np.sqrt(np.pi))**2) /
                   (2*D**2))/(np.pi*D**2)**0.25

    # Normalization
    Norm = 1/np.sqrt(integrate.trapezoid(np.absolute(gkp_wfn)**2, q))
    gkp_wfn = Norm*gkp_wfn

    return q, gkp_wfn


def num_state(n, q):
    """Returns the Fock position wavefunctions defined over q space.

    Args:
        n (int): Fock index
        q (array): position values

    Returns:
        fock (array): nth Fock state wavefunction
    """
    fock = ((1/np.sqrt((2**n)*factorial(n)))*(np.pi**-0.25)
            *np.exp(-(q**2)/2)*eval_hermite(n, q)
           )
    return fock


def gkp_fock_coeff(n, combs=100, alpha=np.sqrt(np.pi), mu=0):
    """Returns the nth Fock coefficient for the ideal logical GKP state.

    Args:
        n (int): index of Fock state
        combs (int): number of delta spikes in the ideal state used to
                        calculate inner product with Fock state
        alpha (float): GKP lattice constant
        mu (0 or 1): logical value of GKP state

    Returns:
        coeff (complex): inner product between ideal GKP and nth
            Fock states.
    """
    samples = (mu+np.arange(-combs/2, 1+combs/2)*2)*alpha
    coeff = np.sum(num_state(n, samples))
    return coeff


def gkp_fock(qubit, eps, q, n_max=120, norm=True):
    """Returns the epsilon GKP q wavefunction and coefficients in the
        Fock basis.

    Args:
        qubit (array): qubit state in logical basis
        eps (float): value of epsilon
        q (array): array of q values for q wavefunction
        n_max (int): Fock cutoff
        norm (Boolean): whether to return a normalized state

    Returns:
        gkp (array): q wavefunction
        coeffs (array): Fock coefficients up to cutoff.
    """
    qubit = qubit/np.linalg.norm(qubit)
    gkp_wfn = 0
    coeffs = np.zeros(n_max+1, dtype=complex)
    N = 0
    for i in range(n_max+1):
        coeff = (qubit[0]*gkp_fock_coeff(i, mu=0)+qubit[1]
                 * gkp_fock_coeff(i, mu=1))*np.exp(-i*eps)
        coeffs[i] = coeff
        gkp_wfn += coeff*num_state(i, q)
        if norm:
            N += np.absolute(coeff)**2
    if norm:
        gkp_wfn = gkp_wfn/np.sqrt(N)
        coeffs = coeffs/np.sqrt(N)
    return gkp_wfn, coeffs
